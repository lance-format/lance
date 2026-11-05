// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::inserted_rows::{KeyExistenceFilterBuilder, extract_key_value_from_batch};
use super::{
    InsertedKeyTracker, MergeInsertParams, MergeStats, SourceDedupeBehavior, WhenMatched,
    WhenNotMatchedBySource, create_duplicate_row_error, format_key_values_on_columns,
};
use crate::dataset::utils::CapturedRowIds;
use crate::io::exec::Planner;
use arrow_array::{
    BooleanArray, RecordBatch, StructArray, UInt32Array, cast::AsArray, types::UInt64Type,
};
use arrow_schema::{DataType, Field, Fields, Schema};
use arrow_select::take::take_record_batch;
use datafusion::error::DataFusionError;
use datafusion::{
    physical_plan::{ColumnarValue, PhysicalExpr},
    scalar::ScalarValue,
};
use futures::{
    Stream,
    stream::{self},
};
use lance_arrow::SchemaExt;
use lance_core::{Error, ROW_ADDR_FIELD, ROW_ID_FIELD, Result};
use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};

// "update if" expressions typically compare fields from the source table to the target table.
// These tables have the same schema and so filter expressions need to differentiate.  To do that
// we wrap the left side and the right side in a struct and make a single "combined schema"
fn combined_schema(source: &Schema, target: &Schema) -> Schema {
    let nullable = |schema: &Schema| -> Fields {
        schema
            .fields
            .iter()
            .map(|f| f.as_ref().clone().with_nullable(true))
            .collect()
    };
    Schema::new(vec![
        Field::new("source", DataType::Struct(nullable(source)), false),
        Field::new("target", DataType::Struct(nullable(target)), false),
    ])
}

// This takes a double-wide table (e.g. the result of the outer join below) and takes the left
// half, puts it into a struct, then takes the right half, and puts that into a struct.  This
// makes the table match the "combined schema" so we can apply an "update if" expression
fn unzip_batch(batch: &RecordBatch, combined: &Schema) -> Result<RecordBatch> {
    let missing =
        |name: &str| Error::internal(format!("merge insert join is missing column '{name}'"));
    let struct_fields = |idx: usize| -> Result<Fields> {
        match combined.field(idx).data_type() {
            DataType::Struct(fields) => Ok(fields.clone()),
            other => Err(Error::internal(format!(
                "merge insert condition schema holds {other} where a struct was expected"
            ))),
        }
    };

    let source_fields = struct_fields(0)?;
    let width = source_fields.len();
    let source_arrays = source_fields
        .iter()
        .map(|f| {
            batch
                .column_by_name(f.name())
                .cloned()
                .ok_or_else(|| missing(f.name()))
        })
        .collect::<Result<Vec<_>>>()?;
    let source = StructArray::try_new(source_fields, source_arrays, None)?;

    let target_fields = struct_fields(1)?;
    let target_arrays = target_fields
        .iter()
        .map(|f| {
            target_column_index(batch.schema_ref(), f.name(), width)
                .map(|i| batch.column(i).clone())
                .ok_or_else(|| missing(&format!("target_{}", f.name())))
        })
        .collect::<Result<Vec<_>>>()?;
    let target = StructArray::try_new(target_fields, target_arrays, None)?;

    Ok(RecordBatch::try_new(
        Arc::new(combined.clone()),
        vec![Arc::new(source), Arc::new(target)],
    )?)
}

// A sync-safe structure that is shared by all of the "process batch" tasks.
//
// Note: we are not currently using parallelism but this still needs to be sync because it is
//       held across an await boundary (and we might use parallelism someday)
#[derive(Debug, Clone)]
pub(super) struct Merger {
    // As the merger runs it will update the list of deleted rows
    pub(super) deleted_rows: Arc<Mutex<Vec<u64>>>,
    // Shared collection to capture row ids that need to be updated
    pub(super) updating_row_ids: Arc<Mutex<CapturedRowIds>>,
    // Physical delete expression, only set if params.delete_not_matched_by_source is DeleteIf
    delete_expr: Option<Arc<dyn PhysicalExpr>>,
    // User statistics for merging
    pub(super) merge_stats: Arc<Mutex<MergeStats>>,
    // Only set if params.when_matched is UpdateIf
    match_filter: Option<MatchFilter>,
    // The parameters controlling the merge
    params: MergeInsertParams,
    /// Whether the output schema should include a row address column
    with_row_addr: bool,
    /// Carry row IDs until full-schema upsert output is ordered for writing.
    pub(super) with_row_id: bool,
    /// The output schema of the stream.
    output_schema: Arc<Schema>,
    /// Whether to enable stable row ids
    enable_stable_row_ids: bool,
    /// Set to track processed row IDs to detect duplicates
    processed_row_ids: Arc<Mutex<HashSet<u64>>>,
    /// Set to track non-null keys of rows inserted by FirstSeen mode
    processed_insert_keys: Arc<Mutex<InsertedKeyTracker>>,
    /// Inserted keys for declared primary-key conflict detection; absent otherwise.
    pub(super) inserted_rows_filter: Arc<Mutex<Option<KeyExistenceFilterBuilder>>>,
}

impl Merger {
    // Creates a new merger with an empty set of deleted rows, compiles expressions, if present
    pub(super) fn try_new(
        params: MergeInsertParams,
        schema: Arc<Schema>,
        joined_schema: &Schema,
        with_row_addr: bool,
        enable_stable_row_ids: bool,
        inserted_rows_filter: Option<KeyExistenceFilterBuilder>,
    ) -> Result<Self> {
        let delete_expr = if let WhenNotMatchedBySource::DeleteIf(expr) =
            &params.delete_not_matched_by_source
        {
            let planner = Planner::new(schema.clone());
            let expr = planner.optimize_expr(expr.clone())?;
            let physical_expr = planner.create_physical_expr(&expr)?;
            let data_type = physical_expr.data_type(&schema)?;
            if data_type != DataType::Boolean {
                return Err(Error::invalid_input(format!(
                    "Merge insert conditions must be expressions that return a boolean value, received expression ({}) which has data type {}",
                    expr, data_type
                )));
            }
            Some(physical_expr)
        } else {
            None
        };
        let match_filter = match &params.when_matched {
            WhenMatched::UpdateIf(_) | WhenMatched::UpdateIfExpr(_) => {
                let target_schema = target_schema_for(&schema, joined_schema)?;
                let combined_schema = Arc::new(combined_schema(&schema, &target_schema));
                let planner = Planner::new(combined_schema.clone());
                let expr = match &params.when_matched {
                    WhenMatched::UpdateIf(expr_str) => planner.parse_filter(expr_str)?,
                    WhenMatched::UpdateIfExpr(expr) => expr.clone(),
                    _ => unreachable!(),
                };
                let expr = planner.optimize_expr(expr)?;
                let match_expr = planner.create_physical_expr(&expr)?;
                let data_type = match_expr.data_type(combined_schema.as_ref())?;
                if data_type != DataType::Boolean {
                    return Err(Error::invalid_input(format!(
                        "Merge insert conditions must be expressions that return a boolean value, received a 'when matched update if' expression ({}) which has data type {}",
                        expr, data_type
                    )));
                }
                Some(MatchFilter {
                    schema: combined_schema,
                    expr: match_expr,
                })
            }
            _ => None,
        };
        let with_row_id = enable_stable_row_ids
            && !with_row_addr
            && params.insert_not_matched
            && matches!(
                params.when_matched,
                WhenMatched::UpdateAll | WhenMatched::UpdateIf(_) | WhenMatched::UpdateIfExpr(_)
            );
        let output_schema = if with_row_addr {
            Arc::new(schema.try_with_column(ROW_ADDR_FIELD.clone())?)
        } else if with_row_id {
            Arc::new(schema.try_with_column(ROW_ID_FIELD.clone())?)
        } else {
            schema
        };

        Ok(Self {
            deleted_rows: Arc::new(Mutex::new(Vec::new())),
            updating_row_ids: Arc::new(Mutex::new(CapturedRowIds::new(enable_stable_row_ids))),
            delete_expr,
            merge_stats: Arc::new(Mutex::new(MergeStats::default())),
            match_filter,
            params,
            with_row_addr,
            with_row_id,
            output_schema,
            enable_stable_row_ids,
            processed_row_ids: Arc::new(Mutex::new(HashSet::new())),
            processed_insert_keys: Arc::new(Mutex::new(InsertedKeyTracker::default())),
            inserted_rows_filter: Arc::new(Mutex::new(inserted_rows_filter)),
        })
    }

    pub(super) fn output_schema(&self) -> &Arc<Schema> {
        &self.output_schema
    }

    // Retrieves a bitmap of rows where at least one of the given columns is
    // not null.
    fn not_all_null(batch: &RecordBatch, cols: &[usize]) -> Result<BooleanArray> {
        // For our purposes we know there is always at least 1 on key
        debug_assert!(!cols.is_empty());
        let mut at_least_one_valid = arrow::compute::is_not_null(batch.column(cols[0]))?;
        for &idx in &cols[1..] {
            let is_valid = arrow::compute::is_not_null(batch.column(idx))?;
            at_least_one_valid = arrow::compute::or(&at_least_one_valid, &is_valid)?;
        }
        Ok(at_least_one_valid)
    }

    // Returns selection bitmaps for source-only, matched, and target-only rows.
    fn extract_selections(
        &self,
        combined_batch: &RecordBatch,
        row_id_col: usize,
    ) -> Result<(BooleanArray, BooleanArray, BooleanArray)> {
        // Locate source keys by name: a reordered partial-schema source can put
        // an all-null payload before its keys (lancedb/lancedb#3515).
        let combined_schema = combined_batch.schema();
        let source_key_cols = self
            .params
            .on
            .iter()
            .map(|key| {
                combined_schema.index_of(key).map_err(|_| {
                    Error::internal(format!(
                        "merge insert key column '{}' not found in joined batch",
                        key
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Every target row has a row id, even when its keys are NULL. A NULL
        // target row id therefore identifies a source-only row regardless of
        // its key values. With SQL equality, matched rows have non-null source
        // keys; target-only rows have NULL-padded source keys.
        let has_source_key = Self::not_all_null(combined_batch, &source_key_cols)?;
        let in_right = arrow::compute::is_not_null(combined_batch.column(row_id_col))?;
        let left_only = arrow::compute::not(&in_right)?;
        let in_both = arrow::compute::and(&has_source_key, &in_right)?;
        let right_only = arrow::compute::and(&arrow::compute::not(&has_source_key)?, &in_right)?;
        Ok((left_only, in_both, right_only))
    }

    // Given a batch of outer join data, split it into three different batches
    //
    // Process each sub-batch according to the merge insert params
    //
    // Returns 0, 1, or 2 batches
    // Potentially updates (as a side-effect) the deleted rows vec
    pub(super) async fn execute_batch(
        self,
        batch: RecordBatch,
    ) -> datafusion::common::Result<impl Stream<Item = datafusion::common::Result<RecordBatch>>>
    {
        let mut merge_statistics = self.merge_stats.lock().unwrap();
        let num_fields = batch.schema().fields.len();
        // The schema of the combined batches will be:
        // source_keys, source_payload, target_keys, target_payload, row_id, row_addr?
        // The keys and non_keys on both sides will be equal
        let (row_id_col, row_addr_col, right_offset) = if num_fields % 2 == 1 {
            // No rowaddr
            assert!(!self.with_row_addr);
            (num_fields - 1, None, num_fields / 2)
        } else {
            // Has rowaddr
            assert!(self.with_row_addr);
            (num_fields - 2, Some(num_fields - 1), (num_fields - 2) / 2)
        };

        let output_row_col = row_addr_col.or(self.with_row_id.then_some(row_id_col));
        let output_cols = (0..right_offset).chain(output_row_col).collect::<Vec<_>>();
        let right_cols_with_id = Vec::from_iter(right_offset..num_fields);

        let mut batches = Vec::with_capacity(2);
        let (left_only, in_both, right_only) = self.extract_selections(&batch, row_id_col)?;

        // There is no contention on this mutex.  We're only using it to bypass the rust
        // borrow checker (the stream needs to be `sync` since it crosses an await point)
        let mut deleted_row_ids = self.deleted_rows.lock().unwrap();

        // Each `WhenMatched` variant handles the matched rows (`in_both`)
        // differently.
        let match_filter = self.match_filter;
        match &self.params.when_matched {
            WhenMatched::DoNothing => {}
            WhenMatched::Delete => {
                // Matched rows are removed, not rewritten: record their row ids
                // for the commit to delete and emit no replacement batch. A
                // source with duplicate keys matches the same target row more
                // than once; apply the same `source_dedupe_behavior` policy as
                // updates so a duplicate either aborts (`Fail`) or is skipped
                // and counted once (`FirstSeen`) — the commit deletes the row a
                // single time regardless.
                let matched = arrow::compute::filter_record_batch(&batch, &in_both)?;
                let row_ids = matched.column(row_id_col).as_primitive::<UInt64Type>();

                let mut processed_row_ids = self.processed_row_ids.lock().unwrap();
                for (row_idx, &row_id) in row_ids.values().iter().enumerate() {
                    if processed_row_ids.insert(row_id) {
                        merge_statistics.num_deleted_rows += 1;
                        deleted_row_ids.push(row_id);
                    } else {
                        match self.params.source_dedupe_behavior {
                            SourceDedupeBehavior::Fail => {
                                return Err(create_duplicate_row_error(
                                    &matched,
                                    row_idx,
                                    &self.params.on,
                                ));
                            }
                            SourceDedupeBehavior::FirstSeen => {
                                merge_statistics.num_skipped_duplicates += 1;
                            }
                        }
                    }
                }
            }
            WhenMatched::Fail => {
                // Any matched row aborts the whole operation.
                if let Some(row_idx) = (0..in_both.len()).find(|&i| in_both.value(i)) {
                    return Err(DataFusionError::Execution(format!(
                        "Merge insert failed: found matching row with key values: {}",
                        format_key_values_on_columns(&batch, row_idx, &self.params.on)
                    )));
                }
            }
            WhenMatched::UpdateAll | WhenMatched::UpdateIf(_) | WhenMatched::UpdateIfExpr(_) => {
                let mut matched = arrow::compute::filter_record_batch(&batch, &in_both)?;

                if let Some(match_filter) = match_filter {
                    let unzipped = unzip_batch(&matched, &match_filter.schema)?;
                    let filtered = match_filter.expr.evaluate(&unzipped)?;
                    match filtered {
                        ColumnarValue::Array(mask) => {
                            // Some rows matched, filter down and replace those rows
                            matched =
                                arrow::compute::filter_record_batch(&matched, mask.as_boolean())?;
                        }
                        ColumnarValue::Scalar(scalar) => {
                            if let ScalarValue::Boolean(Some(true)) = scalar {
                                // All rows matched, go ahead and replace the whole batch
                            } else {
                                // Nothing matched, replace nothing
                                matched = RecordBatch::new_empty(matched.schema());
                            }
                        }
                    }
                }

                merge_statistics.num_updated_rows += matched.num_rows() as u64;

                // If the filter eliminated all rows then its important we don't try and write
                // the batch at all.  Writing an empty batch currently panics
                if matched.num_rows() > 0 {
                    let row_ids = matched.column(row_id_col).as_primitive::<UInt64Type>();

                    let mut processed_row_ids = self.processed_row_ids.lock().unwrap();
                    let mut keep_indices: Vec<u32> = Vec::with_capacity(matched.num_rows());
                    for (row_idx, &row_id) in row_ids.values().iter().enumerate() {
                        if processed_row_ids.insert(row_id) {
                            keep_indices.push(row_idx as u32);
                        } else {
                            match self.params.source_dedupe_behavior {
                                SourceDedupeBehavior::Fail => {
                                    return Err(create_duplicate_row_error(
                                        &matched,
                                        row_idx,
                                        &self.params.on,
                                    ));
                                }
                                SourceDedupeBehavior::FirstSeen => {
                                    // Skip this duplicate row (don't add to keep_indices)
                                }
                            }
                        }
                    }
                    drop(processed_row_ids);

                    // Filter out duplicate rows if any were skipped
                    let num_skipped = matched.num_rows() - keep_indices.len();
                    if num_skipped > 0 {
                        merge_statistics.num_skipped_duplicates += num_skipped as u64;
                        merge_statistics.num_updated_rows -= num_skipped as u64;

                        let indices = UInt32Array::from(keep_indices);
                        matched = take_record_batch(&matched, &indices)?;
                    }

                    // Only process and write if there are remaining rows after filtering duplicates
                    if matched.num_rows() > 0 {
                        // Get row_ids again after filtering (if any duplicates were removed)
                        let row_ids = matched.column(row_id_col).as_primitive::<UInt64Type>();
                        deleted_row_ids.extend(row_ids.values());
                        if self.enable_stable_row_ids {
                            self.updating_row_ids
                                .lock()
                                .unwrap()
                                .capture(row_ids.values())?;
                        }

                        let matched = matched.project(&output_cols)?;
                        // The payload columns of an outer join are always nullable.  We need to restore
                        // non-nullable to columns that were originally non-nullable.  This should be safe
                        // since the not_matched rows should all be valid on the right_cols
                        //
                        // Sadly we can't use with_schema because it doesn't let you toggle nullability
                        let matched = RecordBatch::try_new(
                            self.output_schema.clone(),
                            Vec::from_iter(matched.columns().iter().cloned()),
                        )?;
                        batches.push(Ok(matched));
                    }
                }
            }
        }
        if self.params.insert_not_matched {
            let mut not_matched = arrow::compute::filter_record_batch(&batch, &left_only)?;
            if self.params.source_dedupe_behavior == SourceDedupeBehavior::FirstSeen {
                let mut processed_insert_keys = self.processed_insert_keys.lock().unwrap();
                let mut keep_indices = Vec::with_capacity(not_matched.num_rows());
                for row_idx in 0..not_matched.num_rows() {
                    if processed_insert_keys.insert(&not_matched, row_idx, &self.params.on)? {
                        keep_indices.push(row_idx as u32);
                    } else {
                        merge_statistics.num_skipped_duplicates += 1;
                    }
                }
                drop(processed_insert_keys);

                if keep_indices.len() != not_matched.num_rows() {
                    not_matched =
                        take_record_batch(&not_matched, &UInt32Array::from(keep_indices))?;
                }
            }
            // Track only rows that will actually be inserted, after source deduplication.
            if let Some(filter) = self
                .inserted_rows_filter
                .lock()
                .map_err(|_| Error::internal("Merge insert key filter lock poisoned"))?
                .as_mut()
            {
                for row_idx in 0..not_matched.num_rows() {
                    if let Some(key) =
                        extract_key_value_from_batch(&not_matched, row_idx, &self.params.on)
                    {
                        filter.insert(key)?;
                    }
                }
            }
            let not_matched = not_matched.project(&output_cols)?;
            // See comment above explaining this schema replacement
            let not_matched = RecordBatch::try_new(
                self.output_schema.clone(),
                Vec::from_iter(not_matched.columns().iter().cloned()),
            )?;

            merge_statistics.num_inserted_rows += not_matched.num_rows() as u64;
            batches.push(Ok(not_matched));
        }
        match self.params.delete_not_matched_by_source {
            WhenNotMatchedBySource::Delete => {
                let unmatched = arrow::compute::filter(batch.column(row_id_col), &right_only)?;
                merge_statistics.num_deleted_rows += unmatched.len() as u64;
                let row_ids = unmatched.as_primitive::<UInt64Type>();
                deleted_row_ids.extend(row_ids.values());
            }
            WhenNotMatchedBySource::DeleteIf(_) => {
                let target_data = batch.project(&right_cols_with_id)?;
                let unmatched = arrow::compute::filter_record_batch(&target_data, &right_only)?;
                let row_id_col = unmatched.num_columns() - 1;
                let to_delete = self.delete_expr.unwrap().evaluate(&unmatched)?;

                match to_delete {
                    ColumnarValue::Array(mask) => {
                        let row_ids = arrow::compute::filter(
                            unmatched.column(row_id_col),
                            mask.as_boolean(),
                        )?;
                        let row_ids = row_ids.as_primitive::<UInt64Type>();
                        merge_statistics.num_deleted_rows += row_ids.len() as u64;
                        deleted_row_ids.extend(row_ids.values());
                    }
                    ColumnarValue::Scalar(scalar) => {
                        if let ScalarValue::Boolean(Some(true)) = scalar {
                            let row_ids = unmatched.column(row_id_col).as_primitive::<UInt64Type>();
                            merge_statistics.num_deleted_rows += row_ids.len() as u64;
                            deleted_row_ids.extend(row_ids.values());
                        }
                    }
                }
            }
            WhenNotMatchedBySource::Keep => {}
        }

        Ok(stream::iter(batches))
    }
}

/// Index of the target-side column `target_<name>` in a joined schema; the source
/// columns come first, so the search starts after them.
fn target_column_index(joined: &Schema, name: &str, num_source_columns: usize) -> Option<usize> {
    let target_name = format!("target_{name}");
    joined
        .fields()
        .iter()
        .skip(num_source_columns)
        .position(|f| f.name() == &target_name)
        .map(|i| num_source_columns + i)
}

/// The target side of an "update if" condition: each source column's `target_<name>`
/// field as the join emits it (the indexed take projects nested children to the source's
/// subset), renamed and in source order.
fn target_schema_for(source: &Schema, joined: &Schema) -> Result<Schema> {
    let width = source.fields.len();
    let fields = source
        .fields
        .iter()
        .map(|f| {
            target_column_index(joined, f.name(), width)
                .map(|i| joined.field(i).as_ref().clone().with_name(f.name()))
                .ok_or_else(|| {
                    Error::internal(format!(
                        "merge insert join is missing column 'target_{}'",
                        f.name()
                    ))
                })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Schema::new(fields))
}

/// Physical "when matched update if" condition with the schema it evaluates against.
#[derive(Debug, Clone)]
struct MatchFilter {
    schema: Arc<Schema>,
    expr: Arc<dyn PhysicalExpr>,
}
