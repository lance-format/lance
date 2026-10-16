// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashSet;
use std::ops::Range;
use std::sync::Arc;

use arrow_array::{Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema, SchemaRef, SortOptions};
use arrow_select::concat::concat_batches;
use datafusion::catalog::Session;
use datafusion::common::exec_datafusion_err;
use datafusion::config::ConfigOptions;
use datafusion::logical_expr::Expr;
use datafusion::physical_expr::PhysicalSortExpr;
use datafusion::physical_optimizer::output_requirements::OutputRequirementExec;
use datafusion::physical_plan::display::DisplayableExecutionPlan;
use datafusion::physical_plan::projection::ProjectionExec as DFProjectionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::{ExecutionPlan, SendableRecordBatchStream, expressions};
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::{Distribution, LexOrdering, PhysicalExpr};
use futures::future::BoxFuture;
use futures::stream::StreamExt;
use futures::{FutureExt, TryStreamExt};
use lance_arrow::DataTypeExt;
use lance_core::datatypes::{Field, OnMissing, Projection};
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::{ROW_ADDR, ROW_ID, ROW_OFFSET};
use lance_datafusion::exec::{
    LanceExecutionOptions, StrictBatchSizeExec, analyze_plan, execute_plan,
};
use lance_index::scalar::expression::PlannerIndexExt;
use lance_index::scalar::inverted::{
    DOC_INDEX_COL, DOC_INDEX_FIELD, DocumentGranularity, SCORE_COL,
};
use lance_index::vector::DIST_COL;
use lance_io::stream::RecordBatchStream;
use tracing::instrument;

use crate::datatypes::Schema;
use crate::index::DatasetIndexInternalExt;
use crate::io::exec::filtered_read::filter_read_columns;
use crate::io::exec::knn::{QUERY_INDEX_COL, query_index_field};
use crate::io::exec::{
    AddRowOffsetExec, FilterPlan as ExprFilterPlan, Planner, get_physical_optimizer,
};
use crate::{Error, Result};

use super::aggregate::AggregateExpr;
use super::source::TakeOperation;
use super::{
    ColumnOrdering, DatasetRecordBatchStream, FilterPlan, MaterializationStyle, QueryFilter,
    ScanRangePlan, Scanner, validate_batch_size,
};

impl Scanner {
    /// The Arrow schema of the output, including projections and vector / _distance
    pub async fn schema(&self) -> Result<SchemaRef> {
        let plan = self.create_plan().await?;
        Ok(plan.schema())
    }

    /// Fetches the currently set expr filter
    ///
    /// Note that this forces the filter to be evaluated and the result will depend on
    /// the current state of the scanner (e.g. if with_row_id has been called then _rowid
    /// will be available for filtering but not otherwise) and so you may want to call this
    /// after setting all other options.
    pub fn get_expr_filter(&self) -> Result<Option<Expr>> {
        if let Some(filter) = &self.filter.expr_filter {
            let filter_schema = self.filterable_schema()?;
            Ok(Some(filter.to_datafusion(
                self.dataset.schema(),
                filter_schema.as_ref(),
            )?))
        } else {
            Ok(None)
        }
    }

    fn add_extra_columns(
        &self,
        schema: Schema,
        fts_document_granularity: Option<DocumentGranularity>,
    ) -> Result<Schema> {
        let mut extra_columns = vec![ArrowField::new(ROW_OFFSET, DataType::UInt64, true)];

        if self.nearest.is_some() || self.minhash_query.is_some() {
            extra_columns.push(ArrowField::new(DIST_COL, DataType::Float32, true));
        }
        if self.is_batch_nearest {
            extra_columns.push(query_index_field());
        }

        if self.full_text_query.is_some() {
            extra_columns.push(ArrowField::new(SCORE_COL, DataType::Float32, true));
            if fts_document_granularity
                .map(DocumentGranularity::is_list_element)
                // `get_expr_filter` is synchronous, so an omitted granularity
                // cannot be resolved from index metadata here. Include the
                // column conservatively; `create_plan` uses the resolved value.
                .unwrap_or(true)
            {
                extra_columns.push(DOC_INDEX_FIELD.clone());
            }
        }

        schema.merge(&ArrowSchema::new(extra_columns))
    }

    /// The full schema available to filters
    ///
    /// This is the schema of the dataset, any metadata columns like _rowid or _rowaddr
    /// and any extra columns like _distance or _score
    fn filterable_schema(&self) -> Result<Arc<Schema>> {
        let fts_document_granularity = self
            .full_text_query
            .as_ref()
            .and_then(|query| self.fts_document_granularity(&query.query).ok());
        self.filterable_schema_with_fts_granularity(fts_document_granularity)
    }

    fn filterable_schema_with_fts_granularity(
        &self,
        fts_document_granularity: Option<DocumentGranularity>,
    ) -> Result<Arc<Schema>> {
        let base_schema = Projection::full(self.dataset.clone())
            .with_row_id()
            .with_row_addr()
            .with_row_last_updated_at_version()
            .with_row_created_at_version()
            .to_schema();

        Ok(Arc::new(self.add_extra_columns(
            base_schema,
            fts_document_granularity,
        )?))
    }

    /// This takes the current output, and the user's requested projection, and calculates the
    /// final projection expression.
    ///
    /// This final expression may reorder columns, drop columns, or calculate new columns
    pub(crate) fn calculate_final_projection(
        &self,
        current_schema: &ArrowSchema,
    ) -> Result<Vec<(Arc<dyn PhysicalExpr>, String)>> {
        // Select the columns from the output schema based on the user's projection (or the list
        // of all available columns if the user did not specify a projection)
        let mut output_expr = self.projection_plan.to_physical_exprs(current_schema)?;

        if self.full_text_query.is_some()
            && current_schema.field_with_name(DOC_INDEX_COL).is_ok()
            && output_expr.iter().all(|(_, name)| name != DOC_INDEX_COL)
        {
            output_expr.push((
                expressions::col(DOC_INDEX_COL, current_schema)?,
                DOC_INDEX_COL.to_string(),
            ));
        }

        // Make sure _distance and _score are _always_ in the output unless user has opted out of the legacy
        // projection behavior
        if self.autoproject_scoring_columns {
            if (self.nearest.is_some() || self.minhash_query.is_some())
                && output_expr.iter().all(|(_, name)| name != DIST_COL)
            {
                if self.explicit_projection {
                    log::warn!(
                        "Deprecation warning, this behavior will change in the future. This search specified output columns but did not include `_distance`.  Currently the `_distance` column will be included.  In the future it will not.  Call `disable_scoring_autoprojection` to adopt the future behavior and avoid this warning"
                    );
                }
                let vector_expr = expressions::col(DIST_COL, current_schema)?;
                output_expr.push((vector_expr, DIST_COL.to_string()));
            }
            if self.full_text_query.is_some()
                && output_expr.iter().all(|(_, name)| name != SCORE_COL)
            {
                if self.explicit_projection {
                    log::warn!(
                        "Deprecation warning, this behavior will change in the future. This search specified output columns but did not include `_score`.  Currently the `_score` column will be included.  In the future it will not.  Call `disable_scoring_autoprojection` to adopt the future behavior and avoid this warning"
                    );
                }
                let score_expr = expressions::col(SCORE_COL, current_schema)?;
                output_expr.push((score_expr, SCORE_COL.to_string()));
            }
        }

        // Batch nearest queries expose the synthetic `query_index` discriminator as
        // the first output column for compatibility with LanceDB batch vector search.
        if self.is_batch_nearest {
            let query_index_expr = if let Some(pos) = output_expr
                .iter()
                .position(|(_, name)| name == QUERY_INDEX_COL)
            {
                output_expr.remove(pos)
            } else {
                (
                    expressions::col(QUERY_INDEX_COL, current_schema)?,
                    QUERY_INDEX_COL.to_string(),
                )
            };
            output_expr.insert(0, query_index_expr);
        }

        if self.legacy_with_row_id {
            let row_id_pos = output_expr
                .iter()
                .position(|(_, name)| name == ROW_ID)
                .ok_or_else(|| {
                    Error::internal(
                        "user specified with_row_id but the _rowid column was not in the output"
                            .to_string(),
                    )
                })?;
            if row_id_pos != output_expr.len() - 1 {
                // Row id is not last column.  Need to rotate it to the last spot.
                let row_id_expr = output_expr.remove(row_id_pos);
                output_expr.push(row_id_expr);
            }
        }

        if self.legacy_with_row_addr {
            let row_addr_pos = output_expr.iter().position(|(_, name)| name == ROW_ADDR).ok_or_else(|| {
                Error::internal("user specified with_row_address but the _rowaddr column was not in the output".to_string())
            })?;
            if row_addr_pos != output_expr.len() - 1 {
                // Row addr is not last column.  Need to rotate it to the last spot.
                let row_addr_expr = output_expr.remove(row_addr_pos);
                output_expr.push(row_addr_expr);
            }
        }

        Ok(output_expr)
    }

    /// The `order_by` keys as columns of the final projection's output, or `None`
    /// when a key does not pass through the projection unchanged.
    fn ordering_in_output(
        ordering: &[ColumnOrdering],
        projection: &[(Arc<dyn PhysicalExpr>, String)],
    ) -> Option<LexOrdering> {
        let sort_exprs = ordering
            .iter()
            .map(|col| {
                let index = projection.iter().position(|(expr, _)| {
                    expr.downcast_ref::<Column>()
                        .is_some_and(|column| column.name() == col.column_name)
                })?;
                Some(PhysicalSortExpr {
                    expr: Arc::new(Column::new(&projection[index].1, index)),
                    options: SortOptions {
                        descending: !col.ascending,
                        nulls_first: col.nulls_first,
                    },
                })
            })
            .collect::<Option<Vec<_>>>()?;
        LexOrdering::new(sort_exprs)
    }

    /// Create a stream from the Scanner.
    #[instrument(skip_all)]
    pub fn try_into_stream(&self) -> BoxFuture<'_, Result<DatasetRecordBatchStream>> {
        // Future intentionally boxed here to avoid large futures on the stack
        async move {
            let plan = self.create_plan().await?;

            Ok(DatasetRecordBatchStream::new(execute_plan(
                plan,
                LanceExecutionOptions {
                    batch_size: self.batch_size,
                    execution_stats_callback: self.scan_stats_callback.clone(),
                    ..Default::default()
                },
            )?))
        }
        .boxed()
    }

    pub(crate) async fn try_into_dfstream(
        &self,
        mut options: LanceExecutionOptions,
    ) -> Result<SendableRecordBatchStream> {
        let plan = self.create_plan().await?;

        // Use the scan stats callback if the user didn't set an execution stats callback
        if options.execution_stats_callback.is_none() {
            options.execution_stats_callback = self.scan_stats_callback.clone();
        }

        execute_plan(plan, options)
    }

    pub(crate) fn execution_options(&self) -> LanceExecutionOptions {
        LanceExecutionOptions {
            batch_size: self.batch_size,
            execution_stats_callback: self.scan_stats_callback.clone(),
            ..Default::default()
        }
    }

    pub async fn try_into_batch(&self) -> Result<RecordBatch> {
        let stream = self.try_into_stream().await?;
        let schema = stream.schema();
        let batches = stream.try_collect::<Vec<_>>().await?;
        Ok(concat_batches(&schema, &batches)?)
    }

    /// Scan and return the number of matching rows
    ///
    /// Note: calling [`Dataset::count_rows`] can be more efficient than calling this method
    /// especially if there is no filter.
    ///
    /// [`Dataset::count_rows`]: crate::dataset::Dataset::count_rows
    #[instrument(skip_all)]
    pub fn count_rows(&self) -> BoxFuture<'_, Result<u64>> {
        // Future intentionally boxed here to avoid large futures on the stack
        async move {
            let mut scanner = self.clone();
            scanner.aggregate(AggregateExpr::builder().count_star().build())?;

            let plan = scanner.create_plan().await?;
            let mut stream = execute_plan(plan, LanceExecutionOptions::default())?;

            // A count plan will always return a single batch with a single row.
            if let Some(first_batch) = stream.next().await {
                let batch = first_batch?;
                let array = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or(Error::invalid_input(
                        "Count plan did not return an Int64Array".to_string(),
                    ))?;
                Ok(array.value(0) as u64)
            } else {
                Ok(0)
            }
        }
        .boxed()
    }

    // A "narrow" field is a field that is so small that we are better off reading the
    // entire column and filtering in memory rather than "take"ing the column.
    //
    // The exact threshold depends on a two factors:
    // 1. The number of rows returned by the filter
    // 2. The number of rows in the dataset
    // 3. The IOPS/bandwidth ratio of the storage system
    // 4. The size of each value in the column
    //
    // We don't (today) have a good way of knowing #1 or #4.  #2 is easy to know.  We can
    // combine 1 & 2 into "percentage of rows returned" but since we don't know #1 it
    // doesn't really help.  #3 is complex but as a rule of thumb we can use:
    //
    //   Local storage: 1 IOP for ever ten thousand bytes
    //   Cloud storage: 1 IOP for every million bytes
    //
    // Our current heuristic today is to assume a filter will return 0.1% of the rows in the dataset.
    //
    // This means, for cloud storage, a field is "narrow" if there are 1KB of data per row and
    // for local disk a field is "narrow" if there are 10 bytes of data per row.
    fn is_early_field(&self, field: &Field) -> bool {
        match self.materialization_style {
            MaterializationStyle::AllEarly => true,
            MaterializationStyle::AllLate => false,
            MaterializationStyle::AllEarlyExcept(ref cols) => !cols.contains(&(field.id as u32)),
            MaterializationStyle::Heuristic => {
                if field.is_blob() && self.blob_handling.returns_description(field) {
                    // A blob returned as a description (offset + size) is tiny, so it is
                    // cheaper to read eagerly. When blob_handling materializes the full
                    // binary value instead (e.g. `all_binary`), fall through to the
                    // width-based heuristic so a selective filter can late-materialize it
                    // rather than reading the whole column.
                    return true;
                }

                let byte_width = field.data_type().byte_width_opt();
                let is_cloud = self.dataset.object_store.as_ref().is_cloud();
                if is_cloud {
                    byte_width.is_some_and(|bw| bw < 1000)
                } else {
                    byte_width.is_some_and(|bw| bw < 10)
                }
            }
        }
    }

    // If we are going to filter on `filter_plan`, then which columns are so small it is
    // cheaper to read the entire column and filter in memory.
    //
    // Note: only add columns that we actually need to read
    pub(super) fn calc_eager_projection(
        &self,
        filter_plan: &ExprFilterPlan,
        desired_projection: &Projection,
    ) -> Result<Projection> {
        // Note: We use all_columns and not refine_columns here.  If a column is covered by an index but
        // the user has requested it, then we do not use it for late materialization.
        //
        // Either that column is covered by an exact filter (e.g. string with bitmap/btree) and there is no
        // need for late materialization or that column is covered by an inexact filter (e.g. ngram) in which
        // case we are going to load the column anyways for the recheck.
        let filter_columns = filter_plan
            .full_expr
            .as_ref()
            .map(|expr| filter_read_columns(expr, desired_projection))
            .unwrap_or_default();

        let filter_schema = self
            .dataset
            .empty_projection()
            .union_columns(filter_columns, OnMissing::Error)?
            .into_schema();

        let mut late_ids = HashSet::new();
        for field in self.dataset.schema().fields.iter() {
            self.collect_late_field_ids(field, false, &mut late_ids);
        }

        // Start with the desired fields
        Ok(desired_projection
            .clone()
            // Subtract columns that are expensive
            .subtract_predicate(|f| late_ids.contains(&f.id))
            // Add back columns that we need for filtering
            .union_schema(&filter_schema))
    }

    // Collects the ids of the fields that should be materialized late (see calc_eager_projection).
    //
    // `forced` is true when a non-struct ancestor is late, in which case this field and all of
    // its descendants are late regardless of their own width.
    fn collect_late_field_ids(&self, field: &Field, forced: bool, late_ids: &mut HashSet<i32>) {
        let is_late = forced || !self.is_early_field(field);
        if is_late {
            late_ids.insert(field.id);
        }
        let force_children = forced || (is_late && !field.logical_type.is_struct());
        for child in field.children.iter() {
            self.collect_late_field_ids(child, force_children, late_ids);
        }
    }

    fn validate_options(&self) -> Result<()> {
        if let Some(query) = &self.nearest {
            query.validate_search_effort()?;
        }
        if let Some(QueryFilter::Vector(query)) = &self.filter.query_filter {
            query.validate_search_effort()?;
        }
        if self.batch_readahead == 0 {
            return Err(Error::invalid_input_source(
                "batch_readahead must be greater than 0, got 0".into(),
            ));
        }

        if self.materialization_readahead_bytes == Some(0) {
            return Err(Error::invalid_input_source(
                "materialization_readahead_bytes must be greater than 0, got 0".into(),
            ));
        }

        if let Some(batch_size) = self.batch_size {
            validate_batch_size(batch_size)?;
        }

        if self.strict_batch_size
            && let Some(batch_size_bytes) = self
                .resolved_file_reader_options()
                .and_then(|options| options.batch_size_bytes)
        {
            return Err(Error::invalid_input_source(
                format!(
                    "strict_batch_size=true cannot be combined with batch_size_bytes={batch_size_bytes}; strict row batching can merge batches beyond the byte limit"
                )
                .into(),
            ));
        }

        if self.include_deleted_rows && !self.projection_plan.physical_projection.with_row_id {
            return Err(Error::invalid_input_source(
                "include_deleted_rows is set but with_row_id is false".into(),
            ));
        }

        if self.aggregate.is_some() {
            if self.limit.is_some() || self.offset.is_some() {
                return Err(Error::invalid_input_source(
                    "Cannot use limit/offset with aggregate. Apply limit to the result instead."
                        .into(),
                ));
            }
            if self.ordering.is_some() {
                return Err(Error::invalid_input_source(
                    "Cannot use order_by with aggregate. Apply ordering to the result instead."
                        .into(),
                ));
            }
        }

        if self.index_segments.is_some() && self.nearest.is_none() {
            return Err(Error::not_supported(
                "with_index_segments is only supported for vector search".to_string(),
            ));
        }

        if self.physical_row_addr_prefilter.is_some() {
            if self.nearest.is_some() || self.full_text_query.is_some() {
                return Err(Error::not_supported(
                    "with_physical_row_addr_prefilter is not supported for vector or full-text search",
                ));
            }
            if self
                .dataset
                .manifest()
                .data_storage_format
                .lance_file_format()
                == lance_file::version::ConcreteFileVersion::V1
            {
                return Err(Error::not_supported(
                    "with_physical_row_addr_prefilter is not supported on legacy-storage datasets",
                ));
            }
        }

        Ok(())
    }

    async fn create_filter_plan(
        &self,
        use_scalar_index: bool,
        query_filter: Option<QueryFilter>,
        fts_document_granularity: Option<DocumentGranularity>,
    ) -> Result<FilterPlan> {
        let filter_schema =
            self.filterable_schema_with_fts_granularity(fts_document_granularity)?;
        let planner = Planner::new(Arc::new(filter_schema.as_ref().into()));

        // Check expr filter
        let filter_plan = if let Some(filter) = self.filter.expr_filter.as_ref() {
            let expr = filter.to_datafusion(self.dataset.schema(), filter_schema.as_ref())?;
            let index_info = match self.scalar_index_selection.as_ref() {
                Some(selection) => {
                    self.dataset
                        .scalar_index_info_for_segments(selection.segments.as_ref())
                        .await?
                }
                None => self.dataset.scalar_index_info().await?,
            };
            let filter_plan =
                planner.create_filter_plan(expr.clone(), &index_info, use_scalar_index)?;

            // This tests if any of the fragments are missing the physical_rows property (old style)
            // If they are then we cannot use scalar indices
            if filter_plan.index_query.is_some() {
                let fragments = if let Some(fragments) = self.fragments.as_ref() {
                    fragments
                } else {
                    self.dataset.fragments()
                };
                let mut has_missing_row_count = false;
                for frag in fragments {
                    if frag.physical_rows.is_none() {
                        has_missing_row_count = true;
                        break;
                    }
                }
                if has_missing_row_count {
                    // We need row counts to use scalar indices.  If we don't have them then
                    // fallback to a non-indexed filter
                    let filter_plan =
                        planner.create_filter_plan(expr.clone(), &index_info, false)?;
                    FilterPlan::new(query_filter.clone(), filter_plan)
                } else {
                    FilterPlan::new(query_filter.clone(), filter_plan)
                }
            } else {
                FilterPlan::new(query_filter.clone(), filter_plan)
            }
        } else {
            FilterPlan::new(query_filter, ExprFilterPlan::default())
        };

        if let Some(selection) = self.scalar_index_selection.as_ref()
            && filter_plan.expr_filter_plan.index_query.is_none()
        {
            return Err(Error::invalid_input(format!(
                "Filter cannot be planned using scalar index '{}'",
                selection.index_name
            )));
        }

        // Check query filter
        if filter_plan.query_filter.is_some()
            && self.nearest.is_none()
            && self.full_text_query.is_none()
        {
            return Err(Error::invalid_input_source(
                "Query filter can only be used with full text search or vector search".into(),
            ));
        }
        if self.nearest.is_some() && filter_plan.vector_filter().is_some() {
            return Err(Error::invalid_input_source(
                "Query filter can't be used with vector search".into(),
            ));
        }
        if self.full_text_query.is_some() && filter_plan.fts_filter().is_some() {
            return Err(Error::invalid_input_source(
                "Fts filter can't be used with fts search".into(),
            ));
        }

        Ok(filter_plan)
    }

    fn requested_scan_range(&self, total_rows: u64) -> Option<Range<u64>> {
        let offset = u64::try_from(self.offset.unwrap_or(0)).ok()?;
        let start = offset.min(total_rows);
        let end = match self.limit {
            Some(limit) => offset
                .checked_add(u64::try_from(limit).ok()?)?
                .min(total_rows),
            None => total_rows,
        };
        Some(start..end)
    }

    fn plan_metadata_scan_range(&self) -> Option<ScanRangePlan> {
        self.dataset.manifest().writer_version.as_ref()?;

        let fragments = self
            .fragments
            .as_deref()
            .unwrap_or_else(|| self.dataset.fragments());
        let mut visible_prefix = Vec::with_capacity(fragments.len() + 1);
        visible_prefix.push(0_u64);

        for fragment in fragments {
            // Metadata-only pruning is an optimization. Any missing or inconsistent count must
            // preserve the unpruned path instead of guessing where a visible ordinal lands.
            let physical_rows = u64::try_from(fragment.physical_rows?).ok()?;
            let deleted_rows = match fragment.deletion_file.as_ref() {
                Some(deletion_file) => u64::try_from(deletion_file.num_deleted_rows?).ok()?,
                None => 0,
            };
            let visible_rows = physical_rows.checked_sub(deleted_rows)?;
            visible_prefix.push(visible_prefix.last()?.checked_add(visible_rows)?);
        }

        let requested_range = self.requested_scan_range(*visible_prefix.last()?)?;
        let start = requested_range.start;
        let end = requested_range.end;

        if start == end {
            return Some(ScanRangePlan {
                fragments: Some(Arc::new(Vec::new())),
                range: 0..0,
            });
        }

        let first_fragment = visible_prefix
            .windows(2)
            .position(|window| window[1] > start)?;
        let final_visible_ordinal = end.checked_sub(1)?;
        let last_fragment = visible_prefix
            .windows(2)
            .position(|window| window[1] > final_visible_ordinal)?;
        let visible_before_first = visible_prefix[first_fragment];
        let selected_fragments = fragments[first_fragment..=last_fragment].to_vec();
        Some(ScanRangePlan {
            fragments: Some(Arc::new(selected_fragments)),
            range: start.checked_sub(visible_before_first)?
                ..end.checked_sub(visible_before_first)?,
        })
    }

    pub(super) async fn plan_scan_range(
        &self,
        filter_plan: &ExprFilterPlan,
    ) -> Result<Option<ScanRangePlan>> {
        if filter_plan.has_any_filter() {
            // If there is a filter we can't pushdown limit / offset
            Ok(None)
        } else if self.ordering.is_some() {
            // If there is ordering, we can't pushdown limit / offset
            // because we need to sort all data first before applying the limit
            Ok(None)
        } else if self.include_deleted_rows || (self.limit.is_none() && self.offset.is_none()) {
            // FilteredReadOptions does not support scan ranges when deleted rows are included.
            Ok(None)
        } else {
            let fragments = self
                .fragments
                .as_deref()
                .unwrap_or_else(|| self.dataset.fragments());
            let has_deletions = fragments
                .iter()
                .any(|fragment| fragment.deletion_file.is_some());
            let is_v1 = self
                .dataset
                .manifest()
                .data_storage_format
                .lance_file_format()
                == lance_file::version::ConcreteFileVersion::V1;

            // V2 range planning is independent of row-ID encoding. Manifest live-row counts
            // remove whole fragments first; filtered read loads deletion vectors only for the
            // retained fragments and trims the residual logical range inside them.
            if !is_v1 && let Some(plan) = self.plan_metadata_scan_range() {
                return Ok(Some(plan));
            }

            // Keep V1 behavior unchanged. For V2 fragments with deletions, incomplete metadata
            // cannot safely locate the requested visible ordinals, so retain the non-pushdown
            // fallback instead of guessing.
            if has_deletions && (!is_v1 || self.dataset.manifest().uses_stable_row_ids()) {
                return Ok(None);
            }

            let total_rows = u64::try_from(self.dataset.count_all_rows().await?).map_err(|_| {
                Error::internal("Dataset row count does not fit in u64".to_string())
            })?;
            Ok(self
                .requested_scan_range(total_rows)
                .map(|range| ScanRangePlan {
                    fragments: self.fragments.clone().map(Arc::new),
                    range,
                }))
        }
    }

    /// Create [`ExecutionPlan`] for Scan.
    ///
    /// An ExecutionPlan is a graph of operators that can be executed.
    ///
    /// The following plans are supported:
    ///
    ///  - **Plain scan without filter or limits.**
    ///
    ///  ```ignore
    ///  Scan(projections)
    ///  ```
    ///
    ///  - **Scan with filter and/or limits.**
    ///
    ///  ```ignore
    ///  Scan(filtered_cols) -> Filter(expr)
    ///     -> (*LimitExec(limit, offset))
    ///     -> Take(remaining_cols) -> Projection()
    ///  ```
    ///
    ///  - **Use KNN Index (with filter and/or limits)**
    ///
    /// ```ignore
    /// KNNIndex() -> Take(vector) -> FlatRefine()
    ///     -> Take(filtered_cols) -> Filter(expr)
    ///     -> (*LimitExec(limit, offset))
    ///     -> Take(remaining_cols) -> Projection()
    /// ```
    ///
    /// - **Use KNN flat (brute force) with filter and/or limits**
    ///
    /// ```ignore
    /// Scan(vector) -> FlatKNN()
    ///     -> Take(filtered_cols) -> Filter(expr)
    ///     -> (*LimitExec(limit, offset))
    ///     -> Take(remaining_cols) -> Projection()
    /// ```
    ///
    /// In general, a plan has 5 stages:
    ///
    /// 1. Source (from dataset Scan or from index, may include prefilter)
    /// 2. Filter
    /// 3. Sort
    /// 4. Limit / Offset
    /// 5. Take remaining columns / Projection
    pub fn create_plan(&self) -> BoxFuture<'_, Result<Arc<dyn ExecutionPlan>>> {
        Box::pin(self.create_plan_impl(None))
    }

    pub(crate) fn create_plan_with_session<'a>(
        &'a self,
        session: &'a dyn Session,
    ) -> BoxFuture<'a, Result<Arc<dyn ExecutionPlan>>> {
        Box::pin(self.create_plan_impl(Some(session)))
    }

    #[instrument(level = "debug", skip_all)]
    async fn create_plan_impl(
        &self,
        session: Option<&dyn Session>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        log::trace!("creating scanner plan");
        self.validate_options()?;
        if let Some(rows) = &self.physical_row_addr_prefilter {
            crate::dataset::fragment_slice::validate_physical_rows(&self.dataset, rows).await?;
        }

        let full_text_query = match &self.full_text_query {
            Some(query) => Some(self.resolve_full_text_search_query(query).await?),
            None => None,
        };
        let query_filter = match &self.filter.query_filter {
            Some(QueryFilter::Fts(query)) => Some(QueryFilter::Fts(
                self.resolve_full_text_search_query(query).await?,
            )),
            Some(QueryFilter::Vector(query)) => Some(QueryFilter::Vector(query.clone())),
            None => None,
        };
        let fts_document_granularity = full_text_query
            .as_ref()
            .map(|query| self.fts_document_granularity(&query.query))
            .transpose()?;

        // Scalar indices are only used when prefiltering
        let use_scalar_index = self.use_scalar_index && (self.prefilter || self.nearest.is_none());
        let mut filter_plan = self
            .create_filter_plan(use_scalar_index, query_filter, fts_document_granularity)
            .await?;

        let mut use_limit_node = true;
        // Source: either a (K|A)NN search, full text search, or a (full|indexed) scan
        let mut plan: Arc<dyn ExecutionPlan> = match (
            &self.nearest,
            &full_text_query,
            &self.minhash_query,
        ) {
            (Some(_), None, None) => self.vector_search_source(&mut filter_plan).await?,
            (None, Some(query), None) => self.fts_search_source(&mut filter_plan, query).await?,
            (None, None, Some(query)) => {
                self.minhash_search_source(&mut filter_plan, query).await?
            }
            (None, None, None) => {
                if self.projection_plan.has_output_cols()
                    && self.projection_plan.physical_projection.is_empty()
                {
                    // This means the user is doing something like `SELECT 1 AS foo`.  We don't support this and
                    // I'm not sure we should.  Users should use a full SQL API to do something like this.
                    //
                    // It's also possible we get here from `SELECT does_not_exist`

                    // Note: even though we are just going to return an error we still want to calculate the
                    // final projection here.  This lets us distinguish between a user doing something like:
                    //
                    // SELECT 1 FROM t (not supported error)
                    // SELECT non_existent_column FROM t (column not found error)
                    let output_expr = self.calculate_final_projection(&ArrowSchema::empty())?;
                    return Err(Error::not_supported_source(format!("Scans must request at least one column.  Received only dynamic expressions: {:?}", output_expr).into()));
                }

                let take_op = filter_plan
                    .expr_filter_plan
                    .full_expr
                    .as_ref()
                    .and_then(TakeOperation::try_from_expr);
                if let Some((take_op, remainder)) = take_op {
                    // If there is any remainder use it as the filter (we don't even try and combine an indexed
                    // search on the filter with a take as that seems excessive)
                    filter_plan.expr_filter_plan = remainder
                        .map(ExprFilterPlan::new_refine_only)
                        .unwrap_or(ExprFilterPlan::default());
                    self.take_source(take_op).await?
                } else {
                    let planned_read = self
                        .filtered_read_source(&mut filter_plan.expr_filter_plan, session)
                        .await?;
                    if planned_read.limit_pushed_down {
                        use_limit_node = false;
                    }
                    if planned_read.filter_pushed_down {
                        filter_plan.disable_refine();
                    }
                    planned_read.plan
                }
            }
            (Some(_), Some(_), None) => {
                return Err(Error::invalid_input_source(
                    "Cannot have both nearest and full text search".into(),
                ));
            }
            _ => {
                return Err(Error::invalid_input_source(
                    "Cannot combine a MinHash search with nearest or full text search".into(),
                ));
            }
        };

        // Load columns needed for filter and ordering
        let mut pre_filter_projection = self.dataset.empty_projection();

        // We may need to take filter columns if we are going to refine
        // an indexed scan.
        if filter_plan.has_refine() {
            // It's ok for some filter columns to be missing (e.g. _rowid)
            pre_filter_projection = pre_filter_projection.union_columns(
                filter_plan.refine_columns(&self.dataset).await?,
                OnMissing::Ignore,
            )?;
        }

        // TODO: Does it always make sense to take the ordering columns here?  If there is a filter then
        // maybe we wait until after the filter to take the ordering columns?  Maybe it would be better to
        // grab the ordering column in the initial scan (if it is eager) and if it isn't then we should
        // take it after the filtering phase, if any (we already have a take there).
        if let Some(ordering) = &self.ordering {
            pre_filter_projection = pre_filter_projection.union_columns(
                ordering.iter().map(|col| &col.column_name),
                OnMissing::Error,
            )?;
        }

        plan = self.take(plan, pre_filter_projection)?;

        // Filter
        plan = filter_plan.refine_filter(plan, self, session).await?;

        // Aggregate (if set, applies aggregate and returns early)
        if let Some(agg) = &self.aggregate {
            // Take only columns needed by the aggregate, not the full projection.
            // For COUNT(*), this is empty. For SUM(x), this is just [x].
            let required_columns = agg.required_columns();
            let agg_projection = if required_columns.is_empty() {
                self.dataset.empty_projection()
            } else {
                self.dataset
                    .empty_projection()
                    .union_columns(&required_columns, OnMissing::Error)?
            };
            plan = self.take(plan, agg_projection)?;
            plan = self.apply_aggregate(plan, agg).await?;

            let optimizer = get_physical_optimizer();
            let mut options = ConfigOptions::default();
            options.execution.target_partitions = self
                .target_parallelism
                .unwrap_or_else(get_num_compute_intensive_cpus);
            for rule in optimizer.rules {
                plan = rule.optimize(plan, &options)?;
            }

            return Ok(plan);
        }

        // Sort
        if let Some(ordering) = &self.ordering {
            let ordering_columns = ordering.iter().map(|col| &col.column_name);
            let projection_with_ordering = self
                .dataset
                .empty_projection()
                .union_columns(ordering_columns, OnMissing::Error)?;
            // We haven't loaded the sort column yet so take it now
            plan = self.take(plan, projection_with_ordering)?;
            let col_exprs = ordering
                .iter()
                .map(|col| {
                    Ok(PhysicalSortExpr {
                        expr: Self::create_column_expr(
                            &col.column_name,
                            &self.dataset,
                            plan.schema().as_ref(),
                        )?,
                        options: SortOptions {
                            descending: !col.ascending,
                            nulls_first: col.nulls_first,
                        },
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            plan = Arc::new(SortExec::new(
                LexOrdering::new(col_exprs)
                    .ok_or(exec_datafusion_err!("Unexpected empty sort expressions"))?,
                plan,
            ));
        }

        // Limit / offset
        if use_limit_node && (self.limit.is_some() || self.offset.is_some()) {
            plan = self.limit_node(plan);
        }

        // Take remaining columns required for projection
        plan = self.take(plan, self.projection_plan.physical_projection.clone())?;

        // Add system columns, if requested
        if self.projection_plan.must_add_row_offset {
            plan = Arc::new(AddRowOffsetExec::try_new(plan, self.dataset.clone()).await?);
        }

        // Final projection
        let final_projection = self.calculate_final_projection(plan.schema().as_ref())?;
        let output_sort = self
            .ordering
            .as_ref()
            .and_then(|ordering| Self::ordering_in_output(ordering, &final_projection));

        plan = Arc::new(DFProjectionExec::try_new(final_projection, plan)?);

        // If requested, apply a strict batch size to the final output
        if self.strict_batch_size {
            plan = Arc::new(StrictBatchSizeExec::new(plan, self.get_batch_size()));
        }

        let optimizer = get_physical_optimizer();
        let mut options = ConfigOptions::default();
        options.execution.target_partitions = self
            .target_parallelism
            .unwrap_or_else(get_num_compute_intensive_cpus);
        // Rows must come back in plan order unless the caller opted out and
        // nothing ranks them.
        let order_matters = self.ordered
            || self.ordering.is_some()
            || self.nearest.is_some()
            || self.full_text_query.is_some()
            || self.minhash_query.is_some();
        // The order the output declares. With `order_by` it must cover every sort
        // key, which it cannot if the projection drops or changes one.
        let properties = plan.properties();
        let visible_order = match (&self.ordering, output_sort) {
            (None, _) => properties.output_ordering().cloned(),
            (Some(_), Some(output_sort)) => properties
                .equivalence_properties()
                .ordering_satisfy(output_sort.clone())?
                .then_some(output_sort),
            (Some(_), None) => None,
        };
        // Round-robin partitions are merged back in order only by an ordering the
        // output declares; otherwise they arrive in whichever order they finish.
        options.optimizer.enable_round_robin_repartition =
            !order_matters || visible_order.is_some();
        if let Some(order) = visible_order {
            // `EnforceDistribution` then merges any partitions back in this order
            plan = Arc::new(OutputRequirementExec::new(
                plan,
                Some(order.into()),
                Distribution::SinglePartition,
                None,
            ));
            // Otherwise the merge is replaced by a coalesce and a full re-sort
            options.optimizer.prefer_existing_sort = true;
        }
        for rule in optimizer.rules {
            plan = rule.optimize(plan, &options)?;
        }

        Ok(plan)
    }

    // Check if a filter plan references version columns
    pub(super) fn filter_references_version_columns(&self, filter_plan: &ExprFilterPlan) -> bool {
        use lance_core::{ROW_CREATED_AT_VERSION, ROW_LAST_UPDATED_AT_VERSION};

        if let Some(refine_expr) = &filter_plan.refine_expr {
            let column_names = Planner::column_names_in_expr(refine_expr);
            for col_name in column_names {
                if col_name == ROW_CREATED_AT_VERSION || col_name == ROW_LAST_UPDATED_AT_VERSION {
                    return true;
                }
            }
        }
        false
    }

    #[instrument(level = "info", skip(self))]
    pub async fn analyze_plan(&self) -> Result<String> {
        let plan = self.create_plan().await?;
        analyze_plan(
            plan,
            LanceExecutionOptions {
                batch_size: self.batch_size,
                ..Default::default()
            },
        )
        .await
    }

    #[instrument(level = "info", skip(self))]
    pub async fn explain_plan(&self, verbose: bool) -> Result<String> {
        // Box the plan-building future at the call site: `create_plan`'s inlined async
        // layout otherwise exceeds rustc's depth limit here. It has to be boxed at the
        // call site rather than inside `create_plan` — boxing internally turns the
        // future's `Send` check into a `Box<Future>: Send` trait obligation that
        // overflows the solver through the cache types (E0275 in downstream crates).
        let plan = Box::pin(self.create_plan()).await?;
        let display = DisplayableExecutionPlan::new(plan.as_ref());

        Ok(format!("{}", display.indent(verbose)))
    }

    /// Run [`Self::count_rows`]'s underlying plan and return it formatted with
    /// runtime metrics. Equivalent to [`Self::analyze_plan`] but with a
    /// `COUNT(*)` aggregate auto-applied first — the only way for callers
    /// without a hand-built `AggregateExpr` (e.g. the Python bindings) to
    /// inspect the plan that `count_rows` actually executed.
    #[instrument(level = "info", skip(self))]
    pub async fn analyze_count_plan(&self) -> Result<String> {
        let mut scanner = self.clone();
        scanner.aggregate(AggregateExpr::builder().count_star().build())?;
        let plan = scanner.create_plan().await?;
        analyze_plan(
            plan,
            LanceExecutionOptions {
                batch_size: self.batch_size,
                ..Default::default()
            },
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::datatypes::Int32Type;
    use arrow_schema::Fields;
    use lance_datagen::{BatchCount, RowCount, array, gen_batch};
    use lance_file::version::LanceFileVersion;

    use crate::dataset::Dataset;
    use crate::dataset::scanner::test_dataset::{
        TestVectorDataset, assert_scan_slice_matches, batch_knn_two_queries, find_filtered_read,
    };

    #[tokio::test]
    async fn test_stable_row_id_limit_pushdown_with_deletions() -> Result<()> {
        let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true).await?;
        let mut dataset = test_ds.dataset;

        assert_scan_slice_matches(&dataset, None, 19, 2).await?;
        let mut scan = dataset.scan();
        scan.limit(Some(2), Some(19))?;
        let scan_range_plan = scan
            .plan_scan_range(&ExprFilterPlan::default())
            .await?
            .expect("scan range");
        assert_eq!(scan_range_plan.range, 19..21);

        dataset
            .delete("i IN (0, 20, 25, 198, 200, 205, 399)")
            .await?;

        // Deletions before OFFSET, within the boundary fragment, at the LIMIT endpoint,
        // and a LIMIT spanning fragments all match full-scan slice semantics.
        for (offset, limit) in [
            (0, 10),
            (19, 2),
            (20, 10),
            (190, 20),
            (196, 10),
            (200, 10),
            (390, 10),
            (500, 10),
            (196, 0),
        ] {
            assert_scan_slice_matches(&dataset, None, offset, limit).await?;
        }

        // Fragment 0 has 196 visible rows, so an offset on that exact boundary can
        // discard it and express the remaining range relative to fragment 1.
        let mut scan = dataset.scan();
        scan.limit(Some(10), Some(196))?;
        let scan_range_plan = scan
            .plan_scan_range(&ExprFilterPlan::default())
            .await?
            .expect("scan range");
        assert_eq!(scan_range_plan.range, 0..10);
        let plan = scan.create_plan().await?;
        let filtered = find_filtered_read(plan.as_ref()).expect("filtered read");
        let selected = filtered
            .options()
            .fragments
            .as_ref()
            .expect("pruned fragments");
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].id, 1);

        // Explicit fragment subsets and order define scan order; stable row IDs need not be
        // contiguous or increasing.
        let fragments = dataset.fragments().as_ref().clone();
        assert_scan_slice_matches(&dataset, Some(&fragments[1..]), 10, 10).await?;
        let reversed = vec![fragments[1].clone(), fragments[0].clone()];
        assert_scan_slice_matches(&dataset, Some(&reversed), 197, 10).await?;
        let mut reversed_scan = dataset.scan();
        reversed_scan.with_fragments(reversed);
        reversed_scan.limit(Some(10), Some(197))?;
        let reversed_plan = reversed_scan.create_plan().await?;
        let filtered = find_filtered_read(reversed_plan.as_ref()).expect("filtered read");
        let selected = filtered
            .options()
            .fragments
            .as_ref()
            .expect("pruned fragments");
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].id, 0);

        // A zero-live-row fragment does not consume visible ordinals and can be pruned.
        let mut fully_deleted = fragments[0].clone();
        let physical_rows = fully_deleted.physical_rows;
        fully_deleted
            .deletion_file
            .as_mut()
            .expect("deletion file")
            .num_deleted_rows = physical_rows;
        let mut fully_deleted_scan = dataset.scan();
        fully_deleted_scan.with_fragments(vec![fully_deleted, fragments[1].clone()]);
        fully_deleted_scan.limit(Some(10), Some(0))?;
        let fully_deleted_plan = fully_deleted_scan
            .plan_scan_range(&ExprFilterPlan::default())
            .await?
            .expect("scan range");
        assert_eq!(fully_deleted_plan.range, 0..10);
        assert_eq!(
            fully_deleted_plan.fragments.expect("pruned fragments")[0].id,
            1
        );

        // Unknown or inconsistent metadata conservatively preserves the fallback.
        let mut unknown_count = dataset.fragments()[0].clone();
        unknown_count
            .deletion_file
            .as_mut()
            .expect("deletion file")
            .num_deleted_rows = None;
        let mut unknown_scan = dataset.scan();
        unknown_scan.with_fragments(vec![unknown_count]);
        unknown_scan.limit(Some(10), Some(1))?;
        assert!(
            unknown_scan
                .plan_scan_range(&ExprFilterPlan::default())
                .await?
                .is_none()
        );

        let mut missing_physical_rows = dataset.fragments()[0].clone();
        missing_physical_rows.physical_rows = None;
        let mut missing_rows_scan = dataset.scan();
        missing_rows_scan.with_fragments(vec![missing_physical_rows]);
        missing_rows_scan.limit(Some(10), Some(1))?;
        assert!(
            missing_rows_scan
                .plan_scan_range(&ExprFilterPlan::default())
                .await?
                .is_none()
        );

        let mut invalid_deleted_rows = dataset.fragments()[0].clone();
        let physical_rows = invalid_deleted_rows.physical_rows.expect("physical rows");
        invalid_deleted_rows
            .deletion_file
            .as_mut()
            .expect("deletion file")
            .num_deleted_rows = physical_rows.checked_add(1);
        let mut invalid_rows_scan = dataset.scan();
        invalid_rows_scan.with_fragments(vec![invalid_deleted_rows]);
        invalid_rows_scan.limit(Some(10), Some(1))?;
        assert!(
            invalid_rows_scan
                .plan_scan_range(&ExprFilterPlan::default())
                .await?
                .is_none()
        );

        let mut include_deleted_scan = dataset.scan();
        include_deleted_scan.with_row_id().include_deleted_rows();
        include_deleted_scan.limit(Some(10), Some(1))?;
        assert!(
            include_deleted_scan
                .plan_scan_range(&ExprFilterPlan::default())
                .await?
                .is_none()
        );

        let mut untrusted_dataset = dataset.clone();
        Arc::make_mut(&mut untrusted_dataset.manifest).writer_version = None;
        let mut untrusted_scan = untrusted_dataset.scan();
        untrusted_scan.limit(Some(10), Some(1))?;
        assert!(
            untrusted_scan
                .plan_scan_range(&ExprFilterPlan::default())
                .await?
                .is_none()
        );

        let legacy_test_ds = TestVectorDataset::new(LanceFileVersion::Legacy, true).await?;
        let mut legacy_dataset = legacy_test_ds.dataset;
        legacy_dataset.delete("i = 0").await?;
        let mut legacy_scan = legacy_dataset.scan();
        legacy_scan.limit(Some(10), Some(1))?;
        assert!(
            legacy_scan
                .plan_scan_range(&ExprFilterPlan::default())
                .await?
                .is_none()
        );
        assert_scan_slice_matches(&legacy_dataset, None, 1, 10).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_search_effort_validation_after_probe_setters() {
        let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
            .await
            .unwrap();
        let mut scan = test_ds.dataset.scan();
        let error = scan.search_effort(0.75).err().unwrap();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("requires nearest"));
        let (queries, _) = batch_knn_two_queries();
        scan.nearest("vec", &queries, 2).unwrap();
        scan.search_effort(0.75).unwrap();
        let error = scan.search_effort(f64::NAN).err().unwrap();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("must be finite"));
        assert_eq!(scan.nearest.as_ref().unwrap().search_effort, 0.75);
        scan.nprobes(2);
        scan.validate_options().unwrap();
        assert_eq!(scan.nearest.as_ref().unwrap().minimum_nprobes, 1);
        assert_eq!(scan.nearest.as_ref().unwrap().maximum_nprobes, None);
        scan.search_effort(0.5).unwrap();
        scan.validate_options().unwrap();
        scan.search_effort(1.0).unwrap();
        assert_eq!(scan.nearest.as_ref().unwrap().search_effort, 1.0);
    }

    #[tokio::test]
    async fn test_late_materialize_list_as_unit() {
        // A list's `item` child is narrow, so deciding per field kept it eager and pulled the
        // whole list back in.  A list must be deferred together with its children.
        let struct_fields = Fields::from(vec![
            ArrowField::new("count", DataType::Int32, true),
            ArrowField::new(
                "tokens",
                DataType::List(Arc::new(ArrowField::new("item", DataType::Int32, true))),
                true,
            ),
        ]);
        let data = gen_batch()
            .col("i", array::step::<Int32Type>())
            .col("toks", array::rand_list(&DataType::Int32, false))
            .col("st", array::rand_struct(struct_fields))
            .into_reader_rows(RowCount::from(10), BatchCount::from(1));
        let dataset = Arc::new(Dataset::write(data, "memory://test", None).await.unwrap());
        let scanner = dataset.scan();
        let planner = Planner::new(Arc::new(dataset.schema().into()));
        let eager = |filter: &str, project: [&str; 1]| {
            let filter_plan =
                ExprFilterPlan::new_refine_only(planner.parse_filter(filter).unwrap());
            let desired = dataset
                .empty_projection()
                .union_columns(project, OnMissing::Error)
                .unwrap();
            scanner
                .calc_eager_projection(&filter_plan, &desired)
                .unwrap()
                .to_bare_schema()
        };

        // Top-level list: only the narrow filter column is read eagerly
        let schema = eager("i > 5", ["toks"]);
        assert!(schema.field("i").is_some(), "{schema:?}");
        assert!(schema.field("toks").is_none(), "{schema:?}");

        // List nested in a struct: the narrow sibling is still read eagerly
        let schema = eager("st.count IS NOT NULL", ["st.tokens"]);
        let eager_children = schema.field("st").map(|st| {
            st.children
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>()
        });
        assert_eq!(eager_children, Some(vec!["count"]), "{schema:?}");
    }
}
