// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use arrow_schema::DataType;
use datafusion::catalog::Session;
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::ExecutionPlan;
#[allow(deprecated)]
use datafusion::physical_plan::coalesce_batches::CoalesceBatchesExec;
use datafusion::physical_plan::limit::GlobalLimitExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::union::UnionExec;
use datafusion::scalar::ScalarValue;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use futures::FutureExt;
use futures::future::BoxFuture;
use lance_core::datatypes::{OnMissing, Projection};
use lance_core::error::LanceOptionExt;
use lance_core::utils::address::RowAddress;
use lance_core::{ROW_ADDR, ROW_ID, ROW_OFFSET};
use lance_datafusion::exec::OneShotExec;
use lance_datafusion::expr::safe_coerce_scalar;
use lance_index::scalar::expression::ScalarIndexExpr;
use lance_select::result::IndexExprResultWireFormat;
use lance_select::{IndexExprResult, RowAddrMask, RowAddrTreeMap};
use lance_table::format::{Fragment, IndexMetadata};
use roaring::RoaringBitmap;

use crate::dataset::overlay::{collect_overlay_stale_rows_for_segment, overlaid_fragments};
use crate::dataset::rowids::{live_row_addrs_to_row_ids, translate_addr_treemap_to_row_ids};
use crate::dataset::{row_offsets_to_row_addresses, versions};
use crate::datatypes::Schema;
use crate::index::scalar_logical::load_named_scalar_segments;
use crate::io::exec::filtered_read::{
    FilteredReadExec, FilteredReadOptions, FilteredReadThreadingMode,
};
use crate::io::exec::scalar_index::{MaterializeIndexExec, ScalarIndexExec};
use crate::io::exec::{
    AddRowAddrExec, FilterPlan as ExprFilterPlan, LanceFilterExec, LancePushdownScanExec,
    LanceScanConfig, LanceScanExec, Planner, ScanConfig, TakeExec, project,
};
use crate::{Error, Result};

use super::{
    DEFAULT_IO_BUFFER_SIZE, LEGACY_DEFAULT_FRAGMENT_READAHEAD, PlannedFilteredScan, Scanner,
    validate_batch_size,
};

/// Represents a user-requested take operation
#[derive(Debug, Clone)]
pub enum TakeOperation {
    /// Take rows by row id
    RowIds(Vec<u64>),
    /// Take rows by row address
    RowAddrs(Vec<u64>),
    /// Take rows by row offset
    ///
    /// The row offset is the offset of the row in the dataset.  This can
    /// be converted to row addresses using the fragment sizes.
    RowOffsets(Vec<u64>),
}

impl TakeOperation {
    fn extract_u64_list(list: &[Expr]) -> Option<Vec<u64>> {
        let mut u64s = Vec::with_capacity(list.len());
        for expr in list {
            if let Expr::Literal(lit, _) = expr {
                if let Some(ScalarValue::UInt64(Some(val))) =
                    safe_coerce_scalar(lit, &DataType::UInt64)
                {
                    u64s.push(val);
                } else {
                    return None;
                }
            } else {
                return None;
            }
        }
        Some(u64s)
    }

    fn merge(self, other: Self) -> Option<Self> {
        match (self, other) {
            (Self::RowIds(mut left), Self::RowIds(right)) => {
                left.extend(right);
                Some(Self::RowIds(left))
            }
            (Self::RowAddrs(mut left), Self::RowAddrs(right)) => {
                left.extend(right);
                Some(Self::RowAddrs(left))
            }
            (Self::RowOffsets(mut left), Self::RowOffsets(right)) => {
                left.extend(right);
                Some(Self::RowOffsets(left))
            }
            _ => None,
        }
    }

    /// Attempts to create a take operation from an expression.  This will succeed if the expression
    /// has one of the following forms:
    ///  - `_rowid = 10`
    ///  - `_rowid = 10 OR _rowid = 20 OR _rowid = 30`
    ///  - `_rowid IN (10, 20, 30)`
    ///  - `_rowaddr = 10`
    ///  - `_rowaddr = 10 OR _rowaddr = 20 OR _rowaddr = 30`
    ///  - `_rowaddr IN (10, 20, 30)`
    ///  - `_rowoffset = 10`
    ///  - `_rowoffset = 10 OR _rowoffset = 20 OR _rowoffset = 30`
    ///  - `_rowoffset IN (10, 20, 30)`
    ///
    /// The _rowid / _rowaddr / _rowoffset determine if we are taking by row id, address, or offset.
    ///
    /// If a take expression is combined with some other filter via an AND then the remainder will be
    /// returned as well.  For example, `_rowid = 10` will return (take_op, None) and
    /// `_rowid = 10 AND x > 70` will return (take_op, Some(x > 70)).
    pub(super) fn try_from_expr(expr: &Expr) -> Option<(Self, Option<Expr>)> {
        if let Expr::BinaryExpr(binary) = expr {
            match binary.op {
                datafusion_expr::Operator::And => {
                    let left_take = Self::try_from_expr(&binary.left);
                    let right_take = Self::try_from_expr(&binary.right);
                    match (left_take, right_take) {
                        (Some(_), Some(_)) => {
                            // This is something like...
                            //
                            // _rowid = 10 AND _rowid = 20
                            //
                            // ...which is kind of nonsensical.  Better to just return None.
                            return None;
                        }
                        (Some((left_op, left_rem)), None) => {
                            let remainder = match left_rem {
                                // If there is a remainder on the left side we combine it.  This _should_
                                // be something like converting (_rowid = 10 AND x > 70) AND y > 80
                                // to (_rowid = 10) AND (x > 70 AND y > 80) which should be valid
                                Some(expr) => Expr::and(expr, binary.right.as_ref().clone()),
                                None => binary.right.as_ref().clone(),
                            };
                            return Some((left_op, Some(remainder)));
                        }
                        (None, Some((right_op, right_rem))) => {
                            let remainder = match right_rem {
                                Some(expr) => Expr::and(expr, binary.left.as_ref().clone()),
                                None => binary.left.as_ref().clone(),
                            };
                            return Some((right_op, Some(remainder)));
                        }
                        (None, None) => {
                            return None;
                        }
                    }
                }
                datafusion_expr::Operator::Eq => {
                    // Check for _rowid = literal
                    if let (Expr::Column(col), Expr::Literal(lit, _)) =
                        (binary.left.as_ref(), binary.right.as_ref())
                        && let Some(ScalarValue::UInt64(Some(val))) =
                            safe_coerce_scalar(lit, &DataType::UInt64)
                    {
                        if col.name == ROW_ID {
                            return Some((Self::RowIds(vec![val]), None));
                        } else if col.name == ROW_ADDR {
                            return Some((Self::RowAddrs(vec![val]), None));
                        } else if col.name == ROW_OFFSET {
                            return Some((Self::RowOffsets(vec![val]), None));
                        }
                    }
                }
                datafusion_expr::Operator::Or => {
                    let left_take = Self::try_from_expr(&binary.left);
                    let right_take = Self::try_from_expr(&binary.right);
                    if let (Some(left), Some(right)) = (left_take, right_take) {
                        if left.1.is_some() || right.1.is_some() {
                            // This would be something like...
                            //
                            // (_rowid = 10 AND x > 70) OR _rowid = 20
                            //
                            // I don't think it's correct to convert this into a take operation
                            // which would give us (_rowid = 10 OR _rowid = 20) AND x > 70
                            return None;
                        }
                        return left.0.merge(right.0).map(|op| (op, None));
                    }
                }
                _ => {}
            }
        } else if let Expr::InList(in_expr) = expr
            // A negated InList (`_rowid NOT IN (...)`) is the complement of
            // the listed ids, not a take of them. Small lists are expanded
            // into `!=` conjunctions by the expression simplifier before they
            // get here, but larger ones arrive as `negated: true`.
            && !in_expr.negated
            && let Expr::Column(col) = in_expr.expr.as_ref()
            && let Some(u64s) = Self::extract_u64_list(&in_expr.list)
        {
            if col.name == ROW_ID {
                return Some((Self::RowIds(u64s), None));
            } else if col.name == ROW_ADDR {
                return Some((Self::RowAddrs(u64s), None));
            } else if col.name == ROW_OFFSET {
                return Some((Self::RowOffsets(u64s), None));
            }
        }
        None
    }
}

impl Scanner {
    // Helper function for filtered_read
    //
    // Do not call this directly, use filtered_read instead
    //
    // First return value is the plan, second is whether the limit was pushed down
    pub(in crate::dataset) async fn legacy_filtered_read(
        &self,
        filter_plan: &ExprFilterPlan,
        projection: Projection,
        make_deletions_null: bool,
        fragments: Option<Arc<Vec<Fragment>>>,
        scan_range: Option<Range<u64>>,
        is_prefilter: bool,
    ) -> Result<PlannedFilteredScan> {
        let fragments = fragments.unwrap_or(self.dataset.fragments().clone());
        let mut filter_pushed_down = false;

        let plan: Arc<dyn ExecutionPlan> = if filter_plan.has_index_query() {
            if self.include_deleted_rows {
                return Err(Error::invalid_input_source(
                    "Cannot include deleted rows in a scalar indexed scan".into(),
                ));
            }
            self.scalar_indexed_scan(projection, filter_plan, fragments)
                .await
        } else if !is_prefilter
            && filter_plan.has_refine()
            && self.batch_size.is_none()
            && self.use_stats
            && !self.filter_references_version_columns(filter_plan)
        {
            filter_pushed_down = true;
            self.pushdown_scan(false, filter_plan)
        } else {
            let ordered = if self.ordering.is_some() || self.nearest.is_some() {
                // If we are sorting the results there is no need to scan in order
                false
            } else if projection.with_row_last_updated_at_version
                || projection.with_row_created_at_version
            {
                // Version columns require ordered scanning because version metadata
                // is indexed by position within each fragment
                true
            } else {
                self.ordered
            };

            let projection = if let Some(refine_expr) = filter_plan.refine_expr.as_ref() {
                if is_prefilter {
                    let refine_cols = Planner::column_names_in_expr(refine_expr);
                    projection.union_columns(refine_cols, OnMissing::Error)?
                } else {
                    projection
                }
            } else {
                projection
            };

            // Can't push down limit for legacy scan if there is a refine step
            let scan_range = if filter_plan.has_refine() {
                None
            } else {
                scan_range
            };

            let scan = self.scan_fragments(
                projection.with_row_id,
                self.projection_plan.physical_projection.with_row_addr,
                self.projection_plan
                    .physical_projection
                    .with_row_last_updated_at_version,
                self.projection_plan
                    .physical_projection
                    .with_row_created_at_version,
                make_deletions_null,
                Arc::new(projection.to_bare_schema()),
                fragments,
                scan_range,
                ordered,
            );

            if filter_plan.has_refine() && is_prefilter {
                Ok(Arc::new(LanceFilterExec::try_new(
                    filter_plan.refine_expr.clone().unwrap(),
                    scan,
                )?) as Arc<dyn ExecutionPlan>)
            } else {
                Ok(scan)
            }
        }?;
        Ok(PlannedFilteredScan {
            plan,
            limit_pushed_down: false,
            filter_pushed_down,
        })
    }

    pub(super) fn index_expr_result_format(&self) -> IndexExprResultWireFormat {
        if self.relational_algebra_version > 1 {
            IndexExprResultWireFormat::TwoMask
        } else {
            // In version 1 we used the legacy three-variant format for index expr results
            IndexExprResultWireFormat::ThreeVariant
        }
    }

    // A plain-scan external row mask is fed as the FilteredReadExec row source so
    // only masked rows are read, with any SQL filter applied as a refine on top.
    // Vector and full-text searches apply the mask via their own prefilter paths
    // (KNN external_mask / FTS build_prefilter / MinHash search node and its
    // flat branch), so this plain-scan source is scoped to scans that are none
    // of them. FTS and MinHash in particular have nearest.is_none(), so excluding
    // them here keeps their prefilter's own filtered read unmasked.
    fn use_external_mask(&self) -> bool {
        self.nearest.is_none()
            && self.full_text_query.is_none()
            && self.minhash_query.is_none()
            && self.external_row_mask.is_some()
    }

    // The filter plan actually handed to the filtered read. With an external mask
    // active the mask is the row source, so any SQL filter is demoted to a refine
    // on top of it; otherwise the plan is used as-is. Projection and scan-range
    // planning must be done against this, not the raw filter_plan, so refine
    // columns are retained and limit/offset is not pushed down before masking.
    fn effective_filter_plan(&self, filter_plan: &ExprFilterPlan) -> ExprFilterPlan {
        if self.use_external_mask() {
            match filter_plan.full_expr.clone() {
                Some(expr) => ExprFilterPlan::new_refine_only(expr),
                None => ExprFilterPlan::default(),
            }
        } else {
            filter_plan.clone()
        }
    }

    // Read options carrying the scanner's batch size, readahead and I/O settings.
    fn read_options(&self, projection: Projection) -> Result<FilteredReadOptions> {
        // Bound the decode fan-out by `batch_readahead`.
        let mut read_options = FilteredReadOptions::new(projection).with_threading_mode(
            FilteredReadThreadingMode::OnePartitionMultipleThreads(self.batch_readahead),
        );

        if let Some(batch_size) = self.batch_size {
            read_options = read_options.with_batch_size(validate_batch_size(batch_size)?);
        }

        if let Some(file_reader_options) = self.resolved_file_reader_options() {
            read_options = read_options.with_file_reader_options(file_reader_options);
        }

        if let Some(fragment_readahead) = self.fragment_readahead {
            read_options = read_options.with_fragment_readahead(fragment_readahead);
        }

        if let Some(io_buffer_size_bytes) = self.io_buffer_size {
            read_options = read_options.with_io_buffer_size(io_buffer_size_bytes);
        }

        if let Some(materialization_readahead_bytes) = self.materialization_readahead_bytes {
            read_options =
                read_options.with_materialization_readahead_bytes(materialization_readahead_bytes);
        }

        Ok(read_options)
    }

    // Helper function for filtered_read
    //
    // Do not call this directly, use filtered_read instead
    pub(in crate::dataset) async fn new_filtered_read(
        &self,
        filter_plan: &ExprFilterPlan,
        projection: Projection,
        make_deletions_null: bool,
        fragments: Option<Arc<Vec<Fragment>>>,
        scan_range: Option<Range<u64>>,
        session: Option<&dyn Session>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Kept for the overlay stale-Take path below, which re-evaluates blocked stale rows.
        let user_projection = projection.clone();
        let use_external_mask = self.use_external_mask();
        let effective_filter = self.effective_filter_plan(filter_plan);

        let mut read_options = self
            .read_options(projection)?
            .with_filter_plan(effective_filter);

        if let Some(rows) = &self.physical_row_addr_prefilter {
            read_options = read_options.with_physical_row_addr_prefilter(rows.clone());
        }

        if let Some(fragments) = fragments {
            read_options = read_options.with_fragments(fragments);
        }

        if let Some(scan_range) = scan_range {
            read_options = read_options.with_scan_range_before_filter(scan_range)?;
        }

        if make_deletions_null {
            read_options = read_options.with_deleted_rows()?;
        }

        if self.fast_search && filter_plan.has_index_query() {
            read_options = read_options.with_only_indexed_fragments();
        }

        if let Some(session) = session {
            read_options = read_options.with_physical_filters(session)?;
        }

        // Mask data overlay files: a row with an overlay committed after an index it relies on
        // touched an indexed field can no longer be trusted to that index. Block just those rows
        // from the index result (their fragments stay indexed, so non-stale rows keep the index)
        // and re-evaluate them on a targeted take path below — O(stale_rows), not O(fragment).
        let mut overlay_stale_rows: HashMap<u32, RoaringBitmap> = HashMap::new();
        if let Some(index_query) = filter_plan.index_query.as_ref() {
            let candidate_frags = read_options
                .fragments
                .clone()
                .unwrap_or_else(|| self.dataset.fragments().clone());
            overlay_stale_rows = self
                .overlay_stale_index_rows(index_query, &candidate_frags)
                .await?;
            if let Some(block) = self.stale_rows_block_mask(&overlay_stale_rows).await? {
                read_options = read_options.with_overlay_block(block);
            }
        }

        let result_format = self.index_expr_result_format();
        let index_input = match self.external_row_mask.as_deref() {
            Some(mask) if use_external_mask => Some(self.mask_as_take_input(mask.clone())?),
            _ => {
                if let Some(index_query) = filter_plan.index_query.clone() {
                    let target_fragments = read_options
                        .fragments
                        .as_ref()
                        .unwrap_or_else(|| self.dataset.fragments())
                        .iter()
                        .map(|fragment| fragment.id as u32)
                        .collect::<RoaringBitmap>();
                    let fragment_scope = ScalarIndexExec::fragments_covered_by_index_query(
                        &index_query,
                        self.dataset.as_ref(),
                    )
                    .await?
                        & target_fragments;
                    let mut exec =
                        ScalarIndexExec::new(self.dataset.clone(), index_query, result_format)
                            .with_fragment_scope(fragment_scope);
                    if let Some(segment_uuids) = self
                        .scalar_index_selection
                        .as_ref()
                        .and_then(|selection| selection.segment_uuids.clone())
                    {
                        exec = exec.with_segment_uuids(segment_uuids);
                    }
                    Some(Arc::new(exec) as Arc<dyn ExecutionPlan>)
                } else {
                    None
                }
            }
        };

        let plan: Arc<dyn ExecutionPlan> = Arc::new(FilteredReadExec::try_new(
            self.dataset.clone(),
            read_options,
            index_input,
        )?);

        if overlay_stale_rows.is_empty() {
            return Ok(plan);
        }

        // Stale-Take path: take the stale rows' current (overlay-merged) values and re-apply the
        // full filter, then union with the indexed read. These rows were blocked from the index
        // result above, so this is the only path that can surface them.
        let filter = filter_plan.full_expr.as_ref().expect_ok()?;
        let filter_cols = Planner::column_names_in_expr(filter);
        let take_projection = user_projection.union_columns(filter_cols, OnMissing::Error)?;

        let stale_node = self
            .stale_rows_take(&overlay_stale_rows, take_projection)
            .await?;
        let planner = Planner::new(stale_node.schema());
        let optimized_filter = planner.optimize_expr(filter.clone())?;
        let filtered = Arc::new(match session {
            Some(session) => {
                LanceFilterExec::try_new_with_session(optimized_filter, stale_node, session)?
            }
            None => LanceFilterExec::try_new(optimized_filter, stale_node)?,
        });
        let stale_path: Arc<dyn ExecutionPlan> =
            Arc::new(project(filtered, plan.schema().as_ref())?);

        let unioned = UnionExec::try_new(vec![plan, stale_path])?;
        Ok(Arc::new(RepartitionExec::try_new(
            unioned,
            datafusion::physical_plan::Partitioning::RoundRobinBatch(1),
        )?))
    }

    // Helper function for filtered read
    //
    // Delegates to legacy or new filtered read based on dataset storage version
    #[allow(clippy::too_many_arguments)]
    pub(super) fn filtered_read<'a>(
        &'a self,
        filter_plan: &'a ExprFilterPlan,
        projection: Projection,
        make_deletions_null: bool,
        fragments: Option<Arc<Vec<Fragment>>>,
        scan_range: Option<Range<u64>>,
        is_prefilter: bool,
        session: Option<&'a dyn Session>,
    ) -> BoxFuture<'a, Result<PlannedFilteredScan>> {
        // The plain-scan mask path lives in new_filtered_read; legacy_filtered_read
        // has no equivalent, so a masked plain scan there would silently drop the
        // mask and return every row. Fail loudly instead. Vector and full-text
        // searches apply the mask via their own prefilter paths (ANN prefilter /
        // FTS build_prefilter) plus the RowAddrMaskFilterExec flat wrap, so they
        // are unaffected -- use_external_mask() is false for them.
        let is_legacy = self
            .dataset
            .manifest()
            .data_storage_format
            .lance_file_format()
            == lance_file::version::ConcreteFileVersion::V1;
        if is_legacy && (self.use_external_mask() || self.physical_row_addr_prefilter.is_some()) {
            return std::future::ready(Err(Error::not_supported(
                "row prefilters are not supported for plain scans on \
                 legacy-storage datasets",
            )))
            .boxed();
        }
        versions::filtered_read(
            self.dataset
                .manifest()
                .data_storage_format
                .lance_file_format(),
            self,
            filter_plan,
            projection,
            make_deletions_null,
            fragments,
            scan_range,
            is_prefilter,
            session,
        )
        .boxed()
    }

    fn row_ids_as_take_input(&self, row_ids: RowAddrTreeMap) -> Result<Arc<dyn ExecutionPlan>> {
        self.mask_as_take_input(RowAddrMask::from_allowed(row_ids))
    }

    // Wrap a row-address mask as a one-shot index input for FilteredReadExec, so a
    // plain scan reads only the rows the mask selects.
    //
    // Every take-shaped row source funnels through here: plain takes, the
    // _rowid/_rowaddr predicate shortcut, and the overlay stale-row replay under
    // both scan and ANN. Intersecting the caller's mask once at this boundary is
    // what keeps the invariant on all of them; applying it per branch is how
    // branches get missed. Idempotent, so the branch that passes the external
    // mask itself is unaffected.
    fn mask_as_take_input(&self, mask: RowAddrMask) -> Result<Arc<dyn ExecutionPlan>> {
        let mask = match self.external_row_mask.as_deref() {
            Some(external) => mask.intersect(external.clone()),
            None => mask,
        };
        let index_result = IndexExprResult::exact(mask);
        let fragments_covered = self.dataset.fragment_bitmap.as_ref().clone();
        let format = self.index_expr_result_format();
        let batch = index_result.serialize(&fragments_covered, format)?;
        let schema = batch.schema();
        let stream = futures::stream::once(async move { Ok(batch) });
        let stream = Box::pin(RecordBatchStreamAdapter::new(schema, stream));
        Ok(Arc::new(OneShotExec::new(stream)))
    }

    async fn row_addrs_as_take_input(&self, row_addrs: Vec<u64>) -> Result<Arc<dyn ExecutionPlan>> {
        let row_ids =
            live_row_addrs_to_row_ids(&self.dataset, row_addrs.into_iter().map(Some)).await?;
        self.row_ids_as_take_input(RowAddrTreeMap::from_iter(row_ids.into_iter().flatten()))
    }

    // The columns the source read must produce. If we have an aggregate, we only need the
    // columns referenced by the aggregate, not all the columns from the projection plan.
    fn source_projection(&self) -> Result<Projection> {
        let Some(agg) = &self.aggregate else {
            return Ok(self.projection_plan.physical_projection.clone());
        };
        let required_columns = agg.required_columns();
        if required_columns.is_empty() {
            // COUNT(*) or similar - no columns needed
            Ok(self.dataset.empty_projection())
        } else {
            // Aggregate needs specific columns
            self.dataset
                .empty_projection()
                .union_columns(&required_columns, OnMissing::Error)
        }
    }

    pub(super) async fn take_source(
        &self,
        take_op: TakeOperation,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // We generally assume that late materialization does not make sense for take operations
        // so we can just use the source projection
        let mut projection = self.source_projection()?;
        // Later stages take filter, aggregate and ordering columns by row address, and
        // an empty projection still needs an output column. The address is synthesized.
        projection.with_row_addr = true;

        let input = match take_op {
            TakeOperation::RowIds(ids) => {
                self.row_ids_as_take_input(RowAddrTreeMap::from_iter(ids))
            }
            TakeOperation::RowAddrs(addrs) => self.row_addrs_as_take_input(addrs).await,
            TakeOperation::RowOffsets(offsets) => {
                let mut addrs =
                    row_offsets_to_row_addresses(&self.dataset.get_fragments(), &offsets).await?;
                addrs.retain(|addr| *addr != RowAddress::TOMBSTONE_ROW);
                self.row_addrs_as_take_input(addrs).await
            }
        }?;

        let mut filtered_read_options = self.read_options(projection)?;
        if let Some(fragment) = self.fragments.as_ref() {
            filtered_read_options =
                filtered_read_options.with_fragments(Arc::new(fragment.clone()));
        }
        if let Some(rows) = &self.physical_row_addr_prefilter {
            filtered_read_options =
                filtered_read_options.with_physical_row_addr_prefilter(rows.clone());
        }

        Ok(Arc::new(FilteredReadExec::try_new(
            self.dataset.clone(),
            filtered_read_options,
            Some(input),
        )?))
    }

    pub(super) async fn filtered_read_source(
        &self,
        filter_plan: &mut ExprFilterPlan,
        session: Option<&dyn Session>,
    ) -> Result<PlannedFilteredScan> {
        log::trace!("source is a filtered read");

        let effective_projection = self.source_projection()?;

        // Plan against the effective filter: with an external mask the SQL filter
        // becomes a refine, so its columns must be retained even when the original
        // plan resolved to an exact scalar-index query (has_refine() == false).
        let effective_filter = self.effective_filter_plan(filter_plan);
        let mut projection = if effective_filter.has_refine() {
            // If the filter plan has two steps (a scalar indexed portion and a refine portion) then
            // it makes sense to grab cheap columns during the first step to avoid taking them for
            // the second step.
            self.calc_eager_projection(&effective_filter, &effective_projection)?
                .with_row_id()
        } else {
            // If the filter plan only has one step then we just do a filtered read of all the
            // columns that the user asked for.
            effective_projection
        };

        if projection.is_empty() {
            // If the user is not requesting any columns then we will scan the row address which
            // is cheap
            projection.with_row_addr = true;
        }

        // An external mask is applied as the row source inside new_filtered_read, so
        // limit/offset must not be pushed down as a pre-mask range (that would limit
        // rows before masking). Leaving scan_range None keeps limit_pushed_down false
        // so the limit is applied by a node above the masked source instead.
        let scan_range_plan = if filter_plan.is_empty()
            && !self.use_external_mask()
            && self.physical_row_addr_prefilter.is_none()
        {
            log::trace!("pushing scan_range into filtered_read");
            self.plan_scan_range(filter_plan).await?
        } else {
            None
        };
        let (fragments, scan_range) = match scan_range_plan {
            Some(plan) => (plan.fragments, Some(plan.range)),
            None => (self.fragments.clone().map(Arc::new), None),
        };

        self.filtered_read(
            filter_plan,
            projection,
            self.include_deleted_rows,
            fragments,
            scan_range,
            /*is_prefilter= */ false,
            session,
        )
        .await
    }

    /// Per-row stale offsets for each fragment whose indexed values may be stale because an
    /// overlay committed *after* an index was built touches a field that index covers.
    ///
    /// The check is field-aware (an overlay touching only unindexed fields excludes nothing) and
    /// version-gated (an overlay with `committed_version <= index.dataset_version` is already
    /// incorporated by the index), via
    /// [`lance_table::format::overlay::staleness::overlay_exclusion_offsets`].
    pub(super) async fn overlay_stale_index_rows(
        &self,
        index_expr: &ScalarIndexExpr,
        fragments: &[Fragment],
    ) -> Result<HashMap<u32, RoaringBitmap>> {
        // Overlays are rare; skip all index loading when none of the candidate fragments has one.
        let overlaid_frags = overlaid_fragments(fragments);
        if overlaid_frags.is_empty() {
            return Ok(HashMap::new());
        }

        // Walk the (boolean) index expression tree to collect leaf searches.
        let mut searches = Vec::new();
        let mut stack = vec![index_expr];
        while let Some(expr) = stack.pop() {
            match expr {
                ScalarIndexExpr::Not(inner) => stack.push(inner),
                ScalarIndexExpr::And(lhs, rhs) | ScalarIndexExpr::Or(lhs, rhs) => {
                    stack.push(lhs);
                    stack.push(rhs);
                }
                ScalarIndexExpr::Query(search) => searches.push(search),
            }
        }

        // `load_named_scalar_segments` returns cached index metadata — no disk I/O on the hot
        // path. Even without the cache, this code is only reached when at least one fragment has
        // overlays (rare), so the per-leaf cost is acceptable.
        let mut stale: HashMap<u32, RoaringBitmap> = HashMap::new();
        for search in searches {
            let segments = load_named_scalar_segments(
                self.dataset.as_ref(),
                &search.column,
                &search.index_name,
            )
            .await?;
            for segment in &segments {
                collect_overlay_stale_rows_for_segment(
                    segment,
                    &overlaid_frags,
                    &mut stale,
                    self.dataset.schema(),
                )?;
            }
        }
        Ok(stale)
    }

    /// Compute per-row stale data for a vector index's segments.
    ///
    /// Returns a map from fragment_id to the set of row offsets within that fragment that are stale
    /// (their vector values have been updated by a newer overlay since the index was built). An
    /// empty map means no stale rows — the fast path where no masking is needed.
    pub(super) fn overlay_stale_vector_rows(
        &self,
        segments: &[IndexMetadata],
    ) -> Result<HashMap<u32, RoaringBitmap>> {
        // Scope to the query's target fragments (all dataset fragments if unscoped).
        let dataset_frags = self.dataset.fragments();
        let fragments: &[Fragment] = match self.fragments.as_ref() {
            Some(f) => f.as_slice(),
            None => dataset_frags.as_slice(),
        };
        let overlaid_frags = overlaid_fragments(fragments);
        if overlaid_frags.is_empty() {
            return Ok(HashMap::new());
        }
        let mut stale: HashMap<u32, RoaringBitmap> = HashMap::new();
        for segment in segments {
            collect_overlay_stale_rows_for_segment(
                segment,
                &overlaid_frags,
                &mut stale,
                self.dataset.schema(),
            )?;
        }
        Ok(stale)
    }

    /// Collect the stale rows into a [`RowAddrTreeMap`] in the domain the index results use.
    ///
    /// Index results are in the row-id domain (see `ScalarQuery::evaluate_nullable`), and a
    /// physical row address equals its row id only when the dataset does not use stable row ids.
    /// Under stable row ids the addresses are translated to their row ids so the result lines up
    /// with the index output it is combined with (block mask) or taken against.
    pub(super) async fn stale_rows_in_id_domain(
        &self,
        stale_rows: &HashMap<u32, RoaringBitmap>,
    ) -> Result<RowAddrTreeMap> {
        let mut tree_map = RowAddrTreeMap::new();
        for (&frag_id, offsets) in stale_rows {
            tree_map.insert_bitmap(frag_id, offsets.clone());
        }
        if self.dataset.manifest.uses_stable_row_ids() {
            tree_map = translate_addr_treemap_to_row_ids(&self.dataset, &tree_map).await?;
        }
        Ok(tree_map)
    }

    /// Build a block-list mask over stale rows, or `None` when there are none.
    ///
    /// The mask removes these rows from an index result so the index never emits them; they
    /// are re-evaluated against their current (overlay-merged) values on a targeted take path
    /// (see [`Self::stale_rows_take`]).
    pub(super) async fn stale_rows_block_mask(
        &self,
        stale_rows: &HashMap<u32, RoaringBitmap>,
    ) -> Result<Option<RowAddrMask>> {
        if stale_rows.is_empty() {
            return Ok(None);
        }
        let tree_map = self.stale_rows_in_id_domain(stale_rows).await?;
        Ok(Some(RowAddrMask::from_block(tree_map)))
    }

    /// Take the stale rows by physical address, projecting `projection`, to re-evaluate only
    /// those rows (rather than their whole fragments) against their current overlay-merged values.
    ///
    /// The rows are identified by an address allow list routed through `FilteredReadExec`, not a
    /// `_rowid` column (`_rowid` and row address diverge under stable row ids — see
    /// [`Self::stale_rows_in_id_domain`]).
    pub(super) async fn stale_rows_take(
        &self,
        stale_rows: &HashMap<u32, RoaringBitmap>,
        projection: Projection,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let take_id_map = self.stale_rows_in_id_domain(stale_rows).await?;
        let index_input = self.row_ids_as_take_input(take_id_map)?;
        let mut read_options = FilteredReadOptions::new(projection);
        if let Some(fragments) = self.fragments.as_ref() {
            read_options = read_options.with_fragments(Arc::new(fragments.clone()));
        }
        if let Some(rows) = &self.physical_row_addr_prefilter {
            read_options = read_options.with_physical_row_addr_prefilter(rows.clone());
        }
        Ok(Arc::new(FilteredReadExec::try_new(
            self.dataset.clone(),
            read_options,
            Some(index_input),
        )?))
    }

    // First perform a lookup in a scalar index for ids and then perform a take on the
    // target fragments with those ids
    async fn scalar_indexed_scan(
        &self,
        projection: Projection,
        filter_plan: &ExprFilterPlan,
        fragments: Arc<Vec<Fragment>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        log::trace!("scalar indexed scan");
        // One or more scalar indices cover this data and there is a filter which is
        // compatible with the indices.  Use that filter to perform a take instead of
        // a full scan.

        // If this unwrap fails we have a bug because we shouldn't be using this function unless we've already
        // checked that there is an index query
        let index_expr = filter_plan.index_query.as_ref().unwrap();

        let needs_recheck = index_expr.needs_recheck();

        // Figure out which fragments are covered by ALL indices, and which rows within
        // covered fragments are stale due to data overlay files.
        let (relevant_frags, missing_frags, stale_rows) = self
            .partition_frags_by_coverage(index_expr, fragments)
            .await?;

        // Build the MaterializeIndexExec, blocking stale row addresses so the index never
        // emits them. Stale rows are re-scored separately via a targeted take below.
        let mut mat_exec = MaterializeIndexExec::new(
            self.dataset.clone(),
            index_expr.clone(),
            Arc::new(relevant_frags),
        );
        if let Some(segment_uuids) = self
            .scalar_index_selection
            .as_ref()
            .and_then(|selection| selection.segment_uuids.clone())
        {
            mat_exec = mat_exec.with_segment_uuids(segment_uuids);
        }
        let mat_exec = match self.stale_rows_block_mask(&stale_rows).await? {
            Some(block) => mat_exec.with_overlay_block(block),
            None => mat_exec,
        };
        let mut plan: Arc<dyn ExecutionPlan> = Arc::new(mat_exec);

        let refine_expr = filter_plan.refine_expr.as_ref();

        // If all we want is the row ids then we can skip the take.  However, if there is a refine
        // or a recheck then we still need to do a take because we need filter columns.
        let needs_take =
            needs_recheck || projection.has_data_fields() || filter_plan.refine_expr.is_some();
        if needs_take {
            let mut take_projection = projection.clone();
            if needs_recheck {
                // If we need to recheck then we need to also take the columns used for the filter
                let filter_expr = index_expr.to_expr();
                let filter_cols = Planner::column_names_in_expr(&filter_expr);
                take_projection = take_projection.union_columns(filter_cols, OnMissing::Error)?;
            }
            if let Some(refine_expr) = refine_expr {
                let refine_cols = Planner::column_names_in_expr(refine_expr);
                take_projection = take_projection.union_columns(refine_cols, OnMissing::Error)?;
            }
            log::trace!("need to take additional columns for scalar_indexed_scan");
            plan = self.take(plan, take_projection)?;
        }

        let post_take_filter = match (needs_recheck, refine_expr) {
            (false, None) => None,
            (true, None) => {
                // If we need to recheck then we need to apply the filter to the results
                Some(index_expr.to_expr())
            }
            (true, Some(_)) => Some(filter_plan.full_expr.as_ref().unwrap().clone()),
            (false, Some(refine_expr)) => Some(refine_expr.clone()),
        };

        if let Some(post_take_filter) = post_take_filter {
            let planner = Planner::new(plan.schema());
            let optimized_filter = planner.optimize_expr(post_take_filter)?;

            log::trace!("applying post-take filter to indexed scan");
            plan = Arc::new(LanceFilterExec::try_new(optimized_filter, plan)?);
        }

        if self.projection_plan.physical_projection.with_row_addr {
            plan = Arc::new(AddRowAddrExec::try_new(plan, self.dataset.clone(), 0)?);
        }

        // Both the missing-fragments path (full scan) and the stale-rows path (targeted take)
        // need the user's projection extended with any filter columns. Compute it once.
        let fallback_projection: Option<Projection> =
            if !missing_frags.is_empty() || !stale_rows.is_empty() {
                let filter = filter_plan.full_expr.as_ref().expect_ok()?;
                let filter_cols = Planner::column_names_in_expr(filter);
                Some(
                    projection
                        .clone()
                        .union_columns(filter_cols, OnMissing::Error)?,
                )
            } else {
                None
            };

        let new_data_path: Option<Arc<dyn ExecutionPlan>> = if !missing_frags.is_empty() {
            log::trace!(
                "scalar_indexed_scan will need full scan of {} missing fragments",
                missing_frags.len()
            );

            // If there is new data then we need this:
            //
            // MaterializeIndexExec(old_frags) -> Take -> Union
            // Scan(new_frags) -> Filter -> Project    -|
            //
            // The project is to drop any columns we had to include
            // in the full scan merely for the sake of fulfilling the
            // filter.
            //
            // If there were no extra columns then we still need the project
            // because Materialize -> Take puts the row id at the left and
            // Scan puts the row id at the right
            let scan_projection = fallback_projection.clone().expect_ok()?;
            let filter = filter_plan.full_expr.as_ref().expect_ok()?;
            let scan_schema = Arc::new(scan_projection.to_bare_schema());
            let scan_arrow_schema = Arc::new(scan_schema.as_ref().into());
            let planner = Planner::new(scan_arrow_schema);
            let optimized_filter = planner.optimize_expr(filter.clone())?;

            let new_data_scan = self.scan_fragments(
                true,
                self.projection_plan.physical_projection.with_row_addr,
                self.projection_plan
                    .physical_projection
                    .with_row_last_updated_at_version,
                self.projection_plan
                    .physical_projection
                    .with_row_created_at_version,
                false,
                scan_schema,
                missing_frags.into(),
                // No pushdown of limit/offset when doing scalar indexed scan
                None,
                false,
            );
            let filtered = Arc::new(LanceFilterExec::try_new(optimized_filter, new_data_scan)?);
            Some(Arc::new(project(filtered, plan.schema().as_ref())?))
        } else {
            log::trace!("scalar_indexed_scan will not need full scan of any missing fragments");
            None
        };

        // Stale-Take path: re-evaluate only the stale row addresses against the full filter
        // (row-level optimization). These rows were blocked from the index result above;
        // here we take their current (overlay-merged) values and re-apply the predicate.
        // The schema matches `plan` via `project(…, plan.schema())`.
        let stale_take_path: Option<Arc<dyn ExecutionPlan>> = if stale_rows.is_empty() {
            None
        } else {
            let filter = filter_plan.full_expr.as_ref().expect_ok()?;
            let take_projection = fallback_projection.expect_ok()?;

            let stale_node = self.stale_rows_take(&stale_rows, take_projection).await?;

            let planner = Planner::new(stale_node.schema());
            let optimized_filter = planner.optimize_expr(filter.clone())?;
            let filtered = Arc::new(LanceFilterExec::try_new(optimized_filter, stale_node)?);
            Some(Arc::new(project(filtered, plan.schema().as_ref())?))
        };

        let extra_paths: Vec<Arc<dyn ExecutionPlan>> = [new_data_path, stale_take_path]
            .into_iter()
            .flatten()
            .collect();
        if extra_paths.is_empty() {
            Ok(plan)
        } else {
            let all_paths = std::iter::once(plan).chain(extra_paths).collect();
            let unioned = UnionExec::try_new(all_paths)?;
            Ok(Arc::new(RepartitionExec::try_new(
                unioned,
                datafusion::physical_plan::Partitioning::RoundRobinBatch(1),
            )?))
        }
    }

    fn get_io_buffer_size(&self) -> u64 {
        self.io_buffer_size.unwrap_or(*DEFAULT_IO_BUFFER_SIZE)
    }

    /// Create an Execution plan with a scan node
    ///
    /// Setting `with_make_deletions_null` will use the validity of the _rowid
    /// column as a selection vector. Read more in [crate::io::FileReader].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn scan(
        &self,
        with_row_id: bool,
        with_row_address: bool,
        with_row_last_updated_at_version: bool,
        with_row_created_at_version: bool,
        with_make_deletions_null: bool,
        range: Option<Range<u64>>,
        projection: Arc<Schema>,
    ) -> Arc<dyn ExecutionPlan> {
        let fragments = if let Some(fragment) = self.fragments.as_ref() {
            Arc::new(fragment.clone())
        } else {
            self.dataset.fragments().clone()
        };
        let ordered = if self.ordering.is_some() || self.nearest.is_some() {
            // If we are sorting the results there is no need to scan in order
            false
        } else {
            self.ordered
        };
        self.scan_fragments(
            with_row_id,
            with_row_address,
            with_row_last_updated_at_version,
            with_row_created_at_version,
            with_make_deletions_null,
            projection,
            fragments,
            range,
            ordered,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn scan_fragments(
        &self,
        with_row_id: bool,
        with_row_address: bool,
        with_row_last_updated_at_version: bool,
        with_row_created_at_version: bool,
        with_make_deletions_null: bool,
        projection: Arc<Schema>,
        fragments: Arc<Vec<Fragment>>,
        range: Option<Range<u64>>,
        ordered: bool,
    ) -> Arc<dyn ExecutionPlan> {
        log::trace!("scan_fragments covered {} fragments", fragments.len());
        let config = LanceScanConfig {
            batch_size: self.get_batch_size(),
            batch_readahead: self.batch_readahead,
            fragment_readahead: self.fragment_readahead,
            io_buffer_size: self.get_io_buffer_size(),
            materialization_readahead_bytes: self.materialization_readahead_bytes,
            with_row_id,
            with_row_address,
            with_row_last_updated_at_version,
            with_row_created_at_version,
            with_make_deletions_null,
            ordered_output: ordered,
            file_reader_options: self.resolved_file_reader_options(),
            parallelism_cap: None,
        };
        Arc::new(LanceScanExec::new(
            self.dataset.clone(),
            fragments,
            range,
            projection,
            config,
        ))
    }

    fn pushdown_scan(
        &self,
        make_deletions_null: bool,
        filter_plan: &ExprFilterPlan,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        log::trace!("pushdown_scan");

        let config = ScanConfig {
            batch_readahead: self.batch_readahead,
            fragment_readahead: self
                .fragment_readahead
                .unwrap_or(LEGACY_DEFAULT_FRAGMENT_READAHEAD),
            with_row_id: self.projection_plan.physical_projection.with_row_id,
            with_row_address: self.projection_plan.physical_projection.with_row_addr,
            make_deletions_null,
            ordered_output: self.ordered,
            file_reader_options: self.resolved_file_reader_options(),
        };

        let fragments = if let Some(fragment) = self.fragments.as_ref() {
            Arc::new(fragment.clone())
        } else {
            self.dataset.fragments().clone()
        };

        Ok(Arc::new(LancePushdownScanExec::try_new(
            self.dataset.clone(),
            fragments,
            Arc::new(self.projection_plan.physical_projection.to_bare_schema()),
            filter_plan.refine_expr.clone().unwrap(),
            config,
        )?))
    }

    /// Take row indices produced by input plan from the dataset (with projection)
    ///
    /// Planned as a [`FilteredReadExec`] row-stream read; legacy (v1) storage
    /// keeps using [`TakeExec`].
    #[allow(deprecated)]
    pub(super) fn take(
        &self,
        input: Arc<dyn ExecutionPlan>,
        output_projection: Projection,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let fields_to_take = output_projection
            .clone()
            .subtract_arrow_schema(input.schema().as_ref(), OnMissing::Ignore)?;
        if !fields_to_take.has_data_fields()
            && !fields_to_take.with_row_id
            && !fields_to_take.with_row_addr
        {
            // No new columns needed
            return Ok(input);
        }

        versions::take(
            self.dataset
                .manifest()
                .data_storage_format
                .lance_file_format(),
            self,
            input,
            output_projection,
        )
    }

    pub(in crate::dataset) fn take_current(
        &self,
        input: Arc<dyn ExecutionPlan>,
        output_projection: Projection,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let input_schema = input.schema();
        let has_row_id = input_schema.column_with_name(ROW_ID).is_some();
        let has_row_addr = input_schema.column_with_name(ROW_ADDR).is_some();
        if has_row_id || has_row_addr {
            // Pass the full (un-subtracted) target so a rebuild against a
            // different child re-derives what to fetch, and preserve carried
            // identity columns (downstream nodes may key off them; the final
            // ProjectionExec trims for free)
            let mut projection = output_projection;
            projection.with_row_id |= has_row_id;
            projection.with_row_addr |= has_row_addr;
            let mut read_options = self.read_options(projection)?;
            if self.include_deleted_rows {
                // Forwarded so the row-stream read rejects it: deleted rows
                // carry a null row id, which the take would silently drop
                read_options = read_options.with_deleted_rows()?;
            }
            if let Some(fragments) = &self.fragments {
                read_options = read_options.with_fragments(Arc::new(fragments.clone()));
            }
            return Ok(Arc::new(FilteredReadExec::try_new(
                self.dataset.clone(),
                read_options,
                Some(input),
            )?));
        }

        self.take_legacy(input, output_projection)
    }

    #[allow(deprecated)]
    pub(in crate::dataset) fn take_legacy(
        &self,
        input: Arc<dyn ExecutionPlan>,
        output_projection: Projection,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let coalesced = Arc::new(CoalesceBatchesExec::new(
            input.clone(),
            self.get_batch_size(),
        ));
        if let Some(take_plan) = TakeExec::try_new_with_batch_size(
            self.dataset.clone(),
            coalesced,
            output_projection,
            self.resolved_file_reader_options()
                .and_then(|o| o.batch_size_bytes),
        )? {
            Ok(Arc::new(take_plan))
        } else {
            // No new columns needed
            Ok(input)
        }
    }

    /// Global offset-limit of the result of the input plan
    pub(super) fn limit_node(&self, plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        Arc::new(GlobalLimitExec::new(
            plan,
            *self.offset.as_ref().unwrap_or(&0) as usize,
            self.limit.map(|l| l as usize),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use datafusion::logical_expr::{col, lit};

    // Unit-level companion to the NOT IN case in
    // test_filter_to_take_with_stable_row_ids: exercises the lowering directly,
    // independent of the expression simplifier's InList expansion threshold
    // (which the end-to-end test depends on to keep the list unexpanded).
    #[test]
    fn take_operation_rejects_negated_in_list() {
        // All three virtual columns share the one InList arm, so the guard has
        // to hold for each of them.
        for column in [ROW_ID, ROW_ADDR, ROW_OFFSET] {
            let positive = col(column).in_list(vec![lit(0u64), lit(1u64)], false);
            let lowered = TakeOperation::try_from_expr(&positive);
            let ids = match lowered {
                Some((TakeOperation::RowIds(ids), None))
                | Some((TakeOperation::RowAddrs(ids), None))
                | Some((TakeOperation::RowOffsets(ids), None)) => ids,
                other => panic!("{column} IN (0, 1) must lower into a take, got {other:?}"),
            };
            assert_eq!(ids, vec![0, 1], "wrong ids lowered for {column}");

            let negated = col(column).in_list(vec![lit(0u64), lit(1u64)], true);
            assert!(
                TakeOperation::try_from_expr(&negated).is_none(),
                "a negated InList on {column} is the complement of the listed ids \
                 and must not lower into a take of them"
            );
        }
    }
}
