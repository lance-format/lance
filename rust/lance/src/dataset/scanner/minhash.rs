// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::SortOptions;
use datafusion::physical_expr::PhysicalSortExpr;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::{ExecutionPlan, expressions};
use datafusion_physical_expr::Partitioning;
use datafusion_physical_plan::empty::EmptyExec;
use lance_core::ROW_ID;
use lance_core::datatypes::OnMissing;
use lance_index::IndexCriteria;
use lance_index::scalar::minhash_lsh::{MinHashLshIndexParams, MinHashQuery};
use lance_index::vector::DIST_COL;
use lance_table::format::Fragment;
use roaring::RoaringBitmap;

use crate::dataset::overlay::{collect_overlay_stale_rows_for_segment, overlaid_fragments};
use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};
use crate::io::exec::knn::knn_empty_result_schema;
use crate::io::exec::{
    FilterPlan as ExprFilterPlan, FlatMinHashExec, LanceFilterExec, MinHashSearchExec, Planner,
    RowAddrMaskFilterExec,
};
use crate::{Error, Result};

use super::{FilterPlan, PlannedFilteredScan, Scanner};

impl Scanner {
    /// The source plan of a scan whose `nearest` is a MinHash search.
    ///
    /// With prefiltering the search node applies the whole filter before
    /// ranking, so `filter_plan` is left with nothing to refine; otherwise the
    /// search ranks unfiltered rows and the filter is kept as a refine step
    /// over the ranked output. Deleted rows cannot be included: a deleted row
    /// has no current text to rank.
    pub(super) async fn minhash_search_source(
        &self,
        filter_plan: &mut FilterPlan,
        query: &MinHashQuery,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        log::trace!("source is a minhash search");
        if self.include_deleted_rows {
            return Err(Error::invalid_input_source(
                "Cannot include deleted rows in a MinHash search".into(),
            ));
        }

        if self.prefilter {
            let source = self.minhash(&filter_plan.expr_filter_plan, query).await?;
            // The search node applies the filter before ranking; nothing is left to refine
            filter_plan.disable_refine();
            Ok(source)
        } else {
            // Postfiltering runs the filter in memory on the ranked rows
            filter_plan.make_refine_only();
            self.minhash(&ExprFilterPlan::default(), query).await
        }
    }

    /// Create an execution plan for a MinHash similarity search over every
    /// segment of the column's MinHash LSH index.
    async fn minhash(
        &self,
        filter_plan: &ExprFilterPlan,
        query: &MinHashQuery,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let Some(limit) = self.limit else {
            return Err(Error::invalid_input(
                "MinHash search requires a limit (the number of most similar rows to return)"
                    .to_string(),
            ));
        };
        // Offset rows are skipped by the limit node, so the search must produce them too
        let search_limit = limit
            .checked_add(self.offset.unwrap_or(0))
            .and_then(|search_limit| usize::try_from(search_limit).ok())
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "MinHash search limit {limit} plus offset {:?} is out of range",
                    self.offset
                ))
            })?;

        let index = self
            .dataset
            .load_scalar_index(
                IndexCriteria::default()
                    .for_column(&query.column)
                    .supports_minhash(),
            )
            .await?
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "No MinHash LSH index found for column {}; create a \"minhashlsh\" scalar index on the column before running a MinHash search",
                    query.column
                ))
            })?;
        let segments = self.dataset.load_indices_by_name(&index.name).await?;
        let details = segments
            .first()
            .and_then(|segment| segment.index_details.as_deref())
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "MinHash LSH index {} on column {} has no segment with index details",
                    index.name, query.column
                ))
            })?;
        let params = MinHashLshIndexParams::from_details_any(details)?;

        let target_fragments: &[Fragment] = self
            .fragments
            .as_deref()
            .unwrap_or_else(|| self.dataset.fragments());
        if target_fragments.is_empty() {
            return Ok(Arc::new(EmptyExec::new(knn_empty_result_schema(false))));
        }
        let unindexed_fragments =
            self.retain_target_fragments(self.dataset.unindexed_fragments(&index.name).await?);
        let all_unindexed = unindexed_fragments.len() == target_fragments.len();

        // Rows whose value was replaced by a data overlay newer than the index
        // have stale signatures: the index branch blocks them and the flat
        // branch re-scores them from their current value.
        let mut stale_rows: HashMap<u32, RoaringBitmap> = HashMap::new();
        if target_fragments
            .iter()
            .any(|fragment| !fragment.overlays.is_empty())
        {
            let overlaid_frags = overlaid_fragments(target_fragments);
            for segment in &segments {
                collect_overlay_stale_rows_for_segment(
                    segment,
                    &overlaid_frags,
                    &mut stale_rows,
                    self.dataset.schema(),
                )?;
            }
        }

        let indexed_plan = if all_unindexed {
            None
        } else {
            let overlay_block = self.stale_rows_block_mask(&stale_rows).await?;
            let prefilter_source = self
                .prefilter_source(filter_plan, self.get_indexed_frags(&segments))
                .await?;
            Some(Arc::new(MinHashSearchExec::new(
                self.dataset.clone(),
                query.clone(),
                search_limit,
                segments,
                prefilter_source,
                overlay_block,
                self.external_row_mask.clone(),
            )) as Arc<dyn ExecutionPlan>)
        };
        // fast_search trades the unindexed rows for the index-only cost
        let flat_plan =
            if !self.fast_search && (!unindexed_fragments.is_empty() || !stale_rows.is_empty()) {
                Some(
                    self.plan_flat_minhash(
                        unindexed_fragments,
                        stale_rows,
                        query,
                        &params,
                        search_limit,
                        filter_plan,
                    )
                    .await?,
                )
            } else {
                None
            };

        let plan = match (indexed_plan, flat_plan) {
            (Some(indexed_plan), Some(flat_plan)) => {
                UnionExec::try_new(vec![indexed_plan, flat_plan])?
            }
            // Each branch already emits its top-k in distance order
            (Some(indexed_plan), None) => return Ok(indexed_plan),
            (None, Some(flat_plan)) => return Ok(flat_plan),
            (None, None) => {
                return Ok(Arc::new(EmptyExec::new(knn_empty_result_schema(false))));
            }
        };
        let plan = Arc::new(RepartitionExec::try_new(
            plan,
            Partitioning::RoundRobinBatch(1),
        )?);
        let schema = plan.schema();
        let sort_exprs = [
            PhysicalSortExpr {
                expr: expressions::col(DIST_COL, schema.as_ref())?,
                options: SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            },
            PhysicalSortExpr {
                expr: expressions::col(ROW_ID, schema.as_ref())?,
                options: SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            },
        ];
        Ok(Arc::new(
            SortExec::new(sort_exprs.into(), plan).with_fetch(Some(search_limit)),
        ))
    }

    /// One plan over the rows an index does not cover: whole unindexed
    /// fragments (filtered while reading) and stale rows of a data overlay
    /// (taken by address, then filtered), projected to `_rowid` plus
    /// `columns` and the filter's columns.
    pub(super) async fn plan_uncovered_rows_scan(
        &self,
        fragments: Vec<Fragment>,
        stale_rows: HashMap<u32, RoaringBitmap>,
        mut columns: Vec<String>,
        filter_plan: &ExprFilterPlan,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let filter_expr = if stale_rows.is_empty() {
            filter_plan.refine_expr.as_ref()
        } else {
            filter_plan.full_expr.as_ref()
        };
        if let Some(filter_expr) = filter_expr {
            columns.extend(Planner::column_names_in_expr(filter_expr));
        }
        let scan_projection = self
            .dataset
            .empty_projection()
            .with_row_id()
            .union_columns(&columns, OnMissing::Error)?;

        let mut inputs = Vec::with_capacity(2);
        if !fragments.is_empty() {
            // A prefilter read applies the whole filter plan, refine included
            let PlannedFilteredScan { plan, .. } = self
                .filtered_read(
                    filter_plan,
                    scan_projection.clone(),
                    /*make_deletions_null=*/ false,
                    Some(Arc::new(fragments)),
                    None,
                    /*is_prefilter=*/ true,
                    None,
                )
                .await?;
            inputs.push(plan);
        }
        if !stale_rows.is_empty() {
            // A take by address applies no filter, so the whole plan runs here
            let mut plan = self.stale_rows_take(&stale_rows, scan_projection).await?;
            if let Some(filter) = filter_plan.full_expr.as_ref() {
                let planner = Planner::new(plan.schema());
                let filter = planner.optimize_expr(filter.clone())?;
                plan = Arc::new(LanceFilterExec::try_new(filter, plan)?);
            }
            inputs.push(plan);
        }
        match inputs.len() {
            0 => Err(Error::internal(
                "an uncovered-rows scan requires unindexed fragments or stale rows",
            )),
            1 => inputs
                .pop()
                .ok_or_else(|| Error::internal("uncovered-rows scan input vanished")),
            _ => Ok(UnionExec::try_new(inputs)?),
        }
    }

    /// The flat branch of a MinHash search: rows the index does not cover
    /// (`fragments` indexed by no segment, and `stale_rows` whose text a data
    /// overlay replaced after the index was built) are read with the filter
    /// and the external row mask applied, signed with the index `params` on
    /// the fly, kept when they share a band with the query, and ranked to
    /// `limit` hits. The caller merges them with the index hits;
    /// `fast_search` skips this branch.
    async fn plan_flat_minhash(
        &self,
        fragments: Vec<Fragment>,
        stale_rows: HashMap<u32, RoaringBitmap>,
        query: &MinHashQuery,
        params: &MinHashLshIndexParams,
        limit: usize,
        filter_plan: &ExprFilterPlan,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let plan = self
            .plan_uncovered_rows_scan(
                fragments,
                stale_rows,
                vec![query.column.clone()],
                filter_plan,
            )
            .await?;
        let mut plan = self.ensure_column_alias(plan, &query.column)?;
        // Unindexed rows never reach the index-side prefilter, so the external
        // mask is applied here, before the flat search keeps its top `limit`
        // rows: a masked row must not take one of those slots.
        if let Some(mask) = self.external_row_mask.clone() {
            plan = Arc::new(RowAddrMaskFilterExec::new(plan, mask));
        }
        Ok(Arc::new(FlatMinHashExec::new(
            plan,
            query.clone(),
            params.clone(),
            limit,
        )))
    }
}
