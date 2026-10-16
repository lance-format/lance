// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::Array;
use arrow_schema::{DataType, SortOptions};
use async_recursion::async_recursion;
use datafusion::common::DFSchema;
use datafusion::logical_expr::{Expr, col, lit};
use datafusion::physical_expr::PhysicalSortExpr;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::{ExecutionPlan, expressions};
use datafusion::scalar::ScalarValue;
use datafusion_expr::execution_props::ExecutionProps;
use datafusion_physical_expr::expressions::{Column, Literal};
use datafusion_physical_expr::{Partitioning, PhysicalExpr, create_physical_expr};
use datafusion_physical_plan::empty::EmptyExec;
use datafusion_physical_plan::projection::ProjectionExec;
use lance_core::ROW_ID;
use lance_core::datatypes::OnMissing;
use lance_core::error::LanceOptionExt;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::expression::ScalarIndexExpr;
use lance_index::scalar::inverted::DOC_INDEX_COL;
use lance_index::vector::{DIST_COL, Query};
use lance_linalg::distance::MetricType;
use lance_select::RowAddrMask;
use lance_table::format::{Fragment, IndexMetadata};
use roaring::RoaringBitmap;

use crate::index::scalar_logical::scalar_index_fragment_bitmap;
use crate::index::vector::utils::{
    default_distance_type_for, get_vector_dim, get_vector_type, validate_distance_type_for,
};
use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};
use crate::io::exec::knn::{
    KnnBatchParams, MultivectorScoringExec, QUERY_INDEX_COL, knn_empty_result_schema,
    new_knn_batch_exec, new_knn_exec,
};
use crate::io::exec::scalar_index::ScalarIndexExec;
use crate::io::exec::{
    FilterPlan as ExprFilterPlan, KNNVectorDistanceExec, LanceFilterExec, Planner, PreFilterSource,
    RowAddrMaskFilterExec, project,
};
use crate::{Error, Result};

use super::{DEFAULT_XTR_OVERFETCH, FilterPlan, PlannedFilteredScan, Scanner};

impl Scanner {
    pub(super) async fn vector_search_source(
        &self,
        filter_plan: &mut FilterPlan,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if self.include_deleted_rows {
            return Err(Error::invalid_input_source(
                "Cannot include deleted rows in a nearest neighbor search".into(),
            ));
        }
        let Some(query) = self.nearest.as_ref() else {
            return Err(Error::invalid_input("No nearest query".to_string()));
        };

        if self.prefilter {
            log::trace!("source is a vector search (prefilter)");
            // If we are prefiltering then the ann / knn node will take care of the filter
            let source: Arc<dyn ExecutionPlan> = match &filter_plan.fts_filter() {
                Some(fts_query) => {
                    let mut fts_plan = self.fts(&filter_plan.expr_filter_plan, fts_query).await?;
                    if fts_plan.schema().field_with_name(DOC_INDEX_COL).is_ok() {
                        fts_plan = self.deduplicate_fts_filter_rows(fts_plan)?;
                    }
                    let projection = self
                        .dataset
                        .empty_projection()
                        .union_column(&query.column, OnMissing::Error)?;
                    let plan = self.take(fts_plan, projection)?;

                    self.flat_knn(plan, query)?
                }
                None => {
                    self.vector_search(&filter_plan.expr_filter_plan, query)
                        .await?
                }
            };

            filter_plan.disable_refine();
            Ok(source)
        } else {
            log::trace!("source is a vector search (postfilter)");
            // If we are postfiltering then we can't use scalar indices for the filter
            // and will need to run the postfilter in memory
            filter_plan.make_refine_only();
            self.vector_search(&ExprFilterPlan::default(), query).await
        }
    }

    // ANN/KNN search execution node with optional prefilter
    #[async_recursion]
    pub(super) async fn vector_search(
        &self,
        filter_plan: &ExprFilterPlan,
        q: &Query,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let mut q = q.clone();

        // Sanity check
        let (vector_type, element_type) = get_vector_type(self.dataset.schema(), &q.column)?;

        let column_id = self.dataset.schema().field_id(q.column.as_str())?;
        let use_index = q.use_index;
        let indices = if use_index {
            self.dataset.load_indices().await?
        } else {
            Arc::new(vec![])
        };
        let mut index_and_segments = if use_index {
            if let Some(requested_segments) = self.index_segments.as_ref() {
                let requested_segment_set =
                    requested_segments.iter().copied().collect::<HashSet<_>>();
                let requested_index_segments = indices
                    .iter()
                    .filter(|idx| requested_segment_set.contains(&idx.uuid))
                    .cloned()
                    .collect::<Vec<_>>();

                if requested_index_segments.len() != requested_segment_set.len() {
                    let found_segment_set = requested_index_segments
                        .iter()
                        .map(|idx| idx.uuid)
                        .collect::<HashSet<_>>();
                    let missing_segments = requested_segment_set
                        .difference(&found_segment_set)
                        .map(ToString::to_string)
                        .collect::<Vec<_>>();
                    return Err(Error::invalid_input(format!(
                        "with_index_segments referenced unknown index segments: {missing_segments:?}",
                    )));
                }

                if requested_index_segments
                    .iter()
                    .any(|idx| idx.fields.first() != Some(&column_id))
                {
                    return Err(Error::invalid_input(format!(
                        "with_index_segments contained a segment that does not belong to vector column '{}'",
                        q.column
                    )));
                }

                let index_name = requested_index_segments[0].name.clone();
                if requested_index_segments
                    .iter()
                    .any(|idx| idx.name != index_name)
                {
                    return Err(Error::invalid_input(
                        "with_index_segments must reference segments from a single logical index"
                            .to_string(),
                    ));
                }

                let selected_index_segments =
                    self.retain_relevant_index_segments(requested_index_segments);
                if selected_index_segments.is_empty() {
                    None
                } else {
                    let idx = self
                        .dataset
                        .open_vector_index(
                            q.column.as_str(),
                            &selected_index_segments[0].uuid,
                            &NoOpMetricsCollector,
                        )
                        .await?;
                    let index_metric = idx.metric_type();
                    let use_this_index = match q.metric_type {
                        Some(user_metric) => {
                            if user_metric == index_metric {
                                true
                            } else {
                                return Err(Error::invalid_input(format!(
                                    "with_index_segments requested metric {:?} but the selected index segments use {:?}",
                                    user_metric, index_metric
                                )));
                            }
                        }
                        None => true,
                    };
                    if use_this_index {
                        Some((index_name, selected_index_segments, index_metric))
                    } else {
                        None
                    }
                }
            }
            // An index can only answer a query on the column it is keyed on, which is
            // always `fields[0]`. Not `contains`: `fields` also lists columns the index
            // merely carries values for, which it cannot search. Not a boundary derived
            // from `covering_fields` either -- that is computed from a field older
            // writers drop, so it would widen to the carried columns exactly when the
            // declaration is lost.
            else if let Some(index) = indices.iter().find(|i| {
                i.fields.first() == Some(&column_id) && crate::index::index_type_is_known(i)
            }) {
                // Try to get metric type from index metadata first (fast path for newer indices)
                let index_metric = if let Some(metric) =
                    crate::index::vector::details::metric_type_from_index_metadata(index)
                {
                    metric
                } else {
                    // Fall back to opening the index for legacy indices without details
                    let idx = self
                        .dataset
                        .open_vector_index(q.column.as_str(), &index.uuid, &NoOpMetricsCollector)
                        .await?;
                    idx.metric_type()
                };

                let use_this_index = match q.metric_type {
                    Some(user_metric) => {
                        if user_metric == index_metric {
                            true
                        } else {
                            log::warn!(
                                "Requested metric {:?} is incompatible with index metric {:?}, falling back to brute-force search",
                                user_metric,
                                index_metric
                            );
                            false
                        }
                    }
                    None => true,
                };

                if use_this_index {
                    let index_segments = self.retain_relevant_index_segments(
                        self.dataset.load_indices_by_name(&index.name).await?,
                    );
                    let index_frags = self.get_indexed_frags(&index_segments);
                    if !index_segments.is_empty() && !index_frags.is_empty() {
                        Some((index.name.clone(), index_segments, index_metric))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Explicit segment selection may remove a contributor to FRI-derived coverage.
        // When fragments are requested, they must still be searched completely.
        // Segment-only search intentionally permits partial results and is unchanged.
        if self.index_segments.is_some()
            && self.fragments.is_some()
            && indices
                .iter()
                .any(lance_table::system_index::frag_reuse::metadata::is_tagged)
            && let Some((name, segments, _)) = &index_and_segments
        {
            let selected: HashSet<_> = segments.iter().map(|segment| segment.uuid).collect();
            let covered = self.get_indexed_frags(segments);
            let missing_contributor = indices.iter().any(|index| {
                index.name == *name
                    && !selected.contains(&index.uuid)
                    && index
                        .fragment_bitmap
                        .as_ref()
                        .is_some_and(|bitmap| !bitmap.is_disjoint(&covered))
            });
            if missing_contributor {
                index_and_segments = None;
            }
        }

        if let Some((index_name, index_segments, index_metric)) = index_and_segments {
            if self.is_batch_nearest {
                validate_distance_type_for(index_metric, &element_type)?;
                return self
                    .batch_indexed_vector_search(
                        filter_plan,
                        &q,
                        &index_name,
                        &index_segments,
                        index_metric,
                    )
                    .await;
            }

            log::trace!("index found for vector search");
            // Use the index's metric type
            q.metric_type = Some(index_metric);
            validate_distance_type_for(index_metric, &element_type)?;

            if matches!(q.refine_factor, Some(0)) {
                return Err(Error::invalid_input(
                    "Refine factor cannot be zero".to_string(),
                ));
            }
            // Mask data overlay files: compute which row addresses within each segment have
            // been updated by a newer overlay so their ANN entries may be stale.
            // These stale rows are blocked from ANN results via the prefilter and re-scored
            // on the targeted flat path below — only the specific stale rows, not the whole
            // fragment, so sparse overlays incur near-zero overhead.
            let stale_rows = self.overlay_stale_vector_rows(&index_segments)?;
            // Build a prefilter block mask for stale rows (empty = no-op fast path).
            let overlay_block = self.stale_rows_block_mask(&stale_rows).await?;

            let ann_node = match vector_type {
                DataType::FixedSizeList(_, _) => {
                    self.ann(&q, &index_segments, filter_plan, overlay_block.clone())
                        .await?
                }
                DataType::List(_) => {
                    self.multivec_ann(&q, &index_segments, filter_plan, overlay_block.clone())
                        .await?
                }
                _ => unreachable!(),
            };

            let mut knn_node = if q.refine_factor.is_some() {
                let vector_projection = self
                    .dataset
                    .empty_projection()
                    .union_column(&q.column, OnMissing::Error)
                    .unwrap();
                let knn_node_with_vector = self.take(ann_node, vector_projection)?;
                self.flat_knn(knn_node_with_vector, &q)?
            } else {
                ann_node
            }; // vector, _distance, _rowid

            if !self.fast_search {
                knn_node = self
                    .knn_combined(
                        &q,
                        &index_name,
                        &index_segments,
                        &stale_rows,
                        knn_node,
                        filter_plan,
                    )
                    .await?;
            }

            Ok(knn_node)
        } else {
            if self.fast_search {
                return Ok(Arc::new(EmptyExec::new(knn_empty_result_schema(
                    self.is_batch_nearest,
                ))));
            }
            // Resolve metric type for flat search (use default if not specified)
            let metric = q
                .metric_type
                .unwrap_or_else(|| default_distance_type_for(&element_type));
            q.metric_type = Some(metric);
            validate_distance_type_for(metric, &element_type)?;
            // No index found. use flat search.
            let mut columns = vec![q.column.clone()];
            if let Some(refine_expr) = filter_plan.refine_expr.as_ref() {
                columns.extend(Planner::column_names_in_expr(refine_expr));
            }
            let mut vector_scan_projection = self
                .dataset
                .empty_projection()
                .with_row_id()
                .union_columns(&columns, OnMissing::Error)?;

            vector_scan_projection.with_row_addr =
                self.projection_plan.physical_projection.with_row_addr;

            let PlannedFilteredScan { mut plan, .. } = self
                .filtered_read(
                    filter_plan,
                    vector_scan_projection,
                    /*include_deleted_rows=*/ true,
                    self.fragments.clone().map(Arc::new),
                    None,
                    /*is_prefilter= */ true,
                    None,
                )
                .await?;

            if let Some(refine_expr) = &filter_plan.refine_expr {
                plan = Arc::new(LanceFilterExec::try_new(refine_expr.clone(), plan)?);
            }
            // The flat branch never reaches the index-side prefilter, so apply
            // the external row-address mask here against the scanned _rowid.
            if let Some(mask) = self.external_row_mask.clone() {
                plan = Arc::new(RowAddrMaskFilterExec::new(plan, mask));
            }
            Ok(self.flat_knn(plan, &q)?)
        }
    }

    /// Whether a batch (multi-query) vector search can use the shared-scan
    /// indexed fast path ([`new_knn_batch_exec`]) instead of running one indexed
    /// search per query vector.
    ///
    /// Requires all of:
    /// - no refine step (the batch path does not yet rerank);
    /// - fixed nprobes (`minimum_nprobes == maximum_nprobes`) — see below;
    /// - every segment an IVF index with a flat-style sub-index (i.e. not HNSW);
    /// - every target fragment covered by the *selected* `index_segments` (or
    ///   `fast_search`, which searches only the selected segments anyway).
    ///
    /// The fixed-nprobes requirement is a *correctness* gate, not just an
    /// optimization. The shared-scan path searches exactly `minimum_nprobes`
    /// partitions per query, but the single-query path is adaptive: it applies a
    /// k-dependent `early_pruning` floor and then expands probes up to
    /// `maximum_nprobes` (late search) when a query has fewer than `k` results.
    /// When `minimum_nprobes == maximum_nprobes` neither adjustment can fire
    /// (pruning is capped at the maximum, and the late-search range is empty), so
    /// the batch result is provably identical to repeated single-query search.
    /// With adaptive nprobes the two would diverge, so we fall back to the
    /// per-query loop, which reuses the real adaptive search and stays exact.
    ///
    /// Extending the shared-scan path to adaptive nprobes (a batched early/late
    /// search) is left as a follow-up.
    async fn batch_index_search_supported(
        &self,
        index_name: &str,
        index_segments: &[IndexMetadata],
        q: &Query,
    ) -> Result<bool> {
        // Any refine factor sends the query onto a reranking path that the
        // shared batch scan does not implement: the single-query path reranks
        // with the original vectors even when the factor is 1, and rejects a
        // factor of 0 outright (`Refine factor cannot be zero`). The batch path
        // does neither, so fall back to the per-query loop for every `Some(_)`.
        if q.refine_factor.is_some() {
            return Ok(false);
        }
        // Only fixed nprobes is provably equivalent to single-query search; see
        // the method docs. Adaptive nprobes falls back to the per-query loop.
        if q.maximum_nprobes != Some(q.minimum_nprobes) {
            return Ok(false);
        }
        // `nprobes(0)` is not rejected by the query builder, so `min == max == 0`
        // slips past the fixed-nprobes check above. The single-query path probes
        // nothing and returns an empty result, whereas the batch node would probe
        // one partition's worth of neighbors — a silent divergence. Fall back so
        // the per-query loop defines the semantics of `nprobes(0)`.
        if q.minimum_nprobes == 0 {
            return Ok(false);
        }
        // The per-query path threads a caller-supplied external row-address mask
        // (`with_row_id_prefilter`) into each query's prefilter via
        // `with_external_mask`; the shared batch path builds one prefilter across
        // the batch and does not carry that mask. Rather than silently returning
        // masked-out rows, fall back to the per-query loop whenever a mask is set.
        if self.external_row_mask.is_some() {
            return Ok(false);
        }
        // Decide from the index metadata (no I/O) rather than opening the index
        // to call `supports_batch_partition_search()`: this is a planning-time
        // gate and the single-query path likewise avoids opening the index here.
        // An IVF index with a flat-style sub-index (i.e. not HNSW) is exactly the
        // set for which `supports_batch_partition_search()` is true; the exec
        // re-checks that trait as a defensive invariant. Legacy segments without
        // details fall back.
        let all_ivf_flat_style = index_segments.iter().all(|index| {
            index
                .index_details
                .as_ref()
                .filter(|details| !details.value.is_empty())
                .map(|details| {
                    let index_type =
                        crate::index::vector::details::derive_vector_index_type(details);
                    index_type.starts_with("IVF_") && !index_type.contains("HNSW")
                })
                .unwrap_or(false)
        });
        if !all_ivf_flat_style {
            return Ok(false);
        }
        // The batch node searches only the index's own entries; unlike the
        // single-query path it does not reconcile data overlays (which block
        // overlay-stale rows from the ANN result and re-score them on a flat
        // take path — see `overlay_stale_vector_rows` in `vector_search`). If any
        // indexed row was updated by a newer overlay, the batch path would return
        // that row's stale index entry, so fall back to the per-query loop. This
        // must precede the `fast_search` shortcut below because the single-query
        // path applies the overlay block even in fast-search mode. Cheap in the
        // common case: returns an empty map when no target fragment has overlays.
        if !self.overlay_stale_vector_rows(index_segments)?.is_empty() {
            return Ok(false);
        }
        if self.fast_search {
            return Ok(true);
        }
        // The batch node only searches the selected `index_segments`, so any
        // target fragment those segments do not cover would silently drop rows
        // (the single-query path re-scores such fragments on a flat fallback in
        // `knn_combined`). Measure coverage against the selected segments -- not
        // the whole logical index -- so a subset selected via
        // `with_index_segments` cannot hide a fragment that an unselected
        // segment happens to cover; fall back whenever any remain.
        let uncovered_fragments = self
            .fragments_missing_from_index_segments(index_name, index_segments)
            .await?;
        Ok(uncovered_fragments.is_empty())
    }

    /// Target fragments the given `index_segments` do not cover.
    ///
    /// The ANN scan reads only the selected segments' partitions, so these are
    /// exactly the fragments the single-query path re-scores on a flat fallback
    /// in [`Self::knn_combined`]. Coverage is measured against the *selected*
    /// segments rather than every segment of the logical index (which
    /// `Dataset::unindexed_fragments` would do): a caller may select a subset
    /// via [`with_index_segments`](Self::with_index_segments) while another,
    /// unselected segment covers one of the requested fragments.
    async fn fragments_missing_from_index_segments(
        &self,
        index_name: &str,
        index_segments: &[IndexMetadata],
    ) -> Result<Vec<Fragment>> {
        if let Some(target_fragments) = &self.fragments {
            let indexed_fragments = self.get_indexed_frags(index_segments);
            Ok(target_fragments
                .iter()
                .filter(|fragment| !indexed_fragments.contains(fragment.id as u32))
                .cloned()
                .collect())
        } else if self.index_segments.is_some() {
            // An explicit segment selection with no fragment restriction searches
            // exactly those segments; there is nothing to fall back for.
            Ok(Vec::new())
        } else {
            self.dataset.unindexed_fragments(index_name).await
        }
    }

    async fn batch_indexed_vector_search(
        &self,
        filter_plan: &ExprFilterPlan,
        q: &Query,
        index_name: &str,
        index_segments: &[IndexMetadata],
        index_metric: MetricType,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Fast path: when every index segment is an IVF index with a flat-style
        // sub-index (IVF_FLAT/PQ/SQ/RQ), search all query vectors in a single
        // pass that reads each partition's storage once and shares the prefilter
        // across the batch. HNSW, refine, and mixed indexed/unindexed scans fall
        // back to the per-query loop below, which never regresses behavior.
        if self
            .batch_index_search_supported(index_name, index_segments, q)
            .await?
        {
            let mut batch_query = q.clone();
            batch_query.metric_type = Some(index_metric);
            let prefilter_source = self
                .ann_prefilter_source(filter_plan, index_segments)
                .await?;
            return new_knn_batch_exec(
                self.dataset.clone(),
                index_segments,
                &batch_query,
                self.nearest_query_count,
                prefilter_source,
            );
        }

        let query_dim = q.key.len() / self.nearest_query_count;
        let mut query_plans = Vec::with_capacity(self.nearest_query_count);

        for query_index in 0..self.nearest_query_count {
            let mut single_query = q.clone();
            single_query.key = q.key.slice(query_index * query_dim, query_dim);

            let mut single_scanner = self.clone();
            single_scanner.nearest_query_count = 1;
            single_scanner.is_batch_nearest = false;
            single_scanner.nearest = Some(single_query.clone());

            let single_plan = single_scanner
                .vector_search(filter_plan, &single_query)
                .await?;
            query_plans.push(Self::add_query_index_column(
                single_plan,
                query_index as i32,
            )?);
        }

        let unioned = UnionExec::try_new(query_plans)?;
        let unioned = Arc::new(RepartitionExec::try_new(
            unioned,
            Partitioning::RoundRobinBatch(1),
        )?) as Arc<dyn ExecutionPlan>;

        let query_index_sort = PhysicalSortExpr {
            expr: expressions::col(QUERY_INDEX_COL, unioned.schema().as_ref())?,
            options: SortOptions {
                descending: false,
                nulls_first: false,
            },
        };
        let distance_sort = PhysicalSortExpr {
            expr: expressions::col(DIST_COL, unioned.schema().as_ref())?,
            options: SortOptions {
                descending: false,
                nulls_first: false,
            },
        };
        let row_id_sort = PhysicalSortExpr {
            expr: expressions::col(ROW_ID, unioned.schema().as_ref())?,
            options: SortOptions {
                descending: false,
                nulls_first: false,
            },
        };

        Ok(Arc::new(SortExec::new(
            [query_index_sort, distance_sort, row_id_sort].into(),
            unioned,
        )))
    }

    fn add_query_index_column(
        plan: Arc<dyn ExecutionPlan>,
        query_index: i32,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = plan.schema();
        let mut projection_exprs = Vec::with_capacity(schema.fields().len() + 1);
        projection_exprs.push((
            Arc::new(Literal::new(ScalarValue::Int32(Some(query_index)))) as Arc<dyn PhysicalExpr>,
            QUERY_INDEX_COL.to_string(),
        ));
        for field in schema.fields() {
            projection_exprs.push((
                Arc::new(Column::new_with_schema(field.name(), schema.as_ref())?)
                    as Arc<dyn PhysicalExpr>,
                field.name().clone(),
            ));
        }
        Ok(Arc::new(ProjectionExec::try_new(projection_exprs, plan)?))
    }

    /// Combine ANN results with KNN results for data appended after index creation
    async fn knn_combined(
        &self,
        q: &Query,
        index_name: &str,
        indexed_segments: &[IndexMetadata],
        stale_rows: &HashMap<u32, RoaringBitmap>,
        mut knn_node: Arc<dyn ExecutionPlan>,
        filter_plan: &ExprFilterPlan,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let fallback_fragments = self
            .fragments_missing_from_index_segments(index_name, indexed_segments)
            .await?;

        let has_fallback = !fallback_fragments.is_empty();
        let has_stale = !stale_rows.is_empty();

        if !has_fallback && !has_stale {
            return Ok(knn_node);
        }

        let q = q.clone();
        debug_assert!(q.metric_type.is_some());

        // Ensure the vector column is present for distance computation.
        if knn_node.schema().column_with_name(&q.column).is_none() {
            let vector_projection = self
                .dataset
                .empty_projection()
                .union_column(&q.column, OnMissing::Error)?;
            knn_node = self.take(knn_node, vector_projection)?;
        }

        let mut columns = vec![q.column.clone()];
        if let Some(expr) = filter_plan.full_expr.as_ref() {
            let filter_columns = Planner::column_names_in_expr(expr);
            columns.extend(filter_columns);
        }

        // Collect flat-path plans; union order matches original (flat before ANN) so test snapshots
        // and downstream plan analyses remain stable.
        let mut flat_inputs: Vec<Arc<dyn ExecutionPlan>> = Vec::new();

        // Flat KNN for unindexed (new-data) fragments.
        if has_fallback {
            let vector_scan_projection = Arc::new(self.dataset.schema().project(&columns)?);
            // Note: we could try and use the scalar indices here to reduce the scope of this scan
            // but the most common case is that fragments newer than the vector index are also
            // newer than the scalar indices.
            let mut scan_node = self.scan_fragments(
                true,
                false,
                false,
                false,
                false,
                vector_scan_projection,
                Arc::new(fallback_fragments),
                // Can't pushdown limit/offset in an ANN search
                None,
                // We are re-ordering anyways, so no need to get data in a deterministic order.
                false,
            );
            if let Some(expr) = filter_plan.full_expr.as_ref() {
                scan_node = Arc::new(LanceFilterExec::try_new(expr.clone(), scan_node)?);
            }
            // Appended fragments are not covered by the index, so the external
            // row-address mask must be applied to them here.
            let scan_node = match self.external_row_mask.clone() {
                Some(mask) => Arc::new(RowAddrMaskFilterExec::new(scan_node, mask)) as _,
                None => scan_node,
            };
            let topk_fallback = self.flat_knn(scan_node, &q)?;
            let topk_fallback: Arc<dyn ExecutionPlan> =
                Arc::new(project(topk_fallback, knn_node.schema().as_ref())?);
            flat_inputs.push(topk_fallback);
        }

        // Flat KNN for stale rows only (row-level precision).
        // Only specific row addresses need re-scoring, not the whole fragment, so sparse overlays
        // incur near-zero overhead.
        if has_stale {
            // Fetch vector + filter columns for the stale rows. `flat_knn` sorts by row id, so the
            // take must carry it (the fallback scan above gets it via `scan_fragments`).
            let mut take_proj = self
                .dataset
                .empty_projection()
                .with_row_id()
                .union_column(&q.column, OnMissing::Error)?;
            if let Some(expr) = filter_plan.full_expr.as_ref() {
                let filter_columns = Planner::column_names_in_expr(expr);
                take_proj = take_proj.union_columns(filter_columns, OnMissing::Error)?;
            }
            let mut stale_node = self.stale_rows_take(stale_rows, take_proj).await?;
            if let Some(expr) = filter_plan.full_expr.as_ref() {
                stale_node = Arc::new(LanceFilterExec::try_new(expr.clone(), stale_node)?);
            }
            let topk_stale = self.flat_knn(stale_node, &q)?;
            let topk_stale: Arc<dyn ExecutionPlan> =
                Arc::new(project(topk_stale, knn_node.schema().as_ref())?);
            flat_inputs.push(topk_stale);
        }

        // Union: flat paths first (matching original order), then ANN results.
        flat_inputs.push(knn_node);
        let unioned = UnionExec::try_new(flat_inputs)?;
        let unioned = RepartitionExec::try_new(
            unioned,
            datafusion::physical_plan::Partitioning::RoundRobinBatch(1),
        )?;
        self.flat_knn(Arc::new(unioned), &q)
    }

    #[async_recursion]
    async fn fragments_covered_by_index_query(
        &self,
        index_expr: &ScalarIndexExpr,
    ) -> Result<RoaringBitmap> {
        match index_expr {
            ScalarIndexExpr::And(lhs, rhs) => {
                Ok(self.fragments_covered_by_index_query(lhs).await?
                    & self.fragments_covered_by_index_query(rhs).await?)
            }
            ScalarIndexExpr::Or(lhs, rhs) => Ok(self.fragments_covered_by_index_query(lhs).await?
                & self.fragments_covered_by_index_query(rhs).await?),
            ScalarIndexExpr::Not(expr) => self.fragments_covered_by_index_query(expr).await,
            ScalarIndexExpr::Query(search) => scalar_index_fragment_bitmap(
                self.dataset.as_ref(),
                &search.column,
                &search.index_name,
            )
            .await?
            .ok_or_else(|| {
                crate::Error::internal(format!(
                    "Index not found even though it must have been found earlier: {}",
                    search.index_name
                ))
            }),
        }
    }

    /// Given an index query, split the fragments into two groups and collect per-row stale data.
    ///
    /// - `relevant_frags`: covered by ALL indices. Stale rows within them are returned separately
    ///   so callers can block them from `MaterializeIndexExec` and re-score via a targeted take.
    /// - `missing_frags`: not covered by at least one index; fall back to full scan + filter.
    /// - `stale_rows`: per-fragment row offsets whose indexed values are stale due to a data
    ///   overlay committed after the index was built (field-aware, version-gated). Empty when no
    ///   overlays are present.
    ///
    /// There is no point in partially indexing a fragment (some indices cover it, others do not).
    /// If we have to do a full scan of a fragment for any reason, we do it entirely.
    pub(super) async fn partition_frags_by_coverage(
        &self,
        index_expr: &ScalarIndexExpr,
        fragments: Arc<Vec<Fragment>>,
    ) -> Result<(Vec<Fragment>, Vec<Fragment>, HashMap<u32, RoaringBitmap>)> {
        let covered_frags = self.fragments_covered_by_index_query(index_expr).await?;
        let stale_rows = self
            .overlay_stale_index_rows(index_expr, &fragments)
            .await?;
        let mut relevant_frags = Vec::with_capacity(fragments.len());
        let mut missing_frags = Vec::with_capacity(fragments.len());
        for fragment in fragments.iter() {
            if covered_frags.contains(fragment.id as u32) {
                // Indexed fragments stay on the indexed path. Stale rows within them are blocked
                // from the index result and re-evaluated separately via a targeted take.
                relevant_frags.push(fragment.clone());
            } else {
                missing_frags.push(fragment.clone());
            }
        }
        Ok((relevant_frags, missing_frags, stale_rows))
    }

    /// Add a knn search node to the input plan
    pub(super) fn flat_knn(
        &self,
        input: Arc<dyn ExecutionPlan>,
        q: &Query,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Resolve metric_type if not set (use default for the column's element type)
        let metric_type = match q.metric_type {
            Some(m) => m,
            None => {
                let (_, element_type) = get_vector_type(self.dataset.schema(), &q.column)?;
                default_distance_type_for(&element_type)
            }
        };
        let input = if self.is_batch_nearest {
            Arc::new(CoalescePartitionsExec::new(input)) as Arc<dyn ExecutionPlan>
        } else {
            input
        };
        let retain_vector = if self.is_batch_nearest {
            let vector_field_id = self.dataset.schema().field_id(q.column.as_str())?;
            self.projection_plan
                .physical_projection
                .contains_field_id(vector_field_id)
        } else {
            false
        };
        let flat_dist = Arc::new(KNNVectorDistanceExec::try_new_batch(
            input,
            &q.column,
            q.key.clone(),
            KnnBatchParams {
                is_batch: self.is_batch_nearest,
                query_count: self.nearest_query_count,
                k: q.k,
                lower_bound: q.lower_bound,
                upper_bound: q.upper_bound,
                distance_type: metric_type,
                retain_vector,
            },
        )?);

        if self.is_batch_nearest {
            return Ok(flat_dist);
        }

        let lower: Option<(Expr, Arc<dyn PhysicalExpr>)> = q
            .lower_bound
            .map(|v| -> Result<(Expr, Arc<dyn PhysicalExpr>)> {
                let logical = col(DIST_COL).gt_eq(lit(v));
                let schema = flat_dist.schema();
                let df_schema = DFSchema::try_from(schema)?;
                let physical = create_physical_expr(&logical, &df_schema, &ExecutionProps::new())?;
                Ok::<(Expr, Arc<dyn PhysicalExpr>), _>((logical, physical))
            })
            .transpose()?;

        let upper = q
            .upper_bound
            .map(|v| -> Result<(Expr, Arc<dyn PhysicalExpr>)> {
                let logical = col(DIST_COL).lt(lit(v));
                let schema = flat_dist.schema();
                let df_schema = DFSchema::try_from(schema)?;
                let physical = create_physical_expr(&logical, &df_schema, &ExecutionProps::new())?;
                Ok::<(Expr, Arc<dyn PhysicalExpr>), _>((logical, physical))
            })
            .transpose()?;

        let filter_expr = match (lower, upper) {
            (Some((llog, _)), Some((ulog, _))) => {
                let logical = llog.and(ulog);
                let schema = flat_dist.schema();
                let df_schema = DFSchema::try_from(schema)?;
                let physical = create_physical_expr(&logical, &df_schema, &ExecutionProps::new())?;
                Some((logical, physical))
            }
            (Some((llog, lphys)), None) => Some((llog, lphys)),
            (None, Some((ulog, uphys))) => Some((ulog, uphys)),
            (None, None) => None,
        };

        let knn_plan: Arc<dyn ExecutionPlan> = if let Some(filter_expr) = filter_expr {
            Arc::new(LanceFilterExec::try_new(filter_expr.0, flat_dist)?)
        } else {
            flat_dist
        };

        // Use DataFusion's [SortExec] for Top-K search
        let sort = SortExec::new(
            [
                PhysicalSortExpr {
                    expr: expressions::col(DIST_COL, knn_plan.schema().as_ref())?,
                    options: SortOptions {
                        descending: false,
                        nulls_first: false,
                    },
                },
                PhysicalSortExpr {
                    expr: expressions::col(ROW_ID, knn_plan.schema().as_ref())?,
                    options: SortOptions {
                        descending: false,
                        nulls_first: false,
                    },
                },
            ]
            .into(),
            knn_plan,
        )
        .with_fetch(Some(q.k));

        Self::flat_knn_not_null_filter(Arc::new(sort))
    }

    fn flat_knn_not_null_filter(
        knn_plan: Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let logical_not_null = col(DIST_COL).is_not_null();
        Ok(Arc::new(LanceFilterExec::try_new(
            logical_not_null,
            knn_plan,
        )?))
    }

    pub(super) fn get_fragments_as_bitmap(&self) -> RoaringBitmap {
        if let Some(fragments) = &self.fragments {
            RoaringBitmap::from_iter(fragments.iter().map(|f| f.id as u32))
        } else {
            self.dataset.fragment_bitmap.as_ref().clone()
        }
    }

    fn retain_relevant_index_segments(
        &self,
        index_segments: Vec<IndexMetadata>,
    ) -> Vec<IndexMetadata> {
        if let Some(fragments) = &self.fragments {
            let target_fragments = RoaringBitmap::from_iter(fragments.iter().map(|f| f.id as u32));
            index_segments
                .into_iter()
                .filter(|idx| {
                    idx.fragment_bitmap
                        .as_ref()
                        .is_some_and(|fragmap| !(fragmap & &target_fragments).is_empty())
                })
                .collect()
        } else {
            index_segments
        }
    }

    /// Retain only fragments that are in the user-specified fragment list.
    /// If no fragment list is specified, returns the fragments unchanged.
    pub(super) fn retain_target_fragments(&self, mut fragments: Vec<Fragment>) -> Vec<Fragment> {
        if let Some(target) = &self.fragments {
            let bitmap = RoaringBitmap::from_iter(target.iter().map(|f| f.id as u32));
            fragments.retain(|f| bitmap.contains(f.id as u32));
        }
        fragments
    }

    pub(super) fn get_indexed_frags(&self, index: &[IndexMetadata]) -> RoaringBitmap {
        let all_fragments = self.get_fragments_as_bitmap();

        let mut all_indexed_frags = RoaringBitmap::new();
        for idx in index {
            if let Some(fragmap) = idx.fragment_bitmap.as_ref() {
                all_indexed_frags |= fragmap;
            } else {
                // If any index is missing the fragment bitmap it is safest to just assume we
                // need all fragments
                return all_fragments;
            }
        }

        all_indexed_frags & all_fragments
    }

    /// Choose the prefilter source for an indexed ANN scan.
    ///
    /// Skip the row-ID allow-list when there is no predicate and every selected
    /// segment has known coverage with no currently visible fragment outside the
    /// scan scope. The scope cannot reject an indexed row in that case.
    /// `DatasetPreFilter` still enforces deletions and missing fragments, while
    /// ANN retains overlay and external masks.
    ///
    /// Check the segments' raw bitmaps: [`Self::get_indexed_frags`] intersects
    /// coverage with the scan scope and would hide excluded fragments. Obsolete
    /// fragment IDs do not block this optimization because they are not visible
    /// in the current snapshot. Predicates, unknown coverage, partial coverage,
    /// and an empty segment list fall back to [`Self::prefilter_source`].
    async fn ann_prefilter_source(
        &self,
        filter_plan: &ExprFilterPlan,
        indices: &[IndexMetadata],
    ) -> Result<PreFilterSource> {
        if filter_plan.is_empty() && !indices.is_empty() {
            let excluded_fragments =
                self.dataset.fragment_bitmap.as_ref() - &self.get_fragments_as_bitmap();
            let has_full_index_coverage = indices.iter().all(|index| {
                index
                    .fragment_bitmap
                    .as_ref()
                    .is_some_and(|fragments| fragments.is_disjoint(&excluded_fragments))
            });
            if has_full_index_coverage {
                return Ok(PreFilterSource::None);
            }
        }
        self.prefilter_source(filter_plan, self.get_indexed_frags(indices))
            .await
    }

    /// Create an Execution plan to do indexed ANN search
    async fn ann(
        &self,
        q: &Query,
        index: &[IndexMetadata],
        filter_plan: &ExprFilterPlan,
        overlay_block: Option<RowAddrMask>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let prefilter_source = self.ann_prefilter_source(filter_plan, index).await?;
        let inner_fanout_search = new_knn_exec(
            self.dataset.clone(),
            index,
            q,
            prefilter_source,
            overlay_block,
            self.external_row_mask.clone(),
        )?;
        let sort_expr = PhysicalSortExpr {
            expr: expressions::col(DIST_COL, inner_fanout_search.schema().as_ref())?,
            options: SortOptions {
                descending: false,
                nulls_first: false,
            },
        };
        let sort_expr_row_id = PhysicalSortExpr {
            expr: expressions::col(ROW_ID, inner_fanout_search.schema().as_ref())?,
            options: SortOptions {
                descending: false,
                nulls_first: false,
            },
        };
        Ok(Arc::new(
            SortExec::new([sort_expr, sort_expr_row_id].into(), inner_fanout_search)
                .with_fetch(Some(q.k * q.refine_factor.unwrap_or(1) as usize)),
        ))
    }

    // Create an Execution plan to do ANN over multivectors
    async fn multivec_ann(
        &self,
        q: &Query,
        index: &[IndexMetadata],
        filter_plan: &ExprFilterPlan,
        overlay_block: Option<RowAddrMask>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // we split the query procedure into two steps:
        // 1. collect the candidates by vector searching on each query vector
        // 2. scoring the candidates

        let over_fetch_factor = *DEFAULT_XTR_OVERFETCH;

        let prefilter_source = self.ann_prefilter_source(filter_plan, index).await?;
        let dim = get_vector_dim(self.dataset.schema(), &q.column)?;

        let num_queries = q.key.len() / dim;
        let new_queries = (0..num_queries)
            .map(|i| q.key.slice(i * dim, dim))
            .map(|query_vec| {
                let mut new_query = q.clone();
                new_query.key = query_vec;
                // with XTR, we don't need to refine the result with original vectors,
                // but here we really need to over-fetch the candidates to reach good enough recall.
                // TODO: improve the recall with WARP, expose this parameter to the users.
                new_query.refine_factor = Some(over_fetch_factor);
                new_query
            });
        let mut ann_nodes = Vec::with_capacity(new_queries.len());
        for query in new_queries {
            // this produces `nprobes * k * over_fetch_factor * num_indices` candidates
            let ann_node = new_knn_exec(
                self.dataset.clone(),
                index,
                &query,
                prefilter_source.clone(),
                overlay_block.clone(),
                self.external_row_mask.clone(),
            )?;
            let sort_expr = PhysicalSortExpr {
                expr: expressions::col(DIST_COL, ann_node.schema().as_ref())?,
                options: SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            };
            let sort_expr_row_id = PhysicalSortExpr {
                expr: expressions::col(ROW_ID, ann_node.schema().as_ref())?,
                options: SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            };
            let ann_node = Arc::new(
                SortExec::new([sort_expr, sort_expr_row_id].into(), ann_node)
                    .with_fetch(Some(q.k * over_fetch_factor as usize)),
            );
            ann_nodes.push(ann_node as Arc<dyn ExecutionPlan>);
        }

        let ann_node = Arc::new(MultivectorScoringExec::try_new(ann_nodes, q.clone())?);

        let sort_expr = PhysicalSortExpr {
            expr: expressions::col(DIST_COL, ann_node.schema().as_ref())?,
            options: SortOptions {
                descending: false,
                nulls_first: false,
            },
        };
        let sort_expr_row_id = PhysicalSortExpr {
            expr: expressions::col(ROW_ID, ann_node.schema().as_ref())?,
            options: SortOptions {
                descending: false,
                nulls_first: false,
            },
        };
        let ann_node = Arc::new(
            SortExec::new([sort_expr, sort_expr_row_id].into(), ann_node)
                .with_fetch(Some(q.k * q.refine_factor.unwrap_or(1) as usize)),
        );

        Ok(ann_node)
    }

    /// Create prefilter source from filter plan
    ///
    /// A prefilter is an input to a vector or fts search.  It tells us which rows are eligible
    /// for the search.  A prefilter is calculated by doing a filtered read of the row id column.
    pub(super) async fn prefilter_source(
        &self,
        filter_plan: &ExprFilterPlan,
        required_frags: RoaringBitmap,
    ) -> Result<PreFilterSource> {
        if filter_plan.is_empty() {
            let has_full_fragment_coverage = self.fragments.as_ref().is_none_or(|fragments| {
                let selected: HashSet<_> = fragments.iter().map(|fragment| fragment.id).collect();
                selected.len() == self.dataset.manifest.fragments.len()
                    && self
                        .dataset
                        .manifest
                        .fragments
                        .iter()
                        .all(|fragment| selected.contains(&fragment.id))
            });
            // A full-snapshot scope cannot reject any live row. Avoid materializing its row IDs,
            // but retain self.fragments for unindexed fallback and leave deletion/overlay masks
            // to DatasetPreFilter. Comparing IDs also prevents duplicate fragments hiding a gap.
            if has_full_fragment_coverage {
                log::trace!("no filter plan or effective fragment restriction, no prefilter");
                return Ok(PreFilterSource::None);
            }
        }

        // get fragments covered by index
        let fragments: Vec<Fragment> = self
            .dataset
            .manifest
            .fragments
            .iter()
            .filter(|f| required_frags.contains(f.id as u32))
            .cloned()
            .collect();

        // If explicitly specified fragments with .with_fragments(), intersect with those
        let fragments = Arc::new(self.retain_target_fragments(fragments));

        // Can only use ScalarIndexExec when the scalar index is exact and we are not scanning
        // a subset of the fragments.
        //
        // TODO: We could enhance ScalarIndexExec with a fragment bitmap to filter out rows that
        // are not in the fragments we are scanning.
        if filter_plan.is_exact_index_search() && self.fragments.is_none() {
            let index_query = filter_plan.index_query.as_ref().expect_ok()?;
            let (relevant_frags, missing_frags, stale_rows) = self
                .partition_frags_by_coverage(index_query, fragments.clone())
                .await?;

            // Overlay-stale rows must never reach the direct ScalarIndexExec path: it would hand
            // ANN/FTS a selection vector containing rows whose indexed values are now stale. When
            // any exist, fall through to the filtered-read prefilter, which masks them.
            if stale_rows.is_empty() && (missing_frags.is_empty() || self.fast_search) {
                log::trace!("prefilter entirely satisfied by exact index search");
                let result_format = self.index_expr_result_format();
                // We can only avoid materializing the index for a prefilter if:
                // 1. The search is indexed
                // 2. The index search is an exact search with no recheck or refine
                // 3. The indices cover at least the same fragments as the vector index,
                //    unless fast_search allows skipping uncovered fragments.
                let mut exec =
                    ScalarIndexExec::new(self.dataset.clone(), index_query.clone(), result_format);
                if let Some(segment_uuids) = self
                    .scalar_index_selection
                    .as_ref()
                    .and_then(|selection| selection.segment_uuids.clone())
                {
                    exec = exec.with_segment_uuids(segment_uuids);
                }
                if missing_frags.is_empty() && !relevant_frags.is_empty() {
                    exec = exec.with_fragment_scope(
                        relevant_frags
                            .iter()
                            .map(|fragment| fragment.id as u32)
                            .collect(),
                    );
                }
                return Ok(PreFilterSource::ScalarIndexQuery(Arc::new(exec)));
            } else {
                log::trace!("exact index search did not cover all fragments");
            }
        }

        // If one of our criteria is not met, we need to do a filtered read of just the row id column
        log::trace!(
            "prefilter is a filtered read of {} fragments",
            fragments.len()
        );
        let PlannedFilteredScan { plan, .. } = self
            .filtered_read(
                filter_plan,
                self.dataset.empty_projection().with_row_id(),
                false,
                Some(fragments),
                None,
                /*is_prefilter= */ true,
                None,
            )
            .await?;
        Ok(PreFilterSource::FilteredRowIds(plan))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use datafusion::logical_expr::{col, lit};
    use lance_file::version::LanceFileVersion;
    use rstest::rstest;

    use crate::dataset::scanner::test_dataset::TestVectorDataset;

    #[rstest]
    #[case::covered(vec![Some(vec![0])], true)]
    #[case::partial(vec![Some(vec![0, 1])], false)]
    #[case::removed_fragment(vec![Some(vec![0, 100])], true)]
    #[case::unknown(vec![None], false)]
    #[case::mixed_unknown(vec![Some(vec![0]), None], false)]
    #[case::multiple_segments(vec![Some(vec![0]), Some(vec![1])], false)]
    #[tokio::test]
    async fn test_segment_prefilter_coverage(
        #[case] coverage: Vec<Option<Vec<u32>>>,
        #[case] no_prefilter: bool,
    ) {
        let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
            .await
            .unwrap();
        test_ds.make_vector_index().await.unwrap();
        let metadata = test_ds.dataset.load_indices().await.unwrap();
        let indices: Vec<_> = coverage
            .into_iter()
            .map(|fragments| {
                let mut index = metadata[0].clone();
                index.fragment_bitmap = fragments.map(|ids| ids.into_iter().collect());
                index
            })
            .collect();
        let mut scanner = test_ds.dataset.scan();
        scanner.with_fragments(vec![test_ds.dataset.manifest.fragments[0].clone()]);
        let source = scanner
            .ann_prefilter_source(&ExprFilterPlan::default(), &indices)
            .await
            .unwrap();
        assert_eq!(matches!(source, PreFilterSource::None), no_prefilter);
    }

    #[rstest]
    #[case::all(vec![0, 1], false, true)]
    #[case::reordered(vec![1, 0], false, true)]
    #[case::duplicates(vec![0, 0], false, false)]
    #[case::subset(vec![0], false, false)]
    #[case::empty(vec![], false, false)]
    #[case::filtered(vec![0, 1], true, false)]
    #[tokio::test]
    async fn test_full_snapshot_prefilter(
        #[case] selected: Vec<usize>,
        #[case] has_filter: bool,
        #[case] no_prefilter: bool,
    ) {
        let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
            .await
            .unwrap();
        let fragments = &test_ds.dataset.manifest.fragments;
        assert_eq!(fragments.len(), 2);
        let mut scanner = test_ds.dataset.scan();
        scanner.with_fragments(selected.iter().map(|i| fragments[*i].clone()).collect());
        let filter = if has_filter {
            ExprFilterPlan::new_refine_only(col("i").gt(lit(100)))
        } else {
            ExprFilterPlan::default()
        };
        let source = scanner
            .prefilter_source(&filter, fragments.iter().map(|f| f.id as u32).collect())
            .await
            .unwrap();
        assert_eq!(matches!(source, PreFilterSource::None), no_prefilter);
        // Planning must preserve the explicit scope used by unindexed-fragment fallback.
        assert_eq!(scanner.fragments.as_ref().unwrap().len(), selected.len());
    }
}
