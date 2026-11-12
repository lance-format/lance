// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Vector-search source planning and index-to-flat fallback.
//!
//! This module maps nearest-neighbor queries to indexed or flat plans, applies
//! fragment and segment coverage, combines prefilters, and preserves distance
//! semantics during refinement. Stable-row-ID and deletion handling must match
//! the selected source. Tests live below and in
//! `dataset/tests/dataset_scanner_knn.rs` and
//! `dataset/tests/dataset_scanner_segments.rs`.

use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef};
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
use datafusion_expr::physical_planning_context::PhysicalPlanningContext;
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

use crate::index::DatasetIndexExt;
use crate::index::loading::DatasetIndexInternalExt;
use crate::index::scalar_logical::scalar_index_fragment_bitmap;
use crate::index::vector::utils::{
    default_distance_type_for, get_vector_dim, get_vector_type, validate_distance_type_for,
};
use crate::index::visibility::IndexVisibility;
use crate::io::exec::knn::{
    BatchRefineExec, KnnBatchParams, MultivectorScoringExec, QUERY_INDEX_COL,
    knn_empty_result_schema, new_knn_batch_exec, new_knn_exec,
};
use crate::io::exec::scalar_index::ScalarIndexExec;
use crate::io::exec::{
    FilterPlan as ExprFilterPlan, KNNVectorDistanceExec, LanceFilterExec, Planner, PreFilterSource,
    RowAddrMaskFilterExec, project,
};
use crate::{Error, Result};

use super::{DEFAULT_XTR_OVERFETCH, FilterPlan, PlannedFilteredScan, Scanner};

/// The multiple of the exact distance a vector index ranks by. Lance indexes
/// cosine on SQ, PQ, and RQ as L2 over normalized vectors, twice `1 - cos`;
/// FLAT ranks it, and every other metric, on its own scale.
fn index_distance_scale(metric: MetricType, segments: &[IndexMetadata]) -> f32 {
    let quantized_as_l2 = segments.iter().any(|segment| {
        segment.index_details.as_ref().is_some_and(|details| {
            let index_type = crate::index::vector::details::derive_vector_index_type(details);
            index_type.ends_with("PQ") || index_type.ends_with("SQ") || index_type.ends_with("RQ")
        })
    });
    if metric == MetricType::Cosine && quantized_as_l2 {
        2.0
    } else {
        1.0
    }
}

#[derive(Clone)]
pub(super) struct VectorRefinement {
    pub(super) column: String,
    pub(super) key: ArrayRef,
    pub(super) metric: MetricType,
}

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
                    let mut fts_plan = self
                        .fts(&filter_plan.expr_filter_plan, fts_query, fts_query.params())
                        .await?;
                    if fts_plan.schema().field_with_name(DOC_INDEX_COL).is_ok() {
                        fts_plan = self.deduplicate_fts_filter_rows(fts_plan)?;
                    }
                    let projection = self
                        .dataset
                        .empty_projection()
                        .union_column(&query.column, OnMissing::Error)?;
                    let plan = self.take(fts_plan, projection)?;

                    self.flat_knn(plan, query).await?
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

        if matches!(self.quantized_refine_factor, Some(0)) {
            return Err(Error::invalid_input(
                "Quantized refine factor cannot be zero".to_string(),
            ));
        }

        if let Some(refine) = &self.nearest_refine {
            if self.is_batch_nearest {
                return self.batch_per_query_vector_search(filter_plan, &q).await;
            }
            return self
                .cross_column_vector_search(filter_plan, &q, refine)
                .await;
        }

        // Sanity check
        let (vector_type, element_type) = get_vector_type(self.dataset.schema(), &q.column)?;

        let column_id = self.dataset.schema().field_id(q.column.as_str())?;
        let use_index = q.use_index;
        let indices = if use_index {
            self.dataset.load_indices().await?
        } else {
            Arc::new(vec![])
        };
        // The metric of the index chosen for this query. When that index ends up unused
        // because no searched fragment is in its coverage, or the fragment-reuse check drops
        // it, the flat branch below still ranks by it if the query names no metric, so
        // `_distance` keeps one scale whichever fragments are searched.
        let mut chosen_index_metric = None;
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

                // From the details alone: opening a segment the scope may exclude would add
                // I/O and a failure mode to what can end as a flat search.
                chosen_index_metric = requested_index_segments
                    .iter()
                    .find_map(crate::index::vector::details::metric_type_from_index_metadata);
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
            } else {
                // An index can only answer a query on the column it is keyed on, which
                // is always `fields[0]`. Not `contains`: `fields` also lists columns the
                // index merely carries values for, which it cannot search. Not a boundary
                // derived from `covering_fields` either -- that is computed from a field
                // older writers drop, so it would widen to the carried columns exactly
                // when the declaration is lost.
                let candidates = indices.iter().filter(|i| {
                    i.fields.first() == Some(&column_id)
                        && crate::index::segment::index_type_is_known(i)
                });

                // A column may carry several vector indices with different metrics. The
                // query's metric has to take part in *selecting* one, not merely veto the
                // first candidate: picking the first keyed index and then rejecting it on
                // a metric mismatch brute-forced the whole column even when another index
                // on it had exactly the requested metric (issue #9873).
                let mut selected: Option<(&IndexMetadata, MetricType)> = None;
                let mut saw_candidate = false;
                for index in candidates {
                    saw_candidate = true;
                    // Resolve the metric from index metadata first (fast path for newer
                    // indices); open the index only for legacy candidates without details.
                    let index_metric = if let Some(metric) =
                        crate::index::vector::details::metric_type_from_index_metadata(index)
                    {
                        metric
                    } else {
                        self.dataset
                            .open_vector_index(
                                q.column.as_str(),
                                &index.uuid,
                                &NoOpMetricsCollector,
                            )
                            .await?
                            .metric_type()
                    };
                    match q.metric_type {
                        // No metric requested: the first keyed index answers, as before.
                        None => {
                            selected = Some((index, index_metric));
                            break;
                        }
                        Some(user_metric) if user_metric == index_metric => {
                            selected = Some((index, index_metric));
                            break;
                        }
                        // Metric mismatch: keep looking at the column's other indices.
                        Some(_) => continue,
                    }
                }

                match selected {
                    Some((index, index_metric)) => {
                        chosen_index_metric = Some(index_metric);
                        let index_segments = self.retain_relevant_index_segments(
                            self.dataset.load_indices_by_name(&index.name).await?,
                        );
                        let index_frags = self.get_indexed_frags(&index_segments);
                        if !index_segments.is_empty() && !index_frags.is_empty() {
                            Some((index.name.clone(), index_segments, index_metric))
                        } else {
                            None
                        }
                    }
                    None => {
                        if saw_candidate {
                            // Every index on this column uses a different metric than the
                            // query asked for, so none can accelerate it.
                            log::warn!(
                                "Requested metric {:?} does not match any index on column '{}', falling back to brute-force search",
                                q.metric_type,
                                q.column
                            );
                        }
                        None
                    }
                }
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
            let overlay_block = stale_rows.block_mask().await?;

            let ann_query = if self.quantized_refine_factor.is_some() {
                let exact_factor = q.refine_factor.unwrap_or(1) as usize;
                let mut ann_query = q.clone();
                ann_query.k = q.k.checked_mul(exact_factor).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "quantized refinement result count overflows: k={} exact_refine_factor={exact_factor}",
                        q.k
                    ))
                })?;
                ann_query.refine_factor = None;
                ann_query
            } else {
                q.clone()
            };

            let ann_node = match vector_type {
                DataType::FixedSizeList(_, _) => {
                    // Refine applies the distance range to exact distances, but
                    // the search still has to, or a lower bound leaves it only
                    // candidates refine then drops. It ranks on the index's own
                    // scale, so it gets the range on that scale.
                    let ann_query = match q.refine_factor {
                        Some(_) => {
                            let scale = index_distance_scale(index_metric, &index_segments);
                            Query {
                                lower_bound: q.lower_bound.map(|lower| lower * scale),
                                upper_bound: q.upper_bound.map(|upper| upper * scale),
                                ..ann_query
                            }
                        }
                        None => ann_query,
                    };
                    self.ann(
                        &ann_query,
                        &index_segments,
                        filter_plan,
                        overlay_block.clone(),
                    )
                    .await?
                }
                DataType::List(_) => {
                    if self.quantized_refine_factor.is_some() {
                        return Err(Error::not_supported(
                            "quantized refinement is not supported for multivector columns",
                        ));
                    }
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
                self.flat_knn(knn_node_with_vector, &q).await?
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
            if self.quantized_refine_factor.is_some() {
                return Err(Error::not_supported(
                    "quantized refinement requires an IVF_RQ vector index",
                ));
            }
            if self.fast_search {
                return Ok(Arc::new(EmptyExec::new(knn_empty_result_schema(
                    self.is_batch_nearest,
                ))));
            }
            // Resolve metric type for flat search: the query's, else the chosen index's,
            // else the column type's default. An untrained index can record a metric its
            // column cannot use, so such a metric is skipped.
            let metric = q
                .metric_type
                .or(chosen_index_metric
                    .filter(|metric| validate_distance_type_for(*metric, &element_type).is_ok()))
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
            self.flat_knn(plan, &q).await
        }
    }

    async fn cross_column_vector_search(
        &self,
        filter_plan: &ExprFilterPlan,
        q: &Query,
        refine: &VectorRefinement,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let factor = q
            .refine_factor
            .filter(|factor| *factor > 0)
            .ok_or_else(|| {
                Error::invalid_input(
                    "refine_column requires an explicit positive refine_factor".to_string(),
                )
            })?;
        let mut coarse = q.clone();
        coarse.k = q.k.checked_mul(factor as usize).ok_or_else(|| {
            Error::invalid_input("k * refine_factor overflows the candidate budget".to_string())
        })?;
        coarse.refine_factor = None;
        // Bounds apply to the final distance space, not the coarse index.
        coarse.lower_bound = None;
        coarse.upper_bound = None;

        let mut coarse_scanner = self.clone();
        coarse_scanner.nearest_refine = None;
        coarse_scanner.nearest = Some(coarse.clone());
        // Merge indexed, appended, and updated rows before cross-column scoring.
        let candidates = coarse_scanner.vector_search(filter_plan, &coarse).await?;
        let projection = self
            .dataset
            .empty_projection()
            .union_column(&refine.column, OnMissing::Error)?;
        let candidates = self.take(candidates, projection)?;

        let mut final_query = q.clone();
        final_query.column = refine.column.clone();
        final_query.key = refine.key.clone();
        final_query.metric_type = Some(refine.metric);
        final_query.refine_factor = None;
        final_query.use_index = false;
        self.flat_knn(candidates, &final_query).await
    }

    /// Whether a batch (multi-query) vector search can use the shared-scan
    /// indexed fast path ([`new_knn_batch_exec`]) instead of running one indexed
    /// search per query vector.
    ///
    /// Requires all of:
    /// - no quantized refine step (exact refinement is supported);
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
        // Quantized refinement reranks with Accurate RQ scores, which the shared
        // batch path does not implement. Fall back to the per-query loop.
        if self.quantized_refine_factor.is_some() {
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
        // across the batch. HNSW and mixed indexed/unindexed scans fall
        // back to the per-query loop below, which never regresses behavior.
        if q.refine_factor == Some(0) {
            return Err(Error::invalid_input("Refine factor cannot be zero"));
        }
        if self
            .batch_index_search_supported(index_name, index_segments, q)
            .await?
        {
            let mut batch_query = q.clone();
            batch_query.metric_type = Some(index_metric);
            batch_query.k =
                q.k.checked_mul(q.refine_factor.unwrap_or(1) as usize)
                    .ok_or_else(|| {
                        Error::invalid_input(format!(
                            "Candidate count overflows usize: k={}, refine_factor={:?}",
                            q.k, q.refine_factor
                        ))
                    })?;
            if q.refine_factor.is_some() {
                // Refinement applies the range to exact distances, while the
                // ANN stage ranks on the index's own scale. Transform only the
                // candidate bounds; BatchRefineExec receives the original query.
                let scale = index_distance_scale(index_metric, index_segments);
                batch_query.lower_bound = q.lower_bound.map(|bound| bound * scale);
                batch_query.upper_bound = q.upper_bound.map(|bound| bound * scale);
            }
            batch_query.refine_factor = None;
            let prefilter_source = self
                .ann_prefilter_source(filter_plan, index_segments)
                .await?;
            let input = new_knn_batch_exec(
                self.dataset.clone(),
                index_segments,
                &batch_query,
                self.nearest_query_count,
                prefilter_source,
            )?;
            return if q.refine_factor.is_some() {
                let mut refine_query = q.clone();
                refine_query.metric_type = Some(index_metric);
                Ok(Arc::new(BatchRefineExec::try_new(
                    self.dataset.clone(),
                    input,
                    refine_query,
                    self.nearest_query_count,
                )?))
            } else {
                Ok(input)
            };
        }

        self.batch_per_query_vector_search(filter_plan, q).await
    }

    async fn batch_per_query_vector_search(
        &self,
        filter_plan: &ExprFilterPlan,
        q: &Query,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let query_dim = q.key.len() / self.nearest_query_count;
        let mut query_plans = Vec::with_capacity(self.nearest_query_count);

        for query_index in 0..self.nearest_query_count {
            let mut single_query = q.clone();
            single_query.key = q.key.slice(query_index * query_dim, query_dim);

            let mut single_scanner = self.clone();
            single_scanner.nearest_query_count = 1;
            single_scanner.is_batch_nearest = false;
            single_scanner.nearest = Some(single_query.clone());
            if let Some(refine) = &self.nearest_refine {
                let refine_dim = refine.key.len() / self.nearest_query_count;
                let mut single_refine = refine.clone();
                single_refine.key = refine.key.slice(query_index * refine_dim, refine_dim);
                single_scanner.nearest_refine = Some(single_refine);
            }

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
        stale_rows: &IndexVisibility,
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
        let keep_quantized_distances =
            self.quantized_refine_factor.is_some() && q.refine_factor.is_none();

        // Retained RQ scores use the index's native distance scale. Convert the
        // flat arms' bounds before filtering, then their scores before merging;
        // fetching indexed vectors just to recompute distances defeats quantized refinement.
        let flat_distance_scale = if keep_quantized_distances {
            q.metric_type
                .map_or(1.0, |metric| index_distance_scale(metric, indexed_segments))
        } else {
            1.0
        };
        let flat_query = Query {
            lower_bound: q.lower_bound.map(|bound| bound / flat_distance_scale),
            upper_bound: q.upper_bound.map(|bound| bound / flat_distance_scale),
            ..q.clone()
        };

        // Ensure the vector column is present for distance computation.
        if !keep_quantized_distances && knn_node.schema().column_with_name(&q.column).is_none() {
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
            let topk_fallback = self.flat_knn(scan_node, &flat_query).await?;
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
            let mut stale_node =
                self.row_ids_take(stale_rows.result_rows().await?.clone(), take_proj)?;
            if let Some(expr) = filter_plan.full_expr.as_ref() {
                stale_node = Arc::new(LanceFilterExec::try_new(expr.clone(), stale_node)?);
            }
            let topk_stale = self.flat_knn(stale_node, &flat_query).await?;
            let topk_stale: Arc<dyn ExecutionPlan> =
                Arc::new(project(topk_stale, knn_node.schema().as_ref())?);
            flat_inputs.push(topk_stale);
        }

        if flat_distance_scale != 1.0 {
            for input in &mut flat_inputs {
                let exprs: Vec<_> = input
                    .schema()
                    .fields()
                    .iter()
                    .enumerate()
                    .map(|(index, field)| {
                        let column =
                            Arc::new(Column::new(field.name(), index)) as Arc<dyn PhysicalExpr>;
                        let expr = if field.name() == DIST_COL {
                            Arc::new(expressions::BinaryExpr::new(
                                column,
                                datafusion::logical_expr::Operator::Multiply,
                                Arc::new(Literal::new(ScalarValue::Float32(Some(
                                    flat_distance_scale,
                                )))),
                            )) as Arc<dyn PhysicalExpr>
                        } else {
                            column
                        };
                        (expr, field.name().clone())
                    })
                    .collect();
                *input = Arc::new(ProjectionExec::try_new(exprs, input.clone())?);
            }
        }

        // Union: flat paths first (matching original order), then ANN results.
        flat_inputs.push(knn_node);
        let unioned = UnionExec::try_new(flat_inputs)?;
        let unioned: Arc<dyn ExecutionPlan> = Arc::new(RepartitionExec::try_new(
            unioned,
            datafusion::physical_plan::Partitioning::RoundRobinBatch(1),
        )?);
        if keep_quantized_distances {
            let sort = SortExec::new(
                [
                    PhysicalSortExpr {
                        expr: expressions::col(DIST_COL, unioned.schema().as_ref())?,
                        options: SortOptions {
                            descending: false,
                            nulls_first: false,
                        },
                    },
                    PhysicalSortExpr {
                        expr: expressions::col(ROW_ID, unioned.schema().as_ref())?,
                        options: SortOptions {
                            descending: false,
                            nulls_first: false,
                        },
                    },
                ]
                .into(),
                unioned,
            )
            .with_fetch(Some(q.k));
            Self::flat_knn_not_null_filter(Arc::new(sort))
        } else {
            self.flat_knn(unioned, &q).await
        }
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
    ) -> Result<(Vec<Fragment>, Vec<Fragment>, IndexVisibility)> {
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
    pub(super) async fn flat_knn(
        &self,
        input: Arc<dyn ExecutionPlan>,
        q: &Query,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let (_, element_type) = get_vector_type(self.dataset.schema(), &q.column)?;
        let mut metric_type = q.metric_type;
        // Ranking FTS hits still needs the index's metric, even though the rows
        // are scored directly instead of being retrieved through ANN.
        if metric_type.is_none() && q.use_index {
            let column_id = self.dataset.schema().field_id(&q.column)?;
            let indices = self.dataset.load_indices().await?;
            if let Some(index) = indices.iter().find(|index| {
                index.fields.first() == Some(&column_id)
                    && crate::index::segment::index_type_is_known(index)
            }) {
                let index_metric = if let Some(metric) =
                    crate::index::vector::details::metric_type_from_index_metadata(index)
                {
                    metric
                } else {
                    self.dataset
                        .open_vector_index(&q.column, &index.uuid, &NoOpMetricsCollector)
                        .await?
                        .metric_type()
                };
                // Match the flat fallback in vector_search for untrained indices
                // whose recorded metric is incompatible with the column type.
                if validate_distance_type_for(index_metric, &element_type).is_ok() {
                    metric_type = Some(index_metric);
                }
            }
        }
        let metric_type = metric_type.unwrap_or_else(|| default_distance_type_for(&element_type));
        validate_distance_type_for(metric_type, &element_type)?;
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
                let physical = create_physical_expr(
                    &logical,
                    &df_schema,
                    &ExecutionProps::new(),
                    &PhysicalPlanningContext::default(),
                )?;
                Ok::<(Expr, Arc<dyn PhysicalExpr>), _>((logical, physical))
            })
            .transpose()?;

        let upper = q
            .upper_bound
            .map(|v| -> Result<(Expr, Arc<dyn PhysicalExpr>)> {
                let logical = col(DIST_COL).lt(lit(v));
                let schema = flat_dist.schema();
                let df_schema = DFSchema::try_from(schema)?;
                let physical = create_physical_expr(
                    &logical,
                    &df_schema,
                    &ExecutionProps::new(),
                    &PhysicalPlanningContext::default(),
                )?;
                Ok::<(Expr, Arc<dyn PhysicalExpr>), _>((logical, physical))
            })
            .transpose()?;

        let filter_expr = match (lower, upper) {
            (Some((llog, _)), Some((ulog, _))) => {
                let logical = llog.and(ulog);
                let schema = flat_dist.schema();
                let df_schema = DFSchema::try_from(schema)?;
                let physical = create_physical_expr(
                    &logical,
                    &df_schema,
                    &ExecutionProps::new(),
                    &PhysicalPlanningContext::default(),
                )?;
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
            self.quantized_refine_factor,
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
                None,
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

        let is_legacy = self
            .dataset
            .manifest()
            .data_storage_format
            .lance_file_format()
            == lance_file::version::ConcreteFileVersion::V1;
        // Keep the legacy storage planner's existing direct-mask eligibility.
        if is_legacy && filter_plan.is_exact_index_search() && self.fragments.is_none() {
            let index_query = filter_plan.index_query.as_ref().expect_ok()?;
            let (relevant, missing, stale_rows) = self
                .partition_frags_by_coverage(index_query, fragments.clone())
                .await?;
            if stale_rows.is_empty() && (missing.is_empty() || self.fast_search) {
                let mut exec = ScalarIndexExec::new(
                    self.dataset.clone(),
                    index_query.clone(),
                    self.index_expr_result_format(),
                );
                if let Some(segment_uuids) = self
                    .scalar_index_selection
                    .as_ref()
                    .and_then(|selection| selection.segment_uuids.clone())
                {
                    exec = exec.with_segment_uuids(segment_uuids);
                }
                if missing.is_empty() && !relevant.is_empty() {
                    exec = exec.with_fragment_scope(
                        relevant.iter().map(|fragment| fragment.id as u32).collect(),
                    );
                }
                return Ok(PreFilterSource::ScalarIndexQuery(Arc::new(exec)));
            }
        }
        if filter_plan.is_exact_index_search() && !is_legacy {
            let index_query = filter_plan.index_query.as_ref().expect_ok()?;
            let (relevant_frags, missing_frags, stale_rows) = self
                .partition_frags_by_coverage(index_query, fragments.clone())
                .await?;

            // Stale index values require the existing re-evaluation path. Missing physical
            // counts cannot be recovered from column data on an index-only prefilter path.
            if stale_rows.is_empty()
                && relevant_frags
                    .iter()
                    .all(|fragment| fragment.physical_rows.is_some())
            {
                let scope: RoaringBitmap = relevant_frags
                    .iter()
                    .map(|fragment| fragment.id as u32)
                    .collect();
                let mut coverage = scope.clone();
                let has_missing_fragments = !missing_frags.is_empty();
                let missing = if !self.fast_search && !missing_frags.is_empty() {
                    if filter_plan.full_expr.is_none() {
                        return Err(Error::internal(format!(
                            "exact scalar prefilter {index_query} has no full predicate to evaluate missing fragments"
                        )));
                    }
                    coverage.extend(missing_frags.iter().map(|fragment| fragment.id as u32));
                    let mut refine = filter_plan.clone();
                    refine.make_refine_only();
                    let PlannedFilteredScan { plan, .. } = self
                        .filtered_read(
                            &refine,
                            self.dataset.empty_projection().with_row_id(),
                            false,
                            Some(Arc::new(missing_frags)),
                            None,
                            true,
                            None,
                        )
                        .await?;
                    Some(plan)
                } else {
                    None
                };
                if scope.is_empty() {
                    if let Some(plan) = missing {
                        return Ok(PreFilterSource::FilteredRowIds(plan));
                    }
                    let batch = lance_select::IndexExprResult::exact(
                        lance_select::RowAddrMask::allow_nothing(),
                    )
                    .serialize(&coverage, self.index_expr_result_format())?;
                    let plan = datafusion::datasource::memory::MemorySourceConfig::try_new_exec(
                        &[vec![batch]],
                        self.index_expr_result_format().schema().clone(),
                        None,
                    )?;
                    return Ok(PreFilterSource::ScalarIndexQuery(plan));
                }
                let result_format = self.index_expr_result_format();
                let mut exec =
                    ScalarIndexExec::new(self.dataset.clone(), index_query.clone(), result_format)
                        .with_fragment_scope(scope.clone());
                if let Some(segment_uuids) = self
                    .scalar_index_selection
                    .as_ref()
                    .and_then(|selection| selection.segment_uuids.clone())
                {
                    exec = exec.with_segment_uuids(segment_uuids);
                }
                // The consumer bounds hits to its own index fragments. A separate scope
                // mask is needed only for explicit selections or skipped/missing coverage.
                if self.fragments.is_none() && !has_missing_fragments {
                    return Ok(PreFilterSource::ScalarIndexQuery(Arc::new(exec)));
                }
                let plan = crate::io::exec::PrefilterUnionExec::new(
                    self.dataset.clone(),
                    Arc::new(exec),
                    missing,
                    scope,
                    coverage,
                    result_format,
                );
                return Ok(PreFilterSource::ScalarIndexQuery(Arc::new(plan)));
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
