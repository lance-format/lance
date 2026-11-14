// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shared-scan batch (multi-query) IVF search.

use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Float32Builder, Int32Builder, UInt64Builder};
use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};
use datafusion_physical_expr::{Distribution, EquivalenceProperties};
use futures::{StreamExt, stream};
use lance_core::utils::futures::FinallyStreamExt;
use lance_index::metrics::{IndexTimer, IndexTiming};
use lance_index::prefilter::PreFilter;
use lance_index::scalar::minhash_lsh::{MinHashHit, TopHits};
use lance_index::vector::Query;
use lance_table::format::IndexMetadata;

use super::super::utils::{PreFilterMasks, PreFilterSource, build_prefilter};
use super::{
    AnnIndexMetrics, LateSearchBudget, batch_probe, check_segment_contract,
    knn_empty_result_schema, normalize_batch_query_for_index, prefilter_for_segment,
    segment_ownership_mask,
};
use crate::dataset::Dataset;
use crate::index::loading::DatasetIndexInternalExt;
use crate::index::vector::ivf::VectorQueryContract;
use crate::{Error, Result};

/// Build a batch (multi-query) indexed vector search plan.
///
/// `query.key` must hold all `query_count` query vectors concatenated.
pub fn new_knn_batch_exec(
    dataset: Arc<Dataset>,
    indices: &[IndexMetadata],
    query: &Query,
    query_count: usize,
    prefilter_source: PreFilterSource,
) -> Result<Arc<dyn ExecutionPlan>> {
    Ok(Arc::new(ANNIvfBatchExec::try_new(
        dataset,
        indices.to_vec(),
        query.clone(),
        query_count,
        prefilter_source,
    )?))
}

/// [ExecutionPlan] for batch (multi-query) IVF vector search.
///
/// Where the single-query path uses [`super::ANNIvfPartitionExec`] +
/// [`super::ANNIvfSubIndexExec`], this node ranks every query vector against the IVF
/// centroids and then asks the index to read each probed partition's storage
/// once, scoring all queries that probe it
/// (via [`lance_index::vector::VectorIndex::search_partitions_batch`]). The prefilter is built once
/// and shared across all queries.
///
/// This is a separate node rather than a mode on the two single-query nodes
/// because the two-node pipeline streams one partition-list per delta through a
/// per-query top-k, whereas the shared scan must invert queries onto partitions
/// and keep one heap per query in a single pass. It still reuses the underlying
/// primitives (partition load, prefilter wiring via `build_prefilter`,
/// and the per-partition accumulate the index performs).
///
/// Output schema: `{query_index: Int32, _distance: Float32, _rowid: UInt64}`,
/// sorted by `(query_index, _distance, _rowid)`, with up to `k` rows per query.
///
/// Probing follows the single-query adaptive rules (initial budget from
/// `AutoProbePolicy`, then late search up to `maximum_nprobes` for queries that
/// found fewer than `k` rows) while sharing partition loads across queries; see
/// the `batch_probe` module. With fixed nprobes the late search never runs.
#[derive(Debug)]
pub struct ANNIvfBatchExec {
    dataset: Arc<Dataset>,
    indices: Vec<IndexMetadata>,
    /// Vector query whose `key` holds all `query_count` vectors concatenated.
    query: Query,
    query_count: usize,
    prefilter_source: PreFilterSource,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl ANNIvfBatchExec {
    pub fn try_new(
        dataset: Arc<Dataset>,
        indices: Vec<IndexMetadata>,
        query: Query,
        query_count: usize,
        prefilter_source: PreFilterSource,
    ) -> Result<Self> {
        if indices.is_empty() {
            return Err(Error::index(
                "ANNIvfBatchExec: no index found for query".to_string(),
            ));
        }
        if query_count == 0 || !query.key.len().is_multiple_of(query_count) {
            return Err(Error::invalid_input(format!(
                "ANNIvfBatchExec: query key length {} is not divisible by query count {query_count}",
                query.key.len()
            )));
        }
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(knn_empty_result_schema(true)),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Ok(Self {
            dataset,
            indices,
            query,
            query_count,
            prefilter_source,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    /// Returns a reference to the vector query. Its `key` holds all
    /// [`Self::query_count`] vectors concatenated.
    pub fn query(&self) -> &Query {
        &self.query
    }

    /// Returns the number of query vectors packed into [`Self::query`].
    pub fn query_count(&self) -> usize {
        self.query_count
    }

    /// Returns a reference to the dataset.
    pub fn dataset(&self) -> &Arc<Dataset> {
        &self.dataset
    }

    /// Returns a reference to the index metadata.
    pub fn indices(&self) -> &[IndexMetadata] {
        &self.indices
    }

    /// Returns a reference to the prefilter source.
    pub fn prefilter_source(&self) -> &PreFilterSource {
        &self.prefilter_source
    }
}

impl DisplayAs for ANNIvfBatchExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "ANNIvfBatch: query_count={}, k={}, deltas={}",
                    self.query_count,
                    self.query.k,
                    self.indices.len()
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "ANNIvfBatch\nquery_count={}\nk={}\ndeltas={}",
                    self.query_count,
                    self.query.k,
                    self.indices.len()
                )
            }
        }
    }
}

impl ExecutionPlan for ANNIvfBatchExec {
    fn name(&self) -> &str {
        "ANNIvfBatchExec"
    }

    fn schema(&self) -> SchemaRef {
        knn_empty_result_schema(true)
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        match &self.prefilter_source {
            PreFilterSource::None => vec![],
            PreFilterSource::FilteredRowIds(src) => vec![src],
            PreFilterSource::ScalarIndexQuery(src) => vec![src],
        }
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        self.children()
            .iter()
            .map(|_| Distribution::SinglePartition)
            .collect()
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let prefilter_source = match (&self.prefilter_source, children.len()) {
            (PreFilterSource::None, 0) => PreFilterSource::None,
            (PreFilterSource::FilteredRowIds(_), 1) => {
                PreFilterSource::FilteredRowIds(children.pop().expect("length checked"))
            }
            (PreFilterSource::ScalarIndexQuery(_), 1) => {
                PreFilterSource::ScalarIndexQuery(children.pop().expect("length checked"))
            }
            _ => {
                return Err(DataFusionError::Internal(
                    "ANNIvfBatchExec given an unexpected number of children".to_string(),
                ));
            }
        };
        Ok(Arc::new(Self {
            dataset: self.dataset.clone(),
            indices: self.indices.clone(),
            query: self.query.clone(),
            query_count: self.query_count,
            prefilter_source,
            properties: self.properties.clone(),
            metrics: ExecutionPlanMetricsSet::new(),
        }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::context::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let schema = self.schema();
        let ds = self.dataset.clone();
        let column = self.query.column.clone();
        let indices = self.indices.clone();
        let query = self.query.clone();
        let query_count = self.query_count;
        let metrics = Arc::new(AnnIndexMetrics::new(&self.metrics, partition));
        let metrics_clone = metrics.clone();
        let timer = Instant::now();

        let pre_filter = build_prefilter(
            context,
            partition,
            &self.prefilter_source,
            ds.clone(),
            &indices,
            // Batch queries have no overlay or external row-address mask.
            PreFilterMasks {
                overlay_block: None,
                external_mask: None,
            },
            &self.metrics,
        )?;

        // Same per-segment ownership gate as `ANNIvfSubIndexExec`: only restrict
        // when every segment has a fragment_bitmap.
        let has_segment_bitmaps = indices.iter().all(|idx| idx.fragment_bitmap.is_some());

        let result_schema = schema.clone();
        let fut = async move {
            let dim = query.key.len() / query_count;
            // Each max-heap retains only the best k candidates across deltas.
            let mut candidates = (0..query_count)
                .map(|_| TopHits::new(query.k))
                .collect::<Vec<_>>();
            let mut first_contract = None;
            let mut deltas = Vec::with_capacity(indices.len());
            // Early search: every delta probes each query's initial budget in one
            // shared scan, mirroring the single-query initial search.
            for index_meta in &indices {
                // A segment's index file may still hold rows for fragments pruned from
                // its fragment_bitmap (e.g. after an in-place `update_columns`) that a
                // newer delta now owns. As in the single-query path, search with the
                // segment's ownership prefilter and drop any unowned row, so the stale
                // copy is never returned next to the fresh one.
                let segment_pre_filter =
                    prefilter_for_segment(ds.clone(), index_meta, pre_filter.clone()).await?;
                let seg_mask = if has_segment_bitmaps {
                    segment_ownership_mask(&ds, index_meta).await?
                } else {
                    None
                };
                let index = {
                    let _open_timer =
                        IndexTimer::new(&metrics.index_metrics, IndexTiming::IndexOpen);
                    ds.open_vector_index(&column, &index_meta.uuid, &metrics.index_metrics)
                        .await?
                };
                check_segment_contract(
                    &mut first_contract,
                    index_meta.uuid,
                    VectorQueryContract::of_index(index.as_ref()),
                )?;
                // The scanner's `batch_index_search_supported` gate decides which
                // indices reach this node; this check only guards against that
                // gate and the index implementation disagreeing (an internal
                // invariant, not a user-facing error).
                if !index.supports_batch_partition_search() {
                    return Err(DataFusionError::Internal(format!(
                        "ANNIvfBatchExec reached for index {} that does not support batch \
                         partition search",
                        index_meta.uuid
                    )));
                }
                // Normalize each query vector independently (cosine only)
                // before ranking; see normalize_batch_query_for_index.
                let normalized = normalize_batch_query_for_index(
                    index.as_ref(),
                    query.clone(),
                    query_count,
                    dim,
                )?;
                let plans = batch_probe::plan_batch_probes(
                    index.clone(),
                    normalized.clone(),
                    query_count,
                    dim,
                )
                .await?;
                let (early_partitions, early_dists): (Vec<_>, Vec<_>) =
                    plans.iter().map(|plan| plan.early_partitions()).unzip();
                let per_query = batch_probe::search_batch(
                    index.clone(),
                    normalized.clone(),
                    early_partitions,
                    early_dists,
                    segment_pre_filter.clone(),
                    seg_mask.as_deref(),
                    &metrics,
                )
                .await?;
                for (query_index, batch) in per_query.iter().enumerate() {
                    batch_probe::append_candidates(batch, &mut candidates[query_index])?;
                }
                // As in the single-query path, a delta whose own prefilter selects
                // nothing is counted once and kept out of late search.
                segment_pre_filter.wait_for_ready().await?;
                if segment_pre_filter.selects_nothing() {
                    metrics.segments_skipped.add(1);
                }
                let mut delta = batch_probe::BatchDelta {
                    index,
                    query: normalized,
                    plans,
                    pre_filter: segment_pre_filter,
                    seg_mask,
                };
                if delta.retain_late_work(&candidates, query.k, query.k) {
                    deltas.push(delta);
                }
            }
            // Later deltas' early rows can complete queries an earlier delta was
            // retained for, and once the prefilter is ready a query that already
            // holds every row it can match needs no ranking either.
            pre_filter.wait_for_ready().await?;
            let budget = LateSearchBudget::new(&pre_filter.mask(), query.k).stop_at();
            deltas.retain_mut(|delta| delta.retain_late_work(&candidates, query.k, budget));
            batch_probe::late_search(
                &deltas,
                dim,
                query.k,
                pre_filter.clone(),
                &mut candidates,
                &metrics,
            )
            .await?;
            // Per-query top-k merge across deltas, tagged with query_index.
            let capacity = candidates.iter().map(|hits| query.k - hits.missing()).sum();
            let mut query_index_builder = Int32Builder::with_capacity(capacity);
            let mut distance_builder = Float32Builder::with_capacity(capacity);
            let mut row_id_builder = UInt64Builder::with_capacity(capacity);
            for (query_index, hits) in candidates.into_iter().enumerate() {
                for MinHashHit { distance, row_id } in hits.into_sorted() {
                    query_index_builder.append_value(query_index as i32);
                    distance_builder.append_value(distance.0);
                    row_id_builder.append_value(row_id);
                }
            }
            let batch = RecordBatch::try_new(
                result_schema,
                vec![
                    Arc::new(query_index_builder.finish()),
                    Arc::new(distance_builder.finish()),
                    Arc::new(row_id_builder.finish()),
                ],
            )?;
            metrics.baseline_metrics.record_output(batch.num_rows());
            DataFusionResult::Ok(batch)
        };

        let stream = stream::once(fut).finally(move || {
            metrics_clone.index_metrics.flush_io();
            metrics_clone
                .baseline_metrics
                .elapsed_compute()
                .add_duration(timer.elapsed());
            metrics_clone.baseline_metrics.done();
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream.boxed(),
        )))
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }
}
