// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Phrase execution node.

use std::sync::{Arc, OnceLock};

use arrow_schema::SchemaRef;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion_physical_expr::{Distribution, EquivalenceProperties, Partitioning, PhysicalExpr};
use futures::stream::{self};
use futures::{FutureExt, StreamExt};
use lance_core::{Error, Result, utils::tracing::StreamTracingExt};
use lance_select::RowAddrMask;
use lance_table::format::IndexMetadata;

use crate::Dataset;
use crate::io::exec::PreFilterSource;
use crate::io::exec::utils::{PreFilterMasks, build_prefilter};
use lance_index::prefilter::PreFilter;
use lance_index::scalar::inverted::query::{
    FtsSearchParams, PhraseQuery, try_collect_query_tokens,
};
use lance_index::scalar::inverted::{
    DocumentGranularity, MemBM25Scorer, build_global_bm25_scorer, fts_schema,
};
use tracing::instrument;
use uuid::Uuid;

use super::corpus::SharedFtsScorer;
use super::matches::scored_documents_batch;
use super::query_tokens::{TokenizedQuery, fmt_tokenized_query, record_tokenized_query};
use super::segments::{open_fts_segments, search_segments, segments_the_prefilter_may_match};
use super::selection::{FtsIndexMetrics, FtsSegmentSelection};

/// Indexed phrase search with an explicit document boundary and candidate scope.
///
/// Codecs must preserve the resolved granularity independently of the logical
/// query, as well as the kind of segment selection, masks and scoring override.
/// The example reuses value objects and child plans; a transport codec supplies
/// their decoded equivalents from the same pinned dataset snapshot.
///
/// ```
/// # use lance::{io::exec::fts::PhraseQueryExec, Error, Result};
/// # fn rebuild(exec: &PhraseQueryExec) -> Result<PhraseQueryExec> {
/// if exec.has_shared_scorer() {
///     return Err(Error::invalid_input("reconstruct the enclosing corpus owner"));
/// }
/// let mut rebuilt = if let Some(segments) = exec.preset_segments() {
///     PhraseQueryExec::new_with_segments_and_document_granularity(
///         exec.dataset().clone(), exec.query().clone(), exec.params().clone(),
///         exec.prefilter_source().clone(), segments.to_vec(), exec.document_granularity(),
///     )
/// } else if let Some(uuids) = exec.explicit_segment_uuids() {
///     PhraseQueryExec::new_with_segment_uuids_and_document_granularity(
///         exec.dataset().clone(), exec.query().clone(), exec.params().clone(),
///         exec.prefilter_source().clone(), uuids, exec.document_granularity(),
///     )
/// } else {
///     PhraseQueryExec::new_with_document_granularity(
///         exec.dataset().clone(), exec.query().clone(), exec.params().clone(),
///         exec.prefilter_source().clone(), exec.document_granularity(),
///     )
/// }.with_external_mask(exec.external_mask().cloned());
/// if let Some(mask) = exec.overlay_block() {
///     rebuilt = rebuilt.with_overlay_block(mask.clone());
/// }
/// if let Some(scorer) = exec.base_scorer() {
///     rebuilt = rebuilt.with_base_scorer(scorer.clone());
/// }
/// # Ok(rebuilt)
/// # }
/// ```
#[derive(Debug)]
pub struct PhraseQueryExec {
    dataset: Arc<Dataset>,
    query: PhraseQuery,
    tokenized_query: Arc<OnceLock<TokenizedQuery>>,
    params: FtsSearchParams,
    prefilter_source: PreFilterSource,
    /// Optional override for the BM25 scorer normally built locally inside
    /// `execute()`. See [`MatchQueryExec::with_base_scorer`].
    base_scorer: Option<Arc<MemBM25Scorer>>,
    /// Corpus-wide scorer published by the flat branch of a mixed search.
    shared_scorer: Option<Arc<SharedFtsScorer>>,
    segment_selection: FtsSegmentSelection,
    /// Rows whose indexed values were superseded by newer data overlays.
    overlay_block: Option<RowAddrMask>,
    document_granularity: DocumentGranularity,
    schema: SchemaRef,
    /// Optional external row-address mask combined (logical AND) with the BM25
    /// prefilter so only masked rows are scored (see [`MatchQueryExec::with_external_mask`]).
    external_mask: Option<Arc<RowAddrMask>>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl DisplayAs for PhraseQueryExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "PhraseQuery: column={}, query={}",
                    self.query.column.as_deref().unwrap_or_default(),
                    self.query.terms
                )?;
                fmt_tokenized_query(&self.tokenized_query, ", ", f)
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "PhraseQuery\ncolumn={}\nquery={}",
                    self.query.column.as_deref().unwrap_or_default(),
                    self.query.terms
                )?;
                fmt_tokenized_query(&self.tokenized_query, "\n", f)
            }
        }
    }
}

impl PhraseQueryExec {
    pub fn new(
        dataset: Arc<Dataset>,
        query: PhraseQuery,
        params: FtsSearchParams,
        prefilter_source: PreFilterSource,
    ) -> Result<Self> {
        let document_granularity = query.document_granularity.ok_or_else(|| {
            Error::invalid_input("PhraseQuery document granularity must be resolved".to_string())
        })?;
        Ok(Self::new_with_document_granularity(
            dataset,
            query,
            params,
            prefilter_source,
            document_granularity,
        ))
    }

    pub fn new_with_document_granularity(
        dataset: Arc<Dataset>,
        query: PhraseQuery,
        params: FtsSearchParams,
        prefilter_source: PreFilterSource,
        document_granularity: DocumentGranularity,
    ) -> Self {
        let schema = fts_schema(document_granularity);
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        let params = params.with_phrase_slop(Some(query.slop));

        Self {
            dataset,
            query,
            tokenized_query: Arc::new(OnceLock::new()),
            params,
            prefilter_source,
            base_scorer: None,
            shared_scorer: None,
            segment_selection: FtsSegmentSelection::AllCommitted,
            overlay_block: None,
            document_granularity,
            schema,
            external_mask: None,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// See [`super::MatchQueryExec::new_with_segments`].
    pub fn new_with_segments(
        dataset: Arc<Dataset>,
        query: PhraseQuery,
        params: FtsSearchParams,
        prefilter_source: PreFilterSource,
        segments: Vec<IndexMetadata>,
    ) -> Result<Self> {
        let document_granularity = query.document_granularity.ok_or_else(|| {
            Error::invalid_input("PhraseQuery document granularity must be resolved".to_string())
        })?;
        Ok(Self::new_with_segments_and_document_granularity(
            dataset,
            query,
            params,
            prefilter_source,
            segments,
            document_granularity,
        ))
    }

    pub fn new_with_segments_and_document_granularity(
        dataset: Arc<Dataset>,
        query: PhraseQuery,
        params: FtsSearchParams,
        prefilter_source: PreFilterSource,
        segments: Vec<IndexMetadata>,
        document_granularity: DocumentGranularity,
    ) -> Self {
        let schema = fts_schema(document_granularity);
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        let params = params.with_phrase_slop(Some(query.slop));

        Self {
            dataset,
            query,
            tokenized_query: Arc::new(OnceLock::new()),
            params,
            prefilter_source,
            base_scorer: None,
            shared_scorer: None,
            segment_selection: FtsSegmentSelection::ExactResolved(Arc::from(segments)),
            overlay_block: None,
            external_mask: None,
            document_granularity,
            schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// Construct a `PhraseQueryExec` bound to an exact ordered set of committed
    /// FTS segment UUIDs.
    ///
    /// The UUIDs are resolved from this exec's dataset snapshot when the output
    /// stream is polled. Duplicate UUIDs are removed while preserving their
    /// first-occurrence order. Resolution fails if the list is empty or any UUID
    /// is not committed for the query column.
    pub fn new_with_segment_uuids(
        dataset: Arc<Dataset>,
        query: PhraseQuery,
        params: FtsSearchParams,
        prefilter_source: PreFilterSource,
        segment_uuids: Vec<Uuid>,
    ) -> Result<Self> {
        let document_granularity = query.document_granularity.ok_or_else(|| {
            Error::invalid_input("PhraseQuery document granularity must be resolved".to_string())
        })?;
        Ok(Self::new_with_segment_uuids_and_document_granularity(
            dataset,
            query,
            params,
            prefilter_source,
            segment_uuids,
            document_granularity,
        ))
    }

    /// Bind an exact ordered UUID selection with an already resolved document
    /// granularity, retaining the logical query unchanged.
    ///
    /// This supports reconstruction when the logical query leaves granularity
    /// unset. UUID resolution remains deferred until execution, with the same
    /// validation as [`Self::new_with_segment_uuids`]. See [`Self`] for an example.
    pub fn new_with_segment_uuids_and_document_granularity(
        dataset: Arc<Dataset>,
        query: PhraseQuery,
        params: FtsSearchParams,
        prefilter_source: PreFilterSource,
        segment_uuids: Vec<Uuid>,
        document_granularity: DocumentGranularity,
    ) -> Self {
        let schema = fts_schema(document_granularity);
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        let params = params.with_phrase_slop(Some(query.slop));

        Self {
            dataset,
            query,
            tokenized_query: Arc::new(OnceLock::new()),
            params,
            prefilter_source,
            base_scorer: None,
            shared_scorer: None,
            segment_selection: FtsSegmentSelection::exact_uuids(segment_uuids),
            overlay_block: None,
            document_granularity,
            schema,
            external_mask: None,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// Override the local BM25 scorer; see
    /// [`super::MatchQueryExec::with_base_scorer`].
    pub fn with_base_scorer(mut self, scorer: Arc<MemBM25Scorer>) -> Self {
        self.base_scorer = Some(scorer);
        self
    }

    pub(crate) fn with_shared_scorer(mut self, scorer: Arc<SharedFtsScorer>) -> Self {
        self.shared_scorer = Some(scorer);
        self
    }

    pub(super) fn uses_shared_scorer(&self, scorer: &Arc<SharedFtsScorer>) -> bool {
        self.shared_scorer
            .as_ref()
            .is_some_and(|shared| Arc::ptr_eq(shared, scorer))
    }

    pub(super) fn rebuilt_with_shared_scorer(
        &self,
        scorer: Arc<SharedFtsScorer>,
    ) -> Arc<dyn ExecutionPlan> {
        Arc::new(Self {
            dataset: self.dataset.clone(),
            query: self.query.clone(),
            tokenized_query: self.tokenized_query.clone(),
            params: self.params.clone(),
            prefilter_source: self.prefilter_source.clone(),
            base_scorer: self.base_scorer.clone(),
            shared_scorer: Some(scorer),
            segment_selection: self.segment_selection.clone(),
            overlay_block: self.overlay_block.clone(),
            document_granularity: self.document_granularity,
            schema: self.schema.clone(),
            external_mask: self.external_mask.clone(),
            properties: self.properties.clone(),
            metrics: self.metrics.clone(),
        })
    }

    /// Exclude rows whose indexed text was superseded by a newer data overlay.
    /// The mask uses the index's row-id domain, including stable row IDs when
    /// enabled. Preserve it separately from the candidate prefilter.
    pub fn with_overlay_block(mut self, overlay_block: RowAddrMask) -> Self {
        self.overlay_block = Some(overlay_block);
        self
    }

    /// See [`super::MatchQueryExec::with_external_mask`].
    pub fn with_external_mask(mut self, mask: Option<Arc<RowAddrMask>>) -> Self {
        self.external_mask = mask;
        self
    }

    /// Whether reconstruction must reconnect this leaf to its enclosing corpus
    /// owner. Process-local scorer coordination is not exported.
    pub fn has_shared_scorer(&self) -> bool {
        self.shared_scorer.is_some()
    }

    /// Resolved document boundary, independent of the logical query option.
    /// See [`Self`] for a reconstruction example.
    pub fn document_granularity(&self) -> DocumentGranularity {
        self.document_granularity
    }

    pub fn query(&self) -> &PhraseQuery {
        &self.query
    }

    pub fn params(&self) -> &FtsSearchParams {
        &self.params
    }

    pub fn dataset(&self) -> &Arc<Dataset> {
        &self.dataset
    }

    pub fn prefilter_source(&self) -> &PreFilterSource {
        &self.prefilter_source
    }

    pub fn base_scorer(&self) -> Option<&Arc<MemBM25Scorer>> {
        self.base_scorer.as_ref()
    }

    /// Overlay exclusion in the index's row-id domain; `None` adds no exclusion.
    pub fn overlay_block(&self) -> Option<&RowAddrMask> {
        self.overlay_block.as_ref()
    }

    /// Additional candidate restriction in the index's row-id domain, or `None`.
    /// This mask does not narrow the statistics corpus.
    pub fn external_mask(&self) -> Option<&Arc<RowAddrMask>> {
        self.external_mask.as_ref()
    }

    pub fn preset_segments(&self) -> Option<&[IndexMetadata]> {
        self.segment_selection.preset_segments()
    }

    /// Return the ordered segment UUIDs for an explicit selection.
    ///
    /// Returns `None` when this exec searches all committed segments. UUID-based
    /// selections omit duplicates while preserving first-occurrence order.
    /// Pre-resolved selections preserve the supplied metadata order.
    pub fn explicit_segment_uuids(&self) -> Option<Vec<Uuid>> {
        self.segment_selection.explicit_segment_uuids()
    }
}

impl ExecutionPlan for PhraseQueryExec {
    fn name(&self) -> &str {
        "PhraseQueryExec"
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.prefilter_source.execution_plan().into_iter().collect()
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        // Prefilter inputs must be a single partition
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
        let plan = match children.len() {
            0 => {
                if !matches!(self.prefilter_source, PreFilterSource::None) {
                    return Err(DataFusionError::Internal(
                        "Unexpected prefilter source".to_string(),
                    ));
                }
                Self {
                    dataset: self.dataset.clone(),
                    query: self.query.clone(),
                    tokenized_query: self.tokenized_query.clone(),
                    params: self.params.clone(),
                    prefilter_source: PreFilterSource::None,
                    base_scorer: self.base_scorer.clone(),
                    shared_scorer: self.shared_scorer.clone(),
                    segment_selection: self.segment_selection.clone(),
                    overlay_block: self.overlay_block.clone(),
                    document_granularity: self.document_granularity,
                    schema: self.schema.clone(),
                    external_mask: self.external_mask.clone(),
                    properties: self.properties.clone(),
                    metrics: ExecutionPlanMetricsSet::new(),
                }
            }
            1 => {
                let src = children.pop().unwrap();
                let prefilter_source = self.prefilter_source.with_execution_plan(src)?;
                Self {
                    dataset: self.dataset.clone(),
                    query: self.query.clone(),
                    tokenized_query: self.tokenized_query.clone(),
                    params: self.params.clone(),
                    prefilter_source,
                    base_scorer: self.base_scorer.clone(),
                    shared_scorer: self.shared_scorer.clone(),
                    segment_selection: self.segment_selection.clone(),
                    overlay_block: self.overlay_block.clone(),
                    document_granularity: self.document_granularity,
                    schema: self.schema.clone(),
                    external_mask: self.external_mask.clone(),
                    properties: self.properties.clone(),
                    metrics: ExecutionPlanMetricsSet::new(),
                }
            }
            _ => {
                return Err(DataFusionError::Internal(
                    "Unexpected number of children".to_string(),
                ));
            }
        };
        Ok(Arc::new(plan))
    }

    #[instrument(name = "phrase_query_exec", level = "debug", skip_all)]
    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let query = self.query.clone();
        let tokenized_query = self.tokenized_query.clone();
        let params = self.params.clone();
        let ds = self.dataset.clone();
        let prefilter_source = self.prefilter_source.clone();
        let external_mask = self.external_mask.clone();
        let preset_base_scorer = self.base_scorer.clone();
        let shared_scorer = self.shared_scorer.clone();
        let segment_selection = self.segment_selection.clone();
        let overlay_block = self.overlay_block.clone();
        let document_granularity = self.document_granularity;
        let schema = self.schema.clone();
        let metrics_set = self.metrics.clone();
        let metrics = Arc::new(FtsIndexMetrics::new(&self.metrics, partition));
        let stream = stream::once(async move {
            let _timer = metrics.baseline_metrics.elapsed_compute().timer();
            let column = query.column.ok_or(DataFusionError::Execution(format!(
                "column not set for PhraseQuery {}",
                query.terms
            )))?;
            let segments = segment_selection
                .resolve(
                    &ds,
                    &column,
                    document_granularity,
                    &metrics.segment_bind_duration,
                )
                .await?;
            let address_domain = !ds.manifest.uses_stable_row_ids();
            let ds_for_scope = ds.clone();
            let indices =
                open_fts_segments(&ds, &column, &segments, &metrics.index_metrics).await?;

            let mut pre_filter = build_prefilter(
                context.clone(),
                partition,
                &prefilter_source,
                ds,
                &segments,
                PreFilterMasks {
                    overlay_block,
                    external_mask,
                },
                &metrics_set,
            )?;
            let deleted_fragments =
                indices
                    .iter()
                    .fold(roaring::RoaringBitmap::new(), |mut deleted, index| {
                        deleted |= index.deleted_fragments().clone();
                        deleted
                    });
            if !deleted_fragments.is_empty() {
                Arc::get_mut(&mut pre_filter)
                    .expect("prefilter just created")
                    .set_deleted_fragments(deleted_fragments);
            }
            metrics
                .record_parts_searched(indices.iter().map(|index| index.partition_count()).sum());

            let first_index = indices.first().ok_or(DataFusionError::Execution(format!(
                "FTS index for column {} has no segments",
                column
            )))?;
            let mut tokenizer = first_index.tokenizer();
            let tokens = try_collect_query_tokens(&query.terms, &mut tokenizer)?;
            record_tokenized_query(&tokenized_query, &tokens);
            let base_scorer = match (preset_base_scorer, shared_scorer) {
                (Some(scorer), _) => scorer,
                (None, Some(shared_scorer)) => shared_scorer.wait().await?,
                (None, None) => {
                    let scorer_start = std::time::Instant::now();
                    let scorer = Arc::new(
                        build_global_bm25_scorer(
                            &indices,
                            &tokens,
                            &params,
                            Some(metrics.as_ref()),
                        )
                        .boxed()
                        .await?,
                    );
                    metrics.record_scorer_build(scorer_start.elapsed());
                    scorer
                }
            };

            pre_filter.wait_for_ready().await?;
            let indices = segments_the_prefilter_may_match(
                &ds_for_scope,
                &indices,
                &segments,
                pre_filter.as_ref(),
                address_domain,
                &metrics,
            )
            .await;
            let tokens = Arc::new(tokens);
            let params = Arc::new(params);
            let documents = search_segments(
                &indices,
                tokens,
                params,
                lance_index::scalar::inverted::query::Operator::And,
                pre_filter,
                metrics.clone(),
                base_scorer,
                None,
            )
            .await?;
            metrics.baseline_metrics.record_output(documents.len());
            let batch = scored_documents_batch(schema, documents)?;
            Ok::<_, DataFusionError>(batch)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream.stream_in_current_span().boxed(),
        )))
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
    use std::sync::Arc;

    use datafusion::execution::TaskContext;
    use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
    use datafusion::physical_plan::{ChildrenPropertiesMode, ReplaceChildrenOptions};
    use lance_index::scalar::inverted::DocumentGranularity;
    use lance_index::scalar::inverted::query::{FtsSearchParams, PhraseQuery};
    use uuid::Uuid;

    use crate::io::exec::PreFilterSource;
    use crate::io::exec::fts::FTS_SEGMENT_BIND_DURATION_METRIC;
    use crate::io::exec::fts::tests::{
        assert_execution_error, create_segment_selection_fixture, execute_results, execute_row_ids,
        expected_row_ids, metric_value, segment_uuid_for_fragment,
    };

    #[tokio::test]
    async fn test_phrase_query_exec_segment_selection() {
        let (dataset, segments, fragment_ids) = create_segment_selection_fixture().await;
        let query = PhraseQuery::new("quick brown".to_string())
            .with_column(Some("text".to_string()))
            .with_document_granularity(DocumentGranularity::Row);
        let params = FtsSearchParams::default().with_limit(Some(20));
        let committed_uuids = segments
            .iter()
            .map(|segment| segment.uuid)
            .collect::<Vec<_>>();

        let all_committed = PhraseQueryExec::new(
            dataset.clone(),
            query.clone(),
            params.clone(),
            PreFilterSource::None,
        )
        .unwrap();
        assert!(all_committed.preset_segments().is_none());
        assert!(all_committed.explicit_segment_uuids().is_none());
        let all_results = execute_results(&all_committed).await.unwrap();
        assert_eq!(
            all_results
                .iter()
                .map(|(row_id, _)| *row_id)
                .collect::<Vec<_>>(),
            expected_row_ids(&fragment_ids)
        );
        assert_eq!(
            metric_value(&all_committed, FTS_SEGMENT_BIND_DURATION_METRIC),
            0
        );

        let exact_resolved = PhraseQueryExec::new_with_segments(
            dataset.clone(),
            query.clone(),
            params.clone(),
            PreFilterSource::None,
            segments.clone(),
        )
        .unwrap();
        assert_eq!(exact_resolved.preset_segments(), Some(segments.as_slice()));
        assert_eq!(
            exact_resolved.explicit_segment_uuids(),
            Some(committed_uuids)
        );
        assert_eq!(execute_results(&exact_resolved).await.unwrap(), all_results);
        assert_eq!(
            metric_value(&exact_resolved, FTS_SEGMENT_BIND_DURATION_METRIC),
            0
        );

        let selected_fragment = fragment_ids[1];
        let selected_uuid = segment_uuid_for_fragment(&segments, selected_fragment);
        let unpolled = PhraseQueryExec::new_with_segment_uuids(
            dataset.clone(),
            query.clone(),
            params.clone(),
            PreFilterSource::None,
            vec![selected_uuid],
        )
        .unwrap();
        drop(
            unpolled
                .execute(0, Arc::new(TaskContext::default()))
                .unwrap(),
        );
        assert_eq!(
            metric_value(&unpolled, FTS_SEGMENT_BIND_DURATION_METRIC),
            0,
            "UUID binding should not start until the output stream is polled"
        );

        let exact_uuids = PhraseQueryExec::new_with_segment_uuids(
            dataset.clone(),
            query.clone(),
            params.clone(),
            PreFilterSource::None,
            vec![selected_uuid],
        )
        .unwrap();
        assert!(exact_uuids.preset_segments().is_none());
        assert_eq!(
            exact_uuids.explicit_segment_uuids(),
            Some(vec![selected_uuid])
        );
        assert_eq!(
            execute_row_ids(&exact_uuids).await.unwrap(),
            expected_row_ids(&[selected_fragment])
        );
        assert!(
            metric_value(&exact_uuids, FTS_SEGMENT_BIND_DURATION_METRIC) > 0,
            "successful UUID binding should record a duration"
        );

        let input_uuids = vec![
            segment_uuid_for_fragment(&segments, fragment_ids[2]),
            segment_uuid_for_fragment(&segments, fragment_ids[0]),
            segment_uuid_for_fragment(&segments, fragment_ids[2]),
        ];
        let deduplicated_uuids = input_uuids[..2].to_vec();
        let ordered_plan = Arc::new(
            PhraseQueryExec::new_with_segment_uuids(
                dataset.clone(),
                query.clone(),
                params.clone(),
                PreFilterSource::None,
                input_uuids,
            )
            .unwrap(),
        )
        .replace_children(
            vec![],
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
        .unwrap();
        let rewritten = ordered_plan.downcast_ref::<PhraseQueryExec>().unwrap();
        assert_eq!(
            rewritten.explicit_segment_uuids(),
            Some(deduplicated_uuids.clone())
        );
        assert_eq!(
            execute_row_ids(rewritten).await.unwrap(),
            expected_row_ids(&[fragment_ids[2], fragment_ids[0]])
        );
        let resolver_metrics_set = ExecutionPlanMetricsSet::new();
        let resolver_metrics = super::FtsIndexMetrics::new(&resolver_metrics_set, 0);
        let resolved = rewritten
            .segment_selection
            .resolve(
                &dataset,
                "text",
                DocumentGranularity::Row,
                &resolver_metrics.segment_bind_duration,
            )
            .await
            .unwrap();
        assert_eq!(
            resolved
                .iter()
                .map(|segment| segment.uuid)
                .collect::<Vec<_>>(),
            deduplicated_uuids
        );

        let empty = PhraseQueryExec::new_with_segment_uuids(
            dataset.clone(),
            query.clone(),
            params.clone(),
            PreFilterSource::None,
            vec![],
        )
        .unwrap();
        assert_execution_error(
            execute_row_ids(&empty).await.unwrap_err(),
            "requires at least one segment UUID",
        );

        let missing_uuid = Uuid::new_v4();
        let missing = PhraseQueryExec::new_with_segment_uuids(
            dataset.clone(),
            query,
            params.clone(),
            PreFilterSource::None,
            vec![missing_uuid],
        )
        .unwrap();
        assert_execution_error(
            execute_row_ids(&missing).await.unwrap_err(),
            &missing_uuid.to_string(),
        );

        let wrong_column = PhraseQueryExec::new_with_segment_uuids(
            dataset,
            PhraseQuery::new("quick brown".to_string())
                .with_column(Some("other".to_string()))
                .with_document_granularity(DocumentGranularity::Row),
            params,
            PreFilterSource::None,
            vec![selected_uuid],
        )
        .unwrap();
        assert_execution_error(
            execute_row_ids(&wrong_column).await.unwrap_err(),
            "no Inverted index found",
        );
    }
}
