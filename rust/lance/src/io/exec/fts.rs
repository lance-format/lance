// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{AsArray, BooleanBuilder};
use arrow::datatypes::{Float32Type, UInt64Type};
use arrow_array::{Array, BooleanArray, Float32Array, OffsetSizeTrait, RecordBatch, UInt64Array};
use arrow_schema::{DataType, SchemaRef};
use datafusion::common::{NullEquality, Statistics};
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::empty::EmptyExec;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::{Distribution, EquivalenceProperties, Partitioning};
use datafusion_physical_plan::joins::{HashJoinExec, PartitionMode};
use datafusion_physical_plan::metrics::{BaselineMetrics, Count};
use futures::future::try_join_all;
use futures::stream::{self};
use futures::{FutureExt, StreamExt, TryStreamExt};
use itertools::Itertools;
use lance_core::{
    Error, ROW_ID, Result,
    utils::{tokio::get_num_compute_intensive_cpus, tracing::StreamTracingExt},
};
use lance_datafusion::utils::{ExecutionPlanMetricsSetExt, MetricsExt, PARTITIONS_SEARCHED_METRIC};
use lance_table::format::IndexMetadata;

use super::PreFilterSource;
use super::utils::{IndexMetrics, build_prefilter};
use crate::index::scalar::inverted::{load_segment_details, load_segments};
use crate::{Dataset, index::DatasetIndexInternalExt};
use lance_index::metrics::{
    AND_CANDIDATES_PRUNED_BEFORE_RETURN_METRIC, AND_CANDIDATES_SEEN_METRIC, AND_FULL_SCORES_METRIC,
    FREQS_COLLECTED_METRIC, FTS_BLOCK_METADATA_ROWS_METRIC, FTS_BLOCKS_PRUNED_METRIC,
    FTS_PAYLOAD_BLOCKS_READ_METRIC, FTS_PAYLOAD_READ_RANGES_METRIC,
    FTS_POSITION_BLOCKS_READ_METRIC, FTS_REFINEMENT_BATCHES_METRIC, MetricsCollector,
};
use lance_index::scalar::inverted::builder::ScoredDoc;
use lance_index::scalar::inverted::builder::document_input;
use lance_index::scalar::inverted::document_tokenizer::{DocType, JsonTokenizer, LanceTokenizer};
use lance_index::scalar::inverted::query::{
    BoostQuery, FtsSearchParams, MatchQuery, PhraseQuery, Tokens, collect_query_tokens,
    has_query_token,
};
use lance_index::scalar::inverted::tokenizer::document_tokenizer::TextTokenizer;
use lance_index::scalar::inverted::{
    FTS_SCHEMA, InvertedIndex, MemBM25Scorer, SCORE_COL, build_global_bm25_scorer,
    flat_bm25_search_stream_with_metrics,
};
use lance_index::{prefilter::PreFilter, scalar::inverted::query::BooleanQuery};
use lance_tokenizer::{SimpleTokenizer, TextAnalyzer};
use tracing::instrument;

/// Open one FTS segment as an [`InvertedIndex`].
async fn open_fts_segment(
    dataset: &Dataset,
    column: &str,
    segment: &IndexMetadata,
    metrics: &IndexMetrics,
) -> Result<Arc<InvertedIndex>> {
    let index = dataset
        .open_scalar_index(column, &segment.uuid, metrics)
        .await?;
    let inverted = index
        .as_any()
        .downcast_ref::<InvertedIndex>()
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "Index for column {} and segment {} is not an inverted index",
                column, segment.uuid
            ))
        })?;
    Ok(Arc::new(inverted.clone()))
}

/// Open all committed FTS segments for a column.
///
/// Exact multi-segment BM25 still needs every segment's local corpus statistics, so the
/// current correctness-first path opens each committed segment before scoring.
async fn open_fts_segments(
    dataset: &Dataset,
    column: &str,
    segments: &[IndexMetadata],
    metrics: &IndexMetrics,
) -> Result<Vec<Arc<InvertedIndex>>> {
    try_join_all(
        segments
            .iter()
            .map(|segment| open_fts_segment(dataset, column, segment, metrics)),
    )
    .await
}

async fn search_segments(
    indices: &[Arc<InvertedIndex>],
    tokens: Arc<Tokens>,
    params: Arc<FtsSearchParams>,
    operator: lance_index::scalar::inverted::query::Operator,
    pre_filter: Arc<dyn PreFilter>,
    metrics: Arc<FtsIndexMetrics>,
    base_scorer: Arc<MemBM25Scorer>,
) -> Result<(Vec<u64>, Vec<f32>)> {
    let limit = params.limit.unwrap_or(usize::MAX);
    let mut candidates = std::collections::BinaryHeap::new();
    let searches = indices
        .iter()
        .map(|index| {
            let index = Arc::clone(index);
            let tokens = tokens.clone();
            let params = params.clone();
            let pre_filter = pre_filter.clone();
            let metrics = metrics.clone();
            let base_scorer = base_scorer.clone();
            async move {
                index
                    .bm25_search(
                        tokens,
                        params,
                        operator,
                        pre_filter,
                        metrics,
                        Some(base_scorer.as_ref()),
                    )
                    .await
            }
        })
        .collect::<Vec<_>>();
    let searches = stream::iter(searches).buffer_unordered(get_num_compute_intensive_cpus());
    let mut searches = searches;

    while let Some((doc_ids, scores)) = searches.try_next().await? {
        for (row_id, score) in doc_ids.into_iter().zip(scores.into_iter()) {
            if candidates.len() < limit {
                candidates.push(std::cmp::Reverse(ScoredDoc::new(row_id, score)));
            } else if candidates.peek().unwrap().0.score.0 < score {
                candidates.pop();
                candidates.push(std::cmp::Reverse(ScoredDoc::new(row_id, score)));
            }
        }
    }

    Ok(candidates
        .into_sorted_vec()
        .into_iter()
        .map(|std::cmp::Reverse(doc)| (doc.row_id, doc.score.0))
        .unzip())
}

/// Fall back to the default simple tokenizer when no on-disk FTS segment exists.
fn default_text_tokenizer() -> Box<dyn LanceTokenizer> {
    Box::new(TextTokenizer::new(
        TextAnalyzer::builder(SimpleTokenizer::default()).build(),
    ))
}

pub struct FtsIndexMetrics {
    index_metrics: IndexMetrics,
    partitions_searched: Count,
    and_candidates_seen: Count,
    and_candidates_pruned_before_return: Count,
    and_full_scores: Count,
    freqs_collected: Count,
    fts_block_metadata_rows: Count,
    fts_payload_blocks_read: Count,
    fts_position_blocks_read: Count,
    fts_blocks_pruned: Count,
    fts_payload_read_ranges: Count,
    fts_refinement_batches: Count,
    baseline_metrics: BaselineMetrics,
}

impl FtsIndexMetrics {
    pub fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            index_metrics: IndexMetrics::new(metrics, partition),
            partitions_searched: metrics.new_count(PARTITIONS_SEARCHED_METRIC, partition),
            and_candidates_seen: metrics.new_count(AND_CANDIDATES_SEEN_METRIC, partition),
            and_candidates_pruned_before_return: metrics
                .new_count(AND_CANDIDATES_PRUNED_BEFORE_RETURN_METRIC, partition),
            and_full_scores: metrics.new_count(AND_FULL_SCORES_METRIC, partition),
            freqs_collected: metrics.new_count(FREQS_COLLECTED_METRIC, partition),
            fts_block_metadata_rows: metrics.new_count(FTS_BLOCK_METADATA_ROWS_METRIC, partition),
            fts_payload_blocks_read: metrics.new_count(FTS_PAYLOAD_BLOCKS_READ_METRIC, partition),
            fts_position_blocks_read: metrics.new_count(FTS_POSITION_BLOCKS_READ_METRIC, partition),
            fts_blocks_pruned: metrics.new_count(FTS_BLOCKS_PRUNED_METRIC, partition),
            fts_payload_read_ranges: metrics.new_count(FTS_PAYLOAD_READ_RANGES_METRIC, partition),
            fts_refinement_batches: metrics.new_count(FTS_REFINEMENT_BATCHES_METRIC, partition),
            baseline_metrics: BaselineMetrics::new(metrics, partition),
        }
    }

    pub fn record_parts_searched(&self, num_parts: usize) {
        self.partitions_searched.add(num_parts);
    }
}

impl MetricsCollector for FtsIndexMetrics {
    fn record_parts_loaded(&self, num_parts: usize) {
        self.index_metrics.record_parts_loaded(num_parts);
    }

    fn record_index_loads(&self, num_indexes: usize) {
        self.index_metrics.record_index_loads(num_indexes);
    }

    fn record_comparisons(&self, num_comparisons: usize) {
        self.index_metrics.record_comparisons(num_comparisons);
    }

    fn record_and_candidates_seen(&self, num_candidates: usize) {
        self.and_candidates_seen.add(num_candidates);
    }

    fn record_and_candidates_pruned_before_return(&self, num_candidates: usize) {
        self.and_candidates_pruned_before_return.add(num_candidates);
    }

    fn record_and_full_scores(&self, num_scores: usize) {
        self.and_full_scores.add(num_scores);
    }

    fn record_freqs_collected(&self, num_collections: usize) {
        self.freqs_collected.add(num_collections);
    }

    fn record_fts_block_metadata_rows(&self, num_rows: usize) {
        self.fts_block_metadata_rows.add(num_rows);
    }

    fn record_fts_payload_blocks_read(&self, num_blocks: usize) {
        self.fts_payload_blocks_read.add(num_blocks);
    }

    fn record_fts_position_blocks_read(&self, num_blocks: usize) {
        self.fts_position_blocks_read.add(num_blocks);
    }

    fn record_fts_blocks_pruned(&self, num_blocks: usize) {
        self.fts_blocks_pruned.add(num_blocks);
    }

    fn record_fts_payload_read_ranges(&self, num_ranges: usize) {
        self.fts_payload_read_ranges.add(num_ranges);
    }

    fn record_fts_refinement_batches(&self, num_batches: usize) {
        self.fts_refinement_batches.add(num_batches);
    }
}

#[derive(Debug)]
pub struct MatchQueryExec {
    dataset: Arc<Dataset>,
    query: MatchQuery,
    params: FtsSearchParams,
    prefilter_source: PreFilterSource,
    /// When set, `execute()` skips `build_global_bm25_scorer` and threads this
    /// scorer down to `InvertedIndex::bm25_search`.
    base_scorer: Option<Arc<MemBM25Scorer>>,
    /// When set, `execute()` skips `load_segments` and searches exactly these
    /// segments.
    preset_segments: Option<Vec<IndexMetadata>>,

    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl DisplayAs for MatchQueryExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "MatchQuery: column={}, query={}",
                    self.query.column.as_deref().unwrap_or_default(),
                    self.query.terms
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "MatchQuery\ncolumn={}\nquery={}",
                    self.query.column.as_deref().unwrap_or_default(),
                    self.query.terms
                )
            }
        }
    }
}

impl MatchQueryExec {
    /// Merge the fuzzy fields from `query` into `params` so that the stored
    /// params reflect what BM25 stat collection and search will actually use.
    fn effective_params(query: &MatchQuery, params: FtsSearchParams) -> FtsSearchParams {
        params
            .with_fuzziness(query.fuzziness)
            .with_max_expansions(query.max_expansions)
            .with_prefix_length(query.prefix_length)
    }

    pub fn new(
        dataset: Arc<Dataset>,
        query: MatchQuery,
        params: FtsSearchParams,
        prefilter_source: PreFilterSource,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(FTS_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        let params = Self::effective_params(&query, params);
        Self {
            dataset,
            query,
            params,
            prefilter_source,
            base_scorer: None,
            preset_segments: None,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// Construct a `MatchQueryExec` bound to an explicit, pre-resolved set of
    /// FTS segments. Unlike [`Self::new`], `execute()` will not call
    /// [`load_segments`] — it will search exactly the segments supplied here.
    ///
    /// Useful when a caller has already enumerated segments and wants to scope
    /// this exec to a strict subset — for example, a distributed query that
    /// routes per-segment work across hosts, where each per-host leaf should
    /// only search its own assigned subset of the dataset's committed
    /// segments.
    pub fn new_with_segments(
        dataset: Arc<Dataset>,
        query: MatchQuery,
        params: FtsSearchParams,
        prefilter_source: PreFilterSource,
        segments: Vec<IndexMetadata>,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(FTS_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        let params = Self::effective_params(&query, params);
        Self {
            dataset,
            query,
            params,
            prefilter_source,
            base_scorer: None,
            preset_segments: Some(segments),
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// Override the BM25 scorer used by `execute()`. When set, the local
    /// `build_global_bm25_scorer` call is skipped and the supplied scorer is
    /// threaded down to `InvertedIndex::bm25_search`.
    ///
    /// The default path builds a scorer from the segments this exec searches,
    /// which is correct when those segments are the entire corpus. A caller
    /// would override that scorer to keep BM25 IDFs corpus-wide when the exec
    /// is searching only a subset — for example, a distributed query that
    /// routes per-segment work to multiple hosts and aggregates stats
    /// out-of-band, so each per-host leaf scores against the full corpus
    /// rather than its local segment subset. See [`build_global_bm25_scorer`]
    /// for constructing one.
    pub fn with_base_scorer(mut self, scorer: Arc<MemBM25Scorer>) -> Self {
        self.base_scorer = Some(scorer);
        self
    }

    pub fn query(&self) -> &MatchQuery {
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

    pub fn preset_segments(&self) -> Option<&[IndexMetadata]> {
        self.preset_segments.as_deref()
    }
}

impl ExecutionPlan for MatchQueryExec {
    fn name(&self) -> &str {
        "MatchQueryExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        match &self.prefilter_source {
            PreFilterSource::None => vec![],
            PreFilterSource::FilteredRowIds(src) => vec![&src],
            PreFilterSource::ScalarIndexQuery(src) => vec![&src],
        }
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        // Prefilter inputs must be a single partition
        self.children()
            .iter()
            .map(|_| Distribution::SinglePartition)
            .collect()
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
                    params: self.params.clone(),
                    prefilter_source: PreFilterSource::None,
                    base_scorer: self.base_scorer.clone(),
                    preset_segments: self.preset_segments.clone(),
                    properties: self.properties.clone(),
                    metrics: ExecutionPlanMetricsSet::new(),
                }
            }
            1 => {
                let src = children.pop().unwrap();
                let prefilter_source = match &self.prefilter_source {
                    PreFilterSource::FilteredRowIds(_) => {
                        PreFilterSource::FilteredRowIds(src.clone())
                    }
                    PreFilterSource::ScalarIndexQuery(_) => {
                        PreFilterSource::ScalarIndexQuery(src.clone())
                    }
                    PreFilterSource::None => {
                        return Err(DataFusionError::Internal(
                            "Unexpected prefilter source".to_string(),
                        ));
                    }
                };

                Self {
                    dataset: self.dataset.clone(),
                    query: self.query.clone(),
                    params: self.params.clone(),
                    prefilter_source,
                    base_scorer: self.base_scorer.clone(),
                    preset_segments: self.preset_segments.clone(),
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

    #[instrument(name = "match_query_exec", level = "debug", skip_all)]
    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let query = self.query.clone();
        let params = self.params.clone();
        let ds = self.dataset.clone();
        let prefilter_source = self.prefilter_source.clone();
        let preset_base_scorer = self.base_scorer.clone();
        let preset_segments = self.preset_segments.clone();
        let metrics = Arc::new(FtsIndexMetrics::new(&self.metrics, partition));
        let column = query.column.ok_or(DataFusionError::Execution(format!(
            "column not set for MatchQuery {}",
            query.terms
        )))?;
        let stream = stream::once(async move {
            let _timer = metrics.baseline_metrics.elapsed_compute().timer();
            let segments = match preset_segments {
                Some(segments) => segments,
                None => load_segments(&ds, &column)
                    .await?
                    .ok_or(DataFusionError::Execution(format!(
                        "No Inverted index found for column {}",
                        column,
                    )))?,
            };
            let _details = load_segment_details(&ds, &column, &segments).await?;
            let indices =
                open_fts_segments(&ds, &column, &segments, &metrics.index_metrics).await?;

            let mut pre_filter =
                build_prefilter(context.clone(), partition, &prefilter_source, ds, &segments)?;
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

            let is_fuzzy = matches!(query.fuzziness, Some(n) if n != 0);
            let first_index = indices.first().ok_or(DataFusionError::Execution(format!(
                "FTS index for column {} has no segments",
                column
            )))?;
            let mut tokenizer = match is_fuzzy {
                false => first_index.tokenizer(),
                true => {
                    let tokenizer = TextAnalyzer::from(SimpleTokenizer::default());
                    match first_index.tokenizer().doc_type() {
                        DocType::Text => {
                            Box::new(TextTokenizer::new(tokenizer)) as Box<dyn LanceTokenizer>
                        }
                        DocType::Json => {
                            Box::new(JsonTokenizer::new(tokenizer)) as Box<dyn LanceTokenizer>
                        }
                    }
                }
            };
            let tokens = collect_query_tokens(&query.terms, &mut tokenizer);
            let base_scorer = match preset_base_scorer {
                Some(scorer) => scorer,
                None => Arc::new(
                    build_global_bm25_scorer(&indices, &tokens, &params)
                        .boxed()
                        .await?,
                ),
            };

            pre_filter.wait_for_ready().await?;
            let tokens = Arc::new(tokens);
            let params = Arc::new(params);
            let (doc_ids, mut scores) = search_segments(
                &indices,
                tokens,
                params,
                query.operator,
                pre_filter,
                metrics.clone(),
                base_scorer,
            )
            .await?;
            scores.iter_mut().for_each(|s| {
                *s *= query.boost;
            });
            metrics.baseline_metrics.record_output(doc_ids.len());

            let batch = RecordBatch::try_new(
                FTS_SCHEMA.clone(),
                vec![
                    Arc::new(UInt64Array::from(doc_ids)),
                    Arc::new(Float32Array::from(scores)),
                ],
            )?;
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

/// Filters the input, removing rows that do not share tokens with the query
#[derive(Debug)]
pub struct FlatMatchFilterExec {
    dataset: Arc<Dataset>,
    input: Arc<dyn ExecutionPlan>,
    query: MatchQuery,
    params: FtsSearchParams,
    /// Optional pre-resolved segment list. See
    /// [`MatchQueryExec::new_with_segments`]. `FlatMatchFilterExec` only
    /// uses the first segment's tokenizer, but the full list is preserved so
    /// the field round-trips through `with_new_children`.
    preset_segments: Option<Vec<IndexMetadata>>,

    metrics: ExecutionPlanMetricsSet,
}

impl DisplayAs for FlatMatchFilterExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "FlatMatchFilter: column={}, query={}",
                    self.query.column.as_deref().unwrap_or_default(),
                    self.query.terms
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "FlatMatchFilter\ncolumn={}\nquery={}",
                    self.query.column.as_deref().unwrap_or_default(),
                    self.query.terms
                )
            }
        }
    }
}

impl FlatMatchFilterExec {
    async fn load_tokenizer(
        dataset: &Dataset,
        column: &str,
        metrics: &IndexMetrics,
    ) -> DataFusionResult<Box<dyn LanceTokenizer>> {
        if let Some(segments) = load_segments(dataset, column).await? {
            let index_meta = segments.first().ok_or_else(|| {
                DataFusionError::Execution(format!(
                    "FTS index for column {} has no segments",
                    column
                ))
            })?;
            return Ok(open_fts_segment(dataset, column, index_meta, metrics)
                .await?
                .tokenizer());
        }
        Ok(default_text_tokenizer())
    }

    async fn load_tokenizer_from_preset_segments(
        dataset: &Dataset,
        column: &str,
        segments: &[IndexMetadata],
        metrics: &IndexMetrics,
    ) -> DataFusionResult<Box<dyn LanceTokenizer>> {
        let index_meta = segments.first().ok_or_else(|| {
            DataFusionError::Execution(format!("FTS index for column {} has no segments", column))
        })?;
        Ok(open_fts_segment(dataset, column, index_meta, metrics)
            .await?
            .tokenizer())
    }

    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        dataset: Arc<Dataset>,
        query: MatchQuery,
        params: FtsSearchParams,
    ) -> Self {
        Self {
            dataset,
            input,
            query,
            params,
            preset_segments: None,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// See [`MatchQueryExec::new_with_segments`]. `FlatMatchFilterExec`
    /// uses the first segment's tokenizer; the rest are kept for caller-side
    /// bookkeeping.
    pub fn new_with_segments(
        input: Arc<dyn ExecutionPlan>,
        dataset: Arc<Dataset>,
        query: MatchQuery,
        params: FtsSearchParams,
        segments: Vec<IndexMetadata>,
    ) -> Self {
        Self {
            dataset,
            input,
            query,
            params,
            preset_segments: Some(segments),
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    pub fn query(&self) -> &MatchQuery {
        &self.query
    }

    pub fn params(&self) -> &FtsSearchParams {
        &self.params
    }

    pub fn dataset(&self) -> &Arc<Dataset> {
        &self.dataset
    }

    pub fn preset_segments(&self) -> Option<&[IndexMetadata]> {
        self.preset_segments.as_deref()
    }

    fn find_matches<O: OffsetSizeTrait>(
        text_col: &dyn Array,
        tokenizer: &mut Box<dyn LanceTokenizer>,
        query_tokens: &Tokens,
    ) -> BooleanArray {
        let text_col = text_col.as_string::<O>();
        let mut predicate = BooleanBuilder::with_capacity(text_col.len());
        for idx in 0..text_col.len() {
            let value = text_col.value(idx);
            predicate.append_value(has_query_token(value, tokenizer, query_tokens));
        }
        predicate.finish()
    }

    async fn build_filter_stream(
        input: SendableRecordBatchStream,
        partition: usize,
        schema: SchemaRef,
        dataset: Arc<Dataset>,
        query: MatchQuery,
        preset_segments: Option<Vec<IndexMetadata>>,
        metrics_set: ExecutionPlanMetricsSet,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let metrics = Arc::new(FtsIndexMetrics::new(&metrics_set, partition));
        let column = query
            .column
            .clone()
            .ok_or(DataFusionError::Execution(format!(
                "column not set for MatchQuery {}",
                query.terms
            )))?;
        let mut tokenizer = match preset_segments {
            Some(segments) => {
                Self::load_tokenizer_from_preset_segments(
                    &dataset,
                    &column,
                    &segments,
                    &metrics.index_metrics,
                )
                .await?
            }
            None => Self::load_tokenizer(&dataset, &column, &metrics.index_metrics).await?,
        };
        let query_tokens = Arc::new(collect_query_tokens(&query.terms, &mut tokenizer));

        let baseline = BaselineMetrics::new(&metrics_set, partition);
        let elapsed_compute = baseline.elapsed_compute().clone();
        let stream = input.then(move |batch_result| {
            let column = column.clone();
            let query_tokens = query_tokens.clone();
            let mut tokenizer = tokenizer.box_clone();
            let elapsed_compute = elapsed_compute.clone();
            async move {
                let batch = batch_result?;
                let _t = elapsed_compute.timer();
                let text_column = batch.column_by_name(&column).ok_or_else(|| {
                    DataFusionError::Execution(format!("Column {} not found in batch", column,))
                })?;
                let predicate = match text_column.data_type() {
                    DataType::Utf8 => {
                        Self::find_matches::<i32>(text_column, &mut tokenizer, &query_tokens)
                    }
                    DataType::LargeUtf8 => {
                        Self::find_matches::<i64>(text_column, &mut tokenizer, &query_tokens)
                    }
                    _ => {
                        return Err(DataFusionError::Execution(format!(
                            "Column {} is not a string",
                            column,
                        )));
                    }
                };
                Ok(arrow::compute::filter_record_batch(&batch, &predicate)?)
            }
        });
        let stream = stream.map(move |batch| {
            let poll = baseline.record_poll(std::task::Poll::Ready(Some(batch)));
            match poll {
                std::task::Poll::Ready(Some(b)) => b,
                _ => unreachable!("record_poll preserves Ready(Some) input"),
            }
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

impl ExecutionPlan for FlatMatchFilterExec {
    fn name(&self) -> &str {
        "FlatMatchFilterExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "Unexpected number of children".to_string(),
            ));
        }
        let input = children.pop().ok_or_else(|| {
            DataFusionError::Internal("Unexpected number of children".to_string())
        })?;

        Ok(Arc::new(Self {
            dataset: self.dataset.clone(),
            input,
            query: self.query.clone(),
            params: self.params.clone(),
            preset_segments: self.preset_segments.clone(),
            metrics: ExecutionPlanMetricsSet::new(),
        }))
    }

    #[instrument(name = "flat_match_filter_exec", level = "debug", skip_all)]
    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let schema = self.schema();
        let stream_fut = Self::build_filter_stream(
            input,
            partition,
            schema.clone(),
            self.dataset.clone(),
            self.query.clone(),
            self.preset_segments.clone(),
            self.metrics.clone(),
        );
        let stream = stream::once(stream_fut)
            .try_flatten()
            .stream_in_current_span()
            .boxed();
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }

    fn partition_statistics(&self, partition: Option<usize>) -> DataFusionResult<Statistics> {
        self.input.partition_statistics(partition)
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn supports_limit_pushdown(&self) -> bool {
        true
    }
}

/// Calculates the FTS score for each row in the input
#[derive(Debug)]
pub struct FlatMatchQueryExec {
    dataset: Arc<Dataset>,
    query: MatchQuery,
    params: FtsSearchParams,
    unindexed_input: Arc<dyn ExecutionPlan>,
    /// Optional override for the BM25 scorer normally built locally inside
    /// `execute()`. See [`MatchQueryExec::with_base_scorer`].
    base_scorer: Option<Arc<MemBM25Scorer>>,
    /// Optional pre-resolved segment list. See
    /// [`MatchQueryExec::new_with_segments`].
    preset_segments: Option<Vec<IndexMetadata>>,

    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl DisplayAs for FlatMatchQueryExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "FlatMatchQuery: column={}, query={}",
                    self.query.column.as_deref().unwrap_or_default(),
                    self.query.terms
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "FlatMatchQuery\ncolumn={}\nquery={}",
                    self.query.column.as_deref().unwrap_or_default(),
                    self.query.terms
                )
            }
        }
    }
}

impl FlatMatchQueryExec {
    pub fn new(
        dataset: Arc<Dataset>,
        query: MatchQuery,
        params: FtsSearchParams,
        unindexed_input: Arc<dyn ExecutionPlan>,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(FTS_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self {
            dataset,
            query,
            params,
            unindexed_input,
            base_scorer: None,
            preset_segments: None,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// See [`MatchQueryExec::new_with_segments`].
    pub fn new_with_segments(
        dataset: Arc<Dataset>,
        query: MatchQuery,
        params: FtsSearchParams,
        unindexed_input: Arc<dyn ExecutionPlan>,
        segments: Vec<IndexMetadata>,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(FTS_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self {
            dataset,
            query,
            params,
            unindexed_input,
            base_scorer: None,
            preset_segments: Some(segments),
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// Override the local BM25 scorer; see [`MatchQueryExec::with_base_scorer`].
    pub fn with_base_scorer(mut self, scorer: Arc<MemBM25Scorer>) -> Self {
        self.base_scorer = Some(scorer);
        self
    }

    pub fn query(&self) -> &MatchQuery {
        &self.query
    }

    pub fn params(&self) -> &FtsSearchParams {
        &self.params
    }

    pub fn dataset(&self) -> &Arc<Dataset> {
        &self.dataset
    }

    pub fn base_scorer(&self) -> Option<&Arc<MemBM25Scorer>> {
        self.base_scorer.as_ref()
    }

    pub fn preset_segments(&self) -> Option<&[IndexMetadata]> {
        self.preset_segments.as_deref()
    }
}

impl ExecutionPlan for FlatMatchQueryExec {
    fn name(&self) -> &str {
        "FlatMatchQueryExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.unindexed_input]
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        // `execute()` only reads `unindexed_input.execute(partition)` for the single
        // output partition, so the input must be coalesced to one partition. Without
        // this, EnforceDistribution may round-robin the scan across `target_partitions`
        // and only partition 0 is consumed, silently dropping the other fragments.
        vec![Distribution::SinglePartition]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "Unexpected number of children".to_string(),
            ));
        }
        let unindexed_input = children.pop().unwrap();
        Ok(Arc::new(Self {
            dataset: self.dataset.clone(),
            query: self.query.clone(),
            params: self.params.clone(),
            unindexed_input,
            base_scorer: self.base_scorer.clone(),
            preset_segments: self.preset_segments.clone(),
            properties: self.properties.clone(),
            metrics: ExecutionPlanMetricsSet::new(),
        }))
    }

    #[instrument(name = "flat_match_query_exec", level = "debug", skip_all)]
    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let query = self.query.clone();
        let ds = self.dataset.clone();
        let preset_base_scorer = self.base_scorer.clone();
        let preset_segments = self.preset_segments.clone();
        let metrics = Arc::new(FtsIndexMetrics::new(&self.metrics, partition));
        let metrics_clone = metrics.clone();
        let target_batch_size = context.session_config().batch_size();

        // CPU time accumulator passed into `flat_bm25_search_stream_with_metrics`
        // so it can attribute the spawn_cpu tokenize work and synchronous
        // scoring back onto this node's `elapsed_compute`. Sharing the same
        // `Time` handle that's already inside the FtsIndexMetrics avoids
        // registering a duplicate metric.
        let elapsed_compute = metrics.baseline_metrics.elapsed_compute().clone();

        let column = query.column.ok_or(DataFusionError::Execution(format!(
            "column not set for MatchQuery {}",
            query.terms
        )))?;
        let unindexed_input =
            document_input(self.unindexed_input.execute(partition, context)?, &column)?;

        let stream = stream::once(async move {
            let segments = match preset_segments {
                Some(segments) => Some(segments),
                None => load_segments(&ds, &column).await?,
            };
            let (tokenizer, base_scorer) = match segments {
                Some(segments) => {
                    let _details = load_segment_details(&ds, &column, &segments).await?;
                    let indices =
                        open_fts_segments(&ds, &column, &segments, &metrics.index_metrics).await?;
                    metrics.record_parts_searched(
                        indices.iter().map(|index| index.partition_count()).sum(),
                    );
                    let first_index = indices.first().ok_or(DataFusionError::Execution(
                        format!("FTS index for column {} has no segments", column),
                    ))?;
                    let mut tokenizer = first_index.tokenizer();
                    let base_scorer = match preset_base_scorer {
                        Some(scorer) => (*scorer).clone(),
                        None => {
                            let query_tokens = collect_query_tokens(&query.terms, &mut tokenizer);
                            build_global_bm25_scorer(
                                &indices,
                                &query_tokens,
                                &FtsSearchParams::new(),
                            )
                            .boxed()
                            .await?
                        }
                    };
                    (tokenizer, Some(base_scorer))
                }
                None => (
                    default_text_tokenizer(),
                    preset_base_scorer.map(|s| (*s).clone()),
                ),
            };

            flat_bm25_search_stream_with_metrics(
                unindexed_input,
                column,
                query.terms,
                tokenizer,
                base_scorer,
                target_batch_size,
                Some(elapsed_compute),
            )
            .await
        })
        .try_flatten()
        .map(move |batch| {
            // record_poll records output_rows, output_bytes, and output_batches
            // on the shared BaselineMetrics — same pattern DataFusion's own
            // FilterExec uses inside its hand-written poll_next.
            let poll = metrics_clone
                .baseline_metrics
                .record_poll(std::task::Poll::Ready(Some(batch)));
            match poll {
                std::task::Poll::Ready(Some(b)) => b,
                _ => unreachable!("record_poll preserves Ready(Some) input"),
            }
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

#[derive(Debug)]
pub struct PhraseQueryExec {
    dataset: Arc<Dataset>,
    query: PhraseQuery,
    params: FtsSearchParams,
    prefilter_source: PreFilterSource,
    /// Optional override for the BM25 scorer normally built locally inside
    /// `execute()`. See [`MatchQueryExec::with_base_scorer`].
    base_scorer: Option<Arc<MemBM25Scorer>>,
    /// Optional pre-resolved segment list. See
    /// [`MatchQueryExec::new_with_segments`].
    preset_segments: Option<Vec<IndexMetadata>>,
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
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "PhraseQuery\ncolumn={}\nquery={}",
                    self.query.column.as_deref().unwrap_or_default(),
                    self.query.terms
                )
            }
        }
    }
}

impl PhraseQueryExec {
    pub fn new(
        dataset: Arc<Dataset>,
        query: PhraseQuery,
        mut params: FtsSearchParams,
        prefilter_source: PreFilterSource,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(FTS_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        params = params.with_phrase_slop(Some(query.slop));

        Self {
            dataset,
            query,
            params,
            prefilter_source,
            base_scorer: None,
            preset_segments: None,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// See [`MatchQueryExec::new_with_segments`].
    pub fn new_with_segments(
        dataset: Arc<Dataset>,
        query: PhraseQuery,
        mut params: FtsSearchParams,
        prefilter_source: PreFilterSource,
        segments: Vec<IndexMetadata>,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(FTS_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        params = params.with_phrase_slop(Some(query.slop));

        Self {
            dataset,
            query,
            params,
            prefilter_source,
            base_scorer: None,
            preset_segments: Some(segments),
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// Override the local BM25 scorer; see [`MatchQueryExec::with_base_scorer`].
    pub fn with_base_scorer(mut self, scorer: Arc<MemBM25Scorer>) -> Self {
        self.base_scorer = Some(scorer);
        self
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

    pub fn preset_segments(&self) -> Option<&[IndexMetadata]> {
        self.preset_segments.as_deref()
    }
}

impl ExecutionPlan for PhraseQueryExec {
    fn name(&self) -> &str {
        "PhraseQueryExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        match &self.prefilter_source {
            PreFilterSource::None => vec![],
            PreFilterSource::FilteredRowIds(src) => vec![&src],
            PreFilterSource::ScalarIndexQuery(src) => vec![&src],
        }
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        // Prefilter inputs must be a single partition
        self.children()
            .iter()
            .map(|_| Distribution::SinglePartition)
            .collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let plan = match children.len() {
            0 => Self {
                dataset: self.dataset.clone(),
                query: self.query.clone(),
                params: self.params.clone(),
                prefilter_source: PreFilterSource::None,
                base_scorer: self.base_scorer.clone(),
                preset_segments: self.preset_segments.clone(),
                properties: self.properties.clone(),
                metrics: ExecutionPlanMetricsSet::new(),
            },
            1 => {
                let src = children.pop().unwrap();
                let prefilter_source = match &self.prefilter_source {
                    PreFilterSource::FilteredRowIds(_) => {
                        PreFilterSource::FilteredRowIds(src.clone())
                    }
                    PreFilterSource::ScalarIndexQuery(_) => {
                        PreFilterSource::ScalarIndexQuery(src.clone())
                    }
                    PreFilterSource::None => {
                        return Err(DataFusionError::Internal(
                            "Unexpected prefilter source".to_string(),
                        ));
                    }
                };
                Self {
                    dataset: self.dataset.clone(),
                    query: self.query.clone(),
                    params: self.params.clone(),
                    prefilter_source,
                    base_scorer: self.base_scorer.clone(),
                    preset_segments: self.preset_segments.clone(),
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
        let params = self.params.clone();
        let ds = self.dataset.clone();
        let prefilter_source = self.prefilter_source.clone();
        let preset_base_scorer = self.base_scorer.clone();
        let preset_segments = self.preset_segments.clone();
        let metrics = Arc::new(FtsIndexMetrics::new(&self.metrics, partition));
        let stream = stream::once(async move {
            let _timer = metrics.baseline_metrics.elapsed_compute().timer();
            let column = query.column.ok_or(DataFusionError::Execution(format!(
                "column not set for PhraseQuery {}",
                query.terms
            )))?;
            let segments = match preset_segments {
                Some(segments) => segments,
                None => load_segments(&ds, &column)
                    .await?
                    .ok_or(DataFusionError::Execution(format!(
                        "No Inverted index found for column {}",
                        column,
                    )))?,
            };
            let _details = load_segment_details(&ds, &column, &segments).await?;
            let indices =
                open_fts_segments(&ds, &column, &segments, &metrics.index_metrics).await?;

            let mut pre_filter =
                build_prefilter(context.clone(), partition, &prefilter_source, ds, &segments)?;
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
            let tokens = collect_query_tokens(&query.terms, &mut tokenizer);
            let base_scorer = match preset_base_scorer {
                Some(scorer) => scorer,
                None => Arc::new(
                    build_global_bm25_scorer(&indices, &tokens, &params)
                        .boxed()
                        .await?,
                ),
            };

            pre_filter.wait_for_ready().await?;
            let tokens = Arc::new(tokens);
            let params = Arc::new(params);
            let (doc_ids, scores) = search_segments(
                &indices,
                tokens,
                params,
                lance_index::scalar::inverted::query::Operator::And,
                pre_filter,
                metrics.clone(),
                base_scorer,
            )
            .await?;
            metrics.baseline_metrics.record_output(doc_ids.len());
            let batch = RecordBatch::try_new(
                FTS_SCHEMA.clone(),
                vec![
                    Arc::new(UInt64Array::from(doc_ids)),
                    Arc::new(Float32Array::from(scores)),
                ],
            )?;
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

#[derive(Debug)]
pub struct BoostQueryExec {
    query: BoostQuery,
    params: FtsSearchParams,
    positive: Arc<dyn ExecutionPlan>,
    negative: Arc<dyn ExecutionPlan>,

    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl DisplayAs for BoostQueryExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "BoostQuery: negative_boost={}",
                    self.query.negative_boost
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "BoostQuery\nnegative_boost={}",
                    self.query.negative_boost
                )
            }
        }
    }
}

impl BoostQueryExec {
    pub fn new(
        query: BoostQuery,
        params: FtsSearchParams,
        positive: Arc<dyn ExecutionPlan>,
        negative: Arc<dyn ExecutionPlan>,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(FTS_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Self {
            query,
            params,
            positive,
            negative,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    pub fn query(&self) -> &BoostQuery {
        &self.query
    }

    pub fn params(&self) -> &FtsSearchParams {
        &self.params
    }

    pub fn positive(&self) -> &Arc<dyn ExecutionPlan> {
        &self.positive
    }

    pub fn negative(&self) -> &Arc<dyn ExecutionPlan> {
        &self.negative
    }
}

impl ExecutionPlan for BoostQueryExec {
    fn name(&self) -> &str {
        "BoostQueryExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.positive, &self.negative]
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        // This node fully consumes and re-orders the input rows.
        // It must be run on a single partition.
        self.children()
            .iter()
            .map(|_| Distribution::SinglePartition)
            .collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 2 {
            return Err(DataFusionError::Internal(
                "Unexpected number of children".to_string(),
            ));
        }

        let negative = children.pop().unwrap();
        let positive = children.pop().unwrap();
        Ok(Arc::new(Self {
            query: self.query.clone(),
            params: self.params.clone(),
            positive,
            negative,
            properties: self.properties.clone(),
            metrics: ExecutionPlanMetricsSet::new(),
        }))
    }

    #[instrument(name = "boost_query_exec", level = "debug", skip_all)]
    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let query = self.query.clone();
        let params = self.params.clone();
        let positive = self.positive.execute(partition, context.clone())?;
        let negative = self.negative.execute(partition, context)?;
        let metrics = Arc::new(FtsIndexMetrics::new(&self.metrics, partition));
        let stream = stream::once(async move {
            let positive = positive.try_collect::<Vec<_>>().await?;
            let negative = negative.try_collect::<Vec<_>>().await?;

            let _timer = metrics.baseline_metrics.elapsed_compute().timer();
            let mut res = HashMap::new();
            for batch in positive {
                let doc_ids = batch[ROW_ID].as_primitive::<UInt64Type>().values();
                let scores = batch[SCORE_COL].as_primitive::<Float32Type>().values();

                for (doc_id, score) in std::iter::zip(doc_ids, scores) {
                    res.insert(*doc_id, *score);
                }
            }
            for batch in negative {
                let doc_ids = batch[ROW_ID].as_primitive::<UInt64Type>().values();
                let scores = batch[SCORE_COL].as_primitive::<Float32Type>().values();

                for (doc_id, neg_score) in std::iter::zip(doc_ids, scores) {
                    if let Some(score) = res.get_mut(doc_id) {
                        *score -= query.negative_boost * neg_score;
                    }
                }
            }

            let (doc_ids, scores): (Vec<_>, Vec<_>) = res
                .into_iter()
                .sorted_unstable_by(|(_, a), (_, b)| b.total_cmp(a))
                .take(params.limit.unwrap_or(usize::MAX))
                .unzip();
            metrics.baseline_metrics.record_output(doc_ids.len());

            let batch = RecordBatch::try_new(
                FTS_SCHEMA.clone(),
                vec![
                    Arc::new(UInt64Array::from(doc_ids)),
                    Arc::new(Float32Array::from(scores)),
                ],
            )?;
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

/// Identifies which clause of a [`BooleanQuery`] a list of child execs
/// belongs to. Used by [`build_boolean_query_children`] to pick the
/// right exec shape per slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolSlot {
    Should,
    Must,
    MustNot,
}

/// Combine N children into the per-slot exec shape that
/// [`BooleanQueryExec::new`] expects. Used by `Scanner::plan_fts` to
/// assemble the per-slot exec shape:
///
/// | slot      | 0 children                 | 1 child       | N children                                          |
/// |-----------|----------------------------|---------------|-----------------------------------------------------|
/// | Should    | `Some(EmptyExec(FTS))`     | `Some(child)` | `Some(Union -> Repartition(RoundRobinBatch(1)))`    |
/// | Must      | `None`                     | `Some(child)` | `Some(chained HashJoin on row_id)`                  |
/// | MustNot   | `Some(EmptyExec(FTS))`     | `Some(child)` | `Some(Union -> Repartition(RoundRobinBatch(1)))`    |
///
/// Errors only on internal invariants (HashJoin construction, Schema
/// lookups). Returns `Result<Option<Arc<dyn ExecutionPlan>>>` so the
/// `Must` slot's `None` case is naturally expressible.
pub fn build_boolean_query_children(
    slot: BoolSlot,
    mut children: Vec<Arc<dyn ExecutionPlan>>,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    match slot {
        BoolSlot::Should | BoolSlot::MustNot => {
            if children.is_empty() {
                Ok(Some(Arc::new(EmptyExec::new(FTS_SCHEMA.clone()))))
            } else if children.len() == 1 {
                Ok(Some(children.pop().unwrap()))
            } else {
                let unioned = UnionExec::try_new(children)?;
                Ok(Some(Arc::new(RepartitionExec::try_new(
                    unioned,
                    Partitioning::RoundRobinBatch(1),
                )?)))
            }
        }
        BoolSlot::Must => {
            let mut joined: Option<Arc<dyn ExecutionPlan>> = None;
            for plan in children {
                if let Some(left) = joined {
                    joined = Some(Arc::new(HashJoinExec::try_new(
                        left,
                        plan,
                        vec![(
                            Arc::new(Column::new_with_schema(ROW_ID, &FTS_SCHEMA)?),
                            Arc::new(Column::new_with_schema(ROW_ID, &FTS_SCHEMA)?),
                        )],
                        None,
                        &datafusion_expr::JoinType::Inner,
                        None,
                        PartitionMode::CollectLeft,
                        NullEquality::NullEqualsNothing,
                        false,
                    )?) as _);
                } else {
                    joined = Some(plan);
                }
            }
            Ok(joined)
        }
    }
}

#[derive(Debug)]
pub struct BooleanQueryExec {
    query: BooleanQuery,
    params: FtsSearchParams,
    should: Arc<dyn ExecutionPlan>,
    must: Option<Arc<dyn ExecutionPlan>>,
    must_not: Arc<dyn ExecutionPlan>,

    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl DisplayAs for BooleanQueryExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "BooleanQuery: should={:?}, must={:?}, must_not={:?}",
                    self.query.should, self.query.must, self.query.must_not,
                )
            }
            DisplayFormatType::TreeRender => {
                write!(f, "BooleanQuery")?;
                if !self.query.should.is_empty() {
                    write!(f, "\nshould={:?}", self.query.should)?;
                }
                if !self.query.must.is_empty() {
                    write!(f, "\nmust={:?}", self.query.must)?;
                }
                if !self.query.must_not.is_empty() {
                    write!(f, "\nmust_not={:?}", self.query.must_not)?;
                }
                std::fmt::Result::Ok(())
            }
        }
    }
}

impl BooleanQueryExec {
    pub fn new(
        query: BooleanQuery,
        params: FtsSearchParams,
        should: Arc<dyn ExecutionPlan>,
        must: Option<Arc<dyn ExecutionPlan>>,
        must_not: Arc<dyn ExecutionPlan>,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(FTS_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Self {
            query,
            params,
            must,
            should,
            must_not,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    pub fn query(&self) -> &BooleanQuery {
        &self.query
    }

    pub fn params(&self) -> &FtsSearchParams {
        &self.params
    }

    pub fn should(&self) -> &Arc<dyn ExecutionPlan> {
        &self.should
    }

    pub fn must(&self) -> Option<&Arc<dyn ExecutionPlan>> {
        self.must.as_ref()
    }

    pub fn must_not(&self) -> &Arc<dyn ExecutionPlan> {
        &self.must_not
    }
}

impl ExecutionPlan for BooleanQueryExec {
    fn name(&self) -> &str {
        "BooleanQueryExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        match &self.must {
            Some(must) => vec![&self.should, &self.must_not, must],
            None => vec![&self.should, &self.must_not],
        }
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        // This node fully consumes and re-orders the input rows.
        // It must be run on a single partition.
        self.children()
            .iter()
            .map(|_| Distribution::SinglePartition)
            .collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        match children.len() {
            1 => {
                let should = children.pop().unwrap();
                Ok(Arc::new(Self {
                    query: self.query.clone(),
                    params: self.params.clone(),
                    should,
                    must: None,
                    must_not: self.must_not.clone(),
                    properties: self.properties.clone(),
                    metrics: ExecutionPlanMetricsSet::new(),
                }))
            }
            2 => {
                let must_not = children.pop().unwrap();
                let should = children.pop().unwrap();
                Ok(Arc::new(Self {
                    query: self.query.clone(),
                    params: self.params.clone(),
                    should,
                    must: None,
                    must_not,
                    properties: self.properties.clone(),
                    metrics: ExecutionPlanMetricsSet::new(),
                }))
            }
            3 => {
                let must = children.pop().unwrap();
                let must_not = children.pop().unwrap();
                let should = children.pop().unwrap();
                Ok(Arc::new(Self {
                    query: self.query.clone(),
                    params: self.params.clone(),
                    should,
                    must: Some(must),
                    must_not,
                    properties: self.properties.clone(),
                    metrics: ExecutionPlanMetricsSet::new(),
                }))
            }
            _ => Err(DataFusionError::Internal(
                "Unexpected number of children".to_string(),
            )),
        }
    }

    #[instrument(name = "bool_query_exec", level = "debug", skip_all)]
    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let params = self.params.clone();
        let should_plan = self.should.clone();
        let must_plan = self.must.clone();
        let must_not_plan = self.must_not.clone();
        let must = self
            .must
            .as_ref()
            .map(|m| m.execute(partition, context.clone()))
            .transpose()?;
        let mut should = self.should.execute(partition, context.clone())?;
        let mut must_not = self.must_not.execute(partition, context)?;
        let metrics = Arc::new(FtsIndexMetrics::new(&self.metrics, partition));

        let stream = stream::once(async move {
            let elapsed_time = metrics.baseline_metrics.elapsed_compute();

            let mut res = HashMap::new();
            let has_must = must.is_some();
            if let Some(mut must) = must {
                while let Some(batch) = must.try_next().await? {
                    let _timer = elapsed_time.timer();
                    let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>().values();
                    let scores = batch[SCORE_COL].as_primitive::<Float32Type>().values();
                    res.extend(std::iter::zip(
                        row_ids.iter().copied(),
                        scores.iter().copied(),
                    ));
                }
            }

            // add the scores from the should clause
            while let Some(batch) = should.try_next().await? {
                let _timer = elapsed_time.timer();
                let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>().values();
                let scores = batch[SCORE_COL].as_primitive::<Float32Type>().values();

                for (row_id, score) in std::iter::zip(row_ids, scores) {
                    let entry = res.entry(*row_id).and_modify(|e| *e += score);
                    if !has_must {
                        entry.or_insert(*score);
                    }
                }
            }

            // remove the results from the must_not clause
            while let Some(batch) = must_not.try_next().await? {
                let _timer = elapsed_time.timer();
                let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>().values();
                for row_id in row_ids {
                    res.remove(row_id);
                }
            }

            let mut partitions_searched = 0;
            for plan in [Some(&should_plan), must_plan.as_ref(), Some(&must_not_plan)] {
                let Some(plan) = plan else {
                    continue;
                };
                let Some(metrics) = plan.metrics() else {
                    continue;
                };
                for (metric_name, count) in metrics.iter_counts() {
                    if metric_name.as_ref() == PARTITIONS_SEARCHED_METRIC {
                        partitions_searched += count.value();
                    }
                }
            }
            metrics.record_parts_searched(partitions_searched);

            // sort the results and take the top k
            let _timer = elapsed_time.timer();
            let (row_ids, scores): (Vec<_>, Vec<_>) = res
                .into_iter()
                .sorted_unstable_by(|(_, a), (_, b)| b.total_cmp(a))
                .take(params.limit.unwrap_or(usize::MAX))
                .unzip();
            metrics.baseline_metrics.record_output(row_ids.len());
            let batch = RecordBatch::try_new(
                FTS_SCHEMA.clone(),
                vec![
                    Arc::new(UInt64Array::from(row_ids)),
                    Arc::new(Float32Array::from(scores)),
                ],
            )?;
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
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use crate::index::DatasetIndexExt;
    use arrow_array::{
        ArrayRef, Float32Array, Int32Array, RecordBatch, RecordBatchIterator, StringArray,
        UInt64Array,
    };
    use arrow_schema::DataType;
    use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
    use datafusion::{execution::TaskContext, physical_plan::ExecutionPlan};
    use futures::TryStreamExt;
    use lance_core::ROW_ID;
    use lance_datafusion::datagen::DatafusionDatagenExt;
    use lance_datafusion::exec::{ExecutionStatsCallback, ExecutionSummaryCounts};
    use lance_datafusion::utils::PARTITIONS_SEARCHED_METRIC;
    use lance_datagen::{BatchCount, ByteCount, RowCount};
    use lance_index::metrics::NoOpMetricsCollector;
    use lance_index::scalar::inverted::query::{
        BooleanQuery, BoostQuery, FtsQuery, FtsSearchParams, MatchQuery, Occur, Operator,
        PhraseQuery, collect_query_tokens, has_query_token,
    };
    use lance_index::scalar::inverted::{
        FTS_SCHEMA, InvertedIndex, Language, SCORE_COL, build_global_bm25_scorer,
    };
    use lance_index::scalar::{FullTextSearchQuery, InvertedIndexParams};
    use lance_index::{IndexCriteria, IndexType};
    use lance_table::format::IndexMetadata;

    use crate::{
        Dataset,
        dataset::WriteParams,
        dataset::transaction::{Operation, TransactionBuilder},
        index::DatasetIndexInternalExt,
        io::exec::PreFilterSource,
        utils::test::{DatagenExt, FragmentCount, FragmentRowCount, NoContextTestFixture},
    };

    use super::{
        BoolSlot, BoostQueryExec, FlatMatchFilterExec, FlatMatchQueryExec, MatchQueryExec,
        PhraseQueryExec, build_boolean_query_children, open_fts_segments,
    };
    use crate::io::exec::utils::IndexMetrics;
    use datafusion::physical_plan::empty::EmptyExec;
    use datafusion::physical_plan::repartition::RepartitionExec;
    use datafusion::physical_plan::union::UnionExec;
    use datafusion_physical_plan::joins::HashJoinExec;

    #[derive(Default)]
    struct StatsHolder {
        collected_stats: Arc<Mutex<Option<ExecutionSummaryCounts>>>,
    }

    impl StatsHolder {
        fn get_setter(&self) -> ExecutionStatsCallback {
            let collected_stats = self.collected_stats.clone();
            Arc::new(move |stats| {
                *collected_stats.lock().unwrap() = Some(stats.clone());
            })
        }

        fn consume(self) -> ExecutionSummaryCounts {
            self.collected_stats.lock().unwrap().take().unwrap()
        }
    }

    #[test]
    fn execute_without_context() {
        // These tests ensure we can create nodes and call execute without a tokio Runtime
        // being active.  This is a requirement for proper implementation of a Datafusion foreign
        // table provider.
        let fixture = NoContextTestFixture::new();
        let match_query = MatchQueryExec::new(
            Arc::new(fixture.dataset.clone()),
            MatchQuery::new("blah".to_string()).with_column(Some("text".to_string())),
            FtsSearchParams::default(),
            PreFilterSource::None,
        );
        match_query
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap();
        let metrics = match_query.metrics().unwrap();
        assert!(metrics.elapsed_compute().unwrap() > 0);

        let flat_input = lance_datagen::gen_batch()
            .col(
                "text",
                lance_datagen::array::rand_utf8(ByteCount::from(10), false),
            )
            .into_df_exec(RowCount::from(15), BatchCount::from(2));

        let flat_match_query = FlatMatchQueryExec::new(
            Arc::new(fixture.dataset.clone()),
            MatchQuery::new("blah".to_string()).with_column(Some("text".to_string())),
            FtsSearchParams::default(),
            flat_input,
        );
        flat_match_query
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap();
        let metrics = flat_match_query.metrics().unwrap();
        assert!(metrics.elapsed_compute().unwrap() > 0);

        let phrase_query = PhraseQueryExec::new(
            Arc::new(fixture.dataset.clone()),
            PhraseQuery::new("blah".to_string()),
            FtsSearchParams::new().with_phrase_slop(Some(0)),
            PreFilterSource::None,
        );
        phrase_query
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap();
        let metrics = phrase_query.metrics().unwrap();
        assert!(metrics.elapsed_compute().unwrap() > 0);

        let boost_input_one = MatchQueryExec::new(
            Arc::new(fixture.dataset.clone()),
            MatchQuery::new("blah".to_string()).with_column(Some("text".to_string())),
            FtsSearchParams::default(),
            PreFilterSource::None,
        );

        let boost_input_two = MatchQueryExec::new(
            Arc::new(fixture.dataset),
            MatchQuery::new("blah".to_string()).with_column(Some("text".to_string())),
            FtsSearchParams::default(),
            PreFilterSource::None,
        );

        let boost_query = BoostQueryExec::new(
            BoostQuery::new(
                FtsQuery::Match(
                    MatchQuery::new("blah".to_string()).with_column(Some("text".to_string())),
                ),
                FtsQuery::Match(
                    MatchQuery::new("test".to_string()).with_column(Some("text".to_string())),
                ),
                Some(1.0),
            ),
            FtsSearchParams::default(),
            Arc::new(boost_input_one),
            Arc::new(boost_input_two),
        );
        boost_query
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap();
        let metrics = boost_query.metrics().unwrap();
        assert!(metrics.elapsed_compute().unwrap() > 0);
    }

    #[test]
    fn test_flat_match_filter_find_matches_large_utf8() {
        use arrow_array::LargeStringArray;

        use super::default_text_tokenizer;

        let mut tokenizer = default_text_tokenizer();
        let query_tokens = collect_query_tokens("hello", &mut tokenizer);

        let text_col =
            LargeStringArray::from(vec!["hello world", "no match here", "say hello there"]);

        let result =
            FlatMatchFilterExec::find_matches::<i64>(&text_col, &mut tokenizer, &query_tokens);

        assert_eq!(result.len(), 3);
        assert!(result.value(0), "expected match in 'hello world'");
        assert!(!result.value(1), "expected no match in 'no match here'");
        assert!(result.value(2), "expected match in 'say hello there'");
    }

    #[tokio::test]
    async fn test_flat_match_filter_load_tokenizer_uses_on_disk_params_when_details_missing() {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "text",
                lance_datagen::array::cycle_utf8_literals(&["hello", "HELLO"]),
            )
            .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(2))
            .await
            .unwrap();

        let params = InvertedIndexParams::new("simple".to_string(), Language::English)
            .with_position(false)
            .lower_case(false)
            .stem(false)
            .remove_stop_words(false)
            .ascii_folding(false)
            .max_token_length(None);
        dataset
            .create_index(&["text"], IndexType::Inverted, None, &params, true)
            .await
            .unwrap();

        let index_meta = dataset
            .load_scalar_index(IndexCriteria::default().for_column("text").supports_fts())
            .await
            .unwrap()
            .unwrap();
        let mut legacy_index_meta = index_meta.clone();
        legacy_index_meta.index_details = None;
        let transaction = TransactionBuilder::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![legacy_index_meta],
                removed_indices: vec![index_meta],
            },
        )
        .build();
        dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await
            .unwrap();

        let metrics = IndexMetrics::new(&ExecutionPlanMetricsSet::new(), 0);
        let mut tokenizer = FlatMatchFilterExec::load_tokenizer(&dataset, "text", &metrics)
            .await
            .unwrap();
        let query_tokens = collect_query_tokens("hello", &mut tokenizer);

        let mut tokenizer = FlatMatchFilterExec::load_tokenizer(&dataset, "text", &metrics)
            .await
            .unwrap();
        assert!(has_query_token("hello", &mut tokenizer, &query_tokens));
        assert!(
            !has_query_token("HELLO", &mut tokenizer, &query_tokens),
            "legacy FTS indices should continue using on-disk tokenizer params"
        );
    }

    #[tokio::test]
    async fn test_parts_searched_metrics() {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "text",
                lance_datagen::array::cycle_utf8_literals(&["hello", "lance", "search"]),
            )
            .into_ram_dataset(FragmentCount::from(3), FragmentRowCount::from(5))
            .await
            .unwrap();

        dataset
            .create_index(
                &["text"],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        let index_meta = dataset
            .load_scalar_index(IndexCriteria::default().for_column("text").supports_fts())
            .await
            .unwrap()
            .unwrap();
        let index = dataset
            .open_generic_index("text", &index_meta.uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let inverted_index = index.as_any().downcast_ref::<InvertedIndex>().unwrap();
        let expected_parts = inverted_index.partition_count();

        let stats_holder = StatsHolder::default();
        let mut scanner = dataset.scan();
        scanner
            .scan_stats_callback(stats_holder.get_setter())
            .project(&["text"])
            .unwrap()
            .with_row_id()
            .full_text_search(FullTextSearchQuery::new("hello".to_string()))
            .unwrap();
        let _ = scanner.try_into_batch().await.unwrap();
        let stats = stats_holder.consume();
        let parts_searched = stats
            .all_counts
            .get(PARTITIONS_SEARCHED_METRIC)
            .copied()
            .unwrap_or_default();
        assert_eq!(parts_searched, expected_parts);

        let mut analyze_scanner = dataset.scan();
        analyze_scanner
            .project(&["text"])
            .unwrap()
            .with_row_id()
            .full_text_search(FullTextSearchQuery::new("hello".to_string()))
            .unwrap();
        let analysis = analyze_scanner.analyze_plan().await.unwrap();
        assert!(analysis.contains(PARTITIONS_SEARCHED_METRIC));
    }

    #[tokio::test]
    async fn test_boolean_query_parts_searched_metrics() {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "text",
                lance_datagen::array::cycle_utf8_literals(&["hello", "lance", "search"]),
            )
            .into_ram_dataset(FragmentCount::from(3), FragmentRowCount::from(5))
            .await
            .unwrap();

        dataset
            .create_index(
                &["text"],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        let index_meta = dataset
            .load_scalar_index(IndexCriteria::default().for_column("text").supports_fts())
            .await
            .unwrap()
            .unwrap();
        let index = dataset
            .open_generic_index("text", &index_meta.uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let inverted_index = index.as_any().downcast_ref::<InvertedIndex>().unwrap();
        let expected_parts = inverted_index.partition_count();

        let query = BooleanQuery::new([
            (
                Occur::Should,
                MatchQuery::new("hello".to_string())
                    .with_operator(Operator::And)
                    .into(),
            ),
            (
                Occur::Must,
                MatchQuery::new("lance".to_string())
                    .with_operator(Operator::And)
                    .into(),
            ),
        ]);
        let expected_total = expected_parts * 2;

        let mut scanner = dataset.scan();
        scanner
            .project(&["text"])
            .unwrap()
            .with_row_id()
            .full_text_search(FullTextSearchQuery::new_query(query.into()))
            .unwrap();
        let analysis = scanner.analyze_plan().await.unwrap();
        let boolean_line = analysis
            .lines()
            .find(|line| line.contains("BooleanQuery"))
            .unwrap();
        assert!(
            boolean_line.contains(&format!("{PARTITIONS_SEARCHED_METRIC}={expected_total}")),
            "BooleanQuery metrics missing partitions_searched: {boolean_line}"
        );
    }

    #[tokio::test]
    async fn test_match_query_exec_with_base_scorer_matches_baseline() {
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        // Skewed term distributions across two fragments — "lance" is common in
        // segment 0 and rare in segment 1 — so any local-IDF computation will
        // disagree with the global-IDF baseline. That makes the test sensitive
        // to a bug where `with_base_scorer` is silently ignored.
        let batches = vec![
            RecordBatch::try_from_iter(vec![
                ("id", Arc::new(Int32Array::from(vec![0, 1])) as ArrayRef),
                (
                    "text",
                    Arc::new(StringArray::from(vec![
                        Some("lance database"),
                        Some("lance search"),
                    ])) as ArrayRef,
                ),
            ])
            .unwrap(),
            RecordBatch::try_from_iter(vec![
                ("id", Arc::new(Int32Array::from(vec![2, 3])) as ArrayRef),
                (
                    "text",
                    Arc::new(StringArray::from(vec![
                        Some("alpha beta"),
                        Some("gamma lance"),
                    ])) as ArrayRef,
                ),
            ])
            .unwrap(),
        ];
        let schema = batches[0].schema();
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let mut ds = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 2,
                max_rows_per_group: 2,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let params = InvertedIndexParams::new("simple".to_string(), Language::English)
            .with_position(false)
            .lower_case(true)
            .stem(false)
            .remove_stop_words(false)
            .ascii_folding(false)
            .max_token_length(None);
        let fragment_ids = ds
            .get_fragments()
            .iter()
            .map(|fragment| fragment.id() as u32)
            .collect::<Vec<_>>();
        assert!(
            fragment_ids.len() >= 2,
            "test setup should produce >= 2 fragments, got {}",
            fragment_ids.len()
        );

        let mut metadatas = Vec::<IndexMetadata>::with_capacity(fragment_ids.len());
        for fragment_id in fragment_ids {
            let mut builder = ds
                .create_index_builder(&["text"], IndexType::Inverted, &params)
                .name("seg_fts".to_string())
                .fragments(vec![fragment_id]);
            metadatas.push(builder.execute_uncommitted().await.unwrap());
        }
        ds.commit_existing_index_segments("seg_fts", "text", metadatas.clone())
            .await
            .unwrap();
        assert_eq!(
            ds.load_indices_by_name("seg_fts").await.unwrap().len(),
            metadatas.len(),
            "expected one committed segment per fragment"
        );

        let dataset = Arc::new(ds);
        let query = MatchQuery::new("lance".to_string()).with_column(Some("text".to_string()));
        let search_params = FtsSearchParams::default().with_limit(Some(10));

        // Baseline: the existing path that builds the global scorer locally.
        let baseline_exec = MatchQueryExec::new(
            dataset.clone(),
            query.clone(),
            search_params.clone(),
            PreFilterSource::None,
        );
        let baseline_batches: Vec<RecordBatch> = baseline_exec
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let baseline = concat_score_batches(&baseline_batches);
        assert!(
            !baseline.is_empty(),
            "baseline should return at least one hit"
        );

        // Override: build the global scorer manually via the public helper, then
        // construct the exec with the preset segments and the preset scorer.
        let preset_segments = crate::index::scalar::inverted::load_segments(&dataset, "text")
            .await
            .unwrap()
            .expect("FTS index just created");
        let metrics_set = ExecutionPlanMetricsSet::new();
        let metrics = IndexMetrics::new(&metrics_set, 0);
        let indices = open_fts_segments(&dataset, "text", &preset_segments, &metrics)
            .await
            .unwrap();
        assert!(
            indices.len() >= 2,
            "expected >= 2 segments to exercise global IDF, got {}",
            indices.len()
        );
        let mut tokenizer = indices[0].tokenizer();
        let tokens = collect_query_tokens(&query.terms, &mut tokenizer);
        let global_scorer = Arc::new(
            build_global_bm25_scorer(&indices, &tokens, &search_params)
                .await
                .unwrap(),
        );

        let override_exec = MatchQueryExec::new_with_segments(
            dataset.clone(),
            query.clone(),
            search_params.clone(),
            PreFilterSource::None,
            preset_segments,
        )
        .with_base_scorer(global_scorer);
        let override_batches: Vec<RecordBatch> = override_exec
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let overridden = concat_score_batches(&override_batches);

        assert_eq!(
            baseline.len(),
            overridden.len(),
            "row count differs: baseline={}, override={}",
            baseline.len(),
            overridden.len()
        );
        for (i, (b, o)) in baseline.iter().zip(overridden.iter()).enumerate() {
            assert_eq!(
                b.0, o.0,
                "row id mismatch at rank {}: baseline={}, override={}",
                i, b.0, o.0
            );
            assert_eq!(
                b.1, o.1,
                "score mismatch at rank {} (row id {}): baseline={}, override={}",
                i, b.0, b.1, o.1
            );
        }

        // Sanity check on FTS schema before extracting columns above.
        for batch in baseline_batches.iter().chain(override_batches.iter()) {
            assert!(
                batch.column_by_name(ROW_ID).is_some(),
                "FTS output is expected to carry a row id column"
            );
            assert_eq!(
                batch.column_by_name(SCORE_COL).unwrap().data_type(),
                &DataType::Float32,
                "FTS score column should be Float32"
            );
        }

        // Locally-bound helper: collect (row_id, score) pairs sorted by score desc.
        fn concat_score_batches(batches: &[RecordBatch]) -> Vec<(u64, f32)> {
            let mut out: Vec<(u64, f32)> = Vec::new();
            for batch in batches {
                let row_ids = batch
                    .column_by_name(ROW_ID)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap();
                let scores = batch
                    .column_by_name(SCORE_COL)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap();
                for i in 0..batch.num_rows() {
                    out.push((row_ids.value(i), scores.value(i)));
                }
            }
            // Stable order for diffing — descending score, ties broken by row id.
            out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            out
        }
    }

    fn empty_fts_child() -> Arc<dyn ExecutionPlan> {
        Arc::new(EmptyExec::new(FTS_SCHEMA.clone()))
    }

    #[test]
    fn build_boolean_should_empty_returns_empty_exec() {
        let plan = build_boolean_query_children(BoolSlot::Should, vec![])
            .unwrap()
            .expect("Should slot always returns Some");
        assert!(
            plan.as_any().downcast_ref::<EmptyExec>().is_some(),
            "expected EmptyExec for empty Should slot, got {plan:?}"
        );
    }

    #[test]
    fn build_boolean_should_single_child_passthrough() {
        let child = empty_fts_child();
        let child_ptr = Arc::as_ptr(&child);
        let plan = build_boolean_query_children(BoolSlot::Should, vec![child])
            .unwrap()
            .expect("Should slot always returns Some");
        assert_eq!(
            Arc::as_ptr(&plan),
            child_ptr,
            "single-child Should should return the child unchanged"
        );
    }

    #[test]
    fn build_boolean_should_multi_child_union_repartition() {
        let plan = build_boolean_query_children(
            BoolSlot::Should,
            vec![empty_fts_child(), empty_fts_child()],
        )
        .unwrap()
        .expect("Should slot always returns Some");
        let repartition = plan
            .as_any()
            .downcast_ref::<RepartitionExec>()
            .expect("multi-child Should should be wrapped in RepartitionExec");
        let inner = repartition
            .input()
            .as_any()
            .downcast_ref::<UnionExec>()
            .expect("RepartitionExec should wrap a UnionExec");
        assert_eq!(inner.children().len(), 2);
    }

    #[test]
    fn build_boolean_must_empty_returns_none() {
        let plan = build_boolean_query_children(BoolSlot::Must, vec![]).unwrap();
        assert!(plan.is_none(), "empty Must slot should return None");
    }

    #[test]
    fn build_boolean_must_single_child_passthrough_some() {
        let child = empty_fts_child();
        let child_ptr = Arc::as_ptr(&child);
        let plan = build_boolean_query_children(BoolSlot::Must, vec![child])
            .unwrap()
            .expect("single-child Must should be Some");
        assert_eq!(
            Arc::as_ptr(&plan),
            child_ptr,
            "single-child Must should return the child unchanged"
        );
    }

    #[test]
    fn build_boolean_must_multi_child_chained_hashjoin() {
        let children = vec![empty_fts_child(), empty_fts_child(), empty_fts_child()];
        let n = children.len();
        let plan = build_boolean_query_children(BoolSlot::Must, children)
            .unwrap()
            .expect("multi-child Must should be Some");

        // Walk the left spine: each layer is a HashJoinExec whose left child is
        // either another HashJoinExec or the original leaf. With N children
        // there are N-1 joins.
        let mut joins = 0usize;
        let mut current: Arc<dyn ExecutionPlan> = plan;
        while let Some(join) = current.clone().as_any().downcast_ref::<HashJoinExec>() {
            joins += 1;
            current = join.children()[0].clone();
        }
        assert_eq!(joins, n - 1, "expected {} joins for {n} children", n - 1);
    }

    #[test]
    fn build_boolean_must_not_multi_child_union_repartition() {
        let plan = build_boolean_query_children(
            BoolSlot::MustNot,
            vec![empty_fts_child(), empty_fts_child()],
        )
        .unwrap()
        .expect("MustNot slot always returns Some");
        let repartition = plan
            .as_any()
            .downcast_ref::<RepartitionExec>()
            .expect("multi-child MustNot should be wrapped in RepartitionExec");
        let inner = repartition
            .input()
            .as_any()
            .downcast_ref::<UnionExec>()
            .expect("RepartitionExec should wrap a UnionExec");
        assert_eq!(inner.children().len(), 2);
    }
}
