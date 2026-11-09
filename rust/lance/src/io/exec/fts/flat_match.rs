// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Flat Match filtering and query execution nodes.

use std::collections::HashSet;
use std::sync::{Arc, OnceLock};

use arrow::array::{AsArray, BooleanBuilder};
use arrow_array::{Array, BooleanArray, OffsetSizeTrait};
use arrow_schema::{DataType, SchemaRef};
use datafusion::common::Statistics;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::ChildStats;
use datafusion::physical_plan::StatisticsArgs;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion_physical_expr::{Distribution, EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion_physical_plan::metrics::BaselineMetrics;
use futures::stream::{self};
use futures::{FutureExt, StreamExt, TryStreamExt};
use lance_core::{Error, Result, utils::tracing::StreamTracingExt};
use lance_table::format::IndexMetadata;

use crate::Dataset;
use crate::index::scalar::inverted::{
    ResolvedFtsField, load_segment_details, load_segments, resolve_fts_field,
};
use crate::io::exec::PreFilterSource;
use crate::io::exec::row_addr_mask::apply_mask;
use crate::io::exec::utils::{FilteredRowIdsToPrefilter, IndexMetrics, SelectionVectorToPrefilter};
use lance_index::prefilter::FilterLoader;
use lance_index::scalar::inverted::builder::document_input;
use lance_index::scalar::inverted::document_tokenizer::LanceTokenizer;
use lance_index::scalar::inverted::query::{
    FtsSearchParams, MatchQuery, Operator, Tokens, has_query_token, try_collect_query_tokens,
    uses_fuzzy_expansion,
};
use lance_index::scalar::inverted::{
    DocumentGranularity, FlatBm25SearchOptions, MemBM25Scorer, build_global_bm25_scorer,
    flat_bm25_search_stream_with_options_and_scorer, fts_schema,
};
use tracing::instrument;

use super::corpus::{SharedFtsScorer, SharedFtsScorerProducer};
use super::query_tokens::{
    TokenizedQuery, default_text_tokenizer, fmt_tokenized_query, record_tokenized_query,
};
use super::segments::{open_fts_segment, open_fts_segments};
use super::selection::FtsIndexMetrics;

/// How a flat Match filter obtains each document from its input rows.
///
/// This contains configuration only. Source traversal is derived again from the
/// pinned dataset schema by [`FlatMatchFilterExec::try_new`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FtsFilterDocumentBinding {
    /// A Utf8 or LargeUtf8 input column, possibly aliased from the logical query
    /// field. Its name is literal and is not parsed as a source field path.
    Column(String),
    /// A source field path in the dataset schema. The input contains its complete
    /// root column; each row is retained if any derived document matches.
    ResolvedField {
        /// Canonical source path, independent of input projection aliases.
        path: String,
        /// Physical document boundary; it takes precedence over the logical
        /// query option, including when that option is absent.
        document_granularity: DocumentGranularity,
    },
}

/// Filters the input according to a match query's token operator.
///
/// Reconstruction preserves document binding separately from the logical query
/// column and tokenizer segment selection. It does not copy cached tokens or
/// metrics, and it retains the input schema without adding scores or coordinates.
///
/// ```
/// # use std::sync::Arc;
/// # use datafusion::physical_plan::ExecutionPlan;
/// # use lance::{io::exec::fts::FlatMatchFilterExec, Result};
/// # fn rebuild(exec: &FlatMatchFilterExec, decoded_input: Arc<dyn ExecutionPlan>)
/// #     -> Result<FlatMatchFilterExec> {
/// FlatMatchFilterExec::try_new(
///     decoded_input, exec.dataset().clone(), exec.query().clone(),
///     exec.params().clone(), exec.document_binding(),
///     exec.preset_segments().map(|segments| segments.to_vec()),
/// )
/// # }
/// ```
#[derive(Debug)]

pub struct FlatMatchFilterExec {
    dataset: Arc<Dataset>,
    input: Arc<dyn ExecutionPlan>,
    query: MatchQuery,
    tokenized_query: Arc<OnceLock<TokenizedQuery>>,
    params: FtsSearchParams,
    /// Optional pre-resolved segment list. See
    /// [`MatchQueryExec::new_with_segments`]. `FlatMatchFilterExec` only
    /// uses the first segment's tokenizer, but the full list is preserved so
    /// the field round-trips through `with_new_children`.
    preset_segments: Option<Vec<IndexMetadata>>,
    document_column: String,
    resolved_field: Option<ResolvedFtsField>,

    metrics: ExecutionPlanMetricsSet,
}

struct FlatMatchFilterStreamOptions {
    dataset: Arc<Dataset>,
    query: MatchQuery,
    tokenized_query: Arc<OnceLock<TokenizedQuery>>,
    document_column: String,
    preset_segments: Option<Vec<IndexMetadata>>,
    resolved_field: Option<ResolvedFtsField>,
    metrics_set: ExecutionPlanMetricsSet,
}

fn document_matches_query(
    text: &str,
    tokenizer: &mut Box<dyn LanceTokenizer>,
    query_tokens: &Tokens,
    operator: Operator,
) -> bool {
    match operator {
        Operator::Or => has_query_token(text, tokenizer, query_tokens),
        Operator::And => {
            let mut remaining_groups = (0..query_tokens.len())
                .map(|index| query_tokens.term_group(index))
                .collect::<HashSet<_>>();
            if remaining_groups.is_empty() {
                return false;
            }
            let mut stream = tokenizer.token_stream_for_doc(text);
            while let Some(token) = stream.next() {
                for index in 0..query_tokens.len() {
                    if token.text == query_tokens.get_token(index) {
                        remaining_groups.remove(&query_tokens.term_group(index));
                    }
                }
                if remaining_groups.is_empty() {
                    return true;
                }
            }
            false
        }
    }
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
                )?;
                fmt_tokenized_query(&self.tokenized_query, ", ", f)
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "FlatMatchFilter\ncolumn={}\nquery={}",
                    self.query.column.as_deref().unwrap_or_default(),
                    self.query.terms
                )?;
                fmt_tokenized_query(&self.tokenized_query, "\n", f)
            }
        }
    }
}

impl FlatMatchFilterExec {
    /// Reconstruct a filter from portable document binding and decoded input.
    ///
    /// Resolved fields use the pinned dataset schema and require a complete root
    /// column of the same type. Column bindings require the named string column.
    /// No row-ID column is required. The first explicit segment supplies the
    /// tokenizer; `None` retains native index discovery/default tokenization and
    /// `Some(empty)` retains its execution-time error. Fuzzy-query admission and
    /// unresolved logical granularity errors remain in the native execution path.
    /// See [`Self`] for a reconstruction example.
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        dataset: Arc<Dataset>,
        query: MatchQuery,
        params: FtsSearchParams,
        binding: FtsFilterDocumentBinding,
        preset_segments: Option<Vec<IndexMetadata>>,
    ) -> Result<Self> {
        let input_schema = input.schema();
        let mut result = match binding {
            FtsFilterDocumentBinding::Column(column) => {
                let field = input_schema.field_with_name(&column).map_err(|_| {
                    Error::invalid_input(format!("FTS filter input is missing column '{column}'"))
                })?;
                if !matches!(field.data_type(), DataType::Utf8 | DataType::LargeUtf8) {
                    return Err(Error::invalid_input(format!(
                        "FTS filter column '{column}' must be Utf8 or LargeUtf8, got {}",
                        field.data_type()
                    )));
                }
                Self::new_with_document_column(input, dataset, query, params, column)
            }
            FtsFilterDocumentBinding::ResolvedField {
                path,
                document_granularity,
            } => {
                let resolved = resolve_fts_field(dataset.schema(), &path, document_granularity)?;
                // The root name is already resolved; literal quotes are not path syntax.
                let source = dataset
                    .schema()
                    .fields
                    .iter()
                    .find(|field| field.name == resolved.root_column)
                    .ok_or_else(|| Error::internal("Resolved FTS filter root is missing"))?;
                let field = input_schema
                    .field_with_name(&resolved.root_column)
                    .map_err(|_| {
                        Error::invalid_input(format!(
                            "FTS filter input is missing source root '{}'",
                            resolved.root_column
                        ))
                    })?;
                if field.data_type() != &source.data_type() {
                    return Err(Error::invalid_input(format!(
                        "FTS filter input root '{}' must have source type {}, got {}",
                        resolved.root_column,
                        source.data_type(),
                        field.data_type()
                    )));
                }
                Self::new_with_resolved_field(input, dataset, query, params, resolved)
            }
        };
        result.preset_segments = preset_segments;
        Ok(result)
    }

    /// The active input binding, including the actual flat-column alias or the
    /// canonical source path and resolved document boundary. See [`Self`] for
    /// reconstruction. Private traversal and token caches are not exported.
    pub fn document_binding(&self) -> FtsFilterDocumentBinding {
        match &self.resolved_field {
            Some(resolved) => FtsFilterDocumentBinding::ResolvedField {
                path: resolved.canonical_path.clone(),
                document_granularity: resolved.document_granularity,
            },
            None => FtsFilterDocumentBinding::Column(self.document_column.clone()),
        }
    }

    async fn load_tokenizer(
        dataset: &Dataset,
        column: &str,
        document_granularity: DocumentGranularity,
        metrics: &IndexMetrics,
    ) -> DataFusionResult<Box<dyn LanceTokenizer>> {
        if let Some(segments) = load_segments(dataset, column, document_granularity).await? {
            let index_meta = segments.first().ok_or_else(|| {
                DataFusionError::Execution(format!(
                    "FTS index for column {} has no segments",
                    column
                ))
            })?;
            return Ok(open_fts_segment(dataset, column, index_meta, None, metrics)
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
        Ok(open_fts_segment(dataset, column, index_meta, None, metrics)
            .await?
            .tokenizer())
    }

    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        dataset: Arc<Dataset>,
        query: MatchQuery,
        params: FtsSearchParams,
    ) -> Self {
        let document_column = query.column.clone().unwrap_or_default();
        Self::new_with_document_column(input, dataset, query, params, document_column)
    }

    pub fn new_with_document_column(
        input: Arc<dyn ExecutionPlan>,
        dataset: Arc<Dataset>,
        query: MatchQuery,
        params: FtsSearchParams,
        document_column: String,
    ) -> Self {
        Self {
            dataset,
            input,
            query,
            tokenized_query: Arc::new(OnceLock::new()),
            params,
            preset_segments: None,
            document_column,
            resolved_field: None,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    pub(crate) fn new_with_resolved_field(
        input: Arc<dyn ExecutionPlan>,
        dataset: Arc<Dataset>,
        query: MatchQuery,
        params: FtsSearchParams,
        resolved_field: ResolvedFtsField,
    ) -> Self {
        Self {
            dataset,
            input,
            query,
            tokenized_query: Arc::new(OnceLock::new()),
            params,
            preset_segments: None,
            document_column: resolved_field.root_column.clone(),
            resolved_field: Some(resolved_field),
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// See [`super::MatchQueryExec::new_with_segments`]. `FlatMatchFilterExec`
    /// uses the first segment's tokenizer; the rest are kept for caller-side
    /// bookkeeping.
    pub fn new_with_segments(
        input: Arc<dyn ExecutionPlan>,
        dataset: Arc<Dataset>,
        query: MatchQuery,
        params: FtsSearchParams,
        segments: Vec<IndexMetadata>,
    ) -> Self {
        let document_column = query.column.clone().unwrap_or_default();
        Self {
            dataset,
            input,
            query,
            tokenized_query: Arc::new(OnceLock::new()),
            params,
            preset_segments: Some(segments),
            document_column,
            resolved_field: None,
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
        operator: Operator,
    ) -> BooleanArray {
        let text_col = text_col.as_string::<O>();
        let mut predicate = BooleanBuilder::with_capacity(text_col.len());
        for idx in 0..text_col.len() {
            predicate.append_value(
                !text_col.is_null(idx)
                    && document_matches_query(
                        text_col.value(idx),
                        tokenizer,
                        query_tokens,
                        operator,
                    ),
            );
        }
        predicate.finish()
    }

    async fn build_filter_stream(
        input: SendableRecordBatchStream,
        partition: usize,
        schema: SchemaRef,
        options: FlatMatchFilterStreamOptions,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let FlatMatchFilterStreamOptions {
            dataset,
            query,
            tokenized_query,
            document_column,
            preset_segments,
            resolved_field,
            metrics_set,
        } = options;
        let metrics = Arc::new(FtsIndexMetrics::new(&metrics_set, partition));
        let column = query
            .column
            .clone()
            .ok_or(DataFusionError::Execution(format!(
                "column not set for MatchQuery {}",
                query.terms
            )))?;
        if uses_fuzzy_expansion(query.fuzziness) {
            return Err(DataFusionError::NotImplemented(format!(
                "Fuzzy MatchQuery is not supported when FTS is used as a post-filter: column={}, fuzziness={:?}",
                column, query.fuzziness
            )));
        }
        let document_granularity = resolved_field
            .as_ref()
            .map(|resolved| resolved.document_granularity)
            .or(query.document_granularity)
            .ok_or_else(|| {
                DataFusionError::Execution(
                    "MatchQuery document granularity was not resolved".to_string(),
                )
            })?;
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
            None => {
                Self::load_tokenizer(
                    &dataset,
                    &column,
                    document_granularity,
                    &metrics.index_metrics,
                )
                .await?
            }
        };
        let query_tokens = Arc::new(try_collect_query_tokens(&query.terms, &mut tokenizer)?);
        record_tokenized_query(&tokenized_query, &query_tokens);

        let baseline = BaselineMetrics::new(&metrics_set, partition);
        let elapsed_compute = baseline.elapsed_compute().clone();
        let stream = input.then(move |batch_result| {
            let column = document_column.clone();
            let query_tokens = query_tokens.clone();
            let mut tokenizer = tokenizer.box_clone();
            let elapsed_compute = elapsed_compute.clone();
            let resolved_field = resolved_field.clone();
            let query_operator = query.operator;
            async move {
                let batch = batch_result?;
                let _t = elapsed_compute.timer();
                if let Some(resolved_field) = resolved_field {
                    let documents = resolved_field
                        .documents_from_batch(&batch)
                        .map_err(DataFusionError::from)?;
                    let mut matches = vec![false; batch.num_rows()];
                    for document in documents {
                        if document_matches_query(
                            &document.text,
                            &mut tokenizer,
                            &query_tokens,
                            query_operator,
                        ) {
                            matches[document.row_index] = true;
                        }
                    }
                    let predicate = BooleanArray::from(matches);
                    return Ok(arrow::compute::filter_record_batch(&batch, &predicate)?);
                }
                let text_column = batch.column_by_name(&column).ok_or_else(|| {
                    DataFusionError::Execution(format!("Column {} not found in batch", column,))
                })?;
                let predicate = match text_column.data_type() {
                    DataType::Utf8 => {
                        Self::find_matches::<i32>(
                            text_column,
                            &mut tokenizer,
                            &query_tokens,
                            query_operator,
                        )
                    }
                    DataType::LargeUtf8 => {
                        Self::find_matches::<i64>(
                            text_column,
                            &mut tokenizer,
                            &query_tokens,
                            query_operator,
                        )
                    }
                    _ => {
                        return Err(DataFusionError::Execution(format!(
                            "FTS document column {} is not a string; nested List inputs must be expanded before filtering",
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

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
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
            tokenized_query: self.tokenized_query.clone(),
            params: self.params.clone(),
            preset_segments: self.preset_segments.clone(),
            document_column: self.document_column.clone(),
            resolved_field: self.resolved_field.clone(),
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
            FlatMatchFilterStreamOptions {
                dataset: self.dataset.clone(),
                query: self.query.clone(),
                tokenized_query: self.tokenized_query.clone(),
                document_column: self.document_column.clone(),
                preset_segments: self.preset_segments.clone(),
                resolved_field: self.resolved_field.clone(),
                metrics_set: self.metrics.clone(),
            },
        );
        let stream = stream::once(stream_fut)
            .try_flatten()
            .stream_in_current_span()
            .boxed();
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }

    fn child_stats_requests(&self, partition: Option<usize>) -> Vec<ChildStats> {
        vec![ChildStats::At(partition)]
    }

    fn statistics_from_inputs(
        &self,
        input_stats: &[Arc<Statistics>],
        _args: &StatisticsArgs,
    ) -> DataFusionResult<Arc<Statistics>> {
        Ok(Arc::clone(&input_stats[0]))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    // Drops rows that do not match the query, so a limit cannot move below it.
    fn supports_limit_pushdown(&self) -> bool {
        false
    }
}

/// Calculates the FTS score for each row in the input.
///
/// A plan codec must preserve the corpus input, the candidate filter, and the
/// document column separately. Candidates restrict the emitted rows, while
/// every corpus row still contributes to the BM25 statistics. A node publishing
/// a shared scorer must stay with its consumers; it cannot be reconstructed as
/// an independent node.
///
/// ```
/// use std::sync::Arc;
/// use datafusion::physical_plan::ExecutionPlan;
/// use lance::io::exec::fts::FlatMatchQueryExec;
///
/// fn rebuild_independent(
///     exec: &FlatMatchQueryExec,
///     corpus: Arc<dyn ExecutionPlan>,
/// ) -> Option<FlatMatchQueryExec> {
///     if exec.has_shared_scorer() {
///         return None;
///     }
///     let mut rebuilt = match exec.preset_segments() {
///         Some(segments) => FlatMatchQueryExec::new_with_segments_and_document_granularity(
///             exec.dataset().clone(), exec.query().clone(), exec.params().clone(),
///             corpus, segments.to_vec(), exec.document_granularity(),
///             exec.document_column().to_owned(),
///         ),
///         None => FlatMatchQueryExec::new_with_document_granularity(
///             exec.dataset().clone(), exec.query().clone(), exec.params().clone(),
///             corpus, exec.document_granularity(), exec.document_column().to_owned(),
///         ),
///     }.with_candidate_filter(exec.candidate_filter().clone());
///     if let Some(scorer) = exec.base_scorer() {
///         rebuilt = rebuilt.with_base_scorer(scorer.clone());
///     }
///     Some(rebuilt)
/// }
/// ```
#[derive(Debug)]
pub struct FlatMatchQueryExec {
    dataset: Arc<Dataset>,
    query: MatchQuery,
    tokenized_query: Arc<OnceLock<TokenizedQuery>>,
    params: FtsSearchParams,
    unindexed_input: Arc<dyn ExecutionPlan>,
    /// Selects emitted rows after the full input has contributed corpus statistics.
    candidate_filter: PreFilterSource,
    /// Optional override for the BM25 scorer normally built locally inside
    /// `execute()`. See [`MatchQueryExec::with_base_scorer`].
    base_scorer: Option<Arc<MemBM25Scorer>>,
    /// Publishes the scorer extended with this flat branch's documents.
    shared_scorer: Option<Arc<SharedFtsScorer>>,
    /// Optional pre-resolved segment list. See
    /// [`MatchQueryExec::new_with_segments`].
    preset_segments: Option<Vec<IndexMetadata>>,
    document_granularity: DocumentGranularity,
    document_column: String,
    schema: SchemaRef,

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
                )?;
                fmt_tokenized_query(&self.tokenized_query, ", ", f)
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "FlatMatchQuery\ncolumn={}\nquery={}",
                    self.query.column.as_deref().unwrap_or_default(),
                    self.query.terms
                )?;
                fmt_tokenized_query(&self.tokenized_query, "\n", f)
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
    ) -> Result<Self> {
        let document_column = query.column.clone().unwrap_or_default();
        let document_granularity = query.document_granularity.ok_or_else(|| {
            Error::invalid_input("MatchQuery document granularity must be resolved".to_string())
        })?;
        Ok(Self::new_with_document_granularity(
            dataset,
            query,
            params,
            unindexed_input,
            document_granularity,
            document_column,
        ))
    }

    pub fn new_with_document_granularity(
        dataset: Arc<Dataset>,
        query: MatchQuery,
        params: FtsSearchParams,
        unindexed_input: Arc<dyn ExecutionPlan>,
        document_granularity: DocumentGranularity,
        document_column: String,
    ) -> Self {
        let schema = fts_schema(document_granularity);
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self {
            dataset,
            query,
            tokenized_query: Arc::new(OnceLock::new()),
            params,
            unindexed_input,
            candidate_filter: PreFilterSource::None,
            base_scorer: None,
            shared_scorer: None,
            preset_segments: None,
            document_granularity,
            document_column,
            schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// See [`super::MatchQueryExec::new_with_segments`].
    pub fn new_with_segments(
        dataset: Arc<Dataset>,
        query: MatchQuery,
        params: FtsSearchParams,
        unindexed_input: Arc<dyn ExecutionPlan>,
        segments: Vec<IndexMetadata>,
    ) -> Result<Self> {
        let document_column = query.column.clone().unwrap_or_default();
        let document_granularity = query.document_granularity.ok_or_else(|| {
            Error::invalid_input("MatchQuery document granularity must be resolved".to_string())
        })?;
        Ok(Self::new_with_segments_and_document_granularity(
            dataset,
            query,
            params,
            unindexed_input,
            segments,
            document_granularity,
            document_column,
        ))
    }

    pub fn new_with_segments_and_document_granularity(
        dataset: Arc<Dataset>,
        query: MatchQuery,
        params: FtsSearchParams,
        unindexed_input: Arc<dyn ExecutionPlan>,
        segments: Vec<IndexMetadata>,
        document_granularity: DocumentGranularity,
        document_column: String,
    ) -> Self {
        let schema = fts_schema(document_granularity);
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self {
            dataset,
            query,
            tokenized_query: Arc::new(OnceLock::new()),
            params,
            unindexed_input,
            base_scorer: None,
            shared_scorer: None,
            preset_segments: Some(segments),
            candidate_filter: PreFilterSource::None,
            document_granularity,
            document_column,
            schema,
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
            unindexed_input: self.unindexed_input.clone(),
            candidate_filter: self.candidate_filter.clone(),
            base_scorer: self.base_scorer.clone(),
            shared_scorer: Some(scorer),
            preset_segments: self.preset_segments.clone(),
            document_granularity: self.document_granularity,
            document_column: self.document_column.clone(),
            schema: self.schema.clone(),
            properties: self.properties.clone(),
            metrics: self.metrics.clone(),
        })
    }

    /// Restrict emitted rows after collecting statistics from the complete corpus.
    ///
    /// An empty candidate set still collects corpus statistics and publishes any
    /// shared scorer. [`PreFilterSource::None`] emits every matching corpus row.
    /// See [`Self`] for an example that preserves this filter when rebuilding a plan.
    pub fn with_candidate_filter(mut self, candidate_filter: PreFilterSource) -> Self {
        self.candidate_filter = candidate_filter;
        self
    }

    /// The filter applied after corpus statistics and scores have been computed.
    ///
    /// This is separate from the corpus input. See [`Self`] for a reconstruction
    /// example; applying it to the corpus scan instead would change BM25 scores.
    pub fn candidate_filter(&self) -> &PreFilterSource {
        &self.candidate_filter
    }

    /// The input column containing documents, which may be an alias of the
    /// indexed field in [`Self::query`]. See [`Self`] for a reconstruction example.
    pub fn document_column(&self) -> &str {
        &self.document_column
    }

    /// The logical document boundary used for scoring and output coordinates.
    /// See [`Self`] for a reconstruction example that preserves this boundary.
    pub fn document_granularity(&self) -> DocumentGranularity {
        self.document_granularity
    }

    /// Whether this node publishes statistics for other nodes in the same plan.
    ///
    /// A codec or optimizer that cannot preserve that producer/consumer binding
    /// must keep the complete shared-scoring subtree intact. See [`Self`] for an
    /// example that declines independent reconstruction in this case.
    pub fn has_shared_scorer(&self) -> bool {
        self.shared_scorer.is_some()
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

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        let mut children = vec![&self.unindexed_input];
        if let Some(candidate_filter) = self.candidate_filter.execution_plan() {
            children.push(candidate_filter);
        }
        children
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        // `execute()` only reads `unindexed_input.execute(partition)` for the single
        // output partition, so the input must be coalesced to one partition. Without
        // this, EnforceDistribution may round-robin the scan across `target_partitions`
        // and only partition 0 is consumed, silently dropping the other fragments.
        vec![Distribution::SinglePartition; self.children().len()]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let expected_children = self.children().len();
        if children.len() != expected_children {
            return Err(DataFusionError::Internal(format!(
                "flat Match expected {expected_children} children, got {}",
                children.len()
            )));
        }
        let mut children = children.into_iter();
        let unindexed_input = children.next().ok_or_else(|| {
            DataFusionError::Internal("flat Match corpus input is missing".to_string())
        })?;
        let candidate_filter = match children.next() {
            Some(child) => self.candidate_filter.with_execution_plan(child)?,
            None => PreFilterSource::None,
        };
        Ok(Arc::new(Self {
            dataset: self.dataset.clone(),
            query: self.query.clone(),
            tokenized_query: self.tokenized_query.clone(),
            params: self.params.clone(),
            unindexed_input,
            candidate_filter,
            base_scorer: self.base_scorer.clone(),
            shared_scorer: self.shared_scorer.clone(),
            preset_segments: self.preset_segments.clone(),
            document_granularity: self.document_granularity,
            document_column: self.document_column.clone(),
            schema: self.schema.clone(),
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
        let tokenized_query = self.tokenized_query.clone();
        let ds = self.dataset.clone();
        let candidate_filter = self.candidate_filter.clone();
        let preset_base_scorer = self.base_scorer.clone();
        let shared_scorer_producer = self.shared_scorer.clone().map(SharedFtsScorerProducer::new);
        let preset_segments = self.preset_segments.clone();
        let metrics = Arc::new(FtsIndexMetrics::new(&self.metrics, partition));
        let metrics_clone = metrics.clone();
        let target_batch_size = context.session_config().batch_size();
        let document_granularity = self.document_granularity;
        let document_column = self.document_column.clone();
        let phrase_slop = self.params.phrase_slop;

        // Lets `flat_bm25_search_stream_with_metrics` attribute the spawn_cpu
        // tokenize work and synchronous scoring back onto this node's
        // `elapsed_compute`. Reusing the handle already inside `FtsIndexMetrics`
        // avoids registering a duplicate metric.
        let elapsed_compute = metrics.baseline_metrics.elapsed_compute().clone();

        let column = query.column.ok_or(DataFusionError::Execution(format!(
            "column not set for MatchQuery {}",
            query.terms
        )))?;
        let unindexed_input = self
            .unindexed_input
            .execute(partition, context.clone())
            .and_then(|stream| document_input(stream, &document_column).map_err(Into::into));
        let unindexed_input = match unindexed_input {
            Ok(input) => input,
            Err(error) => {
                if let Some(producer) = shared_scorer_producer {
                    producer.publish_error(&error);
                }
                return Err(error);
            }
        };

        let stream = stream::once(async move {
            let shared_scorer_producer = shared_scorer_producer;
            let result = async {
                let segments = match preset_segments {
                    Some(segments) => Some(segments),
                    None => load_segments(&ds, &column, document_granularity).await?,
                };
                let (tokenizer, base_scorer) = match segments {
                    Some(segments) => {
                        let _details = load_segment_details(&ds, &column, &segments).await?;
                        let indices =
                            open_fts_segments(&ds, &column, &segments, &metrics.index_metrics)
                                .await?;
                        metrics.record_parts_searched(
                            indices.iter().map(|index| index.partition_count()).sum(),
                        );
                        let first_index = indices.first().ok_or(DataFusionError::Execution(
                            format!("FTS index for column {} has no segments", column),
                        ))?;
                        let mut tokenizer = first_index.tokenizer();
                        let query_tokens = try_collect_query_tokens(&query.terms, &mut tokenizer)?;
                        record_tokenized_query(&tokenized_query, &query_tokens);
                        let base_scorer = match preset_base_scorer {
                            Some(scorer) => (*scorer).clone(),
                            None => {
                                let scorer_start = std::time::Instant::now();
                                let scorer = build_global_bm25_scorer(
                                    &indices,
                                    &query_tokens,
                                    &FtsSearchParams::new(),
                                    Some(metrics.as_ref()),
                                )
                                .boxed()
                                .await?;
                                metrics.record_scorer_build(scorer_start.elapsed());
                                scorer
                            }
                        };
                        (tokenizer, Some(base_scorer))
                    }
                    None => {
                        let mut tokenizer = default_text_tokenizer();
                        let query_tokens = try_collect_query_tokens(&query.terms, &mut tokenizer)?;
                        record_tokenized_query(&tokenized_query, &query_tokens);
                        (tokenizer, preset_base_scorer.map(|s| (*s).clone()))
                    }
                };

                flat_bm25_search_stream_with_options_and_scorer(
                    unindexed_input,
                    document_column,
                    query.terms,
                    tokenizer,
                    base_scorer,
                    FlatBm25SearchOptions {
                        target_batch_size,
                        elapsed_compute: Some(elapsed_compute),
                        operator: query.operator,
                        boost: query.boost,
                        document_granularity,
                        phrase_slop,
                    },
                )
                .await
            }
            .await;

            match result {
                Ok((stream, scorer)) => {
                    if let Some(producer) = shared_scorer_producer {
                        producer.publish(Arc::new(scorer));
                    }
                    // Candidate selection is deliberately inside this node: even
                    // an empty selection must not remove the corpus producer.
                    let mask = match candidate_filter {
                        PreFilterSource::FilteredRowIds(input) => {
                            Box::new(FilteredRowIdsToPrefilter::new(
                                input.execute(partition, context)?,
                            ))
                            .load()
                            .await?
                        }
                        PreFilterSource::ScalarIndexQuery(input) => {
                            Box::new(SelectionVectorToPrefilter(
                                input.execute(partition, context)?,
                            ))
                            .load()
                            .await?
                        }
                        PreFilterSource::None => return Ok(stream),
                    };
                    let schema = stream.schema();
                    let stream = stream.map(move |batch| {
                        let _timer = metrics.baseline_metrics.elapsed_compute().timer();
                        apply_mask(&mask, batch?)
                    });
                    Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream))
                        as SendableRecordBatchStream)
                }
                Err(error) => {
                    if let Some(producer) = shared_scorer_producer {
                        producer.publish_error(&error);
                    }
                    Err(error)
                }
            }
        })
        .try_flatten()
        .map(move |batch| {
            // Records output_rows, output_bytes, and output_batches on the shared
            // BaselineMetrics, as DataFusion's own FilterExec does in its
            // hand-written poll_next.
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

#[cfg(test)]
mod tests {
    mod filter_binding;
    use super::*;
    use crate::dataset::transaction::{Operation, TransactionBuilder};
    use crate::index::DatasetIndexExt;
    use crate::index::loading::DatasetIndexInternalExt;
    use crate::io::exec::fts::tests::StatsHolder;
    use crate::io::exec::utils::IndexMetrics;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
    use lance_datafusion::utils::PARTITIONS_SEARCHED_METRIC;
    use lance_index::metrics::NoOpMetricsCollector;
    use lance_index::scalar::FullTextSearchQuery;
    use lance_index::scalar::InvertedIndexParams;
    use lance_index::scalar::inverted::{InvertedIndex, Language};
    use lance_index::{IndexCriteria, IndexType};

    #[rstest::rstest]
    #[case::default(InvertedIndexParams::default(), "alpha", "beta")]
    #[case::ngram(InvertedIndexParams::default()
        .base_tokenizer("ngram".into()).ngram_min_length(2).ngram_max_length(2), "ab", "bc cd")]
    #[test]
    fn document_match_filter_respects_document_boundary(
        #[case] params: InvertedIndexParams,
        #[case] first: &str,
        #[case] rest: &str,
    ) {
        let document = format!("{first} {rest}");
        let mut tokenizer = params.build().unwrap();
        let query_tokens = try_collect_query_tokens(first, &mut tokenizer).unwrap();
        assert!(super::document_matches_query(
            &document,
            &mut tokenizer,
            &query_tokens,
            Operator::Or,
        ));

        let query_tokens = try_collect_query_tokens(&document, &mut tokenizer).unwrap();
        assert!(!super::document_matches_query(
            first,
            &mut tokenizer,
            &query_tokens,
            Operator::And,
        ));
        assert!(super::document_matches_query(
            &document,
            &mut tokenizer,
            &query_tokens,
            Operator::And,
        ));
    }

    #[test]
    fn test_flat_match_filter_find_matches_large_utf8() {
        use arrow_array::LargeStringArray;

        let mut tokenizer = default_text_tokenizer();
        let query_tokens = try_collect_query_tokens("hello", &mut tokenizer).unwrap();

        let text_col =
            LargeStringArray::from(vec!["hello world", "no match here", "say hello there"]);

        let result = FlatMatchFilterExec::find_matches::<i64>(
            &text_col,
            &mut tokenizer,
            &query_tokens,
            Operator::Or,
        );

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
        let mut tokenizer = FlatMatchFilterExec::load_tokenizer(
            &dataset,
            "text",
            DocumentGranularity::Row,
            &metrics,
        )
        .await
        .unwrap();
        let query_tokens = try_collect_query_tokens("hello", &mut tokenizer).unwrap();

        let mut tokenizer = FlatMatchFilterExec::load_tokenizer(
            &dataset,
            "text",
            DocumentGranularity::Row,
            &metrics,
        )
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
}
