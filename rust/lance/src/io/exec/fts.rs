// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

use arrow::array::{AsArray, BooleanBuilder, ListBuilder, UInt32Builder};
use arrow::datatypes::{Float32Type, UInt64Type};
use arrow_array::{Array, BooleanArray, Float32Array, OffsetSizeTrait, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, SchemaRef};
use datafusion::common::{NullEquality, Statistics};
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::empty::EmptyExec;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, Gauge, MetricsSet};
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::{Distribution, EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion_physical_plan::ExecutionPlanProperties;
use datafusion_physical_plan::joins::{HashJoinExec, PartitionMode};
use datafusion_physical_plan::metrics::{BaselineMetrics, Count, Time};
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
use super::utils::{IndexMetrics, build_prefilter, build_prefilter_restricted_to_fragments};
use crate::index::prefilter::DatasetPreFilter;
use crate::index::scalar::inverted::{
    ResolvedFtsField, flatten_fts_document_column, fts_document_schema, load_segment_details,
    load_segments, transform_fts_document_stream,
};
use crate::{Dataset, index::DatasetIndexInternalExt};
use lance_index::metrics::{
    AND_CANDIDATES_PRUNED_BEFORE_RETURN_METRIC, AND_CANDIDATES_SEEN_METRIC, AND_FULL_SCORES_METRIC,
    COMPOUND_ADDRESS_RESOLUTION_BATCHES_METRIC, COMPOUND_ADDRESSES_RESOLVED_METRIC,
    COMPOUND_PEAK_ADDRESS_RESOLUTION_BATCH_SIZE_METRIC, COMPOUND_PEAK_BUFFERED_CANDIDATES_METRIC,
    COMPOUND_SCORE_FLOOR_OVERFLOWS_METRIC, COMPOUND_SHOULD_BOUND_RECOMPUTATIONS_METRIC,
    COMPOUND_SHOULD_ESSENTIAL_EVALUATIONS_METRIC, COMPOUND_SHOULD_NON_ESSENTIAL_EVALUATIONS_METRIC,
    COMPOUND_SHOULD_SKIPPED_WINDOWS_METRIC, CROSS_COLUMN_STAGED_ATTEMPTS_METRIC,
    CROSS_COLUMN_STAGED_CANDIDATES_METRIC, CROSS_COLUMN_STAGED_FALLBACKS_METRIC,
    CROSS_COLUMN_STAGED_SUCCESSES_METRIC, FREQS_COLLECTED_METRIC, MetricsCollector,
};
use lance_index::scalar::inverted::builder::ScoredDoc;
use lance_index::scalar::inverted::builder::document_input;
use lance_index::scalar::inverted::document_tokenizer::{DocType, JsonTokenizer, LanceTokenizer};
use lance_index::scalar::inverted::query::{
    BoostQuery, CombinedFieldsQuery, FtsQuery, FtsQueryNode, FtsSearchParams, MatchQuery, Operator,
    PhraseQuery, Tokens, collect_query_tokens, has_query_token,
};
use lance_index::scalar::inverted::tokenizer::document_tokenizer::TextTokenizer;
use lance_index::scalar::inverted::{
    CombinedCorpusStats, CombinedFieldColumn, CombinedFieldsBM25Scorer, DOC_INDEX_COL,
    DocumentGranularity, FTS_SCHEMA, FlatBm25SearchOptions, InvertedIndex, MemBM25Scorer,
    SCORE_COL, build_combined_bm25_scorer, build_global_bm25_scorer, combined_fields_search,
    compound_search, compound_search_with_base_scorer, cross_column_compound_search,
    flat_bm25_search_stream_with_options_and_scorer, flat_combined_fields_search_stream,
    fts_schema, validate_combined_tokenizers,
};
use lance_index::{prefilter::PreFilter, scalar::inverted::query::BooleanQuery};
use lance_select::{RowAddrMask, RowAddrTreeMap, RowSetOps};
use lance_tokenizer::{SimpleTokenizer, TextAnalyzer};
use tracing::instrument;
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq)]
struct TokenWithPosition {
    text: String,
    position: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TokenizedQuery(Vec<TokenWithPosition>);

impl TokenizedQuery {
    fn from_tokens(tokens: &Tokens) -> Self {
        let mut token_positions = Vec::with_capacity(tokens.len());
        for index in 0..tokens.len() {
            token_positions.push(TokenWithPosition {
                text: tokens.get_token(index).to_string(),
                position: tokens.position(index),
            });
        }
        Self(token_positions)
    }
}

impl std::fmt::Display for TokenizedQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (index, token) in self.0.iter().enumerate() {
            if index > 0 {
                write!(f, ", ")?;
            }
            write!(f, "({:?}, {})", token.text, token.position)?;
        }
        write!(f, "]")
    }
}

fn record_tokenized_query(snapshot: &OnceLock<TokenizedQuery>, tokens: &Tokens) {
    snapshot.get_or_init(|| TokenizedQuery::from_tokens(tokens));
}

fn fmt_tokenized_query(
    snapshot: &OnceLock<TokenizedQuery>,
    separator: &str,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result {
    if let Some(tokens) = snapshot.get() {
        write!(f, "{separator}tokenized_query={tokens}")?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenizedLeafKind {
    Match,
    Phrase,
}

impl std::fmt::Display for TokenizedLeafKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Match => write!(f, "Match"),
            Self::Phrase => write!(f, "Phrase"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TokenizedQueryLeaf {
    kind: TokenizedLeafKind,
    column: Option<String>,
    tokens: TokenizedQuery,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TokenizedCompoundQuery(Vec<TokenizedQueryLeaf>);

impl std::fmt::Display for TokenizedCompoundQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (index, leaf) in self.0.iter().enumerate() {
            if index > 0 {
                write!(f, ", ")?;
            }
            write!(
                f,
                "{}(column={:?}, tokens={})",
                leaf.kind,
                leaf.column.as_deref().unwrap_or_default(),
                leaf.tokens
            )?;
        }
        write!(f, "]")
    }
}

fn fmt_tokenized_compound_query(
    snapshot: &OnceLock<TokenizedCompoundQuery>,
    separator: &str,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result {
    if let Some(tokens) = snapshot.get() {
        write!(f, "{separator}tokenized_query={tokens}")?;
    }
    Ok(())
}

/// Expands a schema-derived nested FTS source into one canonical row per
/// logical document before flat search or index building consumes it.
#[derive(Debug)]
pub struct FtsDocumentExec {
    input: Arc<dyn ExecutionPlan>,
    resolved: ResolvedFtsField,
    properties: Arc<PlanProperties>,
}

impl FtsDocumentExec {
    pub(crate) fn new(input: Arc<dyn ExecutionPlan>, resolved: ResolvedFtsField) -> Self {
        let schema = fts_document_schema(resolved.coordinate_rank());
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema),
            input.output_partitioning().clone(),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self {
            input,
            resolved,
            properties,
        }
    }
}

impl DisplayAs for FtsDocumentExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "FtsDocument: column={}, granularity={:?}",
            self.resolved.canonical_path, self.resolved.document_granularity
        )
    }
}

impl ExecutionPlan for FtsDocumentExec {
    fn name(&self) -> &str {
        "FtsDocumentExec"
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
                "FtsDocumentExec expects one child".to_string(),
            ));
        }
        Ok(Arc::new(Self::new(
            children.pop().unwrap(),
            self.resolved.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        transform_fts_document_stream(
            self.input.execute(partition, context)?,
            self.resolved.clone(),
        )
        .map_err(DataFusionError::from)
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }
}

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
) -> Result<Vec<ScoredDoc>> {
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
                    .bm25_search_documents(
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

    while let Some(documents) = searches.try_next().await? {
        for document in documents {
            if candidates.len() < limit {
                candidates.push(std::cmp::Reverse(document));
            } else if candidates.peek().unwrap().0.score < document.score {
                candidates.pop();
                candidates.push(std::cmp::Reverse(document));
            }
        }
    }

    Ok(candidates
        .into_sorted_vec()
        .into_iter()
        .map(|std::cmp::Reverse(document)| document)
        .collect())
}

fn scored_documents_batch(schema: SchemaRef, documents: Vec<ScoredDoc>) -> Result<RecordBatch> {
    let row_ids = UInt64Array::from_iter_values(documents.iter().map(|document| document.row_id));
    let scores = Float32Array::from_iter_values(documents.iter().map(|document| document.score.0));
    let mut columns = vec![Arc::new(row_ids) as Arc<dyn Array>];
    if schema.field_with_name(DOC_INDEX_COL).is_ok() {
        let mut builder = ListBuilder::new(UInt32Builder::new()).with_field(Field::new(
            "item",
            DataType::UInt32,
            false,
        ));
        for document in &documents {
            builder.values().append_slice(&document.doc_index);
            builder.append(true);
        }
        columns.push(Arc::new(builder.finish()));
    }
    columns.push(Arc::new(scores));
    Ok(RecordBatch::try_new(schema, columns)?)
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct DocumentKey {
    row_id: u64,
    doc_index: Vec<u32>,
}

fn batch_document_keys(batch: &RecordBatch) -> Result<Vec<DocumentKey>> {
    let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
    let doc_indices = batch
        .column_by_name(DOC_INDEX_COL)
        .map(|column| column.as_list::<i32>());
    (0..batch.num_rows())
        .map(|row| {
            let doc_index = if let Some(doc_indices) = doc_indices {
                if doc_indices.is_null(row) {
                    return Err(Error::internal(
                        "element-document FTS produced a null document coordinate".to_string(),
                    ));
                }
                doc_indices
                    .value(row)
                    .as_primitive::<arrow::datatypes::UInt32Type>()
                    .values()
                    .to_vec()
            } else {
                Vec::new()
            };
            Ok(DocumentKey {
                row_id: row_ids.value(row),
                doc_index,
            })
        })
        .collect()
}

fn batch_scored_document_keys(batch: &RecordBatch) -> Result<Vec<(DocumentKey, f32)>> {
    let keys = batch_document_keys(batch)?;
    let scores = batch[SCORE_COL].as_primitive::<Float32Type>();
    Ok(keys
        .into_iter()
        .enumerate()
        .map(|(index, key)| (key, scores.value(index)))
        .collect())
}

fn batch_scored_document_keys_sum_scores(batch: &RecordBatch) -> Result<Vec<(DocumentKey, f32)>> {
    let keys = batch_document_keys(batch)?;
    let schema = batch.schema();
    let score_columns = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, field)| field.name() == SCORE_COL)
        .map(|(index, _)| batch.column(index).as_primitive::<Float32Type>())
        .collect::<Vec<_>>();
    if score_columns.is_empty() {
        return Err(Error::internal(format!(
            "Boolean MUST result is missing required {SCORE_COL} columns"
        )));
    }
    keys.into_iter()
        .enumerate()
        .map(|(row, key)| {
            let score: f32 = score_columns.iter().map(|scores| scores.value(row)).sum();
            if !score.is_finite() {
                return Err(Error::internal(format!(
                    "Boolean MUST score sum must be finite, got {score} for row_id={}",
                    key.row_id
                )));
            }
            Ok((key, score))
        })
        .collect()
}

fn document_key_scores_batch(
    schema: SchemaRef,
    values: impl IntoIterator<Item = (DocumentKey, f32)>,
) -> Result<RecordBatch> {
    scored_documents_batch(
        schema,
        values
            .into_iter()
            .map(|(key, score)| ScoredDoc::with_doc_index(key.row_id, key.doc_index, score))
            .collect(),
    )
}

fn compare_scored_documents(
    (left_key, left_score): &(DocumentKey, f32),
    (right_key, right_score): &(DocumentKey, f32),
) -> Ordering {
    right_score
        .total_cmp(left_score)
        .then_with(|| left_key.cmp(right_key))
}

fn count_fts_leaves(query: &FtsQuery) -> usize {
    match query {
        FtsQuery::Match(_) | FtsQuery::Phrase(_) => 1,
        FtsQuery::Boost(query) => {
            count_fts_leaves(&query.positive) + count_fts_leaves(&query.negative)
        }
        FtsQuery::MultiMatch(query) => query.match_queries.len(),
        // Unreachable while `supports_compound_scorer` rejects BM25F. Counted as
        // the per-column posting scans anyway, so the arm stays honest if that
        // changes.
        FtsQuery::CombinedFields(query) => query.column_names().count(),
        FtsQuery::Boolean(query) => query
            .should
            .iter()
            .chain(&query.must)
            .chain(&query.must_not)
            .map(count_fts_leaves)
            .sum(),
    }
}

/// Return every leaf column, including prohibited Boolean leaves.
///
/// The repeated, ordered list is useful beyond the distinct set returned by
/// `FtsQueryNode::columns`: each leaf contributes its own posting-partition
/// work and must use the tokenizer and statistics of its field.
fn compound_leaf_columns(query: &FtsQuery) -> Result<Vec<&str>> {
    fn visit<'a>(query: &'a FtsQuery, columns: &mut Vec<&'a str>) -> Result<()> {
        let required_column = |column: &'a Option<String>, kind: &str| {
            column.as_deref().ok_or_else(|| {
                Error::invalid_input(format!(
                    "cross-column compound FTS {kind} leaf is missing its resolved column"
                ))
            })
        };

        match query {
            FtsQuery::Match(query) => columns.push(required_column(&query.column, "Match")?),
            FtsQuery::Phrase(query) => columns.push(required_column(&query.column, "Phrase")?),
            FtsQuery::Boost(query) => {
                visit(&query.positive, columns)?;
                visit(&query.negative, columns)?;
            }
            FtsQuery::MultiMatch(query) => {
                for query in &query.match_queries {
                    columns.push(required_column(&query.column, "MultiMatch")?);
                }
            }
            FtsQuery::Boolean(query) => {
                for query in query
                    .should
                    .iter()
                    .chain(&query.must)
                    .chain(&query.must_not)
                {
                    visit(query, columns)?;
                }
            }
            // Unreachable while `supports_compound_scorer` rejects BM25F: a
            // combined_fields leaf blends statistics across its columns, so it
            // cannot name one column per leaf the way this list requires.
            FtsQuery::CombinedFields(_) => {
                return Err(Error::invalid_input(
                    "cross-column compound FTS cannot take a combined_fields leaf".to_string(),
                ));
            }
        }
        Ok(())
    }

    let mut columns = Vec::with_capacity(count_fts_leaves(query));
    visit(query, &mut columns)?;
    Ok(columns)
}

/// One DataFusion boundary around a posting-backed compound scorer tree.
#[derive(Debug)]
pub struct CompoundQueryExec {
    dataset: Arc<Dataset>,
    query: FtsQuery,
    tokenized_query: Arc<OnceLock<TokenizedCompoundQuery>>,
    params: FtsSearchParams,
    prefilter_source: PreFilterSource,
    /// When set, leaf scorers use this instead of building one from the
    /// searched segments — see [`MatchQueryExec::with_base_scorer`].
    base_scorer: Option<Arc<MemBM25Scorer>>,
    segment_selection: FtsSegmentSelection,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl CompoundQueryExec {
    pub fn new_with_segments(
        dataset: Arc<Dataset>,
        query: FtsQuery,
        params: FtsSearchParams,
        prefilter_source: PreFilterSource,
        segments: Vec<IndexMetadata>,
    ) -> Self {
        Self::new_inner(
            dataset,
            query,
            params,
            prefilter_source,
            FtsSegmentSelection::ExactResolved(Arc::from(segments)),
        )
    }

    pub fn new_with_segment_uuids(
        dataset: Arc<Dataset>,
        query: FtsQuery,
        params: FtsSearchParams,
        prefilter_source: PreFilterSource,
        segment_uuids: Vec<Uuid>,
    ) -> Self {
        Self::new_inner(
            dataset,
            query,
            params,
            prefilter_source,
            FtsSegmentSelection::exact_uuids(segment_uuids),
        )
    }

    fn new_inner(
        dataset: Arc<Dataset>,
        query: FtsQuery,
        params: FtsSearchParams,
        prefilter_source: PreFilterSource,
        segment_selection: FtsSegmentSelection,
    ) -> Self {
        Self {
            dataset,
            query,
            tokenized_query: Arc::new(OnceLock::new()),
            params,
            prefilter_source,
            base_scorer: None,
            segment_selection,
            properties: Arc::new(PlanProperties::new(
                EquivalenceProperties::new(FTS_SCHEMA.clone()),
                Partitioning::RoundRobinBatch(1),
                EmissionType::Final,
                Boundedness::Bounded,
            )),
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// Override locally computed BM25 statistics with a corpus-wide scorer.
    ///
    /// The scorer must cover every token in every query leaf, including fuzzy
    /// expansions. Execution returns an error when any required token is absent.
    pub fn with_base_scorer(mut self, scorer: Arc<MemBM25Scorer>) -> Self {
        self.base_scorer = Some(scorer);
        self
    }

    pub fn dataset(&self) -> &Arc<Dataset> {
        &self.dataset
    }

    pub fn query(&self) -> &FtsQuery {
        &self.query
    }

    pub fn params(&self) -> &FtsSearchParams {
        &self.params
    }

    pub fn prefilter_source(&self) -> &PreFilterSource {
        &self.prefilter_source
    }

    pub fn base_scorer(&self) -> Option<&Arc<MemBM25Scorer>> {
        self.base_scorer.as_ref()
    }

    /// See [`MatchQueryExec::explicit_segment_uuids`].
    pub fn explicit_segment_uuids(&self) -> Option<Vec<Uuid>> {
        self.segment_selection.explicit_segment_uuids()
    }
}

impl DisplayAs for CompoundQueryExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "CompoundFtsScorer: query={}", self.query)?;
                fmt_tokenized_compound_query(&self.tokenized_query, ", ", f)
            }
            DisplayFormatType::TreeRender => {
                write!(f, "CompoundFtsScorer\nquery={}", self.query)?;
                fmt_tokenized_compound_query(&self.tokenized_query, "\n", f)
            }
        }
    }
}

impl ExecutionPlan for CompoundQueryExec {
    fn name(&self) -> &str {
        "CompoundQueryExec"
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        match &self.prefilter_source {
            PreFilterSource::None => vec![],
            PreFilterSource::FilteredRowIds(source) | PreFilterSource::ScalarIndexQuery(source) => {
                vec![source]
            }
        }
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        self.children()
            .iter()
            .map(|_| Distribution::SinglePartition)
            .collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let prefilter_source = match children.len() {
            0 if matches!(self.prefilter_source, PreFilterSource::None) => PreFilterSource::None,
            1 => {
                let Some(source) = children.pop() else {
                    return Err(DataFusionError::Internal(
                        "compound FTS lost its prefilter child".to_string(),
                    ));
                };
                match &self.prefilter_source {
                    PreFilterSource::FilteredRowIds(_) => PreFilterSource::FilteredRowIds(source),
                    PreFilterSource::ScalarIndexQuery(_) => {
                        PreFilterSource::ScalarIndexQuery(source)
                    }
                    PreFilterSource::None => {
                        return Err(DataFusionError::Internal(
                            "compound FTS received an unexpected prefilter child".to_string(),
                        ));
                    }
                }
            }
            count => {
                return Err(DataFusionError::Internal(format!(
                    "compound FTS expected at most one prefilter child, got {count}"
                )));
            }
        };
        Ok(Arc::new(Self {
            dataset: self.dataset.clone(),
            query: self.query.clone(),
            tokenized_query: self.tokenized_query.clone(),
            params: self.params.clone(),
            prefilter_source,
            base_scorer: self.base_scorer.clone(),
            segment_selection: self.segment_selection.clone(),
            properties: self.properties.clone(),
            metrics: ExecutionPlanMetricsSet::new(),
        }))
    }

    #[instrument(name = "compound_fts_scorer_exec", level = "debug", skip_all)]
    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let dataset = self.dataset.clone();
        let query = self.query.clone();
        let tokenized_query = self.tokenized_query.clone();
        let params = self.params.clone();
        let prefilter_source = self.prefilter_source.clone();
        let base_scorer = self.base_scorer.clone();
        let segment_selection = self.segment_selection.clone();
        let metrics = Arc::new(FtsIndexMetrics::new(&self.metrics, partition));

        let stream = stream::once(async move {
            let _timer = metrics.baseline_metrics.elapsed_compute().timer();
            let columns = query.columns();
            let column = columns.iter().next().ok_or_else(|| {
                DataFusionError::Execution(
                    "compound FTS query does not reference an indexed column".to_string(),
                )
            })?;
            if columns.len() != 1 {
                return Err(DataFusionError::Execution(
                    "posting-backed compound FTS requires exactly one column".to_string(),
                ));
            }
            let segments = segment_selection
                .resolve(
                    &dataset,
                    column,
                    DocumentGranularity::Row,
                    &metrics.segment_bind_duration,
                )
                .await?;
            let _details = load_segment_details(&dataset, column, &segments).await?;
            let indices =
                open_fts_segments(&dataset, column, &segments, &metrics.index_metrics).await?;
            if let Some(first_index) = indices.first() {
                tokenized_query
                    .get_or_init(|| tokenize_compound_query(&query, first_index.as_ref()));
            }
            let mut prefilter = build_prefilter(
                context,
                partition,
                &prefilter_source,
                dataset,
                &segments,
                None,
            )?;
            let deleted_fragments =
                indices
                    .iter()
                    .fold(roaring::RoaringBitmap::new(), |mut deleted, index| {
                        deleted |= index.deleted_fragments().clone();
                        deleted
                    });
            if !deleted_fragments.is_empty() {
                let prefilter = Arc::get_mut(&mut prefilter).ok_or_else(|| {
                    DataFusionError::Internal(
                        "compound FTS prefilter was unexpectedly shared before initialization"
                            .to_string(),
                    )
                })?;
                prefilter.set_deleted_fragments(deleted_fragments);
            }
            metrics.record_parts_searched(
                indices
                    .iter()
                    .map(|index| index.partition_count())
                    .sum::<usize>()
                    .saturating_mul(count_fts_leaves(&query)),
            );
            let (row_ids, scores) = match base_scorer {
                Some(base_scorer) => {
                    compound_search_with_base_scorer(
                        &indices,
                        &query,
                        &params,
                        prefilter,
                        metrics.clone(),
                        base_scorer,
                    )
                    .await?
                }
                None => {
                    compound_search(&indices, &query, &params, prefilter, metrics.clone()).await?
                }
            };
            metrics.baseline_metrics.record_output(row_ids.len());
            Ok::<_, DataFusionError>(RecordBatch::try_new(
                FTS_SCHEMA.clone(),
                vec![
                    Arc::new(UInt64Array::from(row_ids)),
                    Arc::new(Float32Array::from(scores)),
                ],
            )?)
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

#[derive(Debug, Clone)]
struct CompoundColumnSelection {
    column: String,
    segment_selection: FtsSegmentSelection,
}

/// One DataFusion boundary around a cross-column posting-backed scorer tree.
///
/// Each column keeps its own ordered segment selection and tokenizer. The
/// lower-level scorer joins leaves in the common row-address domain; segment
/// ordinals are deliberately never paired across columns.
#[derive(Debug)]
pub struct CrossColumnCompoundQueryExec {
    dataset: Arc<Dataset>,
    query: FtsQuery,
    tokenized_query: Arc<OnceLock<TokenizedCompoundQuery>>,
    params: FtsSearchParams,
    prefilter_source: PreFilterSource,
    columns: Arc<[CompoundColumnSelection]>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl CrossColumnCompoundQueryExec {
    pub fn new_with_segments(
        dataset: Arc<Dataset>,
        query: FtsQuery,
        params: FtsSearchParams,
        prefilter_source: PreFilterSource,
        columns: Vec<(String, Vec<IndexMetadata>)>,
    ) -> Result<Self> {
        if params.limit.is_none() {
            return Err(Error::invalid_input(
                "cross-column compound FTS requires a bounded result limit",
            ));
        }
        let leaf_columns = compound_leaf_columns(&query)?;
        let query_columns = leaf_columns.iter().copied().collect::<HashSet<_>>();
        if query_columns.len() < 2 {
            return Err(Error::invalid_input(format!(
                "cross-column compound FTS requires at least two query columns, got {}",
                query_columns.len()
            )));
        }

        let mut selected_columns = HashSet::with_capacity(columns.len());
        for (column, segments) in &columns {
            if column.is_empty() {
                return Err(Error::invalid_input(
                    "cross-column compound FTS segment selection has an empty column name",
                ));
            }
            if segments.is_empty() {
                return Err(Error::invalid_input(format!(
                    "cross-column compound FTS requires at least one segment for column {column}"
                )));
            }
            if !selected_columns.insert(column.as_str()) {
                return Err(Error::invalid_input(format!(
                    "cross-column compound FTS has duplicate segment selections for column {column}"
                )));
            }
        }

        if selected_columns != query_columns {
            let mut missing = query_columns
                .difference(&selected_columns)
                .copied()
                .collect::<Vec<_>>();
            let mut unexpected = selected_columns
                .difference(&query_columns)
                .copied()
                .collect::<Vec<_>>();
            missing.sort_unstable();
            unexpected.sort_unstable();
            return Err(Error::invalid_input(format!(
                "cross-column compound FTS segment selections do not match query leaves: \
                 missing={missing:?}, unexpected={unexpected:?}"
            )));
        }

        let columns = columns
            .into_iter()
            .map(|(column, segments)| CompoundColumnSelection {
                column,
                segment_selection: FtsSegmentSelection::ExactResolved(Arc::from(segments)),
            })
            .collect::<Vec<_>>();
        Ok(Self {
            dataset,
            query,
            tokenized_query: Arc::new(OnceLock::new()),
            params,
            prefilter_source,
            columns: Arc::from(columns),
            properties: Arc::new(PlanProperties::new(
                EquivalenceProperties::new(FTS_SCHEMA.clone()),
                Partitioning::RoundRobinBatch(1),
                EmissionType::Final,
                Boundedness::Bounded,
            )),
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    pub fn dataset(&self) -> &Arc<Dataset> {
        &self.dataset
    }

    pub fn query(&self) -> &FtsQuery {
        &self.query
    }

    pub fn params(&self) -> &FtsSearchParams {
        &self.params
    }

    pub fn prefilter_source(&self) -> &PreFilterSource {
        &self.prefilter_source
    }
}

impl DisplayAs for CrossColumnCompoundQueryExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "CrossColumnCompoundFtsScorer: query={}", self.query)?;
                fmt_tokenized_compound_query(&self.tokenized_query, ", ", f)
            }
            DisplayFormatType::TreeRender => {
                write!(f, "CrossColumnCompoundFtsScorer\nquery={}", self.query)?;
                fmt_tokenized_compound_query(&self.tokenized_query, "\n", f)
            }
        }
    }
}

impl ExecutionPlan for CrossColumnCompoundQueryExec {
    fn name(&self) -> &str {
        "CrossColumnCompoundQueryExec"
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        match &self.prefilter_source {
            PreFilterSource::None => vec![],
            PreFilterSource::FilteredRowIds(source) | PreFilterSource::ScalarIndexQuery(source) => {
                vec![source]
            }
        }
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        self.children()
            .iter()
            .map(|_| Distribution::SinglePartition)
            .collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let prefilter_source = match children.len() {
            0 if matches!(self.prefilter_source, PreFilterSource::None) => PreFilterSource::None,
            1 => {
                let Some(source) = children.pop() else {
                    return Err(DataFusionError::Internal(
                        "cross-column compound FTS lost its prefilter child".to_string(),
                    ));
                };
                match &self.prefilter_source {
                    PreFilterSource::FilteredRowIds(_) => PreFilterSource::FilteredRowIds(source),
                    PreFilterSource::ScalarIndexQuery(_) => {
                        PreFilterSource::ScalarIndexQuery(source)
                    }
                    PreFilterSource::None => {
                        return Err(DataFusionError::Internal(
                            "cross-column compound FTS received an unexpected prefilter child"
                                .to_string(),
                        ));
                    }
                }
            }
            count => {
                return Err(DataFusionError::Internal(format!(
                    "cross-column compound FTS expected at most one prefilter child, got {count}"
                )));
            }
        };

        Ok(Arc::new(Self {
            dataset: self.dataset.clone(),
            query: self.query.clone(),
            tokenized_query: self.tokenized_query.clone(),
            params: self.params.clone(),
            prefilter_source,
            columns: self.columns.clone(),
            properties: self.properties.clone(),
            metrics: ExecutionPlanMetricsSet::new(),
        }))
    }

    #[instrument(
        name = "cross_column_compound_fts_scorer_exec",
        level = "debug",
        skip_all
    )]
    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let dataset = self.dataset.clone();
        let query = self.query.clone();
        let tokenized_query = self.tokenized_query.clone();
        let params = self.params.clone();
        let prefilter_source = self.prefilter_source.clone();
        let columns = self.columns.clone();
        let metrics = Arc::new(FtsIndexMetrics::new(&self.metrics, partition));

        let stream = stream::once(async move {
            let _timer = metrics.baseline_metrics.elapsed_compute().timer();
            let selected_segments = columns
                .iter()
                .flat_map(|selection| {
                    selection
                        .segment_selection
                        .preset_segments()
                        .into_iter()
                        .flatten()
                        .cloned()
                })
                .collect::<Vec<_>>();
            if selected_segments.is_empty() {
                return Err(DataFusionError::Internal(
                    "cross-column compound FTS lost its exact segment selections".to_string(),
                ));
            }
            // DatasetPreFilter starts its deletion and filter prerequisites in
            // the background. Construct it before opening index segments so
            // both I/O paths can make progress concurrently.
            let mut prefilter = build_prefilter(
                context,
                partition,
                &prefilter_source,
                dataset.clone(),
                &selected_segments,
                None,
            )?;
            let opened_columns = try_join_all(columns.iter().cloned().map(|selection| {
                let dataset = dataset.clone();
                let metrics = metrics.clone();
                async move {
                    let segments = selection
                        .segment_selection
                        .resolve(
                            &dataset,
                            &selection.column,
                            DocumentGranularity::Row,
                            &metrics.segment_bind_duration,
                        )
                        .await?;
                    let indices = open_fts_segments(
                        &dataset,
                        &selection.column,
                        &segments,
                        &metrics.index_metrics,
                    )
                    .await?;
                    Ok::<_, DataFusionError>((selection.column, indices))
                }
            }))
            .await?;

            let mut tokenizer_indices = HashMap::with_capacity(opened_columns.len());
            let mut partition_counts = HashMap::with_capacity(opened_columns.len());
            for (column, indices) in &opened_columns {
                let first_index = indices.first().ok_or_else(|| {
                    DataFusionError::Execution(format!(
                        "cross-column compound FTS opened no segments for column {column}"
                    ))
                })?;
                tokenizer_indices.insert(column.as_str(), first_index.as_ref());
                partition_counts.insert(
                    column.as_str(),
                    indices
                        .iter()
                        .map(|index| index.partition_count())
                        .sum::<usize>(),
                );
            }
            let tokens = tokenize_cross_column_compound_query(&query, &tokenizer_indices)?;
            tokenized_query.get_or_init(|| tokens);

            let searched_parts = compound_leaf_columns(&query)?.into_iter().try_fold(
                0usize,
                |searched, column| {
                    let column_parts = partition_counts.get(column).copied().ok_or_else(|| {
                        DataFusionError::Execution(format!(
                            "cross-column compound FTS has no opened index for query column \
                             {column}"
                        ))
                    })?;
                    Ok::<_, DataFusionError>(searched.saturating_add(column_parts))
                },
            )?;
            metrics.record_parts_searched(searched_parts);

            let deleted_fragments = opened_columns.iter().flat_map(|(_, indices)| indices).fold(
                roaring::RoaringBitmap::new(),
                |mut deleted, index| {
                    deleted |= index.deleted_fragments().clone();
                    deleted
                },
            );
            if !deleted_fragments.is_empty() {
                let prefilter = Arc::get_mut(&mut prefilter).ok_or_else(|| {
                    DataFusionError::Internal(
                        "cross-column compound FTS prefilter was unexpectedly shared before \
                         initialization"
                            .to_string(),
                    )
                })?;
                prefilter.set_deleted_fragments(deleted_fragments);
            }

            let search_columns = opened_columns;
            let (row_ids, scores) = cross_column_compound_search(
                &search_columns,
                &query,
                &params,
                prefilter,
                metrics.clone(),
            )
            .await?;
            metrics.baseline_metrics.record_output(row_ids.len());
            Ok::<_, DataFusionError>(RecordBatch::try_new(
                FTS_SCHEMA.clone(),
                vec![
                    Arc::new(UInt64Array::from(row_ids)),
                    Arc::new(Float32Array::from(scores)),
                ],
            )?)
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

/// Fall back to the default simple tokenizer when no on-disk FTS segment exists.
fn default_text_tokenizer() -> Box<dyn LanceTokenizer> {
    Box::new(TextTokenizer::new(
        TextAnalyzer::builder(SimpleTokenizer::default()).build(),
    ))
}

fn tokenizer_for_match_query(
    index: &InvertedIndex,
    fuzziness: Option<u32>,
) -> Box<dyn LanceTokenizer> {
    if !matches!(fuzziness, Some(distance) if distance != 0) {
        return index.tokenizer();
    }

    let analyzer = TextAnalyzer::from(SimpleTokenizer::default());
    match index.tokenizer().doc_type() {
        DocType::Text => Box::new(TextTokenizer::new(analyzer)),
        DocType::Json => Box::new(JsonTokenizer::new(analyzer)),
    }
}

fn tokenize_compound_query(query: &FtsQuery, index: &InvertedIndex) -> TokenizedCompoundQuery {
    fn visit(query: &FtsQuery, index: &InvertedIndex, leaves: &mut Vec<TokenizedQueryLeaf>) {
        match query {
            FtsQuery::Match(query) => {
                let mut tokenizer = tokenizer_for_match_query(index, query.fuzziness);
                let tokens = collect_query_tokens(&query.terms, &mut tokenizer);
                leaves.push(TokenizedQueryLeaf {
                    kind: TokenizedLeafKind::Match,
                    column: query.column.clone(),
                    tokens: TokenizedQuery::from_tokens(&tokens),
                });
            }
            FtsQuery::Phrase(query) => {
                let mut tokenizer = index.tokenizer();
                let tokens = collect_query_tokens(&query.terms, &mut tokenizer);
                leaves.push(TokenizedQueryLeaf {
                    kind: TokenizedLeafKind::Phrase,
                    column: query.column.clone(),
                    tokens: TokenizedQuery::from_tokens(&tokens),
                });
            }
            FtsQuery::Boost(query) => {
                visit(&query.positive, index, leaves);
                visit(&query.negative, index, leaves);
            }
            FtsQuery::MultiMatch(query) => {
                for query in &query.match_queries {
                    let mut tokenizer = tokenizer_for_match_query(index, query.fuzziness);
                    let tokens = collect_query_tokens(&query.terms, &mut tokenizer);
                    leaves.push(TokenizedQueryLeaf {
                        kind: TokenizedLeafKind::Match,
                        column: query.column.clone(),
                        tokens: TokenizedQuery::from_tokens(&tokens),
                    });
                }
            }
            // Unreachable: only `CompoundQueryExec` tokenizes this way, and
            // `supports_compound_scorer` rejects BM25F at any depth. The
            // combined_fields execs snapshot their own tokens.
            FtsQuery::CombinedFields(_) => {}
            FtsQuery::Boolean(query) => {
                for query in query
                    .should
                    .iter()
                    .chain(&query.must)
                    .chain(&query.must_not)
                {
                    visit(query, index, leaves);
                }
            }
        }
    }

    let mut leaves = Vec::with_capacity(count_fts_leaves(query));
    visit(query, index, &mut leaves);
    TokenizedCompoundQuery(leaves)
}

fn tokenize_cross_column_compound_query(
    query: &FtsQuery,
    indices: &HashMap<&str, &InvertedIndex>,
) -> Result<TokenizedCompoundQuery> {
    fn index_for_leaf<'a>(
        column: Option<&str>,
        kind: &str,
        indices: &HashMap<&str, &'a InvertedIndex>,
    ) -> Result<(&'a InvertedIndex, String)> {
        let column = column.ok_or_else(|| {
            Error::invalid_input(format!(
                "cross-column compound FTS {kind} leaf is missing its resolved column"
            ))
        })?;
        let index = indices.get(column).copied().ok_or_else(|| {
            Error::invalid_input(format!(
                "cross-column compound FTS has no opened index for {kind} column {column}"
            ))
        })?;
        Ok((index, column.to_string()))
    }

    fn visit(
        query: &FtsQuery,
        indices: &HashMap<&str, &InvertedIndex>,
        leaves: &mut Vec<TokenizedQueryLeaf>,
    ) -> Result<()> {
        match query {
            FtsQuery::Match(query) => {
                let (index, column) = index_for_leaf(query.column.as_deref(), "Match", indices)?;
                let mut tokenizer = tokenizer_for_match_query(index, query.fuzziness);
                let tokens = collect_query_tokens(&query.terms, &mut tokenizer);
                leaves.push(TokenizedQueryLeaf {
                    kind: TokenizedLeafKind::Match,
                    column: Some(column),
                    tokens: TokenizedQuery::from_tokens(&tokens),
                });
            }
            FtsQuery::Phrase(query) => {
                let (index, column) = index_for_leaf(query.column.as_deref(), "Phrase", indices)?;
                let mut tokenizer = index.tokenizer();
                let tokens = collect_query_tokens(&query.terms, &mut tokenizer);
                leaves.push(TokenizedQueryLeaf {
                    kind: TokenizedLeafKind::Phrase,
                    column: Some(column),
                    tokens: TokenizedQuery::from_tokens(&tokens),
                });
            }
            FtsQuery::Boost(query) => {
                visit(&query.positive, indices, leaves)?;
                visit(&query.negative, indices, leaves)?;
            }
            FtsQuery::MultiMatch(query) => {
                for query in &query.match_queries {
                    let (index, column) =
                        index_for_leaf(query.column.as_deref(), "MultiMatch", indices)?;
                    let mut tokenizer = tokenizer_for_match_query(index, query.fuzziness);
                    let tokens = collect_query_tokens(&query.terms, &mut tokenizer);
                    leaves.push(TokenizedQueryLeaf {
                        kind: TokenizedLeafKind::Match,
                        column: Some(column),
                        tokens: TokenizedQuery::from_tokens(&tokens),
                    });
                }
            }
            FtsQuery::Boolean(query) => {
                for query in query
                    .should
                    .iter()
                    .chain(&query.must)
                    .chain(&query.must_not)
                {
                    visit(query, indices, leaves)?;
                }
            }
            // Unreachable while `supports_compound_scorer` rejects BM25F; see
            // `compound_leaf_columns`.
            FtsQuery::CombinedFields(_) => {
                return Err(Error::invalid_input(
                    "cross-column compound FTS cannot take a combined_fields leaf".to_string(),
                ));
            }
        }
        Ok(())
    }

    let mut leaves = Vec::with_capacity(count_fts_leaves(query));
    visit(query, indices, &mut leaves)?;
    Ok(TokenizedCompoundQuery(leaves))
}

type SharedScorerResult<S> = std::result::Result<Arc<S>, Arc<str>>;

/// Coordinates BM25 corpus statistics between the indexed and flat branches
/// of a mixed search. The flat branch extends the indexed statistics with the
/// unindexed documents, then publishes the resulting corpus-wide scorer.
///
/// Generic over the scorer so the single-column path can share a
/// [`MemBM25Scorer`] and `combined_fields` a [`CombinedFieldsBM25Scorer`]; the
/// coordination is the same either way.
#[derive(Debug)]
pub(crate) struct SharedFtsScorer<S = MemBM25Scorer> {
    sender: tokio::sync::watch::Sender<Option<SharedScorerResult<S>>>,
}

impl<S: Send + Sync + 'static> SharedFtsScorer<S> {
    pub(crate) fn new() -> Self {
        let (sender, _) = tokio::sync::watch::channel(None);
        Self { sender }
    }

    fn publish(&self, scorer: Arc<S>) {
        self.sender.send_replace(Some(Ok(scorer)));
    }

    fn publish_error(&self, error: &DataFusionError) {
        self.sender
            .send_replace(Some(Err(Arc::from(error.to_string()))));
    }

    async fn wait(&self) -> DataFusionResult<Arc<S>> {
        let mut receiver = self.sender.subscribe();
        loop {
            let result = receiver.borrow_and_update().clone();
            if let Some(result) = result {
                return result.map_err(|message| DataFusionError::Execution(message.to_string()));
            }
            receiver.changed().await.map_err(|_| {
                DataFusionError::Execution(
                    "mixed FTS corpus scorer producer stopped before publishing statistics"
                        .to_string(),
                )
            })?;
        }
    }
}

struct SharedFtsScorerProducer<S: Send + Sync + 'static = MemBM25Scorer> {
    scorer: Arc<SharedFtsScorer<S>>,
    completed: bool,
}

impl<S: Send + Sync + 'static> SharedFtsScorerProducer<S> {
    fn new(scorer: Arc<SharedFtsScorer<S>>) -> Self {
        Self {
            scorer,
            completed: false,
        }
    }

    fn publish(mut self, scorer: Arc<S>) {
        self.scorer.publish(scorer);
        self.completed = true;
    }

    fn publish_error(mut self, error: &DataFusionError) {
        self.scorer.publish_error(error);
        self.completed = true;
    }
}

impl<S: Send + Sync + 'static> Drop for SharedFtsScorerProducer<S> {
    fn drop(&mut self) {
        if !self.completed {
            self.scorer.sender.send_replace(Some(Err(Arc::from(
                "mixed FTS corpus scorer producer was cancelled before publishing statistics",
            ))));
        }
    }
}

/// Time spent resolving an exact ordered UUID selection to committed FTS segments.
pub const FTS_SEGMENT_BIND_DURATION_METRIC: &str = "fts_segment_bind_duration";

#[derive(Debug, Clone)]
enum FtsSegmentSelection {
    AllCommitted,
    ExactResolved(Arc<[IndexMetadata]>),
    ExactUuids(Arc<[Uuid]>),
}

impl FtsSegmentSelection {
    fn exact_uuids(mut uuids: Vec<Uuid>) -> Self {
        let mut seen = HashSet::with_capacity(uuids.len());
        uuids.retain(|uuid| seen.insert(*uuid));
        Self::ExactUuids(Arc::from(uuids))
    }

    fn preset_segments(&self) -> Option<&[IndexMetadata]> {
        match self {
            Self::ExactResolved(segments) => Some(segments),
            Self::AllCommitted | Self::ExactUuids(_) => None,
        }
    }

    fn explicit_segment_uuids(&self) -> Option<Vec<Uuid>> {
        match self {
            Self::AllCommitted => None,
            Self::ExactResolved(segments) => {
                Some(segments.iter().map(|segment| segment.uuid).collect())
            }
            Self::ExactUuids(uuids) => Some(uuids.to_vec()),
        }
    }

    async fn resolve(
        &self,
        dataset: &Dataset,
        column: &str,
        document_granularity: DocumentGranularity,
        segment_bind_duration: &Time,
    ) -> DataFusionResult<Arc<[IndexMetadata]>> {
        let segments = match self {
            Self::AllCommitted => load_segments(dataset, column, document_granularity)
                .await?
                .map(Arc::from)
                .ok_or_else(|| {
                    DataFusionError::Execution(format!(
                        "No Inverted index found for column {}",
                        column,
                    ))
                }),
            Self::ExactResolved(segments) => Ok(segments.clone()),
            Self::ExactUuids(uuids) => {
                let _timer = segment_bind_duration.timer();
                let dataset_version = dataset.version_id();
                if uuids.is_empty() {
                    return Err(DataFusionError::Execution(format!(
                        "Exact FTS segment selection for column {} at dataset version {} \
                         requires at least one segment UUID",
                        column, dataset_version
                    )));
                }

                let committed_segments = load_segments(dataset, column, document_granularity)
                    .await?
                    .ok_or_else(|| {
                        DataFusionError::Execution(format!(
                            "Cannot resolve exact FTS segment selection for column {} at dataset \
                             version {}: no Inverted index found",
                            column, dataset_version
                        ))
                    })?;
                let mut segments_by_uuid = HashMap::with_capacity(committed_segments.len());
                for segment in committed_segments {
                    let uuid = segment.uuid;
                    if segments_by_uuid.insert(uuid, segment).is_some() {
                        return Err(DataFusionError::Execution(format!(
                            "FTS metadata for column {} at dataset version {} contains duplicate \
                             segment UUID {}",
                            column, dataset_version, uuid
                        )));
                    }
                }

                let mut resolved = Vec::with_capacity(uuids.len());
                for uuid in uuids.iter() {
                    let segment = segments_by_uuid.get(uuid).ok_or_else(|| {
                        DataFusionError::Execution(format!(
                            "Requested FTS segment UUID {} for column {} is not committed in \
                             dataset version {}",
                            uuid, column, dataset_version
                        ))
                    })?;
                    resolved.push(segment.clone());
                }
                Ok(Arc::from(resolved))
            }
        }?;
        let details = load_segment_details(dataset, column, &segments).await?;
        let indexed_granularity = DocumentGranularity::try_from(details.document_granularity)?;
        if indexed_granularity != document_granularity {
            return Err(DataFusionError::Execution(format!(
                "FTS segments selected for column {column} use {indexed_granularity:?} document \
                 granularity, but the query was resolved as {document_granularity:?}"
            )));
        }
        Ok(segments)
    }
}

pub struct FtsIndexMetrics {
    index_metrics: IndexMetrics,
    partitions_searched: Count,
    and_candidates_seen: Count,
    and_candidates_pruned_before_return: Count,
    and_full_scores: Count,
    freqs_collected: Count,
    compound_addresses_resolved: Count,
    compound_address_resolution_batches: Count,
    compound_peak_address_resolution_batch_size: Gauge,
    compound_score_floor_overflows: Count,
    compound_peak_buffered_candidates: Gauge,
    compound_should_skipped_windows: Count,
    compound_should_bound_recomputations: Count,
    compound_should_essential_evaluations: Count,
    compound_should_non_essential_evaluations: Count,
    cross_column_staged_attempts: Count,
    cross_column_staged_successes: Count,
    cross_column_staged_fallbacks: Count,
    cross_column_staged_candidates: Count,
    /// Wall time (ms) of the exec-local `build_global_bm25_scorer`
    /// fallback; zero when a preset base scorer was injected.
    scorer_build_ms: Gauge,
    segment_bind_duration: Time,
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
            compound_addresses_resolved: metrics
                .new_count(COMPOUND_ADDRESSES_RESOLVED_METRIC, partition),
            compound_address_resolution_batches: metrics
                .new_count(COMPOUND_ADDRESS_RESOLUTION_BATCHES_METRIC, partition),
            compound_peak_address_resolution_batch_size: metrics.new_gauge(
                COMPOUND_PEAK_ADDRESS_RESOLUTION_BATCH_SIZE_METRIC,
                partition,
            ),
            compound_score_floor_overflows: metrics
                .new_count(COMPOUND_SCORE_FLOOR_OVERFLOWS_METRIC, partition),
            compound_peak_buffered_candidates: metrics
                .new_gauge(COMPOUND_PEAK_BUFFERED_CANDIDATES_METRIC, partition),
            compound_should_skipped_windows: metrics
                .new_count(COMPOUND_SHOULD_SKIPPED_WINDOWS_METRIC, partition),
            compound_should_bound_recomputations: metrics
                .new_count(COMPOUND_SHOULD_BOUND_RECOMPUTATIONS_METRIC, partition),
            compound_should_essential_evaluations: metrics
                .new_count(COMPOUND_SHOULD_ESSENTIAL_EVALUATIONS_METRIC, partition),
            compound_should_non_essential_evaluations: metrics
                .new_count(COMPOUND_SHOULD_NON_ESSENTIAL_EVALUATIONS_METRIC, partition),
            cross_column_staged_attempts: metrics
                .new_count(CROSS_COLUMN_STAGED_ATTEMPTS_METRIC, partition),
            cross_column_staged_successes: metrics
                .new_count(CROSS_COLUMN_STAGED_SUCCESSES_METRIC, partition),
            cross_column_staged_fallbacks: metrics
                .new_count(CROSS_COLUMN_STAGED_FALLBACKS_METRIC, partition),
            cross_column_staged_candidates: metrics
                .new_count(CROSS_COLUMN_STAGED_CANDIDATES_METRIC, partition),
            scorer_build_ms: metrics.new_gauge("scorer_build_ms", partition),
            segment_bind_duration: metrics.new_time(FTS_SEGMENT_BIND_DURATION_METRIC, partition),
            baseline_metrics: BaselineMetrics::new(metrics, partition),
        }
    }

    pub fn record_parts_searched(&self, num_parts: usize) {
        self.partitions_searched.add(num_parts);
    }

    pub fn record_scorer_build(&self, elapsed: std::time::Duration) {
        self.scorer_build_ms.set(elapsed.as_millis() as usize);
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

    fn record_index_cache_hits(&self, num_hits: usize) {
        self.index_metrics.record_index_cache_hits(num_hits);
    }

    fn record_index_cache_misses(&self, num_misses: usize) {
        self.index_metrics.record_index_cache_misses(num_misses);
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

    fn record_compound_addresses_resolved(&self, num_addresses: usize) {
        self.compound_addresses_resolved.add(num_addresses);
    }

    fn record_compound_address_resolution_batches(&self, num_batches: usize) {
        self.compound_address_resolution_batches.add(num_batches);
    }

    fn record_compound_peak_address_resolution_batch_size(&self, num_addresses: usize) {
        self.compound_peak_address_resolution_batch_size
            .set_max(num_addresses);
    }

    fn record_compound_score_floor_overflows(&self, num_overflows: usize) {
        self.compound_score_floor_overflows.add(num_overflows);
    }

    fn record_compound_peak_buffered_candidates(&self, num_candidates: usize) {
        self.compound_peak_buffered_candidates
            .set_max(num_candidates);
    }

    fn record_compound_should_skipped_windows(&self, num_windows: usize) {
        self.compound_should_skipped_windows.add(num_windows);
    }

    fn record_compound_should_bound_recomputations(&self, num_recomputations: usize) {
        self.compound_should_bound_recomputations
            .add(num_recomputations);
    }

    fn record_compound_should_essential_evaluations(&self, num_evaluations: usize) {
        self.compound_should_essential_evaluations
            .add(num_evaluations);
    }

    fn record_compound_should_non_essential_evaluations(&self, num_evaluations: usize) {
        self.compound_should_non_essential_evaluations
            .add(num_evaluations);
    }

    fn record_cross_column_staged_attempts(&self, num_attempts: usize) {
        self.cross_column_staged_attempts.add(num_attempts);
    }

    fn record_cross_column_staged_successes(&self, num_successes: usize) {
        self.cross_column_staged_successes.add(num_successes);
    }

    fn record_cross_column_staged_fallbacks(&self, num_fallbacks: usize) {
        self.cross_column_staged_fallbacks.add(num_fallbacks);
    }

    fn record_cross_column_staged_candidates(&self, num_candidates: usize) {
        self.cross_column_staged_candidates.add(num_candidates);
    }
}

#[derive(Debug)]
pub struct MatchQueryExec {
    dataset: Arc<Dataset>,
    query: MatchQuery,
    tokenized_query: Arc<OnceLock<TokenizedQuery>>,
    params: FtsSearchParams,
    prefilter_source: PreFilterSource,
    /// When set, `execute()` skips `build_global_bm25_scorer` and threads this
    /// scorer down to `InvertedIndex::bm25_search`.
    base_scorer: Option<Arc<MemBM25Scorer>>,
    /// Corpus-wide scorer published by the flat branch of a mixed search.
    shared_scorer: Option<Arc<SharedFtsScorer>>,
    segment_selection: FtsSegmentSelection,
    /// Rows whose indexed values were superseded by newer data overlays.
    overlay_block: Option<RowAddrMask>,
    document_granularity: DocumentGranularity,
    schema: SchemaRef,

    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl DisplayAs for MatchQueryExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "MatchQuery: column={}, query=[{}]",
                    self.query.column.as_deref().unwrap_or_default(),
                    self.query.terms
                )?;
                fmt_tokenized_query(&self.tokenized_query, ", ", f)
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "MatchQuery\ncolumn={}\nquery={}",
                    self.query.column.as_deref().unwrap_or_default(),
                    self.query.terms
                )?;
                fmt_tokenized_query(&self.tokenized_query, "\n", f)
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
    ) -> Result<Self> {
        let document_granularity = query.document_granularity.ok_or_else(|| {
            Error::invalid_input("MatchQuery document granularity must be resolved".to_string())
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
        query: MatchQuery,
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
        let params = Self::effective_params(&query, params);
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
    ) -> Result<Self> {
        let document_granularity = query.document_granularity.ok_or_else(|| {
            Error::invalid_input("MatchQuery document granularity must be resolved".to_string())
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
        query: MatchQuery,
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
        let params = Self::effective_params(&query, params);
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
            document_granularity,
            schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// Construct a `MatchQueryExec` bound to an exact ordered set of committed
    /// FTS segment UUIDs.
    ///
    /// The UUIDs are resolved from this exec's dataset snapshot when the output
    /// stream is polled. Duplicate UUIDs are removed while preserving their
    /// first-occurrence order. Resolution fails if the list is empty or any UUID
    /// is not committed for the query column.
    pub fn new_with_segment_uuids(
        dataset: Arc<Dataset>,
        query: MatchQuery,
        params: FtsSearchParams,
        prefilter_source: PreFilterSource,
        segment_uuids: Vec<Uuid>,
    ) -> Result<Self> {
        let document_granularity = query.document_granularity.ok_or_else(|| {
            Error::invalid_input("MatchQuery document granularity must be resolved".to_string())
        })?;
        let schema = fts_schema(document_granularity);
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        let params = Self::effective_params(&query, params);
        Ok(Self {
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
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
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

    pub(crate) fn with_shared_scorer(mut self, scorer: Arc<SharedFtsScorer>) -> Self {
        self.shared_scorer = Some(scorer);
        self
    }

    /// Exclude rows whose indexed text was superseded by a newer data overlay.
    pub(crate) fn with_overlay_block(mut self, overlay_block: RowAddrMask) -> Self {
        self.overlay_block = Some(overlay_block);
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

impl ExecutionPlan for MatchQueryExec {
    fn name(&self) -> &str {
        "MatchQueryExec"
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
                    tokenized_query: self.tokenized_query.clone(),
                    params: self.params.clone(),
                    prefilter_source: PreFilterSource::None,
                    base_scorer: self.base_scorer.clone(),
                    shared_scorer: self.shared_scorer.clone(),
                    segment_selection: self.segment_selection.clone(),
                    overlay_block: self.overlay_block.clone(),
                    document_granularity: self.document_granularity,
                    schema: self.schema.clone(),
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
                    tokenized_query: self.tokenized_query.clone(),
                    params: self.params.clone(),
                    prefilter_source,
                    base_scorer: self.base_scorer.clone(),
                    shared_scorer: self.shared_scorer.clone(),
                    segment_selection: self.segment_selection.clone(),
                    overlay_block: self.overlay_block.clone(),
                    document_granularity: self.document_granularity,
                    schema: self.schema.clone(),
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
        let tokenized_query = self.tokenized_query.clone();
        let params = self.params.clone();
        let ds = self.dataset.clone();
        let prefilter_source = self.prefilter_source.clone();
        let preset_base_scorer = self.base_scorer.clone();
        let shared_scorer = self.shared_scorer.clone();
        let segment_selection = self.segment_selection.clone();
        let overlay_block = self.overlay_block.clone();
        let document_granularity = self.document_granularity;
        let schema = self.schema.clone();
        let metrics = Arc::new(FtsIndexMetrics::new(&self.metrics, partition));
        let column = query.column.ok_or(DataFusionError::Execution(format!(
            "column not set for MatchQuery {}",
            query.terms
        )))?;
        let stream = stream::once(async move {
            let _timer = metrics.baseline_metrics.elapsed_compute().timer();
            let segments = segment_selection
                .resolve(
                    &ds,
                    &column,
                    document_granularity,
                    &metrics.segment_bind_duration,
                )
                .await?;
            let indices =
                open_fts_segments(&ds, &column, &segments, &metrics.index_metrics).await?;

            let mut pre_filter = build_prefilter(
                context.clone(),
                partition,
                &prefilter_source,
                ds,
                &segments,
                overlay_block,
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
            let mut tokenizer = tokenizer_for_match_query(first_index, query.fuzziness);
            let tokens = collect_query_tokens(&query.terms, &mut tokenizer);
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
            let tokens = Arc::new(tokens);
            let params = Arc::new(params);
            let mut documents = search_segments(
                &indices,
                tokens,
                params,
                query.operator,
                pre_filter,
                metrics.clone(),
                base_scorer,
            )
            .await?;
            documents.iter_mut().for_each(|document| {
                document.score.0 *= query.boost;
            });
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

/// Cross-field BM25F full-text search (`combined_fields`).
///
/// Unlike a `MultiMatch` plan (one [`MatchQueryExec`] per column fused by a
/// `max` aggregate, i.e. `best_fields`), this is a single node: it opens every
/// target column's segments, blends their corpus statistics, and runs one merged
/// scan so term statistics are shared across fields. Per-column boosts are baked
/// into the blended term frequency, so there is no post-scan boost multiplier or
/// `AggregateExec(max)`. Emits `(ROW_ID, SCORE)` in [`FTS_SCHEMA`].
#[derive(Debug)]
pub struct CombinedFieldsQueryExec {
    dataset: Arc<Dataset>,
    query: CombinedFieldsQuery,
    /// Tokens `execute()` actually produced, for `analyze_plan`. One snapshot
    /// covers every target column because `validate_combined_tokenizers`
    /// requires them to share a tokenizer, so the scan tokenizes once.
    tokenized_query: Arc<OnceLock<TokenizedQuery>>,
    params: FtsSearchParams,
    prefilter_source: PreFilterSource,
    /// When set, `execute()` skips `build_combined_bm25_scorer` and threads this
    /// blended scorer down to the merged scan (distributed corpus-global stats).
    base_scorer: Option<Arc<CombinedFieldsBM25Scorer>>,
    /// Waits for the flat sibling's blended scorer; see
    /// [`Self::with_shared_scorer`].
    shared_scorer: Option<Arc<SharedFtsScorer<CombinedFieldsBM25Scorer>>>,
    /// When set, restrict this scan to exactly these fragments: the ones every
    /// target column's index covers. See [`Self::with_covered_fragments`].
    covered_fragments: Option<roaring::RoaringBitmap>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl CombinedFieldsQueryExec {
    pub fn new(
        dataset: Arc<Dataset>,
        query: CombinedFieldsQuery,
        params: FtsSearchParams,
        prefilter_source: PreFilterSource,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(FTS_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Self {
            dataset,
            query,
            tokenized_query: Arc::new(OnceLock::new()),
            params,
            prefilter_source,
            base_scorer: None,
            shared_scorer: None,
            covered_fragments: None,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// Restrict this scan to `fragments`, the ones every target column's index
    /// covers.
    ///
    /// A BM25F score is only complete when every target column's index covers the
    /// row's fragment: `dl'` sums each column's document length, and the per-column
    /// lookup returns 0 for a row that column's index holds no document for, so a
    /// fragment indexed for some target columns but not others would score with a
    /// partial `tf'`/`dl'`. The planner routes those fragments to the flat scan and
    /// passes the remainder here.
    ///
    /// This has to be an allow list rather than a fragment block list: the inverted
    /// index trains with `_rowid`, which is a logical stable row id when the dataset
    /// uses stable row ids, so a fragment-address block list would match nothing and
    /// both sides of the union would emit the same row.
    pub fn with_covered_fragments(mut self, fragments: roaring::RoaringBitmap) -> Self {
        self.covered_fragments = Some(fragments);
        self
    }

    /// Override the blended BM25F scorer used by `execute()`. When set, the
    /// local `build_combined_bm25_scorer` call is skipped. Mirrors
    /// [`MatchQueryExec::with_base_scorer`] for distributed queries that
    /// aggregate cross-column corpus statistics out-of-band.
    pub fn with_base_scorer(mut self, scorer: Arc<CombinedFieldsBM25Scorer>) -> Self {
        self.base_scorer = Some(scorer);
        self
    }

    /// Score against the blended scorer the flat sibling publishes instead of
    /// building one from this side's index statistics alone.
    ///
    /// Only the flat scan sees the rows no index covers, so only it can produce
    /// the statistics that describe the whole scanned corpus. Without this the two
    /// children of a mixed plan score against different `docCount'`/`docFreq'`/
    /// `avgdl'`, which makes their scores incomparable and the union's sort wrong.
    pub(crate) fn with_shared_scorer(
        mut self,
        scorer: Arc<SharedFtsScorer<CombinedFieldsBM25Scorer>>,
    ) -> Self {
        self.shared_scorer = Some(scorer);
        self
    }

    pub fn query(&self) -> &CombinedFieldsQuery {
        &self.query
    }

    pub fn params(&self) -> &FtsSearchParams {
        &self.params
    }

    pub fn prefilter_source(&self) -> &PreFilterSource {
        &self.prefilter_source
    }

    fn clone_with_prefilter_source(&self, prefilter_source: PreFilterSource) -> Self {
        Self {
            dataset: self.dataset.clone(),
            query: self.query.clone(),
            tokenized_query: self.tokenized_query.clone(),
            params: self.params.clone(),
            prefilter_source,
            base_scorer: self.base_scorer.clone(),
            shared_scorer: self.shared_scorer.clone(),
            covered_fragments: self.covered_fragments.clone(),
            properties: self.properties.clone(),
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl DisplayAs for CombinedFieldsQueryExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let columns = self.query.column_names().collect::<Vec<_>>().join(", ");
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "CombinedFieldsQuery: columns=[{}], query={}",
                    columns,
                    self.query.terms()
                )?;
                fmt_tokenized_query(&self.tokenized_query, ", ", f)
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "CombinedFieldsQuery\ncolumns=[{}]\nquery={}",
                    columns,
                    self.query.terms()
                )?;
                fmt_tokenized_query(&self.tokenized_query, "\n", f)
            }
        }
    }
}

/// Which target columns of a `combined_fields` scan need an inverted index.
#[derive(Debug, Clone, Copy)]
enum CombinedFieldsIndexRequirement {
    /// Every column, because the indexed side scores each of them from its
    /// index.
    EveryColumn,
    /// At least one column, for the tokenizer. The flat side reads the document
    /// text from the scan, so a column without an index only misses out on
    /// contributing corpus statistics.
    AtLeastOneColumn,
}

/// What [`open_combined_fields_scan`] hands back to either `combined_fields` exec.
struct CombinedFieldsScan {
    columns: Vec<CombinedFieldColumn>,
    /// Every segment opened across all target columns, for the indexed side's
    /// prefilter.
    segments: Vec<IndexMetadata>,
    tokens: Tokens,
    tokenizer: Box<dyn LanceTokenizer>,
}

/// Open every target column's segments, pair each with its boost, and tokenize
/// the query once.
///
/// Always every committed segment at row granularity: BM25F blends the target
/// columns per row, and query planning rejects a column that only has a
/// list-element index. The query stores each column together with its weight, so
/// no column can be dropped here for want of one.
///
/// Both sides count the opened segments toward parts searched: the flat side needs
/// them for the tokenizer and the blended corpus statistics, the indexed side for
/// scoring.
async fn open_combined_fields_scan(
    dataset: &Dataset,
    query: &CombinedFieldsQuery,
    index_requirement: CombinedFieldsIndexRequirement,
    tokenized_query: &OnceLock<TokenizedQuery>,
    metrics: &FtsIndexMetrics,
) -> DataFusionResult<CombinedFieldsScan> {
    let mut columns = Vec::with_capacity(query.column_names().len());
    let mut all_segments = Vec::new();
    for (column, weight) in query.weighted_columns() {
        let segments = match index_requirement {
            CombinedFieldsIndexRequirement::EveryColumn => Some(
                FtsSegmentSelection::AllCommitted
                    .resolve(
                        dataset,
                        column,
                        DocumentGranularity::Row,
                        &metrics.segment_bind_duration,
                    )
                    .await?,
            ),
            CombinedFieldsIndexRequirement::AtLeastOneColumn => {
                load_segments(dataset, column, DocumentGranularity::Row)
                    .await?
                    .map(Arc::from)
            }
        };
        let indices = match segments {
            Some(segments) => {
                let _details = load_segment_details(dataset, column, &segments).await?;
                let indices =
                    open_fts_segments(dataset, column, &segments, &metrics.index_metrics).await?;
                all_segments.extend(segments.iter().cloned());
                indices
            }
            None => Vec::new(),
        };
        columns.push(CombinedFieldColumn {
            column: column.to_string(),
            weight,
            indices,
        });
    }
    validate_combined_tokenizers(&columns)?;
    metrics.record_parts_searched(
        columns
            .iter()
            .flat_map(|column| &column.indices)
            .map(|index| index.partition_count())
            .sum(),
    );

    // Fields share a tokenizer (validated above), so tokenize once.
    let first_index = columns
        .iter()
        .find_map(|column| column.indices.first())
        .ok_or_else(|| match index_requirement {
            // Not a user error: `EveryColumn` resolution already failed for any
            // column without an index, and `CombinedFieldsQuery::try_new` rejects
            // an empty column list, so reaching this means one of those two
            // guarantees broke.
            CombinedFieldsIndexRequirement::EveryColumn => DataFusionError::Internal(
                "combined_fields query reached execution with no target columns".to_string(),
            ),
            CombinedFieldsIndexRequirement::AtLeastOneColumn => {
                DataFusionError::Execution(format!(
                    "combined_fields requires an inverted index on at least one of {:?}",
                    query.column_names().collect::<Vec<_>>()
                ))
            }
        })?;
    let mut tokenizer = first_index.tokenizer();
    let tokens = collect_query_tokens(query.terms(), &mut tokenizer);
    record_tokenized_query(tokenized_query, &tokens);

    Ok(CombinedFieldsScan {
        columns,
        segments: all_segments,
        tokens,
        tokenizer,
    })
}

impl ExecutionPlan for CombinedFieldsQueryExec {
    fn name(&self) -> &str {
        "CombinedFieldsQueryExec"
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        match &self.prefilter_source {
            PreFilterSource::None => vec![],
            PreFilterSource::FilteredRowIds(src) => vec![&src],
            PreFilterSource::ScalarIndexQuery(src) => vec![&src],
        }
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
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
                self.clone_with_prefilter_source(PreFilterSource::None)
            }
            1 => {
                let src = children.pop().unwrap();
                let prefilter_source = match &self.prefilter_source {
                    PreFilterSource::FilteredRowIds(_) => PreFilterSource::FilteredRowIds(src),
                    PreFilterSource::ScalarIndexQuery(_) => PreFilterSource::ScalarIndexQuery(src),
                    PreFilterSource::None => {
                        return Err(DataFusionError::Internal(
                            "Unexpected prefilter source".to_string(),
                        ));
                    }
                };
                self.clone_with_prefilter_source(prefilter_source)
            }
            _ => {
                return Err(DataFusionError::Internal(
                    "Unexpected number of children".to_string(),
                ));
            }
        };
        Ok(Arc::new(plan))
    }

    #[instrument(name = "combined_fields_query_exec", level = "debug", skip_all)]
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
        let preset_base_scorer = self.base_scorer.clone();
        let shared_scorer = self.shared_scorer.clone();
        let covered_fragments = self.covered_fragments.clone();
        let metrics = Arc::new(FtsIndexMetrics::new(&self.metrics, partition));
        let stream = stream::once(async move {
            let _timer = metrics.baseline_metrics.elapsed_compute().timer();

            let CombinedFieldsScan {
                columns,
                segments: all_segments,
                tokens,
                ..
            } = open_combined_fields_scan(
                &ds,
                &query,
                CombinedFieldsIndexRequirement::EveryColumn,
                &tokenized_query,
                metrics.as_ref(),
            )
            .await?;

            // With mixed coverage the planner supplies the fragments every target
            // column indexes, and the scan is restricted to exactly those: the
            // rest go to the flat sibling. Otherwise the prefilter spans the union
            // of the target columns' segments.
            let mut pre_filter = match covered_fragments {
                Some(covered) => build_prefilter_restricted_to_fragments(
                    context.clone(),
                    partition,
                    &prefilter_source,
                    ds,
                    covered,
                )?,
                None => build_prefilter(
                    context.clone(),
                    partition,
                    &prefilter_source,
                    ds,
                    &all_segments,
                    None,
                )?,
            };
            let deleted_fragments = columns.iter().flat_map(|column| &column.indices).fold(
                roaring::RoaringBitmap::new(),
                |mut deleted, index| {
                    deleted |= index.deleted_fragments().clone();
                    deleted
                },
            );
            if !deleted_fragments.is_empty() {
                Arc::get_mut(&mut pre_filter)
                    .expect("prefilter just created")
                    .set_deleted_fragments(deleted_fragments);
            }
            let scorer = match (preset_base_scorer, shared_scorer) {
                (Some(scorer), _) => scorer,
                // A mixed plan routes the rows no index covers to a flat sibling,
                // so folding only this side's index statistics here would score the
                // two children against different `docCount'`/`docFreq'`/`avgdl'`.
                // Wait for the blend that describes the whole scanned corpus.
                (None, Some(shared_scorer)) => shared_scorer.wait().await?,
                (None, None) => {
                    let scorer_start = std::time::Instant::now();
                    let scorer = Arc::new(
                        build_combined_bm25_scorer(
                            &columns,
                            &tokens,
                            CombinedCorpusStats::IndexOnly,
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
            let (doc_ids, scores) = combined_fields_search(
                &columns,
                &tokens,
                &params,
                query.operator(),
                scorer.as_ref(),
                pre_filter,
                metrics.as_ref(),
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

/// Filters the input according to a match query's token operator.
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
            let mut remaining_positions = (0..query_tokens.len())
                .map(|index| query_tokens.position(index))
                .collect::<HashSet<_>>();
            if remaining_positions.is_empty() {
                return false;
            }
            let mut stream = tokenizer.token_stream_for_doc(text);
            while let Some(token) = stream.next() {
                for index in 0..query_tokens.len() {
                    if token.text == query_tokens.get_token(index) {
                        remaining_positions.remove(&query_tokens.position(index));
                    }
                }
                if remaining_positions.is_empty() {
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
        if query.fuzziness != Some(0) {
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
        let query_tokens = Arc::new(collect_query_tokens(&query.terms, &mut tokenizer));
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

    fn partition_statistics(&self, partition: Option<usize>) -> DataFusionResult<Arc<Statistics>> {
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
    tokenized_query: Arc<OnceLock<TokenizedQuery>>,
    params: FtsSearchParams,
    unindexed_input: Arc<dyn ExecutionPlan>,
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

    /// See [`MatchQueryExec::new_with_segments`].
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
            document_granularity,
            document_column,
            schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// Override the local BM25 scorer; see [`MatchQueryExec::with_base_scorer`].
    pub fn with_base_scorer(mut self, scorer: Arc<MemBM25Scorer>) -> Self {
        self.base_scorer = Some(scorer);
        self
    }

    pub(crate) fn with_shared_scorer(mut self, scorer: Arc<SharedFtsScorer>) -> Self {
        self.shared_scorer = Some(scorer);
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
            tokenized_query: self.tokenized_query.clone(),
            params: self.params.clone(),
            unindexed_input,
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
        let unindexed_input = document_input(
            self.unindexed_input.execute(partition, context)?,
            &document_column,
        )?;

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
                        let query_tokens = collect_query_tokens(&query.terms, &mut tokenizer);
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
                        let query_tokens = collect_query_tokens(&query.terms, &mut tokenizer);
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
                    Ok(stream)
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

/// Cross-field (BM25F) scoring for rows no index covers.
///
/// One target column's share of the rows a flat `combined_fields` scan may fold
/// into corpus statistics; see [`FlatCombinedFieldsExec::with_stats_coverage`].
#[derive(Debug, Clone, Default)]
pub struct FlatStatsCoverage {
    /// Fragments this column's index does not cover at all.
    pub fragments: roaring::RoaringBitmap,
    /// Rows it does cover, but whose entries a newer overlay made stale, in the id
    /// domain the scan's `_rowid` uses. The index still counts their pre-overlay
    /// values, so folding these over-counts the column by at most their number.
    pub stale_rows: RowAddrTreeMap,
}

impl FlatStatsCoverage {
    /// Coverage by whole fragments, with no individually named stale rows.
    pub fn for_fragments(fragments: roaring::RoaringBitmap) -> Self {
        Self {
            fragments,
            stale_rows: RowAddrTreeMap::new(),
        }
    }
}

/// The sibling of [`CombinedFieldsQueryExec`] for the fragments that at least one
/// target column's index is missing. Those rows cannot be scored from the indexes
/// (`tf'`/`dl'` would omit the unindexed columns), so they are scored straight
/// from their column values here. The two plans are unioned and re-sorted by the
/// planner, exactly as the single-column match path does.
///
/// The input must carry `_rowid` plus every target column of the query.
#[derive(Debug)]
pub struct FlatCombinedFieldsExec {
    dataset: Arc<Dataset>,
    query: CombinedFieldsQuery,
    /// Tokens `execute()` actually produced, for `analyze_plan`. Single snapshot
    /// for the same reason as [`CombinedFieldsQueryExec::tokenized_query`].
    tokenized_query: Arc<OnceLock<TokenizedQuery>>,
    params: FtsSearchParams,
    unindexed_input: Arc<dyn ExecutionPlan>,
    /// Optional override for the blended scorer normally built locally inside
    /// `execute()`. See [`CombinedFieldsQueryExec::with_base_scorer`].
    base_scorer: Option<Arc<CombinedFieldsBM25Scorer>>,
    /// Per target column (aligned with `query.columns`): the fragments that
    /// column's index does not cover. Only rows in those fragments may be folded
    /// into that column's corpus statistics; see
    /// [`Self::with_stats_coverage`].
    stats_coverage: Vec<FlatStatsCoverage>,
    /// Set when the scan was read unfiltered: this then selects the rows to emit,
    /// over `scanned_fragments`. See [`Self::with_emit_prefilter`].
    emit_prefilter: Option<PreFilterSource>,
    /// The fragments `unindexed_input` reads, so the emit mask spans exactly them.
    scanned_fragments: roaring::RoaringBitmap,
    /// Whether the target columns' index statistics are dropped in favour of this
    /// scan's own. See [`Self::with_whole_corpus_stats`].
    whole_corpus_stats: bool,
    /// Per target column (aligned with `query.columns`): the schema-derived form
    /// of its field path, which decides how `execute` turns the scanned column
    /// into document text. See [`Self::with_resolved_fields`].
    resolved_fields: Vec<ResolvedFtsField>,
    /// Publishes the blended scorer to an indexed sibling; see
    /// [`Self::with_shared_scorer`].
    shared_scorer: Option<Arc<SharedFtsScorer<CombinedFieldsBM25Scorer>>>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl DisplayAs for FlatCombinedFieldsExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let columns = self.query.column_names().collect::<Vec<_>>().join(", ");
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "FlatCombinedFields: columns=[{}], query={}",
                    columns,
                    self.query.terms()
                )?;
                fmt_tokenized_query(&self.tokenized_query, ", ", f)
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "FlatCombinedFields\ncolumns=[{}]\nquery={}",
                    columns,
                    self.query.terms()
                )?;
                fmt_tokenized_query(&self.tokenized_query, "\n", f)
            }
        }
    }
}

impl FlatCombinedFieldsExec {
    pub fn new(
        dataset: Arc<Dataset>,
        query: CombinedFieldsQuery,
        params: FtsSearchParams,
        unindexed_input: Arc<dyn ExecutionPlan>,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(FTS_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        let stats_coverage = vec![FlatStatsCoverage::default(); query.column_names().len()];
        Self {
            dataset,
            query,
            tokenized_query: Arc::new(OnceLock::new()),
            params,
            unindexed_input,
            base_scorer: None,
            stats_coverage,
            emit_prefilter: None,
            scanned_fragments: roaring::RoaringBitmap::new(),
            whole_corpus_stats: false,
            resolved_fields: Vec::new(),
            shared_scorer: None,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// Supply the schema-derived form of each target column's field path, in
    /// query order.
    ///
    /// The planner resolves these to decide what the scan projects, and `execute`
    /// needs the same answers to read the projected batches back, so one entry
    /// per target column is required.
    pub(crate) fn with_resolved_fields(mut self, resolved_fields: Vec<ResolvedFtsField>) -> Self {
        self.resolved_fields = resolved_fields;
        self
    }

    /// Publish the blended scorer this scan builds to an indexed sibling.
    ///
    /// Only this side sees the unindexed rows, so only it can produce the corpus
    /// statistics both sides must score against.
    pub(crate) fn with_shared_scorer(
        mut self,
        scorer: Arc<SharedFtsScorer<CombinedFieldsBM25Scorer>>,
    ) -> Self {
        self.shared_scorer = Some(scorer);
        self
    }

    /// Record, per target column, which scanned rows its index does not already
    /// account for.
    ///
    /// The flat scan must read all of a row's target columns or `tf'`/`dl'` would be
    /// partial, but a scanned fragment can still be indexed for some of those
    /// columns. Without this, such a row would be folded into that column's corpus
    /// statistics on top of the index statistics that already include it, inflating
    /// `docCount_f`, `sumTotalTermFreq_f`, and `docFreq_f`.
    pub fn with_stats_coverage(mut self, coverage: Vec<FlatStatsCoverage>) -> Self {
        self.stats_coverage = coverage;
        self
    }

    /// Select emitted rows with `prefilter` rather than with the scan.
    ///
    /// The caller reads the scan unfiltered so that a folded row's contribution to the
    /// corpus statistics does not depend on the query's filter, and hands the filter
    /// here instead, which is how [`CombinedFieldsQueryExec`] treats its own.
    pub fn with_emit_prefilter(
        mut self,
        prefilter: PreFilterSource,
        scanned_fragments: roaring::RoaringBitmap,
    ) -> Self {
        self.emit_prefilter = Some(prefilter);
        self.scanned_fragments = scanned_fragments;
        self
    }

    /// Drop the target columns' index statistics in favour of this scan's own; see
    /// [`CombinedCorpusStats::FlatOnly`](lance_index::scalar::inverted::CombinedCorpusStats::FlatOnly).
    ///
    /// Every entry of [`Self::with_stats_coverage`] must then span every target
    /// fragment, or the fold drops rows nothing else now accounts for.
    pub fn with_whole_corpus_stats(mut self) -> Self {
        self.whole_corpus_stats = true;
        self
    }

    /// Override the local blended scorer; see
    /// [`CombinedFieldsQueryExec::with_base_scorer`].
    pub fn with_base_scorer(mut self, scorer: Arc<CombinedFieldsBM25Scorer>) -> Self {
        self.base_scorer = Some(scorer);
        self
    }

    pub fn query(&self) -> &CombinedFieldsQuery {
        &self.query
    }

    pub fn params(&self) -> &FtsSearchParams {
        &self.params
    }

    pub fn dataset(&self) -> &Arc<Dataset> {
        &self.dataset
    }
}

impl ExecutionPlan for FlatCombinedFieldsExec {
    fn name(&self) -> &str {
        "FlatCombinedFieldsExec"
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        let mut children = vec![&self.unindexed_input];
        if let Some(
            PreFilterSource::FilteredRowIds(source) | PreFilterSource::ScalarIndexQuery(source),
        ) = &self.emit_prefilter
        {
            children.push(source);
        }
        children
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        // `execute()` reads only `unindexed_input.execute(partition)` for the
        // single output partition, so the input must be coalesced to one
        // partition or fragments would be silently dropped. Same reasoning as
        // `FlatMatchQueryExec`.
        self.children()
            .iter()
            .map(|_| Distribution::SinglePartition)
            .collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != self.children().len() {
            return Err(DataFusionError::Internal(
                "Unexpected number of children".to_string(),
            ));
        }
        // Reverse order: the optional prefilter child is last.
        let emit_prefilter = match (&self.emit_prefilter, children.pop()) {
            (Some(PreFilterSource::FilteredRowIds(_)), Some(source)) => {
                Some(PreFilterSource::FilteredRowIds(source))
            }
            (Some(PreFilterSource::ScalarIndexQuery(_)), Some(source)) => {
                Some(PreFilterSource::ScalarIndexQuery(source))
            }
            // No prefilter child, so the popped plan is the input itself.
            (source, popped) => {
                children.extend(popped);
                source.clone()
            }
        };
        let Some(unindexed_input) = children.pop() else {
            return Err(DataFusionError::Internal(
                "combined_fields flat scan lost its input child".to_string(),
            ));
        };
        Ok(Arc::new(Self {
            dataset: self.dataset.clone(),
            query: self.query.clone(),
            tokenized_query: self.tokenized_query.clone(),
            params: self.params.clone(),
            unindexed_input,
            base_scorer: self.base_scorer.clone(),
            stats_coverage: self.stats_coverage.clone(),
            emit_prefilter,
            scanned_fragments: self.scanned_fragments.clone(),
            whole_corpus_stats: self.whole_corpus_stats,
            resolved_fields: self.resolved_fields.clone(),
            shared_scorer: self.shared_scorer.clone(),
            properties: self.properties.clone(),
            metrics: ExecutionPlanMetricsSet::new(),
        }))
    }

    #[instrument(name = "flat_combined_fields_exec", level = "debug", skip_all)]
    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let query = self.query.clone();
        let tokenized_query = self.tokenized_query.clone();
        let ds = self.dataset.clone();
        let preset_base_scorer = self.base_scorer.clone();
        let stats_coverage = self.stats_coverage.clone();
        let emit_prefilter = self.emit_prefilter.clone();
        let scanned_fragments = self.scanned_fragments.clone();
        let whole_corpus_stats = self.whole_corpus_stats;
        let prefilter_context = context.clone();
        let stats_mask_ds = self.dataset.clone();
        let metrics = Arc::new(FtsIndexMetrics::new(&self.metrics, partition));
        let metrics_clone = metrics.clone();
        let target_batch_size = context.session_config().batch_size();
        let elapsed_compute = metrics.baseline_metrics.elapsed_compute().clone();

        let shared_scorer_producer = self.shared_scorer.clone().map(SharedFtsScorerProducer::new);

        // One resolved field per target column. Without the check a short list
        // would silently fall back to reading the raw column, which for a
        // `List<Utf8>` target still tokenizes and so reverts to a per-element
        // `dl_f` instead of erroring.
        if self.resolved_fields.len() != query.column_names().len() {
            return Err(DataFusionError::Internal(format!(
                "combined_fields flat scan got {} resolved fields for {} target columns",
                self.resolved_fields.len(),
                query.column_names().len()
            )));
        }
        let mut input = self.unindexed_input.execute(partition, context)?;
        for (column, resolved) in query.column_names().zip(&self.resolved_fields) {
            // A path that continues past a `List` cannot be projected, so the scan
            // brought its list root instead and the leaf text is derived here.
            input = if resolved.has_lists() {
                flatten_fts_document_column(input, resolved.clone())?
            } else {
                document_input(input, column)?
            };
        }

        let stream =
            stream::once(async move {
                let shared_scorer_producer = shared_scorer_producer;
                let result = async {
                let CombinedFieldsScan {
                    columns,
                    tokens,
                    tokenizer,
                    ..
                } = open_combined_fields_scan(
                    &ds,
                    &query,
                    CombinedFieldsIndexRequirement::AtLeastOneColumn,
                    &tokenized_query,
                    metrics.as_ref(),
                )
                .await?;

                let input_schema = input.schema();
                let doc_col_indices = query
                    .column_names()
                    .map(|column| input_schema.index_of(column))
                    .collect::<std::result::Result<Vec<_>, _>>()?;

                if stats_coverage.len() != columns.len() {
                    return Err(DataFusionError::Internal(format!(
                        "combined_fields flat scan got {} statistics coverage sets for {} columns",
                        stats_coverage.len(),
                        columns.len()
                    )));
                }
                // One mask per target column, allowing the rows that column's index is
                // missing. Built through `create_restricted_deletion_mask` so the
                // fragment restriction lands in the same id space the flat scan's
                // `_rowid` uses; a hand-rolled fragment-address allow list would match
                // nothing under stable row ids. `None` means nothing needed filtering,
                // i.e. every scanned row is missing from that index.
                let mut stats_masks = Vec::with_capacity(stats_coverage.len());
                for coverage in &stats_coverage {
                    let mask = match DatasetPreFilter::create_restricted_deletion_mask(
                        stats_mask_ds.clone(),
                        coverage.fragments.clone(),
                    ) {
                        Some(fut) => fut.await?,
                        None => Arc::new(RowAddrMask::all_rows()),
                    };
                    // The overlay-stale rows sit in fragments this column's index does
                    // cover, so they are named individually rather than by fragment.
                    let mask = if coverage.stale_rows.is_empty() {
                        mask
                    } else {
                        Arc::new(mask.as_ref().clone().also_allow(coverage.stale_rows.clone()))
                    };
                    stats_masks.push(mask);
                }

                // Set when the scan was read unfiltered so a folded row's contribution
                // does not move with the filter; the filter then picks what to emit.
                let emit_mask = match emit_prefilter {
                    Some(prefilter) => {
                        let pre_filter = build_prefilter_restricted_to_fragments(
                            prefilter_context,
                            partition,
                            &prefilter,
                            stats_mask_ds.clone(),
                            scanned_fragments,
                        )?;
                        pre_filter.wait_for_ready().await?;
                        Some(pre_filter.mask())
                    }
                    None => None,
                };

                flat_combined_fields_search_stream(
                    input,
                    &columns,
                    doc_col_indices,
                    &stats_masks,
                    emit_mask,
                    whole_corpus_stats,
                    &tokens,
                    tokenizer,
                    query.operator(),
                    preset_base_scorer,
                    target_batch_size,
                    Some(elapsed_compute),
                    Some(metrics.as_ref()),
                )
                .await
            }
            .await;

                match result {
                    Ok((stream, scorer)) => {
                        if let Some(producer) = shared_scorer_producer {
                            producer.publish(scorer);
                        }
                        Ok(stream)
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
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// See [`MatchQueryExec::new_with_segments`].
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
        mut params: FtsSearchParams,
        prefilter_source: PreFilterSource,
        segment_uuids: Vec<Uuid>,
    ) -> Result<Self> {
        let document_granularity = query.document_granularity.ok_or_else(|| {
            Error::invalid_input("PhraseQuery document granularity must be resolved".to_string())
        })?;
        let schema = fts_schema(document_granularity);
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        params = params.with_phrase_slop(Some(query.slop));

        Ok(Self {
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
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    /// Override the local BM25 scorer; see [`MatchQueryExec::with_base_scorer`].
    pub fn with_base_scorer(mut self, scorer: Arc<MemBM25Scorer>) -> Self {
        self.base_scorer = Some(scorer);
        self
    }

    pub(crate) fn with_shared_scorer(mut self, scorer: Arc<SharedFtsScorer>) -> Self {
        self.shared_scorer = Some(scorer);
        self
    }

    /// Exclude rows whose indexed text was superseded by a newer data overlay.
    pub(crate) fn with_overlay_block(mut self, overlay_block: RowAddrMask) -> Self {
        self.overlay_block = Some(overlay_block);
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
                tokenized_query: self.tokenized_query.clone(),
                params: self.params.clone(),
                prefilter_source: PreFilterSource::None,
                base_scorer: self.base_scorer.clone(),
                shared_scorer: self.shared_scorer.clone(),
                segment_selection: self.segment_selection.clone(),
                overlay_block: self.overlay_block.clone(),
                document_granularity: self.document_granularity,
                schema: self.schema.clone(),
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
                    tokenized_query: self.tokenized_query.clone(),
                    params: self.params.clone(),
                    prefilter_source,
                    base_scorer: self.base_scorer.clone(),
                    shared_scorer: self.shared_scorer.clone(),
                    segment_selection: self.segment_selection.clone(),
                    overlay_block: self.overlay_block.clone(),
                    document_granularity: self.document_granularity,
                    schema: self.schema.clone(),
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
        let preset_base_scorer = self.base_scorer.clone();
        let shared_scorer = self.shared_scorer.clone();
        let segment_selection = self.segment_selection.clone();
        let overlay_block = self.overlay_block.clone();
        let document_granularity = self.document_granularity;
        let schema = self.schema.clone();
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
            let indices =
                open_fts_segments(&ds, &column, &segments, &metrics.index_metrics).await?;

            let mut pre_filter = build_prefilter(
                context.clone(),
                partition,
                &prefilter_source,
                ds,
                &segments,
                overlay_block,
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
            let tokens = collect_query_tokens(&query.terms, &mut tokenizer);
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

#[derive(Debug)]
pub struct BoostQueryExec {
    query: BoostQuery,
    params: FtsSearchParams,
    positive: Arc<dyn ExecutionPlan>,
    negative: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,

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
        let schema = positive.schema();
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Self {
            query,
            params,
            positive,
            negative,
            schema,
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
            schema: self.schema.clone(),
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
        let schema = self.schema.clone();
        let metrics = Arc::new(FtsIndexMetrics::new(&self.metrics, partition));
        let stream = stream::once(async move {
            let positive = positive.try_collect::<Vec<_>>().await?;
            let negative = negative.try_collect::<Vec<_>>().await?;

            let _timer = metrics.baseline_metrics.elapsed_compute().timer();
            let mut res = HashMap::new();
            for batch in positive {
                for (key, score) in batch_scored_document_keys(&batch)? {
                    res.insert(key, score);
                }
            }
            for batch in negative {
                for (key, neg_score) in batch_scored_document_keys(&batch)? {
                    if let Some(score) = res.get_mut(&key) {
                        *score -= query.negative_boost * neg_score;
                    }
                }
            }

            let documents = res
                .into_iter()
                .sorted_unstable_by(compare_scored_documents)
                .take(params.limit.unwrap_or(usize::MAX))
                .collect::<Vec<_>>();
            metrics.baseline_metrics.record_output(documents.len());

            let batch = document_key_scores_batch(schema, documents)?;
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
    children: Vec<Arc<dyn ExecutionPlan>>,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    build_boolean_query_children_with_schema(slot, children, FTS_SCHEMA.clone())
}

pub fn build_boolean_query_children_with_schema(
    slot: BoolSlot,
    mut children: Vec<Arc<dyn ExecutionPlan>>,
    schema: SchemaRef,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    match slot {
        BoolSlot::Should | BoolSlot::MustNot => {
            if children.is_empty() {
                Ok(Some(Arc::new(EmptyExec::new(schema))))
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
                    let mut on: Vec<(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)> = vec![(
                        Arc::new(Column::new_with_schema(ROW_ID, &schema)?),
                        Arc::new(Column::new_with_schema(ROW_ID, &schema)?),
                    )];
                    if schema.field_with_name(DOC_INDEX_COL).is_ok() {
                        on.push((
                            Arc::new(Column::new_with_schema(DOC_INDEX_COL, &schema)?),
                            Arc::new(Column::new_with_schema(DOC_INDEX_COL, &schema)?),
                        ));
                    }
                    joined = Some(Arc::new(HashJoinExec::try_new(
                        left,
                        plan,
                        on,
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
    schema: SchemaRef,

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
        let schema = should.schema();
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
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
            schema,
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
                    schema: self.schema.clone(),
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
                    schema: self.schema.clone(),
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
                    schema: self.schema.clone(),
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
        let schema = self.schema.clone();

        let stream = stream::once(async move {
            let elapsed_time = metrics.baseline_metrics.elapsed_compute();

            let mut res = HashMap::new();
            let has_must = must.is_some();
            if let Some(mut must) = must {
                while let Some(batch) = must.try_next().await? {
                    let _timer = elapsed_time.timer();
                    res.extend(batch_scored_document_keys_sum_scores(&batch)?);
                }
            }

            // add the scores from the should clause
            while let Some(batch) = should.try_next().await? {
                let _timer = elapsed_time.timer();
                for (key, score) in batch_scored_document_keys(&batch)? {
                    let entry = res.entry(key).and_modify(|value| *value += score);
                    if !has_must {
                        entry.or_insert(score);
                    }
                }
            }

            // remove the results from the must_not clause
            while let Some(batch) = must_not.try_next().await? {
                let _timer = elapsed_time.timer();
                for key in batch_document_keys(&batch)? {
                    res.remove(&key);
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
            let documents = res
                .into_iter()
                .sorted_unstable_by(compare_scored_documents)
                .take(params.limit.unwrap_or(usize::MAX))
                .collect::<Vec<_>>();
            metrics.baseline_metrics.record_output(documents.len());
            let batch = document_key_scores_batch(schema, documents)?;
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
    use datafusion::error::{DataFusionError, Result as DataFusionResult};
    use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
    use datafusion::{execution::TaskContext, physical_plan::ExecutionPlan};
    use futures::TryStreamExt;
    use lance_core::{ROW_ID, utils::address::RowAddress};
    use lance_datafusion::datagen::DatafusionDatagenExt;
    use lance_datafusion::exec::{ExecutionStatsCallback, ExecutionSummaryCounts};
    use lance_datafusion::utils::{
        INDEX_CACHE_HITS_METRIC, INDEX_CACHE_MISSES_METRIC, PARTITIONS_SEARCHED_METRIC,
    };
    use lance_datagen::{BatchCount, ByteCount, RowCount};
    use lance_index::metrics::{MetricsCollector, NoOpMetricsCollector};
    use lance_index::scalar::inverted::query::{
        BooleanQuery, BoostQuery, CombinedFieldsQuery, FtsQuery, FtsSearchParams, MatchQuery,
        Occur, Operator, PhraseQuery, collect_query_tokens, has_query_token,
    };
    use lance_index::scalar::inverted::{
        CombinedCorpusStats, CombinedFieldColumn, DocumentGranularity, FTS_SCHEMA, InvertedIndex,
        Language, SCORE_COL, build_combined_bm25_scorer, build_global_bm25_scorer,
    };
    use lance_index::scalar::{FullTextSearchQuery, InvertedIndexParams};
    use lance_index::{IndexCriteria, IndexType};
    use lance_table::format::IndexMetadata;
    use uuid::Uuid;

    use crate::{
        Dataset,
        dataset::WriteParams,
        dataset::transaction::{Operation, TransactionBuilder},
        index::DatasetIndexInternalExt,
        io::exec::PreFilterSource,
        utils::test::{DatagenExt, FragmentCount, FragmentRowCount, NoContextTestFixture},
    };

    use super::{
        BoolSlot, BoostQueryExec, CombinedFieldsQueryExec, CompoundQueryExec,
        CrossColumnCompoundQueryExec, FTS_SEGMENT_BIND_DURATION_METRIC, FlatCombinedFieldsExec,
        FlatMatchFilterExec, FlatMatchQueryExec, MatchQueryExec, PhraseQueryExec,
        build_boolean_query_children, default_text_tokenizer, open_fts_segments,
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
    fn test_compound_should_metrics_are_counted_independently() {
        let metrics_set = ExecutionPlanMetricsSet::new();
        let metrics = super::FtsIndexMetrics::new(&metrics_set, 0);

        metrics.record_compound_should_skipped_windows(2);
        metrics.record_compound_should_bound_recomputations(3);
        metrics.record_compound_should_essential_evaluations(5);
        metrics.record_compound_should_non_essential_evaluations(7);

        assert_eq!(metrics.compound_should_skipped_windows.value(), 2);
        assert_eq!(metrics.compound_should_bound_recomputations.value(), 3);
        assert_eq!(metrics.compound_should_essential_evaluations.value(), 5);
        assert_eq!(metrics.compound_should_non_essential_evaluations.value(), 7);
    }

    #[test]
    fn test_cross_column_staged_metrics_are_counted_independently() {
        let metrics_set = ExecutionPlanMetricsSet::new();
        let metrics = super::FtsIndexMetrics::new(&metrics_set, 0);

        metrics.record_cross_column_staged_attempts(2);
        metrics.record_cross_column_staged_successes(1);
        metrics.record_cross_column_staged_fallbacks(1);
        metrics.record_cross_column_staged_candidates(17);

        assert_eq!(metrics.cross_column_staged_attempts.value(), 2);
        assert_eq!(metrics.cross_column_staged_successes.value(), 1);
        assert_eq!(metrics.cross_column_staged_fallbacks.value(), 1);
        assert_eq!(metrics.cross_column_staged_candidates.value(), 17);
    }

    async fn create_segment_selection_fixture() -> (Arc<Dataset>, Vec<IndexMetadata>, Vec<u32>) {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "text",
                lance_datagen::array::cycle_utf8_literals(&["quick brown fox"]),
            )
            .col(
                "other",
                lance_datagen::array::cycle_utf8_literals(&["not indexed"]),
            )
            .into_ram_dataset(FragmentCount::from(3), FragmentRowCount::from(2))
            .await
            .unwrap();
        let fragment_ids = dataset
            .get_fragments()
            .iter()
            .map(|fragment| fragment.id() as u32)
            .collect::<Vec<_>>();
        assert_eq!(fragment_ids.len(), 3);

        let params = InvertedIndexParams::default().with_position(true);
        let mut segments = Vec::with_capacity(fragment_ids.len());
        for fragment_id in &fragment_ids {
            let mut builder = dataset
                .create_index_builder(&["text"], IndexType::Inverted, &params)
                .name("segment_selection_fts".to_string())
                .fragments(vec![*fragment_id]);
            segments.push(builder.execute_uncommitted().await.unwrap());
        }
        dataset
            .commit_existing_index_segments("segment_selection_fts", "text", segments.clone())
            .await
            .unwrap();

        let committed = crate::index::scalar::inverted::load_segments(
            &dataset,
            "text",
            DocumentGranularity::Row,
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(committed.len(), fragment_ids.len());
        (Arc::new(dataset), committed, fragment_ids)
    }

    fn tokenized_query_index_params() -> InvertedIndexParams {
        InvertedIndexParams::new("simple".to_string(), Language::English)
            .with_position(true)
            .lower_case(true)
            .stem(false)
            .remove_stop_words(true)
            .ascii_folding(false)
    }

    /// Builds a dataset where every column in `columns` holds "first and second"
    /// and carries its own inverted index. `combined_fields` needs an FTS index
    /// per target column, so the columns are indexed one at a time rather than
    /// as a single multi-column index.
    ///
    /// With `with_unindexed_append`, extra rows land outside the indices, which
    /// forces the mixed plan where an indexed exec and its flat sibling both run.
    async fn create_tokenized_query_fixture(
        columns: &[&str],
        with_unindexed_append: bool,
    ) -> Dataset {
        let corpus = || {
            columns
                .iter()
                .fold(lance_datagen::gen_batch(), |batch, column| {
                    batch.col(
                        *column,
                        lance_datagen::array::cycle_utf8_literals(&["first and second"]),
                    )
                })
        };

        let mut dataset = corpus()
            .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(2))
            .await
            .unwrap();
        for column in columns {
            dataset
                .create_index(
                    &[*column],
                    IndexType::Inverted,
                    None,
                    &tokenized_query_index_params(),
                    true,
                )
                .await
                .unwrap();
        }

        if with_unindexed_append {
            let appended = corpus().into_reader_rows(RowCount::from(2), BatchCount::from(1));
            dataset.append(appended, None).await.unwrap();
        }
        dataset
    }

    fn find_plan_line<'a>(analysis: &'a str, node: &str) -> &'a str {
        analysis
            .lines()
            .find(|line| line.trim_start().starts_with(node))
            .unwrap_or_else(|| panic!("{node} missing from plan:\n{analysis}"))
    }

    fn segment_uuid_for_fragment(segments: &[IndexMetadata], fragment_id: u32) -> Uuid {
        segments
            .iter()
            .find(|segment| {
                segment
                    .fragment_bitmap
                    .as_ref()
                    .is_some_and(|fragments| fragments.contains(fragment_id))
            })
            .map(|segment| segment.uuid)
            .unwrap()
    }

    fn expected_row_ids(fragment_ids: &[u32]) -> Vec<u64> {
        let mut row_ids = fragment_ids
            .iter()
            .flat_map(|fragment_id| {
                (0..2).map(|offset| u64::from(RowAddress::new_from_parts(*fragment_id, offset)))
            })
            .collect::<Vec<_>>();
        row_ids.sort_unstable();
        row_ids
    }

    async fn execute_results(plan: &dyn ExecutionPlan) -> DataFusionResult<Vec<(u64, f32)>> {
        let batches: Vec<RecordBatch> = plan
            .execute(0, Arc::new(TaskContext::default()))?
            .try_collect()
            .await?;
        let mut results = Vec::new();
        for batch in batches {
            let row_ids = batch[ROW_ID]
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            let scores = batch[SCORE_COL]
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap();
            results.extend(
                row_ids
                    .values()
                    .iter()
                    .copied()
                    .zip(scores.values().iter().copied()),
            );
        }
        results.sort_by_key(|(row_id, _)| *row_id);
        Ok(results)
    }

    async fn execute_row_ids(plan: &dyn ExecutionPlan) -> DataFusionResult<Vec<u64>> {
        Ok(execute_results(plan)
            .await?
            .into_iter()
            .map(|(row_id, _)| row_id)
            .collect())
    }

    fn metric_value(plan: &dyn ExecutionPlan, name: &str) -> usize {
        plan.metrics()
            .unwrap()
            .iter()
            .find(|metric| metric.value().name() == name)
            .unwrap()
            .value()
            .as_usize()
    }

    fn assert_execution_error(error: DataFusionError, expected_message: &str) {
        assert!(
            matches!(&error, DataFusionError::Execution(_)),
            "expected execution error, got {error:?}"
        );
        assert!(
            error.to_string().contains(expected_message),
            "expected error containing {expected_message:?}, got {error}"
        );
    }

    #[test]
    fn document_match_filter_respects_document_boundary() {
        let mut tokenizer = default_text_tokenizer();
        let query_tokens = collect_query_tokens("alpha", &mut tokenizer);
        assert!(super::document_matches_query(
            "alpha beta",
            &mut tokenizer,
            &query_tokens,
            Operator::Or,
        ));

        let mut tokenizer = default_text_tokenizer();
        let query_tokens = collect_query_tokens("alpha beta", &mut tokenizer);
        assert!(!super::document_matches_query(
            "alpha",
            &mut tokenizer,
            &query_tokens,
            Operator::And,
        ));
        assert!(super::document_matches_query(
            "alpha beta",
            &mut tokenizer,
            &query_tokens,
            Operator::And,
        ));
    }

    #[tokio::test]
    async fn shared_fts_scorer_reports_cancelled_producer() {
        let scorer = Arc::new(super::SharedFtsScorer::<super::MemBM25Scorer>::new());
        let producer = super::SharedFtsScorerProducer::new(scorer.clone());
        drop(producer);

        let error = tokio::time::timeout(std::time::Duration::from_secs(1), scorer.wait())
            .await
            .expect("cancelled producer must wake scorer waiters")
            .unwrap_err();
        assert!(
            error.to_string().contains("producer was cancelled"),
            "{error}"
        );
    }

    #[test]
    fn execute_without_context() {
        // These tests ensure we can create nodes and call execute without a tokio Runtime
        // being active.  This is a requirement for proper implementation of a Datafusion foreign
        // table provider.
        let fixture = NoContextTestFixture::new();
        let match_query = MatchQueryExec::new(
            Arc::new(fixture.dataset.clone()),
            MatchQuery::new("blah".to_string())
                .with_column(Some("text".to_string()))
                .with_document_granularity(DocumentGranularity::Row),
            FtsSearchParams::default(),
            PreFilterSource::None,
        )
        .unwrap();
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
            MatchQuery::new("blah".to_string())
                .with_column(Some("text".to_string()))
                .with_document_granularity(DocumentGranularity::Row),
            FtsSearchParams::default(),
            flat_input,
        )
        .unwrap();
        flat_match_query
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap();
        let metrics = flat_match_query.metrics().unwrap();
        assert!(metrics.elapsed_compute().unwrap() > 0);

        let phrase_query = PhraseQueryExec::new(
            Arc::new(fixture.dataset.clone()),
            PhraseQuery::new("blah".to_string())
                .with_document_granularity(DocumentGranularity::Row),
            FtsSearchParams::new().with_phrase_slop(Some(0)),
            PreFilterSource::None,
        )
        .unwrap();
        phrase_query
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap();
        let metrics = phrase_query.metrics().unwrap();
        assert!(metrics.elapsed_compute().unwrap() > 0);

        let boost_input_one = MatchQueryExec::new(
            Arc::new(fixture.dataset.clone()),
            MatchQuery::new("blah".to_string())
                .with_column(Some("text".to_string()))
                .with_document_granularity(DocumentGranularity::Row),
            FtsSearchParams::default(),
            PreFilterSource::None,
        )
        .unwrap();

        let boost_input_two = MatchQueryExec::new(
            Arc::new(fixture.dataset),
            MatchQuery::new("blah".to_string())
                .with_column(Some("text".to_string()))
                .with_document_granularity(DocumentGranularity::Row),
            FtsSearchParams::default(),
            PreFilterSource::None,
        )
        .unwrap();

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
        let query_tokens = collect_query_tokens("hello", &mut tokenizer);

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

    #[tokio::test]
    async fn test_analyze_plan_shows_indexed_and_flat_match_tokens() {
        let dataset = create_tokenized_query_fixture(&["text"], true).await;
        let query = MatchQuery::new("FIRST and SECOND".to_string())
            .with_column(Some("text".to_string()))
            .with_operator(Operator::And);
        let mut scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new_query(query.into()))
            .unwrap();

        let explained = scanner.explain_plan(false).await.unwrap();
        assert!(
            !explained.contains("tokenized_query="),
            "explain_plan should not claim runtime tokenization: {explained}"
        );

        let analysis = scanner.analyze_plan().await.unwrap();
        let expected = r#"tokenized_query=[("first", 0), ("second", 2)]"#;
        assert!(
            find_plan_line(&analysis, "MatchQuery:").contains(expected),
            "indexed MatchQuery is missing token positions:\n{analysis}"
        );
        assert!(
            find_plan_line(&analysis, "FlatMatchQuery:").contains(expected),
            "flat MatchQuery is missing token positions:\n{analysis}"
        );
    }

    #[tokio::test]
    async fn test_analyze_plan_shows_indexed_and_flat_phrase_tokens() {
        let dataset = create_tokenized_query_fixture(&["text"], true).await;
        let query =
            PhraseQuery::new("FIRST and SECOND".to_string()).with_column(Some("text".to_string()));
        let mut scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new_query(query.into()))
            .unwrap();

        let analysis = scanner.analyze_plan().await.unwrap();
        let expected = r#"tokenized_query=[("first", 0), ("second", 2)]"#;
        assert!(
            find_plan_line(&analysis, "PhraseQuery:").contains(expected),
            "indexed PhraseQuery is missing token positions:\n{analysis}"
        );
        assert!(
            find_plan_line(&analysis, "FlatMatchQuery:").contains(expected),
            "flat phrase path is missing token positions:\n{analysis}"
        );
    }

    #[tokio::test]
    async fn test_analyze_plan_shows_compound_leaf_tokens() {
        let dataset = create_tokenized_query_fixture(&["text"], false).await;
        let query = BooleanQuery::new([
            (
                Occur::Should,
                MatchQuery::new("FIRST and SECOND".to_string())
                    .with_column(Some("text".to_string()))
                    .into(),
            ),
            (
                Occur::Must,
                PhraseQuery::new("SECOND FIRST".to_string())
                    .with_column(Some("text".to_string()))
                    .with_slop(2)
                    .into(),
            ),
        ]);
        let mut scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new_query(query.into()))
            .unwrap();

        let analysis = scanner.analyze_plan().await.unwrap();
        let compound = find_plan_line(&analysis, "CompoundFtsScorer:");
        assert!(
            compound.contains(r#"Match(column="text", tokens=[("first", 0), ("second", 2)])"#),
            "compound Match leaf is missing token positions: {compound}"
        );
        assert!(
            compound.contains(r#"Phrase(column="text", tokens=[("second", 0), ("first", 1)])"#),
            "compound Phrase leaf is missing token positions: {compound}"
        );
    }

    /// A `combined_fields` scan is a leaf FTS node, so `analyze_plan` must show
    /// its tokens the way it does for `match` and `phrase`. Both children of a
    /// mixed plan report, or a `Boolean{combined_fields, match}` plan would show
    /// tokens on the `match` sibling only.
    #[tokio::test]
    async fn test_analyze_plan_shows_combined_fields_tokens() {
        let dataset = create_tokenized_query_fixture(&["title", "body"], true).await;
        let query = CombinedFieldsQuery::try_new(
            "FIRST and SECOND".to_string(),
            vec!["title".to_string(), "body".to_string()],
        )
        .unwrap();
        let mut scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new_query(query.into()))
            .unwrap();

        let explained = scanner.explain_plan(false).await.unwrap();
        assert!(
            !explained.contains("tokenized_query="),
            "explain_plan should not claim runtime tokenization: {explained}"
        );

        let analysis = scanner.analyze_plan().await.unwrap();
        let expected = r#"tokenized_query=[("first", 0), ("second", 2)]"#;
        assert!(
            find_plan_line(&analysis, "CombinedFieldsQuery:").contains(expected),
            "indexed combined_fields is missing token positions:\n{analysis}"
        );
        assert!(
            find_plan_line(&analysis, "FlatCombinedFields:").contains(expected),
            "flat combined_fields is missing token positions:\n{analysis}"
        );
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
        let compound_line = analysis
            .lines()
            .find(|line| line.contains("CompoundFtsScorer"))
            .unwrap();
        assert!(
            compound_line.contains(&format!("{PARTITIONS_SEARCHED_METRIC}={expected_total}")),
            "compound FTS scorer metrics missing partitions_searched: {compound_line}"
        );
    }

    /// The cross-field scorer build reads one posting-metadata row per
    /// (term, column, partition), and both `combined_fields` execs must report
    /// those to their query's index-cache counters. Unreported, `EXPLAIN ANALYZE`
    /// on a cold cross-field query undercounts `index_cache_misses`, so its
    /// `index_cache_hit_ratio` is not comparable with `match`'s on the same data.
    #[tokio::test]
    async fn test_combined_fields_scorer_build_reports_index_cache_metrics() {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "title",
                lance_datagen::array::cycle_utf8_literals(&[
                    "hello world",
                    "lance search",
                    "hello lance",
                ]),
            )
            .col(
                "body",
                lance_datagen::array::cycle_utf8_literals(&["lance", "hello", "search hello"]),
            )
            .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(9))
            .await
            .unwrap();
        for column in ["title", "body"] {
            dataset
                .create_index(
                    &[column],
                    IndexType::Inverted,
                    None,
                    &InvertedIndexParams::default(),
                    true,
                )
                .await
                .unwrap();
        }
        let dataset = Arc::new(dataset);

        let query = CombinedFieldsQuery::try_new(
            "hello lance".to_string(),
            vec!["title".to_string(), "body".to_string()],
        )
        .unwrap();
        let params = FtsSearchParams::default().with_limit(Some(10));

        // Open the same segments the execs will, both to size the expected lookup
        // count and to hand the preset runs a scorer identical to the one they
        // would have built.
        let mut columns = Vec::new();
        for (column, weight) in query.weighted_columns() {
            let segments = crate::index::scalar::inverted::load_segments(
                &dataset,
                column,
                DocumentGranularity::Row,
            )
            .await
            .unwrap()
            .expect("FTS index just created");
            let metrics_set = ExecutionPlanMetricsSet::new();
            let indices = open_fts_segments(
                &dataset,
                column,
                &segments,
                &IndexMetrics::new(&metrics_set, 0),
            )
            .await
            .unwrap();
            columns.push(CombinedFieldColumn {
                column: column.to_string(),
                weight,
                indices,
            });
        }
        let mut tokenizer = columns[0].indices[0].tokenizer();
        let tokens = collect_query_tokens(query.terms(), &mut tokenizer);
        let mut terms = (0..tokens.len())
            .map(|index| tokens.get_token(index).to_string())
            .collect::<Vec<_>>();
        terms.sort_unstable();
        terms.dedup();
        let partitions: usize = columns
            .iter()
            .flat_map(|column| &column.indices)
            .map(|index| index.partition_count())
            .sum();
        // One posting-metadata row per (term, column, partition).
        let expected_lookups = terms.len() * partitions;
        assert!(expected_lookups > 0);
        let scorer = Arc::new(
            build_combined_bm25_scorer(&columns, &tokens, CombinedCorpusStats::IndexOnly, None)
                .await
                .unwrap(),
        );

        // Warm the index cache first: a cold run's counts depend on which entry
        // each lookup happens to load, while on a warm cache every lookup is a
        // hit and the two runs below differ only in the scorer build.
        let warmup = CombinedFieldsQueryExec::new(
            dataset.clone(),
            query.clone(),
            params.clone(),
            PreFilterSource::None,
        );
        let expected = execute_results(&warmup).await.unwrap();
        assert!(!expected.is_empty());

        let built = CombinedFieldsQueryExec::new(
            dataset.clone(),
            query.clone(),
            params.clone(),
            PreFilterSource::None,
        );
        assert_eq!(execute_results(&built).await.unwrap(), expected);
        let preset = CombinedFieldsQueryExec::new(
            dataset.clone(),
            query.clone(),
            params.clone(),
            PreFilterSource::None,
        )
        .with_base_scorer(scorer.clone());
        assert_eq!(
            execute_results(&preset).await.unwrap(),
            expected,
            "the preset scorer must match the one the exec builds",
        );
        assert_eq!(
            metric_value(&built, INDEX_CACHE_HITS_METRIC)
                - metric_value(&preset, INDEX_CACHE_HITS_METRIC),
            expected_lookups,
            "CombinedFieldsQueryExec must report the scorer build's cache lookups",
        );

        // The flat sibling builds its own scorer from the same per-column index
        // statistics. An empty input isolates those reads: the scanned rows
        // contribute none, so the count is absolute rather than a delta.
        let flat_schema = Arc::new(arrow_schema::Schema::new(vec![
            lance_core::ROW_ID_FIELD.clone(),
            arrow_schema::Field::new("title", DataType::Utf8, true),
            arrow_schema::Field::new("body", DataType::Utf8, true),
        ]));
        let resolved_fields = query
            .column_names()
            .map(|column| {
                crate::index::scalar::inverted::resolve_fts_field(
                    dataset.schema(),
                    column,
                    DocumentGranularity::Row,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let flat = FlatCombinedFieldsExec::new(
            dataset.clone(),
            query.clone(),
            params.clone(),
            Arc::new(EmptyExec::new(flat_schema)),
        )
        .with_resolved_fields(resolved_fields);
        assert!(execute_results(&flat).await.unwrap().is_empty());
        assert_eq!(
            metric_value(&flat, INDEX_CACHE_HITS_METRIC)
                + metric_value(&flat, INDEX_CACHE_MISSES_METRIC),
            expected_lookups,
            "FlatCombinedFieldsExec must report the scorer build's cache lookups",
        );
    }

    #[tokio::test]
    async fn test_match_query_exec_segment_selection() {
        let (dataset, segments, fragment_ids) = create_segment_selection_fixture().await;
        let query = MatchQuery::new("quick".to_string())
            .with_column(Some("text".to_string()))
            .with_document_granularity(DocumentGranularity::Row);
        let params = FtsSearchParams::default().with_limit(Some(20));
        let committed_uuids = segments
            .iter()
            .map(|segment| segment.uuid)
            .collect::<Vec<_>>();

        let all_committed = MatchQueryExec::new(
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

        let exact_resolved = MatchQueryExec::new_with_segments(
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
            Some(committed_uuids.clone())
        );
        assert_eq!(execute_results(&exact_resolved).await.unwrap(), all_results);
        assert_eq!(
            metric_value(&exact_resolved, FTS_SEGMENT_BIND_DURATION_METRIC),
            0
        );

        let mismatched_granularity = MatchQueryExec::new_with_segments_and_document_granularity(
            dataset.clone(),
            query.clone(),
            params.clone(),
            PreFilterSource::None,
            segments.clone(),
            DocumentGranularity::ListElement,
        );
        assert_execution_error(
            execute_row_ids(&mismatched_granularity).await.unwrap_err(),
            "use Row document granularity",
        );

        let selected_fragment = fragment_ids[1];
        let selected_uuid = segment_uuid_for_fragment(&segments, selected_fragment);
        let unpolled = MatchQueryExec::new_with_segment_uuids(
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

        let exact_uuids = MatchQueryExec::new_with_segment_uuids(
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
            MatchQueryExec::new_with_segment_uuids(
                dataset.clone(),
                query.clone(),
                params.clone(),
                PreFilterSource::None,
                input_uuids,
            )
            .unwrap(),
        )
        .with_new_children(vec![])
        .unwrap();
        let rewritten = ordered_plan.downcast_ref::<MatchQueryExec>().unwrap();
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

        let empty = MatchQueryExec::new_with_segment_uuids(
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
        let missing = MatchQueryExec::new_with_segment_uuids(
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

        let wrong_column = MatchQueryExec::new_with_segment_uuids(
            dataset,
            MatchQuery::new("quick".to_string())
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

    #[tokio::test]
    async fn test_combined_fields_exec_opens_all_committed_segments() {
        let (dataset, _segments, fragment_ids) = create_segment_selection_fixture().await;
        let query = CombinedFieldsQuery::try_new("quick".to_string(), vec!["text".to_string()])
            .unwrap()
            .try_with_boosts(vec![2.0])
            .unwrap();
        let params = FtsSearchParams::default().with_limit(Some(20));

        // One segment per fragment, so a scan that opens every committed segment
        // returns every fragment's matching row.
        let exec = CombinedFieldsQueryExec::new(dataset, query, params, PreFilterSource::None);
        assert_eq!(
            execute_row_ids(&exec).await.unwrap(),
            expected_row_ids(&fragment_ids)
        );
    }

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
        .with_new_children(vec![])
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
        let query = MatchQuery::new("lance".to_string())
            .with_column(Some("text".to_string()))
            .with_document_granularity(DocumentGranularity::Row);
        let search_params = FtsSearchParams::default().with_limit(Some(10));

        // Baseline: the existing path that builds the global scorer locally.
        let baseline_exec = MatchQueryExec::new(
            dataset.clone(),
            query.clone(),
            search_params.clone(),
            PreFilterSource::None,
        )
        .unwrap();
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
        let preset_segments = crate::index::scalar::inverted::load_segments(
            &dataset,
            "text",
            DocumentGranularity::Row,
        )
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
            build_global_bm25_scorer(&indices, &tokens, &search_params, None)
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
        .unwrap()
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

    #[tokio::test]
    async fn test_compound_query_exec_validates_base_scorer() {
        let (dataset, segments, _) = create_segment_selection_fixture().await;
        let search_params = FtsSearchParams::default().with_limit(Some(10));
        let metrics_set = ExecutionPlanMetricsSet::new();
        let metrics = IndexMetrics::new(&metrics_set, 0);
        let indices = open_fts_segments(&dataset, "text", &segments, &metrics)
            .await
            .unwrap();

        let query: FtsQuery = BooleanQuery::new([
            (
                Occur::Should,
                MatchQuery::new("quick".to_string())
                    .with_column(Some("text".to_string()))
                    .into(),
            ),
            (
                Occur::Should,
                MatchQuery::new("brown".to_string())
                    .with_column(Some("text".to_string()))
                    .into(),
            ),
        ])
        .into();

        let baseline = CompoundQueryExec::new_with_segments(
            dataset.clone(),
            query.clone(),
            search_params.clone(),
            PreFilterSource::None,
            segments.clone(),
        );
        let baseline_results = execute_results(&baseline).await.unwrap();

        let mut tokenizer = indices[0].tokenizer();
        let complete_tokens = collect_query_tokens("quick brown", &mut tokenizer);
        let complete_scorer = Arc::new(
            build_global_bm25_scorer(&indices, &complete_tokens, &search_params, None)
                .await
                .unwrap(),
        );
        let complete_override = CompoundQueryExec::new_with_segments(
            dataset.clone(),
            query.clone(),
            search_params.clone(),
            PreFilterSource::None,
            segments.clone(),
        )
        .with_base_scorer(complete_scorer);
        assert_eq!(
            execute_results(&complete_override).await.unwrap(),
            baseline_results
        );

        let mut tokenizer = indices[0].tokenizer();
        let incomplete_tokens = collect_query_tokens("quick", &mut tokenizer);
        let incomplete_scorer = Arc::new(
            build_global_bm25_scorer(&indices, &incomplete_tokens, &search_params, None)
                .await
                .unwrap(),
        );
        let incomplete_override = CompoundQueryExec::new_with_segments(
            dataset.clone(),
            query,
            search_params.clone(),
            PreFilterSource::None,
            segments.clone(),
        )
        .with_base_scorer(incomplete_scorer);

        let error = execute_results(&incomplete_override).await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("injected BM25 scorer is missing compound FTS token 'brown'"),
            "unexpected incomplete-scorer error: {error}"
        );

        let mut tokenizer = indices[0].tokenizer();
        let brown_tokens = collect_query_tokens("brown", &mut tokenizer);
        let scorer_without_fuzzy_expansion = Arc::new(
            build_global_bm25_scorer(&indices, &brown_tokens, &search_params, None)
                .await
                .unwrap(),
        );
        let fuzzy_query = BooleanQuery::new([
            (
                Occur::Should,
                MatchQuery::new("quik".to_string())
                    .with_column(Some("text".to_string()))
                    .with_fuzziness(Some(1))
                    .into(),
            ),
            (
                Occur::Should,
                MatchQuery::new("brown".to_string())
                    .with_column(Some("text".to_string()))
                    .into(),
            ),
        ]);
        let fuzzy_override = CompoundQueryExec::new_with_segments(
            dataset,
            fuzzy_query.into(),
            search_params,
            PreFilterSource::None,
            segments,
        )
        .with_base_scorer(scorer_without_fuzzy_expansion);
        let error = execute_results(&fuzzy_override).await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("injected BM25 scorer is missing compound FTS token 'quick'"),
            "unexpected fuzzy-scorer error: {error}"
        );
    }

    #[tokio::test]
    async fn test_cross_column_compound_exec_validates_constructor_inputs() {
        let (dataset, segments, _) = create_segment_selection_fixture().await;
        let query: FtsQuery = BooleanQuery::new([
            (
                Occur::Must,
                MatchQuery::new("quick".to_string())
                    .with_column(Some("title".to_string()))
                    .into(),
            ),
            (
                Occur::MustNot,
                MatchQuery::new("blocked".to_string())
                    .with_column(Some("body".to_string()))
                    .into(),
            ),
        ])
        .into();
        let params = FtsSearchParams::default().with_limit(Some(10));

        let error = CrossColumnCompoundQueryExec::new_with_segments(
            dataset.clone(),
            query.clone(),
            params.clone(),
            PreFilterSource::None,
            vec![("title".to_string(), segments.clone())],
        )
        .unwrap_err();
        assert!(
            error.to_string().contains(r#"missing=["body"]"#),
            "unexpected missing-column error: {error}"
        );

        let error = CrossColumnCompoundQueryExec::new_with_segments(
            dataset.clone(),
            query.clone(),
            FtsSearchParams::default(),
            PreFilterSource::None,
            vec![
                ("title".to_string(), segments.clone()),
                ("body".to_string(), segments.clone()),
            ],
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("requires a bounded result limit"),
            "unexpected unbounded-query error: {error}"
        );

        let exec = CrossColumnCompoundQueryExec::new_with_segments(
            dataset,
            query,
            params,
            PreFilterSource::None,
            vec![
                ("title".to_string(), segments.clone()),
                ("body".to_string(), segments),
            ],
        )
        .unwrap();
        let display = format!(
            "{}",
            datafusion::physical_plan::displayable(&exec).one_line()
        );
        assert!(
            display.contains("CrossColumnCompoundFtsScorer:"),
            "unexpected display name: {display}"
        );
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
            plan.downcast_ref::<EmptyExec>().is_some(),
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
            .downcast_ref::<RepartitionExec>()
            .expect("multi-child Should should be wrapped in RepartitionExec");
        let inner = repartition
            .input()
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
        while let Some(join) = current.clone().downcast_ref::<HashJoinExec>() {
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
            .downcast_ref::<RepartitionExec>()
            .expect("multi-child MustNot should be wrapped in RepartitionExec");
        let inner = repartition
            .input()
            .downcast_ref::<UnionExec>()
            .expect("RepartitionExec should wrap a UnionExec");
        assert_eq!(inner.children().len(), 2);
    }
}
