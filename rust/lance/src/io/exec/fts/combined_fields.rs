// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Cross-field BM25F (`combined_fields`) execution node.

use std::sync::{Arc, OnceLock};

use arrow_array::{Float32Array, RecordBatch, UInt64Array};
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
use lance_core::utils::tracing::StreamTracingExt;
use lance_select::RowAddrMask;
use lance_table::format::IndexMetadata;

use crate::Dataset;
use crate::index::scalar::inverted::{load_segment_details, load_segments};
use crate::io::exec::PreFilterSource;
use crate::io::exec::utils::{
    PreFilterMasks, build_prefilter, build_prefilter_restricted_to_fragments,
};
use lance_index::prefilter::PreFilter;
use lance_index::scalar::inverted::document_tokenizer::LanceTokenizer;
use lance_index::scalar::inverted::query::{
    CombinedFieldsQuery, FtsSearchParams, Tokens, try_collect_query_tokens,
};
use lance_index::scalar::inverted::{
    CombinedCorpusStats, CombinedFieldColumn, CombinedFieldsBM25Scorer, DocumentGranularity,
    FTS_SCHEMA, build_combined_bm25_scorer, combined_fields_search, validate_combined_tokenizers,
};
use lance_select::{RowAddrTreeMap, RowSetOps};
use tracing::instrument;

use super::corpus::SharedFtsScorer;
use super::query_tokens::{TokenizedQuery, fmt_tokenized_query, record_tokenized_query};
use super::segments::{combined_columns_the_prefilter_may_match, open_fts_segments};
use super::selection::{FtsIndexMetrics, FtsSegmentSelection};

/// Cross-field BM25F full-text search (`combined_fields`).
///
/// Unlike a `MultiMatch` plan (one [`super::MatchQueryExec`] per column fused by a
/// `max` aggregate, i.e. `best_fields`), this is a single node: it opens every
/// target column's segments, blends their corpus statistics, and runs one merged
/// scan so term statistics are shared across fields. Per-column boosts are baked
/// into the blended term frequency, so there is no post-scan boost multiplier or
/// `AggregateExec(max)`. Emits `(ROW_ID, SCORE)` in [`FTS_SCHEMA`].
///
/// Reconstruct the leaf with its complete candidate scope and scorer override.
/// A shared scorer must be reconnected by the enclosing corpus owner; it is not
/// an independently executable leaf after decoding.
///
/// ```
/// # use lance::{io::exec::fts::CombinedFieldsQueryExec, Error, Result};
/// # fn rebuild(exec: &CombinedFieldsQueryExec) -> Result<CombinedFieldsQueryExec> {
/// if exec.has_shared_scorer() {
///     return Err(Error::invalid_input("reconstruct the enclosing corpus owner"));
/// }
/// let mut rebuilt = CombinedFieldsQueryExec::new(
///     exec.dataset().clone(), exec.query().clone(), exec.params().clone(),
///     exec.prefilter_source().clone(),
/// ).with_external_mask(exec.external_mask().cloned());
/// if let Some(fragments) = exec.covered_fragments() {
///     rebuilt = rebuilt.with_covered_fragments(fragments.clone());
/// }
/// if let Some(mask) = exec.overlay_block() {
///     rebuilt = rebuilt.with_overlay_block(mask.clone());
/// }
/// if let Some(scorer) = exec.base_scorer() {
///     rebuilt = rebuilt.with_base_scorer(scorer.clone());
/// }
/// # Ok(rebuilt)
/// # }
/// ```
#[derive(Debug, Clone)]

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
    /// See [`Self::with_overlay_block`].
    overlay_block: Option<RowAddrMask>,
    /// Optional external row-address mask ANDead into the prefilter so only
    /// masked rows are scored (see [`super::MatchQueryExec::with_external_mask`]).
    external_mask: Option<Arc<RowAddrMask>>,
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
            overlay_block: None,
            external_mask: None,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// See [`super::MatchQueryExec::with_external_mask`].
    pub fn with_external_mask(mut self, mask: Option<Arc<RowAddrMask>>) -> Self {
        self.external_mask = mask;
        self
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

    /// Exclude the rows whose indexed text a newer data overlay superseded in any
    /// target column. The flat sibling scores them from their current values.
    /// The mask uses the index's row-id domain, including stable row IDs when
    /// enabled. See [`Self`] for an example preserving it during reconstruction.
    pub fn with_overlay_block(mut self, overlay_block: RowAddrMask) -> Self {
        self.overlay_block = Some(overlay_block);
        self
    }

    /// Override the blended BM25F scorer used by `execute()`. When set, the
    /// local `build_combined_bm25_scorer` call is skipped. Mirrors
    /// [`super::MatchQueryExec::with_base_scorer`] for distributed queries that
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

    pub(super) fn uses_shared_scorer(
        &self,
        scorer: &Arc<SharedFtsScorer<CombinedFieldsBM25Scorer>>,
    ) -> bool {
        self.shared_scorer
            .as_ref()
            .is_some_and(|shared| Arc::ptr_eq(shared, scorer))
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

    /// The opened snapshot that supplies the schema and committed indices.
    /// See [`Self`] for a reconstruction example.
    pub fn dataset(&self) -> &Arc<Dataset> {
        &self.dataset
    }

    /// Candidate fragments covered by every target column, or `None` for the
    /// normal index-derived scope. `Some(empty)` selects no candidates.
    pub fn covered_fragments(&self) -> Option<&roaring::RoaringBitmap> {
        self.covered_fragments.as_ref()
    }

    /// Overlay-stale rows excluded from indexed results, in the index's row-id
    /// domain. `None` means no additional overlay exclusion.
    pub fn overlay_block(&self) -> Option<&RowAddrMask> {
        self.overlay_block.as_ref()
    }

    /// Additional candidate mask, or `None` for no external restriction.
    /// It uses the index's row-id domain and does not narrow the statistics corpus.
    pub fn external_mask(&self) -> Option<&Arc<RowAddrMask>> {
        self.external_mask.as_ref()
    }

    /// Scorer override, or `None` to use shared or locally built statistics.
    /// An override takes precedence over the enclosing owner's shared scorer.
    pub fn base_scorer(&self) -> Option<&Arc<CombinedFieldsBM25Scorer>> {
        self.base_scorer.as_ref()
    }

    /// Whether reconstruction must reconnect this leaf to its corpus owner.
    /// The process-local coordination object is deliberately not exported.
    pub fn has_shared_scorer(&self) -> bool {
        self.shared_scorer.is_some()
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
            overlay_block: self.overlay_block.clone(),
            external_mask: self.external_mask.clone(),
            properties: self.properties.clone(),
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl DisplayAs for CombinedFieldsQueryExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt_combined_fields(
            "CombinedFieldsQuery",
            &self.query,
            &self.tokenized_query,
            t,
            f,
        )
    }
}

/// The plan display every `combined_fields` exec uses: the label, the target
/// columns and the query text, plus the tokens `execute()` produced once it has
/// run.
pub(super) fn fmt_combined_fields(
    label: &str,
    query: &CombinedFieldsQuery,
    tokenized_query: &OnceLock<TokenizedQuery>,
    t: DisplayFormatType,
    f: &mut std::fmt::Formatter,
) -> std::fmt::Result {
    let columns = query.column_names().collect::<Vec<_>>().join(", ");
    // The one-line forms label with `: ` and separate fields with `, `; the tree
    // render puts every field on its own line.
    let (after_label, separator) = match t {
        DisplayFormatType::TreeRender => ("\n", "\n"),
        DisplayFormatType::Default | DisplayFormatType::Verbose => (": ", ", "),
    };
    write!(
        f,
        "{label}{after_label}columns=[{columns}]{separator}query={}",
        query.terms()
    )?;
    fmt_tokenized_query(tokenized_query, separator, f)
}

/// Which target columns of a `combined_fields` scan need an inverted index.
#[derive(Debug, Clone, Copy)]
pub(super) enum CombinedFieldsIndexRequirement {
    /// Every column, because the indexed side scores each of them from its
    /// index.
    EveryColumn,
    /// At least one column, for the tokenizer. The flat side reads the document
    /// text from the scan, so a column without an index only misses out on
    /// contributing corpus statistics.
    AtLeastOneColumn,
}

/// What [`open_combined_fields_scan`] hands back to either `combined_fields` exec.
pub(super) struct CombinedFieldsScan {
    pub(super) columns: Vec<CombinedFieldColumn>,
    /// Every segment opened across all target columns, for the indexed side's
    /// prefilter.
    segments: Vec<IndexMetadata>,
    pub(super) tokens: Tokens,
    pub(super) tokenizer: Box<dyn LanceTokenizer>,
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
///
/// `stale_rows` is empty, or one entry per target column in query column order.
/// See [`CombinedFieldColumn::stale_rows`].
pub(super) async fn open_combined_fields_scan(
    dataset: &Dataset,
    query: &CombinedFieldsQuery,
    index_requirement: CombinedFieldsIndexRequirement,
    stale_rows: &[Arc<RowAddrTreeMap>],
    tokenized_query: &OnceLock<TokenizedQuery>,
    metrics: &FtsIndexMetrics,
) -> DataFusionResult<CombinedFieldsScan> {
    let mut columns = Vec::with_capacity(query.column_names().len());
    let mut all_segments = Vec::new();
    for (slot, (column, weight)) in query.weighted_columns().enumerate() {
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
            stale_rows: stale_rows
                .get(slot)
                .filter(|rows| !rows.is_empty())
                .cloned(),
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
    let tokens = try_collect_query_tokens(query.terms(), &mut tokenizer)?;
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
        self.prefilter_source.execution_plan().into_iter().collect()
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
        let expected = self.children().len();
        if children.len() != expected {
            return Err(DataFusionError::Internal(format!(
                "CombinedFieldsQueryExec expected {expected} children, got {}",
                children.len()
            )));
        }
        let prefilter_source = match children.pop() {
            Some(source) => self.prefilter_source.with_execution_plan(source)?,
            None => PreFilterSource::None,
        };
        Ok(Arc::new(self.clone_with_prefilter_source(prefilter_source)))
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
        let masks = PreFilterMasks {
            overlay_block: self.overlay_block.clone(),
            external_mask: self.external_mask.clone(),
        };
        let exec_metrics = self.metrics.clone();
        let metrics = Arc::new(FtsIndexMetrics::new(&self.metrics, partition));
        let stream = stream::once(async move {
            let _timer = metrics.baseline_metrics.elapsed_compute().timer();
            let address_domain = !ds.manifest.uses_stable_row_ids();
            let ds_for_scope = ds.clone();

            let CombinedFieldsScan {
                columns,
                segments: all_segments,
                tokens,
                ..
            } = open_combined_fields_scan(
                &ds,
                &query,
                CombinedFieldsIndexRequirement::EveryColumn,
                // Nothing to subtract: this side either scores against index
                // statistics alone or against the scorer the flat sibling built,
                // and only that sibling folds a stale row in.
                &[],
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
                    masks,
                    &exec_metrics,
                )?,
                None => build_prefilter(
                    context.clone(),
                    partition,
                    &prefilter_source,
                    ds,
                    &all_segments,
                    masks,
                    &exec_metrics,
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
            let columns = combined_columns_the_prefilter_may_match(
                &ds_for_scope,
                columns,
                &all_segments,
                pre_filter.as_ref(),
                address_domain,
                &metrics,
            )
            .await;
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
