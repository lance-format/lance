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
use futures::{FutureExt, StreamExt, TryStreamExt};
use lance_core::utils::tracing::StreamTracingExt;
use lance_select::RowAddrMask;
use lance_table::format::IndexMetadata;

use crate::Dataset;
use crate::index::scalar::inverted::{load_segment_details, load_segment_params, load_segments};
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
    CombinedCorpusStats, CombinedFieldColumn, CombinedFieldsBM25Scorer, CombinedFieldsSearch,
    DocumentGranularity, FTS_SCHEMA, InvertedIndexParams, combined_fields_search,
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
/// if let Some(domain) = exec.statistics_fragments() {
///     rebuilt = rebuilt.with_statistics_fragments(domain.clone())?;
/// }
/// # Ok(rebuilt)
/// # }
/// ```
#[derive(Debug, Clone)]

pub struct CombinedFieldsQueryExec {
    dataset: Arc<Dataset>,
    query: CombinedFieldsQuery,
    statistics_fragments: Option<roaring::RoaringBitmap>,
    /// Tokens `execute()` actually produced, for `analyze_plan`. One snapshot
    /// covers every target column because their index metadata
    /// must declare the same tokenizer, so the scan tokenizes once.
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
    /// Use a fixed physical-fragment statistics domain instead of the complete
    /// corpus. Set [`Self::with_covered_fragments`] first: candidates must stay
    /// within the domain. Retained committed-index rows still count after
    /// deletion, while candidate filtering never changes these statistics.
    ///
    /// This requires physical row IDs and modern indices without overlays or a
    /// preset scorer. An empty domain requires empty candidate coverage. A mixed
    /// plan's flat sibling must use exactly the same statistics domain.
    ///
    /// ```
    /// # use lance::io::exec::fts::CombinedFieldsQueryExec;
    /// # use datafusion::error::Result;
    /// # fn configure(exec: CombinedFieldsQueryExec) -> Result<CombinedFieldsQueryExec> {
    /// let domain = roaring::RoaringBitmap::from_iter([0, 1]);
    /// exec.with_covered_fragments(domain.clone()).with_statistics_fragments(domain)
    /// # }
    /// ```
    pub fn with_statistics_fragments(
        mut self,
        fragments: roaring::RoaringBitmap,
    ) -> DataFusionResult<Self> {
        self.statistics_fragments = Some(fragments);
        self.validate_statistics_scope()?;
        Ok(self)
    }

    fn validate_statistics_scope(&self) -> DataFusionResult<()> {
        let Some(scope) = &self.statistics_fragments else {
            return Ok(());
        };
        if self.dataset.manifest().uses_stable_row_ids()
            || self.overlay_block.is_some()
            || self.base_scorer.is_some()
        {
            return Err(DataFusionError::Plan(
                "BM25F fragment statistics require physical row IDs without overlay or preset scorer".into()));
        }
        if self
            .covered_fragments
            .as_ref()
            .is_none_or(|candidates| !candidates.is_subset(scope))
        {
            return Err(DataFusionError::Plan(
                "BM25F candidate fragments must be explicitly contained in the statistics domain"
                    .into(),
            ));
        }
        Ok(())
    }

    /// The fixed fragment statistics domain. `None` uses the complete corpus;
    /// `Some(empty)` contributes no indexed rows.
    pub fn statistics_fragments(&self) -> Option<&roaring::RoaringBitmap> {
        self.statistics_fragments.as_ref()
    }

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
            statistics_fragments: None,
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
            statistics_fragments: self.statistics_fragments.clone(),
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
    pub(super) statistics_rows: Option<Arc<RowAddrTreeMap>>,
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
/// Both sides count opened segments toward parts searched. The flat side needs
/// covered segments for blended corpus statistics; a completely uncovered shard
/// uses tokenizer metadata without opening unrelated segments.
///
/// `stale_rows` is empty, or one entry per target column in query column order.
/// See [`CombinedFieldColumn::stale_rows`].
pub(super) async fn open_combined_fields_scan(
    dataset: &Dataset,
    query: &CombinedFieldsQuery,
    statistics_fragments: Option<&roaring::RoaringBitmap>,
    index_requirement: CombinedFieldsIndexRequirement,
    stale_rows: &[Arc<RowAddrTreeMap>],
    tokenized_query: &OnceLock<TokenizedQuery>,
    metrics: &FtsIndexMetrics,
) -> DataFusionResult<CombinedFieldsScan> {
    let statistics_rows = if let Some(fragments) = statistics_fragments {
        if dataset.manifest().uses_stable_row_ids() {
            return Err(DataFusionError::Plan(
                "BM25F fragment statistics require physical row IDs".into(),
            ));
        }
        let mut rows = RowAddrTreeMap::new();
        // Candidate membership still uses live fragments. Statistics retain
        // committed indexed rows, including fragments removed by deletion.
        for fragment in fragments {
            rows.insert_fragment(fragment);
        }
        Some(Arc::new(rows))
    } else {
        None
    };
    let mut columns = Vec::with_capacity(query.column_names().len());
    let mut all_segments = Vec::new();
    let mut tokenizer_params: Option<(&str, InvertedIndexParams)> = None;
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
        if let Some(segments) = &segments {
            load_segment_details(dataset, column, segments).await?;
        }
        let in_statistics_domain = |segment: &IndexMetadata| {
            statistics_fragments.is_none_or(|scope| {
                segment
                    .fragment_bitmap
                    .as_ref()
                    .is_none_or(|coverage| !coverage.is_disjoint(scope))
            })
        };
        let selected = segments.as_ref().map(|segments| {
            if statistics_fragments.is_none() {
                return segments.clone();
            }
            Arc::from(
                segments
                    .iter()
                    .filter(|segment| in_statistics_domain(segment))
                    .cloned()
                    .collect::<Vec<_>>(),
            )
        });
        let indices = match selected {
            Some(selected) => {
                let indices =
                    open_fts_segments(dataset, column, &selected, &metrics.index_metrics).await?;
                all_segments.extend(selected.iter().cloned());
                indices
            }
            None => Vec::new(),
        };

        // Opened indices already contain the complete persisted parameters,
        // including explicit document tokenizers absent from manifest details.
        // Reopening their metadata adds an extra cold read per segment.
        let selected_params = stream::iter(indices.iter().map(|index| Ok(index.params().clone())));
        // Omitted segments still participate in configuration validation. Read
        // only their parameters, without partitions, and bound concurrent I/O.
        let mut omitted_params = Vec::new();
        if let Some(segments) = &segments {
            for segment in segments.iter() {
                if !in_statistics_domain(segment) {
                    omitted_params.push(load_segment_params(dataset, segment));
                }
            }
        }
        let omitted_params =
            stream::iter(omitted_params).buffered(dataset.object_store.io_parallelism());
        let mut field_params = selected_params.chain(omitted_params);
        while let Some(params) = field_params.try_next().await? {
            if let Some((reference, expected)) = &tokenizer_params {
                if !expected.same_tokenization(&params) {
                    return Err(DataFusionError::Execution(format!(
                        "combined_fields requires every target column to share the same tokenizer/index configuration; column '{column}' differs from column '{reference}'"
                    )));
                }
            } else {
                tokenizer_params = Some((column, params));
            }
        }
        if statistics_fragments.is_some() && indices.iter().any(|index| index.is_legacy()) {
            return Err(DataFusionError::Plan(
                "BM25F fragment statistics require modern DocId-keyed indices".into(),
            ));
        }
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
    metrics.record_parts_searched(
        columns
            .iter()
            .flat_map(|column| &column.indices)
            .map(|index| index.partition_count())
            .sum(),
    );

    // Fields share a tokenizer (validated above), so tokenize once.
    let (_, params) = tokenizer_params.ok_or_else(|| match index_requirement {
        // Not a user error: `EveryColumn` resolution already failed for any
        // column without an index, and `CombinedFieldsQuery::try_new` rejects
        // an empty column list, so reaching this means one of those two
        // guarantees broke.
        CombinedFieldsIndexRequirement::EveryColumn => DataFusionError::Internal(
            "combined_fields query reached execution with no target columns".to_string(),
        ),
        CombinedFieldsIndexRequirement::AtLeastOneColumn => DataFusionError::Execution(format!(
            "combined_fields requires an inverted index on at least one of {:?}",
            query.column_names().collect::<Vec<_>>()
        )),
    })?;
    let mut tokenizer = params.build()?;
    let tokens = try_collect_query_tokens(query.terms(), &mut tokenizer)?;
    record_tokenized_query(tokenized_query, &tokens);

    Ok(CombinedFieldsScan {
        statistics_rows,
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
        self.validate_statistics_scope()?;
        let query = self.query.clone();
        let statistics_fragments = self.statistics_fragments.clone();
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
                statistics_rows,
                columns,
                segments: all_segments,
                tokens,
                ..
            } = open_combined_fields_scan(
                &ds,
                &query,
                statistics_fragments.as_ref(),
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
                    let scorer = Arc::new({
                        let mut search = CombinedFieldsSearch::new(&columns);
                        if let Some(rows) = &statistics_rows {
                            search = search.with_statistics_rows(rows);
                        }
                        search
                            .build_scorer(
                                &tokens,
                                CombinedCorpusStats::IndexOnly,
                                Some(metrics.as_ref()),
                            )
                            .boxed()
                            .await?
                    });
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::WriteParams;
    use crate::index::DatasetIndexExt;
    use crate::io::exec::fts::tests::{candidate_input, execute_results, memory_input};
    use crate::io::exec::fts::{
        FlatCombinedFieldsExec, FlatStatsCoverage, FlatStatsScope, FtsCorpusBinding,
        SharedFtsScorerExec,
    };
    use arrow_array::RecordBatchIterator;
    use datafusion::physical_plan::{
        coalesce_partitions::CoalescePartitionsExec, union::UnionExec,
    };
    use lance_core::ROW_ID;
    use lance_core::utils::address::RowAddress;
    use lance_index::IndexType;
    use lance_index::scalar::inverted::query::Operator;

    #[rstest::rstest]
    #[case::same_analyzer("text", "text", false)]
    #[case::incompatible_analyzer("text", "text", true)]
    #[case::explicit_json_on_text("json", "json", false)]
    #[case::incompatible_document_tokenizer("json", "text", false)]
    #[tokio::test]
    async fn combined_fields_preserves_document_tokenizers(
        #[case] title_tokenizer: &str,
        #[case] body_tokenizer: &str,
        #[case] incompatible_analyzer: bool,
        #[values(None, Some(0), Some(1))] fragment: Option<u32>,
    ) {
        let batch = arrow_array::record_batch!(
            ("title", Utf8, [r#"{"title":"Alpha"}"#]),
            ("body", Utf8, [r#"{"title":"beta"}"#])
        )
        .unwrap();
        let schema = batch.schema();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            "memory://",
            Some(WriteParams {
                enable_stable_row_ids: false,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        for (column, tokenizer) in [("title", title_tokenizer), ("body", body_tokenizer)] {
            let params = InvertedIndexParams::default()
                .lance_tokenizer(tokenizer.to_string())
                .lower_case(!(incompatible_analyzer && column == "body"));
            dataset
                .create_index(&[column], IndexType::Inverted, None, &params, true)
                .await
                .unwrap();
        }
        let (terms, expected_tokens) = if title_tokenizer == "json" {
            (
                "title,str,ALPHA beta",
                ["title,str,alpha", "title,str,beta"],
            )
        } else {
            ("ALPHA beta", ["alpha", "beta"])
        };
        let query = CombinedFieldsQuery::try_new(terms.into(), vec!["title".into(), "body".into()])
            .unwrap();
        let scope = fragment.map(|fragment| roaring::RoaringBitmap::from_iter([fragment]));
        let metrics = FtsIndexMetrics::new(&ExecutionPlanMetricsSet::new(), 0);
        let scan = open_combined_fields_scan(
            &dataset,
            &query,
            scope.as_ref(),
            CombinedFieldsIndexRequirement::AtLeastOneColumn,
            &[],
            &OnceLock::new(),
            &metrics,
        )
        .await;
        if incompatible_analyzer || title_tokenizer != body_tokenizer {
            let error = scan.err().unwrap();
            assert!(matches!(&error, DataFusionError::Execution(_)));
            assert!(error.to_string().contains("share the same tokenizer"));
        } else {
            let scan = scan.unwrap();
            let indexed = fragment != Some(1);
            assert_eq!(scan.segments.len(), if indexed { 2 } else { 0 });
            assert!(
                scan.columns
                    .iter()
                    .all(|column| column.indices.len() == usize::from(indexed))
            );
            assert_eq!(scan.tokens.len(), 2);
            assert_eq!(scan.tokens.get_token(0), expected_tokens[0]);
            assert_eq!(scan.tokens.get_token(1), expected_tokens[1]);
        }
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn combined_fields_validates_omitted_field_tokenizer(
        #[values("json", "text")] body_tokenizer: &str,
        #[values(0, 1)] fragment: u32,
    ) {
        let batch = arrow_array::record_batch!(
            (
                "title",
                Utf8,
                [r#"{"title":"Alpha"}"#, r#"{"title":"Alpha"}"#]
            ),
            ("body", Utf8, [r#"{"title":"beta"}"#, r#"{"title":"beta"}"#])
        )
        .unwrap();
        let schema = batch.schema();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            "memory://",
            Some(WriteParams {
                enable_stable_row_ids: false,
                max_rows_per_file: 1,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        for (column, tokenizer, covered) in [("title", "json", 0), ("body", body_tokenizer, 1)] {
            let params = InvertedIndexParams::default().lance_tokenizer(tokenizer.to_string());
            let name = format!("{column}_idx");
            let segment = dataset
                .create_index_builder(&[column], IndexType::Inverted, &params)
                .name(name.clone())
                .fragments(vec![covered])
                .execute_uncommitted()
                .await
                .unwrap();
            dataset
                .commit_existing_index_segments(&name, column, vec![segment])
                .await
                .unwrap();
        }
        let query = CombinedFieldsQuery::try_new(
            "title,str,ALPHA beta".into(),
            vec!["title".into(), "body".into()],
        )
        .unwrap();
        let scope = roaring::RoaringBitmap::from_iter([fragment]);
        let metrics = FtsIndexMetrics::new(&ExecutionPlanMetricsSet::new(), 0);
        let scan = open_combined_fields_scan(
            &dataset,
            &query,
            Some(&scope),
            CombinedFieldsIndexRequirement::AtLeastOneColumn,
            &[],
            &OnceLock::new(),
            &metrics,
        )
        .await;
        if body_tokenizer == "json" {
            let scan = scan.unwrap();
            assert_eq!(scan.segments.len(), 1);
            assert_eq!(scan.columns[fragment as usize].indices.len(), 1);
            assert!(scan.columns[1 - fragment as usize].indices.is_empty());
            assert_eq!(scan.tokens.len(), 2);
            assert_eq!(scan.tokens.get_token(0), "title,str,alpha");
            assert_eq!(scan.tokens.get_token(1), "title,str,beta");
        } else {
            let error = scan.err().unwrap();
            assert!(matches!(&error, DataFusionError::Execution(_)));
            assert!(error.to_string().contains("share the same tokenizer"));
        }
    }

    // Raw text is the oracle: no index statistics, scorer, or plan helper is
    // used to calculate expected scores. The third fragment is outside the domain.
    const TEXT: [[&str; 2]; 6] = [
        ["alpha alpha", "beta"],
        ["beta", "alpha beta"],
        ["alpha", ""],
        ["", "beta beta"],
        ["alpha alpha alpha alpha", "alpha"],
        ["beta beta beta", "beta"],
    ];

    async fn domain_dataset(stable: bool, mixed: bool) -> Arc<Dataset> {
        let batch = arrow_array::record_batch!(
            (
                "title",
                Utf8,
                TEXT.iter().map(|row| row[0]).collect::<Vec<_>>()
            ),
            (
                "body",
                Utf8,
                TEXT.iter().map(|row| row[1]).collect::<Vec<_>>()
            ),
            ("id", Int32, [0, 1, 2, 3, 4, 5])
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(
                vec![Ok(batch.slice(0, if mixed { 2 } else { 6 }))],
                batch.schema(),
            ),
            "memory://",
            Some(WriteParams {
                enable_stable_row_ids: stable,
                max_rows_per_file: 2,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        for column in ["title", "body"] {
            dataset
                .create_index(
                    &[column],
                    IndexType::Inverted,
                    None,
                    &InvertedIndexParams::default()
                        .stem(false)
                        .remove_stop_words(false),
                    true,
                )
                .await
                .unwrap();
        }
        if mixed {
            dataset
                .append(
                    RecordBatchIterator::new(vec![Ok(batch.slice(2, 4))], batch.schema()),
                    Some(WriteParams {
                        max_rows_per_file: 2,
                        ..Default::default()
                    }),
                )
                .await
                .unwrap();
        }
        Arc::new(dataset)
    }

    fn domain_query(operator: Operator) -> CombinedFieldsQuery {
        CombinedFieldsQuery::try_new("alpha beta".into(), vec!["title".into(), "body".into()])
            .unwrap()
            .try_with_boosts(vec![3.0, 1.0])
            .unwrap()
            .with_operator(operator)
    }

    fn domain_oracle(rows: usize, candidates: &[usize], operator: Operator) -> Vec<(u64, f32)> {
        let count = (0..2)
            .map(|field| TEXT[..rows].iter().filter(|r| !r[field].is_empty()).count())
            .max()
            .unwrap() as f64;
        let weights = [3.0, 1.0];
        let lengths: Vec<f64> = TEXT[..rows]
            .iter()
            .map(|row| {
                (0..2)
                    .map(|f| weights[f] * row[f].split_whitespace().count() as f64)
                    .sum()
            })
            .collect();
        let average = lengths.iter().sum::<f64>() / count;
        candidates
            .iter()
            .filter_map(|&row| {
                let mut score = 0.0;
                for term in ["alpha", "beta"] {
                    let frequency = (0..2)
                        .map(|f| {
                            weights[f]
                                * TEXT[row][f]
                                    .split_whitespace()
                                    .filter(|t| *t == term)
                                    .count() as f64
                        })
                        .sum::<f64>();
                    if operator == Operator::And && frequency == 0.0 {
                        return None;
                    }
                    let df = (0..2)
                        .map(|f| {
                            TEXT[..rows]
                                .iter()
                                .filter(|r| r[f].split_whitespace().any(|t| t == term))
                                .count()
                        })
                        .max()
                        .unwrap() as f64;
                    let idf = (1.0 + (count - df + 0.5) / (df + 0.5)).ln();
                    score += idf * 2.2 * frequency
                        / (frequency + 1.2 * (0.25 + 0.75 * lengths[row] / average));
                }
                Some((
                    u64::from(RowAddress::new_from_parts(
                        (row / 2) as u32,
                        (row % 2) as u32,
                    )),
                    score as f32,
                ))
            })
            .collect()
    }

    #[rstest::rstest]
    #[case::indexed_or(false, false, Operator::Or)]
    #[case::indexed_and(false, false, Operator::And)]
    #[case::deleted_fragment(false, true, Operator::Or)]
    #[case::mixed_or(true, false, Operator::Or)]
    #[case::mixed_and(true, false, Operator::And)]
    #[tokio::test]
    async fn statistics_domain_matches_raw_text_with_candidate_filter(
        #[case] mixed: bool,
        #[case] delete_fragment: bool,
        #[case] operator: Operator,
    ) {
        let mut dataset = domain_dataset(false, mixed).await;
        if delete_fragment {
            Arc::make_mut(&mut dataset)
                .delete("id >= 2 AND id < 4")
                .await
                .unwrap();
        }
        let scope = roaring::RoaringBitmap::from_iter([0, 1]);
        let candidates = if delete_fragment {
            vec![0]
        } else {
            vec![0, 2, 3]
        };
        let candidate_rows = candidates
            .iter()
            .map(|row| {
                u64::from(RowAddress::new_from_parts(
                    (row / 2) as u32,
                    (row % 2) as u32,
                ))
            })
            .collect::<Vec<_>>();
        let indexed = CombinedFieldsQueryExec::new(
            dataset.clone(),
            domain_query(operator),
            FtsSearchParams::default(),
            candidate_input(candidate_rows.clone(), false),
        )
        .with_covered_fragments(if mixed || delete_fragment {
            roaring::RoaringBitmap::from_iter([0])
        } else {
            scope.clone()
        })
        .with_statistics_fragments(scope.clone())
        .unwrap();
        let plan: Arc<dyn ExecutionPlan> = if mixed {
            let tail = arrow_array::record_batch!(
                ("title", Utf8, [TEXT[2][0], TEXT[3][0]]),
                ("body", Utf8, [TEXT[2][1], TEXT[3][1]]),
                (
                    ROW_ID,
                    UInt64,
                    [
                        u64::from(RowAddress::new_from_parts(1, 0)),
                        u64::from(RowAddress::new_from_parts(1, 1))
                    ]
                )
            )
            .unwrap();
            let tail_fragments = roaring::RoaringBitmap::from_iter([1]);
            let flat = FlatCombinedFieldsExec::try_new(
                dataset,
                domain_query(operator),
                FtsSearchParams::default(),
                memory_input(tail),
                tail_fragments.clone(),
                FlatStatsScope::PerColumn(vec![
                    FlatStatsCoverage {
                        fragments: tail_fragments,
                        stale_rows: Arc::new(RowAddrTreeMap::new())
                    };
                    2
                ]),
                candidate_input(candidate_rows, false),
            )
            .unwrap()
            .with_statistics_fragments(scope)
            .unwrap();
            let union = UnionExec::try_new(vec![Arc::new(indexed), Arc::new(flat)]).unwrap();
            Arc::new(
                SharedFtsScorerExec::try_new(
                    Arc::new(CoalescePartitionsExec::new(union)),
                    FtsCorpusBinding {
                        producer_path: vec![0, 1],
                        consumer_path: vec![0, 0],
                    },
                )
                .unwrap(),
            )
        } else {
            Arc::new(indexed)
        };
        let expected = domain_oracle(4, &candidates, operator);
        for _ in 0..2 {
            let actual = execute_results(plan.as_ref()).await.unwrap();
            assert_eq!(actual.len(), expected.len());
            for ((row, score), (expected_row, expected_score)) in actual.iter().zip(&expected) {
                assert_eq!(row, expected_row);
                assert!(
                    (score - expected_score).abs() < 1e-5,
                    "{row}: {score} != {expected_score}"
                );
            }
        }
    }

    #[rstest::rstest]
    #[case::stable(true, false)]
    #[case::unscoped(false, false)]
    #[case::outside(false, true)]
    #[tokio::test]
    async fn statistics_domain_rejects_incompatible_candidate_scope(
        #[case] stable: bool,
        #[case] explicit_outside: bool,
    ) {
        let dataset = domain_dataset(stable, false).await;
        let mut exec = CombinedFieldsQueryExec::new(
            dataset,
            domain_query(Operator::Or),
            FtsSearchParams::default(),
            PreFilterSource::None,
        );
        if explicit_outside {
            exec = exec.with_covered_fragments(roaring::RoaringBitmap::from_iter([1]));
        }
        let error = exec
            .with_statistics_fragments(roaring::RoaringBitmap::from_iter([0]))
            .unwrap_err();
        assert!(matches!(error, DataFusionError::Plan(_)));
        assert!(error.to_string().contains(if stable {
            "physical row IDs"
        } else {
            "candidate fragments"
        }));
    }

    #[tokio::test]
    async fn statistics_domain_revalidates_late_candidate_change() {
        let dataset = domain_dataset(false, false).await;
        let scope = roaring::RoaringBitmap::from_iter([0]);
        let exec = CombinedFieldsQueryExec::new(
            dataset,
            domain_query(Operator::Or),
            FtsSearchParams::default(),
            PreFilterSource::None,
        )
        .with_covered_fragments(scope.clone())
        .with_statistics_fragments(scope)
        .unwrap()
        .with_covered_fragments(roaring::RoaringBitmap::from_iter([1]));
        let error = execute_results(&exec).await.unwrap_err();
        assert!(matches!(error, DataFusionError::Plan(_)));
        assert!(error.to_string().contains("candidate fragments"));
    }
}
