// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Flat cross-field BM25F execution node.

use std::sync::{Arc, OnceLock};

use arrow_schema::DataType;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion_physical_expr::{Distribution, EquivalenceProperties, Partitioning, PhysicalExpr};
use futures::stream::{self};
use futures::{StreamExt, TryStreamExt};
use lance_arrow::json::JsonEncoding;
use lance_core::{Error, ROW_ID, Result, utils::tracing::StreamTracingExt};
use lance_select::RowAddrMask;

use crate::Dataset;
use crate::index::prefilter::DatasetPreFilter;
use crate::index::scalar::inverted::{
    ResolvedFtsField, flatten_fts_document_column, resolve_fts_field,
};
use crate::io::exec::PreFilterSource;
use crate::io::exec::utils::{PreFilterMasks, build_prefilter_restricted_to_fragments};
use lance_index::prefilter::PreFilter;
use lance_index::scalar::inverted::builder::document_input;
use lance_index::scalar::inverted::query::{CombinedFieldsQuery, FtsSearchParams};
use lance_index::scalar::inverted::{
    CombinedFieldsBM25Scorer, DocumentGranularity, FTS_SCHEMA, flat_combined_fields_search_stream,
};
use lance_select::{RowAddrTreeMap, RowSetOps};
use tracing::instrument;

use super::combined_fields::{
    CombinedFieldsIndexRequirement, CombinedFieldsScan, fmt_combined_fields,
    open_combined_fields_scan,
};
use super::corpus::{SharedFtsScorer, SharedFtsScorerProducer};
use super::query_tokens::TokenizedQuery;
use super::selection::FtsIndexMetrics;

/// The rows of one target column that a flat `combined_fields` scan folds into
/// that column's corpus statistics: the ones its index does not already count.
///
/// The scan reads every target column of every row it scores, but a scanned
/// fragment can still be indexed for some of those columns. Folding such a row
/// into that column's statistics as well would count it twice, inflating
/// `docCount_f`, `sumTotalTermFreq_f` and `docFreq_f`.
#[derive(Debug, Clone, Default)]

pub struct FlatStatsCoverage {
    /// Fragments this column's index does not cover at all.
    pub fragments: roaring::RoaringBitmap,
    /// Rows it does cover, but whose entries a newer overlay made stale, in the id
    /// domain the scan's `_rowid` uses. Their current values are folded in, and
    /// their pre-overlay values are subtracted from the index statistics (see
    /// [`lance_index::scalar::inverted::CombinedFieldColumn::stale_rows`]), so
    /// each counts once.
    pub stale_rows: Arc<RowAddrTreeMap>,
}

/// Which corpus statistics a flat `combined_fields` scan contributes.
#[derive(Debug, Clone)]
pub enum FlatStatsScope {
    /// Per target column, in query order, the rows to fold on top of that
    /// column's index statistics.
    PerColumn(Vec<FlatStatsCoverage>),
    /// The scan reads every target fragment and its statistics replace the index
    /// statistics; see [`lance_index::scalar::inverted::CombinedCorpusStats::FlatOnly`].
    WholeCorpus,
}

/// The sibling of [`super::CombinedFieldsQueryExec`] for the fragments that at
/// least one target column's index is missing. Those rows cannot be scored from
/// the indexes (`tf'`/`dl'` would omit the unindexed columns), so they are scored
/// straight from their column values here. The two plans are unioned and
/// re-sorted by the planner, exactly as the single-column match path does.
///
/// The input must carry `_rowid` plus every target column of the query.
/// Field traversal is derived from the pinned dataset schema and query paths,
/// always at row granularity. A codec preserves the corpus input separately from
/// the emission prefilter and reconnects shared leaves through their owner.
///
/// ```
/// # use lance::{io::exec::fts::FlatCombinedFieldsExec, Error, Result};
/// # use datafusion::physical_plan::ExecutionPlan;
/// # fn rebuild(exec: &FlatCombinedFieldsExec) -> Result<FlatCombinedFieldsExec> {
/// if exec.has_shared_scorer() {
///     return Err(Error::invalid_input("reconstruct the enclosing corpus owner"));
/// }
/// FlatCombinedFieldsExec::try_new(
///     exec.dataset().clone(), exec.query().clone(), exec.params().clone(),
///     exec.children()[0].clone(), exec.scanned_fragments().clone(),
///     exec.stats_scope().clone(), exec.emit_prefilter().clone(),
/// )
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct FlatCombinedFieldsExec {
    dataset: Arc<Dataset>,
    query: CombinedFieldsQuery,
    /// Tokens `execute()` actually produced, for `analyze_plan`. Single snapshot
    /// for the same reason as [`super::CombinedFieldsQueryExec::tokenized_query`].
    tokenized_query: Arc<OnceLock<TokenizedQuery>>,
    params: FtsSearchParams,
    unindexed_input: Arc<dyn ExecutionPlan>,
    /// The fragments `unindexed_input` reads.
    scanned_fragments: roaring::RoaringBitmap,
    /// Per target column, in query order: how `execute` turns the scanned column
    /// into document text.
    resolved_fields: Vec<ResolvedFtsField>,
    stats_scope: FlatStatsScope,
    /// Picks the emitted rows. The planner reads `unindexed_input` unfiltered, so
    /// that the filter does not change the corpus statistics.
    emit_prefilter: PreFilterSource,
    /// Publishes the blended scorer to an indexed sibling; see
    /// [`Self::with_shared_scorer`].
    shared_scorer: Option<Arc<SharedFtsScorer<CombinedFieldsBM25Scorer>>>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl DisplayAs for FlatCombinedFieldsExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt_combined_fields(
            "FlatCombinedFields",
            &self.query,
            &self.tokenized_query,
            t,
            f,
        )
    }
}

impl FlatCombinedFieldsExec {
    /// Reconstruct a flat BM25F scan with its native corpus and candidate scopes.
    ///
    /// `scanned_fragments` includes fragments read whole and those supplying
    /// overlay-stale rows. A [`FlatStatsScope::PerColumn`] entry describes only
    /// rows missing from that column's index statistics, in query column order;
    /// its stale rows retain the dataset's row-id domain. Neither this scope nor
    /// the corpus input may be narrowed to the emission prefilter.
    ///
    /// Paths crossing lists read the complete list root; other paths require a
    /// string or JSON column aliased to the query path. The input also needs UInt64
    /// `_rowid`. See [`Self`] for a reconstruction example.
    pub fn try_new(
        dataset: Arc<Dataset>,
        query: CombinedFieldsQuery,
        params: FtsSearchParams,
        unindexed_input: Arc<dyn ExecutionPlan>,
        scanned_fragments: roaring::RoaringBitmap,
        stats_scope: FlatStatsScope,
        emit_prefilter: PreFilterSource,
    ) -> Result<Self> {
        let num_columns = query.column_names().len();
        let num_coverage = match &stats_scope {
            FlatStatsScope::PerColumn(coverage) => coverage.len(),
            FlatStatsScope::WholeCorpus => num_columns,
        };
        if num_coverage != num_columns {
            return Err(Error::invalid_input(format!(
                "combined_fields flat scan got {} statistics coverage sets for {} target columns",
                num_coverage, num_columns
            )));
        }
        let resolved_fields = query
            .column_names()
            .map(|column| resolve_fts_field(dataset.schema(), column, DocumentGranularity::Row))
            .collect::<Result<Vec<_>>>()?;
        let input_schema = unindexed_input.schema();
        let row_id = input_schema
            .field_with_name(ROW_ID)
            .map_err(|_| Error::invalid_input("combined_fields flat input is missing _rowid"))?;
        if row_id.data_type() != &DataType::UInt64 {
            return Err(Error::invalid_input(format!(
                "combined_fields flat input _rowid must be UInt64, got {}",
                row_id.data_type()
            )));
        }
        for (column, resolved) in query.column_names().zip(&resolved_fields) {
            let input_name = if resolved.has_lists() {
                resolved.root_column.as_str()
            } else {
                column
            };
            let field = input_schema.field_with_name(input_name).map_err(|_| {
                Error::invalid_input(format!(
                    "combined_fields flat input is missing column '{input_name}' for '{column}'"
                ))
            })?;
            let is_valid = if resolved.has_lists() {
                // This root is already resolved; do not parse literal quotes again.
                dataset
                    .schema()
                    .fields
                    .iter()
                    .find(|source| source.name == input_name)
                    .is_some_and(|source| field.data_type() == &source.data_type())
            } else {
                matches!(
                    field.data_type(),
                    DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
                ) || JsonEncoding::of_field(field).is_some()
            };
            if !is_valid {
                return Err(Error::invalid_input(format!(
                    "combined_fields flat input column '{input_name}' has incompatible type {} for '{column}'",
                    field.data_type()
                )));
            }
        }
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(FTS_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Ok(Self {
            dataset,
            query,
            tokenized_query: Arc::new(OnceLock::new()),
            params,
            unindexed_input,
            scanned_fragments,
            resolved_fields,
            stats_scope,
            emit_prefilter,
            shared_scorer: None,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
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

    pub(super) fn uses_shared_scorer(
        &self,
        scorer: &Arc<SharedFtsScorer<CombinedFieldsBM25Scorer>>,
    ) -> bool {
        self.shared_scorer
            .as_ref()
            .is_some_and(|shared| Arc::ptr_eq(shared, scorer))
    }

    /// Snapshot whose schema resolves query paths and whose indices supply
    /// tokenizer and corpus statistics. See [`Self`] for reconstruction.
    pub fn dataset(&self) -> &Arc<Dataset> {
        &self.dataset
    }

    /// Query paths and field weights in the order used by the statistics scope.
    pub fn query(&self) -> &CombinedFieldsQuery {
        &self.query
    }

    /// Native search parameters. Sorting and top-k remain in the parent plan.
    pub fn params(&self) -> &FtsSearchParams {
        &self.params
    }

    /// Every fragment read by the corpus input, including overlay row reads.
    pub fn scanned_fragments(&self) -> &roaring::RoaringBitmap {
        &self.scanned_fragments
    }

    /// Contributions to each column's statistics, independent of candidates.
    pub fn stats_scope(&self) -> &FlatStatsScope {
        &self.stats_scope
    }

    /// Candidate filter applied after measuring corpus statistics.
    pub fn emit_prefilter(&self) -> &PreFilterSource {
        &self.emit_prefilter
    }

    /// Whether reconstruction must reconnect this producer to its corpus owner.
    /// The process-local coordination object is deliberately not exported.
    pub fn has_shared_scorer(&self) -> bool {
        self.shared_scorer.is_some()
    }
}

impl ExecutionPlan for FlatCombinedFieldsExec {
    fn name(&self) -> &str {
        "FlatCombinedFieldsExec"
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        let mut children = vec![&self.unindexed_input];
        children.extend(self.emit_prefilter.execution_plan());
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
        if children.len() != self.children().len() {
            return Err(DataFusionError::Internal(
                "Unexpected number of children".to_string(),
            ));
        }
        // Reverse order: the optional prefilter child is last, so pop it first.
        let emit_prefilter = if self.emit_prefilter.execution_plan().is_some() {
            self.emit_prefilter
                .with_execution_plan(children.pop().expect("child count checked above"))?
        } else {
            // A prefilter source with no child of its own leaves the input last.
            self.emit_prefilter.clone()
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
            scanned_fragments: self.scanned_fragments.clone(),
            resolved_fields: self.resolved_fields.clone(),
            stats_scope: self.stats_scope.clone(),
            emit_prefilter,
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
        let stats_scope = self.stats_scope.clone();
        let stale_rows_per_column: Vec<Arc<RowAddrTreeMap>> = match &stats_scope {
            FlatStatsScope::PerColumn(coverage) => coverage
                .iter()
                .map(|coverage| coverage.stale_rows.clone())
                .collect(),
            FlatStatsScope::WholeCorpus => Vec::new(),
        };
        let emit_prefilter = self.emit_prefilter.clone();
        let scanned_fragments = self.scanned_fragments.clone();
        let prefilter_context = context.clone();
        let stats_mask_ds = self.dataset.clone();
        let metrics_set = self.metrics.clone();
        let metrics = Arc::new(FtsIndexMetrics::new(&self.metrics, partition));
        let metrics_clone = metrics.clone();
        let target_batch_size = context.session_config().batch_size();
        let elapsed_compute = metrics.baseline_metrics.elapsed_compute().clone();

        let shared_scorer_producer = self.shared_scorer.clone().map(SharedFtsScorerProducer::new);

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

        let stream = stream::once(async move {
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
                    &stale_rows_per_column,
                    &tokenized_query,
                    metrics.as_ref(),
                )
                .await?;

                let input_schema = input.schema();
                let doc_col_indices = query
                    .column_names()
                    .map(|column| input_schema.index_of(column))
                    .collect::<std::result::Result<Vec<_>, _>>()?;

                // One mask per target column over the rows it may fold. Built through
                // `create_restricted_deletion_mask` so the fragment restriction lands in
                // the id space the scan's `_rowid` uses; a hand-rolled fragment-address
                // allow list would match nothing under stable row ids.
                let fragment_mask = |fragments: roaring::RoaringBitmap| {
                    let ds = stats_mask_ds.clone();
                    async move {
                        match DatasetPreFilter::create_restricted_deletion_mask(ds, fragments) {
                            Some(mask) => mask.await,
                            None => Ok(Arc::new(RowAddrMask::all_rows())),
                        }
                    }
                };
                let stats_masks = match &stats_scope {
                    FlatStatsScope::PerColumn(coverage) => {
                        let mut stats_masks = Vec::with_capacity(coverage.len());
                        for coverage in coverage {
                            let mask = fragment_mask(coverage.fragments.clone()).await?;
                            // Stale rows sit in fragments the index does cover, so they
                            // are allowed by address rather than by fragment.
                            stats_masks.push(if coverage.stale_rows.is_empty() {
                                mask
                            } else {
                                Arc::new(
                                    mask.as_ref()
                                        .clone()
                                        .also_allow(coverage.stale_rows.as_ref().clone()),
                                )
                            });
                        }
                        stats_masks
                    }
                    FlatStatsScope::WholeCorpus => {
                        vec![fragment_mask(scanned_fragments.clone()).await?; columns.len()]
                    }
                };

                let emit_mask = {
                    let pre_filter = build_prefilter_restricted_to_fragments(
                        prefilter_context,
                        partition,
                        &emit_prefilter,
                        stats_mask_ds.clone(),
                        scanned_fragments,
                        // The external row-address mask is applied by the
                        // planner's `RowAddrMaskFilterExec` wrap around this
                        // node, so it stays out of the corpus statistics.
                        PreFilterMasks::default(),
                        &metrics_set,
                    )?;
                    pre_filter.wait_for_ready().await?;
                    pre_filter.mask()
                };

                flat_combined_fields_search_stream(
                    input,
                    &columns,
                    doc_col_indices,
                    &stats_masks,
                    Some(emit_mask),
                    matches!(stats_scope, FlatStatsScope::WholeCorpus),
                    &tokens,
                    tokenizer,
                    query.operator(),
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
