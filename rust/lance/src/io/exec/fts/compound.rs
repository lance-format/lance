// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compound-query execution nodes and their append-only residual handling.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

use arrow::array::AsArray;
use arrow::datatypes::{Float32Type, UInt64Type};
use arrow_array::{Array, Float32Array, RecordBatch, UInt64Array};
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion_physical_expr::{Distribution, EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion_physical_plan::ExecutionPlanProperties;
use futures::future::try_join_all;
use futures::stream::{self, FuturesUnordered};
use futures::{StreamExt, TryStreamExt};
use lance_core::{
    Error, ROW_ID, Result,
    utils::{
        tokio::{get_num_compute_intensive_cpus, spawn_cpu},
        tracing::StreamTracingExt,
    },
};
use lance_select::RowAddrMask;
use lance_table::format::IndexMetadata;
use rustc_hash::FxHashSet;

use crate::Dataset;
use crate::dataset::mem_wal::index::{QueryLocalFtsIndex, QueryLocalFtsStats};
use crate::index::scalar::inverted::load_segment_details;
use crate::io::exec::PreFilterSource;
use crate::io::exec::utils::{PreFilterMasks, build_prefilter};
use lance_index::prefilter::PreFilter;
use lance_index::scalar::inverted::builder::ScoredDoc;
use lance_index::scalar::inverted::query::{
    FtsQuery, FtsQueryNode, FtsSearchParams, Tokens, try_collect_query_tokens,
};
use lance_index::scalar::inverted::{
    DocumentGranularity, FTS_SCHEMA, InvertedIndex, MemBM25Scorer, PreparedBm25Query, SCORE_COL,
    Scorer, build_global_bm25_scorer, compound_search, compound_search_prepared_match,
    compound_search_prepared_match_with_score_floor, compound_search_with_base_scorer,
    cross_column_compound_search, exclusive_scaled_score_floor, materialized_compound_top_k,
};
use tracing::instrument;
use uuid::Uuid;

use super::match_query::MatchQueryExec;
use super::matches::{
    PreparedMatch, compound_leaf_columns, compound_query_uses_fuzzy_expansion, count_fts_leaves,
    scored_documents_batch,
};
use super::query_tokens::{
    TokenizedCompoundQuery, fmt_tokenized_compound_query, tokenize_compound_query,
    tokenize_cross_column_compound_query, tokenizer_for_match_query,
};
use super::segments::{
    open_fts_segments, search_prepared_segments, segments_the_prefilter_may_match,
};
use super::selection::{FtsIndexMetrics, FtsSegmentSelection};

/// Maximum number of additional kth-score rows retained before exact replay.
/// One extra probe slot is reserved for the strict lower-score guard.
const WAND_TIE_COMPLETION_BUDGET: usize = 128;

/// One DataFusion boundary around a posting-backed compound scorer tree.
///
/// Reconstruction preserves the explicit selection kind, candidate scope and
/// either the base scorer or canonical prepared Match override. Codecs supply
/// decoded child plans and value objects; the example reuses existing children
/// and uses the portable prepared-state constructor from `lance-index`.
///
/// ```
/// # use std::sync::Arc;
/// # use lance::{io::exec::fts::CompoundQueryExec, Error, Result};
/// # use lance_index::scalar::inverted::PreparedBm25Query;
/// # fn rebuild(exec: &CompoundQueryExec) -> Result<CompoundQueryExec> {
/// let mut rebuilt = if let Some(segments) = exec.preset_segments() {
///     CompoundQueryExec::new_with_segments(
///         exec.dataset().clone(), exec.query().clone(), exec.params().clone(),
///         exec.prefilter_source().clone(), segments.to_vec(),
///     )
/// } else {
///     let uuids = exec.explicit_segment_uuids()
///         .ok_or_else(|| Error::invalid_input("compound FTS needs an explicit selection"))?;
///     CompoundQueryExec::new_with_segment_uuids(
///         exec.dataset().clone(), exec.query().clone(), exec.params().clone(),
///         exec.prefilter_source().clone(), uuids,
///     )
/// }.with_external_mask(exec.external_mask().cloned());
/// if let Some(scorer) = exec.base_scorer() {
///     rebuilt = rebuilt.with_base_scorer(scorer.clone());
/// }
/// if let Some(prepared) = exec.prepared_match() {
///     rebuilt = rebuilt.with_prepared_match(Arc::new(PreparedBm25Query::from_parts(
///         prepared.tokens().clone(), prepared.scorer().clone(),
///         prepared.has_all_query_positions(),
///     )));
/// }
/// # Ok(rebuilt)
/// # }
/// ```
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
    /// Canonical vocabulary/scorer pair for a root Match query prepared over
    /// the complete corpus before this exec was restricted to a segment
    /// subset.
    prepared_match: Option<Arc<PreparedBm25Query>>,
    segment_selection: FtsSegmentSelection,
    /// Caller-supplied row-address mask, intersected into the prefilter so the
    /// compound scorer ranks only surviving rows (see
    /// [`MatchQueryExec::with_external_mask`]).
    external_mask: Option<Arc<RowAddrMask>>,
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
            prepared_match: None,
            segment_selection,
            external_mask: None,
            properties: Arc::new(PlanProperties::new(
                EquivalenceProperties::new(FTS_SCHEMA.clone()),
                Partitioning::RoundRobinBatch(1),
                EmissionType::Final,
                Boundedness::Bounded,
            )),
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// See [`MatchQueryExec::with_external_mask`].
    pub fn with_external_mask(mut self, mask: Option<Arc<RowAddrMask>>) -> Self {
        self.external_mask = mask;
        self
    }

    /// Override locally computed BM25 statistics with a corpus-wide scorer.
    ///
    /// The scorer must cover every token in every query leaf, including fuzzy
    /// expansions. Execution returns an error when any required token is absent.
    pub fn with_base_scorer(mut self, scorer: Arc<MemBM25Scorer>) -> Self {
        self.base_scorer = Some(scorer);
        self.prepared_match = None;
        self
    }

    /// Override root-Match preparation with one canonical vocabulary/scorer
    /// pair built against the complete corpus.
    ///
    /// This is required for distributed fuzzy execution over a segment subset;
    /// a scorer alone cannot preserve the globally capped rewrite.
    pub fn with_prepared_match(mut self, prepared: Arc<PreparedBm25Query>) -> Self {
        self.prepared_match = Some(prepared);
        self.base_scorer = None;
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

    /// Canonical root-Match vocabulary and scorer, or `None` for native
    /// preparation. Restore its values with [`PreparedBm25Query::from_parts`]
    /// so dictionary term IDs and other local preparation caches are not copied.
    /// See [`Self`] for a reconstruction example.
    pub fn prepared_match(&self) -> Option<&Arc<PreparedBm25Query>> {
        self.prepared_match.as_ref()
    }

    /// Additional candidate mask in the indices' row-id domain, or `None` for
    /// no external restriction. It does not narrow the statistics corpus.
    /// See [`Self`] for a reconstruction example.
    pub fn external_mask(&self) -> Option<&Arc<RowAddrMask>> {
        self.external_mask.as_ref()
    }

    /// Exact resolved metadata in its original order. `None` denotes deferred
    /// UUID selection; use [`Self::explicit_segment_uuids`] in that case.
    /// An empty resolved selection remains explicit and must not widen to all
    /// committed segments. See [`Self`] for a reconstruction example.
    pub fn preset_segments(&self) -> Option<&[IndexMetadata]> {
        self.segment_selection.preset_segments()
    }

    /// Re-cut this scorer at `limit`, keeping its segment selection, prepared
    /// scorers and masks.
    ///
    /// The FTS top-k lives in `FtsSearchParams`, not in an enclosing fetch
    /// node, so a caller that needs more candidates than the user asked for —
    /// over-fetching to survive a later dedup, say — has no other way to raise
    /// it. Everything that decides *which* rows are eligible is carried over
    /// untouched, so this widens the cut without widening the domain.
    pub fn with_limit(&self, limit: usize) -> Self {
        let mut params = self.params.clone();
        params.limit = Some(limit);
        Self {
            dataset: self.dataset.clone(),
            query: self.query.clone(),
            tokenized_query: self.tokenized_query.clone(),
            params,
            prefilter_source: self.prefilter_source.clone(),
            base_scorer: self.base_scorer.clone(),
            prepared_match: self.prepared_match.clone(),
            segment_selection: self.segment_selection.clone(),
            external_mask: self.external_mask.clone(),
            properties: self.properties.clone(),
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// See [`MatchQueryExec::explicit_segment_uuids`].
    pub fn explicit_segment_uuids(&self) -> Option<Vec<Uuid>> {
        self.segment_selection.explicit_segment_uuids()
    }
}

#[derive(Debug)]
struct QueryLocalResidualShard {
    index: QueryLocalFtsIndex,
    stats: QueryLocalFtsStats,
}

async fn index_query_local_residual_batch(
    mut residual: QueryLocalResidualShard,
    batch: RecordBatch,
    allowed_terms: Arc<FxHashSet<String>>,
) -> Result<QueryLocalResidualShard> {
    spawn_cpu(move || {
        let row_ids = batch
            .column_by_name(ROW_ID)
            .ok_or_else(|| {
                Error::invalid_input(
                    "hybrid compound FTS residual input is missing _rowid".to_string(),
                )
            })?
            .as_primitive::<UInt64Type>();
        let stats = residual.index.insert_with_row_ids_for_terms(
            &batch,
            row_ids,
            allowed_terms.as_ref(),
        )?;
        residual.stats.checked_add_assign(stats)?;
        Ok(residual)
    })
    .await
}

/// Build a bounded set of independent residual posting shards.
///
/// A single [`QueryLocalFtsIndex`] intentionally has one writer. Reusing one
/// index per CPU worker preserves that contract while allowing different scan
/// batches to tokenize in parallel. Completed workers immediately take the
/// next batch, so the entire stream is never collected in memory and the
/// number of live tokenizers/posting maps is bounded by the CPU pool size.
async fn index_query_local_residual(
    residual_input: SendableRecordBatchStream,
    seed: QueryLocalFtsIndex,
    allowed_terms: Arc<FxHashSet<String>>,
) -> DataFusionResult<Vec<QueryLocalResidualShard>> {
    // Match flat FTS's CPU-task sizing. Dataset scan batches are normally
    // row-bounded (often 8,192 rows), which can leave a small residual with
    // only one or two tokenizer tasks. Byte rechunking keeps tasks substantial
    // while exposing enough parallelism for variable-width text.
    const ACCUMULATE_BYTES: usize = 256 * 1024;
    const SLICE_BYTES: usize = 512 * 1024;
    let input_schema = residual_input.schema();
    let mut residual_input = Box::pin(lance_arrow::stream::rechunk_stream_by_size(
        residual_input,
        input_schema,
        ACCUMULATE_BYTES,
        SLICE_BYTES,
    ));
    let parallelism = get_num_compute_intensive_cpus().max(1);
    let mut initial_batches = Vec::with_capacity(parallelism);
    let mut is_input_exhausted = false;

    while initial_batches.len() < parallelism {
        let Some(batch) = residual_input.try_next().await? else {
            is_input_exhausted = true;
            break;
        };
        initial_batches.push(batch);
    }

    if initial_batches.is_empty() {
        return Ok(vec![QueryLocalResidualShard {
            index: seed,
            stats: QueryLocalFtsStats::default(),
        }]);
    }

    // Construct every shard from the already-loaded seed before dispatching
    // CPU work. This keeps tokenizer model I/O out of `spawn_cpu` closures.
    let mut initial_shards = Vec::with_capacity(initial_batches.len());
    for _ in 1..initial_batches.len() {
        initial_shards.push(QueryLocalResidualShard {
            index: seed.empty_sibling(),
            stats: QueryLocalFtsStats::default(),
        });
    }
    initial_shards.push(QueryLocalResidualShard {
        index: seed,
        stats: QueryLocalFtsStats::default(),
    });

    let mut in_flight = FuturesUnordered::new();
    for (shard, batch) in initial_shards.into_iter().zip(initial_batches) {
        in_flight.push(index_query_local_residual_batch(
            shard,
            batch,
            allowed_terms.clone(),
        ));
    }

    let mut shards = Vec::with_capacity(parallelism.min(in_flight.len()));
    while let Some(shard) = in_flight.try_next().await? {
        if is_input_exhausted {
            shards.push(shard);
            continue;
        }
        match residual_input.try_next().await? {
            Some(batch) => in_flight.push(index_query_local_residual_batch(
                shard,
                batch,
                allowed_terms.clone(),
            )),
            None => {
                is_input_exhausted = true;
                shards.push(shard);
            }
        }
    }
    Ok(shards)
}

async fn query_local_residual_leaves(
    shards: Vec<QueryLocalResidualShard>,
    query: FtsQuery,
    scorer: Arc<MemBM25Scorer>,
) -> Result<Vec<Vec<(u64, f32)>>> {
    let shard_leaves = stream::iter(shards.into_iter().map(|shard| {
        let query = query.clone();
        let scorer = scorer.clone();
        spawn_cpu(move || shard.index.exact_leaf_results(&query, scorer.as_ref()))
    }))
    .buffered(get_num_compute_intensive_cpus().max(1))
    .try_collect::<Vec<_>>()
    .await?;

    let leaf_count = shard_leaves.first().map_or(0, Vec::len);
    let mut merged = vec![Vec::new(); leaf_count];
    for leaves in shard_leaves {
        if leaves.len() != leaf_count {
            return Err(Error::internal(format!(
                "hybrid compound FTS residual shards produced inconsistent leaf counts: expected {leaf_count}, got {}",
                leaves.len()
            )));
        }
        for (merged, rows) in merged.iter_mut().zip(leaves) {
            merged.extend(rows);
        }
    }
    Ok(merged)
}

fn residual_bm25_scorer(
    committed_scorer: &MemBM25Scorer,
    shards: &[QueryLocalResidualShard],
) -> Result<MemBM25Scorer> {
    let mut scorer = committed_scorer.clone();
    for shard in shards {
        shard.stats.add_to_scorer(&mut scorer)?;
    }
    Ok(scorer)
}

/// Compound FTS over committed postings plus a small append-only residual scan.
///
/// The residual documents are tokenized once into query-local postings, rather
/// than once for every compound leaf. The indexed arm uses committed-index
/// BM25 statistics. The residual arm extends those statistics with the
/// query-local materialized documents, which matches the established mixed
/// flat-search approximation without rescanning the residual input or rebuilding
/// exact corpus statistics.
#[derive(Debug)]
pub struct HybridCompoundQueryExec {
    dataset: Arc<Dataset>,
    query: FtsQuery,
    params: FtsSearchParams,
    column: String,
    segments: Arc<[IndexMetadata]>,
    residual_input: Arc<dyn ExecutionPlan>,
    indexed_input: Option<(Arc<dyn ExecutionPlan>, Arc<MemBM25Scorer>)>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl HybridCompoundQueryExec {
    pub fn new(
        dataset: Arc<Dataset>,
        query: FtsQuery,
        params: FtsSearchParams,
        column: String,
        segments: Vec<IndexMetadata>,
        residual_input: Arc<dyn ExecutionPlan>,
    ) -> Self {
        Self {
            dataset,
            query,
            params,
            column,
            segments: Arc::from(segments),
            residual_input,
            indexed_input: None,
            properties: Arc::new(PlanProperties::new(
                EquivalenceProperties::new(FTS_SCHEMA.clone()),
                Partitioning::RoundRobinBatch(1),
                EmissionType::Final,
                Boundedness::Bounded,
            )),
            metrics: ExecutionPlanMetricsSet::new(),
        }
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

    pub fn column(&self) -> &str {
        &self.column
    }

    /// The indexed segments this scorer reads. Paired with
    /// [`Self::residual_input`], [`Self::indexed_input`] and [`Self::new`], this
    /// lets a caller rebuild the node. The FTS top-k lives in [`Self::params`], not in a
    /// fetch node, so changing it means reconstruction.
    pub fn segments(&self) -> &[IndexMetadata] {
        &self.segments
    }

    /// Replace the indexed search with an externally computed bounded top-k.
    ///
    /// `input` must search exactly [`Self::segments`] with the complete query
    /// and the supplied corpus-wide committed-index scorer. It must produce
    /// at most [`FtsSearchParams::limit`] rows, ordered by score descending and
    /// row ID ascending. The residual arm extends the same scorer with its
    /// query-local statistics, preserving the mixed-search scoring contract.
    ///
    /// A distributed planner can merge bounded per-worker [`CompoundQueryExec`]
    /// results before passing that single-partition input to this method.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use datafusion::physical_plan::ExecutionPlan;
    /// # use lance::io::exec::fts::HybridCompoundQueryExec;
    /// # use lance_index::scalar::inverted::MemBM25Scorer;
    /// # fn replace_indexed_search(
    /// #     hybrid: &HybridCompoundQueryExec,
    /// #     merged: Arc<dyn ExecutionPlan>,
    /// #     scorer: Arc<MemBM25Scorer>,
    /// # ) -> datafusion::error::Result<HybridCompoundQueryExec> {
    /// hybrid.with_indexed_input(merged, scorer)
    /// # }
    /// ```
    pub fn with_indexed_input(
        &self,
        input: Arc<dyn ExecutionPlan>,
        committed_scorer: Arc<MemBM25Scorer>,
    ) -> DataFusionResult<Self> {
        if input.schema() != self.schema()
            || input.output_partitioning().partition_count() != 1
            || self.params.limit.is_none()
        {
            return Err(DataFusionError::Plan(
                "hybrid indexed input requires the FTS schema, one partition, and a bounded limit"
                    .to_string(),
            ));
        }
        let mut result = Self::new(
            self.dataset.clone(),
            self.query.clone(),
            self.params.clone(),
            self.column.clone(),
            self.segments.to_vec(),
            self.residual_input.clone(),
        );
        result.indexed_input = Some((input, committed_scorer));
        Ok(result)
    }

    /// The externally computed indexed candidates and their committed-index
    /// scorer, or `None` when this node searches its indexed segments locally.
    ///
    /// Preserve the pair together through [`Self::with_indexed_input`]. The
    /// residual input remains separate, and its query-local index is rebuilt
    /// during execution. A codec supplies decoded children; this example reuses
    /// the existing children and reconstructs the scorer from its values.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use lance::io::exec::fts::HybridCompoundQueryExec;
    /// # use lance_index::scalar::inverted::MemBM25Scorer;
    /// # fn rebuild(exec: &HybridCompoundQueryExec)
    /// #     -> datafusion::error::Result<HybridCompoundQueryExec> {
    /// let rebuilt = HybridCompoundQueryExec::new(
    ///     exec.dataset().clone(), exec.query().clone(), exec.params().clone(),
    ///     exec.column().to_owned(), exec.segments().to_vec(),
    ///     exec.residual_input().clone(),
    /// );
    /// match exec.indexed_input() {
    ///     Some((input, scorer)) => rebuilt.with_indexed_input(
    ///         input.clone(), Arc::new(MemBM25Scorer::new(
    ///             scorer.total_tokens, scorer.num_docs, scorer.token_docs.clone(),
    ///         )),
    ///     ),
    ///     None => Ok(rebuilt),
    /// }
    /// # }
    /// ```
    pub fn indexed_input(&self) -> Option<&(Arc<dyn ExecutionPlan>, Arc<MemBM25Scorer>)> {
        self.indexed_input.as_ref()
    }

    /// The scan over the fragments no segment covers.
    pub fn residual_input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.residual_input
    }
}

impl DisplayAs for HybridCompoundQueryExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "HybridCompoundFtsScorer: column={}, query={}",
            self.column, self.query
        )
    }
}

impl ExecutionPlan for HybridCompoundQueryExec {
    fn name(&self) -> &str {
        "HybridCompoundQueryExec"
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        let mut children = vec![&self.residual_input];
        if let Some((input, _)) = &self.indexed_input {
            children.push(input);
        }
        children
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
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
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != self.children().len() {
            return Err(DataFusionError::Internal(format!(
                "hybrid compound FTS expected {} children, got {}",
                self.children().len(),
                children.len()
            )));
        }
        let indexed_input = if self.indexed_input.is_some() {
            children.pop()
        } else {
            None
        };
        let residual_input = children.pop().ok_or_else(|| {
            DataFusionError::Internal("hybrid compound FTS lost its residual child".to_string())
        })?;
        let result = Self::new(
            self.dataset.clone(),
            self.query.clone(),
            self.params.clone(),
            self.column.clone(),
            self.segments.to_vec(),
            residual_input,
        );
        match (indexed_input, &self.indexed_input) {
            (Some(input), Some((_, scorer))) => {
                Ok(Arc::new(result.with_indexed_input(input, scorer.clone())?))
            }
            _ => Ok(Arc::new(result)),
        }
    }

    #[instrument(name = "hybrid_compound_fts_exec", level = "debug", skip_all)]
    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let dataset = self.dataset.clone();
        let query = self.query.clone();
        let params = self.params.clone();
        let column = self.column.clone();
        let segments = self.segments.clone();
        let residual_input = self.residual_input.clone();
        let indexed_input = self.indexed_input.clone();
        let metrics_set = self.metrics.clone();
        let metrics = Arc::new(FtsIndexMetrics::new(&self.metrics, partition));
        let schema = self.schema();

        let stream = stream::once(async move {
            let _timer = metrics.baseline_metrics.elapsed_compute().timer();
            // An external indexed arm supplies the committed statistics and
            // candidates. Only the representative tokenizer is needed here.
            let segments_to_open = if indexed_input.is_some() {
                segments.get(..1).ok_or_else(|| {
                    DataFusionError::Execution(format!(
                        "hybrid FTS for column {column} requires a committed segment"
                    ))
                })?
            } else {
                &segments
            };
            let indices =
                open_fts_segments(&dataset, &column, segments_to_open, &metrics.index_metrics)
                    .await?;
            let first_index = indices.first().ok_or_else(|| {
                DataFusionError::Execution(format!(
                    "FTS index for column {column} has no committed segments"
                ))
            })?;
            let field_id = dataset.schema().field_id(&column)?;
            let tokenizer = first_index.tokenizer();
            let doc_type = tokenizer.doc_type();
            let residual_seed = QueryLocalFtsIndex::try_with_loaded_tokenizer(
                field_id,
                column.clone(),
                first_index.params().clone(),
                tokenizer,
            )?;
            let terms = residual_seed.exact_query_terms(&query)?;
            if terms.is_empty() {
                metrics.baseline_metrics.record_output(0);
                return scored_documents_batch(schema, Vec::new()).map_err(DataFusionError::from);
            }
            let allowed_terms = Arc::new(terms.iter().cloned().collect::<FxHashSet<_>>());
            let query_tokens = Tokens::new(terms.clone(), doc_type);
            let exact_params = params
                .clone()
                .with_fuzziness(Some(0))
                .with_phrase_slop(None);

            let residual_context = context.clone();
            let residual_indexing = async move {
                let residual_input = residual_input.execute(partition, residual_context)?;
                index_query_local_residual(residual_input, residual_seed, allowed_terms).await
            };
            let scorer_build = async {
                if let Some((_, scorer)) = &indexed_input {
                    return Ok(scorer.clone());
                }
                let scorer = build_global_bm25_scorer(
                    &indices,
                    &query_tokens,
                    &exact_params,
                    Some(metrics.as_ref()),
                )
                .await?;
                DataFusionResult::<Arc<MemBM25Scorer>>::Ok(Arc::new(scorer))
            };
            let (residual_shards, committed_scorer) =
                futures::future::try_join(residual_indexing, scorer_build).await?;
            let residual_scorer = Arc::new(residual_bm25_scorer(
                committed_scorer.as_ref(),
                &residual_shards,
            )?);
            let limit = params.limit.ok_or_else(|| {
                DataFusionError::Execution(
                    "hybrid compound FTS requires a bounded result limit".to_string(),
                )
            })?;

            let indexed_search = async {
                if let Some((input, _)) = &indexed_input {
                    let stream = input.execute(0, context.clone())?;
                    return collect_bounded_fts_input(stream, limit).await;
                }
                let prefilter = build_prefilter(
                    context,
                    partition,
                    &PreFilterSource::None,
                    dataset,
                    &segments,
                    PreFilterMasks {
                        overlay_block: None,
                        external_mask: None,
                    },
                    &metrics_set,
                )?;
                compound_search_with_base_scorer(
                    &indices,
                    &query,
                    &params,
                    prefilter,
                    metrics.clone(),
                    committed_scorer,
                )
                .await
                .map_err(DataFusionError::from)
            };
            let residual_query = query.clone();
            let residual_search = async move {
                let residual_leaves = query_local_residual_leaves(
                    residual_shards,
                    residual_query.clone(),
                    residual_scorer,
                )
                .await?;
                spawn_cpu(move || {
                    materialized_compound_top_k(&residual_query, residual_leaves, limit)
                })
                .await
                .map_err(DataFusionError::from)
            };
            let ((indexed_row_ids, indexed_scores), (residual_row_ids, residual_scores)) =
                futures::future::try_join(indexed_search, residual_search).await?;

            let mut documents = indexed_row_ids
                .into_iter()
                .zip(indexed_scores)
                .chain(residual_row_ids.into_iter().zip(residual_scores))
                .map(|(row_id, score)| ScoredDoc::new(row_id, score))
                .collect::<Vec<_>>();
            documents.sort_unstable_by(|left, right| {
                right
                    .score
                    .0
                    .total_cmp(&left.score.0)
                    .then_with(|| left.row_id.cmp(&right.row_id))
            });
            documents.truncate(limit);
            metrics.baseline_metrics.record_output(documents.len());
            scored_documents_batch(schema, documents).map_err(DataFusionError::from)
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

async fn collect_bounded_fts_input(
    mut input: SendableRecordBatchStream,
    limit: usize,
) -> DataFusionResult<(Vec<u64>, Vec<f32>)> {
    let mut row_ids = Vec::new();
    let mut scores = Vec::new();
    while let Some(batch) = input.try_next().await? {
        if batch.num_rows() > limit.saturating_sub(row_ids.len()) {
            return Err(DataFusionError::Execution(format!(
                "hybrid indexed input exceeded its bounded result limit of {limit} rows"
            )));
        }
        let ids = batch
            .column_by_name(ROW_ID)
            .ok_or_else(|| {
                DataFusionError::Execution("hybrid indexed input is missing row IDs".to_string())
            })?
            .as_primitive::<UInt64Type>();
        let values = batch
            .column_by_name(SCORE_COL)
            .ok_or_else(|| {
                DataFusionError::Execution("hybrid indexed input is missing scores".to_string())
            })?
            .as_primitive::<Float32Type>();
        if ids.null_count() != 0 || values.null_count() != 0 {
            return Err(DataFusionError::Execution(
                "hybrid indexed input contains null row IDs or scores".to_string(),
            ));
        }
        row_ids.extend_from_slice(ids.values());
        scores.extend_from_slice(values.values());
    }
    Ok((row_ids, scores))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WandExactnessCertificate {
    Exhaustive,
    Strict,
    Ambiguous,
}

/// Classify a globally merged bounded Match WAND result.
///
/// Sorting before classification is essential: per-segment WAND output is not
/// a final cross-segment ordering. A strict score gap after result k proves
/// that score-only pruning could not have discarded a row-id tie at the final
/// boundary. Returning fewer rows than requested proves exhaustion. Merely
/// observing a lower score during collection is not a proof because other
/// partitions may still contain kth-score ties.
fn classify_wand_exactness_certificate(
    documents: &mut [ScoredDoc],
    limit: usize,
    probe_limit: usize,
) -> WandExactnessCertificate {
    if limit == 0
        || probe_limit <= limit
        || documents.len() > probe_limit
        || documents
            .iter()
            .any(|document| !document.score.0.is_finite())
    {
        return WandExactnessCertificate::Ambiguous;
    }
    documents.sort_unstable_by(|left, right| {
        right
            .score
            .0
            .total_cmp(&left.score.0)
            .then_with(|| left.row_id.cmp(&right.row_id))
    });
    if documents.len() < probe_limit {
        WandExactnessCertificate::Exhaustive
    } else if documents[limit - 1].score.0.total_cmp(
        &documents
            .last()
            .expect("a full bounded probe has a guard candidate")
            .score
            .0,
    ) == Ordering::Greater
    {
        WandExactnessCertificate::Strict
    } else {
        WandExactnessCertificate::Ambiguous
    }
}

pub(super) fn finish_wand_documents(
    mut documents: Vec<ScoredDoc>,
    limit: usize,
) -> (Vec<u64>, Vec<f32>) {
    documents.truncate(limit);
    documents
        .into_iter()
        .map(|document| (document.row_id, document.score.0))
        .unzip()
}

async fn exact_prepared_match_fallback(
    indices: &[Arc<InvertedIndex>],
    query: &FtsQuery,
    params: &FtsSearchParams,
    prefilter: Arc<dyn PreFilter>,
    metrics: Arc<FtsIndexMetrics>,
    prepared_match: Arc<PreparedBm25Query>,
    score_floor: Option<f32>,
) -> Result<(Vec<u64>, Vec<f32>)> {
    if let Some(score_floor) = score_floor {
        compound_search_prepared_match_with_score_floor(
            indices,
            query,
            params,
            prefilter,
            metrics,
            prepared_match,
            score_floor,
        )
        .await
    } else {
        compound_search_prepared_match(indices, query, params, prefilter, metrics, prepared_match)
            .await
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
        let prefilter_source = match children.len() {
            0 if matches!(self.prefilter_source, PreFilterSource::None) => PreFilterSource::None,
            1 => {
                let Some(source) = children.pop() else {
                    return Err(DataFusionError::Internal(
                        "compound FTS lost its prefilter child".to_string(),
                    ));
                };
                self.prefilter_source.with_execution_plan(source)?
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
            prepared_match: self.prepared_match.clone(),
            segment_selection: self.segment_selection.clone(),
            external_mask: self.external_mask.clone(),
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
        let preset_base_scorer = self.base_scorer.clone();
        let preset_prepared_match = self.prepared_match.clone();
        let segment_selection = self.segment_selection.clone();
        let external_mask = self.external_mask.clone();
        let metrics_set = self.metrics.clone();
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
            if preset_prepared_match.is_some() && !matches!(&query, FtsQuery::Match(_)) {
                return Err(DataFusionError::Execution(
                    "CompoundQueryExec prepared vocabulary requires a root Match query".to_string(),
                ));
            }
            let scorer_only_fuzzy = preset_prepared_match.is_none()
                && compound_query_uses_fuzzy_expansion(&query)
                && preset_base_scorer.is_some();
            let scorer_override_covers_all = if scorer_only_fuzzy {
                segment_selection
                    .covers_all_committed(&dataset, column, DocumentGranularity::Row, &segments)
                    .await?
            } else {
                true
            };
            let _details = load_segment_details(&dataset, column, &segments).await?;
            let address_domain = !dataset.manifest.uses_stable_row_ids();
            let dataset_for_scope = dataset.clone();
            let indices =
                open_fts_segments(&dataset, column, &segments, &metrics.index_metrics).await?;
            if let Some(first_index) = indices.first() {
                let snapshot = tokenize_compound_query(&query, first_index.as_ref())?;
                tokenized_query.get_or_init(|| snapshot);
            }
            let mut prefilter = build_prefilter(
                context,
                partition,
                &prefilter_source,
                dataset,
                &segments,
                PreFilterMasks {
                    overlay_block: None,
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
            let base_scorer = match (preset_prepared_match.is_some(), preset_base_scorer) {
                (true, _) => None,
                (false, scorer) => scorer,
            };
            if base_scorer.is_some() && scorer_only_fuzzy && !scorer_override_covers_all {
                return Err(DataFusionError::Execution(
                    "fuzzy CompoundQueryExec cannot use a scorer-only override over a segment subset; prepare the canonical vocabulary with prepare_bm25_query and pass it with with_prepared_match"
                    .to_string(),
                ));
            }
            let certificate_limit = match (&query, params.limit) {
                (FtsQuery::Match(match_query), Some(limit))
                    if limit > 0
                        && params.wand_factor == 1.0
                        && match_query.boost.is_finite()
                        && match_query.boost > 0.0
                        && base_scorer.is_none()
                        && indices
                            .iter()
                            .all(|index| index.supports_wand_exactness_certificate()) =>
                {
                    limit
                        .checked_add(1)
                        .map(|wand_limit| (match_query.clone(), limit, wand_limit))
                }
                _ => None,
            };
            let (row_ids, scores) = if let Some((match_query, limit, wand_limit)) =
                certificate_limit
            {
                let wand_params = MatchQueryExec::effective_params(&match_query, params.clone())
                    .with_phrase_slop(None)
                    .with_limit(Some(wand_limit));
                let prepared = if let Some(prepared_match) = preset_prepared_match.clone() {
                    Arc::new(PreparedMatch {
                        query: prepared_match,
                        params: Arc::new(wand_params),
                        operator: match_query.operator,
                    })
                } else {
                    let first_index = indices.first().ok_or_else(|| {
                        DataFusionError::Execution(format!(
                            "FTS index for column {column} has no segments"
                        ))
                    })?;
                    let mut tokenizer =
                        tokenizer_for_match_query(first_index.as_ref(), match_query.fuzziness);
                    let tokens = try_collect_query_tokens(&match_query.terms, &mut tokenizer)?;
                    let scorer_start = std::time::Instant::now();
                    let prepared = Arc::new(
                        PreparedMatch::new(
                            &indices,
                            tokens,
                            wand_params,
                            match_query.operator,
                            metrics.as_ref(),
                            None,
                        )
                        .await?,
                    );
                    metrics.record_scorer_build(scorer_start.elapsed());
                    prepared
                };

                // Zero-weight terms can match documents without contributing a
                // positive score. A short score-only WAND result therefore does
                // not prove exhaustion. Preserve exact membership semantics for
                // those rare corpora without recording a certificate attempt.
                if prepared.query.scorer().token_docs.keys().any(|token| {
                    let weight = prepared.query.scorer().query_weight(token);
                    !weight.is_finite() || weight <= 0.0
                }) {
                    prefilter.wait_for_ready().await?;
                    let indices = segments_the_prefilter_may_match(
                        &dataset_for_scope,
                        &indices,
                        &segments,
                        prefilter.as_ref(),
                        address_domain,
                        &metrics,
                    )
                    .await;
                    if indices.is_empty() {
                        return Ok::<_, DataFusionError>(RecordBatch::new_empty(
                            FTS_SCHEMA.clone(),
                        ));
                    }
                    compound_search_prepared_match(
                        &indices,
                        &query,
                        &params,
                        prefilter,
                        metrics.clone(),
                        prepared.query.clone(),
                    )
                    .await?
                } else {
                    prefilter.wait_for_ready().await?;
                    let indices = segments_the_prefilter_may_match(
                        &dataset_for_scope,
                        &indices,
                        &segments,
                        prefilter.as_ref(),
                        address_domain,
                        &metrics,
                    )
                    .await;
                    if indices.is_empty() {
                        return Ok::<_, DataFusionError>(RecordBatch::new_empty(
                            FTS_SCHEMA.clone(),
                        ));
                    }
                    let mut documents = search_prepared_segments(
                        &indices,
                        prepared.clone(),
                        prefilter.clone(),
                        metrics.clone(),
                        None,
                    )
                    .await?;
                    documents.iter_mut().for_each(|document| {
                        document.score.0 *= match_query.boost;
                    });
                    match classify_wand_exactness_certificate(&mut documents, limit, wand_limit) {
                        WandExactnessCertificate::Exhaustive => {
                            finish_wand_documents(documents, limit)
                        }
                        WandExactnessCertificate::Strict => finish_wand_documents(documents, limit),
                        WandExactnessCertificate::Ambiguous => {
                            let score_floor = documents
                                .get(limit - 1)
                                .map(|document| document.score.0)
                                .filter(|score| score.is_finite());
                            let completion_limit = limit
                                .checked_add(WAND_TIE_COMPLETION_BUDGET)
                                .and_then(|limit| limit.checked_add(1));
                            if let (Some(score_floor), Some(completion_limit)) =
                                (score_floor, completion_limit)
                            {
                                let completion_prepared = Arc::new(PreparedMatch {
                                    query: prepared.query.clone(),
                                    params: Arc::new(
                                        prepared
                                            .params
                                            .as_ref()
                                            .clone()
                                            .with_limit(Some(completion_limit)),
                                    ),
                                    operator: prepared.operator,
                                });
                                let raw_score_floor =
                                    exclusive_scaled_score_floor(score_floor, match_query.boost);
                                let mut completion = search_prepared_segments(
                                    &indices,
                                    completion_prepared,
                                    prefilter.clone(),
                                    metrics.clone(),
                                    raw_score_floor,
                                )
                                .await?;
                                completion.iter_mut().for_each(|document| {
                                    document.score.0 *= match_query.boost;
                                });
                                match classify_wand_exactness_certificate(
                                    &mut completion,
                                    limit,
                                    completion_limit,
                                ) {
                                    WandExactnessCertificate::Exhaustive => {
                                        finish_wand_documents(completion, limit)
                                    }
                                    WandExactnessCertificate::Strict => {
                                        finish_wand_documents(completion, limit)
                                    }
                                    WandExactnessCertificate::Ambiguous => {
                                        let seeded_floor = completion
                                            .iter()
                                            .all(|document| document.score.0.is_finite())
                                            .then_some(score_floor);
                                        exact_prepared_match_fallback(
                                            &indices,
                                            &query,
                                            &params,
                                            prefilter,
                                            metrics.clone(),
                                            prepared.query.clone(),
                                            seeded_floor,
                                        )
                                        .await?
                                    }
                                }
                            } else {
                                exact_prepared_match_fallback(
                                    &indices,
                                    &query,
                                    &params,
                                    prefilter,
                                    metrics.clone(),
                                    prepared.query.clone(),
                                    score_floor,
                                )
                                .await?
                            }
                        }
                    }
                }
            } else {
                // With statistics prepared elsewhere the candidate search can
                // skip segments the prefilter excludes; without them the
                // search derives its statistics from the segments it is given
                // and must see all of them.
                let candidate_indices = async {
                    prefilter.wait_for_ready().await?;
                    Ok::<_, DataFusionError>(
                        segments_the_prefilter_may_match(
                            &dataset_for_scope,
                            &indices,
                            &segments,
                            prefilter.as_ref(),
                            address_domain,
                            &metrics,
                        )
                        .await,
                    )
                };
                match (preset_prepared_match, base_scorer) {
                    (Some(prepared_match), _) => {
                        let indices = candidate_indices.await?;
                        if indices.is_empty() {
                            return Ok::<_, DataFusionError>(RecordBatch::new_empty(
                                FTS_SCHEMA.clone(),
                            ));
                        }
                        compound_search_prepared_match(
                            &indices,
                            &query,
                            &params,
                            prefilter,
                            metrics.clone(),
                            prepared_match,
                        )
                        .await?
                    }
                    (None, Some(base_scorer)) => {
                        // A scorer alone does not fix the vocabulary: fuzzy
                        // expansion still runs over the segments handed in
                        // here, so the candidate set stays complete.
                        drop(candidate_indices);
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
                    (None, None) => {
                        drop(candidate_indices);
                        compound_search(&indices, &query, &params, prefilter, metrics.clone())
                            .await?
                    }
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
    segments: Arc<[IndexMetadata]>,
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
    /// Combined into the prefilter so only masked rows are scored (see
    /// [`MatchQueryExec::with_external_mask`]).
    external_mask: Option<Arc<RowAddrMask>>,
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
                segments: Arc::from(segments),
            })
            .collect::<Vec<_>>();
        Ok(Self {
            dataset,
            query,
            tokenized_query: Arc::new(OnceLock::new()),
            params,
            prefilter_source,
            columns: Arc::from(columns),
            external_mask: None,
            properties: Arc::new(PlanProperties::new(
                EquivalenceProperties::new(FTS_SCHEMA.clone()),
                Partitioning::RoundRobinBatch(1),
                EmissionType::Final,
                Boundedness::Bounded,
            )),
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    /// See [`MatchQueryExec::with_external_mask`].
    pub fn with_external_mask(mut self, mask: Option<Arc<RowAddrMask>>) -> Self {
        self.external_mask = mask;
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

    /// The exact resolved segments for each column, preserving both column order
    /// and each column's segment order. Segment ordinals across columns are not
    /// aligned and must not be zipped together.
    ///
    /// These are the inputs accepted by [`Self::new_with_segments`]. Execution
    /// still validates the selected segments' document granularity before search.
    ///
    /// ```
    /// # use lance::{io::exec::fts::CrossColumnCompoundQueryExec, Result};
    /// # fn rebuild(exec: &CrossColumnCompoundQueryExec)
    /// #     -> Result<CrossColumnCompoundQueryExec> {
    /// let columns = exec.column_segments()
    ///     .map(|(column, segments)| (column.to_owned(), segments.to_vec()))
    ///     .collect();
    /// Ok(CrossColumnCompoundQueryExec::new_with_segments(
    ///     exec.dataset().clone(), exec.query().clone(), exec.params().clone(),
    ///     exec.prefilter_source().clone(), columns,
    /// )?.with_external_mask(exec.external_mask().cloned()))
    /// # }
    /// ```
    pub fn column_segments(&self) -> impl ExactSizeIterator<Item = (&str, &[IndexMetadata])> {
        self.columns
            .iter()
            .map(|selection| (selection.column.as_str(), selection.segments.as_ref()))
    }

    /// Additional candidate restriction in the indices' row-id domain, or
    /// `None` for no external mask. Corpus statistics remain independent of it.
    /// See [`Self::column_segments`] for a reconstruction example.
    pub fn external_mask(&self) -> Option<&Arc<RowAddrMask>> {
        self.external_mask.as_ref()
    }

    /// Re-cut this scorer at `limit`, keeping its segment selection, prepared
    /// scorers and masks.
    ///
    /// The FTS top-k lives in `FtsSearchParams`, not in an enclosing fetch
    /// node, so a caller that needs more candidates than the user asked for —
    /// over-fetching to survive a later dedup, say — has no other way to raise
    /// it. Everything that decides *which* rows are eligible is carried over
    /// untouched, so this widens the cut without widening the domain.
    pub fn with_limit(&self, limit: usize) -> Self {
        let mut params = self.params.clone();
        params.limit = Some(limit);
        Self {
            dataset: self.dataset.clone(),
            query: self.query.clone(),
            tokenized_query: self.tokenized_query.clone(),
            params,
            prefilter_source: self.prefilter_source.clone(),
            columns: self.columns.clone(),
            external_mask: self.external_mask.clone(),
            properties: self.properties.clone(),
            metrics: ExecutionPlanMetricsSet::new(),
        }
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
        let prefilter_source = match children.len() {
            0 if matches!(self.prefilter_source, PreFilterSource::None) => PreFilterSource::None,
            1 => {
                let Some(source) = children.pop() else {
                    return Err(DataFusionError::Internal(
                        "cross-column compound FTS lost its prefilter child".to_string(),
                    ));
                };
                self.prefilter_source.with_execution_plan(source)?
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
            external_mask: self.external_mask.clone(),
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
        let external_mask = self.external_mask.clone();
        let metrics_set = self.metrics.clone();
        let metrics = Arc::new(FtsIndexMetrics::new(&self.metrics, partition));

        let stream = stream::once(async move {
            let _timer = metrics.baseline_metrics.elapsed_compute().timer();
            let selected_segments = columns
                .iter()
                .flat_map(|selection| selection.segments.iter().cloned())
                .collect::<Vec<_>>();
            // DatasetPreFilter starts its deletion and filter prerequisites in
            // the background. Construct it before opening index segments so
            // both I/O paths can make progress concurrently.
            let mut prefilter = build_prefilter(
                context,
                partition,
                &prefilter_source,
                dataset.clone(),
                &selected_segments,
                PreFilterMasks {
                    overlay_block: None,
                    external_mask,
                },
                &metrics_set,
            )?;
            let opened_columns = try_join_all(columns.iter().cloned().map(|selection| {
                let dataset = dataset.clone();
                let metrics = metrics.clone();
                async move {
                    let segments = FtsSegmentSelection::ExactResolved(selection.segments)
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

#[cfg(test)]
mod tests {
    use super::*;
    use lance_index::scalar::inverted::builder::ScoredDoc;

    #[test]
    fn test_wand_exactness_certificate_classification() {
        let documents = |scores: &[f32]| {
            scores
                .iter()
                .enumerate()
                .rev()
                .map(|(row_id, score)| ScoredDoc::new(row_id as u64, *score))
                .collect::<Vec<_>>()
        };

        let mut exhaustive = documents(&[3.0, 2.0]);
        assert_eq!(
            classify_wand_exactness_certificate(&mut exhaustive, 3, 4),
            WandExactnessCertificate::Exhaustive
        );

        let mut strict = documents(&[4.0, 3.0, 3.0, 1.0]);
        assert_eq!(
            classify_wand_exactness_certificate(&mut strict, 3, 4),
            WandExactnessCertificate::Strict
        );
        assert_eq!(
            strict
                .iter()
                .map(|document| document.row_id)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3],
            "ties wholly inside top-k must use row-id ordering without forcing fallback"
        );

        let mut ambiguous = documents(&[4.0, 3.0, 2.0, 2.0]);
        assert_eq!(
            classify_wand_exactness_certificate(&mut ambiguous, 3, 4),
            WandExactnessCertificate::Ambiguous
        );

        let mut non_finite = documents(&[4.0, f32::INFINITY]);
        assert_eq!(
            classify_wand_exactness_certificate(&mut non_finite, 1, 2),
            WandExactnessCertificate::Ambiguous
        );

        let mut zero_limit = documents(&[1.0]);
        assert_eq!(
            classify_wand_exactness_certificate(&mut zero_limit, 0, 1),
            WandExactnessCertificate::Ambiguous
        );

        let mut reversed_segments = vec![
            ScoredDoc::new(99, 2.0),
            ScoredDoc::new(50, 3.0),
            ScoredDoc::new(1, 2.0),
        ];
        assert_eq!(
            classify_wand_exactness_certificate(&mut reversed_segments, 2, 4),
            WandExactnessCertificate::Exhaustive
        );
        assert_eq!(
            reversed_segments
                .iter()
                .map(|document| document.row_id)
                .collect::<Vec<_>>(),
            vec![50, 1, 99],
            "completed ties must use final row-id order, not segment arrival order"
        );

        let completion_limit = 1 + WAND_TIE_COMPLETION_BUDGET + 1;
        let mut at_budget = (0..=WAND_TIE_COMPLETION_BUDGET)
            .rev()
            .map(|row_id| ScoredDoc::new(row_id as u64, 2.0))
            .chain(std::iter::once(ScoredDoc::new(u64::MAX, 1.0)))
            .collect::<Vec<_>>();
        assert_eq!(at_budget.len(), completion_limit);
        assert_eq!(
            classify_wand_exactness_certificate(&mut at_budget, 1, completion_limit),
            WandExactnessCertificate::Strict,
            "the completion budget includes a slot for a strict lower-score guard"
        );
        assert_eq!(at_budget[0].row_id, 0);

        let mut overflow = (0..completion_limit)
            .rev()
            .map(|row_id| ScoredDoc::new(row_id as u64, 2.0))
            .collect::<Vec<_>>();
        assert_eq!(
            classify_wand_exactness_certificate(&mut overflow, 1, completion_limit),
            WandExactnessCertificate::Ambiguous,
            "a full probe with no lower-score guard must replay exactly"
        );
    }
}
