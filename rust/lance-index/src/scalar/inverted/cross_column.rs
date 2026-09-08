// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Posting-backed compound FTS over more than one indexed column.
//!
//! A partition-local FTS scorer iterates `DocId`s, but `DocId` is only stable
//! within one partition of one column.  This module maps every leaf source to
//! the dataset row-address domain before composing the query.  Consequently,
//! columns may have different segment and partition boundaries without an
//! intermediate hash join or a materialized result set per query node.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use futures::{StreamExt, TryStreamExt, stream};
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu};
use lance_core::{Error, Result};
use lance_select::RowAddrMask;
use roaring::{RoaringBitmap, RoaringTreemap};

use super::compound::{
    BoxScorer, ComposableScorer, CompoundLeafPlanInput, CompoundPlanAnalysis, CompoundScorerPlan,
    EmptyScorer, LeafQuery, MaterializedScorer, RowAddressMergeScorer, RowAddressSource,
    ScoreBounds, ScoredRow, TopKCollector, collect_leaf_queries, map_scorer_to_row_addresses,
    prepare_row_address_projection, tokenize_leaf,
};
use super::documents::{
    DocId, DocLengths, DocVisibility, PartitionDocuments, ResidentAddressProjection,
};
use super::index::{InvertedPartition, PostingLoadOptions};
use super::query::{FtsQuery, FtsSearchParams, Operator};
use super::scorer::MemBM25Scorer;
use super::wand::{
    FLAT_SEARCH_PERCENT_THRESHOLD, FlatDocuments, PostingIterator, WandCursor, WandDocuments,
};
use super::{
    DocInfo, DocumentGranularity, InvertedIndex, PreparedBm25Query, final_query_tokens,
    has_all_query_positions,
};
use crate::metrics::MetricsCollector;
use crate::prefilter::PreFilter;

const MAX_CONCURRENT_SOURCE_LOADS: usize = 32;

/// Stage optional/prohibited source I/O only when the positive generator is
/// expected to match at most this percentage of its column corpus.
const MAX_STAGED_GENERATOR_PERCENT: usize = 1;

/// Very small candidate sets are cheap enough to stage regardless of corpus
/// size and keep the path useful for small shards.
const MIN_STAGED_GENERATOR_CANDIDATES: usize = 128;

/// Bound the row-address candidate buffer even for very large corpora.
const MAX_STAGED_GENERATOR_CANDIDATES: usize = 1_000_000;

struct PreparedCrossColumnLeaf {
    column_ordinal: usize,
    query: Arc<PreparedBm25Query>,
    params: Arc<FtsSearchParams>,
    operator: Operator,
}

struct LoadedCrossColumnLeaf {
    leaf_ordinal: usize,
    postings: Vec<PostingIterator>,
    params: Arc<FtsSearchParams>,
    operator: Operator,
    scorer: Arc<MemBM25Scorer>,
}

fn local_candidate_lower_bound(operator: Operator, postings: &[PostingIterator]) -> Option<u64> {
    match operator {
        Operator::Or => postings
            .iter()
            .filter_map(PostingIterator::current_doc_id)
            .min(),
        Operator::And => postings
            .iter()
            .map(PostingIterator::current_doc_id)
            .try_fold(0, |lower_bound, doc| doc.map(|doc| lower_bound.max(doc))),
    }
}

fn compare_leaf_mapping_priority(
    left_leaf_ordinal: usize,
    left_cost: usize,
    right_leaf_ordinal: usize,
    right_cost: usize,
) -> Ordering {
    right_cost
        .cmp(&left_cost)
        .then_with(|| left_leaf_ordinal.cmp(&right_leaf_ordinal))
}

struct LoadedCrossColumnSource {
    num_docs: usize,
    lengths: LoadedScoringLengths,
    visibility: DocVisibility,
    projection: ResidentAddressProjection,
    leaves: Vec<LoadedCrossColumnLeaf>,
}

enum LoadedScoringLengths {
    Dense(Arc<DocLengths>),
    Sparse(Vec<ResolvedCandidateDocument>),
}

#[derive(Clone, Copy)]
struct ResolvedCandidateDocument {
    doc_id: u32,
    row_address: u64,
    scoring_length: u32,
}

struct LoadedGeneratorSource {
    documents: Arc<PartitionDocuments>,
    visibility: DocVisibility,
    leaves: Vec<LoadedCrossColumnLeaf>,
}

/// Document view used by the two CPU phases of staged generation.
///
/// The membership pass supplies no lengths and runs with a negative WAND
/// floor, so every exact match remains visible. The scoring pass supplies
/// lengths only for selected generator candidates; non-selected posting docs
/// use zero solely while they are advanced past and can never escape through
/// `document_key`.
#[derive(Clone, Copy)]
enum ScoringLengths<'a> {
    Missing,
    Dense(&'a DocLengths),
    Sparse(&'a [ResolvedCandidateDocument]),
}

struct StagedWandDocuments<'a> {
    num_docs: usize,
    visibility: &'a DocVisibility,
    scoring_lengths: ScoringLengths<'a>,
}

impl StagedWandDocuments<'_> {
    fn scoring_length(&self, doc_id: u32) -> u32 {
        match self.scoring_lengths {
            ScoringLengths::Missing => 0,
            ScoringLengths::Dense(lengths) => lengths.scoring(DocId::new(doc_id)),
            ScoringLengths::Sparse(documents) => documents
                .binary_search_by_key(&doc_id, |document| document.doc_id)
                .ok()
                .map(|index| documents[index].scoring_length)
                .unwrap_or_default(),
        }
    }

    fn row_address(&self, doc_id: u32) -> Option<u64> {
        let ScoringLengths::Sparse(documents) = self.scoring_lengths else {
            return None;
        };
        documents
            .binary_search_by_key(&doc_id, |document| document.doc_id)
            .ok()
            .map(|index| documents[index].row_address)
    }
}

impl WandDocuments for StagedWandDocuments<'_> {
    type Candidate = DocId;

    fn len(&self) -> usize {
        self.num_docs
    }

    fn visible_cost_upper_bound(&self) -> usize {
        self.visibility.len(self.num_docs)
    }

    fn scoring_norms(&self) -> Option<&[u8]> {
        match self.scoring_lengths {
            ScoringLengths::Dense(lengths) => lengths.scoring_norms(),
            ScoringLengths::Missing | ScoringLengths::Sparse(_) => None,
        }
    }

    fn scoring_num_tokens(&self, doc_id: u32) -> u32 {
        self.scoring_length(doc_id)
    }

    fn doc_length(&self, doc: &DocInfo) -> u32 {
        match doc {
            DocInfo::Raw(doc) => self.scoring_length(doc.doc_id),
            DocInfo::Located(_) => unreachable!("modern posting lists contain dense DocIds"),
        }
    }

    fn document_key(&self, doc: &DocInfo) -> Option<u64> {
        match doc {
            DocInfo::Raw(doc) if self.visibility.selected(DocId::new(doc.doc_id)) => {
                Some(u64::from(doc.doc_id))
            }
            DocInfo::Raw(_) => None,
            DocInfo::Located(_) => unreachable!("modern posting lists contain dense DocIds"),
        }
    }

    fn document_key_for_doc_id(&self, doc_id: u32) -> Option<u64> {
        self.visibility
            .selected(DocId::new(doc_id))
            .then_some(u64::from(doc_id))
    }

    fn candidate_from_key(&self, key: u64) -> Self::Candidate {
        DocId::new(key as u32)
    }

    fn flat_documents(&self) -> Option<FlatDocuments<'_>> {
        if matches!(self.scoring_lengths, ScoringLengths::Missing) {
            return None;
        }
        self.visibility.iter().map(|doc_ids| {
            let len = self.visibility.len(self.num_docs);
            let docs = doc_ids.map(|doc_id| {
                let value = u64::from(doc_id.get());
                (value, value)
            });
            (len, Box::new(docs) as Box<dyn Iterator<Item = (u64, u64)>>)
        })
    }

    fn flat_doc_length(&self, doc_id: u64, _document_key: u64, _compressed: bool) -> u32 {
        u32::try_from(doc_id)
            .ok()
            .map(|doc_id| self.scoring_length(doc_id))
            .unwrap_or_default()
    }
}

#[derive(Clone)]
struct SourceDescriptor {
    column_ordinal: usize,
    segment_ordinal: usize,
    partition: Arc<InvertedPartition>,
}

fn staged_candidate_budget(num_docs: usize, limit: usize) -> usize {
    let percentage_budget = num_docs
        .saturating_mul(MAX_STAGED_GENERATOR_PERCENT)
        .div_ceil(100);
    percentage_budget.max(limit).clamp(
        MIN_STAGED_GENERATOR_CANDIDATES,
        MAX_STAGED_GENERATOR_CANDIDATES,
    )
}

fn leaf_plan_input(leaf: &PreparedCrossColumnLeaf) -> Result<CompoundLeafPlanInput> {
    let mut seen_terms = HashSet::<(u32, String)>::new();
    let mut costs_by_position = HashMap::<u32, usize>::new();
    let tokens = leaf.query.tokens();
    for token_index in 0..tokens.len() {
        let position = tokens.position(token_index);
        let token = tokens.get_token(token_index);
        if seen_terms.insert((position, token.to_owned())) {
            let frequency = leaf.query.scorer().num_docs_containing_token(token);
            let position_cost = costs_by_position.entry(position).or_default();
            *position_cost = position_cost.saturating_add(frequency);
        }
    }

    let requires_every_position =
        leaf.operator == Operator::And || leaf.params.phrase_slop.is_some();
    let possible = (!requires_every_position || leaf.query.has_all_query_positions())
        && !costs_by_position.is_empty()
        && if requires_every_position {
            costs_by_position.values().all(|cost| *cost > 0)
        } else {
            costs_by_position.values().any(|cost| *cost > 0)
        };
    if !possible {
        return Ok(CompoundLeafPlanInput::new(false, 0, ScoreBounds::ZERO));
    }

    // Position alternatives can overlap, so this is deliberately an upper
    // estimate. Overestimating only disables staging; the actual candidate
    // count is guarded independently before any probe source is skipped.
    let cost = if requires_every_position {
        costs_by_position.values().copied().min().unwrap_or(0)
    } else {
        costs_by_position
            .values()
            .copied()
            .fold(0, usize::saturating_add)
    }
    .min(leaf.query.scorer().num_docs());
    Ok(CompoundLeafPlanInput::new(
        true,
        cost,
        ScoreBounds::try_new(0.0, f32::INFINITY)?,
    ))
}

fn staged_generator(
    analysis: &CompoundPlanAnalysis,
    inputs: &[CompoundLeafPlanInput],
    leaves: &[PreparedCrossColumnLeaf],
    limit: usize,
) -> Option<(Vec<usize>, usize)> {
    if analysis.generator_leaves.is_empty() {
        return None;
    }
    let generator_leaves = analysis
        .generator_leaves
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    if !inputs
        .iter()
        .enumerate()
        .any(|(leaf_ordinal, input)| input.possible && !generator_leaves.contains(&leaf_ordinal))
    {
        return None;
    }
    let num_docs = analysis
        .generator_leaves
        .iter()
        .map(|&leaf_ordinal| {
            leaves
                .get(leaf_ordinal)
                .map(|leaf| leaf.query.scorer().num_docs())
        })
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .max()
        .unwrap_or(0);
    let candidate_budget = staged_candidate_budget(num_docs, limit);
    (analysis.generator_cost <= candidate_budget)
        .then_some((analysis.generator_leaves.clone(), candidate_budget))
}

fn query_state_is_prewarmed(
    columns: &[(String, Vec<Arc<InvertedIndex>>)],
    leaves: &[PreparedCrossColumnLeaf],
) -> bool {
    columns
        .iter()
        .enumerate()
        .all(|(column_ordinal, (_, indices))| {
            let with_position = leaves.iter().any(|leaf| {
                leaf.column_ordinal == column_ordinal && leaf.params.phrase_slop.is_some()
            });
            indices
                .iter()
                .all(|index| index.prewarmed_query_state_ready(with_position))
        })
}

fn leaf_column(leaf: &LeafQuery) -> Option<&str> {
    match leaf {
        LeafQuery::Match(query) => query.column.as_deref(),
        LeafQuery::Phrase(query) => query.column.as_deref(),
    }
}

fn validate_row_leaf_granularities(leaves: &[LeafQuery]) -> Result<()> {
    for (leaf_ordinal, leaf) in leaves.iter().enumerate() {
        let (leaf_kind, column, document_granularity) = match leaf {
            LeafQuery::Match(query) => {
                ("Match", query.column.as_deref(), query.document_granularity)
            }
            LeafQuery::Phrase(query) => (
                "Phrase",
                query.column.as_deref(),
                query.document_granularity,
            ),
        };
        if document_granularity == Some(DocumentGranularity::ListElement) {
            let column = column.unwrap_or("<unspecified>");
            return Err(Error::invalid_input(format!(
                "cross-column compound FTS {leaf_kind} leaf {leaf_ordinal} for column '{column}' requested ListElement document granularity, but only Row is supported"
            )));
        }
    }
    Ok(())
}

/// Validate the query-local column domain and return the input column ordinal
/// for every query leaf.  Keeping this separate from index validation makes it
/// possible to test plan/leaf alignment without constructing index fixtures.
fn resolve_leaf_columns(column_names: &[String], leaves: &[LeafQuery]) -> Result<Vec<usize>> {
    if column_names.len() < 2 {
        return Err(Error::invalid_input(
            "cross-column compound FTS requires at least two columns",
        ));
    }

    let mut columns_by_name = HashMap::with_capacity(column_names.len());
    for (column_ordinal, column) in column_names.iter().enumerate() {
        if column.is_empty() {
            return Err(Error::invalid_input(
                "cross-column compound FTS column names cannot be empty",
            ));
        }
        if columns_by_name
            .insert(column.as_str(), column_ordinal)
            .is_some()
        {
            return Err(Error::invalid_input(format!(
                "cross-column compound FTS received duplicate column '{column}'"
            )));
        }
    }

    let mut referenced_columns = HashSet::with_capacity(column_names.len());
    let mut leaf_columns = Vec::with_capacity(leaves.len());
    for (leaf_ordinal, leaf) in leaves.iter().enumerate() {
        let column = leaf_column(leaf).ok_or_else(|| {
            Error::invalid_input(format!(
                "cross-column compound FTS leaf {leaf_ordinal} is missing a column"
            ))
        })?;
        let column_ordinal = columns_by_name.get(column).copied().ok_or_else(|| {
            Error::invalid_input(format!(
                "cross-column compound FTS leaf {leaf_ordinal} references column '{column}', which has no supplied index"
            ))
        })?;
        referenced_columns.insert(column_ordinal);
        leaf_columns.push(column_ordinal);
    }

    if referenced_columns.len() != column_names.len() {
        let unused = column_names
            .iter()
            .enumerate()
            .filter(|(ordinal, _)| !referenced_columns.contains(ordinal))
            .map(|(_, column)| column.as_str())
            .collect::<Vec<_>>();
        return Err(Error::invalid_input(format!(
            "cross-column compound FTS received indices for unreferenced columns: {}",
            unused.join(", ")
        )));
    }

    Ok(leaf_columns)
}

fn validate_modern_row_indices(columns: &[(String, Vec<Arc<InvertedIndex>>)]) -> Result<()> {
    for (column, indices) in columns {
        if indices.is_empty() {
            return Err(Error::invalid_input(format!(
                "cross-column compound FTS column '{column}' has no index segments"
            )));
        }
        for (segment_ordinal, index) in indices.iter().enumerate() {
            if index.is_legacy() {
                return Err(Error::invalid_input(format!(
                    "cross-column compound FTS requires modern indices, but column '{column}' segment {segment_ordinal} is legacy"
                )));
            }
            for (partition_ordinal, partition) in index.partitions.iter().enumerate() {
                if partition.docs.modern().is_none() {
                    return Err(Error::invalid_input(format!(
                        "cross-column compound FTS requires modern documents, but column '{column}' segment {segment_ordinal} partition {partition_ordinal} is legacy"
                    )));
                }
                if partition.docs.coordinate_rank() != 0 {
                    return Err(Error::invalid_input(format!(
                        "cross-column compound FTS only supports row documents, but column '{column}' segment {segment_ordinal} partition {partition_ordinal} has coordinate rank {}",
                        partition.docs.coordinate_rank()
                    )));
                }
            }
        }
    }
    Ok(())
}

async fn build_column_scorer(
    indices: &[Arc<InvertedIndex>],
    terms: Vec<String>,
    metrics: Arc<dyn MetricsCollector>,
) -> Result<Arc<MemBM25Scorer>> {
    let terms = Arc::new(terms);
    let parallelism = get_num_compute_intensive_cpus()
        .clamp(1, MAX_CONCURRENT_SOURCE_LOADS)
        .min(indices.len().max(1));
    let stats = stream::iter(indices.iter().cloned().map(|index| {
        let terms = terms.clone();
        let metrics = metrics.clone();
        async move {
            index
                .bm25_stats_for_terms(terms.as_ref(), Some(metrics.as_ref()))
                .await
        }
    }))
    .buffer_unordered(parallelism)
    .try_collect::<Vec<_>>()
    .await?;

    let mut total_tokens = 0_u64;
    let mut num_docs = 0_usize;
    let mut term_doc_freqs = vec![0_usize; terms.len()];
    for (segment_total_tokens, segment_num_docs, segment_term_doc_freqs) in stats {
        if segment_term_doc_freqs.len() != terms.len() {
            return Err(Error::internal(format!(
                "FTS segment returned {} document frequencies for {} requested terms",
                segment_term_doc_freqs.len(),
                terms.len()
            )));
        }
        total_tokens = total_tokens
            .checked_add(segment_total_tokens)
            .ok_or_else(|| Error::index("cross-column FTS corpus token count overflows u64"))?;
        num_docs = num_docs.checked_add(segment_num_docs).ok_or_else(|| {
            Error::index("cross-column FTS corpus document count overflows usize")
        })?;
        for (total, segment) in term_doc_freqs.iter_mut().zip(segment_term_doc_freqs) {
            *total = total.checked_add(segment).ok_or_else(|| {
                Error::index("cross-column FTS term document frequency overflows usize")
            })?;
        }
    }

    let token_docs = terms
        .iter()
        .cloned()
        .zip(term_doc_freqs)
        .collect::<HashMap<_, _>>();
    Ok(Arc::new(MemBM25Scorer::new(
        total_tokens,
        num_docs,
        token_docs,
    )))
}

async fn prepare_column_leaves(
    column_ordinal: usize,
    indices: &[Arc<InvertedIndex>],
    leaf_queries: &[(usize, LeafQuery)],
    params: &FtsSearchParams,
    metrics: Arc<dyn MetricsCollector>,
) -> Result<Vec<(usize, PreparedCrossColumnLeaf)>> {
    let first_index = indices.first().ok_or_else(|| {
        Error::invalid_input("cross-column compound FTS requires at least one index segment")
    })?;
    let mut leaf_metadata = Vec::with_capacity(leaf_queries.len());
    let mut union_terms = Vec::new();
    let mut seen_terms = HashSet::new();

    for (leaf_ordinal, leaf) in leaf_queries {
        let effective_params = leaf.effective_params(params);
        let tokens = tokenize_leaf(first_index, leaf, &effective_params);
        let final_tokens = Arc::new(final_query_tokens(indices, &tokens, &effective_params)?);
        for token in final_tokens.as_ref() {
            if seen_terms.insert(token.clone()) {
                union_terms.push(token.clone());
            }
        }
        let has_all_positions = has_all_query_positions(&tokens, final_tokens.as_ref());
        leaf_metadata.push((
            *leaf_ordinal,
            final_tokens,
            has_all_positions,
            Arc::new(effective_params),
            leaf.operator(),
        ));
    }

    // One union-term scorer per column preserves the existing metadata-I/O
    // boundary while every leaf keeps its own canonical vocabulary budget.
    let scorer = build_column_scorer(indices, union_terms, metrics).await?;
    Ok(leaf_metadata
        .into_iter()
        .map(
            |(leaf_ordinal, tokens, has_all_positions, params, operator)| {
                (
                    leaf_ordinal,
                    PreparedCrossColumnLeaf {
                        column_ordinal,
                        query: Arc::new(PreparedBm25Query::from_parts(
                            tokens,
                            scorer.clone(),
                            has_all_positions,
                        )),
                        params,
                        operator,
                    },
                )
            },
        )
        .collect())
}

fn viable_leaf_ordinals(
    column_ordinal: usize,
    _segment_ordinal: usize,
    partition: &InvertedPartition,
    prepared_leaves: &[PreparedCrossColumnLeaf],
    leaf_ordinals: &[usize],
) -> Result<Vec<usize>> {
    let mut viable_leaf_ordinals = Vec::with_capacity(leaf_ordinals.len());
    for &leaf_ordinal in leaf_ordinals {
        let leaf = prepared_leaves.get(leaf_ordinal).ok_or_else(|| {
            Error::internal(format!(
                "cross-column FTS source references missing leaf {leaf_ordinal}"
            ))
        })?;
        if leaf.column_ordinal != column_ordinal {
            return Err(Error::internal(format!(
                "cross-column FTS leaf {leaf_ordinal} belongs to column {}, not {column_ordinal}",
                leaf.column_ordinal
            )));
        }
        let tokens = leaf.query.tokens();
        if partition.may_match_tokens(
            tokens.as_ref(),
            leaf.operator,
            leaf.params.phrase_slop.is_some(),
        ) {
            viable_leaf_ordinals.push(leaf_ordinal);
        }
    }
    Ok(viable_leaf_ordinals)
}

async fn load_source_leaves(
    partition: Arc<InvertedPartition>,
    _segment_ordinal: usize,
    viable_leaf_ordinals: Vec<usize>,
    prepared_leaves: Arc<Vec<PreparedCrossColumnLeaf>>,
    metrics: Arc<dyn MetricsCollector>,
) -> Result<Vec<LoadedCrossColumnLeaf>> {
    let leaf_parallelism = partition
        .store()
        .io_parallelism()
        .max(1)
        .min(viable_leaf_ordinals.len());
    let leaves = stream::iter(viable_leaf_ordinals.into_iter().map(|leaf_ordinal| {
        let partition = partition.clone();
        let prepared_leaves = prepared_leaves.clone();
        let metrics = metrics.clone();
        async move {
            let leaf = prepared_leaves.get(leaf_ordinal).ok_or_else(|| {
                Error::internal(format!(
                    "cross-column FTS source references missing leaf {leaf_ordinal}"
                ))
            })?;
            let tokens = leaf.query.tokens();
            let postings = if tokens.is_empty()
                || ((leaf.operator == Operator::And || leaf.params.phrase_slop.is_some())
                    && !leaf.query.has_all_query_positions())
            {
                Vec::new()
            } else {
                partition
                    .load_posting_lists_with_policy(
                        tokens.as_ref(),
                        leaf.params.as_ref(),
                        leaf.operator,
                        leaf.query.scorer().as_ref(),
                        metrics.as_ref(),
                        PostingLoadOptions::cache_aware_exact(true),
                    )
                    .await?
                    .postings
            };
            Result::Ok(LoadedCrossColumnLeaf {
                leaf_ordinal,
                postings,
                params: leaf.params.clone(),
                operator: leaf.operator,
                scorer: leaf.query.scorer().clone(),
            })
        }
    }))
    .buffer_unordered(leaf_parallelism)
    .try_collect::<Vec<_>>()
    .await?
    .into_iter()
    .filter(|leaf| !leaf.postings.is_empty())
    .collect::<Vec<_>>();
    Ok(leaves)
}

async fn source_visibility(
    documents: &Arc<PartitionDocuments>,
    mask: Arc<RowAddrMask>,
) -> Result<DocVisibility> {
    let materialize_selected = mask.max_len().is_some_and(|selected| {
        u128::from(selected).saturating_mul(100)
            <= u128::from(*FLAT_SEARCH_PERCENT_THRESHOLD).saturating_mul(documents.len() as u128)
    });
    match documents.immediate_visibility(mask.clone(), materialize_selected) {
        Some(visibility) => Ok(visibility),
        None => documents.visibility(mask, materialize_selected).await,
    }
}

async fn load_masked_cross_column_source_for_leaves(
    descriptor: SourceDescriptor,
    prepared_leaves: Arc<Vec<PreparedCrossColumnLeaf>>,
    leaf_ordinals: Vec<usize>,
    mask: Arc<RowAddrMask>,
    metrics: Arc<dyn MetricsCollector>,
) -> Result<Option<LoadedCrossColumnSource>> {
    let SourceDescriptor {
        column_ordinal,
        segment_ordinal,
        partition,
    } = descriptor;
    let viable_leaf_ordinals = viable_leaf_ordinals(
        column_ordinal,
        segment_ordinal,
        partition.as_ref(),
        prepared_leaves.as_ref(),
        &leaf_ordinals,
    )?;
    if viable_leaf_ordinals.is_empty() {
        return Ok(None);
    }

    let documents = partition.docs.modern().cloned().ok_or_else(|| {
        Error::internal("cross-column FTS source changed from modern to legacy documents")
    })?;
    let visibility = source_visibility(&documents, mask).await?;
    if visibility.is_empty() {
        return Ok(None);
    }

    // Visibility is resolved before posting reads so a filtered-out source
    // never touches its posting payloads.
    let leaves = load_source_leaves(
        partition,
        segment_ordinal,
        viable_leaf_ordinals,
        prepared_leaves,
        metrics,
    )
    .await?;
    if leaves.is_empty() {
        return Ok(None);
    }

    // Only sources with at least one matching posting need scoring lengths or
    // row-address projection. These independent document columns load once and
    // in parallel.
    let lengths = async {
        match documents.cached_lengths() {
            Some(lengths) => Ok(lengths),
            None => documents.lengths().await,
        }
    };
    let projection = documents.address_projection();
    let (lengths, projection) = futures::try_join!(lengths, projection)?;

    Ok(Some(LoadedCrossColumnSource {
        num_docs: documents.len(),
        lengths: LoadedScoringLengths::Dense(lengths),
        visibility,
        projection,
        leaves,
    }))
}

async fn load_masked_generator_source(
    descriptor: SourceDescriptor,
    prepared_leaves: Arc<Vec<PreparedCrossColumnLeaf>>,
    leaf_ordinals: Vec<usize>,
    mask: Arc<RowAddrMask>,
    metrics: Arc<dyn MetricsCollector>,
) -> Result<Option<LoadedGeneratorSource>> {
    let SourceDescriptor {
        column_ordinal,
        segment_ordinal,
        partition,
    } = descriptor;
    let viable_leaf_ordinals = viable_leaf_ordinals(
        column_ordinal,
        segment_ordinal,
        partition.as_ref(),
        prepared_leaves.as_ref(),
        &leaf_ordinals,
    )?;
    if viable_leaf_ordinals.is_empty() {
        return Ok(None);
    }

    let documents = partition.docs.modern().cloned().ok_or_else(|| {
        Error::internal("cross-column FTS source changed from modern to legacy documents")
    })?;
    let visibility = source_visibility(&documents, mask).await?;
    if visibility.is_empty() {
        return Ok(None);
    }
    let leaves = load_source_leaves(
        partition,
        segment_ordinal,
        viable_leaf_ordinals,
        prepared_leaves,
        metrics,
    )
    .await?;
    if leaves.is_empty() {
        return Ok(None);
    }
    Ok(Some(LoadedGeneratorSource {
        documents,
        visibility,
        leaves,
    }))
}

async fn load_masked_cross_column_source(
    descriptor: SourceDescriptor,
    prepared_leaves: Arc<Vec<PreparedCrossColumnLeaf>>,
    leaves_by_column: Arc<Vec<Vec<usize>>>,
    mask: Arc<RowAddrMask>,
    metrics: Arc<dyn MetricsCollector>,
) -> Result<Option<LoadedCrossColumnSource>> {
    let leaf_ordinals = leaves_by_column
        .get(descriptor.column_ordinal)
        .cloned()
        .ok_or_else(|| Error::internal("cross-column FTS source references a missing column"))?;
    load_masked_cross_column_source_for_leaves(
        descriptor,
        prepared_leaves,
        leaf_ordinals,
        mask,
        metrics,
    )
    .await
}

async fn load_candidate_cross_column_source(
    descriptor: SourceDescriptor,
    prepared_leaves: Arc<Vec<PreparedCrossColumnLeaf>>,
    leaves_by_column: Arc<Vec<Vec<usize>>>,
    candidates: Arc<Vec<u64>>,
    metrics: Arc<dyn MetricsCollector>,
) -> Result<Option<LoadedCrossColumnSource>> {
    let SourceDescriptor {
        column_ordinal,
        segment_ordinal,
        partition,
    } = descriptor;
    let leaf_ordinals = leaves_by_column
        .get(column_ordinal)
        .ok_or_else(|| Error::internal("cross-column FTS source references a missing column"))?;
    let viable_leaf_ordinals = viable_leaf_ordinals(
        column_ordinal,
        segment_ordinal,
        partition.as_ref(),
        prepared_leaves.as_ref(),
        leaf_ordinals,
    )?;
    if viable_leaf_ordinals.is_empty() {
        return Ok(None);
    }

    let documents = partition.docs.modern().cloned().ok_or_else(|| {
        Error::internal("cross-column FTS source changed from modern to legacy documents")
    })?;
    let projection = documents.address_projection().await?;
    let selected = projection
        .select_sorted_addresses(candidates.as_slice())
        .await?;
    let candidate_doc_ids = selected.iter().map(DocId::new).collect::<Vec<_>>();
    let visibility = DocVisibility::Selected(selected);
    if visibility.is_empty() {
        return Ok(None);
    }

    // Candidate projection is exact in this source's local DocId domain. Only
    // a non-empty intersection is allowed to trigger posting or length I/O.
    let leaves = load_source_leaves(
        partition,
        segment_ordinal,
        viable_leaf_ordinals,
        prepared_leaves,
        metrics,
    )
    .await?;
    if leaves.is_empty() {
        return Ok(None);
    }
    let lengths = match documents.cached_lengths() {
        Some(lengths) => LoadedScoringLengths::Dense(lengths),
        None if documents.prefer_sparse_document_read(candidate_doc_ids.len()) => {
            let resolved = documents
                .resolve_scoring_documents(&candidate_doc_ids)
                .await?
                .into_iter()
                .map(
                    |(doc_id, row_address, scoring_length)| ResolvedCandidateDocument {
                        doc_id,
                        row_address,
                        scoring_length,
                    },
                )
                .collect();
            LoadedScoringLengths::Sparse(resolved)
        }
        None => LoadedScoringLengths::Dense(documents.lengths().await?),
    };

    Ok(Some(LoadedCrossColumnSource {
        num_docs: documents.len(),
        lengths,
        visibility,
        projection,
        leaves,
    }))
}

struct StagedGeneratorCandidates {
    addresses: Vec<u64>,
    materialized_leaves: Vec<(usize, Vec<ScoredRow>)>,
}

struct LocalGeneratorLeaf {
    leaf_ordinal: usize,
    postings: Vec<PostingIterator>,
    params: Arc<FtsSearchParams>,
    operator: Operator,
    scorer: Arc<MemBM25Scorer>,
    candidate_docs: RoaringBitmap,
}

struct LocalGeneratorCandidates {
    documents: Arc<PartitionDocuments>,
    leaves: Vec<LocalGeneratorLeaf>,
    candidate_docs: Vec<DocId>,
}

struct ResolvedGeneratorCandidates {
    num_docs: usize,
    leaves: Vec<LocalGeneratorLeaf>,
    /// Sorted by local DocId.
    documents: Vec<ResolvedCandidateDocument>,
}

fn collect_local_generator_candidates(
    sources: Vec<LoadedGeneratorSource>,
    generator_leaf_ordinals: Vec<usize>,
    max_candidates: usize,
    metrics: Arc<dyn MetricsCollector>,
) -> Result<Option<Vec<LocalGeneratorCandidates>>> {
    let mut materialized_row_count = 0_usize;
    let generator_leaf_set = generator_leaf_ordinals
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let mut collected = Vec::with_capacity(sources.len());
    for source in sources {
        let documents = StagedWandDocuments {
            num_docs: source.documents.len(),
            visibility: &source.visibility,
            scoring_lengths: ScoringLengths::Missing,
        };
        let mut candidate_docs = RoaringBitmap::new();
        let mut collected_leaves = Vec::with_capacity(source.leaves.len());
        for leaf in source.leaves {
            if !generator_leaf_set.contains(&leaf.leaf_ordinal) {
                return Err(Error::internal(format!(
                    "staged cross-column FTS loaded non-generator leaf {}",
                    leaf.leaf_ordinal
                )));
            }
            let Some(local_document_lower_bound) =
                local_candidate_lower_bound(leaf.operator, &leaf.postings)
            else {
                continue;
            };
            let score_postings = leaf
                .postings
                .iter()
                .map(PostingIterator::fork_from_start)
                .collect::<Vec<_>>();
            let mut local_scorer = WandCursor::new(
                leaf.operator,
                leaf.postings,
                &documents,
                leaf.scorer.clone(),
                leaf.params.as_ref(),
                metrics.as_ref(),
            );
            let mut leaf_candidate_docs = RoaringBitmap::new();
            let mut candidate = local_scorer.advance(local_document_lower_bound)?;
            while let Some(local_doc) = candidate {
                if local_scorer.matches()? {
                    let local_doc = u32::try_from(local_doc).map_err(|_| {
                        Error::index(format!(
                            "staged cross-column FTS local document {local_doc} exceeds the modern u32 domain"
                        ))
                    })?;
                    if leaf_candidate_docs.insert(local_doc) {
                        materialized_row_count = materialized_row_count.saturating_add(1);
                    }
                    candidate_docs.insert(local_doc);
                    if materialized_row_count > max_candidates {
                        // A partial candidate set is never used. Returning
                        // None makes the async caller run the complete eager
                        // execution path.
                        return Ok(None);
                    }
                }
                candidate = local_scorer.next()?;
            }
            if !leaf_candidate_docs.is_empty() {
                collected_leaves.push(LocalGeneratorLeaf {
                    leaf_ordinal: leaf.leaf_ordinal,
                    postings: score_postings,
                    params: leaf.params,
                    operator: leaf.operator,
                    scorer: leaf.scorer,
                    candidate_docs: leaf_candidate_docs,
                });
            }
        }
        if !candidate_docs.is_empty() {
            collected.push(LocalGeneratorCandidates {
                documents: source.documents,
                leaves: collected_leaves,
                candidate_docs: candidate_docs.iter().map(DocId::new).collect(),
            });
        }
    }
    Ok(Some(collected))
}

async fn resolve_generator_candidates(
    source: LocalGeneratorCandidates,
) -> Result<ResolvedGeneratorCandidates> {
    let documents = source
        .documents
        .resolve_scoring_documents(&source.candidate_docs)
        .await?
        .into_iter()
        .map(
            |(doc_id, row_address, scoring_length)| ResolvedCandidateDocument {
                doc_id,
                row_address,
                scoring_length,
            },
        )
        .collect();
    Ok(ResolvedGeneratorCandidates {
        num_docs: source.documents.len(),
        leaves: source.leaves,
        documents,
    })
}

fn score_generator_candidates(
    sources: Vec<ResolvedGeneratorCandidates>,
    generator_leaf_ordinals: Vec<usize>,
    max_candidates: usize,
    metrics: Arc<dyn MetricsCollector>,
) -> Result<Option<StagedGeneratorCandidates>> {
    let mut addresses = RoaringTreemap::new();
    let mut materialized_row_count = 0_usize;
    let mut scored_rows_by_leaf = generator_leaf_ordinals
        .iter()
        .map(|&leaf_ordinal| {
            (
                leaf_ordinal,
                (
                    RoaringTreemap::new(),
                    Vec::with_capacity(max_candidates.min(MIN_STAGED_GENERATOR_CANDIDATES)),
                ),
            )
        })
        .collect::<HashMap<_, _>>();

    for source in sources {
        let mut source_address_owners = HashMap::<u64, u32>::new();
        for leaf in source.leaves {
            let visibility = DocVisibility::Selected(leaf.candidate_docs.clone());
            let documents = StagedWandDocuments {
                num_docs: source.num_docs,
                visibility: &visibility,
                scoring_lengths: ScoringLengths::Sparse(&source.documents),
            };
            let expected_matches = leaf.candidate_docs.len();
            let mut actual_matches = 0_u64;
            let mut local_scorer = WandCursor::new(
                leaf.operator,
                leaf.postings,
                &documents,
                leaf.scorer,
                leaf.params.as_ref(),
                metrics.as_ref(),
            );
            for local_doc in leaf.candidate_docs.iter() {
                let positioned = local_scorer.advance(u64::from(local_doc))?;
                if positioned != Some(u64::from(local_doc)) || !local_scorer.matches()? {
                    return Err(Error::internal(format!(
                        "staged cross-column FTS could not reproduce generator leaf {} candidate {local_doc}",
                        leaf.leaf_ordinal
                    )));
                }
                let row_address = documents.row_address(local_doc).ok_or_else(|| {
                    Error::internal(format!(
                        "staged cross-column FTS did not resolve generator local document {local_doc}"
                    ))
                })?;
                if let Some(existing_doc) = source_address_owners.insert(row_address, local_doc)
                    && existing_doc != local_doc
                {
                    return Err(Error::index(format!(
                        "invalid FTS row-address projection: row address {row_address} is shared by local documents {existing_doc} and {local_doc}"
                    )));
                }
                let (leaf_addresses, scored_rows) = scored_rows_by_leaf
                    .get_mut(&leaf.leaf_ordinal)
                    .ok_or_else(|| {
                        Error::internal(format!(
                            "staged cross-column FTS lost generator leaf {}",
                            leaf.leaf_ordinal
                        ))
                    })?;
                if !leaf_addresses.insert(row_address) {
                    return Err(Error::internal(format!(
                        "cross-column FTS generator leaf {} produced duplicate row address {row_address}",
                        leaf.leaf_ordinal
                    )));
                }
                scored_rows.push(ScoredRow {
                    row_id: row_address,
                    score: local_scorer.score()?,
                });
                materialized_row_count = materialized_row_count.saturating_add(1);
                addresses.insert(row_address);
                actual_matches += 1;
                if addresses.len() > max_candidates as u64
                    || materialized_row_count > max_candidates
                {
                    return Ok(None);
                }
            }
            if actual_matches != expected_matches {
                return Err(Error::internal(format!(
                    "staged cross-column FTS rescored {actual_matches} of {expected_matches} candidates for generator leaf {}",
                    leaf.leaf_ordinal
                )));
            }
        }
    }
    let mut materialized_leaves = scored_rows_by_leaf
        .into_iter()
        .map(|(leaf_ordinal, (_, mut rows))| {
            rows.sort_unstable_by_key(|row| row.row_id);
            (leaf_ordinal, rows)
        })
        .collect::<Vec<_>>();
    materialized_leaves.sort_unstable_by_key(|(leaf_ordinal, _)| *leaf_ordinal);
    Ok(Some(StagedGeneratorCandidates {
        addresses: addresses.into_iter().collect(),
        materialized_leaves,
    }))
}

struct SourceDocuments {
    num_docs: usize,
    lengths: LoadedScoringLengths,
    visibility: DocVisibility,
    projection: ResidentAddressProjection,
}

fn score_cross_column_sources(
    sources: Vec<LoadedCrossColumnSource>,
    plan: CompoundScorerPlan,
    plan_bounds: ScoreBounds,
    num_leaves: usize,
    materialized_leaves: Vec<(usize, Vec<ScoredRow>)>,
    limit: usize,
    metrics: Arc<dyn MetricsCollector>,
) -> Result<(Vec<u64>, Vec<f32>)> {
    let mut source_documents = Vec::with_capacity(sources.len());
    let mut source_leaves = Vec::with_capacity(sources.len());
    for source in sources {
        source_documents.push(SourceDocuments {
            num_docs: source.num_docs,
            lengths: source.lengths,
            visibility: source.visibility,
            projection: source.projection,
        });
        source_leaves.push(source.leaves);
    }
    let wand_documents = source_documents
        .iter()
        .map(|source| StagedWandDocuments {
            num_docs: source.num_docs,
            visibility: &source.visibility,
            scoring_lengths: match &source.lengths {
                LoadedScoringLengths::Dense(lengths) => ScoringLengths::Dense(lengths.as_ref()),
                LoadedScoringLengths::Sparse(documents) => {
                    ScoringLengths::Sparse(documents.as_slice())
                }
            },
        })
        .collect::<Vec<_>>();
    let address_projections = source_documents
        .iter()
        .map(|source| prepare_row_address_projection(&source.projection))
        .collect::<Vec<_>>();

    let mut sources_by_leaf = (0..num_leaves)
        .map(|_| Vec::<RowAddressSource<'_>>::new())
        .collect::<Vec<_>>();
    for (source_ordinal, leaves) in source_leaves.into_iter().enumerate() {
        let mut local_scorers = Vec::with_capacity(leaves.len());
        for leaf in leaves {
            if leaf.postings.is_empty() {
                continue;
            }
            let Some(local_document_lower_bound) =
                local_candidate_lower_bound(leaf.operator, &leaf.postings)
            else {
                continue;
            };
            let local_scorer: BoxScorer<'_> = Box::new(WandCursor::new(
                leaf.operator,
                leaf.postings,
                &wand_documents[source_ordinal],
                leaf.scorer,
                leaf.params.as_ref(),
                metrics.as_ref(),
            ));
            let cost = local_scorer.cost();
            local_scorers.push((
                leaf.leaf_ordinal,
                cost,
                local_document_lower_bound,
                local_scorer,
            ));
        }

        // A dense leaf validates an unknown projection once and caches the
        // ordered result for sparse siblings. Mapping high-cost leaves first
        // therefore avoids materializing sparse leaves before that validation.
        local_scorers.sort_unstable_by(|left, right| {
            compare_leaf_mapping_priority(left.0, left.1, right.0, right.1)
        });
        for (leaf_ordinal, _, local_document_lower_bound, local_scorer) in local_scorers {
            let Some(address_source) = map_scorer_to_row_addresses(
                local_scorer,
                &address_projections[source_ordinal],
                local_document_lower_bound,
            )?
            else {
                continue;
            };
            sources_by_leaf
                .get_mut(leaf_ordinal)
                .ok_or_else(|| {
                    Error::internal(format!(
                        "cross-column FTS loaded unexpected leaf {}",
                        leaf_ordinal
                    ))
                })?
                .push(address_source);
        }
    }
    // Materialized fallbacks retain a source-wide collision map only while
    // sibling leaves are being mapped. Scorers no longer borrow this state.
    drop(address_projections);

    let mut materialized_leaf_ordinals = vec![false; num_leaves];
    for (leaf_ordinal, _) in &materialized_leaves {
        let materialized = materialized_leaf_ordinals
            .get_mut(*leaf_ordinal)
            .ok_or_else(|| {
                Error::internal(format!(
                    "staged cross-column FTS materialized unexpected leaf {leaf_ordinal}"
                ))
            })?;
        if std::mem::replace(materialized, true) {
            return Err(Error::internal(format!(
                "staged cross-column FTS materialized leaf {leaf_ordinal} more than once"
            )));
        }
    }
    let mut leaf_scorers = sources_by_leaf
        .into_iter()
        .enumerate()
        .map(|(leaf_ordinal, mut sources)| {
            if materialized_leaf_ordinals[leaf_ordinal] {
                if !sources.is_empty() {
                    return Err(Error::internal(format!(
                        "staged cross-column FTS loaded materialized generator leaf {leaf_ordinal} twice"
                    )));
                }
                return Ok(None);
            }
            let scorer: BoxScorer<'_> = match sources.len() {
                0 => Box::new(EmptyScorer),
                1 => sources
                    .pop()
                    .ok_or_else(|| Error::internal("cross-column FTS lost its only leaf source"))?
                    .into_scorer(),
                _ => Box::new(RowAddressMergeScorer::try_new(sources)?),
            };
            Ok(Some(scorer))
        })
        .collect::<Result<Vec<_>>>()?;
    for (leaf_ordinal, rows) in materialized_leaves {
        let slot = leaf_scorers.get_mut(leaf_ordinal).ok_or_else(|| {
            Error::internal(format!(
                "staged cross-column FTS materialized unexpected leaf {leaf_ordinal}"
            ))
        })?;
        debug_assert!(slot.is_none());
        *slot = Some(Box::new(MaterializedScorer::try_new(rows)?));
    }
    let mut scorer = plan.build(&mut leaf_scorers)?;
    if leaf_scorers.iter().any(Option::is_some) {
        return Err(Error::internal(
            "cross-column compound FTS scorer did not consume every prepared leaf",
        ));
    }
    let rows = TopKCollector::new(limit).collect(scorer.as_mut())?;
    if let Some(row) = rows.iter().find(|row| !plan_bounds.contains(row.score)) {
        return Err(Error::internal(format!(
            "cross-column compound FTS score {} for row address {} escaped plan bounds {plan_bounds:?}",
            row.score, row.row_id
        )));
    }
    Ok(rows.into_iter().map(|row| (row.row_id, row.score)).unzip())
}

/// Internal cross-crate hook for a bounded compound FTS query over multiple
/// indexed columns.
///
/// Each leaf is scored with corpus statistics from its own column.  Partition
/// scorers are mapped to current row addresses and heap-merged before the
/// Boolean/Boost/MultiMatch tree is composed, so one exact global collector
/// owns both the competitive score and final row-address tie breaking.
///
/// Every Match and Phrase leaf must omit document granularity or request
/// [`DocumentGranularity::Row`]. `columns` must contain at least two modern,
/// row-document indices and every supplied column must be referenced. `params`
/// must specify a bounded result limit. Invalid query or index shapes return
/// [`Error::InvalidInput`]; list-element requests are rejected before any
/// prefilter or index work begins.
#[doc(hidden)]
pub async fn cross_column_compound_search(
    columns: &[(String, Vec<Arc<InvertedIndex>>)],
    query: &FtsQuery,
    params: &FtsSearchParams,
    prefilter: Arc<dyn PreFilter>,
    metrics: Arc<dyn MetricsCollector>,
) -> Result<(Vec<u64>, Vec<f32>)> {
    let mut leaf_queries = Vec::new();
    collect_leaf_queries(query, &mut leaf_queries)?;
    validate_row_leaf_granularities(&leaf_queries)?;

    let limit = params.limit.ok_or_else(|| {
        Error::invalid_input("cross-column compound FTS requires a bounded result limit")
    })?;
    if limit == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let column_names = columns
        .iter()
        .map(|(column, _)| column.clone())
        .collect::<Vec<_>>();
    let leaf_columns = resolve_leaf_columns(&column_names, &leaf_queries)?;
    validate_modern_row_indices(columns)?;

    let mut num_plan_leaves = 0;
    let plan = CompoundScorerPlan::from_query(query, &mut num_plan_leaves)?;
    if num_plan_leaves != leaf_queries.len() {
        return Err(Error::internal(format!(
            "cross-column compound FTS planned {num_plan_leaves} leaves but prepared {}",
            leaf_queries.len()
        )));
    }

    // The prefilter owns potentially asynchronous deletion/filter work. It is
    // shared by every column but reaches readiness exactly once here. An empty
    // filter avoids all token-statistics and posting I/O.
    prefilter.wait_for_ready().await?;
    let mask = prefilter.mask();
    if mask.max_len() == Some(0) {
        return Ok((Vec::new(), Vec::new()));
    }

    let mut queries_by_column = (0..columns.len())
        .map(|_| Vec::<(usize, LeafQuery)>::new())
        .collect::<Vec<_>>();
    for (leaf_ordinal, (leaf, column_ordinal)) in
        leaf_queries.into_iter().zip(leaf_columns).enumerate()
    {
        queries_by_column[column_ordinal].push((leaf_ordinal, leaf));
    }

    // Own the work items before building the stream. Besides keeping this
    // future `Send`, it prevents borrowed iterator closure types from leaking
    // into callers that box their execution stream.
    let preparation_work = columns
        .iter()
        .enumerate()
        .map(|(column_ordinal, (_, indices))| {
            (
                column_ordinal,
                indices.clone(),
                queries_by_column[column_ordinal].clone(),
            )
        })
        .collect::<Vec<_>>();
    let preparation_parallelism = get_num_compute_intensive_cpus()
        .clamp(1, MAX_CONCURRENT_SOURCE_LOADS)
        .min(preparation_work.len());
    let prepared_by_column = stream::iter(preparation_work.into_iter().map(
        |(column_ordinal, indices, leaf_queries)| {
            let params = params.clone();
            let metrics = metrics.clone();
            async move {
                prepare_column_leaves(column_ordinal, &indices, &leaf_queries, &params, metrics)
                    .await
            }
        },
    ))
    .buffer_unordered(preparation_parallelism)
    .try_collect::<Vec<_>>()
    .await?;
    let mut prepared_leaves = (0..num_plan_leaves)
        .map(|_| None)
        .collect::<Vec<Option<PreparedCrossColumnLeaf>>>();
    for column_leaves in prepared_by_column {
        for (leaf_ordinal, leaf) in column_leaves {
            let slot = prepared_leaves.get_mut(leaf_ordinal).ok_or_else(|| {
                Error::internal(format!(
                    "cross-column FTS prepared unexpected leaf {leaf_ordinal}"
                ))
            })?;
            if slot.replace(leaf).is_some() {
                return Err(Error::internal(format!(
                    "cross-column FTS prepared leaf {leaf_ordinal} more than once"
                )));
            }
        }
    }
    let prepared_leaves = prepared_leaves
        .into_iter()
        .enumerate()
        .map(|(leaf_ordinal, leaf)| {
            leaf.ok_or_else(|| {
                Error::internal(format!(
                    "cross-column FTS did not prepare leaf {leaf_ordinal}"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let plan_inputs = prepared_leaves
        .iter()
        .map(leaf_plan_input)
        .collect::<Result<Vec<_>>>()?;
    let plan_analysis = plan.analyze_leaves(&plan_inputs)?;
    if !plan_analysis.possible {
        return Ok((Vec::new(), Vec::new()));
    }
    // Staging trades a second bounded CPU pass for deferred cold I/O. Once an
    // explicit prewarm has made every selected index query-ready, that trade is
    // strictly worse: the original bounded coordinator can consume resident
    // postings and document columns directly. The hint is checked without I/O;
    // any mixed or uncertain state conservatively keeps the staged cold path.
    let staged_generator = (!query_state_is_prewarmed(columns, &prepared_leaves))
        .then(|| staged_generator(&plan_analysis, &plan_inputs, &prepared_leaves, limit))
        .flatten();
    let leaves_by_column = prepared_leaves.iter().enumerate().fold(
        (0..columns.len()).map(|_| Vec::new()).collect::<Vec<_>>(),
        |mut by_column, (leaf_ordinal, leaf)| {
            by_column[leaf.column_ordinal].push(leaf_ordinal);
            by_column
        },
    );

    let prepared_leaves = Arc::new(prepared_leaves);
    let leaves_by_column = Arc::new(leaves_by_column);
    let descriptors = columns
        .iter()
        .enumerate()
        .flat_map(|(column_ordinal, (_, indices))| {
            indices
                .iter()
                .enumerate()
                .flat_map(move |(segment_ordinal, index)| {
                    index
                        .partitions
                        .iter()
                        .cloned()
                        .map(move |partition| SourceDescriptor {
                            column_ordinal,
                            segment_ordinal,
                            partition,
                        })
                })
        })
        .collect::<Vec<_>>();
    let parallelism = get_num_compute_intensive_cpus().clamp(1, MAX_CONCURRENT_SOURCE_LOADS);
    let staged_candidates = if let Some((generator_leaf_ordinals, candidate_budget)) =
        staged_generator
    {
        let generator_leaf_set = generator_leaf_ordinals
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let generator_leaves_by_column = Arc::new(
            leaves_by_column
                .iter()
                .map(|leaves| {
                    leaves
                        .iter()
                        .copied()
                        .filter(|leaf| generator_leaf_set.contains(leaf))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        );
        let generator_descriptors = descriptors
            .iter()
            .filter(|descriptor| {
                generator_leaves_by_column
                    .get(descriptor.column_ordinal)
                    .is_some_and(|leaves| !leaves.is_empty())
            })
            .cloned()
            .collect::<Vec<_>>();
        let generator_sources = stream::iter(generator_descriptors.into_iter().map(|descriptor| {
            let leaf_ordinals = generator_leaves_by_column
                .get(descriptor.column_ordinal)
                .cloned()
                .unwrap_or_default();
            load_masked_generator_source(
                descriptor,
                prepared_leaves.clone(),
                leaf_ordinals,
                mask.clone(),
                metrics.clone(),
            )
        }))
        .buffer_unordered(parallelism)
        .try_collect::<Vec<_>>()
        .await?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        let candidate_metrics = metrics.clone();
        let collection_leaf_ordinals = generator_leaf_ordinals.clone();
        let local_candidates = spawn_cpu(move || {
            collect_local_generator_candidates(
                generator_sources,
                collection_leaf_ordinals,
                candidate_budget,
                candidate_metrics,
            )
        })
        .await?;
        if let Some(local_candidates) = local_candidates {
            let resolved = stream::iter(
                local_candidates
                    .into_iter()
                    .map(resolve_generator_candidates),
            )
            .buffer_unordered(parallelism)
            .try_collect::<Vec<_>>()
            .await?;
            let candidate_metrics = metrics.clone();
            spawn_cpu(move || {
                score_generator_candidates(
                    resolved,
                    generator_leaf_ordinals,
                    candidate_budget,
                    candidate_metrics,
                )
            })
            .await?
        } else {
            None
        }
    } else {
        None
    };

    let (sources, materialized_leaves) = if let Some(candidates) = staged_candidates {
        if candidates.addresses.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let StagedGeneratorCandidates {
            addresses,
            materialized_leaves,
        } = candidates;
        let generator_leaf_set = materialized_leaves
            .iter()
            .map(|(leaf_ordinal, _)| *leaf_ordinal)
            .collect::<HashSet<_>>();
        let candidates = Arc::new(addresses);
        let deferred_leaves_by_column = Arc::new(
            leaves_by_column
                .iter()
                .map(|leaves| {
                    leaves
                        .iter()
                        .copied()
                        .filter(|leaf| !generator_leaf_set.contains(leaf))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        );
        let sources = stream::iter(descriptors.into_iter().map(|descriptor| {
            load_candidate_cross_column_source(
                descriptor,
                prepared_leaves.clone(),
                deferred_leaves_by_column.clone(),
                candidates.clone(),
                metrics.clone(),
            )
        }))
        .buffer_unordered(parallelism)
        .try_collect::<Vec<_>>()
        .await?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        (sources, materialized_leaves)
    } else {
        let sources = stream::iter(descriptors.into_iter().map(|descriptor| {
            load_masked_cross_column_source(
                descriptor,
                prepared_leaves.clone(),
                leaves_by_column.clone(),
                mask.clone(),
                metrics.clone(),
            )
        }))
        .buffer_unordered(parallelism)
        .try_collect::<Vec<_>>()
        .await?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        (sources, Vec::new())
    };

    spawn_cpu(move || {
        score_cross_column_sources(
            sources,
            plan,
            plan_analysis.bounds,
            num_plan_leaves,
            materialized_leaves,
            limit,
            metrics,
        )
    })
    .await
}

#[cfg(test)]
mod tests {
    use arrow::buffer::ScalarBuffer;

    use super::*;
    use crate::metrics::NoOpMetricsCollector;
    use crate::prefilter::NoFilter;
    use crate::scalar::inverted::encoding::compress_posting_list;
    use crate::scalar::inverted::query::{
        BooleanQuery, MatchQuery, MultiMatchQuery, Occur, PhraseQuery, Tokens,
    };
    use crate::scalar::inverted::tokenizer::document_tokenizer::DocType;
    use crate::scalar::inverted::{
        CompressedPostingList, DocumentGranularity, LEGACY_BLOCK_SIZE, PlainPostingList,
        PostingList, PostingTailCodec,
    };

    fn match_query(column: Option<&str>, terms: &str) -> FtsQuery {
        FtsQuery::Match(
            MatchQuery::new(terms.to_owned()).with_column(column.map(ToOwned::to_owned)),
        )
    }

    fn posting(doc_ids: &[u64]) -> PostingIterator {
        if doc_ids.is_empty() {
            return PostingIterator::new(
                "term".to_owned(),
                0,
                0,
                PostingList::Plain(PlainPostingList::new(
                    ScalarBuffer::from(Vec::<u64>::new()),
                    ScalarBuffer::from(Vec::<f32>::new()),
                    Some(0.0),
                    None,
                )),
                100,
            );
        }
        let doc_ids = doc_ids
            .iter()
            .map(|&doc_id| u32::try_from(doc_id).unwrap())
            .collect::<Vec<_>>();
        let frequencies = vec![1_u32; doc_ids.len()];
        let blocks = compress_posting_list(
            doc_ids.len(),
            doc_ids.iter(),
            frequencies.iter(),
            vec![1.0; doc_ids.len()].into_iter(),
        )
        .unwrap();
        PostingIterator::new(
            "term".to_owned(),
            0,
            0,
            PostingList::Compressed(CompressedPostingList::new(
                blocks,
                1.0,
                doc_ids.len() as u32,
                PostingTailCodec::VarintDelta,
                LEGACY_BLOCK_SIZE,
                None,
                None,
            )),
            100,
        )
    }

    fn prepared_leaf(
        tokens_by_segment: Vec<Tokens>,
        operator: Operator,
        phrase_slop: Option<u32>,
        num_docs: usize,
        token_docs: impl IntoIterator<Item = (&'static str, usize)>,
    ) -> PreparedCrossColumnLeaf {
        let mut segment_tokens = tokens_by_segment.into_iter();
        let tokens = Arc::new(segment_tokens.next().unwrap());
        for segment_tokens in segment_tokens {
            assert_eq!(segment_tokens.len(), tokens.len());
            for index in 0..segment_tokens.len() {
                assert_eq!(segment_tokens.get_token(index), tokens.get_token(index));
                assert_eq!(segment_tokens.position(index), tokens.position(index));
            }
        }
        let scorer = Arc::new(MemBM25Scorer::new(
            num_docs as u64,
            num_docs,
            token_docs
                .into_iter()
                .map(|(token, count)| (token.to_owned(), count))
                .collect(),
        ));
        PreparedCrossColumnLeaf {
            column_ordinal: 0,
            query: Arc::new(PreparedBm25Query::from_parts(tokens, scorer, true)),
            params: Arc::new(FtsSearchParams::new().with_phrase_slop(phrase_slop)),
            operator,
        }
    }

    fn loaded_generator_source(
        leaf_ordinal: usize,
        addresses: Vec<u64>,
        matching_docs: &[u64],
    ) -> ResolvedGeneratorCandidates {
        let num_docs = addresses.len();
        let candidate_docs = matching_docs
            .iter()
            .map(|&doc_id| u32::try_from(doc_id).unwrap())
            .collect::<RoaringBitmap>();
        ResolvedGeneratorCandidates {
            num_docs,
            documents: candidate_docs
                .iter()
                .map(|doc_id| ResolvedCandidateDocument {
                    doc_id,
                    row_address: addresses[doc_id as usize],
                    scoring_length: 1,
                })
                .collect(),
            leaves: vec![LoadedCrossColumnLeaf {
                leaf_ordinal,
                postings: vec![posting(matching_docs)],
                params: Arc::new(FtsSearchParams::new()),
                operator: Operator::Or,
                scorer: Arc::new(MemBM25Scorer::new(
                    num_docs as u64,
                    num_docs,
                    HashMap::from([("term".to_owned(), matching_docs.len())]),
                )),
            }]
            .into_iter()
            .map(|leaf| LocalGeneratorLeaf {
                leaf_ordinal: leaf.leaf_ordinal,
                postings: leaf.postings,
                params: leaf.params,
                operator: leaf.operator,
                scorer: leaf.scorer,
                candidate_docs: candidate_docs.clone(),
            })
            .collect(),
        }
    }

    #[tokio::test]
    async fn rejects_list_element_leaves_before_column_or_index_validation() {
        let mut multi_match = MultiMatchQuery::try_new(
            "term".to_owned(),
            vec!["title".to_owned(), "body".to_owned()],
        )
        .unwrap();
        multi_match.match_queries[1].document_granularity = Some(DocumentGranularity::ListElement);
        let cases = [
            (
                "Match leaf 0 for column 'title'",
                FtsQuery::Match(
                    MatchQuery::new("term".to_owned())
                        .with_column(Some("title".to_owned()))
                        .with_document_granularity(DocumentGranularity::ListElement),
                ),
            ),
            (
                "Phrase leaf 0 for column 'body'",
                FtsQuery::Phrase(
                    PhraseQuery::new("two terms".to_owned())
                        .with_column(Some("body".to_owned()))
                        .with_document_granularity(DocumentGranularity::ListElement),
                ),
            ),
            (
                "Match leaf 1 for column 'body'",
                FtsQuery::MultiMatch(multi_match),
            ),
        ];

        for (leaf_context, query) in cases {
            let error = cross_column_compound_search(
                &[],
                &query,
                &FtsSearchParams::new().with_limit(Some(10)),
                Arc::new(NoFilter),
                Arc::new(NoOpMetricsCollector),
            )
            .await
            .unwrap_err();

            assert!(matches!(&error, Error::InvalidInput { .. }));
            let message = error.to_string();
            assert!(
                message.contains(leaf_context),
                "unexpected error: {message}"
            );
            assert!(
                message.contains(
                    "requested ListElement document granularity, but only Row is supported"
                ),
                "unexpected error: {message}"
            );
        }
    }

    #[tokio::test]
    async fn permits_unspecified_and_row_leaf_granularity() {
        for document_granularity in [None, Some(DocumentGranularity::Row)] {
            let mut match_query =
                MatchQuery::new("term".to_owned()).with_column(Some("title".to_owned()));
            match_query.document_granularity = document_granularity;
            let error = cross_column_compound_search(
                &[],
                &FtsQuery::Match(match_query),
                &FtsSearchParams::new().with_limit(Some(10)),
                Arc::new(NoFilter),
                Arc::new(NoOpMetricsCollector),
            )
            .await
            .unwrap_err();

            assert!(matches!(&error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains("requires at least two columns"));
        }
    }

    #[test]
    fn derives_conservative_local_candidate_lower_bound() {
        let postings = vec![posting(&[10, 20]), posting(&[5, 30])];
        assert_eq!(
            local_candidate_lower_bound(Operator::Or, &postings),
            Some(5)
        );
        assert_eq!(
            local_candidate_lower_bound(Operator::And, &postings),
            Some(10)
        );

        let postings_with_empty = vec![posting(&[10]), posting(&[])];
        assert_eq!(
            local_candidate_lower_bound(Operator::Or, &postings_with_empty),
            Some(10)
        );
        assert_eq!(
            local_candidate_lower_bound(Operator::And, &postings_with_empty),
            None
        );
    }

    #[test]
    fn maps_dense_leaves_before_sparse_siblings() {
        let mut leaves = vec![(3, 10), (2, 100), (1, 10), (0, 100)];
        leaves.sort_unstable_by(|left, right| {
            compare_leaf_mapping_priority(left.0, left.1, right.0, right.1)
        });

        assert_eq!(leaves, vec![(0, 100), (2, 100), (1, 10), (3, 10)]);
    }

    #[test]
    fn estimates_generator_cost_by_unique_query_positions() {
        let tokens = Tokens::with_positions(
            vec![
                "rare".to_owned(),
                "rare_alt".to_owned(),
                "common".to_owned(),
            ],
            vec![0, 0, 1],
            DocType::Text,
        );
        let leaf = prepared_leaf(
            vec![tokens.clone(), tokens],
            Operator::And,
            None,
            10_000,
            [("rare", 7), ("rare_alt", 3), ("common", 1_000)],
        );
        let input = leaf_plan_input(&leaf).unwrap();
        assert!(input.possible);
        assert_eq!(input.cost, 10);
        assert_eq!(input.bounds.lower(), 0.0);
        assert_eq!(input.bounds.upper(), f32::INFINITY);

        let missing = prepared_leaf(
            vec![Tokens::with_positions(
                vec!["rare".to_owned(), "missing".to_owned()],
                vec![0, 1],
                DocType::Text,
            )],
            Operator::And,
            None,
            10_000,
            [("rare", 7), ("missing", 0)],
        );
        assert!(!leaf_plan_input(&missing).unwrap().possible);
    }

    #[test]
    fn stages_only_selective_single_leaf_generators_with_probe_work() {
        let leaves = vec![
            prepared_leaf(
                vec![Tokens::new(vec!["rare".to_owned()], DocType::Text)],
                Operator::Or,
                None,
                10_000,
                [("rare", 100)],
            ),
            prepared_leaf(
                vec![Tokens::new(vec!["optional".to_owned()], DocType::Text)],
                Operator::Or,
                None,
                10_000,
                [("optional", 2_000)],
            ),
            prepared_leaf(
                vec![Tokens::new(vec!["probe".to_owned()], DocType::Text)],
                Operator::Or,
                None,
                10_000,
                [("probe", 500)],
            ),
        ];
        let inputs = leaves
            .iter()
            .map(leaf_plan_input)
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let analysis = CompoundPlanAnalysis {
            possible: true,
            bounds: ScoreBounds::UNBOUNDED,
            generator_cost: 100,
            generator_leaves: vec![0],
        };
        assert_eq!(
            staged_generator(&analysis, &inputs, &leaves, 10),
            Some((vec![0], 128))
        );
        let no_probe_inputs = vec![inputs[0]];
        assert_eq!(
            staged_generator(&analysis, &no_probe_inputs, &leaves[..1], 10),
            None
        );

        let multi_generator = CompoundPlanAnalysis {
            generator_cost: 100,
            generator_leaves: vec![0, 1],
            ..analysis
        };
        assert_eq!(
            staged_generator(&multi_generator, &inputs, &leaves, 10),
            Some((vec![0, 1], 128))
        );

        let too_dense = CompoundPlanAnalysis {
            generator_cost: 129,
            ..multi_generator
        };
        assert_eq!(staged_generator(&too_dense, &inputs, &leaves, 10), None);
    }

    #[test]
    fn generator_collection_is_complete_and_overflow_falls_back() {
        let make_sources = || {
            vec![
                loaded_generator_source(1, vec![10, 20, 30], &[0, 2]),
                loaded_generator_source(1, vec![40, 50], &[1]),
            ]
        };
        let metrics: Arc<dyn MetricsCollector> = Arc::new(NoOpMetricsCollector);
        let candidates = score_generator_candidates(make_sources(), vec![1], 3, metrics.clone())
            .unwrap()
            .unwrap();
        assert_eq!(candidates.addresses, vec![10, 30, 50]);
        assert_eq!(candidates.materialized_leaves.len(), 1);
        let (leaf_ordinal, scored_rows) = &candidates.materialized_leaves[0];
        assert_eq!(*leaf_ordinal, 1);
        assert_eq!(
            scored_rows.iter().map(|row| row.row_id).collect::<Vec<_>>(),
            vec![10, 30, 50]
        );
        assert!(
            scored_rows
                .iter()
                .all(|row| row.score.is_finite() && row.score > 0.0)
        );

        assert!(
            score_generator_candidates(make_sources(), vec![1], 2, metrics)
                .unwrap()
                .is_none(),
            "overflow must abandon the entire staged candidate set"
        );
    }

    #[test]
    fn generator_collection_materializes_every_leaf_in_a_union_cover() {
        let sources = vec![
            loaded_generator_source(0, vec![10, 20], &[0]),
            loaded_generator_source(1, vec![10, 20], &[0, 1]),
        ];
        let metrics: Arc<dyn MetricsCollector> = Arc::new(NoOpMetricsCollector);
        let candidates = score_generator_candidates(sources, vec![0, 1], 3, metrics)
            .unwrap()
            .unwrap();

        assert_eq!(candidates.addresses, vec![10, 20]);
        assert_eq!(
            candidates
                .materialized_leaves
                .iter()
                .map(|(leaf, rows)| {
                    (*leaf, rows.iter().map(|row| row.row_id).collect::<Vec<_>>())
                })
                .collect::<Vec<_>>(),
            vec![(0, vec![10]), (1, vec![10, 20])]
        );
    }

    #[test]
    fn generator_collection_rejects_same_leaf_duplicates_across_sources() {
        let sources = vec![
            loaded_generator_source(0, vec![10], &[0]),
            loaded_generator_source(0, vec![10], &[0]),
        ];
        let metrics: Arc<dyn MetricsCollector> = Arc::new(NoOpMetricsCollector);

        let Err(error) = score_generator_candidates(sources, vec![0], 2, metrics) else {
            panic!("duplicate row addresses should be rejected");
        };
        assert!(
            error
                .to_string()
                .contains("generator leaf 0 produced duplicate row address 10")
        );
    }

    #[test]
    fn resolves_leaf_columns_in_compound_plan_order() {
        let query = FtsQuery::Boolean(BooleanQuery::new([
            (Occur::Should, match_query(Some("body"), "optional")),
            (
                Occur::Must,
                FtsQuery::Phrase(
                    PhraseQuery::new("required phrase".to_owned())
                        .with_column(Some("title".to_owned())),
                ),
            ),
            (Occur::MustNot, match_query(Some("body"), "blocked")),
        ]));
        let mut leaves = Vec::new();
        collect_leaf_queries(&query, &mut leaves).unwrap();

        assert_eq!(
            resolve_leaf_columns(&["title".to_owned(), "body".to_owned()], &leaves).unwrap(),
            vec![1, 0, 1]
        );
        let mut num_plan_leaves = 0;
        CompoundScorerPlan::from_query(&query, &mut num_plan_leaves).unwrap();
        assert_eq!(num_plan_leaves, leaves.len());
    }

    #[test]
    fn rejects_missing_duplicate_and_unreferenced_columns() {
        let leaves = vec![LeafQuery::Match(
            MatchQuery::new("term".to_owned()).with_column(Some("title".to_owned())),
        )];

        let error = resolve_leaf_columns(&["title".to_owned()], &leaves).unwrap_err();
        assert!(error.to_string().contains("at least two columns"));

        let error =
            resolve_leaf_columns(&["title".to_owned(), "title".to_owned()], &leaves).unwrap_err();
        assert!(error.to_string().contains("duplicate column 'title'"));

        let error =
            resolve_leaf_columns(&["title".to_owned(), "body".to_owned()], &leaves).unwrap_err();
        assert!(error.to_string().contains("unreferenced columns: body"));
    }

    #[test]
    fn rejects_leaf_without_a_supplied_column_index() {
        let leaves = vec![LeafQuery::Match(MatchQuery::new("term".to_owned()))];
        let error =
            resolve_leaf_columns(&["title".to_owned(), "body".to_owned()], &leaves).unwrap_err();
        assert!(error.to_string().contains("leaf 0 is missing a column"));

        let leaves = vec![LeafQuery::Match(
            MatchQuery::new("term".to_owned()).with_column(Some("summary".to_owned())),
        )];
        let error =
            resolve_leaf_columns(&["title".to_owned(), "body".to_owned()], &leaves).unwrap_err();
        assert!(error.to_string().contains("no supplied index"));
    }
}
