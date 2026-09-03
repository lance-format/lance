// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

pub mod builder;
mod cache_codec;
mod compound;
mod cross_column;
mod documents;
mod encoding;
mod impact;
mod index;
mod iter;
pub mod json;
pub mod parser;
pub mod query;
mod scorer;
pub mod tokenizer;
mod wand;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, OnceLock};

use arrow_schema::{DataType, Field};
use async_trait::async_trait;
pub use builder::InvertedIndexBuilder;
pub use compound::{
    compound_search, compound_search_prepared_match,
    compound_search_prepared_match_with_score_floor, compound_search_with_base_scorer,
    compound_search_with_base_scorer_and_score_floor, exclusive_scaled_score_floor,
    materialized_compound_top_k,
};
#[doc(hidden)]
pub use cross_column::cross_column_compound_search;
use datafusion::execution::SendableRecordBatchStream;
pub use index::*;
use lance_core::{Result, cache::LanceCache};
pub use lance_tokenizer::Language;
pub use scorer::{MemBM25Scorer, Scorer};
pub use tokenizer::*;

use crate::scalar::inverted::query::{FtsSearchParams, Tokens, uses_fuzzy_expansion};

/// Canonical token vocabulary and BM25 statistics for one indexed query leaf.
///
/// Keeping these values together prevents a search path from expanding one
/// vocabulary while scoring another. Positions on `tokens` identify fuzzy
/// alternatives belonging to the same original query position.
#[doc(hidden)]
#[derive(Clone)]
pub struct PreparedBm25Query {
    tokens: Arc<Tokens>,
    scorer: Arc<MemBM25Scorer>,
    has_all_query_positions: bool,
    shared_norm_addends: OnceLock<Arc<QueryNormAddends>>,
}

/// Query-wide cache for the 256 document-norm denominator addends used by
/// quantized modern indexes. The key binds the cache to the immutable scorer
/// snapshot stored in [`PreparedBm25Query`].
pub(super) struct QueryNormAddends {
    scorer_key: u64,
    addends: OnceLock<Option<Arc<[f32; 256]>>>,
    #[cfg(test)]
    build_count: AtomicUsize,
}

impl QueryNormAddends {
    fn new(scorer_key: u64) -> Self {
        Self {
            scorer_key,
            addends: OnceLock::new(),
            #[cfg(test)]
            build_count: AtomicUsize::new(0),
        }
    }

    pub(super) fn get_or_init<S: Scorer>(&self, scorer: &S) -> Option<Arc<[f32; 256]>> {
        if scorer.doc_weight_cache_key() != Some(self.scorer_key) {
            return None;
        }
        self.addends
            .get_or_init(|| {
                #[cfg(test)]
                self.build_count.fetch_add(1, Ordering::Relaxed);
                let mut addends = [0.0; 256];
                for (code, addend) in addends.iter_mut().enumerate() {
                    *addend = scorer.doc_norm(dequantize_doc_length(code as u8))?;
                }
                Some(Arc::new(addends))
            })
            .clone()
    }

    #[cfg(test)]
    pub(super) fn build_count(&self) -> usize {
        self.build_count.load(Ordering::Relaxed)
    }
}

impl std::fmt::Debug for PreparedBm25Query {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedBm25Query")
            .field("token_count", &self.tokens.len())
            .field("has_all_query_positions", &self.has_all_query_positions)
            .field("scorer", &self.scorer)
            .finish()
    }
}

impl PreparedBm25Query {
    pub(crate) fn from_parts(
        tokens: Arc<Tokens>,
        scorer: Arc<MemBM25Scorer>,
        has_all_query_positions: bool,
    ) -> Self {
        Self {
            tokens,
            scorer,
            has_all_query_positions,
            shared_norm_addends: OnceLock::new(),
        }
    }

    #[doc(hidden)]
    pub fn tokens(&self) -> &Arc<Tokens> {
        &self.tokens
    }

    #[doc(hidden)]
    pub fn scorer(&self) -> &Arc<MemBM25Scorer> {
        &self.scorer
    }

    pub(crate) fn has_all_query_positions(&self) -> bool {
        self.has_all_query_positions
    }

    pub(super) fn should_share_norm_addends(
        &self,
        is_v3: bool,
        is_exact: bool,
        operator: query::Operator,
        is_phrase: bool,
    ) -> bool {
        *SHARED_NORM_LUT_ENABLED
            && is_v3
            && is_exact
            && operator == query::Operator::And
            && !is_phrase
            && matches!(self.tokens.len(), 2 | 3)
    }

    pub(super) fn shared_norm_addends(&self) -> Option<Arc<QueryNormAddends>> {
        self.shared_norm_addends_with_enabled(*SHARED_NORM_LUT_ENABLED)
    }

    fn shared_norm_addends_with_enabled(&self, enabled: bool) -> Option<Arc<QueryNormAddends>> {
        if !enabled {
            return None;
        }
        let scorer_key = self.scorer.doc_weight_cache_key()?;
        Some(
            self.shared_norm_addends
                .get_or_init(|| Arc::new(QueryNormAddends::new(scorer_key)))
                .clone(),
        )
    }
}

const LANCE_FTS_SHARED_NORM_LUT_ENV: &str = "LANCE_FTS_SHARED_NORM_LUT";

fn shared_norm_lut_enabled_from_value(value: Option<&str>) -> bool {
    !value.is_some_and(|value| {
        let value = value.trim();
        value == "0" || value.eq_ignore_ascii_case("off")
    })
}

static SHARED_NORM_LUT_ENABLED: LazyLock<bool> = LazyLock::new(|| {
    shared_norm_lut_enabled_from_value(std::env::var(LANCE_FTS_SHARED_NORM_LUT_ENV).ok().as_deref())
});

pub(crate) fn final_query_tokens(
    indices: &[Arc<InvertedIndex>],
    query_tokens: &Tokens,
    params: &FtsSearchParams,
) -> Result<Tokens> {
    if !uses_fuzzy_expansion(params.fuzziness) {
        return Ok(query_tokens.clone());
    }

    let initial_capacity = query_tokens.len().min(params.max_expansions);
    let mut expanded_tokens = Vec::with_capacity(initial_capacity);
    let mut expanded_positions = Vec::with_capacity(initial_capacity);
    let mut seen = HashSet::new();
    let mut source_terms_by_position = BTreeMap::<u32, Vec<&str>>::new();
    for token_idx in 0..query_tokens.len() {
        source_terms_by_position
            .entry(query_tokens.position(token_idx))
            .or_default()
            .push(query_tokens.get_token(token_idx));
    }
    for (position, source_terms) in source_terms_by_position {
        let remaining = params.max_expansions.saturating_sub(expanded_tokens.len());
        if remaining == 0 {
            break;
        }
        let mut candidates = BTreeSet::new();
        let mut seen_source_terms = HashSet::new();
        for source_term in source_terms {
            if !seen_source_terms.insert(source_term) {
                continue;
            }
            // One source token has one canonical automaton across every
            // selected segment. Drop it after this source term so peak DFA
            // memory is independent of the number of query terms.
            let automaton = FuzzyAutomaton::new(source_term, query_tokens.token_type(), params)?;
            for index in indices {
                index.collect_fuzzy_candidates_with_automaton(
                    &automaton,
                    remaining,
                    &mut candidates,
                )?;
            }
        }
        for candidate in candidates {
            if expanded_tokens.len() >= params.max_expansions {
                break;
            }
            if seen.insert((candidate.clone(), position)) {
                expanded_tokens.push(candidate);
                expanded_positions.push(position);
            }
        }
    }
    Ok(Tokens::with_positions(
        expanded_tokens,
        expanded_positions,
        query_tokens.token_type().clone(),
    ))
}

fn unique_terms(tokens: &Tokens) -> Vec<String> {
    let mut terms = Vec::with_capacity(tokens.len());
    let mut seen = HashSet::new();
    for token in tokens {
        if seen.insert(token.clone()) {
            terms.push(token.clone());
        }
    }
    terms
}

pub(crate) fn has_all_query_positions(query_tokens: &Tokens, final_tokens: &Tokens) -> bool {
    let surviving_positions = (0..final_tokens.len())
        .map(|index| final_tokens.position(index))
        .collect::<HashSet<_>>();
    (0..query_tokens.len()).all(|index| surviving_positions.contains(&query_tokens.position(index)))
}

const LANCE_FTS_SYNC_DF_ENV: &str = "LANCE_FTS_SYNC_DF";

fn sync_df_enabled_from_value(value: Option<&str>) -> bool {
    !value.is_some_and(|value| {
        let value = value.trim();
        value == "0" || value.eq_ignore_ascii_case("off")
    })
}

static LANCE_FTS_SYNC_DF_ENABLED: LazyLock<bool> = LazyLock::new(|| {
    sync_df_enabled_from_value(std::env::var(LANCE_FTS_SYNC_DF_ENV).ok().as_deref())
});

/// Build the global scorer without futures when a completed full prewarm made
/// every segment statistic and posting-length table synchronously available.
/// Returning `None` leaves the existing asynchronous path entirely in charge.
fn bm25_scorer_from_loaded_stats(
    indices: &[Arc<InvertedIndex>],
    terms: &[String],
) -> Result<Option<Arc<MemBM25Scorer>>> {
    bm25_scorer_from_loaded_stats_with_enabled(indices, terms, *LANCE_FTS_SYNC_DF_ENABLED)
}

fn bm25_scorer_from_loaded_stats_with_enabled(
    indices: &[Arc<InvertedIndex>],
    terms: &[String],
    is_enabled: bool,
) -> Result<Option<Arc<MemBM25Scorer>>> {
    // Keep the kill switch first so an ablation does not even probe prewarm
    // state or resident metadata.
    if !is_enabled {
        return Ok(None);
    }

    let mut loaded_stats = Vec::with_capacity(indices.len());
    for index in indices {
        let Some(stats) = index.bm25_stats_for_terms_if_loaded(terms)? else {
            return Ok(None);
        };
        loaded_stats.push(stats);
    }

    merge_loaded_bm25_stats(terms, loaded_stats)
}

fn merge_loaded_bm25_stats(
    terms: &[String],
    loaded_stats: Vec<(u64, usize, Vec<usize>)>,
) -> Result<Option<Arc<MemBM25Scorer>>> {
    let mut loaded_stats = loaded_stats.into_iter();
    let Some((mut total_tokens, mut num_docs, first_token_docs)) = loaded_stats.next() else {
        return Ok(None);
    };
    if first_token_docs.len() != terms.len() {
        return Err(lance_core::Error::internal(format!(
            "loaded FTS document-frequency count is {}, expected {}",
            first_token_docs.len(),
            terms.len()
        )));
    }
    let mut token_docs = HashMap::with_capacity(terms.len());
    for (term, count) in terms.iter().cloned().zip(first_token_docs) {
        token_docs.insert(term, count);
    }
    for (segment_total_tokens, segment_num_docs, segment_token_docs) in loaded_stats {
        if segment_token_docs.len() != terms.len() {
            return Err(lance_core::Error::internal(format!(
                "loaded FTS document-frequency count is {}, expected {}",
                segment_token_docs.len(),
                terms.len()
            )));
        }
        total_tokens = total_tokens
            .checked_add(segment_total_tokens)
            .ok_or_else(|| lance_core::Error::index("FTS corpus token count overflows u64"))?;
        num_docs = num_docs
            .checked_add(segment_num_docs)
            .ok_or_else(|| lance_core::Error::index("FTS corpus document count overflows usize"))?;
        for (term, count) in terms.iter().zip(segment_token_docs) {
            let total = token_docs.get_mut(term).ok_or_else(|| {
                lance_core::Error::internal(format!(
                    "global scorer term '{term}' was not initialized"
                ))
            })?;
            *total = total.checked_add(count).ok_or_else(|| {
                lance_core::Error::index(format!(
                    "FTS document frequency for term '{term}' overflows usize"
                ))
            })?;
        }
    }
    Ok(Some(Arc::new(MemBM25Scorer::new(
        total_tokens,
        num_docs,
        token_docs,
    ))))
}

/// Expand and score one indexed query leaf exactly once across all segments.
///
/// Expansion consumes one deterministic `max_expansions` budget in query
/// position order, with terms ordered lexicographically across every physical
/// segment and partition and deduplicated by `(term, position)`. The scorer's
/// document frequencies are then merged for exactly those final terms.
///
/// `base_scorer` is an API-compatibility hook for distributed/mixed callers
/// that already own corpus-wide statistics. It is validated against the final
/// vocabulary before being paired with the tokens.
#[doc(hidden)]
pub async fn prepare_bm25_query(
    indices: &[Arc<InvertedIndex>],
    query_tokens: Tokens,
    params: &FtsSearchParams,
    metrics: Option<&dyn crate::scalar::MetricsCollector>,
    base_scorer: Option<Arc<MemBM25Scorer>>,
) -> Result<PreparedBm25Query> {
    let first_index = indices.first().ok_or_else(|| {
        lance_core::Error::invalid_input("FTS index requires at least one segment")
    })?;
    let (tokens, has_all_query_positions) = if uses_fuzzy_expansion(params.fuzziness) {
        let tokens = Arc::new(final_query_tokens(indices, &query_tokens, params)?);
        let has_all_query_positions = has_all_query_positions(&query_tokens, tokens.as_ref());
        (tokens, has_all_query_positions)
    } else {
        (Arc::new(query_tokens), true)
    };
    let terms = unique_terms(tokens.as_ref());
    let scorer = if let Some(scorer) = base_scorer {
        if let Some(missing) = terms
            .iter()
            .find(|term| !scorer.token_docs.contains_key(term.as_str()))
        {
            return Err(lance_core::Error::invalid_input(format!(
                "injected BM25 scorer is missing compound FTS token '{missing}'"
            )));
        }
        scorer
    } else if let Some(scorer) = bm25_scorer_from_loaded_stats(indices, &terms)? {
        scorer
    } else {
        let (mut total_tokens, mut num_docs, first_token_docs) =
            first_index.bm25_stats_for_terms(&terms, metrics).await?;
        let mut token_docs = HashMap::with_capacity(terms.len());
        for (term, count) in terms.iter().cloned().zip(first_token_docs) {
            token_docs.insert(term, count);
        }

        for index in indices.iter().skip(1) {
            let (segment_total_tokens, segment_num_docs, segment_token_docs) =
                index.bm25_stats_for_terms(&terms, metrics).await?;
            total_tokens = total_tokens
                .checked_add(segment_total_tokens)
                .ok_or_else(|| lance_core::Error::index("FTS corpus token count overflows u64"))?;
            num_docs = num_docs.checked_add(segment_num_docs).ok_or_else(|| {
                lance_core::Error::index("FTS corpus document count overflows usize")
            })?;
            for (term, count) in terms.iter().zip(segment_token_docs) {
                let total = token_docs.get_mut(term).ok_or_else(|| {
                    lance_core::Error::internal(format!(
                        "global scorer term '{term}' was not initialized"
                    ))
                })?;
                *total = total.checked_add(count).ok_or_else(|| {
                    lance_core::Error::index(format!(
                        "FTS document frequency for term '{term}' overflows usize"
                    ))
                })?;
            }
        }
        Arc::new(MemBM25Scorer::new(total_tokens, num_docs, token_docs))
    };

    Ok(PreparedBm25Query {
        tokens,
        scorer,
        has_all_query_positions,
        shared_norm_addends: OnceLock::new(),
    })
}

/// Build a shared [`MemBM25Scorer`] across a set of FTS index segments.
///
/// Compatibility wrapper for callers that only need statistics. Indexed
/// execution should retain the [`PreparedBm25Query`] returned by
/// [`prepare_bm25_query`] so the same final vocabulary reaches search.
///
/// Aggregates each segment's `(total_tokens, num_docs, per_term_doc_freq)`
/// statistics into a single corpus-wide scorer.
///
/// `metrics`, when provided, is forwarded to the per-token metadata cache
/// boundary on each segment so callers running under an `ExecutionPlan`
/// (e.g. `MatchQueryExec`) see the reads triggered here in their per-query
/// `index_cache_hits`/`index_cache_misses` counters.
///
/// For exact queries this remains the compatibility producer paired with the
/// `with_base_scorer` consumer on FTS exec types. Fuzzy distributed execution
/// must retain the full [`PreparedBm25Query`] from [`prepare_bm25_query`]; a
/// scorer alone cannot preserve the canonical expansion vocabulary.
pub async fn build_global_bm25_scorer(
    indices: &[Arc<InvertedIndex>],
    query_tokens: &Tokens,
    params: &FtsSearchParams,
    metrics: Option<&dyn crate::scalar::MetricsCollector>,
) -> Result<MemBM25Scorer> {
    let prepared = prepare_bm25_query(indices, query_tokens.clone(), params, metrics, None).await?;
    Ok(prepared.scorer.as_ref().clone())
}

use lance_core::Error;

use crate::pbold;
use crate::progress::IndexBuildProgress;
use crate::scalar::{
    CreatedIndex, RowIdRemapper, ScalarIndex,
    expression::{FtsQueryParser, ScalarQueryParser},
    registry::{
        BasicTrainer, ScalarIndexCacheKey, ScalarIndexLoad, ScalarIndexPlugin, TrainingCriteria,
        TrainingOrdering, TrainingRequest,
    },
};

use super::IndexStore;

#[derive(Debug, Default)]
pub struct InvertedIndexPlugin;

impl InvertedIndexPlugin {
    pub async fn train_inverted_index(
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        params: InvertedIndexParams,
        fragment_ids: Option<Vec<u32>>,
        progress: Arc<dyn IndexBuildProgress>,
    ) -> Result<CreatedIndex> {
        let fragment_mask = fragment_ids.as_ref().and_then(|frag_ids| {
            if !frag_ids.is_empty() {
                // Create a mask with fragment_id in high 32 bits for distributed indexing
                // This mask is used to filter partitions belonging to specific fragments
                // If multiple fragments processed, use first fragment_id <<32 as mask
                Some((frag_ids[0] as u64) << 32)
            } else {
                None
            }
        });

        params.validate_format_version()?;
        let format_version = params.resolved_format_version();
        let is_element_document = params.get_document_granularity().is_list_element();
        let details = pbold::InvertedIndexDetails::try_from(&params)?;
        let mut inverted_index =
            InvertedIndexBuilder::new_with_fragment_mask(params, fragment_mask)
                .with_progress(progress);
        let files = inverted_index.update(data, index_store, None).await?;
        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&details).unwrap(),
            index_version: if is_element_document {
                INVERTED_INDEX_VERSION_V3
            } else {
                format_version.index_version()
            },
            files,
        })
    }

    /// Return true if the query can be used to speed up contains_tokens queries
    fn can_accelerate_queries(details: &pbold::InvertedIndexDetails) -> bool {
        details.base_tokenizer == Some("simple".to_string())
            && details.max_token_length.is_none()
            && details.language == serde_json::to_string(&Language::English).unwrap()
            && !details.stem
    }
}

struct InvertedIndexTrainingRequest {
    parameters: InvertedIndexParams,
    criteria: TrainingCriteria,
}

impl InvertedIndexTrainingRequest {
    pub fn new(parameters: InvertedIndexParams) -> Self {
        Self {
            parameters,
            criteria: TrainingCriteria::new(TrainingOrdering::None).with_row_id(),
        }
    }
}

impl TrainingRequest for InvertedIndexTrainingRequest {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn criteria(&self) -> &TrainingCriteria {
        &self.criteria
    }
}

#[async_trait]
impl BasicTrainer for InvertedIndexPlugin {
    fn new_training_request(
        &self,
        params: &str,
        field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        match field.data_type() {
            DataType::Utf8 | DataType::LargeUtf8 | DataType::LargeBinary => (),
            DataType::List(f) if matches!(f.data_type(), DataType::Utf8 | DataType::LargeUtf8) => (),
            DataType::LargeList(f) if matches!(f.data_type(), DataType::Utf8 | DataType::LargeUtf8) => (),

            _ => return Err(Error::invalid_input_source(format!(
                "A inverted index can only be created on a Utf8 or LargeUtf8 field/list or LargeBinary field. Column has type {:?}",
                field.data_type()
            )
                .into()))
        }

        let params = InvertedIndexParams::from_training_json(params)?;
        Ok(Box::new(InvertedIndexTrainingRequest::new(params)))
    }

    /// Train a new index
    ///
    /// The provided data must fulfill all the criteria returned by `training_criteria`.
    /// It is the caller's responsibility to ensure this.
    ///
    /// Returns index details that describe the index.  These details can potentially be
    /// useful for planning (although this will currently require inside information on
    /// the index type) and they will need to be provided when loading the index.
    ///
    /// It is the caller's responsibility to store these details somewhere.
    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        request: Box<dyn TrainingRequest>,
        fragment_ids: Option<Vec<u32>>,
        progress: Arc<dyn IndexBuildProgress>,
    ) -> Result<CreatedIndex> {
        let request = (request as Box<dyn std::any::Any>)
            .downcast::<InvertedIndexTrainingRequest>()
            .map_err(|_| {
                Error::invalid_input_source(
                    "must provide training request created by new_training_request".into(),
                )
            })?;
        Self::train_inverted_index(
            data,
            index_store,
            request.parameters.clone(),
            fragment_ids,
            progress,
        )
        .await
    }
}

#[async_trait]
impl ScalarIndexPlugin for InvertedIndexPlugin {
    fn basic_trainer(&self) -> Option<&dyn BasicTrainer> {
        Some(self)
    }

    fn name(&self) -> &str {
        "Inverted"
    }

    fn provides_exact_answer(&self) -> bool {
        false
    }

    fn version(&self) -> u32 {
        INVERTED_INDEX_VERSION_V3
    }

    fn new_query_parser(
        &self,
        index_name: String,
        _index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        let Ok(index_details) = _index_details.to_msg::<pbold::InvertedIndexDetails>() else {
            return None;
        };

        if Self::can_accelerate_queries(&index_details) {
            Some(Box::new(FtsQueryParser::new(
                index_name,
                self.name().to_string(),
            )))
        } else {
            None
        }
    }

    /// Load an index from storage
    ///
    /// The index details should match the details that were returned when the index was
    /// originally trained.
    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        index_details: &prost_types::Any,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        let index = InvertedIndex::load(index_store, frag_reuse_index, cache).await?;
        let details = index_details.to_msg::<pbold::InvertedIndexDetails>()?;
        let expected_granularity = DocumentGranularity::try_from(details.document_granularity)?;
        let physical_granularity = index.params().get_document_granularity();
        if physical_granularity != expected_granularity {
            return Err(Error::index(format!(
                "FTS document granularity in index details is {expected_granularity:?}, but the physical document schema implies {physical_granularity:?}"
            )));
        }
        Ok(index as Arc<dyn ScalarIndex>)
    }

    async fn get_or_insert_in_cache(
        &self,
        _index_store: Arc<dyn IndexStore>,
        _frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        cache: &LanceCache,
        load: ScalarIndexLoad<'_>,
    ) -> Result<Arc<dyn ScalarIndex>> {
        cache
            .get_or_insert_unsized_with_key(ScalarIndexCacheKey, || load)
            .await
    }

    fn details_as_json(&self, details: &prost_types::Any) -> Result<serde_json::Value> {
        let index_details = details.to_msg::<pbold::InvertedIndexDetails>()?;
        let index_params = InvertedIndexParams::try_from(&index_details)?;
        Ok(index_params.to_details_json()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::inverted::document_tokenizer::DocType;
    use crate::scalar::inverted::scorer::bm25_doc_weight_with_norm;
    use crate::scalar::{BuiltinIndexType, ScalarIndexParams};

    #[test]
    fn test_shared_norm_lut_kill_switch_parser() {
        for value in [None, Some(""), Some("1"), Some("on"), Some("false")] {
            assert!(
                shared_norm_lut_enabled_from_value(value),
                "unexpected disable: {value:?}"
            );
        }
        for value in [Some("0"), Some("off"), Some("OFF"), Some(" off ")] {
            assert!(
                !shared_norm_lut_enabled_from_value(value),
                "unexpected enable: {value:?}"
            );
        }
    }

    #[test]
    fn test_shared_norm_addends_are_bit_exact_and_built_once() {
        let scorer = Arc::new(MemBM25Scorer::new(4_096, 64, HashMap::new()));
        let key = scorer.doc_weight_cache_key().unwrap();
        let shared = Arc::new(QueryNormAddends::new(key));
        let threads = (0..8)
            .map(|_| {
                let scorer = scorer.clone();
                let shared = shared.clone();
                std::thread::spawn(move || shared.get_or_init(&scorer).unwrap())
            })
            .collect::<Vec<_>>();
        let built_addends = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(shared.build_count(), 1);
        assert!(
            built_addends[1..]
                .iter()
                .all(|addends| Arc::ptr_eq(&built_addends[0], addends))
        );
        for code in 0..=u8::MAX {
            let doc_length = dequantize_doc_length(code);
            for frequency in [0, 1, 2, 7, 255, u32::MAX] {
                assert_eq!(
                    bm25_doc_weight_with_norm(frequency, built_addends[0][code as usize]).to_bits(),
                    scorer.doc_weight(frequency, doc_length).to_bits()
                );
            }
        }

        let mismatched = MemBM25Scorer::new(4_097, 64, HashMap::new());
        assert!(shared.get_or_init(&mismatched).is_none());
        assert_eq!(shared.build_count(), 1);
    }

    #[test]
    fn test_shared_norm_addend_query_routing_and_snapshot_invalidation() {
        let make_prepared = |num_terms, scorer: Arc<MemBM25Scorer>| {
            let tokens = (0..num_terms).map(|term| format!("t{term}")).collect();
            PreparedBm25Query::from_parts(
                Arc::new(Tokens::new(tokens, DocType::Text)),
                scorer,
                true,
            )
        };
        let scorer = Arc::new(MemBM25Scorer::new(100, 10, HashMap::new()));
        for num_terms in [1, 4, 5] {
            let prepared = make_prepared(num_terms, scorer.clone());
            assert!(!prepared.should_share_norm_addends(true, true, query::Operator::And, false));
            assert!(prepared.shared_norm_addends.get().is_none());
        }

        let prepared = make_prepared(2, scorer.clone());
        assert!(!prepared.should_share_norm_addends(false, true, query::Operator::And, false));
        assert!(!prepared.should_share_norm_addends(true, false, query::Operator::And, false));
        assert!(!prepared.should_share_norm_addends(true, true, query::Operator::Or, false));
        assert!(!prepared.should_share_norm_addends(true, true, query::Operator::And, true));
        assert!(prepared.shared_norm_addends.get().is_none());
        assert!(prepared.shared_norm_addends_with_enabled(false).is_none());
        assert!(prepared.shared_norm_addends.get().is_none());
        assert!(prepared.should_share_norm_addends(true, true, query::Operator::And, false));

        let old_shared = prepared.shared_norm_addends().unwrap();
        let mut updated_scorer = scorer.as_ref().clone();
        updated_scorer.update(&HashMap::new(), 25);
        let updated = make_prepared(2, Arc::new(updated_scorer));
        let updated_shared = updated.shared_norm_addends().unwrap();
        assert_ne!(old_shared.scorer_key, updated_shared.scorer_key);
        assert!(!Arc::ptr_eq(&old_shared, &updated_shared));
    }

    #[test]
    fn test_sync_df_kill_switch_parser() {
        for value in [None, Some(""), Some("1"), Some("on"), Some("false")] {
            assert!(
                sync_df_enabled_from_value(value),
                "unexpected disable: {value:?}"
            );
        }
        for value in [Some("0"), Some("off"), Some("OFF"), Some(" off ")] {
            assert!(
                !sync_df_enabled_from_value(value),
                "unexpected enable: {value:?}"
            );
        }
    }

    #[test]
    fn test_sync_df_merge_reports_checked_overflow_and_invalid_shape() {
        let terms = vec!["term".to_string()];
        for (stats, expected) in [
            (
                vec![(u64::MAX, 1, vec![1]), (1, 1, vec![1])],
                "corpus token count overflows u64",
            ),
            (
                vec![(1, usize::MAX, vec![1]), (1, 1, vec![1])],
                "corpus document count overflows usize",
            ),
            (
                vec![(1, 1, vec![usize::MAX]), (1, 1, vec![1])],
                "document frequency for term 'term' overflows usize",
            ),
        ] {
            let error = merge_loaded_bm25_stats(&terms, stats).unwrap_err();
            assert!(
                error.to_string().contains(expected),
                "unexpected error: {error}"
            );
        }

        let error = merge_loaded_bm25_stats(&terms, vec![(1, 1, Vec::new())]).unwrap_err();
        assert!(matches!(error, lance_core::Error::Internal { .. }));
        assert!(
            error
                .to_string()
                .contains("document-frequency count is 0, expected 1"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_plugin_version_tracks_v3_capability_gate() {
        let plugin = InvertedIndexPlugin;
        assert_eq!(plugin.version(), INVERTED_INDEX_VERSION_V3);
    }

    #[test]
    fn test_details_json_includes_document_granularity() {
        let details = pbold::InvertedIndexDetails {
            document_granularity: pbold::inverted_index_details::DocumentGranularity::ListElement
                as i32,
            ..Default::default()
        };
        let details = prost_types::Any::from_msg(&details).unwrap();

        let json = InvertedIndexPlugin.details_as_json(&details).unwrap();

        assert_eq!(json["document_granularity"], "list_element");
    }

    #[test]
    fn test_new_training_request_defaults_missing_block_size_to_128() {
        let plugin = InvertedIndexPlugin;
        let field = Field::new("text", DataType::Utf8, true);

        let cases = [
            (
                ScalarIndexParams::for_builtin(BuiltinIndexType::Inverted),
                false,
            ),
            (ScalarIndexParams::new("inverted".to_string()), false),
            (
                ScalarIndexParams::new("inverted".to_string())
                    .with_params(&serde_json::json!({ "with_position": true })),
                true,
            ),
        ];

        for (params, expected_with_position) in cases {
            let request = plugin
                .new_training_request(params.params.as_deref().unwrap_or("{}"), &field)
                .unwrap();
            let request = request
                .as_any()
                .downcast_ref::<InvertedIndexTrainingRequest>()
                .unwrap();

            assert_eq!(request.parameters.posting_block_size(), DEFAULT_BLOCK_SIZE);
            assert_eq!(request.parameters.has_positions(), expected_with_position);
        }
    }
}
