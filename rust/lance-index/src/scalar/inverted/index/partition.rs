// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use smallvec::SmallVec;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PositionMatchSummary {
    exact_scoring_required: bool,
    every_position_matched: bool,
}

#[derive(Debug, Clone, Copy)]
pub(in super::super) struct PostingLoadOptions {
    force_global_scorer: bool,
    read_policy: PostingReadPolicy,
}

impl PostingLoadOptions {
    const fn read_ahead(force_global_scorer: bool) -> Self {
        Self {
            force_global_scorer,
            read_policy: PostingReadPolicy::ReadAhead,
        }
    }

    pub(in super::super) const fn cache_aware_exact(force_global_scorer: bool) -> Self {
        Self {
            force_global_scorer,
            read_policy: PostingReadPolicy::CacheAwareExact,
        }
    }
}

fn summarize_position_matches(mut positions: SmallVec<[(u32, bool); 8]>) -> PositionMatchSummary {
    positions.sort_unstable_by_key(|(position, _)| *position);

    let mut exact_scoring_required = false;
    let mut every_position_matched = true;
    let mut group_start = 0;
    while group_start < positions.len() {
        let position = positions[group_start].0;
        let mut group_end = group_start + 1;
        let mut group_matched = positions[group_start].1;
        while group_end < positions.len() && positions[group_end].0 == position {
            exact_scoring_required = true;
            group_matched |= positions[group_end].1;
            group_end += 1;
        }
        every_position_matched &= group_matched;
        group_start = group_end;
    }

    PositionMatchSummary {
        exact_scoring_required,
        every_position_matched,
    }
}

fn token_dictionary_may_match(
    dictionary: &TokenSet,
    tokens: &Tokens,
    operator: Operator,
    is_phrase_query: bool,
) -> bool {
    if tokens.is_empty() {
        return false;
    }
    if operator != Operator::And && !is_phrase_query {
        return (0..tokens.len()).any(|index| dictionary.get(tokens.get_token(index)).is_some());
    }

    // Query positions are normally adjacent, but `Tokens::with_positions` is
    // public and does not require that ordering. Keep the exact behavior
    // without allocating two hash tables for every leaf in every partition.
    let mut positions = SmallVec::<[(u32, bool); 8]>::new();
    for index in 0..tokens.len() {
        positions.push((
            tokens.position(index),
            dictionary.get(tokens.get_token(index)).is_some(),
        ));
    }
    summarize_position_matches(positions).every_position_matched
}

fn posting_group_demand_counts(
    inverted_list: &PostingListReader,
    token_ids: &[(u32, String, u32)],
) -> HashMap<(u32, u32), usize> {
    let mut demanded_token_ids = token_ids
        .iter()
        .map(|(token_id, _, _)| *token_id)
        .collect::<Vec<_>>();
    demanded_token_ids.sort_unstable();
    demanded_token_ids.dedup();

    let mut counts = HashMap::new();
    for token_id in demanded_token_ids {
        if let Some(group) = inverted_list.group_range_for_token(token_id) {
            *counts.entry(group).or_default() += 1;
        }
    }
    counts
}

fn effective_posting_read_policy(
    inverted_list: &PostingListReader,
    requested_policy: PostingReadPolicy,
    group_demand_counts: &HashMap<(u32, u32), usize>,
    token_id: u32,
) -> PostingReadPolicy {
    if requested_policy == PostingReadPolicy::ReadAhead {
        return PostingReadPolicy::ReadAhead;
    }
    let Some(group) = inverted_list.group_range_for_token(token_id) else {
        return PostingReadPolicy::CacheAwareExact;
    };
    let demand_count = group_demand_counts.get(&group).copied();
    debug_assert!(
        demand_count.is_some(),
        "posting group {group:?} must have a demand count for token {token_id}"
    );
    if demand_count == Some(1) {
        PostingReadPolicy::CacheAwareExact
    } else {
        PostingReadPolicy::ReadAhead
    }
}

/// Query-level inputs for a grouped-term score upper bound.
///
/// The exact grouped score multiplies and then sums every expansion term in a
/// stable order. Computing the bound from the `f32` sum of query weights can
/// round below that score, so retain the `f64` sum and widen once for the term
/// multiplications plus the final `f32` additions.
#[derive(Debug, Clone, Copy)]
struct GroupedScoreUpperBound {
    query_weight: f32,
    exact_query_weight_sum: f64,
    rounding_factor: f64,
}

impl GroupedScoreUpperBound {
    fn new(query_weights: impl Iterator<Item = f32>) -> Self {
        let mut query_weight = 0.0_f32;
        let mut exact_query_weight_sum = 0.0_f64;
        let mut num_terms = 0usize;
        for weight in query_weights {
            query_weight += weight;
            exact_query_weight_sum += f64::from(weight);
            num_terms += 1;
        }
        Self {
            query_weight,
            exact_query_weight_sum,
            // Two extra stages cover each term's rounded BM25 evaluation and
            // multiplication before the grouped f32 sum.
            rounding_factor: score_sum_upper_bound_factor(num_terms.saturating_add(2)),
        }
    }

    #[inline]
    fn score(self, union_freq: u32, doc_length: u32, scorer: &MemBM25Scorer) -> f32 {
        outward_f32_upper_bound(
            self.exact_query_weight_sum
                * f64::from(scorer.doc_weight(union_freq, doc_length))
                * self.rounding_factor,
        )
    }
}

#[derive(Debug, Clone, DeepSizeOf)]
pub struct InvertedPartition {
    // 0 for legacy format
    pub(super) id: u64,
    pub(super) store: Arc<dyn IndexStore>,
    pub(crate) tokens: TokenSet,
    pub(crate) inverted_list: Arc<PostingListReader>,
    /// Legacy documents stay in their original complete `DocSet`; modern
    /// documents use typed, independently-loaded lengths and addresses.
    pub(in super::super) docs: PartitionDocumentStore,
    pub(super) token_set_format: TokenSetFormat,
}

impl InvertedPartition {
    /// Check if this partition belongs to the specified fragment.
    ///
    /// This method encapsulates the bit manipulation logic for fragment filtering
    /// in distributed indexing scenarios.
    ///
    /// # Arguments
    /// * `fragment_mask` - A mask with fragment_id in high 32 bits
    ///
    /// # Returns
    /// * `true` if the partition belongs to the fragment, `false` otherwise
    pub fn belongs_to_fragment(&self, fragment_mask: u64) -> bool {
        (self.id() & fragment_mask) == fragment_mask
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn store(&self) -> &dyn IndexStore {
        self.store.as_ref()
    }

    pub fn is_legacy(&self) -> bool {
        self.inverted_list.is_legacy_layout()
    }

    /// This partition's `(num_rows, per_term_row_freq)` counted over distinct
    /// row ids rather than documents. See
    /// [`InvertedIndex::bm25_row_stats_for_terms`] for why the distinction
    /// exists and when this runs.
    ///
    /// A partition whose documents map one-to-one onto rows takes the same
    /// single-metadata-row `posting_len_for_token` lookup the document-granularity
    /// statistics use, so only a partition that really indexed one document per
    /// list element pays for reading posting lists.
    pub(super) async fn row_stats_for_terms(
        &self,
        terms: &[String],
        metrics: Option<&dyn MetricsCollector>,
    ) -> Result<(usize, Vec<usize>)> {
        let docs = self.docs.address_keyed().await?;
        let num_rows = docs.num_distinct_rows();
        let one_document_per_row = num_rows == docs.len();
        let is_legacy = self.is_legacy();
        let mut row_freqs = Vec::with_capacity(terms.len());
        for term in terms {
            let Some(token_id) = self.tokens.get(term) else {
                row_freqs.push(0);
                continue;
            };
            if one_document_per_row {
                row_freqs.push(
                    self.inverted_list
                        .posting_len_for_token(token_id, metrics)
                        .await?,
                );
                continue;
            }
            // Deduplicating needs the postings themselves; `posting_len_for_token`
            // only knows how many documents there are. The read is cached, and a
            // partition that reaches this branch forces the full-read fallback in
            // `combined_fields_search` anyway (legacy layout or non-ascending
            // row_ids), so nothing that would otherwise have been pruned is read.
            let posting = self
                .inverted_list
                .posting_list(token_id, false, metrics.unwrap_or(&NoOpMetricsCollector))
                .await?;
            let mut rows = RoaringTreemap::new();
            for (row_id, _) in live_posting_rows(&posting, &docs, is_legacy) {
                rows.insert(row_id);
            }
            row_freqs.push(rows.len() as usize);
        }
        Ok((num_rows, row_freqs))
    }

    pub async fn load(
        store: Arc<dyn IndexStore>,
        id: u64,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        index_cache: &LanceCache,
        token_set_format: TokenSetFormat,
    ) -> Result<Self> {
        let token_file = store.open_index_file(&token_file_path(id)).await?;
        let tokens = TokenSet::load(token_file, token_set_format).await?;
        let invert_list_file = store.open_index_file(&posting_file_path(id)).await?;
        let mut inverted_list = PostingListReader::try_new(invert_list_file, index_cache).await?;
        let docs_path = doc_file_path(id);
        let docs_reader = store.open_index_file(&docs_path).await?;
        let docs = PartitionDocuments::try_new(
            store.clone(),
            docs_path,
            id,
            WeakLanceCache::from(index_cache),
            docs_reader.as_ref(),
            frag_reuse_index,
            // 256-document blocks score with quantized document lengths.
            inverted_list.block_size() == MAX_POSTING_BLOCK_SIZE,
        )?;
        inverted_list.modern_num_docs = Some(docs.len());

        Ok(Self {
            id,
            store,
            tokens,
            inverted_list: Arc::new(inverted_list),
            docs: PartitionDocumentStore::Modern(Arc::new(docs)),
            token_set_format,
        })
    }

    fn map(&self, token: &str) -> Option<u32> {
        self.tokens.get(token)
    }

    /// Return whether this partition's token dictionary can satisfy a query
    /// leaf without reading any posting data.
    ///
    /// Fuzzy expansions and identifier subwords that belong to the same
    /// original query position are alternatives. AND and phrase leaves need
    /// at least one dictionary term for every original position; OR leaves
    /// need any term. A `false` result is therefore an exact empty-source
    /// proof, while `true` remains conservative until postings are loaded.
    pub(in super::super) fn may_match_tokens(
        &self,
        tokens: &Tokens,
        operator: Operator,
        is_phrase_query: bool,
    ) -> bool {
        token_dictionary_may_match(&self.tokens, tokens, operator, is_phrase_query)
    }

    pub fn expand_fuzzy(&self, tokens: &Tokens, params: &FtsSearchParams) -> Result<Tokens> {
        let mut new_tokens = Vec::with_capacity(min(tokens.len(), params.max_expansions));
        let mut new_positions = Vec::with_capacity(new_tokens.capacity());
        let mut seen = HashSet::new();
        for token_idx in 0..tokens.len() {
            let remaining = params.max_expansions.saturating_sub(new_tokens.len());
            if remaining == 0 {
                break;
            }
            let token = tokens.get_token(token_idx);
            let position = tokens.position(token_idx);
            let base_prefix_len = tokens.token_type().prefix_len(token) as u32;
            let mut candidates = BTreeSet::new();
            self.collect_fuzzy_candidates(
                token,
                base_prefix_len,
                params,
                remaining,
                &mut candidates,
            )?;
            for candidate in candidates {
                if new_tokens.len() >= params.max_expansions {
                    break;
                }
                if seen.insert((candidate.clone(), position)) {
                    new_tokens.push(candidate);
                    new_positions.push(position);
                }
            }
        }
        Ok(Tokens::with_positions(
            new_tokens,
            new_positions,
            tokens.token_type().clone(),
        ))
    }

    /// Collect up to `limit` fuzzy candidates for one query token from this
    /// partition's token FST, in key (lexicographic) order. Callers merge
    /// candidates across partitions and apply the query-wide
    /// `max_expansions` budget; truncating each partition at `limit` is
    /// lossless for that selection because any term among the merged
    /// lexicographically-smallest `limit` is also among its own partition's
    /// smallest `limit`.
    pub(super) fn collect_fuzzy_candidates(
        &self,
        token: &str,
        base_prefix_len: u32,
        params: &FtsSearchParams,
        limit: usize,
        candidates: &mut BTreeSet<String>,
    ) -> Result<()> {
        let fuzziness = match params.fuzziness {
            Some(fuzziness) => fuzziness,
            None => MatchQuery::auto_fuzziness(token),
        };
        let lev = fst::automaton::Levenshtein::new(token, fuzziness)
            .map_err(|e| Error::index(format!("failed to construct the fuzzy query: {}", e)))?;

        if let TokenMap::Fst(ref map) = self.tokens.tokens {
            let mut expanded = Vec::new();
            match base_prefix_len + params.prefix_length {
                0 => take_fst_keys(map.search(lev), &mut expanded, limit),
                prefix_length => {
                    let prefix = &token[..min(prefix_length as usize, token.len())];
                    let prefix = fst::automaton::Str::new(prefix).starts_with();
                    take_fst_keys(map.search(lev.intersection(prefix)), &mut expanded, limit)
                }
            }
            candidates.extend(expanded);
            Ok(())
        } else {
            Err(Error::index(
                "tokens is not fst, which is not expected".to_owned(),
            ))
        }
    }

    #[inline]
    fn grouped_score_upper_bound(
        score_upper_bound: GroupedScoreUpperBound,
        union_freq: u32,
        doc_length: u32,
        scorer: &MemBM25Scorer,
    ) -> f32 {
        // BM25's document weight is monotonic in frequency and every IDF is
        // non-negative. Scoring the summed frequency with the summed IDF is
        // therefore an upper bound on the sum of the individual term scores.
        score_upper_bound.score(union_freq, doc_length, scorer)
    }

    fn grouped_block_max_scores(
        doc_ids: &[u32],
        frequencies: &[u32],
        block_size: usize,
        docs: &LoadedDocLengths,
        score_upper_bound: GroupedScoreUpperBound,
        scorer: &MemBM25Scorer,
    ) -> Vec<f32> {
        doc_ids
            .chunks(block_size)
            .zip(frequencies.chunks(block_size))
            .map(|(doc_ids, frequencies)| {
                doc_ids
                    .iter()
                    .zip(frequencies)
                    .map(|(doc_id, freq)| {
                        Self::grouped_score_upper_bound(
                            score_upper_bound,
                            *freq,
                            docs.scoring_num_tokens(*doc_id),
                            scorer,
                        )
                    })
                    .fold(0.0, f32::max)
            })
            .collect()
    }

    fn union_plain_posting_lists(
        postings: Vec<PostingList>,
        docs: &LoadedDocLengths,
        score_upper_bound: GroupedScoreUpperBound,
        scorer: &MemBM25Scorer,
    ) -> Result<PostingList> {
        let mut freqs_by_row_id = BTreeMap::new();
        for posting in postings {
            for (row_id, freq, _) in posting.iter() {
                let entry = freqs_by_row_id.entry(row_id).or_insert(0u32);
                *entry = entry.checked_add(freq).ok_or_else(|| {
                    Error::index(format!("posting frequency overflow for row id {}", row_id))
                })?;
            }
        }
        let mut row_ids = Vec::with_capacity(freqs_by_row_id.len());
        let mut frequencies = Vec::with_capacity(freqs_by_row_id.len());
        let mut max_score = 0.0_f32;
        for (row_id, freq) in freqs_by_row_id {
            max_score = max_score.max(Self::grouped_score_upper_bound(
                score_upper_bound,
                freq,
                docs.num_tokens_by_row_id(row_id),
                scorer,
            ));
            row_ids.push(row_id);
            frequencies.push(freq as f32);
        }
        Ok(PostingList::Plain(PlainPostingList::new(
            ScalarBuffer::from(row_ids),
            ScalarBuffer::from(frequencies),
            Some(max_score),
            None,
        )))
    }

    fn union_plain_posting_lists_with_positions(
        postings: Vec<PostingList>,
        docs: &LoadedDocLengths,
        score_upper_bound: GroupedScoreUpperBound,
        scorer: &MemBM25Scorer,
    ) -> Result<PostingList> {
        let mut positions_by_row_id = BTreeMap::<u64, Vec<u32>>::new();
        for posting in postings {
            for (row_id, _, positions) in posting.iter() {
                let positions = positions.ok_or_else(|| {
                    Error::index("cannot union grouped phrase terms without positions".to_string())
                })?;
                positions_by_row_id
                    .entry(row_id)
                    .or_default()
                    .extend(positions);
            }
        }
        if positions_by_row_id.is_empty() {
            return Ok(PostingList::Plain(PlainPostingList::new(
                ScalarBuffer::from(Vec::<u64>::new()),
                ScalarBuffer::from(Vec::<f32>::new()),
                None,
                None,
            )));
        }

        let mut row_ids = Vec::with_capacity(positions_by_row_id.len());
        let mut frequencies = Vec::with_capacity(positions_by_row_id.len());
        let mut positions_builder = ListBuilder::new(Int32Builder::new());
        let mut max_score = 0.0_f32;
        for (row_id, mut positions) in positions_by_row_id {
            positions.sort_unstable();
            let frequency = positions.len() as u32;
            max_score = max_score.max(Self::grouped_score_upper_bound(
                score_upper_bound,
                frequency,
                docs.num_tokens_by_row_id(row_id),
                scorer,
            ));
            row_ids.push(row_id);
            frequencies.push(frequency as f32);
            for position in positions {
                positions_builder.values().append_value(position as i32);
            }
            positions_builder.append(true);
        }

        Ok(PostingList::Plain(PlainPostingList::new(
            ScalarBuffer::from(row_ids),
            ScalarBuffer::from(frequencies),
            Some(max_score),
            Some(positions_builder.finish()),
        )))
    }

    fn union_compressed_posting_lists(
        postings: Vec<PostingList>,
        docs: &LoadedDocLengths,
        score_upper_bound: GroupedScoreUpperBound,
        scorer: &MemBM25Scorer,
    ) -> Result<PostingList> {
        let block_size = postings
            .iter()
            .find_map(|posting| match posting {
                PostingList::Compressed(posting) => Some(posting.block_size),
                PostingList::Plain(_) => None,
            })
            .unwrap_or(LEGACY_BLOCK_SIZE);
        let mut freqs_by_doc_id = BTreeMap::new();
        for posting in postings {
            for (doc_id, freq, _) in posting.iter() {
                let doc_id = u32::try_from(doc_id).map_err(|_| {
                    Error::index(format!(
                        "compressed posting doc id {} exceeds u32::MAX",
                        doc_id
                    ))
                })?;
                let entry = freqs_by_doc_id.entry(doc_id).or_insert(0u32);
                *entry = entry.checked_add(freq).ok_or_else(|| {
                    Error::index(format!("posting frequency overflow for doc id {}", doc_id))
                })?;
            }
        }
        if freqs_by_doc_id.is_empty() {
            return Ok(PostingList::Plain(PlainPostingList::new(
                ScalarBuffer::from(Vec::<u64>::new()),
                ScalarBuffer::from(Vec::<f32>::new()),
                None,
                None,
            )));
        }

        let mut builder = PostingListBuilder::new_with_block_size(false, block_size);
        let mut doc_ids = Vec::with_capacity(freqs_by_doc_id.len());
        let mut frequencies = Vec::with_capacity(freqs_by_doc_id.len());
        for (doc_id, freq) in freqs_by_doc_id {
            builder.add(doc_id, PositionRecorder::Count(freq));
            doc_ids.push(doc_id);
            frequencies.push(freq);
        }
        let block_max_scores = Self::grouped_block_max_scores(
            &doc_ids,
            &frequencies,
            block_size,
            docs,
            score_upper_bound,
            scorer,
        );
        let batch = builder.to_batch(block_max_scores)?;
        let max_score = batch[MAX_SCORE_COL].as_primitive::<Float32Type>().value(0);
        let length = batch[LENGTH_COL].as_primitive::<UInt32Type>().value(0);
        PostingList::from_batch(&batch, Some(max_score), Some(length))
    }

    fn union_compressed_posting_lists_with_positions(
        postings: Vec<PostingList>,
        docs: &LoadedDocLengths,
        score_upper_bound: GroupedScoreUpperBound,
        scorer: &MemBM25Scorer,
    ) -> Result<PostingList> {
        let block_size = postings
            .iter()
            .find_map(|posting| match posting {
                PostingList::Compressed(posting) => Some(posting.block_size),
                PostingList::Plain(_) => None,
            })
            .unwrap_or(LEGACY_BLOCK_SIZE);
        let mut positions_by_doc_id = BTreeMap::<u32, Vec<u32>>::new();
        for posting in postings {
            for (doc_id, _, positions) in posting.iter() {
                let doc_id = u32::try_from(doc_id).map_err(|_| {
                    Error::index(format!(
                        "compressed posting doc id {} exceeds u32::MAX",
                        doc_id
                    ))
                })?;
                let positions = positions.ok_or_else(|| {
                    Error::index("cannot union grouped phrase terms without positions".to_string())
                })?;
                positions_by_doc_id
                    .entry(doc_id)
                    .or_default()
                    .extend(positions);
            }
        }
        if positions_by_doc_id.is_empty() {
            return Ok(PostingList::Plain(PlainPostingList::new(
                ScalarBuffer::from(Vec::<u64>::new()),
                ScalarBuffer::from(Vec::<f32>::new()),
                None,
                None,
            )));
        }

        let mut builder = PostingListBuilder::new_with_block_size(true, block_size);
        let mut doc_ids = Vec::with_capacity(positions_by_doc_id.len());
        let mut frequencies = Vec::with_capacity(positions_by_doc_id.len());
        for (doc_id, mut positions) in positions_by_doc_id {
            positions.sort_unstable();
            let frequency = positions.len() as u32;
            builder.add(doc_id, PositionRecorder::Position(positions.into()));
            doc_ids.push(doc_id);
            frequencies.push(frequency);
        }
        let block_max_scores = Self::grouped_block_max_scores(
            &doc_ids,
            &frequencies,
            block_size,
            docs,
            score_upper_bound,
            scorer,
        );
        let batch = builder.to_batch(block_max_scores)?;
        let max_score = batch[MAX_SCORE_COL].as_primitive::<Float32Type>().value(0);
        let length = batch[LENGTH_COL].as_primitive::<UInt32Type>().value(0);
        PostingList::from_batch(&batch, Some(max_score), Some(length))
    }

    fn union_posting_lists(
        postings: Vec<PostingList>,
        docs: &LoadedDocLengths,
        with_positions: bool,
        score_upper_bound: GroupedScoreUpperBound,
        scorer: &MemBM25Scorer,
    ) -> Result<PostingList> {
        let has_plain = postings
            .iter()
            .any(|posting| matches!(posting, PostingList::Plain(_)));
        let has_compressed = postings
            .iter()
            .any(|posting| matches!(posting, PostingList::Compressed(_)));
        match (has_plain, has_compressed) {
            (true, true) => Err(Error::index(
                "cannot union mixed plain and compressed posting lists".to_owned(),
            )),
            (true, false) if with_positions => Self::union_plain_posting_lists_with_positions(
                postings,
                docs,
                score_upper_bound,
                scorer,
            ),
            (true, false) => {
                Self::union_plain_posting_lists(postings, docs, score_upper_bound, scorer)
            }
            (false, true) if with_positions => Self::union_compressed_posting_lists_with_positions(
                postings,
                docs,
                score_upper_bound,
                scorer,
            ),
            (false, true) => {
                Self::union_compressed_posting_lists(postings, docs, score_upper_bound, scorer)
            }
            (false, false) => Ok(PostingList::Plain(PlainPostingList::new(
                ScalarBuffer::from(Vec::<u64>::new()),
                ScalarBuffer::from(Vec::<f32>::new()),
                None,
                None,
            ))),
        }
    }

    // search the documents that contain the query
    // return the doc info and the doc length
    // ref: https://en.wikipedia.org/wiki/Okapi_BM25
    //
    // `force_global_scorer` is used by compound search, where leaf scores and
    // bounds must share corpus-level statistics before the global collector
    // can safely propagate its threshold. Old posting formats without impacts
    // fall back to a scorer-derived global upper bound in that mode.
    pub(in super::super) async fn load_posting_lists(
        &self,
        tokens: &Tokens,
        params: &FtsSearchParams,
        operator: Operator,
        impact_scorer: &MemBM25Scorer,
        metrics: &dyn MetricsCollector,
        force_global_scorer: bool,
    ) -> Result<LoadedPostings> {
        self.load_posting_lists_with_policy(
            tokens,
            params,
            operator,
            impact_scorer,
            metrics,
            PostingLoadOptions::read_ahead(force_global_scorer),
        )
        .await
    }

    #[instrument(name = "load_posting_lists", level = "debug", skip_all)]
    pub(in super::super) async fn load_posting_lists_with_policy(
        &self,
        tokens: &Tokens,
        params: &FtsSearchParams,
        operator: Operator,
        impact_scorer: &MemBM25Scorer,
        metrics: &dyn MetricsCollector,
        options: PostingLoadOptions,
    ) -> Result<LoadedPostings> {
        let PostingLoadOptions {
            force_global_scorer,
            read_policy: requested_read_policy,
        } = options;
        let is_phrase_query = params.phrase_slop.is_some();
        let is_and_query = operator == Operator::And;
        // Fuzzy expansion already ran once at the index level (see
        // `InvertedIndex::bm25_search`) under the global `max_expansions`
        // budget. Positions identify alternatives that must share one posting
        // iterator, including code identifier subwords and fuzzy expansions.
        let mut token_ids = Vec::with_capacity(tokens.len());
        let mut position_matches = SmallVec::<[(u32, bool); 8]>::new();
        for index in 0..tokens.len() {
            let token = tokens.get_token(index);
            let position = tokens.position(index);
            let token_id = self.map(token);
            position_matches.push((position, token_id.is_some()));
            if let Some(token_id) = token_id {
                token_ids.push((token_id, token.to_owned(), position));
            }
        }
        let position_summary = summarize_position_matches(position_matches);
        let exact_scoring_required = position_summary.exact_scoring_required;
        if token_ids.is_empty() {
            return Ok(LoadedPostings::empty());
        }
        if (is_and_query || is_phrase_query) && !position_summary.every_position_matched {
            return Ok(LoadedPostings::empty());
        }

        token_ids.sort_unstable_by_key(|(token_id, _, position)| (*position, *token_id));
        token_ids.dedup_by(|lhs, rhs| lhs.0 == rhs.0 && lhs.2 == rhs.2);

        let group_demand_counts = if requested_read_policy == PostingReadPolicy::CacheAwareExact {
            posting_group_demand_counts(self.inverted_list.as_ref(), &token_ids)
        } else {
            HashMap::new()
        };

        let num_docs = self.docs.len();
        let loaded_postings = stream::iter(token_ids)
            .map(|(token_id, token, position)| {
                let read_policy = effective_posting_read_policy(
                    self.inverted_list.as_ref(),
                    requested_read_policy,
                    &group_demand_counts,
                    token_id,
                );
                async move {
                    let posting = match read_policy {
                        PostingReadPolicy::ReadAhead => {
                            self.inverted_list
                                .posting_list(token_id, is_phrase_query, metrics)
                                .await?
                        }
                        PostingReadPolicy::CacheAwareExact => {
                            self.inverted_list
                                .posting_list_with_policy(
                                    token_id,
                                    is_phrase_query,
                                    metrics,
                                    read_policy,
                                )
                                .await?
                        }
                    };

                    Result::Ok((token_id, token, position, posting))
                }
            })
            .buffered(self.store.io_parallelism())
            .try_collect::<Vec<_>>()
            .await?;

        let needs_union = loaded_postings
            .windows(2)
            .any(|window| window[0].2 == window[1].2);
        if (is_and_query || is_phrase_query)
            && !needs_union
            && loaded_postings
                .iter()
                .any(|(_, _, _, posting)| posting.is_empty())
        {
            return Ok(LoadedPostings::empty());
        }

        if !needs_union {
            let impact_safe = loaded_postings
                .iter()
                .all(|(_, _, _, posting)| posting.has_impacts());
            return Ok(LoadedPostings {
                postings: loaded_postings
                    .into_iter()
                    .map(|(token_id, token, position, posting)| {
                        let needs_scorer_upper_bound = (exact_scoring_required
                            || force_global_scorer)
                            && !posting.has_impacts();
                        let query_weight =
                            if impact_safe || exact_scoring_required || force_global_scorer {
                                impact_scorer.query_weight(&token)
                            } else {
                                idf(posting.len(), num_docs)
                            };
                        let posting = PostingIterator::with_query_weight(
                            token,
                            token_id,
                            position,
                            query_weight,
                            posting,
                            num_docs,
                        );
                        if needs_scorer_upper_bound {
                            posting.with_scorer_upper_bound()
                        } else {
                            posting
                        }
                    })
                    .collect(),
                grouped_expansions: Vec::new(),
                impact_safe,
                exact_scoring_required,
            });
        }

        let docs_for_union = if needs_union {
            Some(match &self.docs {
                PartitionDocumentStore::Legacy(docs) => LoadedDocLengths::Legacy(docs.clone()),
                PartitionDocumentStore::Modern(documents) => {
                    LoadedDocLengths::Modern(documents.lengths().await?)
                }
            })
        } else {
            None
        };

        // WAND's AND mode treats every iterator as required, so expansions from
        // one original query position must be merged before scoring.
        let mut grouped_postings = Vec::new();
        let mut grouped_expansions = Vec::new();
        let mut iter = loaded_postings.into_iter().peekable();
        while let Some((token_id, token, position, posting)) = iter.next() {
            let mut group = vec![(token_id, token, posting)];
            while matches!(iter.peek(), Some((_, _, next_position, _)) if *next_position == position)
            {
                let (token_id, token, _, posting) = iter.next().expect("peeked item must exist");
                group.push((token_id, token, posting));
            }

            let (token_id, token, posting) = if group.len() == 1 {
                group.pop().expect("single-item group must exist")
            } else {
                let token_id = group[0].0;
                let token = group[0].1.clone();
                let terms = group
                    .iter()
                    .map(|(_, token, posting)| {
                        GroupedTermScorer::new(impact_scorer.query_weight(token), posting)
                    })
                    .collect::<Vec<_>>();
                let terms = Arc::<[GroupedTermScorer]>::from(terms);
                let score_upper_bound =
                    GroupedScoreUpperBound::new(terms.iter().map(GroupedTermScorer::query_weight));
                let query_weight = score_upper_bound.query_weight;
                grouped_expansions.push(GroupedExpansionTerms {
                    position,
                    terms: terms.clone(),
                });
                let postings = group
                    .into_iter()
                    .map(|(_, _, posting)| posting)
                    .collect::<Vec<_>>();
                let docs = docs_for_union.as_ref().ok_or_else(|| {
                    Error::index("union docs were not loaded for grouped query terms".to_string())
                })?;
                let posting = Self::union_posting_lists(
                    postings,
                    docs,
                    is_phrase_query,
                    score_upper_bound,
                    impact_scorer,
                )?;
                if posting.is_empty() && (is_and_query || is_phrase_query) {
                    return Ok(LoadedPostings::empty());
                }
                grouped_postings.push(
                    PostingIterator::with_query_weight(
                        token,
                        token_id,
                        position,
                        query_weight,
                        posting,
                        num_docs,
                    )
                    .with_grouped_terms(terms),
                );
                continue;
            };
            if posting.is_empty() {
                if is_and_query || is_phrase_query {
                    return Ok(LoadedPostings::empty());
                }
                continue;
            }

            let query_weight = impact_scorer.query_weight(&token);
            let needs_scorer_upper_bound = !posting.has_impacts();
            let posting = PostingIterator::with_query_weight(
                token,
                token_id,
                position,
                query_weight,
                posting,
                num_docs,
            );
            grouped_postings.push(if needs_scorer_upper_bound {
                posting.with_scorer_upper_bound()
            } else {
                posting
            });
        }

        Ok(LoadedPostings {
            postings: grouped_postings,
            grouped_expansions,
            impact_safe: false,
            exact_scoring_required: true,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn bm25_search_legacy(
        &self,
        docs: &DocSet,
        params: &FtsSearchParams,
        operator: Operator,
        mask: &RowAddrMask,
        postings: Vec<PostingIterator>,
        impact_scorer: Option<Arc<MemBM25Scorer>>,
        metrics: &dyn MetricsCollector,
        shared_threshold: Arc<AtomicU32>,
    ) -> Result<Vec<DocCandidate<u64>>> {
        let documents = LegacyWandDocuments::new(docs, mask);
        self.bm25_search_with_documents(
            &documents,
            params,
            operator,
            postings,
            impact_scorer,
            metrics,
            shared_threshold,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn bm25_search_modern(
        &self,
        lengths: &DocLengths,
        visibility: &DocVisibility,
        params: &FtsSearchParams,
        operator: Operator,
        postings: Vec<PostingIterator>,
        impact_scorer: Option<Arc<MemBM25Scorer>>,
        metrics: &dyn MetricsCollector,
        shared_threshold: Arc<AtomicU32>,
    ) -> Result<Vec<DocCandidate<DocId>>> {
        if visibility.is_all() {
            let documents = ModernWandDocuments::all(lengths);
            self.bm25_search_with_documents(
                &documents,
                params,
                operator,
                postings,
                impact_scorer,
                metrics,
                shared_threshold,
            )
        } else {
            let documents = ModernWandDocuments::filtered(lengths, visibility);
            self.bm25_search_with_documents(
                &documents,
                params,
                operator,
                postings,
                impact_scorer,
                metrics,
                shared_threshold,
            )
        }
    }

    #[instrument(level = "debug", skip_all)]
    #[allow(clippy::too_many_arguments)]
    fn bm25_search_with_documents<D: WandDocuments>(
        &self,
        documents: &D,
        params: &FtsSearchParams,
        operator: Operator,
        postings: Vec<PostingIterator>,
        impact_scorer: Option<Arc<MemBM25Scorer>>,
        metrics: &dyn MetricsCollector,
        shared_threshold: Arc<AtomicU32>,
    ) -> Result<Vec<DocCandidate<D::Candidate>>> {
        if postings.is_empty() {
            return Ok(Vec::new());
        }

        let hits = if let Some(scorer) = impact_scorer {
            let mut wand = Wand::new(operator, postings.into_iter(), documents, scorer)
                .with_shared_threshold(shared_threshold);
            wand.search(params, metrics)?
        } else {
            let scorer = IndexBM25Scorer::new(std::iter::once(self));
            let mut wand = Wand::new(operator, postings.into_iter(), documents, scorer)
                .with_shared_threshold(shared_threshold);
            wand.search(params, metrics)?
        };
        Ok(hits)
    }

    pub async fn into_builder(self) -> Result<InnerBuilder> {
        Ok(self.into_builder_chunked(None, None).await?.0)
    }

    async fn into_builder_chunked(
        self,
        chunk_tokens_override: Option<usize>,
        max_list_children_override: Option<u64>,
    ) -> Result<(InnerBuilder, usize)> {
        let mut builder = InnerBuilder::new_with_posting_tail_codec_and_block_size(
            self.id,
            self.inverted_list.has_positions(),
            self.token_set_format,
            self.inverted_list.posting_tail_codec(),
            self.inverted_list.block_size(),
        );
        builder.tokens = self.tokens.into_mutable();
        builder.docs = self.docs.load_build_docset().await?;

        builder
            .posting_lists
            .reserve_exact(self.inverted_list.len());
        let chunk_count = self
            .inverted_list
            .for_each_posting_list_chunked(
                self.inverted_list.has_positions(),
                chunk_tokens_override,
                max_list_children_override,
                |posting_list| {
                    builder
                        .posting_lists
                        .push(posting_list.into_builder(&builder.docs));
                    Ok(())
                },
            )
            .await?;
        Ok((builder, chunk_count))
    }

    #[cfg(test)]
    pub(super) async fn into_builder_with_chunk_limits(
        self,
        chunk_tokens: usize,
        max_list_children: u64,
    ) -> Result<(InnerBuilder, usize)> {
        self.into_builder_chunked(Some(chunk_tokens), Some(max_list_children))
            .await
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::scalar::inverted::document_tokenizer::DocType;

    fn dictionary(terms: &[&str]) -> TokenSet {
        let mut dictionary = TokenSet::default();
        for term in terms {
            dictionary.add((*term).to_owned());
        }
        dictionary
    }

    fn position_summary(entries: &[(u32, bool)]) -> PositionMatchSummary {
        summarize_position_matches(entries.iter().copied().collect())
    }

    #[test]
    fn position_summary_marks_or_duplicates_for_exact_scoring() {
        let summary = position_summary(&[(0, true), (0, false), (1, false)]);

        assert!(summary.exact_scoring_required);
        assert!(!summary.every_position_matched);
    }

    #[test]
    fn position_summary_requires_a_match_in_every_and_group() {
        let complete = position_summary(&[(0, false), (0, true), (1, true)]);
        let incomplete = position_summary(&[(0, true), (1, false), (1, false)]);

        assert!(complete.exact_scoring_required);
        assert!(complete.every_position_matched);
        assert!(incomplete.exact_scoring_required);
        assert!(!incomplete.every_position_matched);
    }

    #[test]
    fn position_summary_groups_nonadjacent_positions() {
        let summary = position_summary(&[(9, false), (1, true), (4, true), (9, true)]);

        assert!(summary.exact_scoring_required);
        assert!(summary.every_position_matched);
    }

    #[test]
    fn position_summary_spills_past_eight_tokens_without_losing_exactness() {
        let mut positions = SmallVec::<[(u32, bool); 8]>::new();
        positions.extend((0..10).map(|position| (position, true)));
        positions.push((3, false));
        assert!(positions.spilled());

        let summary = summarize_position_matches(positions);
        assert!(summary.exact_scoring_required);
        assert!(summary.every_position_matched);
    }

    #[test]
    fn token_dictionary_or_needs_any_expansion() {
        let tokens = Tokens::with_positions(
            vec!["missing".to_owned(), "alpha".to_owned()],
            vec![0, 1],
            DocType::Text,
        );

        assert!(token_dictionary_may_match(
            &dictionary(&["alpha"]),
            &tokens,
            Operator::Or,
            false,
        ));
        assert!(!token_dictionary_may_match(
            &dictionary(&["other"]),
            &tokens,
            Operator::Or,
            false,
        ));
    }

    #[test]
    fn token_dictionary_and_and_phrase_need_every_original_position() {
        let expanded = Tokens::with_positions(
            vec!["alpha".to_owned(), "alphi".to_owned(), "beta".to_owned()],
            vec![0, 0, 1],
            DocType::Text,
        );
        let complete = dictionary(&["alphi", "beta"]);
        let missing_position = dictionary(&["alpha", "alphi"]);

        assert!(token_dictionary_may_match(
            &complete,
            &expanded,
            Operator::And,
            false,
        ));
        assert!(!token_dictionary_may_match(
            &missing_position,
            &expanded,
            Operator::And,
            false,
        ));
        assert!(!token_dictionary_may_match(
            &missing_position,
            &expanded,
            Operator::Or,
            true,
        ));

        let non_adjacent_positions = Tokens::with_positions(
            vec!["beta".to_owned(), "alpha".to_owned(), "alphi".to_owned()],
            vec![1, 0, 1],
            DocType::Text,
        );
        assert!(token_dictionary_may_match(
            &dictionary(&["alpha", "alphi"]),
            &non_adjacent_positions,
            Operator::And,
            false,
        ));
    }

    #[rstest]
    #[case::plain(false)]
    #[case::v3_compressed(true)]
    fn grouped_union_bound_covers_exact_grouped_score(#[case] compressed: bool) {
        let num_docs = 1_975_725usize;
        let total_tokens = num_docs as u64 * u64::from(u32::MAX);
        let token_names = ["t0", "t1"];
        let token_docs = [1_970_713usize, 334_819];
        let frequencies = [138_872u32, 794_767];
        let doc_length = frequencies.into_iter().sum::<u32>();
        let scorer = Arc::new(MemBM25Scorer::new(
            total_tokens,
            num_docs,
            token_names
                .into_iter()
                .zip(token_docs)
                .map(|(token, docs)| (token.to_owned(), docs))
                .collect(),
        ));
        let query_weights = token_names.map(|token| scorer.query_weight(token));
        let exact_score = query_weights.into_iter().zip(frequencies).fold(
            0.0_f32,
            |score, (query_weight, frequency)| {
                score + query_weight * scorer.doc_weight(frequency, doc_length)
            },
        );
        let query_weight = query_weights.into_iter().sum::<f32>();
        let score_upper_bound = GroupedScoreUpperBound::new(query_weights.into_iter());
        let union_frequency = frequencies.into_iter().sum::<u32>();
        let naive_proxy_bound = query_weight * scorer.doc_weight(union_frequency, doc_length);
        let term_postings = query_weights
            .into_iter()
            .zip(frequencies)
            .map(|(query_weight, frequency)| {
                let max_score = query_weight * scorer.doc_weight(frequency, doc_length);
                if compressed {
                    let mut builder =
                        PostingListBuilder::new_with_block_size(false, MAX_POSTING_BLOCK_SIZE);
                    builder.add(0, PositionRecorder::Count(frequency));
                    let batch = builder.to_batch(vec![max_score]).unwrap();
                    let max_score = batch[MAX_SCORE_COL].as_primitive::<Float32Type>().value(0);
                    let length = batch[LENGTH_COL].as_primitive::<UInt32Type>().value(0);
                    PostingList::from_batch(&batch, Some(max_score), Some(length)).unwrap()
                } else {
                    PostingList::Plain(PlainPostingList::new(
                        ScalarBuffer::from(vec![0u64]),
                        ScalarBuffer::from(vec![frequency as f32]),
                        Some(max_score),
                        None,
                    ))
                }
            })
            .collect::<Vec<_>>();
        let grouped_terms = term_postings
            .iter()
            .zip(query_weights)
            .map(|(posting, query_weight)| GroupedTermScorer::new(query_weight, posting))
            .collect::<Arc<[GroupedTermScorer]>>();
        let mut documents = DocSet::default();
        documents.append(0, doc_length);
        let loaded_documents = LoadedDocLengths::Legacy(Arc::new(documents.clone()));
        let union_posting = InvertedPartition::union_posting_lists(
            term_postings,
            &loaded_documents,
            false,
            score_upper_bound,
            &scorer,
        )
        .unwrap();
        let grouped_bound = union_posting.max_score().unwrap();

        assert_eq!(exact_score.to_bits(), 0x407a_4aa8);
        assert_eq!(naive_proxy_bound.to_bits(), 0x407a_4aa7);
        assert!(grouped_bound >= exact_score);

        let posting = PostingIterator::with_query_weight(
            "group".to_owned(),
            0,
            0,
            query_weight,
            union_posting,
            1,
        )
        .with_grouped_terms(grouped_terms);
        let metrics = NoOpMetricsCollector;
        let params = FtsSearchParams::default();
        let mut cursor = WandCursor::new(
            Operator::Or,
            vec![posting],
            &documents,
            scorer,
            &params,
            &metrics,
        );
        cursor.set_min_competitive_score(exact_score).unwrap();

        assert_eq!(cursor.next().unwrap(), Some(0));
        assert_eq!(
            cursor.current_score().unwrap().to_bits(),
            exact_score.to_bits()
        );
    }
}
