// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use std::sync::Weak;

#[derive(Debug)]
pub(in super::super) struct PartitionCandidates<C> {
    /// Final-scorer IDF of each query position; see [`idf_by_position`].
    pub(super) idf_by_position: Vec<f32>,
    pub(super) grouped_expansions: Vec<GroupedExpansionTerms>,
    pub(super) candidates: Vec<DocCandidate<C>>,
}

/// IDF of the term each posting matches, indexed by query position.
///
/// `scorer` must carry the query's final corpus statistics. Positions without
/// a posting keep 0.0; no candidate reports a frequency for them.
pub(super) fn idf_by_position(postings: &[PostingIterator], scorer: &MemBM25Scorer) -> Vec<f32> {
    let num_positions = postings
        .iter()
        .map(|posting| posting.term_index() as usize + 1)
        .max()
        .unwrap_or_default();
    let mut idf_by_position = vec![0.0_f32; num_positions];
    for posting in postings {
        idf_by_position[posting.term_index() as usize] = scorer.query_weight(posting.token());
    }
    idf_by_position
}

pub(super) struct ModernSearchRequest<'a> {
    pub(super) tokens: Arc<Tokens>,
    pub(super) params: Arc<FtsSearchParams>,
    pub(super) operator: Operator,
    pub(super) mask: Arc<RowAddrMask>,
    pub(super) metrics: Arc<dyn MetricsCollector>,
    /// The query's final scorer, or a clone of it; it both drives WAND and
    /// rescores candidates.
    pub(super) impact_scorer: Arc<MemBM25Scorer>,
    pub(super) limit: usize,
    /// Exclusive raw-score floor used to seed standalone Match WAND.
    pub(super) initial_score_floor: Option<f32>,
    /// Dictionary lookups already done for `tokens` while preparing the
    /// query's statistics, when they were resolved against this segment.
    pub(super) term_ids: Option<SegmentTermIds<'a>>,
}

/// Per-partition token ids of a prepared query's unique terms, recorded while
/// summing document frequencies so search does not look every term up in
/// every partition dictionary a second time.
pub(in super::super) struct PreparedTermIds {
    /// Unique-term ordinal of each final query token.
    term_by_token: Box<[usize]>,
    term_count: usize,
    segments: Vec<PreparedSegmentTermIds>,
}

struct PreparedSegmentTermIds {
    // The weak reference pins the segment's allocation without keeping its
    // dictionaries alive, so no other segment can later occupy that address:
    // a pointer-equal segment at search time is the one these ids came from.
    segment: Weak<InvertedIndex>,
    // Partition-major: `ids[partition_ordinal * term_count + term]`.
    ids: Box<[Option<u32>]>,
}

impl PreparedTermIds {
    pub(in super::super) fn new(term_by_token: Box<[usize]>, term_count: usize) -> Self {
        debug_assert!(term_by_token.iter().all(|&term| term < term_count));
        Self {
            term_by_token,
            term_count,
            segments: Vec::new(),
        }
    }

    pub(in super::super) fn push_segment(
        &mut self,
        segment: &Arc<InvertedIndex>,
        ids: Box<[Option<u32>]>,
    ) -> Result<()> {
        let expected = segment.partitions.len().saturating_mul(self.term_count);
        if ids.len() != expected {
            return Err(Error::internal(format!(
                "resolved FTS token id count is {}, expected {expected} for {} partitions and {} terms",
                ids.len(),
                segment.partitions.len(),
                self.term_count
            )));
        }
        self.segments.push(PreparedSegmentTermIds {
            segment: Arc::downgrade(segment),
            ids,
        });
        Ok(())
    }

    /// Token ids resolved against exactly this segment object for a query
    /// with `token_count` final tokens, if any.
    pub(super) fn for_segment(
        &self,
        segment: &InvertedIndex,
        token_count: usize,
    ) -> Option<SegmentTermIds<'_>> {
        if token_count != self.term_by_token.len() {
            return None;
        }
        self.segments
            .iter()
            .find(|prepared| std::ptr::eq(prepared.segment.as_ptr(), segment))
            .map(|prepared| SegmentTermIds {
                term_by_token: &self.term_by_token,
                term_count: self.term_count,
                ids: &prepared.ids,
            })
    }
}

#[derive(Clone, Copy)]
pub(super) struct SegmentTermIds<'a> {
    term_by_token: &'a [usize],
    term_count: usize,
    ids: &'a [Option<u32>],
}

impl<'a> SegmentTermIds<'a> {
    pub(super) fn partition(&self, partition_ordinal: usize) -> PartitionTermIds<'a> {
        let start = partition_ordinal * self.term_count;
        PartitionTermIds {
            term_by_token: self.term_by_token,
            ids: &self.ids[start..start + self.term_count],
        }
    }
}

/// One partition's view of [`PreparedTermIds`], indexed by final token.
#[derive(Clone, Copy)]
pub(super) struct PartitionTermIds<'a> {
    term_by_token: &'a [usize],
    ids: &'a [Option<u32>],
}

impl PartitionTermIds<'_> {
    pub(super) fn token_id(&self, token_index: usize) -> Option<u32> {
        self.ids[self.term_by_token[token_index]]
    }
}

/// Typed identity for one modern candidate after partition-local scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PartitionDocId {
    pub(super) partition_ordinal: u32,
    pub(super) doc_id: DocId,
}

impl PartitionDocId {
    pub(super) fn try_new(partition_ordinal: usize, doc_id: DocId) -> Result<Self> {
        Ok(Self {
            partition_ordinal: u32::try_from(partition_ordinal).map_err(|_| {
                Error::index(format!(
                    "FTS partition ordinal {partition_ordinal} exceeds candidate identity capacity"
                ))
            })?,
            doc_id,
        })
    }

    pub(super) fn partition_ordinal(self) -> usize {
        self.partition_ordinal as usize
    }
}

#[derive(Debug, Clone)]
pub(super) struct ScoredPartitionDoc {
    pub(super) document: PartitionDocId,
    pub(super) score: OrderedFloat,
}

impl ScoredPartitionDoc {
    fn new(document: PartitionDocId, score: f32) -> Self {
        Self {
            document,
            score: OrderedFloat(score),
        }
    }
}

impl PartialEq for ScoredPartitionDoc {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for ScoredPartitionDoc {}

impl PartialOrd for ScoredPartitionDoc {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredPartitionDoc {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.cmp(&other.score)
    }
}

pub(super) const MAX_CONCURRENT_ADDRESS_READ_BYTES: usize = 64 * 1024 * 1024;

pub(super) fn address_read_concurrency(io_parallelism: usize, largest_read_bytes: usize) -> usize {
    let io_parallelism = io_parallelism.max(1);
    if largest_read_bytes == 0 {
        return io_parallelism;
    }
    io_parallelism.min(
        MAX_CONCURRENT_ADDRESS_READ_BYTES
            .checked_div(largest_read_bytes)
            .unwrap_or(0)
            .max(1),
    )
}

/// Merge one legacy candidate into the global top-k.
///
/// Legacy candidates already carry row addresses, so the heap evicts on the full
/// `(score DESC, row_id ASC)` key and `into_sorted_vec` yields the final order
/// directly. A score tie with a lower row_id wins, a higher one loses, whatever
/// order candidates arrive in.
pub(super) fn push_scored_key(
    candidates: &mut BinaryHeap<Reverse<ScoredDoc>>,
    limit: usize,
    key: u64,
    score: f32,
) {
    // A NaN score is never competitive. It has to be rejected before the total
    // order sees it, because `total_cmp` ranks NaN above every real score, which
    // would put it at the top of the result.
    if score.is_nan() {
        return;
    }
    let candidate = ScoredDoc::new(key, score);
    if candidates.len() < limit {
        candidates.push(Reverse(candidate));
    } else if candidates.peek().is_some_and(|worst| worst.0 < candidate) {
        candidates.pop();
        candidates.push(Reverse(candidate));
    }
}

/// The global top-k of a modern search, as the merge can express it before row
/// addresses are known.
#[derive(Default)]
pub(super) struct ModernCandidates {
    /// The best `limit` candidates by score.
    heap: BinaryHeap<Reverse<ScoredPartitionDoc>>,
    /// Candidates tied at the current k-th score that the heap has no room for.
    /// Cleared whenever the k-th score rises, so it holds just the current tie
    /// band and stays empty for a query without ties.
    tie_band: Vec<ScoredPartitionDoc>,
}

impl ModernCandidates {
    /// Merge one candidate.
    ///
    /// Modern candidates carry dense DocIds whose row addresses are resolved only
    /// after selection, so a k-th-score tie cannot be ordered here. Every tied
    /// candidate is retained instead, and [`rank_resolved_documents`] picks the
    /// exact top-k once the addresses are known.
    pub(super) fn push(&mut self, limit: usize, document: PartitionDocId, score: f32) {
        if score.is_nan() {
            return;
        }
        let candidate = ScoredPartitionDoc::new(document, score);
        if self.heap.len() < limit {
            self.heap.push(Reverse(candidate));
            return;
        }
        let Some(Reverse(worst)) = self.heap.peek() else {
            return;
        };
        let worst_score = worst.score;
        match candidate.score.cmp(&worst_score) {
            // Below the k-th score: never competitive.
            std::cmp::Ordering::Less => {}
            // Tied at the k-th score: a potential winner once addresses resolve.
            std::cmp::Ordering::Equal => self.tie_band.push(candidate),
            std::cmp::Ordering::Greater => {
                let Some(Reverse(displaced)) = self.heap.pop() else {
                    return;
                };
                self.heap.push(Reverse(candidate));
                let kth = self.heap.peek().map(|Reverse(entry)| entry.score);
                if kth.is_some_and(|kth| kth > worst_score) {
                    // The k-th score rose past the band: none of its members can win.
                    self.tie_band.clear();
                } else {
                    self.tie_band.push(displaced);
                }
            }
        }
    }

    /// Candidates held for the final selection: the top-k plus the tie band.
    pub(super) fn buffered(&self) -> usize {
        self.heap.len() + self.tie_band.len()
    }

    /// The candidates whose addresses have to be resolved: the top-k plus the tie
    /// band. Returning more than `limit` is safe, the final ranking truncates.
    pub(super) fn into_vec(self) -> Vec<ScoredPartitionDoc> {
        let mut candidates = self
            .heap
            .into_vec()
            .into_iter()
            .map(|Reverse(candidate)| candidate)
            .collect::<Vec<_>>();
        candidates.extend(self.tie_band);
        candidates
    }

    /// Rebuild from candidates already ordered best first, as the exact top-k with
    /// an empty tie band.
    pub(super) fn from_ranked(ordered: Vec<ScoredPartitionDoc>) -> Self {
        Self {
            heap: ordered.into_iter().map(Reverse).collect(),
            tie_band: Vec::new(),
        }
    }
}

/// Order resolved FTS documents by `(score DESC, row_id ASC)` and keep the exact
/// top-k.
///
/// This is where the deterministic tiebreak is finally settled for modern
/// indexes: the merge above hands over the top-k plus its k-th-score tie group,
/// and the row addresses resolved in between decide which tied documents win.
pub(super) fn rank_resolved_documents(
    mut documents: Vec<ScoredDoc>,
    limit: usize,
) -> Vec<ScoredDoc> {
    documents.sort_unstable_by(|left, right| right.cmp(left));
    documents.truncate(limit);
    documents
}

/// What one WAND search had to give up on its retained tie band.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct WandBandStatus {
    /// See [`Wand::score_floor_overflow`]. When set, the candidates are missing
    /// part of the partition's k-th-score tie band and must not be merged: the
    /// partition is searched again with row addresses attached.
    pub(super) score_floor_overflow: bool,
    /// See [`Wand::band_truncated`]. A retry cannot improve on it, so it is only
    /// reported.
    pub(super) band_truncated: bool,
}

impl WandBandStatus {
    pub(super) fn of<S: Scorer, D: WandDocuments>(wand: &Wand<'_, S, D>) -> Self {
        Self {
            score_floor_overflow: wand.score_floor_overflow(),
            band_truncated: wand.band_truncated(),
        }
    }
}

/// One modern partition's inputs to the CPU scoring phase.
pub(super) struct LoadedModernPartition {
    pub(super) partition_ordinal: usize,
    pub(super) part: Arc<InvertedPartition>,
    pub(super) lengths: Arc<DocLengths>,
    pub(super) visibility: DocVisibility,
    pub(super) postings: FetchedPostings,
    /// Row addresses the walk orders k-th-score ties by. Attached when they are
    /// already in memory, and always on the retry that follows a tie overflow.
    /// Without them the walk defers ties to the global merge.
    pub(super) addresses: Option<ResidentAddressProjection>,
}

/// Load everything the CPU scoring phase needs for one modern partition.
///
/// `Ok(None)` means the partition cannot contribute: nothing visible under the
/// prefilter, or no matching postings.
#[allow(clippy::too_many_arguments)]
pub(super) async fn load_modern_partition(
    partition_ordinal: usize,
    part: Arc<InvertedPartition>,
    tokens: &Tokens,
    params: &FtsSearchParams,
    operator: Operator,
    mask: &Arc<RowAddrMask>,
    metrics: &dyn MetricsCollector,
    term_ids: Option<PartitionTermIds<'_>>,
    load_addresses: bool,
) -> Result<Option<LoadedModernPartition>> {
    let documents = part
        .docs
        .modern()
        .ok_or_else(|| Error::internal("modern index contains legacy partition documents"))?;
    let materialize_selected = operator == Operator::Or
        && mask.max_len().is_some_and(|selected| {
            u128::from(selected).saturating_mul(100)
                <= u128::from(*FLAT_SEARCH_PERCENT_THRESHOLD)
                    .saturating_mul(documents.len() as u128)
        });
    let visibility = match documents.immediate_visibility(mask.clone(), materialize_selected) {
        Some(visibility) => visibility,
        None => {
            documents
                .visibility(mask.clone(), materialize_selected)
                .await?
        }
    };
    if !visibility.may_match().await? {
        return Ok(None);
    }
    let Some(postings) = part
        .fetch_posting_lists(
            tokens,
            params,
            operator,
            metrics,
            PostingLoadOptions::cache_aware_exact(),
            term_ids,
        )
        .await?
    else {
        return Ok(None);
    };
    let lengths = match documents.cached_lengths() {
        Some(lengths) => lengths,
        None => documents.lengths().await?,
    };
    let addresses = if load_addresses {
        // The retry reads the addresses so it can order the tie band exactly with
        // a heap bounded by `limit`. A partition that reaches the final top-k has
        // its addresses resolved either way.
        Some(documents.address_projection().await?)
    } else {
        documents.resident_address_projection()
    };
    Ok(Some(LoadedModernPartition {
        partition_ordinal,
        part,
        lengths,
        visibility,
        postings,
        addresses,
    }))
}

/// Query-wide state every modern partition is scored with.
#[derive(Clone)]
pub(super) struct ModernScoringContext {
    pub(super) tokens: Arc<Tokens>,
    pub(super) params: Arc<FtsSearchParams>,
    pub(super) operator: Operator,
    pub(super) metrics: Arc<dyn MetricsCollector>,
    pub(super) impact_scorer: Arc<MemBM25Scorer>,
    pub(super) impact_shared_threshold: Arc<AtomicU32>,
    pub(super) shared_norm_addends: SharedNormAddends,
}

/// What one modern partition contributes to the global merge.
pub(super) enum ModernPartitionOutcome {
    /// Rescored candidates: the partition's top-k plus its k-th-score tie band.
    Scored {
        partition_ordinal: usize,
        scored: Vec<(DocId, f32)>,
    },
    /// The k-th-score tie band outgrew the collector, so the candidates are not
    /// the exact top-k. The partition must be searched again with addresses.
    Overflow {
        partition_ordinal: usize,
        part: Arc<InvertedPartition>,
    },
}

/// Run one modern partition's WAND search and rescore its candidates. Pure CPU
/// work, so it runs on the compute pool, and the frequency lists it allocates
/// are freed by the same thread.
pub(super) fn score_modern_partition(
    loaded: LoadedModernPartition,
    context: &ModernScoringContext,
) -> Result<Option<ModernPartitionOutcome>> {
    let LoadedModernPartition {
        partition_ordinal,
        part,
        lengths,
        visibility,
        postings,
        addresses,
    } = loaded;
    let LoadedPostings {
        postings,
        grouped_expansions,
        impact_safe,
        exact_scoring_required,
        ..
    } = postings.into_loaded(context.tokens.as_ref(), context.impact_scorer.as_ref())?;
    if postings.is_empty() {
        return Ok(None);
    }
    let use_global_scorer = impact_safe || exact_scoring_required;
    let threshold = if use_global_scorer {
        context.impact_shared_threshold.clone()
    } else {
        Arc::new(AtomicU32::new(f32::NEG_INFINITY.to_bits()))
    };
    let wand_scorer = use_global_scorer.then(|| context.impact_scorer.clone());
    let idf_by_position = idf_by_position(&postings, context.impact_scorer.as_ref());
    let (candidates, band) = part.bm25_search_modern(
        lengths.as_ref(),
        &visibility,
        addresses.as_ref(),
        context.params.as_ref(),
        context.operator,
        postings,
        wand_scorer,
        context.shared_norm_addends.clone(),
        context.metrics.as_ref(),
        threshold,
    )?;
    if band.score_floor_overflow {
        return Ok(Some(ModernPartitionOutcome::Overflow {
            partition_ordinal,
            part,
        }));
    }
    if band.band_truncated {
        debug!(
            partition_ordinal,
            "FTS element tie band truncated; elements of one row past the cap order by DocId"
        );
    }
    let scored = rescore_partition_candidates(
        PartitionCandidates {
            idf_by_position,
            grouped_expansions,
            candidates,
        },
        context.impact_scorer.as_ref(),
    );
    Ok(
        (!scored.is_empty()).then_some(ModernPartitionOutcome::Scored {
            partition_ordinal,
            scored,
        }),
    )
}

pub(super) fn rescore_partition_candidates<C>(
    partition: PartitionCandidates<C>,
    scorer: &MemBM25Scorer,
) -> Vec<(C, f32)> {
    let PartitionCandidates {
        idf_by_position,
        grouped_expansions,
        candidates,
    } = partition;
    let grouped_positions = grouped_expansions
        .iter()
        .map(|group| group.position)
        .collect::<HashSet<_>>();
    let mut position_scores = vec![0.0_f32; idf_by_position.len()];

    candidates
        .into_iter()
        .map(
            |DocCandidate {
                 document,
                 posting_doc_id,
                 freqs,
                 doc_length,
             }| {
                position_scores.fill(0.0);
                for (term_index, freq) in freqs {
                    if grouped_positions.contains(&term_index) {
                        continue;
                    }
                    debug_assert!((term_index as usize) < idf_by_position.len());
                    position_scores[term_index as usize] +=
                        idf_by_position[term_index as usize] * scorer.doc_weight(freq, doc_length);
                }
                for group in &grouped_expansions {
                    debug_assert!((group.position as usize) < position_scores.len());
                    let grouped_score = group
                        .terms
                        .iter()
                        .filter_map(|term| {
                            term.frequency(posting_doc_id).map(|freq| {
                                term.query_weight() * scorer.doc_weight(freq, doc_length)
                            })
                        })
                        .fold(0.0_f32, |sum, score| sum + score);
                    position_scores[group.position as usize] += grouped_score;
                }
                let score = position_scores
                    .iter()
                    .fold(0.0_f32, |sum, score| sum + *score);
                (document, score)
            },
        )
        .collect()
}

#[derive(Debug)]
pub(in super::super) struct LoadedPostings {
    pub(in super::super) postings: Vec<PostingIterator>,
    pub(super) grouped_expansions: Vec<GroupedExpansionTerms>,
    pub(super) impact_safe: bool,
    pub(super) exact_scoring_required: bool,
    #[cfg(test)]
    pub(super) no_impact_fallback: bool,
}

pub(super) enum LoadedDocLengths {
    Legacy(Arc<DocSet>),
    Modern(Arc<DocLengths>),
}

impl LoadedDocLengths {
    pub(super) fn scoring_num_tokens(&self, doc_id: u32) -> u32 {
        match self {
            Self::Legacy(docs) => docs.scoring_num_tokens(doc_id),
            Self::Modern(lengths) => lengths.scoring(DocId::new(doc_id)),
        }
    }

    pub(super) fn num_tokens_by_row_id(&self, row_id: u64) -> u32 {
        match self {
            Self::Legacy(docs) => docs.num_tokens_by_row_id(row_id),
            Self::Modern(_) => unreachable!("modern posting lists use dense DocIds"),
        }
    }
}

impl LoadedPostings {
    pub(super) fn empty() -> Self {
        Self {
            postings: Vec::new(),
            grouped_expansions: Vec::new(),
            impact_safe: false,
            exact_scoring_required: false,
            #[cfg(test)]
            no_impact_fallback: false,
        }
    }
}

#[derive(Debug)]
pub(super) struct GroupedExpansionTerms {
    pub(super) position: u32,
    pub(super) terms: Arc<[GroupedTermScorer]>,
}
