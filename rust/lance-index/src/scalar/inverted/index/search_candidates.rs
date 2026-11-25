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
        self.cmp(other).is_eq()
    }
}

impl Eq for ScoredPartitionDoc {}

impl PartialOrd for ScoredPartitionDoc {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Greater compares better: higher score, then the earlier `(partition, doc
/// id)` coordinate.
///
/// The coordinate tiebreak is what makes this a total order. Score alone left a
/// tie to heap and arrival order, and partitions are searched concurrently, so
/// which of two equal-scoring documents a bounded search kept was not stable
/// run to run -- and a bounded search's result has to be a *prefix* of the
/// order the plan above emits in or paging through a tie skips and repeats
/// rows.
///
/// Candidates are still coordinates here, not addresses, so the tiebreak is the
/// coordinate rather than the address the caller finally emits. The two agree
/// for an index built the only way `InvertedIndexBuilder` builds one: its
/// partitions are filled in scan order, which is ascending row address, so
/// `(partition ordinal, doc id)` ascends with the address.
impl Ord for ScoredPartitionDoc {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .cmp(&other.score)
            .then_with(|| {
                other
                    .document
                    .partition_ordinal
                    .cmp(&self.document.partition_ordinal)
            })
            .then_with(|| other.document.doc_id.cmp(&self.document.doc_id))
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

/// Admit `(key, score)` into a bounded best-`limit` heap, keyed on `(score
/// DESC, key ASC)` -- the order the plan above emits in.
///
/// `ScoredDoc`'s own `Ord` compares the score alone, so ties would otherwise be
/// resolved by heap and arrival order; `key` is the stored identifier the
/// caller emits, so comparing it makes the retained set a prefix of that order
/// and the result stable run to run.
pub(super) fn push_scored_key(
    candidates: &mut BinaryHeap<Reverse<ScoredDoc>>,
    limit: usize,
    key: u64,
    score: f32,
) {
    let beats_worst = |worst: &Reverse<ScoredDoc>| match worst.0.score.0.partial_cmp(&score) {
        Some(std::cmp::Ordering::Less) => true,
        Some(std::cmp::Ordering::Equal) => key < worst.0.row_id,
        _ => false,
    };
    if candidates.len() < limit {
        candidates.push(Reverse(ScoredDoc::new(key, score)));
    } else if candidates.peek().is_some_and(beats_worst) {
        candidates.pop();
        candidates.push(Reverse(ScoredDoc::new(key, score)));
    }
}

pub(super) fn push_scored_partition_doc(
    candidates: &mut BinaryHeap<Reverse<ScoredPartitionDoc>>,
    limit: usize,
    document: PartitionDocId,
    score: f32,
) {
    if candidates.len() < limit {
        candidates.push(Reverse(ScoredPartitionDoc::new(document, score)));
    } else if candidates
        .peek()
        .is_some_and(|candidate| candidate.0.score.0 < score)
    {
        candidates.pop();
        candidates.push(Reverse(ScoredPartitionDoc::new(document, score)));
    }
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
