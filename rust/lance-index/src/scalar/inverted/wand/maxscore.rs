// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bulk MAXSCORE disjunction: clause partitioning, the outer window loop and
//! the inner-window completion kernel.

use lance_core::Result;
use smallvec::SmallVec;

use crate::metrics::MetricsCollector;
use crate::scalar::inverted::builder::{BLOCK_SIZE, ScoredDoc};
use crate::scalar::inverted::encoding::MAX_POSTING_BLOCK_SIZE;
use crate::scalar::inverted::query::FtsSearchParams;
use crate::scalar::inverted::scorer::{Scorer, bm25_doc_weight_with_norm};
use crate::scalar::inverted::{DocInfo, PostingList, RawDocInfo};

use super::bounds::{CompetitiveFloorMode, score_sum_cannot_compete, score_sum_upper_bound_factor};
use super::collector::TopKCollector;
use super::doc_source::{DocCandidate, WandDocuments};
use super::posting::PostingIterator;
use super::{TERMINATED_DOC_ID, Wand};

impl PostingIterator {
    /// Bulk-score every posting in `[current doc, up_to]` into the window
    /// accumulator (slot = doc - window_min) and leave the iterator on the
    /// first doc beyond `up_to`. This is the Lucene `nextDocsAndScores`
    /// equivalent: it walks the decompressed block arrays directly, with no
    /// per-doc heap traffic.
    // The norm cache arg tips this hot-path fn over the limit; bundling the
    // scoring inputs isn't worth the churn here.
    #[allow(clippy::too_many_arguments)]
    fn collect_window_scores<S: Scorer + ?Sized, D: WandDocuments>(
        &mut self,
        window_min: u64,
        up_to: u64,
        clause_idx: usize,
        docs: &D,
        scorer: &S,
        norm_k: Option<(&[u8], &[f32; 256])>,
        acc: &mut WindowAccumulator,
    ) {
        if self.doc().is_some_and(|doc| doc.doc_id() < window_min) {
            self.next(window_min);
        }
        match self.list {
            PostingList::Compressed(ref list) => {
                let shift = list.block_shift();
                let mask = list.block_mask();
                'blocks: while let Some(doc) = self.current_doc {
                    if doc.doc_id() > up_to {
                        break;
                    }
                    let block_idx = self.index >> shift;
                    let block_offset = self.index & mask;
                    let compressed =
                        unsafe { &mut *self.ensure_compressed_block_ptr(list, block_idx) };
                    for offset in block_offset..compressed.doc_ids.len() {
                        let doc_id = compressed.doc_ids[offset];
                        if u64::from(doc_id) > up_to {
                            self.index = (block_idx << shift) + offset;
                            self.block_idx = block_idx;
                            self.current_doc = Some(DocInfo::Raw(RawDocInfo {
                                doc_id,
                                frequency: compressed.freqs[offset],
                            }));
                            break 'blocks;
                        }
                        let freq = compressed.freqs[offset];
                        // One byte-norm load plus a cached addend replaces
                        // recomputing the BM25 denominator per doc.
                        let doc_weight = match norm_k {
                            Some((norms, cache)) => bm25_doc_weight_with_norm(
                                freq,
                                cache[norms[doc_id as usize] as usize],
                            ),
                            None => scorer.doc_weight(freq, docs.scoring_num_tokens(doc_id)),
                        };
                        let score = self.query_weight * doc_weight;
                        let slot = (u64::from(doc_id) - window_min) as usize;
                        acc.add(clause_idx, slot, score, freq);
                    }
                    // Block exhausted: step into the next block (or finish).
                    let next_start = (block_idx + 1) << shift;
                    if next_start >= list.length as usize {
                        self.index = list.length as usize;
                        self.block_idx = self.index >> shift;
                        self.current_doc = None;
                        break;
                    }
                    self.index = next_start;
                    self.block_idx = block_idx + 1;
                    let compressed =
                        unsafe { &mut *self.ensure_compressed_block_ptr(list, block_idx + 1) };
                    self.current_doc = Some(DocInfo::Raw(RawDocInfo {
                        doc_id: compressed.doc_ids[0],
                        frequency: compressed.freqs[0],
                    }));
                }
            }
            PostingList::Plain(_) => {
                while let Some(doc) = self.doc() {
                    let doc_id = doc.doc_id();
                    if doc_id > up_to {
                        break;
                    }
                    let doc_length = docs.doc_length(&doc);
                    let score = self.score(scorer, doc.frequency(), doc_length);
                    let slot = (doc_id - window_min) as usize;
                    acc.add(clause_idx, slot, score, doc.frequency());
                    self.next(doc_id + 1);
                }
            }
        }
    }
}

/// Inner window span (in doc ids) of the bulk MAXSCORE path. Same as Lucene's
/// `MaxScoreBulkScorer.INNER_WINDOW_SIZE`.
pub(super) const MAXSCORE_INNER_WINDOW: usize = 1 << 12;

/// One MAXSCORE clause: posting plus the current window bound / prefix used
/// to pick essential vs optional and to complete scores in query order.
pub(super) struct MaxScoreClause {
    pub(super) posting: Box<PostingIterator>,
    pub(super) query_rank: usize,
    pub(super) bound: f32,
    pub(super) prefix_bound: f64,
}

/// Per-window score/frequency accumulator for the bulk MAXSCORE path. Slot i
/// covers doc id `window_min + i`; `freqs` is laid out clause-major so accept
/// paths can recover (term, freq) pairs of the essential clauses.
struct WindowAccumulator {
    scores: Vec<f32>,
    freqs: Vec<u32>,
    words: Vec<u64>,
    num_clauses: usize,
}

impl WindowAccumulator {
    fn new(num_clauses: usize) -> Self {
        Self {
            scores: vec![0.0; MAXSCORE_INNER_WINDOW],
            freqs: vec![0; num_clauses * MAXSCORE_INNER_WINDOW],
            words: vec![0; MAXSCORE_INNER_WINDOW / 64],
            num_clauses,
        }
    }

    #[inline]
    fn add(&mut self, clause_idx: usize, slot: usize, score: f32, freq: u32) {
        self.scores[slot] += score;
        // Doc-major layout: one slot's clause frequencies share a cache line.
        self.freqs[slot * self.num_clauses + clause_idx] = freq;
        self.words[slot >> 6] |= 1u64 << (slot & 63);
    }

    #[inline]
    fn clause_freq(&self, clause_idx: usize, slot: usize) -> u32 {
        self.freqs[slot * self.num_clauses + clause_idx]
    }

    #[inline]
    fn clear_slot(&mut self, slot: usize) {
        self.scores[slot] = 0.0;
        self.freqs[slot * self.num_clauses..(slot + 1) * self.num_clauses].fill(0);
    }
}

impl<'a, S: Scorer, D: WandDocuments> Wand<'a, S, D> {
    /// Bulk MAXSCORE top-k disjunction, mirroring Lucene's MaxScoreBulkScorer.
    ///
    /// Per outer window (bounded by the essential clauses' block boundaries):
    /// clauses are partitioned into a non-essential prefix — sorted by window
    /// max score, as many as fit under the threshold — and the essential rest.
    /// Essential clauses stream their postings into a window accumulator in
    /// bulk; only accumulated candidates that could still beat the threshold
    /// probe the non-essential clauses. No per-doc heap maintenance happens
    /// anywhere on this path.
    pub(super) fn maxscore_search(
        &mut self,
        params: &FtsSearchParams,
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<DocCandidate<D::Candidate>>> {
        let limit = params.limit.unwrap_or(usize::MAX);
        let mut clauses = std::mem::take(&mut self.head)
            .into_vec()
            .into_iter()
            .map(|head| MaxScoreClause {
                posting: head.posting,
                query_rank: 0,
                bound: 0.0,
                prefix_bound: 0.0,
            })
            .collect::<Vec<_>>();
        // Before a competitive floor exists every clause is essential. Keep
        // that exhaustive accumulator in canonical query order so its score
        // can be emitted directly instead of recomputing every contribution.
        clauses.sort_unstable_by_key(|clause| (clause.posting.position, clause.posting.token_id));
        for (query_rank, clause) in clauses.iter_mut().enumerate() {
            clause.query_rank = query_rank;
        }
        let num_query_terms = clauses.len();
        let total_sum_upper_bound_factor = score_sum_upper_bound_factor(num_query_terms);

        let mut acc = WindowAccumulator::new(clauses.len());
        let mut essential_chunk = Vec::with_capacity(MAX_POSTING_BLOCK_SIZE);
        let mut soa_scratch = MaxScoreSoaScratch::new();
        let mut candidates = TopKCollector::new(limit, std::cmp::min(limit, BLOCK_SIZE * 10));
        let norm_k = self.norm_k_cache();
        let norm_k_ref = norm_k
            .as_ref()
            .map(|(norms, cache)| (*norms, cache.as_ref()));
        let mut num_comparisons = 0usize;
        // Adaptive minimum window size (Lucene): grow windows when they yield
        // too few candidates to amortize the per-window bound computations.
        let mut min_window_size = 1u64;
        let mut num_windows = 0u64;
        let mut prev_first_essential = 0usize;

        let mut window_min = clauses
            .iter()
            .filter_map(|clause| clause.posting.doc().map(|doc| doc.doc_id()))
            .min()
            .unwrap_or(TERMINATED_DOC_ID);

        while window_min < TERMINATED_DOC_ID {
            clauses.retain(|clause| clause.posting.doc().is_some());
            if clauses.is_empty() {
                break;
            }
            self.raise_to_shared_floor(params.wand_factor);

            // Window boundary from the previous window's essential clauses
            // only: dense non-essential clauses must not fragment the window.
            let first_window_lead = prev_first_essential.min(clauses.len() - 1);
            let mut window_max = TERMINATED_DOC_ID;
            for clause in &mut clauses {
                let doc = clause
                    .posting
                    .doc()
                    .map(|doc| doc.doc_id())
                    .expect("exhausted clauses were retained out");
                clause.posting.shallow_next(doc.max(window_min));
            }
            for clause in &clauses[first_window_lead..] {
                window_max = window_max.min(clause.posting.block_end_doc());
            }
            if clauses.len() > 1 {
                // Target at least 32 candidates per clause per window on
                // average before shrinking windows back to block granularity.
                if (num_comparisons as u64) < num_windows * 32 * clauses.len() as u64 {
                    min_window_size = (min_window_size * 2).min(MAXSCORE_INNER_WINDOW as u64);
                } else {
                    min_window_size = 1;
                }
                window_max = window_max.max(window_min.saturating_add(min_window_size - 1));
            }

            for clause in &mut clauses {
                let doc = clause
                    .posting
                    .doc()
                    .map(|doc| doc.doc_id())
                    .expect("exhausted clauses were retained out");
                clause.bound = if doc > window_max {
                    0.0
                } else {
                    clause
                        .posting
                        .block_max_score_up_to_with_stats(window_max, &self.scorer)
                        .score
                };
            }
            let mut clauses_in_query_order = true;
            let mut first_essential = 0;
            let mut prefix = 0.0_f64;
            if self.threshold > 0.0 {
                clauses.sort_unstable_by(|a, b| a.bound.total_cmp(&b.bound));
                clauses_in_query_order = clauses
                    .windows(2)
                    .all(|window| window[0].query_rank <= window[1].query_rank);
                for (i, clause) in clauses.iter_mut().enumerate() {
                    let next_prefix = prefix + f64::from(clause.bound);
                    let widened = next_prefix * score_sum_upper_bound_factor(i + 1);
                    if widened > f64::from(self.threshold) {
                        break;
                    }
                    prefix = next_prefix;
                    clause.prefix_bound = prefix;
                    first_essential = i + 1;
                }
            }
            prev_first_essential = first_essential;
            num_windows += 1;

            if first_essential == clauses.len() {
                // No clause combination inside this window can beat the
                // threshold: skip it wholesale.
                window_min = match window_max {
                    TERMINATED_DOC_ID => TERMINATED_DOC_ID,
                    max => max + 1,
                };
                // Single live clause: instead of re-running the window
                // machinery once per block, scan the baked per-block bounds
                // for the next block that can beat the threshold. This is the
                // slab form of Lucene's getSkipUpTo and turns dead stretches
                // of a dominant term into a tight load-mul-compare loop.
                if clauses.len() == 1 && window_min != TERMINATED_DOC_ID && self.threshold > 0.0 {
                    let posting = &clauses[0].posting;
                    if let PostingList::Compressed(ref list) = posting.list
                        && let Some(impacts) = list.impacts.as_ref()
                    {
                        let compressed = unsafe { &mut *posting.compressed_state_ptr() };
                        let bounds = impacts.level0_doc_weight_bounds_cached(
                            &self.scorer,
                            &mut compressed.block_max_window.impact_score_cache,
                        );
                        let query_weight = posting.query_weight;
                        // Position by binary search on the first-doc slab: the
                        // deep cursor lags arbitrarily far behind during long
                        // skip runs, and walking from it re-scans the same
                        // blocks on every dead window.
                        let first_docs = list.block_first_docs();
                        let mut block_idx = first_docs
                            .partition_point(|&first| u64::from(first) <= window_min)
                            .saturating_sub(1);
                        while block_idx < bounds.len()
                            && query_weight * bounds[block_idx] <= self.threshold
                        {
                            block_idx += 1;
                        }
                        window_min = if block_idx < bounds.len() {
                            window_min.max(u64::from(list.block_least_doc_id(block_idx)))
                        } else {
                            TERMINATED_DOC_ID
                        };
                    }
                }
                continue;
            }

            let total_non_essential_bound = if first_essential > 0 {
                clauses[first_essential - 1].prefix_bound
            } else {
                0.0
            };

            // Single essential clause (the common case once the threshold is
            // competitive): skip dead blocks, take one posting block at a
            // time, and complete ≤2 optionals in SoA buffers. More optionals
            // keep the per-doc LiveHit completion below.
            if first_essential + 1 == clauses.len() {
                #[cfg(test)]
                {
                    self.maxscore_single_essential_windows += 1;
                }
                if first_essential <= 2 {
                    let essential_query_rank = clauses[first_essential].query_rank;
                    let essential_term = clauses[first_essential].posting.term_index();
                    let essential_weight = clauses[first_essential].posting.query_weight;
                    let essential_window_bound = clauses[first_essential].bound;
                    if clauses[first_essential]
                        .posting
                        .doc()
                        .is_some_and(|doc| doc.doc_id() < window_min)
                    {
                        clauses[first_essential].posting.next(window_min);
                    }
                    self.score_single_essential_upto(
                        &mut clauses,
                        first_essential,
                        first_essential,
                        window_max,
                        essential_query_rank,
                        essential_term,
                        essential_weight,
                        essential_window_bound,
                        num_query_terms,
                        total_non_essential_bound,
                        total_sum_upper_bound_factor,
                        norm_k_ref,
                        &mut essential_chunk,
                        &mut soa_scratch,
                        &mut candidates,
                        &mut num_comparisons,
                        params.wand_factor,
                    )?;
                    window_min = match window_max {
                        TERMINATED_DOC_ID => TERMINATED_DOC_ID,
                        max => max + 1,
                    };
                    continue;
                }
                let (non_essential, essential) = clauses.split_at_mut(first_essential);
                let essential_query_rank = essential[0].query_rank;
                let posting = &mut essential[0].posting;
                if posting.doc().is_some_and(|doc| doc.doc_id() < window_min) {
                    posting.next(window_min);
                }
                let essential_term = posting.term_index();
                let essential_weight = posting.query_weight;
                let sole_query_clause = non_essential.is_empty();

                macro_rules! consider_candidate {
                    ($doc:expr, $freq:expr) => {{
                        let doc = $doc;
                        let freq = $freq;
                        num_comparisons += 1;
                        // One byte-norm load + cached addend when available;
                        // the exact doc length is only needed at insert time.
                        let norm_addend =
                            norm_k_ref.map(|(norms, cache)| cache[norms[doc as usize] as usize]);
                        let score = match norm_addend {
                            Some(addend) => {
                                essential_weight * bm25_doc_weight_with_norm(freq, addend)
                            }
                            None => {
                                essential_weight
                                    * self.scorer.doc_weight(
                                        freq,
                                        self.documents.scoring_num_tokens(doc as u32),
                                    )
                            }
                        };
                        if !(self.threshold > 0.0
                            && score_sum_cannot_compete(
                                score,
                                total_non_essential_bound,
                                self.threshold,
                                total_sum_upper_bound_factor,
                                CompetitiveFloorMode::Exclusive,
                            ))
                        {
                            if let Some(document_key) =
                                self.documents.document_key_for_doc_id(doc as u32)
                            {
                                let mut total = score;
                                let mut scores_by_query_rank = SmallVec::<[f32; 8]>::new();
                                if !sole_query_clause {
                                    scores_by_query_rank.resize(num_query_terms, 0.0);
                                    scores_by_query_rank[essential_query_rank] = score;
                                }
                                let mut rejected = false;
                                for i in (0..non_essential.len()).rev() {
                                    if self.threshold > 0.0
                                        && score_sum_cannot_compete(
                                            total,
                                            non_essential[i].prefix_bound,
                                            self.threshold,
                                            total_sum_upper_bound_factor,
                                            CompetitiveFloorMode::Exclusive,
                                        )
                                    {
                                        rejected = true;
                                        break;
                                    }
                                    let query_rank = non_essential[i].query_rank;
                                    let probe = &mut non_essential[i].posting;
                                    if probe.doc().is_some_and(|d| d.doc_id() < doc) {
                                        probe.next(doc);
                                    }
                                    if let Some(d) = probe.doc()
                                        && d.doc_id() == doc
                                    {
                                        let contribution = match norm_addend {
                                            Some(addend) => {
                                                probe.query_weight
                                                    * bm25_doc_weight_with_norm(
                                                        d.frequency(),
                                                        addend,
                                                    )
                                            }
                                            None => probe.score(
                                                &self.scorer,
                                                d.frequency(),
                                                self.documents.scoring_num_tokens(doc as u32),
                                            ),
                                        };
                                        total += contribution;
                                        scores_by_query_rank[query_rank] = contribution;
                                    }
                                }

                                // Apply the caller's floor mode only after the
                                // canonical query-order score is available.
                                if !rejected {
                                    let canonical_score = if sole_query_clause {
                                        score
                                    } else {
                                        scores_by_query_rank
                                            .into_iter()
                                            .fold(0.0_f32, |sum, contribution| sum + contribution)
                                    };
                                    if canonical_score > self.threshold {
                                        let doc_length =
                                            self.documents.scoring_num_tokens(doc as u32);
                                        if candidates.insert(
                                            ScoredDoc::new(document_key, canonical_score),
                                            doc_length,
                                            doc,
                                            std::iter::once((essential_term, freq)).chain(
                                                non_essential.iter().filter_map(|clause| {
                                                    clause.posting.doc().and_then(|d| {
                                                        (d.doc_id() == doc).then(|| {
                                                            (
                                                                clause.posting.term_index(),
                                                                d.frequency(),
                                                            )
                                                        })
                                                    })
                                                }),
                                            ),
                                        )? && let Some(kth) = candidates.kth_score_if_full()
                                        {
                                            self.update_threshold(kth, params.wand_factor);
                                        }
                                    }
                                }
                            }
                        }
                    }};
                }

                match posting.list {
                    PostingList::Compressed(ref list) => {
                        let shift = list.block_shift();
                        let mask = list.block_mask();
                        'stream: while let Some(cur) = posting.current_doc {
                            if cur.doc_id() > window_max {
                                break;
                            }
                            let block_idx = posting.index >> shift;
                            let block_offset = posting.index & mask;
                            let compressed = unsafe {
                                &mut *posting.ensure_compressed_block_ptr(list, block_idx)
                            };
                            for offset in block_offset..compressed.doc_ids.len() {
                                let doc_id = compressed.doc_ids[offset];
                                if u64::from(doc_id) > window_max {
                                    posting.index = (block_idx << shift) + offset;
                                    posting.block_idx = block_idx;
                                    posting.current_doc = Some(DocInfo::Raw(RawDocInfo {
                                        doc_id,
                                        frequency: compressed.freqs[offset],
                                    }));
                                    break 'stream;
                                }
                                consider_candidate!(u64::from(doc_id), compressed.freqs[offset]);
                            }
                            let next_start = (block_idx + 1) << shift;
                            if next_start >= list.length as usize {
                                posting.index = list.length as usize;
                                posting.block_idx = posting.index >> shift;
                                posting.current_doc = None;
                                break;
                            }
                            posting.index = next_start;
                            posting.block_idx = block_idx + 1;
                            let compressed = unsafe {
                                &mut *posting.ensure_compressed_block_ptr(list, block_idx + 1)
                            };
                            posting.current_doc = Some(DocInfo::Raw(RawDocInfo {
                                doc_id: compressed.doc_ids[0],
                                frequency: compressed.freqs[0],
                            }));
                        }
                    }
                    PostingList::Plain(_) => {
                        while let Some(cur) = posting.doc() {
                            let doc = cur.doc_id();
                            if doc > window_max {
                                break;
                            }
                            consider_candidate!(doc, cur.frequency());
                            posting.next(doc + 1);
                        }
                    }
                }

                window_min = match window_max {
                    TERMINATED_DOC_ID => TERMINATED_DOC_ID,
                    max => max + 1,
                };
                continue;
            }

            // Stream the essential clauses through inner windows.
            #[cfg(test)]
            {
                self.maxscore_general_windows += 1;
            }
            let mut inner_min = window_min;
            loop {
                // Lucene `scoreInnerWindow`: when the second essential is at
                // least INNER_WINDOW/2 ahead of the first, the prefix matches
                // a single clause. Complete it with the SoA path instead of
                // scattering two sparse lists into a 4096-slot accumulator.
                //
                // One pass serves both purposes: `top_doc` is also the closest
                // essential cursor, which is what raises `inner_min` to the
                // window floor.
                let mut top_idx = None;
                let mut top_doc = TERMINATED_DOC_ID;
                let mut top2_doc = TERMINATED_DOC_ID;
                for (idx, clause) in clauses.iter().enumerate().skip(first_essential) {
                    let Some(doc) = clause.posting.doc() else {
                        continue;
                    };
                    let id = doc.doc_id();
                    if id < top_doc {
                        top2_doc = top_doc;
                        top_doc = id;
                        top_idx = Some(idx);
                    } else if id < top2_doc {
                        top2_doc = id;
                    }
                }
                inner_min = inner_min.max(top_doc);
                if inner_min == TERMINATED_DOC_ID || inner_min > window_max {
                    break;
                }
                if let Some(ess_idx) = top_idx
                    && top2_doc.saturating_sub(top_doc) >= (MAXSCORE_INNER_WINDOW / 2) as u64
                    && first_essential <= 2
                    // Reachable, not merely defensive. `shallow_next` only moves
                    // `block_idx` and leaves `current_doc` where it was, so a
                    // window that ends early -- wholesale skip, `inner_min >
                    // window_max`, or the single-essential stream -- can advance
                    // `window_min` past cursors that were never consumed. The
                    // next window then starts with `inner_min` above the smallest
                    // essential doc. Since `upto` is `top2_doc - 1`, the span is
                    // empty when `top2_doc == inner_min` (the call would return
                    // without advancing, leaving `inner_min` stuck) and ends
                    // below `inner_min` when `top2_doc < inner_min` (it would
                    // move `inner_min` backwards onto a span this window has
                    // already passed). Both fall through to the general path.
                    && {
                        let non_empty_span = top2_doc > inner_min;
                        #[cfg(test)]
                        if !non_empty_span {
                            self.maxscore_top2_gap_empty_span += 1;
                        }
                        non_empty_span
                    }
                {
                    let upto = window_max.min(top2_doc.saturating_sub(1));
                    if clauses[ess_idx]
                        .posting
                        .doc()
                        .is_some_and(|doc| doc.doc_id() < inner_min)
                    {
                        clauses[ess_idx].posting.next(inner_min);
                    }
                    let essential_query_rank = clauses[ess_idx].query_rank;
                    let essential_term = clauses[ess_idx].posting.term_index();
                    let essential_weight = clauses[ess_idx].posting.query_weight;
                    let essential_window_bound = clauses[ess_idx].bound;
                    #[cfg(test)]
                    {
                        self.maxscore_single_essential_windows += 1;
                        self.maxscore_top2_gap_windows += 1;
                    }
                    self.score_single_essential_upto(
                        &mut clauses,
                        first_essential,
                        ess_idx,
                        upto,
                        essential_query_rank,
                        essential_term,
                        essential_weight,
                        essential_window_bound,
                        num_query_terms,
                        total_non_essential_bound,
                        total_sum_upper_bound_factor,
                        norm_k_ref,
                        &mut essential_chunk,
                        &mut soa_scratch,
                        &mut candidates,
                        &mut num_comparisons,
                        params.wand_factor,
                    )?;
                    inner_min = match upto {
                        TERMINATED_DOC_ID => TERMINATED_DOC_ID,
                        max => max + 1,
                    };
                    continue;
                }
                let inner_max =
                    window_max.min(inner_min.saturating_add(MAXSCORE_INNER_WINDOW as u64 - 1));

                for (clause_idx, clause) in clauses.iter_mut().enumerate().skip(first_essential) {
                    clause.posting.collect_window_scores(
                        inner_min,
                        inner_max,
                        clause_idx,
                        self.documents,
                        &self.scorer,
                        norm_k_ref,
                        &mut acc,
                    );
                }

                // Drain candidates in doc order, completing them with the
                // non-essential clauses ordered by descending bound.
                for word_idx in 0..acc.words.len() {
                    let mut word = acc.words[word_idx];
                    if word == 0 {
                        continue;
                    }
                    acc.words[word_idx] = 0;
                    while word != 0 {
                        let bit = word.trailing_zeros() as usize;
                        word &= word - 1;
                        let slot = (word_idx << 6) | bit;
                        let doc = inner_min + slot as u64;
                        let mut score = acc.scores[slot];
                        num_comparisons += 1;

                        if self.threshold > 0.0
                            && score_sum_cannot_compete(
                                score,
                                total_non_essential_bound,
                                self.threshold,
                                total_sum_upper_bound_factor,
                                CompetitiveFloorMode::Exclusive,
                            )
                        {
                            acc.clear_slot(slot);
                            continue;
                        }

                        let Some(document_key) = self.documents.document_key_for_doc_id(doc as u32)
                        else {
                            acc.clear_slot(slot);
                            continue;
                        };

                        // Doc length is only needed at heap-insert time; the
                        // non-essential completion scores go through the norm
                        // cache when available.
                        let norm_addend =
                            norm_k_ref.map(|(norms, cache)| cache[norms[doc as usize] as usize]);
                        let mut doc_length_cell: Option<u32> = None;
                        let needs_canonical_rescore =
                            first_essential != 0 || !clauses_in_query_order;
                        let mut scores_by_query_rank = SmallVec::<[f32; 8]>::new();
                        if needs_canonical_rescore {
                            scores_by_query_rank.resize(num_query_terms, 0.0);
                        }
                        let mut rejected = false;
                        for i in (0..first_essential).rev() {
                            if self.threshold > 0.0
                                && score_sum_cannot_compete(
                                    score,
                                    clauses[i].prefix_bound,
                                    self.threshold,
                                    total_sum_upper_bound_factor,
                                    CompetitiveFloorMode::Exclusive,
                                )
                            {
                                rejected = true;
                                break;
                            }
                            let query_rank = clauses[i].query_rank;
                            let posting = &mut clauses[i].posting;
                            if posting.doc().is_some_and(|d| d.doc_id() < doc) {
                                posting.next(doc);
                            }
                            if let Some(d) = posting.doc()
                                && d.doc_id() == doc
                            {
                                let contribution = match norm_addend {
                                    Some(addend) => {
                                        posting.query_weight
                                            * bm25_doc_weight_with_norm(d.frequency(), addend)
                                    }
                                    None => {
                                        let doc_length =
                                            *doc_length_cell.get_or_insert_with(|| {
                                                self.documents.scoring_num_tokens(doc as u32)
                                            });
                                        posting.score(&self.scorer, d.frequency(), doc_length)
                                    }
                                };
                                score += contribution;
                                if needs_canonical_rescore {
                                    scores_by_query_rank[query_rank] = contribution;
                                }
                            }
                        }

                        if !rejected {
                            let doc_length = doc_length_cell
                                .unwrap_or_else(|| self.documents.scoring_num_tokens(doc as u32));
                            let score = if !needs_canonical_rescore {
                                // `collect_window_scores` visited every
                                // essential clause in query order, so the
                                // accumulator already contains the canonical
                                // f32 fold. Avoid rescoring exhaustive hits.
                                score
                            } else {
                                for (i, clause) in clauses.iter().enumerate().skip(first_essential)
                                {
                                    let freq = acc.clause_freq(i, slot);
                                    if freq == 0 {
                                        continue;
                                    }
                                    let contribution = match norm_addend {
                                        Some(addend) => {
                                            clause.posting.query_weight
                                                * bm25_doc_weight_with_norm(freq, addend)
                                        }
                                        None => {
                                            clause.posting.score(&self.scorer, freq, doc_length)
                                        }
                                    };
                                    scores_by_query_rank[clause.query_rank] = contribution;
                                }
                                scores_by_query_rank
                                    .iter()
                                    .fold(0.0_f32, |sum, contribution| sum + contribution)
                            };
                            if score <= self.threshold {
                                acc.clear_slot(slot);
                                continue;
                            }
                            if candidates.insert(
                                ScoredDoc::new(document_key, score),
                                doc_length,
                                doc,
                                clauses.iter().enumerate().filter_map(|(i, clause)| {
                                    if i >= first_essential {
                                        let freq = acc.clause_freq(i, slot);
                                        (freq > 0).then(|| (clause.posting.term_index(), freq))
                                    } else {
                                        clause.posting.doc().and_then(|d| {
                                            (d.doc_id() == doc).then(|| {
                                                (clause.posting.term_index(), d.frequency())
                                            })
                                        })
                                    }
                                }),
                            )? && let Some(kth) = candidates.kth_score_if_full()
                            {
                                self.update_threshold(kth, params.wand_factor);
                            }
                        }
                        acc.clear_slot(slot);
                    }
                }
                if inner_max >= window_max {
                    break;
                }
                inner_min = inner_max + 1;
            }

            window_min = match window_max {
                TERMINATED_DOC_ID => TERMINATED_DOC_ID,
                max => max + 1,
            };
        }

        metrics.record_comparisons(num_comparisons);

        candidates.into_candidates(|key| self.documents.candidate_from_key(key))
    }
}

// Inner-window work for MAXSCORE: block skip, two-pointer optional freqs,
// and SoA completion for one essential plus at most two optionals.
//
// Lucene's `scoreInnerWindowSingleEssentialClause`. Two-essential compact
// union is intentionally not ported.

#[inline]
pub(super) fn exclusive_cannot_compete(
    partial: f32,
    remaining_upper_bound: f64,
    floor: f32,
    upper_bound_factor: f64,
) -> bool {
    floor > 0.0
        && score_sum_cannot_compete(
            partial,
            remaining_upper_bound,
            floor,
            upper_bound_factor,
            CompetitiveFloorMode::Exclusive,
        )
}

fn bm25_tf_from_caches(
    query_weight: f32,
    freq: u32,
    doc: u32,
    quantized: Option<(&[u8], &[f32; 256])>,
) -> Option<f32> {
    let (norms, cache) = quantized?;
    Some(query_weight * bm25_doc_weight_with_norm(freq, cache[norms[doc as usize] as usize]))
}

fn bulk_bm25_tf(
    query_weight: f32,
    hits: &[(u64, u32)],
    quantized: Option<(&[u8], &[f32; 256])>,
    fallback: impl Fn(u64, u32) -> f32,
    out: &mut Vec<f32>,
) {
    out.clear();
    out.reserve(hits.len());
    if let Some((norms, cache)) = quantized {
        for &(doc, freq) in hits {
            out.push(
                query_weight * bm25_doc_weight_with_norm(freq, cache[norms[doc as usize] as usize]),
            );
        }
        return;
    }
    for &(doc, freq) in hits {
        out.push(fallback(doc, freq));
    }
}

/// Lucene `DocAndScoreAccBuffer`: SoA vectors reused across essential blocks.
pub(super) struct MaxScoreSoaScratch {
    docs: Vec<u64>,
    ess_scores: Vec<f32>,
    ess_freqs: Vec<u32>,
    raw_scores: Vec<f32>,
    opt_scores: Vec<f32>,
    opt_freqs: Vec<u32>,
    opt_s: [Vec<f32>; 2],
    opt_f: [Vec<u32>; 2],
    partial: Vec<f32>,
}

impl MaxScoreSoaScratch {
    pub(super) fn new() -> Self {
        // One decompressed posting block is the largest chunk
        // `take_docs_one_block_upto` can hand over, so sizing to it keeps the
        // dense case off the allocator's path.
        Self {
            docs: Vec::with_capacity(MAX_POSTING_BLOCK_SIZE),
            ess_scores: Vec::with_capacity(MAX_POSTING_BLOCK_SIZE),
            ess_freqs: Vec::with_capacity(MAX_POSTING_BLOCK_SIZE),
            raw_scores: Vec::with_capacity(MAX_POSTING_BLOCK_SIZE),
            opt_scores: Vec::with_capacity(MAX_POSTING_BLOCK_SIZE),
            opt_freqs: Vec::with_capacity(MAX_POSTING_BLOCK_SIZE),
            opt_s: [
                Vec::with_capacity(MAX_POSTING_BLOCK_SIZE),
                Vec::with_capacity(MAX_POSTING_BLOCK_SIZE),
            ],
            opt_f: [
                Vec::with_capacity(MAX_POSTING_BLOCK_SIZE),
                Vec::with_capacity(MAX_POSTING_BLOCK_SIZE),
            ],
            partial: Vec::with_capacity(MAX_POSTING_BLOCK_SIZE),
        }
    }
}

impl PostingIterator {
    /// Lucene `ScorerUtil.applyOptionalClause`: fill `out_freqs[i]` with this
    /// clause's frequency at sorted `docs[i]`, or 0 on a miss.
    pub(super) fn merge_optional_freqs(&mut self, docs: &[u64], out_freqs: &mut [u32]) {
        debug_assert_eq!(docs.len(), out_freqs.len());
        out_freqs.fill(0);
        if docs.is_empty() {
            return;
        }
        if self.doc().is_some_and(|cur| cur.doc_id() < docs[0]) {
            self.next(docs[0]);
        }
        if matches!(self.list, PostingList::Plain(_)) {
            for (i, &doc) in docs.iter().enumerate() {
                if self.doc().is_some_and(|cur| cur.doc_id() < doc) {
                    self.next(doc);
                }
                if let Some(cur) = self.doc()
                    && cur.doc_id() == doc
                {
                    out_freqs[i] = cur.frequency();
                }
            }
            return;
        }
        let shift = match &self.list {
            PostingList::Compressed(list) => list.block_shift(),
            PostingList::Plain(_) => return,
        };
        let length = match &self.list {
            PostingList::Compressed(list) => list.length as usize,
            PostingList::Plain(_) => return,
        };
        let mut i = 0usize;
        while i < docs.len() {
            let Some(cur) = self.current_doc else {
                return;
            };
            let cur_id = cur.doc_id();
            while i < docs.len() && docs[i] < cur_id {
                i += 1;
            }
            if i >= docs.len() {
                return;
            }
            if docs[i] > cur_id {
                self.next(docs[i]);
                continue;
            }
            let block_idx = self.index >> shift;
            let mask = match &self.list {
                PostingList::Compressed(list) => list.block_mask(),
                PostingList::Plain(_) => return,
            };
            let (stay, end_of_block) = match &self.list {
                PostingList::Compressed(list) => {
                    let compressed = unsafe { &*self.ensure_compressed_block_ptr(list, block_idx) };
                    let mut j = self.index & mask;
                    let n = compressed.doc_ids.len();
                    while i < docs.len() && j < n {
                        let block_doc = u64::from(compressed.doc_ids[j]);
                        if block_doc < docs[i] {
                            j += 1;
                        } else if block_doc == docs[i] {
                            out_freqs[i] = compressed.freqs[j];
                            i += 1;
                            j += 1;
                        } else {
                            i += 1;
                        }
                    }
                    let stay = (j < n).then(|| (compressed.doc_ids[j], compressed.freqs[j]));
                    (stay, j)
                }
                PostingList::Plain(_) => return,
            };
            if let Some((doc_id, frequency)) = stay {
                self.index = (block_idx << shift) + end_of_block;
                self.block_idx = block_idx;
                self.current_doc = Some(DocInfo::Raw(RawDocInfo { doc_id, frequency }));
            } else {
                let next_start = (block_idx + 1) << shift;
                if next_start >= length {
                    self.index = length;
                    self.block_idx = self.index >> shift;
                    self.current_doc = None;
                } else {
                    self.index = next_start;
                    self.block_idx = block_idx + 1;
                    let next_doc = match &self.list {
                        PostingList::Compressed(list) => {
                            let next =
                                unsafe { &*self.ensure_compressed_block_ptr(list, block_idx + 1) };
                            (next.doc_ids[0], next.freqs[0])
                        }
                        PostingList::Plain(_) => return,
                    };
                    self.current_doc = Some(DocInfo::Raw(RawDocInfo {
                        doc_id: next_doc.0,
                        frequency: next_doc.1,
                    }));
                }
            }
        }
    }

    /// Copy one decompressed posting block (or a 128-doc Plain slice) with
    /// `doc_id <= window_max` into `out`. Leaves the cursor on the first doc
    /// beyond that chunk.
    pub(super) fn take_docs_one_block_upto(&mut self, window_max: u64, out: &mut Vec<(u64, u32)>) {
        out.clear();
        match self.list {
            PostingList::Compressed(ref list) => {
                let Some(cur) = self.current_doc else {
                    return;
                };
                if cur.doc_id() > window_max {
                    return;
                }
                let shift = list.block_shift();
                let mask = list.block_mask();
                let block_idx = self.index >> shift;
                let block_offset = self.index & mask;
                let compressed = unsafe { &mut *self.ensure_compressed_block_ptr(list, block_idx) };
                for offset in block_offset..compressed.doc_ids.len() {
                    let doc_id = compressed.doc_ids[offset];
                    if u64::from(doc_id) > window_max {
                        self.index = (block_idx << shift) + offset;
                        self.block_idx = block_idx;
                        self.current_doc = Some(DocInfo::Raw(RawDocInfo {
                            doc_id,
                            frequency: compressed.freqs[offset],
                        }));
                        return;
                    }
                    out.push((u64::from(doc_id), compressed.freqs[offset]));
                }
                let next_start = (block_idx + 1) << shift;
                if next_start >= list.length as usize {
                    self.index = list.length as usize;
                    self.block_idx = self.index >> shift;
                    self.current_doc = None;
                    return;
                }
                self.index = next_start;
                self.block_idx = block_idx + 1;
                let compressed =
                    unsafe { &mut *self.ensure_compressed_block_ptr(list, block_idx + 1) };
                self.current_doc = Some(DocInfo::Raw(RawDocInfo {
                    doc_id: compressed.doc_ids[0],
                    frequency: compressed.freqs[0],
                }));
            }
            PostingList::Plain(_) => {
                while let Some(cur) = self.doc() {
                    let doc = cur.doc_id();
                    if doc > window_max {
                        break;
                    }
                    out.push((doc, cur.frequency()));
                    self.next(doc + 1);
                    if out.len() >= 128 {
                        break;
                    }
                }
            }
        }
    }

    /// Advance to the first doc of the next posting block, or exhaust.
    pub(super) fn skip_current_block(&mut self) -> bool {
        match self.next_block_first_doc() {
            Some(doc) => {
                self.next(doc);
                self.current_doc.is_some()
            }
            None => {
                if let PostingList::Compressed(ref list) = self.list {
                    self.index = list.length as usize;
                    self.block_idx = self.index >> list.block_shift();
                }
                self.current_doc = None;
                false
            }
        }
    }
}

impl<'a, S: Scorer, D: WandDocuments> Wand<'a, S, D> {
    /// Skip dead essential blocks and complete surviving docs with at most
    /// two optional clauses via SoA buffers.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn score_single_essential_upto(
        &mut self,
        clauses: &mut [MaxScoreClause],
        first_essential: usize,
        ess_idx: usize,
        upto: u64,
        essential_query_rank: usize,
        essential_term: u32,
        essential_weight: f32,
        essential_window_bound: f32,
        num_query_terms: usize,
        total_non_essential_bound: f64,
        total_sum_upper_bound_factor: f64,
        norm_k_ref: Option<(&[u8], &[f32; 256])>,
        essential_chunk: &mut Vec<(u64, u32)>,
        scratch: &mut MaxScoreSoaScratch,
        candidates: &mut TopKCollector,
        num_comparisons: &mut usize,
        wand_factor: f32,
    ) -> Result<()> {
        // Deleted rows are still named by the posting list and are only
        // rejected when the hit is inserted, so on a partition with deletions
        // this path would merge and score the optionals for documents it is
        // about to drop -- the per-document path it replaced asked about
        // visibility before touching any optional. Asking up front cannot
        // change a result (an invisible document is dropped either way), but it
        // is only worth doing when something can actually be invisible: on a
        // fully visible partition it is a scan that rejects nothing.
        // `any_invisible` is O(1) and conservative -- it errs towards probing,
        // so skipping the probe can only give up the saving, never a document.
        // (`visible_cost_upper_bound` cannot stand in for it: a deletion
        // vector too large to materialize still reports the full count.)
        let probe_visibility = self.documents.any_invisible();
        loop {
            // Lucene `nextPostings(upTo)`: do not advance past this call's
            // exclusive end. After `take_docs` the cursor already sits on the
            // first doc of the next block, which may belong to a later window
            // whose optional remainder is larger.
            if clauses[ess_idx]
                .posting
                .doc()
                .is_none_or(|doc| doc.doc_id() > upto)
            {
                break;
            }
            if self.threshold > 0.0 {
                let block_max = clauses[ess_idx].posting.block_max_score(&self.scorer);
                if exclusive_cannot_compete(
                    block_max,
                    total_non_essential_bound,
                    self.threshold,
                    total_sum_upper_bound_factor,
                ) {
                    // `skip_current_block()` also drops the documents past `upto`,
                    // which belong to a later window. Meeting them again through a
                    // different essential would score them without this clause's
                    // contribution, so only skip wholesale when the whole block
                    // lies inside the span this call was asked to complete.
                    if upto < clauses[ess_idx].posting.block_end_doc() {
                        // `upto` can be the terminator, in which case there is
                        // nothing left for this call to complete.
                        if upto >= u32::MAX as u64 {
                            break;
                        }
                        clauses[ess_idx].posting.next(upto + 1);
                        break;
                    }
                    #[cfg(test)]
                    {
                        self.maxscore_essential_blocks_skipped += 1;
                    }
                    let more = clauses[ess_idx].posting.skip_current_block();
                    if !more
                        || clauses[ess_idx]
                            .posting
                            .doc()
                            .is_none_or(|doc| doc.doc_id() > upto)
                    {
                        break;
                    }
                    continue;
                }
            }
            clauses[ess_idx]
                .posting
                .take_docs_one_block_upto(upto, essential_chunk);
            if essential_chunk.is_empty() {
                break;
            }
            if probe_visibility {
                essential_chunk.retain(|(doc, _)| {
                    self.documents
                        .document_key_for_doc_id(*doc as u32)
                        .is_some()
                });
                if essential_chunk.is_empty() {
                    // The whole block was invisible; `take_docs` already left
                    // the cursor on the next one.
                    continue;
                }
            }
            let (non_essential, _) = clauses.split_at_mut(first_essential);
            self.complete_maxscore_single_essential(
                essential_chunk,
                non_essential,
                essential_query_rank,
                essential_term,
                essential_weight,
                essential_window_bound,
                num_query_terms,
                total_non_essential_bound,
                total_sum_upper_bound_factor,
                norm_k_ref,
                scratch,
                candidates,
                num_comparisons,
                wand_factor,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn complete_maxscore_single_essential(
        &mut self,
        hits: &[(u64, u32)],
        non_essential: &mut [MaxScoreClause],
        essential_query_rank: usize,
        essential_term: u32,
        essential_weight: f32,
        essential_window_bound: f32,
        num_query_terms: usize,
        total_non_essential_bound: f64,
        total_sum_upper_bound_factor: f64,
        norm_k_ref: Option<(&[u8], &[f32; 256])>,
        scratch: &mut MaxScoreSoaScratch,
        candidates: &mut TopKCollector,
        num_comparisons: &mut usize,
        wand_factor: f32,
    ) -> Result<()> {
        debug_assert!(non_essential.len() <= 2);
        scratch.docs.clear();
        scratch.ess_scores.clear();
        scratch.ess_freqs.clear();
        scratch.opt_scores.clear();
        scratch.opt_freqs.clear();
        bulk_bm25_tf(
            essential_weight,
            hits,
            norm_k_ref,
            |doc, freq| {
                essential_weight
                    * self
                        .scorer
                        .doc_weight(freq, self.documents.scoring_num_tokens(doc as u32))
            },
            &mut scratch.raw_scores,
        );
        for (hit, &score) in hits.iter().zip(scratch.raw_scores.iter()) {
            *num_comparisons += 1;
            if exclusive_cannot_compete(
                score,
                total_non_essential_bound,
                self.threshold,
                total_sum_upper_bound_factor,
            ) {
                continue;
            }
            scratch.docs.push(hit.0);
            scratch.ess_scores.push(score);
            scratch.ess_freqs.push(hit.1);
        }
        if scratch.docs.is_empty() {
            return Ok(());
        }

        let mut opt_term = 0u32;
        let mut opt_rank = 0usize;

        if non_essential.len() == 1 {
            let optional = &mut non_essential[0];
            opt_term = optional.posting.term_index();
            opt_rank = optional.query_rank;
            let require_best = essential_window_bound.is_finite()
                && essential_window_bound > 0.0
                && exclusive_cannot_compete(
                    essential_window_bound,
                    0.0,
                    self.threshold,
                    total_sum_upper_bound_factor,
                );
            let query_weight = optional.posting.query_weight;
            // With a single optional clause `total_non_essential_bound` is that
            // clause's own `prefix_bound`, so the entry filter above already
            // applied this predicate; re-running it here was a no-op pass.
            scratch.opt_scores.clear();
            scratch.opt_scores.resize(scratch.docs.len(), 0.0);
            scratch.opt_freqs.clear();
            scratch.opt_freqs.resize(scratch.docs.len(), 0);
            let probe = &mut optional.posting;
            probe.merge_optional_freqs(&scratch.docs, &mut scratch.opt_freqs);
            let mut w = 0usize;
            for r in 0..scratch.docs.len() {
                let freq = scratch.opt_freqs[r];
                if freq == 0 && require_best {
                    continue;
                }
                let contribution = if freq == 0 {
                    0.0
                } else {
                    bm25_tf_from_caches(query_weight, freq, scratch.docs[r] as u32, norm_k_ref)
                        .unwrap_or_else(|| {
                            probe.score(
                                &self.scorer,
                                freq,
                                self.documents.scoring_num_tokens(scratch.docs[r] as u32),
                            )
                        })
                };
                scratch.docs[w] = scratch.docs[r];
                scratch.ess_scores[w] = scratch.ess_scores[r];
                scratch.ess_freqs[w] = scratch.ess_freqs[r];
                scratch.opt_scores[w] = contribution;
                scratch.opt_freqs[w] = freq;
                w += 1;
            }
            scratch.docs.truncate(w);
            scratch.ess_scores.truncate(w);
            scratch.ess_freqs.truncate(w);
            scratch.opt_scores.truncate(w);
            scratch.opt_freqs.truncate(w);
        } else if non_essential.len() == 2 {
            let n_live = scratch.docs.len();
            for slot in 0..2 {
                scratch.opt_s[slot].clear();
                scratch.opt_s[slot].resize(n_live, 0.0);
                scratch.opt_f[slot].clear();
                scratch.opt_f[slot].resize(n_live, 0);
            }
            scratch.partial.clear();
            scratch.partial.extend_from_slice(&scratch.ess_scores);
            let opt_term = [
                non_essential[0].posting.term_index(),
                non_essential[1].posting.term_index(),
            ];
            let opt_rank = [non_essential[0].query_rank, non_essential[1].query_rank];
            let remaining_without_best = non_essential[0].prefix_bound;
            let require_best = essential_window_bound.is_finite()
                && essential_window_bound > 0.0
                && remaining_without_best.is_finite()
                && exclusive_cannot_compete(
                    essential_window_bound,
                    remaining_without_best,
                    self.threshold,
                    total_sum_upper_bound_factor,
                );
            let apply = [1usize, 0];
            for (step, &idx) in apply.iter().enumerate() {
                let required = require_best && step == 0;
                let query_weight = non_essential[idx].posting.query_weight;
                // Step 0 probes the second optional, whose `prefix_bound` is
                // `total_non_essential_bound`; the entry filter already applied
                // that predicate while `partial` still equalled the essential
                // score, so re-filtering here would be a no-op pass.
                if !required && step > 0 {
                    let prefix_bound = non_essential[idx].prefix_bound;
                    let mut w = 0usize;
                    for r in 0..scratch.docs.len() {
                        if exclusive_cannot_compete(
                            scratch.partial[r],
                            prefix_bound,
                            self.threshold,
                            total_sum_upper_bound_factor,
                        ) {
                            continue;
                        }
                        scratch.docs[w] = scratch.docs[r];
                        scratch.ess_scores[w] = scratch.ess_scores[r];
                        scratch.ess_freqs[w] = scratch.ess_freqs[r];
                        scratch.opt_s[0][w] = scratch.opt_s[0][r];
                        scratch.opt_s[1][w] = scratch.opt_s[1][r];
                        scratch.opt_f[0][w] = scratch.opt_f[0][r];
                        scratch.opt_f[1][w] = scratch.opt_f[1][r];
                        scratch.partial[w] = scratch.partial[r];
                        w += 1;
                    }
                    scratch.docs.truncate(w);
                    scratch.ess_scores.truncate(w);
                    scratch.ess_freqs.truncate(w);
                    scratch.partial.truncate(w);
                    scratch.opt_s[0].truncate(w);
                    scratch.opt_s[1].truncate(w);
                    scratch.opt_f[0].truncate(w);
                    scratch.opt_f[1].truncate(w);
                    if scratch.docs.is_empty() {
                        return Ok(());
                    }
                }
                let n = scratch.docs.len();
                scratch.opt_f[idx].clear();
                scratch.opt_f[idx].resize(n, 0);
                let probe = &mut non_essential[idx].posting;
                probe.merge_optional_freqs(&scratch.docs, &mut scratch.opt_f[idx]);
                let mut w = 0usize;
                for r in 0..n {
                    let freq = scratch.opt_f[idx][r];
                    if freq == 0 && required {
                        continue;
                    }
                    let contribution = if freq == 0 {
                        0.0
                    } else {
                        bm25_tf_from_caches(query_weight, freq, scratch.docs[r] as u32, norm_k_ref)
                            .unwrap_or_else(|| {
                                probe.score(
                                    &self.scorer,
                                    freq,
                                    self.documents.scoring_num_tokens(scratch.docs[r] as u32),
                                )
                            })
                    };
                    scratch.docs[w] = scratch.docs[r];
                    scratch.ess_scores[w] = scratch.ess_scores[r];
                    scratch.ess_freqs[w] = scratch.ess_freqs[r];
                    scratch.opt_s[0][w] = scratch.opt_s[0][r];
                    scratch.opt_s[1][w] = scratch.opt_s[1][r];
                    scratch.opt_f[0][w] = scratch.opt_f[0][r];
                    scratch.opt_f[1][w] = scratch.opt_f[1][r];
                    scratch.opt_s[idx][w] = contribution;
                    scratch.opt_f[idx][w] = freq;
                    scratch.partial[w] = scratch.partial[r] + contribution;
                    w += 1;
                }
                scratch.docs.truncate(w);
                scratch.ess_scores.truncate(w);
                scratch.ess_freqs.truncate(w);
                scratch.partial.truncate(w);
                scratch.opt_s[0].truncate(w);
                scratch.opt_s[1].truncate(w);
                scratch.opt_f[0].truncate(w);
                scratch.opt_f[1].truncate(w);
                if scratch.docs.is_empty() {
                    return Ok(());
                }
            }
            for i in 0..scratch.docs.len() {
                let mut by_rank = SmallVec::<[f32; 8]>::from_elem(0.0, num_query_terms);
                by_rank[essential_query_rank] = scratch.ess_scores[i];
                for (o, &rank) in opt_rank.iter().enumerate() {
                    if scratch.opt_f[o][i] > 0 {
                        by_rank[rank] = scratch.opt_s[o][i];
                    }
                }
                let canonical = by_rank
                    .into_iter()
                    .fold(0.0_f32, |sum, contribution| sum + contribution);
                if canonical <= self.threshold || candidates.rejects_score(canonical) {
                    continue;
                }
                let Some(document_key) = self
                    .documents
                    .document_key_for_doc_id(scratch.docs[i] as u32)
                else {
                    continue;
                };
                let doc_length = self.documents.scoring_num_tokens(scratch.docs[i] as u32);
                let mut freqs = SmallVec::<[(u32, u32); 3]>::new();
                freqs.push((essential_term, scratch.ess_freqs[i]));
                for (o, &term) in opt_term.iter().enumerate() {
                    if scratch.opt_f[o][i] > 0 {
                        freqs.push((term, scratch.opt_f[o][i]));
                    }
                }
                if candidates.insert(
                    ScoredDoc::new(document_key, canonical),
                    doc_length,
                    scratch.docs[i],
                    freqs.into_iter(),
                )? && let Some(kth) = candidates.kth_score_if_full()
                {
                    self.update_threshold(kth, wand_factor);
                }
            }
            return Ok(());
        }

        for i in 0..scratch.docs.len() {
            let opt_freq = if non_essential.len() == 1 {
                scratch.opt_freqs.get(i).copied().unwrap_or(0)
            } else {
                0
            };
            let canonical = if non_essential.is_empty() {
                scratch.ess_scores[i]
            } else {
                let mut by_rank = SmallVec::<[f32; 8]>::from_elem(0.0, num_query_terms);
                by_rank[essential_query_rank] = scratch.ess_scores[i];
                if opt_freq > 0 {
                    by_rank[opt_rank] = scratch.opt_scores[i];
                }
                by_rank
                    .into_iter()
                    .fold(0.0_f32, |sum, contribution| sum + contribution)
            };
            if canonical <= self.threshold || candidates.rejects_score(canonical) {
                continue;
            }
            let Some(document_key) = self
                .documents
                .document_key_for_doc_id(scratch.docs[i] as u32)
            else {
                continue;
            };
            let doc_length = self.documents.scoring_num_tokens(scratch.docs[i] as u32);
            let mut freqs = SmallVec::<[(u32, u32); 2]>::new();
            freqs.push((essential_term, scratch.ess_freqs[i]));
            if opt_freq > 0 {
                freqs.push((opt_term, opt_freq));
            }
            if candidates.insert(
                ScoredDoc::new(document_key, canonical),
                doc_length,
                scratch.docs[i],
                freqs.into_iter(),
            )? && let Some(kth) = candidates.kth_score_if_full()
            {
                self.update_threshold(kth, wand_factor);
            }
        }
        Ok(())
    }
}
