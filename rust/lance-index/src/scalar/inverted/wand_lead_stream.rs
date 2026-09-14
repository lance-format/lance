// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Skewed conjunction search: the rare list's decompressed block is the
//! buffer, dense followers only `next(target)`.
//!
//! Community Auto still uses N-way bulk for balanced 2/3 (and modern 4+).
//! When `max_cost / min_cost >= AND_SKEW_RATIO`, bulk would decompress the
//! stopword in every window the rare term barely touches. Three or more
//! skewed clauses therefore take this lead-stream; a skewed pair stays on
//! classic leapfrog (Lucene `ConjunctionDISI` — one follower makes the
//! buffer pure overhead).
//!
//! Inner score-first is not a dispatch table: with three or more clauses
//! whose second list is at least `AND_SCORE_FIRST_COST_RATIO` times the
//! lead, the rare BM25 is scored before seeking dense followers. Phrase
//! confirmation stays on the existing score-then-positions path.

use lance_core::Result;
use smallvec::SmallVec;

use super::super::builder::ScoredDoc;
use super::super::encoding::MAX_POSTING_BLOCK_SIZE;
use super::super::query::FtsSearchParams;
use super::super::scorer::Scorer;
use super::{
    BLOCK_SIZE, CompetitiveFloorMode, DocCandidate, DocInfo, PostingIterator, PostingList,
    RawDocInfo, TERMINATED_DOC_ID, TopKCollector, Wand, WandDocuments, conservative_score_sum,
    score_sum_cannot_compete, score_sum_upper_bound_factor,
};
use crate::metrics::MetricsCollector;

/// Cost ratio at which Auto leaves N-way bulk. Same 32× Wikipedia number
/// the analysis tree used for n≤3; not a term-count bucket.
pub(super) const AND_SKEW_RATIO: usize = 32;

/// Score the rare lead before seeking followers when the second list is
/// at least this many times longer. Same 3× shape as IU promotion.
const AND_SCORE_FIRST_COST_RATIO: usize = 3;

const FREQ_LUT_BUCKETS: usize = 64;

enum LeadFollowerSeek {
    Match,
    Leap(u64),
    Exhausted,
}

pub(super) fn conjunction_lists_are_skewed(min_cost: usize, max_cost: usize) -> bool {
    min_cost > 0 && max_cost / min_cost >= AND_SKEW_RATIO
}

fn and_score_first_for(num_lists: usize, lead_cost: usize, second_cost: usize) -> bool {
    num_lists >= 3 && second_cost >= lead_cost.saturating_mul(AND_SCORE_FIRST_COST_RATIO)
}

impl PostingIterator {
    /// Copy remaining docs in the current decompressed block at or before
    /// `window_max` without moving the cursor. Returns the absolute posting
    /// index of the first copied doc.
    fn peek_remaining_block_docs_upto(
        &mut self,
        window_max: u64,
        docs: &mut Vec<u32>,
        freqs: &mut Vec<u32>,
    ) -> Option<usize> {
        docs.clear();
        freqs.clear();
        let PostingList::Compressed(ref list) = self.list else {
            return None;
        };
        let cur = self.current_doc?;
        if cur.doc_id() > window_max {
            return None;
        }
        let shift = list.block_shift();
        let block_idx = self.index >> shift;
        let block_offset = self.index & list.block_mask();
        // SAFETY: `ensure_compressed_block_ptr` hands back a pointer into this
        // iterator's own `UnsafeCell` compressed state for `block_idx`, which it
        // guarantees is populated. We hold `&mut self`, and the reads below
        // complete before the state is touched again, so no second reference to
        // it is live across them. Same contract as the other call sites.
        let compressed = unsafe { &mut *self.ensure_compressed_block_ptr(list, block_idx) };
        for offset in block_offset..compressed.doc_ids.len() {
            let doc_id = compressed.doc_ids[offset];
            if u64::from(doc_id) > window_max {
                break;
            }
            docs.push(doc_id);
            freqs.push(compressed.freqs[offset]);
        }
        if docs.is_empty() {
            None
        } else {
            Some((block_idx << shift) + block_offset)
        }
    }
}

impl<'a, S: Scorer, D: WandDocuments> Wand<'a, S, D> {
    /// Seek `lead[from..to]` onto `doc`.
    fn seek_lead_followers(&mut self, from: usize, to: usize, doc: u32) -> LeadFollowerSeek {
        for posting in self.lead.iter_mut().take(to).skip(from) {
            if posting
                .doc()
                .is_none_or(|cur| cur.doc_id() < u64::from(doc))
            {
                posting.next(u64::from(doc));
            }
            match posting.doc() {
                None => return LeadFollowerSeek::Exhausted,
                Some(cur) if cur.doc_id() > u64::from(doc) => {
                    return LeadFollowerSeek::Leap(cur.doc_id());
                }
                Some(_) => {}
            }
        }
        LeadFollowerSeek::Match
    }

    /// After the two rarest lists sit on `doc`, skip remaining dense
    /// followers when even a conservative Exclusive upper cannot enter
    /// the heap. `rest_block_max` is `lead[2..]` over the current window.
    fn matched_pair_cannot_beat_floor(
        &self,
        lead_freq: u32,
        doc_length: u32,
        rest_block_max: f64,
        num_lists: usize,
    ) -> bool {
        if self.threshold <= 0.0 || num_lists < 3 {
            return false;
        }
        let Some(second) = self.lead.get(1).and_then(|posting| posting.doc()) else {
            return false;
        };
        score_sum_cannot_compete(
            self.lead[0].score(&self.scorer, lead_freq, doc_length),
            f64::from(self.lead[1].score(&self.scorer, second.frequency(), doc_length))
                + rest_block_max,
            self.threshold,
            score_sum_upper_bound_factor(num_lists),
            CompetitiveFloorMode::Exclusive,
        )
    }

    fn park_lead_at(&mut self, index: usize, doc: u32, freq: u32) {
        self.lead[0].index = index;
        if let PostingList::Compressed(ref list) = self.lead[0].list {
            self.lead[0].block_idx = index >> list.block_shift();
        }
        self.lead[0].current_doc = Some(DocInfo::Raw(RawDocInfo {
            doc_id: doc,
            frequency: freq,
        }));
    }

    /// Skewed conjunction: the rare lead's decompressed block is the buffer,
    /// followers only `next(target)`. Scoring, phrase confirm, and heap
    /// semantics match the classic loop.
    pub(super) fn and_lead_stream_search(
        &mut self,
        params: &FtsSearchParams,
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<DocCandidate<D::Candidate>>> {
        let limit = params.limit.unwrap_or(usize::MAX);
        if limit == 0 {
            return Ok(vec![]);
        }
        let phrase_slop = params.phrase_slop;
        let num_lists = self.lead.len();
        let mut freq_bound_lut = [f32::INFINITY; FREQ_LUT_BUCKETS];
        for (freq, slot) in freq_bound_lut
            .iter_mut()
            .enumerate()
            .take(FREQ_LUT_BUCKETS - 1)
        {
            *slot = self.lead[0].score(&self.scorer, freq as u32, 0);
        }
        // Every LUT entry is consumed as an upper bound for any document
        // length, which only holds while the BM25 document weight is
        // non-increasing in length. Lock that premise down here. Use `>=`:
        // scorers that ignore document length (e.g. `UnitScorer`) tie.
        #[cfg(debug_assertions)]
        for freq in [1_usize, FREQ_LUT_BUCKETS / 2, FREQ_LUT_BUCKETS - 2] {
            let upper = freq_bound_lut[freq];
            for doc_length in [1_u32, 64, 4096] {
                debug_assert!(
                    upper >= self.lead[0].score(&self.scorer, freq as u32, doc_length),
                    "freq LUT entry {freq} is not an upper bound at doc length {doc_length}"
                );
            }
        }
        freq_bound_lut[FREQ_LUT_BUCKETS - 1] =
            self.lead[0].frequency_clamp_upper_bound(&self.scorer);

        let score_first = and_score_first_for(
            num_lists,
            self.lead[0].cost(),
            self.lead.get(1).map(|posting| posting.cost()).unwrap_or(0),
        );

        let mut candidates = TopKCollector::new(limit, std::cmp::min(limit, BLOCK_SIZE * 10));
        let mut num_comparisons: usize = 0;
        let mut lead_docs: Vec<u32> = Vec::with_capacity(MAX_POSTING_BLOCK_SIZE);
        let mut lead_freqs: Vec<u32> = Vec::with_capacity(MAX_POSTING_BLOCK_SIZE);

        let mut target: u64 = 0;
        for posting in &self.lead {
            match posting.doc() {
                Some(doc) => target = target.max(doc.doc_id()),
                None => return Ok(vec![]),
            }
        }

        'window: loop {
            self.raise_to_shared_floor(params.wand_factor);
            self.lead[0].next(target);
            let Some(lead_doc) = self.lead[0].doc() else {
                break;
            };
            target = target.max(lead_doc.doc_id());
            let win_end = Self::posting_block_up_to(&self.lead[0], target);

            if self.threshold > 0.0 {
                for posting in &mut self.lead {
                    posting.shallow_next(target);
                }
                let wide_max = conservative_score_sum(self.lead.iter().map(|posting| {
                    posting
                        .block_max_score_up_to_with_stats(win_end, &self.scorer)
                        .score
                }));
                if wide_max < self.threshold {
                    if win_end == TERMINATED_DOC_ID {
                        break;
                    }
                    target = win_end + 1;
                    continue;
                }
            }

            let Some(first_index) = self.lead[0].peek_remaining_block_docs_upto(
                win_end,
                &mut lead_docs,
                &mut lead_freqs,
            ) else {
                if win_end == TERMINATED_DOC_ID {
                    break;
                }
                target = win_end + 1;
                continue;
            };

            // These bounds must be computed even when the floor is zero at
            // window entry: the floor can become live part-way through the
            // window, and `freq_cannot_beat` below consumes them as an upper
            // bound. Skipping the computation would under-estimate that bound
            // and prune documents that can still compete.
            let others_bounds: SmallVec<[f64; 8]> = self.lead[1..]
                .iter()
                .map(|posting| {
                    f64::from(
                        posting
                            .block_max_score_up_to_with_stats(win_end, &self.scorer)
                            .score,
                    )
                })
                .collect();
            let others_block_max = others_bounds.iter().copied().sum::<f64>();
            let rest_block_max = others_bounds.iter().skip(1).copied().sum::<f64>();
            let freq_cannot_beat = if self.threshold > 0.0 && num_lists >= 2 {
                std::array::from_fn(|frequency| {
                    score_sum_cannot_compete(
                        freq_bound_lut[frequency],
                        others_block_max,
                        self.threshold,
                        score_sum_upper_bound_factor(num_lists),
                        CompetitiveFloorMode::Exclusive,
                    )
                })
            } else {
                [false; FREQ_LUT_BUCKETS]
            };

            let mut pos = 0;
            while pos < lead_docs.len() {
                let doc = lead_docs[pos];
                let freq = lead_freqs[pos];
                let freq_bucket = (freq as usize).min(FREQ_LUT_BUCKETS - 1);
                if freq_cannot_beat[freq_bucket] {
                    pos += 1;
                    continue;
                }

                self.park_lead_at(first_index + pos, doc, freq);
                let Some(parked) = self.lead[0].doc() else {
                    pos += 1;
                    continue;
                };
                let Some(document_key) = self.documents.document_key(&parked) else {
                    pos += 1;
                    continue;
                };
                let doc_length = self.documents.doc_length(&parked);

                if score_first
                    && self.threshold > 0.0
                    && score_sum_cannot_compete(
                        self.lead[0].score(&self.scorer, freq, doc_length),
                        others_block_max,
                        self.threshold,
                        score_sum_upper_bound_factor(num_lists),
                        CompetitiveFloorMode::Exclusive,
                    )
                {
                    pos += 1;
                    continue;
                }

                let split_followers = num_lists >= 3 && self.threshold > 0.0;
                let first_followers = if split_followers { 2 } else { num_lists };
                match self.seek_lead_followers(1, first_followers, doc) {
                    LeadFollowerSeek::Exhausted => break 'window,
                    LeadFollowerSeek::Leap(next) => {
                        pos = skip_lead_docs(&lead_docs, pos, next);
                        if next > win_end {
                            target = next;
                            continue 'window;
                        }
                        continue;
                    }
                    LeadFollowerSeek::Match => {}
                }

                if self.matched_pair_cannot_beat_floor(freq, doc_length, rest_block_max, num_lists)
                {
                    pos += 1;
                    continue;
                }

                if split_followers {
                    match self.seek_lead_followers(2, num_lists, doc) {
                        LeadFollowerSeek::Exhausted => break 'window,
                        LeadFollowerSeek::Leap(next) => {
                            pos = skip_lead_docs(&lead_docs, pos, next);
                            if next > win_end {
                                target = next;
                                continue 'window;
                            }
                            continue;
                        }
                        LeadFollowerSeek::Match => {}
                    }
                }

                let score = self.score_in_query_order(doc_length);
                num_comparisons += 1;
                if let Some(slop) = phrase_slop {
                    if self.exclusive_score_cannot_beat_floor(score) {
                        pos += 1;
                        continue;
                    }
                    if !self.check_positions(slop as i32)? {
                        pos += 1;
                        continue;
                    }
                }

                if candidates.insert(
                    ScoredDoc::new(document_key, score),
                    doc_length,
                    u64::from(doc),
                    self.iter_term_freqs(),
                )? && let Some(kth) = candidates.kth_score_if_full()
                {
                    self.update_threshold(kth, params.wand_factor);
                }
                pos += 1;
            }

            if win_end == TERMINATED_DOC_ID {
                break;
            }
            target = win_end + 1;
        }

        metrics.record_comparisons(num_comparisons);
        candidates.into_candidates(|key| self.documents.candidate_from_key(key))
    }
}

fn skip_lead_docs(lead_docs: &[u32], pos: usize, next: u64) -> usize {
    let next32 = u32::try_from(next).unwrap_or(u32::MAX);
    pos + lead_docs[pos + 1..].partition_point(|&id| id < next32) + 1
}

#[cfg(test)]
mod tests {
    use super::{
        AND_SKEW_RATIO, and_score_first_for, conjunction_lists_are_skewed, skip_lead_docs,
    };

    #[test]
    fn skew_gate_uses_integer_max_over_min() {
        assert!(!conjunction_lists_are_skewed(0, 1_000_000));
        assert!(!conjunction_lists_are_skewed(16, 16 * (AND_SKEW_RATIO - 1)));
        assert!(conjunction_lists_are_skewed(16, 16 * AND_SKEW_RATIO));
        assert!(conjunction_lists_are_skewed(1, AND_SKEW_RATIO));
    }

    #[test]
    fn score_first_needs_three_clauses_and_a_3x_second() {
        assert!(!and_score_first_for(2, 100, 10_000));
        assert!(!and_score_first_for(3, 100, 299));
        assert!(and_score_first_for(3, 100, 300));
        assert!(and_score_first_for(6, 10, 30));
    }

    #[test]
    fn skip_lead_docs_advances_to_the_first_id_at_or_after_next() {
        let docs = [10, 20, 30, 40];
        assert_eq!(skip_lead_docs(&docs, 0, 30), 2);
        assert_eq!(skip_lead_docs(&docs, 0, 25), 2);
        assert_eq!(skip_lead_docs(&docs, 1, 50), 4);
    }
}
