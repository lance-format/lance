// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Doc-at-a-time conjunction with block-max windows and score-first pruning.

use arrow_array::Array;
use smallvec::SmallVec;

use crate::scalar::inverted::query::Operator;
use crate::scalar::inverted::scorer::{Scorer, bm25_doc_weight_with_norm};
use crate::scalar::inverted::{DocInfo, PostingList};

use super::bounds::{
    ScoreContribution, conservative_score_sum, next_up_f64, score_contributions_in_query_order,
    score_sum_cannot_compete, score_sum_upper_bound_factor,
};
use super::doc_source::WandDocuments;
use super::posting::PostingIterator;
use super::{TERMINATED_DOC_ID, Wand};

impl<'a, S: Scorer, D: WandDocuments> Wand<'a, S, D> {
    fn and_candidate_cannot_beat_threshold(&self, doc_length: u32) -> bool {
        if self.operator != Operator::And
            || self.threshold <= 0.0
            || self.num_terms < 2
            || self.lead.len() != self.num_terms
        {
            return false;
        }

        let Some((first, remaining)) = self.lead.split_first() else {
            return false;
        };
        let Some(doc) = first.doc() else {
            return false;
        };

        let remaining_upper_bound = remaining
            .iter()
            .map(|posting| f64::from(posting.block_max_score(&self.scorer)))
            .sum::<f64>();
        score_sum_cannot_compete(
            first.score(&self.scorer, doc.frequency(), doc_length),
            remaining_upper_bound,
            self.threshold,
            score_sum_upper_bound_factor(self.num_terms),
            self.floor_mode,
        )
    }

    pub(super) fn invalidate_score_first_and_window(&mut self) {
        if self.score_first_and_enabled {
            self.score_first_and_suffix_bounds.clear();
            self.score_first_and_bounds_up_to = None;
            self.score_first_and_dense_range = None;
        }
        self.and_candidate_score = None;
    }

    pub(super) fn prepare_score_first_and_window(&mut self) {
        if !self.score_first_and_enabled {
            return;
        }
        self.score_first_and_suffix_bounds.clear();
        self.score_first_and_bounds_up_to = None;
        self.score_first_and_dense_range = None;
        if self.threshold <= 0.0 {
            return;
        }
        #[cfg(test)]
        if self.disable_score_first_and {
            return;
        }
        let Some(up_to) = self.up_to else {
            return;
        };

        if self.and_max_score < self.threshold {
            return;
        }

        let mut dense_from = 0;
        let mut dense_to = up_to;
        for posting in &self.lead {
            let Some(current_doc) = posting.current_doc_id() else {
                return;
            };
            let PostingList::Compressed(list) = &posting.list else {
                return;
            };
            if list.block_size == 0 {
                return;
            }
            let expected_blocks = (list.length as usize).div_ceil(list.block_size);
            if list.blocks.len() != expected_blocks || posting.block_idx >= list.blocks.len() {
                return;
            }
            let Some(block_start) = posting.block_idx.checked_mul(list.block_size) else {
                return;
            };
            let Some(remaining) = (list.length as usize).checked_sub(block_start) else {
                return;
            };
            let block_count = remaining.min(list.block_size);
            if block_count == 0 {
                return;
            }
            let block_first = list.block_least_doc_id(posting.block_idx);
            let Some(block_last) = list
                .impacts
                .as_ref()
                .and_then(|impacts| impacts.level0_doc_up_to(posting.block_idx))
            else {
                return;
            };
            if block_last < block_first
                || u64::from(block_last) - u64::from(block_first) + 1 != block_count as u64
            {
                return;
            }
            dense_from = dense_from.max(current_doc).max(u64::from(block_first));
            dense_to = dense_to.min(u64::from(block_last));
        }
        if dense_from > dense_to {
            return;
        }

        self.score_first_and_suffix_bounds
            .resize(self.lead.len() + 1, 0.0);
        for index in (0..self.lead.len()).rev() {
            let bound = self.lead[index].block_max_score(&self.scorer);
            if !bound.is_finite() || bound < 0.0 {
                self.score_first_and_suffix_bounds.clear();
                return;
            }
            self.score_first_and_suffix_bounds[index] =
                next_up_f64(f64::from(bound) + self.score_first_and_suffix_bounds[index + 1]);
        }
        self.score_first_and_bounds_up_to = Some(up_to);
        self.score_first_and_dense_range = Some((dense_from, dense_to));
    }

    #[inline]
    fn score_first_and_cannot_compete(&self, partial_score: f32, next_clause: usize) -> bool {
        self.score_first_and_bounds_up_to == self.up_to
            && self
                .score_first_and_suffix_bounds
                .get(next_clause)
                .is_some_and(|remaining_upper_bound| {
                    score_sum_cannot_compete(
                        partial_score,
                        *remaining_upper_bound,
                        self.threshold,
                        score_sum_upper_bound_factor(self.num_terms),
                        self.floor_mode,
                    )
                })
    }

    fn and_next_target_after(&self, doc: u64) -> Option<u64> {
        if self
            .lead
            .first()
            .is_some_and(|posting| posting.is_compressed())
            && doc >= u64::from(u32::MAX)
        {
            None
        } else {
            doc.checked_add(1)
        }
    }

    pub(super) fn next_and_candidate(&mut self) -> Option<DocInfo> {
        self.and_candidate_score = None;
        if self.lead.len() < self.num_terms {
            return None;
        }
        // Two- and three-clause conjunctions use the bulk SIMD kernels by
        // default. Restrict this classic-path optimization to four and five
        // clauses, keeping one clause as the benchmark's drift control.
        let is_vectorized_search_enabled = matches!(self.lead.len(), 4 | 5);
        if let Some(last_doc) = self.and_last_doc
            && self
                .lead
                .first()
                .and_then(|posting| posting.current_doc_id())
                == Some(last_doc)
        {
            let next_target = self.and_advance_target(self.and_next_target_after(last_doc)?);
            if next_target == TERMINATED_DOC_ID {
                return None;
            }
            self.lead[0].next_doc_id(next_target, is_vectorized_search_enabled);
        }

        'advance_head: loop {
            let doc = self
                .lead
                .first()
                .and_then(|posting| posting.current_doc_id())?;
            if self.up_to.is_none_or(|up_to| doc > up_to) {
                let next_target = self.and_advance_target(doc);
                if next_target == TERMINATED_DOC_ID {
                    return None;
                }
                if next_target != doc {
                    self.lead[0].next_doc_id(next_target, is_vectorized_search_enabled);
                    continue;
                }
            }

            let mut is_score_first = self.score_first_and_bounds_up_to == self.up_to
                && self.score_first_and_suffix_bounds.len() == self.lead.len() + 1
                && self
                    .score_first_and_dense_range
                    .is_some_and(|(from, to)| doc >= from && doc <= to);
            let mut partial_score = 0.0_f32;
            let mut score_contributions = SmallVec::<[ScoreContribution; 8]>::new();
            let mut score_first_doc_norm = 0.0;
            if is_score_first {
                let lead_doc = self.lead.first().and_then(|posting| posting.doc())?;
                let cached_doc_norm = self
                    .score_first_and_norm_k
                    .get_or_init(|| {
                        self.norm_k_cache().filter(|(_, cache)| {
                            cache
                                .iter()
                                .all(|addend| addend.is_finite() && *addend >= 0.0)
                        })
                    })
                    .as_ref()
                    .and_then(|(norms, cache)| {
                        norms.get(doc as usize).map(|&code| cache[code as usize])
                    });
                let doc_norm = cached_doc_norm
                    .or_else(|| self.scorer.doc_norm(self.documents.doc_length(&lead_doc)));
                if let Some(doc_norm) = doc_norm
                    && doc_norm.is_finite()
                    && doc_norm >= 0.0
                {
                    score_first_doc_norm = doc_norm;
                    let contribution = self.lead[0].query_weight
                        * bm25_doc_weight_with_norm(lead_doc.frequency(), doc_norm);
                    partial_score = contribution;
                    score_contributions
                        .push(((self.lead[0].position, self.lead[0].token_id), contribution));
                    if self.score_first_and_cannot_compete(partial_score, 1) {
                        #[cfg(test)]
                        {
                            self.and_window_stats.score_first_rejections += 1;
                        }
                        let next_target = self.and_advance_target(self.and_next_target_after(doc)?);
                        if next_target == TERMINATED_DOC_ID {
                            return None;
                        }
                        self.lead[0].next_doc_id(next_target, is_vectorized_search_enabled);
                        continue;
                    }
                } else {
                    is_score_first = false;
                }
            }

            for index in 1..self.lead.len() {
                let posting = &mut self.lead[index];
                if posting.current_doc_id()? < doc {
                    posting.next_doc_id(doc, is_vectorized_search_enabled);
                }
                let next = posting.current_doc_id()?;
                if next > doc {
                    let next_target = self.and_advance_target(next);
                    if next_target == TERMINATED_DOC_ID {
                        return None;
                    }
                    self.lead[0].next_doc_id(next_target, is_vectorized_search_enabled);
                    continue 'advance_head;
                }

                if is_score_first {
                    let posting_doc = posting.doc()?;
                    let contribution = posting.query_weight
                        * bm25_doc_weight_with_norm(posting_doc.frequency(), score_first_doc_norm);
                    partial_score += contribution;
                    score_contributions.push(((posting.position, posting.token_id), contribution));
                    if self.score_first_and_cannot_compete(partial_score, index + 1) {
                        #[cfg(test)]
                        {
                            self.and_window_stats.score_first_rejections += 1;
                        }
                        let next_target = self.and_advance_target(self.and_next_target_after(doc)?);
                        if next_target == TERMINATED_DOC_ID {
                            return None;
                        }
                        self.lead[0].next_doc_id(next_target, is_vectorized_search_enabled);
                        continue 'advance_head;
                    }
                }
            }

            let lead_doc = self.lead.first().and_then(|posting| posting.doc())?;
            let doc_length = self.documents.doc_length(&lead_doc);
            if !is_score_first && self.and_candidate_cannot_beat_threshold(doc_length) {
                let next_target = self.and_advance_target(self.and_next_target_after(doc)?);
                if next_target == TERMINATED_DOC_ID {
                    return None;
                }
                self.lead[0].next_doc_id(next_target, is_vectorized_search_enabled);
                continue;
            }
            if is_score_first {
                self.and_candidate_score =
                    Some(score_contributions_in_query_order(score_contributions));
            }
            self.and_last_doc = Some(doc);
            return Some(lead_doc);
        }
    }

    pub(super) fn posting_block_up_to(posting: &PostingIterator, target: u64) -> u64 {
        posting
            .next_block_first_doc()
            .map(|doc| doc.saturating_sub(1))
            .unwrap_or(TERMINATED_DOC_ID)
            .max(target)
    }

    pub(super) fn and_move_to_next_block(&mut self, target: u64) {
        self.invalidate_score_first_and_window();
        if self.threshold <= 0.0 {
            // Eligible conjunctions need no bounds until a positive floor.
            // Their posting cursors advance blocks independently, and every
            // zero-to-positive floor transition invalidates this window.
            self.up_to = Some(if self.score_first_and_enabled {
                TERMINATED_DOC_ID
            } else {
                target
            });
            self.and_max_score = f32::INFINITY;
            return;
        }

        if self.lead.is_empty() {
            self.up_to = Some(TERMINATED_DOC_ID);
            self.and_max_score = 0.0;
            return;
        }

        for posting in &mut self.lead {
            posting.shallow_next(target);
        }

        let narrow_up_to = self
            .lead
            .iter()
            .map(|posting| Self::posting_block_up_to(posting, target))
            .min()
            .unwrap_or(TERMINATED_DOC_ID);
        let narrow_max_score = conservative_score_sum(
            self.lead
                .iter()
                .map(|posting| posting.block_max_score(&self.scorer)),
        );

        if narrow_max_score >= self.threshold {
            self.up_to = Some(narrow_up_to);
            self.and_max_score = narrow_max_score;
            self.prepare_score_first_and_window();
            #[cfg(test)]
            {
                self.and_window_stats.windows_narrow += 1;
            }
            return;
        }

        let lead_up_to = self
            .lead
            .first()
            .map(|posting| Self::posting_block_up_to(posting, target))
            .unwrap_or(TERMINATED_DOC_ID);
        let can_try_wide = lead_up_to > narrow_up_to
            && lead_up_to != TERMINATED_DOC_ID
            && self.lead.iter().all(|posting| posting.is_compressed());

        if can_try_wide {
            let mut wide_bounds = SmallVec::<[f32; 8]>::new();
            #[cfg(test)]
            let mut range_blocks_scanned = 0;
            for posting in &mut self.lead {
                let block_max = posting.block_max_score_up_to_with_stats(lead_up_to, &self.scorer);
                wide_bounds.push(block_max.score);
                #[cfg(test)]
                {
                    range_blocks_scanned += block_max.blocks_scanned;
                }
            }
            let wide_max_score = conservative_score_sum(wide_bounds.into_iter());
            #[cfg(test)]
            {
                self.and_window_stats.range_blocks_scanned += range_blocks_scanned;
            }

            if wide_max_score < self.threshold {
                self.up_to = Some(lead_up_to);
                self.and_max_score = wide_max_score;
                #[cfg(test)]
                {
                    self.and_window_stats.windows_wide += 1;
                }
                return;
            }
        }

        self.up_to = Some(narrow_up_to);
        self.and_max_score = narrow_max_score;
        self.prepare_score_first_and_window();
        #[cfg(test)]
        {
            self.and_window_stats.windows_narrow += 1;
        }
    }

    pub(super) fn and_advance_target(&mut self, mut target: u64) -> u64 {
        if self.up_to.is_none_or(|up_to| target > up_to) {
            self.and_move_to_next_block(target);
        }

        loop {
            let Some(up_to) = self.up_to else {
                return TERMINATED_DOC_ID;
            };
            if self.and_max_score >= self.threshold {
                return target;
            }
            #[cfg(test)]
            {
                self.and_window_stats.windows_skipped += 1;
            }
            if up_to == TERMINATED_DOC_ID {
                return TERMINATED_DOC_ID;
            }
            target = up_to + 1;
            self.and_move_to_next_block(target);
        }
    }
}
