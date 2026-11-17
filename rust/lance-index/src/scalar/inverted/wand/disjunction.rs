// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Block-max WAND disjunction: the head / lead / tail posting loop.

use std::collections::BinaryHeap;

use lance_core::Result;
use smallvec::SmallVec;

use crate::scalar::inverted::DocInfo;
use crate::scalar::inverted::query::Operator;
use crate::scalar::inverted::scorer::Scorer;

use super::bounds::{
    conservative_score_sum, next_up_f64, score_sum_cannot_compete, score_sum_upper_bound_factor,
};
use super::doc_source::WandDocuments;
use super::posting::PostingIterator;
use super::{HeadPosting, TERMINATED_DOC_ID, TailPosting, Wand};

impl<'a, S: Scorer, D: WandDocuments> Wand<'a, S, D> {
    // find the next doc candidate
    // Find the next term-level candidate doc. The returned score is the exact
    // contribution from the current `lead` set; additional score can still come
    // from `tail` iterators that are advanced to the same doc later.
    pub(super) fn next(&mut self) -> Result<Option<(DocInfo, f32)>> {
        if self.operator == Operator::And {
            let candidate = self.next_and_candidate();
            #[cfg(test)]
            {
                if candidate.is_some() {
                    self.and_window_stats.candidates_returned += 1;
                }
            }
            return Ok(candidate.map(|doc| (doc, self.and_candidate_score.unwrap_or(0.0))));
        }

        loop {
            let Some(target) = self.head_doc() else {
                if self.advance_tail_to_next_or_window() {
                    continue;
                }
                return Ok(None);
            };

            if self.up_to.is_none_or(|up_to| target > up_to) {
                // The tail's bounds expire at the old window end. Refresh from
                // the first unchecked doc, before the next head posting, and
                // re-read the head since a competitive tail may now precede it.
                let refresh_target = self.up_to.map_or(self.or_next_target, |up_to| {
                    self.or_next_target.max(up_to + 1)
                });
                self.update_max_scores(refresh_target);
                continue;
            }
            self.move_head_doc_to_lead(target);
            if self.lead.is_empty() {
                continue;
            }

            // Block-Max WAND pruning: skip the whole window when its score upper
            // bound cannot reach the top-k threshold.
            if self.threshold > 0.0
                && self
                    .floor_mode
                    .rejects_upper_bound(f64::from(self.or_block_window_max()), self.threshold)
            {
                // On the final block `up_to` is the `u64::MAX` sentinel; step once
                // there to avoid seeking past the valid doc id range.
                let mut skip_to = match self.up_to {
                    Some(up_to) if up_to < u32::MAX as u64 => up_to + 1,
                    _ => target + 1,
                };
                // The narrow window is dead; if the whole level1 group is dead
                // too, hop over it in one advance.
                let group_skip = self.or_group_skip_to();
                if let Some(group_skip_to) = group_skip {
                    skip_to = skip_to.max(group_skip_to);
                }
                self.push_back_leads(skip_to);
                continue;
            }

            let Some(first_doc) = self.lead.first().and_then(|posting| posting.doc()) else {
                self.push_back_leads(target + 1);
                continue;
            };
            let doc_length = self.documents.doc_length(&first_doc);
            let mut lead_score = 0.0;
            if let Some(first_posting) = self.lead.first() {
                lead_score += first_posting.score(&self.scorer, first_doc.frequency(), doc_length);
            }
            for posting in self.lead.iter().skip(1) {
                if let Some(lead_doc) = posting.doc() {
                    lead_score += posting.score(&self.scorer, lead_doc.frequency(), doc_length);
                }
            }

            while !self.floor_mode.accepts_score(lead_score, self.threshold) {
                if score_sum_cannot_compete(
                    lead_score,
                    self.tail_upper_bound_sum(),
                    self.threshold,
                    score_sum_upper_bound_factor(self.num_terms),
                    self.floor_mode,
                ) {
                    self.push_back_leads(first_doc.doc_id() + 1);
                    break;
                }
                if !self.advance_tail_top(target, doc_length, &mut lead_score) {
                    // Dynamic heap-order accumulation may be below the query-
                    // order score. Let the final canonical scorer decide.
                    break;
                }
            }

            if !self.lead.is_empty() {
                return Ok(Some((first_doc, lead_score)));
            }
        }
    }

    #[allow(clippy::vec_box)]
    pub(super) fn head_doc(&self) -> Option<u64> {
        self.head.peek().map(HeadPosting::doc_id)
    }

    fn push_head(&mut self, posting: Box<PostingIterator>) {
        if posting.doc().is_some() {
            self.head.push(HeadPosting::new(posting));
        }
    }

    pub(super) fn move_head_doc_to_lead(&mut self, target: u64) {
        while self.head_doc() == Some(target) {
            if let Some(posting) = self.head.pop() {
                self.lead.push(posting.posting);
            }
        }
    }

    // Move all head iterators that are already known to be behind `target`
    // into `tail`, possibly overflowing low-value entries back into `head`.
    pub(super) fn move_head_before_target_to_tail(&mut self, target: u64) {
        if self.threshold <= 0.0 {
            while matches!(self.head_doc(), Some(doc_id) if doc_id < target) {
                if let Some(mut posting) = self.head.pop().map(|posting| posting.posting) {
                    posting.next(target);
                    self.push_head(posting);
                }
            }
            return;
        }

        while matches!(self.head_doc(), Some(doc_id) if doc_id < target) {
            if let Some(posting) = self.head.pop() {
                let upper_bound = posting.posting.global_upper_bound(&self.scorer);
                if let Some(mut evicted) =
                    self.insert_tail_with_overflow(posting.posting, upper_bound)
                {
                    evicted.next(target);
                    self.push_head(evicted);
                }
            }
        }
    }

    /// Upper bound on the score of any document in the window `[target, up_to]`
    /// for a disjunction. Sums the block-max of every overlapping iterator:
    /// `lead`, `head` (later docs still in the window, which
    /// `can_target_beat_threshold` omits), and the individual tail bounds.
    pub(super) fn or_block_window_max(&self) -> f32 {
        conservative_score_sum(
            self.lead
                .iter()
                .map(|posting| posting.window_max_score(self.up_to, &self.scorer))
                .chain(
                    self.head
                        .iter()
                        .map(|posting| posting.posting.window_max_score(self.up_to, &self.scorer)),
                )
                .chain(self.tail.iter().map(|posting| posting.upper_bound)),
        )
    }

    pub(super) fn can_target_beat_threshold(&mut self, target: u64) -> bool {
        if self.up_to.is_none_or(|up_to| target > up_to) {
            self.update_max_scores(target);
        }

        let mut possible_matches = self.lead.len();
        for posting in &self.tail {
            if matches!(posting.posting.block_first_doc(), Some(block_doc) if block_doc <= target) {
                possible_matches += 1;
            }
        }
        let sum = conservative_score_sum(
            self.lead
                .iter()
                .map(|posting| posting.window_max_score(self.up_to, &self.scorer))
                .chain(
                    self.tail
                        .iter()
                        .filter(|posting| {
                            matches!(
                                posting.posting.block_first_doc(),
                                Some(block_doc) if block_doc <= target
                            )
                        })
                        .map(|posting| posting.posting.window_max_score(self.up_to, &self.scorer)),
                ),
        );

        match self.operator {
            Operator::And => {
                possible_matches >= self.num_terms
                    && !self
                        .floor_mode
                        .rejects_upper_bound(f64::from(sum), self.threshold)
            }
            Operator::Or => !self
                .floor_mode
                .rejects_upper_bound(f64::from(sum), self.threshold),
        }
    }

    pub(super) fn update_max_scores(&mut self, target: u64) {
        // Refresh the block-max window for the current target. The resulting
        // `up_to` is the furthest doc id for which this block-max view remains
        // valid. Like Lucene's WANDScorer, the boundary comes from the cheap
        // clauses only, and the refresh avoids allocating: heaps are recycled
        // through their backing vectors (shallow_next never changes the doc a
        // head entry is ordered by, so heapify restores the same shape).
        let lead_cost = self
            .lead
            .iter()
            .map(|posting| posting.cost())
            .min()
            .unwrap_or(usize::MAX);
        let mut narrow_up_to = TERMINATED_DOC_ID;
        for posting in &mut self.lead {
            posting.shallow_next(target);
            narrow_up_to = narrow_up_to.min(posting.block_end_doc());
        }

        let mut head_postings = std::mem::take(&mut self.head).into_vec();
        for posting in &mut head_postings {
            // Unlike Lucene, every head clause participates in the boundary:
            // the refresh is allocation-free and answers from the anchored
            // level caches, so frequent refreshes are cheap, while keeping the
            // window inside every clause's current block keeps all the bounds
            // at tight level0 values.
            let doc_id = posting.doc_id();
            posting.posting.shallow_next(doc_id);
            narrow_up_to = narrow_up_to.min(posting.posting.block_end_doc());
        }

        let mut tail_postings = std::mem::take(&mut self.tail).into_vec();
        for tail_posting in &mut tail_postings {
            tail_posting.posting.shallow_next(target);
        }

        if narrow_up_to == TERMINATED_DOC_ID
            && let Some(top) = tail_postings
                .iter()
                .min_by_key(|posting| posting.posting.cost())
            && top.posting.cost() <= lead_cost
        {
            narrow_up_to = narrow_up_to.min(top.posting.block_end_doc().max(target));
        }

        self.up_to = Some(narrow_up_to);
        self.head = BinaryHeap::from(head_postings);

        self.tail_max_score = 0.0;
        for tail_posting in tail_postings {
            let posting = tail_posting.posting;
            let upper_bound = match posting.block_first_doc() {
                Some(block_doc) if block_doc <= target => {
                    posting.window_max_score(self.up_to, &self.scorer)
                }
                _ => 0.0,
            };
            if let Some(mut evicted) = self.insert_tail_with_overflow(posting, upper_bound) {
                evicted.next(target);
                self.push_head(evicted);
            }
        }
    }

    /// After the narrow window proved skippable, try widening the skip to the
    /// level1 group boundary, in the spirit of Lucene's `getSkipUpTo`. All
    /// bounds come from the anchored level caches, so a failed attempt costs a
    /// few loads and float adds — unlike an eager wide-window probe.
    ///
    /// The group boundary is the minimum current-group end over every live
    /// iterator, so each iterator's level1 score is a valid bound over
    /// `[target, group_up_to]`.
    fn or_group_skip_to(&self) -> Option<u64> {
        let mut group_up_to = TERMINATED_DOC_ID;
        for posting in &self.lead {
            let (doc_up_to, _) = posting.impact_group_bound(&self.scorer)?;
            group_up_to = group_up_to.min(doc_up_to);
        }
        for posting in self.head.iter() {
            let (doc_up_to, _) = posting.posting.impact_group_bound(&self.scorer)?;
            group_up_to = group_up_to.min(doc_up_to);
        }
        for tail_posting in self.tail.iter() {
            let (doc_up_to, _) = tail_posting.posting.impact_group_bound(&self.scorer)?;
            group_up_to = group_up_to.min(doc_up_to);
        }
        if self.up_to.is_some_and(|up_to| group_up_to <= up_to) {
            // No gain over the narrow window skip.
            return None;
        }

        // Second pass over the memoized bounds: sum only the iterators that
        // can produce a doc inside the skipped range.
        let mut bounds = SmallVec::<[f32; 8]>::new();
        for posting in &self.lead {
            bounds.push(posting.impact_group_bound(&self.scorer)?.1);
        }
        for posting in self
            .head
            .iter()
            .filter(|posting| posting.doc_id() <= group_up_to)
        {
            bounds.push(posting.posting.impact_group_bound(&self.scorer)?.1);
        }
        for tail_posting in self.tail.iter().filter(|tail_posting| {
            matches!(
                tail_posting.posting.block_first_doc(),
                Some(block_doc) if block_doc <= group_up_to
            )
        }) {
            bounds.push(tail_posting.posting.impact_group_bound(&self.scorer)?.1);
        }
        let bounds_sum = conservative_score_sum(bounds.into_iter());
        self.floor_mode
            .rejects_upper_bound(f64::from(bounds_sum), self.threshold)
            .then_some(group_up_to.saturating_add(1))
    }

    pub(super) fn refine_or_candidate(&mut self, target: u64, doc_length: u32) -> bool {
        if self.threshold <= 0.0 {
            return true;
        }

        let mut lead_score = self
            .lead
            .iter()
            .filter_map(|posting| {
                posting
                    .doc()
                    .map(|doc| posting.score(&self.scorer, doc.frequency(), doc_length))
            })
            .sum::<f32>();

        while !self.floor_mode.accepts_score(lead_score, self.threshold) {
            if score_sum_cannot_compete(
                lead_score,
                self.tail_upper_bound_sum(),
                self.threshold,
                score_sum_upper_bound_factor(self.num_terms),
                self.floor_mode,
            ) {
                return false;
            }
            if !self.advance_tail_top(target, doc_length, &mut lead_score) {
                // No remaining posting can be advanced, but f32 execution-
                // order rounding may still leave the canonical score at or
                // above the inclusive floor.
                return true;
            }
        }

        true
    }

    pub(super) fn collect_tail_matches(&mut self, target: u64) {
        let mut remaining = Vec::with_capacity(self.tail.len());
        let tail = std::mem::take(&mut self.tail);
        self.tail_max_score = 0.0;
        for tail_posting in tail.into_vec() {
            let mut posting = tail_posting.posting;
            posting.next(target);
            match posting.doc().map(|doc| doc.doc_id()) {
                Some(doc_id) if doc_id == target => self.lead.push(posting),
                Some(_) => remaining.push(posting),
                None => {}
            }
        }

        for posting in remaining {
            self.push_head(posting);
        }
    }

    pub(super) fn advance_tail_and_lead_to_head(&mut self, least_id: u64) {
        let mut postings = Vec::with_capacity(self.tail.len() + self.lead.len());
        while let Some(tail) = self.tail.pop() {
            postings.push(tail.posting);
        }
        self.tail_max_score = 0.0;
        postings.append(&mut self.lead);
        for mut posting in postings {
            posting.next(least_id);
            self.push_head(posting);
        }
    }

    pub(super) fn advance_lead_to_head(&mut self, least_id: u64) {
        let lead = std::mem::take(&mut self.lead);
        for mut posting in lead {
            posting.next(least_id);
            self.push_head(posting);
        }
        // In the flat-search path this is only called after `collect_tail_matches`,
        // which drains the current tail into either `lead` or `head`. At this
        // point `tail` is expected to be empty, so clearing it is a no-op that
        // just resets the cached `tail_max_score`.
        debug_assert!(self.tail.is_empty());
        self.clear_tail();
    }

    fn clear_tail(&mut self) {
        self.tail.clear();
        self.tail_max_score = 0.0;
    }

    fn insert_tail(&mut self, posting: Box<PostingIterator>, upper_bound: f32) {
        self.tail_max_score = next_up_f64(self.tail_max_score + f64::from(upper_bound));
        self.tail
            .push(TailPosting::new(upper_bound, posting.cost(), posting));
    }

    fn insert_tail_with_overflow(
        &mut self,
        posting: Box<PostingIterator>,
        upper_bound: f32,
    ) -> Option<Box<PostingIterator>> {
        // Keep only the lagging iterators that are most useful for deciding the
        // current candidate. If a stronger tail entry arrives, evict the weakest
        // one back to the caller so it can be advanced into `head`.
        if self.threshold <= 0.0 || upper_bound <= 0.0 {
            return Some(posting);
        }

        let parked_upper_bound = self.tail_upper_bound_sum() + f64::from(upper_bound);
        if score_sum_cannot_compete(
            0.0,
            parked_upper_bound,
            self.threshold,
            score_sum_upper_bound_factor(self.num_terms),
            self.floor_mode,
        ) {
            self.insert_tail(posting, upper_bound);
            return None;
        }

        if self.tail.is_empty() {
            return Some(posting);
        }

        let candidate = TailPosting::new(upper_bound, posting.cost(), posting);
        if let Some(top) = self.tail.peek()
            && top > &candidate
        {
            let evicted = self.tail.pop().expect("peeked tail posting should exist");
            self.remove_tail_upper_bound(evicted.upper_bound);
            self.tail_max_score = next_up_f64(self.tail_max_score + f64::from(upper_bound));
            self.tail.push(candidate);
            return Some(evicted.posting);
        }

        Some(candidate.posting)
    }

    fn lead_to_tail_upper_bound(&self, posting: &PostingIterator, target: u64) -> f32 {
        if self.operator == Operator::Or
            && posting.is_compressed()
            && self.up_to.is_some_and(|up_to| target <= up_to)
        {
            posting.window_max_score(self.up_to, &self.scorer)
        } else {
            posting.global_upper_bound(&self.scorer)
        }
    }

    pub(super) fn advance_tail_to_next_or_window(&mut self) -> bool {
        if self.operator != Operator::Or || self.tail.is_empty() {
            return false;
        }

        let Some(up_to) = self.up_to else {
            return false;
        };
        if up_to >= u32::MAX as u64 {
            return false;
        }
        if !self
            .tail
            .iter()
            .any(|tail| tail.posting.has_next_compressed_block())
        {
            return false;
        }

        // A low-scoring tail can be the only iterator left in the current
        // window. Move to the next window so a later high-scoring block is still
        // reachable instead of ending the disjunction early.
        self.update_max_scores(self.or_next_target.max(up_to + 1));
        true
    }

    pub(super) fn push_back_leads(&mut self, target: u64) {
        self.or_next_target = target;
        // After finishing a candidate doc, convert the aligned iterators back
        // into lagging iterators. Entries that do not stay in `tail` are
        // advanced to `target` and returned to `head`.
        // pop() drains in place, keeping self.lead's capacity for reuse.
        if self.threshold <= 0.0 {
            while let Some(mut posting) = self.lead.pop() {
                posting.next(target);
                self.push_head(posting);
            }
            return;
        }

        while let Some(posting) = self.lead.pop() {
            let upper_bound = self.lead_to_tail_upper_bound(&posting, target);
            if let Some(mut evicted) = self.insert_tail_with_overflow(posting, upper_bound) {
                evicted.next(target);
                self.push_head(evicted);
            }
        }
    }

    fn advance_tail_top(&mut self, target: u64, doc_length: u32, lead_score: &mut f32) -> bool {
        // Advance the most promising lagging iterator to the current target.
        // If it lands on the target, fold its exact contribution into
        // `lead_score`; otherwise put it back into `head`.
        let Some(TailPosting {
            upper_bound,
            cost: _,
            mut posting,
        }) = self.tail.pop()
        else {
            return false;
        };
        self.remove_tail_upper_bound(upper_bound);
        posting.next(target);
        match posting.doc() {
            Some(doc) if doc.doc_id() == target => {
                *lead_score += posting.score(&self.scorer, doc.frequency(), doc_length);
                self.lead.push(posting);
            }
            Some(_) => self.push_head(posting),
            None => {}
        }
        true
    }

    #[inline]
    fn tail_upper_bound_sum(&self) -> f64 {
        self.tail_max_score
    }

    #[inline]
    fn remove_tail_upper_bound(&mut self, upper_bound: f32) {
        if self.tail.is_empty() {
            self.tail_max_score = 0.0;
            return;
        }
        self.tail_max_score = next_up_f64((self.tail_max_score - f64::from(upper_bound)).max(0.0));
    }

    pub(super) fn advance_all_tail(
        &mut self,
        target: u64,
        doc_length: Option<u32>,
        mut score: Option<&mut f32>,
    ) {
        // Materialize all remaining lagging iterators for `target`. This is
        // only done once we have already decided to fully score / validate the
        // candidate.
        let tail = std::mem::take(&mut self.tail);
        self.tail_max_score = 0.0;
        for tail_posting in tail.into_vec() {
            let mut posting = tail_posting.posting;
            posting.next(target);
            match posting.doc() {
                Some(doc) if doc.doc_id() == target => {
                    if let (Some(doc_length), Some(score)) = (doc_length, score.as_deref_mut()) {
                        *score += posting.score(&self.scorer, doc.frequency(), doc_length);
                    }
                    self.lead.push(posting)
                }
                Some(_) => self.push_head(posting),
                None => {}
            }
        }
    }
}
