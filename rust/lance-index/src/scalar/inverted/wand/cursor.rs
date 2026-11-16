// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Posting-backed cursor that exposes a `Wand` to the compound collector.

use std::cell::OnceCell;
use std::sync::Arc;

use lance_core::{Error, Result};
use smallvec::SmallVec;

use crate::metrics::MetricsCollector;
use crate::scalar::inverted::DocInfo;
use crate::scalar::inverted::query::{FtsSearchParams, Operator};
use crate::scalar::inverted::scorer::{MemBM25Scorer, Scorer};

use super::bounds::{CompetitiveFloorMode, conservative_score_sum};
use super::doc_source::WandDocuments;
use super::posting::PostingIterator;
use super::{HeadPosting, TERMINATED_DOC_ID, Wand};

/// Incremental posting-backed scorer used by compound FTS.
///
/// Unlike [`Wand::search`], this cursor does not own a top-k heap. It exposes
/// exact candidates and conservative block bounds to the compound collector,
/// which owns the only competitive score for the whole scorer tree.
pub(in crate::scalar::inverted) struct WandCursor<'a, D: WandDocuments> {
    pub(super) wand: Wand<'a, Arc<MemBM25Scorer>, D>,
    phrase_slop: Option<u32>,
    wand_factor: f32,
    cost: usize,
    global_score_upper_bound: OnceCell<Option<f32>>,
    current_doc: Option<DocInfo>,
    current_document_key: Option<u64>,
    current_score: f32,
    confirmation: Option<bool>,
    shallow: Option<(u64, u64, f32)>,
    comparisons: usize,
    metrics_recorded: bool,
    metrics: &'a dyn MetricsCollector,
}

impl<'a, D: WandDocuments> WandCursor<'a, D> {
    pub(in crate::scalar::inverted) fn new(
        operator: Operator,
        postings: Vec<PostingIterator>,
        documents: &'a D,
        scorer: Arc<MemBM25Scorer>,
        params: &FtsSearchParams,
        metrics: &'a dyn MetricsCollector,
    ) -> Self {
        let cost = match operator {
            Operator::And => postings.iter().map(PostingIterator::cost).min(),
            Operator::Or => Some(
                postings
                    .iter()
                    .map(PostingIterator::cost)
                    .fold(0, usize::saturating_add),
            ),
        }
        .unwrap_or_default()
        .min(documents.visible_cost_upper_bound());
        let mut wand = Wand::new(operator, postings.into_iter(), documents, scorer)
            .with_floor_mode(CompetitiveFloorMode::Inclusive);
        if params.phrase_slop.is_some() {
            wand.score_first_and_enabled = false;
        }
        Self {
            wand,
            phrase_slop: params.phrase_slop,
            wand_factor: params.wand_factor,
            cost,
            global_score_upper_bound: OnceCell::new(),
            current_doc: None,
            current_document_key: None,
            current_score: 0.0,
            confirmation: None,
            shallow: None,
            comparisons: 0,
            metrics_recorded: false,
            metrics,
        }
    }

    pub(in crate::scalar::inverted) fn doc(&self) -> Option<u64> {
        self.current_doc.map(|doc| doc.doc_id())
    }

    pub(in crate::scalar::inverted) fn document_key(&self) -> Option<u64> {
        self.current_document_key
    }

    fn clear_current(&mut self) {
        self.current_doc = None;
        self.current_document_key = None;
        self.current_score = 0.0;
        self.confirmation = None;
        self.shallow = None;
    }

    fn record_metrics(&mut self) {
        if !self.metrics_recorded {
            self.metrics.record_comparisons(self.comparisons);
            self.metrics_recorded = true;
        }
    }

    fn position_next(&mut self) -> Result<Option<u64>> {
        loop {
            let Some((doc, and_score)) = self.wand.next()? else {
                self.clear_current();
                self.record_metrics();
                return Ok(None);
            };
            self.comparisons += 1;
            let doc_id = doc.doc_id();
            let Some(document_key) = self.wand.documents.document_key(&doc) else {
                if self.wand.operator == Operator::Or {
                    self.wand.push_back_leads(doc_id.saturating_add(1));
                }
                continue;
            };
            let doc_length = self.wand.documents.doc_length(&doc);
            self.wand.advance_all_tail(doc_id, None, None);
            let score = if self.wand.and_candidate_score.is_some() {
                and_score
            } else {
                self.wand.score_in_query_order(doc_length)
            };
            self.current_doc = Some(doc);
            self.current_document_key = Some(document_key);
            self.current_score = score;
            self.confirmation = self.phrase_slop.is_none().then_some(true);
            self.shallow = None;
            return Ok(Some(doc_id));
        }
    }

    pub(in crate::scalar::inverted) fn next(&mut self) -> Result<Option<u64>> {
        if let Some(doc) = self.current_doc
            && self.wand.operator == Operator::Or
        {
            self.wand.push_back_leads(doc.doc_id().saturating_add(1));
        }
        self.clear_current();
        self.position_next()
    }

    pub(in crate::scalar::inverted) fn advance(&mut self, target: u64) -> Result<Option<u64>> {
        if self.doc().is_some_and(|doc| doc >= target) {
            return Ok(self.doc());
        }
        self.clear_current();
        self.wand.seek(target);
        self.position_next()
    }

    pub(in crate::scalar::inverted) fn cost(&self) -> usize {
        self.cost
    }

    pub(in crate::scalar::inverted) fn global_score_upper_bound(&self) -> Option<f32> {
        *self
            .global_score_upper_bound
            .get_or_init(|| self.wand.compound_global_score_upper_bound())
    }

    pub(in crate::scalar::inverted) fn current_score(&self) -> Result<f32> {
        self.current_doc
            .map(|_| self.current_score)
            .ok_or_else(|| Error::internal("posting FTS scorer is not positioned on a document"))
    }

    pub(in crate::scalar::inverted) fn matches(&mut self) -> Result<bool> {
        let Some(_) = self.current_doc else {
            return Ok(false);
        };
        if let Some(confirmed) = self.confirmation {
            return Ok(confirmed);
        }
        let phrase_slop = self.phrase_slop.ok_or_else(|| {
            Error::internal("posting FTS scorer requires phrase slop for position confirmation")
        })?;
        let confirmed = self.wand.check_positions(phrase_slop as i32)?;
        self.confirmation = Some(confirmed);
        Ok(confirmed)
    }

    pub(in crate::scalar::inverted) fn match_cost(&self) -> Option<f32> {
        self.phrase_slop.map(|_| self.wand.num_terms.max(1) as f32)
    }

    pub(in crate::scalar::inverted) fn advance_shallow(&mut self, target: u64) -> Result<u64> {
        let (up_to, upper) = self.wand.compound_shallow_bound(target);
        self.shallow = Some((target, up_to, upper));
        Ok(up_to)
    }

    pub(in crate::scalar::inverted) fn score_upper_bound(&self, up_to: u64) -> Result<f32> {
        let (target, shallow_up_to, upper) = self.shallow.ok_or_else(|| {
            Error::internal("score bound requires advance_shallow on the posting FTS scorer")
        })?;
        if up_to < target || up_to > shallow_up_to {
            return Err(Error::internal(format!(
                "posting FTS score bound up_to={up_to} is outside shallow range [{target}, {shallow_up_to}]"
            )));
        }
        Ok(upper)
    }

    pub(in crate::scalar::inverted) fn set_min_competitive_score(
        &mut self,
        min_score: f32,
    ) -> Result<()> {
        if min_score.is_nan() {
            return Err(Error::invalid_input(
                "minimum competitive FTS score cannot be NaN",
            ));
        }
        let floor = min_score * self.wand_factor;
        if floor > self.wand.threshold {
            if self.wand.score_first_and_enabled && self.wand.threshold <= 0.0 && floor > 0.0 {
                self.wand.up_to = None;
                self.wand.invalidate_score_first_and_window();
            }
            self.wand.threshold = floor;
        }
        Ok(())
    }
}

impl<D: WandDocuments> Drop for WandCursor<'_, D> {
    fn drop(&mut self) {
        self.record_metrics();
    }
}

impl<S: Scorer, D: WandDocuments> Wand<'_, S, D> {
    pub(super) fn compound_global_score_upper_bound(&self) -> Option<f32> {
        if self.lead.len() + self.head.len() + self.tail.len() != self.num_terms {
            return None;
        }
        // Grouped expansions score their terms separately, while their union
        // posting stores one aggregate bound whose f32 rounding is not proven
        // conservative for that exact sum.
        if self.lead.iter().any(|posting| posting.has_grouped_terms())
            || self
                .head
                .iter()
                .any(|posting| posting.posting.has_grouped_terms())
            || self
                .tail
                .iter()
                .any(|posting| posting.posting.has_grouped_terms())
        {
            return None;
        }
        let upper = conservative_score_sum(
            self.lead
                .iter()
                .map(|posting| posting.global_upper_bound(&self.scorer))
                .chain(
                    self.head
                        .iter()
                        .map(|posting| posting.posting.global_upper_bound(&self.scorer)),
                )
                .chain(
                    self.tail
                        .iter()
                        .map(|posting| posting.posting.global_upper_bound(&self.scorer)),
                ),
        );
        (upper.is_finite() && upper >= 0.0).then_some(upper)
    }

    pub(super) fn seek(&mut self, target: u64) {
        self.up_to = None;
        self.and_max_score = f32::INFINITY;
        self.and_last_doc = None;
        self.invalidate_score_first_and_window();
        if self.operator == Operator::And {
            for posting in &mut self.lead {
                if posting.doc().is_some_and(|doc| doc.doc_id() < target) {
                    posting.next(target);
                }
            }
            return;
        }

        self.or_next_target = target;
        let mut postings = std::mem::take(&mut self.head)
            .into_vec()
            .into_iter()
            .map(|posting| posting.posting)
            .chain(self.lead.drain(..))
            .chain(
                std::mem::take(&mut self.tail)
                    .into_vec()
                    .into_iter()
                    .map(|posting| posting.posting),
            )
            .collect::<Vec<_>>();
        self.tail_max_score = 0.0;
        for posting in &mut postings {
            if posting.doc().is_some_and(|doc| doc.doc_id() < target) {
                posting.next(target);
            }
        }
        self.head = postings
            .into_iter()
            .filter(|posting| posting.doc().is_some())
            .map(HeadPosting::new)
            .collect();
    }

    pub(super) fn compound_shallow_bound(&mut self, target: u64) -> (u64, f32) {
        if self.operator == Operator::Or {
            self.update_max_scores(target);
            return (
                self.up_to.unwrap_or(TERMINATED_DOC_ID),
                conservative_score_sum(
                    self.lead
                        .iter()
                        .map(|posting| posting.window_max_score(self.up_to, &self.scorer))
                        .chain(self.head.iter().map(|posting| {
                            posting.posting.window_max_score(self.up_to, &self.scorer)
                        }))
                        .chain(self.tail.iter().map(|posting| posting.upper_bound)),
                ),
            );
        }

        let saved_block_indices = self.score_first_and_enabled.then(|| {
            self.lead
                .iter()
                .map(|posting| posting.block_idx)
                .collect::<SmallVec<[usize; 8]>>()
        });
        let mut up_to = TERMINATED_DOC_ID;
        for posting in &mut self.lead {
            posting.shallow_next(target);
            up_to = up_to.min(Self::posting_block_up_to(posting, target));
        }
        let upper = conservative_score_sum(
            self.lead
                .iter()
                .map(|posting| posting.block_max_score(&self.scorer)),
        );
        if let Some(saved_block_indices) = saved_block_indices {
            // The returned bound is immutable. Restore the internal shallow
            // anchors so an external probe cannot steer exact iteration or
            // discard a still-valid score-first window.
            for (posting, block_idx) in self.lead.iter_mut().zip(saved_block_indices) {
                posting.block_idx = block_idx;
            }
        }
        (up_to, upper)
    }
}
