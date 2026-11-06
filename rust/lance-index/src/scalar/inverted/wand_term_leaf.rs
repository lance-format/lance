// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! One-term Boolean Match leaf scorer: a leaf with exactly one ungrouped
//! posting steps its posting directly instead of paying the WAND wrapper.
//! See [`TermLeafScorer`] for the floor contract.

use std::cell::OnceCell;
use std::sync::Arc;

use lance_core::{Error, Result};

use super::super::DocInfo;
use super::super::compound::{ComposableScorer, ScoreBounds};
use super::super::scorer::MemBM25Scorer;
use crate::metrics::MetricsCollector;

use super::{PostingIterator, WandDocuments, conservative_score_sum, shallow_score_bound};

/// One-term Boolean Match leaf: posting GEQ for approximation, BM25 on `score()`.
///
/// Unlike [`WandCursor`], this is not a WAND machine: it never uses a
/// competitive floor to skip blocks or drop documents. `ReqOptScorer` holds
/// the floor itself and forwards it to neither side, so this scorer must keep
/// every hit with `required < floor ≤ required + optional`.
pub struct TermLeafScorer<'a, D: WandDocuments> {
    posting: PostingIterator,
    documents: &'a D,
    scorer: Arc<MemBM25Scorer>,
    cost: usize,
    global_score_upper_bound: OnceCell<Option<f32>>,
    current_doc: Option<DocInfo>,
    current_document_key: Option<u64>,
    current_score: f32,
    score_ready: bool,
    shallow: Option<(u64, u64, f32)>,
    comparisons: usize,
    metrics_recorded: bool,
    metrics: &'a dyn MetricsCollector,
}

impl<'a, D: WandDocuments> TermLeafScorer<'a, D> {
    pub fn new(
        posting: PostingIterator,
        documents: &'a D,
        scorer: Arc<MemBM25Scorer>,
        metrics: &'a dyn MetricsCollector,
    ) -> Self {
        let cost = posting.cost().min(documents.visible_cost_upper_bound());
        Self {
            posting,
            documents,
            scorer,
            cost,
            global_score_upper_bound: OnceCell::new(),
            current_doc: None,
            current_document_key: None,
            current_score: 0.0,
            score_ready: false,
            shallow: None,
            comparisons: 0,
            metrics_recorded: false,
            metrics,
        }
    }

    pub fn doc(&self) -> Option<u64> {
        self.current_doc.map(|doc| doc.doc_id())
    }

    pub fn document_key(&self) -> Option<u64> {
        self.current_document_key
    }

    fn clear_current(&mut self) {
        self.current_doc = None;
        self.current_document_key = None;
        self.current_score = 0.0;
        self.score_ready = false;
        self.shallow = None;
    }

    fn record_metrics(&mut self) {
        if !self.metrics_recorded {
            self.metrics.record_comparisons(self.comparisons);
            self.metrics_recorded = true;
        }
    }

    fn position_geq(&mut self, mut target: u64) -> Result<Option<u64>> {
        self.clear_current();
        loop {
            self.posting.next_doc_id(target, true);
            let Some(doc) = self.posting.current_doc else {
                self.record_metrics();
                return Ok(None);
            };
            self.comparisons += 1;
            let Some(document_key) = self.documents.document_key(&doc) else {
                target = doc.doc_id().saturating_add(1);
                continue;
            };
            self.current_doc = Some(doc);
            self.current_document_key = Some(document_key);
            self.current_score = 0.0;
            self.score_ready = false;
            return Ok(Some(doc.doc_id()));
        }
    }

    pub fn next(&mut self) -> Result<Option<u64>> {
        let target = match self.doc() {
            None => 0,
            Some(u64::MAX) => {
                self.clear_current();
                self.record_metrics();
                return Ok(None);
            }
            Some(current) => current.saturating_add(1),
        };
        self.position_geq(target)
    }

    pub fn advance(&mut self, target: u64) -> Result<Option<u64>> {
        if self.doc().is_some_and(|doc| doc >= target) {
            return Ok(self.doc());
        }
        self.position_geq(target)
    }

    pub fn cost(&self) -> usize {
        self.cost
    }

    pub fn global_score_upper_bound(&self) -> Option<f32> {
        *self.global_score_upper_bound.get_or_init(|| {
            if self.posting.has_grouped_terms() {
                return None;
            }
            let upper = conservative_score_sum(std::iter::once(
                self.posting.global_upper_bound(self.scorer.as_ref()),
            ));
            (upper.is_finite() && upper >= 0.0).then_some(upper)
        })
    }

    pub fn current_score(&mut self) -> Result<f32> {
        if self.current_doc.is_none() {
            return Err(Error::internal(
                "posting FTS scorer is not positioned on a document",
            ));
        }
        if self.score_ready {
            return Ok(self.current_score);
        }
        let scored = self.posting.doc().ok_or_else(|| {
            Error::internal("single-term posting is not positioned on a document")
        })?;
        let doc_length = self.documents.doc_length(&scored);
        let score = self
            .posting
            .score(self.scorer.as_ref(), scored.frequency(), doc_length);
        self.current_doc = Some(scored);
        self.current_score = score;
        self.score_ready = true;
        Ok(score)
    }

    pub fn advance_shallow(&mut self, target: u64) -> Result<u64> {
        self.posting.shallow_next(target);
        if let Some(first) = self.posting.block_first_doc()
            && first > target
        {
            let up_to = first.saturating_sub(1).max(target);
            self.shallow = Some((target, up_to, 0.0));
            return Ok(up_to);
        }
        let up_to = self.posting.block_end_doc().max(target);
        let upper = conservative_score_sum(std::iter::once(
            self.posting
                .window_max_score(Some(up_to), self.scorer.as_ref()),
        ));
        self.shallow = Some((target, up_to, upper));
        Ok(up_to)
    }

    pub fn score_upper_bound(&self, up_to: u64) -> Result<f32> {
        shallow_score_bound(self.shallow, up_to)
    }

    pub fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()> {
        if min_score.is_nan() {
            return Err(Error::invalid_input(
                "minimum competitive FTS score cannot be NaN",
            ));
        }
        Ok(())
    }

    pub fn scored_upper_bound(&self) -> Option<f32> {
        self.score_ready.then_some(self.current_score)
    }

    #[cfg(test)]
    pub fn frequency_blocks_decoded(&self) -> usize {
        self.posting.frequency_blocks_decoded()
    }
}

impl<D: WandDocuments> Drop for TermLeafScorer<'_, D> {
    fn drop(&mut self) {
        self.record_metrics();
    }
}

impl<D: WandDocuments + Sync> ComposableScorer for TermLeafScorer<'_, D> {
    fn doc(&self) -> Option<u64> {
        self.doc()
    }

    fn document_key(&self) -> Option<u64> {
        self.document_key()
    }

    fn next(&mut self) -> Result<Option<u64>> {
        self.next()
    }

    fn advance(&mut self, target: u64) -> Result<Option<u64>> {
        self.advance(target)
    }

    fn cost(&self) -> usize {
        self.cost()
    }

    fn score(&mut self) -> Result<f32> {
        self.current_score()
    }

    fn advance_shallow(&mut self, target: u64) -> Result<u64> {
        self.advance_shallow(target)
    }

    fn score_bounds(&mut self, up_to: u64) -> Result<ScoreBounds> {
        Ok(ScoreBounds {
            lower: 0.0,
            upper: self.score_upper_bound(up_to)?,
        })
    }

    fn global_score_upper_bound(&self) -> Option<f32> {
        TermLeafScorer::global_score_upper_bound(self)
    }

    fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()> {
        self.set_min_competitive_score(min_score)
    }

    fn current_score_upper_bound(&mut self) -> Result<Option<f32>> {
        Ok(self.scored_upper_bound())
    }

    fn matches(&mut self) -> Result<bool> {
        Ok(self.doc().is_some())
    }

    fn scores_non_negative(&self) -> bool {
        true
    }
}
