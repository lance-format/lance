// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

/// Below three clauses, maintaining MAXSCORE windows costs more than the
/// generic document-at-a-time union is likely to save.
const MIN_SHOULD_MAXSCORE_CLAUSES: usize = 3;

#[derive(Clone, Copy)]
struct WindowBounds {
    start: u64,
    up_to: u64,
    floor: f32,
    combined_upper: f32,
}

#[derive(Clone, Copy)]
struct ReportedBounds {
    target: u64,
    up_to: u64,
    bounds: ScoreBounds,
}

/// Exact windowed MAXSCORE scorer for same-column Boolean SHOULD sums.
///
/// List-wide maxima split clauses into a non-essential prefix whose total
/// score cannot reach the current floor and an essential suffix that drives
/// candidate iteration and shallow-window boundaries. Non-essential clauses
/// are only probed when they can still make an essential candidate competitive.
pub(super) struct ShouldMaxScoreScorer<'a> {
    children: Vec<BoxScorer<'a>>,
    initialized: bool,
    exhausted: bool,
    current: Option<u64>,
    confirmed_doc: Option<u64>,
    confirmed: bool,
    current_score: Option<f32>,
    min_competitive_score: f32,
    window: Option<WindowBounds>,
    reported_bounds: Option<ReportedBounds>,
    global_upper_bounds: Vec<f32>,
    global_score_upper_bound: f32,
    child_upper_bounds: Vec<f32>,
    essential: Vec<bool>,
    bound_order: Vec<usize>,
    child_scores: Vec<Option<f32>>,
}

impl<'a> ShouldMaxScoreScorer<'a> {
    pub(super) fn global_bounds(children: &[BoxScorer<'a>]) -> Option<Vec<f32>> {
        if children.len() < MIN_SHOULD_MAXSCORE_CLAUSES {
            return None;
        }
        if !children.iter().all(|child| child.scores_non_negative()) {
            return None;
        }
        let bounds = children
            .iter()
            .map(|child| {
                child
                    .global_score_upper_bound()
                    .filter(|upper| upper.is_finite() && *upper >= 0.0)
            })
            .collect::<Option<Vec<_>>>()?;
        Self::sum_uppers(bounds.iter())
            .is_finite()
            .then_some(bounds)
    }

    pub(super) fn new(children: Vec<BoxScorer<'a>>, global_upper_bounds: Vec<f32>) -> Self {
        debug_assert_eq!(children.len(), global_upper_bounds.len());
        let num_children = children.len();
        let global_score_upper_bound = Self::sum_uppers(global_upper_bounds.iter());
        let mut bound_order = (0..num_children).collect::<Vec<_>>();
        bound_order.sort_by(|left, right| {
            global_upper_bounds[*left]
                .total_cmp(&global_upper_bounds[*right])
                .then_with(|| left.cmp(right))
        });
        Self {
            children,
            initialized: false,
            exhausted: false,
            current: None,
            confirmed_doc: None,
            confirmed: false,
            current_score: None,
            min_competitive_score: f32::NEG_INFINITY,
            window: None,
            reported_bounds: None,
            global_upper_bounds,
            global_score_upper_bound,
            child_upper_bounds: vec![0.0; num_children],
            essential: vec![true; num_children],
            bound_order,
            child_scores: vec![None; num_children],
        }
    }

    fn reset_current(&mut self) {
        self.current = None;
        self.confirmed_doc = None;
        self.confirmed = false;
        self.current_score = None;
        self.child_scores.fill(None);
        self.reported_bounds = None;
    }

    fn set_current(&mut self, current: u64) {
        self.reset_current();
        self.current = Some(current);
    }

    fn exhaust(&mut self) -> Option<u64> {
        self.exhausted = true;
        self.reset_current();
        self.window = None;
        None
    }

    fn initialize_next(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }
        for child in &mut self.children {
            child.next()?;
        }
        self.initialized = true;
        Ok(())
    }

    fn initialize_advance(&mut self, target: u64) -> Result<()> {
        if self.initialized {
            return Ok(());
        }
        for child in &mut self.children {
            child.advance(target)?;
        }
        self.initialized = true;
        Ok(())
    }

    fn align_all_children(&mut self, target: u64) -> Result<()> {
        for child in &mut self.children {
            if child.doc().is_some_and(|doc| doc < target) {
                child.advance(target)?;
            }
        }
        Ok(())
    }

    fn select_essential_children(&mut self) {
        self.essential.fill(false);
        let floor = self.min_competitive_score;
        if floor <= 0.0 || floor.is_nan() {
            for (is_essential, child) in self.essential.iter_mut().zip(&self.children) {
                *is_essential = child.doc().is_some();
            }
            return;
        }

        let mut non_essential = 0.0_f64;
        let mut num_non_essential = 0;
        let mut found_essential = false;
        for index in &self.bound_order {
            if self.children[*index].doc().is_none() {
                continue;
            }
            if found_essential {
                self.essential[*index] = true;
                continue;
            }
            let next = non_essential + f64::from(self.global_upper_bounds[*index]);
            let widened_exact = next * score_sum_upper_bound_factor(num_non_essential + 1);
            let rounded = widened_exact as f32;
            let widened = if f64::from(rounded) < widened_exact {
                next_up(rounded)
            } else {
                rounded
            };
            if widened < floor {
                non_essential = next;
                num_non_essential += 1;
            } else {
                self.essential[*index] = true;
                found_essential = true;
            }
        }
    }

    fn usable_bounds(bounds: ScoreBounds) -> bool {
        bounds.lower.is_finite()
            && bounds.upper.is_finite()
            && bounds.lower <= bounds.upper
            && bounds.upper >= 0.0
    }

    fn add_upper(bounds: ScoreBounds, upper: f32) -> ScoreBounds {
        bounds.add(ScoreBounds { lower: 0.0, upper })
    }

    fn sum_uppers<'b>(uppers: impl Iterator<Item = &'b f32>) -> f32 {
        uppers
            .fold(ScoreBounds::ZERO, |sum, upper| Self::add_upper(sum, *upper))
            .upper
    }

    fn prepare_window(&mut self, target: u64) -> Result<()> {
        self.child_upper_bounds.fill(0.0);
        self.reported_bounds = None;

        self.select_essential_children();
        if !self.essential.iter().any(|is_essential| *is_essential) {
            self.exhaust();
            return Ok(());
        }

        let mut up_to = u64::MAX;
        let mut has_active_child = false;
        for (child, is_essential) in self.children.iter_mut().zip(&mut self.essential) {
            if !*is_essential {
                continue;
            }
            if child.doc().is_some_and(|doc| doc < target) {
                child.advance(target)?;
            }
            if let Some(doc) = child.doc() {
                has_active_child = true;
                let child_target = target.max(doc);
                let child_up_to = child.advance_shallow(child_target)?;
                if child_up_to < child_target {
                    return Err(Error::internal(format!(
                        "FTS SHOULD child returned shallow range ending at {child_up_to} before target {child_target}"
                    )));
                }
                up_to = up_to.min(child_up_to);
            } else {
                *is_essential = false;
            }
        }
        if !has_active_child {
            self.exhaust();
            return Ok(());
        }

        for (index, child) in self.children.iter().enumerate() {
            if !self.essential[index] && child.doc().is_some() {
                self.child_upper_bounds[index] = self.global_upper_bounds[index];
            }
        }
        for (index, child) in self.children.iter_mut().enumerate() {
            if self.essential[index] && child.doc().is_some_and(|doc| doc <= up_to) {
                let bounds = child.score_bounds(up_to)?;
                if Self::usable_bounds(bounds) {
                    self.child_upper_bounds[index] =
                        bounds.upper.max(0.0).min(self.global_upper_bounds[index]);
                } else {
                    self.child_upper_bounds[index] = self.global_upper_bounds[index];
                }
            }
        }

        let combined_upper = Self::sum_uppers(self.child_upper_bounds.iter());
        let floor = self.min_competitive_score;
        self.window = Some(WindowBounds {
            start: target,
            up_to,
            floor,
            combined_upper,
        });
        Ok(())
    }

    fn position(&mut self, mut target: u64) -> Result<Option<u64>> {
        if self.exhausted {
            return Ok(None);
        }

        loop {
            let needs_window = self.window.is_none_or(|window| {
                target < window.start
                    || target > window.up_to
                    || self.min_competitive_score > window.floor
            });
            if needs_window {
                self.window = None;
                self.prepare_window(target)?;
                if self.exhausted {
                    return Ok(None);
                }
            }
            let window = self
                .window
                .ok_or_else(|| Error::internal("FTS SHOULD scorer did not prepare a window"))?;

            if window.combined_upper < self.min_competitive_score {
                if window.up_to == u64::MAX {
                    return Ok(self.exhaust());
                }
                target = window.up_to + 1;
                self.window = None;
                continue;
            }

            let next = self
                .children
                .iter()
                .zip(&self.essential)
                .filter_map(|(child, is_essential)| {
                    (*is_essential)
                        .then(|| child.doc())
                        .flatten()
                        .filter(|doc| *doc >= target && *doc <= window.up_to)
                })
                .min();
            if let Some(next) = next {
                self.set_current(next);
                return Ok(self.current);
            }

            if window.up_to == u64::MAX {
                return Ok(self.exhaust());
            }
            target = window.up_to + 1;
            self.window = None;
        }
    }

    fn partial_score_upper(&self) -> f32 {
        let mut bounds = ScoreBounds::ZERO;
        for (index, score) in self.child_scores.iter().enumerate() {
            if let Some(score) = score {
                bounds = bounds.add(ScoreBounds {
                    lower: *score,
                    upper: *score,
                });
            } else if !self.essential[index] {
                bounds = Self::add_upper(bounds, self.child_upper_bounds[index]);
            }
        }
        bounds.upper
    }

    fn ensure_confirmed(&mut self) -> Result<bool> {
        let Some(current) = self.current else {
            return Ok(false);
        };
        if self.confirmed_doc == Some(current) {
            return Ok(self.confirmed);
        }

        self.child_scores.fill(None);
        for index in 0..self.children.len() {
            if !self.essential[index] || self.children[index].doc() != Some(current) {
                continue;
            }
            if self.children[index].matches()? {
                self.child_scores[index] = Some(self.children[index].score()?);
            }
        }

        if self.partial_score_upper() >= self.min_competitive_score {
            for index in 0..self.children.len() {
                if self.essential[index] || self.child_upper_bounds[index] == 0.0 {
                    continue;
                }
                if self.children[index].doc().is_some_and(|doc| doc < current) {
                    self.children[index].advance(current)?;
                }
                if self.children[index].doc() == Some(current) && self.children[index].matches()? {
                    self.child_scores[index] = Some(self.children[index].score()?);
                }
            }
        }

        let mut has_match = false;
        let mut score = 0.0_f32;
        for child_score in self.child_scores.iter().flatten() {
            has_match = true;
            score += *child_score;
        }
        score = checked_score(score, "FTS SHOULD MAXSCORE")?;
        self.confirmed = has_match && score >= self.min_competitive_score;
        self.current_score = self.confirmed.then_some(score);
        self.confirmed_doc = Some(current);
        Ok(self.confirmed)
    }

    fn combined_shallow_bounds(&self, target: u64) -> ReportedBounds {
        ReportedBounds {
            target,
            up_to: u64::MAX,
            bounds: ScoreBounds {
                lower: 0.0,
                upper: self.global_score_upper_bound,
            },
        }
    }
}

impl ComposableScorer for ShouldMaxScoreScorer<'_> {
    fn doc(&self) -> Option<u64> {
        self.current
    }

    fn document_key(&self) -> Option<u64> {
        let current = self.current?;
        self.children
            .iter()
            .find(|child| child.doc() == Some(current))
            .and_then(|child| child.document_key())
    }

    fn next(&mut self) -> Result<Option<u64>> {
        if self.exhausted {
            return Ok(None);
        }
        if !self.initialized {
            self.initialize_next()?;
            return self.position(0);
        }
        let Some(current) = self.current else {
            return Ok(self.exhaust());
        };
        if current == u64::MAX {
            return Ok(self.exhaust());
        }
        for child in &mut self.children {
            if child.doc() == Some(current) {
                child.next()?;
            }
        }
        self.reset_current();
        self.position(current + 1)
    }

    fn advance(&mut self, target: u64) -> Result<Option<u64>> {
        if self.current.is_some_and(|current| current >= target) {
            return Ok(self.current);
        }
        if self.exhausted {
            return Ok(None);
        }
        self.initialize_advance(target)?;
        self.align_all_children(target)?;
        self.reset_current();
        self.window = None;
        self.position(target)
    }

    fn cost(&self) -> usize {
        self.children
            .iter()
            .map(|child| child.cost())
            .fold(0, usize::saturating_add)
    }

    fn score(&mut self) -> Result<f32> {
        if !self.ensure_confirmed()? {
            return Err(Error::internal(
                "score requested from an unconfirmed FTS SHOULD MAXSCORE document",
            ));
        }
        self.current_score.ok_or_else(|| {
            Error::internal("confirmed FTS SHOULD MAXSCORE document has no exact score")
        })
    }

    fn advance_shallow(&mut self, target: u64) -> Result<u64> {
        let target = self.current.map_or(target, |current| target.max(current));
        let reported = if let Some(window) = self.window
            && target >= window.start
            && target <= window.up_to
        {
            ReportedBounds {
                target,
                up_to: window.up_to,
                bounds: ScoreBounds {
                    lower: 0.0,
                    upper: window.combined_upper,
                },
            }
        } else {
            self.combined_shallow_bounds(target)
        };
        self.reported_bounds = Some(reported);
        Ok(reported.up_to)
    }

    fn score_bounds(&mut self, up_to: u64) -> Result<ScoreBounds> {
        let reported = self.reported_bounds.ok_or_else(|| {
            Error::internal("score_bounds requires advance_shallow on the FTS SHOULD scorer")
        })?;
        if up_to < reported.target || up_to > reported.up_to {
            return Err(Error::internal(format!(
                "FTS SHOULD score bound up_to={up_to} is outside shallow range [{}, {}]",
                reported.target, reported.up_to
            )));
        }
        Ok(reported.bounds)
    }

    fn global_score_upper_bound(&self) -> Option<f32> {
        Some(self.global_score_upper_bound)
    }

    fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()> {
        if min_score.is_nan() {
            return Err(Error::invalid_input(
                "minimum competitive FTS score cannot be NaN",
            ));
        }
        if min_score > self.min_competitive_score {
            self.min_competitive_score = min_score;
        }
        Ok(())
    }

    fn current_score_upper_bound(&mut self) -> Result<Option<f32>> {
        let Some(current) = self.current else {
            return Ok(None);
        };
        self.child_scores.fill(None);
        for (index, child) in self.children.iter_mut().enumerate() {
            if self.essential[index] {
                if child.doc() != Some(current) {
                    self.child_scores[index] = Some(0.0);
                    continue;
                }
                let Some(upper) = child.current_score_upper_bound()? else {
                    return Ok(None);
                };
                if !upper.is_finite() {
                    return Ok(None);
                }
                self.child_scores[index] = Some(upper.max(0.0));
            }
        }
        let mut upper = self.partial_score_upper();
        if upper < self.min_competitive_score {
            return Ok(Some(upper));
        }

        // The residual range bound was inconclusive. Tighten it with each
        // non-essential posting approximation for this document, largest
        // global bound first. Stop as soon as the unresolved residual can no
        // longer reach the floor, still without touching phrase positions.
        for index in self.bound_order.iter().rev().copied() {
            if self.essential[index] {
                continue;
            }
            let child = &mut self.children[index];
            if child.doc().is_some_and(|doc| doc < current) {
                child.advance(current)?;
            }
            self.child_scores[index] = if child.doc() == Some(current) {
                let Some(upper) = child.current_score_upper_bound()? else {
                    return Ok(None);
                };
                if !upper.is_finite() {
                    return Ok(None);
                }
                Some(upper.max(0.0))
            } else {
                Some(0.0)
            };
            upper = self.partial_score_upper();
            if upper < self.min_competitive_score {
                return Ok(Some(upper));
            }
        }
        Ok(upper.is_finite().then_some(upper))
    }

    fn supports_doc_local_confirmation_pruning(&self) -> bool {
        self.children
            .iter()
            .any(|child| child.supports_doc_local_confirmation_pruning())
    }

    fn matches(&mut self) -> Result<bool> {
        self.ensure_confirmed()
    }

    fn match_cost(&self) -> Option<f32> {
        self.children
            .iter()
            .filter_map(|child| child.match_cost())
            .reduce(|left, right| left + right)
    }

    fn scores_non_negative(&self) -> bool {
        true
    }
}
