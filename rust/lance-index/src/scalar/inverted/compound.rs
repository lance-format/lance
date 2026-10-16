// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

mod should_maxscore;

use std::cell::{Cell, RefCell};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};

use futures::{StreamExt, TryStreamExt, stream};
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu};
use lance_core::{Error, Result};
use lance_select::RowAddrMask;
use lance_tokenizer::{SimpleTokenizer, TextAnalyzer};

use super::{
    InvertedIndex, PreparedBm25Query,
    document_tokenizer::{DocType, JsonTokenizer, LanceTokenizer},
    documents::{
        CachedRowAddressOrder, DocId, DocLengths, DocVisibility, OrderedRowAddressProjection,
        PartitionDocuments, ResidentAddressProjection, RowAddressProjectionOrderError,
    },
    index::{DocSet, InvertedPartition},
    prepare_bm25_query,
    query::{
        FtsQuery, FtsSearchParams, MatchQuery, Operator, PhraseQuery, Tokens,
        try_collect_query_tokens,
    },
    scorer::MemBM25Scorer,
    tokenizer::document_tokenizer::TextTokenizer,
    wand::{
        FLAT_SEARCH_PERCENT_THRESHOLD, LegacyWandDocuments, ModernWandDocuments, PostingIterator,
        TermLeafScorer, WandCursor, WandDocuments, iu_tight_enabled, iu_tight_search,
        iu_tight_search_multi, score_sum_upper_bound_factor,
    },
};
use crate::{metrics::MetricsCollector, prefilter::PreFilter};

use self::should_maxscore::ShouldMaxScoreScorer;

const DEFAULT_BLOCK_SIZE: usize = 128;
const SCORE_FLOOR_RESOLUTION_BATCH_SIZE: usize = DEFAULT_BLOCK_SIZE;

/// One exact FTS result in a compound collector's candidate domain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ScoredRow<K = u64> {
    pub row_id: K,
    pub score: f32,
}

#[cfg(test)]
impl ScoredRow<u64> {
    pub(super) fn new(row_id: u64, score: f32) -> Result<Self> {
        if !score.is_finite() {
            return Err(Error::invalid_input(format!(
                "FTS score for row_id={row_id} must be finite, got {score}"
            )));
        }
        Ok(Self { row_id, score })
    }
}

/// Conservative score bounds for a document range.
///
/// The lower bound is needed by signed compositions such as [`BoostScorer`].
/// Arithmetic widens both sides by one representable `f32` so nested
/// operations cannot round an upper bound below an exact score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ScoreBounds {
    pub(super) lower: f32,
    pub(super) upper: f32,
}

impl ScoreBounds {
    pub(super) const ZERO: Self = Self {
        lower: 0.0,
        upper: 0.0,
    };
    pub(super) const UNBOUNDED: Self = Self {
        lower: f32::NEG_INFINITY,
        upper: f32::INFINITY,
    };

    pub(super) fn try_new(lower: f32, upper: f32) -> Result<Self> {
        let bounds = Self { lower, upper };
        if !bounds.is_valid_for_finite_scores() {
            return Err(Error::invalid_input(format!(
                "FTS score bounds require an ordered interval that can contain finite scores, got [{lower}, {upper}]"
            )));
        }
        Ok(bounds)
    }

    #[cfg(test)]
    pub(super) fn lower(self) -> f32 {
        self.lower
    }

    #[cfg(test)]
    pub(super) fn upper(self) -> f32 {
        self.upper
    }

    fn is_valid_for_finite_scores(self) -> bool {
        !self.lower.is_nan()
            && !self.upper.is_nan()
            && self.lower <= self.upper
            && self.lower != f32::INFINITY
            && self.upper != f32::NEG_INFINITY
    }

    pub(super) fn contains(self, score: f32) -> bool {
        score.is_finite() && self.lower <= score && score <= self.upper
    }

    fn point(score: f32) -> Result<Self> {
        if !score.is_finite() {
            return Err(Error::invalid_input(format!(
                "FTS score bounds require a finite score, got {score}"
            )));
        }
        Ok(Self {
            lower: score,
            upper: score,
        })
    }

    fn scale_non_negative(self, factor: f32) -> Self {
        debug_assert!(factor.is_finite() && factor >= 0.0);
        if factor == 0.0 {
            return Self::ZERO;
        }
        if !self.lower.is_finite() || !self.upper.is_finite() {
            return Self::UNBOUNDED;
        }
        Self {
            lower: next_down(self.lower * factor),
            upper: next_up(self.upper * factor),
        }
    }

    fn include_zero(self) -> Self {
        Self {
            lower: self.lower.min(0.0),
            upper: self.upper.max(0.0),
        }
    }

    fn add(self, other: Self) -> Self {
        if !self.lower.is_finite()
            || !self.upper.is_finite()
            || !other.lower.is_finite()
            || !other.upper.is_finite()
        {
            return Self::UNBOUNDED;
        }
        Self {
            lower: next_down(self.lower + other.lower),
            upper: next_up(self.upper + other.upper),
        }
    }

    fn subtract_scaled(self, other: Self, factor: f32) -> Self {
        let penalty = other.scale_non_negative(factor);
        if !self.lower.is_finite()
            || !self.upper.is_finite()
            || !penalty.lower.is_finite()
            || !penalty.upper.is_finite()
        {
            return Self::UNBOUNDED;
        }
        Self {
            lower: next_down(self.lower - penalty.upper),
            upper: next_up(self.upper - penalty.lower),
        }
    }
}

fn next_up(value: f32) -> f32 {
    if !value.is_finite() {
        return value;
    }
    if value == 0.0 {
        return f32::from_bits(1);
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f32::from_bits(bits + 1)
    } else {
        f32::from_bits(bits - 1)
    }
}

fn next_down(value: f32) -> f32 {
    if !value.is_finite() {
        return value;
    }
    if value == 0.0 {
        return f32::from_bits((1_u32 << 31) | 1);
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f32::from_bits(bits - 1)
    } else {
        f32::from_bits(bits + 1)
    }
}

/// Find the greatest finite non-negative raw score whose actual `f32`
/// multiplication remains strictly below an inclusive scaled score floor.
///
/// Non-negative finite `f32` values have the same order as their bit patterns,
/// and multiplication by a finite positive factor is monotonic. Binary search
/// therefore proves the returned exclusive child floor cannot discard a raw
/// score whose scaled value is equal to `scaled_score_floor`.
#[doc(hidden)]
pub fn exclusive_scaled_score_floor(scaled_score_floor: f32, factor: f32) -> Option<f32> {
    if !scaled_score_floor.is_finite()
        || scaled_score_floor <= 0.0
        || !factor.is_finite()
        || factor <= 0.0
    {
        return None;
    }

    let mut lower_bits = 0_u32;
    let mut upper_bits = f32::MAX.to_bits();
    while lower_bits < upper_bits {
        let midpoint = lower_bits + (upper_bits - lower_bits).div_ceil(2);
        let raw_score = f32::from_bits(midpoint);
        if raw_score * factor < scaled_score_floor {
            lower_bits = midpoint;
        } else {
            upper_bits = midpoint - 1;
        }
    }
    Some(f32::from_bits(lower_bits))
}

fn checked_score(score: f32, context: &str) -> Result<f32> {
    if score.is_finite() {
        Ok(score)
    } else {
        Err(Error::invalid_input(format!(
            "{context} produced a non-finite FTS score: {score}"
        )))
    }
}

/// Internal document-at-a-time scorer protocol for compound FTS.
///
/// Implementations iterate matching partition-local document ids in ascending
/// order and expose the corresponding candidate key separately. A collector
/// may shallow-advance independently of the exact iterator, inspect a
/// conservative range bound, and monotonically raise the competitive score.
/// `matches` is the optional two-phase confirmation hook: cheap approximations
/// return a candidate from `next` / `advance` and defer expensive checks such
/// as phrase positions until confirmation.
pub(super) trait ComposableScorer: Send {
    fn doc(&self) -> Option<u64>;
    fn document_key(&self) -> Option<u64> {
        self.doc()
    }
    fn next(&mut self) -> Result<Option<u64>>;
    fn advance(&mut self, target: u64) -> Result<Option<u64>>;
    fn cost(&self) -> usize;
    fn score(&mut self) -> Result<f32>;
    fn advance_shallow(&mut self, target: u64) -> Result<u64>;
    fn score_bounds(&mut self, up_to: u64) -> Result<ScoreBounds>;
    /// Conservative list-wide score upper bound, independent of iterator
    /// position: it must cover every document of the list, not a range.
    /// Understating it lets a parent prune documents that can still compete,
    /// so implementations return `None` rather than guess. `None` keeps the
    /// scorer on exact eager composition paths.
    fn global_score_upper_bound(&self) -> Option<f32> {
        None
    }
    fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()>;

    /// Conservative score upper bound for the current approximation without
    /// running two-phase confirmation. `None` disables doc-local pruning.
    fn current_score_upper_bound(&mut self) -> Result<Option<f32>> {
        Ok(None)
    }

    fn supports_doc_local_confirmation_pruning(&self) -> bool {
        false
    }

    fn matches(&mut self) -> Result<bool> {
        Ok(true)
    }

    /// Estimated relative cost of [`Self::matches`], stable for this scorer's
    /// lifetime. `None` means no ordering hint, not that confirmation may be skipped.
    fn match_cost(&self) -> Option<f32> {
        None
    }

    fn scores_non_negative(&self) -> bool {
        false
    }

    /// Drain one cached shallow window of confirmed hits.
    ///
    /// An empty window is not exhaustion: the collector must call again after a
    /// skipped or hitless range. Emit every confirmed document whose score is
    /// at least `min_score` (inclusive). `max_hits` caps this window only; the
    /// scorer stays on the first document that did not fit so the next call
    /// can continue without re-emitting.
    fn collect_confirmed_window(
        &mut self,
        min_score: f32,
        out: &mut Vec<ConfirmedHit>,
        max_hits: usize,
    ) -> Result<WindowCollect> {
        collect_confirmed_window_default(self, min_score, out, max_hits)
    }

    #[cfg(test)]
    fn debug_type_name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    #[cfg(test)]
    fn debug_child_type_names(&self) -> Vec<&'static str> {
        Vec::new()
    }

    #[cfg(test)]
    fn debug_last_min_competitive_score(&self) -> Option<f32> {
        None
    }
}

/// Upper bound on hits one `collect_confirmed_window` drain may buffer. The
/// drain pauses and re-offers, so this bounds scratch memory without changing
/// which documents are collected.
pub(super) const CONFIRMED_WINDOW_MAX_HITS: usize = 4096;

/// One confirmed hit produced by [`ComposableScorer::collect_confirmed_window`].
#[derive(Debug, Clone, Copy)]
pub(super) struct ConfirmedHit {
    pub document_key: u64,
    pub score: f32,
}

/// Status of one shallow-window drain.
///
/// Zero hits does not mean the scorer is finished. Only [`Self::exhausted`]
/// tells the collector to stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct WindowCollect {
    pub exhausted: bool,
}

pub(super) type BoxScorer<'a> = Box<dyn ComposableScorer + 'a>;

fn collect_confirmed_window_default<S: ComposableScorer + ?Sized>(
    scorer: &mut S,
    min_score: f32,
    out: &mut Vec<ConfirmedHit>,
    max_hits: usize,
) -> Result<WindowCollect> {
    if max_hits == 0 {
        return Ok(WindowCollect { exhausted: true });
    }
    scorer.set_min_competitive_score(min_score)?;
    let Some(doc_id) = (match scorer.doc() {
        Some(doc) => Some(doc),
        None => scorer.next()?,
    }) else {
        return Ok(WindowCollect { exhausted: true });
    };

    let up_to = scorer.advance_shallow(doc_id)?;
    let bounds = scorer.score_bounds(up_to)?;
    if bounds.upper < min_score {
        let next = if up_to == u64::MAX {
            None
        } else {
            scorer.advance(up_to + 1)?
        };
        return Ok(WindowCollect {
            exhausted: next.is_none(),
        });
    }

    loop {
        if out.len() >= max_hits {
            return Ok(WindowCollect { exhausted: false });
        }
        if min_score.is_finite()
            && scorer.supports_doc_local_confirmation_pruning()
            && scorer
                .current_score_upper_bound()?
                .is_some_and(|upper| upper < min_score)
        {
            match scorer.next()? {
                // `next` feeds only this guard; the walk resumes from the
                // scorer's own position.
                Some(next) if next <= up_to => continue,
                Some(_) => return Ok(WindowCollect { exhausted: false }),
                None => return Ok(WindowCollect { exhausted: true }),
            }
        }
        if let Some(match_cost) = scorer.match_cost()
            && (!match_cost.is_finite() || match_cost < 0.0)
        {
            return Err(Error::internal(format!(
                "FTS scorer reported invalid two-phase match cost: {match_cost}"
            )));
        }
        if scorer.matches()? {
            let score = checked_score(scorer.score()?, "compound scorer")?;
            if score >= min_score {
                let document_key = scorer.document_key().ok_or_else(|| {
                    Error::internal("compound FTS scorer did not expose its current document key")
                })?;
                out.push(ConfirmedHit {
                    document_key,
                    score,
                });
            }
        }
        match scorer.next()? {
            // Same as above: the guard is the only read of `next`.
            Some(next) if next <= up_to => continue,
            Some(_) => return Ok(WindowCollect { exhausted: false }),
            None => return Ok(WindowCollect { exhausted: true }),
        }
    }
}

fn sum_global_score_upper_bounds(children: &[BoxScorer<'_>]) -> Option<f32> {
    children.iter().try_fold(0.0, |upper, child| {
        let child_upper = child.global_score_upper_bound()?;
        if !child_upper.is_finite() || child_upper < 0.0 {
            return None;
        }
        let combined = ScoreBounds { lower: 0.0, upper }
            .add(ScoreBounds {
                lower: 0.0,
                upper: child_upper,
            })
            .upper;
        combined.is_finite().then_some(combined)
    })
}

/// Posting-payload-independent metadata for one semantic leaf in a staged
/// search task.
///
/// Bounds describe the unboosted leaf score. [`CompoundScorerPlan`] applies
/// the `MatchQuery` boost recorded in its leaf node while composing the root
/// interval. An impossible leaf contributes neither candidates nor score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct CompoundLeafPlanInput {
    pub(super) possible: bool,
    pub(super) cost: usize,
    pub(super) bounds: ScoreBounds,
}

impl CompoundLeafPlanInput {
    pub(super) fn new(possible: bool, cost: usize, bounds: ScoreBounds) -> Self {
        Self {
            possible,
            cost,
            bounds,
        }
    }
}

/// Pure metadata analysis used before staged source I/O begins.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct CompoundPlanAnalysis {
    pub(super) possible: bool,
    pub(super) bounds: ScoreBounds,
    pub(super) generator_cost: usize,
    pub(super) generator_leaves: Vec<usize>,
}

#[derive(Debug)]
struct NodePlanAnalysis {
    possible: bool,
    bounds: ScoreBounds,
    generator_cost: usize,
    generator_leaves: Vec<usize>,
}

impl NodePlanAnalysis {
    fn impossible() -> Self {
        Self {
            possible: false,
            bounds: ScoreBounds::ZERO,
            generator_cost: 0,
            generator_leaves: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) enum CompoundScorerPlan {
    Leaf {
        index: usize,
        boost: f32,
    },
    Boost {
        positive: Box<Self>,
        negative: Box<Self>,
        negative_boost: f32,
    },
    MultiMatch(Vec<Self>),
    Boolean {
        should: Vec<Self>,
        must: Vec<Self>,
        must_not: Vec<Self>,
    },
}

impl CompoundScorerPlan {
    pub(super) fn leaf_count(&self) -> usize {
        match self {
            Self::Leaf { .. } => 1,
            Self::Boost {
                positive, negative, ..
            } => positive.leaf_count().saturating_add(negative.leaf_count()),
            Self::MultiMatch(children) => children
                .iter()
                .fold(0, |count, child| count.saturating_add(child.leaf_count())),
            Self::Boolean {
                should,
                must,
                must_not,
            } => should
                .iter()
                .chain(must)
                .chain(must_not)
                .fold(0, |count, child| count.saturating_add(child.leaf_count())),
        }
    }

    /// Select a complete positive generator cover and compose conservative
    /// root score bounds without constructing or loading any scorer.
    pub(super) fn analyze_leaves(
        &self,
        leaves: &[CompoundLeafPlanInput],
    ) -> Result<CompoundPlanAnalysis> {
        let leaf_count = self.leaf_count();
        if leaves.len() != leaf_count {
            return Err(Error::internal(format!(
                "compound FTS plan has {leaf_count} leaves but received {} staged leaf inputs",
                leaves.len()
            )));
        }
        for (index, leaf) in leaves.iter().enumerate() {
            if !leaf.bounds.is_valid_for_finite_scores() {
                return Err(Error::internal(format!(
                    "compound FTS staged leaf {index} reported invalid score bounds [{}, {}]",
                    leaf.bounds.lower, leaf.bounds.upper
                )));
            }
        }

        let mut seen = vec![false; leaf_count];
        self.validate_leaf_indices(&mut seen)?;
        if let Some(missing) = seen.iter().position(|seen| !*seen) {
            return Err(Error::internal(format!(
                "compound FTS plan does not reference staged leaf {missing}"
            )));
        }

        let mut node = self.analyze_node(leaves)?;
        node.generator_leaves.sort_unstable();
        node.generator_leaves.dedup();
        Ok(CompoundPlanAnalysis {
            possible: node.possible,
            bounds: node.bounds,
            generator_cost: node.generator_cost,
            generator_leaves: node.generator_leaves,
        })
    }

    fn validate_leaf_indices(&self, seen: &mut [bool]) -> Result<()> {
        match self {
            Self::Leaf { index, .. } => {
                let slot_count = seen.len();
                let slot = seen.get_mut(*index).ok_or_else(|| {
                    Error::internal(format!(
                        "compound FTS plan references staged leaf {index}, but only {} slots exist",
                        slot_count
                    ))
                })?;
                if *slot {
                    return Err(Error::internal(format!(
                        "compound FTS plan references staged leaf {index} more than once"
                    )));
                }
                *slot = true;
                Ok(())
            }
            Self::Boost {
                positive, negative, ..
            } => {
                positive.validate_leaf_indices(seen)?;
                negative.validate_leaf_indices(seen)
            }
            Self::MultiMatch(children) => {
                for child in children {
                    child.validate_leaf_indices(seen)?;
                }
                Ok(())
            }
            Self::Boolean {
                should,
                must,
                must_not,
            } => {
                for child in should.iter().chain(must).chain(must_not) {
                    child.validate_leaf_indices(seen)?;
                }
                Ok(())
            }
        }
    }

    fn analyze_node(&self, leaves: &[CompoundLeafPlanInput]) -> Result<NodePlanAnalysis> {
        match self {
            Self::Leaf { index, boost } => {
                if !boost.is_finite() || *boost < 0.0 {
                    return Err(Error::invalid_input(format!(
                        "MatchQuery boost must be finite and non-negative, got {boost}"
                    )));
                }
                let leaf = leaves.get(*index).ok_or_else(|| {
                    Error::internal(format!(
                        "compound FTS plan references missing staged leaf {index}"
                    ))
                })?;
                if !leaf.possible {
                    return Ok(NodePlanAnalysis::impossible());
                }
                Ok(NodePlanAnalysis {
                    possible: true,
                    bounds: leaf.bounds.scale_non_negative(*boost),
                    generator_cost: leaf.cost,
                    generator_leaves: vec![*index],
                })
            }
            Self::Boost {
                positive,
                negative,
                negative_boost,
            } => {
                if !negative_boost.is_finite() || *negative_boost < 0.0 {
                    return Err(Error::invalid_input(format!(
                        "BoostQuery negative_boost must be finite and non-negative, got {negative_boost}"
                    )));
                }
                let positive = positive.analyze_node(leaves)?;
                let negative = negative.analyze_node(leaves)?;
                if !positive.possible {
                    return Ok(NodePlanAnalysis::impossible());
                }
                let bounds = if negative.possible {
                    positive
                        .bounds
                        .subtract_scaled(negative.bounds.include_zero(), *negative_boost)
                } else {
                    positive.bounds
                };
                Ok(NodePlanAnalysis {
                    possible: true,
                    bounds,
                    generator_cost: positive.generator_cost,
                    generator_leaves: positive.generator_leaves,
                })
            }
            Self::MultiMatch(children) => {
                let children = children
                    .iter()
                    .map(|child| child.analyze_node(leaves))
                    .collect::<Result<Vec<_>>>()?;
                let mut possible = children.iter().filter(|child| child.possible);
                let Some(first) = possible.next() else {
                    return Ok(NodePlanAnalysis::impossible());
                };
                let mut bounds = first.bounds;
                for child in possible {
                    bounds.lower = bounds.lower.min(child.bounds.lower);
                    bounds.upper = bounds.upper.max(child.bounds.upper);
                }
                let mut generator_cost = 0_usize;
                let mut generator_leaves = Vec::new();
                for child in children.into_iter().filter(|child| child.possible) {
                    generator_cost = generator_cost.saturating_add(child.generator_cost);
                    generator_leaves.extend(child.generator_leaves);
                }
                Ok(NodePlanAnalysis {
                    possible: true,
                    bounds,
                    generator_cost,
                    generator_leaves,
                })
            }
            Self::Boolean {
                should,
                must,
                must_not,
            } => {
                let should = should
                    .iter()
                    .map(|child| child.analyze_node(leaves))
                    .collect::<Result<Vec<_>>>()?;
                let must = must
                    .iter()
                    .map(|child| child.analyze_node(leaves))
                    .collect::<Result<Vec<_>>>()?;
                for child in must_not {
                    child.analyze_node(leaves)?;
                }

                if !must.is_empty() {
                    if must.iter().any(|child| !child.possible) {
                        return Ok(NodePlanAnalysis::impossible());
                    }
                    let mut bounds = ScoreBounds::ZERO;
                    for child in &must {
                        bounds = bounds.add(child.bounds);
                    }
                    for child in should.iter().filter(|child| child.possible) {
                        bounds = bounds.add(child.bounds.include_zero());
                    }
                    let mut must = must.into_iter();
                    let mut generator = must.next().ok_or_else(|| {
                        Error::internal("compound FTS Boolean MUST analysis lost its generator")
                    })?;
                    for child in must {
                        if child.generator_cost < generator.generator_cost {
                            generator = child;
                        }
                    }
                    return Ok(NodePlanAnalysis {
                        possible: true,
                        bounds,
                        generator_cost: generator.generator_cost,
                        generator_leaves: generator.generator_leaves,
                    });
                }

                let possible_should = should
                    .into_iter()
                    .filter(|child| child.possible)
                    .collect::<Vec<_>>();
                if possible_should.is_empty() {
                    return Ok(NodePlanAnalysis::impossible());
                }
                let mut bounds = ScoreBounds::ZERO;
                let mut generator_cost = 0_usize;
                let mut generator_leaves = Vec::new();
                for child in possible_should {
                    bounds = bounds.add(child.bounds.include_zero());
                    generator_cost = generator_cost.saturating_add(child.generator_cost);
                    generator_leaves.extend(child.generator_leaves);
                }
                Ok(NodePlanAnalysis {
                    possible: true,
                    bounds,
                    generator_cost,
                    generator_leaves,
                })
            }
        }
    }

    pub(super) fn from_query(query: &FtsQuery, num_leaves: &mut usize) -> Result<Self> {
        match query {
            FtsQuery::Match(query) => {
                let index = *num_leaves;
                *num_leaves += 1;
                Ok(Self::Leaf {
                    index,
                    boost: query.boost,
                })
            }
            FtsQuery::Phrase(_) => {
                let index = *num_leaves;
                *num_leaves += 1;
                Ok(Self::Leaf { index, boost: 1.0 })
            }
            FtsQuery::Boost(query) => Ok(Self::Boost {
                positive: Box::new(Self::from_query(&query.positive, num_leaves)?),
                negative: Box::new(Self::from_query(&query.negative, num_leaves)?),
                negative_boost: query.negative_boost,
            }),
            FtsQuery::MultiMatch(query) => Ok(Self::MultiMatch(
                query
                    .match_queries
                    .iter()
                    .map(|query| Self::from_query(&FtsQuery::Match(query.clone()), num_leaves))
                    .collect::<Result<Vec<_>>>()?,
            )),
            FtsQuery::Boolean(query) => Ok(Self::Boolean {
                should: query
                    .should
                    .iter()
                    .map(|query| Self::from_query(query, num_leaves))
                    .collect::<Result<Vec<_>>>()?,
                must: query
                    .must
                    .iter()
                    .map(|query| Self::from_query(query, num_leaves))
                    .collect::<Result<Vec<_>>>()?,
                must_not: query
                    .must_not
                    .iter()
                    .map(|query| Self::from_query(query, num_leaves))
                    .collect::<Result<Vec<_>>>()?,
            }),
            FtsQuery::CombinedFields(_) => Err(combined_fields_unsupported()),
        }
    }

    pub(super) fn build<'a>(&self, leaves: &mut [Option<BoxScorer<'a>>]) -> Result<BoxScorer<'a>> {
        match self {
            Self::Leaf { index, boost } => {
                let leaf = leaves
                    .get_mut(*index)
                    .and_then(Option::take)
                    .ok_or_else(|| {
                        Error::internal(format!(
                            "compound FTS scorer references missing leaf index {index}"
                        ))
                    })?;
                Ok(Box::new(ScaleScorer::try_new(leaf, *boost)?))
            }
            Self::Boost {
                positive,
                negative,
                negative_boost,
            } => Ok(Box::new(BoostScorer::try_new(
                positive.build(leaves)?,
                negative.build(leaves)?,
                *negative_boost,
            )?)),
            Self::MultiMatch(children) => Ok(Box::new(DisjunctionScorer::try_new(
                children
                    .iter()
                    .map(|child| child.build(leaves))
                    .collect::<Result<Vec<_>>>()?,
                DisjunctionScore::Max,
            )?)),
            Self::Boolean {
                should,
                must,
                must_not,
            } => BooleanScorer::try_root(
                should
                    .iter()
                    .map(|child| child.build(leaves))
                    .collect::<Result<Vec<_>>>()?,
                must.iter()
                    .map(|child| child.build(leaves))
                    .collect::<Result<Vec<_>>>()?,
                must_not
                    .iter()
                    .map(|child| child.build(leaves))
                    .collect::<Result<Vec<_>>>()?,
            ),
        }
    }
}

impl<D: WandDocuments + Sync> ComposableScorer for WandCursor<'_, D> {
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
        WandCursor::global_score_upper_bound(self)
    }

    fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()> {
        self.set_min_competitive_score(min_score)
    }

    fn current_score_upper_bound(&mut self) -> Result<Option<f32>> {
        self.current_score().map(Some)
    }

    fn supports_doc_local_confirmation_pruning(&self) -> bool {
        self.match_cost().is_some()
    }

    fn matches(&mut self) -> Result<bool> {
        self.matches()
    }

    fn match_cost(&self) -> Option<f32> {
        self.match_cost()
    }

    fn scores_non_negative(&self) -> bool {
        true
    }

    fn collect_confirmed_window(
        &mut self,
        min_score: f32,
        out: &mut Vec<ConfirmedHit>,
        max_hits: usize,
    ) -> Result<WindowCollect> {
        if max_hits == 0 {
            return Ok(WindowCollect { exhausted: true });
        }
        self.set_min_competitive_score(min_score)?;
        let Some(start) = (match self.doc() {
            Some(doc) => Some(doc),
            None => self.next()?,
        }) else {
            return Ok(WindowCollect { exhausted: true });
        };
        let up_to = self.advance_shallow(start)?;
        let upper = self.score_upper_bound(up_to)?;
        if upper < min_score {
            let next = if up_to == u64::MAX {
                None
            } else {
                self.advance(up_to + 1)?
            };
            return Ok(WindowCollect {
                exhausted: next.is_none(),
            });
        }

        loop {
            if out.len() >= max_hits {
                return Ok(WindowCollect { exhausted: false });
            }
            let Some(doc) = self.doc() else {
                return Ok(WindowCollect { exhausted: true });
            };
            if doc > up_to {
                return Ok(WindowCollect { exhausted: false });
            }
            let score = checked_score(self.current_score()?, "window drain emit")?;
            // Inclusive collector floor. wand_factor only raises the skip
            // threshold inside next(); it must not tighten this emit test.
            let competitive = score >= min_score;
            let confirmed = if competitive && self.match_cost().is_some() {
                self.matches()?
            } else {
                competitive
            };
            if confirmed && competitive {
                let document_key = self.document_key().ok_or_else(|| {
                    Error::internal("posting FTS scorer did not expose its current document key")
                })?;
                out.push(ConfirmedHit {
                    document_key,
                    score,
                });
            }
            match self.next()? {
                Some(next) if next <= up_to => {}
                Some(_) => return Ok(WindowCollect { exhausted: false }),
                None => return Ok(WindowCollect { exhausted: true }),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ShallowRange {
    target: u64,
    up_to: u64,
    block_index: Option<usize>,
}

/// Exact in-memory scorer used for unordered-address fallbacks and unit tests.
pub(super) struct MaterializedScorer {
    rows: Vec<ScoredRow>,
    block_size: usize,
    block_bounds: Box<[ScoreBounds]>,
    global_score_upper_bound: f32,
    index: Option<usize>,
    shallow: Option<ShallowRange>,
    min_competitive_score: f32,
    scores_non_negative: bool,
    #[cfg(test)]
    bound_score_visits: usize,
}

impl MaterializedScorer {
    pub(super) fn try_new(mut rows: Vec<ScoredRow>) -> Result<Self> {
        rows.sort_unstable_by_key(|row| row.row_id);
        for row in &rows {
            ScoreBounds::point(row.score)?;
        }
        for pair in rows.windows(2) {
            if pair[0].row_id == pair[1].row_id {
                return Err(Error::internal(format!(
                    "FTS leaf scorer produced duplicate row_id={}",
                    pair[0].row_id
                )));
            }
        }
        let block_size = DEFAULT_BLOCK_SIZE;
        let block_bounds = Self::build_block_bounds(&rows, block_size);
        let global_score_upper_bound = Self::global_upper_bound(&block_bounds);
        let scores_non_negative = rows.iter().all(|row| row.score >= 0.0);
        #[cfg(test)]
        let bound_score_visits = rows.len();
        Ok(Self {
            rows,
            block_size,
            block_bounds,
            global_score_upper_bound,
            index: None,
            shallow: None,
            min_competitive_score: f32::NEG_INFINITY,
            scores_non_negative,
            #[cfg(test)]
            bound_score_visits,
        })
    }

    #[cfg(test)]
    fn with_block_size(mut self, block_size: usize) -> Self {
        assert!(block_size > 0);
        self.block_size = block_size;
        self.block_bounds = Self::build_block_bounds(&self.rows, block_size);
        self.global_score_upper_bound = Self::global_upper_bound(&self.block_bounds);
        self.bound_score_visits = self.rows.len();
        self
    }

    fn build_block_bounds(rows: &[ScoredRow], block_size: usize) -> Box<[ScoreBounds]> {
        debug_assert!(block_size > 0);
        rows.chunks(block_size)
            .map(|block| {
                let first = block[0].score;
                let mut bounds = ScoreBounds {
                    lower: first,
                    upper: first,
                };
                for row in &block[1..] {
                    bounds.lower = bounds.lower.min(row.score);
                    bounds.upper = bounds.upper.max(row.score);
                }
                bounds
            })
            .collect()
    }

    fn global_upper_bound(block_bounds: &[ScoreBounds]) -> f32 {
        block_bounds
            .iter()
            .map(|bounds| bounds.upper)
            .max_by(f32::total_cmp)
            .unwrap_or(0.0)
    }

    fn block_bounds_at(&self, block_index: usize) -> Result<ScoreBounds> {
        self.block_bounds.get(block_index).copied().ok_or_else(|| {
            Error::internal(format!(
                "materialized FTS scorer has no score bounds for block {block_index}"
            ))
        })
    }

    #[cfg(test)]
    fn bound_score_visits(&self) -> usize {
        self.bound_score_visits
    }

    #[cfg(test)]
    fn num_bound_blocks(&self) -> usize {
        self.block_bounds.len()
    }

    fn block_end(&self, block_index: usize) -> usize {
        (block_index + 1)
            .saturating_mul(self.block_size)
            .min(self.rows.len())
    }

    fn block_index(&self, row_index: usize) -> usize {
        row_index / self.block_size
    }

    fn skip_non_competitive_block(&self, row_index: usize) -> Result<Option<usize>> {
        let block_index = self.block_index(row_index);
        if self.block_bounds_at(block_index)?.upper < self.min_competitive_score {
            Ok(Some(self.block_end(block_index)))
        } else {
            Ok(None)
        }
    }

    fn position_at(&mut self, mut index: usize) -> Result<Option<u64>> {
        while index < self.rows.len() {
            if let Some(next_index) = self.skip_non_competitive_block(index)? {
                index = next_index;
                continue;
            }
            self.index = Some(index);
            self.shallow = None;
            return Ok(Some(self.rows[index].row_id));
        }
        self.index = None;
        self.shallow = None;
        Ok(None)
    }
}

impl ComposableScorer for MaterializedScorer {
    fn doc(&self) -> Option<u64> {
        self.index.map(|index| self.rows[index].row_id)
    }

    fn next(&mut self) -> Result<Option<u64>> {
        self.position_at(self.index.map_or(0, |index| index + 1))
    }

    fn advance(&mut self, target: u64) -> Result<Option<u64>> {
        if self.doc().is_some_and(|doc| doc >= target) {
            return Ok(self.doc());
        }
        let start = self.index.map_or(0, |index| index + 1);
        let offset = self.rows[start..].partition_point(|row| row.row_id < target);
        self.position_at(start + offset)
    }

    fn cost(&self) -> usize {
        self.rows.len()
    }

    fn score(&mut self) -> Result<f32> {
        self.index
            .map(|index| self.rows[index].score)
            .ok_or_else(|| Error::internal("FTS scorer is not positioned on a document"))
    }

    fn advance_shallow(&mut self, target: u64) -> Result<u64> {
        let start = self.rows.partition_point(|row| row.row_id < target);
        if start == self.rows.len() {
            self.shallow = Some(ShallowRange {
                target,
                up_to: u64::MAX,
                block_index: None,
            });
            return Ok(u64::MAX);
        }
        let block_index = self.block_index(start);
        let end = self.block_end(block_index);
        let up_to = self
            .rows
            .get(end)
            .map(|next| next.row_id.saturating_sub(1))
            .unwrap_or(u64::MAX);
        self.shallow = Some(ShallowRange {
            target,
            up_to,
            block_index: Some(block_index),
        });
        Ok(up_to)
    }

    fn score_bounds(&mut self, up_to: u64) -> Result<ScoreBounds> {
        let shallow = self.shallow.ok_or_else(|| {
            Error::internal("score_bounds requires advance_shallow on the FTS scorer")
        })?;
        if up_to < shallow.target || up_to > shallow.up_to {
            return Err(Error::internal(format!(
                "FTS score bound up_to={up_to} is outside shallow range [{}, {}]",
                shallow.target, shallow.up_to
            )));
        }
        shallow
            .block_index
            .map_or(Ok(ScoreBounds::ZERO), |block_index| {
                self.block_bounds_at(block_index)
            })
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
        self.score().map(Some)
    }

    fn scores_non_negative(&self) -> bool {
        self.scores_non_negative
    }

    #[cfg(test)]
    fn debug_last_min_competitive_score(&self) -> Option<f32> {
        Some(self.min_competitive_score)
    }
}

#[derive(Debug, Clone, Copy)]
struct MappedShallowRange {
    target: u64,
    up_to: u64,
    source_target: u64,
    source_up_to: u64,
    has_source_docs: bool,
}

/// Project a strictly ordered partition-local scorer into the shared physical
/// row-address domain.
struct RowAddressScorer<'a> {
    source: BoxScorer<'a>,
    projection: OrderedRowAddressProjection,
    current: Option<u64>,
    exhausted: bool,
    shallow: Option<MappedShallowRange>,
}

impl<'a> RowAddressScorer<'a> {
    fn new(source: BoxScorer<'a>, projection: OrderedRowAddressProjection) -> Self {
        Self {
            source,
            projection,
            current: None,
            exhausted: false,
            shallow: None,
        }
    }

    fn set_source_position(&mut self, source_doc: Option<u64>) -> Result<Option<u64>> {
        self.shallow = None;
        let Some(source_doc) = source_doc else {
            self.current = None;
            self.exhausted = true;
            return Ok(None);
        };
        if self.source.doc() != Some(source_doc) {
            return Err(Error::internal(format!(
                "FTS source returned local document {source_doc} but reported position {:?}",
                self.source.doc()
            )));
        }
        let row_address = self.projection.address(source_doc).ok_or_else(|| {
            Error::internal(format!(
                "FTS source returned non-live or out-of-range local document {source_doc} for a projection with {} slots",
                self.projection.len()
            ))
        })?;
        if self.current.is_some_and(|current| row_address <= current) {
            return Err(Error::internal(format!(
                "FTS row-address projection moved from {:?} to non-increasing address {row_address}",
                self.current
            )));
        }
        self.current = Some(row_address);
        Ok(self.current)
    }

    fn ensure_positioned(&self) -> Result<()> {
        if self.current.is_none() {
            Err(Error::internal(
                "row-address FTS scorer is not positioned on a document",
            ))
        } else {
            Ok(())
        }
    }

    fn local_upper_bound(&self, global_up_to: u64, shallow: MappedShallowRange) -> Option<u64> {
        if global_up_to >= shallow.up_to {
            return Some(shallow.source_up_to);
        }
        let next_global = global_up_to.checked_add(1)?;
        match self.projection.lower_bound(next_global) {
            Some(next_local) => next_local
                .checked_sub(1)
                .map(|local| local.min(shallow.source_up_to)),
            None => Some(shallow.source_up_to),
        }
    }
}

impl ComposableScorer for RowAddressScorer<'_> {
    fn doc(&self) -> Option<u64> {
        self.current
    }

    fn document_key(&self) -> Option<u64> {
        self.current
    }

    fn next(&mut self) -> Result<Option<u64>> {
        if self.exhausted {
            return Ok(None);
        }
        let source_doc = self.source.next()?;
        self.set_source_position(source_doc)
    }

    fn advance(&mut self, target: u64) -> Result<Option<u64>> {
        if self.current.is_some_and(|current| current >= target) {
            return Ok(self.current);
        }
        if self.exhausted {
            return Ok(None);
        }
        let Some(source_target) = self.projection.lower_bound(target) else {
            self.current = None;
            self.exhausted = true;
            self.shallow = None;
            return Ok(None);
        };
        let source_doc = self.source.advance(source_target)?;
        if source_doc.is_some_and(|doc| doc < source_target) {
            return Err(Error::internal(format!(
                "FTS source advanced to local document {:?} before target {source_target}",
                source_doc
            )));
        }
        let row_address = self.set_source_position(source_doc)?;
        if row_address.is_some_and(|address| address < target) {
            return Err(Error::internal(format!(
                "FTS projection advanced to row address {:?} before target {target}",
                row_address
            )));
        }
        Ok(row_address)
    }

    fn cost(&self) -> usize {
        self.source.cost().min(self.projection.live_len())
    }

    fn score(&mut self) -> Result<f32> {
        self.ensure_positioned()?;
        self.source.score()
    }

    fn advance_shallow(&mut self, target: u64) -> Result<u64> {
        if self.exhausted {
            self.shallow = Some(MappedShallowRange {
                target,
                up_to: u64::MAX,
                source_target: 0,
                source_up_to: 0,
                has_source_docs: false,
            });
            return Ok(u64::MAX);
        }
        let Some(source_target) = self.projection.lower_bound(target) else {
            self.shallow = Some(MappedShallowRange {
                target,
                up_to: u64::MAX,
                source_target: 0,
                source_up_to: 0,
                has_source_docs: false,
            });
            return Ok(u64::MAX);
        };
        let source_target = self
            .source
            .doc()
            .map_or(source_target, |doc| source_target.max(doc));
        let source_up_to = self.source.advance_shallow(source_target)?;
        if source_up_to < source_target {
            return Err(Error::internal(format!(
                "FTS source returned shallow range ending at local document {source_up_to} before target {source_target}"
            )));
        }
        let up_to = self
            .projection
            .next_address(source_up_to)
            .map(|next| next.saturating_sub(1))
            .unwrap_or(u64::MAX);
        if up_to < target {
            return Err(Error::internal(format!(
                "FTS projection mapped local shallow end {source_up_to} to row address {up_to} before target {target}"
            )));
        }
        self.shallow = Some(MappedShallowRange {
            target,
            up_to,
            source_target,
            source_up_to,
            has_source_docs: true,
        });
        Ok(up_to)
    }

    fn score_bounds(&mut self, up_to: u64) -> Result<ScoreBounds> {
        let shallow = self.shallow.ok_or_else(|| {
            Error::internal("score_bounds requires advance_shallow on the row-address FTS scorer")
        })?;
        if up_to < shallow.target || up_to > shallow.up_to {
            return Err(Error::internal(format!(
                "FTS row-address score bound up_to={up_to} is outside shallow range [{}, {}]",
                shallow.target, shallow.up_to
            )));
        }
        if !shallow.has_source_docs {
            return Ok(ScoreBounds::ZERO);
        }
        let Some(source_up_to) = self.local_upper_bound(up_to, shallow) else {
            return Ok(ScoreBounds::ZERO);
        };
        if source_up_to < shallow.source_target {
            return Ok(ScoreBounds::ZERO);
        }
        self.source.score_bounds(source_up_to)
    }

    fn global_score_upper_bound(&self) -> Option<f32> {
        self.source.global_score_upper_bound()
    }

    fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()> {
        self.source.set_min_competitive_score(min_score)
    }

    fn current_score_upper_bound(&mut self) -> Result<Option<f32>> {
        self.ensure_positioned()?;
        self.source.current_score_upper_bound()
    }

    fn supports_doc_local_confirmation_pruning(&self) -> bool {
        self.source.supports_doc_local_confirmation_pruning()
    }

    fn matches(&mut self) -> Result<bool> {
        self.ensure_positioned()?;
        self.source.matches()
    }

    fn match_cost(&self) -> Option<f32> {
        self.source.match_cost()
    }

    fn scores_non_negative(&self) -> bool {
        self.source.scores_non_negative()
    }
}

fn projected_address(projection: &ResidentAddressProjection, source_doc: u64) -> Result<u64> {
    let source_doc = u32::try_from(source_doc).map_err(|_| {
        Error::index(format!(
            "FTS local document {source_doc} exceeds the modern u32 domain"
        ))
    })?;
    projection.address(DocId::new(source_doc)).ok_or_else(|| {
        Error::internal(format!(
            "FTS source returned non-live local document {source_doc} while materializing row addresses"
        ))
    })
}

fn materialize_row_address_scorer<'a>(
    source: BoxScorer<'a>,
    projection: &ResidentAddressProjection,
    collisions: &MaterializedProjectionCollisions,
) -> Result<Option<RowAddressSource<'a>>> {
    let mut mapped_documents = Vec::with_capacity(source.cost().min(DEFAULT_BLOCK_SIZE));
    let scorer = materialize_mapped_scorer(source, |local_doc| {
        let row_address = projected_address(projection, local_doc)?;
        mapped_documents.push((row_address, local_doc));
        Ok(row_address)
    })?;

    // Constructing the materialized scorer first preserves its more local
    // duplicate diagnostic when one leaf produces both colliding documents.
    // Only a valid leaf is then published to the source-wide tracker.
    collisions.register(&mapped_documents)?;
    Ok(scorer)
}

fn materialize_mapped_scorer<'a>(
    mut source: BoxScorer<'a>,
    mut map_document: impl FnMut(u64) -> Result<u64>,
) -> Result<Option<RowAddressSource<'a>>> {
    let mut rows = Vec::with_capacity(source.cost().min(DEFAULT_BLOCK_SIZE));
    let mut source_doc = source.next()?;
    while let Some(doc) = source_doc {
        if source.matches()? {
            rows.push(ScoredRow {
                row_id: map_document(doc)?,
                score: checked_score(source.score()?, "materialized row-address FTS scorer")?,
            });
        }
        source_doc = source.next()?;
    }
    let scorer = MaterializedScorer::try_new(rows)?;
    let min_possible_row_address = scorer.rows.first().map(|row| row.row_id);
    Ok(
        min_possible_row_address.map(|min_possible_row_address| RowAddressSource {
            min_possible_row_address,
            scorer: Box::new(scorer),
        }),
    )
}

#[derive(Debug, Default)]
struct MaterializedProjectionCollisions {
    local_doc_by_address: RefCell<std::collections::HashMap<u64, u64>>,
}

impl MaterializedProjectionCollisions {
    fn register(&self, mapped_documents: &[(u64, u64)]) -> Result<()> {
        let mut local_doc_by_address =
            self.local_doc_by_address.try_borrow_mut().map_err(|_| {
                Error::internal(
                    "materialized FTS row-address collision tracker is already borrowed",
                )
            })?;

        for &(row_address, local_doc) in mapped_documents {
            if let Some(&existing_local_doc) = local_doc_by_address.get(&row_address)
                && existing_local_doc != local_doc
            {
                return Err(Error::index(format!(
                    "FTS row address {row_address} maps to distinct local documents {existing_local_doc} and {local_doc} in one physical source"
                )));
            }
        }
        for &(row_address, local_doc) in mapped_documents {
            local_doc_by_address.insert(row_address, local_doc);
        }
        Ok(())
    }
}

/// Query-scoped projection state shared by every leaf from one physical
/// source. Preparation is O(1); ordered validation is triggered lazily only
/// when a dense source makes streaming cheaper than candidate materialization.
#[derive(Debug)]
pub(super) struct PreparedRowAddressProjection {
    projection: ResidentAddressProjection,
    collisions: MaterializedProjectionCollisions,
}

pub(super) fn prepare_row_address_projection(
    projection: &ResidentAddressProjection,
) -> PreparedRowAddressProjection {
    PreparedRowAddressProjection {
        projection: projection.clone(),
        collisions: MaterializedProjectionCollisions::default(),
    }
}

fn invalid_row_address_projection(error: RowAddressProjectionOrderError) -> Error {
    Error::index(format!("invalid FTS row-address projection: {error}"))
}

fn map_validated_scorer_to_row_addresses<'a>(
    source: BoxScorer<'a>,
    projection: &PreparedRowAddressProjection,
    validation: std::result::Result<OrderedRowAddressProjection, RowAddressProjectionOrderError>,
    local_document_lower_bound: u64,
) -> Result<Option<RowAddressSource<'a>>> {
    match validation {
        Ok(ordered) => {
            let Some(first_row_address) = ordered
                .address(local_document_lower_bound)
                .or_else(|| ordered.next_address(local_document_lower_bound))
            else {
                return Ok(None);
            };
            Ok(Some(RowAddressSource::new(
                first_row_address,
                Box::new(RowAddressScorer::new(source, ordered)),
            )))
        }
        Err(error @ RowAddressProjectionOrderError::Duplicate { .. }) => {
            Err(invalid_row_address_projection(error))
        }
        Err(RowAddressProjectionOrderError::OutOfOrder { .. }) => {
            materialize_row_address_scorer(source, &projection.projection, &projection.collisions)
        }
    }
}

/// A lazily initialized scorer source and a conservative lower bound on its
/// first possible row address.
pub(super) struct RowAddressSource<'a> {
    min_possible_row_address: u64,
    scorer: BoxScorer<'a>,
}

impl<'a> RowAddressSource<'a> {
    pub(super) fn new(min_possible_row_address: u64, scorer: BoxScorer<'a>) -> Self {
        Self {
            min_possible_row_address,
            scorer,
        }
    }

    pub(super) fn into_scorer(self) -> BoxScorer<'a> {
        self.scorer
    }
}

/// Map a partition-local scorer into the shared row-address domain.
///
/// Strictly ordered projections stay streaming. Descending or sufficiently
/// sparse unknown projections use an exact materialized fallback. Duplicate
/// projections are rejected because two local documents cannot share one row.
pub(super) fn map_scorer_to_row_addresses<'a>(
    source: BoxScorer<'a>,
    projection: &PreparedRowAddressProjection,
    local_document_lower_bound: u64,
) -> Result<Option<RowAddressSource<'a>>> {
    map_scorer_to_row_addresses_with_threshold(
        source,
        projection,
        local_document_lower_bound,
        *FLAT_SEARCH_PERCENT_THRESHOLD,
    )
}

fn map_scorer_to_row_addresses_with_threshold<'a>(
    source: BoxScorer<'a>,
    projection: &PreparedRowAddressProjection,
    local_document_lower_bound: u64,
    flat_search_percent_threshold: u64,
) -> Result<Option<RowAddressSource<'a>>> {
    match projection.projection.cached_row_address_order() {
        CachedRowAddressOrder::Ordered => map_validated_scorer_to_row_addresses(
            source,
            projection,
            projection.projection.try_ordered_row_addresses(),
            local_document_lower_bound,
        ),
        CachedRowAddressOrder::Duplicate => map_validated_scorer_to_row_addresses(
            source,
            projection,
            projection.projection.try_ordered_row_addresses(),
            local_document_lower_bound,
        ),
        CachedRowAddressOrder::OutOfOrder => {
            materialize_row_address_scorer(source, &projection.projection, &projection.collisions)
        }
        CachedRowAddressOrder::Unknown
            if projection.projection.should_materialize_unknown_projection(
                source.cost(),
                flat_search_percent_threshold,
            ) =>
        {
            materialize_row_address_scorer(source, &projection.projection, &projection.collisions)
        }
        CachedRowAddressOrder::Unknown => map_validated_scorer_to_row_addresses(
            source,
            projection,
            projection.projection.try_ordered_row_addresses(),
            local_document_lower_bound,
        ),
    }
}

#[derive(Debug, Clone, Copy)]
enum MergeShallowBounds {
    Current { source_index: usize },
    Global(ScoreBounds),
    Empty,
}

#[derive(Debug, Clone, Copy)]
struct MergeShallowRange {
    target: u64,
    up_to: u64,
    bounds: MergeShallowBounds,
}

/// Merge disjoint sources for one semantic leaf in their shared row-address
/// domain.
///
/// Exactly one source must own each row address. Keeping the current source
/// outside the heap makes both `next` and one-source `advance` O(log P), where
/// P is the number of sources, instead of scanning every source per hit.
pub(super) struct RowAddressMergeScorer<'a> {
    sources: Vec<BoxScorer<'a>>,
    source_minimums: Vec<u64>,
    pending: BinaryHeap<Reverse<(u64, usize)>>,
    heads: BinaryHeap<Reverse<(u64, usize)>>,
    active_sources: HashSet<usize>,
    current: Option<(u64, usize)>,
    shallow: Option<MergeShallowRange>,
    min_competitive_score: f32,
}

impl<'a> RowAddressMergeScorer<'a> {
    pub(super) fn try_new(sources: Vec<RowAddressSource<'a>>) -> Result<Self> {
        if sources.is_empty() {
            return Err(Error::internal(
                "row-address merge scorer requires at least one source",
            ));
        }
        let mut pending = BinaryHeap::with_capacity(sources.len());
        let mut source_minimums = Vec::with_capacity(sources.len());
        let mut scorers = Vec::with_capacity(sources.len());
        for (source_index, source) in sources.into_iter().enumerate() {
            pending.push(Reverse((source.min_possible_row_address, source_index)));
            source_minimums.push(source.min_possible_row_address);
            scorers.push(source.scorer);
        }
        Ok(Self {
            heads: BinaryHeap::with_capacity(scorers.len()),
            active_sources: HashSet::with_capacity(scorers.len()),
            sources: scorers,
            source_minimums,
            pending,
            current: None,
            shallow: None,
            min_competitive_score: f32::NEG_INFINITY,
        })
    }

    fn push_positioned_source(&mut self, source_index: usize, doc: u64) -> Result<()> {
        if self.sources[source_index].doc() != Some(doc) {
            return Err(Error::internal(format!(
                "FTS source {source_index} returned row address {doc} but reported position {:?}",
                self.sources[source_index].doc()
            )));
        }
        self.heads.push(Reverse((doc, source_index)));
        Ok(())
    }

    fn initialize_source(&mut self, source_index: usize, target: u64) -> Result<()> {
        if self.min_competitive_score > f32::NEG_INFINITY {
            self.sources[source_index].set_min_competitive_score(self.min_competitive_score)?;
        }
        let source_target = target.max(self.source_minimums[source_index]);
        if let Some(doc) = self.sources[source_index].advance(source_target)? {
            if doc < source_target {
                return Err(Error::internal(format!(
                    "FTS source {source_index} initialized at row address {doc} before target {source_target}"
                )));
            }
            self.active_sources.insert(source_index);
            self.push_positioned_source(source_index, doc)?;
        }
        Ok(())
    }

    fn ensure_candidate_head(&mut self, target: u64) -> Result<()> {
        loop {
            let actual_head = self.heads.peek().map(|Reverse((doc, _))| *doc);
            let pending_head = self.pending.peek().map(|Reverse((minimum, _))| *minimum);
            let should_initialize = match (actual_head, pending_head) {
                (_, None) => false,
                (None, Some(_)) => true,
                (Some(actual), Some(pending)) => pending <= actual,
            };
            if !should_initialize {
                return Ok(());
            }
            let Reverse((_, source_index)) = self.pending.pop().ok_or_else(|| {
                Error::internal("FTS pending source heap unexpectedly became empty")
            })?;
            self.initialize_source(source_index, target)?;
        }
    }

    fn select_current(&mut self, target: u64) -> Result<Option<u64>> {
        self.shallow = None;
        self.ensure_candidate_head(target)?;
        let Some(Reverse((doc, source_index))) = self.heads.pop() else {
            self.current = None;
            return Ok(None);
        };
        if let Some(Reverse((duplicate, duplicate_source))) = self.heads.peek()
            && *duplicate == doc
        {
            self.current = None;
            return Err(Error::internal(format!(
                "FTS sources {source_index} and {duplicate_source} produced duplicate row address {doc}"
            )));
        }
        self.current = Some((doc, source_index));
        Ok(Some(doc))
    }

    fn advance_source(&mut self, source_index: usize, target: u64) -> Result<()> {
        if let Some(doc) = self.sources[source_index].advance(target)? {
            if doc < target {
                return Err(Error::internal(format!(
                    "FTS source {source_index} advanced to row address {doc} before target {target}"
                )));
            }
            self.push_positioned_source(source_index, doc)?;
        } else {
            self.active_sources.remove(&source_index);
        }
        Ok(())
    }

    fn next_source(&mut self, source_index: usize) -> Result<()> {
        if let Some(doc) = self.sources[source_index].next()? {
            self.push_positioned_source(source_index, doc)?;
        } else {
            self.active_sources.remove(&source_index);
        }
        Ok(())
    }

    fn current_source_mut(&mut self) -> Result<&mut BoxScorer<'a>> {
        let (_, source_index) = self.current.ok_or_else(|| {
            Error::internal("row-address merge scorer is not positioned on a document")
        })?;
        Ok(&mut self.sources[source_index])
    }

    fn next_source_boundary(&self) -> Option<u64> {
        self.heads
            .peek()
            .map(|Reverse((doc, _))| *doc)
            .into_iter()
            .chain(self.pending.peek().map(|Reverse((minimum, _))| *minimum))
            .min()
    }

    fn global_range_bounds(&self) -> ScoreBounds {
        ScoreBounds {
            lower: if self.scores_non_negative() {
                0.0
            } else {
                f32::NEG_INFINITY
            },
            upper: self.global_score_upper_bound().unwrap_or(f32::INFINITY),
        }
    }
}

impl ComposableScorer for RowAddressMergeScorer<'_> {
    fn doc(&self) -> Option<u64> {
        self.current.map(|(doc, _)| doc)
    }

    fn document_key(&self) -> Option<u64> {
        self.doc()
    }

    fn next(&mut self) -> Result<Option<u64>> {
        let target = match self.current.take() {
            Some((u64::MAX, source_index)) => {
                self.active_sources.remove(&source_index);
                self.shallow = None;
                return Ok(None);
            }
            Some((doc, source_index)) => {
                self.next_source(source_index)?;
                doc + 1
            }
            None if self.heads.is_empty() && self.pending.is_empty() => return Ok(None),
            None => 0,
        };
        self.select_current(target)
    }

    fn advance(&mut self, target: u64) -> Result<Option<u64>> {
        if self.doc().is_some_and(|doc| doc >= target) {
            return Ok(self.doc());
        }
        if let Some((_, source_index)) = self.current.take() {
            self.advance_source(source_index, target)?;
        }
        while let Some(Reverse((doc, _))) = self.heads.peek() {
            if *doc >= target {
                break;
            }
            let Reverse((_, source_index)) = self
                .heads
                .pop()
                .ok_or_else(|| Error::internal("FTS source heap unexpectedly became empty"))?;
            self.advance_source(source_index, target)?;
        }
        self.select_current(target)
    }

    fn cost(&self) -> usize {
        self.sources
            .iter()
            .map(|source| source.cost())
            .fold(0, usize::saturating_add)
    }

    fn score(&mut self) -> Result<f32> {
        self.current_source_mut()?.score()
    }

    fn advance_shallow(&mut self, target: u64) -> Result<u64> {
        let Some((current, source_index)) = self.current else {
            self.shallow = Some(MergeShallowRange {
                target,
                up_to: u64::MAX,
                bounds: MergeShallowBounds::Empty,
            });
            return Ok(u64::MAX);
        };
        let next_source = self.next_source_boundary();
        if next_source.is_some_and(|boundary| boundary <= target) {
            let bounds = self.global_range_bounds();
            self.shallow = Some(MergeShallowRange {
                target,
                up_to: target,
                bounds: MergeShallowBounds::Global(bounds),
            });
            return Ok(target);
        }

        let source_target = target.max(current);
        let source_up_to = self.sources[source_index].advance_shallow(source_target)?;
        if source_up_to < source_target {
            return Err(Error::internal(format!(
                "FTS source {source_index} returned shallow range ending at {source_up_to} before target {source_target}"
            )));
        }
        let up_to = next_source
            .map(|boundary| source_up_to.min(boundary.saturating_sub(1)))
            .unwrap_or(source_up_to);
        self.shallow = Some(MergeShallowRange {
            target,
            up_to,
            bounds: MergeShallowBounds::Current { source_index },
        });
        Ok(up_to)
    }

    fn score_bounds(&mut self, up_to: u64) -> Result<ScoreBounds> {
        let shallow = self.shallow.ok_or_else(|| {
            Error::internal("score_bounds requires advance_shallow on the row-address merge scorer")
        })?;
        if up_to < shallow.target || up_to > shallow.up_to {
            return Err(Error::internal(format!(
                "FTS row-address merge bound up_to={up_to} is outside shallow range [{}, {}]",
                shallow.target, shallow.up_to
            )));
        }
        match shallow.bounds {
            MergeShallowBounds::Current { source_index } => {
                self.sources[source_index].score_bounds(up_to)
            }
            MergeShallowBounds::Global(bounds) => Ok(bounds),
            MergeShallowBounds::Empty => Ok(ScoreBounds::ZERO),
        }
    }

    fn global_score_upper_bound(&self) -> Option<f32> {
        self.sources
            .iter()
            .map(|source| source.global_score_upper_bound())
            .try_fold(f32::NEG_INFINITY, |upper, source_upper| {
                let source_upper = source_upper?;
                source_upper.is_finite().then_some(upper.max(source_upper))
            })
    }

    fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()> {
        if min_score.is_nan() {
            return Err(Error::invalid_input(
                "minimum competitive FTS score cannot be NaN",
            ));
        }
        if min_score <= self.min_competitive_score {
            return Ok(());
        }
        for source_index in self.active_sources.iter().copied() {
            self.sources[source_index].set_min_competitive_score(min_score)?;
        }
        self.min_competitive_score = min_score;
        Ok(())
    }

    fn current_score_upper_bound(&mut self) -> Result<Option<f32>> {
        self.current_source_mut()?.current_score_upper_bound()
    }

    fn supports_doc_local_confirmation_pruning(&self) -> bool {
        self.sources
            .iter()
            .any(|source| source.supports_doc_local_confirmation_pruning())
    }

    fn matches(&mut self) -> Result<bool> {
        self.current_source_mut()?.matches()
    }

    fn match_cost(&self) -> Option<f32> {
        self.sources
            .iter()
            .map(|source| source.match_cost())
            .try_fold(0.0_f32, |cost, source_cost| {
                source_cost.map(|source_cost| cost.max(source_cost))
            })
    }

    fn scores_non_negative(&self) -> bool {
        self.sources
            .iter()
            .all(|source| source.scores_non_negative())
    }
}

/// Monotonic score-only floor shared by partition-local top-k collectors.
///
/// Equal-score candidates are never pruned because final ordering also uses
/// row id. The score-only floor is therefore a safe lower bound even when
/// partitions encounter ties in different orders.
#[derive(Debug)]
pub(super) struct CompetitiveScore {
    bits: AtomicU32,
}

impl Default for CompetitiveScore {
    fn default() -> Self {
        Self {
            bits: AtomicU32::new(f32::NEG_INFINITY.to_bits()),
        }
    }
}

impl CompetitiveScore {
    fn get(&self) -> f32 {
        f32::from_bits(self.bits.load(AtomicOrdering::Relaxed))
    }

    fn raise(&self, score: f32) {
        debug_assert!(!score.is_nan());
        let mut current = self.bits.load(AtomicOrdering::Relaxed);
        while score > f32::from_bits(current) {
            match self.bits.compare_exchange_weak(
                current,
                score.to_bits(),
                AtomicOrdering::Relaxed,
                AtomicOrdering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct HeapRow<K>(ScoredRow<K>);

impl<K: PartialEq> PartialEq for HeapRow<K> {
    fn eq(&self, other: &Self) -> bool {
        self.0.row_id == other.0.row_id && self.0.score.to_bits() == other.0.score.to_bits()
    }
}

impl<K: Eq> Eq for HeapRow<K> {}

impl<K: Ord> PartialOrd for HeapRow<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: Ord> Ord for HeapRow<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        // The worst result is the heap maximum: lower score, then higher row id.
        other
            .0
            .score
            .total_cmp(&self.0.score)
            .then_with(|| self.0.row_id.cmp(&other.0.row_id))
    }
}

fn compare_scored_rows<K: Ord>(left: &ScoredRow<K>, right: &ScoredRow<K>) -> Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| left.row_id.cmp(&right.row_id))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CollectionStatus {
    Complete,
    ScoreFloorOverflow,
}

#[derive(Debug, Clone, Copy)]
enum TieHandling {
    ResolveByKey,
    RetainScoreFloor { max_buffered: usize },
}

/// The sole owner of top-k state for a compound scorer tree.
///
/// Resolved document keys use the normal `(score DESC, row_id ASC)` ordering
/// and keep exactly `limit` rows. An unresolved partition temporarily retains
/// its kth-score floor, but stops once the bounded resolution buffer fills.
pub(super) struct TopKCollector<K = u64> {
    limit: usize,
    heap: BinaryHeap<HeapRow<K>>,
    competitive_score: Arc<CompetitiveScore>,
    tie_handling: TieHandling,
}

impl<K: Copy + Ord> TopKCollector<K> {
    pub(super) fn new(limit: usize) -> Self {
        Self::with_competitive_score(limit, Arc::new(CompetitiveScore::default()))
    }

    pub(super) fn with_competitive_score(
        limit: usize,
        competitive_score: Arc<CompetitiveScore>,
    ) -> Self {
        Self::with_tie_handling(limit, competitive_score, TieHandling::ResolveByKey)
    }

    fn retaining_score_floor(
        limit: usize,
        competitive_score: Arc<CompetitiveScore>,
        max_buffered: usize,
    ) -> Self {
        debug_assert!(max_buffered >= limit);
        Self::with_tie_handling(
            limit,
            competitive_score,
            TieHandling::RetainScoreFloor { max_buffered },
        )
    }

    fn with_tie_handling(
        limit: usize,
        competitive_score: Arc<CompetitiveScore>,
        tie_handling: TieHandling,
    ) -> Self {
        Self {
            limit,
            heap: BinaryHeap::with_capacity(limit.min(DEFAULT_BLOCK_SIZE)),
            competitive_score,
            tie_handling,
        }
    }

    fn insert(&mut self, row: ScoredRow<K>) -> CollectionStatus {
        if self.limit == 0 {
            return CollectionStatus::Complete;
        }
        if self.heap.len() < self.limit {
            self.heap.push(HeapRow(row));
        } else {
            let worst = self.heap.peek().expect("a full top-k heap is non-empty").0;
            match row.score.total_cmp(&worst.score) {
                Ordering::Less => {}
                Ordering::Equal => match self.tie_handling {
                    TieHandling::ResolveByKey => {
                        if row.row_id < worst.row_id {
                            self.heap.pop();
                            self.heap.push(HeapRow(row));
                        }
                    }
                    TieHandling::RetainScoreFloor { max_buffered } => {
                        if self.heap.len() >= max_buffered {
                            self.raise_competitive_score();
                            return CollectionStatus::ScoreFloorOverflow;
                        }
                        self.heap.push(HeapRow(row));
                    }
                },
                Ordering::Greater => {
                    match self.tie_handling {
                        TieHandling::ResolveByKey => {
                            self.heap.pop();
                            self.heap.push(HeapRow(row));
                        }
                        TieHandling::RetainScoreFloor { max_buffered } => {
                            self.heap.push(HeapRow(row));
                            self.prune_obsolete_score_floors();
                            if self.heap.len() > max_buffered {
                                // This collector is discarded and the partition is
                                // retried with resolved keys. Keep its observable
                                // working set within the advertised bound meanwhile.
                                self.heap.pop();
                                self.raise_competitive_score();
                                return CollectionStatus::ScoreFloorOverflow;
                            }
                        }
                    }
                }
            }
        }
        self.raise_competitive_score();
        CollectionStatus::Complete
    }

    fn raise_competitive_score(&self) {
        if self.heap.len() >= self.limit
            && let Some(worst) = self.heap.peek()
        {
            self.competitive_score.raise(worst.0.score);
        }
    }

    fn min_competitive_score(&self) -> f32 {
        self.competitive_score.get()
    }

    fn prune_obsolete_score_floors(&mut self) {
        while self.heap.len() > self.limit {
            let floor = self
                .heap
                .peek()
                .expect("a non-empty top-k heap has a score floor")
                .0
                .score;
            let floor_count = self
                .heap
                .iter()
                .filter(|row| row.0.score.total_cmp(&floor) == Ordering::Equal)
                .count();
            if self.heap.len() - floor_count < self.limit {
                break;
            }
            self.heap
                .retain(|row| row.0.score.total_cmp(&floor) != Ordering::Equal);
        }
    }

    pub(super) fn collect_mapped(
        &mut self,
        scorer: &mut dyn ComposableScorer,
        mut map_document: impl FnMut(u64) -> Result<K>,
    ) -> Result<CollectionStatus> {
        if self.limit == 0 {
            return Ok(CollectionStatus::Complete);
        }
        let capacity_limit = match self.tie_handling {
            TieHandling::ResolveByKey => self.limit,
            TieHandling::RetainScoreFloor { max_buffered } => max_buffered,
        };
        let expected = scorer.cost().min(capacity_limit);
        self.heap
            .reserve(expected.saturating_sub(self.heap.capacity()));

        // A COUNT-style collector carries limit == usize::MAX; draining with
        // that as max_hits would buffer the whole match set in the scratch
        // window (24 bytes per hit) before a single insert. The drain pauses
        // and re-offers, so the cap only bounds that scratch vector.
        let window_cap = capacity_limit.min(CONFIRMED_WINDOW_MAX_HITS);

        let mut window = Vec::with_capacity(DEFAULT_BLOCK_SIZE.min(window_cap));
        loop {
            let min_score = self.competitive_score.get();
            if min_score <= 0.0 {
                // No usable floor: the drain filters nothing, so its per-window
                // bookkeeping is pure cost -- measured ~4x slower on COUNT for
                // small match sets. Walk the scorer document by document.
                scorer.set_min_competitive_score(min_score)?;
                let Some(doc_id) = (match scorer.doc() {
                    Some(doc) => Some(doc),
                    None => scorer.next()?,
                }) else {
                    break;
                };
                let up_to = scorer.advance_shallow(doc_id)?;
                // score_bounds positions the two-phase match cost; the drain
                // does the same before deciding whether a window can compete.
                let _bounds = scorer.score_bounds(up_to)?;
                if let Some(match_cost) = scorer.match_cost()
                    && (!match_cost.is_finite() || match_cost < 0.0)
                {
                    return Err(Error::internal(format!(
                        "FTS scorer reported invalid two-phase match cost: {match_cost}"
                    )));
                }
                if scorer.matches()? {
                    let score = checked_score(scorer.score()?, "compound scorer")?;
                    if score >= self.competitive_score.get() {
                        let document_key = scorer.document_key().ok_or_else(|| {
                            Error::internal(
                                "compound FTS scorer did not expose its current document key",
                            )
                        })?;
                        let status = self.insert(ScoredRow {
                            row_id: map_document(document_key)?,
                            score,
                        });
                        if status == CollectionStatus::ScoreFloorOverflow {
                            return Ok(status);
                        }
                    }
                }
                if scorer.next()?.is_none() {
                    break;
                }
                continue;
            }
            window.clear();
            let collect = scorer.collect_confirmed_window(min_score, &mut window, window_cap)?;
            for hit in &window {
                // A shared partition floor is already known to be globally
                // competitive. Scores strictly below it cannot enter final top-k.
                // Window emit is inclusive; re-check here because the heap
                // floor can rise while this window is being inserted.
                if hit.score >= self.competitive_score.get() {
                    let status = self.insert(ScoredRow {
                        row_id: map_document(hit.document_key)?,
                        score: hit.score,
                    });
                    if status == CollectionStatus::ScoreFloorOverflow {
                        return Ok(status);
                    }
                }
            }
            if collect.exhausted {
                break;
            }
        }

        Ok(CollectionStatus::Complete)
    }

    fn into_candidates(self) -> Vec<ScoredRow<K>> {
        let mut rows = self.heap.into_iter().map(|row| row.0).collect::<Vec<_>>();
        rows.sort_unstable_by(compare_scored_rows);
        rows
    }

    pub(super) fn into_rows(self) -> Vec<ScoredRow<K>> {
        let limit = self.limit;
        let mut rows = self.into_candidates();
        rows.truncate(limit);
        rows
    }
}

impl TopKCollector<u64> {
    pub(super) fn collect(mut self, scorer: &mut dyn ComposableScorer) -> Result<Vec<ScoredRow>> {
        self.collect_mapped(scorer, Ok)?;
        Ok(self.into_rows())
    }
}

/// Evaluate a compound query over exact, materialized leaf result sets.
///
/// This is the bridge used by query-local residual postings: it keeps Boolean,
/// Boost, and MultiMatch semantics in the same scorer tree as the on-disk
/// compound path while allowing a different posting source.
#[doc(hidden)]
pub fn materialized_compound_top_k(
    query: &FtsQuery,
    leaves: Vec<Vec<(u64, f32)>>,
    limit: usize,
) -> Result<(Vec<u64>, Vec<f32>)> {
    let mut leaf_count = 0;
    let plan = CompoundScorerPlan::from_query(query, &mut leaf_count)?;
    if leaf_count != leaves.len() {
        return Err(Error::internal(format!(
            "compound FTS planned {leaf_count} leaves but received {} materialized leaves",
            leaves.len()
        )));
    }
    let mut scorers = leaves
        .into_iter()
        .map(|rows| {
            let rows = rows
                .into_iter()
                .map(|(row_id, score)| ScoredRow { row_id, score })
                .collect();
            MaterializedScorer::try_new(rows).map(|scorer| Some(Box::new(scorer) as BoxScorer<'_>))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut scorer = plan.build(&mut scorers)?;
    let rows = TopKCollector::new(limit).collect(scorer.as_mut())?;
    Ok(rows.into_iter().map(|row| (row.row_id, row.score)).unzip())
}

#[derive(Debug, Clone, Copy)]
pub(super) enum DisjunctionScore {
    Sum,
    Max,
}

pub(super) struct EmptyScorer;

impl ComposableScorer for EmptyScorer {
    fn doc(&self) -> Option<u64> {
        None
    }

    fn next(&mut self) -> Result<Option<u64>> {
        Ok(None)
    }

    fn advance(&mut self, _target: u64) -> Result<Option<u64>> {
        Ok(None)
    }

    fn cost(&self) -> usize {
        0
    }

    fn score(&mut self) -> Result<f32> {
        Err(Error::internal(
            "score requested from an empty compound FTS scorer",
        ))
    }

    fn advance_shallow(&mut self, _target: u64) -> Result<u64> {
        Ok(u64::MAX)
    }

    fn score_bounds(&mut self, _up_to: u64) -> Result<ScoreBounds> {
        Ok(ScoreBounds::ZERO)
    }

    fn global_score_upper_bound(&self) -> Option<f32> {
        Some(0.0)
    }

    fn set_min_competitive_score(&mut self, _min_score: f32) -> Result<()> {
        Ok(())
    }

    fn current_score_upper_bound(&mut self) -> Result<Option<f32>> {
        Ok(Some(0.0))
    }

    fn scores_non_negative(&self) -> bool {
        true
    }
}

struct ScaleScorer<'a> {
    child: BoxScorer<'a>,
    factor: f32,
    last_parent_score_floor: f32,
    #[cfg(test)]
    score_floor_translations: usize,
}

impl<'a> ScaleScorer<'a> {
    fn try_new(child: BoxScorer<'a>, factor: f32) -> Result<Self> {
        if !factor.is_finite() || factor < 0.0 {
            return Err(Error::invalid_input(format!(
                "MatchQuery boost must be finite and non-negative, got {factor}"
            )));
        }
        Ok(Self {
            child,
            factor,
            last_parent_score_floor: f32::NEG_INFINITY,
            #[cfg(test)]
            score_floor_translations: 0,
        })
    }
}

impl ComposableScorer for ScaleScorer<'_> {
    fn doc(&self) -> Option<u64> {
        self.child.doc()
    }

    fn document_key(&self) -> Option<u64> {
        self.child.document_key()
    }

    fn next(&mut self) -> Result<Option<u64>> {
        self.child.next()
    }

    fn advance(&mut self, target: u64) -> Result<Option<u64>> {
        self.child.advance(target)
    }

    fn cost(&self) -> usize {
        self.child.cost()
    }

    fn score(&mut self) -> Result<f32> {
        checked_score(self.child.score()? * self.factor, "MatchQuery boost")
    }

    fn advance_shallow(&mut self, target: u64) -> Result<u64> {
        self.child.advance_shallow(target)
    }

    fn score_bounds(&mut self, up_to: u64) -> Result<ScoreBounds> {
        Ok(self
            .child
            .score_bounds(up_to)?
            .scale_non_negative(self.factor))
    }

    fn global_score_upper_bound(&self) -> Option<f32> {
        self.child
            .global_score_upper_bound()
            .map(|upper| {
                ScoreBounds { lower: 0.0, upper }
                    .scale_non_negative(self.factor)
                    .upper
            })
            .filter(|upper| upper.is_finite() && *upper >= 0.0)
    }

    fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()> {
        if min_score.is_nan() {
            return Err(Error::invalid_input(
                "minimum competitive MatchQuery score cannot be NaN",
            ));
        }
        if min_score <= self.last_parent_score_floor {
            return Ok(());
        }
        #[cfg(test)]
        {
            self.score_floor_translations += 1;
        }
        if let Some(child_floor) = exclusive_scaled_score_floor(min_score, self.factor) {
            self.child.set_min_competitive_score(child_floor)?;
        }
        self.last_parent_score_floor = min_score;
        Ok(())
    }

    fn current_score_upper_bound(&mut self) -> Result<Option<f32>> {
        Ok(self.child.current_score_upper_bound()?.map(|upper| {
            ScoreBounds { lower: 0.0, upper }
                .scale_non_negative(self.factor)
                .upper
        }))
    }

    fn supports_doc_local_confirmation_pruning(&self) -> bool {
        self.child.supports_doc_local_confirmation_pruning()
    }

    fn matches(&mut self) -> Result<bool> {
        self.child.matches()
    }

    fn match_cost(&self) -> Option<f32> {
        self.child.match_cost()
    }

    fn scores_non_negative(&self) -> bool {
        self.child.scores_non_negative()
    }

    #[cfg(test)]
    fn debug_child_type_names(&self) -> Vec<&'static str> {
        vec![self.child.debug_type_name()]
    }
}

/// Union scorer used for Boolean SHOULD sums and MultiMatch DisMax.
pub(super) struct DisjunctionScorer<'a> {
    children: Vec<BoxScorer<'a>>,
    mode: DisjunctionScore,
    current: Option<u64>,
    confirmed_doc: Option<u64>,
    confirmed: Vec<bool>,
    min_competitive_score: f32,
}

impl<'a> DisjunctionScorer<'a> {
    pub(super) fn try_new(children: Vec<BoxScorer<'a>>, mode: DisjunctionScore) -> Result<Self> {
        if children.is_empty() {
            return Err(Error::internal(
                "FTS disjunction scorer requires at least one child",
            ));
        }
        let confirmed = vec![false; children.len()];
        Ok(Self {
            children,
            mode,
            current: None,
            confirmed_doc: None,
            confirmed,
            min_competitive_score: f32::NEG_INFINITY,
        })
    }

    fn set_current_from_children(&mut self) -> Option<u64> {
        self.current = self.children.iter().filter_map(|child| child.doc()).min();
        self.confirmed_doc = None;
        self.confirmed.fill(false);
        self.current
    }

    fn ensure_confirmed(&mut self) -> Result<bool> {
        let Some(current) = self.current else {
            return Ok(false);
        };
        if self.confirmed_doc == Some(current) {
            return Ok(self.confirmed.iter().any(|matched| *matched));
        }
        self.confirmed.fill(false);
        for (matched, child) in self.confirmed.iter_mut().zip(&mut self.children) {
            if child.doc() == Some(current) {
                *matched = child.matches()?;
            }
        }
        self.confirmed_doc = Some(current);
        Ok(self.confirmed.iter().any(|matched| *matched))
    }
}

impl ComposableScorer for DisjunctionScorer<'_> {
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
        match self.current {
            None => {
                for child in &mut self.children {
                    child.next()?;
                }
            }
            Some(current) => {
                for child in &mut self.children {
                    if child.doc() == Some(current) {
                        child.next()?;
                    }
                }
            }
        }
        Ok(self.set_current_from_children())
    }

    fn advance(&mut self, target: u64) -> Result<Option<u64>> {
        if self.current.is_some_and(|current| current >= target) {
            return Ok(self.current);
        }
        for child in &mut self.children {
            if child.doc().is_none_or(|doc| doc < target) {
                child.advance(target)?;
            }
        }
        Ok(self.set_current_from_children())
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
                "FTS disjunction score requested for an unconfirmed document",
            ));
        }
        let mut score = match self.mode {
            DisjunctionScore::Sum => 0.0_f32,
            DisjunctionScore::Max => f32::NEG_INFINITY,
        };
        for (matched, child) in self.confirmed.iter().zip(&mut self.children) {
            if !matched {
                continue;
            }
            let child_score = child.score()?;
            score = match self.mode {
                DisjunctionScore::Sum => score + child_score,
                DisjunctionScore::Max => score.max(child_score),
            };
        }
        checked_score(score, "FTS disjunction")
    }

    fn advance_shallow(&mut self, target: u64) -> Result<u64> {
        let mut up_to = u64::MAX;
        for child in &mut self.children {
            if let Some(doc) = child.doc() {
                up_to = up_to.min(child.advance_shallow(target.max(doc))?);
            }
        }
        Ok(up_to)
    }

    fn score_bounds(&mut self, up_to: u64) -> Result<ScoreBounds> {
        let mut bounds = match self.mode {
            DisjunctionScore::Sum => ScoreBounds::ZERO,
            DisjunctionScore::Max => ScoreBounds {
                lower: f32::INFINITY,
                upper: f32::NEG_INFINITY,
            },
        };
        for child in &mut self.children {
            let child_bounds = if child.doc().is_some_and(|doc| doc <= up_to) {
                child.score_bounds(up_to)?
            } else {
                ScoreBounds::ZERO
            };
            bounds = match self.mode {
                DisjunctionScore::Sum => bounds.add(child_bounds.include_zero()),
                DisjunctionScore::Max => ScoreBounds {
                    lower: bounds.lower.min(child_bounds.lower),
                    upper: bounds.upper.max(child_bounds.upper),
                },
            };
        }
        if bounds.lower == f32::INFINITY {
            Ok(ScoreBounds::ZERO)
        } else {
            Ok(bounds)
        }
    }

    fn global_score_upper_bound(&self) -> Option<f32> {
        match self.mode {
            DisjunctionScore::Sum => sum_global_score_upper_bounds(&self.children),
            DisjunctionScore::Max => self.children.iter().try_fold(0.0_f32, |upper, child| {
                let child_upper = child.global_score_upper_bound()?;
                child_upper.is_finite().then_some(upper.max(child_upper))
            }),
        }
    }

    fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()> {
        if min_score.is_nan() {
            return Err(Error::invalid_input(
                "minimum competitive FTS score cannot be NaN",
            ));
        }
        if min_score <= self.min_competitive_score {
            return Ok(());
        }
        self.min_competitive_score = min_score;
        // A child below a DisMax threshold cannot affect a competitive max.
        // Sum scorers need sibling-global bounds before translating the floor,
        // so they keep it at this node and prune from their combined block bound.
        if matches!(self.mode, DisjunctionScore::Max) {
            for child in &mut self.children {
                child.set_min_competitive_score(min_score)?;
            }
        }
        Ok(())
    }

    fn current_score_upper_bound(&mut self) -> Result<Option<f32>> {
        let Some(current) = self.current else {
            return Ok(None);
        };
        let mut upper = 0.0_f32;
        for child in &mut self.children {
            if child.doc() != Some(current) {
                continue;
            }
            let Some(child_upper) = child.current_score_upper_bound()? else {
                return Ok(None);
            };
            if !child_upper.is_finite() {
                return Ok(None);
            }
            upper = match self.mode {
                DisjunctionScore::Sum => {
                    ScoreBounds { lower: 0.0, upper }
                        .add(ScoreBounds {
                            lower: 0.0,
                            upper: child_upper.max(0.0),
                        })
                        .upper
                }
                DisjunctionScore::Max => upper.max(child_upper),
            };
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
        self.children
            .iter()
            .all(|child| child.scores_non_negative())
    }

    #[cfg(test)]
    fn debug_child_type_names(&self) -> Vec<&'static str> {
        self.children
            .iter()
            .map(|child| child.debug_type_name())
            .collect()
    }
}

/// Intersection scorer that requires and scores every Boolean MUST child.
pub(super) struct RequiredConjunctionScorer<'a> {
    children: Vec<BoxScorer<'a>>,
    /// Child indices sorted by approximation cost, omitted when query order is
    /// already cheapest-first. `children` remains in query order so scoring and
    /// score-bound arithmetic stay bit-for-bit stable.
    approximation_order: Option<Vec<usize>>,
    /// Child indices sorted by two-phase confirmation cost. Children without a
    /// cost hint remain in query order after costed confirmations.
    confirmation_order: Option<Vec<usize>>,
    current: Option<u64>,
    confirmed_doc: Option<u64>,
    confirmed: bool,
}

fn align_conjunction_children(
    children: &mut [BoxScorer<'_>],
    mut target: u64,
    child_index: impl Fn(usize) -> usize,
) -> Result<Option<u64>> {
    loop {
        for position in 0..children.len() {
            let child = &mut children[child_index(position)];
            if child.doc().is_none_or(|doc| doc < target) {
                let Some(doc) = child.advance(target)? else {
                    return Ok(None);
                };
                target = target.max(doc);
            }
        }
        let min_doc = children.iter().filter_map(|child| child.doc()).min();
        let max_doc = children.iter().filter_map(|child| child.doc()).max();
        if min_doc == max_doc {
            return Ok(min_doc);
        }
        target = max_doc.ok_or_else(|| {
            Error::internal("FTS conjunction lost a child while aligning scorers")
        })?;
    }
}

fn compare_confirmation_cost(
    left: &dyn ComposableScorer,
    right: &dyn ComposableScorer,
) -> Ordering {
    match (left.match_cost(), right.match_cost()) {
        (Some(left), Some(right)) => left.total_cmp(&right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn confirm_conjunction_children(
    children: &mut [BoxScorer<'_>],
    child_index: impl Fn(usize) -> usize,
) -> Result<bool> {
    for position in 0..children.len() {
        if !children[child_index(position)].matches()? {
            return Ok(false);
        }
    }
    Ok(true)
}

impl<'a> RequiredConjunctionScorer<'a> {
    pub(super) fn try_new(children: Vec<BoxScorer<'a>>) -> Result<Self> {
        if children.is_empty() {
            return Err(Error::internal(
                "FTS conjunction scorer requires at least one child",
            ));
        }
        let approximation_order = if children
            .windows(2)
            .all(|pair| pair[0].cost() <= pair[1].cost())
        {
            None
        } else {
            let mut order = (0..children.len()).collect::<Vec<_>>();
            order.sort_by_key(|&index| (children[index].cost(), index));
            Some(order)
        };
        for (index, child) in children.iter().enumerate() {
            if let Some(match_cost) = child.match_cost()
                && (!match_cost.is_finite() || match_cost < 0.0)
            {
                return Err(Error::internal(format!(
                    "FTS conjunction child {index} reported invalid two-phase match cost: {match_cost}"
                )));
            }
        }
        let confirmation_order = if children.windows(2).all(|pair| {
            compare_confirmation_cost(pair[0].as_ref(), pair[1].as_ref()) != Ordering::Greater
        }) {
            None
        } else {
            let mut order = (0..children.len()).collect::<Vec<_>>();
            order.sort_by(|&left, &right| {
                compare_confirmation_cost(children[left].as_ref(), children[right].as_ref())
                    .then_with(|| left.cmp(&right))
            });
            Some(order)
        };
        Ok(Self {
            children,
            approximation_order,
            confirmation_order,
            current: None,
            confirmed_doc: None,
            confirmed: false,
        })
    }

    fn align(&mut self, target: u64) -> Result<Option<u64>> {
        self.current = if let Some(order) = &self.approximation_order {
            align_conjunction_children(&mut self.children, target, |position| order[position])?
        } else {
            align_conjunction_children(&mut self.children, target, |position| position)?
        };
        if self.current.is_some() {
            self.confirmed_doc = None;
            self.confirmed = false;
        }
        Ok(self.current)
    }

    fn ensure_confirmed(&mut self) -> Result<bool> {
        let Some(current) = self.current else {
            return Ok(false);
        };
        if self.confirmed_doc == Some(current) {
            return Ok(self.confirmed);
        }
        self.confirmed = if let Some(order) = &self.confirmation_order {
            confirm_conjunction_children(&mut self.children, |position| order[position])?
        } else {
            confirm_conjunction_children(&mut self.children, |position| position)?
        };
        self.confirmed_doc = Some(current);
        Ok(self.confirmed)
    }
}

impl ComposableScorer for RequiredConjunctionScorer<'_> {
    fn doc(&self) -> Option<u64> {
        self.current
    }

    fn document_key(&self) -> Option<u64> {
        self.children.first().and_then(|child| child.document_key())
    }

    fn next(&mut self) -> Result<Option<u64>> {
        let target = match self.current {
            None => 0,
            Some(u64::MAX) => return Ok(None),
            Some(current) => current + 1,
        };
        self.align(target)
    }

    fn advance(&mut self, target: u64) -> Result<Option<u64>> {
        if self.current.is_some_and(|current| current >= target) {
            return Ok(self.current);
        }
        self.align(target)
    }

    fn cost(&self) -> usize {
        self.children
            .iter()
            .map(|child| child.cost())
            .min()
            .unwrap_or(0)
    }

    fn score(&mut self) -> Result<f32> {
        if !self.ensure_confirmed()? {
            return Err(Error::internal(
                "FTS conjunction score requested for an unconfirmed document",
            ));
        }
        let mut score = 0.0_f32;
        for child in &mut self.children {
            score += child.score()?;
        }
        checked_score(score, "FTS conjunction")
    }

    fn advance_shallow(&mut self, target: u64) -> Result<u64> {
        let mut up_to = u64::MAX;
        for child in &mut self.children {
            let child_target = child.doc().map_or(target, |doc| target.max(doc));
            up_to = up_to.min(child.advance_shallow(child_target)?);
        }
        Ok(up_to)
    }

    fn score_bounds(&mut self, up_to: u64) -> Result<ScoreBounds> {
        let mut bounds = ScoreBounds::ZERO;
        for child in &mut self.children {
            bounds = bounds.add(child.score_bounds(up_to)?);
        }
        Ok(bounds)
    }

    fn global_score_upper_bound(&self) -> Option<f32> {
        self.scores_non_negative()
            .then(|| sum_global_score_upper_bounds(&self.children))
            .flatten()
    }

    fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()> {
        if min_score.is_nan() {
            return Err(Error::invalid_input(
                "minimum competitive FTS score cannot be NaN",
            ));
        }
        // Propagating the full conjunction floor to one child is unsafe because
        // individually sub-threshold MUST scores may sum to a competitive hit.
        if self.children.len() == 1 {
            self.children[0].set_min_competitive_score(min_score)?;
        }
        Ok(())
    }

    fn current_score_upper_bound(&mut self) -> Result<Option<f32>> {
        let Some(current) = self.current else {
            return Ok(None);
        };
        let mut bounds = ScoreBounds::ZERO;
        for child in &mut self.children {
            if child.doc() != Some(current) {
                return Ok(None);
            }
            let Some(upper) = child.current_score_upper_bound()? else {
                return Ok(None);
            };
            if !upper.is_finite() {
                return Ok(None);
            }
            bounds = bounds.add(ScoreBounds {
                lower: 0.0,
                upper: upper.max(0.0),
            });
        }
        Ok(bounds.upper.is_finite().then_some(bounds.upper))
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
        self.children
            .iter()
            .all(|child| child.scores_non_negative())
    }

    fn collect_confirmed_window(
        &mut self,
        min_score: f32,
        out: &mut Vec<ConfirmedHit>,
        max_hits: usize,
    ) -> Result<WindowCollect> {
        if self.children.len() == 1 {
            let status = self.children[0].collect_confirmed_window(min_score, out, max_hits)?;
            self.current = self.children[0].doc();
            self.confirmed_doc = None;
            self.confirmed = false;
            return Ok(status);
        }
        collect_confirmed_window_default(self, min_score, out, max_hits)
    }

    #[cfg(test)]
    fn debug_child_type_names(&self) -> Vec<&'static str> {
        self.children
            .iter()
            .map(|child| child.debug_type_name())
            .collect()
    }
}

/// Positive-driven Boost scorer with signed conservative bounds.
pub(super) struct BoostScorer<'a> {
    positive: BoxScorer<'a>,
    negative: BoxScorer<'a>,
    negative_boost: f32,
    negative_matches_doc: Option<u64>,
    negative_matches: bool,
}

impl<'a> BoostScorer<'a> {
    pub(super) fn try_new(
        positive: BoxScorer<'a>,
        negative: BoxScorer<'a>,
        negative_boost: f32,
    ) -> Result<Self> {
        if !negative_boost.is_finite() || negative_boost < 0.0 {
            return Err(Error::invalid_input(format!(
                "BoostQuery negative_boost must be finite and non-negative, got {negative_boost}"
            )));
        }
        Ok(Self {
            positive,
            negative,
            negative_boost,
            negative_matches_doc: None,
            negative_matches: false,
        })
    }

    fn reset_confirmation(&mut self) {
        self.negative_matches_doc = None;
        self.negative_matches = false;
    }

    fn confirm_negative(&mut self) -> Result<bool> {
        let Some(current) = self.positive.doc() else {
            return Ok(false);
        };
        if self.negative_matches_doc == Some(current) {
            return Ok(self.negative_matches);
        }
        self.negative_matches =
            self.negative.advance(current)? == Some(current) && self.negative.matches()?;
        self.negative_matches_doc = Some(current);
        Ok(self.negative_matches)
    }
}

impl ComposableScorer for BoostScorer<'_> {
    fn doc(&self) -> Option<u64> {
        self.positive.doc()
    }

    fn document_key(&self) -> Option<u64> {
        self.positive.document_key()
    }

    fn next(&mut self) -> Result<Option<u64>> {
        self.reset_confirmation();
        self.positive.next()
    }

    fn advance(&mut self, target: u64) -> Result<Option<u64>> {
        self.reset_confirmation();
        self.positive.advance(target)
    }

    fn cost(&self) -> usize {
        self.positive.cost()
    }

    fn score(&mut self) -> Result<f32> {
        let positive = self.positive.score()?;
        let score = if self.confirm_negative()? {
            positive - self.negative_boost * self.negative.score()?
        } else {
            positive
        };
        checked_score(score, "BoostQuery scorer")
    }

    fn advance_shallow(&mut self, target: u64) -> Result<u64> {
        let mut up_to = self.positive.advance_shallow(target)?;
        if self.negative.doc().is_none_or(|doc| doc < target) {
            self.negative.advance(target)?;
        }
        if let Some(doc) = self.negative.doc() {
            up_to = up_to.min(self.negative.advance_shallow(target.max(doc))?);
        }
        Ok(up_to)
    }

    fn score_bounds(&mut self, up_to: u64) -> Result<ScoreBounds> {
        let positive = self.positive.score_bounds(up_to)?;
        let negative = if self.negative.doc().is_some_and(|doc| doc <= up_to) {
            self.negative.score_bounds(up_to)?.include_zero()
        } else {
            ScoreBounds::ZERO
        };
        Ok(positive.subtract_scaled(negative, self.negative_boost))
    }

    fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()> {
        // With a non-negative negative scorer, Boost can only demote the
        // positive score, so the parent's floor is safe for the positive side.
        if self.negative.scores_non_negative() {
            self.positive.set_min_competitive_score(min_score)?;
        }
        Ok(())
    }

    fn current_score_upper_bound(&mut self) -> Result<Option<f32>> {
        if self.negative.scores_non_negative() {
            self.positive.current_score_upper_bound()
        } else {
            Ok(None)
        }
    }

    fn supports_doc_local_confirmation_pruning(&self) -> bool {
        self.positive.supports_doc_local_confirmation_pruning()
    }

    fn matches(&mut self) -> Result<bool> {
        self.positive.matches()
    }

    fn match_cost(&self) -> Option<f32> {
        self.positive.match_cost()
    }
}

/// Required-plus-optional scorer that only touches the optional side when it
/// can change competitiveness or an exact score is requested.
///
/// The required scorer drives iteration. Once a score floor is available,
/// block bounds may either skip the whole range or temporarily turn the
/// optional approximation into a required iterator when the required score
/// cannot reach the floor on its own.
#[derive(Clone, Copy)]
struct ReqOptBounds {
    up_to: u64,
    required: ScoreBounds,
    combined: ScoreBounds,
}

struct ReqOptScorer<'a> {
    required: BoxScorer<'a>,
    optional: BoxScorer<'a>,
    current: Option<u64>,
    exhausted: bool,
    optional_initialized: bool,
    optional_is_required: bool,
    optional_checked_doc: Option<u64>,
    optional_matches: bool,
    confirmed_doc: Option<u64>,
    confirmed: bool,
    min_competitive_score: f32,
    shallow_bounds: Option<ReqOptBounds>,
    /// Set once the heap floor passes the required clause's list-wide upper
    /// bound: from then on no document can reach the heap on the required
    /// clause alone, so the optional clause is mandatory for the rest of the
    /// scan and the query is an intersection.
    optional_globally_required: bool,
}

impl<'a> ReqOptScorer<'a> {
    fn new(required: BoxScorer<'a>, optional: BoxScorer<'a>) -> Self {
        debug_assert!(required.scores_non_negative());
        debug_assert!(optional.scores_non_negative());
        Self {
            required,
            optional,
            current: None,
            exhausted: false,
            optional_initialized: false,
            optional_is_required: false,
            optional_checked_doc: None,
            optional_matches: false,
            confirmed_doc: None,
            confirmed: false,
            min_competitive_score: f32::NEG_INFINITY,
            shallow_bounds: None,
            optional_globally_required: false,
        }
    }

    fn set_current(&mut self, current: Option<u64>) {
        if self.current != current {
            self.optional_checked_doc = None;
            self.optional_matches = false;
            self.confirmed_doc = None;
            self.confirmed = false;
        }
        self.current = current;
        self.optional_is_required = false;
    }

    fn set_optional_required(&mut self, required: bool) {
        if self.optional_is_required != required {
            self.confirmed_doc = None;
            self.confirmed = false;
        }
        self.optional_is_required = required;
    }

    /// Promote the optional clause to required once the required clause's
    /// list-wide upper bound can no longer reach the heap. The floor only ever
    /// rises, so this decision is permanent. Past this point there is nothing
    /// for a required-only walk to contribute, and the per-window bookkeeping
    /// is pure overhead.
    fn promote_optional_if_required(&mut self) -> Result<()> {
        if self.optional_globally_required || !self.min_competitive_score.is_finite() {
            return Ok(());
        }
        let Some(required_upper) = self.required.global_score_upper_bound() else {
            return Ok(());
        };
        if !required_upper.is_finite() || required_upper >= self.min_competitive_score {
            return Ok(());
        }
        self.optional_globally_required = true;
        self.shallow_bounds = None;
        Ok(())
    }

    /// Intersect both sides directly. Used once the optional clause is known to
    /// be mandatory: the required clause alone can no longer produce a
    /// competitive document, so the window bookkeeping earns nothing. The
    /// cheaper list drives, so a huge MUST paired with a sparse SHOULD walks the
    /// sparse list instead of the MUST tail.
    fn position_intersection(&mut self, target: u64) -> Result<Option<u64>> {
        self.shallow_bounds = None;
        if self.optional.cost() < self.required.cost() {
            self.intersect_from_optional(target)
        } else {
            self.intersect_from_required(target)
        }
    }

    /// Leapfrog driven by the required (MUST) list.
    fn intersect_from_required(&mut self, target: u64) -> Result<Option<u64>> {
        let Some(mut required_doc) = self.required.advance(target)? else {
            return Ok(self.exhaust());
        };
        loop {
            self.set_current(Some(required_doc));
            self.set_optional_required(true);
            let Some(optional_doc) = self.ensure_optional_at_or_after(required_doc)? else {
                return Ok(self.exhaust());
            };
            if optional_doc == required_doc {
                return Ok(self.current);
            }
            let Some(next_required) = self.required.advance(optional_doc)? else {
                return Ok(self.exhaust());
            };
            required_doc = next_required;
        }
    }

    /// Leapfrog driven by the optional (SHOULD) list, which is the cheaper list
    /// once the optional is mandatory.
    fn intersect_from_optional(&mut self, target: u64) -> Result<Option<u64>> {
        self.optional_initialized = true;
        let Some(mut optional_doc) = self.optional.advance(target)? else {
            return Ok(self.exhaust());
        };
        loop {
            let Some(required_doc) = self.required.advance(optional_doc)? else {
                return Ok(self.exhaust());
            };
            if required_doc == optional_doc {
                self.set_current(Some(optional_doc));
                self.set_optional_required(true);
                return Ok(self.current);
            }
            let Some(next_optional) = self.optional.advance(required_doc)? else {
                return Ok(self.exhaust());
            };
            optional_doc = next_optional;
        }
    }

    fn exhaust(&mut self) -> Option<u64> {
        self.exhausted = true;
        self.set_current(None);
        None
    }

    fn ensure_optional_at_or_after(&mut self, target: u64) -> Result<Option<u64>> {
        if !self.optional_initialized || self.optional.doc().is_some_and(|doc| doc < target) {
            self.optional.advance(target)?;
            self.optional_initialized = true;
        }
        Ok(self.optional.doc())
    }

    fn optional_matches_current(&mut self) -> Result<bool> {
        let Some(current) = self.current else {
            return Ok(false);
        };
        if self.optional_checked_doc == Some(current) {
            return Ok(self.optional_matches);
        }
        self.optional_matches = self.ensure_optional_at_or_after(current)? == Some(current)
            && self.optional.matches()?;
        self.optional_checked_doc = Some(current);
        Ok(self.optional_matches)
    }

    fn usable_bounds(bounds: ScoreBounds) -> bool {
        bounds.lower.is_finite() && bounds.upper.is_finite() && bounds.lower <= bounds.upper
    }

    fn bounds(&mut self, up_to: u64) -> Result<ReqOptBounds> {
        if let Some(bounds) = self.shallow_bounds
            && bounds.up_to == up_to
        {
            return Ok(bounds);
        }

        let required = self.required.score_bounds(up_to)?;
        let optional = if self.optional.doc().is_some_and(|doc| doc <= up_to) {
            let bounds = self.optional.score_bounds(up_to)?;
            if Self::usable_bounds(bounds) {
                bounds.include_zero()
            } else {
                ScoreBounds::UNBOUNDED
            }
        } else {
            ScoreBounds::ZERO
        };
        let combined = required.add(optional);
        let bounds = ReqOptBounds {
            up_to,
            required,
            combined,
        };
        self.shallow_bounds = Some(bounds);
        Ok(bounds)
    }

    fn position(&mut self, mut target: u64) -> Result<Option<u64>> {
        if self.exhausted {
            return Ok(None);
        }
        if self.optional_globally_required {
            return self.position_intersection(target);
        }

        'search: loop {
            let bounds = self.shallow_bounds.filter(|bounds| target <= bounds.up_to);

            if self.min_competitive_score.is_finite()
                && self.min_competitive_score > 0.0
                && let Some(bounds) = bounds
                && Self::usable_bounds(bounds.required)
                && Self::usable_bounds(bounds.combined)
            {
                if bounds.combined.upper < self.min_competitive_score {
                    if bounds.up_to == u64::MAX {
                        return Ok(self.exhaust());
                    }
                    target = bounds.up_to + 1;
                    self.shallow_bounds = None;
                    continue;
                }

                if bounds.required.upper < self.min_competitive_score {
                    // The optional contribution is necessary throughout this
                    // cached shallow range. Intersect approximations until
                    // both sides agree or the range is exhausted.
                    let Some(mut required_doc) = self.required.advance(target)? else {
                        return Ok(self.exhaust());
                    };
                    if required_doc > bounds.up_to {
                        target = required_doc;
                        self.shallow_bounds = None;
                        continue;
                    }
                    self.set_current(Some(required_doc));
                    self.set_optional_required(true);
                    loop {
                        self.set_current(Some(required_doc));
                        self.set_optional_required(true);
                        let Some(optional_doc) = self.ensure_optional_at_or_after(required_doc)?
                        else {
                            if bounds.up_to == u64::MAX {
                                return Ok(self.exhaust());
                            }
                            target = bounds.up_to + 1;
                            self.shallow_bounds = None;
                            continue 'search;
                        };
                        if optional_doc > bounds.up_to {
                            if bounds.up_to == u64::MAX {
                                return Ok(self.exhaust());
                            }
                            target = bounds.up_to + 1;
                            self.shallow_bounds = None;
                            continue 'search;
                        }
                        if optional_doc == required_doc {
                            return Ok(self.current);
                        }
                        let Some(next_required) = self.required.advance(optional_doc)? else {
                            return Ok(self.exhaust());
                        };
                        if next_required > bounds.up_to {
                            target = next_required;
                            self.shallow_bounds = None;
                            continue 'search;
                        }
                        required_doc = next_required;
                    }
                }
            }

            let Some(required_doc) = self.required.advance(target)? else {
                return Ok(self.exhaust());
            };
            self.set_current(Some(required_doc));
            self.set_optional_required(false);
            return Ok(self.current);
        }
    }

    fn ensure_confirmed(&mut self) -> Result<bool> {
        let Some(current) = self.current else {
            return Ok(false);
        };
        if self.confirmed_doc == Some(current) {
            return Ok(self.confirmed);
        }
        self.confirmed = self.required.matches()?
            && (!self.optional_is_required || self.optional_matches_current()?);
        self.confirmed_doc = Some(current);
        Ok(self.confirmed)
    }
}

impl ComposableScorer for ReqOptScorer<'_> {
    fn doc(&self) -> Option<u64> {
        self.current
    }

    fn document_key(&self) -> Option<u64> {
        self.required.document_key()
    }

    fn next(&mut self) -> Result<Option<u64>> {
        let target = match self.current {
            None => 0,
            Some(u64::MAX) => return Ok(self.exhaust()),
            Some(current) => current + 1,
        };
        self.position(target)
    }

    fn advance(&mut self, target: u64) -> Result<Option<u64>> {
        if self.current.is_some_and(|current| current >= target) {
            return Ok(self.current);
        }
        self.position(target)
    }

    fn cost(&self) -> usize {
        self.required.cost()
    }

    fn score(&mut self) -> Result<f32> {
        if !self.ensure_confirmed()? {
            return Err(Error::internal(
                "required-plus-optional FTS score requested for an unconfirmed document",
            ));
        }
        let mut score = self.required.score()?;
        if self.optional_matches_current()? {
            score += self.optional.score()?;
        }
        checked_score(score, "required-plus-optional FTS scorer")
    }

    fn advance_shallow(&mut self, target: u64) -> Result<u64> {
        let target = self.current.map_or(target, |current| target.max(current));
        if let Some(bounds) = self.shallow_bounds
            && target <= bounds.up_to
        {
            return Ok(bounds.up_to);
        }
        self.shallow_bounds = None;
        let mut up_to = self.required.advance_shallow(target)?;
        match self.ensure_optional_at_or_after(target)? {
            Some(optional_doc) if optional_doc <= target => {
                up_to = up_to.min(self.optional.advance_shallow(target)?);
            }
            Some(optional_doc) => {
                up_to = up_to.min(optional_doc.saturating_sub(1));
            }
            None => {}
        }
        Ok(up_to)
    }

    fn score_bounds(&mut self, up_to: u64) -> Result<ScoreBounds> {
        Ok(self.bounds(up_to)?.combined)
    }

    fn global_score_upper_bound(&self) -> Option<f32> {
        let required = self.required.global_score_upper_bound()?;
        let optional = self.optional.global_score_upper_bound()?;
        let combined = ScoreBounds {
            lower: 0.0,
            upper: required,
        }
        .add(ScoreBounds {
            lower: 0.0,
            upper: optional,
        })
        .upper;
        combined.is_finite().then_some(combined)
    }

    fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()> {
        if min_score.is_nan() {
            return Err(Error::invalid_input(
                "minimum competitive FTS score cannot be NaN",
            ));
        }
        if min_score > self.min_competitive_score {
            self.min_competitive_score = min_score;
            // Once the required clause alone can no longer reach the heap the
            // optional clause is mandatory, so switch to a direct intersection.
            self.promote_optional_if_required()?;
        }
        Ok(())
    }

    fn current_score_upper_bound(&mut self) -> Result<Option<f32>> {
        let Some(current) = self.current else {
            return Ok(None);
        };
        let Some(required_upper) = self.required.current_score_upper_bound()? else {
            return Ok(None);
        };
        if !required_upper.is_finite() {
            return Ok(None);
        }
        if required_upper >= self.min_competitive_score {
            // The required side alone keeps this document competitive. Keep
            // the optional iterator lazy; scoring will align it only for a
            // surviving candidate.
            return Ok(Some(f32::INFINITY));
        }
        let optional_upper = if self.ensure_optional_at_or_after(current)? == Some(current) {
            let Some(optional_upper) = self.optional.current_score_upper_bound()? else {
                return Ok(None);
            };
            if !optional_upper.is_finite() {
                return Ok(None);
            }
            optional_upper.max(0.0)
        } else {
            0.0
        };
        let upper = ScoreBounds {
            lower: 0.0,
            upper: required_upper.max(0.0),
        }
        .add(ScoreBounds {
            lower: 0.0,
            upper: optional_upper,
        })
        .upper;
        Ok(upper.is_finite().then_some(upper))
    }

    fn supports_doc_local_confirmation_pruning(&self) -> bool {
        self.required.supports_doc_local_confirmation_pruning()
            || self.optional.supports_doc_local_confirmation_pruning()
    }

    fn matches(&mut self) -> Result<bool> {
        self.ensure_confirmed()
    }

    fn match_cost(&self) -> Option<f32> {
        self.required
            .match_cost()
            .into_iter()
            .chain(self.optional.match_cost())
            .reduce(|left, right| left + right)
    }

    fn scores_non_negative(&self) -> bool {
        true
    }

    #[cfg(test)]
    fn debug_child_type_names(&self) -> Vec<&'static str> {
        vec![
            self.required.debug_type_name(),
            self.optional.debug_type_name(),
        ]
    }
}
pub(super) struct BooleanScorer<'a> {
    driver: BoxScorer<'a>,
    optional: Option<BoxScorer<'a>>,
    prohibited: Option<BoxScorer<'a>>,
    current: Option<u64>,
    confirmed_doc: Option<u64>,
    confirmed: bool,
    optional_matches: bool,
    defer_confirmation: bool,
}

impl<'a> BooleanScorer<'a> {
    pub(super) fn try_new(
        should: Vec<BoxScorer<'a>>,
        must: Vec<BoxScorer<'a>>,
        must_not: Vec<BoxScorer<'a>>,
    ) -> Result<Self> {
        let (driver, optional) = if must.is_empty() {
            if should.is_empty() {
                return Err(Error::invalid_input(
                    "boolean query must have at least one should/must query",
                ));
            }
            let driver = if let Some(global_bounds) = ShouldMaxScoreScorer::global_bounds(&should) {
                Box::new(ShouldMaxScoreScorer::new(should, global_bounds)) as BoxScorer<'a>
            } else {
                Box::new(DisjunctionScorer::try_new(should, DisjunctionScore::Sum)?)
                    as BoxScorer<'a>
            };
            (driver, None)
        } else {
            let mut optional = if should.is_empty() {
                None
            } else {
                Some(
                    Box::new(DisjunctionScorer::try_new(should, DisjunctionScore::Sum)?)
                        as BoxScorer<'a>,
                )
            };
            let required = if must.len() == 1 {
                must.into_iter().next().expect("must has one child")
            } else {
                Box::new(RequiredConjunctionScorer::try_new(must)?) as BoxScorer<'a>
            };
            let driver = if required.scores_non_negative()
                && optional
                    .as_ref()
                    .is_some_and(|optional| optional.scores_non_negative())
            {
                Box::new(ReqOptScorer::new(
                    required,
                    optional
                        .take()
                        .expect("checked that the optional scorer is present"),
                )) as BoxScorer<'a>
            } else {
                required
            };
            (driver, optional)
        };
        let prohibited = if must_not.is_empty() {
            None
        } else {
            Some(
                Box::new(DisjunctionScorer::try_new(must_not, DisjunctionScore::Max)?)
                    as BoxScorer<'a>,
            )
        };
        let scores_non_negative = driver.scores_non_negative()
            && optional
                .as_ref()
                .is_none_or(|optional| optional.scores_non_negative());
        let has_doc_local_confirmation = driver.supports_doc_local_confirmation_pruning()
            || optional
                .as_ref()
                .is_some_and(|optional| optional.supports_doc_local_confirmation_pruning())
            || prohibited
                .as_ref()
                .is_some_and(|prohibited| prohibited.supports_doc_local_confirmation_pruning());
        Ok(Self {
            driver,
            optional,
            prohibited,
            current: None,
            confirmed_doc: None,
            confirmed: false,
            optional_matches: false,
            defer_confirmation: scores_non_negative && has_doc_local_confirmation,
        })
    }

    pub(super) fn try_root(
        should: Vec<BoxScorer<'a>>,
        must: Vec<BoxScorer<'a>>,
        must_not: Vec<BoxScorer<'a>>,
    ) -> Result<BoxScorer<'a>> {
        let boolean = Self::try_new(should, must, must_not)?;
        if boolean.optional.is_none() && boolean.prohibited.is_none() {
            Ok(boolean.driver)
        } else {
            Ok(Box::new(boolean))
        }
    }

    fn set_current(&mut self, current: Option<u64>) -> Option<u64> {
        if self.current != current {
            self.confirmed_doc = None;
            self.confirmed = false;
            self.optional_matches = false;
        }
        self.current = current;
        current
    }

    fn ensure_confirmed(&mut self) -> Result<bool> {
        let Some(current) = self.current else {
            return Ok(false);
        };
        if self.confirmed_doc == Some(current) {
            return Ok(self.confirmed);
        }
        if !self.driver.matches()? {
            self.confirmed_doc = Some(current);
            self.confirmed = false;
            return Ok(false);
        }
        if let Some(prohibited) = &mut self.prohibited
            && prohibited.advance(current)? == Some(current)
            && prohibited.matches()?
        {
            self.confirmed_doc = Some(current);
            self.confirmed = false;
            return Ok(false);
        }
        self.optional_matches = if let Some(optional) = &mut self.optional {
            optional.advance(current)? == Some(current) && optional.matches()?
        } else {
            false
        };
        self.confirmed_doc = Some(current);
        self.confirmed = true;
        Ok(true)
    }

    fn next_candidate(&mut self, target: Option<u64>) -> Result<Option<u64>> {
        let mut candidate = match target {
            Some(target) => self.driver.advance(target)?,
            None => self.driver.next()?,
        };
        loop {
            self.set_current(candidate);
            if self.defer_confirmation || candidate.is_none() || self.ensure_confirmed()? {
                return Ok(self.current);
            }
            candidate = self.driver.next()?;
        }
    }
}

impl ComposableScorer for BooleanScorer<'_> {
    fn doc(&self) -> Option<u64> {
        self.current
    }

    fn document_key(&self) -> Option<u64> {
        self.driver.document_key()
    }

    fn next(&mut self) -> Result<Option<u64>> {
        self.next_candidate(None)
    }

    fn advance(&mut self, target: u64) -> Result<Option<u64>> {
        if self.current.is_some_and(|current| current >= target) {
            return Ok(self.current);
        }
        self.next_candidate(Some(target))
    }

    fn cost(&self) -> usize {
        self.driver.cost()
    }

    fn score(&mut self) -> Result<f32> {
        if !self.ensure_confirmed()? {
            return Err(Error::internal(
                "Boolean FTS score requested for an unconfirmed document",
            ));
        }
        let mut score = self.driver.score()?;
        if self.optional_matches
            && let Some(optional) = &mut self.optional
        {
            score += optional.score()?;
        }
        checked_score(score, "BooleanQuery scorer")
    }

    fn advance_shallow(&mut self, target: u64) -> Result<u64> {
        let mut up_to = self.driver.advance_shallow(target)?;
        if let Some(optional) = &mut self.optional
            && let Some(doc) = optional.doc()
        {
            up_to = up_to.min(optional.advance_shallow(target.max(doc))?);
        }
        Ok(up_to)
    }

    fn score_bounds(&mut self, up_to: u64) -> Result<ScoreBounds> {
        let mut bounds = self.driver.score_bounds(up_to)?;
        if let Some(optional) = &mut self.optional
            && optional.doc().is_some_and(|doc| doc <= up_to)
        {
            bounds = bounds.add(optional.score_bounds(up_to)?.include_zero());
        }
        Ok(bounds)
    }

    fn global_score_upper_bound(&self) -> Option<f32> {
        if !self.scores_non_negative() {
            return None;
        }
        let driver = self.driver.global_score_upper_bound()?;
        let combined = if let Some(optional) = &self.optional {
            let optional = optional.global_score_upper_bound()?;
            ScoreBounds {
                lower: 0.0,
                upper: driver,
            }
            .add(ScoreBounds {
                lower: 0.0,
                upper: optional,
            })
            .upper
        } else {
            driver
        };
        (combined.is_finite() && combined >= 0.0).then_some(combined)
    }

    fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()> {
        // When SHOULD is also present, a global sibling bound is required to
        // translate the parent threshold safely. The combined block bound still
        // prunes at this node. Without SHOULD, driver score is the full score.
        if self.optional.is_none() {
            self.driver.set_min_competitive_score(min_score)?;
        }
        Ok(())
    }

    fn current_score_upper_bound(&mut self) -> Result<Option<f32>> {
        let Some(current) = self.current else {
            return Ok(None);
        };
        if !self.scores_non_negative() {
            return Ok(None);
        }
        let Some(driver_upper) = self.driver.current_score_upper_bound()? else {
            return Ok(None);
        };
        if !driver_upper.is_finite() {
            return Ok(None);
        }
        let optional_upper = if let Some(optional) = &mut self.optional {
            if optional.advance(current)? == Some(current) {
                let Some(optional_upper) = optional.current_score_upper_bound()? else {
                    return Ok(None);
                };
                if !optional_upper.is_finite() {
                    return Ok(None);
                }
                optional_upper.max(0.0)
            } else {
                0.0
            }
        } else {
            0.0
        };
        let upper = ScoreBounds {
            lower: 0.0,
            upper: driver_upper,
        }
        .add(ScoreBounds {
            lower: 0.0,
            upper: optional_upper,
        })
        .upper;
        Ok(upper.is_finite().then_some(upper))
    }

    fn supports_doc_local_confirmation_pruning(&self) -> bool {
        self.defer_confirmation
    }

    fn matches(&mut self) -> Result<bool> {
        self.ensure_confirmed()
    }

    fn match_cost(&self) -> Option<f32> {
        self.driver
            .match_cost()
            .into_iter()
            .chain(
                self.optional
                    .as_ref()
                    .and_then(|optional| optional.match_cost()),
            )
            .chain(
                self.prohibited
                    .as_ref()
                    .and_then(|prohibited| prohibited.match_cost()),
            )
            .reduce(|left, right| left + right)
    }

    fn scores_non_negative(&self) -> bool {
        self.driver.scores_non_negative()
            && self
                .optional
                .as_ref()
                .is_none_or(|optional| optional.scores_non_negative())
    }

    fn collect_confirmed_window(
        &mut self,
        min_score: f32,
        out: &mut Vec<ConfirmedHit>,
        max_hits: usize,
    ) -> Result<WindowCollect> {
        if self.optional.is_none() && self.prohibited.is_none() {
            let status = self
                .driver
                .collect_confirmed_window(min_score, out, max_hits)?;
            self.current = self.driver.doc();
            self.confirmed_doc = None;
            self.confirmed = false;
            return Ok(status);
        }
        collect_confirmed_window_default(self, min_score, out, max_hits)
    }

    #[cfg(test)]
    fn debug_child_type_names(&self) -> Vec<&'static str> {
        let mut names = vec![self.driver.debug_type_name()];
        if let Some(optional) = &self.optional {
            names.push(optional.debug_type_name());
        }
        if let Some(prohibited) = &self.prohibited {
            names.push(prohibited.debug_type_name());
        }
        names
    }
}

#[derive(Clone)]
pub(super) enum LeafQuery {
    Match(MatchQuery),
    Phrase(PhraseQuery),
}

impl LeafQuery {
    pub(super) fn terms(&self) -> &str {
        match self {
            Self::Match(query) => &query.terms,
            Self::Phrase(query) => &query.terms,
        }
    }

    pub(super) fn operator(&self) -> Operator {
        match self {
            Self::Match(query) => query.operator,
            Self::Phrase(_) => Operator::And,
        }
    }

    pub(super) fn effective_params(&self, params: &FtsSearchParams) -> FtsSearchParams {
        match self {
            Self::Match(query) => params
                .clone()
                .with_limit(None)
                .with_phrase_slop(None)
                .with_fuzziness(query.fuzziness)
                .with_max_expansions(query.max_expansions)
                .with_prefix_length(query.prefix_length),
            Self::Phrase(query) => params
                .clone()
                .with_limit(None)
                .with_phrase_slop(Some(query.slop)),
        }
    }
}

pub(super) fn collect_leaf_queries(query: &FtsQuery, leaves: &mut Vec<LeafQuery>) -> Result<()> {
    match query {
        FtsQuery::Match(query) => leaves.push(LeafQuery::Match(query.clone())),
        FtsQuery::Phrase(query) => leaves.push(LeafQuery::Phrase(query.clone())),
        FtsQuery::Boost(query) => {
            collect_leaf_queries(&query.positive, leaves)?;
            collect_leaf_queries(&query.negative, leaves)?;
        }
        FtsQuery::MultiMatch(query) => {
            leaves.extend(query.match_queries.iter().cloned().map(LeafQuery::Match));
        }
        FtsQuery::Boolean(query) => {
            for child in query
                .should
                .iter()
                .chain(&query.must)
                .chain(&query.must_not)
            {
                collect_leaf_queries(child, leaves)?;
            }
        }
        FtsQuery::CombinedFields(_) => return Err(combined_fields_unsupported()),
    }
    Ok(())
}

/// A `combined_fields` node reaching the compound scorer is a planner bug.
///
/// Every leaf here draws its postings from one column's index and scores them
/// with a `MemBM25Scorer`, whereas BM25F blends `tf'`/`dl'`/`docFreq'` across
/// several columns before scoring. The planner keeps such trees out via
/// `supports_compound_scorer`, so this is an explicit error.
fn combined_fields_unsupported() -> Error {
    Error::not_supported(
        "the compound FTS scorer cannot score a combined_fields (BM25F) node: its statistics \
         are blended across columns and do not fit the single-index leaf protocol",
    )
}

struct PreparedLeaf {
    query: Arc<PreparedBm25Query>,
    params: Arc<FtsSearchParams>,
    operator: Operator,
}

pub(super) fn tokenize_leaf(
    index: &InvertedIndex,
    leaf: &LeafQuery,
    params: &FtsSearchParams,
) -> Result<Tokens> {
    // Keep the legacy explicit-fuzzy rewrite independent of index analysis.
    // AUTO fuzziness still expands later, but its source terms must first use
    // the same normalization and filtering as the indexed vocabulary.
    let is_explicit_fuzzy_match = matches!(leaf, LeafQuery::Match(_))
        && matches!(params.fuzziness, Some(distance) if distance > 0);
    let mut tokenizer = if is_explicit_fuzzy_match {
        let analyzer = TextAnalyzer::from(SimpleTokenizer::default());
        match index.tokenizer().doc_type() {
            DocType::Text => Box::new(TextTokenizer::new(analyzer)) as Box<dyn LanceTokenizer>,
            DocType::Json => Box::new(JsonTokenizer::new(analyzer)) as Box<dyn LanceTokenizer>,
        }
    } else {
        index.tokenizer()
    };
    try_collect_query_tokens(leaf.terms(), &mut tokenizer)
}

async fn prepare_compound_query(
    indices: &[Arc<InvertedIndex>],
    query: &FtsQuery,
    params: &FtsSearchParams,
    metrics: &dyn MetricsCollector,
    base_scorer: Option<Arc<MemBM25Scorer>>,
    prepared_match: Option<Arc<PreparedBm25Query>>,
) -> Result<(CompoundScorerPlan, Vec<PreparedLeaf>)> {
    let first_index = indices
        .first()
        .ok_or_else(|| Error::invalid_input("compound FTS requires at least one index segment"))?;
    let mut leaf_queries = Vec::new();
    collect_leaf_queries(query, &mut leaf_queries)?;
    let mut num_plan_leaves = 0;
    let plan = CompoundScorerPlan::from_query(query, &mut num_plan_leaves)?;
    if num_plan_leaves != leaf_queries.len() {
        return Err(Error::internal(format!(
            "compound FTS planned {num_plan_leaves} leaves but prepared {}",
            leaf_queries.len()
        )));
    }

    let mut leaves = Vec::with_capacity(leaf_queries.len());
    if prepared_match.is_some() && leaf_queries.len() != 1 {
        return Err(Error::internal(
            "prepared Match replay requires exactly one compound FTS leaf",
        ));
    }
    for leaf in leaf_queries {
        let effective_params = leaf.effective_params(params);
        let tokens = tokenize_leaf(first_index, &leaf, &effective_params)?;
        let prepared = match &prepared_match {
            Some(prepared) => prepared.clone(),
            None => Arc::new(
                prepare_bm25_query(
                    indices,
                    tokens,
                    &effective_params,
                    Some(metrics),
                    base_scorer.clone(),
                )
                .await?,
            ),
        };
        leaves.push(PreparedLeaf {
            query: prepared,
            params: Arc::new(effective_params),
            operator: leaf.operator(),
        });
    }
    Ok((plan, leaves))
}

struct LoadedLeaf {
    postings: Vec<PostingIterator>,
    params: Arc<FtsSearchParams>,
    operator: Operator,
    scorer: Arc<MemBM25Scorer>,
}

enum LoadedDocuments {
    Legacy(Arc<DocSet>),
    Modern {
        documents: Arc<PartitionDocuments>,
        lengths: Arc<DocLengths>,
        visibility: DocVisibility,
        projection: Option<ResidentAddressProjection>,
    },
}

struct LoadedPartition {
    segment_ordinal: usize,
    partition_ordinal: usize,
    partition: Arc<InvertedPartition>,
    documents: LoadedDocuments,
    leaves: Vec<LoadedLeaf>,
}

async fn load_compound_partition(
    segment_ordinal: usize,
    partition_ordinal: usize,
    partition: Arc<InvertedPartition>,
    leaves: &[PreparedLeaf],
    mask: Arc<RowAddrMask>,
    metrics: Arc<dyn MetricsCollector>,
) -> Result<Option<LoadedPartition>> {
    let leaf_loads = leaves.iter().map(|leaf| {
        let partition = partition.clone();
        let tokens = leaf.query.tokens().clone();
        let params = leaf.params.clone();
        let scorer = leaf.query.scorer().clone();
        let metrics = metrics.clone();
        let operator = leaf.operator;
        let has_all_query_positions = leaf.query.has_all_query_positions();
        async move {
            let postings = if tokens.is_empty()
                || ((operator == Operator::And || params.phrase_slop.is_some())
                    && !has_all_query_positions)
            {
                Vec::new()
            } else {
                partition
                    .load_posting_lists(
                        tokens.as_ref(),
                        params.as_ref(),
                        operator,
                        scorer.as_ref(),
                        metrics.as_ref(),
                        true,
                    )
                    .await?
                    .postings
            };
            Result::Ok(LoadedLeaf {
                postings,
                params,
                operator,
                scorer,
            })
        }
    });
    let leaves = futures::future::try_join_all(leaf_loads).await?;

    let documents = if let Some(docs) = partition.docs.legacy() {
        LoadedDocuments::Legacy(docs.clone())
    } else {
        let documents = partition.docs.modern().cloned().ok_or_else(|| {
            Error::internal("FTS partition contains neither legacy nor modern documents")
        })?;
        let materialize_selected = mask.max_len().is_some_and(|selected| {
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
        if visibility.is_empty() {
            return Ok(None);
        }
        let lengths = match documents.cached_lengths() {
            Some(lengths) => lengths,
            None => documents.lengths().await?,
        };
        let projection = documents.resident_address_projection();
        LoadedDocuments::Modern {
            documents,
            lengths,
            visibility,
            projection,
        }
    };

    Ok(Some(LoadedPartition {
        segment_ordinal,
        partition_ordinal,
        partition,
        documents,
        leaves,
    }))
}

struct DeferredCompoundRows {
    documents: Arc<PartitionDocuments>,
    rows: Vec<ScoredRow<DocId>>,
}

struct OverflowedCompoundPartition {
    segment_ordinal: usize,
    partition_ordinal: usize,
    partition: Arc<InvertedPartition>,
    documents: Arc<PartitionDocuments>,
}

enum PartitionCollectionBoundary {
    Deferred(DeferredCompoundRows),
    Overflow(OverflowedCompoundPartition),
}

struct CollectedPartitions {
    collector: TopKCollector<u64>,
    remaining: Vec<LoadedPartition>,
    boundary: Option<PartitionCollectionBoundary>,
}

fn is_compound_term_leaf(leaf: &LoadedLeaf) -> bool {
    leaf.postings.len() == 1
        && leaf.params.phrase_slop.is_none()
        && !leaf.postings[0].has_grouped_terms()
}

/// The leaf children of a flat Boolean plan, split by strength. Children that
/// are not single leaves are skipped; `must_children` counts the raw MUST
/// children and `all_leaves` reports whether every child (including MUST_NOT)
/// was a leaf.
struct BooleanLeafSplit {
    must_leaves: Vec<(usize, f32)>,
    should_leaves: Vec<(usize, f32)>,
    must_children: usize,
    has_must_not: bool,
    all_leaves: bool,
}

fn boolean_leaf_split(plan: &CompoundScorerPlan) -> Option<BooleanLeafSplit> {
    let CompoundScorerPlan::Boolean {
        should,
        must,
        must_not,
    } = plan
    else {
        return None;
    };
    fn leaf_children(children: &[CompoundScorerPlan]) -> Vec<(usize, f32)> {
        children
            .iter()
            .filter_map(|child| match child {
                CompoundScorerPlan::Leaf { index, boost } => Some((*index, *boost)),
                _ => None,
            })
            .collect()
    }
    let all_leaves = should
        .iter()
        .chain(must_not.iter())
        .chain(must.iter())
        .all(|child| matches!(child, CompoundScorerPlan::Leaf { .. }));
    Some(BooleanLeafSplit {
        must_leaves: leaf_children(must),
        should_leaves: leaf_children(should),
        must_children: must.len(),
        has_must_not: !must_not.is_empty(),
        all_leaves,
    })
}

/// Whether a leaf may step its single posting directly instead of paying the
/// WAND wrapper. TermLeafScorer consults no competitive floor —
/// `set_min_competitive_score` is a no-op and it never skips — so the swap
/// also gives up the wrapper's block-max skipping. Where that skipping earns
/// nothing the wrapper is pure overhead; where it earns, the leaf keeps its
/// cursor.
///
/// The ReqOpt scorer holds its floor itself and never forwards it to either
/// side (`ReqOptScorer::set_min_competitive_score`); it prunes by consulting
/// each side's `advance_shallow`/`score_bounds`, and it probes the optional side
/// by exact `advance(required_doc)`. So on that path neither leaf is steered by
/// a floor and the wrapper's skipping earns nothing — swapping the leaf in wins
/// on the required/optional Boolean shapes:
///
/// - the single MUST child of a 1-MUST Boolean: its candidates are admitted
///   one by one and the wrapper spent its time on window bookkeeping;
/// - the SHOULD children of the same 1-MUST Boolean: the wrapper has no union
///   machine to steer and only adds seek and scoring bookkeeping;
/// - the leaves of a multi-MUST conjunction are driven by their parent's own
///   skipping, and the optional children of a SHOULD-only union sit inside a
///   union machine — both measured slower with the swap.
///
/// The remaining arms (MUST_NOT, Boost-positive, MultiMatch, nested shapes) are
/// left on the WAND wrapper: TermLeafScorer never skips, and those arms are
/// correctness-safe either way but have no measurement behind the swap. Note
/// that a MUST_NOT Boolean still drives through `ReqOptScorer`; excluding it is
/// deliberate conservatism, not a structural difference.
fn leaf_admits_term_leaf(plan: &CompoundScorerPlan, index: usize) -> bool {
    match plan {
        CompoundScorerPlan::Boolean { .. } => {
            let Some(split) = boolean_leaf_split(plan) else {
                return false;
            };
            if split.must_children != 1 {
                return false;
            }
            // The single MUST leaf keeps its existing admission.
            let Some(&(must_index, _)) = split.must_leaves.first() else {
                return false;
            };
            if must_index == index {
                return true;
            }
            // In a 1-MUST ReqOpt, the SHOULD leaves are probed by
            // `advance(doc)` once per required document, and the ReqOpt scorer
            // holds the floor itself (it forwards it to neither side). A direct
            // posting seek beats the WAND machine there, so admit them too.
            // MUST_NOT is deliberately excluded — the ReqOpt scorer still drives
            // in that case, but admitting it is unmeasured — and a multi-MUST
            // conjunction or a SHOULD-only union stays on the WAND cursor, which
            // the parent's or the union's block skipping steers.
            !split.has_must_not
                && split
                    .should_leaves
                    .iter()
                    .any(|&(should_index, _)| should_index == index)
        }
        // A standalone single-leaf plan (one term query): TermLeafScorer is
        // the whole scorer, so there is no wrapper bookkeeping to save.
        CompoundScorerPlan::Leaf {
            index: leaf_index,
            boost: _,
        } => *leaf_index == index,
        _ => false,
    }
}

fn box_leaf_scorer<'a, D>(
    leaf: LoadedLeaf,
    documents: &'a D,
    metrics: &'a dyn MetricsCollector,
    admits_term_leaf: bool,
) -> BoxScorer<'a>
where
    D: WandDocuments + Sync,
{
    if leaf.postings.is_empty() {
        return Box::new(EmptyScorer);
    }
    if admits_term_leaf && is_compound_term_leaf(&leaf) {
        let posting = leaf
            .postings
            .into_iter()
            .next()
            .expect("is_compound_term_leaf requires one posting");
        return Box::new(TermLeafScorer::new(
            posting,
            documents,
            leaf.scorer,
            metrics,
        ));
    }
    Box::new(WandCursor::new(
        leaf.operator,
        leaf.postings,
        documents,
        leaf.scorer,
        leaf.params.as_ref(),
        metrics,
    ))
}

fn collect_partition_with_documents<D, K>(
    documents: &D,
    leaves: Vec<LoadedLeaf>,
    plan: &CompoundScorerPlan,
    metrics: &dyn MetricsCollector,
    collector: &mut TopKCollector<K>,
    mut map_document: impl FnMut(u64) -> Result<K>,
) -> Result<CollectionStatus>
where
    D: WandDocuments + Sync,
    K: Copy + Ord,
{
    // The IU fast paths replace the generic scorer tree for the "K required
    // + N optional" shapes over plain single-posting leaves. `LANCE_FTS_IU_TIGHT`
    // gates them for A/B and rollout. Known imbalance: when every required
    // list is dense while an optional list is rare, the optional clause's
    // large upper bound keeps the multi kernel's floor prune inert and the
    // generic path's block-max window skipping can win; an optional-driven
    // intersection is the planned fix, the env switch is the escape hatch.
    if iu_tight_enabled()
        && let Some((must_indices, should_indices)) = iu_tight_clause_indices(plan)
        && iu_tight_clauses_ready(&leaves, &must_indices, &should_indices)
    {
        // Every `K >= 1 required + N >= 1 optional` shape over plain
        // single-posting leaves takes the tight kernels. The relative cost
        // gate that used to hold K=1 back is gone: with the bulk optional
        // merge in place the kernel wins for close-cost shapes too.
        return collect_iu_tight(
            documents,
            leaves,
            &must_indices,
            &should_indices,
            metrics,
            collector,
            map_document,
        );
    }
    let mut leaf_scorers = leaves
        .into_iter()
        .enumerate()
        .map(|(index, leaf)| {
            let admits = leaf_admits_term_leaf(plan, index);
            Some(box_leaf_scorer(leaf, documents, metrics, admits))
        })
        .collect::<Vec<_>>();
    let mut scorer = plan.build(&mut leaf_scorers)?;
    if leaf_scorers.iter().any(Option::is_some) {
        return Err(Error::internal(
            "compound FTS scorer did not consume every prepared leaf",
        ));
    }
    collector.collect_mapped(scorer.as_mut(), &mut map_document)
}

/// IU clause indices for K >= 1 required clauses plus N >= 1 optional
/// clauses, all identity-boost single-leaf children of one flat Boolean plan
/// with no MUST_NOT.
fn iu_tight_clause_indices(plan: &CompoundScorerPlan) -> Option<(Vec<usize>, Vec<usize>)> {
    let split = boolean_leaf_split(plan)?;
    if split.all_leaves && !split.has_must_not {
        let identity_boost = |children: &[(usize, f32)]| {
            !children.is_empty() && children.iter().all(|&(_, boost)| boost == 1.0)
        };
        if identity_boost(&split.must_leaves) && identity_boost(&split.should_leaves) {
            return Some((
                split
                    .must_leaves
                    .into_iter()
                    .map(|(index, _)| index)
                    .collect(),
                split
                    .should_leaves
                    .into_iter()
                    .map(|(index, _)| index)
                    .collect(),
            ));
        }
    }
    None
}

/// Whether every IU clause leaf is a single compressed ungrouped posting.
fn iu_tight_clauses_ready(
    leaves: &[LoadedLeaf],
    must_indices: &[usize],
    should_indices: &[usize],
) -> bool {
    if must_indices.iter().any(|&index| index >= leaves.len())
        || should_indices.iter().any(|&index| index >= leaves.len())
    {
        return false;
    }
    must_indices
        .iter()
        .chain(should_indices.iter())
        .all(|&index| {
            let leaf = &leaves[index];
            is_compound_term_leaf(leaf) && leaf.postings[0].is_compressed()
        })
}

/// Shared collection tail for the two IU kernels: floor-checked top-k
/// insertion with overflow tracking and comparison counting. The `floor`
/// handle passed into [`Self::hit`] carries the latest competitive floor back
/// to the kernels.
struct IuCollector<'a, K: Copy + Ord, F> {
    collector: &'a mut TopKCollector<K>,
    map_document: F,
    overflow: bool,
    comparisons: usize,
}

impl<K: Copy + Ord, F: FnMut(u64) -> Result<K>> IuCollector<'_, K, F> {
    /// The kernels' `on_hit`.
    fn hit(&mut self, floor: &Cell<f32>, key: u64, score: f32) -> Result<bool> {
        self.comparisons += 1;
        let score = checked_score(score, "iu_tight kernel")?;
        if score < self.collector.min_competitive_score() {
            return Ok(true);
        }
        let row_id = (self.map_document)(key)?;
        let status = self.collector.insert(ScoredRow { row_id, score });
        floor.set(self.collector.min_competitive_score());
        if status == CollectionStatus::ScoreFloorOverflow {
            self.overflow = true;
            return Ok(false);
        }
        Ok(true)
    }

    fn finish(&self, metrics: &dyn MetricsCollector) -> CollectionStatus {
        // Counted here: candidates scored (on_hit calls). WandCursor's
        // num_comparisons counts heap advances instead, so the two metrics
        // use different units and must not be compared directly.
        metrics.record_comparisons(self.comparisons);
        if self.overflow {
            CollectionStatus::ScoreFloorOverflow
        } else {
            CollectionStatus::Complete
        }
    }
}

/// Collect the IU shapes with the tight kernels: one required clause drives
/// MUST blocks with a bulk optional merge; two or more required clauses align
/// their cursors by seek-to-max. Both score the required clauses first and
/// then the matching optional clauses, in plan order, so the f32 result
/// matches the generic `ReqOptScorer` path.
fn collect_iu_tight<D, K>(
    documents: &D,
    mut leaves: Vec<LoadedLeaf>,
    must_indices: &[usize],
    should_indices: &[usize],
    metrics: &dyn MetricsCollector,
    collector: &mut TopKCollector<K>,
    map_document: impl FnMut(u64) -> Result<K>,
) -> Result<CollectionStatus>
where
    D: WandDocuments + Sync,
    K: Copy + Ord,
{
    let mut take_posting = |index: usize| -> Result<PostingIterator> {
        leaves[index]
            .postings
            .drain(..)
            .next()
            .ok_or_else(|| Error::internal("iu_tight_clauses_ready checked a single posting"))
    };
    let required = must_indices
        .iter()
        .map(|&leaf_index| take_posting(leaf_index))
        .collect::<Result<Vec<_>>>()?;
    let optional = should_indices
        .iter()
        .map(|&leaf_index| take_posting(leaf_index))
        .collect::<Result<Vec<_>>>()?;
    let scorer = leaves[must_indices[0]].scorer.clone();
    let floor = Cell::new(collector.min_competitive_score());
    let mut iu = IuCollector {
        collector,
        map_document,
        overflow: false,
        comparisons: 0,
    };
    let result = if required.len() == 1 {
        let mut required = required;
        iu_tight_search(
            documents,
            scorer.as_ref(),
            required.pop().expect("checked one required clause"),
            optional,
            &mut |key, score| iu.hit(&floor, key, score),
            || floor.get(),
        )
    } else {
        iu_tight_search_multi(
            documents,
            scorer.as_ref(),
            required,
            optional,
            &mut |key, score| iu.hit(&floor, key, score),
            || floor.get(),
        )
    };
    let status = iu.finish(metrics);
    result?;
    Ok(status)
}

fn collect_loaded_partitions(
    partitions: Vec<LoadedPartition>,
    plan: &CompoundScorerPlan,
    mask: &RowAddrMask,
    metrics: &dyn MetricsCollector,
    mut collector: TopKCollector<u64>,
) -> Result<CollectedPartitions> {
    let mut partitions = partitions.into_iter();
    while let Some(partition) = partitions.next() {
        let LoadedPartition {
            segment_ordinal,
            partition_ordinal,
            partition: source,
            documents,
            leaves,
        } = partition;
        match documents {
            LoadedDocuments::Legacy(docs) => {
                let documents = LegacyWandDocuments::new(docs.as_ref(), mask);
                let status = collect_partition_with_documents(
                    &documents,
                    leaves,
                    plan,
                    metrics,
                    &mut collector,
                    Ok,
                )?;
                debug_assert_eq!(status, CollectionStatus::Complete);
            }
            LoadedDocuments::Modern {
                documents: partition_documents,
                lengths,
                visibility,
                projection,
            } => {
                let documents = ModernWandDocuments::filtered(lengths.as_ref(), &visibility);
                if let Some(projection) = projection {
                    let status = collect_partition_with_documents(
                        &documents,
                        leaves,
                        plan,
                        metrics,
                        &mut collector,
                        |doc_id| {
                            let doc_id = DocId::new(u32::try_from(doc_id).map_err(|_| {
                                Error::index(format!(
                                    "FTS DocId {doc_id} exceeds the modern u32 domain"
                                ))
                            })?);
                            let row_id = projection.address(doc_id).ok_or_else(|| {
                                Error::internal(format!(
                                    "compound FTS scorer returned non-visible DocId {} in segment {segment_ordinal}, partition {partition_ordinal}",
                                    doc_id.get()
                                ))
                            })?;
                            Ok(row_id)
                        },
                    )?;
                    debug_assert_eq!(status, CollectionStatus::Complete);
                } else {
                    let max_buffered = collector
                        .limit
                        .saturating_add(SCORE_FLOOR_RESOLUTION_BATCH_SIZE);
                    let mut local_collector = TopKCollector::retaining_score_floor(
                        collector.limit,
                        collector.competitive_score.clone(),
                        max_buffered,
                    );
                    let status = collect_partition_with_documents(
                        &documents,
                        leaves,
                        plan,
                        metrics,
                        &mut local_collector,
                        |doc_id| {
                            Ok(DocId::new(u32::try_from(doc_id).map_err(|_| {
                                Error::index(format!(
                                    "FTS DocId {doc_id} exceeds the modern u32 domain"
                                ))
                            })?))
                        },
                    )?;
                    let boundary = match status {
                        CollectionStatus::Complete => {
                            PartitionCollectionBoundary::Deferred(DeferredCompoundRows {
                                documents: partition_documents,
                                rows: local_collector.into_candidates(),
                            })
                        }
                        CollectionStatus::ScoreFloorOverflow => {
                            PartitionCollectionBoundary::Overflow(OverflowedCompoundPartition {
                                segment_ordinal,
                                partition_ordinal,
                                partition: source,
                                documents: partition_documents,
                            })
                        }
                    };
                    return Ok(CollectedPartitions {
                        collector,
                        remaining: partitions.collect(),
                        boundary: Some(boundary),
                    });
                }
            }
        }
    }
    Ok(CollectedPartitions {
        collector,
        remaining: Vec::new(),
        boundary: None,
    })
}

async fn merge_resolved_compound_rows(
    collector: &mut TopKCollector<u64>,
    deferred: DeferredCompoundRows,
) -> Result<()> {
    for rows in deferred.rows.chunks(SCORE_FLOOR_RESOLUTION_BATCH_SIZE) {
        let doc_ids = rows.iter().map(|row| row.row_id).collect::<Vec<_>>();
        let addresses = deferred.documents.resolve_addresses(&doc_ids).await?;
        if addresses.len() != rows.len() {
            return Err(Error::internal(format!(
                "compound FTS resolved {} addresses for {} DocIds",
                addresses.len(),
                rows.len()
            )));
        }
        for (row, row_id) in rows.iter().zip(addresses) {
            let status = collector.insert(ScoredRow {
                row_id,
                score: row.score,
            });
            debug_assert_eq!(status, CollectionStatus::Complete);
        }
    }
    Ok(())
}

async fn reload_compound_partition_with_projection(
    overflow: OverflowedCompoundPartition,
    leaves: &[PreparedLeaf],
    mask: Arc<RowAddrMask>,
    metrics: Arc<dyn MetricsCollector>,
) -> Result<LoadedPartition> {
    let projection = overflow.documents.address_projection().await?;
    let mut loaded = load_compound_partition(
        overflow.segment_ordinal,
        overflow.partition_ordinal,
        overflow.partition,
        leaves,
        mask,
        metrics,
    )
    .await?
    .ok_or_else(|| {
        Error::internal(format!(
            "compound FTS retry lost visible documents in segment {}, partition {}",
            overflow.segment_ordinal, overflow.partition_ordinal
        ))
    })?;
    match &mut loaded.documents {
        LoadedDocuments::Modern {
            projection: loaded_projection,
            ..
        } => *loaded_projection = Some(projection),
        LoadedDocuments::Legacy(_) => {
            return Err(Error::internal(format!(
                "compound FTS retry changed segment {}, partition {} from modern to legacy documents",
                overflow.segment_ordinal, overflow.partition_ordinal
            )));
        }
    }
    Ok(loaded)
}

/// Search one-column compound FTS directly over posting-backed scorers.
///
/// The caller must provide all committed index segments for the column and a
/// ready prefilter. One collector owns the global top-k heap and propagates its
/// score floor through every partition-local scorer tree. Modern partitions
/// resolve candidates in bounded batches; an oversized kth-score tie is retried
/// against a resident row-address projection so final row-id ordering stays exact.
pub async fn compound_search(
    indices: &[Arc<InvertedIndex>],
    query: &FtsQuery,
    params: &FtsSearchParams,
    prefilter: Arc<dyn PreFilter>,
    metrics: Arc<dyn MetricsCollector>,
) -> Result<(Vec<u64>, Vec<f32>)> {
    compound_search_impl(indices, query, params, prefilter, metrics, None, None, None).await
}

/// Search one-column compound FTS with caller-supplied corpus-wide BM25 statistics.
///
/// The scorer must contain an entry for every token used by every query leaf,
/// including terms produced by fuzzy expansion. An incomplete scorer is rejected
/// instead of treating missing token statistics as zero.
pub async fn compound_search_with_base_scorer(
    indices: &[Arc<InvertedIndex>],
    query: &FtsQuery,
    params: &FtsSearchParams,
    prefilter: Arc<dyn PreFilter>,
    metrics: Arc<dyn MetricsCollector>,
    base_scorer: Arc<MemBM25Scorer>,
) -> Result<(Vec<u64>, Vec<f32>)> {
    compound_search_impl(
        indices,
        query,
        params,
        prefilter,
        metrics,
        Some(base_scorer),
        None,
        None,
    )
    .await
}

/// Search one-column compound FTS with corpus-wide BM25 statistics and an
/// inclusive initial score floor.
///
/// The floor may only remove scores strictly below it. Equal-score rows must
/// still be visited because final ordering uses row id as its secondary key.
pub async fn compound_search_with_base_scorer_and_score_floor(
    indices: &[Arc<InvertedIndex>],
    query: &FtsQuery,
    params: &FtsSearchParams,
    prefilter: Arc<dyn PreFilter>,
    metrics: Arc<dyn MetricsCollector>,
    base_scorer: Arc<MemBM25Scorer>,
    score_floor: f32,
) -> Result<(Vec<u64>, Vec<f32>)> {
    compound_search_impl(
        indices,
        query,
        params,
        prefilter,
        metrics,
        Some(base_scorer),
        Some(score_floor),
        None,
    )
    .await
}

/// Replay one root Match query with the exact vocabulary/scorer pair used by
/// an earlier bounded WAND probe.
#[doc(hidden)]
pub async fn compound_search_prepared_match(
    indices: &[Arc<InvertedIndex>],
    query: &FtsQuery,
    params: &FtsSearchParams,
    prefilter: Arc<dyn PreFilter>,
    metrics: Arc<dyn MetricsCollector>,
    prepared_match: Arc<PreparedBm25Query>,
) -> Result<(Vec<u64>, Vec<f32>)> {
    if !matches!(query, FtsQuery::Match(_)) {
        return Err(Error::invalid_input(
            "prepared Match replay requires a root Match query",
        ));
    }
    compound_search_impl(
        indices,
        query,
        params,
        prefilter,
        metrics,
        None,
        None,
        Some(prepared_match),
    )
    .await
}

/// Replay one root Match query with a prepared vocabulary/scorer pair and an
/// inclusive initial score floor.
#[doc(hidden)]
pub async fn compound_search_prepared_match_with_score_floor(
    indices: &[Arc<InvertedIndex>],
    query: &FtsQuery,
    params: &FtsSearchParams,
    prefilter: Arc<dyn PreFilter>,
    metrics: Arc<dyn MetricsCollector>,
    prepared_match: Arc<PreparedBm25Query>,
    score_floor: f32,
) -> Result<(Vec<u64>, Vec<f32>)> {
    if !matches!(query, FtsQuery::Match(_)) {
        return Err(Error::invalid_input(
            "prepared Match replay requires a root Match query",
        ));
    }
    compound_search_impl(
        indices,
        query,
        params,
        prefilter,
        metrics,
        None,
        Some(score_floor),
        Some(prepared_match),
    )
    .await
}

// These arguments keep the public entry points explicit while centralizing the
// shared search loop; bundling them would only move the same independent inputs
// into an internal forwarding struct.
#[allow(clippy::too_many_arguments)]
async fn compound_search_impl(
    indices: &[Arc<InvertedIndex>],
    query: &FtsQuery,
    params: &FtsSearchParams,
    prefilter: Arc<dyn PreFilter>,
    metrics: Arc<dyn MetricsCollector>,
    base_scorer: Option<Arc<MemBM25Scorer>>,
    initial_score_floor: Option<f32>,
    prepared_match: Option<Arc<PreparedBm25Query>>,
) -> Result<(Vec<u64>, Vec<f32>)> {
    let limit = params.limit.unwrap_or(usize::MAX);
    if limit == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let (plan, leaves) = prepare_compound_query(
        indices,
        query,
        params,
        metrics.as_ref(),
        base_scorer,
        prepared_match,
    )
    .await?;
    prefilter.wait_for_ready().await?;
    let mask = prefilter.mask();
    let competitive_score = Arc::new(CompetitiveScore::default());
    if let Some(score_floor) = initial_score_floor {
        competitive_score.raise(checked_score(score_floor, "initial compound score floor")?);
    }
    let mut collector = TopKCollector::with_competitive_score(limit, competitive_score);

    for (segment_ordinal, index) in indices.iter().enumerate() {
        let loads =
            index
                .partitions
                .iter()
                .cloned()
                .enumerate()
                .map(|(partition_ordinal, partition)| {
                    load_compound_partition(
                        segment_ordinal,
                        partition_ordinal,
                        partition,
                        &leaves,
                        mask.clone(),
                        metrics.clone(),
                    )
                });
        let mut partitions = stream::iter(loads)
            .buffer_unordered(get_num_compute_intensive_cpus().clamp(1, 32))
            .try_collect::<Vec<_>>()
            .await?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        while !partitions.is_empty() {
            let cpu_plan = plan.clone();
            let cpu_mask = mask.clone();
            let cpu_metrics = metrics.clone();
            let collected = spawn_cpu(move || {
                collect_loaded_partitions(
                    partitions,
                    &cpu_plan,
                    cpu_mask.as_ref(),
                    cpu_metrics.as_ref(),
                    collector,
                )
            })
            .await?;
            collector = collected.collector;
            partitions = collected.remaining;
            match collected.boundary {
                Some(PartitionCollectionBoundary::Deferred(deferred)) => {
                    merge_resolved_compound_rows(&mut collector, deferred).await?;
                }
                Some(PartitionCollectionBoundary::Overflow(overflow)) => {
                    let retry = reload_compound_partition_with_projection(
                        overflow,
                        &leaves,
                        mask.clone(),
                        metrics.clone(),
                    )
                    .await?;
                    partitions.insert(0, retry);
                }
                None => debug_assert!(partitions.is_empty()),
            }
        }
    }

    let rows = collector.into_rows();
    Ok(rows.into_iter().map(|row| (row.row_id, row.score)).unzip())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::AtomicUsize;

    use arrow::buffer::ScalarBuffer;
    use rand::{Rng, SeedableRng, rngs::SmallRng};

    use super::super::documents::{
        ordered_row_address_projection_for_test, resident_row_address_projection_for_test,
    };
    use super::super::encoding::compress_posting_list;
    use super::super::index::{PlainPostingList, PostingList};
    use super::super::scorer::Scorer;
    use super::super::wand::GroupedTermScorer;
    use super::super::{CompressedPostingList, LEGACY_BLOCK_SIZE, PostingTailCodec};
    use super::*;
    use crate::metrics::{LocalMetricsCollector, NoOpMetricsCollector};
    use crate::scalar::inverted::query::MultiMatchQuery;

    fn rows(values: &[(u64, f32)]) -> Vec<ScoredRow> {
        values
            .iter()
            .map(|(row_id, score)| ScoredRow::new(*row_id, *score).unwrap())
            .collect()
    }

    fn materialized(values: &[(u64, f32)]) -> Box<dyn ComposableScorer> {
        Box::new(MaterializedScorer::try_new(rows(values)).unwrap())
    }

    #[test]
    fn materialized_compound_top_k_preserves_multimatch_and_tie_order() {
        let query = FtsQuery::MultiMatch(MultiMatchQuery {
            match_queries: vec![
                MatchQuery::new("alpha".to_string()).with_column(Some("text".to_string())),
                MatchQuery::new("alpha".to_string()).with_column(Some("text".to_string())),
            ],
        });
        let (row_ids, scores) = materialized_compound_top_k(
            &query,
            vec![vec![(7, 1.0), (3, 2.0)], vec![(7, 3.0), (5, 3.0)]],
            2,
        )
        .unwrap();

        assert_eq!(row_ids, vec![5, 7]);
        assert_eq!(scores, vec![3.0, 3.0]);
    }

    fn zero_weight_wand<'a>(
        documents: &'a DocSet,
        scorer: Arc<MemBM25Scorer>,
        params: &'a FtsSearchParams,
        metrics: &'a dyn MetricsCollector,
    ) -> BoxScorer<'a> {
        let query_weight = scorer.query_weight("common");
        assert_eq!(query_weight, 0.0);
        let posting = PostingIterator::with_query_weight(
            "common".to_owned(),
            0,
            0,
            query_weight,
            PostingList::Plain(PlainPostingList::new(
                ScalarBuffer::from(vec![0_u64]),
                ScalarBuffer::from(vec![1.0_f32]),
                Some(0.0),
                None,
            )),
            1,
        );
        Box::new(WandCursor::new(
            Operator::Or,
            vec![posting],
            documents,
            scorer,
            params,
            metrics,
        ))
    }

    fn mapping_error(result: Result<Option<RowAddressSource<'_>>>) -> Error {
        match result {
            Err(error) => error,
            Ok(_) => panic!("row-address mapping unexpectedly succeeded"),
        }
    }

    fn should_maxscore<'a>(children: Vec<BoxScorer<'a>>) -> ShouldMaxScoreScorer<'a> {
        let global_bounds = ShouldMaxScoreScorer::global_bounds(&children).unwrap();
        ShouldMaxScoreScorer::new(children, global_bounds)
    }

    #[test]
    fn score_bounds_are_conservative_under_nested_sum_and_boost() {
        let should = DisjunctionScorer::try_new(
            vec![
                materialized(&[(0, 0.1), (2, 2.0)]),
                materialized(&[(0, 0.2)]),
            ],
            DisjunctionScore::Sum,
        )
        .unwrap();
        let negative = materialized(&[(0, 0.3), (2, 5.0)]);
        let mut scorer = BoostScorer::try_new(Box::new(should), negative, 0.5).unwrap();

        assert_eq!(scorer.next().unwrap(), Some(0));
        let up_to = scorer.advance_shallow(0).unwrap();
        let bounds = scorer.score_bounds(up_to).unwrap();
        let first_score = scorer.score().unwrap();
        assert!(bounds.lower <= first_score);
        assert!(bounds.upper >= first_score);

        assert_eq!(scorer.next().unwrap(), Some(2));
        let second_score = scorer.score().unwrap();
        assert!(second_score.is_sign_negative());
        let up_to = scorer.advance_shallow(2).unwrap();
        let bounds = scorer.score_bounds(up_to).unwrap();
        assert!(bounds.lower <= second_score);
        assert!(bounds.upper >= second_score);
    }

    #[test]
    fn scaled_score_floor_is_maximal_and_preserves_equalities() {
        let cases = [
            (3.75_f32, 2.5_f32),
            (f32::from_bits(1.0_f32.to_bits() + 1), 1.000_000_2_f32),
            (2.0_f32, f32::MIN_POSITIVE),
            (1.0_f32, f32::from_bits(1)),
        ];
        for (raw_score, factor) in cases {
            let scaled_score = raw_score * factor;
            assert!(scaled_score.is_finite() && scaled_score > 0.0);
            let child_floor = exclusive_scaled_score_floor(scaled_score, factor).unwrap();
            assert!(child_floor * factor < scaled_score);
            assert!(child_floor < raw_score);
            let next_raw = f32::from_bits(child_floor.to_bits() + 1);
            assert!(next_raw * factor >= scaled_score);

            let mut scorer = ScaleScorer::try_new(materialized(&[(0, raw_score)]), factor).unwrap();
            scorer.set_min_competitive_score(scaled_score).unwrap();
            assert_eq!(scorer.next().unwrap(), Some(0));
            assert_eq!(scorer.score().unwrap(), scaled_score);
        }

        let subnormal_factor = f32::from_bits(1);
        let raw_score = 0.25_f32;
        let scaled_score = raw_score * subnormal_factor;
        assert_eq!(scaled_score, 0.0);
        assert_eq!(
            exclusive_scaled_score_floor(scaled_score, subnormal_factor),
            None
        );
        let mut scorer =
            ScaleScorer::try_new(materialized(&[(0, raw_score)]), subnormal_factor).unwrap();
        scorer.set_min_competitive_score(scaled_score).unwrap();
        assert_eq!(scorer.next().unwrap(), Some(0));
        assert_eq!(scorer.score().unwrap(), 0.0);
    }

    #[test]
    fn scale_scorer_only_translates_strictly_higher_floors() {
        let (child, work) = instrumented(materialized(&[(0, 10.0)]));
        let mut scorer = ScaleScorer::try_new(child, 2.0).unwrap();

        scorer.set_min_competitive_score(4.0).unwrap();
        scorer.set_min_competitive_score(4.0).unwrap();
        scorer.set_min_competitive_score(3.0).unwrap();
        assert_eq!(scorer.score_floor_translations, 1);
        assert_eq!(work.floors.load(AtomicOrdering::Relaxed), 1);

        scorer.set_min_competitive_score(5.0).unwrap();
        assert_eq!(scorer.score_floor_translations, 2);
        assert_eq!(work.floors.load(AtomicOrdering::Relaxed), 2);

        let error = scorer.set_min_competitive_score(f32::NAN).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("cannot be NaN"));
        assert_eq!(scorer.score_floor_translations, 2);

        let subnormal_factor = f32::from_bits(1);
        let (child, work) = instrumented(materialized(&[(0, 1.0)]));
        let mut scorer = ScaleScorer::try_new(child, subnormal_factor).unwrap();
        scorer.set_min_competitive_score(0.0).unwrap();
        scorer.set_min_competitive_score(0.0).unwrap();
        assert_eq!(scorer.score_floor_translations, 1);
        assert_eq!(work.floors.load(AtomicOrdering::Relaxed), 0);

        scorer.set_min_competitive_score(f32::from_bits(1)).unwrap();
        assert_eq!(scorer.score_floor_translations, 2);
        assert_eq!(work.floors.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    fn materialized_scorer_precomputes_multi_block_bounds() {
        assert_eq!(std::mem::size_of::<ScoreBounds>(), 8);
        let values = [
            (0, 5.0),
            (1, 1.0),
            (2, 3.0),
            (3, -2.0),
            (4, 7.0),
            (5, 4.0),
            (6, 8.0),
            (7, 6.0),
            (8, 9.0),
            (9, 0.0),
        ];
        let mut scorer = MaterializedScorer::try_new(rows(&values))
            .unwrap()
            .with_block_size(3);

        assert_eq!(scorer.num_bound_blocks(), 4);
        assert_eq!(scorer.bound_score_visits(), values.len());
        assert_eq!(scorer.global_score_upper_bound(), Some(9.0));
        assert_eq!(scorer.advance_shallow(4).unwrap(), 5);
        assert_eq!(
            scorer.score_bounds(4).unwrap(),
            ScoreBounds {
                lower: -2.0,
                upper: 7.0,
            }
        );
        for _ in 0..100 {
            assert_eq!(scorer.score_bounds(5).unwrap().upper, 7.0);
        }
        assert_eq!(scorer.bound_score_visits(), values.len());

        scorer.set_min_competitive_score(8.0).unwrap();
        assert_eq!(scorer.next().unwrap(), Some(6));
        assert_eq!(scorer.bound_score_visits(), values.len());
    }

    #[test]
    fn collector_propagates_threshold_across_partitions_and_keeps_ties() {
        let mut collector = TopKCollector::new(2);
        let mut first = MaterializedScorer::try_new(rows(&[(8, 9.0), (4, 10.0), (3, 9.0)]))
            .unwrap()
            .with_block_size(1);
        collector.collect_mapped(&mut first, Ok).unwrap();
        assert_eq!(collector.competitive_score.get(), 9.0);

        let mut second = MaterializedScorer::try_new(rows(&[(1, 1.0), (2, 9.0)]))
            .unwrap()
            .with_block_size(1);
        collector.collect_mapped(&mut second, Ok).unwrap();
        assert_eq!(
            collector.into_rows(),
            vec![
                ScoredRow {
                    row_id: 4,
                    score: 10.0
                },
                ScoredRow {
                    row_id: 2,
                    score: 9.0
                }
            ]
        );
    }

    #[test]
    fn seeded_collector_keeps_floor_equalities_for_row_id_ordering() {
        let competitive_score = Arc::new(CompetitiveScore::default());
        competitive_score.raise(5.0);
        let mut collector = TopKCollector::with_competitive_score(2, competitive_score);

        let mut later_segment = MaterializedScorer::try_new(rows(&[(2, 4.0), (99, 5.0)])).unwrap();
        collector.collect_mapped(&mut later_segment, Ok).unwrap();
        let mut earlier_segment =
            MaterializedScorer::try_new(rows(&[(1, 5.0), (50, 6.0)])).unwrap();
        collector.collect_mapped(&mut earlier_segment, Ok).unwrap();

        assert_eq!(collector.into_rows(), rows(&[(50, 6.0), (1, 5.0)]));
    }

    #[test]
    fn collector_bounds_equal_score_candidates() {
        let limit = 1;
        let num_candidates = DEFAULT_BLOCK_SIZE * 4;
        let values = (0..num_candidates)
            .map(|row_id| (row_id as u64, 1.0))
            .collect::<Vec<_>>();
        let mut scorer = MaterializedScorer::try_new(rows(&values)).unwrap();
        let max_buffered = limit + SCORE_FLOOR_RESOLUTION_BATCH_SIZE;
        let mut collector = TopKCollector::retaining_score_floor(
            limit,
            Arc::new(CompetitiveScore::default()),
            max_buffered,
        );

        let status = collector.collect_mapped(&mut scorer, Ok).unwrap();

        assert_eq!(status, CollectionStatus::ScoreFloorOverflow);
        assert_eq!(collector.heap.len(), max_buffered);
    }

    #[test]
    fn collector_reclaims_obsolete_score_floor_before_overflowing() {
        let limit = 2;
        let max_buffered = 4;
        let values = [
            (0, 1.0),
            (1, 1.0),
            (2, 1.0),
            (3, 2.0),
            (4, 2.0),
            (5, 2.0),
            (6, 2.0),
            (7, 2.0),
        ];
        let mut scorer = MaterializedScorer::try_new(rows(&values)).unwrap();
        let competitive_score = Arc::new(CompetitiveScore::default());
        let mut collector =
            TopKCollector::retaining_score_floor(limit, competitive_score.clone(), max_buffered);

        let status = collector.collect_mapped(&mut scorer, Ok).unwrap();

        assert_eq!(status, CollectionStatus::ScoreFloorOverflow);
        assert_eq!(collector.heap.len(), max_buffered);
        assert!(collector.heap.iter().all(|row| row.0.score == 2.0));
        assert_eq!(competitive_score.get(), 2.0);
    }

    struct TwoPhaseScorer {
        inner: MaterializedScorer,
        accepted: Vec<u64>,
        match_cost: Option<f32>,
        has_doc_upper: bool,
        approximations: Arc<AtomicUsize>,
        confirmations: Arc<AtomicUsize>,
    }

    impl ComposableScorer for TwoPhaseScorer {
        fn doc(&self) -> Option<u64> {
            self.inner.doc()
        }

        fn next(&mut self) -> Result<Option<u64>> {
            let doc = self.inner.next()?;
            if doc.is_some() {
                self.approximations.fetch_add(1, AtomicOrdering::Relaxed);
            }
            Ok(doc)
        }

        fn advance(&mut self, target: u64) -> Result<Option<u64>> {
            let doc = self.inner.advance(target)?;
            if doc.is_some() {
                self.approximations.fetch_add(1, AtomicOrdering::Relaxed);
            }
            Ok(doc)
        }

        fn cost(&self) -> usize {
            self.inner.cost()
        }

        fn score(&mut self) -> Result<f32> {
            self.inner.score()
        }

        fn advance_shallow(&mut self, target: u64) -> Result<u64> {
            self.inner.advance_shallow(target)
        }

        fn score_bounds(&mut self, up_to: u64) -> Result<ScoreBounds> {
            self.inner.score_bounds(up_to)
        }

        fn global_score_upper_bound(&self) -> Option<f32> {
            self.inner.global_score_upper_bound()
        }

        fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()> {
            self.inner.set_min_competitive_score(min_score)
        }

        fn current_score_upper_bound(&mut self) -> Result<Option<f32>> {
            if self.has_doc_upper {
                self.inner.score().map(Some)
            } else {
                Ok(None)
            }
        }

        fn supports_doc_local_confirmation_pruning(&self) -> bool {
            true
        }

        fn matches(&mut self) -> Result<bool> {
            self.confirmations.fetch_add(1, AtomicOrdering::Relaxed);
            Ok(self
                .doc()
                .is_some_and(|doc| self.accepted.binary_search(&doc).is_ok()))
        }

        fn match_cost(&self) -> Option<f32> {
            self.match_cost
        }

        fn scores_non_negative(&self) -> bool {
            true
        }
    }

    fn two_phase(
        values: &[(u64, f32)],
        accepted: Vec<u64>,
        match_cost: Option<f32>,
    ) -> (
        Box<dyn ComposableScorer>,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
    ) {
        let approximations = Arc::new(AtomicUsize::new(0));
        let confirmations = Arc::new(AtomicUsize::new(0));
        let scorer = TwoPhaseScorer {
            inner: MaterializedScorer::try_new(rows(values)).unwrap(),
            accepted,
            match_cost,
            has_doc_upper: true,
            approximations: approximations.clone(),
            confirmations: confirmations.clone(),
        };
        (Box::new(scorer), approximations, confirmations)
    }

    fn two_phase_without_doc_upper(
        values: &[(u64, f32)],
        accepted: Vec<u64>,
    ) -> (
        Box<dyn ComposableScorer>,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
    ) {
        let approximations = Arc::new(AtomicUsize::new(0));
        let confirmations = Arc::new(AtomicUsize::new(0));
        let scorer = TwoPhaseScorer {
            inner: MaterializedScorer::try_new(rows(values)).unwrap(),
            accepted,
            match_cost: Some(1.0),
            has_doc_upper: false,
            approximations: approximations.clone(),
            confirmations: confirmations.clone(),
        };
        (Box::new(scorer), approximations, confirmations)
    }

    #[derive(Default)]
    struct ScorerWork {
        advances: AtomicUsize,
        confirmations: AtomicUsize,
        shallow_advances: AtomicUsize,
        bounds: AtomicUsize,
        floors: AtomicUsize,
    }

    struct InstrumentedScorer<'a> {
        inner: BoxScorer<'a>,
        work: Arc<ScorerWork>,
    }

    impl ComposableScorer for InstrumentedScorer<'_> {
        fn doc(&self) -> Option<u64> {
            self.inner.doc()
        }

        fn document_key(&self) -> Option<u64> {
            self.inner.document_key()
        }

        fn next(&mut self) -> Result<Option<u64>> {
            let doc = self.inner.next()?;
            if doc.is_some() {
                self.work.advances.fetch_add(1, AtomicOrdering::Relaxed);
            }
            Ok(doc)
        }

        fn advance(&mut self, target: u64) -> Result<Option<u64>> {
            let doc = self.inner.advance(target)?;
            if doc.is_some() {
                self.work.advances.fetch_add(1, AtomicOrdering::Relaxed);
            }
            Ok(doc)
        }

        fn cost(&self) -> usize {
            self.inner.cost()
        }

        fn score(&mut self) -> Result<f32> {
            self.inner.score()
        }

        fn advance_shallow(&mut self, target: u64) -> Result<u64> {
            self.work
                .shallow_advances
                .fetch_add(1, AtomicOrdering::Relaxed);
            self.inner.advance_shallow(target)
        }

        fn score_bounds(&mut self, up_to: u64) -> Result<ScoreBounds> {
            self.work.bounds.fetch_add(1, AtomicOrdering::Relaxed);
            self.inner.score_bounds(up_to)
        }

        fn global_score_upper_bound(&self) -> Option<f32> {
            self.inner.global_score_upper_bound()
        }

        fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()> {
            self.work.floors.fetch_add(1, AtomicOrdering::Relaxed);
            self.inner.set_min_competitive_score(min_score)
        }

        fn current_score_upper_bound(&mut self) -> Result<Option<f32>> {
            self.inner.current_score_upper_bound()
        }

        fn supports_doc_local_confirmation_pruning(&self) -> bool {
            self.inner.supports_doc_local_confirmation_pruning()
        }

        fn matches(&mut self) -> Result<bool> {
            self.work
                .confirmations
                .fetch_add(1, AtomicOrdering::Relaxed);
            self.inner.matches()
        }

        fn match_cost(&self) -> Option<f32> {
            self.inner.match_cost()
        }

        fn scores_non_negative(&self) -> bool {
            self.inner.scores_non_negative()
        }
    }

    fn instrumented<'a>(inner: BoxScorer<'a>) -> (BoxScorer<'a>, Arc<ScorerWork>) {
        let work = Arc::new(ScorerWork::default());
        (
            Box::new(InstrumentedScorer {
                inner,
                work: work.clone(),
            }),
            work,
        )
    }

    fn row_address_source(values: &[(u64, f32)]) -> RowAddressSource<'static> {
        let min_possible_row_address = values
            .iter()
            .map(|(row_address, _)| *row_address)
            .min()
            .unwrap();
        RowAddressSource::new(min_possible_row_address, materialized(values))
    }

    #[test]
    fn row_address_scorer_maps_gaps_advance_and_shallow_bounds() {
        let projection = ordered_row_address_projection_for_test(vec![10, 20, 50, 100, 200]);
        let source = Box::new(
            MaterializedScorer::try_new(rows(&[(0, 1.0), (2, 5.0), (4, 9.0)]))
                .unwrap()
                .with_block_size(2),
        );
        let mut scorer = RowAddressScorer::new(source, projection);

        assert_eq!(scorer.next().unwrap(), Some(10));
        assert_eq!(scorer.document_key(), Some(10));
        let up_to = scorer.advance_shallow(10).unwrap();
        assert_eq!(up_to, 199);
        assert_eq!(
            scorer.score_bounds(150).unwrap(),
            ScoreBounds {
                lower: 1.0,
                upper: 5.0,
            }
        );
        assert_eq!(scorer.global_score_upper_bound(), Some(9.0));

        assert_eq!(scorer.advance(21).unwrap(), Some(50));
        assert_eq!(scorer.score().unwrap(), 5.0);
        assert_eq!(scorer.advance(51).unwrap(), Some(200));
        assert_eq!(scorer.score().unwrap(), 9.0);
        assert_eq!(scorer.next().unwrap(), None);
    }

    #[test]
    fn prepared_row_address_projection_is_reusable_across_leaf_scorers() {
        let ordered_projection = resident_row_address_projection_for_test(vec![10, 20, 50]);
        let ordered = prepare_row_address_projection(&ordered_projection);
        assert_eq!(
            ordered_projection.cached_row_address_order(),
            CachedRowAddressOrder::Unknown
        );
        assert_eq!(ordered_projection.ordered_validation_visited_docs(), 0);

        let first = map_scorer_to_row_addresses_with_threshold(
            materialized(&[(0, 1.0), (2, 3.0)]),
            &ordered,
            0,
            10,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            ordered_projection.cached_row_address_order(),
            CachedRowAddressOrder::Ordered
        );
        assert_eq!(ordered_projection.ordered_validation_visited_docs(), 3);
        let second = map_scorer_to_row_addresses(materialized(&[(1, 2.0)]), &ordered, 0)
            .unwrap()
            .unwrap();
        assert_eq!(ordered_projection.ordered_validation_visited_docs(), 3);
        let first = TopKCollector::new(10)
            .collect(first.into_scorer().as_mut())
            .unwrap();
        let second = TopKCollector::new(10)
            .collect(second.into_scorer().as_mut())
            .unwrap();
        assert_eq!(first, rows(&[(50, 3.0), (10, 1.0)]));
        assert_eq!(second, rows(&[(20, 2.0)]));

        let delayed = map_scorer_to_row_addresses(materialized(&[(2, 3.0)]), &ordered, 2)
            .unwrap()
            .unwrap();
        assert_eq!(delayed.min_possible_row_address, 50);

        let unordered_projection = resident_row_address_projection_for_test(vec![30, 10, 20]);
        let unordered = prepare_row_address_projection(&unordered_projection);
        let first = map_scorer_to_row_addresses_with_threshold(
            materialized(&[(0, 1.0), (1, 2.0)]),
            &unordered,
            0,
            10,
        )
        .unwrap()
        .unwrap();
        assert!(matches!(
            unordered_projection.cached_row_address_order(),
            CachedRowAddressOrder::OutOfOrder
        ));
        assert_eq!(unordered_projection.ordered_validation_visited_docs(), 2);
        let second = map_scorer_to_row_addresses(materialized(&[(2, 4.0)]), &unordered, 0)
            .unwrap()
            .unwrap();
        assert_eq!(unordered_projection.ordered_validation_visited_docs(), 2);
        let first = TopKCollector::new(10)
            .collect(first.into_scorer().as_mut())
            .unwrap();
        let second = TopKCollector::new(10)
            .collect(second.into_scorer().as_mut())
            .unwrap();
        assert_eq!(first, rows(&[(10, 2.0), (30, 1.0)]));
        assert_eq!(second, rows(&[(20, 4.0)]));
    }

    #[test]
    fn adaptive_projection_materializes_sparse_unknown_without_validation() {
        let projection = resident_row_address_projection_for_test(
            (0..100).map(|local_doc| local_doc * 10).collect(),
        );
        let prepared = prepare_row_address_projection(&projection);

        let source = map_scorer_to_row_addresses_with_threshold(
            materialized(&[(73, 4.0)]),
            &prepared,
            73,
            10,
        )
        .unwrap()
        .unwrap();

        assert_eq!(projection.ordered_validation_visited_docs(), 0);
        assert_eq!(
            projection.cached_row_address_order(),
            CachedRowAddressOrder::Unknown
        );
        let result = TopKCollector::new(1)
            .collect(source.into_scorer().as_mut())
            .unwrap();
        assert_eq!(result, rows(&[(730, 4.0)]));
    }

    #[test]
    fn adaptive_projection_amortizes_repeated_sparse_materialization() {
        let projection = resident_row_address_projection_for_test(
            (0..100).map(|local_doc| local_doc * 10).collect(),
        );
        let prepared = prepare_row_address_projection(&projection);

        for _ in 0..10 {
            map_scorer_to_row_addresses_with_threshold(
                materialized(&[(73, 4.0)]),
                &prepared,
                73,
                10,
            )
            .unwrap();
        }
        assert_eq!(projection.ordered_validation_visited_docs(), 0);
        assert_eq!(
            projection.cached_row_address_order(),
            CachedRowAddressOrder::Unknown
        );

        map_scorer_to_row_addresses_with_threshold(materialized(&[(73, 4.0)]), &prepared, 73, 10)
            .unwrap();
        assert_eq!(projection.ordered_validation_visited_docs(), 100);
        assert_eq!(
            projection.cached_row_address_order(),
            CachedRowAddressOrder::Ordered
        );

        map_scorer_to_row_addresses_with_threshold(materialized(&[(73, 4.0)]), &prepared, 73, 10)
            .unwrap();
        assert_eq!(projection.ordered_validation_visited_docs(), 100);
    }

    #[test]
    fn adaptive_projection_validates_dense_unknown_once() {
        let projection = resident_row_address_projection_for_test(vec![10, 20, 30, 40]);
        let prepared = prepare_row_address_projection(&projection);

        map_scorer_to_row_addresses_with_threshold(
            materialized(&[(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)]),
            &prepared,
            0,
            10,
        )
        .unwrap();
        assert_eq!(projection.ordered_validation_visited_docs(), 4);
        assert_eq!(
            projection.cached_row_address_order(),
            CachedRowAddressOrder::Ordered
        );

        map_scorer_to_row_addresses_with_threshold(
            materialized(&[(0, 2.0), (1, 2.0), (2, 2.0), (3, 2.0)]),
            &prepared,
            0,
            10,
        )
        .unwrap();
        assert_eq!(projection.ordered_validation_visited_docs(), 4);
    }

    #[test]
    fn adaptive_projection_tracks_unknown_duplicates_across_leaves() {
        let projection = resident_row_address_projection_for_test(vec![10, 10]);
        let prepared = prepare_row_address_projection(&projection);

        map_scorer_to_row_addresses_with_threshold(materialized(&[(0, 1.0)]), &prepared, 0, 1000)
            .unwrap();
        // The same physical local document can match more than one leaf.
        map_scorer_to_row_addresses_with_threshold(materialized(&[(0, 2.0)]), &prepared, 0, 1000)
            .unwrap();
        let error = mapping_error(map_scorer_to_row_addresses_with_threshold(
            materialized(&[(1, 3.0)]),
            &prepared,
            1,
            1000,
        ));

        assert!(
            error
                .to_string()
                .contains("distinct local documents 0 and 1")
        );
        assert_eq!(projection.ordered_validation_visited_docs(), 0);
        assert_eq!(
            projection.cached_row_address_order(),
            CachedRowAddressOrder::Unknown
        );
    }

    #[test]
    fn adaptive_projection_tracks_out_of_order_duplicates_across_leaves() {
        let projection = resident_row_address_projection_for_test(vec![10, 30, 10]);
        let prepared = prepare_row_address_projection(&projection);

        map_scorer_to_row_addresses_with_threshold(materialized(&[(0, 1.0)]), &prepared, 0, 0)
            .unwrap();
        assert!(matches!(
            projection.cached_row_address_order(),
            CachedRowAddressOrder::OutOfOrder
        ));
        assert_eq!(projection.ordered_validation_visited_docs(), 3);

        let error = mapping_error(map_scorer_to_row_addresses_with_threshold(
            materialized(&[(2, 2.0)]),
            &prepared,
            2,
            100,
        ));
        assert!(
            error
                .to_string()
                .contains("distinct local documents 0 and 2")
        );
        assert_eq!(projection.ordered_validation_visited_docs(), 3);
    }

    #[test]
    fn adaptive_projection_rejects_cached_duplicates_without_materializing() {
        let projection = resident_row_address_projection_for_test(vec![10, 10]);
        let prepared = prepare_row_address_projection(&projection);

        let first_error = mapping_error(map_scorer_to_row_addresses_with_threshold(
            materialized(&[(0, 1.0)]),
            &prepared,
            0,
            0,
        ));
        assert!(
            first_error
                .to_string()
                .contains("shared by local documents 0 and 1")
        );
        assert_eq!(projection.ordered_validation_visited_docs(), 2);
        assert!(matches!(
            projection.cached_row_address_order(),
            CachedRowAddressOrder::Duplicate
        ));

        let second_error = mapping_error(map_scorer_to_row_addresses_with_threshold(
            materialized(&[(1, 2.0)]),
            &prepared,
            1,
            100,
        ));
        assert!(
            second_error
                .to_string()
                .contains("shared by local documents 0 and 1")
        );
        // The atomic cache stores only the invalid category. Reconstructing
        // exact duplicate diagnostics is a rare error-path rescan.
        assert_eq!(projection.ordered_validation_visited_docs(), 4);
    }

    #[test]
    fn materialized_projection_reports_single_leaf_duplicates_locally() {
        let projection = resident_row_address_projection_for_test(vec![10, 10]);
        let prepared = prepare_row_address_projection(&projection);
        let error = mapping_error(map_scorer_to_row_addresses_with_threshold(
            materialized(&[(0, 1.0), (1, 2.0)]),
            &prepared,
            0,
            100,
        ));

        assert!(error.to_string().contains("duplicate row_id=10"));
        assert_eq!(projection.ordered_validation_visited_docs(), 0);
    }

    #[test]
    fn row_address_merge_orders_gapped_sources_and_keeps_score_ties() {
        let mut scorer = RowAddressMergeScorer::try_new(vec![
            row_address_source(&[(10, 5.0), (100, 2.0)]),
            row_address_source(&[(20, 5.0), (70, 8.0)]),
        ])
        .unwrap();

        let results = TopKCollector::new(10).collect(&mut scorer).unwrap();
        assert_eq!(
            results,
            rows(&[(70, 8.0), (10, 5.0), (20, 5.0), (100, 2.0)])
        );
    }

    #[test]
    fn row_address_merge_advance_skips_sources_with_a_heap() {
        let mut scorer = RowAddressMergeScorer::try_new(vec![
            row_address_source(&[(10, 1.0), (100, 2.0)]),
            row_address_source(&[(20, 3.0), (70, 4.0)]),
            row_address_source(&[(30, 5.0), (90, 6.0)]),
        ])
        .unwrap();

        assert_eq!(scorer.advance(65).unwrap(), Some(70));
        assert_eq!(scorer.next().unwrap(), Some(90));
        assert_eq!(scorer.next().unwrap(), Some(100));
        assert_eq!(scorer.next().unwrap(), None);
    }

    #[test]
    fn row_address_merge_delays_pending_sources_and_shallow_work() {
        let first = Box::new(
            MaterializedScorer::try_new(rows(&[(10, 1.0), (20, 4.0), (2_000, 8.0)]))
                .unwrap()
                .with_block_size(3),
        );
        let second = Box::new(
            MaterializedScorer::try_new(rows(&[(1_000, 10.0)]))
                .unwrap()
                .with_block_size(1),
        );
        let (first, first_work) = instrumented(first);
        let (second, second_work) = instrumented(second);
        let mut scorer = RowAddressMergeScorer::try_new(vec![
            RowAddressSource::new(10, first),
            RowAddressSource::new(1_000, second),
        ])
        .unwrap();

        assert_eq!(scorer.next().unwrap(), Some(10));
        assert_eq!(first_work.advances.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(second_work.advances.load(AtomicOrdering::Relaxed), 0);

        let up_to = scorer.advance_shallow(10).unwrap();
        assert_eq!(up_to, 999);
        // Materialized bounds conservatively cover the whole source block,
        // including its row at 2_000, while the merge window still stops
        // before the pending source at 1_000.
        assert_eq!(scorer.score_bounds(up_to).unwrap().upper, 8.0);
        assert_eq!(first_work.shallow_advances.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(first_work.bounds.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(
            second_work.shallow_advances.load(AtomicOrdering::Relaxed),
            0
        );
        assert_eq!(second_work.bounds.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(scorer.global_score_upper_bound(), Some(10.0));
    }

    #[test]
    fn row_address_merge_pushes_new_floors_only_to_active_sources() {
        let (first, first_work) = instrumented(materialized(&[(10, 1.0), (20, 2.0)]));
        let (second, second_work) = instrumented(materialized(&[(1_000, 10.0)]));
        let mut scorer = RowAddressMergeScorer::try_new(vec![
            RowAddressSource::new(10, first),
            RowAddressSource::new(1_000, second),
        ])
        .unwrap();

        assert_eq!(scorer.next().unwrap(), Some(10));
        scorer.set_min_competitive_score(5.0).unwrap();
        scorer.set_min_competitive_score(5.0).unwrap();
        assert_eq!(first_work.floors.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(second_work.floors.load(AtomicOrdering::Relaxed), 0);

        assert_eq!(scorer.advance(1_000).unwrap(), Some(1_000));
        assert_eq!(second_work.floors.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    fn row_address_merge_rejects_duplicate_source_addresses() {
        let mut scorer = RowAddressMergeScorer::try_new(vec![
            row_address_source(&[(20, 1.0)]),
            row_address_source(&[(20, 2.0)]),
        ])
        .unwrap();

        let error = scorer.next().unwrap_err();
        assert!(error.to_string().contains("duplicate row address 20"));
    }

    #[test]
    fn materialized_address_fallback_sorts_nonmonotonic_projection() {
        let addresses = [30, 10, 20];
        let source = materialized(&[(0, 1.0), (1, 2.0), (2, 3.0)]);
        let source = materialize_mapped_scorer(source, |doc| {
            addresses
                .get(doc as usize)
                .copied()
                .ok_or_else(|| Error::internal(format!("missing test address for document {doc}")))
        })
        .unwrap()
        .unwrap();
        let mut scorer = source.into_scorer();

        assert_eq!(scorer.next().unwrap(), Some(10));
        assert_eq!(scorer.score().unwrap(), 2.0);
        assert_eq!(scorer.next().unwrap(), Some(20));
        assert_eq!(scorer.score().unwrap(), 3.0);
        assert_eq!(scorer.next().unwrap(), Some(30));
        assert_eq!(scorer.score().unwrap(), 1.0);
        assert_eq!(scorer.next().unwrap(), None);
    }

    #[test]
    fn materialized_address_fallback_rejects_duplicate_matches() {
        let source = materialized(&[(0, 1.0), (1, 2.0)]);
        let Err(error) = materialize_mapped_scorer(source, |_| Ok(10)) else {
            panic!("duplicate projected addresses must fail");
        };

        assert!(error.to_string().contains("duplicate row_id=10"));
    }

    struct UnboundedScorer {
        inner: MaterializedScorer,
    }

    impl ComposableScorer for UnboundedScorer {
        fn doc(&self) -> Option<u64> {
            self.inner.doc()
        }

        fn next(&mut self) -> Result<Option<u64>> {
            self.inner.next()
        }

        fn advance(&mut self, target: u64) -> Result<Option<u64>> {
            self.inner.advance(target)
        }

        fn cost(&self) -> usize {
            self.inner.cost()
        }

        fn score(&mut self) -> Result<f32> {
            self.inner.score()
        }

        fn advance_shallow(&mut self, target: u64) -> Result<u64> {
            self.inner.advance_shallow(target)
        }

        fn score_bounds(&mut self, _up_to: u64) -> Result<ScoreBounds> {
            Ok(ScoreBounds::UNBOUNDED)
        }

        fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()> {
            self.inner.set_min_competitive_score(min_score)
        }

        fn scores_non_negative(&self) -> bool {
            true
        }
    }

    struct CountingScorer {
        inner: MaterializedScorer,
        cost: usize,
        advance_calls: Arc<AtomicUsize>,
    }

    impl ComposableScorer for CountingScorer {
        fn doc(&self) -> Option<u64> {
            self.inner.doc()
        }

        fn next(&mut self) -> Result<Option<u64>> {
            self.inner.next()
        }

        fn advance(&mut self, target: u64) -> Result<Option<u64>> {
            self.advance_calls.fetch_add(1, AtomicOrdering::Relaxed);
            self.inner.advance(target)
        }

        fn cost(&self) -> usize {
            self.cost
        }

        fn score(&mut self) -> Result<f32> {
            self.inner.score()
        }

        fn advance_shallow(&mut self, target: u64) -> Result<u64> {
            self.inner.advance_shallow(target)
        }

        fn score_bounds(&mut self, up_to: u64) -> Result<ScoreBounds> {
            self.inner.score_bounds(up_to)
        }

        fn global_score_upper_bound(&self) -> Option<f32> {
            self.inner.global_score_upper_bound()
        }

        fn set_min_competitive_score(&mut self, min_score: f32) -> Result<()> {
            self.inner.set_min_competitive_score(min_score)
        }

        fn matches(&mut self) -> Result<bool> {
            self.inner.matches()
        }

        fn scores_non_negative(&self) -> bool {
            self.inner.scores_non_negative()
        }
    }

    fn counting(
        values: &[(u64, f32)],
        cost: usize,
    ) -> (Box<dyn ComposableScorer>, Arc<AtomicUsize>) {
        let advance_calls = Arc::new(AtomicUsize::new(0));
        let scorer = CountingScorer {
            inner: MaterializedScorer::try_new(rows(values)).unwrap(),
            cost,
            advance_calls: advance_calls.clone(),
        };
        (Box::new(scorer), advance_calls)
    }

    #[test]
    fn collector_confirms_two_phase_matches_without_a_cost_hint() {
        let (mut scorer, approximations, confirmations) =
            two_phase(&[(1, 100.0), (2, 2.0), (3, 1.0)], vec![2, 3], None);
        let results = TopKCollector::new(2).collect(scorer.as_mut()).unwrap();
        assert_eq!(results, rows(&[(2, 2.0), (3, 1.0)]));
        assert_eq!(approximations.load(AtomicOrdering::Relaxed), 3);
        assert_eq!(confirmations.load(AtomicOrdering::Relaxed), 3);
        assert_eq!(scorer.match_cost(), None);
    }

    #[test]
    fn required_conjunction_confirms_cheapest_first_and_short_circuits() {
        let values = (0..100).map(|doc| (doc, 1.0)).collect::<Vec<_>>();
        let accepted_by_cheap = (0..100).step_by(5).collect::<Vec<_>>();

        let (expensive, expensive_approximations, expensive_confirmations) =
            two_phase(&values, (0..100).collect(), Some(10.0));
        let (cheap, cheap_approximations, cheap_confirmations) =
            two_phase(&values, accepted_by_cheap.clone(), Some(1.0));
        let mut scorer = RequiredConjunctionScorer::try_new(vec![expensive, cheap]).unwrap();
        assert_eq!(scorer.confirmation_order.as_deref(), Some(&[1, 0][..]));

        let results = TopKCollector::new(100).collect(&mut scorer).unwrap();
        let expected = accepted_by_cheap
            .iter()
            .map(|doc| (*doc, 2.0))
            .collect::<Vec<_>>();
        assert_eq!(results, rows(&expected));
        assert_eq!(cheap_confirmations.load(AtomicOrdering::Relaxed), 100);
        assert_eq!(expensive_confirmations.load(AtomicOrdering::Relaxed), 20);
        let approximations = cheap_approximations.load(AtomicOrdering::Relaxed)
            + expensive_approximations.load(AtomicOrdering::Relaxed);
        let confirmations = cheap_confirmations.load(AtomicOrdering::Relaxed)
            + expensive_confirmations.load(AtomicOrdering::Relaxed);
        assert_eq!(approximations, 200);
        assert_eq!(confirmations, 120);
        assert!(
            confirmations * 5 <= approximations * 4,
            "confirmation ordering should reduce work by at least 20%: {confirmations}/{approximations}"
        );

        let (cheap, _, _) = two_phase(&values, accepted_by_cheap, Some(1.0));
        let (expensive, _, _) = two_phase(&values, (0..100).collect(), Some(10.0));
        let scorer = RequiredConjunctionScorer::try_new(vec![cheap, expensive]).unwrap();
        assert!(scorer.confirmation_order.is_none());
    }

    #[test]
    fn required_conjunction_confirms_children_without_cost_hints() {
        let (unknown, _, unknown_confirmations) = two_phase(&[(0, 1.0)], Vec::new(), None);
        let (costed, _, costed_confirmations) = two_phase(&[(0, 1.0)], vec![0], Some(1.0));
        let mut scorer = RequiredConjunctionScorer::try_new(vec![unknown, costed]).unwrap();

        assert_eq!(scorer.confirmation_order.as_deref(), Some(&[1, 0][..]));
        assert!(
            TopKCollector::new(1)
                .collect(&mut scorer)
                .unwrap()
                .is_empty()
        );
        assert_eq!(costed_confirmations.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(unknown_confirmations.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    fn required_conjunction_rejects_invalid_match_cost() {
        let (invalid, _, _) = two_phase(&[(0, 1.0)], vec![0], Some(f32::NAN));

        let error = RequiredConjunctionScorer::try_new(vec![invalid])
            .err()
            .unwrap();
        assert!(matches!(error, Error::Internal { .. }));
        assert!(
            error
                .to_string()
                .contains("child 0 reported invalid two-phase match cost: NaN")
        );
    }

    #[test]
    fn required_conjunction_uses_all_must_scores_for_competitive_bounds() {
        let left = Box::new(
            MaterializedScorer::try_new(rows(&[(1, 3.0), (3, 1.0)]))
                .unwrap()
                .with_block_size(1),
        );
        let right = Box::new(
            MaterializedScorer::try_new(rows(&[(1, 30.0), (3, 10.0)]))
                .unwrap()
                .with_block_size(1),
        );
        let mut scorer = RequiredConjunctionScorer::try_new(vec![left, right]).unwrap();
        let competitive_score = Arc::new(CompetitiveScore::default());
        competitive_score.raise(10.0);

        let results = TopKCollector::with_competitive_score(10, competitive_score)
            .collect(&mut scorer)
            .unwrap();

        assert_eq!(results, rows(&[(1, 33.0), (3, 11.0)]));
    }

    #[test]
    fn required_conjunction_aligns_cheapest_clause_first() {
        let dense_rows = (0..=100).map(|row_id| (row_id, 1.0)).collect::<Vec<_>>();
        let (dense, dense_advance_calls) = counting(&dense_rows, 101);
        let (rare, rare_advance_calls) = counting(&[(50, 1.0)], 1);
        let mut scorer = RequiredConjunctionScorer::try_new(vec![dense, rare]).unwrap();
        assert_eq!(scorer.approximation_order.as_deref(), Some(&[1, 0][..]));

        assert_eq!(scorer.next().unwrap(), Some(50));
        assert_eq!(scorer.next().unwrap(), None);
        assert_eq!(dense_advance_calls.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(rare_advance_calls.load(AtomicOrdering::Relaxed), 2);

        let (rare, _) = counting(&[(50, 1.0)], 1);
        let (dense, _) = counting(&dense_rows, 101);
        let scorer = RequiredConjunctionScorer::try_new(vec![rare, dense]).unwrap();
        assert!(scorer.approximation_order.is_none());
    }

    #[test]
    fn required_conjunction_preserves_query_score_order() {
        let (large, _) = counting(&[(0, 16_777_216.0)], 3);
        let (first_small, _) = counting(&[(0, 1.0)], 1);
        let (second_small, _) = counting(&[(0, 1.0)], 2);
        let mut scorer =
            RequiredConjunctionScorer::try_new(vec![large, first_small, second_small]).unwrap();

        assert_eq!(scorer.next().unwrap(), Some(0));
        assert_eq!(scorer.score().unwrap(), 16_777_216.0);
    }

    #[test]
    fn boolean_sums_all_matching_clause_scores() {
        let must = vec![
            materialized(&[(1, 3.0), (2, 2.0), (3, 1.0)]),
            materialized(&[(1, 30.0), (3, 10.0)]),
        ];
        let should = vec![
            materialized(&[(1, 0.5), (3, 4.0)]),
            materialized(&[(3, 2.0)]),
        ];
        let must_not = vec![materialized(&[(1, 9.0)])];
        let mut boolean = BooleanScorer::try_new(should, must, must_not).unwrap();
        let results = TopKCollector::new(10).collect(&mut boolean).unwrap();
        assert_eq!(
            results,
            vec![ScoredRow {
                row_id: 3,
                score: 17.0
            }]
        );

        let mut dismax = DisjunctionScorer::try_new(
            vec![
                materialized(&[(1, 2.0), (3, 3.0)]),
                materialized(&[(1, 4.0), (2, 4.0)]),
            ],
            DisjunctionScore::Max,
        )
        .unwrap();
        let results = TopKCollector::new(2).collect(&mut dismax).unwrap();
        assert_eq!(results, rows(&[(1, 4.0), (2, 4.0)]));
    }

    #[test]
    fn boolean_preserves_zero_weight_required_and_prohibited_membership() {
        let mut token_docs = HashMap::new();
        token_docs.insert("common".to_owned(), 10_000_000);
        let scorer = Arc::new(MemBM25Scorer::new(10_000_000, 10_000_000, token_docs));
        assert_eq!(scorer.query_weight("common"), 0.0);

        let mut documents = DocSet::default();
        documents.append(0, 1);
        let params = FtsSearchParams::default();
        let metrics = NoOpMetricsCollector;

        let mut required = BooleanScorer::try_new(
            Vec::new(),
            vec![zero_weight_wand(
                &documents,
                scorer.clone(),
                &params,
                &metrics,
            )],
            Vec::new(),
        )
        .unwrap();
        assert_eq!(required.next().unwrap(), Some(0));
        assert!(required.matches().unwrap());
        assert_eq!(required.score().unwrap(), 0.0);
        drop(required);

        let mut excluded = BooleanScorer::try_new(
            Vec::new(),
            vec![materialized(&[(0, 1.0)])],
            vec![zero_weight_wand(&documents, scorer, &params, &metrics)],
        )
        .unwrap();
        assert_eq!(excluded.next().unwrap(), None);
    }

    #[test]
    fn phrase_doc_bound_records_exact_and_sloppy_confirmation_avoidance() {
        let mut token_docs = HashMap::new();
        token_docs.insert("common".to_owned(), 10_000_000);
        let scorer = Arc::new(MemBM25Scorer::new(10_000_000, 10_000_000, token_docs));
        let mut documents = DocSet::default();
        documents.append(0, 1);

        for slop in [0, 2] {
            let params = FtsSearchParams::default().with_phrase_slop(Some(slop));
            let metrics = NoOpMetricsCollector;
            let phrase = zero_weight_wand(&documents, scorer.clone(), &params, &metrics);
            // The high-scoring sibling keeps the shared block competitive, while
            // doc 0 still has a combined doc-local upper below the score floor.
            let mut phrase = DisjunctionScorer::try_new(
                vec![phrase, materialized(&[(0, 0.0), (1, 2.0)])],
                DisjunctionScore::Sum,
            )
            .unwrap();
            let competitive_score = Arc::new(CompetitiveScore::default());
            competitive_score.raise(1.0);

            assert_eq!(
                TopKCollector::with_competitive_score(1, competitive_score)
                    .collect(&mut phrase)
                    .unwrap(),
                rows(&[(1, 2.0)])
            );
        }
    }

    #[test]
    fn phrase_positive_boost_uses_positive_upper_before_confirmation() {
        let (phrase, _, confirmations) = two_phase(&[(0, 2.0), (1, 10.0)], vec![1], Some(1.0));
        let mut scorer = BoostScorer::try_new(phrase, materialized(&[(1, 1.0)]), 0.5).unwrap();
        let competitive_score = Arc::new(CompetitiveScore::default());
        competitive_score.raise(9.5);

        assert_eq!(
            TopKCollector::with_competitive_score(1, competitive_score)
                .collect(&mut scorer)
                .unwrap(),
            rows(&[(1, 9.5)])
        );
        assert_eq!(confirmations.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    fn phrase_must_not_never_contributes_to_score_upper_bound() {
        let (prohibited, approximations, confirmations) =
            two_phase(&[(0, 100.0)], Vec::new(), Some(1.0));
        let mut scorer = BooleanScorer::try_new(
            Vec::new(),
            vec![materialized(&[(0, 1.0), (1, 10.0)])],
            vec![prohibited],
        )
        .unwrap();
        let competitive_score = Arc::new(CompetitiveScore::default());
        competitive_score.raise(10.0);

        assert_eq!(
            TopKCollector::with_competitive_score(1, competitive_score)
                .collect(&mut scorer)
                .unwrap(),
            rows(&[(1, 10.0)])
        );
        // Doc 0 is rejected from the positive score alone. The prohibited
        // phrase is never advanced, so it cannot inflate that upper bound.
        assert_eq!(approximations.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(confirmations.load(AtomicOrdering::Relaxed), 0);
    }

    #[test]
    fn signed_and_unknown_doc_bounds_use_exact_confirmation_fallback() {
        let (signed_phrase, _, signed_confirmations) = two_phase(&[(0, 2.0)], vec![0], Some(1.0));
        let signed_optional =
            Box::new(BoostScorer::try_new(signed_phrase, materialized(&[(0, 2.0)]), 2.0).unwrap());
        let mut signed = BooleanScorer::try_new(
            vec![signed_optional],
            vec![materialized(&[(0, 1.0)])],
            Vec::new(),
        )
        .unwrap();
        let competitive_score = Arc::new(CompetitiveScore::default());
        competitive_score.raise(10.0);

        assert!(
            TopKCollector::with_competitive_score(1, competitive_score)
                .collect(&mut signed)
                .unwrap()
                .is_empty()
        );
        assert_eq!(signed_confirmations.load(AtomicOrdering::Relaxed), 1);
        let (unknown_phrase, approximations, confirmations) =
            two_phase_without_doc_upper(&[(0, 1.0), (1, 100.0)], vec![1]);
        let mut unknown = BooleanScorer::try_new(
            vec![unknown_phrase, materialized(&[(0, 0.0), (1, 0.0)])],
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        let competitive_score = Arc::new(CompetitiveScore::default());
        competitive_score.raise(50.0);

        assert_eq!(
            TopKCollector::with_competitive_score(1, competitive_score)
                .collect(&mut unknown)
                .unwrap(),
            rows(&[(1, 100.0)])
        );
        assert_eq!(approximations.load(AtomicOrdering::Relaxed), 2);
        assert_eq!(confirmations.load(AtomicOrdering::Relaxed), 2);
    }

    #[test]
    fn row_address_merge_forwards_phrase_doc_bound_and_avoidance() {
        // Cross-column eager execution maps each leaf into row-address space
        // and merges disjoint sources through this same wrapper stack.
        let (phrase, approximations, confirmations) =
            two_phase(&[(0, 2.0), (1, 10.0)], vec![1], Some(1.0));
        let mapped_phrase = Box::new(RowAddressScorer::new(
            phrase,
            ordered_row_address_projection_for_test(vec![100, 200]),
        ));
        let mut scorer = RowAddressMergeScorer::try_new(vec![
            RowAddressSource::new(100, mapped_phrase),
            row_address_source(&[(300, 100.0)]),
        ])
        .unwrap();
        let competitive_score = Arc::new(CompetitiveScore::default());
        competitive_score.raise(10.0);

        assert_eq!(
            TopKCollector::with_competitive_score(2, competitive_score)
                .collect(&mut scorer)
                .unwrap(),
            rows(&[(300, 100.0), (200, 10.0)])
        );
        assert_eq!(approximations.load(AtomicOrdering::Relaxed), 2);
        assert_eq!(confirmations.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    fn phrase_outward_upper_equal_to_ulp_floor_still_confirms() {
        let exact_score = next_down(1.0);
        let floor = 1.0;
        assert_eq!(
            ScoreBounds::ZERO
                .add(ScoreBounds::point(exact_score).unwrap())
                .upper(),
            floor
        );
        let (phrase, _, confirmations) = two_phase(&[(7, exact_score)], vec![7], Some(1.0));
        let mut phrase = DisjunctionScorer::try_new(vec![phrase], DisjunctionScore::Sum).unwrap();
        let competitive_score = Arc::new(CompetitiveScore::default());
        competitive_score.raise(floor);

        assert!(
            TopKCollector::with_competitive_score(1, competitive_score)
                .collect(&mut phrase)
                .unwrap()
                .is_empty()
        );
        assert_eq!(confirmations.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    fn surviving_nested_phrase_is_confirmed_exactly_once() {
        let (phrase, _, confirmations) = two_phase(&[(3, 5.0)], vec![3], Some(1.0));
        let nested = Box::new(
            DisjunctionScorer::try_new(
                vec![phrase, materialized(&[(3, 0.0)])],
                DisjunctionScore::Sum,
            )
            .unwrap(),
        );
        let mut scorer = BooleanScorer::try_new(vec![nested], Vec::new(), Vec::new()).unwrap();

        assert_eq!(
            TopKCollector::new(1).collect(&mut scorer).unwrap(),
            rows(&[(3, 5.0)])
        );
        assert_eq!(confirmations.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    fn reqopt_delays_sparse_optional_probes() {
        let values = (0..100).map(|doc| (doc, 1.0)).collect::<Vec<_>>();
        let build = || {
            let required =
                Box::new(RequiredConjunctionScorer::try_new(vec![materialized(&values)]).unwrap());
            let optional = Box::new(
                DisjunctionScorer::try_new(
                    vec![materialized(&[(99, 100.0)])],
                    DisjunctionScore::Sum,
                )
                .unwrap(),
            );
            let (required, required_work) = instrumented(required);
            let (optional, optional_work) = instrumented(optional);
            (required, required_work, optional, optional_work)
        };

        let (required, _, optional, eager_optional_work) = build();
        let mut eager = BooleanScorer {
            driver: required,
            optional: Some(optional),
            prohibited: None,
            current: None,
            confirmed_doc: None,
            confirmed: false,
            optional_matches: false,
            defer_confirmation: false,
        };
        let eager_results = TopKCollector::new(1).collect(&mut eager).unwrap();

        let (required, required_work, optional, optional_work) = build();
        let mut scorer = ReqOptScorer::new(required, optional);
        let results = TopKCollector::new(1).collect(&mut scorer).unwrap();

        assert_eq!(eager_results, rows(&[(99, 101.0)]));
        assert_eq!(results, rows(&[(99, 101.0)]));
        let required_advances = required_work.advances.load(AtomicOrdering::Relaxed);
        let eager_optional_probes = eager_optional_work.advances.load(AtomicOrdering::Relaxed);
        let optional_probes = optional_work.advances.load(AtomicOrdering::Relaxed);
        assert_eq!(required_advances, 100);
        assert_eq!(eager_optional_probes, 100);
        assert_eq!(optional_probes, 1);
        assert_eq!(
            required_work.shallow_advances.load(AtomicOrdering::Relaxed),
            2
        );
        assert_eq!(required_work.bounds.load(AtomicOrdering::Relaxed), 2);
        assert_eq!(
            required_work.confirmations.load(AtomicOrdering::Relaxed),
            100
        );
        assert_eq!(optional_work.confirmations.load(AtomicOrdering::Relaxed), 1);
        assert!(
            optional_probes * 5 <= eager_optional_probes * 4,
            "lazy required-plus-optional scoring should reduce optional probes by at least 20%: \
             {optional_probes}/{eager_optional_probes}"
        );
    }

    #[test]
    fn empty_window_does_not_stop_collection() {
        // A later optional max of 3.0 keeps the sticky MUST floor loose, so
        // next() still lands in the first block. That block's own optional
        // upper is only 0.15, so the window skip must fire and must not be
        // treated as end-of-stream.
        let mut required_values = (0..32).map(|doc| (doc, 1.5)).collect::<Vec<_>>();
        required_values.push((32, 4.0));
        let mut optional_values = (0..32).map(|doc| (doc, 0.15)).collect::<Vec<_>>();
        optional_values.push((32, 1.0));
        optional_values.push((200, 3.0));
        let required = Box::new(
            MaterializedScorer::try_new(rows(&required_values))
                .unwrap()
                .with_block_size(32),
        );
        let optional = Box::new(
            MaterializedScorer::try_new(rows(&optional_values))
                .unwrap()
                .with_block_size(32),
        );
        let mut scorer = ReqOptScorer::new(required, optional);

        let mut first_window = Vec::new();
        let first = scorer
            .collect_confirmed_window(4.5, &mut first_window, 8)
            .unwrap();
        assert!(
            first_window.is_empty(),
            "non-competitive first window should emit nothing: {first_window:?}"
        );
        assert!(
            !first.exhausted,
            "skipping a non-competitive window must not look like end-of-stream"
        );

        let mut second_window = Vec::new();
        let second = scorer
            .collect_confirmed_window(4.5, &mut second_window, 8)
            .unwrap();
        assert_eq!(
            second_window
                .iter()
                .map(|hit| (hit.document_key, hit.score))
                .collect::<Vec<_>>(),
            vec![(32, 5.0)]
        );
        assert!(second.exhausted || second_window.len() == 1);
    }

    #[test]
    fn reqopt_temporarily_requires_optional_contribution() {
        let values = (0..100).map(|doc| (doc, 1.0)).collect::<Vec<_>>();
        let (required, required_work) = instrumented(materialized(&values));
        let (optional, optional_work) = instrumented(materialized(&[(50, 10.0)]));
        let mut scorer = ReqOptScorer::new(required, optional);
        let competitive_score = Arc::new(CompetitiveScore::default());
        competitive_score.raise(10.0);

        let results = TopKCollector::with_competitive_score(1, competitive_score)
            .collect(&mut scorer)
            .unwrap();

        assert_eq!(results, rows(&[(50, 11.0)]));
        assert!(required_work.advances.load(AtomicOrdering::Relaxed) < 10);
        assert_eq!(required_work.confirmations.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(optional_work.advances.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(optional_work.confirmations.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    fn reqopt_promotes_optional_once_required_cannot_reach_the_heap() {
        // The required clause tops out at 4.5 while doc 40 reaches 5.0 only
        // through the optional clause. Once the heap floor passes 4.5 the
        // required clause alone is hopeless, so the scan switches to a direct
        // intersection and must still surface doc 40 -- a document in the
        // sparse tail of the required list.
        let mut required_values = (0..32).map(|doc| (doc, 4.5)).collect::<Vec<_>>();
        required_values.extend((32..64).map(|doc| (doc, 2.0)));
        let mut optional_values = (0..32).map(|doc| (doc, 0.15)).collect::<Vec<_>>();
        optional_values.push((40, 3.0));
        let required = Box::new(
            MaterializedScorer::try_new(rows(&required_values))
                .unwrap()
                .with_block_size(32),
        );
        let optional = Box::new(
            MaterializedScorer::try_new(rows(&optional_values))
                .unwrap()
                .with_block_size(32),
        );
        let mut scorer = ReqOptScorer::new(required, optional);
        assert!(!scorer.optional_globally_required);

        let results = TopKCollector::new(3).collect(&mut scorer).unwrap();

        assert_eq!(results, rows(&[(40, 5.0), (0, 4.65), (1, 4.65)]));
        assert!(
            scorer.optional_globally_required,
            "a heap floor above the required clause's global upper must promote \
             the optional clause"
        );
    }

    #[test]
    fn boolean_reqopt_keeps_current_confirmation_stable_after_bounds() {
        let mut scorer = BooleanScorer::try_new(
            vec![materialized(&[(1, 10.0)])],
            vec![materialized(&[(0, 1.0), (1, 1.0)])],
            Vec::new(),
        )
        .unwrap();
        let competitive_score = Arc::new(CompetitiveScore::default());
        competitive_score.raise(5.0);

        let results = TopKCollector::with_competitive_score(1, competitive_score)
            .collect(&mut scorer)
            .unwrap();

        assert_eq!(results, rows(&[(1, 11.0)]));
    }

    #[test]
    fn boolean_gates_reqopt_and_signed_boost_uses_exact_fallback() {
        let supported = BooleanScorer::try_new(
            vec![materialized(&[(1, 4.0)])],
            vec![materialized(&[(1, 2.0)])],
            Vec::new(),
        )
        .unwrap();
        assert!(supported.optional.is_none());

        let signed_optional = Box::new(
            BoostScorer::try_new(
                materialized(&[(1, 4.0), (2, 1.0)]),
                materialized(&[(1, 1.0), (2, 4.0)]),
                0.5,
            )
            .unwrap(),
        );
        let mut fallback = BooleanScorer::try_new(
            vec![signed_optional],
            vec![materialized(&[(1, 2.0), (2, 2.0)])],
            Vec::new(),
        )
        .unwrap();
        assert!(fallback.optional.is_some());
        assert_eq!(
            TopKCollector::new(2).collect(&mut fallback).unwrap(),
            rows(&[(1, 5.5), (2, 1.0)])
        );
    }

    #[test]
    fn reqopt_uses_exact_iteration_for_unbounded_scorers() {
        let required = Box::new(UnboundedScorer {
            inner: MaterializedScorer::try_new(rows(&[(1, 1.0), (2, 1.0)])).unwrap(),
        });
        let optional = materialized(&[(2, 10.0)]);
        let mut scorer = ReqOptScorer::new(required, optional);
        let competitive_score = Arc::new(CompetitiveScore::default());
        competitive_score.raise(5.0);

        let results = TopKCollector::with_competitive_score(1, competitive_score)
            .collect(&mut scorer)
            .unwrap();

        assert_eq!(results, rows(&[(2, 11.0)]));
    }

    fn pure_should_canary_children() -> (Vec<BoxScorer<'static>>, Vec<Arc<ScorerWork>>) {
        let mut children = Vec::new();
        let mut work = Vec::new();
        let mut push = |child: BoxScorer<'static>| {
            let (child, child_work) = instrumented(child);
            children.push(child);
            work.push(child_work);
        };

        push(materialized(&[(0, 2.0)]));
        let dense = (1..=1024).map(|doc| (doc, 0.125)).collect::<Vec<_>>();
        for _ in 0..8 {
            push(materialized(&dense));
        }
        let sparse = (127..=1023)
            .step_by(128)
            .map(|doc| (doc, 1.5))
            .collect::<Vec<_>>();
        push(materialized(&sparse));
        (children, work)
    }

    fn scorer_advances(work: &[Arc<ScorerWork>]) -> usize {
        work.iter()
            .map(|work| work.advances.load(AtomicOrdering::Relaxed))
            .sum()
    }

    fn exhaustive_should_top_k(children: &[Vec<(u64, f32)>], limit: usize) -> Vec<ScoredRow> {
        let mut scores = HashMap::<u64, f32>::new();
        for child in children {
            for (doc, score) in child {
                *scores.entry(*doc).or_default() += *score;
            }
        }
        let mut rows = scores
            .into_iter()
            .map(|(doc, score)| ScoredRow::new(doc, score).unwrap())
            .collect::<Vec<_>>();
        rows.sort_unstable_by(compare_scored_rows);
        rows.truncate(limit);
        rows
    }

    fn exhaustive_compound_scores(
        plan: &CompoundScorerPlan,
        leaves: &[HashMap<u64, f32>],
    ) -> HashMap<u64, f32> {
        match plan {
            CompoundScorerPlan::Leaf { index, boost } => leaves[*index]
                .iter()
                .map(|(row_address, score)| (*row_address, *score * *boost))
                .collect(),
            CompoundScorerPlan::Boost {
                positive,
                negative,
                negative_boost,
            } => {
                let mut positive = exhaustive_compound_scores(positive, leaves);
                let negative = exhaustive_compound_scores(negative, leaves);
                for (row_address, score) in &mut positive {
                    if let Some(negative_score) = negative.get(row_address) {
                        *score -= *negative_boost * *negative_score;
                    }
                }
                positive
            }
            CompoundScorerPlan::MultiMatch(children) => {
                let mut scores = HashMap::<u64, f32>::new();
                for child in children {
                    for (row_address, score) in exhaustive_compound_scores(child, leaves) {
                        scores
                            .entry(row_address)
                            .and_modify(|current| *current = current.max(score))
                            .or_insert(score);
                    }
                }
                scores
            }
            CompoundScorerPlan::Boolean {
                should,
                must,
                must_not,
            } => {
                let mut scores = if let Some((first, remaining)) = must.split_first() {
                    let mut scores = exhaustive_compound_scores(first, leaves);
                    for child in remaining {
                        let required = exhaustive_compound_scores(child, leaves);
                        scores.retain(|row_address, score| {
                            if let Some(required_score) = required.get(row_address) {
                                *score += required_score;
                                true
                            } else {
                                false
                            }
                        });
                    }
                    for child in should {
                        for (row_address, optional_score) in
                            exhaustive_compound_scores(child, leaves)
                        {
                            if let Some(score) = scores.get_mut(&row_address) {
                                *score += optional_score;
                            }
                        }
                    }
                    scores
                } else {
                    let mut scores = HashMap::<u64, f32>::new();
                    for child in should {
                        for (row_address, score) in exhaustive_compound_scores(child, leaves) {
                            *scores.entry(row_address).or_default() += score;
                        }
                    }
                    scores
                };

                for child in must_not {
                    for row_address in exhaustive_compound_scores(child, leaves).into_keys() {
                        scores.remove(&row_address);
                    }
                }
                scores
            }
        }
    }

    fn exhaustive_compound_top_k(
        plan: &CompoundScorerPlan,
        leaves: &[HashMap<u64, f32>],
        limit: usize,
    ) -> Vec<ScoredRow> {
        let mut scores = exhaustive_compound_scores(plan, leaves)
            .into_iter()
            .collect::<Vec<_>>();
        scores.sort_unstable_by(|(left_row, left_score), (right_row, right_score)| {
            right_score
                .total_cmp(left_score)
                .then_with(|| left_row.cmp(right_row))
        });
        scores.truncate(limit);
        scores
            .into_iter()
            .map(|(row_address, score)| ScoredRow::new(row_address, score).unwrap())
            .collect()
    }

    fn randomized_mapped_leaf(
        scores: &HashMap<u64, f32>,
        canonical_row_addresses: &[u64],
        rng: &mut SmallRng,
    ) -> BoxScorer<'static> {
        let source_count = rng.random_range(2..=3);
        let source_by_row = (0..256)
            .find_map(|_| {
                let source_by_row = (0..canonical_row_addresses.len())
                    .map(|_| rng.random_range(0..source_count))
                    .collect::<Vec<_>>();
                let mut projection_lengths = vec![0_u64; source_count];
                let mut local_matches = vec![Vec::<u64>::new(); source_count];
                for (row_index, row_address) in canonical_row_addresses.iter().enumerate() {
                    let source_index = source_by_row[row_index];
                    let local_doc = projection_lengths[source_index];
                    projection_lengths[source_index] += 1;
                    if scores.contains_key(row_address) {
                        local_matches[source_index].push(local_doc);
                    }
                }

                let local_gap_patterns = local_matches
                    .iter()
                    .map(|matches| {
                        matches
                            .windows(2)
                            .map(|pair| pair[1] - pair[0])
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let has_distinct_gapped_sources =
                    local_matches
                        .iter()
                        .enumerate()
                        .all(|(source_index, matches)| {
                            matches.len() >= 2
                                && matches.windows(2).any(|pair| pair[1] > pair[0] + 1)
                                && matches.len() * 4 > projection_lengths[source_index] as usize
                        })
                        && local_gap_patterns
                            .iter()
                            .enumerate()
                            .all(|(source_index, gaps)| {
                                local_gap_patterns[..source_index]
                                    .iter()
                                    .all(|previous| previous != gaps)
                            });
                has_distinct_gapped_sources.then_some(source_by_row)
            })
            .expect("randomized physical sources should have distinct local-document gaps");

        let mut sources = Vec::with_capacity(source_count);
        for source_index in 0..source_count {
            let projection_addresses = canonical_row_addresses
                .iter()
                .enumerate()
                .filter_map(|(row_index, row_address)| {
                    (source_by_row[row_index] == source_index).then_some(*row_address)
                })
                .collect::<Vec<_>>();
            let local_rows = projection_addresses
                .iter()
                .enumerate()
                .filter_map(|(local_doc, row_address)| {
                    scores
                        .get(row_address)
                        .map(|score| (local_doc as u64, *score))
                })
                .collect::<Vec<_>>();
            let first_match = local_rows
                .first()
                .expect("every randomized physical source should have postings")
                .0;
            let local_document_lower_bound = rng.random_range(0..=first_match);
            let projection = resident_row_address_projection_for_test(projection_addresses.clone());
            let prepared = prepare_row_address_projection(&projection);
            let source = map_scorer_to_row_addresses(
                materialized(&local_rows),
                &prepared,
                local_document_lower_bound,
            )
            .unwrap()
            .expect("a randomized physical source with postings should map");
            assert_eq!(
                projection.cached_row_address_order(),
                CachedRowAddressOrder::Ordered
            );
            assert_eq!(
                projection.ordered_validation_visited_docs(),
                projection_addresses.len()
            );
            sources.push(source);
        }

        Box::new(RowAddressMergeScorer::try_new(sources).unwrap())
    }

    fn plan_leaf(index: usize) -> CompoundScorerPlan {
        CompoundScorerPlan::Leaf { index, boost: 1.0 }
    }

    #[test]
    fn iu_tight_leaf_indices_matches_flat_must_should_terms() {
        let plan = CompoundScorerPlan::Boolean {
            should: vec![plan_leaf(0), plan_leaf(1)],
            must: vec![plan_leaf(2)],
            must_not: Vec::new(),
        };
        assert_eq!(iu_tight_clause_indices(&plan), Some((vec![2], vec![0, 1])));

        let boosted = CompoundScorerPlan::Boolean {
            should: vec![CompoundScorerPlan::Leaf {
                index: 0,
                boost: 2.0,
            }],
            must: vec![plan_leaf(1)],
            must_not: Vec::new(),
        };
        assert_eq!(iu_tight_clause_indices(&boosted), None);

        let nested = CompoundScorerPlan::Boolean {
            should: vec![CompoundScorerPlan::Boolean {
                should: vec![plan_leaf(0)],
                must: Vec::new(),
                must_not: Vec::new(),
            }],
            must: vec![plan_leaf(1)],
            must_not: Vec::new(),
        };
        assert_eq!(iu_tight_clause_indices(&nested), None);
    }

    #[test]
    fn phrase_doc_bounds_match_exhaustive_oracle_across_boolean_shapes() {
        // Leaf 0 models an exact phrase and leaf 1 a sloppy phrase. Their
        // approximation scores include false position candidates, including a
        // high-scoring false candidate at doc 4. Docs 1 and 3 deliberately tie.
        let exact_approximations = [(0, 2.0), (1, 4.0), (2, 3.0), (3, 4.0), (4, 100.0)];
        let sloppy_approximations = [(0, 1.0), (1, 2.0), (2, 3.0), (3, 2.0), (4, 1.0)];
        let exact_matches = HashMap::from([(1, 4.0), (3, 4.0)]);
        let sloppy_matches = HashMap::from([(1, 2.0), (2, 3.0), (3, 2.0)]);
        let optional = HashMap::from([(0, 1.0), (1, 4.0), (2, 2.0), (3, 4.0), (4, 1.0)]);
        let required = HashMap::from([(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0)]);
        let oracle_leaves = vec![exact_matches, sloppy_matches, optional.clone(), required];

        let pure_should = CompoundScorerPlan::Boolean {
            should: vec![plan_leaf(0), plan_leaf(1), plan_leaf(2)],
            must: Vec::new(),
            must_not: Vec::new(),
        };
        let must_should = CompoundScorerPlan::Boolean {
            should: vec![plan_leaf(0), plan_leaf(1), plan_leaf(2)],
            must: vec![plan_leaf(3)],
            must_not: Vec::new(),
        };
        let nested = CompoundScorerPlan::Boolean {
            should: vec![pure_should.clone()],
            must: vec![plan_leaf(3)],
            must_not: Vec::new(),
        };

        for (shape, plan, floor) in [
            ("should", pure_should, 10.0),
            ("must_should", must_should, 11.0),
            ("nested", nested, 11.0),
        ] {
            let (exact, _, exact_confirmations) =
                two_phase(&exact_approximations, vec![1, 3], Some(1.0));
            let (sloppy, _, sloppy_confirmations) =
                two_phase(&sloppy_approximations, vec![1, 2, 3], Some(2.0));
            let mut leaves = vec![
                Some(exact),
                Some(sloppy),
                Some(materialized(
                    &optional
                        .iter()
                        .map(|(doc, score)| (*doc, *score))
                        .collect::<Vec<_>>(),
                )),
                Some(materialized(&[
                    (0, 1.0),
                    (1, 1.0),
                    (2, 1.0),
                    (3, 1.0),
                    (4, 1.0),
                ])),
            ];
            let mut scorer = plan.build(&mut leaves).unwrap();
            let competitive_score = Arc::new(CompetitiveScore::default());
            competitive_score.raise(floor);
            let actual = TopKCollector::with_competitive_score(2, competitive_score)
                .collect(scorer.as_mut())
                .unwrap();
            let expected = exhaustive_compound_top_k(&plan, &oracle_leaves, 2);

            assert_eq!(actual, expected, "shape={shape}");
            assert_eq!(actual, rows(&[(1, floor), (3, floor)]), "shape={shape}");
            assert!(
                exact_confirmations.load(AtomicOrdering::Relaxed) > 0
                    && sloppy_confirmations.load(AtomicOrdering::Relaxed) > 0,
                "shape={shape} must confirm surviving equal-floor phrase candidates"
            );
        }
    }

    fn plan_input(possible: bool, cost: usize, lower: f32, upper: f32) -> CompoundLeafPlanInput {
        CompoundLeafPlanInput::new(possible, cost, ScoreBounds::try_new(lower, upper).unwrap())
    }

    #[test]
    fn staged_plan_analysis_selects_stable_must_generator() {
        let plan = CompoundScorerPlan::Boolean {
            should: vec![plan_leaf(0)],
            must: vec![
                CompoundScorerPlan::MultiMatch(vec![plan_leaf(1), plan_leaf(2)]),
                CompoundScorerPlan::Boost {
                    positive: Box::new(plan_leaf(3)),
                    negative: Box::new(plan_leaf(4)),
                    negative_boost: 0.5,
                },
            ],
            must_not: vec![plan_leaf(5)],
        };
        let analysis = plan
            .analyze_leaves(&[
                plan_input(true, 9, 1.0, 2.0),
                plan_input(true, 2, 2.0, 4.0),
                plan_input(true, 2, -1.0, 3.0),
                plan_input(true, 4, 5.0, 6.0),
                plan_input(true, 1, 1.0, 2.0),
                plan_input(true, 1, 0.0, 10.0),
            ])
            .unwrap();

        assert_eq!(plan.leaf_count(), 6);
        assert!(analysis.possible);
        assert_eq!(analysis.generator_cost, 4);
        // Equal-cost MUST covers retain query order, then expose a canonical
        // sorted/deduplicated leaf list to the I/O scheduler.
        assert_eq!(analysis.generator_leaves, vec![1, 2]);
        assert!(analysis.bounds.lower() <= 3.0);
        assert!(analysis.bounds.upper() >= 12.0);
    }

    #[test]
    fn staged_plan_analysis_handles_optional_missing_and_impossible_required() {
        let pure_should = CompoundScorerPlan::Boolean {
            // Deliberately use non-canonical traversal order so the public
            // generator list must sort independently of query-tree layout.
            should: vec![plan_leaf(2), plan_leaf(0), plan_leaf(1)],
            must: Vec::new(),
            must_not: Vec::new(),
        };
        let analysis = pure_should
            .analyze_leaves(&[
                plan_input(true, 7, 1.0, 2.0),
                plan_input(false, 1, 100.0, 200.0),
                plan_input(true, 3, -4.0, -1.0),
            ])
            .unwrap();
        assert!(analysis.possible);
        assert_eq!(analysis.generator_cost, 10);
        assert_eq!(analysis.generator_leaves, vec![0, 2]);
        assert!(analysis.bounds.lower() <= -4.0);
        assert!(analysis.bounds.upper() >= 2.0);

        let missing_must = CompoundScorerPlan::Boolean {
            should: vec![plan_leaf(0)],
            must: vec![plan_leaf(1)],
            must_not: Vec::new(),
        };
        let analysis = missing_must
            .analyze_leaves(&[
                plan_input(true, 1, 0.0, 10.0),
                plan_input(false, 1, 0.0, 10.0),
            ])
            .unwrap();
        assert!(!analysis.possible);
        assert_eq!(analysis.bounds, ScoreBounds::ZERO);
        assert!(analysis.generator_leaves.is_empty());

        let only_must_not = CompoundScorerPlan::Boolean {
            should: Vec::new(),
            must: Vec::new(),
            must_not: vec![plan_leaf(0)],
        };
        let analysis = only_must_not
            .analyze_leaves(&[plan_input(true, 1, 0.0, 10.0)])
            .unwrap();
        assert!(!analysis.possible);
        assert!(analysis.generator_leaves.is_empty());
    }

    #[test]
    fn staged_plan_analysis_composes_signed_nested_boost_and_unbounded_inputs() {
        let plan = CompoundScorerPlan::Boost {
            positive: Box::new(plan_leaf(0)),
            negative: Box::new(CompoundScorerPlan::Boost {
                positive: Box::new(plan_leaf(1)),
                negative: Box::new(plan_leaf(2)),
                negative_boost: 1.0,
            }),
            negative_boost: 0.5,
        };
        let analysis = plan
            .analyze_leaves(&[
                plan_input(true, 3, 2.0, 3.0),
                plan_input(true, 1, 1.0, 2.0),
                plan_input(true, 1, 4.0, 5.0),
            ])
            .unwrap();
        assert_eq!(analysis.generator_leaves, vec![0]);
        // The nested negative can itself be negative, so subtracting it may
        // increase the outer Boost score.
        assert!(analysis.bounds.lower() <= 1.0);
        assert!(analysis.bounds.upper() >= 5.0);

        let unbounded = CompoundScorerPlan::Boolean {
            should: vec![plan_leaf(0), plan_leaf(1)],
            must: Vec::new(),
            must_not: Vec::new(),
        }
        .analyze_leaves(&[
            CompoundLeafPlanInput::new(true, 1, ScoreBounds::UNBOUNDED),
            plan_input(true, 1, 0.0, 1.0),
        ])
        .unwrap();
        assert_eq!(unbounded.bounds, ScoreBounds::UNBOUNDED);

        assert!(ScoreBounds::try_new(f32::NAN, 1.0).is_err());
        let invalid = [CompoundLeafPlanInput {
            possible: true,
            cost: 1,
            bounds: ScoreBounds {
                lower: 2.0,
                upper: 1.0,
            },
        }];
        assert!(plan_leaf(0).analyze_leaves(&invalid).is_err());
    }

    fn random_staged_plan(
        rng: &mut SmallRng,
        depth: usize,
        next_leaf: &mut usize,
    ) -> CompoundScorerPlan {
        if depth == 0 || rng.random_bool(0.35) {
            let index = *next_leaf;
            *next_leaf += 1;
            return CompoundScorerPlan::Leaf {
                index,
                boost: [0.0, 0.5, 1.0, 2.0][rng.random_range(0..4)],
            };
        }
        match rng.random_range(0..3) {
            0 => CompoundScorerPlan::Boost {
                positive: Box::new(random_staged_plan(rng, depth - 1, next_leaf)),
                negative: Box::new(random_staged_plan(rng, depth - 1, next_leaf)),
                negative_boost: [0.0, 0.5, 1.0, 1.5][rng.random_range(0..4)],
            },
            1 => CompoundScorerPlan::MultiMatch(
                (0..rng.random_range(1..=3))
                    .map(|_| random_staged_plan(rng, depth - 1, next_leaf))
                    .collect(),
            ),
            _ => {
                let must_count = rng.random_range(0..=2);
                let should_count = if must_count == 0 {
                    rng.random_range(1..=3)
                } else {
                    rng.random_range(0..=2)
                };
                let should = (0..should_count)
                    .map(|_| random_staged_plan(rng, depth - 1, next_leaf))
                    .collect();
                let must = (0..must_count)
                    .map(|_| random_staged_plan(rng, depth - 1, next_leaf))
                    .collect();
                let must_not = (0..rng.random_range(0..=2))
                    .map(|_| random_staged_plan(rng, depth - 1, next_leaf))
                    .collect();
                CompoundScorerPlan::Boolean {
                    should,
                    must,
                    must_not,
                }
            }
        }
    }

    #[test]
    fn randomized_staged_plan_bounds_and_generator_cover_exact_matches() {
        for seed in 0..64 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let mut leaf_count = 0;
            let plan = random_staged_plan(&mut rng, 3, &mut leaf_count);
            assert_eq!(plan.leaf_count(), leaf_count, "seed={seed}");
            let inputs = (0..leaf_count)
                .map(|_| {
                    let possible = rng.random_bool(0.8);
                    let lower = rng.random_range(-4..=2) as f32;
                    let upper = rng.random_range(lower as i32..=5) as f32;
                    plan_input(possible, rng.random_range(1..=32), lower, upper)
                })
                .collect::<Vec<_>>();
            let analysis = plan.analyze_leaves(&inputs).unwrap();
            assert!(
                analysis
                    .generator_leaves
                    .windows(2)
                    .all(|pair| pair[0] < pair[1])
            );

            for document in 0..128_u64 {
                let mut leaves = vec![HashMap::new(); leaf_count];
                for (leaf_index, input) in inputs.iter().enumerate() {
                    if input.possible && rng.random_bool(0.5) {
                        let score = rng.random_range(input.bounds.lower()..=input.bounds.upper());
                        leaves[leaf_index].insert(document, score);
                    }
                }
                if let Some(score) = exhaustive_compound_scores(&plan, &leaves).get(&document) {
                    assert!(analysis.possible, "seed={seed}, document={document}");
                    assert!(
                        analysis.bounds.lower() <= *score && *score <= analysis.bounds.upper(),
                        "seed={seed}, document={document}, score={score}, bounds={:?}",
                        analysis.bounds
                    );
                    assert!(
                        analysis
                            .generator_leaves
                            .iter()
                            .any(|leaf| leaves[*leaf].contains_key(&document)),
                        "seed={seed}, document={document}, generators={:?}",
                        analysis.generator_leaves
                    );
                }
            }
        }
    }

    #[test]
    fn randomized_mapped_sources_match_recursive_exhaustive_oracle() {
        let plans = [
            (
                "should_sum",
                CompoundScorerPlan::Boolean {
                    should: (0..4).map(plan_leaf).collect(),
                    must: Vec::new(),
                    must_not: Vec::new(),
                },
            ),
            (
                "multimatch_max",
                CompoundScorerPlan::MultiMatch((0..4).map(plan_leaf).collect()),
            ),
            (
                "must_sum",
                CompoundScorerPlan::Boolean {
                    should: Vec::new(),
                    must: (0..4).map(plan_leaf).collect(),
                    must_not: Vec::new(),
                },
            ),
            (
                "required_optional",
                CompoundScorerPlan::Boolean {
                    should: vec![plan_leaf(1), plan_leaf(2), plan_leaf(3)],
                    must: vec![plan_leaf(0)],
                    must_not: Vec::new(),
                },
            ),
            (
                "signed_boost",
                CompoundScorerPlan::Boost {
                    positive: Box::new(CompoundScorerPlan::MultiMatch(vec![
                        plan_leaf(0),
                        CompoundScorerPlan::Leaf {
                            index: 1,
                            boost: 0.5,
                        },
                    ])),
                    negative: Box::new(CompoundScorerPlan::Boolean {
                        should: vec![plan_leaf(2), plan_leaf(3)],
                        must: Vec::new(),
                        must_not: Vec::new(),
                    }),
                    negative_boost: 1.5,
                },
            ),
            (
                "must_not",
                CompoundScorerPlan::Boolean {
                    should: vec![plan_leaf(1)],
                    must: vec![plan_leaf(0)],
                    must_not: vec![CompoundScorerPlan::MultiMatch(vec![
                        plan_leaf(2),
                        plan_leaf(3),
                    ])],
                },
            ),
        ];

        for seed in 0..12 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let num_rows = rng.random_range(32..=64);
            let mut next_row_address = (seed + 1) << 32;
            let canonical_row_addresses = (0..num_rows)
                .map(|_| {
                    next_row_address += rng.random_range(1..=16);
                    next_row_address
                })
                .collect::<Vec<_>>();
            let mut leaves = vec![HashMap::<u64, f32>::new(); 4];
            for (row_index, row_address) in canonical_row_addresses.iter().enumerate() {
                let mut matched = false;
                for (leaf_index, leaf) in leaves.iter_mut().enumerate() {
                    let is_required_only_canary = row_index < 16 && leaf_index < 2;
                    let is_random_match = row_index >= 16 && rng.random_bool(0.5);
                    if row_index < 8 || is_required_only_canary || is_random_match {
                        let score = if row_index < 8 {
                            2.0
                        } else {
                            rng.random_range(1..=4) as f32 * 0.5
                        };
                        leaf.insert(*row_address, score);
                        matched = true;
                    }
                }
                if !matched {
                    let leaf_index = rng.random_range(0..leaves.len());
                    let score = rng.random_range(1..=4) as f32 * 0.5;
                    leaves[leaf_index].insert(*row_address, score);
                }
            }

            let max_scores = exhaustive_compound_scores(&plans[1].1, &leaves);
            let max_score = max_scores.values().copied().max_by(f32::total_cmp).unwrap();
            assert!(
                max_scores
                    .values()
                    .filter(|score| **score == max_score)
                    .count()
                    >= 8,
                "seed={seed} should retain enough top-score ties to cross every tested limit"
            );
            assert!(
                exhaustive_compound_scores(&plans[4].1, &leaves)
                    .values()
                    .any(|score| *score < 0.0),
                "seed={seed} should exercise signed Boost scores"
            );
            let must_not_scores = exhaustive_compound_scores(&plans[5].1, &leaves);
            assert!(
                canonical_row_addresses[..8]
                    .iter()
                    .all(|row_address| !must_not_scores.contains_key(row_address))
                    && canonical_row_addresses[8..16]
                        .iter()
                        .all(|row_address| must_not_scores.contains_key(row_address)),
                "seed={seed} should exercise both prohibited and retained candidates"
            );

            for (shape, plan) in &plans {
                for limit in [1, 3, 7] {
                    let expected = exhaustive_compound_top_k(plan, &leaves, limit);
                    let mut mapped_leaves = leaves
                        .iter()
                        .map(|leaf| {
                            Some(randomized_mapped_leaf(
                                leaf,
                                &canonical_row_addresses,
                                &mut rng,
                            ))
                        })
                        .collect::<Vec<_>>();
                    let mut scorer = plan.build(&mut mapped_leaves).unwrap();
                    assert!(mapped_leaves.iter().all(Option::is_none));
                    let actual = TopKCollector::new(limit).collect(scorer.as_mut()).unwrap();
                    assert_eq!(actual, expected, "seed={seed} shape={shape} limit={limit}");
                }
            }
        }
    }

    #[test]
    fn pure_should_maxscore_reduces_posting_comparisons() {
        let (children, eager_work) = pure_should_canary_children();
        let mut eager = DisjunctionScorer::try_new(children, DisjunctionScore::Sum).unwrap();
        let eager_results = TopKCollector::new(1).collect(&mut eager).unwrap();
        let eager_comparisons = scorer_advances(&eager_work);

        let (children, optimized_work) = pure_should_canary_children();
        let optimized_results = {
            let mut optimized = should_maxscore(children);
            TopKCollector::new(1).collect(&mut optimized).unwrap()
        };
        let optimized_comparisons = scorer_advances(&optimized_work);

        assert_eq!(eager_results, rows(&[(127, 2.5)]));
        assert_eq!(optimized_results, eager_results);
        assert!(eager_comparisons > 0);
        assert!(
            optimized_comparisons * 5 <= eager_comparisons * 4,
            "pure-SHOULD MAXSCORE should reduce posting candidate probes by at least 20%: \
             optimized={optimized_comparisons} eager={eager_comparisons}"
        );
    }

    #[test]
    fn pure_should_maxscore_matches_randomized_exhaustive_top_k() {
        for seed in 0..8 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let children = (0..6)
                .map(|_| {
                    (0..256)
                        .filter_map(|doc| {
                            if rng.random_bool(0.35) {
                                let score = rng.random_range(1..=16) as f32 * 0.25;
                                Some((doc, score))
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            for limit in [1, 7, 31, 512] {
                let expected = exhaustive_should_top_k(&children, limit);

                let mut optimized =
                    should_maxscore(children.iter().map(|values| materialized(values)).collect());
                let actual = TopKCollector::new(limit).collect(&mut optimized).unwrap();
                assert_eq!(actual, expected, "seed={seed} limit={limit}");
            }
        }
    }

    #[test]
    fn pure_should_maxscore_confirms_two_phase_children_before_scoring() {
        let (phrase, _, confirmations) = two_phase(&[(1, 100.0)], Vec::new(), Some(10.0));
        let competitive_score = Arc::new(CompetitiveScore::default());
        competitive_score.raise(10.0);
        let results = {
            let mut scorer = should_maxscore(vec![
                materialized(&[(0, 10.0)]),
                phrase,
                materialized(&[(1, 6.0)]),
                materialized(&[(1, 5.0)]),
            ]);
            TopKCollector::with_competitive_score(1, competitive_score)
                .collect(&mut scorer)
                .unwrap()
        };

        assert_eq!(results, rows(&[(1, 11.0)]));
        assert_eq!(confirmations.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    fn pure_should_maxscore_preserves_query_score_order_and_terminal_doc() {
        let mut scorer = should_maxscore(vec![
            materialized(&[(u64::MAX, 16_777_216.0)]),
            materialized(&[(u64::MAX, 1.0)]),
            materialized(&[(u64::MAX, 1.0)]),
            materialized(&[]),
        ]);

        assert_eq!(
            TopKCollector::new(1).collect(&mut scorer).unwrap(),
            rows(&[(u64::MAX, 16_777_216.0)])
        );
    }

    #[test]
    fn pure_should_maxscore_keeps_equal_floor_across_bound_ordering() {
        let scores = [
            f32::from_bits(0x4783_798b),
            f32::from_bits(0x4dd3_8b75),
            f32::from_bits(0x48e7_7236),
            f32::from_bits(0x418e_5b26),
            f32::from_bits(0x4241_b1eb),
        ];
        let exact_score = scores
            .into_iter()
            .fold(0.0_f32, |total, score| total + score);
        assert_eq!(exact_score.to_bits(), 0x4dd3_cd8d);

        let mut scorer = should_maxscore(
            scores
                .into_iter()
                .map(|score| materialized(&[(7, score)]))
                .collect(),
        );
        let competitive_score = Arc::new(CompetitiveScore::default());
        competitive_score.raise(exact_score);

        assert_eq!(
            TopKCollector::with_competitive_score(1, competitive_score)
                .collect(&mut scorer)
                .unwrap(),
            rows(&[(7, exact_score)])
        );
    }

    #[test]
    fn pure_should_maxscore_supports_nested_non_negative_children() {
        let nested_dismax = Box::new(
            DisjunctionScorer::try_new(
                vec![
                    materialized(&[(0, 1.0), (1, 5.0)]),
                    materialized(&[(0, 3.0), (2, 4.0)]),
                ],
                DisjunctionScore::Max,
            )
            .unwrap(),
        );
        let nested_boolean = Box::new(
            BooleanScorer::try_new(
                Vec::new(),
                vec![materialized(&[(0, 2.0), (1, 2.0), (2, 2.0)])],
                vec![materialized(&[(1, 0.0)])],
            )
            .unwrap(),
        );
        let results = {
            let mut scorer = BooleanScorer::try_new(
                vec![
                    nested_dismax,
                    nested_boolean,
                    materialized(&[(0, 0.5), (1, 0.5), (2, 0.5)]),
                ],
                Vec::new(),
                Vec::new(),
            )
            .unwrap();
            TopKCollector::new(1).collect(&mut scorer).unwrap()
        };

        assert_eq!(results, rows(&[(2, 6.5)]));
    }

    #[test]
    fn pure_should_maxscore_applies_must_not_before_raising_the_floor() {
        let results = {
            let mut scorer = BooleanScorer::try_new(
                vec![
                    materialized(&[(0, 10.0), (1, 5.0)]),
                    materialized(&[(0, 1.0), (1, 1.0)]),
                    materialized(&[(2, 8.0)]),
                ],
                Vec::new(),
                vec![materialized(&[(0, 1.0)])],
            )
            .unwrap();
            TopKCollector::new(1).collect(&mut scorer).unwrap()
        };

        assert_eq!(results, rows(&[(2, 8.0)]));
    }

    #[test]
    fn pure_should_uses_exact_fallback_for_unsupported_shapes() {
        let signed_results = {
            let signed = Box::new(
                BoostScorer::try_new(
                    materialized(&[(0, 5.0), (1, 1.0)]),
                    materialized(&[(0, 2.0), (1, 4.0)]),
                    1.0,
                )
                .unwrap(),
            );
            let mut scorer = BooleanScorer::try_new(
                vec![
                    signed,
                    materialized(&[(0, 1.0), (1, 1.0)]),
                    materialized(&[(1, 5.0)]),
                ],
                Vec::new(),
                Vec::new(),
            )
            .unwrap();
            TopKCollector::new(2).collect(&mut scorer).unwrap()
        };
        assert_eq!(signed_results, rows(&[(0, 4.0), (1, 3.0)]));

        let unbounded_results = {
            let unbounded = Box::new(UnboundedScorer {
                inner: MaterializedScorer::try_new(rows(&[(0, 1.0), (2, 3.0)])).unwrap(),
            });
            let mut scorer = BooleanScorer::try_new(
                vec![
                    unbounded,
                    materialized(&[(0, 2.0), (1, 2.0)]),
                    materialized(&[(1, 4.0)]),
                ],
                Vec::new(),
                Vec::new(),
            )
            .unwrap();
            TopKCollector::new(3).collect(&mut scorer).unwrap()
        };
        assert_eq!(unbounded_results, rows(&[(1, 6.0), (0, 3.0), (2, 3.0)]));

        {
            let mut scorer = BooleanScorer::try_new(
                vec![materialized(&[(0, 1.0)]), materialized(&[(1, 2.0)])],
                Vec::new(),
                Vec::new(),
            )
            .unwrap();
            assert_eq!(
                TopKCollector::new(2).collect(&mut scorer).unwrap(),
                rows(&[(1, 2.0), (0, 1.0)])
            );
        }

        let large_score = f32::MAX / 2.0;
        {
            let mut scorer = BooleanScorer::try_new(
                vec![
                    materialized(&[(0, large_score)]),
                    materialized(&[(1, large_score)]),
                    materialized(&[(2, large_score)]),
                ],
                Vec::new(),
                Vec::new(),
            )
            .unwrap();
            assert_eq!(
                TopKCollector::new(3).collect(&mut scorer).unwrap(),
                rows(&[(0, large_score), (1, large_score), (2, large_score)])
            );
        }
    }

    fn plain_unit_posting(token: &str, doc_ids: Vec<u64>) -> PostingIterator {
        let freqs = vec![1.0_f32; doc_ids.len()];
        PostingIterator::with_query_weight(
            token.to_owned(),
            0,
            0,
            1.0,
            PostingList::Plain(PlainPostingList::new(
                ScalarBuffer::from(doc_ids),
                ScalarBuffer::from(freqs),
                Some(1.0),
                None,
            )),
            1,
        )
    }

    fn loaded_leaf(postings: Vec<PostingIterator>, phrase_slop: Option<u32>) -> LoadedLeaf {
        let params = FtsSearchParams::default().with_phrase_slop(phrase_slop);
        LoadedLeaf {
            postings,
            params: Arc::new(params),
            operator: Operator::Or,
            scorer: Arc::new(MemBM25Scorer::new(1, 1, HashMap::new())),
        }
    }

    #[test]
    fn term_leaf_optional_probe_matches_wand_cursor() {
        // After the ReqOpt change a SHOULD leaf of a 1-MUST Boolean is driven by
        // TermLeafScorer instead of WandCursor. For the probe pattern the parent
        // uses (`advance(required_doc)`), the two must visit exactly the same
        // documents and report the same scores.
        let mut docs = DocSet::default();
        for row in 0..8u64 {
            docs.append(row, 3);
        }
        let ids = vec![1u64, 3, 5, 6];
        let metrics = NoOpMetricsCollector;
        let leaf = || loaded_leaf(vec![plain_unit_posting("s", ids.clone())], None);
        let mut term = box_leaf_scorer(leaf(), &docs, &metrics, true);
        let mut wand = box_leaf_scorer(leaf(), &docs, &metrics, false);

        assert!(
            term.debug_type_name().contains("TermLeafScorer"),
            "got {}",
            term.debug_type_name()
        );
        assert!(
            !wand.debug_type_name().contains("TermLeafScorer"),
            "got {}",
            wand.debug_type_name()
        );

        let mut visited = std::collections::BTreeSet::new();
        for target in [0u64, 1, 2, 3, 4, 5, 6, 7, 8] {
            let t = term.advance(target).unwrap();
            let w = wand.advance(target).unwrap();
            assert_eq!(t, w, "advance({target})");
            assert_eq!(term.doc(), wand.doc(), "doc after advance({target})");
            if let Some(doc) = t {
                visited.insert(doc);
                assert_eq!(
                    term.score().unwrap(),
                    wand.score().unwrap(),
                    "score at {target}"
                );
            }
        }
        assert_eq!(
            visited,
            ids.iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>(),
            "every posting document was probed"
        );
    }

    #[test]
    fn box_leaf_scorer_uses_term_leaf_for_one_ungrouped_term() {
        let documents = DocSet::default();
        let metrics = NoOpMetricsCollector;
        let leaf = loaded_leaf(vec![plain_unit_posting("a", vec![0])], None);
        let scorer = box_leaf_scorer(leaf, &documents, &metrics, true);
        assert!(
            scorer.debug_type_name().contains("TermLeafScorer"),
            "got {}",
            scorer.debug_type_name()
        );
    }

    #[test]
    fn reqopt_admits_one_must_and_its_shoulds() {
        // The ReqOpt shape (one MUST + N SHOULD): the parent holds the floor
        // itself and probes the optional side by `advance(required_doc)` once
        // per required document. Neither side is steered by a union machine or
        // by a forwarded floor, so both may step a direct posting: WandCursor's
        // window bookkeeping is pure overhead there, and TermLeafScorer's
        // no-floor contract costs nothing.
        let leaf_node = |index: usize| CompoundScorerPlan::Leaf { index, boost: 1.0 };
        let plan = CompoundScorerPlan::Boolean {
            must: vec![leaf_node(0)],
            should: vec![leaf_node(1)],
            must_not: Vec::new(),
        };
        assert!(leaf_admits_term_leaf(&plan, 0), "required side");
        assert!(
            leaf_admits_term_leaf(&plan, 1),
            "optional side of a 1-MUST Boolean"
        );

        // A MUST_NOT still drives through ReqOptScorer, but admitting its
        // optional side is unmeasured, so it keeps the WAND cursor.
        let plan = CompoundScorerPlan::Boolean {
            must: vec![leaf_node(0)],
            should: vec![leaf_node(1)],
            must_not: vec![leaf_node(2)],
        };
        assert!(leaf_admits_term_leaf(&plan, 0));
        assert!(!leaf_admits_term_leaf(&plan, 1));

        // A multi-MUST conjunction is driven by the parent's skipping; its
        // leaves are not admitted.
        let plan = CompoundScorerPlan::Boolean {
            must: vec![leaf_node(0), leaf_node(1)],
            should: Vec::new(),
            must_not: Vec::new(),
        };
        assert!(!leaf_admits_term_leaf(&plan, 0));
        assert!(!leaf_admits_term_leaf(&plan, 1));

        // A SHOULD-only union: every leaf keeps its WAND cursor.
        let plan = CompoundScorerPlan::Boolean {
            must: Vec::new(),
            should: vec![leaf_node(0), leaf_node(1)],
            must_not: Vec::new(),
        };
        assert!(!leaf_admits_term_leaf(&plan, 0));
        assert!(!leaf_admits_term_leaf(&plan, 1));
    }

    #[test]
    fn reqopt_fallback_enumerates_every_must_doc_for_counting() {
        // The COUNT path has no competitive floor: it relies on the IU fast
        // path and the ReqOptScorer fallback both enumerating every MUST
        // document. Plain postings fail `iu_tight_clauses_ready` (not
        // compressed), so this exercises the ReqOptScorer fallback; the
        // kernel side is pinned by
        // `iu_tight_search_sums_must_and_matching_shoulds` (floor = -inf).
        let mut docs = DocSet::default();
        for doc_id in 0..50_u64 {
            docs.append(doc_id, 1);
        }
        let must_docs: Vec<u64> = (0..50).step_by(2).collect();
        let should_docs: Vec<u64> = (0..50).step_by(3).collect();

        let leaf_node = |index: usize| CompoundScorerPlan::Leaf { index, boost: 1.0 };
        let plan = CompoundScorerPlan::Boolean {
            must: vec![leaf_node(0)],
            should: vec![leaf_node(1)],
            must_not: Vec::new(),
        };

        let build_leaves = || {
            vec![
                loaded_leaf(vec![plain_unit_posting("must", must_docs.clone())], None),
                loaded_leaf(
                    vec![plain_unit_posting("should", should_docs.clone())],
                    None,
                ),
            ]
        };

        let leaves = build_leaves();
        let (must_indices, should_indices) =
            iu_tight_clause_indices(&plan).expect("1+1 shape must resolve");
        assert_eq!(must_indices, vec![0]);
        assert_eq!(should_indices, vec![1]);
        assert!(
            !iu_tight_clauses_ready(&leaves, &must_indices, &should_indices),
            "plain postings must route to the ReqOptScorer fallback"
        );

        let mut seen = std::collections::BTreeSet::new();
        let mut collector = TopKCollector::new(10_000);
        let status = collect_partition_with_documents(
            &docs,
            build_leaves(),
            &plan,
            &NoOpMetricsCollector,
            &mut collector,
            |key| {
                seen.insert(key);
                Ok(key)
            },
        )
        .unwrap();
        assert_eq!(status, CollectionStatus::Complete);

        let want: std::collections::BTreeSet<u64> = must_docs.iter().copied().collect();
        assert_eq!(
            seen, want,
            "a floor-less collection must enumerate every MUST document"
        );
    }

    fn compressed_unit_posting(token: &str, doc_ids: Vec<u32>) -> PostingIterator {
        let max_score = 1.0_f32;
        let freqs = vec![1_u32; doc_ids.len()];
        let block_max_scores = vec![max_score; doc_ids.len()];
        let blocks = compress_posting_list(
            doc_ids.len(),
            doc_ids.iter(),
            freqs.iter(),
            block_max_scores.into_iter(),
        )
        .unwrap();
        let num_docs = doc_ids.iter().copied().max().unwrap_or(0) as usize + 1;
        PostingIterator::with_query_weight(
            token.to_owned(),
            0,
            0,
            1.0,
            PostingList::Compressed(CompressedPostingList::new(
                blocks,
                max_score,
                doc_ids.len() as u32,
                PostingTailCodec::VarintDelta,
                LEGACY_BLOCK_SIZE,
                None,
                None,
            )),
            num_docs,
        )
    }

    #[test]
    fn iu_tight_multi_dry_optional_ends_dense_intersection() {
        // Two dense required clauses over 65,536 documents plus an optional
        // whose only document is doc 0. Once that optional runs dry, every
        // remaining candidate is provably below the floor, so the walk must
        // end after a couple of candidates instead of scoring the whole
        // intersection. Without the budget the kernel scored all 65,536
        // candidates.
        let n = 65_536;
        let mut docs = DocSet::default();
        for id in 0..n {
            docs.append(id, 1);
        }
        let leaves = || {
            vec![
                loaded_leaf(
                    vec![compressed_unit_posting("a", (0..n as u32).collect())],
                    None,
                ),
                loaded_leaf(
                    vec![compressed_unit_posting("b", (0..n as u32).collect())],
                    None,
                ),
                loaded_leaf(vec![compressed_unit_posting("c", vec![0])], None),
            ]
        };
        let metrics = LocalMetricsCollector::default();
        let mut tight = TopKCollector::new(1);
        collect_iu_tight(&docs, leaves(), &[0, 1], &[2], &metrics, &mut tight, Ok).unwrap();
        let rows = tight.into_rows();
        assert_eq!(rows.len(), 1, "top-1");
        assert_eq!(rows[0].row_id, 0);
        assert_eq!(rows[0].score, 3.0);
        let comparisons = metrics
            .comparisons
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            comparisons <= 2,
            "the dry optional must end the walk within a couple of candidates, got {comparisons}"
        );
    }

    #[test]
    fn iu_tight_multi_finite_floor_keeps_far_optional_hit() {
        // The dry budget must only count cursors that are truly exhausted.
        // Two optionals hit only doc 500: at candidate 0 their cursors sit
        // past the candidate without being dry. Once the near optional dries
        // at candidate 10, a walk that mistook "past the candidate" for dry
        // would drop both bounds, prove 2 < floor=3, and terminate — losing
        // doc 500, whose two optional hits make it the top hit.
        let n = 501;
        let mut docs = DocSet::default();
        for id in 0..n {
            docs.append(id, 1);
        }
        let leaves = || {
            vec![
                loaded_leaf(
                    vec![compressed_unit_posting("a", (0..n as u32).collect())],
                    None,
                ),
                loaded_leaf(
                    vec![compressed_unit_posting("b", (0..n as u32).collect())],
                    None,
                ),
                loaded_leaf(
                    vec![compressed_unit_posting("c1", (0..10_u32).collect())],
                    None,
                ),
                loaded_leaf(vec![compressed_unit_posting("c2", vec![500])], None),
                loaded_leaf(vec![compressed_unit_posting("c3", vec![500])], None),
            ]
        };
        let metrics = LocalMetricsCollector::default();
        let mut tight = TopKCollector::new(1);
        collect_iu_tight(
            &docs,
            leaves(),
            &[0, 1],
            &[2, 3, 4],
            &metrics,
            &mut tight,
            Ok,
        )
        .unwrap();
        let rows = tight.into_rows();
        assert_eq!(rows.len(), 1, "top-1");
        assert_eq!(rows[0].row_id, 500, "the far double optional hit must win");
        assert_eq!(rows[0].score, 4.0);
        // Generic-tree contract check on the same shape.
        let leaf_node = |index| CompoundScorerPlan::Leaf { index, boost: 1.0 };
        let plan = CompoundScorerPlan::Boolean {
            must: vec![leaf_node(0), leaf_node(1)],
            should: vec![leaf_node(2), leaf_node(3), leaf_node(4)],
            must_not: Vec::new(),
        };
        let generic_metrics = NoOpMetricsCollector;
        let mut generic_leaves = leaves()
            .into_iter()
            .map(|leaf| Some(box_leaf_scorer(leaf, &docs, &generic_metrics, false)))
            .collect::<Vec<_>>();
        let mut generic = plan.build(&mut generic_leaves).unwrap();
        let mut collector = TopKCollector::new(1);
        let mut current = generic.next().unwrap();
        while let Some(id) = current {
            let floor = collector.min_competitive_score();
            generic.set_min_competitive_score(floor).unwrap();
            let up_to = generic.advance_shallow(id).unwrap();
            if generic.score_bounds(up_to).unwrap().upper < floor {
                current = if up_to == u64::MAX {
                    None
                } else {
                    generic.advance(up_to + 1).unwrap()
                };
                continue;
            }
            if generic.matches().unwrap() {
                let score = generic.score().unwrap();
                collector.insert(ScoredRow {
                    row_id: generic.document_key().unwrap(),
                    score,
                });
            }
            current = generic.next().unwrap();
        }
        assert_eq!(rows, collector.into_rows());
    }

    #[test]
    fn iu_tight_multi_misaligned_blocks_match_generic() {
        // The two required cursors pack their 128-doc blocks at different
        // boundaries (must_a has a hole at 1..255, must_b is dense from 0),
        // so their block ends are misaligned while the block prune walks.
        // The optional's hits sit far apart. Results must stay bit-identical
        // to the generic tree under a real collector floor.
        let n = 3_000;
        let mut docs = DocSet::default();
        for id in 0..n {
            docs.append(id, 1);
        }
        let must_a_docs: Vec<u32> = [0_u32].into_iter().chain(256..n as u32).collect();
        let must_b_docs: Vec<u32> = (0..n as u32).collect();
        let optional_docs: Vec<u32> = vec![50, 500, 1500, 2900];
        let leaves = || {
            vec![
                loaded_leaf(
                    vec![compressed_unit_posting("a", must_a_docs.clone())],
                    None,
                ),
                loaded_leaf(
                    vec![compressed_unit_posting("b", must_b_docs.clone())],
                    None,
                ),
                loaded_leaf(
                    vec![compressed_unit_posting("c", optional_docs.clone())],
                    None,
                ),
            ]
        };
        let metrics = LocalMetricsCollector::default();
        let mut tight = TopKCollector::new(10);
        collect_iu_tight(&docs, leaves(), &[0, 1], &[2], &metrics, &mut tight, Ok).unwrap();
        let tight_rows = tight.into_rows();
        let generic_metrics = NoOpMetricsCollector;
        let mut generic_leaves = leaves()
            .into_iter()
            .map(|leaf| Some(box_leaf_scorer(leaf, &docs, &generic_metrics, false)))
            .collect::<Vec<_>>();
        let leaf_node = |index| CompoundScorerPlan::Leaf { index, boost: 1.0 };
        let plan = CompoundScorerPlan::Boolean {
            must: vec![leaf_node(0), leaf_node(1)],
            should: vec![leaf_node(2)],
            must_not: Vec::new(),
        };
        let mut generic = plan.build(&mut generic_leaves).unwrap();
        let mut collector = TopKCollector::new(10);
        let mut current = generic.next().unwrap();
        while let Some(id) = current {
            let floor = collector.min_competitive_score();
            generic.set_min_competitive_score(floor).unwrap();
            let up_to = generic.advance_shallow(id).unwrap();
            if generic.score_bounds(up_to).unwrap().upper < floor {
                current = if up_to == u64::MAX {
                    None
                } else {
                    generic.advance(up_to + 1).unwrap()
                };
                continue;
            }
            if generic.matches().unwrap() {
                let score = generic.score().unwrap();
                collector.insert(ScoredRow {
                    row_id: generic.document_key().unwrap(),
                    score,
                });
            }
            current = generic.next().unwrap();
        }
        assert_eq!(tight_rows, collector.into_rows());

        // The far optional hit must be part of the result.
        assert!(tight_rows.iter().any(|r| r.row_id == 2900));
    }

    #[test]
    fn multi_promotion_keeps_far_optional_hit_cheaply() {
        // A promotion must drive from the lone sparse optional once the
        // floor rises past the required lists' combined bound: the far hit
        // (doc 299999, must+optional) is scored exactly once and the walk
        // never touches the other ~300k members. Deleting the promotion or
        // mis-recomputing required_ub blows the comparisons bound.
        let n = 300_000;
        let mut docs = DocSet::default();
        for id in 0..n {
            docs.append(id, 1);
        }
        let mut optional_docs: Vec<u32> = (0..10_u32).collect();
        optional_docs.push(n as u32 - 1);
        let leaves = || {
            vec![
                loaded_leaf(
                    vec![compressed_unit_posting("a", (0..n as u32).collect())],
                    None,
                ),
                loaded_leaf(
                    vec![compressed_unit_posting("b", (0..n as u32).collect())],
                    None,
                ),
                loaded_leaf(
                    vec![compressed_unit_posting("c", optional_docs.clone())],
                    None,
                ),
            ]
        };
        let metrics = LocalMetricsCollector::default();
        let mut tight = TopKCollector::new(10);
        collect_iu_tight(&docs, leaves(), &[0, 1], &[2], &metrics, &mut tight, Ok).unwrap();
        let comparisons = metrics
            .comparisons
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            (11..=100).contains(&comparisons),
            "promotion must drive from the sparse optional, got {comparisons} comparisons"
        );
        let rows = tight.into_rows();
        assert_eq!(rows.len(), 10);
        assert!(rows.iter().all(|r| r.score == 3.0));
    }

    #[test]
    fn iu_tight_collector_admits_score_equal_floor_with_smaller_row() {
        // `IuCollector::hit` must admit a document whose score exactly equals
        // the competitive floor: TopKCollector resolves score ties by row id,
        // so a later document with a smaller mapped row can still displace
        // the incumbent. This walks the real collector (the differential
        // kernel tests carry their own hit closures and never see it).
        let mut docs = DocSet::default();
        for id in 0..6_u64 {
            docs.append(id, 1);
        }
        let leaves = || {
            vec![
                loaded_leaf(vec![compressed_unit_posting("a", vec![0, 1])], None),
                loaded_leaf(vec![compressed_unit_posting("b", vec![0, 1])], None),
                loaded_leaf(vec![compressed_unit_posting("s", vec![5])], None),
            ]
        };
        let metrics = LocalMetricsCollector::default();
        let mut tight = TopKCollector::new(1);
        collect_iu_tight(
            &docs,
            leaves(),
            &[0, 1],
            &[2],
            &metrics,
            &mut tight,
            |key| Ok(1000 - key),
        )
        .unwrap();
        let rows = tight.into_rows();
        assert_eq!(rows.len(), 1, "top-1");
        assert_eq!(
            rows[0].row_id, 999,
            "doc 1 maps to the smaller row and wins the tie"
        );
        assert_eq!(rows[0].score, 2.0);
    }

    #[test]
    fn iu_tight_routing_ignores_clause_cost() {
        // Routing is a shape predicate: a K=1 IU plan over single compressed
        // postings takes the kernel whatever the clause costs are. This pins
        // that cost stays out of the decision — a regenerated relative cost
        // gate inside the condition would show up here as a false match.
        // (It cannot catch a gate added *after* the two helpers, so the
        // ablation in the commit message stays the performance record.)
        let mut docs = DocSet::default();
        for doc_id in 0..50_u64 {
            docs.append(doc_id, 1);
        }
        let leaf_node = |index: usize| CompoundScorerPlan::Leaf { index, boost: 1.0 };
        let plan = CompoundScorerPlan::Boolean {
            must: vec![leaf_node(0)],
            should: vec![leaf_node(1)],
            must_not: Vec::new(),
        };
        // Close-cost shapes measured faster through the kernel than through
        // the fallback path, so clause cost stays out of the decision.
        let leaves = vec![
            loaded_leaf(
                vec![compressed_unit_posting("must", (0..50).collect())],
                None,
            ),
            loaded_leaf(
                vec![compressed_unit_posting("should", (0..50).collect())],
                None,
            ),
        ];
        let (must_indices, should_indices) = iu_tight_clause_indices(&plan).expect("1+1 shape");
        assert_eq!(must_indices, vec![0]);
        assert_eq!(should_indices, vec![1]);
        assert!(
            iu_tight_clauses_ready(&leaves, &must_indices, &should_indices),
            "a close-cost 1+1 shape must still route to the kernel"
        );
    }

    #[test]
    fn iu_tight_fast_paths_record_comparisons() {
        // The IU fast paths bypass the scorer tree, so they must report their
        // own comparison counts; a zero here means the metric was dropped.
        let mut docs = DocSet::default();
        for doc_id in 0..50_u64 {
            docs.append(doc_id, 1);
        }
        let leaf_node = |index: usize| CompoundScorerPlan::Leaf { index, boost: 1.0 };

        // K=1: the dense MUST against two sparse SHOULDs clears the 3x gate.
        let k1_plan = CompoundScorerPlan::Boolean {
            must: vec![leaf_node(0)],
            should: vec![leaf_node(1)],
            must_not: Vec::new(),
        };
        let k1_leaves = vec![
            loaded_leaf(
                vec![compressed_unit_posting("must", (0..50).collect())],
                None,
            ),
            loaded_leaf(vec![compressed_unit_posting("should", vec![3, 9])], None),
        ];
        let (k1_must, k1_should) = iu_tight_clause_indices(&k1_plan).expect("1+1 shape");
        assert!(iu_tight_clauses_ready(&k1_leaves, &k1_must, &k1_should));

        let metrics = LocalMetricsCollector::default();
        let mut collector = TopKCollector::new(10);
        let status = collect_partition_with_documents(
            &docs,
            k1_leaves,
            &k1_plan,
            &metrics,
            &mut collector,
            Ok,
        )
        .unwrap();
        assert_eq!(status, CollectionStatus::Complete);
        assert!(
            metrics
                .comparisons
                .load(std::sync::atomic::Ordering::Relaxed)
                > 0,
            "K=1 IU fast path must record comparisons"
        );

        // K>=2: two dense MUSTs plus a sparse SHOULD take the multi kernel.
        let k2_plan = CompoundScorerPlan::Boolean {
            must: vec![leaf_node(0), leaf_node(1)],
            should: vec![leaf_node(2)],
            must_not: Vec::new(),
        };
        let k2_leaves = vec![
            loaded_leaf(
                vec![compressed_unit_posting("must_a", (0..50).collect())],
                None,
            ),
            loaded_leaf(
                vec![compressed_unit_posting(
                    "must_b",
                    (0..50).step_by(2).collect(),
                )],
                None,
            ),
            loaded_leaf(vec![compressed_unit_posting("should", vec![3, 9])], None),
        ];
        let (k2_must, k2_should) = iu_tight_clause_indices(&k2_plan).expect("2+1 shape");
        assert_eq!(k2_must.len(), 2);
        assert!(iu_tight_clauses_ready(&k2_leaves, &k2_must, &k2_should));

        let metrics = LocalMetricsCollector::default();
        let mut collector = TopKCollector::new(10);
        let status = collect_partition_with_documents(
            &docs,
            k2_leaves,
            &k2_plan,
            &metrics,
            &mut collector,
            Ok,
        )
        .unwrap();
        assert_eq!(status, CollectionStatus::Complete);
        assert!(
            metrics
                .comparisons
                .load(std::sync::atomic::Ordering::Relaxed)
                > 0,
            "K>=2 IU fast path must record comparisons"
        );
    }

    #[test]
    fn box_leaf_scorer_keeps_wand_cursor_for_phrase_grouped_and_multi_term() {
        let documents = DocSet::default();
        let metrics = NoOpMetricsCollector;

        let phrase = loaded_leaf(vec![plain_unit_posting("a", vec![0])], Some(0));
        let phrase_scorer = box_leaf_scorer(phrase, &documents, &metrics, true);
        assert!(
            phrase_scorer.debug_type_name().contains("WandCursor"),
            "got {}",
            phrase_scorer.debug_type_name()
        );

        let list = PostingList::Plain(PlainPostingList::new(
            ScalarBuffer::from(vec![0_u64]),
            ScalarBuffer::from(vec![1.0_f32]),
            Some(1.0),
            None,
        ));
        let grouped =
            PostingIterator::with_query_weight("g".to_owned(), 0, 0, 1.0, list.clone(), 1)
                .with_grouped_terms(Arc::from([GroupedTermScorer::new(1.0, &list)]));
        let grouped_scorer =
            box_leaf_scorer(loaded_leaf(vec![grouped], None), &documents, &metrics, true);
        assert!(
            grouped_scorer.debug_type_name().contains("WandCursor"),
            "got {}",
            grouped_scorer.debug_type_name()
        );

        let multi = loaded_leaf(
            vec![
                plain_unit_posting("a", vec![0]),
                plain_unit_posting("b", vec![0]),
            ],
            None,
        );
        let multi_scorer = box_leaf_scorer(multi, &documents, &metrics, true);
        assert!(
            multi_scorer.debug_type_name().contains("WandCursor"),
            "got {}",
            multi_scorer.debug_type_name()
        );
    }

    #[test]
    fn try_root_unwraps_reqopt_for_one_must_and_keeps_boolean_with_must_not() {
        let root = BooleanScorer::try_root(
            vec![
                materialized(&[(0, 1.0), (1, 1.0)]),
                materialized(&[(1, 2.0)]),
            ],
            vec![materialized(&[(0, 3.0), (1, 3.0)])],
            Vec::new(),
        )
        .unwrap();
        assert!(
            root.debug_type_name().contains("ReqOptScorer"),
            "got {}",
            root.debug_type_name()
        );
        let children = root.debug_child_type_names();
        assert!(
            !children[0].contains("RequiredConjunctionScorer"),
            "required child was {}",
            children[0]
        );

        let with_must_not = BooleanScorer::try_root(
            vec![materialized(&[(0, 1.0)])],
            vec![materialized(&[(0, 3.0)])],
            vec![materialized(&[(1, 1.0)])],
        )
        .unwrap();
        assert!(
            with_must_not.debug_type_name().contains("BooleanScorer"),
            "got {}",
            with_must_not.debug_type_name()
        );
    }

    #[test]
    fn plan_build_still_wraps_identity_boost_in_scale() {
        let mut leaves = vec![Some(materialized(&[(0, 1.0)]))];
        let identity = CompoundScorerPlan::Leaf {
            index: 0,
            boost: 1.0,
        }
        .build(&mut leaves)
        .unwrap();
        assert!(
            identity.debug_type_name().contains("ScaleScorer"),
            "got {}",
            identity.debug_type_name()
        );

        let mut leaves = vec![Some(materialized(&[(0, 1.0)]))];
        let boosted = CompoundScorerPlan::Leaf {
            index: 0,
            boost: 2.0,
        }
        .build(&mut leaves)
        .unwrap();
        assert!(
            boosted.debug_type_name().contains("ScaleScorer"),
            "got {}",
            boosted.debug_type_name()
        );
        assert!(
            boosted.debug_child_type_names()[0].contains("MaterializedScorer"),
            "got {:?}",
            boosted.debug_child_type_names()
        );
    }

    #[test]
    fn two_must_children_stay_conjunction_and_do_not_forward_full_floor() {
        let mut conjunction = RequiredConjunctionScorer::try_new(vec![
            materialized(&[(0, 1.0)]),
            materialized(&[(0, 2.0)]),
        ])
        .unwrap();
        assert_eq!(conjunction.children.len(), 2);
        conjunction.set_min_competitive_score(10.0).unwrap();
        for child in &conjunction.children {
            assert_eq!(
                child.debug_last_min_competitive_score(),
                Some(f32::NEG_INFINITY)
            );
        }

        let boolean = BooleanScorer::try_new(
            Vec::new(),
            vec![materialized(&[(0, 1.0)]), materialized(&[(0, 2.0)])],
            Vec::new(),
        )
        .unwrap();
        assert!(
            boolean
                .driver
                .debug_type_name()
                .contains("RequiredConjunctionScorer"),
            "got {}",
            boolean.driver.debug_type_name()
        );
    }

    #[test]
    fn signed_should_keeps_boolean_root_and_includes_optional_score() {
        let signed_should = Box::new(
            BoostScorer::try_new(materialized(&[(0, 2.0)]), materialized(&[(0, 1.0)]), 1.0)
                .unwrap(),
        );
        assert!(!signed_should.scores_non_negative());
        let mut root = BooleanScorer::try_root(
            vec![signed_should],
            vec![materialized(&[(0, 3.0)])],
            Vec::new(),
        )
        .unwrap();
        assert!(
            root.debug_type_name().contains("BooleanScorer"),
            "got {}",
            root.debug_type_name()
        );
        assert_eq!(
            TopKCollector::new(1).collect(root.as_mut()).unwrap(),
            rows(&[(0, 4.0)])
        );
    }
}
