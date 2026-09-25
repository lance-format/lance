// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What a memtable index can be asked, and what it answers.
//!
//! One vocabulary covers every family. A scalar index, a vector index and a
//! full-text index differ in the question they take and the shape of the answer
//! they give, not in how they are reached, so there is one query type and one
//! result type rather than a method per family.
//!
//! The query type is open: [`MemQuery`] is implemented for anything, and
//! blanket-implemented for every [`AnyQuery`] Lance already defines. A plugin
//! that invents a question needs no change here, and a scalar plugin reuses the
//! query types its on-disk index already answers.
//!
//! The answer comes in two shapes because there really are two. A filter names
//! a set of rows and the set can be bracketed — some rows certainly match, some
//! might. A ranked search names an ordered list with scores, where bracketing
//! means nothing. [`MemMatches`] is that distinction and nothing more.

use std::any::Any;
use std::fmt::Debug;

use std::sync::Arc;

use arrow_array::FixedSizeListArray;
use arrow_schema::{DataType, Field};
use lance_index::scalar::AnyQuery;
use lance_index::scalar::inverted::DocumentGranularity;
use lance_linalg::distance::DistanceType;
use roaring::RoaringTreemap;

use super::RowPosition;

/// A question put to a memtable index.
///
/// Implemented for free by every [`AnyQuery`], so a scalar plugin asks its
/// memtable index exactly what it asks its on-disk index. A family with no
/// on-disk query type — vector search and full-text search, which arrive from
/// the scan API rather than from a filter expression — implements this
/// directly.
pub trait MemQuery: Debug + Send + Sync {
    /// Downcast hook. An index knows the concrete queries it answers and
    /// recognises them here; one it does not recognise it declines.
    fn as_any(&self) -> &dyn Any;
}

impl<T: AnyQuery> MemQuery for T {
    fn as_any(&self) -> &dyn Any {
        AnyQuery::as_any(self)
    }
}

/// A Lance scalar query that is already behind a trait object.
///
/// The blanket implementation above covers a concrete query type, but Rust
/// cannot re-point a `&dyn AnyQuery` at another trait, and the expression pass
/// hands out exactly that. Wrapping is the whole difference: an index sees the
/// same query through [`MemQuery::as_any`] either way, so nothing downstream
/// distinguishes them.
#[derive(Debug)]
pub struct ScalarQuery<'a>(pub &'a dyn AnyQuery);

impl MemQuery for ScalarQuery<'_> {
    fn as_any(&self) -> &dyn Any {
        self.0.as_any()
    }
}

/// A set of positions in one memtable.
///
/// Roaring rather than a `Vec` because posting lists intersect and union on
/// every compound filter, and because a set covering most of a large memtable
/// is a run rather than millions of entries.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PositionSet(RoaringTreemap);

impl PositionSet {
    /// The empty set.
    pub fn empty() -> Self {
        Self(RoaringTreemap::new())
    }

    /// Every position a reader may see, which is every position up to and
    /// including `max_visible`.
    pub fn all_visible(max_visible: RowPosition) -> Self {
        Self(RoaringTreemap::from_sorted_iter(0..=max_visible).expect("ascending"))
    }

    /// Whether the set holds no position.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// How many positions the set holds.
    pub fn len(&self) -> u64 {
        self.0.len()
    }

    /// Whether `position` is in the set.
    pub fn contains(&self, position: RowPosition) -> bool {
        self.0.contains(position)
    }

    /// Add one position.
    pub fn insert(&mut self, position: RowPosition) {
        self.0.insert(position);
    }

    /// The positions in ascending order.
    pub fn iter(&self) -> impl Iterator<Item = RowPosition> + '_ {
        self.0.iter()
    }

    /// Drop everything above `max_visible`.
    ///
    /// An index may hold rows a reader must not see yet, because a writer runs
    /// ahead of the watermark that publishes its rows.
    pub fn truncate_to(mut self, max_visible: RowPosition) -> Self {
        self.0.remove_range(max_visible.saturating_add(1)..);
        self
    }
}

impl FromIterator<RowPosition> for PositionSet {
    fn from_iter<I: IntoIterator<Item = RowPosition>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl From<RoaringTreemap> for PositionSet {
    fn from(map: RoaringTreemap) -> Self {
        Self(map)
    }
}

impl From<PositionSet> for Vec<RowPosition> {
    fn from(set: PositionSet) -> Self {
        set.0.into_iter().collect()
    }
}

impl std::ops::BitAnd for PositionSet {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl std::ops::BitOr for PositionSet {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// Which rows a filter matched, bracketed.
///
/// An index that locates rows knows the answer. One that only narrows knows a
/// superset. The two are the endpoints of one interval, so a compound filter
/// combines them elementwise and the caller asks one question of the result —
/// is it settled — rather than tracking which index was which.
///
/// * `certain` — rows that definitely match.
/// * `possible` — rows that might; a row outside it definitely does not.
///
/// A settled answer has the two equal. A narrowing index leaves `certain`
/// empty and the caller re-checks `possible`.
#[derive(Debug, Clone, PartialEq)]
pub struct MemSearchResult {
    /// Rows guaranteed to match.
    pub certain: PositionSet,
    /// Rows that may match. Nothing outside this set matches.
    pub possible: PositionSet,
}

impl MemSearchResult {
    /// A settled answer: exactly these rows match.
    pub fn exact(positions: PositionSet) -> Self {
        Self {
            certain: positions.clone(),
            possible: positions,
        }
    }

    /// A narrowed answer: nothing outside `positions` matches, and the caller
    /// re-checks what is inside.
    pub fn at_most(positions: PositionSet) -> Self {
        Self {
            certain: PositionSet::empty(),
            possible: positions,
        }
    }

    /// Nothing matches, and that is settled.
    pub fn empty() -> Self {
        Self::exact(PositionSet::empty())
    }

    /// Whether the answer is settled and needs no re-check.
    pub fn is_exact(&self) -> bool {
        self.certain == self.possible
    }

    /// Drop everything above `max_visible` from both endpoints.
    pub fn truncate_to(self, max_visible: RowPosition) -> Self {
        Self {
            certain: self.certain.truncate_to(max_visible),
            possible: self.possible.truncate_to(max_visible),
        }
    }
}

impl std::ops::BitAnd for MemSearchResult {
    type Output = Self;

    /// Rows matching both. Certain on both sides stays certain; possible on
    /// either side bounds the result.
    fn bitand(self, rhs: Self) -> Self {
        Self {
            certain: self.certain & rhs.certain,
            possible: self.possible & rhs.possible,
        }
    }
}

impl std::ops::BitOr for MemSearchResult {
    type Output = Self;

    /// Rows matching either.
    fn bitor(self, rhs: Self) -> Self {
        Self {
            certain: self.certain | rhs.certain,
            possible: self.possible | rhs.possible,
        }
    }
}

/// One row a ranked search returned.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedMatch {
    /// Where the row sits in the memtable.
    pub position: RowPosition,
    /// How well it scored. Lower is nearer for a vector distance; higher is
    /// better for a relevance score. Which way round is the query's business.
    pub score: f32,
    /// For a document nested inside a list, which element matched. `None` when
    /// the whole row is the document.
    pub element: Option<Vec<u32>>,
}

impl RankedMatch {
    /// A match on a whole row.
    pub fn new(position: RowPosition, score: f32) -> Self {
        Self {
            position,
            score,
            element: None,
        }
    }

    /// A match on one element of a nested document.
    pub fn nested(position: RowPosition, score: f32, element: Vec<u32>) -> Self {
        Self {
            position,
            score,
            element: Some(element),
        }
    }
}

/// What a search answered.
#[derive(Debug, Clone, PartialEq)]
pub enum MemMatches {
    /// A set of rows, bracketed. What a filter asks for.
    Filter(MemSearchResult),
    /// Rows in rank order, best first. What a vector or full-text search asks
    /// for, where a set with no order would lose the answer.
    Ranked(Vec<RankedMatch>),
}

impl MemMatches {
    /// A settled set of rows.
    pub fn exact(positions: impl IntoIterator<Item = RowPosition>) -> Self {
        Self::Filter(MemSearchResult::exact(positions.into_iter().collect()))
    }

    /// A narrowed set the caller re-checks.
    pub fn at_most(positions: impl IntoIterator<Item = RowPosition>) -> Self {
        Self::Filter(MemSearchResult::at_most(positions.into_iter().collect()))
    }

    /// Rows in rank order.
    pub fn ranked(matches: Vec<RankedMatch>) -> Self {
        Self::Ranked(matches)
    }

    /// The filter result, when that is what was asked for.
    pub fn as_filter(&self) -> Option<&MemSearchResult> {
        match self {
            Self::Filter(result) => Some(result),
            Self::Ranked(_) => None,
        }
    }

    /// The ranked matches, when that is what was asked for.
    pub fn as_ranked(&self) -> Option<&[RankedMatch]> {
        match self {
            Self::Ranked(matches) => Some(matches),
            Self::Filter(_) => None,
        }
    }
}

/// What a search needs to know besides the query itself.
#[derive(Debug, Clone, Copy)]
pub struct SearchContext {
    /// The highest position a reader may see. An index holds rows past it —
    /// a writer runs ahead of the watermark that publishes them — and must not
    /// return one.
    pub max_visible: RowPosition,
}

impl SearchContext {
    /// A search over everything visible up to `max_visible`.
    pub fn new(max_visible: RowPosition) -> Self {
        Self { max_visible }
    }
}

/// Nearest-neighbour search over one vector column.
///
/// A query the scan API raises rather than a filter expression, so it has no
/// on-disk counterpart to borrow and is defined here.
#[derive(Debug)]
pub struct VectorMemQuery {
    /// Exactly one query vector.
    pub vector: FixedSizeListArray,
    /// How many neighbours to return.
    pub k: usize,
    /// Search breadth, or `None` for the index's own default.
    pub ef: Option<usize>,
    /// The metric the caller asked for, or `None` to accept the index's own.
    ///
    /// A graph's metric is baked into its structure, so an index built for a
    /// different one declines rather than answering in the wrong space.
    pub distance_type: Option<DistanceType>,
}

impl VectorMemQuery {
    /// A query carrying only what index routing turns on.
    ///
    /// Planning asks an index whether it can answer before there is a query to
    /// hand it, and for nearest-neighbour search the answer turns on the metric
    /// alone. Passing a real query shape rather than a capability flag is what
    /// keeps routing open: a plugin decides for itself what it can serve.
    pub fn probe(distance_type: Option<DistanceType>) -> Self {
        Self {
            vector: FixedSizeListArray::new_null(
                Arc::new(Field::new("item", DataType::Float32, true)),
                1,
                0,
            ),
            k: 0,
            ef: None,
            distance_type,
        }
    }
}

impl MemQuery for VectorMemQuery {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A full-text query tree, with the recall and limit knobs a text search needs
/// and a filter predicate has no equivalent of.
#[derive(Debug)]
pub struct FtsMemQuery {
    /// The query.
    pub expr: super::fts::FtsQueryExpr,
    /// Recall, pruning and limit.
    pub options: super::fts::SearchOptions,
    /// Whether the caller is searching whole rows or the elements of a list.
    ///
    /// Two full-text indexes may cover one column at different granularities,
    /// and only the one that was built the way the query asks can answer it.
    pub granularity: DocumentGranularity,
}

impl FtsMemQuery {
    /// A query carrying only what index routing turns on.
    ///
    /// As [`VectorMemQuery::probe`]; for full-text search the answer turns on
    /// document granularity alone.
    pub fn probe(granularity: DocumentGranularity) -> Self {
        Self {
            expr: super::fts::FtsQueryExpr::Match {
                column: None,
                query: String::new(),
                operator: Default::default(),
                boost: 1.0,
            },
            options: super::fts::SearchOptions::new(),
            granularity,
        }
    }
}

impl MemQuery for FtsMemQuery {
    fn as_any(&self) -> &dyn Any {
        self
    }
}
