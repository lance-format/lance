// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What a memtable index can be asked, and what it answers.
//!
//! [`MemQuery`] is open to any query type, Lance's scalar index queries
//! included. [`MemMatches`] is either a filter's set of rows or a search's
//! ranked list.

use std::any::Any;
use std::fmt::Debug;

use arrow_array::FixedSizeListArray;
use lance_index::scalar::AnyQuery;
use lance_index::scalar::inverted::DocumentGranularity;
use lance_linalg::distance::DistanceType;
use roaring::RoaringTreemap;

use super::RowPosition;

/// A question put to a memtable index.
pub trait MemQuery: Debug + Send + Sync {
    /// The concrete query, for an index to recognise the ones it answers.
    fn as_any(&self) -> &dyn Any;
}

impl<T: AnyQuery> MemQuery for T {
    fn as_any(&self) -> &dyn Any {
        AnyQuery::as_any(self)
    }
}

/// A set of positions in one memtable.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PositionSet(RoaringTreemap);

impl PositionSet {
    /// The empty set.
    pub fn empty() -> Self {
        Self(RoaringTreemap::new())
    }

    /// Every position up to and including `max_visible`.
    pub fn all_visible(max_visible: RowPosition) -> Self {
        let mut positions = RoaringTreemap::new();
        positions.insert_range(0..=max_visible);
        Self(positions)
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
    pub fn truncate_to(mut self, max_visible: RowPosition) -> Self {
        if let Some(first_hidden) = max_visible.checked_add(1) {
            self.0.remove_range(first_hidden..);
        }
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

/// Which rows a filter matched, as two bounds like Lance's scalar
/// `SearchResult`: exact when they are equal, otherwise the caller re-checks
/// the rows in `at_most`.
#[derive(Debug, Clone, PartialEq)]
pub struct MemSearchResult {
    /// Rows that match.
    pub at_least: PositionSet,
    /// Rows that may match; nothing outside them does.
    pub at_most: PositionSet,
}

impl MemSearchResult {
    /// A settled answer: exactly these rows match.
    pub fn exact(positions: PositionSet) -> Self {
        Self {
            at_least: positions.clone(),
            at_most: positions,
        }
    }

    /// A narrowed answer: nothing outside `positions` matches, and the caller
    /// re-checks what is inside.
    pub fn at_most(positions: PositionSet) -> Self {
        Self {
            at_least: PositionSet::empty(),
            at_most: positions,
        }
    }

    /// Nothing matches, and that is settled.
    pub fn empty() -> Self {
        Self::exact(PositionSet::empty())
    }

    /// Whether the answer is settled and needs no re-check.
    pub fn is_exact(&self) -> bool {
        self.at_least == self.at_most
    }

    /// Drop everything above `max_visible` from both endpoints.
    pub fn truncate_to(self, max_visible: RowPosition) -> Self {
        Self {
            at_least: self.at_least.truncate_to(max_visible),
            at_most: self.at_most.truncate_to(max_visible),
        }
    }
}

impl std::ops::BitAnd for MemSearchResult {
    type Output = Self;

    /// Rows matching both.
    fn bitand(self, rhs: Self) -> Self {
        Self {
            at_least: self.at_least & rhs.at_least,
            at_most: self.at_most & rhs.at_most,
        }
    }
}

impl std::ops::BitOr for MemSearchResult {
    type Output = Self;

    /// Rows matching either.
    fn bitor(self, rhs: Self) -> Self {
        Self {
            at_least: self.at_least | rhs.at_least,
            at_most: self.at_most | rhs.at_most,
        }
    }
}

/// One row a ranked search returned.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedMatch {
    /// Where the row sits in the memtable.
    pub position: RowPosition,
    /// Its score: a distance, lower is better, or a full-text relevance,
    /// higher is better.
    pub score: f32,
    /// The indices of the list element that matched, outermost first, for a
    /// document nested in lists.
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
    /// The rows a filter matched.
    Filter(MemSearchResult),
    /// The rows a vector or full-text search found, best first.
    Ranked(Vec<RankedMatch>),
}

impl MemMatches {
    /// A settled set of rows.
    pub fn exact(positions: impl IntoIterator<Item = RowPosition>) -> Self {
        Self::Filter(MemSearchResult::exact(positions.into_iter().collect()))
    }

    /// Candidate rows the caller re-checks.
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
    /// The highest position a reader may see. An index may hold rows past it
    /// and must not return them.
    pub max_visible: RowPosition,
}

impl SearchContext {
    /// A search over everything visible up to `max_visible`.
    pub fn new(max_visible: RowPosition) -> Self {
        Self { max_visible }
    }
}

/// Nearest-neighbour search over one vector column.
#[derive(Debug)]
pub struct VectorMemQuery {
    /// Exactly one query vector.
    pub vector: FixedSizeListArray,
    /// How many neighbours to return.
    pub k: usize,
    /// Search breadth, or `None` for the index's own default.
    pub ef: Option<usize>,
    /// The metric the caller asked for, or `None` to accept the index's own.
    pub distance_type: Option<DistanceType>,
}

impl MemQuery for VectorMemQuery {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A full-text query tree with its search options.
#[derive(Debug)]
pub struct FtsMemQuery {
    /// The query.
    pub expr: super::fts::FtsQueryExpr,
    /// Recall, pruning and limit.
    pub options: super::fts::SearchOptions,
    /// Whether the search covers whole rows or the elements of a list.
    pub granularity: DocumentGranularity,
}

impl FtsMemQuery {
    /// A query carrying only the granularity, for asking an index whether it
    /// serves one.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn set(positions: &[RowPosition]) -> PositionSet {
        positions.iter().copied().collect()
    }

    /// Combining answers combines each bound, so a re-check still covers every
    /// row the combination may match.
    #[test]
    fn answers_combine_bound_by_bound() {
        let settled = MemSearchResult::exact(set(&[1, 2]));
        let candidates = MemSearchResult::at_most(set(&[2, 3]));
        assert_eq!(
            settled.clone() & candidates.clone(),
            MemSearchResult {
                at_least: PositionSet::empty(),
                at_most: set(&[2]),
            }
        );
        assert_eq!(
            settled | candidates,
            MemSearchResult {
                at_least: set(&[1, 2]),
                at_most: set(&[1, 2, 3]),
            }
        );
        assert!(MemSearchResult::empty().is_exact());
    }

    /// Truncating drops the hidden rows from both bounds, and the last position
    /// hides nothing.
    #[test]
    fn truncating_keeps_only_the_visible_rows() {
        let answer = MemSearchResult::exact(set(&[0, 5, u64::MAX]));
        assert_eq!(
            answer.clone().truncate_to(4),
            MemSearchResult::exact(set(&[0]))
        );
        assert_eq!(answer.clone().truncate_to(u64::MAX), answer);
    }
}
