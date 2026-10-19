// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Greedy search over one partition's graph: Algorithm 1 of the DiskANN paper.
//!
//! A build runs it once per vertex to collect the visited set a prune works
//! from, and a query runs it to answer, so both share [`SearchList`].

use std::cell::Cell;
use std::collections::BinaryHeap;

use lance_core::{Error, Result};

use super::graph::Adjacency;
use crate::vector::graph::{OrderedFloat, OrderedNode};
use crate::vector::storage::DistCalculator;

/// Counts distance computations.
#[derive(Debug, Default)]
pub struct Comparisons(Cell<u64>);

impl Comparisons {
    #[inline]
    pub fn record(&self, count: u64) {
        self.0.set(self.0.get().saturating_add(count));
    }

    pub fn get(&self) -> u64 {
        self.0.get()
    }
}

/// Reusable visited marks.
///
/// Stamped with a generation rather than cleared per search: a build runs one
/// search per vertex. A byte per vertex rather than four leaves more cache to
/// the codes a walk reads, at the price of clearing the buffer once every 255
/// searches.
#[derive(Debug)]
pub struct SearchScratch {
    seen: Vec<u8>,
    generation: u8,
}

impl SearchScratch {
    pub fn new(num_vertices: usize) -> Self {
        Self {
            seen: vec![0; num_vertices],
            generation: 0,
        }
    }

    /// Start a search: every mark from the previous one stops counting.
    pub(crate) fn begin(&mut self) {
        self.generation = match self.generation.checked_add(1) {
            Some(next) => next,
            // The whole buffer, not one partition's share of it: a reused
            // scratch is as long as the largest partition it has served.
            None => {
                self.seen.fill(0);
                1
            }
        };
    }

    /// Mark `id` as reached, returning whether this search had not reached it.
    #[inline]
    pub(crate) fn mark(&mut self, id: u32) -> bool {
        let slot = &mut self.seen[id as usize];
        if *slot == self.generation {
            false
        } else {
            *slot = self.generation;
            true
        }
    }

    /// Make room for a graph of `num_vertices`. A new buffer rather than a
    /// grown one: every mark is stale by the next [`Self::begin`] anyway.
    pub(crate) fn cover(&mut self, num_vertices: usize) {
        if self.seen.len() < num_vertices {
            self.seen = vec![0; num_vertices];
        }
    }

    pub(crate) fn capacity(&self) -> usize {
        self.seen.len()
    }
}

/// A vertex in the search list, and whether its out-edges have been followed.
#[derive(Debug, Clone)]
struct Entry {
    node: OrderedNode,
    expanded: bool,
}

/// When a walk stops before its list runs out of vertices to expand: once none
/// of them is among its `rank` nearest candidates or nearer than `1 + margin`
/// times the length to the `rank`-th. This is adaptive beam search
/// (Al-Jazzazi et al., NeurIPS 2025) with `gamma = margin` and `k = rank`.
///
/// The nearest `keep` candidates stay in the list whatever their distance,
/// because a re-score is owed them. `keep` is at least `rank`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StopRule {
    pub margin: f32,
    pub rank: usize,
    pub keep: usize,
}

/// A [`StopRule`] as a list applies it.
#[derive(Debug, Clone, Copy)]
struct Margin {
    rank: usize,
    keep: usize,
    /// `(1 + margin)^2`: the rule is about lengths, and the distances a walk
    /// measures are squared.
    factor: f32,
    /// `factor` times the distance of the entry at `rank - 1`. It only falls,
    /// because that entry can only be replaced by a nearer one, which is what
    /// makes it safe to drop what lies beyond it.
    bar: OrderedFloat,
}

/// `L`: the beam a walk keeps, nearest first.
///
/// Among equal distances the first offered stays, so a walk is deterministic
/// for a given offer order.
#[derive(Debug)]
pub struct SearchList {
    list: Vec<Entry>,
    size: usize,
    /// Every entry before this one has been expanded.
    cursor: usize,
    margin: Option<Margin>,
}

impl SearchList {
    /// A list of at most `search_list_size` entries over `num_vertices`.
    ///
    /// Capped by the graph too: `search_list_size` may come from a caller that
    /// passed `usize::MAX`, which would otherwise reach the allocator.
    pub fn new(search_list_size: usize, num_vertices: usize) -> Self {
        Self {
            list: Vec::with_capacity(search_list_size.min(num_vertices).saturating_add(1)),
            size: search_list_size,
            cursor: 0,
            margin: None,
        }
    }

    /// A list that stops its walk by `stop`, capped at `search_list_size`.
    ///
    /// The caller checks `1 <= rank <= keep <= search_list_size`.
    pub(crate) fn with_margin(
        search_list_size: usize,
        num_vertices: usize,
        stop: StopRule,
    ) -> Self {
        debug_assert!(
            1 <= stop.rank && stop.rank <= stop.keep && stop.keep <= search_list_size,
            "a stop rule at rank {} keeping {} does not fit a list of {search_list_size}",
            stop.rank,
            stop.keep
        );
        let widened = 1.0 + stop.margin;
        Self {
            margin: Some(Margin {
                rank: stop.rank,
                keep: stop.keep,
                factor: widened * widened,
                bar: OrderedFloat(f32::INFINITY),
            }),
            ..Self::new(search_list_size, num_vertices)
        }
    }

    pub fn len(&self) -> usize {
        self.list.len()
    }

    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    /// Whether an offer at `distance` could be kept: false once the list is
    /// full and its back is no farther. Lets a caller skip work it would do
    /// only for a vertex the list keeps.
    #[inline]
    pub fn would_keep(&self, distance: f32) -> bool {
        self.list.len() < self.size
            || self
                .list
                .last()
                .is_some_and(|back| OrderedFloat(distance) < back.node.dist)
    }

    /// Offer a vertex at `distance`, keeping it if it beats the back of the
    /// list.
    ///
    /// Nothing checks whether `id` is already in the list: the caller marks it
    /// in its [`SearchScratch`] first, so a vertex is measured and offered
    /// once.
    ///
    /// A list with a [`StopRule`] also turns away a vertex that would land past
    /// its `keep` nearest and is not nearer than the bar, and cuts off every
    /// entry past `keep` that is not nearer than it either: the rule would
    /// never expand them and the bar only falls.
    pub fn offer(&mut self, id: u32, distance: f32) {
        let distance = OrderedFloat(distance);
        if self.list.len() >= self.size
            && self
                .list
                .last()
                .is_some_and(|back| back.node.dist <= distance)
        {
            return;
        }
        if let Some(margin) = &self.margin
            && self.list.len() >= margin.keep
            && margin.bar <= distance
            && self.list[margin.keep - 1].node.dist <= distance
        {
            return;
        }
        let at = self
            .list
            .partition_point(|entry| entry.node.dist <= distance);
        if at >= self.size {
            return;
        }
        self.list.insert(
            at,
            Entry {
                node: OrderedNode::new(id, distance),
                expanded: false,
            },
        );
        self.list.truncate(self.size);
        self.cursor = self.cursor.min(at);
        if let Some(margin) = &mut self.margin {
            if at < margin.rank && self.list.len() >= margin.rank {
                margin.bar = OrderedFloat(margin.factor * self.list[margin.rank - 1].node.dist.0);
            }
            // An insert before `keep` pushes the entry at `keep - 1` past it,
            // and a new bar may have passed over some of those already there.
            if at < margin.keep && self.list.len() > margin.keep {
                let nearer =
                    self.list[margin.keep..].partition_point(|entry| entry.node.dist < margin.bar);
                self.list.truncate(margin.keep + nearer);
            }
        }
        debug_assert!(
            self.cursor <= self.list.len(),
            "a search list's cursor {} ran past its {} entries",
            self.cursor,
            self.list.len()
        );
    }

    /// Measure and offer every id in `ids`, asking for the data of the one
    /// `ahead` further on before the current one is measured, and hand each
    /// measured `(id, distance)` to `also`.
    ///
    /// Every id must be a vertex of `calculator`'s store and already marked,
    /// as [`Self::offer`] requires. `ahead` of zero asks for nothing. The ask
    /// is [`DistCalculator::prefetch`], a hint that cannot change a distance:
    /// a hop knows every id it will measure before it measures any, which no
    /// hardware prefetcher can guess.
    pub fn offer_all_with(
        &mut self,
        ids: &[u32],
        calculator: &impl DistCalculator,
        ahead: usize,
        mut also: impl FnMut(u32, f32),
    ) {
        for id in ids.iter().take(ahead) {
            calculator.prefetch(*id);
        }
        for (position, id) in ids.iter().enumerate() {
            if ahead != 0
                && let Some(later) = ids.get(position.saturating_add(ahead))
            {
                calculator.prefetch(*later);
            }
            let distance = calculator.distance(*id);
            self.offer(*id, distance);
            also(*id, distance);
        }
    }

    /// [`Self::offer_all_with`] for a walk that keeps no second list.
    pub fn offer_all(&mut self, ids: &[u32], calculator: &impl DistCalculator, ahead: usize) {
        self.offer_all_with(ids, calculator, ahead, |_, _| {});
    }

    /// The nearest vertex whose out-edges have not been followed, marked as
    /// followed.
    ///
    /// Called `n` times without an [`Self::offer`] in between, it yields the
    /// `n` nearest unexpanded vertices. With a [`StopRule`] it is also `None`
    /// when that vertex is past the `rank` nearest and not nearer than the
    /// bar, until an offer lands a nearer one.
    pub fn next_unexpanded(&mut self) -> Option<OrderedNode> {
        let position = self.cursor
            + self.list[self.cursor..]
                .iter()
                .position(|entry| !entry.expanded)?;
        if let Some(margin) = &self.margin
            && position >= margin.rank
            && margin.bar <= self.list[position].node.dist
        {
            self.cursor = position;
            return None;
        }
        self.list[position].expanded = true;
        self.cursor = position + 1;
        Some(self.list[position].node.clone())
    }

    /// The list itself, nearest first.
    pub fn into_candidates(self) -> Vec<OrderedNode> {
        self.list.into_iter().map(|entry| entry.node).collect()
    }
}

#[derive(Debug)]
pub struct SearchResult {
    /// The search list, nearest first: `L` in the paper.
    pub candidates: Vec<OrderedNode>,
    /// Vertices whose out-edges were followed, in the order they were
    /// followed: `V` in the paper, and what a build's prune works from. A
    /// vertex pushed out of the list before its turn is not in it, which keeps
    /// the prune's input bounded by the search's work.
    pub visited: Vec<OrderedNode>,
}

/// Walk `graph` from `entry_point` towards `query`, keeping `search_list_size`
/// candidates, and asking for each neighbor's data `prefetch_ahead` neighbors
/// before it is measured.
///
/// The neighbors of each expanded vertex that the walk had not reached are
/// collected first and then measured in the order the graph lists them, which
/// is what puts the ids in hand before the loads.
pub fn greedy_search_with_lookahead(
    graph: &impl Adjacency,
    query: &impl DistCalculator,
    entry_point: u32,
    search_list_size: usize,
    prefetch_ahead: usize,
    scratch: &mut SearchScratch,
    comparisons: &Comparisons,
) -> Result<SearchResult> {
    let mut list = start_walk(
        graph,
        query,
        entry_point,
        search_list_size,
        scratch,
        comparisons,
    )?;
    let mut visited = Vec::new();
    let mut fresh = Vec::new();

    while let Some(nearest_unexpanded) = list.next_unexpanded() {
        fresh.clear();
        fresh.extend(
            graph
                .neighbors(nearest_unexpanded.id)
                .iter()
                .copied()
                .filter(|neighbor| scratch.mark(*neighbor)),
        );
        visited.push(nearest_unexpanded);
        comparisons.record(fresh.len() as u64);
        list.offer_all(&fresh, query, prefetch_ahead);
    }

    Ok(SearchResult {
        candidates: list.into_candidates(),
        visited,
    })
}

/// Check a build-time walk's inputs and start it: the scratch begun, the entry
/// point marked, measured and offered.
fn start_walk(
    graph: &impl Adjacency,
    query: &impl DistCalculator,
    entry_point: u32,
    search_list_size: usize,
    scratch: &mut SearchScratch,
    comparisons: &Comparisons,
) -> Result<SearchList> {
    if search_list_size == 0 {
        return Err(Error::invalid_input(
            "Vamana search list size must be greater than zero".to_string(),
        ));
    }
    let num_vertices = graph.num_vertices();
    if entry_point as usize >= num_vertices {
        return Err(Error::invalid_input(format!(
            "Vamana entry point {entry_point} is outside a graph of {num_vertices} vertices"
        )));
    }
    if scratch.capacity() < num_vertices {
        return Err(Error::invalid_input(format!(
            "Vamana search scratch holds {} vertices but the graph has {num_vertices}",
            scratch.capacity()
        )));
    }

    scratch.begin();
    scratch.mark(entry_point);
    comparisons.record(1);
    let mut list = SearchList::new(search_list_size, num_vertices);
    list.offer(entry_point, query.distance(entry_point));
    Ok(list)
}

/// A calculator whose ask names at most `lines` cache lines of a vector,
/// stepping a line at a time from its start, where Lance's flat calculator asks
/// for all of it.
///
/// A whole vector asked for ahead pays only while it is short: two neighbors
/// ahead, it took a build's search to 0.735 of its time per distance on SIFT
/// (8 lines a vector) but to 1.155 on GIST (60), and the first eight lines took
/// GIST to 0.910.
///
/// `vectors` must be the buffer the wrapped calculator reads, vector `id` at
/// `id * dimension`. An id past the buffer is asked for nothing, and nothing is
/// asked for off x86_64. Every other answer is the wrapped calculator's.
pub struct LeadingLines<'a, C> {
    calculator: &'a C,
    vectors: &'a [f32],
    dimension: usize,
    lines: usize,
}

impl<'a, C: DistCalculator> LeadingLines<'a, C> {
    pub fn new(calculator: &'a C, vectors: &'a [f32], dimension: usize, lines: usize) -> Self {
        Self {
            calculator,
            vectors,
            dimension,
            lines,
        }
    }
}

impl<C: DistCalculator> DistCalculator for LeadingLines<'_, C> {
    fn distance(&self, id: u32) -> f32 {
        self.calculator.distance(id)
    }

    fn distance_all(&self, k_hint: usize) -> Vec<f32> {
        self.calculator.distance_all(k_hint)
    }

    fn distance_all_with_scratch(
        &self,
        k_hint: usize,
        dists: &mut Vec<f32>,
        u16_scratch: &mut Vec<u16>,
        u8_scratch: &mut Vec<u8>,
        u32_scratch: &mut Vec<u32>,
    ) {
        self.calculator.distance_all_with_scratch(
            k_hint,
            dists,
            u16_scratch,
            u8_scratch,
            u32_scratch,
        );
    }

    fn has_exact_topk_scan(&self) -> bool {
        self.calculator.has_exact_topk_scan()
    }

    fn prefetch(&self, id: u32) {
        let vector = (id as usize)
            .checked_mul(self.dimension)
            .and_then(|start| self.vectors.get(start..))
            .and_then(|rest| rest.get(..self.dimension));
        if let Some(vector) = vector {
            ask_leading_lines(vector, self.lines);
        }
    }

    fn accumulate_topk_with_scratch(
        &self,
        k: usize,
        lower_bound: Option<f32>,
        upper_bound: Option<f32>,
        row_id: impl Fn(u32) -> u64,
        res: &mut BinaryHeap<OrderedNode<u64>>,
        dists: &mut Vec<f32>,
        u16_scratch: &mut Vec<u16>,
        u8_scratch: &mut Vec<u8>,
        u32_scratch: &mut Vec<u32>,
    ) {
        self.calculator.accumulate_topk_with_scratch(
            k,
            lower_bound,
            upper_bound,
            row_id,
            res,
            dists,
            u16_scratch,
            u8_scratch,
            u32_scratch,
        );
    }

    fn accumulate_filtered_topk_with_scratch(
        &self,
        k: usize,
        lower_bound: Option<f32>,
        upper_bound: Option<f32>,
        row_ids: impl Iterator<Item = (u32, u64)>,
        accept_row: impl Fn(u64) -> bool,
        res: &mut BinaryHeap<OrderedNode<u64>>,
        dists: &mut Vec<f32>,
        u16_scratch: &mut Vec<u16>,
        u8_scratch: &mut Vec<u8>,
        u32_scratch: &mut Vec<u32>,
    ) {
        self.calculator.accumulate_filtered_topk_with_scratch(
            k,
            lower_bound,
            upper_bound,
            row_ids,
            accept_row,
            res,
            dists,
            u16_scratch,
            u8_scratch,
            u32_scratch,
        );
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
thread_local! {
    static ASKED: std::cell::RefCell<Option<Vec<usize>>> = const {
        std::cell::RefCell::new(None)
    };
}

/// Start recording the lines asked for on this thread.
#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) fn record_asks() {
    ASKED.with(|asked| *asked.borrow_mut() = Some(Vec::new()));
}

/// The lines asked for on this thread since [`record_asks`], which stops
/// recording.
#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) fn recorded_asks() -> Vec<usize> {
    ASKED.with(|asked| asked.borrow_mut().take().unwrap_or_default())
}

#[cfg(target_arch = "x86_64")]
const CACHE_LINE: usize = 64;

#[inline]
fn ask_leading_lines(vector: &[f32], lines: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        let end = vector.as_ptr_range().end.cast::<i8>();
        let mut line = vector.as_ptr().cast::<i8>();
        for _ in 0..lines {
            if line >= end {
                break;
            }
            #[cfg(test)]
            ASKED.with(|asked| {
                if let Some(asked) = asked.borrow_mut().as_mut() {
                    asked.push(line as usize);
                }
            });
            // SAFETY: SSE is part of every x86_64 processor, and a prefetch
            // never faults whatever the address. `wrapping_add` because the
            // step past the last line leaves the allocation.
            unsafe { _mm_prefetch::<_MM_HINT_T0>(line) };
            line = line.wrapping_add(CACHE_LINE);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (vector, lines);
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::FixedSizeListArray;
    use arrow_array::types::Float32Type;
    use lance_linalg::distance::DistanceType;
    use rand::rngs::SmallRng;
    use rand::seq::SliceRandom;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::vector::flat::storage::FlatFloatStorage;
    use crate::vector::storage::VectorStore;
    use crate::vector::vamana::graph::PartitionGraph;

    fn line_storage(num_vertices: usize) -> FlatFloatStorage {
        let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            (0..num_vertices).map(|i| Some(vec![Some(i as f32)])),
            1,
        );
        FlatFloatStorage::new(vectors, DistanceType::L2)
    }

    fn path_graph(num_vertices: usize) -> PartitionGraph {
        let adjacency = (0..num_vertices as u32)
            .map(|vertex| {
                if vertex + 1 < num_vertices as u32 {
                    vec![vertex + 1]
                } else {
                    vec![]
                }
            })
            .collect::<Vec<_>>();
        PartitionGraph::try_new(4, (0..num_vertices as u64).collect(), &adjacency).unwrap()
    }

    fn search(
        graph: &PartitionGraph,
        storage: &FlatFloatStorage,
        target: f32,
        entry_point: u32,
        search_list_size: usize,
    ) -> SearchResult {
        let query = Arc::new(arrow_array::Float32Array::from(vec![target]));
        let calculator = storage.dist_calculator(query, 0.0);
        let mut scratch = SearchScratch::new(graph.len());
        greedy_search_with_lookahead(
            graph,
            &calculator,
            entry_point,
            search_list_size,
            2,
            &mut scratch,
            &Comparisons::default(),
        )
        .unwrap()
    }

    #[test]
    fn a_walk_along_a_path_reaches_the_far_end() {
        let storage = line_storage(10);
        let graph = path_graph(10);
        let found = search(&graph, &storage, 9.0, 0, 2);
        assert_eq!(found.candidates[0].id, 9);
        assert_eq!(found.visited.len(), 10);
    }

    #[test]
    fn a_walk_cannot_leave_its_component() {
        let storage = line_storage(6);
        let graph = PartitionGraph::try_new(
            2,
            (0..6).collect(),
            &[vec![1], vec![0], vec![], vec![4], vec![5], vec![]],
        )
        .unwrap();
        let found = search(&graph, &storage, 5.0, 0, 6);
        let mut reached = found
            .candidates
            .iter()
            .map(|node| node.id)
            .collect::<Vec<_>>();
        reached.sort_unstable();
        assert_eq!(reached, vec![0, 1]);
    }

    #[test]
    fn bad_walk_inputs_are_rejected() {
        let storage = line_storage(4);
        let graph = path_graph(4);
        let query = Arc::new(arrow_array::Float32Array::from(vec![1.0]));
        let calculator = storage.dist_calculator(query, 0.0);
        let comparisons = Comparisons::default();
        let mut scratch = SearchScratch::new(4);
        assert!(
            greedy_search_with_lookahead(&graph, &calculator, 4, 2, 0, &mut scratch, &comparisons)
                .is_err()
        );
        assert!(
            greedy_search_with_lookahead(&graph, &calculator, 0, 0, 0, &mut scratch, &comparisons)
                .is_err()
        );
        let mut small = SearchScratch::new(3);
        assert!(
            greedy_search_with_lookahead(&graph, &calculator, 0, 2, 0, &mut small, &comparisons)
                .is_err()
        );
    }

    #[test]
    fn comparisons_count_each_vertex_once() {
        let storage = line_storage(10);
        let graph = PartitionGraph::try_new(
            9,
            (0..10).collect(),
            &(0..10u32)
                .map(|vertex| (0..10u32).filter(|other| *other != vertex).collect())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let query = Arc::new(arrow_array::Float32Array::from(vec![3.0]));
        let calculator = storage.dist_calculator(query, 0.0);
        let comparisons = Comparisons::default();
        let mut scratch = SearchScratch::new(10);
        greedy_search_with_lookahead(&graph, &calculator, 0, 3, 0, &mut scratch, &comparisons)
            .unwrap();
        assert_eq!(comparisons.get(), 10);
    }

    #[test]
    fn reused_scratch_does_not_leak_between_searches() {
        let storage = line_storage(10);
        let graph = path_graph(10);
        let query = Arc::new(arrow_array::Float32Array::from(vec![9.0]));
        let calculator = storage.dist_calculator(query, 0.0);
        let mut scratch = SearchScratch::new(10);
        // Past the wrap of the generation, twice.
        for _ in 0..600 {
            let found = greedy_search_with_lookahead(
                &graph,
                &calculator,
                0,
                2,
                0,
                &mut scratch,
                &Comparisons::default(),
            )
            .unwrap();
            assert_eq!(found.candidates[0].id, 9);
            assert_eq!(found.visited.len(), 10);
        }
    }

    #[test]
    fn a_full_list_keeps_the_first_of_equals() {
        let mut list = SearchList::new(2, 10);
        list.offer(5, 1.0);
        list.offer(6, 1.0);
        list.offer(7, 1.0);
        assert_eq!(
            list.into_candidates()
                .iter()
                .map(|node| node.id)
                .collect::<Vec<_>>(),
            vec![5, 6]
        );
    }

    #[test]
    fn would_keep_agrees_with_offer() {
        let mut list = SearchList::new(2, 10);
        assert!(list.would_keep(5.0));
        list.offer(1, 2.0);
        list.offer(2, 3.0);
        assert!(!list.would_keep(3.0));
        assert!(list.would_keep(2.5));
    }

    /// [`SearchList`] without a stop rule, a cursor or a turn-away at the
    /// back: a binary search on every offer and a scan from the front on
    /// every expansion.
    struct PlainList {
        list: Vec<(OrderedNode, bool)>,
        size: usize,
    }

    impl PlainList {
        fn new(size: usize) -> Self {
            Self {
                list: Vec::new(),
                size,
            }
        }

        fn offer(&mut self, id: u32, distance: f32) {
            let distance = OrderedFloat(distance);
            let at = self.list.partition_point(|(node, _)| node.dist <= distance);
            if at >= self.size {
                return;
            }
            self.list
                .insert(at, (OrderedNode::new(id, distance), false));
            self.list.truncate(self.size);
        }

        fn next_unexpanded(&mut self) -> Option<OrderedNode> {
            let position = self.list.iter().position(|(_, expanded)| !expanded)?;
            self.list[position].1 = true;
            Some(self.list[position].0.clone())
        }
    }

    /// A node down to the bits of its distance: `OrderedFloat` derives its
    /// equality, under which a NaN is not equal to itself.
    fn bits(node: OrderedNode) -> (u32, u32) {
        (node.id, node.dist.0.to_bits())
    }

    fn held(list: &SearchList) -> Vec<(u32, u32, bool)> {
        list.list
            .iter()
            .map(|entry| (entry.node.id, entry.node.dist.0.to_bits(), entry.expanded))
            .collect()
    }

    fn plainly_held(list: &PlainList) -> Vec<(u32, u32, bool)> {
        list.list
            .iter()
            .map(|(node, expanded)| (node.id, node.dist.0.to_bits(), *expanded))
            .collect()
    }

    /// [`PlainList`] ruled by a [`StopRule`] straight from its definition:
    /// nothing is thrown away, and the rule is consulted only when a vertex is
    /// asked for - expand the nearest unexpanded one if it is among the `rank`
    /// nearest or nearer than `(1 + margin)^2` times the `rank`-th's distance.
    struct RuledList {
        plain: PlainList,
        rank: usize,
        keep: usize,
        factor: f32,
    }

    impl RuledList {
        fn new(size: usize, stop: StopRule) -> Self {
            Self {
                plain: PlainList::new(size),
                rank: stop.rank,
                keep: stop.keep,
                factor: (1.0 + stop.margin).powi(2),
            }
        }

        fn offer(&mut self, id: u32, distance: f32) {
            self.plain.offer(id, distance);
        }

        fn bar(&self) -> Option<OrderedFloat> {
            self.plain
                .list
                .get(self.rank - 1)
                .map(|(node, _)| OrderedFloat(self.factor * node.dist.0))
        }

        fn next_unexpanded(&mut self) -> Option<OrderedNode> {
            let position = self.plain.list.iter().position(|(_, expanded)| !expanded)?;
            let beyond = self
                .bar()
                .is_some_and(|bar| bar <= self.plain.list[position].0.dist);
            if position >= self.rank && beyond {
                return None;
            }
            self.plain.list[position].1 = true;
            Some(self.plain.list[position].0.clone())
        }

        /// What the rule has any use for: the nearest `keep`, and after them
        /// whatever is nearer than the bar.
        fn kept(&self) -> Vec<(u32, u32, bool)> {
            let held = plainly_held(&self.plain);
            let cut = match self.bar() {
                Some(bar) if held.len() > self.keep => {
                    self.keep
                        + self.plain.list[self.keep..]
                            .iter()
                            .take_while(|(node, _)| node.dist < bar)
                            .count()
                }
                _ => held.len(),
            };
            held[..cut].to_vec()
        }
    }

    /// A list with a stop rule keeps the front of what the ruled plain list
    /// holds, and expands exactly what the rule allows, after every offer and
    /// every expansion.
    ///
    /// The front and not the whole: a list may throw away what the rule has no
    /// use for, and what it keeps has to be exactly the rest, or a candidate
    /// the re-score is owed - or one the walk should still expand - went
    /// missing. At a margin of zero it also expands exactly what the plain list
    /// of length `rank` does, whatever it keeps, and at `rank == keep` it is
    /// that list entry for entry: the walk as it was.
    ///
    /// The distances are chosen so that bars of 1.5625, 2.25 and 4 times one of
    /// them land exactly on another, and so exercise the strict side of every
    /// comparison. They are finite, as every distance a walk measures is.
    #[test]
    fn a_list_with_a_stop_rule_is_the_front_of_the_list_it_rules() {
        const DISTANCES: [f32; 12] = [
            -1.0, 0.0, 0.5, 1.0, 2.0, 2.25, 4.0, 4.5, 6.25, 9.0, 16.0, 25.0,
        ];
        let mut rng = SmallRng::seed_from_u64(13);
        // Where each half of the rule acted in the list itself: a stop with
        // unexpanded entries still held, and an insert that did not lengthen a
        // list the cap had room in. A fixture that never reached one of them
        // would pass the comparisons below without testing it.
        let (mut stopped, mut cut) = (0usize, 0usize);
        for (rank, keep) in [(1, 1), (1, 3), (2, 5), (3, 3), (5, 16), (16, 16)] {
            for size in [keep, keep + 1, 64, usize::MAX] {
                for margin in [0.0f32, 0.25, 0.5, 1.0] {
                    let stop = StopRule { margin, rank, keep };
                    for sequence in 0..20 {
                        let mut list = SearchList::with_margin(size, 256, stop);
                        let mut ruled = RuledList::new(size, stop);
                        let mut plain = (margin == 0.0).then(|| PlainList::new(rank));
                        for step in 0..100 {
                            let context = || {
                                format!(
                                    "rank {rank}, keep {keep}, L = {size}, margin {margin}, \
                                     sequence {sequence}, step {step}"
                                )
                            };
                            if rng.random_bool(0.7) {
                                let id = rng.random_range(0..256);
                                let distance = DISTANCES[rng.random_range(0..DISTANCES.len())];
                                let before = held(&list);
                                list.offer(id, distance);
                                let after = held(&list);
                                cut += usize::from(
                                    after != before
                                        && after.len() <= before.len()
                                        && before.len() < size,
                                );
                                ruled.offer(id, distance);
                                if let Some(plain) = plain.as_mut() {
                                    plain.offer(id, distance);
                                }
                            } else {
                                let expansions = if rng.random_bool(0.25) { 4 } else { 1 };
                                for _ in 0..expansions {
                                    let expected = ruled.next_unexpanded().map(bits);
                                    let expanded = list.next_unexpanded().map(bits);
                                    stopped += usize::from(
                                        expanded.is_none()
                                            && list.list.iter().any(|entry| !entry.expanded),
                                    );
                                    assert_eq!(
                                        expanded,
                                        expected,
                                        "{}: the lists expanded different vertices",
                                        context()
                                    );
                                    if let Some(plain) = plain.as_mut() {
                                        assert_eq!(
                                            plain.next_unexpanded().map(bits),
                                            expected,
                                            "{}: a margin of zero expanded what the list of \
                                             length rank does not",
                                            context()
                                        );
                                    }
                                }
                            }
                            let kept = ruled.kept();
                            assert_eq!(
                                held(&list),
                                kept,
                                "{}: the lists hold different entries",
                                context()
                            );
                            if let Some(plain) = plain.as_ref().filter(|_| rank == keep) {
                                assert_eq!(
                                    plainly_held(plain),
                                    kept,
                                    "{}: a margin of zero kept what the list of that length \
                                     does not",
                                    context()
                                );
                            }
                        }
                    }
                }
            }
        }
        assert!(
            stopped > 0 && cut > 0,
            "the list stopped {stopped} walks with entries left and cut {cut} times, so one half \
             of the rule was never exercised"
        );
    }

    /// A calculator that reads every distance off a table.
    struct Table(Vec<f32>);

    impl DistCalculator for Table {
        fn distance(&self, id: u32) -> f32 {
            self.0[id as usize]
        }

        fn distance_all(&self, _k_hint: usize) -> Vec<f32> {
            self.0.clone()
        }
    }

    /// A graph the write path accepts, chosen at random: out-edges distinct,
    /// none a self-edge, degrees anywhere from zero to `max_degree`.
    fn random_graph(num_vertices: usize, max_degree: usize, rng: &mut SmallRng) -> PartitionGraph {
        let adjacency = (0..num_vertices as u32)
            .map(|vertex| {
                let mut others = (0..num_vertices as u32)
                    .filter(|other| *other != vertex)
                    .collect::<Vec<_>>();
                others.shuffle(rng);
                others.truncate(rng.random_range(0..=max_degree));
                others
            })
            .collect::<Vec<_>>();
        PartitionGraph::try_new(
            max_degree as u32,
            (0..num_vertices as u64).collect(),
            &adjacency,
        )
        .unwrap()
    }

    /// Algorithm 1 with a stop rule, one vertex at a time.
    fn ruled_walk(
        graph: &PartitionGraph,
        table: &Table,
        entry_point: u32,
        stop: StopRule,
        search_list_size: usize,
    ) -> (Vec<u32>, u64) {
        let mut scratch = SearchScratch::new(graph.len());
        scratch.begin();
        scratch.mark(entry_point);
        let mut list = SearchList::with_margin(search_list_size, graph.len(), stop);
        list.offer(entry_point, table.distance(entry_point));
        let mut comparisons = 1;
        let mut expanded = Vec::new();
        while let Some(node) = list.next_unexpanded() {
            expanded.push(node.id);
            for neighbor in graph.neighbors(node.id) {
                if scratch.mark(*neighbor) {
                    comparisons += 1;
                    list.offer(*neighbor, table.distance(*neighbor));
                }
            }
        }
        (expanded, comparisons)
    }

    /// A wider margin walks on from where a narrower one stopped: the same
    /// vertices expanded in the same order, then more, and never fewer
    /// distances.
    ///
    /// This is what makes a sweep over the margin a sweep over one walk's
    /// stopping points rather than over different walks, so that recall and
    /// work move one way along it. It holds for one vertex a hop, under any cap
    /// both walks share - one just past `keep` binds all the time here - and for
    /// distances of at least zero: below zero a wider margin puts the bar lower,
    /// not higher. A hop of four can take vertices a narrower margin left, and
    /// the two walks then part.
    #[test]
    fn a_wider_margin_walks_on_from_where_a_narrower_one_stopped() {
        const DISTANCES: [f32; 6] = [0.0, 1.0, 1.0, 2.0, 3.0, 5.0];
        let mut rng = SmallRng::seed_from_u64(17);
        let mut further = 0usize;
        for num_vertices in [2, 17, 64, 200] {
            for _ in 0..24 {
                let graph = random_graph(num_vertices, 4, &mut rng);
                let table = Table(
                    (0..num_vertices)
                        .map(|_| DISTANCES[rng.random_range(0..DISTANCES.len())])
                        .collect(),
                );
                let entry_point = rng.random_range(0..num_vertices as u32);
                for (rank, keep) in [(1, 1), (2, 4), (5, 5)] {
                    for size in [keep + 1, usize::MAX] {
                        let mut narrower: Option<(Vec<u32>, u64)> = None;
                        for margin in [0.0f32, 0.25, 1.0, 4.0] {
                            let walked = ruled_walk(
                                &graph,
                                &table,
                                entry_point,
                                StopRule { margin, rank, keep },
                                size,
                            );
                            if let Some((expanded, comparisons)) = &narrower {
                                assert!(
                                    walked.0.starts_with(expanded),
                                    "over {num_vertices} vertices at rank {rank}, L = {size}, \
                                     margin {margin} expanded {:?} where a narrower margin \
                                     expanded {expanded:?}",
                                    walked.0
                                );
                                assert!(
                                    walked.1 >= *comparisons,
                                    "margin {margin} measured {} distances against \
                                     {comparisons} for a narrower one",
                                    walked.1
                                );
                                further += usize::from(walked.0.len() > expanded.len());
                            }
                            narrower = Some(walked);
                        }
                    }
                }
            }
        }
        assert!(
            further > 0,
            "no wider margin ever expanded more than a narrower one, so the fixture never let the \
             margin decide anything"
        );
    }

    #[test]
    fn a_stop_rule_stops_expanding_past_its_bar() {
        let mut list = SearchList::with_margin(
            10,
            10,
            StopRule {
                margin: 0.0,
                rank: 1,
                keep: 1,
            },
        );
        list.offer(0, 1.0);
        list.offer(1, 4.0);
        assert_eq!(list.next_unexpanded().map(|node| node.id), Some(0));
        // The bar is 1.0, so the vertex at 4.0 is never expanded.
        assert_eq!(list.next_unexpanded(), None);
        list.offer(2, 0.5);
        assert_eq!(list.next_unexpanded().map(|node| node.id), Some(2));
    }

    /// Every answer differs from the one the trait's default would give, so
    /// only a forwarded call returns it.
    struct Marked;

    impl DistCalculator for Marked {
        fn distance(&self, id: u32) -> f32 {
            id as f32
        }

        fn distance_all(&self, _k_hint: usize) -> Vec<f32> {
            vec![1.0]
        }

        fn distance_all_with_scratch(
            &self,
            _k_hint: usize,
            dists: &mut Vec<f32>,
            _: &mut Vec<u16>,
            _: &mut Vec<u8>,
            _: &mut Vec<u32>,
        ) {
            *dists = vec![2.0];
        }

        fn has_exact_topk_scan(&self) -> bool {
            true
        }

        fn accumulate_topk_with_scratch(
            &self,
            _k: usize,
            _lower_bound: Option<f32>,
            _upper_bound: Option<f32>,
            _row_id: impl Fn(u32) -> u64,
            res: &mut BinaryHeap<OrderedNode<u64>>,
            _: &mut Vec<f32>,
            _: &mut Vec<u16>,
            _: &mut Vec<u8>,
            _: &mut Vec<u32>,
        ) {
            res.push(OrderedNode::new(7, OrderedFloat(0.0)));
        }

        fn accumulate_filtered_topk_with_scratch(
            &self,
            _k: usize,
            _lower_bound: Option<f32>,
            _upper_bound: Option<f32>,
            _row_ids: impl Iterator<Item = (u32, u64)>,
            _accept_row: impl Fn(u64) -> bool,
            res: &mut BinaryHeap<OrderedNode<u64>>,
            _: &mut Vec<f32>,
            _: &mut Vec<u16>,
            _: &mut Vec<u8>,
            _: &mut Vec<u32>,
        ) {
            res.push(OrderedNode::new(9, OrderedFloat(0.0)));
        }
    }

    #[test]
    fn leading_lines_forward_everything_but_the_ask() {
        let leading = LeadingLines::new(&Marked, &[], 1, 2);
        assert_eq!(leading.distance(5), 5.0);
        assert_eq!(leading.distance_all(3), vec![1.0]);
        let mut dists = Vec::new();
        leading.distance_all_with_scratch(
            3,
            &mut dists,
            &mut Vec::new(),
            &mut Vec::new(),
            &mut Vec::new(),
        );
        assert_eq!(dists, vec![2.0]);
        assert!(leading.has_exact_topk_scan());
        let mut heap = BinaryHeap::new();
        leading.accumulate_topk_with_scratch(
            1,
            None,
            None,
            u64::from,
            &mut heap,
            &mut Vec::new(),
            &mut Vec::new(),
            &mut Vec::new(),
            &mut Vec::new(),
        );
        leading.accumulate_filtered_topk_with_scratch(
            1,
            None,
            None,
            std::iter::empty(),
            |_| true,
            &mut heap,
            &mut Vec::new(),
            &mut Vec::new(),
            &mut Vec::new(),
            &mut Vec::new(),
        );
        let mut ids = heap.into_iter().map(|node| node.id).collect::<Vec<_>>();
        ids.sort_unstable();
        assert_eq!(ids, vec![7, 9]);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn leading_lines_ask_for_the_first_lines_of_a_vector_only() {
        let dimension = 64;
        let values = vec![0.0f32; dimension * 3];
        let vectors = FixedSizeListArray::try_new(
            Arc::new(arrow_schema::Field::new(
                "item",
                arrow_schema::DataType::Float32,
                true,
            )),
            dimension as i32,
            Arc::new(arrow_array::Float32Array::from(values)),
            None,
        )
        .unwrap();
        let storage = FlatFloatStorage::new(vectors.clone(), DistanceType::L2);
        let buffer = vectors
            .values()
            .as_any()
            .downcast_ref::<arrow_array::Float32Array>()
            .unwrap()
            .values()
            .to_vec();
        let calculator = storage.dist_calculator_from_id(0);
        let leading = LeadingLines::new(&calculator, &buffer, dimension, 2);
        record_asks();
        leading.prefetch(1);
        leading.prefetch(3);
        let asked = recorded_asks();
        let start = buffer[dimension..].as_ptr() as usize;
        assert_eq!(asked, vec![start, start + 64]);
    }
}
