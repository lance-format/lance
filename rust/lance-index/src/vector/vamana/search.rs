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

    /// A plain list: the nearest `size` offers, first of equals kept.
    fn plain_front(offers: &[(u32, f32)], size: usize) -> Vec<u32> {
        let mut sorted = offers.to_vec();
        sorted.sort_by(|a, b| a.1.total_cmp(&b.1));
        sorted.into_iter().take(size).map(|(id, _)| id).collect()
    }

    #[test]
    fn a_list_with_a_stop_rule_is_the_front_of_the_list_it_rules() {
        let mut rng = SmallRng::seed_from_u64(7);
        for _ in 0..200 {
            let size = rng.random_range(4..20);
            let keep = rng.random_range(1..=size);
            let rank = rng.random_range(1..=keep);
            let stop = StopRule {
                margin: rng.random_range(0.0..0.5),
                rank,
                keep,
            };
            let mut ruled = SearchList::with_margin(size, 100, stop);
            let mut offers = Vec::new();
            for id in 0..rng.random_range(1..60u32) {
                let distance = rng.random_range(0.0..10.0f32);
                ruled.offer(id, distance);
                offers.push((id, distance));
            }
            let ruled = ruled
                .into_candidates()
                .iter()
                .map(|node| node.id)
                .collect::<Vec<_>>();
            let plain = plain_front(&offers, size);
            assert!(ruled.len() >= keep.min(offers.len()));
            assert_eq!(ruled[..], plain[..ruled.len()]);
        }
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
