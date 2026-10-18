// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Adding vertices to a graph.
//!
//! A build is repeated insertion, and `insert_point` is the step it repeats;
//! extending a built graph with new rows runs the same step once per new
//! vertex. Sharing it keeps a graph built whole and one built up by insertion
//! from drifting apart in a way no invariant check would notice.

use arrow_array::Float32Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use lance_core::{Error, Result};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

use super::VAMANA_SEED;
use super::build::{Admission, BUILD_PREFETCH_AHEAD, BUILD_PREFETCH_LINES, Pruned, admit, prune};
use super::graph::{Adjacency, PartitionGraph, reached};
use super::params::VamanaBuildParams;
use super::search::{Comparisons, LeadingLines, SearchScratch, greedy_search_with_lookahead};
use crate::vector::flat::storage::{FLAT_COLUMN, FlatFloatStorage};
use crate::vector::graph::{OrderedFloat, OrderedNode};
use crate::vector::storage::{DistCalculator, VectorStore};

/// What one insertion runs under. `alpha` is separate from the build
/// parameters because a build makes two passes at two values of it.
pub(crate) struct Linking<'a> {
    pub alpha: f32,
    pub search_list_size: usize,
    /// The vectors the store's distances read, for the search to ask for
    /// ahead; `None` asks for nothing.
    pub vectors: Option<&'a ResidentVectors>,
}

/// The vectors a flat store's distances read, in the buffer they read them
/// from.
pub(crate) struct ResidentVectors {
    values: Float32Array,
    dimension: usize,
}

impl ResidentVectors {
    /// The vectors of a flat `f32` store, or `None` for any other store.
    ///
    /// Checked against the store at both ends rather than taken from its
    /// batch on trust: the ask names addresses in this buffer, and a copy
    /// elsewhere would ask for the right data at the wrong address.
    pub(crate) fn of<S: VectorStore>(store: &S) -> Option<Self> {
        let flat = store.as_any().downcast_ref::<FlatFloatStorage>()?;
        let batch = flat.to_batches().ok()?.next()?;
        let vectors = batch
            .column_by_name(FLAT_COLUMN)?
            .as_fixed_size_list_opt()?;
        let dimension = vectors.value_length() as usize;
        let values = vectors.values().as_primitive_opt::<Float32Type>()?.clone();
        let last = flat.len().checked_sub(1)?;
        if values.len() != flat.len() * dimension {
            return None;
        }
        let read_in_place = [0, last].into_iter().all(|id| {
            flat.vector(id as u32)
                .as_primitive_opt::<Float32Type>()
                .is_some_and(|read| {
                    read.values().as_ptr() == values.values()[id * dimension..].as_ptr()
                })
        });
        read_in_place.then_some(Self { values, dimension })
    }
}

/// Buffers reused across a run of insertions.
pub(crate) struct InsertScratch {
    search: SearchScratch,
    list: OpenList,
    /// Whether each vertex's out-edges are still exactly what a prune without
    /// its coincident fill returned, the one kind of list [`admit`] can take a
    /// back-edge into. Such a list holds at the alpha it was pruned at and any
    /// larger one, so the insertions one scratch serves must not lower alpha.
    is_prune_output: Vec<bool>,
}

impl InsertScratch {
    pub(crate) fn new(num_vertices: usize, max_degree: u32) -> Self {
        Self {
            search: SearchScratch::new(num_vertices),
            list: OpenList::new(max_degree),
            is_prune_output: vec![false; num_vertices],
        }
    }

    /// Each vertex's flag, for a run of insertions that links in batches.
    pub(crate) fn prune_outputs_mut(&mut self) -> &mut [bool] {
        &mut self.is_prune_output
    }

    /// Make every full list go through the prune, as if none were a prune's
    /// output: what a test compares the shortcut against.
    #[cfg(test)]
    pub(crate) fn forget_prune_outputs(&mut self) {
        self.is_prune_output.fill(false);
    }
}

/// One vertex's out-edges, taken out of the graph to be offered back-edges.
pub(crate) struct OpenList {
    owner: u32,
    edges: Vec<u32>,
    /// What [`InsertScratch`] keeps for the vertex: whether `edges` is still
    /// exactly a prune's output.
    is_prune_output: bool,
}

/// What [`OpenList::offer`] did with a newcomer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Offered {
    /// The list already held it.
    Present,
    /// The list had room.
    Appended,
    /// [`admit`] found that the prune would keep the list as it stands.
    Refused,
    /// [`admit`] found the list the prune would return.
    Admitted,
    /// [`admit`] could not decide, and the prune did.
    Undecided,
    /// The list was not a prune's output, and the prune decided.
    Repruned,
}

impl Offered {
    /// Whether the list must be written back: every outcome but the two that
    /// leave the list and its flag as they were opened. A prune may hand back
    /// the list it was given, which is written back all the same.
    pub(crate) fn changed(self) -> bool {
        matches!(
            self,
            Self::Appended | Self::Admitted | Self::Undecided | Self::Repruned
        )
    }
}

impl OpenList {
    pub(crate) fn new(max_degree: u32) -> Self {
        Self {
            owner: 0,
            edges: Vec::with_capacity(max_degree as usize),
            is_prune_output: false,
        }
    }

    /// Start offering to `owner`, whose out-edges are `edges`.
    pub(crate) fn open(&mut self, owner: u32, edges: &[u32], is_prune_output: bool) {
        self.owner = owner;
        self.edges.clear();
        self.edges.extend_from_slice(edges);
        self.is_prune_output = is_prune_output;
    }

    pub(crate) fn edges(&self) -> &[u32] {
        &self.edges
    }

    pub(crate) fn is_prune_output(&self) -> bool {
        self.is_prune_output
    }

    /// Give the owner an edge to `newcomer`: added outright while the list
    /// has room, and fought for through the prune when it does not. A full
    /// list that is still a prune's output decides through [`admit`] instead,
    /// with the same answer.
    pub(crate) fn offer<S: VectorStore>(
        &mut self,
        store: &S,
        newcomer: u32,
        alpha: f32,
        max_degree: usize,
        comparisons: &Comparisons,
    ) -> Result<Offered> {
        if self.edges.contains(&newcomer) {
            return Ok(Offered::Present);
        }
        if self.edges.len() < max_degree {
            self.edges.push(newcomer);
            self.is_prune_output = false;
            return Ok(Offered::Appended);
        }
        let mut offered = Offered::Repruned;
        if self.is_prune_output {
            match admit(
                store,
                self.owner,
                &self.edges,
                newcomer,
                alpha,
                max_degree,
                comparisons,
            )? {
                Admission::Refused => return Ok(Offered::Refused),
                Admission::Admitted(admitted) => {
                    self.edges = admitted;
                    return Ok(Offered::Admitted);
                }
                Admission::Undecided => offered = Offered::Undecided,
            }
        }
        let from_owner = store.dist_calculator_from_id(self.owner);
        comparisons.record(self.edges.len() as u64 + 1);
        let contenders = self
            .edges
            .iter()
            .chain(std::iter::once(&newcomer))
            .map(|id| OrderedNode::new(*id, OrderedFloat(from_owner.distance(*id))))
            .collect();
        let repruned = prune(
            store,
            self.owner,
            contenders,
            alpha,
            max_degree,
            comparisons,
        )?;
        self.edges = repruned.neighbors;
        self.is_prune_output = !repruned.has_coincident_fill;
        Ok(offered)
    }
}

/// Give `point` its out-edges, and give the vertices it chose an edge back.
///
/// `point` must already be a vertex of `graph`, with whatever out-edges it
/// has. Each back-edge is added outright while the neighbor has room, and
/// fought for through the prune when it does not; a full list that is still a
/// prune's output decides through [`admit`] instead, with the same answer.
pub(crate) fn insert_point<S: VectorStore>(
    graph: &mut PartitionGraph,
    store: &S,
    scratch: &mut InsertScratch,
    linking: &Linking,
    point: u32,
    entry_point: u32,
    comparisons: &Comparisons,
) -> Result<()> {
    let max_degree = graph.max_degree() as usize;
    let selected = choose_out_edges(
        graph,
        store,
        &mut scratch.search,
        linking,
        point,
        entry_point,
        comparisons,
    )?;
    graph.set_neighbors(point, &selected.neighbors)?;
    scratch.is_prune_output[point as usize] = !selected.has_coincident_fill;

    for neighbor in &selected.neighbors {
        let neighbor = *neighbor;
        scratch.list.open(
            neighbor,
            graph.neighbors(neighbor),
            scratch.is_prune_output[neighbor as usize],
        );
        let offered = scratch
            .list
            .offer(store, point, linking.alpha, max_degree, comparisons)?;
        if offered.changed() {
            graph.set_neighbors(neighbor, scratch.list.edges())?;
        }
        scratch.is_prune_output[neighbor as usize] = scratch.list.is_prune_output();
    }
    Ok(())
}

/// Search `graph` from `entry_point` for `point`, and prune what the search
/// visited, together with `point`'s current out-edges, into its new out-edges.
///
/// Writes nothing, so a batch of points can choose against one graph at once.
pub(crate) fn choose_out_edges<S: VectorStore>(
    graph: &PartitionGraph,
    store: &S,
    search: &mut SearchScratch,
    linking: &Linking,
    point: u32,
    entry_point: u32,
    comparisons: &Comparisons,
) -> Result<Pruned> {
    let from_point = store.dist_calculator_from_id(point);
    let found = match linking.vectors {
        Some(resident) => greedy_search_with_lookahead(
            graph,
            &LeadingLines::new(
                &from_point,
                resident.values.values(),
                resident.dimension,
                BUILD_PREFETCH_LINES,
            ),
            entry_point,
            linking.search_list_size,
            BUILD_PREFETCH_AHEAD,
            search,
            comparisons,
        ),
        None => greedy_search_with_lookahead(
            graph,
            &from_point,
            entry_point,
            linking.search_list_size,
            BUILD_PREFETCH_AHEAD,
            search,
            comparisons,
        ),
    };
    let mut candidates = found?.visited;
    // The paper folds the current out-edges into the candidates inside the
    // prune; doing it here keeps the prune ignorant of the graph, so the
    // back-edge case can reuse it.
    comparisons.record(graph.neighbors(point).len() as u64);
    candidates.extend(
        graph.neighbors(point).iter().map(|neighbor| {
            OrderedNode::new(*neighbor, OrderedFloat(from_point.distance(*neighbor)))
        }),
    );
    prune(
        store,
        point,
        candidates,
        linking.alpha,
        graph.max_degree() as usize,
        comparisons,
    )
}

/// Link vertices `first_new..` of `graph` into the graph before them.
///
/// One pass at the full `alpha`, where a build makes two: the graph the new
/// vertices search is already navigable. `entry_point` must be one of the
/// existing vertices. The new vertices are inserted in a seeded random order
/// rather than in storage order, which is scan order and would insert a whole
/// region of a sorted dataset before its neighbors exist.
pub(crate) fn insert_vertices<S: VectorStore>(
    graph: &mut PartitionGraph,
    store: &S,
    first_new: u32,
    entry_point: u32,
    params: &VamanaBuildParams,
    seed: u64,
    comparisons: &Comparisons,
) -> Result<()> {
    params.validate()?;
    if params.max_degree != graph.max_degree() {
        return Err(Error::invalid_input(format!(
            "Vamana cannot insert at max_degree {} into a graph built at {}",
            params.max_degree,
            graph.max_degree()
        )));
    }
    if store.len() != graph.len() {
        return Err(Error::invalid_input(format!(
            "Vamana graph of {} vertices cannot be extended over a store of {} vectors",
            graph.len(),
            store.len()
        )));
    }
    if entry_point >= first_new {
        return Err(Error::invalid_input(format!(
            "Vamana insertion must start from an existing vertex, got {entry_point} with \
             {first_new} existing"
        )));
    }
    let resident = ResidentVectors::of(store);
    let linking = Linking {
        alpha: params.alpha,
        search_list_size: params.search_list_size,
        vectors: resident.as_ref(),
    };
    let mut scratch = InsertScratch::new(graph.len(), graph.max_degree());
    let mut order = (first_new..graph.len() as u32).collect::<Vec<_>>();
    order.shuffle(&mut SmallRng::seed_from_u64(seed));
    for point in order {
        insert_point(
            graph,
            store,
            &mut scratch,
            &linking,
            point,
            entry_point,
            comparisons,
        )?;
    }
    Ok(())
}

/// Insert again each of the vertices below `carried` that a walk from
/// `start` cannot reach or that has no out-edges, and say whether every
/// vertex of `must_reach` is reachable from `start` afterwards.
///
/// Consolidation's one-hop inheritance leaves no edge dangling but can cut
/// survivors off: a route that ran through two dead vertices in a row is
/// gone, and a survivor whose every neighbor died can be left with no
/// out-edges at all. Inserting those again costs a search each, where a
/// rebuild costs one per vertex, twice. It can fail, because back-edges are
/// fought for through a prune: an inserted vertex can be left with no
/// in-edges, and a prune can take another vertex's only one away.
///
/// The searches start at `start`, or at the first vertex with out-edges when
/// `start` has none. `false` when a vertex of `must_reach` ends up cut off,
/// or when there are two or more vertices and none has an out-edge to
/// search from. A count of the vertices reached would not do: one vertex
/// linked in can hide another cut off.
pub(crate) fn relink_stranded<S: VectorStore>(
    graph: &mut PartitionGraph,
    store: &S,
    carried: u32,
    start: u32,
    must_reach: &[u32],
    params: &VamanaBuildParams,
    comparisons: &Comparisons,
) -> Result<bool> {
    if start >= carried || carried as usize > graph.len() {
        return Err(Error::invalid_input(format!(
            "Vamana relinking needs a start among the first {carried} vertices of a graph of {}, \
             got {start}",
            graph.len()
        )));
    }
    if let Some(vertex) = must_reach.iter().find(|vertex| **vertex >= carried) {
        return Err(Error::invalid_input(format!(
            "Vamana relinking cannot keep vertex {vertex} reachable: it relinks the first \
             {carried} only"
        )));
    }
    let Some(search_start) = std::iter::once(start)
        .chain(0..carried)
        .find(|vertex| !graph.neighbors(*vertex).is_empty())
    else {
        return Ok(carried <= 1);
    };
    let reached_from_search = reached(graph, search_start);
    let mut stranded = (0..carried)
        .filter(|vertex| {
            !reached_from_search[*vertex as usize] || graph.neighbors(*vertex).is_empty()
        })
        .collect::<Vec<_>>();
    if stranded.is_empty() {
        // `start` itself had out-edges and reached every vertex.
        return Ok(true);
    }
    // Shuffled for the reason `insert_vertices` shuffles.
    stranded.shuffle(&mut SmallRng::seed_from_u64(VAMANA_SEED));
    let resident = ResidentVectors::of(store);
    let linking = Linking {
        alpha: params.alpha,
        search_list_size: params.search_list_size,
        vectors: resident.as_ref(),
    };
    let mut scratch = InsertScratch::new(graph.len(), graph.max_degree());
    for vertex in stranded {
        insert_point(
            graph,
            store,
            &mut scratch,
            &linking,
            vertex,
            search_start,
            comparisons,
        )?;
    }
    let reached = reached(graph, start);
    Ok(must_reach.iter().all(|vertex| reached[*vertex as usize]))
}

/// The new positions of the vertices a walk from `start` reaches in `old`,
/// among those `new_position` keeps: what a relink must keep reachable.
pub(crate) fn reached_survivors(
    old: &impl Adjacency,
    start: u32,
    new_position: &[Option<u32>],
) -> Vec<u32> {
    reached(old, start)
        .into_iter()
        .zip(new_position)
        .filter_map(|(is_reached, position)| position.filter(|_| is_reached))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use lance_linalg::distance::DistanceType;

    use super::*;
    use crate::vector::vamana::build::build_partition;
    use crate::vector::vamana::graph::reachable;
    use crate::vector::vamana::testing::{flat_storage, scattered_values, walk_recall_at_ten};

    const DIMENSION: usize = 8;

    fn params() -> VamanaBuildParams {
        VamanaBuildParams {
            max_degree: 16,
            search_list_size: 32,
            alpha: 1.2,
        }
    }

    /// A graph built over the first `built` rows of `store`, extended by the
    /// rest through insertion.
    fn grown(store: &FlatFloatStorage, built: usize) -> (PartitionGraph, u32) {
        let first = flat_storage(
            scattered_values(store.len() * DIMENSION)[..built * DIMENSION].to_vec(),
            DIMENSION,
            DistanceType::L2,
        );
        let comparisons = Comparisons::default();
        let base = build_partition(&first, &params(), 42, &comparisons).unwrap();
        let mut graph = base.graph;
        graph
            .extend(&(built as u64..store.len() as u64).collect::<Vec<_>>())
            .unwrap();
        insert_vertices(
            &mut graph,
            store,
            built as u32,
            base.medoid,
            &params(),
            42,
            &comparisons,
        )
        .unwrap();
        (graph, base.medoid)
    }

    #[test]
    fn inserted_vertices_are_linked_in_both_directions() {
        let store = flat_storage(
            scattered_values(400 * DIMENSION),
            DIMENSION,
            DistanceType::L2,
        );
        let (graph, medoid) = grown(&store, 300);
        let mut pointed_at = HashSet::new();
        for vertex in 0..400u32 {
            let neighbors = graph.neighbors(vertex);
            assert!(
                !neighbors.is_empty() && neighbors.len() <= 16,
                "vertex {vertex}"
            );
            pointed_at.extend(neighbors.iter().copied());
        }
        let reached_new = (300..400u32)
            .filter(|vertex| pointed_at.contains(vertex))
            .count();
        assert!(
            reached_new >= 98,
            "only {reached_new} of 100 new vertices have an in-edge"
        );
        assert!(reachable(&graph, medoid, |_| true) >= 398);
    }

    #[test]
    fn a_grown_graph_answers_nearly_as_well_as_one_built_whole() {
        const ROWS: usize = 1000;
        let values = scattered_values((ROWS + 50) * DIMENSION);
        let store = flat_storage(
            values[..ROWS * DIMENSION].to_vec(),
            DIMENSION,
            DistanceType::L2,
        );
        let (grown, medoid) = grown(&store, 750);
        let whole = build_partition(&store, &params(), 42, &Comparisons::default()).unwrap();
        let recall_of = |graph: &PartitionGraph, start: u32| {
            walk_recall_at_ten(
                &store,
                graph,
                start,
                &values[ROWS * DIMENSION..],
                DIMENSION,
                32,
            )
            .0
        };
        let grown_recall = recall_of(&grown, medoid);
        let whole_recall = recall_of(&whole.graph, whole.medoid);
        assert!(
            grown_recall >= whole_recall - 0.02,
            "grown {grown_recall}, whole {whole_recall}"
        );
    }

    /// A start with no out-edges cannot be searched from: the searches start
    /// at the first vertex that has some, and only what that one cannot reach
    /// is inserted again, the start among it. Vertex 1's list, in an order no
    /// prune writes, shows it was not.
    #[test]
    fn a_start_without_out_edges_is_relinked_from_another_vertex() {
        let store = flat_storage((0..6).map(|i| i as f32).collect(), 1, DistanceType::L2);
        let mut graph = PartitionGraph::try_new(
            4,
            (0..6).collect(),
            &[vec![1], vec![2, 0], vec![], vec![4], vec![3, 5], vec![]],
        )
        .unwrap();
        let params = params().with_max_degree(4);
        let relinked = relink_stranded(
            &mut graph,
            &store,
            6,
            2,
            &[0, 1, 2, 3, 4, 5],
            &params,
            &Comparisons::default(),
        )
        .unwrap();
        assert!(relinked);
        assert!((0..6).all(|vertex| !graph.neighbors(vertex).is_empty()));
        assert_eq!(reachable(&graph, 2, |_| true), 6);
        assert!(graph.neighbors(1).starts_with(&[2, 0]));
    }

    /// At one out-edge a vertex, vertex 0 inserted again takes vertex 1 as
    /// its neighbor, which keeps its nearer edge to 2 rather than take 0 back:
    /// 0 stays cut off. A relink that had to keep 0 reachable says so, even
    /// one under which as many vertices stay reachable as had to.
    #[test]
    fn a_relink_that_leaves_a_vertex_cut_off_says_so() {
        let store = flat_storage(vec![0.0, 3.0, 4.0], 1, DistanceType::L2);
        let graph =
            PartitionGraph::try_new(1, (0..3).collect(), &[vec![], vec![2], vec![]]).unwrap();
        let params = params().with_max_degree(1);
        let relink = |must_reach: &[u32]| {
            let mut graph = graph.clone();
            let kept = relink_stranded(
                &mut graph,
                &store,
                3,
                1,
                must_reach,
                &params,
                &Comparisons::default(),
            )
            .unwrap();
            assert_eq!(graph.neighbors(0), &[1]);
            assert_eq!(reachable(&graph, 1, |_| true), 2);
            kept
        };
        assert!(!relink(&[0, 1, 2]));
        assert!(relink(&[1, 2]));
        assert!(!relink(&[0, 1]));
        let error = relink_stranded(
            &mut graph.clone(),
            &store,
            3,
            1,
            &[3],
            &params,
            &Comparisons::default(),
        )
        .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    }

    /// Only what the old graph reached and the remap kept: 3 is reached but
    /// dies, 2 survives but was reached from nowhere.
    #[test]
    fn a_relink_must_keep_reached_what_was_reached_and_survives() {
        let graph = PartitionGraph::try_new(
            4,
            (0..5).collect(),
            &[vec![1], vec![3], vec![0], vec![4], vec![]],
        )
        .unwrap();
        let new_position = [Some(0), Some(1), Some(2), None, Some(3)];
        assert_eq!(reached_survivors(&graph, 0, &new_position), vec![0, 1, 3]);
    }

    #[test]
    fn insertion_refuses_what_it_cannot_do() {
        let store = flat_storage(
            scattered_values(40 * DIMENSION),
            DIMENSION,
            DistanceType::L2,
        );
        let comparisons = Comparisons::default();
        let mut graph = PartitionGraph::edgeless(16, (0..40).collect()).unwrap();
        let error =
            insert_vertices(&mut graph, &store, 30, 30, &params(), 42, &comparisons).unwrap_err();
        assert!(error.to_string().contains("existing vertex"), "{error}");
        let error = insert_vertices(
            &mut graph,
            &store,
            30,
            0,
            &params().with_max_degree(8),
            42,
            &comparisons,
        )
        .unwrap_err();
        assert!(error.to_string().contains("built at 16"), "{error}");
        let mut short = PartitionGraph::edgeless(16, (0..39).collect()).unwrap();
        let error =
            insert_vertices(&mut short, &store, 30, 0, &params(), 42, &comparisons).unwrap_err();
        assert!(error.to_string().contains("cannot be extended"), "{error}");
    }
}
