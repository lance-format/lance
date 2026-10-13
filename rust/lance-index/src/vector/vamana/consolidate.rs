// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Taking dead vertices out of a graph: Algorithm 4 of the FreshVamana paper.
//!
//! A live vertex that pointed at a dead one inherits that vertex's live
//! out-edges in its place, so the routes through the dead survive them:
//!
//! ```text
//! C <- (N_out(p) \ D) union (union of N_out(v) for v in N_out(p) intersect D)
//! N_out(p) <- RobustPrune(p, C \ D, alpha, R)   when |C| > R, else C itself
//! ```
//!
//! The inheritance is one hop, so no edge dangles afterwards, but the graph
//! can still come apart: removing 90% of a 1000-vertex graph of four
//! dimensions evenly cut 2 of the 100 survivors off in [`super::remap`]'s
//! tests, and 98 on the `lance-vamana` crate's fixture. Callers link the
//! survivors it cuts off back in with `insert::relink_stranded`.

use lance_core::{Error, Result};

use super::build::robust_prune;
use super::graph::{Adjacency, NO_NEIGHBOR, PartitionGraph};
use super::search::Comparisons;
use crate::vector::graph::{OrderedFloat, OrderedNode};
use crate::vector::storage::{DistCalculator, VectorStore};

/// Rewrite `old` over a store whose positions `new_position` gives.
///
/// Old vertex `v` becomes position `new_position[v]`, or is dead when that is
/// `None`; survivors must keep their relative order. `row_ids` names every
/// position of `store`, which may hold positions no old vertex maps to: those
/// start without out-edges. A re-prune measures with `store`, in the new
/// numbering.
pub(crate) fn consolidate<S: VectorStore>(
    old: &impl Adjacency,
    new_position: &[Option<u32>],
    store: &S,
    row_ids: Vec<u64>,
    max_degree: u32,
    alpha: f32,
    comparisons: &Comparisons,
) -> Result<PartitionGraph> {
    if new_position.len() != old.num_vertices() {
        return Err(Error::invalid_input(format!(
            "Vamana consolidation was given {} positions for {} vertices",
            new_position.len(),
            old.num_vertices()
        )));
    }
    if row_ids.len() != store.len() {
        return Err(Error::invalid_input(format!(
            "Vamana consolidation was given {} row ids for a store of {} vectors",
            row_ids.len(),
            store.len()
        )));
    }
    let mut last = None;
    for position in new_position.iter().flatten() {
        if last.is_some_and(|last| last >= *position) || *position as usize >= row_ids.len() {
            return Err(Error::invalid_input(format!(
                "Vamana consolidation positions must ascend within a store of {} vectors, got \
                 {position} after {last:?}",
                row_ids.len()
            )));
        }
        last = Some(*position);
    }

    let mut graph = PartitionGraph::edgeless(max_degree, row_ids)?;
    let mut candidates = Vec::with_capacity(max_degree as usize);
    // The point each position was last taken as a candidate of. A list read
    // from a file can name its own vertex or an edge twice, which no writer
    // here does; both are dropped rather than refused.
    let mut taken_by = vec![NO_NEIGHBOR; store.len()];
    for (point, position) in new_position.iter().enumerate() {
        let Some(position) = *position else {
            continue;
        };
        candidates.clear();
        taken_by[position as usize] = position;
        let mut touches_dead = false;
        for neighbor in old.neighbors(point as u32) {
            // A live neighbor stands for itself, a dead one for its own live
            // out-neighbors.
            let hops = match new_position[*neighbor as usize] {
                Some(_) => std::slice::from_ref(neighbor),
                None => {
                    touches_dead = true;
                    old.neighbors(*neighbor)
                }
            };
            for hop in hops {
                if let Some(alive) = new_position[*hop as usize]
                    && taken_by[alive as usize] != position
                {
                    taken_by[alive as usize] = position;
                    candidates.push(alive);
                }
            }
        }
        // An untouched list is carried over in the order its prune wrote it,
        // nearest first; a list is never wider than `max_degree` without a
        // dead neighbor, so only a touched one can need a prune.
        if touches_dead {
            candidates.sort_unstable();
            if candidates.len() > max_degree as usize {
                let from_point = store.dist_calculator_from_id(position);
                comparisons.record(candidates.len() as u64);
                let pool = candidates
                    .iter()
                    .map(|id| OrderedNode::new(*id, OrderedFloat(from_point.distance(*id))))
                    .collect();
                candidates = robust_prune(
                    store,
                    position,
                    pool,
                    alpha,
                    max_degree as usize,
                    comparisons,
                )?;
            }
        }
        graph.set_neighbors(position, &candidates)?;
    }
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use arrow_array::builder::{ListBuilder, UInt32Builder};
    use arrow_array::{FixedSizeListArray, Float32Array, UInt64Array};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_linalg::distance::DistanceType;

    use super::*;
    use crate::vector::flat::storage::FlatFloatStorage;
    use crate::vector::vamana::build::build_partition;
    use crate::vector::vamana::graph::{LoadedGraph, neighbor_item_field, reachable};
    use crate::vector::vamana::params::VamanaBuildParams;
    use crate::vector::vamana::testing::scattered_values;

    fn storage_of(values: Vec<f32>, dimension: usize) -> FlatFloatStorage {
        FlatFloatStorage::new(
            FixedSizeListArray::try_new_from_values(Float32Array::from(values), dimension as i32)
                .unwrap(),
            DistanceType::L2,
        )
    }

    /// Vertices on a line, each pointing at its two neighbors.
    fn line_graph(num_vertices: u32) -> PartitionGraph {
        let adjacency = (0..num_vertices)
            .map(|vertex| {
                [vertex.checked_sub(1), Some(vertex + 1)]
                    .into_iter()
                    .flatten()
                    .filter(|neighbor| *neighbor < num_vertices)
                    .collect()
            })
            .collect::<Vec<_>>();
        PartitionGraph::try_new(4, (100..100 + num_vertices as u64).collect(), &adjacency).unwrap()
    }

    /// The new positions when `dead` are taken out and the rest close ranks.
    fn closing_ranks(num_vertices: u32, dead: &[u32]) -> Vec<Option<u32>> {
        let mut next = 0;
        (0..num_vertices)
            .map(|vertex| {
                (!dead.contains(&vertex)).then(|| {
                    next += 1;
                    next - 1
                })
            })
            .collect()
    }

    fn survivors_store(num_vertices: u32, dead: &[u32]) -> (FlatFloatStorage, Vec<u64>) {
        let kept = (0..num_vertices).filter(|vertex| !dead.contains(vertex));
        let row_ids = kept.clone().map(|vertex| 100 + vertex as u64).collect();
        (
            storage_of(kept.map(|vertex| vertex as f32).collect(), 1),
            row_ids,
        )
    }

    #[test]
    fn a_hole_in_the_line_is_closed_by_inheritance() {
        let graph = line_graph(10);
        let (store, row_ids) = survivors_store(10, &[5]);
        let consolidated = consolidate(
            &graph,
            &closing_ranks(10, &[5]),
            &store,
            row_ids.clone(),
            4,
            1.2,
            &Comparisons::default(),
        )
        .unwrap();
        assert_eq!(consolidated.row_ids(), row_ids);
        // Old 4 pointed at 3 and 5; it inherits 5's live neighbor 6, now at 5.
        assert_eq!(consolidated.neighbors(4), &[3, 5]);
        // Old 6 is now 5 and inherits old 4.
        assert_eq!(consolidated.neighbors(5), &[4, 6]);
        // Untouched lists keep their order.
        assert_eq!(consolidated.neighbors(2), &[1, 3]);
        assert_eq!(reachable(&consolidated, 0, |_| true), 9);
    }

    /// A real build over scattered points, where inherited lists overflow and
    /// are pruned back. Removing every other row keeps every survivor
    /// reachable; the crate measured the one-hop repair losing survivors only
    /// past 70% removed evenly.
    #[test]
    fn a_built_graph_survives_losing_half_its_rows() {
        const VERTICES: u32 = 1000;
        const DIMENSION: usize = 8;
        let values = scattered_values(VERTICES as usize * DIMENSION);
        let store = storage_of(values.clone(), DIMENSION);
        let params = VamanaBuildParams {
            max_degree: 16,
            search_list_size: 32,
            alpha: 1.2,
        };
        let built = build_partition(&store, &params, 42, &Comparisons::default()).unwrap();
        let dead = (0..VERTICES)
            .filter(|vertex| vertex % 2 == 0)
            .collect::<Vec<_>>();
        let survivors = (0..VERTICES)
            .filter(|vertex| vertex % 2 == 1)
            .collect::<Vec<_>>();
        let survivor_values = survivors
            .iter()
            .flat_map(|vertex| values[*vertex as usize * DIMENSION..][..DIMENSION].to_vec())
            .collect::<Vec<_>>();
        let survivor_store = storage_of(survivor_values, DIMENSION);
        let comparisons = Comparisons::default();
        let positions = closing_ranks(VERTICES, &dead);
        let consolidated = consolidate(
            &built.graph,
            &positions,
            &survivor_store,
            survivors.iter().map(|vertex| *vertex as u64).collect(),
            params.max_degree,
            params.alpha,
            &comparisons,
        )
        .unwrap();
        assert!(comparisons.get() > 0, "no inherited list was pruned back");
        for vertex in 0..survivors.len() as u32 {
            let neighbors = consolidated.neighbors(vertex);
            assert!(!neighbors.is_empty() && neighbors.len() <= params.max_degree as usize);
        }
        let start = positions[built.medoid as usize].unwrap_or(0);
        assert_eq!(reachable(&consolidated, start, |_| true), survivors.len());
    }

    #[test]
    fn positions_no_old_vertex_maps_to_start_without_edges() {
        let graph = line_graph(4);
        let store = storage_of(vec![0.0, 1.0, 2.0, 3.0, 9.0, 10.0], 1);
        let consolidated = consolidate(
            &graph,
            &[Some(0), Some(1), Some(2), Some(3)],
            &store,
            vec![100, 101, 102, 103, 7, 8],
            4,
            1.2,
            &Comparisons::default(),
        )
        .unwrap();
        assert_eq!(consolidated.neighbors(3), &[2]);
        assert_eq!(consolidated.neighbors(4), &[] as &[u32]);
        assert_eq!(consolidated.neighbors(5), &[] as &[u32]);
    }

    /// A list read from a file can name its own vertex or an edge twice,
    /// which no writer here does: both are dropped, from a list that loses a
    /// neighbor and from one that does not.
    #[test]
    fn self_edges_and_repeats_from_a_file_are_dropped() {
        let mut builder = ListBuilder::new(UInt32Builder::new()).with_field(neighbor_item_field());
        for list in [vec![0, 3, 1, 1], vec![2, 2, 0], vec![1], vec![0, 2, 3]] {
            builder.values().append_slice(&list);
            builder.append(true);
        }
        let loaded = LoadedGraph::try_new(
            &UInt64Array::from(vec![100, 101, 102, 103]),
            &builder.finish(),
            4,
        )
        .unwrap();
        let consolidated = consolidate(
            &loaded,
            &[Some(0), Some(1), Some(2), None],
            &storage_of(vec![0.0, 1.0, 2.0], 1),
            vec![100, 101, 102],
            4,
            1.2,
            &Comparisons::default(),
        )
        .unwrap();
        assert_eq!(consolidated.neighbors(0), &[1, 2]);
        assert_eq!(consolidated.neighbors(1), &[2, 0]);
        assert_eq!(consolidated.neighbors(2), &[1]);
    }

    #[test]
    fn positions_that_do_not_ascend_are_refused() {
        let graph = line_graph(3);
        let store = storage_of(vec![0.0, 1.0, 2.0], 1);
        let comparisons = Comparisons::default();
        let row_ids = vec![1, 2, 3];
        for positions in [
            vec![Some(1), Some(0), Some(2)],
            vec![Some(0), Some(0), Some(2)],
            vec![Some(0), Some(1), Some(3)],
            vec![Some(0), Some(1)],
        ] {
            assert!(
                consolidate(
                    &graph,
                    &positions,
                    &store,
                    row_ids.clone(),
                    4,
                    1.2,
                    &comparisons
                )
                .is_err(),
                "{positions:?}"
            );
        }
        assert!(
            consolidate(
                &graph,
                &[Some(0), Some(1), Some(2)],
                &store,
                vec![1, 2],
                4,
                1.2,
                &comparisons
            )
            .is_err()
        );
    }
}
