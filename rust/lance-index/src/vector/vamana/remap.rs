// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Following a compaction without rebuilding the graph.
//!
//! The storage arrives remapped already: rows the mapping drops are gone and
//! the survivors keep their order. Applying the same rule to the graph's own
//! row ids says which vertex every storage position is, so the graph only has
//! to lose its dead vertices, which consolidation does at a fraction of a
//! build, and the survivors it cuts off are inserted again. A store that does
//! not line up with the graph, or a graph those insertions leave with a
//! survivor cut off that the old graph reached, is rebuilt instead.

use lance_core::Result;
use lance_core::utils::row_addr_remap::RowAddrRemap;

use super::VAMANA_SEED;
use super::build::medoid;
use super::consolidate::consolidate;
use super::entry_points::train_entry_points;
use super::index::{Vamana, check_store};
use super::insert::{reached_survivors, relink_stranded};
use super::search::Comparisons;
use crate::vector::storage::VectorStore;
use crate::vector::v3::subindex::IvfSubIndex;

pub(crate) fn remap(
    index: &Vamana,
    mapping: &RowAddrRemap,
    store: &impl VectorStore,
) -> Result<Vamana> {
    check_store(store)?;
    let params = index.params().clone();
    if store.is_empty() {
        return Ok(Vamana::empty(params));
    }
    if index.is_empty() {
        return Vamana::index_vectors(store, params);
    }

    let old_row_ids = index.graph().row_ids();
    let mut new_position = Vec::with_capacity(old_row_ids.len());
    let mut row_ids = Vec::with_capacity(store.len());
    for row_id in old_row_ids {
        let kept = match mapping.get(*row_id) {
            Some(None) => None,
            Some(Some(new_row_id)) => Some(new_row_id),
            None => Some(*row_id),
        };
        new_position.push(kept.map(|new_row_id| {
            row_ids.push(new_row_id);
            row_ids.len() as u32 - 1
        }));
    }
    let aligned = row_ids.len() == store.len()
        && row_ids
            .iter()
            .enumerate()
            .all(|(position, row_id)| store.row_id(position as u32) == *row_id);
    if !aligned {
        return Vamana::index_vectors(store, params);
    }

    let comparisons = Comparisons::default();
    let mut graph = consolidate(
        index.graph(),
        &new_position,
        store,
        row_ids,
        params.max_degree,
        params.alpha,
        &comparisons,
    )?;
    if graph.len() == index.len() {
        // Nothing died: the same vertices at the same positions.
        let metadata = index.metadata();
        return Vamana::from_graph(
            &graph,
            metadata.medoid,
            metadata.entry_points.clone(),
            params,
        );
    }

    let new_medoid = medoid(store, VAMANA_SEED, &comparisons)?;
    let survivors = graph.len() as u32;
    let must_reach = reached_survivors(index.graph(), index.metadata().medoid, &new_position);
    if !relink_stranded(
        &mut graph,
        store,
        survivors,
        new_medoid,
        &must_reach,
        &params,
        &comparisons,
    )? {
        return Vamana::index_vectors(store, params);
    }
    let entry_points = train_entry_points(store, VAMANA_SEED)?;
    Vamana::from_graph(&graph, new_medoid, entry_points, params)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, UInt32Array};
    use lance_linalg::distance::DistanceType;

    use super::*;
    use crate::metrics::NoOpMetricsCollector;
    use crate::prefilter::NoFilter;
    use crate::vector::flat::storage::FlatFloatStorage;
    use crate::vector::vamana::graph::{Adjacency, PartitionGraph, reachable};
    use crate::vector::vamana::params::{VamanaBuildParams, VamanaQueryParams};
    use crate::vector::vamana::testing::{
        cut_off_after_relink, exact_nearest, flat_storage_with_row_ids, recall, scattered_values,
        torn_line, vectors_of,
    };

    const DIMENSION: usize = 8;

    fn params() -> VamanaBuildParams {
        VamanaBuildParams {
            max_degree: 16,
            search_list_size: 40,
            alpha: 1.2,
        }
    }

    fn vectors(num_rows: usize) -> FixedSizeListArray {
        vectors_of(
            scattered_values(num_rows * DIMENSION)
                .into_iter()
                .map(|value| value - 0.5)
                .collect(),
            DIMENSION,
        )
    }

    fn row_id(position: u32) -> u64 {
        10 * position as u64 + 3
    }

    /// The rows at `positions`, in that order, named `row_ids`.
    fn store_of(
        vectors: &FixedSizeListArray,
        positions: &[u32],
        row_ids: Vec<u64>,
    ) -> FlatFloatStorage {
        let taken = arrow_select::take::take(vectors, &UInt32Array::from(positions.to_vec()), None)
            .unwrap();
        flat_storage_with_row_ids(
            taken.as_fixed_size_list().clone(),
            row_ids,
            DistanceType::L2,
        )
    }

    fn recall_over(index: &Vamana, store: &FlatFloatStorage) -> f64 {
        let queries = scattered_values(20 * DIMENSION);
        let mut total = 0.0;
        for query in queries.as_chunks::<DIMENSION>().0 {
            let query = Arc::new(Float32Array::from(
                query.iter().map(|v| v - 0.5).collect::<Vec<_>>(),
            )) as ArrayRef;
            let expected = exact_nearest(store, query.clone(), 10, |_, _| true);
            let params = VamanaQueryParams {
                ef: 30,
                lower_bound: None,
                upper_bound: None,
                dist_q_c: 0.0,
                stop: None,
            };
            let found = index
                .search(
                    query,
                    10,
                    params,
                    store,
                    Arc::new(NoFilter),
                    &NoOpMetricsCollector,
                )
                .unwrap();
            let found = found[lance_core::ROW_ID]
                .as_primitive::<arrow_array::types::UInt64Type>()
                .values()
                .to_vec();
            total += recall(&found, &expected);
        }
        total / 20.0
    }

    #[test]
    fn a_remap_without_deaths_keeps_every_edge() {
        let vectors = vectors(500);
        let positions = (0..500).collect::<Vec<u32>>();
        let old = store_of(
            &vectors,
            &positions,
            positions.iter().map(|p| row_id(*p)).collect(),
        );
        let index = Vamana::index_vectors(&old, params()).unwrap();
        let moved = positions
            .iter()
            .map(|p| row_id(*p) + 1_000_000)
            .collect::<Vec<_>>();
        let mapping = RowAddrRemap::direct(
            positions
                .iter()
                .map(|p| (row_id(*p), Some(row_id(*p) + 1_000_000)))
                .collect::<HashMap<_, _>>(),
        );
        let new = store_of(&vectors, &positions, moved.clone());
        let remapped = index.remap(&mapping, &new).unwrap();
        assert_eq!(remapped.graph().row_ids(), moved);
        for vertex in 0..500 {
            assert_eq!(
                remapped.graph().neighbors(vertex),
                index.graph().neighbors(vertex)
            );
        }
        assert_eq!(remapped.metadata().medoid, index.metadata().medoid);
        assert_eq!(
            remapped.metadata().entry_points,
            index.metadata().entry_points
        );
    }

    #[test]
    fn deaths_are_consolidated_rather_than_rebuilt() {
        let vectors = vectors(1000);
        let positions = (0..1000).collect::<Vec<u32>>();
        let old = store_of(
            &vectors,
            &positions,
            positions.iter().map(|p| row_id(*p)).collect(),
        );
        let index = Vamana::index_vectors(&old, params()).unwrap();
        let is_dead = |position: u32| position.is_multiple_of(5);
        let mapping = RowAddrRemap::direct(
            positions
                .iter()
                .filter(|p| is_dead(**p))
                .map(|p| (row_id(*p), None))
                .collect::<HashMap<_, _>>(),
        );
        let survivors = positions
            .iter()
            .copied()
            .filter(|p| !is_dead(*p))
            .collect::<Vec<_>>();
        let new = store_of(
            &vectors,
            &survivors,
            survivors.iter().map(|p| row_id(*p)).collect(),
        );
        let remapped = index.remap(&mapping, &new).unwrap();
        assert_eq!(
            remapped.graph().row_ids(),
            new.row_ids().copied().collect::<Vec<_>>()
        );

        // A survivor none of whose neighbors died keeps its list as it was,
        // which a rebuild would not.
        let new_position = |old: u32| survivors.binary_search(&old).unwrap() as u32;
        let mut untouched = 0;
        for (new_vertex, old_vertex) in survivors.iter().enumerate() {
            let old_neighbors = index.graph().neighbors(*old_vertex);
            if old_neighbors.iter().any(|neighbor| is_dead(*neighbor)) {
                continue;
            }
            untouched += 1;
            let translated = old_neighbors
                .iter()
                .map(|n| new_position(*n))
                .collect::<Vec<_>>();
            assert_eq!(remapped.graph().neighbors(new_vertex as u32), translated);
        }
        assert!(untouched > 10, "only {untouched} untouched survivors");
        let recall = recall_over(&remapped, &new);
        assert!(recall >= 0.9, "recall {recall}");
    }

    #[test]
    fn a_store_that_does_not_line_up_is_rebuilt() {
        let vectors = vectors(300);
        let positions = (0..300).collect::<Vec<u32>>();
        let old = store_of(
            &vectors,
            &positions,
            positions.iter().map(|p| row_id(*p)).collect(),
        );
        let index = Vamana::index_vectors(&old, params()).unwrap();
        let mapping = RowAddrRemap::direct(HashMap::from([(row_id(7), None)]));
        let mut survivors = positions
            .iter()
            .copied()
            .filter(|p| *p != 7)
            .collect::<Vec<_>>();
        survivors.reverse();
        let new = store_of(
            &vectors,
            &survivors,
            survivors.iter().map(|p| row_id(*p)).collect(),
        );
        let remapped = index.remap(&mapping, &new).unwrap();
        let rebuilt = Vamana::index_vectors(&new, params()).unwrap();
        assert_eq!(remapped.to_batch().unwrap(), rebuilt.to_batch().unwrap());
    }

    /// The far side of the torn line is inserted again rather than the
    /// partition rebuilt.
    #[test]
    fn a_remap_that_tears_the_graph_apart_relinks_it() {
        let (index, line) = torn_line();
        let mapping = RowAddrRemap::direct(HashMap::from([(row_id(4), None), (row_id(5), None)]));
        let survivors = (0..10).filter(|p| *p != 4 && *p != 5).collect::<Vec<u32>>();
        let new = store_of(
            &line,
            &survivors,
            survivors.iter().map(|p| row_id(*p)).collect(),
        );
        let remapped = index.remap(&mapping, &new).unwrap();
        let graph = remapped.graph();
        // Carried over as it was, in an order no rebuild writes.
        assert_eq!(graph.neighbors(0), &[2, 1]);
        assert!((0..8).all(|vertex| !graph.neighbors(vertex).is_empty()));
        assert_eq!(reachable(graph, remapped.metadata().medoid, |_| true), 8);
    }

    /// Nine rows in ten gone, evenly, from a graph of four dimensions, whose
    /// edges are local enough that inheritance alone cuts survivors off.
    /// Inserting them again leaves every survivor reachable, with out-edges
    /// for a walk to leave by.
    #[test]
    fn a_remap_that_loses_most_rows_relinks_the_rest() {
        const ROWS: u32 = 1000;
        let vectors = vectors_of(
            scattered_values(ROWS as usize * 4)
                .into_iter()
                .map(|value| value - 0.5)
                .collect(),
            4,
        );
        let positions = (0..ROWS).collect::<Vec<u32>>();
        let old = store_of(
            &vectors,
            &positions,
            positions.iter().map(|p| row_id(*p)).collect(),
        );
        let index = Vamana::index_vectors(&old, params()).unwrap();
        let is_dead = |position: u32| position % 100 < 90;
        let mapping = RowAddrRemap::direct(
            positions
                .iter()
                .filter(|p| is_dead(**p))
                .map(|p| (row_id(*p), None))
                .collect::<HashMap<_, _>>(),
        );
        let survivors = positions
            .iter()
            .copied()
            .filter(|p| !is_dead(*p))
            .collect::<Vec<_>>();
        let survivor_ids = survivors.iter().map(|p| row_id(*p)).collect::<Vec<_>>();
        let new = store_of(&vectors, &survivors, survivor_ids.clone());

        let comparisons = Comparisons::default();
        let mut next = 0;
        let new_position = positions
            .iter()
            .map(|p| {
                (!is_dead(*p)).then(|| {
                    next += 1;
                    next - 1
                })
            })
            .collect::<Vec<_>>();
        let torn = consolidate(
            index.graph(),
            &new_position,
            &new,
            survivor_ids,
            params().max_degree,
            params().alpha,
            &comparisons,
        )
        .unwrap();
        let start = medoid(&new, VAMANA_SEED, &comparisons).unwrap();
        let cut_off = survivors.len() - reachable(&torn, start, |_| true);
        assert!(cut_off > 0, "inheritance cut off no survivor");

        let remapped = index.remap(&mapping, &new).unwrap();
        let graph = remapped.graph();
        assert!((0..100).all(|vertex| !graph.neighbors(vertex).is_empty()));
        assert_eq!(reachable(graph, remapped.metadata().medoid, |_| true), 100);
        // A rebuild passes both checks above too.
        let rebuilt = Vamana::index_vectors(&new, params()).unwrap();
        assert_ne!(remapped.to_batch().unwrap(), rebuilt.to_batch().unwrap());
    }

    /// The end of a line pointing only at the vertex before it, which dies:
    /// the end inherits nothing but itself and is left a dead end that the
    /// walk still reaches, as an entry point would be. It is inserted again.
    #[test]
    fn a_survivor_left_without_out_edges_is_linked_again() {
        let line = vectors_of((0..6).map(|i| i as f32).collect(), 1);
        let graph = PartitionGraph::try_new(
            4,
            (0..6).map(row_id).collect(),
            &[
                vec![1],
                vec![0, 2],
                vec![1, 3],
                vec![2, 4],
                vec![5],
                vec![4],
            ],
        )
        .unwrap();
        let index = Vamana::from_graph(&graph, 2, vec![], params().with_max_degree(4)).unwrap();
        let mapping = RowAddrRemap::direct(HashMap::from([(row_id(4), None)]));
        let survivors = [0, 1, 2, 3, 5];
        let new = store_of(
            &line,
            &survivors,
            survivors.iter().map(|p| row_id(*p)).collect(),
        );
        let remapped = index.remap(&mapping, &new).unwrap();
        // Old 5, now at 4, points back at old 3 rather than nowhere.
        assert_eq!(remapped.graph().neighbors(4), &[3]);
    }

    /// Two pairs that point only at each other, each losing one vertex: no
    /// survivor keeps an out-edge to search from, so the partition is built
    /// again.
    #[test]
    fn a_remap_that_leaves_no_edges_at_all_is_rebuilt() {
        let line = vectors_of((0..4).map(|i| i as f32).collect(), 1);
        let graph = PartitionGraph::try_new(
            4,
            (0..4).map(row_id).collect(),
            &[vec![1], vec![0], vec![3], vec![2]],
        )
        .unwrap();
        let params = params().with_max_degree(4);
        let index = Vamana::from_graph(&graph, 0, vec![], params.clone()).unwrap();
        let mapping = RowAddrRemap::direct(HashMap::from([(row_id(1), None), (row_id(3), None)]));
        let new = store_of(&line, &[0, 2], vec![row_id(0), row_id(2)]);
        let remapped = index.remap(&mapping, &new).unwrap();
        let rebuilt = Vamana::index_vectors(&new, params).unwrap();
        assert_eq!(remapped.to_batch().unwrap(), rebuilt.to_batch().unwrap());
    }

    /// The ring's way in dies with the medoid: inserted again, 0 is still
    /// reached from nowhere, so the partition is rebuilt rather than kept
    /// with a survivor no walk finds.
    #[test]
    fn a_relink_that_leaves_a_survivor_cut_off_is_rebuilt() {
        let (index, line) = cut_off_after_relink();
        let mapping = RowAddrRemap::direct(HashMap::from([
            (row_id(5), None),
            (row_id(6), None),
            (row_id(7), None),
        ]));
        let survivors = (0..5).collect::<Vec<u32>>();
        let new = store_of(
            &line,
            &survivors,
            survivors.iter().map(|p| row_id(*p)).collect(),
        );
        let remapped = index.remap(&mapping, &new).unwrap();
        let rebuilt = Vamana::index_vectors(&new, index.params().clone()).unwrap();
        // Kept, the graph would still hold 3 -> 4, which no rebuild writes.
        assert_ne!(rebuilt.graph().neighbors(3), &[4]);
        assert_eq!(remapped.to_batch().unwrap(), rebuilt.to_batch().unwrap());
    }

    #[test]
    fn a_fully_deleted_partition_becomes_empty() {
        let vectors = vectors(50);
        let positions = (0..50).collect::<Vec<u32>>();
        let old = store_of(
            &vectors,
            &positions,
            positions.iter().map(|p| row_id(*p)).collect(),
        );
        let index = Vamana::index_vectors(&old, params()).unwrap();
        let mapping = RowAddrRemap::direct(positions.iter().map(|p| (row_id(*p), None)).collect());
        let new = store_of(&vectors, &[], vec![]);
        let remapped = index.remap(&mapping, &new).unwrap();
        assert!(remapped.is_empty());
        assert_eq!(remapped.to_batch().unwrap().num_rows(), 0);
    }
}
