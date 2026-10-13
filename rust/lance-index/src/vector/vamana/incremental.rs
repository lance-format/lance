// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Extending a partition's graph with the rows an index merge adds.
//!
//! A merged partition's storage starts with the rows of the partition it
//! extends, in order, some of them filtered out, and continues with rows from
//! elsewhere. Matching the base graph's row ids against that prefix says which
//! vertex each matched position is; the unmatched base vertices are dead and
//! consolidated away, survivors that leaves cut off are inserted again, and
//! the remaining positions are linked in by insertion. Every vertex the base
//! graph reached must still be reached from the merged rows' medoid, where
//! walks start, or the partition is rebuilt. Correctness does not
//! depend on how much matches: every position becomes a vertex named by its
//! storage row id, and only edges between matched vertices carry over.

use lance_core::{Error, Result};

use super::VAMANA_SEED;
use super::build::medoid;
use super::consolidate::consolidate;
use super::entry_points::train_entry_points;
use super::index::{Vamana, check_store};
use super::insert::{insert_vertices, reached_survivors, relink_stranded};
use super::params::VamanaBuildParams;
use super::search::Comparisons;
use crate::vector::storage::VectorStore;
use crate::vector::v3::subindex::IvfSubIndex;

/// Build `storage`'s graph by extending `base`'s, or from scratch when too
/// little of `base` carries over: nothing matched, more than half of it dead,
/// more than twice as many new rows as matched ones, or a vertex the base
/// graph reached that inserting the cut-off vertices again leaves cut off
/// from the merged rows' medoid.
///
/// The first `base_rows` positions of `storage` are the rows `base` was built
/// over, in order, some of them filtered out, and only they can match a
/// vertex of `base`. Past them a row id proves nothing: an update under stable
/// row ids deletes a row and adds it back with a new vector under the same id.
pub fn build_incremental(
    base: &Vamana,
    storage: &impl VectorStore,
    base_rows: usize,
    params: VamanaBuildParams,
) -> Result<Vamana> {
    params.validate()?;
    check_store(storage)?;
    let num_rows = storage.len();
    if base_rows > num_rows {
        return Err(Error::invalid_input(format!(
            "Vamana merge was told {base_rows} rows come from the base partition, but the merged \
             partition holds {num_rows}"
        )));
    }
    if storage.is_empty() {
        return Ok(Vamana::empty(params));
    }
    if base.is_empty() || base.params() != &params {
        return Vamana::index_vectors(storage, params);
    }

    let base_row_ids = base.graph().row_ids();
    let mut new_position = Vec::with_capacity(base_row_ids.len());
    let mut matched = 0usize;
    for row_id in base_row_ids {
        if matched < base_rows && storage.row_id(matched as u32) == *row_id {
            new_position.push(Some(matched as u32));
            matched += 1;
        } else {
            new_position.push(None);
        }
    }
    let dead = base_row_ids.len() - matched;
    let added = num_rows - matched;
    if matched == 0 || dead * 2 > base_row_ids.len() || added > matched * 2 {
        return Vamana::index_vectors(storage, params);
    }

    let comparisons = Comparisons::default();
    let row_ids = (0..num_rows as u32)
        .map(|position| storage.row_id(position))
        .collect();
    let mut graph = consolidate(
        base.graph(),
        &new_position,
        storage,
        row_ids,
        params.max_degree,
        params.alpha,
        &comparisons,
    )?;

    let metadata = base.metadata();
    let start = std::iter::once(metadata.medoid)
        .chain(metadata.entry_points.iter().copied())
        .find_map(|vertex| new_position[vertex as usize])
        .unwrap_or(0);
    let must_reach = reached_survivors(base.graph(), metadata.medoid, &new_position);
    // What the deaths cut off is linked in again first, so the new rows'
    // searches see the whole graph.
    if dead > 0
        && !relink_stranded(
            &mut graph,
            storage,
            matched as u32,
            start,
            &must_reach,
            &params,
            &comparisons,
        )?
    {
        return Vamana::index_vectors(storage, params);
    }

    if added > 0 {
        insert_vertices(
            &mut graph,
            storage,
            matched as u32,
            start,
            &params,
            VAMANA_SEED,
            &comparisons,
        )?;
    }
    // Walks start at the merged rows' medoid rather than at `start`, and the
    // new rows' back-edges are fought for through prunes too.
    let medoid = medoid(storage, VAMANA_SEED, &comparisons)?;
    if !relink_stranded(
        &mut graph,
        storage,
        num_rows as u32,
        medoid,
        &must_reach,
        &params,
        &comparisons,
    )? {
        return Vamana::index_vectors(storage, params);
    }
    let entry_points = train_entry_points(storage, VAMANA_SEED)?;
    Vamana::from_graph(&graph, medoid, entry_points, params)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use arrow_array::cast::AsArray;
    use arrow_array::{FixedSizeListArray, UInt32Array};
    use lance_linalg::distance::DistanceType;
    use rstest::rstest;

    use super::*;
    use crate::vector::flat::storage::FlatFloatStorage;
    use crate::vector::vamana::graph::{Adjacency, PartitionGraph, reachable};
    use crate::vector::vamana::testing::{
        cut_off_after_relink, flat_storage_with_row_ids, scattered_values, torn_line, vectors_of,
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
        vectors_of(scattered_values(num_rows * DIMENSION), DIMENSION)
    }

    fn row_id(position: u32) -> u64 {
        10 * position as u64 + 3
    }

    /// The rows at `positions`, in that order, named `row_ids`.
    fn store_named(
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

    fn store_of(vectors: &FixedSizeListArray, positions: &[u32]) -> FlatFloatStorage {
        store_named(
            vectors,
            positions,
            positions.iter().map(|p| row_id(*p)).collect(),
        )
    }

    /// Insertion only ever adds a new vertex to an old vertex's list or
    /// re-prunes the list with it, so every old vertex's new out-edges come
    /// from its old ones or from the new vertices. A rebuild breaks this.
    fn extends(
        result: &Vamana,
        base: &Vamana,
        base_position: &dyn Fn(u32) -> Option<u32>,
        is_checked: &dyn Fn(u32) -> bool,
        first_new: u32,
    ) -> bool {
        (0..base.len() as u32)
            .filter(|old| is_checked(*old))
            .all(|old| {
                let Some(position) = base_position(old) else {
                    return true;
                };
                let allowed = base
                    .graph()
                    .neighbors(old)
                    .iter()
                    .filter_map(|neighbor| base_position(*neighbor))
                    .collect::<HashSet<_>>();
                result
                    .graph()
                    .neighbors(position)
                    .iter()
                    .all(|neighbor| *neighbor >= first_new || allowed.contains(neighbor))
            })
    }

    #[test]
    fn new_rows_are_linked_into_the_base_graph() {
        let vectors = vectors(800);
        let base =
            Vamana::index_vectors(&store_of(&vectors, &(0..600).collect::<Vec<_>>()), params())
                .unwrap();
        let merged = store_of(&vectors, &(0..800).collect::<Vec<_>>());
        let result = build_incremental(&base, &merged, 600, params()).unwrap();
        assert_eq!(
            result.graph().row_ids(),
            merged.row_ids().copied().collect::<Vec<_>>()
        );
        assert!(extends(&result, &base, &|old| Some(old), &|_| true, 600));
        assert_ne!(
            result.to_batch().unwrap(),
            Vamana::index_vectors(&merged, params())
                .unwrap()
                .to_batch()
                .unwrap()
        );
    }

    #[test]
    fn base_rows_filtered_out_of_the_merge_are_consolidated() {
        let vectors = vectors(800);
        let base =
            Vamana::index_vectors(&store_of(&vectors, &(0..600).collect::<Vec<_>>()), params())
                .unwrap();
        let kept = (0..600u32).filter(|p| p % 7 != 0).collect::<Vec<_>>();
        let positions = kept.iter().copied().chain(600..800).collect::<Vec<_>>();
        let merged = store_of(&vectors, &positions);
        let result = build_incremental(&base, &merged, kept.len(), params()).unwrap();
        assert_eq!(
            result.graph().row_ids(),
            merged.row_ids().copied().collect::<Vec<_>>()
        );
        let position_of = |old: u32| kept.binary_search(&old).ok().map(|at| at as u32);
        // Only lists that lost no neighbor are bound by the old edges.
        let untouched = |old: u32| base.graph().neighbors(old).iter().all(|n| n % 7 != 0);
        assert!(extends(
            &result,
            &base,
            &position_of,
            &untouched,
            kept.len() as u32
        ));
    }

    #[rstest]
    #[case::other_params(600, 600, 0, Some(params().with_max_degree(12)))]
    #[case::nothing_matches(600, 0, 200, None)]
    #[case::too_many_new_rows(200, 200, 450, None)]
    #[case::most_of_the_base_died(600, 250, 100, None)]
    fn too_little_carried_over_is_a_fresh_build(
        #[case] base_rows: u32,
        #[case] kept_rows: u32,
        #[case] new_rows: u32,
        #[case] merge_params: Option<VamanaBuildParams>,
    ) {
        let vectors = vectors(1500);
        let base = Vamana::index_vectors(
            &store_of(&vectors, &(0..base_rows).collect::<Vec<_>>()),
            params(),
        )
        .unwrap();
        let positions = (0..kept_rows)
            .chain(1000..1000 + new_rows)
            .collect::<Vec<_>>();
        let merged = store_of(&vectors, &positions);
        let merge_params = merge_params.unwrap_or_else(params);
        let result =
            build_incremental(&base, &merged, kept_rows as usize, merge_params.clone()).unwrap();
        let fresh = Vamana::index_vectors(&merged, merge_params).unwrap();
        assert_eq!(result.to_batch().unwrap(), fresh.to_batch().unwrap());
    }

    /// The last ten base rows were updated under stable row ids: gone from
    /// the base's prefix and back among the new rows, under the same ids and
    /// with new vectors. Matched by id, their old vertices would keep the
    /// edges of the old vectors and never be inserted. The graph must be the
    /// one fresh ids give.
    #[test]
    fn a_reused_row_id_names_a_new_row() {
        let vectors = vectors(800);
        let base =
            Vamana::index_vectors(&store_of(&vectors, &(0..600).collect::<Vec<_>>()), params())
                .unwrap();
        let positions = (0..590).chain(600..800).collect::<Vec<u32>>();
        let merged_with = |updated_ids: &dyn Fn(u32) -> u64| {
            let row_ids = positions
                .iter()
                .map(|p| match p {
                    600..610 => updated_ids(*p),
                    _ => row_id(*p),
                })
                .collect();
            let merged = store_named(&vectors, &positions, row_ids);
            build_incremental(&base, &merged, 590, params()).unwrap()
        };
        let reused = merged_with(&|p| row_id(p - 10));
        let fresh = merged_with(&|p| row_id(p));
        assert_eq!(reused.graph().row_ids()[590], row_id(590));
        assert_eq!(reused.metadata(), fresh.metadata());
        for vertex in 0..790 {
            assert_eq!(
                reused.graph().neighbors(vertex),
                fresh.graph().neighbors(vertex),
                "vertex {vertex}"
            );
        }
    }

    #[test]
    fn a_base_row_count_past_the_merged_rows_is_refused() {
        let vectors = vectors(100);
        let base =
            Vamana::index_vectors(&store_of(&vectors, &(0..50).collect::<Vec<_>>()), params())
                .unwrap();
        let error = build_incremental(
            &base,
            &store_of(&vectors, &(0..60).collect::<Vec<_>>()),
            61,
            params(),
        )
        .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("61 rows"), "{error}");
    }

    /// The torn line merged with two new rows: the side cut off is inserted
    /// again and the new rows after it, rather than the partition rebuilt.
    #[test]
    fn a_merge_that_tears_the_base_apart_relinks_it() {
        let (base, _) = torn_line();
        let line = vectors_of((0..12).map(|i| i as f32).collect(), 1);
        let positions = (0..12).filter(|p| *p != 4 && *p != 5).collect::<Vec<u32>>();
        let merged = store_of(&line, &positions);
        let result = build_incremental(&base, &merged, 8, base.params().clone()).unwrap();
        let graph = result.graph();
        // Carried over as it was, in an order no rebuild writes.
        assert_eq!(graph.neighbors(0), &[2, 1]);
        assert!((0..10).all(|vertex| !graph.neighbors(vertex).is_empty()));
        assert_eq!(reachable(graph, result.metadata().medoid, |_| true), 10);
        // The new row at 10 found its nearest, the survivor at 9 on the side
        // cut off: that side was linked in again before the new rows went in.
        assert!(graph.neighbors(8).contains(&7), "{:?}", graph.neighbors(8));
    }

    /// The same graph merged with no new rows. The merged rows' medoid is 3,
    /// from which 0 is reached from nowhere even after it is inserted again,
    /// so the partition is rebuilt.
    #[test]
    fn a_merge_that_leaves_a_survivor_cut_off_is_rebuilt() {
        let (base, line) = cut_off_after_relink();
        let merged = store_of(&line, &(0..5).collect::<Vec<u32>>());
        let result = build_incremental(&base, &merged, 5, base.params().clone()).unwrap();
        let rebuilt = Vamana::index_vectors(&merged, base.params().clone()).unwrap();
        // Kept, the graph would still hold 3 -> 4, which no rebuild writes.
        assert_ne!(rebuilt.graph().neighbors(3), &[4]);
        assert_eq!(result.to_batch().unwrap(), rebuilt.to_batch().unwrap());
    }

    /// Three vertices at one out-edge each, 0 -> 1 -> 2 -> 1 from the medoid
    /// 0, merged with a row at 11. The new row takes 1's one edge from 2, and
    /// the merged rows' medoid, 1, reaches neither 0 nor 2 even after they
    /// are inserted again, so the partition is rebuilt.
    #[test]
    fn a_merge_whose_new_rows_cut_a_survivor_off_is_rebuilt() {
        let params = VamanaBuildParams {
            max_degree: 1,
            search_list_size: 8,
            alpha: 1.2,
        };
        let graph = PartitionGraph::try_new(
            1,
            (0..3).map(row_id).collect(),
            &[vec![1], vec![2], vec![1]],
        )
        .unwrap();
        let base = Vamana::from_graph(&graph, 0, vec![], params.clone()).unwrap();
        let line = vectors_of(vec![0.0, 10.0, 13.0, 11.0], 1);
        let merged = store_of(&line, &[0, 1, 2, 3]);
        let result = build_incremental(&base, &merged, 3, params.clone()).unwrap();
        let rebuilt = Vamana::index_vectors(&merged, params).unwrap();
        // Kept, 2 would still point at 1 though the new row is nearer.
        assert_ne!(rebuilt.graph().neighbors(2), &[1]);
        assert_eq!(result.to_batch().unwrap(), rebuilt.to_batch().unwrap());
    }
}
