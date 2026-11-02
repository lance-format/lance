// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Batched insertion of a delta HNSW graph into a base graph.
//!
//! Every delta vector is searched against the frozen base graph in parallel,
//! then all forward edges are written and each affected node is pruned once
//! for the whole batch, instead of once per inserted vector. Delta vectors
//! never see each other through the base search, so each one also takes its
//! neighbors in the delta graph as extra candidates.

use std::ops::Range;
use std::sync::RwLock;

use lance_core::{Error, Result};
use rayon::prelude::*;

use super::builder::{BuilderLevel0, HNSW, HnswLevelView, HnswQueryParams};
use super::select_neighbors_heuristic_owned;
use crate::vector::graph::builder::GraphBuilderNode;
use crate::vector::graph::{OrderedNode, VisitedGenerator, beam_search, greedy_search};
use crate::vector::hnsw::builder::connect_stranded_level0;
use crate::vector::storage::{DistCalculator, VectorStore};

impl HNSW {
    /// Merge `delta` into `self` without rebuilding either graph.
    ///
    /// `storage` holds the vectors of the merged graph: the `self.len()` base
    /// vectors first, then the `delta.len()` delta vectors, so delta node `i`
    /// becomes node `self.len() + i`. Delta nodes keep the levels they already
    /// have. The result uses the build parameters of `self`.
    ///
    /// This is standalone: nothing in the index build path calls it.
    ///
    /// # Examples
    ///
    /// ```
    /// use arrow_array::{FixedSizeListArray, Float32Array};
    /// use lance_arrow::FixedSizeListArrayExt;
    /// use lance_index::vector::flat::storage::FlatFloatStorage;
    /// use lance_index::vector::hnsw::{HNSW, builder::HnswBuildParams};
    /// use lance_index::vector::v3::subindex::IvfSubIndex;
    /// use lance_linalg::distance::DistanceType;
    ///
    /// # fn main() -> lance_core::Result<()> {
    /// let values = Float32Array::from_iter_values((0..64 * 4).map(|i| ((i * 37) % 101) as f32));
    /// let vectors = FixedSizeListArray::try_new_from_values(values, 4)?;
    /// let split = 48;
    /// let storage_of = |start: usize, len: usize| {
    ///     FlatFloatStorage::new(vectors.slice(start, len), DistanceType::L2)
    /// };
    /// let params = HnswBuildParams::default().num_edges(8).ef_construction(16);
    /// let base = HNSW::index_vectors(&storage_of(0, split), params.clone())?;
    /// let delta = HNSW::index_vectors(&storage_of(split, 64 - split), params)?;
    ///
    /// let merged = base.merge_delta(&delta, &storage_of(0, 64))?;
    /// assert_eq!(merged.len(), 64);
    /// # Ok(())
    /// # }
    /// ```
    pub fn merge_delta(&self, delta: &Self, storage: &impl VectorStore) -> Result<Self> {
        let base_len = self.len();
        let delta_len = delta.len();
        if storage.len() != base_len + delta_len {
            return Err(Error::invalid_input(format!(
                "merge storage has {} rows, but the base graph has {base_len} nodes and the \
                 delta graph has {delta_len}",
                storage.len()
            )));
        }
        if delta.is_empty() {
            return Ok(self.clone());
        }
        if self.is_empty() {
            return Ok(delta.clone());
        }

        let params = self.build_params().clone();
        params.validate()?;
        let total = base_len + delta_len;
        let base_levels = self.max_level() as usize;
        let delta_levels: Vec<usize> = (0..delta_len as u32)
            .map(|id| delta.node_num_levels(id))
            .collect();
        let merged_levels = delta_levels
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .max(base_levels);
        let entry_point = if delta.max_level() as usize > base_levels {
            base_len as u32 + delta.entry_point()
        } else {
            self.entry_point()
        };

        let mut nodes = self.clone_nodes(storage);
        nodes.extend(
            delta_levels
                .iter()
                .enumerate()
                .map(|(i, levels)| GraphBuilderNode::new((base_len + i) as u32, *levels)),
        );
        let nodes: Vec<RwLock<GraphBuilderNode>> = nodes.into_iter().map(RwLock::new).collect();

        // Every search reads the base graph only: delta nodes have no edges yet,
        // so no traversal can reach one before the writes below.
        let selected: Vec<Vec<Vec<OrderedNode>>> = (0..delta_len)
            .into_par_iter()
            .map_init(
                || VisitedGenerator::new(total),
                |visited_generator, i| {
                    let q = (base_len + i) as u32;
                    let dist_calc = storage.dist_calculator_from_id(q);
                    let mut ep = OrderedNode::new(
                        self.entry_point(),
                        dist_calc.distance(self.entry_point()).into(),
                    );
                    for level in (delta_levels[i]..base_levels).rev() {
                        let view = HnswLevelView::new(level as u16, &nodes);
                        ep = greedy_search(&view, ep, &dist_calc, params.prefetch_distance).0;
                    }

                    let mut per_level = vec![Vec::new(); delta_levels[i]];
                    for level in (0..delta_levels[i]).rev() {
                        // Levels above the base graph have no base node to
                        // search; only the delta graph supplies candidates there.
                        let mut candidates = if level < base_levels {
                            let view = HnswLevelView::new(level as u16, &nodes);
                            let mut visited = visited_generator.generate(total);
                            let found = beam_search(
                                &view,
                                &ep,
                                &HnswQueryParams {
                                    ef: params.ef_construction,
                                    lower_bound: None,
                                    upper_bound: None,
                                    dist_q_c: 0.0,
                                    use_acorn: false,
                                },
                                &dist_calc,
                                None,
                                params.prefetch_distance,
                                &mut visited,
                            )
                            .0;
                            ep = found[0].clone();
                            found
                        } else {
                            Vec::new()
                        };
                        candidates.extend(delta.neighbors_at(level, i as u32).into_iter().map(
                            |neighbor| {
                                let id = base_len as u32 + neighbor;
                                OrderedNode::new(id, dist_calc.distance(id).into())
                            },
                        ));
                        per_level[level] =
                            select_neighbors_heuristic_owned(storage, candidates, params.m);
                    }
                    per_level
                },
            )
            .collect();

        for level in 0..merged_levels {
            let limit = if level == 0 { params.m * 2 } else { params.m };

            let mut reverse_edges: Vec<(u32, OrderedNode)> = Vec::new();
            for (i, per_level) in selected.iter().enumerate() {
                let Some(forward) = per_level.get(level) else {
                    continue;
                };
                let q = (base_len + i) as u32;
                let mut node = nodes[q as usize].write().unwrap();
                node.level_neighbors_ranked[level].clone_from(forward);
                node.update_from_ranked_neighbors(level as u16);
                reverse_edges.extend(
                    forward
                        .iter()
                        .map(|target| (target.id, OrderedNode::new(q, target.dist))),
                );
            }

            // Each target is pruned once, however many delta nodes picked it.
            reverse_edges.par_sort_unstable_by_key(|(target, from)| (*target, from.id));
            let groups = group_by_target(&reverse_edges);
            groups.into_par_iter().for_each(|group| {
                let target = reverse_edges[group.start].0;
                let mut node = nodes[target as usize].write().unwrap();
                for (_, from) in &reverse_edges[group] {
                    // A delta pair that picked each other already has this edge.
                    if node.level_neighbors_ranked[level]
                        .iter()
                        .all(|existing| existing.id != from.id)
                    {
                        node.add_neighbor(from.id, from.dist, level as u16);
                    }
                }
                if node.level_neighbors_ranked[level].len() > limit {
                    let ranked = std::mem::take(&mut node.level_neighbors_ranked[level]);
                    node.level_neighbors_ranked[level] =
                        select_neighbors_heuristic_owned(storage, ranked, limit);
                }
                node.update_from_ranked_neighbors(level as u16);
            });
        }

        connect_stranded_level0(
            &mut BuilderLevel0 { nodes: &nodes },
            entry_point,
            params.ef_construction,
            storage,
        );

        let nodes: Vec<GraphBuilderNode> = nodes
            .into_iter()
            .map(|node| node.into_inner().expect("merge lock poisoned"))
            .collect();
        let entry_levels = nodes[entry_point as usize].level_neighbors.len();
        let level_count = (0..entry_levels)
            .map(|level| {
                nodes
                    .iter()
                    .filter(|node| node.level_neighbors.len() > level)
                    .count()
            })
            .collect();
        Ok(Self::from_parts(params, nodes, level_count, entry_point))
    }

    /// Copy this graph into builder nodes, recomputing the neighbor distances
    /// a disk-loaded graph does not carry.
    fn clone_nodes(&self, storage: &impl VectorStore) -> Vec<GraphBuilderNode> {
        if let Some(nodes) = self.nodes() {
            return nodes.as_ref().clone();
        }
        (0..self.len() as u32)
            .into_par_iter()
            .map(|id| {
                let levels = self.node_num_levels(id);
                let mut node = GraphBuilderNode::new(id, levels);
                for level in 0..levels {
                    for neighbor in self.neighbors_at(level, id) {
                        node.add_neighbor(
                            neighbor,
                            storage.dist_between(id, neighbor).into(),
                            level as u16,
                        );
                    }
                    node.update_from_ranked_neighbors(level as u16);
                }
                node
            })
            .collect()
    }
}

/// Ranges of `edges`, sorted by target, that share one target.
fn group_by_target(edges: &[(u32, OrderedNode)]) -> Vec<Range<usize>> {
    let mut groups = Vec::new();
    let mut start = 0;
    for end in 1..=edges.len() {
        if end == edges.len() || edges[end].0 != edges[start].0 {
            groups.push(start..end);
            start = end;
        }
    }
    groups
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use arrow_array::{Array, ArrayRef, FixedSizeListArray};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::Error;
    use lance_linalg::distance::DistanceType;
    use lance_testing::datagen::generate_random_array_with_seed;
    use rstest::rstest;

    use crate::vector::flat::storage::FlatFloatStorage;
    use crate::vector::hnsw::HNSW;
    use crate::vector::hnsw::builder::{HnswBuildParams, HnswQueryParams};
    use crate::vector::storage::{DistCalculator, VectorStore};
    use crate::vector::v3::subindex::IvfSubIndex;

    const DIM: usize = 16;
    const BASE: usize = 1500;
    const DELTA: usize = 150;

    fn vectors(n: usize) -> FixedSizeListArray {
        FixedSizeListArray::try_new_from_values(
            generate_random_array_with_seed::<arrow_array::types::Float32Type>(n * DIM, [7; 32]),
            DIM as i32,
        )
        .unwrap()
    }

    fn storage(vectors: &FixedSizeListArray, start: usize, len: usize) -> FlatFloatStorage {
        FlatFloatStorage::new(vectors.slice(start, len), DistanceType::L2)
    }

    fn params() -> HnswBuildParams {
        HnswBuildParams::default().num_edges(12).ef_construction(40)
    }

    fn search(hnsw: &HNSW, store: &FlatFloatStorage, query: ArrayRef, k: usize) -> Vec<u32> {
        let query_params = HnswQueryParams {
            ef: 64,
            lower_bound: None,
            upper_bound: None,
            dist_q_c: 0.0,
            use_acorn: false,
        };
        let (results, _) = hnsw
            .search_basic(query, k, &query_params, None, store)
            .unwrap();
        results.into_iter().map(|node| node.id).collect()
    }

    fn brute_force(store: &FlatFloatStorage, query: ArrayRef, k: usize) -> Vec<u32> {
        let dist_calc = store.dist_calculator(query, 0.0);
        let mut all: Vec<(f32, u32)> = (0..store.len() as u32)
            .map(|id| (dist_calc.distance(id), id))
            .collect();
        all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        all.into_iter().take(k).map(|(_, id)| id).collect()
    }

    /// Fraction of the exact top-`k` that a search over `hnsw` returns, over a
    /// mix of base and delta vectors as queries.
    fn recall(hnsw: &HNSW, vectors: &FixedSizeListArray, store: &FlatFloatStorage) -> f32 {
        const K: usize = 10;
        let total = vectors.len();
        let queries: Vec<usize> = (0..50).map(|i| i * total / 50).collect();
        let hits: usize = queries
            .iter()
            .map(|&q| {
                let truth: HashSet<u32> = brute_force(store, vectors.value(q), K)
                    .into_iter()
                    .collect();
                search(hnsw, store, vectors.value(q), K)
                    .iter()
                    .filter(|id| truth.contains(id))
                    .count()
            })
            .sum();
        hits as f32 / (queries.len() * K) as f32
    }

    /// Every node is reachable from the entry point on level 0 and no list
    /// exceeds its degree bound (`connect_stranded_level0` may add one edge).
    fn assert_well_formed(hnsw: &HNSW, params: &HnswBuildParams) {
        let mut seen = vec![false; hnsw.len()];
        let mut stack = vec![hnsw.entry_point()];
        seen[hnsw.entry_point() as usize] = true;
        while let Some(id) = stack.pop() {
            for neighbor in hnsw.neighbors_at(0, id) {
                if !seen[neighbor as usize] {
                    seen[neighbor as usize] = true;
                    stack.push(neighbor);
                }
            }
        }
        assert!(seen.iter().all(|reached| *reached));

        for id in 0..hnsw.len() as u32 {
            for level in 0..hnsw.node_num_levels(id) {
                let limit = if level == 0 {
                    2 * params.m + 1
                } else {
                    params.m
                };
                let neighbors = hnsw.neighbors_at(level, id);
                assert!(neighbors.len() <= limit, "node {id} level {level}");
                assert!(!neighbors.contains(&id));
                assert!(neighbors.iter().all(|n| hnsw.node_num_levels(*n) > level));
            }
        }
    }

    #[rstest]
    #[case::built_base_built_delta(false)]
    #[case::loaded_base_loaded_delta(true)]
    fn test_merge_delta_recall_and_structure(#[case] load: bool) {
        let all = vectors(BASE + DELTA);
        let params = params();
        let mut base = HNSW::index_vectors(&storage(&all, 0, BASE), params.clone()).unwrap();
        let mut delta = HNSW::index_vectors(&storage(&all, BASE, DELTA), params.clone()).unwrap();
        if load {
            base = HNSW::load(base.to_batch().unwrap()).unwrap();
            delta = HNSW::load(delta.to_batch().unwrap()).unwrap();
        }

        let merged_store = storage(&all, 0, BASE + DELTA);
        let merged = base.merge_delta(&delta, &merged_store).unwrap();

        assert_eq!(merged.len(), BASE + DELTA);
        assert_eq!(merged.num_nodes(0), BASE + DELTA);
        assert_well_formed(&merged, &params);
        assert!(recall(&merged, &all, &merged_store) >= 0.9);

        // The merged graph survives the on-disk round trip.
        let reloaded = HNSW::load(merged.to_batch().unwrap()).unwrap();
        assert_eq!(reloaded.len(), merged.len());
        assert!(recall(&reloaded, &all, &merged_store) >= 0.9);
    }

    #[test]
    fn test_merge_delta_finds_every_delta_vector() {
        let all = vectors(BASE + DELTA);
        let base = HNSW::index_vectors(&storage(&all, 0, BASE), params()).unwrap();
        let delta = HNSW::index_vectors(&storage(&all, BASE, DELTA), params()).unwrap();
        let merged_store = storage(&all, 0, BASE + DELTA);
        let merged = base.merge_delta(&delta, &merged_store).unwrap();

        let found = (BASE..BASE + DELTA)
            .filter(|&id| search(&merged, &merged_store, all.value(id), 1) == vec![id as u32])
            .count();
        assert!(found >= DELTA * 95 / 100, "found {found} of {DELTA}");
    }

    #[test]
    fn test_merge_delta_handles_delta_taller_than_base() {
        // A one-node base has a single level, so any delta level above 0 lies
        // outside the base graph and is wired from delta edges alone.
        let all = vectors(1 + 600);
        let params = params();
        let base = HNSW::index_vectors(&storage(&all, 0, 1), params.clone()).unwrap();
        let delta = HNSW::index_vectors(&storage(&all, 1, 600), params.clone()).unwrap();
        assert!(delta.max_level() > base.max_level());

        let merged_store = storage(&all, 0, 601);
        let merged = base.merge_delta(&delta, &merged_store).unwrap();
        assert_eq!(merged.max_level(), delta.max_level());
        assert_well_formed(&merged, &params);
        assert!(recall(&merged, &all, &merged_store) >= 0.9);
    }

    #[test]
    fn test_merge_delta_with_an_empty_side() {
        let all = vectors(200);
        let full = HNSW::index_vectors(&storage(&all, 0, 200), params()).unwrap();
        let empty = HNSW::empty();
        let store = storage(&all, 0, 200);

        assert_eq!(full.merge_delta(&empty, &store).unwrap().len(), 200);
        assert_eq!(empty.merge_delta(&full, &store).unwrap().len(), 200);
    }

    #[test]
    fn test_merge_delta_rejects_mismatched_storage() {
        let all = vectors(100);
        let base = HNSW::index_vectors(&storage(&all, 0, 60), params()).unwrap();
        let delta = HNSW::index_vectors(&storage(&all, 60, 40), params()).unwrap();

        let error = base.merge_delta(&delta, &storage(&all, 0, 99)).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("99 rows"), "{error}");
    }
}
