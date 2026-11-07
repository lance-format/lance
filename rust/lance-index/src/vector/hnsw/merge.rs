// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Merging the base and delta HNSW graphs of a partition.
//!
//! [`decide_deltas`] merges a partition once its deltas and tombstones are
//! estimated to slow queries past a budget. [`merge_deltas`] folds the delta
//! graphs into one graph, either by rebuilding it from the live rows or by a
//! [`HNSW::batched_insert`] of the live delta rows into the base graph.
//!
//! A batched insert searches every inserted vector against the frozen base
//! graph in parallel, then writes all forward edges and prunes each affected
//! node once for the whole batch, instead of once per inserted vector. Inserted
//! vectors never see each other through the base search, so each one also
//! takes its neighbors in its source graph as extra candidates.

use std::ops::Range;
use std::sync::RwLock;

use lance_core::{Error, Result};
use rand::Rng;
use rayon::prelude::*;
use roaring::RoaringBitmap;

use super::builder::{BuilderLevel0, HNSW, HnswLevelView, HnswQueryParams};
use super::select_neighbors_heuristic_owned;
use crate::vector::graph::builder::GraphBuilderNode;
use crate::vector::graph::{OrderedNode, VisitedGenerator, beam_search, greedy_search};
use crate::vector::hnsw::builder::connect_stranded_level0;
use crate::vector::segment_ef::SegmentRows;
use crate::vector::storage::{DistCalculator, VectorStore};
use crate::vector::v3::subindex::IvfSubIndex;

/// Chance that [`merge_deltas`] rebuilds a partition its other rules would
/// batch insert into, so repeated batched inserts cannot degrade a graph
/// forever.
pub const REBUILD_PROBABILITY: f64 = 0.01;

/// Largest fraction of deleted rows in a partition, as
/// `MAX_DELETED_NUMERATOR / MAX_DELETED_DENOMINATOR`, that [`merge_deltas`]
/// keeps as tombstones in the base graph instead of rebuilding.
const MAX_DELETED_NUMERATOR: usize = 3;
const MAX_DELETED_DENOMINATOR: usize = 10;

/// Rows of a source graph to add to a base graph with [`HNSW::batched_insert`].
#[derive(Debug, Clone, Copy)]
pub struct InsertedRows<'a> {
    /// The graph the rows were indexed in, whose edges among the inserted rows
    /// seed their neighbor search.
    pub graph: &'a HNSW,
    /// Node ids of `graph` to insert.
    pub rows: &'a RoaringBitmap,
}

/// One HNSW graph of a partition, base or delta.
#[derive(Debug, Clone, Copy)]
pub struct PartitionGraph<'a> {
    pub graph: &'a HNSW,
    /// Node ids of `graph` whose rows are deleted.
    pub deleted: &'a RoaringBitmap,
}

impl PartitionGraph<'_> {
    fn rows(&self) -> Result<SegmentRows> {
        if let Some(max) = self.deleted.max()
            && max as usize >= self.graph.len()
        {
            return Err(Error::invalid_input(format!(
                "deleted rows include node {max}, but the graph has {} nodes",
                self.graph.len()
            )));
        }
        Ok(SegmentRows {
            rows: self.graph.len(),
            dead_rows: self.deleted.len() as usize,
        })
    }
}

/// How [`merge_deltas`] folds a partition's deltas into one graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaMergeStrategy {
    /// Build a new graph from scratch over the live rows of every graph,
    /// dropping all tombstones.
    Rebuild,
    /// Insert the live rows of every delta into the base graph in one batch.
    /// The base graph keeps its tombstones; delta tombstones are dropped.
    BatchedInsert,
}

/// Row `row` of graph `graph` of a partition: graph 0 is the base and graph
/// `i + 1` is delta `i`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartitionRow {
    pub graph: usize,
    pub row: u32,
}

/// A partition's graphs merged into one, with the vectors it indexes.
#[derive(Debug, Clone)]
pub struct MergedPartition<S> {
    pub strategy: DeltaMergeStrategy,
    pub graph: HNSW,
    /// Node `i` of `graph` holds row `rows[i]` of the partition.
    pub rows: Vec<PartitionRow>,
    pub storage: S,
}

/// Choose how [`merge_deltas`] merges `deltas` into `base`.
///
/// With probability [`REBUILD_PROBABILITY`] the partition is rebuilt. Otherwise
/// it is batch inserted when at most 30% of its rows, base and deltas together,
/// are deleted and the base holds at least as many live rows as all deltas
/// together, and rebuilt when not.
pub fn choose_merge_strategy(
    base: &PartitionGraph<'_>,
    deltas: &[PartitionGraph<'_>],
    rng: &mut impl Rng,
) -> Result<DeltaMergeStrategy> {
    let base_rows = base.rows()?;
    let mut delta_rows = SegmentRows {
        rows: 0,
        dead_rows: 0,
    };
    for delta in deltas {
        let rows = delta.rows()?;
        delta_rows.rows += rows.rows;
        delta_rows.dead_rows += rows.dead_rows;
    }
    Ok(strategy_for(base_rows, delta_rows, rng.random()))
}

/// The strategy for a base and the sum of its deltas, given a uniform draw
/// `roll` from `[0, 1)`.
fn strategy_for(base: SegmentRows, deltas: SegmentRows, roll: f64) -> DeltaMergeStrategy {
    if roll < REBUILD_PROBABILITY {
        return DeltaMergeStrategy::Rebuild;
    }
    let live = |rows: SegmentRows| rows.rows - rows.dead_rows;
    let rows = base.rows + deltas.rows;
    let dead_rows = base.dead_rows + deltas.dead_rows;
    let few_deleted = dead_rows * MAX_DELETED_DENOMINATOR <= rows * MAX_DELETED_NUMERATOR;
    if few_deleted && live(base) >= live(deltas) {
        DeltaMergeStrategy::BatchedInsert
    } else {
        DeltaMergeStrategy::Rebuild
    }
}

/// Merge a partition's `deltas` into its `base` graph, choosing the strategy
/// with [`choose_merge_strategy`].
///
/// `gather` returns the vectors of the given partition rows, in that order;
/// the merged graph indexes them. The result uses the build parameters of
/// `base`.
///
/// This is standalone: nothing in the index build path calls it.
///
/// # Examples
///
/// ```
/// use arrow_array::{Array, FixedSizeListArray, Float32Array};
/// use lance_arrow::FixedSizeListArrayExt;
/// use lance_index::vector::flat::storage::FlatFloatStorage;
/// use lance_index::vector::hnsw::merge::{PartitionGraph, PartitionRow, merge_deltas};
/// use lance_index::vector::hnsw::{HNSW, builder::HnswBuildParams};
/// use lance_index::vector::v3::subindex::IvfSubIndex;
/// use lance_linalg::distance::DistanceType;
/// use roaring::RoaringBitmap;
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
/// let deleted = RoaringBitmap::from_iter([3, 7]);
/// let none = RoaringBitmap::new();
///
/// let merged = merge_deltas(
///     &PartitionGraph { graph: &base, deleted: &deleted },
///     &[PartitionGraph { graph: &delta, deleted: &none }],
///     &mut rand::rng(),
///     |rows: &[PartitionRow]| {
///         let offsets = [0, split as u32];
///         let values: Vec<f32> = rows
///             .iter()
///             .flat_map(|row| {
///                 let vector = vectors.value((offsets[row.graph] + row.row) as usize);
///                 let vector = vector.as_any().downcast_ref::<Float32Array>().unwrap();
///                 vector.values().to_vec()
///             })
///             .collect();
///         let vectors = FixedSizeListArray::try_new_from_values(Float32Array::from(values), 4)?;
///         Ok(FlatFloatStorage::new(vectors, DistanceType::L2))
///     },
/// )?;
/// assert_eq!(merged.graph.len(), merged.rows.len());
/// # Ok(())
/// # }
/// ```
pub fn merge_deltas<S: VectorStore>(
    base: &PartitionGraph<'_>,
    deltas: &[PartitionGraph<'_>],
    rng: &mut impl Rng,
    gather: impl FnOnce(&[PartitionRow]) -> Result<S>,
) -> Result<MergedPartition<S>> {
    let strategy = choose_merge_strategy(base, deltas, rng)?;
    merge_deltas_with(strategy, base, deltas, gather)
}

/// [`merge_deltas`] with a `strategy` chosen by the caller, for one that must
/// prepare the graphs for it, such as repairing deleted rows out of the base
/// before a [`DeltaMergeStrategy::BatchedInsert`].
pub fn merge_deltas_with<S: VectorStore>(
    strategy: DeltaMergeStrategy,
    base: &PartitionGraph<'_>,
    deltas: &[PartitionGraph<'_>],
    gather: impl FnOnce(&[PartitionRow]) -> Result<S>,
) -> Result<MergedPartition<S>> {
    let live = |graph: &PartitionGraph<'_>| {
        let mut live = RoaringBitmap::from_sorted_iter(0..graph.graph.len() as u32)
            .expect("a range is sorted");
        live -= graph.deleted;
        live
    };
    let delta_live: Vec<RoaringBitmap> = deltas.iter().map(live).collect();
    let base_rows = match strategy {
        DeltaMergeStrategy::Rebuild => live(base),
        // The base graph keeps every node, tombstones included.
        DeltaMergeStrategy::BatchedInsert => {
            RoaringBitmap::from_sorted_iter(0..base.graph.len() as u32).expect("a range is sorted")
        }
    };
    let rows: Vec<PartitionRow> = std::iter::once(&base_rows)
        .chain(&delta_live)
        .enumerate()
        .flat_map(|(graph, rows)| rows.iter().map(move |row| PartitionRow { graph, row }))
        .collect();
    let storage = gather(&rows)?;
    if storage.len() != rows.len() {
        return Err(Error::invalid_input(format!(
            "gather returned {} vectors for {} partition rows",
            storage.len(),
            rows.len()
        )));
    }

    let graph = match strategy {
        DeltaMergeStrategy::Rebuild => {
            HNSW::index_vectors(&storage, base.graph.build_params().clone())?
        }
        DeltaMergeStrategy::BatchedInsert => {
            let inserted: Vec<InsertedRows<'_>> = deltas
                .iter()
                .zip(&delta_live)
                .map(|(delta, rows)| InsertedRows {
                    graph: delta.graph,
                    rows,
                })
                .collect();
            base.graph.batched_insert(&inserted, &storage)?
        }
    };
    Ok(MergedPartition {
        strategy,
        graph,
        rows,
        storage,
    })
}

/// Slowdown over a clean base graph that [`decide_deltas`] tolerates by
/// default before merging.
pub const DEFAULT_SLOWDOWN_BUDGET: f64 = 5.0;

/// Smallest `ef` a delta is searched with, as in the default
/// [`SegmentEfParams`](crate::vector::segment_ef::SegmentEfParams).
const DELTA_EF_FLOOR: f64 = 20.0;
/// How much wider a delta's beam is than its share of the live rows, as in
/// the default [`SegmentEfParams`](crate::vector::segment_ef::SegmentEfParams).
const DELTA_EF_MULTIPLIER: f64 = 4.0;
/// Exponent of the cost of base tombstones, fit to measured query latency.
const TOMBSTONE_EXPONENT: f64 = 0.85;

/// Estimated query latency of a partition relative to its base graph without
/// deltas or tombstones, for a query with beam width `ef`.
///
/// Each delta costs `c_i = min(1, max(20 / ef, 4 * rows_i / N))` base searches,
/// where `rows_i` is its live rows and `N` the base's live rows, and base
/// tombstones scale the whole by `(1 - tau)^-0.85`, where `tau` is the
/// fraction of base rows deleted:
///
/// ```text
/// slowdown = (1 - tau)^-0.85 * (1 + sum_i c_i)
/// ```
///
/// On 1M and 4M row partitions this tracks measured latency within about 15%,
/// erring low. It counts rows only, so it costs one pass over the deltas.
pub fn estimated_slowdown(
    base: &PartitionGraph<'_>,
    deltas: &[PartitionGraph<'_>],
    ef: usize,
) -> Result<f64> {
    if ef == 0 {
        return Err(Error::invalid_input("ef must be at least 1, got 0"));
    }
    let base_rows = base.rows()?;
    let delta_live = deltas
        .iter()
        .map(|delta| delta.rows().map(|rows| rows.rows - rows.dead_rows))
        .collect::<Result<Vec<_>>>()?;
    Ok(slowdown_for(base_rows, delta_live, ef))
}

fn slowdown_for(base: SegmentRows, delta_live: impl IntoIterator<Item = usize>, ef: usize) -> f64 {
    let base_live = base.rows - base.dead_rows;
    let floor = DELTA_EF_FLOOR / ef as f64;
    let delta_cost: f64 = delta_live
        .into_iter()
        .map(|live| {
            // A base with no live rows leaves any live delta as the whole
            // partition.
            let share = match (live, base_live) {
                (0, _) => 0.0,
                (_, 0) => f64::INFINITY,
                _ => DELTA_EF_MULTIPLIER * live as f64 / base_live as f64,
            };
            share.max(floor).min(1.0)
        })
        .sum();
    let live_fraction = if base.rows == 0 {
        1.0
    } else {
        base_live as f64 / base.rows as f64
    };
    // A fully deleted base gives an infinite slowdown, which always merges.
    live_fraction.powf(-TOMBSTONE_EXPONENT) * (1.0 + delta_cost)
}

/// Merge a partition's deltas with [`merge_deltas`] if its
/// [`estimated_slowdown`] at beam width `ef` exceeds `slowdown_budget`
/// (about [`DEFAULT_SLOWDOWN_BUDGET`]), and do nothing otherwise.
///
/// Returns the merged partition, or `None` when no merge is needed, in which
/// case `gather` is never called.
///
/// This is standalone: nothing in the index build path calls it.
///
/// # Examples
///
/// ```
/// use arrow_array::{FixedSizeListArray, Float32Array};
/// use lance_arrow::FixedSizeListArrayExt;
/// use lance_index::vector::flat::storage::FlatFloatStorage;
/// use lance_index::vector::hnsw::merge::{
///     DEFAULT_SLOWDOWN_BUDGET, PartitionGraph, PartitionRow, decide_deltas,
/// };
/// use lance_index::vector::hnsw::{HNSW, builder::HnswBuildParams};
/// use lance_index::vector::v3::subindex::IvfSubIndex;
/// use lance_linalg::distance::DistanceType;
/// use roaring::RoaringBitmap;
///
/// # fn main() -> lance_core::Result<()> {
/// let values = Float32Array::from_iter_values((0..64 * 4).map(|i| ((i * 37) % 101) as f32));
/// let vectors = FixedSizeListArray::try_new_from_values(values, 4)?;
/// let params = HnswBuildParams::default().num_edges(8).ef_construction(16);
/// let base = HNSW::index_vectors(&FlatFloatStorage::new(vectors, DistanceType::L2), params)?;
/// let none = RoaringBitmap::new();
///
/// // A clean base without deltas is never merged.
/// let merged = decide_deltas(
///     &PartitionGraph { graph: &base, deleted: &none },
///     &[],
///     128,
///     DEFAULT_SLOWDOWN_BUDGET,
///     &mut rand::rng(),
///     |_: &[PartitionRow]| -> lance_core::Result<FlatFloatStorage> { unreachable!() },
/// )?;
/// assert!(merged.is_none());
/// # Ok(())
/// # }
/// ```
pub fn decide_deltas<S: VectorStore>(
    base: &PartitionGraph<'_>,
    deltas: &[PartitionGraph<'_>],
    ef: usize,
    slowdown_budget: f64,
    rng: &mut impl Rng,
    gather: impl FnOnce(&[PartitionRow]) -> Result<S>,
) -> Result<Option<MergedPartition<S>>> {
    if !slowdown_budget.is_finite() || slowdown_budget < 1.0 {
        return Err(Error::invalid_input(format!(
            "slowdown_budget must be finite and at least 1, got {slowdown_budget}"
        )));
    }
    if estimated_slowdown(base, deltas, ef)? <= slowdown_budget {
        return Ok(None);
    }
    merge_deltas(base, deltas, rng, gather).map(Some)
}

impl HNSW {
    /// Insert `inserted` into `self` in one batch, without rebuilding.
    ///
    /// `storage` holds the vectors of the result: the `self.len()` base vectors
    /// first, then the inserted rows in order, so the `i`-th inserted row
    /// becomes node `self.len() + i`. Inserted rows keep the levels they have in
    /// their source graphs. The result uses the build parameters of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use arrow_array::{FixedSizeListArray, Float32Array};
    /// use lance_arrow::FixedSizeListArrayExt;
    /// use lance_index::vector::flat::storage::FlatFloatStorage;
    /// use lance_index::vector::hnsw::merge::InsertedRows;
    /// use lance_index::vector::hnsw::{HNSW, builder::HnswBuildParams};
    /// use lance_index::vector::v3::subindex::IvfSubIndex;
    /// use lance_linalg::distance::DistanceType;
    /// use roaring::RoaringBitmap;
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
    /// let rows = RoaringBitmap::from_sorted_iter(0..(64 - split) as u32).unwrap();
    /// let merged = base.batched_insert(
    ///     &[InsertedRows { graph: &delta, rows: &rows }],
    ///     &storage_of(0, 64),
    /// )?;
    /// assert_eq!(merged.len(), 64);
    /// # Ok(())
    /// # }
    /// ```
    pub fn batched_insert(
        &self,
        inserted: &[InsertedRows<'_>],
        storage: &impl VectorStore,
    ) -> Result<Self> {
        let base_len = self.len();
        for (i, rows) in inserted.iter().enumerate() {
            if let Some(max) = rows.rows.max()
                && max as usize >= rows.graph.len()
            {
                return Err(Error::invalid_input(format!(
                    "inserted rows {i} include node {max}, but its graph has {} nodes",
                    rows.graph.len()
                )));
            }
        }
        // The graph and node of each inserted row, in the order of `storage`.
        let sources: Vec<(usize, u32)> = inserted
            .iter()
            .enumerate()
            .flat_map(|(graph, rows)| rows.rows.iter().map(move |id| (graph, id)))
            .collect();
        let insert_len = sources.len();
        if storage.len() != base_len + insert_len {
            return Err(Error::invalid_input(format!(
                "batched insert storage has {} rows, but the base graph has {base_len} nodes and \
                 {insert_len} rows are inserted",
                storage.len()
            )));
        }
        if insert_len == 0 {
            return Ok(self.clone());
        }
        // An empty base has no graph to search, so the inserted rows are built
        // from scratch.
        if self.is_empty() {
            return Self::index_vectors(storage, self.build_params().clone());
        }

        let mut new_ids: Vec<Vec<Option<u32>>> = inserted
            .iter()
            .map(|rows| vec![None; rows.graph.len()])
            .collect();
        for (i, (graph, id)) in sources.iter().enumerate() {
            new_ids[*graph][*id as usize] = Some((base_len + i) as u32);
        }

        let params = self.build_params().clone();
        params.validate()?;
        let total = base_len + insert_len;
        let base_levels = self.max_level() as usize;
        let insert_levels: Vec<usize> = sources
            .iter()
            .map(|(graph, id)| inserted[*graph].graph.node_num_levels(*id))
            .collect();
        let insert_max_levels = insert_levels.iter().copied().max().unwrap_or(0);
        let merged_levels = insert_max_levels.max(base_levels);
        let entry_point = if insert_max_levels > base_levels {
            let tallest = insert_levels
                .iter()
                .position(|levels| *levels == insert_max_levels)
                .expect("an inserted row has the most levels");
            (base_len + tallest) as u32
        } else {
            self.entry_point()
        };

        let mut nodes = self.clone_nodes(storage);
        nodes.extend(
            insert_levels
                .iter()
                .enumerate()
                .map(|(i, levels)| GraphBuilderNode::new((base_len + i) as u32, *levels)),
        );
        let nodes: Vec<RwLock<GraphBuilderNode>> = nodes.into_iter().map(RwLock::new).collect();

        // Every search reads the base graph only: inserted nodes have no edges yet,
        // so no traversal can reach one before the writes below.
        let selected: Vec<Vec<Vec<OrderedNode>>> = (0..insert_len)
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
                    for level in (insert_levels[i]..base_levels).rev() {
                        let view = HnswLevelView::new(level as u16, &nodes);
                        ep = greedy_search(&view, ep, &dist_calc, params.prefetch_distance).0;
                    }

                    let mut per_level = vec![Vec::new(); insert_levels[i]];
                    for level in (0..insert_levels[i]).rev() {
                        // Levels above the base graph have no base node to
                        // search; only the source graph supplies candidates there.
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
                        let (graph, source_id) = sources[i];
                        candidates.extend(
                            inserted[graph]
                                .graph
                                .neighbors_at(level, source_id)
                                .into_iter()
                                .filter_map(|neighbor| new_ids[graph][neighbor as usize])
                                .map(|id| OrderedNode::new(id, dist_calc.distance(id).into())),
                        );
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

            // Each target is pruned once, however many inserted nodes picked it.
            reverse_edges.par_sort_unstable_by_key(|(target, from)| (*target, from.id));
            let groups = group_by_target(&reverse_edges);
            groups.into_par_iter().for_each(|group| {
                let target = reverse_edges[group.start].0;
                let mut node = nodes[target as usize].write().unwrap();
                for (_, from) in &reverse_edges[group] {
                    // An inserted pair that picked each other already has this edge.
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
    use lance_linalg::distance::DistanceType;
    use lance_testing::datagen::generate_random_array_with_seed;
    use rstest::rstest;

    use arrow_array::Float32Array;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use roaring::RoaringBitmap;

    use super::*;
    use crate::vector::flat::storage::FlatFloatStorage;
    use crate::vector::hnsw::HNSW;
    use crate::vector::hnsw::builder::{HnswBuildParams, HnswQueryParams};

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

    fn all_rows(graph: &HNSW) -> RoaringBitmap {
        RoaringBitmap::from_sorted_iter(0..graph.len() as u32).unwrap()
    }

    /// Batch insert every node of `delta` into `base`.
    fn insert_all(base: &HNSW, delta: &HNSW, store: &FlatFloatStorage) -> Result<HNSW> {
        let rows = all_rows(delta);
        base.batched_insert(
            &[InsertedRows {
                graph: delta,
                rows: &rows,
            }],
            store,
        )
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
    fn test_batched_insert_recall_and_structure(#[case] load: bool) {
        let all = vectors(BASE + DELTA);
        let params = params();
        let mut base = HNSW::index_vectors(&storage(&all, 0, BASE), params.clone()).unwrap();
        let mut delta = HNSW::index_vectors(&storage(&all, BASE, DELTA), params.clone()).unwrap();
        if load {
            base = HNSW::load(base.to_batch().unwrap()).unwrap();
            delta = HNSW::load(delta.to_batch().unwrap()).unwrap();
        }

        let merged_store = storage(&all, 0, BASE + DELTA);
        let merged = insert_all(&base, &delta, &merged_store).unwrap();

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
    fn test_batched_insert_finds_every_delta_vector() {
        let all = vectors(BASE + DELTA);
        let base = HNSW::index_vectors(&storage(&all, 0, BASE), params()).unwrap();
        let delta = HNSW::index_vectors(&storage(&all, BASE, DELTA), params()).unwrap();
        let merged_store = storage(&all, 0, BASE + DELTA);
        let merged = insert_all(&base, &delta, &merged_store).unwrap();

        let found = (BASE..BASE + DELTA)
            .filter(|&id| search(&merged, &merged_store, all.value(id), 1) == vec![id as u32])
            .count();
        assert!(found >= DELTA * 95 / 100, "found {found} of {DELTA}");
    }

    #[test]
    fn test_batched_insert_handles_delta_taller_than_base() {
        // A one-node base has a single level, so any delta level above 0 lies
        // outside the base graph and is wired from delta edges alone.
        let all = vectors(1 + 600);
        let params = params();
        let base = HNSW::index_vectors(&storage(&all, 0, 1), params.clone()).unwrap();
        let delta = HNSW::index_vectors(&storage(&all, 1, 600), params.clone()).unwrap();
        assert!(delta.max_level() > base.max_level());

        let merged_store = storage(&all, 0, 601);
        let merged = insert_all(&base, &delta, &merged_store).unwrap();
        assert_eq!(merged.max_level(), delta.max_level());
        assert_well_formed(&merged, &params);
        assert!(recall(&merged, &all, &merged_store) >= 0.9);
    }

    #[test]
    fn test_batched_insert_with_an_empty_side() {
        let all = vectors(200);
        let full = HNSW::index_vectors(&storage(&all, 0, 200), params()).unwrap();
        let empty = HNSW::empty();
        let store = storage(&all, 0, 200);

        assert_eq!(insert_all(&full, &empty, &store).unwrap().len(), 200);
        assert_eq!(insert_all(&empty, &full, &store).unwrap().len(), 200);
    }

    #[test]
    fn test_batched_insert_rejects_mismatched_storage() {
        let all = vectors(100);
        let base = HNSW::index_vectors(&storage(&all, 0, 60), params()).unwrap();
        let delta = HNSW::index_vectors(&storage(&all, 60, 40), params()).unwrap();

        let error = insert_all(&base, &delta, &storage(&all, 0, 99)).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("99 rows"), "{error}");
    }

    #[test]
    fn test_batched_insert_rows_from_several_graphs() {
        // Two deltas, each inserting every other row: only those rows join the
        // base, and edges come only from inserted rows of the same delta.
        let all = vectors(BASE + 2 * DELTA);
        let params = params();
        let base = HNSW::index_vectors(&storage(&all, 0, BASE), params.clone()).unwrap();
        let deltas = [BASE, BASE + DELTA].map(|start| {
            HNSW::index_vectors(&storage(&all, start, DELTA), params.clone()).unwrap()
        });
        let rows = RoaringBitmap::from_iter((0..DELTA as u32).step_by(2));
        let inserted = [
            InsertedRows {
                graph: &deltas[0],
                rows: &rows,
            },
            InsertedRows {
                graph: &deltas[1],
                rows: &rows,
            },
        ];
        let picked: Vec<usize> = (0..BASE)
            .chain(
                [BASE, BASE + DELTA]
                    .into_iter()
                    .flat_map(|start| rows.iter().map(move |row| start + row as usize)),
            )
            .collect();
        let picked_vectors = take(&all, &picked);
        let store = FlatFloatStorage::new(picked_vectors.clone(), DistanceType::L2);

        let merged = base.batched_insert(&inserted, &store).unwrap();
        assert_eq!(merged.len(), BASE + DELTA);
        assert_well_formed(&merged, &params);
        assert!(recall(&merged, &picked_vectors, &store) >= 0.9);
    }

    #[test]
    fn test_batched_insert_rejects_rows_outside_graph() {
        let all = vectors(100);
        let base = HNSW::index_vectors(&storage(&all, 0, 60), params()).unwrap();
        let delta = HNSW::index_vectors(&storage(&all, 60, 40), params()).unwrap();
        let rows = RoaringBitmap::from_iter([40]);

        let error = base
            .batched_insert(
                &[InsertedRows {
                    graph: &delta,
                    rows: &rows,
                }],
                &storage(&all, 0, 61),
            )
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("node 40"), "{error}");
    }

    fn rows(rows: usize, dead_rows: usize) -> SegmentRows {
        SegmentRows { rows, dead_rows }
    }

    #[rstest]
    #[case::small_deltas(rows(1000, 0), rows(100, 0), 0.5, DeltaMergeStrategy::BatchedInsert)]
    #[case::random_rebuild(rows(1000, 0), rows(100, 0), 0.005, DeltaMergeStrategy::Rebuild)]
    #[case::deleted_at_limit(rows(100, 30), rows(0, 0), 0.5, DeltaMergeStrategy::BatchedInsert)]
    #[case::deleted_over_limit(rows(100, 31), rows(0, 0), 0.5, DeltaMergeStrategy::Rebuild)]
    // Deleted rows count across base and deltas: 35 of 150, then 50 of 150.
    #[case::deleted_across_graphs(
        rows(100, 10),
        rows(50, 25),
        0.5,
        DeltaMergeStrategy::BatchedInsert
    )]
    #[case::deleted_in_deltas(rows(100, 10), rows(50, 40), 0.5, DeltaMergeStrategy::Rebuild)]
    #[case::live_rows_equal(rows(100, 0), rows(100, 0), 0.5, DeltaMergeStrategy::BatchedInsert)]
    #[case::deltas_outgrow_base(rows(100, 0), rows(101, 0), 0.5, DeltaMergeStrategy::Rebuild)]
    // Live rows compare, not total rows: 80 live in the base against 90.
    #[case::live_deltas_outgrow_base(rows(100, 20), rows(90, 0), 0.5, DeltaMergeStrategy::Rebuild)]
    fn test_strategy_for(
        #[case] base: SegmentRows,
        #[case] deltas: SegmentRows,
        #[case] roll: f64,
        #[case] expected: DeltaMergeStrategy,
    ) {
        assert_eq!(strategy_for(base, deltas, roll), expected);
    }

    #[test]
    fn test_choose_merge_strategy_rebuilds_one_percent() {
        let all = vectors(30);
        let base = HNSW::index_vectors(&storage(&all, 0, 20), params()).unwrap();
        let delta = HNSW::index_vectors(&storage(&all, 20, 10), params()).unwrap();
        let none = RoaringBitmap::new();
        let base = PartitionGraph {
            graph: &base,
            deleted: &none,
        };
        let deltas = [PartitionGraph {
            graph: &delta,
            deleted: &none,
        }];

        let mut rng = SmallRng::seed_from_u64(42);
        let rebuilds = (0..20_000)
            .filter(|_| {
                choose_merge_strategy(&base, &deltas, &mut rng).unwrap()
                    == DeltaMergeStrategy::Rebuild
            })
            .count();
        // 200 expected; the bounds are about 4 standard deviations wide.
        assert!((140..=260).contains(&rebuilds), "{rebuilds} rebuilds");
    }

    #[test]
    fn test_choose_merge_strategy_rejects_deleted_rows_outside_graph() {
        let all = vectors(20);
        let base = HNSW::index_vectors(&storage(&all, 0, 20), params()).unwrap();
        let deleted = RoaringBitmap::from_iter([20]);

        let error = choose_merge_strategy(
            &PartitionGraph {
                graph: &base,
                deleted: &deleted,
            },
            &[],
            &mut SmallRng::seed_from_u64(0),
        )
        .unwrap_err();
        assert!(error.to_string().contains("node 20"), "{error}");
    }

    fn take(vectors: &FixedSizeListArray, ids: &[usize]) -> FixedSizeListArray {
        let values: Vec<f32> = ids
            .iter()
            .flat_map(|id| {
                let vector = vectors.value(*id);
                let vector = vector.as_any().downcast_ref::<Float32Array>().unwrap();
                vector.values().to_vec()
            })
            .collect();
        FixedSizeListArray::try_new_from_values(Float32Array::from(values), DIM as i32).unwrap()
    }

    #[rstest]
    #[case::rebuild(DeltaMergeStrategy::Rebuild)]
    #[case::batched_insert(DeltaMergeStrategy::BatchedInsert)]
    fn test_merge_deltas_with(#[case] strategy: DeltaMergeStrategy) {
        // A base with every 10th row deleted and three deltas with every 5th.
        let starts = [0, BASE, BASE + DELTA, BASE + 2 * DELTA];
        let lens = [BASE, DELTA, DELTA, DELTA];
        let all = vectors(BASE + 3 * DELTA);
        let params = params();
        let graphs: Vec<HNSW> = starts
            .iter()
            .zip(lens)
            .map(|(start, len)| {
                HNSW::index_vectors(&storage(&all, *start, len), params.clone()).unwrap()
            })
            .collect();
        let deleted: Vec<RoaringBitmap> = lens
            .iter()
            .enumerate()
            .map(|(graph, len)| {
                let step = if graph == 0 { 10 } else { 5 };
                RoaringBitmap::from_iter((0..*len as u32).step_by(step))
            })
            .collect();
        let partition: Vec<PartitionGraph<'_>> = graphs
            .iter()
            .zip(&deleted)
            .map(|(graph, deleted)| PartitionGraph { graph, deleted })
            .collect();

        let merged = merge_deltas_with(strategy, &partition[0], &partition[1..], |rows| {
            let ids: Vec<usize> = rows
                .iter()
                .map(|row| starts[row.graph] + row.row as usize)
                .collect();
            Ok(FlatFloatStorage::new(take(&all, &ids), DistanceType::L2))
        })
        .unwrap();

        let base_kept = match strategy {
            DeltaMergeStrategy::Rebuild => BASE - BASE / 10,
            DeltaMergeStrategy::BatchedInsert => BASE,
        };
        assert_eq!(merged.strategy, strategy);
        assert_eq!(merged.rows.len(), base_kept + 3 * (DELTA - DELTA / 5));
        assert_eq!(merged.graph.len(), merged.rows.len());
        // Deleted delta rows are always dropped; deleted base rows only by a rebuild.
        for row in &merged.rows {
            let dead = deleted[row.graph].contains(row.row);
            assert!(!dead || (row.graph == 0 && strategy == DeltaMergeStrategy::BatchedInsert));
        }
        assert_well_formed(&merged.graph, &params);
        let ids: Vec<usize> = merged
            .rows
            .iter()
            .map(|row| starts[row.graph] + row.row as usize)
            .collect();
        assert!(recall(&merged.graph, &take(&all, &ids), &merged.storage) >= 0.9);
    }

    /// Live rows of each of `count` deltas holding `total` rows together.
    fn split(total: usize, count: usize) -> Vec<usize> {
        vec![total / count; count]
    }

    // Partitions of 1M rows at ef 128; slowdowns from the merge rule.
    #[rstest]
    #[case::clean(rows(1 << 20, 0), vec![], 1.0)]
    #[case::tombstones_60(rows(1_000_000, 600_000), vec![], 2.18)]
    #[case::deltas_4(rows(1_000_000, 0), split(125_000, 4), 1.63)]
    #[case::deltas_8(rows(1_000_000, 0), split(125_000, 8), 2.25)]
    #[case::deltas_4_tombstones_35(rows(1_000_000, 350_000), split(125_000, 4), 2.55)]
    #[case::deltas_16_tombstones_20(rows(1_000_000, 200_000), split(125_000, 16), 4.23)]
    #[case::deltas_32(rows(1_000_000, 0), split(125_000, 32), 6.0)]
    // Each delta costs at most a whole base search.
    #[case::large_deltas(rows(1_000_000, 0), split(1_000_000, 4), 5.0)]
    #[case::empty_deltas(rows(1_000_000, 0), vec![0, 0], 1.0 + 2.0 * 20.0 / 128.0)]
    #[case::empty_base(rows(0, 0), vec![10], 2.0)]
    #[case::dead_base(rows(10, 10), vec![], f64::INFINITY)]
    fn test_slowdown_for(
        #[case] base: SegmentRows,
        #[case] delta_live: Vec<usize>,
        #[case] expected: f64,
    ) {
        let slowdown = slowdown_for(base, delta_live, 128);
        assert!(
            slowdown == expected || (slowdown - expected).abs() < 0.01,
            "{slowdown} != {expected}"
        );
    }

    #[test]
    fn test_slowdown_for_small_ef_costs_whole_searches() {
        // An ef below the floor searches every delta with the whole beam.
        assert_eq!(slowdown_for(rows(1000, 0), [1, 1], 10), 3.0);
    }

    /// Run `decide_deltas` on a 200-row base with `dead` rows deleted and
    /// `deltas` deltas of 20 rows, returning whether `gather` ran.
    fn decide(dead: u32, deltas: usize) -> (Option<DeltaMergeStrategy>, bool) {
        let all = vectors(200 + 20 * deltas);
        let graphs: Vec<HNSW> = std::iter::once((0, 200))
            .chain((0..deltas).map(|i| (200 + 20 * i, 20)))
            .map(|(start, len)| HNSW::index_vectors(&storage(&all, start, len), params()).unwrap())
            .collect();
        let deleted = RoaringBitmap::from_sorted_iter(0..dead).unwrap();
        let none = RoaringBitmap::new();
        let partition: Vec<PartitionGraph<'_>> = graphs
            .iter()
            .enumerate()
            .map(|(i, graph)| PartitionGraph {
                graph,
                deleted: if i == 0 { &deleted } else { &none },
            })
            .collect();

        let mut gathered = false;
        let merged = decide_deltas(
            &partition[0],
            &partition[1..],
            128,
            DEFAULT_SLOWDOWN_BUDGET,
            &mut SmallRng::seed_from_u64(1),
            |rows| {
                gathered = true;
                let ids: Vec<usize> = rows
                    .iter()
                    .map(|row| match row.graph {
                        0 => row.row as usize,
                        graph => 200 + 20 * (graph - 1) + row.row as usize,
                    })
                    .collect();
                Ok(FlatFloatStorage::new(take(&all, &ids), DistanceType::L2))
            },
        )
        .unwrap();
        if let Some(merged) = &merged {
            assert_eq!(merged.graph.len(), merged.rows.len());
        }
        (merged.map(|merged| merged.strategy), gathered)
    }

    #[rstest]
    // 1 + 4 * 0.4: within budget, so nothing is gathered.
    #[case::few_deltas(0, 4, None)]
    // 1 + 6 * 0.4 = 3.4, scaled by (0.5)^-0.85 = 1.8 to 6.1: merged, and
    // rebuilt, since half the base is deleted.
    #[case::deltas_and_tombstones(100, 6, Some(DeltaMergeStrategy::Rebuild))]
    // (0.1)^-0.85 = 7.1: merged on tombstones alone.
    #[case::tombstones_only(180, 0, Some(DeltaMergeStrategy::Rebuild))]
    fn test_decide_deltas(
        #[case] dead: u32,
        #[case] deltas: usize,
        #[case] expected: Option<DeltaMergeStrategy>,
    ) {
        let (strategy, gathered) = decide(dead, deltas);
        assert_eq!(strategy, expected);
        assert_eq!(gathered, expected.is_some());
    }

    #[rstest]
    #[case::zero_ef(0, DEFAULT_SLOWDOWN_BUDGET, "ef must be at least 1")]
    #[case::budget_below_one(128, 0.5, "slowdown_budget")]
    #[case::nan_budget(128, f64::NAN, "slowdown_budget")]
    fn test_decide_deltas_rejects_invalid_input(
        #[case] ef: usize,
        #[case] budget: f64,
        #[case] message: &str,
    ) {
        let all = vectors(20);
        let base = HNSW::index_vectors(&storage(&all, 0, 20), params()).unwrap();
        let none = RoaringBitmap::new();
        let error = decide_deltas(
            &PartitionGraph {
                graph: &base,
                deleted: &none,
            },
            &[],
            ef,
            budget,
            &mut SmallRng::seed_from_u64(0),
            |_| Ok(storage(&all, 0, 20)),
        )
        .unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
    }
}
