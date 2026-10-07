// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Carry a serialized HNSW graph across a row-address remap.
//!
//! HNSW edges name local vector ids (row numbers in the partition storage),
//! not row addresses. Compaction rewrites row addresses and may delete rows,
//! but every storage `remap` keeps the surviving vectors in their original
//! order. So a remap never invalidates the surviving edges: without deletions
//! the graph is unchanged. With deletions, surviving edges stay, and each node
//! that lost a neighbor gets a new list from the builder's own search: a
//! construction-`ef` beam over the surviving graph, plus the deleted node's
//! other neighbors, chosen with the same heuristic used at build time.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{ArrayBuilder, AsArray, Float32Builder, ListBuilder, UInt32Builder};
use arrow::datatypes::{Float32Type, UInt32Type};
use arrow_array::{Array, RecordBatch};
use itertools::Itertools;
use lance_core::{Error, Result};
use rayon::prelude::*;

use super::builder::{HNSW_METADATA_KEY, HnswQueryParams, Level0Links, connect_stranded_level0};
use super::{HNSW, HnswMetadata, VECTOR_ID_COL, select_neighbors_heuristic_owned};
use crate::vector::DIST_COL;
use crate::vector::graph::{
    BorrowingGraph, NEIGHBORS_COL, OrderedNode, VisitedGenerator, beam_search_borrowed,
    greedy_search_borrowed,
};
use crate::vector::storage::{DistCalculator, VectorStore};
use crate::vector::v3::subindex::IvfSubIndex;

/// Rewrite a serialized HNSW graph after some of its nodes were removed.
///
/// `graph` is the batch written by [`HNSW::to_batch`], including the distance
/// column. `new_ids[old_id]` is the id node `old_id` takes after the removal,
/// or `None` if it was removed; surviving ids must keep their relative order,
/// which is what every vector storage `remap` produces.
///
/// Surviving adjacency lists and their distances are kept, minus the edges to
/// removed nodes; nothing is re-linked, so node degree drops by the number of
/// removed neighbors. If the entry point is removed, the first surviving node
/// on the highest non-empty level replaces it, so search still starts from
/// the top of the graph.
///
/// `new_ids` must have one entry per graph node. If no node is removed, the
/// input batch is returned as is. Returns an empty batch if no node survives.
pub fn remap_graph_batch(graph: &RecordBatch, new_ids: &[Option<u32>]) -> Result<RecordBatch> {
    let metadata_json = graph
        .schema_ref()
        .metadata()
        .get(HNSW_METADATA_KEY)
        .ok_or_else(|| Error::index(format!("{HNSW_METADATA_KEY} not found in HNSW batch")))?;
    let metadata: HnswMetadata = serde_json::from_str(metadata_json).map_err(|e| {
        Error::index(format!(
            "Failed to decode HNSW metadata: {e}, json: {metadata_json}"
        ))
    })?;
    let num_nodes = match metadata.level_offsets.as_slice() {
        [start, end, ..] => end.saturating_sub(*start),
        _ => 0,
    };
    if num_nodes != new_ids.len() {
        return Err(Error::invalid_input(format!(
            "HNSW graph has {num_nodes} nodes but the remap covers {} nodes",
            new_ids.len()
        )));
    }
    let column = |name: &str| {
        graph.column_by_name(name).ok_or_else(|| {
            Error::index(format!(
                "HNSW batch has no {name} column; remapping needs every column the writer emits"
            ))
        })
    };
    let ids = column(VECTOR_ID_COL)?
        .as_primitive_opt::<UInt32Type>()
        .ok_or_else(|| Error::index(format!("{VECTOR_ID_COL} must be UInt32")))?;
    let neighbors = column(NEIGHBORS_COL)?
        .as_list_opt::<i32>()
        .ok_or_else(|| Error::index(format!("{NEIGHBORS_COL} must be List<UInt32>")))?;
    let distances = column(DIST_COL)?
        .as_list_opt::<i32>()
        .ok_or_else(|| Error::index(format!("{DIST_COL} must be List<Float32>")))?;
    let neighbor_values = neighbors
        .values()
        .as_primitive_opt::<UInt32Type>()
        .ok_or_else(|| Error::index(format!("{NEIGHBORS_COL} must be List<UInt32>")))?
        .values();
    let distance_values = distances
        .values()
        .as_primitive_opt::<Float32Type>()
        .ok_or_else(|| Error::index(format!("{DIST_COL} must be List<Float32>")))?
        .values();
    if new_ids
        .iter()
        .enumerate()
        .all(|(old_id, new_id)| *new_id == Some(old_id as u32))
    {
        return Ok(graph.clone());
    }
    let new_id = |old_id: u32| new_ids.get(old_id as usize).copied().flatten();

    let num_rows = graph.num_rows();
    let mut id_builder = UInt32Builder::with_capacity(num_rows);
    let mut neighbors_builder = ListBuilder::with_capacity(
        UInt32Builder::with_capacity(neighbor_values.len()),
        num_rows,
    );
    let mut distances_builder = ListBuilder::with_capacity(
        Float32Builder::with_capacity(neighbor_values.len()),
        num_rows,
    );
    let mut level_offsets = Vec::with_capacity(metadata.level_offsets.len());
    level_offsets.push(0);
    // First surviving node of each level, to replace a removed entry point.
    let mut level_first_node = Vec::with_capacity(metadata.level_offsets.len());
    for (&start, &end) in metadata.level_offsets.iter().tuple_windows() {
        if start > end || end > num_rows {
            return Err(Error::index(format!(
                "HNSW level range {start}..{end} is invalid for a batch of {num_rows} rows"
            )));
        }
        let mut first_node = None;
        for row in start..end {
            let Some(node) = new_id(ids.value(row)) else {
                continue;
            };
            first_node.get_or_insert(node);
            id_builder.append_value(node);
            if !neighbors.is_null(row) {
                let edges = neighbors.value_offsets()[row] as usize
                    ..neighbors.value_offsets()[row + 1] as usize;
                let dists = distances.value_offsets()[row] as usize
                    ..distances.value_offsets()[row + 1] as usize;
                if edges.len() != dists.len() {
                    return Err(Error::index(format!(
                        "HNSW row {row} has {} neighbors but {} distances",
                        edges.len(),
                        dists.len()
                    )));
                }
                for (&neighbor, &dist) in neighbor_values[edges].iter().zip(&distance_values[dists])
                {
                    if let Some(neighbor) = new_id(neighbor) {
                        neighbors_builder.values().append_value(neighbor);
                        distances_builder.values().append_value(dist);
                    }
                }
            }
            neighbors_builder.append(true);
            distances_builder.append(true);
        }
        level_offsets.push(id_builder.len());
        level_first_node.push(first_node);
    }

    if id_builder.is_empty() {
        return Ok(RecordBatch::new_empty(HNSW::schema()));
    }
    let entry_point = match new_id(metadata.entry_point) {
        Some(entry_point) => entry_point,
        None => level_first_node
            .iter()
            .rev()
            .find_map(|node| *node)
            .ok_or_else(|| Error::internal("HNSW remap kept nodes on no level".to_string()))?,
    };
    let metadata = HnswMetadata {
        entry_point,
        params: metadata.params,
        level_offsets,
    };

    let mut schema_metadata = graph.schema_ref().metadata().clone();
    schema_metadata.insert(
        HNSW_METADATA_KEY.to_string(),
        serde_json::to_string(&metadata)?,
    );
    let schema = HNSW::schema()
        .as_ref()
        .clone()
        .with_metadata(schema_metadata);
    Ok(RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(id_builder.finish()),
            Arc::new(neighbors_builder.finish()),
            Arc::new(distances_builder.finish()),
        ],
    )?)
}

struct LevelRows {
    rows: Vec<GraphRow>,
}

struct GraphRow {
    /// New id when this node survives. `None` when it was deleted.
    node: Option<u32>,
    is_entry: bool,
    /// Each edge's new id, when that neighbor survives, and its stored distance.
    edges: Vec<(Option<u32>, f32)>,
}

struct ParsedGraph {
    identity: bool,
    num_rows: usize,
    neighbor_count: usize,
    metadata: HnswMetadata,
    /// New id of the entry point when it survives.
    new_entry: Option<u32>,
    levels: Vec<LevelRows>,
}

fn parse_graph(graph: &RecordBatch, new_ids: &[Option<u32>]) -> Result<ParsedGraph> {
    let metadata_json = graph
        .schema_ref()
        .metadata()
        .get(HNSW_METADATA_KEY)
        .ok_or_else(|| Error::index(format!("{HNSW_METADATA_KEY} not found in HNSW batch")))?;
    let metadata: HnswMetadata = serde_json::from_str(metadata_json).map_err(|e| {
        Error::index(format!(
            "Failed to decode HNSW metadata: {e}, json: {metadata_json}"
        ))
    })?;
    let num_nodes = match metadata.level_offsets.as_slice() {
        [start, end, ..] => end.saturating_sub(*start),
        _ => 0,
    };
    if num_nodes != new_ids.len() {
        return Err(Error::invalid_input(format!(
            "HNSW graph has {num_nodes} nodes but the remap covers {} nodes",
            new_ids.len()
        )));
    }
    let column = |name: &str| {
        graph.column_by_name(name).ok_or_else(|| {
            Error::index(format!(
                "HNSW batch has no {name} column; remapping needs every column the writer emits"
            ))
        })
    };
    let ids = column(VECTOR_ID_COL)?
        .as_primitive_opt::<UInt32Type>()
        .ok_or_else(|| Error::index(format!("{VECTOR_ID_COL} must be UInt32")))?;
    let neighbors = column(NEIGHBORS_COL)?
        .as_list_opt::<i32>()
        .ok_or_else(|| Error::index(format!("{NEIGHBORS_COL} must be List<UInt32>")))?;
    let distances = column(DIST_COL)?
        .as_list_opt::<i32>()
        .ok_or_else(|| Error::index(format!("{DIST_COL} must be List<Float32>")))?;
    let neighbor_values = neighbors
        .values()
        .as_primitive_opt::<UInt32Type>()
        .ok_or_else(|| Error::index(format!("{NEIGHBORS_COL} must be List<UInt32>")))?
        .values();
    let distance_values = distances
        .values()
        .as_primitive_opt::<Float32Type>()
        .ok_or_else(|| Error::index(format!("{DIST_COL} must be List<Float32>")))?
        .values();
    let identity = new_ids
        .iter()
        .enumerate()
        .all(|(old_id, new_id)| *new_id == Some(old_id as u32));
    let new_id = |old_id: u32| new_ids.get(old_id as usize).copied().flatten();
    let num_rows = graph.num_rows();
    let mut levels = Vec::with_capacity(metadata.level_offsets.len().saturating_sub(1));
    for (&start, &end) in metadata.level_offsets.iter().tuple_windows() {
        if start > end || end > num_rows {
            return Err(Error::index(format!(
                "HNSW level range {start}..{end} is invalid for a batch of {num_rows} rows"
            )));
        }
        let mut rows = Vec::with_capacity(end - start);
        for row in start..end {
            let old = ids.value(row);
            let mut edges = Vec::new();
            if !neighbors.is_null(row) {
                let edge_range = neighbors.value_offsets()[row] as usize
                    ..neighbors.value_offsets()[row + 1] as usize;
                let dist_range = distances.value_offsets()[row] as usize
                    ..distances.value_offsets()[row + 1] as usize;
                if edge_range.len() != dist_range.len() {
                    return Err(Error::index(format!(
                        "HNSW row {row} has {} neighbors but {} distances",
                        edge_range.len(),
                        dist_range.len()
                    )));
                }
                edges.extend(
                    neighbor_values[edge_range]
                        .iter()
                        .zip(&distance_values[dist_range])
                        .map(|(&neighbor, &dist)| (new_id(neighbor), dist)),
                );
            }
            rows.push(GraphRow {
                node: new_id(old),
                is_entry: old == metadata.entry_point,
                edges,
            });
        }
        levels.push(LevelRows { rows });
    }
    Ok(ParsedGraph {
        identity,
        num_rows,
        neighbor_count: neighbor_values.len(),
        new_entry: new_id(metadata.entry_point),
        metadata,
        levels,
    })
}

fn linked(links: &[(u32, f32)], neighbor: u32) -> bool {
    links.iter().any(|(id, _)| *id == neighbor)
}

/// One HNSW level after deleted nodes have been removed.
struct LevelAdj {
    neighbors: Vec<Vec<(u32, f32)>>,
    /// Neighbor ids only, kept in sync with `neighbors` for search.
    search_ids: Vec<Vec<u32>>,
    /// Degree the node had before deletions, including edges that were dropped.
    target: Vec<usize>,
    on_level: Vec<bool>,
    /// Surviving nodes in their original row order.
    order: Vec<u32>,
    /// Nodes that lost at least one edge.
    damaged: Vec<u32>,
}

struct SearchView<'a> {
    neighbors: &'a [Vec<u32>],
}

impl BorrowingGraph for SearchView<'_> {
    fn len(&self) -> usize {
        self.neighbors.len()
    }

    fn neighbors(&self, key: u32) -> &[u32] {
        &self.neighbors[key as usize]
    }
}

/// Connect the surviving neighbors of each deleted node so search can still
/// step across the hole. The edges are a scaffold: nodes that lost a neighbor
/// are reselected afterwards.
fn fill_holes<S: VectorStore>(level: &mut LevelAdj, holes: &[Vec<u32>], storage: &S) {
    for hole in holes {
        for &node in hole {
            if !level.on_level[node as usize]
                || level.neighbors[node as usize].len() >= level.target[node as usize]
            {
                continue;
            }
            let distances = storage.dist_calculator_from_id(node);
            let mut candidates = Vec::new();
            for &other in hole {
                if other != node && !linked(&level.neighbors[node as usize], other) {
                    candidates.push((other, distances.distance(other)));
                }
            }
            candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            for (other, dist) in candidates {
                if level.neighbors[node as usize].len() >= level.target[node as usize] {
                    break;
                }
                if !linked(&level.neighbors[node as usize], other) {
                    level.neighbors[node as usize].push((other, dist));
                }
            }
        }
    }
}

fn refresh_search_ids(level: &mut LevelAdj) {
    for (ids, edges) in level.search_ids.iter_mut().zip(&level.neighbors) {
        ids.clear();
        ids.extend(edges.iter().map(|(id, _)| *id));
    }
}

/// Neighbor list a damaged node would have been given at build time.
///
/// The search runs on the scaffold (surviving edges plus the hole links), from
/// high levels down to this one, with the same `ef` the graph was built with.
/// Candidates are that beam plus the node's current neighbors, and Algorithm 4
/// keeps `target` of them. Existing edges stay in the candidate set, so a
/// diverse surviving edge is not thrown away just because a closer local edge
/// exists.
fn choose_neighbors<S: VectorStore>(
    level: &LevelAdj,
    higher: &[LevelAdj],
    entry: u32,
    node: u32,
    query: &HnswQueryParams,
    storage: &S,
    visited_gen: &mut VisitedGenerator,
) -> Vec<(u32, f32)> {
    let distances = storage.dist_calculator_from_id(node);
    let on_this_level = |id: u32| level.on_level.get(id as usize).copied().unwrap_or(false);
    let mut ep = if on_this_level(entry) { entry } else { node };
    for higher_level in higher.iter().rev() {
        let on_higher = higher_level
            .on_level
            .get(ep as usize)
            .copied()
            .unwrap_or(false);
        if !on_higher {
            continue;
        }
        let view = SearchView {
            neighbors: &higher_level.search_ids,
        };
        let start = OrderedNode::new(ep, distances.distance(ep).into());
        ep = greedy_search_borrowed(&view, start, &distances, None).id;
    }
    if !on_this_level(ep) {
        ep = node;
    }
    let view = SearchView {
        neighbors: &level.search_ids,
    };
    let start = OrderedNode::new(ep, distances.distance(ep).into());
    let mut visited = visited_gen.generate(level.neighbors.len());
    let found = beam_search_borrowed(&view, &start, query, &distances, None, None, &mut visited);
    drop(visited);

    let mut seen = HashSet::new();
    let mut candidates = Vec::new();
    let mut push = |id: u32, dist: f32| {
        if id != node && seen.insert(id) {
            candidates.push(OrderedNode::new(id, dist.into()));
        }
    };
    for hit in found {
        push(hit.id, hit.dist.0);
    }
    for &(id, dist) in &level.neighbors[node as usize] {
        push(id, dist);
    }
    if candidates.is_empty() {
        return level.neighbors[node as usize].clone();
    }
    let k = level.target[node as usize].max(1);
    select_neighbors_heuristic_owned(storage, candidates, k)
        .into_iter()
        .map(|neighbor| (neighbor.id, neighbor.dist.0))
        .collect()
}

fn reselect_damaged<S: VectorStore + Sync>(
    levels: &mut [LevelAdj],
    entry: u32,
    ef: usize,
    storage: &S,
) {
    let query = HnswQueryParams {
        ef: ef.max(1),
        lower_bound: None,
        upper_bound: None,
        dist_q_c: 0.0,
        use_acorn: false,
    };
    for level_idx in (0..levels.len()).rev() {
        if levels[level_idx].damaged.is_empty() {
            refresh_search_ids(&mut levels[level_idx]);
            continue;
        }
        let (below, higher) = levels.split_at_mut(level_idx + 1);
        let level = &mut below[level_idx];
        refresh_search_ids(level);
        let n = level.neighbors.len();
        let chosen: Vec<Vec<(u32, f32)>> = level
            .damaged
            .par_iter()
            .map_init(
                || VisitedGenerator::new(n),
                |visited_gen, &node| {
                    choose_neighbors(level, higher, entry, node, &query, storage, visited_gen)
                },
            )
            .collect();
        for (&node, edges) in level.damaged.iter().zip(chosen) {
            level.neighbors[node as usize] = edges;
        }
        add_reciprocals(level, storage);
        refresh_search_ids(level);
    }
}

/// Add the reverse of every edge. A list that grows past the degree the node
/// had before the deletion is cut back to that degree with the builder's heuristic.
fn add_reciprocals<S: VectorStore>(level: &mut LevelAdj, storage: &S) {
    let n = level.neighbors.len();
    let mut incoming = vec![Vec::<(u32, f32)>::new(); n];
    for (node, edges) in level.neighbors.iter().enumerate() {
        for &(other, dist) in edges {
            if other as usize >= n || other == node as u32 {
                continue;
            }
            incoming[other as usize].push((node as u32, dist));
        }
    }
    for (node, incoming_edges) in incoming.iter_mut().enumerate() {
        if !level.on_level[node] {
            continue;
        }
        let edges = &mut level.neighbors[node];
        for (src, dist) in incoming_edges.drain(..) {
            if !linked(edges, src) {
                edges.push((src, dist));
            }
        }
        let limit = level.target[node].max(1);
        if edges.len() > limit {
            let candidates = edges
                .iter()
                .map(|&(id, dist)| OrderedNode::new(id, dist.into()))
                .collect();
            *edges = select_neighbors_heuristic_owned(storage, candidates, limit)
                .into_iter()
                .map(|neighbor| (neighbor.id, neighbor.dist.0))
                .collect();
        }
    }
}

fn level_from_rows<S: VectorStore>(
    rows: &[GraphRow],
    kept: usize,
    storage: &S,
    entry_replacement: &mut Option<(u32, f32)>,
) -> LevelAdj {
    let mut level = LevelAdj {
        neighbors: vec![Vec::new(); kept],
        search_ids: vec![Vec::new(); kept],
        target: vec![0; kept],
        on_level: vec![false; kept],
        order: Vec::with_capacity(rows.len()),
        damaged: Vec::new(),
    };
    let mut holes = Vec::new();
    for row in rows {
        let Some(node) = row.node else {
            if row.is_entry
                && let Some(best) = row
                    .edges
                    .iter()
                    .filter_map(|(neighbor, dist)| neighbor.map(|id| (id, *dist)))
                    .min_by(|left, right| left.1.total_cmp(&right.1))
            {
                // Higher levels are visited later and overwrite, so the
                // replacement is the closest survivor on the highest level
                // that still has one.
                *entry_replacement = Some(best);
            }
            let survivors = row
                .edges
                .iter()
                .filter_map(|(neighbor, _)| *neighbor)
                .collect::<Vec<_>>();
            if survivors.len() > 1 {
                holes.push(survivors);
            }
            continue;
        };
        level.on_level[node as usize] = true;
        level.order.push(node);
        level.target[node as usize] = row.edges.len();
        let surviving = row
            .edges
            .iter()
            .filter_map(|(neighbor, dist)| neighbor.map(|id| (id, *dist)))
            .collect::<Vec<_>>();
        if surviving.len() < row.edges.len() {
            level.damaged.push(node);
        }
        level.neighbors[node as usize] = surviving;
    }
    fill_holes(&mut level, &holes, storage);
    level
}

/// Remap a graph and replace edges that pointed at deleted nodes.
///
/// Every edge between two surviving nodes is kept. A node that lost a neighbor
/// is given a new list by searching the surviving graph with the `ef` it was
/// built with and selecting with the builder's heuristic, so the links that
/// used to step through a deleted node are replaced by links a rebuild would
/// have drawn. Nodes that lost nothing keep their edges.
///
/// Reciprocal trimming can drop the only inbound edge of a node. Search only
/// walks level 0 from the entry point, so those nodes are then linked with
/// `connect_stranded_level0`, the same pass the builder runs after a
/// parallel insert.
///
/// `storage` is the remapped partition, in the new local-id order `new_ids`
/// assigns. With no deletions the input batch is returned unchanged.
pub fn remap_graph_repair<S: VectorStore + Sync>(
    graph: &RecordBatch,
    new_ids: &[Option<u32>],
    storage: &S,
) -> Result<RecordBatch> {
    let parsed = parse_graph(graph, new_ids)?;
    if parsed.identity {
        return Ok(graph.clone());
    }
    let kept = new_ids.iter().filter(|id| id.is_some()).count();
    if storage.len() != kept {
        return Err(Error::invalid_input(format!(
            "remapped storage has {} rows but the graph kept {kept} nodes",
            storage.len()
        )));
    }
    if kept == 0 {
        return Ok(RecordBatch::new_empty(HNSW::schema()));
    }

    let mut entry_replacement: Option<(u32, f32)> = None;
    let mut highest_first: Option<u32> = None;
    let mut levels = Vec::with_capacity(parsed.levels.len());
    for level in &parsed.levels {
        let adj = level_from_rows(&level.rows, kept, storage, &mut entry_replacement);
        if let Some(&node) = adj.order.first() {
            highest_first = Some(node);
        }
        levels.push(adj);
    }
    let entry_point = parsed
        .new_entry
        .or(entry_replacement.map(|(id, _)| id))
        .or(highest_first)
        .ok_or_else(|| Error::internal("HNSW repair kept nodes on no level".to_string()))?;
    let ef = parsed.metadata.params.ef_construction.min(kept.max(1));
    reselect_damaged(&mut levels, entry_point, ef, storage);
    {
        let mut level0 = Level0Adj::new(&mut levels[0]);
        connect_stranded_level0(
            &mut level0,
            entry_point,
            parsed.metadata.params.ef_construction,
            storage,
        );
    }
    refresh_search_ids(&mut levels[0]);

    let mut id_builder = UInt32Builder::with_capacity(parsed.num_rows);
    let mut neighbors_builder = ListBuilder::with_capacity(
        UInt32Builder::with_capacity(parsed.neighbor_count),
        parsed.num_rows,
    );
    let mut distances_builder = ListBuilder::with_capacity(
        Float32Builder::with_capacity(parsed.neighbor_count),
        parsed.num_rows,
    );
    let mut level_offsets = Vec::with_capacity(parsed.metadata.level_offsets.len());
    level_offsets.push(0);
    for level in &levels {
        for &node in &level.order {
            id_builder.append_value(node);
            for &(neighbor, dist) in &level.neighbors[node as usize] {
                neighbors_builder.values().append_value(neighbor);
                distances_builder.values().append_value(dist);
            }
            neighbors_builder.append(true);
            distances_builder.append(true);
        }
        level_offsets.push(id_builder.len());
    }

    let metadata = HnswMetadata {
        entry_point,
        params: parsed.metadata.params.clone(),
        level_offsets,
    };
    let mut schema_metadata = graph.schema_ref().metadata().clone();
    schema_metadata.insert(
        HNSW_METADATA_KEY.to_string(),
        serde_json::to_string(&metadata)?,
    );
    let schema = HNSW::schema()
        .as_ref()
        .clone()
        .with_metadata(schema_metadata);
    Ok(RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(id_builder.finish()),
            Arc::new(neighbors_builder.finish()),
            Arc::new(distances_builder.finish()),
        ],
    )?)
}

/// Level 0 of a repaired graph, as the builder's stranded-node linker sees it.
struct Level0Adj<'a> {
    edges: &'a mut [Vec<(u32, f32)>],
    ids: Vec<Arc<Vec<u32>>>,
}

impl<'a> Level0Adj<'a> {
    fn new(level: &'a mut LevelAdj) -> Self {
        let ids = level
            .neighbors
            .iter()
            .map(|edges| Arc::new(edges.iter().map(|(id, _)| *id).collect()))
            .collect();
        Self {
            edges: level.neighbors.as_mut_slice(),
            ids,
        }
    }
}

impl Level0Links for Level0Adj<'_> {
    fn len(&self) -> usize {
        self.edges.len()
    }

    fn neighbors(&self, id: u32) -> Arc<Vec<u32>> {
        self.ids[id as usize].clone()
    }

    fn ranked(&self, id: u32) -> Vec<OrderedNode> {
        self.edges[id as usize]
            .iter()
            .map(|(id, dist)| OrderedNode::new(*id, (*dist).into()))
            .collect()
    }

    fn link(&mut self, anchor: OrderedNode, node: u32) {
        let edges = &mut self.edges[anchor.id as usize];
        edges.push((node, anchor.dist.0));
        self.ids[anchor.id as usize] = Arc::new(edges.iter().map(|(id, _)| *id).collect());
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use arrow::array::AsArray;
    use arrow::compute::take;
    use arrow::datatypes::{Float32Type, UInt32Type};
    use arrow_array::{Array, FixedSizeListArray, RecordBatch, UInt32Array};
    use itertools::Itertools;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::Error;
    use lance_linalg::distance::DistanceType;
    use lance_testing::datagen::generate_random_array_with_seed;
    use rstest::rstest;

    use super::{remap_graph_batch, remap_graph_repair};
    use crate::vector::DIST_COL;
    use crate::vector::flat::storage::FlatFloatStorage;
    use crate::vector::graph::NEIGHBORS_COL;
    use crate::vector::hnsw::builder::{HNSW_METADATA_KEY, HnswBuildParams, HnswQueryParams};
    use crate::vector::hnsw::{HNSW, HnswMetadata, VECTOR_ID_COL};
    use crate::vector::v3::subindex::IvfSubIndex;

    const DIM: usize = 16;
    const TOTAL: usize = 2000;
    const K: usize = 10;

    fn build(total: usize) -> (FixedSizeListArray, HNSW) {
        let data = generate_random_array_with_seed::<Float32Type>(total * DIM, [7; 32]);
        let fsl = FixedSizeListArray::try_new_from_values(data, DIM as i32).unwrap();
        let store = FlatFloatStorage::new(fsl.clone(), DistanceType::L2);
        let hnsw = HNSW::index_vectors(
            &store,
            HnswBuildParams::default().num_edges(16).ef_construction(64),
        )
        .unwrap();
        (fsl, hnsw)
    }

    fn metadata(batch: &RecordBatch) -> HnswMetadata {
        serde_json::from_str(&batch.schema_ref().metadata()[HNSW_METADATA_KEY]).unwrap()
    }

    fn num_level_members(hnsw: &HNSW) -> usize {
        (0..hnsw.max_level() as usize)
            .map(|level| hnsw.num_nodes(level))
            .sum()
    }

    fn query_params() -> HnswQueryParams {
        HnswQueryParams {
            ef: 64,
            lower_bound: None,
            upper_bound: None,
            dist_q_c: 0.0,
            use_acorn: false,
        }
    }

    fn recall(hnsw: &HNSW, vectors: &FixedSizeListArray, queries: &FixedSizeListArray) -> f32 {
        let store = FlatFloatStorage::new(vectors.clone(), DistanceType::L2);
        let mut hits = 0;
        for i in 0..queries.len() {
            let query = queries.value(i);
            let truth = (0..vectors.len())
                .map(|j| {
                    let v = vectors.value(j);
                    let v = v.as_primitive::<Float32Type>();
                    let q = query.as_primitive::<Float32Type>();
                    let dist: f32 = v
                        .values()
                        .iter()
                        .zip(q.values())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    (dist, j as u32)
                })
                .sorted_by(|a, b| a.0.total_cmp(&b.0))
                .take(K)
                .map(|(_, id)| id)
                .collect::<HashSet<_>>();
            let found = hnsw
                .search_basic(query, K, &query_params(), None, &store)
                .unwrap();
            hits += found.iter().filter(|n| truth.contains(&n.id)).count();
        }
        hits as f32 / (queries.len() * K) as f32
    }

    #[test]
    fn test_remap_without_removals_keeps_graph() {
        let (_, hnsw) = build(TOTAL);
        let batch = HNSW::load(hnsw.to_batch().unwrap())
            .unwrap()
            .to_batch()
            .unwrap();
        assert_eq!(batch.num_rows(), num_level_members(&hnsw));

        let new_ids = (0..TOTAL as u32).map(Some).collect::<Vec<_>>();
        let remapped = remap_graph_batch(&batch, &new_ids).unwrap();
        assert_eq!(remapped, batch);
    }

    /// Repair has to stay with a rebuild. The tolerance is wider than the bench
    /// because this graph is small; a broken reconnection misses by much more.
    #[test]
    fn test_repair_recall_stays_with_rebuild() {
        let (vectors, hnsw) = build(TOTAL);
        let batch = hnsw.to_batch().unwrap();
        let params = metadata(&batch).params;
        let queries = generate_random_array_with_seed::<Float32Type>(20 * DIM, [9; 32]);
        let queries = FixedSizeListArray::try_new_from_values(queries, DIM as i32).unwrap();
        for remove_every in [100, 10, 2] {
            let mut new_ids = Vec::with_capacity(TOTAL);
            let mut kept_idx = Vec::new();
            for old_id in 0..TOTAL {
                if old_id % remove_every == 0 {
                    new_ids.push(None);
                } else {
                    new_ids.push(Some(kept_idx.len() as u32));
                    kept_idx.push(old_id as u32);
                }
            }
            let kept_idx = UInt32Array::from(kept_idx);
            let kept = take(&vectors, &kept_idx, None).unwrap();
            let kept = kept.as_fixed_size_list().clone();
            let store = FlatFloatStorage::new(kept.clone(), DistanceType::L2);
            let repaired =
                HNSW::load(remap_graph_repair(&batch, &new_ids, &store).unwrap()).unwrap();
            let rebuilt = HNSW::index_vectors(&store, params.clone()).unwrap();
            let repair_recall = recall(&repaired, &kept, &queries);
            let rebuild_recall = recall(&rebuilt, &kept, &queries);
            assert!(
                repair_recall + 0.08 >= rebuild_recall,
                "remove_every={remove_every} repair {repair_recall:.4} rebuild {rebuild_recall:.4}"
            );
        }
    }

    #[rstest]
    #[case::one_percent(100)]
    #[case::ten_percent(10)]
    #[case::half(2)]
    fn test_remap_drops_removed_nodes(#[case] remove_every: usize) {
        let (vectors, hnsw) = build(TOTAL);
        let batch = hnsw.to_batch().unwrap();
        let old_metadata = metadata(&batch);

        let mut new_ids = Vec::with_capacity(TOTAL);
        let mut kept = Vec::new();
        for old_id in 0..TOTAL {
            if old_id % remove_every == 0 {
                new_ids.push(None);
            } else {
                new_ids.push(Some(kept.len() as u32));
                kept.push(old_id as u32);
            }
        }
        let remapped = remap_graph_batch(&batch, &new_ids).unwrap();
        let new_metadata = metadata(&remapped);
        assert_eq!(
            new_metadata.level_offsets.len(),
            old_metadata.level_offsets.len()
        );
        assert_eq!(new_metadata.level_offsets[1], kept.len());

        // Every level keeps exactly its surviving members, renumbered in order.
        let old_ids = batch[VECTOR_ID_COL].as_primitive::<UInt32Type>();
        let new_ids_col = remapped[VECTOR_ID_COL].as_primitive::<UInt32Type>();
        let old_levels = old_metadata.level_offsets.iter().tuple_windows::<(_, _)>();
        let new_levels = new_metadata.level_offsets.iter().tuple_windows::<(_, _)>();
        for (level, ((old_start, old_end), (new_start, new_end))) in
            old_levels.zip(new_levels).enumerate()
        {
            let expected = (*old_start..*old_end)
                .filter_map(|row| new_ids[old_ids.value(row) as usize])
                .collect::<Vec<_>>();
            let actual = new_ids_col.values()[*new_start..*new_end].to_vec();
            assert_eq!(actual, expected, "level {level}");
        }

        // No edge names a removed or out-of-range node, and each surviving
        // edge is an old edge with its distance carried along.
        let old_neighbors = batch[NEIGHBORS_COL].as_list::<i32>();
        let old_dists = batch[DIST_COL].as_list::<i32>();
        let new_neighbors = remapped[NEIGHBORS_COL].as_list::<i32>();
        let new_dists = remapped[DIST_COL].as_list::<i32>();
        let mut old_row = 0;
        for new_row in 0..remapped.num_rows() {
            while new_ids[old_ids.value(old_row) as usize].is_none() {
                old_row += 1;
            }
            let expected = old_neighbors
                .value(old_row)
                .as_primitive::<UInt32Type>()
                .values()
                .iter()
                .zip(
                    old_dists
                        .value(old_row)
                        .as_primitive::<Float32Type>()
                        .values(),
                )
                .filter_map(|(n, d)| new_ids[*n as usize].map(|n| (n, *d)))
                .collect::<Vec<_>>();
            let actual = new_neighbors
                .value(new_row)
                .as_primitive::<UInt32Type>()
                .values()
                .iter()
                .copied()
                .zip(
                    new_dists
                        .value(new_row)
                        .as_primitive::<Float32Type>()
                        .values()
                        .iter()
                        .copied(),
                )
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "row {new_row}");
            assert!(actual.iter().all(|(n, _)| (*n as usize) < kept.len()));
            old_row += 1;
        }

        let loaded = HNSW::load(remapped.clone()).unwrap();
        assert_eq!(loaded.len(), kept.len());
        assert_eq!(loaded.to_batch().unwrap(), remapped);
        assert_eq!(remapped.num_rows(), num_level_members(&loaded));

        let kept_vectors = take(&vectors, &UInt32Array::from(kept), None).unwrap();
        let kept_vectors = kept_vectors.as_fixed_size_list().clone();
        let queries = kept_vectors.slice(0, 50);
        let recall = recall(&loaded, &kept_vectors, &queries);
        assert!(
            recall >= 0.5,
            "recall {recall} after removing 1/{remove_every}"
        );
    }

    #[test]
    fn test_remap_replaces_removed_entry_point() {
        let (_, hnsw) = build(TOTAL);
        let batch = hnsw.to_batch().unwrap();
        let old_metadata = metadata(&batch);
        let entry_point = old_metadata.entry_point as usize;

        let new_ids = (0..TOTAL)
            .map(|old_id| match old_id.cmp(&entry_point) {
                std::cmp::Ordering::Less => Some(old_id as u32),
                std::cmp::Ordering::Equal => None,
                std::cmp::Ordering::Greater => Some(old_id as u32 - 1),
            })
            .collect::<Vec<_>>();
        let remapped = remap_graph_batch(&batch, &new_ids).unwrap();
        let new_metadata = metadata(&remapped);
        let loaded = HNSW::load(remapped.clone()).unwrap();

        // The replacement sits on the highest level that still has nodes.
        let top = loaded.max_level() as usize - 1;
        let ids = remapped[VECTOR_ID_COL].as_primitive::<UInt32Type>();
        let top_ids =
            &ids.values()[new_metadata.level_offsets[top]..new_metadata.level_offsets[top + 1]];
        assert!(top_ids.contains(&new_metadata.entry_point));
    }

    #[test]
    fn test_remap_removing_every_node() {
        let (_, hnsw) = build(64);
        let batch = hnsw.to_batch().unwrap();
        let remapped = remap_graph_batch(&batch, &[None; 64]).unwrap();
        assert_eq!(remapped.num_rows(), 0);
        assert!(HNSW::load(remapped).unwrap().is_empty());
    }

    #[test]
    fn test_remap_rejects_mismatched_node_count() {
        let (_, hnsw) = build(64);
        let batch = hnsw.to_batch().unwrap();
        let err = remap_graph_batch(&batch, &[Some(0); 63]).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
        assert!(err.to_string().contains("64 nodes"), "{err}");

        // A search-projected batch lacks the distances the index file keeps.
        let projected = batch.project(&[0, 1]).unwrap();
        let new_ids = (0..64).map(Some).collect::<Vec<_>>();
        let err = remap_graph_batch(&projected, &new_ids).unwrap_err();
        assert!(err.to_string().contains(DIST_COL), "{err}");
    }

    /// Deleting from a graph of identical vectors drops the only inbound edge of
    /// many nodes. The repair links each of them back, and a wide search can
    /// return them.
    #[test]
    fn test_repair_keeps_survivors_reachable() {
        const N: usize = 500;
        let values = arrow_array::Float32Array::from(vec![0.0f32; N * DIM]);
        let vectors = FixedSizeListArray::try_new_from_values(values, DIM as i32).unwrap();
        let store = FlatFloatStorage::new(vectors.clone(), DistanceType::L2);
        let hnsw = HNSW::index_vectors(
            &store,
            HnswBuildParams::default().num_edges(4).ef_construction(4),
        )
        .unwrap();
        let batch = hnsw.to_batch().unwrap();

        let mut new_ids = Vec::with_capacity(N);
        let mut kept_idx = Vec::new();
        for old_id in 0..N {
            if old_id % 100 == 0 {
                new_ids.push(None);
            } else {
                new_ids.push(Some(kept_idx.len() as u32));
                kept_idx.push(old_id as u32);
            }
        }
        let kept = take(&vectors, &UInt32Array::from(kept_idx), None).unwrap();
        let kept = kept.as_fixed_size_list().clone();
        let kept_store = FlatFloatStorage::new(kept, DistanceType::L2);

        let repaired = remap_graph_repair(&batch, &new_ids, &kept_store).unwrap();
        let (reached, total) = reachable_from_entry(&repaired);
        assert_eq!(reached, total, "repair left nodes stranded");
        assert_eq!(total, N - N / 100);

        let params = HnswQueryParams {
            ef: 300,
            lower_bound: None,
            upper_bound: None,
            dist_q_c: 0.0,
            use_acorn: false,
        };
        let hits = HNSW::load(repaired)
            .unwrap()
            .search_basic(vectors.value(0), 300, &params, None, &kept_store)
            .unwrap()
            .len();
        assert_eq!(hits, 300);
    }

    fn reachable_from_entry(batch: &RecordBatch) -> (usize, usize) {
        let meta = metadata(batch);
        let ids = batch[VECTOR_ID_COL].as_primitive::<UInt32Type>();
        let neighbors = batch[NEIGHBORS_COL].as_list::<i32>();
        let level0 = meta.level_offsets[1];
        let n = ids.values()[..level0].iter().copied().max().unwrap_or(0) as usize + 1;
        let mut adj = vec![Vec::new(); n];
        let mut present = vec![false; n];
        for row in 0..level0 {
            let id = ids.value(row) as usize;
            present[id] = true;
            adj[id] = neighbors
                .value(row)
                .as_primitive::<UInt32Type>()
                .values()
                .to_vec();
        }
        let mut reachable = vec![false; n];
        let mut queue = std::collections::VecDeque::new();
        let entry = meta.entry_point as usize;
        if entry < n {
            reachable[entry] = true;
            queue.push_back(entry);
        }
        while let Some(current) = queue.pop_front() {
            for &neighbor in &adj[current] {
                let neighbor = neighbor as usize;
                if neighbor < n && !reachable[neighbor] {
                    reachable[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
        let total = present.iter().filter(|on| **on).count();
        let reached = present
            .iter()
            .zip(&reachable)
            .filter(|(on, reached)| **on && **reached)
            .count();
        (reached, total)
    }
}
