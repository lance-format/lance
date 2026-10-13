// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The adjacency of one partition's Vamana graph.
//!
//! Vertices are addressed by their position in the partition's storage, so
//! edges never name a row id: compaction and consolidation rewrite the row ids
//! and renumber positions, but never have to look an edge up by row.

use std::sync::Arc;

use arrow::buffer::{OffsetBuffer, ScalarBuffer};
use arrow_array::builder::{ListBuilder, UInt32Builder};
use arrow_array::cast::AsArray;
use arrow_array::types::UInt32Type;
use arrow_array::{Array, ListArray, UInt64Array};
use arrow_schema::{DataType, Field};
use lance_core::{Error, Result};

/// Pads a vertex's slots past its last out-edge.
pub const NO_NEIGHBOR: u32 = u32::MAX;

/// The widest neighbor list a graph may keep.
pub const MAX_DEGREE: u32 = 1024;

/// The most vertices a partition can address: local ids are `u32` and
/// [`NO_NEIGHBOR`] is reserved.
pub const MAX_VERTICES: usize = NO_NEIGHBOR as usize;

/// Out-edges by vertex, in whichever form the graph is held.
pub trait Adjacency {
    fn num_vertices(&self) -> usize;

    /// The out-edges of `vertex`, which must be below [`Self::num_vertices`].
    fn neighbors(&self, vertex: u32) -> &[u32];
}

/// A graph being built or rewritten: `max_degree` slots per vertex,
/// tail-padded with [`NO_NEIGHBOR`], so that a prune rewrites one vertex in
/// place without moving any other.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionGraph {
    max_degree: u32,
    row_ids: Vec<u64>,
    slots: Vec<u32>,
}

impl PartitionGraph {
    /// A graph whose vertices have no out-edges yet.
    pub fn edgeless(max_degree: u32, row_ids: Vec<u64>) -> Result<Self> {
        check_max_degree(max_degree)?;
        check_addressable(row_ids.len())?;
        let slots = vec![NO_NEIGHBOR; row_ids.len() * max_degree as usize];
        Ok(Self {
            max_degree,
            row_ids,
            slots,
        })
    }

    /// A graph from one out-edge list per vertex.
    pub fn try_new(max_degree: u32, row_ids: Vec<u64>, adjacency: &[Vec<u32>]) -> Result<Self> {
        if row_ids.len() != adjacency.len() {
            return Err(Error::invalid_input(format!(
                "Vamana graph has {} row ids but {} adjacency lists",
                row_ids.len(),
                adjacency.len()
            )));
        }
        let mut graph = Self::edgeless(max_degree, row_ids)?;
        for (vertex, neighbors) in adjacency.iter().enumerate() {
            graph.set_neighbors(vertex as u32, neighbors)?;
        }
        Ok(graph)
    }

    /// Append vertices with no out-edges, leaving every existing vertex where
    /// it is: the edges already in the graph name those positions.
    pub fn extend(&mut self, row_ids: &[u64]) -> Result<()> {
        let total = self.row_ids.len() + row_ids.len();
        check_addressable(total)?;
        self.row_ids.extend_from_slice(row_ids);
        self.slots
            .resize(total * self.max_degree as usize, NO_NEIGHBOR);
        Ok(())
    }

    pub fn max_degree(&self) -> u32 {
        self.max_degree
    }

    pub fn len(&self) -> usize {
        self.row_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.row_ids.is_empty()
    }

    pub fn row_ids(&self) -> &[u64] {
        &self.row_ids
    }

    /// Replace the out-edges of `vertex`.
    ///
    /// Refuses an edge that leaves the graph, points at its own vertex or
    /// repeats, so that no caller can write a list a walk would mis-read.
    pub fn set_neighbors(&mut self, vertex: u32, neighbors: &[u32]) -> Result<()> {
        let num_vertices = self.row_ids.len();
        if vertex as usize >= num_vertices {
            return Err(Error::invalid_input(format!(
                "Vamana vertex {vertex} is outside a graph of {num_vertices} vertices"
            )));
        }
        check_adjacency(vertex, neighbors, num_vertices, self.max_degree)?;
        let width = self.max_degree as usize;
        let start = vertex as usize * width;
        self.slots[start..start + neighbors.len()].copy_from_slice(neighbors);
        self.slots[start + neighbors.len()..start + width].fill(NO_NEIGHBOR);
        Ok(())
    }

    /// The out-edge lists as an Arrow list column, without the padding.
    pub fn to_list_array(&self) -> Result<ListArray> {
        let num_edges = (0..self.len() as u32)
            .map(|vertex| self.neighbors(vertex).len())
            .sum();
        check_list_edges(num_edges)?;
        let mut builder =
            ListBuilder::with_capacity(UInt32Builder::with_capacity(num_edges), self.row_ids.len())
                .with_field(neighbor_item_field());
        for vertex in 0..self.len() as u32 {
            builder.values().append_slice(self.neighbors(vertex));
            builder.append(true);
        }
        Ok(builder.finish())
    }
}

impl Adjacency for PartitionGraph {
    fn num_vertices(&self) -> usize {
        self.row_ids.len()
    }

    fn neighbors(&self, vertex: u32) -> &[u32] {
        let width = self.max_degree as usize;
        let slots = &self.slots[vertex as usize * width..][..width];
        let degree = slots
            .iter()
            .position(|neighbor| *neighbor == NO_NEIGHBOR)
            .unwrap_or(width);
        &slots[..degree]
    }
}

/// A graph read back from an index file: views over the Arrow buffers it was
/// read into, so loading copies nothing.
#[derive(Debug, Clone)]
pub struct LoadedGraph {
    row_ids: ScalarBuffer<u64>,
    offsets: OffsetBuffer<i32>,
    neighbors: ScalarBuffer<u32>,
}

impl LoadedGraph {
    pub fn empty() -> Self {
        Self {
            row_ids: ScalarBuffer::from(Vec::<u64>::new()),
            offsets: OffsetBuffer::new_empty(),
            neighbors: ScalarBuffer::from(Vec::<u32>::new()),
        }
    }

    /// Every edge is checked once here, so that a walk can index by them
    /// without a bounds check per hop.
    pub fn try_new(row_ids: &UInt64Array, neighbors: &ListArray, max_degree: u32) -> Result<Self> {
        let num_vertices = row_ids.len();
        if neighbors.len() != num_vertices {
            return Err(Error::index(format!(
                "Vamana graph has {num_vertices} row ids but {} neighbor lists",
                neighbors.len()
            )));
        }
        if row_ids.null_count() != 0 || neighbors.null_count() != 0 {
            return Err(Error::index(
                "Vamana graph holds a null row id or neighbor list".to_string(),
            ));
        }
        let values = neighbors
            .values()
            .as_primitive_opt::<UInt32Type>()
            .ok_or_else(|| {
                Error::index(format!(
                    "Vamana neighbor ids must be UInt32, got {}",
                    neighbors.values().data_type()
                ))
            })?;
        let offsets = neighbors.offsets().clone();
        let used = offsets[0] as usize..offsets[num_vertices] as usize;
        if values.slice(used.start, used.len()).null_count() != 0 {
            return Err(Error::index(
                "Vamana graph holds a null neighbor id".to_string(),
            ));
        }
        let values = values.values().clone();
        for (vertex, window) in offsets.windows(2).enumerate() {
            let edges = &values[window[0] as usize..window[1] as usize];
            if edges.len() > max_degree as usize {
                return Err(Error::index(format!(
                    "Vamana vertex {vertex} has {} out-edges, more than max_degree {max_degree}",
                    edges.len()
                )));
            }
            if let Some(edge) = edges.iter().find(|edge| **edge as usize >= num_vertices) {
                return Err(Error::index(format!(
                    "Vamana vertex {vertex} points at vertex {edge}, outside a graph of \
                     {num_vertices} vertices"
                )));
            }
        }
        Ok(Self {
            row_ids: row_ids.values().clone(),
            offsets,
            neighbors: values,
        })
    }

    pub fn row_ids(&self) -> &[u64] {
        &self.row_ids
    }

    /// A copy in the mutable form, for a rewrite to start from.
    pub fn to_partition_graph(&self, max_degree: u32) -> Result<PartitionGraph> {
        let mut graph = PartitionGraph::edgeless(max_degree, self.row_ids.to_vec())?;
        for vertex in 0..self.num_vertices() as u32 {
            graph.set_neighbors(vertex, self.neighbors(vertex))?;
        }
        Ok(graph)
    }
}

impl Adjacency for LoadedGraph {
    fn num_vertices(&self) -> usize {
        self.row_ids.len()
    }

    fn neighbors(&self, vertex: u32) -> &[u32] {
        let start = self.offsets[vertex as usize] as usize;
        let end = self.offsets[vertex as usize + 1] as usize;
        &self.neighbors[start..end]
    }
}

/// Whether a walk from `start` can reach each vertex along out-edges.
pub fn reached(graph: &impl Adjacency, start: u32) -> Vec<bool> {
    let mut seen = vec![false; graph.num_vertices()];
    seen[start as usize] = true;
    let mut frontier = vec![start];
    while let Some(vertex) = frontier.pop() {
        for neighbor in graph.neighbors(vertex) {
            if !seen[*neighbor as usize] {
                seen[*neighbor as usize] = true;
                frontier.push(*neighbor);
            }
        }
    }
    seen
}

/// How many of the vertices `counted` accepts a walk from `start` can reach
/// along out-edges, passing through any vertex on the way.
pub fn reachable(graph: &impl Adjacency, start: u32, counted: impl Fn(u32) -> bool) -> usize {
    reached(graph, start)
        .iter()
        .enumerate()
        .filter(|(vertex, is_reached)| **is_reached && counted(*vertex as u32))
        .count()
}

pub(crate) fn neighbor_item_field() -> Arc<Field> {
    Arc::new(Field::new("item", DataType::UInt32, false))
}

pub(crate) fn check_max_degree(max_degree: u32) -> Result<()> {
    if max_degree == 0 || max_degree > MAX_DEGREE {
        return Err(Error::invalid_input(format!(
            "Vamana max_degree must be between 1 and {MAX_DEGREE}, got {max_degree}"
        )));
    }
    Ok(())
}

/// A list column's offsets are `i32`, so it holds at most `i32::MAX` edges:
/// about 33.6 million vertices at `R = 64`.
fn check_list_edges(num_edges: usize) -> Result<()> {
    if num_edges > i32::MAX as usize {
        return Err(Error::invalid_input(format!(
            "Vamana graph has {num_edges} edges, more than the {} a list column with 32-bit \
             offsets holds",
            i32::MAX
        )));
    }
    Ok(())
}

fn check_addressable(num_vertices: usize) -> Result<()> {
    if num_vertices > MAX_VERTICES {
        return Err(Error::invalid_input(format!(
            "Vamana partition holds {num_vertices} vectors, more than the addressable \
             {MAX_VERTICES}"
        )));
    }
    Ok(())
}

fn check_adjacency(
    vertex: u32,
    neighbors: &[u32],
    num_vertices: usize,
    max_degree: u32,
) -> Result<()> {
    if neighbors.len() > max_degree as usize {
        return Err(Error::invalid_input(format!(
            "Vamana vertex {vertex} has degree {} which exceeds max_degree {max_degree}",
            neighbors.len()
        )));
    }
    for (position, neighbor) in neighbors.iter().enumerate() {
        if *neighbor as usize >= num_vertices {
            return Err(Error::invalid_input(format!(
                "Vamana vertex {vertex} points at vertex {neighbor}, outside a graph of \
                 {num_vertices} vertices"
            )));
        }
        if *neighbor == vertex {
            return Err(Error::invalid_input(format!(
                "Vamana vertex {vertex} points at itself"
            )));
        }
        // Quadratic rather than sorted: a list is tens of slots, and it is
        // written beside a prune that spends a distance per slot pair.
        if neighbors[..position].contains(neighbor) {
            return Err(Error::invalid_input(format!(
                "Vamana vertex {vertex} has a duplicate out-edge {neighbor}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_graph(max_degree: u32) -> PartitionGraph {
        PartitionGraph::try_new(
            max_degree,
            vec![10, 11, 12, 13],
            &[vec![1, 2], vec![0], vec![], vec![0, 1, 2]],
        )
        .unwrap()
    }

    #[test]
    fn neighbors_are_trimmed_at_the_padding() {
        let graph = sample_graph(3);
        assert_eq!(graph.neighbors(0), &[1, 2]);
        assert_eq!(graph.neighbors(2), &[] as &[u32]);
        assert_eq!(graph.neighbors(3), &[0, 1, 2]);
    }

    #[test]
    fn set_neighbors_rewrites_one_vertex_and_repads_it() {
        let mut graph = sample_graph(3);
        graph.set_neighbors(3, &[2]).unwrap();
        assert_eq!(graph.neighbors(3), &[2]);
        assert_eq!(graph.neighbors(0), &[1, 2]);
    }

    #[test]
    fn set_neighbors_refuses_what_it_cannot_store() {
        let mut graph = sample_graph(3);
        assert!(graph.set_neighbors(0, &[1, 2, 3, 0]).is_err());
        assert!(graph.set_neighbors(0, &[4]).is_err());
        assert!(graph.set_neighbors(0, &[0]).is_err());
        assert!(graph.set_neighbors(0, &[1, 1]).is_err());
        assert!(graph.set_neighbors(4, &[1]).is_err());
        assert!(PartitionGraph::edgeless(0, vec![1]).is_err());
        assert!(PartitionGraph::edgeless(MAX_DEGREE + 1, vec![1]).is_err());
    }

    #[test]
    fn extend_keeps_existing_vertices_in_place() {
        let mut graph = sample_graph(3);
        graph.extend(&[14, 15]).unwrap();
        assert_eq!(graph.len(), 6);
        assert_eq!(graph.row_ids(), &[10, 11, 12, 13, 14, 15]);
        assert_eq!(graph.neighbors(3), &[0, 1, 2]);
        assert_eq!(graph.neighbors(5), &[] as &[u32]);
    }

    #[test]
    fn a_list_column_round_trips_through_the_loaded_form() {
        let graph = sample_graph(3);
        let lists = graph.to_list_array().unwrap();
        let row_ids = UInt64Array::from(graph.row_ids().to_vec());
        let loaded = LoadedGraph::try_new(&row_ids, &lists, 3).unwrap();
        for vertex in 0..4 {
            assert_eq!(loaded.neighbors(vertex), graph.neighbors(vertex));
        }
        assert_eq!(loaded.row_ids(), graph.row_ids());
        assert_eq!(loaded.to_partition_graph(3).unwrap(), graph);
    }

    #[test]
    fn a_sliced_list_column_loads_at_its_own_offsets() {
        let graph = sample_graph(3);
        let lists = graph.to_list_array().unwrap().slice(1, 3);
        let row_ids = UInt64Array::from(vec![11, 12, 13]);
        // Vertex 0 of the slice is vertex 1 of the graph, whose only edge
        // points at old vertex 0, which is still in range for three vertices.
        let loaded = LoadedGraph::try_new(&row_ids, &lists, 3).unwrap();
        assert_eq!(loaded.neighbors(0), &[0]);
        assert_eq!(loaded.neighbors(1), &[] as &[u32]);
        assert_eq!(loaded.neighbors(2), &[0, 1, 2]);
    }

    #[test]
    fn a_loaded_graph_refuses_edges_it_cannot_follow() {
        let row_ids = UInt64Array::from(vec![1, 2]);
        let mut builder = ListBuilder::new(UInt32Builder::new()).with_field(neighbor_item_field());
        builder.values().append_slice(&[1]);
        builder.append(true);
        builder.values().append_slice(&[2]);
        builder.append(true);
        let out_of_range = builder.finish();
        assert!(LoadedGraph::try_new(&row_ids, &out_of_range, 4).is_err());

        let mut builder = ListBuilder::new(UInt32Builder::new()).with_field(neighbor_item_field());
        builder.values().append_slice(&[1]);
        builder.append(true);
        builder.values().append_slice(&[0]);
        builder.append(true);
        let fine = builder.finish();
        assert!(LoadedGraph::try_new(&row_ids, &fine, 1).is_ok());
        assert!(LoadedGraph::try_new(&UInt64Array::from(vec![1]), &fine, 1).is_err());

        let mut builder = ListBuilder::new(UInt32Builder::new()).with_field(neighbor_item_field());
        builder.values().append_slice(&[1]);
        builder.append(true);
        builder.values().append_slice(&[0, 1]);
        builder.append(true);
        let too_wide = builder.finish();
        assert!(LoadedGraph::try_new(&row_ids, &too_wide, 1).is_err());
    }

    #[test]
    fn reachable_counts_only_what_it_is_asked_to() {
        let graph = PartitionGraph::try_new(
            4,
            vec![0, 1, 2, 3, 4],
            &[vec![1], vec![2], vec![], vec![4], vec![]],
        )
        .unwrap();
        assert_eq!(reachable(&graph, 0, |_| true), 3);
        assert_eq!(reachable(&graph, 0, |vertex| vertex != 1), 2);
        assert_eq!(reachable(&graph, 3, |_| true), 2);
        assert_eq!(reached(&graph, 3), vec![false, false, false, true, true]);
    }

    /// The check rather than a graph past it: one would take 8 GiB of slots.
    #[test]
    fn a_list_column_refuses_more_edges_than_its_offsets_hold() {
        assert!(check_list_edges(i32::MAX as usize).is_ok());
        let error = check_list_edges(i32::MAX as usize + 1).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("2147483648 edges"), "{error}");
    }
}
