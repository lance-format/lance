// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::cmp::{Ordering as CmpOrdering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::ops::Range;
use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use arc_swap::ArcSwap;

use arrow_array::builder::{Float32Builder, ListBuilder, UInt32Builder};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use crossbeam_queue::ArrayQueue;
use lance_core::{Error, Result};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::storage::VectorSource;

const HNSW_METADATA_KEY: &str = "lance:hnsw";
const VECTOR_ID_COL: &str = "__vector_id";
const NEIGHBORS_COL: &str = "__neighbors";
const DIST_COL: &str = "_distance";
const DEFAULT_SEED: u64 = 100;
const MIN_HNSW_M: usize = 4;
const WORD_BITS: usize = usize::BITS as usize;

/// Parameters for HNSW graph construction.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BuildParams {
    /// Maximum number of graph levels.
    pub max_level: u16,
    /// Number of neighbors selected for a new node on every level.
    /// Existing level-0 nodes may retain up to `2 * m` reciprocal connections.
    pub m: usize,
    /// Beam width used while inserting nodes.
    pub ef_construction: usize,
    /// Lance-compatible prefetch hint retained in exported metadata.
    pub prefetch_distance: Option<usize>,
    /// Random seed used for deterministic level assignment.
    #[serde(skip, default = "default_seed")]
    pub seed: u64,
}

impl Default for BuildParams {
    fn default() -> Self {
        Self {
            max_level: 7,
            m: 20,
            ef_construction: 150,
            prefetch_distance: Some(2),
            seed: DEFAULT_SEED,
        }
    }
}

impl BuildParams {
    /// Defaults intended for active MemTable indexing.
    pub fn mem_wal_default() -> Self {
        Self {
            m: 12,
            ef_construction: 64,
            ..Self::default()
        }
    }

    /// Set the maximum number of graph levels.
    pub fn max_level(mut self, max_level: u16) -> Self {
        self.max_level = max_level;
        self
    }

    /// Set the HNSW `M` parameter.
    pub fn num_edges(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    /// Set the construction beam width.
    pub fn ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = ef_construction;
        self
    }

    /// Set the deterministic level-assignment seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    fn validate(&self) -> Result<()> {
        if self.max_level == 0 {
            return Err(Error::invalid_input("max_level must be greater than 0"));
        }
        if self.max_level as u32 > u64::BITS {
            return Err(Error::invalid_input(format!(
                "max_level must be <= {}, got {}",
                u64::BITS,
                self.max_level
            )));
        }
        if self.m < MIN_HNSW_M {
            return Err(Error::invalid_input(format!(
                "BuildParams::m must be at least {MIN_HNSW_M} to avoid severely fragmented graphs, got {}",
                self.m
            )));
        }
        if self.m > usize::MAX / 2 {
            return Err(Error::invalid_input(format!(
                "BuildParams::m must be at most {} so the level-0 reciprocal limit can be represented, got {}",
                usize::MAX / 2,
                self.m
            )));
        }
        let neighbor_capacity = max_reciprocal_neighbors(self.m, 0);
        let max_vec_capacity = (isize::MAX as usize) / std::mem::size_of::<ScoredPoint>();
        if neighbor_capacity > max_vec_capacity {
            return Err(Error::invalid_input(format!(
                "BuildParams::m={} requires neighbor_capacity={neighbor_capacity} at level 0, but Vec<ScoredPoint> can hold at most {max_vec_capacity} values",
                self.m
            )));
        }
        if self.ef_construction < self.m {
            return Err(Error::invalid_input(format!(
                "BuildParams::ef_construction must be at least m ({}), got {}",
                self.m, self.ef_construction
            )));
        }
        Ok(())
    }
}

fn default_seed() -> u64 {
    DEFAULT_SEED
}

/// Query parameters for graph search.
#[derive(Debug, Clone, Copy)]
pub struct SearchParams {
    /// Number of nearest neighbors to return.
    pub k: usize,
    /// Beam width used by the bottom-level search.
    pub ef: usize,
}

impl SearchParams {
    /// Create search params, using `max(k, ef)` internally.
    pub fn new(k: usize, ef: usize) -> Self {
        Self { k, ef }
    }
}

/// Candidate with an internal vector id and distance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredPoint {
    pub id: u32,
    pub distance: f32,
}

impl ScoredPoint {
    fn new(id: u32, distance: f32) -> Self {
        Self { id, distance }
    }
}

impl Eq for ScoredPoint {}

impl PartialOrd for ScoredPoint {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredPoint {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(CmpOrdering::Equal)
    }
}

/// Search result with Lance row id attached.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchResult {
    pub id: u32,
    pub row_id: u64,
    pub distance: f32,
}

#[derive(Debug, Clone, Copy)]
struct BeamLimits {
    ef: usize,
    output_limit: usize,
}

#[derive(Debug, Clone, Copy)]
struct BuildBeamLimits {
    ef: usize,
    visible_len: usize,
    visited_capacity: usize,
}

/// Metadata stored under `lance:hnsw` in Lance HNSW batches.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LanceHnswMetadata {
    pub entry_point: u32,
    pub params: BuildParams,
    pub level_offsets: Vec<usize>,
}

struct PackedLevel {
    offsets: Vec<usize>,
    neighbors: Vec<u32>,
}

impl PackedLevel {
    fn empty() -> Self {
        Self {
            offsets: vec![0],
            neighbors: Vec::new(),
        }
    }

    fn neighbors(&self, id: u32) -> Option<&[u32]> {
        let idx = id as usize;
        let start = *self.offsets.get(idx)?;
        let end = *self.offsets.get(idx + 1)?;
        Some(&self.neighbors[start..end])
    }
}

struct LevelLinks {
    /// Published neighbor ids for this level, read lock-free by searchers.
    /// `ArcSwap` reclaims the previous snapshot once no reader guard references
    /// it, so the repeated publishes of an incremental build do not accumulate
    /// (the prior manual `AtomicPtr` + retired-`Vec` scheme leaked every old
    /// snapshot until graph drop, growing O(N^2/batch) and OOMing at ~1M rows).
    ///
    /// `None` on level 0, which searchers read from the packed level-0
    /// snapshot; one `ArcSwap` per node makes a large graph slow to drop.
    published: Option<ArcSwap<Vec<u32>>>,
    /// Empty until the node is inserted or first linked to, then sized to
    /// exactly `max_neighbors` once and reused (see [`size_ranked`]).
    ranked: Mutex<Vec<ScoredPoint>>,
}

impl LevelLinks {
    /// `empty` is one empty list the whole graph shares until a level first
    /// publishes, so an unfilled slot allocates nothing per level.
    fn new(level: u16, empty: &Arc<Vec<u32>>) -> Self {
        Self {
            published: (level > 0).then(|| ArcSwap::new(Arc::clone(empty))),
            ranked: Mutex::new(Vec::new()),
        }
    }

    /// Heap bytes a level adds once its node is in the graph: `ranked` at
    /// exactly `max_neighbors`, and above level 0 a published list counted at
    /// full width (a publish may hold fewer) plus its `Arc` block (refcounts
    /// and the Vec header), rounded to the 16-byte granule allocators hand out
    /// for a block that small.
    fn linked_bytes(level: u16, max_neighbors: usize) -> usize {
        let ranked = max_neighbors * std::mem::size_of::<ScoredPoint>();
        if level == 0 {
            return ranked;
        }
        let published_block = (2 * std::mem::size_of::<usize>() + std::mem::size_of::<Vec<u32>>())
            .next_multiple_of(16);
        published_block + ranked + max_neighbors * std::mem::size_of::<u32>()
    }

    fn publish_from_ranked(&self, ranked: &[ScoredPoint]) {
        if let Some(published) = &self.published {
            published.store(Arc::new(
                ranked.iter().map(|point| point.id).collect::<Vec<_>>(),
            ));
        }
    }
}

struct Node {
    target_level: u16,
    levels: Vec<LevelLinks>,
    dirty_levels: AtomicU64,
}

impl Node {
    fn new(target_level: u16, empty: &Arc<Vec<u32>>) -> Self {
        let levels = (0..=target_level)
            .map(|level| LevelLinks::new(level, empty))
            .collect();
        Self {
            target_level,
            levels,
            dirty_levels: AtomicU64::new(0),
        }
    }

    /// Heap bytes a node of `target_level` holds from construction, excluding
    /// the `Node` itself (which lives inline in the graph's node arena).
    fn fixed_bytes(target_level: u16) -> usize {
        (target_level as usize + 1) * std::mem::size_of::<LevelLinks>()
    }

    /// Heap bytes a node of `target_level` adds once it is in the graph.
    fn linked_bytes(target_level: u16, m: usize) -> usize {
        (0..=target_level)
            .map(|level| LevelLinks::linked_bytes(level, max_reciprocal_neighbors(m, level)))
            .sum()
    }

    fn has_level(&self, level: u16) -> bool {
        (level as usize) < self.levels.len()
    }

    fn ranked(&self, level: u16) -> Result<MutexGuard<'_, Vec<ScoredPoint>>> {
        self.levels[level as usize]
            .ranked
            .lock()
            .map_err(|_| Error::internal("HNSW neighbor mutex poisoned"))
    }

    fn mark_dirty(&self, level: u16) {
        self.dirty_levels
            .fetch_or(1_u64 << level, Ordering::Release);
    }
}

/// Multi-reader / single-writer HNSW graph.
///
/// Public readers search only the published visible prefix. A writer may build
/// a new contiguous id range with worker threads and then publish that range
/// atomically when the batch completes.
pub struct HnswGraph {
    params: BuildParams,
    nodes: Vec<Node>,
    build_entry_point: AtomicU32,
    build_max_level: AtomicU16,
    visible_entry_point: AtomicU32,
    visible_max_level: AtomicU16,
    indexed_len: AtomicUsize,
    visible_len: AtomicUsize,
    visited_pool: ArrayQueue<VisitedList>,
    packed_level0: ArcSwap<PackedLevel>,
    /// Heap bytes of the node arena, each node's per-level headers and the
    /// visited pool. Fixed at construction: all are sized from `capacity`.
    base_bytes: usize,
    /// Heap bytes of the neighbor lists of every node inserted so far. A node's
    /// lists are allocated as it is inserted, so slots never filled cost only
    /// their share of `base_bytes`.
    linked_bytes: AtomicUsize,
    /// Heap bytes of the current `packed_level0` snapshot, which is rebuilt
    /// wholesale on each level-0 publish rather than grown.
    packed_bytes: AtomicUsize,
}

impl HnswGraph {
    /// Pre-allocate graph nodes and their random levels.
    pub fn try_new(capacity: usize, params: BuildParams) -> Result<Self> {
        params.validate()?;
        if capacity == 0 {
            return Err(Error::invalid_input("capacity must be greater than 0"));
        }
        if capacity > u32::MAX as usize {
            return Err(Error::invalid_input(format!(
                "capacity must fit in u32, got {capacity}"
            )));
        }

        let mut rng = SmallRng::seed_from_u64(params.seed);
        let empty = Arc::new(Vec::new());
        let mut nodes = Vec::with_capacity(capacity);
        let mut node_bytes = 0;
        for _ in 0..capacity {
            let target_level = random_level(&params, &mut rng);
            node_bytes += Node::fixed_bytes(target_level);
            nodes.push(Node::new(target_level, &empty));
        }

        let pool_size = visited_pool_size();
        let visited_pool = ArrayQueue::new(pool_size);
        for _ in 0..pool_size {
            let _ = visited_pool.push(VisitedList::new(0));
        }

        Ok(Self {
            params,
            base_bytes: capacity * std::mem::size_of::<Node>()
                + node_bytes
                + visited_pool_bytes(capacity),
            nodes,
            build_entry_point: AtomicU32::new(0),
            build_max_level: AtomicU16::new(0),
            visible_entry_point: AtomicU32::new(0),
            visible_max_level: AtomicU16::new(0),
            indexed_len: AtomicUsize::new(0),
            visible_len: AtomicUsize::new(0),
            visited_pool,
            packed_level0: ArcSwap::from_pointee(PackedLevel::empty()),
            packed_bytes: AtomicUsize::new(0),
            linked_bytes: AtomicUsize::new(0),
        })
    }

    /// Upper bound on the graph's dominant heap allocations.
    ///
    /// A fixed part sized from `capacity` and allocated in full at
    /// construction, so a vector memtable owes it from its first row, plus the
    /// neighbor lists of the nodes inserted so far.
    pub(crate) fn resident_bytes(&self) -> usize {
        self.base_bytes
            + self.linked_bytes.load(Ordering::Relaxed)
            + self.packed_bytes.load(Ordering::Relaxed)
    }

    /// What [`Self::resident_bytes`] will report for a graph of this shape
    /// before its first insert, answerable before one is built.
    ///
    /// Everything `try_new` allocates is sized from `capacity`; the only random
    /// input is how nodes divide across levels, and that division is a
    /// geometric ladder — every node holds level 0, and the share reaching each
    /// level above it falls by a factor of `m`. Walking that ladder lands close
    /// to the built graph instead of sampling it, and level 0 — which dominates
    /// — is not an estimate at all.
    ///
    /// Exists because the allocation is committed well before it happens: the
    /// first vector row into a memtable materializes the whole fixed part.
    /// Charging it only from that row on would put the largest single
    /// allocation in a vector memtable beyond the reach of admission control.
    /// Neighbor lists are not part of it: they are allocated per inserted node,
    /// and charged as each node is.
    pub(crate) fn reserved_bytes(capacity: usize, params: &BuildParams) -> usize {
        // Guard the ladder's divisor rather than `params.m` itself: `validate`
        // rejects m < 2, but this is reachable before that runs.
        let ratio = params.m.max(2);
        let mut reaching = capacity;
        let mut links = 0;
        for _ in 0..params.max_level {
            links += reaching * std::mem::size_of::<LevelLinks>();
            reaching /= ratio;
            if reaching == 0 {
                break;
            }
        }
        capacity * std::mem::size_of::<Node>() + links + visited_pool_bytes(capacity)
    }

    /// Number of nodes visible to readers.
    pub fn len(&self) -> usize {
        self.visible_len.load(Ordering::Acquire)
    }

    /// Returns true if no nodes are visible to readers.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of nodes already inserted into the graph, including unpublished
    /// in-flight writer batches after insertion completes.
    pub fn indexed_len(&self) -> usize {
        self.indexed_len.load(Ordering::Acquire)
    }

    /// Graph build parameters.
    pub fn params(&self) -> &BuildParams {
        &self.params
    }

    /// Insert and publish a single vector id.
    pub fn insert(&self, id: u32, vectors: &impl VectorSource) -> Result<()> {
        let expected = self.indexed_len.load(Ordering::Acquire);
        if id as usize != expected {
            return Err(Error::invalid_input(format!(
                "insert id must match indexed_len: id={}, indexed_len={expected}",
                id
            )));
        }
        self.validate_source(vectors, id as usize + 1)?;
        self.insert_inner(id, vectors)?;
        self.indexed_len.store(id as usize + 1, Ordering::Release);
        self.publish_dirty(id as usize + 1)?;
        self.publish_visible(id as usize + 1);
        Ok(())
    }

    /// Insert a contiguous id range using Rayon worker threads.
    ///
    /// Readers continue to search the previous visible prefix while this runs.
    /// The full range becomes visible only after every insert in the batch has
    /// completed.
    pub fn insert_batch(&self, ids: Range<u32>, vectors: &impl VectorSource) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let expected = self.indexed_len.load(Ordering::Acquire);
        if ids.start as usize != expected {
            return Err(Error::invalid_input(format!(
                "insert range must start at indexed_len: range_start={}, indexed_len={expected}",
                ids.start
            )));
        }
        self.validate_source(vectors, ids.end as usize)?;

        let parallel_start = if ids.start == 0 {
            self.insert_inner(0, vectors)?;
            1
        } else {
            ids.start
        };

        (parallel_start..ids.end)
            .into_par_iter()
            .try_for_each_init(
                || VisitedList::new(0),
                |visited, id| self.insert_inner_with_visited(id, vectors, visited),
            )?;

        self.indexed_len.store(ids.end as usize, Ordering::Release);
        self.publish_dirty(ids.end as usize)?;
        self.publish_visible(ids.end as usize);
        Ok(())
    }

    /// Insert a contiguous id range serially.
    ///
    /// This is useful when comparing against single-threaded baselines or
    /// debugging graph quality without scheduler effects.
    pub fn insert_batch_serial(&self, ids: Range<u32>, vectors: &impl VectorSource) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let expected = self.indexed_len.load(Ordering::Acquire);
        if ids.start as usize != expected {
            return Err(Error::invalid_input(format!(
                "insert range must start at indexed_len: range_start={}, indexed_len={expected}",
                ids.start
            )));
        }
        self.validate_source(vectors, ids.end as usize)?;

        let mut visited = VisitedList::new(0);
        for id in ids.clone() {
            self.insert_inner_with_visited(id, vectors, &mut visited)?;
        }

        self.indexed_len.store(ids.end as usize, Ordering::Release);
        self.publish_dirty(ids.end as usize)?;
        self.publish_visible(ids.end as usize);
        Ok(())
    }

    /// Insert a contiguous id range using the current thread.
    pub fn insert_batch_with_threads(
        &self,
        ids: Range<u32>,
        vectors: &impl VectorSource,
        threads: usize,
    ) -> Result<()> {
        if threads <= 1 {
            return self.insert_batch_serial(ids, vectors);
        }
        self.insert_batch(ids, vectors)
    }

    /// Search the published visible graph.
    pub fn search(
        &self,
        query: &[f32],
        params: SearchParams,
        vectors: &impl VectorSource,
    ) -> Result<Vec<SearchResult>> {
        if params.k == 0 {
            return Ok(Vec::new());
        }
        if query.len() != vectors.dim() {
            return Err(Error::invalid_input(format!(
                "query dimension mismatch: expected {}, got {}",
                vectors.dim(),
                query.len()
            )));
        }

        // The vector snapshot and graph publication are captured separately.
        // Cap search to the common visible prefix so a reader racing with the
        // writer never follows a newly-published graph edge into an older
        // vector snapshot.
        let visible_len = self.visible_len.load(Ordering::Acquire).min(vectors.len());
        if visible_len == 0 {
            return Ok(Vec::new());
        }
        let neighbor_visible_len = visible_len;
        let visible_max_level = self.visible_max_level.load(Ordering::Acquire);
        let entry = self.visible_entry_point.load(Ordering::Acquire);
        if entry as usize >= visible_len {
            return Ok(Vec::new());
        }

        let mut visited = self
            .visited_pool
            .pop()
            .unwrap_or_else(|| VisitedList::new(0));
        let mut ep = ScoredPoint::new(entry, vectors.distance_to(query, entry));
        for level in (1..visible_max_level).rev() {
            ep = self.greedy_search_query(ep, level, neighbor_visible_len, |id| {
                vectors.distance_to(query, id)
            });
        }

        // A beam wider than the graph finds nothing more; cap what it allocates.
        let ef = params.ef.max(params.k).min(visible_len);
        let limits = BeamLimits {
            ef,
            output_limit: params.k,
        };
        let candidates =
            self.beam_search_query(ep, 0, limits, neighbor_visible_len, &mut visited, |id| {
                vectors.distance_to(query, id)
            });
        let _ = self.visited_pool.push(visited);

        Ok(candidates
            .into_iter()
            .take(params.k)
            .map(|point| SearchResult {
                id: point.id,
                row_id: vectors.row_id(point.id),
                distance: point.distance,
            })
            .collect())
    }

    /// Emit the Lance HNSW sub-index record batch.
    ///
    /// The resulting batch uses the same schema and `lance:hnsw` metadata
    /// expected by `lance-index`'s `HNSW::load`.
    ///
    /// `max_nodes` caps the prefix, for a caller that has already captured a
    /// companion artifact and needs this one to agree with it: vector storage
    /// is materialized separately, and a graph that advanced past it would name
    /// rows the storage batch has no vector for.
    ///
    /// Ordinary search readers can run concurrently with insertion; a flush
    /// export snapshots a completed prefix.
    pub fn to_lance_hnsw_batch(&self, max_nodes: Option<usize>) -> Result<RecordBatch> {
        let visible_len = self.visible_len.load(Ordering::Acquire);
        let visible_len = match max_nodes {
            Some(max_nodes) => visible_len.min(max_nodes),
            None => visible_len,
        };
        let max_level = self.params.max_level as usize;
        let mut level_counts = vec![0usize; max_level];
        for id in 0..visible_len {
            let node = &self.nodes[id];
            for count in level_counts
                .iter_mut()
                .take(node.levels.len().min(max_level))
            {
                *count += 1;
            }
        }

        let total_rows: usize = level_counts.iter().sum();
        let mut vector_id_builder = UInt32Builder::with_capacity(total_rows);
        let mut neighbors_builder = ListBuilder::with_capacity(UInt32Builder::new(), total_rows);
        let mut distances_builder = ListBuilder::with_capacity(Float32Builder::new(), total_rows);

        for level in 0..max_level {
            for id in 0..visible_len {
                let node = &self.nodes[id];
                if level >= node.levels.len() {
                    continue;
                }
                let ranked = node.ranked(level as u16)?;
                vector_id_builder.append_value(id as u32);
                // `visible_len` is snapshotted but adjacency is read live, so a
                // batch landing mid-export can append itself to a node already
                // emitted. Those ids are not rows in this batch: `HNSW::load`
                // slices level 0 to the rows below, and `neighbors_at` would
                // address past them. `search` caps in-memory traversal the same
                // way; the export has to persist the bound. Both columns filter
                // together so a reader pairing them sees equal lengths.
                neighbors_builder.append_value(
                    ranked
                        .iter()
                        .filter(|point| (point.id as usize) < visible_len)
                        .map(|point| Some(point.id)),
                );
                distances_builder.append_value(
                    ranked
                        .iter()
                        .filter(|point| (point.id as usize) < visible_len)
                        .map(|point| Some(point.distance)),
                );
            }
        }

        // `publish_visible` stores the entry point before `visible_len`, so it
        // can name a node outside this prefix. `search` returns no results in
        // that case; an exported index cannot, so fall back to the deepest node
        // the prefix does hold.
        let entry_point = {
            let published = self.visible_entry_point.load(Ordering::Acquire);
            if (published as usize) < visible_len {
                published
            } else {
                (0..visible_len)
                    .max_by_key(|&id| self.nodes[id].levels.len())
                    .map(|id| id as u32)
                    .unwrap_or(0)
            }
        };
        let metadata = LanceHnswMetadata {
            entry_point,
            params: self.params.clone(),
            level_offsets: level_counts
                .iter()
                .chain(std::iter::once(&0))
                .scan(0, |state, count| {
                    let start = *state;
                    *state += *count;
                    Some(start)
                })
                .collect(),
        };
        let metadata = serde_json::to_string(&metadata)?;
        let schema = lance_hnsw_schema()
            .as_ref()
            .clone()
            .with_metadata(HashMap::from_iter([(
                HNSW_METADATA_KEY.to_string(),
                metadata,
            )]));
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(vector_id_builder.finish()) as ArrayRef,
                Arc::new(neighbors_builder.finish()) as ArrayRef,
                Arc::new(distances_builder.finish()) as ArrayRef,
            ],
        )
        .map_err(Error::from)
    }

    fn validate_source(&self, vectors: &impl VectorSource, needed_len: usize) -> Result<()> {
        // Not caller input: the graph was sized below what the memtable holds.
        // See the matching note in `storage.rs::append_batch`.
        if needed_len > self.nodes.len() {
            return Err(Error::internal(format!(
                "HNSW graph capacity {} exhausted: need {needed_len}; \
                 the graph is sized below the memtable's row capacity",
                self.nodes.len()
            )));
        }
        if vectors.len() < needed_len {
            return Err(Error::invalid_input(format!(
                "vector source has {} rows but graph insert needs {needed_len}",
                vectors.len()
            )));
        }
        Ok(())
    }

    fn insert_inner(&self, id: u32, vectors: &impl VectorSource) -> Result<()> {
        let mut visited = self
            .visited_pool
            .pop()
            .unwrap_or_else(|| VisitedList::new(0));
        let result = self.insert_inner_with_visited(id, vectors, &mut visited);
        let _ = self.visited_pool.push(visited);
        result
    }

    fn insert_inner_with_visited(
        &self,
        id: u32,
        vectors: &impl VectorSource,
        visited: &mut VisitedList,
    ) -> Result<()> {
        // Charged before any of this node's lists is allocated.
        self.linked_bytes.fetch_add(
            Node::linked_bytes(self.nodes[id as usize].target_level, self.params.m),
            Ordering::Relaxed,
        );
        if id == 0 {
            let target_level = self.nodes[0].target_level;
            self.build_entry_point.store(0, Ordering::Release);
            self.build_max_level
                .store(target_level + 1, Ordering::Release);
            return Ok(());
        }

        let target_level = self.nodes[id as usize].target_level;
        let current_max_level = self.build_max_level.load(Ordering::Acquire).max(1);
        let entry = self.build_entry_point.load(Ordering::Acquire);
        let mut ep = ScoredPoint::new(entry, vectors.distance_between(id, entry));

        for level in (target_level + 1..current_max_level).rev() {
            ep = self.greedy_search_build(ep, level, usize::MAX, |candidate| {
                vectors.distance_between(id, candidate)
            })?;
        }

        let connect_max_level = target_level.min(current_max_level - 1);
        let mut selected_by_level: Vec<Vec<ScoredPoint>> =
            vec![Vec::new(); (target_level + 1) as usize];
        for level in (0..=connect_max_level).rev() {
            let candidates = self.beam_search_build(
                ep,
                level,
                BuildBeamLimits {
                    ef: self.params.ef_construction,
                    visible_len: usize::MAX,
                    visited_capacity: vectors.len(),
                },
                visited,
                |candidate| vectors.distance_between(id, candidate),
            )?;
            let candidates: Vec<_> = candidates
                .into_iter()
                .filter(|point| point.id != id && self.nodes[point.id as usize].has_level(level))
                .collect();

            // Algorithm 1 selects M neighbors for the new node. Mmax0 = 2M is
            // reserved for reciprocal connections on existing level-0 nodes.
            let selected = self.select_neighbors(vectors, &candidates, self.params.m);
            self.set_node_neighbors(id, level, &selected)?;
            if let Some(next) = selected.first().copied() {
                ep = next;
            }
            selected_by_level[level as usize] = selected;
        }

        for (level, selected) in selected_by_level.into_iter().enumerate() {
            let level = level as u16;
            for neighbor in selected {
                self.add_reverse_edge(vectors, neighbor.id, id, neighbor.distance, level)?;
            }
        }

        self.promote_build_entry(id, target_level);
        Ok(())
    }

    fn greedy_search_query<F>(
        &self,
        mut current: ScoredPoint,
        level: u16,
        visible_len: usize,
        distance: F,
    ) -> ScoredPoint
    where
        F: Fn(u32) -> f32,
    {
        loop {
            let mut next = None;
            self.visit_published_neighbors(current.id, level, visible_len, |neighbor| {
                let candidate_distance = distance(neighbor);
                if candidate_distance < current.distance
                    && next
                        .map(|point: ScoredPoint| candidate_distance < point.distance)
                        .unwrap_or(true)
                {
                    next = Some(ScoredPoint::new(neighbor, candidate_distance));
                }
            });

            let Some(next_point) = next else {
                break;
            };
            current = next_point;
        }
        current
    }

    fn beam_search_query<F>(
        &self,
        ep: ScoredPoint,
        level: u16,
        limits: BeamLimits,
        visible_len: usize,
        visited: &mut VisitedList,
        distance: F,
    ) -> Vec<ScoredPoint>
    where
        F: Fn(u32) -> f32,
    {
        let mut candidates = BinaryHeap::with_capacity(limits.ef);
        let mut results = BinaryHeap::with_capacity(limits.ef);
        let visited_capacity = if visible_len == usize::MAX {
            self.visible_len.load(Ordering::Acquire)
        } else {
            visible_len
        };
        visited.reset(visited_capacity);
        let _ = visited.insert(ep.id);
        candidates.push(Reverse(ep));
        results.push(ep);

        while let Some(Reverse(current)) = candidates.pop() {
            let furthest = results
                .peek()
                .map(|point| point.distance)
                .unwrap_or(f32::INFINITY);
            if current.distance > furthest && results.len() == limits.ef {
                break;
            }

            self.visit_published_neighbors(current.id, level, visible_len, |neighbor| {
                if !visited.insert(neighbor) {
                    return;
                }
                let candidate = ScoredPoint::new(neighbor, distance(neighbor));
                let furthest = results
                    .peek()
                    .map(|point| point.distance)
                    .unwrap_or(f32::INFINITY);
                if results.len() < limits.ef || candidate.distance < furthest {
                    if results.len() == limits.ef {
                        results.pop();
                    }
                    results.push(candidate);
                    candidates.push(Reverse(candidate));
                }
            });
        }

        let output_limit = limits.output_limit.min(results.len());
        while results.len() > output_limit {
            results.pop();
        }
        results.into_sorted_vec()
    }

    fn greedy_search_build<F>(
        &self,
        mut current: ScoredPoint,
        level: u16,
        visible_len: usize,
        distance: F,
    ) -> Result<ScoredPoint>
    where
        F: Fn(u32) -> f32,
    {
        loop {
            let mut next = None;
            self.visit_build_neighbors(current.id, level, visible_len, |neighbor| {
                let candidate_distance = distance(neighbor);
                if candidate_distance < current.distance
                    && next
                        .map(|point: ScoredPoint| candidate_distance < point.distance)
                        .unwrap_or(true)
                {
                    next = Some(ScoredPoint::new(neighbor, candidate_distance));
                }
            })?;

            let Some(next_point) = next else {
                break;
            };
            current = next_point;
        }
        Ok(current)
    }

    fn beam_search_build<F>(
        &self,
        ep: ScoredPoint,
        level: u16,
        limits: BuildBeamLimits,
        visited: &mut VisitedList,
        distance: F,
    ) -> Result<Vec<ScoredPoint>>
    where
        F: Fn(u32) -> f32,
    {
        let mut candidates = BinaryHeap::with_capacity(limits.ef);
        let mut results = BinaryHeap::with_capacity(limits.ef);
        let visited_capacity = if limits.visible_len == usize::MAX {
            limits.visited_capacity
        } else {
            limits.visible_len
        };
        visited.reset(visited_capacity);
        let _ = visited.insert(ep.id);
        candidates.push(Reverse(ep));
        results.push(ep);

        while let Some(Reverse(current)) = candidates.pop() {
            let furthest = results
                .peek()
                .map(|point| point.distance)
                .unwrap_or(f32::INFINITY);
            if current.distance > furthest && results.len() == limits.ef {
                break;
            }

            self.visit_build_neighbors(current.id, level, limits.visible_len, |neighbor| {
                if !visited.insert(neighbor) {
                    return;
                }
                let candidate = ScoredPoint::new(neighbor, distance(neighbor));
                let furthest = results
                    .peek()
                    .map(|point| point.distance)
                    .unwrap_or(f32::INFINITY);
                if results.len() < limits.ef || candidate.distance < furthest {
                    if results.len() == limits.ef {
                        results.pop();
                    }
                    results.push(candidate);
                    candidates.push(Reverse(candidate));
                }
            })?;
        }

        Ok(results.into_sorted_vec())
    }

    fn visit_published_neighbors<F>(&self, id: u32, level: u16, visible_len: usize, mut visit: F)
    where
        F: FnMut(u32),
    {
        let node = &self.nodes[id as usize];
        if !node.has_level(level) {
            return;
        }
        if level == 0 {
            let packed = self.packed_level0.load();
            if let Some(neighbors) = packed.neighbors(id) {
                if visible_len == usize::MAX {
                    for neighbor in neighbors.iter().copied() {
                        visit(neighbor);
                    }
                    return;
                }
                for neighbor in neighbors.iter().copied() {
                    if neighbor as usize >= visible_len {
                        continue;
                    }
                    visit(neighbor);
                }
                return;
            }
        }
        let Some(published) = &node.levels[level as usize].published else {
            return;
        };
        let published = published.load();
        if visible_len == usize::MAX {
            for neighbor in published.iter().copied() {
                visit(neighbor);
            }
            return;
        }
        for neighbor in published.iter().copied() {
            if neighbor as usize >= visible_len {
                continue;
            }
            visit(neighbor);
        }
    }

    fn visit_build_neighbors<F>(
        &self,
        id: u32,
        level: u16,
        visible_len: usize,
        mut visit: F,
    ) -> Result<()>
    where
        F: FnMut(u32),
    {
        let node = &self.nodes[id as usize];
        if !node.has_level(level) {
            return Ok(());
        }
        let ranked = node.ranked(level)?;
        if visible_len == usize::MAX {
            for neighbor in ranked.iter().map(|point| point.id) {
                visit(neighbor);
            }
            return Ok(());
        }
        for neighbor in ranked.iter().map(|point| point.id) {
            if neighbor as usize >= visible_len {
                continue;
            }
            visit(neighbor);
        }
        Ok(())
    }

    /// Algorithm 4 with `extendCandidates = false` and
    /// `keepPrunedConnections = true`.
    ///
    /// Callers supply the complete candidate set. Candidates rejected by the
    /// diversity heuristic refill the result up to `limit`, preserving graph
    /// connectivity without exceeding the role-specific degree bound.
    fn select_neighbors(
        &self,
        vectors: &impl VectorSource,
        candidates: &[ScoredPoint],
        limit: usize,
    ) -> Vec<ScoredPoint> {
        let mut candidates = candidates.to_vec();
        prune_neighbors(vectors, &mut candidates, limit);
        candidates
    }

    fn set_node_neighbors(&self, id: u32, level: u16, neighbors: &[ScoredPoint]) -> Result<()> {
        let node = &self.nodes[id as usize];
        let mut ranked = node.ranked(level)?;
        size_ranked(&mut ranked, max_reciprocal_neighbors(self.params.m, level));
        ranked.clear();
        ranked.extend_from_slice(neighbors);
        node.mark_dirty(level);
        Ok(())
    }

    fn add_reverse_edge(
        &self,
        vectors: &impl VectorSource,
        target: u32,
        neighbor: u32,
        distance: f32,
        level: u16,
    ) -> Result<()> {
        let node = &self.nodes[target as usize];
        if !node.has_level(level) {
            return Ok(());
        }
        let mut ranked = node.ranked(level)?;
        if ranked.iter().any(|point| point.id == neighbor) {
            return Ok(());
        }
        let limit = max_reciprocal_neighbors(self.params.m, level);
        // A level the node was never connected at (the first node, or a level
        // above the graph's top when it was inserted) is first reached here.
        size_ranked(&mut ranked, limit);
        let point = ScoredPoint::new(neighbor, distance);
        if ranked.len() < limit {
            ranked.push(point);
            ranked.sort_unstable();
        } else {
            // Full: prune the list plus the newcomer in one scratch buffer and
            // copy the survivors back, so `ranked` keeps its capacity.
            let mut candidates = Vec::with_capacity(ranked.len() + 1);
            candidates.extend_from_slice(&ranked);
            candidates.push(point);
            prune_neighbors(vectors, &mut candidates, limit);
            ranked.clear();
            ranked.extend_from_slice(&candidates);
        }
        node.mark_dirty(level);
        Ok(())
    }

    fn promote_build_entry(&self, id: u32, target_level: u16) {
        loop {
            let entry = self.build_entry_point.load(Ordering::Acquire);
            let entry_level = self.nodes[entry as usize].target_level;
            // Sequential strict promotion keeps the first node at the global
            // maximum level. Prefer the lower id on equal levels so parallel
            // insertion converges to that same deterministic fixpoint.
            if target_level < entry_level || (target_level == entry_level && id >= entry) {
                break;
            }
            if self
                .build_entry_point
                .compare_exchange(entry, id, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                self.build_max_level
                    .fetch_max(target_level + 1, Ordering::AcqRel);
                break;
            }
        }
    }

    fn publish_visible(&self, len: usize) {
        let entry = self.build_entry_point.load(Ordering::Acquire);
        self.visible_entry_point.store(entry, Ordering::Release);
        self.visible_max_level.store(
            self.build_max_level.load(Ordering::Acquire),
            Ordering::Release,
        );
        self.visible_len.store(len, Ordering::Release);
    }

    fn publish_dirty(&self, len: usize) -> Result<()> {
        let mut has_level0_update = false;
        for node in self.nodes.iter().take(len) {
            let mut mask = node.dirty_levels.swap(0, Ordering::AcqRel);
            while mask != 0 {
                let level = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                if level >= node.levels.len() {
                    continue;
                }
                let ranked = node.ranked(level as u16)?;
                node.levels[level].publish_from_ranked(&ranked);
                has_level0_update |= level == 0;
            }
        }
        if has_level0_update {
            self.rebuild_packed_level0(len)?;
        }
        Ok(())
    }

    fn rebuild_packed_level0(&self, len: usize) -> Result<()> {
        let mut offsets = Vec::with_capacity(len + 1);
        let mut neighbors =
            Vec::with_capacity(len.saturating_mul(max_reciprocal_neighbors(self.params.m, 0)));
        offsets.push(0);
        for node in self.nodes.iter().take(len) {
            let ranked = node.ranked(0)?;
            neighbors.extend(ranked.iter().map(|point| point.id));
            offsets.push(neighbors.len());
        }

        let packed_bytes = offsets.capacity() * std::mem::size_of::<usize>()
            + neighbors.capacity() * std::mem::size_of::<u32>();

        // ArcSwap reclaims the prior snapshot once no reader guard holds it.
        self.packed_level0
            .store(Arc::new(PackedLevel { offsets, neighbors }));
        self.packed_bytes.store(packed_bytes, Ordering::Relaxed);
        Ok(())
    }
}

fn lance_hnsw_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(VECTOR_ID_COL, DataType::UInt32, true),
        Field::new(
            NEIGHBORS_COL,
            DataType::List(Field::new_list_field(DataType::UInt32, true).into()),
            true,
        ),
        Field::new(
            DIST_COL,
            DataType::List(Field::new_list_field(DataType::Float32, true).into()),
            true,
        ),
    ]))
}

fn random_level(params: &BuildParams, rng: &mut SmallRng) -> u16 {
    let ml = 1.0 / (params.m as f32).ln();
    ((-rng.random::<f32>().ln() * ml) as u16).min(params.max_level - 1)
}

fn max_reciprocal_neighbors(m: usize, level: u16) -> usize {
    if level == 0 { m * 2 } else { m }
}

/// Apply Algorithm 4's diversity heuristic with `keepPrunedConnections = true`.
///
/// Candidates rejected by the heuristic refill the result in nearest-first
/// order, reducing sparse directed components without exceeding `limit`.
fn prune_neighbors(vectors: &impl VectorSource, candidates: &mut Vec<ScoredPoint>, limit: usize) {
    candidates.sort_unstable();
    if candidates.len() <= limit {
        return;
    }
    let mut kept = 0;
    let mut pruned = Vec::with_capacity(candidates.len());
    for read in 0..candidates.len() {
        if kept == limit {
            break;
        }
        let candidate = candidates[read];
        if kept == 0 || vectors.prefers_candidate(candidate, &candidates[..kept]) {
            candidates[kept] = candidate;
            kept += 1;
        } else {
            pruned.push(candidate);
        }
    }
    let refill = (limit - kept).min(pruned.len());
    candidates[kept..kept + refill].copy_from_slice(&pruned[..refill]);
    candidates.truncate(kept + refill);
}

/// Give a node's `ranked` list exactly `max_neighbors` of capacity the first
/// time it is written. Every later write fits in it (a full list is pruned
/// through a scratch buffer), so its capacity is what
/// [`LevelLinks::linked_bytes`] charges.
fn size_ranked(ranked: &mut Vec<ScoredPoint>, max_neighbors: usize) {
    if ranked.capacity() < max_neighbors {
        ranked.reserve_exact(max_neighbors - ranked.len());
    }
}

/// One list per worker, doubled so a searcher never blocks on the queue.
fn visited_pool_size() -> usize {
    rayon::current_num_threads().max(1) * 2
}

/// Heap the visited pool settles at for a graph of `capacity` nodes. The lists
/// are pushed empty but `VisitedList::reset` resizes each to one bit per node
/// on first use, so the pool is charged at its grown size from the start.
fn visited_pool_bytes(capacity: usize) -> usize {
    visited_pool_size()
        * (std::mem::size_of::<VisitedList>()
            + capacity.div_ceil(WORD_BITS) * std::mem::size_of::<usize>())
}

#[derive(Debug)]
struct VisitedList {
    words: Vec<usize>,
    touched: Vec<u32>,
}

impl VisitedList {
    fn new(capacity: usize) -> Self {
        Self {
            words: vec![0; capacity.div_ceil(WORD_BITS)],
            touched: Vec::new(),
        }
    }

    fn reset(&mut self, capacity: usize) {
        for id in self.touched.drain(..) {
            let idx = id as usize;
            self.words[idx / WORD_BITS] &= !(1usize << (idx % WORD_BITS));
        }
        let needed_words = capacity.div_ceil(WORD_BITS);
        if self.words.len() < needed_words {
            self.words.resize(needed_words, 0);
        }
    }

    fn insert(&mut self, id: u32) -> bool {
        let idx = id as usize;
        let word = idx / WORD_BITS;
        let bit = 1usize << (idx % WORD_BITS);
        if self.words[word] & bit == 0 {
            self.words[word] |= bit;
            self.touched.push(id);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use arrow_array::types::Float32Type;
    use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array};
    use arrow_schema::{DataType, Field};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::vector::hnsw::HNSW;
    use lance_index::vector::v3::subindex::IvfSubIndex;
    use lance_linalg::distance::DistanceType;
    use lance_testing::datagen::generate_random_array_with_seed;
    use rstest::rstest;

    use super::super::{ArrowFixedSizeListVectorStore, VectorSource, VectorStoreSnapshot};
    use super::*;

    fn fsl_from_values(values: Vec<f32>, dim: usize) -> Arc<FixedSizeListArray> {
        let values = Arc::new(Float32Array::from(values)) as ArrayRef;
        Arc::new(
            FixedSizeListArray::try_new(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
                values,
                None,
            )
            .unwrap(),
        )
    }

    fn fsl(rows: usize, dim: usize) -> Arc<FixedSizeListArray> {
        let mut values = Vec::with_capacity(rows * dim);
        for row in 0..rows {
            for col in 0..dim {
                values.push(row as f32 + col as f32 * 0.001);
            }
        }
        fsl_from_values(values, dim)
    }

    fn vector_snapshot(values: Vec<f32>, dim: usize) -> VectorStoreSnapshot {
        assert_eq!(values.len() % dim, 0);
        let rows = values.len() / dim;
        let store = Arc::new(
            ArrowFixedSizeListVectorStore::try_new(rows, 1, dim, DistanceType::L2).unwrap(),
        );
        let ids = store.append_batch(fsl_from_values(values, dim), 0).unwrap();
        assert_eq!(ids, 0..rows as u32);
        store.snapshot()
    }

    #[rstest]
    #[case::zero_max_level(
        BuildParams::default().max_level(0),
        "max_level must be greater than 0"
    )]
    #[case::too_many_levels(
        BuildParams::default().max_level(u64::BITS as u16 + 1),
        "max_level must be <= 64, got 65"
    )]
    #[case::small_m(
        BuildParams::default().num_edges(MIN_HNSW_M - 1),
        "m must be at least 4"
    )]
    #[case::small_ef(
        BuildParams::default().num_edges(20).ef_construction(19),
        "ef_construction must be at least m (20), got 19"
    )]
    #[case::overflowing_level_zero_limit(
        BuildParams::default()
            .num_edges(usize::MAX)
            .ef_construction(usize::MAX),
        "level-0 reciprocal limit can be represented"
    )]
    #[case::unallocatable_level_zero_limit(
        BuildParams::default()
            .num_edges(usize::MAX / 2)
            .ef_construction(usize::MAX / 2),
        "neighbor_capacity"
    )]
    fn test_rejects_invalid_build_params(
        #[case] params: BuildParams,
        #[case] expected_message: &str,
    ) {
        let Err(error) = HnswGraph::try_new(1, params) else {
            panic!("expected invalid build parameters to be rejected");
        };
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error.to_string().contains(expected_message),
            "unexpected error: {error}"
        );
    }

    /// Random level assignment includes node 0, and promotion converges to
    /// the first globally highest node regardless of insertion scheduling.
    #[test]
    fn test_entry_point_matches_sequential_strict_promotion() {
        const CAPACITY: usize = 64;
        let params = BuildParams::default()
            .max_level(2)
            .num_edges(4)
            .ef_construction(16)
            .seed(7);
        let graph = HnswGraph::try_new(CAPACITY, params.clone()).unwrap();

        let mut rng = SmallRng::seed_from_u64(params.seed);
        let expected_levels = (0..CAPACITY)
            .map(|_| random_level(&params, &mut rng))
            .collect::<Vec<_>>();
        let actual_levels = graph
            .nodes
            .iter()
            .map(|node| node.target_level)
            .collect::<Vec<_>>();
        assert_eq!(actual_levels, expected_levels);

        let highest_level = *expected_levels.iter().max().unwrap();
        assert!(
            expected_levels
                .iter()
                .filter(|level| **level == highest_level)
                .count()
                > 1,
            "test seed must produce an equal-height promotion race"
        );
        let expected_entry = expected_levels
            .iter()
            .position(|level| *level == highest_level)
            .unwrap() as u32;

        graph
            .build_max_level
            .store(graph.nodes[0].target_level + 1, Ordering::Release);
        for id in (1..CAPACITY as u32).rev() {
            graph.promote_build_entry(id, graph.nodes[id as usize].target_level);
        }

        assert_eq!(
            graph.build_entry_point.load(Ordering::Acquire),
            expected_entry
        );
        assert_eq!(
            graph.build_max_level.load(Ordering::Acquire),
            highest_level + 1
        );
    }

    /// Algorithm 1 limits the newly inserted node to M connections at level
    /// 0; Mmax0 = 2M remains the capacity for reciprocal edges on old nodes.
    #[test]
    fn test_new_node_uses_m_connections() {
        const M: usize = 4;
        const DIM: usize = 8;
        let mut values = vec![0.0; (DIM + 1) * DIM];
        for row in 0..DIM {
            values[row * DIM + row] = 1.0;
        }
        let vectors = vector_snapshot(values, DIM);
        let graph = HnswGraph::try_new(
            DIM + 1,
            BuildParams::default()
                .max_level(1)
                .num_edges(M)
                .ef_construction(DIM + 1),
        )
        .unwrap();

        graph
            .insert_batch_serial(0..DIM as u32 + 1, &vectors)
            .unwrap();

        assert_eq!(graph.nodes[DIM].ranked(0).unwrap().len(), M);
        assert!(
            graph
                .nodes
                .iter()
                .all(|node| { node.ranked(0).unwrap().len() <= max_reciprocal_neighbors(M, 0) })
        );
    }

    /// Reciprocal candidates join the complete old-plus-new candidate set
    /// before Algorithm 4. The farther diverse edge survives and rejected
    /// candidates refill the list to Mmax0.
    #[test]
    fn test_reciprocal_edge_uses_complete_candidates_and_refills() {
        const M: usize = 4;
        let mut values = vec![0.0, 0.0];
        for x in 1..=8 {
            values.extend([x as f32, 0.0]);
        }
        values.extend([0.0, 9.0]);
        let vectors = vector_snapshot(values, 2);
        let graph = HnswGraph::try_new(
            10,
            BuildParams::default()
                .max_level(1)
                .num_edges(M)
                .ef_construction(16),
        )
        .unwrap();
        let existing: Vec<_> = (1..=8)
            .map(|id| ScoredPoint::new(id, vectors.distance_between(0, id)))
            .collect();
        graph.set_node_neighbors(0, 0, &existing).unwrap();

        graph
            .add_reverse_edge(&vectors, 0, 9, vectors.distance_between(0, 9), 0)
            .unwrap();

        let selected = graph.nodes[0]
            .ranked(0)
            .unwrap()
            .iter()
            .map(|point| point.id)
            .collect::<Vec<_>>();
        assert_eq!(selected, vec![1, 9, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_parallel_insert_searches_visible_graph() {
        let rows = 256;
        let dim = 16;
        let store = Arc::new(
            ArrowFixedSizeListVectorStore::try_new(512, 4, dim, DistanceType::L2).unwrap(),
        );
        let ids = store.append_batch(fsl(rows, dim), 100).unwrap();
        let snapshot = store.snapshot();
        let graph = HnswGraph::try_new(
            512,
            BuildParams::mem_wal_default()
                .num_edges(8)
                .ef_construction(32)
                .seed(7),
        )
        .unwrap();

        graph.insert_batch(ids, &snapshot).unwrap();
        assert_eq!(graph.len(), rows);
        for node in &graph.nodes {
            assert!(node.levels[0].published.is_none());
            assert!(
                node.levels[1..]
                    .iter()
                    .all(|links| links.published.is_some())
            );
        }

        let query = snapshot.vector(42);
        let result = graph
            .search(query, SearchParams::new(5, 32), &snapshot)
            .unwrap();
        assert!(result.iter().any(|point| point.id == 42));

        let widest = graph
            .search(query, SearchParams::new(5, usize::MAX), &snapshot)
            .unwrap();
        assert!(widest.iter().any(|point| point.id == 42));
    }

    /// Each node's neighbor lists are charged as it is inserted and hold
    /// exactly `max_neighbors` once written, across incremental batches and a
    /// full list taking one more reverse edge; slots never filled allocate
    /// none.
    #[test]
    fn test_neighbor_lists_are_charged_and_bounded_per_inserted_node() {
        let (rows, capacity, dim, m) = (400, 512, 8, 4);
        let store = Arc::new(
            ArrowFixedSizeListVectorStore::try_new(capacity, 4, dim, DistanceType::L2).unwrap(),
        );
        // Random points, so the diversity heuristic fills lists.
        let values = generate_random_array_with_seed::<Float32Type>(rows * dim, [7; 32]);
        let vectors = FixedSizeListArray::try_new_from_values(values, dim as i32).unwrap();
        store.append_batch(Arc::new(vectors), 0).unwrap();
        let snapshot = store.snapshot();
        let graph = HnswGraph::try_new(
            capacity,
            BuildParams::mem_wal_default()
                .num_edges(m)
                .ef_construction(32)
                .seed(11),
        )
        .unwrap();
        let fixed = graph.resident_bytes();

        let check = |inserted: usize| {
            for (id, node) in graph.nodes.iter().enumerate() {
                for level in 0..node.levels.len() as u16 {
                    let capacity = node.ranked(level).unwrap().capacity();
                    let allowed = if id < inserted {
                        max_reciprocal_neighbors(m, level)
                    } else {
                        0
                    };
                    assert!(
                        capacity == 0 || capacity == allowed,
                        "node {id} level {level} holds {capacity} slots, allowed {allowed}"
                    );
                }
            }
            let linked: usize = graph.nodes[..inserted]
                .iter()
                .map(|node| Node::linked_bytes(node.target_level, m))
                .sum();
            assert_eq!(
                graph.resident_bytes(),
                fixed + linked + graph.packed_bytes.load(Ordering::Relaxed),
                "charge after {inserted} rows"
            );
        };
        for start in (0..rows as u32).step_by(50) {
            graph.insert_batch(start..start + 50, &snapshot).unwrap();
            check(start as usize + 50);
        }

        // A full list taking one more edge is pruned in place, not grown.
        let limit = max_reciprocal_neighbors(m, 0);
        let target = (0..rows as u32)
            .find(|&id| graph.nodes[id as usize].ranked(0).unwrap().len() == limit)
            .expect("some level-0 list should be full");
        let linked: Vec<u32> = graph.nodes[target as usize]
            .ranked(0)
            .unwrap()
            .iter()
            .map(|point| point.id)
            .collect();
        let newcomer = (0..rows as u32)
            .find(|id| *id != target && !linked.contains(id))
            .unwrap();
        let distance = snapshot.distance_between(target, newcomer);
        graph
            .add_reverse_edge(&snapshot, target, newcomer, distance, 0)
            .unwrap();
        check(rows);
    }

    /// The graph must be exportable to a boundary its caller chose, because the
    /// companion vector storage is captured separately: a graph that advanced
    /// past it would name rows the storage batch has no vector for, and a search
    /// would score them.
    #[test]
    fn to_lance_hnsw_batch_honors_a_caller_supplied_prefix() {
        const ROWS: usize = 256;
        const DIM: usize = 8;
        const PREFIX: usize = 64;
        let store = Arc::new(
            ArrowFixedSizeListVectorStore::try_new(512, 4, DIM, DistanceType::L2).unwrap(),
        );
        let ids = store.append_batch(fsl(ROWS, DIM), 0).unwrap();
        let snapshot = store.snapshot();
        let graph = HnswGraph::try_new(
            512,
            BuildParams::mem_wal_default()
                .num_edges(8)
                .ef_construction(32)
                .seed(17),
        )
        .unwrap();
        graph.insert_batch(ids, &snapshot).unwrap();

        let full = graph.to_lance_hnsw_batch(None).unwrap();
        let bounded = graph.to_lance_hnsw_batch(Some(PREFIX)).unwrap();
        assert!(
            bounded.num_rows() < full.num_rows(),
            "the cap has to actually bound the export"
        );
        assert_eq!(HNSW::load(bounded.clone()).unwrap().len(), PREFIX);

        // Every edge must stay inside the requested prefix, not merely inside
        // whatever the graph had published.
        let neighbors = bounded
            .column(1)
            .as_any()
            .downcast_ref::<arrow_array::ListArray>()
            .expect("neighbors column is a list");
        for row in 0..bounded.num_rows() {
            let ids = neighbors.value(row);
            let ids = ids
                .as_any()
                .downcast_ref::<arrow_array::UInt32Array>()
                .expect("neighbor ids are u32");
            for i in 0..ids.len() {
                assert!(
                    (ids.value(i) as usize) < PREFIX,
                    "row {row} points at {} outside the {PREFIX}-node prefix",
                    ids.value(i)
                );
            }
        }
    }

    /// An export racing inserts must persist only edges inside the prefix it
    /// publishes.
    ///
    /// `to_lance_hnsw_batch` snapshots `visible_len` but reads each node's ranked
    /// list live, so a batch landing mid-export can append itself to a node
    /// already emitted. `HNSW::load` then slices level 0 to the exported rows and
    /// a walk over that edge addresses past them. `search` bounds in-memory
    /// traversal by `visible_len` for the same reason.
    #[test]
    fn test_lance_hnsw_batch_edges_stay_inside_the_exported_prefix() {
        const ROWS: usize = 1024;
        const DIM: usize = 16;
        const CHUNK: usize = 32;
        let store = Arc::new(
            ArrowFixedSizeListVectorStore::try_new(2048, 8, DIM, DistanceType::L2).unwrap(),
        );
        let ids = store.append_batch(fsl(ROWS, DIM), 0).unwrap();
        let snapshot = store.snapshot();
        let graph = Arc::new(
            HnswGraph::try_new(
                2048,
                BuildParams::mem_wal_default()
                    .num_edges(8)
                    .ef_construction(32)
                    .seed(13),
            )
            .unwrap(),
        );

        // Seed a prefix so an export has real adjacency to walk.
        let chunk = CHUNK as u32;
        graph
            .insert_batch(ids.start..ids.start + chunk, &snapshot)
            .unwrap();

        let writing = Arc::new(AtomicBool::new(true));
        let writer = {
            let graph = Arc::clone(&graph);
            let writing = Arc::clone(&writing);
            std::thread::spawn(move || {
                let mut next = ids.start + chunk;
                while next < ids.end {
                    let stop = (next + chunk).min(ids.end);
                    graph.insert_batch(next..stop, &snapshot).unwrap();
                    next = stop;
                }
                writing.store(false, Ordering::Release);
            })
        };

        // Export while the writer runs rather than a fixed count, so the overlap
        // does not depend on how fast this machine inserts.
        let mut exports = 0;
        let mut edges_checked = 0;
        while writing.load(Ordering::Acquire) || exports < 5 {
            // Read as a search does: every visible node is in the packed level 0.
            let visible = graph.len() as u32;
            let packed = graph.packed_level0.load();
            assert!(
                (0..visible).all(|id| packed.neighbors(id).is_some()),
                "a visible node is missing from the packed level-0 snapshot"
            );
            let batch = graph.to_lance_hnsw_batch(None).unwrap();
            let rows = batch.num_rows();
            let neighbors = batch
                .column(1)
                .as_any()
                .downcast_ref::<arrow_array::ListArray>()
                .expect("neighbors column is a list");
            let distances = batch
                .column(2)
                .as_any()
                .downcast_ref::<arrow_array::ListArray>()
                .expect("distances column is a list");
            // Level 0 holds every visible node, so its row count is the prefix.
            let prefix = HNSW::load(batch.clone()).unwrap().len() as u32;
            for row in 0..rows {
                let ids = neighbors.value(row);
                let ids = ids
                    .as_any()
                    .downcast_ref::<arrow_array::UInt32Array>()
                    .expect("neighbor ids are u32");
                assert_eq!(
                    ids.len(),
                    distances.value(row).len(),
                    "row {row} pairs {} ids with {} distances",
                    ids.len(),
                    distances.value(row).len()
                );
                for i in 0..ids.len() {
                    let nid = ids.value(i);
                    assert!(
                        nid < prefix,
                        "exported row {row} points at {nid}, outside the \
                         {prefix}-node prefix it was exported with"
                    );
                    edges_checked += 1;
                }
            }
            exports += 1;
        }
        writer.join().unwrap();
        assert!(edges_checked > 0, "test never inspected an edge");
    }

    #[test]
    fn test_lance_hnsw_batch_loads_with_lance_index() {
        let rows = 64;
        let dim = 8;
        let store = Arc::new(
            ArrowFixedSizeListVectorStore::try_new(128, 2, dim, DistanceType::L2).unwrap(),
        );
        let ids = store.append_batch(fsl(rows, dim), 0).unwrap();
        let snapshot = store.snapshot();
        let graph = HnswGraph::try_new(
            128,
            BuildParams::mem_wal_default()
                .max_level(12)
                .num_edges(8)
                .ef_construction(32)
                .seed(11),
        )
        .unwrap();
        graph.insert_batch(ids, &snapshot).unwrap();

        let batch = graph.to_lance_hnsw_batch(None).unwrap();
        let metadata: LanceHnswMetadata = serde_json::from_str(
            batch
                .schema_ref()
                .metadata()
                .get(HNSW_METADATA_KEY)
                .unwrap(),
        )
        .unwrap();
        let actual_levels = graph.nodes[..rows]
            .iter()
            .map(|node| node.target_level as usize + 1)
            .max()
            .unwrap();
        let highest_level = actual_levels - 1;
        let expected_entry = graph.nodes[..rows]
            .iter()
            .position(|node| node.target_level as usize == highest_level)
            .unwrap() as u32;
        assert_eq!(metadata.entry_point, expected_entry);
        assert_eq!(metadata.level_offsets.len(), 13);
        assert_eq!(*metadata.level_offsets.last().unwrap(), batch.num_rows());
        assert!(
            metadata.level_offsets[actual_levels..]
                .iter()
                .all(|offset| *offset == batch.num_rows()),
            "configured trailing levels must serialize as empty ranges"
        );
        let loaded = HNSW::load(batch).unwrap();
        assert_eq!(loaded.len(), rows);
    }

    #[test]
    fn test_minimum_params_reachability() {
        let rows = 2048;
        let dim = 32;
        let mut rng = SmallRng::seed_from_u64(0);
        let values = (0..rows * dim).map(|_| rng.random::<f32>()).collect();
        let vectors = vector_snapshot(values, dim);
        let graph = HnswGraph::try_new(
            rows,
            BuildParams::default()
                .num_edges(MIN_HNSW_M)
                .ef_construction(MIN_HNSW_M),
        )
        .unwrap();
        graph.insert_batch(0..rows as u32, &vectors).unwrap();

        let results = graph
            .search(vectors.vector(0), SearchParams::new(rows, rows), &vectors)
            .unwrap();
        assert!(
            results.len() >= rows * 90 / 100,
            "minimum construction settings reached only {} of {rows}",
            results.len()
        );
    }

    /// Rows written again under new keys repeat their vectors exactly; each
    /// distinct vector must keep a reachable copy, including one stored more
    /// often than a level-0 list holds (8 at `m = 4`). Not every copy is
    /// reachable: a list holds one copy of a vector, so a search for many
    /// identical rows may return only some of them. Not run under dot product,
    /// where a vector's nearest row need not hold the same vector.
    #[rstest::rstest]
    fn test_each_repeated_vector_keeps_a_reachable_copy(
        #[values(2, 12)] copies: usize,
        #[values(DistanceType::L2, DistanceType::Cosine)] distance_type: DistanceType,
    ) {
        let (distinct, dim) = (100, 8);
        let values = generate_random_array_with_seed::<Float32Type>(distinct * dim, [3; 32]);
        let rows = distinct * copies;
        let repeated: Vec<f32> = values
            .values()
            .iter()
            .copied()
            .cycle()
            .take(rows * dim)
            .collect();
        let vectors =
            FixedSizeListArray::try_new_from_values(Float32Array::from(repeated), dim as i32)
                .unwrap();
        let store =
            Arc::new(ArrowFixedSizeListVectorStore::try_new(rows, 4, dim, distance_type).unwrap());
        let ids = store.append_batch(Arc::new(vectors), 0).unwrap();
        let snapshot = store.snapshot();
        let params = BuildParams::mem_wal_default()
            .num_edges(4)
            .ef_construction(16)
            .seed(1);
        let graph = HnswGraph::try_new(rows, params).unwrap();
        graph.insert_batch_serial(ids, &snapshot).unwrap();

        let unreachable = (0..distinct as u32)
            .filter(|&id| {
                let query = snapshot.vector(id);
                let found = graph
                    .search(query, SearchParams::new(1, rows), &snapshot)
                    .unwrap();
                !found.iter().any(|point| snapshot.vector(point.id) == query)
            })
            .count();
        assert_eq!(
            unreachable, 0,
            "{unreachable} of {distinct} vectors have no reachable copy"
        );
    }
}
