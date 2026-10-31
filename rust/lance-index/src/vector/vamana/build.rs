// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Building a partition's graph.

use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{ArrayRef, Float32Array, RecordBatch};
use lance_core::{Error, Result};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

use super::batch::{Batches, link_in_batches};
use super::graph::{MAX_VERTICES, PartitionGraph};
use super::insert::{InsertScratch, Linking, ResidentVectors, insert_point};
use super::params::VamanaBuildParams;
use super::search::Comparisons;
use crate::vector::graph::{OrderedFloat, OrderedNode};
use crate::vector::storage::{DistCalculator, VectorStore};

/// How many neighbors ahead a build's search asks for a vector while it
/// measures the current one.
pub const BUILD_PREFETCH_AHEAD: usize = 2;

/// How many cache lines of a vector a build's search asks for: all of a SIFT
/// vector, the first eight of a longer one. Eight had the best geometric mean
/// over SIFT, GloVe-200, Cohere and GIST among 2 to 32 lines and whole vectors.
pub const BUILD_PREFETCH_LINES: usize = 8;

/// How many vertices a store without float vectors samples for its medoid.
const MEDOID_SAMPLE: usize = 256;

/// From this many vertices up, a partition's sweeps link vertices in batches,
/// in parallel. A smaller partition keeps the graph a vertex-at-a-time sweep
/// gives, which batches give up a little of because the vertices of a batch do
/// not see one another; and an index of small partitions has many of them,
/// which Lance already builds side by side.
pub(crate) const BATCHED_FROM_VERTICES: usize = 100_000;

/// The share of a partition one batch holds at most, as a divisor: ParlayANN's
/// 2%, past which too many vertices would search a graph that does not hold
/// one another yet.
const MAX_BATCH_DIVISOR: usize = 50;

/// How a build's sweeps link the vertices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Sweep {
    /// One vertex at a time, each searching the graph its predecessors left.
    Sequential,
    /// At most `max_batch` vertices at a time, each searching the graph as it
    /// stood when its batch began; see [`super::batch`].
    Batched { max_batch: NonZeroUsize },
}

impl Sweep {
    /// The sweep a build over `num_vertices` vertices takes.
    pub(crate) fn for_vertices(num_vertices: usize) -> Self {
        match NonZeroUsize::new(num_vertices / MAX_BATCH_DIVISOR) {
            Some(max_batch) if num_vertices >= BATCHED_FROM_VERTICES => Self::Batched { max_batch },
            _ => Self::Sequential,
        }
    }
}

/// A partition's graph and the vertex a search of it should start from.
#[derive(Debug)]
pub struct BuiltPartition {
    pub graph: PartitionGraph,
    pub medoid: u32,
}

/// Build one partition's graph over every vector of `store`.
///
/// Algorithm 3 of the DiskANN paper: start from a random `R`-regular graph,
/// then sweep every vertex in random order, and for each one search the graph
/// as it stands, prune the visited set into its out-edges and add the
/// back-edges. The sweep runs twice, at `alpha = 1` and then at `params.alpha`:
/// the second pass works on a graph that is already navigable and can afford
/// to keep the shorter edges the first one dropped. A large partition sweeps
/// in batches; see [`Sweep::for_vertices`].
pub(crate) fn build_partition<S: VectorStore>(
    store: &S,
    params: &VamanaBuildParams,
    seed: u64,
    comparisons: &Comparisons,
) -> Result<BuiltPartition> {
    build_partition_by(
        store,
        params,
        seed,
        Sweep::for_vertices(store.len()),
        comparisons,
    )
}

/// [`build_partition`] with the sweep chosen by the caller.
pub(crate) fn build_partition_by<S: VectorStore>(
    store: &S,
    params: &VamanaBuildParams,
    seed: u64,
    sweep: Sweep,
    comparisons: &Comparisons,
) -> Result<BuiltPartition> {
    params.validate()?;
    let num_vertices = addressable_len(store.len())?;
    if num_vertices == 0 {
        return Err(Error::invalid_input(
            "Vamana cannot build a graph over an empty partition".to_string(),
        ));
    }

    let mut rng = SmallRng::seed_from_u64(seed);
    let row_ids = (0..num_vertices)
        .map(|id| store.row_id(id))
        .collect::<Vec<_>>();
    let mut graph = PartitionGraph::edgeless(params.max_degree, row_ids)?;
    randomize(&mut graph, &mut rng)?;
    let medoid = medoid(store, seed, comparisons)?;

    let mut scratch = InsertScratch::new(num_vertices as usize, params.max_degree);
    let mut order = (0..num_vertices).collect::<Vec<_>>();
    let resident = ResidentVectors::of(store);

    for alpha in [1.0, params.alpha] {
        let linking = Linking {
            alpha,
            search_list_size: params.search_list_size,
            vectors: resident.as_ref(),
        };
        order.shuffle(&mut rng);
        match sweep {
            Sweep::Sequential => {
                for point in &order {
                    insert_point(
                        &mut graph,
                        store,
                        &mut scratch,
                        &linking,
                        *point,
                        medoid,
                        comparisons,
                    )?;
                }
            }
            Sweep::Batched { max_batch } => link_in_batches(
                &mut graph,
                store,
                scratch.prune_outputs_mut(),
                &linking,
                &Batches {
                    order: &order,
                    entry_point: medoid,
                    max_batch,
                },
                comparisons,
            )?,
        }
    }

    Ok(BuiltPartition { graph, medoid })
}

/// A store's length as the `u32` ids its vertices are addressed by.
pub(crate) fn addressable_len(num_vertices: usize) -> Result<u32> {
    if num_vertices > MAX_VERTICES {
        return Err(Error::invalid_input(format!(
            "Vamana cannot build over {num_vertices} vectors, more than the addressable \
             {MAX_VERTICES}"
        )));
    }
    Ok(num_vertices as u32)
}

/// Give every vertex `min(R, n - 1)` random out-edges, so the first pass has
/// somewhere to walk.
pub(crate) fn randomize(graph: &mut PartitionGraph, rng: &mut SmallRng) -> Result<()> {
    let num_vertices = graph.len();
    let degree = (graph.max_degree() as usize).min(num_vertices.saturating_sub(1));
    if degree == 0 {
        return Ok(());
    }

    let mut neighbors = Vec::with_capacity(degree);
    for point in 0..num_vertices as u32 {
        neighbors.clear();
        // Drawn from the other `n - 1` vertices and shifted past `point`, so a
        // vertex never draws itself or a duplicate.
        neighbors.extend(
            rand::seq::index::sample(rng, num_vertices - 1, degree)
                .into_iter()
                .map(|drawn| {
                    let drawn = drawn as u32;
                    if drawn >= point { drawn + 1 } else { drawn }
                }),
        );
        graph.set_neighbors(point, &neighbors)?;
    }
    Ok(())
}

/// The vertex a search should start from.
///
/// Over float vectors, the vertex nearest the centroid: Lance's `L2` is the
/// squared distance, and `sum_j ||x_i - x_j||^2 = n ||x_i - c||^2 + const`, so
/// that vertex is the medoid exactly, at `O(n d)` rather than `O(n^2 d)`. Over
/// codes there is no centroid to measure from, so it is the medoid of a sample.
pub(crate) fn medoid<S: VectorStore>(
    store: &S,
    seed: u64,
    comparisons: &Comparisons,
) -> Result<u32> {
    let num_vertices = addressable_len(store.len())?;
    if num_vertices == 0 {
        return Err(Error::invalid_input(
            "Vamana cannot pick a medoid from an empty partition".to_string(),
        ));
    }
    let Some(centroid) = centroid(store, num_vertices as usize)? else {
        return Ok(sampled_medoid(store, seed, comparisons));
    };
    let from_centroid = store.dist_calculator(Arc::new(centroid) as ArrayRef, 0.0);
    comparisons.record(num_vertices as u64);
    let mut best = (f32::INFINITY, 0);
    for candidate in 0..num_vertices {
        let distance = from_centroid.distance(candidate);
        if distance < best.0 {
            best = (distance, candidate);
        }
    }
    Ok(best.1)
}

/// The vertex of a seeded sample with the least summed distance to the rest of
/// the sample.
fn sampled_medoid<S: VectorStore>(store: &S, seed: u64, comparisons: &Comparisons) -> u32 {
    let num_vertices = store.len();
    let sample = if num_vertices <= MEDOID_SAMPLE {
        (0..num_vertices as u32).collect::<Vec<_>>()
    } else {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut sample = rand::seq::index::sample(&mut rng, num_vertices, MEDOID_SAMPLE)
            .into_iter()
            .map(|id| id as u32)
            .collect::<Vec<_>>();
        sample.sort_unstable();
        sample
    };
    comparisons.record((sample.len() * sample.len()) as u64);
    let mut best = (f64::INFINITY, sample[0]);
    for candidate in &sample {
        let from_candidate = store.dist_calculator_from_id(*candidate);
        let total = sample
            .iter()
            .map(|other| from_candidate.distance(*other).max(0.0) as f64)
            .sum::<f64>();
        if total < best.0 {
            best = (total, *candidate);
        }
    }
    best.1
}

/// The mean of a store's float vectors, accumulated in `f64`: a naive `f32`
/// sum over a million values drifts by a millionth of the total. `None` when
/// the store holds no `FixedSizeList<Float32>` column to average.
fn centroid<S: VectorStore>(store: &S, num_vertices: usize) -> Result<Option<Float32Array>> {
    let mut sums: Vec<f64> = Vec::new();
    let mut counted = 0usize;
    for batch in store.to_batches()? {
        let Some((values, dimension)) = float_vectors(&batch)? else {
            return Ok(None);
        };
        if sums.is_empty() {
            sums = vec![0.0; dimension];
        } else if sums.len() != dimension {
            return Err(Error::invalid_input(format!(
                "Vamana partition mixes vectors of {} and {dimension} dimensions",
                sums.len()
            )));
        }
        let wanted = batch.num_rows() * dimension;
        let Some(values) = values.values().get(..wanted) else {
            return Err(Error::invalid_input(format!(
                "Vamana partition holds {} values for {} vectors of {dimension} dimensions",
                values.len(),
                batch.num_rows()
            )));
        };
        for vector in values.chunks_exact(dimension) {
            for (sum, value) in sums.iter_mut().zip(vector) {
                *sum += *value as f64;
            }
        }
        counted += batch.num_rows();
    }
    if counted != num_vertices {
        return Err(Error::invalid_input(format!(
            "Vamana partition reports {num_vertices} vectors but offers {counted}"
        )));
    }
    let scale = 1.0 / counted as f64;
    Ok(Some(Float32Array::from(
        sums.iter()
            .map(|sum| (sum * scale) as f32)
            .collect::<Vec<_>>(),
    )))
}

/// The one `FixedSizeList<Float32>` column of a store's batch and its width.
///
/// Found by type rather than by name because the column name is the store's
/// business. The type is also what keeps a centroid honest: a quantized store
/// offers codes, and a `Float32` query against a store of another value type
/// would not even be accepted.
pub(crate) fn float_vectors(batch: &RecordBatch) -> Result<Option<(&Float32Array, usize)>> {
    let mut columns = batch.columns().iter().filter_map(|column| {
        let vectors = column.as_fixed_size_list_opt()?;
        let values = vectors.values().as_primitive_opt::<Float32Type>()?;
        Some((values, vectors.value_length() as usize))
    });
    let found = columns.next();
    if columns.next().is_some() {
        return Err(Error::invalid_input(format!(
            "Vamana cannot tell which of several FixedSizeList<Float32> columns holds the \
             vectors of {}",
            batch.schema_ref()
        )));
    }
    if let Some((_, 0)) = found {
        return Err(Error::invalid_input(
            "Vamana cannot build over vectors of no width".to_string(),
        ));
    }
    Ok(found)
}

/// Choose up to `max_degree` out-edges for `point` from `candidates`.
///
/// Algorithm 2 of the DiskANN paper. Candidates are taken nearest first, and
/// after each pick a candidate that sits `alpha` times closer to the pick than
/// to `point` is dropped: the pick already routes there. At `alpha = 1` this is
/// exactly the diversity heuristic Lance applies to HNSW; above it, more short
/// edges survive.
///
/// `candidates` carry their distance to `point`, and may contain `point`
/// itself and duplicates.
pub(crate) fn robust_prune<S: VectorStore>(
    store: &S,
    point: u32,
    candidates: Vec<OrderedNode>,
    alpha: f32,
    max_degree: usize,
    comparisons: &Comparisons,
) -> Result<Vec<u32>> {
    prune(store, point, candidates, alpha, max_degree, comparisons).map(|pruned| pruned.neighbors)
}

/// What [`prune`] chose.
pub(crate) struct Pruned {
    /// The picks nearest first, ties broken by id, then the coincident fill.
    pub(crate) neighbors: Vec<u32>,
    /// Whether the coincident fill supplied part of `neighbors`, which makes
    /// the list one [`admit`] cannot take a newcomer into.
    pub(crate) has_coincident_fill: bool,
}

/// [`robust_prune`], saying as well whether the coincident fill was used.
pub(crate) fn prune<S: VectorStore>(
    store: &S,
    point: u32,
    candidates: Vec<OrderedNode>,
    alpha: f32,
    max_degree: usize,
    comparisons: &Comparisons,
) -> Result<Pruned> {
    validate_alpha(alpha)?;
    if max_degree == 0 {
        return Err(Error::invalid_input(
            "Vamana max_degree must be greater than zero".to_string(),
        ));
    }
    // An id past the store would slice the vector buffer out of bounds, and a
    // non-finite distance makes every comparison false, so the sweep would
    // drop candidate after candidate and leave the vertex a single edge.
    let store_len = store.len();
    if point as usize >= store_len {
        return Err(Error::invalid_input(format!(
            "Vamana vertex {point} is outside a store of {store_len} vectors"
        )));
    }
    for candidate in &candidates {
        if candidate.id as usize >= store_len {
            return Err(Error::invalid_input(format!(
                "Vamana candidate {} is outside a store of {store_len} vectors",
                candidate.id
            )));
        }
        if !candidate.dist.0.is_finite() {
            return Err(Error::invalid_input(format!(
                "Vamana candidate {} has distance {}, which no comparison can order; the vectors \
                 are most likely not finite",
                candidate.id, candidate.dist.0
            )));
        }
    }

    let mut pool = candidates;
    pool.retain(|candidate| candidate.id != point);
    // Pinned at zero: under cosine, `1 - dot` in f32 lands a few ULPs below
    // zero for a vector against itself, and a negative distance inverts the
    // rule, since `alpha > 1` then lowers the left-hand side.
    for candidate in &mut pool {
        candidate.dist = OrderedFloat(candidate.dist.0.max(0.0));
    }
    // Deduplicated by id before ordering by distance: the same id twice at two
    // distances would not be adjacent under a distance order.
    pool.sort_unstable_by(|a, b| a.id.cmp(&b.id).then(a.dist.cmp(&b.dist)));
    pool.dedup_by_key(|candidate| candidate.id);
    pool.sort_unstable_by(|a, b| a.dist.cmp(&b.dist).then(a.id.cmp(&b.id)));
    let mut pool = VecDeque::from(pool);

    let mut selected = Vec::with_capacity(max_degree);
    // Candidates exactly on top of a pick. The rule occludes them at every
    // alpha, since `alpha * 0 > 0` is false, which would turn a partition of
    // duplicates into a chain; they are the same point, not a worse one.
    let mut coincident: Vec<OrderedNode> = Vec::new();
    while let Some(nearest) = pool.pop_front() {
        selected.push(nearest.id);
        if selected.len() == max_degree {
            break;
        }
        let from_nearest = store.dist_calculator_from_id(nearest.id);
        comparisons.record(pool.len() as u64);
        pool.retain(|candidate| {
            let separation = from_nearest.distance(candidate.id).max(0.0);
            if alpha * separation > candidate.dist.0 {
                return true;
            }
            if separation == 0.0 {
                coincident.push(candidate.clone());
            }
            false
        });
    }

    // Filling the leftover slots with candidates occluded at a non-zero
    // separation was measured on SIFT1M and rejected: 3-5% more distances per
    // query at equal recall, and 7x the distances to build.
    let mut has_coincident_fill = false;
    if selected.len() < max_degree {
        coincident.sort_unstable_by(|a, b| a.dist.cmp(&b.dist).then(a.id.cmp(&b.id)));
        for candidate in coincident {
            if selected.len() == max_degree {
                break;
            }
            selected.push(candidate.id);
            has_coincident_fill = true;
        }
    }
    Ok(Pruned {
        neighbors: selected,
        has_coincident_fill,
    })
}

/// What [`admit`] decided about a newcomer to a full list.
#[derive(Debug, PartialEq)]
pub(crate) enum Admission {
    /// The prune would return the list as it stands.
    Refused,
    /// The list the prune would return.
    Admitted(Vec<u32>),
    /// A zero separation or a distance the prune would refuse: left to it.
    Undecided,
}

/// [`prune`] over a full list and one newcomer, when the list is itself what a
/// prune without a coincident fill returned at `alpha` or a smaller one.
///
/// Over such a list every member closer than the newcomer is picked again, the
/// newcomer survives only if none of them occludes it, and a farther member
/// drops only if the newcomer occludes it: no member occludes another. That
/// takes about `2 R` distances where the prune spends about `R^2 / 2`, and the
/// answer is the prune's bit for bit. Back-edge re-prunes were 57% of a GloVe
/// build before this.
pub(crate) fn admit<S: VectorStore>(
    store: &S,
    owner: u32,
    members: &[u32],
    newcomer: u32,
    alpha: f32,
    max_degree: usize,
    comparisons: &Comparisons,
) -> Result<Admission> {
    debug_assert_eq!(members.len(), max_degree, "vertex {owner} is not full");
    let from_owner = store.dist_calculator_from_id(owner);
    comparisons.record(members.len() as u64 + 1);
    let distances = members
        .iter()
        .map(|member| from_owner.distance(*member))
        .collect::<Vec<_>>();
    let newcomer_distance = from_owner.distance(newcomer);
    if !newcomer_distance.is_finite() || distances.iter().any(|distance| !distance.is_finite()) {
        return Ok(Admission::Undecided);
    }
    let key = |distance: f32, id: u32| (OrderedFloat(distance.max(0.0)), id);
    let newcomer_key = key(newcomer_distance, newcomer);
    debug_assert!(
        distances
            .windows(2)
            .zip(members.windows(2))
            .all(|(pair, ids)| key(pair[0], ids[0]) < key(pair[1], ids[1])),
        "vertex {owner}: a list taken for a prune's output is not in prune order"
    );

    let place = members
        .iter()
        .zip(&distances)
        .position(|(member, distance)| key(*distance, *member) > newcomer_key)
        .unwrap_or(members.len());
    if place >= max_degree {
        return Ok(Admission::Refused);
    }
    for member in &members[..place] {
        comparisons.record(1);
        // One distance from each member: over product codes a calculator
        // would first copy a whole distance table for it.
        let separation = store.dist_between(*member, newcomer).max(0.0);
        if alpha * separation > newcomer_key.0.0 {
            continue;
        }
        return Ok(if separation == 0.0 {
            Admission::Undecided
        } else {
            Admission::Refused
        });
    }

    let mut admitted = Vec::with_capacity(max_degree);
    admitted.extend_from_slice(&members[..place]);
    admitted.push(newcomer);
    let from_newcomer = store.dist_calculator_from_id(newcomer);
    for (member, distance) in members[place..].iter().zip(&distances[place..]) {
        if admitted.len() == max_degree {
            break;
        }
        comparisons.record(1);
        let separation = from_newcomer.distance(*member).max(0.0);
        if alpha * separation > distance.max(0.0) {
            admitted.push(*member);
        } else if separation == 0.0 {
            return Ok(Admission::Undecided);
        }
    }
    Ok(Admission::Admitted(admitted))
}

/// Refuse a pruning slack the graph could not survive. Infinity is refused as
/// well as NaN: `serde_json` writes either as `null`, which reads back as an
/// error, so the index could be written once and never opened again.
pub(crate) fn validate_alpha(alpha: f32) -> Result<()> {
    if !alpha.is_finite() || alpha < 1.0 {
        return Err(Error::invalid_input(format!(
            "Vamana alpha must be a finite value of at least 1.0, got {alpha}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use lance_linalg::distance::DistanceType;
    use lance_linalg::kernels::normalize_fsl;
    use rstest::rstest;

    use super::*;
    use crate::vector::flat::storage::FlatFloatStorage;
    use crate::vector::vamana::graph::{Adjacency, MAX_DEGREE, reachable};
    use crate::vector::vamana::testing::{
        Store, flat_storage, flat_storage_with_row_ids, pq_storage, scattered_values, sq_storage,
        vectors_of, walk_recall_at_ten,
    };

    fn storage_of(
        values: Vec<f32>,
        dimension: usize,
        distance_type: DistanceType,
    ) -> FlatFloatStorage {
        flat_storage(values, dimension, distance_type)
    }

    fn scattered_storage(num_vertices: usize, dimension: usize) -> FlatFloatStorage {
        flat_storage(
            scattered_values(num_vertices * dimension),
            dimension,
            DistanceType::L2,
        )
    }

    fn line_storage(num_vertices: usize) -> FlatFloatStorage {
        storage_of(
            (0..num_vertices).map(|i| i as f32).collect(),
            1,
            DistanceType::L2,
        )
    }

    fn small_params() -> VamanaBuildParams {
        VamanaBuildParams {
            max_degree: 16,
            search_list_size: 32,
            alpha: 1.2,
        }
    }

    fn build(storage: &FlatFloatStorage, params: &VamanaBuildParams) -> BuiltPartition {
        build_partition(storage, params, 42, &Comparisons::default()).unwrap()
    }

    /// Batches small enough that a fixture of a few hundred vertices is cut
    /// into many, each searching a graph its predecessors left.
    const BATCHED: Sweep = Sweep::Batched {
        max_batch: NonZeroUsize::new(16).unwrap(),
    };

    fn build_by(
        storage: &FlatFloatStorage,
        params: &VamanaBuildParams,
        sweep: Sweep,
    ) -> Result<BuiltPartition> {
        build_partition_by(storage, params, 42, sweep, &Comparisons::default())
    }

    fn all_candidates(storage: &FlatFloatStorage, point: u32) -> Vec<OrderedNode> {
        let calculator = storage.dist_calculator_from_id(point);
        (0..storage.len() as u32)
            .map(|id| OrderedNode::new(id, OrderedFloat(calculator.distance(id))))
            .collect()
    }

    /// Lance's own diversity heuristic, walking candidates nearest first.
    fn lance_selection(
        storage: &FlatFloatStorage,
        point: u32,
        candidates: &[OrderedNode],
        max_degree: usize,
    ) -> Vec<u32> {
        let mut sorted = candidates.to_vec();
        sorted.retain(|candidate| candidate.id != point);
        sorted.sort_unstable_by(|a, b| a.dist.cmp(&b.dist).then(a.id.cmp(&b.id)));
        let mut selected: Vec<OrderedNode> = Vec::with_capacity(max_degree);
        for candidate in sorted {
            if selected.len() == max_degree {
                break;
            }
            if storage.prefers_candidate(&candidate, &selected) {
                selected.push(candidate);
            }
        }
        selected.into_iter().map(|node| node.id).collect()
    }

    #[test]
    fn alpha_one_reproduces_lance_diversity_exactly() {
        let storage = scattered_storage(200, 8);
        for point in [0u32, 7, 63, 199] {
            let candidates = all_candidates(&storage, point);
            let ours = robust_prune(
                &storage,
                point,
                candidates.clone(),
                1.0,
                16,
                &Comparisons::default(),
            )
            .unwrap();
            assert_eq!(
                ours,
                lance_selection(&storage, point, &candidates, 16),
                "vertex {point}"
            );
        }
    }

    /// With enough slack nothing is dropped, which pins the direction of the
    /// alpha test: an inverted comparison would drop everything instead.
    #[test]
    fn unbounded_alpha_selects_the_nearest_candidates() {
        let storage = scattered_storage(64, 4);
        let candidates = all_candidates(&storage, 0);
        let mut nearest = candidates.clone();
        nearest.retain(|candidate| candidate.id != 0);
        nearest.sort_unstable_by(|a, b| a.dist.cmp(&b.dist).then(a.id.cmp(&b.id)));
        let nearest = nearest
            .iter()
            .take(8)
            .map(|node| node.id)
            .collect::<Vec<_>>();
        let selected = robust_prune(
            &storage,
            0,
            candidates,
            f32::MAX,
            8,
            &Comparisons::default(),
        )
        .unwrap();
        assert_eq!(selected, nearest);

        let all = all_candidates(&storage, 0);
        let diverse = robust_prune(&storage, 0, all, 1.0, 64, &Comparisons::default()).unwrap();
        assert!(diverse.len() < 63, "alpha 1 pruned nothing");
    }

    #[test]
    fn the_point_and_its_duplicates_never_become_edges() {
        let storage = scattered_storage(32, 4);
        let mut candidates = all_candidates(&storage, 5);
        candidates.extend(all_candidates(&storage, 5));
        let selected =
            robust_prune(&storage, 5, candidates, 1.2, 16, &Comparisons::default()).unwrap();
        assert!(!selected.contains(&5));
        assert_eq!(
            selected.iter().collect::<HashSet<_>>().len(),
            selected.len()
        );
        assert!(selected.len() <= 16);
    }

    /// Under cosine a vector measured against itself can land a few ULPs
    /// below zero. Read as zero, the two candidates on top of the point tie,
    /// the first by id is picked and the other is occluded at zero
    /// separation, which leaves the far one a slot. Read as they are, the
    /// more negative one is picked, the other passes `alpha * 0 > -1e-7`,
    /// and the far candidate loses its slot.
    #[test]
    fn a_distance_just_below_zero_is_read_as_zero() {
        // The point, two vectors on top of it and one far away.
        let storage = storage_of(
            vec![0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
            2,
            DistanceType::L2,
        );
        let candidates = vec![
            OrderedNode::new(1, OrderedFloat(-1e-7)),
            OrderedNode::new(2, OrderedFloat(100.0)),
            OrderedNode::new(3, OrderedFloat(-2e-7)),
        ];
        let selected =
            robust_prune(&storage, 0, candidates, 1.2, 2, &Comparisons::default()).unwrap();
        assert_eq!(selected, vec![1, 2]);
    }

    #[test]
    fn invalid_prune_inputs_are_rejected() {
        let storage = scattered_storage(8, 2);
        let candidates = all_candidates(&storage, 0);
        assert!(
            robust_prune(
                &storage,
                0,
                candidates.clone(),
                0.9,
                4,
                &Comparisons::default()
            )
            .is_err()
        );
        assert!(
            robust_prune(
                &storage,
                0,
                candidates.clone(),
                1.2,
                0,
                &Comparisons::default()
            )
            .is_err()
        );
        assert!(
            robust_prune(
                &storage,
                8,
                candidates.clone(),
                1.2,
                4,
                &Comparisons::default()
            )
            .is_err()
        );
        let mut outside = candidates.clone();
        outside.push(OrderedNode::new(8, OrderedFloat(1.0)));
        assert!(robust_prune(&storage, 0, outside, 1.2, 4, &Comparisons::default()).is_err());
        let mut not_finite = candidates;
        not_finite.push(OrderedNode::new(3, OrderedFloat(f32::NAN)));
        assert!(robust_prune(&storage, 0, not_finite, 1.2, 4, &Comparisons::default()).is_err());
    }

    #[test]
    fn a_built_graph_holds_its_shape_and_is_reachable() {
        let storage = scattered_storage(500, 8);
        let params = small_params();
        let built = build(&storage, &params);
        assert_eq!(built.graph.row_ids(), (0..500).collect::<Vec<u64>>());
        for vertex in 0..500u32 {
            let neighbors = built.graph.neighbors(vertex);
            assert!(!neighbors.is_empty(), "vertex {vertex} is a dead end");
            assert!(neighbors.len() <= params.max_degree as usize);
        }
        assert_eq!(reachable(&built.graph, built.medoid, |_| true), 500);
    }

    #[test]
    fn the_graph_names_the_stores_row_ids() {
        let row_ids = (0..40).map(|i| 1000 + 7 * i).collect::<Vec<u64>>();
        let storage = flat_storage_with_row_ids(
            vectors_of(scattered_values(40 * 4), 4),
            row_ids.clone(),
            DistanceType::L2,
        );
        let built = build(&storage, &small_params());
        assert_eq!(built.graph.row_ids(), row_ids);
    }

    /// Recall against brute force on queries outside the index, with the walk
    /// reading less than a quarter of the partition.
    #[rstest]
    #[case::l2(DistanceType::L2)]
    #[case::cosine(DistanceType::Cosine)]
    fn recall_at_ten_beats_the_bar(#[case] distance_type: DistanceType) {
        const VERTICES: usize = 1200;
        const QUERIES: usize = 50;
        let values = scattered_values((VERTICES + QUERIES) * 8)
            .into_iter()
            .map(|value| value - 0.5)
            .collect::<Vec<_>>();
        let (indexed, queries) = values.split_at(VERTICES * 8);
        let indexed = storage_of(indexed.to_vec(), 8, distance_type);
        let params = small_params();
        let built = build(&indexed, &params);

        let (recall, walked) = walk_recall_at_ten(
            &indexed,
            &built.graph,
            built.medoid,
            queries,
            8,
            params.search_list_size,
        );
        let scanned = walked as f64 / QUERIES as f64;
        assert!(
            recall >= 0.95,
            "{distance_type}: recall@10 fell to {recall:.4}"
        );
        assert!(
            scanned < 0.25 * VERTICES as f64,
            "{distance_type}: the walk measured {scanned:.1} of {VERTICES} vectors"
        );
    }

    /// Duplicates occlude each other at zero separation at every alpha; the
    /// coincident fill keeps their slots filled. Under cosine `d(x, x)` rounds
    /// either side of zero, which only the pin at zero keeps from inverting the
    /// rule.
    #[rstest]
    #[case::l2(DistanceType::L2)]
    #[case::cosine(DistanceType::Cosine)]
    fn a_partition_of_duplicates_keeps_its_edges(
        #[case] distance_type: DistanceType,
        #[values(Sweep::Sequential, BATCHED)] sweep: Sweep,
    ) {
        const VERTICES: usize = 400;
        const DIMENSION: usize = 8;
        let params = small_params();
        for distinct in [1usize, 2, 4] {
            let values = Float32Array::from(
                (0..VERTICES)
                    .flat_map(|vertex| {
                        (0..DIMENSION)
                            .map(move |axis| ((vertex % distinct) * DIMENSION + axis) as f32)
                    })
                    .collect::<Vec<_>>(),
            );
            let vectors = vectors_of(values.values().to_vec(), DIMENSION);
            let vectors = if distance_type == DistanceType::Cosine {
                normalize_fsl(&vectors).unwrap()
            } else {
                vectors
            };
            let built = build_by(
                &FlatFloatStorage::new(vectors, distance_type),
                &params,
                sweep,
            )
            .unwrap();
            for vertex in 0..VERTICES as u32 {
                assert_eq!(
                    built.graph.neighbors(vertex).len(),
                    params.max_degree as usize,
                    "{distance_type}, distinct={distinct}: vertex {vertex} was left short"
                );
            }
            assert!(reachable(&built.graph, built.medoid, |_| true) > params.max_degree as usize);
        }
    }

    #[rstest]
    fn the_same_seed_builds_the_same_graph(#[values(Sweep::Sequential, BATCHED)] sweep: Sweep) {
        let storage = scattered_storage(300, 6);
        let params = small_params();
        let first = build_by(&storage, &params, sweep).unwrap();
        let second = build_by(&storage, &params, sweep).unwrap();
        assert_eq!(first.graph, second.graph);
        assert_eq!(first.medoid, second.medoid);
        let other =
            build_partition_by(&storage, &params, 43, sweep, &Comparisons::default()).unwrap();
        assert_ne!(
            other.graph, first.graph,
            "the seed does not reach the build"
        );

        let plain = build_by(
            &storage,
            &VamanaBuildParams {
                alpha: 1.0,
                ..params
            },
            sweep,
        )
        .unwrap();
        assert_ne!(plain.graph, first.graph, "alpha does not reach the build");
    }

    #[rstest]
    fn a_partition_smaller_than_the_degree_bound_builds(
        #[values(Sweep::Sequential, BATCHED)] sweep: Sweep,
    ) {
        for vertices in [1usize, 2, 3, 17] {
            let built = build_by(&scattered_storage(vertices, 4), &small_params(), sweep).unwrap();
            assert_eq!(built.graph.len(), vertices);
            assert_eq!(reachable(&built.graph, built.medoid, |_| true), vertices);
        }
        assert!(build_by(&scattered_storage(0, 4), &small_params(), sweep).is_err());
        let wide = VamanaBuildParams {
            max_degree: MAX_DEGREE,
            ..small_params()
        };
        build_by(&scattered_storage(8, 2), &wide, sweep).unwrap();
        let too_wide = VamanaBuildParams {
            max_degree: MAX_DEGREE + 1,
            ..small_params()
        };
        assert!(build_by(&scattered_storage(8, 2), &too_wide, sweep).is_err());
    }

    #[rstest]
    fn a_build_over_non_finite_vectors_stops(#[values(Sweep::Sequential, BATCHED)] sweep: Sweep) {
        let mut values = scattered_values(20 * 2);
        values[7] = f32::NAN;
        let storage = storage_of(values, 2, DistanceType::L2);
        let error = build_by(&storage, &small_params(), sweep).unwrap_err();
        assert!(
            error.to_string().contains("no comparison can order"),
            "{error}"
        );
    }

    /// The medoid by its definition, the summed distance to every vertex.
    fn exhaustive_medoid(storage: &FlatFloatStorage) -> u32 {
        let summed = |point: u32| {
            let from = storage.dist_calculator_from_id(point);
            (0..storage.len() as u32)
                .map(|other| from.distance(other) as f64)
                .sum::<f64>()
        };
        (0..storage.len() as u32)
            .min_by(|left, right| summed(*left).total_cmp(&summed(*right)))
            .unwrap()
    }

    #[test]
    fn the_medoid_minimises_the_summed_distance() {
        let mut outlier = scattered_values(64 * 4);
        outlier.extend(std::iter::repeat_n(100.0, 4));
        for (storage, what) in [
            (scattered_storage(200, 4), "a scattered cloud"),
            (line_storage(41), "a line"),
            (
                storage_of(outlier, 4, DistanceType::L2),
                "a cloud with a far outlier",
            ),
        ] {
            let chosen = medoid(&storage, 42, &Comparisons::default()).unwrap();
            assert_eq!(chosen, exhaustive_medoid(&storage), "{what}");
        }
        assert_eq!(
            medoid(&line_storage(11), 42, &Comparisons::default()).unwrap(),
            5
        );
    }

    #[test]
    fn a_store_of_codes_takes_the_medoid_of_a_sample() {
        // Codes have no centroid to measure from; on a line the sampled medoid
        // still lands in the middle.
        let vectors = vectors_of((0..100).map(|i| i as f32).collect(), 1);
        let storage = sq_storage(&vectors, (0..100).collect(), DistanceType::L2);
        let chosen = medoid(&storage, 42, &Comparisons::default()).unwrap();
        assert!((45..=54).contains(&chosen), "medoid {chosen}");
    }

    /// A build that re-prunes every full list rather than admitting into one:
    /// the reference the shortcut must reproduce exactly.
    fn build_with_full_reprunes<S: VectorStore>(
        storage: &S,
        params: &VamanaBuildParams,
        comparisons: &Comparisons,
    ) -> PartitionGraph {
        let num_vertices = storage.len() as u32;
        let mut rng = SmallRng::seed_from_u64(42);
        let row_ids = (0..num_vertices).map(|id| storage.row_id(id)).collect();
        let mut graph = PartitionGraph::edgeless(params.max_degree, row_ids).unwrap();
        randomize(&mut graph, &mut rng).unwrap();
        let medoid = medoid(storage, 42, comparisons).unwrap();
        let mut scratch = InsertScratch::new(num_vertices as usize, params.max_degree);
        let mut order = (0..num_vertices).collect::<Vec<_>>();
        for alpha in [1.0, params.alpha] {
            let linking = Linking {
                alpha,
                search_list_size: params.search_list_size,
                vectors: None,
            };
            order.shuffle(&mut rng);
            for point in &order {
                scratch.forget_prune_outputs();
                insert_point(
                    &mut graph,
                    storage,
                    &mut scratch,
                    &linking,
                    *point,
                    medoid,
                    comparisons,
                )
                .unwrap();
            }
        }
        graph
    }

    fn assert_pair_distances_are_the_calculators<S: VectorStore>(storage: &S) {
        for from in 0..storage.len() as u32 {
            let calculator = storage.dist_calculator_from_id(from);
            for to in 0..storage.len() as u32 {
                assert_eq!(
                    storage.dist_between(from, to).to_bits(),
                    calculator.distance(to).to_bits(),
                    "{from} -> {to}"
                );
            }
        }
    }

    /// `admit` measures each separation with `dist_between`, the prune with a
    /// calculator; over every store a graph is built on, they must give the
    /// same distance to the bit for the shortcut to reproduce the prune.
    #[rstest]
    #[case::flat(Store::Flat(scattered_storage(100, 8)))]
    #[case::scalar_codes(Store::Scalar(sq_storage(
        &vectors_of(scattered_values(100 * 8), 8),
        (0..100).collect(),
        DistanceType::L2,
    )))]
    #[case::product_codes(Store::Product(pq_storage(&vectors_of(scattered_values(300 * 8), 8), 8)))]
    #[case::four_bit_product_codes(Store::Product(pq_storage(&vectors_of(scattered_values(100 * 8), 8), 4)))]
    fn a_pair_distance_is_the_calculators_to_the_bit(#[case] store: Store) {
        match &store {
            Store::Flat(storage) => assert_pair_distances_are_the_calculators(storage),
            Store::Scalar(storage) => assert_pair_distances_are_the_calculators(storage),
            Store::Product(storage) => assert_pair_distances_are_the_calculators(storage),
        }
    }

    fn assert_admitting_reproduces_repruning<S: VectorStore>(storage: &S, admits_cheaper: bool) {
        let params = VamanaBuildParams {
            max_degree: 8,
            search_list_size: 24,
            alpha: 1.2,
        };
        let admitting = Comparisons::default();
        let built = build_partition(storage, &params, 42, &admitting).unwrap();
        let repruning = Comparisons::default();
        assert_eq!(
            built.graph,
            build_with_full_reprunes(storage, &params, &repruning)
        );
        // On the lattice nearly every decision is a tie or a zero separation,
        // which `admit` leaves to the prune.
        assert!(
            admitting.get() < repruning.get()
                || !admits_cheaper && admitting.get() == repruning.get(),
            "admitting cost {} distances, re-pruning {}",
            admitting.get(),
            repruning.get()
        );
    }

    /// The shortcut reproduces the prune over every store a graph is built on.
    #[rstest]
    #[case::scattered(Store::Flat(scattered_storage(300, 6)), true)]
    #[case::lattice(Store::Flat(storage_of(
        (0..64).flat_map(|point| [point % 4, point / 4 % 4, point / 16].map(|axis| axis as f32)).collect(),
        3,
        DistanceType::L2,
    )), false)]
    #[case::cosine(Store::Flat({
        let vectors = vectors_of(scattered_values(300 * 6).into_iter().map(|v| v - 0.5).collect(), 6);
        FlatFloatStorage::new(normalize_fsl(&vectors).unwrap(), DistanceType::Cosine)
    }), true)]
    #[case::scalar_codes(Store::Scalar(sq_storage(
        &vectors_of(scattered_values(300 * 6), 6),
        (0..300).collect(),
        DistanceType::L2,
    )), true)]
    #[case::product_codes(Store::Product(pq_storage(&vectors_of(scattered_values(300 * 8), 8), 8)), true)]
    #[case::four_bit_product_codes(Store::Product(pq_storage(&vectors_of(scattered_values(300 * 8), 8), 4)), true)]
    fn a_build_equals_one_that_re_prunes_every_full_list(
        #[case] store: Store,
        #[case] admits_cheaper: bool,
    ) {
        match &store {
            Store::Flat(storage) => assert_admitting_reproduces_repruning(storage, admits_cheaper),
            Store::Scalar(storage) => {
                assert_admitting_reproduces_repruning(storage, admits_cheaper)
            }
            Store::Product(storage) => {
                assert_admitting_reproduces_repruning(storage, admits_cheaper)
            }
        }
    }
}
