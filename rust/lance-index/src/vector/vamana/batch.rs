// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Linking a sweep's vertices in batches, in parallel.
//!
//! ParlayANN's prefix doubling: a sweep's order is cut into batches of 1, 2,
//! 4, ... vertices, up to a cap. Every vertex of a batch searches and prunes
//! against the graph as it stood when the batch began, all of them at once.
//! Then every row the batch touched is rewritten on its own: a vertex of the
//! batch takes the out-edges it chose, and a vertex it chose takes its
//! back-edges in the batch's order, through the step an insertion takes them
//! through. A row is read and written by its own task alone, so the graph and
//! every distance count come out the same on any number of threads.
//!
//! The vertices of one batch do not see one another, which costs a little of
//! the graph's quality; the early batches are small so that the graph is
//! navigable before many vertices search it blind.

use std::num::NonZeroUsize;
use std::ops::Range;

use lance_core::{Error, Result};
use rayon::prelude::*;

use super::build::Pruned;
use super::graph::{PartitionGraph, RowMut};
use super::insert::{Linking, OpenList, choose_out_edges};
use super::search::Comparisons;
use super::walk::with_scratch;
use crate::vector::storage::VectorStore;

/// The batches a sweep of `len` vertices is cut into: 1, 2, 4, ... vertices,
/// never more than `max_batch`, the last one whatever remains.
pub fn batch_bounds(len: usize, max_batch: NonZeroUsize) -> impl Iterator<Item = Range<usize>> {
    let mut start = 0;
    let mut size = 1usize;
    std::iter::from_fn(move || {
        if start >= len {
            return None;
        }
        let end = start + (len - start).min(size).min(max_batch.get());
        let batch = start..end;
        start = end;
        size = size.saturating_mul(2);
        Some(batch)
    })
}

/// How a sweep is cut: the vertices in the order they are linked, the vertex
/// every search starts from, and the most vertices that search one graph.
pub struct Batches<'a> {
    pub order: &'a [u32],
    pub entry_point: u32,
    pub max_batch: NonZeroUsize,
}

/// Link every vertex of `batches.order` into `graph`, batch by batch.
///
/// `is_prune_output` holds the flag an insertion keeps for each vertex (see
/// [`super::insert::InsertScratch`]) and, as there, must live across both
/// sweeps of a build, whose alpha only grows. On an error the graph and the
/// flags are left part of the way through a batch and must not be used.
///
/// The work runs on the current rayon pool, the global one unless the caller
/// installed another, as Lance's HNSW build does. The calling thread, often a
/// `spawn_cpu` closure's, waits for the pool's workers, and they never wait
/// on Lance's threads.
pub fn link_in_batches<S: VectorStore>(
    graph: &mut PartitionGraph,
    store: &S,
    is_prune_output: &mut [bool],
    linking: &Linking,
    batches: &Batches,
    comparisons: &Comparisons,
) -> Result<()> {
    let num_vertices = graph.len();
    if is_prune_output.len() != num_vertices {
        return Err(Error::invalid_input(format!(
            "Vamana batches need one prune-output flag per vertex, got {} for {num_vertices} \
             vertices",
            is_prune_output.len()
        )));
    }
    if let Some(vertex) = batches
        .order
        .iter()
        .find(|vertex| **vertex as usize >= num_vertices)
    {
        return Err(Error::invalid_input(format!(
            "Vamana cannot link vertex {vertex} into a graph of {num_vertices} vertices"
        )));
    }
    let mut buffers = Buffers::default();
    for batch in batch_bounds(batches.order.len(), batches.max_batch) {
        let measured = link_batch(
            graph,
            store,
            is_prune_output,
            linking,
            &batches.order[batch],
            batches.entry_point,
            &mut buffers,
        )?;
        comparisons.record(measured);
    }
    Ok(())
}

/// What every batch of a sweep fills anew, kept so that a batch reuses the
/// allocations of the one before.
#[derive(Default)]
struct Buffers {
    /// Each back-edge as its target over its source's position in the batch,
    /// so that one sort groups a target's back-edges in the batch's order.
    offers: Vec<u64>,
    /// The batch's vertices with their positions, in vertex order.
    members: Vec<(u32, u32)>,
    touched: Vec<Touched>,
    /// The vertex of each touched row.
    vertices: Vec<u32>,
    /// The distances each touched row's rewrite measured.
    measured: Vec<u64>,
}

/// A row one batch rewrites.
struct Touched {
    vertex: u32,
    /// The vertex's position in the batch, when it is one of the batch's.
    own: Option<u32>,
    /// The entries of the batch's sorted offers that name the vertex.
    offers: Range<usize>,
}

/// Link one batch, returning the distances it measured.
fn link_batch<S: VectorStore>(
    graph: &mut PartitionGraph,
    store: &S,
    is_prune_output: &mut [bool],
    linking: &Linking,
    batch: &[u32],
    entry_point: u32,
    buffers: &mut Buffers,
) -> Result<u64> {
    let frozen = &*graph;
    let num_vertices = frozen.len();
    let chosen = batch
        .par_iter()
        .map(|point| {
            let counted = Comparisons::default();
            let pruned = with_scratch(|search| {
                search.cover(num_vertices);
                choose_out_edges(
                    frozen,
                    store,
                    search,
                    linking,
                    *point,
                    entry_point,
                    &counted,
                )
            })?;
            Ok((pruned, counted.get()))
        })
        .collect::<Vec<Result<_>>>();
    // The first failure in the batch's order, not the first some thread met.
    let mut measured = 0u64;
    let mut own = Vec::with_capacity(batch.len());
    for result in chosen {
        let (pruned, counted) = result?;
        measured = measured.saturating_add(counted);
        own.push(pruned);
    }

    buffers.offers.clear();
    buffers.offers.par_extend(
        own.par_iter()
            .enumerate()
            .flat_map_iter(|(position, pruned)| {
                pruned
                    .neighbors
                    .iter()
                    .map(move |target| (u64::from(*target) << 32) | position as u64)
            }),
    );
    buffers.offers.par_sort_unstable();
    touch_rows(batch, buffers);

    let max_degree = graph.max_degree();
    let rows = graph.rows_mut(&buffers.vertices)?;
    let flags = entries_mut(is_prune_output, &buffers.vertices)?;
    buffers.measured.clear();
    buffers.measured.resize(buffers.touched.len(), 0);
    let rewrite = Rewrite {
        store,
        alpha: linking.alpha,
        max_degree,
        batch,
        own: &own,
        offers: &buffers.offers,
    };
    let failure = buffers
        .touched
        .par_iter()
        .zip(rows)
        .zip(flags)
        .zip(buffers.measured.par_iter_mut())
        .map_init(
            || OpenList::new(rewrite.max_degree),
            |list, (((touched, mut row), is_prune_output), measured)| {
                let counted = Comparisons::default();
                let rewritten = rewrite.row(list, touched, &mut row, is_prune_output, &counted);
                *measured = counted.get();
                rewritten.err()
            },
        )
        // The failure of the lowest row, not the first some thread met.
        .find_map_first(|failure| failure);
    match failure {
        Some(error) => Err(error),
        None => Ok(buffers
            .measured
            .iter()
            .fold(measured, |sum, counted| sum.saturating_add(*counted))),
    }
}

/// What every row a batch touched is rewritten from.
struct Rewrite<'a, S> {
    store: &'a S,
    alpha: f32,
    max_degree: u32,
    batch: &'a [u32],
    /// The out-edges each vertex of the batch chose, by position.
    own: &'a [Pruned],
    offers: &'a [u64],
}

impl<S: VectorStore> Rewrite<'_, S> {
    /// Give a row its own list when its vertex is one of the batch's, then its
    /// back-edges in the batch's order, keeping its flag as an insertion does.
    fn row(
        &self,
        list: &mut OpenList,
        touched: &Touched,
        row: &mut RowMut,
        is_prune_output: &mut bool,
        comparisons: &Comparisons,
    ) -> Result<()> {
        if let Some(position) = touched.own {
            let own = &self.own[position as usize];
            row.set(&own.neighbors)?;
            *is_prune_output = !own.has_coincident_fill;
        }
        if touched.offers.is_empty() {
            return Ok(());
        }
        list.open(touched.vertex, row.neighbors(), *is_prune_output);
        let mut is_changed = false;
        for offer in &self.offers[touched.offers.clone()] {
            let source = self.batch[(offer & u64::from(u32::MAX)) as usize];
            is_changed |= list
                .offer(
                    self.store,
                    source,
                    self.alpha,
                    self.max_degree as usize,
                    comparisons,
                )?
                .changed();
        }
        if is_changed {
            row.set(list.edges())?;
        }
        *is_prune_output = list.is_prune_output();
        Ok(())
    }
}

/// Fill `buffers.touched` and `buffers.vertices` with every row a batch
/// rewrites, in ascending vertex order: the batch's own vertices and the
/// targets of its sorted offers.
fn touch_rows(batch: &[u32], buffers: &mut Buffers) {
    let Buffers {
        offers,
        members,
        touched,
        vertices,
        ..
    } = buffers;
    members.clear();
    members.extend(batch.iter().copied().zip(0u32..));
    members.sort_unstable();
    touched.clear();
    vertices.clear();
    let mut members = members.iter().peekable();
    let mut start = 0;
    loop {
        let target = offers.get(start).map(|offer| (offer >> 32) as u32);
        let member = members.peek().map(|(vertex, _)| *vertex);
        let Some(vertex) = target.into_iter().chain(member).min() else {
            break;
        };
        let own = if member == Some(vertex) {
            members.next().map(|(_, position)| *position)
        } else {
            None
        };
        // A target draws a couple of offers on average: a linear walk beats a
        // search over the rest of the batch's offers.
        let mut end = start;
        while offers
            .get(end)
            .is_some_and(|offer| (offer >> 32) as u32 == vertex)
        {
            end += 1;
        }
        touched.push(Touched {
            vertex,
            own,
            offers: start..end,
        });
        vertices.push(vertex);
        start = end;
    }
}

/// The entries of `values` at `positions`, which must ascend strictly, each
/// writable apart from the others.
fn entries_mut<'a, T>(mut values: &'a mut [T], positions: &[u32]) -> Result<Vec<&'a mut T>> {
    let mut entries = Vec::with_capacity(positions.len());
    let mut first_in_values = 0usize;
    for position in positions {
        let entry_and_rest = (*position as usize)
            .checked_sub(first_in_values)
            .and_then(|skip| std::mem::take(&mut values).get_mut(skip..))
            .and_then(|from_entry| from_entry.split_first_mut());
        let Some((entry, rest)) = entry_and_rest else {
            return Err(Error::invalid_input(format!(
                "Vamana entries must be taken in strictly ascending order within the \
                 values, got position {position}"
            )));
        };
        entries.push(entry);
        values = rest;
        first_in_values = *position as usize + 1;
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use lance_linalg::distance::DistanceType;
    use lance_linalg::kernels::normalize_fsl;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use rand::seq::SliceRandom;
    use rstest::rstest;

    use super::*;
    use crate::vector::flat::storage::FlatFloatStorage;
    use crate::vector::vamana::build::{
        BATCHED_FROM_VERTICES, Sweep, build_partition_by, medoid, randomize,
    };
    use crate::vector::vamana::graph::Adjacency;
    use crate::vector::vamana::insert::Offered;
    use crate::vector::vamana::params::VamanaBuildParams;
    use crate::vector::vamana::search::SearchScratch;
    use crate::vector::vamana::testing::{
        Store, flat_storage, graph_fingerprint, pq_storage, scattered_values, sq_storage,
        vectors_of, walk_recall_at_ten,
    };

    const SEED: u64 = 42;

    /// Batches cut a fixture of a few hundred vertices into a dozen or so.
    const MAX_BATCH: usize = 64;

    fn tight_params() -> VamanaBuildParams {
        VamanaBuildParams {
            max_degree: 8,
            search_list_size: 24,
            alpha: 1.2,
        }
    }

    fn batched(max_batch: usize) -> Sweep {
        Sweep::Batched {
            max_batch: NonZeroUsize::new(max_batch).unwrap(),
        }
    }

    /// A build's graph, its medoid and the distances it measured.
    fn build<S: VectorStore>(
        store: &S,
        params: &VamanaBuildParams,
        sweep: Sweep,
    ) -> (PartitionGraph, u32, u64) {
        let comparisons = Comparisons::default();
        let built = build_partition_by(store, params, SEED, sweep, &comparisons).unwrap();
        (built.graph, built.medoid, comparisons.get())
    }

    fn centred(count: usize) -> Vec<f32> {
        scattered_values(count)
            .into_iter()
            .map(|value| value - 0.5)
            .collect()
    }

    fn flat_l2(vertices: usize, dimension: usize) -> Store {
        Store::Flat(flat_storage(
            centred(vertices * dimension),
            dimension,
            DistanceType::L2,
        ))
    }

    fn flat_cosine(vertices: usize, dimension: usize) -> Store {
        let vectors = normalize_fsl(&vectors_of(centred(vertices * dimension), dimension)).unwrap();
        Store::Flat(FlatFloatStorage::new(vectors, DistanceType::Cosine))
    }

    fn scalar_codes(vertices: usize, dimension: usize) -> Store {
        Store::Scalar(sq_storage(
            &vectors_of(scattered_values(vertices * dimension), dimension),
            (0..vertices as u64).collect(),
            DistanceType::L2,
        ))
    }

    /// Four-bit product codes, the first sixteen vectors for centroids.
    fn product_codes(vertices: usize) -> Store {
        Store::Product(pq_storage(
            &vectors_of(scattered_values(vertices * 8), 8),
            4,
        ))
    }

    /// The `{0..5}^3` lattice: nearly every decision a tie. Its prunes keep
    /// fewer than `R` edges, so a list fills by appends, and a full one is
    /// re-pruned rather than admitted into.
    fn lattice() -> Store {
        let values = (0..216)
            .flat_map(|point| [point % 6, point / 6 % 6, point / 36].map(|axis| axis as f32))
            .collect();
        Store::Flat(flat_storage(values, 3, DistanceType::L2))
    }

    /// Each of `distinct` points three times over, so prunes meet coincident
    /// candidates.
    fn triplicates(distinct: usize, dimension: usize) -> Store {
        let points = scattered_values(distinct * dimension);
        let values = (0..3).flat_map(|_| points.iter().copied()).collect();
        Store::Flat(flat_storage(values, dimension, DistanceType::L2))
    }

    /// A batch of one vertex is an insertion, so batches of one must give the
    /// vertex-at-a-time build to the byte and the distance: what ties the
    /// batching to every test of the sequential sweep.
    #[rstest]
    #[case::flat_l2(flat_l2(300, 6))]
    #[case::flat_cosine(flat_cosine(300, 6))]
    #[case::scalar_codes(scalar_codes(300, 6))]
    #[case::product_codes(product_codes(150))]
    fn batches_of_one_are_the_sequential_build(#[case] store: Store) {
        fn check<S: VectorStore>(store: &S) {
            let params = tight_params();
            assert_eq!(
                build(store, &params, batched(1)),
                build(store, &params, Sweep::Sequential)
            );
        }
        match &store {
            Store::Flat(storage) => check(storage),
            Store::Scalar(storage) => check(storage),
            Store::Product(storage) => check(storage),
        }
    }

    /// What the offers of a build in batches came to.
    #[derive(Debug, Default)]
    struct Coverage {
        /// Targets offered two or more back-edges in one batch, at least one
        /// of them while full.
        crowded_targets: usize,
        /// Targets that were themselves vertices of the batch that offered.
        member_targets: usize,
        /// Offers to a target an earlier offer of the same batch changed.
        offers_after_a_change: usize,
        admitted: usize,
        refused: usize,
        undecided: usize,
        repruned: usize,
        /// Own lists a prune completed with coincident candidates.
        coincident_fills: usize,
    }

    /// What one batch offered one target.
    #[derive(Default)]
    struct Offers {
        count: usize,
        any_full: bool,
        any_change: bool,
    }

    /// A build in batches written one step at a time: the schedule cut anew,
    /// every own list of a batch chosen against a copy of the graph taken when
    /// the batch began, then the back-edges offered vertex by vertex in the
    /// batch's order. With `forget_prune_outputs` every offer to a full list
    /// goes through the prune.
    fn build_plainly<S: VectorStore>(
        store: &S,
        params: &VamanaBuildParams,
        max_batch: usize,
        forget_prune_outputs: bool,
    ) -> ((PartitionGraph, u32, u64), Coverage) {
        let comparisons = Comparisons::default();
        let num_vertices = store.len() as u32;
        let max_degree = params.max_degree as usize;
        let mut rng = SmallRng::seed_from_u64(SEED);
        let row_ids = (0..num_vertices).map(|id| store.row_id(id)).collect();
        let mut graph = PartitionGraph::edgeless(params.max_degree, row_ids).unwrap();
        randomize(&mut graph, &mut rng).unwrap();
        let medoid = medoid(store, SEED, &comparisons).unwrap();
        let mut is_prune_output = vec![false; num_vertices as usize];
        let mut search = SearchScratch::new(num_vertices as usize);
        let mut list = OpenList::new(params.max_degree);
        let mut coverage = Coverage::default();
        let mut order = (0..num_vertices).collect::<Vec<_>>();
        for alpha in [1.0, params.alpha] {
            let linking = Linking {
                alpha,
                search_list_size: params.search_list_size,
                vectors: None,
            };
            order.shuffle(&mut rng);
            let (mut start, mut size) = (0, 1);
            while start < order.len() {
                let batch = &order[start..order.len().min(start + size.min(max_batch))];
                start += batch.len();
                size *= 2;
                let frozen = graph.clone();
                let chosen = batch
                    .iter()
                    .map(|point| {
                        choose_out_edges(
                            &frozen,
                            store,
                            &mut search,
                            &linking,
                            *point,
                            medoid,
                            &comparisons,
                        )
                        .unwrap()
                    })
                    .collect::<Vec<_>>();
                for (point, own) in batch.iter().zip(&chosen) {
                    graph.set_neighbors(*point, &own.neighbors).unwrap();
                    is_prune_output[*point as usize] = !own.has_coincident_fill;
                    coverage.coincident_fills += usize::from(own.has_coincident_fill);
                }
                let mut offered = HashMap::<u32, Offers>::new();
                for (point, own) in batch.iter().zip(&chosen) {
                    for target in &own.neighbors {
                        let flag = is_prune_output[*target as usize] && !forget_prune_outputs;
                        list.open(*target, graph.neighbors(*target), flag);
                        let was_full = list.edges().len() == max_degree;
                        let outcome = list
                            .offer(store, *point, alpha, max_degree, &comparisons)
                            .unwrap();
                        // Written whatever the outcome, so that what
                        // `Offered::changed` leaves unwritten is checked here
                        // rather than shared.
                        graph.set_neighbors(*target, list.edges()).unwrap();
                        is_prune_output[*target as usize] = list.is_prune_output();
                        match outcome {
                            Offered::Admitted => coverage.admitted += 1,
                            Offered::Refused => coverage.refused += 1,
                            Offered::Undecided => coverage.undecided += 1,
                            Offered::Repruned => coverage.repruned += 1,
                            Offered::Present | Offered::Appended => {}
                        }
                        let offers = offered.entry(*target).or_default();
                        offers.count += 1;
                        offers.any_full |= was_full;
                        coverage.offers_after_a_change += usize::from(offers.any_change);
                        offers.any_change |= outcome.changed();
                    }
                }
                coverage.crowded_targets += offered
                    .values()
                    .filter(|offers| offers.count >= 2 && offers.any_full)
                    .count();
                coverage.member_targets += offered
                    .keys()
                    .filter(|target| batch.contains(target))
                    .count();
            }
        }
        ((graph, medoid, comparisons.get()), coverage)
    }

    fn every_offer_was_made(coverage: &Coverage) -> bool {
        coverage.crowded_targets > 0
            && coverage.member_targets > 0
            && coverage.offers_after_a_change > 0
            && coverage.admitted > 0
            && coverage.refused > 0
    }

    fn ties_were_repruned(coverage: &Coverage) -> bool {
        coverage.crowded_targets > 0 && coverage.member_targets > 0 && coverage.repruned > 0
    }

    fn coincident_lists_were_made(coverage: &Coverage) -> bool {
        coverage.crowded_targets > 0 && coverage.coincident_fills > 0 && coverage.undecided > 0
    }

    /// The parallel batches give what their plain statement gives, to the
    /// byte and the distance, over fixtures where the kinds of offer the
    /// statement counts all occur.
    #[rstest]
    #[case::flat_l2(flat_l2(300, 6), every_offer_was_made as fn(&Coverage) -> bool)]
    #[case::flat_cosine(flat_cosine(300, 6), every_offer_was_made as fn(&Coverage) -> bool)]
    #[case::scalar_codes(scalar_codes(300, 6), every_offer_was_made as fn(&Coverage) -> bool)]
    #[case::product_codes(product_codes(300), every_offer_was_made as fn(&Coverage) -> bool)]
    #[case::lattice(lattice(), ties_were_repruned as fn(&Coverage) -> bool)]
    #[case::triplicates(triplicates(100, 4), coincident_lists_were_made as fn(&Coverage) -> bool)]
    fn batches_are_their_plain_statement(
        #[case] store: Store,
        #[case] premise: fn(&Coverage) -> bool,
    ) {
        fn check<S: VectorStore>(store: &S, premise: fn(&Coverage) -> bool) {
            let params = tight_params();
            let (plain, coverage) = build_plainly(store, &params, MAX_BATCH, false);
            assert!(
                premise(&coverage),
                "the fixture misses an offer: {coverage:?}"
            );
            assert_eq!(build(store, &params, batched(MAX_BATCH)), plain);
        }
        match &store {
            Store::Flat(storage) => check(storage, premise),
            Store::Scalar(storage) => check(storage, premise),
            Store::Product(storage) => check(storage, premise),
        }
    }

    /// Admitting into flagged lists inside batches decides what re-pruning
    /// every full list decides, for fewer distances where `admit` decides.
    /// Among coincident points it measures and then leaves the decision to the
    /// prune, whose list the batches must write all the same.
    #[rstest]
    #[case::flat_l2(flat_l2(300, 6), true)]
    #[case::flat_cosine(flat_cosine(300, 6), true)]
    #[case::scalar_codes(scalar_codes(300, 6), true)]
    #[case::product_codes(product_codes(300), true)]
    #[case::triplicates(triplicates(100, 4), false)]
    fn batches_admit_what_a_reprune_would_keep(#[case] store: Store, #[case] decides: bool) {
        fn check<S: VectorStore>(store: &S, decides: bool) {
            let params = tight_params();
            let (graph, medoid, admitting) = build(store, &params, batched(MAX_BATCH));
            let ((repruned, repruned_medoid, repruning), _) =
                build_plainly(store, &params, MAX_BATCH, true);
            assert_eq!(graph, repruned);
            assert_eq!(medoid, repruned_medoid);
            if decides {
                assert!(
                    admitting < repruning,
                    "admitting cost {admitting} distances, re-pruning {repruning}"
                );
            } else {
                let (_, coverage) = build_plainly(store, &params, MAX_BATCH, false);
                assert!(
                    coverage.undecided > 0,
                    "admit decided everything, so nothing was left to the prune: {coverage:?}"
                );
            }
        }
        match &store {
            Store::Flat(storage) => check(storage, decides),
            Store::Scalar(storage) => check(storage, decides),
            Store::Product(storage) => check(storage, decides),
        }
    }

    /// One thread or eight, and two builds sharing a pool, give the same
    /// graphs and distance counts; and the eight did search on more than one
    /// thread, which leaves each searching thread's walk scratch as long as
    /// the partition.
    #[test]
    fn the_thread_count_changes_nothing() {
        let flat = flat_storage(centred(600 * 8), 8, DistanceType::L2);
        let codes = sq_storage(
            &vectors_of(scattered_values(300 * 8), 8),
            (0..300).collect(),
            DistanceType::L2,
        );
        let params = tight_params();
        let sweep = batched(MAX_BATCH);
        let pool = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
        };
        let alone = pool(1).install(|| build(&flat, &params, sweep));
        let eight = pool(8);
        for _ in 0..2 {
            assert_eq!(eight.install(|| build(&flat, &params, sweep)), alone);
        }
        let searched = eight
            .broadcast(|_| with_scratch(|scratch| scratch.capacity() >= flat.len()))
            .into_iter()
            .filter(|searched| *searched)
            .count();
        assert!(
            searched > 1,
            "the eight-thread builds searched on {searched} thread(s)"
        );
        let codes_alone = pool(1).install(|| build(&codes, &params, sweep));
        let together = pool(8).install(|| {
            rayon::join(
                || build(&flat, &params, sweep),
                || build(&codes, &params, sweep),
            )
        });
        assert_eq!(together, (alone, codes_alone));
    }

    /// Batches cut at the production share of the partition cost a graph at
    /// most a point of recall@10 against the vertex-at-a-time build.
    #[rstest]
    #[case::l2(DistanceType::L2)]
    #[case::cosine(DistanceType::Cosine)]
    fn batches_keep_recall_within_a_point(#[case] distance_type: DistanceType) {
        const VERTICES: usize = 1200;
        const QUERIES: usize = 50;
        const DIMENSION: usize = 8;
        let values = centred((VERTICES + QUERIES) * DIMENSION);
        let (indexed, queries) = values.split_at(VERTICES * DIMENSION);
        let storage = flat_storage(indexed.to_vec(), DIMENSION, distance_type);
        let params = VamanaBuildParams {
            max_degree: 16,
            search_list_size: 32,
            alpha: 1.2,
        };
        let max_batch = VERTICES / 50;
        let (sequential, sequential_medoid, _) = build(&storage, &params, Sweep::Sequential);
        let (in_batches, medoid, _) = build(&storage, &params, batched(max_batch));
        assert!(
            batch_bounds(VERTICES, NonZeroUsize::new(max_batch).unwrap())
                .any(|batch| batch.len() > 1),
            "no batch holds more than one vertex"
        );
        assert_ne!(
            in_batches, sequential,
            "the batches made the sequential graph, so the comparison proves nothing"
        );
        let (baseline, _) = walk_recall_at_ten(
            &storage,
            &sequential,
            sequential_medoid,
            queries,
            DIMENSION,
            params.search_list_size,
        );
        let (recall, _) = walk_recall_at_ten(
            &storage,
            &in_batches,
            medoid,
            queries,
            DIMENSION,
            params.search_list_size,
        );
        assert!(
            recall >= 0.95 && recall >= baseline - 0.01,
            "{distance_type}: recall@10 {recall:.3} in batches, {baseline:.3} one at a time"
        );
    }

    /// Points with whole coordinates below 1000: every squared distance between
    /// two of them is a whole number below 2^24, which any order of summation
    /// measures exactly, so the graphs recorded below hold on any processor.
    fn integers(vertices: usize, dimension: usize) -> Vec<f32> {
        scattered_values(vertices * dimension)
            .into_iter()
            .map(|value| (value * 1000.0).floor())
            .collect()
    }

    /// Graphs, medoids and distance counts recorded from the sequential build,
    /// which this change left byte for byte as it was, and from the batches it
    /// adds. They pin what every other test shares with its reference, such as
    /// an `admit` that stops deciding: the graph stays the same, but every
    /// decision goes to the prune at the price of its distances.
    #[rstest]
    #[case::scattered_one_at_a_time(Sweep::Sequential, integers(300, 6), 6, (0xcf30_ba0f_a861_7631, 187, 107_827))]
    #[case::scattered_in_batches(batched(MAX_BATCH), integers(300, 6), 6, (0x316e_7b1c_1303_40e3, 187, 107_856))]
    #[case::triplicates_one_at_a_time(Sweep::Sequential, [integers(100, 4), integers(100, 4), integers(100, 4)].concat(), 4, (0x055a_a478_a2cd_d719, 34, 100_302))]
    #[case::triplicates_in_batches(batched(MAX_BATCH), [integers(100, 4), integers(100, 4), integers(100, 4)].concat(), 4, (0x6e59_0948_4691_9af9, 34, 102_978))]
    fn integer_fixtures_build_their_recorded_graphs(
        #[case] sweep: Sweep,
        #[case] values: Vec<f32>,
        #[case] dimension: usize,
        #[case] recorded: (u64, u32, u64),
    ) {
        let store = flat_storage(values, dimension, DistanceType::L2);
        let (graph, medoid, measured) = build(&store, &tight_params(), sweep);
        assert_eq!(
            (graph_fingerprint(&graph), medoid, measured),
            recorded,
            "fingerprint {:#x}, medoid {medoid}, distances {measured}",
            graph_fingerprint(&graph)
        );
    }

    fn doubling_to(cap: usize) -> Vec<usize> {
        std::iter::successors(Some(1), |size| Some(size * 2))
            .take_while(|size| *size < cap)
            .collect()
    }

    #[rstest]
    #[case::empty(0, 4, vec![])]
    #[case::one(1, 4, vec![1])]
    #[case::batches_of_one(5, 1, vec![1, 1, 1, 1, 1])]
    #[case::doubling(7, 8, vec![1, 2, 4])]
    #[case::remainder(10, 8, vec![1, 2, 4, 3])]
    #[case::capped(20, 4, vec![1, 2, 4, 4, 4, 4, 1])]
    #[case::recall_fixture(1200, 24, [doubling_to(24), vec![24; 48], vec![17]].concat())]
    #[case::threshold(100_000, 2000, [doubling_to(2000), vec![2000; 48], vec![1953]].concat())]
    fn batches_double_up_to_the_cap(
        #[case] len: usize,
        #[case] max_batch: usize,
        #[case] sizes: Vec<usize>,
    ) {
        let bounds = batch_bounds(len, NonZeroUsize::new(max_batch).unwrap()).collect::<Vec<_>>();
        assert_eq!(
            bounds.iter().map(|batch| batch.len()).collect::<Vec<_>>(),
            sizes
        );
        let mut next = 0;
        for batch in &bounds {
            assert_eq!(batch.start, next, "batches must tile the sweep");
            next = batch.end;
        }
        assert_eq!(next, len);
    }

    #[test]
    fn partitions_from_the_threshold_link_in_batches() {
        assert_eq!(Sweep::for_vertices(0), Sweep::Sequential);
        assert_eq!(
            Sweep::for_vertices(BATCHED_FROM_VERTICES - 1),
            Sweep::Sequential
        );
        assert_eq!(Sweep::for_vertices(BATCHED_FROM_VERTICES), batched(2000));
        assert_eq!(Sweep::for_vertices(1 << 20), batched((1 << 20) / 50));
    }

    #[test]
    fn touched_rows_are_the_batch_and_its_targets_in_order() {
        let batch = [7u32, 2, 9];
        // Position 0 (vertex 7) chose 2 and 4, position 2 (vertex 9) chose 4.
        let mut buffers = Buffers {
            offers: vec![2u64 << 32, 4 << 32, (4 << 32) | 2],
            ..Default::default()
        };
        touch_rows(&batch, &mut buffers);
        let rows = buffers
            .touched
            .iter()
            .map(|row| (row.vertex, row.own, row.offers.clone()))
            .collect::<Vec<_>>();
        assert_eq!(
            rows,
            vec![
                (2, Some(1), 0..1),
                (4, None, 1..3),
                (7, Some(0), 3..3),
                (9, Some(2), 3..3),
            ]
        );
        assert_eq!(buffers.vertices, vec![2, 4, 7, 9]);

        // A second batch reuses the buffers and leaves nothing of the first.
        buffers.offers = vec![1u64 << 32];
        touch_rows(&[5], &mut buffers);
        assert_eq!(buffers.vertices, vec![1, 5]);
        assert_eq!(buffers.touched.len(), 2);
    }

    #[test]
    fn entries_are_taken_apart_in_ascending_order() {
        let mut values = [0u8, 1, 2, 3, 4];
        let entries = entries_mut(&mut values, &[0, 2, 4]).unwrap();
        for entry in entries {
            *entry += 10;
        }
        assert_eq!(values, [10, 1, 12, 3, 14]);
        for positions in [&[2u32, 2][..], &[3, 1], &[5]] {
            let error = entries_mut(&mut values, positions).unwrap_err();
            assert!(
                matches!(error, Error::InvalidInput { .. })
                    && error.to_string().contains("strictly ascending"),
                "{positions:?}: {error}"
            );
        }
    }
}
