// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! A query's walk over one partition's graph.

use std::cell::RefCell;

use lance_core::{Error, Result};

use super::graph::Adjacency;
use super::search::{SearchList, SearchScratch, StopRule};
use crate::vector::graph::OrderedNode;
use crate::vector::storage::DistCalculator;

/// How many vertices one hop expands. Its neighbors are measured together,
/// which is what lets their data be asked for ahead of the distances.
pub const DEFAULT_BEAM_WIDTH: usize = 4;

/// How many neighbors ahead a walk asks for a code while it measures the
/// current one.
pub const DEFAULT_PREFETCH_AHEAD: usize = 2;

#[derive(Debug, Clone, Copy)]
pub struct WalkParams {
    /// `L`: the most candidates the walk keeps.
    pub search_list_size: usize,
    pub beam_width: usize,
    pub prefetch_ahead: usize,
    /// When the walk stops with candidates still unexpanded; `None` walks
    /// until nothing in the list is left to expand.
    pub stop: Option<StopRule>,
}

impl WalkParams {
    pub fn new(search_list_size: usize) -> Self {
        Self {
            search_list_size,
            beam_width: DEFAULT_BEAM_WIDTH,
            prefetch_ahead: DEFAULT_PREFETCH_AHEAD,
            stop: None,
        }
    }
}

/// The vertices a walk may return, kept apart from the vertices it steers by.
///
/// A walk that returned only admitted vertices from its own list would stop
/// short whenever the admitted ones are rare; steering by every vertex and
/// keeping the admitted ones beside the list walks exactly as an unfiltered
/// walk does.
pub struct Admission<'a> {
    /// How many admitted vertices to keep, nearest first.
    pub keep: usize,
    /// Whether the vertex at a position, at the measured distance, may be
    /// returned.
    pub accept: &'a mut dyn FnMut(u32, f32) -> bool,
}

#[derive(Debug)]
pub struct Walked {
    /// The walk's own list, nearest first.
    pub list: Vec<OrderedNode>,
    /// The admitted vertices, nearest first; empty unless the walk was given
    /// an [`Admission`].
    pub admitted: Vec<OrderedNode>,
    /// Distances measured.
    pub comparisons: u64,
}

/// Walk `graph` towards the query `calculator` measures.
///
/// The walk starts at the nearest of `entry_points`, or at `medoid` when there
/// are none, and expands `beam_width` vertices per hop. `scratch` may come
/// from any earlier walk.
pub fn walk(
    graph: &impl Adjacency,
    calculator: &impl DistCalculator,
    entry_points: &[u32],
    medoid: u32,
    params: &WalkParams,
    mut admission: Option<Admission<'_>>,
    scratch: &mut SearchScratch,
) -> Result<Walked> {
    let num_vertices = graph.num_vertices();
    if params.search_list_size == 0 || params.beam_width == 0 {
        return Err(Error::invalid_input(format!(
            "Vamana walk needs a search list and a beam of at least one, got {} and {}",
            params.search_list_size, params.beam_width
        )));
    }
    // The list squares `1 + margin`; past about 1.8e19 that overflows, and an
    // infinite factor times a zero distance makes its bar NaN.
    if let Some(stop) = params.stop
        && !(1 <= stop.rank
            && stop.rank <= stop.keep
            && stop.keep <= params.search_list_size
            && stop.margin >= 0.0
            && ((1.0 + stop.margin) * (1.0 + stop.margin)).is_finite())
    {
        return Err(Error::invalid_input(format!(
            "Vamana stop rule needs 1 <= rank <= keep <= search list size and a margin of at \
             least zero whose (1 + margin)^2 is finite, got rank {}, keep {}, search list size \
             {} and margin {}",
            stop.rank, stop.keep, params.search_list_size, stop.margin
        )));
    }
    if medoid as usize >= num_vertices
        || entry_points
            .iter()
            .any(|entry| *entry as usize >= num_vertices)
    {
        return Err(Error::index(format!(
            "Vamana walk was given a start outside a graph of {num_vertices} vertices"
        )));
    }

    scratch.cover(num_vertices);
    scratch.begin();
    let mut list = match params.stop {
        None => SearchList::new(params.search_list_size, num_vertices),
        Some(stop) => SearchList::with_margin(params.search_list_size, num_vertices, stop),
    };
    let mut admitted = admission
        .as_ref()
        .map(|admission| SearchList::new(admission.keep, num_vertices));

    let (start, distance, mut comparisons) =
        match nearest_entry(entry_points, |entry| calculator.distance(entry)) {
            Some((start, distance)) => (start, distance, entry_points.len() as u64),
            None => (medoid, calculator.distance(medoid), 1),
        };
    // Only the start is marked: an entry point it was chosen over is left for
    // the walk to reach like any other vertex.
    scratch.mark(start);
    list.offer(start, distance);
    if let (Some(admitted), Some(admission)) = (admitted.as_mut(), admission.as_mut()) {
        consider(admitted, admission, start, distance);
    }

    let mut frontier = Vec::with_capacity(params.beam_width.min(num_vertices));
    let mut fresh = Vec::new();
    loop {
        frontier.clear();
        while frontier.len() < params.beam_width {
            let Some(node) = list.next_unexpanded() else {
                break;
            };
            frontier.push(node.id);
        }
        if frontier.is_empty() {
            break;
        }
        // Sorted, so that which of two equally distant candidates survives a
        // full list does not depend on the order the hop was taken in.
        frontier.sort_unstable();
        fresh.clear();
        for vertex in &frontier {
            fresh.extend(
                graph
                    .neighbors(*vertex)
                    .iter()
                    .copied()
                    .filter(|neighbor| scratch.mark(*neighbor)),
            );
        }
        comparisons += fresh.len() as u64;
        match (admitted.as_mut(), admission.as_mut()) {
            (Some(admitted), Some(admission)) => {
                list.offer_all_with(&fresh, calculator, params.prefetch_ahead, |id, distance| {
                    consider(admitted, admission, id, distance)
                })
            }
            _ => list.offer_all(&fresh, calculator, params.prefetch_ahead),
        }
    }

    Ok(Walked {
        list: list.into_candidates(),
        admitted: admitted
            .map(SearchList::into_candidates)
            .unwrap_or_default(),
        comparisons,
    })
}

#[inline]
fn consider(admitted: &mut SearchList, admission: &mut Admission<'_>, id: u32, distance: f32) {
    if admitted.would_keep(distance) && (admission.accept)(id, distance) {
        admitted.offer(id, distance);
    }
}

/// The nearest of `entries` and its distance, the first of the nearest on a
/// tie. A distance that is not below infinity is passed over; if every one
/// is, the first entry is taken at the distance measured there. `None` only
/// when there are no entries.
pub(crate) fn nearest_entry(entries: &[u32], distance: impl Fn(u32) -> f32) -> Option<(u32, f32)> {
    let mut first = None;
    let mut nearest: Option<(u32, f32)> = None;
    for &entry in entries {
        let measured = distance(entry);
        first.get_or_insert((entry, measured));
        if measured < nearest.map_or(f32::INFINITY, |(_, closest)| closest) {
            nearest = Some((entry, measured));
        }
    }
    nearest.or(first)
}

thread_local! {
    static SCRATCH: RefCell<SearchScratch> = RefCell::new(SearchScratch::new(0));
}

/// Run `f` with this thread's walk scratch.
///
/// Thread-local rather than pooled behind a lock: a walk runs on the CPU pool,
/// where nothing may block, and never yields while it holds the scratch. A
/// re-entrant call gets a scratch of its own.
pub fn with_scratch<T>(f: impl FnOnce(&mut SearchScratch) -> T) -> T {
    SCRATCH.with(|cell| match cell.try_borrow_mut() {
        Ok(mut scratch) => f(&mut scratch),
        Err(_) => f(&mut SearchScratch::new(0)),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::types::Float32Type;
    use arrow_array::{FixedSizeListArray, Float32Array};
    use lance_linalg::distance::DistanceType;

    use super::*;
    use crate::vector::flat::storage::FlatFloatStorage;
    use crate::vector::storage::VectorStore;
    use crate::vector::vamana::graph::PartitionGraph;

    fn line_storage(num_vertices: usize) -> FlatFloatStorage {
        let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            (0..num_vertices).map(|i| Some(vec![Some(i as f32)])),
            1,
        );
        FlatFloatStorage::new(vectors, DistanceType::L2)
    }

    /// Each vertex points at its two neighbors on the line.
    fn line_graph(num_vertices: usize) -> PartitionGraph {
        let adjacency = (0..num_vertices as u32)
            .map(|vertex| {
                [vertex.checked_sub(1), Some(vertex + 1)]
                    .into_iter()
                    .flatten()
                    .filter(|neighbor| (*neighbor as usize) < num_vertices)
                    .collect()
            })
            .collect::<Vec<_>>();
        PartitionGraph::try_new(2, (0..num_vertices as u64).collect(), &adjacency).unwrap()
    }

    fn walk_to(target: f32, entry_points: &[u32], params: &WalkParams) -> Walked {
        let storage = line_storage(100);
        let graph = line_graph(100);
        let calculator = storage.dist_calculator(Arc::new(Float32Array::from(vec![target])), 0.0);
        let mut scratch = SearchScratch::new(0);
        walk(
            &graph,
            &calculator,
            entry_points,
            0,
            params,
            None,
            &mut scratch,
        )
        .unwrap()
    }

    #[test]
    fn a_walk_reaches_its_query_from_the_medoid() {
        let walked = walk_to(70.2, &[], &WalkParams::new(4));
        let ids = walked.list.iter().map(|node| node.id).collect::<Vec<_>>();
        assert_eq!(ids, vec![70, 71, 69, 72]);
        assert!(walked.admitted.is_empty());
    }

    #[test]
    fn a_walk_starts_at_the_nearest_entry_point() {
        let from_medoid = walk_to(90.0, &[], &WalkParams::new(4));
        let from_entry = walk_to(90.0, &[10, 80, 50], &WalkParams::new(4));
        assert_eq!(from_entry.list[0].id, 90);
        assert!(from_entry.comparisons < from_medoid.comparisons);
    }

    #[test]
    fn every_beam_width_finds_the_same_answer_on_a_line() {
        for beam_width in [1, 2, 4, 8] {
            let params = WalkParams {
                beam_width,
                ..WalkParams::new(4)
            };
            assert_eq!(walk_to(33.0, &[], &params).list[0].id, 33);
        }
    }

    #[test]
    fn an_admission_keeps_only_what_it_accepts_and_walks_as_before() {
        let storage = line_storage(100);
        let graph = line_graph(100);
        let calculator = storage.dist_calculator(Arc::new(Float32Array::from(vec![40.0])), 0.0);
        let mut scratch = SearchScratch::new(0);
        let mut accept = |id: u32, _: f32| id % 2 == 1;
        let admitted = walk(
            &graph,
            &calculator,
            &[],
            0,
            &WalkParams::new(8),
            Some(Admission {
                keep: 3,
                accept: &mut accept,
            }),
            &mut scratch,
        )
        .unwrap();
        let unfiltered = walk(
            &graph,
            &calculator,
            &[],
            0,
            &WalkParams::new(8),
            None,
            &mut scratch,
        )
        .unwrap();
        let ids = admitted
            .admitted
            .iter()
            .map(|node| node.id)
            .collect::<Vec<_>>();
        assert_eq!(ids, vec![39, 41, 37]);
        assert_eq!(admitted.list, unfiltered.list);
        assert_eq!(admitted.comparisons, unfiltered.comparisons);
    }

    #[test]
    fn an_admission_considers_the_start_too() {
        let storage = line_storage(100);
        let graph = line_graph(100);
        let calculator = storage.dist_calculator(Arc::new(Float32Array::from(vec![0.0])), 0.0);
        let mut scratch = SearchScratch::new(0);
        let mut accept = |_: u32, _: f32| true;
        let walked = walk(
            &graph,
            &calculator,
            &[],
            0,
            &WalkParams::new(4),
            Some(Admission {
                keep: 2,
                accept: &mut accept,
            }),
            &mut scratch,
        )
        .unwrap();
        let ids = walked
            .admitted
            .iter()
            .map(|node| node.id)
            .collect::<Vec<_>>();
        assert_eq!(ids, vec![0, 1]);
    }

    #[test]
    fn a_start_outside_the_graph_is_refused() {
        let storage = line_storage(10);
        let graph = line_graph(10);
        let calculator = storage.dist_calculator(Arc::new(Float32Array::from(vec![1.0])), 0.0);
        let mut scratch = SearchScratch::new(0);
        let params = WalkParams::new(4);
        assert!(walk(&graph, &calculator, &[], 10, &params, None, &mut scratch).is_err());
        assert!(
            walk(
                &graph,
                &calculator,
                &[3, 12],
                0,
                &params,
                None,
                &mut scratch
            )
            .is_err()
        );
        let zero = WalkParams::new(0);
        assert!(walk(&graph, &calculator, &[], 0, &zero, None, &mut scratch).is_err());
    }

    /// A rule the list could not apply is refused rather than left to a
    /// `debug_assert` a release build skips.
    #[test]
    fn a_stop_rule_that_does_not_fit_the_list_is_refused() {
        let storage = line_storage(10);
        let graph = line_graph(10);
        let calculator = storage.dist_calculator(Arc::new(Float32Array::from(vec![1.0])), 0.0);
        let mut scratch = SearchScratch::new(0);
        let fits = StopRule {
            margin: 0.1,
            rank: 2,
            keep: 3,
        };
        for (stop, refused) in [
            (fits, false),
            // Every bound reached at once: the list holds 4.
            (
                StopRule {
                    margin: 0.0,
                    rank: 1,
                    keep: 4,
                },
                false,
            ),
            (StopRule { rank: 3, ..fits }, false),
            (StopRule { rank: 0, ..fits }, true),
            (StopRule { rank: 4, ..fits }, true),
            (StopRule { keep: 5, ..fits }, true),
            (
                StopRule {
                    margin: 2e19,
                    ..fits
                },
                true,
            ),
            (
                StopRule {
                    margin: -0.1,
                    ..fits
                },
                true,
            ),
            (
                StopRule {
                    margin: f32::NAN,
                    ..fits
                },
                true,
            ),
            (
                StopRule {
                    margin: f32::INFINITY,
                    ..fits
                },
                true,
            ),
        ] {
            let params = WalkParams {
                stop: Some(stop),
                ..WalkParams::new(4)
            };
            let walked = walk(&graph, &calculator, &[], 0, &params, None, &mut scratch);
            assert_eq!(walked.is_err(), refused, "{stop:?}");
            if let Err(error) = walked {
                assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
                assert!(error.to_string().contains("stop rule"), "{error}");
            }
        }
    }

    #[test]
    fn the_nearest_entry_skips_what_cannot_be_ordered() {
        let distances = [f32::NAN, 3.0, 1.0, 1.0];
        assert_eq!(
            nearest_entry(&[0, 1, 2, 3], |entry| distances[entry as usize]),
            Some((2, 1.0))
        );
        let found = nearest_entry(&[0], |_| f32::NAN).unwrap();
        assert_eq!(found.0, 0);
        assert!(found.1.is_nan());
        assert_eq!(nearest_entry(&[], |_| 0.0), None);
    }

    #[test]
    fn a_reentrant_scratch_request_gets_its_own() {
        let outer = with_scratch(|outer| {
            outer.cover(5);
            with_scratch(|inner| inner.capacity())
        });
        assert_eq!(outer, 0);
    }
}
