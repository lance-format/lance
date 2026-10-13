// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Where a query's walk starts: near its query rather than at the medoid.
//!
//! A walk from the medoid first covers the same ground whatever it was asked.
//! Entry points spread the start over the partition, and a walk measures its
//! query against each of them and starts at the nearest. Counted at equal
//! recall on SIFT, GloVe-200, Cohere and GIST at a million rows a partition,
//! 64 entry points cost 6-16% fewer distances at top-10, the 64 included; at
//! 8 192 rows a partition, 16 measured as few as 64.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array};
use arrow_schema::{DataType, Field};
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use super::build::float_vectors;
use crate::vector::kmeans::{KMeans, KMeansParams};
use crate::vector::storage::{DistCalculator, VectorStore};

/// Vectors sampled per entry point.
const SAMPLE_RATE: usize = 256;

/// Entry points of a partition above [`SMALL_PARTITION_ROWS`].
const NUM_ENTRIES: usize = 64;

/// The geometric middle of the two partition sizes the rule was measured at,
/// 8 192 and 65 536 rows.
const SMALL_PARTITION_ROWS: usize = 23_170;

/// Entry points of a partition of at most [`SMALL_PARTITION_ROWS`]: as few
/// distances as 64 at 8 192 rows, trained in a quarter of the time.
const SMALL_PARTITION_ENTRIES: usize = 16;

const MAX_ITERS: u32 = 50;

/// Fewest vertices one piece of the nearest-vertex scan takes.
const MIN_CHUNK: usize = 4096;

/// How many centroids a partition of `num_vertices` trains.
pub(crate) fn entries_for(num_vertices: usize) -> usize {
    if num_vertices <= SMALL_PARTITION_ROWS {
        SMALL_PARTITION_ENTRIES
    } else {
        NUM_ENTRIES
    }
}

/// A partition's entry points, ascending and distinct; none for a partition no
/// larger than the number it would train, whose walks start at its medoid.
///
/// Over float vectors, the vertex nearest each centroid of a k-means over a
/// sample, as the measurements were taken. Over codes there are no vectors to
/// average, so the entry points are a k-means++ seeding of the sample under
/// the store's own distance.
pub(crate) fn train_entry_points<S: VectorStore>(store: &S, seed: u64) -> Result<Vec<u32>> {
    let num_vertices = store.len();
    let num_entries = entries_for(num_vertices);
    if num_vertices <= num_entries {
        return Ok(Vec::new());
    }
    let mut rng = SmallRng::seed_from_u64(seed);
    let sample_size = num_entries * SAMPLE_RATE;
    // In the order it is drawn, as the crate the rule was measured with keeps
    // it: the k-means starts from rows picked by their place in the sample.
    let sample = if num_vertices > sample_size {
        rand::seq::index::sample(&mut rng, num_vertices, sample_size)
            .into_iter()
            .map(|id| id as u32)
            .collect::<Vec<_>>()
    } else {
        (0..num_vertices as u32).collect()
    };

    let mut entries = match sampled_vectors(store, &sample)? {
        Some(training) => kmeans_entries(store, training, num_entries, &mut rng)?,
        None => seeded_entries(store, &sample, num_entries, &mut rng),
    };
    entries.sort_unstable();
    entries.dedup();
    Ok(entries)
}

/// The float vectors at the distinct positions `sample`, in its order, or
/// `None` when the store holds codes.
fn sampled_vectors<S: VectorStore>(
    store: &S,
    sample: &[u32],
) -> Result<Option<FixedSizeListArray>> {
    // Read in position order, a batch at a time, each into its slot.
    let mut by_position = sample
        .iter()
        .enumerate()
        .map(|(slot, position)| (*position, slot))
        .collect::<Vec<_>>();
    by_position.sort_unstable();
    let mut values = Vec::new();
    let mut dimension = 0;
    let mut next = by_position.iter().peekable();
    let mut start = 0usize;
    for batch in store.to_batches()? {
        let Some((vectors, width)) = float_vectors(&batch)? else {
            return Ok(None);
        };
        if dimension == 0 {
            dimension = width;
            values = vec![0.0; sample.len() * width];
        } else if dimension != width {
            return Err(Error::invalid_input(format!(
                "Vamana partition mixes vectors of {dimension} and {width} dimensions"
            )));
        }
        let end = start + batch.num_rows();
        while let Some((position, slot)) = next.next_if(|(position, _)| (*position as usize) < end)
        {
            let row = *position as usize - start;
            values[slot * width..(slot + 1) * width]
                .copy_from_slice(&vectors.values()[row * width..(row + 1) * width]);
        }
        start = end;
    }
    if next.peek().is_some() || dimension == 0 {
        return Err(Error::internal(format!(
            "Vamana entry-point sample reaches past a store of {start} vectors"
        )));
    }
    Ok(Some(FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dimension as i32,
        Arc::new(Float32Array::from(values)),
        None,
    )?))
}

fn kmeans_entries<S: VectorStore>(
    store: &S,
    training: FixedSizeListArray,
    num_entries: usize,
    rng: &mut SmallRng,
) -> Result<Vec<u32>> {
    let init_rows = rand::seq::index::sample(rng, training.len(), num_entries)
        .into_iter()
        .map(|row| row as u32)
        .collect::<Vec<_>>();
    let init =
        arrow_select::take::take(&training, &arrow_array::UInt32Array::from(init_rows), None)?;
    // Cosine vectors are normalized before they are stored, and k-means only
    // understands L2 and dot.
    let distance_type = match store.distance_type() {
        DistanceType::Cosine => DistanceType::L2,
        other => other,
    };
    let params = KMeansParams::new(
        Some(Arc::new(init.as_fixed_size_list().clone())),
        MAX_ITERS,
        1,
        distance_type,
    )
    .with_hierarchical_k(1);
    let kmeans = KMeans::new_with_params(&training, num_entries, &params)
        .map_err(|e| Error::index(format!("Vamana entry-point k-means failed: {e}")))?;
    let dimension = training.value_length() as usize;
    let centroids = kmeans
        .centroids
        .as_primitive_opt::<Float32Type>()
        .ok_or_else(|| {
            Error::internal(format!(
                "k-means returned {} centroids, not the f32 it was trained on",
                kmeans.centroids.data_type()
            ))
        })?
        .values()
        .chunks_exact(dimension)
        .map(|centroid| Arc::new(Float32Array::from(centroid.to_vec())) as ArrayRef)
        .collect::<Vec<_>>();
    Ok(nearest_vertices(store, &centroids))
}

/// For each centroid, the vertex nearest it, the first of the nearest in id
/// order; a centroid no vertex measures a comparable distance to contributes
/// none.
fn nearest_vertices<S: VectorStore>(store: &S, centroids: &[ArrayRef]) -> Vec<u32> {
    let num_vertices = store.len();
    let chunk = num_vertices
        .div_ceil(rayon::current_num_threads().max(1))
        .max(MIN_CHUNK);
    let starts = (0..num_vertices).step_by(chunk).collect::<Vec<_>>();
    let by_chunk = starts
        .into_par_iter()
        .map(|start| {
            let end = (start + chunk).min(num_vertices);
            centroids
                .iter()
                .map(|centroid| {
                    let calculator = store.dist_calculator(centroid.clone(), 0.0);
                    let mut nearest = (f32::INFINITY, u32::MAX);
                    for id in start as u32..end as u32 {
                        let distance = calculator.distance(id);
                        if distance < nearest.0 {
                            nearest = (distance, id);
                        }
                    }
                    nearest
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    (0..centroids.len())
        .filter_map(|centroid| {
            by_chunk
                .iter()
                .map(|nearest| nearest[centroid])
                .filter(|(_, id)| *id != u32::MAX)
                .min_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)))
                .map(|(_, id)| id)
        })
        .collect()
}

/// k-means++ seeding of `sample`: each next entry point is drawn with
/// probability proportional to its distance to the nearest one drawn so far.
/// Lance's `L2` is squared already, so that is the usual `D^2` weighting.
fn seeded_entries<S: VectorStore>(
    store: &S,
    sample: &[u32],
    num_entries: usize,
    rng: &mut SmallRng,
) -> Vec<u32> {
    let mut entries = Vec::with_capacity(num_entries);
    let mut nearest = vec![f32::INFINITY; sample.len()];
    let mut next = sample[rng.random_range(0..sample.len())];
    loop {
        entries.push(next);
        if entries.len() == num_entries {
            break;
        }
        let calculator = store.dist_calculator_from_id(next);
        for (distance, id) in nearest.iter_mut().zip(sample) {
            let measured = calculator.distance(*id).max(0.0);
            // NaN stays out of the weights: `min` keeps the other operand.
            *distance = distance.min(measured);
        }
        let total = nearest
            .iter()
            .filter(|distance| distance.is_finite())
            .map(|distance| *distance as f64)
            .sum::<f64>();
        if total <= 0.0 {
            break;
        }
        let mut target = rng.random::<f64>() * total;
        let mut drawn = None;
        for (distance, id) in nearest.iter().zip(sample) {
            if !distance.is_finite() || *distance <= 0.0 {
                continue;
            }
            target -= *distance as f64;
            drawn = Some(*id);
            if target <= 0.0 {
                break;
            }
        }
        match drawn {
            Some(id) => next = id,
            None => break,
        }
    }
    entries
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::vector::flat::storage::FlatFloatStorage;
    use crate::vector::vamana::testing::{flat_storage, scattered_values, sq_storage, vectors_of};

    #[test]
    fn the_count_follows_the_partition_size() {
        assert_eq!(entries_for(0), 16);
        assert_eq!(entries_for(SMALL_PARTITION_ROWS), 16);
        assert_eq!(entries_for(SMALL_PARTITION_ROWS + 1), 64);
    }

    #[test]
    fn a_partition_no_larger_than_its_count_keeps_none() {
        let store = flat_storage(scattered_values(16 * 4), 4, DistanceType::L2);
        assert!(train_entry_points(&store, 42).unwrap().is_empty());
        let store = flat_storage(scattered_values(17 * 4), 4, DistanceType::L2);
        assert!(!train_entry_points(&store, 42).unwrap().is_empty());
    }

    /// Four tight clusters far apart: whichever way the entry points are
    /// trained, every cluster gets one, so a walk never starts across the
    /// space from its query.
    fn clustered(points_per_cluster: usize) -> FixedSizeListArray {
        let jitter = scattered_values(4 * points_per_cluster * 2);
        let values = (0..4 * points_per_cluster)
            .flat_map(|row| {
                let center = (row % 4) as f32 * 100.0;
                [center + jitter[2 * row], center - jitter[2 * row + 1]]
            })
            .collect();
        vectors_of(values, 2)
    }

    #[rstest]
    #[case::float_vectors(false)]
    #[case::scalar_codes(true)]
    fn every_cluster_gets_an_entry_point(#[case] quantized: bool) {
        let vectors = clustered(100);
        let entries = if quantized {
            train_entry_points(
                &sq_storage(&vectors, (0..400).collect(), DistanceType::L2),
                42,
            )
            .unwrap()
        } else {
            train_entry_points(&FlatFloatStorage::new(vectors, DistanceType::L2), 42).unwrap()
        };
        assert!(!entries.is_empty() && entries.len() <= 16, "{entries:?}");
        assert!(
            entries.windows(2).all(|pair| pair[0] < pair[1]),
            "{entries:?}"
        );
        assert!(entries.iter().all(|entry| *entry < 400));
        for cluster in 0..4 {
            assert!(
                entries.iter().any(|entry| *entry as usize % 4 == cluster),
                "cluster {cluster} has no entry point: {entries:?}"
            );
        }
    }

    #[test]
    fn the_same_seed_trains_the_same_entry_points() {
        let store = flat_storage(scattered_values(3000 * 4), 4, DistanceType::L2);
        let first = train_entry_points(&store, 42).unwrap();
        assert_eq!(first, train_entry_points(&store, 42).unwrap());
        assert_ne!(first, train_entry_points(&store, 43).unwrap());
    }

    /// The `lance-vamana` crate's recipe, written out: a sample of 256 rows
    /// an entry point kept in the order it is drawn, starting centroids drawn
    /// from its rows by their place in it, and the vertex nearest each
    /// centroid. A flat partition trains what the crate trains, which is what
    /// lets the two be compared.
    #[test]
    fn a_flat_partition_trains_what_the_crate_trains() {
        const ROWS: usize = 5000;
        let vectors = vectors_of(scattered_values(ROWS * 4), 4);
        let mut rng = SmallRng::seed_from_u64(42);
        let take_rows = |from: &FixedSizeListArray, rows: Vec<u32>| {
            arrow_select::take::take(from, &arrow_array::UInt32Array::from(rows), None)
                .unwrap()
                .as_fixed_size_list()
                .clone()
        };
        let drawn = rand::seq::index::sample(&mut rng, ROWS, 16 * 256)
            .into_iter()
            .map(|row| row as u32)
            .collect();
        let training = take_rows(&vectors, drawn);
        let starts = rand::seq::index::sample(&mut rng, training.len(), 16)
            .into_iter()
            .map(|row| row as u32)
            .collect();
        let params = KMeansParams::new(
            Some(Arc::new(take_rows(&training, starts))),
            50,
            1,
            DistanceType::L2,
        )
        .with_hierarchical_k(1);
        let kmeans = KMeans::new_with_params(&training, 16, &params).unwrap();
        let store = FlatFloatStorage::new(vectors, DistanceType::L2);
        let mut expected = kmeans
            .centroids
            .as_primitive::<Float32Type>()
            .values()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|centroid| {
                let query = Arc::new(Float32Array::from(centroid.to_vec()));
                let calculator = store.dist_calculator(query, 0.0);
                (0..ROWS as u32)
                    .min_by(|left, right| {
                        calculator
                            .distance(*left)
                            .total_cmp(&calculator.distance(*right))
                    })
                    .unwrap()
            })
            .collect::<Vec<_>>();
        expected.sort_unstable();
        expected.dedup();
        assert_eq!(train_entry_points(&store, 42).unwrap(), expected);
    }
}
