// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Nearest-centroid assignment for e-means batches.
//!
//! `f32` vectors are scored with one matrix product per tile, using
//! `||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2`: the `x.c` terms of a tile of
//! vectors against a block of centroids come from a single GEMM instead of one
//! kernel call per vector-centroid pair. Other element types score each pair
//! with the L2 kernel.
//!
//! The expansion rounds in proportion to the norms, so it can lose the gap
//! between two nearby centroids far from the origin. Vectors and centroids are
//! scored relative to the centroids' mean, which leaves distances unchanged
//! and shrinks the norms. That alone is not enough when centroids are close to
//! each other but spread around the mean, so a vector whose two best scores
//! are within the rounding bound of each other is reassigned with the direct
//! L2 kernel. Reported distances always come from the direct kernel.

use lance_linalg::distance::{dot::dot, l2::L2, l2::l2_distance_batch_into};
use lance_linalg::kernels::argmin_value_float;
use ndarray::linalg::general_mat_mul;
use ndarray::{ArrayView2, ArrayViewMut2};
use num_traits::Float;

/// `matrixmultiply`, the GEMM behind `ndarray` here, hands a product to its
/// global pool of at most four threads once `m*k + k*n` reaches 2^15 elements.
/// Products issued from many rayon tasks at once would queue on that small
/// pool, so every tile stays below the threshold and runs on its caller.
const GEMM_TILE_MAX_ELEMENTS: usize = (1 << 15) - 1;
/// Vectors a tile keeps at least when all centroids share one block: one
/// 8-row panel of the GEMM micro-kernel.
const MIN_TILE_ROWS: usize = 8;

/// The nearest centroid of a vector and the squared L2 distance to it, or
/// `None` when every distance is NaN.
pub(in crate::vector::kmeans) type Assignment = Option<(u32, f32)>;

/// Assigns vectors to their nearest centroid under squared L2 distance.
pub(in crate::vector::kmeans) trait NearestCentroid:
    Float + L2 + Send + Sync
{
    /// Per-centroid data that [`Self::assign_rows`] reuses across calls with
    /// the same centroids.
    type Prepared: Sync;

    fn prepare_centroids(centroids: &[Self], dimension: usize) -> Self::Prepared;

    /// Vectors [`Self::assign_rows`] scores together against `k` centroids;
    /// fewer rows per call leave its kernels partly idle.
    fn tile_rows(_k: usize, _dimension: usize) -> usize {
        1
    }

    /// Appends to `assignments` the [`Assignment`] of every row id of `rows`
    /// into `values`. `prepared` is [`Self::prepare_centroids`] of
    /// `centroids`.
    fn assign_rows(
        values: &[Self],
        dimension: usize,
        centroids: &[Self],
        _prepared: &Self::Prepared,
        rows: &[usize],
        assignments: &mut Vec<Assignment>,
    ) {
        assign_rows_pairwise(values, dimension, centroids, rows, assignments);
    }
}

impl NearestCentroid for half::f16 {
    type Prepared = ();

    fn prepare_centroids(_centroids: &[Self], _dimension: usize) {}
}

impl NearestCentroid for f64 {
    type Prepared = ();

    fn prepare_centroids(_centroids: &[Self], _dimension: usize) {}
}

/// Centroids relative to their mean, as the GEMM path scores them.
pub(in crate::vector::kmeans) struct CenteredCentroids {
    center: Vec<f32>,
    centroids: Vec<f32>,
    /// The squared norm of every centered centroid.
    norms: Vec<f32>,
    /// The largest norm of a centered centroid.
    max_norm: f32,
}

impl NearestCentroid for f32 {
    type Prepared = CenteredCentroids;

    fn prepare_centroids(centroids: &[Self], dimension: usize) -> CenteredCentroids {
        let center = centroid_mean(centroids, dimension);
        let centroids: Vec<Self> = centroids
            .chunks_exact(dimension)
            .flat_map(|centroid| centroid.iter().zip(&center).map(|(c, m)| c - m))
            .collect();
        let norms: Vec<Self> = centroids
            .chunks_exact(dimension)
            .map(|centroid| dot(centroid, centroid))
            .collect();
        let max_norm = norms.iter().copied().fold(0.0, Self::max).sqrt();
        CenteredCentroids {
            center,
            centroids,
            norms,
            max_norm,
        }
    }

    fn tile_rows(k: usize, dimension: usize) -> usize {
        GemmTiles::new(usize::MAX, k, dimension).map_or(1, |tiles| tiles.rows)
    }

    fn assign_rows(
        values: &[Self],
        dimension: usize,
        centroids: &[Self],
        centered: &CenteredCentroids,
        rows: &[usize],
        assignments: &mut Vec<Assignment>,
    ) {
        match GemmTiles::new(rows.len(), centered.norms.len(), dimension) {
            Some(tiles) => assign_rows_gemm(
                values,
                dimension,
                centroids,
                centered,
                rows,
                tiles,
                assignments,
            ),
            None => assign_rows_pairwise(values, dimension, centroids, rows, assignments),
        }
    }
}

fn assign_rows_pairwise<N: Float + L2>(
    values: &[N],
    dimension: usize,
    centroids: &[N],
    rows: &[usize],
    assignments: &mut Vec<Assignment>,
) {
    let mut distances = vec![0.0f32; centroids.len() / dimension];
    assignments.extend(rows.iter().map(|&row| {
        let vector = &values[row * dimension..(row + 1) * dimension];
        l2_distance_batch_into(vector, centroids, dimension, &mut distances);
        argmin_value_float(distances.iter().copied())
    }));
}

/// The mean of `centroids`, or the origin if it is not finite, so that one
/// non-finite centroid cannot leave every vector unassigned.
fn centroid_mean(centroids: &[f32], dimension: usize) -> Vec<f32> {
    let mut sum = vec![0.0f64; dimension];
    for centroid in centroids.chunks_exact(dimension) {
        for (total, &value) in sum.iter_mut().zip(centroid) {
            *total += value as f64;
        }
    }
    let num_centroids = (centroids.len() / dimension).max(1) as f64;
    let mean: Vec<f32> = sum
        .iter()
        .map(|total| (total / num_centroids) as f32)
        .collect();
    if mean.iter().all(|value| value.is_finite()) {
        mean
    } else {
        vec![0.0; dimension]
    }
}

/// How far apart two GEMM scores `||c||^2 - 2 x.c` of one vector must be for
/// rounding to be unable to swap them, given the norms of the centered vector
/// and of the largest centered centroid.
///
/// A dot product or squared norm of `dimension` terms is within
/// `dimension * u * (sum of |term|)` of its exact value for any summation
/// order, where `u` is the unit roundoff, and centering rounds every component
/// by at most `u` of itself. All of these are below
/// `(dimension + 4) * u * (|x| + |c|)^2` per score. `f32::EPSILON` is `2u`,
/// which covers the two scores being compared.
fn ambiguity_margin(dimension: usize, vector_norm: f32, max_centroid_norm: f32) -> f32 {
    let scale = vector_norm + max_centroid_norm;
    (dimension + 4) as f32 * f32::EPSILON * scale * scale
}

/// Vectors and centroids per GEMM tile, sized so that a tile's product stays
/// under [`GEMM_TILE_MAX_ELEMENTS`].
#[derive(Clone, Copy)]
struct GemmTiles {
    rows: usize,
    centroids: usize,
}

impl GemmTiles {
    /// `None` when there are no rows or centroids, or the dimension is too
    /// large for even a one-by-one tile.
    fn new(num_rows: usize, k: usize, dimension: usize) -> Option<Self> {
        let vectors_per_tile = GEMM_TILE_MAX_ELEMENTS / dimension;
        if vectors_per_tile < 2 || num_rows == 0 || k == 0 {
            return None;
        }
        // A small table, such as a 16-centroid node of hierarchical training,
        // stays in one block: split, its last block would be a GEMM of a few
        // centroids that costs the packing of a full one.
        let centroids = if k + MIN_TILE_ROWS <= vectors_per_tile {
            k
        } else {
            k.min(vectors_per_tile / 2).max(1)
        };
        let rows = (vectors_per_tile - centroids).min(num_rows);
        Some(Self { rows, centroids })
    }
}

fn assign_rows_gemm(
    values: &[f32],
    dimension: usize,
    centroids: &[f32],
    centered: &CenteredCentroids,
    rows: &[usize],
    tiles: GemmTiles,
    assignments: &mut Vec<Assignment>,
) {
    let mut vectors = vec![0.0f32; tiles.rows * dimension];
    let mut products = vec![0.0f32; tiles.rows * tiles.centroids];
    // Per vector of the tile: the two smallest `||c||^2 - 2 x.c` so far and
    // the centroid of the smallest. `||x||^2` is the same for every centroid.
    let mut nearest: Vec<(f32, f32, Option<u32>)> = Vec::with_capacity(tiles.rows);
    let mut distances = Vec::new();
    assignments.reserve(rows.len());
    for tile_rows in rows.chunks(tiles.rows) {
        let num_vectors = tile_rows.len();
        for (vector, &row) in vectors.chunks_exact_mut(dimension).zip(tile_rows) {
            let original = &values[row * dimension..(row + 1) * dimension];
            for ((value, &original), &mean) in vector.iter_mut().zip(original).zip(&centered.center)
            {
                *value = original - mean;
            }
        }
        let vectors = &vectors[..num_vectors * dimension];
        nearest.clear();
        nearest.resize(num_vectors, (f32::INFINITY, f32::INFINITY, None));

        for (block_index, block) in centered
            .centroids
            .chunks(tiles.centroids * dimension)
            .enumerate()
        {
            let first_centroid = block_index * tiles.centroids;
            let num_centroids = block.len() / dimension;
            let products = &mut products[..num_vectors * num_centroids];
            general_mat_mul(
                1.0,
                &ArrayView2::from_shape((num_vectors, dimension), vectors)
                    .expect("the vector tile holds num_vectors rows"),
                &ArrayView2::from_shape((num_centroids, dimension), block)
                    .expect("the centroid block holds num_centroids rows")
                    .t(),
                0.0,
                &mut ArrayViewMut2::from_shape((num_vectors, num_centroids), &mut *products)
                    .expect("the product tile holds num_vectors x num_centroids values"),
            );
            for (best, row_products) in nearest.iter_mut().zip(products.chunks_exact(num_centroids))
            {
                for (offset, (&product, &norm)) in row_products
                    .iter()
                    .zip(&centered.norms[first_centroid..first_centroid + num_centroids])
                    .enumerate()
                {
                    let score = norm - 2.0 * product;
                    // A NaN score never compares less, so a vector with NaN
                    // components stays unassigned, as in the pairwise path.
                    if score < best.0 {
                        *best = (score, best.0, Some((first_centroid + offset) as u32));
                    } else if score < best.1 {
                        best.1 = score;
                    }
                }
            }
        }

        for ((vector, &row), &(score, runner_up, centroid)) in
            vectors.chunks_exact(dimension).zip(tile_rows).zip(&nearest)
        {
            let original = &values[row * dimension..(row + 1) * dimension];
            let Some(centroid) = centroid else {
                assignments.push(None);
                continue;
            };
            let margin = ambiguity_margin(dimension, dot(vector, vector).sqrt(), centered.max_norm);
            // An infinite runner-up means a single centroid, nothing to swap.
            if runner_up - score > margin || runner_up.is_infinite() {
                let chosen = &centroids[centroid as usize * dimension..][..dimension];
                assignments.push(Some((centroid, f32::l2(original, chosen))));
            } else {
                distances.resize(centered.norms.len(), 0.0);
                l2_distance_batch_into(original, centroids, dimension, &mut distances);
                assignments.push(argmin_value_float(distances.iter().copied()));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use rstest::rstest;

    use super::*;

    fn random_values(rng: &mut SmallRng, len: usize) -> Vec<f32> {
        (0..len).map(|_| rng.random_range(-1.0..1.0)).collect()
    }

    /// The GEMM path picks the centroid the pairwise path picks for every
    /// vector and reports the same distance up to rounding.
    #[rstest]
    #[case::hierarchy_node(500, 16, 128, 0.0)]
    #[case::many_centroid_blocks(300, 700, 96, 0.0)]
    #[case::tiny_dimension(100, 8, 3, 0.0)]
    #[case::high_dimension(64, 16, 4096, 0.0)]
    #[case::far_from_origin(500, 16, 128, 1000.0)]
    fn test_gemm_assignment_matches_pairwise(
        #[case] num_rows: usize,
        #[case] k: usize,
        #[case] dimension: usize,
        #[case] offset: f32,
    ) {
        let mut rng = SmallRng::seed_from_u64(3);
        let mut values = random_values(&mut rng, num_rows * dimension);
        let mut centroids = random_values(&mut rng, k * dimension);
        for value in values.iter_mut().chain(&mut centroids) {
            *value += offset;
        }
        let mut rows: Vec<usize> = (0..num_rows).collect();
        rows.shuffle(&mut rng);

        let prepared = f32::prepare_centroids(&centroids, dimension);
        let mut gemm = Vec::new();
        f32::assign_rows(&values, dimension, &centroids, &prepared, &rows, &mut gemm);
        let mut pairwise = Vec::new();
        assign_rows_pairwise(&values, dimension, &centroids, &rows, &mut pairwise);

        assert_eq!(gemm.len(), rows.len());
        for (&row, (gemm, pairwise)) in rows.iter().zip(gemm.iter().zip(&pairwise)) {
            let (gemm_centroid, gemm_distance) = gemm.unwrap();
            let (pairwise_centroid, pairwise_distance) = pairwise.unwrap();
            assert_eq!(gemm_centroid, pairwise_centroid, "row {row}");
            let tolerance = 1e-5 * pairwise_distance.max(1.0);
            assert!(
                (gemm_distance - pairwise_distance).abs() <= tolerance,
                "row {row}: distance {gemm_distance} vs {pairwise_distance}"
            );
        }
    }

    #[test]
    fn test_gemm_assignment_leaves_nan_vectors_unassigned() {
        let dimension = 4;
        let values = vec![1.0, 2.0, 3.0, 4.0, f32::NAN, 0.0, 0.0, 0.0];
        let centroids = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        let prepared = f32::prepare_centroids(&centroids, dimension);
        let mut assignments = Vec::new();
        f32::assign_rows(
            &values,
            dimension,
            &centroids,
            &prepared,
            &[0, 1],
            &mut assignments,
        );
        assert_eq!(assignments, vec![Some((0, 0.0)), None]);
    }

    #[test]
    fn test_gemm_tiles_stay_below_matrixmultiply_threading() {
        for (num_rows, k, dimension) in [(4096, 16, 128), (4096, 65_536, 768), (10, 4, 8)] {
            let tiles = GemmTiles::new(num_rows, k, dimension).unwrap();
            assert!((tiles.rows + tiles.centroids) * dimension <= GEMM_TILE_MAX_ELEMENTS);
            assert!(tiles.rows >= 1 && tiles.rows <= num_rows);
            assert!(tiles.centroids >= 1 && tiles.centroids <= k);
        }
        assert!(GemmTiles::new(10, 4, 1 << 15).is_none());
        assert!(GemmTiles::new(10, 0, 8).is_none());
        // A hierarchy node keeps its 16 centroids in one block even at
        // dimension 1,024, where a tile holds 31 vectors in all.
        let tiles = GemmTiles::new(4096, 16, 1024).unwrap();
        assert_eq!((tiles.rows, tiles.centroids), (15, 16));
    }
}
