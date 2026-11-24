// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Nearest-centroid assignment for e-means batches.
//!
//! `f32` vectors are scored with one matrix product per tile, using
//! `||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2`: the `x.c` terms of a tile of
//! vectors against a block of centroids come from a single GEMM instead of one
//! kernel call per vector-centroid pair. Other element types score each pair
//! with the L2 kernel.

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
    /// the same centroids. Empty unless an implementation needs it.
    fn prepare_centroids(_centroids: &[Self], _dimension: usize) -> Vec<f32> {
        Vec::new()
    }

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
        _prepared: &[f32],
        rows: &[usize],
        assignments: &mut Vec<Assignment>,
    ) {
        assign_rows_pairwise(values, dimension, centroids, rows, assignments);
    }
}

impl NearestCentroid for half::f16 {}
impl NearestCentroid for f64 {}

impl NearestCentroid for f32 {
    /// The squared norm of every centroid.
    fn prepare_centroids(centroids: &[Self], dimension: usize) -> Vec<f32> {
        centroids
            .chunks_exact(dimension)
            .map(|centroid| dot(centroid, centroid))
            .collect()
    }

    fn tile_rows(k: usize, dimension: usize) -> usize {
        GemmTiles::new(usize::MAX, k, dimension).map_or(1, |tiles| tiles.rows)
    }

    fn assign_rows(
        values: &[Self],
        dimension: usize,
        centroids: &[Self],
        centroid_norms: &[f32],
        rows: &[usize],
        assignments: &mut Vec<Assignment>,
    ) {
        match GemmTiles::new(rows.len(), centroid_norms.len(), dimension) {
            Some(tiles) => assign_rows_gemm(
                values,
                dimension,
                centroids,
                centroid_norms,
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
    centroid_norms: &[f32],
    rows: &[usize],
    tiles: GemmTiles,
    assignments: &mut Vec<Assignment>,
) {
    let mut vectors = vec![0.0f32; tiles.rows * dimension];
    let mut products = vec![0.0f32; tiles.rows * tiles.centroids];
    // Per vector of the tile: the smallest `||c||^2 - 2 x.c` so far and its
    // centroid. `||x||^2` is the same for every centroid and is added after.
    let mut nearest: Vec<(f32, Option<u32>)> = Vec::with_capacity(tiles.rows);
    assignments.reserve(rows.len());
    for tile_rows in rows.chunks(tiles.rows) {
        let num_vectors = tile_rows.len();
        for (vector, &row) in vectors.chunks_exact_mut(dimension).zip(tile_rows) {
            vector.copy_from_slice(&values[row * dimension..(row + 1) * dimension]);
        }
        let vectors = &vectors[..num_vectors * dimension];
        nearest.clear();
        nearest.resize(num_vectors, (f32::INFINITY, None));

        for (block_index, block) in centroids.chunks(tiles.centroids * dimension).enumerate() {
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
                    .zip(&centroid_norms[first_centroid..first_centroid + num_centroids])
                    .enumerate()
                {
                    let score = norm - 2.0 * product;
                    // A NaN score never compares less, so a vector with NaN
                    // components stays unassigned, as in the pairwise path.
                    if score < best.0 {
                        *best = (score, Some((first_centroid + offset) as u32));
                    }
                }
            }
        }

        assignments.extend(vectors.chunks_exact(dimension).zip(&nearest).map(
            |(vector, &(score, centroid))| {
                // The expansion can round a tiny distance below zero.
                centroid.map(|centroid| (centroid, (dot(vector, vector) + score).max(0.0)))
            },
        ));
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

    /// The GEMM path picks the same centroid as the pairwise path for every
    /// vector, except where two centroids are within float rounding of each
    /// other, and reports the same distance up to rounding.
    #[rstest]
    #[case::hierarchy_node(500, 16, 128)]
    #[case::many_centroid_blocks(300, 700, 96)]
    #[case::tiny_dimension(100, 8, 3)]
    #[case::high_dimension(64, 16, 4096)]
    fn test_gemm_assignment_matches_pairwise(
        #[case] num_rows: usize,
        #[case] k: usize,
        #[case] dimension: usize,
    ) {
        let mut rng = SmallRng::seed_from_u64(3);
        let values = random_values(&mut rng, num_rows * dimension);
        let centroids = random_values(&mut rng, k * dimension);
        let mut rows: Vec<usize> = (0..num_rows).collect();
        rows.shuffle(&mut rng);

        let norms = f32::prepare_centroids(&centroids, dimension);
        let mut gemm = Vec::new();
        f32::assign_rows(&values, dimension, &centroids, &norms, &rows, &mut gemm);
        let mut pairwise = Vec::new();
        assign_rows_pairwise(&values, dimension, &centroids, &rows, &mut pairwise);

        assert_eq!(gemm.len(), rows.len());
        for (&row, (gemm, pairwise)) in rows.iter().zip(gemm.iter().zip(&pairwise)) {
            let (gemm_centroid, gemm_distance) = gemm.unwrap();
            let (pairwise_centroid, pairwise_distance) = pairwise.unwrap();
            let tolerance = 1e-3 * pairwise_distance.max(1.0);
            assert!(
                (gemm_distance - pairwise_distance).abs() <= tolerance,
                "row {row}: distance {gemm_distance} vs {pairwise_distance}"
            );
            if gemm_centroid != pairwise_centroid {
                let vector = &values[row * dimension..(row + 1) * dimension];
                let chosen = &centroids[gemm_centroid as usize * dimension..][..dimension];
                let chosen_distance = f32::l2(vector, chosen);
                assert!(
                    (chosen_distance - pairwise_distance).abs() <= tolerance,
                    "row {row}: centroid {gemm_centroid} at {chosen_distance} is not tied with \
                     {pairwise_centroid} at {pairwise_distance}"
                );
            }
        }
    }

    #[test]
    fn test_gemm_assignment_leaves_nan_vectors_unassigned() {
        let dimension = 4;
        let values = vec![1.0, 2.0, 3.0, 4.0, f32::NAN, 0.0, 0.0, 0.0];
        let centroids = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        let norms = f32::prepare_centroids(&centroids, dimension);
        let mut assignments = Vec::new();
        f32::assign_rows(
            &values,
            dimension,
            &centroids,
            &norms,
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
