// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use core::panic;

use super::utils::get_sub_vector_centroids;
use lance_core::assume_eq;
use lance_linalg::distance::{Dot, L2, dot_distance_batch, l2::L2Prepared, l2_distance_batch};

/// Build a Distance Table from the query to each PQ centroid
/// using L2 distance.
pub fn build_distance_table_l2<T: L2>(
    codebook: &[T],
    num_bits: u32,
    num_sub_vectors: usize,
    query: &[T],
) -> Vec<f32> {
    match num_bits {
        4 => build_distance_table_l2_impl::<4, T>(codebook, num_sub_vectors, query),
        8 => build_distance_table_l2_impl::<8, T>(codebook, num_sub_vectors, query),
        _ => panic!("Unsupported number of bits: {}", num_bits),
    }
}

#[inline]
pub fn build_distance_table_l2_impl<const NUM_BITS: u32, T: L2>(
    codebook: &[T],
    num_sub_vectors: usize,
    query: &[T],
) -> Vec<f32> {
    let dimension = query.len();
    let sub_vector_length = dimension / num_sub_vectors;
    let num_centroids = 2_usize.pow(NUM_BITS);
    let mut result = Vec::with_capacity(num_sub_vectors * num_centroids);
    // Legacy writers allowed non-divisible dimensions and truncated the tail.
    // Limit iteration to the sub-vectors that were persisted by those writers.
    for (i, sub_vec) in query
        .chunks_exact(sub_vector_length)
        .take(num_sub_vectors)
        .enumerate()
    {
        let subvec_centroids =
            get_sub_vector_centroids::<NUM_BITS, _>(codebook, dimension, num_sub_vectors, i);
        result.extend(l2_distance_batch(
            sub_vec,
            subvec_centroids,
            sub_vector_length,
        ));
    }
    result
}

/// Build an L2 distance table using pre-prepared [L2Prepared] per sub-vector.
///
/// This avoids the per-call AoS→SoA transpose by reusing targets that were
/// transposed once at `ProductQuantizer` construction time.
pub fn build_distance_table_l2_prepared(l2_targets: &[L2Prepared], query: &[f32]) -> Vec<f32> {
    let sub_dim = query.len() / l2_targets.len();
    let num_targets = l2_targets[0].num_targets();

    let mut result = vec![0.0f32; l2_targets.len() * num_targets];
    // The target count also bounds legacy codebooks whose writers truncated
    // a non-divisible vector tail.
    for (i, (target, sub_vec)) in l2_targets
        .iter()
        .zip(query.chunks_exact(sub_dim))
        .enumerate()
    {
        target.distances_into(sub_vec, &mut result[i * num_targets..][..num_targets]);
    }
    result
}

/// Build a Distance Table from the query to each PQ centroid
/// using Dot distance.
pub fn build_distance_table_dot<T: Dot>(
    codebook: &[T],
    num_bits: u32,
    num_sub_vectors: usize,
    query: &[T],
) -> Vec<f32> {
    match num_bits {
        4 => build_distance_table_dot_impl::<4, T>(codebook, num_sub_vectors, query),
        8 => build_distance_table_dot_impl::<8, T>(codebook, num_sub_vectors, query),
        _ => panic!("Unsupported number of bits: {}", num_bits),
    }
}

#[inline]
pub fn build_distance_table_dot_impl<const NUM_BITS: u32, T: Dot>(
    codebook: &[T],
    num_sub_vectors: usize,
    query: &[T],
) -> Vec<f32> {
    let dimension = query.len();
    let sub_vector_length = dimension / num_sub_vectors;
    let num_centroids = 2_usize.pow(NUM_BITS);
    let mut result = Vec::with_capacity(num_sub_vectors * num_centroids);
    // Legacy writers allowed non-divisible dimensions and truncated the tail.
    // Limit iteration to the sub-vectors that were persisted by those writers.
    for (i, sub_vec) in query
        .chunks_exact(sub_vector_length)
        .take(num_sub_vectors)
        .enumerate()
    {
        let subvec_centroids =
            get_sub_vector_centroids::<NUM_BITS, _>(codebook, dimension, num_sub_vectors, i);
        result.extend(dot_distance_batch(
            sub_vec,
            subvec_centroids,
            sub_vector_length,
        ));
    }
    result
}

/// Compute L2 distance from the query to all code.
///
/// Parameters
/// ----------
/// - distance_table: the pre-computed L2 distance table.
///   It is a flatten array of [num_sub_vectors, num_centroids] f32.
/// - num_bits: the number of bits used for PQ.
/// - num_sub_vectors: the number of sub-vectors.
/// - code: the transposed PQ code to be used to compute the distances.
///
/// Returns
/// -------
///  The squared L2 distance.
///
#[inline]
pub(super) fn compute_pq_distance(
    distance_table: &[f32],
    num_bits: u32,
    num_sub_vectors: usize,
    code: &[u8],
) -> Vec<f32> {
    if code.is_empty() {
        return Vec::new();
    }
    if num_bits == 4 {
        return compute_pq_distance_4bit(distance_table, num_sub_vectors, code);
    }
    // here `code` has been transposed,
    // so code[i][j] is the code of i-th sub-vector of the j-th vector,
    // and `code` is a flatten array of [num_sub_vectors, num_vectors] u8,
    // so code[i * num_vectors + j] is the code of i-th sub-vector of the j-th vector.
    let num_vectors = code.len() / num_sub_vectors;
    let mut distances = vec![0.0; num_vectors];
    // it must be 8
    const NUM_CENTROIDS: usize = 2_usize.pow(8);
    for (sub_vec_idx, vec_indices) in code.chunks_exact(num_vectors).enumerate() {
        let dist_table =
            &distance_table[sub_vec_idx * NUM_CENTROIDS..(sub_vec_idx + 1) * NUM_CENTROIDS];
        assume_eq!(dist_table.len(), NUM_CENTROIDS);
        assume_eq!(vec_indices.len(), distances.len());
        vec_indices
            .iter()
            .zip(distances.iter_mut())
            .for_each(|(&centroid_idx, sum)| {
                *sum += dist_table[centroid_idx as usize];
            });
    }

    distances
}

#[inline]
fn compute_pq_distance_4bit(
    distance_table: &[f32],
    num_sub_vectors: usize,
    code: &[u8],
) -> Vec<f32> {
    let num_vectors = code.len() * 2 / num_sub_vectors;
    let mut distances = vec![0.0f32; num_vectors];
    const NUM_CENTROIDS: usize = 16;

    // Use the same f32 lookup and pairwise addition as PQDistCalculator::distance.
    // Quantizing only part of a partition to u8 changes candidate ordering when
    // a redundant prefilter is removed, and refinement cannot recover lost rows.
    for (sub_vec_idx, vec_indices) in code.chunks_exact(num_vectors).enumerate() {
        let tables = &distance_table[sub_vec_idx * 2 * NUM_CENTROIDS..][..2 * NUM_CENTROIDS];
        let (dist_table, next_dist_table) = tables.split_at(NUM_CENTROIDS);
        for (&centroid_idx, distance) in vec_indices.iter().zip(distances.iter_mut()) {
            let current_idx = (centroid_idx & 0x0f) as usize;
            let next_idx = (centroid_idx >> 4) as usize;
            *distance += dist_table[current_idx] + next_dist_table[next_idx];
        }
    }
    distances
}

/// Compute L2 distance from the query to all code without transposing the code.
/// for testing only
///
/// Type parameters
/// ---------------
/// - C: the tile size of code-book to run at once.
/// - V: the tile size of PQ code to run at once.
///
#[cfg(test)]
fn compute_l2_distance_without_transposing<const C: usize, const V: usize>(
    distance_table: &[f32],
    num_bits: u32,
    num_sub_vectors: usize,
    code: &[u8],
) -> Vec<f32> {
    let num_centroids = super::num_centroids(num_bits);
    let iter = code.chunks_exact(num_sub_vectors * V);
    let distances = iter.clone().flat_map(|c| {
        let mut sums = [0.0_f32; V];
        for i in (0..num_sub_vectors).step_by(C) {
            for (vec_idx, sum) in sums.iter_mut().enumerate() {
                let vec_start = vec_idx * num_sub_vectors;
                let s = c[vec_start + i..]
                    .iter()
                    .take(C.min(num_sub_vectors - i))
                    .enumerate()
                    .map(|(k, c)| distance_table[(i + k) * num_centroids + *c as usize])
                    .sum::<f32>();
                *sum += s;
            }
        }
        sums.into_iter()
    });
    // Remainder
    let remainder = iter.remainder().chunks(num_sub_vectors).map(|c| {
        c.iter()
            .enumerate()
            .map(|(sub_vec_idx, code)| distance_table[sub_vec_idx * num_centroids + *code as usize])
            .sum::<f32>()
    });
    distances.chain(remainder).collect()
}

#[cfg(test)]
mod tests {
    use crate::vector::pq::storage::transpose;

    use super::*;
    use arrow_array::UInt8Array;

    #[test]
    fn test_compute_on_transposed_codes() {
        let num_vectors = 100;
        let num_sub_vectors = 4;
        let num_bits = 8;
        let dimension = 16;
        let codebook =
            Vec::from_iter((0..num_sub_vectors * num_vectors * dimension).map(|v| v as f32));
        let query = Vec::from_iter((0..dimension).map(|v| v as f32));
        let distance_table = build_distance_table_l2(&codebook, num_bits, num_sub_vectors, &query);

        let pq_codes = Vec::from_iter((0..num_vectors * num_sub_vectors).map(|v| v as u8));
        let pq_codes = UInt8Array::from_iter_values(pq_codes);
        let transposed_codes = transpose(&pq_codes, num_vectors, num_sub_vectors);
        let distances = compute_pq_distance(
            &distance_table,
            num_bits,
            num_sub_vectors,
            transposed_codes.values(),
        );
        let expected = compute_l2_distance_without_transposing::<4, 1>(
            &distance_table,
            num_bits,
            num_sub_vectors,
            pq_codes.values(),
        );
        assert_eq!(distances, expected);
    }

    #[rstest::rstest]
    #[case::positive(1.25, 1.0 / 7.0)]
    #[case::negative(-20.5, 1.0 / 7.0)]
    #[case::zero(0.0, 0.0)]
    #[case::positive_constant(1.25, 0.0)]
    #[case::negative_constant(-1.25, 0.0)]
    fn test_compute_4bit_bulk_distance_matches_float_lookup(
        #[case] offset: f32,
        #[case] scale: f32,
        #[values(0, 1, 199, 200, 201, 224, 227)] num_vectors: usize,
    ) {
        const NUM_SUB_VECTORS: usize = 4;
        const NUM_PACKED_CODES: usize = NUM_SUB_VECTORS / 2;
        const NUM_CENTROIDS: usize = 16;

        let distance_table = (0..NUM_SUB_VECTORS * NUM_CENTROIDS)
            .map(|value| offset + value as f32 * scale)
            .collect::<Vec<_>>();
        let packed_codes = (0..num_vectors * NUM_PACKED_CODES)
            .map(|value| {
                let low = (value % NUM_CENTROIDS) as u8;
                let high = ((value * 7 + 3) % NUM_CENTROIDS) as u8;
                low | (high << 4)
            })
            .collect::<Vec<_>>();
        let packed_codes = UInt8Array::from(packed_codes);
        let transposed = if num_vectors == 0 {
            packed_codes.clone()
        } else {
            transpose(&packed_codes, num_vectors, NUM_PACKED_CODES)
        };
        let actual = compute_pq_distance(&distance_table, 4, NUM_SUB_VECTORS, transposed.values());
        let expected = packed_codes
            .values()
            .chunks_exact(NUM_PACKED_CODES)
            .map(|codes| {
                codes
                    .iter()
                    .enumerate()
                    .map(|(byte_idx, code)| {
                        distance_table[byte_idx * 2 * NUM_CENTROIDS + (code & 0x0f) as usize]
                            + distance_table
                                [(byte_idx * 2 + 1) * NUM_CENTROIDS + (code >> 4) as usize]
                    })
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_compute_4bit_bulk_distance_preserves_topk_across_prefix() {
        // The old bulk kernel scored the first 200 rows in f32 and the remaining
        // full SIMD blocks in u8, making identical PQ distances incomparable.
        let mut distance_table = vec![1.0; 32];
        for sub_vector in 0..2 {
            distance_table[sub_vector * 16 + 1] = 1.6;
            distance_table[sub_vector * 16 + 2] = 2.0;
            distance_table[sub_vector * 16 + 3] = 1.8;
        }
        let mut codes = vec![0x22; 224];
        codes[0] = 0x11;
        codes[200] = 0x33;
        let distances = compute_pq_distance(&distance_table, 4, 2, &codes);
        let best = distances
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(best, 0);
        assert_eq!(distances[0], 3.2);
        assert_eq!(distances[200], 3.6);
    }
}
