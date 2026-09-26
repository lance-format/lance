// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use core::panic;

use super::utils::get_sub_vector_centroids;
use lance_core::assume_eq;
use lance_linalg::distance::{Dot, L2, dot_distance_batch, l2::L2Prepared, l2_distance_batch};
use lance_linalg::simd::dist_table::{
    sum_4bit_dist_table_f32_row, sum_4bit_dist_table_f32_rows, sum_4bit_dist_table_f32_transposed,
};

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
        let code_len = num_sub_vectors / 2;
        let mut distances = vec![0.0; code.len() / code_len];
        sum_4bit_dist_table_f32_transposed(
            distances.len(),
            code_len,
            code,
            distance_table,
            &mut distances,
        );
        return distances;
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

/// Exact 4-bit PQ distance of row `id` over column-major codes, in the
/// reference order every exact 4-bit path follows; see
/// [`sum_4bit_dist_table_f32_row`].
pub(super) fn compute_pq_distance_4bit_row(distance_table: &[f32], code: &[u8], id: usize) -> f32 {
    let code_len = distance_table.len() / 32;
    sum_4bit_dist_table_f32_row(code.len() / code_len, code_len, code, distance_table, id)
}

/// Exact 4-bit PQ distances of rows `ids` over column-major codes, writing the
/// distance of `ids[i]` to `distances[i]`, bit-identical to
/// [`compute_pq_distance_4bit_row`].
pub(super) fn compute_pq_distance_4bit_rows(
    distance_table: &[f32],
    code: &[u8],
    ids: &[u32],
    distances: &mut [f32],
) {
    let code_len = distance_table.len() / 32;
    sum_4bit_dist_table_f32_rows(
        code.len() / code_len,
        code_len,
        code,
        distance_table,
        ids,
        distances,
    );
}

/// A 4-bit PQ distance table quantized to `u8` lookup entries.
///
/// Like faiss' fast-scan LUTs, every 16-entry table is shifted by its own
/// minimum and all tables share one scale, chosen so a row's sum of `2 *
/// code_len` entries cannot overflow a `u16` accumulator.
///
/// # Error bounds
///
/// With `q_t(j)` the byte of entry `d_t(j)` and `e_t(j) = q_t(j) * step -
/// (d_t(j) - min_t)` its signed rounding error, a row picking `c_t` in every
/// table has the real sum `S = bias + q * step - sum_t e_t(c_t)`, so
/// `bias + q * step - err_below <= S <= bias + q * step + err_above`.
///
/// `distance()` rounds `S` in `f32` as `-0.0 + fl(lo + hi) + ...`, one
/// rounding per pair and per accumulation. The first accumulation is exact, so
/// each entry passes through at most `C = code_len` roundings, and additions
/// that underflow are exact. Every summand then carries a factor `1 + theta`
/// with `|theta| <= gamma_C = C u / (1 - C u)`, `u = 2^-24` (Higham, Lemma
/// 3.1), which gives `|fl(S) - S| <= gamma_C * magnitude`, and `fl(S)` within
/// `S * (1 -+ gamma_C)` when no entry is negative. Dot scores subtract `diff =
/// num_sub_vectors - 1` once more, off by at most `u * (|fl(S)| + diff)`.
/// `gamma` counts `C + 1` roundings for headroom, and `slack` covers the `f64`
/// rounding of these bounds; both only widen them.
#[derive(Debug, Clone, Copy)]
pub(super) struct QuantizedDistanceTable {
    /// Sum of the per-table minimums.
    bias: f64,
    /// Distance represented by one quantized unit.
    step: f32,
    /// Sum over tables of the largest rounding error, `max_j e_t(j) >= 0`.
    err_below: f64,
    /// Sum over tables of `-min_j e_t(j) >= 0`.
    err_above: f64,
    /// Sum of the largest entry magnitude of each table, which bounds the
    /// magnitude of every partial sum of a row.
    magnitude: f64,
    /// Whether no entry is negative, so the `f32` rounding error of a row
    /// scales with its sum instead of `magnitude`.
    nonnegative: bool,
    /// Relative rounding bound `gamma_{C+1}` of a row's `f32` sum.
    gamma: f64,
    /// `1 / step` and `1 / (1 - gamma)`, which only steer the first guess of
    /// [`Self::sum_cutoff`], so their rounding cannot change its result.
    inv_step: f64,
    inv_lower_gain: f64,
    /// Allowance for the `f64` rounding of `bias`, the errors and the bounds.
    slack: f64,
}

/// Quantize `distance_table` into `quantized`, or return `None` when callers
/// must score exactly: the table has a non-finite entry, no positive span to
/// quantize, or entries so large that a row's `f32` sum could overflow.
pub(super) fn quantize_4bit_distance_table(
    distance_table: &[f32],
    quantized: &mut Vec<u8>,
) -> Option<QuantizedDistanceTable> {
    const NUM_CENTROIDS: usize = 16;
    let (tables, _) = distance_table.as_chunks::<NUM_CENTROIDS>();
    let num_tables = tables.len();
    // Each rounded entry may exceed `span * scale` by half a unit.
    let max_total = (u16::MAX as usize).checked_sub(num_tables)? as f32;

    let mut max_span = 0.0f32;
    let mut span_sum = 0.0f32;
    let mut bias = 0.0f64;
    let mut magnitude = 0.0f64;
    let mut nonnegative = true;
    // The minimum and maximum below would skip or misplace NaN, which would
    // leave NaN entries out of every bound.
    let mut finite = true;
    for table in tables {
        finite &= reduce_16(&table.map(f32::is_finite), |a, b| a & b);
        let (min, max) = (reduce_16(table, min_of), reduce_16(table, max_of));
        max_span = max_span.max(max - min);
        span_sum += max - min;
        bias += f64::from(min);
        magnitude += f64::from(min.abs().max(max.abs()));
        nonnegative &= min >= 0.0;
    }
    if !(finite && span_sum.is_finite() && max_span > 0.0 && magnitude <= f64::from(f32::MAX / 4.0))
    {
        return None;
    }
    let scale = (u8::MAX as f32 / max_span).min(max_total / span_sum);
    let step = scale.recip();
    if !(scale.is_finite() && scale > 0.0 && step.is_finite()) {
        return None;
    }

    quantized.clear();
    quantized.reserve(num_tables * NUM_CENTROIDS);
    let (mut err_below, mut err_above) = (0.0f64, 0.0f64);
    let mut max_sum = 0u32;
    for table in tables {
        let min = reduce_16(table, min_of);
        let codes = table.map(|d| ((d - min) * scale + 0.5).floor().min(u8::MAX as f32) as u8);
        // The minimum entry quantizes to 0 with no error, so the largest error
        // is `>= 0` and the smallest `<= 0`. `code * step` and, short of a
        // 29-bit exponent gap, `d - min` are exact in `f64`.
        let errors: [f64; NUM_CENTROIDS] = std::array::from_fn(|j| {
            f64::from(codes[j]) * f64::from(step) - (f64::from(table[j]) - f64::from(min))
        });
        err_below += reduce_16(&errors, max_of);
        err_above -= reduce_16(&errors, min_of);
        max_sum += u32::from(reduce_16(&codes, max_of));
        quantized.extend_from_slice(&codes);
    }
    // The scale rules this out; checked so that the kernel's `u16` sums
    // provably never wrap.
    if max_sum > u32::from(u16::MAX) {
        return None;
    }

    const U: f64 = f32::EPSILON as f64 / 2.0;
    let roundings = num_tables.div_ceil(2) as f64 + 1.0;
    let gamma = roundings * U / (1.0 - roundings * U);
    // Each of the `num_tables + O(1)` `f64` operations behind a bound rounds
    // by at most `2^-53` of these magnitudes; `2^-50` per operation plus 64
    // spare operations is generous.
    let slack = (num_tables as f64 + 64.0)
        * 2.0f64.powi(-50)
        * (2.0 * magnitude + err_below + err_above + f64::from(u16::MAX) * f64::from(step));
    Some(QuantizedDistanceTable {
        bias,
        step,
        err_below,
        err_above,
        magnitude,
        nonnegative,
        gamma,
        inv_step: f64::from(step).recip(),
        inv_lower_gain: (1.0 - gamma).recip(),
        slack,
    })
}

/// Reduce the 16 entries of a table pairwise.
///
/// Quantization runs once per partition and query. Unlike a sequential fold
/// of `f32::min`, which must handle NaN, the pairwise form vectorizes; for
/// the finite values it sees, it finds the same minimum up to the sign of
/// zero, which never changes a quantized byte.
#[inline(always)]
fn reduce_16<T: Copy>(values: &[T; 16], f: impl Fn(T, T) -> T) -> T {
    let halves: [T; 8] = std::array::from_fn(|i| f(values[i], values[i + 8]));
    let quarters: [T; 4] = std::array::from_fn(|i| f(halves[i], halves[i + 4]));
    f(f(quarters[0], quarters[2]), f(quarters[1], quarters[3]))
}

fn min_of<T: PartialOrd>(a: T, b: T) -> T {
    if b < a { b } else { a }
}

fn max_of<T: PartialOrd>(a: T, b: T) -> T {
    if b > a { b } else { a }
}

/// Whether every entry of a 4-bit distance table is finite and the per-table
/// maximum magnitudes sum to less than `f32::MAX / 4`.
///
/// A row score adds one entry per table, so every partial sum stays within
/// that sum times a rounding factor far below 2, and the dot offset of
/// `num_sub_vectors - 1` cannot push it out of `(f32::MIN, f32::MAX)` either.
pub(super) fn bounded_4bit_scores(distance_table: &[f32]) -> bool {
    const NUM_CENTROIDS: usize = 16;
    let mut magnitude_sum = 0.0f32;
    for table in distance_table.chunks_exact(NUM_CENTROIDS) {
        let mut max_magnitude = 0.0f32;
        for &d in table {
            if !d.is_finite() {
                return false;
            }
            max_magnitude = max_magnitude.max(d.abs());
        }
        magnitude_sum += max_magnitude;
    }
    magnitude_sum < f32::MAX / 4.0
}

impl QuantizedDistanceTable {
    /// The amount `distance()` subtracts from a row's sum to get its score,
    /// and a bound on the rounding of that subtraction. Only dot subtracts,
    /// with `score_diff = num_sub_vectors - 1 > 0`.
    fn score_shift(&self, score_diff: f32) -> (f64, f64) {
        if score_diff == 0.0 {
            return (0.0, 0.0);
        }
        const U: f64 = f32::EPSILON as f64 / 2.0;
        let diff = f64::from(score_diff);
        // The `2^-20` margin covers the `f64` rounding of the `diff` terms.
        let rounding =
            U * (self.magnitude * (1.0 + self.gamma) + diff.abs()) * (1.0 + 2.0f64.powi(-20));
        (diff, rounding)
    }

    /// `bias` lowered by the largest quantization error.
    fn lower_offset(&self) -> f64 {
        self.bias - self.err_below - self.slack
    }

    /// `bias + sum * step` lowered by the largest quantization error.
    fn lower_row_sum(&self, sum: u16) -> f64 {
        f64::from(sum) * f64::from(self.step) + self.lower_offset()
    }

    /// A lower bound of the score of every row whose quantized sum is `sum`,
    /// nondecreasing in `sum` since every `f64` operation rounds monotonically.
    fn lower_score(&self, sum: u16, (diff, rounding): (f64, f64)) -> f64 {
        let row_sum = self.lower_row_sum(sum);
        let dist = if self.nonnegative {
            row_sum.max(0.0) * (1.0 - self.gamma)
        } else {
            row_sum - self.gamma * self.magnitude
        };
        dist - diff - rounding
    }

    /// An upper bound of the score of every row whose quantized sum is `sum`,
    /// nondecreasing in `sum`.
    fn upper_score(&self, sum: u16, (diff, rounding): (f64, f64)) -> f64 {
        let row_sum =
            f64::from(sum) * f64::from(self.step) + (self.bias + self.err_above + self.slack);
        let dist = if self.nonnegative {
            row_sum.max(0.0) * (1.0 + self.gamma)
        } else {
            row_sum + self.gamma * self.magnitude
        };
        dist - diff + rounding
    }

    /// The largest quantized sum of a row whose score (exact distance minus
    /// `score_diff`) may still order below `threshold`, or `None` when no row's
    /// can. Every row with a larger sum scores at or above `threshold`.
    ///
    /// Scores compare in the total order of
    /// [`OrderedFloat`](crate::vector::graph::OrderedFloat): a positive NaN
    /// threshold ranks above every score and cuts nothing, a negative NaN
    /// below all, and `-0.0` orders below `+0.0`.
    pub(super) fn sum_cutoff(&self, threshold: f32, score_diff: f32) -> Option<u16> {
        if threshold.is_nan() {
            return threshold.is_sign_positive().then_some(u16::MAX);
        }
        // A score whose real bound is `>= 0` may still be `-0.0`, which orders
        // below a `+0.0` threshold, so that threshold keeps bounds `<= 0`.
        let threshold = if threshold.to_bits() == 0 {
            f64::from_bits(1)
        } else {
            f64::from(threshold)
        };
        let shift = self.score_shift(score_diff);
        let below = |sum| self.lower_score(sum, shift) < threshold;
        // Invert `lower_score` in closed form, then step to the exact largest
        // qualifying sum, which the inversion's `f64` rounding can miss by a
        // unit. This runs on every heap replacement, so a good guess costs two
        // evaluations of `below`.
        let dist = threshold + shift.0 + shift.1;
        let row_sum = if self.nonnegative {
            dist * self.inv_lower_gain
        } else {
            dist + self.gamma * self.magnitude
        };
        let estimate = ((row_sum - self.lower_offset()) * self.inv_step).ceil() - 1.0;
        let mut sum = estimate.clamp(0.0, f64::from(u16::MAX)) as u16;
        if below(sum) {
            while sum < u16::MAX && below(sum + 1) {
                sum += 1;
            }
        } else {
            if !below(0) {
                return None;
            }
            // `below(0)` ends the walk.
            while !below(sum) {
                sum -= 1;
            }
        }
        Some(sum)
    }

    /// A distance strictly above the score (exact distance minus
    /// `score_diff`) of every row whose quantized sum is at most `sum`.
    pub(super) fn kth_bound_distance(&self, sum: u16, score_diff: f32) -> f32 {
        let upper = self.upper_score(sum, self.score_shift(score_diff));
        let bound = upper as f32;
        if f64::from(bound) > upper {
            bound
        } else {
            bound.next_up()
        }
    }
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
    use rand::{Rng, SeedableRng, rngs::StdRng};

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

    /// Per-row 4-bit PQ distances over row-major packed codes, summed in the
    /// same order as `PQDistCalculator::distance`.
    fn reference_4bit_distances(
        distance_table: &[f32],
        packed_codes: &[u8],
        code_len: usize,
    ) -> Vec<f32> {
        packed_codes
            .chunks_exact(code_len)
            .map(|codes| {
                codes
                    .iter()
                    .enumerate()
                    .map(|(column, code)| {
                        distance_table[column * 32 + (code & 0x0f) as usize]
                            + distance_table[column * 32 + 16 + (code >> 4) as usize]
                    })
                    .sum::<f32>()
            })
            .collect()
    }

    fn random_4bit_codes(rng: &mut StdRng, num_vectors: usize, code_len: usize) -> UInt8Array {
        UInt8Array::from_iter_values((0..num_vectors * code_len).map(|_| rng.random::<u8>()))
    }

    fn transposed_4bit_codes(
        packed_codes: &UInt8Array,
        num_vectors: usize,
        code_len: usize,
    ) -> UInt8Array {
        if num_vectors == 0 {
            return packed_codes.clone();
        }
        transpose(packed_codes, num_vectors, code_len)
    }

    #[rstest::rstest]
    fn test_compute_4bit_bulk_distance_matches_per_row(#[values(0, 17, 600)] num_vectors: usize) {
        const NUM_SUB_VECTORS: usize = 6;
        const CODE_LEN: usize = NUM_SUB_VECTORS / 2;
        let mut rng = StdRng::seed_from_u64(num_vectors as u64);
        let distance_table = (0..NUM_SUB_VECTORS * 16)
            .map(|_| rng.random_range(-3.0f32..5.0))
            .collect::<Vec<_>>();
        let packed_codes = random_4bit_codes(&mut rng, num_vectors, CODE_LEN);
        let transposed = transposed_4bit_codes(&packed_codes, num_vectors, CODE_LEN);

        let actual = compute_pq_distance(&distance_table, 4, NUM_SUB_VECTORS, transposed.values());
        let expected = reference_4bit_distances(&distance_table, packed_codes.values(), CODE_LEN);
        let bits = |dists: &[f32]| dists.iter().map(|d| d.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&actual), bits(&expected));
    }
}
