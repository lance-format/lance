// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use num_traits::AsPrimitive;
use rand::RngCore;

// Fast random rotation used by the RabitQ "fast" path.
//
// The transform is a composition of:
// 1) random diagonal sign flips (Rademacher variables),
// 2) FWHT-style mixing on a power-of-two window,
// 3) a Kac-style pairwise mixing step for non-power-of-two dimensions.
//
// Background:
// - Hadamard transform: https://en.wikipedia.org/wiki/Hadamard_transform
// - Fast Walsh-Hadamard transform (FWHT):
//   https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform
// - Rademacher random signs:
//   https://en.wikipedia.org/wiki/Rademacher_distribution
// - Kac-walk-based fast dimension reduction (uses fixed-angle pair rotations):
//   https://arxiv.org/abs/2003.10069
// - Givens / plane rotation:
//   https://en.wikipedia.org/wiki/Givens_rotation
const FAST_ROTATION_ROUNDS: usize = 4;

#[inline]
fn fwht_in_place(values: &mut [f32]) {
    // In-place FWHT butterfly network.
    // For each stage, pair entries (x, y) and map to (x + y, x - y).
    // Complexity: O(n log n) operations, no extra heap allocation.
    debug_assert!(values.len().is_power_of_two());
    let mut half = 1usize;
    while half < values.len() {
        let step = half * 2;
        for block in values.chunks_exact_mut(step) {
            let (left, right) = block.split_at_mut(half);
            for (x, y) in left.iter_mut().zip(right.iter_mut()) {
                let lx = *x;
                let ry = *y;
                *x = lx + ry;
                *y = lx - ry;
            }
        }
        half = step;
    }
}

/// Applies a random diagonal matrix with +/-1 entries by toggling the f32
/// sign bit: bit `i` of `signs` (LSB first) flips `values[i]`.
type FlipSignsFn = unsafe fn(&mut [f32], &[u8]);

lance_core::dispatch! {
    static FLIP_SIGNS: Dispatch<FlipSignsFn> = {
        #[cfg(target_arch = "x86_64")]
        {
            &[x86::FLIP_SIGNS_AVX2, FLIP_SIGNS_SCALAR]
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            &[FLIP_SIGNS_SCALAR]
        }
    };
}

lance_core::target_kernel! {
    #[target_features()]
    #[candidate(FLIP_SIGNS_SCALAR: FlipSignsFn)]
    #[inline]
    fn flip_signs_scalar(values: &mut [f32], signs: &[u8]) {
        for (byte_idx, &mask) in signs.iter().enumerate() {
            let start = byte_idx * 8;
            if start >= values.len() {
                break;
            }
            let end = (start + 8).min(values.len());
            for (bit_idx, value) in values[start..end].iter_mut().enumerate() {
                let sign_mask = (((mask >> bit_idx) & 1) as u32) << 31;
                *value = f32::from_bits(value.to_bits() ^ sign_mask);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    use super::{FlipSignsFn, flip_signs_scalar};

    lance_core::target_kernel! {
        #[target_features("avx2")]
        #[candidate(pub(super) FLIP_SIGNS_AVX2: FlipSignsFn)]
        /// Vectorized `flip_signs_scalar`: consume 8 f32 values per AVX2 lane.
        /// The sign mask is expanded from one byte to 8 lane-wise sign-bit masks.
        fn flip_signs_avx2(values: &mut [f32], signs: &[u8]) {
            let bit_select = _mm256_setr_epi32(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80);
            let sign_flip = _mm256_set1_epi32(0x80000000u32 as i32);

            let (chunks, _) = values.as_chunks_mut::<8>();
            let full_chunks = chunks.len();
            for (chunk, &mask) in chunks.iter_mut().zip(signs) {
                let mask_bits = _mm256_set1_epi32(mask as i32);
                let test = _mm256_and_si256(mask_bits, bit_select);
                let cmp = _mm256_cmpeq_epi32(test, bit_select);
                let xor_mask = _mm256_and_si256(cmp, sign_flip);

                // SAFETY: `chunk` holds exactly 8 floats.
                unsafe {
                    let vec = _mm256_loadu_ps(chunk.as_ptr());
                    let out = _mm256_xor_ps(vec, _mm256_castsi256_ps(xor_mask));
                    _mm256_storeu_ps(chunk.as_mut_ptr(), out);
                }
            }

            if full_chunks * 8 < values.len() && full_chunks < signs.len() {
                flip_signs_scalar(&mut values[full_chunks * 8..], &signs[full_chunks..]);
            }
        }
    }
}

#[inline]
fn flip_signs(values: &mut [f32], signs: &[u8]) {
    debug_assert!(signs.len() * 8 >= values.len());
    FLIP_SIGNS.get().call(values, signs);
}

#[inline]
fn kacs_walk(values: &mut [f32]) {
    // A fixed-angle (pi/4) plane-rotation-like sweep over paired coordinates:
    // (x, y) -> (x + y, x - y). Up to normalization, this is a 2x2 Hadamard block
    // and corresponds to one Kac-style mixing step.
    let half = values.len() / 2;
    let (left, right) = values.split_at_mut(half);
    for (x, y) in left.iter_mut().zip(right.iter_mut()) {
        let lx = *x;
        let ry = *y;
        *x = lx + ry;
        *y = lx - ry;
    }
}

#[inline]
fn rescale(values: &mut [f32], factor: f32) {
    // Keep the transform numerically stable and approximately orthonormal.
    for value in values.iter_mut() {
        *value *= factor;
    }
}

#[inline]
fn sign_bytes_per_round(dim: usize) -> usize {
    dim.div_ceil(8)
}

pub(crate) fn fast_rotation_signs_len(dim: usize) -> usize {
    FAST_ROTATION_ROUNDS * sign_bytes_per_round(dim)
}

pub fn random_fast_rotation_signs(dim: usize) -> Vec<u8> {
    // Each round needs one random sign bit per dimension.
    let mut signs = vec![0u8; fast_rotation_signs_len(dim)];
    rand::rng().fill_bytes(&mut signs);
    signs
}

#[inline]
pub fn apply_fast_rotation<T: AsPrimitive<f32>>(input: &[T], output: &mut [f32], signs: &[u8]) {
    // Fast random rotation pipeline, aligned with RaBitQ-Library's FhtKacRotator:
    // - power-of-two dims: repeat [random signs -> FWHT -> scale] for 4 rounds
    // - non-power-of-two dims: alternate FWHT on head/tail + Kac mixing
    //
    // This keeps the fast path matrix-free: no dense orthogonal matrix materialization.
    let dim = output.len();
    let input_len = input.len().min(dim);
    output[..input_len]
        .iter_mut()
        .zip(input[..input_len].iter())
        .for_each(|(dst, src)| *dst = src.as_());
    if input_len < dim {
        output[input_len..].fill(0.0);
    }

    apply_fast_rotation_in_place(output, signs);
}

#[inline]
pub fn apply_fast_rotation_in_place(output: &mut [f32], signs: &[u8]) {
    let dim = output.len();
    let bytes_per_round = sign_bytes_per_round(dim);
    debug_assert_eq!(signs.len(), FAST_ROTATION_ROUNDS * bytes_per_round);
    if dim == 0 {
        return;
    }

    let trunc_dim = 1usize << dim.ilog2();
    let scale = 1.0f32 / (trunc_dim as f32).sqrt();
    if trunc_dim == dim {
        for round in 0..FAST_ROTATION_ROUNDS {
            let offset = round * bytes_per_round;
            flip_signs(output, &signs[offset..offset + bytes_per_round]);
            fwht_in_place(output);
            rescale(output, scale);
        }
        return;
    }

    let start = dim - trunc_dim;
    for round in 0..FAST_ROTATION_ROUNDS {
        let offset = round * bytes_per_round;
        flip_signs(output, &signs[offset..offset + bytes_per_round]);

        if round % 2 == 0 {
            let head = &mut output[..trunc_dim];
            fwht_in_place(head);
            rescale(head, scale);
        } else {
            let tail = &mut output[start..];
            fwht_in_place(tail);
            rescale(tail, scale);
        }

        kacs_walk(output);
    }

    // Matches RaBitQ-Library FhtKacRotator behavior for non-power-of-two dimensions.
    // The extra factor compensates the alternating truncated FWHT + Kac steps above.
    rescale(output, 0.25);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_rotation_sign_bytes() {
        assert_eq!(random_fast_rotation_signs(128).len(), 64);
        assert_eq!(random_fast_rotation_signs(130).len(), 68);
    }

    /// Every host-supported sign-flip kernel matches a per-element reference
    /// bit for bit, across full 8-lane chunks and tails.
    #[rstest::rstest]
    fn test_flip_signs_candidates_match_reference(
        #[values(0, 1, 7, 8, 9, 15, 16, 17, 64, 129)] len: usize,
    ) {
        let values: Vec<f32> = (0..len).map(|i| i as f32 - 3.5).collect();
        let signs = random_fast_rotation_signs(len);
        let signs = &signs[..sign_bytes_per_round(len)];
        let expected: Vec<u32> = values
            .iter()
            .enumerate()
            .map(|(i, v)| v.to_bits() ^ (u32::from((signs[i / 8] >> (i % 8)) & 1) << 31))
            .collect();
        for candidate in FLIP_SIGNS.candidates() {
            let Some(kernel) = candidate.kernel() else {
                continue;
            };
            let mut actual = values.clone();
            kernel.call(&mut actual, signs);
            let actual: Vec<u32> = actual.iter().map(|v| v.to_bits()).collect();
            assert_eq!(actual, expected, "{} len={len}", candidate.name());
        }
    }

    #[test]
    fn test_fast_rotation_preserves_shape() {
        let input = vec![1.0f32; 129];
        let mut output = vec![0.0f32; 129];
        let signs = random_fast_rotation_signs(129);
        apply_fast_rotation(&input, &mut output, &signs);
        assert_eq!(output.len(), 129);
    }
}
