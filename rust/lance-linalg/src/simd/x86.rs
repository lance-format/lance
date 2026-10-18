// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reduction helpers shared by the x86_64 AVX kernels in [`crate::distance`].
//!
//! Each distance kernel accumulates into a 256-bit register and folds it down
//! to a scalar once, after the main loop. The fold is identical across
//! `cosine`, `dot`, `l2` and `norm_l2`, so it lives here instead of being
//! copied into each kernel's private `mod x86`. So do the batch loops that
//! the `dot` and `l2` batch kernels share.
//!
//! The module itself is `pub(crate)`, which is what keeps these helpers off the
//! public API; the items are `pub` rather than `pub(crate)` only because
//! `clippy::redundant_pub_crate` fires on the narrower visibility.

use std::arch::x86_64::*;

/// Horizontal sum of the eight `f32` lanes of an `__m256`.
///
/// Folds the upper 128-bit lane into the lower one, then reduces the
/// remaining four lanes pairwise. Uses `movehl`/`shuffle` plus scalar adds
/// rather than two `vhaddps`, which is one fewer uop on most cores.
///
/// # Safety
///
/// The host must support AVX. Callers are `#[target_feature]`-annotated
/// kernels that the runtime dispatcher only selects after checking.
#[inline]
#[target_feature(enable = "avx")]
pub unsafe fn hsum256_ps(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    // 0x55 broadcasts lane 1 into lane 0, so the scalar add below lands the
    // last of the four partial sums.
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55));
    _mm_cvtss_f32(sum32)
}

/// [`hsum256_ps`] of eight vectors at once: lane `i` of the result is
/// `hsum256_ps(v[i])`, bit for bit.
///
/// The eight registers are transposed while they are reduced, so the adds
/// pair lanes in the same order as [`hsum256_ps`] (`l + l+4`, then `l + l+2`,
/// then `l + l+1`), with 14 shuffles for eight sums instead of 24, and the
/// result is stored with one write instead of eight.
///
/// # Safety
///
/// The host must support AVX.
#[inline]
#[target_feature(enable = "avx")]
pub unsafe fn hsum256_ps_x8(v: [__m256; 8]) -> __m256 {
    // `[v[i].lo + v[i].hi | v[i+4].lo + v[i+4].hi]`, the first fold of
    // `hsum256_ps` for vectors i and i+4.
    let fold = |a: __m256, b: __m256| {
        _mm256_add_ps(
            _mm256_permute2f128_ps(a, b, 0x20),
            _mm256_permute2f128_ps(a, b, 0x31),
        )
    };
    let p0 = fold(v[0], v[4]);
    let p1 = fold(v[1], v[5]);
    let p2 = fold(v[2], v[6]);
    let p3 = fold(v[3], v[7]);
    // Per 128-bit half, `[a0 + a2, a1 + a3, b0 + b2, b1 + b3]`: the
    // `movehl` fold of `hsum256_ps`.
    let pair = |a: __m256, b: __m256| {
        _mm256_add_ps(
            _mm256_shuffle_ps(a, b, 0b01_00_01_00),
            _mm256_shuffle_ps(a, b, 0b11_10_11_10),
        )
    };
    let q01 = pair(p0, p1);
    let q23 = pair(p2, p3);
    // Lane 0 plus lane 1 of each vector's pair: its final scalar add.
    _mm256_add_ps(
        _mm256_shuffle_ps(q01, q23, 0b10_00_10_00),
        _mm256_shuffle_ps(q01, q23, 0b11_01_11_01),
    )
}

/// Horizontal sum of the four `f64` lanes of an `__m256d`.
///
/// Folds the upper 128-bit lane into the lower one, then adds the remaining
/// pair.
///
/// # Safety
///
/// The host must support AVX. Callers are `#[target_feature]`-annotated
/// kernels that the runtime dispatcher only selects after checking.
#[inline]
#[target_feature(enable = "avx")]
pub unsafe fn hsum256_pd(v: __m256d) -> f64 {
    let lo = _mm256_castpd256_pd128(v);
    let hi = _mm256_extractf128_pd(v, 1);
    let sum128 = _mm_add_pd(lo, hi);
    let sum64 = _mm_add_pd(sum128, _mm_unpackhi_pd(sum128, sum128));
    _mm_cvtsd_f64(sum64)
}

/// Writes `hsum256_ps(acc) + tail` for each `D`-wide vector `y` of `batch`,
/// where `(acc, tail) == parts(y)`, into the matching element of `output`:
/// the result of a pair kernel that ends in `hsum256_ps`.
///
/// Batch kernels call this once per batch for each common PQ sub-vector
/// width. With `D` constant, `parts` inlines fully unrolled with no tail, and
/// each width gets a loop, and a register allocation, of its own: sharing one
/// loop across widths left its invariants spilled to the stack.
/// Eight vectors at a time share one [`hsum256_ps_x8`], which sums each
/// accumulator in the same order as `hsum256_ps`, and one store, so the
/// result is unchanged.
#[inline]
#[target_feature(enable = "avx")]
pub fn fixed_width_batch<const D: usize>(
    batch: &[f32],
    output: &mut [f32],
    parts: impl Fn(&[f32; D]) -> (__m256, f32),
) {
    let (vectors, _) = batch.as_chunks::<D>();
    let len = vectors.len().min(output.len());
    let (vector_blocks, vectors_rest) = vectors[..len].as_chunks::<8>();
    let (output_blocks, output_rest) = output[..len].as_chunks_mut::<8>();
    for (distances, block) in output_blocks.iter_mut().zip(vector_blocks) {
        // A plain loop rather than `array::map`: `map` has no target
        // features, so unless AVX is in the build's baseline it cannot
        // inline `parts`, and each call then returns its `__m256` through
        // memory (+100-250% on x86-64-v2 builds).
        let mut accs = [_mm256_setzero_ps(); 8];
        let mut tails = [0.0f32; 8];
        for ((acc, tail), y) in accs.iter_mut().zip(&mut tails).zip(block) {
            (*acc, *tail) = parts(y);
        }
        // SAFETY: AVX is enabled here, and `tails` and `distances` hold
        // eight f32s each.
        unsafe {
            let sums = hsum256_ps_x8(accs);
            let distance = _mm256_add_ps(sums, _mm256_loadu_ps(tails.as_ptr()));
            _mm256_storeu_ps(distances.as_mut_ptr(), distance);
        }
    }
    for (distance, y) in output_rest.iter_mut().zip(vectors_rest) {
        let (acc, tail) = parts(y);
        // SAFETY: AVX is enabled here.
        *distance = unsafe { hsum256_ps(acc) } + tail;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::ascending([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 36.0)]
    #[case::negative_lanes([-1.5, 2.0, -3.0, 4.5, 0.0, -0.5, 1.0, 2.5], 5.0)]
    #[case::zeros([0.0; 8], 0.0)]
    #[case::cancelling([1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0], 0.0)]
    fn hsum256_ps_sums_every_lane(#[case] lanes: [f32; 8], #[case] expected: f32) {
        if !std::is_x86_feature_detected!("avx") {
            return;
        }
        let sum = unsafe { hsum256_ps(_mm256_loadu_ps(lanes.as_ptr())) };
        assert_eq!(sum, expected);
    }

    /// Lanes whose sum depends on the order of the adds (cancellation,
    /// mixed magnitudes, signed zeros), so any other pairing changes bits.
    #[test]
    fn hsum256_ps_x8_matches_hsum256_ps_bitwise() {
        if !std::is_x86_feature_detected!("avx") {
            return;
        }
        let pool = [
            1.0e8, -1.0e8, 1.0, -0.0, 0.0, 3.5e-3, -7.25, 1.0e-30, 2.0e7, 0.1, -0.3, 1.0e30,
        ];
        for offset in 0..pool.len() {
            let lanes: [[f32; 8]; 8] = std::array::from_fn(|v| {
                std::array::from_fn(|l| pool[(offset + v * 5 + l * 7) % pool.len()])
            });
            let mut got = [0.0f32; 8];
            unsafe {
                let v = lanes.map(|l| _mm256_loadu_ps(l.as_ptr()));
                _mm256_storeu_ps(got.as_mut_ptr(), hsum256_ps_x8(v));
            }
            for (i, l) in lanes.iter().enumerate() {
                let want = unsafe { hsum256_ps(_mm256_loadu_ps(l.as_ptr())) };
                assert_eq!(
                    got[i].to_bits(),
                    want.to_bits(),
                    "vector {i} of offset {offset}: {} != {want}",
                    got[i]
                );
            }
        }
    }

    /// Every output is `hsum256_ps(acc) + tail` of its own vector, through
    /// the eight-vector blocks and the remainder alike. A width of 12 gives
    /// every vector a nonzero tail.
    #[test]
    fn fixed_width_batch_adds_each_vectors_tail() {
        if !std::is_x86_feature_detected!("avx") {
            return;
        }
        let parts = |y: &[f32; 12]| {
            let acc = unsafe { _mm256_loadu_ps(y.as_ptr()) };
            (acc, y[8..].iter().sum::<f32>())
        };
        let batch = (0..12 * 19)
            .map(|i| ((i * 7919) % 101) as f32 * 0.37 - 17.0)
            .collect::<Vec<_>>();
        let mut got = vec![f32::NAN; 19];
        unsafe { fixed_width_batch(&batch, &mut got, parts) };
        for (i, y) in batch.as_chunks::<12>().0.iter().enumerate() {
            let (acc, tail) = parts(y);
            let want = unsafe { hsum256_ps(acc) } + tail;
            assert_eq!(
                got[i].to_bits(),
                want.to_bits(),
                "vector {i}: {} != {want}",
                got[i]
            );
        }
    }

    #[rstest]
    #[case::ascending([1.0, 2.0, 3.0, 4.0], 10.0)]
    #[case::negative_lanes([-1.5, 2.0, -3.0, 4.5], 2.0)]
    #[case::zeros([0.0; 4], 0.0)]
    #[case::cancelling([1.0, -1.0, 2.0, -2.0], 0.0)]
    fn hsum256_pd_sums_every_lane(#[case] lanes: [f64; 4], #[case] expected: f64) {
        if !std::is_x86_feature_detected!("avx") {
            return;
        }
        let sum = unsafe { hsum256_pd(_mm256_loadu_pd(lanes.as_ptr())) };
        assert_eq!(sum, expected);
    }
}
