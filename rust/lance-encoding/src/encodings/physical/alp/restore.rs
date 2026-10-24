// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! SIMD kernels for the ALP chunk restore.
//!
//! ALP decode spends most of its per-value time converting the unpacked
//! offsets back to floats and writing them to the output buffer. The scalar
//! loop issues one tiny 4/8-byte store per value, which neither vectorizes
//! the integer-to-float conversion nor saturates the store path. These
//! explicit AVX-512 / AVX2 (x86_64) and NEON (aarch64) kernels convert and
//! write whole vectors while preserving the exact IEEE operation sequence of
//! the scalar path: the divide path stays a single division and the multiply
//! path stays two multiplies, so output is bitwise identical.

use super::{F10_F32, F10_F64, IF10_F32, IF10_F64};

#[cfg(target_arch = "x86_64")]
use lance_core::utils::cpu::CpuFeatures;

/// Restore `f32` values from unpacked offsets (FOR mode).
pub(super) fn restore_chunk32(
    offsets: &[u32],
    base: i64,
    div: bool,
    e: usize,
    f: usize,
    out: &mut [u8],
) {
    #[cfg(target_arch = "x86_64")]
    {
        let host = CpuFeatures::host();
        if host.contains(CpuFeatures::AVX512F) {
            return unsafe { restore_chunk32_avx512(offsets, base, div, e, f, out) };
        }
        if host.contains(CpuFeatures::AVX2) {
            return unsafe { restore_chunk32_avx2(offsets, base, div, e, f, out) };
        }
        restore_chunk32_scalar(offsets, base, div, e, f, out)
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { restore_chunk32_neon(offsets, base, div, e, f, out) }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        restore_chunk32_scalar(offsets, base, div, e, f, out);
    }
}

/// Restore `f64` values from unpacked offsets (FOR mode).
pub(super) fn restore_chunk64(
    offsets: &[u64],
    base: i64,
    div: bool,
    e: usize,
    f: usize,
    out: &mut [u8],
) {
    #[cfg(target_arch = "x86_64")]
    {
        // `i64 -> f64` conversion needs AVX-512; older x86 falls back.
        let host = CpuFeatures::host();
        if host.contains(CpuFeatures::AVX512F) {
            return unsafe { restore_chunk64_avx512(offsets, base, div, e, f, out) };
        }
        restore_chunk64_scalar(offsets, base, div, e, f, out)
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { restore_chunk64_neon(offsets, base, div, e, f, out) }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        restore_chunk64_scalar(offsets, base, div, e, f, out);
    }
}

fn restore_chunk32_scalar(
    offsets: &[u32],
    base: i64,
    div: bool,
    e: usize,
    f: usize,
    out: &mut [u8],
) {
    // `(base as i32).wrapping_add(offset as i32)` is the low 32 bits of
    // `(base + offset as i64)`, which is what the original decode loop casts
    // to `i32`; both spellings yield identical values.
    let base = base as i32;
    for (i, &offset) in offsets.iter().enumerate() {
        let enc = base.wrapping_add(offset as i32);
        let value = if div {
            (enc as f32) / F10_F32[e]
        } else {
            (enc as f32) * F10_F32[f] * IF10_F32[e]
        };
        out[i * 4..i * 4 + 4].copy_from_slice(&value.to_bits().to_le_bytes());
    }
}

fn restore_chunk64_scalar(
    offsets: &[u64],
    base: i64,
    div: bool,
    e: usize,
    f: usize,
    out: &mut [u8],
) {
    for (i, &offset) in offsets.iter().enumerate() {
        let enc = base.wrapping_add(offset as i64);
        let value = if div {
            (enc as f64) / F10_F64[e]
        } else {
            (enc as f64) * F10_F64[f] * IF10_F64[e]
        };
        out[i * 8..i * 8 + 8].copy_from_slice(&value.to_bits().to_le_bytes());
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn restore_chunk32_avx2(
    offsets: &[u32],
    base: i64,
    div: bool,
    e: usize,
    f: usize,
    out: &mut [u8],
) {
    use std::arch::x86_64::*;

    let n = offsets.len();
    let base_v = _mm256_set1_epi32(base as i32);
    let mut i = 0;
    if div {
        let factor = _mm256_set1_ps(F10_F32[e]);
        while i + 8 <= n {
            let enc = _mm256_add_epi32(
                _mm256_loadu_si256(offsets.as_ptr().add(i) as *const __m256i),
                base_v,
            );
            let v = _mm256_cvtepi32_ps(enc);
            _mm256_storeu_si256(
                out.as_mut_ptr().add(i * 4) as *mut __m256i,
                _mm256_castps_si256(_mm256_div_ps(v, factor)),
            );
            i += 8;
        }
    } else {
        let factor = _mm256_set1_ps(F10_F32[f]);
        let inv = _mm256_set1_ps(IF10_F32[e]);
        while i + 8 <= n {
            let enc = _mm256_add_epi32(
                _mm256_loadu_si256(offsets.as_ptr().add(i) as *const __m256i),
                base_v,
            );
            let v = _mm256_cvtepi32_ps(enc);
            _mm256_storeu_si256(
                out.as_mut_ptr().add(i * 4) as *mut __m256i,
                _mm256_castps_si256(_mm256_mul_ps(_mm256_mul_ps(v, factor), inv)),
            );
            i += 8;
        }
    }
    restore_chunk32_scalar(&offsets[i..], base, div, e, f, &mut out[i * 4..]);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn restore_chunk32_avx512(
    offsets: &[u32],
    base: i64,
    div: bool,
    e: usize,
    f: usize,
    out: &mut [u8],
) {
    use std::arch::x86_64::*;

    let n = offsets.len();
    let base_v = _mm512_set1_epi32(base as i32);
    let mut i = 0;
    if div {
        let factor = _mm512_set1_ps(F10_F32[e]);
        while i + 16 <= n {
            let enc = _mm512_add_epi32(
                _mm512_loadu_si512(offsets.as_ptr().add(i) as *const __m512i),
                base_v,
            );
            let v = _mm512_cvtepi32_ps(enc);
            _mm512_storeu_si512(
                out.as_mut_ptr().add(i * 4) as *mut __m512i,
                _mm512_castps_si512(_mm512_div_ps(v, factor)),
            );
            i += 16;
        }
    } else {
        let factor = _mm512_set1_ps(F10_F32[f]);
        let inv = _mm512_set1_ps(IF10_F32[e]);
        while i + 16 <= n {
            let enc = _mm512_add_epi32(
                _mm512_loadu_si512(offsets.as_ptr().add(i) as *const __m512i),
                base_v,
            );
            let v = _mm512_cvtepi32_ps(enc);
            _mm512_storeu_si512(
                out.as_mut_ptr().add(i * 4) as *mut __m512i,
                _mm512_castps_si512(_mm512_mul_ps(_mm512_mul_ps(v, factor), inv)),
            );
            i += 16;
        }
    }
    restore_chunk32_scalar(&offsets[i..], base, div, e, f, &mut out[i * 4..]);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn restore_chunk32_neon(
    offsets: &[u32],
    base: i64,
    div: bool,
    e: usize,
    f: usize,
    out: &mut [u8],
) {
    use std::arch::aarch64::*;

    let n = offsets.len();
    let base_v = vdupq_n_s32(base as i32);
    let mut i = 0;
    if div {
        let factor = vdupq_n_f32(F10_F32[e]);
        while i + 4 <= n {
            let enc = vaddq_s32(
                vreinterpretq_s32_u32(vld1q_u32(offsets.as_ptr().add(i))),
                base_v,
            );
            let v = vcvtq_f32_s32(enc);
            vst1q_u32(
                out.as_mut_ptr().add(i * 4) as *mut u32,
                vreinterpretq_u32_f32(vdivq_f32(v, factor)),
            );
            i += 4;
        }
    } else {
        let factor = vdupq_n_f32(F10_F32[f]);
        let inv = vdupq_n_f32(IF10_F32[e]);
        while i + 4 <= n {
            let enc = vaddq_s32(
                vreinterpretq_s32_u32(vld1q_u32(offsets.as_ptr().add(i))),
                base_v,
            );
            let v = vcvtq_f32_s32(enc);
            vst1q_u32(
                out.as_mut_ptr().add(i * 4) as *mut u32,
                vreinterpretq_u32_f32(vmulq_f32(vmulq_f32(v, factor), inv)),
            );
            i += 4;
        }
    }
    restore_chunk32_scalar(&offsets[i..], base, div, e, f, &mut out[i * 4..]);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn restore_chunk64_avx512(
    offsets: &[u64],
    base: i64,
    div: bool,
    e: usize,
    f: usize,
    out: &mut [u8],
) {
    use std::arch::x86_64::*;

    let n = offsets.len();
    let base_v = _mm512_set1_epi64(base);
    let mut i = 0;
    if div {
        let factor = _mm512_set1_pd(F10_F64[e]);
        while i + 8 <= n {
            let enc = _mm512_add_epi64(
                _mm512_loadu_si512(offsets.as_ptr().add(i) as *const __m512i),
                base_v,
            );
            let v = _mm512_cvtepi64_pd(enc);
            _mm512_storeu_si512(
                out.as_mut_ptr().add(i * 8) as *mut __m512i,
                _mm512_castpd_si512(_mm512_div_pd(v, factor)),
            );
            i += 8;
        }
    } else {
        let factor = _mm512_set1_pd(F10_F64[f]);
        let inv = _mm512_set1_pd(IF10_F64[e]);
        while i + 8 <= n {
            let enc = _mm512_add_epi64(
                _mm512_loadu_si512(offsets.as_ptr().add(i) as *const __m512i),
                base_v,
            );
            let v = _mm512_cvtepi64_pd(enc);
            _mm512_storeu_si512(
                out.as_mut_ptr().add(i * 8) as *mut __m512i,
                _mm512_castpd_si512(_mm512_mul_pd(_mm512_mul_pd(v, factor), inv)),
            );
            i += 8;
        }
    }
    restore_chunk64_scalar(&offsets[i..], base, div, e, f, &mut out[i * 8..]);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn restore_chunk64_neon(
    offsets: &[u64],
    base: i64,
    div: bool,
    e: usize,
    f: usize,
    out: &mut [u8],
) {
    use std::arch::aarch64::*;

    let n = offsets.len();
    let base_v = vdupq_n_s64(base);
    let mut i = 0;
    if div {
        let factor = vdupq_n_f64(F10_F64[e]);
        // NEON `float64x2_t` has two lanes; process two values at a time.
        while i + 2 <= n {
            let enc = vaddq_s64(
                vreinterpretq_s64_u64(vld1q_u64(offsets.as_ptr().add(i))),
                base_v,
            );
            let v = vcvtq_f64_s64(enc);
            vst1q_u64(
                out.as_mut_ptr().add(i * 8) as *mut u64,
                vreinterpretq_u64_f64(vdivq_f64(v, factor)),
            );
            i += 2;
        }
    } else {
        let factor = vdupq_n_f64(F10_F64[f]);
        let inv = vdupq_n_f64(IF10_F64[e]);
        while i + 2 <= n {
            let enc = vaddq_s64(
                vreinterpretq_s64_u64(vld1q_u64(offsets.as_ptr().add(i))),
                base_v,
            );
            let v = vcvtq_f64_s64(enc);
            vst1q_u64(
                out.as_mut_ptr().add(i * 8) as *mut u64,
                vreinterpretq_u64_f64(vmulq_f64(vmulq_f64(v, factor), inv)),
            );
            i += 2;
        }
    }
    restore_chunk64_scalar(&offsets[i..], base, div, e, f, &mut out[i * 8..]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn assert_restore32_equivalent(kernel: unsafe fn(&[u32], i64, bool, usize, usize, &mut [u8])) {
        let mut rng = StdRng::seed_from_u64(0x00de_c0de_2026_1024);
        for &(div, e, f) in &[
            (true, 1, 0),
            (true, 5, 0),
            (true, 10, 0),
            (false, 5, 3),
            (false, 0, 7),
            (false, 10, 10),
        ] {
            for &n in &[0usize, 1, 3, 4, 7, 8, 15, 16, 17, 31, 63, 64, 65, 129, 1024] {
                let offsets: Vec<u32> = (0..n).map(|_| rng.random()).collect();
                let base = rng.random_range(i32::MIN as i64..=i32::MAX as i64);
                let mut expected = vec![0u8; n * 4];
                let mut got = vec![0u8; n * 4];
                restore_chunk32_scalar(&offsets, base, div, e, f, &mut expected);
                unsafe { kernel(&offsets, base, div, e, f, &mut got) };
                assert_eq!(
                    got, expected,
                    "restore32 mismatch div={div} e={e} f={f} n={n} base={base}"
                );
            }
        }
    }

    fn assert_restore64_equivalent(kernel: unsafe fn(&[u64], i64, bool, usize, usize, &mut [u8])) {
        let mut rng = StdRng::seed_from_u64(0x00de_c0de_2026_1025);
        for &(div, e, f) in &[
            (true, 1, 0),
            (true, 9, 0),
            (true, 18, 0),
            (false, 5, 3),
            (false, 0, 7),
            (false, 18, 18),
        ] {
            for &n in &[0usize, 1, 3, 4, 7, 8, 15, 16, 17, 31, 63, 64, 65, 129, 1024] {
                let offsets: Vec<u64> = (0..n).map(|_| rng.random()).collect();
                let base = rng.random_range(i64::MIN / 4..=i64::MAX / 4);
                let mut expected = vec![0u8; n * 8];
                let mut got = vec![0u8; n * 8];
                restore_chunk64_scalar(&offsets, base, div, e, f, &mut expected);
                unsafe { kernel(&offsets, base, div, e, f, &mut got) };
                assert_eq!(
                    got, expected,
                    "restore64 mismatch div={div} e={e} f={f} n={n} base={base}"
                );
            }
        }
    }

    /// The runtime-dispatched host kernel must agree with the scalar
    /// fallback for every lane and tail element.
    #[test]
    fn test_restore_matches_scalar() {
        assert_restore32_equivalent(restore_chunk32);
        assert_restore64_equivalent(restore_chunk64);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        assert_restore32_equivalent(restore_chunk32_avx2);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx512_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        assert_restore32_equivalent(restore_chunk32_avx512);
        assert_restore64_equivalent(restore_chunk64_avx512);
    }
}
