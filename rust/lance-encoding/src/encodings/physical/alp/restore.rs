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

/// Right-shift-by-k index vectors for the AVX-512 16-lane inclusive scans.
/// `IDX[k][i] = i - k` for `i >= k`; the masked permute zeroes lanes `i < k`,
/// which is the zero-filled shift the Hillis-Steele steps need. On non-x86
/// targets the tables exist only so the scan emulation test can exercise them.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
const AVX512_SCAN_IDX: [[i32; 16]; 4] = [
    [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7],
];

/// Right-shift-by-k index vectors for the AVX-512 8-lane and AVX2 8-lane
/// inclusive scans; see [`AVX512_SCAN_IDX`].
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
const AVX8_SCAN_IDX: [[i64; 8]; 3] = [
    [0, 0, 1, 2, 3, 4, 5, 6],
    [0, 0, 0, 1, 2, 3, 4, 5],
    [0, 0, 0, 0, 0, 1, 2, 3],
];

/// `i32` spelling of [`AVX8_SCAN_IDX`] for the AVX2 8-lane scan, which loads
/// a 32-byte index vector; the `i64` table must not be byte-reinterpreted
/// for it.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
const AVX8_SCAN_IDX_I32: [[i32; 8]; 3] = [
    [0, 0, 1, 2, 3, 4, 5, 6],
    [0, 0, 0, 1, 2, 3, 4, 5],
    [0, 0, 0, 0, 0, 1, 2, 3],
];

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
        // `i64 -> f64` conversion needs AVX-512DQ as well as AVX-512F.
        let host = CpuFeatures::host();
        if host.contains(CpuFeatures::AVX512F.union(CpuFeatures::AVX512DQ)) {
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

/// Delta-mode restore for `f32`, combining the prefix scan and the
/// convert-and-store pass.
///
/// The decoder's chain is `cursor = cursor.wrapping_add(offset + delta_base)`
/// over `offsets.len()` increments, emitting `offsets.len() + 1` values. The
/// scalar loop serializes one wrap-add dependency per value, so these
/// entry points dispatch to kernels that scan a block of offsets with a
/// Hillis-Steele SIMD prefix (log2(lanes) serial steps) while carrying only
/// the block total across blocks; whole vectors are converted and stored
/// straight to the output, with no temporary buffer. Every step is modulo
/// 2^32/2^64, matching the scalar chain bit-for-bit.
pub(super) fn restore_delta_chunk32(
    offsets: &[u32],
    base: i64,
    delta_base: i64,
    div: bool,
    e: usize,
    f: usize,
    out: &mut [u8],
) {
    #[cfg(target_arch = "x86_64")]
    {
        let host = CpuFeatures::host();
        if host.contains(CpuFeatures::AVX512F) {
            return unsafe {
                restore_delta_chunk32_avx512(offsets, base, delta_base, div, e, f, out)
            };
        }
        if host.contains(CpuFeatures::AVX2) {
            return unsafe {
                restore_delta_chunk32_avx2(offsets, base, delta_base, div, e, f, out)
            };
        }
        restore_delta_chunk32_scalar(offsets, base, delta_base, div, e, f, out)
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { restore_delta_chunk32_neon(offsets, base, delta_base, div, e, f, out) }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        restore_delta_chunk32_scalar(offsets, base, delta_base, div, e, f, out);
    }
}

/// Delta-mode restore for `f64`; see [`restore_delta_chunk32`].
pub(super) fn restore_delta_chunk64(
    offsets: &[u64],
    base: i64,
    delta_base: i64,
    div: bool,
    e: usize,
    f: usize,
    out: &mut [u8],
) {
    #[cfg(target_arch = "x86_64")]
    {
        // `i64 -> f64` conversion needs AVX-512DQ as well as AVX-512F.
        let host = CpuFeatures::host();
        if host.contains(CpuFeatures::AVX512F.union(CpuFeatures::AVX512DQ)) {
            return unsafe {
                restore_delta_chunk64_avx512(offsets, base, delta_base, div, e, f, out)
            };
        }
        restore_delta_chunk64_scalar(offsets, base, delta_base, div, e, f, out)
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { restore_delta_chunk64_neon(offsets, base, delta_base, div, e, f, out) }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        restore_delta_chunk64_scalar(offsets, base, delta_base, div, e, f, out);
    }
}

fn restore_delta_chunk32_scalar(
    offsets: &[u32],
    base: i64,
    delta_base: i64,
    div: bool,
    e: usize,
    f: usize,
    out: &mut [u8],
) {
    // `base as i32` is the low 32 bits of `(base + 0) as i32`; the loop emits
    // the current value before consuming its offset, so the final write lands
    // at position `offsets.len()` (one value per offset plus the base).
    let mut cursor = base as i32;
    let delta_base = delta_base as i32;
    for (i, &offset) in offsets.iter().enumerate() {
        let value = if div {
            (cursor as f32) / F10_F32[e]
        } else {
            (cursor as f32) * F10_F32[f] * IF10_F32[e]
        };
        out[i * 4..i * 4 + 4].copy_from_slice(&value.to_bits().to_le_bytes());
        cursor = cursor.wrapping_add(offset as i32).wrapping_add(delta_base);
    }
    let value = if div {
        (cursor as f32) / F10_F32[e]
    } else {
        (cursor as f32) * F10_F32[f] * IF10_F32[e]
    };
    out[offsets.len() * 4..].copy_from_slice(&value.to_bits().to_le_bytes());
}

fn restore_delta_chunk64_scalar(
    offsets: &[u64],
    base: i64,
    delta_base: i64,
    div: bool,
    e: usize,
    f: usize,
    out: &mut [u8],
) {
    let mut cursor = base;
    for (i, &offset) in offsets.iter().enumerate() {
        let value = if div {
            (cursor as f64) / F10_F64[e]
        } else {
            (cursor as f64) * F10_F64[f] * IF10_F64[e]
        };
        out[i * 8..i * 8 + 8].copy_from_slice(&value.to_bits().to_le_bytes());
        cursor = cursor.wrapping_add(offset as i64).wrapping_add(delta_base);
    }
    let value = if div {
        (cursor as f64) / F10_F64[e]
    } else {
        (cursor as f64) * F10_F64[f] * IF10_F64[e]
    };
    out[offsets.len() * 8..].copy_from_slice(&value.to_bits().to_le_bytes());
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn restore_delta_chunk32_avx2(
    offsets: &[u32],
    base: i64,
    delta_base: i64,
    div: bool,
    e: usize,
    f: usize,
    out: &mut [u8],
) {
    use std::arch::x86_64::*;

    // `_mm256_permutevar8x32_epi32` duplicates the left edge of the vector;
    // a blend then zeroes the lanes that wrapped across the edge, giving a
    // proper zero-filled shift for the scan step.
    let idx1 = _mm256_loadu_si256(AVX8_SCAN_IDX_I32[0].as_ptr() as *const __m256i);
    let idx2 = _mm256_loadu_si256(AVX8_SCAN_IDX_I32[1].as_ptr() as *const __m256i);
    let idx4 = _mm256_loadu_si256(AVX8_SCAN_IDX_I32[2].as_ptr() as *const __m256i);

    let n = offsets.len();
    let x_adj = _mm256_set1_epi32(delta_base as i32);
    let zero = _mm256_setzero_si256();
    let mut carry = base as i32;
    let mut i = 0;
    if div {
        let factor = _mm256_set1_ps(F10_F32[e]);
        while i + 8 <= n {
            let x = _mm256_add_epi32(
                _mm256_loadu_si256(offsets.as_ptr().add(i) as *const __m256i),
                x_adj,
            );
            let mut s = x;
            let shifted = _mm256_permutevar8x32_epi32(s, idx1);
            s = _mm256_add_epi32(s, _mm256_blend_epi32(shifted, zero, 0x01));
            let shifted = _mm256_permutevar8x32_epi32(s, idx2);
            s = _mm256_add_epi32(s, _mm256_blend_epi32(shifted, zero, 0x03));
            let shifted = _mm256_permutevar8x32_epi32(s, idx4);
            s = _mm256_add_epi32(s, _mm256_blend_epi32(shifted, zero, 0x0F));
            // `s - x` is the exclusive prefix; lane 7 holds the block total.
            let enc = _mm256_add_epi32(_mm256_sub_epi32(s, x), _mm256_set1_epi32(carry));
            let v = _mm256_cvtepi32_ps(enc);
            _mm256_storeu_si256(
                out.as_mut_ptr().add(i * 4) as *mut __m256i,
                _mm256_castps_si256(_mm256_div_ps(v, factor)),
            );
            carry = carry.wrapping_add(_mm256_extract_epi32(s, 7));
            i += 8;
        }
    } else {
        let factor = _mm256_set1_ps(F10_F32[f]);
        let inv = _mm256_set1_ps(IF10_F32[e]);
        while i + 8 <= n {
            let x = _mm256_add_epi32(
                _mm256_loadu_si256(offsets.as_ptr().add(i) as *const __m256i),
                x_adj,
            );
            let mut s = x;
            let shifted = _mm256_permutevar8x32_epi32(s, idx1);
            s = _mm256_add_epi32(s, _mm256_blend_epi32(shifted, zero, 0x01));
            let shifted = _mm256_permutevar8x32_epi32(s, idx2);
            s = _mm256_add_epi32(s, _mm256_blend_epi32(shifted, zero, 0x03));
            let shifted = _mm256_permutevar8x32_epi32(s, idx4);
            s = _mm256_add_epi32(s, _mm256_blend_epi32(shifted, zero, 0x0F));
            let enc = _mm256_add_epi32(_mm256_sub_epi32(s, x), _mm256_set1_epi32(carry));
            let v = _mm256_cvtepi32_ps(enc);
            _mm256_storeu_si256(
                out.as_mut_ptr().add(i * 4) as *mut __m256i,
                _mm256_castps_si256(_mm256_mul_ps(_mm256_mul_ps(v, factor), inv)),
            );
            carry = carry.wrapping_add(_mm256_extract_epi32(s, 7));
            i += 8;
        }
    }
    restore_delta_chunk32_scalar(
        &offsets[i..],
        carry as i64,
        delta_base,
        div,
        e,
        f,
        &mut out[i * 4..],
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn restore_delta_chunk32_avx512(
    offsets: &[u32],
    base: i64,
    delta_base: i64,
    div: bool,
    e: usize,
    f: usize,
    out: &mut [u8],
) {
    use std::arch::x86_64::*;

    // Masked permute zeroes the lanes that wrap across the left edge.
    let idx1 = _mm512_loadu_si512(AVX512_SCAN_IDX[0].as_ptr() as *const __m512i);
    let idx2 = _mm512_loadu_si512(AVX512_SCAN_IDX[1].as_ptr() as *const __m512i);
    let idx4 = _mm512_loadu_si512(AVX512_SCAN_IDX[2].as_ptr() as *const __m512i);
    let idx8 = _mm512_loadu_si512(AVX512_SCAN_IDX[3].as_ptr() as *const __m512i);

    let n = offsets.len();
    let x_adj = _mm512_set1_epi32(delta_base as i32);
    let mut carry = base as i32;
    let mut i = 0;
    if div {
        let factor = _mm512_set1_ps(F10_F32[e]);
        while i + 16 <= n {
            let x = _mm512_add_epi32(
                _mm512_loadu_si512(offsets.as_ptr().add(i) as *const __m512i),
                x_adj,
            );
            let mut s = x;
            s = _mm512_add_epi32(s, _mm512_maskz_permutexvar_epi32(0xFFFE, idx1, s));
            s = _mm512_add_epi32(s, _mm512_maskz_permutexvar_epi32(0xFFFC, idx2, s));
            s = _mm512_add_epi32(s, _mm512_maskz_permutexvar_epi32(0xFFF0, idx4, s));
            s = _mm512_add_epi32(s, _mm512_maskz_permutexvar_epi32(0xFF00, idx8, s));
            // `s - x` is the exclusive prefix; lane 15 holds the block total.
            let enc = _mm512_add_epi32(_mm512_sub_epi32(s, x), _mm512_set1_epi32(carry));
            let v = _mm512_cvtepi32_ps(enc);
            _mm512_storeu_si512(
                out.as_mut_ptr().add(i * 4) as *mut __m512i,
                _mm512_castps_si512(_mm512_div_ps(v, factor)),
            );
            let total = _mm_extract_epi32(_mm512_extracti32x4_epi32(s, 3), 3);
            carry = carry.wrapping_add(total);
            i += 16;
        }
    } else {
        let factor = _mm512_set1_ps(F10_F32[f]);
        let inv = _mm512_set1_ps(IF10_F32[e]);
        while i + 16 <= n {
            let x = _mm512_add_epi32(
                _mm512_loadu_si512(offsets.as_ptr().add(i) as *const __m512i),
                x_adj,
            );
            let mut s = x;
            s = _mm512_add_epi32(s, _mm512_maskz_permutexvar_epi32(0xFFFE, idx1, s));
            s = _mm512_add_epi32(s, _mm512_maskz_permutexvar_epi32(0xFFFC, idx2, s));
            s = _mm512_add_epi32(s, _mm512_maskz_permutexvar_epi32(0xFFF0, idx4, s));
            s = _mm512_add_epi32(s, _mm512_maskz_permutexvar_epi32(0xFF00, idx8, s));
            let enc = _mm512_add_epi32(_mm512_sub_epi32(s, x), _mm512_set1_epi32(carry));
            let v = _mm512_cvtepi32_ps(enc);
            _mm512_storeu_si512(
                out.as_mut_ptr().add(i * 4) as *mut __m512i,
                _mm512_castps_si512(_mm512_mul_ps(_mm512_mul_ps(v, factor), inv)),
            );
            let total = _mm_extract_epi32(_mm512_extracti32x4_epi32(s, 3), 3);
            carry = carry.wrapping_add(total);
            i += 16;
        }
    }
    restore_delta_chunk32_scalar(
        &offsets[i..],
        carry as i64,
        delta_base,
        div,
        e,
        f,
        &mut out[i * 4..],
    );
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn restore_delta_chunk32_neon(
    offsets: &[u32],
    base: i64,
    delta_base: i64,
    div: bool,
    e: usize,
    f: usize,
    out: &mut [u8],
) {
    use std::arch::aarch64::*;

    // `vextq(zero, v, 4 - k)` shifts the 4 lanes right by k with zero fill.
    let x_adj = vdupq_n_s32(delta_base as i32);
    let zero = vdupq_n_s32(0);
    let mut carry = base as i32;
    let n = offsets.len();
    let mut i = 0;
    if div {
        let factor = vdupq_n_f32(F10_F32[e]);
        while i + 4 <= n {
            let x = vaddq_s32(
                vreinterpretq_s32_u32(vld1q_u32(offsets.as_ptr().add(i))),
                x_adj,
            );
            let mut s = x;
            s = vaddq_s32(s, vextq_s32(zero, s, 3));
            s = vaddq_s32(s, vextq_s32(zero, s, 2));
            // `s - x` is the exclusive prefix; lane 3 holds the block total.
            let enc = vaddq_s32(vsubq_s32(s, x), vdupq_n_s32(carry));
            let v = vcvtq_f32_s32(enc);
            vst1q_u32(
                out.as_mut_ptr().add(i * 4) as *mut u32,
                vreinterpretq_u32_f32(vdivq_f32(v, factor)),
            );
            carry = carry.wrapping_add(vgetq_lane_s32(s, 3));
            i += 4;
        }
    } else {
        let factor = vdupq_n_f32(F10_F32[f]);
        let inv = vdupq_n_f32(IF10_F32[e]);
        while i + 4 <= n {
            let x = vaddq_s32(
                vreinterpretq_s32_u32(vld1q_u32(offsets.as_ptr().add(i))),
                x_adj,
            );
            let mut s = x;
            s = vaddq_s32(s, vextq_s32(zero, s, 3));
            s = vaddq_s32(s, vextq_s32(zero, s, 2));
            let enc = vaddq_s32(vsubq_s32(s, x), vdupq_n_s32(carry));
            let v = vcvtq_f32_s32(enc);
            vst1q_u32(
                out.as_mut_ptr().add(i * 4) as *mut u32,
                vreinterpretq_u32_f32(vmulq_f32(vmulq_f32(v, factor), inv)),
            );
            carry = carry.wrapping_add(vgetq_lane_s32(s, 3));
            i += 4;
        }
    }
    restore_delta_chunk32_scalar(
        &offsets[i..],
        carry as i64,
        delta_base,
        div,
        e,
        f,
        &mut out[i * 4..],
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn restore_delta_chunk64_avx512(
    offsets: &[u64],
    base: i64,
    delta_base: i64,
    div: bool,
    e: usize,
    f: usize,
    out: &mut [u8],
) {
    use std::arch::x86_64::*;

    let idx1 = _mm512_loadu_si512(AVX8_SCAN_IDX[0].as_ptr() as *const __m512i);
    let idx2 = _mm512_loadu_si512(AVX8_SCAN_IDX[1].as_ptr() as *const __m512i);
    let idx4 = _mm512_loadu_si512(AVX8_SCAN_IDX[2].as_ptr() as *const __m512i);

    let n = offsets.len();
    let x_adj = _mm512_set1_epi64(delta_base);
    let mut carry = base;
    let mut i = 0;
    if div {
        let factor = _mm512_set1_pd(F10_F64[e]);
        while i + 8 <= n {
            let x = _mm512_add_epi64(
                _mm512_loadu_si512(offsets.as_ptr().add(i) as *const __m512i),
                x_adj,
            );
            let mut s = x;
            s = _mm512_add_epi64(s, _mm512_maskz_permutexvar_epi64(0xFE, idx1, s));
            s = _mm512_add_epi64(s, _mm512_maskz_permutexvar_epi64(0xFC, idx2, s));
            s = _mm512_add_epi64(s, _mm512_maskz_permutexvar_epi64(0xF0, idx4, s));
            // `s - x` is the exclusive prefix; lane 7 holds the block total.
            let enc = _mm512_add_epi64(_mm512_sub_epi64(s, x), _mm512_set1_epi64(carry));
            let v = _mm512_cvtepi64_pd(enc);
            _mm512_storeu_si512(
                out.as_mut_ptr().add(i * 8) as *mut __m512i,
                _mm512_castpd_si512(_mm512_div_pd(v, factor)),
            );
            let total = _mm256_extract_epi64(_mm512_extracti64x4_epi64(s, 1), 3);
            carry = carry.wrapping_add(total);
            i += 8;
        }
    } else {
        let factor = _mm512_set1_pd(F10_F64[f]);
        let inv = _mm512_set1_pd(IF10_F64[e]);
        while i + 8 <= n {
            let x = _mm512_add_epi64(
                _mm512_loadu_si512(offsets.as_ptr().add(i) as *const __m512i),
                x_adj,
            );
            let mut s = x;
            s = _mm512_add_epi64(s, _mm512_maskz_permutexvar_epi64(0xFE, idx1, s));
            s = _mm512_add_epi64(s, _mm512_maskz_permutexvar_epi64(0xFC, idx2, s));
            s = _mm512_add_epi64(s, _mm512_maskz_permutexvar_epi64(0xF0, idx4, s));
            let enc = _mm512_add_epi64(_mm512_sub_epi64(s, x), _mm512_set1_epi64(carry));
            let v = _mm512_cvtepi64_pd(enc);
            _mm512_storeu_si512(
                out.as_mut_ptr().add(i * 8) as *mut __m512i,
                _mm512_castpd_si512(_mm512_mul_pd(_mm512_mul_pd(v, factor), inv)),
            );
            let total = _mm256_extract_epi64(_mm512_extracti64x4_epi64(s, 1), 3);
            carry = carry.wrapping_add(total);
            i += 8;
        }
    }
    restore_delta_chunk64_scalar(
        &offsets[i..],
        carry,
        delta_base,
        div,
        e,
        f,
        &mut out[i * 8..],
    );
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn restore_delta_chunk64_neon(
    offsets: &[u64],
    base: i64,
    delta_base: i64,
    div: bool,
    e: usize,
    f: usize,
    out: &mut [u8],
) {
    use std::arch::aarch64::*;

    // `vextq(zero, v, 1)` shifts the 2 lanes right by 1 with zero fill.
    let x_adj = vdupq_n_s64(delta_base);
    let zero = vdupq_n_s64(0);
    let mut carry = base;
    let n = offsets.len();
    let mut i = 0;
    if div {
        let factor = vdupq_n_f64(F10_F64[e]);
        while i + 2 <= n {
            let x = vaddq_s64(
                vreinterpretq_s64_u64(vld1q_u64(offsets.as_ptr().add(i))),
                x_adj,
            );
            let s = vaddq_s64(x, vextq_s64(zero, x, 1));
            // `s - x` is the exclusive prefix; lane 1 holds the block total.
            let enc = vaddq_s64(vsubq_s64(s, x), vdupq_n_s64(carry));
            let v = vcvtq_f64_s64(enc);
            vst1q_u64(
                out.as_mut_ptr().add(i * 8) as *mut u64,
                vreinterpretq_u64_f64(vdivq_f64(v, factor)),
            );
            carry = carry.wrapping_add(vgetq_lane_s64(s, 1));
            i += 2;
        }
    } else {
        let factor = vdupq_n_f64(F10_F64[f]);
        let inv = vdupq_n_f64(IF10_F64[e]);
        while i + 2 <= n {
            let x = vaddq_s64(
                vreinterpretq_s64_u64(vld1q_u64(offsets.as_ptr().add(i))),
                x_adj,
            );
            let s = vaddq_s64(x, vextq_s64(zero, x, 1));
            let enc = vaddq_s64(vsubq_s64(s, x), vdupq_n_s64(carry));
            let v = vcvtq_f64_s64(enc);
            vst1q_u64(
                out.as_mut_ptr().add(i * 8) as *mut u64,
                vreinterpretq_u64_f64(vmulq_f64(vmulq_f64(v, factor), inv)),
            );
            carry = carry.wrapping_add(vgetq_lane_s64(s, 1));
            i += 2;
        }
    }
    restore_delta_chunk64_scalar(
        &offsets[i..],
        carry,
        delta_base,
        div,
        e,
        f,
        &mut out[i * 8..],
    );
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
#[target_feature(enable = "avx512f,avx512dq")]
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

    /// Byte-exact reimplementation of the original scalar delta decode loop
    /// (i64 cursor chain + per-value emit), used as the differential
    /// reference for the delta restore kernels.
    fn delta_reference32(
        offsets: &[u32],
        base: i64,
        delta_base: i64,
        div: bool,
        e: usize,
        f: usize,
    ) -> Vec<u8> {
        let mut out = Vec::with_capacity(offsets.len() * 4 + 4);
        let mut cursor = base;
        let emit = |enc: i64, out: &mut Vec<u8>| {
            let value = if div {
                (enc as i32 as f32) / F10_F32[e]
            } else {
                (enc as i32 as f32) * F10_F32[f] * IF10_F32[e]
            };
            out.extend_from_slice(&value.to_bits().to_le_bytes());
        };
        emit(cursor, &mut out);
        for &offset in offsets {
            cursor = cursor.wrapping_add((offset as i64).wrapping_add(delta_base));
            emit(cursor, &mut out);
        }
        out
    }

    fn delta_reference64(
        offsets: &[u64],
        base: i64,
        delta_base: i64,
        div: bool,
        e: usize,
        f: usize,
    ) -> Vec<u8> {
        let mut out = Vec::with_capacity(offsets.len() * 8 + 8);
        let mut cursor = base;
        let emit = |enc: i64, out: &mut Vec<u8>| {
            let value = if div {
                (enc as f64) / F10_F64[e]
            } else {
                (enc as f64) * F10_F64[f] * IF10_F64[e]
            };
            out.extend_from_slice(&value.to_bits().to_le_bytes());
        };
        emit(cursor, &mut out);
        for &offset in offsets {
            cursor = cursor.wrapping_add((offset as i64).wrapping_add(delta_base));
            emit(cursor, &mut out);
        }
        out
    }

    /// The scalar fallback must reproduce the original delta loop
    /// byte-for-byte, including negative deltas and i32/i64 wrapping.
    #[test]
    fn test_delta_restore32_scalar_matches_reference() {
        let mut rng = StdRng::seed_from_u64(0x00de_1a1a_2026_1024);
        for &(div, e, f) in &[
            (true, 1, 0),
            (true, 5, 0),
            (true, 10, 0),
            (false, 5, 3),
            (false, 0, 7),
            (false, 10, 10),
        ] {
            for &n in &[
                0usize, 1, 2, 3, 7, 8, 15, 16, 17, 31, 63, 64, 65, 1023, 1024, 1025,
            ] {
                let offsets: Vec<u32> = (0..n).map(|_| rng.random()).collect();
                for &(base, delta_base) in &[
                    (0i64, 0i64),
                    (
                        rng.random_range(i32::MIN as i64..=i32::MAX as i64),
                        rng.random(),
                    ),
                    (i32::MIN as i64, -1i64),
                    (i32::MAX as i64, 1i64),
                ] {
                    let expected = delta_reference32(&offsets, base, delta_base, div, e, f);
                    let mut out = vec![0u8; expected.len()];
                    restore_delta_chunk32_scalar(&offsets, base, delta_base, div, e, f, &mut out);
                    assert_eq!(
                        out, expected,
                        "delta32 mismatch n={n} base={base} db={delta_base}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_delta_restore64_scalar_matches_reference() {
        let mut rng = StdRng::seed_from_u64(0x00de_1a1b_2026_1025);
        for &(div, e, f) in &[
            (true, 1, 0),
            (true, 9, 0),
            (true, 18, 0),
            (false, 5, 3),
            (false, 0, 7),
            (false, 18, 18),
        ] {
            for &n in &[
                0usize, 1, 2, 3, 7, 8, 15, 16, 17, 31, 63, 64, 65, 1023, 1024, 1025,
            ] {
                let offsets: Vec<u64> = (0..n).map(|_| rng.random()).collect();
                for &(base, delta_base) in &[
                    (0i64, 0i64),
                    (rng.random_range(i64::MIN / 4..=i64::MAX / 4), rng.random()),
                    (i64::MIN, -1i64),
                    (i64::MAX, 1i64),
                ] {
                    let expected = delta_reference64(&offsets, base, delta_base, div, e, f);
                    let mut out = vec![0u8; expected.len()];
                    restore_delta_chunk64_scalar(&offsets, base, delta_base, div, e, f, &mut out);
                    assert_eq!(
                        out, expected,
                        "delta64 mismatch n={n} base={base} db={delta_base}"
                    );
                }
            }
        }
    }

    /// Software emulation of the AVX-512 16-lane scan (masked permute with
    /// zeroed wrap lanes) and the 8-lane scans, using the same index tables
    /// as the kernels. This runs on every architecture so an index-table
    /// mistake is caught locally instead of only on AVX hardware.
    fn emulate_scan16(
        offsets: &[u32],
        base: i64,
        delta_base: i64,
        div: bool,
        e: usize,
        f: usize,
        out: &mut [u8],
    ) {
        let db = delta_base as i32;
        let mut carry = base as i32;
        let mut i = 0;
        let m = offsets.len();
        while i + 16 <= m {
            let mut x = [0i32; 16];
            for j in 0..16 {
                x[j] = (offsets[i + j] as i32).wrapping_add(db);
            }
            let mut s = x;
            for (k, table) in [1usize, 2, 4, 8].into_iter().zip(AVX512_SCAN_IDX) {
                let mut shifted = [0i32; 16];
                for j in k..16 {
                    shifted[j] = s[table[j] as usize];
                }
                for j in 0..16 {
                    s[j] = s[j].wrapping_add(shifted[j]);
                }
            }
            for j in 0..16 {
                let enc = s[j].wrapping_sub(x[j]).wrapping_add(carry);
                let value = if div {
                    (enc as f32) / F10_F32[e]
                } else {
                    (enc as f32) * F10_F32[f] * IF10_F32[e]
                };
                out[(i + j) * 4..(i + j) * 4 + 4].copy_from_slice(&value.to_bits().to_le_bytes());
            }
            carry = carry.wrapping_add(s[15]);
            i += 16;
        }
        restore_delta_chunk32_scalar(
            &offsets[i..],
            carry as i64,
            delta_base,
            div,
            e,
            f,
            &mut out[i * 4..],
        );
    }

    fn emulate_scan8(
        offsets: &[u32],
        base: i64,
        delta_base: i64,
        div: bool,
        e: usize,
        f: usize,
        out: &mut [u8],
    ) {
        let db = delta_base as i32;
        let mut carry = base as i32;
        let mut i = 0;
        let m = offsets.len();
        while i + 8 <= m {
            let mut x = [0i32; 8];
            for j in 0..8 {
                x[j] = (offsets[i + j] as i32).wrapping_add(db);
            }
            let mut s = x;
            for (k, table) in [1usize, 2, 4].into_iter().zip(AVX8_SCAN_IDX_I32) {
                let mut shifted = [0i32; 8];
                for j in k..8 {
                    shifted[j] = s[table[j] as usize];
                }
                for j in 0..8 {
                    s[j] = s[j].wrapping_add(shifted[j]);
                }
            }
            for j in 0..8 {
                let enc = s[j].wrapping_sub(x[j]).wrapping_add(carry);
                let value = if div {
                    (enc as f32) / F10_F32[e]
                } else {
                    (enc as f32) * F10_F32[f] * IF10_F32[e]
                };
                out[(i + j) * 4..(i + j) * 4 + 4].copy_from_slice(&value.to_bits().to_le_bytes());
            }
            carry = carry.wrapping_add(s[7]);
            i += 8;
        }
        restore_delta_chunk32_scalar(
            &offsets[i..],
            carry as i64,
            delta_base,
            div,
            e,
            f,
            &mut out[i * 4..],
        );
    }

    fn emulate_scan8_64(
        offsets: &[u64],
        base: i64,
        delta_base: i64,
        div: bool,
        e: usize,
        f: usize,
        out: &mut [u8],
    ) {
        let mut carry = base;
        let mut i = 0;
        let m = offsets.len();
        while i + 8 <= m {
            let mut x = [0i64; 8];
            for j in 0..8 {
                x[j] = (offsets[i + j] as i64).wrapping_add(delta_base);
            }
            let mut s = x;
            for (k, table) in [1usize, 2, 4].into_iter().zip(AVX8_SCAN_IDX) {
                let mut shifted = [0i64; 8];
                for j in k..8 {
                    shifted[j] = s[table[j] as usize];
                }
                for j in 0..8 {
                    s[j] = s[j].wrapping_add(shifted[j]);
                }
            }
            for j in 0..8 {
                let enc = s[j].wrapping_sub(x[j]).wrapping_add(carry);
                let value = if div {
                    (enc as f64) / F10_F64[e]
                } else {
                    (enc as f64) * F10_F64[f] * IF10_F64[e]
                };
                out[(i + j) * 8..(i + j) * 8 + 8].copy_from_slice(&value.to_bits().to_le_bytes());
            }
            carry = carry.wrapping_add(s[7]);
            i += 8;
        }
        restore_delta_chunk64_scalar(
            &offsets[i..],
            carry,
            delta_base,
            div,
            e,
            f,
            &mut out[i * 8..],
        );
    }

    #[test]
    fn test_avx_scan_tables_produce_exclusive_prefix() {
        let mut rng = StdRng::seed_from_u64(0x00de_5caa_2026_1028);
        for &(div, e, f) in &[
            (true, 1, 0),
            (true, 5, 0),
            (true, 10, 0),
            (false, 5, 3),
            (false, 0, 7),
            (false, 10, 10),
        ] {
            for &n in &[
                0usize, 1, 3, 7, 8, 15, 16, 17, 31, 63, 64, 65, 127, 128, 129, 1023, 1024, 1025,
            ] {
                let offs32: Vec<u32> = (0..n).map(|_| rng.random()).collect();
                let offs64: Vec<u64> = (0..n).map(|_| rng.random()).collect();
                for &(base, delta_base) in &[
                    (0i64, 0i64),
                    (i32::MAX as i64, 1i64),
                    (i32::MIN as i64, -1i64),
                    (i64::MAX, 1i64),
                    (i64::MIN, -1i64),
                ] {
                    let mut expected = vec![0u8; (n + 1) * 4];
                    let mut got = vec![0u8; (n + 1) * 4];
                    restore_delta_chunk32_scalar(
                        &offs32,
                        base,
                        delta_base,
                        div,
                        e,
                        f,
                        &mut expected,
                    );
                    emulate_scan16(&offs32, base, delta_base, div, e, f, &mut got);
                    assert_eq!(
                        got, expected,
                        "scan16 mismatch div={div} e={e} f={f} n={n} base={base} db={delta_base}"
                    );
                    let mut got = vec![0u8; (n + 1) * 4];
                    emulate_scan8(&offs32, base, delta_base, div, e, f, &mut got);
                    assert_eq!(
                        got, expected,
                        "scan8 mismatch div={div} e={e} f={f} n={n} base={base} db={delta_base}"
                    );
                    let mut expected64 = vec![0u8; (n + 1) * 8];
                    let mut got64 = vec![0u8; (n + 1) * 8];
                    restore_delta_chunk64_scalar(
                        &offs64,
                        base,
                        delta_base,
                        div,
                        e,
                        f,
                        &mut expected64,
                    );
                    emulate_scan8_64(&offs64, base, delta_base, div, e, f, &mut got64);
                    assert_eq!(
                        got64, expected64,
                        "scan8_64 mismatch div={div} e={e} f={f} n={n} base={base} db={delta_base}"
                    );
                }
            }
        }
    }

    #[allow(clippy::type_complexity)]
    fn assert_delta32_equivalent(
        kernel: unsafe fn(&[u32], i64, i64, bool, usize, usize, &mut [u8]),
    ) {
        let mut rng = StdRng::seed_from_u64(0x00de_1a1c_2026_1026);
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
                for &(base, delta_base) in &[
                    (0i64, 0i64),
                    (
                        rng.random_range(i32::MIN as i64..=i32::MAX as i64),
                        rng.random(),
                    ),
                    (i32::MIN as i64, -1i64),
                    (i32::MAX as i64, 1i64),
                ] {
                    let mut expected = vec![0u8; (n + 1) * 4];
                    let mut got = vec![0u8; (n + 1) * 4];
                    restore_delta_chunk32_scalar(
                        &offsets,
                        base,
                        delta_base,
                        div,
                        e,
                        f,
                        &mut expected,
                    );
                    unsafe { kernel(&offsets, base, delta_base, div, e, f, &mut got) };
                    assert_eq!(
                        got, expected,
                        "delta32 mismatch div={div} e={e} f={f} n={n} base={base} db={delta_base}"
                    );
                }
            }
        }
    }

    #[allow(clippy::type_complexity)]
    fn assert_delta64_equivalent(
        kernel: unsafe fn(&[u64], i64, i64, bool, usize, usize, &mut [u8]),
    ) {
        let mut rng = StdRng::seed_from_u64(0x00de_1a1d_2026_1027);
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
                for &(base, delta_base) in &[
                    (0i64, 0i64),
                    (rng.random_range(i64::MIN / 4..=i64::MAX / 4), rng.random()),
                    (i64::MIN, -1i64),
                    (i64::MAX, 1i64),
                ] {
                    let mut expected = vec![0u8; (n + 1) * 8];
                    let mut got = vec![0u8; (n + 1) * 8];
                    restore_delta_chunk64_scalar(
                        &offsets,
                        base,
                        delta_base,
                        div,
                        e,
                        f,
                        &mut expected,
                    );
                    unsafe { kernel(&offsets, base, delta_base, div, e, f, &mut got) };
                    assert_eq!(
                        got, expected,
                        "delta64 mismatch div={div} e={e} f={f} n={n} base={base} db={delta_base}"
                    );
                }
            }
        }
    }

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
        assert_delta32_equivalent(restore_delta_chunk32);
        assert_delta64_equivalent(restore_delta_chunk64);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        assert_restore32_equivalent(restore_chunk32_avx2);
        assert_delta32_equivalent(restore_delta_chunk32_avx2);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx512_f32_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        assert_restore32_equivalent(restore_chunk32_avx512);
        assert_delta32_equivalent(restore_delta_chunk32_avx512);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx512_f64_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx512f")
            || !std::arch::is_x86_feature_detected!("avx512dq")
        {
            return;
        }
        assert_restore64_equivalent(restore_chunk64_avx512);
        assert_delta64_equivalent(restore_delta_chunk64_avx512);
    }
}
