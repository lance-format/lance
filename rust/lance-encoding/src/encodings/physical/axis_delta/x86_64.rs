// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Integer prediction with vector prefix sums for short strides and independent
//! lane arithmetic for long strides. Every span resets the predictor.

use std::arch::x86_64::*;

#[inline]
fn update_tail<const DECODE: bool, const WIDTH: usize>(
    span: &mut [u8],
    stride: usize,
    offset: usize,
) {
    let mut current = [0; 8];
    let mut previous = [0; 8];
    current[..WIDTH].copy_from_slice(&span[offset..offset + WIDTH]);
    previous[..WIDTH].copy_from_slice(&span[offset - stride..offset - stride + WIDTH]);
    let current = u64::from_le_bytes(current);
    let previous = u64::from_le_bytes(previous);
    let value = if DECODE {
        current.wrapping_add(previous)
    } else {
        current.wrapping_sub(previous)
    };
    span[offset..offset + WIDTH].copy_from_slice(&value.to_le_bytes()[..WIDTH]);
}

#[inline]
fn add128<const WIDTH: usize>(current: __m128i, previous: __m128i) -> __m128i {
    // SAFETY: SSE2 is available on every x86_64 target. Width is validated by
    // the dispatch below; each instruction wraps at the corresponding bit width.
    unsafe {
        match WIDTH {
            1 => _mm_add_epi8(current, previous),
            2 => _mm_add_epi16(current, previous),
            4 => _mm_add_epi32(current, previous),
            8 => _mm_add_epi64(current, previous),
            _ => unreachable!("validated element width"),
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn combine256<const DECODE: bool, const WIDTH: usize>(
    current: __m256i,
    previous: __m256i,
) -> __m256i {
    match (DECODE, WIDTH) {
        (true, 1) => _mm256_add_epi8(current, previous),
        (true, 2) => _mm256_add_epi16(current, previous),
        (true, 4) => _mm256_add_epi32(current, previous),
        (true, 8) => _mm256_add_epi64(current, previous),
        (false, 1) => _mm256_sub_epi8(current, previous),
        (false, 2) => _mm256_sub_epi16(current, previous),
        (false, 4) => _mm256_sub_epi32(current, previous),
        (false, 8) => _mm256_sub_epi64(current, previous),
        _ => unreachable!("validated element width"),
    }
}

/// Transform complete predictor spans without touching neighboring bytes.
///
/// # Safety
/// AVX2 must be available. Width must be 1, 2, 4, or 8; stride and span size
/// must be positive multiples of width, with stride dividing span size.
/// The buffer must contain complete spans. Unaligned buffers are supported.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn transform<const DECODE: bool>(
    bytes: &mut [u8],
    span_bytes: usize,
    stride: usize,
    width: usize,
) {
    // SAFETY: The caller validates the CPU and layout; dispatch fixes the lane
    // width outside all element loops.
    unsafe {
        match width {
            1 => transform_width::<DECODE, 1>(bytes, span_bytes, stride),
            2 => transform_width::<DECODE, 2>(bytes, span_bytes, stride),
            4 => transform_width::<DECODE, 4>(bytes, span_bytes, stride),
            8 => transform_width::<DECODE, 8>(bytes, span_bytes, stride),
            _ => unreachable!("validated element width"),
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn transform_width<const DECODE: bool, const WIDTH: usize>(
    bytes: &mut [u8],
    span_bytes: usize,
    stride: usize,
) {
    // A short stride creates dependencies within a vector. Doubling the shift
    // computes an inclusive prefix sum for each interleaved predecessor chain.
    // The preceding vector contributes only its last `stride` bytes. This also
    // handles strides (e.g. RGB's three bytes) that do not divide vector width.
    macro_rules! prefix {
        ($stride:literal) => {{
            for span in bytes.chunks_exact_mut(span_bytes) {
                let end = span.len() / 16 * 16;
                let mut previous = _mm_setzero_si128();
                for offset in (0..end).step_by(16) {
                    // SAFETY: `end` excludes partial vectors. Both accesses
                    // cover exactly 16 bytes inside this span, without alignment
                    // requirements. Immediate shifts never access memory.
                    unsafe {
                        let dest = span.as_mut_ptr().add(offset);
                        let mut value = _mm_loadu_si128(dest.cast());
                        value =
                            add128::<WIDTH>(value, _mm_srli_si128::<{ 16 - $stride }>(previous));
                        value = add128::<WIDTH>(value, _mm_slli_si128::<$stride>(value));
                        if $stride * 2 < 16 {
                            value =
                                add128::<WIDTH>(value, _mm_slli_si128::<{ $stride * 2 }>(value));
                        }
                        if $stride * 4 < 16 {
                            value =
                                add128::<WIDTH>(value, _mm_slli_si128::<{ $stride * 4 }>(value));
                        }
                        if $stride * 8 < 16 {
                            value =
                                add128::<WIDTH>(value, _mm_slli_si128::<{ $stride * 8 }>(value));
                        }
                        _mm_storeu_si128(dest.cast(), value);
                        previous = value;
                    }
                }
                for offset in (end.max(stride)..span.len()).step_by(WIDTH) {
                    update_tail::<true, WIDTH>(span, stride, offset);
                }
            }
        }};
    }
    if DECODE && stride < 16 {
        match stride {
            1 => prefix!(1),
            2 => prefix!(2),
            3 => prefix!(3),
            4 => prefix!(4),
            5 => prefix!(5),
            6 => prefix!(6),
            7 => prefix!(7),
            8 => prefix!(8),
            9 => prefix!(9),
            10 => prefix!(10),
            11 => prefix!(11),
            12 => prefix!(12),
            13 => prefix!(13),
            14 => prefix!(14),
            15 => prefix!(15),
            _ => unreachable!("validated nonzero stride"),
        }
        return;
    }
    for span in bytes.chunks_exact_mut(span_bytes) {
        if DECODE {
            let mut offset = stride;
            // A vector cannot contain its own predecessors on this path.
            if stride >= 32 {
                while offset + 32 <= span.len() {
                    // SAFETY: stride >= 32 keeps every predecessor in already
                    // decoded bytes. The bounds keep both 32-byte windows inside
                    // this span. AVX2 is enabled and accesses are unaligned.
                    unsafe {
                        let dest = span.as_mut_ptr().add(offset);
                        let value = _mm256_loadu_si256(dest.cast());
                        let previous = _mm256_loadu_si256(dest.sub(stride).cast());
                        _mm256_storeu_si256(
                            dest.cast(),
                            combine256::<true, WIDTH>(value, previous),
                        );
                    }
                    offset += 32;
                }
            } else {
                // Retain the complete preceding step in registers. Reloading
                // recently stored bytes is especially costly for steps such as
                // 24 bytes that do not divide the vector width.
                // SAFETY: This path has stride in 16..32 and complete spans.
                unsafe {
                    let mut previous = _mm_loadu_si128(span.as_ptr().cast());
                    if stride == 16 {
                        while offset < span.len() {
                            let dest = span.as_mut_ptr().add(offset);
                            previous = add128::<WIDTH>(_mm_loadu_si128(dest.cast()), previous);
                            _mm_storeu_si128(dest.cast(), previous);
                            offset += stride;
                        }
                    } else {
                        let tail = stride - 16;
                        let mut previous_tail = _mm_loadu_si128(span.as_ptr().add(tail).cast());
                        while offset < span.len() {
                            let dest = span.as_mut_ptr().add(offset);
                            // The two windows overlap, but both compute identical
                            // values there. Load both before either store.
                            let first = _mm_loadu_si128(dest.cast());
                            let last = _mm_loadu_si128(dest.add(tail).cast());
                            previous = add128::<WIDTH>(first, previous);
                            previous_tail = add128::<WIDTH>(last, previous_tail);
                            _mm_storeu_si128(dest.cast(), previous);
                            _mm_storeu_si128(dest.add(tail).cast(), previous_tail);
                            offset += stride;
                        }
                    }
                }
            }
            for offset in (offset..span.len()).step_by(WIDTH) {
                update_tail::<true, WIDTH>(span, stride, offset);
            }
        } else {
            let mut end = span.len();
            while end - stride >= 32 {
                end -= 32;
                // SAFETY: The descending traversal retains original predecessors.
                // Both complete windows lie inside the span; load both before
                // storing because they may overlap when stride < 32.
                unsafe {
                    let dest = span.as_mut_ptr().add(end);
                    let value = _mm256_loadu_si256(dest.cast());
                    let previous = _mm256_loadu_si256(dest.sub(stride).cast());
                    _mm256_storeu_si256(dest.cast(), combine256::<false, WIDTH>(value, previous));
                }
            }
            for offset in (stride..end).step_by(WIDTH).rev() {
                update_tail::<false, WIDTH>(span, stride, offset);
            }
        }
    }
}
