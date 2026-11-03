// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Pairwise intersection of the sorted, distinct DocIDs in posting blocks.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm_castsi128_ps, _mm_cmpeq_epi32, _mm_loadu_si128, _mm_movemask_ps, _mm_or_si128,
    _mm_shuffle_epi32,
};

#[inline]
pub(super) fn intersect(left: &[u32], right: &[u32], out: &mut Vec<u32>) {
    out.clear();
    out.reserve(left.len());
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 support was checked above.
        unsafe { intersect_avx2(left, right, out) };
        return;
    }
    intersect_scalar(left, right, out);
}

#[inline]
fn intersect_scalar(left: &[u32], right: &[u32], out: &mut Vec<u32>) {
    let (mut a, mut b) = (0, 0);
    while a < left.len() && b < right.len() {
        match left[a].cmp(&right[b]) {
            std::cmp::Ordering::Less => a += 1,
            std::cmp::Ordering::Greater => b += 1,
            std::cmp::Ordering::Equal => {
                out.push(left[a]);
                a += 1;
                b += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn intersect_avx2(left: &[u32], right: &[u32], out: &mut Vec<u32>) {
    let (mut a, mut b) = (0, 0);
    while a + 4 <= left.len() && b + 4 <= right.len() {
        let a_max = left[a + 3];
        let b_max = right[b + 3];
        if a_max < right[b] {
            a += 4;
            continue;
        }
        if b_max < left[a] {
            b += 4;
            continue;
        }
        // Compare every lane in the left tile with every lane in the right
        // tile. Equality needs no signedness conversion for high DocIDs.
        // SAFETY: both slices have at least four entries at these offsets;
        // unaligned loads are permitted and AVX2 was checked by the caller.
        let mut mask = unsafe {
            let av = _mm_loadu_si128(left.as_ptr().add(a).cast());
            let mut bv = _mm_loadu_si128(right.as_ptr().add(b).cast());
            let mut eq = _mm_cmpeq_epi32(av, bv);
            bv = _mm_shuffle_epi32::<0x39>(bv);
            eq = _mm_or_si128(eq, _mm_cmpeq_epi32(av, bv));
            bv = _mm_shuffle_epi32::<0x39>(bv);
            eq = _mm_or_si128(eq, _mm_cmpeq_epi32(av, bv));
            bv = _mm_shuffle_epi32::<0x39>(bv);
            eq = _mm_or_si128(eq, _mm_cmpeq_epi32(av, bv));
            _mm_movemask_ps(_mm_castsi128_ps(eq)) as u32
        };
        while mask != 0 {
            out.push(left[a + mask.trailing_zeros() as usize]);
            mask &= mask - 1;
        }
        // Retain the tile with the larger maximum: its remaining documents
        // may match the next tile on the other side. Distinct sorted inputs
        // ensure emitted matches remain ordered and cannot be repeated.
        a += usize::from(a_max <= b_max) * 4;
        b += usize::from(b_max <= a_max) * 4;
    }
    intersect_scalar(&left[a..], &right[b..], out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn assert_intersection(left: &[u32], right: &[u32]) {
        let expected = left
            .iter()
            .filter(|doc| right.contains(doc))
            .copied()
            .collect::<Vec<_>>();
        let mut actual = Vec::new();
        intersect_scalar(left, right, &mut actual);
        assert_eq!(actual, expected);
        intersect(left, right, &mut actual);
        assert_eq!(actual, expected, "intersection of {left:?} and {right:?}");
    }

    #[test]
    fn exhaustive_short_sets() {
        for left_mask in 0u32..256 {
            let left = (0..8)
                .filter(|bit| left_mask & (1 << bit) != 0)
                .collect::<Vec<_>>();
            for right_mask in 0u32..256 {
                let right = (0..8)
                    .filter(|bit| right_mask & (1 << bit) != 0)
                    .collect::<Vec<_>>();
                assert_intersection(&left, &right);
            }
        }
    }

    #[rstest]
    fn sparse_full_domain_blocks(
        #[values(0, 1, 3, 4, 5, 7, 8, 9, 127, 128, 255, 256)] right_len: u32,
    ) {
        for left_len in 0..=256 {
            let left = (0..left_len)
                .map(|i| u32::MAX - (255 - i) * 10_000_000)
                .collect::<Vec<_>>();
            let right = (0..right_len)
                .map(|i| u32::MAX - (255 - i) * 10_000_001)
                .collect::<Vec<_>>();
            assert_intersection(&left, &right);
        }
    }
}
