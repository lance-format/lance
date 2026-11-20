// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The 1-bit graph walk of `IVF_HNSW_RQ`: it scores the query against the sign
//! bits only, with the query quantized to 4 bits so that AND + popcount over bit
//! planes replaces the float inner product. `VectorStore::rerank` rescores.

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::vector::utils::do_prefetch;

/// Bits per dim of the walk's query.
const QUERY_BITS: usize = 4;
const DIM_512: usize = 512;
const WORDS_512: usize = DIM_512 / u64::BITS as usize;
/// From this many `u64` words (512 dims) on, each query plane is stored whole,
/// which the compiler vectorizes; below it, each word keeps its planes together.
const PLANE_MAJOR_WORDS: usize = WORDS_512;

/// Packs each row of row-major `_rabit_codes` into little-endian `u64` words,
/// dim `d` at bit `d % 64` of word `d / 64`.
#[cfg(test)]
pub fn sign_words(codes: &[u8], code_len: usize) -> Vec<u64> {
    let mut words = Vec::with_capacity(codes.len() / code_len * code_len.div_ceil(8));
    words.extend(
        codes
            .chunks_exact(code_len)
            .flat_map(|row| row.chunks(8))
            .map(|bytes| {
                let mut word = [0u8; 8];
                word[..bytes.len()].copy_from_slice(bytes);
                u64::from_le_bytes(word)
            }),
    );
    words
}

/// The walk over a partition's row-major sign words, with the rotated query quantized
/// uniformly over its range to 4-bit codes that reconstruct as `delta * code + vl`.
pub struct OneBitWalk<'a> {
    signs: &'a [u64],
    words: usize,
    query: WalkQuery,
    delta: f32,
    vl: f32,
    ip_error_bound: f32,
}

enum WalkQuery {
    Planes(Vec<u64>),
    #[cfg(target_arch = "x86_64")]
    Bytes512(Box<[u8; DIM_512]>),
}

impl WalkQuery {
    fn new(codes: Vec<u8>) -> Self {
        #[cfg(target_arch = "x86_64")]
        let codes = if codes.len() == DIM_512 && !has_vector_popcnt() && has_masked_bytes() {
            // The masked loads consume the query codes directly.
            match codes.into_boxed_slice().try_into() {
                Ok(codes) => return Self::Bytes512(codes),
                Err(codes) => codes.into_vec(),
            }
        } else {
            codes
        };
        Self::Planes(transpose(&codes))
    }

    #[inline(always)]
    fn ip_counts(&self, signs: &[u64]) -> (u64, u64) {
        match self {
            Self::Planes(planes) => ip_counts(signs, planes),
            #[cfg(target_arch = "x86_64")]
            Self::Bytes512(codes) => {
                let signs = signs.first_chunk::<WORDS_512>().expect("512-bit sign row");
                // SAFETY: Bytes512 is constructed only after detecting AVX-512BW
                // and POPCNT support.
                unsafe { ip_counts_512_masked(signs, codes) }
            }
        }
    }
}

impl<'a> OneBitWalk<'a> {
    pub fn new(signs: &'a [u64], rotated: &[f32]) -> Self {
        let (lo, hi) = rotated
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        let max_code = ((1 << QUERY_BITS) - 1) as f32;
        let (vl, delta) = if hi > lo {
            (lo, (hi - lo) / max_code)
        } else {
            (lo.min(0.0).max(hi), 0.0)
        };
        let (mut codes, mut sum_error, mut error_sq) =
            (Vec::with_capacity(rotated.len()), 0.0, 0.0);
        for &v in rotated {
            let code = if delta > 0.0 {
                ((v - vl) / delta).round().clamp(0.0, max_code)
            } else {
                0.0
            };
            let error = delta * code + vl - v;
            sum_error += error;
            error_sq += error * error;
            codes.push(code as u8);
        }
        Self {
            signs,
            words: rotated.len().div_ceil(64),
            query: WalkQuery::new(codes),
            delta,
            vl,
            // A sign row `b` sums the reconstruction error `err` over its set
            // bits, `(sum(err) + <2b - 1, err>) / 2`, and padded bits are never
            // set; Cauchy-Schwarz bounds `|<2b - 1, err>|` by `sqrt(dim) |err|`.
            ip_error_bound: 0.5
                * (f32::abs(sum_error) + (rotated.len() as f32).sqrt() * f32::sqrt(error_sq)),
        }
    }

    /// The reconstructed query summed over the dims whose sign bit is set in row `id`.
    #[inline(always)]
    pub fn ip(&self, id: usize) -> f32 {
        let (ip, count) = self.query.ip_counts(self.row(id));
        self.delta * ip as f32 + self.vl * count as f32
    }

    /// Bounds how far [`Self::ip`] lies from the exact inner product with any row.
    pub fn ip_error_bound(&self) -> f32 {
        self.ip_error_bound
    }

    #[inline(always)]
    pub fn prefetch(&self, id: usize) {
        #[cfg(target_arch = "x86_64")]
        if self.words == WORDS_512 {
            // SAFETY: row provides a pointer to the non-empty sign code. For
            // 64 bytes the generic helper would issue the same one prefetch.
            unsafe { _mm_prefetch::<_MM_HINT_T0>(self.row(id).as_ptr().cast()) };
            return;
        }
        do_prefetch(self.row(id).as_ptr_range());
    }

    #[inline(always)]
    fn row(&self, id: usize) -> &'a [u64] {
        &self.signs[id * self.words..(id + 1) * self.words]
    }
}

/// Transposes 4-bit query codes into bit planes, dim `d` at bit `d % 64` of word
/// `d / 64`: plane by plane from [`PLANE_MAJOR_WORDS`] words on, else word by word.
fn transpose(codes: &[u8]) -> Vec<u64> {
    let words = codes.len().div_ceil(64);
    let mut planes = vec![0u64; words * QUERY_BITS];
    for (d, &code) in codes.iter().enumerate() {
        for j in 0..QUERY_BITS {
            let at = if words >= PLANE_MAJOR_WORDS {
                j * words + d / 64
            } else {
                d / 64 * QUERY_BITS + j
            };
            planes[at] |= u64::from((code >> j) & 1) << (d % 64);
        }
    }
    planes
}

/// The sum of the query codes over the set bits of `signs`, and their count.
#[inline(always)]
fn ip_counts(signs: &[u64], planes: &[u64]) -> (u64, u64) {
    if let (Ok(signs), (planes, [])) = (signs.try_into(), planes.as_chunks::<WORDS_512>())
        && let Ok(planes) = planes.try_into()
    {
        return ip_counts_512(signs, planes);
    }
    let count = signs.iter().map(|x| u64::from(x.count_ones())).sum();
    let mut ip = 0u64;
    if signs.len() >= PLANE_MAJOR_WORDS {
        for (j, plane) in planes.chunks_exact(signs.len()).enumerate() {
            let sum: u64 = signs
                .iter()
                .zip(plane)
                .map(|(x, q)| u64::from((x & q).count_ones()))
                .sum();
            ip += sum << j;
        }
    } else {
        for (&x, p) in signs.iter().zip(planes.as_chunks::<QUERY_BITS>().0) {
            ip += u64::from((x & p[0]).count_ones())
                + (u64::from((x & p[1]).count_ones()) << 1)
                + (u64::from((x & p[2]).count_ones()) << 2)
                + (u64::from((x & p[3]).count_ones()) << 3);
        }
    }
    (ip, count)
}

#[inline(always)]
fn ip_counts_512(signs: &[u64; WORDS_512], planes: &[[u64; WORDS_512]; QUERY_BITS]) -> (u64, u64) {
    #[cfg(target_arch = "x86_64")]
    if has_vector_popcnt() {
        // SAFETY: runtime detection established both required CPU features.
        return unsafe { ip_counts_512_avx512(signs, planes) };
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    return ip_counts_512_neon(signs, planes);
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    ip_counts_512_portable(signs, planes)
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn has_vector_popcnt() -> bool {
    is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vpopcntdq")
}

#[cfg(target_arch = "x86_64")]
fn has_masked_bytes() -> bool {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("popcnt")
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,popcnt")]
unsafe fn ip_counts_512_masked(signs: &[u64; WORDS_512], codes: &[u8; DIM_512]) -> (u64, u64) {
    let chunks = codes.as_chunks::<64>().0;
    let masked = |i: usize| {
        // SAFETY: each chunk holds the 64 bytes accessible to its masked load.
        unsafe { _mm512_maskz_loadu_epi8(signs[i], chunks[i].as_ptr().cast()) }
    };
    let pair = |i: usize| _mm512_add_epi8(masked(i), masked(i + 1));
    let low = _mm512_add_epi8(pair(0), pair(2));
    let high = _mm512_add_epi8(pair(4), pair(6));
    // Each lane sums at most eight 4-bit codes, fitting in a byte.
    let sums = _mm512_sad_epu8(_mm512_add_epi8(low, high), _mm512_setzero_si512());
    let ip = _mm512_reduce_add_epi64(sums) as u64;
    let count = signs.iter().map(|x| u64::from(x.count_ones())).sum();
    (ip, count)
}

#[cfg(any(test, not(all(target_arch = "aarch64", target_feature = "neon"))))]
#[inline(always)]
fn ip_counts_512_portable(
    signs: &[u64; WORDS_512],
    [p0, p1, p2, p3]: &[[u64; WORDS_512]; QUERY_BITS],
) -> (u64, u64) {
    let count: u32 = signs.iter().map(|x| x.count_ones()).sum();
    let sum = |plane: &[u64; WORDS_512]| -> u32 {
        signs
            .iter()
            .zip(plane)
            .map(|(x, q)| (x & q).count_ones())
            .sum()
    };
    let ip = sum(p0) + (sum(p1) << 1) + (sum(p2) << 2) + (sum(p3) << 3);
    (u64::from(ip), u64::from(count))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn ip_counts_512_avx512(
    signs: &[u64; WORDS_512],
    [p0, p1, p2, p3]: &[[u64; WORDS_512]; QUERY_BITS],
) -> (u64, u64) {
    // SAFETY: each input array has 64 bytes and these loads allow unaligned addresses.
    let (signs, p0, p1, p2, p3) = unsafe {
        (
            _mm512_loadu_si512(signs.as_ptr().cast()),
            _mm512_loadu_si512(p0.as_ptr().cast()),
            _mm512_loadu_si512(p1.as_ptr().cast()),
            _mm512_loadu_si512(p2.as_ptr().cast()),
            _mm512_loadu_si512(p3.as_ptr().cast()),
        )
    };
    let count = _mm512_popcnt_epi64(signs);
    let sum0 = _mm512_popcnt_epi64(_mm512_and_si512(signs, p0));
    let sum1 = _mm512_popcnt_epi64(_mm512_and_si512(signs, p1));
    let sum2 = _mm512_popcnt_epi64(_mm512_and_si512(signs, p2));
    let sum3 = _mm512_popcnt_epi64(_mm512_and_si512(signs, p3));
    let low = _mm512_add_epi64(sum0, _mm512_slli_epi64::<1>(sum1));
    let high = _mm512_add_epi64(sum2, _mm512_slli_epi64::<1>(sum3));
    let ip = _mm512_add_epi64(low, _mm512_slli_epi64::<2>(high));
    // The largest possible IP is 512 * 15, so the sign count can be packed
    // above the low 16 bits and both totals share one horizontal reduction.
    let packed = _mm512_reduce_add_epi64(_mm512_or_si512(ip, _mm512_slli_epi64::<16>(count)));
    (packed as u64 & 0xffff, packed as u64 >> 16)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
fn ip_counts_512_neon(
    signs: &[u64; WORDS_512],
    planes: &[[u64; WORDS_512]; QUERY_BITS],
) -> (u64, u64) {
    // SAFETY: NEON is enabled at compile time, and each 16-byte load is within
    // its fixed-size input array.
    unsafe {
        let mut count = vdupq_n_u8(0);
        let mut sums = [vdupq_n_u8(0); QUERY_BITS];
        for offset in [0, 2, 4, 6] {
            let signs = vld1q_u8(signs.as_ptr().add(offset).cast());
            count = vaddq_u8(count, vcntq_u8(signs));
            for (sum, plane) in sums.iter_mut().zip(planes) {
                let query = vld1q_u8(plane.as_ptr().add(offset).cast());
                *sum = vaddq_u8(*sum, vcntq_u8(vandq_u8(signs, query)));
            }
        }
        // Each lane has at most 32 bits set across the four vectors. Pairwise
        // addition widens the lanes before applying the high planes' weight.
        let [sum0, sum1, sum2, sum3] = sums;
        let low = vaddq_u8(sum0, vshlq_n_u8::<1>(sum1));
        let high = vaddq_u8(sum2, vshlq_n_u8::<1>(sum3));
        let ip = vaddq_u16(vpaddlq_u8(low), vshlq_n_u16::<2>(vpaddlq_u8(high)));
        (u64::from(vaddvq_u16(ip)), u64::from(vaddlvq_u8(count)))
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::*;

    #[test]
    fn test_walk_512_bit_counts_match_portable() {
        let mut rng = SmallRng::seed_from_u64(7);
        for _ in 0..32 {
            let signs: [_; WORDS_512] = std::array::from_fn(|_| rng.random::<u64>());
            let planes = std::array::from_fn(|_| std::array::from_fn(|_| rng.random::<u64>()));
            assert_eq!(
                ip_counts_512(&signs, &planes),
                ip_counts_512_portable(&signs, &planes)
            );
        }
        assert_eq!(
            ip_counts_512(&[u64::MAX; WORDS_512], &[[u64::MAX; WORDS_512]; QUERY_BITS]),
            (512 * 15, 512)
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_walk_512_masked_bytes_match_portable() {
        if !has_masked_bytes() {
            return;
        }
        let mut rng = SmallRng::seed_from_u64(13);
        for i in 0..32 {
            let signs = std::array::from_fn(|_| if i == 0 { u64::MAX } else { rng.random() });
            let codes = std::array::from_fn(|_| if i == 0 { 15 } else { rng.random_range(0..16) });
            let planes = transpose(&codes);
            let planes = planes.as_chunks::<WORDS_512>().0.try_into().unwrap();
            // SAFETY: the required features were detected above.
            let masked = unsafe { ip_counts_512_masked(&signs, &codes) };
            assert_eq!(masked, ip_counts_512_portable(&signs, planes));
        }
    }

    /// Both plane layouts sum the query codes over the set sign bits, and the
    /// walk stays within its error bound of the exact inner product. 8 and 200
    /// dims end in a partial word; 512 and 960 store the planes whole.
    #[test]
    fn test_walk_inner_products() {
        let mut rng = SmallRng::seed_from_u64(7);
        for dim in [8usize, 64, 200, 512, 960] {
            let codes: Vec<u8> = (0..4 * dim / 8).map(|_| rng.random()).collect();
            let signs = sign_words(&codes, dim / 8);
            let is_set = |id: usize, d: usize| (codes[id * dim / 8 + d / 8] >> (d % 8)) & 1 == 1;
            let query: Vec<u8> = (0..dim).map(|_| rng.random_range(0..16)).collect();
            let rotated: Vec<f32> = (0..dim).map(|_| rng.random_range(-3.0..3.0)).collect();
            let walk = OneBitWalk::new(&signs, &rotated);
            // Each dim reconstructs within half a step, which caps the bound.
            assert!(walk.ip_error_bound() <= 0.5 * dim as f32 * walk.delta + 1e-4);
            for id in 0..4 {
                let row = &signs[id * walk.words..(id + 1) * walk.words];
                let set = (0..dim).filter(|&d| is_set(id, d));
                let want = (
                    set.clone().map(|d| query[d] as u64).sum(),
                    set.clone().count() as u64,
                );
                assert_eq!(ip_counts(row, &transpose(&query)), want, "{dim}/{id}");
                let exact: f32 = set.map(|d| rotated[d]).sum();
                let error = (walk.ip(id) - exact).abs();
                assert!(error <= walk.ip_error_bound() + 1e-3, "{dim}/{id}: {error}");
            }
        }
    }
}
