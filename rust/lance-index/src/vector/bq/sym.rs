// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! SymRaBitQ: distances between two RaBitQ codes.
//!
//! `IVF_HNSW_RQ` builds its graph without the raw vectors, so the edges are
//! scored between two RaBitQ codes with the symmetric estimator in
//! [`sym_dist`]. The graph walk scores the query against the 1-bit codes
//! only: the query is quantized to 4 bits ([`prepare_warmup_query`])
//! so that AND + popcount replaces the float inner product.
//!
//! "Lib" refers to the reference RaBitQ-Library. These kernels reproduce its
//! layouts, and the code-to-code ones its numerics; `testdata/symrabitq` holds
//! its outputs.

use std::sync::LazyLock;

use lance_linalg::distance::DistanceType;

use crate::vector::bq::ex_dot::unpack_blocked_row;

/// Bytes per row of sign words: one `u64` per 64 dims, the last one
/// zero-padded.
pub fn sign_word_bytes(dim: usize) -> usize {
    dim.div_ceil(64) * 8
}

/// Regroups the sign bits of row-major `_rabit_codes` (dim `i` at bit `i % 8`
/// of byte `i / 8`) into the sign words the kernels read: one little-endian
/// `u64` per 64 dims with the first dim at bit 63, as Lib packs them.
pub fn sign_words_from_codes(codes: &[u8], code_len: usize) -> Vec<u8> {
    let word_bytes = sign_word_bytes(code_len * 8);
    let mut words = vec![0u8; codes.len() / code_len * word_bytes];
    for (row, row_words) in codes
        .chunks_exact(code_len)
        .zip(words.chunks_exact_mut(word_bytes))
    {
        for (bytes, word) in row.chunks(8).zip(row_words.as_chunks_mut::<8>().0) {
            let mut le = [0u8; 8];
            le[..bytes.len()].copy_from_slice(bytes);
            *word = u64::from_le_bytes(le).reverse_bits().to_le_bytes();
        }
    }
    words
}

/// Per-row factors of the symmetric estimator. Nothing stores them:
/// [`derive_sym_factors`] rebuilds them from the asymmetric RQ factors.
#[derive(Debug, Clone, Copy)]
pub struct SymFactors {
    /// `κ = ‖r‖² / ⟨r, ũ⟩` for the rotated residual `r` and its centered code
    /// `ũ`, so that `⟨ra, rb⟩ ≈ κa κb ⟨ũa, ũb⟩`.
    pub kappa: f32,
    /// `‖r‖²` (L2) or `⟨r, c⟩ + ‖c‖² / 2` (Dot) for the rotated centroid `c`.
    pub base: f32,
}

/// One row as [`sym_dist`] reads it: sign words, blocked ex codes, factors.
#[derive(Debug, Clone, Copy)]
pub struct SymCode<'a> {
    pub signs: &'a [u8],
    pub ex: &'a [u8],
    pub factors: SymFactors,
}

/// The code of dim `d`: its sign bit over its unpacked ex code (none at 1 bit).
fn split_code(signs: &[u8], ex_codes: &[u8], ex_bits: u8, d: usize) -> u32 {
    let word = u64::from_le_bytes(signs.as_chunks::<8>().0[d / 64]);
    let sign = ((word >> (63 - d % 64)) & 1) as u32;
    (sign << ex_bits) + ex_codes.get(d).map_or(0, |&ex| ex as u32)
}

fn code_bias(num_bits: u8) -> f32 {
    ((1u32 << num_bits) - 1) as f32 / 2.0
}

type CenteredIpFn = fn(&[u8], &[u8], &[u8], &[u8], usize, u8) -> f32;

static CENTERED_IP: LazyLock<CenteredIpFn> = LazyLock::new(|| {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return x86::centered_code_ip;
        }
    }
    centered_code_ip_scalar
});

/// `Σ (ua - cb)(ub - cb)` from `Σ ua ub` and the code sums. `i64` keeps
/// 9-bit codes of high-dimensional vectors from overflowing.
fn finish_centered(ip_uu: i64, sum_a: i64, sum_b: i64, dim: usize, cb: f32) -> f32 {
    let center = cb as f64;
    ((ip_uu as f64) - (center * (sum_a + sum_b) as f64) + (dim as f64 * center * center)) as f32
}

fn centered_code_ip_scalar(
    signs_a: &[u8],
    ex_a: &[u8],
    signs_b: &[u8],
    ex_b: &[u8],
    dim: usize,
    num_bits: u8,
) -> f32 {
    let ex_bits = num_bits - 1;
    let (mut codes_a, mut codes_b) = (Vec::new(), Vec::new());
    if ex_bits > 0 {
        unpack_blocked_row(ex_a, ex_bits, dim, &mut codes_a);
        unpack_blocked_row(ex_b, ex_bits, dim, &mut codes_b);
    }
    let (mut ip_uu, mut sum_a, mut sum_b) = (0i64, 0i64, 0i64);
    for d in 0..dim {
        let ua = split_code(signs_a, &codes_a, ex_bits, d) as i64;
        let ub = split_code(signs_b, &codes_b, ex_bits, d) as i64;
        ip_uu += ua * ub;
        sum_a += ua;
        sum_b += ub;
    }
    finish_centered(ip_uu, sum_a, sum_b, dim, code_bias(num_bits))
}

/// `⟨ũa, ũb⟩` of two codes centered at `((1 << num_bits) - 1) / 2`.
fn centered_code_ip(a: &SymCode, b: &SymCode, dim: usize, num_bits: u8) -> f32 {
    debug_assert_eq!(a.signs.len(), sign_word_bytes(dim));
    debug_assert_eq!(b.signs.len(), sign_word_bytes(dim));
    debug_assert!(
        num_bits == 1
            || a.ex.len() == crate::vector::bq::ex_dot::blocked_ex_code_bytes(dim, num_bits - 1)
    );
    CENTERED_IP(a.signs, a.ex, b.signs, b.ex, dim, num_bits)
}

/// Lib `sym_dist_split`: the L2 or Dot distance between two rows of the same
/// partition, estimated from their codes alone. Any metric other than Dot is
/// scored as L2, the metric a normalized Cosine index stores.
pub fn sym_dist(
    a: &SymCode,
    b: &SymCode,
    dim: usize,
    num_bits: u8,
    distance_type: DistanceType,
) -> f32 {
    let (fa, fb) = (&a.factors, &b.factors);
    let ip = fa.kappa * fb.kappa * centered_code_ip(a, b, dim, num_bits);
    match distance_type {
        DistanceType::Dot => 1.0 - (fa.base + fb.base + ip),
        _ => fa.base + fb.base - 2.0 * ip,
    }
}

/// The [`SymFactors`] of one row from its sign words, its unpacked `ex_codes`
/// (empty at 1 bit), its rotated partition `centroid` and its raw-query RQ
/// factors: `scale` is `-2κ` (L2) or `-κ` (Dot), and `add + scale⟨c, ũ⟩` is
/// `‖r‖²` (L2) or `1 - ⟨r, c⟩` (Dot).
pub fn derive_sym_factors(
    signs: &[u8],
    ex_codes: &[u8],
    centroid: &[f32],
    add: f32,
    scale: f32,
    num_bits: u8,
    distance_type: DistanceType,
) -> SymFactors {
    let (ex_bits, cb) = (num_bits - 1, code_bias(num_bits));
    let centroid_ip: f32 = centroid
        .iter()
        .enumerate()
        .map(|(d, c)| c * (split_code(signs, ex_codes, ex_bits, d) as f32 - cb))
        .sum();
    let base = add + scale * centroid_ip;
    match distance_type {
        DistanceType::Dot => SymFactors {
            kappa: -scale,
            base: 1.0 - base + 0.5 * centroid.iter().map(|c| c * c).sum::<f32>(),
        },
        _ => SymFactors {
            kappa: -0.5 * scale,
            base: base.max(0.0),
        },
    }
}

/// Lib `SplitSingleQuery::kNumBits`. Query is 4-bit; data stays 1-bit.
const QUERY_WARMUP_BITS: u8 = 4;

/// `dim` rounded up to a 64-dim word, matching Lib `padded_dim`.
fn padded_code_dim(dim: usize) -> usize {
    dim.div_ceil(64) * 64
}

/// `u64` words in a transposed 4-bit query: one word per 64 dims per bitplane.
fn query_plane_words(dim: usize) -> usize {
    (padded_code_dim(dim) / 64) * QUERY_WARMUP_BITS as usize
}

/// A query's 4-bit codes and their reconstruction `delta * code + vl`.
#[derive(Debug, Clone)]
struct ScalarCode {
    codes: Vec<u8>,
    delta: f32,
    vl: f32,
    /// L2 norm of the reconstruction error `delta * code + vl - query`.
    error_norm: f32,
}

/// The 4-bit query of the `IVF_HNSW_RQ` walk: bit planes plus Lib `k1xsumq`.
#[derive(Debug, Clone)]
pub struct WarmupQuery {
    /// Query 4-bit planes. dim<512 is Lib block-major; dim>=512 is 512-plane-major.
    pub planes: Vec<u64>,
    pub delta: f32,
    pub vl: f32,
    pub k1xsumq: f32,
    /// Upper bound on `|warmup IP - exact 1-bit IP|` for any binary code.
    pub ip_error_bound: f32,
}

/// The query quantized uniformly over its range to 4-bit codes, each dim
/// reconstructed as `delta * code + vl` to within `delta / 2`. Codes past the
/// query's dims, up to the padded dimension, stay 0.
fn quantize_query(query: &[f32]) -> ScalarCode {
    let (lo, hi) = query
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    let max_code = ((1u32 << QUERY_WARMUP_BITS) - 1) as f32;
    let (vl, delta) = if hi > lo {
        (lo, (hi - lo) / max_code)
    } else {
        (lo.min(0.0).max(hi), 0.0)
    };
    let mut codes = vec![0u8; padded_code_dim(query.len())];
    let mut error_sq = 0.0f32;
    for (code, &v) in codes.iter_mut().zip(query) {
        let level = if delta > 0.0 {
            ((v - vl) / delta).round().clamp(0.0, max_code)
        } else {
            0.0
        };
        *code = level as u8;
        let error = delta * level + vl - v;
        error_sq += error * error;
    }
    ScalarCode {
        codes,
        delta,
        vl,
        error_norm: error_sq.sqrt(),
    }
}

const WARMUP_SUPER_CHUNKS: usize = 8;

/// Lib `new_transpose_bin_512`: dim0 at bit 63; each 512-dim superblock is
/// plane-major (`tq[j * nchunk + k]`).
fn transpose_query_bin_be(codes: &[u8], out: &mut [u64]) {
    debug_assert_eq!(out.len(), query_plane_words(codes.len()));
    out.fill(0);
    let bits = QUERY_WARMUP_BITS as usize;
    let padded = padded_code_dim(codes.len());
    let mut dim = 0;
    let mut out_off = 0;
    while dim < padded {
        let nchunk = ((padded - dim) / 64).min(WARMUP_SUPER_CHUNKS);
        for k in 0..nchunk {
            let start = dim + k * 64;
            for offset in 0..64 {
                let idx = start + offset;
                if idx >= codes.len() {
                    break;
                }
                let code = codes[idx];
                for j in 0..bits {
                    if (code >> j) & 1 == 1 {
                        out[out_off + j * nchunk + k] |= 1u64 << (63 - offset);
                    }
                }
            }
        }
        out_off += nchunk * bits;
        dim += nchunk * 64;
    }
}

/// Lib scalar `warmup_ip_x0_q` layout: `[block0 planes 0..3, block1 planes 0..3, ...]`.
fn transpose_query_bin_block_major(codes: &[u8], out: &mut [u64]) {
    debug_assert_eq!(out.len(), query_plane_words(codes.len()));
    out.fill(0);
    let bits = QUERY_WARMUP_BITS as usize;
    let padded = padded_code_dim(codes.len());
    let nblk = padded / 64;
    for k in 0..nblk {
        let start = k * 64;
        for offset in 0..64 {
            let idx = start + offset;
            if idx >= codes.len() {
                break;
            }
            let code = codes[idx];
            for j in 0..bits {
                if (code >> j) & 1 == 1 {
                    out[k * bits + j] |= 1u64 << (63 - offset);
                }
            }
        }
    }
}

fn transpose_query_for_warmup(codes: &[u8], out: &mut [u64]) {
    if padded_code_dim(codes.len()) >= 512 {
        transpose_query_bin_be(codes, out);
    } else {
        transpose_query_bin_block_major(codes, out);
    }
}

#[inline(always)]
fn warmup_ip_accumulate_block_major(data_be: &[u8], query: &[u64]) -> (u64, u64) {
    let num_blk = data_be.len() / 8;
    debug_assert_eq!(query.len(), num_blk * QUERY_WARMUP_BITS as usize);
    let data = data_be.as_ptr() as *const u64;
    let query = query.as_ptr();
    let mut ip = 0u64;
    let mut ppc = 0u64;
    // SAFETY: `data_be` is `num_blk` little-endian u64 words; query is block-major.
    unsafe {
        for k in 0..num_blk {
            let x = data.add(k).read_unaligned();
            ppc += x.count_ones() as u64;
            let q = query.add(k * QUERY_WARMUP_BITS as usize);
            ip += (x & *q).count_ones() as u64;
            ip += ((x & *q.add(1)).count_ones() as u64) << 1;
            ip += ((x & *q.add(2)).count_ones() as u64) << 2;
            ip += ((x & *q.add(3)).count_ones() as u64) << 3;
        }
    }
    (ip, ppc)
}

#[inline(always)]
fn warmup_ip_accumulate_scalar(data_be: &[u8], query_planes: &[u64]) -> (u64, u64) {
    let num_blk = data_be.len() / 8;
    debug_assert_eq!(query_planes.len(), num_blk * QUERY_WARMUP_BITS as usize);
    let data = data_be.as_ptr() as *const u64;
    let query = query_planes.as_ptr();
    let mut ip = 0u64;
    let mut ppc = 0u64;
    let mut blk = 0;
    let mut q_off = 0;
    while blk < num_blk {
        let nchunk = (num_blk - blk).min(WARMUP_SUPER_CHUNKS);
        // SAFETY: `data_be` is `num_blk` little-endian u64 words; query is
        // plane-major inside each 512-dim superblock (same as Lib).
        unsafe {
            for k in 0..nchunk {
                let x = data.add(blk + k).read_unaligned();
                ppc += x.count_ones() as u64;
                ip += (x & *query.add(q_off + k)).count_ones() as u64;
                ip += ((x & *query.add(q_off + nchunk + k)).count_ones() as u64) << 1;
                ip += ((x & *query.add(q_off + 2 * nchunk + k)).count_ones() as u64) << 2;
                ip += ((x & *query.add(q_off + 3 * nchunk + k)).count_ones() as u64) << 3;
            }
        }
        blk += nchunk;
        q_off += nchunk * QUERY_WARMUP_BITS as usize;
    }
    (ip, ppc)
}

/// Lib scalar `warmup_ip_x0_q`.
#[inline(always)]
fn warmup_ip_x0_q_scalar(data_be: &[u8], query_planes: &[u64], delta: f32, vl: f32) -> f32 {
    debug_assert!(data_be.len().is_multiple_of(8));
    debug_assert_eq!(
        query_planes.len(),
        (data_be.len() / 8) * QUERY_WARMUP_BITS as usize
    );
    let (ip, ppc) = if data_be.len() >= WARMUP_SUPER_CHUNKS * 8 {
        warmup_ip_accumulate_scalar(data_be, query_planes)
    } else {
        warmup_ip_accumulate_block_major(data_be, query_planes)
    };
    delta * ip as f32 + vl * ppc as f32
}

/// Lib `warmup_ip_x0_q_512` when dim>=512 and AVX2 exist; else Lib scalar POPCNT.
#[inline(always)]
pub fn warmup_ip_x0_q(data_be: &[u8], query_planes: &[u64], delta: f32, vl: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        // The AVX2 kernel reads 512-dim superblocks; below that, hardware
        // POPCNT on block-major words is faster.
        if data_be.len() >= WARMUP_SUPER_CHUNKS * 8 && std::is_x86_feature_detected!("avx2") {
            return unsafe { warmup_avx2::warmup_ip_x0_q_512(data_be, query_planes, delta, vl) };
        }
    }
    warmup_ip_x0_q_scalar(data_be, query_planes, delta, vl)
}

#[cfg(target_arch = "x86_64")]
mod warmup_avx2 {
    use super::QUERY_WARMUP_BITS;
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn popcount_avx2(v: __m256i) -> __m256i {
        let lookup = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2,
            3, 3, 4,
        );
        let low_mask = _mm256_set1_epi8(0x0f);
        let lo = _mm256_and_si256(v, low_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
        _mm256_sad_epu8(
            _mm256_add_epi8(
                _mm256_shuffle_epi8(lookup, lo),
                _mm256_shuffle_epi8(lookup, hi),
            ),
            _mm256_setzero_si256(),
        )
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn mm256_reduce_add_epi64(v: __m256i) -> u64 {
        let low = _mm256_castsi256_si128(v);
        let high = _mm256_extracti128_si256::<1>(v);
        let sum = _mm_add_epi64(low, high);
        _mm_extract_epi64::<0>(sum) as u64 + _mm_extract_epi64::<1>(sum) as u64
    }

    /// Lib `warmup_ip_x0_q_512_avx2`. Caller must have checked AVX2 and dim>=512.
    #[target_feature(enable = "avx2")]
    pub unsafe fn warmup_ip_x0_q_512(data_be: &[u8], query: &[u64], delta: f32, vl: f32) -> f32 {
        debug_assert!(data_be.len().is_multiple_of(8));
        let num_blk = data_be.len() / 8;
        debug_assert_eq!(query.len(), num_blk * QUERY_WARMUP_BITS as usize);
        let padded_dim = num_blk * 64;
        let mut data_ptr = data_be.as_ptr() as *const u64;
        let mut query_ptr = query.as_ptr();

        let mut acc_ip = _mm256_setzero_si256();
        let mut acc_ppc = _mm256_setzero_si256();
        let mut acc_bits = [_mm256_setzero_si256(); 4];

        let dim_end_512 = (padded_dim / 512) * 512;
        let mut dim = 0;
        while dim < dim_end_512 {
            let data_lo = _mm256_loadu_si256(data_ptr as *const __m256i);
            let data_hi = _mm256_loadu_si256(data_ptr.add(4) as *const __m256i);
            data_ptr = data_ptr.add(8);
            acc_ppc = _mm256_add_epi64(acc_ppc, popcount_avx2(data_lo));
            acc_ppc = _mm256_add_epi64(acc_ppc, popcount_avx2(data_hi));
            for acc in acc_bits.iter_mut() {
                let q_lo = _mm256_loadu_si256(query_ptr as *const __m256i);
                let q_hi = _mm256_loadu_si256(query_ptr.add(4) as *const __m256i);
                query_ptr = query_ptr.add(8);
                *acc = _mm256_add_epi64(*acc, popcount_avx2(_mm256_and_si256(data_lo, q_lo)));
                *acc = _mm256_add_epi64(*acc, popcount_avx2(_mm256_and_si256(data_hi, q_hi)));
            }
            dim += 512;
        }

        let remaining = padded_dim - dim;
        if remaining > 0 {
            let num_chunks_64 = remaining / 64;
            let num_chunks_32 = remaining / 32;
            let chunks_lo = num_chunks_32.min(8) as i32;
            let chunks_hi = num_chunks_32.saturating_sub(8) as i32;
            let sequence = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            let mask_lo = _mm256_cmpgt_epi32(_mm256_set1_epi32(chunks_lo), sequence);
            let mask_hi = _mm256_cmpgt_epi32(_mm256_set1_epi32(chunks_hi), sequence);
            // SAFETY: `add(4)` is only formed when ≥5 leftover u64 words so the
            // high 256-bit pointer stays inside the allocation. Lib always
            // computes `data+4`; that is UB in Rust when leftover is 1..=3.
            let data_lo = _mm256_maskload_epi32(data_ptr as *const i32, mask_lo);
            let data_hi = if num_chunks_64 > 4 {
                _mm256_maskload_epi32(data_ptr.add(4) as *const i32, mask_hi)
            } else {
                _mm256_setzero_si256()
            };
            acc_ppc = _mm256_add_epi64(acc_ppc, popcount_avx2(data_lo));
            acc_ppc = _mm256_add_epi64(acc_ppc, popcount_avx2(data_hi));
            for acc in acc_bits.iter_mut() {
                let q_lo = _mm256_maskload_epi32(query_ptr as *const i32, mask_lo);
                let q_hi = if num_chunks_64 > 4 {
                    _mm256_maskload_epi32(query_ptr.add(4) as *const i32, mask_hi)
                } else {
                    _mm256_setzero_si256()
                };
                query_ptr = query_ptr.add(num_chunks_64);
                *acc = _mm256_add_epi64(*acc, popcount_avx2(_mm256_and_si256(data_lo, q_lo)));
                *acc = _mm256_add_epi64(*acc, popcount_avx2(_mm256_and_si256(data_hi, q_hi)));
            }
        }

        for (j, acc) in acc_bits.iter().enumerate() {
            let shift = _mm_cvtsi32_si128(j as i32);
            acc_ip = _mm256_add_epi64(acc_ip, _mm256_sll_epi64(*acc, shift));
        }
        let ip = mm256_reduce_add_epi64(acc_ip);
        let ppc = mm256_reduce_add_epi64(acc_ppc);
        delta * ip as f32 + vl * ppc as f32
    }
}

/// Quantizes and transposes the rotated query of the `IVF_HNSW_RQ` walk.
pub fn prepare_warmup_query(rotated: &[f32]) -> WarmupQuery {
    let quantized = quantize_query(rotated);
    let mut planes = vec![0u64; query_plane_words(rotated.len())];
    transpose_query_for_warmup(&quantized.codes, &mut planes);
    let dim = rotated.len();
    let sum_q = rotated.iter().copied().sum::<f32>();
    let code_sum = quantized.codes[..dim]
        .iter()
        .map(|&code| code as u32)
        .sum::<u32>();
    let sum_error = quantized.delta * code_sum as f32 + quantized.vl * dim as f32 - sum_q;
    WarmupQuery {
        planes,
        delta: quantized.delta,
        vl: quantized.vl,
        k1xsumq: sum_q * -0.5,
        // A binary code `b` sums the reconstruction error `err` over its set
        // bits, `(sum(err) + <2b - 1, err>) / 2`, and padded bits are never
        // set; Cauchy-Schwarz bounds `|<2b - 1, err>|` by `sqrt(dim) |err|`.
        ip_error_bound: 0.5 * (sum_error.abs() + (dim as f32).sqrt() * quantized.error_norm),
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    const fn sign_expand_u8(bits: u8) -> u64 {
        let mut word = 0u64;
        let mut i = 0u32;
        while i < 8 {
            if (bits >> (7 - i)) & 1 == 1 {
                word |= 16u64 << (8 * i);
            }
            i += 1;
        }
        word
    }

    const fn make_sign_lut() -> [u64; 256] {
        let mut lut = [0u64; 256];
        let mut i = 0;
        while i < 256 {
            lut[i] = sign_expand_u8(i as u8);
            i += 1;
        }
        lut
    }

    const SIGN_EXPAND: [u64; 256] = make_sign_lut();

    const fn bit_expand_le_u8(bits: u8) -> u64 {
        let mut word = 0u64;
        let mut i = 0u32;
        while i < 8 {
            if (bits >> i) & 1 == 1 {
                word |= 1u64 << (8 * i);
            }
            i += 1;
        }
        word
    }

    const fn make_bit_expand_lut() -> [u64; 256] {
        let mut lut = [0u64; 256];
        let mut i = 0;
        while i < 256 {
            lut[i] = bit_expand_le_u8(i as u8);
            i += 1;
        }
        lut
    }

    /// Lib `kBitExpandLe`: byte `i` of the word is bit `i` of the input byte.
    const BIT_EXPAND_LE: [u64; 256] = make_bit_expand_lut();

    #[inline(always)]
    unsafe fn word_at(bin: *const u8, w: usize) -> u64 {
        // SAFETY: caller sized `bin` to at least `(w + 1) * 8` bytes.
        (bin as *const u64).add(w).read_unaligned()
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn unpack_4bit_xmm(raw8: __m128i) -> __m128i {
        let lo_mask = _mm_set1_epi8(0x0f);
        let lo = _mm_and_si128(raw8, lo_mask);
        let hi = _mm_and_si128(
            _mm_srli_epi16(_mm_and_si128(raw8, _mm_set1_epi8(0xf0u8 as i8)), 4),
            lo_mask,
        );
        _mm_unpacklo_epi64(lo, hi)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn signs_to_bytes(sign_bits: u32) -> __m128i {
        _mm_set_epi64x(
            SIGN_EXPAND[(sign_bits & 255) as usize] as i64,
            SIGN_EXPAND[(sign_bits >> 8) as usize] as i64,
        )
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn decode_u16(packed: *const u8, sign_bits: u32) -> __m128i {
        let raw = _mm_loadl_epi64(packed as *const __m128i);
        _mm_add_epi8(unpack_4bit_xmm(raw), signs_to_bytes(sign_bits))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn acc_16(
        ua: __m128i,
        ub: __m128i,
        ones: __m128i,
        zero: __m128i,
        ip: &mut __m128i,
        sum_a: &mut __m128i,
        sum_b: &mut __m128i,
    ) {
        *ip = _mm_add_epi32(*ip, _mm_madd_epi16(_mm_maddubs_epi16(ua, ub), ones));
        *sum_a = _mm_add_epi64(*sum_a, _mm_sad_epu8(ua, zero));
        *sum_b = _mm_add_epi64(*sum_b, _mm_sad_epu8(ub, zero));
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_epi32_128(v: __m128i) -> i32 {
        let y = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0x4e));
        let z = _mm_add_epi32(y, _mm_shuffle_epi32(y, 0xb1));
        _mm_cvtsi128_si32(z)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_sad_128(v: __m128i) -> i32 {
        _mm_cvtsi128_si32(v) + _mm_extract_epi32(v, 2)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn acc_word_4bit(
        sa: u64,
        sb: u64,
        pa: *const u8,
        pb: *const u8,
        ones: __m128i,
        zero: __m128i,
        ip: &mut __m128i,
        sum_a: &mut __m128i,
        sum_b: &mut __m128i,
    ) {
        acc_16(
            decode_u16(pa, (sa >> 48) as u32),
            decode_u16(pb, (sb >> 48) as u32),
            ones,
            zero,
            ip,
            sum_a,
            sum_b,
        );
        acc_16(
            decode_u16(pa.add(8), ((sa >> 32) as u32) & 0xffff),
            decode_u16(pb.add(8), ((sb >> 32) as u32) & 0xffff),
            ones,
            zero,
            ip,
            sum_a,
            sum_b,
        );
        acc_16(
            decode_u16(pa.add(16), ((sa >> 16) as u32) & 0xffff),
            decode_u16(pb.add(16), ((sb >> 16) as u32) & 0xffff),
            ones,
            zero,
            ip,
            sum_a,
            sum_b,
        );
        acc_16(
            decode_u16(pa.add(24), sa as u32 & 0xffff),
            decode_u16(pb.add(24), sb as u32 & 0xffff),
            ones,
            zero,
            ip,
            sum_a,
            sum_b,
        );
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn decode_2bit(field: __m128i, sign_bits: u32) -> __m128i {
        _mm_add_epi8(field, _mm_srli_epi16(signs_to_bytes(sign_bits), 2))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn acc_word_2bit(
        sa: u64,
        sb: u64,
        pa: *const u8,
        pb: *const u8,
        ones: __m128i,
        zero: __m128i,
        ip: &mut __m128i,
        sum_a: &mut __m128i,
        sum_b: &mut __m128i,
    ) {
        // SAFETY: each 64-dim extra block is 16 bytes for 2-bit codes.
        let raw_a = _mm_loadu_si128(pa as *const __m128i);
        let raw_b = _mm_loadu_si128(pb as *const __m128i);
        let m03 = _mm_set1_epi8(3);
        let m0c = _mm_set1_epi8(0x0c);
        let m30 = _mm_set1_epi8(0x30);
        let mc0 = _mm_set1_epi8(0xc0u8 as i8);
        acc_16(
            decode_2bit(_mm_and_si128(raw_a, m03), (sa >> 48) as u32),
            decode_2bit(_mm_and_si128(raw_b, m03), (sb >> 48) as u32),
            ones,
            zero,
            ip,
            sum_a,
            sum_b,
        );
        acc_16(
            decode_2bit(
                _mm_srli_epi16(_mm_and_si128(raw_a, m0c), 2),
                ((sa >> 32) as u32) & 0xffff,
            ),
            decode_2bit(
                _mm_srli_epi16(_mm_and_si128(raw_b, m0c), 2),
                ((sb >> 32) as u32) & 0xffff,
            ),
            ones,
            zero,
            ip,
            sum_a,
            sum_b,
        );
        acc_16(
            decode_2bit(
                _mm_srli_epi16(_mm_and_si128(raw_a, m30), 4),
                ((sa >> 16) as u32) & 0xffff,
            ),
            decode_2bit(
                _mm_srli_epi16(_mm_and_si128(raw_b, m30), 4),
                ((sb >> 16) as u32) & 0xffff,
            ),
            ones,
            zero,
            ip,
            sum_a,
            sum_b,
        );
        acc_16(
            decode_2bit(
                _mm_srli_epi16(_mm_and_si128(raw_a, mc0), 6),
                sa as u32 & 0xffff,
            ),
            decode_2bit(
                _mm_srli_epi16(_mm_and_si128(raw_b, mc0), 6),
                sb as u32 & 0xffff,
            ),
            ones,
            zero,
            ip,
            sum_a,
            sum_b,
        );
    }

    #[target_feature(enable = "avx2")]
    unsafe fn centered_code_ip_2bit(
        bin_a: &[u8],
        ex_a: &[u8],
        bin_b: &[u8],
        ex_b: &[u8],
        dim: usize,
    ) -> f32 {
        let ones = _mm_set1_epi16(1);
        let zero = _mm_setzero_si128();
        let mut ip = zero;
        let mut sum_a = zero;
        let mut sum_b = zero;
        // Iterate padded 64-dim blocks (see centered_code_ip_8bit).
        let words = bin_a.len() / 8;
        for w in 0..words {
            acc_word_2bit(
                word_at(bin_a.as_ptr(), w),
                word_at(bin_b.as_ptr(), w),
                ex_a.as_ptr().add(w * 16),
                ex_b.as_ptr().add(w * 16),
                ones,
                zero,
                &mut ip,
                &mut sum_a,
                &mut sum_b,
            );
        }
        super::finish_centered(
            hsum_epi32_128(ip) as i64,
            hsum_sad_128(sum_a) as i64,
            hsum_sad_128(sum_b) as i64,
            dim,
            3.5,
        )
    }

    #[target_feature(enable = "avx2")]
    unsafe fn centered_code_ip_4bit(
        bin_a: &[u8],
        ex_a: &[u8],
        bin_b: &[u8],
        ex_b: &[u8],
        dim: usize,
    ) -> f32 {
        let ones = _mm_set1_epi16(1);
        let zero = _mm_setzero_si128();
        let mut ip = zero;
        let mut sum_a = zero;
        let mut sum_b = zero;
        // Iterate padded 64-dim blocks (see centered_code_ip_8bit).
        let words = bin_a.len() / 8;
        for w in 0..words {
            acc_word_4bit(
                word_at(bin_a.as_ptr(), w),
                word_at(bin_b.as_ptr(), w),
                ex_a.as_ptr().add(w * 32),
                ex_b.as_ptr().add(w * 32),
                ones,
                zero,
                &mut ip,
                &mut sum_a,
                &mut sum_b,
            );
        }
        super::finish_centered(
            hsum_epi32_128(ip) as i64,
            hsum_sad_128(sum_a) as i64,
            hsum_sad_128(sum_b) as i64,
            dim,
            15.5,
        )
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn acc_u8_products(a: __m128i, b: __m128i, u16_mask: __m128i, ip_xy: &mut __m128i) {
        let p0 = _mm_mullo_epi16(_mm_cvtepu8_epi16(a), _mm_cvtepu8_epi16(b));
        *ip_xy = _mm_add_epi32(
            *ip_xy,
            _mm_add_epi32(_mm_and_si128(p0, u16_mask), _mm_srli_epi32(p0, 16)),
        );
        let p1 = _mm_mullo_epi16(
            _mm_cvtepu8_epi16(_mm_srli_si128(a, 8)),
            _mm_cvtepu8_epi16(_mm_srli_si128(b, 8)),
        );
        *ip_xy = _mm_add_epi32(
            *ip_xy,
            _mm_add_epi32(_mm_and_si128(p1, u16_mask), _mm_srli_epi32(p1, 16)),
        );
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn acc_16_u8(
        xa: *const u8,
        xb: *const u8,
        sa: u32,
        sb: u32,
        zero: __m128i,
        sign_on: __m128i,
        u16_mask: __m128i,
        ip_xy: &mut __m128i,
        sum_x: &mut __m128i,
        sum_y: &mut __m128i,
        sum_sa_xb: &mut __m128i,
        sum_sb_xa: &mut __m128i,
    ) {
        // SAFETY: each 8-bit extra run is 16 bytes.
        let a = _mm_loadu_si128(xa as *const __m128i);
        let b = _mm_loadu_si128(xb as *const __m128i);
        let ma = _mm_cmpeq_epi8(signs_to_bytes(sa), sign_on);
        let mb = _mm_cmpeq_epi8(signs_to_bytes(sb), sign_on);
        acc_u8_products(a, b, u16_mask, ip_xy);
        *sum_x = _mm_add_epi64(*sum_x, _mm_sad_epu8(a, zero));
        *sum_y = _mm_add_epi64(*sum_y, _mm_sad_epu8(b, zero));
        *sum_sa_xb = _mm_add_epi64(*sum_sa_xb, _mm_sad_epu8(_mm_and_si128(b, ma), zero));
        *sum_sb_xa = _mm_add_epi64(*sum_sb_xa, _mm_sad_epu8(_mm_and_si128(a, mb), zero));
    }

    #[target_feature(enable = "avx2")]
    unsafe fn centered_code_ip_8bit(
        bin_a: &[u8],
        ex_a: &[u8],
        bin_b: &[u8],
        ex_b: &[u8],
        dim: usize,
    ) -> f32 {
        let zero = _mm_setzero_si128();
        let sign_on = _mm_set1_epi8(16);
        let u16_mask = _mm_set1_epi32(0x0000_ffff);
        let mut ip_uu = 0i64;
        let mut sum_x = zero;
        let mut sum_y = zero;
        let mut sum_sa_xb = zero;
        let mut sum_sb_xa = zero;
        let mut n_sa = 0i32;
        let mut n_sb = 0i32;
        let mut n_and = 0i32;
        // Buffers are zero-padded to whole 64-dim blocks, so iterate by block
        // count; the zero tail contributes nothing to the accumulators.
        let words = bin_a.len() / 8;
        for w in 0..words {
            let mut ip_xy = zero;
            let sa = word_at(bin_a.as_ptr(), w);
            let sb = word_at(bin_b.as_ptr(), w);
            let pa = ex_a.as_ptr().add(w * 64);
            let pb = ex_b.as_ptr().add(w * 64);
            let s0 = (sa >> 48) as u32;
            let t0 = (sb >> 48) as u32;
            let s1 = ((sa >> 32) as u32) & 0xffff;
            let t1 = ((sb >> 32) as u32) & 0xffff;
            let s2 = ((sa >> 16) as u32) & 0xffff;
            let t2 = ((sb >> 16) as u32) & 0xffff;
            let s3 = sa as u32 & 0xffff;
            let t3 = sb as u32 & 0xffff;
            acc_16_u8(
                pa,
                pb,
                s0,
                t0,
                zero,
                sign_on,
                u16_mask,
                &mut ip_xy,
                &mut sum_x,
                &mut sum_y,
                &mut sum_sa_xb,
                &mut sum_sb_xa,
            );
            acc_16_u8(
                pa.add(16),
                pb.add(16),
                s1,
                t1,
                zero,
                sign_on,
                u16_mask,
                &mut ip_xy,
                &mut sum_x,
                &mut sum_y,
                &mut sum_sa_xb,
                &mut sum_sb_xa,
            );
            acc_16_u8(
                pa.add(32),
                pb.add(32),
                s2,
                t2,
                zero,
                sign_on,
                u16_mask,
                &mut ip_xy,
                &mut sum_x,
                &mut sum_y,
                &mut sum_sa_xb,
                &mut sum_sb_xa,
            );
            acc_16_u8(
                pa.add(48),
                pb.add(48),
                s3,
                t3,
                zero,
                sign_on,
                u16_mask,
                &mut ip_xy,
                &mut sum_x,
                &mut sum_y,
                &mut sum_sa_xb,
                &mut sum_sb_xa,
            );
            n_sa += sa.count_ones() as i32;
            n_sb += sb.count_ones() as i32;
            n_and += (sa & sb).count_ones() as i32;
            // An i32 sum of 8-bit products overflows past ~33k dims.
            ip_uu += hsum_epi32_128(ip_xy) as i64;
        }
        let ip_uu = ip_uu
            + 256 * (hsum_sad_128(sum_sa_xb) + hsum_sad_128(sum_sb_xa)) as i64
            + 65536 * n_and as i64;
        let sum_a = hsum_sad_128(sum_x) as i64 + 256 * n_sa as i64;
        let sum_b = hsum_sad_128(sum_y) as i64 + 256 * n_sb as i64;
        super::finish_centered(ip_uu, sum_a, sum_b, dim, 255.5)
    }

    /// Lib `extra_1bit_16`: 16 one-bit codes from a little-endian u16 word.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn extra_1bit_16(word: u16) -> __m128i {
        _mm_set_epi64x(
            BIT_EXPAND_LE[(word >> 8) as usize] as i64,
            BIT_EXPAND_LE[(word & 255) as usize] as i64,
        )
    }

    /// Lib `top_bits_group`: byte `b` holds the top bit of dim `16*group + b`
    /// at bit position `p`. The plane packs dim `i` at byte `i % 8`, bit
    /// `i / 8` of the little-endian u64.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn top_bits_group(top: u64, p: u32, group: u32) -> __m128i {
        let high_shift = p as i32 - 1 - 2 * group as i32;
        let low_shift = p as i32 - 2 * group as i32;
        let high = if high_shift >= 0 {
            top << high_shift
        } else {
            top >> (-high_shift) as u32
        };
        let low = if low_shift >= 0 {
            top << low_shift
        } else {
            top >> (-low_shift) as u32
        };
        _mm_and_si128(
            _mm_set_epi64x(high as i64, low as i64),
            _mm_set1_epi8((1u8 << p) as i8),
        )
    }

    /// Lib `sign_contrib_bytes`: each set sign bit contributes `1 << ex_bits`.
    /// Sign bytes start at 16 (`1 << 4`) and shift to the target magnitude;
    /// `ex_bits <= 7` keeps every value inside its byte lane.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sign_contrib_bytes(sign_bits: u32, ex_bits: u8) -> __m128i {
        let s16 = signs_to_bytes(sign_bits);
        if ex_bits <= 4 {
            _mm_srl_epi16(s16, _mm_cvtsi32_si128(4 - ex_bits as i32))
        } else {
            _mm_sll_epi16(s16, _mm_cvtsi32_si128(ex_bits as i32 - 4))
        }
    }

    /// Lib `unpack_2bit_fields`: byte `b` holds dims `b`, `b+16`, `b+32`,
    /// `b+48` at bit pairs 0/2/4/6.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn unpack_2bit_fields(ex: *const u8, out: &mut [__m128i; 4]) {
        // SAFETY: `ex` points to one 64-dim block of 2-bit codes (16 bytes).
        let raw = _mm_loadu_si128(ex as *const __m128i);
        let mask = _mm_set1_epi8(3);
        out[0] = _mm_and_si128(raw, mask);
        out[1] = _mm_and_si128(_mm_srli_epi16(raw, 2), mask);
        out[2] = _mm_and_si128(_mm_srli_epi16(raw, 4), mask);
        out[3] = _mm_and_si128(_mm_srli_epi16(raw, 6), mask);
    }

    /// Lib `unpack_6bit_fields`: three 16-byte chunks hold the low 6 bits of
    /// dims 0-47; dim `48+b` gathers the top 2 bits of byte `b` per chunk.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn unpack_6bit_fields(ex: *const u8, out: &mut [__m128i; 4]) {
        // SAFETY: `ex` points to one 64-dim block of 6-bit codes (48 bytes).
        let cpt1 = _mm_loadu_si128(ex as *const __m128i);
        let cpt2 = _mm_loadu_si128(ex.add(16) as *const __m128i);
        let cpt3 = _mm_loadu_si128(ex.add(32) as *const __m128i);
        let mask6 = _mm_set1_epi8(0x3f);
        let mask2 = _mm_set1_epi8(0xc0u8 as i8);
        out[0] = _mm_and_si128(cpt1, mask6);
        out[1] = _mm_and_si128(cpt2, mask6);
        out[2] = _mm_and_si128(cpt3, mask6);
        out[3] = _mm_or_si128(
            _mm_or_si128(
                _mm_srli_epi16(_mm_and_si128(cpt1, mask2), 6),
                _mm_srli_epi16(_mm_and_si128(cpt2, mask2), 4),
            ),
            _mm_srli_epi16(_mm_and_si128(cpt3, mask2), 2),
        );
    }

    /// Lib `unpack_ex_64`: decode one 64-dim block of blocked-layout extra
    /// codes into four 16-byte groups (`out[g]` = dims `16g..16g+15`).
    ///
    /// `ex` must point to a full 64-dim block of `8 * ex_bits` bytes. Total
    /// widths 3, 5 and 9 have dedicated kernels and never get here.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn unpack_ex_64(ex: *const u8, ex_bits: u8, out: &mut [__m128i; 4]) {
        match ex_bits {
            0 => {
                *out = [_mm_setzero_si128(); 4];
            }
            1 => {
                for (group, slot) in out.iter_mut().enumerate() {
                    // SAFETY: 8 bytes per 64-dim block; each group reads 2.
                    *slot = extra_1bit_16((ex.add(group * 2) as *const u16).read_unaligned());
                }
            }
            3 => {
                unpack_2bit_fields(ex, out);
                // SAFETY: 24-byte block; reads the top-bit plane (bytes 16..24).
                let top = (ex.add(16) as *const u64).read_unaligned();
                for (group, slot) in out.iter_mut().enumerate() {
                    *slot = _mm_or_si128(*slot, top_bits_group(top, 2, group as u32));
                }
            }
            5 => {
                // SAFETY: 40-byte block; two 16-byte chunks + top plane at 32..40.
                let c0 = _mm_loadu_si128(ex as *const __m128i);
                let c1 = _mm_loadu_si128(ex.add(16) as *const __m128i);
                let mask = _mm_set1_epi8(0x0f);
                out[0] = _mm_and_si128(c0, mask);
                out[1] = _mm_and_si128(_mm_srli_epi16(c0, 4), mask);
                out[2] = _mm_and_si128(c1, mask);
                out[3] = _mm_and_si128(_mm_srli_epi16(c1, 4), mask);
                let top = (ex.add(32) as *const u64).read_unaligned();
                for (group, slot) in out.iter_mut().enumerate() {
                    *slot = _mm_or_si128(*slot, top_bits_group(top, 4, group as u32));
                }
            }
            6 => unpack_6bit_fields(ex, out),
            7 => {
                unpack_6bit_fields(ex, out);
                // SAFETY: 56-byte block; reads the top-bit plane (bytes 48..56).
                let top = (ex.add(48) as *const u64).read_unaligned();
                for (group, slot) in out.iter_mut().enumerate() {
                    *slot = _mm_or_si128(*slot, top_bits_group(top, 6, group as u32));
                }
            }
            _ => unreachable!("ex_bits {ex_bits} has a dedicated kernel"),
        }
    }

    /// Lib `decode_u_64`: extra codes plus the sign-bit contribution, giving
    /// the unsigned `num_bits`-wide codes of one 64-dim block.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn decode_u_64(signs: u64, ex: *const u8, ex_bits: u8, codes: &mut [__m128i; 4]) {
        unpack_ex_64(ex, ex_bits, codes);
        let groups = [
            (signs >> 48) as u32,
            ((signs >> 32) & 0xffff) as u32,
            ((signs >> 16) & 0xffff) as u32,
            (signs & 0xffff) as u32,
        ];
        for (code, sign_bits) in codes.iter_mut().zip(groups) {
            *code = _mm_add_epi8(*code, sign_contrib_bytes(sign_bits, ex_bits));
        }
    }

    /// Lib `acc_16_u8_full`: widening u8 products for codes above the
    /// `maddubs` signed range (total bits = 8).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn acc_16_u8_full(
        ua: __m128i,
        ub: __m128i,
        zero: __m128i,
        u16_mask: __m128i,
        ip: &mut __m128i,
        sum_a: &mut __m128i,
        sum_b: &mut __m128i,
    ) {
        acc_u8_products(ua, ub, u16_mask, ip);
        *sum_a = _mm_add_epi64(*sum_a, _mm_sad_epu8(ua, zero));
        *sum_b = _mm_add_epi64(*sum_b, _mm_sad_epu8(ub, zero));
    }

    /// Lib `centered_code_ip_fitted`: one generic AVX2 kernel for total bit
    /// widths 1-8. `maddubs` covers codes up to 127 (bits <= 7); bits = 8
    /// uses widening 16-bit products.
    #[target_feature(enable = "avx2")]
    unsafe fn centered_code_ip_fitted(
        bin_a: &[u8],
        ex_a: &[u8],
        bin_b: &[u8],
        ex_b: &[u8],
        dim: usize,
        num_bits: u8,
    ) -> f32 {
        let ex_bits = num_bits - 1;
        let stride = (64 * ex_bits as usize) / 8;
        let use_maddubs = num_bits <= 7;
        let ones = _mm_set1_epi16(1);
        let zero = _mm_setzero_si128();
        let u16_mask = _mm_set1_epi32(0x0000_ffff);
        let mut ip_uu = 0i64;
        let mut sum_a = zero;
        let mut sum_b = zero;
        // Iterate padded 64-dim blocks (see centered_code_ip_8bit).
        let words = bin_a.len() / 8;
        for w in 0..words {
            let mut ip = zero;
            let mut ua = [zero; 4];
            let mut ub = [zero; 4];
            // SAFETY: `stride` is the per-block extra-code byte count; with
            // `ex_bits == 0` the pointer is never dereferenced.
            decode_u_64(
                word_at(bin_a.as_ptr(), w),
                ex_a.as_ptr().add(w * stride),
                ex_bits,
                &mut ua,
            );
            decode_u_64(
                word_at(bin_b.as_ptr(), w),
                ex_b.as_ptr().add(w * stride),
                ex_bits,
                &mut ub,
            );
            for g in 0..4 {
                if use_maddubs {
                    acc_16(ua[g], ub[g], ones, zero, &mut ip, &mut sum_a, &mut sum_b);
                } else {
                    acc_16_u8_full(
                        ua[g], ub[g], zero, u16_mask, &mut ip, &mut sum_a, &mut sum_b,
                    );
                }
            }
            ip_uu += hsum_epi32_128(ip) as i64;
        }
        super::finish_centered(
            ip_uu,
            hsum_sad_128(sum_a) as i64,
            hsum_sad_128(sum_b) as i64,
            dim,
            super::code_bias(num_bits),
        )
    }

    /// AVX2 [`super::centered_code_ip`]. Both rows are zero-padded to whole
    /// 64-dim blocks, so the kernels iterate every padded block: zero codes
    /// add nothing, and `finish_centered` centers with the true `dim`.
    pub(super) fn centered_code_ip(
        signs_a: &[u8],
        ex_a: &[u8],
        signs_b: &[u8],
        ex_b: &[u8],
        dim: usize,
        num_bits: u8,
    ) -> f32 {
        // SAFETY: `CENTERED_IP` selects this function only when AVX2 is present.
        unsafe {
            match num_bits {
                3 => centered_code_ip_2bit(signs_a, ex_a, signs_b, ex_b, dim),
                5 => centered_code_ip_4bit(signs_a, ex_a, signs_b, ex_b, dim),
                9 => centered_code_ip_8bit(signs_a, ex_a, signs_b, ex_b, dim),
                _ => centered_code_ip_fitted(signs_a, ex_a, signs_b, ex_b, dim, num_bits),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use serde::Deserialize;

    use super::*;
    use crate::vector::bq::ex_dot::{blocked_ex_code_bytes, pack_blocked_row};

    /// Lib outputs: per-vector factors and pairwise `sym_dist`, with f32
    /// values as hex IEEE-754 bit patterns.
    const SYM_DIST_GOLDEN: &str = include_str!("../../../testdata/symrabitq/sym_dist.json");

    #[derive(Deserialize)]
    struct Golden {
        vectors: Vec<GoldenVector>,
        pairs: Vec<GoldenPair>,
    }

    #[derive(Deserialize)]
    struct GoldenVector {
        id: String,
        metric: String,
        num_bits: u8,
        dim: usize,
        bin_hex: String,
        ex_hex: String,
        rho: String,
        gamma: String,
        unorm: String,
        ip_cent: String,
    }

    #[derive(Deserialize)]
    struct GoldenPair {
        a: String,
        b: String,
        sym_dist: String,
    }

    impl GoldenVector {
        /// Lib's `(ρ, γ, ‖ũ‖, ip_cent)` as `κ = ρ / (‖ũ‖ γ)` and the base term.
        fn factors(&self) -> SymFactors {
            let [rho, gamma, unorm, ip_cent] =
                [&self.rho, &self.gamma, &self.unorm, &self.ip_cent].map(|hex| f32_from_hex(hex));
            SymFactors {
                kappa: rho / (unorm * gamma),
                base: match self.distance_type() {
                    DistanceType::Dot => ip_cent,
                    _ => rho * rho,
                },
            }
        }

        fn distance_type(&self) -> DistanceType {
            match self.metric.as_str() {
                "l2" => DistanceType::L2,
                "dot" => DistanceType::Dot,
                other => panic!("unsupported golden metric {other}"),
            }
        }
    }

    fn golden() -> Golden {
        serde_json::from_str(SYM_DIST_GOLDEN).unwrap()
    }

    fn bytes_from_hex(hex: &str) -> Vec<u8> {
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect()
    }

    fn f32_from_hex(hex: &str) -> f32 {
        f32::from_bits(u32::from_str_radix(hex, 16).unwrap())
    }

    fn assert_close(got: f32, expected: f32, context: &str) {
        let err = (got - expected).abs() / expected.abs().max(1.0);
        assert!(err <= 1e-5, "{context}: got {got}, expected {expected}");
    }

    #[test]
    fn test_sym_dist_matches_lib() {
        let golden = golden();
        let vectors: HashMap<_, _> = golden
            .vectors
            .iter()
            .map(|vector| (vector.id.as_str(), vector))
            .collect();
        for pair in &golden.pairs {
            let (a, b) = (vectors[pair.a.as_str()], vectors[pair.b.as_str()]);
            let (signs_a, ex_a) = (bytes_from_hex(&a.bin_hex), bytes_from_hex(&a.ex_hex));
            let (signs_b, ex_b) = (bytes_from_hex(&b.bin_hex), bytes_from_hex(&b.ex_hex));
            let got = sym_dist(
                &SymCode {
                    signs: &signs_a,
                    ex: &ex_a,
                    factors: a.factors(),
                },
                &SymCode {
                    signs: &signs_b,
                    ex: &ex_b,
                    factors: b.factors(),
                },
                a.dim,
                a.num_bits,
                a.distance_type(),
            );
            let context = format!("{}-{}", pair.a, pair.b);
            assert_close(got, f32_from_hex(&pair.sym_dist), &context);
        }
    }

    /// Random codes in the production layout: sign words regrouped from
    /// `_rabit_codes` bytes, ex codes packed with zero padding.
    fn random_code(rng: &mut StdRng, dim: usize, num_bits: u8) -> (Vec<u8>, Vec<u8>) {
        let codes: Vec<u8> = (0..dim / 8).map(|_| rng.random()).collect();
        let signs = sign_words_from_codes(&codes, dim / 8);
        if num_bits == 1 {
            return (signs, Vec::new());
        }
        let ex_bits = num_bits - 1;
        let max_code = ((1u16 << ex_bits) - 1) as u8;
        let values: Vec<u8> = (0..dim).map(|_| rng.random_range(0..=max_code)).collect();
        let mut ex = vec![0; blocked_ex_code_bytes(dim, ex_bits)];
        pack_blocked_row(&values, ex_bits, &mut ex);
        (signs, ex)
    }

    /// At the largest codes, an i32 sum of 8-bit products overflows past
    /// ~33k dims.
    #[test]
    fn test_centered_ip_simd_matches_scalar_past_i32_range() {
        let dim = 34_048;
        for num_bits in [8u8, 9] {
            let ex_bits = num_bits - 1;
            let signs = vec![u8::MAX; sign_word_bytes(dim)];
            let mut ex = vec![0; blocked_ex_code_bytes(dim, ex_bits)];
            pack_blocked_row(&vec![u8::MAX >> (8 - ex_bits); dim], ex_bits, &mut ex);
            let dispatched = CENTERED_IP(&signs, &ex, &signs, &ex, dim, num_bits);
            let scalar = centered_code_ip_scalar(&signs, &ex, &signs, &ex, dim, num_bits);
            assert_close(dispatched, scalar, &format!("num_bits={num_bits}"));
        }
    }

    #[test]
    fn test_centered_ip_simd_matches_scalar() {
        let mut rng = StdRng::seed_from_u64(20260914);
        // 72/104/200 end in a zero-padded 64-dim block the SIMD kernels read.
        for dim in [8usize, 64, 72, 104, 128, 200, 576, 960] {
            for num_bits in 1u8..=9 {
                let (signs_a, ex_a) = random_code(&mut rng, dim, num_bits);
                let (signs_b, ex_b) = random_code(&mut rng, dim, num_bits);
                let dispatched = CENTERED_IP(&signs_a, &ex_a, &signs_b, &ex_b, dim, num_bits);
                let scalar =
                    centered_code_ip_scalar(&signs_a, &ex_a, &signs_b, &ex_b, dim, num_bits);
                assert_close(
                    dispatched,
                    scalar,
                    &format!("dim={dim} num_bits={num_bits}"),
                );
            }
        }
    }

    /// The warmup inner product of a row sums the 4-bit query codes over the
    /// dims whose `_rabit_codes` sign bit is set, in both query layouts.
    #[test]
    fn test_warmup_ip_sums_query_codes_over_set_signs() {
        let mut rng = StdRng::seed_from_u64(7);
        // Below 512 dims the query is block-major; 576/640/960 end in a
        // partial 512-dim superblock.
        for dim in [8usize, 64, 128, 512, 576, 640, 960] {
            let codes: Vec<u8> = (0..dim / 8).map(|_| rng.random()).collect();
            let signs = sign_words_from_codes(&codes, dim / 8);
            let query: Vec<u8> = (0..dim).map(|_| rng.random_range(0..16)).collect();
            let mut planes = vec![0u64; query_plane_words(dim)];
            transpose_query_for_warmup(&query, &mut planes);
            let is_set = |d: usize| (codes[d / 8] >> (d % 8)) & 1 == 1;
            let ip: u32 = (0..dim)
                .filter(|&d| is_set(d))
                .map(|d| query[d] as u32)
                .sum();
            let count = (0..dim).filter(|&d| is_set(d)).count();
            // `delta * ip + vl * popcount(signs)`
            assert_eq!(
                warmup_ip_x0_q(&signs, &planes, 1.0, 0.0),
                ip as f32,
                "{dim}"
            );
            assert_eq!(
                warmup_ip_x0_q_scalar(&signs, &planes, 1.0, 0.0),
                ip as f32,
                "{dim}"
            );
            assert_eq!(
                warmup_ip_x0_q(&signs, &planes, 0.0, 1.0),
                count as f32,
                "{dim}"
            );
        }
    }

    /// The query's extremes map to codes 0 and 15, and every dim reconstructs
    /// to within half a step.
    #[test]
    fn test_quantize_query_spans_its_range() {
        let query: Vec<f32> = (0..72).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
        let got = quantize_query(&query);
        assert_eq!(got.codes.len(), 128);
        assert_eq!(*got.codes[..72].iter().min().unwrap(), 0);
        assert_eq!(*got.codes[..72].iter().max().unwrap(), 15);
        for (&code, &v) in got.codes.iter().zip(&query) {
            assert!((got.delta * code as f32 + got.vl - v).abs() <= got.delta / 2.0 + 1e-6);
        }
    }
}
