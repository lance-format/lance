// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! SymRaBitQ helpers for HNSW graph construction and Residual 1-bit search.
//!
//! BE 1-bit pack plus the symmetric estimator. Residual `distance(id)` uses
//! Lib `quantize_scalar` + `warmup_ip_x0_q`. Flat `distance_all` stays on
//! the existing FastScan path.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::LazyLock;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// Bytes needed for a Lib `pack_binary` column: one `u64` word per 64 dims,
/// rounded up, last word zero-padded.
pub fn sym_bin_code_bytes(code_dim: usize) -> usize {
    code_dim.div_ceil(64) * 8
}

/// Pack residual signs the way RaBitQ-Lib `pack_binary` does:
/// `residual > 0` → 1; each 64-dim word is bit-endian (dim 0 at bit 63);
/// each word is stored as little-endian bytes.
pub fn pack_binary_be(residual: &[f32], out: &mut [u8]) {
    debug_assert_eq!(out.len(), sym_bin_code_bytes(residual.len()));
    out.fill(0);
    for (word_idx, chunk) in out.chunks_exact_mut(8).enumerate() {
        let start = word_idx * 64;
        let end = residual.len().min(start + 64);
        let mut word = 0u64;
        for (offset, &value) in residual[start..end].iter().enumerate() {
            if value > 0.0 {
                word |= 1u64 << (63 - offset);
            }
        }
        chunk.copy_from_slice(&word.to_le_bytes());
    }
}

/// Rewrite LSB-first `_rabit_codes` bytes into Lib `pack_binary` BE words.
pub fn pack_binary_be_from_le(le: &[u8], dim: usize, out: &mut [u8]) {
    debug_assert_eq!(out.len(), sym_bin_code_bytes(dim));
    debug_assert!(le.len() >= dim.div_ceil(8));
    out.fill(0);
    for (word_idx, chunk) in out.chunks_exact_mut(8).enumerate() {
        let start = word_idx * 64;
        let end = dim.min(start + 64);
        let mut word = 0u64;
        for dim_idx in start..end {
            if le[dim_idx / 8] & (1u8 << (dim_idx % 8)) != 0 {
                word |= 1u64 << (63 - (dim_idx - start));
            }
        }
        chunk.copy_from_slice(&word.to_le_bytes());
    }
}

/// Lib `mask_ip_x0_q`: sum of `query[d]` where the BE 1-bit code is set.
///
/// `query` must cover every 64-dim word in `data_be` (zero-pad the tail).
#[inline]
pub fn mask_ip_x0_q(query: &[f32], data_be: &[u8]) -> f32 {
    debug_assert!(data_be.len().is_multiple_of(8));
    debug_assert!(query.len() >= (data_be.len() / 8) * 64);
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return unsafe { mask_ip_avx512::mask_ip_x0_q(query, data_be) };
        }
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { mask_ip_avx2::mask_ip_x0_q(query, data_be) };
        }
    }
    mask_ip_x0_q_scalar(query, data_be)
}

#[inline]
pub(crate) fn mask_ip_x0_q_scalar(query: &[f32], data_be: &[u8]) -> f32 {
    let mut sum = 0.0f32;
    for (word_idx, chunk) in data_be.chunks_exact(8).enumerate() {
        let bits = u64::from_le_bytes(chunk.try_into().expect("8-byte BE word")).reverse_bits();
        let base = word_idx * 64;
        let mut bits = bits;
        for offset in 0..64 {
            if bits & 1 != 0 {
                sum += query[base + offset];
            }
            bits >>= 1;
        }
    }
    sum
}

#[cfg(target_arch = "x86_64")]
mod mask_ip_avx2 {
    use std::arch::x86_64::*;

    /// Lib AVX2 `mask_ip_x0_q`: 8-wide bit-mask AND of the reversed word.
    #[target_feature(enable = "avx2")]
    pub unsafe fn mask_ip_x0_q(query: &[f32], data_be: &[u8]) -> f32 {
        let num_blk = data_be.len() / 8;
        let mut data_ptr = data_be.as_ptr() as *const u64;
        let mut query_ptr = query.as_ptr();
        let bit_checker = _mm256_set_epi32(0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01);
        let mut sum = _mm256_setzero_ps();
        for _ in 0..num_blk {
            let bits = (*data_ptr).reverse_bits();
            data_ptr = data_ptr.add(1);
            for lane in 0..8 {
                let current_byte = ((bits >> (lane * 8)) & 0xff) as i32;
                let v_byte = _mm256_set1_epi32(current_byte);
                let masked_bits = _mm256_and_si256(v_byte, bit_checker);
                let mask = _mm256_cmpgt_epi32(masked_bits, _mm256_setzero_si256());
                let q_vals = _mm256_loadu_ps(query_ptr);
                sum = _mm256_add_ps(sum, _mm256_and_ps(q_vals, _mm256_castsi256_ps(mask)));
                query_ptr = query_ptr.add(8);
            }
        }
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
        lanes.iter().sum()
    }
}

#[cfg(target_arch = "x86_64")]
mod mask_ip_avx512 {
    use std::arch::x86_64::*;

    /// Lib AVX-512 `mask_ip_x0_q`: four 16-wide masked loads per 64-dim word.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn mask_ip_x0_q(query: &[f32], data_be: &[u8]) -> f32 {
        let num_blk = data_be.len() / 8;
        let mut data_ptr = data_be.as_ptr() as *const u64;
        let mut query_ptr = query.as_ptr();
        let mut sum = _mm512_setzero_ps();
        for _ in 0..num_blk {
            let bits = (*data_ptr).reverse_bits();
            data_ptr = data_ptr.add(1);
            sum = _mm512_add_ps(sum, _mm512_maskz_loadu_ps(bits as u16, query_ptr));
            sum = _mm512_add_ps(
                sum,
                _mm512_maskz_loadu_ps((bits >> 16) as u16, query_ptr.add(16)),
            );
            sum = _mm512_add_ps(
                sum,
                _mm512_maskz_loadu_ps((bits >> 32) as u16, query_ptr.add(32)),
            );
            sum = _mm512_add_ps(
                sum,
                _mm512_maskz_loadu_ps((bits >> 48) as u16, query_ptr.add(48)),
            );
            query_ptr = query_ptr.add(64);
        }
        _mm512_reduce_add_ps(sum)
    }
}

/// Symmetric scalars. Not the asymmetric `F_add` / `F_scale` / `F_error` fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymFactors {
    pub rho: f32,
    pub gamma: f32,
    pub unorm: f32,
    pub ip_cent: f32,
}

fn le_u64_at(bytes: &[u8], offset: usize) -> u64 {
    debug_assert!(bytes.len() >= offset + 8);
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("8-byte little-endian word"),
    )
}

fn bin_word(bin_be: &[u8], dim: usize) -> u64 {
    le_u64_at(bin_be, (dim / 64) * 8)
}

fn split_sign_bit(bin_be: &[u8], dim: usize) -> u32 {
    let word = bin_word(bin_be, dim);
    ((word >> (63 - (dim % 64))) & 1) as u32
}

fn split_ex_code(ex: &[u8], dim: usize, ex_bits: u8) -> u32 {
    match ex_bits {
        0 => 0,
        1..=8 => crate::vector::bq::ex_dot::blocked_ex_code_value(ex, dim, ex_bits) as u32,
        _ => unreachable!("ex_bits must be in 0..=8"),
    }
}

fn split_total_code(bin_be: &[u8], ex: &[u8], dim: usize, num_bits: u8) -> u32 {
    let ex_bits = num_bits.saturating_sub(1);
    split_ex_code(ex, dim, ex_bits) + (split_sign_bit(bin_be, dim) << ex_bits)
}

fn code_bias(num_bits: u8) -> f32 {
    ((1u32 << num_bits) - 1) as f32 / 2.0
}

/// ρ, γ, ‖ũ‖, and ip_cent from a rotated residual, its centroid, and codes.
pub fn compute_sym_factors(
    residual: &[f32],
    centroid: &[f32],
    bin_be: &[u8],
    blocked_ex: &[u8],
    num_bits: u8,
) -> SymFactors {
    debug_assert_eq!(residual.len(), centroid.len());
    let dim = residual.len();
    let cb = code_bias(num_bits);
    let mut rho_sq = 0.0f32;
    let mut unorm_sq = 0.0f32;
    let mut ip = 0.0f32;
    let mut ip_cent = 0.0f32;
    let mut c_norm2 = 0.0f32;
    for i in 0..dim {
        let centered = split_total_code(bin_be, blocked_ex, i, num_bits) as f32 - cb;
        rho_sq += residual[i] * residual[i];
        unorm_sq += centered * centered;
        ip += residual[i] * centered;
        ip_cent += residual[i] * centroid[i];
        c_norm2 += centroid[i] * centroid[i];
    }
    ip_cent += 0.5 * c_norm2;
    let rho = rho_sq.sqrt();
    let unorm = unorm_sq.sqrt();
    if rho == 0.0 || unorm == 0.0 {
        return SymFactors {
            rho,
            gamma: 1.0,
            unorm: 1.0,
            ip_cent,
        };
    }
    SymFactors {
        rho,
        gamma: ip / (rho * unorm),
        unorm,
        ip_cent,
    }
}

/// Encode-path factors from unpacked extra codes. Same `r > 0` sign as
/// [`pack_binary_be`]; does not unpack blocked extra.
pub(crate) fn compute_sym_factors_from_values(
    residual: &[f32],
    centroid: &[f32],
    ex_values: &[u8],
    centroid_norm_sq: f32,
    num_bits: u8,
) -> SymFactors {
    debug_assert_eq!(residual.len(), centroid.len());
    let dim = residual.len();
    let ex_bits = num_bits.saturating_sub(1);
    // 1-bit rows carry no ex codes; the slice may be empty then.
    debug_assert!(ex_values.len() == dim || (ex_bits == 0 && ex_values.is_empty()));
    let cb = code_bias(num_bits);
    let mut residual_norm_sq = 0.0f32;
    let mut unorm_sq = 0.0f32;
    let mut ip = 0.0f32;
    let mut ip_cent = 0.0f32;
    for i in 0..dim {
        let sign = u32::from(residual[i] > 0.0);
        let ex = if ex_bits == 0 {
            0
        } else {
            u32::from(ex_values[i])
        };
        let centered = ((sign << ex_bits) + ex) as f32 - cb;
        residual_norm_sq += residual[i] * residual[i];
        unorm_sq += centered * centered;
        ip += residual[i] * centered;
        ip_cent += residual[i] * centroid[i];
    }
    ip_cent += 0.5 * centroid_norm_sq;
    let rho = residual_norm_sq.sqrt();
    let unorm = unorm_sq.sqrt();
    if rho == 0.0 || unorm == 0.0 {
        return SymFactors {
            rho,
            gamma: 1.0,
            unorm: 1.0,
            ip_cent,
        };
    }
    SymFactors {
        rho,
        gamma: ip / (rho * unorm),
        unorm,
        ip_cent,
    }
}

type CenteredIpFn = fn(&[u8], &[u8], &[u8], &[u8], usize, u8) -> f32;

fn select_centered_ip() -> CenteredIpFn {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return x86::centered_code_ip_dispatch;
        }
    }
    centered_code_ip_scalar
}

static CENTERED_IP: LazyLock<CenteredIpFn> = LazyLock::new(select_centered_ip);

fn finish_centered(ip_uu: i32, sum_a: i32, sum_b: i32, dim: usize, cb: f32) -> f32 {
    let center = cb as f64;
    ((ip_uu as f64) - (center * (sum_a + sum_b) as f64) + (dim as f64 * center * center)) as f32
}

fn centered_code_ip_scalar(
    bin_a: &[u8],
    ex_a: &[u8],
    bin_b: &[u8],
    ex_b: &[u8],
    dim: usize,
    num_bits: u8,
) -> f32 {
    if num_bits == 5 && dim.is_multiple_of(64) {
        return centered_code_ip_scalar_4ex(bin_a, ex_a, bin_b, ex_b, dim);
    }
    let cb = code_bias(num_bits);
    let mut ip_uu: i32 = 0;
    let mut sum_a: i32 = 0;
    let mut sum_b: i32 = 0;
    for i in 0..dim {
        let ua = split_total_code(bin_a, ex_a, i, num_bits) as i32;
        let ub = split_total_code(bin_b, ex_b, i, num_bits) as i32;
        ip_uu += ua * ub;
        sum_a += ua;
        sum_b += ub;
    }
    finish_centered(ip_uu, sum_a, sum_b, dim, cb)
}

fn centered_code_ip_scalar_4ex(
    bin_a: &[u8],
    ex_a: &[u8],
    bin_b: &[u8],
    ex_b: &[u8],
    dim: usize,
) -> f32 {
    let mut ip_uu: i32 = 0;
    let mut sum_a: i32 = 0;
    let mut sum_b: i32 = 0;
    let words = dim / 64;
    for w in 0..words {
        let sa = le_u64_at(bin_a, w * 8);
        let sb = le_u64_at(bin_b, w * 8);
        let pa = &ex_a[w * 32..w * 32 + 32];
        let pb = &ex_b[w * 32..w * 32 + 32];
        for g in 0..4 {
            let ga = &pa[g * 8..g * 8 + 8];
            let gb = &pb[g * 8..g * 8 + 8];
            let shift0 = 63 - (g * 16);
            for i in 0..8 {
                let s0 = ((sa >> (shift0 - i)) & 1) as u32;
                let t0 = ((sb >> (shift0 - i)) & 1) as u32;
                let s1 = ((sa >> (shift0 - 8 - i)) & 1) as u32;
                let t1 = ((sb >> (shift0 - 8 - i)) & 1) as u32;
                let ua0 = (ga[i] as u32 & 0x0f) + (s0 << 4);
                let ub0 = (gb[i] as u32 & 0x0f) + (t0 << 4);
                let ua1 = (ga[i] as u32 >> 4) + (s1 << 4);
                let ub1 = (gb[i] as u32 >> 4) + (t1 << 4);
                ip_uu += (ua0 * ub0 + ua1 * ub1) as i32;
                sum_a += (ua0 + ua1) as i32;
                sum_b += (ub0 + ub1) as i32;
            }
        }
    }
    finish_centered(ip_uu, sum_a, sum_b, dim, 15.5)
}

/// ⟨ũ, ṽ⟩ after centering codes at `((1 << bits) - 1) / 2`.
pub(crate) fn centered_code_ip(
    bin_a: &[u8],
    ex_a: &[u8],
    bin_b: &[u8],
    ex_b: &[u8],
    dim: usize,
    num_bits: u8,
) -> f32 {
    debug_assert_eq!(bin_a.len(), sym_bin_code_bytes(dim));
    debug_assert_eq!(bin_b.len(), sym_bin_code_bytes(dim));
    if num_bits > 1 {
        let ex_len = crate::vector::bq::ex_dot::blocked_ex_code_bytes(dim, num_bits - 1);
        debug_assert_eq!(ex_a.len(), ex_len);
        debug_assert_eq!(ex_b.len(), ex_len);
    }
    CENTERED_IP(bin_a, ex_a, bin_b, ex_b, dim, num_bits)
}

/// Lib `sym_dist_split` estimator. L2 and Dot only.
#[allow(clippy::too_many_arguments)]
pub fn sym_dist(
    a: &SymFactors,
    bin_a: &[u8],
    ex_a: &[u8],
    b: &SymFactors,
    bin_b: &[u8],
    ex_b: &[u8],
    dim: usize,
    num_bits: u8,
    distance_type: lance_linalg::distance::DistanceType,
) -> f32 {
    if a.rho == 0.0 && b.rho == 0.0 {
        return match distance_type {
            lance_linalg::distance::DistanceType::Dot => 1.0 - (a.ip_cent + b.ip_cent),
            lance_linalg::distance::DistanceType::L2 => 0.0,
            _ => unreachable!("SymRaBitQ only supports L2 and Dot"),
        };
    }
    let ip_u = centered_code_ip(bin_a, ex_a, bin_b, ex_b, dim, num_bits);
    let ip_bar = ip_u / (a.unorm * b.unorm);
    let hat = ip_bar / (a.gamma * b.gamma);
    match distance_type {
        lance_linalg::distance::DistanceType::Dot => {
            1.0 - (a.ip_cent + b.ip_cent + (a.rho * b.rho * hat))
        }
        lance_linalg::distance::DistanceType::L2 => {
            (a.rho * a.rho) + (b.rho * b.rho) - (2.0 * a.rho * b.rho * hat)
        }
        _ => unreachable!("SymRaBitQ only supports L2 and Dot"),
    }
}

/// Lib `SplitSingleQuery::kNumBits`. Query is 4-bit; data stays 1-bit.
pub(crate) const QUERY_WARMUP_BITS: u8 = 4;
const _: () = assert!(QUERY_WARMUP_BITS == 4);

/// `dim` rounded up to a 64-dim word, matching Lib `padded_dim`.
pub(crate) fn padded_code_dim(dim: usize) -> usize {
    dim.div_ceil(64) * 64
}

/// `u64` words in a transposed 4-bit query: one word per 64 dims per bitplane.
pub(crate) fn query_plane_words(dim: usize) -> usize {
    (padded_code_dim(dim) / 64) * QUERY_WARMUP_BITS as usize
}

/// 4-bit reconstruction codes plus `delta` / `vl` from Lib `quantize_scalar`.
#[derive(Debug, Clone)]
pub(crate) struct ScalarCode {
    pub codes: Vec<u8>,
    pub delta: f32,
    pub vl: f32,
}

/// Query state for Residual `distance(id)`: BE planes + Lib `k1xsumq`.
#[derive(Debug, Clone)]
pub struct ResidualWarmupQuery {
    /// Query 4-bit planes. dim<512 is Lib block-major; dim>=512 is 512-plane-major.
    pub planes: Vec<u64>,
    pub delta: f32,
    pub vl: f32,
    pub k1xsumq: f32,
}

const QUERY_EX_BITS: u8 = QUERY_WARMUP_BITS - 1;
const T_CONST_SAMPLES: usize = 100;
const T_CONST_SEED: u64 = 42;
const TIGHT_START_3EX: f64 = 0.52;

/// `get_const_scaling_factors(padded_dim, 3)` used by `faster_config(dim, 4)`.
///
/// Sampling this is expensive, so callers resolve it once per index rather
/// than once per query. See [`CachedSymPair::t_const`].
pub fn const_scaling_factor_3ex(padded_dim: usize) -> f64 {
    if padded_dim == 0 {
        return 0.0;
    }
    let normal = Normal::new(0.0, 1.0).expect("unit gaussian");
    let mut rng = StdRng::seed_from_u64(T_CONST_SEED);
    let mut sum = 0.0;
    let mut row = vec![0.0f64; padded_dim];
    for _ in 0..T_CONST_SAMPLES {
        let mut norm_sq = 0.0;
        for value in row.iter_mut() {
            *value = normal.sample(&mut rng);
            norm_sq += *value * *value;
        }
        let norm = norm_sq.sqrt();
        if norm == 0.0 {
            continue;
        }
        for value in row.iter_mut() {
            *value = value.abs() / norm;
        }
        sum += best_rescale_factor_3ex(&row);
    }
    sum / T_CONST_SAMPLES as f64
}

fn quantized_level_at_scale(magnitude: f64, t: f64, max_code: i32) -> i32 {
    if magnitude == 0.0 {
        return 0;
    }
    let mut code = (t * magnitude).min(max_code as f64) as i32;
    if code < max_code && (code as f64 + 1.0) / magnitude <= t {
        code += 1;
    } else if code > 0 && code as f64 / magnitude > t {
        code -= 1;
    }
    code
}

fn best_rescale_factor_3ex(o_abs: &[f64]) -> f64 {
    let dim = o_abs.len();
    if dim == 0 {
        return 0.0;
    }
    let max_o = o_abs.iter().copied().fold(0.0, f64::max);
    if max_o == 0.0 {
        return 0.0;
    }
    let max_code = ((1u32 << QUERY_EX_BITS) - 1) as i32;
    let t_end = (max_code as f64 + 10.0) / max_o;
    let t_start = t_end * TIGHT_START_3EX;
    let mut next_t: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
    let mut cur_o_bar = vec![0i32; dim];
    let mut sqr_denominator = dim as f64 * 0.25;
    let mut numerator = 0.0;
    let enqueue = |next_t: &mut BinaryHeap<Reverse<(u64, usize)>>, i: usize, code: i32| {
        let magnitude = o_abs[i];
        if magnitude > 0.0 && code < max_code {
            let next = (code as f64 + 1.0) / magnitude;
            if next < t_end {
                next_t.push(Reverse((next.to_bits(), i)));
            }
        }
    };
    for i in 0..dim {
        let cur = quantized_level_at_scale(o_abs[i], t_start, max_code);
        cur_o_bar[i] = cur;
        sqr_denominator += (cur * cur + cur) as f64;
        numerator += (cur as f64 + 0.5) * o_abs[i];
        enqueue(&mut next_t, i, cur);
    }
    let mut max_ip = numerator / sqr_denominator.sqrt();
    let mut best_t = t_start;
    while let Some(Reverse((t_bits, _))) = next_t.peek().copied() {
        let cur_t = f64::from_bits(t_bits);
        while let Some(Reverse((peek_bits, i))) = next_t.peek().copied() {
            if peek_bits != t_bits {
                break;
            }
            next_t.pop();
            cur_o_bar[i] += 1;
            sqr_denominator += 2.0 * cur_o_bar[i] as f64;
            numerator += o_abs[i];
            enqueue(&mut next_t, i, cur_o_bar[i]);
        }
        let cur_ip = numerator / sqr_denominator.sqrt();
        if cur_ip > max_ip {
            max_ip = cur_ip;
            best_t = cur_t;
        }
    }
    best_t
}

/// Lib `quantize_scalar` with `RECONSTRUCTION` and a precomputed `t_const`.
pub(crate) fn quantize_scalar_reconstruct(query: &[f32], t_const: f64) -> ScalarCode {
    let dim = padded_code_dim(query.len());
    let mut residual = vec![0.0f32; dim];
    residual[..query.len()].copy_from_slice(query);
    let norm = residual
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if norm == 0.0 {
        return ScalarCode {
            codes: vec![0; dim],
            delta: 0.0,
            vl: 0.0,
        };
    }
    let max_code = (1u32 << QUERY_EX_BITS) - 1;
    let mut codes = vec![0u8; dim];
    for i in 0..dim {
        let abs_n = residual[i].abs() / norm;
        let mut code = ((t_const * abs_n as f64) + 1e-5) as i32;
        if code < 0 {
            code = 0;
        }
        if code as u32 > max_code {
            code = max_code as i32;
        }
        if residual[i] > 0.0 {
            code += 1 << QUERY_EX_BITS;
        } else {
            // Lib `ex_bits_code`: r<=0 → `(~tmp) & ((1<<ex_bits)-1)`.
            code = (!code) & (max_code as i32);
        }
        codes[i] = code as u8;
    }
    let cb = -((1u32 << QUERY_EX_BITS) as f32 - 0.5);
    let mut dot = 0.0f32;
    let mut quan_sq = 0.0f32;
    for i in 0..dim {
        let u_cb = codes[i] as f32 + cb;
        dot += residual[i] * u_cb;
        quan_sq += u_cb * u_cb;
    }
    let norm_quan = quan_sq.sqrt();
    let cos = dot / (norm * norm_quan);
    let delta = norm / norm_quan * cos;
    ScalarCode {
        codes,
        delta,
        vl: delta * cb,
    }
}

/// 64-dim words in one Lib `new_transpose_bin_512` superblock.
const WARMUP_SUPER_CHUNKS: usize = 8;

/// Lib `new_transpose_bin_512`: dim0 at bit 63; each 512-dim superblock is
/// plane-major (`tq[j * nchunk + k]`).
pub(crate) fn transpose_query_bin_be(codes: &[u8], out: &mut [u64]) {
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
pub(crate) fn transpose_query_bin_block_major(codes: &[u8], out: &mut [u64]) {
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

/// Lib scalar `warmup_ip_x0_q` (always scalar; for tests / benches).
#[inline(always)]
pub(crate) fn warmup_ip_x0_q_scalar(
    data_be: &[u8],
    query_planes: &[u64],
    delta: f32,
    vl: f32,
) -> f32 {
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
        // AVX2 is a 512-dim kernel. Below that, hardware POPCNT on block-major
        // words beats AVX2 popcount variants: Lib's maskload remainder loses at
        // dim=128, and a per-block broadcast/AND/`popcount_avx2`/sllv kernel
        // loses ~25-50% across dim=64..384 (benches/sym_dist.rs warmup_scan).
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

/// Quantize + transpose a rotated residual query for Residual HNSW search.
/// `t_const` comes from [`const_scaling_factor_3ex`] for the padded dimension.
pub fn prepare_residual_warmup_query(rotated: &[f32], t_const: f64) -> ResidualWarmupQuery {
    let quantized = quantize_scalar_reconstruct(rotated, t_const);
    let mut planes = vec![0u64; query_plane_words(rotated.len())];
    transpose_query_for_warmup(&quantized.codes, &mut planes);
    ResidualWarmupQuery {
        planes,
        delta: quantized.delta,
        vl: quantized.vl,
        k1xsumq: rotated.iter().copied().sum::<f32>() * -0.5,
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
            hsum_epi32_128(ip),
            hsum_sad_128(sum_a),
            hsum_sad_128(sum_b),
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
            hsum_epi32_128(ip),
            hsum_sad_128(sum_a),
            hsum_sad_128(sum_b),
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
        let mut ip_xy = zero;
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
        }
        let ip_uu = hsum_epi32_128(ip_xy)
            + (256 * (hsum_sad_128(sum_sa_xb) + hsum_sad_128(sum_sb_xa)))
            + (65536 * n_and);
        let sum_a = hsum_sad_128(sum_x) + (256 * n_sa);
        let sum_b = hsum_sad_128(sum_y) + (256 * n_sb);
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
    /// `ex` must point to a full 64-dim block of `8 * ex_bits` bytes. Arms
    /// 2/4/8 are never reached by the current dispatch (total bits 3/5/9 use
    /// the specialized kernels); they are kept for parity with the lib's
    /// generic kernel.
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
            2 => unpack_2bit_fields(ex, out),
            3 => {
                unpack_2bit_fields(ex, out);
                // SAFETY: 24-byte block; reads the top-bit plane (bytes 16..24).
                let top = (ex.add(16) as *const u64).read_unaligned();
                for (group, slot) in out.iter_mut().enumerate() {
                    *slot = _mm_or_si128(*slot, top_bits_group(top, 2, group as u32));
                }
            }
            4 => {
                for (group, slot) in out.iter_mut().enumerate() {
                    // SAFETY: 32-byte block; each group reads 8 bytes.
                    let raw = _mm_loadl_epi64(ex.add(group * 8) as *const __m128i);
                    *slot = unpack_4bit_xmm(raw);
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
            8 => {
                for (group, slot) in out.iter_mut().enumerate() {
                    // SAFETY: 64-byte block; each group reads 16 bytes.
                    *slot = _mm_loadu_si128(ex.add(group * 16) as *const __m128i);
                }
            }
            _ => unreachable!("ex_bits must be in 0..=8"),
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
        let mut ip = zero;
        let mut sum_a = zero;
        let mut sum_b = zero;
        // Iterate padded 64-dim blocks (see centered_code_ip_8bit).
        let words = bin_a.len() / 8;
        for w in 0..words {
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
        }
        super::finish_centered(
            hsum_epi32_128(ip),
            hsum_sad_128(sum_a),
            hsum_sad_128(sum_b),
            dim,
            super::code_bias(num_bits),
        )
    }

    #[target_feature(enable = "avx2")]
    unsafe fn centered_code_ip_avx2(
        bin_a: &[u8],
        ex_a: &[u8],
        bin_b: &[u8],
        ex_b: &[u8],
        dim: usize,
        num_bits: u8,
    ) -> f32 {
        match num_bits {
            3 => centered_code_ip_2bit(bin_a, ex_a, bin_b, ex_b, dim),
            5 => centered_code_ip_4bit(bin_a, ex_a, bin_b, ex_b, dim),
            9 => centered_code_ip_8bit(bin_a, ex_a, bin_b, ex_b, dim),
            1..=8 => centered_code_ip_fitted(bin_a, ex_a, bin_b, ex_b, dim, num_bits),
            _ => super::centered_code_ip_scalar(bin_a, ex_a, bin_b, ex_b, dim, num_bits),
        }
    }

    pub(super) fn centered_code_ip_dispatch(
        bin_a: &[u8],
        ex_a: &[u8],
        bin_b: &[u8],
        ex_b: &[u8],
        dim: usize,
        num_bits: u8,
    ) -> f32 {
        if matches!(num_bits, 1..=9) {
            debug_assert_eq!(bin_a.len(), super::sym_bin_code_bytes(dim));
            debug_assert_eq!(bin_b.len(), super::sym_bin_code_bytes(dim));
            if num_bits > 1 {
                let ex_len = crate::vector::bq::ex_dot::blocked_ex_code_bytes(dim, num_bits - 1);
                debug_assert_eq!(ex_a.len(), ex_len);
                debug_assert_eq!(ex_b.len(), ex_len);
            }
            // SAFETY: AVX2 was selected in `select_centered_ip`. Both buffers
            // are zero-padded to whole 64-dim blocks (`sym_bin_code_bytes` /
            // `blocked_ex_code_bytes`), so the kernels may iterate every padded
            // block: zero tail codes add nothing to ip/sum accumulators, and
            // `finish_centered` still centers with the true `dim`.
            unsafe { centered_code_ip_avx2(bin_a, ex_a, bin_b, ex_b, dim, num_bits) }
        } else {
            super::centered_code_ip_scalar(bin_a, ex_a, bin_b, ex_b, dim, num_bits)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use lance_linalg::distance::DistanceType;
    use serde::Deserialize;

    use crate::vector::bq::ex_dot::{blocked_ex_code_bytes, pack_blocked_row};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::{
        QUERY_WARMUP_BITS, ResidualWarmupQuery, SymFactors, centered_code_ip_scalar,
        compute_sym_factors, compute_sym_factors_from_values, const_scaling_factor_3ex,
        mask_ip_x0_q, mask_ip_x0_q_scalar, pack_binary_be, pack_binary_be_from_le, padded_code_dim,
        prepare_residual_warmup_query, quantize_scalar_reconstruct, query_plane_words,
        sym_bin_code_bytes, sym_dist, transpose_query_bin_be, transpose_query_bin_block_major,
        warmup_ip_x0_q, warmup_ip_x0_q_scalar,
    };

    const SYM_DIST_GOLDEN: &str = include_str!("../../../testdata/symrabitq/sym_dist.json");

    fn f32_from_bits_hex(hex: &str) -> f32 {
        let bits = u32::from_str_radix(hex, 16).expect("f32 bits hex");
        f32::from_bits(bits)
    }

    fn to_hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[test]
    fn mask_ip_x0_q_matches_positive_residual_dot() {
        let dim = 960usize;
        let mut rng = StdRng::seed_from_u64(20260915);
        let residual: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.5f32..1.5)).collect();
        let query: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0f32..1.0)).collect();
        let mut bin = vec![0u8; sym_bin_code_bytes(dim)];
        pack_binary_be(&residual, &mut bin);
        let want: f32 = residual
            .iter()
            .zip(query.iter())
            .map(|(&r, &q)| if r > 0.0 { q } else { 0.0 })
            .sum();
        let dispatch = mask_ip_x0_q(&query, &bin);
        let scalar = mask_ip_x0_q_scalar(&query, &bin);
        assert!(
            (dispatch - want).abs() <= 1e-4 * want.abs().max(1.0),
            "dispatch={dispatch} want={want}"
        );
        assert!(
            (scalar - want).abs() <= 1e-4 * want.abs().max(1.0),
            "scalar={scalar} want={want}"
        );

        let mut le = vec![0u8; dim.div_ceil(8)];
        for (dim_idx, &value) in residual.iter().enumerate() {
            if value > 0.0 {
                le[dim_idx / 8] |= 1u8 << (dim_idx % 8);
            }
        }
        let mut from_le = vec![0u8; bin.len()];
        pack_binary_be_from_le(&le, dim, &mut from_le);
        assert_eq!(from_le, bin);
    }

    #[derive(Debug, Deserialize)]
    struct DistGoldenFile {
        vectors: Vec<DistVector>,
        pairs: Vec<DistPair>,
    }

    #[derive(Debug, Deserialize)]
    struct DistVector {
        id: String,
        metric: String,
        num_bits: u8,
        dim: usize,
        centroid_hex: String,
        residual_hex: String,
        bin_hex: String,
        ex_hex: String,
        rho: String,
        gamma: String,
        unorm: String,
        ip_cent: String,
    }

    #[derive(Debug, Deserialize)]
    struct DistPair {
        a: String,
        b: String,
        sym_dist: String,
    }

    fn from_hex(hex: &str) -> Vec<u8> {
        assert_eq!(hex.len() % 2, 0, "odd hex length");
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex byte"))
            .collect()
    }

    /// Golden f32 arrays are stored as concatenated IEEE-754 bit patterns,
    /// 8 hex chars per value.
    fn load_f32_hex(hex: &str) -> Vec<f32> {
        assert_eq!(hex.len() % 8, 0, "f32 hex length");
        (0..hex.len())
            .step_by(8)
            .map(|i| f32_from_bits_hex(&hex[i..i + 8]))
            .collect()
    }

    fn rel_close(got: f32, expected: f32, id: &str, field: &str) {
        let scale = expected.abs().max(1.0);
        let err = (got - expected).abs() / scale;
        assert!(
            err <= 1e-5,
            "{id}.{field}: got {got} expected {expected} rel_err={err}"
        );
    }

    fn metric_type(metric: &str) -> DistanceType {
        match metric {
            "l2" => DistanceType::L2,
            "dot" => DistanceType::Dot,
            other => panic!("unsupported golden metric {other}"),
        }
    }

    fn load_dist_golden() -> DistGoldenFile {
        serde_json::from_str(SYM_DIST_GOLDEN).expect("parse sym_dist.json")
    }

    #[test]
    fn compute_sym_factors_matches_lib_golden() {
        let golden = load_dist_golden();
        let mut saw_l2 = false;
        let mut saw_dot = false;
        let mut saw_zero_rho = false;
        for vec in &golden.vectors {
            let residual = load_f32_hex(&vec.residual_hex);
            let centroid = load_f32_hex(&vec.centroid_hex);
            assert_eq!(residual.len(), vec.dim, "case {}", vec.id);
            let bin = from_hex(&vec.bin_hex);
            let ex = from_hex(&vec.ex_hex);

            let mut packed = vec![0u8; sym_bin_code_bytes(vec.dim)];
            pack_binary_be(&residual, &mut packed);
            assert_eq!(to_hex(&packed), vec.bin_hex, "case {}", vec.id);

            let got = compute_sym_factors(&residual, &centroid, &bin, &ex, vec.num_bits);
            rel_close(got.rho, f32_from_bits_hex(&vec.rho), &vec.id, "rho");
            rel_close(got.gamma, f32_from_bits_hex(&vec.gamma), &vec.id, "gamma");
            rel_close(got.unorm, f32_from_bits_hex(&vec.unorm), &vec.id, "unorm");
            rel_close(
                got.ip_cent,
                f32_from_bits_hex(&vec.ip_cent),
                &vec.id,
                "ip_cent",
            );
            saw_l2 |= vec.metric == "l2";
            saw_dot |= vec.metric == "dot";
            saw_zero_rho |= got.rho == 0.0;
        }
        assert!(saw_l2 && saw_dot && saw_zero_rho);
    }

    #[test]
    fn compute_sym_factors_from_values_matches_packed() {
        use crate::vector::bq::ex_dot::{blocked_ex_code_bytes, pack_blocked_row};

        let mut rng = StdRng::seed_from_u64(20260915);
        for dim in [8usize, 64, 96] {
            for num_bits in [2u8, 5, 8] {
                let ex_bits = num_bits - 1;
                let max_ex = (1u16 << ex_bits) - 1;
                let mut residual = vec![0.0f32; dim];
                let mut centroid = vec![0.0f32; dim];
                let mut ex_values = vec![0u8; dim];
                for i in 0..dim {
                    residual[i] = rng.random_range(-2.0..2.0);
                    centroid[i] = rng.random_range(-1.0..1.0);
                    ex_values[i] = rng.random_range(0..=max_ex) as u8;
                }
                residual[0] = 0.0;
                residual[1] = -0.0;
                residual[2] = -1.5;
                let centroid_norm_sq = centroid.iter().map(|v| v * v).sum::<f32>();
                let mut bin = vec![0u8; super::sym_bin_code_bytes(dim)];
                pack_binary_be(&residual, &mut bin);
                let mut blocked = vec![0u8; blocked_ex_code_bytes(dim, ex_bits)];
                pack_blocked_row(&ex_values, ex_bits, &mut blocked);
                let packed = compute_sym_factors(&residual, &centroid, &bin, &blocked, num_bits);
                let got = compute_sym_factors_from_values(
                    &residual,
                    &centroid,
                    &ex_values,
                    centroid_norm_sq,
                    num_bits,
                );
                rel_close(got.rho, packed.rho, "from_values", "rho");
                rel_close(got.gamma, packed.gamma, "from_values", "gamma");
                rel_close(got.unorm, packed.unorm, "from_values", "unorm");
                rel_close(got.ip_cent, packed.ip_cent, "from_values", "ip_cent");
            }
        }

        let residual = vec![0.0f32; 8];
        let centroid = vec![0.5f32; 8];
        let ex_values = vec![1u8; 8];
        let packed = {
            let mut bin = vec![0u8; super::sym_bin_code_bytes(8)];
            pack_binary_be(&residual, &mut bin);
            let mut blocked = vec![0u8; blocked_ex_code_bytes(8, 1)];
            pack_blocked_row(&ex_values, 1, &mut blocked);
            compute_sym_factors(&residual, &centroid, &bin, &blocked, 2)
        };
        let got = compute_sym_factors_from_values(&residual, &centroid, &ex_values, 2.0, 2);
        assert_eq!(got.gamma, 1.0);
        assert_eq!(got.unorm, 1.0);
        rel_close(got.rho, packed.rho, "zero_rho", "rho");
        rel_close(got.ip_cent, packed.ip_cent, "zero_rho", "ip_cent");
    }

    #[test]
    fn sym_dist_matches_lib_golden_pairs() {
        let golden = load_dist_golden();
        let by_id: HashMap<&str, &DistVector> = golden
            .vectors
            .iter()
            .map(|vec| (vec.id.as_str(), vec))
            .collect();
        assert!(!golden.pairs.is_empty());
        for pair in &golden.pairs {
            let left = by_id[pair.a.as_str()];
            let right = by_id[pair.b.as_str()];
            assert_eq!(left.num_bits, right.num_bits);
            assert_eq!(left.dim, right.dim);
            assert_eq!(left.metric, right.metric);
            let a = decode_vector(left);
            let b = decode_vector(right);
            let got = sym_dist(
                &a.factors,
                &a.bin,
                &a.ex,
                &b.factors,
                &b.bin,
                &b.ex,
                left.dim,
                left.num_bits,
                metric_type(&left.metric),
            );
            rel_close(
                got,
                f32_from_bits_hex(&pair.sym_dist),
                &format!("{}-{}", pair.a, pair.b),
                "sym_dist",
            );
        }
    }

    struct Decoded {
        factors: SymFactors,
        bin: Vec<u8>,
        ex: Vec<u8>,
    }

    fn decode_vector(vec: &DistVector) -> Decoded {
        let residual = load_f32_hex(&vec.residual_hex);
        let centroid = load_f32_hex(&vec.centroid_hex);
        let bin = from_hex(&vec.bin_hex);
        let ex = from_hex(&vec.ex_hex);
        let factors = compute_sym_factors(&residual, &centroid, &bin, &ex, vec.num_bits);
        Decoded { factors, bin, ex }
    }

    #[test]
    fn simd_centered_ip_matches_scalar() {
        let mut rng = StdRng::seed_from_u64(20260914);
        // 72/96/100/200 are not multiples of 64: they pin the zero-padded
        // tail blocks the AVX2 kernels iterate over.
        for dim in [8usize, 64, 72, 96, 100, 128, 200, 576, 960] {
            for num_bits in 1u8..=9 {
                let mut residual_a = vec![0.0f32; dim];
                let mut residual_b = vec![0.0f32; dim];
                for value in residual_a.iter_mut().chain(residual_b.iter_mut()) {
                    *value = rng.random_range(-2.0..2.0);
                }
                let mut bin_a = vec![0u8; super::sym_bin_code_bytes(dim)];
                let mut bin_b = vec![0u8; super::sym_bin_code_bytes(dim)];
                super::pack_binary_be(&residual_a, &mut bin_a);
                super::pack_binary_be(&residual_b, &mut bin_b);
                let ex_len = if num_bits > 1 {
                    blocked_ex_code_bytes(dim, num_bits - 1)
                } else {
                    0
                };
                // The kernels read every padded 64-dim block, so the ex codes
                // must carry the production zero-padding invariant: random
                // per-dim values packed with `pack_blocked_row`, not random
                // bytes (whose padding tail would be nonzero garbage).
                let random_ex = |rng: &mut StdRng| {
                    if num_bits == 1 {
                        return Vec::new();
                    }
                    let ex_bits = num_bits - 1;
                    let max_ex = ((1u32 << ex_bits) - 1) as u8;
                    let values = (0..dim)
                        .map(|_| rng.random_range(0..=max_ex))
                        .collect::<Vec<u8>>();
                    let mut ex = vec![0u8; ex_len];
                    pack_blocked_row(&values, ex_bits, &mut ex);
                    ex
                };
                let ex_a = random_ex(&mut rng);
                let ex_b = random_ex(&mut rng);
                let simd = super::centered_code_ip(&bin_a, &ex_a, &bin_b, &ex_b, dim, num_bits);
                let scalar = centered_code_ip_scalar(&bin_a, &ex_a, &bin_b, &ex_b, dim, num_bits);
                let scale = scalar.abs().max(1.0);
                assert!(
                    (simd - scalar).abs() / scale <= 1e-5,
                    "bits={num_bits} dim={dim} simd={simd} scalar={scalar}"
                );
            }
        }
    }

    #[test]
    fn warmup_ip_x0_q_matches_and_popcount_formula() {
        let data_word = 0xF0F0F0F0F0F0F0F0u64;
        let data_be = data_word.to_le_bytes();
        let planes = [
            0xAAAAAAAAAAAAAAAA,
            0xCCCCCCCCCCCCCCCC,
            0x0000000000000000,
            data_word,
        ];
        let delta = 0.25f32;
        let vl = -1.875f32;
        let mut ip = 0usize;
        let ppc = data_word.count_ones() as usize;
        for (j, plane) in planes.iter().enumerate() {
            ip += ((data_word & plane).count_ones() as usize) << j;
        }
        let want = delta * ip as f32 + vl * ppc as f32;
        let got = warmup_ip_x0_q(&data_be, &planes, delta, vl);
        assert!(
            (got - want).abs() <= 1e-6,
            "warmup_ip_x0_q={got} want={want} ip={ip} ppc={ppc}"
        );
    }

    #[test]
    fn transpose_query_bin_be_puts_dim0_at_bit63_and_lsb_in_plane0() {
        let mut codes = vec![0u8; 64];
        codes[0] = 1;
        codes[1] = 2;
        codes[2] = 4;
        codes[3] = 8;
        let mut planes = vec![0u64; query_plane_words(64)];
        transpose_query_bin_be(&codes, &mut planes);
        assert_eq!(planes.len(), QUERY_WARMUP_BITS as usize);
        assert_eq!(
            planes[0],
            1u64 << 63,
            "LSB of dim0 must sit at bit 63 of plane 0"
        );
        assert_eq!(
            planes[1],
            1u64 << 62,
            "bit1 of dim1 must sit at bit 62 of plane 1"
        );
        assert_eq!(planes[2], 1u64 << 61);
        assert_eq!(planes[3], 1u64 << 60);
    }

    #[test]
    fn transpose_query_bin_block_major_is_sequential_planes() {
        let mut codes = vec![0u8; 128];
        codes[0] = 1;
        codes[64] = 1;
        let mut block = vec![0u64; query_plane_words(128)];
        let mut plane = vec![0u64; query_plane_words(128)];
        transpose_query_bin_block_major(&codes, &mut block);
        transpose_query_bin_be(&codes, &mut plane);
        assert_eq!(block[0], 1u64 << 63, "block0 plane0");
        assert_eq!(block[4], 1u64 << 63, "block1 plane0 follows block0 planes");
        assert_ne!(block, plane, "dim=128 block-major != 512 plane-major");
    }

    #[test]
    fn transpose_query_bin_be_is_plane_major_inside_512_superblock() {
        let mut codes = vec![0u8; 128];
        codes[0] = 1;
        codes[64] = 1;
        let mut planes = vec![0u64; query_plane_words(128)];
        transpose_query_bin_be(&codes, &mut planes);
        assert_eq!(planes.len(), 8);
        assert_eq!(planes[0], 1u64 << 63, "plane0 block0");
        assert_eq!(planes[1], 1u64 << 63, "plane0 block1 sits next to block0");
        assert_eq!(planes[2], 0);
    }

    fn warmup_ip_popcount_oracle(
        data_be: &[u8],
        query: &[u64],
        delta: f32,
        vl: f32,
        plane_major: bool,
    ) -> f32 {
        let nblk = data_be.len() / 8;
        let bits = QUERY_WARMUP_BITS as usize;
        let mut ip = 0u64;
        let mut ppc = 0u64;
        let mut blk = 0;
        let mut q_off = 0;
        while blk < nblk {
            let nchunk = if plane_major { (nblk - blk).min(8) } else { 1 };
            for k in 0..nchunk {
                let off = (blk + k) * 8;
                let x = u64::from_le_bytes(data_be[off..off + 8].try_into().unwrap());
                ppc += x.count_ones() as u64;
                for j in 0..bits {
                    let y = if plane_major {
                        query[q_off + j * nchunk + k]
                    } else {
                        query[(blk + k) * bits + j]
                    };
                    ip += ((x & y).count_ones() as u64) << j;
                }
            }
            blk += nchunk;
            q_off += nchunk * bits;
        }
        delta * ip as f32 + vl * ppc as f32
    }

    #[test]
    fn warmup_ip_dispatch_matches_independent_popcount_oracle() {
        let mut rng = StdRng::seed_from_u64(7);
        // 100/600 are not multiples of 64: they pin partial-tail handling in
        // the block-major layout (100) and a partial 512-superblock (600).
        // 8/32 exercise the smallest block-major inputs.
        for dim in [8usize, 32, 64, 100, 128, 512, 576, 600, 640, 704, 768, 960] {
            let nblk = padded_code_dim(dim) / 64;
            let mut data = vec![0u8; nblk * 8];
            rng.fill(data.as_mut_slice());
            let mut codes = vec![0u8; dim];
            rng.fill(codes.as_mut_slice());
            let mut planes = vec![0u64; query_plane_words(dim)];
            let plane_major = padded_code_dim(dim) >= 512;
            if plane_major {
                transpose_query_bin_be(&codes, &mut planes);
            } else {
                transpose_query_bin_block_major(&codes, &mut planes);
            }
            let delta = 0.31f32;
            let vl = -2.25f32;
            let want = warmup_ip_popcount_oracle(&data, &planes, delta, vl, plane_major);
            let dispatch = warmup_ip_x0_q(&data, &planes, delta, vl);
            let scalar = warmup_ip_x0_q_scalar(&data, &planes, delta, vl);
            assert!(
                (dispatch - want).abs() <= 1e-5,
                "dim={dim} dispatch={dispatch} oracle={want}"
            );
            assert!(
                (scalar - want).abs() <= 1e-5,
                "dim={dim} scalar={scalar} oracle={want}"
            );
        }
    }

    #[test]
    fn prepare_residual_warmup_query_picks_layout_by_dim() {
        let mut rng = StdRng::seed_from_u64(3);
        for (dim, plane_major) in [(128usize, false), (512, true)] {
            let query: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0f32..1.0)).collect();
            let t_const = const_scaling_factor_3ex(padded_code_dim(dim));
            let warmup = prepare_residual_warmup_query(&query, t_const);
            let quantized = quantize_scalar_reconstruct(&query, t_const);
            let mut planes = vec![0u64; query_plane_words(dim)];
            if plane_major {
                transpose_query_bin_be(&quantized.codes, &mut planes);
            } else {
                transpose_query_bin_block_major(&quantized.codes, &mut planes);
            }
            assert_eq!(
                warmup.planes, planes,
                "prepare dim={dim} layout mismatch plane_major={plane_major}"
            );
        }
    }

    #[test]
    fn quantize_scalar_zero_residual_is_zero() {
        let got = quantize_scalar_reconstruct(&[0.0f32; 64], 12.0);
        assert!(got.codes.iter().all(|&code| code == 0));
        assert_eq!(got.delta, 0.0);
        assert_eq!(got.vl, 0.0);
    }

    #[test]
    fn quantize_scalar_flips_ex_bits_when_non_positive() {
        // Lib `ex_bits_code` + `combined_code`: extra bits from |r|/||r||, then
        // r<=0 → `(~tmp) & 7`; r>0 → tmp + (1<<3).
        let mut query = vec![0.0f32; 64];
        query[0] = 1.0;
        query[1] = -1.0;
        let t_const = 14.0;
        let norm = 2.0f32.sqrt();
        let tmp = ((t_const * (1.0 / norm as f64)) + 1e-5) as i32;
        let tmp = tmp.clamp(0, 7);
        let got = quantize_scalar_reconstruct(&query, t_const);
        assert_eq!(
            got.codes[0],
            (tmp + 8) as u8,
            "positive dim keeps extra bits + sign"
        );
        assert_eq!(
            got.codes[1],
            ((!tmp) & 7) as u8,
            "negative dim must invert extra bits, no sign"
        );
        assert_eq!(got.codes[2], 7, "zero dim is non-positive: invert 0 → 7");
    }

    #[test]
    fn quantize_scalar_reconstruct_pins_vl_to_delta() {
        let query: Vec<f32> = (0..64).map(|i| ((i % 11) as f32 - 5.0) * 0.17).collect();
        let got = quantize_scalar_reconstruct(&query, 14.0);
        // Lib derives the lower bound from the code bias, so vl tracks delta.
        assert!((got.vl - got.delta * -7.5).abs() <= 1e-6);
    }

    #[test]
    fn prepare_residual_warmup_query_sets_k1xsumq_and_planes() {
        let mut query = vec![0.0f32; 64];
        for (i, value) in query.iter_mut().enumerate() {
            *value = ((i % 9) as f32 - 4.0) * 0.11;
        }
        let t_const = const_scaling_factor_3ex(padded_code_dim(64));
        assert!(t_const.is_finite() && t_const > 0.0);
        let warmup: ResidualWarmupQuery = prepare_residual_warmup_query(&query, t_const);
        let sum_q: f32 = query.iter().sum();
        assert!((warmup.k1xsumq - sum_q * -0.5).abs() <= 1e-6);
        assert_eq!(warmup.planes.len(), query_plane_words(64));
        let quantized = quantize_scalar_reconstruct(&query, t_const);
        let mut planes = vec![0u64; query_plane_words(64)];
        transpose_query_bin_block_major(&quantized.codes, &mut planes);
        assert_eq!(warmup.planes, planes);
        assert!((warmup.delta - quantized.delta).abs() <= 1e-6);
        assert!((warmup.vl - quantized.vl).abs() <= 1e-6);
    }
}
