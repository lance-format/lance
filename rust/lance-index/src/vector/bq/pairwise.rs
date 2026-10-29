// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Symmetric RQ code-to-code scoring with exact integer code products.
//!
//! A row reconstructs `x̂ = c + s·ō` in the rotated space, where `c` is the
//! partition centroid, `ō_d = code_d − (2^bits − 1)/2` is the centered code and
//! `s` the row's scale. Distances are closed-form in the integer `Σ ōa·ōb` and
//! per-row scalars staged once. Multi-bit codes are staged one integer per
//! dimension (u8 up to 8 bits, u16 for 9); 1-bit codes stay packed sign words.

use std::{ops::Range, sync::Arc};

use arrow::buffer::{Buffer, ScalarBuffer};
use arrow_array::{
    ArrayRef, ArrowPrimitiveType, FixedSizeListArray, Float32Array, Float64Array, PrimitiveArray,
    RecordBatch, UInt64Array,
    cast::AsArray,
    types::{Float32Type, Float64Type, UInt8Type, UInt16Type, UInt64Type},
};
use arrow_schema::{DataType, Field};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, Result};
use lance_linalg::distance::{DistanceType, dot_u8::dot_u8_u64};

use super::{
    builder::RabitQuantizer,
    ex_dot::{blocked_ex_code_bytes, unpack_blocked_row},
    storage::{RABIT_BLOCKED_EX_CODE_COLUMN, RABIT_CODE_COLUMN, RabitQueryEstimator, unpack_codes},
    transform::{EX_SCALE_FACTORS_COLUMN, SCALE_FACTORS_COLUMN},
};
use crate::vector::pairwise::{
    NORM_COLUMN, PairKernel, PairScorer, column_values, l2_to_cosine, list_values, norm_field,
};

const CODES: &str = "__pairwise_rq_codes";
const SCALE: &str = "__pairwise_rq_scale";
const SUM: &str = "__pairwise_rq_sum";
const CODE_NORM: &str = "__pairwise_rq_code_norm";
const CENTER_DOT: &str = "__pairwise_rq_center_dot";

/// Partition-local RQ code scoring.
pub struct RQPairScorer {
    dim: usize,
    bits: u8,
    packed: bool,
    metric: DistanceType,
    /// Rotated partition centroid, for dot and cosine; empty for l2, where
    /// the centroid cancels.
    centroid: Vec<f32>,
    centroid_norm: f32,
}

impl RQPairScorer {
    pub(crate) fn new(
        rq: &RabitQuantizer,
        centroid: ArrayRef,
        metric: DistanceType,
    ) -> Result<Self> {
        let meta = rq.metadata_ref();
        if !(1..=9).contains(&meta.num_bits)
            || meta.rotated_dim() == 0
            || !meta.rotated_dim().is_multiple_of(8)
        {
            return Err(Error::invalid_input(format!(
                "RQ pair scoring requires 1..=9 bits and a positive code dimension divisible by 8, got bits={} dim={}",
                meta.num_bits,
                meta.rotated_dim()
            )));
        }
        if meta.query_estimator != RabitQueryEstimator::RawQuery {
            return Err(Error::not_supported(
                "pair enumeration requires a current RQ index; rebuild the index",
            ));
        }
        if !matches!(
            metric,
            DistanceType::L2 | DistanceType::Cosine | DistanceType::Dot
        ) {
            return Err(Error::not_supported(format!(
                "RQ pair scoring with {metric} distance"
            )));
        }
        let centroid = if metric == DistanceType::L2 {
            Vec::new()
        } else {
            let dim = centroid.len();
            rq.rotate_fsl_to_f32(&FixedSizeListArray::try_new_from_values(
                centroid, dim as i32,
            )?)?
        };
        let centroid_norm = centroid.iter().map(|v| v * v).sum();
        Ok(Self {
            dim: meta.rotated_dim(),
            bits: meta.num_bits,
            packed: meta.packed,
            metric,
            centroid,
            centroid_norm,
        })
    }

    fn words(&self) -> usize {
        self.dim.div_ceil(64)
    }

    /// Stored scale factors are `-2s` (l2, cosine) or `-s` (dot).
    fn scale_divisor(&self) -> f32 {
        if self.metric == DistanceType::Dot {
            -1.0
        } else {
            -2.0
        }
    }
}

impl PairScorer for RQPairScorer {
    type Kernel<'a> = RQKernel<'a>;

    fn row_bytes(&self) -> usize {
        let codes = match self.bits {
            1 => self.words() * 8,
            2..=8 => self.dim,
            _ => self.dim * 2,
        };
        let sum = if self.bits > 1 { 8 } else { 0 };
        let center_dot = if self.centroid.is_empty() { 0 } else { 4 };
        let norm = if self.metric == DistanceType::Cosine {
            8
        } else {
            0
        };
        // Scale and code norm are always staged.
        codes + 8 + sum + center_dot + norm
    }

    fn stage(&self, source: &RecordBatch, rows: Range<usize>) -> Result<Vec<(Field, ArrayRef)>> {
        // Batches start at multiples of 32 rows, so packed sign groups stay whole.
        let source = source.slice(rows.start, rows.len());
        let num_rows = source.num_rows();
        let signs = source
            .column_by_name(RABIT_CODE_COLUMN)
            .and_then(|codes| codes.as_fixed_size_list_opt())
            .ok_or_else(|| Error::invalid_input("RQ pair batch missing sign codes"))?;
        let signs = if self.packed {
            unpack_codes(signs)
        } else {
            signs.clone()
        };
        let signs = signs.values().as_primitive::<UInt8Type>().values();
        let sign_bytes = self.dim / 8;
        if signs.len() != num_rows * sign_bytes {
            return Err(Error::invalid_input(format!(
                "RQ sign codes have {} bytes for {num_rows} rows of dim {}",
                signs.len(),
                self.dim
            )));
        }
        let ex_bits = self.bits - 1;
        let (extended, ex_width) = if ex_bits == 0 {
            (&[][..], 0)
        } else {
            let width = blocked_ex_code_bytes(self.dim, ex_bits);
            let values = list_values::<UInt8Type>(&source, RABIT_BLOCKED_EX_CODE_COLUMN)?;
            if values.len() != num_rows * width {
                return Err(Error::invalid_input(format!(
                    "RQ extended codes have {} bytes for {num_rows} rows of {width} bytes",
                    values.len()
                )));
            }
            (values, width)
        };
        let scale_column = if self.bits == 1 {
            SCALE_FACTORS_COLUMN
        } else {
            EX_SCALE_FACTORS_COLUMN
        };
        let divisor = self.scale_divisor();
        let scales: Vec<f32> = column_values::<Float32Type>(&source, scale_column)?
            .iter()
            .map(|scale| scale / divisor)
            .collect();

        let dim = self.dim;
        let words = self.words();
        let bias = (1i64 << self.bits) - 1;
        let mut sign_words = Vec::new();
        let mut codes_u8 = Vec::new();
        let mut codes_u16 = Vec::new();
        match self.bits {
            1 => sign_words.reserve(num_rows * words),
            2..=8 => codes_u8.reserve(num_rows * dim),
            _ => codes_u16.reserve(num_rows * dim),
        }
        let mut sums = Vec::with_capacity(num_rows);
        let mut code_norms = Vec::with_capacity(num_rows);
        let mut center_dots = Vec::with_capacity(num_rows);
        let mut ex = Vec::new();
        for row in 0..num_rows {
            let signs = &signs[row * sign_bytes..(row + 1) * sign_bytes];
            if ex_bits == 0 {
                // Little-endian words keep dimension d at bit d % 64 of word d / 64.
                for bytes in signs.chunks(8) {
                    let mut word = [0u8; 8];
                    word[..bytes.len()].copy_from_slice(bytes);
                    sign_words.push(u64::from_le_bytes(word));
                }
            } else {
                unpack_blocked_row(
                    &extended[row * ex_width..(row + 1) * ex_width],
                    ex_bits,
                    dim,
                    &mut ex,
                );
            }
            let mut sum = 0u64;
            let mut code_norm = 0i64;
            let mut center_dot = 0.0f32;
            for d in 0..dim {
                let sign = u16::from((signs[d / 8] >> (d % 8)) & 1);
                let code = (sign << ex_bits) | ex.get(d).copied().map_or(0, u16::from);
                match self.bits {
                    1 => {}
                    2..=8 => codes_u8.push(code as u8),
                    _ => codes_u16.push(code),
                }
                let centered = 2 * i64::from(code) - bias;
                sum += u64::from(code);
                code_norm += centered * centered;
                if !self.centroid.is_empty() {
                    center_dot += self.centroid[d] * centered as f32 * 0.5;
                }
            }
            sums.push(sum);
            code_norms.push(code_norm as f32 * 0.25);
            center_dots.push(center_dot);
        }

        let codes: ArrayRef = match self.bits {
            1 => Arc::new(FixedSizeListArray::try_new_from_values(
                aligned::<UInt64Type>(&sign_words),
                words as i32,
            )?),
            2..=8 => Arc::new(FixedSizeListArray::try_new_from_values(
                aligned::<UInt8Type>(&codes_u8),
                dim as i32,
            )?),
            _ => Arc::new(FixedSizeListArray::try_new_from_values(
                aligned::<UInt16Type>(&codes_u16),
                dim as i32,
            )?),
        };
        let norms = (self.metric == DistanceType::Cosine).then(|| {
            // ‖c + s·ō‖² = ‖c‖² + 2s⟨c, ō⟩ + s²‖ō‖²
            let centroid_norm: f64 = self.centroid.iter().map(|&c| f64::from(c).powi(2)).sum();
            scales
                .iter()
                .zip(&center_dots)
                .zip(&code_norms)
                .map(|((&s, &cd), &n)| {
                    let (s, cd, n) = (f64::from(s), f64::from(cd), f64::from(n));
                    (centroid_norm + 2.0 * s * cd + s * s * n).max(0.0).sqrt()
                })
                .collect::<Vec<_>>()
        });
        let mut columns: Vec<(Field, ArrayRef)> = vec![
            (Field::new(CODES, codes.data_type().clone(), false), codes),
            (
                Field::new(SCALE, DataType::Float32, false),
                Arc::new(Float32Array::from(scales)),
            ),
            (
                Field::new(CODE_NORM, DataType::Float32, false),
                Arc::new(Float32Array::from(code_norms)),
            ),
        ];
        if self.bits > 1 {
            columns.push((
                Field::new(SUM, DataType::UInt64, false),
                Arc::new(UInt64Array::from(sums)),
            ));
        }
        if !self.centroid.is_empty() {
            columns.push((
                Field::new(CENTER_DOT, DataType::Float32, false),
                Arc::new(Float32Array::from(center_dots)),
            ));
        }
        if let Some(norms) = norms {
            columns.push((norm_field(), Arc::new(Float64Array::from(norms))));
        }
        Ok(columns)
    }

    fn kernel<'a>(
        &'a self,
        anchor: &'a RecordBatch,
        candidates: &'a RecordBatch,
    ) -> Result<Self::Kernel<'a>> {
        RQKernel::new(self, anchor, candidates, true)
    }
}

/// Copies staged codes into an Arrow-aligned buffer. A large `Vec` from the
/// system allocator is often only 16-byte aligned, which would make every
/// 64-byte SIMD load of a code row straddle two cache lines.
fn aligned<T: ArrowPrimitiveType>(values: &[T::Native]) -> PrimitiveArray<T> {
    PrimitiveArray::new(ScalarBuffer::from(Buffer::from_slice_ref(values)), None)
}

/// Writes `Σ ōa·ōb` of one anchor row against consecutive candidate rows.
/// Every backend computes the same exact integer and rounds it to f32 once,
/// so all backends agree bit for bit.
type SignDotsFn = fn(anchor: &[u64], candidates: &[u64], dim: usize, out: &mut [f32]);
/// [`SignDotsFn`] for one integer code per dimension, given the staged code
/// sums of the anchor and of each candidate, and `bias = 2^bits − 1`.
type CodeDotsFn<T> =
    fn(anchor: &[T], candidates: &[T], anchor_sum: u64, sums: &[u64], bias: i64, out: &mut [f32]);

enum RQCodes<'a> {
    Signs {
        anchor: &'a [u64],
        candidates: &'a [u64],
        dots: SignDotsFn,
    },
    U8 {
        anchor: &'a [u8],
        candidates: &'a [u8],
        dots: CodeDotsFn<u8>,
    },
    U16 {
        anchor: &'a [u16],
        candidates: &'a [u16],
        dots: CodeDotsFn<u16>,
    },
}

impl<'a> RQCodes<'a> {
    /// Resolve the code kernel once per binding. `simd = false` forces the
    /// portable kernels, the reference for the SIMD ones.
    fn bind(
        scorer: &RQPairScorer,
        anchor: &'a RecordBatch,
        candidates: &'a RecordBatch,
        simd: bool,
    ) -> Result<Self> {
        Ok(match scorer.bits {
            1 => Self::Signs {
                anchor: list_values::<UInt64Type>(anchor, CODES)?,
                candidates: list_values::<UInt64Type>(candidates, CODES)?,
                dots: select_sign_dots(simd),
            },
            2..=8 => Self::U8 {
                anchor: list_values::<UInt8Type>(anchor, CODES)?,
                candidates: list_values::<UInt8Type>(candidates, CODES)?,
                dots: select_u8_dots(simd && centered_fits_i32(scorer.dim, scorer.bits)),
            },
            _ => Self::U16 {
                anchor: list_values::<UInt16Type>(anchor, CODES)?,
                candidates: list_values::<UInt16Type>(candidates, CODES)?,
                dots: select_u16_dots(simd && centered_fits_i32(scorer.dim, scorer.bits)),
            },
        })
    }
}

/// SIMD code kernels evaluate `Σ(2ōa)(2ōb)` in wrapping i32 arithmetic, which
/// is exact while its bound `dim·bias²` fits in i32.
fn centered_fits_i32(dim: usize, bits: u8) -> bool {
    let bias = (1u64 << bits) - 1;
    (dim as u64)
        .checked_mul(bias * bias)
        .is_some_and(|bound| bound <= i32::MAX as u64)
}

fn select_sign_dots(simd: bool) -> SignDotsFn {
    #[cfg(target_arch = "x86_64")]
    if simd && *x86::HAS_AVX512_POPCNT {
        // SAFETY: the required CPU features were detected.
        return |anchor, candidates, dim, out| unsafe {
            x86::sign_dots(anchor, candidates, dim, out)
        };
    }
    let _ = simd;
    sign_dots_scalar
}

fn select_u8_dots(simd: bool) -> CodeDotsFn<u8> {
    #[cfg(target_arch = "x86_64")]
    if simd && *x86::HAS_AVX512_VNNI {
        // SAFETY: the required CPU features were detected.
        return |anchor, candidates, anchor_sum, sums, bias, out| unsafe {
            x86::u8_dots(anchor, candidates, anchor_sum, sums, bias, out)
        };
    }
    let _ = simd;
    u8_dots_scalar
}

fn select_u16_dots(simd: bool) -> CodeDotsFn<u16> {
    #[cfg(target_arch = "x86_64")]
    if simd && *x86::HAS_AVX512_VNNI {
        // SAFETY: the required CPU features were detected.
        return |anchor, candidates, anchor_sum, sums, bias, out| unsafe {
            x86::u16_dots(anchor, candidates, anchor_sum, sums, bias, out)
        };
    }
    let _ = simd;
    u16_dots_scalar
}

/// Sign codes center to `±1/2`: `Σ ōa·ōb = (dim − 2·popcount(a ⊕ b)) / 4`.
fn sign_dots_scalar(anchor: &[u64], candidates: &[u64], dim: usize, out: &mut [f32]) {
    for (out, y) in out.iter_mut().zip(candidates.chunks_exact(anchor.len())) {
        let differences: u32 = anchor
            .iter()
            .zip(y)
            .map(|(x, y)| (x ^ y).count_ones())
            .sum();
        *out = (dim as f32 - 2.0 * differences as f32) * 0.25;
    }
}

/// With `2ō = 2·code − bias`:
/// `Σ(2ōa)(2ōb) = 4Σab − 2·bias·(Σa + Σb) + dim·bias²`, exact in i64.
#[inline]
fn centered_from_product(product: u64, sums: u64, dim: usize, bias: i64) -> f32 {
    let centered = 4 * product as i64 - 2 * bias * sums as i64 + dim as i64 * bias * bias;
    centered as f32 * 0.25
}

fn u8_dots_scalar(
    anchor: &[u8],
    candidates: &[u8],
    anchor_sum: u64,
    sums: &[u64],
    bias: i64,
    out: &mut [f32],
) {
    let dim = anchor.len();
    for ((out, y), &sum) in out.iter_mut().zip(candidates.chunks_exact(dim)).zip(sums) {
        *out = centered_from_product(dot_u8_u64(anchor, y), anchor_sum + sum, dim, bias);
    }
}

fn u16_dots_scalar(
    anchor: &[u16],
    candidates: &[u16],
    anchor_sum: u64,
    sums: &[u64],
    bias: i64,
    out: &mut [f32],
) {
    let dim = anchor.len();
    for ((out, y), &sum) in out.iter_mut().zip(candidates.chunks_exact(dim)).zip(sums) {
        let product = anchor
            .iter()
            .zip(y)
            .map(|(&x, &y)| u64::from(x) * u64::from(y))
            .sum();
        *out = centered_from_product(product, anchor_sum + sum, dim, bias);
    }
}

/// AVX-512 code kernels. They score one anchor row against groups of up to 16
/// candidate rows: each anchor vector is loaded once per group and step, every
/// candidate has its own accumulator, and a transpose-add reduces the
/// accumulators to one vector of per-candidate totals. The integers are exact,
/// and the f32 conversion repeats the portable expression, so values are
/// bit-identical to the portable kernels.
#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;
    use std::sync::LazyLock;

    pub(super) static HAS_AVX512_VNNI: LazyLock<bool> = LazyLock::new(|| {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni")
    });

    pub(super) static HAS_AVX512_POPCNT: LazyLock<bool> = LazyLock::new(|| {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vpopcntdq")
    });

    /// Candidates per group, one accumulator register each.
    const GROUP: usize = 16;

    /// Lane `j` of the result is the wrapping sum of the i32 lanes of `acc[j]`.
    #[inline]
    #[target_feature(enable = "avx512f")]
    fn transpose_sum(acc: &[__m512i; GROUP]) -> __m512i {
        // Each 128-bit block of pairs[k] holds partials of [2k, 2k+1, 2k, 2k+1].
        let mut pairs = [_mm512_setzero_si512(); 8];
        for (k, pair) in pairs.iter_mut().enumerate() {
            let (a, b) = (acc[2 * k], acc[2 * k + 1]);
            *pair = _mm512_add_epi32(_mm512_unpacklo_epi32(a, b), _mm512_unpackhi_epi32(a, b));
        }
        // Each 128-bit block of quads[k] holds partials of [4k, .., 4k+3].
        let mut quads = [_mm512_setzero_si512(); 4];
        for (k, quad) in quads.iter_mut().enumerate() {
            let (a, b) = (pairs[2 * k], pairs[2 * k + 1]);
            *quad = _mm512_add_epi32(_mm512_unpacklo_epi64(a, b), _mm512_unpackhi_epi64(a, b));
        }
        // Blocks [0..4, 0..4, 4..8, 4..8], then [0..4, 4..8, 8..12, 12..16].
        let low = fold_blocks(quads[0], quads[1]);
        let high = fold_blocks(quads[2], quads[3]);
        fold_blocks(low, high)
    }

    /// `[a0 + a1, a2 + a3, b0 + b1, b2 + b3]` over 128-bit blocks.
    #[inline]
    #[target_feature(enable = "avx512f")]
    fn fold_blocks(a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi32(
            _mm512_shuffle_i64x2::<0x88>(a, b),
            _mm512_shuffle_i64x2::<0xDD>(a, b),
        )
    }

    /// Lanes `0..m` of a group.
    #[inline]
    fn group_mask(m: usize) -> __mmask16 {
        ((1u32 << m) - 1) as __mmask16
    }

    /// Starts of the `N` rows of width `dim` in a group of `rows.len() / dim`
    /// candidates, repeating the last row to fill the group; lanes past the
    /// real candidates are never stored. Each start has at least `dim`
    /// readable elements, as the slicing is bounds-checked.
    #[inline]
    fn group_rows<T, const N: usize>(rows: &[T], dim: usize) -> [*const T; N] {
        let last = rows.len() / dim - 1;
        std::array::from_fn(|j| rows[j.min(last) * dim..].as_ptr())
    }

    /// Stores lanes `mask` of `(4r + coef·Σb + offset) / 4`, evaluated in
    /// wrapping i32 arithmetic: exact because the true value fits in i32.
    ///
    /// # Safety
    /// The CPU supports AVX-512F, and `mask` is a [`group_mask`] of at most
    /// `sums.len()` and `out.len()` lanes.
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn store_centered(
        r: __m512i,
        coef: i64,
        offset: i64,
        sums: &[u64],
        mask: __mmask16,
        out: &mut [f32],
    ) {
        let lanes = (u16::BITS - mask.leading_zeros()) as usize;
        debug_assert!(
            lanes <= sums.len() && lanes <= out.len(),
            "RQ store mask={mask:#06x} covers {lanes} lanes, sums.len()={}, out.len()={}",
            sums.len(),
            out.len()
        );
        // SAFETY: masked loads read only the lanes set in `mask`, which are in
        // bounds of `sums`; the high half reads nothing unless `mask` sets a
        // lane past 8.
        let (low, high) = unsafe {
            (
                _mm512_maskz_loadu_epi64(mask as __mmask8, sums.as_ptr().cast()),
                _mm512_maskz_loadu_epi64(
                    (mask >> 8) as __mmask8,
                    sums.as_ptr().wrapping_add(8).cast(),
                ),
            )
        };
        let sums = _mm512_inserti64x4::<1>(
            _mm512_castsi256_si512(_mm512_cvtepi64_epi32(low)),
            _mm512_cvtepi64_epi32(high),
        );
        let centered = _mm512_add_epi32(
            _mm512_add_epi32(
                _mm512_slli_epi32::<2>(r),
                _mm512_mullo_epi32(sums, _mm512_set1_epi32(coef as i32)),
            ),
            _mm512_set1_epi32(offset as i32),
        );
        let values = _mm512_mul_ps(_mm512_cvtepi32_ps(centered), _mm512_set1_ps(0.25));
        // SAFETY: the masked store writes only the lanes set in `mask`, which
        // are in bounds of `out`.
        unsafe { _mm512_mask_storeu_ps(out.as_mut_ptr(), mask, values) };
    }

    /// Accumulators for a group of `m` candidates. Short tails use a narrower
    /// group rather than a padded full one; unused accumulators stay zero.
    macro_rules! group_by_width {
        ($m:expr, $group:ident($($arg:expr),*)) => {
            match $m {
                9.. => $group::<16>($($arg),*),
                5..=8 => $group::<8>($($arg),*),
                _ => $group::<4>($($arg),*),
            }
        };
    }

    /// `Σ ōa·ōb` for u8 codes. VPDPBUSD takes the candidate as the unsigned
    /// operand and the anchor biased to `a − 128` as the signed one, so
    /// `r = Σ b·(a − 128)` and `Σab = r + 128·Σb`.
    ///
    /// # Safety
    /// The CPU supports AVX-512F, AVX-512BW and AVX-512 VNNI.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    pub(super) unsafe fn u8_dots(
        anchor: &[u8],
        candidates: &[u8],
        anchor_sum: u64,
        sums: &[u64],
        bias: i64,
        out: &mut [f32],
    ) {
        let dim = anchor.len();
        let n = out.len();
        assert!(
            dim > 0 && candidates.len() >= n * dim && sums.len() >= n,
            "RQ u8 kernel needs n={n} rows of dim={dim} > 0, got candidates.len()={} and sums.len()={}",
            candidates.len(),
            sums.len()
        );
        // 4Σab − 2·bias·(Σa + Σb) + dim·bias² = 4r + coef·Σb + offset.
        let coef = 512 - 2 * bias;
        let offset = dim as i64 * bias * bias - 2 * bias * anchor_sum as i64;
        let mut first = 0;
        while first < n {
            let m = (n - first).min(GROUP);
            let rows = &candidates[first * dim..(first + m) * dim];
            // SAFETY: the caller guarantees this function's CPU features.
            let acc = unsafe { group_by_width!(m, u8_group(anchor, rows)) };
            // SAFETY: as above, and `sums` and `out` hold at least
            // `n - first >= m` values from `first`.
            unsafe {
                store_centered(
                    transpose_sum(&acc),
                    coef,
                    offset,
                    &sums[first..],
                    group_mask(m),
                    &mut out[first..],
                )
            };
            first += m;
        }
    }

    /// Accumulators of `Σ b·(a − 128)` per candidate row of a group.
    ///
    /// # Safety
    /// The CPU supports AVX-512F, AVX-512BW and AVX-512 VNNI.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn u8_group<const N: usize>(anchor: &[u8], rows: &[u8]) -> [__m512i; GROUP] {
        let dim = anchor.len();
        let rows = group_rows::<u8, N>(rows, dim);
        let flip = _mm512_set1_epi8(i8::MIN);
        let x = anchor.as_ptr();
        let steps = dim / 64;
        let mut acc = [_mm512_setzero_si512(); GROUP];
        for step in 0..steps {
            let at = step * 64;
            // SAFETY: `at + 64 <= dim`, and the anchor and every group row
            // hold `dim` bytes.
            unsafe {
                let a = _mm512_xor_si512(_mm512_loadu_si512(x.add(at).cast()), flip);
                for (sum, row) in acc.iter_mut().zip(&rows) {
                    let b = _mm512_loadu_si512(row.add(at).cast());
                    *sum = _mm512_dpbusd_epi32(*sum, b, a);
                }
            }
        }
        let tail = (1u64 << (dim % 64)) - 1;
        if tail != 0 {
            let at = steps * 64;
            // SAFETY: masked loads touch only the `dim - at` in-bounds bytes.
            unsafe {
                let a = _mm512_xor_si512(_mm512_maskz_loadu_epi8(tail, x.add(at).cast()), flip);
                for (sum, row) in acc.iter_mut().zip(&rows) {
                    let b = _mm512_maskz_loadu_epi8(tail, row.add(at).cast());
                    *sum = _mm512_dpbusd_epi32(*sum, b, a);
                }
            }
        }
        acc
    }

    /// `Σ ōa·ōb` for u16 codes below 2^15, which VPDPWSSD multiplies exactly
    /// as signed words.
    ///
    /// # Safety
    /// The CPU supports AVX-512F, AVX-512BW and AVX-512 VNNI.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    pub(super) unsafe fn u16_dots(
        anchor: &[u16],
        candidates: &[u16],
        anchor_sum: u64,
        sums: &[u64],
        bias: i64,
        out: &mut [f32],
    ) {
        let dim = anchor.len();
        let n = out.len();
        assert!(
            dim > 0 && candidates.len() >= n * dim && sums.len() >= n,
            "RQ u16 kernel needs n={n} rows of dim={dim} > 0, got candidates.len()={} and sums.len()={}",
            candidates.len(),
            sums.len()
        );
        let coef = -2 * bias;
        let offset = dim as i64 * bias * bias - 2 * bias * anchor_sum as i64;
        let mut first = 0;
        while first < n {
            let m = (n - first).min(GROUP);
            let rows = &candidates[first * dim..(first + m) * dim];
            // SAFETY: the caller guarantees this function's CPU features.
            let acc = unsafe { group_by_width!(m, u16_group(anchor, rows)) };
            // SAFETY: as above, and `sums` and `out` hold at least
            // `n - first >= m` values from `first`.
            unsafe {
                store_centered(
                    transpose_sum(&acc),
                    coef,
                    offset,
                    &sums[first..],
                    group_mask(m),
                    &mut out[first..],
                )
            };
            first += m;
        }
    }

    /// Accumulators of `Σ a·b` per candidate row of a group.
    ///
    /// # Safety
    /// The CPU supports AVX-512F, AVX-512BW and AVX-512 VNNI.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn u16_group<const N: usize>(anchor: &[u16], rows: &[u16]) -> [__m512i; GROUP] {
        let dim = anchor.len();
        let rows = group_rows::<u16, N>(rows, dim);
        let x = anchor.as_ptr();
        let steps = dim / 32;
        let mut acc = [_mm512_setzero_si512(); GROUP];
        for step in 0..steps {
            let at = step * 32;
            // SAFETY: `at + 32 <= dim`, and the anchor and every group row
            // hold `dim` words.
            unsafe {
                let a = _mm512_loadu_si512(x.add(at).cast());
                for (sum, row) in acc.iter_mut().zip(&rows) {
                    let b = _mm512_loadu_si512(row.add(at).cast());
                    *sum = _mm512_dpwssd_epi32(*sum, a, b);
                }
            }
        }
        let tail = ((1u64 << (dim % 32)) - 1) as __mmask32;
        if tail != 0 {
            let at = steps * 32;
            // SAFETY: masked loads touch only the `dim - at` in-bounds words.
            unsafe {
                let a = _mm512_maskz_loadu_epi16(tail, x.add(at).cast());
                for (sum, row) in acc.iter_mut().zip(&rows) {
                    let b = _mm512_maskz_loadu_epi16(tail, row.add(at).cast());
                    *sum = _mm512_dpwssd_epi32(*sum, a, b);
                }
            }
        }
        acc
    }

    /// `(dim − 2·popcount(a ⊕ b)) / 4` for packed sign words.
    ///
    /// # Safety
    /// The CPU supports AVX-512F and AVX-512 VPOPCNTDQ.
    #[target_feature(enable = "avx512f,avx512vpopcntdq")]
    pub(super) unsafe fn sign_dots(
        anchor: &[u64],
        candidates: &[u64],
        dim: usize,
        out: &mut [f32],
    ) {
        let words = anchor.len();
        let n = out.len();
        assert!(
            words > 0 && candidates.len() >= n * words,
            "RQ sign kernel needs n={n} rows of {words} > 0 words (dim={dim}), got candidates.len()={}",
            candidates.len()
        );
        let dim = _mm512_set1_ps(dim as f32);
        let mut first = 0;
        while first < n {
            let m = (n - first).min(GROUP);
            let rows = &candidates[first * words..(first + m) * words];
            // SAFETY: the caller guarantees this function's CPU features.
            let acc = unsafe { group_by_width!(m, sign_group(anchor, rows)) };
            let differences = _mm512_cvtepi32_ps(transpose_sum(&acc));
            let values = _mm512_mul_ps(
                _mm512_sub_ps(dim, _mm512_mul_ps(_mm512_set1_ps(2.0), differences)),
                _mm512_set1_ps(0.25),
            );
            // SAFETY: the masked store writes only the first `m` lanes, and
            // `out` holds `n - first >= m` floats from `first`.
            unsafe { _mm512_mask_storeu_ps(out[first..].as_mut_ptr(), group_mask(m), values) };
            first += m;
        }
    }

    /// Accumulators of `popcount(a ⊕ b)` per candidate row of a group.
    ///
    /// # Safety
    /// The CPU supports AVX-512F and AVX-512 VPOPCNTDQ.
    #[inline]
    #[target_feature(enable = "avx512f,avx512vpopcntdq")]
    unsafe fn sign_group<const N: usize>(anchor: &[u64], rows: &[u64]) -> [__m512i; GROUP] {
        let words = anchor.len();
        let rows = group_rows::<u64, N>(rows, words);
        let x = anchor.as_ptr();
        let steps = words / 8;
        let mut acc = [_mm512_setzero_si512(); GROUP];
        for step in 0..steps {
            let at = step * 8;
            // SAFETY: `at + 8 <= words`, and the anchor and every group row
            // hold `words` words.
            unsafe {
                let a = _mm512_loadu_si512(x.add(at).cast());
                for (sum, row) in acc.iter_mut().zip(&rows) {
                    let b = _mm512_loadu_si512(row.add(at).cast());
                    *sum = _mm512_add_epi32(*sum, _mm512_popcnt_epi32(_mm512_xor_si512(a, b)));
                }
            }
        }
        let tail = ((1u32 << (words % 8)) - 1) as __mmask8;
        if tail != 0 {
            let at = steps * 8;
            // SAFETY: masked loads touch only the `words - at` in-bounds words.
            unsafe {
                let a = _mm512_maskz_loadu_epi64(tail, x.add(at).cast());
                for (sum, row) in acc.iter_mut().zip(&rows) {
                    let b = _mm512_maskz_loadu_epi64(tail, row.add(at).cast());
                    *sum = _mm512_add_epi32(*sum, _mm512_popcnt_epi32(_mm512_xor_si512(a, b)));
                }
            }
        }
        acc
    }
}

/// Per-row scalars of one staged batch.
struct RQRows<'a> {
    scales: &'a [f32],
    code_norms: &'a [f32],
    sums: &'a [u64],
    center_dots: &'a [f32],
    norms: &'a [f64],
}

impl<'a> RQRows<'a> {
    fn try_new(scorer: &RQPairScorer, batch: &'a RecordBatch) -> Result<Self> {
        Ok(Self {
            scales: column_values::<Float32Type>(batch, SCALE)?,
            code_norms: column_values::<Float32Type>(batch, CODE_NORM)?,
            sums: if scorer.bits > 1 {
                column_values::<UInt64Type>(batch, SUM)?
            } else {
                &[]
            },
            center_dots: if scorer.centroid.is_empty() {
                &[]
            } else {
                column_values::<Float32Type>(batch, CENTER_DOT)?
            },
            norms: if scorer.metric == DistanceType::Cosine {
                column_values::<Float64Type>(batch, NORM_COLUMN)?
            } else {
                &[]
            },
        })
    }
}

pub struct RQKernel<'a> {
    scorer: &'a RQPairScorer,
    codes: RQCodes<'a>,
    anchor: RQRows<'a>,
    candidates: RQRows<'a>,
}

impl<'a> RQKernel<'a> {
    fn new(
        scorer: &'a RQPairScorer,
        anchor: &'a RecordBatch,
        candidates: &'a RecordBatch,
        simd: bool,
    ) -> Result<Self> {
        Ok(Self {
            scorer,
            codes: RQCodes::bind(scorer, anchor, candidates, simd)?,
            anchor: RQRows::try_new(scorer, anchor)?,
            candidates: RQRows::try_new(scorer, candidates)?,
        })
    }

    /// `Σ ōa·ōb` per candidate. Integer products are exact, so the result
    /// does not depend on the evaluation order or the kernel.
    fn centered_dots(&self, anchor_row: usize, candidates: Range<usize>, out: &mut [f32]) {
        let dim = self.scorer.dim;
        let bias = (1i64 << self.scorer.bits) - 1;
        match &self.codes {
            RQCodes::Signs {
                anchor,
                candidates: values,
                dots,
            } => {
                let words = self.scorer.words();
                let x = &anchor[anchor_row * words..(anchor_row + 1) * words];
                let y = &values[candidates.start * words..candidates.end * words];
                dots(x, y, dim, out);
            }
            RQCodes::U8 {
                anchor,
                candidates: values,
                dots,
            } => {
                let x = &anchor[anchor_row * dim..(anchor_row + 1) * dim];
                let y = &values[candidates.start * dim..candidates.end * dim];
                let sums = &self.candidates.sums[candidates];
                dots(x, y, self.anchor.sums[anchor_row], sums, bias, out);
            }
            RQCodes::U16 {
                anchor,
                candidates: values,
                dots,
            } => {
                let x = &anchor[anchor_row * dim..(anchor_row + 1) * dim];
                let y = &values[candidates.start * dim..candidates.end * dim];
                let sums = &self.candidates.sums[candidates];
                dots(x, y, self.anchor.sums[anchor_row], sums, bias, out);
            }
        }
    }
}

impl PairKernel for RQKernel<'_> {
    fn distances(&self, anchor_row: usize, candidates: Range<usize>, out: &mut [f32]) {
        self.centered_dots(anchor_row, candidates.clone(), out);
        let a_scale = self.anchor.scales[anchor_row];
        let b_scales = &self.candidates.scales[candidates.clone()];
        if self.scorer.metric == DistanceType::Dot {
            let a_center = self.anchor.center_dots[anchor_row];
            let b_centers = &self.candidates.center_dots[candidates];
            for ((out, &scale), &center) in out.iter_mut().zip(b_scales).zip(b_centers) {
                *out = 1.0
                    - (self.scorer.centroid_norm
                        + a_scale * a_center
                        + scale * center
                        + a_scale * scale * *out);
            }
            return;
        }
        let a_norm = self.anchor.code_norms[anchor_row];
        let b_norms = &self.candidates.code_norms[candidates.clone()];
        for ((out, &scale), &norm) in out.iter_mut().zip(b_scales).zip(b_norms) {
            // Identical codes and scales cancel exactly; negative roundoff
            // elsewhere is not a negative squared distance.
            let squared =
                a_scale * a_scale * a_norm + scale * scale * norm - 2.0 * a_scale * scale * *out;
            *out = if squared < 0.0 { 0.0 } else { squared };
        }
        if self.scorer.metric == DistanceType::Cosine {
            l2_to_cosine(
                out,
                self.anchor.norms[anchor_row],
                &self.candidates.norms[candidates],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{ex_dot::pack_blocked_row, storage::pack_codes};
    use super::*;
    use arrow_array::{Array, UInt8Array};
    use arrow_schema::Schema;
    use lance_core::ROW_ID;
    use rstest::rstest;

    fn staged(scorer: &RQPairScorer, source: &RecordBatch, rows: Range<usize>) -> RecordBatch {
        RecordBatch::try_from_iter(
            scorer
                .stage(source, rows)
                .unwrap()
                .into_iter()
                .map(|(field, column)| (field.name().clone(), column)),
        )
        .unwrap()
    }

    #[rstest]
    #[case::rq1(1)]
    #[case::rq2(2)]
    #[case::rq3(3)]
    #[case::rq4(4)]
    #[case::rq5(5)]
    #[case::rq6(6)]
    #[case::rq7(7)]
    #[case::rq8(8)]
    #[case::rq9(9)]
    fn test_native_batch_distance(
        #[case] bits: u8,
        #[values(false, true)] packed: bool,
        #[values(DistanceType::L2, DistanceType::Cosine, DistanceType::Dot)] metric: DistanceType,
    ) {
        // 72 dimensions exercise padding in sign words and extended codes;
        // 35 rows exercise both a full packed group and its partial tail.
        let dim = 72;
        let rows = 35;
        let centroid: Vec<f32> = (0..dim).map(|d| (d % 7) as f32 * 0.125).collect();
        let scorer = RQPairScorer {
            dim,
            bits,
            packed,
            metric,
            centroid_norm: centroid.iter().map(|v| v * v).sum(),
            centroid: if metric == DistanceType::L2 {
                Vec::new()
            } else {
                centroid.clone()
            },
        };
        let mask = (1u16 << bits) - 1;
        let ex_bits = bits - 1;
        let mut sign_codes = vec![0u8; rows * dim / 8];
        let ex_width = if ex_bits == 0 {
            0
        } else {
            blocked_ex_code_bytes(dim, ex_bits)
        };
        let mut ex_codes = vec![0u8; rows * ex_width];
        let mut exact = Vec::new();
        let scales: Vec<f32> = (0..rows)
            .map(|r| match r {
                0 => 0.0,
                // Row 2 duplicates row 1's codes and scale.
                2 => 0.125,
                _ => (r % 4 + 1) as f32 * 0.0625,
            })
            .collect();
        for (row, &scale) in scales.iter().enumerate() {
            let seed = if row == 2 { 1 } else { row };
            let codes: Vec<u16> = (0..dim)
                .map(|d| ((seed * 17 + d * 13) as u16) & mask)
                .collect();
            for (d, &code) in codes.iter().enumerate() {
                sign_codes[row * (dim / 8) + d / 8] |= ((code >> ex_bits) as u8) << (d % 8);
            }
            if ex_bits > 0 {
                let ex: Vec<u8> = codes
                    .iter()
                    .map(|c| (c & ((1 << ex_bits) - 1)) as u8)
                    .collect();
                pack_blocked_row(
                    &ex,
                    ex_bits,
                    &mut ex_codes[row * ex_width..(row + 1) * ex_width],
                );
            }
            // Exactly representable lattice points provide an independent
            // scalar oracle, including scale zero and nonzero centroids.
            exact.push(
                codes
                    .iter()
                    .enumerate()
                    .map(|(d, &q)| {
                        f64::from(centroid[d])
                            + f64::from(scale) * (f64::from(q) - f64::from(mask) * 0.5)
                    })
                    .collect::<Vec<_>>(),
            );
        }
        let signs =
            FixedSizeListArray::try_new_from_values(UInt8Array::from(sign_codes), (dim / 8) as i32)
                .unwrap();
        let signs = if packed { pack_codes(&signs) } else { signs };
        let scale_name = if bits == 1 {
            SCALE_FACTORS_COLUMN
        } else {
            EX_SCALE_FACTORS_COLUMN
        };
        let divisor = scorer.scale_divisor();
        let mut fields = vec![
            Field::new(ROW_ID, DataType::UInt64, false),
            Field::new(RABIT_CODE_COLUMN, signs.data_type().clone(), false),
            Field::new(scale_name, DataType::Float32, false),
        ];
        let mut arrays: Vec<ArrayRef> = vec![
            Arc::new(UInt64Array::from_iter_values(0..rows as u64)),
            Arc::new(signs),
            Arc::new(Float32Array::from(
                scales.iter().map(|s| s * divisor).collect::<Vec<_>>(),
            )),
        ];
        if bits > 1 {
            let ex = FixedSizeListArray::try_new_from_values(
                UInt8Array::from(ex_codes),
                ex_width as i32,
            )
            .unwrap();
            fields.push(Field::new(
                RABIT_BLOCKED_EX_CODE_COLUMN,
                ex.data_type().clone(),
                false,
            ));
            arrays.push(Arc::new(ex));
        }
        let source = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap();
        // A full packed group followed by the partition's unpacked tail.
        let head = staged(&scorer, &source, 0..32);
        let tail = staged(&scorer, &source, 32..rows);
        assert!(head.column_by_name(RABIT_CODE_COLUMN).is_none());
        // Code rows start cache-line aligned for the SIMD kernels.
        let codes = head[CODES].as_fixed_size_list().values().to_data();
        assert_eq!(codes.buffers()[0].as_ptr().align_offset(64), 0);
        assert_eq!(
            head[CODES].as_fixed_size_list().value_type(),
            match bits {
                1 => DataType::UInt64,
                2..=8 => DataType::UInt8,
                _ => DataType::UInt16,
            }
        );
        let same = scorer.kernel(&head, &head).unwrap();
        let later = scorer.kernel(&head, &tail).unwrap();
        let portable = [
            RQKernel::new(&scorer, &head, &head, false).unwrap(),
            RQKernel::new(&scorer, &head, &tail, false).unwrap(),
        ];
        for row in [0, 1, 2, 17, 31] {
            let mut actual = vec![0.0; 32];
            same.distances(row, 0..32, &mut actual);
            let mut rest = vec![0.0; rows - 32];
            later.distances(row, 0..rows - 32, &mut rest);
            let mut part = vec![0.0; 4];
            same.distances(row, 5..9, &mut part);
            assert_eq!(part, actual[5..9]);
            // The dispatched kernels match the portable ones bit for bit.
            let mut reference = vec![0.0; 32];
            portable[0].distances(row, 0..32, &mut reference);
            assert_eq!(to_bits(&reference), to_bits(&actual));
            let mut reference = vec![0.0; rows - 32];
            portable[1].distances(row, 0..rows - 32, &mut reference);
            assert_eq!(to_bits(&reference), to_bits(&rest));
            for (i, distance) in actual.into_iter().chain(rest).enumerate() {
                let (x, y) = (&exact[row], &exact[i]);
                let dot: f64 = x.iter().zip(y).map(|(a, b)| a * b).sum();
                let norm = |v: &[f64]| v.iter().map(|v| v * v).sum::<f64>().sqrt();
                let expected: f64 = match metric {
                    DistanceType::Dot => 1.0 - dot,
                    DistanceType::L2 => x.iter().zip(y).map(|(a, b)| (a - b).powi(2)).sum(),
                    _ => 1.0 - dot / (norm(x) * norm(y)),
                };
                assert!(
                    (distance as f64 - expected).abs() <= 1e-5 * expected.abs().max(1.0),
                    "bits={bits} packed={packed} {metric} row={row} candidate={i} actual={distance} expected={expected}"
                );
            }
        }
        if metric != DistanceType::Dot {
            // Identical codes and scales score exactly zero.
            let mut out = [1.0];
            same.distances(1, 2..3, &mut out);
            assert_eq!(out, [0.0]);
        }
    }

    fn to_bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    /// The dispatched code kernels agree bit for bit with the portable ones
    /// on candidate groups of every length and offset, dimension tails, and
    /// the extreme codes at the largest dimension the SIMD kernels accept.
    #[rstest]
    #[case::signs_tail(1, 648)]
    #[case::u8_tail(5, 200)]
    #[case::u8_bound(8, 33024)]
    #[case::u16_tail(9, 200)]
    #[case::u16_bound(9, 8224)]
    fn test_code_dots_match_portable(#[case] bits: u8, #[case] dim: usize) {
        let rows = 37;
        let max = (1u16 << bits) - 1;
        // Rows 0 and 1 hold the extreme codes, so their product is the most
        // negative centered value, `−dim·bias²/4`.
        let codes: Vec<u16> = (0..rows * dim)
            .map(|i| match i / dim {
                0 => 0,
                1 => max,
                r => ((r * 31 + (i % dim) * 17 + (i % dim) % 5 * 101) as u16) & max,
            })
            .collect();
        let sums: Vec<u64> = codes
            .chunks(dim)
            .map(|row| row.iter().map(|&c| u64::from(c)).sum())
            .collect();
        let bias = i64::from(max);
        let signs: Vec<u64> = codes
            .chunks(dim)
            .flat_map(|row| {
                let mut words = vec![0u64; dim.div_ceil(64)];
                for (d, &code) in row.iter().enumerate() {
                    words[d / 64] |= u64::from(code) << (d % 64);
                }
                words
            })
            .collect();
        let bytes: Vec<u8> = codes.iter().map(|&c| c as u8).collect();
        let dots = |simd: bool, anchor: usize, candidates: Range<usize>| {
            let mut out = vec![f32::NAN; candidates.len()];
            let sum = sums[anchor];
            let row_sums = &sums[candidates.clone()];
            match bits {
                1 => {
                    let words = dim.div_ceil(64);
                    select_sign_dots(simd)(
                        &signs[anchor * words..(anchor + 1) * words],
                        &signs[candidates.start * words..candidates.end * words],
                        dim,
                        &mut out,
                    );
                }
                2..=8 => select_u8_dots(simd)(
                    &bytes[anchor * dim..(anchor + 1) * dim],
                    &bytes[candidates.start * dim..candidates.end * dim],
                    sum,
                    row_sums,
                    bias,
                    &mut out,
                ),
                _ => select_u16_dots(simd)(
                    &codes[anchor * dim..(anchor + 1) * dim],
                    &codes[candidates.start * dim..candidates.end * dim],
                    sum,
                    row_sums,
                    bias,
                    &mut out,
                ),
            }
            out
        };
        if bits > 1 {
            assert!(centered_fits_i32(dim, bits));
            let extreme = dots(true, 0, 1..2)[0];
            assert_eq!(extreme, -((dim as i64 * bias * bias) as f32) * 0.25);
        }
        for anchor in [0, 1, 5] {
            for candidates in [0..rows, 1..rows, 3..19, 2..17, 3..10, 7..8, 4..4] {
                assert_eq!(
                    to_bits(&dots(true, anchor, candidates.clone())),
                    to_bits(&dots(false, anchor, candidates.clone())),
                    "bits={bits} dim={dim} anchor={anchor} candidates={candidates:?}"
                );
            }
        }
    }

    #[test]
    fn test_centered_fits_i32() {
        assert!(centered_fits_i32(33024, 8));
        assert!(!centered_fits_i32(33032, 8));
        assert!(centered_fits_i32(8224, 9));
        assert!(!centered_fits_i32(8232, 9));
    }
}
