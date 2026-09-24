// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Symmetric SQ code-to-code scoring with exact integer kernels.
//!
//! A code reconstructs `x̃ = lower + value_scale * code`, so squared L2 is the
//! integer `Σ(a − b)²` times `value_scale²`, and the dot product expands into
//! the integer `Σ a·b` plus per-row code sums staged once.
//!
//! On x86-64 with AVX-512 VNNI, one anchor row is scored against eight
//! candidate rows per pass, with squared L2 in its dot form
//! `Σa² + Σb² − 2Σa·b` over staged per-row code statistics. Every integer is
//! exact, so each distance is bit-identical to the portable per-pair kernels.

use std::{ops::Range, sync::Arc};

use arrow_array::{
    ArrayRef, Float64Array, RecordBatch, UInt64Array,
    cast::AsArray,
    types::{Float64Type, UInt8Type, UInt64Type},
};
use arrow_schema::{DataType, Field};
use lance_core::{Error, Result};
use lance_linalg::distance::{DistanceType, dot_u8::dot_u8_u64, l2_u8::l2_u8_u64};

use super::{
    ScalarQuantizer,
    storage::{sq_distance_scale, sq_value_scale},
};
use crate::vector::SQ_CODE_COLUMN;
use crate::vector::pairwise::{
    NORM_COLUMN, PairKernel, PairScorer, column_values, l2_to_cosine, list_values, norm_field,
};
use crate::vector::quantizer::Quantization;

/// Staged `Σ code` per row, for dot and for the batched kernels.
const CODE_SUM_COLUMN: &str = "__pairwise_sq_code_sum";
/// Staged `Σ code²` per row, for the batched l2 and cosine kernels.
const CODE_SQUARES_COLUMN: &str = "__pairwise_sq_code_squares";

/// Largest dimension whose code statistics, `Σ a·b` and squared L2 all fit
/// in an i32 (`dim · 255² < 2³¹`), as the batched SIMD kernels require.
const MAX_BATCHED_DIM: usize = 32768;

pub struct SQPairScorer {
    dim: usize,
    metric: DistanceType,
    lower_bound: f32,
    value_scale: f32,
    /// `value_scale²`, the factor from integer to reconstructed squared L2.
    distance_scale: f32,
    /// Whether this host runs the batched SIMD kernels, which need the staged
    /// code statistics. Staged batches, spilled or not, never leave the
    /// process that staged them, so staging and scoring agree on it.
    batched: bool,
}

impl SQPairScorer {
    pub(crate) fn new(sq: &ScalarQuantizer, metric: DistanceType) -> Result<Self> {
        if sq.num_bits() != 8 {
            return Err(Error::not_supported(format!(
                "SQ pair scoring requires 8-bit codes, got {}",
                sq.num_bits()
            )));
        }
        if !matches!(
            metric,
            DistanceType::L2 | DistanceType::Cosine | DistanceType::Dot
        ) {
            return Err(Error::not_supported(format!(
                "SQ pair scoring with {metric} distance"
            )));
        }
        let bounds = sq.bounds();
        Ok(Self {
            dim: sq.code_dim(),
            metric,
            lower_bound: bounds.start as f32,
            value_scale: sq_value_scale(&bounds),
            distance_scale: sq_distance_scale(&bounds),
            batched: batched_kernels_available(sq.code_dim()),
        })
    }
}

impl PairScorer for SQPairScorer {
    type Kernel<'a> = SQKernel<'a>;

    fn row_bytes(&self) -> usize {
        let is_dot = self.metric == DistanceType::Dot;
        let sums = if is_dot || self.batched { 8 } else { 0 };
        let squares = if !is_dot && self.batched { 8 } else { 0 };
        let norms = if self.metric == DistanceType::Cosine {
            8
        } else {
            0
        };
        self.dim + sums + squares + norms
    }

    fn stage(&self, source: &RecordBatch, rows: Range<usize>) -> Result<Vec<(Field, ArrayRef)>> {
        let (index, field) = source
            .schema()
            .column_with_name(SQ_CODE_COLUMN)
            .map(|(index, field)| (index, field.clone()))
            .ok_or_else(|| Error::internal(format!("SQ batch missing {SQ_CODE_COLUMN}")))?;
        let codes = source.column(index).slice(rows.start, rows.len());
        let values = codes
            .as_fixed_size_list_opt()
            .and_then(|codes| codes.values().as_primitive_opt::<UInt8Type>())
            .ok_or_else(|| Error::internal("SQ codes must be a fixed-size list of u8"))?
            .values();
        let is_dot = self.metric == DistanceType::Dot;
        let row_stat = |name: &str, stat: fn(u64) -> u64| -> (Field, ArrayRef) {
            let stats: Vec<u64> = values
                .chunks_exact(self.dim)
                .map(|codes| codes.iter().map(|&code| stat(u64::from(code))).sum())
                .collect();
            (
                Field::new(name, DataType::UInt64, false),
                Arc::new(UInt64Array::from(stats)),
            )
        };
        let mut columns = vec![(field, codes.clone())];
        if is_dot || self.batched {
            columns.push(row_stat(CODE_SUM_COLUMN, |code| code));
        }
        if !is_dot && self.batched {
            columns.push(row_stat(CODE_SQUARES_COLUMN, |code| code * code));
        }
        if self.metric == DistanceType::Cosine {
            let lower = f64::from(self.lower_bound);
            let scale = f64::from(self.value_scale);
            let squares: Vec<f64> = (0..=255u8)
                .map(|code| (lower + scale * f64::from(code)).powi(2))
                .collect();
            let norms: Float64Array = values
                .chunks_exact(self.dim)
                .map(|codes| {
                    codes
                        .iter()
                        .map(|&code| squares[usize::from(code)])
                        .sum::<f64>()
                        .sqrt()
                })
                .collect::<Vec<_>>()
                .into();
            columns.push((norm_field(), Arc::new(norms)));
        }
        Ok(columns)
    }

    fn kernel<'a>(
        &'a self,
        anchor: &'a RecordBatch,
        candidates: &'a RecordBatch,
    ) -> Result<Self::Kernel<'a>> {
        let is_dot = self.metric == DistanceType::Dot;
        let rows = |batch: &'a RecordBatch| -> Result<StagedRows<'a>> {
            let staged = StagedRows {
                codes: list_values::<UInt8Type>(batch, SQ_CODE_COLUMN)?,
                sums: if is_dot || self.batched {
                    column_values::<UInt64Type>(batch, CODE_SUM_COLUMN)?
                } else {
                    &[]
                },
                #[cfg(target_arch = "x86_64")]
                squares: if !is_dot && self.batched {
                    column_values::<UInt64Type>(batch, CODE_SQUARES_COLUMN)?
                } else {
                    &[]
                },
                norms: if self.metric == DistanceType::Cosine {
                    column_values::<Float64Type>(batch, NORM_COLUMN)?
                } else {
                    &[]
                },
            };
            if staged.codes.len() != batch.num_rows() * self.dim {
                return Err(Error::internal(format!(
                    "SQ pairwise batch has {} code bytes for {} rows of dim {}",
                    staged.codes.len(),
                    batch.num_rows(),
                    self.dim
                )));
            }
            Ok(staged)
        };
        Ok(SQKernel {
            scorer: self,
            anchor: rows(anchor)?,
            candidates: rows(candidates)?,
            batched: self.batched,
        })
    }
}

/// Whether this host runs the batched SIMD kernels at dimension `dim`.
fn batched_kernels_available(dim: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    let has_vnni = std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vnni");
    #[cfg(not(target_arch = "x86_64"))]
    let has_vnni = false;
    dim <= MAX_BATCHED_DIM && has_vnni
}

/// One staged batch's rows as raw slices.
#[derive(Clone, Copy)]
struct StagedRows<'a> {
    codes: &'a [u8],
    /// For dot or the batched kernels; empty otherwise.
    sums: &'a [u64],
    /// For the batched l2 and cosine kernels; empty otherwise.
    #[cfg(target_arch = "x86_64")]
    squares: &'a [u64],
    /// Cosine only.
    norms: &'a [f64],
}

/// The float terms of the reconstructed dot product
/// `dim·lower² + lower·value_scale·(Sa + Sb) + value_scale²·Σa·b`, in the
/// expression order of search's code-to-code SQ dot distance.
#[derive(Clone, Copy)]
struct DotTerms {
    constant: f32,
    lower_scale: f32,
    distance_scale: f32,
    anchor_sum: f32,
}

impl DotTerms {
    #[inline]
    fn distance(&self, dot: u64, candidate_sum: u64) -> f32 {
        let dot = self.constant
            + self.lower_scale * (candidate_sum as f32 + self.anchor_sum)
            + self.distance_scale * dot as f32;
        1.0 - dot
    }
}

pub struct SQKernel<'a> {
    scorer: &'a SQPairScorer,
    anchor: StagedRows<'a>,
    candidates: StagedRows<'a>,
    /// Whether the batched SIMD kernels apply, from the scorer; tests clear
    /// it to compare against the portable kernels.
    batched: bool,
}

impl PairKernel for SQKernel<'_> {
    fn distances(&self, anchor_row: usize, candidates: Range<usize>, out: &mut [f32]) {
        let scorer = self.scorer;
        let dim = scorer.dim;
        let a = &self.anchor.codes[anchor_row * dim..(anchor_row + 1) * dim];
        let codes = &self.candidates.codes[candidates.start * dim..candidates.end * dim];
        let out = &mut out[..candidates.len()];
        if scorer.metric == DistanceType::Dot {
            let sums = &self.candidates.sums[candidates];
            let lower = scorer.lower_bound;
            let terms = DotTerms {
                constant: dim as f32 * lower * lower,
                lower_scale: lower * scorer.value_scale,
                distance_scale: scorer.distance_scale,
                anchor_sum: self.anchor.sums[anchor_row] as f32,
            };
            #[cfg(target_arch = "x86_64")]
            if self.batched {
                // SAFETY: `batched` is set only when AVX-512 VNNI was detected
                // and `dim <= MAX_BATCHED_DIM`.
                unsafe { x86::dot_distances(a, codes, sums, &terms, out) };
                return;
            }
            for ((out, b), &b_sum) in out.iter_mut().zip(codes.chunks_exact(dim)).zip(sums) {
                *out = terms.distance(dot_u8_u64(b, a), b_sum);
            }
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if self.batched {
            // SAFETY: as for dot.
            unsafe {
                x86::l2_distances(
                    a,
                    self.anchor.squares[anchor_row],
                    codes,
                    &self.candidates.sums[candidates.clone()],
                    &self.candidates.squares[candidates.clone()],
                    scorer.distance_scale,
                    out,
                )
            };
        }
        if !self.batched {
            for (out, b) in out.iter_mut().zip(codes.chunks_exact(dim)) {
                *out = l2_u8_u64(b, a) as f32 * scorer.distance_scale;
            }
        }
        if scorer.metric == DistanceType::Cosine {
            l2_to_cosine(
                out,
                self.anchor.norms[anchor_row],
                &self.candidates.norms[candidates],
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    //! AVX-512 VNNI kernels scoring one anchor row against 8 candidate rows
    //! per pass: each 64-byte anchor block is loaded once for all 8 rows, and
    //! the 8 accumulators are reduced together.
    //!
    //! `VPDPBUSD` multiplies unsigned by signed bytes, so the anchor is
    //! biased into the signed domain (`a ⊕ 0x80 = a − 128`) and
    //! `D = Σ b·(a − 128) = Σ a·b − 128·Sb` is corrected with the staged
    //! candidate code sum `Sb`. For `dim <= MAX_BATCHED_DIM` every lane, `D`,
    //! `Σ a·b` and the squared L2 fit in an i32; intermediate sums wrap
    //! modulo 2³², which leaves the in-range final value exact.

    use std::arch::x86_64::*;

    use super::{DotTerms, MAX_BATCHED_DIM};

    const ROWS: usize = 8;

    /// Biased dots `D` of the anchor against `ROWS` consecutive candidate rows
    /// starting at `rows`, in row order.
    ///
    /// # Safety
    /// `anchor` points at `dim` bytes and `rows` at `ROWS * dim` bytes.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    #[inline]
    unsafe fn biased_dots(anchor: *const u8, rows: *const u8, dim: usize) -> __m256i {
        let flip = _mm512_set1_epi8(i8::MIN);
        let mut acc = [_mm512_setzero_si512(); ROWS];
        let full = dim / 64 * 64;
        let mut k = 0;
        while k < full {
            // SAFETY: `k + 64 <= dim` for the anchor and every row.
            unsafe {
                let a = _mm512_xor_si512(_mm512_loadu_si512(anchor.add(k).cast()), flip);
                for (r, acc) in acc.iter_mut().enumerate() {
                    let b = _mm512_loadu_si512(rows.add(r * dim + k).cast());
                    *acc = _mm512_dpbusd_epi32(*acc, b, a);
                }
            }
            k += 64;
        }
        if k < dim {
            // Masked-out candidate bytes load as zero, so they add nothing.
            let mask = u64::MAX >> (64 - (dim - k));
            // SAFETY: masked loads touch only the `dim - k` in-bounds bytes.
            unsafe {
                let a = _mm512_xor_si512(_mm512_maskz_loadu_epi8(mask, anchor.add(k).cast()), flip);
                for (r, acc) in acc.iter_mut().enumerate() {
                    let b = _mm512_maskz_loadu_epi8(mask, rows.add(r * dim + k).cast());
                    *acc = _mm512_dpbusd_epi32(*acc, b, a);
                }
            }
        }
        reduce(acc)
    }

    /// Horizontal sums of 8 accumulators as one vector, in accumulator order.
    #[target_feature(enable = "avx512f")]
    #[inline]
    fn reduce(acc: [__m512i; ROWS]) -> __m256i {
        // Per 128-bit lane: [x y x y] partial sums, then [w x y z].
        let pair = |x: __m512i, y: __m512i| {
            _mm512_add_epi32(_mm512_unpacklo_epi32(x, y), _mm512_unpackhi_epi32(x, y))
        };
        let quad = |x: __m512i, y: __m512i| {
            _mm512_add_epi32(_mm512_unpacklo_epi64(x, y), _mm512_unpackhi_epi64(x, y))
        };
        let low = quad(pair(acc[0], acc[1]), pair(acc[2], acc[3]));
        let high = quad(pair(acc[4], acc[5]), pair(acc[6], acc[7]));
        // 128-bit lanes [low0 + low1, low2 + low3, high0 + high1, high2 + high3].
        let halves = _mm512_add_epi32(
            _mm512_shuffle_i32x4::<0b10_00_10_00>(low, high),
            _mm512_shuffle_i32x4::<0b11_01_11_01>(low, high),
        );
        let sums = _mm512_add_epi32(
            _mm512_shuffle_i32x4::<0b00_00_10_00>(halves, halves),
            _mm512_shuffle_i32x4::<0b00_00_11_01>(halves, halves),
        );
        _mm512_castsi512_si256(sums)
    }

    /// The exact `Σ a·b` of the anchor and one candidate row, for fewer than
    /// `ROWS` candidates.
    ///
    /// # Safety
    /// Both pointers point at `dim` bytes.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    #[inline]
    unsafe fn dot(anchor: *const u8, row: *const u8, dim: usize, row_sum: u64) -> u64 {
        let flip = _mm512_set1_epi8(i8::MIN);
        let mut acc = _mm512_setzero_si512();
        let mut k = 0;
        while k < dim {
            let mask = u64::MAX >> (64 - (dim - k).min(64));
            // SAFETY: masked loads touch only in-bounds bytes.
            unsafe {
                let a = _mm512_xor_si512(_mm512_maskz_loadu_epi8(mask, anchor.add(k).cast()), flip);
                let b = _mm512_maskz_loadu_epi8(mask, row.add(k).cast());
                acc = _mm512_dpbusd_epi32(acc, b, a);
            }
            k += 64;
        }
        (i64::from(_mm512_reduce_add_epi32(acc)) + 128 * row_sum as i64) as u64
    }

    /// Low 32 bits of 8 staged u64 statistics.
    ///
    /// # Safety
    /// `values` points at 8 u64s.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn load_low32(values: *const u64) -> __m256i {
        // SAFETY: the caller guarantees 8 readable u64s.
        _mm512_cvtepi64_epi32(unsafe { _mm512_loadu_si512(values.cast()) })
    }

    /// First rows of the 8-row groups covering `num_rows >= ROWS` rows. A
    /// partial tail is covered by one more group ending at the last row: it
    /// rescores a few rows, whose exact values are rewritten unchanged.
    fn groups(num_rows: usize) -> impl Iterator<Item = usize> {
        let tail = (!num_rows.is_multiple_of(ROWS)).then(|| num_rows - ROWS);
        (0..num_rows / ROWS).map(|g| g * ROWS).chain(tail)
    }

    /// Checks the slice shapes every unchecked access relies on; returns `dim`.
    fn check_shapes(anchor: &[u8], codes: &[u8], sums: &[u64], out: &[f32]) -> usize {
        let dim = anchor.len();
        assert!(
            (1..=MAX_BATCHED_DIM).contains(&dim),
            "SQ batched kernel dim={dim} must be in 1..={MAX_BATCHED_DIM}"
        );
        assert_eq!(codes.len(), sums.len() * dim);
        assert_eq!(out.len(), sums.len());
        dim
    }

    /// Squared L2 times `scale` of the anchor against every candidate row.
    ///
    /// # Safety
    /// The CPU supports AVX-512F, AVX-512BW and AVX-512 VNNI.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    pub(super) unsafe fn l2_distances(
        anchor: &[u8],
        anchor_squares: u64,
        codes: &[u8],
        sums: &[u64],
        squares: &[u64],
        scale: f32,
        out: &mut [f32],
    ) {
        let dim = check_shapes(anchor, codes, sums, out);
        assert_eq!(squares.len(), sums.len());
        let num_rows = sums.len();
        if num_rows < ROWS {
            for (i, out) in out.iter_mut().enumerate() {
                // SAFETY: row `i` is in bounds.
                let row = unsafe { codes.as_ptr().add(i * dim) };
                let dot = unsafe { dot(anchor.as_ptr(), row, dim, sums[i]) };
                *out = (anchor_squares + squares[i] - 2 * dot) as f32 * scale;
            }
            return;
        }
        let anchor_squares = _mm256_set1_epi32(anchor_squares as i32);
        let scale = _mm256_set1_ps(scale);
        for i in groups(num_rows) {
            // SAFETY: rows `i..i + ROWS` are in bounds of every slice.
            unsafe {
                let dots = biased_dots(anchor.as_ptr(), codes.as_ptr().add(i * dim), dim);
                let sums = load_low32(sums.as_ptr().add(i));
                let squares = load_low32(squares.as_ptr().add(i));
                // Σa² + Σb² − 2(D + 128·Sb), wrapping.
                let l2 = _mm256_sub_epi32(
                    _mm256_sub_epi32(
                        _mm256_add_epi32(anchor_squares, squares),
                        _mm256_slli_epi32::<1>(dots),
                    ),
                    _mm256_slli_epi32::<8>(sums),
                );
                let distances = _mm256_mul_ps(_mm256_cvtepi32_ps(l2), scale);
                _mm256_storeu_ps(out.as_mut_ptr().add(i), distances);
            }
        }
    }

    /// Dot distances of the anchor against every candidate row.
    ///
    /// # Safety
    /// The CPU supports AVX-512F, AVX-512BW and AVX-512 VNNI.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    pub(super) unsafe fn dot_distances(
        anchor: &[u8],
        codes: &[u8],
        sums: &[u64],
        terms: &DotTerms,
        out: &mut [f32],
    ) {
        let dim = check_shapes(anchor, codes, sums, out);
        let num_rows = sums.len();
        if num_rows < ROWS {
            for (i, out) in out.iter_mut().enumerate() {
                // SAFETY: row `i` is in bounds.
                let row = unsafe { codes.as_ptr().add(i * dim) };
                let dot = unsafe { dot(anchor.as_ptr(), row, dim, sums[i]) };
                *out = terms.distance(dot, sums[i]);
            }
            return;
        }
        let one = _mm256_set1_ps(1.0);
        let constant = _mm256_set1_ps(terms.constant);
        let lower_scale = _mm256_set1_ps(terms.lower_scale);
        let distance_scale = _mm256_set1_ps(terms.distance_scale);
        let anchor_sum = _mm256_set1_ps(terms.anchor_sum);
        for i in groups(num_rows) {
            // SAFETY: rows `i..i + ROWS` are in bounds of every slice.
            unsafe {
                let dots = biased_dots(anchor.as_ptr(), codes.as_ptr().add(i * dim), dim);
                let sums = load_low32(sums.as_ptr().add(i));
                let dot = _mm256_add_epi32(dots, _mm256_slli_epi32::<7>(sums));
                // Separate multiplies and adds in `DotTerms::distance` order.
                let sum_term = _mm256_mul_ps(
                    lower_scale,
                    _mm256_add_ps(_mm256_cvtepi32_ps(sums), anchor_sum),
                );
                let code_term = _mm256_mul_ps(distance_scale, _mm256_cvtepi32_ps(dot));
                let dot = _mm256_add_ps(_mm256_add_ps(constant, sum_term), code_term);
                _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_sub_ps(one, dot));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{FixedSizeListArray, UInt8Array};
    use arrow_schema::Schema;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::ROW_ID;
    use rstest::rstest;

    fn stage(scorer: &SQPairScorer, codes: &[u8], rows: Range<usize>) -> RecordBatch {
        let dim = scorer.dim;
        let num_rows = codes.len() / dim;
        let source = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new(ROW_ID, DataType::UInt64, false),
                Field::new(
                    SQ_CODE_COLUMN,
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::UInt8, true)),
                        dim as i32,
                    ),
                    false,
                ),
            ])),
            vec![
                Arc::new(UInt64Array::from_iter_values(0..num_rows as u64)),
                Arc::new(
                    FixedSizeListArray::try_new_from_values(
                        UInt8Array::from(codes.to_vec()),
                        dim as i32,
                    )
                    .unwrap(),
                ),
            ],
        )
        .unwrap();
        RecordBatch::try_from_iter(
            scorer
                .stage(&source, rows)
                .unwrap()
                .into_iter()
                .map(|(field, column)| (field.name().clone(), column)),
        )
        .unwrap()
    }

    /// Distances from the batched kernel (where the CPU supports it) and from
    /// the portable per-pair kernel, which must agree bit for bit.
    fn distances(kernel: &SQKernel, anchor: usize, candidates: Range<usize>) -> Vec<f32> {
        let mut out = vec![f32::NAN; candidates.len()];
        kernel.distances(anchor, candidates.clone(), &mut out);
        let portable = SQKernel {
            batched: false,
            ..*kernel
        };
        let mut expected = vec![f32::NAN; candidates.len()];
        portable.distances(anchor, candidates, &mut expected);
        assert_eq!(
            out.iter().map(|d| d.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|d| d.to_bits()).collect::<Vec<_>>(),
            "batched and portable kernels disagree"
        );
        out
    }

    // Dimensions cover a tail-only row, whole 64-byte blocks, and blocks plus
    // a tail; 20 rows cover full 8-row groups plus a partial one.
    #[rstest]
    fn test_sq_distances(
        #[values(DistanceType::L2, DistanceType::Cosine, DistanceType::Dot)] metric: DistanceType,
        #[values(19, 64, 131)] dim: usize,
    ) {
        let rows = 20;
        let sq = ScalarQuantizer::with_bounds(8, dim, -1.0..1.5);
        let scorer = SQPairScorer::new(&sq, metric).unwrap();
        let mut codes: Vec<u8> = (0..rows * dim)
            .map(|i| ((i * 37 + 11) % 256) as u8)
            .collect();
        // Row 1 repeats row 0.
        codes.copy_within(0..dim, dim);
        let staged = stage(&scorer, &codes, 0..rows);
        // Codes plus one 8-byte statistic or norm per extra column.
        assert_eq!(scorer.row_bytes(), dim + 8 * (staged.num_columns() - 1));
        // Stage a slice so offsets into the source are exercised.
        let tail = stage(&scorer, &codes, 4..rows);
        let reconstruct = |r: usize| -> Vec<f64> {
            codes[r * dim..(r + 1) * dim]
                .iter()
                .map(|&c| {
                    f64::from(scorer.lower_bound) + f64::from(scorer.value_scale) * f64::from(c)
                })
                .collect()
        };
        let kernel = scorer.kernel(&staged, &staged).unwrap();
        let tail_kernel = scorer.kernel(&staged, &tail).unwrap();
        for a in 0..rows {
            let out = distances(&kernel, a, 0..rows);
            // Other candidate ranges shift the 8-row groups or fall below one
            // group; values must not change.
            assert_eq!(distances(&tail_kernel, a, 0..rows - 4), out[4..]);
            assert_eq!(distances(&kernel, a, 3..rows), out[3..]);
            assert_eq!(distances(&kernel, a, 13..rows), out[13..]);
            for (b, &distance) in out.iter().enumerate() {
                let (x, y) = (reconstruct(a), reconstruct(b));
                let dot: f64 = x.iter().zip(&y).map(|(x, y)| x * y).sum();
                let norm = |v: &[f64]| v.iter().map(|v| v * v).sum::<f64>().sqrt();
                let expected = match metric {
                    DistanceType::L2 => x.iter().zip(&y).map(|(x, y)| (x - y).powi(2)).sum(),
                    DistanceType::Dot => 1.0 - dot,
                    _ => 1.0 - dot / (norm(&x) * norm(&y)),
                };
                // f32 roundoff scales with the per-dimension terms.
                let tolerance = 1e-5 * expected.abs().max(dim as f64 * 0.25);
                assert!(
                    (f64::from(distance) - expected).abs() <= tolerance,
                    "{metric} dim={dim} a={a} b={b} actual={distance} expected={expected}"
                );
            }
            if metric != DistanceType::Dot {
                assert_eq!(out[a], 0.0);
            }
        }
        if metric != DistanceType::Dot {
            assert_eq!(distances(&kernel, 0, 1..2), [0.0]);
        }
    }

    /// Extreme codes at the largest batched dimension reach the i32 limits of
    /// the batched kernels, which must still match the u64 portable kernels.
    #[rstest]
    fn test_sq_batched_limits(
        #[values(DistanceType::L2, DistanceType::Cosine, DistanceType::Dot)] metric: DistanceType,
    ) {
        let dim = MAX_BATCHED_DIM;
        let rows = 11;
        let sq = ScalarQuantizer::with_bounds(8, dim, 0.0..1.0);
        let scorer = SQPairScorer::new(&sq, metric).unwrap();
        let codes: Vec<u8> = (0..rows)
            .flat_map(|r| {
                (0..dim).map(move |i| match r % 4 {
                    0 => 0,
                    1 => 255,
                    2 => (i * 131 + r) as u8,
                    _ => 255 - (i % 2) as u8,
                })
            })
            .collect();
        let staged = stage(&scorer, &codes, 0..rows);
        let kernel = scorer.kernel(&staged, &staged).unwrap();
        for a in 0..rows {
            let out = distances(&kernel, a, 0..rows);
            if metric == DistanceType::L2 {
                // All-zero vs all-255 codes: the largest squared L2.
                let expected = (dim as u64 * 255 * 255) as f32 * scorer.distance_scale;
                match a % 4 {
                    0 => assert_eq!(out[1], expected),
                    1 => assert_eq!(out[0], expected),
                    _ => {}
                }
            }
        }
        assert!(!batched_kernels_available(MAX_BATCHED_DIM + 1));
    }
}
