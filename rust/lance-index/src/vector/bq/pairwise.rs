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

use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, Float64Array, RecordBatch, UInt8Array, UInt16Array,
    UInt64Array,
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
                UInt64Array::from(sign_words),
                words as i32,
            )?),
            2..=8 => Arc::new(FixedSizeListArray::try_new_from_values(
                UInt8Array::from(codes_u8),
                dim as i32,
            )?),
            _ => Arc::new(FixedSizeListArray::try_new_from_values(
                UInt16Array::from(codes_u16),
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
        let codes = match self.bits {
            1 => RQCodes::Signs {
                anchor: list_values::<UInt64Type>(anchor, CODES)?,
                candidates: list_values::<UInt64Type>(candidates, CODES)?,
            },
            2..=8 => RQCodes::U8 {
                anchor: list_values::<UInt8Type>(anchor, CODES)?,
                candidates: list_values::<UInt8Type>(candidates, CODES)?,
            },
            _ => RQCodes::U16 {
                anchor: list_values::<UInt16Type>(anchor, CODES)?,
                candidates: list_values::<UInt16Type>(candidates, CODES)?,
            },
        };
        Ok(RQKernel {
            scorer: self,
            codes,
            anchor: RQRows::try_new(self, anchor)?,
            candidates: RQRows::try_new(self, candidates)?,
        })
    }
}

enum RQCodes<'a> {
    Signs {
        anchor: &'a [u64],
        candidates: &'a [u64],
    },
    U8 {
        anchor: &'a [u8],
        candidates: &'a [u8],
    },
    U16 {
        anchor: &'a [u16],
        candidates: &'a [u16],
    },
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

impl RQKernel<'_> {
    /// `Σ ōa·ōb` per candidate. Integer products are exact, so the result
    /// does not depend on the evaluation order.
    fn centered_dots(&self, anchor_row: usize, candidates: Range<usize>, out: &mut [f32]) {
        let dim = self.scorer.dim;
        let bias = (1i64 << self.scorer.bits) - 1;
        // With 2ō = 2·code − bias: Σ(2ōa)(2ōb) = 4Σab − 2·bias·(Σa + Σb) + dim·bias².
        let centered = |product: u64, sums: u64| {
            let centered = 4 * product as i64 - 2 * bias * sums as i64 + dim as i64 * bias * bias;
            centered as f32 * 0.25
        };
        let b_sums = &self.candidates.sums;
        match &self.codes {
            RQCodes::Signs {
                anchor,
                candidates: values,
            } => {
                let words = self.scorer.words();
                let x = &anchor[anchor_row * words..(anchor_row + 1) * words];
                let y = &values[candidates.start * words..candidates.end * words];
                for (out, y) in out.iter_mut().zip(y.chunks_exact(words)) {
                    let differences: u32 = x.iter().zip(y).map(|(x, y)| (x ^ y).count_ones()).sum();
                    *out = (dim as f32 - 2.0 * differences as f32) * 0.25;
                }
            }
            RQCodes::U8 {
                anchor,
                candidates: values,
            } => {
                let x = &anchor[anchor_row * dim..(anchor_row + 1) * dim];
                let y = &values[candidates.start * dim..candidates.end * dim];
                let a_sum = self.anchor.sums[anchor_row];
                for ((out, y), &b_sum) in out
                    .iter_mut()
                    .zip(y.chunks_exact(dim))
                    .zip(&b_sums[candidates])
                {
                    *out = centered(dot_u8_u64(x, y), a_sum + b_sum);
                }
            }
            RQCodes::U16 {
                anchor,
                candidates: values,
            } => {
                let x = &anchor[anchor_row * dim..(anchor_row + 1) * dim];
                let y = &values[candidates.start * dim..candidates.end * dim];
                let a_sum = self.anchor.sums[anchor_row];
                for ((out, y), &b_sum) in out
                    .iter_mut()
                    .zip(y.chunks_exact(dim))
                    .zip(&b_sums[candidates])
                {
                    let product = x
                        .iter()
                        .zip(y)
                        .map(|(&x, &y)| u64::from(x) * u64::from(y))
                        .sum();
                    *out = centered(product, a_sum + b_sum);
                }
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
    use arrow_array::Array;
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
        for row in [0, 1, 2, 17, 31] {
            let mut actual = vec![0.0; 32];
            same.distances(row, 0..32, &mut actual);
            let mut rest = vec![0.0; rows - 32];
            later.distances(row, 0..rows - 32, &mut rest);
            let mut part = vec![0.0; 4];
            same.distances(row, 5..9, &mut part);
            assert_eq!(part, actual[5..9]);
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
}
