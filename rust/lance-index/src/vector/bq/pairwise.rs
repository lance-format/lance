// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Symmetric RQ code-to-code batch scoring. Bit planes retain the quantized
//! representation; no floating-point vector is materialized. Scales stay scalar.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, UInt64Array,
    cast::AsArray,
    types::{Float32Type, UInt8Type, UInt64Type},
};
use arrow_schema::{DataType, Field};
use lance_arrow::{FixedSizeListArrayExt, RecordBatchExt};
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;

use super::{
    builder::RabitQuantizer,
    ex_dot::{blocked_ex_code_bytes, unpack_blocked_row},
    storage::{RABIT_BLOCKED_EX_CODE_COLUMN, RABIT_CODE_COLUMN, RabitQueryEstimator, unpack_codes},
    transform::{EX_SCALE_FACTORS_COLUMN, SCALE_FACTORS_COLUMN},
};

const PLANES: &str = "__pairwise_rq_planes";
const SUM: &str = "__pairwise_rq_sum";
const NORM: &str = "__pairwise_rq_norm";
const CENTER_DOT: &str = "__pairwise_rq_center_dot";

/// Partition-local RQ code scoring, with a shared centroid for dot distance.
pub struct RQCodeDistance {
    dim: usize,
    bits: u8,
    packed: bool,
    metric: DistanceType,
    centroid: Vec<f32>,
    centroid_norm: f32,
}

impl RQCodeDistance {
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
        let centroid = if metric == DistanceType::Dot {
            let dim = centroid.len();
            rq.rotate_fsl_to_f32(&FixedSizeListArray::try_new_from_values(
                centroid, dim as i32,
            )?)?
        } else {
            Vec::new()
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

    /// Repack existing sign/extended codes into bit planes once during staging.
    /// The last word is zero-padded; sums and norms exclude those padding bits.
    pub(crate) fn prepare(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let codes = batch
            .column_by_name(RABIT_CODE_COLUMN)
            .ok_or_else(|| Error::invalid_input("RQ pair batch missing sign codes"))?
            .as_fixed_size_list();
        let signs = if self.packed {
            unpack_codes(codes)
        } else {
            codes.clone()
        };
        let signs = signs.values().as_primitive::<UInt8Type>().values();
        let ex_bits = self.bits - 1;
        let extended = if ex_bits == 0 {
            None
        } else {
            Some(
                batch
                    .column_by_name(RABIT_BLOCKED_EX_CODE_COLUMN)
                    .ok_or_else(|| Error::invalid_input("RQ pair batch missing extended codes"))?
                    .as_fixed_size_list(),
            )
        };
        let words = self.dim.div_ceil(64);
        let width = words * self.bits as usize;
        let mut planes = vec![0u64; batch.num_rows() * width];
        let mut sums = Vec::with_capacity(batch.num_rows());
        let mut norms = Vec::with_capacity(batch.num_rows());
        let mut center_dots = Vec::with_capacity(batch.num_rows());
        let bias = (1i64 << self.bits) - 1;
        for row in 0..batch.num_rows() {
            let ex = if let Some(extended) = extended {
                let data = extended.value(row);
                let data = data.as_primitive::<UInt8Type>().values();
                if data.len() != blocked_ex_code_bytes(self.dim, ex_bits) {
                    return Err(Error::invalid_input("invalid RQ extended code width"));
                }
                unpack_blocked_row(data, ex_bits, self.dim)
            } else {
                Vec::new()
            };
            let mut sum = 0u64;
            let mut norm = 0i64;
            let mut center_dot = 0.0;
            for d in 0..self.dim {
                let sign = (signs[row * (self.dim / 8) + d / 8] >> (d % 8)) & 1;
                let code = ((sign as u16) << ex_bits) + ex.get(d).copied().unwrap_or(0) as u16;
                for bit in 0..self.bits as usize {
                    planes[row * width + bit * words + d / 64] |=
                        ((u64::from(code) >> bit) & 1) << (d % 64);
                }
                let centered = 2 * i64::from(code) - bias;
                sum += u64::from(code);
                norm += centered * centered;
                if self.metric == DistanceType::Dot {
                    center_dot += self.centroid[d] * centered as f32 * 0.5;
                }
            }
            sums.push(sum);
            norms.push(norm as f32 * 0.25);
            center_dots.push(center_dot);
        }
        let planes =
            FixedSizeListArray::try_new_from_values(UInt64Array::from(planes), width as i32)?;
        let batch = batch.drop_column(RABIT_CODE_COLUMN)?;
        let batch = if extended.is_some() {
            batch.drop_column(RABIT_BLOCKED_EX_CODE_COLUMN)?
        } else {
            batch
        };
        Ok(batch
            .try_with_column(
                Field::new(PLANES, planes.data_type().clone(), false),
                Arc::new(planes),
            )?
            .try_with_column(
                Field::new(SUM, DataType::UInt64, false),
                Arc::new(UInt64Array::from(sums)),
            )?
            .try_with_column(
                Field::new(NORM, DataType::Float32, false),
                Arc::new(Float32Array::from(norms)),
            )?
            .try_with_column(
                Field::new(CENTER_DOT, DataType::Float32, false),
                Arc::new(Float32Array::from(center_dots)),
            )?)
    }

    pub(crate) fn distance_batch(
        &self,
        anchor: &RecordBatch,
        row: usize,
        candidates: &RecordBatch,
    ) -> Result<Vec<f32>> {
        let planes = |batch: &RecordBatch| -> Result<FixedSizeListArray> {
            Ok(batch
                .column_by_name(PLANES)
                .ok_or_else(|| Error::internal("RQ pair batch missing bit planes"))?
                .as_fixed_size_list()
                .clone())
        };
        let scalar = |batch: &RecordBatch, name: &str| -> Result<Float32Array> {
            Ok(batch
                .column_by_name(name)
                .ok_or_else(|| Error::internal(format!("RQ pair batch missing {name}")))?
                .as_primitive::<Float32Type>()
                .clone())
        };
        let sums = |batch: &RecordBatch| -> Result<UInt64Array> {
            Ok(batch
                .column_by_name(SUM)
                .ok_or_else(|| Error::internal("RQ pair batch missing code sums"))?
                .as_primitive::<UInt64Type>()
                .clone())
        };
        let a_planes = planes(anchor)?.value(row);
        let a_planes = a_planes.as_primitive::<UInt64Type>().values();
        let b_planes = planes(candidates)?;
        let b_planes = b_planes.values().as_primitive::<UInt64Type>().values();
        let scale_column = if self.bits == 1 {
            SCALE_FACTORS_COLUMN
        } else {
            EX_SCALE_FACTORS_COLUMN
        };
        let divisor = if self.metric == DistanceType::Dot {
            -1.0
        } else {
            -2.0
        };
        let a_scale = scalar(anchor, scale_column)?.value(row) / divisor;
        let b_scale = scalar(candidates, scale_column)?;
        let a_norm = scalar(anchor, NORM)?.value(row);
        let b_norm = scalar(candidates, NORM)?;
        let a_center = scalar(anchor, CENTER_DOT)?.value(row);
        let b_center = scalar(candidates, CENTER_DOT)?;
        let a_sum = sums(anchor)?.value(row);
        let b_sum = sums(candidates)?;
        let words = self.dim.div_ceil(64);
        let mut dots = vec![0.0; candidates.num_rows()];
        code_dot_batch(
            a_planes,
            b_planes,
            words,
            self.bits,
            self.dim,
            a_sum,
            b_sum.values(),
            &mut dots,
        );
        let mut distances = Vec::with_capacity(candidates.num_rows());
        for (i, dot) in dots.into_iter().enumerate() {
            let scale = b_scale.value(i) / divisor;
            let distance = if self.metric == DistanceType::Dot {
                1.0 - (self.centroid_norm
                    + a_scale * a_center
                    + scale * b_center.value(i)
                    + a_scale * scale * dot)
            } else {
                // SDC uses code norms and the RQ scale factors. Negative roundoff
                // at zero is not a negative squared distance.
                let squared = a_scale * a_scale * a_norm + scale * scale * b_norm.value(i)
                    - 2.0 * a_scale * scale * dot;
                if squared < 0.0 { 0.0 } else { squared }
            };
            distances.push(distance);
        }
        Ok(distances)
    }
}

// Dispatch once per vector batch, not once per candidate or bit plane.
#[allow(clippy::too_many_arguments)]
fn code_dot_batch(
    a: &[u64],
    b: &[u64],
    words: usize,
    bits: u8,
    dim: usize,
    a_sum: u64,
    b_sum: &[u64],
    output: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512vpopcntdq")
    {
        // SAFETY: Both target features were checked. The implementation uses
        // bounds-checked slices, including the partial final code word.
        return unsafe { code_dot_batch_avx512(a, b, words, bits, dim, a_sum, b_sum, output) };
    }
    code_dot_batch_scalar(a, b, words, bits, dim, a_sum, b_sum, output);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
#[allow(clippy::too_many_arguments)]
unsafe fn code_dot_batch_avx512(
    a: &[u64],
    b: &[u64],
    words: usize,
    bits: u8,
    dim: usize,
    a_sum: u64,
    b_sum: &[u64],
    output: &mut [f32],
) {
    code_dot_batch_scalar(a, b, words, bits, dim, a_sum, b_sum, output);
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn code_dot_batch_scalar(
    a: &[u64],
    b: &[u64],
    words: usize,
    bits: u8,
    dim: usize,
    a_sum: u64,
    b_sum: &[u64],
    output: &mut [f32],
) {
    let width = words * bits as usize;
    let bias = (1i64 << bits) - 1;
    for (i, (b, out)) in b.chunks_exact(width).zip(output).enumerate() {
        *out = if bits == 1 {
            let differences: u32 = a.iter().zip(b).map(|(a, b)| (a ^ b).count_ones()).sum();
            (dim as f32 - 2.0 * differences as f32) * 0.25
        } else {
            let mut product = 0i64;
            for (a_bit, a) in a.chunks_exact(words).enumerate() {
                for (b_bit, b) in b.chunks_exact(words).enumerate() {
                    let intersections: u32 =
                        a.iter().zip(b).map(|(a, b)| (a & b).count_ones()).sum();
                    product += i64::from(intersections) << (a_bit + b_bit);
                }
            }
            let centered =
                4 * product - 2 * bias * (a_sum + b_sum[i]) as i64 + dim as i64 * bias * bias;
            centered as f32 * 0.25
        };
    }
}

#[cfg(test)]
mod tests {
    use super::super::{ex_dot::pack_blocked_row, storage::pack_codes};
    use super::*;
    use arrow_array::UInt8Array;
    use arrow_schema::Schema;
    use lance_core::ROW_ID;
    use rstest::rstest;

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
        #[values(DistanceType::L2, DistanceType::Dot)] metric: DistanceType,
    ) {
        // 72 dimensions exercise padding in bit planes and extended codes;
        // 35 rows exercise both a full FastScan group and its partial tail.
        let dim = 72;
        let rows = 35;
        let centroid: Vec<f32> = (0..dim).map(|d| (d % 7) as f32 * 0.125).collect();
        let scorer = RQCodeDistance {
            dim,
            bits,
            packed,
            metric,
            centroid_norm: centroid.iter().map(|v| v * v).sum(),
            centroid: centroid.clone(),
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
            .map(|r| {
                if r == 0 {
                    0.0
                } else {
                    (r % 4 + 1) as f32 * 0.0625
                }
            })
            .collect();
        for (row, &scale) in scales.iter().enumerate() {
            let codes: Vec<u16> = (0..dim)
                .map(|d| ((row * 17 + d * 13) as u16) & mask)
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
        let divisor = if metric == DistanceType::Dot {
            -1.0
        } else {
            -2.0
        };
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
        let batch = scorer
            .prepare(RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap())
            .unwrap();
        assert!(batch.column_by_name(RABIT_CODE_COLUMN).is_none());
        let planes = batch[PLANES].as_fixed_size_list();
        assert_eq!(planes.value_type(), DataType::UInt64);
        assert_eq!(
            planes.value_length() as usize * 8,
            dim.div_ceil(64) * 8 * bits as usize
        );
        let candidates = batch.slice(1, rows - 2);
        for row in [0, 1, 17, 34] {
            let a = planes.value(row);
            let b = candidates[PLANES].as_fixed_size_list().values();
            let sums = candidates[SUM].as_primitive::<UInt64Type>();
            let mut scalar = vec![0.0; candidates.num_rows()];
            let mut dispatched = scalar.clone();
            code_dot_batch_scalar(
                a.as_primitive::<UInt64Type>().values(),
                b.as_primitive::<UInt64Type>().values(),
                dim.div_ceil(64),
                bits,
                dim,
                batch[SUM].as_primitive::<UInt64Type>().value(row),
                sums.values(),
                &mut scalar,
            );
            code_dot_batch(
                a.as_primitive::<UInt64Type>().values(),
                b.as_primitive::<UInt64Type>().values(),
                dim.div_ceil(64),
                bits,
                dim,
                batch[SUM].as_primitive::<UInt64Type>().value(row),
                sums.values(),
                &mut dispatched,
            );
            assert_eq!(scalar, dispatched);
            let actual = scorer.distance_batch(&batch, row, &candidates).unwrap();
            assert_eq!(actual.len(), rows - 2);
            for (i, &distance) in actual.iter().enumerate() {
                let expected: f64 = if metric == DistanceType::Dot {
                    1.0 - exact[row]
                        .iter()
                        .zip(&exact[i + 1])
                        .map(|(a, b)| a * b)
                        .sum::<f64>()
                } else {
                    exact[row]
                        .iter()
                        .zip(&exact[i + 1])
                        .map(|(a, b)| (a - b).powi(2))
                        .sum()
                };
                assert!(
                    (distance as f64 - expected).abs() <= 1e-5 * expected.abs().max(1.0),
                    "bits={bits} packed={packed} row={row} candidate={} actual={distance} expected={expected}",
                    i + 1
                );
            }
        }
    }
}
