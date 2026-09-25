// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Symmetric SQ code-to-code scoring with exact integer kernels.
//!
//! A code reconstructs `x̃ = lower + value_scale * code`, so squared L2 is the
//! integer `Σ(a − b)²` times `value_scale²`, and the dot product expands into
//! the integer `Σ a·b` plus per-row code sums staged once.

use std::{ops::Range, sync::Arc};

use arrow_array::{
    ArrayRef, Float32Array, Float64Array, RecordBatch,
    cast::AsArray,
    types::{Float32Type, Float64Type, UInt8Type},
};
use arrow_schema::{DataType, Field};
use lance_core::{Error, Result};
use lance_linalg::distance::{DistanceType, dot_u8::dot_u8_u64, l2_u8::l2_u8_u64};

use super::{
    ScalarQuantizer,
    storage::{sq_code_sum, sq_distance_scale, sq_value_scale},
};
use crate::vector::SQ_CODE_COLUMN;
use crate::vector::pairwise::{
    NORM_COLUMN, PairKernel, PairScorer, column_values, l2_to_cosine, list_values, norm_field,
};
use crate::vector::quantizer::Quantization;

const CODE_SUM_COLUMN: &str = "__pairwise_sq_code_sum";

pub struct SQPairScorer {
    dim: usize,
    metric: DistanceType,
    lower_bound: f32,
    value_scale: f32,
    /// `value_scale²`, the factor from integer to reconstructed squared L2.
    distance_scale: f32,
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
        })
    }
}

impl PairScorer for SQPairScorer {
    type Kernel<'a> = SQKernel<'a>;

    fn row_bytes(&self) -> usize {
        self.dim
            + match self.metric {
                DistanceType::Dot => 4,
                DistanceType::Cosine => 8,
                _ => 0,
            }
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
        let mut columns = vec![(field, codes.clone())];
        match self.metric {
            DistanceType::Dot => {
                let sums: Float32Array = values
                    .chunks_exact(self.dim)
                    .map(sq_code_sum)
                    .collect::<Vec<_>>()
                    .into();
                columns.push((
                    Field::new(CODE_SUM_COLUMN, DataType::Float32, false),
                    Arc::new(sums),
                ));
            }
            DistanceType::Cosine => {
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
            _ => {}
        }
        Ok(columns)
    }

    fn kernel<'a>(
        &'a self,
        anchor: &'a RecordBatch,
        candidates: &'a RecordBatch,
    ) -> Result<Self::Kernel<'a>> {
        let sums = |batch: &'a RecordBatch| -> Result<&'a [f32]> {
            if self.metric == DistanceType::Dot {
                column_values::<Float32Type>(batch, CODE_SUM_COLUMN)
            } else {
                Ok(&[])
            }
        };
        let norms = |batch: &'a RecordBatch| -> Result<&'a [f64]> {
            if self.metric == DistanceType::Cosine {
                column_values::<Float64Type>(batch, NORM_COLUMN)
            } else {
                Ok(&[])
            }
        };
        Ok(SQKernel {
            scorer: self,
            anchor: list_values::<UInt8Type>(anchor, SQ_CODE_COLUMN)?,
            candidates: list_values::<UInt8Type>(candidates, SQ_CODE_COLUMN)?,
            anchor_sums: sums(anchor)?,
            candidate_sums: sums(candidates)?,
            anchor_norms: norms(anchor)?,
            candidate_norms: norms(candidates)?,
        })
    }
}

pub struct SQKernel<'a> {
    scorer: &'a SQPairScorer,
    anchor: &'a [u8],
    candidates: &'a [u8],
    anchor_sums: &'a [f32],
    candidate_sums: &'a [f32],
    anchor_norms: &'a [f64],
    candidate_norms: &'a [f64],
}

impl PairKernel for SQKernel<'_> {
    fn distances(&self, anchor_row: usize, candidates: Range<usize>, out: &mut [f32]) {
        let scorer = self.scorer;
        let dim = scorer.dim;
        let a = &self.anchor[anchor_row * dim..(anchor_row + 1) * dim];
        let codes = self.candidates[candidates.start * dim..candidates.end * dim].chunks_exact(dim);
        if scorer.metric == DistanceType::Dot {
            // The same expansion as search's code-to-code SQ dot distance.
            let lower = scorer.lower_bound;
            let constant = dim as f32 * lower * lower;
            let a_sum = self.anchor_sums[anchor_row];
            for ((out, b), &b_sum) in out
                .iter_mut()
                .zip(codes)
                .zip(&self.candidate_sums[candidates])
            {
                let dot = constant
                    + lower * scorer.value_scale * (b_sum + a_sum)
                    + scorer.distance_scale * dot_u8_u64(b, a) as f32;
                *out = 1.0 - dot;
            }
            return;
        }
        for (out, b) in out.iter_mut().zip(codes) {
            *out = l2_u8_u64(b, a) as f32 * scorer.distance_scale;
        }
        if scorer.metric == DistanceType::Cosine {
            l2_to_cosine(
                out,
                self.anchor_norms[anchor_row],
                &self.candidate_norms[candidates],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{FixedSizeListArray, UInt8Array, UInt64Array};
    use arrow_schema::Schema;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::ROW_ID;
    use rstest::rstest;

    #[rstest]
    fn test_sq_distances(
        #[values(DistanceType::L2, DistanceType::Cosine, DistanceType::Dot)] metric: DistanceType,
    ) {
        let dim = 19;
        let rows = 9;
        let sq = ScalarQuantizer::with_bounds(8, dim, -1.0..1.5);
        let scorer = SQPairScorer::new(&sq, metric).unwrap();
        let mut codes: Vec<u8> = (0..rows * dim)
            .map(|i| ((i * 37 + 11) % 256) as u8)
            .collect();
        // Row 1 repeats row 0.
        codes.copy_within(0..dim, dim);
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
                Arc::new(UInt64Array::from_iter_values(0..rows as u64)),
                Arc::new(
                    FixedSizeListArray::try_new_from_values(
                        UInt8Array::from(codes.clone()),
                        dim as i32,
                    )
                    .unwrap(),
                ),
            ],
        )
        .unwrap();
        // Stage a slice so offsets into the source are exercised.
        let staged = RecordBatch::try_from_iter(
            scorer
                .stage(&source, 0..rows)
                .unwrap()
                .into_iter()
                .map(|(field, column)| (field.name().clone(), column)),
        )
        .unwrap();
        let tail = RecordBatch::try_from_iter(
            scorer
                .stage(&source, 4..rows)
                .unwrap()
                .into_iter()
                .map(|(field, column)| (field.name().clone(), column)),
        )
        .unwrap();
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
            let mut out = vec![0.0; rows];
            kernel.distances(a, 0..rows, &mut out);
            let mut later = vec![0.0; rows - 4];
            tail_kernel.distances(a, 0..rows - 4, &mut later);
            assert_eq!(later, out[4..]);
            for (b, &distance) in out.iter().enumerate() {
                let (x, y) = (reconstruct(a), reconstruct(b));
                let dot: f64 = x.iter().zip(&y).map(|(x, y)| x * y).sum();
                let norm = |v: &[f64]| v.iter().map(|v| v * v).sum::<f64>().sqrt();
                let expected = match metric {
                    DistanceType::L2 => x.iter().zip(&y).map(|(x, y)| (x - y).powi(2)).sum(),
                    DistanceType::Dot => 1.0 - dot,
                    _ => 1.0 - dot / (norm(&x) * norm(&y)),
                };
                assert!(
                    (f64::from(distance) - expected).abs() <= 1e-5 * expected.abs().max(1.0),
                    "{metric} a={a} b={b} actual={distance} expected={expected}"
                );
            }
            if metric != DistanceType::Dot {
                assert_eq!(out[a], 0.0);
            }
        }
        if metric != DistanceType::Dot {
            let mut out = [1.0];
            kernel.distances(0, 1..2, &mut out);
            assert_eq!(out, [0.0]);
        }
    }
}
