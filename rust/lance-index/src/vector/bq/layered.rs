// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Fixed prefix layouts for opt-in layered RaBitQ indices.

use lance_core::{Error, Result};

/// Plane widths and build-time scale-search policy. The scale is not persisted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RQLayout {
    pub high_bits: u8,
    pub low_bits: u8,
    pub search_bits: u8,
}

impl RQLayout {
    /// Resolve the only supported layered layouts: 1+2+2, 1+4+2 and 1+4+4.
    pub fn try_new(num_bits: u8) -> Result<Self> {
        let (high_bits, low_bits, search_bits) = match num_bits {
            5 => (2, 2, 3),
            7 => (4, 2, 5),
            9 => (4, 4, 6),
            _ => {
                return Err(Error::invalid_input(format!(
                    "IVF_RQ layered requires num_bits=5, 7 or 9, got {num_bits}"
                )));
            }
        };
        Ok(Self {
            high_bits,
            low_bits,
            search_bits,
        })
    }
}

use super::ex_dot::{blocked_ex_code_bytes, pack_blocked_row};
use super::storage::{RABIT_BLOCKED_EX_CODE_COLUMN, RABIT_BLOCKED_EX_CODE_LO_COLUMN};
use super::transform::{EX_ADD_FACTORS_FIELD, EX_SCALE_FACTORS_FIELD};
use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, UInt8Array};
use arrow_schema::{DataType, Field};
use lance_arrow::FixedSizeListArrayExt;
use std::sync::Arc;

pub const HIGH_ADD_FACTORS_COLUMN: &str = "__add_factors_ex_hi";
pub const HIGH_SCALE_FACTORS_COLUMN: &str = "__scale_factors_ex_hi";
pub const HIGH_BOUNDS_COLUMN: &str = "__rq_bounds_hi";
pub const FULL_BOUNDS_COLUMN: &str = "__rq_bounds_full";

// Each level stores ||w_level-w_sign||, |add_level-add_sign| and a
// coefficient for floating-point/LUT error. These bound the stored estimator,
// rather than its error relative to the unquantized vector.
fn bounds_field(name: &str) -> Field {
    Field::new(
        name,
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 3),
        true,
    )
}

pub(crate) fn estimator_bounds(
    residuals: &[f32],
    codes: &[u8],
    dim: usize,
    bits: u8,
    binary: &super::transform::RabitRawQueryFactors,
    level: &super::transform::RabitRawQueryFactors,
) -> Result<ArrayRef> {
    let adds = level
        .ex_add_factors
        .as_ref()
        .ok_or_else(|| Error::internal("missing level add factors"))?;
    let scales = level
        .ex_scale_factors
        .as_ref()
        .ok_or_else(|| Error::internal("missing level scale factors"))?;
    let code_scale = (1u32 << bits) as f64;
    let bias = -(code_scale - 0.5);
    let mut bounds = Vec::with_capacity(residuals.len() / dim * 3);
    for (row, (residual, codes)) in residuals
        .chunks_exact(dim)
        .zip(codes.chunks_exact(dim))
        .enumerate()
    {
        let binary_scale = binary.scale_factors.value(row) as f64;
        let scale = scales.value(row) as f64;
        let norm = residual
            .iter()
            .zip(codes)
            .map(|(&r, &c)| {
                let sign = f64::from(u8::from(r.is_sign_positive()));
                let delta =
                    (code_scale * sign + f64::from(c) + bias) * scale - (sign - 0.5) * binary_scale;
                delta * delta
            })
            .sum::<f64>()
            .sqrt();
        for value in [
            norm,
            (f64::from(adds.value(row)) - f64::from(binary.add_factors.value(row))).abs(),
            binary_scale.abs() + code_scale * scale.abs(),
        ] {
            let rounded = value as f32;
            bounds.push(if f64::from(rounded) < value {
                rounded.next_up()
            } else {
                rounded
            });
        }
    }
    Ok(Arc::new(FixedSizeListArray::try_new_from_values(
        Float32Array::from(bounds),
        3,
    )?))
}

pub(crate) fn storage_fields(
    dim: usize,
    num_bits: u8,
    mut fields: Vec<Field>,
) -> Result<Vec<Field>> {
    let layout = RQLayout::try_new(num_bits)?;
    for (name, bits) in [
        (RABIT_BLOCKED_EX_CODE_COLUMN, layout.high_bits),
        (RABIT_BLOCKED_EX_CODE_LO_COLUMN, layout.low_bits),
    ] {
        fields.push(Field::new(
            name,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt8, true)),
                blocked_ex_code_bytes(dim, bits) as i32,
            ),
            true,
        ));
    }
    fields.extend([
        EX_ADD_FACTORS_FIELD.clone(),
        EX_SCALE_FACTORS_FIELD.clone(),
        Field::new(HIGH_ADD_FACTORS_COLUMN, DataType::Float32, true),
        Field::new(HIGH_SCALE_FACTORS_COLUMN, DataType::Float32, true),
        bounds_field(HIGH_BOUNDS_COLUMN),
        bounds_field(FULL_BOUNDS_COLUMN),
    ]);
    Ok(fields)
}

/// Split already-quantized full codes. No second quantization or scale search.
pub(crate) fn split_codes(
    values: &[u8],
    dim: usize,
    layout: RQLayout,
) -> Result<(ArrayRef, ArrayRef, Vec<u8>)> {
    let rows = values.len() / dim;
    let hi_bytes = blocked_ex_code_bytes(dim, layout.high_bits);
    let lo_bytes = blocked_ex_code_bytes(dim, layout.low_bits);
    let mut hi = vec![0; rows * hi_bytes];
    let mut lo = vec![0; rows * lo_bytes];
    let hi_values: Vec<u8> = values.iter().map(|v| v >> layout.low_bits).collect();
    let mask = ((1u16 << layout.low_bits) - 1) as u8;
    let mut low_values = vec![0; dim];
    for row in 0..rows {
        for (dst, src) in low_values
            .iter_mut()
            .zip(&values[row * dim..(row + 1) * dim])
        {
            *dst = src & mask;
        }
        pack_blocked_row(
            &hi_values[row * dim..(row + 1) * dim],
            layout.high_bits,
            &mut hi[row * hi_bytes..(row + 1) * hi_bytes],
        );
        pack_blocked_row(
            &low_values,
            layout.low_bits,
            &mut lo[row * lo_bytes..(row + 1) * lo_bytes],
        );
    }
    Ok((
        Arc::new(FixedSizeListArray::try_new_from_values(
            UInt8Array::from(hi),
            hi_bytes as i32,
        )?),
        Arc::new(FixedSizeListArray::try_new_from_values(
            UInt8Array::from(lo),
            lo_bytes as i32,
        )?),
        hi_values,
    ))
}

/// Query precision on a layered IVF_RQ index. Other layouts accept only `Full`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RQPrecision {
    /// Scan sign codes with the binary estimator.
    Sign,
    /// Scan sign and high codes with the high-level factor pair.
    High,
    /// Evaluate all stored bits.
    #[default]
    Full,
}

impl std::str::FromStr for RQPrecision {
    type Err = Error;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "sign" => Ok(Self::Sign),
            "high" => Ok(Self::High),
            "full" => Ok(Self::Full),
            _ => Err(Error::invalid_input(format!(
                "rq_precision must be sign, high or full, got {value}"
            ))),
        }
    }
}

use arrow_array::RecordBatch;
use lance_core::cache::{CacheCodec, CacheKey};
use lance_core::deepsize::{Context, DeepSizeOf};
use std::borrow::Cow;

/// A plane and its factors share one persistent cache entry.
#[derive(Debug)]
pub struct PlaneBatch(pub RecordBatch);
impl DeepSizeOf for PlaneBatch {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.0.deep_size_of_children(context)
    }
}
/// Plane keys live under the immutable index UUID namespace.
pub struct PlaneKey {
    pub partition: usize,
    pub plane: u8,
}
impl CacheKey for PlaneKey {
    type ValueType = PlaneBatch;
    fn key(&self) -> Cow<'_, str> {
        format!("{}:{}", self.partition, self.plane).into()
    }
    fn type_name() -> &'static str {
        "RQPlane"
    }
    fn codec_for_key(&self) -> Option<CacheCodec> {
        Self::codec().map(|codec| codec.with_memory_priority(3 - self.plane))
    }
    fn codec() -> Option<CacheCodec> {
        Some(CacheCodec::from_impl::<PlaneBatch>())
    }
}

/// Column projection for sign (0), high (1), or low (2), including factors.
pub fn plane_columns(plane: u8) -> &'static [&'static str] {
    use super::storage::RABIT_CODE_COLUMN;
    use super::transform::{
        ADD_FACTORS_COLUMN, ERROR_FACTORS_COLUMN, EX_ADD_FACTORS_COLUMN, EX_SCALE_FACTORS_COLUMN,
        SCALE_FACTORS_COLUMN,
    };
    match plane {
        0 => &[
            lance_core::ROW_ID,
            RABIT_CODE_COLUMN,
            ADD_FACTORS_COLUMN,
            SCALE_FACTORS_COLUMN,
            ERROR_FACTORS_COLUMN,
            HIGH_BOUNDS_COLUMN,
            FULL_BOUNDS_COLUMN,
        ],
        1 => &[
            RABIT_BLOCKED_EX_CODE_COLUMN,
            HIGH_ADD_FACTORS_COLUMN,
            HIGH_SCALE_FACTORS_COLUMN,
        ],
        2 => &[
            RABIT_BLOCKED_EX_CODE_LO_COLUMN,
            EX_ADD_FACTORS_COLUMN,
            EX_SCALE_FACTORS_COLUMN,
        ],
        _ => &[],
    }
}

/// A quantized candidate tied to an immutable index and its physical partition.
///
/// Distributed executors may perform a global distance cut on these values, then
/// return the survivors to the same index for full-code reranking with the same query.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QuantizedCandidate {
    pub index_uuid: uuid::Uuid,
    pub partition_id: usize,
    pub row_offset: u32,
    pub row_id: u64,
    pub distance: f32,
    pub centroid_distance: f32,
    /// Original-partition binary LUT value, including its SIMD rounding policy.
    pub binary_inner_product: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::bq::{
        RQBuildParams, builder::RabitQuantizer, storage::RabitQuantizationStorage,
        transform::RQTransformer,
    };
    use crate::vector::quantizer::{Quantization, QuantizerStorage};
    use crate::vector::storage::{DistCalculator, DistanceCalculatorOptions, VectorStore};
    use crate::vector::transform::Transformer;
    use crate::vector::{CENTROID_DIST_COLUMN, PART_ID_COLUMN};
    use arrow_array::types::Float32Type;
    use arrow_array::{Float32Array, UInt32Array, UInt64Array, cast::AsArray};
    use lance_arrow::RecordBatchExt;
    use lance_linalg::distance::DistanceType;
    use rstest::rstest;

    #[rstest]
    #[case::rq5_l2(5, DistanceType::L2)]
    #[case::rq7_l2(7, DistanceType::L2)]
    #[case::rq9_l2(9, DistanceType::L2)]
    #[case::rq5_dot(5, DistanceType::Dot)]
    #[case::rq7_dot(7, DistanceType::Dot)]
    #[case::rq9_dot(9, DistanceType::Dot)]
    fn layered_roundtrip_and_levels(
        #[case] bits: u8,
        #[case] distance_type: DistanceType,
        #[values(64, 72)] dim: usize,
    ) {
        const ROWS: usize = 37;
        let values: Vec<f32> = (0..dim * ROWS)
            .map(|i| ((i * 17 % 101) as f32 - 50.) / 50.)
            .collect();
        let vectors =
            FixedSizeListArray::try_new_from_values(Float32Array::from(values.clone()), dim as i32)
                .unwrap();
        let rq = RabitQuantizer::build(
            &vectors,
            distance_type,
            &RQBuildParams::new(bits).with_layered(true),
        )
        .unwrap();
        let norms = Float32Array::from(
            values
                .chunks(dim)
                .map(|r| r.iter().map(|v| v * v).sum::<f32>())
                .collect::<Vec<_>>(),
        );
        let input = RecordBatch::try_from_iter(vec![
            ("vector", Arc::new(vectors.clone()) as ArrayRef),
            (
                lance_core::ROW_ID,
                Arc::new(UInt64Array::from((0..ROWS as u64).collect::<Vec<_>>())) as ArrayRef,
            ),
            (
                PART_ID_COLUMN,
                Arc::new(UInt32Array::from(vec![0; ROWS])) as ArrayRef,
            ),
            (CENTROID_DIST_COLUMN, Arc::new(norms) as ArrayRef),
        ])
        .unwrap();
        let centroids =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.; dim]), dim as i32)
                .unwrap();
        let batch = RQTransformer::new(rq.clone(), distance_type, centroids, "vector")
            .unwrap()
            .transform(&input)
            .unwrap();
        let full = RabitQuantizationStorage::try_from_batch(
            batch.clone(),
            rq.metadata_ref(),
            distance_type,
            None,
        )
        .unwrap();
        let codes = rq.quantize_split(&vectors).unwrap().ex_codes.unwrap();
        let single_batch = batch
            .drop_column(RABIT_BLOCKED_EX_CODE_LO_COLUMN)
            .unwrap()
            .drop_column(HIGH_ADD_FACTORS_COLUMN)
            .unwrap()
            .drop_column(HIGH_SCALE_FACTORS_COLUMN)
            .unwrap()
            .drop_column(RABIT_BLOCKED_EX_CODE_COLUMN)
            .unwrap()
            .try_with_column(
                super::super::storage::rabit_ex_code_field(dim, bits)
                    .unwrap()
                    .unwrap(),
                codes,
            )
            .unwrap();
        let mut metadata = rq.metadata_ref().clone();
        metadata.layered = false;
        let single =
            RabitQuantizationStorage::try_from_batch(single_batch, &metadata, distance_type, None)
                .unwrap();
        let query: ArrayRef = Arc::new(Float32Array::from(values[..dim].to_vec()));
        assert_eq!(
            full.dist_calculator(query.clone(), 1.).distance_all(ROWS),
            single.dist_calculator(query.clone(), 1.).distance_all(ROWS)
        );
        for precision in [RQPrecision::Sign, RQPrecision::High, RQPrecision::Full] {
            let projected = RabitQuantizationStorage::try_from_batch_at_precision(
                batch.clone(),
                rq.metadata_ref(),
                distance_type,
                None,
                precision,
            )
            .unwrap();
            let mut scratch = Vec::new();
            let calc = full.dist_calculator_with_scratch(
                query.clone(),
                1.,
                None,
                &mut scratch,
                DistanceCalculatorOptions {
                    approx_mode: Default::default(),
                    rq_precision: precision,
                },
            );
            assert_eq!(
                calc.distance_all(ROWS),
                projected
                    .dist_calculator(query.clone(), 1.)
                    .distance_all(ROWS)
            );
            if precision != RQPrecision::Sign {
                let mut audit_scratch = Vec::new();
                let audit = full.dist_calculator_with_scratch(
                    query.clone(),
                    1.,
                    None,
                    &mut audit_scratch,
                    DistanceCalculatorOptions {
                        approx_mode: crate::vector::ApproxMode::Accurate,
                        rq_precision: precision,
                    },
                );
                let binary = audit.binary_inner_products();
                let distances = audit.distance_all(ROWS);
                for (row, &ip) in binary.iter().enumerate() {
                    let bound = audit.raw_query_lower_bound(row, ip).unwrap();
                    assert!(
                        bound <= distances[row],
                        "invalid bound at row {row}: {bound} > {}",
                        distances[row]
                    );
                }
                let best = distances.iter().copied().min_by(f32::total_cmp).unwrap();
                assert!(
                    binary
                        .iter()
                        .enumerate()
                        .any(|(row, &ip)| audit.raw_query_lower_bound(row, ip).unwrap() > best),
                    "fixture must exercise pruning"
                );
            }
            // A precision-specific bound must preserve the unpruned result,
            // including non-block-aligned code dimensions and heap/range cuts.
            for (lower, upper) in [(None, None), (Some(-0.5), Some(20.0))] {
                let distances = calc.distance_all(ROWS);
                let mut expected: Vec<_> = distances
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(_, d)| lower.is_none_or(|v| *d >= v) && upper.is_none_or(|v| *d < v))
                    .map(|(id, d)| (id as u64, d))
                    .collect();
                expected.sort_by(|a, b| a.1.total_cmp(&b.1));
                expected.truncate(3);
                let mut heap = std::collections::BinaryHeap::new();
                calc.accumulate_topk_with_scratch(
                    3,
                    lower,
                    upper,
                    u64::from,
                    &mut heap,
                    &mut Vec::new(),
                    &mut Vec::new(),
                    &mut Vec::new(),
                    &mut Vec::new(),
                );
                let actual: Vec<_> = heap
                    .into_sorted_vec()
                    .into_iter()
                    .map(|n| (n.id, n.dist.0))
                    .collect();
                assert_eq!(
                    actual, expected,
                    "bits={bits} precision={precision:?} dim={dim}"
                );
            }
        }
        let missing = batch.drop_column(RABIT_BLOCKED_EX_CODE_LO_COLUMN).unwrap();
        assert!(
            RabitQuantizationStorage::try_from_batch(
                missing,
                rq.metadata_ref(),
                distance_type,
                None
            )
            .unwrap_err()
            .to_string()
            .contains("missing column")
        );
        assert_eq!(
            batch[HIGH_ADD_FACTORS_COLUMN]
                .as_primitive::<Float32Type>()
                .len(),
            ROWS
        );
    }

    #[rstest]
    #[case(1)]
    #[case(2)]
    #[case(3)]
    #[case(4)]
    #[case(6)]
    #[case(8)]
    fn rejects_unsupported_layout(#[case] bits: u8) {
        assert!(
            RQLayout::try_new(bits)
                .unwrap_err()
                .to_string()
                .contains("requires num_bits=5, 7 or 9")
        );
    }
}
