// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fmt::{Debug, Formatter};
use std::sync::{Arc, LazyLock};

use arrow::array::AsArray;
use arrow::datatypes::{Float16Type, Float32Type, Float64Type, UInt32Type};
#[cfg(test)]
use arrow_array::UInt32Array;
use arrow_array::{Array, ArrowNativeTypeOp, FixedSizeListArray, Float32Array, RecordBatch};
use arrow_schema::DataType;
use lance_arrow::RecordBatchExt;
use lance_core::{Error, Result};
use lance_linalg::distance::{DistanceType, norm_squared_fsl};
use tracing::instrument;

use crate::vector::bq::builder::RabitQuantizer;
use crate::vector::bq::rabit_ex_bits;
use crate::vector::bq::storage::{
    RABIT_BLOCKED_EX_CODE_COLUMN, RABIT_CODE_COLUMN, RabitQueryEstimator,
};
use crate::vector::quantizer::Quantization;
use crate::vector::transform::Transformer;
use crate::vector::{CENTROID_DIST_COLUMN, PART_ID_COLUMN};

// the inner product of quantized vector and the residual vector.
pub const ADD_FACTORS_COLUMN: &str = "__add_factors";
// the inner product of quantized vector and the centroid vector.
pub const SCALE_FACTORS_COLUMN: &str = "__scale_factors";
pub const EX_ADD_FACTORS_COLUMN: &str = "__add_factors_ex";
pub const EX_SCALE_FACTORS_COLUMN: &str = "__scale_factors_ex";
pub const ERROR_FACTORS_COLUMN: &str = "__error_factors";

const RABIT_ERROR_EPSILON: f32 = 1.9;

pub static ADD_FACTORS_FIELD: LazyLock<arrow_schema::Field> = LazyLock::new(|| {
    arrow_schema::Field::new(ADD_FACTORS_COLUMN, arrow_schema::DataType::Float32, true)
});
pub static SCALE_FACTORS_FIELD: LazyLock<arrow_schema::Field> = LazyLock::new(|| {
    arrow_schema::Field::new(SCALE_FACTORS_COLUMN, arrow_schema::DataType::Float32, true)
});
pub static EX_ADD_FACTORS_FIELD: LazyLock<arrow_schema::Field> = LazyLock::new(|| {
    arrow_schema::Field::new(EX_ADD_FACTORS_COLUMN, arrow_schema::DataType::Float32, true)
});
pub static EX_SCALE_FACTORS_FIELD: LazyLock<arrow_schema::Field> = LazyLock::new(|| {
    arrow_schema::Field::new(
        EX_SCALE_FACTORS_COLUMN,
        arrow_schema::DataType::Float32,
        true,
    )
});
pub static ERROR_FACTORS_FIELD: LazyLock<arrow_schema::Field> = LazyLock::new(|| {
    arrow_schema::Field::new(ERROR_FACTORS_COLUMN, arrow_schema::DataType::Float32, true)
});

pub struct RQTransformer {
    rq: RabitQuantizer,
    distance_type: DistanceType,
    centroids_norm_square: Option<Float32Array>,
    rotated_centroids: Option<Vec<f32>>,
    vector_column: String,
}

impl RQTransformer {
    fn transform_raw_query(
        &self,
        batch: &RecordBatch,
        residual_vectors: &FixedSizeListArray,
        res_norm_square: &Float32Array,
    ) -> Result<RecordBatch> {
        let part_ids = batch
            .column_by_name(PART_ID_COLUMN)
            .ok_or_else(|| Error::index("RQ Transform: partition ids not found"))?
            .as_primitive::<UInt32Type>();
        let rotated_centroids = self.rotated_centroids.as_ref().ok_or_else(|| {
            Error::internal("RabitQ raw-query transformer is missing rotated centroids")
        })?;
        let dim = self.rq.dim();
        let num_centroids = rotated_centroids.len() / dim;
        for &part_id in part_ids.values() {
            if part_id as usize >= num_centroids {
                return Err(Error::invalid_input(format!(
                    "RQ Transform: partition id {part_id} out of range for {num_centroids} rotated centroids"
                )));
            }
        }
        let ex_bits = rabit_ex_bits(self.rq.num_bits())?;
        let (codes, factors) = self.rq.quantize_with_factors(
            residual_vectors,
            |row, residual, ex_values, ex_dot| {
                let start = part_ids.value(row) as usize * dim;
                compute_row_factors(
                    self.distance_type,
                    res_norm_square.value(row),
                    residual,
                    &rotated_centroids[start..start + dim],
                    ex_values,
                    ex_dot,
                    ex_bits,
                )
            },
        )?;
        let mut add = Vec::with_capacity(factors.len());
        let mut scale = Vec::with_capacity(factors.len());
        let mut error = Vec::with_capacity(factors.len());
        let mut ex_add = Vec::new();
        let mut ex_scale = Vec::new();
        if ex_bits != 0 {
            ex_add.reserve(factors.len());
            ex_scale.reserve(factors.len());
        }
        for factor in factors {
            add.push(factor.add);
            scale.push(factor.scale);
            error.push(factor.error);
            if ex_bits != 0 {
                ex_add.push(factor.ex_add);
                ex_scale.push(factor.ex_scale);
            }
        }
        let mut batch = batch
            .try_with_column(self.rq.field(), codes.binary_codes)?
            .try_with_column(ADD_FACTORS_FIELD.clone(), Arc::new(Float32Array::from(add)))?
            .try_with_column(
                SCALE_FACTORS_FIELD.clone(),
                Arc::new(Float32Array::from(scale)),
            )?
            .try_with_column(
                ERROR_FACTORS_FIELD.clone(),
                Arc::new(Float32Array::from(error)),
            )?;
        if let Some(ex_codes) = codes.ex_codes {
            let field = crate::vector::bq::storage::rabit_ex_code_field(dim, self.rq.num_bits())?
                .ok_or_else(|| Error::internal("Missing RabitQ extra-code field"))?;
            batch = batch
                .try_with_column(field, ex_codes)?
                .try_with_column(
                    EX_ADD_FACTORS_FIELD.clone(),
                    Arc::new(Float32Array::from(ex_add)),
                )?
                .try_with_column(
                    EX_SCALE_FACTORS_FIELD.clone(),
                    Arc::new(Float32Array::from(ex_scale)),
                )?;
        }
        Ok(batch
            .drop_column(&self.vector_column)?
            .drop_column(CENTROID_DIST_COLUMN)?)
    }

    pub fn new(
        rq: RabitQuantizer,
        distance_type: DistanceType,
        centroids: FixedSizeListArray,
        vector_column: impl Into<String>,
    ) -> Result<Self> {
        // for dot product, the add factor is `1 - v*c + |c|^2`, so we need to compute |c|^2
        let centroids_norm_square = (distance_type == DistanceType::Dot)
            .then(|| Float32Array::from(norm_squared_fsl(&centroids)));
        let rotated_centroids = (rq.metadata_ref().query_estimator
            == RabitQueryEstimator::RawQuery)
            .then(|| rq.rotate_fsl_to_f32(&centroids))
            .transpose()?;

        Ok(Self {
            rq,
            distance_type,
            centroids_norm_square,
            rotated_centroids,
            vector_column: vector_column.into(),
        })
    }
}

#[cfg(test)]
struct RabitRawQueryFactors {
    add_factors: Float32Array,
    scale_factors: Float32Array,
    error_factors: Float32Array,
    ex_add_factors: Option<Float32Array>,
    ex_scale_factors: Option<Float32Array>,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct RabitRowFactors {
    add: f32,
    scale: f32,
    error: f32,
    ex_add: f32,
    ex_scale: f32,
}

fn compute_row_factors(
    distance_type: DistanceType,
    norm_square: f32,
    residual: &[f32],
    centroid: &[f32],
    ex_values: Option<&[u8]>,
    ex_res_dot: f32,
    ex_bits: u8,
) -> RabitRowFactors {
    let ex_code_bias = -((1u32 << ex_bits) as f32 - 0.5);
    let mut binary_res_dot = 0.0f32;
    let mut binary_cent_dot = 0.0f32;
    let mut ex_cent_dot = 0.0f32;
    let mut residual_centroid_dot = 0.0f32;
    for (dim_idx, (&residual_value, &centroid_value)) in residual.iter().zip(centroid).enumerate() {
        let binary_code = u32::from(residual_value.is_sign_positive());
        let binary_factor = binary_factor_value(residual_value);
        binary_res_dot += residual_value * binary_factor;
        binary_cent_dot += centroid_value * binary_factor;
        if let Some(ex_values) = ex_values {
            let ex_factor =
                ((binary_code << ex_bits) + ex_values[dim_idx] as u32) as f32 + ex_code_bias;
            ex_cent_dot += centroid_value * ex_factor;
        }
        residual_centroid_dot += residual_value * centroid_value;
    }
    let binary_correction = factor_ratio(norm_square * binary_cent_dot, binary_res_dot);
    let ex_correction = factor_ratio(norm_square * ex_cent_dot, ex_res_dot);
    let error = error_factor_value(distance_type, norm_square, binary_res_dot, residual.len());
    match distance_type {
        DistanceType::L2 => RabitRowFactors {
            add: norm_square + 2.0 * binary_correction,
            scale: factor_ratio(-2.0 * norm_square, binary_res_dot),
            error,
            ex_add: norm_square + 2.0 * ex_correction,
            ex_scale: factor_ratio(-2.0 * norm_square, ex_res_dot),
        },
        DistanceType::Dot => {
            let dot_base = 1.0 - residual_centroid_dot;
            RabitRowFactors {
                add: dot_base + binary_correction,
                scale: factor_ratio(-norm_square, binary_res_dot),
                error,
                ex_add: dot_base + ex_correction,
                ex_scale: factor_ratio(-norm_square, ex_res_dot),
            }
        }
        _ => unreachable!(),
    }
}

#[inline]
fn factor_ratio(numerator: f32, denominator: f32) -> f32 {
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

#[inline]
fn binary_factor_value(rotated_residual: f32) -> f32 {
    if rotated_residual.is_sign_positive() {
        0.5
    } else {
        -0.5
    }
}

#[inline]
fn error_factor_value(
    distance_type: DistanceType,
    norm_square: f32,
    binary_res_dot: f32,
    code_dim: usize,
) -> f32 {
    if code_dim <= 1 || norm_square <= 0.0 || binary_res_dot == 0.0 {
        return 0.0;
    }

    let code_norm_square = code_dim as f32 * 0.25;
    let alignment = norm_square * code_norm_square / binary_res_dot.powi(2);
    let angular_error = ((alignment - 1.0).max(0.0) / (code_dim as f32 - 1.0)).sqrt();
    let error = norm_square.sqrt() * RABIT_ERROR_EPSILON * angular_error;
    match distance_type {
        DistanceType::L2 => 2.0 * error,
        DistanceType::Dot => error,
        _ => unreachable!(),
    }
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn compute_raw_query_factors(
    distance_type: DistanceType,
    res_norm_square: &Float32Array,
    rotated_residuals: &[f32],
    rotated_centroids: &[f32],
    part_ids: &UInt32Array,
    ex_code_values: Option<&[u8]>,
    ex_res_dot_dists: Option<&[f32]>,
    ex_bits: u8,
    code_dim: usize,
) -> Result<RabitRawQueryFactors> {
    if !matches!(distance_type, DistanceType::L2 | DistanceType::Dot) {
        return Err(Error::index(format!(
            "RQ Transform: distance type {} not supported",
            distance_type
        )));
    }

    let num_rows = res_norm_square.len();
    debug_assert_eq!(rotated_residuals.len(), num_rows * code_dim);
    if let Some(ex_code_values) = ex_code_values {
        debug_assert_eq!(ex_code_values.len(), num_rows * code_dim);
    }
    if let Some(ex_res_dot_dists) = ex_res_dot_dists {
        debug_assert_eq!(ex_res_dot_dists.len(), num_rows);
    }

    let has_ex_codes = ex_bits != 0;
    let ex_code_bias = -((1u32 << ex_bits) as f32 - 0.5);
    let mut add_factors = Vec::with_capacity(num_rows);
    let mut scale_factors = Vec::with_capacity(num_rows);
    let mut error_factors = Vec::with_capacity(num_rows);
    let mut ex_add_factors = has_ex_codes.then(|| Vec::with_capacity(num_rows));
    let mut ex_scale_factors = has_ex_codes.then(|| Vec::with_capacity(num_rows));

    for (row_idx, &norm_square) in res_norm_square.values().iter().enumerate() {
        let part_id = part_ids.value(row_idx) as usize;
        let centroid_start = part_id.checked_mul(code_dim).ok_or_else(|| {
            Error::invalid_input(format!(
                "RQ Transform: partition id {} overflows code_dim {}",
                part_id, code_dim
            ))
        })?;
        let centroid_end = centroid_start.checked_add(code_dim).ok_or_else(|| {
            Error::invalid_input(format!(
                "RQ Transform: partition id {} plus code_dim {} overflows",
                part_id, code_dim
            ))
        })?;
        if centroid_end > rotated_centroids.len() {
            return Err(Error::invalid_input(format!(
                "RQ Transform: partition id {} out of range for {} rotated centroids",
                part_id,
                rotated_centroids.len() / code_dim
            )));
        }

        let row_start = row_idx * code_dim;
        let row_end = row_start + code_dim;
        let residual = &rotated_residuals[row_start..row_end];
        let centroid = &rotated_centroids[centroid_start..centroid_end];
        let ex_values = ex_code_values.map(|values| &values[row_start..row_end]);

        let mut binary_res_dot = 0.0f32;
        let mut binary_cent_dot = 0.0f32;
        let mut ex_cent_dot = 0.0f32;
        let mut residual_centroid_dot = 0.0f32;
        for (dim_idx, (&residual_value, &centroid_value)) in
            residual.iter().zip(centroid.iter()).enumerate()
        {
            let residual_value: f32 = residual_value;
            let centroid_value: f32 = centroid_value;
            let binary_code = if residual_value.is_sign_positive() {
                1u32
            } else {
                0u32
            };
            let binary_factor = binary_factor_value(residual_value);

            binary_res_dot += residual_value * binary_factor;
            binary_cent_dot += centroid_value * binary_factor;
            if let Some(ex_values) = ex_values {
                let ex_code_value = ex_values[dim_idx];
                let ex_factor =
                    ((binary_code << ex_bits) + ex_code_value as u32) as f32 + ex_code_bias;
                ex_cent_dot += centroid_value * ex_factor;
            }
            residual_centroid_dot += residual_value * centroid_value;
        }

        let binary_correction = factor_ratio(norm_square * binary_cent_dot, binary_res_dot);
        let ex_res_dot = ex_res_dot_dists
            .map(|values| values[row_idx])
            .unwrap_or_default();
        let ex_correction = factor_ratio(norm_square * ex_cent_dot, ex_res_dot);
        error_factors.push(error_factor_value(
            distance_type,
            norm_square,
            binary_res_dot,
            code_dim,
        ));

        match distance_type {
            DistanceType::L2 => {
                add_factors.push(norm_square + 2.0 * binary_correction);
                scale_factors.push(factor_ratio(-2.0 * norm_square, binary_res_dot));
                if let Some(ex_add_factors) = ex_add_factors.as_mut() {
                    ex_add_factors.push(norm_square + 2.0 * ex_correction);
                }
                if let Some(ex_scale_factors) = ex_scale_factors.as_mut() {
                    ex_scale_factors.push(factor_ratio(-2.0 * norm_square, ex_res_dot));
                }
            }
            DistanceType::Dot => {
                let dot_base = 1.0 - residual_centroid_dot;
                add_factors.push(dot_base + binary_correction);
                scale_factors.push(factor_ratio(-norm_square, binary_res_dot));
                if let Some(ex_add_factors) = ex_add_factors.as_mut() {
                    ex_add_factors.push(dot_base + ex_correction);
                }
                if let Some(ex_scale_factors) = ex_scale_factors.as_mut() {
                    ex_scale_factors.push(factor_ratio(-norm_square, ex_res_dot));
                }
            }
            _ => unreachable!(),
        }
    }

    Ok(RabitRawQueryFactors {
        add_factors: Float32Array::from(add_factors),
        scale_factors: Float32Array::from(scale_factors),
        error_factors: Float32Array::from(error_factors),
        ex_add_factors: ex_add_factors.map(Float32Array::from),
        ex_scale_factors: ex_scale_factors.map(Float32Array::from),
    })
}

impl Debug for RQTransformer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "RabitTransformer(vector_column={})", self.vector_column)
    }
}

impl Transformer for RQTransformer {
    #[instrument(name = "RQTransformer::transform", level = "debug", skip_all)]
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let has_split_codes = self.rq.num_bits() == 1
            || (batch.column_by_name(RABIT_BLOCKED_EX_CODE_COLUMN).is_some()
                && batch.column_by_name(EX_ADD_FACTORS_COLUMN).is_some()
                && batch.column_by_name(EX_SCALE_FACTORS_COLUMN).is_some());
        if batch.column_by_name(RABIT_CODE_COLUMN).is_some() && has_split_codes {
            return Ok(batch.clone());
        }

        let residual_vectors = batch
            .column_by_name(&self.vector_column)
            .ok_or(Error::index(format!(
                "RQ Transform: column {} not found in batch",
                self.vector_column
            )))?;
        let residual_vectors = residual_vectors
            .as_fixed_size_list_opt()
            .ok_or(Error::index(format!(
                "RQ Transform: column {} is not a fixed size list, got {}",
                self.vector_column,
                residual_vectors.data_type(),
            )))?;

        let dist_v_c = batch
            .column_by_name(CENTROID_DIST_COLUMN)
            .ok_or(Error::index(format!(
                "RQ Transform: column {} not found in batch",
                CENTROID_DIST_COLUMN
            )))?;
        let dist_v_c = dist_v_c.as_primitive::<Float32Type>();

        let res_norm_square = match self.distance_type {
            // for L2, |v-c|^2 is just the distance to the centroid
            DistanceType::L2 => dist_v_c.clone(),
            DistanceType::Dot => Float32Array::from(norm_squared_fsl(residual_vectors)),
            _ => {
                return Err(Error::index(format!(
                    "RQ Transform: distance type {} not supported",
                    self.distance_type
                )));
            }
        };

        if self.rq.metadata_ref().query_estimator == RabitQueryEstimator::RawQuery {
            return self.transform_raw_query(batch, residual_vectors, &res_norm_square);
        }

        let rq_codes = self.rq.quantize_split(residual_vectors)?;
        let codes_fsl = rq_codes.binary_codes.as_fixed_size_list();
        debug_assert_eq!(codes_fsl.len(), batch.num_rows());

        let mut batch = batch.try_with_column(self.rq.field(), rq_codes.binary_codes)?;
        if self.rq.metadata_ref().query_estimator == RabitQueryEstimator::ResidualQuery {
            // Preserve the released residual-query estimator and factor layout.
            let ip_rq_res = match residual_vectors.value_type() {
                DataType::Float16 => Float32Array::from(
                    self.rq
                        .codes_res_dot_dists::<Float16Type>(residual_vectors)?,
                ),
                DataType::Float32 => Float32Array::from(
                    self.rq
                        .codes_res_dot_dists::<Float32Type>(residual_vectors)?,
                ),
                DataType::Float64 => Float32Array::from(
                    self.rq
                        .codes_res_dot_dists::<Float64Type>(residual_vectors)?,
                ),
                _ => {
                    return Err(Error::index(format!(
                        "RQ Transform: unsupported residual vector data type: {}",
                        residual_vectors.data_type()
                    )));
                }
            };

            let add_factors = match self.distance_type {
                DistanceType::L2 => res_norm_square.clone(),
                DistanceType::Dot => {
                    // for dot, the add factor is `1 - v*c + |c|^2 = dist_v_c + |c|^2`
                    let part_ids = &batch[PART_ID_COLUMN];
                    let part_ids = part_ids.as_primitive::<UInt32Type>();
                    let centroids_norm_square = self.centroids_norm_square.as_ref().ok_or(
                        Error::index("RQ Transform: centroids norm square not found".to_string()),
                    )?;
                    let centroids_norm_square =
                        arrow::compute::take(centroids_norm_square, part_ids, None)?;
                    let centroids_norm_square = centroids_norm_square.as_primitive::<Float32Type>();
                    Float32Array::from_iter_values(
                        dist_v_c
                            .values()
                            .iter()
                            .zip(centroids_norm_square.values().iter())
                            .map(|(dist_v_c, centroids_norm_square)| {
                                dist_v_c + centroids_norm_square
                            }),
                    )
                }
                _ => {
                    return Err(Error::index(format!(
                        "RQ Transform: distance type {} not supported",
                        self.distance_type
                    )));
                }
            };

            let scale_factors = match self.distance_type {
                DistanceType::L2 => Float32Array::from_iter_values(
                    res_norm_square.values().iter().zip(ip_rq_res.values()).map(
                        |(res_norm_square, ip_rq_res)| {
                            (-2.0 * res_norm_square)
                                .div_checked(*ip_rq_res)
                                .unwrap_or_default()
                        },
                    ),
                ),
                DistanceType::Dot => Float32Array::from_iter_values(
                    res_norm_square.values().iter().zip(ip_rq_res.values()).map(
                        |(res_norm_square, ip_rq_res)| {
                            -res_norm_square.div_checked(*ip_rq_res).unwrap_or_default()
                        },
                    ),
                ),
                _ => {
                    return Err(Error::index(format!(
                        "RQ Transform: distance type {} not supported",
                        self.distance_type
                    )));
                }
            };

            batch = batch
                .try_with_column(ADD_FACTORS_FIELD.clone(), Arc::new(add_factors))?
                .try_with_column(SCALE_FACTORS_FIELD.clone(), Arc::new(scale_factors))?;
        }

        let batch = batch
            .drop_column(&self.vector_column)?
            .drop_column(CENTROID_DIST_COLUMN)?;
        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::AsArray;
    use arrow::datatypes::{Float32Type, UInt8Type};
    use arrow_array::{
        Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, UInt32Array,
    };
    use lance_arrow::{FixedSizeListArrayExt, RecordBatchExt};
    use lance_linalg::distance::DistanceType;
    use rstest::rstest;

    use crate::vector::bq::RQRotationType;
    use crate::vector::bq::builder::RabitQuantizer;
    use crate::vector::bq::ex_dot::blocked_ex_code_bytes;
    use crate::vector::bq::storage::RABIT_BLOCKED_EX_CODE_COLUMN;
    use crate::vector::transform::Transformer;
    use crate::vector::{CENTROID_DIST_COLUMN, PART_ID_COLUMN};

    use super::{
        ADD_FACTORS_COLUMN, ERROR_FACTORS_COLUMN, EX_ADD_FACTORS_COLUMN, EX_SCALE_FACTORS_COLUMN,
        RQTransformer, compute_raw_query_factors, error_factor_value,
    };

    #[rstest]
    #[case(RQRotationType::Fast, 8)]
    #[case(RQRotationType::Fast, 768)]
    #[case(RQRotationType::Matrix, 8)]
    fn test_fused_quantization_matches_split_pipeline(
        #[case] rotation: RQRotationType,
        #[case] dim: usize,
        #[values(1, 3, 5, 9)] bits: u8,
        #[values(DistanceType::L2, DistanceType::Dot)] distance: DistanceType,
        #[values(
            arrow_schema::DataType::Float16,
            arrow_schema::DataType::Float32,
            arrow_schema::DataType::Float64
        )]
        value_type: arrow_schema::DataType,
    ) {
        let rq = match value_type {
            arrow_schema::DataType::Float16 => RabitQuantizer::new_with_rotation::<
                arrow::datatypes::Float16Type,
            >(bits, dim as i32, rotation),
            arrow_schema::DataType::Float64 => RabitQuantizer::new_with_rotation::<
                arrow::datatypes::Float64Type,
            >(bits, dim as i32, rotation),
            _ => RabitQuantizer::new_with_rotation::<Float32Type>(bits, dim as i32, rotation),
        };
        let centroid_values = Float32Array::from(
            (0..3 * dim)
                .map(|i| ((i * 17 % 103) as f32 - 51.0) / 64.0)
                .collect::<Vec<_>>(),
        );
        let centroids = FixedSizeListArray::try_new_from_values(
            arrow::compute::cast(&centroid_values, &value_type).unwrap(),
            dim as i32,
        )
        .unwrap();
        let transformer = RQTransformer::new(rq.clone(), distance, centroids, "vector").unwrap();
        let values = Float32Array::from(
            (0..18 * dim)
                .map(|i| {
                    if i / dim == 2 {
                        0.0
                    } else {
                        ((i * 31 % 107) as f32 - 53.0) / 64.0
                    }
                })
                .collect::<Vec<_>>(),
        );
        let values = arrow::compute::cast(&values, &value_type).unwrap();
        let residuals = FixedSizeListArray::try_new_from_values(values, dim as i32)
            .unwrap()
            .slice(1, 16);
        let norm = Float32Array::from(lance_linalg::distance::norm_squared_fsl(&residuals));
        let parts = UInt32Array::from((0..16).map(|i| i % 3).collect::<Vec<u32>>());
        let batch = RecordBatch::try_from_iter(vec![
            ("vector", Arc::new(residuals.clone()) as ArrayRef),
            (CENTROID_DIST_COLUMN, Arc::new(norm.clone()) as ArrayRef),
            (PART_ID_COLUMN, Arc::new(parts.clone()) as ArrayRef),
        ])
        .unwrap();
        let split = rq.quantize_split(&residuals).unwrap();
        let expected = compute_raw_query_factors(
            distance,
            &norm,
            split.rotated_residuals.as_ref().unwrap(),
            transformer.rotated_centroids.as_ref().unwrap(),
            &parts,
            split.ex_code_values.as_deref(),
            split.ex_res_dot_dists.as_deref(),
            bits - 1,
            dim,
        )
        .unwrap();
        let actual = transformer.transform(&batch).unwrap();
        assert_eq!(
            actual
                .column_by_name(crate::vector::bq::storage::RABIT_CODE_COLUMN)
                .unwrap()
                .to_data(),
            split.binary_codes.to_data()
        );
        if let Some(ex_codes) = split.ex_codes {
            assert_eq!(
                actual
                    .column_by_name(RABIT_BLOCKED_EX_CODE_COLUMN)
                    .unwrap()
                    .to_data(),
                ex_codes.to_data()
            );
        }
        let mut expected_columns = vec![
            (ADD_FACTORS_COLUMN, expected.add_factors),
            (super::SCALE_FACTORS_COLUMN, expected.scale_factors),
            (ERROR_FACTORS_COLUMN, expected.error_factors),
        ];
        if let (Some(add), Some(scale)) = (expected.ex_add_factors, expected.ex_scale_factors) {
            expected_columns.push((EX_ADD_FACTORS_COLUMN, add));
            expected_columns.push((EX_SCALE_FACTORS_COLUMN, scale));
        }
        for (name, expected) in expected_columns {
            let actual = actual
                .column_by_name(name)
                .unwrap()
                .as_primitive::<Float32Type>();
            assert_eq!(
                actual
                    .values()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .values()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "{name}"
            );
        }
        assert_eq!(
            transformer
                .transform(&batch.slice(0, 0))
                .unwrap()
                .num_rows(),
            0
        );
        let bad_parts = batch
            .drop_column(PART_ID_COLUMN)
            .unwrap()
            .try_with_column(
                arrow_schema::Field::new(PART_ID_COLUMN, arrow_schema::DataType::UInt32, false),
                Arc::new(UInt32Array::from(vec![u32::MAX; 16])),
            )
            .unwrap();
        let error = transformer.transform(&bad_parts).unwrap_err();
        assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
        assert!(error.to_string().contains("out of range"));
    }

    #[test]
    fn test_rq_transformer_writes_multi_bit_ex_factors() {
        let rq = RabitQuantizer::new_with_rotation::<Float32Type>(4, 8, RQRotationType::Fast);
        let centroids =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0f32; 8]), 8)
                .unwrap();
        let transformer =
            RQTransformer::new(rq.clone(), DistanceType::L2, centroids, "vector").unwrap();

        let residual_vectors = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![
                1.0, -2.0, 3.0, -4.0, 1.5, -2.5, 3.5, -4.5, 0.5, -1.0, 1.5, -2.0, 2.5, -3.0, 3.5,
                -4.0,
            ]),
            8,
        )
        .unwrap();
        let res_norm_square = Float32Array::from(vec![73.0f32, 47.0]);
        let batch = RecordBatch::try_from_iter(vec![
            ("vector", Arc::new(residual_vectors.clone()) as ArrayRef),
            (
                PART_ID_COLUMN,
                Arc::new(UInt32Array::from(vec![0, 0])) as ArrayRef,
            ),
            (
                CENTROID_DIST_COLUMN,
                Arc::new(res_norm_square.clone()) as ArrayRef,
            ),
        ])
        .unwrap();

        let transformed = transformer.transform(&batch).unwrap();
        assert!(
            transformed
                .column_by_name(RABIT_BLOCKED_EX_CODE_COLUMN)
                .is_some()
        );
        assert_eq!(
            transformed[RABIT_BLOCKED_EX_CODE_COLUMN]
                .as_fixed_size_list()
                .value_length(),
            blocked_ex_code_bytes(8, 3) as i32
        );
        assert!(
            transformed[RABIT_BLOCKED_EX_CODE_COLUMN]
                .as_fixed_size_list()
                .values()
                .as_primitive::<UInt8Type>()
                .values()
                .iter()
                .any(|value| *value != 0)
        );
        let expected_ex_dots = rq
            .quantize_split(&residual_vectors)
            .unwrap()
            .ex_res_dot_dists
            .unwrap();
        let ex_add_factors = transformed[EX_ADD_FACTORS_COLUMN].as_primitive::<Float32Type>();
        assert_eq!(ex_add_factors.values(), res_norm_square.values());
        let ex_scale_factors = transformed[EX_SCALE_FACTORS_COLUMN].as_primitive::<Float32Type>();
        for ((actual, norm), ex_dot) in ex_scale_factors
            .values()
            .iter()
            .zip(res_norm_square.values())
            .zip(expected_ex_dots)
        {
            let expected = if ex_dot == 0.0 {
                0.0
            } else {
                -2.0 * norm / ex_dot
            };
            assert!((actual - expected).abs() < 1e-6);
        }
        assert!(transformed.column_by_name("vector").is_none());
        assert!(transformed.column_by_name(CENTROID_DIST_COLUMN).is_none());
        assert!(transformed.column_by_name(ADD_FACTORS_COLUMN).is_some());
        assert!(transformed.column_by_name(ERROR_FACTORS_COLUMN).is_some());
    }

    #[test]
    fn test_rq_transformer_caches_rotated_centroids_for_raw_query() {
        let centroids =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0f32; 8]), 8)
                .unwrap();
        let raw_query_rq =
            RabitQuantizer::new_with_rotation::<Float32Type>(1, 8, RQRotationType::Fast);
        let raw_query_transformer =
            RQTransformer::new(raw_query_rq, DistanceType::L2, centroids.clone(), "vector")
                .unwrap();
        assert_eq!(
            raw_query_transformer
                .rotated_centroids
                .as_ref()
                .unwrap()
                .len(),
            8
        );

        let multi_bit_rq =
            RabitQuantizer::new_with_rotation::<Float32Type>(4, 8, RQRotationType::Fast);
        let multi_bit_transformer =
            RQTransformer::new(multi_bit_rq, DistanceType::L2, centroids, "vector").unwrap();
        assert_eq!(
            multi_bit_transformer
                .rotated_centroids
                .as_ref()
                .unwrap()
                .len(),
            8
        );
    }

    #[test]
    fn test_raw_query_factors_match_reference_l2_formula() {
        let res_norm_square = Float32Array::from(vec![5.0f32, 7.0]);
        let rotated_residuals = vec![2.0, -1.0, 0.0, 0.0];
        let rotated_centroids = vec![3.0, 4.0];
        let part_ids = UInt32Array::from(vec![0, 0]);
        let ex_code_values = vec![1, 0, 0, 0];
        let ex_res_dot_dists = vec![4.5, 0.0];

        let factors = compute_raw_query_factors(
            DistanceType::L2,
            &res_norm_square,
            &rotated_residuals,
            &rotated_centroids,
            &part_ids,
            Some(&ex_code_values),
            Some(&ex_res_dot_dists),
            1,
            2,
        )
        .unwrap();

        assert!((factors.add_factors.value(0) - 1.6666667).abs() < 1e-5);
        assert!((factors.scale_factors.value(0) + 6.6666665).abs() < 1e-5);
        let expected_error = error_factor_value(DistanceType::L2, 5.0, 1.5, 2);
        assert!((factors.error_factors.value(0) - expected_error).abs() < 1e-5);
        let ex_add_factors = factors.ex_add_factors.unwrap();
        let ex_scale_factors = factors.ex_scale_factors.unwrap();
        assert!((ex_add_factors.value(0) - 1.6666667).abs() < 1e-5);
        assert!((ex_scale_factors.value(0) + 2.2222223).abs() < 1e-5);
        assert_eq!(factors.add_factors.value(1), 7.0);
        assert_eq!(factors.scale_factors.value(1), 0.0);
        assert_eq!(factors.error_factors.value(1), 0.0);
        assert_eq!(ex_add_factors.value(1), 7.0);
        assert_eq!(ex_scale_factors.value(1), 0.0);
    }

    #[test]
    fn test_raw_query_factors_match_reference_dot_formula() {
        let res_norm_square = Float32Array::from(vec![5.0f32]);
        let rotated_residuals = vec![2.0, -1.0];
        let rotated_centroids = vec![3.0, 4.0];
        let part_ids = UInt32Array::from(vec![0]);
        let ex_code_values = vec![1, 0];
        let ex_res_dot_dists = vec![4.5];

        let factors = compute_raw_query_factors(
            DistanceType::Dot,
            &res_norm_square,
            &rotated_residuals,
            &rotated_centroids,
            &part_ids,
            Some(&ex_code_values),
            Some(&ex_res_dot_dists),
            1,
            2,
        )
        .unwrap();

        assert!((factors.add_factors.value(0) + 2.6666667).abs() < 1e-5);
        assert!((factors.scale_factors.value(0) + 3.3333333).abs() < 1e-5);
        let expected_error = error_factor_value(DistanceType::Dot, 5.0, 1.5, 2);
        assert!((factors.error_factors.value(0) - expected_error).abs() < 1e-5);
        let ex_add_factors = factors.ex_add_factors.unwrap();
        let ex_scale_factors = factors.ex_scale_factors.unwrap();
        assert!((ex_add_factors.value(0) + 2.6666667).abs() < 1e-5);
        assert!((ex_scale_factors.value(0) + 1.1111112).abs() < 1e-5);
    }
}
