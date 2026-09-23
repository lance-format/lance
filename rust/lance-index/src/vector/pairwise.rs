// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bounded reconstruction of index representations for symmetric pair scoring.

use arrow_array::cast::AsArray;
use arrow_array::types::{Float16Type, Float32Type, Float64Type, UInt8Type};
use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float64Array, RecordBatch, UInt64Array};
use arrow_schema::DataType;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;

use super::bq::ex_dot::{blocked_ex_code_bytes, unpack_blocked_row};
use super::bq::storage::{RABIT_BLOCKED_EX_CODE_COLUMN, RabitQueryEstimator, unpack_codes};
use super::bq::transform::{EX_SCALE_FACTORS_COLUMN, SCALE_FACTORS_COLUMN};
use super::quantizer::{Quantization, Quantizer};

/// Internal index batch. Vectors may be reconstructed and, for RQ, rotated.
/// All batches of a partition use the same coordinate system.
#[derive(Clone, Debug)]
pub struct PairwiseVectorBatch {
    pub row_ids: UInt64Array,
    pub vectors: FixedSizeListArray,
}

fn float_values(array: &dyn Array) -> Result<Vec<f64>> {
    match array.data_type() {
        DataType::Float16 => Ok(array
            .as_primitive::<Float16Type>()
            .values()
            .iter()
            .map(|v| v.to_f64())
            .collect()),
        DataType::Float32 => Ok(array
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .map(|&v| f64::from(v))
            .collect()),
        DataType::Float64 => Ok(array.as_primitive::<Float64Type>().values().to_vec()),
        other => Err(Error::not_supported(format!(
            "pair reconstruction requires floating-point values, got {other}"
        ))),
    }
}

fn codes<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a FixedSizeListArray> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_fixed_size_list_opt())
        .ok_or_else(|| Error::invalid_input(format!("missing fixed-size vector column {name}")))
}

/// Reconstruct only the supplied batch, never fetching source-table vectors.
/// Quantized pair distances are distances between these reconstructions, not
/// asymmetric query-to-code estimates. This gives identical codes zero L2.
pub(crate) fn reconstruct(
    quantizer: &Quantizer,
    batch: &RecordBatch,
    centroid: ArrayRef,
    metric: DistanceType,
) -> Result<FixedSizeListArray> {
    let encoded = codes(batch, quantizer.column())?;
    if matches!(quantizer, Quantizer::Flat(_) | Quantizer::FlatBin(_)) {
        return Ok(encoded.clone());
    }
    let center = float_values(centroid.as_ref())?;
    let dim;
    let mut values = Vec::new();
    match quantizer {
        Quantizer::Product(pq) => {
            dim = pq.dimension;
            let width = dim / pq.num_sub_vectors;
            let codebook = float_values(pq.codebook.values().as_ref())?;
            let num_centroids = 1usize << pq.num_bits;
            let raw = encoded.values().as_primitive::<UInt8Type>();
            values.reserve(batch.num_rows() * dim);
            for row in 0..batch.num_rows() {
                let bytes = &raw.values()[row * encoded.value_length() as usize
                    ..(row + 1) * encoded.value_length() as usize];
                for sub in 0..pq.num_sub_vectors {
                    let code = if pq.num_bits == 4 {
                        (bytes[sub / 2] >> (4 * (sub % 2))) & 15
                    } else {
                        bytes[sub]
                    } as usize;
                    let start = (sub * num_centroids + code) * width;
                    for (offset, &value) in codebook[start..start + width].iter().enumerate() {
                        let c = if super::pq::ProductQuantizer::use_residual(metric) {
                            center[sub * width + offset]
                        } else {
                            0.0
                        };
                        values.push(value + c);
                    }
                }
            }
        }
        Quantizer::Scalar(sq) => {
            dim = encoded.value_length() as usize;
            let bounds = sq.bounds();
            let scale = (bounds.end - bounds.start) / 255.0;
            values = encoded
                .values()
                .as_primitive::<UInt8Type>()
                .values()
                .iter()
                .map(|&v| bounds.start + f64::from(v) * scale)
                .collect();
        }
        Quantizer::Rabit(rq) => {
            let meta = rq.metadata_ref();
            if meta.query_estimator != RabitQueryEstimator::RawQuery {
                return Err(Error::not_supported(
                    "pair enumeration requires a current RQ index; rebuild the index",
                ));
            }
            dim = meta.rotated_dim();
            let centroid = FixedSizeListArray::try_new_from_values(centroid, center.len() as i32)?;
            let rotated_center = rq.rotate_fsl_to_f32(&centroid)?;
            // FastScan packs groups of 32 rows. The reader aligns the batch to
            // those groups before calling this decoder.
            let signs = if meta.packed {
                unpack_codes(encoded)
            } else {
                encoded.clone()
            };
            let signs = signs.values().as_primitive::<UInt8Type>();
            let ex_bits = meta.num_bits - 1;
            let scales_name = if ex_bits == 0 {
                SCALE_FACTORS_COLUMN
            } else {
                EX_SCALE_FACTORS_COLUMN
            };
            let scales = batch
                .column_by_name(scales_name)
                .and_then(|a| a.as_primitive_opt::<Float32Type>())
                .ok_or_else(|| {
                    Error::invalid_input(format!("missing RQ scale column {scales_name}"))
                })?;
            let extended = if ex_bits == 0 {
                None
            } else {
                Some(codes(batch, RABIT_BLOCKED_EX_CODE_COLUMN)?)
            };
            values.reserve(batch.num_rows() * dim);
            for row in 0..batch.num_rows() {
                let ex = if let Some(extended) = extended {
                    let data = extended.value(row);
                    let raw = data.as_primitive::<UInt8Type>();
                    if raw.len() != blocked_ex_code_bytes(dim, ex_bits) {
                        return Err(Error::invalid_input("invalid RQ extended code width"));
                    }
                    unpack_blocked_row(raw.values(), ex_bits, dim)
                } else {
                    Vec::new()
                };
                let scale = -f64::from(scales.value(row))
                    / if metric == DistanceType::Dot {
                        1.0
                    } else {
                        2.0
                    };
                let bias = (1u32 << ex_bits) as f64 - 0.5;
                for d in 0..dim {
                    let sign = (signs.value(row * (dim / 8) + d / 8) >> (d % 8)) & 1;
                    let code = (u32::from(sign) << ex_bits)
                        + if ex_bits == 0 { 0 } else { u32::from(ex[d]) };
                    values.push(f64::from(rotated_center[d]) + scale * (f64::from(code) - bias));
                }
            }
        }
        Quantizer::Flat(_) | Quantizer::FlatBin(_) => return Ok(encoded.clone()),
    }
    Ok(FixedSizeListArray::try_new_from_values(
        Float64Array::from(values),
        dim as i32,
    )?)
}
