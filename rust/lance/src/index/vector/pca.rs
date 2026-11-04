// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Dataset training for the IVF_RQ PCA pre-transform.

use arrow::datatypes::{Float16Type, Float32Type, Float64Type};
use arrow_array::Array;
use arrow_array::cast::AsArray;
use lance_index::vector::pca::{PcaModel, PcaParams, accumulate_centered_scatter};

use super::utils::{get_vector_dim, sample_finite_vectors_seeded};
use crate::dataset::Dataset;
use crate::{Error, Result};

const SCATTER_CHUNK_ROWS: usize = 1024;

pub struct PcaTrainer;

impl PcaTrainer {
    /// Trains one global model from finite rows of `column`.
    ///
    /// Sampling keeps the configured seed and draws more candidates until
    /// `sample_size` valid rows are collected or the dataset is exhausted.
    /// Mean and scatter are accumulated in row chunks, without retaining a
    /// centered copy of the sample.
    pub async fn train_dataset(
        dataset: &Dataset,
        column: &str,
        params: &PcaParams,
    ) -> Result<PcaModel> {
        let input_dimension = get_vector_dim(dataset.schema(), column)?;
        let sample =
            sample_finite_vectors_seeded(dataset, column, params.sample_size, params.sampling_seed)
                .await?;
        let row_count = sample.len();
        if sample.value_length() < 0 || sample.value_length() as usize != input_dimension {
            return Err(Error::invalid_input(format!(
                "PCA sample dimension {} does not match column dimension {input_dimension}",
                sample.value_length()
            )));
        }
        params.validate(input_dimension, row_count)?;

        let mut sums = vec![0.0; input_dimension];
        for start in (0..row_count).step_by(SCATTER_CHUNK_ROWS) {
            let rows = SCATTER_CHUNK_ROWS.min(row_count - start);
            let values = rows_as_f64(&sample, start, rows)?;
            for row in values.chunks_exact(input_dimension) {
                for (column_idx, value) in row.iter().enumerate() {
                    sums[column_idx] += *value;
                }
            }
        }
        let denominator = row_count as f64;
        for sum in &mut sums {
            *sum /= denominator;
            if !sum.is_finite() {
                return Err(Error::invalid_input(
                    "PCA mean calculation produced a non-finite value".to_string(),
                ));
            }
        }

        let mut scatter = vec![0.0; input_dimension * input_dimension];
        for start in (0..row_count).step_by(SCATTER_CHUNK_ROWS) {
            let rows = SCATTER_CHUNK_ROWS.min(row_count - start);
            let values = rows_as_f64(&sample, start, rows)?;
            accumulate_centered_scatter(&values, &sums, &mut scatter)?;
        }
        PcaModel::train_from_scatter(&sums, row_count, &scatter, params)
    }
}

fn rows_as_f64(
    sample: &arrow_array::FixedSizeListArray,
    start: usize,
    rows: usize,
) -> Result<Vec<f64>> {
    let dimension = sample.value_length() as usize;
    let mut values = Vec::with_capacity(rows * dimension);
    for row_idx in start..start + rows {
        let row = sample.value(row_idx);
        match row.data_type() {
            arrow::datatypes::DataType::Float32 => {
                for value in row.as_primitive::<Float32Type>().values() {
                    values.push(f64::from(*value));
                }
            }
            arrow::datatypes::DataType::Float64 => {
                values.extend_from_slice(row.as_primitive::<Float64Type>().values());
            }
            arrow::datatypes::DataType::Float16 => {
                for value in row.as_primitive::<Float16Type>().values() {
                    values.push(value.to_f64());
                }
            }
            other => {
                return Err(Error::invalid_input(format!(
                    "PCA training does not support vector values of type {other}"
                )));
            }
        }
    }
    Ok(values)
}
