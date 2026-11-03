// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Principal component analysis for vector pre-transforms.

use std::alloc::Layout;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
use std::borrow::Cow;
use std::mem::size_of;
use std::sync::{Arc, OnceLock};

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array,
    cast::AsArray,
    types::{Float16Type, Float32Type, Float64Type},
};
use arrow_schema::DataType;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::{Error, Result};
#[cfg(not(target_arch = "aarch64"))]
use lance_linalg::distance::dot::dot_f32 as dot_row;
use ndarray::{Array2, ArrayView2};
use prost::Message;

use crate::pb;

/// Arrow schema metadata key containing the PCA model global-buffer index.
pub const PCA_METADATA_KEY: &str = "lance:pca";

/// Current experimental PCA model serialization version.
pub const PCA_MODEL_VERSION: u32 = 1;

/// A PCA projection that produced a non-finite output component.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PcaNonFiniteOutput {
    pub row: usize,
    pub output_dimension: usize,
    pub value: f32,
}

impl std::fmt::Display for PcaNonFiniteOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PCA transform produced a non-finite value at row={}, output_dimension={}, value={}",
            self.row, self.output_dimension, self.value
        )
    }
}

impl std::error::Error for PcaNonFiniteOutput {}

/// Returns the structured non-finite PCA output carried by `error`, if any.
pub(crate) fn non_finite_output(error: &Error) -> Option<PcaNonFiniteOutput> {
    let mut current = Some(error as &(dyn std::error::Error + 'static));
    while let Some(err) = current {
        if let Some(found) = err.downcast_ref::<PcaNonFiniteOutput>() {
            return Some(*found);
        }
        current = err.source();
    }
    None
}

const PCA_MODEL_NAME: &str = "PCA model";
const PCA_FINGERPRINT_DOMAIN: &[u8] = b"lance-vector-pre-transform:pca:v1";

const DEFAULT_SAMPLE_SIZE: usize = 65_536;
const DEFAULT_SAMPLING_SEED: u64 = 42;
const TRANSFORM_BATCH_SIZE: usize = 4096;

/// Configuration for training a PCA vector transform.
///
/// This is an experimental format feature. The parameters and stored model may
/// change incompatibly, in which case the index must be rebuilt. The source
/// data is not rewritten.
///
/// ```
/// use lance_index::vector::pca::PcaParams;
///
/// let params = PcaParams::new(64);
/// assert_eq!(params.dimension, 64);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PcaParams {
    /// Number of dimensions produced by the transform.
    pub dimension: usize,
    /// Target number of valid rows used by dataset-aware training.
    pub sample_size: usize,
    /// Seed used by dataset-aware sampling.
    pub sampling_seed: u64,
}

impl PcaParams {
    /// Creates PCA parameters with the documented defaults.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            sample_size: DEFAULT_SAMPLE_SIZE,
            sampling_seed: DEFAULT_SAMPLING_SEED,
        }
    }

    pub fn with_sample_size(mut self, sample_size: usize) -> Self {
        self.sample_size = sample_size;
        self
    }

    pub fn with_sampling_seed(mut self, sampling_seed: u64) -> Self {
        self.sampling_seed = sampling_seed;
        self
    }

    /// Validates these parameters for a training input shape.
    ///
    /// `row_count` is the number of valid vectors that will contribute to the
    /// model; `input_dimension` is the number of scalar values in each vector.
    pub fn validate(&self, input_dimension: usize, row_count: usize) -> Result<()> {
        if self.dimension == 0 || self.dimension >= input_dimension {
            return Err(Error::invalid_input(format!(
                "pca.dimension={} must be greater than 0 and smaller than input_dimension={input_dimension}",
                self.dimension
            )));
        }
        if self.sample_size == 0 {
            return Err(Error::invalid_input(
                "pca.sample_size must be greater than 0".to_string(),
            ));
        }
        let required_rows = self
            .dimension
            .checked_add(1)
            .ok_or_else(|| Error::invalid_input("pca.dimension+1 overflows usize"))?;
        if self.sample_size < required_rows {
            return Err(Error::invalid_input(format!(
                "pca.sample_size={} must be at least dimension+1={required_rows}",
                self.sample_size
            )));
        }
        if row_count < required_rows {
            return Err(Error::invalid_input(format!(
                "PCA training requires at least output_dimension+1={required_rows} valid vectors, got {row_count}"
            )));
        }
        Ok(())
    }
}

/// A trained PCA transform represented as `y = A * (x - mean)`.
///
/// This is an experimental format feature. The stored model may change
/// incompatibly, in which case the index must be rebuilt. The source data is
/// not rewritten.
///
/// ```
/// use arrow_array::{FixedSizeListArray, Float32Array};
/// use lance_arrow::FixedSizeListArrayExt;
/// use lance_index::vector::pca::{PcaModel, PcaParams};
///
/// # fn train_and_transform() -> lance_core::Result<()> {
/// let vectors = FixedSizeListArray::try_new_from_values(
///     Float32Array::from(vec![
///         2.0, 0.0, 0.0,
///        -2.0, 0.0, 0.0,
///         0.0, 1.0, 0.0,
///         0.0,-1.0, 0.0,
///     ]),
///     3,
/// )?;
/// let model = PcaModel::train(&vectors, &PcaParams::new(1))?;
/// let projected = model.transform(&vectors)?;
/// assert_eq!(projected.value_length(), 1);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct PcaModel {
    input_dimension: usize,
    output_dimension: usize,
    transform_matrix: Float32Array,
    mean: Float32Array,
    fingerprint: OnceLock<[u8; 32]>,
}

impl DeepSizeOf for PcaModel {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        (&self.transform_matrix as &dyn Array).deep_size_of_children(context)
            + (&self.mean as &dyn Array).deep_size_of_children(context)
    }
}

fn encode_f32_tensor(values: &[f32], shape: Vec<u32>, label: &str) -> Result<pb::Tensor> {
    let byte_len = values
        .len()
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| Error::index(format!("{PCA_MODEL_NAME} {label} data length overflows")))?;
    let mut data = Vec::with_capacity(byte_len);
    for value in values {
        data.extend_from_slice(&value.to_le_bytes());
    }
    Ok(pb::Tensor {
        data_type: pb::tensor::DataType::Float32 as i32,
        shape,
        data,
    })
}

fn decode_f32_tensor(
    tensor: &pb::Tensor,
    expected_rank: usize,
    label: &str,
    model_version: u32,
) -> Result<Vec<f32>> {
    if tensor.data_type != pb::tensor::DataType::Float32 as i32 {
        return Err(Error::corrupt_file_named(
            PCA_MODEL_NAME,
            format!(
                "model_version={model_version} {label} has data_type={}, expected FLOAT32; shape={:?}, data length={}",
                tensor.data_type,
                tensor.shape,
                tensor.data.len()
            ),
        ));
    }
    if tensor.shape.len() != expected_rank {
        return Err(Error::corrupt_file_named(
            PCA_MODEL_NAME,
            format!(
                "model_version={model_version} {label} has rank={}, expected rank={expected_rank}; shape={:?}, data length={}",
                tensor.shape.len(),
                tensor.shape,
                tensor.data.len()
            ),
        ));
    }
    if tensor.shape.contains(&0) {
        return Err(Error::corrupt_file_named(
            PCA_MODEL_NAME,
            format!(
                "model_version={model_version} {label} has zero dimension; shape={:?}, data length={}",
                tensor.shape,
                tensor.data.len()
            ),
        ));
    }
    let element_count = tensor.shape.iter().try_fold(1_usize, |count, dimension| {
        let dimension = usize::try_from(*dimension).map_err(|_| {
            Error::corrupt_file_named(
                PCA_MODEL_NAME,
                format!(
                    "model_version={model_version} {label} dimension={dimension} cannot convert to usize; shape={:?}, data length={}",
                    tensor.shape,
                    tensor.data.len()
                ),
            )
        })?;
        count.checked_mul(dimension).ok_or_else(|| {
            Error::corrupt_file_named(
                PCA_MODEL_NAME,
                format!(
                    "model_version={model_version} {label} element count overflows usize; shape={:?}, data length={}",
                    tensor.shape,
                    tensor.data.len()
                ),
            )
        })
    })?;
    let expected_data_len = element_count.checked_mul(size_of::<f32>()).ok_or_else(|| {
        Error::corrupt_file_named(
            PCA_MODEL_NAME,
            format!(
                    "model_version={model_version} {label} byte length overflows usize; shape={:?}, data length={}",
                tensor.shape,
                tensor.data.len()
            ),
        )
    })?;
    if tensor.data.len() != expected_data_len {
        return Err(Error::corrupt_file_named(
            PCA_MODEL_NAME,
            format!(
                "model_version={model_version} {label} shape={:?} requires data length={expected_data_len}, got data length={}",
                tensor.shape,
                tensor.data.len()
            ),
        ));
    }
    let mut values = Vec::with_capacity(element_count);
    for (index, bytes) in tensor.data.as_chunks::<4>().0.iter().enumerate() {
        let value = f32::from_le_bytes(*bytes);
        if !value.is_finite() {
            return Err(Error::corrupt_file_named(
                PCA_MODEL_NAME,
                format!(
                    "model_version={model_version} {label} contains NaN or infinity at element={index}; shape={:?}, data length={}",
                    tensor.shape,
                    tensor.data.len()
                ),
            ));
        }
        values.push(value);
    }
    Ok(values)
}

impl TryFrom<&PcaModel> for pb::PcaModel {
    type Error = Error;

    fn try_from(model: &PcaModel) -> Result<Self> {
        let output_dimension = u32::try_from(model.output_dimension).map_err(|_| {
            Error::index(format!(
                "{PCA_MODEL_NAME} output_dimension={} exceeds u32",
                model.output_dimension
            ))
        })?;
        let input_dimension = u32::try_from(model.input_dimension).map_err(|_| {
            Error::index(format!(
                "{PCA_MODEL_NAME} input_dimension={} exceeds u32",
                model.input_dimension
            ))
        })?;
        Ok(Self {
            model_version: PCA_MODEL_VERSION,
            transform_matrix: Some(encode_f32_tensor(
                model.transform_matrix.values(),
                vec![output_dimension, input_dimension],
                "transform_matrix",
            )?),
            mean: Some(encode_f32_tensor(
                model.mean.values(),
                vec![input_dimension],
                "mean",
            )?),
        })
    }
}

impl TryFrom<pb::PcaModel> for PcaModel {
    type Error = Error;

    fn try_from(model: pb::PcaModel) -> Result<Self> {
        if model.model_version != PCA_MODEL_VERSION {
            return Err(Error::corrupt_file_named(
                PCA_MODEL_NAME,
                format!(
                    "unsupported model_version={}, expected version={PCA_MODEL_VERSION}",
                    model.model_version
                ),
            ));
        }
        let transform_tensor = model.transform_matrix.as_ref().ok_or_else(|| {
            Error::corrupt_file_named(
                PCA_MODEL_NAME,
                format!(
                    "missing transform_matrix message for model_version={}",
                    model.model_version
                ),
            )
        })?;
        let mean_tensor = model.mean.as_ref().ok_or_else(|| {
            Error::corrupt_file_named(
                PCA_MODEL_NAME,
                format!(
                    "missing mean message for model_version={}; matrix shape={:?}, data length={}",
                    model.model_version,
                    transform_tensor.shape,
                    transform_tensor.data.len()
                ),
            )
        })?;
        let transform_matrix =
            decode_f32_tensor(transform_tensor, 2, "transform_matrix", model.model_version)?;
        let mean = decode_f32_tensor(mean_tensor, 1, "mean", model.model_version)?;
        let output_dimension = usize::try_from(transform_tensor.shape[0]).map_err(|_| {
            Error::corrupt_file_named(
                PCA_MODEL_NAME,
                format!(
                    "output dimension={} cannot convert to usize; matrix shape={:?}, data length={}",
                    transform_tensor.shape[0],
                    transform_tensor.shape,
                    transform_tensor.data.len()
                ),
            )
        })?;
        let input_dimension = usize::try_from(transform_tensor.shape[1]).map_err(|_| {
            Error::corrupt_file_named(
                PCA_MODEL_NAME,
                format!(
                    "input dimension={} cannot convert to usize; matrix shape={:?}, data length={}",
                    transform_tensor.shape[1],
                    transform_tensor.shape,
                    transform_tensor.data.len()
                ),
            )
        })?;
        if output_dimension >= input_dimension {
            return Err(Error::corrupt_file_named(
                PCA_MODEL_NAME,
                format!(
                    "output_dimension={output_dimension} must be smaller than input_dimension={input_dimension}; model_version={}, matrix shape={:?}, data length={}",
                    model.model_version,
                    transform_tensor.shape,
                    transform_tensor.data.len()
                ),
            ));
        }
        if mean_tensor.shape[0] != transform_tensor.shape[1] || mean.len() != input_dimension {
            return Err(Error::corrupt_file_named(
                PCA_MODEL_NAME,
                format!(
                    "mean shape={:?} and data length={} do not match input_dimension={input_dimension}; model_version={}, matrix shape={:?}, data length={}",
                    mean_tensor.shape,
                    mean_tensor.data.len(),
                    model.model_version,
                    transform_tensor.shape,
                    transform_tensor.data.len()
                ),
            ));
        }
        Ok(Self {
            input_dimension,
            output_dimension,
            transform_matrix: Float32Array::from(transform_matrix),
            mean: Float32Array::from(mean),
            fingerprint: OnceLock::new(),
        })
    }
}

/// Encodes a PCA model as the versioned protobuf payload stored in a global buffer.
pub fn encode_pca_model(model: &PcaModel) -> Result<Vec<u8>> {
    Ok(pb::PcaModel::try_from(model)?.encode_to_vec())
}

/// Decodes and strictly validates a versioned PCA protobuf payload.
///
/// Malformed protobuf data and invalid model contents are reported as
/// [`Error::CorruptFile`].
pub fn decode_pca_model(data: &[u8]) -> Result<PcaModel> {
    let model = pb::PcaModel::decode(data).map_err(|error| {
        Error::corrupt_file_named(
            PCA_MODEL_NAME,
            format!(
                "failed to decode protobuf with data length={}: {error}",
                data.len()
            ),
        )
    })?;
    model.try_into()
}

impl PcaModel {
    /// Trains a model from an in-memory fixed-size-list vector array.
    ///
    /// Training rejects null or non-finite values. All statistics and matrix
    /// decompositions use `f64`; the returned runtime model uses `f32`.
    /// Input values are converted while accumulating statistics, without a
    /// full-size `f64` copy of the training data.
    pub fn train(data: &FixedSizeListArray, params: &PcaParams) -> Result<Self> {
        let (input_dimension, values) = vector_values(data)?;
        params.validate(input_dimension, data.len())?;
        if values.null_count() != 0 {
            return Err(Error::invalid_input(format!(
                "PCA input contains {} null scalar values",
                values.null_count()
            )));
        }
        match values.data_type() {
            DataType::Float16 => train_values(
                values.as_primitive::<Float16Type>().values(),
                input_dimension,
                params,
            ),
            DataType::Float32 => train_values(
                values.as_primitive::<Float32Type>().values(),
                input_dimension,
                params,
            ),
            DataType::Float64 => train_values(
                values.as_primitive::<Float64Type>().values(),
                input_dimension,
                params,
            ),
            data_type => Err(Error::invalid_input(format!(
                "PCA input must be Float16, Float32, or Float64, got {data_type}"
            ))),
        }
    }

    /// Trains a model from a mean and row-major centered scatter matrix.
    ///
    /// `scatter` must equal the sum of centered outer products and must be a
    /// finite, symmetric `mean.len() * mean.len()` matrix. Population
    /// covariance has the same eigenvectors as scatter. This is the
    /// statistics entry point used by streaming dataset training.
    pub fn train_from_scatter(
        mean: &[f64],
        sample_count: usize,
        scatter: &[f64],
        params: &PcaParams,
    ) -> Result<Self> {
        let input_dimension = mean.len();
        params.validate(input_dimension, sample_count)?;
        validate_finite_slice(mean, "mean")?;
        validate_symmetric_matrix(scatter, input_dimension, "scatter")?;
        let transform_matrix =
            principal_components(scatter, input_dimension, sample_count, params.dimension)?;
        let mean = Float32Array::from(
            mean.iter()
                .map(|value| canonical_f32(*value))
                .collect::<Result<Vec<_>>>()?,
        );
        Ok(Self {
            input_dimension,
            output_dimension: params.dimension,
            transform_matrix,
            mean,
            fingerprint: OnceLock::new(),
        })
    }

    /// Applies the PCA transform to a batch of vectors.
    pub fn transform(&self, vectors: &FixedSizeListArray) -> Result<FixedSizeListArray> {
        let (dimension, values) = vector_values(vectors)?;
        if dimension != self.input_dimension {
            return Err(Error::invalid_input(format!(
                "PCA vector dimension {dimension} does not match input_dimension={}",
                self.input_dimension,
            )));
        }
        let row_count = vectors.len();
        let input = values_as_f32(values.as_ref())?;
        let output_len = row_count
            .checked_mul(self.output_dimension)
            .ok_or_else(|| Error::invalid_input("PCA output shape overflows usize"))?;
        let mut output = Vec::with_capacity(output_len);
        let matrix = ArrayView2::from_shape(
            (self.output_dimension, self.input_dimension),
            self.transform_matrix.values().as_ref(),
        )
        .map_err(|error| Error::internal(format!("invalid PCA model matrix shape: {error}")))?;
        for chunk_start in (0..row_count).step_by(TRANSFORM_BATCH_SIZE) {
            let chunk_end = chunk_start
                .saturating_add(TRANSFORM_BATCH_SIZE)
                .min(row_count);
            let chunk_rows = chunk_end - chunk_start;
            let centered =
                Array2::from_shape_fn((chunk_rows, self.input_dimension), |(row, column)| {
                    input[(chunk_start + row) * self.input_dimension + column]
                        - self.mean.value(column)
                });
            let transformed = centered.dot(&matrix.t());
            for row in 0..chunk_rows {
                for column in 0..self.output_dimension {
                    let value = transformed[[row, column]];
                    if !value.is_finite() {
                        return Err(Error::invalid_input_source(Box::new(PcaNonFiniteOutput {
                            row: chunk_start + row,
                            output_dimension: column,
                            value,
                        })));
                    }
                    output.push(value);
                }
            }
        }
        let output_dimension = i32::try_from(self.output_dimension).map_err(|_| {
            Error::invalid_input(format!(
                "PCA output_dimension={} exceeds i32",
                self.output_dimension
            ))
        })?;
        Ok(FixedSizeListArray::try_new_from_values(
            Float32Array::from(output),
            output_dimension,
        )?)
    }

    /// Applies the PCA transform to one one-dimensional query vector.
    pub fn transform_query(&self, query: &dyn Array) -> Result<ArrayRef> {
        if query.len() != self.input_dimension {
            return Err(Error::invalid_input(format!(
                "PCA query dimension {} does not match input_dimension={}",
                query.len(),
                self.input_dimension
            )));
        }
        let input = values_as_f32(query)?;
        let centered: Vec<f32> = input
            .iter()
            .zip(self.mean.values())
            .map(|(value, mean)| value - mean)
            .collect();
        let mut output = Vec::with_capacity(self.output_dimension);
        let matrix = self.transform_matrix.values().as_ref();
        #[cfg(target_arch = "aarch64")]
        let matrix = {
            let mut groups = matrix.chunks_exact(4 * self.input_dimension);
            for rows in groups.by_ref() {
                output.extend(dot_4rows(&centered, rows));
            }
            groups.remainder()
        };
        output.extend(
            matrix
                .chunks_exact(self.input_dimension)
                .map(|row| dot_row(&centered, row)),
        );
        if let Some((row, _)) = output
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(Error::invalid_input(format!(
                "PCA transform produced a non-finite value at output dimension {row}"
            )));
        }
        Ok(Arc::new(Float32Array::from(output)))
    }

    pub fn input_dimension(&self) -> usize {
        self.input_dimension
    }

    pub fn output_dimension(&self) -> usize {
        self.output_dimension
    }

    pub fn transform_matrix(&self) -> &Float32Array {
        &self.transform_matrix
    }

    pub fn mean(&self) -> &Float32Array {
        &self.mean
    }

    /// Returns the stable BLAKE3 identity of this model's numerical transform.
    ///
    /// The digest covers the PCA v1 domain separator, model version, row-major
    /// transform matrix, and mean. Floating-point values use little-endian
    /// bytes, with negative zero normalized to positive zero.
    pub fn fingerprint(&self) -> [u8; 32] {
        *self.fingerprint.get_or_init(|| self.compute_fingerprint())
    }

    fn compute_fingerprint(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(PCA_FINGERPRINT_DOMAIN);
        hasher.update(&PCA_MODEL_VERSION.to_le_bytes());
        for value in self
            .transform_matrix
            .values()
            .iter()
            .chain(self.mean.values())
        {
            let canonical = if *value == 0.0 { 0.0 } else { *value };
            hasher.update(&canonical.to_le_bytes());
        }
        *hasher.finalize().as_bytes()
    }
}

fn vector_values(data: &FixedSizeListArray) -> Result<(usize, ArrayRef)> {
    let dimension = usize::try_from(data.value_length()).map_err(|_| {
        Error::invalid_input(format!(
            "PCA vector value_length={} is invalid",
            data.value_length()
        ))
    })?;
    if data.null_count() != 0 {
        return Err(Error::invalid_input(format!(
            "PCA vectors contain {} null rows",
            data.null_count(),
        )));
    }
    let start = data
        .offset()
        .checked_mul(dimension)
        .ok_or_else(|| Error::invalid_input("PCA vector value offset overflows usize"))?;
    let length = data
        .len()
        .checked_mul(dimension)
        .ok_or_else(|| Error::invalid_input("PCA vector values length overflows usize"))?;
    let end = start
        .checked_add(length)
        .ok_or_else(|| Error::invalid_input("PCA vector value range overflows usize"))?;
    if end > data.values().len() {
        return Err(Error::invalid_input(format!(
            "PCA vector value range [{start}, {end}) exceeds child values length {}",
            data.values().len(),
        )));
    }
    Ok((dimension, data.values().slice(start, length)))
}

fn validate_finite_slice(values: &[f64], label: &str) -> Result<()> {
    if let Some((index, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(Error::invalid_input(format!(
            "PCA {label} contains non-finite value={value} at element={index}, data length={}",
            values.len()
        )));
    }
    Ok(())
}

fn validate_symmetric_matrix(values: &[f64], dimension: usize, label: &str) -> Result<()> {
    let expected_len = dimension.checked_mul(dimension).ok_or_else(|| {
        Error::invalid_input(format!(
            "PCA {label} shape [{dimension}, {dimension}] overflows usize"
        ))
    })?;
    if values.len() != expected_len {
        return Err(Error::invalid_input(format!(
            "PCA {label} values length={} does not match row-major shape [{dimension}, {dimension}] requiring {expected_len} values",
            values.len()
        )));
    }
    validate_finite_slice(values, label)?;
    for row in 0..dimension {
        for column in row + 1..dimension {
            let upper = values[row * dimension + column];
            let lower = values[column * dimension + row];
            let tolerance = 64.0 * f64::EPSILON * upper.abs().max(lower.abs()).max(1.0);
            if (upper - lower).abs() > tolerance {
                return Err(Error::invalid_input(format!(
                    "PCA {label} must be symmetric: value[{row}, {column}]={upper}, value[{column}, {row}]={lower}, tolerance={tolerance}, shape=[{dimension}, {dimension}]"
                )));
            }
        }
    }
    Ok(())
}

fn train_values<T: Copy + Into<f64>>(
    values: &[T],
    dimension: usize,
    params: &PcaParams,
) -> Result<PcaModel> {
    let rows = values.len() / dimension;
    let mean = compensated_mean(values, rows, dimension)?;
    let mut scatter = new_scatter(dimension)?;
    let batch_values = TRANSFORM_BATCH_SIZE
        .checked_mul(dimension)
        .ok_or_else(|| Error::invalid_input("PCA training batch shape overflows usize"))?;
    for batch in values.chunks(batch_values) {
        accumulate_scatter(batch, &mean, &mut scatter)?;
    }
    PcaModel::train_from_scatter(&mean, rows, &scatter, params)
}

fn values_as_f32(values: &dyn Array) -> Result<Cow<'_, [f32]>> {
    if values.null_count() != 0 {
        return Err(Error::invalid_input(format!(
            "PCA input contains {} null scalar values",
            values.null_count()
        )));
    }
    let converted: Cow<'_, [_]> = match values.data_type() {
        DataType::Float16 => values
            .as_primitive::<Float16Type>()
            .values()
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>()
            .into(),
        DataType::Float32 => Cow::Borrowed(values.as_primitive::<Float32Type>().values().as_ref()),
        DataType::Float64 => values
            .as_primitive::<Float64Type>()
            .values()
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>()
            .into(),
        data_type => {
            return Err(Error::invalid_input(format!(
                "PCA input must be Float16, Float32, or Float64, got {data_type}"
            )));
        }
    };
    if converted.iter().any(|value| !value.is_finite()) {
        return Err(Error::invalid_input(
            "PCA input contains non-finite values or values outside the Float32 range".to_string(),
        ));
    }
    Ok(converted)
}

fn compensated_mean<T: Copy + Into<f64>>(
    values: &[T],
    rows: usize,
    dimension: usize,
) -> Result<Vec<f64>> {
    let mut sums = vec![0.0; dimension];
    let mut corrections = vec![0.0; dimension];
    for row in values.chunks_exact(dimension) {
        for (column, value) in row.iter().enumerate() {
            let value = (*value).into();
            let adjusted = value - corrections[column];
            let next = sums[column] + adjusted;
            corrections[column] = (next - sums[column]) - adjusted;
            sums[column] = next;
        }
    }
    let denominator = rows as f64;
    for sum in &mut sums {
        *sum /= denominator;
        if !sum.is_finite() {
            return Err(Error::invalid_input(
                "PCA mean calculation produced a non-finite value".to_string(),
            ));
        }
    }
    Ok(sums)
}

/// Accumulates a centered scatter matrix from one row-major batch.
///
/// `values` contains complete observations with `mean.len()` values per row.
/// `scatter` is a full row-major `mean.len() * mean.len()` matrix and is
/// updated in place with `(values - mean)^T * (values - mean)`.
pub fn accumulate_centered_scatter(
    values: &[f64],
    mean: &[f64],
    scatter: &mut [f64],
) -> Result<()> {
    let dimension = mean.len();
    if dimension == 0 {
        return Err(Error::invalid_input(
            "PCA scatter accumulation requires a non-empty mean".to_string(),
        ));
    }
    if values.is_empty() || !values.len().is_multiple_of(dimension) {
        return Err(Error::invalid_input(format!(
            "PCA scatter batch values length={} must be a non-zero multiple of dimension={dimension}",
            values.len()
        )));
    }
    validate_finite_slice(values, "scatter batch")?;
    validate_finite_slice(mean, "scatter mean")?;
    validate_symmetric_matrix(scatter, dimension, "accumulated scatter")?;

    accumulate_scatter(values, mean, scatter)
}

fn accumulate_scatter<T: Copy + Into<f64>>(
    values: &[T],
    mean: &[f64],
    scatter: &mut [f64],
) -> Result<()> {
    let dimension = mean.len();
    let rows = values.len() / dimension;
    let centered = Array2::from_shape_fn((rows, dimension), |(row, column)| {
        values[row * dimension + column].into() - mean[column]
    });
    let batch_scatter = centered.t().dot(&centered);
    for row in 0..dimension {
        for column in row..dimension {
            let value = scatter[row * dimension + column] + batch_scatter[[row, column]];
            if !value.is_finite() {
                return Err(Error::invalid_input(format!(
                    "PCA scatter accumulation produced non-finite value={value} at row={row}, column={column}, shape=[{dimension}, {dimension}]"
                )));
            }
            scatter[row * dimension + column] = value;
            scatter[column * dimension + row] = value;
        }
    }
    Ok(())
}

fn new_scatter(dimension: usize) -> Result<Vec<f64>> {
    let matrix_len = dimension
        .checked_mul(dimension)
        .ok_or_else(|| Error::invalid_input("PCA scatter matrix shape overflows usize"))?;
    Layout::array::<f64>(matrix_len).map_err(|_| {
        Error::invalid_input(format!(
            "PCA scatter shape [{dimension}, {dimension}] exceeds the maximum allocation size"
        ))
    })?;
    Ok(vec![0.0; matrix_len])
}

fn principal_components(
    scatter: &[f64],
    dimension: usize,
    sample_count: usize,
    output_dimension: usize,
) -> Result<Float32Array> {
    // Normalize while creating the solver-owned matrix: raw scatter eigenvalues
    // can overflow even when the covariance eigenvalues are finite.
    let dense = nalgebra::DMatrix::from_fn(dimension, dimension, |row, column| {
        scatter[row * dimension + column] / sample_count as f64
    });
    // Bound solver work so a non-converging matrix fails instead of looping forever.
    let max_iterations = dimension
        .saturating_mul(dimension)
        .saturating_mul(15)
        .max(32);
    let tolerance = f64::EPSILON * (dimension as f64);
    let decomposition = nalgebra::SymmetricEigen::try_new(dense, tolerance, max_iterations)
        .ok_or_else(|| {
            Error::index(format!(
                "PCA eigendecomposition failed to converge within {max_iterations} iterations"
            ))
        })?;
    let mut order: Vec<usize> = (0..dimension).collect();
    order.sort_by(|&left, &right| {
        decomposition.eigenvalues[right].total_cmp(&decomposition.eigenvalues[left])
    });
    let eigenvalues = order
        .iter()
        .map(|&index| decomposition.eigenvalues[index])
        .collect::<Vec<_>>();
    validate_eigenvalues(&eigenvalues, output_dimension)?;
    let mut basis = Vec::with_capacity(output_dimension * dimension);
    for component in order.into_iter().take(output_dimension) {
        for row in 0..dimension {
            basis.push(decomposition.eigenvectors[(row, component)]);
        }
    }
    normalize_component_signs(&mut basis, output_dimension, dimension)?;
    Ok(Float32Array::from(
        basis
            .into_iter()
            .map(canonical_f32)
            .collect::<Result<Vec<_>>>()?,
    ))
}

fn validate_eigenvalues(eigenvalues: &[f64], output_dimension: usize) -> Result<()> {
    let Some(&lambda_max) = eigenvalues.first() else {
        return Err(Error::index(
            "PCA eigendecomposition returned no eigenvalues",
        ));
    };
    let tau = f64::MIN_POSITIVE.max(1.0e-10 * lambda_max.abs());
    let mut effective_rank = 0;
    for (index, &eigenvalue) in eigenvalues.iter().enumerate() {
        if !eigenvalue.is_finite() {
            return Err(Error::index(format!(
                "PCA eigenvalue at descending index {index} is non-finite"
            )));
        }
        if eigenvalue < -tau {
            return Err(Error::index(format!(
                "PCA eigenvalue at descending index {index} is {eigenvalue}, below -tau={}",
                -tau,
            )));
        }
        if eigenvalue > tau {
            effective_rank += 1;
        }
    }
    if effective_rank < output_dimension {
        return Err(Error::invalid_input(format!(
            "PCA training data has effective_rank={effective_rank}, smaller than output_dimension={output_dimension}; lambda_max={lambda_max}, tau={tau}"
        )));
    }
    Ok(())
}

fn normalize_vector(vector: &mut [f64], label: &str) -> Result<()> {
    let norm_squared: f64 = vector.iter().map(|value| value * value).sum();
    let norm = norm_squared.sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return Err(Error::index(format!(
            "PCA {label} has invalid L2 norm={norm}"
        )));
    }
    for value in vector {
        *value /= norm;
    }
    Ok(())
}

fn normalize_component_signs(basis: &mut [f64], components: usize, dimension: usize) -> Result<()> {
    for component in 0..components {
        let vector = &mut basis[component * dimension..(component + 1) * dimension];
        normalize_vector(vector, "eigenvector")?;
        canonicalize_sign(vector)?;
    }
    Ok(())
}

fn canonicalize_sign(vector: &mut [f64]) -> Result<()> {
    let Some((pivot, _)) =
        vector
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.abs()
                    .total_cmp(&right.abs())
                    .then_with(|| right_index.cmp(left_index))
            })
    else {
        return Err(Error::index(
            "PCA cannot canonicalize an empty vector".to_string(),
        ));
    };
    if vector[pivot] < 0.0 {
        for value in vector {
            *value = -*value;
        }
    }
    Ok(())
}

fn canonical_f32(value: f64) -> Result<f32> {
    let value = value as f32;
    if !value.is_finite() {
        return Err(Error::index(
            "PCA model conversion to Float32 produced a non-finite value".to_string(),
        ));
    }
    Ok(if value == 0.0 { 0.0 } else { value })
}

// Sharing input loads across four rows beats independent dot products on ARM.
#[cfg(target_arch = "aarch64")]
#[inline]
fn dot_4rows(centered: &[f32], rows: &[f32]) -> [f32; 4] {
    let n = centered.len();
    let stride = n;
    debug_assert_eq!(rows.len(), 4 * n);
    let full = n & !7;
    // SAFETY: rows contains four complete n-element rows; each vector load
    // stays within its row and the centered input. NEON is required on aarch64.
    let (mut out, mut i) = unsafe {
        let cp = centered.as_ptr();
        let m0 = rows.as_ptr();
        let m1 = m0.add(stride);
        let m2 = m1.add(stride);
        let m3 = m2.add(stride);
        let zero = vdupq_n_f32(0.0);
        let (mut a0, mut b0, mut a1, mut b1) = (zero, zero, zero, zero);
        let (mut a2, mut b2, mut a3, mut b3) = (zero, zero, zero, zero);
        let mut i = 0;
        while i < full {
            let c0 = vld1q_f32(cp.add(i));
            let c1 = vld1q_f32(cp.add(i + 4));
            a0 = vfmaq_f32(a0, vld1q_f32(m0.add(i)), c0);
            b0 = vfmaq_f32(b0, vld1q_f32(m0.add(i + 4)), c1);
            a1 = vfmaq_f32(a1, vld1q_f32(m1.add(i)), c0);
            b1 = vfmaq_f32(b1, vld1q_f32(m1.add(i + 4)), c1);
            a2 = vfmaq_f32(a2, vld1q_f32(m2.add(i)), c0);
            b2 = vfmaq_f32(b2, vld1q_f32(m2.add(i + 4)), c1);
            a3 = vfmaq_f32(a3, vld1q_f32(m3.add(i)), c0);
            b3 = vfmaq_f32(b3, vld1q_f32(m3.add(i + 4)), c1);
            i += 8;
        }
        (
            [
                vaddvq_f32(vaddq_f32(a0, b0)),
                vaddvq_f32(vaddq_f32(a1, b1)),
                vaddvq_f32(vaddq_f32(a2, b2)),
                vaddvq_f32(vaddq_f32(a3, b3)),
            ],
            i,
        )
    };
    while i < n {
        let c = centered[i];
        out[0] += rows[i] * c;
        out[1] += rows[stride + i] * c;
        out[2] += rows[2 * stride + i] * c;
        out[3] += rows[3 * stride + i] * c;
        i += 1;
    }
    out
}

// Handle the remaining rows with the existing ARM row kernel.
#[cfg(target_arch = "aarch64")]
#[inline]
fn dot_row(centered: &[f32], matrix_row: &[f32]) -> f32 {
    let n = centered.len();
    debug_assert_eq!(matrix_row.len(), n);
    let full = n & !15;
    // SAFETY: both slices contain n elements and every vector load ends
    // before full <= n. NEON is required on aarch64.
    let (mut value, mut i) = unsafe {
        let cp = centered.as_ptr();
        let mp = matrix_row.as_ptr();
        let zero = vdupq_n_f32(0.0);
        let (mut a0, mut a1, mut a2, mut a3) = (zero, zero, zero, zero);
        let mut i = 0;
        while i < full {
            a0 = vfmaq_f32(a0, vld1q_f32(mp.add(i)), vld1q_f32(cp.add(i)));
            a1 = vfmaq_f32(a1, vld1q_f32(mp.add(i + 4)), vld1q_f32(cp.add(i + 4)));
            a2 = vfmaq_f32(a2, vld1q_f32(mp.add(i + 8)), vld1q_f32(cp.add(i + 8)));
            a3 = vfmaq_f32(a3, vld1q_f32(mp.add(i + 12)), vld1q_f32(cp.add(i + 12)));
            i += 16;
        }
        (
            vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3))),
            i,
        )
    };
    while i < n {
        value += matrix_row[i] * centered[i];
        i += 1;
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;
    use arrow::compute::cast;
    use arrow_array::{Float16Array, Float64Array};
    use half::f16;
    use rstest::rstest;

    fn persisted_model(matrix: Vec<f32>, mean: Vec<f32>) -> PcaModel {
        PcaModel {
            input_dimension: mean.len(),
            output_dimension: matrix.len() / mean.len(),
            transform_matrix: Float32Array::from(matrix),
            mean: Float32Array::from(mean),
            fingerprint: OnceLock::new(),
        }
    }

    fn valid_pb_model() -> pb::PcaModel {
        pb::PcaModel::try_from(&persisted_model(
            vec![1.0, 2.0, 3.0, -1.0, 0.5, 4.0],
            vec![0.25, -0.5, 2.0],
        ))
        .expect("valid model must encode")
    }

    fn assert_corrupt(error: Error, context: &str) {
        assert!(
            matches!(error, Error::CorruptFile { .. }),
            "expected CorruptFile, got {error:?}"
        );
        assert!(
            error.to_string().contains(context),
            "error {error} must contain {context:?}"
        );
    }

    fn f32_vectors(values: Vec<f32>, dimension: i32) -> FixedSizeListArray {
        FixedSizeListArray::try_new_from_values(Float32Array::from(values), dimension)
            .expect("test vectors must be valid")
    }

    fn full_rank_values(rows: usize, dimension: usize) -> Vec<f32> {
        let mut values = Vec::with_capacity(rows * dimension);
        for row in 0..rows {
            for column in 0..dimension {
                let value =
                    ((row + 1) as f32).powi((column + 1) as i32) + (row * column) as f32 * 0.17;
                values.push(value);
            }
        }
        values
    }

    #[test]
    fn covariance_path_centers_and_reduces() {
        let vectors = f32_vectors(full_rank_values(8, 3), 3);
        let model = PcaModel::train(&vectors, &PcaParams::new(2)).expect("training should work");
        let output = model.transform(&vectors).expect("transform should work");
        assert_eq!(output.len(), 8);
        assert_eq!(output.value_length(), 2);
        assert_eq!(output.value_type(), DataType::Float32);
        let values = output.values().as_primitive::<Float32Type>();
        for column in 0..2 {
            let mean: f32 = (0..8)
                .map(|row| values.value(row * 2 + column))
                .sum::<f32>()
                / 8.0;
            assert!(mean.abs() < 1.0e-3);
        }
    }

    #[test]
    fn covariance_query_matches_batch_when_samples_are_fewer_than_dimensions() {
        let vectors = f32_vectors(full_rank_values(4, 6), 6);
        let model = PcaModel::train(&vectors, &PcaParams::new(2)).expect("training should work");
        let batch = model.transform(&vectors).expect("transform should work");
        let query = model
            .transform_query(vectors.value(2).as_ref())
            .expect("query transform should work");
        let query = query.as_primitive::<Float32Type>();
        let batch_row = batch.value(2);
        let batch_row = batch_row.as_primitive::<Float32Type>();
        for index in 0..2 {
            assert_relative_eq!(query.value(index), batch_row.value(index), epsilon = 1.0e-5);
        }
    }

    #[test]
    fn scatter_statistics_match_in_memory_training() {
        let rows = 8;
        let dimension = 3;
        let values = full_rank_values(rows, dimension);
        let vectors = f32_vectors(values.clone(), dimension as i32);
        let values = values.into_iter().map(f64::from).collect::<Vec<_>>();
        let mean =
            compensated_mean(&values, rows, dimension).expect("mean calculation should work");
        let mut scatter = vec![0.0; dimension * dimension];
        accumulate_centered_scatter(&values, &mean, &mut scatter)
            .expect("scatter accumulation should work");
        let params = PcaParams::new(2);

        let in_memory = PcaModel::train(&vectors, &params).expect("array training should work");
        let from_scatter = PcaModel::train_from_scatter(&mean, rows, &scatter, &params)
            .expect("scatter training should work");

        assert_eq!(from_scatter.fingerprint(), in_memory.fingerprint());
        assert_eq!(
            from_scatter
                .transform(&vectors)
                .expect("scatter model transform should work"),
            in_memory
                .transform(&vectors)
                .expect("in-memory model transform should work")
        );
    }

    #[rstest]
    #[case::small_batches(17)]
    #[case::training_batches(TRANSFORM_BATCH_SIZE)]
    fn scatter_training_matches_across_batch_boundaries(
        #[case] batch_rows: usize,
        #[values(DataType::Float16, DataType::Float32, DataType::Float64)] data_type: DataType,
    ) {
        let rows = TRANSFORM_BATCH_SIZE + 3;
        let values: Vec<f64> = (0..rows + 2)
            .flat_map(|row| {
                if row == 0 || row == rows + 1 {
                    return [f64::NAN; 3];
                }
                let phase = row as f64 * 0.037;
                [
                    phase.sin(),
                    0.5 * (2.0 * phase).cos(),
                    0.1 * (3.0 * phase).sin(),
                ]
            })
            .collect();
        let values = cast(&Float64Array::from(values), &data_type).unwrap();
        let vectors = FixedSizeListArray::try_new_from_values(values, 3)
            .unwrap()
            .slice(1, rows);
        let values = cast(vectors.values().as_ref(), &DataType::Float64).unwrap();
        let values = values.as_primitive::<Float64Type>().values();
        let mean = compensated_mean(values, rows, 3).unwrap();
        let mut scatter = new_scatter(3).unwrap();
        for batch in values.chunks(batch_rows * 3) {
            accumulate_centered_scatter(batch, &mean, &mut scatter).unwrap();
        }
        let params = PcaParams::new(2);
        let from_array = PcaModel::train(&vectors, &params).unwrap();
        let from_scatter = PcaModel::train_from_scatter(&mean, rows, &scatter, &params).unwrap();
        let expected = from_array.transform(&vectors.slice(0, 8)).unwrap();
        let actual = from_scatter.transform(&vectors.slice(0, 8)).unwrap();
        for (actual, expected) in actual
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .zip(expected.values().as_primitive::<Float32Type>().values())
        {
            assert_relative_eq!(actual, expected, epsilon = 1.0e-5);
        }
    }

    #[test]
    fn scatter_training_accepts_finite_covariance_with_large_scatter() {
        let model =
            PcaModel::train_from_scatter(&[0.0; 3], 4, &[f64::MAX * 0.5; 9], &PcaParams::new(1))
                .expect("finite covariance must not overflow when decomposing scatter");
        let expected = 1.0 / 3.0_f32.sqrt();
        for value in model.transform_matrix().values() {
            assert_relative_eq!(*value, expected, epsilon = 1.0e-6);
        }
    }

    #[test]
    fn transform_accepts_an_empty_batch() {
        let model = persisted_model(vec![1.0, 0.0, 0.0], vec![0.0; 3]);
        let result = model.transform(&f32_vectors(Vec::new(), 3)).unwrap();
        assert_eq!(result.len(), 0);
        assert_eq!(result.value_length(), 1);
        assert_eq!(result.value_type(), DataType::Float32);
    }

    #[test]
    fn statistics_apis_validate_shapes_and_values() {
        let params = PcaParams::new(1);
        let error = PcaModel::train_from_scatter(&[0.0, 0.0], 3, &[1.0, 0.0, 0.0], &params)
            .expect_err("incorrect scatter length must fail");
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("requiring 4 values"));

        let error = PcaModel::train_from_scatter(&[0.0, 0.0], 3, &[1.0, 2.0, 0.0, 1.0], &params)
            .expect_err("non-symmetric scatter must fail");
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("must be symmetric"));

        let error =
            PcaModel::train_from_scatter(&[0.0, f64::NAN], 3, &[1.0, 0.0, 0.0, 1.0], &params)
                .expect_err("non-finite mean must fail");
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("mean"));
        assert!(error.to_string().contains("element=1"));
    }

    #[test]
    fn centered_scatter_accumulates_matrix_batches() {
        let mean = [2.0, 1.0, -1.0];
        let mut scatter = vec![0.0; 9];
        accumulate_centered_scatter(&[1.0, 2.0, 0.0, 3.0, 0.0, -2.0], &mean, &mut scatter)
            .expect("first scatter batch should accumulate");
        accumulate_centered_scatter(&[4.0, 2.0, 1.0], &mean, &mut scatter)
            .expect("second scatter batch should accumulate");
        assert_eq!(scatter, vec![6.0, 0.0, 2.0, 0.0, 3.0, 4.0, 2.0, 4.0, 6.0]);

        let error = accumulate_centered_scatter(&[1.0, 2.0], &mean, &mut scatter)
            .expect_err("incomplete scatter rows must fail");
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("non-zero multiple"));
    }

    #[test]
    fn covariance_path_selects_largest_axes_in_descending_order() {
        let vectors = f32_vectors(
            vec![
                3.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                0.0, -1.0,
            ],
            3,
        );
        let model = PcaModel::train(&vectors, &PcaParams::new(2)).expect("training should work");
        let matrix = model.transform_matrix().values();
        assert_relative_eq!(matrix[0], 1.0, epsilon = 1.0e-6);
        assert_relative_eq!(matrix[1], 0.0, epsilon = 1.0e-6);
        assert_relative_eq!(matrix[2], 0.0, epsilon = 1.0e-6);
        assert_relative_eq!(matrix[3], 0.0, epsilon = 1.0e-6);
        assert_relative_eq!(matrix[4], 1.0, epsilon = 1.0e-6);
        assert_relative_eq!(matrix[5], 0.0, epsilon = 1.0e-6);
    }

    #[test]
    fn covariance_path_selects_axes_when_samples_are_fewer_than_dimensions() {
        let vectors = f32_vectors(
            vec![
                3.0, 0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                -1.0, 0.0, 0.0, 0.0,
            ],
            5,
        );
        let model = PcaModel::train(&vectors, &PcaParams::new(2)).expect("training should work");
        let matrix = model.transform_matrix().values();
        assert_relative_eq!(matrix[0], 1.0, epsilon = 1.0e-6);
        assert_relative_eq!(matrix[1], 0.0, epsilon = 1.0e-6);
        assert_relative_eq!(matrix[5], 0.0, epsilon = 1.0e-6);
        assert_relative_eq!(matrix[6], 1.0, epsilon = 1.0e-6);
    }

    #[test]
    fn accepts_all_supported_float_types() {
        let values = full_rank_values(6, 3);
        let f16_values = values
            .iter()
            .map(|value| f16::from_f32(*value))
            .collect::<Vec<_>>();
        let f64_values = values
            .iter()
            .map(|value| f64::from(*value))
            .collect::<Vec<_>>();
        let f16_vectors =
            FixedSizeListArray::try_new_from_values(Float16Array::from(f16_values), 3)
                .expect("f16 vectors should be valid");
        let f64_vectors =
            FixedSizeListArray::try_new_from_values(Float64Array::from(f64_values), 3)
                .expect("f64 vectors should be valid");
        let f32_vectors = f32_vectors(values, 3);
        for vectors in [&f16_vectors, &f32_vectors, &f64_vectors] {
            let model = PcaModel::train(vectors, &PcaParams::new(2)).expect("training should work");
            let batch = model.transform(vectors).expect("transform should work");
            assert_eq!(batch.value_type(), DataType::Float32);
            for row in 0..vectors.len() {
                let query = model.transform_query(vectors.value(row).as_ref()).unwrap();
                let expected = batch.value(row);
                for (actual, expected) in query
                    .as_primitive::<Float32Type>()
                    .values()
                    .iter()
                    .zip(expected.as_primitive::<Float32Type>().values())
                {
                    assert_relative_eq!(actual, expected, epsilon = 1.0e-5);
                }
            }
        }
    }

    #[test]
    fn sliced_arrays_use_only_logical_values() {
        let all_vectors = f32_vectors(full_rank_values(10, 3), 3);
        let sliced = all_vectors.slice(2, 6);
        let sliced = sliced
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .expect("slice must preserve fixed-size-list type");
        let model =
            PcaModel::train(sliced, &PcaParams::new(2)).expect("sliced training should work");
        let transformed = model
            .transform(sliced)
            .expect("sliced transform should work");
        assert_eq!(transformed.len(), 6);

        let first_query = model
            .transform_query(sliced.value(0).as_ref())
            .expect("query transform should work");
        let first_batch = transformed.value(0);
        let first_query = first_query.as_primitive::<Float32Type>();
        let first_batch = first_batch.as_primitive::<Float32Type>();
        for index in 0..2 {
            assert_relative_eq!(
                first_query.value(index),
                first_batch.value(index),
                epsilon = 1.0e-5
            );
        }
    }

    #[test]
    fn rejects_invalid_shape_null_and_non_finite_inputs() {
        let vectors = f32_vectors(full_rank_values(2, 3), 3);
        let error = PcaModel::train(&vectors, &PcaParams::new(2)).expect_err("n=d_out must fail");
        assert!(error.to_string().contains("at least output_dimension+1=3"));

        let non_finite = f32_vectors(vec![1.0, 2.0, 3.0, f32::NAN, 5.0, 6.0, 7.0, 8.0], 2);
        let error = PcaModel::train(&non_finite, &PcaParams::new(1))
            .expect_err("non-finite training input must fail");
        assert!(error.to_string().contains("non-finite"));

        let nullable = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![Some(1.0), None, Some(3.0), Some(4.0)]),
            2,
        )
        .expect("nullable vectors should construct");
        let error = PcaModel::train(&nullable, &PcaParams::new(1))
            .expect_err("null training input must fail");
        assert!(error.to_string().contains("null"));
    }

    #[rstest]
    #[case::shape_overflow(usize::MAX, "shape overflows")]
    #[case::allocation_overflow(1 << (usize::BITS / 2 - 1), "maximum allocation size")]
    fn rejects_scatter_allocation_overflow(#[case] dimension: usize, #[case] message: &str) {
        let error = new_scatter(dimension)
            .expect_err("unrepresentable scatter allocation must fail before allocating");
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(message));
    }

    #[test]
    fn rejects_low_rank_training_data() {
        let low_rank = f32_vectors(
            vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0],
            3,
        );
        let error =
            PcaModel::train(&low_rank, &PcaParams::new(2)).expect_err("low-rank input must fail");
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("effective_rank=1"));
    }

    #[test]
    fn eigenvalue_threshold_handles_zero_and_negative_values() {
        validate_eigenvalues(&[4.0, 1.0e-12, -1.0e-12], 1)
            .expect("near-zero eigenvalues should be tolerated");
        let error = validate_eigenvalues(&[4.0, 1.0e-12, -1.0e-12], 2)
            .expect_err("near-zero eigenvalues must not contribute to rank");
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("effective_rank=1"));
        let error = validate_eigenvalues(&[4.0, -1.0], 1)
            .expect_err("significant negative eigenvalue must fail");
        assert!(matches!(error, Error::Index { .. }));
        assert!(error.to_string().contains("below -tau"));
    }

    #[test]
    fn query_rejects_wrong_shape_and_non_finite_values() {
        let vectors = f32_vectors(full_rank_values(8, 3), 3);
        let model = PcaModel::train(&vectors, &PcaParams::new(2)).expect("training should work");
        let wrong_shape = Float32Array::from(vec![1.0, 2.0]);
        let error = model
            .transform_query(&wrong_shape)
            .expect_err("wrong query shape must fail");
        assert!(
            error
                .to_string()
                .contains("does not match input_dimension=3")
        );

        let non_finite = Float32Array::from(vec![1.0, f32::INFINITY, 3.0]);
        let error = model
            .transform_query(&non_finite)
            .expect_err("non-finite query must fail");
        assert!(error.to_string().contains("non-finite"));
    }

    #[test]
    fn sign_normalization_uses_first_maximum() {
        let mut vector = vec![-2.0, 2.0, 1.0];
        canonicalize_sign(&mut vector).expect("sign normalization should work");
        assert_eq!(vector, vec![2.0, -2.0, -1.0]);
    }

    #[test]
    fn protobuf_bytes_are_deterministic() {
        let model = persisted_model(vec![1.0, 2.0, 3.0, -1.0, 0.5, 4.0], vec![0.25, -0.5, 2.0]);
        let actual = encode_pca_model(&model).expect("encoding must succeed");
        let mut matrix_data = Vec::new();
        for value in [1.0_f32, 2.0, 3.0, -1.0, 0.5, 4.0] {
            matrix_data.extend_from_slice(&value.to_le_bytes());
        }
        let mut mean_data = Vec::new();
        for value in [0.25_f32, -0.5, 2.0] {
            mean_data.extend_from_slice(&value.to_le_bytes());
        }
        let mut expected = vec![
            0x08, 0x01, 0x12, 0x20, 0x08, 0x02, 0x12, 0x02, 0x02, 0x03, 0x1a, 0x18,
        ];
        expected.extend(matrix_data);
        expected.extend([0x1a, 0x13, 0x08, 0x02, 0x12, 0x01, 0x03, 0x1a, 0x0c]);
        expected.extend(mean_data);
        assert_eq!(actual, expected);
        assert_eq!(
            encode_pca_model(&model).expect("repeated encoding must succeed"),
            expected
        );
    }

    #[test]
    fn persisted_model_round_trip_preserves_transform() {
        let model = persisted_model(vec![1.0, 0.0, 0.5, 0.0, 1.0, -0.25], vec![2.0, 3.0, 4.0]);
        let vectors = f32_vectors(vec![3.0, 5.0, 8.0, 1.0, 3.0, 0.0], 3);
        let expected = model.transform(&vectors).expect("transform must succeed");
        let decoded = decode_pca_model(&encode_pca_model(&model).expect("encode must succeed"))
            .expect("decode must succeed");
        assert_eq!(decoded.transform_matrix(), model.transform_matrix());
        assert_eq!(decoded.mean(), model.mean());
        assert_eq!(
            decoded
                .transform(&vectors)
                .expect("decoded transform must succeed"),
            expected
        );
    }

    #[test]
    fn fingerprint_matches_independent_blake3_input() {
        let model = persisted_model(vec![-0.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]);
        assert!(model.fingerprint.get().is_none());
        let mut input = PCA_FINGERPRINT_DOMAIN.to_vec();
        input.extend_from_slice(&1_u32.to_le_bytes());
        for value in [0.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] {
            input.extend_from_slice(&value.to_le_bytes());
        }
        let expected: [u8; 32] = *blake3::hash(&input).as_bytes();
        assert_eq!(model.fingerprint(), expected);
        assert_eq!(model.fingerprint.get(), Some(&expected));
        assert_eq!(model.fingerprint(), expected);

        let positive_zero =
            persisted_model(vec![0.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]);
        assert_eq!(model.fingerprint(), positive_zero.fingerprint());
        let changed_matrix =
            persisted_model(vec![0.0, 2.5, 3.0, 4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]);
        let changed_mean = persisted_model(vec![0.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![7.0, 8.0, 9.5]);
        assert_ne!(model.fingerprint(), changed_matrix.fingerprint());
        assert_ne!(model.fingerprint(), changed_mean.fingerprint());
    }

    #[rstest]
    #[case(0)]
    #[case(2)]
    fn rejects_unsupported_persisted_versions(#[case] version: u32) {
        let mut model = valid_pb_model();
        model.model_version = version;
        assert_corrupt(
            PcaModel::try_from(model).expect_err("unsupported version must fail"),
            "model_version",
        );
    }

    #[test]
    fn rejects_missing_tensors() {
        let mut missing_matrix = valid_pb_model();
        missing_matrix.transform_matrix = None;
        assert_corrupt(
            PcaModel::try_from(missing_matrix).expect_err("missing matrix must fail"),
            "missing transform_matrix",
        );
        let mut missing_mean = valid_pb_model();
        missing_mean.mean = None;
        assert_corrupt(
            PcaModel::try_from(missing_mean).expect_err("missing mean must fail"),
            "missing mean",
        );
    }

    #[test]
    fn rejects_invalid_tensor_type_rank_and_dimensions() {
        let mut wrong_type = valid_pb_model();
        wrong_type
            .transform_matrix
            .as_mut()
            .expect("matrix exists")
            .data_type = pb::tensor::DataType::Float64 as i32;
        assert_corrupt(
            PcaModel::try_from(wrong_type).expect_err("wrong dtype must fail"),
            "FLOAT32",
        );

        let mut wrong_rank = valid_pb_model();
        wrong_rank
            .transform_matrix
            .as_mut()
            .expect("matrix exists")
            .shape = vec![6];
        assert_corrupt(
            PcaModel::try_from(wrong_rank).expect_err("wrong rank must fail"),
            "rank=1",
        );

        let mut zero_dimension = valid_pb_model();
        zero_dimension
            .transform_matrix
            .as_mut()
            .expect("matrix exists")
            .shape = vec![0, 3];
        assert_corrupt(
            PcaModel::try_from(zero_dimension).expect_err("zero dimension must fail"),
            "zero dimension",
        );

        let mut unreduced = valid_pb_model();
        unreduced
            .transform_matrix
            .as_mut()
            .expect("matrix exists")
            .shape = vec![3, 2];
        assert_corrupt(
            PcaModel::try_from(unreduced).expect_err("d_out >= d_in must fail"),
            "must be smaller",
        );

        let mut wrong_mean_shape = valid_pb_model();
        wrong_mean_shape.mean.as_mut().expect("mean exists").shape = vec![2];
        wrong_mean_shape
            .mean
            .as_mut()
            .expect("mean exists")
            .data
            .truncate(8);
        assert_corrupt(
            PcaModel::try_from(wrong_mean_shape).expect_err("mean mismatch must fail"),
            "do not match input_dimension",
        );
    }

    #[rstest]
    #[case(true)]
    #[case(false)]
    fn rejects_incorrect_tensor_data_lengths(#[case] is_short: bool) {
        let mut model = valid_pb_model();
        let data = &mut model.transform_matrix.as_mut().expect("matrix exists").data;
        if is_short {
            data.pop();
        } else {
            data.push(0);
        }
        assert_corrupt(
            PcaModel::try_from(model).expect_err("incorrect length must fail"),
            "data length",
        );
    }

    #[rstest]
    #[case(f32::NAN)]
    #[case(f32::INFINITY)]
    fn rejects_non_finite_persisted_values(#[case] value: f32) {
        let mut model = valid_pb_model();
        model
            .mean
            .as_mut()
            .expect("mean exists")
            .data
            .splice(0..4, value.to_le_bytes());
        assert_corrupt(
            PcaModel::try_from(model).expect_err("non-finite value must fail"),
            "NaN or infinity",
        );
    }

    #[test]
    fn rejects_tensor_dimension_multiplication_overflow() {
        let tensor = pb::Tensor {
            data_type: pb::tensor::DataType::Float32 as i32,
            shape: vec![u32::MAX, u32::MAX, u32::MAX],
            data: Vec::new(),
        };
        assert_corrupt(
            decode_f32_tensor(&tensor, 3, "overflow_tensor", PCA_MODEL_VERSION)
                .expect_err("dimension multiplication must overflow"),
            "overflows",
        );
    }

    #[rstest]
    #[case::small(2, 1)]
    #[case::tail(5, 3)]
    #[case::aligned(16, 4)]
    #[case::unaligned(23, 5)]
    #[case::medium(64, 7)]
    #[case::wide(960, 9)]
    #[case::embedding(960, 256)]
    fn query_projection_matches_scalar_reference(
        #[case] input_dimension: usize,
        #[case] output_dimension: usize,
    ) {
        let input: Vec<f32> = (0..input_dimension)
            .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.25)
            .collect();
        let mean: Vec<f32> = (0..input_dimension)
            .map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.5)
            .collect();
        let matrix: Vec<f32> = (0..input_dimension * output_dimension)
            .map(|i| ((i * 11 % 17) as f32 - 8.0) * 0.125)
            .collect();
        let expected: Vec<f64> = matrix
            .chunks_exact(input_dimension)
            .map(|row| {
                row.iter()
                    .zip(input.iter().zip(&mean))
                    .map(|(coefficient, (value, mean))| {
                        f64::from(*coefficient) * f64::from(value - mean)
                    })
                    .sum()
            })
            .collect();
        let model = persisted_model(matrix, mean);
        let actual = model.transform_query(&Float32Array::from(input)).unwrap();
        let actual = actual.as_primitive::<Float32Type>();
        assert_eq!(actual.len(), output_dimension);
        for (actual, expected) in actual.values().iter().zip(expected) {
            assert_relative_eq!(
                f64::from(*actual),
                expected,
                epsilon = 1.0e-4,
                max_relative = 1.0e-4
            );
        }
    }

    #[rstest]
    #[case::single_row(1)]
    #[case::four_rows(4)]
    #[case::four_rows_and_tail(5)]
    fn query_projection_rejects_non_finite_output(#[case] output_dimension: usize) {
        let model = persisted_model(vec![1.0; output_dimension * 6], vec![0.0; 6]);
        let error = model
            .transform_query(&Float32Array::from(vec![f32::MAX; 6]))
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("non-finite value at output dimension 0")
        );
    }
}
