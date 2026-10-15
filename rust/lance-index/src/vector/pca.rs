// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Principal component analysis for vector pre-transforms.

use std::f64::consts::TAU;
use std::mem::size_of;
use std::sync::{Arc, LazyLock, OnceLock};

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array,
    cast::AsArray,
    types::{Float16Type, Float32Type, Float64Type},
};
use arrow_schema::DataType;
use bytes::Bytes;
use faer::{Mat, Side, linalg::solvers::SelfAdjointEigen};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::{Error, Result};
use prost::Message;
use rand::rngs::SmallRng;
use rand::{RngCore, SeedableRng};

use crate::pb;
use crate::scalar::{IndexReader, IndexWriter};

/// Arrow schema metadata key containing the PCA model global-buffer index.
pub const PCA_METADATA_KEY: &str = "lance:pca";

/// Current stable PCA model serialization version.
pub const PCA_MODEL_VERSION: u32 = 1;

/// A PCA projection that produced a non-finite output component.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcaNonFiniteOutput {
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
pub fn non_finite_output(error: &Error) -> Option<PcaNonFiniteOutput> {
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
const DEFAULT_MAX_TRAINING_MEMORY_BYTES: usize = 1_073_741_824;
const DEFAULT_SAMPLING_SEED: u64 = 42;
const DEFAULT_ROTATION_SEED: u64 = 5;
const DEFAULT_EPSILON: f32 = 1.0e-5;
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
/// let params = PcaParams::new(64)
///     .with_whitening(true)
///     .with_random_rotation(false);
/// assert_eq!(params.dimension, 64);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PcaParams {
    /// Number of dimensions produced by the transform.
    pub dimension: usize,
    /// Whether to apply a deterministic orthogonal rotation after PCA.
    pub random_rotation: bool,
    /// Whether to scale retained components by their regularized variance.
    pub whitening: bool,
    /// Regularizer used by whitening.
    ///
    /// This must be finite and positive when whitening is enabled. Otherwise,
    /// it may be zero but must remain finite and non-negative.
    pub epsilon: f32,
    /// Target number of valid rows used by dataset-aware training.
    pub sample_size: usize,
    /// Maximum estimated peak bytes permitted for training.
    pub max_training_memory_bytes: usize,
    /// Seed used by dataset-aware sampling.
    pub sampling_seed: u64,
    /// Seed used to construct the deterministic random rotation.
    pub rotation_seed: u64,
}

impl PcaParams {
    /// Creates PCA parameters with the documented defaults.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            random_rotation: false,
            whitening: false,
            epsilon: DEFAULT_EPSILON,
            sample_size: DEFAULT_SAMPLE_SIZE,
            max_training_memory_bytes: DEFAULT_MAX_TRAINING_MEMORY_BYTES,
            sampling_seed: DEFAULT_SAMPLING_SEED,
            rotation_seed: DEFAULT_ROTATION_SEED,
        }
    }

    pub fn with_random_rotation(mut self, enabled: bool) -> Self {
        self.random_rotation = enabled;
        self
    }

    pub fn with_whitening(mut self, enabled: bool) -> Self {
        self.whitening = enabled;
        self
    }

    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn with_sample_size(mut self, sample_size: usize) -> Self {
        self.sample_size = sample_size;
        self
    }

    pub fn with_max_training_memory_bytes(mut self, bytes: usize) -> Self {
        self.max_training_memory_bytes = bytes;
        self
    }

    pub fn with_sampling_seed(mut self, sampling_seed: u64) -> Self {
        self.sampling_seed = sampling_seed;
        self
    }

    pub fn with_rotation_seed(mut self, rotation_seed: u64) -> Self {
        self.rotation_seed = rotation_seed;
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
        if self.max_training_memory_bytes == 0 {
            return Err(Error::invalid_input(
                "pca.max_training_memory_bytes must be greater than 0".to_string(),
            ));
        }
        if !self.epsilon.is_finite()
            || (self.whitening && self.epsilon <= 0.0)
            || (!self.whitening && self.epsilon < 0.0)
        {
            let requirement = if self.whitening {
                "finite and greater than 0"
            } else {
                "finite and non-negative"
            };
            return Err(Error::invalid_input(format!(
                "pca.epsilon must be {requirement} when whitening={}, got {}",
                self.whitening, self.epsilon
            )));
        }
        Ok(())
    }

    /// Rejects a build when the estimated peak training memory exceeds the budget.
    pub fn validate_streaming_memory_budget(
        &self,
        input_dimension: usize,
        row_count: usize,
        batch_rows: usize,
    ) -> Result<()> {
        self.validate(input_dimension, row_count)?;
        let path = if row_count >= input_dimension {
            TrainingPath::Covariance
        } else {
            TrainingPath::Gram
        };
        validate_memory_budget(row_count, input_dimension, batch_rows, path, true, self)
    }
}

/// Diagnostics produced while training a PCA model.
#[derive(Debug, Clone)]
pub struct PcaTrainingStats {
    sampled_rows: usize,
    valid_rows: usize,
    effective_rank: usize,
    total_variance: f64,
    retained_variance: f64,
}

impl PcaTrainingStats {
    /// Number of rows supplied to the in-memory trainer.
    pub fn sampled_rows(&self) -> usize {
        self.sampled_rows
    }

    /// Number of valid rows used for training.
    pub fn valid_rows(&self) -> usize {
        self.valid_rows
    }

    /// Number of eigenvalues above the numerical rank threshold.
    pub fn effective_rank(&self) -> usize {
        self.effective_rank
    }

    /// Sum of all sanitized population-covariance eigenvalues.
    pub fn total_variance(&self) -> f64 {
        self.total_variance
    }

    /// Sum of the retained population-covariance eigenvalues.
    pub fn retained_variance(&self) -> f64 {
        self.retained_variance
    }

    /// Fraction of total variance retained by the model.
    pub fn retained_variance_ratio(&self) -> f64 {
        self.retained_variance / self.total_variance
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
    transform_matrix: Arc<Float32Array>,
    mean: Arc<Float32Array>,
    training_stats: Option<PcaTrainingStats>,
    fingerprint: OnceLock<[u8; 32]>,
}

impl DeepSizeOf for PcaModel {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        fn array_size(array: &Arc<Float32Array>, context: &mut Context) -> usize {
            if context.mark_seen(Arc::as_ptr(array) as usize) {
                std::mem::size_of_val(array.as_ref())
                    + (array.as_ref() as &dyn Array).deep_size_of_children(context)
            } else {
                0
            }
        }

        array_size(&self.transform_matrix, context) + array_size(&self.mean, context)
    }
}

fn encode_f32_tensor(values: &[f32], shape: Vec<u32>, label: &str) -> Result<pb::Tensor> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Error::index(format!(
            "{PCA_MODEL_NAME} {label} contains NaN or infinity"
        )));
    }
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
    for (index, bytes) in tensor.data.chunks_exact(size_of::<f32>()).enumerate() {
        let value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
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
        let expected_matrix_len = model
            .output_dimension
            .checked_mul(model.input_dimension)
            .ok_or_else(|| {
                Error::index(format!("{PCA_MODEL_NAME} matrix shape overflows usize"))
            })?;
        if model.output_dimension == 0
            || model.input_dimension == 0
            || model.output_dimension >= model.input_dimension
            || model.transform_matrix.len() != expected_matrix_len
            || model.mean.len() != model.input_dimension
        {
            return Err(Error::index(format!(
                "{PCA_MODEL_NAME} has invalid shape: matrix=[{}, {}] matrix data length={}, mean shape=[{}] mean data length={}",
                model.output_dimension,
                model.input_dimension,
                model.transform_matrix.len(),
                model.input_dimension,
                model.mean.len()
            )));
        }
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
            transform_matrix: Arc::new(Float32Array::from(transform_matrix)),
            mean: Arc::new(Float32Array::from(mean)),
            training_stats: None,
            fingerprint: OnceLock::new(),
        })
    }
}

/// Encodes a PCA model as the stable protobuf payload stored in a global buffer.
pub fn encode_pca_model(model: &PcaModel) -> Result<Vec<u8>> {
    Ok(pb::PcaModel::try_from(model)?.encode_to_vec())
}

/// Decodes and strictly validates a stable PCA protobuf payload.
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

/// Writes a PCA protobuf payload to an index-file global buffer.
///
/// Returns the global-buffer index stored under [`PCA_METADATA_KEY`].
pub async fn write_pca_model<W: IndexWriter + ?Sized>(
    writer: &mut W,
    model: &PcaModel,
) -> Result<u32> {
    writer
        .add_global_buffer(Bytes::from(encode_pca_model(model)?))
        .await
}

/// Loads a PCA model from the global buffer referenced by schema metadata.
///
/// `metadata_value` must represent a non-negative `u32` global-buffer index.
pub async fn read_pca_model<R: IndexReader + ?Sized>(
    reader: &R,
    metadata_value: &str,
) -> Result<PcaModel> {
    let buffer_index = metadata_value.parse::<u32>().map_err(|error| {
        Error::corrupt_file_named(
            PCA_MODEL_NAME,
            format!(
                "metadata key {PCA_METADATA_KEY} has invalid global buffer index {metadata_value:?}: {error}"
            ),
        )
    })?;
    let data = match reader.read_global_buffer(buffer_index).await {
        Ok(data) => data,
        Err(error @ (Error::NotFound { .. } | Error::InvalidInput { .. })) => {
            return Err(Error::corrupt_file_named(
                PCA_MODEL_NAME,
                format!(
                    "metadata key {PCA_METADATA_KEY} references missing or invalid global buffer index={buffer_index}: {error}"
                ),
            ));
        }
        Err(error) => return Err(error),
    };
    decode_pca_model(&data)
}

impl PcaModel {
    /// Trains a model from an in-memory fixed-size-list vector array.
    ///
    /// Training rejects null or non-finite values. All statistics and matrix
    /// decompositions use `f64`; the returned runtime model uses `f32`.
    pub fn train(data: &FixedSizeListArray, params: &PcaParams) -> Result<Self> {
        let input_dimension = usize::try_from(data.value_length()).map_err(|_| {
            Error::invalid_input(format!(
                "PCA input value_length={} is not a valid dimension",
                data.value_length()
            ))
        })?;
        let row_count = data.len();
        params.validate(input_dimension, row_count)?;
        validate_training_array(data)?;
        let path = if row_count >= input_dimension {
            TrainingPath::Covariance
        } else {
            TrainingPath::Gram
        };
        validate_memory_budget(row_count, input_dimension, row_count, path, false, params)?;

        let logical_values = logical_values(data)?;
        let values = values_as_f64(logical_values.as_ref())?;
        let value_count = row_count.checked_mul(input_dimension).ok_or_else(|| {
            Error::invalid_input(format!(
                "PCA input shape [{row_count}, {input_dimension}] overflows usize"
            ))
        })?;
        if values.len() != value_count {
            return Err(Error::invalid_input(format!(
                "PCA input values length {} does not match shape [{row_count}, {input_dimension}]",
                values.len()
            )));
        }

        let mean = compensated_mean(&values, row_count, input_dimension)?;
        let (eigenvalues, basis) = if row_count >= input_dimension {
            let covariance = covariance_matrix(&values, &mean, row_count, input_dimension)?;
            eigenpairs_descending(covariance)?
        } else {
            let centered = center_values(&values, &mean, row_count, input_dimension)?;
            gram_eigenvectors(&centered, row_count, input_dimension, params.dimension)?
        };
        let model =
            Self::from_eigenpairs(mean, row_count, input_dimension, eigenvalues, basis, params)?;
        model.validate_trained(data)?;
        Ok(model)
    }

    /// Trains a model from a mean and row-major population covariance matrix.
    ///
    /// The covariance must be a finite, symmetric `input_dimension *
    /// input_dimension` matrix, where `input_dimension == mean.len()`.
    /// `sample_count` records how many valid vectors contributed to the
    /// statistics. This entry point reuses the same eigendecomposition,
    /// whitening, and random-rotation kernel as [`Self::train`].
    pub fn train_from_covariance(
        mean: &[f64],
        sample_count: usize,
        covariance: &[f64],
        params: &PcaParams,
    ) -> Result<Self> {
        Self::train_from_second_moment(
            mean,
            sample_count,
            covariance,
            SecondMomentScale::Covariance,
            params,
        )
    }

    /// Trains a model from a mean and row-major centered scatter matrix.
    ///
    /// `scatter` must equal the sum of centered outer products and must be a
    /// finite, symmetric `mean.len() * mean.len()` matrix. Population
    /// covariance is computed as `scatter / sample_count`.
    pub fn train_from_scatter(
        mean: &[f64],
        sample_count: usize,
        scatter: &[f64],
        params: &PcaParams,
    ) -> Result<Self> {
        Self::train_from_second_moment(
            mean,
            sample_count,
            scatter,
            SecondMomentScale::Scatter,
            params,
        )
    }

    /// Trains a model from a mean and complete row-major centered observations.
    ///
    /// This Gram-matrix entry point is restricted to `sample_count <
    /// mean.len()`. `centered` must contain exactly `sample_count * mean.len()`
    /// finite values and each row is expected to have already had `mean`
    /// subtracted from it.
    pub fn train_from_centered(
        mean: &[f64],
        sample_count: usize,
        centered: &[f64],
        params: &PcaParams,
    ) -> Result<Self> {
        let input_dimension = mean.len();
        params.validate(input_dimension, sample_count)?;
        if sample_count >= input_dimension {
            return Err(Error::invalid_input(format!(
                "PCA Gram training requires sample_count < input_dimension, got sample_count={sample_count}, input_dimension={input_dimension}"
            )));
        }
        validate_finite_slice(mean, "mean")?;
        let expected_len = sample_count.checked_mul(input_dimension).ok_or_else(|| {
            Error::invalid_input(format!(
                "PCA centered shape [{sample_count}, {input_dimension}] overflows usize"
            ))
        })?;
        if centered.len() != expected_len {
            return Err(Error::invalid_input(format!(
                "PCA centered values length={} does not match shape [{sample_count}, {input_dimension}] requiring {expected_len} values",
                centered.len()
            )));
        }
        validate_finite_slice(centered, "centered values")?;
        validate_memory_budget(
            sample_count,
            input_dimension,
            0,
            TrainingPath::Gram,
            false,
            params,
        )?;
        let (eigenvalues, basis) =
            gram_eigenvectors(centered, sample_count, input_dimension, params.dimension)?;
        Self::from_eigenpairs(
            mean.to_vec(),
            sample_count,
            input_dimension,
            eigenvalues,
            basis,
            params,
        )
    }

    fn train_from_second_moment(
        mean: &[f64],
        sample_count: usize,
        second_moment: &[f64],
        scale: SecondMomentScale,
        params: &PcaParams,
    ) -> Result<Self> {
        let input_dimension = mean.len();
        params.validate(input_dimension, sample_count)?;
        validate_finite_slice(mean, "mean")?;
        validate_symmetric_matrix(second_moment, input_dimension, scale.label())?;
        validate_memory_budget(
            sample_count,
            input_dimension,
            0,
            TrainingPath::Covariance,
            false,
            params,
        )?;
        let denominator = match scale {
            SecondMomentScale::Covariance => 1.0,
            SecondMomentScale::Scatter => sample_count as f64,
        };
        let (eigenvalues, basis) =
            covariance_eigenvectors(second_moment, input_dimension, denominator)?;
        Self::from_eigenpairs(
            mean.to_vec(),
            sample_count,
            input_dimension,
            eigenvalues,
            basis,
            params,
        )
    }

    fn from_eigenpairs(
        mean: Vec<f64>,
        sample_count: usize,
        input_dimension: usize,
        eigenvalues: Vec<f64>,
        mut basis: Vec<f64>,
        params: &PcaParams,
    ) -> Result<Self> {
        let (sanitized, effective_rank, tau) = sanitize_eigenvalues(eigenvalues, params.dimension)?;
        if effective_rank < params.dimension {
            let selected_min = sanitized
                .get(params.dimension.saturating_sub(1))
                .copied()
                .unwrap_or(0.0);
            return Err(Error::invalid_input(format!(
                "PCA training data has effective_rank={effective_rank}, smaller than output_dimension={}; lambda_max={}, minimum_selected_eigenvalue={selected_min}, tau={tau}",
                params.dimension, sanitized[0]
            )));
        }

        let retained_basis_len = params
            .dimension
            .checked_mul(input_dimension)
            .ok_or_else(|| Error::invalid_input("PCA retained basis shape overflows usize"))?;
        if basis.len() < retained_basis_len {
            return Err(Error::index(format!(
                "PCA eigendecomposition basis length={} is smaller than retained shape [{}, {input_dimension}] requiring {retained_basis_len} values",
                basis.len(),
                params.dimension
            )));
        }
        basis.truncate(retained_basis_len);
        normalize_component_signs(&mut basis, params.dimension, input_dimension)?;
        let rotation = if params.random_rotation {
            Some(deterministic_rotation(
                params.dimension,
                params.rotation_seed,
            )?)
        } else {
            None
        };
        let matrix_f64 = assemble_transform(
            &basis,
            &sanitized[..params.dimension],
            rotation.as_deref(),
            input_dimension,
            params,
        )?;
        let transform_matrix = Arc::new(Float32Array::from(
            matrix_f64
                .iter()
                .map(|value| canonical_f32(*value))
                .collect::<Result<Vec<_>>>()?,
        ));
        let mean = Arc::new(Float32Array::from(
            mean.iter()
                .map(|value| canonical_f32(*value))
                .collect::<Result<Vec<_>>>()?,
        ));
        let total_variance = sanitized.iter().sum();
        let retained_variance = sanitized[..params.dimension].iter().sum();
        let model = Self {
            input_dimension,
            output_dimension: params.dimension,
            transform_matrix,
            mean,
            training_stats: Some(PcaTrainingStats {
                sampled_rows: sample_count,
                valid_rows: sample_count,
                effective_rank,
                total_variance,
                retained_variance,
            }),
            fingerprint: OnceLock::new(),
        };
        model.validate_model()?;
        Ok(model)
    }

    /// Applies the PCA transform to a batch of vectors.
    pub fn transform(&self, vectors: &FixedSizeListArray) -> Result<FixedSizeListArray> {
        validate_runtime_array(vectors, self.input_dimension)?;
        let row_count = vectors.len();
        let logical_values = logical_values(vectors)?;
        let input = values_as_f32(logical_values.as_ref())?;
        let output_len = row_count
            .checked_mul(self.output_dimension)
            .ok_or_else(|| Error::invalid_input("PCA output shape overflows usize"))?;
        let mut output = Vec::with_capacity(output_len);
        let matrix = Mat::<f32>::from_fn(
            self.output_dimension,
            self.input_dimension,
            |row, column| {
                self.transform_matrix
                    .value(row * self.input_dimension + column)
            },
        );
        for chunk_start in (0..row_count).step_by(TRANSFORM_BATCH_SIZE) {
            let chunk_end = chunk_start
                .saturating_add(TRANSFORM_BATCH_SIZE)
                .min(row_count);
            let chunk_rows = chunk_end - chunk_start;
            let centered = Mat::<f32>::from_fn(chunk_rows, self.input_dimension, |row, column| {
                input[(chunk_start + row) * self.input_dimension + column] - self.mean.value(column)
            });
            let transformed = &centered * matrix.transpose();
            for row in 0..chunk_rows {
                for column in 0..self.output_dimension {
                    let value = transformed[(row, column)];
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
        if query.null_count() != 0 {
            return Err(Error::invalid_input(format!(
                "PCA query contains {} null values",
                query.null_count()
            )));
        }
        if query.len() != self.input_dimension {
            return Err(Error::invalid_input(format!(
                "PCA query dimension {} does not match input_dimension={}",
                query.len(),
                self.input_dimension
            )));
        }
        let input = values_as_f32(query)?;
        let mut output = Vec::with_capacity(self.output_dimension);
        transform_row(
            &input,
            self.mean.values(),
            self.transform_matrix.values(),
            self.input_dimension,
            self.output_dimension,
            &mut output,
        )?;
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

    /// Returns training diagnostics when this model was trained in memory.
    pub fn training_stats(&self) -> Option<&PcaTrainingStats> {
        self.training_stats.as_ref()
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

    fn validate_trained(&self, training_data: &FixedSizeListArray) -> Result<()> {
        self.validate_model()?;

        let transformed = self.transform(training_data)?;
        if transformed
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(Error::index(
                "PCA training sample transform produced non-finite values".to_string(),
            ));
        }
        if !training_data.is_empty() {
            let first = training_data.value(0);
            let query = self.transform_query(first.as_ref())?;
            let query = query.as_primitive::<Float32Type>();
            let batch = transformed.value(0);
            let batch = batch.as_primitive::<Float32Type>();
            for (query_value, batch_value) in query.values().iter().zip(batch.values()) {
                let tolerance = 1.0e-5 + 1.0e-5 * batch_value.abs();
                if (query_value - batch_value).abs() > tolerance {
                    return Err(Error::index(
                        "PCA query and batch transforms differ beyond tolerance".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_model(&self) -> Result<()> {
        let expected_matrix_len = self
            .output_dimension
            .checked_mul(self.input_dimension)
            .ok_or_else(|| Error::index("PCA trained model matrix shape overflows usize"))?;
        if self.transform_matrix.len() != expected_matrix_len
            || self.mean.len() != self.input_dimension
        {
            return Err(Error::index(format!(
                "PCA trained model has matrix length {} and mean length {}, expected {} and {}",
                self.transform_matrix.len(),
                self.mean.len(),
                expected_matrix_len,
                self.input_dimension
            )));
        }
        if self
            .transform_matrix
            .values()
            .iter()
            .chain(self.mean.values())
            .any(|value| !value.is_finite())
        {
            return Err(Error::index(
                "PCA trained model contains non-finite values".to_string(),
            ));
        }

        let mut centered_mean = Vec::with_capacity(self.output_dimension);
        transform_row(
            self.mean.values(),
            self.mean.values(),
            self.transform_matrix.values(),
            self.input_dimension,
            self.output_dimension,
            &mut centered_mean,
        )?;
        let max_matrix = self
            .transform_matrix
            .values()
            .iter()
            .fold(0.0_f64, |value, item| value.max(f64::from(item.abs())));
        let max_mean = self
            .mean
            .values()
            .iter()
            .fold(0.0_f64, |value, item| value.max(f64::from(item.abs())));
        let tolerance = 32.0
            * f64::from(f32::EPSILON)
            * (self.input_dimension as f64 * max_matrix * max_mean + 1.0);
        if centered_mean
            .iter()
            .any(|value| f64::from(value.abs()) > tolerance)
        {
            return Err(Error::index(format!(
                "PCA trained model failed centered-mean validation with tolerance={tolerance}"
            )));
        }

        Ok(())
    }
}

#[derive(Clone, Copy)]
enum TrainingPath {
    Covariance,
    Gram,
}

#[derive(Clone, Copy)]
enum SecondMomentScale {
    Covariance,
    Scatter,
}

impl SecondMomentScale {
    fn label(self) -> &'static str {
        match self {
            Self::Covariance => "covariance",
            Self::Scatter => "scatter",
        }
    }
}

fn validate_training_array(data: &FixedSizeListArray) -> Result<()> {
    let logical_values = logical_values(data)?;
    if data.null_count() != 0 || logical_values.null_count() != 0 {
        return Err(Error::invalid_input(format!(
            "PCA training data contains nulls: vector_nulls={}, value_nulls={}",
            data.null_count(),
            logical_values.null_count()
        )));
    }
    match data.value_type() {
        DataType::Float16 | DataType::Float32 | DataType::Float64 => Ok(()),
        data_type => Err(Error::invalid_input(format!(
            "PCA training data must contain Float16, Float32, or Float64 values, got {data_type}"
        ))),
    }
}

fn logical_values(data: &FixedSizeListArray) -> Result<ArrayRef> {
    let dimension = usize::try_from(data.value_length()).map_err(|_| {
        Error::invalid_input(format!(
            "PCA vector value_length={} is not a valid dimension",
            data.value_length()
        ))
    })?;
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
            data.values().len()
        )));
    }
    Ok(data.values().slice(start, length))
}

fn validate_runtime_array(data: &FixedSizeListArray, expected_dimension: usize) -> Result<()> {
    validate_training_array(data)?;
    let dimension = usize::try_from(data.value_length()).map_err(|_| {
        Error::invalid_input(format!(
            "PCA vector dimension {} is invalid",
            data.value_length()
        ))
    })?;
    if dimension != expected_dimension {
        return Err(Error::invalid_input(format!(
            "PCA vector dimension {dimension} does not match input_dimension={expected_dimension}"
        )));
    }
    Ok(())
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

fn checked_bytes(elements: usize, element_size: usize, label: &str) -> Result<usize> {
    elements.checked_mul(element_size).ok_or_else(|| {
        Error::invalid_input(format!(
            "PCA training memory estimate for {label} overflows usize"
        ))
    })
}

fn validate_memory_budget(
    row_count: usize,
    input_dimension: usize,
    batch_rows: usize,
    path: TrainingPath,
    uses_matrix_accumulation: bool,
    params: &PcaParams,
) -> Result<()> {
    let matrix_dimension = match path {
        TrainingPath::Covariance => input_dimension,
        TrainingPath::Gram => row_count,
    };
    let matrix_elements = matrix_dimension
        .checked_mul(matrix_dimension)
        .ok_or_else(|| Error::invalid_input("PCA statistics matrix shape overflows usize"))?;
    let stat_elements = match path {
        TrainingPath::Covariance => matrix_elements,
        TrainingPath::Gram => row_count
            .checked_mul(input_dimension)
            .and_then(|centered| centered.checked_add(matrix_elements))
            .ok_or_else(|| Error::invalid_input("PCA Gram statistics shape overflows usize"))?,
    };
    let stat = checked_bytes(stat_elements, size_of::<f64>(), "statistics")?;
    let eig = checked_bytes(
        matrix_elements.checked_mul(3).ok_or_else(|| {
            Error::invalid_input("PCA eigensolver workspace estimate overflows usize")
        })?,
        size_of::<f64>(),
        "eigensolver",
    )?;
    let final_model_elements = params
        .dimension
        .checked_mul(input_dimension)
        .and_then(|matrix| matrix.checked_add(input_dimension))
        .ok_or_else(|| Error::invalid_input("PCA model shape overflows usize"))?;
    let final_model = checked_bytes(final_model_elements, size_of::<f32>(), "model")?;
    let transform_elements = params
        .dimension
        .checked_mul(input_dimension)
        .ok_or_else(|| Error::invalid_input("PCA transform workspace shape overflows usize"))?;
    // Training can concurrently hold the retained basis, its whitened copy,
    // and the final rotated transform before conversion to f32.
    let transform_workspace = checked_bytes(
        transform_elements
            .checked_mul(3)
            .ok_or_else(|| Error::invalid_input("PCA transform workspace overflows usize"))?,
        size_of::<f64>(),
        "transform workspace",
    )?;
    // Rotation construction holds the Gaussian input, QR factors, and Q.
    let rotation_elements = params
        .dimension
        .checked_mul(params.dimension)
        .and_then(|elements| elements.checked_mul(3))
        .ok_or_else(|| Error::invalid_input("PCA rotation workspace shape overflows usize"))?;
    let rotation_workspace = if params.random_rotation {
        checked_bytes(rotation_elements, size_of::<f64>(), "rotation workspace")?
    } else {
        0
    };
    let model = final_model
        .checked_add(transform_workspace)
        .and_then(|value| value.checked_add(rotation_workspace))
        .ok_or_else(|| Error::invalid_input("PCA model memory estimate overflows usize"))?;
    // The in-memory entry point converts its complete Arrow input to f64. The
    // dataset trainer can replace this term with its bounded streaming batch.
    let effective_batch_rows = batch_rows.min(row_count);
    let batch_elements = effective_batch_rows
        .checked_mul(input_dimension)
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "PCA training batch shape [{effective_batch_rows}, {input_dimension}] overflows usize"
            ))
        })?;
    let batch = checked_bytes(batch_elements, size_of::<f64>(), "batch")?;
    // Streaming covariance uses a centered batch matrix and a batch-local
    // scatter matrix in addition to the converted input batch and accumulated
    // statistics.
    let matrix_accumulation_workspace =
        if uses_matrix_accumulation && matches!(path, TrainingPath::Covariance) {
            batch.checked_add(stat).ok_or_else(|| {
                Error::invalid_input("PCA matrix accumulation workspace estimate overflows usize")
            })?
        } else {
            0
        };
    let total = stat
        .checked_add(eig)
        .and_then(|value| value.checked_add(model))
        .and_then(|value| value.checked_add(batch))
        .and_then(|value| value.checked_add(matrix_accumulation_workspace))
        .ok_or_else(|| Error::invalid_input("PCA total memory estimate overflows usize"))?;
    if total > params.max_training_memory_bytes {
        return Err(Error::invalid_input(format!(
            "PCA training memory estimate exceeds max_training_memory_bytes={}: M_stat={stat}, M_eig={eig}, M_model={model}, M_batch={batch}, M_matrix_accumulation={matrix_accumulation_workspace}, total={total}",
            params.max_training_memory_bytes
        )));
    }
    Ok(())
}

fn values_as_f64(values: &dyn Array) -> Result<Vec<f64>> {
    if values.null_count() != 0 {
        return Err(Error::invalid_input(format!(
            "PCA input contains {} null scalar values",
            values.null_count()
        )));
    }
    let converted = match values.data_type() {
        DataType::Float16 => values
            .as_primitive::<Float16Type>()
            .values()
            .iter()
            .map(|value| f64::from(value.to_f32()))
            .collect(),
        DataType::Float32 => values
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .map(|value| f64::from(*value))
            .collect(),
        DataType::Float64 => values.as_primitive::<Float64Type>().values().to_vec(),
        data_type => {
            return Err(Error::invalid_input(format!(
                "PCA input must be Float16, Float32, or Float64, got {data_type}"
            )));
        }
    };
    if converted.iter().any(|value| !value.is_finite()) {
        return Err(Error::invalid_input(
            "PCA input contains non-finite values".to_string(),
        ));
    }
    Ok(converted)
}

fn values_as_f32(values: &dyn Array) -> Result<Vec<f32>> {
    if values.null_count() != 0 {
        return Err(Error::invalid_input(format!(
            "PCA input contains {} null scalar values",
            values.null_count()
        )));
    }
    let converted = match values.data_type() {
        DataType::Float16 => values
            .as_primitive::<Float16Type>()
            .values()
            .iter()
            .map(|value| value.to_f32())
            .collect(),
        DataType::Float32 => values.as_primitive::<Float32Type>().values().to_vec(),
        DataType::Float64 => values
            .as_primitive::<Float64Type>()
            .values()
            .iter()
            .map(|value| *value as f32)
            .collect(),
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

fn compensated_mean(values: &[f64], rows: usize, dimension: usize) -> Result<Vec<f64>> {
    let mut sums = vec![0.0; dimension];
    let mut corrections = vec![0.0; dimension];
    for row in values.chunks_exact(dimension) {
        for (column, value) in row.iter().enumerate() {
            let adjusted = *value - corrections[column];
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

fn center_values(values: &[f64], mean: &[f64], rows: usize, dimension: usize) -> Result<Vec<f64>> {
    let length = rows
        .checked_mul(dimension)
        .ok_or_else(|| Error::invalid_input("PCA centered matrix shape overflows usize"))?;
    let mut centered = Vec::with_capacity(length);
    for row in values.chunks_exact(dimension) {
        centered.extend(row.iter().zip(mean).map(|(value, mean)| value - mean));
    }
    if centered.iter().any(|value| !value.is_finite()) {
        return Err(Error::invalid_input(
            "PCA centering produced a non-finite value".to_string(),
        ));
    }
    Ok(centered)
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

    let rows = values.len() / dimension;
    let centered = Mat::<f64>::from_fn(rows, dimension, |row, column| {
        values[row * dimension + column] - mean[column]
    });
    let batch_scatter = centered.transpose() * &centered;
    for row in 0..dimension {
        for column in row..dimension {
            let value = scatter[row * dimension + column] + batch_scatter[(row, column)];
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

fn covariance_matrix(
    values: &[f64],
    mean: &[f64],
    rows: usize,
    dimension: usize,
) -> Result<Mat<f64>> {
    let mut covariance = Mat::<f64>::zeros(dimension, dimension);
    let denominator = rows as f64;
    for column in 0..dimension {
        for other in column..dimension {
            let mut sum = 0.0;
            for row in 0..rows {
                sum += (values[row * dimension + column] - mean[column])
                    * (values[row * dimension + other] - mean[other]);
            }
            let value = sum / denominator;
            if !value.is_finite() {
                return Err(Error::invalid_input(format!(
                    "PCA population covariance contains non-finite value={value} at row={column}, column={other}, shape=[{dimension}, {dimension}]"
                )));
            }
            covariance[(column, other)] = value;
        }
    }
    Ok(covariance)
}

fn covariance_eigenvectors(
    covariance: &[f64],
    dimension: usize,
    denominator: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let covariance = Mat::<f64>::from_fn(dimension, dimension, |row, column| {
        covariance[row * dimension + column] / denominator
    });
    eigenpairs_descending(covariance)
}

fn gram_eigenvectors(
    centered: &[f64],
    rows: usize,
    input_dimension: usize,
    output_dimension: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let mut gram = Mat::<f64>::zeros(rows, rows);
    let denominator = rows as f64;
    for row in 0..rows {
        for other in row..rows {
            let mut sum = 0.0;
            for column in 0..input_dimension {
                sum += centered[row * input_dimension + column]
                    * centered[other * input_dimension + column];
            }
            gram[(row, other)] = sum / denominator;
        }
    }
    let (eigenvalues, gram_vectors) = eigenpairs_descending(gram)?;
    let (_, effective_rank, tau) = sanitize_eigenvalues(eigenvalues.clone(), output_dimension)?;
    if effective_rank < output_dimension {
        return Ok((eigenvalues, Vec::new()));
    }

    let basis_len = output_dimension
        .checked_mul(input_dimension)
        .ok_or_else(|| Error::invalid_input("PCA Gram basis shape overflows usize"))?;
    let mut basis = vec![0.0; basis_len];
    for component in 0..output_dimension {
        let denominator = (rows as f64 * eigenvalues[component]).sqrt();
        if !denominator.is_finite() || denominator <= tau.sqrt() {
            return Err(Error::index(format!(
                "PCA Gram eigenvector lift has invalid denominator={denominator} for component={component}"
            )));
        }
        for column in 0..input_dimension {
            let mut value = 0.0;
            for row in 0..rows {
                value +=
                    centered[row * input_dimension + column] * gram_vectors[component * rows + row];
            }
            basis[component * input_dimension + column] = value / denominator;
        }
        normalize_vector(
            &mut basis[component * input_dimension..(component + 1) * input_dimension],
            "Gram-lifted eigenvector",
        )?;
    }
    Ok((eigenvalues, basis))
}

fn eigenpairs_descending(matrix: Mat<f64>) -> Result<(Vec<f64>, Vec<f64>)> {
    let dimension = matrix.nrows();
    let decomposition = SelfAdjointEigen::new(matrix.as_ref(), Side::Upper)
        .map_err(|error| Error::index(format!("PCA eigendecomposition failed: {error:?}")))?;
    let eigenvalues = decomposition.S().column_vector();
    let eigenvectors = decomposition.U();
    let mut descending_values = Vec::with_capacity(dimension);
    let element_count = dimension
        .checked_mul(dimension)
        .ok_or_else(|| Error::invalid_input("PCA eigenvector shape overflows usize"))?;
    let mut descending_vectors = Vec::with_capacity(element_count);
    for component in (0..dimension).rev() {
        descending_values.push(eigenvalues[component]);
        for row in 0..dimension {
            descending_vectors.push(eigenvectors[(row, component)]);
        }
    }
    Ok((descending_values, descending_vectors))
}

fn sanitize_eigenvalues(
    eigenvalues: Vec<f64>,
    output_dimension: usize,
) -> Result<(Vec<f64>, usize, f64)> {
    let Some(lambda_max) = eigenvalues.first().copied() else {
        return Err(Error::invalid_input(
            "PCA eigendecomposition returned no eigenvalues".to_string(),
        ));
    };
    if !lambda_max.is_finite() {
        return Err(Error::index(
            "PCA eigendecomposition returned a non-finite maximum eigenvalue".to_string(),
        ));
    }
    let tau = f64::MIN_POSITIVE.max(1.0e-10 * lambda_max.abs());
    let mut sanitized = Vec::with_capacity(eigenvalues.len());
    let mut effective_rank = 0;
    for (index, eigenvalue) in eigenvalues.into_iter().enumerate() {
        if !eigenvalue.is_finite() {
            return Err(Error::index(format!(
                "PCA eigenvalue at descending index {index} is non-finite"
            )));
        }
        if eigenvalue < -tau {
            return Err(Error::index(format!(
                "PCA eigenvalue at descending index {index} is {eigenvalue}, below -tau={}",
                -tau
            )));
        }
        if eigenvalue.abs() <= tau {
            sanitized.push(0.0);
        } else {
            effective_rank += 1;
            sanitized.push(eigenvalue);
        }
    }
    if sanitized.len() < output_dimension {
        return Err(Error::index(format!(
            "PCA eigendecomposition returned {} eigenvalues, fewer than output_dimension={output_dimension}",
            sanitized.len()
        )));
    }
    Ok((sanitized, effective_rank, tau))
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

fn deterministic_gaussian_matrix(dimension: usize, seed: u64) -> Result<Vec<f64>> {
    let length = dimension
        .checked_mul(dimension)
        .ok_or_else(|| Error::invalid_input("PCA random rotation shape overflows usize"))?;
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut values = Vec::with_capacity(length);
    while values.len() < length {
        let a = rng.next_u64() >> 11;
        let b = rng.next_u64() >> 11;
        let u1 = (a + 1) as f64 * 2.0_f64.powi(-53);
        let u2 = b as f64 * 2.0_f64.powi(-53);
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = TAU * u2;
        values.push(radius * angle.cos());
        if values.len() < length {
            values.push(radius * angle.sin());
        }
    }
    Ok(values)
}

fn deterministic_rotation(dimension: usize, seed: u64) -> Result<Vec<f64>> {
    let gaussian = deterministic_gaussian_matrix(dimension, seed)?;
    let matrix = Mat::from_fn(dimension, dimension, |row, column| {
        gaussian[row * dimension + column]
    });
    let mut rotation = matrix.qr().compute_Q();
    for column in 0..dimension {
        let mut pivot = 0;
        for row in 1..dimension {
            if rotation[(row, column)].abs() > rotation[(pivot, column)].abs() {
                pivot = row;
            }
        }
        if rotation[(pivot, column)] < 0.0 {
            for row in 0..dimension {
                rotation[(row, column)] = -rotation[(row, column)];
            }
        }
    }
    let tolerance = 64.0 * dimension as f64 * f64::EPSILON;
    let mut row_major = Vec::with_capacity(dimension.saturating_mul(dimension));
    for row in 0..dimension {
        for column in 0..dimension {
            let mut dot = 0.0;
            for inner in 0..dimension {
                dot += rotation[(inner, row)] * rotation[(inner, column)];
            }
            let expected = if row == column { 1.0 } else { 0.0 };
            if (dot - expected).abs() > tolerance {
                return Err(Error::index(format!(
                    "PCA random rotation failed orthogonality validation: row={row}, column={column}, value={dot}, tolerance={tolerance}"
                )));
            }
            row_major.push(rotation[(row, column)]);
        }
    }
    Ok(row_major)
}

fn assemble_transform(
    basis: &[f64],
    eigenvalues: &[f64],
    rotation: Option<&[f64]>,
    input_dimension: usize,
    params: &PcaParams,
) -> Result<Vec<f64>> {
    let output_dimension = params.dimension;
    let length = output_dimension
        .checked_mul(input_dimension)
        .ok_or_else(|| Error::invalid_input("PCA transform matrix shape overflows usize"))?;
    let mut scaled_basis = vec![0.0; length];
    for component in 0..output_dimension {
        let scale = if params.whitening {
            (eigenvalues[component] + f64::from(params.epsilon))
                .sqrt()
                .recip()
        } else {
            1.0
        };
        if !scale.is_finite() {
            return Err(Error::index(format!(
                "PCA whitening scale for component={component} is non-finite"
            )));
        }
        for column in 0..input_dimension {
            scaled_basis[component * input_dimension + column] =
                basis[component * input_dimension + column] * scale;
        }
    }
    let Some(rotation) = rotation else {
        return Ok(scaled_basis);
    };
    let mut transform = vec![0.0; length];
    for row in 0..output_dimension {
        for column in 0..input_dimension {
            let mut value = 0.0;
            for inner in 0..output_dimension {
                value += rotation[row * output_dimension + inner]
                    * scaled_basis[inner * input_dimension + column];
            }
            transform[row * input_dimension + column] = value;
        }
    }
    Ok(transform)
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

/// Environment variable that restores the original single-accumulator
/// scalar query transform, for A/B measurement.
pub const PCA_QUERY_TRANSFORM_SCALAR_ENV: &str = "LANCE_PCA_QUERY_TRANSFORM_SCALAR";

static QUERY_TRANSFORM_SCALAR: LazyLock<bool> = LazyLock::new(|| {
    std::env::var_os(PCA_QUERY_TRANSFORM_SCALAR_ENV)
        .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
});

/// Inner kernel used by the per-query PCA transform `y = M (x - mean)`.
///
/// Production code goes through [`TransformKernel::selected`].
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformKernel {
    /// Original loop: one accumulator per output row, the input re-centred
    /// for every row. Reference for the others.
    Scalar,
    /// Input centred once, eight independent accumulator chains per row,
    /// no fused multiply-add. Used off aarch64.
    Portable,
    /// NEON: four output rows per pass share each loaded chunk of the
    /// centred input. aarch64 only, and the default there.
    Neon,
}

impl TransformKernel {
    /// Kernel used by [`PcaModel::transform_query`] on this host.
    pub fn selected() -> Self {
        if *QUERY_TRANSFORM_SCALAR {
            return Self::Scalar;
        }
        if cfg!(target_arch = "aarch64") {
            Self::Neon
        } else {
            Self::Portable
        }
    }

    /// Whether this kernel can run in this build on this host.
    pub fn is_available(self) -> bool {
        match self {
            Self::Scalar | Self::Portable => true,
            Self::Neon => cfg!(target_arch = "aarch64"),
        }
    }
}

/// Appends `matrix * (input - mean)` to `output` with the kernel selected
/// for this host.
fn transform_row(
    input: &[f32],
    mean: &[f32],
    matrix: &[f32],
    input_dimension: usize,
    output_dimension: usize,
    output: &mut Vec<f32>,
) -> Result<()> {
    transform_row_with(
        TransformKernel::selected(),
        input,
        mean,
        matrix,
        input_dimension,
        output_dimension,
        output,
    )
}

/// [`transform_row`] with an explicit kernel. Every kernel checks each output
/// value for finiteness.
#[doc(hidden)]
pub fn transform_row_with(
    kernel: TransformKernel,
    input: &[f32],
    mean: &[f32],
    matrix: &[f32],
    input_dimension: usize,
    output_dimension: usize,
    output: &mut Vec<f32>,
) -> Result<()> {
    debug_assert_eq!(input.len(), input_dimension);
    debug_assert_eq!(mean.len(), input_dimension);
    debug_assert_eq!(matrix.len(), input_dimension * output_dimension);
    if kernel == TransformKernel::Scalar {
        for row in 0..output_dimension {
            let mut value = 0.0_f32;
            for column in 0..input_dimension {
                value += matrix[row * input_dimension + column] * (input[column] - mean[column]);
            }
            push_finite(value, row, output)?;
        }
        return Ok(());
    }

    let centered: Vec<f32> = input.iter().zip(mean).map(|(x, m)| x - m).collect();
    output.reserve(output_dimension);
    match kernel {
        TransformKernel::Scalar => unreachable!("handled above"),
        TransformKernel::Portable => {
            for (row, matrix_row) in matrix.chunks_exact(input_dimension).enumerate() {
                push_finite(dot_row_portable(&centered, matrix_row), row, output)?;
            }
        }
        TransformKernel::Neon => {
            #[cfg(target_arch = "aarch64")]
            {
                let mut row = 0;
                while row + 4 <= output_dimension {
                    let values = neon::dot_4rows(
                        &centered,
                        &matrix[row * input_dimension..],
                        input_dimension,
                    );
                    for (offset, value) in values.into_iter().enumerate() {
                        push_finite(value, row + offset, output)?;
                    }
                    row += 4;
                }
                while row < output_dimension {
                    let matrix_row = &matrix[row * input_dimension..(row + 1) * input_dimension];
                    push_finite(neon::dot_row(&centered, matrix_row), row, output)?;
                    row += 1;
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                let _ = centered;
                return Err(Error::invalid_input(
                    "PCA NEON transform kernel is only available on aarch64".to_string(),
                ));
            }
        }
    }
    Ok(())
}

#[inline]
fn push_finite(value: f32, row: usize, output: &mut Vec<f32>) -> Result<()> {
    if !value.is_finite() {
        return Err(Error::invalid_input(format!(
            "PCA transform produced a non-finite value at output dimension {row}"
        )));
    }
    output.push(value);
    Ok(())
}

/// Dot product of the centred input with one matrix row.
fn dot_row_portable(centered: &[f32], matrix_row: &[f32]) -> f32 {
    debug_assert_eq!(centered.len(), matrix_row.len());
    let mut acc = [0.0_f32; 8];
    let full = centered.len() & !7;
    for (c, m) in centered[..full]
        .chunks_exact(8)
        .zip(matrix_row[..full].chunks_exact(8))
    {
        for lane in 0..8 {
            acc[lane] += m[lane] * c[lane];
        }
    }
    let mut value =
        ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for (c, m) in centered[full..].iter().zip(&matrix_row[full..]) {
        value += m * c;
    }
    value
}

/// NEON kernels for the per-query transform. Results differ from the scalar
/// loop at the ulp level because of fused multiply-add.
#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// Dot products of `centered` with the first four `stride`-long rows of
    /// `rows`.
    #[inline]
    pub(super) fn dot_4rows(centered: &[f32], rows: &[f32], stride: usize) -> [f32; 4] {
        let n = centered.len();
        debug_assert!(stride >= n);
        debug_assert!(rows.len() >= 3 * stride + n);
        let full = n & !7;
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

    /// Dot product of `centered` with one matrix row.
    #[inline]
    pub(super) fn dot_row(centered: &[f32], matrix_row: &[f32]) -> f32 {
        let n = centered.len();
        debug_assert_eq!(matrix_row.len(), n);
        let full = n & !15;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;
    use arrow_array::{Float16Array, Float64Array, RecordBatch};
    use async_trait::async_trait;
    use half::f16;
    use lance_core::datatypes::Schema;
    use rstest::rstest;
    use std::collections::HashMap;
    use std::sync::Mutex;

    fn persisted_model(matrix: Vec<f32>, mean: Vec<f32>) -> PcaModel {
        PcaModel {
            input_dimension: mean.len(),
            output_dimension: matrix.len() / mean.len(),
            transform_matrix: Arc::new(Float32Array::from(matrix)),
            mean: Arc::new(Float32Array::from(mean)),
            training_stats: None,
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

    struct BufferWriter {
        buffers: Arc<Mutex<Vec<Bytes>>>,
    }

    #[async_trait]
    impl IndexWriter for BufferWriter {
        async fn write_record_batch(&mut self, _batch: RecordBatch) -> Result<u64> {
            Err(Error::not_supported(
                "record batches are unused by this test",
            ))
        }

        async fn add_global_buffer(&mut self, data: Bytes) -> Result<u32> {
            let mut buffers = self
                .buffers
                .lock()
                .expect("test mutex must not be poisoned");
            buffers.push(data);
            u32::try_from(buffers.len() - 1)
                .map_err(|_| Error::index("test global buffer index exceeds u32"))
        }

        async fn finish(&mut self) -> Result<crate::scalar::IndexFile> {
            Err(Error::not_supported("finish is unused by this test"))
        }

        async fn finish_with_metadata(
            &mut self,
            _metadata: HashMap<String, String>,
        ) -> Result<crate::scalar::IndexFile> {
            Err(Error::not_supported("finish is unused by this test"))
        }
    }

    struct BufferReader {
        buffers: Arc<Mutex<Vec<Bytes>>>,
        schema: Schema,
        fail_reads_with_io: bool,
    }

    #[async_trait]
    impl IndexReader for BufferReader {
        async fn read_record_batch(&self, _n: u64, _batch_size: u64) -> Result<RecordBatch> {
            Err(Error::not_supported(
                "record batches are unused by this test",
            ))
        }

        async fn read_global_buffer(&self, index: u32) -> Result<Bytes> {
            if self.fail_reads_with_io {
                return Err(Error::io("injected PCA global buffer read failure"));
            }
            self.buffers
                .lock()
                .expect("test mutex must not be poisoned")
                .get(index as usize)
                .cloned()
                .ok_or_else(|| Error::not_found(format!("global buffer index={index}")))
        }

        async fn read_range(
            &self,
            _range: std::ops::Range<usize>,
            _projection: Option<&[&str]>,
        ) -> Result<RecordBatch> {
            Err(Error::not_supported("ranges are unused by this test"))
        }

        async fn num_batches(&self, _batch_size: u64) -> u32 {
            0
        }

        fn num_rows(&self) -> usize {
            0
        }

        fn schema(&self) -> &Schema {
            &self.schema
        }
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
    fn params_have_documented_defaults() {
        let params = PcaParams::new(8);
        assert!(!params.random_rotation);
        assert!(!params.whitening);
        assert_eq!(params.epsilon, 1.0e-5);
        assert_eq!(params.sample_size, 65_536);
        assert_eq!(params.max_training_memory_bytes, 1_073_741_824);
        assert_eq!(params.sampling_seed, 42);
        assert_eq!(params.rotation_seed, 5);
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
        let stats = model.training_stats().expect("training has stats");
        assert_eq!(stats.sampled_rows(), 8);
        assert!(stats.effective_rank() >= 2);
        assert!(stats.retained_variance_ratio() > 0.0);
        assert!(stats.retained_variance_ratio() <= 1.0 + 1.0e-12);
    }

    #[test]
    fn gram_path_trains_and_query_matches_batch() {
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
    fn covariance_and_scatter_statistics_match_in_memory_training() {
        let rows = 8;
        let dimension = 3;
        let values = full_rank_values(rows, dimension);
        let vectors = f32_vectors(values.clone(), dimension as i32);
        let values = values.into_iter().map(f64::from).collect::<Vec<_>>();
        let mean =
            compensated_mean(&values, rows, dimension).expect("mean calculation should work");
        let covariance_matrix =
            covariance_matrix(&values, &mean, rows, dimension).expect("covariance should work");
        let mut covariance = Vec::with_capacity(dimension * dimension);
        for row in 0..dimension {
            for column in 0..dimension {
                covariance.push(covariance_matrix[(row.min(column), row.max(column))]);
            }
        }
        let scatter = covariance
            .iter()
            .map(|value| value * rows as f64)
            .collect::<Vec<_>>();
        let params = PcaParams::new(2)
            .with_whitening(true)
            .with_random_rotation(true)
            .with_rotation_seed(31);

        let in_memory = PcaModel::train(&vectors, &params).expect("array training should work");
        let from_covariance = PcaModel::train_from_covariance(&mean, rows, &covariance, &params)
            .expect("covariance training should work");
        let from_scatter = PcaModel::train_from_scatter(&mean, rows, &scatter, &params)
            .expect("scatter training should work");

        assert_eq!(from_covariance.fingerprint(), in_memory.fingerprint());
        assert_eq!(from_scatter.fingerprint(), in_memory.fingerprint());
        assert_eq!(
            from_covariance
                .transform(&vectors)
                .expect("covariance model transform should work"),
            in_memory
                .transform(&vectors)
                .expect("in-memory model transform should work")
        );
    }

    #[test]
    fn centered_gram_statistics_match_in_memory_training() {
        let rows = 4;
        let dimension = 6;
        let values = full_rank_values(rows, dimension);
        let vectors = f32_vectors(values.clone(), dimension as i32);
        let values = values.into_iter().map(f64::from).collect::<Vec<_>>();
        let mean =
            compensated_mean(&values, rows, dimension).expect("mean calculation should work");
        let centered =
            center_values(&values, &mean, rows, dimension).expect("centering should work");
        let params = PcaParams::new(2)
            .with_whitening(true)
            .with_random_rotation(true)
            .with_rotation_seed(37);

        let in_memory = PcaModel::train(&vectors, &params).expect("array training should work");
        let from_centered = PcaModel::train_from_centered(&mean, rows, &centered, &params)
            .expect("Gram training should work");

        assert_eq!(from_centered.fingerprint(), in_memory.fingerprint());
        assert_eq!(
            from_centered
                .transform(&vectors)
                .expect("Gram model transform should work"),
            in_memory
                .transform(&vectors)
                .expect("in-memory model transform should work")
        );
    }

    #[test]
    fn statistics_apis_validate_shapes_values_and_streaming_memory() {
        let params = PcaParams::new(1);
        let error = PcaModel::train_from_covariance(&[0.0, 0.0], 3, &[1.0, 0.0, 0.0], &params)
            .expect_err("incorrect covariance length must fail");
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("requiring 4 values"));

        let error = PcaModel::train_from_scatter(&[0.0, 0.0], 3, &[1.0, 2.0, 0.0, 1.0], &params)
            .expect_err("non-symmetric scatter must fail");
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("must be symmetric"));

        let error = PcaModel::train_from_centered(&[0.0, f64::NAN, 0.0], 2, &[0.0; 6], &params)
            .expect_err("non-finite mean must fail");
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("mean"));
        assert!(error.to_string().contains("element=1"));

        let error = PcaParams::new(1)
            .with_max_training_memory_bytes(1)
            .validate_streaming_memory_budget(3, 8, 2)
            .expect_err("small streaming budget must fail");
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("M_batch=48"));
        assert!(error.to_string().contains("M_matrix_accumulation="));
        assert!(error.to_string().contains("total="));
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
    fn streaming_matrix_memory_budget_has_exact_boundary() {
        let accepted = PcaParams::new(1).with_max_training_memory_bytes(552);
        accepted
            .validate_streaming_memory_budget(3, 8, 2)
            .expect("exact streaming matrix memory estimate should pass");

        let rejected = PcaParams::new(1).with_max_training_memory_bytes(551);
        let error = rejected
            .validate_streaming_memory_budget(3, 8, 2)
            .expect_err("one byte below the matrix memory estimate must fail");
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("M_stat=72"));
        assert!(error.to_string().contains("M_batch=48"));
        assert!(error.to_string().contains("M_matrix_accumulation=120"));
        assert!(error.to_string().contains("total=552"));
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
    fn gram_path_lifts_largest_axes_in_descending_order() {
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
        assert_relative_eq!(
            model
                .training_stats()
                .expect("training has stats")
                .retained_variance(),
            5.0,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn whitening_uses_population_eigenvalues() {
        let vectors = f32_vectors(full_rank_values(10, 3), 3);
        let epsilon = 1.0e-3;
        let plain =
            PcaModel::train(&vectors, &PcaParams::new(2)).expect("plain training should work");
        let whitened = PcaModel::train(
            &vectors,
            &PcaParams::new(2).with_whitening(true).with_epsilon(epsilon),
        )
        .expect("whitening training should work");
        let plain_output = plain
            .transform(&vectors)
            .expect("plain transform should work");
        let whitened_output = whitened
            .transform(&vectors)
            .expect("whitened transform should work");
        let plain_values = plain_output.values().as_primitive::<Float32Type>();
        let whitened_values = whitened_output.values().as_primitive::<Float32Type>();
        for column in 0..2 {
            let plain_variance: f64 = (0..10)
                .map(|row| f64::from(plain_values.value(row * 2 + column)).powi(2))
                .sum::<f64>()
                / 10.0;
            let whitened_variance: f64 = (0..10)
                .map(|row| f64::from(whitened_values.value(row * 2 + column)).powi(2))
                .sum::<f64>()
                / 10.0;
            let expected = plain_variance / (plain_variance + f64::from(epsilon));
            assert_relative_eq!(
                whitened_variance,
                expected,
                epsilon = 5.0e-3,
                max_relative = 5.0e-2
            );
        }
    }

    #[test]
    fn random_rotation_is_deterministic_and_preserves_distance() {
        let vectors = f32_vectors(full_rank_values(9, 4), 4);
        let plain =
            PcaModel::train(&vectors, &PcaParams::new(3)).expect("plain training should work");
        let rotated_params = PcaParams::new(3)
            .with_random_rotation(true)
            .with_rotation_seed(17);
        let rotated =
            PcaModel::train(&vectors, &rotated_params).expect("rotation training should work");
        let repeated =
            PcaModel::train(&vectors, &rotated_params).expect("repeated training should work");
        assert_eq!(
            rotated.transform_matrix.values(),
            repeated.transform_matrix.values()
        );
        let plain_output = plain
            .transform(&vectors)
            .expect("plain transform should work");
        let rotated_output = rotated
            .transform(&vectors)
            .expect("rotation transform should work");
        let plain_values = plain_output.values().as_primitive::<Float32Type>();
        let rotated_values = rotated_output.values().as_primitive::<Float32Type>();
        let plain_distance: f32 = (0..3)
            .map(|column| {
                let difference = plain_values.value(column) - plain_values.value(3 + column);
                difference * difference
            })
            .sum();
        let rotated_distance: f32 = (0..3)
            .map(|column| {
                let difference = rotated_values.value(column) - rotated_values.value(3 + column);
                difference * difference
            })
            .sum();
        assert_relative_eq!(plain_distance, rotated_distance, max_relative = 1.0e-4);
    }

    #[test]
    fn whitening_and_rotation_can_be_combined() {
        let vectors = f32_vectors(full_rank_values(9, 4), 4);
        let params = PcaParams::new(3)
            .with_whitening(true)
            .with_epsilon(1.0e-4)
            .with_random_rotation(true)
            .with_rotation_seed(23);
        let model = PcaModel::train(&vectors, &params).expect("PCAWR training should work");
        let transformed = model
            .transform(&vectors)
            .expect("PCAWR transform should work");
        assert_eq!(transformed.value_length(), 3);
        assert!(
            transformed
                .values()
                .as_primitive::<Float32Type>()
                .values()
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn box_muller_has_stable_prefix() {
        let values = deterministic_gaussian_matrix(2, 5).expect("generation should work");
        let mut rng = SmallRng::seed_from_u64(5);
        let mut expected = Vec::with_capacity(4);
        for _ in 0..2 {
            let a = rng.next_u64() >> 11;
            let b = rng.next_u64() >> 11;
            let u1 = (a + 1) as f64 * 2.0_f64.powi(-53);
            let u2 = b as f64 * 2.0_f64.powi(-53);
            let radius = (-2.0 * u1.ln()).sqrt();
            expected.push(radius * (TAU * u2).cos());
            expected.push(radius * (TAU * u2).sin());
        }
        for (actual, expected) in values.iter().zip(expected) {
            assert_relative_eq!(*actual, expected, epsilon = 1.0e-12);
        }
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
        for vectors in [&f16_vectors, &f64_vectors] {
            let model = PcaModel::train(vectors, &PcaParams::new(2)).expect("training should work");
            assert_eq!(
                model
                    .transform(vectors)
                    .expect("transform should work")
                    .value_type(),
                DataType::Float32
            );
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

    #[test]
    fn rejects_low_rank_and_small_memory_budget() {
        let low_rank = f32_vectors(
            vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0],
            3,
        );
        let error =
            PcaModel::train(&low_rank, &PcaParams::new(2)).expect_err("low-rank input must fail");
        assert!(error.to_string().contains("effective_rank=1"));

        let full_rank = f32_vectors(full_rank_values(8, 3), 3);
        let error = PcaModel::train(
            &full_rank,
            &PcaParams::new(2).with_max_training_memory_bytes(1),
        )
        .expect_err("small memory budget must fail");
        assert!(error.to_string().contains("M_stat="));
    }

    #[test]
    fn eigenvalue_threshold_handles_zero_and_negative_values() {
        let (values, rank, tau) = sanitize_eigenvalues(vec![4.0, 1.0e-12, -1.0e-12], 1)
            .expect("near-zero eigenvalues should be sanitized");
        assert_eq!(rank, 1);
        assert_eq!(values, vec![4.0, 0.0, 0.0]);
        assert_relative_eq!(tau, 4.0e-10);

        let error = sanitize_eigenvalues(vec![4.0, -1.0], 1)
            .expect_err("significant negative eigenvalue must fail");
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
        assert!(decoded.training_stats().is_none());
        assert_eq!(
            decoded
                .transform(&vectors)
                .expect("decoded transform must succeed"),
            expected
        );
    }

    #[test]
    fn whitening_rotation_model_round_trips() {
        let vectors = f32_vectors(full_rank_values(9, 4), 4);
        let params = PcaParams::new(3)
            .with_whitening(true)
            .with_random_rotation(true)
            .with_rotation_seed(29);
        let trained = PcaModel::train(&vectors, &params).expect("training must succeed");
        let decoded = decode_pca_model(&encode_pca_model(&trained).expect("encode must succeed"))
            .expect("decode must succeed");
        assert_eq!(decoded.transform_matrix(), trained.transform_matrix());
        assert_eq!(decoded.mean(), trained.mean());
        assert_eq!(
            decoded.transform(&vectors).expect("transform must succeed"),
            trained.transform(&vectors).expect("transform must succeed")
        );
        assert!(decoded.training_stats().is_none());
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

    #[test]
    fn same_seed_produces_same_fingerprint() {
        let vectors = f32_vectors(full_rank_values(9, 4), 4);
        let params = PcaParams::new(3)
            .with_random_rotation(true)
            .with_rotation_seed(41);
        let first = PcaModel::train(&vectors, &params).expect("training must succeed");
        let second = PcaModel::train(&vectors, &params).expect("training must succeed");
        assert_eq!(first.fingerprint(), second.fingerprint());
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
    #[case("")]
    #[case("-1")]
    #[case("abc")]
    #[case("4294967296")]
    #[tokio::test]
    async fn rejects_invalid_metadata_indices(#[case] metadata: &str) {
        let reader = BufferReader {
            buffers: Arc::new(Mutex::new(Vec::new())),
            schema: Schema::default(),
            fail_reads_with_io: false,
        };
        assert_corrupt(
            read_pca_model(&reader, metadata)
                .await
                .expect_err("invalid metadata must fail"),
            PCA_METADATA_KEY,
        );
    }

    #[tokio::test]
    async fn preserves_global_buffer_io_error_variant() {
        let reader = BufferReader {
            buffers: Arc::new(Mutex::new(Vec::new())),
            schema: Schema::default(),
            fail_reads_with_io: true,
        };
        let error = read_pca_model(&reader, "0")
            .await
            .expect_err("I/O errors must be propagated");
        assert!(
            matches!(&error, Error::IO { .. }),
            "expected IO, got {error:?}"
        );
        assert!(
            error
                .to_string()
                .contains("injected PCA global buffer read failure"),
            "propagated error must preserve its source: {error}"
        );
    }

    #[tokio::test]
    async fn global_buffer_helpers_round_trip_and_reject_missing_index() {
        let buffers = Arc::new(Mutex::new(vec![
            Bytes::from_static(b"zero"),
            Bytes::from_static(b"one"),
            Bytes::from_static(b"two"),
        ]));
        let model = persisted_model(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![1.0, 2.0, 3.0]);
        let mut writer = BufferWriter {
            buffers: buffers.clone(),
        };
        let index = write_pca_model(&mut writer, &model)
            .await
            .expect("write must succeed");
        assert_eq!(index, 3);
        let reader = BufferReader {
            buffers,
            schema: Schema::default(),
            fail_reads_with_io: false,
        };
        let decoded = read_pca_model(&reader, "3")
            .await
            .expect("metadata index 3 must load");
        assert_eq!(decoded.transform_matrix(), model.transform_matrix());
        assert_eq!(decoded.mean(), model.mean());
        read_pca_model(&reader, "03")
            .await
            .expect("any valid decimal u32 representation must load");

        assert_corrupt(
            read_pca_model(&reader, "4")
                .await
                .expect_err("missing buffer must fail"),
            "index=4",
        );
    }

    fn transform_kernels_under_test() -> Vec<TransformKernel> {
        [TransformKernel::Portable, TransformKernel::Neon]
            .into_iter()
            .filter(|kernel| kernel.is_available())
            .collect()
    }

    #[test]
    fn transform_kernels_match_scalar_loop() {
        for (input_dimension, output_dimension) in [
            (1usize, 1usize),
            (5, 3),
            (16, 4),
            (23, 5),
            (64, 7),
            (960, 9),
            (960, 256),
        ] {
            let input: Vec<f32> = (0..input_dimension)
                .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.25)
                .collect();
            let mean: Vec<f32> = (0..input_dimension)
                .map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.5)
                .collect();
            let matrix: Vec<f32> = (0..input_dimension * output_dimension)
                .map(|i| ((i * 11 % 17) as f32 - 8.0) * 0.125)
                .collect();
            let mut expected = Vec::new();
            transform_row_with(
                TransformKernel::Scalar,
                &input,
                &mean,
                &matrix,
                input_dimension,
                output_dimension,
                &mut expected,
            )
            .expect("scalar kernel must succeed");
            for kernel in transform_kernels_under_test() {
                let mut actual = Vec::new();
                transform_row_with(
                    kernel,
                    &input,
                    &mean,
                    &matrix,
                    input_dimension,
                    output_dimension,
                    &mut actual,
                )
                .unwrap_or_else(|error| panic!("{kernel:?} kernel must succeed: {error}"));
                for (row, (actual_value, expected_value)) in actual.iter().zip(&expected).enumerate()
                {
                    assert!(
                        (actual_value - expected_value).abs() <= 1.0e-4 * (1.0 + expected_value.abs()),
                        "{kernel:?} {input_dimension}x{output_dimension} row {row}: {actual_value} vs {expected_value}"
                    );
                }
            }
        }
    }

    #[test]
    fn transform_kernels_reject_non_finite_output() {
        let input = vec![1.0_f32; 24];
        let mean = vec![0.0_f32; 24];
        let mut matrix = vec![1.0_f32; 24 * 5];
        matrix[24 * 3 + 20] = f32::INFINITY;
        for kernel in std::iter::once(TransformKernel::Scalar).chain(transform_kernels_under_test())
        {
            let mut output = Vec::new();
            let error = transform_row_with(kernel, &input, &mean, &matrix, 24, 5, &mut output)
                .expect_err("non-finite output must fail");
            assert!(
                error.to_string().contains("output dimension 3"),
                "{kernel:?}: {error}"
            );
        }
    }

    #[test]
    fn selected_transform_kernel_is_available() {
        let kernel = TransformKernel::selected();
        assert!(kernel.is_available(), "{kernel:?}");
        if cfg!(target_arch = "aarch64") {
            assert_ne!(kernel, TransformKernel::Portable);
        } else {
            assert!(matches!(
                kernel,
                TransformKernel::Portable | TransformKernel::Scalar
            ));
        }
    }
}
