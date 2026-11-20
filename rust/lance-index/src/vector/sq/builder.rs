// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_schema::DataType;
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;
use serde::{Deserialize, Serialize};

use super::{ScalarQuantizer, storage::ScalarQuantizationMetadata};
use crate::vector::quantizer::QuantizerBuildParams;

#[derive(Debug, Clone)]
pub struct SQBuildParams {
    /// Number of bits of scaling range.
    pub num_bits: u16,

    /// Sample rate for training.
    pub sample_rate: usize,

    /// Trained bounds to reuse. A supplied model replaces SQ training and sampling.
    pub model: Option<SqModel>,
}

/// The vector coordinates used to train SQ bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SqInputSpace {
    /// Original coordinates, used for L2 and dot product.
    Raw,
    /// Unit-length coordinates, used for cosine.
    L2Normalized,
}

impl TryFrom<DistanceType> for SqInputSpace {
    type Error = Error;

    fn try_from(distance_type: DistanceType) -> Result<Self> {
        match distance_type {
            DistanceType::L2 | DistanceType::Dot => Ok(Self::Raw),
            DistanceType::Cosine => Ok(Self::L2Normalized),
            DistanceType::Hamming => Err(Error::invalid_input(
                "SQ does not support the hamming metric",
            )),
        }
    }
}

/// Trained SQ8 bounds and the input space they were trained on.
///
/// Cosine encodes normalized vectors. L2 and dot product encode raw vectors
/// and may share the same SQ model. A physical merge also requires shared IVF
/// centroids and a common distance metric.
#[derive(Debug, Clone)]
pub struct SqModel {
    metadata: ScalarQuantizationMetadata,
    input_space: SqInputSpace,
}

impl SqModel {
    /// Restore trained bounds and the input space they apply to.
    ///
    /// ```
    /// use lance_index::vector::sq::builder::{SqInputSpace, SqModel};
    /// use lance_index::vector::sq::storage::ScalarQuantizationMetadata;
    /// let model = SqModel::try_new(
    ///     ScalarQuantizationMetadata { dim: 32, num_bits: 8, bounds: -1.0..1.0 },
    ///     SqInputSpace::Raw,
    /// )?;
    /// assert_eq!(model.metadata().dim, 32);
    /// # Ok::<(), lance_core::Error>(())
    /// ```
    pub fn try_new(
        metadata: ScalarQuantizationMetadata,
        input_space: SqInputSpace,
    ) -> Result<Self> {
        if metadata.dim == 0 || metadata.dim > i32::MAX as usize {
            return Err(Error::invalid_input(format!(
                "SQ model dimension must be in 1..={}, got {}",
                i32::MAX,
                metadata.dim
            )));
        }
        if metadata.num_bits != 8 {
            return Err(Error::invalid_input(format!(
                "SQ model num_bits must be 8, got {}",
                metadata.num_bits
            )));
        }
        if !metadata.bounds.start.is_finite()
            || !metadata.bounds.end.is_finite()
            || metadata.bounds.start > metadata.bounds.end
            || !(metadata.bounds.end - metadata.bounds.start).is_finite()
        {
            return Err(Error::invalid_input(format!(
                "SQ model bounds must be finite and ordered with a finite range, got {:?}",
                metadata.bounds
            )));
        }
        Ok(Self {
            metadata,
            input_space,
        })
    }

    /// Return the dimension, bit width and bounds used to encode vectors.
    pub fn metadata(&self) -> &ScalarQuantizationMetadata {
        &self.metadata
    }

    /// Return the vector input space used to train the bounds.
    pub fn input_space(&self) -> SqInputSpace {
        self.input_space
    }

    /// Check that this model can encode vectors with the requested type, dimension, and settings.
    pub fn validate_for_build(
        &self,
        dimension: usize,
        element_type: &DataType,
        distance_type: DistanceType,
        num_bits: u16,
    ) -> Result<()> {
        let input_space = SqInputSpace::try_from(distance_type)?;
        if self.input_space != input_space {
            return Err(Error::invalid_input(format!(
                "SQ model was trained in the {:?} input space and cannot encode vectors for {distance_type}, which requires {input_space:?}",
                self.input_space,
            )));
        }
        if !matches!(
            element_type,
            DataType::Float16 | DataType::Float32 | DataType::Float64
        ) {
            return Err(Error::invalid_input(format!(
                "SQ requires floating-point vectors, got {element_type}"
            )));
        }
        if self.metadata.dim != dimension || self.metadata.num_bits != num_bits {
            return Err(Error::invalid_input(format!(
                "SQ model dimension {} and num_bits {} do not match requested dimension {dimension} and num_bits {num_bits}",
                self.metadata.dim, self.metadata.num_bits,
            )));
        }
        Ok(())
    }

    /// Create the scalar quantizer with these trained bounds.
    pub fn quantizer(&self) -> ScalarQuantizer {
        ScalarQuantizer::with_bounds(
            self.metadata.num_bits,
            self.metadata.dim,
            self.metadata.bounds.clone(),
        )
    }
}

impl From<&SQBuildParams> for crate::pb::vector_index_details::ScalarQuantization {
    fn from(params: &SQBuildParams) -> Self {
        Self {
            num_bits: params.num_bits as u32,
        }
    }
}

impl Default for SQBuildParams {
    fn default() -> Self {
        Self {
            num_bits: 8,
            sample_rate: 256,
            model: None,
        }
    }
}

impl QuantizerBuildParams for SQBuildParams {
    fn sample_size(&self) -> usize {
        self.sample_rate * 2usize.pow(self.num_bits as u32)
    }

    fn try_sample_size(&self) -> Result<usize> {
        1usize
            .checked_shl(u32::from(self.num_bits))
            .and_then(|codes| self.sample_rate.checked_mul(codes))
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "SQ training sample size overflows usize: sample_rate={}, num_bits={}",
                    self.sample_rate, self.num_bits
                ))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sq_sample_size_rejects_overflow() {
        let params = SQBuildParams {
            num_bits: 8,
            sample_rate: usize::MAX,
            ..Default::default()
        };
        let error = params.try_sample_size().unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("sample size overflows"));
    }
}
