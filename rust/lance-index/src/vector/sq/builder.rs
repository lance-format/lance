// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::storage::ScalarQuantizationMetadata;
use crate::vector::quantizer::QuantizerBuildParams;
use lance_core::{Error, Result};

#[derive(Debug, Clone)]
pub struct SQBuildParams {
    /// Number of bits of scaling range. Currently only 8 is supported.
    pub num_bits: u16,

    /// Sample rate for training (default 256). Ignored when a model is supplied.
    pub sample_rate: usize,

    /// Shared quantizer trained in the same vector space as the build (normalized
    /// for cosine). When present, no training rows are sampled.
    pub model: Option<ScalarQuantizationMetadata>,
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
        if self.model.is_some() {
            0
        } else {
            self.sample_rate * 2usize.pow(self.num_bits as u32)
        }
    }
}

impl SQBuildParams {
    /// Validate SQ8 settings and a supplied model against the indexed vector dimension.
    /// A supplied model must have matching dimensions and bits, and valid finite bounds.
    ///
    /// ```
    /// use lance_index::vector::sq::builder::SQBuildParams;
    /// SQBuildParams::default().validate(128)?;
    /// # Ok::<(), lance_core::Error>(())
    /// ```
    pub fn validate(&self, dimension: usize) -> Result<()> {
        if self.num_bits != 8 {
            return Err(Error::invalid_input(format!(
                "SQ num_bits must be 8, got {}",
                self.num_bits
            )));
        }
        if let Some(model) = &self.model {
            model.validate_for_build()?;
            if model.dim != dimension || model.num_bits != self.num_bits {
                return Err(Error::invalid_input(format!(
                    "SQ model dimension {} and num_bits {} do not match requested dimension {dimension} and num_bits {}",
                    model.dim, model.num_bits, self.num_bits
                )));
            }
        }
        Ok(())
    }
}
