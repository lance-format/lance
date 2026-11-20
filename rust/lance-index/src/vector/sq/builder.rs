// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::vector::quantizer::QuantizerBuildParams;
use lance_core::{Error, Result};

#[derive(Debug, Clone)]
pub struct SQBuildParams {
    /// Number of bits of scaling range.
    pub num_bits: u16,

    /// Sample rate for training.
    pub sample_rate: usize,
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
        }
    }
}

impl QuantizerBuildParams for SQBuildParams {
    fn sample_size(&self) -> usize {
        self.try_sample_size()
            .expect("SQ training sample size must fit in usize")
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
        };
        let error = params.try_sample_size().unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("sample size overflows"));
    }
}
