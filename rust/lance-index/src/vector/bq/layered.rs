// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Fixed prefix layouts for opt-in layered RaBitQ indices.

use lance_core::{Error, Result};

/// Plane widths for splitting a native full-precision code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RQLayout {
    pub high_bits: u8,
    pub low_bits: u8,
}

impl RQLayout {
    /// Resolve the only supported layered layouts: 1+2+2, 1+4+2 and 1+4+4.
    pub fn try_new(num_bits: u8) -> Result<Self> {
        let (high_bits, low_bits) = match num_bits {
            5 => (2, 2),
            7 => (4, 2),
            9 => (4, 4),
            _ => {
                return Err(Error::invalid_input(format!(
                    "IVF_RQ layered requires num_bits=5, 7 or 9, got {num_bits}"
                )));
            }
        };
        Ok(Self {
            high_bits,
            low_bits,
        })
    }
}

