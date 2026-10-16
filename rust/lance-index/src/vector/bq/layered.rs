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
    /// Resolve a sign/high/low layout for a total width from 3 through 9 bits.
    /// High uses 1, 2 or 4 bits; low holds the remaining full-code bits.
    pub fn try_new(num_bits: u8) -> Result<Self> {
        let (high_bits, low_bits) = match num_bits {
            3 => (1, 1),
            4 => (2, 1),
            5 => (2, 2),
            6 => (4, 1),
            7 => (4, 2),
            8 => (4, 3),
            9 => (4, 4),
            _ => {
                return Err(Error::invalid_input(format!(
                    "IVF_RQ layered requires num_bits in 3..=9, got {num_bits}"
                )));
            }
        };
        Ok(Self {
            high_bits,
            low_bits,
        })
    }
}
