// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::{Error, Result};

/// Physical output shape shared by all admitted ranges from one MiniBlock codec.
/// This describes payload only, not typed Arrow or rep/def reconstruction.
///
/// ```
/// use lance_encoding::compression::MiniBlockOutputLayout;
/// let layout = MiniBlockOutputLayout::FixedSizeList {
///     dimension: 3,
///     child: Box::new(MiniBlockOutputLayout::Nullable(Box::new(
///         MiniBlockOutputLayout::FixedWidth { bits_per_value: 32 },
///     ))),
/// };
/// assert_eq!(layout.payload_bytes(2).unwrap(), 25); // Six f32s and one child bitmap byte.
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MiniBlockOutputLayout {
    /// Packed values; padding is discarded when ranges are concatenated.
    FixedWidth { bits_per_value: u64 },
    /// A validity bitmap alongside an equally sized child.
    Nullable(Box<Self>),
    /// Each parent contains exactly `dimension` child values.
    FixedSizeList { dimension: u64, child: Box<Self> },
}

impl MiniBlockOutputLayout {
    /// Exact new payload bytes for this many root values, including child bitmaps.
    /// Checked geometry prevents overflow before an allocation is admitted.
    pub fn payload_bytes(&self, values: u64) -> Result<usize> {
        match self {
            Self::FixedWidth { bits_per_value } => {
                if *bits_per_value == 0 {
                    return Err(Error::not_supported(
                        "zero-width output needs typed layout planning",
                    ));
                }
                values
                    .checked_mul(*bits_per_value)
                    .and_then(|bits| usize::try_from(bits).ok())
                    .map(|bits| bits.div_ceil(8))
                    .ok_or_else(|| Error::invalid_input("MiniBlock output bit count overflows"))
            }
            Self::Nullable(child) => {
                let bitmap = usize::try_from(values)
                    .map_err(|_| Error::invalid_input("MiniBlock bitmap count overflows"))?
                    .div_ceil(8);
                child
                    .payload_bytes(values)?
                    .checked_add(bitmap)
                    .ok_or_else(|| Error::invalid_input("MiniBlock nullable output size overflows"))
            }
            Self::FixedSizeList { dimension, child } => {
                let count = values
                    .checked_mul(*dimension)
                    .filter(|_| *dimension > 0)
                    .ok_or_else(|| {
                        Error::invalid_input("MiniBlock list dimension or child count is invalid")
                    })?;
                child.payload_bytes(count)
            }
        }
    }
}
