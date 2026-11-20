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
    /// Normalized offsets and selected byte payload. Payload size depends on input.
    VariableWidth { bits_per_offset: u8 },
    /// A validity bitmap alongside an equally sized child.
    Nullable(Box<Self>),
    /// Each parent contains exactly `dimension` child values.
    FixedSizeList { dimension: u64, child: Box<Self> },
}

impl MiniBlockOutputLayout {
    /// Exact new payload bytes for a fixed-size shape, including child bitmaps.
    /// Variable-width shapes return `NotSupported`: their byte lengths require a
    /// validated codec range plan. Checked geometry prevents allocation overflow.
    pub fn payload_bytes(&self, values: u64) -> Result<usize> {
        match self {
            Self::VariableWidth { .. } => Err(Error::not_supported(
                "variable-width output needs the validated range payload size",
            )),
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

    fn offset_bytes(bits: u8, values: u64) -> Result<usize> {
        if !matches!(bits, 32 | 64) {
            return Err(Error::invalid_input(
                "MiniBlock offsets require 32 or 64 bits",
            ));
        }
        values
            .checked_add(1)
            .and_then(|n| n.checked_mul(u64::from(bits / 8)))
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| Error::invalid_input("MiniBlock offset size overflows"))
    }

    pub(crate) fn validate_range_bytes(&self, values: u64, bytes: usize) -> Result<()> {
        match self {
            Self::VariableWidth { bits_per_offset } => {
                let offsets = Self::offset_bytes(*bits_per_offset, values)?;
                let data = bytes.checked_sub(offsets).ok_or_else(|| {
                    Error::invalid_input("MiniBlock value plan omits required offsets")
                })?;
                let max = if *bits_per_offset == 32 {
                    i32::MAX as u64
                } else {
                    i64::MAX as u64
                };
                if data as u64 > max {
                    return Err(Error::invalid_input(
                        "MiniBlock binary payload exceeds signed offset range",
                    ));
                }
                for size in [offsets, data] {
                    std::alloc::Layout::from_size_align(size, arrow_buffer::alloc::ALIGNMENT)
                        .map_err(|_| {
                            Error::invalid_input("MiniBlock binary output exceeds addressable size")
                        })?;
                }
                Ok(())
            }
            _ if self.payload_bytes(values)? == bytes => Ok(()),
            _ => Err(Error::invalid_input(
                "MiniBlock value plan disagrees with concat layout",
            )),
        }
    }

    pub(crate) fn concat_payload_bytes(
        &self,
        parts: impl IntoIterator<Item = (u64, usize)>,
    ) -> Result<usize> {
        let mut values = 0_u64;
        let mut data_bytes = 0_usize;
        for (count, bytes) in parts {
            self.validate_range_bytes(count, bytes)?;
            values = values
                .checked_add(count)
                .ok_or_else(|| Error::invalid_input("MiniBlock concat count overflows"))?;
            if let Self::VariableWidth { bits_per_offset } = self {
                data_bytes = data_bytes
                    .checked_add(bytes - Self::offset_bytes(*bits_per_offset, count)?)
                    .ok_or_else(|| Error::invalid_input("MiniBlock concat payload overflows"))?;
            }
        }
        if let Self::VariableWidth { bits_per_offset } = self {
            let bytes = data_bytes
                .checked_add(Self::offset_bytes(*bits_per_offset, values)?)
                .ok_or_else(|| Error::invalid_input("MiniBlock concat output overflows"))?;
            self.validate_range_bytes(values, bytes)?;
            Ok(bytes)
        } else {
            self.payload_bytes(values)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    fn variable_concat_counts_one_final_leading_offset(#[values(32, 64)] bits: u8) {
        let width = usize::from(bits / 8);
        let layout = MiniBlockOutputLayout::VariableWidth {
            bits_per_offset: bits,
        };
        assert_eq!(
            layout
                .concat_payload_bytes([(2, 3 * width + 3), (0, width), (1, 2 * width + 9)])
                .unwrap(),
            4 * width + 12
        );
        assert_eq!(
            layout
                .concat_payload_bytes([(0, width), (0, width)])
                .unwrap(),
            width
        );
        assert!(matches!(
            layout.payload_bytes(2),
            Err(Error::NotSupported { .. })
        ));
    }

    #[rstest]
    #[case::missing_offset(0)]
    #[case::signed_overflow(1)]
    #[case::count_overflow(2)]
    #[case::invalid_width(3)]
    fn invalid_binary_concat_is_rejected_before_materialization(#[case] fault: u8) {
        let layout = MiniBlockOutputLayout::VariableWidth {
            bits_per_offset: if fault == 3 { 8 } else { 32 },
        };
        let parts = match fault {
            0 => vec![(1, 7)],
            // Each child fits i32 offsets, but their complete concatenation does not.
            1 => vec![(1, 8 + i32::MAX as usize), (1, 9)],
            2 => vec![(u64::MAX, 0)],
            _ => vec![(1, 8)],
        };
        let error = layout.concat_payload_bytes(parts).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        let expected = ["required offsets", "signed offset", "overflows", "32 or 64"];
        assert!(
            error.to_string().contains(expected[fault as usize]),
            "{error}"
        );
    }

    #[rstest]
    #[case::bits(MiniBlockOutputLayout::FixedWidth { bits_per_value: 1 }, vec![(3, 1), (6, 1)], 2)]
    #[case::nullable(MiniBlockOutputLayout::Nullable(Box::new(MiniBlockOutputLayout::FixedWidth { bits_per_value: 32 })), vec![(1, 5), (2, 9)], 13)]
    #[case::vectors(MiniBlockOutputLayout::FixedSizeList { dimension: 3, child: Box::new(MiniBlockOutputLayout::FixedWidth { bits_per_value: 32 }) }, vec![(1, 12), (2, 24)], 36)]
    fn fixed_concat_layout_stays_exact(
        #[case] layout: MiniBlockOutputLayout,
        #[case] parts: Vec<(u64, usize)>,
        #[case] bytes: usize,
    ) {
        assert_eq!(layout.concat_payload_bytes(parts).unwrap(), bytes);
        let error = layout.concat_payload_bytes([(1, 123)]).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("disagrees"));
    }
}
