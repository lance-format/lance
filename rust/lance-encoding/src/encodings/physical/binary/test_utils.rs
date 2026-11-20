// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::{BinaryMiniBlockDecompressor, BinaryMiniBlockLayout};
use crate::{
    encodings::physical::{
        block::{CompressionConfig, CompressionScheme},
        general::FixedWidthGeneralBlockDecompressor,
        value::ValueDecompressor,
    },
    format::metadata,
};

// Noop is a runtime codec, not a serializable General compression scheme.
// Construct it directly for feature-independent admission and lifetime tests;
// serialized LZ4/Zstd fixtures must still use the real writer and factory.
pub fn uncompressed_general_offsets(bits: u8) -> BinaryMiniBlockDecompressor {
    assert!(matches!(bits, 32 | 64));
    BinaryMiniBlockDecompressor {
        layout: BinaryMiniBlockLayout::Generic {
            bits_per_offset: bits,
            offsets: Box::new(FixedWidthGeneralBlockDecompressor::new(
                Box::new(ValueDecompressor::from_flat(&metadata::Flat {
                    bits_per_value: u64::from(bits),
                    data: None,
                })),
                CompressionConfig::new(CompressionScheme::None, None),
                u64::from(bits),
            )),
            offsets_have_payload: true,
        },
    }
}
