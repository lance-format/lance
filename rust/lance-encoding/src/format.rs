// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

/// Protobuf definitions for encodings
///
/// These are the messages used for describing encoding in the 2.0 format
pub mod pb {
    #![allow(clippy::all)]
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(unused)]
    #![allow(improper_ctypes)]
    #![allow(clippy::upper_case_acronyms)]
    #![allow(clippy::use_self)]
    include!(concat!(env!("OUT_DIR"), "/lance.encodings.rs"));
}

/// Protobuf definitions for encodings21
///
/// These are the messages used for describing encoding in the 2.1 format
/// and the stable 2.2 format.
pub mod pb21 {
    #![allow(clippy::all)]
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(unused)]
    #![allow(improper_ctypes)]
    #![allow(clippy::upper_case_acronyms)]
    #![allow(clippy::use_self)]
    include!(concat!(env!("OUT_DIR"), "/lance.encodings21.rs"));
}

/// Metadata for the unstable 2.3 planar page format.
pub mod pb23 {
    #![allow(clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/lance.encodings23.rs"));
}

pub mod metadata;
pub mod planar;
mod stable;

use pb::{
    ArrayEncoding, Binary, Bitpacked, BitpackedForNonNeg, Block, Dictionary, FixedSizeBinary,
    FixedSizeList, Flat, Fsst, InlineBitpacking, Nullable, OutOfLineBitpacking, PackedStruct,
    PackedStructFixedWidthMiniBlock, Rle, Variable,
    array_encoding::ArrayEncoding as ArrayEncodingEnum,
    buffer::BufferType,
    nullable::{AllNull, NoNull, Nullability, SomeNull},
};

use crate::{encodings::physical::block::CompressionConfig, repdef::DefinitionInterpretation};

use self::pb::Constant;
use lance_core::Result;

// Utility functions for creating complex protobuf objects
pub struct ProtobufUtils {}

impl ProtobufUtils {
    pub fn constant(value: Vec<u8>) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::Constant(Constant {
                value: value.into(),
            })),
        }
    }

    pub fn basic_all_null_encoding() -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::Nullable(Box::new(Nullable {
                nullability: Some(Nullability::AllNulls(AllNull {})),
            }))),
        }
    }

    pub fn basic_some_null_encoding(
        validity: ArrayEncoding,
        values: ArrayEncoding,
    ) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::Nullable(Box::new(Nullable {
                nullability: Some(Nullability::SomeNulls(Box::new(SomeNull {
                    validity: Some(Box::new(validity)),
                    values: Some(Box::new(values)),
                }))),
            }))),
        }
    }

    pub fn basic_no_null_encoding(values: ArrayEncoding) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::Nullable(Box::new(Nullable {
                nullability: Some(Nullability::NoNulls(Box::new(NoNull {
                    values: Some(Box::new(values)),
                }))),
            }))),
        }
    }

    pub fn block(scheme: &str) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::Block(Block {
                scheme: scheme.to_string(),
            })),
        }
    }

    pub fn flat_encoding(
        bits_per_value: u64,
        buffer_index: u32,
        compression: Option<CompressionConfig>,
    ) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::Flat(Flat {
                bits_per_value,
                buffer: Some(pb::Buffer {
                    buffer_index,
                    buffer_type: BufferType::Page as i32,
                }),
                compression: compression.map(|compression_config| pb::Compression {
                    scheme: compression_config.scheme.to_string(),
                    level: compression_config.level,
                }),
            })),
        }
    }

    pub fn fsl_encoding(dimension: u64, items: ArrayEncoding, has_validity: bool) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::FixedSizeList(Box::new(FixedSizeList {
                dimension: dimension.try_into().unwrap(),
                items: Some(Box::new(items)),
                has_validity,
            }))),
        }
    }

    pub fn bitpacked_encoding(
        compressed_bits_per_value: u64,
        uncompressed_bits_per_value: u64,
        buffer_index: u32,
        signed: bool,
    ) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::Bitpacked(Bitpacked {
                compressed_bits_per_value,
                buffer: Some(pb::Buffer {
                    buffer_index,
                    buffer_type: BufferType::Page as i32,
                }),
                uncompressed_bits_per_value,
                signed,
            })),
        }
    }

    pub fn bitpacked_for_non_neg_encoding(
        compressed_bits_per_value: u64,
        uncompressed_bits_per_value: u64,
        buffer_index: u32,
    ) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::BitpackedForNonNeg(BitpackedForNonNeg {
                compressed_bits_per_value,
                buffer: Some(pb::Buffer {
                    buffer_index,
                    buffer_type: BufferType::Page as i32,
                }),
                uncompressed_bits_per_value,
            })),
        }
    }
    pub fn inline_bitpacking(uncompressed_bits_per_value: u64) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::InlineBitpacking(InlineBitpacking {
                uncompressed_bits_per_value,
            })),
        }
    }
    pub fn out_of_line_bitpacking(
        uncompressed_bits_per_value: u64,
        compressed_bits_per_value: u64,
    ) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::OutOfLineBitpacking(
                OutOfLineBitpacking {
                    uncompressed_bits_per_value,
                    compressed_bits_per_value,
                },
            )),
        }
    }

    pub fn variable(bits_per_offset: u8) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::Variable(Variable {
                bits_per_offset: bits_per_offset as u32,
            })),
        }
    }

    // Construct a `FsstMiniBlock` ArrayEncoding, the inner `binary_mini_block` encoding is actually
    // not used and `FsstMiniBlockDecompressor` constructs a `binary_mini_block` in a `hard-coded` fashion.
    // This can be an optimization later.
    pub fn fsst(data: ArrayEncoding, symbol_table: Vec<u8>) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::Fsst(Box::new(Fsst {
                binary: Some(Box::new(data)),
                symbol_table: symbol_table.into(),
            }))),
        }
    }

    pub fn rle(bits_per_value: u64) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::Rle(Rle { bits_per_value })),
        }
    }

    pub fn byte_stream_split(bits_per_value: u64) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::ByteStreamSplit(pb::ByteStreamSplit {
                bits_per_value,
            })),
        }
    }

    pub fn general_mini_block(
        inner: ArrayEncoding,
        compression: CompressionConfig,
    ) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::GeneralMiniBlock(Box::new(
                pb::GeneralMiniBlock {
                    inner: Some(Box::new(inner)),
                    compression: Some(pb::Compression {
                        scheme: compression.scheme.to_string(),
                        level: compression.level,
                    }),
                },
            ))),
        }
    }

    pub fn packed_struct(
        child_encodings: Vec<ArrayEncoding>,
        packed_buffer_index: u32,
    ) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::PackedStruct(PackedStruct {
                inner: child_encodings,
                buffer: Some(pb::Buffer {
                    buffer_index: packed_buffer_index,
                    buffer_type: BufferType::Page as i32,
                }),
            })),
        }
    }

    pub fn packed_struct_fixed_width_mini_block(
        data: ArrayEncoding,
        bits_per_values: Vec<u32>,
    ) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::PackedStructFixedWidthMiniBlock(
                Box::new(PackedStructFixedWidthMiniBlock {
                    flat: Some(Box::new(data)),
                    bits_per_values,
                }),
            )),
        }
    }

    pub fn binary(
        indices_encoding: ArrayEncoding,
        bytes_encoding: ArrayEncoding,
        null_adjustment: u64,
    ) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::Binary(Box::new(Binary {
                bytes: Some(Box::new(bytes_encoding)),
                indices: Some(Box::new(indices_encoding)),
                null_adjustment,
            }))),
        }
    }

    pub fn dict_encoding(
        indices: ArrayEncoding,
        items: ArrayEncoding,
        num_items: u32,
    ) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::Dictionary(Box::new(Dictionary {
                indices: Some(Box::new(indices)),
                items: Some(Box::new(items)),
                num_dictionary_items: num_items,
            }))),
        }
    }

    pub fn fixed_size_binary(data: ArrayEncoding, byte_width: u32) -> ArrayEncoding {
        ArrayEncoding {
            array_encoding: Some(ArrayEncodingEnum::FixedSizeBinary(Box::new(
                FixedSizeBinary {
                    bytes: Some(Box::new(data)),
                    byte_width,
                },
            ))),
        }
    }
}

/// Constructors for compression trees and structural execution descriptions.
pub struct EncodingUtils {}

impl EncodingUtils {
    pub fn flat(
        bits_per_value: u64,
        values_compression: Option<metadata::BufferCompression>,
    ) -> metadata::CompressiveEncoding {
        metadata::CompressiveEncoding {
            compression: Some(metadata::compressive_encoding::Compression::Flat(
                metadata::Flat {
                    bits_per_value,
                    data: values_compression,
                },
            )),
        }
    }

    pub fn constant(value: Option<bytes::Bytes>) -> metadata::CompressiveEncoding {
        metadata::CompressiveEncoding {
            compression: Some(metadata::compressive_encoding::Compression::Constant(
                metadata::Constant { value },
            )),
        }
    }

    pub fn fsl(
        items_per_value: u64,
        has_validity: bool,
        values: metadata::CompressiveEncoding,
    ) -> metadata::CompressiveEncoding {
        metadata::CompressiveEncoding {
            compression: Some(metadata::compressive_encoding::Compression::FixedSizeList(
                Box::new(metadata::FixedSizeList {
                    items_per_value,
                    has_validity,
                    values: Some(Box::new(values)),
                }),
            )),
        }
    }

    pub fn variable(
        offsets_desc: metadata::CompressiveEncoding,
        values_compression: Option<metadata::BufferCompression>,
    ) -> metadata::CompressiveEncoding {
        metadata::CompressiveEncoding {
            compression: Some(metadata::compressive_encoding::Compression::Variable(
                Box::new(metadata::Variable {
                    offsets: Some(Box::new(offsets_desc)),
                    values: values_compression,
                }),
            )),
        }
    }

    pub fn inline_bitpacking(
        uncompressed_bits_per_value: u64,
        values_compression: Option<metadata::BufferCompression>,
    ) -> metadata::CompressiveEncoding {
        metadata::CompressiveEncoding {
            compression: Some(
                metadata::compressive_encoding::Compression::InlineBitpacking(
                    metadata::InlineBitpacking {
                        uncompressed_bits_per_value,
                        values: values_compression,
                    },
                ),
            ),
        }
    }

    pub fn out_of_line_bitpacking(
        uncompressed_bits_per_value: u64,
        values: metadata::CompressiveEncoding,
    ) -> metadata::CompressiveEncoding {
        metadata::CompressiveEncoding {
            compression: Some(
                metadata::compressive_encoding::Compression::OutOfLineBitpacking(Box::new(
                    metadata::OutOfLineBitpacking {
                        uncompressed_bits_per_value,
                        values: Some(Box::new(values)),
                    },
                )),
            ),
        }
    }

    pub fn buffer_compression(
        compression: CompressionConfig,
    ) -> Result<metadata::BufferCompression> {
        Ok(metadata::BufferCompression {
            scheme: metadata::CompressionScheme::try_from(compression.scheme)? as i32,
            level: compression.level,
        })
    }

    pub fn wrapped(
        compression: CompressionConfig,
        values: metadata::CompressiveEncoding,
    ) -> Result<metadata::CompressiveEncoding> {
        Ok(metadata::CompressiveEncoding {
            compression: Some(metadata::compressive_encoding::Compression::General(
                Box::new(metadata::General {
                    compression: Some(Self::buffer_compression(compression)?),
                    values: Some(Box::new(values)),
                }),
            )),
        })
    }

    pub fn rle(
        values: metadata::CompressiveEncoding,
        run_lengths: metadata::CompressiveEncoding,
    ) -> metadata::CompressiveEncoding {
        metadata::CompressiveEncoding {
            compression: Some(metadata::compressive_encoding::Compression::Rle(Box::new(
                metadata::Rle {
                    values: Some(Box::new(values)),
                    run_lengths: Some(Box::new(run_lengths)),
                },
            ))),
        }
    }

    pub fn byte_stream_split(
        values: metadata::CompressiveEncoding,
    ) -> metadata::CompressiveEncoding {
        metadata::CompressiveEncoding {
            compression: Some(
                metadata::compressive_encoding::Compression::ByteStreamSplit(Box::new(
                    metadata::ByteStreamSplit {
                        values: Some(Box::new(values)),
                    },
                )),
            ),
        }
    }

    pub fn fsst(
        data: metadata::CompressiveEncoding,
        symbol_table: Vec<u8>,
    ) -> metadata::CompressiveEncoding {
        metadata::CompressiveEncoding {
            compression: Some(metadata::compressive_encoding::Compression::Fsst(Box::new(
                metadata::Fsst {
                    symbol_table: symbol_table.into(),
                    values: Some(Box::new(data)),
                },
            ))),
        }
    }

    fn def_inter_to_repdef_layer(def: DefinitionInterpretation) -> i32 {
        match def {
            DefinitionInterpretation::AllValidItem => {
                metadata::RepDefLayer::RepdefAllValidItem as i32
            }
            DefinitionInterpretation::AllValidList => {
                metadata::RepDefLayer::RepdefAllValidList as i32
            }
            DefinitionInterpretation::NullableItem => {
                metadata::RepDefLayer::RepdefNullableItem as i32
            }
            DefinitionInterpretation::NullableList => {
                metadata::RepDefLayer::RepdefNullableList as i32
            }
            DefinitionInterpretation::EmptyableList => {
                metadata::RepDefLayer::RepdefEmptyableList as i32
            }
            DefinitionInterpretation::NullableAndEmptyableList => {
                metadata::RepDefLayer::RepdefNullAndEmptyList as i32
            }
        }
    }

    pub fn repdef_layer_to_def_interp(layer: i32) -> DefinitionInterpretation {
        let layer = metadata::RepDefLayer::try_from(layer).unwrap();
        match layer {
            metadata::RepDefLayer::RepdefAllValidItem => DefinitionInterpretation::AllValidItem,
            metadata::RepDefLayer::RepdefAllValidList => DefinitionInterpretation::AllValidList,
            metadata::RepDefLayer::RepdefNullableItem => DefinitionInterpretation::NullableItem,
            metadata::RepDefLayer::RepdefNullableList => DefinitionInterpretation::NullableList,
            metadata::RepDefLayer::RepdefEmptyableList => DefinitionInterpretation::EmptyableList,
            metadata::RepDefLayer::RepdefNullAndEmptyList => {
                DefinitionInterpretation::NullableAndEmptyableList
            }
            metadata::RepDefLayer::RepdefUnspecified => {
                panic!("Unspecified repdef layer")
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn miniblock_layout(
        rep_encoding: Option<metadata::CompressiveEncoding>,
        def_encoding: Option<metadata::CompressiveEncoding>,
        value_encoding: metadata::CompressiveEncoding,
        repetition_index_depth: u32,
        num_buffers: u64,
        dictionary_encoding: Option<(metadata::CompressiveEncoding, u64)>,
        def_meaning: &[DefinitionInterpretation],
        num_items: u64,
        has_large_chunk: bool,
    ) -> metadata::PageLayout {
        assert!(!def_meaning.is_empty());
        let (dictionary, num_dictionary_items) = dictionary_encoding
            .map(|(d, i)| (Some(d), i))
            .unwrap_or((None, 0));
        metadata::PageLayout {
            layout: Some(metadata::page_layout::Layout::MiniBlockLayout(
                metadata::MiniBlockLayout {
                    def_compression: def_encoding,
                    rep_compression: rep_encoding,
                    value_compression: Some(value_encoding),
                    repetition_index_depth,
                    num_buffers,
                    dictionary,
                    num_dictionary_items,
                    layers: def_meaning
                        .iter()
                        .map(|&def| Self::def_inter_to_repdef_layer(def))
                        .collect(),
                    num_items,
                    has_large_chunk,
                },
            )),
        }
    }

    fn full_zip_layout(
        bits_rep: u8,
        bits_def: u8,
        details: metadata::full_zip_layout::Details,
        value_encoding: metadata::CompressiveEncoding,
        def_meaning: &[DefinitionInterpretation],
        num_items: u32,
        num_visible_items: u32,
    ) -> metadata::PageLayout {
        metadata::PageLayout {
            layout: Some(metadata::page_layout::Layout::FullZipLayout(
                metadata::FullZipLayout {
                    bits_rep: bits_rep as u32,
                    bits_def: bits_def as u32,
                    details: Some(details),
                    value_compression: Some(value_encoding),
                    num_items,
                    num_visible_items,
                    layers: def_meaning
                        .iter()
                        .map(|&def| Self::def_inter_to_repdef_layer(def))
                        .collect(),
                },
            )),
        }
    }

    pub fn fixed_full_zip_layout(
        bits_rep: u8,
        bits_def: u8,
        bits_per_value: u32,
        value_encoding: metadata::CompressiveEncoding,
        def_meaning: &[DefinitionInterpretation],
        num_items: u32,
        num_visible_items: u32,
    ) -> metadata::PageLayout {
        Self::full_zip_layout(
            bits_rep,
            bits_def,
            metadata::full_zip_layout::Details::BitsPerValue(bits_per_value),
            value_encoding,
            def_meaning,
            num_items,
            num_visible_items,
        )
    }

    pub fn variable_full_zip_layout(
        bits_rep: u8,
        bits_def: u8,
        bits_per_offset: u32,
        value_encoding: metadata::CompressiveEncoding,
        def_meaning: &[DefinitionInterpretation],
        num_items: u32,
        num_visible_items: u32,
    ) -> metadata::PageLayout {
        Self::full_zip_layout(
            bits_rep,
            bits_def,
            metadata::full_zip_layout::Details::BitsPerOffset(bits_per_offset),
            value_encoding,
            def_meaning,
            num_items,
            num_visible_items,
        )
    }

    pub fn blob_layout(
        inner_layout: metadata::PageLayout,
        def_meaning: &[DefinitionInterpretation],
    ) -> metadata::PageLayout {
        metadata::PageLayout {
            layout: Some(metadata::page_layout::Layout::BlobLayout(Box::new(
                metadata::BlobLayout {
                    inner_layout: Some(Box::new(inner_layout)),
                    layers: def_meaning
                        .iter()
                        .map(|&def| Self::def_inter_to_repdef_layer(def))
                        .collect(),
                },
            ))),
        }
    }

    pub fn range(
        uncompressed_bits_per_value: u64,
        start: u64,
        step: u64,
    ) -> metadata::CompressiveEncoding {
        metadata::CompressiveEncoding {
            compression: Some(metadata::compressive_encoding::Compression::Range(
                metadata::Range {
                    uncompressed_bits_per_value,
                    start,
                    step,
                },
            )),
        }
    }

    pub fn delta(
        uncompressed_bits_per_value: u64,
        base: u64,
        deltas: metadata::CompressiveEncoding,
    ) -> metadata::CompressiveEncoding {
        metadata::CompressiveEncoding {
            compression: Some(metadata::compressive_encoding::Compression::Delta(
                Box::new(metadata::Delta {
                    uncompressed_bits_per_value,
                    base,
                    deltas: Some(Box::new(deltas)),
                }),
            )),
        }
    }

    pub fn alp(bits_per_value: u32) -> metadata::CompressiveEncoding {
        metadata::CompressiveEncoding {
            compression: Some(metadata::compressive_encoding::Compression::Alp(
                metadata::Alp { bits_per_value },
            )),
        }
    }

    pub fn dictionary(
        indices: metadata::CompressiveEncoding,
        items: metadata::CompressiveEncoding,
        num_dictionary_items: u32,
    ) -> metadata::CompressiveEncoding {
        metadata::CompressiveEncoding {
            compression: Some(metadata::compressive_encoding::Compression::Dictionary(
                Box::new(metadata::Dictionary {
                    indices: Some(Box::new(indices)),
                    items: Some(Box::new(items)),
                    num_dictionary_items,
                }),
            )),
        }
    }

    pub fn constant_layout(
        def_meaning: &[DefinitionInterpretation],
        inline_value: Option<Vec<u8>>,
    ) -> metadata::PageLayout {
        metadata::PageLayout {
            layout: Some(metadata::page_layout::Layout::ConstantLayout(
                metadata::ConstantLayout {
                    inline_value: inline_value.map(bytes::Bytes::from),
                    rep_compression: None,
                    def_compression: None,
                    num_rep_values: 0,
                    num_def_values: 0,
                    layers: def_meaning
                        .iter()
                        .map(|&def| Self::def_inter_to_repdef_layer(def))
                        .collect(),
                },
            )),
        }
    }

    pub fn compressed_all_null_constant_layout(
        def_meaning: &[DefinitionInterpretation],
        rep_compression: Option<metadata::CompressiveEncoding>,
        def_compression: Option<metadata::CompressiveEncoding>,
        num_rep_values: u64,
        num_def_values: u64,
    ) -> metadata::PageLayout {
        metadata::PageLayout {
            layout: Some(metadata::page_layout::Layout::ConstantLayout(
                metadata::ConstantLayout {
                    inline_value: None,
                    rep_compression,
                    def_compression,
                    num_rep_values,
                    num_def_values,
                    layers: def_meaning
                        .iter()
                        .map(|&def| Self::def_inter_to_repdef_layer(def))
                        .collect(),
                },
            )),
        }
    }

    pub fn packed_struct(
        values: metadata::CompressiveEncoding,
        bits_per_values: Vec<u64>,
    ) -> metadata::CompressiveEncoding {
        metadata::CompressiveEncoding {
            compression: Some(metadata::compressive_encoding::Compression::PackedStruct(
                Box::new(metadata::PackedStruct {
                    bits_per_value: bits_per_values,
                    values: Some(Box::new(values)),
                }),
            )),
        }
    }

    pub fn packed_struct_variable(
        fields: Vec<metadata::variable_packed_struct::FieldEncoding>,
    ) -> metadata::CompressiveEncoding {
        metadata::CompressiveEncoding {
            compression: Some(
                metadata::compressive_encoding::Compression::VariablePackedStruct(
                    metadata::VariablePackedStruct { fields },
                ),
            ),
        }
    }

    pub fn packed_struct_field_fixed(
        value_encoding: metadata::CompressiveEncoding,
        bits_per_value: u64,
    ) -> metadata::variable_packed_struct::FieldEncoding {
        metadata::variable_packed_struct::FieldEncoding {
            value: Some(value_encoding),
            layout: Some(
                metadata::variable_packed_struct::field_encoding::Layout::BitsPerValue(
                    bits_per_value,
                ),
            ),
        }
    }

    pub fn packed_struct_field_variable(
        value_encoding: metadata::CompressiveEncoding,
        bits_per_length: u64,
    ) -> metadata::variable_packed_struct::FieldEncoding {
        metadata::variable_packed_struct::FieldEncoding {
            value: Some(value_encoding),
            layout: Some(
                metadata::variable_packed_struct::field_encoding::Layout::BitsPerLength(
                    bits_per_length,
                ),
            ),
        }
    }
}
