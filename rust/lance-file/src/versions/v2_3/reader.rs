// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::BTreeMap, sync::Arc};

use lance_core::{Error, Result, datatypes::Schema};
use lance_encoding::{
    compression::{
        DecompressionStrategy, DefaultDecompressionStrategy, create_fixed_width_block_decompressor,
    },
    decoder::ColumnInfo,
    format::{metadata, metadata::compressive_encoding::Compression, pb23},
};

use crate::{
    format::pbfile,
    reader::{
        BufferDescriptor, CachedFileMetadata, FileMetadataIndex, RawFileMetadata, ReaderProjection,
        structural,
    },
    version::ConcreteFileVersion,
};

fn required<'a, T>(value: Option<&'a T>, label: &str) -> Result<&'a T> {
    value.ok_or_else(|| {
        Error::invalid_input_source(
            format!("Lance v2.3 {label} is missing its nested encoding").into(),
        )
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EncodingPosition {
    OptionalPayloadBlock,
    BufferBackedBlock,
    MiniBlock,
    PerValue { is_variable: bool },
}

impl EncodingPosition {
    fn name(self) -> &'static str {
        match self {
            Self::OptionalPayloadBlock => "optional-payload block",
            Self::BufferBackedBlock => "buffer-backed block",
            Self::MiniBlock => "mini-block",
            Self::PerValue { .. } => "per-value",
        }
    }
}

fn validate_compressive_encoding(
    encoding: &metadata::CompressiveEncoding,
    position: EncodingPosition,
) -> Result<()> {
    let strategy = DefaultDecompressionStrategy::default();
    let validate = || -> Result<()> {
        match position {
            // Indexed sparse values use the typed UInt32 block grammar, not the
            // buffer-backed grammar used for dense levels and dictionary items.
            EncodingPosition::OptionalPayloadBlock => {
                create_fixed_width_block_decompressor(encoding, 32).map(|_| ())
            }
            EncodingPosition::BufferBackedBlock => {
                match encoding.compression.as_ref() {
                    Some(Compression::Range(_) | Compression::MonotonicDelta(_)) => {
                        return Err(Error::invalid_input(
                            "Range and MonotonicDelta compression require an optional-payload block",
                        ));
                    }
                    Some(Compression::General(general)) => validate_compressive_encoding(
                        required(general.values.as_deref(), "general-compression values")?,
                        position,
                    )?,
                    Some(Compression::Variable(variable)) => {
                        validate_variable_encoding(variable)?;
                    }
                    _ => {}
                }
                let decoder = strategy.create_block_decompressor(encoding)?;
                if !decoder.requires_payload() {
                    return Err(Error::invalid_input(
                        "Metadata-only compression requires an optional-payload block",
                    ));
                }
                Ok(())
            }
            EncodingPosition::MiniBlock => strategy
                .create_miniblock_decompressor(encoding, &strategy)
                .map(|_| ()),
            EncodingPosition::PerValue { is_variable: false } => strategy
                .create_fixed_per_value_decompressor(encoding)
                .map(|_| ()),
            EncodingPosition::PerValue { is_variable: true } => {
                // These factories do not all inspect their child descriptors:
                // FSST uses VariableDecoder directly, and General only builds a
                // child decoder for fixed-width tensors.
                match encoding.compression.as_ref() {
                    Some(Compression::Variable(variable)) => {
                        validate_variable_encoding(variable)?;
                    }
                    Some(Compression::Fsst(fsst)) => {
                        let values = required(fsst.values.as_deref(), "FSST values")?;
                        if fsst.symbol_table.is_empty()
                            || !matches!(values.compression, Some(Compression::Variable(_)))
                        {
                            return Err(Error::invalid_input(
                                "FSST requires a symbol table and Variable values",
                            ));
                        }
                        validate_compressive_encoding(values, position)?;
                    }
                    Some(Compression::General(general)) => {
                        let values =
                            required(general.values.as_deref(), "general-compression values")?;
                        let is_variable = match values.compression.as_ref() {
                            Some(Compression::Variable(_)) => true,
                            Some(Compression::FixedSizeList(_) | Compression::AxisDelta(_)) => {
                                false
                            }
                            _ => {
                                return Err(Error::invalid_input(
                                    "Per-value General requires Variable, FixedSizeList, or AxisDelta values",
                                ));
                            }
                        };
                        validate_compressive_encoding(
                            values,
                            EncodingPosition::PerValue { is_variable },
                        )?;
                    }
                    Some(Compression::VariablePackedStruct(packed)) => {
                        for field in &packed.fields {
                            let is_variable = match required(
                                field.layout.as_ref(),
                                "packed-struct field layout",
                            )? {
                                metadata::variable_packed_struct::field_encoding::Layout::BitsPerValue(_) => false,
                                metadata::variable_packed_struct::field_encoding::Layout::BitsPerLength(_) => true,
                            };
                            validate_compressive_encoding(
                                required(field.value.as_ref(), "variable packed-struct field")?,
                                EncodingPosition::PerValue { is_variable },
                            )?;
                        }
                    }
                    _ => {}
                }
                strategy
                    .create_variable_per_value_decompressor(encoding)
                    .map(|_| ())
            }
        }
    };
    validate().map_err(|error| {
        Error::invalid_input(format!(
            "Lance v2.3 encoding is invalid in {} positions: {error}",
            position.name()
        ))
    })
}

fn validate_variable_encoding(variable: &metadata::Variable) -> Result<()> {
    let offsets = required(variable.offsets.as_deref(), "variable offsets")?;
    if variable.values.is_some()
        || !matches!(offsets.compression.as_ref(), Some(Compression::Flat(flat))
            if matches!(flat.bits_per_value, 32 | 64) && flat.data.is_none())
    {
        return Err(Error::invalid_input(
            "Variable block and per-value encodings require plain 32 or 64-bit Flat offsets and no values compression",
        ));
    }
    Ok(())
}

fn validate_sparse_sequence(
    positions: Option<&metadata::CompressiveEncoding>,
    count: u64,
) -> Result<()> {
    lance_encoding::compression::validate_fixed_width_block_sequence(
        required(positions, "sparse structural sequence")?,
        64,
        count,
    )
}

fn validate_sparse_validity(validity: Option<&pb23::SparseValiditySet>) -> Result<()> {
    let validity = required(validity, "sparse validity")?;
    match pb23::sparse_validity_set::Meaning::try_from(validity.meaning) {
        Ok(
            pb23::sparse_validity_set::Meaning::SparseValidityNullPositions
            | pb23::sparse_validity_set::Meaning::SparseValidityValidPositions,
        ) => {}
        _ => {
            return Err(Error::invalid_input(
                "Planar sparse validity has an invalid meaning",
            ));
        }
    }
    validate_sparse_sequence(validity.positions.as_ref(), validity.num_positions)
}

fn validate_sparse_structure(layout: &pb23::SparseStructure) -> Result<()> {
    use pb23::sparse_structural_layer::Layer;

    for layer in &layout.layers {
        match layer.layer.as_ref() {
            Some(Layer::Validity(validity)) => validate_sparse_validity(Some(validity))?,
            Some(Layer::List(list)) => {
                validate_sparse_sequence(
                    list.non_empty_positions.as_ref(),
                    list.num_non_empty_positions,
                )?;
                validate_sparse_sequence(list.counts.as_ref(), list.num_non_empty_positions)?;
                validate_sparse_validity(list.validity.as_ref())?;
            }
            Some(Layer::FixedSizeList(list)) => validate_sparse_validity(list.validity.as_ref())?,
            None => {
                return Err(Error::invalid_input_source(
                    "Lance v2.3 sparse structural layer is missing its layer variant".into(),
                ));
            }
        }
    }
    Ok(())
}

fn validate_page_layout(layout: &metadata::PageLayout) -> Result<()> {
    use metadata::page_layout::Layout;

    match layout.layout.as_ref() {
        Some(Layout::MiniBlockLayout(miniblock)) => {
            if !miniblock.has_large_chunk {
                return Err(Error::invalid_input_source(
                    "Lance v2.3 miniblock pages require the u32 chunk grammar".into(),
                ));
            }
            if let Some(rep) = miniblock.rep_compression.as_ref() {
                validate_compressive_encoding(rep, EncodingPosition::BufferBackedBlock)?;
            }
            if let Some(def) = miniblock.def_compression.as_ref() {
                validate_compressive_encoding(def, EncodingPosition::BufferBackedBlock)?;
            }
            validate_compressive_encoding(
                required(miniblock.value_compression.as_ref(), "miniblock values")?,
                EncodingPosition::MiniBlock,
            )?;
            if let Some(dictionary) = miniblock.dictionary.as_ref() {
                validate_compressive_encoding(dictionary, EncodingPosition::BufferBackedBlock)?;
            }
            Ok(())
        }
        Some(Layout::FullZipLayout(fullzip)) => {
            let is_variable = match required(fullzip.details.as_ref(), "full-zip details")? {
                metadata::full_zip_layout::Details::BitsPerValue(_) => false,
                metadata::full_zip_layout::Details::BitsPerOffset(_) => true,
            };
            validate_compressive_encoding(
                required(fullzip.value_compression.as_ref(), "full-zip values")?,
                EncodingPosition::PerValue { is_variable },
            )
        }
        Some(Layout::ConstantLayout(constant)) => {
            if let Some(rep) = constant.rep_compression.as_ref() {
                validate_compressive_encoding(rep, EncodingPosition::BufferBackedBlock)?;
            }
            if let Some(def) = constant.def_compression.as_ref() {
                validate_compressive_encoding(def, EncodingPosition::BufferBackedBlock)?;
            }
            Ok(())
        }
        Some(Layout::BlobLayout(_)) => Err(Error::invalid_input(
            "Blob v1 pages are not part of the Lance 2.3 grammar",
        )),
        None => Err(Error::invalid_input_source(
            "Lance v2.3 page is missing its page layout".into(),
        )),
    }
}

fn validate_planar_layout(layout: &pb23::PlanarLayout) -> Result<()> {
    if let Some(structure) = layout.sparse() {
        validate_sparse_structure(structure)?;
        use pb23::value_plane::Codec;
        match layout.values.as_ref().and_then(|v| v.codec.as_ref()) {
            Some(Codec::Chunked(v)) => {
                validate_compressive_encoding(
                    required(v.compression.as_ref(), "chunked values")?,
                    EncodingPosition::MiniBlock,
                )?;
                if let Some(dictionary) = &v.dictionary {
                    validate_compressive_encoding(dictionary, EncodingPosition::BufferBackedBlock)?;
                }
                Ok(())
            }
            Some(Codec::Zipped(v)) => validate_zipped_values(v),
            Some(Codec::Indexed(v)) => {
                validate_compressive_encoding(
                    required(v.indices.as_ref(), "dictionary indices")?,
                    EncodingPosition::OptionalPayloadBlock,
                )?;
                let items = required(v.items.as_ref(), "dictionary items")?;
                // Indexed dictionary items always use variable per-value decoding.
                if !matches!(
                    items.addressing,
                    Some(pb23::zipped_values::Addressing::BitsPerOffset(_))
                ) {
                    return Err(Error::invalid_input(
                        "Indexed dictionary items require variable-width addressing",
                    ));
                }
                validate_zipped_values(items)
            }
            Some(Codec::Constant(_)) | Some(Codec::None(_)) => Ok(()),
            None => Err(Error::invalid_input("Planar page is missing values")),
        }
    } else {
        validate_page_layout(&layout.dense_execution_layout()?)
    }
}

fn validate_zipped_values(values: &pb23::ZippedValues) -> Result<()> {
    let is_variable = match required(values.addressing.as_ref(), "zipped addressing")? {
        pb23::zipped_values::Addressing::BitsPerValue(_) => false,
        pb23::zipped_values::Addressing::BitsPerOffset(_) => true,
    };
    validate_compressive_encoding(
        required(values.compression.as_ref(), "zipped values")?,
        EncodingPosition::PerValue { is_variable },
    )
}

pub fn decode_column(
    column_index: u32,
    metadata: &pbfile::ColumnMetadata,
) -> Result<Arc<ColumnInfo>> {
    let page_infos = metadata
        .pages
        .iter()
        .enumerate()
        .map(|(page_index, page)| {
            let layout: pb23::PlanarLayout = structural::fetch_encoding(
                page.encoding
                    .as_ref()
                    .ok_or_else(|| Error::invalid_input("Planar page is missing its encoding"))?,
            )?;
            validate_planar_layout(&layout)?;
            structural::build_page_info(
                column_index,
                page_index,
                page,
                lance_encoding::decoder::PageEncoding::Planar(layout),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    structural::build_column_info(column_index, metadata, page_infos)
}

pub fn decode_column_metadata(
    column_metadatas: &[pbfile::ColumnMetadata],
) -> Result<Vec<Arc<ColumnInfo>>> {
    column_metadatas
        .iter()
        .enumerate()
        .map(|(column_index, metadata)| {
            let column_index = u32::try_from(column_index).map_err(|_| {
                Error::invalid_input_source("File has more than u32::MAX columns".into())
            })?;
            decode_column(column_index, metadata)
        })
        .collect()
}

pub fn projection_from_field_ids(
    schema: &Schema,
    field_id_to_column_index: &BTreeMap<u32, u32>,
) -> ReaderProjection {
    structural::projection_from_field_ids(schema, field_id_to_column_index)
}

pub fn projection_from_whole_schema(schema: &Schema) -> ReaderProjection {
    structural::projection_from_field_ids(schema, &super::field_id_to_column_index(schema))
}

pub fn projection_from_column_names(
    schema: &Schema,
    column_names: &[&str],
) -> Result<ReaderProjection> {
    structural::projection_from_column_names(
        schema,
        column_names,
        &super::field_id_to_column_index(schema),
    )
}

pub fn finish_metadata(raw: RawFileMetadata) -> Result<CachedFileMetadata> {
    if (raw.footer.major_version, raw.footer.minor_version) != (2, 3) {
        return Err(Error::version_conflict(
            "Attempt to use the Lance v2.3 reader for a different file version".to_string(),
            raw.footer.major_version,
            raw.footer.minor_version,
        ));
    }
    validate_global_buffers(&raw.file_buffers)?;
    let column_infos = decode_column_metadata(&raw.column_metadatas)?;
    Ok(CachedFileMetadata {
        file_schema: raw.file_schema,
        column_metadatas: raw.column_metadatas,
        column_infos,
        num_rows: raw.num_rows,
        file_buffers: raw.file_buffers,
        num_data_bytes: raw.num_data_bytes,
        num_column_metadata_bytes: raw.num_column_metadata_bytes,
        num_global_buffer_bytes: raw.num_global_buffer_bytes,
        num_footer_bytes: raw.num_footer_bytes,
        major_version: raw.footer.major_version,
        minor_version: raw.footer.minor_version,
        version: ConcreteFileVersion::V2_3,
        file_size_bytes: raw.file_size_bytes,
        retained_global_buffers: raw.retained_global_buffers,
    })
}

pub fn validate_global_buffers(buffers: &[BufferDescriptor]) -> Result<()> {
    structural::validate_global_buffers(buffers)
}

pub fn finish_metadata_index(index: FileMetadataIndex) -> Result<FileMetadataIndex> {
    if index.version != ConcreteFileVersion::V2_3 {
        let (major, minor) = index.version.to_standard_footer_numbers();
        return Err(Error::version_conflict(
            "Attempt to use the Lance v2.3 reader for a different metadata index".to_string(),
            major,
            minor,
        ));
    }
    validate_global_buffers(&index.file_buffers)?;
    Ok(index)
}

#[cfg(test)]
mod grammar_tests {
    use super::*;
    use lance_encoding::{
        encodings::physical::block::{CompressionConfig, CompressionScheme},
        format::EncodingUtils,
    };
    use metadata::{
        CompressiveEncoding, FullZipLayout, MiniBlockLayout, PageLayout,
        compressive_encoding::Compression, page_layout::Layout,
    };

    fn flat(bits_per_value: u64) -> CompressiveEncoding {
        EncodingUtils::flat(bits_per_value, None)
    }

    fn general(values: CompressiveEncoding) -> CompressiveEncoding {
        EncodingUtils::wrapped(CompressionConfig::new(CompressionScheme::Lz4, None), values)
            .unwrap()
    }

    fn axis_delta() -> CompressiveEncoding {
        CompressiveEncoding {
            compression: Some(Compression::AxisDelta(Box::new(metadata::AxisDelta {
                values: Some(Box::new(EncodingUtils::fsl(64, false, flat(16)))),
                axis_length: 8,
                inner_stride: 1,
            }))),
        }
    }

    // Expected support: optional-payload block, buffer-backed block, mini-block,
    // fixed per-value, variable per-value.
    #[rstest::rstest]
    #[case::flat(flat(32), [true, true, true, true, false])]
    #[case::constant(EncodingUtils::constant(Some(0_u32.to_le_bytes().to_vec().into())), [true, false, false, true, false])]
    #[case::variable(EncodingUtils::variable(flat(32), None), [false, true, true, false, true])]
    #[case::inline(EncodingUtils::inline_bitpacking(32, None), [true, true, true, false, false])]
    #[case::out_of_line(EncodingUtils::out_of_line_bitpacking(32, flat(4)), [true, true, false, false, false])]
    #[case::fsst(EncodingUtils::fsst(EncodingUtils::variable(flat(32), None), vec![1]), [false, false, true, false, true])]
    #[case::dictionary(EncodingUtils::dictionary(flat(32), flat(32), 1), [true, false, false, false, false])]
    #[case::rle(EncodingUtils::rle(flat(32), flat(32)), [true, true, true, false, false])]
    #[case::byte_stream_split(EncodingUtils::byte_stream_split(flat(32)), [false, false, true, false, false])]
    #[case::alp(EncodingUtils::alp(32), [false, false, true, false, false])]
    #[case::pcodec(EncodingUtils::pcodec(flat(32)), [false, false, true, false, false])]
    #[case::general_flat(general(flat(32)), [true, true, true, false, false])]
    #[case::general_variable(general(EncodingUtils::variable(flat(32), None)), [false, true, true, false, true])]
    #[case::fixed_size_list(EncodingUtils::fsl(2, false, flat(32)), [false, true, true, true, false])]
    #[case::packed_struct(EncodingUtils::packed_struct(flat(32), vec![32]), [false, false, true, true, false])]
    #[case::variable_packed_struct(EncodingUtils::packed_struct_variable(vec![
        EncodingUtils::packed_struct_field_fixed(flat(32), 32),
        EncodingUtils::packed_struct_field_variable(EncodingUtils::variable(flat(32), None), 32),
    ]), [false, false, false, false, true])]
    #[case::range(EncodingUtils::range(32, 0, 1), [true, false, false, false, false])]
    #[case::monotonic_delta(EncodingUtils::delta(32, 0, flat(32)), [true, false, false, false, false])]
    #[case::axis_delta(axis_delta(), [false, false, false, true, false])]
    #[case::missing(CompressiveEncoding::default(), [false; 5])]
    fn codec_roots_match_decoder_contracts(
        #[case] encoding: CompressiveEncoding,
        #[case] supported: [bool; 5],
        #[values(
            EncodingPosition::OptionalPayloadBlock,
            EncodingPosition::BufferBackedBlock,
            EncodingPosition::MiniBlock,
            EncodingPosition::PerValue { is_variable: false },
            EncodingPosition::PerValue { is_variable: true }
        )]
        position: EncodingPosition,
    ) {
        let index = match position {
            EncodingPosition::OptionalPayloadBlock => 0,
            EncodingPosition::BufferBackedBlock => 1,
            EncodingPosition::MiniBlock => 2,
            EncodingPosition::PerValue { is_variable: false } => 3,
            EncodingPosition::PerValue { is_variable: true } => 4,
        };
        let result = validate_compressive_encoding(&encoding, position);
        if supported[index] {
            result.unwrap();
        } else {
            let error = result.unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains(position.name()));
        }
    }

    #[rstest::rstest]
    #[case::dictionary_offsets(
        EncodingUtils::variable(EncodingUtils::dictionary(flat(32), flat(32), 1), None),
        true
    )]
    #[case::rle_offsets(
        EncodingUtils::variable(EncodingUtils::rle(flat(32), flat(32)), None),
        true
    )]
    #[case::monotonic_delta_offsets(
        EncodingUtils::variable(EncodingUtils::delta(32, 0, flat(32)), None),
        true
    )]
    #[case::dictionary_rle_offsets(
        EncodingUtils::variable(
            EncodingUtils::dictionary(EncodingUtils::rle(flat(32), flat(32)), flat(32), 1),
            None
        ),
        false
    )]
    fn variable_miniblocks_use_typed_offset_grammar(
        #[case] encoding: CompressiveEncoding,
        #[case] supported: bool,
    ) {
        let layout = PageLayout {
            layout: Some(Layout::MiniBlockLayout(MiniBlockLayout {
                value_compression: Some(encoding),
                has_large_chunk: true,
                ..Default::default()
            })),
        };
        if supported {
            validate_page_layout(&layout).unwrap();
        } else {
            let error = validate_page_layout(&layout).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains("generic block encoding"));
        }
    }

    #[rstest::rstest]
    #[case::fixed(false)]
    #[case::variable(true)]
    fn rejects_fullzip_dictionary_rle(#[case] is_variable: bool) {
        let dictionary =
            EncodingUtils::dictionary(EncodingUtils::rle(flat(32), flat(16)), flat(32), 1);
        let layout = PageLayout {
            layout: Some(Layout::FullZipLayout(FullZipLayout {
                value_compression: Some(dictionary),
                details: Some(if is_variable {
                    metadata::full_zip_layout::Details::BitsPerOffset(32)
                } else {
                    metadata::full_zip_layout::Details::BitsPerValue(32)
                }),
                ..Default::default()
            })),
        };
        let error = validate_page_layout(&layout).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("dictionary is not supported"));
    }

    #[rstest::rstest]
    #[case::monotonic_delta_of_monotonic_delta(
        EncodingPosition::OptionalPayloadBlock,
        EncodingUtils::delta(32, 0, EncodingUtils::delta(32, 0, flat(32)))
    )]
    #[case::dictionary_of_rle(
        EncodingPosition::OptionalPayloadBlock,
        EncodingUtils::dictionary(EncodingUtils::rle(flat(32), flat(32)), flat(32), 1)
    )]
    #[case::bitpacking_of_rle(
        EncodingPosition::BufferBackedBlock,
        EncodingUtils::out_of_line_bitpacking(32, EncodingUtils::rle(flat(32), flat(16)))
    )]
    #[case::buffer_range(
        EncodingPosition::BufferBackedBlock,
        general(EncodingUtils::range(32, 0, 1))
    )]
    #[case::buffer_monotonic_delta(
        EncodingPosition::BufferBackedBlock,
        general(EncodingUtils::delta(32, 0, flat(32)))
    )]
    #[case::miniblock_fsst(EncodingPosition::MiniBlock, EncodingUtils::fsst(flat(32), vec![1]))]
    #[case::miniblock_split(
        EncodingPosition::MiniBlock,
        EncodingUtils::byte_stream_split(EncodingUtils::rle(flat(32), flat(16)))
    )]
    #[case::miniblock_rle(
        EncodingPosition::MiniBlock,
        EncodingUtils::rle(EncodingUtils::inline_bitpacking(32, None), flat(16))
    )]
    #[case::miniblock_fsl(
        EncodingPosition::MiniBlock,
        EncodingUtils::fsl(2, false, EncodingUtils::rle(flat(32), flat(16)))
    )]
    #[case::per_value_fsl(EncodingPosition::PerValue { is_variable: false }, EncodingUtils::fsl(2, false, EncodingUtils::rle(flat(32), flat(16))))]
    #[case::per_value_offsets(EncodingPosition::PerValue { is_variable: true }, EncodingUtils::variable(EncodingUtils::rle(flat(32), flat(16)), None))]
    #[case::per_value_fsst(EncodingPosition::PerValue { is_variable: true }, EncodingUtils::fsst(flat(32), vec![1]))]
    #[case::per_value_general(EncodingPosition::PerValue { is_variable: true }, general(EncodingUtils::rle(flat(32), flat(16))))]
    #[case::packed_fixed_field(EncodingPosition::PerValue { is_variable: true }, EncodingUtils::packed_struct_variable(vec![
        EncodingUtils::packed_struct_field_fixed(EncodingUtils::variable(flat(32), None), 32),
    ]))]
    #[case::packed_variable_field(EncodingPosition::PerValue { is_variable: true }, EncodingUtils::packed_struct_variable(vec![
        EncodingUtils::packed_struct_field_variable(flat(32), 32),
    ]))]
    fn rejects_invalid_codec_nesting(
        #[case] position: EncodingPosition,
        #[case] encoding: CompressiveEncoding,
    ) {
        let error = validate_compressive_encoding(&encoding, position).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(position.name()));
    }

    #[rstest::rstest]
    #[case::flat_fixed(flat(32), false, true)]
    #[case::flat_variable(flat(32), true, false)]
    #[case::variable_fixed(EncodingUtils::variable(flat(32), None), false, false)]
    #[case::variable_variable(EncodingUtils::variable(flat(32), None), true, true)]
    #[case::general_tensor(general(EncodingUtils::fsl(2, false, flat(32))), true, true)]
    #[case::general_axis_delta(general(axis_delta()), true, true)]
    #[case::rle(EncodingUtils::rle(flat(32), flat(16)), false, false)]
    fn planar_zipped_values_match_addressing(
        #[case] encoding: CompressiveEncoding,
        #[case] is_variable: bool,
        #[case] supported: bool,
        #[values(false, true)] is_sparse: bool,
    ) {
        let structure = if is_sparse {
            pb23::structural_plane::Codec::Sparse(pb23::SparseStructure::default())
        } else {
            pb23::structural_plane::Codec::Dense(pb23::DenseStructure {
                layers: vec![metadata::RepDefLayer::RepdefAllValidItem as i32],
            })
        };
        let layout = pb23::PlanarLayout {
            structure: Some(pb23::StructuralPlane {
                codec: Some(structure),
            }),
            values: Some(pb23::ValuePlane {
                codec: Some(pb23::value_plane::Codec::Zipped(pb23::ZippedValues {
                    compression: Some(encoding),
                    levels: (!is_sparse).then_some(pb23::DenseZipLevels::default()),
                    addressing: Some(if is_variable {
                        pb23::zipped_values::Addressing::BitsPerOffset(32)
                    } else {
                        pb23::zipped_values::Addressing::BitsPerValue(32)
                    }),
                    ..Default::default()
                })),
            }),
        };
        if supported {
            validate_planar_layout(&layout).unwrap();
        } else {
            let error = validate_planar_layout(&layout).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains("per-value positions"));
        }
    }

    #[rstest::rstest]
    #[case::flat(flat(32), EncodingUtils::variable(flat(32), None), true)]
    #[case::range(
        EncodingUtils::range(32, 0, 1),
        EncodingUtils::variable(flat(32), None),
        true
    )]
    #[case::dictionary(
        EncodingUtils::dictionary(flat(32), flat(32), 1),
        EncodingUtils::variable(flat(32), None),
        true
    )]
    #[case::nested_dictionary(
        EncodingUtils::dictionary(EncodingUtils::rle(flat(32), flat(32)), flat(32), 1),
        EncodingUtils::variable(flat(32), None),
        false
    )]
    #[case::wrong_index_width(flat(64), EncodingUtils::variable(flat(32), None), false)]
    #[case::invalid_items(flat(32), EncodingUtils::dictionary(flat(32), flat(32), 1), false)]
    fn planar_indexed_values_use_typed_indices_and_variable_items(
        #[case] indices: CompressiveEncoding,
        #[case] items: CompressiveEncoding,
        #[case] supported: bool,
    ) {
        let layout = pb23::PlanarLayout {
            structure: Some(pb23::StructuralPlane {
                codec: Some(pb23::structural_plane::Codec::Sparse(
                    pb23::SparseStructure::default(),
                )),
            }),
            values: Some(pb23::ValuePlane {
                codec: Some(pb23::value_plane::Codec::Indexed(pb23::IndexedValues {
                    indices: Some(indices),
                    items: Some(pb23::ZippedValues {
                        compression: Some(items),
                        addressing: Some(pb23::zipped_values::Addressing::BitsPerOffset(32)),
                        num_items: 1,
                        num_visible_items: 1,
                        ..Default::default()
                    }),
                })),
            }),
        };
        if supported {
            validate_planar_layout(&layout).unwrap();
        } else {
            let error = validate_planar_layout(&layout).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains("positions"));
        }
    }
}
