// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::BTreeMap, sync::Arc};

use lance_core::{Error, Result, datatypes::Schema};
use lance_encoding::{decoder::ColumnInfo, format::pb21};

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
    PerValue,
}

impl EncodingPosition {
    fn child(self) -> Self {
        match self {
            Self::OptionalPayloadBlock | Self::BufferBackedBlock => Self::BufferBackedBlock,
            Self::MiniBlock => Self::MiniBlock,
            Self::PerValue => Self::PerValue,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::OptionalPayloadBlock => "optional-payload block",
            Self::BufferBackedBlock => "buffer-backed block",
            Self::MiniBlock => "mini-block",
            Self::PerValue => "per-value",
        }
    }
}

fn validate_compressive_encoding(
    encoding: &pb21::CompressiveEncoding,
    position: EncodingPosition,
) -> Result<()> {
    use pb21::compressive_encoding::Compression;

    match encoding.compression.as_ref() {
        Some(Compression::Flat(_))
        | Some(Compression::InlineBitpacking(_))
        | Some(Compression::Constant(_)) => Ok(()),
        Some(Compression::Variable(variable)) => validate_compressive_encoding(
            required(variable.offsets.as_deref(), "variable offsets")?,
            position.child(),
        ),
        Some(Compression::OutOfLineBitpacking(bitpacking)) => validate_compressive_encoding(
            required(
                bitpacking.values.as_deref(),
                "out-of-line bitpacking values",
            )?,
            position.child(),
        ),
        Some(Compression::Fsst(fsst)) => validate_compressive_encoding(
            required(fsst.values.as_deref(), "FSST values")?,
            position.child(),
        ),
        Some(Compression::Dictionary(dictionary)) => {
            validate_compressive_encoding(
                required(dictionary.indices.as_deref(), "dictionary indices")?,
                position.child(),
            )?;
            validate_compressive_encoding(
                required(dictionary.items.as_deref(), "dictionary items")?,
                position.child(),
            )
        }
        Some(Compression::Rle(rle)) => {
            validate_compressive_encoding(
                required(rle.values.as_deref(), "RLE values")?,
                position.child(),
            )?;
            validate_compressive_encoding(
                required(rle.run_lengths.as_deref(), "RLE run lengths")?,
                position.child(),
            )
        }
        Some(Compression::ByteStreamSplit(split)) => validate_compressive_encoding(
            required(split.values.as_deref(), "byte-stream-split values")?,
            position.child(),
        ),
        Some(Compression::General(general)) => validate_compressive_encoding(
            required(general.values.as_deref(), "general-compression values")?,
            position.child(),
        ),
        Some(Compression::FixedSizeList(list)) => validate_compressive_encoding(
            required(list.values.as_deref(), "fixed-size-list values")?,
            position.child(),
        ),
        Some(Compression::PackedStruct(packed)) => validate_compressive_encoding(
            required(packed.values.as_deref(), "packed-struct values")?,
            position.child(),
        ),
        Some(Compression::VariablePackedStruct(packed)) => {
            for field in &packed.fields {
                validate_compressive_encoding(
                    required(field.value.as_ref(), "variable packed-struct field")?,
                    position.child(),
                )?;
            }
            Ok(())
        }
        Some(Compression::Range(_)) if position == EncodingPosition::OptionalPayloadBlock => Ok(()),
        Some(Compression::Delta(delta)) if position == EncodingPosition::OptionalPayloadBlock => {
            validate_compressive_encoding(
                required(delta.deltas.as_deref(), "delta values")?,
                EncodingPosition::OptionalPayloadBlock,
            )
        }
        Some(Compression::Range(_) | Compression::Delta(_)) => Err(Error::invalid_input_source(
            format!(
                "Range and delta compression are not supported in {} positions",
                position.name()
            )
            .into(),
        )),
        None => Err(Error::invalid_input_source(
            "Lance v2.3 compressive encoding is missing its compression variant".into(),
        )),
    }
}

fn validate_sparse_positions(positions: Option<&pb21::SparsePositionSet>) -> Result<()> {
    use pb21::sparse_position_set::Positions;

    if let Some(Positions::Explicit(encoding)) =
        positions.and_then(|positions| positions.positions.as_ref())
    {
        validate_compressive_encoding(encoding, EncodingPosition::BufferBackedBlock)?;
    }
    Ok(())
}

fn validate_sparse_validity(validity: Option<&pb21::SparseValiditySet>) -> Result<()> {
    if let Some(validity) = validity {
        validate_sparse_positions(validity.positions.as_ref())?;
    }
    Ok(())
}

fn validate_sparse_layout(layout: &pb21::SparseLayout) -> Result<()> {
    use pb21::{sparse_count_set::Counts, sparse_structural_layer::Layer};

    if !layout.has_large_chunk {
        return Err(Error::invalid_input_source(
            "Lance v2.3 sparse pages require the u32 chunk grammar".into(),
        ));
    }
    validate_compressive_encoding(
        required(layout.value_compression.as_ref(), "sparse values")?,
        EncodingPosition::MiniBlock,
    )?;
    for layer in &layout.structural_layers {
        match layer.layer.as_ref() {
            Some(Layer::Validity(validity)) => {
                validate_sparse_validity(validity.validity.as_ref())?
            }
            Some(Layer::List(list)) => {
                validate_sparse_positions(list.non_empty_positions.as_ref())?;
                if let Some(Counts::Explicit(encoding)) = list
                    .counts
                    .as_ref()
                    .and_then(|counts| counts.counts.as_ref())
                {
                    validate_compressive_encoding(encoding, EncodingPosition::BufferBackedBlock)?;
                }
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

fn validate_page_layout(layout: &pb21::PageLayout) -> Result<()> {
    use pb21::page_layout::Layout;

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
        Some(Layout::FullZipLayout(fullzip)) => validate_compressive_encoding(
            required(fullzip.value_compression.as_ref(), "full-zip values")?,
            EncodingPosition::PerValue,
        ),
        Some(Layout::ConstantLayout(constant)) => {
            if let Some(rep) = constant.rep_compression.as_ref() {
                validate_compressive_encoding(rep, EncodingPosition::BufferBackedBlock)?;
            }
            if let Some(def) = constant.def_compression.as_ref() {
                validate_compressive_encoding(def, EncodingPosition::BufferBackedBlock)?;
            }
            Ok(())
        }
        Some(Layout::SparseLayout(sparse)) => validate_sparse_layout(sparse),
        Some(Layout::BlobLayout(blob)) => {
            let inner = blob.inner_layout.as_deref().ok_or_else(|| {
                Error::invalid_input_source(
                    "Lance v2.3 blob page layout is missing its inner layout".into(),
                )
            })?;
            validate_page_layout(inner)
        }
        None => Err(Error::invalid_input_source(
            "Lance v2.3 page is missing its page layout".into(),
        )),
    }
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
            let page_layout = structural::decode_page_layout(column_index, page_index, page)?;
            validate_page_layout(&page_layout)?;
            structural::build_page_info(column_index, page_index, page, page_layout)
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
    use pb21::{
        CompressiveEncoding, Dictionary, FixedSizeList, Flat, FullZipLayout, MiniBlockLayout,
        PageLayout, Rle, compressive_encoding::Compression, page_layout::Layout,
    };

    fn flat(bits_per_value: u64) -> CompressiveEncoding {
        CompressiveEncoding {
            compression: Some(Compression::Flat(Flat {
                bits_per_value,
                data: None,
            })),
        }
    }

    #[test]
    fn accepts_nested_variable_width_rle() {
        let rle = CompressiveEncoding {
            compression: Some(Compression::Rle(Box::new(Rle {
                values: Some(Box::new(flat(32))),
                run_lengths: Some(Box::new(flat(16))),
            }))),
        };
        let dictionary = CompressiveEncoding {
            compression: Some(Compression::Dictionary(Box::new(Dictionary {
                indices: Some(Box::new(rle)),
                items: Some(Box::new(flat(32))),
                num_dictionary_items: 1,
            }))),
        };
        let layout = PageLayout {
            layout: Some(Layout::FullZipLayout(FullZipLayout {
                value_compression: Some(dictionary),
                ..Default::default()
            })),
        };

        validate_page_layout(&layout).unwrap();
    }

    #[test]
    fn accepts_range_and_delta() {
        let range = CompressiveEncoding {
            compression: Some(Compression::Range(pb21::Range {
                uncompressed_bits_per_value: 32,
                start: 0,
                step: 1,
            })),
        };
        let delta = CompressiveEncoding {
            compression: Some(Compression::Delta(Box::new(pb21::Delta {
                uncompressed_bits_per_value: 32,
                base: 0,
                deltas: Some(Box::new(range.clone())),
            }))),
        };

        validate_compressive_encoding(&range, EncodingPosition::OptionalPayloadBlock).unwrap();
        validate_compressive_encoding(&delta, EncodingPosition::OptionalPayloadBlock).unwrap();
    }

    #[test]
    fn rejects_metadata_only_codecs_in_buffer_backed_block_positions() {
        let range = CompressiveEncoding {
            compression: Some(Compression::Range(pb21::Range {
                uncompressed_bits_per_value: 32,
                start: 0,
                step: 1,
            })),
        };
        let delta = CompressiveEncoding {
            compression: Some(Compression::Delta(Box::new(pb21::Delta {
                uncompressed_bits_per_value: 32,
                base: 0,
                deltas: Some(Box::new(range.clone())),
            }))),
        };
        let wrapped_range = CompressiveEncoding {
            compression: Some(Compression::FixedSizeList(Box::new(FixedSizeList {
                items_per_value: 1,
                values: Some(Box::new(range.clone())),
                has_validity: false,
            }))),
        };

        for encoding in [range, delta, wrapped_range] {
            let layout = PageLayout {
                layout: Some(Layout::MiniBlockLayout(MiniBlockLayout {
                    rep_compression: Some(encoding),
                    value_compression: Some(flat(32)),
                    has_large_chunk: true,
                    ..Default::default()
                })),
            };

            let error = validate_page_layout(&layout).unwrap_err();
            assert!(matches!(&error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains("buffer-backed block positions"));
        }
    }

    #[test]
    fn rejects_range_and_delta_outside_block_positions() {
        let range = CompressiveEncoding {
            compression: Some(Compression::Range(pb21::Range {
                uncompressed_bits_per_value: 32,
                start: 0,
                step: 1,
            })),
        };
        let delta = CompressiveEncoding {
            compression: Some(Compression::Delta(Box::new(pb21::Delta {
                uncompressed_bits_per_value: 32,
                base: 0,
                deltas: Some(Box::new(range.clone())),
            }))),
        };

        for encoding in [range, delta] {
            let miniblock = PageLayout {
                layout: Some(Layout::MiniBlockLayout(MiniBlockLayout {
                    value_compression: Some(encoding.clone()),
                    has_large_chunk: true,
                    ..Default::default()
                })),
            };
            let miniblock_error = validate_page_layout(&miniblock).unwrap_err();
            assert!(matches!(&miniblock_error, Error::InvalidInput { .. }));
            assert!(miniblock_error.to_string().contains("mini-block positions"));

            let fullzip = PageLayout {
                layout: Some(Layout::FullZipLayout(FullZipLayout {
                    value_compression: Some(encoding),
                    ..Default::default()
                })),
            };
            let fullzip_error = validate_page_layout(&fullzip).unwrap_err();
            assert!(matches!(&fullzip_error, Error::InvalidInput { .. }));
            assert!(fullzip_error.to_string().contains("per-value positions"));
        }
    }
}
