// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The 2.3 writer chooses structure independently from value storage.

use super::*;
use crate::format::planar::from_dense_execution;

pub(super) fn inline_type(data_type: &DataType) -> bool {
    *data_type == DataType::Boolean
        || data_type.byte_width_opt().is_some_and(|w| w <= 32)
            && !matches!(
                data_type,
                DataType::FixedSizeList(_, _) | DataType::Struct(_)
            )
}

pub(super) fn fixed_width_bytes(data_type: &DataType) -> Option<usize> {
    match data_type {
        DataType::FixedSizeList(child, dimension) => {
            fixed_width_bytes(child.data_type())?.checked_mul(usize::try_from(*dimension).ok()?)
        }
        _ => data_type.byte_width_opt(),
    }
}

fn is_narrow(arrays: &[ArrayRef], num_values: u64) -> bool {
    match arrays.first().map(|array| array.data_type()) {
        Some(DataType::Binary) => arrays.iter().all(|array| {
            array
                .as_binary::<i32>()
                .value_offsets()
                .windows(2)
                .all(|v| v[1] - v[0] < 256)
        }),
        Some(DataType::LargeBinary) => arrays.iter().all(|array| {
            array
                .as_binary::<i64>()
                .value_offsets()
                .windows(2)
                .all(|v| v[1] - v[0] < 256)
        }),
        Some(DataType::Utf8) => arrays.iter().all(|array| {
            array
                .as_string::<i32>()
                .value_offsets()
                .windows(2)
                .all(|v| v[1] - v[0] < 256)
        }),
        Some(DataType::LargeUtf8) => arrays.iter().all(|array| {
            array
                .as_string::<i64>()
                .value_offsets()
                .windows(2)
                .all(|v| v[1] - v[0] < 256)
        }),
        Some(data_type) if fixed_width_bytes(data_type).is_some() => {
            fixed_width_bytes(data_type).is_some_and(|width| width < 256)
        }
        _ => PrimitiveStructuralEncoder::is_narrow(&DataBlock::from_arrays(arrays, num_values)),
    }
}

#[derive(Debug)]
pub(super) struct PlanarPrimitiveEncoding {
    pub compression: Arc<dyn CompressionStrategy>,
}

impl PrimitivePageEncodingBehavior for PlanarPrimitiveEncoding {
    fn try_plan_pages(
        &self,
        ctx: &PrimitivePlanContext<'_>,
        arrays: &[ArrayRef],
        normalized: &NormalizedStructuralPlan,
        row_number: u64,
        num_rows: u64,
        num_values: u64,
    ) -> Result<Option<Vec<PrimitivePageData>>> {
        let requested = ctx.encoding_metadata.get(STRUCTURAL_ENCODING_META_KEY);
        let explicit_sparse =
            requested.is_some_and(|v| v.eq_ignore_ascii_case(STRUCTURAL_ENCODING_SPARSE));
        let trivial = normalized.layers().all(|layer| {
            matches!(layer,
            crate::repdef::NormalizedStructuralLayer::Validity { validity, .. }
                if validity.is_none_or(|v| v.count_set_bits() == v.len()))
        });
        let has_lists = normalized
            .layers()
            .any(|layer| matches!(layer, crate::repdef::NormalizedStructuralLayer::List { .. }));
        let has_fsl = normalized.layers().any(|layer| {
            matches!(
                layer,
                crate::repdef::NormalizedStructuralLayer::FixedSizeList { .. }
            )
        });
        // Slot domains avoid expanding nested fixed-size-list children into
        // dense structural events before a sparse page has even been selected.
        if (explicit_sparse && !trivial)
            || (requested.is_none()
                && has_lists
                && (has_fsl
                    || (num_rows > miniblock::max_repdef_levels_per_chunk(1)
                        && num_values < num_rows / 16)))
        {
            return Ok(Some(vec![PrimitivePageData {
                arrays: arrays.to_vec(),
                structure: PrimitivePageStructure::Sparse {
                    plan: sparse::writer::plan(normalized, num_values)?,
                    constant: None,
                },
                row_number,
                num_rows,
            }]));
        }
        let (repdef, budget) = normalized.serialize_with_miniblock_repdef_budget(
            miniblock::max_repdef_levels_per_chunk,
            num_rows,
            num_values,
        )?;
        let validity = PrimitiveStructuralEncoder::leaf_validity(&repdef, num_values as usize)?;
        let no_values = num_values == 0
            || validity.as_ref().is_some_and(|v| v.count_set_bits() == 0)
            || matches!(ctx.field.data_type(), DataType::Struct(fields) if fields.is_empty());
        let constant = if no_values || trivial || !inline_type(&ctx.field.data_type()) {
            None
        } else {
            PrimitiveStructuralEncoder::find_constant_scalar(arrays, validity.as_ref())?
                .and_then(|scalar| lance_arrow::scalar::try_inline_value(&scalar))
        };
        let nontrivial = repdef.repetition_levels.is_some() || repdef.definition_levels.is_some();
        let sparse_for_budget = requested.is_none()
            && matches!(
                budget,
                crate::repdef::MiniBlockRepDefBudget::RequiresPageSplit(_)
                    | crate::repdef::MiniBlockRepDefBudget::SingleRowOverBudget(_)
            );
        // Wide mostly-null leaves benefit from per-value addressing even when the
        // dense level stream fits a page. Explicit dense requests remain authoritative.
        let sparse_wide = requested.is_none()
            && validity
                .as_ref()
                .is_some_and(|v| v.count_set_bits() < v.len() / 2)
            && !is_narrow(arrays, num_values);
        if (nontrivial
            && (explicit_sparse || sparse_for_budget || sparse_wide || constant.is_some()))
            || no_values
        {
            return Ok(Some(vec![PrimitivePageData {
                arrays: arrays.to_vec(),
                structure: PrimitivePageStructure::Sparse {
                    plan: sparse::writer::plan(normalized, num_values)?,
                    constant,
                },
                row_number,
                num_rows,
            }]));
        }
        Ok(Some(
            PrimitiveStructuralEncoder::split_pages_for_miniblock_repdef_budget(
                arrays.to_vec(),
                repdef,
                budget,
                row_number,
                num_rows,
            )?,
        ))
    }

    fn try_encode_page(
        &self,
        ctx: &PrimitiveEncodeContext,
        page: PrimitivePageData,
    ) -> Result<PrimitiveEncodeAttempt> {
        if matches!(page.structure, PrimitivePageStructure::Sparse { .. }) {
            let PrimitivePageData {
                arrays,
                structure: PrimitivePageStructure::Sparse { plan, constant },
                row_number,
                num_rows,
            } = page
            else {
                return Err(Error::internal(
                    "Planar sparse page lost its structural plan",
                ));
            };
            return Ok(PrimitiveEncodeAttempt::Encoded(sparse::planar::encode(
                ctx,
                arrays,
                plan,
                constant,
                row_number,
                num_rows,
                self.compression.clone(),
            )?));
        }
        // Nontrivial constant pages were selected before dense page splitting.
        // Only all-valid pages still need constant detection here.
        let all_valid = matches!(&page.structure, PrimitivePageStructure::Dense { repdef, .. }
            if repdef.repetition_levels.is_none() && repdef.definition_levels.is_none());
        let attempt = if all_valid && inline_type(&ctx.field.data_type()) {
            ConstantPrimitiveEncoding.try_encode_page(ctx, page)?
        } else {
            PrimitiveEncodeAttempt::Unhandled(page)
        };
        let page = match attempt {
            PrimitiveEncodeAttempt::Encoded(mut page) => {
                let PageEncoding::Structural(layout) = page.description else {
                    return Err(Error::internal(
                        "Dense constant encoder returned unexpected metadata",
                    ));
                };
                // Arbitrary scalar buffers remain a value-engine concern. Encode
                // these with the ordinary dense path instead of inventing a second
                // scalar wire representation in the planar format.
                if matches!(&layout.layout, Some(metadata::page_layout::Layout::ConstantLayout(c)) if c.inline_value.is_some())
                {
                    page.description = PageEncoding::Planar(from_dense_execution(layout)?);
                    return Ok(PrimitiveEncodeAttempt::Encoded(page));
                }
                return Err(Error::internal(
                    "Non-inline constant reached the planar constant encoder",
                ));
            }
            PrimitiveEncodeAttempt::Unhandled(page) => page,
        };
        let mut dense_ctx = ctx.clone();
        if dense_ctx
            .encoding_metadata
            .get(STRUCTURAL_ENCODING_META_KEY)
            .is_some_and(|v| v.eq_ignore_ascii_case(STRUCTURAL_ENCODING_SPARSE))
        {
            let mut metadata = (*dense_ctx.encoding_metadata).clone();
            metadata.remove(STRUCTURAL_ENCODING_META_KEY);
            dense_ctx.encoding_metadata = Arc::new(metadata);
        }
        let mut encoded = PrimitiveStructuralEncoder::encode_dense_page(
            dense_ctx,
            page,
            self.compression.clone(),
            MiniblockChunkSize::U32,
            ComplexNullEncoding::CompressedLevels,
            FixedWidthDictionaryEncoding::Include64Bit,
        )?;
        let PageEncoding::Structural(layout) = encoded.description else {
            return Err(Error::internal(
                "Dense encoder returned unexpected metadata",
            ));
        };
        encoded.description = PageEncoding::Planar(from_dense_execution(layout)?);
        Ok(PrimitiveEncodeAttempt::Encoded(encoded))
    }
}

/// Validate the planar envelope before entering shared value engines. These
/// checks belong to 2.3 and do not change the stable readers' accepted grammar.
pub(super) fn validate_page(
    page: &PageInfo,
    layout: &crate::format::pb23::PlanarLayout,
    field: &Field,
    decompressors: &dyn DecompressionStrategy,
) -> Result<()> {
    use crate::format::pb23::{structural_plane::Codec as Structure, value_plane::Codec as Values};
    let structure = layout
        .structure
        .as_ref()
        .and_then(|s| s.codec.as_ref())
        .ok_or_else(|| Error::invalid_input("Planar page is missing structure"))?;
    let values = layout
        .values
        .as_ref()
        .and_then(|s| s.codec.as_ref())
        .ok_or_else(|| Error::invalid_input("Planar page is missing values"))?;
    for &(position, size) in page.buffer_offsets_and_sizes.iter() {
        position
            .checked_add(size)
            .ok_or_else(|| Error::invalid_input("Planar buffer range overflows"))?;
    }
    let dense = matches!(structure, Structure::Dense(_));
    let expected_buffers = match values {
        Values::Chunked(v) => {
            let compression = v
                .compression
                .as_ref()
                .ok_or_else(|| Error::invalid_input("Planar chunked compression is missing"))?;
            let decoder =
                decompressors.create_miniblock_decompressor(compression, decompressors)?;
            if v.num_buffers != decoder.num_buffers() as u64 {
                return Err(Error::invalid_input(format!(
                    "Planar chunk declares {} value buffers, codec requires {}",
                    v.num_buffers,
                    decoder.num_buffers()
                )));
            }
            if v.dictionary.is_some() != (v.num_dictionary_items > 0) {
                return Err(Error::invalid_input(
                    "Planar dictionary count does not match its encoding",
                ));
            }
            if let Some(levels) = &v.levels
                && levels.repetition_index_depth > u16::MAX as u32
            {
                return Err(Error::invalid_input("Planar repetition depth exceeds u16"));
            }
            2 + usize::from(v.dictionary.is_some())
                + usize::from(
                    v.levels
                        .as_ref()
                        .is_some_and(|l| l.repetition_index_depth > 0),
                )
        }
        Values::Zipped(v) => {
            if v.compression.is_none() {
                return Err(Error::invalid_input("Planar zipped compression is missing"));
            }
            if v.levels
                .as_ref()
                .is_some_and(|l| l.bits_repetition > 16 || l.bits_definition > 16)
            {
                return Err(Error::invalid_input("Planar control levels exceed 16 bits"));
            }
            match v.addressing {
                Some(crate::format::pb23::zipped_values::Addressing::BitsPerValue(bits))
                    if bits > 0 && bits.is_multiple_of(8) => {}
                Some(crate::format::pb23::zipped_values::Addressing::BitsPerOffset(bits))
                    if [32, 64].contains(&bits) => {}
                _ => return Err(Error::invalid_input("Planar zipped addressing is invalid")),
            }
            1 + usize::from(
                matches!(
                    v.addressing,
                    Some(crate::format::pb23::zipped_values::Addressing::BitsPerOffset(_))
                ) || v.levels.as_ref().is_some_and(|l| l.bits_repetition > 0),
            )
        }
        Values::Indexed(_) => {
            return Err(Error::not_supported(
                "Indexed planar values are not implemented",
            ));
        }
        Values::Constant(v) => {
            let dt = field.data_type();
            let width = if dt == DataType::Boolean {
                Some(1)
            } else {
                dt.byte_width_opt()
            };
            if v.scalar.is_empty()
                || v.scalar.len() > 32
                || width != Some(v.scalar.len())
                || matches!(dt, DataType::FixedSizeList(_, _) | DataType::Struct(_))
            {
                return Err(Error::invalid_input(
                    "Planar scalar length does not match its leaf type",
                ));
            }
            0
        }
        Values::None(_) => 0,
    };
    if dense && page.buffer_offsets_and_sizes.len() != expected_buffers {
        return Err(Error::invalid_input(format!(
            "Planar dense page has {} buffers, expected {expected_buffers}",
            page.buffer_offsets_and_sizes.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{TestCases, TestEncoding, check_round_trip_encoding_of_data};
    use arrow_array::{FixedSizeListArray, Int32Array, StringArray};
    use arrow_buffer::NullBuffer;

    #[rstest::rstest]
    #[case::dense(false, false)]
    #[case::sparse(true, false)]
    #[case::wide_sparse(true, true)]
    #[tokio::test]
    async fn planar_round_trip(#[case] sparse: bool, #[case] wide: bool) {
        let array: ArrayRef = if wide {
            Arc::new(StringArray::from_iter((0..40).map(|i| {
                (i % 11 == 0).then(|| format!("{i:04}{}", "x".repeat(1024)))
            })))
        } else {
            Arc::new(Int32Array::from_iter(
                (0..40).map(|i| (!sparse || i % 3 == 0).then_some(i)),
            ))
        };
        let cases = TestCases::default()
            .with_encoding(TestEncoding::StructuralSparse)
            .with_page_sizes(vec![1 << 20])
            .with_ingest_batch_counts([1, 3])
            .with_range(2..27)
            .with_indices(vec![0, 1, 11, 22, 39])
            .with_batch_size(7)
            .with_verify_encoding(Arc::new(move |columns, _| {
                for column in columns {
                    for page in &column.final_pages {
                        assert!(matches!(page.description, PageEncoding::Planar(_)));
                    }
                }
            }));
        let metadata = if sparse {
            HashMap::from([(STRUCTURAL_ENCODING_META_KEY.to_owned(), "sparse".to_owned())])
        } else {
            HashMap::new()
        };
        check_round_trip_encoding_of_data(vec![array], &cases, metadata).await;
    }

    #[rstest::rstest]
    #[case::constant(false)]
    #[case::none(true)]
    #[tokio::test]
    async fn planar_sparse_constant_and_none(#[case] all_null: bool) {
        let array: ArrayRef = Arc::new(Int32Array::from_iter(
            (0..40).map(|i| (!all_null && i % 3 == 0).then_some(42)),
        ));
        let cases = TestCases::default()
            .with_encoding(TestEncoding::StructuralSparse)
            .with_page_sizes(vec![1 << 20])
            .with_ingest_batch_counts([1])
            .with_range(2..27)
            .with_indices(vec![0, 1, 3, 39])
            .with_batch_size(7)
            .with_verify_encoding(Arc::new(move |columns, _| {
                for column in columns {
                    for page in &column.final_pages {
                        let PageEncoding::Planar(layout) = &page.description else {
                            panic!("Expected planar page");
                        };
                        assert!(layout.sparse().is_some());
                        let values = layout.values.as_ref().unwrap().codec.as_ref().unwrap();
                        assert_eq!(
                            matches!(values, crate::format::pb23::value_plane::Codec::None(_)),
                            all_null
                        );
                        if !all_null {
                            assert!(matches!(
                                values,
                                crate::format::pb23::value_plane::Codec::Constant(_)
                            ));
                        }
                    }
                }
            }));
        check_round_trip_encoding_of_data(vec![array], &cases, HashMap::new()).await;
    }

    #[tokio::test]
    async fn planar_sparse_wide_fixed_size_list() {
        let values: ArrayRef = Arc::new(Int32Array::from_iter_values(0..4096));
        let array: ArrayRef = Arc::new(FixedSizeListArray::new(
            Arc::new(ArrowField::new("item", DataType::Int32, true)),
            128,
            values,
            Some(NullBuffer::from_iter((0..32).map(|i| i % 7 == 0))),
        ));
        let cases = TestCases::default()
            .with_encoding(TestEncoding::StructuralSparse)
            .with_page_sizes(vec![1 << 20])
            .with_ingest_batch_counts([1])
            .with_range(2..27)
            .with_indices(vec![0, 1, 7, 14, 31])
            .with_batch_size(7);
        check_round_trip_encoding_of_data(
            vec![array],
            &cases,
            HashMap::from([(STRUCTURAL_ENCODING_META_KEY.to_owned(), "sparse".to_owned())]),
        )
        .await;
    }
    #[rstest::rstest]
    #[case::struct_only(false)]
    #[case::inside_list(true)]
    #[tokio::test]
    async fn sparse_zipped_respects_null_ancestors(#[case] inside_list: bool) {
        use arrow_array::{ListArray, StructArray};
        use arrow_buffer::{OffsetBuffer, ScalarBuffer};
        let values: ArrayRef =
            Arc::new(StringArray::from_iter((0..12).map(|i| {
                (i % 5 != 0).then(|| format!("{i}{}", "x".repeat(1024)))
            })));
        let structs: ArrayRef = Arc::new(StructArray::new(
            vec![Arc::new(ArrowField::new("value", DataType::Utf8, true))].into(),
            vec![values],
            Some(NullBuffer::from_iter((0..12).map(|i| i % 3 != 0))),
        ));
        let array = if inside_list {
            Arc::new(ListArray::new(
                Arc::new(ArrowField::new("item", structs.data_type().clone(), true)),
                OffsetBuffer::new(ScalarBuffer::from(vec![0, 2, 2, 5, 8, 10, 12])),
                structs,
                Some(NullBuffer::from(vec![true, false, true, true, false, true])),
            )) as ArrayRef
        } else {
            structs
        };
        let cases = TestCases::default()
            .with_encoding(TestEncoding::StructuralSparse)
            .with_page_sizes(vec![1 << 20])
            .with_ingest_batch_counts([1, 3])
            .with_range(1..5)
            .with_indices(vec![0, 1, 3, 5])
            .with_batch_size(2)
            .with_verify_encoding(Arc::new(|columns, _| {
                for column in columns {
                    for page in &column.final_pages {
                        let PageEncoding::Planar(layout) = &page.description else {
                            panic!("expected planar page");
                        };
                        assert!(layout.sparse().is_some());
                        assert!(matches!(
                            layout.values.as_ref().unwrap().codec,
                            Some(crate::format::pb23::value_plane::Codec::Zipped(_))
                        ));
                    }
                }
            }));
        check_round_trip_encoding_of_data(
            vec![array],
            &cases,
            HashMap::from([(STRUCTURAL_ENCODING_META_KEY.to_owned(), "sparse".to_owned())]),
        )
        .await;
    }
}
