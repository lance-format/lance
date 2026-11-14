// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Range extraction with one admission for the leaf and every nested child bitmap.

use std::ops::Range;

use arrow_buffer::{Buffer, bit_mask};
use lance_core::{Error, Result, utils::allocation::Reservation};

use super::{ValueDecompressor, validate_fixed_buffer};
use crate::{
    buffer::LanceBuffer,
    compression::MiniBlockDecompressor,
    data::{BlockInfo, DataBlock, FixedSizeListBlock, NullableDataBlock},
};

fn scaled(range: &Range<u64>, dimension: u64) -> Result<Range<u64>> {
    let start = range.start.checked_mul(dimension);
    let end = range.end.checked_mul(dimension);
    match (start, end) {
        (Some(start), Some(end)) if dimension > 0 => Ok(start..end),
        _ => Err(Error::invalid_input("FSL selection or dimension overflows")),
    }
}

fn bit_count(values: u64, bits: u64) -> Result<usize> {
    values
        .checked_mul(bits)
        .and_then(|count| usize::try_from(count).ok())
        .ok_or_else(|| Error::invalid_input("Flat/FSL bit range exceeds addressable size"))
}

fn output_bytes(range: &Range<u64>, bits: u64) -> Result<usize> {
    Ok(bit_count(range.end - range.start, bits)?.div_ceil(8))
}

fn copy_selected(
    input: &LanceBuffer,
    range: &Range<u64>,
    bits: u64,
    workspace: &mut Reservation,
) -> Result<LanceBuffer> {
    let start_bit = bit_count(range.start, bits)?;
    let len_bits = bit_count(range.end - range.start, bits)?;
    let mut output = workspace
        .try_allocate(len_bits.div_ceil(8))?
        .ok_or_else(|| Error::internal("Flat/FSL workspace changed after complete admission"))?;
    if start_bit.is_multiple_of(8) && len_bits.is_multiple_of(8) {
        output
            .as_mut()
            .copy_from_slice(&input[start_bit / 8..(start_bit + len_bits) / 8]);
    } else {
        // The complete stored layout and selected endpoints were checked before
        // admission. Arrow's bounded bit copy uses no heap scratch and preserves
        // zero padding in the initialized destination's final byte.
        bit_mask::set_bits(output.as_mut(), input, 0, start_bit, len_bits);
    }
    Ok(LanceBuffer::from(Buffer::from(output.freeze())))
}

pub(super) fn decode(
    codec: &ValueDecompressor,
    data: &[LanceBuffer],
    num_values: u64,
    range: Range<u64>,
    workspace: &mut Reservation,
) -> Result<Option<DataBlock>> {
    if range.start > range.end || range.end > num_values {
        return Err(Error::corrupt_file_named(
            "mini-block",
            format!("Flat/FSL selection {range:?} exceeds {num_values} values"),
        ));
    }
    if codec.bits_per_item == 0 {
        // The storage sentinel does not describe the eventual typed Arrow layout.
        // Do not claim zero output bytes and materialize that layout unaccounted later.
        return Err(Error::not_supported(
            "admitted zero-width leaf needs a typed output layout",
        ));
    }
    if data.len() != codec.num_buffers() {
        return Err(Error::invalid_input(format!(
            "Flat/FSL mini-block has {} buffers, expected {}",
            data.len(),
            codec.num_buffers()
        )));
    }
    let mut stored = num_values;
    let mut selected = range;
    let mut total = 0_usize;
    let mut buffer_index = 0;
    // Validate all input lengths and row/child geometry before reserving any output.
    // Revisit the layers below, avoiding even a per-selection plan Vec before admission.
    for layer in &codec.layers {
        stored = stored
            .checked_mul(layer.dimension)
            .ok_or_else(|| Error::invalid_input("FSL child value count overflows"))?;
        selected = scaled(&selected, layer.dimension)?;
        if layer.has_validity {
            validate_fixed_buffer(&data[buffer_index], stored, 1)?;
            bit_count(stored, 1)?;
            total = total
                .checked_add(output_bytes(&selected, 1)?)
                .ok_or_else(|| Error::invalid_input("FSL output size overflows"))?;
            buffer_index += 1;
        }
    }
    validate_fixed_buffer(&data[buffer_index], stored, codec.bits_per_item)?;
    bit_count(stored, codec.bits_per_item)?;
    // set_bits uses byte-length * 8 for bounds checks; reject unaddressable padding too.
    for buffer in data {
        buffer
            .len()
            .checked_mul(8)
            .ok_or_else(|| Error::invalid_input("Flat/FSL padded bit length overflows"))?;
    }
    total = total
        .checked_add(output_bytes(&selected, codec.bits_per_item)?)
        .ok_or_else(|| Error::invalid_input("Flat/FSL output size overflows"))?;
    if total > workspace.len() {
        return Ok(None);
    }
    let values = copy_selected(
        &data[buffer_index],
        &selected,
        codec.bits_per_item,
        workspace,
    )?;
    let mut output = codec.buffer_to_block(values, selected.end - selected.start);
    for layer in codec.layers.iter().rev() {
        if layer.has_validity {
            buffer_index -= 1;
            output = DataBlock::Nullable(NullableDataBlock {
                data: Box::new(output),
                nulls: copy_selected(&data[buffer_index], &selected, 1, workspace)?,
                block_info: BlockInfo::new(),
            });
        }
        output = DataBlock::FixedSizeList(FixedSizeListBlock {
            child: Box::new(output),
            dimension: layer.dimension,
        });
        selected = selected.start / layer.dimension..selected.end / layer.dimension;
    }
    Ok(Some(output))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        Array, ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, make_array,
    };
    use arrow_buffer::{BooleanBuffer, NullBuffer};
    use arrow_schema::{DataType, Field};
    use lance_core::utils::allocation::AllocationBudget;
    use rstest::rstest;

    use super::*;
    use crate::{
        encodings::{
            logical::primitive::miniblock::{MiniBlockCompressionContext, MiniBlockCompressor},
            physical::value::ValueEncoder,
        },
        format::{
            EncodingUtils,
            metadata::{CompressiveEncoding, compressive_encoding::Compression},
        },
    };

    fn codec(encoding: CompressiveEncoding) -> ValueDecompressor {
        match encoding.compression.unwrap() {
            Compression::Flat(flat) => ValueDecompressor::from_flat(&flat),
            Compression::FixedSizeList(fsl) => ValueDecompressor::from_fsl(&fsl).unwrap(),
            other => panic!("unexpected fixture encoding: {other:?}"),
        }
    }

    fn fixture(depth: usize, boolean: bool, nullable_layers: u8) -> ArrayRef {
        let dimensions = &[3, 2, 3][..depth];
        let count = 32 * dimensions.iter().product::<usize>();
        let valid = |i| depth == 0 || nullable_layers & (1 << (depth - 1)) == 0 || i % 5 != 1;
        let mut values: ArrayRef = if boolean {
            Arc::new(BooleanArray::from_iter(
                (0..count).map(|i| valid(i).then_some(i % 3 == 0)),
            ))
        } else {
            Arc::new(Float32Array::from_iter(
                (0..count).map(|i| valid(i).then_some(i as f32 * 0.25 - 17.0)),
            ))
        };
        for (level, &dimension) in dimensions.iter().enumerate().rev() {
            let rows = values.len() / dimension;
            let nulls = (level > 0 && nullable_layers & (1 << (level - 1)) != 0).then(|| {
                NullBuffer::new(BooleanBuffer::from(
                    (0..rows).map(|i| i % 3 != 1).collect::<Vec<_>>(),
                ))
            });
            values = Arc::new(FixedSizeListArray::new(
                Arc::new(Field::new("item", values.data_type().clone(), true)),
                dimension as i32,
                values,
                nulls,
            ));
        }
        values
    }

    // Price the expected Arrow payload from the fixture, independently of codec metadata.
    fn selected_payload(array: &dyn Array, rows: usize) -> usize {
        let validity = if array.nulls().is_some() {
            rows.div_ceil(8)
        } else {
            0
        };
        validity
            + match array.data_type() {
                DataType::FixedSizeList(_, dimension) => {
                    let list = array.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
                    selected_payload(list.values().as_ref(), rows * *dimension as usize)
                }
                DataType::Float32 => rows * 4,
                DataType::Boolean => rows.div_ceil(8),
                other => panic!("unexpected fixture type: {other}"),
            }
    }

    fn check_selection(
        expected: ArrayRef,
        codec: ValueDecompressor,
        encoded: Vec<LanceBuffer>,
        range: Range<u64>,
        unaligned: bool,
    ) {
        let output_bytes = selected_payload(expected.as_ref(), (range.end - range.start) as usize);
        let prefix = usize::from(unaligned);
        let input_bytes = prefix + encoded.iter().map(|buffer| buffer.len()).sum::<usize>();
        let budget = AllocationBudget::new(input_bytes + output_bytes);
        let mut workspace = budget.try_reserve(input_bytes + output_bytes).unwrap();
        let mut input = workspace.try_allocate(input_bytes).unwrap().unwrap();
        let mut cursor = prefix;
        for buffer in &encoded {
            input.as_mut()[cursor..cursor + buffer.len()].copy_from_slice(buffer);
            cursor += buffer.len();
        }
        let source = input.freeze();
        cursor = prefix;
        let buffers = encoded
            .iter()
            .map(|buffer| {
                let bytes = source.slice(cursor..cursor + buffer.len());
                cursor += buffer.len();
                LanceBuffer::try_from_bytes_in(&bytes, 1, &mut workspace)
                    .unwrap()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        drop(encoded);

        if output_bytes > 0 {
            let mut short = workspace.try_split(output_bytes - 1).unwrap();
            assert!(
                codec
                    .try_decompress_range_in(
                        &buffers,
                        expected.len() as u64,
                        range.clone(),
                        &mut short,
                    )
                    .unwrap()
                    .is_none()
            );
            assert_eq!(short.len(), output_bytes - 1);
            assert_eq!(budget.reserved(), input_bytes + output_bytes);
            drop(short);
            drop(workspace);
            workspace = budget.try_reserve(output_bytes).unwrap();
        }
        let output = codec
            .try_decompress_range_in(
                &buffers,
                expected.len() as u64,
                range.clone(),
                &mut workspace,
            )
            .unwrap()
            .unwrap();
        assert!(workspace.is_empty());
        assert_eq!(budget.reserved(), input_bytes + output_bytes);
        let actual_data = output
            .into_arrow(expected.data_type().clone(), true)
            .unwrap();
        // Arrow may discard an all-valid bitmap, but must not create an uncharged payload.
        let retained = actual_data.get_buffer_memory_size();
        assert!(retained <= output_bytes);
        assert_eq!(budget.reserved(), input_bytes + retained);
        let actual = make_array(actual_data);
        assert_eq!(
            actual.to_data(),
            expected.slice(range.start as usize, actual.len()).to_data()
        );
        let last = actual.slice(0, usize::from(!actual.is_empty()));
        drop(actual);
        drop(workspace);
        drop(buffers);
        drop(source);
        assert_eq!(budget.reserved(), retained);
        drop(last);
        assert_eq!(budget.reserved(), 0);
    }

    #[rstest]
    #[case::empty(7..7)]
    #[case::partial(3..27)]
    #[case::full(0..32)]
    #[case::tail(31..32)]
    fn encoded_ranges_preserve_values_and_owner_credit(
        #[case] range: Range<u64>,
        #[values(0, 1, 3)] depth: usize,
        #[values(false, true)] boolean: bool,
        #[values(false, true)] unaligned: bool,
    ) {
        let expected = fixture(depth, boolean, u8::MAX);
        let (encoded, encoding) = MiniBlockCompressor::compress(
            &ValueEncoder::default(),
            MiniBlockCompressionContext::new(0, true, true),
            DataBlock::from_array(expected.clone()),
        )
        .unwrap();
        assert_eq!(encoded.chunks.len(), 1);
        check_selection(expected, codec(encoding), encoded.data, range, unaligned);
    }

    #[rstest]
    #[case::no_bitmaps(0)]
    #[case::only_middle_bitmap(2)]
    #[case::outer_and_leaf_bitmaps(5)]
    fn omitted_child_bitmaps_keep_buffer_order(
        #[case] nullable_layers: u8,
        #[values(false, true)] boolean: bool,
    ) {
        let expected = fixture(3, boolean, nullable_layers);
        let (encoded, encoding) = MiniBlockCompressor::compress(
            &ValueEncoder::default(),
            MiniBlockCompressionContext::new(0, true, true),
            DataBlock::from_array(expected.clone()),
        )
        .unwrap();
        assert_eq!(encoded.chunks.len(), 1);
        check_selection(expected, codec(encoding), encoded.data, 3..27, true);
    }

    #[rstest]
    #[case::all_null(false)]
    #[case::all_valid(true)]
    fn raw_child_validity_preserves_typed_values(#[case] valid: bool) {
        // Construct raw storage explicitly: the encoder may choose a constant sentinel
        // for all-null children, which is a separate typed materialization capability.
        let values = Float32Array::from_iter_values((0..96).map(|i| i as f32));
        let expected_values = Float32Array::new(
            values.values().clone(),
            Some(NullBuffer::from(vec![valid; 96])),
        );
        let expected = Arc::new(FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            3,
            Arc::new(expected_values),
            None,
        ));
        let encoded = vec![
            LanceBuffer::from(vec![if valid { 255 } else { 0 }; 12]),
            LanceBuffer::from(values.values().inner().clone()),
        ];
        check_selection(
            expected,
            codec(EncodingUtils::fsl(3, true, EncodingUtils::flat(32, None))),
            encoded,
            3..27,
            true,
        );
    }

    #[rstest]
    #[case::truncated_validity(0, "Flat buffer")]
    #[case::truncated_leaf(1, "Flat buffer")]
    #[case::missing_buffer(2, "buffers, expected")]
    #[case::extra_buffer(3, "buffers, expected")]
    #[case::range_past_end(4, "selection")]
    fn invalid_layout_does_not_consume_credit(#[case] kind: u8, #[case] message: &str) {
        let decoder = codec(EncodingUtils::fsl(3, true, EncodingUtils::flat(32, None)));
        let mut buffers = vec![
            LanceBuffer::from(vec![255; 12]),
            LanceBuffer::from(vec![0; 384]),
        ];
        match kind {
            0 | 1 => {
                let index = kind as usize;
                buffers[index] = buffers[index].slice_with_length(0, buffers[index].len() - 1);
            }
            2 => {
                buffers.pop();
            }
            3 => buffers.push(LanceBuffer::from(vec![0])),
            _ => (),
        }
        let budget = AllocationBudget::new(1024);
        let mut workspace = budget.try_reserve(1024).unwrap();
        let range = if kind == 4 { 31..33 } else { 3..27 };
        let error = decoder
            .try_decompress_range_in(&buffers, 32, range, &mut workspace)
            .unwrap_err();
        assert!(matches!(
            error,
            Error::InvalidInput { .. } | Error::CorruptFile { .. }
        ));
        assert!(error.to_string().contains(message), "{error}");
        assert_eq!(workspace.len(), 1024);
        assert_eq!(budget.reserved(), 1024);
    }

    #[test]
    fn child_count_overflow_does_not_allocate() {
        let decoder = codec(EncodingUtils::fsl(3, false, EncodingUtils::flat(32, None)));
        let budget = AllocationBudget::new(0);
        let mut workspace = budget.try_reserve(0).unwrap();
        let error = decoder
            .try_decompress_range_in(&[LanceBuffer::from(vec![])], u64::MAX, 0..1, &mut workspace)
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("child value count overflows"));
        assert_eq!(budget.reserved(), 0);
    }

    #[test]
    fn zero_width_requires_typed_materialization_capability() {
        let decoder = codec(EncodingUtils::flat(0, None));
        let budget = AllocationBudget::new(0);
        let mut workspace = budget.try_reserve(0).unwrap();
        let error = decoder
            .try_decompress_range_in(&[LanceBuffer::from(vec![])], 32, 3..27, &mut workspace)
            .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(error.to_string().contains("typed output layout"));
        assert_eq!(budget.reserved(), 0);
    }
}
