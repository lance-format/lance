// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::{
    encodings::physical::{
        constant::ConstantBlockDecompressor, delta::DeltaDecompressor, range::RangeDecompressor,
    },
    format::{EncodingUtils, metadata::CompressiveEncoding},
};

fn description(bits: u8, kind: u8) -> CompressiveEncoding {
    let value = if kind == 0 || kind == 3 { 0_u64 } else { 3 };
    let constant = EncodingUtils::constant(Some(bytes::Bytes::copy_from_slice(
        &value.to_le_bytes()[..usize::from(bits / 8)],
    )));
    let offsets = match kind {
        0 => constant,
        1 => EncodingUtils::range(u64::from(bits), 0, 3),
        _ => EncodingUtils::delta(u64::from(bits), 0, constant),
    };
    EncodingUtils::variable(offsets, None)
}

fn decoder(bits: u8, kind: u8) -> BinaryMiniBlockDecompressor {
    let encoding = description(bits, kind);
    let Some(Compression::Variable(variable)) = encoding.compression else {
        panic!("expected Variable")
    };
    BinaryMiniBlockDecompressor::from_variable(&variable).unwrap()
}

#[rstest]
#[case::full(0..4)]
#[case::middle(1..3)]
#[case::empty(2..2)]
fn metadata_binary_ranges_need_only_selected_output(
    #[case] range: Range<u64>,
    #[values(32, 64)] bits: u8,
    #[values(0, 1, 2, 3)] kind: u8,
) {
    let all: &[&[u8]] = if kind == 0 || kind == 3 {
        &[b"", b"", b"", b""]
    } else {
        &[b"abc", b"def", b"ghi", b"jkl"]
    };
    let raw = all.concat();
    let selected = &all[range.start as usize..range.end as usize];
    let required = (selected.len() + 1) * usize::from(bits / 8)
        + selected.iter().map(|v| v.len()).sum::<usize>();
    let physical = raw.len() + 41;
    let pool = AllocationBudget::new(physical + required);
    let mut work = pool.try_reserve(physical + required).unwrap();
    let mut input = work.try_allocate(physical).unwrap().unwrap();
    input.as_mut()[1..1 + raw.len()].copy_from_slice(&raw);
    let owner = Buffer::from(input.freeze());
    let sibling = owner.slice_with_length(physical - 1, 1);
    let data = [LanceBuffer::from(owner.slice_with_length(1, raw.len()))];
    drop(owner);
    let codec = decoder(bits, kind);
    let plan = codec.plan_range(&data, 4, range.clone()).unwrap();
    assert_eq!(plan.output_bytes, required);
    assert_eq!(plan.workspace_bytes, required);
    let mut short = work.try_split(required - 1).unwrap();
    for _ in 0..2 {
        assert!(
            codec
                .try_decompress_range_in(&data, 4, range.clone(), &mut short)
                .unwrap()
                .is_none()
        );
        assert_eq!(short.len(), required - 1);
        assert_eq!(pool.reserved(), physical + required);
    }
    drop(short);
    drop(work);
    let mut work = pool.try_reserve(required).unwrap();
    let result = codec
        .try_decompress_range_in(&data, 4, range, &mut work)
        .unwrap()
        .unwrap();
    assert!(work.is_empty());
    let ty = if bits == 32 {
        DataType::Binary
    } else {
        DataType::LargeBinary
    };
    let actual = result.try_to_arrow_in(&ty, &mut work).unwrap().unwrap();
    assert_eq!(actual, expected(selected, bits));
    assert_eq!(actual.get_buffer_memory_size(), required);
    drop(result);
    drop(data);
    drop(work);
    assert_eq!(pool.reserved(), physical + required);
    drop(sibling);
    assert_eq!(pool.reserved(), required);
    drop(actual);
    assert_eq!(pool.reserved(), 0);
}

#[rstest]
fn many_empty_values_do_not_materialize_all_offsets(#[values(32, 64)] bits: u8) {
    let codec = decoder(bits, 0);
    let count = 1_u64 << 40;
    let required = 4 * usize::from(bits / 8);
    let pool = AllocationBudget::new(required);
    let mut work = pool.try_reserve(required).unwrap();
    let data = [LanceBuffer::from(Vec::<u8>::new())];
    let result = codec
        .try_decompress_range_in(&data, count, count - 3..count, &mut work)
        .unwrap()
        .unwrap();
    let block = result.as_variable_width_ref().unwrap();
    assert_eq!(block.num_values, 3);
    assert!(block.data.is_empty());
    assert_eq!(block.offsets.as_ref(), vec![0; required]);
    assert!(work.is_empty());
    drop(result);
    drop(work);
    assert_eq!(pool.reserved(), 0);
}

#[rstest]
fn current_writer_metadata_offsets_round_trip(
    #[values(32, 64)] bits: u8,
    #[values(false, true)] empty: bool,
) {
    let value: &[u8] = if empty { b"" } else { b"abc" };
    let values = vec![value; 2048];
    let array: arrow_array::ArrayRef = if bits == 32 {
        std::sync::Arc::new(BinaryArray::from_vec(values))
    } else {
        std::sync::Arc::new(LargeBinaryArray::from_vec(values))
    };
    let (encoded, encoding) =
        BinaryMiniBlockEncoder::with_generic_offsets(Some(256), CompressionFieldParams::default())
            .compress(
                MiniBlockCompressionContext::new(0, true, true),
                DataBlock::from(array),
            )
            .unwrap();
    let Some(Compression::Variable(variable)) = encoding.compression else {
        panic!("expected Variable")
    };
    assert!(matches!(
        variable.offsets.as_ref().unwrap().compression,
        Some(Compression::Range(_) | Compression::Constant(_))
    ));
    assert_eq!(encoded.data.len(), 1);
    let first = &encoded.chunks[0];
    let count = if first.log_num_values == 0 {
        encoded.num_values
    } else {
        1_u64 << first.log_num_values
    };
    assert!(count >= 3);
    let data = [encoded.data[0].slice_with_length(0, first.buffer_sizes[0] as usize)];
    let codec = BinaryMiniBlockDecompressor::from_variable(&variable).unwrap();
    let required = 3 * usize::from(bits / 8) + 2 * value.len();
    let pool = AllocationBudget::new(required);
    let mut work = pool.try_reserve(required).unwrap();
    let result = codec
        .try_decompress_range_in(&data, count, 1..3, &mut work)
        .unwrap()
        .unwrap();
    let ty = if bits == 32 {
        DataType::Binary
    } else {
        DataType::LargeBinary
    };
    assert_eq!(
        result.try_to_arrow_in(&ty, &mut work).unwrap().unwrap(),
        expected(&[value, value], bits)
    );
    assert!(work.is_empty());
}

#[rstest]
#[case::nonzero_start(0, "start at zero")]
#[case::wrong_end(1, "value-buffer length")]
#[case::extra_payload(2, "one value buffer")]
#[case::missing_payload(3, "one value buffer")]
#[case::count_overflow(4, "count overflows")]
#[case::reversed_selection(5, "selected range")]
fn invalid_metadata_binary_preserves_credit(
    #[case] fault: u8,
    #[case] message: &str,
    #[values(false, true)] empty_selection: bool,
) {
    let mut codec = decoder(32, 1);
    let mut data = vec![LanceBuffer::from(vec![0_u8; 12])];
    let mut count = 4;
    let mut range = 0..1;
    match fault {
        0 => {
            codec = BinaryMiniBlockDecompressor {
                layout: BinaryMiniBlockLayout::Generic {
                    bits_per_offset: 32,
                    offsets: Box::new(RangeDecompressor::new(32, 1, 3)),
                    offsets_have_payload: false,
                },
            }
        }
        1 => data[0] = LanceBuffer::from(vec![0_u8; 13]),
        2 => data.push(LanceBuffer::from(vec![0_u8; 1])),
        3 => data.clear(),
        4 => count = u64::MAX,
        5 => range = Range { start: 2, end: 1 },
        _ => unreachable!(),
    }
    if empty_selection && fault != 5 {
        range = 0..0;
    }
    let pool = AllocationBudget::new(64);
    let mut work = pool.try_reserve(64).unwrap();
    for _ in 0..2 {
        let error = codec
            .try_decompress_range_in(&data, count, range.clone(), &mut work)
            .unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains(message), "{error}");
        assert_eq!(work.len(), 64);
    }
}

#[rstest]
fn empty_binary_metadata_still_needs_a_leading_offset(#[values(32, 64)] bits: u8) {
    let codec = decoder(bits, 0);
    let width = usize::from(bits / 8);
    let pool = AllocationBudget::new(width);
    let mut work = pool.try_reserve(width).unwrap();
    let data = [LanceBuffer::from(Vec::<u8>::new())];
    let output = codec
        .try_decompress_range_in(&data, 0, 0..0, &mut work)
        .unwrap()
        .unwrap();
    let ty = if bits == 32 {
        DataType::Binary
    } else {
        DataType::LargeBinary
    };
    let actual = output.try_to_arrow_in(&ty, &mut work).unwrap().unwrap();
    assert_eq!(actual, expected(&[], bits));
    assert_eq!(actual.get_buffer_memory_size(), width);
    assert!(work.is_empty());
    drop(output);
    drop(actual);
    drop(work);
    assert_eq!(pool.reserved(), 0);
}

#[rstest]
#[case::range_width(Box::new(RangeDecompressor::new(32, 0, 1)), 64, 4, "width")]
#[case::range_step(Box::new(RangeDecompressor::new(32, 0, 0)), 32, 4, "positive")]
#[case::range_count(Box::new(RangeDecompressor::new(32, 0, 1)), 32, 1, "at least 2")]
#[case::range_overflow(Box::new(RangeDecompressor::new(64, 0, u64::MAX)), 64, 3, "overflows")]
#[case::range_u32(Box::new(RangeDecompressor::new(32, 0, u32::MAX as u64)), 32, 3, "exceeds")]
#[case::constant_value(
    Box::new(ConstantBlockDecompressor::new(32, Some(u64::MAX))),
    32,
    4,
    "exceeds"
)]
#[case::constant_width(Box::new(ConstantBlockDecompressor::new(32, Some(0))), 64, 4, "width")]
#[case::constant_absent(Box::new(ConstantBlockDecompressor::new(32, None)), 32, 4, "absent")]
#[case::constant_present(Box::new(ConstantBlockDecompressor::new(32, Some(0))), 32, 0, "absent")]
#[case::delta_count(
    Box::new(DeltaDecompressor::new(
        32,
        0,
        Box::new(ConstantBlockDecompressor::new(32, Some(1)))
    )),
    32,
    1,
    "at least 2"
)]
#[case::delta_child_width(
    Box::new(DeltaDecompressor::new(
        32,
        0,
        Box::new(ConstantBlockDecompressor::new(64, Some(1)))
    )),
    32,
    4,
    "width"
)]
#[case::delta_overflow(Box::new(DeltaDecompressor::new(32, u32::MAX as u64, Box::new(ConstantBlockDecompressor::new(32, Some(1))))), 32, 2, "exceeds")]
fn metadata_plans_validate_full_sequence_geometry(
    #[case] codec: Box<dyn BlockDecompressor>,
    #[case] bits: u64,
    #[case] count: u64,
    #[case] message: &str,
) {
    let error = codec.plan_unsigned_metadata(bits, count).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains(message), "{error}");
}

#[test]
fn unsupported_metadata_never_uses_ordinary_decode() {
    let codec = DeltaDecompressor::new(32, 0, Box::new(ForbiddenOffsets));
    let error = codec.plan_unsigned_metadata(32, 4).unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("metadata plan"));
    let codec = DeltaDecompressor::new(32, 0, Box::new(RangeDecompressor::new(32, 1, 1)));
    let error = codec.plan_unsigned_metadata(32, 4).unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("constant deltas"));
}
