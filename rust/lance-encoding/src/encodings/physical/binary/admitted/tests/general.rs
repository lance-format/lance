// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::{
    compression::BlockCompressor,
    data::FixedWidthDataBlock,
    encodings::physical::{
        binary::test_utils::uncompressed_general_offsets,
        block::{CompressionConfig, CompressionScheme},
        general::GeneralBlockCompressor,
        value::ValueEncoder,
    },
    format::EncodingUtils,
};

fn encoded(
    bits: u8,
    scheme: CompressionScheme,
    offsets: &[u64],
) -> (BinaryMiniBlockDecompressor, LanceBuffer) {
    let bytes = offsets
        .iter()
        .flat_map(|v| v.to_le_bytes()[..usize::from(bits / 8)].to_vec())
        .collect::<Vec<_>>();
    if scheme == CompressionScheme::None {
        return (uncompressed_general_offsets(bits), LanceBuffer::from(bytes));
    }
    // Use the same General(Flat) writer and descriptor as the sequence candidate.
    let (payload, encoding) = GeneralBlockCompressor::new(
        Box::new(ValueEncoder::default()),
        CompressionConfig::new(scheme, None),
    )
    .compress(DataBlock::FixedWidth(FixedWidthDataBlock {
        data: LanceBuffer::from(bytes),
        num_values: offsets.len() as u64,
        bits_per_value: u64::from(bits),
        block_info: BlockInfo::new(),
    }))
    .unwrap();
    assert!(matches!(
        encoding.compression,
        Some(Compression::General(_))
    ));
    let description = EncodingUtils::variable(encoding, None);
    let Some(Compression::Variable(variable)) = description.compression else {
        unreachable!()
    };
    (
        BinaryMiniBlockDecompressor::from_variable(&variable).unwrap(),
        payload.unwrap(),
    )
}

fn input(
    encoded: &LanceBuffer,
    value_bytes: &[u8],
    capacity: usize,
) -> (
    std::sync::Arc<AllocationBudget>,
    Reservation,
    Vec<LanceBuffer>,
    Buffer,
    usize,
) {
    let physical = encoded.len() + value_bytes.len() + 37;
    let pool = AllocationBudget::new(physical + capacity);
    let mut work = pool.try_reserve(physical + capacity).unwrap();
    let mut input = work.try_allocate(physical).unwrap().unwrap();
    input.as_mut()[1..1 + encoded.len()].copy_from_slice(encoded);
    let start = encoded.len() + 2;
    input.as_mut()[start..start + value_bytes.len()].copy_from_slice(value_bytes);
    let owner = Buffer::from(input.freeze());
    let buffers = vec![
        LanceBuffer::from(owner.slice_with_length(1, encoded.len())),
        LanceBuffer::from(owner.slice_with_length(start, value_bytes.len())),
    ];
    let sibling = owner.slice_with_length(physical - 1, 1);
    (pool, work, buffers, sibling, physical)
}

#[rstest]
#[case::noop(CompressionScheme::None)]
#[cfg_attr(feature = "lz4", case::lz4(CompressionScheme::Lz4))]
fn general_offsets_preserve_ranges_and_complete_overlap(
    #[case] scheme: CompressionScheme,
    #[values(32, 64)] bits: u8,
    #[values(0..4, 1..3, 1..2, 4..4)] range: Range<u64>,
) {
    let values: &[&[u8]] = &[b"a", b"", b"bc", b"d"];
    let selected = &values[range.start as usize..range.end as usize];
    let width = usize::from(bits / 8);
    let scratch = 5 * width;
    let output = (selected.len() + 1) * width + selected.concat().len();
    let capacity = scratch + output;
    let (codec, encoded) = encoded(bits, scheme, &[0, 1, 1, 3, 4]);
    let (pool, mut work, buffers, sibling, physical) = input(&encoded, b"abcd", capacity);
    // Pure planning must stay allocation-free and cannot decompress General.
    assert!(matches!(
        codec.plan_range(&buffers, 4, range.clone()),
        Err(Error::NotSupported { .. })
    ));
    let mut short = work.try_split(capacity - 1).unwrap();
    for _ in 0..2 {
        assert!(
            codec
                .try_prepare_and_decompress_range_in(&buffers, 4, range.clone(), &mut short)
                .unwrap()
                .is_none()
        );
        assert_eq!(short.len(), capacity - 1);
        assert_eq!(pool.reserved(), physical + capacity);
    }
    drop((short, work));
    let mut work = pool.try_reserve(capacity).unwrap();
    let output = codec
        .try_prepare_and_decompress_range_in(&buffers, 4, range, &mut work)
        .unwrap()
        .unwrap();
    assert_eq!(work.len(), scratch);
    let ty = if bits == 32 {
        DataType::Binary
    } else {
        DataType::LargeBinary
    };
    let actual = output.try_to_arrow_in(&ty, &mut work).unwrap().unwrap();
    assert_eq!(actual, expected(selected, bits));
    let output_bytes = actual.get_buffer_memory_size();
    drop((output, buffers, work));
    assert_eq!(pool.reserved(), physical + output_bytes);
    drop(sibling);
    assert_eq!(pool.reserved(), output_bytes);
    drop(actual);
    assert_eq!(pool.reserved(), 0);
}

#[rstest]
#[case::noop(CompressionScheme::None)]
#[cfg_attr(feature = "lz4", case::lz4(CompressionScheme::Lz4))]
fn general_offsets_validate_the_unselected_tail_before_publication(
    #[case] scheme: CompressionScheme,
    #[values(32, 64)] bits: u8,
    #[values([0, 1, 3, 2, 4], [0, 1, 1, 3, 5])] offsets: [u64; 5],
) {
    let (codec, encoded) = encoded(bits, scheme, &offsets);
    let (pool, mut work, buffers, sibling, physical) = input(&encoded, b"abcd", 128);
    for _ in 0..2 {
        let error = codec
            .try_prepare_and_decompress_range_in(&buffers, 4, 0..1, &mut work)
            .unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("offset"), "{error}");
        assert_eq!(work.len(), 128);
        assert_eq!(pool.reserved(), physical + 128);
    }
    drop((work, buffers, sibling));
    assert_eq!(pool.reserved(), 0);
}

#[cfg(feature = "lz4")]
#[rstest]
#[case::short_decoded_frame(false)]
#[case::truncated_frame(true)]
fn general_offsets_reject_malformed_frames_without_leaking_credit(#[case] truncated: bool) {
    let offsets: &[u64] = if truncated {
        &[0, 1, 1, 3, 4]
    } else {
        &[0, 1, 1, 3]
    };
    let (codec, mut encoded) = encoded(32, CompressionScheme::Lz4, offsets);
    if truncated {
        encoded = encoded.slice_with_length(0, encoded.len() - 1);
    }
    let (pool, mut work, buffers, sibling, physical) = input(&encoded, b"abcd", 128);
    for _ in 0..2 {
        let error = codec
            .try_prepare_and_decompress_range_in(&buffers, 4, 0..1, &mut work)
            .unwrap_err();
        assert!(matches!(
            error,
            Error::InvalidInput { .. } | Error::CorruptFile { .. }
        ));
        assert!(
            error.to_string().contains(if truncated {
                "LZ4 decompression error"
            } else {
                "LZ4 declares"
            }),
            "{error}"
        );
        assert_eq!(work.len(), 128);
        assert_eq!(pool.reserved(), physical + 128);
    }
    drop((work, buffers, sibling));
    assert_eq!(pool.reserved(), 0);
}

#[cfg(feature = "zstd")]
#[test]
fn general_offsets_do_not_fall_back_to_unaccounted_zstd() {
    let (codec, encoded) = encoded(32, CompressionScheme::Zstd, &[0, 1, 1, 3, 4]);
    let (pool, mut work, buffers, sibling, physical) = input(&encoded, b"abcd", 128);
    let error = codec
        .try_prepare_and_decompress_range_in(&buffers, 4, 0..1, &mut work)
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(
        error.to_string().contains("no admitted exact decoder"),
        "{error}"
    );
    assert_eq!(work.len(), 128);
    assert_eq!(pool.reserved(), physical + 128);
    drop((work, buffers, sibling));
    assert_eq!(pool.reserved(), 0);
}

#[rstest]
#[case::noop(CompressionScheme::None)]
#[cfg_attr(feature = "lz4", case::lz4(CompressionScheme::Lz4))]
fn general_offsets_preparation_can_be_abandoned_or_recycled(#[case] scheme: CompressionScheme) {
    let (codec, encoded) = encoded(32, scheme, &[0, 1, 1, 3, 4]);
    let (pool, mut work, buffers, sibling, physical) = input(&encoded, b"abcd", 40);
    let first = codec
        .try_prepare_range_in(&buffers, 4, 1..3, &mut work)
        .unwrap()
        .unwrap();
    let second = codec
        .try_prepare_range_in(&buffers, 4, 2..4, &mut work)
        .unwrap()
        .unwrap();
    assert!(work.is_empty());
    first.recycle(&mut work).unwrap();
    assert_eq!(work.len(), 20);
    assert_eq!(pool.reserved(), physical + 40);
    drop(second);
    assert_eq!(work.len(), 20);
    assert_eq!(pool.reserved(), physical + 20);
    drop((work, buffers, sibling));
    assert_eq!(pool.reserved(), 0);
}
