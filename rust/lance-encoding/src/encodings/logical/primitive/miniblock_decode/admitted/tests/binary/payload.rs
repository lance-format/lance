// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::{
    compression::BlockCompressor,
    encodings::physical::{
        dictionary::BlockDictionaryEncoder, general::GeneralBlockCompressor, rle::BlockRleEncoder,
        value::ValueEncoder,
    },
    format::{EncodingUtils, metadata::compressive_encoding::Compression},
};

#[derive(Clone, Copy, Debug)]
enum OffsetCodec {
    Delta,
    Rle,
    Dictionary,
    General(CompressionScheme),
}

fn payload_fixture(
    bits: u8,
    large: bool,
    nested: bool,
    available: usize,
    kind: OffsetCodec,
) -> Fixture {
    if !matches!(kind, OffsetCodec::Delta) {
        let mut values = Vec::new();
        for value in [0_u64, 1, 1, 3, 4] {
            values.extend_from_slice(&value.to_le_bytes()[..usize::from(bits / 8)]);
        }
        let compressor: Box<dyn BlockCompressor> = if matches!(kind, OffsetCodec::Rle) {
            Box::new(
                BlockRleEncoder::try_new(
                    u64::from(bits),
                    Box::<ValueEncoder>::default(),
                    Box::<ValueEncoder>::default(),
                )
                .unwrap(),
            )
        } else if let OffsetCodec::General(scheme) = kind {
            Box::new(GeneralBlockCompressor::new(
                Box::<ValueEncoder>::default(),
                CompressionConfig::new(scheme, None),
            ))
        } else {
            Box::new(
                BlockDictionaryEncoder::try_new(
                    u64::from(bits),
                    std::sync::Arc::from([4, 0, 1, 3]),
                    Box::<ValueEncoder>::default(),
                    Box::<ValueEncoder>::default(),
                )
                .unwrap(),
            )
        };
        let (payload, encoding) = compressor
            .compress(DataBlock::FixedWidth(crate::data::FixedWidthDataBlock {
                bits_per_value: u64::from(bits),
                num_values: 5,
                data: LanceBuffer::from(values),
                block_info: crate::data::BlockInfo::new(),
            }))
            .unwrap();
        let Some(Compression::Variable(variable)) =
            EncodingUtils::variable(encoding, None).compression
        else {
            panic!("expected Variable")
        };
        return framed_fixture(
            &[payload.as_ref().unwrap(), b"abcd"],
            BinaryMiniBlockDecompressor::from_variable(&variable).unwrap(),
            large,
            nested,
            available,
        );
    }
    let mut offsets = Vec::new();
    for delta in [1_u64, 0, 2, 1] {
        offsets.extend_from_slice(&delta.to_le_bytes()[..usize::from(bits / 8)]);
    }
    let encoding = EncodingUtils::variable(
        EncodingUtils::delta(
            u64::from(bits),
            0,
            EncodingUtils::flat(u64::from(bits), None),
        ),
        None,
    );
    let Some(Compression::Variable(variable)) = encoding.compression else {
        panic!("expected Variable")
    };
    framed_fixture(
        &[&offsets, b"abcd"],
        BinaryMiniBlockDecompressor::from_variable(&variable).unwrap(),
        large,
        nested,
        available,
    )
}

#[rstest]
#[case::delta(OffsetCodec::Delta)]
#[case::rle(OffsetCodec::Rle)]
#[case::dictionary(OffsetCodec::Dictionary)]
#[case::general_none(OffsetCodec::General(CompressionScheme::None))]
#[cfg_attr(
    feature = "lz4",
    case::general_lz4(OffsetCodec::General(CompressionScheme::Lz4))
)]
fn encoded_offsets_preserve_task_overlap(
    #[case] kind: OffsetCodec,
    #[values(32, 64)] bits: u8,
    #[values(false, true)] large: bool,
    #[values(false, true)] multiple: bool,
    #[values(false, true)] nested: bool,
) {
    let width = usize::from(bits / 8);
    let selected: &[&[u8]] = if multiple || nested {
        &[b"a", b"", b"bc", b"d"]
    } else {
        &[b"", b"bc"]
    };
    let output_bytes =
        (selected.len() + 1) * width + selected.iter().map(|v| v.len()).sum::<usize>();
    let preparation = if matches!(kind, OffsetCodec::General(_)) {
        // Each instruction retains five decoded offsets until complete preflight.
        5 * width * if multiple { 2 } else { 1 }
    } else {
        0
    };
    let required = preparation
        + if multiple {
            11 * width + 8 + if nested { 24 } else { 0 }
        } else {
            output_bytes + if nested { 48 } else { 0 }
        };
    let Fixture {
        mut task,
        budget,
        mut work,
        physical,
    } = payload_fixture(bits, large, nested, required, kind);
    select(
        &mut task,
        if multiple {
            &[(0, 2), (2, 2)]
        } else if nested {
            &[(0, 4)]
        } else {
            &[(1, 2)]
        },
    );
    let mut short = work.try_split(required - 1).unwrap();
    for _ in 0..2 {
        assert!(task.try_decode_in(&mut short).unwrap().is_none());
        assert_eq!(short.len(), required - 1);
        assert_eq!(budget.reserved(), physical + required);
    }
    drop(short);
    drop(work);
    let mut work = budget.try_reserve(required).unwrap();
    let page = task.try_decode_in(&mut work).unwrap().unwrap();
    let ty = if bits == 32 {
        DataType::Binary
    } else {
        DataType::LargeBinary
    };
    let actual = page.data.try_to_arrow_in(&ty, &mut work).unwrap().unwrap();
    let expected = if bits == 32 {
        BinaryArray::from_vec(selected.to_vec()).to_data()
    } else {
        LargeBinaryArray::from_vec(selected.to_vec()).to_data()
    };
    assert_eq!(actual, expected);
    assert_eq!(actual.get_buffer_memory_size(), output_bytes);
    assert_eq!(
        work.len(),
        preparation + if nested && !multiple { 24 } else { 0 }
    );
    drop(task);
    drop(work);
    assert_eq!(
        budget.reserved(),
        output_bytes + if nested { 24 } else { 0 }
    );
    drop(page);
    assert_eq!(budget.reserved(), output_bytes);
    drop(actual);
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn later_encoded_offset_overflow_recycles_structural_scratch() {
    let Fixture {
        mut task,
        budget,
        mut work,
        ..
    } = payload_fixture(32, false, true, 1024, OffsetCodec::Delta);
    select(&mut task, &[(0, 2), (2, 2)]);
    let mut bytes = task.instructions[1].1.data.to_vec();
    // 16-byte header, two padded 12-byte level streams, then four u32 deltas.
    bytes[60..64].fill(255);
    let mut corrupt = work.try_allocate(bytes.len()).unwrap().unwrap();
    corrupt.as_mut().copy_from_slice(&bytes);
    task.instructions[1].1.data = LanceBuffer::from(Buffer::from(corrupt.freeze()));
    let available = work.len();
    let reserved = budget.reserved();
    for _ in 0..2 {
        let error = task.try_decode_in(&mut work).err().unwrap();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error.to_string().contains("prefix sum overflows"),
            "{error}"
        );
        assert_eq!(work.len(), available);
        assert_eq!(budget.reserved(), reserved);
    }
    drop(task);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::noop(CompressionScheme::None)]
#[cfg_attr(feature = "lz4", case::lz4(CompressionScheme::Lz4))]
fn later_general_offset_error_recycles_all_preparation(#[case] scheme: CompressionScheme) {
    let Fixture {
        mut task,
        budget,
        mut work,
        ..
    } = payload_fixture(32, false, true, 1024, OffsetCodec::General(scheme));
    select(&mut task, &[(0, 2), (2, 2)]);
    let config = CompressionConfig::new(scheme, None);
    let compressor = GeneralBufferCompressor::get_compressor(config).unwrap();
    let mut payload = Vec::new();
    compressor
        .compress(bytemuck::cast_slice(&[0_u32, 1, 1, 3, 5]), &mut payload)
        .unwrap();
    let original = &task.instructions[1].1.data;
    // Preserve both structural streams and frame the new offset payload separately.
    let mut frame = original[..48].to_vec();
    frame[6..8].copy_from_slice(&(payload.len() as u16).to_le_bytes());
    frame.extend_from_slice(&payload);
    frame.resize(frame.len().next_multiple_of(MINIBLOCK_ALIGNMENT), 0);
    frame.extend_from_slice(b"abcd");
    frame.resize(frame.len().next_multiple_of(MINIBLOCK_ALIGNMENT), 0);
    let mut input = work.try_allocate(frame.len()).unwrap().unwrap();
    input.as_mut().copy_from_slice(&frame);
    task.instructions[1].1.data = LanceBuffer::from(Buffer::from(input.freeze()));
    let before = work.len();
    let reserved = budget.reserved();
    for _ in 0..2 {
        let error = task.try_decode_in(&mut work).err().unwrap();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("offset"), "{error}");
        assert_eq!(work.len(), before);
        assert_eq!(budget.reserved(), reserved);
    }
    drop((task, work));
    assert_eq!(budget.reserved(), 0);
}
