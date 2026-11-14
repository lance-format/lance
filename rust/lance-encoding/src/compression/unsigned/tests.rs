// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::{compression::create_fixed_width_block_decompressor, format::EncodingUtils};
use rstest::rstest;

#[cfg(feature = "bitpacking")]
use crate::{compression::BlockDecompressor, encodings::physical::bitpacking::OutOfLineBitpacking};
#[cfg(feature = "bitpacking")]
use lance_bitpacking::BitPacking;

#[rstest]
fn flat_and_delta_visit_without_alignment_copies(
    #[values(32, 64)] bits: u64,
    #[values(false, true)] delta: bool,
    #[values(0, 1)] displacement: usize,
) {
    let expected = [0_u64, 1, 1, 7, 12];
    let values: &[u64] = if delta { &[1, 0, 6, 5] } else { &expected };
    let mut bytes = vec![255; displacement];
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes()[..(bits / 8) as usize]);
    }
    let input = LanceBuffer::from(bytes);
    let input = input.slice_with_length(displacement, input.len() - displacement);
    let child = EncodingUtils::flat(bits, None);
    let encoding = if delta {
        EncodingUtils::delta(bits, 0, child)
    } else {
        child
    };
    let codec = create_fixed_width_block_decompressor(&encoding, bits).unwrap();
    let mut actual = Vec::new();
    codec
        .visit_unsigned(Some(&input), bits, 5, &mut |v| {
            actual.push(v);
            Ok(())
        })
        .unwrap();
    assert_eq!(actual, expected);
    if !delta {
        for index in (0..expected.len()).rev() {
            assert_eq!(
                codec
                    .unsigned_value_at(Some(&input), bits, 5, index as u64)
                    .unwrap(),
                expected[index]
            );
        }
        assert!(matches!(
            codec.unsigned_value_at(Some(&input), bits, 5, 5),
            Err(Error::InvalidInput { .. })
        ));
    }
    let mut calls = 0;
    let error = codec
        .visit_unsigned(Some(&input), bits, 5, &mut |_| {
            calls += 1;
            Err(Error::invalid_input("consumer stopped"))
        })
        .unwrap_err();
    assert!(error.to_string().contains("consumer stopped"));
    assert_eq!(calls, 1);
}

#[rstest]
#[case::truncated(0)]
#[case::extra_word(1)]
#[case::overflow(2)]
#[case::missing(3)]
#[case::wrong_width(4)]
fn invalid_unsigned_payload_is_rejected(#[case] fault: u8, #[values(32, 64)] bits: u64) {
    let max = if bits == 32 {
        u32::MAX as u64
    } else {
        u64::MAX
    };
    let codec = create_fixed_width_block_decompressor(
        &EncodingUtils::delta(bits, 1, EncodingUtils::flat(bits, None)),
        bits,
    )
    .unwrap();
    let mut bytes = max.to_le_bytes()[..(bits / 8) as usize].to_vec();
    match fault {
        0 => {
            bytes.pop();
        }
        1 => bytes.extend_from_slice(&0_u64.to_le_bytes()[..(bits / 8) as usize]),
        _ => {}
    }
    let input = LanceBuffer::from(bytes);
    let error = codec
        .visit_unsigned(
            if fault == 3 { None } else { Some(&input) },
            if fault == 4 { 16 } else { bits },
            2,
            &mut |_| Ok(()),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::InvalidInput { .. } | Error::CorruptFile { .. }
    ));
}

#[cfg(feature = "bitpacking")]
#[derive(Clone, Copy, Debug)]
enum Tail {
    None,
    Raw,
    Padded,
    Tie,
}

#[cfg(feature = "bitpacking")]
fn packed_fixture(bits: u64, width: u64, tail: Tail) -> (Vec<u8>, Vec<u64>) {
    let remaining = match tail {
        Tail::None => 0,
        Tail::Tie => 1024 * width / bits,
        _ => 19,
    } as usize;
    let count = 2048 + remaining;
    let mask = (1_u64 << width) - 1;
    let mut expected = (0..count)
        .map(|i| (i as u64).wrapping_mul(0x9e3779b97f4a7c15) & mask)
        .collect::<Vec<_>>();
    let raw_tail = matches!(tail, Tail::Raw | Tail::Tie);
    if raw_tail {
        for value in &mut expected[2048..] {
            *value = u64::MAX >> (64 - bits);
        }
    }
    let mut bytes = Vec::new();
    for chunk in expected.chunks(1024) {
        if chunk.len() < 1024 && raw_tail {
            for value in chunk {
                bytes.extend_from_slice(&value.to_le_bytes()[..(bits / 8) as usize]);
            }
        } else {
            macro_rules! pack {
                ($word:ty) => {{
                    let mut values = [0 as $word; 1024];
                    for (dst, src) in values.iter_mut().zip(chunk) {
                        *dst = *src as $word;
                    }
                    let mut packed = vec![0 as $word; 1024 * width as usize / bits as usize];
                    // SAFETY: a full input block, valid width, and exact packed word count.
                    unsafe { <$word>::unchecked_pack(width as usize, &values, &mut packed) };
                    bytes.extend_from_slice(bytemuck::cast_slice(&packed));
                }};
            }
            match bits {
                8 => pack!(u8),
                16 => pack!(u16),
                32 => pack!(u32),
                64 => pack!(u64),
                _ => unreachable!(),
            }
        }
    }
    (bytes, expected)
}

#[cfg(feature = "bitpacking")]
#[rstest]
#[case::complete(Tail::None)]
#[case::raw(Tail::Raw)]
#[case::padded(Tail::Padded)]
#[case::equal_size_raw_precedence(Tail::Tie)]
fn bitpacked_visit_matches_encoder_at_every_width(
    #[case] tail: Tail,
    #[values(8, 16, 32, 64)] bits: u64,
    #[values(0, 1)] displacement: usize,
) {
    for width in 1..bits {
        let (payload, expected) = packed_fixture(bits, width, tail);
        let mut bytes = vec![255; displacement];
        bytes.extend_from_slice(&payload);
        let input = LanceBuffer::from(bytes);
        let input = input.slice_with_length(displacement, input.len() - displacement);
        let codec = OutOfLineBitpacking::new(width, bits);
        let mut index = 0;
        codec
            .visit_unsigned(Some(&input), bits, expected.len() as u64, &mut |v| {
                assert_eq!(
                    v, expected[index],
                    "bits={bits} width={width} index={index}"
                );
                index += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(index, expected.len());
        for index in [0, 31, 127, 128, 511, 1023, 1024, expected.len() - 1] {
            assert_eq!(
                codec
                    .unsigned_value_at(Some(&input), bits, expected.len() as u64, index as u64)
                    .unwrap(),
                expected[index]
            );
        }
        assert!(matches!(
            codec.unsigned_value_at(
                Some(&input),
                bits,
                expected.len() as u64,
                expected.len() as u64
            ),
            Err(Error::InvalidInput { .. })
        ));
    }
}

#[cfg(feature = "bitpacking")]
#[rstest]
fn malformed_bitpacked_framing_precedes_visits(
    #[values(8, 16, 32, 64)] bits: u64,
    #[values(-1, 1)] change: i8,
) {
    let (mut payload, expected) = packed_fixture(bits, 3, Tail::Padded);
    if change < 0 {
        payload.pop();
    } else {
        payload.push(0);
    }
    let input = LanceBuffer::from(payload);
    let mut calls = 0;
    let error = OutOfLineBitpacking::new(3, bits)
        .visit_unsigned(Some(&input), bits, expected.len() as u64, &mut |_| {
            calls += 1;
            Ok(())
        })
        .unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert_eq!(calls, 0);
    assert!(matches!(
        OutOfLineBitpacking::new(3, bits).unsigned_value_at(
            Some(&input),
            bits,
            expected.len() as u64,
            0
        ),
        Err(Error::CorruptFile { .. })
    ));
}

#[rstest]
fn unsigned_metadata_lookup_checks_full_sequence(
    #[values(32, 64)] bits: u64,
    #[values(false, true)] is_constant: bool,
) {
    let encoding = if is_constant {
        EncodingUtils::constant(Some(bytes::Bytes::copy_from_slice(
            &7_u64.to_le_bytes()[..(bits / 8) as usize],
        )))
    } else {
        EncodingUtils::range(bits, 7, 3)
    };
    let codec = create_fixed_width_block_decompressor(&encoding, bits).unwrap();
    assert_eq!(
        codec.unsigned_value_at(None, bits, 5, 4).unwrap(),
        if is_constant { 7 } else { 19 }
    );
    assert!(matches!(
        codec.unsigned_value_at(None, bits, 5, 5),
        Err(Error::InvalidInput { .. })
    ));
    assert!(matches!(
        codec.unsigned_value_at(None, bits, 0, 0),
        Err(Error::InvalidInput { .. })
    ));
    let max = if bits == 32 {
        u32::MAX as u64
    } else {
        u64::MAX
    };
    let codec =
        create_fixed_width_block_decompressor(&EncodingUtils::range(bits, max - 1, 2), bits)
            .unwrap();
    let error = codec.unsigned_value_at(None, bits, 2, 0).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("range") || error.to_string().contains("Range"));
}

#[rstest]
#[case::truncated(0)]
#[case::extra(1)]
#[case::missing(2)]
#[case::wrong_width(3)]
#[case::count_overflow(4)]
fn unsigned_lookup_rejects_invalid_flat_framing(#[case] fault: u8, #[values(32, 64)] bits: u64) {
    let codec =
        create_fixed_width_block_decompressor(&EncodingUtils::flat(bits, None), bits).unwrap();
    let mut bytes = vec![0; (bits / 8 * 2) as usize];
    if fault == 0 {
        bytes.pop();
    } else if fault == 1 {
        bytes.push(0);
    }
    let input = LanceBuffer::from(bytes);
    let error = codec
        .unsigned_value_at(
            if fault == 2 { None } else { Some(&input) },
            if fault == 3 { 16 } else { bits },
            if fault == 4 { u64::MAX } else { 2 },
            0,
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::InvalidInput { .. } | Error::CorruptFile { .. }
    ));
    assert!(error.to_string().contains("Flat"));
}
