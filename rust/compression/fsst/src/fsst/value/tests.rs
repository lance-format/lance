// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::fsst::{FSST_MAGIC, FSST_SYMBOL_TABLE_SIZE, compress_into};

const SYMBOLS: [&[u8]; 8] = [
    b"a",
    b"bc",
    b"def",
    b"ghij",
    b"klmno",
    b"pqrstu",
    b"vwxyz01",
    b"23456789",
];

fn decoder(enabled: bool) -> FsstDecompressor {
    let mut table = [0_u8; FSST_SYMBOL_TABLE_SIZE];
    let header = FSST_MAGIC | (u64::from(enabled) << 24) | SYMBOLS.len() as u64;
    table[..8].copy_from_slice(&header.to_ne_bytes());
    for (i, symbol) in SYMBOLS.iter().enumerate() {
        table[8 + 8 * i..8 + 8 * i + symbol.len()].copy_from_slice(symbol);
        table[8 + 8 * SYMBOLS.len() + i] = symbol.len() as u8;
    }
    FsstDecompressor::try_new(&table).unwrap()
}

fn assert_exact(decoder: &FsstDecompressor, input: &[u8], expected: &[u8]) {
    assert_eq!(
        decoder.decompressed_value_len(input).unwrap(),
        expected.len()
    );
    // The destination is intentionally unaligned and has no writable padding.
    let mut output = vec![0xa5; expected.len() + 2];
    decoder
        .decompress_value_into(input, &mut output[1..1 + expected.len()])
        .unwrap();
    assert_eq!(&output[1..1 + expected.len()], expected);
    assert_eq!(output[0], 0xa5);
    assert_eq!(output[expected.len() + 1], 0xa5);
}

#[test]
fn exact_value_handles_all_symbol_lengths_literals_and_empty_values() {
    let decoder = decoder(true);
    assert_exact(&decoder, &[], &[]);
    for (code, expected) in SYMBOLS.iter().enumerate() {
        assert_exact(&decoder, &[code as u8], expected);
    }
    for literal in 0..=255 {
        assert_exact(&decoder, &[FSST_ESC, literal], &[literal]);
    }
    for count in 0..96 {
        let mut encoded = Vec::new();
        let mut expected = Vec::new();
        for i in 0..count {
            encoded.extend_from_slice(&[(i % SYMBOLS.len()) as u8, FSST_ESC, i as u8]);
            expected.extend_from_slice(SYMBOLS[i % SYMBOLS.len()]);
            expected.push(i as u8);
        }
        assert_exact(&decoder, &encoded, &expected);
    }
}

#[test]
fn exact_value_passthrough_does_not_interpret_codes_or_escapes() {
    let decoder = decoder(false);
    let bytes = (0..=255).collect::<Vec<_>>();
    assert_exact(&decoder, &[], &[]);
    assert_exact(&decoder, &bytes, &bytes);
    assert_exact(&decoder, &[FSST_ESC], &[FSST_ESC]);
}

#[test]
fn exact_value_rejects_wrong_output_sizes_before_writing() {
    let decoder = decoder(true);
    for size in [0, 7, 9, 64] {
        let mut output = vec![0xa5; size];
        let error = decoder
            .decompress_value_into(&[7], &mut output)
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("expected 8"));
        assert_eq!(output, vec![0xa5; size]);
    }
}

#[test]
fn exact_value_rejects_late_corruption_without_partial_output() {
    let decoder = decoder(true);
    for (input, message) in [
        (vec![0, 7, FSST_ESC], "missing a payload"),
        (vec![0, 7, 8], "undefined symbol"),
        (vec![0, 7, 254], "undefined symbol"),
    ] {
        let error = decoder.decompressed_value_len(&input).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains(message));
        let mut output = [0xa5; 32];
        let error = decoder
            .decompress_value_into(&input, &mut output)
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains(message));
        assert_eq!(output, [0xa5; 32]);
    }
    // A following value's first byte cannot complete the preceding escape.
    let adjacent = [0, FSST_ESC, 1];
    let error = decoder.decompressed_value_len(&adjacent[..2]).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    assert!(error.to_string().contains("missing a payload"));
    assert_exact(&decoder, &adjacent[2..], b"bc");
}

fn real_writer<T: arrow_array::OffsetSizeTrait>() {
    for repeats in [1, 2048] {
        let long = b"the rain in spain stays mainly in the plain ".repeat(repeats);
        let values: [&[u8]; 5] = [b"", &long, &[0, 255, 1, 254], b"tail", b""];
        let mut bytes = Vec::new();
        let mut offsets = vec![T::from_usize(0).unwrap()];
        for value in values {
            bytes.extend_from_slice(value);
            offsets.push(T::from_usize(bytes.len()).unwrap());
        }
        let mut table = [0_u8; FSST_SYMBOL_TABLE_SIZE];
        let (encoded, encoded_offsets) = compress_into(&mut table, &bytes, &offsets).unwrap();
        let decoder = FsstDecompressor::try_new(&table).unwrap();
        assert_eq!(decoder.decoder_switch_on, repeats > 1);
        for (pair, expected) in encoded_offsets.windows(2).zip(values) {
            let range = pair[0].as_usize()..pair[1].as_usize();
            assert_exact(&decoder, &encoded[range], expected);
        }
        // Keep compatibility evidence against the ordinary bulk reader as well.
        let (ordinary, ordinary_offsets) =
            decoder.decompress_into(&encoded, &encoded_offsets).unwrap();
        assert_eq!(ordinary, bytes);
        assert_eq!(ordinary_offsets, offsets);
    }
}

#[test]
fn exact_value_matches_real_writer_with_i32_offsets() {
    real_writer::<i32>();
}

#[test]
fn exact_value_matches_real_writer_with_i64_offsets() {
    real_writer::<i64>();
}
