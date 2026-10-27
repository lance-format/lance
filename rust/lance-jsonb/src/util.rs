// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-FileCopyrightText: Copyright 2023 Datafuse Labs
// Adapted from databendlabs/jsonb at commit fba895c5ebe77ce2539e187f9c652f51cbf195c3.

use super::constants::*;
use super::error::Error;
use super::error::ParseErrorCode;

#[allow(clippy::zero_prefixed_literal)]
static HEX: [u8; 256] = {
    const __: u8 = 255; // not a hex digit
    [
        //   1   2   3   4   5   6   7   8   9   A   B   C   D   E   F
        __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, // 0
        __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, // 1
        __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, // 2
        00, 01, 02, 03, 04, 05, 06, 07, 08, 09, __, __, __, __, __, __, // 3
        __, 10, 11, 12, 13, 14, 15, __, __, __, __, __, __, __, __, __, // 4
        __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, // 5
        __, 10, 11, 12, 13, 14, 15, __, __, __, __, __, __, __, __, __, // 6
        __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, // 7
        __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, // 8
        __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, // 9
        __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, // A
        __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, // B
        __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, // C
        __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, // D
        __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, // E
        __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, // F
    ]
};

pub fn parse_string(mut data: &[u8], len: usize, idx: &mut usize) -> Result<String, Error> {
    let mut buf = Vec::with_capacity(len);
    let mut str_buf = String::with_capacity(4);
    while !data.is_empty() {
        *idx += 1;
        let byte = data[0];
        if byte == b'\\' {
            data = &data[1..];
            data = parse_escaped_string(data, idx, &mut str_buf)?;
            buf.extend_from_slice(str_buf.as_bytes());
            str_buf.clear();
        } else {
            buf.push(byte);
            data = &data[1..];
        }
    }
    String::from_utf8(buf).map_err(|_| Error::Syntax(ParseErrorCode::InvalidStringValue, *idx))
}

fn parse_escaped_string<'a>(
    mut data: &'a [u8],
    idx: &mut usize,
    str_buf: &mut String,
) -> Result<&'a [u8], Error> {
    if data.is_empty() {
        return Err(Error::Syntax(
            ParseErrorCode::UnexpectedEndOfHexEscape,
            *idx,
        ));
    }

    let byte = data[0];
    *idx += 1;
    data = &data[1..];
    match byte {
        b'\\' => str_buf.push(BS),
        b'"' => str_buf.push(QU),
        b'/' => str_buf.push(SD),
        b'b' => str_buf.push(BB),
        b'f' => str_buf.push(FF),
        b'n' => str_buf.push(NN),
        b'r' => str_buf.push(RR),
        b't' => str_buf.push(TT),
        b'u' => {
            let mut numbers = [0u8; UNICODE_LEN];
            // Parse the first Unicode escape sequence
            data = parse_unicode_escape(data, idx, &mut numbers)?;
            let hex = decode_hex_escape(&numbers, idx)?;

            let c = match hex {
                0xDC00..=0xDFFF => {
                    // Low surrogate without preceding high surrogate
                    encode_invalid_unicode(&numbers, str_buf);
                    return Ok(data);
                }

                // Non-BMP characters are encoded as a sequence of two hex
                // escapes, representing UTF-16 surrogates.
                n1 @ 0xD800..=0xDBFF => {
                    // High surrogate - check for following low surrogate
                    if data.len() < 2 {
                        encode_invalid_unicode(&numbers, str_buf);
                        return Ok(data);
                    }

                    // Check for \u sequence
                    if data[0] == b'\\' && data[1] == b'u' {
                        *idx += 2;
                        data = &data[2..];
                    } else {
                        encode_invalid_unicode(&numbers, str_buf);
                        return Ok(data);
                    }

                    let mut lower_numbers = [0u8; UNICODE_LEN];
                    // Parse the second Unicode escape sequence
                    data = parse_unicode_escape(data, idx, &mut lower_numbers)?;
                    let n2 = decode_hex_escape(&lower_numbers, idx)?;
                    if !(0xDC00..=0xDFFF).contains(&n2) {
                        encode_invalid_unicode(&numbers, str_buf);
                        encode_invalid_unicode(&lower_numbers, str_buf);
                        return Ok(data);
                    }

                    #[allow(clippy::precedence)]
                    let n = (((n1 - 0xD800) as u32) << 10 | (n2 - 0xDC00) as u32) + 0x1_0000;

                    match char::from_u32(n) {
                        Some(ch) => ch,
                        None => {
                            // Handle invalid Unicode code points gracefully
                            // If we somehow got an invalid code point, preserve the original escape sequence
                            encode_invalid_unicode(&numbers, str_buf);
                            encode_invalid_unicode(&lower_numbers, str_buf);
                            return Ok(data);
                        }
                    }
                }

                // Regular Unicode code points
                n => match char::from_u32(n as u32) {
                    Some(ch) => ch,
                    None => {
                        // Handle invalid code points gracefully
                        encode_invalid_unicode(&numbers, str_buf);
                        return Ok(data);
                    }
                },
            };
            str_buf.push(c);
        }
        other => return Err(Error::Syntax(ParseErrorCode::InvalidEscaped(other), *idx)),
    }
    Ok(data)
}

/// Parse a Unicode escape sequence and return the updated data slice
///
/// This helper function handles both standard \uXXXX and extended \u{XXXX} formats,
/// extracting the hex digits into the provided buffer.
#[inline]
fn parse_unicode_escape<'a>(
    mut data: &'a [u8],
    idx: &mut usize,
    numbers: &mut [u8; UNICODE_LEN],
) -> Result<&'a [u8], Error> {
    if data.len() < UNICODE_LEN {
        return Err(Error::Syntax(
            ParseErrorCode::UnexpectedEndOfHexEscape,
            *idx,
        ));
    }
    // Handle \u{XXXX} format (with braces)
    if data[0] == b'{' {
        if data.len() < UNICODE_LEN + 2 {
            return Err(Error::Syntax(
                ParseErrorCode::UnexpectedEndOfHexEscape,
                *idx,
            ));
        }

        numbers.copy_from_slice(&data[1..UNICODE_LEN + 1]);
        if data[UNICODE_LEN + 1] != b'}' {
            return Err(Error::Syntax(
                ParseErrorCode::UnexpectedEndOfHexEscape,
                *idx,
            ));
        }

        data = &data[UNICODE_LEN + 2..];
        *idx += UNICODE_LEN + 2;
    } else {
        // Standard \uXXXX format
        numbers.copy_from_slice(&data[..UNICODE_LEN]);
        data = &data[UNICODE_LEN..];
        *idx += UNICODE_LEN;
    }

    Ok(data)
}

// https://datatracker.ietf.org/doc/html/rfc8259#section-8.2
// RFC8259 allow invalid Unicode
#[inline]
fn encode_invalid_unicode(numbers: &[u8], str_buf: &mut String) {
    str_buf.push('\\');
    str_buf.push('u');
    for n in numbers {
        str_buf.push((*n).into());
    }
}

#[inline]
fn decode_hex_val(val: u8) -> Option<u16> {
    let n = HEX[val as usize] as u16;
    if n == 255 { None } else { Some(n) }
}

#[inline]
fn decode_hex_escape(numbers: &[u8], idx: &usize) -> Result<u16, Error> {
    let mut n = 0;
    for number in numbers {
        if let Some(hex) = decode_hex_val(*number) {
            n = (n << 4) + hex;
        } else {
            return Err(Error::Syntax(ParseErrorCode::InvalidHex(*number), *idx));
        }
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::fmt::Write;

    #[test]
    fn test_parse_string() {
        // Test cases with expected results
        let test_cases = vec![
            // Basic strings
            ("hello", "hello"),
            ("", ""),
            ("123", "123"),
            // Escaped characters
            (r#"hello\nworld"#, "hello\nworld"),
            (r#"\"\\\b\f\n\r\t"#, "\"\\\u{8}\u{c}\n\r\t"),
            (r#"escaped \"quotes\""#, "escaped \"quotes\""),
            (r#"forward\/slash"#, "forward/slash"),
            // Unicode escapes - Basic
            (r#"\u0041\u0042\u0043"#, "ABC"),
            (r#"Unicode: \u00A9 \u00AE"#, "Unicode: © ®"),
            // Unicode escapes - Braces syntax
            (r#"\u{0041}\u{0042}\u{0043}"#, "ABC"),
            (r#"Unicode: \u{00A9} \u{00AE}"#, "Unicode: © ®"),
            // Unicode escapes - Surrogate pairs
            (r#"\uD834\uDD1E"#, "𝄞"),     // G-clef (musical symbol)
            (r#"\u{D834}\u{DD1E}"#, "𝄞"), // Same with braces
            // Mixed content
            (r#"Mixed: \u0041\n\t\"test\""#, "Mixed: A\n\t\"test\""),
            (r#"CJK: \u4E2D\u6587"#, "CJK: 中文"),
            // Edge cases
            (r#"\u007F"#, "\u{7F}"), // DEL character
            (r#"\u0000"#, "\u{0}"),  // NULL character
        ];

        // Run all test cases
        for (input, expected) in test_cases {
            let input_bytes = input.as_bytes();
            let mut idx = 0;
            let result = parse_string(input_bytes, input_bytes.len(), &mut idx);

            assert!(result.is_ok(), "Failed to parse valid string: {}", input);
            assert_eq!(
                result.unwrap(),
                expected,
                "Incorrect parsing result for: {}",
                input
            );
            assert_eq!(
                idx,
                input_bytes.len(),
                "Index not advanced correctly for: {}",
                input
            );
        }

        // Error cases
        let error_cases = vec![
            // Invalid escape sequence
            r#"\z"#,
            // Incomplete Unicode escape
            r#"\u123"#,
            // Invalid hex in Unicode escape
            r#"\uGHIJ"#,
        ];

        for input in error_cases {
            let input_bytes = if let Ok(s) = std::str::from_utf8(input.as_ref()) {
                s.as_bytes()
            } else {
                input.as_ref()
            };
            let mut idx = 0;
            let result = parse_string(input_bytes, input_bytes.len(), &mut idx);
            assert!(
                result.is_err(),
                "Expected error for invalid input: {:?}",
                input_bytes
            );
        }
    }

    proptest! {
        /// Property-based test for parse_string using randomly generated strings
        ///
        /// This test generates:
        /// 1. Regular ASCII strings
        /// 2. Strings with escaped characters
        /// 3. Strings with Unicode characters including CJK
        /// 4. Strings with Unicode escape sequences
        #[test]
        fn proptest_parse_string(
            // Generate regular ASCII strings
            s1 in r#"[a-zA-Z0-9 ]{0,50}"#,
            // Generate strings with standard JSON escape sequences
            s2 in r#"(\\[\"\\\/bfnrt]){0,10}"#,
            // Generate Unicode characters including CJK
            s3 in prop::collection::vec(prop::char::range('\u{0020}', '\u{FFFF}'), 0..20).prop_map(|chars| chars.into_iter().collect::<String>()),
            // Generate valid Unicode escape sequences
            s4 in prop::collection::vec(0u16..0xD800, 0..5).prop_map(|nums| {
                nums.into_iter()
                    .fold(String::new(), |mut output, b| {
                        let _ = write!(output, r#"\u{:04X}"#, b);
                        output
                    })
            }),
            // Generate valid Unicode surrogate pairs
            s5 in prop::collection::vec((0xD800u16..0xDC00, 0xDC00u16..0xE000), 0..3).prop_map(|pairs| {
                pairs.into_iter()
                    .fold(String::new(), |mut output, (high, low)| {
                        let _ = write!(output, r#"\u{:04X}\u{:04X}"#, high, low);
                        output
                    })
            }),
        ) {
            // Combine all generated strings
            let combined = format!("{}{}{}{}{}", s1, s2, s3, s4, s5);

            // Skip empty strings as they're already tested in the unit tests
            prop_assume!(!combined.is_empty());

            // Convert to a properly escaped JSON string
            let json_string = serde_json::to_string(&combined).unwrap();
            // Remove the surrounding quotes that serde_json adds
            let json_content = &json_string[1..json_string.len()-1];

            // Parse the string using our function
            let input_bytes = json_content.as_bytes();
            let mut idx = 0;
            let result = parse_string(input_bytes, input_bytes.len(), &mut idx);

            // Verify parsing succeeded and produced the expected result
            prop_assert!(result.is_ok(), "Failed to parse valid string: {}", json_content);
            prop_assert_eq!(result.unwrap(), combined, "Incorrect parsing result");
            prop_assert_eq!(idx, input_bytes.len(), "Index not advanced correctly");
        }

        /// Property-based test for parse_string with focus on edge cases
        ///
        /// This test specifically targets edge cases like:
        /// 1. Strings with many escape sequences
        /// 2. Very long strings
        /// 3. Strings with complex Unicode patterns
        #[test]
        fn proptest_parse_string_edge_cases(
            // Generate strings with many escape sequences
            heavy_escapes in prop::collection::vec(
                prop::sample::select(vec![r#"\\"#, r#"\""#, r#"\n"#, r#"\t"#, r#"\b"#, r#"\f"#, r#"\r"#, r#"\/"#, r#"\u0020"#, r#"\u00A9"#]),
                1..100
            ).prop_map(|v| v.join("")),

            // Generate long regular strings
            long_string in r#"[a-zA-Z0-9 ]{100,500}"#,

            // Generate strings with repeating Unicode patterns
            unicode_pattern in prop::collection::vec(
                prop::sample::select(vec![
                    // ASCII
                    "ABC",
                    // Emoji
                    "😀😁😂",
                    // CJK
                    "中文日本語",
                    // Mixed scripts
                    "Latin Кириллица العربية",
                    // Unicode escapes
                    r#"\u0041\u0042\u0043"#,
                    // Surrogate pairs
                    r#"\uD834\uDD1E\uD834\uDD1F"#
                ]),
                1..10
            ).prop_map(|v| v.join("")),
        ) {
            // Test each generated string separately
            for test_str in [heavy_escapes, long_string, unicode_pattern] {
                // Skip empty strings
                if test_str.is_empty() {
                    continue;
                }

                // Convert to a properly escaped JSON string
                let json_string = serde_json::to_string(&test_str).unwrap();
                // Remove the surrounding quotes
                let json_content = &json_string[1..json_string.len()-1];

                // Parse the string
                let input_bytes = json_content.as_bytes();
                let mut idx = 0;
                let result = parse_string(input_bytes, input_bytes.len(), &mut idx);

                // Verify parsing
                prop_assert!(result.is_ok(), "Failed to parse valid string: {}", json_content);
                prop_assert_eq!(result.unwrap(), test_str, "Incorrect parsing result");
                prop_assert_eq!(idx, input_bytes.len(), "Index not advanced correctly");
            }
        }
    }
}
