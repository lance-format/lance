// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-FileCopyrightText: Copyright 2023 Datafuse Labs
// Adapted from databendlabs/jsonb at commit fba895c5ebe77ce2539e187f9c652f51cbf195c3.

use std::borrow::Cow;

use super::constants::*;
use super::error::Error;
use super::error::ParseErrorCode;
use super::error::Result;
use super::number::Number;
use super::util::parse_string;
use super::value::Object;
use super::value::Value;

use crate::constants::MAX_DECIMAL128_PRECISION;
use crate::constants::MAX_DECIMAL256_PRECISION;

use crate::constants::INT64_MAX;
use crate::constants::INT64_MIN;
use crate::constants::UINT64_MAX;
use crate::constants::UINT64_MIN;

// JSON literal constants
const NULL_LOWERCASE: [u8; 4] = *b"null";
const NULL_UPPERCASE: [u8; 4] = *b"NULL";
const TRUE_LOWERCASE: [u8; 4] = *b"true";
const TRUE_UPPERCASE: [u8; 4] = *b"TRUE";
const FALSE_LOWERCASE: [u8; 5] = *b"false";
const FALSE_UPPERCASE: [u8; 5] = *b"FALSE";

const NAN_LOWERCASE: [u8; 3] = *b"nan";
const NAN_UPPERCASE: [u8; 3] = *b"NAN";
const INFINITY_LOWERCASE: [u8; 8] = *b"infinity";
const INFINITY_UPPERCASE: [u8; 8] = *b"INFINITY";

/// Intermediate Abstract Syntax Tree representation of JSON values optimized for parsing performance.
///
/// `JsonAst` serves as an efficient intermediate representation during the JSON parsing process,
/// providing several performance optimizations:
///
/// 1. **Zero-copy string handling**: Uses `Cow<'a, str>` to avoid unnecessary string allocations
///    when the input can be directly borrowed, for both string values and object keys.
///
/// 2. **Efficient object representation**: Stores object entries as a vector of tuples with
///    keys using the same zero-copy `Cow<'a, str>` approach, reducing memory overhead and
///    avoiding unnecessary allocations during parsing.
///
/// 3. **Immediate validation**: Performs object key uniqueness validation and sorting during
///    the parsing process, eliminating the need for a separate validation pass and reducing
///    overall processing time.
///
/// 4. **Lifetime preservation**: Maintains the lifetime of the original input buffer throughout
///    the parsing process, minimizing unnecessary copying of data.
///
/// 5. **Direct conversion path**: Provides an optimized conversion to the final `OwnedJsonb` type
///    through the `into_owned_jsonb` method.
///
/// This approach separates the parsing concerns from the final representation concerns,
/// allowing each to be optimized independently.
#[derive(Clone, PartialEq, Default, Eq, Debug)]
pub enum JsonAst<'a> {
    #[default]
    Null,
    Bool(bool),
    String(Cow<'a, str>),
    Number(Number),
    Array(Vec<Self>),
    Object(Vec<(Cow<'a, str>, Self, usize)>),
}

impl<'a> JsonAst<'a> {
    /// Converts the intermediate `JsonAst` representation to the final `Value` type.
    fn into_value(self) -> Result<Value<'a>> {
        let value = match self {
            JsonAst::Null => Value::Null,
            JsonAst::Bool(v) => Value::Bool(v),
            JsonAst::String(v) => Value::String(v),
            JsonAst::Number(v) => Value::Number(v),
            JsonAst::Array(vals) => {
                let mut values = Vec::with_capacity(vals.len());
                for val in vals.into_iter() {
                    let value = val.into_value()?;
                    values.push(value);
                }
                Value::Array(values)
            }
            JsonAst::Object(kvs) => {
                let mut object = Object::new();
                for (key, val, _) in kvs.into_iter() {
                    let key_str = key.to_string();
                    let value = val.into_value()?;
                    object.insert(key_str, value);
                }
                Value::Object(object)
            }
        };
        Ok(value)
    }
}

/// Parse JSON text to JSONB Value with extended mode.
/// The parser will follow extended JSON syntax rules like leading plus signs,
/// multiple leading zeros, decimal points without digits, and empty array elements.
/// Numeric values are preferentially parsed as decimal values to ensure that precision is not lost.
///
/// Inspired by `https://github.com/jorgecarleitao/json-deserializer`
/// Thanks Jorge Leitao.
pub fn parse_value(buf: &[u8]) -> Result<Value<'_>> {
    let mut parser = Parser::new(buf);
    let json_ast = parser.parse()?;
    json_ast.into_value()
}

/// JSON parser with optimized parsing strategies.
///
/// This parser implements both standard JSON parsing and an extended syntax with additional features.
/// It uses a single-pass approach for better performance and provides detailed error reporting.
struct Parser<'a> {
    /// Input buffer containing the JSON text to parse
    buf: &'a [u8],
    /// Current position in the buffer
    idx: usize,
    /// Function pointer for parsing json value based on the mode
    parse_value_fn: fn(&mut Self) -> Result<JsonAst<'a>>,
    /// Function pointer for parsing array value based on the mode
    parse_array_value_fn: fn(&mut Self) -> Result<JsonAst<'a>>,
    /// Function pointer for parsing object_key based on the mode
    parse_object_key_fn: fn(&mut Self) -> Result<Cow<'a, str>>,
}

impl<'a> Parser<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            idx: 0,
            parse_value_fn: Self::parse_json_value,
            parse_array_value_fn: Self::parse_array_value,
            parse_object_key_fn: Self::parse_object_key,
        }
    }

    /// Parse a complete JSON document from the input buffer.
    fn parse(&mut self) -> Result<JsonAst<'a>> {
        let value = (self.parse_value_fn)(self)?;

        self.skip_unused();
        if self.idx < self.buf.len() {
            self.step();
            return Err(self.error(ParseErrorCode::UnexpectedTrailingCharacters));
        }
        Ok(value)
    }

    /// Parse a JSON value in extended mode with more lenient syntax rules
    #[inline]
    fn parse_json_value(&mut self) -> Result<JsonAst<'a>> {
        self.skip_unused();
        // Parse empty string to Null value
        let Ok(c) = self.next() else {
            return Ok(JsonAst::Null);
        };
        match c {
            b'n' | b'N' => self.parse_json_null_or_nan(),
            b't' | b'T' => self.parse_json_true(),
            b'f' | b'F' => self.parse_json_false(),
            b'i' | b'I' => self.parse_json_infinity(),
            b'0'..=b'9' | b'-' | b'+' | b'.' => self.parse_json_number(),
            b'"' | b'\'' => self.parse_json_string(),
            b'[' => self.parse_json_array(),
            b'{' => self.parse_json_object(),
            _ => {
                self.step();
                Err(self.error(ParseErrorCode::ExpectedSomeValue))
            }
        }
    }

    #[inline]
    fn next(&mut self) -> Result<u8> {
        match self.buf.get(self.idx) {
            Some(c) => Ok(*c),
            None => Err(self.error(ParseErrorCode::InvalidEOF)),
        }
    }

    #[inline]
    fn must_is(&mut self, c: &u8) -> Result<()> {
        match self.buf.get(self.idx) {
            Some(v) => {
                self.step();
                if v == c {
                    Ok(())
                } else {
                    Err(self.error(ParseErrorCode::ExpectedSomeIdent))
                }
            }
            None => Err(self.error(ParseErrorCode::InvalidEOF)),
        }
    }

    #[inline]
    fn must_either(&mut self, c1: &u8, c2: &u8) -> Result<u8> {
        match self.buf.get(self.idx) {
            Some(v) => {
                self.step();
                if v == c1 || v == c2 {
                    Ok(*v)
                } else {
                    Err(self.error(ParseErrorCode::ExpectedSomeIdent))
                }
            }
            None => Err(self.error(ParseErrorCode::InvalidEOF)),
        }
    }

    #[inline]
    fn check_next(&mut self, c: &u8) -> bool {
        if let Some(v) = self.buf.get(self.idx)
            && v == c
        {
            return true;
        }
        false
    }

    #[inline]
    fn check_next_either(&mut self, c1: &u8, c2: &u8) -> Option<u8> {
        if let Some(v) = self.buf.get(self.idx)
            && (v == c1 || v == c2)
        {
            return Some(*v);
        }
        None
    }

    #[inline]
    fn check_digit(&mut self) -> Option<u8> {
        if let Some(v) = self.buf.get(self.idx)
            && v.is_ascii_digit()
        {
            let digit = v - b'0';
            return Some(digit);
        }
        None
    }

    #[inline]
    fn step_digits(&mut self) -> usize {
        let mut len = 0;
        while let Some(v) = self.buf.get(self.idx) {
            if !v.is_ascii_digit() {
                break;
            }
            len += 1;
            self.step();
        }
        len
    }

    #[inline]
    fn step_hexdigits(&mut self) -> usize {
        let mut len = 0;
        while let Some(v) = self.buf.get(self.idx) {
            if !v.is_ascii_hexdigit() {
                break;
            }
            len += 1;
            self.step();
        }
        len
    }

    #[inline]
    fn step(&mut self) {
        self.idx += 1;
    }

    #[inline]
    fn step_by(&mut self, n: usize) {
        self.idx += n;
    }

    fn error(&self, code: ParseErrorCode) -> Error {
        let pos = self.idx;
        Error::Syntax(code, pos)
    }

    #[inline]
    fn skip_unused(&mut self) {
        while self.idx < self.buf.len() {
            let c = self.buf[self.idx];

            // Fast path: handle common whitespace characters
            if c.is_ascii_whitespace() {
                self.idx += 1;
                continue;
            }

            // Slow path: handle escape sequences
            if c == b'\\' && self.idx + 1 < self.buf.len() {
                let next_c = self.buf[self.idx + 1];

                // Handle simple escapes \n, \r, \t
                let simple_escape = matches!(next_c, b'n' | b'r' | b't');
                if simple_escape {
                    self.idx += 2;
                    continue;
                }

                // Handle \x0C escape
                let hex_escape = self.idx + 3 < self.buf.len()
                    && next_c == b'x'
                    && self.buf[self.idx + 2] == b'0'
                    && self.buf[self.idx + 3] == b'C';
                if hex_escape {
                    self.idx += 4;
                    continue;
                }
            }

            break;
        }
    }

    /// Parse a JSON null or NaN literal in extended mode with case-insensitivity
    #[inline]
    fn parse_json_null_or_nan(&mut self) -> Result<JsonAst<'a>> {
        let idx = self.idx;
        if let Ok(null) = self.parse_json_null() {
            Ok(null)
        } else {
            // fallback idx to check if it is NaN
            self.idx = idx;
            self.parse_json_nan()
        }
    }

    /// Parse a JSON null literal in extended mode with case-insensitivity
    /// Accepts any case variation of "null" (e.g., "Null", "NULL", "nUlL").
    #[inline]
    fn parse_json_null(&mut self) -> Result<JsonAst<'a>> {
        for (v1, v2) in NULL_LOWERCASE.iter().zip(NULL_UPPERCASE.iter()) {
            self.must_either(v1, v2)?;
        }
        Ok(JsonAst::Null)
    }

    /// Parse a JSON true literal in extended mode with case-insensitivity
    /// Accepts any case variation of "true" (e.g., "True", "TRUE", "tRuE").
    #[inline]
    fn parse_json_true(&mut self) -> Result<JsonAst<'a>> {
        for (v1, v2) in TRUE_LOWERCASE.iter().zip(TRUE_UPPERCASE.iter()) {
            self.must_either(v1, v2)?;
        }
        Ok(JsonAst::Bool(true))
    }

    /// Parse a JSON false literal in extended mode with case-insensitivity
    /// Accepts any case variation of "false" (e.g., "False", "FALSE", "fAlSe").
    #[inline]
    fn parse_json_false(&mut self) -> Result<JsonAst<'a>> {
        for (v1, v2) in FALSE_LOWERCASE.iter().zip(FALSE_UPPERCASE.iter()) {
            self.must_either(v1, v2)?;
        }
        Ok(JsonAst::Bool(false))
    }

    /// Parse a JSON infinity literal in extended mode with case-insensitivity
    /// Accepts any case variation of "infinity" (e.g., "Infinity", "INFINITY").
    #[inline]
    fn parse_json_infinity(&mut self) -> Result<JsonAst<'a>> {
        for (v1, v2) in INFINITY_LOWERCASE.iter().zip(INFINITY_UPPERCASE.iter()) {
            self.must_either(v1, v2)?;
        }
        Ok(JsonAst::Number(Number::Float64(f64::INFINITY)))
    }

    /// Parse a JSON NaN literal in extended mode with case-insensitivity
    /// Accepts any case variation of "NaN" (e.g., "nan", "NAN").
    #[inline]
    fn parse_json_nan(&mut self) -> Result<JsonAst<'a>> {
        for (v1, v2) in NAN_LOWERCASE.iter().zip(NAN_UPPERCASE.iter()) {
            self.must_either(v1, v2)?;
        }
        Ok(JsonAst::Number(Number::Float64(f64::NAN)))
    }

    /// Parse extended JSON numbers (supporting non-standard syntax)
    ///
    /// This function implements a high-performance JSON number parsing algorithm with extended syntax:
    /// 1. Support for leading plus sign (e.g., `+123`)
    /// 2. Support for multiple leading zeros (e.g., `000123`)
    /// 3. Support for decimal point without digits on either side (e.g., `.123` or `123.`)
    /// 4. Support for special values like `NaN`, `Infinity`, and `-Infinity` (case-insensitive)
    /// 5. Support for hexadecimal notation with optional fractional part (e.g., `0xFF`, `0x1A.B`)
    ///
    /// Zero-allocation parsing strategy:
    /// 1. Uses direct digit accumulation without intermediate string conversions
    /// 2. For standard numeric types (Int64/UInt64), directly builds the value during parsing
    /// 3. For hexadecimal numbers, uses specialized parsing with support for fractional parts
    /// 4. Falls back to Float64 parsing for fractions, exponents, and integers outside the i64/u64 range
    ///
    /// This implementation prioritizes performance through:
    /// - Single-pass approach with minimal branching
    /// - Avoiding heap allocations and string conversions
    /// - Optimized handling of common number formats
    /// - Specialized handling for extended syntax elements
    fn parse_json_number(&mut self) -> Result<JsonAst<'a>> {
        // Store the starting position for potential fallback parsing
        let start_idx = self.idx;

        let mut negative = false;
        let mut leading_zeros = 0;

        // Handle sign prefix (+ or -), extending JSON to support leading plus sign
        let c = self.next()?;
        if c == b'-' {
            negative = true;
            self.step();
        } else if c == b'+' {
            // Extended syntax: Support for leading plus sign
            self.step();
        }

        // Extended syntax: Support for multiple leading zeros (e.g., 000123)
        loop {
            if self.check_next(&b'0') {
                leading_zeros += 1;
                self.step();
            } else {
                break;
            }
        }

        // Initialize parsing state
        let mut hi_value = 0_i128; // Stores high digits (for large values)
        let mut lo_value = 0_i128; // Stores low digits (for very large values)
        let mut scale = 0_u32; // Tracks decimal places
        let mut precision = 0; // Count of significant digits
        let mut has_fraction = false; // Whether the number has an fraction part
        let mut has_exponent = false; // Whether the number has an exponent part

        // Parse digits, supporting up to MAX_DECIMAL256_PRECISION digits
        while precision < MAX_DECIMAL256_PRECISION {
            if let Some(digit) = self.check_digit() {
                // Parse digit and accumulate value
                // Store in hi_value or lo_value based on precision
                if precision < MAX_DECIMAL128_PRECISION {
                    hi_value = unsafe { hi_value.unchecked_mul(10_i128) };
                    hi_value = unsafe { hi_value.unchecked_add(digit as i128) };
                } else {
                    lo_value = unsafe { lo_value.unchecked_mul(10_i128) };
                    lo_value = unsafe { lo_value.unchecked_add(digit as i128) };
                }
                self.step();
            } else if self.check_next(&b'.') {
                // Handle decimal point - can only appear once
                if has_fraction {
                    return Err(self.error(ParseErrorCode::InvalidNumberValue));
                }
                has_fraction = true;
                self.step();
                // Continue to next iteration without incrementing precision
                continue;
            } else {
                // Not a digit or decimal point, exit the parsing loop
                break;
            }
            precision += 1;
            // Track scale (number of digits after decimal point)
            if has_fraction {
                scale += 1;
            }
        }

        // Handle numbers that exceed MAX_DECIMAL256_PRECISION
        if precision == MAX_DECIMAL256_PRECISION {
            // If we haven't seen a decimal point yet, continue parsing integer part
            if !has_fraction {
                let len = self.step_digits();
                precision += len;
                if self.check_next(&b'.') {
                    has_fraction = true;
                    self.step();
                }
            }
            // Parse fractional part if decimal point exists
            if has_fraction {
                let len = self.step_digits();
                precision += len;
                scale += len as u32;
            }
        }

        if leading_zeros == 0 && precision == 0 {
            // Handle special values
            if !has_fraction && let Ok(c) = self.next() {
                match c {
                    b'i' | b'I' => {
                        let val = self.parse_json_infinity()?;
                        if negative {
                            return Ok(JsonAst::Number(Number::Float64(f64::NEG_INFINITY)));
                        } else {
                            return Ok(val);
                        }
                    }
                    b'n' | b'N' => {
                        let val = self.parse_json_nan()?;
                        if negative {
                            // `-Nan` is not allowed
                            return Err(self.error(ParseErrorCode::InvalidNumberValue));
                        } else {
                            return Ok(val);
                        }
                    }
                    _ => {}
                }
            }
            return Err(self.error(ParseErrorCode::InvalidNumberValue));
        } else if leading_zeros == 1 && precision == 0 && !has_fraction {
            // Handle hexadecimal number (0x...)
            if self.check_next_either(&b'x', &b'X').is_some() {
                self.step();

                // Mark the start position of hex digits
                let hex_start = self.idx;
                let int_len = self.step_hexdigits();
                if int_len == 0 {
                    return Err(self.error(ParseErrorCode::InvalidNumberValue));
                }

                // Check if we have a fractional part
                if self.check_next(&b'.') {
                    // Skip the decimal point
                    self.step();

                    // Mark the start of fractional digits
                    let frac_start = self.idx;
                    let frac_len = self.step_hexdigits();
                    if frac_len == 0 {
                        return Err(self.error(ParseErrorCode::InvalidNumberValue));
                    }

                    let int_str = std::str::from_utf8(&self.buf[hex_start..hex_start + int_len])
                        .map_err(|_| self.error(ParseErrorCode::InvalidNumberValue))?;
                    let frac_str =
                        std::str::from_utf8(&self.buf[frac_start..frac_start + frac_len])
                            .map_err(|_| self.error(ParseErrorCode::InvalidNumberValue))?;

                    // Parse integer part
                    let int_val = u128::from_str_radix(int_str, 16)
                        .map_err(|_| self.error(ParseErrorCode::InvalidNumberValue))?;

                    // Parse fractional part and calculate its value
                    let frac_val = u128::from_str_radix(frac_str, 16)
                        .map_err(|_| self.error(ParseErrorCode::InvalidNumberValue))?;
                    let frac_divisor = 16.0_f64.powi(frac_len as i32);

                    // Combine integer and fractional parts
                    let mut final_val = int_val as f64 + (frac_val as f64 / frac_divisor);
                    if negative {
                        final_val = -final_val;
                    }
                    return Ok(JsonAst::Number(Number::Float64(final_val)));
                } else {
                    // Integer-only hex value
                    let int_str = std::str::from_utf8(&self.buf[hex_start..self.idx])
                        .map_err(|_| self.error(ParseErrorCode::InvalidNumberValue))?;

                    // Parse the hex value
                    let value = u128::from_str_radix(int_str, 16)
                        .map_err(|_| self.error(ParseErrorCode::InvalidNumberValue))?;

                    // Convert to appropriate number type based on size
                    if negative {
                        // Handle negative values
                        if value <= (i64::MAX as u128 + 1) {
                            let i_val = -(value as i64);
                            return Ok(JsonAst::Number(Number::Int64(i_val)));
                        }
                        {
                            return Ok(JsonAst::Number(Number::Float64(-(value as f64))));
                        }
                    } else {
                        // Handle positive values
                        if value <= u64::MAX as u128 {
                            return Ok(JsonAst::Number(Number::UInt64(value as u64)));
                        }
                        {
                            return Ok(JsonAst::Number(Number::Float64(value as f64)));
                        }
                    }
                }
            }
        }

        // Handle exponent notation (e.g., 1e10, 1.5E-7)
        if self.check_next_either(&b'E', &b'e').is_some() {
            has_exponent = true;
            self.step();
            // Handle exponent sign
            if self.check_next_either(&b'+', &b'-').is_some() {
                self.step();
            }
            // Parse exponent digits
            let len = self.step_digits();
            if len == 0 {
                return Err(self.error(ParseErrorCode::InvalidNumberValue));
            }
        }

        // If no exponent and precision is within limits, try to return the most appropriate numeric type
        if !has_exponent && precision <= MAX_DECIMAL128_PRECISION {
            let value = if negative { -hi_value } else { hi_value };

            // Try to fit the value into the most appropriate numeric type
            if scale == 0 && (UINT64_MIN..=UINT64_MAX).contains(&value) {
                return Ok(JsonAst::Number(Number::UInt64(
                    u64::try_from(value).unwrap(),
                )));
            } else if scale == 0 && (INT64_MIN..=INT64_MAX).contains(&value) {
                return Ok(JsonAst::Number(Number::Int64(
                    i64::try_from(value).unwrap(),
                )));
            }
        }

        // Second parsing strategy: Try to parse as i256 for very large numbers

        // Final fallback strategy: Parse as Float64 using fast_float2 library
        // This handles scientific notation and very large/small numbers
        let s = unsafe { std::str::from_utf8_unchecked(&self.buf[start_idx..self.idx]) };
        match fast_float2::parse(s) {
            Ok(v) => Ok(JsonAst::Number(Number::Float64(v))),
            Err(_) => Err(self.error(ParseErrorCode::InvalidNumberValue)),
        }
    }

    /// Parse a JSON string with extended syntax support
    ///
    /// Extended syntax allows both double quotes (") and single quotes (')
    /// as string delimiters, which is not allowed in standard JSON.
    #[inline]
    fn parse_json_string(&mut self) -> Result<JsonAst<'a>> {
        let end_quote = self.must_either(&b'"', &b'\'')?;
        let val = self.parse_quoted_string(end_quote)?;
        Ok(JsonAst::String(val))
    }

    /// Parse a quoted string with support for escape sequences
    ///
    /// Handles both standard and extended Unicode escape sequences:
    /// - Standard: \uXXXX (4 hex digits)
    /// - Extended: \u{XXXX} (variable number of hex digits in braces)
    ///
    /// Uses a two-pass approach for efficiency:
    /// 1. First pass: Find string boundaries and count escapes
    /// 2. Second pass: Process escapes only when necessary
    fn parse_quoted_string(&mut self, end_quote: u8) -> Result<Cow<'a, str>> {
        // Mark the starting position (after the opening quote)
        let start_idx = self.idx;
        let mut escapes = 0;

        // First pass: Scan the string to find the end and count escape sequences
        loop {
            let c = self.next()?;
            match c {
                b'\\' => {
                    // Handle escape sequence
                    self.step();
                    escapes += 1;
                    let next_c = self.next()?;
                    if next_c == b'u' {
                        // Handle Unicode escape sequence
                        self.step();
                        let next_c = self.next()?;
                        if next_c == b'{' {
                            // Extended Unicode format: \u{XXXX}
                            self.step_by(UNICODE_LEN + 2);
                        } else {
                            // Standard Unicode format: \uXXXX
                            self.step_by(UNICODE_LEN);
                        }
                    } else {
                        // Simple escape sequence like \n, \t, etc.
                        self.step();
                    }
                }
                _ => {
                    self.step();
                    if c == end_quote {
                        break;
                    }
                }
            }
        }

        // Get the string data (excluding quotes)
        let data = &self.buf[start_idx..self.idx - 1];

        // Second pass: Process the string based on whether it contains escape sequences
        let val = if escapes > 0 {
            // String contains escape sequences, need to process them
            let len = self.idx - 1 - start_idx - escapes;
            let mut idx = start_idx + 1;
            let s = parse_string(data, len, &mut idx)?;
            Cow::Owned(s)
        } else {
            // Fast path: No escape sequences, can use the string as-is
            std::str::from_utf8(data)
                .map(Cow::Borrowed)
                .map_err(|_| self.error(ParseErrorCode::InvalidStringValue))?
        };
        Ok(val)
    }

    /// Parse an unquoted string literal for object keys
    ///
    /// Extended syntax feature that allows object keys without quotes.
    ///
    /// Restrictions:
    /// - Only letters, numbers, underscore, dollar and UTF-8 multi-byte characters are allowed
    /// - First character cannot be a number
    /// - Must contain at least one character
    fn parse_unquoted_string(&mut self) -> Result<Cow<'a, str>> {
        let start_idx = self.idx;

        let c = self.next()?;
        if c.is_ascii_digit() {
            self.step();
            return Err(self.error(ParseErrorCode::ObjectKeyInvalidNumber));
        }

        loop {
            let c = self.next()?;
            if c.is_ascii_alphanumeric() || matches!(c, b'_' | b'$') {
                self.step();
            } else if c >= 0x80 {
                // Handle UTF-8 multi-byte characters (including Chinese)
                // UTF-8 continuation bytes start with binary 10xxxxxx (0x80-0xBF)
                // Determine how many continuation bytes to expect based on the first byte
                let continuation_bytes = if c >= 0xF0 {
                    4 // 4-byte sequence (U+10000 to U+10FFFF)
                } else if c >= 0xE0 {
                    3 // 3-byte sequence (U+0800 to U+FFFF) - includes most Chinese characters
                } else if c >= 0xC0 {
                    2 // 2-byte sequence (U+0080 to U+07FF)
                } else {
                    // Invalid UTF-8 start byte
                    return Err(self.error(ParseErrorCode::ObjectKeyInvalidCharacter));
                };

                // Consume the expected continuation bytes
                self.step_by(continuation_bytes);
            } else {
                break;
            }
        }
        if self.idx == start_idx {
            return Err(self.error(ParseErrorCode::ObjectKeyInvalidCharacter));
        }

        // Get the string data
        let data = &self.buf[start_idx..self.idx];
        let val = std::str::from_utf8(data)
            .map(Cow::Borrowed)
            .map_err(|_| self.error(ParseErrorCode::InvalidStringValue))?;
        Ok(val)
    }

    /// Parse an array value with support for empty elements
    ///
    /// Extended syntax feature that treats empty elements as null:
    /// - [1,,3] is parsed as [1,null,3]
    /// - [1,2,] is parsed as [1,2,null]
    ///
    /// This is not allowed in standard JSON but supported in extended mode.
    #[inline]
    fn parse_array_value(&mut self) -> Result<JsonAst<'a>> {
        if self.check_next_either(&b',', &b']').is_some() {
            Ok(JsonAst::Null)
        } else {
            self.parse_json_value()
        }
    }

    /// Parse a JSON array with support for extended syntax
    ///
    /// - Parses arrays enclosed in square brackets [...]
    /// - Handles comma-separated values
    /// - Validates proper syntax for separators and closing brackets
    /// - Allows empty elements, which are treated as null
    ///
    /// Examples of valid arrays:
    /// - [1,2,3]     (standard JSON)
    /// - [1,,3]      (empty element treated as null)
    /// - [1,2,]      (trailing comma treated as null element)
    fn parse_json_array(&mut self) -> Result<JsonAst<'a>> {
        // Ensure the array starts with an opening bracket
        self.must_is(&b'[')?;

        let mut first = true;
        let mut values = Vec::with_capacity(8);

        // Parse array elements until closing bracket is found
        loop {
            self.skip_unused();
            let c = self.next()?;

            // Check for end of array
            if c == b']' {
                self.step();
                break;
            }

            // Handle comma separator between elements (not for the first element)
            if !first {
                if c != b',' {
                    return Err(self.error(ParseErrorCode::ExpectedArrayCommaOrEnd));
                }
                self.step();
            }
            first = false;

            self.skip_unused();
            // Parse a regular array element
            let value = (self.parse_array_value_fn)(self)?;
            values.push(value);
        }
        Ok(JsonAst::Array(values))
    }

    /// Parse an object key with extended syntax support
    ///
    /// Extended syntax allows:
    /// 1. Double-quoted strings (")
    /// 2. Single-quoted strings (')
    /// 3. Unquoted identifiers (letters, numbers, underscore, dollar and UTF-8 characters)
    ///    with the restriction that they cannot start with a number
    #[inline]
    fn parse_object_key(&mut self) -> Result<Cow<'a, str>> {
        if let Some(end_quote) = self.check_next_either(&b'"', &b'\'') {
            self.step();
            self.parse_quoted_string(end_quote)
        } else {
            self.parse_unquoted_string()
        }
    }

    /// Parse a JSON object with support for both standard and extended syntax
    ///
    /// - Parses objects enclosed in curly braces {...}
    /// - Handles key-value pairs separated by colons
    /// - Validates proper syntax for separators and closing braces
    /// - Detects and reports duplicate keys
    /// - Accepts quoted (double/single) and unquoted keys, and extended syntax for values
    ///
    /// Examples of valid objects:
    /// - {"key": "value"}      (standard JSON)
    /// - {'key': 'value'}      (single quotes)
    /// - {key: "value"}        (unquoted key)
    /// - {_user123: 'value'}   (unquoted key with underscore)
    fn parse_json_object(&mut self) -> Result<JsonAst<'a>> {
        // Ensure the object starts with an opening brace
        self.must_is(&b'{')?;

        let mut first = true;
        let mut obj = Vec::with_capacity(16);

        // Parse key-value pairs until closing brace is found
        loop {
            self.skip_unused();
            let c = self.next()?;

            // Check for end of object
            if c == b'}' {
                self.step();
                break;
            }

            // Handle comma separator between key-value pairs (not for the first pair)
            if !first {
                if c != b',' {
                    return Err(self.error(ParseErrorCode::ExpectedObjectCommaOrEnd));
                }
                self.step();
            }
            first = false;

            self.skip_unused();
            let key_str = (self.parse_object_key_fn)(self)?;
            let pos = self.idx;

            self.skip_unused();

            // Ensure key and value are separated by a colon
            let c = self.next()?;
            if c != b':' {
                return Err(self.error(ParseErrorCode::ExpectedColon));
            }
            self.step();

            // Parse the value
            let value = (self.parse_value_fn)(self)?;

            // Add the key-value pair to the object
            obj.push((key_str, value, pos));
        }

        // Sort the Object fields by key and check for duplicate keys.
        // Returns an error if duplicate keys are found.
        obj.sort_by(|a, b| a.0.cmp(&b.0));
        for i in 1..obj.len() {
            if obj[i - 1].0 == obj[i].0 {
                let key_str = obj[i].0.clone().to_string();
                let pos = obj[i].2;
                let code = ParseErrorCode::ObjectDuplicateKey(key_str);
                return Err(Error::Syntax(code, pos));
            }
        }
        Ok(JsonAst::Object(obj))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OwnedJsonb;
    use proptest::prelude::*;
    use std::collections::BTreeMap;
    use std::fmt::Display;
    use std::fmt::Formatter;

    /// Json5Value represents the extended [JSON5 syntax](https://json5.org/) for testing purposes
    ///
    /// This enum is used to generate test data that conforms to the JSON5 specification,
    /// including features like hexadecimal numbers, single-quoted strings, and unquoted object keys.
    #[derive(Clone, PartialEq, Default, Eq, Debug)]
    pub enum Json5Value {
        #[default]
        Null,
        Bool(bool),
        Number(Number),
        HexNumber(String),
        DoubleQuotedString(String),
        SingleQuotedString(String),
        Array(Vec<Self>),
        DoubleQuotedKeyObject(BTreeMap<String, Self>),
        SingleQuotedKeyObject(BTreeMap<String, Self>),
        UnquotedKeyObject(BTreeMap<String, Self>),
    }

    /// Display implementation for Json5Value that formats values according to JSON5 syntax
    ///
    /// This implementation handles proper escaping of special characters in strings
    /// and ensures the output conforms to the JSON5 specification.
    impl Display for Json5Value {
        fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
            match self {
                Self::Null => write!(f, "null"),
                Self::Bool(v) => {
                    if *v {
                        write!(f, "true")
                    } else {
                        write!(f, "false")
                    }
                }
                Self::Number(v) => write!(f, "{}", v),
                Self::HexNumber(v) => write!(f, "{}", v),
                Self::DoubleQuotedString(v) => {
                    write!(f, "\"")?;
                    for c in v.chars() {
                        match c {
                            '"' => write!(f, "\\\"")?,
                            '\\' => write!(f, "\\\\")?,
                            c => write!(f, "{}", c)?,
                        }
                    }
                    write!(f, "\"")
                }
                Self::SingleQuotedString(v) => {
                    write!(f, "'")?;
                    for c in v.chars() {
                        match c {
                            '\'' => write!(f, "\\\'")?,
                            '\\' => write!(f, "\\\\")?,
                            c => write!(f, "{}", c)?,
                        }
                    }
                    write!(f, "'")
                }
                Self::Array(vs) => {
                    write!(f, "[")?;
                    for (i, v) in vs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ",")?;
                        }
                        write!(f, "{v}")?;
                    }
                    write!(f, "]")
                }
                Self::DoubleQuotedKeyObject(vs) => {
                    write!(f, "{{")?;
                    for (i, (k, v)) in vs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ",")?;
                        }
                        write!(f, "\"")?;
                        for c in k.chars() {
                            match c {
                                '"' => write!(f, "\\\"")?,
                                '\\' => write!(f, "\\\\")?,
                                c => write!(f, "{}", c)?,
                            }
                        }
                        write!(f, "\"")?;
                        write!(f, ":{v}")?;
                    }
                    write!(f, "}}")
                }
                Self::SingleQuotedKeyObject(vs) => {
                    write!(f, "{{")?;
                    for (i, (k, v)) in vs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ",")?;
                        }
                        write!(f, "'")?;
                        for c in k.chars() {
                            match c {
                                '\'' => write!(f, "\\\'")?,
                                '\\' => write!(f, "\\\\")?,
                                c => write!(f, "{}", c)?,
                            }
                        }
                        write!(f, "'")?;
                        write!(f, ":{v}")?;
                    }
                    write!(f, "}}")
                }
                Self::UnquotedKeyObject(vs) => {
                    write!(f, "{{")?;
                    for (i, (k, v)) in vs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ",")?;
                        }
                        write!(f, "{k}:{v}")?;
                    }
                    write!(f, "}}")
                }
            }
        }
    }

    /// Strategy to generate standard strings for testing
    ///
    /// Generates strings containing ASCII characters and CJK Unicode characters
    /// for testing standard JSON string handling
    fn string_strategy() -> impl Strategy<Value = String> {
        let ascii = '!'..='~';
        // CJK Unified Ideographs
        let cjk = '\u{4E00}'..='\u{9FFF}';

        let chars: Vec<char> = ascii.chain(cjk).collect();
        prop::collection::vec(prop::sample::select(chars), 1..50)
            .prop_map(|v| v.into_iter().collect())
    }

    /// Strategy to generate strings suitable for quoted keys and values in JSON5
    ///
    /// Excludes quote characters (single and double) and backslashes to simplify testing
    /// while still providing a diverse set of characters including CJK Unicode
    fn quoted_string_strategy() -> impl Strategy<Value = String> {
        // ignore ' " \
        let ascii1 = '('..='[';
        let ascii2 = ']'..='~';
        // CJK Unified Ideographs
        let cjk = '\u{4E00}'..='\u{9FFF}';

        let chars: Vec<char> = ascii1.chain(ascii2).chain(cjk).collect();
        prop::collection::vec(prop::sample::select(chars), 1..50)
            .prop_map(|v| v.into_iter().collect())
    }

    /// Strategy to generate strings suitable for unquoted object keys in JSON5
    ///
    /// Generates strings containing alphanumeric characters, underscores, dollar signs,
    /// and CJK Unicode characters that are valid as unquoted keys in JSON5.
    /// This tests the parser's ability to handle extended syntax for object keys.
    fn unquoted_string_strategy() -> impl Strategy<Value = String> {
        let number = '0'..='9';
        let lowercase = 'a'..='f';
        let uppercase = 'A'..='F';
        let underline = '_';
        let dollar = '$';
        // CJK Unified Ideographs
        let cjk = '\u{4E00}'..='\u{9FFF}';

        let mut chars: Vec<char> = number
            .chain(lowercase)
            .chain(uppercase)
            .chain(cjk)
            .collect();
        chars.push(underline);
        chars.push(dollar);
        prop::collection::vec(prop::sample::select(chars), 1..50)
            .prop_map(|v| v.into_iter().collect())
    }

    /// Strategy to generate standard JSON number values
    ///
    /// Generates integers (signed and unsigned) and floating-point numbers
    /// while excluding special cases like -0.0 that might cause comparison issues
    fn standard_number_strategy() -> impl Strategy<Value = Number> {
        prop_oneof![
            any::<u64>().prop_map(Number::UInt64),
            any::<i64>().prop_map(Number::Int64),
            any::<f64>()
                .prop_filter("Exclude -0.0", |x| *x != -0.0)
                .prop_map(Number::Float64),
        ]
    }

    /// Strategy to generate random hexadecimal numbers for testing
    ///
    /// Generates hexadecimal numbers with 0x/0X prefix (e.g., "0xFF", "0X1A3")
    /// to test the parser's ability to handle extended JSON5 hex number syntax
    fn hex_number_strategy() -> impl Strategy<Value = String> {
        let number = '0'..='9';
        let lowercase = 'a'..='f';
        let uppercase = 'A'..='F';

        let hex_digit =
            prop::sample::select(number.chain(lowercase).chain(uppercase).collect::<Vec<_>>());
        let hex_prefix = prop::sample::select(vec!['x', 'X']);
        let int_part =
            prop::collection::vec(hex_digit, 1..16).prop_map(|v| v.into_iter().collect::<String>());

        (hex_prefix, int_part).prop_map(|(x, i)| format!("0{}{}", x, i))
    }

    /// Strategy to generate JSON5 values for testing the extended JSON parser
    ///
    /// Creates a comprehensive set of JSON5 values including all extended syntax features:
    /// - Standard JSON literals (null, true, false)
    /// - Numbers (standard format)
    /// - Hexadecimal numbers (with 0x/0X prefix)
    /// - Double-quoted strings
    /// - Single-quoted strings
    /// - Arrays
    /// - Objects with different key styles (double-quoted, single-quoted, and unquoted)
    ///
    /// This strategy is used to verify that our parser correctly handles all JSON5 extensions
    fn json5_strategy() -> impl Strategy<Value = Json5Value> {
        let leaf = prop_oneof![
            Just(Json5Value::Null),
            any::<bool>().prop_map(Json5Value::Bool),
            standard_number_strategy().prop_map(Json5Value::Number),
            hex_number_strategy().prop_map(Json5Value::HexNumber),
            quoted_string_strategy().prop_map(Json5Value::DoubleQuotedString),
            quoted_string_strategy().prop_map(Json5Value::SingleQuotedString),
        ];

        leaf.prop_recursive(8, 256, 30, |inner| {
            prop_oneof![
                prop::collection::vec(inner.clone(), 0..10).prop_map(Json5Value::Array),
                prop::collection::btree_map(quoted_string_strategy(), inner.clone(), 0..20)
                    .prop_map(Json5Value::DoubleQuotedKeyObject),
                prop::collection::btree_map(quoted_string_strategy(), inner.clone(), 0..20)
                    .prop_map(Json5Value::SingleQuotedKeyObject),
                prop::collection::btree_map(unquoted_string_strategy(), inner, 0..20)
                    .prop_map(Json5Value::UnquotedKeyObject),
            ]
        })
    }

    /// Strategy to generate standard JSON values
    fn standard_json_strategy() -> impl Strategy<Value = Value<'static>> {
        let leaf = prop_oneof![
            Just(Value::Null),
            any::<bool>().prop_map(Value::Bool),
            standard_number_strategy().prop_map(Value::Number),
            string_strategy().prop_map(|v| Value::String(Cow::Owned(v))),
        ];

        leaf.prop_recursive(8, 256, 30, |inner| {
            prop_oneof![
                prop::collection::vec(inner.clone(), 0..10).prop_map(Value::Array),
                prop::collection::btree_map(string_strategy(), inner, 0..20)
                    .prop_map(Value::Object),
            ]
        })
    }

    proptest! {
        /// Tests the parser's ability to handle JSON5 syntax
        ///
        /// Generates JSON5 values and verifies that our parser produces
        /// the same results as the json_five crate
        #[test]
        fn test_json5_parser(json in json5_strategy()) {
            let source = format!("{}", json);

            let res1 = json_five::from_str::<serde_json::Value>(&source);
            let res2 = parse_value(source.as_bytes());
            let res3 = source.parse::<OwnedJsonb>();
            assert_eq!(res1.is_ok(), res2.is_ok());
            assert_eq!(res1.is_ok(), res3.is_ok());
            if res1.is_ok() {
                let res1 = format!("{}", res1.unwrap());
                let res2 = format!("{}", res2.unwrap());
                let res3 = format!("{}", res3.unwrap());
                assert_eq!(res1, res2);
                assert_eq!(res1, res3);
            }
        }
    }

    proptest! {
        /// Tests that standard JSON input round-trips through the parser
        #[test]
        fn test_standard_json_parser(json in standard_json_strategy()) {
            let source = format!("{}", json);
            let res1 = serde_json::from_slice::<serde_json::Value>(source.as_bytes());
            let res2 = parse_value(source.as_bytes());
            let res3 = source.parse::<OwnedJsonb>();
            assert_eq!(res1.is_ok(), res2.is_ok());
            assert_eq!(res1.is_ok(), res3.is_ok());
            if res1.is_ok() {
                let res2 = format!("{}", res2.unwrap());
                let res3 = format!("{}", res3.unwrap());
                assert_eq!(source, res2);
                assert_eq!(source, res3);
            }
        }
    }
}
