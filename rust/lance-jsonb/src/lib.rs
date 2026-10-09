// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-FileCopyrightText: Copyright 2023 Datafuse Labs
// Adapted from databendlabs/jsonb at commit fba895c5ebe77ce2539e187f9c652f51cbf195c3.

//! JSONB encoding for Lance JSON columns.
//!
//! Lance stores `lance.json` columns as JSONB values in `LargeBinary` arrays.
//! This crate owns that encoding: it parses JSON text into JSONB
//! ([`parse_value`], [`OwnedJsonb`]), reads JSONB values ([`RawJsonb`],
//! [`from_raw_jsonb`]), and evaluates SQL/JSONPath expressions over them
//! ([`jsonpath`]).
//!
//! ```
//! use lance_jsonb::OwnedJsonb;
//! use lance_jsonb::jsonpath::{Selector, parse_json_path};
//!
//! let jsonb: OwnedJsonb = r#"{"user":{"name":"Ada","tags":["a","b"]}}"#.parse().unwrap();
//! let path = parse_json_path(b"$.user.tags[1]").unwrap();
//! let value = Selector::new(jsonb.as_raw()).select_value(&path).unwrap().unwrap();
//! assert_eq!(value.to_string(), r#""b""#);
//! ```
//!
//! The implementation is adapted from
//! [databendlabs/jsonb](https://github.com/databendlabs/jsonb) and keeps only
//! what Lance uses. See the crate README for the upstream revision and the
//! differences from it.
//!
//! ## Compatibility
//!
//! The byte layout below is persisted in Lance data files. Every change must
//! keep reading values written by released Lance versions, and writers must keep
//! producing bytes that those versions can read. `tests/it/compat.rs` pins the
//! bytes produced for representative inputs.
//!
//! ## Encoding format
//!
//! The JSONB encoding format is a tree-like structure. Each node contains a container header, a number of JEntry headers, and nested encoding values.
//!
//! - 32-bit container header. 3 bits identify the type of value, including `scalar`, `object` and `array`, and 29 bits identify the number of JEntries in the `array` or `object`. The root node of the `jsonb` value is always a container header.
//!   - `scalar` container header: `0x20000000`
//!   - `object` container header: `0x40000000`
//!   - `array` container header: `0x80000000`
//! - 32-bit JEntry header. 1 bit identifies whether the JEntry stores a length or an offset, 3 bits identify the type of value, including `null`, `string`, `number`, `false`, `true` and `container`, and the remaining 28 bits identify the length or offset of the encoding value.
//!   - `null` JEntry header: `0x00000000`
//!   - `string` JEntry header: `0x10000000`
//!   - `number` JEntry header: `0x20000000`
//!   - `false` JEntry header: `0x30000000`
//!   - `true` JEntry header: `0x40000000`
//!   - `container` JEntry header `0x50000000`
//! - Encoding value. Different types of JEntry header have different encoding values.
//!   - `null`, `true`, `false`: no encoding value, identified by the JEntry header.
//!   - `string`: a normal UTF-8 string.
//!   - `number`: an encoded number to represent uint64s, int64s and float64s.
//!   - `container`: a nested `json` value with a recursive structure.
//!
//! #### An encoding example
//!
//! ```text
//! // JSON value
//! [false, 10, {"k":"v"}]
//!
//! // JSONB encoding
//! 0x80000003    array container header (3 JEntries)
//! 0x30000000    false JEntry header (no encoding value)
//! 0x20000002    number JEntry header (encoding value length 2)
//! 0x5000000e    container JEntry header (encoding value length 14)
//! 0x500a        number encoding value (10)
//! 0x40000001    object container header (1 JEntry)
//! 0x10000001    string key JEntry header (encoding value length 1)
//! 0x10000001    string value JEntry header (encoding value length 1)
//! 0x6b          string encoding value ("k")
//! 0x76          string encoding value ("v")
//! ```

mod constants;
mod core;
mod error;
mod extension;
mod functions;
pub mod jsonpath;
mod number;
mod owned;
mod parser;
mod raw;
mod util;
mod value;

pub use error::Error;
pub use extension::Date;
pub use extension::Interval;
pub use extension::Timestamp;
pub use extension::TimestampTz;
pub use number::Decimal64;
pub use number::Decimal128;
pub use number::Decimal256;
pub use number::Number;
pub use owned::OwnedJsonb;
pub use parser::parse_value;
pub use raw::RawJsonb;
pub use raw::from_raw_jsonb;
pub use value::Object;
pub use value::Value;
