// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-FileCopyrightText: Copyright 2023 Datafuse Labs
// Adapted from databendlabs/jsonb at commit fba895c5ebe77ce2539e187f9c652f51cbf195c3.

use std::collections::VecDeque;

use byteorder::BigEndian;
use byteorder::WriteBytesExt;
use serde::ser;
use serde::ser::Serialize;
use serde::ser::SerializeMap;
use serde::ser::SerializeSeq;

use super::constants::*;
use super::jentry::JEntry;
use crate::RawJsonb;
use crate::extension::ExtensionValue;
use crate::number::Number;
use crate::value::Object;
use crate::value::Value;

impl Serialize for RawJsonb<'_> {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> core::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut index = 0;
        let (header_type, header_len) = self
            .read_header(index)
            .map_err(|e| ser::Error::custom(format!("{e}")))?;
        index += 4;

        match header_type {
            SCALAR_CONTAINER_TAG => {
                let jentry = self
                    .read_jentry(index)
                    .map_err(|e| ser::Error::custom(format!("{e}")))?;
                index += 4;

                let payload_start = index;
                let payload_end = index + jentry.length as usize;
                match jentry.type_code {
                    NULL_TAG => serializer.serialize_unit(),
                    TRUE_TAG => serializer.serialize_bool(true),
                    FALSE_TAG => serializer.serialize_bool(false),
                    NUMBER_TAG => {
                        let num = Number::decode(&self.data[payload_start..payload_end])
                            .map_err(|e| ser::Error::custom(format!("{e}")))?;
                        num.serialize(serializer)
                    }
                    STRING_TAG => {
                        let s = unsafe {
                            std::str::from_utf8_unchecked(&self.data[payload_start..payload_end])
                        };
                        serializer.serialize_str(s)
                    }
                    EXTENSION_TAG => {
                        let val = ExtensionValue::decode(&self.data[payload_start..payload_end])
                            .map_err(|e| ser::Error::custom(format!("{e}")))?;
                        let s = format!("{}", val);
                        serializer.serialize_str(&s)
                    }
                    CONTAINER_TAG => {
                        // Scalar header can't have container jentry tag
                        Err(ser::Error::custom("Invalid jsonb".to_string()))
                    }
                    _ => Err(ser::Error::custom("Invalid jsonb".to_string())),
                }
            }
            ARRAY_CONTAINER_TAG => {
                let mut serialize_seq = serializer.serialize_seq(Some(header_len))?;

                let mut payload_start = index + 4 * header_len;
                for _ in 0..header_len {
                    let jentry = self
                        .read_jentry(index)
                        .map_err(|e| ser::Error::custom(format!("{e}")))?;
                    index += 4;

                    let payload_end = payload_start + jentry.length as usize;
                    match jentry.type_code {
                        NULL_TAG => serialize_seq.serialize_element(&())?,
                        TRUE_TAG => serialize_seq.serialize_element(&true)?,
                        FALSE_TAG => serialize_seq.serialize_element(&false)?,
                        NUMBER_TAG => {
                            let num = Number::decode(&self.data[payload_start..payload_end])
                                .map_err(|e| ser::Error::custom(format!("{e}")))?;
                            serialize_seq.serialize_element(&num)?;
                        }
                        STRING_TAG => {
                            let s = unsafe {
                                std::str::from_utf8_unchecked(
                                    &self.data[payload_start..payload_end],
                                )
                            };
                            serialize_seq.serialize_element(&s)?;
                        }
                        EXTENSION_TAG => {
                            let val =
                                ExtensionValue::decode(&self.data[payload_start..payload_end])
                                    .map_err(|e| ser::Error::custom(format!("{e}")))?;
                            let s = format!("{}", val);
                            serialize_seq.serialize_element(&s)?;
                        }
                        CONTAINER_TAG => {
                            let inner_raw_jsonb =
                                RawJsonb::new(&self.data[payload_start..payload_end]);
                            serialize_seq.serialize_element(&inner_raw_jsonb)?;
                        }
                        _ => {
                            return Err(ser::Error::custom("Invalid jsonb".to_string()));
                        }
                    }
                    payload_start = payload_end;
                }
                serialize_seq.end()
            }
            OBJECT_CONTAINER_TAG => {
                let mut serialize_map = serializer.serialize_map(Some(header_len))?;

                let mut keys = VecDeque::with_capacity(header_len);
                let mut payload_start = index + 8 * header_len;
                for _ in 0..header_len {
                    let jentry = self
                        .read_jentry(index)
                        .map_err(|e| ser::Error::custom(format!("{e}")))?;
                    index += 4;

                    let payload_end = payload_start + jentry.length as usize;
                    match jentry.type_code {
                        STRING_TAG => {
                            let s = unsafe {
                                std::str::from_utf8_unchecked(
                                    &self.data[payload_start..payload_end],
                                )
                            };
                            keys.push_back(s);
                        }
                        _ => {
                            return Err(ser::Error::custom("Invalid jsonb".to_string()));
                        }
                    }
                    payload_start = payload_end;
                }

                for _ in 0..header_len {
                    let jentry = self
                        .read_jentry(index)
                        .map_err(|e| ser::Error::custom(format!("{e}")))?;
                    index += 4;

                    let payload_end = payload_start + jentry.length as usize;
                    let k = keys.pop_front().unwrap();
                    match jentry.type_code {
                        NULL_TAG => serialize_map.serialize_entry(&k, &())?,
                        TRUE_TAG => serialize_map.serialize_entry(&k, &true)?,
                        FALSE_TAG => serialize_map.serialize_entry(&k, &false)?,
                        NUMBER_TAG => {
                            let num = Number::decode(&self.data[payload_start..payload_end])
                                .map_err(|e| ser::Error::custom(format!("{e}")))?;
                            serialize_map.serialize_entry(&k, &num)?;
                        }
                        STRING_TAG => {
                            let s = unsafe {
                                std::str::from_utf8_unchecked(
                                    &self.data[payload_start..payload_end],
                                )
                            };
                            serialize_map.serialize_entry(&k, &s)?;
                        }
                        EXTENSION_TAG => {
                            let val =
                                ExtensionValue::decode(&self.data[payload_start..payload_end])
                                    .map_err(|e| ser::Error::custom(format!("{e}")))?;
                            let s = format!("{}", val);
                            serialize_map.serialize_entry(&k, &s)?;
                        }
                        CONTAINER_TAG => {
                            let inner_raw_jsonb =
                                RawJsonb::new(&self.data[payload_start..payload_end]);
                            serialize_map.serialize_entry(&k, &inner_raw_jsonb)?;
                        }
                        _ => {
                            return Err(ser::Error::custom("Invalid jsonb".to_string()));
                        }
                    }
                    payload_start = payload_end;
                }
                serialize_map.end()
            }
            _ => Err(ser::Error::custom("Invalid jsonb".to_string())),
        }
    }
}

/// `BaseEncoder` provides common buffer management functionality for `JSONB` encoding.
/// It handles low-level operations like reserving space for JEntries, writing encoded
/// JEntries back to the buffer, and encoding container headers.
struct BaseEncoder<'a> {
    buf: &'a mut Vec<u8>,
}

impl<'a> BaseEncoder<'a> {
    /// Creates a new `BaseEncoder` with the given buffer.
    fn new(buf: &'a mut Vec<u8>) -> Self {
        Self { buf }
    }

    /// Reserves space in the buffer for JEntries that will be filled in later.
    /// Returns the starting index where the JEntries will be placed.
    fn reserve_jentries(&mut self, len: usize) -> usize {
        let old_len = self.buf.len();
        let new_len = old_len + len;
        self.buf.resize(new_len, 0);
        old_len
    }

    /// Writes an encoded `JEntry` to the buffer at the specified index.
    /// Updates the index to point to the next `JEntry` position.
    fn replace_jentry(&mut self, jentry: JEntry, jentry_index: &mut usize) {
        let jentry_bytes = jentry.encoded().to_be_bytes();
        for (i, b) in jentry_bytes.iter().enumerate() {
            self.buf[*jentry_index + i] = *b;
        }
        *jentry_index += 4;
    }

    /// Encodes a scalar container header and reserves space for its `JEntry`.
    /// Returns the total length of the scalar container and the index where its JEntry will be placed.
    fn encode_scalar_header(&mut self) -> (usize, usize) {
        let header = SCALAR_CONTAINER_TAG;
        self.buf.write_u32::<BigEndian>(header).unwrap();

        // Scalar Value only has one JEntry
        let scalar_len = 4 + 4;
        let jentry_index = self.reserve_jentries(4);
        (scalar_len, jentry_index)
    }

    /// Encodes an array container header and reserves space for its JEntries.
    /// Returns the total length of the array container and the index where its JEntries will be placed.
    fn encode_array_header(&mut self, len: usize) -> (usize, usize) {
        let header = ARRAY_CONTAINER_TAG | len as u32;
        self.buf.write_u32::<BigEndian>(header).unwrap();

        // `Array` has N `JEntries`
        let array_len = 4 + len * 4;
        let jentry_index = self.reserve_jentries(len * 4);

        (array_len, jentry_index)
    }

    /// Encodes an object container header and reserves space for its JEntries.
    /// Returns the total length of the object container and the index where its JEntries will be placed.
    fn encode_object_header(&mut self, len: usize) -> (usize, usize) {
        let header = OBJECT_CONTAINER_TAG | len as u32;
        self.buf.write_u32::<BigEndian>(header).unwrap();

        // `Object` has 2 * N `JEntries`
        let object_len = 4 + len * 8;
        let jentry_index = self.reserve_jentries(len * 8);

        (object_len, jentry_index)
    }
}

/// Encoder for serializing Value types to `JSONB` binary format.
/// Uses `BaseEncoder` for common buffer management operations.
pub struct Encoder<'a> {
    base_encoder: BaseEncoder<'a>,
}

impl<'a> Encoder<'a> {
    /// Creates a new `Encoder` with the given buffer.
    pub(crate) fn new(buf: &'a mut Vec<u8>) -> Self {
        let base_encoder = BaseEncoder::new(buf);
        Self { base_encoder }
    }

    /// Encodes a `Value` into `JSONB` binary format.
    /// Dispatches to the appropriate encoding method based on the value type.
    pub(crate) fn encode(&mut self, value: &Value<'a>) {
        match value {
            Value::Array(array) => self.encode_array(array),
            Value::Object(obj) => self.encode_object(obj),
            _ => self.encode_scalar(value),
        };
    }

    /// Encodes a scalar `Value` (null, bool, number, string, or extension types).
    /// Returns the total length of the encoded scalar.
    fn encode_scalar(&mut self, value: &Value<'a>) -> usize {
        let (mut scalar_len, mut jentry_index) = self.base_encoder.encode_scalar_header();

        let jentry = self.encode_value(value);
        scalar_len += jentry.length as usize;
        self.base_encoder.replace_jentry(jentry, &mut jentry_index);

        scalar_len
    }

    /// Encodes an array of Values.
    /// Returns the total length of the encoded array.
    fn encode_array(&mut self, values: &[Value<'a>]) -> usize {
        let (mut array_len, mut jentry_index) = self.base_encoder.encode_array_header(values.len());

        // encode all values
        for value in values.iter() {
            let jentry = self.encode_value(value);
            array_len += jentry.length as usize;
            self.base_encoder.replace_jentry(jentry, &mut jentry_index);
        }

        array_len
    }

    /// Encodes an object of Values (map of string keys to Values).
    /// Returns the total length of the encoded object.
    fn encode_object(&mut self, obj: &Object<'a>) -> usize {
        let (mut object_len, mut jentry_index) = self.base_encoder.encode_object_header(obj.len());

        // encode all keys first
        for key in obj.keys() {
            let len = key.len();
            object_len += len;
            self.base_encoder.buf.extend_from_slice(key.as_bytes());
            let jentry = JEntry::make_string_jentry(len);
            self.base_encoder.replace_jentry(jentry, &mut jentry_index);
        }

        // encode all values
        for value in obj.values() {
            let jentry = self.encode_value(value);
            object_len += jentry.length as usize;
            self.base_encoder.replace_jentry(jentry, &mut jentry_index);
        }

        object_len
    }

    /// Encodes a single `Value` and returns its `JEntry`.
    /// The `JEntry` contains metadata about the encoded value.
    fn encode_value(&mut self, value: &Value<'a>) -> JEntry {
        let old_off = self.base_encoder.buf.len();

        match value {
            Value::Null => JEntry::make_null_jentry(),
            Value::Bool(v) => {
                if *v {
                    JEntry::make_true_jentry()
                } else {
                    JEntry::make_false_jentry()
                }
            }
            Value::Number(v) => {
                let _ = v.compact_encode(&mut self.base_encoder.buf).unwrap();
                let len = self.base_encoder.buf.len() - old_off;
                JEntry::make_number_jentry(len)
            }
            Value::String(s) => {
                let len = s.len();
                self.base_encoder
                    .buf
                    .extend_from_slice(s.as_ref().as_bytes());
                JEntry::make_string_jentry(len)
            }
            Value::Binary(v) => {
                let val = ExtensionValue::Binary(v);
                let _ = val.compact_encode(&mut self.base_encoder.buf).unwrap();
                let len = self.base_encoder.buf.len() - old_off;
                JEntry::make_extension_jentry(len)
            }
            Value::Date(v) => {
                let val = ExtensionValue::Date(v.clone());
                let _ = val.compact_encode(&mut self.base_encoder.buf).unwrap();
                let len = self.base_encoder.buf.len() - old_off;
                JEntry::make_extension_jentry(len)
            }
            Value::Timestamp(v) => {
                let val = ExtensionValue::Timestamp(v.clone());
                let _ = val.compact_encode(&mut self.base_encoder.buf).unwrap();
                let len = self.base_encoder.buf.len() - old_off;
                JEntry::make_extension_jentry(len)
            }
            Value::TimestampTz(v) => {
                let val = ExtensionValue::TimestampTz(v.clone());
                let _ = val.compact_encode(&mut self.base_encoder.buf).unwrap();
                let len = self.base_encoder.buf.len() - old_off;
                JEntry::make_extension_jentry(len)
            }
            Value::Interval(v) => {
                let val = ExtensionValue::Interval(v.clone());
                let _ = val.compact_encode(&mut self.base_encoder.buf).unwrap();
                let len = self.base_encoder.buf.len() - old_off;
                JEntry::make_extension_jentry(len)
            }
            Value::Array(array) => {
                let len = self.encode_array(array);
                JEntry::make_container_jentry(len)
            }
            Value::Object(obj) => {
                let len = self.encode_object(obj);
                JEntry::make_container_jentry(len)
            }
        }
    }
}
