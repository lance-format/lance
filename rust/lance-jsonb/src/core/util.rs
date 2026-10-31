// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-FileCopyrightText: Copyright 2023 Datafuse Labs
// Adapted from databendlabs/jsonb at commit fba895c5ebe77ce2539e187f9c652f51cbf195c3.

use core::ops::Range;
use std::borrow::Cow;
use std::io::Write;

use byteorder::BigEndian;
use byteorder::WriteBytesExt;
use ethnum::i256;

use super::constants::*;
use super::jentry::JEntry;
use crate::Number;
use crate::OwnedJsonb;
use crate::RawJsonb;
use crate::core::ExtensionItem;
use crate::core::JsonbItem;
use crate::core::JsonbItemType;
use crate::core::NumberItem;
use crate::error::*;
use crate::extension::Date;
use crate::extension::ExtensionValue;
use crate::extension::Interval;
use crate::extension::Timestamp;
use crate::extension::TimestampTz;
use crate::number::Decimal64;
use crate::number::Decimal128;
use crate::number::Decimal256;

impl<'a> JsonbItem<'a> {
    pub(crate) fn from_raw_jsonb(raw_jsonb: RawJsonb<'a>) -> Result<Self> {
        let (header_type, _) = raw_jsonb.read_header(0)?;
        match header_type {
            SCALAR_CONTAINER_TAG => {
                let (jentry, data) = raw_jsonb.read_jentry_and_data(4, 8)?;
                let item = match jentry.type_code {
                    NULL_TAG => JsonbItem::Null,
                    TRUE_TAG => JsonbItem::Boolean(true),
                    FALSE_TAG => JsonbItem::Boolean(false),
                    NUMBER_TAG => JsonbItem::Number(NumberItem::Raw(data)),
                    STRING_TAG => {
                        let s = Cow::Borrowed(unsafe { std::str::from_utf8_unchecked(data) });
                        JsonbItem::String(s)
                    }
                    EXTENSION_TAG => JsonbItem::Extension(ExtensionItem::Raw(data)),
                    _ => {
                        return Err(Error::InvalidJsonb);
                    }
                };
                Ok(item)
            }
            OBJECT_CONTAINER_TAG | ARRAY_CONTAINER_TAG => Ok(JsonbItem::Raw(raw_jsonb)),
            _ => Err(Error::InvalidJsonb),
        }
    }
}

impl<'a> RawJsonb<'a> {
    /// Returns the type information of a JSONB data item
    ///
    /// # Returns
    ///
    /// Returns `Result<JsonbItemType>`, where `JsonbItemType` can be one of:
    /// - `JsonbItemType::Null` - represents a null value
    /// - `JsonbItemType::Boolean` - represents a boolean value (true or false)
    /// - `JsonbItemType::Number` - represents a numeric type
    /// - `JsonbItemType::String` - represents a string type
    /// - `JsonbItemType::Extension` - represents an extension type
    /// - `JsonbItemType::Array(n)` - represents an array type, where n is the number of array elements
    /// - `JsonbItemType::Object(n)` - represents an object type, where n is the number of key-value pairs
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidJsonb` if the JSONB data format is invalid
    pub(crate) fn jsonb_item_type(&self) -> Result<JsonbItemType> {
        let (header_type, header_len) = self.read_header(0)?;
        match header_type {
            SCALAR_CONTAINER_TAG => {
                let jentry = self.read_jentry(4)?;

                match jentry.type_code {
                    NULL_TAG => Ok(JsonbItemType::Null),
                    TRUE_TAG => Ok(JsonbItemType::Boolean),
                    FALSE_TAG => Ok(JsonbItemType::Boolean),
                    NUMBER_TAG => Ok(JsonbItemType::Number),
                    STRING_TAG => Ok(JsonbItemType::String),
                    EXTENSION_TAG => Ok(JsonbItemType::Extension),
                    _ => Err(Error::InvalidJsonb),
                }
            }
            ARRAY_CONTAINER_TAG => Ok(JsonbItemType::Array(header_len)),
            OBJECT_CONTAINER_TAG => Ok(JsonbItemType::Object(header_len)),
            _ => Err(Error::InvalidJsonb),
        }
    }

    pub(crate) fn get_object_value_by_key_name(
        &self,
        key_name: &Cow<'a, str>,
        eq_func: impl Fn(&[u8], &[u8]) -> bool,
    ) -> Result<Option<JsonbItem<'a>>> {
        let (header_type, length) = self.read_header(0)?;
        if header_type != OBJECT_CONTAINER_TAG || length == 0 {
            return Ok(None);
        }
        let mut index = 0;
        let mut jentry_offset = 4;
        let mut item_offset = 4 + 8 * length;

        let mut key_matched = false;
        let name_len = key_name.len();
        let name_bytes = key_name.as_bytes();
        while index < length {
            let key_jentry = self.read_jentry(jentry_offset)?;
            let key_len = key_jentry.length as usize;

            index += 1;
            jentry_offset += 4;
            item_offset += key_len;

            // check if key match the name.
            if name_len == key_len {
                let key_range = Range {
                    start: item_offset - key_len,
                    end: item_offset,
                };
                let key_data = self.slice(key_range)?;
                if eq_func(name_bytes, key_data) {
                    key_matched = true;
                    break;
                }
            }
        }

        if !key_matched {
            return Ok(None);
        }
        let val_index = index - 1;
        // skip unmatched keys and values.
        while index < length + val_index {
            let jentry = self.read_jentry(jentry_offset)?;
            index += 1;
            jentry_offset += 4;
            item_offset += jentry.length as usize;
        }

        // read value item data
        let value_jentry = self.read_jentry(jentry_offset)?;
        let value_len = value_jentry.length as usize;

        let value_range = Range {
            start: item_offset,
            end: item_offset + value_len,
        };
        let value_data = self.slice(value_range)?;
        let value_item = jentry_to_jsonb_item(value_jentry, value_data);
        Ok(Some(value_item))
    }

    pub(super) fn read_header(&self, index: usize) -> Result<(u32, usize)> {
        let header = self.read_u32(index)?;
        let header_type = header & CONTAINER_HEADER_TYPE_MASK;
        match header_type {
            SCALAR_CONTAINER_TAG | OBJECT_CONTAINER_TAG | ARRAY_CONTAINER_TAG => {}
            _ => {
                return Err(Error::InvalidJsonb);
            }
        }
        let header_len = header & CONTAINER_HEADER_LEN_MASK;
        Ok((header_type, header_len as usize))
    }

    pub(super) fn read_jentry(&self, index: usize) -> Result<JEntry> {
        let jentry_encoded = self.read_u32(index)?;
        let jentry = JEntry::decode_jentry(jentry_encoded);
        Ok(jentry)
    }

    pub(super) fn read_jentry_and_data(
        &self,
        index: usize,
        data_index: usize,
    ) -> Result<(JEntry, &'a [u8])> {
        let jentry = self.read_jentry(index)?;
        let range = Range {
            start: data_index,
            end: data_index + jentry.length as usize,
        };
        let data = self.slice(range)?;
        Ok((jentry, data))
    }

    pub(super) fn read_u32(&self, idx: usize) -> Result<u32> {
        let bytes: [u8; 4] = self
            .data
            .get(idx..idx + 4)
            .ok_or(Error::InvalidEOF)?
            .try_into()
            .unwrap();
        Ok(u32::from_be_bytes(bytes))
    }

    pub(super) fn slice(&self, range: Range<usize>) -> Result<&'a [u8]> {
        // Check for potential out-of-bounds access before creating item
        if range.end > self.len() {
            return Err(Error::InvalidJsonb);
        }
        Ok(&self.data[range])
    }
}

impl OwnedJsonb {
    pub(crate) fn from_item(item: JsonbItem<'_>) -> Result<Self> {
        match item {
            JsonbItem::Raw(raw_jsonb) => Ok(raw_jsonb.to_owned()),
            JsonbItem::Owned(owned_jsonb) => Ok(owned_jsonb),
            _ => {
                let mut len = match item {
                    JsonbItem::Null => 0,
                    JsonbItem::Boolean(_) => 0,
                    JsonbItem::Number(NumberItem::Raw(data)) => data.len(),
                    JsonbItem::String(ref s) => s.len(),
                    JsonbItem::Extension(ExtensionItem::Raw(data)) => data.len(),
                    // The estimated lengths for number and extension.
                    _ => 10,
                };

                // add the length of header and jentry.
                len += 8;
                let mut buf = Vec::with_capacity(len);
                let header = SCALAR_CONTAINER_TAG;
                buf.write_u32::<BigEndian>(header)?;

                match item {
                    JsonbItem::Null => {
                        let jentry = JEntry::make_null_jentry();
                        buf.write_u32::<BigEndian>(jentry.encoded())?;
                    }
                    JsonbItem::Boolean(v) => {
                        let jentry = if v {
                            JEntry::make_true_jentry()
                        } else {
                            JEntry::make_false_jentry()
                        };
                        buf.write_u32::<BigEndian>(jentry.encoded())?;
                    }
                    JsonbItem::Number(num) => match num {
                        NumberItem::Raw(data) => {
                            let jentry = JEntry::make_number_jentry(data.len());
                            buf.write_u32::<BigEndian>(jentry.encoded())?;
                            buf.extend_from_slice(data);
                        }
                        NumberItem::Number(num) => {
                            let mut data = vec![];
                            let len = num.compact_encode(&mut data)?;
                            let jentry = JEntry::make_number_jentry(len);
                            buf.write_u32::<BigEndian>(jentry.encoded())?;
                            buf.extend_from_slice(&data);
                        }
                    },
                    JsonbItem::String(s) => {
                        let jentry = JEntry::make_string_jentry(s.len());
                        buf.write_u32::<BigEndian>(jentry.encoded())?;
                        buf.extend_from_slice(s.as_bytes());
                    }
                    JsonbItem::Extension(ext) => match ext {
                        ExtensionItem::Raw(data) => {
                            let jentry = JEntry::make_extension_jentry(data.len());
                            buf.write_u32::<BigEndian>(jentry.encoded())?;
                            buf.extend_from_slice(data);
                        }
                        ExtensionItem::Extension(ext) => {
                            let mut data = vec![];
                            let len = ext.compact_encode(&mut data)?;
                            let jentry = JEntry::make_extension_jentry(len);
                            buf.write_u32::<BigEndian>(jentry.encoded())?;
                            buf.extend_from_slice(&data);
                        }
                    },
                    _ => unreachable!(),
                }
                Ok(Self::new(buf))
            }
        }
    }
}

impl Number {
    #[inline]
    pub(crate) fn compact_encode<W: Write>(&self, mut writer: W) -> Result<usize> {
        match self {
            Self::Int64(v) => {
                if *v == 0 {
                    writer.write_all(&[NUMBER_ZERO])?;
                    return Ok(1);
                }
                writer.write_all(&[NUMBER_INT])?;
                if *v >= i8::MIN.into() && *v <= i8::MAX.into() {
                    writer.write_all(&(*v as i8).to_be_bytes())?;
                    Ok(2)
                } else if *v >= i16::MIN.into() && *v <= i16::MAX.into() {
                    writer.write_all(&(*v as i16).to_be_bytes())?;
                    Ok(3)
                } else if *v >= i32::MIN.into() && *v <= i32::MAX.into() {
                    writer.write_all(&(*v as i32).to_be_bytes())?;
                    Ok(5)
                } else {
                    writer.write_all(&v.to_be_bytes())?;
                    Ok(9)
                }
            }
            Self::UInt64(v) => {
                if *v == 0 {
                    writer.write_all(&[NUMBER_ZERO])?;
                    return Ok(1);
                }
                writer.write_all(&[NUMBER_UINT])?;
                if *v <= u8::MAX.into() {
                    writer.write_all(&(*v as u8).to_be_bytes())?;
                    Ok(2)
                } else if *v <= u16::MAX.into() {
                    writer.write_all(&(*v as u16).to_be_bytes())?;
                    Ok(3)
                } else if *v <= u32::MAX.into() {
                    writer.write_all(&(*v as u32).to_be_bytes())?;
                    Ok(5)
                } else {
                    writer.write_all(&v.to_be_bytes())?;
                    Ok(9)
                }
            }
            Self::Float64(v) => {
                if v.is_nan() {
                    writer.write_all(&[NUMBER_NAN])?;
                    return Ok(1);
                } else if v.is_infinite() {
                    if v.is_sign_negative() {
                        writer.write_all(&[NUMBER_NEG_INF])?;
                    } else {
                        writer.write_all(&[NUMBER_INF])?;
                    }
                    return Ok(1);
                }
                writer.write_all(&[NUMBER_FLOAT])?;
                writer.write_all(&v.to_be_bytes())?;
                Ok(9)
            }
            Self::Decimal64(v) => {
                writer.write_all(&[NUMBER_DECIMAL])?;
                writer.write_all(&v.value.to_be_bytes())?;
                writer.write_all(&v.scale.to_be_bytes())?;
                Ok(10)
            }
            Self::Decimal128(v) => {
                writer.write_all(&[NUMBER_DECIMAL])?;
                writer.write_all(&v.value.to_be_bytes())?;
                writer.write_all(&v.scale.to_be_bytes())?;
                Ok(18)
            }
            Self::Decimal256(v) => {
                writer.write_all(&[NUMBER_DECIMAL])?;
                writer.write_all(&v.value.to_be_bytes())?;
                writer.write_all(&v.scale.to_be_bytes())?;
                Ok(34)
            }
        }
    }

    #[inline]
    pub(crate) fn decode(bytes: &[u8]) -> Result<Self> {
        let mut len = bytes.len();
        assert!(len > 0);
        len -= 1;

        let ty = bytes[0];
        let num = match ty {
            NUMBER_ZERO => Self::UInt64(0),
            NUMBER_NAN => Self::Float64(f64::NAN),
            NUMBER_INF => Self::Float64(f64::INFINITY),
            NUMBER_NEG_INF => Self::Float64(f64::NEG_INFINITY),
            NUMBER_INT => match len {
                1 => Self::Int64(i8::from_be_bytes(bytes[1..].try_into().unwrap()) as i64),
                2 => Self::Int64(i16::from_be_bytes(bytes[1..].try_into().unwrap()) as i64),
                4 => Self::Int64(i32::from_be_bytes(bytes[1..].try_into().unwrap()) as i64),
                8 => Self::Int64(i64::from_be_bytes(bytes[1..].try_into().unwrap())),
                _ => {
                    return Err(Error::InvalidJsonbNumber);
                }
            },
            NUMBER_UINT => match len {
                1 => Self::UInt64(u8::from_be_bytes(bytes[1..].try_into().unwrap()) as u64),
                2 => Self::UInt64(u16::from_be_bytes(bytes[1..].try_into().unwrap()) as u64),
                4 => Self::UInt64(u32::from_be_bytes(bytes[1..].try_into().unwrap()) as u64),
                8 => Self::UInt64(u64::from_be_bytes(bytes[1..].try_into().unwrap())),
                _ => {
                    return Err(Error::InvalidJsonbNumber);
                }
            },
            NUMBER_FLOAT => Self::Float64(f64::from_be_bytes(bytes[1..].try_into().unwrap())),
            NUMBER_DECIMAL => match len {
                9 => {
                    let value = i64::from_be_bytes(bytes[1..9].try_into().unwrap());
                    let scale = u8::from_be_bytes(bytes[9..10].try_into().unwrap());
                    let dec = Decimal64 { scale, value };
                    Self::Decimal64(dec)
                }
                17 => {
                    let value = i128::from_be_bytes(bytes[1..17].try_into().unwrap());
                    let scale = u8::from_be_bytes(bytes[17..18].try_into().unwrap());
                    let dec = Decimal128 { scale, value };
                    Self::Decimal128(dec)
                }
                18 => {
                    // Compatible with deprecated Decimal128 formats, including precision
                    let value = i128::from_be_bytes(bytes[1..17].try_into().unwrap());
                    let scale = u8::from_be_bytes(bytes[18..19].try_into().unwrap());
                    let dec = Decimal128 { scale, value };
                    Self::Decimal128(dec)
                }
                33 => {
                    let value = i256::from_be_bytes(bytes[1..33].try_into().unwrap());
                    let scale = u8::from_be_bytes(bytes[33..34].try_into().unwrap());
                    let dec = Decimal256 { scale, value };
                    Self::Decimal256(dec)
                }
                34 => {
                    // Compatible with deprecated Decimal256 formats, including precision
                    let value = i256::from_be_bytes(bytes[1..33].try_into().unwrap());
                    let scale = u8::from_be_bytes(bytes[34..35].try_into().unwrap());
                    let dec = Decimal256 { scale, value };
                    Self::Decimal256(dec)
                }
                _ => {
                    return Err(Error::InvalidJsonbNumber);
                }
            },
            _ => {
                return Err(Error::InvalidJsonbNumber);
            }
        };
        Ok(num)
    }
}

impl ExtensionValue<'_> {
    #[inline]
    pub(crate) fn compact_encode<W: Write>(&self, mut writer: W) -> Result<usize> {
        match self {
            ExtensionValue::Binary(v) => {
                writer.write_all(&[EXTENSION_BINARY])?;
                let len = v.len() + 1;
                writer.write_all(v)?;
                Ok(len)
            }
            ExtensionValue::Date(v) => {
                writer.write_all(&[EXTENSION_DATE])?;
                writer.write_all(&v.value.to_be_bytes())?;
                Ok(5)
            }
            ExtensionValue::Timestamp(v) => {
                writer.write_all(&[EXTENSION_TIMESTAMP])?;
                writer.write_all(&v.value.to_be_bytes())?;
                Ok(9)
            }
            ExtensionValue::TimestampTz(v) => {
                writer.write_all(&[EXTENSION_TIMESTAMP_TZ])?;
                writer.write_all(&v.value.to_be_bytes())?;
                writer.write_all(&v.offset.to_be_bytes())?;
                Ok(13)
            }
            ExtensionValue::Interval(v) => {
                writer.write_all(&[EXTENSION_INTERVAL])?;
                writer.write_all(&v.months.to_be_bytes())?;
                writer.write_all(&v.days.to_be_bytes())?;
                writer.write_all(&v.micros.to_be_bytes())?;
                Ok(17)
            }
        }
    }

    #[inline]
    pub(crate) fn decode(bytes: &[u8]) -> Result<ExtensionValue<'_>> {
        let mut len = bytes.len();
        assert!(len > 0);
        len -= 1;

        let ty = bytes[0];
        let val = match ty {
            EXTENSION_BINARY => {
                let v = &bytes[1..len + 1];
                ExtensionValue::Binary(v)
            }
            EXTENSION_DATE => {
                if len != 4 {
                    return Err(Error::InvalidJsonbNumber);
                }
                let value = i32::from_be_bytes(bytes[1..5].try_into().unwrap());

                ExtensionValue::Date(Date { value })
            }
            EXTENSION_TIMESTAMP => {
                if len != 8 {
                    return Err(Error::InvalidJsonbNumber);
                }
                let value = i64::from_be_bytes(bytes[1..9].try_into().unwrap());

                ExtensionValue::Timestamp(Timestamp { value })
            }
            EXTENSION_TIMESTAMP_TZ => {
                if len != 9 && len != 12 {
                    return Err(Error::InvalidJsonbNumber);
                }
                let value = i64::from_be_bytes(bytes[1..9].try_into().unwrap());
                let offset = if len == 9 {
                    let hours = i8::from_be_bytes(bytes[9..10].try_into().unwrap());
                    (hours as i32) * 3_600
                } else {
                    i32::from_be_bytes(bytes[9..13].try_into().unwrap())
                };

                ExtensionValue::TimestampTz(TimestampTz { offset, value })
            }
            EXTENSION_INTERVAL => {
                if len != 16 {
                    return Err(Error::InvalidJsonbNumber);
                }
                let months = i32::from_be_bytes(bytes[1..5].try_into().unwrap());
                let days = i32::from_be_bytes(bytes[5..9].try_into().unwrap());
                let micros = i64::from_be_bytes(bytes[9..17].try_into().unwrap());
                ExtensionValue::Interval(Interval {
                    months,
                    days,
                    micros,
                })
            }
            _ => {
                return Err(Error::InvalidJsonbExtension);
            }
        };
        Ok(val)
    }
}

pub(super) fn jentry_to_jsonb_item(jentry: JEntry, data: &[u8]) -> JsonbItem<'_> {
    match jentry.type_code {
        NULL_TAG => JsonbItem::Null,
        TRUE_TAG => JsonbItem::Boolean(true),
        FALSE_TAG => JsonbItem::Boolean(false),
        NUMBER_TAG => JsonbItem::Number(NumberItem::Raw(data)),
        STRING_TAG => {
            let s = Cow::Borrowed(unsafe { std::str::from_utf8_unchecked(data) });
            JsonbItem::String(s)
        }
        EXTENSION_TAG => JsonbItem::Extension(ExtensionItem::Raw(data)),
        CONTAINER_TAG => JsonbItem::Raw(RawJsonb::new(data)),
        _ => unreachable!(),
    }
}
