// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-FileCopyrightText: Copyright 2023 Datafuse Labs
// Adapted from databendlabs/jsonb at commit fba895c5ebe77ce2539e187f9c652f51cbf195c3.

use std::borrow::Cow;
use std::collections::VecDeque;
use std::convert::TryFrom;

use num_traits::FromPrimitive;
use serde::de;
use serde::de::Deserialize;
use serde::de::IntoDeserializer;
use serde::de::Visitor;

use super::constants::*;
use super::jentry::JEntry;
use crate::RawJsonb;
use crate::error::Error;
use crate::error::Result;
use crate::extension::ExtensionValue;
use crate::number::Number;

/// `Deserializer` is a custom deserializer for JSONB data, implementing the
/// `serde::de::Deserializer` trait. It allows deserializing a `RawJsonb` into
/// Rust data structures using Serde.
pub struct Deserializer<'de> {
    index: usize,
    current_jentry: Option<JEntry>,
    raw: &'de RawJsonb<'de>,
}

impl<'de> Deserializer<'de> {
    /// Creates a new `Deserializer` from a `RawJsonb`.
    ///
    /// This function initializes the deserializer with the provided `RawJsonb` data.
    ///
    /// # Arguments
    ///
    /// * `raw_jsonb`: A reference to the `RawJsonb` containing the JSONB data to deserialize.
    pub(crate) fn new(raw_jsonb: &'de RawJsonb) -> Self {
        Self {
            index: 0,
            current_jentry: None,
            raw: raw_jsonb,
        }
    }

    /// Checks if the deserializer has reached the end of the `RawJsonb` data.
    ///
    /// This function returns `true` if the current index is equal to the length of
    /// the `RawJsonb` data, indicating that all data has been processed.
    pub(crate) fn end(&self) -> bool {
        self.index == self.raw.len()
    }

    fn read_header(&mut self) -> Result<(u32, usize)> {
        let header = self.raw.read_header(self.index)?;
        self.index += 4;
        Ok(header)
    }

    fn read_jentry(&mut self) -> Result<JEntry> {
        let jentry = self.raw.read_jentry(self.index)?;
        self.index += 4;
        Ok(jentry)
    }

    fn set_jentry(&mut self, jentry: JEntry) {
        self.current_jentry = Some(jentry);
    }

    fn set_jentry_with_index(&mut self, jentry: JEntry, index: usize) {
        self.current_jentry = Some(jentry);
        self.index = index;
    }

    fn read_scalar_jentry(&mut self) -> Result<JEntry> {
        if let Some(jentry) = &self.current_jentry {
            Ok(jentry.clone())
        } else {
            let (header_type, _) = self.read_header()?;
            match header_type {
                SCALAR_CONTAINER_TAG => {
                    let jentry = self.read_jentry()?;
                    Ok(jentry)
                }
                ARRAY_CONTAINER_TAG | OBJECT_CONTAINER_TAG => Err(Error::UnexpectedType),
                _ => Err(Error::InvalidJsonb),
            }
        }
    }

    fn read_array_jentries(&mut self, header_len: usize) -> Result<VecDeque<JEntry>> {
        let mut jentries = VecDeque::with_capacity(header_len);
        for _ in 0..header_len {
            let jentry = self.read_jentry()?;
            jentries.push_back(jentry);
        }
        Ok(jentries)
    }

    fn read_object_jentries(
        &mut self,
        header_len: usize,
    ) -> Result<(VecDeque<JEntry>, VecDeque<JEntry>)> {
        let mut key_jentries = VecDeque::with_capacity(header_len);
        for _ in 0..header_len {
            let jentry = self.read_jentry()?;
            key_jentries.push_back(jentry);
        }
        let mut value_jentries = VecDeque::with_capacity(header_len);
        for _ in 0..header_len {
            let jentry = self.read_jentry()?;
            value_jentries.push_back(jentry);
        }

        Ok((key_jentries, value_jentries))
    }

    fn read_payload_number(&mut self, length: usize) -> Result<Number> {
        let start = self.index;
        let end = self.index + length;
        let num = Number::decode(&self.raw.data[start..end])?;
        self.index = end;
        Ok(num)
    }

    fn read_payload_str(&mut self, length: usize) -> Result<Cow<'_, str>> {
        let start = self.index;
        let end = self.index + length;
        let s = unsafe { std::str::from_utf8_unchecked(&self.raw.data[start..end]) };
        self.index = end;
        Ok(Cow::Borrowed(s))
    }

    fn read_payload_extension(&mut self, length: usize) -> Result<ExtensionValue<'_>> {
        let start = self.index;
        let end = self.index + length;
        let val = ExtensionValue::decode(&self.raw.data[start..end])?;
        self.index = end;
        Ok(val)
    }

    fn read_null(&mut self) -> Result<()> {
        let jentry_res = self.read_scalar_jentry();
        if jentry_res == Err(Error::UnexpectedType) {
            return Err(Error::UnexpectedType);
        }
        let jentry = jentry_res?;
        match jentry.type_code {
            NULL_TAG => Ok(()),
            FALSE_TAG | TRUE_TAG | NUMBER_TAG | STRING_TAG | CONTAINER_TAG | EXTENSION_TAG => {
                Err(Error::UnexpectedType)
            }
            _ => Err(Error::InvalidJsonb),
        }
    }

    fn read_bool(&mut self) -> Result<bool> {
        let jentry_res = self.read_scalar_jentry();
        if jentry_res == Err(Error::UnexpectedType) {
            return Err(Error::UnexpectedType);
        }
        let jentry = jentry_res?;
        match jentry.type_code {
            FALSE_TAG => Ok(false),
            TRUE_TAG => Ok(true),
            NULL_TAG | NUMBER_TAG | STRING_TAG | CONTAINER_TAG | EXTENSION_TAG => {
                Err(Error::UnexpectedType)
            }
            _ => Err(Error::InvalidJsonb),
        }
    }

    fn read_number(&mut self) -> Result<Number> {
        let jentry_res = self.read_scalar_jentry();
        if jentry_res == Err(Error::UnexpectedType) {
            return Err(Error::UnexpectedType);
        }
        let jentry = jentry_res?;
        match jentry.type_code {
            NUMBER_TAG => {
                let length = jentry.length as usize;
                let num = self.read_payload_number(length)?;
                Ok(num)
            }
            NULL_TAG | FALSE_TAG | TRUE_TAG | STRING_TAG | CONTAINER_TAG | EXTENSION_TAG => {
                Err(Error::UnexpectedType)
            }
            _ => Err(Error::InvalidJsonb),
        }
    }

    fn read_integer<T>(&mut self) -> Result<T>
    where
        for<'a> T: Deserialize<'a> + FromPrimitive,
    {
        let num = self.read_number()?;
        match num {
            Number::Int64(n) => T::from_i64(n).ok_or(Error::UnexpectedType),
            Number::UInt64(n) => T::from_u64(n).ok_or(Error::UnexpectedType),
            Number::Float64(_)
            | Number::Decimal64(_)
            | Number::Decimal128(_)
            | Number::Decimal256(_) => Err(Error::UnexpectedType),
        }
    }

    fn read_float<T>(&mut self) -> Result<T>
    where
        for<'a> T: Deserialize<'a> + FromPrimitive,
    {
        let num = self.read_number()?;
        match num {
            Number::Int64(n) => T::from_i64(n).ok_or(Error::UnexpectedType),
            Number::UInt64(n) => T::from_u64(n).ok_or(Error::UnexpectedType),
            Number::Float64(n) => T::from_f64(n).ok_or(Error::UnexpectedType),
            Number::Decimal64(v) => {
                let n = v.to_float64();
                T::from_f64(n).ok_or(Error::UnexpectedType)
            }
            Number::Decimal128(v) => {
                let n = v.to_float64();
                T::from_f64(n).ok_or(Error::UnexpectedType)
            }
            Number::Decimal256(v) => {
                let n = v.to_float64();
                T::from_f64(n).ok_or(Error::UnexpectedType)
            }
        }
    }

    fn read_str(&mut self) -> Result<Cow<'_, str>> {
        let jentry_res = self.read_scalar_jentry();
        if jentry_res == Err(Error::UnexpectedType) {
            return Err(Error::UnexpectedType);
        }
        let jentry = jentry_res?;
        match jentry.type_code {
            STRING_TAG => {
                let length = jentry.length as usize;
                let s = self.read_payload_str(length)?;
                Ok(s)
            }
            EXTENSION_TAG => {
                let length = jentry.length as usize;
                let val = self.read_payload_extension(length)?;
                let s = format!("{}", val);
                Ok(Cow::Owned(s))
            }
            NULL_TAG | FALSE_TAG | TRUE_TAG | NUMBER_TAG | CONTAINER_TAG => {
                Err(Error::UnexpectedType)
            }
            _ => Err(Error::InvalidJsonb),
        }
    }

    fn read_string(&mut self) -> Result<String> {
        let s = self.read_str()?;
        Ok(s.to_string())
    }

    fn read_with_jentry<V>(&mut self, jentry: JEntry, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        match jentry.type_code {
            NULL_TAG => visitor.visit_unit(),
            TRUE_TAG => visitor.visit_bool(true),
            FALSE_TAG => visitor.visit_bool(false),
            STRING_TAG => {
                let length = jentry.length as usize;
                let s = self.read_payload_str(length)?;
                visitor.visit_string(s.to_string())
            }
            NUMBER_TAG => {
                let length = jentry.length as usize;
                let num = self.read_payload_number(length)?;
                match num {
                    Number::Int64(i) => {
                        if let Ok(x) = u8::try_from(i) {
                            visitor.visit_u8(x)
                        } else if let Ok(x) = i8::try_from(i) {
                            visitor.visit_i8(x)
                        } else if let Ok(x) = u16::try_from(i) {
                            visitor.visit_u16(x)
                        } else if let Ok(x) = i16::try_from(i) {
                            visitor.visit_i16(x)
                        } else if let Ok(x) = u32::try_from(i) {
                            visitor.visit_u32(x)
                        } else if let Ok(x) = i32::try_from(i) {
                            visitor.visit_i32(x)
                        } else if let Ok(x) = u64::try_from(i) {
                            visitor.visit_u64(x)
                        } else {
                            visitor.visit_i64(i)
                        }
                    }
                    Number::UInt64(i) => {
                        if let Ok(x) = u8::try_from(i) {
                            visitor.visit_u8(x)
                        } else if let Ok(x) = u16::try_from(i) {
                            visitor.visit_u16(x)
                        } else if let Ok(x) = u32::try_from(i) {
                            visitor.visit_u32(x)
                        } else {
                            visitor.visit_u64(i)
                        }
                    }
                    Number::Float64(i) => visitor.visit_f64(i),
                    Number::Decimal64(i) => {
                        let v = i.to_float64();
                        visitor.visit_f64(v)
                    }
                    Number::Decimal128(i) => {
                        let v = i.to_float64();
                        visitor.visit_f64(v)
                    }
                    Number::Decimal256(i) => {
                        let v = i.to_float64();
                        visitor.visit_f64(v)
                    }
                }
            }
            EXTENSION_TAG => {
                let length = jentry.length as usize;
                let val = self.read_payload_extension(length)?;
                let s = format!("{}", val);
                visitor.visit_string(s)
            }
            CONTAINER_TAG => Err(Error::UnexpectedType),
            _ => Err(Error::InvalidJsonb),
        }
    }
}

impl<'de> de::Deserializer<'de> for &mut Deserializer<'de> {
    type Error = Error;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        if let Some(jentry) = &self.current_jentry {
            match jentry.type_code {
                CONTAINER_TAG => {
                    self.current_jentry = None;
                    self.deserialize_any(visitor)
                }
                _ => self.read_with_jentry(jentry.clone(), visitor),
            }
        } else {
            let (header_type, header_len) = self.read_header()?;
            match header_type {
                SCALAR_CONTAINER_TAG => {
                    let jentry = self.read_jentry()?;
                    match jentry.type_code {
                        CONTAINER_TAG => {
                            self.current_jentry = None;
                            self.deserialize_any(visitor)
                        }
                        _ => self.read_with_jentry(jentry, visitor),
                    }
                }
                ARRAY_CONTAINER_TAG => {
                    let jentries = self.read_array_jentries(header_len)?;
                    visitor.visit_seq(ArrayDeserializer::new(self, jentries))
                }
                OBJECT_CONTAINER_TAG => {
                    let (key_jentries, value_jentries) = self.read_object_jentries(header_len)?;

                    let origin_index = self.index;
                    let key_length: usize = key_jentries.iter().map(|j| j.length as usize).sum();
                    let value_length: usize =
                        value_jentries.iter().map(|j| j.length as usize).sum();
                    let key_payload_offset = self.index;
                    let value_payload_offset = self.index + key_length;

                    let value = visitor.visit_map(ObjectDeserializer::new(
                        self,
                        key_payload_offset,
                        key_jentries,
                        value_payload_offset,
                        value_jentries,
                    ))?;
                    self.index = origin_index + key_length + value_length;
                    Ok(value)
                }
                _ => Err(Error::InvalidJsonb),
            }
        }
    }

    fn deserialize_bool<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_bool(self.read_bool()?)
    }

    fn deserialize_i8<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i8(self.read_integer()?)
    }

    fn deserialize_i16<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i16(self.read_integer()?)
    }

    fn deserialize_i32<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i32(self.read_integer()?)
    }

    fn deserialize_i64<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i64(self.read_integer()?)
    }

    fn deserialize_u8<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_u8(self.read_integer()?)
    }

    fn deserialize_u16<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_u16(self.read_integer()?)
    }

    fn deserialize_u32<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_u32(self.read_integer()?)
    }

    fn deserialize_u64<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_u64(self.read_integer()?)
    }

    fn deserialize_f32<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_f32(self.read_float()?)
    }

    fn deserialize_f64<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_f64(self.read_float()?)
    }

    fn deserialize_char<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        let s = self.read_string()?;
        if s.len() != 1 {
            return Err(Error::Message("invalid string length for char".into()));
        }
        visitor.visit_char(s.chars().next().unwrap())
    }

    fn deserialize_str<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_str(&self.read_str()?)
    }

    fn deserialize_string<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_string(self.read_string()?)
    }

    fn deserialize_bytes<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_string(self.read_string()?)
    }

    fn deserialize_byte_buf<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_string(self.read_string()?)
    }

    fn deserialize_option<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        let origin_index = self.index;
        if self.read_null().is_ok() {
            visitor.visit_none()
        } else {
            self.index = origin_index;
            visitor.visit_some(self)
        }
    }

    fn deserialize_unit<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.read_null()?;
        visitor.visit_unit()
    }

    fn deserialize_unit_struct<V>(self, _name: &'static str, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.deserialize_unit(visitor)
    }

    fn deserialize_newtype_struct<V>(self, _name: &'static str, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_seq<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        let (header_type, header_len) = self.read_header()?;
        match header_type {
            ARRAY_CONTAINER_TAG => {
                let jentries = self.read_array_jentries(header_len)?;
                visitor.visit_seq(ArrayDeserializer::new(self, jentries))
            }
            SCALAR_CONTAINER_TAG | OBJECT_CONTAINER_TAG => Err(Error::UnexpectedType),
            _ => Err(Error::InvalidJsonb),
        }
    }

    fn deserialize_tuple<V>(self, _len: usize, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.deserialize_seq(visitor)
    }

    fn deserialize_tuple_struct<V>(
        self,
        _name: &'static str,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.deserialize_seq(visitor)
    }

    fn deserialize_map<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        let (header_type, header_len) = self.read_header()?;
        match header_type {
            OBJECT_CONTAINER_TAG => {
                let (key_jentries, value_jentries) = self.read_object_jentries(header_len)?;

                let origin_index = self.index;
                let key_length: usize = key_jentries.iter().map(|j| j.length as usize).sum();
                let value_length: usize = value_jentries.iter().map(|j| j.length as usize).sum();
                let key_payload_offset = self.index;
                let value_payload_offset = self.index + key_length;

                let value = visitor.visit_map(ObjectDeserializer::new(
                    self,
                    key_payload_offset,
                    key_jentries,
                    value_payload_offset,
                    value_jentries,
                ))?;
                self.index = origin_index + key_length + value_length;
                Ok(value)
            }
            SCALAR_CONTAINER_TAG | ARRAY_CONTAINER_TAG => Err(Error::UnexpectedType),
            _ => Err(Error::InvalidJsonb),
        }
    }

    fn deserialize_struct<V>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.deserialize_map(visitor)
    }

    fn deserialize_enum<V>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        let (header_type, header_len) = self.read_header()?;
        match header_type {
            OBJECT_CONTAINER_TAG => {
                if header_len != 1 {
                    return Err(Error::UnexpectedType);
                }
                let key_jentry = self.read_jentry()?;
                let value_jentry = self.read_jentry()?;

                let origin_index = self.index;

                let key_length = key_jentry.length as usize;
                let value_length = value_jentry.length as usize;

                self.set_jentry(key_jentry);
                let key = self.read_string()?;
                let value_payload_offset = self.index;
                let value = visitor.visit_enum(EnumDeserializer::new(
                    self,
                    key,
                    value_jentry,
                    value_payload_offset,
                ))?;
                self.index = origin_index + key_length + value_length;
                Ok(value)
            }
            SCALAR_CONTAINER_TAG | ARRAY_CONTAINER_TAG => Err(Error::UnexpectedType),
            _ => Err(Error::InvalidJsonb),
        }
    }

    fn deserialize_identifier<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.deserialize_str(visitor)
    }

    fn deserialize_ignored_any<V>(self, _visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        todo!()
    }

    #[inline]
    fn is_human_readable(&self) -> bool {
        false
    }
}

struct ArrayDeserializer<'a, 'de: 'a> {
    size: usize,
    de: &'a mut Deserializer<'de>,
    jentries: VecDeque<JEntry>,
}

impl<'a, 'de> ArrayDeserializer<'a, 'de> {
    fn new(de: &'a mut Deserializer<'de>, jentries: VecDeque<JEntry>) -> Self {
        let size = jentries.len();
        ArrayDeserializer { size, de, jentries }
    }
}

impl<'de> de::SeqAccess<'de> for ArrayDeserializer<'_, 'de> {
    type Error = Error;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>>
    where
        T: de::DeserializeSeed<'de>,
    {
        if let Some(jentry) = self.jentries.pop_front() {
            self.de.set_jentry(jentry);
            // Deserialize an array element.
            seed.deserialize(&mut *self.de).map(Some)
        } else {
            Ok(None)
        }
    }

    #[inline]
    fn size_hint(&self) -> Option<usize> {
        Some(self.size)
    }
}

struct ObjectDeserializer<'a, 'de: 'a> {
    size: usize,
    de: &'a mut Deserializer<'de>,
    key_jentries: VecDeque<JEntry>,
    key_payload_offset: usize,
    value_jentries: VecDeque<JEntry>,
    value_payload_offset: usize,
}

impl<'a, 'de> ObjectDeserializer<'a, 'de> {
    fn new(
        de: &'a mut Deserializer<'de>,
        key_payload_offset: usize,
        key_jentries: VecDeque<JEntry>,
        value_payload_offset: usize,
        value_jentries: VecDeque<JEntry>,
    ) -> Self {
        let size = key_jentries.len();
        ObjectDeserializer {
            size,
            de,
            key_payload_offset,
            key_jentries,
            value_payload_offset,
            value_jentries,
        }
    }
}

impl<'de> de::MapAccess<'de> for ObjectDeserializer<'_, 'de> {
    type Error = Error;

    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>>
    where
        K: de::DeserializeSeed<'de>,
    {
        if let Some(jentry) = self.key_jentries.pop_front() {
            let key_payload_offset = self.key_payload_offset;
            self.key_payload_offset += jentry.length as usize;
            self.de.set_jentry_with_index(jentry, key_payload_offset);
            // Deserialize an object key.
            seed.deserialize(&mut *self.de).map(Some)
        } else {
            Ok(None)
        }
    }

    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value>
    where
        V: de::DeserializeSeed<'de>,
    {
        if let Some(jentry) = self.value_jentries.pop_front() {
            let value_payload_offset = self.value_payload_offset;
            self.value_payload_offset += jentry.length as usize;
            self.de.set_jentry_with_index(jentry, value_payload_offset);
            // Deserialize an object value.
            seed.deserialize(&mut *self.de)
        } else {
            Err(Error::InvalidJsonb)
        }
    }

    #[inline]
    fn size_hint(&self) -> Option<usize> {
        Some(self.size)
    }
}

struct EnumDeserializer<'a, 'de: 'a> {
    de: &'a mut Deserializer<'de>,
    key: String,
    jentry: JEntry,
    payload_offset: usize,
}

impl<'a, 'de> EnumDeserializer<'a, 'de> {
    fn new(
        de: &'a mut Deserializer<'de>,
        key: String,
        jentry: JEntry,
        payload_offset: usize,
    ) -> Self {
        EnumDeserializer {
            de,
            key,
            jentry,
            payload_offset,
        }
    }
}

impl<'de> de::EnumAccess<'de> for EnumDeserializer<'_, 'de> {
    type Error = Error;
    type Variant = Self;

    fn variant_seed<V>(self, seed: V) -> Result<(V::Value, Self::Variant)>
    where
        V: de::DeserializeSeed<'de>,
    {
        let key_variant = self.key.clone().into_deserializer();
        seed.deserialize(key_variant).map(|v| (v, self))
    }
}

impl<'de> de::VariantAccess<'de> for EnumDeserializer<'_, 'de> {
    type Error = Error;

    fn unit_variant(self) -> Result<()> {
        if self.jentry.type_code == NULL_TAG {
            Ok(())
        } else {
            Err(Error::UnexpectedType)
        }
    }

    fn newtype_variant_seed<T>(self, seed: T) -> Result<T::Value>
    where
        T: de::DeserializeSeed<'de>,
    {
        self.de
            .set_jentry_with_index(self.jentry, self.payload_offset);
        seed.deserialize(&mut *self.de)
    }

    fn tuple_variant<V>(self, _len: usize, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        if self.jentry.type_code == CONTAINER_TAG {
            let (header_type, header_len) = self.de.read_header()?;
            match header_type {
                ARRAY_CONTAINER_TAG => {
                    let jentries = self.de.read_array_jentries(header_len)?;
                    visitor.visit_seq(ArrayDeserializer::new(self.de, jentries))
                }
                SCALAR_CONTAINER_TAG | OBJECT_CONTAINER_TAG => Err(Error::UnexpectedType),
                _ => Err(Error::InvalidJsonb),
            }
        } else {
            Err(Error::UnexpectedType)
        }
    }

    fn struct_variant<V>(self, _fields: &'static [&'static str], visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.de
            .set_jentry_with_index(self.jentry.clone(), self.payload_offset);
        if self.jentry.type_code == CONTAINER_TAG {
            let (header_type, header_len) = self.de.read_header()?;
            match header_type {
                OBJECT_CONTAINER_TAG => {
                    let (key_jentries, value_jentries) =
                        self.de.read_object_jentries(header_len)?;

                    let origin_index = self.de.index;
                    let key_length: usize = key_jentries.iter().map(|j| j.length as usize).sum();
                    let value_length: usize =
                        value_jentries.iter().map(|j| j.length as usize).sum();
                    let key_payload_offset = self.de.index;
                    let value_payload_offset = self.de.index + key_length;

                    let value = visitor.visit_map(ObjectDeserializer::new(
                        self.de,
                        key_payload_offset,
                        key_jentries,
                        value_payload_offset,
                        value_jentries,
                    ))?;
                    self.de.index = origin_index + key_length + value_length;
                    Ok(value)
                }
                SCALAR_CONTAINER_TAG | ARRAY_CONTAINER_TAG => Err(Error::UnexpectedType),
                _ => Err(Error::UnexpectedType),
            }
        } else {
            Err(Error::UnexpectedType)
        }
    }
}
