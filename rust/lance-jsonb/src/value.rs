// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-FileCopyrightText: Copyright 2023 Datafuse Labs
// Adapted from databendlabs/jsonb at commit fba895c5ebe77ce2539e187f9c652f51cbf195c3.

use crate::core::JsonbItemType;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;

use crate::extension::ExtensionValue;

use crate::Date;
use crate::Interval;
use crate::Number;
use crate::Timestamp;
use crate::TimestampTz;
use crate::core::Encoder;

pub type Object<'a> = BTreeMap<String, Value<'a>>;

/// Represents a JSON or extended JSON value.
///
/// This enum supports both standard JSON types (Null, Bool, String, Number, Array, Object)
/// and extended types for specialized data representation (Binary, Date, Timestamp, etc.).
/// The extended types provide additional functionality beyond the JSON specification,
/// making this implementation more suitable for database applications and other
/// systems requiring richer data type support.
#[derive(Clone, Default)]
pub enum Value<'a> {
    /// Represents a JSON null value
    #[default]
    Null,
    /// Represents a JSON boolean value (true or false)
    Bool(bool),
    /// Represents a JSON string value
    String(Cow<'a, str>),
    /// Represents a JSON number value with various internal representations
    Number(Number),
    /// Extended type: Represents binary data not supported in standard JSON
    /// Useful for storing raw bytes, images, or other binary content
    Binary(&'a [u8]),
    /// Extended type: Represents a calendar date (year, month, day)
    /// Stored as days since epoch for efficient comparison and manipulation
    Date(Date),
    /// Extended type: Represents a timestamp without timezone information
    /// Stored as microseconds since epoch
    Timestamp(Timestamp),
    /// Extended type: Represents a timestamp with timezone information
    /// Includes both timestamp and timezone offset
    TimestampTz(TimestampTz),
    /// Extended type: Represents a time interval or duration
    /// Useful for time difference calculations and scheduling
    Interval(Interval),
    /// Represents a JSON array of values
    Array(Vec<Self>),
    /// Represents a JSON object as key-value pairs
    Object(Object<'a>),
}

impl Eq for Value<'_> {}

impl PartialEq for Value<'_> {
    fn eq(&self, other: &Self) -> bool {
        let result = self.cmp(other);
        result == Ordering::Equal
    }
}

impl PartialOrd for Value<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Value<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        let self_type = self.jsonb_item_type();
        let other_type = other.jsonb_item_type();

        if let Some(ord) = self_type.partial_cmp(&other_type) {
            return ord;
        }

        match (self, other) {
            (Value::Null, Value::Null) => Ordering::Equal,
            (Value::Bool(v1), Value::Bool(v2)) => v1.cmp(v2),
            (Value::Number(v1), Value::Number(v2)) => v1.cmp(v2),
            (Value::String(v1), Value::String(v2)) => v1.cmp(v2),
            (Value::Array(arr1), Value::Array(arr2)) => {
                for (v1, v2) in arr1.iter().zip(arr2.iter()) {
                    let ord = v1.cmp(v2);
                    if ord != Ordering::Equal {
                        return ord;
                    }
                }
                arr1.len().cmp(&arr2.len())
            }
            (Value::Object(obj1), Value::Object(obj2)) => {
                for ((k1, v1), (k2, v2)) in obj1.iter().zip(obj2.iter()) {
                    let ord = k1.cmp(k2);
                    if ord != Ordering::Equal {
                        return ord;
                    }
                    let ord = v1.cmp(v2);
                    if ord != Ordering::Equal {
                        return ord;
                    }
                }
                obj1.len().cmp(&obj2.len())
            }
            (_, _) => match (self.as_extension_value(), other.as_extension_value()) {
                (Some(self_ext), Some(other_ext)) => {
                    if let Some(ord) = self_ext.partial_cmp(&other_ext) {
                        return ord;
                    }
                    Ordering::Equal
                }
                (_, _) => Ordering::Equal,
            },
        }
    }
}

impl Debug for Value<'_> {
    fn fmt(&self, formatter: &mut Formatter) -> std::fmt::Result {
        match *self {
            Value::Null => formatter.debug_tuple("Null").finish(),
            Value::Bool(v) => formatter.debug_tuple("Bool").field(&v).finish(),
            Value::Number(ref v) => Debug::fmt(v, formatter),
            Value::String(ref v) => formatter.debug_tuple("String").field(v).finish(),
            Value::Binary(ref v) => formatter.debug_tuple("Binary").field(v).finish(),
            Value::Date(ref v) => formatter.debug_tuple("Date").field(v).finish(),
            Value::Timestamp(ref v) => formatter.debug_tuple("Timestamp").field(v).finish(),
            Value::TimestampTz(ref v) => formatter.debug_tuple("TimestampTz").field(v).finish(),
            Value::Interval(ref v) => formatter.debug_tuple("Interval").field(v).finish(),
            Value::Array(ref v) => {
                formatter.write_str("Array(")?;
                Debug::fmt(v, formatter)?;
                formatter.write_str(")")
            }
            Value::Object(ref v) => {
                formatter.write_str("Object(")?;
                Debug::fmt(v, formatter)?;
                formatter.write_str(")")
            }
        }
    }
}

impl Display for Value<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Null => write!(f, "null"),
            Value::Bool(v) => {
                if *v {
                    write!(f, "true")
                } else {
                    write!(f, "false")
                }
            }
            Value::Number(v) => write!(f, "{}", v),
            Value::String(v) => {
                write!(f, "{:?}", v)
            }
            Value::Binary(v) => {
                write!(f, "\"")?;
                for c in *v {
                    write!(f, "{c:02X}")?;
                }
                write!(f, "\"")?;
                Ok(())
            }
            Value::Date(v) => {
                write!(f, "\"{}\"", v)
            }
            Value::Timestamp(v) => {
                write!(f, "\"{}\"", v)
            }
            Value::TimestampTz(v) => {
                write!(f, "\"{}\"", v)
            }
            Value::Interval(v) => {
                write!(f, "\"{}\"", v)
            }
            Value::Array(vs) => {
                write!(f, "[")?;
                for (i, v) in vs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, "]")
            }
            Value::Object(vs) => {
                write!(f, "{{")?;
                for (i, (k, v)) in vs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "\"")?;
                    write!(f, "{k}")?;
                    write!(f, "\"")?;
                    write!(f, ":")?;
                    write!(f, "{v}")?;
                }
                write!(f, "}}")
            }
        }
    }
}

impl<'a> Value<'a> {
    /// Serializes this value into JSONB bytes, appending to `buf`.
    pub(crate) fn write_to_vec(&self, buf: &mut Vec<u8>) {
        let mut encoder = Encoder::new(buf);
        encoder.encode(self);
    }

    /// Serializes this value into JSONB bytes and returns the buffer.
    pub fn to_vec(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.write_to_vec(&mut buf);
        buf
    }

    fn jsonb_item_type(&self) -> JsonbItemType {
        match self {
            Value::Null => JsonbItemType::Null,
            Value::Bool(_) => JsonbItemType::Boolean,
            Value::Number(_) => JsonbItemType::Number,
            Value::String(_) => JsonbItemType::String,
            Value::Binary(_) => JsonbItemType::Extension,
            Value::Date(_) => JsonbItemType::Extension,
            Value::Timestamp(_) => JsonbItemType::Extension,
            Value::TimestampTz(_) => JsonbItemType::Extension,
            Value::Interval(_) => JsonbItemType::Extension,
            Value::Array(arr) => JsonbItemType::Array(arr.len()),
            Value::Object(obj) => JsonbItemType::Object(obj.len()),
        }
    }

    fn as_extension_value(&self) -> Option<ExtensionValue<'_>> {
        match self {
            Value::Binary(v) => Some(ExtensionValue::Binary(v)),
            Value::Date(v) => Some(ExtensionValue::Date(v.clone())),
            Value::Timestamp(v) => Some(ExtensionValue::Timestamp(v.clone())),
            Value::TimestampTz(v) => Some(ExtensionValue::TimestampTz(v.clone())),
            Value::Interval(v) => Some(ExtensionValue::Interval(v.clone())),
            _ => None,
        }
    }
}
