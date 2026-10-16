// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The semantic type model behind `logical_type`.
//!
//! In a table that follows the semantic type contract, `logical_type` names a
//! [`SemanticType`]: the value domain and query semantics of a column. The
//! Arrow layout that reads return is an [`OutputEncoding`], recorded in the
//! field metadata entry [`OUTPUT_ENCODING_META_KEY`] when it differs from the
//! type's default. Legacy tables and data file schemas instead name one exact
//! Arrow type per `logical_type`; each of those strings is an alias for a
//! semantic type plus an implied output encoding, which
//! [`LogicalType::semantic`] recovers.

use std::fmt::{self, Display, Formatter};
use std::str::FromStr;
use std::sync::Arc;

use arrow_schema::{DataType, Field as ArrowField};

use super::LogicalType;
use crate::{Error, Result};

/// Field metadata entry naming the Arrow layout that reads return for a field
/// of a representation-only or value-transforming semantic type.
pub const OUTPUT_ENCODING_META_KEY: &str = "lance-schema:output-encoding";

const MAX_DECIMAL128_PRECISION: u8 = 38;
const MAX_DECIMAL256_PRECISION: u8 = 76;

/// The value domain and query semantics a `logical_type` names, independent of
/// the Arrow layout that holds the values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticType {
    /// UTF-8 strings, stored as `Utf8`, `LargeUtf8`, or a dictionary of them.
    String,
    /// Byte strings, stored as `Binary`, `LargeBinary`, or a dictionary of them.
    Binary,
    /// Variable-length lists. The child field describes the elements.
    List,
    /// Decimals with a fixed precision and scale, stored as `Decimal128` or
    /// `Decimal256`.
    Decimal { precision: u8, scale: i8 },
    /// JSON values, stored as JSONB.
    Json,
    /// Blobs, whose representations are defined by the blob format.
    Blob,
    /// A type with exactly one Arrow representation, named by its
    /// `logical_type` string. Nested types in this class (`struct`, `map`,
    /// fixed-size lists of structs) describe their children with child fields.
    Unchanged(LogicalType),
}

/// How a semantic type relates its input, storage, and output representations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticTypeClass {
    /// One Arrow representation for input, storage, and output.
    Unchanged,
    /// Several Arrow layouts hold the same values; writers store any of them
    /// unchanged and readers may return any of them.
    RepresentationOnly,
    /// Input values are converted to a different stored representation, and
    /// reads convert them back.
    ValueTransforming,
}

impl SemanticType {
    /// The class this type belongs to.
    pub fn class(&self) -> SemanticTypeClass {
        match self {
            Self::String | Self::Binary | Self::List | Self::Decimal { .. } => {
                SemanticTypeClass::RepresentationOnly
            }
            Self::Json | Self::Blob => SemanticTypeClass::ValueTransforming,
            Self::Unchanged(_) => SemanticTypeClass::Unchanged,
        }
    }

    /// The canonical `logical_type` string of this type.
    pub fn logical_type(&self) -> LogicalType {
        match self {
            Self::String => LogicalType::from("string"),
            Self::Binary => LogicalType::from("binary"),
            Self::List => LogicalType::from("list"),
            Self::Decimal { precision, scale } => {
                LogicalType(format!("decimal:{precision}:{scale}"))
            }
            Self::Json => LogicalType::from("json"),
            Self::Blob => LogicalType::from(super::BLOB_LOGICAL_TYPE),
            Self::Unchanged(logical_type) => logical_type.clone(),
        }
    }

    /// The output encoding reads use when neither the caller nor the field
    /// names one. `None` for types whose output is not selected by an encoding.
    pub fn default_output_encoding(&self) -> Option<OutputEncoding> {
        match self {
            Self::String => Some(OutputEncoding::Utf8),
            Self::Binary => Some(OutputEncoding::Binary),
            Self::List => Some(OutputEncoding::List),
            Self::Decimal { precision, .. } if *precision <= MAX_DECIMAL128_PRECISION => {
                Some(OutputEncoding::Decimal128)
            }
            Self::Decimal { .. } => Some(OutputEncoding::Decimal256),
            Self::Json => Some(OutputEncoding::ArrowJson),
            Self::Blob | Self::Unchanged(_) => None,
        }
    }

    /// Check that `encoding` is one of this type's output encodings.
    pub fn check_output_encoding(&self, encoding: &OutputEncoding) -> Result<()> {
        use OutputEncoding::*;
        let valid = match (self, encoding) {
            (Self::String, Utf8 | LargeUtf8 | Utf8View)
            | (Self::Binary, Binary | LargeBinary | BinaryView)
            | (Self::List, List | LargeList)
            | (Self::Decimal { .. }, Decimal256)
            | (Self::Json, ArrowJson | LanceJson) => true,
            (Self::Decimal { precision, .. }, Decimal128) => *precision <= MAX_DECIMAL128_PRECISION,
            (Self::String, Dictionary { value, .. }) => matches!(**value, Utf8 | LargeUtf8),
            (Self::Binary, Dictionary { value, .. }) => matches!(**value, Binary | LargeBinary),
            _ => false,
        };
        if valid {
            Ok(())
        } else {
            Err(Error::schema(format!(
                "output encoding '{encoding}' is not valid for semantic type '{}'",
                self.logical_type()
            )))
        }
    }

    /// The Arrow type holding this type's values in the layout `encoding`
    /// names. `item` is the child field of a list.
    ///
    /// Only representation-only types have a layout selected by an output
    /// encoding; the output encodings of value-transforming types select a
    /// conversion instead.
    pub fn layout_data_type(
        &self,
        encoding: &OutputEncoding,
        item: Option<&ArrowField>,
    ) -> Result<DataType> {
        self.check_output_encoding(encoding)?;
        let item = || {
            item.cloned().map(Arc::new).ok_or_else(|| {
                Error::schema(format!(
                    "a '{}' layout needs its item field",
                    self.logical_type()
                ))
            })
        };
        match (self, encoding) {
            (Self::List, OutputEncoding::List) => Ok(DataType::List(item()?)),
            (Self::List, OutputEncoding::LargeList) => Ok(DataType::LargeList(item()?)),
            (Self::Decimal { precision, scale }, OutputEncoding::Decimal128) => {
                Ok(DataType::Decimal128(*precision, *scale))
            }
            (Self::Decimal { precision, scale }, OutputEncoding::Decimal256) => {
                Ok(DataType::Decimal256(*precision, *scale))
            }
            (Self::String | Self::Binary, encoding) => Ok(encoding
                .byte_layout()
                .expect("string and binary encodings are byte layouts")),
            _ => Err(Error::schema(format!(
                "output encoding '{encoding}' of semantic type '{}' does not select an Arrow layout",
                self.logical_type()
            ))),
        }
    }
}

/// An Arrow layout that reads return for a field, as named by the
/// [`OUTPUT_ENCODING_META_KEY`] field metadata entry.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OutputEncoding {
    Utf8,
    LargeUtf8,
    Utf8View,
    Binary,
    LargeBinary,
    BinaryView,
    /// A dictionary with integer `key` indices over `value`, which is one of
    /// `Utf8`, `LargeUtf8`, `Binary`, or `LargeBinary`.
    Dictionary {
        key: DataType,
        value: Box<Self>,
    },
    List,
    LargeList,
    Decimal128,
    Decimal256,
    /// JSON text with the `arrow.json` extension.
    ArrowJson,
    /// JSONB with the `lance.json` extension.
    LanceJson,
}

impl OutputEncoding {
    /// The layout of an Arrow type that holds representation-only values, if
    /// it has one.
    ///
    /// Types of other semantic types, and dictionaries whose values are not
    /// strings or byte strings, have no layout.
    pub fn of_data_type(data_type: &DataType) -> Option<Self> {
        match data_type {
            DataType::Utf8 => Some(Self::Utf8),
            DataType::LargeUtf8 => Some(Self::LargeUtf8),
            DataType::Utf8View => Some(Self::Utf8View),
            DataType::Binary => Some(Self::Binary),
            DataType::LargeBinary => Some(Self::LargeBinary),
            DataType::BinaryView => Some(Self::BinaryView),
            DataType::List(_) => Some(Self::List),
            DataType::LargeList(_) => Some(Self::LargeList),
            DataType::Decimal128(..) => Some(Self::Decimal128),
            DataType::Decimal256(..) => Some(Self::Decimal256),
            DataType::Dictionary(key, value) if key.is_dictionary_key_type() => {
                let value = match value.as_ref() {
                    DataType::Utf8 => Self::Utf8,
                    DataType::LargeUtf8 => Self::LargeUtf8,
                    DataType::Binary => Self::Binary,
                    DataType::LargeBinary => Self::LargeBinary,
                    _ => return None,
                };
                Some(Self::Dictionary {
                    key: key.as_ref().clone(),
                    value: Box::new(value),
                })
            }
            _ => None,
        }
    }

    /// The Arrow type of a string or byte string layout.
    fn byte_layout(&self) -> Option<DataType> {
        match self {
            Self::Utf8 => Some(DataType::Utf8),
            Self::LargeUtf8 => Some(DataType::LargeUtf8),
            Self::Utf8View => Some(DataType::Utf8View),
            Self::Binary => Some(DataType::Binary),
            Self::LargeBinary => Some(DataType::LargeBinary),
            Self::BinaryView => Some(DataType::BinaryView),
            Self::Dictionary { key, value } => Some(DataType::Dictionary(
                Box::new(key.clone()),
                Box::new(value.byte_layout()?),
            )),
            _ => None,
        }
    }
}

impl Display for OutputEncoding {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Utf8 => write!(f, "utf8"),
            Self::LargeUtf8 => write!(f, "large_utf8"),
            Self::Utf8View => write!(f, "utf8_view"),
            Self::Binary => write!(f, "binary"),
            Self::LargeBinary => write!(f, "large_binary"),
            Self::BinaryView => write!(f, "binary_view"),
            Self::Dictionary { key, value } => {
                // Keys are always integer types, which have a logical type name.
                let key = LogicalType::try_from(key).map_err(|_| fmt::Error)?;
                write!(f, "dictionary:{key}:{value}")
            }
            Self::List => write!(f, "list"),
            Self::LargeList => write!(f, "large_list"),
            Self::Decimal128 => write!(f, "decimal128"),
            Self::Decimal256 => write!(f, "decimal256"),
            Self::ArrowJson => write!(f, "arrow.json"),
            Self::LanceJson => write!(f, "lance.json"),
        }
    }
}

impl FromStr for OutputEncoding {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        let encoding = match s {
            "utf8" => Self::Utf8,
            "large_utf8" => Self::LargeUtf8,
            "utf8_view" => Self::Utf8View,
            "binary" => Self::Binary,
            "large_binary" => Self::LargeBinary,
            "binary_view" => Self::BinaryView,
            "list" => Self::List,
            "large_list" => Self::LargeList,
            "decimal128" => Self::Decimal128,
            "decimal256" => Self::Decimal256,
            "arrow.json" => Self::ArrowJson,
            "lance.json" => Self::LanceJson,
            _ => {
                let parts = s.split(':').collect::<Vec<_>>();
                let [_, key, value] = parts.as_slice() else {
                    return Err(unknown_output_encoding(s));
                };
                if parts[0] != "dictionary" {
                    return Err(unknown_output_encoding(s));
                }
                let key = DataType::try_from(&LogicalType::from(*key))
                    .ok()
                    .filter(DataType::is_dictionary_key_type)
                    .ok_or_else(|| unknown_output_encoding(s))?;
                let value = match *value {
                    "utf8" => Self::Utf8,
                    "large_utf8" => Self::LargeUtf8,
                    "binary" => Self::Binary,
                    "large_binary" => Self::LargeBinary,
                    _ => return Err(unknown_output_encoding(s)),
                };
                Self::Dictionary {
                    key,
                    value: Box::new(value),
                }
            }
        };
        Ok(encoding)
    }
}

fn unknown_output_encoding(value: &str) -> Error {
    Error::schema(format!("unknown output encoding '{value}'"))
}

/// A `logical_type` string read as a semantic type, together with the output
/// encoding a legacy alias implies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticLogicalType {
    pub semantic_type: SemanticType,
    /// The layout the string names when it is not the default of
    /// `semantic_type`, such as `large_utf8` for `large_string`.
    pub implied_encoding: Option<OutputEncoding>,
}

impl SemanticLogicalType {
    fn new(semantic_type: SemanticType, layout: Option<OutputEncoding>) -> Result<Self> {
        if let Some(layout) = &layout {
            semantic_type.check_output_encoding(layout)?;
        }
        let implied_encoding = layout
            .filter(|layout| semantic_type.default_output_encoding().as_ref() != Some(layout));
        Ok(Self {
            semantic_type,
            implied_encoding,
        })
    }

    /// The layout this string names: the implied encoding, or the default.
    pub fn output_encoding(&self) -> Option<OutputEncoding> {
        self.implied_encoding
            .clone()
            .or_else(|| self.semantic_type.default_output_encoding())
    }
}

impl LogicalType {
    /// Read this string as a semantic type.
    ///
    /// Accepts canonical names and the legacy aliases that name one Arrow
    /// layout of a representation-only type. A dictionary whose values are not
    /// strings or byte strings has no semantic type and is rejected.
    pub fn semantic(&self) -> Result<SemanticLogicalType> {
        use OutputEncoding as E;
        use SemanticType as T;
        let (semantic_type, layout) = match self.0.as_str() {
            "string" => (T::String, None),
            "large_string" => (T::String, Some(E::LargeUtf8)),
            "binary" => (T::Binary, None),
            "large_binary" => (T::Binary, Some(E::LargeBinary)),
            "list" | "list.struct" => (T::List, None),
            "large_list" | "large_list.struct" => (T::List, Some(E::LargeList)),
            "json" => (T::Json, None),
            super::BLOB_LOGICAL_TYPE => (T::Blob, None),
            name if name.starts_with("decimal:") => self.semantic_decimal()?,
            name if name.starts_with("dict:") => self.semantic_dictionary()?,
            // Nested types take their Arrow type from child fields.
            _ if self.is_struct() || self.is_map() || self.is_fixed_size_list_struct() => {
                (T::Unchanged(self.clone()), None)
            }
            _ => {
                // Parse to reject strings that name no type at all.
                DataType::try_from(self)?;
                (T::Unchanged(self.clone()), None)
            }
        };
        SemanticLogicalType::new(semantic_type, layout)
    }

    fn semantic_decimal(&self) -> Result<(SemanticType, Option<OutputEncoding>)> {
        let parts = self.0.split(':').collect::<Vec<_>>();
        let (width, precision, scale) = match parts.as_slice() {
            ["decimal", precision, scale] => (None, precision, scale),
            ["decimal", width, precision, scale] => (Some(*width), precision, scale),
            _ => {
                return Err(Error::schema(format!("Unsupported decimal type: {self}")));
            }
        };
        let precision = precision
            .parse::<u8>()
            .map_err(|err| Error::schema(format!("invalid decimal type {self}: {err}")))?;
        let scale = scale
            .parse::<i8>()
            .map_err(|err| Error::schema(format!("invalid decimal type {self}: {err}")))?;
        if precision == 0
            || precision > MAX_DECIMAL256_PRECISION
            || i16::from(scale) > i16::from(precision)
        {
            return Err(Error::schema(format!(
                "invalid decimal type {self}: precision must be between 1 and {MAX_DECIMAL256_PRECISION} and scale at most the precision"
            )));
        }
        let layout = match width {
            None => None,
            Some("128") => Some(OutputEncoding::Decimal128),
            Some("256") => Some(OutputEncoding::Decimal256),
            Some(width) => {
                return Err(Error::schema(format!(
                    "Only Decimal128 and Decimal256 is supported. Found {width}"
                )));
            }
        };
        Ok((SemanticType::Decimal { precision, scale }, layout))
    }

    fn semantic_dictionary(&self) -> Result<(SemanticType, Option<OutputEncoding>)> {
        let DataType::Dictionary(key, value) = DataType::try_from(self)? else {
            unreachable!("a dict: logical type parses to a dictionary");
        };
        let layout = OutputEncoding::of_data_type(&DataType::Dictionary(key, value.clone()))
            .ok_or_else(|| {
                Error::schema(format!(
                    "dictionary type {self} has no semantic type: dictionary encoding is a layout of string and binary values only"
                ))
            })?;
        let semantic_type = match value.as_ref() {
            DataType::Utf8 | DataType::LargeUtf8 => SemanticType::String,
            _ => SemanticType::Binary,
        };
        Ok((semantic_type, Some(layout)))
    }
}

/// How schema compatibility compares the types of two fields.
///
/// [`Field::type_matches`](super::Field::type_matches) applies it; it is the
/// single entry point for deciding whether two field types are compatible for
/// append, merge, and update.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum TypeComparison {
    /// The rule of legacy tables: each `logical_type` names one Arrow type, so
    /// types match only when the strings are equal.
    #[default]
    Exact,
    /// The rule of tables that follow the semantic type contract: types match
    /// when their semantic types and semantic parameters are equal. Output
    /// encodings, physical layouts, and dictionary key types are ignored.
    Semantic,
}

impl TypeComparison {
    pub(super) fn logical_types_match(self, actual: &LogicalType, expected: &LogicalType) -> bool {
        if actual == expected {
            return true;
        }
        match self {
            Self::Exact => false,
            Self::Semantic => match (actual.semantic(), expected.semantic()) {
                (Ok(actual), Ok(expected)) => actual.semantic_type == expected.semantic_type,
                // A string that names no semantic type only matches itself.
                _ => false,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    /// Every legacy alias reads as its canonical type plus the layout it names,
    /// and that layout maps back to the same alias.
    #[rstest]
    #[case::string("string", "string", None)]
    #[case::large_string("large_string", "string", Some("large_utf8"))]
    #[case::binary("binary", "binary", None)]
    #[case::large_binary("large_binary", "binary", Some("large_binary"))]
    #[case::list("list", "list", None)]
    #[case::list_struct("list.struct", "list", None)]
    #[case::large_list("large_list", "list", Some("large_list"))]
    #[case::large_list_struct("large_list.struct", "list", Some("large_list"))]
    #[case::dict_string("dict:string:int16:false", "string", Some("dictionary:int16:utf8"))]
    #[case::dict_large_string(
        "dict:large_string:int8:false",
        "string",
        Some("dictionary:int8:large_utf8")
    )]
    #[case::dict_binary("dict:binary:uint32:false", "binary", Some("dictionary:uint32:binary"))]
    #[case::decimal128("decimal:128:10:2", "decimal:10:2", None)]
    #[case::decimal256_narrow("decimal:256:10:2", "decimal:10:2", Some("decimal256"))]
    #[case::decimal256_wide("decimal:256:40:2", "decimal:40:2", None)]
    #[case::canonical_decimal("decimal:10:2", "decimal:10:2", None)]
    #[case::canonical_wide_decimal("decimal:60:-3", "decimal:60:-3", None)]
    #[case::json("json", "json", None)]
    #[case::blob("blob", "blob", None)]
    #[case::int64("int64", "int64", None)]
    #[case::timestamp("timestamp:us:+08:00", "timestamp:us:+08:00", None)]
    #[case::fixed_size_list("fixed_size_list:float:4", "fixed_size_list:float:4", None)]
    #[case::bfloat16(
        "fixed_size_list:lance.bfloat16:8",
        "fixed_size_list:lance.bfloat16:8",
        None
    )]
    #[case::struct_type("struct", "struct", None)]
    #[case::map("map", "map", None)]
    fn test_semantic_logical_type(
        #[case] logical_type: &str,
        #[case] canonical: &str,
        #[case] implied_encoding: Option<&str>,
    ) {
        let semantic = LogicalType::from(logical_type).semantic().unwrap();
        assert_eq!(semantic.semantic_type.logical_type().to_string(), canonical);
        assert_eq!(
            semantic
                .implied_encoding
                .map(|encoding| encoding.to_string()),
            implied_encoding.map(str::to_string)
        );
        // The canonical name reads as the same semantic type with no implied layout.
        let canonical = semantic.semantic_type.logical_type().semantic().unwrap();
        assert_eq!(canonical.semantic_type, semantic.semantic_type);
        assert_eq!(canonical.implied_encoding, None);
    }

    #[rstest]
    #[case::dictionary_of_integers("dict:int32:int8:false")]
    #[case::decimal128_too_wide("decimal:128:40:2")]
    #[case::decimal_zero_precision("decimal:0:0")]
    #[case::decimal_too_precise("decimal:77:2")]
    #[case::decimal_scale_above_precision("decimal:5:6")]
    #[case::decimal_bad_width("decimal:64:10:2")]
    #[case::unknown("string_view")]
    fn test_semantic_logical_type_rejected(#[case] logical_type: &str) {
        assert!(
            LogicalType::from(logical_type).semantic().is_err(),
            "expected {logical_type} to be rejected"
        );
    }

    #[rstest]
    #[case::utf8("utf8")]
    #[case::large_utf8("large_utf8")]
    #[case::utf8_view("utf8_view")]
    #[case::binary("binary")]
    #[case::large_binary("large_binary")]
    #[case::binary_view("binary_view")]
    #[case::dictionary("dictionary:int16:utf8")]
    #[case::dictionary_unsigned("dictionary:uint64:large_binary")]
    #[case::list("list")]
    #[case::large_list("large_list")]
    #[case::decimal128("decimal128")]
    #[case::decimal256("decimal256")]
    #[case::arrow_json("arrow.json")]
    #[case::lance_json("lance.json")]
    fn test_output_encoding_round_trip(#[case] value: &str) {
        let encoding = value.parse::<OutputEncoding>().unwrap();
        assert_eq!(encoding.to_string(), value);
    }

    #[rstest]
    #[case::unknown("utf16")]
    #[case::dictionary_float_key("dictionary:float:utf8")]
    #[case::dictionary_view_value("dictionary:int16:utf8_view")]
    #[case::dictionary_nested("dictionary:int16:dictionary:int8:utf8")]
    #[case::dictionary_missing_value("dictionary:int16")]
    #[case::legacy_name("large_string")]
    fn test_output_encoding_rejected(#[case] value: &str) {
        assert!(value.parse::<OutputEncoding>().is_err(), "{value}");
    }

    #[rstest]
    #[case::string_large("string", "large_utf8", true)]
    #[case::string_view("string", "utf8_view", true)]
    #[case::string_dictionary("string", "dictionary:int32:large_utf8", true)]
    #[case::string_binary_dictionary("string", "dictionary:int32:binary", false)]
    #[case::string_binary("string", "binary", false)]
    #[case::binary_view("binary", "binary_view", true)]
    #[case::binary_utf8("binary", "utf8", false)]
    #[case::list_large("list", "large_list", true)]
    #[case::decimal128_narrow("decimal:38:2", "decimal128", true)]
    #[case::decimal128_wide("decimal:40:2", "decimal128", false)]
    #[case::decimal256_narrow("decimal:10:2", "decimal256", true)]
    #[case::json_text("json", "arrow.json", true)]
    #[case::json_jsonb("json", "lance.json", true)]
    #[case::json_utf8("json", "utf8", false)]
    #[case::int_utf8("int32", "utf8", false)]
    #[case::blob_binary("blob", "large_binary", false)]
    fn test_check_output_encoding(
        #[case] logical_type: &str,
        #[case] encoding: &str,
        #[case] valid: bool,
    ) {
        let semantic_type = LogicalType::from(logical_type)
            .semantic()
            .unwrap()
            .semantic_type;
        let encoding = encoding.parse::<OutputEncoding>().unwrap();
        assert_eq!(
            semantic_type.check_output_encoding(&encoding).is_ok(),
            valid
        );
    }

    #[rstest]
    #[case::string_aliases("string", "large_string", true)]
    #[case::string_dictionary("dict:string:int32:false", "large_string", true)]
    #[case::binary_dictionary("dict:large_binary:int8:false", "binary", true)]
    #[case::string_binary("string", "binary", false)]
    #[case::list_aliases("large_list.struct", "list", true)]
    #[case::decimal_widths("decimal:256:10:2", "decimal:10:2", true)]
    #[case::decimal_legacy_widths("decimal:128:10:2", "decimal:256:10:2", true)]
    #[case::decimal_precision("decimal:128:12:2", "decimal:10:2", false)]
    #[case::decimal_scale("decimal:10:3", "decimal:10:2", false)]
    #[case::int_widths("int32", "int64", false)]
    #[case::float_widths("float", "double", false)]
    #[case::json_binary("json", "large_binary", false)]
    #[case::blob_binary("blob", "large_binary", false)]
    #[case::timestamps("timestamp:us:UTC", "timestamp:us:-", false)]
    fn test_type_comparison(
        #[case] actual: &str,
        #[case] expected: &str,
        #[case] semantic_match: bool,
    ) {
        let actual = LogicalType::from(actual);
        let expected = LogicalType::from(expected);
        assert_eq!(
            TypeComparison::Semantic.logical_types_match(&actual, &expected),
            semantic_match
        );
        assert_eq!(
            TypeComparison::Semantic.logical_types_match(&expected, &actual),
            semantic_match
        );
        // Legacy tables keep comparing the strings.
        assert!(!TypeComparison::Exact.logical_types_match(&actual, &expected));
        assert!(TypeComparison::Exact.logical_types_match(&actual, &actual));
    }
}
