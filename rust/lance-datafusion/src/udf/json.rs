// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::builder::{
    BooleanBuilder, Float64Builder, Int64Builder, LargeBinaryBuilder, StringBuilder,
};
use arrow_array::{Array, ArrayRef, LargeBinaryArray, StringArray};
use arrow_schema::DataType;
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{ScalarUDF, Volatility};
use datafusion::physical_plan::ColumnarValue;
use datafusion::prelude::create_udf;
use lance_arrow::json::{json_path_exists, select_json_path, select_json_path_values};
use std::sync::Arc;

/// Represents the type of a JSONB value
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonbType {
    Null = 0,
    Boolean = 1,
    Int64 = 2,
    Float64 = 3,
    String = 4,
    Array = 5,
    Object = 6,
}

impl JsonbType {
    /// Convert from u8 value
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Null),
            1 => Some(Self::Boolean),
            2 => Some(Self::Int64),
            3 => Some(Self::Float64),
            4 => Some(Self::String),
            5 => Some(Self::Array),
            6 => Some(Self::Object),
            _ => None,
        }
    }

    /// Convert to u8 value for storage in Arrow arrays
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Common helper functions and types for JSON UDFs
mod common {
    use super::*;

    /// Convert ColumnarValue arguments to ArrayRef vector
    ///
    /// Note: This implementation currently broadcasts scalars to arrays.
    /// Future optimization: handle scalars directly without broadcasting
    /// to improve performance for scalar inputs.
    pub fn columnar_to_arrays(args: &[ColumnarValue]) -> Vec<ArrayRef> {
        args.iter()
            .map(|arg| match arg {
                ColumnarValue::Array(arr) => arr.clone(),
                ColumnarValue::Scalar(scalar) => scalar.to_array().unwrap(),
            })
            .collect()
    }

    /// Create DataFusionError for execution failures (simplified error wrapping)
    pub fn execution_error(msg: impl Into<String>) -> DataFusionError {
        DataFusionError::Execution(msg.into())
    }

    /// Validate argument count for UDF
    pub fn validate_arg_count(
        args: &[ArrayRef],
        expected: usize,
        function_name: &str,
    ) -> Result<()> {
        if args.len() != expected {
            return Err(execution_error(format!(
                "{} requires exactly {} arguments",
                function_name, expected
            )));
        }
        Ok(())
    }

    /// Extract and validate LargeBinaryArray from first argument
    pub fn extract_jsonb_array(args: &[ArrayRef]) -> Result<&LargeBinaryArray> {
        args[0]
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .ok_or_else(|| execution_error("First argument must be LargeBinary"))
    }

    /// Extract and validate StringArray from specified argument
    pub fn extract_string_array(args: &[ArrayRef], arg_index: usize) -> Result<&StringArray> {
        args[arg_index]
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| execution_error(format!("Argument {} must be String", arg_index + 1)))
    }

    /// Get string value at index, handling scalar broadcast case
    /// When a scalar is converted to an array, it becomes a single-element array
    /// This function handles accessing that value repeatedly for all rows
    pub fn get_string_value_at(string_array: &StringArray, index: usize) -> Option<&str> {
        // Handle scalar broadcast case: if array has only 1 element, always use index 0
        let actual_index = if string_array.len() == 1 { 0 } else { index };

        if string_array.is_null(actual_index) {
            None
        } else {
            Some(string_array.value(actual_index))
        }
    }

    /// Get a JSON field or array element by key.
    pub fn get_json_value_by_key(
        raw_jsonb: &lance_jsonb::RawJsonb,
        key: &str,
    ) -> Result<Option<lance_jsonb::OwnedJsonb>> {
        if raw_jsonb.is_object().unwrap_or(false) {
            raw_jsonb
                .get_by_name(key, false)
                .map_err(|e| execution_error(format!("Failed to get field '{}': {}", key, e)))
        } else if raw_jsonb.is_array().unwrap_or(false) {
            match key.parse::<usize>() {
                Ok(index) => raw_jsonb.get_by_index(index).map_err(|e| {
                    execution_error(format!("Failed to get array element [{}]: {}", index, e))
                }),
                Err(_) => Ok(None),
            }
        } else {
            Ok(None)
        }
    }

    /// Parse JSONPath with proper error handling (no false returns)
    pub fn parse_json_path(path: &str) -> Result<lance_jsonb::jsonpath::JsonPath<'_>> {
        lance_arrow::json::parse_json_path(path).map_err(|e| execution_error(e.to_string()))
    }
}

/// Convert JSONB value to string using jsonb's built-in serde (strict mode)
///
/// Values that cannot be represented as a string (arrays, objects) are null,
/// per the documented "null if conversion fails" contract for the
/// `json_get_string` UDF family.
fn json_value_to_string(value: lance_jsonb::OwnedJsonb) -> Result<Option<String>> {
    let raw_jsonb = value.as_raw();

    // Check for null first
    if raw_jsonb
        .is_null()
        .map_err(|e| common::execution_error(format!("Failed to check null: {}", e)))?
    {
        return Ok(None);
    }

    // Use jsonb's built-in to_str() method - strict conversion
    match raw_jsonb.to_str() {
        Ok(s) => Ok(Some(s)),
        Err(lance_jsonb::Error::InvalidCast) => Ok(None),
        Err(e) => Err(common::execution_error(format!(
            "Failed to convert to string: {}",
            e
        ))),
    }
}

/// Convert JSONB value to integer using jsonb's built-in serde (strict mode)
///
/// Values that cannot be represented as an integer (non-numeric strings,
/// arrays, objects) are null, per the documented "null if conversion fails"
/// contract for the `json_get_int` UDF family.
fn json_value_to_int(value: lance_jsonb::OwnedJsonb) -> Result<Option<i64>> {
    let raw_jsonb = value.as_raw();

    // Check for null first
    if raw_jsonb
        .is_null()
        .map_err(|e| common::execution_error(format!("Failed to check null: {}", e)))?
    {
        return Ok(None);
    }

    // Use jsonb's built-in to_i64() method - strict conversion
    match raw_jsonb.to_i64() {
        Ok(v) => Ok(Some(v)),
        Err(lance_jsonb::Error::InvalidCast) => Ok(None),
        Err(e) => Err(common::execution_error(format!(
            "Failed to convert to integer: {}",
            e
        ))),
    }
}

/// Convert JSONB value to float using jsonb's built-in serde (strict mode)
///
/// Values that cannot be represented as a float (non-numeric strings, arrays,
/// objects) are null, per the documented "null if conversion fails" contract
/// for the `json_get_float` UDF family.
fn json_value_to_float(value: lance_jsonb::OwnedJsonb) -> Result<Option<f64>> {
    let raw_jsonb = value.as_raw();

    // Check for null first
    if raw_jsonb
        .is_null()
        .map_err(|e| common::execution_error(format!("Failed to check null: {}", e)))?
    {
        return Ok(None);
    }

    // Use jsonb's built-in to_f64() method - strict conversion
    match raw_jsonb.to_f64() {
        Ok(v) => Ok(Some(v)),
        Err(lance_jsonb::Error::InvalidCast) => Ok(None),
        Err(e) => Err(common::execution_error(format!(
            "Failed to convert to float: {}",
            e
        ))),
    }
}

/// Convert JSONB value to boolean using jsonb's built-in serde (strict mode)
///
/// Numbers coerce per the documented contract (0 -> false, non-zero -> true).
/// Other values that cannot be represented as a boolean (arrays, objects,
/// unrecognized strings) are null, per the documented "null if conversion
/// fails" contract for the `json_get_bool` UDF family.
fn json_value_to_bool(value: lance_jsonb::OwnedJsonb) -> Result<Option<bool>> {
    let raw_jsonb = value.as_raw();

    // Check for null first
    if raw_jsonb
        .is_null()
        .map_err(|e| common::execution_error(format!("Failed to check null: {}", e)))?
    {
        return Ok(None);
    }

    if let Some(number) = raw_jsonb
        .as_number()
        .map_err(|e| common::execution_error(format!("Failed to convert to boolean: {}", e)))?
    {
        return Ok(Some(number != lance_jsonb::Number::Int64(0)));
    }

    // Use jsonb's built-in to_bool() method - strict conversion
    match raw_jsonb.to_bool() {
        Ok(v) => Ok(Some(v)),
        Err(lance_jsonb::Error::InvalidCast) => Ok(None),
        Err(e) => Err(common::execution_error(format!(
            "Failed to convert to boolean: {}",
            e
        ))),
    }
}

/// Create the json_extract UDF for extracting JSONPath from JSON data
///
/// # Arguments
/// * First parameter: JSONB binary data (LargeBinary)
/// * Second parameter: JSONPath expression as string (Utf8)
///
/// # Returns
/// String representation of the extracted value, or null if path not found
pub fn json_extract_udf() -> ScalarUDF {
    create_udf(
        "json_extract",
        vec![DataType::LargeBinary, DataType::Utf8],
        DataType::Utf8,
        Volatility::Immutable,
        Arc::new(json_extract_columnar_impl),
    )
}

/// Create the json_extract_with_type UDF that returns JSONB bytes with type information
///
/// # Arguments
/// * First parameter: JSONB binary data (LargeBinary)
/// * Second parameter: JSONPath expression as string (Utf8)
///
/// # Returns
/// A struct with two fields:
/// - value: LargeBinary (the extracted JSONB value)
/// - type_tag: UInt8 (type information: 0=null, 1=bool, 2=int64, 3=float64, 4=string, 5=array, 6=object)
pub fn json_extract_with_type_udf() -> ScalarUDF {
    use arrow_schema::Fields;

    let return_type = DataType::Struct(Fields::from(vec![
        arrow_schema::Field::new("value", DataType::LargeBinary, true),
        arrow_schema::Field::new("type_tag", DataType::UInt8, false),
    ]));

    create_udf(
        "json_extract_with_type",
        vec![DataType::LargeBinary, DataType::Utf8],
        return_type,
        Volatility::Immutable,
        Arc::new(json_extract_with_type_columnar_impl),
    )
}

/// Implementation of json_extract function with ColumnarValue
fn json_extract_columnar_impl(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    let arrays = common::columnar_to_arrays(args);
    let result = json_extract_impl(&arrays)?;
    Ok(ColumnarValue::Array(result))
}

/// Implementation of json_extract_with_type function with ColumnarValue
fn json_extract_with_type_columnar_impl(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    let arrays = common::columnar_to_arrays(args);
    let result = json_extract_with_type_impl(&arrays)?;
    Ok(ColumnarValue::Array(result))
}

/// Implementation of json_extract function
fn json_extract_impl(args: &[ArrayRef]) -> Result<ArrayRef> {
    common::validate_arg_count(args, 2, "json_extract")?;

    let jsonb_array = common::extract_jsonb_array(args)?;
    let path_array = common::extract_string_array(args, 1)?;
    let mut builder = StringBuilder::with_capacity(jsonb_array.len(), 1024);

    for i in 0..jsonb_array.len() {
        if jsonb_array.is_null(i) {
            builder.append_null();
        } else if let Some(path) = common::get_string_value_at(path_array, i) {
            let jsonb_bytes = jsonb_array.value(i);
            match extract_json_path(jsonb_bytes, path)? {
                Some(value) => builder.append_value(&value),
                None => builder.append_null(),
            }
        } else {
            builder.append_null();
        }
    }

    Ok(Arc::new(builder.finish()))
}

/// Implementation of json_extract_with_type function
fn json_extract_with_type_impl(args: &[ArrayRef]) -> Result<ArrayRef> {
    use arrow_array::StructArray;
    use arrow_array::builder::{LargeBinaryBuilder, UInt8Builder};

    common::validate_arg_count(args, 2, "json_extract_with_type")?;

    let jsonb_array = common::extract_jsonb_array(args)?;
    let path_array = common::extract_string_array(args, 1)?;

    let mut value_builder = LargeBinaryBuilder::with_capacity(jsonb_array.len(), 1024);
    let mut type_builder = UInt8Builder::with_capacity(jsonb_array.len());

    for i in 0..jsonb_array.len() {
        if jsonb_array.is_null(i) {
            value_builder.append_null();
            type_builder.append_value(JsonbType::Null.as_u8());
        } else if let Some(path) = common::get_string_value_at(path_array, i) {
            let jsonb_bytes = jsonb_array.value(i);
            match extract_json_path_with_type(jsonb_bytes, path)? {
                Some((value_bytes, type_tag)) => {
                    value_builder.append_value(&value_bytes);
                    type_builder.append_value(type_tag);
                }
                None => {
                    value_builder.append_null();
                    type_builder.append_value(JsonbType::Null.as_u8());
                }
            }
        } else {
            value_builder.append_null();
            type_builder.append_value(JsonbType::Null.as_u8());
        }
    }

    // Create struct array with two fields
    let value_array = Arc::new(value_builder.finish()) as ArrayRef;
    let type_array = Arc::new(type_builder.finish()) as ArrayRef;

    let struct_array = StructArray::from(vec![
        (
            Arc::new(arrow_schema::Field::new(
                "value",
                DataType::LargeBinary,
                true,
            )),
            value_array,
        ),
        (
            Arc::new(arrow_schema::Field::new("type_tag", DataType::UInt8, false)),
            type_array,
        ),
    ]);

    Ok(Arc::new(struct_array))
}

/// Extract value from JSONB using JSONPath and return with type information
/// Returns (JSONB bytes, type_tag) where type_tag represents the JsonbType
fn extract_json_path_with_type(jsonb_bytes: &[u8], path: &str) -> Result<Option<(Vec<u8>, u8)>> {
    let json_path = common::parse_json_path(path)?;

    match select_json_path(jsonb_bytes, &json_path) {
        Ok(Some(owned_value)) => {
            let raw = owned_value.as_raw();

            // Determine type using JsonbType enum
            let jsonb_type = if raw.is_null().unwrap_or(false) {
                JsonbType::Null
            } else if raw.is_boolean().unwrap_or(false) {
                JsonbType::Boolean
            } else if raw.is_number().unwrap_or(false) {
                let is_float_storage =
                    matches!(raw.as_number(), Ok(Some(lance_jsonb::Number::Float64(_))));
                if !is_float_storage && raw.is_i64().unwrap_or(false) {
                    JsonbType::Int64
                } else {
                    JsonbType::Float64
                }
            } else if raw.is_string().unwrap_or(false) {
                JsonbType::String
            } else if raw.is_array().unwrap_or(false) {
                JsonbType::Array
            } else if raw.is_object().unwrap_or(false) {
                JsonbType::Object
            } else {
                JsonbType::String // default to string
            };

            // Return the JSONB bytes and type tag as u8
            Ok(Some((owned_value.to_vec(), jsonb_type.as_u8())))
        }
        Ok(None) => Ok(None),
        Err(e) => Err(common::execution_error(format!(
            "Failed to select value from path '{}': {}",
            path, e
        ))),
    }
}

/// Extract value from JSONB using JSONPath
///
/// Note: Uses `select_value` so JSONPath expressions matching multiple values
/// return a JSON array instead of silently dropping all but the first match.
fn extract_json_path(jsonb_bytes: &[u8], path: &str) -> Result<Option<String>> {
    let json_path = common::parse_json_path(path)?;

    match select_json_path(jsonb_bytes, &json_path) {
        Ok(value) => Ok(value.map(|value| value.to_string())),
        Err(e) => Err(common::execution_error(format!(
            "Failed to select value from path '{}': {}",
            path, e
        ))),
    }
}

/// Create the json_exists UDF for checking if a JSONPath exists
///
/// # Arguments
/// * First parameter: JSONB binary data (LargeBinary)
/// * Second parameter: JSONPath expression as string (Utf8)
///
/// # Returns
/// Boolean indicating whether the path exists in the JSON data
pub fn json_exists_udf() -> ScalarUDF {
    create_udf(
        "json_exists",
        vec![DataType::LargeBinary, DataType::Utf8],
        DataType::Boolean,
        Volatility::Immutable,
        Arc::new(json_exists_columnar_impl),
    )
}

/// Implementation of json_exists function with ColumnarValue
fn json_exists_columnar_impl(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    let arrays = common::columnar_to_arrays(args);
    let result = json_exists_impl(&arrays)?;
    Ok(ColumnarValue::Array(result))
}

/// Implementation of json_exists function
fn json_exists_impl(args: &[ArrayRef]) -> Result<ArrayRef> {
    common::validate_arg_count(args, 2, "json_exists")?;

    let jsonb_array = common::extract_jsonb_array(args)?;
    let path_array = common::extract_string_array(args, 1)?;

    let mut builder = BooleanBuilder::with_capacity(jsonb_array.len());

    for i in 0..jsonb_array.len() {
        if jsonb_array.is_null(i) {
            builder.append_null();
        } else if let Some(path) = common::get_string_value_at(path_array, i) {
            let jsonb_bytes = jsonb_array.value(i);
            let exists = check_json_path_exists(jsonb_bytes, path)?;
            builder.append_value(exists);
        } else {
            builder.append_null();
        }
    }

    Ok(Arc::new(builder.finish()))
}

/// Check if a JSONPath exists in JSONB
fn check_json_path_exists(jsonb_bytes: &[u8], path: &str) -> Result<bool> {
    let json_path = common::parse_json_path(path)?;

    match json_path_exists(jsonb_bytes, &json_path) {
        Ok(exists) => Ok(exists),
        Err(e) => Err(common::execution_error(format!(
            "Failed to check existence of path '{}': {}",
            path, e
        ))),
    }
}

/// Create the json_get UDF for getting a field value as JSON string
///
/// # Arguments
/// * First parameter: JSONB binary data (LargeBinary)
/// * Second parameter: Field name or array index as string (Utf8)
///
/// # Returns
/// Raw JSONB bytes of the field value, or null if not found
pub fn json_get_udf() -> ScalarUDF {
    create_udf(
        "json_get",
        vec![DataType::LargeBinary, DataType::Utf8],
        DataType::LargeBinary,
        Volatility::Immutable,
        Arc::new(json_get_columnar_impl),
    )
}

/// Implementation of json_get function with ColumnarValue
fn json_get_columnar_impl(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    let arrays = common::columnar_to_arrays(args);
    let result = json_get_impl(&arrays)?;
    Ok(ColumnarValue::Array(result))
}

/// Implementation of json_get function
fn json_get_impl(args: &[ArrayRef]) -> Result<ArrayRef> {
    common::validate_arg_count(args, 2, "json_get")?;

    let jsonb_array = common::extract_jsonb_array(args)?;
    let key_array = common::extract_string_array(args, 1)?;

    let mut builder = LargeBinaryBuilder::with_capacity(jsonb_array.len(), 0);

    for i in 0..jsonb_array.len() {
        if jsonb_array.is_null(i) {
            builder.append_null();
        } else if let Some(key) = common::get_string_value_at(key_array, i) {
            let jsonb_bytes = jsonb_array.value(i);
            let raw_jsonb = lance_jsonb::RawJsonb::new(jsonb_bytes);

            match common::get_json_value_by_key(&raw_jsonb, key)? {
                Some(value) => builder.append_value(value.as_raw().as_ref()),
                None => builder.append_null(),
            }
        } else {
            builder.append_null();
        }
    }

    Ok(Arc::new(builder.finish()))
}

/// Create the json_get_string UDF for getting a string value
///
/// # Arguments
/// * First parameter: JSONB binary data (LargeBinary)
/// * Second parameter: Field name or array index as string (Utf8)
///
/// # Returns
/// String value with type coercion (numbers/booleans converted to strings)
pub fn json_get_string_udf() -> ScalarUDF {
    create_udf(
        "json_get_string",
        vec![DataType::LargeBinary, DataType::Utf8],
        DataType::Utf8,
        Volatility::Immutable,
        Arc::new(json_get_string_columnar_impl),
    )
}

/// Implementation of json_get_string function with ColumnarValue
fn json_get_string_columnar_impl(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    let arrays = common::columnar_to_arrays(args);
    let result = json_get_string_impl(&arrays)?;
    Ok(ColumnarValue::Array(result))
}

/// Implementation of json_get_string function
fn json_get_string_impl(args: &[ArrayRef]) -> Result<ArrayRef> {
    common::validate_arg_count(args, 2, "json_get_string")?;

    let jsonb_array = common::extract_jsonb_array(args)?;
    let key_array = common::extract_string_array(args, 1)?;

    let mut builder = StringBuilder::with_capacity(jsonb_array.len(), 1024);

    for i in 0..jsonb_array.len() {
        if jsonb_array.is_null(i) {
            builder.append_null();
        } else if let Some(key) = common::get_string_value_at(key_array, i) {
            let jsonb_bytes = jsonb_array.value(i);
            let raw_jsonb = lance_jsonb::RawJsonb::new(jsonb_bytes);

            match common::get_json_value_by_key(&raw_jsonb, key)? {
                Some(value) => match json_value_to_string(value)? {
                    Some(string_val) => builder.append_value(&string_val),
                    None => builder.append_null(),
                },
                None => builder.append_null(),
            }
        } else {
            builder.append_null();
        }
    }

    Ok(Arc::new(builder.finish()))
}

/// Create the json_get_int UDF for getting an integer value
///
/// # Arguments
/// * First parameter: JSONB binary data (LargeBinary)
/// * Second parameter: Field name or array index as string (Utf8)
///
/// # Returns
/// Integer value with type coercion (strings/floats/booleans converted to int)
pub fn json_get_int_udf() -> ScalarUDF {
    create_udf(
        "json_get_int",
        vec![DataType::LargeBinary, DataType::Utf8],
        DataType::Int64,
        Volatility::Immutable,
        Arc::new(json_get_int_columnar_impl),
    )
}

/// Implementation of json_get_int function with ColumnarValue
fn json_get_int_columnar_impl(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    let arrays = common::columnar_to_arrays(args);
    let result = json_get_int_impl(&arrays)?;
    Ok(ColumnarValue::Array(result))
}

/// Implementation of json_get_int function
fn json_get_int_impl(args: &[ArrayRef]) -> Result<ArrayRef> {
    common::validate_arg_count(args, 2, "json_get_int")?;

    let jsonb_array = common::extract_jsonb_array(args)?;
    let key_array = common::extract_string_array(args, 1)?;

    let mut builder = Int64Builder::with_capacity(jsonb_array.len());

    for i in 0..jsonb_array.len() {
        if jsonb_array.is_null(i) {
            builder.append_null();
        } else if let Some(key) = common::get_string_value_at(key_array, i) {
            let jsonb_bytes = jsonb_array.value(i);
            let raw_jsonb = lance_jsonb::RawJsonb::new(jsonb_bytes);

            match common::get_json_value_by_key(&raw_jsonb, key)? {
                Some(value) => match json_value_to_int(value)? {
                    Some(int_val) => builder.append_value(int_val),
                    None => builder.append_null(),
                },
                None => builder.append_null(),
            }
        } else {
            builder.append_null();
        }
    }

    Ok(Arc::new(builder.finish()))
}

/// Create the json_get_float UDF for getting a float value
///
/// # Arguments
/// * First parameter: JSONB binary data (LargeBinary)
/// * Second parameter: Field name or array index as string (Utf8)
///
/// # Returns
/// Float value with type coercion (strings/integers/booleans converted to float)
pub fn json_get_float_udf() -> ScalarUDF {
    create_udf(
        "json_get_float",
        vec![DataType::LargeBinary, DataType::Utf8],
        DataType::Float64,
        Volatility::Immutable,
        Arc::new(json_get_float_columnar_impl),
    )
}

/// Implementation of json_get_float function with ColumnarValue
fn json_get_float_columnar_impl(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    let arrays = common::columnar_to_arrays(args);
    let result = json_get_float_impl(&arrays)?;
    Ok(ColumnarValue::Array(result))
}

/// Implementation of json_get_float function
fn json_get_float_impl(args: &[ArrayRef]) -> Result<ArrayRef> {
    common::validate_arg_count(args, 2, "json_get_float")?;

    let jsonb_array = common::extract_jsonb_array(args)?;
    let key_array = common::extract_string_array(args, 1)?;

    let mut builder = Float64Builder::with_capacity(jsonb_array.len());

    for i in 0..jsonb_array.len() {
        if jsonb_array.is_null(i) {
            builder.append_null();
        } else if let Some(key) = common::get_string_value_at(key_array, i) {
            let jsonb_bytes = jsonb_array.value(i);
            let raw_jsonb = lance_jsonb::RawJsonb::new(jsonb_bytes);

            match common::get_json_value_by_key(&raw_jsonb, key)? {
                Some(value) => match json_value_to_float(value)? {
                    Some(float_val) => builder.append_value(float_val),
                    None => builder.append_null(),
                },
                None => builder.append_null(),
            }
        } else {
            builder.append_null();
        }
    }

    Ok(Arc::new(builder.finish()))
}

/// Create the json_get_bool UDF for getting a boolean value
///
/// # Arguments
/// * First parameter: JSONB binary data (LargeBinary)
/// * Second parameter: Field name or array index as string (Utf8)
///
/// # Returns
/// Boolean value with flexible type coercion (strings like 'true'/'yes' become true, numbers follow 0 -> false / non-zero -> true, other values null)
pub fn json_get_bool_udf() -> ScalarUDF {
    create_udf(
        "json_get_bool",
        vec![DataType::LargeBinary, DataType::Utf8],
        DataType::Boolean,
        Volatility::Immutable,
        Arc::new(json_get_bool_columnar_impl),
    )
}

/// Implementation of json_get_bool function with ColumnarValue
fn json_get_bool_columnar_impl(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    let arrays = common::columnar_to_arrays(args);
    let result = json_get_bool_impl(&arrays)?;
    Ok(ColumnarValue::Array(result))
}

/// Implementation of json_get_bool function
fn json_get_bool_impl(args: &[ArrayRef]) -> Result<ArrayRef> {
    common::validate_arg_count(args, 2, "json_get_bool")?;

    let jsonb_array = common::extract_jsonb_array(args)?;
    let key_array = common::extract_string_array(args, 1)?;

    let mut builder = BooleanBuilder::with_capacity(jsonb_array.len());

    for i in 0..jsonb_array.len() {
        if jsonb_array.is_null(i) {
            builder.append_null();
        } else if let Some(key) = common::get_string_value_at(key_array, i) {
            let jsonb_bytes = jsonb_array.value(i);
            let raw_jsonb = lance_jsonb::RawJsonb::new(jsonb_bytes);

            match common::get_json_value_by_key(&raw_jsonb, key)? {
                Some(value) => match json_value_to_bool(value)? {
                    Some(bool_val) => builder.append_value(bool_val),
                    None => builder.append_null(),
                },
                None => builder.append_null(),
            }
        } else {
            builder.append_null();
        }
    }

    Ok(Arc::new(builder.finish()))
}

/// Create the json_array_contains UDF for checking if array contains a value
///
/// # Arguments
/// * First parameter: JSONB binary data (LargeBinary)
/// * Second parameter: JSONPath to array location (Utf8)
/// * Third parameter: Value to search for as string (Utf8)
///
/// # Returns
/// Boolean indicating whether the array contains the specified value
pub fn json_array_contains_udf() -> ScalarUDF {
    create_udf(
        "json_array_contains",
        vec![DataType::LargeBinary, DataType::Utf8, DataType::Utf8],
        DataType::Boolean,
        Volatility::Immutable,
        Arc::new(json_array_contains_columnar_impl),
    )
}

/// Implementation of json_array_contains function with ColumnarValue
fn json_array_contains_columnar_impl(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    let arrays = common::columnar_to_arrays(args);
    let result = json_array_contains_impl(&arrays)?;
    Ok(ColumnarValue::Array(result))
}

/// Implementation of json_array_contains function
fn json_array_contains_impl(args: &[ArrayRef]) -> Result<ArrayRef> {
    common::validate_arg_count(args, 3, "json_array_contains")?;

    let jsonb_array = common::extract_jsonb_array(args)?;
    let path_array = common::extract_string_array(args, 1)?;
    let value_array = common::extract_string_array(args, 2)?;

    let mut builder = BooleanBuilder::with_capacity(jsonb_array.len());

    for i in 0..jsonb_array.len() {
        if jsonb_array.is_null(i) {
            builder.append_null();
        } else {
            let path = common::get_string_value_at(path_array, i);
            let value = common::get_string_value_at(value_array, i);

            match (path, value) {
                (Some(p), Some(v)) => {
                    let jsonb_bytes = jsonb_array.value(i);
                    let contains = check_array_contains(jsonb_bytes, p, v)?;
                    builder.append_value(contains);
                }
                _ => builder.append_null(),
            }
        }
    }

    Ok(Arc::new(builder.finish()))
}

/// Check if a JSON array at path contains a value
fn check_array_contains(jsonb_bytes: &[u8], path: &str, value: &str) -> Result<bool> {
    let json_path = common::parse_json_path(path)?;

    match select_json_path_values(jsonb_bytes, &json_path) {
        Ok(values) => {
            for v in values {
                // Convert to raw JSONB for direct access
                let raw = v.as_raw();
                let elements = raw.array_elements().map_err(|e| {
                    common::execution_error(format!(
                        "Failed to check array contains at path '{}': {}",
                        path, e
                    ))
                })?;
                let Some(elements) = elements else {
                    continue; // Not an array
                };
                for elem in elements {
                    let elem = elem.map_err(|e| {
                        common::execution_error(format!(
                            "Failed to check array contains at path '{}': {}",
                            path, e
                        ))
                    })?;
                    // A needle matches an element's JSON text (`"python"`,
                    // `2`) or its content: to_str() returns strings
                    // unquoted and unescaped, so `say "hi"` matches the
                    // element serialized as `"say \"hi\""`.
                    let matched = elem.to_string() == value
                        || elem.as_raw().to_str().is_ok_and(|s| s == value);
                    if matched {
                        return Ok(true);
                    }
                }
            }
            Ok(false)
        }
        Err(e) => Err(common::execution_error(format!(
            "Failed to check array contains at path '{}': {}",
            path, e
        ))),
    }
}

/// Create the json_array_length UDF for getting array length
///
/// # Arguments
/// * First parameter: JSONB binary data (LargeBinary)
/// * Second parameter: JSONPath to array location (Utf8)
///
/// # Returns
/// Integer length of the JSON array, or null if path doesn't point to an array
pub fn json_array_length_udf() -> ScalarUDF {
    create_udf(
        "json_array_length",
        vec![DataType::LargeBinary, DataType::Utf8],
        DataType::Int64,
        Volatility::Immutable,
        Arc::new(json_array_length_columnar_impl),
    )
}

/// Implementation of json_array_length function with ColumnarValue
fn json_array_length_columnar_impl(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    let arrays = common::columnar_to_arrays(args);
    let result = json_array_length_impl(&arrays)?;
    Ok(ColumnarValue::Array(result))
}

/// Implementation of json_array_length function
fn json_array_length_impl(args: &[ArrayRef]) -> Result<ArrayRef> {
    common::validate_arg_count(args, 2, "json_array_length")?;

    let jsonb_array = common::extract_jsonb_array(args)?;
    let path_array = common::extract_string_array(args, 1)?;

    let mut builder = Int64Builder::with_capacity(jsonb_array.len());

    for i in 0..jsonb_array.len() {
        if jsonb_array.is_null(i) {
            builder.append_null();
        } else if let Some(path) = common::get_string_value_at(path_array, i) {
            let jsonb_bytes = jsonb_array.value(i);
            match get_array_length(jsonb_bytes, path)? {
                Some(len) => builder.append_value(len),
                None => builder.append_null(),
            }
        } else {
            builder.append_null();
        }
    }

    Ok(Arc::new(builder.finish()))
}

/// Get the length of a JSON array at path
fn get_array_length(jsonb_bytes: &[u8], path: &str) -> Result<Option<i64>> {
    let json_path = common::parse_json_path(path)?;

    match select_json_path_values(jsonb_bytes, &json_path) {
        Ok(values) => {
            if values.is_empty() {
                return Ok(None);
            }
            let first = &values[0];
            let raw = first.as_raw();

            // Read the length from the array container header in O(1). A path
            // that resolves to a non-array is null, indistinguishable from a
            // missing path.
            let len = raw.array_len().map_err(|e| {
                common::execution_error(format!(
                    "Failed to get array length at path '{}': {}",
                    path, e
                ))
            })?;
            Ok(len.map(|len| len as i64))
        }
        Err(e) => Err(common::execution_error(format!(
            "Failed to get array length at path '{}': {}",
            path, e
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::builder::LargeBinaryBuilder;
    use arrow_array::{BooleanArray, Float64Array, Int64Array};

    fn create_test_jsonb(json_str: &str) -> Vec<u8> {
        lance_jsonb::parse_value(json_str.as_bytes())
            .unwrap()
            .to_vec()
    }

    #[test]
    fn test_jsonb_type_enum() {
        // Test enum conversion to/from u8
        assert_eq!(JsonbType::Null.as_u8(), 0);
        assert_eq!(JsonbType::Boolean.as_u8(), 1);
        assert_eq!(JsonbType::Int64.as_u8(), 2);
        assert_eq!(JsonbType::Float64.as_u8(), 3);
        assert_eq!(JsonbType::String.as_u8(), 4);
        assert_eq!(JsonbType::Array.as_u8(), 5);
        assert_eq!(JsonbType::Object.as_u8(), 6);

        // Test from_u8 conversion
        assert_eq!(JsonbType::from_u8(0), Some(JsonbType::Null));
        assert_eq!(JsonbType::from_u8(1), Some(JsonbType::Boolean));
        assert_eq!(JsonbType::from_u8(2), Some(JsonbType::Int64));
        assert_eq!(JsonbType::from_u8(3), Some(JsonbType::Float64));
        assert_eq!(JsonbType::from_u8(4), Some(JsonbType::String));
        assert_eq!(JsonbType::from_u8(5), Some(JsonbType::Array));
        assert_eq!(JsonbType::from_u8(6), Some(JsonbType::Object));
        assert_eq!(JsonbType::from_u8(7), None); // Invalid value
    }

    #[tokio::test]
    async fn test_json_extract_udf() -> Result<()> {
        let json = r#"{"user": {"name": "Alice", "age": 30}, "tags": ["python", "ml"]}"#;
        let jsonb_bytes = create_test_jsonb(json);

        let mut binary_builder = LargeBinaryBuilder::new();
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_null();

        let jsonb_array = Arc::new(binary_builder.finish());
        let path_array = Arc::new(StringArray::from(vec![
            Some("$.user.name"),
            Some("$.user.age"),
            Some("$.tags[*]"),
            Some("$.user.name"),
        ]));

        let result = json_extract_impl(&[jsonb_array, path_array])?;
        let string_array = result.as_any().downcast_ref::<StringArray>().unwrap();

        assert_eq!(string_array.len(), 4);
        assert_eq!(string_array.value(0), "\"Alice\"");
        assert_eq!(string_array.value(1), "30");
        assert_eq!(string_array.value(2), "[\"python\",\"ml\"]");
        assert!(string_array.is_null(3));

        Ok(())
    }

    #[tokio::test]
    async fn test_json_exists_udf() -> Result<()> {
        let json = r#"{"user": {"name": "Alice", "age": 30}, "tags": ["rust", "json"]}"#;
        let jsonb_bytes = create_test_jsonb(json);

        let mut binary_builder = LargeBinaryBuilder::new();
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_null();

        let jsonb_array = Arc::new(binary_builder.finish());
        let path_array = Arc::new(StringArray::from(vec![
            Some("$.user.name"),
            Some("$.user.email"),
            Some("$.tags"),
            Some("$.any"),
        ]));

        let result = json_exists_impl(&[jsonb_array, path_array])?;
        let bool_array = result.as_any().downcast_ref::<BooleanArray>().unwrap();

        assert_eq!(bool_array.len(), 4);
        assert!(bool_array.value(0));
        assert!(!bool_array.value(1));
        assert!(bool_array.value(2));
        assert!(bool_array.is_null(3));

        Ok(())
    }

    #[tokio::test]
    async fn test_json_get_string_udf() -> Result<()> {
        // Test valid string conversions
        let json = r#"{"str": "hello", "num": 123, "bool": true, "null": null, "obj": {"a": 1}, "arr": [1]}"#;
        let jsonb_bytes = create_test_jsonb(json);

        let mut binary_builder = LargeBinaryBuilder::new();
        for _ in 0..6 {
            binary_builder.append_value(&jsonb_bytes);
        }

        let jsonb_array = Arc::new(binary_builder.finish());
        let key_array = Arc::new(StringArray::from(vec![
            Some("str"),
            Some("num"),
            Some("bool"),
            Some("null"),
            Some("obj"),
            Some("arr"),
        ]));

        let result = json_get_string_impl(&[jsonb_array, key_array])?;
        let string_array = result.as_any().downcast_ref::<StringArray>().unwrap();

        assert_eq!(string_array.len(), 6);
        assert_eq!(string_array.value(0), "hello");
        assert_eq!(string_array.value(1), "123");
        assert_eq!(string_array.value(2), "true");
        assert!(string_array.is_null(3));
        // Containers do not convert to a string, so they are null.
        assert!(string_array.is_null(4));
        assert!(string_array.is_null(5));

        Ok(())
    }

    #[tokio::test]
    async fn test_json_get_int_udf() -> Result<()> {
        let json =
            r#"{"int": 42, "str_num": "99", "bool": true, "0": 7, "str_bad": "abc", "arr": [1]}"#;
        let jsonb_bytes = create_test_jsonb(json);

        let mut binary_builder = LargeBinaryBuilder::new();
        for _ in 0..6 {
            binary_builder.append_value(&jsonb_bytes);
        }

        let jsonb_array = Arc::new(binary_builder.finish());
        let key_array = Arc::new(StringArray::from(vec![
            Some("int"),
            Some("str_num"),
            Some("bool"),
            Some("0"),
            Some("str_bad"),
            Some("arr"),
        ]));

        let result = json_get_int_impl(&[jsonb_array, key_array])?;
        let int_array = result.as_any().downcast_ref::<Int64Array>().unwrap();

        assert_eq!(int_array.len(), 6);
        assert_eq!(int_array.value(0), 42);
        assert_eq!(int_array.value(1), 99);
        assert_eq!(int_array.value(2), 1); // jsonb converts true to 1
        assert_eq!(int_array.value(3), 7);
        // Non-castable values are null per the documented "null if conversion
        // fails" contract, instead of failing the whole query.
        assert!(int_array.is_null(4));
        assert!(int_array.is_null(5));

        Ok(())
    }

    #[tokio::test]
    async fn test_json_get_float_udf() -> Result<()> {
        let json = r#"{
            "float_decimal": 1.5,
            "float_neg": -2.5,
            "float_int_value": 1.0,
            "float_exp": 1e2,
            "int_pos": 42,
            "int_neg": -7,
            "big_int": 9223372036854775808,
            "str_num": "3.5",
            "str_bad": "not_a_number",
            "bool_true": true,
            "null_val": null
        }"#;
        let jsonb_bytes = create_test_jsonb(json);

        let mut binary_builder = LargeBinaryBuilder::new();
        for _ in 0..12 {
            binary_builder.append_value(&jsonb_bytes);
        }
        let jsonb_array = Arc::new(binary_builder.finish());
        let key_array = Arc::new(StringArray::from(vec![
            Some("float_decimal"),
            Some("float_neg"),
            Some("float_int_value"),
            Some("float_exp"),
            Some("int_pos"),
            Some("int_neg"),
            Some("big_int"),
            Some("str_num"),
            Some("str_bad"),
            Some("bool_true"),
            Some("null_val"),
            Some("missing"),
        ]));

        let result = json_get_float_impl(&[jsonb_array, key_array])?;
        let float_array = result.as_any().downcast_ref::<Float64Array>().unwrap();

        assert_eq!(float_array.len(), 12);
        assert_eq!(float_array.value(0), 1.5);
        assert_eq!(float_array.value(1), -2.5);
        assert_eq!(float_array.value(2), 1.0);
        assert_eq!(float_array.value(3), 100.0);
        assert_eq!(float_array.value(4), 42.0);
        assert_eq!(float_array.value(5), -7.0);
        // 2^63 is exactly representable in f64.
        assert_eq!(float_array.value(6), 9223372036854775808.0);
        assert_eq!(float_array.value(7), 3.5);
        // Non-castable strings are null, not a query-fatal error.
        assert!(float_array.is_null(8));
        assert_eq!(float_array.value(9), 1.0); // jsonb converts true to 1.0
        assert!(float_array.is_null(10));
        assert!(float_array.is_null(11));

        Ok(())
    }

    #[tokio::test]
    async fn test_json_get_bool_udf() -> Result<()> {
        let json = r#"{"bool_true": true, "bool_false": false, "str_true": "true", "str_false": "false", "num_one": 1, "num_zero": 0, "num_float": 2.5, "str_one": "1", "str_maybe": "maybe", "float_zero": 0.0}"#;
        let jsonb_bytes = create_test_jsonb(json);

        let mut binary_builder = LargeBinaryBuilder::new();
        for _ in 0..10 {
            binary_builder.append_value(&jsonb_bytes);
        }

        let jsonb_array = Arc::new(binary_builder.finish());
        let key_array = Arc::new(StringArray::from(vec![
            Some("bool_true"),
            Some("bool_false"),
            Some("str_true"),
            Some("str_false"),
            Some("num_one"),
            Some("num_zero"),
            Some("num_float"),
            Some("str_one"),
            Some("str_maybe"),
            Some("float_zero"),
        ]));

        let result = json_get_bool_impl(&[jsonb_array, key_array])?;
        let bool_array = result.as_any().downcast_ref::<BooleanArray>().unwrap();

        assert_eq!(bool_array.len(), 10);
        assert!(bool_array.value(0));
        assert!(!bool_array.value(1));
        assert!(bool_array.value(2)); // "true" string converts to true
        assert!(!bool_array.value(3)); // "false" string converts to false
        // Numbers coerce per the documented contract: 0 -> false, non-zero -> true.
        assert!(bool_array.value(4));
        assert!(!bool_array.value(5));
        assert!(bool_array.value(6));
        // Non-boolean strings are null, not a query-fatal error.
        assert!(bool_array.is_null(7));
        assert!(bool_array.is_null(8));
        // A float zero is zero too, not only the integer 0.
        assert!(!bool_array.value(9));

        Ok(())
    }

    #[tokio::test]
    async fn test_json_array_contains_udf() -> Result<()> {
        let json = r#"{"tags": ["rust", "json", "database"], "nums": [1, 2, 3], "esc": ["say \"hi\"", "a\\b"]}"#;
        let jsonb_bytes = create_test_jsonb(json);

        let cases = [
            ("$.tags", "rust", true),
            ("$.tags", "python", false),
            ("$.nums", "2", true),
            // A needle written as JSON text, with its quotes, still matches.
            ("$.tags", "\"rust\"", true),
            // Needles containing quotes or backslashes match by content, not
            // only by their re-quoted serialization.
            ("$.esc", "say \"hi\"", true),
            ("$.esc", "a\\b", true),
        ];
        let mut binary_builder = LargeBinaryBuilder::new();
        for _ in &cases {
            binary_builder.append_value(&jsonb_bytes);
        }
        binary_builder.append_null();

        let jsonb_array = Arc::new(binary_builder.finish());
        let path_array = Arc::new(StringArray::from_iter(
            cases.iter().map(|c| Some(c.0)).chain([Some("$.tags")]),
        ));
        let value_array = Arc::new(StringArray::from_iter(
            cases.iter().map(|c| Some(c.1)).chain([Some("any")]),
        ));

        let result = json_array_contains_impl(&[jsonb_array, path_array, value_array])?;
        let bool_array = result.as_any().downcast_ref::<BooleanArray>().unwrap();

        assert_eq!(bool_array.len(), cases.len() + 1);
        for (i, (path, needle, expected)) in cases.iter().enumerate() {
            assert_eq!(bool_array.value(i), *expected, "{path} contains {needle}");
        }
        assert!(bool_array.is_null(cases.len()));

        Ok(())
    }

    #[tokio::test]
    async fn test_json_array_length_udf() -> Result<()> {
        let json =
            r#"{"empty": [], "tags": ["a", "b", "c"], "nested": {"arr": [1, 2]}, "name": "x"}"#;
        let jsonb_bytes = create_test_jsonb(json);

        let mut binary_builder = LargeBinaryBuilder::new();
        for _ in 0..5 {
            binary_builder.append_value(&jsonb_bytes);
        }
        binary_builder.append_null();

        let jsonb_array = Arc::new(binary_builder.finish());
        let path_array = Arc::new(StringArray::from(vec![
            Some("$.empty"),
            Some("$.tags"),
            Some("$.nested.arr"),
            Some("$.nested"),
            Some("$.name"),
            Some("$.any"),
        ]));

        let result = json_array_length_impl(&[jsonb_array, path_array])?;
        let int_array = result.as_any().downcast_ref::<Int64Array>().unwrap();

        assert_eq!(int_array.len(), 6);
        assert_eq!(int_array.value(0), 0);
        assert_eq!(int_array.value(1), 3);
        assert_eq!(int_array.value(2), 2);
        // A path that resolves to a non-array (object or scalar) is null, not
        // 0, so `json_array_length(...) = 0` only matches empty arrays.
        assert!(int_array.is_null(3));
        assert!(int_array.is_null(4));
        assert!(int_array.is_null(5));

        Ok(())
    }

    #[tokio::test]
    async fn test_json_extract_with_type() -> Result<()> {
        use arrow_array::StructArray;
        use arrow_array::UInt8Array;

        let cases: &[(&str, JsonbType)] = &[
            (r#"{"v": 1}"#, JsonbType::Int64),
            (r#"{"v": 0}"#, JsonbType::Int64),
            (r#"{"v": -42}"#, JsonbType::Int64),
            (r#"{"v": 9223372036854775807}"#, JsonbType::Int64), // i64::MAX
            (r#"{"v": 9223372036854775808}"#, JsonbType::Float64), // i64::MAX + 1
            (r#"{"v": 1.0}"#, JsonbType::Float64),
            (r#"{"v": 2.7}"#, JsonbType::Float64),
            (r#"{"v": 1.5}"#, JsonbType::Float64),
            (r#"{"v": -1.5}"#, JsonbType::Float64),
            (r#"{"v": 1e2}"#, JsonbType::Float64),
        ];

        for (json, expected) in cases {
            let bytes = create_test_jsonb(json);
            let mut binary_builder = LargeBinaryBuilder::new();
            binary_builder.append_value(&bytes);
            let jsonb_array: ArrayRef = Arc::new(binary_builder.finish());
            let path_array: ArrayRef = Arc::new(StringArray::from(vec![Some("$.v")]));

            let result = json_extract_with_type_impl(&[jsonb_array, path_array])?;
            let struct_array = result.as_any().downcast_ref::<StructArray>().unwrap();
            let type_tags = struct_array
                .column_by_name("type_tag")
                .unwrap()
                .as_any()
                .downcast_ref::<UInt8Array>()
                .unwrap();
            assert_eq!(type_tags.value(0), expected.as_u8());
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_json_extract_with_wildcard() -> Result<()> {
        use arrow_array::StructArray;
        use arrow_array::UInt8Array;

        let json = r#"{"items": [{"price": 1}, {"price": 2}]}"#;
        let jsonb_bytes = create_test_jsonb(json);

        let mut binary_builder = LargeBinaryBuilder::new();
        binary_builder.append_value(&jsonb_bytes);

        let jsonb_array: ArrayRef = Arc::new(binary_builder.finish());
        let path_array: ArrayRef = Arc::new(StringArray::from(vec![Some("$.items[*].price")]));

        let result = json_extract_with_type_impl(&[jsonb_array, path_array])?;
        let struct_array = result.as_any().downcast_ref::<StructArray>().unwrap();
        let values = struct_array
            .column_by_name("value")
            .unwrap()
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap();
        let type_tags = struct_array
            .column_by_name("type_tag")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap();

        assert_eq!(
            lance_jsonb::RawJsonb::new(values.value(0)).to_string(),
            "[1,2]"
        );
        assert_eq!(type_tags.value(0), JsonbType::Array.as_u8());

        Ok(())
    }

    #[tokio::test]
    async fn test_json_array_access() -> Result<()> {
        let json = r#"["first", "second", "third"]"#;
        let jsonb_bytes = create_test_jsonb(json);

        let mut binary_builder = LargeBinaryBuilder::new();
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_value(&jsonb_bytes);
        binary_builder.append_value(&jsonb_bytes);

        let jsonb_array = Arc::new(binary_builder.finish());
        let key_array = Arc::new(StringArray::from(vec![
            Some("0"),
            Some("1"),
            Some("10"), // Out of bounds
        ]));

        let result = json_get_string_impl(&[jsonb_array, key_array])?;
        let string_array = result.as_any().downcast_ref::<StringArray>().unwrap();

        assert_eq!(string_array.len(), 3);
        assert_eq!(string_array.value(0), "first");
        assert_eq!(string_array.value(1), "second");
        assert!(string_array.is_null(2));

        Ok(())
    }

    #[tokio::test]
    async fn test_json_get_numeric_object_key() -> Result<()> {
        let obj_bytes = create_test_jsonb(r#"{"0": "from_object", "1": 42}"#);
        let arr_bytes = create_test_jsonb(r#"["zero", "one", "two"]"#);

        let mut binary_builder = LargeBinaryBuilder::new();
        binary_builder.append_value(&obj_bytes);
        binary_builder.append_value(&arr_bytes);
        binary_builder.append_value(&arr_bytes);

        let jsonb_array = Arc::new(binary_builder.finish());
        let key_array = Arc::new(StringArray::from(vec![
            Some("0"),   // Numeric key on object: looks up the "0" field.
            Some("0"),   // Numeric key on array: looks up index 0.
            Some("foo"), // Non-numeric key on array: no match.
        ]));

        let result = json_get_string_impl(&[jsonb_array, key_array])?;
        let string_array = result.as_any().downcast_ref::<StringArray>().unwrap();

        assert_eq!(string_array.len(), 3);
        assert_eq!(string_array.value(0), "from_object");
        assert_eq!(string_array.value(1), "zero");
        assert!(string_array.is_null(2));

        Ok(())
    }
}
