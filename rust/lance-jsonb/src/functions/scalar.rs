// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-FileCopyrightText: Copyright 2023 Datafuse Labs
// Adapted from databendlabs/jsonb at commit fba895c5ebe77ce2539e187f9c652f51cbf195c3.

// This file contains functions that specifically operate on JSONB scalar values.

use std::borrow::Cow;

use crate::RawJsonb;
use crate::core::JsonbItem;
use crate::core::JsonbItemType;
use crate::error::*;
use crate::extension::ExtensionValue;
use crate::number::Number;

impl RawJsonb<'_> {
    /// Checks if the JSONB value is null.
    ///
    /// This function determines whether the JSONB value represents a JSON `null`.
    ///
    /// # Arguments
    ///
    /// * `self` - The JSONB value.
    ///
    /// # Returns
    ///
    /// * `Ok(true)` if the value is null.
    /// * `Ok(false)` if the value is not null.
    /// * `Err(Error)` - If the input JSONB data is invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lance_jsonb::OwnedJsonb;
    ///
    /// let null_jsonb = "null".parse::<OwnedJsonb>().unwrap();
    /// let raw_jsonb = null_jsonb.as_raw();
    /// assert!(raw_jsonb.is_null().unwrap());
    ///
    /// let obj_jsonb = r#"{"a": 1}"#.parse::<OwnedJsonb>().unwrap();
    /// let raw_jsonb = obj_jsonb.as_raw();
    /// assert!(!raw_jsonb.is_null().unwrap());
    /// ```
    pub fn is_null(&self) -> Result<bool> {
        let jsonb_item_type = self.jsonb_item_type()?;
        Ok(matches!(jsonb_item_type, JsonbItemType::Null))
    }

    /// Checks if the JSONB value is a boolean.
    ///
    /// This function checks if the JSONB value represents a JSON boolean (`true` or `false`).
    ///
    /// # Arguments
    ///
    /// * `self` - The JSONB value.
    ///
    /// # Returns
    ///
    /// * `Ok(true)` - If the value is a boolean (`true` or `false`).
    /// * `Ok(false)` - If the value is not a boolean.
    /// * `Err(Error)` - If the JSONB data is invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lance_jsonb::OwnedJsonb;
    ///
    /// // Boolean values
    /// let true_jsonb = "true".parse::<OwnedJsonb>().unwrap();
    /// let raw_true = true_jsonb.as_raw();
    /// assert!(raw_true.is_boolean().unwrap());
    ///
    /// let false_jsonb = "false".parse::<OwnedJsonb>().unwrap();
    /// let raw_false = false_jsonb.as_raw();
    /// assert!(raw_false.is_boolean().unwrap());
    ///
    /// // Non-boolean values
    /// let num_jsonb = "1".parse::<OwnedJsonb>().unwrap();
    /// let raw_num = num_jsonb.as_raw();
    /// assert!(!raw_num.is_boolean().unwrap());
    ///
    /// let arr_jsonb = "[1, 2, 3]".parse::<OwnedJsonb>().unwrap();
    /// let raw_arr = arr_jsonb.as_raw();
    /// assert!(!raw_arr.is_boolean().unwrap());
    ///
    /// let obj_jsonb = r#"{"a": 1}"#.parse::<OwnedJsonb>().unwrap();
    /// let raw_obj = obj_jsonb.as_raw();
    /// assert!(!raw_obj.is_boolean().unwrap());
    ///
    /// let null_jsonb = "null".parse::<OwnedJsonb>().unwrap();
    /// let raw_null = null_jsonb.as_raw();
    /// assert!(!raw_null.is_boolean().unwrap());
    ///
    /// // Invalid JSONB
    /// let invalid_jsonb = OwnedJsonb::from(vec![1, 2, 3, 4]);
    /// let invalid_raw_jsonb = invalid_jsonb.as_raw();
    /// let result = invalid_raw_jsonb.is_boolean();
    /// assert!(result.is_err());
    /// ```
    pub fn is_boolean(&self) -> Result<bool> {
        let jsonb_item_type = self.jsonb_item_type()?;
        Ok(matches!(jsonb_item_type, JsonbItemType::Boolean))
    }

    /// Extracts a boolean value from a JSONB value.
    ///
    /// This function attempts to extract a boolean value (`true` or `false`) from the JSONB value.
    /// If the JSONB value is a boolean, the corresponding boolean value is returned, and return `None` otherwise.
    ///
    /// # Arguments
    ///
    /// * `self` - The JSONB value.
    ///
    /// # Returns
    ///
    /// * `Ok(Some(true))` - If the value is JSON `true`.
    /// * `Ok(Some(false))` - If the value is JSON `false`.
    /// * `Ok(None)` - If the value is not a boolean.
    /// * `Err(Error)` - If the JSONB data is invalid.
    pub(crate) fn as_bool(&self) -> Result<Option<bool>> {
        let jsonb_item = JsonbItem::from_raw_jsonb(*self)?;
        match jsonb_item {
            JsonbItem::Boolean(v) => Ok(Some(v)),
            _ => Ok(None),
        }
    }

    /// Converts a JSONB value to a boolean.
    ///
    /// This function attempts to convert a JSONB value to a boolean. It prioritizes extracting a boolean value directly if possible.
    /// If the value is a string, it converts the string to lowercase and checks if it's "true" or "false". Otherwise, it returns an error.
    ///
    /// # Arguments
    ///
    /// * `self` - The JSONB value.
    ///
    /// # Returns
    ///
    /// * `Ok(true)` - If the value is JSON `true` or a string that is "true" (case-insensitive).
    /// * `Ok(false)` - If the value is JSON `false` or a string that is "false" (case-insensitive).
    /// * `Err(Error::InvalidCast)` - If the value cannot be converted to a boolean.
    /// * `Err(Error)` - If the JSONB data is invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lance_jsonb::OwnedJsonb;
    ///
    /// // Boolean values
    /// let true_jsonb = "true".parse::<OwnedJsonb>().unwrap();
    /// assert!(true_jsonb.as_raw().to_bool().unwrap());
    ///
    /// let false_jsonb = "false".parse::<OwnedJsonb>().unwrap();
    /// assert!(!false_jsonb.as_raw().to_bool().unwrap());
    ///
    /// // String representations of booleans
    /// let true_str = r#""true""#.parse::<OwnedJsonb>().unwrap();
    /// assert!(true_str.as_raw().to_bool().unwrap());
    ///
    /// let false_str = r#""false""#.parse::<OwnedJsonb>().unwrap();
    /// assert!(!false_str.as_raw().to_bool().unwrap());
    ///
    /// let true_str_lowercase = r#""TRUE""#.parse::<OwnedJsonb>().unwrap();
    /// assert!(true_str_lowercase.as_raw().to_bool().unwrap());
    ///
    /// // Invalid conversions
    /// let num_jsonb = "1".parse::<OwnedJsonb>().unwrap();
    /// let result = num_jsonb.as_raw().to_bool();
    /// assert!(result.is_err());
    ///
    /// let arr_jsonb = "[1, 2, 3]".parse::<OwnedJsonb>().unwrap();
    /// let result = arr_jsonb.as_raw().to_bool();
    /// assert!(result.is_err());
    ///
    /// let obj_jsonb = r#"{"a": 1}"#.parse::<OwnedJsonb>().unwrap();
    /// let result = obj_jsonb.as_raw().to_bool();
    /// assert!(result.is_err());
    ///
    /// let null_jsonb = "null".parse::<OwnedJsonb>().unwrap();
    /// let result = null_jsonb.as_raw().to_bool();
    /// assert!(result.is_err());
    ///
    /// let invalid_str = r#""maybe""#.parse::<OwnedJsonb>().unwrap();
    /// let result = invalid_str.as_raw().to_bool();
    /// assert!(result.is_err());
    ///
    /// // Invalid JSONB
    /// let invalid_jsonb = OwnedJsonb::from(vec![1, 2, 3, 4]);
    /// let invalid_raw_jsonb = invalid_jsonb.as_raw();
    /// let result = invalid_raw_jsonb.to_bool();
    /// assert!(result.is_err());
    /// ```
    pub fn to_bool(&self) -> Result<bool> {
        let jsonb_item = JsonbItem::from_raw_jsonb(*self)?;
        match jsonb_item {
            JsonbItem::Boolean(v) => {
                return Ok(v);
            }
            JsonbItem::String(s) => {
                if s.eq_ignore_ascii_case("true") || s.eq_ignore_ascii_case("yes") {
                    return Ok(true);
                } else if s.eq_ignore_ascii_case("false") || s.eq_ignore_ascii_case("no") {
                    return Ok(false);
                }
            }
            _ => {}
        }
        Err(Error::InvalidCast)
    }

    /// Checks if the JSONB value is a number.
    ///
    /// This function checks if the JSONB value represents a JSON number.
    ///
    /// # Arguments
    ///
    /// * `self` - The JSONB value.
    ///
    /// # Returns
    ///
    /// * `Ok(true)` - If the value is a number.
    /// * `Ok(false)` - If the value is not a number.
    /// * `Err(Error)` - If the JSONB data is invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lance_jsonb::OwnedJsonb;
    ///
    /// // Number values
    /// let num_jsonb = "123.45".parse::<OwnedJsonb>().unwrap();
    /// let raw_num = num_jsonb.as_raw();
    /// assert!(raw_num.is_number().unwrap());
    ///
    /// let num_jsonb = "123".parse::<OwnedJsonb>().unwrap();
    /// let raw_num = num_jsonb.as_raw();
    /// assert!(raw_num.is_number().unwrap());
    ///
    /// let num_jsonb = "-123.45".parse::<OwnedJsonb>().unwrap();
    /// let raw_num = num_jsonb.as_raw();
    /// assert!(raw_num.is_number().unwrap());
    ///
    /// // Non-number values
    /// let bool_jsonb = "true".parse::<OwnedJsonb>().unwrap();
    /// let raw_bool = bool_jsonb.as_raw();
    /// assert!(!raw_bool.is_number().unwrap());
    ///
    /// let str_jsonb = r#""hello""#.parse::<OwnedJsonb>().unwrap();
    /// let raw_str = str_jsonb.as_raw();
    /// assert!(!raw_str.is_number().unwrap());
    ///
    /// let arr_jsonb = "[1, 2, 3]".parse::<OwnedJsonb>().unwrap();
    /// let raw_arr = arr_jsonb.as_raw();
    /// assert!(!raw_arr.is_number().unwrap());
    ///
    /// let obj_jsonb = r#"{"a": 1}"#.parse::<OwnedJsonb>().unwrap();
    /// let raw_obj = obj_jsonb.as_raw();
    /// assert!(!raw_obj.is_number().unwrap());
    ///
    /// let null_jsonb = "null".parse::<OwnedJsonb>().unwrap();
    /// let raw_null = null_jsonb.as_raw();
    /// assert!(!raw_null.is_number().unwrap());
    ///
    /// // Invalid JSONB
    /// let invalid_jsonb = OwnedJsonb::from(vec![1, 2, 3, 4]);
    /// let invalid_raw_jsonb = invalid_jsonb.as_raw();
    /// let result = invalid_raw_jsonb.is_number();
    /// assert!(result.is_err());
    /// ```
    pub fn is_number(&self) -> Result<bool> {
        let jsonb_item_type = self.jsonb_item_type()?;
        Ok(matches!(jsonb_item_type, JsonbItemType::Number))
    }

    /// Extracts a number from a JSONB value.
    ///
    /// This function attempts to extract a number from the JSONB value.
    /// If the JSONB value is a number, it returns the number; otherwise, it returns `None`.
    ///
    /// # Arguments
    ///
    /// * `self` - The JSONB value.
    ///
    /// # Returns
    ///
    /// * `Ok(Some(Number))` - If the value is a number, the extracted number.
    /// * `Ok(None)` - If the value is not a number.
    /// * `Err(Error)` - If the JSONB data is invalid or if the number cannot be decoded.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lance_jsonb::Number;
    /// use lance_jsonb::OwnedJsonb;
    /// use lance_jsonb::RawJsonb;
    ///
    /// // Number value
    /// let num_jsonb = "123.45".parse::<OwnedJsonb>().unwrap();
    /// let raw_num = num_jsonb.as_raw();
    /// assert_eq!(raw_num.as_number().unwrap(), Some(Number::Float64(123.45)));
    ///
    /// let num_jsonb = "-123".parse::<OwnedJsonb>().unwrap();
    /// let raw_num = num_jsonb.as_raw();
    /// assert_eq!(raw_num.as_number().unwrap(), Some(Number::Int64(-123)));
    ///
    /// // Non-number values
    /// let bool_jsonb = "true".parse::<OwnedJsonb>().unwrap();
    /// let raw_bool = bool_jsonb.as_raw();
    /// assert_eq!(raw_bool.as_number().unwrap(), None);
    ///
    /// let str_jsonb = r#""hello""#.parse::<OwnedJsonb>().unwrap();
    /// let raw_str = str_jsonb.as_raw();
    /// assert_eq!(raw_str.as_number().unwrap(), None);
    ///
    /// let arr_jsonb = "[1, 2, 3]".parse::<OwnedJsonb>().unwrap();
    /// let raw_arr = arr_jsonb.as_raw();
    /// assert_eq!(raw_arr.as_number().unwrap(), None);
    ///
    /// let obj_jsonb = r#"{"a": 1}"#.parse::<OwnedJsonb>().unwrap();
    /// let raw_obj = obj_jsonb.as_raw();
    /// assert_eq!(raw_obj.as_number().unwrap(), None);
    ///
    /// let null_jsonb = "null".parse::<OwnedJsonb>().unwrap();
    /// let raw_null = null_jsonb.as_raw();
    /// assert_eq!(raw_null.as_number().unwrap(), None);
    ///
    /// // Invalid JSONB
    /// let invalid_jsonb = OwnedJsonb::from(vec![1, 2, 3, 4]);
    /// let invalid_raw_jsonb = invalid_jsonb.as_raw();
    /// let result = invalid_raw_jsonb.as_number();
    /// assert!(result.is_err());
    ///
    /// // Invalid Number (corrupted data)
    /// let corrupted_num_jsonb = OwnedJsonb::from(vec![10, 0, 0, 0, 16, 0, 0, 0, 0, 1]);
    /// let corrupted_raw_num_jsonb = corrupted_num_jsonb.as_raw();
    /// let result = corrupted_raw_num_jsonb.as_number();
    /// assert!(result.is_err()); // Decodes should return Err
    /// ```
    pub fn as_number(&self) -> Result<Option<Number>> {
        let jsonb_item = JsonbItem::from_raw_jsonb(*self)?;
        match jsonb_item {
            JsonbItem::Number(num) => {
                let value = num.as_number()?;
                Ok(Some(value))
            }
            _ => Ok(None),
        }
    }

    /// Checks whether the JSONB value is an exact `i64`.
    ///
    /// Decimal and floating-point numbers must already be integral.
    ///
    /// # Arguments
    ///
    /// * `self` - The JSONB value.
    ///
    /// # Returns
    ///
    /// * `Ok(true)` - If the value is an exact `i64`.
    /// * `Ok(false)` - Otherwise.
    /// * `Err(Error)` - If the JSONB data is invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lance_jsonb::OwnedJsonb;
    ///
    /// // i64 values
    /// let i64_jsonb = "123456789012345678".parse::<OwnedJsonb>().unwrap();
    /// let raw_i64 = i64_jsonb.as_raw();
    /// assert!(raw_i64.is_i64().unwrap());
    ///
    /// let i64_jsonb = "-123456789012345678.0".parse::<OwnedJsonb>().unwrap();
    /// let raw_i64 = i64_jsonb.as_raw();
    /// assert!(raw_i64.is_i64().unwrap());
    ///
    /// let float_jsonb = "1.5e0".parse::<OwnedJsonb>().unwrap();
    /// let raw_float = float_jsonb.as_raw();
    /// assert!(!raw_float.is_i64().unwrap());
    ///
    /// // Out-of-range values
    /// let float_jsonb = "1e100".parse::<OwnedJsonb>().unwrap();
    /// let raw_float = float_jsonb.as_raw();
    /// assert!(!raw_float.is_i64().unwrap());
    ///
    /// let str_jsonb = r#""hello""#.parse::<OwnedJsonb>().unwrap();
    /// let raw_str = str_jsonb.as_raw();
    /// assert!(!raw_str.is_i64().unwrap());
    ///
    /// // Invalid JSONB
    /// let invalid_jsonb = OwnedJsonb::from(vec![1, 2, 3, 4]);
    /// let invalid_raw_jsonb = invalid_jsonb.as_raw();
    /// let result = invalid_raw_jsonb.is_i64();
    /// assert!(result.is_err());
    /// ```
    pub fn is_i64(&self) -> Result<bool> {
        self.as_i64().map(|v| v.is_some())
    }

    /// Extracts an exact `i64` from a JSONB value.
    ///
    /// Decimal and floating-point numbers are accepted only when they already
    /// represent an integer.
    ///
    /// # Arguments
    ///
    /// * `self` - The JSONB value.
    ///
    /// # Returns
    ///
    /// * `Ok(Some(i64))` - If the value is an exact `i64`.
    /// * `Ok(None)` - Otherwise.
    /// * `Err(Error)` - If the JSONB data is invalid.
    pub(crate) fn as_i64(&self) -> Result<Option<i64>> {
        let jsonb_item = JsonbItem::from_raw_jsonb(*self)?;
        match jsonb_item {
            JsonbItem::Number(num) => {
                let value = num.as_number()?;
                Ok(value.as_i64())
            }
            _ => Ok(None),
        }
    }

    /// Converts a JSONB value to `i64`.
    ///
    /// Numbers are rounded to the nearest integer. Booleans map to `1` and
    /// `0`. Strings are parsed as `i64`, or as `f64` and then rounded.
    ///
    /// # Arguments
    ///
    /// * `self` - The JSONB value.
    ///
    /// # Returns
    ///
    /// * `Ok(i64)` - The converted value.
    /// * `Err(Error::InvalidCast)` - If the value cannot be converted to an `i64`.
    /// * `Err(Error)` - If the JSONB data is invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lance_jsonb::OwnedJsonb;
    ///
    /// // Integer values
    /// let i64_jsonb = "123".parse::<OwnedJsonb>().unwrap();
    /// assert_eq!(i64_jsonb.as_raw().to_i64().unwrap(), 123);
    ///
    /// let i64_jsonb = "-42".parse::<OwnedJsonb>().unwrap();
    /// assert_eq!(i64_jsonb.as_raw().to_i64().unwrap(), -42);
    ///
    /// // Boolean values
    /// let true_jsonb = "true".parse::<OwnedJsonb>().unwrap();
    /// assert_eq!(true_jsonb.as_raw().to_i64().unwrap(), 1);
    ///
    /// let false_jsonb = "false".parse::<OwnedJsonb>().unwrap();
    /// assert_eq!(false_jsonb.as_raw().to_i64().unwrap(), 0);
    ///
    /// // String representation of an integer
    /// let str_jsonb = r#""123""#.parse::<OwnedJsonb>().unwrap();
    /// assert_eq!(str_jsonb.as_raw().to_i64().unwrap(), 123);
    ///
    /// // Decimal and float numbers are rounded before conversion
    /// let decimal_jsonb = "123.5".parse::<OwnedJsonb>().unwrap();
    /// assert_eq!(decimal_jsonb.as_raw().to_i64().unwrap(), 124);
    ///
    /// let float_jsonb = "1.5e0".parse::<OwnedJsonb>().unwrap();
    /// assert_eq!(float_jsonb.as_raw().to_i64().unwrap(), 2);
    ///
    /// let str_jsonb = r#""1.5""#.parse::<OwnedJsonb>().unwrap();
    /// assert_eq!(str_jsonb.as_raw().to_i64().unwrap(), 2);
    ///
    /// // Invalid conversions
    /// let float_jsonb = "1e100".parse::<OwnedJsonb>().unwrap();
    /// let result = float_jsonb.as_raw().to_i64();
    /// assert!(result.is_err());
    ///
    /// let arr_jsonb = "[1, 2, 3]".parse::<OwnedJsonb>().unwrap();
    /// let result = arr_jsonb.as_raw().to_i64();
    /// assert!(result.is_err());
    ///
    /// let obj_jsonb = r#"{"a": 1}"#.parse::<OwnedJsonb>().unwrap();
    /// let result = obj_jsonb.as_raw().to_i64();
    /// assert!(result.is_err());
    ///
    /// let null_jsonb = "null".parse::<OwnedJsonb>().unwrap();
    /// let result = null_jsonb.as_raw().to_i64();
    /// assert!(result.is_err());
    ///
    /// let invalid_str_jsonb = r#""abc""#.parse::<OwnedJsonb>().unwrap();
    /// let result = invalid_str_jsonb.as_raw().to_i64();
    /// assert!(result.is_err());
    ///
    /// // Invalid JSONB
    /// let invalid_jsonb = OwnedJsonb::from(vec![1, 2, 3, 4]);
    /// let invalid_raw_jsonb = invalid_jsonb.as_raw();
    /// let result = invalid_raw_jsonb.to_i64();
    /// assert!(result.is_err());
    /// ```
    pub fn to_i64(&self) -> Result<i64> {
        let jsonb_item = JsonbItem::from_raw_jsonb(*self)?;
        match jsonb_item {
            JsonbItem::Boolean(v) => {
                if v {
                    return Ok(1_i64);
                } else {
                    return Ok(0_i64);
                }
            }
            JsonbItem::Number(num) => {
                let value = num.as_number()?;
                if let Some(v) = value.to_i64() {
                    return Ok(v);
                }
            }
            JsonbItem::String(s) => {
                if let Some(v) = parse_string_to_i64(&s) {
                    return Ok(v);
                }
            }
            _ => {}
        }
        Err(Error::InvalidCast)
    }

    /// Converts a JSONB value to an f64 floating-point number.
    ///
    /// This function attempts to convert a JSONB value to an `f64` floating-point number.
    /// It prioritizes direct conversion from a number if possible.
    /// If the value is a boolean, it's converted to 1.0 (for `true`) or 0.0 (for `false`).
    /// If the value is a string that can be parsed as an `f64`, that parsed value is returned.
    /// Otherwise, an error is returned.
    ///
    /// # Arguments
    ///
    /// * `self` - The JSONB value.
    ///
    /// # Returns
    ///
    /// * `Ok(f64)` - The `f64` representation of the JSONB value.
    /// * `Err(Error::InvalidCast)` - If the value cannot be converted to an `f64`
    ///   (e.g., it's an array, an object, a string that is not a valid number, or a null value).
    /// * `Err(Error)` - If the JSONB data is invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lance_jsonb::OwnedJsonb;
    ///
    /// // f64 values
    /// let f64_jsonb = "123.45".parse::<OwnedJsonb>().unwrap();
    /// assert_eq!(f64_jsonb.as_raw().to_f64().unwrap(), 123.45);
    ///
    /// let int_jsonb = "123".parse::<OwnedJsonb>().unwrap();
    /// assert_eq!(int_jsonb.as_raw().to_f64().unwrap(), 123.0);
    ///
    /// // Boolean values
    /// let true_jsonb = "true".parse::<OwnedJsonb>().unwrap();
    /// assert_eq!(true_jsonb.as_raw().to_f64().unwrap(), 1.0);
    ///
    /// let false_jsonb = "false".parse::<OwnedJsonb>().unwrap();
    /// assert_eq!(false_jsonb.as_raw().to_f64().unwrap(), 0.0);
    ///
    /// // String representation of a number
    /// let str_jsonb = r#""123.45""#.parse::<OwnedJsonb>().unwrap();
    /// assert_eq!(str_jsonb.as_raw().to_f64().unwrap(), 123.45);
    ///
    /// // Invalid conversions
    /// let arr_jsonb = "[1, 2, 3]".parse::<OwnedJsonb>().unwrap();
    /// let result = arr_jsonb.as_raw().to_f64();
    /// assert!(result.is_err());
    ///
    /// let obj_jsonb = r#"{"a": 1}"#.parse::<OwnedJsonb>().unwrap();
    /// let result = obj_jsonb.as_raw().to_f64();
    /// assert!(result.is_err());
    ///
    /// let null_jsonb = "null".parse::<OwnedJsonb>().unwrap();
    /// let result = null_jsonb.as_raw().to_f64();
    /// assert!(result.is_err());
    ///
    /// let invalid_str_jsonb = r#""abc""#.parse::<OwnedJsonb>().unwrap();
    /// let result = invalid_str_jsonb.as_raw().to_f64();
    /// assert!(result.is_err());
    ///
    /// // Invalid JSONB
    /// let invalid_jsonb = OwnedJsonb::from(vec![1, 2, 3, 4]);
    /// let invalid_raw_jsonb = invalid_jsonb.as_raw();
    /// let result = invalid_raw_jsonb.to_f64();
    /// assert!(result.is_err());
    /// ```
    pub fn to_f64(&self) -> Result<f64> {
        let jsonb_item = JsonbItem::from_raw_jsonb(*self)?;
        match jsonb_item {
            JsonbItem::Boolean(v) => {
                if v {
                    return Ok(1_f64);
                } else {
                    return Ok(0_f64);
                }
            }
            JsonbItem::Number(num) => {
                let value = num.as_number()?;
                let v = value.as_f64();
                return Ok(v);
            }
            JsonbItem::String(s) => {
                if let Ok(v) = s.parse::<f64>() {
                    return Ok(v);
                }
            }
            _ => {}
        }
        Err(Error::InvalidCast)
    }

    /// Checks if the JSONB value is a string.
    ///
    /// This function checks if the JSONB value represents a JSON string.
    ///
    /// # Arguments
    ///
    /// * `self` - The JSONB value.
    ///
    /// # Returns
    ///
    /// * `Ok(true)` - If the value is a string.
    /// * `Ok(false)` - If the value is not a string.
    /// * `Err(Error)` - If the JSONB data is invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lance_jsonb::OwnedJsonb;
    ///
    /// // String value
    /// let str_jsonb = r#""hello""#.parse::<OwnedJsonb>().unwrap();
    /// let raw_str = str_jsonb.as_raw();
    /// assert!(raw_str.is_string().unwrap());
    ///
    /// // Non-string values
    /// let num_jsonb = "123".parse::<OwnedJsonb>().unwrap();
    /// let raw_num = num_jsonb.as_raw();
    /// assert!(!raw_num.is_string().unwrap());
    ///
    /// let bool_jsonb = "true".parse::<OwnedJsonb>().unwrap();
    /// let raw_bool = bool_jsonb.as_raw();
    /// assert!(!raw_bool.is_string().unwrap());
    ///
    /// let arr_jsonb = "[1, 2, 3]".parse::<OwnedJsonb>().unwrap();
    /// let raw_arr = arr_jsonb.as_raw();
    /// assert!(!raw_arr.is_string().unwrap());
    ///
    /// let obj_jsonb = r#"{"a": 1}"#.parse::<OwnedJsonb>().unwrap();
    /// let raw_obj = obj_jsonb.as_raw();
    /// assert!(!raw_obj.is_string().unwrap());
    ///
    /// let null_jsonb = "null".parse::<OwnedJsonb>().unwrap();
    /// let raw_null = null_jsonb.as_raw();
    /// assert!(!raw_null.is_string().unwrap());
    ///
    /// // Invalid JSONB
    /// let invalid_jsonb = OwnedJsonb::from(vec![1, 2, 3, 4]);
    /// let invalid_raw_jsonb = invalid_jsonb.as_raw();
    /// let result = invalid_raw_jsonb.is_string();
    /// assert!(result.is_err());
    /// ```
    pub fn is_string(&self) -> Result<bool> {
        let jsonb_item_type = self.jsonb_item_type()?;
        Ok(matches!(jsonb_item_type, JsonbItemType::String))
    }

    /// Extracts a string from a JSONB value.
    ///
    /// This function attempts to extract a string from the JSONB value.
    /// If the JSONB value is a string, it returns the string as a `Cow<'_, str>`.
    /// Otherwise, it returns `None`.
    ///
    /// # Arguments
    ///
    /// * `self` - The JSONB value.
    ///
    /// # Returns
    ///
    /// * `Ok(Some(Cow<'_, str>))` - If the value is a string, the extracted string.
    /// * `Ok(None)` - If the value is not a string (number, boolean, null, array, object).
    /// * `Err(Error)` - If the JSONB data is invalid or if the string is not valid UTF-8.
    pub(crate) fn as_str(&self) -> Result<Option<Cow<'_, str>>> {
        let jsonb_item = JsonbItem::from_raw_jsonb(*self)?;
        match jsonb_item {
            JsonbItem::String(s) => Ok(Some(s)),
            _ => Ok(None),
        }
    }

    /// Converts a JSONB value to a String.
    ///
    /// This function attempts to convert a JSONB value to a string representation.
    /// It prioritizes direct conversion from strings.
    /// Booleans are converted to "true" or "false", and numbers are converted to their string representations.
    /// Other types (arrays, objects, null) will result in an error.
    ///
    /// # Arguments
    ///
    /// * `self` - The JSONB value.
    ///
    /// # Returns
    ///
    /// * `Ok(String)` - The string representation of the JSONB value.
    /// * `Err(Error::InvalidCast)` - If the JSONB value cannot be converted to a string (e.g., it's an array, an object, or a null value).
    /// * `Err(Error)` - If the JSONB data is invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lance_jsonb::OwnedJsonb;
    ///
    /// // String value
    /// let str_jsonb = r#""hello""#.parse::<OwnedJsonb>().unwrap();
    /// assert_eq!(str_jsonb.as_raw().to_str().unwrap(), "hello");
    ///
    /// // Number value
    /// let num_jsonb = "123.45".parse::<OwnedJsonb>().unwrap();
    /// assert_eq!(num_jsonb.as_raw().to_str().unwrap(), "123.45");
    ///
    /// // Boolean values
    /// let true_jsonb = "true".parse::<OwnedJsonb>().unwrap();
    /// assert_eq!(true_jsonb.as_raw().to_str().unwrap(), "true");
    ///
    /// let false_jsonb = "false".parse::<OwnedJsonb>().unwrap();
    /// assert_eq!(false_jsonb.as_raw().to_str().unwrap(), "false");
    ///
    /// // Invalid conversions
    /// let arr_jsonb = "[1, 2, 3]".parse::<OwnedJsonb>().unwrap();
    /// let result = arr_jsonb.as_raw().to_str();
    /// assert!(result.is_err());
    ///
    /// let obj_jsonb = r#"{"a": 1}"#.parse::<OwnedJsonb>().unwrap();
    /// let result = obj_jsonb.as_raw().to_str();
    /// assert!(result.is_err());
    ///
    /// let null_jsonb = "null".parse::<OwnedJsonb>().unwrap();
    /// let result = null_jsonb.as_raw().to_str();
    /// assert!(result.is_err());
    ///
    /// // Invalid JSONB
    /// let invalid_jsonb = OwnedJsonb::from(vec![1, 2, 3, 4]);
    /// let invalid_raw_jsonb = invalid_jsonb.as_raw();
    /// let result = invalid_raw_jsonb.to_str();
    /// assert!(result.is_err());
    /// ```
    pub fn to_str(&self) -> Result<String> {
        let jsonb_item = JsonbItem::from_raw_jsonb(*self)?;
        match jsonb_item {
            JsonbItem::Boolean(v) => {
                if v {
                    Ok("true".to_string())
                } else {
                    Ok("false".to_string())
                }
            }
            JsonbItem::Number(num) => {
                let value = num.as_number()?;
                Ok(format!("{}", value))
            }
            JsonbItem::String(s) => Ok(s.to_string()),
            _ => Err(Error::InvalidCast),
        }
    }

    /// Checks if the JSONB value is an array.
    ///
    /// This function checks if the JSONB value represents a JSON array.
    ///
    /// # Arguments
    ///
    /// * `self` - The JSONB value.
    ///
    /// # Returns
    ///
    /// * `Ok(true)` - If the value is an array.
    /// * `Ok(false)` - If the value is not an array.
    /// * `Err(Error)` - If the JSONB data is invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lance_jsonb::OwnedJsonb;
    ///
    /// // Array value
    /// let arr_jsonb = "[1, 2, 3]".parse::<OwnedJsonb>().unwrap();
    /// let raw_arr = arr_jsonb.as_raw();
    /// assert!(raw_arr.is_array().unwrap());
    ///
    /// // Non-array values
    /// let num_jsonb = "123".parse::<OwnedJsonb>().unwrap();
    /// let raw_num = num_jsonb.as_raw();
    /// assert!(!raw_num.is_array().unwrap());
    ///
    /// let bool_jsonb = "true".parse::<OwnedJsonb>().unwrap();
    /// let raw_bool = bool_jsonb.as_raw();
    /// assert!(!raw_bool.is_array().unwrap());
    ///
    /// let str_jsonb = r#""hello""#.parse::<OwnedJsonb>().unwrap();
    /// let raw_str = str_jsonb.as_raw();
    /// assert!(!raw_str.is_array().unwrap());
    ///
    /// let obj_jsonb = r#"{"a": 1}"#.parse::<OwnedJsonb>().unwrap();
    /// let raw_obj = obj_jsonb.as_raw();
    /// assert!(!raw_obj.is_array().unwrap());
    ///
    /// let null_jsonb = "null".parse::<OwnedJsonb>().unwrap();
    /// let raw_null = null_jsonb.as_raw();
    /// assert!(!raw_null.is_array().unwrap());
    ///
    /// // Invalid JSONB
    /// let invalid_jsonb = OwnedJsonb::from(vec![1, 2, 3, 4]);
    /// let invalid_raw_jsonb = invalid_jsonb.as_raw();
    /// let result = invalid_raw_jsonb.is_array();
    /// assert!(result.is_err());
    /// ```
    pub fn is_array(&self) -> Result<bool> {
        let jsonb_item_type = self.jsonb_item_type()?;
        Ok(matches!(jsonb_item_type, JsonbItemType::Array(_)))
    }

    /// Checks if the JSONB value is an object.
    ///
    /// This function checks if the JSONB value represents a JSON object.
    ///
    /// # Arguments
    ///
    /// * `self` - The JSONB value.
    ///
    /// # Returns
    ///
    /// * `Ok(true)` - If the value is an object.
    /// * `Ok(false)` - If the value is not an object.
    /// * `Err(Error)` - If the JSONB data is invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lance_jsonb::OwnedJsonb;
    ///
    /// // Object value
    /// let obj_jsonb = r#"{"a": 1}"#.parse::<OwnedJsonb>().unwrap();
    /// let raw_obj = obj_jsonb.as_raw();
    /// assert!(raw_obj.is_object().unwrap());
    ///
    /// // Non-object values
    /// let num_jsonb = "123".parse::<OwnedJsonb>().unwrap();
    /// let raw_num = num_jsonb.as_raw();
    /// assert!(!raw_num.is_object().unwrap());
    ///
    /// let bool_jsonb = "true".parse::<OwnedJsonb>().unwrap();
    /// let raw_bool = bool_jsonb.as_raw();
    /// assert!(!raw_bool.is_object().unwrap());
    ///
    /// let str_jsonb = r#""hello""#.parse::<OwnedJsonb>().unwrap();
    /// let raw_str = str_jsonb.as_raw();
    /// assert!(!raw_str.is_object().unwrap());
    ///
    /// let arr_jsonb = "[1, 2, 3]".parse::<OwnedJsonb>().unwrap();
    /// let raw_arr = arr_jsonb.as_raw();
    /// assert!(!raw_arr.is_object().unwrap());
    ///
    /// let null_jsonb = "null".parse::<OwnedJsonb>().unwrap();
    /// let raw_null = null_jsonb.as_raw();
    /// assert!(!raw_null.is_object().unwrap());
    ///
    /// // Invalid JSONB
    /// let invalid_jsonb = OwnedJsonb::from(vec![1, 2, 3, 4]);
    /// let invalid_raw_jsonb = invalid_jsonb.as_raw();
    /// let result = invalid_raw_jsonb.is_object();
    /// assert!(result.is_err());
    /// ```
    pub fn is_object(&self) -> Result<bool> {
        let jsonb_item_type = self.jsonb_item_type()?;
        Ok(matches!(jsonb_item_type, JsonbItemType::Object(_)))
    }

    /// Extracts a extension value from a JSONB value.
    ///
    /// This function attempts to extract a extension value from the JSONB value.
    ///
    /// # Arguments
    ///
    /// * `self` - The JSONB value.
    ///
    /// # Returns
    ///
    /// * `Ok(Some(ExtensionValue))` - If the value is a extension value, the extracted extension value.
    /// * `Ok(None)` - If the value is not a extension value.
    /// * `Err(Error)` - If the JSONB data is invalid or if the extension value cannot be decoded.
    pub(crate) fn as_extension_value(&self) -> Result<Option<ExtensionValue<'_>>> {
        let jsonb_item = JsonbItem::from_raw_jsonb(*self)?;
        match jsonb_item {
            JsonbItem::Extension(ext) => {
                let val = ext.as_extension_value()?;
                Ok(Some(val))
            }
            _ => Ok(None),
        }
    }
}

fn parse_string_to_i64(s: &str) -> Option<i64> {
    if let Ok(v) = s.parse::<i64>() {
        return Some(v);
    }

    s.parse::<f64>()
        .ok()
        .and_then(|value| Number::Float64(value).to_i64())
}
