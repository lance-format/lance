// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-FileCopyrightText: Copyright 2023 Datafuse Labs
// Adapted from databendlabs/jsonb at commit fba895c5ebe77ce2539e187f9c652f51cbf195c3.

// This file contains functions that dealing with path-based access to JSONB data.

use std::borrow::Cow;

use crate::OwnedJsonb;
use crate::RawJsonb;
use crate::core::ArrayIterator;
use crate::error::*;

impl RawJsonb<'_> {
    /// Gets the element at the specified index in a JSONB array.
    ///
    /// If the JSONB value is an array, this function returns the element at the given `index` as an `OwnedJsonb`.
    /// If the `index` is out of bounds, it returns `Ok(None)`.
    /// If the JSONB value is not an array (e.g., it's an object or a scalar), this function also returns `Ok(None)`.
    ///
    /// # Arguments
    ///
    /// * `self` - The JSONB value.
    /// * `index` - The index of the desired element.
    ///
    /// # Returns
    ///
    /// * `Ok(Some(OwnedJsonb))` - The element at the specified index as an `OwnedJsonb` if the input is an array and the index is valid.
    /// * `Ok(None)` - If the input is not an array, or if the index is out of bounds.
    /// * `Err(Error)` - If an error occurred during decoding (e.g., invalid JSONB data).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lance_jsonb::OwnedJsonb;
    ///
    /// let arr_jsonb = r#"[1, "hello", {"a": 1}]"#.parse::<OwnedJsonb>().unwrap();
    /// let raw_jsonb = arr_jsonb.as_raw();
    ///
    /// let element0 = raw_jsonb.get_by_index(0).unwrap();
    /// assert_eq!(element0.unwrap().to_string(), "1");
    ///
    /// let element1 = raw_jsonb.get_by_index(1).unwrap();
    /// assert_eq!(element1.unwrap().to_string(), r#""hello""#);
    ///
    /// let element2 = raw_jsonb.get_by_index(2).unwrap();
    /// assert_eq!(element2.unwrap().to_string(), r#"{"a":1}"#);
    ///
    /// let element3 = raw_jsonb.get_by_index(3).unwrap();
    /// assert!(element3.is_none()); // Index out of bounds
    ///
    /// let obj_jsonb = r#"{"a": 1}"#.parse::<OwnedJsonb>().unwrap();
    /// let raw_jsonb = obj_jsonb.as_raw();
    /// let element = raw_jsonb.get_by_index(0).unwrap();
    /// assert!(element.is_none()); // Not an array
    /// ```
    pub fn get_by_index(&self, index: usize) -> Result<Option<OwnedJsonb>> {
        let array_iter_opt = ArrayIterator::new(*self)?;
        if let Some(mut array_iter) = array_iter_opt
            && let Some(item_result) = array_iter.nth(index)
        {
            let item = item_result?;
            let value = OwnedJsonb::from_item(item)?;
            return Ok(Some(value));
        }
        Ok(None)
    }

    /// Gets the value associated with a given key in a JSONB object.
    ///
    /// If the JSONB value is an object, this function searches for a key matching the provided `name`
    /// and returns the associated value as an `OwnedJsonb`.
    /// The `ignore_case` parameter controls whether the key search is case-sensitive.
    /// If the key is not found, it returns `Ok(None)`.
    /// If the JSONB value is not an object (e.g., it's an array or a scalar), this function also returns `Ok(None)`.
    ///
    /// # Arguments
    ///
    /// * `self` - The JSONB value.
    /// * `name` - The key to search for.
    /// * `ignore_case` - Whether the key search should be case-insensitive.
    ///
    /// # Returns
    ///
    /// * `Ok(Some(OwnedJsonb))` - The value associated with the key as an `OwnedJsonb`, if the input is an object and the key is found.
    /// * `Ok(None)` - If the input is not an object, or if the key is not found.
    /// * `Err(Error)` - If an error occurred during decoding (e.g., invalid JSONB data).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lance_jsonb::OwnedJsonb;
    ///
    /// let obj_jsonb = r#"{"a": 1, "b": "hello", "c": [1, 2]}"#.parse::<OwnedJsonb>().unwrap();
    /// let raw_jsonb = obj_jsonb.as_raw();
    ///
    /// let value_a = raw_jsonb.get_by_name("a", false).unwrap();
    /// assert_eq!(value_a.unwrap().to_string(), "1");
    ///
    /// let value_b = raw_jsonb.get_by_name("b", false).unwrap();
    /// assert_eq!(value_b.unwrap().to_string(), r#""hello""#);
    ///
    /// let value_c = raw_jsonb.get_by_name("c", false).unwrap();
    /// assert_eq!(value_c.unwrap().to_string(), "[1,2]");
    ///
    /// let value_d = raw_jsonb.get_by_name("d", false).unwrap();
    /// assert!(value_d.is_none()); // Key not found
    ///
    /// // Case-insensitive search
    /// let value_a_case_insensitive = raw_jsonb.get_by_name("A", true).unwrap();
    /// assert_eq!(value_a_case_insensitive.unwrap().to_string(), "1");
    ///
    /// let arr_jsonb = "[1, 2, 3]".parse::<OwnedJsonb>().unwrap();
    /// let raw_jsonb = arr_jsonb.as_raw();
    /// let value = raw_jsonb.get_by_name("a", false).unwrap();
    /// assert!(value.is_none()); // Not an object
    /// ```
    pub fn get_by_name(&self, name: &str, ignore_case: bool) -> Result<Option<OwnedJsonb>> {
        let key_name = Cow::Borrowed(name);
        if let Some(val_item) =
            self.get_object_value_by_key_name(&key_name, |name, key| key.eq(name))?
        {
            let value = OwnedJsonb::from_item(val_item)?;
            return Ok(Some(value));
        }
        if ignore_case
            && let Some(val_item) = self.get_object_value_by_key_name(&key_name, |name, key| {
                key.eq_ignore_ascii_case(name)
            })?
        {
            let value = OwnedJsonb::from_item(val_item)?;
            return Ok(Some(value));
        }
        Ok(None)
    }
}
