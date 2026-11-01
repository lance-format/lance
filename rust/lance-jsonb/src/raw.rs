// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-FileCopyrightText: Copyright 2023 Datafuse Labs
// Adapted from databendlabs/jsonb at commit fba895c5ebe77ce2539e187f9c652f51cbf195c3.

use std::cmp::Ordering;

use serde::Serialize;

use crate::OwnedJsonb;
use crate::core::ArrayIterator;
use crate::core::Deserializer;
use crate::core::JsonbItemType;
use crate::core::ObjectIterator;
use crate::error::*;

/// Represents JSONB data wrapped around a raw, immutable slice of bytes.
///
/// It does not own the underlying data, allowing various operations to be performed on the JSONB data *without copying*.
/// This is critical for performance when dealing with large JSONB values.
/// `RawJsonb` provides various methods to inspect and manipulate the JSONB data efficiently.
#[derive(Debug, Clone, Copy)]
pub struct RawJsonb<'a> {
    /// The underlying byte slice representing the JSONB data.
    pub(crate) data: &'a [u8],
}

impl<'a> RawJsonb<'a> {
    /// Creates a new RawJsonb from a byte slice.
    ///
    /// # Arguments
    ///
    /// * `data` - The byte slice containing the JSONB data.
    ///
    /// # Returns
    ///
    /// A new `RawJsonb` instance.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data }
    }

    /// Returns the length of the JSONB data in bytes.
    ///
    /// # Returns
    ///
    /// The length of the data in bytes.
    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    /// Creates an `OwnedJsonb` from the `RawJsonb` by copying the underlying data.
    ///
    /// This method converts a `RawJsonb`, which holds a reference to JSONB data,
    /// into an `OwnedJsonb`, which owns its own copy of the JSONB data. This is
    /// achieved by cloning the byte slice held by the `RawJsonb` into a new `Vec<u8>`.
    ///
    /// # Returns
    ///
    /// An `OwnedJsonb` instance containing a copy of the JSONB data from the `RawJsonb`.
    pub(crate) fn to_owned(self) -> OwnedJsonb {
        OwnedJsonb::new(self.data.to_vec())
    }

    /// Converts the JSONB value to a JSON string.
    ///
    /// This function serializes the JSONB value into a human-readable JSON string representation.
    /// If the JSONB data is invalid, treat it as a text JSON string and return directly.
    /// If the JSONB data is empty, return a JSON null for compatibility.
    ///
    /// # Returns
    ///
    /// * `String` - The JSON string representation of the value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lance_jsonb::OwnedJsonb;
    ///
    /// let arr_jsonb = "[1, 2, 3]".parse::<OwnedJsonb>().unwrap();
    /// let raw_jsonb = arr_jsonb.as_raw();
    /// assert_eq!(raw_jsonb.to_string(), "[1,2,3]");
    ///
    /// let obj_jsonb = r#"{"a": 1, "b": "hello"}"#.parse::<OwnedJsonb>().unwrap();
    /// let raw_jsonb = obj_jsonb.as_raw();
    /// assert_eq!(raw_jsonb.to_string(), r#"{"a":1,"b":"hello"}"#);
    ///
    /// let num_jsonb = "123.45".parse::<OwnedJsonb>().unwrap();
    /// let raw_jsonb = num_jsonb.as_raw();
    /// assert_eq!(raw_jsonb.to_string(), "123.45");
    ///
    /// let string_jsonb = r#""hello, world!""#.parse::<OwnedJsonb>().unwrap();
    /// let raw_jsonb = string_jsonb.as_raw();
    /// assert_eq!(raw_jsonb.to_string(), r#""hello, world!""#);
    ///
    /// let true_jsonb = "true".parse::<OwnedJsonb>().unwrap();
    /// let raw_jsonb = true_jsonb.as_raw();
    /// assert_eq!(raw_jsonb.to_string(), "true");
    ///
    /// // Example with invalid JSONB data (fallback to treat as text JSON string)
    /// let invalid_jsonb = OwnedJsonb::from(vec![1, 2, 3, 4]); // Invalid binary JSONB
    /// let invalid_raw_jsonb = invalid_jsonb.as_raw();
    ///
    /// // It will treat as text JSON string.
    /// assert_eq!(invalid_raw_jsonb.to_string(), "\u{1}\u{2}\u{3}\u{4}");
    /// ```
    #[allow(clippy::inherent_to_string)]
    pub fn to_string(&self) -> String {
        let mut buf = Vec::with_capacity(self.len());
        let formatter = serde_json::ser::CompactFormatter {};
        let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
        match self.serialize(&mut ser) {
            Ok(_) => String::from_utf8(buf).unwrap(),
            Err(_) => {
                if self.data.is_empty() {
                    "null".to_string()
                } else {
                    String::from_utf8_lossy(self.data).to_string()
                }
            }
        }
    }
}

/// Converts a borrowed byte slice into a RawJsonb.
/// This provides a convenient way to create a RawJsonb from existing data without copying.
impl<'a> From<&'a [u8]> for RawJsonb<'a> {
    fn from(data: &'a [u8]) -> Self {
        Self { data }
    }
}

/// Allows accessing the underlying byte slice as a reference.
/// This enables easy integration with functions that expect a &[u8].
impl AsRef<[u8]> for RawJsonb<'_> {
    fn as_ref(&self) -> &[u8] {
        self.data
    }
}

impl Eq for RawJsonb<'_> {}

impl PartialEq for RawJsonb<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.partial_cmp(other) == Some(Ordering::Equal)
    }
}

/// Implements `PartialOrd` for `RawJsonb`, allowing comparison of two `RawJsonb` values.
///
/// The comparison logic handles different JSONB types (scalar, array, object) and considers null values.
/// The ordering is defined as follows:
///
/// 1. Null is considered greater than any other type.
/// 2. Scalars are compared based on their type and value (String > Number > Boolean > ExtensionValue).
/// 3. Arrays are compared element by element.
/// 4. Objects are compared based on their keys and values.
/// 5. Arrays are greater than objects and scalars.
/// 6. Objects are greater than scalars.
/// 7. If the types are incompatible, None is returned.
#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for RawJsonb<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let self_type = self.jsonb_item_type().ok()?;
        let other_type = other.jsonb_item_type().ok()?;

        // First use JSONB type to determine the order,
        // different types must have different orders.
        if let Some(ord) = self_type.partial_cmp(&other_type) {
            return Some(ord);
        }

        match (self_type, other_type) {
            (JsonbItemType::Array(self_len), JsonbItemType::Array(other_len)) => {
                let self_array_iter = ArrayIterator::new(*self).ok()?.unwrap();
                let mut other_array_iter = ArrayIterator::new(*other).ok()?.unwrap();
                for (self_res, other_res) in &mut self_array_iter.zip(&mut other_array_iter) {
                    let self_item = self_res.ok()?;
                    let other_item = other_res.ok()?;

                    let ord = self_item.partial_cmp(&other_item)?;
                    if ord != Ordering::Equal {
                        return Some(ord);
                    }
                }
                Some(self_len.cmp(&other_len))
            }
            (JsonbItemType::Object(self_len), JsonbItemType::Object(other_len)) => {
                let self_object_iter = ObjectIterator::new(*self).ok()?.unwrap();
                let mut other_object_iter = ObjectIterator::new(*other).ok()?.unwrap();
                for (self_res, other_res) in &mut self_object_iter.zip(&mut other_object_iter) {
                    let (self_key, self_val) = self_res.ok()?;
                    let (other_key, other_val) = other_res.ok()?;

                    let key_ord = self_key.partial_cmp(other_key)?;
                    if key_ord != Ordering::Equal {
                        return Some(key_ord);
                    }
                    let val_ord = self_val.partial_cmp(&other_val)?;
                    if val_ord != Ordering::Equal {
                        return Some(val_ord);
                    }
                }
                Some(self_len.cmp(&other_len))
            }
            (JsonbItemType::String, JsonbItemType::String) => {
                let self_val = self.as_str();
                let other_val = other.as_str();
                match (self_val, other_val) {
                    (Ok(Some(self_val)), Ok(Some(other_val))) => self_val.partial_cmp(&other_val),
                    (_, _) => None,
                }
            }
            (JsonbItemType::Number, JsonbItemType::Number) => {
                let self_val = self.as_number();
                let other_val = other.as_number();
                match (self_val, other_val) {
                    (Ok(Some(self_val)), Ok(Some(other_val))) => self_val.partial_cmp(&other_val),
                    (_, _) => None,
                }
            }
            (JsonbItemType::Boolean, JsonbItemType::Boolean) => {
                let self_val = self.as_bool();
                let other_val = other.as_bool();
                match (self_val, other_val) {
                    (Ok(Some(self_val)), Ok(Some(other_val))) => self_val.partial_cmp(&other_val),
                    (_, _) => None,
                }
            }
            (JsonbItemType::Extension, JsonbItemType::Extension) => {
                let self_val = self.as_extension_value();
                let other_val = other.as_extension_value();
                match (self_val, other_val) {
                    (Ok(Some(self_val)), Ok(Some(other_val))) => self_val.partial_cmp(&other_val),
                    (_, _) => None,
                }
            }
            (_, _) => None,
        }
    }
}

/// Implements `Ord` for `RawJsonb`, allowing comparison of two `RawJsonb` values using the total ordering.
/// This implementation leverages the `PartialOrd` implementation, returning `Ordering::Equal` for incomparable values.
impl Ord for RawJsonb<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.partial_cmp(other) {
            Some(ordering) => ordering,
            None => Ordering::Equal,
        }
    }
}

/// Deserializes a `RawJsonb` into a Rust data structure using Serde.
///
/// This function takes a `RawJsonb` (a borrowed slice of JSONB data) and attempts
/// to deserialize it into a Rust type `T` that implements the `Deserialize` trait.
/// It uses a custom `Deserializer` to handle the JSONB data.
///
/// # Arguments
///
/// * `raw_jsonb`: A reference to the `RawJsonb` containing the JSONB data to deserialize.
///
/// # Type Parameters
///
/// * `T`: The Rust type to deserialize the JSONB data into.  This type must implement
///   the `serde::de::Deserialize` trait.
///
/// # Returns
///
/// * `Ok(t)`: If the deserialization is successful, returns the deserialized value of type `T`.
/// * `Err(Error::InvalidJsonb)`: If the deserialization fails due to invalid JSONB data
///   (e.g., trailing characters after the expected JSONB structure).
/// * `Err(e)`: If any other Serde deserialization error occurs.
///
/// # Examples
///
/// ```
/// use lance_jsonb::from_raw_jsonb;
/// use lance_jsonb::OwnedJsonb;
/// use lance_jsonb::RawJsonb;
/// use serde::Deserialize;
///
/// #[derive(Deserialize, Debug, PartialEq, Eq)]
/// struct Person {
///     name: String,
///     age: u32,
/// }
///
/// let owned_jsonb = r#"{"name": "Alice", "age": 20}"#.parse::<OwnedJsonb>().unwrap();
/// let raw_jsonb = owned_jsonb.as_raw();
///
/// let person: Person = from_raw_jsonb(&raw_jsonb).unwrap();
/// assert_eq!(person, Person { name: "Alice".to_string(), age: 20 });
/// println!("{:?}", person); // Output: Person { name: "Alice", age: 20 }
/// ```
pub fn from_raw_jsonb<'de, T>(raw_jsonb: &'de RawJsonb) -> Result<T>
where
    T: serde::de::Deserialize<'de>,
{
    let mut deserializer = Deserializer::new(raw_jsonb);
    let t = T::deserialize(&mut deserializer)?;
    if deserializer.end() {
        Ok(t)
    } else {
        // Trailing characters
        Err(Error::InvalidJsonb)
    }
}
