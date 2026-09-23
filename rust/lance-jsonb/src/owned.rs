// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-FileCopyrightText: Copyright 2023 Datafuse Labs
// Adapted from databendlabs/jsonb at commit fba895c5ebe77ce2539e187f9c652f51cbf195c3.

use std::cmp::Ordering;
use std::fmt::Display;
use std::str::FromStr;

use crate::RawJsonb;
use crate::error::Error;
use crate::parse_value;

/// Represents a JSONB data that owns its underlying data.
///
/// This struct provides ownership over the binary JSONB representation.
/// `OwnedJsonb` is primarily used to create JSONB data from other data types (such as JSON String).
/// However, for most operations, it's necessary to convert an `OwnedJsonb` to a `RawJsonb` using the `as_raw()` method
/// to avoid unnecessary copying and to take advantage of the performance benefits of the read-only access of the `RawJsonb`.
#[derive(Debug, Clone)]
pub struct OwnedJsonb {
    /// The underlying `Vec<u8>` containing the binary JSONB data.
    pub(crate) data: Vec<u8>,
}

impl OwnedJsonb {
    /// Creates a new OwnedJsonb from a Vec<u8>.
    ///
    /// # Arguments
    ///
    /// * `data` - The `Vec<u8>` containing the JSONB data.
    ///
    /// # Returns
    ///
    /// A new `OwnedJsonb` instance.
    pub(crate) fn new(data: Vec<u8>) -> Self {
        Self { data }
    }

    /// Creates a `RawJsonb` view of the owned data.
    /// This is useful for passing the data to functions that expect a `RawJsonb`.
    /// This does *not* transfer ownership.
    ///
    /// # Returns
    ///
    /// A `RawJsonb` instance referencing the owned data.
    pub fn as_raw(&self) -> RawJsonb<'_> {
        RawJsonb::new(self.data.as_slice())
    }

    /// Consumes the OwnedJsonb and returns the underlying Vec<u8>.
    ///
    /// # Returns
    ///
    /// The underlying `Vec<u8>` containing the JSONB data.
    pub fn to_vec(self) -> Vec<u8> {
        self.data
    }
}

/// Creates an `OwnedJsonb` from a borrowed byte slice.  The byte slice is copied into a new `Vec<u8>`.
impl From<&[u8]> for OwnedJsonb {
    fn from(data: &[u8]) -> Self {
        Self {
            data: data.to_vec(),
        }
    }
}

/// Creates an `OwnedJsonb` from a `Vec<u8>`. This is a simple ownership transfer.
impl From<Vec<u8>> for OwnedJsonb {
    fn from(data: Vec<u8>) -> Self {
        Self { data }
    }
}

/// Parses a string into an `OwnedJsonb`.
/// The string is parsed into a JSON value, then encoded into the binary JSONB format.
impl FromStr for OwnedJsonb {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let value = parse_value(s.as_bytes())?;
        let mut data = Vec::new();
        value.write_to_vec(&mut data);
        Ok(Self { data })
    }
}

/// Allows accessing the underlying byte slice as a reference.
/// This enables easy integration with functions that expect a `&[u8]`.
impl AsRef<[u8]> for OwnedJsonb {
    fn as_ref(&self) -> &[u8] {
        self.data.as_ref()
    }
}

/// Implements the Display trait, allowing OwnedJsonb to be formatted as a string using the `{}` format specifier.
impl Display for OwnedJsonb {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let raw_jsonb = self.as_raw();
        write!(f, "{}", raw_jsonb.to_string())
    }
}

impl Eq for OwnedJsonb {}

impl PartialEq for OwnedJsonb {
    fn eq(&self, other: &Self) -> bool {
        self.partial_cmp(other) == Some(Ordering::Equal)
    }
}

/// Implements `PartialOrd` for `OwnedJsonb`, allowing comparison of two `OwnedJsonb` values.
///
/// The comparison logic handles different JSONB types (scalar, array, object) and considers null values.
/// The ordering is defined as follows:
///
/// 1. Null is considered greater than any other type.
/// 2. Scalars are compared based on their type and value (String > Number > Boolean).
/// 3. Arrays are compared element by element.
/// 4. Objects are compared based on their keys and values.
/// 5. Arrays are greater than objects and scalars.
/// 6. Objects are greater than scalars.
/// 7. If the types are incompatible, None is returned.
#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for OwnedJsonb {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let self_raw = self.as_raw();
        let other_raw = other.as_raw();
        self_raw.partial_cmp(&other_raw)
    }
}

/// Implements `Ord` for `OwnedJsonb`, allowing comparison of two `OwnedJsonb` values using the total ordering.
/// This implementation leverages the `PartialOrd` implementation, returning `Ordering::Equal` for incomparable values.
impl Ord for OwnedJsonb {
    fn cmp(&self, other: &Self) -> Ordering {
        let self_raw = self.as_raw();
        let other_raw = other.as_raw();
        match self_raw.partial_cmp(&other_raw) {
            Some(ordering) => ordering,
            None => Ordering::Equal,
        }
    }
}
