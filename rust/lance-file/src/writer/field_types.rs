// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Checks that arrays handed to a file writer have the Arrow types recorded in
//! the file schema.
//!
//! The file schema is what readers use to decode a column, so an array whose
//! type or extension differs from it is written in a layout the file does not
//! describe. Callers own any conversion from their input representation (for
//! example Arrow JSON text to Lance JSONB); this check only rejects arrays that
//! skipped it, before any page is encoded.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field as ArrowField, SchemaRef};
use lance_arrow::{ARROW_EXT_META_KEY, ARROW_EXT_NAME_KEY, BLOB_V2_EXT_NAME};
use lance_core::{Error, Result, datatypes::Schema};

/// The Arrow type and extension of a field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrowFieldType {
    pub data_type: DataType,
    /// The `ARROW:extension:name` metadata entry, if any.
    pub extension_name: Option<String>,
    /// The `ARROW:extension:metadata` metadata entry, if any.
    pub extension_metadata: Option<String>,
}

impl ArrowFieldType {
    fn new(data_type: &DataType, metadata: Option<&HashMap<String, String>>) -> Self {
        let extension_entry = |key| metadata.and_then(|metadata| metadata.get(key).cloned());
        Self {
            data_type: data_type.clone(),
            extension_name: extension_entry(ARROW_EXT_NAME_KEY),
            extension_metadata: extension_entry(ARROW_EXT_META_KEY),
        }
    }
}

impl fmt::Display for ArrowFieldType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.data_type)?;
        match (&self.extension_name, &self.extension_metadata) {
            (Some(name), Some(metadata)) => write!(f, " (extension {name}, metadata {metadata})"),
            (Some(name), None) => write!(f, " (extension {name})"),
            (None, Some(metadata)) => write!(f, " (extension metadata {metadata})"),
            (None, None) => write!(f, " (no extension)"),
        }
    }
}

/// An array whose Arrow type or extension differs from the field the file
/// schema records for it.
///
/// Returned as the source of an [`Error::InvalidInput`] by the file writers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldTypeMismatch {
    /// Dot-separated path of the mismatched field, starting at the top-level
    /// column.
    pub field_path: String,
    pub expected: ArrowFieldType,
    pub actual: ArrowFieldType,
}

impl fmt::Display for FieldTypeMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "field `{}` does not match the file schema: expected {}, got {}",
            self.field_path, self.expected, self.actual
        )
    }
}

impl std::error::Error for FieldTypeMismatch {}

/// The Arrow fields a file schema expects, built once per writer.
pub struct ExpectedTypes {
    fields: Vec<ArrowField>,
    /// The last batch schema that passed. Batches of one stream normally share
    /// a schema, so later batches are accepted without walking it again.
    accepted: Option<SchemaRef>,
}

impl ExpectedTypes {
    pub fn new(schema: &Schema) -> Self {
        Self {
            fields: schema.fields.iter().map(ArrowField::from).collect(),
            accepted: None,
        }
    }

    /// Check every column of `batch` that the file schema writes.
    ///
    /// Columns are matched by name, the same way the encoders select them. A
    /// missing column is left for the encoder to report.
    pub fn check_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        let batch_schema = batch.schema();
        if self.accepted.as_ref().is_some_and(|accepted| {
            Arc::ptr_eq(accepted, &batch_schema) || accepted == &batch_schema
        }) {
            return Ok(());
        }
        for (index, expected) in self.fields.iter().enumerate() {
            // Batches usually list columns in schema order; only fall back to
            // a name search when they do not, so wide schemas stay linear.
            let actual = batch_schema
                .fields()
                .get(index)
                .filter(|actual| actual.name() == expected.name())
                .map(|actual| actual.as_ref())
                .or_else(|| batch_schema.field_with_name(expected.name()).ok());
            if let Some(actual) = actual {
                check_field(
                    &mut Vec::new(),
                    expected,
                    actual.data_type(),
                    Some(actual.metadata()),
                )?;
            }
        }
        self.accepted = Some(batch_schema);
        Ok(())
    }

    /// Check an array written to the top-level column at `index`.
    ///
    /// A bare array carries no field metadata, so the column's own extension
    /// is not compared; nested extensions live in the array's data type and
    /// are.
    pub fn check_column(&self, index: usize, array: &ArrayRef) -> Result<()> {
        check_field(
            &mut Vec::new(),
            &self.fields[index],
            array.data_type(),
            None,
        )
    }
}

fn check_field<'a>(
    path: &mut Vec<&'a str>,
    expected: &'a ArrowField,
    actual_type: &DataType,
    actual_metadata: Option<&HashMap<String, String>>,
) -> Result<()> {
    path.push(expected.name());
    let result = check_field_at_path(path, expected, actual_type, actual_metadata);
    path.pop();
    result
}

fn check_field_at_path<'a>(
    path: &mut Vec<&'a str>,
    expected: &'a ArrowField,
    actual_type: &DataType,
    actual_metadata: Option<&HashMap<String, String>>,
) -> Result<()> {
    let expected_metadata = expected.metadata();
    let same_extension = actual_metadata.is_none_or(|actual_metadata| {
        [ARROW_EXT_NAME_KEY, ARROW_EXT_META_KEY]
            .iter()
            .all(|key| expected_metadata.get(*key) == actual_metadata.get(*key))
    });
    let is_blob_v2 = expected_metadata
        .get(ARROW_EXT_NAME_KEY)
        .is_some_and(|name| name == BLOB_V2_EXT_NAME);
    // Names and nullability of nested fields are not compared: the encoders
    // address children by position and verify nulls against the values.
    let children_match = same_extension
        && match (expected.data_type(), actual_type) {
            // The file schema records the Blob v2 struct users write, but blob
            // preprocessing hands the encoder a descriptor struct, which the
            // blob encoder validates itself.
            (DataType::Struct(_), DataType::Struct(_)) if is_blob_v2 => true,
            (DataType::Struct(expected_children), DataType::Struct(actual_children)) => {
                if expected_children.len() != actual_children.len() {
                    false
                } else {
                    for (expected_child, actual_child) in
                        expected_children.iter().zip(actual_children.iter())
                    {
                        check_field(
                            path,
                            expected_child,
                            actual_child.data_type(),
                            Some(actual_child.metadata()),
                        )?;
                    }
                    true
                }
            }
            (DataType::List(expected_item), DataType::List(actual_item))
            | (DataType::LargeList(expected_item), DataType::LargeList(actual_item))
            | (DataType::Map(expected_item, _), DataType::Map(actual_item, _)) => {
                check_field(
                    path,
                    expected_item,
                    actual_item.data_type(),
                    Some(actual_item.metadata()),
                )?;
                true
            }
            (
                DataType::FixedSizeList(expected_item, expected_size),
                DataType::FixedSizeList(actual_item, actual_size),
            ) if expected_size == actual_size => {
                check_field(
                    path,
                    expected_item,
                    actual_item.data_type(),
                    Some(actual_item.metadata()),
                )?;
                true
            }
            // The encoders write view arrays in the equivalent offset layout.
            (DataType::Utf8, DataType::Utf8View) | (DataType::Binary, DataType::BinaryView) => true,
            (expected_type, actual_type) => expected_type == actual_type,
        };
    if children_match {
        return Ok(());
    }
    Err(Error::invalid_input_source(Box::new(FieldTypeMismatch {
        field_path: path.join("."),
        expected: ArrowFieldType::new(expected.data_type(), Some(expected_metadata)),
        actual: ArrowFieldType::new(actual_type, actual_metadata),
    })))
}
