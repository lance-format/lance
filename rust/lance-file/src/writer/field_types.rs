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

use std::fmt;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field as ArrowField, FieldRef, SchemaRef};
use lance_arrow::BLOB_V2_EXT_NAME;
use lance_core::{Error, Result, datatypes::Schema};

/// An array whose Arrow type or extension differs from the field the file
/// schema records for it.
///
/// Returned as the source of an [`Error::InvalidInput`] by the file writers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldTypeMismatch {
    /// Dot-separated path of the mismatched field, starting at the top-level
    /// column.
    pub field_path: String,
    /// The field the file schema records.
    pub expected: FieldRef,
    /// The field the input carries. A column written on its own is a bare
    /// array, so it carries no extension.
    pub actual: FieldRef,
}

impl fmt::Display for FieldTypeMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "field `{}` does not match the file schema: expected {}, got {}",
            self.field_path,
            FieldType(&self.expected),
            FieldType(&self.actual)
        )
    }
}

impl std::error::Error for FieldTypeMismatch {}

struct FieldType<'a>(&'a ArrowField);

impl fmt::Display for FieldType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.data_type())?;
        match (
            self.0.extension_type_name(),
            self.0.extension_type_metadata(),
        ) {
            (Some(name), Some(metadata)) => write!(f, " (extension {name}, metadata {metadata})"),
            (Some(name), None) => write!(f, " (extension {name})"),
            _ => write!(f, " (no extension)"),
        }
    }
}

/// The Arrow fields a file schema expects, built once per writer.
pub struct ExpectedTypes {
    fields: Vec<FieldRef>,
    /// The last batch schema that passed. Batches of one stream normally share
    /// a schema, so later batches are accepted without walking it again.
    accepted: Option<SchemaRef>,
}

impl ExpectedTypes {
    pub fn new(schema: &Schema) -> Self {
        Self {
            fields: schema
                .fields
                .iter()
                .map(|field| Arc::new(ArrowField::from(field)))
                .collect(),
            accepted: None,
        }
    }

    /// Check every column of `batch` that the file schema writes.
    ///
    /// Columns are matched by name, the same way the encoders select them. A
    /// missing column is left for the encoder to report.
    pub fn check_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        let batch_schema = batch.schema();
        if self.accepted.as_ref() == Some(&batch_schema) {
            return Ok(());
        }
        for (index, expected) in self.fields.iter().enumerate() {
            // Batches usually list columns in schema order; only fall back to
            // a name search when they do not, so wide schemas stay linear.
            let actual = batch_schema
                .fields()
                .get(index)
                .filter(|actual| actual.name() == expected.name())
                .or_else(|| {
                    let (index, _) = batch_schema.column_with_name(expected.name())?;
                    batch_schema.fields().get(index)
                });
            if let Some(actual) = actual {
                check_field(expected, actual, true).map_err(into_error)?;
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
        let expected = &self.fields[index];
        let actual = Arc::new(ArrowField::new(
            expected.name(),
            array.data_type().clone(),
            true,
        ));
        check_field(expected, &actual, false).map_err(into_error)
    }
}

fn into_error(mismatch: FieldTypeMismatch) -> Error {
    Error::invalid_input_source(Box::new(mismatch))
}

/// Names and nullability of nested fields are not compared: the encoders
/// address children by position and verify nulls against the values.
fn check_field(
    expected: &FieldRef,
    actual: &FieldRef,
    compare_extension: bool,
) -> std::result::Result<(), FieldTypeMismatch> {
    let mismatch = || FieldTypeMismatch {
        field_path: expected.name().clone(),
        expected: expected.clone(),
        actual: actual.clone(),
    };
    if compare_extension
        && (expected.extension_type_name() != actual.extension_type_name()
            || expected.extension_type_metadata() != actual.extension_type_metadata())
    {
        return Err(mismatch());
    }
    let children = match (expected.data_type(), actual.data_type()) {
        // The file schema records the Blob v2 struct users write, but blob
        // preprocessing hands the encoder a descriptor struct, which the blob
        // encoder validates itself.
        (DataType::Struct(_), DataType::Struct(_))
            if expected.extension_type_name() == Some(BLOB_V2_EXT_NAME) =>
        {
            return Ok(());
        }
        (DataType::Struct(expected_children), DataType::Struct(actual_children))
            if expected_children.len() == actual_children.len() =>
        {
            expected_children
                .iter()
                .zip(actual_children.iter())
                .collect()
        }
        (DataType::List(expected_item), DataType::List(actual_item))
        | (DataType::LargeList(expected_item), DataType::LargeList(actual_item))
        | (DataType::Map(expected_item, _), DataType::Map(actual_item, _)) => {
            vec![(expected_item, actual_item)]
        }
        (
            DataType::FixedSizeList(expected_item, expected_size),
            DataType::FixedSizeList(actual_item, actual_size),
        ) if expected_size == actual_size => vec![(expected_item, actual_item)],
        // The encoders write view arrays in the equivalent offset layout.
        (DataType::Utf8, DataType::Utf8View) | (DataType::Binary, DataType::BinaryView) => {
            return Ok(());
        }
        (expected_type, actual_type) if expected_type == actual_type => return Ok(()),
        _ => return Err(mismatch()),
    };
    for (expected_child, actual_child) in children {
        check_field(expected_child, actual_child, true).map_err(|mut mismatch| {
            mismatch.field_path = format!("{}.{}", expected.name(), mismatch.field_path);
            mismatch
        })?;
    }
    Ok(())
}
