// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Converts caller string and binary columns to the representation a table
//! stores.
//!
//! A table column records one Arrow representation, such as `Utf8`, while
//! callers may hold the same values as `LargeUtf8`, `Utf8View` or a
//! dictionary of either. Writes to an existing table convert those columns to
//! the table's representation before anything else sees them, so the table
//! schema, the existing files and what reads return stay unchanged.
//!
//! Only conversions within one family are made: the string types (`Utf8`,
//! `LargeUtf8`, `Utf8View`), the binary types (`Binary`, `LargeBinary`,
//! `BinaryView`), and dictionaries whose values are of the same family. A
//! field whose extension type differs from the table's, or that is a blob, is
//! left as it is, for the schema check to report.
//!
//! ```text
//! input                              table
//! Utf8 | LargeUtf8 | Utf8View    --> Utf8 | LargeUtf8
//! Dictionary(_, any of these)    --> Dictionary(K, Utf8 | LargeUtf8)
//! (and the same for Binary, LargeBinary and BinaryView)
//! ```

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, GenericListArray, MapArray, OffsetSizeTrait, RecordBatch,
    RecordBatchOptions, StructArray,
};
use arrow_buffer::NullBuffer;
use arrow_cast::{CastOptions, cast_with_options};
use arrow_schema::{DataType, Field as ArrowField, FieldRef, Schema as ArrowSchema, SchemaRef};
use futures::StreamExt;
use lance_arrow::schema::FieldExt;
use lance_core::{Error, Result, datatypes::Schema};
use lance_io::stream::{RecordBatchStreamAdapter, SendableRecordBatchStream};

/// Largest number of value bytes an array with 32-bit offsets can address.
const MAX_SMALL_OFFSET_BYTES: usize = i32::MAX as usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
    String,
    Binary,
}

fn flat_family(data_type: &DataType) -> Option<Family> {
    match data_type {
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => Some(Family::String),
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView => Some(Family::Binary),
        _ => None,
    }
}

fn family(data_type: &DataType) -> Option<Family> {
    match data_type {
        DataType::Dictionary(_, value_type) => flat_family(value_type),
        other => flat_family(other),
    }
}

/// The type with 64-bit offsets that holds the same values as `data_type`.
fn with_large_offsets(data_type: &DataType) -> DataType {
    match data_type {
        DataType::Utf8 => DataType::LargeUtf8,
        DataType::Binary => DataType::LargeBinary,
        other => other.clone(),
    }
}

fn same_extension(input: &ArrowField, table: &ArrowField) -> bool {
    input.extension_type_name() == table.extension_type_name()
        && input.extension_type_metadata() == table.extension_type_metadata()
}

/// The caller's field with the table's string or binary representation, or
/// `None` when nothing in it needs converting.
fn target_field(input: &ArrowField, table: &ArrowField) -> Option<ArrowField> {
    if input.data_type() == table.data_type() || !same_extension(input, table) || table.is_blob() {
        return None;
    }
    let data_type = match (input.data_type(), table.data_type()) {
        (DataType::Struct(input_children), DataType::Struct(table_children)) => {
            let mut changed = false;
            let children = input_children
                .iter()
                .map(|input_child| {
                    let converted = table_children
                        .iter()
                        .find(|table_child| table_child.name() == input_child.name())
                        .and_then(|table_child| target_field(input_child, table_child));
                    changed |= converted.is_some();
                    converted
                        .map(Arc::new)
                        .unwrap_or_else(|| input_child.clone())
                })
                .collect::<Vec<_>>();
            if !changed {
                return None;
            }
            DataType::Struct(children.into())
        }
        (DataType::List(input_item), DataType::List(table_item)) => {
            DataType::List(Arc::new(target_field(input_item, table_item)?))
        }
        (DataType::LargeList(input_item), DataType::LargeList(table_item)) => {
            DataType::LargeList(Arc::new(target_field(input_item, table_item)?))
        }
        (
            DataType::FixedSizeList(input_item, input_size),
            DataType::FixedSizeList(table_item, table_size),
        ) if input_size == table_size => {
            DataType::FixedSizeList(Arc::new(target_field(input_item, table_item)?), *input_size)
        }
        (DataType::Map(input_entries, sorted), DataType::Map(table_entries, _)) => DataType::Map(
            Arc::new(target_field(input_entries, table_entries)?),
            *sorted,
        ),
        (input_type, table_type)
            if family(input_type).is_some()
                && family(input_type) == family(table_type)
                && !matches!(table_type, DataType::Utf8View | DataType::BinaryView) =>
        {
            table_type.clone()
        }
        _ => return None,
    };
    Some(input.clone().with_data_type(data_type))
}

/// Converts the string and binary columns of a write to an existing table into
/// the representations the table stores.
#[derive(Debug, Clone)]
pub struct TableRepresentation {
    schema: SchemaRef,
}

impl TableRepresentation {
    /// Plan the conversion of `input` columns to the representations of
    /// `table`, matching columns by name.
    ///
    /// Returns `None` when no column needs converting. Columns that the table
    /// does not have, or whose types differ in any other way, are left as they
    /// are for the schema check to report.
    pub fn try_new(input: &ArrowSchema, table: &Schema) -> Option<Self> {
        let table = ArrowSchema::from(table);
        let mut changed = false;
        let fields = input
            .fields()
            .iter()
            .map(|input_field| {
                let converted = table
                    .field_with_name(input_field.name())
                    .ok()
                    .and_then(|table_field| target_field(input_field, table_field));
                changed |= converted.is_some();
                converted
                    .map(Arc::new)
                    .unwrap_or_else(|| input_field.clone())
            })
            .collect::<Vec<_>>();
        changed.then(|| Self {
            schema: Arc::new(ArrowSchema::new_with_metadata(
                fields,
                input.metadata().clone(),
            )),
        })
    }

    /// The input schema with the table's representations.
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Convert one batch of the input.
    ///
    /// Fails, naming the field, when a value does not fit the table's
    /// representation: more value bytes than 32-bit offsets address, or more
    /// distinct values than the dictionary key type addresses.
    pub fn convert_batch(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let columns = batch
            .columns()
            .iter()
            .zip(self.schema.fields())
            .map(|(column, field)| convert_array(column, field, field.name()))
            .collect::<Result<Vec<_>>>()?;
        Ok(RecordBatch::try_new_with_options(
            self.schema.clone(),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
        )?)
    }

    /// Convert every batch of `stream`.
    pub fn convert_stream(self, stream: SendableRecordBatchStream) -> SendableRecordBatchStream {
        let schema = self.schema.clone();
        let converted = stream.map(move |batch| batch.and_then(|batch| self.convert_batch(&batch)));
        Box::pin(RecordBatchStreamAdapter::new(schema, converted))
    }
}

fn convert_array(array: &ArrayRef, target: &FieldRef, path: &str) -> Result<ArrayRef> {
    if array.data_type() == target.data_type() {
        return Ok(array.clone());
    }
    let child_path = |child: &ArrowField| format!("{path}.{}", child.name());
    Ok(match target.data_type() {
        DataType::Struct(children) => {
            let array = array.as_struct();
            let columns = array
                .columns()
                .iter()
                .zip(children.iter())
                .map(|(column, child)| convert_array(column, child, &child_path(child)))
                .collect::<Result<Vec<_>>>()?;
            Arc::new(StructArray::try_new(
                children.clone(),
                columns,
                array.nulls().cloned(),
            )?)
        }
        DataType::List(item) => convert_list::<i32>(array, item, &child_path(item))?,
        DataType::LargeList(item) => convert_list::<i64>(array, item, &child_path(item))?,
        DataType::FixedSizeList(item, size) => {
            let array = array.as_fixed_size_list();
            let values = convert_array(array.values(), item, &child_path(item))?;
            Arc::new(FixedSizeListArray::try_new(
                item.clone(),
                *size,
                values,
                array.nulls().cloned(),
            )?)
        }
        DataType::Map(entries, sorted) => {
            let array = array.as_map();
            let values: ArrayRef = Arc::new(array.entries().clone());
            let converted = convert_array(&values, entries, &child_path(entries))?;
            Arc::new(MapArray::try_new(
                entries.clone(),
                array.offsets().clone(),
                converted.as_struct().clone(),
                array.nulls().cloned(),
                *sorted,
            )?)
        }
        target_type => convert_values(array, target_type, path)?,
    })
}

fn convert_list<O: OffsetSizeTrait>(
    array: &ArrayRef,
    item: &FieldRef,
    item_path: &str,
) -> Result<ArrayRef> {
    let array = array.as_list::<O>();
    let values = convert_array(array.values(), item, item_path)?;
    Ok(Arc::new(GenericListArray::<O>::try_new(
        item.clone(),
        array.offsets().clone(),
        values,
        array.nulls().cloned(),
    )?))
}

/// Convert a string or binary array, or a dictionary of one, to `target`.
fn convert_values(array: &ArrayRef, target: &DataType, path: &str) -> Result<ArrayRef> {
    let cast = |array: &dyn Array, to: &DataType| {
        // `safe: false` turns values that do not fit into an error instead of
        // nulls.
        let options = CastOptions {
            safe: false,
            ..Default::default()
        };
        cast_with_options(array, to, &options).map_err(|error| {
            Error::invalid_input(format!(
                "cannot convert field `{path}` from {} to the table's {target}: {error}",
                array.data_type()
            ))
        })
    };
    match target {
        // A dictionary holds each distinct value once, so its values can fit
        // 32-bit offsets when the rows would not. Build it with 64-bit offsets
        // and narrow only the values.
        DataType::Dictionary(key_type, value_type) => {
            let wide_type = with_large_offsets(value_type);
            let wide = cast(
                array.as_ref(),
                &DataType::Dictionary(key_type.clone(), Box::new(wide_type.clone())),
            )?;
            if &wide_type == value_type.as_ref() {
                return Ok(wide);
            }
            let dictionary = wide.as_any_dictionary();
            check_small_offsets(dictionary.values().as_ref(), target, path)?;
            let values = cast(dictionary.values().as_ref(), value_type)?;
            Ok(dictionary.with_values(values))
        }
        // Arrow builds view and dictionary conversions with the target's own
        // offsets, which panics on overflow, so go through 64-bit offsets and
        // check the total first.
        _ => {
            let wide_type = with_large_offsets(target);
            if &wide_type != target {
                check_small_offsets(array.as_ref(), target, path)?;
            }
            let wide = cast(array.as_ref(), &wide_type)?;
            cast(wide.as_ref(), target)
        }
    }
}

fn check_small_offsets(array: &dyn Array, target: &DataType, path: &str) -> Result<()> {
    let bytes = referenced_value_bytes(array);
    if bytes > MAX_SMALL_OFFSET_BYTES {
        return Err(Error::invalid_input(format!(
            "cannot convert field `{path}` from {} to the table's {target}: its {bytes} value \
             bytes exceed the {MAX_SMALL_OFFSET_BYTES} bytes 32-bit offsets address",
            array.data_type()
        )));
    }
    Ok(())
}

/// The number of value bytes the valid rows of a string or binary array, or a
/// dictionary of one, refer to.
fn referenced_value_bytes(array: &dyn Array) -> usize {
    match array.data_type() {
        DataType::Dictionary(_, _) => {
            let dictionary = array.as_any_dictionary();
            // Only an all-null dictionary has no values, and it references no
            // bytes. `normalized_keys` rejects an empty values array.
            if dictionary.values().is_empty() {
                return 0;
            }
            let lengths = value_lengths(dictionary.values().as_ref());
            // The keys of null rows are arbitrary, and may index an empty
            // values array.
            let keys = dictionary.normalized_keys();
            sum_valid(
                dictionary.keys().nulls(),
                keys.into_iter()
                    .map(|key| lengths.get(key).copied().unwrap_or_default()),
            )
        }
        _ => value_lengths(array).into_iter().sum(),
    }
}

/// The length of every row, zero for null rows.
fn value_lengths(array: &dyn Array) -> Vec<usize> {
    fn offset_lengths<O: OffsetSizeTrait>(offsets: &[O]) -> Vec<usize> {
        offsets
            .windows(2)
            .map(|window| (window[1] - window[0]).as_usize())
            .collect()
    }
    let lengths = match array.data_type() {
        DataType::Utf8 => offset_lengths(array.as_string::<i32>().value_offsets()),
        DataType::LargeUtf8 => offset_lengths(array.as_string::<i64>().value_offsets()),
        DataType::Binary => offset_lengths(array.as_binary::<i32>().value_offsets()),
        DataType::LargeBinary => offset_lengths(array.as_binary::<i64>().value_offsets()),
        DataType::Utf8View => view_lengths(array.as_string_view().views()),
        DataType::BinaryView => view_lengths(array.as_binary_view().views()),
        _ => vec![0; array.len()],
    };
    match array.nulls() {
        Some(nulls) => lengths
            .into_iter()
            .zip(nulls.iter())
            .map(|(length, is_valid)| if is_valid { length } else { 0 })
            .collect(),
        None => lengths,
    }
}

fn view_lengths(views: &[u128]) -> Vec<usize> {
    // The low 32 bits of a view hold the value's length.
    views.iter().map(|view| *view as u32 as usize).collect()
}

fn sum_valid(nulls: Option<&NullBuffer>, lengths: impl Iterator<Item = usize>) -> usize {
    match nulls {
        Some(nulls) => lengths
            .zip(nulls.iter())
            .filter_map(|(length, is_valid)| is_valid.then_some(length))
            .sum(),
        None => lengths.sum(),
    }
}
