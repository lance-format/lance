// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Raw fixed-width bodies for RaBitQ cache entries.
//!
//! Arrow IPC writes a validity bitmap for every array, all-valid ones
//! included, so a persisted `FixedSizeList<u8, n>` column costs `n / 8`
//! bytes per row beyond its values: 12.5-13% of a sign plane or a native
//! partition. RaBitQ cache entries hold only non-null fixed-width columns,
//! so they are written as their schema and value buffers instead:
//!
//! ```text
//! [kind: u8 = 1]
//! [schema_len: u32 LE][Arrow IPC schema message]
//! [rows: u64 LE]
//! per column, in schema order:
//!   [values, little-endian][zero padding to a multiple of 8 bytes]
//! ```
//!
//! A primitive column contributes its values and a `FixedSizeList` of a
//! primitive its child values. The schema message keeps the names, types,
//! nullability and metadata, so a decoded batch has the source's schema.
//! Any other batch (a column with nulls or of another type, or one written
//! on a big-endian host) is written as `[kind: u8 = 0]` followed by an
//! Arrow IPC section, so every batch round trips.
//!
//! Decoding copies the values into buffers aligned to their type, sized
//! exactly to them, as the ex-plane codec does (see `plane_cache`).

use std::sync::Arc;

use arrow::array::ArrayData;
use arrow::buffer::Buffer;
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, RecordBatch, RecordBatchOptions, cast::AsArray, make_array,
};
use arrow_ipc::writer::{DictionaryTracker, IpcDataGenerator, IpcWriteOptions};
use arrow_schema::{DataType, Schema};
use lance_core::cache::{CacheEntryReader, CacheEntryWriter};
use lance_core::{Error, Result};

/// Body kind of an Arrow IPC section, the fallback for any batch.
const IPC_BODY_KIND: u8 = 0;
/// Body kind of a raw fixed-width body.
pub const RAW_BODY_KIND: u8 = 1;
/// Every column's values are zero-padded to a multiple of this many bytes.
const VALUE_PADDING: usize = 8;
const ZEROS: [u8; VALUE_PADDING] = [0; VALUE_PADDING];
/// Name reported by decode errors, which have no file path.
const BODY_NAME: &str = "raw cache body";

/// How one column's values are stored in a raw body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RawColumn {
    /// Bytes of one primitive value.
    value_width: usize,
    /// Primitive values per row: 1, or the list size of a `FixedSizeList`.
    values_per_row: usize,
}

impl RawColumn {
    /// The layout of a column of `data_type`, or `None` if a raw body cannot
    /// hold it: only integer and floating-point values, alone or in a
    /// `FixedSizeList`, qualify.
    fn of(data_type: &DataType) -> Option<Self> {
        match data_type {
            DataType::FixedSizeList(item, size) => Some(Self {
                value_width: raw_value_width(item.data_type())?,
                values_per_row: usize::try_from(*size).ok()?,
            }),
            data_type => Some(Self {
                value_width: raw_value_width(data_type)?,
                values_per_row: 1,
            }),
        }
    }

    /// Bytes of the values of `rows` rows, before padding.
    fn value_bytes(&self, rows: usize) -> Option<usize> {
        rows.checked_mul(self.values_per_row)?
            .checked_mul(self.value_width)
    }
}

fn raw_value_width(data_type: &DataType) -> Option<usize> {
    if data_type.is_integer() || data_type.is_floating() {
        data_type.primitive_width()
    } else {
        None
    }
}

fn padding(len: usize) -> usize {
    (VALUE_PADDING - len % VALUE_PADDING) % VALUE_PADDING
}

/// Write `batch` as the rest of the entry body: a raw fixed-width body when
/// every column qualifies (see the module docs), an Arrow IPC section
/// otherwise. Nothing may be written after it.
pub fn write_raw_batch(writer: &mut CacheEntryWriter<'_>, batch: &RecordBatch) -> Result<()> {
    let Some(values) = raw_values(batch) else {
        writer.write_u8(IPC_BODY_KIND)?;
        return writer.write_ipc(batch);
    };
    writer.write_u8(RAW_BODY_KIND)?;
    let schema = schema_message(batch.schema_ref());
    let schema_len = u32::try_from(schema.len()).map_err(|_| {
        Error::invalid_input(format!(
            "raw cache body schema message has {} bytes, more than a u32 length",
            schema.len()
        ))
    })?;
    let out = writer.raw_writer();
    out.write_all(&schema_len.to_le_bytes())?;
    out.write_all(&schema)?;
    out.write_all(&(batch.num_rows() as u64).to_le_bytes())?;
    for column in values {
        out.write_all(column.as_slice())?;
        out.write_all(&ZEROS[..padding(column.len())])?;
    }
    Ok(())
}

/// Read a body written by [`write_raw_batch`]: the rest of the entry body.
pub fn read_raw_batch(reader: &mut CacheEntryReader<'_>) -> Result<RecordBatch> {
    match reader.read_u8()? {
        IPC_BODY_KIND => reader.read_ipc(),
        RAW_BODY_KIND => decode_raw(&reader.body()),
        kind => Err(Error::corrupt_file_named(
            BODY_NAME,
            format!("unknown body kind {kind}"),
        )),
    }
}

/// The value bytes of every column of `batch`, or `None` if some column
/// cannot go in a raw body. Raw bodies are little-endian, which is the
/// in-memory layout only on a little-endian host.
fn raw_values(batch: &RecordBatch) -> Option<Vec<Buffer>> {
    if cfg!(target_endian = "big") {
        return None;
    }
    batch
        .columns()
        .iter()
        .map(|column| column_values(column.as_ref()))
        .collect()
}

fn column_values(array: &dyn Array) -> Option<Buffer> {
    let layout = RawColumn::of(array.data_type())?;
    if array.null_count() != 0 {
        return None;
    }
    let values = match array.data_type() {
        DataType::FixedSizeList(..) => {
            let values = array.as_fixed_size_list_opt()?.values();
            // A list's child holds exactly its rows' values once sliced.
            if values.len() != array.len().checked_mul(layout.values_per_row)? {
                return None;
            }
            values.to_data()
        }
        _ => array.to_data(),
    };
    if values.null_count() != 0 {
        return None;
    }
    let buffer = values.buffers().first()?;
    Some(buffer.slice_with_length(
        values.offset() * layout.value_width,
        values.len() * layout.value_width,
    ))
}

fn schema_message(schema: &Schema) -> Vec<u8> {
    IpcDataGenerator::default()
        .schema_to_bytes_with_dictionary_tracker(
            schema,
            &mut DictionaryTracker::new(false),
            &IpcWriteOptions::default(),
        )
        .ipc_message
}

fn parse_schema(message: &[u8]) -> Result<Schema> {
    let corrupt = |reason: String| Error::corrupt_file_named(BODY_NAME, reason);
    let message = arrow_ipc::root_as_message(message)
        .map_err(|e| corrupt(format!("invalid schema message: {e}")))?;
    let schema = message
        .header_as_schema()
        .ok_or_else(|| corrupt("schema message holds no schema".to_string()))?;
    // `fb_to_schema` assumes the field list is present.
    if schema.fields().is_none() {
        return Err(corrupt("schema message has no field list".to_string()));
    }
    Ok(arrow_ipc::convert::fb_to_schema(schema))
}

/// A bounds-checked cursor over a raw body.
struct BodyCursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> BodyCursor<'a> {
    fn take(&mut self, len: usize, what: &str) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .filter(|&end| end <= self.bytes.len())
            .ok_or_else(|| {
                Error::corrupt_file_named(
                    BODY_NAME,
                    format!(
                        "truncated at {what}: needs {len} bytes at offset {}, body has {}",
                        self.pos,
                        self.bytes.len()
                    ),
                )
            })?;
        let taken = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(taken)
    }

    fn take_array<const N: usize>(&mut self, what: &str) -> Result<[u8; N]> {
        let mut array = [0; N];
        array.copy_from_slice(self.take(N, what)?);
        Ok(array)
    }
}

fn decode_raw(body: &[u8]) -> Result<RecordBatch> {
    let mut cursor = BodyCursor {
        bytes: body,
        pos: 0,
    };
    let schema_len = u32::from_le_bytes(cursor.take_array("schema length")?) as usize;
    let schema = Arc::new(parse_schema(cursor.take(schema_len, "schema")?)?);
    let rows = u64::from_le_bytes(cursor.take_array("row count")?);
    let rows = usize::try_from(rows).map_err(|_| {
        Error::corrupt_file_named(BODY_NAME, format!("row count {rows} overflows usize"))
    })?;
    let mut columns = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        let layout = RawColumn::of(field.data_type()).ok_or_else(|| {
            Error::corrupt_file_named(
                BODY_NAME,
                format!(
                    "column {} of type {} is not fixed-width",
                    field.name(),
                    field.data_type()
                ),
            )
        })?;
        let len = layout.value_bytes(rows).ok_or_else(|| {
            Error::corrupt_file_named(
                BODY_NAME,
                format!("column {} of {rows} rows overflows usize", field.name()),
            )
        })?;
        let values = cursor.take(len, field.name())?;
        cursor.take(padding(len), field.name())?;
        columns.push(decode_column(field.data_type(), layout, rows, values)?);
    }
    if cursor.pos != body.len() {
        return Err(Error::corrupt_file_named(
            BODY_NAME,
            format!(
                "{} trailing bytes after the last column",
                body.len() - cursor.pos
            ),
        ));
    }
    Ok(RecordBatch::try_new_with_options(
        schema,
        columns,
        &RecordBatchOptions::new().with_row_count(Some(rows)),
    )?)
}

fn decode_column(
    data_type: &DataType,
    layout: RawColumn,
    rows: usize,
    values: &[u8],
) -> Result<ArrayRef> {
    // `values` holds `rows * values_per_row` values, as `value_bytes` checked.
    let len = rows * layout.values_per_row;
    match data_type {
        DataType::FixedSizeList(item, size) => {
            let child = decode_primitive(item.data_type(), layout.value_width, len, values)?;
            Ok(Arc::new(FixedSizeListArray::try_new(
                item.clone(),
                *size,
                child,
                None,
            )?))
        }
        data_type => decode_primitive(data_type, layout.value_width, len, values),
    }
}

fn decode_primitive(
    data_type: &DataType,
    value_width: usize,
    len: usize,
    values: &[u8],
) -> Result<ArrayRef> {
    let buffer = match value_width {
        1 => Buffer::from_vec(values.to_vec()),
        2 => Buffer::from_vec(le_values(values, u16::from_le_bytes)),
        4 => Buffer::from_vec(le_values(values, u32::from_le_bytes)),
        8 => Buffer::from_vec(le_values(values, u64::from_le_bytes)),
        _ => {
            return Err(Error::corrupt_file_named(
                BODY_NAME,
                format!("values of type {data_type} are {value_width} bytes wide"),
            ));
        }
    };
    let data = ArrayData::builder(data_type.clone())
        .len(len)
        .add_buffer(buffer)
        .build()?;
    Ok(make_array(data))
}

/// `bytes` as little-endian `N`-byte values, in a vector of exactly their
/// count, aligned for `T`.
fn le_values<const N: usize, T>(bytes: &[u8], from_le_bytes: fn([u8; N]) -> T) -> Vec<T> {
    bytes
        .chunks_exact(N)
        .map(|chunk| {
            let mut value = [0; N];
            value.copy_from_slice(chunk);
            from_le_bytes(value)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{Float32Array, StringArray, UInt8Array, UInt64Array};
    use arrow_schema::Field;
    use bytes::Bytes;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::deepsize::DeepSizeOf;
    use std::collections::HashMap;

    use crate::vector::bq::layered::{
        FULL_BOUNDS_COLUMN, HIGH_ADD_FACTORS_COLUMN, HIGH_BOUNDS_COLUMN, HIGH_SCALE_FACTORS_COLUMN,
        SIGN_BOUNDS_PLANE, SignBounds, plane_columns,
    };
    use crate::vector::bq::storage::{
        RABIT_BLOCKED_EX_CODE_COLUMN, RABIT_BLOCKED_EX_CODE_LO_COLUMN, RABIT_CODE_COLUMN,
    };
    use crate::vector::bq::transform::{
        ADD_FACTORS_COLUMN, ERROR_FACTORS_COLUMN, EX_ADD_FACTORS_COLUMN, EX_SCALE_FACTORS_COLUMN,
        SCALE_FACTORS_COLUMN,
    };

    /// Sign code bytes of a 1024-dimensional index, as on MS MARCO.
    const SIGN_CODE_BYTES: i32 = 128;
    /// Blocked ex-code bytes of a 1024-dimensional RQ7 (6 ex bits) index.
    const EX_CODE_BYTES: i32 = 768;
    const LOW_CODE_BYTES: i32 = 256;
    const BOUNDS_VALUES: i32 = 3;

    /// A column of an RQ cache entry with deterministic, row-dependent values.
    fn column(name: &str, rows: usize) -> (Field, ArrayRef) {
        let codes = |width: i32| -> ArrayRef {
            let values = (0..rows * width as usize)
                .map(|v| (v % 251) as u8)
                .collect::<Vec<_>>();
            Arc::new(
                FixedSizeListArray::try_new_from_values(UInt8Array::from(values), width).unwrap(),
            )
        };
        let array: ArrayRef = match name {
            lance_core::ROW_ID => Arc::new(UInt64Array::from(
                (0..rows as u64)
                    .map(|v| v * 7 + (1 << 40))
                    .collect::<Vec<_>>(),
            )),
            RABIT_CODE_COLUMN => codes(SIGN_CODE_BYTES),
            RABIT_BLOCKED_EX_CODE_COLUMN => codes(EX_CODE_BYTES),
            RABIT_BLOCKED_EX_CODE_LO_COLUMN => codes(LOW_CODE_BYTES),
            HIGH_BOUNDS_COLUMN | FULL_BOUNDS_COLUMN => {
                let values = (0..rows * BOUNDS_VALUES as usize)
                    .map(|v| v as f32 * 0.5 - 3.0)
                    .collect::<Vec<_>>();
                Arc::new(
                    FixedSizeListArray::try_new_from_values(
                        Float32Array::from(values),
                        BOUNDS_VALUES,
                    )
                    .unwrap(),
                )
            }
            _ => Arc::new(Float32Array::from(
                (0..rows).map(|v| v as f32 * 0.25 + 1.0).collect::<Vec<_>>(),
            )),
        };
        (Field::new(name, array.data_type().clone(), true), array)
    }

    fn batch(names: &[&str], rows: usize) -> RecordBatch {
        let (fields, columns): (Vec<_>, Vec<_>) =
            names.iter().map(|name| column(name, rows)).unzip();
        let metadata = HashMap::from([("lance:rq".to_string(), "entry".to_string())]);
        RecordBatch::try_new(
            Arc::new(Schema::new_with_metadata(fields, metadata)),
            columns,
        )
        .unwrap()
    }

    /// The column sets of every sign, bounds and native entry: full entries
    /// and code-only ones (the file columns), under both bounds placements.
    fn entry_schemas() -> Vec<(&'static str, Vec<&'static str>)> {
        let native = vec![
            lance_core::ROW_ID,
            RABIT_CODE_COLUMN,
            ADD_FACTORS_COLUMN,
            SCALE_FACTORS_COLUMN,
            ERROR_FACTORS_COLUMN,
            RABIT_BLOCKED_EX_CODE_COLUMN,
            EX_ADD_FACTORS_COLUMN,
            EX_SCALE_FACTORS_COLUMN,
        ];
        let layered = vec![
            lance_core::ROW_ID,
            RABIT_CODE_COLUMN,
            ADD_FACTORS_COLUMN,
            SCALE_FACTORS_COLUMN,
            ERROR_FACTORS_COLUMN,
            RABIT_BLOCKED_EX_CODE_COLUMN,
            RABIT_BLOCKED_EX_CODE_LO_COLUMN,
            HIGH_ADD_FACTORS_COLUMN,
            HIGH_SCALE_FACTORS_COLUMN,
            EX_ADD_FACTORS_COLUMN,
            EX_SCALE_FACTORS_COLUMN,
            HIGH_BOUNDS_COLUMN,
            FULL_BOUNDS_COLUMN,
        ];
        let is_code = |name: &&str| {
            [
                RABIT_CODE_COLUMN,
                RABIT_BLOCKED_EX_CODE_COLUMN,
                RABIT_BLOCKED_EX_CODE_LO_COLUMN,
                HIGH_BOUNDS_COLUMN,
                FULL_BOUNDS_COLUMN,
            ]
            .contains(name)
        };
        let code_only = |names: &[&'static str]| -> Vec<&'static str> {
            names.iter().copied().filter(is_code).collect()
        };
        let sign_lazy = plane_columns(0, SignBounds::Lazy).to_vec();
        let sign_eager = plane_columns(0, SignBounds::Eager).to_vec();
        let bounds = plane_columns(SIGN_BOUNDS_PLANE, SignBounds::Lazy).to_vec();
        vec![
            ("sign lazy codes", code_only(&sign_lazy)),
            ("sign eager codes", code_only(&sign_eager)),
            ("sign lazy", sign_lazy),
            ("sign eager", sign_eager),
            ("bounds", bounds),
            ("native codes", code_only(&native)),
            ("native", native),
            ("layered partition codes", code_only(&layered)),
            ("layered partition", layered),
        ]
    }

    fn encode(batch: &RecordBatch) -> Bytes {
        let mut buf = Vec::new();
        write_raw_batch(&mut CacheEntryWriter::new(&mut buf), batch).unwrap();
        Bytes::from(buf)
    }

    fn decode(bytes: &Bytes) -> Result<RecordBatch> {
        read_raw_batch(&mut CacheEntryReader::new(bytes, 0, 1))
    }

    /// Bytes one row of `schema` holds in a raw body: its logical width.
    fn row_bytes(schema: &Schema) -> Vec<usize> {
        schema
            .fields()
            .iter()
            .map(|field| {
                let layout = RawColumn::of(field.data_type()).unwrap();
                layout.value_width * layout.values_per_row
            })
            .collect()
    }

    #[test]
    fn entry_batches_round_trip_with_schema_and_size() {
        for (name, columns) in entry_schemas() {
            for rows in [0, 1, 7, 300] {
                let source = batch(&columns, rows);
                let encoded = encode(&source);
                assert_eq!(encoded[0], RAW_BODY_KIND, "{name} {rows}");
                let decoded = decode(&encoded).unwrap();
                assert_eq!(decoded, source, "{name} {rows}");
                assert_eq!(decoded.schema(), source.schema(), "{name} {rows}");
                assert_eq!(
                    decoded.deep_size_of(),
                    source.deep_size_of(),
                    "{name} {rows}"
                );
            }
        }
    }

    #[test]
    fn body_holds_schema_and_logical_row_bytes_only() {
        for (name, columns) in entry_schemas() {
            let rows = 301;
            let source = batch(&columns, rows);
            let encoded = encode(&source);
            let schema_len = u32::from_le_bytes(encoded[1..5].try_into().unwrap()) as usize;
            assert_eq!(
                schema_len,
                schema_message(source.schema_ref()).len(),
                "{name}"
            );
            let row_bytes = row_bytes(source.schema_ref());
            let values: usize = row_bytes.iter().map(|bytes| rows * bytes).sum();
            let padding: usize = row_bytes.iter().map(|bytes| padding(rows * bytes)).sum();
            assert!(padding < VALUE_PADDING * row_bytes.len(), "{name}");
            // Kind, schema length, schema, row count, then the values: no
            // validity bitmap, which IPC adds per column.
            assert_eq!(
                encoded.len(),
                1 + 4 + schema_len + 8 + values + padding,
                "{name}"
            );
        }
    }

    #[test]
    fn sliced_columns_write_their_rows_only() {
        let columns = plane_columns(0, SignBounds::Eager);
        let sliced = batch(columns, 50).slice(7, 30);
        let encoded = encode(&sliced);
        assert_eq!(decode(&encoded).unwrap(), sliced);
        assert_eq!(encoded.len(), encode(&batch(columns, 30)).len());
    }

    #[test]
    fn other_batches_fall_back_to_ipc() {
        let with_nulls = RecordBatch::try_from_iter([(
            ADD_FACTORS_COLUMN,
            Arc::new(Float32Array::from(vec![Some(1.0), None, Some(3.0)])) as ArrayRef,
        )])
        .unwrap();
        let null_child = RecordBatch::try_from_iter([(
            HIGH_BOUNDS_COLUMN,
            Arc::new(
                FixedSizeListArray::try_new_from_values(
                    Float32Array::from(vec![Some(1.0), None, Some(3.0)]),
                    BOUNDS_VALUES,
                )
                .unwrap(),
            ) as ArrayRef,
        )])
        .unwrap();
        let variable_width = RecordBatch::try_from_iter([
            (
                lance_core::ROW_ID,
                Arc::new(UInt64Array::from(vec![1u64, 2])) as ArrayRef,
            ),
            (
                "name",
                Arc::new(StringArray::from(vec!["a", "bc"])) as ArrayRef,
            ),
        ])
        .unwrap();
        for source in [with_nulls, null_child, variable_width] {
            let encoded = encode(&source);
            assert_eq!(encoded[0], IPC_BODY_KIND);
            assert_eq!(decode(&encoded).unwrap(), source);
        }
    }

    #[test]
    fn truncated_or_corrupt_bodies_are_errors() {
        let source = batch(plane_columns(0, SignBounds::Eager), 5);
        let encoded = encode(&source);
        for len in 0..encoded.len() {
            assert!(decode(&encoded.slice(..len)).is_err(), "truncated to {len}");
        }
        let mut trailing = encoded.to_vec();
        trailing.push(0);
        let err = decode(&Bytes::from(trailing)).unwrap_err().to_string();
        assert!(err.contains("trailing"), "{err}");
        let mut unknown_kind = encoded.to_vec();
        unknown_kind[0] = 7;
        let err = decode(&Bytes::from(unknown_kind)).unwrap_err().to_string();
        assert!(err.contains("unknown body kind 7"), "{err}");
        let mut bad_schema = encoded.to_vec();
        bad_schema[5..13].fill(0xFF);
        assert!(decode(&Bytes::from(bad_schema)).is_err());
    }
}
