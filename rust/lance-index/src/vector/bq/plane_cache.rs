// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Persistent plane entries. The ex planes (high and low) are
//! row-addressable: fixed-stride rows of codes, each followed by its
//! factors, so a sparse gather reads only its rows. The sign plane and the
//! bounds plane are read whole and keep a raw fixed-width body (see
//! [`super::raw_body`]); sign codes keep their partition-local transposition.
//!
//! Body layouts by entry version:
//!
//! ```text
//! v1           : Arrow IPC section, every plane
//! v2 sign      : [plane: u8 = 0][Arrow IPC section], the bounds plane too
//! v2 ex plane  : [plane: u8][rows: u64 LE][width: u32 LE][rows]
//! v3 sign      : [plane: u8 = 0, or 3 for the bounds plane][raw body]
//! v3 ex plane  : [plane: u8][factor_columns: u8][rows: u64 LE][width: u32 LE][rows]
//! ex-plane row : [codes: width bytes][add: f32 LE][scale: f32 LE]  2 factor columns
//!                [codes: width bytes]                              0 factor columns
//! ```
//!
//! v2 ex-plane rows always carry both factors. A code-only entry
//! (`EntryColumns::Codes`) holds the file columns of its plane alone: its ex
//! plane rows carry no factor columns, and its sign plane only the codes (and
//! the bounds under eager bounds).
use std::ops::Range;
use std::sync::Arc;

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, UInt8Array,
    cast::AsArray,
    types::{Float32Type, UInt8Type},
};
use arrow_schema::{DataType, Field, Schema};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::cache::{CacheCodecImpl, CacheEntryReader, CacheEntryWriter, CacheRangeReader};
use lance_core::{Error, Result};

use super::layered::{PlaneBatch, SIGN_BOUNDS_PLANE, SignBounds, plane_columns};
use super::raw_body::{read_raw_batch, write_raw_batch};
use super::storage::{RABIT_BLOCKED_EX_CODE_COLUMN, RABIT_BLOCKED_EX_CODE_LO_COLUMN};

/// Body layouts of the plane entry versions this build reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BodyVersion {
    /// Every plane as an Arrow IPC section.
    Ipc = 1,
    /// Ex planes as rows with both factors behind a 13-byte header; the sign
    /// and bounds planes as a tagged Arrow IPC section.
    RowsIpcSign = 2,
    /// Ex planes as rows with 0 or 2 factor columns behind a 14-byte header;
    /// the sign and bounds planes as a tagged raw body.
    RowsRawSign = 3,
}

impl BodyVersion {
    fn of(version: u32) -> Result<Self> {
        match version {
            1 => Ok(Self::Ipc),
            2 => Ok(Self::RowsIpcSign),
            3 => Ok(Self::RowsRawSign),
            _ => Err(Error::invalid_input(format!(
                "unsupported plane cache version {version}"
            ))),
        }
    }

    /// Bytes of the header ahead of an ex plane's rows.
    fn ex_header_bytes(self) -> Result<usize> {
        match self {
            Self::Ipc => Err(Error::invalid_input(
                "plane cache version has no row directory",
            )),
            Self::RowsIpcSign => Ok(V2_HEADER_BYTES),
            Self::RowsRawSign => Ok(V3_HEADER_BYTES),
        }
    }
}

const V2_HEADER_BYTES: usize = 13;
const V3_HEADER_BYTES: usize = 14;
/// Factor columns of an ex-plane row with its add and scale factors.
const EX_FACTOR_COLUMNS: u8 = 2;
const FACTOR_BYTES: usize = size_of::<f32>();
const COALESCE_GAP_BYTES: usize = 4096;

struct Header {
    plane: u8,
    /// 0 or [`EX_FACTOR_COLUMNS`].
    factor_columns: u8,
    rows: usize,
    width: usize,
    /// Bytes of the header itself; the rows follow it.
    len: usize,
}

impl Header {
    fn parse(bytes: &[u8], version: BodyVersion) -> Result<Self> {
        let invalid = || Error::invalid_input("invalid ex-plane cache header");
        let len = version.ex_header_bytes()?;
        let header = bytes.get(..len).ok_or_else(invalid)?;
        // v3 adds the factor column count after the plane; v2 rows always
        // carry both factors.
        let (factor_columns, fields) = match version {
            BodyVersion::RowsRawSign => (header[1], &header[2..]),
            _ => (EX_FACTOR_COLUMNS, &header[1..]),
        };
        let plane = header[0];
        if !matches!(plane, 1 | 2) || !matches!(factor_columns, 0 | EX_FACTOR_COLUMNS) {
            return Err(invalid());
        }
        let (rows, width) = fields.split_at(size_of::<u64>());
        let rows = usize::try_from(u64::from_le_bytes(rows.try_into().map_err(|_| invalid())?))
            .map_err(|_| Error::invalid_input("plane row count overflow"))?;
        let width = u32::from_le_bytes(width.try_into().map_err(|_| invalid())?) as usize;
        if width == 0 || width > i32::MAX as usize {
            return Err(Error::invalid_input("invalid cached code width"));
        }
        Ok(Self {
            plane,
            factor_columns,
            rows,
            width,
            len,
        })
    }

    fn stride(&self) -> usize {
        self.width + FACTOR_BYTES * self.factor_columns as usize
    }

    fn batch(&self, codes: Vec<u8>, adds: Vec<f32>, scales: Vec<f32>) -> Result<PlaneBatch> {
        // Ex planes hold the same columns under either bounds placement.
        let names = plane_columns(self.plane, SignBounds::default());
        let codes =
            FixedSizeListArray::try_new_from_values(UInt8Array::from(codes), self.width as i32)?;
        let mut fields = vec![Field::new(names[0], codes.data_type().clone(), true)];
        let mut columns: Vec<Arc<dyn Array>> = vec![Arc::new(codes)];
        if self.factor_columns == EX_FACTOR_COLUMNS {
            for (name, values) in names[1..].iter().zip([adds, scales]) {
                fields.push(Field::new(*name, DataType::Float32, true));
                columns.push(Arc::new(Float32Array::from(values)));
            }
        }
        Ok(PlaneBatch(RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            columns,
        )?))
    }

    fn decode(&self, bytes: &[u8]) -> Result<PlaneBatch> {
        let size = self
            .rows
            .checked_mul(self.stride())
            .ok_or_else(|| Error::invalid_input("plane size overflow"))?;
        if bytes.len() != size {
            return Err(Error::invalid_input("truncated ex-plane cache body"));
        }
        let factor_rows = if self.factor_columns == EX_FACTOR_COLUMNS {
            self.rows
        } else {
            0
        };
        let mut codes = Vec::with_capacity(self.rows * self.width);
        let mut adds = Vec::with_capacity(factor_rows);
        let mut scales = Vec::with_capacity(factor_rows);
        for row in bytes.chunks_exact(self.stride()) {
            self.append(row, &mut codes, &mut adds, &mut scales);
        }
        self.batch(codes, adds, scales)
    }

    /// Append one row of [`Self::stride`] bytes.
    fn append(&self, row: &[u8], codes: &mut Vec<u8>, adds: &mut Vec<f32>, scales: &mut Vec<f32>) {
        let (row_codes, factors) = row.split_at(self.width);
        codes.extend_from_slice(row_codes);
        if self.factor_columns == EX_FACTOR_COLUMNS {
            let (add, scale) = factors.split_at(FACTOR_BYTES);
            adds.push(le_f32(add));
            scales.push(le_f32(scale));
        }
    }
}

/// The little-endian `f32` at the start of `bytes`.
fn le_f32(bytes: &[u8]) -> f32 {
    let mut value = [0; FACTOR_BYTES];
    value.copy_from_slice(&bytes[..FACTOR_BYTES]);
    f32::from_le_bytes(value)
}

/// The ex plane (1 or 2) whose codes `batch` holds, or `None` for the sign
/// and bounds planes.
fn ex_plane(batch: &RecordBatch) -> Option<u8> {
    if batch
        .column_by_name(RABIT_BLOCKED_EX_CODE_LO_COLUMN)
        .is_some()
    {
        Some(2)
    } else if batch.column_by_name(RABIT_BLOCKED_EX_CODE_COLUMN).is_some() {
        Some(1)
    } else {
        None
    }
}

/// The plane tag of a sign or bounds plane batch: [`SIGN_BOUNDS_PLANE`] for
/// exactly the bounds plane's columns, 0 otherwise.
fn whole_plane(batch: &RecordBatch) -> u8 {
    let bounds = plane_columns(SIGN_BOUNDS_PLANE, SignBounds::Lazy);
    let schema = batch.schema_ref();
    if schema.fields().len() == bounds.len()
        && schema
            .fields()
            .iter()
            .zip(bounds)
            .all(|(field, name)| field.name() == name)
    {
        SIGN_BOUNDS_PLANE
    } else {
        0
    }
}

/// A float factor column of an ex-plane batch, which must hold no nulls.
fn factor_values<'a>(batch: &'a RecordBatch, name: &str) -> Result<Option<&'a [f32]>> {
    let Some(column) = batch.column_by_name(name) else {
        return Ok(None);
    };
    let values = column
        .as_primitive_opt::<Float32Type>()
        .filter(|values| values.null_count() == 0)
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "cached ex-plane factor column {name} must be non-null float32, got {}",
                column.data_type()
            ))
        })?;
    let values: &[f32] = values.values();
    Ok(Some(values))
}

impl PlaneBatch {
    /// Write an ex plane as rows. Its factor columns are both present (2) or
    /// both absent (0).
    fn serialize_ex_plane(&self, plane: u8, writer: &mut CacheEntryWriter<'_>) -> Result<()> {
        let names = plane_columns(plane, SignBounds::default());
        let batch = &self.0;
        let codes = batch
            .column_by_name(names[0])
            .and_then(|codes| codes.as_fixed_size_list_opt())
            .filter(|codes| codes.null_count() == 0)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "cached ex plane {plane} needs a non-null fixed-size list column {}",
                    names[0]
                ))
            })?;
        let values = codes
            .values()
            .as_primitive_opt::<UInt8Type>()
            .filter(|values| values.null_count() == 0)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "cached ex-plane codes {} must be non-null uint8, got {}",
                    names[0],
                    codes.value_type()
                ))
            })?
            .values();
        let factors = match (
            factor_values(batch, names[1])?,
            factor_values(batch, names[2])?,
        ) {
            (Some(adds), Some(scales)) => Some((adds, scales)),
            (None, None) => None,
            _ => {
                return Err(Error::invalid_input(format!(
                    "cached ex plane {plane} has only one of its factor columns {} and {}",
                    names[1], names[2]
                )));
            }
        };
        let width = codes.value_length() as usize;
        let width_u32 = u32::try_from(width)
            .ok()
            .filter(|&width| width > 0)
            .ok_or_else(|| {
                Error::invalid_input(format!("invalid cached ex-plane code width {width}"))
            })?;
        writer.write_u8(plane)?;
        writer.write_u8(if factors.is_some() {
            EX_FACTOR_COLUMNS
        } else {
            0
        })?;
        let writer = writer.raw_writer();
        writer.write_all(&(batch.num_rows() as u64).to_le_bytes())?;
        writer.write_all(&width_u32.to_le_bytes())?;
        for row in 0..batch.num_rows() {
            writer.write_all(&values[row * width..(row + 1) * width])?;
            if let Some((adds, scales)) = factors {
                writer.write_all(&adds[row].to_le_bytes())?;
                writer.write_all(&scales[row].to_le_bytes())?;
            }
        }
        Ok(())
    }
}

impl CacheCodecImpl for PlaneBatch {
    const TYPE_ID: &'static str = "lance.vector.rq.plane";
    const CURRENT_VERSION: u32 = BodyVersion::RowsRawSign as u32;
    const SUPPORTS_ROW_SELECTION: bool = true;

    fn serialize(&self, writer: &mut CacheEntryWriter<'_>) -> Result<()> {
        match ex_plane(&self.0) {
            Some(plane) => self.serialize_ex_plane(plane, writer),
            None => {
                writer.write_u8(whole_plane(&self.0))?;
                write_raw_batch(writer, &self.0)
            }
        }
    }

    fn deserialize(reader: &mut CacheEntryReader<'_>) -> Result<Self> {
        let version = BodyVersion::of(reader.version())?;
        let body = reader.body();
        match (version, body.first().copied()) {
            (BodyVersion::Ipc, _) => Ok(Self(reader.read_ipc()?)),
            (BodyVersion::RowsIpcSign, Some(0)) => {
                reader.read_u8()?;
                Ok(Self(reader.read_ipc()?))
            }
            (BodyVersion::RowsRawSign, Some(0 | SIGN_BOUNDS_PLANE)) => {
                reader.read_u8()?;
                Ok(Self(read_raw_batch(reader)?))
            }
            _ => {
                let header = Header::parse(&body, version)?;
                header.decode(&body[header.len..])
            }
        }
    }

    fn deserialize_rows(
        reader: &dyn CacheRangeReader,
        offset: usize,
        version: u32,
        rows: &[u32],
    ) -> Result<Self> {
        let header = read_row_header(reader, offset, version)?;
        let stride = header.stride();
        let mut codes = Vec::new();
        let mut adds = Vec::new();
        let mut scales = Vec::new();
        for (bytes_range, run) in row_runs(&header, offset, rows)? {
            let first = rows[run.start] as usize;
            let len = bytes_range.len();
            let bytes = reader.read_range(bytes_range)?;
            if bytes.len() != len {
                return Err(Error::invalid_input("short plane range read"));
            }
            for &row in &rows[run] {
                let local = (row as usize - first) * stride;
                header.append(
                    &bytes[local..local + stride],
                    &mut codes,
                    &mut adds,
                    &mut scales,
                );
            }
        }
        header.batch(codes, adds, scales)
    }

    fn plan_row_ranges(
        reader: &dyn CacheRangeReader,
        offset: usize,
        version: u32,
        rows: &[u32],
    ) -> Result<Option<Vec<Range<usize>>>> {
        let header = read_row_header(reader, offset, version)?;
        Ok(Some(
            row_runs(&header, offset, rows)?
                .into_iter()
                .map(|(bytes, _)| bytes)
                .collect(),
        ))
    }
}

fn read_row_header(reader: &dyn CacheRangeReader, offset: usize, version: u32) -> Result<Header> {
    let version = BodyVersion::of(version)?;
    let end = offset
        .checked_add(version.ex_header_bytes()?)
        .ok_or_else(|| Error::invalid_input("plane offset overflow"))?;
    Header::parse(&reader.read_range(offset..end)?, version)
}

/// Payload byte ranges for sorted, unique `rows`, each paired with the span of
/// `rows` it covers. Rows closer than [`COALESCE_GAP_BYTES`] share one range.
fn row_runs(
    header: &Header,
    offset: usize,
    rows: &[u32],
) -> Result<Vec<(Range<usize>, Range<usize>)>> {
    if rows.windows(2).any(|pair| pair[0] >= pair[1])
        || rows.last().is_some_and(|&row| row as usize >= header.rows)
    {
        return Err(Error::invalid_input("invalid cached candidate offsets"));
    }
    let stride = header.stride();
    let base = offset
        .checked_add(header.len)
        .ok_or_else(|| Error::invalid_input("plane offset overflow"))?;
    let byte_offset = |row: usize| {
        row.checked_mul(stride)
            .and_then(|v| base.checked_add(v))
            .ok_or_else(|| Error::invalid_input("plane offset overflow"))
    };
    let mut runs = Vec::new();
    let mut start = 0;
    while start < rows.len() {
        let mut end = start + 1;
        while end < rows.len()
            && (rows[end] - rows[end - 1] - 1) as usize <= COALESCE_GAP_BYTES / stride
        {
            end += 1;
        }
        let bytes = byte_offset(rows[start] as usize)?..byte_offset(rows[end - 1] as usize + 1)?;
        runs.push((bytes, start..end));
        start = end;
    }
    Ok(runs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::bq::layered::{
        EntryColumns, FULL_BOUNDS_COLUMN, HIGH_BOUNDS_COLUMN, PlaneKey, plane_entry_columns,
    };
    use crate::vector::bq::raw_body::RAW_BODY_KIND;
    use crate::vector::bq::storage::RABIT_CODE_COLUMN;
    use arrow_array::{ArrayRef, UInt64Array};
    use bytes::Bytes;
    use lance_arrow::RecordBatchExt;
    use lance_core::cache::{CacheCodec, CacheDecode, CacheKey, CacheMissReason};
    use std::cell::{Cell, RefCell};

    /// Ex-plane code widths, including one that is not a multiple of 8.
    const WIDTHS: [usize; 4] = [56, 128, 512, 1024];

    /// Envelope bytes ahead of a plane entry's body.
    fn body_offset() -> usize {
        4 + 1 + 2 + PlaneBatch::TYPE_ID.len() + 4
    }

    fn encode(codec: &CacheCodec, batch: &RecordBatch) -> Bytes {
        let mut encoded = Vec::new();
        codec
            .serialize(
                &(Arc::new(PlaneBatch(batch.clone())) as Arc<dyn std::any::Any + Send + Sync>),
                &mut encoded,
            )
            .unwrap();
        Bytes::from(encoded)
    }

    fn decode(codec: &CacheCodec, encoded: &Bytes) -> CacheDecode<RecordBatch> {
        match codec.deserialize(encoded) {
            CacheDecode::Hit(decoded) => {
                CacheDecode::Hit(decoded.downcast::<PlaneBatch>().unwrap().0.clone())
            }
            CacheDecode::Miss(reason) => CacheDecode::Miss(reason),
        }
    }

    fn decode_rows(codec: &CacheCodec, encoded: &Bytes, rows: &[u32]) -> CacheDecode<RecordBatch> {
        let read = |range: Range<usize>| -> Result<Bytes> { Ok(encoded.slice(range)) };
        match codec.deserialize_rows(&read, rows) {
            CacheDecode::Hit(decoded) => {
                CacheDecode::Hit(decoded.downcast::<PlaneBatch>().unwrap().0.clone())
            }
            CacheDecode::Miss(reason) => CacheDecode::Miss(reason),
        }
    }

    fn ex_plane_batch(plane: u8, rows: usize, width: usize, factor_columns: u8) -> RecordBatch {
        Header {
            plane,
            factor_columns,
            rows,
            width,
            len: V3_HEADER_BYTES,
        }
        .batch(
            (0..rows * width).map(|v| (v % 251) as u8).collect(),
            (0..rows).map(|v| v as f32).collect(),
            (0..rows).map(|v| v as f32 * 0.5 + 2.0).collect(),
        )
        .unwrap()
        .0
    }

    /// A sign or bounds plane batch with the columns `names`.
    fn whole_plane_batch(names: &[&str], rows: usize) -> RecordBatch {
        let columns = names
            .iter()
            .map(|name| {
                let array: ArrayRef = match *name {
                    lance_core::ROW_ID => Arc::new(UInt64Array::from(
                        (0..rows as u64).map(|v| v * 3).collect::<Vec<_>>(),
                    )),
                    RABIT_CODE_COLUMN => Arc::new(
                        FixedSizeListArray::try_new_from_values(
                            UInt8Array::from(
                                (0..rows * 16).map(|v| (v % 253) as u8).collect::<Vec<_>>(),
                            ),
                            16,
                        )
                        .unwrap(),
                    ),
                    HIGH_BOUNDS_COLUMN | FULL_BOUNDS_COLUMN => Arc::new(
                        FixedSizeListArray::try_new_from_values(
                            Float32Array::from(
                                (0..rows * 3).map(|v| v as f32 - 1.5).collect::<Vec<_>>(),
                            ),
                            3,
                        )
                        .unwrap(),
                    ),
                    _ => Arc::new(Float32Array::from(
                        (0..rows).map(|v| v as f32 + 0.25).collect::<Vec<_>>(),
                    )),
                };
                (*name, array)
            })
            .collect::<Vec<_>>();
        RecordBatch::try_from_iter(columns).unwrap()
    }

    #[test]
    fn ex_planes_round_trip_whole_and_by_rows() {
        const ROWS: usize = 700;
        let codec = CacheCodec::from_impl::<PlaneBatch>();
        for plane in [1, 2] {
            for width in WIDTHS {
                for factor_columns in [EX_FACTOR_COLUMNS, 0] {
                    let context = format!("plane {plane} width {width} factors {factor_columns}");
                    let original = ex_plane_batch(plane, ROWS, width, factor_columns);
                    let encoded = encode(&codec, &original);
                    let body = &encoded[body_offset()..];
                    assert_eq!(&body[..2], &[plane, factor_columns], "{context}");
                    let stride = width + FACTOR_BYTES * factor_columns as usize;
                    assert_eq!(body.len(), V3_HEADER_BYTES + ROWS * stride, "{context}");
                    let CacheDecode::Hit(full) = decode(&codec, &encoded) else {
                        panic!("whole decode failed: {context}")
                    };
                    assert_eq!(full, original, "{context}");
                    for rows in [
                        vec![],
                        vec![0],
                        vec![ROWS as u32 - 1],
                        vec![0, 1, 350, 351, 699],
                        (0..ROWS as u32).step_by(3).collect(),
                        (0..ROWS as u32).collect(),
                    ] {
                        let CacheDecode::Hit(selected) = decode_rows(&codec, &encoded, &rows)
                        else {
                            panic!("row decode failed: {context} {rows:?}")
                        };
                        assert_eq!(
                            selected,
                            original
                                .take(&arrow_array::UInt32Array::from(rows.clone()))
                                .unwrap(),
                            "{context} {rows:?}"
                        );
                    }
                    for rows in [vec![ROWS as u32], vec![1, 1], vec![2, 1]] {
                        assert!(
                            matches!(decode_rows(&codec, &encoded, &rows), CacheDecode::Miss(_)),
                            "{context} {rows:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn sparse_plane_cache_reads_match_full_decode() {
        for plane in [1, 2] {
            let original = ex_plane_batch(plane, 4096, 128, EX_FACTOR_COLUMNS);
            let codec = CacheCodec::from_impl::<PlaneBatch>();
            let encoded = encode(&codec, &original);
            let bytes_read = Cell::new(0);
            let read = |range: std::ops::Range<usize>| -> Result<Bytes> {
                bytes_read.set(bytes_read.get() + range.len());
                Ok(encoded.slice(range))
            };
            let rows = [0, 1, 2048, 4095];
            let CacheDecode::Hit(selected) = codec.deserialize_rows(&read, &rows) else {
                panic!("range decode failed")
            };
            assert_eq!(
                selected.downcast::<PlaneBatch>().unwrap().0,
                original
                    .take(&arrow_array::UInt32Array::from(rows.to_vec()))
                    .unwrap()
            );
            assert!(bytes_read.get() < encoded.len() / 10);
        }
    }

    #[test]
    fn row_plan_matches_ranges_read_by_decode() {
        const ROWS: usize = 4096;
        for width in WIDTHS {
            for factor_columns in [EX_FACTOR_COLUMNS, 0] {
                let original = ex_plane_batch(1, ROWS, width, factor_columns);
                let codec = CacheCodec::from_impl::<PlaneBatch>();
                let encoded = encode(&codec, &original);
                let requested = RefCell::new(Vec::new());
                let read = |range: Range<usize>| -> Result<Bytes> {
                    requested.borrow_mut().push(range.clone());
                    Ok(encoded.slice(range))
                };
                let stride = width + FACTOR_BYTES * factor_columns as usize;
                let gap = COALESCE_GAP_BYTES / stride;
                let row_sets: Vec<Vec<u32>> = vec![
                    vec![],
                    vec![0],
                    vec![ROWS as u32 - 1],
                    (0..ROWS as u32).collect(),
                    (0..ROWS as u32).step_by(gap + 1).collect(),
                    (0..ROWS as u32).step_by(gap + 2).collect(),
                    vec![1, 2, 3, 900, 901, 3000, 4095],
                ];
                for rows in row_sets {
                    requested.borrow_mut().clear();
                    let CacheDecode::Hit(_) = codec.deserialize_rows(&read, &rows) else {
                        panic!("range decode failed for {rows:?}")
                    };
                    // The envelope and body header come first; the rest are row reads.
                    let decoded_rows = requested.borrow()[3..].to_vec();
                    let plan = codec.plan_rows(&read, &rows).expect("plane codec plans");
                    assert_eq!(plan, decoded_rows);
                    let expected_runs = if rows.is_empty() {
                        0
                    } else {
                        1 + rows
                            .windows(2)
                            .filter(|w| (w[1] - w[0] - 1) as usize > gap)
                            .count()
                    };
                    assert_eq!(plan.len(), expected_runs, "width {width} {rows:?}");
                }
                assert!(codec.plan_rows(&read, &[5, 5]).is_none());
                assert!(codec.plan_rows(&read, &[ROWS as u32]).is_none());
            }
        }
    }

    /// Sign and bounds plane entries hold a raw body: they round trip whole,
    /// with their schema, and have no row directory, so a row read misses.
    #[test]
    fn sign_and_bounds_planes_round_trip_through_raw_bodies() {
        let sign_lazy = plane_columns(0, SignBounds::Lazy);
        let sign_eager = plane_columns(0, SignBounds::Eager);
        let bounds = plane_columns(SIGN_BOUNDS_PLANE, SignBounds::Lazy);
        let cases: [(&[&str], u8, SignBounds); 5] = [
            (sign_lazy, 0, SignBounds::Lazy),
            (sign_eager, 0, SignBounds::Eager),
            (bounds, SIGN_BOUNDS_PLANE, SignBounds::Lazy),
            // Code-only sign entries: the file columns alone.
            (&[RABIT_CODE_COLUMN], 0, SignBounds::Lazy),
            (
                &[RABIT_CODE_COLUMN, HIGH_BOUNDS_COLUMN, FULL_BOUNDS_COLUMN],
                0,
                SignBounds::Eager,
            ),
        ];
        for (names, tag, sign_bounds) in cases {
            for rows in [0, 70] {
                let codec = PlaneKey {
                    partition: 0,
                    plane: tag,
                    sign_bounds,
                    entry_columns: EntryColumns::default(),
                }
                .codec_for_key()
                .unwrap();
                let original = whole_plane_batch(names, rows);
                let encoded = encode(&codec, &original);
                let body = &encoded[body_offset()..];
                assert_eq!(&body[..2], &[tag, RAW_BODY_KIND], "{names:?}");
                let CacheDecode::Hit(decoded) = decode(&codec, &encoded) else {
                    panic!("{names:?} decode failed")
                };
                assert_eq!(decoded, original, "{names:?}");
                assert_eq!(decoded.schema(), original.schema(), "{names:?}");
                assert!(matches!(
                    decode_rows(&codec, &encoded, &[0]),
                    CacheDecode::Miss(_)
                ));
                let read = |range: Range<usize>| -> Result<Bytes> { Ok(encoded.slice(range)) };
                assert!(codec.plan_rows(&read, &[0]).is_none());
            }
        }
    }

    /// Code-only entries (`EntryColumns::Codes`) hold the file columns of
    /// their plane: an ex plane's rows carry no factor columns, and the sign
    /// and bounds planes keep raw bodies, whole and by rows.
    #[test]
    fn code_only_entries_round_trip() {
        const ROWS: usize = 90;
        for sign_bounds in [SignBounds::Lazy, SignBounds::Eager] {
            for plane in [0, 1, 2, SIGN_BOUNDS_PLANE] {
                let names = plane_entry_columns(plane, sign_bounds, EntryColumns::Codes);
                if names.is_empty() {
                    continue;
                }
                let codec = PlaneKey {
                    partition: 0,
                    plane,
                    sign_bounds,
                    entry_columns: EntryColumns::Codes,
                }
                .codec_for_key()
                .unwrap();
                let context = format!("plane {plane} {sign_bounds:?} {names:?}");
                let original = match plane {
                    1 | 2 => ex_plane_batch(plane, ROWS, 128, 0),
                    _ => whole_plane_batch(&names, ROWS),
                };
                let schema = original.schema();
                let columns: Vec<&str> = schema
                    .fields()
                    .iter()
                    .map(|field| field.name().as_str())
                    .collect();
                assert_eq!(columns, names, "{context}");
                let encoded = encode(&codec, &original);
                let body = &encoded[body_offset()..];
                match plane {
                    1 | 2 => {
                        assert_eq!(&body[..2], &[plane, 0], "{context}");
                        assert_eq!(body.len(), V3_HEADER_BYTES + ROWS * 128, "{context}");
                        let rows = [0, 7, 8, 89];
                        let CacheDecode::Hit(selected) = decode_rows(&codec, &encoded, &rows)
                        else {
                            panic!("{context}: row decode missed")
                        };
                        assert_eq!(
                            selected,
                            original
                                .take(&arrow_array::UInt32Array::from(rows.to_vec()))
                                .unwrap(),
                            "{context}"
                        );
                    }
                    _ => assert_eq!(body[1], RAW_BODY_KIND, "{context}"),
                }
                let CacheDecode::Hit(decoded) = decode(&codec, &encoded) else {
                    panic!("{context}: decode missed")
                };
                assert_eq!(decoded, original, "{context}");
            }
        }
    }

    type AnyValue = Arc<dyn std::any::Any + Send + Sync>;

    /// A v1 body: an IPC section for every plane.
    fn write_v1(any: &AnyValue, writer: &mut CacheEntryWriter<'_>) -> Result<()> {
        writer.write_ipc(&any.downcast_ref::<PlaneBatch>().unwrap().0)
    }

    /// A v2 body: a tagged IPC section for the sign and bounds planes, rows
    /// with both factors behind a 13-byte header for the ex planes.
    fn write_v2(any: &AnyValue, writer: &mut CacheEntryWriter<'_>) -> Result<()> {
        let batch = &any.downcast_ref::<PlaneBatch>().unwrap().0;
        let Some(plane) = ex_plane(batch) else {
            writer.write_u8(0)?;
            return writer.write_ipc(batch);
        };
        let names = plane_columns(plane, SignBounds::default());
        let codes = batch[names[0]].as_fixed_size_list();
        let width = codes.value_length() as usize;
        let values = codes.values().as_primitive::<UInt8Type>().values();
        let adds = batch[names[1]].as_primitive::<Float32Type>();
        let scales = batch[names[2]].as_primitive::<Float32Type>();
        writer.write_u8(plane)?;
        let writer = writer.raw_writer();
        writer.write_all(&(batch.num_rows() as u64).to_le_bytes())?;
        writer.write_all(&(width as u32).to_le_bytes())?;
        for row in 0..batch.num_rows() {
            writer.write_all(&values[row * width..(row + 1) * width])?;
            writer.write_all(&adds.value(row).to_le_bytes())?;
            writer.write_all(&scales.value(row).to_le_bytes())?;
        }
        Ok(())
    }

    fn write_only(_: &mut CacheEntryReader<'_>) -> Result<AnyValue> {
        Err(Error::invalid_input("legacy test codecs only serialize"))
    }

    /// A codec that writes the bodies of an earlier plane cache version.
    fn legacy_codec(version: u32) -> CacheCodec {
        let serialize = match version {
            1 => write_v1,
            2 => write_v2,
            _ => unreachable!("no legacy plane cache version {version}"),
        };
        CacheCodec::new(PlaneBatch::TYPE_ID, version, serialize, write_only)
    }

    #[test]
    fn earlier_versions_still_decode() {
        let codec = CacheCodec::from_impl::<PlaneBatch>();
        let sign = whole_plane_batch(plane_columns(0, SignBounds::Eager), 40);
        let bounds = whole_plane_batch(plane_columns(SIGN_BOUNDS_PLANE, SignBounds::Lazy), 40);
        let ex = ex_plane_batch(2, 300, 56, EX_FACTOR_COLUMNS);
        for version in [1, 2] {
            let legacy = legacy_codec(version);
            for original in [&sign, &bounds, &ex] {
                let encoded = encode(&legacy, original);
                let CacheDecode::Hit(decoded) = decode(&codec, &encoded) else {
                    panic!("v{version} decode failed")
                };
                assert_eq!(&decoded, original, "v{version}");
            }
            let encoded = encode(&legacy, &ex);
            let rows = [0, 5, 299];
            let selected = decode_rows(&codec, &encoded, &rows);
            if version == 1 {
                assert!(matches!(selected, CacheDecode::Miss(_)));
            } else {
                let CacheDecode::Hit(selected) = selected else {
                    panic!("v2 row decode failed")
                };
                assert_eq!(
                    selected,
                    ex.take(&arrow_array::UInt32Array::from(rows.to_vec()))
                        .unwrap()
                );
            }
        }
    }

    #[test]
    fn mismatched_factor_columns_are_errors() {
        let ex = ex_plane_batch(1, 10, 56, EX_FACTOR_COLUMNS);
        let names = plane_columns(1, SignBounds::default());
        let only_add = ex.project(&[0, 1]).unwrap();
        let only_scale = ex.project(&[0, 2]).unwrap();
        let (codes, adds, scales) = (
            ex[names[0]].clone(),
            ex[names[1]].clone(),
            ex[names[2]].clone(),
        );
        let float_codes: ArrayRef = Arc::new(
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.5f32; 10 * 4]), 4)
                .unwrap(),
        );
        let wrong_codes = RecordBatch::try_from_iter([
            (names[0], float_codes),
            (names[1], adds),
            (names[2], scales.clone()),
        ])
        .unwrap();
        let integer_adds: ArrayRef = Arc::new(UInt64Array::from((0..10u64).collect::<Vec<_>>()));
        let wrong_factor = RecordBatch::try_from_iter([
            (names[0], codes),
            (names[1], integer_adds),
            (names[2], scales),
        ])
        .unwrap();
        for batch in [only_add, only_scale, wrong_codes, wrong_factor] {
            let mut buf = Vec::new();
            let result = PlaneBatch(batch).serialize(&mut CacheEntryWriter::new(&mut buf));
            assert!(result.is_err());
        }

        // A header that claims one factor column misses rather than panics.
        let codec = CacheCodec::from_impl::<PlaneBatch>();
        let mut encoded = encode(&codec, &ex).to_vec();
        encoded[body_offset() + 1] = 1;
        let encoded = Bytes::from(encoded);
        assert!(matches!(
            decode(&codec, &encoded),
            CacheDecode::Miss(CacheMissReason::BodyError)
        ));
        assert!(matches!(
            decode_rows(&codec, &encoded, &[0]),
            CacheDecode::Miss(_)
        ));
    }

    #[test]
    fn newer_version_is_a_miss() {
        let codec = CacheCodec::from_impl::<PlaneBatch>();
        for original in [
            ex_plane_batch(1, 20, 128, EX_FACTOR_COLUMNS),
            whole_plane_batch(plane_columns(0, SignBounds::Lazy), 20),
        ] {
            let mut encoded = encode(&codec, &original).to_vec();
            let version_at = body_offset() - 4;
            assert_eq!(
                u32::from_le_bytes(encoded[version_at..body_offset()].try_into().unwrap()),
                PlaneBatch::CURRENT_VERSION
            );
            encoded[version_at..body_offset()]
                .copy_from_slice(&(PlaneBatch::CURRENT_VERSION + 1).to_le_bytes());
            let encoded = Bytes::from(encoded);
            assert!(matches!(
                decode(&codec, &encoded),
                CacheDecode::Miss(CacheMissReason::VersionTooNew)
            ));
            assert!(matches!(
                decode_rows(&codec, &encoded, &[0]),
                CacheDecode::Miss(_)
            ));
            let read = |range: Range<usize>| -> Result<Bytes> { Ok(encoded.slice(range)) };
            assert!(codec.plan_rows(&read, &[0]).is_none());
        }
    }
}
