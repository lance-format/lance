// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Persistent native plane entries. The sign plane keeps its transposed
//! codes in a raw fixed-width body. Ex entries are addressable by row:
//!
//! ```text
//! sign : [plane = 0: u8][raw fixed-width body]
//! ex   : [plane = 1: u8][factor_columns: u8][rows: u64 LE][width: u32 LE][rows]
//! row  : [codes: width bytes][add: f32 LE][scale: f32 LE] (all columns)
//!        [codes: width bytes]                          (code-only)
//! ```
//! Index files are unchanged; these bodies belong only to the persistent cache.

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

use super::planes::{PlaneBatch, plane_columns};
use super::raw_body::{read_raw_batch, write_raw_batch};
use super::storage::RABIT_BLOCKED_EX_CODE_COLUMN;

const BODY_VERSION: u32 = 1;
const EX_HEADER_BYTES: usize = 14;
const EX_FACTOR_COLUMNS: u8 = 2;
const FACTOR_BYTES: usize = size_of::<f32>();
const COALESCE_GAP_BYTES: usize = 4096;

fn validate_version(version: u32) -> Result<()> {
    if version != BODY_VERSION {
        return Err(Error::invalid_input(format!(
            "unsupported native plane cache version {version}"
        )));
    }
    Ok(())
}

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
    fn parse(bytes: &[u8], version: u32) -> Result<Self> {
        let invalid = || Error::invalid_input("invalid ex-plane cache header");
        validate_version(version)?;
        let len = EX_HEADER_BYTES;
        let header = bytes.get(..len).ok_or_else(invalid)?;
        let (factor_columns, fields) = (header[1], &header[2..]);
        let plane = header[0];
        let supported_plane = plane == 1;
        if !supported_plane || !matches!(factor_columns, 0 | EX_FACTOR_COLUMNS) {
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
        let names = plane_columns(self.plane);
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
        let names = plane_columns(plane);
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
    const TYPE_ID: &'static str = "lance.vector.rabit.plane";
    const CURRENT_VERSION: u32 = BODY_VERSION;
    const SUPPORTS_ROW_SELECTION: bool = true;

    fn serialize(&self, writer: &mut CacheEntryWriter<'_>) -> Result<()> {
        if self
            .0
            .column_by_name(RABIT_BLOCKED_EX_CODE_COLUMN)
            .is_some()
        {
            self.serialize_ex_plane(1, writer)
        } else {
            writer.write_u8(0)?;
            write_raw_batch(writer, &self.0)
        }
    }

    fn deserialize(reader: &mut CacheEntryReader<'_>) -> Result<Self> {
        validate_version(reader.version())?;
        if reader.body().first().copied() == Some(0) {
            reader.read_u8()?;
            Ok(Self(read_raw_batch(reader)?))
        } else {
            let body = reader.body();
            let header = Header::parse(&body, reader.version())?;
            header.decode(&body[header.len..])
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
    validate_version(version)?;
    let end = offset
        .checked_add(EX_HEADER_BYTES)
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
    use crate::vector::bq::planes::{EntryColumns, PlaneKey, plane_entry_columns};
    use crate::vector::bq::raw_body::RAW_BODY_KIND;
    use crate::vector::bq::storage::RABIT_CODE_COLUMN;
    use crate::vector::bq::transform::{EX_ADD_FACTORS_COLUMN, EX_SCALE_FACTORS_COLUMN};
    use arrow_array::{ArrayRef, UInt64Array};
    use bytes::Bytes;
    use lance_arrow::RecordBatchExt;
    use lance_core::cache::{CacheCodec, CacheDecode, CacheKey, CacheMissReason};
    use std::cell::{Cell, RefCell};
    const WIDTHS: [usize; 4] = [56, 128, 512, 1024];
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
            len: EX_HEADER_BYTES,
        }
        .batch(
            (0..rows * width).map(|v| (v % 251) as u8).collect(),
            (0..rows).map(|v| v as f32).collect(),
            (0..rows).map(|v| v as f32 * 0.5 + 2.0).collect(),
        )
        .unwrap()
        .0
    }
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
        {
            let plane = 1;
            for width in WIDTHS {
                for factor_columns in [EX_FACTOR_COLUMNS, 0] {
                    let context = format!("plane {plane} width {width} factors {factor_columns}");
                    let original = ex_plane_batch(plane, ROWS, width, factor_columns);
                    let encoded = encode(&codec, &original);
                    let body = &encoded[body_offset()..];
                    assert_eq!(&body[..2], &[plane, factor_columns], "{context}");
                    let stride = width + FACTOR_BYTES * factor_columns as usize;
                    assert_eq!(body.len(), EX_HEADER_BYTES + ROWS * stride, "{context}");
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
        {
            let plane = 1;
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
    #[test]
    fn mismatched_factor_columns_are_errors() {
        let ex = ex_plane_batch(1, 10, 56, EX_FACTOR_COLUMNS);
        let names = plane_columns(1);
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
            whole_plane_batch(plane_columns(0), 20),
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
    #[rstest::rstest]
    #[case::codes(false)]
    #[case::all(true)]
    fn native_ex_plane_round_trip_preserves_factors(#[case] with_factors: bool) {
        let columns = ex_plane_batch(1, 40, 56, if with_factors { 2 } else { 0 });
        let mut native_columns = vec![(RABIT_BLOCKED_EX_CODE_COLUMN, columns.column(0).clone())];
        if with_factors {
            native_columns.push((EX_ADD_FACTORS_COLUMN, columns.column(1).clone()));
            native_columns.push((EX_SCALE_FACTORS_COLUMN, columns.column(2).clone()));
        }
        let fields: Vec<_> = native_columns
            .iter()
            .map(|(name, values)| Field::new(*name, values.data_type().clone(), true))
            .collect();
        let original = RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            native_columns
                .into_iter()
                .map(|(_, values)| values)
                .collect(),
        )
        .unwrap();
        let codec = CacheCodec::from_impl::<PlaneBatch>();
        let encoded = encode(&codec, &original);
        let CacheDecode::Hit(restored) = decode(&codec, &encoded) else {
            panic!("native whole-plane cache entry did not decode")
        };
        assert_eq!(restored, original);
        for rows in [vec![], vec![0], vec![0, 17, 39], (0..40).collect()] {
            let CacheDecode::Hit(selected) = decode_rows(&codec, &encoded, &rows) else {
                panic!("native sparse cache entry did not decode")
            };
            assert_eq!(
                selected,
                original
                    .take(&arrow_array::UInt32Array::from(rows))
                    .unwrap()
            );
        }
    }

    #[rstest::rstest]
    fn sign_planes_round_trip_through_raw_bodies(
        #[values(0, 1, 40)] rows: usize,
        #[values(EntryColumns::All, EntryColumns::Codes)] columns: EntryColumns,
    ) {
        let names = plane_entry_columns(0, columns);
        let original = whole_plane_batch(&names, rows);
        let key = PlaneKey {
            partition: 0,
            plane: 0,
            entry_columns: columns,
        };
        let codec = key.codec_for_key().unwrap();
        let encoded = encode(&codec, &original);
        assert_eq!(
            &encoded[body_offset()..body_offset() + 2],
            &[0, RAW_BODY_KIND]
        );
        let CacheDecode::Hit(decoded) = decode(&codec, &encoded) else {
            panic!("sign plane did not decode")
        };
        assert_eq!(decoded, original);
        assert!(matches!(
            decode_rows(&codec, &encoded, &[0]),
            CacheDecode::Miss(_)
        ));
    }

    #[test]
    fn key_compositions_are_disjoint() {
        let keys: Vec<_> = [0, 1]
            .into_iter()
            .flat_map(|plane| {
                [EntryColumns::All, EntryColumns::Codes]
                    .into_iter()
                    .map(move |entry_columns| PlaneKey {
                        partition: 7,
                        plane,
                        entry_columns,
                    })
            })
            .map(|key| {
                let mut builder = lance_core::cache::KeyBuilder::new(
                    lance_core::cache::CacheNamespace::root(),
                    PlaneBatch::TYPE_ID,
                    PlaneKey::schema(),
                );
                key.write_key(&mut builder);
                builder.finish()
            })
            .collect();
        for (i, key) in keys.iter().enumerate() {
            assert!(!keys[i + 1..].contains(key));
        }
    }
}
