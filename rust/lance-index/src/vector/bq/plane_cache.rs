// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Row-addressable persistent ex planes. Sign codes retain their transposed
//! IPC body, and so do the bounds columns, which scans read whole.
use std::ops::Range;
use std::sync::Arc;

use arrow_array::{Array, Float32Array, RecordBatch, UInt8Array, cast::AsArray, types::UInt8Type};
use arrow_schema::{DataType, Field, Schema};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::cache::{CacheCodecImpl, CacheEntryReader, CacheEntryWriter, CacheRangeReader};
use lance_core::{Error, Result};

use super::layered::{PlaneBatch, SignBounds, plane_columns};
use super::storage::{RABIT_BLOCKED_EX_CODE_COLUMN, RABIT_BLOCKED_EX_CODE_LO_COLUMN};

const HEADER_BYTES: usize = 13;
const COALESCE_GAP_BYTES: usize = 4096;

struct Header {
    plane: u8,
    rows: usize,
    width: usize,
}
impl Header {
    fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_BYTES || !matches!(bytes[0], 1 | 2) {
            return Err(Error::invalid_input("invalid ex-plane cache header"));
        }
        let rows = usize::try_from(u64::from_le_bytes(bytes[1..9].try_into().unwrap()))
            .map_err(|_| Error::invalid_input("plane row count overflow"))?;
        let width = u32::from_le_bytes(bytes[9..13].try_into().unwrap()) as usize;
        if width == 0 || width > i32::MAX as usize {
            return Err(Error::invalid_input("invalid cached code width"));
        }
        Ok(Self {
            plane: bytes[0],
            rows,
            width,
        })
    }
    fn stride(&self) -> usize {
        self.width + 8
    }
    fn batch(&self, codes: Vec<u8>, adds: Vec<f32>, scales: Vec<f32>) -> Result<PlaneBatch> {
        // Ex planes hold the same columns under either bounds placement.
        let names = plane_columns(self.plane, SignBounds::default());
        let codes = arrow_array::FixedSizeListArray::try_new_from_values(
            UInt8Array::from(codes),
            self.width as i32,
        )?;
        let fields = vec![
            Field::new(names[0], codes.data_type().clone(), true),
            Field::new(names[1], DataType::Float32, true),
            Field::new(names[2], DataType::Float32, true),
        ];
        Ok(PlaneBatch(RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            vec![
                Arc::new(codes),
                Arc::new(Float32Array::from(adds)),
                Arc::new(Float32Array::from(scales)),
            ],
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
        let mut codes = Vec::with_capacity(self.rows * self.width);
        let mut adds = Vec::with_capacity(self.rows);
        let mut scales = Vec::with_capacity(self.rows);
        for row in bytes.chunks_exact(self.stride()) {
            self.append(row, &mut codes, &mut adds, &mut scales);
        }
        self.batch(codes, adds, scales)
    }
    fn append(&self, row: &[u8], codes: &mut Vec<u8>, adds: &mut Vec<f32>, scales: &mut Vec<f32>) {
        codes.extend_from_slice(&row[..self.width]);
        adds.push(f32::from_le_bytes(
            row[self.width..self.width + 4].try_into().unwrap(),
        ));
        scales.push(f32::from_le_bytes(
            row[self.width + 4..self.width + 8].try_into().unwrap(),
        ));
    }
}

impl CacheCodecImpl for PlaneBatch {
    const TYPE_ID: &'static str = "lance.vector.rq.plane";
    const CURRENT_VERSION: u32 = 2;
    const SUPPORTS_ROW_SELECTION: bool = true;

    fn serialize(&self, writer: &mut CacheEntryWriter<'_>) -> Result<()> {
        let plane = if self
            .0
            .column_by_name(RABIT_BLOCKED_EX_CODE_LO_COLUMN)
            .is_some()
        {
            2
        } else if self
            .0
            .column_by_name(RABIT_BLOCKED_EX_CODE_COLUMN)
            .is_some()
        {
            1
        } else {
            0
        };
        writer.write_u8(plane)?;
        if plane == 0 {
            return writer.write_ipc(&self.0);
        }
        let names = plane_columns(plane, SignBounds::default());
        let codes = self.0[names[0]].as_fixed_size_list();
        let adds = self.0[names[1]].as_primitive::<arrow_array::types::Float32Type>();
        let scales = self.0[names[2]].as_primitive::<arrow_array::types::Float32Type>();
        if codes.null_count() != 0 || adds.null_count() != 0 || scales.null_count() != 0 {
            return Err(Error::invalid_input("null cached ex-plane row"));
        }
        let width = codes.value_length() as usize;
        let values = codes.values().as_primitive::<UInt8Type>().values();
        let writer = writer.raw_writer();
        writer.write_all(&(self.0.num_rows() as u64).to_le_bytes())?;
        writer.write_all(&(width as u32).to_le_bytes())?;
        for row in 0..self.0.num_rows() {
            writer.write_all(&values[row * width..(row + 1) * width])?;
            writer.write_all(&adds.value(row).to_le_bytes())?;
            writer.write_all(&scales.value(row).to_le_bytes())?;
        }
        Ok(())
    }
    fn deserialize(reader: &mut CacheEntryReader<'_>) -> Result<Self> {
        if reader.version() == 1 {
            return Ok(Self(reader.read_ipc()?));
        }
        if reader.version() != Self::CURRENT_VERSION {
            return Err(Error::invalid_input("unsupported plane cache version"));
        }
        let body = reader.body();
        if body.first() == Some(&0) {
            reader.read_u8()?;
            return Ok(Self(reader.read_ipc()?));
        }
        let header = Header::parse(&body)?;
        header.decode(&body[HEADER_BYTES..])
    }
    fn deserialize_rows(
        reader: &dyn CacheRangeReader,
        offset: usize,
        version: u32,
        rows: &[u32],
    ) -> Result<Self> {
        let header = read_row_header(reader, offset, version)?;
        let width = header.width;
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
                    &bytes[local..local + width + 8],
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
    if version != PlaneBatch::CURRENT_VERSION {
        return Err(Error::invalid_input(
            "plane cache version has no row directory",
        ));
    }
    Header::parse(&reader.read_range(offset..offset + HEADER_BYTES)?)
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
        .checked_add(HEADER_BYTES)
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
    use crate::vector::bq::layered::{PlaneKey, SIGN_BOUNDS_PLANE};
    use bytes::Bytes;
    use lance_arrow::RecordBatchExt;
    use lance_core::cache::{CacheCodec, CacheDecode, CacheKey};
    use std::cell::Cell;

    /// Bounds plane entries keep an IPC body, as sign planes do: they round
    /// trip whole and have no row directory, so a row read misses.
    #[test]
    fn bounds_plane_round_trips_through_ipc() {
        const ROWS: usize = 70;
        let names = plane_columns(SIGN_BOUNDS_PLANE, SignBounds::Lazy);
        let bounds = |offset: f32| {
            let values: Vec<f32> = (0..ROWS * 3).map(|v| v as f32 + offset).collect();
            arrow_array::FixedSizeListArray::try_new_from_values(Float32Array::from(values), 3)
                .unwrap()
        };
        let (high, full) = (bounds(0.5), bounds(0.25));
        let original = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new(names[0], high.data_type().clone(), true),
                Field::new(names[1], full.data_type().clone(), true),
            ])),
            vec![Arc::new(high), Arc::new(full)],
        )
        .unwrap();
        let codec = PlaneKey {
            partition: 0,
            plane: SIGN_BOUNDS_PLANE,
            sign_bounds: SignBounds::Lazy,
        }
        .codec_for_key()
        .unwrap();
        let mut encoded = Vec::new();
        codec
            .serialize(
                &(Arc::new(PlaneBatch(original.clone())) as Arc<dyn std::any::Any + Send + Sync>),
                &mut encoded,
            )
            .unwrap();
        let encoded = Bytes::from(encoded);
        let CacheDecode::Hit(decoded) = codec.deserialize(&encoded) else {
            panic!("bounds plane decode failed")
        };
        assert_eq!(decoded.downcast::<PlaneBatch>().unwrap().0, original);
        let read = |range: Range<usize>| -> Result<Bytes> { Ok(encoded.slice(range)) };
        assert!(matches!(
            codec.deserialize_rows(&read, &[0, 5]),
            CacheDecode::Miss(_)
        ));
        assert!(codec.plan_rows(&read, &[0, 5]).is_none());
    }

    #[test]
    fn sparse_plane_cache_reads_match_full_decode() {
        for plane in [1, 2] {
            let header = Header {
                plane,
                rows: 4096,
                width: 128,
            };
            let original = header
                .batch(
                    (0..4096 * 128).map(|v| (v % 251) as u8).collect(),
                    (0..4096).map(|v| v as f32).collect(),
                    vec![2.; 4096],
                )
                .unwrap();
            let codec = CacheCodec::from_impl::<PlaneBatch>();
            let mut encoded = Vec::new();
            codec
                .serialize(
                    &(Arc::new(original) as Arc<dyn std::any::Any + Send + Sync>),
                    &mut encoded,
                )
                .unwrap();
            let encoded = Bytes::from(encoded);
            let CacheDecode::Hit(full) = codec.deserialize(&encoded) else {
                panic!("full decode failed")
            };
            let full = full.downcast::<PlaneBatch>().unwrap();
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
                full.0
                    .take(&arrow_array::UInt32Array::from(rows.to_vec()))
                    .unwrap()
            );
            assert!(bytes_read.get() < encoded.len() / 10);
            assert!(matches!(
                codec.deserialize_rows(&read, &[4096]),
                CacheDecode::Miss(_)
            ));
            assert!(matches!(
                codec.deserialize_rows(&read, &[1, 1]),
                CacheDecode::Miss(_)
            ));
        }
    }

    #[test]
    fn row_plan_matches_ranges_read_by_decode() {
        const ROWS: usize = 4096;
        const WIDTH: usize = 56;
        let header = Header {
            plane: 1,
            rows: ROWS,
            width: WIDTH,
        };
        let original = header
            .batch(
                (0..ROWS * WIDTH).map(|v| (v % 253) as u8).collect(),
                (0..ROWS).map(|v| v as f32).collect(),
                vec![0.5; ROWS],
            )
            .unwrap();
        let codec = CacheCodec::from_impl::<PlaneBatch>();
        let mut encoded = Vec::new();
        codec
            .serialize(
                &(Arc::new(original) as Arc<dyn std::any::Any + Send + Sync>),
                &mut encoded,
            )
            .unwrap();
        let encoded = Bytes::from(encoded);
        let requested = std::cell::RefCell::new(Vec::new());
        let read = |range: Range<usize>| -> Result<Bytes> {
            requested.borrow_mut().push(range.clone());
            Ok(encoded.slice(range))
        };
        let stride = WIDTH + 8;
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
            assert_eq!(plan.len(), expected_runs);
        }
        assert!(codec.plan_rows(&read, &[5, 5]).is_none());
        assert!(codec.plan_rows(&read, &[ROWS as u32]).is_none());
    }
}
