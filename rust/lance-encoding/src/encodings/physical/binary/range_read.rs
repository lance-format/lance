// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{ops::Range, sync::Arc};

use bytes::Bytes;
use lance_core::{Error, Result};

use super::{BinaryMiniBlockDecompressor, BinaryMiniBlockLayout};
use crate::{
    EncodingsIo, buffer::LanceBuffer, compression::MiniBlockDecompressor, data::DataBlock,
};

// This is an optional fast path. Wider offset tables keep the ordinary reader.
const MAX_OFFSET_BYTES: u64 = 64 * 1024;

/// An immutable binary value's validated file range, without an I/O scope.
///
/// A location may be cached with metadata for the same immutable file. It does
/// not retain a reader, request recorder, payload buffer or I/O priority.
#[derive(Clone, Debug)]
pub struct BinaryValueLocation {
    range: Range<u64>,
}

impl BinaryValueLocation {
    /// Bind the location to the same file in the current request's I/O scope.
    /// The supplied priority applies only to reads through the returned handle.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use lance_encoding::{EncodingsIo, encodings::physical::binary::{BinaryValueLocation, PreparedBinaryValue}};
    /// fn bind(location: &BinaryValueLocation, io: Arc<dyn EncodingsIo>) -> PreparedBinaryValue {
    ///     location.with_io(io, 0)
    /// }
    /// ```
    pub fn with_io(&self, io: Arc<dyn EncodingsIo>, priority: u64) -> PreparedBinaryValue {
        PreparedBinaryValue {
            range: self.range.clone(),
            io,
            priority,
        }
    }
}

/// A validated binary value located without loading its payload.
///
/// Preparation retains only its file range and I/O service. Repeated slices
/// reuse the location instead of re-reading chunk metadata and offsets. Bind a
/// handle to its request's I/O scope; it can contain a request-specific recorder.
#[derive(Clone)]
pub struct PreparedBinaryValue {
    range: Range<u64>,
    io: Arc<dyn EncodingsIo>,
    priority: u64,
}

impl std::fmt::Debug for PreparedBinaryValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedBinaryValue")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

impl PreparedBinaryValue {
    /// Length of the binary value, available without reading its payload.
    pub fn len(&self) -> u64 {
        self.range.end - self.range.start
    }

    /// Whether this value has no payload bytes.
    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }

    /// Copy only the immutable file range, releasing this handle's I/O scope
    /// independently of the returned location's lifetime.
    ///
    /// ```
    /// # use lance_encoding::encodings::physical::binary::{BinaryValueLocation, PreparedBinaryValue};
    /// fn detach(value: PreparedBinaryValue) -> BinaryValueLocation {
    ///     let location = value.location();
    ///     drop(value);
    ///     location
    /// }
    /// ```
    pub fn location(&self) -> BinaryValueLocation {
        BinaryValueLocation {
            range: self.range.clone(),
        }
    }

    /// Read value-relative slices, preserving order, overlap and empty slices.
    ///
    /// Returned buffers can share coalesced I/O backing; visible byte lengths
    /// are not retained-allocation bounds. Every requested slice is validated
    /// before payload I/O, and short reads return a typed error.
    ///
    /// ```
    /// # use lance_encoding::encodings::physical::binary::PreparedBinaryValue;
    /// # async fn example(value: &PreparedBinaryValue) -> lance_core::Result<()> {
    /// let tail = value.read_slices(&[value.len().saturating_sub(8)..value.len()]).await?;
    /// assert!(tail[0].len() <= 8);
    /// # Ok(()) }
    /// ```
    pub async fn read_slices(&self, slices: &[Range<u64>]) -> Result<Vec<Bytes>> {
        let corrupt = |message: &str| Error::corrupt_file_named("binary mini-block", message);
        let value_len = self.len();
        let mut requests = Vec::with_capacity(slices.len());
        let mut result = vec![Bytes::new(); slices.len()];
        for (index, slice) in slices.iter().enumerate() {
            if slice.start > slice.end || slice.end > value_len {
                return Err(Error::invalid_input(format!(
                    "binary byte range {slice:?} exceeds {value_len} bytes"
                )));
            }
            if !slice.is_empty() {
                let start = self
                    .range
                    .start
                    .checked_add(slice.start)
                    .ok_or_else(|| corrupt("binary file offset overflow"))?;
                let end = start
                    .checked_add(slice.end - slice.start)
                    .ok_or_else(|| corrupt("binary file range overflow"))?;
                requests.push((index, start..end));
            }
        }
        requests.sort_unstable_by_key(|(_, range)| (range.start, range.end));
        if !requests.is_empty() {
            let read = self
                .io
                .submit_request(
                    requests.iter().map(|(_, range)| range.clone()).collect(),
                    self.priority,
                )
                .await?;
            if read.len() != requests.len() {
                return Err(corrupt("binary range result count mismatch"));
            }
            for ((index, range), bytes) in requests.into_iter().zip(read) {
                if bytes.len() as u64 != range.end - range.start {
                    return Err(corrupt("truncated binary value range"));
                }
                result[index] = bytes;
            }
        }
        Ok(result)
    }
}

impl BinaryMiniBlockDecompressor {
    pub(crate) async fn prepare_value(
        &self,
        buffers: &[Range<u64>],
        num_values: u64,
        row: u64,
        io: &Arc<dyn EncodingsIo>,
        priority: u64,
    ) -> Result<Option<PreparedBinaryValue>> {
        let corrupt = |message: &str| Error::corrupt_file_named("binary mini-block", message);
        if buffers.len() != self.num_buffers() || row >= num_values {
            return Err(corrupt("invalid buffer count or selected value index"));
        }
        let count = num_values
            .checked_add(1)
            .ok_or_else(|| corrupt("offset count overflow"))?;
        let width = match &self.layout {
            BinaryMiniBlockLayout::Legacy { bits_per_offset }
            | BinaryMiniBlockLayout::Generic {
                bits_per_offset, ..
            } => *bits_per_offset,
        };
        let offset_bytes = count
            .checked_mul(u64::from(width / 8))
            .ok_or_else(|| corrupt("offset byte count overflow"))?;
        if offset_bytes > MAX_OFFSET_BYTES {
            return Ok(None);
        }
        let payload = buffers
            .last()
            .ok_or_else(|| corrupt("missing value buffer"))?;
        if buffers.iter().any(|range| range.start > range.end) {
            return Err(corrupt("reversed binary buffer"));
        }
        let payload_len = payload
            .end
            .checked_sub(payload.start)
            .ok_or_else(|| corrupt("reversed value buffer"))?;
        let (offset_data, embedded) = match &self.layout {
            BinaryMiniBlockLayout::Legacy { .. } => {
                if offset_bytes > payload_len {
                    return Err(corrupt("offset table exceeds its value buffer"));
                }
                let mut read = io
                    .submit_request(vec![payload.start..payload.start + offset_bytes], priority)
                    .await?;
                if read.len() != 1 || read[0].len() as u64 != offset_bytes {
                    return Err(corrupt("truncated binary offset table"));
                }
                (
                    LanceBuffer::from_bytes(
                        read.pop().ok_or_else(|| corrupt("missing offset buffer"))?,
                        1,
                    ),
                    true,
                )
            }
            BinaryMiniBlockLayout::Generic {
                offsets,
                offsets_have_payload,
                ..
            } => {
                let encoded = if *offsets_have_payload {
                    let range = &buffers[0];
                    let len = range
                        .end
                        .checked_sub(range.start)
                        .ok_or_else(|| corrupt("reversed offset buffer"))?;
                    if len > MAX_OFFSET_BYTES {
                        return Ok(None);
                    }
                    let mut read = io.submit_request(vec![range.clone()], priority).await?;
                    if read.len() != 1 || read[0].len() as u64 != len {
                        return Err(corrupt("truncated encoded offset buffer"));
                    }
                    Some(LanceBuffer::from_bytes(
                        read.pop().ok_or_else(|| corrupt("missing offset buffer"))?,
                        1,
                    ))
                } else {
                    None
                };
                let DataBlock::FixedWidth(decoded) = offsets.decompress(encoded, count)? else {
                    return Err(corrupt("offset codec did not produce fixed-width offsets"));
                };
                if decoded.bits_per_value != u64::from(width)
                    || decoded.num_values != count
                    || decoded.data.len() as u64 != offset_bytes
                {
                    return Err(corrupt("offset codec returned an invalid layout"));
                }
                (decoded.data, false)
            }
        };
        let offsets: Vec<u64> = if width == 32 {
            offset_data
                .borrow_to_typed_slice::<u32>()
                .iter()
                .map(|&v| u64::from(v))
                .collect()
        } else {
            offset_data
                .borrow_to_typed_slice::<u64>()
                .iter()
                .copied()
                .collect()
        };
        if (embedded && offsets[0] < offset_bytes)
            || (!embedded && offsets[0] != 0)
            || offsets.windows(2).any(|pair| pair[0] > pair[1])
            || offsets
                .last()
                .copied()
                .ok_or_else(|| corrupt("missing terminal offset"))?
                > payload_len
            || (!embedded
                && offsets
                    .last()
                    .copied()
                    .ok_or_else(|| corrupt("missing terminal offset"))?
                    != payload_len)
        {
            return Err(corrupt("invalid binary offsets"));
        }
        let row = row as usize;
        let value_start = offsets[row];
        let value_len = offsets[row + 1] - value_start;
        if (width == 32 && value_len > i32::MAX as u64) || value_len > i64::MAX as u64 {
            return Err(corrupt("binary value exceeds its signed offset limit"));
        }
        let start = payload
            .start
            .checked_add(value_start)
            .ok_or_else(|| corrupt("binary file offset overflow"))?;
        let end = start
            .checked_add(value_len)
            .ok_or_else(|| corrupt("binary file range overflow"))?;
        Ok(Some(PreparedBinaryValue {
            range: start..end,
            io: io.clone(),
            priority,
        }))
    }
}
