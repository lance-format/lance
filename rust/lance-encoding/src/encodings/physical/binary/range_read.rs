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

impl BinaryMiniBlockDecompressor {
    pub(crate) async fn read_value_slices(
        &self,
        buffers: &[Range<u64>],
        num_values: u64,
        row: u64,
        slices: &[Range<u64>],
        io: &Arc<dyn EncodingsIo>,
        priority: u64,
    ) -> Result<Option<Vec<Bytes>>> {
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
        let mut requests = Vec::with_capacity(slices.len());
        let mut result = vec![Bytes::new(); slices.len()];
        for (index, slice) in slices.iter().enumerate() {
            if slice.start > slice.end || slice.end > value_len {
                return Err(Error::invalid_input(format!(
                    "binary byte range {slice:?} exceeds {value_len} bytes"
                )));
            }
            if !slice.is_empty() {
                let start = payload
                    .start
                    .checked_add(value_start)
                    .and_then(|start| start.checked_add(slice.start))
                    .ok_or_else(|| corrupt("binary file offset overflow"))?;
                let end = start
                    .checked_add(slice.end - slice.start)
                    .ok_or_else(|| corrupt("binary file range overflow"))?;
                requests.push((index, start..end));
            }
        }
        requests.sort_unstable_by_key(|(_, range)| (range.start, range.end));
        if !requests.is_empty() {
            let read = io
                .submit_request(
                    requests.iter().map(|(_, range)| range.clone()).collect(),
                    priority,
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
        Ok(Some(result))
    }
}
