// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Retain mini-block headers and structural streams without reading value buffers.

use super::super::MINIBLOCK_ALIGNMENT;
use crate::EncodingsIo;
use bytes::Bytes;
use futures::{FutureExt, StreamExt, TryStreamExt, future::BoxFuture, stream};
use lance_core::{Error, Result};
use std::{ops::Range, sync::Arc};

pub(super) fn read_chunks(
    io: Arc<dyn EncodingsIo>,
    ranges: Vec<Range<u64>>,
    num_buffers: u64,
    has_rep: bool,
    has_def: bool,
    wide: bool,
    priority: u64,
) -> BoxFuture<'static, Result<Vec<Bytes>>> {
    let io = io.with_coalesce_gap(0).unwrap_or(io);
    async move {
        let sizes_start = 2 + 2 * (usize::from(has_rep) + usize::from(has_def));
        let header_size = usize::try_from(num_buffers)
            .ok()
            .and_then(|n| n.checked_mul(if wide { 4 } else { 2 }))
            .and_then(|n| n.checked_add(sizes_start))
            .ok_or_else(|| {
                Error::corrupt_file_named("miniblock_structure", "Header size overflow")
            })?;
        stream::iter(ranges)
            .map(|range| {
                let io = io.clone();
                async move {
                    if range.start > range.end || header_size as u64 > range.end - range.start {
                        return Err(Error::corrupt_file_named(
                            "miniblock_structure",
                            "Truncated header",
                        ));
                    }
                    let mut header = io
                        .submit_single(range.start..range.start + header_size as u64, priority)
                        .await?
                        .to_vec();
                    if header.len() != header_size {
                        return Err(Error::corrupt_file_named(
                            "miniblock_structure",
                            "Short header read",
                        ));
                    }
                    let mut structural_end = header_size.next_multiple_of(MINIBLOCK_ALIGNMENT);
                    let mut cursor = 2;
                    for present in [has_rep, has_def] {
                        if present {
                            let size =
                                u16::from_le_bytes([header[cursor], header[cursor + 1]]) as usize;
                            structural_end = structural_end
                                .checked_add(size)
                                .and_then(|n| n.checked_next_multiple_of(MINIBLOCK_ALIGNMENT))
                                .ok_or_else(|| {
                                    Error::corrupt_file_named(
                                        "miniblock_structure",
                                        "Structural size overflow",
                                    )
                                })?;
                            cursor += 2;
                        }
                    }
                    let mut stored_end = structural_end;
                    for size in header[sizes_start..].chunks_exact(if wide { 4 } else { 2 }) {
                        let size = if wide {
                            u32::from_le_bytes(size.try_into().map_err(|_| {
                                Error::corrupt_file_named(
                                    "miniblock_structure",
                                    "Invalid size word",
                                )
                            })?) as usize
                        } else {
                            u16::from_le_bytes([size[0], size[1]]) as usize
                        };
                        stored_end = stored_end
                            .checked_add(size)
                            .and_then(|n| n.checked_next_multiple_of(MINIBLOCK_ALIGNMENT))
                            .ok_or_else(|| {
                                Error::corrupt_file_named(
                                    "miniblock_structure",
                                    "Value size overflow",
                                )
                            })?;
                    }
                    if stored_end as u64 > range.end - range.start {
                        return Err(Error::corrupt_file_named(
                            "miniblock_structure",
                            "Buffers exceed chunk bounds",
                        ));
                    }
                    // The ordinary structural decoder still parses the header. Zero
                    // sizes describe the deliberately omitted, unobservable values.
                    header[sizes_start..].fill(0);
                    if structural_end > header_size {
                        header.extend_from_slice(
                            &io.submit_single(
                                range.start + header_size as u64
                                    ..range.start + structural_end as u64,
                                priority,
                            )
                            .await?,
                        );
                    }
                    if header.len() != structural_end {
                        return Err(Error::corrupt_file_named(
                            "miniblock_structure",
                            "Short structural buffer read",
                        ));
                    }
                    Ok(Bytes::from(header))
                }
            })
            .buffered(super::super::structure::METADATA_READ_CONCURRENCY)
            .try_collect()
            .await
    }
    .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encodings::logical::primitive::structure::tests::GuardIo;
    use std::sync::Mutex;

    #[rstest::rstest]
    #[tokio::test]
    async fn reads_levels_without_value_buffers(#[values(false, true)] wide: bool) {
        let mut bytes = Vec::new();
        bytes.extend(4_u16.to_le_bytes());
        bytes.extend(8_u16.to_le_bytes());
        bytes.extend(8_u16.to_le_bytes());
        if wide {
            bytes.extend(16_u32.to_le_bytes());
        } else {
            bytes.extend(16_u16.to_le_bytes());
        }
        bytes.resize(bytes.len().next_multiple_of(MINIBLOCK_ALIGNMENT), 0);
        bytes.extend([1_u16, 0, 1, 0].into_iter().flat_map(u16::to_le_bytes));
        bytes.extend([0_u16, 1, 0, 0].into_iter().flat_map(u16::to_le_bytes));
        let payload_start = bytes.len();
        bytes.extend([0xA5; 16]);
        let end = bytes.len() as u64;
        let io = Arc::new(GuardIo {
            bytes: bytes.into(),
            payloads: std::iter::once(payload_start as u64..end).collect(),
            reads: Mutex::default(),
        });
        let chunks = read_chunks(
            io.clone(),
            std::iter::once(0..end).collect(),
            1,
            true,
            true,
            wide,
            0,
        )
        .await
        .unwrap();
        assert_eq!(chunks[0].len(), payload_start);
        assert_eq!(
            &chunks[0][6..if wide { 10 } else { 8 }],
            if wide { &[0_u8; 4][..] } else { &[0_u8; 2][..] }
        );
        assert_eq!(
            io.reads.lock().unwrap().last().unwrap().end,
            payload_start as u64
        );
    }
}
