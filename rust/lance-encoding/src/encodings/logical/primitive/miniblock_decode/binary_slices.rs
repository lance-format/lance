// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{borrow::Cow, sync::Arc};

use bytes::Bytes;
use lance_core::{Error, Result};

use super::{ChunkInstructions, MINIBLOCK_ALIGNMENT, MiniBlockScheduler, StructuralPageScheduler};
use crate::{
    EncodingsIo,
    compression::{DefaultDecompressionStrategy, MiniBlockDecompressor},
    decoder::{PageEncoding, PageInfo},
    encodings::physical::binary::{BinaryMiniBlockDecompressor, PreparedBinaryValue},
    format::metadata::{compressive_encoding::Compression, page_layout::Layout},
};

// Match the bounded offset-table fast path below. Large metadata keeps the
// ordinary reader instead of allocating an unbounded auxiliary index per call.
const MAX_METADATA_BYTES: u64 = 64 * 1024;

impl MiniBlockScheduler {
    /// Locate one flat binary value without reading its payload.
    ///
    /// Returns `None` for unsupported layouts or metadata outside the bounded
    /// fast path. Callers must retain ordinary decoding as a fallback. The
    /// handle exposes the value length and supports repeated byte-range reads
    /// without reinitializing chunk metadata. No persisted encoding changes.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use lance_encoding::{EncodingsIo, decoder::PageInfo, encodings::logical::primitive::MiniBlockScheduler};
    /// # async fn example(page: &PageInfo, io: &Arc<dyn EncodingsIo>) -> lance_core::Result<()> {
    /// if let Some(value) = MiniBlockScheduler::try_prepare_binary_value(page, 0, io).await? {
    ///     let bytes = value.read_slices(&[0..value.len().min(4)]).await?;
    ///     assert!(bytes[0].len() <= 4);
    /// }
    /// # Ok(()) }
    /// ```
    pub async fn try_prepare_binary_value(
        page: &PageInfo,
        row: u64,
        io: &Arc<dyn EncodingsIo>,
    ) -> Result<Option<PreparedBinaryValue>> {
        if row >= page.num_rows {
            return Err(Error::invalid_input("invalid binary row"));
        }
        let page_layout = match &page.encoding {
            PageEncoding::Structural(layout) => Cow::Borrowed(layout),
            PageEncoding::Planar(layout) => {
                if !matches!(
                    layout
                        .structure
                        .as_ref()
                        .and_then(|structure| structure.codec.as_ref()),
                    Some(crate::format::pb23::structural_plane::Codec::Dense(_))
                ) {
                    return Ok(None);
                }
                Cow::Owned(layout.dense_execution_layout()?)
            }
            PageEncoding::Legacy(_) => return Ok(None),
        };
        let Some(Layout::MiniBlockLayout(layout)) = &page_layout.layout else {
            return Ok(None);
        };
        if layout.rep_compression.is_some()
            || layout.def_compression.is_some()
            || layout.dictionary.is_some()
            || layout.repetition_index_depth != 0
            || layout.num_items != page.num_rows
        {
            return Ok(None);
        }
        let Some(Compression::Variable(variable)) = layout
            .value_compression
            .as_ref()
            .and_then(|v| v.compression.as_ref())
        else {
            return Ok(None);
        };
        let decoder = BinaryMiniBlockDecompressor::from_variable(variable)?;
        if layout.num_buffers != decoder.num_buffers() as u64
            || page.buffer_offsets_and_sizes.len() < 2
        {
            return Err(Error::corrupt_file_named(
                "binary mini-block",
                "invalid binary page buffers",
            ));
        }
        if page
            .buffer_offsets_and_sizes
            .iter()
            .any(|(offset, size)| offset.checked_add(*size).is_none())
        {
            return Err(Error::corrupt_file_named(
                "binary mini-block",
                "page buffer range overflows u64",
            ));
        }
        let metadata_len = page.buffer_offsets_and_sizes[0].1;
        let word_bytes = if layout.has_large_chunk { 4 } else { 2 };
        if metadata_len == 0 || !metadata_len.is_multiple_of(word_bytes) {
            return Err(Error::corrupt_file_named(
                "binary mini-block",
                "invalid metadata word length",
            ));
        }
        let mut scheduler = Self::try_new(
            &page.buffer_offsets_and_sizes,
            page.priority,
            page.num_rows,
            layout,
            &DefaultDecompressionStrategy {},
        )?;
        let initialization = scheduler.init_layout()?;
        let ranges = initialization
            .required_ranges()
            .cloned()
            .collect::<Vec<_>>();
        let metadata_bytes = ranges
            .iter()
            .try_fold(0_u64, |total, range| {
                range
                    .end
                    .checked_sub(range.start)
                    .and_then(|len| total.checked_add(len))
            })
            .ok_or_else(|| {
                Error::corrupt_file_named("binary mini-block", "invalid metadata ranges")
            })?;
        if metadata_bytes > MAX_METADATA_BYTES {
            return Ok(None);
        }
        let buffers = io.submit_request(ranges.clone(), page.priority).await?;
        if buffers.len() != ranges.len()
            || buffers
                .iter()
                .zip(&ranges)
                .any(|(bytes, range)| bytes.len() as u64 != range.end - range.start)
        {
            return Err(Error::corrupt_file_named(
                "binary mini-block",
                "truncated metadata read",
            ));
        }
        let state = scheduler
            .init_from_buffers(initialization.with_buffers(buffers)?, io)
            .await?;
        scheduler.try_load(&state)?;
        let index = &scheduler
            .page_meta
            .as_ref()
            .ok_or_else(|| {
                Error::internal("binary mini-block initialization did not load its index")
            })?
            .chunk_index;
        let instructions = ChunkInstructions::schedule_instructions(index, &[row..row + 1]);
        if instructions.len() != 1 {
            return Err(Error::corrupt_file_named(
                "binary mini-block",
                "flat value spans multiple chunks",
            ));
        }
        let instruction = &instructions[0];
        let chunk = index.byte_range(instruction.chunk_idx);
        let header_len = 2 + decoder.num_buffers() * if layout.has_large_chunk { 4 } else { 2 };
        if chunk
            .end
            .checked_sub(chunk.start)
            .is_none_or(|len| len < header_len as u64)
        {
            return Err(Error::corrupt_file_named(
                "binary mini-block",
                "truncated binary chunk header",
            ));
        }
        let mut header = io
            .submit_request(
                vec![chunk.start..chunk.start + header_len as u64],
                page.priority,
            )
            .await?;
        if header.len() != 1 || header[0].len() != header_len {
            return Err(Error::corrupt_file_named(
                "binary mini-block",
                "truncated binary chunk header read",
            ));
        }
        let header = header.pop().ok_or_else(|| {
            Error::corrupt_file_named("binary mini-block", "missing binary chunk header")
        })?;
        let mut relative = header_len.next_multiple_of(MINIBLOCK_ALIGNMENT) as u64;
        let mut value_buffers = Vec::with_capacity(decoder.num_buffers());
        for buffer in 0..decoder.num_buffers() {
            let size = if layout.has_large_chunk {
                let start = 2 + buffer * 4;
                u64::from(u32::from_le_bytes(
                    header[start..start + 4].try_into().map_err(|_| {
                        Error::corrupt_file_named("binary mini-block", "invalid large buffer size")
                    })?,
                ))
            } else {
                let start = 2 + buffer * 2;
                u64::from(u16::from_le_bytes(
                    header[start..start + 2].try_into().map_err(|_| {
                        Error::corrupt_file_named("binary mini-block", "invalid buffer size")
                    })?,
                ))
            };
            let start = chunk.start.checked_add(relative);
            let end = start.and_then(|start| start.checked_add(size));
            let Some((start, end)) = start.zip(end).filter(|(_, end)| *end <= chunk.end) else {
                return Err(Error::corrupt_file_named(
                    "binary mini-block",
                    "binary chunk buffer exceeds its chunk",
                ));
            };
            value_buffers.push(start..end);
            relative = relative.checked_add(size).ok_or_else(|| {
                Error::corrupt_file_named("binary mini-block", "binary chunk size overflow")
            })?;
            relative = relative
                .checked_add(
                    (MINIBLOCK_ALIGNMENT as u64 - relative % MINIBLOCK_ALIGNMENT as u64)
                        % MINIBLOCK_ALIGNMENT as u64,
                )
                .ok_or_else(|| {
                    Error::corrupt_file_named("binary mini-block", "binary chunk padding overflow")
                })?;
        }
        decoder
            .prepare_value(
                &value_buffers,
                index.items_in_chunk(instruction.chunk_idx),
                instruction.rows_to_skip,
                io,
                page.priority,
            )
            .await
    }
}

#[cfg(test)]
mod tests {
    use std::{ops::Range, sync::Mutex};

    use futures::{FutureExt, future::BoxFuture};
    use rstest::rstest;

    use super::*;
    use crate::{
        buffer::LanceBuffer,
        compression_config::CompressionFieldParams,
        data::{BlockInfo, DataBlock, VariableWidthBlock},
        encodings::{
            logical::primitive::miniblock::{MiniBlockCompressionContext, MiniBlockCompressor},
            physical::binary::BinaryMiniBlockEncoder,
        },
        format::metadata,
    };

    #[derive(Debug)]
    struct RecordedIo {
        bytes: Bytes,
        requests: Mutex<Vec<(Range<u64>, u64)>>,
        short_read_at: Option<usize>,
    }

    impl EncodingsIo for RecordedIo {
        fn submit_request(
            &self,
            ranges: Vec<Range<u64>>,
            priority: u64,
        ) -> BoxFuture<'static, Result<Vec<Bytes>>> {
            let result = ranges
                .into_iter()
                .map(|range| {
                    let request_index = {
                        let mut requests = self.requests.lock().unwrap();
                        requests.push((range.clone(), priority));
                        requests.len()
                    };
                    let bytes = self
                        .bytes
                        .get(range.start as usize..range.end as usize)
                        .ok_or_else(|| Error::io("test read outside file"))?;
                    let len = bytes
                        .len()
                        .saturating_sub(usize::from(self.short_read_at == Some(request_index)));
                    Ok(Bytes::copy_from_slice(&bytes[..len]))
                })
                .collect();
            std::future::ready(result).boxed()
        }
    }

    async fn read_selected(
        page: &PageInfo,
        row: u64,
        slices: &[Range<u64>],
        io: &Arc<dyn EncodingsIo>,
    ) -> Result<Option<Vec<Bytes>>> {
        match MiniBlockScheduler::try_prepare_binary_value(page, row, io).await? {
            Some(value) => value.read_slices(slices).await.map(Some),
            None => Ok(None),
        }
    }

    fn page_fixture(
        width: u8,
        wide: bool,
        generic: bool,
        payload_len: usize,
    ) -> (PageInfo, RecordedIo, Vec<Vec<u8>>) {
        let values = vec![vec![b'a'; payload_len], vec![b'b'; payload_len]];
        let offsets = if width == 32 {
            LanceBuffer::reinterpret_vec(vec![0_i32, payload_len as i32, (payload_len * 2) as i32])
        } else {
            LanceBuffer::reinterpret_vec(vec![0_i64, payload_len as i64, (payload_len * 2) as i64])
        };
        let block = DataBlock::VariableWidth(VariableWidthBlock {
            data: LanceBuffer::from(values.concat()),
            offsets,
            bits_per_offset: width,
            num_values: 2,
            block_info: BlockInfo::new(),
        });
        let encoder = if generic {
            BinaryMiniBlockEncoder::with_generic_offsets(
                Some(32768),
                CompressionFieldParams::default(),
            )
        } else {
            BinaryMiniBlockEncoder::new(Some(32768))
        };
        let (encoded, encoding) = encoder
            .compress(MiniBlockCompressionContext::new(0, wide, generic), block)
            .unwrap();
        assert_eq!(encoded.chunks.len(), 1);
        // Place real codec output into one final structural chunk. Its metadata
        // word has log_num_values=0, the existing final-chunk sentinel.
        let mut chunk = vec![0_u8; 2];
        for size in &encoded.chunks[0].buffer_sizes {
            if wide {
                chunk.extend_from_slice(&size.to_le_bytes());
            } else {
                chunk.extend_from_slice(&u16::try_from(*size).unwrap().to_le_bytes());
            }
        }
        chunk.resize(chunk.len().next_multiple_of(MINIBLOCK_ALIGNMENT), 0);
        for buffer in &encoded.data {
            chunk.extend_from_slice(buffer);
            chunk.resize(chunk.len().next_multiple_of(MINIBLOCK_ALIGNMENT), 0);
        }
        let word = ((chunk.len() / MINIBLOCK_ALIGNMENT - 1) << 4) as u32;
        let mut file = vec![0_u8; 128];
        let metadata_len = if wide {
            file[32..36].copy_from_slice(&word.to_le_bytes());
            4
        } else {
            file[32..34].copy_from_slice(&u16::try_from(word).unwrap().to_le_bytes());
            2
        };
        file.extend_from_slice(&chunk);
        let page = PageInfo {
            num_rows: 2,
            priority: 7,
            buffer_offsets_and_sizes: Arc::from([(32, metadata_len), (128, chunk.len() as u64)]),
            encoding: PageEncoding::Structural(metadata::PageLayout {
                layout: Some(Layout::MiniBlockLayout(metadata::MiniBlockLayout {
                    value_compression: Some(encoding),
                    num_buffers: encoded.data.len() as u64,
                    has_large_chunk: wide,
                    num_items: 2,
                    ..Default::default()
                })),
            }),
        };
        (
            page,
            RecordedIo {
                bytes: file.into(),
                requests: Mutex::default(),
                short_read_at: None,
            },
            values,
        )
    }

    #[rstest]
    #[tokio::test]
    async fn binary_slices_preserve_selection_without_whole_value_reads(
        #[values(32, 64)] width: u8,
        #[values(false, true)] wide: bool,
        #[values(false, true)] generic: bool,
        #[values(0, 1)] row: usize,
        #[values(0, 8192)] payload_len: usize,
    ) {
        let (page, recorded, values) = page_fixture(width, wide, generic, payload_len);
        let recorded = Arc::new(recorded);
        let io: Arc<dyn EncodingsIo> = recorded.clone();
        let ranges = if payload_len == 0 {
            [0..0, 0..0, 0..0, 0..0]
        } else {
            [100..200, 0..0, 50..80, 100..120]
        };
        let value = MiniBlockScheduler::try_prepare_binary_value(&page, row as u64, &io)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(value.len(), payload_len as u64);
        assert_eq!(value.is_empty(), payload_len == 0);
        let metadata_requests = recorded.requests.lock().unwrap().len();
        let result = value.read_slices(&ranges).await.unwrap();
        let tail_start = value.len().saturating_sub(1);
        let tail = value
            .clone()
            .read_slices(&[tail_start..value.len()])
            .await
            .unwrap();
        assert_eq!(tail[0].as_ref(), &values[row][tail_start as usize..]);
        assert_eq!(result.len(), ranges.len());
        for (bytes, range) in result.iter().zip(&ranges) {
            assert_eq!(
                bytes.as_ref(),
                &values[row][range.start as usize..range.end as usize]
            );
        }
        let requests = recorded.requests.lock().unwrap();
        assert!(requests.iter().all(|(_, priority)| *priority == 7));
        let body_requests =
            ranges.iter().filter(|range| !range.is_empty()).count() + usize::from(payload_len > 0);
        assert_eq!(
            requests.len(),
            metadata_requests + body_requests,
            "preparation metadata must be reused on every body read"
        );
        let requested_bytes: u64 = requests
            .iter()
            .map(|(range, _)| range.end - range.start)
            .sum();
        assert!(
            requested_bytes < 512,
            "bounded metadata and150 selected payload bytes, got {requested_bytes}"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn binary_slices_reject_bounds_and_short_reads(
        #[values(1, 2, 3, 4)] short_read_at: usize,
    ) {
        let (page, mut recorded, _) = page_fixture(64, true, false, 8192);
        recorded.short_read_at = Some(short_read_at);
        let io: Arc<dyn EncodingsIo> = Arc::new(recorded);
        let error = read_selected(&page, 0, &[0..1], &io).await.unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("truncated"));
        let error = read_selected(&page, 2, &[0..1], &io).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("invalid binary row"));
    }

    #[tokio::test]
    async fn binary_slices_prepared_clone_owns_only_its_io_scope() {
        let (page, recorded, values) = page_fixture(64, true, false, 8192);
        let recorded = Arc::new(recorded);
        let weak = Arc::downgrade(&recorded);
        let io: Arc<dyn EncodingsIo> = recorded.clone();
        let value = MiniBlockScheduler::try_prepare_binary_value(&page, 1, &io)
            .await
            .unwrap()
            .unwrap();
        let clone = value.clone();
        drop(value);
        drop(io);
        drop(recorded);
        let bytes = clone.read_slices(&[0..4]).await.unwrap();
        assert_eq!(bytes[0].as_ref(), &values[1][..4]);
        assert!(weak.upgrade().is_some());
        drop(clone);
        assert!(
            weak.upgrade().is_none(),
            "a completed read must not retain the handle's I/O scope"
        );
        assert_eq!(bytes[0].as_ref(), &values[1][..4]);
    }

    #[tokio::test]
    async fn binary_slices_prepared_value_can_retry_payload_failure() {
        let (page, mut recorded, values) = page_fixture(64, true, false, 8192);
        recorded.short_read_at = Some(4);
        let recorded = Arc::new(recorded);
        let io: Arc<dyn EncodingsIo> = recorded.clone();
        let value = MiniBlockScheduler::try_prepare_binary_value(&page, 1, &io)
            .await
            .unwrap()
            .unwrap();
        let error = value
            .read_slices(&[value.len() - 4..value.len()])
            .await
            .unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("truncated binary value range"));
        let bytes = value
            .read_slices(&[value.len() - 4..value.len()])
            .await
            .unwrap();
        assert_eq!(bytes[0].as_ref(), &values[1][8188..]);
        assert_eq!(
            recorded.requests.lock().unwrap().len(),
            5,
            "retry must reuse the prepared metadata"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn binary_slices_reject_misaligned_metadata_before_io(#[values(false, true)] wide: bool) {
        let (mut page, recorded, _) = page_fixture(64, wide, false, 8192);
        let recorded = Arc::new(recorded);
        let io: Arc<dyn EncodingsIo> = recorded.clone();
        page.buffer_offsets_and_sizes = Arc::from([(32, if wide { 3 } else { 1 }), (128, 16416)]);
        let error = read_selected(&page, 0, &[0..1], &io).await.unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("metadata word length"));
        assert!(recorded.requests.lock().unwrap().is_empty());
    }

    #[rstest]
    #[case::reversed(Range { start: 20, end: 10 })]
    #[case::outside_value(8191..8193)]
    #[tokio::test]
    async fn binary_slices_reject_invalid_value_ranges(#[case] range: Range<u64>) {
        let (page, recorded, _) = page_fixture(32, false, false, 8192);
        let io: Arc<dyn EncodingsIo> = Arc::new(recorded);
        let error = read_selected(&page, 0, &[range], &io).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("range"));
    }

    #[rstest]
    #[tokio::test]
    async fn binary_slices_fallback_before_reading_unsupported_metadata(
        #[values("metadata", "nonflat", "codec")] unsupported: &str,
    ) {
        let (mut page, recorded, _) = page_fixture(32, false, false, 8192);
        let recorded = Arc::new(recorded);
        let io: Arc<dyn EncodingsIo> = recorded.clone();
        match unsupported {
            "metadata" => {
                page.buffer_offsets_and_sizes =
                    Arc::from([(32, MAX_METADATA_BYTES + 2), (128, 16416)]);
            }
            "nonflat" | "codec" => {
                let PageEncoding::Structural(metadata::PageLayout {
                    layout: Some(Layout::MiniBlockLayout(layout)),
                }) = &mut page.encoding
                else {
                    unreachable!()
                };
                if unsupported == "nonflat" {
                    layout.num_items += 1;
                } else {
                    layout.value_compression = None;
                }
            }
            _ => unreachable!(),
        }
        assert!(
            read_selected(&page, 0, &[0..1], &io)
                .await
                .unwrap()
                .is_none()
        );
        assert!(recorded.requests.lock().unwrap().is_empty());
    }
}
