// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::any::Any;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;
use std::vec;

use arrow_buffer::ScalarBuffer;
use bytes::Bytes;
use futures::FutureExt;
use futures::future::BoxFuture;
use itertools::Itertools;
use lance_core::Result;
use lance_core::cache::{Context, DeepSizeOf};
use lance_core::error::Error;

use crate::EncodingsIo;
use crate::buffer::LanceBuffer;
use crate::compression::{
    DecompressionStrategy, FixedPerValueDecompressor, VariablePerValueDecompressor,
};
use crate::data::{BlockInfo, DataBlockBuilder, FixedWidthDataBlock, VariableWidthBlock};
use crate::decoder::{DecodePageTask, DecodedPage, StructuralPageDecoder};
use crate::encodings::logical::primitive::fullzip::PerValueDataBlock;
use crate::format::{EncodingUtils, metadata};
use crate::repdef::{
    ControlWordParser, DefinitionInterpretation, RepDefUnraveler, max_visible_level,
};
use crate::utils::bytepack::ByteUnpacker;

use super::{
    CachedPageData, NoCachedPageData, PageInitialization, PageInitializationBuffers, PageLoadTask,
    StructuralPageScheduler,
};

#[derive(Debug, Clone, Copy)]
struct FullZipRepIndexDetails {
    buf_position: u64,
    bytes_per_value: u64, // Will be 1, 2, 4, or 8
}

#[derive(Debug)]
enum PerValueDecompressor {
    Fixed(Arc<dyn FixedPerValueDecompressor>),
    Variable(Arc<dyn VariablePerValueDecompressor>),
}

#[derive(Debug)]
struct FullZipDecodeDetails {
    value_decompressor: PerValueDecompressor,
    def_meaning: Arc<[DefinitionInterpretation]>,
    ctrl_word_parser: ControlWordParser,
    max_rep: u16,
    max_visible_def: u16,
}

/// Describes where FullZip byte ranges should be read from.
///
/// FullZip decoding always needs a list of byte ranges, but those bytes can come
/// from two different places:
/// - Remote I/O (normal path): ranges are fetched from the underlying `EncodingsIo`.
/// - A prefetched full page (full scan fast path): the entire page has already been
///   loaded once and ranges should be sliced from memory.
///
/// This abstraction keeps scheduling code focused on "which ranges are needed"
/// instead of "how bytes are fetched", and it lets full-page scans avoid the
/// two-stage rep-index -> data I/O pipeline.
#[derive(Debug, Clone)]
enum FullZipReadSource {
    /// Fetch ranges from the storage backend through the encoding I/O interface.
    Remote(Arc<dyn EncodingsIo>),
    /// Slice ranges from an already-loaded FullZip page buffer.
    PrefetchedPage { base_offset: u64, data: LanceBuffer },
}

impl FullZipReadSource {
    /// Materialize the requested ranges as decode-ready `LanceBuffer`s.
    ///
    /// The returned buffers preserve the input range order.
    fn fetch(
        &self,
        ranges: &[Range<u64>],
        priority: u64,
    ) -> BoxFuture<'static, Result<VecDeque<LanceBuffer>>> {
        match self {
            Self::Remote(io) => {
                let io = io.clone();
                let ranges = ranges.to_vec();
                async move {
                    let data = io.submit_request(ranges, priority).await?;
                    Ok(data
                        .into_iter()
                        .map(|bytes| LanceBuffer::from_bytes(bytes, 1))
                        .collect::<VecDeque<_>>())
                }
                .boxed()
            }
            Self::PrefetchedPage { base_offset, data } => {
                let base_offset = *base_offset;
                let data = data.clone();
                let page_end = base_offset + data.len() as u64;
                std::future::ready(
                    ranges
                        .iter()
                        .map(|range| {
                            if range.start > range.end
                                || range.start < base_offset
                                || range.end > page_end
                            {
                                return Err(Error::internal(format!(
                                    "Requested range {:?} is outside page range {}..{}",
                                    range, base_offset, page_end
                                )));
                            }
                            let start = (range.start - base_offset) as usize;
                            let len = (range.end - range.start) as usize;
                            Ok(data.slice_with_length(start, len))
                        })
                        .collect::<Result<VecDeque<_>>>(),
                )
                .boxed()
            }
        }
    }
}

/// A scheduler for full-zip encoded data
///
/// When the data type has a fixed-width then we simply need to map from
/// row ranges to byte ranges using the fixed-width of the data type.
///
/// When the data type is variable-width or has any repetition then a
/// repetition index is required.
#[derive(Debug)]
pub struct FullZipScheduler {
    data_buf_position: u64,
    pub(super) data_buf_size: u64,
    rep_index: Option<FullZipRepIndexDetails>,
    priority: u64,
    rows_in_page: u64,
    bits_per_offset: u8,
    details: Arc<FullZipDecodeDetails>,
    /// Cached state containing the decoded repetition index
    cached_state: Option<Arc<FullZipCacheableState>>,
    /// Whether repetition index metadata should be cached during initialize.
    pub(super) enable_cache: bool,
}

impl FullZipScheduler {
    pub(super) fn try_new(
        buffer_offsets_and_sizes: &[(u64, u64)],
        priority: u64,
        rows_in_page: u64,
        layout: &metadata::FullZipLayout,
        decompressors: &dyn DecompressionStrategy,
    ) -> Result<Self> {
        let (data_buf_position, data_buf_size) =
            *buffer_offsets_and_sizes.first().ok_or_else(|| {
                Error::corrupt_file_named("full_zip_layout", "Full-zip page has no data buffer")
            })?;
        let rep_index = buffer_offsets_and_sizes
            .get(1)
            .map(|(pos, len)| {
                let num_reps = rows_in_page.checked_add(1).ok_or_else(|| {
                    Error::corrupt_file_named(
                        "full_zip_layout",
                        format!(
                            "Full-zip page row count {rows_in_page} overflows the repetition index"
                        ),
                    )
                })?;
                if len % num_reps != 0 {
                    return Err(Error::corrupt_file_named(
                        "full_zip_layout",
                        format!(
                            "Full-zip repetition index has {len} bytes, which is not a whole number of {num_reps} rows"
                        ),
                    ));
                }
                let bytes_per_rep = len / num_reps;
                if !matches!(bytes_per_rep, 1 | 2 | 4 | 8) {
                    return Err(Error::corrupt_file_named(
                        "full_zip_layout",
                        format!(
                            "Full-zip repetition index has an unsupported width of {bytes_per_rep} bytes per row"
                        ),
                    ));
                }
                Ok(FullZipRepIndexDetails {
                    buf_position: *pos,
                    bytes_per_value: bytes_per_rep,
                })
            })
            .transpose()?;

        let bits_rep = u8::try_from(layout.bits_rep).map_err(|_| {
            Error::corrupt_file_named(
                "full_zip_layout",
                format!("Full-zip bits_rep {} does not fit in a u8", layout.bits_rep),
            )
        })?;
        let bits_def = u8::try_from(layout.bits_def).map_err(|_| {
            Error::corrupt_file_named(
                "full_zip_layout",
                format!("Full-zip bits_def {} does not fit in a u8", layout.bits_def),
            )
        })?;
        // Levels are u16 values, and the parser supports at most a 32-bit word.
        if bits_rep > 16 || bits_def > 16 {
            return Err(Error::corrupt_file_named(
                "full_zip_layout",
                format!(
                    "Full-zip control word uses {bits_rep} repetition and {bits_def} definition bits, but each side supports at most 16 bits (32 bits total)"
                ),
            ));
        }
        let ctrl_word_parser = ControlWordParser::new(bits_rep, bits_def);
        let def_meaning = layout
            .layers
            .iter()
            .map(|l| EncodingUtils::try_repdef_layer_to_def_interp(*l))
            .collect::<Result<Vec<_>>>()?;

        let max_rep = def_meaning.iter().filter(|d| d.is_list()).count() as u16;
        let max_visible_def = max_visible_level(&def_meaning);

        let value_compression = layout.value_compression.as_ref().ok_or_else(|| {
            Error::corrupt_file_named(
                "full_zip_layout",
                "Full-zip layout is missing its value compression",
            )
        })?;
        let value_decompressor = match layout.details {
            Some(metadata::full_zip_layout::Details::BitsPerValue(_)) => {
                let decompressor =
                    decompressors.create_fixed_per_value_decompressor(value_compression)?;
                PerValueDecompressor::Fixed(decompressor.into())
            }
            Some(metadata::full_zip_layout::Details::BitsPerOffset(_)) => {
                let decompressor =
                    decompressors.create_variable_per_value_decompressor(value_compression)?;
                PerValueDecompressor::Variable(decompressor.into())
            }
            None => {
                return Err(Error::corrupt_file_named(
                    "full_zip_layout",
                    "Full-zip layout must have a `details` field",
                ));
            }
        };

        let bits_per_offset = match layout.details {
            Some(metadata::full_zip_layout::Details::BitsPerValue(_)) => 32,
            Some(metadata::full_zip_layout::Details::BitsPerOffset(bits_per_offset)) => {
                bits_per_offset as u8
            }
            None => {
                return Err(Error::corrupt_file_named(
                    "full_zip_layout",
                    "Full-zip layout must have a `details` field",
                ));
            }
        };

        let details = Arc::new(FullZipDecodeDetails {
            value_decompressor,
            def_meaning: def_meaning.into(),
            ctrl_word_parser,
            max_rep,
            max_visible_def,
        });
        Ok(Self {
            data_buf_position,
            data_buf_size,
            rep_index,
            details,
            priority,
            rows_in_page,
            bits_per_offset,
            cached_state: None,
            enable_cache: false,
        })
    }

    fn covers_entire_page(ranges: &[Range<u64>], rows_in_page: u64) -> bool {
        if ranges.is_empty() {
            return false;
        }
        let mut expected_start = 0;
        for range in ranges {
            if range.start != expected_start || range.end > rows_in_page || range.end < range.start
            {
                return false;
            }
            expected_start = range.end;
        }
        expected_start == rows_in_page
    }

    fn create_page_load_task(
        io_future: BoxFuture<'static, Result<Vec<Bytes>>>,
        num_rows: u64,
        details: Arc<FullZipDecodeDetails>,
        bits_per_offset: u8,
    ) -> PageLoadTask {
        let load_task = async move {
            let buffers = io_future.await?;
            let data = buffers
                .into_iter()
                .map(|bytes| LanceBuffer::from_bytes(bytes, 1))
                .collect::<VecDeque<_>>();
            Self::create_decoder(details, data, num_rows, bits_per_offset)
        }
        .boxed();
        PageLoadTask {
            decoder_fut: load_task,
            num_rows,
        }
    }

    /// Creates a decoder from the loaded data
    fn create_decoder(
        details: Arc<FullZipDecodeDetails>,
        data: VecDeque<LanceBuffer>,
        num_rows: u64,
        bits_per_offset: u8,
    ) -> Result<Box<dyn StructuralPageDecoder>> {
        match &details.value_decompressor {
            PerValueDecompressor::Fixed(decompressor) => {
                let bits_per_value = decompressor.bits_per_value();
                if bits_per_value % 8 != 0 {
                    return Err(lance_core::Error::not_supported_source("Bit-packed full-zip encoding (non-byte-aligned values) is not yet implemented".into()));
                }
                let bytes_per_value = bits_per_value / 8;
                let total_bytes_per_value =
                    bytes_per_value as usize + details.ctrl_word_parser.bytes_per_word();
                // total_bytes_per_value == 0 is valid for constant-null FSL pages written by
                // earlier encoders that produced bits_per_value=0 with no ctrl-word bytes.
                // FixedFullZipDecoder::drain handles this case by producing AllNull output
                // without touching the (empty) data buffer.
                Ok(Box::new(FixedFullZipDecoder {
                    details,
                    data,
                    num_rows,
                    offset_in_current: 0,
                    bytes_per_value: bytes_per_value as usize,
                    total_bytes_per_value,
                }) as Box<dyn StructuralPageDecoder>)
            }
            PerValueDecompressor::Variable(_decompressor) => {
                Ok(Box::new(VariableFullZipDecoder::new(
                    details,
                    data,
                    num_rows,
                    bits_per_offset,
                    bits_per_offset,
                )?))
            }
        }
    }

    /// Extracts byte ranges from a repetition index buffer
    /// The buffer contains pairs of (start, end) values for each range
    fn extract_byte_ranges_from_pairs(
        buffer: LanceBuffer,
        bytes_per_value: u64,
        data_buf_position: u64,
    ) -> Vec<Range<u64>> {
        ByteUnpacker::new(buffer, bytes_per_value as usize)
            .chunks(2)
            .into_iter()
            .map(|mut c| {
                let start = c.next().unwrap() + data_buf_position;
                let end = c.next().unwrap() + data_buf_position;
                start..end
            })
            .collect::<Vec<_>>()
    }

    /// Extracts byte ranges from a cached repetition index buffer
    /// The buffer contains all values and we need to extract specific ranges
    fn extract_byte_ranges_from_cached(
        buffer: &LanceBuffer,
        ranges: &[Range<u64>],
        bytes_per_value: u64,
        data_buf_position: u64,
    ) -> Vec<Range<u64>> {
        ranges
            .iter()
            .map(|r| {
                let start_offset = (r.start * bytes_per_value) as usize;
                let end_offset = (r.end * bytes_per_value) as usize;

                let start_slice = &buffer[start_offset..start_offset + bytes_per_value as usize];
                let start_val =
                    ByteUnpacker::new(start_slice.iter().copied(), bytes_per_value as usize)
                        .next()
                        .unwrap();

                let end_slice = &buffer[end_offset..end_offset + bytes_per_value as usize];
                let end_val =
                    ByteUnpacker::new(end_slice.iter().copied(), bytes_per_value as usize)
                        .next()
                        .unwrap();

                (data_buf_position + start_val)..(data_buf_position + end_val)
            })
            .collect()
    }

    /// Computes the ranges in the repetition index that need to be loaded
    fn compute_rep_index_ranges(
        ranges: &[Range<u64>],
        rep_index: &FullZipRepIndexDetails,
    ) -> Vec<Range<u64>> {
        ranges
            .iter()
            .flat_map(|r| {
                let first_val_start =
                    rep_index.buf_position + (r.start * rep_index.bytes_per_value);
                let first_val_end = first_val_start + rep_index.bytes_per_value;
                let last_val_start = rep_index.buf_position + (r.end * rep_index.bytes_per_value);
                let last_val_end = last_val_start + rep_index.bytes_per_value;
                [first_val_start..first_val_end, last_val_start..last_val_end]
            })
            .collect()
    }

    /// Schedules ranges in the presence of a repetition index
    fn schedule_ranges_rep(
        &self,
        ranges: &[Range<u64>],
        io: &Arc<dyn EncodingsIo>,
        rep_index: FullZipRepIndexDetails,
    ) -> Result<Vec<PageLoadTask>> {
        let num_rows = ranges.iter().map(|r| r.end - r.start).sum();
        let data_buf_position = self.data_buf_position;
        let priority = self.priority;
        let details = self.details.clone();
        let bits_per_offset = self.bits_per_offset;

        if Self::covers_entire_page(ranges, self.rows_in_page) {
            let full_range = self.data_buf_position..(self.data_buf_position + self.data_buf_size);
            let page_data = io.submit_single(full_range.clone(), priority);
            let load_task = async move {
                let page_data = page_data.await?;
                let source = FullZipReadSource::PrefetchedPage {
                    base_offset: full_range.start,
                    data: LanceBuffer::from_bytes(page_data, 1),
                };
                let read_ranges = vec![full_range];
                let data = source.fetch(&read_ranges, priority).await?;
                Self::create_decoder(details, data, num_rows, bits_per_offset)
            }
            .boxed();
            let page_load_task = PageLoadTask {
                decoder_fut: load_task,
                num_rows,
            };
            return Ok(vec![page_load_task]);
        }

        if let Some(cached_state) = &self.cached_state {
            let byte_ranges = Self::extract_byte_ranges_from_cached(
                &cached_state.rep_index_buffer,
                ranges,
                rep_index.bytes_per_value,
                data_buf_position,
            );
            let io_future = io.submit_request(byte_ranges, priority);
            let page_load_task =
                Self::create_page_load_task(io_future, num_rows, details, bits_per_offset);
            return Ok(vec![page_load_task]);
        }

        let rep_ranges = Self::compute_rep_index_ranges(ranges, &rep_index);
        let rep_data = io.submit_request(rep_ranges, priority);
        let io_clone = io.clone();
        let load_task = async move {
            let rep_data = rep_data.await?;
            let rep_buffer = LanceBuffer::concat(
                &rep_data
                    .into_iter()
                    .map(|d| LanceBuffer::from_bytes(d, 1))
                    .collect::<Vec<_>>(),
            );
            let byte_ranges = Self::extract_byte_ranges_from_pairs(
                rep_buffer,
                rep_index.bytes_per_value,
                data_buf_position,
            );
            let source = FullZipReadSource::Remote(io_clone);
            let data = source.fetch(&byte_ranges, priority).await?;
            Self::create_decoder(details, data, num_rows, bits_per_offset)
        }
        .boxed();
        let page_load_task = PageLoadTask {
            decoder_fut: load_task,
            num_rows,
        };
        Ok(vec![page_load_task])
    }

    // In the simple case there is no repetition and we just have large fixed-width
    // rows of data.  We can just map row ranges to byte ranges directly using the
    // fixed-width of the data type.
    fn schedule_ranges_simple(
        &self,
        ranges: &[Range<u64>],
        io: &Arc<dyn EncodingsIo>,
    ) -> Result<Vec<PageLoadTask>> {
        // Convert row ranges to item ranges (i.e. multiply by items per row)
        let num_rows = ranges.iter().map(|r| r.end - r.start).sum();

        let PerValueDecompressor::Fixed(decompressor) = &self.details.value_decompressor else {
            unreachable!()
        };

        // Convert item ranges to byte ranges (i.e. multiply by bytes per item)
        let bits_per_value = decompressor.bits_per_value();
        if !bits_per_value.is_multiple_of(8) {
            return Err(Error::invalid_input_source(
                format!(
                    "Full-zip fixed-width values must be byte aligned, got {} bits per value",
                    bits_per_value
                )
                .into(),
            ));
        }
        let bytes_per_value = bits_per_value / 8;
        let bytes_per_cw = self.details.ctrl_word_parser.bytes_per_word();
        let total_bytes_per_value = bytes_per_value + bytes_per_cw as u64;
        let byte_ranges = ranges
            .iter()
            .map(|r| {
                debug_assert!(r.end <= self.rows_in_page);
                let start = self.data_buf_position + r.start * total_bytes_per_value;
                let end = self.data_buf_position + r.end * total_bytes_per_value;
                start..end
            })
            .collect::<Vec<_>>();

        let io_future = io.submit_request(byte_ranges, self.priority);
        let page_load_task = Self::create_page_load_task(
            io_future,
            num_rows,
            self.details.clone(),
            self.bits_per_offset,
        );
        Ok(vec![page_load_task])
    }
}

/// Cacheable state for FullZip encoding, storing the decoded repetition index
#[derive(Debug)]
struct FullZipCacheableState {
    /// The raw repetition index buffer for future decoding
    rep_index_buffer: LanceBuffer,
}

impl DeepSizeOf for FullZipCacheableState {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        self.rep_index_buffer.len()
    }
}

impl CachedPageData for FullZipCacheableState {
    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }
}

impl StructuralPageScheduler for FullZipScheduler {
    fn needs_initialization(&self) -> bool {
        self.enable_cache && self.rep_index.is_some()
    }

    fn init_layout(&self) -> Result<PageInitialization> {
        let repetition_index_range = if self.enable_cache
            && let Some(rep_index) = self.rep_index
        {
            let total_size = (self.rows_in_page + 1) * rep_index.bytes_per_value;
            Some(rep_index.buf_position..(rep_index.buf_position + total_size))
        } else {
            None
        };
        Ok(PageInitialization::new([repetition_index_range]))
    }

    fn init_from_buffers<'a>(
        &'a mut self,
        buffers: PageInitializationBuffers,
        _io: &Arc<dyn EncodingsIo>,
    ) -> BoxFuture<'a, Result<Arc<dyn CachedPageData>>> {
        let [rep_index_buffer] = match buffers.try_into_array("full-zip") {
            Ok(buffers) => buffers,
            Err(error) => return std::future::ready(Err(error)).boxed(),
        };
        if let Some(rep_index_buffer) = rep_index_buffer {
            let state = Arc::new(FullZipCacheableState {
                rep_index_buffer: LanceBuffer::from_bytes(rep_index_buffer, 1),
            });
            std::future::ready(Ok(state as Arc<dyn CachedPageData>)).boxed()
        } else {
            std::future::ready(Ok(Arc::new(NoCachedPageData) as Arc<dyn CachedPageData>)).boxed()
        }
    }

    /// Loads previously cached repetition index data from the cache system.
    /// This method is called when a scheduler instance needs to use cached data
    /// that was initialized by another instance or in a previous operation.
    fn try_load(&mut self, cache: &Arc<dyn CachedPageData>) -> Result<()> {
        self.cached_state = Some(cache.clone().as_arc_any().downcast().map_err(|_| {
            Error::invalid_input_source("Cached full-zip page data has an unexpected type".into())
        })?);
        Ok(())
    }

    fn schedule_ranges(
        &self,
        ranges: &[Range<u64>],
        io: &Arc<dyn EncodingsIo>,
    ) -> Result<Vec<PageLoadTask>> {
        if let Some(rep_index) = self.rep_index {
            self.schedule_ranges_rep(ranges, io, rep_index)
        } else {
            self.schedule_ranges_simple(ranges, io)
        }
    }
}

/// A decoder for full-zip encoded data when the data has a fixed-width
///
/// Here we need to unzip the control words from the values themselves and
/// then decompress the requested values.
///
/// We use a PerValueDecompressor because we will only be decompressing the
/// requested data.  This decoder / scheduler does not do any read amplification.
#[derive(Debug)]
struct FixedFullZipDecoder {
    details: Arc<FullZipDecodeDetails>,
    data: VecDeque<LanceBuffer>,
    offset_in_current: usize,
    bytes_per_value: usize,
    total_bytes_per_value: usize,
    num_rows: u64,
}

impl FixedFullZipDecoder {
    fn slice_next_task(&mut self, num_rows: u64) -> Result<FullZipDecodeTaskItem> {
        debug_assert!(num_rows > 0);
        let cur_buf = self.data.front_mut().ok_or_else(|| {
            Error::corrupt_file_named(
                "fixed_full_zip",
                format!("page data ran out with {num_rows} row(s) left to decode"),
            )
        })?;
        let start = self.offset_in_current;
        if self.details.ctrl_word_parser.has_rep() {
            // This is a slightly slower path.  In order to figure out where to split we need to
            // examine the rep index so we can convert num_lists to num_rows
            let mut rows_started = 0;
            // We always need at least one value.  Now loop through until we have passed num_rows
            // values
            let mut num_items = 0;
            let bytes_per_word = self.details.ctrl_word_parser.bytes_per_word();
            while self.offset_in_current < cur_buf.len() {
                // The item walk is driven by the control words themselves, so a page that
                // disagrees with its own layout can step past the end of the buffer.  Check
                // before every read and every advance so that a malformed page is reported
                // instead of panicking in `slice_with_length`.
                let bytes_avail = cur_buf.len() - self.offset_in_current;
                if bytes_avail < bytes_per_word {
                    return Err(Error::corrupt_file_named(
                        "fixed_full_zip",
                        format!(
                            "truncated control word: {bytes_avail} byte(s) remain in the page \
                             buffer but a control word requires {bytes_per_word}"
                        ),
                    ));
                }
                let control = self.details.ctrl_word_parser.parse_desc(
                    &cur_buf[self.offset_in_current..],
                    self.details.max_rep,
                    self.details.max_visible_def,
                );
                if control.is_new_row {
                    if rows_started == num_rows {
                        break;
                    }
                    rows_started += 1;
                }
                num_items += 1;
                let bytes_in_item = if control.is_visible {
                    self.total_bytes_per_value
                } else {
                    bytes_per_word
                };
                if bytes_in_item > bytes_avail {
                    return Err(Error::corrupt_file_named(
                        "fixed_full_zip",
                        format!(
                            "truncated item: {bytes_avail} byte(s) remain in the page buffer at \
                             offset {} but the item requires {bytes_in_item}",
                            self.offset_in_current
                        ),
                    ));
                }
                self.offset_in_current += bytes_in_item;
            }

            let task_slice = cur_buf.slice_with_length(start, self.offset_in_current - start);
            if self.offset_in_current == cur_buf.len() {
                self.data.pop_front();
                self.offset_in_current = 0;
            }

            Ok(FullZipDecodeTaskItem {
                data: PerValueDataBlock::Fixed(FixedWidthDataBlock {
                    data: task_slice,
                    bits_per_value: self.bytes_per_value as u64 * 8,
                    num_values: num_items,
                    block_info: BlockInfo::new(),
                }),
                rows_in_buf: rows_started,
            })
        } else {
            // If there's no repetition we can calculate the slicing point by just multiplying
            // the number of rows by the total bytes per value
            let cur_buf = self.data.front_mut().unwrap();
            let bytes_avail = cur_buf.len() - self.offset_in_current;
            let offset_in_cur = self.offset_in_current;

            let bytes_needed = num_rows as usize * self.total_bytes_per_value;
            let mut rows_taken = num_rows;
            let task_slice = if bytes_needed >= bytes_avail {
                self.offset_in_current = 0;
                rows_taken = bytes_avail as u64 / self.total_bytes_per_value as u64;
                self.data
                    .pop_front()
                    .unwrap()
                    .slice_with_length(offset_in_cur, bytes_avail)
            } else {
                self.offset_in_current += bytes_needed;
                cur_buf.slice_with_length(offset_in_cur, bytes_needed)
            };
            Ok(FullZipDecodeTaskItem {
                data: PerValueDataBlock::Fixed(FixedWidthDataBlock {
                    data: task_slice,
                    bits_per_value: self.bytes_per_value as u64 * 8,
                    num_values: rows_taken,
                    block_info: BlockInfo::new(),
                }),
                rows_in_buf: rows_taken,
            })
        }
    }
}

impl StructuralPageDecoder for FixedFullZipDecoder {
    fn drain(&mut self, num_rows: u64) -> Result<Box<dyn DecodePageTask>> {
        if self.total_bytes_per_value == 0 {
            // No bytes per row: constant-null page with no ctrl-word bytes.
            // The decompressor (ConstantDecompressor) ignores its input and returns AllNull.
            return Ok(Box::new(FixedFullZipDecodeTask {
                details: self.details.clone(),
                data: vec![FullZipDecodeTaskItem {
                    data: PerValueDataBlock::Fixed(FixedWidthDataBlock {
                        data: LanceBuffer::empty(),
                        bits_per_value: 0,
                        num_values: num_rows,
                        block_info: BlockInfo::new(),
                    }),
                    rows_in_buf: num_rows,
                }],
                bytes_per_value: 0,
                num_rows: num_rows as usize,
            }));
        }
        let mut task_data = Vec::with_capacity(self.data.len());
        let mut remaining = num_rows;
        while remaining > 0 {
            let task_item = self.slice_next_task(remaining)?;
            remaining -= task_item.rows_in_buf;
            task_data.push(task_item);
        }
        Ok(Box::new(FixedFullZipDecodeTask {
            details: self.details.clone(),
            data: task_data,
            bytes_per_value: self.bytes_per_value,
            num_rows: num_rows as usize,
        }))
    }

    fn num_rows(&self) -> u64 {
        self.num_rows
    }
}

/// A decoder for full-zip encoded data when the data has a variable-width
///
/// Here we need to unzip the control words AND lengths from the values and
/// then decompress the requested values.
#[derive(Debug)]
struct VariableFullZipDecoder {
    details: Arc<FullZipDecodeDetails>,
    decompressor: Arc<dyn VariablePerValueDecompressor>,
    data: LanceBuffer,
    offsets: LanceBuffer,
    rep: ScalarBuffer<u16>,
    def: ScalarBuffer<u16>,
    repdef_starts: Vec<usize>,
    data_starts: Vec<usize>,
    offset_starts: Vec<usize>,
    visible_item_counts: Vec<u64>,
    bits_per_offset: u8,
    current_idx: usize,
    num_rows: u64,
}

impl VariableFullZipDecoder {
    fn new(
        details: Arc<FullZipDecodeDetails>,
        data: VecDeque<LanceBuffer>,
        num_rows: u64,
        in_bits_per_length: u8,
        out_bits_per_offset: u8,
    ) -> Result<Self> {
        let decompressor = match details.value_decompressor {
            PerValueDecompressor::Variable(ref d) => d.clone(),
            _ => unreachable!(),
        };

        assert_eq!(in_bits_per_length % 8, 0);
        assert!(out_bits_per_offset == 32 || out_bits_per_offset == 64);

        let mut decoder = Self {
            details,
            decompressor,
            data: LanceBuffer::empty(),
            offsets: LanceBuffer::empty(),
            rep: LanceBuffer::empty().borrow_to_typed_slice(),
            def: LanceBuffer::empty().borrow_to_typed_slice(),
            bits_per_offset: out_bits_per_offset,
            repdef_starts: Vec::with_capacity(num_rows as usize + 1),
            data_starts: Vec::with_capacity(num_rows as usize + 1),
            offset_starts: Vec::with_capacity(num_rows as usize + 1),
            visible_item_counts: Vec::with_capacity(num_rows as usize + 1),
            current_idx: 0,
            num_rows,
        };

        // There's no great time to do this and this is the least worst time.  If we don't unzip then
        // we can't slice the data during the decode phase.  This is because we need the offsets to be
        // unpacked to know where the values start and end.
        //
        // We don't want to unzip on the decode thread because that is a single-threaded path
        // We don't want to unzip on the scheduling thread because that is a single-threaded path
        //
        // Fortunately, we know variable length data will always be read indirectly and so we can do it
        // here, which should be on the indirect thread.  The primary disadvantage to doing it here is that
        // we load all the data into memory and then throw it away only to load it all into memory again during
        // the decode.
        //
        // There are some alternatives to investigate:
        //   - Instead of just reading the beginning and end of the rep index we could read the entire
        //     range in between.  This will give us the break points that we need for slicing and won't increase
        //     the number of IOPs but it will mean we are doing more total I/O and we need to load the rep index
        //     even when doing a full scan.
        //   - We could force each decode task to do a full unzip of all the data.  Each decode task now
        //     has to do more work but the work is all fused.
        //   - We could just try doing this work on the decode thread and see if it is a problem.
        decoder.unzip(data, in_bits_per_length, out_bits_per_offset, num_rows)?;

        Ok(decoder)
    }

    fn slice_batch_data_and_rebase_offsets_typed<T>(
        data: &LanceBuffer,
        offsets: &LanceBuffer,
    ) -> Result<(LanceBuffer, LanceBuffer)>
    where
        T: arrow_buffer::ArrowNativeType
            + Copy
            + PartialOrd
            + std::ops::Sub<Output = T>
            + std::fmt::Display
            + TryInto<usize>,
    {
        let offsets_slice = offsets.borrow_to_typed_slice::<T>();
        let offsets_slice = offsets_slice.as_ref();
        if offsets_slice.is_empty() {
            return Err(Error::internal(
                "Variable offsets cannot be empty".to_string(),
            ));
        }

        let base = offsets_slice[0];
        let end = *offsets_slice.last().unwrap();
        if end < base {
            return Err(Error::internal(format!(
                "Invalid variable offsets: end ({end}) is less than base ({base})"
            )));
        }

        let data_start = base.try_into().map_err(|_| {
            Error::internal(format!("Variable offset ({base}) does not fit into usize"))
        })?;
        let data_end = end.try_into().map_err(|_| {
            Error::internal(format!("Variable offset ({end}) does not fit into usize"))
        })?;
        if data_end > data.len() {
            return Err(Error::internal(format!(
                "Invalid variable offsets: end ({data_end}) exceeds data len ({})",
                data.len()
            )));
        }

        let mut rebased_offsets = Vec::with_capacity(offsets_slice.len());
        for &offset in offsets_slice {
            if offset < base {
                return Err(Error::internal(format!(
                    "Invalid variable offsets: offset ({offset}) is less than base ({base})"
                )));
            }
            rebased_offsets.push(offset - base);
        }

        let sliced_data = data.slice_with_length(data_start, data_end - data_start);
        // Copy into a compact buffer so each output batch owns only what it references.
        let sliced_data = LanceBuffer::copy_slice(&sliced_data);
        let rebased_offsets = LanceBuffer::reinterpret_vec(rebased_offsets);
        Ok((sliced_data, rebased_offsets))
    }

    fn slice_batch_data_and_rebase_offsets(
        data: &LanceBuffer,
        offsets: &LanceBuffer,
        bits_per_offset: u8,
    ) -> Result<(LanceBuffer, LanceBuffer)> {
        match bits_per_offset {
            32 => Self::slice_batch_data_and_rebase_offsets_typed::<u32>(data, offsets),
            64 => Self::slice_batch_data_and_rebase_offsets_typed::<u64>(data, offsets),
            _ => Err(Error::internal(format!(
                "Unsupported bits_per_offset={bits_per_offset}"
            ))),
        }
    }

    /// Reads a single length prefix from the front of `data`.
    ///
    /// The bytes come from the file. A page whose item walk ends with a partial
    /// trailing item leaves fewer than `bits_per_offset / 8` bytes here, so this
    /// is bounds checked and reports a corrupt file rather than reading past the
    /// end of the buffer.
    fn parse_length(data: &[u8], bits_per_offset: u8) -> Result<u64> {
        let width = bits_per_offset as usize / 8;
        if data.len() < width {
            return Err(Error::corrupt_file_named(
                "variable_full_zip",
                format!(
                    "truncated length prefix: {} byte(s) remain in the page buffer but a \
                     {}-bit length prefix requires {}",
                    data.len(),
                    bits_per_offset,
                    width
                ),
            ));
        }
        Ok(match bits_per_offset {
            8 => data[0] as u64,
            16 => u16::from_le_bytes(data[..2].try_into().unwrap()) as u64,
            32 => u32::from_le_bytes(data[..4].try_into().unwrap()) as u64,
            64 => u64::from_le_bytes(data[..8].try_into().unwrap()),
            _ => unreachable!(),
        })
    }

    fn unzip(
        &mut self,
        data: VecDeque<LanceBuffer>,
        in_bits_per_length: u8,
        out_bits_per_offset: u8,
        num_rows: u64,
    ) -> Result<()> {
        // This undercounts if there are lists but, at this point, we don't really know how many items we have
        let mut rep = Vec::with_capacity(num_rows as usize);
        let mut def = Vec::with_capacity(num_rows as usize);
        let bytes_cw = self
            .details
            .ctrl_word_parser
            .bytes_per_word()
            .saturating_mul(num_rows as usize);

        // This undercounts if there are lists
        // It can also overcount if there are invisible items
        let bytes_per_offset = out_bits_per_offset as usize / 8;
        let bytes_offsets = bytes_per_offset.saturating_mul((num_rows as usize).saturating_add(1));
        let mut offsets_data = Vec::with_capacity(bytes_offsets);

        let bytes_per_length = in_bits_per_length as usize / 8;
        let bytes_lengths = bytes_per_length.saturating_mul(num_rows as usize);

        let bytes_data = data
            .iter()
            .fold(0usize, |total, buffer| total.saturating_add(buffer.len()));
        // This overcounts since bytes_lengths and bytes_cw are undercounts
        // It can also undercount if there are invisible items (hence the saturating_sub).
        // A malformed page can make the control-word size exceed the payload, so both
        // subtractions saturate instead of underflowing into a huge allocation.
        let mut unzipped_data = Vec::with_capacity(
            bytes_data
                .saturating_sub(bytes_cw)
                .saturating_sub(bytes_lengths),
        );

        let mut current_offset = 0_u64;
        let mut visible_item_count = 0_u64;
        for databuf in data.into_iter() {
            let mut databuf = databuf.as_ref();
            while !databuf.is_empty() {
                let data_start = unzipped_data.len();
                let offset_start = offsets_data.len();
                // We might have only-rep or only-def, neither, or both.  They move at the same
                // speed though so we only need one index into it
                let repdef_start = rep.len().max(def.len());
                // TODO: Kind of inefficient we parse the control word twice here
                let ctrl_desc = self.details.ctrl_word_parser.parse_desc(
                    databuf,
                    self.details.max_rep,
                    self.details.max_visible_def,
                );
                self.details
                    .ctrl_word_parser
                    .parse(databuf, &mut rep, &mut def);
                databuf = &databuf[self.details.ctrl_word_parser.bytes_per_word()..];

                if ctrl_desc.is_new_row {
                    self.repdef_starts.push(repdef_start);
                    self.data_starts.push(data_start);
                    self.offset_starts.push(offset_start);
                    self.visible_item_counts.push(visible_item_count);
                }
                if ctrl_desc.is_visible {
                    visible_item_count += 1;
                    if ctrl_desc.is_valid_item {
                        let length = Self::parse_length(databuf, in_bits_per_length)?;
                        match out_bits_per_offset {
                            32 => offsets_data
                                .extend_from_slice(&(current_offset as u32).to_le_bytes()),
                            64 => offsets_data.extend_from_slice(&current_offset.to_le_bytes()),
                            _ => unreachable!(),
                        };
                        databuf = &databuf[bytes_per_offset..];
                        if length as usize > databuf.len() {
                            return Err(Error::corrupt_file_named(
                                "variable_full_zip",
                                format!(
                                    "item length {} exceeds the {} byte(s) remaining in the \
                                     page buffer",
                                    length,
                                    databuf.len()
                                ),
                            ));
                        }
                        unzipped_data.extend_from_slice(&databuf[..length as usize]);
                        databuf = &databuf[length as usize..];
                        current_offset += length;
                    } else {
                        // Null items still get an offset
                        match out_bits_per_offset {
                            32 => offsets_data
                                .extend_from_slice(&(current_offset as u32).to_le_bytes()),
                            64 => offsets_data.extend_from_slice(&current_offset.to_le_bytes()),
                            _ => unreachable!(),
                        }
                    }
                }
            }
        }
        self.repdef_starts.push(rep.len().max(def.len()));
        self.data_starts.push(unzipped_data.len());
        self.offset_starts.push(offsets_data.len());
        self.visible_item_counts.push(visible_item_count);
        let expected_starts = (num_rows as usize).checked_add(1).ok_or_else(|| {
            Error::corrupt_file_named(
                "variable_full_zip",
                format!("page row count {num_rows} overflows the row index"),
            )
        })?;
        if self.offset_starts.len() != expected_starts
            || self.repdef_starts.len() != expected_starts
            || self.data_starts.len() != expected_starts
            || self.visible_item_counts.len() != expected_starts
        {
            return Err(Error::corrupt_file_named(
                "variable_full_zip",
                format!(
                    "page declares {num_rows} rows but the zipped data contains {} row starts",
                    self.offset_starts.len().saturating_sub(1)
                ),
            ));
        }
        match out_bits_per_offset {
            32 => offsets_data.extend_from_slice(&(current_offset as u32).to_le_bytes()),
            64 => offsets_data.extend_from_slice(&current_offset.to_le_bytes()),
            _ => unreachable!(),
        };
        self.rep = ScalarBuffer::from(rep);
        self.def = ScalarBuffer::from(def);
        self.data = LanceBuffer::from(unzipped_data);
        self.offsets = LanceBuffer::from(offsets_data);
        Ok(())
    }
}

impl StructuralPageDecoder for VariableFullZipDecoder {
    fn drain(&mut self, num_rows: u64) -> Result<Box<dyn DecodePageTask>> {
        let start = self.current_idx;
        let end = start + num_rows as usize;

        let offset_start = self.offset_starts[start];
        let offset_end = self.offset_starts[end] + (self.bits_per_offset as usize / 8);
        let offsets = self
            .offsets
            .slice_with_length(offset_start, offset_end - offset_start);
        // Keep each batch's variable data buffer bounded to the selected rows.
        let (data, offsets) =
            Self::slice_batch_data_and_rebase_offsets(&self.data, &offsets, self.bits_per_offset)?;

        let repdef_start = self.repdef_starts[start];
        let repdef_end = self.repdef_starts[end];
        let rep = if self.rep.is_empty() {
            self.rep.clone()
        } else {
            self.rep.slice(repdef_start, repdef_end - repdef_start)
        };
        let def = if self.def.is_empty() {
            self.def.clone()
        } else {
            self.def.slice(repdef_start, repdef_end - repdef_start)
        };

        let visible_item_counts_start = self.visible_item_counts[start];
        let visible_item_counts_end = self.visible_item_counts[end];
        let num_visible_items = visible_item_counts_end - visible_item_counts_start;

        self.current_idx += num_rows as usize;

        Ok(Box::new(VariableFullZipDecodeTask {
            details: self.details.clone(),
            decompressor: self.decompressor.clone(),
            data,
            offsets,
            bits_per_offset: self.bits_per_offset,
            num_visible_items,
            rep,
            def,
        }))
    }

    fn num_rows(&self) -> u64 {
        self.num_rows
    }
}

#[derive(Debug)]
struct VariableFullZipDecodeTask {
    details: Arc<FullZipDecodeDetails>,
    decompressor: Arc<dyn VariablePerValueDecompressor>,
    data: LanceBuffer,
    offsets: LanceBuffer,
    bits_per_offset: u8,
    num_visible_items: u64,
    rep: ScalarBuffer<u16>,
    def: ScalarBuffer<u16>,
}

impl DecodePageTask for VariableFullZipDecodeTask {
    fn decode(self: Box<Self>) -> Result<DecodedPage> {
        let block = VariableWidthBlock {
            data: self.data,
            offsets: self.offsets,
            bits_per_offset: self.bits_per_offset,
            num_values: self.num_visible_items,
            block_info: BlockInfo::new(),
        };
        let decomopressed = self.decompressor.decompress(block)?;
        let rep = if self.rep.is_empty() {
            None
        } else {
            Some(self.rep.to_vec())
        };
        let def = if self.def.is_empty() {
            None
        } else {
            Some(self.def.to_vec())
        };
        let unraveler = RepDefUnraveler::new(
            rep,
            def,
            self.details.def_meaning.clone(),
            self.num_visible_items,
        );
        Ok(DecodedPage {
            data: decomopressed,
            repdef: unraveler,
        })
    }
}

#[derive(Debug)]
struct FullZipDecodeTaskItem {
    data: PerValueDataBlock,
    rows_in_buf: u64,
}

/// A task to unzip and decompress full-zip encoded data when that data
/// has a fixed-width.
#[derive(Debug)]
struct FixedFullZipDecodeTask {
    details: Arc<FullZipDecodeDetails>,
    data: Vec<FullZipDecodeTaskItem>,
    num_rows: usize,
    bytes_per_value: usize,
}

impl DecodePageTask for FixedFullZipDecodeTask {
    fn decode(self: Box<Self>) -> Result<DecodedPage> {
        let estimated_size_bytes = if self.details.ctrl_word_parser.bytes_per_word() == 0 {
            let PerValueDecompressor::Fixed(decompressor) = &self.details.value_decompressor else {
                return Err(Error::internal(
                    "FixedFullZipDecodeTask requires a fixed-width decompressor",
                ));
            };
            decompressor
                .decoded_size_bytes(self.num_rows as u64)
                .unwrap_or_else(|| {
                    self.data
                        .iter()
                        .map(|task_item| task_item.data.data_size())
                        .sum::<u64>()
                        * 2
                })
        } else {
            // Rep/def levels can suppress values, so the exact output size is not known
            // until they are decoded. Keep the existing conservative estimate.
            self.data
                .iter()
                .map(|task_item| task_item.data.data_size())
                .sum::<u64>()
                * 2
        };
        let mut data_builder = DataBlockBuilder::with_capacity_estimate(estimated_size_bytes);

        if self.details.ctrl_word_parser.bytes_per_word() == 0 {
            // Fast path, no need to unzip because there is no rep/def
            //
            // We decompress each buffer and add it to our output buffer
            for task_item in self.data.into_iter() {
                let PerValueDataBlock::Fixed(fixed_data) = task_item.data else {
                    unreachable!()
                };
                let PerValueDecompressor::Fixed(decompressor) = &self.details.value_decompressor
                else {
                    unreachable!()
                };
                debug_assert_eq!(fixed_data.num_values, task_item.rows_in_buf);
                let decompressed = decompressor.decompress(fixed_data, task_item.rows_in_buf)?;
                data_builder.append(&decompressed, 0..task_item.rows_in_buf)?;
            }

            let unraveler = RepDefUnraveler::new(
                None,
                None,
                self.details.def_meaning.clone(),
                self.num_rows as u64,
            );

            Ok(DecodedPage {
                data: data_builder.finish(),
                repdef: unraveler,
            })
        } else {
            // Slow path, unzipping needed
            let mut rep = Vec::with_capacity(self.num_rows);
            let mut def = Vec::with_capacity(self.num_rows);
            let mut num_visible_items: u64 = 0;

            for task_item in self.data.into_iter() {
                let PerValueDataBlock::Fixed(fixed_data) = task_item.data else {
                    unreachable!()
                };
                let mut buf_slice = fixed_data.data.as_ref();
                let num_values = fixed_data.num_values as usize;
                // We will be unzipping repdef in to `rep` and `def` and the
                // values into `values` (which contains the compressed values)
                let mut values = Vec::with_capacity(
                    fixed_data.data.len()
                        - (self.details.ctrl_word_parser.bytes_per_word() * num_values),
                );
                let mut visible_items = 0;
                for _ in 0..num_values {
                    // Extract rep/def
                    self.details
                        .ctrl_word_parser
                        .parse(buf_slice, &mut rep, &mut def);
                    buf_slice = &buf_slice[self.details.ctrl_word_parser.bytes_per_word()..];

                    let is_visible = def
                        .last()
                        .map(|d| *d <= self.details.max_visible_def)
                        .unwrap_or(true);
                    if is_visible {
                        // Extract value
                        values.extend_from_slice(buf_slice[..self.bytes_per_value].as_ref());
                        buf_slice = &buf_slice[self.bytes_per_value..];
                        visible_items += 1;
                    }
                }

                // Finally, we decompress the values and add them to our output buffer
                let values_buf = LanceBuffer::from(values);
                let fixed_data = FixedWidthDataBlock {
                    bits_per_value: self.bytes_per_value as u64 * 8,
                    block_info: BlockInfo::new(),
                    data: values_buf,
                    num_values: visible_items,
                };
                let PerValueDecompressor::Fixed(decompressor) = &self.details.value_decompressor
                else {
                    unreachable!()
                };
                let decompressed = decompressor.decompress(fixed_data, visible_items)?;
                data_builder.append(&decompressed, 0..visible_items)?;
                num_visible_items += visible_items;
            }

            let repetition = if rep.is_empty() { None } else { Some(rep) };
            let definition = if def.is_empty() { None } else { Some(def) };

            // Items, not rows: a list row spans any number of items and a null or
            // empty list spans none.
            let unraveler = RepDefUnraveler::new(
                repetition,
                definition,
                self.details.def_meaning.clone(),
                num_visible_items,
            );
            let data = data_builder.finish();

            Ok(DecodedPage {
                data,
                repdef: unraveler,
            })
        }
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::{Array, FixedSizeListArray, Float32Array, UInt8Array, make_array};
    use arrow_schema::{DataType, Field as ArrowField};

    use crate::buffer::LanceBuffer;
    use crate::compression::FixedPerValueDecompressor;
    use crate::constants::{STRUCTURAL_ENCODING_META_KEY, STRUCTURAL_ENCODING_MINIBLOCK};
    use crate::data::{BlockInfo, DataBlock, FixedWidthDataBlock};
    use crate::decoder::DecodePageTask;
    use crate::encodings::logical::primitive::StructuralPageScheduler;
    use crate::encodings::logical::primitive::fullzip::{PerValueCompressor, PerValueDataBlock};
    use crate::encodings::physical::value::{ValueDecompressor, ValueEncoder};
    use crate::format::metadata::compressive_encoding::Compression;
    use crate::testing::{TestCases, check_round_trip_encoding_of_data};

    use super::{
        FixedFullZipDecodeTask, FullZipCacheableState, FullZipDecodeDetails, FullZipDecodeTaskItem,
        FullZipReadSource, FullZipRepIndexDetails, FullZipScheduler, PerValueDecompressor,
        VariableFullZipDecoder,
    };

    fn decode_fixed_fullzip_no_levels(
        decompressor: Arc<dyn FixedPerValueDecompressor>,
        data: Vec<FullZipDecodeTaskItem>,
        num_rows: usize,
        bytes_per_value: usize,
    ) -> DataBlock {
        Box::new(FixedFullZipDecodeTask {
            details: Arc::new(FullZipDecodeDetails {
                value_decompressor: PerValueDecompressor::Fixed(decompressor),
                def_meaning: Arc::from([]),
                ctrl_word_parser: crate::repdef::ControlWordParser::new(0, 0),
                max_rep: 0,
                max_visible_def: u16::MAX,
            }),
            data,
            num_rows,
            bytes_per_value,
        })
        .decode()
        .unwrap()
        .data
    }

    #[test]
    fn test_fixed_fullzip_decode_preallocates_exact_output_size() {
        #[derive(Debug)]
        struct IdentityFixedDecompressor;

        impl FixedPerValueDecompressor for IdentityFixedDecompressor {
            fn decompress(
                &self,
                data: FixedWidthDataBlock,
                num_rows: u64,
            ) -> crate::Result<DataBlock> {
                assert_eq!(data.num_values, num_rows);
                Ok(DataBlock::FixedWidth(data))
            }

            fn bits_per_value(&self) -> u64 {
                32
            }

            fn decoded_size_bytes(&self, num_values: u64) -> Option<u64> {
                num_values.checked_mul(4)
            }
        }

        let make_item = |num_rows: u64| FullZipDecodeTaskItem {
            data: PerValueDataBlock::Fixed(FixedWidthDataBlock {
                data: LanceBuffer::from(vec![7_u8; num_rows as usize * 4]),
                bits_per_value: 32,
                num_values: num_rows,
                block_info: BlockInfo::new(),
            }),
            rows_in_buf: num_rows,
        };

        let num_rows = 512;
        let decoded = decode_fixed_fullzip_no_levels(
            Arc::new(IdentityFixedDecompressor),
            vec![make_item(128), make_item(384)],
            num_rows,
            4,
        );
        let values = decoded.as_fixed_width_ref().unwrap();
        let expected_size = num_rows * 4;
        assert_eq!(values.data.len(), expected_size);
        assert_eq!(values.data.clone().into_buffer().capacity(), expected_size);
    }

    #[test]
    fn test_fixed_fullzip_decode_falls_back_when_output_size_is_not_exact() {
        #[derive(Debug)]
        struct FallbackFixedDecompressor;

        impl FixedPerValueDecompressor for FallbackFixedDecompressor {
            fn decompress(
                &self,
                data: FixedWidthDataBlock,
                num_rows: u64,
            ) -> crate::Result<DataBlock> {
                assert_eq!(data.num_values, num_rows);
                Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
                    data: LanceBuffer::from(vec![7_u8; num_rows as usize * 4]),
                    bits_per_value: 32,
                    num_values: num_rows,
                    block_info: BlockInfo::new(),
                }))
            }

            fn bits_per_value(&self) -> u64 {
                // This deliberately cannot be multiplied by num_rows. If FullZip treats
                // bits_per_value as an exact decoded-size estimate, decoding will fail.
                u64::MAX - 7
            }
        }

        let num_rows = 2;
        let decoded = decode_fixed_fullzip_no_levels(
            Arc::new(FallbackFixedDecompressor),
            vec![FullZipDecodeTaskItem {
                data: PerValueDataBlock::Fixed(FixedWidthDataBlock {
                    data: LanceBuffer::from(vec![0_u8; num_rows * 4]),
                    bits_per_value: 32,
                    num_values: num_rows as u64,
                    block_info: BlockInfo::new(),
                }),
                rows_in_buf: num_rows as u64,
            }],
            num_rows,
            4,
        );
        let values = decoded.as_fixed_width_ref().unwrap();
        assert_eq!(values.num_values, num_rows as u64);
        assert_eq!(values.data.len(), num_rows * 4);
    }

    #[test]
    fn test_fixed_fullzip_real_fsl_preallocates_exact_output_size() {
        let num_rows = 64;
        let dimension = 32;
        let items = Arc::new(Float32Array::from_iter_values(
            (0..num_rows * dimension).map(|value| value as f32),
        ));
        let item_field = Arc::new(ArrowField::new("item", DataType::Float32, false));
        let sample = FixedSizeListArray::new(item_field, dimension as i32, items, None);

        let (data, compression) = PerValueCompressor::compress(
            &ValueEncoder::default(),
            DataBlock::from_array(sample.clone()),
        )
        .unwrap();
        let Compression::FixedSizeList(fsl) = compression.compression.unwrap() else {
            panic!("expected fixed-size-list compression");
        };
        let decompressor = ValueDecompressor::from_fsl(fsl.as_ref()).unwrap();
        let expected_size = num_rows * dimension * size_of::<f32>();
        assert_eq!(
            FixedPerValueDecompressor::decoded_size_bytes(&decompressor, num_rows as u64),
            Some(expected_size as u64)
        );

        let decoded = decode_fixed_fullzip_no_levels(
            Arc::new(decompressor),
            vec![FullZipDecodeTaskItem {
                data,
                rows_in_buf: num_rows as u64,
            }],
            num_rows,
            dimension * size_of::<f32>(),
        );
        let fsl = decoded.as_fixed_size_list_ref().unwrap();
        let values = fsl.child.as_fixed_width_ref().unwrap();
        assert_eq!(values.data.len(), expected_size);
        assert_eq!(values.data.clone().into_buffer().capacity(), expected_size);

        let decoded_array = make_array(
            decoded
                .into_arrow(sample.data_type().clone(), true)
                .unwrap(),
        );
        assert_eq!(decoded_array.as_ref(), &sample);
    }

    #[test]
    fn test_fixed_fullzip_nullable_fsl_uses_fallback_end_to_end() {
        #[derive(Debug)]
        struct NullableFslDecompressor {
            inner: ValueDecompressor,
        }

        impl FixedPerValueDecompressor for NullableFslDecompressor {
            fn decompress(
                &self,
                data: FixedWidthDataBlock,
                num_rows: u64,
            ) -> crate::Result<DataBlock> {
                FixedPerValueDecompressor::decompress(&self.inner, data, num_rows)
            }

            fn bits_per_value(&self) -> u64 {
                // FullZip must not use this physical row width as an exact decoded-size
                // estimate for the nullable, multi-buffer Arrow output.
                u64::MAX - 7
            }

            fn decoded_size_bytes(&self, num_values: u64) -> Option<u64> {
                FixedPerValueDecompressor::decoded_size_bytes(&self.inner, num_values)
            }
        }

        let num_rows = 64;
        let items = Arc::new(UInt8Array::from_iter(
            (0..num_rows).map(|value| (value % 3 != 0).then_some(value as u8)),
        ));
        let item_field = Arc::new(ArrowField::new("item", DataType::UInt8, true));
        let sample = FixedSizeListArray::new(item_field, 1, items, None);

        let (data, compression) = PerValueCompressor::compress(
            &ValueEncoder::default(),
            DataBlock::from_array(sample.clone()),
        )
        .unwrap();
        let Compression::FixedSizeList(fsl) = compression.compression.unwrap() else {
            panic!("expected fixed-size-list compression");
        };
        let decompressor = NullableFslDecompressor {
            inner: ValueDecompressor::from_fsl(fsl.as_ref()).unwrap(),
        };
        assert_eq!(
            FixedPerValueDecompressor::decoded_size_bytes(&decompressor, num_rows as u64),
            None
        );

        let decoded = decode_fixed_fullzip_no_levels(
            Arc::new(decompressor),
            vec![FullZipDecodeTaskItem {
                data,
                rows_in_buf: num_rows as u64,
            }],
            num_rows,
            2,
        );
        let decoded_array = make_array(
            decoded
                .into_arrow(sample.data_type().clone(), true)
                .unwrap(),
        );
        assert_eq!(decoded_array.as_ref(), &sample);
    }

    #[test]
    fn test_slice_batch_data_and_rebase_offsets_u32() {
        let data = LanceBuffer::copy_slice(b"0123456789abcdefghij");
        let offsets = LanceBuffer::reinterpret_vec(vec![6_u32, 8_u32, 8_u32, 12_u32]);

        let (sliced_data, normalized_offsets) =
            VariableFullZipDecoder::slice_batch_data_and_rebase_offsets(&data, &offsets, 32)
                .unwrap();

        assert_eq!(sliced_data.as_ref(), b"6789ab");
        let normalized = normalized_offsets.borrow_to_typed_slice::<u32>();
        assert_eq!(normalized.as_ref(), &[0, 2, 2, 6]);
    }

    #[test]
    fn test_slice_batch_data_and_rebase_offsets_u64() {
        let data = LanceBuffer::copy_slice(b"abcdefghijklmnopqrstuvwxyz");
        let offsets = LanceBuffer::reinterpret_vec(vec![10_u64, 12_u64, 16_u64, 20_u64]);

        let (sliced_data, normalized_offsets) =
            VariableFullZipDecoder::slice_batch_data_and_rebase_offsets(&data, &offsets, 64)
                .unwrap();

        assert_eq!(sliced_data.as_ref(), b"klmnopqrst");
        let normalized = normalized_offsets.borrow_to_typed_slice::<u64>();
        assert_eq!(normalized.as_ref(), &[0, 2, 6, 10]);
    }

    #[test]
    fn test_slice_batch_data_and_rebase_offsets_rejects_invalid_offsets() {
        let data = LanceBuffer::copy_slice(b"abcd");
        let offsets = LanceBuffer::reinterpret_vec(vec![3_u32, 2_u32]);

        let err = VariableFullZipDecoder::slice_batch_data_and_rebase_offsets(&data, &offsets, 32)
            .expect_err("offset end before start should error");
        assert!(err.to_string().contains("less than base"));
    }

    #[tokio::test]
    async fn test_fullzip_initialize_is_lazy() {
        use futures::{FutureExt, future::BoxFuture};
        use std::ops::Range;
        use std::sync::Mutex;

        #[derive(Debug, Clone)]
        struct RecordingScheduler {
            data: bytes::Bytes,
            requests: Arc<Mutex<Vec<Vec<Range<u64>>>>>,
        }

        impl RecordingScheduler {
            fn new(data: bytes::Bytes) -> Self {
                Self {
                    data,
                    requests: Arc::new(Mutex::new(Vec::new())),
                }
            }

            fn requests(&self) -> Vec<Vec<Range<u64>>> {
                self.requests.lock().unwrap().clone()
            }
        }

        impl crate::EncodingsIo for RecordingScheduler {
            fn submit_request(
                &self,
                ranges: Vec<Range<u64>>,
                _priority: u64,
            ) -> BoxFuture<'static, crate::Result<Vec<bytes::Bytes>>> {
                self.requests.lock().unwrap().push(ranges.clone());
                let data = ranges
                    .into_iter()
                    .map(|range| self.data.slice(range.start as usize..range.end as usize))
                    .collect::<Vec<_>>();
                std::future::ready(Ok(data)).boxed()
            }
        }

        #[derive(Debug)]
        struct TestFixedDecompressor;

        impl FixedPerValueDecompressor for TestFixedDecompressor {
            fn decompress(
                &self,
                _data: FixedWidthDataBlock,
                _num_rows: u64,
            ) -> crate::Result<DataBlock> {
                unimplemented!("Test decompressor")
            }

            fn bits_per_value(&self) -> u64 {
                32
            }
        }

        let io = Arc::new(RecordingScheduler::new(bytes::Bytes::from(vec![
            0;
            16 * 1024
        ])));
        let mut scheduler = FullZipScheduler {
            data_buf_position: 0,
            data_buf_size: 4096,
            rep_index: Some(FullZipRepIndexDetails {
                buf_position: 1000,
                bytes_per_value: 4,
            }),
            priority: 0,
            rows_in_page: 100,
            bits_per_offset: 32,
            details: Arc::new(FullZipDecodeDetails {
                value_decompressor: PerValueDecompressor::Fixed(Arc::new(TestFixedDecompressor)),
                def_meaning: Arc::new([crate::repdef::DefinitionInterpretation::NullableItem]),
                ctrl_word_parser: crate::repdef::ControlWordParser::new(0, 1),
                max_rep: 0,
                max_visible_def: 0,
            }),
            cached_state: None,
            enable_cache: false,
        };

        let io_dyn: Arc<dyn crate::EncodingsIo> = io.clone();
        let initialization = scheduler.init_layout().unwrap();
        assert!(
            initialization.required_ranges().next().is_none(),
            "FullZip should not request any ranges when caching is disabled"
        );
        let buffers = initialization.with_buffers(vec![]).unwrap();
        let cached_data = scheduler.init_from_buffers(buffers, &io_dyn).await.unwrap();

        assert!(
            cached_data
                .as_arc_any()
                .downcast_ref::<super::NoCachedPageData>()
                .is_some(),
            "FullZip initialize should not eagerly load repetition index data"
        );
        assert!(scheduler.cached_state.is_none());
        assert!(
            io.requests().is_empty(),
            "FullZip initialize should not issue any I/O"
        );
    }

    #[tokio::test]
    async fn test_fullzip_read_source_slices_prefetched_page() {
        let page_start = 200_u64;
        let page_data = LanceBuffer::copy_slice(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let source = FullZipReadSource::PrefetchedPage {
            base_offset: page_start,
            data: page_data,
        };
        let ranges = vec![
            page_start..(page_start + 3),
            (page_start + 4)..(page_start + 8),
        ];
        let mut data = source.fetch(&ranges, 0).await.unwrap();
        assert_eq!(data.pop_front().unwrap().as_ref(), &[0, 1, 2]);
        assert_eq!(data.pop_front().unwrap().as_ref(), &[4, 5, 6, 7]);
    }

    #[tokio::test]
    async fn test_fullzip_initialize_caches_rep_index_when_enabled() {
        use futures::{FutureExt, future::BoxFuture};
        use std::ops::Range;
        use std::sync::Mutex;

        #[derive(Debug, Clone)]
        struct RecordingScheduler {
            data: bytes::Bytes,
            requests: Arc<Mutex<Vec<Vec<Range<u64>>>>>,
        }

        impl RecordingScheduler {
            fn new(data: bytes::Bytes) -> Self {
                Self {
                    data,
                    requests: Arc::new(Mutex::new(Vec::new())),
                }
            }

            fn requests(&self) -> Vec<Vec<Range<u64>>> {
                self.requests.lock().unwrap().clone()
            }
        }

        impl crate::EncodingsIo for RecordingScheduler {
            fn submit_request(
                &self,
                ranges: Vec<Range<u64>>,
                _priority: u64,
            ) -> BoxFuture<'static, crate::Result<Vec<bytes::Bytes>>> {
                self.requests.lock().unwrap().push(ranges.clone());
                let data = ranges
                    .into_iter()
                    .map(|range| self.data.slice(range.start as usize..range.end as usize))
                    .collect::<Vec<_>>();
                std::future::ready(Ok(data)).boxed()
            }
        }

        #[derive(Debug)]
        struct TestFixedDecompressor;

        impl FixedPerValueDecompressor for TestFixedDecompressor {
            fn decompress(
                &self,
                _data: FixedWidthDataBlock,
                _num_rows: u64,
            ) -> crate::Result<DataBlock> {
                unimplemented!("Test decompressor")
            }

            fn bits_per_value(&self) -> u64 {
                32
            }
        }

        let rows_in_page = 100_u64;
        let bytes_per_value = 4_u64;
        let rep_start = 1000_u64;
        let rep_size = ((rows_in_page + 1) * bytes_per_value) as usize;
        let mut data = vec![0_u8; 16 * 1024];
        data[rep_start as usize..rep_start as usize + rep_size].fill(7);
        let io = Arc::new(RecordingScheduler::new(bytes::Bytes::from(data)));

        let mut scheduler = FullZipScheduler {
            data_buf_position: 0,
            data_buf_size: 4096,
            rep_index: Some(FullZipRepIndexDetails {
                buf_position: rep_start,
                bytes_per_value,
            }),
            priority: 0,
            rows_in_page,
            bits_per_offset: 32,
            details: Arc::new(FullZipDecodeDetails {
                value_decompressor: PerValueDecompressor::Fixed(Arc::new(TestFixedDecompressor)),
                def_meaning: Arc::new([crate::repdef::DefinitionInterpretation::NullableItem]),
                ctrl_word_parser: crate::repdef::ControlWordParser::new(0, 1),
                max_rep: 0,
                max_visible_def: 0,
            }),
            cached_state: None,
            enable_cache: true,
        };

        let io_dyn: Arc<dyn crate::EncodingsIo> = io.clone();
        let initialization = scheduler.init_layout().unwrap();
        let ranges = initialization
            .required_ranges()
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(
            ranges,
            vec![rep_start..(rep_start + (rows_in_page + 1) * bytes_per_value)]
        );
        // The field scheduler fetches the declared ranges (coalesced across all
        // pages) and hands the bytes back to the page to finish initialization.
        let buffers = io_dyn.submit_request(ranges, 0).await.unwrap();
        let buffers = initialization.with_buffers(buffers).unwrap();
        let cached_data = scheduler.init_from_buffers(buffers, &io_dyn).await.unwrap();
        assert!(
            cached_data
                .clone()
                .as_arc_any()
                .downcast_ref::<FullZipCacheableState>()
                .is_some()
        );
        assert!(scheduler.cached_state.is_none());
        scheduler.try_load(&cached_data).unwrap();
        assert!(scheduler.cached_state.is_some());
        assert_eq!(
            io.requests(),
            vec![vec![
                rep_start..(rep_start + (rows_in_page + 1) * bytes_per_value)
            ]]
        );
    }

    #[tokio::test]
    async fn test_fullzip_full_page_bypasses_rep_index_io() {
        use futures::{FutureExt, future::BoxFuture};
        use std::ops::Range;
        use std::sync::Mutex;

        #[derive(Debug, Clone)]
        struct RecordingScheduler {
            data: bytes::Bytes,
            requests: Arc<Mutex<Vec<Vec<Range<u64>>>>>,
        }

        impl RecordingScheduler {
            fn new(data: bytes::Bytes) -> Self {
                Self {
                    data,
                    requests: Arc::new(Mutex::new(Vec::new())),
                }
            }

            fn requests(&self) -> Vec<Vec<Range<u64>>> {
                self.requests.lock().unwrap().clone()
            }
        }

        impl crate::EncodingsIo for RecordingScheduler {
            fn submit_request(
                &self,
                ranges: Vec<Range<u64>>,
                _priority: u64,
            ) -> BoxFuture<'static, crate::Result<Vec<bytes::Bytes>>> {
                self.requests.lock().unwrap().push(ranges.clone());
                let data = ranges
                    .into_iter()
                    .map(|range| self.data.slice(range.start as usize..range.end as usize))
                    .collect::<Vec<_>>();
                std::future::ready(Ok(data)).boxed()
            }
        }

        #[derive(Debug)]
        struct TestFixedDecompressor;

        impl FixedPerValueDecompressor for TestFixedDecompressor {
            fn decompress(
                &self,
                _data: FixedWidthDataBlock,
                _num_rows: u64,
            ) -> crate::Result<DataBlock> {
                unimplemented!("Test decompressor")
            }

            fn bits_per_value(&self) -> u64 {
                32
            }
        }

        let rows_in_page = 100_u64;
        let data_start = 256_u64;
        let data_size = 500_u64;
        let rep_start = 4096_u64;
        let bytes_per_value = 4_u64;

        let mut bytes = vec![0_u8; 16 * 1024];
        for i in 0..=rows_in_page {
            let offset = (i * 5) as u32;
            let pos = rep_start as usize + (i * bytes_per_value) as usize;
            bytes[pos..pos + 4].copy_from_slice(&offset.to_le_bytes());
        }
        let io = Arc::new(RecordingScheduler::new(bytes::Bytes::from(bytes)));

        let scheduler = FullZipScheduler {
            data_buf_position: data_start,
            data_buf_size: data_size,
            rep_index: Some(FullZipRepIndexDetails {
                buf_position: rep_start,
                bytes_per_value,
            }),
            priority: 0,
            rows_in_page,
            bits_per_offset: 32,
            details: Arc::new(FullZipDecodeDetails {
                value_decompressor: PerValueDecompressor::Fixed(Arc::new(TestFixedDecompressor)),
                def_meaning: Arc::new([crate::repdef::DefinitionInterpretation::NullableItem]),
                ctrl_word_parser: crate::repdef::ControlWordParser::new(0, 1),
                max_rep: 0,
                max_visible_def: 0,
            }),
            cached_state: None,
            enable_cache: false,
        };

        let io_dyn: Arc<dyn crate::EncodingsIo> = io.clone();
        let tasks = scheduler
            .schedule_ranges_rep(
                &[0..rows_in_page],
                &io_dyn,
                FullZipRepIndexDetails {
                    buf_position: rep_start,
                    bytes_per_value,
                },
            )
            .unwrap();

        let requests = io.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0], vec![data_start..(data_start + data_size)]);

        let _ = tasks.into_iter().next().unwrap().decoder_fut.await.unwrap();
        let requests_after_await = io.requests();
        assert_eq!(
            requests_after_await.len(),
            1,
            "full page path should not issue rep-index I/O"
        );
    }

    /// This test is used to reproduce fuzz test https://github.com/lancedb/lance/issues/4492
    #[tokio::test]
    async fn test_fuzz_issue_4492_empty_rep_values() {
        use lance_datagen::{RowCount, Seed, array, gen_batch};

        let seed = 1823859942947654717u64;
        let num_rows = 2741usize;

        // Generate the exact same data that caused the failure
        let batch_gen = gen_batch().with_seed(Seed::from(seed));
        let base_generator = array::rand_type(&DataType::FixedSizeBinary(32));
        let list_generator = array::rand_list_any(base_generator, false);

        let batch = batch_gen
            .anon_col(list_generator)
            .into_batch_rows(RowCount::from(num_rows as u64))
            .unwrap();

        let list_array = batch.column(0).clone();

        // Force miniblock encoding
        let mut metadata = HashMap::new();
        metadata.insert(
            STRUCTURAL_ENCODING_META_KEY.to_string(),
            STRUCTURAL_ENCODING_MINIBLOCK.to_string(),
        );

        let test_cases = TestCases::default()
            .with_structural_encodings()
            .with_batch_size(100)
            .with_range(0..num_rows.min(500) as u64)
            .with_indices(vec![0, num_rows as u64 / 2, (num_rows - 1) as u64]);

        check_round_trip_encoding_of_data(vec![list_array], &test_cases, metadata).await
    }

    /// One list of vectors per entry: `None` is a null list and `Some(items)` holds one
    /// vector per item, where `None` is a null vector.  The vectors are wide enough that
    /// the leaf is full-zip encoded.
    fn list_of_vectors(lists: &[Option<Vec<Option<f32>>>]) -> arrow_array::ArrayRef {
        use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};

        const DIMENSION: i32 = 128;
        let mut values = Vec::new();
        let mut vector_validity = Vec::new();
        let mut offsets = vec![0_i32];
        let mut list_validity = Vec::new();
        for list in lists {
            if let Some(items) = list {
                for item in items {
                    values.extend(std::iter::repeat_n(item.unwrap_or(0.0), DIMENSION as usize));
                    vector_validity.push(item.is_some());
                }
            }
            list_validity.push(list.is_some());
            offsets.push(vector_validity.len() as i32);
        }
        let vectors = FixedSizeListArray::new(
            Arc::new(ArrowField::new("item", DataType::Float32, true)),
            DIMENSION,
            Arc::new(Float32Array::from(values)),
            vector_validity
                .contains(&false)
                .then(|| NullBuffer::from(vector_validity)),
        );
        Arc::new(arrow_array::ListArray::new(
            Arc::new(ArrowField::new("item", vectors.data_type().clone(), true)),
            OffsetBuffer::new(ScalarBuffer::from(offsets)),
            Arc::new(vectors),
            Some(NullBuffer::from(list_validity)),
        ))
    }

    /// A read that spans a page whose vectors are all valid and a page holding a null
    /// vector must restore one validity bit per vector, not one per list.
    #[tokio::test]
    async fn test_fullzip_list_read_across_pages_with_mixed_item_validity() {
        let all_valid_page = list_of_vectors(&[
            Some(vec![Some(1.0), Some(2.0), Some(3.0)]),
            None,
            Some(vec![Some(4.0), Some(5.0)]),
        ]);
        let null_vector_page = list_of_vectors(&[
            Some(vec![Some(6.0), None, Some(7.0)]),
            Some(vec![Some(8.0)]),
        ]);
        check_round_trip_encoding_of_data(
            vec![all_valid_page, null_vector_page],
            &TestCases::default()
                .with_structural_encodings()
                // One page per input array
                .with_page_sizes(vec![1])
                .with_indices(vec![2, 3, 4]),
            HashMap::new(),
        )
        .await;
    }

    fn truncated_tail_details() -> std::sync::Arc<super::FullZipDecodeDetails> {
        use crate::compression::VariablePerValueDecompressor;
        use crate::encodings::physical::binary::VariableDecoder;
        use crate::repdef::{ControlWordParser, DefinitionInterpretation};
        use std::sync::Arc;
        Arc::new(super::FullZipDecodeDetails {
            value_decompressor: super::PerValueDecompressor::Variable(Arc::new(
                VariableDecoder::default(),
            )
                as Arc<dyn VariablePerValueDecompressor>),
            def_meaning: vec![DefinitionInterpretation::NullableItem].into(),
            ctrl_word_parser: ControlWordParser::new(0, 0),
            max_rep: 0,
            max_visible_def: 0,
        })
    }

    fn decode_variable_full_zip(
        buf: Vec<u8>,
        bits_per_offset: u8,
    ) -> lance_core::Result<super::VariableFullZipDecoder> {
        use std::collections::VecDeque;
        let mut data = VecDeque::new();
        data.push_back(crate::buffer::LanceBuffer::from(buf));
        super::VariableFullZipDecoder::new(
            truncated_tail_details(),
            data,
            1,
            bits_per_offset,
            bits_per_offset,
        )
    }

    /// A well-formed length prefix decodes without incident, for both widths.
    #[test]
    fn variable_full_zip_wellformed_length_prefix() {
        assert!(decode_variable_full_zip(0u32.to_le_bytes().to_vec(), 32).is_ok());
        assert!(decode_variable_full_zip(0u64.to_le_bytes().to_vec(), 64).is_ok());
    }

    /// A page whose item walk ends with a partial length prefix must surface a
    /// corrupt-file error rather than read past the end of the buffer.
    ///
    /// This asserts the error variant and message rather than merely expecting a
    /// panic: before the length prefix was bounds checked, the read was
    /// `get_unchecked` behind a `debug_assert!`, so a debug build panicked here
    /// (which a `#[should_panic]` test would have accepted as a pass) while a
    /// release build read up to 8 bytes out of a 4 byte allocation.
    #[test]
    fn variable_full_zip_truncated_length_prefix_is_corrupt_file() {
        use lance_core::Error;

        for (bits, buf_len) in [(32u8, 3usize), (64u8, 4usize)] {
            let err = decode_variable_full_zip(vec![0xAA; buf_len], bits)
                .expect_err("a truncated length prefix must not decode");
            assert!(
                matches!(err, Error::CorruptFile { .. }),
                "expected CorruptFile for a {}-bit prefix with {} byte(s), got: {:?}",
                bits,
                buf_len,
                err
            );
            let msg = err.to_string();
            assert!(
                msg.contains("truncated length prefix"),
                "error should say what is wrong, got: {msg}"
            );
        }
    }

    /// Drains `num_rows` from a single-buffer fixed full-zip page built from `buf`.
    ///
    /// `bits_rep` picks the control word width (1 -> one byte, 9 -> two bytes) and each
    /// visible item is a one-word control plus a 4 byte value.
    fn drain_fixed_full_zip(buf: Vec<u8>, bits_rep: u8, num_rows: u64) -> lance_core::Result<()> {
        use crate::compression::FixedPerValueDecompressor;
        use crate::decoder::StructuralPageDecoder;
        use crate::repdef::{ControlWordParser, DefinitionInterpretation};
        use std::collections::VecDeque;
        use std::sync::Arc;

        #[derive(Debug)]
        struct UnusedFixedDecompressor;

        impl FixedPerValueDecompressor for UnusedFixedDecompressor {
            fn decompress(&self, _: FixedWidthDataBlock, _: u64) -> crate::Result<DataBlock> {
                unreachable!("these cases fail while slicing, before any value is decoded")
            }

            fn bits_per_value(&self) -> u64 {
                32
            }
        }

        let details = Arc::new(super::FullZipDecodeDetails {
            value_decompressor: super::PerValueDecompressor::Fixed(Arc::new(
                UnusedFixedDecompressor,
            )),
            def_meaning: vec![DefinitionInterpretation::AllValidList].into(),
            ctrl_word_parser: ControlWordParser::new(bits_rep, 0),
            max_rep: 1,
            max_visible_def: 0,
        });
        let bytes_per_word = bits_rep.div_ceil(8) as usize;

        let mut data = VecDeque::new();
        data.push_back(crate::buffer::LanceBuffer::from(buf));
        let mut decoder = super::FixedFullZipDecoder {
            details,
            data,
            offset_in_current: 0,
            bytes_per_value: 4,
            total_bytes_per_value: 4 + bytes_per_word,
            num_rows,
        };
        decoder.drain(num_rows).map(|_| ())
    }

    /// Two one-item rows, each a control word plus a 4 byte value, slice cleanly.
    #[test]
    fn fixed_full_zip_wellformed_page_slices() {
        assert!(drain_fixed_full_zip(vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 0], 1, 2).is_ok());
    }

    /// The item walk is driven by the page's own control words, so a page that disagrees
    /// with its layout can run off the end of the buffer.  Each way of doing so must
    /// surface a corrupt-file error.
    ///
    /// These assert the error variant and message rather than merely expecting a panic:
    /// before the walk was bounds checked these aborted the process inside
    /// `LanceBuffer::slice_with_length`, which a `#[should_panic]` test would have
    /// accepted as a pass.
    #[rstest::rstest]
    // Second value is one byte short of a full 4 byte value
    #[case::truncated_item(vec![1, 0, 0, 0, 0, 1, 0, 0, 0], 1, 2, "truncated item")]
    // Two byte control words, with a lone trailing byte after the first item
    #[case::truncated_control_word(
        vec![1, 0, 0, 0, 0, 0, 1],
        9,
        2,
        "truncated control word"
    )]
    // A well-formed two row page cannot satisfy a request for three rows
    #[case::page_exhausted(vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 0], 1, 3, "page data ran out")]
    fn fixed_full_zip_malformed_page_is_corrupt_file(
        #[case] buf: Vec<u8>,
        #[case] bits_rep: u8,
        #[case] num_rows: u64,
        #[case] expected_msg: &str,
    ) {
        use lance_core::Error;

        let err = drain_fixed_full_zip(buf, bits_rep, num_rows)
            .expect_err("a malformed page must not slice");
        assert!(
            matches!(err, Error::CorruptFile { .. }),
            "expected CorruptFile, got: {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains(expected_msg),
            "error should say what is wrong, got: {msg}"
        );
    }

    /// A page whose item length prefix decodes to more bytes than remain in the
    /// buffer must surface a corrupt-file error rather than panic slicing the
    /// payload. Such a length means the item walk is reading bytes that are not
    /// a well-formed full-zip stream (a corrupt read or a misaligned walk), and
    /// the panic aborts the process when it crosses an FFI boundary.
    #[test]
    fn variable_full_zip_oversized_item_length_is_corrupt_file() {
        use lance_core::Error;

        for bits in [32u8, 64u8] {
            let prefix_width = bits as usize / 8;
            // Length prefix decodes to u32/u64::MAX with a single payload byte behind it.
            let mut buf = vec![0xFF; prefix_width];
            buf.push(0xAA);
            let err = decode_variable_full_zip(buf, bits)
                .expect_err("an oversized item length must not decode");
            assert!(
                matches!(err, Error::CorruptFile { .. }),
                "expected CorruptFile for an oversized {}-bit length, got: {:?}",
                bits,
                err
            );
            let msg = err.to_string();
            assert!(
                msg.contains("exceeds") && msg.contains("remaining"),
                "error should say what is wrong, got: {msg}"
            );
        }
    }

    #[test]
    fn test_full_zip_scheduler_rejects_bad_rep_index() {
        use crate::compression::DefaultDecompressionStrategy;
        use crate::format::{EncodingUtils, metadata};

        let layout = metadata::FullZipLayout::default();
        // Two buffers: data + repetition index.  With 3 rows the index must be a
        // multiple of 4 entries, so 7 bytes is not a whole number of rows.
        let error = FullZipScheduler::try_new(
            &[(0, 100), (0, 7)],
            0,
            3,
            &layout,
            &DefaultDecompressionStrategy::default(),
        )
        .expect_err("a partial repetition-index width must fail");
        assert!(
            error.to_string().contains("not a whole number"),
            "unexpected error: {error}"
        );

        // 3 bytes per row is not a supported width (only 1, 2, 4 or 8).
        let error = FullZipScheduler::try_new(
            &[(0, 100), (0, 9)],
            0,
            2,
            &layout,
            &DefaultDecompressionStrategy::default(),
        )
        .expect_err("an unsupported repetition-index width must fail");
        assert!(
            error.to_string().contains("unsupported width"),
            "unexpected error: {error}"
        );

        // A page without a data buffer must fail rather than panic on `[0]`.
        let error =
            FullZipScheduler::try_new(&[], 0, 2, &layout, &DefaultDecompressionStrategy::default())
                .expect_err("a page without a data buffer must fail");
        assert!(
            error.to_string().contains("no data buffer"),
            "unexpected error: {error}"
        );

        let layout = metadata::FullZipLayout {
            bits_rep: 200,
            bits_def: 100,
            details: Some(metadata::full_zip_layout::Details::BitsPerValue(32)),
            value_compression: Some(EncodingUtils::flat(32, None)),
            ..Default::default()
        };
        let error = FullZipScheduler::try_new(
            &[(0, 100)],
            0,
            2,
            &layout,
            &DefaultDecompressionStrategy::default(),
        )
        .expect_err("a control word wider than 32 bits must fail");
        assert!(
            error.to_string().contains("at most 16 bits"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn variable_full_zip_row_count_mismatch_is_corrupt_file() {
        use lance_core::Error;

        let error = decode_variable_full_zip(Vec::new(), 32)
            .expect_err("a page with no row starts must not decode");
        assert!(
            matches!(error, Error::CorruptFile { .. }),
            "expected CorruptFile, got: {error:?}"
        );
        assert!(
            error.to_string().contains("row starts"),
            "unexpected error: {error}"
        );
    }
}
