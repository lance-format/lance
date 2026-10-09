// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Basic encodings for variable width data
//!
//! These are not compression but represent the "leaf" encodings for variable length data
//! where we simply match the data with the rules of the structural encoding.
//!
//! These encodings are transparent since we aren't actually doing any compression.  No information
//! is needed in the encoding description.

use arrow_array::OffsetSizeTrait;
use byteorder::{ByteOrder, LittleEndian};
use prost::Message;

use crate::compression::{
    BlockCompressor, BlockDecompressor, MiniBlockDecompressor, VariablePerValueDecompressor,
    create_fixed_width_block_decompressor, infer_fixed_width_block_bits, require_block_payload,
};

use crate::buffer::LanceBuffer;
use crate::compression_config::CompressionFieldParams;
use crate::data::{BlockInfo, DataBlock, FixedWidthDataBlock, VariableWidthBlock};
use crate::encodings::logical::primitive::fullzip::{PerValueCompressor, PerValueDataBlock};
use crate::encodings::logical::primitive::miniblock::{
    MAX_MINIBLOCK_VALUES, MiniBlockChunk, MiniBlockCompressed, MiniBlockCompressionContext,
    MiniBlockCompressor,
};
use crate::encodings::physical::checked_fixed_values;
use crate::format::pb21::CompressiveEncoding;
use crate::format::pb21::compressive_encoding::Compression;
use crate::format::{ProtobufUtils21, pb21};

use lance_core::utils::bit::pad_bytes_to;
use lance_core::{Error, Result};

mod offsets;

use offsets::{BlockCost, OffsetBlockCodec, select_offset_block_codec};

#[derive(Debug)]
pub struct BinaryMiniBlockEncoder {
    minichunk_size: i64,
    generic_offsets: Option<CompressionFieldParams>,
}

impl Default for BinaryMiniBlockEncoder {
    fn default() -> Self {
        Self {
            minichunk_size: *AIM_MINICHUNK_SIZE,
            generic_offsets: None,
        }
    }
}

const DEFAULT_AIM_MINICHUNK_SIZE: i64 = 4 * 1024;

pub static AIM_MINICHUNK_SIZE: std::sync::LazyLock<i64> = std::sync::LazyLock::new(|| {
    std::env::var("LANCE_BINARY_MINIBLOCK_CHUNK_SIZE")
        .unwrap_or_else(|_| DEFAULT_AIM_MINICHUNK_SIZE.to_string())
        .parse::<i64>()
        .unwrap_or(DEFAULT_AIM_MINICHUNK_SIZE)
});

#[derive(Debug, Clone, Copy)]
struct BinaryChunkRange {
    start_offset_index: usize,
    end_offset_index: usize,
}

fn binary_chunk_ranges<N: OffsetSizeTrait>(
    offsets: &[N],
    minichunk_size: i64,
) -> Vec<BinaryChunkRange> {
    let mut ranges = Vec::new();
    let mut start = 0;
    loop {
        let end = search_next_offset_idx(offsets, start, minichunk_size);
        ranges.push(BinaryChunkRange {
            start_offset_index: start,
            end_offset_index: end,
        });
        if end == offsets.len() - 1 {
            return ranges;
        }
        start = end;
    }
}

// Make it to support both u32 and u64
fn chunk_offsets<N: OffsetSizeTrait>(
    offsets: &[N],
    data: &[u8],
    alignment: usize,
    minichunk_size: i64,
) -> (Vec<LanceBuffer>, Vec<MiniBlockChunk>) {
    #[derive(Debug)]
    struct ChunkInfo {
        chunk_start_offset_in_orig_idx: usize,
        chunk_last_offset_in_orig_idx: usize,
        // the bytes in every chunk starts at `chunk.bytes_start_offset`
        bytes_start_offset: usize,
        // every chunk is padded to 8 bytes.
        // we need to interpret every chunk as &[u32] so we need it to padded at least to 4 bytes,
        // this field can actually be eliminated and I can use `num_bytes` in `MiniBlockChunk` to compute
        // the `output_total_bytes`.
        padded_chunk_size: usize,
    }

    let byte_width: usize = N::get_byte_width();
    let mut chunks_info = vec![];
    let mut chunks = vec![];
    let mut last_offset_in_orig_idx = 0;
    loop {
        let this_last_offset_in_orig_idx =
            search_next_offset_idx(offsets, last_offset_in_orig_idx, minichunk_size);

        let num_values_in_this_chunk = this_last_offset_in_orig_idx - last_offset_in_orig_idx;
        let chunk_bytes = offsets[this_last_offset_in_orig_idx] - offsets[last_offset_in_orig_idx];
        let this_chunk_size =
            (num_values_in_this_chunk + 1) * byte_width + chunk_bytes.to_usize().unwrap();

        let padded_chunk_size = this_chunk_size.next_multiple_of(alignment);
        debug_assert!(padded_chunk_size > 0);

        let this_chunk_bytes_start_offset = (num_values_in_this_chunk + 1) * byte_width;
        chunks_info.push(ChunkInfo {
            chunk_start_offset_in_orig_idx: last_offset_in_orig_idx,
            chunk_last_offset_in_orig_idx: this_last_offset_in_orig_idx,
            bytes_start_offset: this_chunk_bytes_start_offset,
            padded_chunk_size,
        });
        chunks.push(MiniBlockChunk {
            log_num_values: if this_last_offset_in_orig_idx == offsets.len() - 1 {
                0
            } else {
                num_values_in_this_chunk.trailing_zeros() as u8
            },
            buffer_sizes: vec![padded_chunk_size as u32],
        });
        if this_last_offset_in_orig_idx == offsets.len() - 1 {
            break;
        }
        last_offset_in_orig_idx = this_last_offset_in_orig_idx;
    }

    let output_total_bytes = chunks_info
        .iter()
        .map(|chunk_info| chunk_info.padded_chunk_size)
        .sum::<usize>();

    let mut output: Vec<u8> = Vec::with_capacity(output_total_bytes);

    for chunk in chunks_info {
        let this_chunk_offsets: Vec<N> = offsets
            [chunk.chunk_start_offset_in_orig_idx..=chunk.chunk_last_offset_in_orig_idx]
            .iter()
            .map(|offset| {
                *offset - offsets[chunk.chunk_start_offset_in_orig_idx]
                    + N::from_usize(chunk.bytes_start_offset).unwrap()
            })
            .collect();

        let this_chunk_offsets = LanceBuffer::reinterpret_vec(this_chunk_offsets);
        output.extend_from_slice(&this_chunk_offsets);

        let start_in_orig = offsets[chunk.chunk_start_offset_in_orig_idx]
            .to_usize()
            .unwrap();
        let end_in_orig = offsets[chunk.chunk_last_offset_in_orig_idx]
            .to_usize()
            .unwrap();
        output.extend_from_slice(&data[start_in_orig..end_in_orig]);

        // pad this chunk to make it align to desired bytes.
        const PAD_BYTE: u8 = 72;
        let pad_len = pad_bytes_to(output.len(), alignment);

        // Compare with usize literal to avoid type mismatch with N
        if pad_len > 0_usize {
            output.extend(std::iter::repeat_n(PAD_BYTE, pad_len));
        }
    }
    (vec![LanceBuffer::reinterpret_vec(output)], chunks)
}

// search for the next offset index to cut the values into a chunk.
// this function incrementally peek the number of values in a chunk,
// each time multiplies the number of values by 2.
// It returns the offset_idx in `offsets` that belongs to this chunk.
fn search_next_offset_idx<N: OffsetSizeTrait>(
    offsets: &[N],
    last_offset_idx: usize,
    minichunk_size: i64,
) -> usize {
    // MiniBlockChunk uses `log_num_values == 0` as a sentinel for the final chunk. This means we
    // must avoid creating 1-value chunks except for the final chunk, even if the configured
    // `minichunk_size` is too small to fit more than one value.
    let remaining_values = offsets.len().saturating_sub(last_offset_idx + 1);
    if remaining_values <= 1 {
        return offsets.len() - 1;
    }

    let mut num_values = 2;
    let mut new_num_values = num_values * 2;
    loop {
        if last_offset_idx + new_num_values >= offsets.len() {
            let existing_bytes = offsets[offsets.len() - 1] - offsets[last_offset_idx];
            // existing bytes plus the new offset size
            let new_size = existing_bytes
                + N::from_usize((offsets.len() - last_offset_idx) * N::get_byte_width()).unwrap();
            if new_size.to_i64().unwrap() <= minichunk_size {
                // case 1: can fit the rest of all data into a miniblock
                return offsets.len() - 1;
            } else {
                // case 2: can only fit the last tried `num_values` into a miniblock
                return last_offset_idx + num_values;
            }
        }
        let existing_bytes = offsets[last_offset_idx + new_num_values] - offsets[last_offset_idx];
        let new_size =
            existing_bytes + N::from_usize((new_num_values + 1) * N::get_byte_width()).unwrap();
        if new_size.to_i64().unwrap() <= minichunk_size {
            if new_num_values * 2 > *MAX_MINIBLOCK_VALUES as usize {
                // hit the max number of values limit
                break;
            }
            num_values = new_num_values;
            new_num_values *= 2;
        } else {
            break;
        }
    }
    last_offset_idx + num_values
}

fn validate_variable_offsets<N: OffsetSizeTrait>(
    offsets: &[N],
    num_values: u64,
    data_len: usize,
) -> Result<()> {
    let expected_offsets = usize::try_from(num_values)
        .ok()
        .and_then(|num_values| num_values.checked_add(1))
        .ok_or_else(|| Error::invalid_input("Variable-width offset count overflows usize"))?;
    if offsets.len() != expected_offsets {
        return Err(Error::invalid_input(format!(
            "Variable-width block has {} offsets, expected {expected_offsets}",
            offsets.len()
        )));
    }
    let mut previous = None;
    for (index, offset) in offsets.iter().enumerate() {
        let offset = offset.to_usize().ok_or_else(|| {
            Error::invalid_input(format!(
                "Variable-width offset at index {index} is negative or does not fit usize"
            ))
        })?;
        if previous.is_some_and(|previous| offset < previous) {
            return Err(Error::invalid_input(format!(
                "Variable-width offsets decrease at index {index}"
            )));
        }
        previous = Some(offset);
    }
    if offsets[0].to_usize() != Some(0) {
        return Err(Error::invalid_input(
            "Variable-width offsets must start at zero",
        ));
    }
    if previous != Some(data_len) {
        return Err(Error::invalid_input(format!(
            "Final variable-width offset {:?} does not equal {data_len} data bytes",
            previous
        )));
    }
    Ok(())
}

fn chunk_value_range<N: OffsetSizeTrait>(
    offsets: &[N],
    range: BinaryChunkRange,
) -> Result<std::ops::Range<usize>> {
    let start = offsets[range.start_offset_index]
        .to_usize()
        .ok_or_else(|| Error::invalid_input("Variable chunk start offset does not fit usize"))?;
    let end = offsets[range.end_offset_index]
        .to_usize()
        .ok_or_else(|| Error::invalid_input("Variable chunk end offset does not fit usize"))?;
    if start > end {
        return Err(Error::invalid_input(
            "Variable chunk offsets are decreasing",
        ));
    }
    Ok(start..end)
}

fn serialized_variable_cost(
    compressed: &MiniBlockCompressed,
    encoding: &CompressiveEncoding,
    context: MiniBlockCompressionContext,
) -> u64 {
    compressed
        .chunks
        .iter()
        .fold(encoding.encoded_len() as u64, |total, chunk| {
            chunk.buffer_sizes.iter().fold(
                total.saturating_add(context.chunk_header_bytes(chunk.buffer_sizes.len() as u64)),
                |total, size| total.saturating_add(u64::from(*size).next_multiple_of(8)),
            )
        })
}

fn build_generic_chunks<N: OffsetSizeTrait>(
    data: &VariableWidthBlock,
    offsets: &[N],
    ranges: &[BinaryChunkRange],
    codec: &OffsetBlockCodec,
) -> Result<(MiniBlockCompressed, CompressiveEncoding)> {
    let bytes_per_offset = (data.bits_per_offset / 8) as usize;
    let mut offset_data = Vec::new();
    let mut value_data = Vec::new();
    let mut chunks = Vec::with_capacity(ranges.len());
    let mut actual_encoding = None;

    for (index, range) in ranges.iter().enumerate() {
        let num_offsets = range.end_offset_index - range.start_offset_index + 1;
        let offset_start = range
            .start_offset_index
            .checked_mul(bytes_per_offset)
            .ok_or_else(|| Error::invalid_input("Offset chunk start overflows usize"))?;
        let offset_bytes = num_offsets
            .checked_mul(bytes_per_offset)
            .ok_or_else(|| Error::invalid_input("Offset chunk size overflows usize"))?;
        let offset_block = FixedWidthDataBlock {
            bits_per_value: data.bits_per_offset as u64,
            data: data.offsets.slice_with_length(offset_start, offset_bytes),
            num_values: num_offsets as u64,
            block_info: BlockInfo::default(),
        };
        let (payload, encoding) = codec.compress(offset_block)?;
        if actual_encoding
            .as_ref()
            .is_some_and(|actual| actual != &encoding)
        {
            return Err(Error::internal(format!(
                "Offset codec produced inconsistent descriptors for chunk {index}"
            )));
        }
        actual_encoding.get_or_insert(encoding);

        let value_range = chunk_value_range(offsets, *range)?;
        let value_bytes = data.data.get(value_range.clone()).ok_or_else(|| {
            Error::invalid_input(format!(
                "Variable chunk range {}..{} exceeds {} bytes",
                value_range.start,
                value_range.end,
                data.data.len()
            ))
        })?;
        let mut buffer_sizes = Vec::with_capacity(1 + usize::from(codec.has_payload()));
        if let Some(payload) = payload {
            buffer_sizes.push(
                u32::try_from(payload.len())
                    .map_err(|_| Error::invalid_input("Offset payload exceeds u32::MAX bytes"))?,
            );
            offset_data.extend_from_slice(&payload);
        }
        buffer_sizes.push(
            u32::try_from(value_bytes.len()).map_err(|_| {
                Error::invalid_input("Variable value payload exceeds u32::MAX bytes")
            })?,
        );
        value_data.extend_from_slice(value_bytes);
        let num_values = range.end_offset_index - range.start_offset_index;
        chunks.push(MiniBlockChunk {
            log_num_values: if index + 1 == ranges.len() {
                0
            } else {
                num_values.trailing_zeros() as u8
            },
            buffer_sizes,
        });
    }

    let mut buffers = Vec::with_capacity(1 + usize::from(codec.has_payload()));
    if codec.has_payload() {
        buffers.push(LanceBuffer::from(offset_data));
    }
    buffers.push(LanceBuffer::from(value_data));
    let offsets_encoding = actual_encoding.ok_or_else(|| {
        Error::internal("Variable-width page did not contain an offset chunk".to_string())
    })?;
    Ok((
        MiniBlockCompressed {
            data: buffers,
            chunks,
            num_values: data.num_values,
        },
        ProtobufUtils21::variable(offsets_encoding, None),
    ))
}

impl BinaryMiniBlockEncoder {
    pub fn new(minichunk_size: Option<i64>) -> Self {
        Self {
            minichunk_size: minichunk_size.unwrap_or(*AIM_MINICHUNK_SIZE),
            generic_offsets: None,
        }
    }

    pub fn with_generic_offsets(
        minichunk_size: Option<i64>,
        field_params: CompressionFieldParams,
    ) -> Self {
        Self {
            minichunk_size: minichunk_size.unwrap_or(*AIM_MINICHUNK_SIZE),
            generic_offsets: Some(field_params),
        }
    }

    // put binary data into chunks, every chunk is less than or equal to `minichunk_size`.
    // In each chunk, offsets are put first then followed by binary bytes data, each chunk is padded to 8 bytes.
    // the offsets in the chunk points to the bytes offset in this chunk.
    fn chunk_data(
        &self,
        data: VariableWidthBlock,
        context: MiniBlockCompressionContext,
    ) -> Result<(MiniBlockCompressed, CompressiveEncoding)> {
        match data.bits_per_offset {
            32 => {
                let offsets_buffer = data.offsets.clone();
                let offsets = offsets_buffer.borrow_to_typed_slice::<i32>();
                self.chunk_typed_data(offsets.as_ref(), data, 4, context)
            }
            64 => {
                let offsets_buffer = data.offsets.clone();
                let offsets = offsets_buffer.borrow_to_typed_slice::<i64>();
                self.chunk_typed_data(offsets.as_ref(), data, 8, context)
            }
            _ => Err(Error::invalid_input(format!(
                "Unsupported bits_per_offset={}",
                data.bits_per_offset
            ))),
        }
    }

    fn chunk_typed_data<N: OffsetSizeTrait>(
        &self,
        offsets: &[N],
        data: VariableWidthBlock,
        legacy_alignment: usize,
        context: MiniBlockCompressionContext,
    ) -> Result<(MiniBlockCompressed, CompressiveEncoding)> {
        validate_variable_offsets(offsets, data.num_values, data.data.len())?;
        let bits_per_offset = data.bits_per_offset as u64;
        let legacy_encoding =
            ProtobufUtils21::variable(ProtobufUtils21::flat(bits_per_offset, None), None);
        let (legacy_buffers, legacy_chunks) =
            chunk_offsets(offsets, &data.data, legacy_alignment, self.minichunk_size);
        let legacy = MiniBlockCompressed {
            data: legacy_buffers,
            chunks: legacy_chunks,
            num_values: data.num_values,
        };
        let Some(field_params) = self
            .generic_offsets
            .as_ref()
            .filter(|_| context.allows_generic_offsets())
        else {
            return Ok((legacy, legacy_encoding));
        };

        let ranges = binary_chunk_ranges(offsets, self.minichunk_size);
        let member_ranges = ranges
            .iter()
            .map(|range| range.start_offset_index..range.end_offset_index + 1)
            .collect::<Vec<_>>();
        let offsets_block = FixedWidthDataBlock {
            bits_per_value: bits_per_offset,
            data: data.offsets.clone(),
            num_values: offsets.len() as u64,
            block_info: BlockInfo::default(),
        };
        let extra_payload_header = context
            .chunk_header_bytes(2)
            .saturating_sub(context.chunk_header_bytes(1));
        let codec = select_offset_block_codec(
            &offsets_block,
            &member_ranges,
            field_params,
            BlockCost::new(extra_payload_header, 8),
        )?;
        if matches!(
            codec.expected_encoding().compression.as_ref(),
            Some(Compression::Flat(_))
        ) {
            return Ok((legacy, legacy_encoding));
        }

        let (generic, generic_encoding) = build_generic_chunks(&data, offsets, &ranges, &codec)?;
        if serialized_variable_cost(&generic, &generic_encoding, context)
            < serialized_variable_cost(&legacy, &legacy_encoding, context)
        {
            Ok((generic, generic_encoding))
        } else {
            Ok((legacy, legacy_encoding))
        }
    }
}

impl MiniBlockCompressor for BinaryMiniBlockEncoder {
    fn compress(
        &self,
        context: MiniBlockCompressionContext,
        data: DataBlock,
    ) -> Result<(MiniBlockCompressed, CompressiveEncoding)> {
        match data {
            DataBlock::VariableWidth(variable_width) => self.chunk_data(variable_width, context),
            _ => Err(Error::invalid_input_source(
                format!(
                    "Cannot compress a data block of type {} with BinaryMiniBlockEncoder",
                    data.name()
                )
                .into(),
            )),
        }
    }
}

#[derive(Debug)]
pub struct BinaryMiniBlockDecompressor {
    layout: BinaryMiniBlockLayout,
}

#[derive(Debug)]
enum BinaryMiniBlockLayout {
    Legacy {
        bits_per_offset: u8,
    },
    Generic {
        bits_per_offset: u8,
        offsets: Box<dyn BlockDecompressor>,
        offsets_have_payload: bool,
    },
}

impl BinaryMiniBlockDecompressor {
    pub fn new(bits_per_offset: u8) -> Result<Self> {
        if !matches!(bits_per_offset, 32 | 64) {
            return Err(Error::invalid_input(format!(
                "Binary mini-block offsets require 32 or 64 bits, got {bits_per_offset}"
            )));
        }
        Ok(Self {
            layout: BinaryMiniBlockLayout::Legacy { bits_per_offset },
        })
    }

    pub fn from_variable(variable: &pb21::Variable) -> Result<Self> {
        if variable.values.is_some() {
            return Err(Error::invalid_input(
                "Binary mini-block Variable encoding cannot contain a values codec",
            ));
        }
        let offsets = variable.offsets.as_deref().ok_or_else(|| {
            Error::invalid_input("Binary mini-block Variable encoding is missing offsets")
        })?;
        let compression = offsets.compression.as_ref().ok_or_else(|| {
            Error::invalid_input("Binary mini-block offsets are missing a compression variant")
        })?;
        if let Compression::Flat(flat) = compression {
            if flat.data.is_some() || !matches!(flat.bits_per_value, 32 | 64) {
                return Err(Error::invalid_input(format!(
                    "Legacy binary mini-block offsets require plain 32 or 64-bit Flat encoding, got {} bits",
                    flat.bits_per_value
                )));
            }
            return Self::new(flat.bits_per_value as u8);
        }

        let bits_per_offset = infer_fixed_width_block_bits(offsets)?;
        let offsets = create_fixed_width_block_decompressor(offsets, bits_per_offset)?;
        let offsets_have_payload = offsets.requires_payload();
        Ok(Self {
            layout: BinaryMiniBlockLayout::Generic {
                bits_per_offset: bits_per_offset as u8,
                offsets,
                offsets_have_payload,
            },
        })
    }
}

fn decode_generic_binary_miniblock(
    bits_per_offset: u8,
    offsets: &dyn BlockDecompressor,
    offsets_have_payload: bool,
    data: Vec<LanceBuffer>,
    num_values: u64,
) -> Result<DataBlock> {
    let expected_buffers = 1 + usize::from(offsets_have_payload);
    if data.len() != expected_buffers {
        return Err(Error::corrupt_file_named(
            "binary mini-block",
            format!(
                "generic chunk has {} buffers, expected {expected_buffers}",
                data.len()
            ),
        ));
    }
    let mut buffers = data.into_iter();
    let offset_payload =
        offsets_have_payload.then(|| buffers.next().expect("buffer count was checked"));
    let values = buffers.next().expect("buffer count was checked");
    let num_offsets = num_values.checked_add(1).ok_or_else(|| {
        Error::corrupt_file_named(
            "binary mini-block",
            format!("cannot decode offsets for {num_values} values"),
        )
    })?;
    let decoded = offsets.decompress(offset_payload, num_offsets)?;
    let DataBlock::FixedWidth(offsets) = decoded else {
        return Err(Error::corrupt_file_named(
            "binary mini-block",
            "generic offset codec did not produce fixed-width offsets",
        ));
    };
    if offsets.bits_per_value != u64::from(bits_per_offset) || offsets.num_values != num_offsets {
        return Err(Error::corrupt_file_named(
            "binary mini-block",
            format!(
                "generic offset codec produced {} {}-bit offsets, expected {num_offsets} {bits_per_offset}-bit offsets",
                offsets.num_values, offsets.bits_per_value
            ),
        ));
    }

    match bits_per_offset {
        32 => {
            let typed = checked_fixed_values::<u32>(&offsets, "Binary mini-block offsets")?;
            let mut previous = 0_u32;
            for (position, &offset) in typed.iter().enumerate() {
                if (position == 0 && offset != 0) || offset < previous || offset > i32::MAX as u32 {
                    return Err(Error::corrupt_file_named(
                        "binary mini-block",
                        format!("invalid 32-bit generic offset {offset} at position {position}"),
                    ));
                }
                previous = offset;
            }
            if previous as usize != values.len() {
                return Err(Error::corrupt_file_named(
                    "binary mini-block",
                    format!(
                        "final generic offset {previous} does not equal the {}-byte value buffer",
                        values.len()
                    ),
                ));
            }
        }
        64 => {
            let typed = checked_fixed_values::<u64>(&offsets, "Binary mini-block offsets")?;
            let mut previous = 0_u64;
            for (position, &offset) in typed.iter().enumerate() {
                if (position == 0 && offset != 0) || offset < previous || offset > i64::MAX as u64 {
                    return Err(Error::corrupt_file_named(
                        "binary mini-block",
                        format!("invalid 64-bit generic offset {offset} at position {position}"),
                    ));
                }
                previous = offset;
            }
            let value_len = u64::try_from(values.len()).map_err(|_| {
                Error::corrupt_file_named(
                    "binary mini-block",
                    "value buffer length does not fit u64",
                )
            })?;
            if previous != value_len {
                return Err(Error::corrupt_file_named(
                    "binary mini-block",
                    format!(
                        "final generic offset {previous} does not equal the {}-byte value buffer",
                        values.len()
                    ),
                ));
            }
        }
        _ => unreachable!("generic offset width was validated during decoder construction"),
    }

    Ok(DataBlock::VariableWidth(VariableWidthBlock {
        data: values,
        offsets: offsets.data,
        bits_per_offset,
        num_values,
        block_info: BlockInfo::new(),
    }))
}

/// Cold path: pinpoint why the chunk-relative offsets of a binary mini-block
/// chunk failed validation.
fn chunk_offset_violation_error<T: Copy + Into<u64>>(offsets: &[T], chunk_len: usize) -> Error {
    let mut previous: u64 = offsets[0].into();
    for (position, &offset) in offsets.iter().enumerate().skip(1) {
        let offset: u64 = offset.into();
        if offset < previous {
            return Error::corrupt_file_named(
                "binary mini-block",
                format!(
                    "value offset at position {position} decreases: {offset} < {previous} \
                     (chunk is {chunk_len} bytes)"
                ),
            );
        }
        previous = offset;
    }
    Error::corrupt_file_named(
        "binary mini-block",
        format!("value offset {previous} is out of bounds for a chunk of {chunk_len} bytes"),
    )
}

impl MiniBlockDecompressor for BinaryMiniBlockDecompressor {
    // decompress a MiniBlock of binary data, the num_values must be less than or equal
    // to the number of values this MiniBlock has, BinaryMiniBlock doesn't store `the number of values`
    // it has so assertion can not be done here and the caller of `decompress` must ensure
    // `num_values` <= number of values in the chunk.
    //
    // The chunk-relative value offsets at the front of the chunk come straight
    // from the file and are used to slice the chunk buffer, so corrupt values
    // must surface as a typed error instead of a panic or an out-of-bounds
    // read.  The monotonicity check rides along the existing rebase loop (the
    // `&=` accumulation keeps it branchless) so validation adds no extra pass.
    fn decompress(&self, data: Vec<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        let bits_per_offset = match &self.layout {
            BinaryMiniBlockLayout::Legacy { bits_per_offset } => *bits_per_offset,
            BinaryMiniBlockLayout::Generic {
                bits_per_offset,
                offsets,
                offsets_have_payload,
            } => {
                return decode_generic_binary_miniblock(
                    *bits_per_offset,
                    offsets.as_ref(),
                    *offsets_have_payload,
                    data,
                    num_values,
                );
            }
        };
        if data.len() != 1 {
            return Err(Error::corrupt_file_named(
                "binary mini-block",
                format!("legacy chunk has {} buffers, expected 1", data.len()),
            ));
        }
        let data = data.into_iter().next().expect("buffer count was checked");

        let bytes_per_offset = bits_per_offset as usize / 8;
        if !data.len().is_multiple_of(bytes_per_offset) {
            return Err(Error::corrupt_file_named(
                "binary mini-block",
                format!(
                    "chunk size {} is not a multiple of the {}-byte offset width",
                    data.len(),
                    bytes_per_offset
                ),
            ));
        }
        let num_offsets = (num_values as usize).checked_add(1).ok_or_else(|| {
            Error::corrupt_file_named(
                "binary mini-block",
                format!("cannot decode {num_values} values from a single chunk"),
            )
        })?;
        if data.len() / bytes_per_offset < num_offsets {
            return Err(Error::corrupt_file_named(
                "binary mini-block",
                format!(
                    "chunk of {} bytes holds {} offsets but decoding {} values requires {}",
                    data.len(),
                    data.len() / bytes_per_offset,
                    num_values,
                    num_offsets
                ),
            ));
        }

        // The value region must start past the offsets being decoded, otherwise
        // the offset table itself aliases into the value bytes.  A lower bound
        // (not equality) because a prefix read of the chunk legitimately leaves
        // unrequested offsets between the requested prefix and the values.
        let min_value_region_start = num_offsets * bytes_per_offset;
        let value_region_overlap_error = |first: u64| {
            Error::corrupt_file_named(
                "binary mini-block",
                format!(
                    "value region starts at offset {first} which overlaps the {num_offsets} \
                     requested offsets ({min_value_region_start} bytes)"
                ),
            )
        };

        if bits_per_offset == 64 {
            let offsets_buffer = data.borrow_to_typed_slice::<u64>();
            let offsets = &offsets_buffer.as_ref()[..num_offsets];

            let first = offsets[0];
            if first < min_value_region_start as u64 {
                return Err(value_region_overlap_error(first));
            }
            let mut previous = first;
            let mut is_monotonic = true;
            let result_offsets = offsets
                .iter()
                .map(|&offset| {
                    is_monotonic &= previous <= offset;
                    previous = offset;
                    offset.wrapping_sub(first)
                })
                .collect::<Vec<u64>>();
            let last = offsets[num_offsets - 1];
            if !is_monotonic || last as usize > data.len() {
                return Err(chunk_offset_violation_error(offsets, data.len()));
            }

            Ok(DataBlock::VariableWidth(VariableWidthBlock {
                data: LanceBuffer::from(data[first as usize..last as usize].to_vec()),
                offsets: LanceBuffer::reinterpret_vec(result_offsets),
                bits_per_offset: 64,
                num_values,
                block_info: BlockInfo::new(),
            }))
        } else {
            let offsets_buffer = data.borrow_to_typed_slice::<u32>();
            let offsets = &offsets_buffer.as_ref()[..num_offsets];

            let first = offsets[0];
            if (first as u64) < min_value_region_start as u64 {
                return Err(value_region_overlap_error(first as u64));
            }
            let mut previous = first;
            let mut is_monotonic = true;
            let result_offsets = offsets
                .iter()
                .map(|&offset| {
                    is_monotonic &= previous <= offset;
                    previous = offset;
                    offset.wrapping_sub(first)
                })
                .collect::<Vec<u32>>();
            let last = offsets[num_offsets - 1];
            if !is_monotonic || last as usize > data.len() {
                return Err(chunk_offset_violation_error(offsets, data.len()));
            }

            Ok(DataBlock::VariableWidth(VariableWidthBlock {
                data: LanceBuffer::from(data[first as usize..last as usize].to_vec()),
                offsets: LanceBuffer::reinterpret_vec(result_offsets),
                bits_per_offset: 32,
                num_values,
                block_info: BlockInfo::new(),
            }))
        }
    }
}

/// Most basic encoding for variable-width data which does no compression at all
/// The DataBlock memory layout looks like below:
///
/// | bits_per_offset           | bytes_start_offset        | offsets data | bytes data |
/// | ------------------------- | ------------------------- | ------------ | ---------- |
/// | <bits_per_offset>/8 bytes | <bits_per_offset>/8 bytes | offsets_len  | data_len   |
///
/// It's used in VariableEncoder and BinaryBlockDecompressor
///
#[derive(Debug, Default)]
pub struct VariableEncoder {}

impl BlockCompressor for VariableEncoder {
    fn compress(&self, mut data: DataBlock) -> Result<(Option<LanceBuffer>, CompressiveEncoding)> {
        let bits_per_offset = match &data {
            DataBlock::VariableWidth(data) => data.bits_per_offset,
            _ => {
                return Err(Error::invalid_input(
                    "BinaryBlockEncoder requires a variable-width block",
                ));
            }
        };
        match data {
            DataBlock::VariableWidth(ref mut variable_width_data) => {
                match variable_width_data.bits_per_offset {
                    32 => {
                        let offsets = variable_width_data.offsets.borrow_to_typed_slice::<u32>();
                        let offsets = offsets.as_ref();
                        // The first 4 bytes store the bits per offset, the next 4 bytes store the start
                        // offset of the bytes data, then offsets data, then bytes data.
                        let bytes_start_offset = 4 + 4 + std::mem::size_of_val(offsets) as u32;

                        let output_total_bytes =
                            bytes_start_offset as usize + variable_width_data.data.len();
                        let mut output: Vec<u8> = Vec::with_capacity(output_total_bytes);

                        // Store bit_per_offset info
                        output.extend_from_slice(&(32_u32).to_le_bytes());

                        // store `bytes_start_offset` in the next 4 bytes of output buffer
                        output.extend_from_slice(&(bytes_start_offset).to_le_bytes());

                        // store offsets
                        output.extend_from_slice(&variable_width_data.offsets);

                        // store bytes
                        output.extend_from_slice(&variable_width_data.data);
                        Ok(LanceBuffer::from(output))
                    }
                    64 => {
                        let offsets = variable_width_data.offsets.borrow_to_typed_slice::<u64>();
                        let offsets = offsets.as_ref();
                        // The first 8 bytes store the bits per offset, the next 8 bytes store the start
                        // offset of the bytes data, then offsets data, then bytes data.
                        let bytes_start_offset = 8 + 8 + std::mem::size_of_val(offsets) as u64;

                        let output_total_bytes =
                            bytes_start_offset as usize + variable_width_data.data.len();
                        let mut output: Vec<u8> = Vec::with_capacity(output_total_bytes);

                        // Store bit_per_offset info
                        output.extend_from_slice(&(64_u64).to_le_bytes());

                        // store `bytes_start_offset` in the next 8 bytes of output buffer
                        output.extend_from_slice(&(bytes_start_offset).to_le_bytes());

                        // store offsets
                        output.extend_from_slice(&variable_width_data.offsets);

                        // store bytes
                        output.extend_from_slice(&variable_width_data.data);
                        Ok(LanceBuffer::from(output))
                    }
                    _ => Err(Error::invalid_input(format!(
                        "BinaryBlockEncoder does not support {}-bit offsets",
                        variable_width_data.bits_per_offset
                    ))),
                }
            }
            _ => unreachable!("variable-width input was validated above"),
        }
        .map(|payload| {
            (
                Some(payload),
                ProtobufUtils21::variable(
                    ProtobufUtils21::flat(bits_per_offset as u64, None),
                    None,
                ),
            )
        })
    }
}

impl PerValueCompressor for VariableEncoder {
    fn compress(&self, data: DataBlock) -> Result<(PerValueDataBlock, CompressiveEncoding)> {
        let DataBlock::VariableWidth(variable) = data else {
            panic!("BinaryPerValueCompressor can only work with Variable Width DataBlock.");
        };

        let encoding = ProtobufUtils21::variable(
            ProtobufUtils21::flat(variable.bits_per_offset as u64, None),
            None,
        );
        Ok((PerValueDataBlock::Variable(variable), encoding))
    }
}

#[derive(Debug, Default)]
pub struct VariableDecoder {}

impl VariablePerValueDecompressor for VariableDecoder {
    fn decompress(&self, data: VariableWidthBlock) -> Result<DataBlock> {
        Ok(DataBlock::VariableWidth(data))
    }
}

#[derive(Debug, Default)]
pub struct BinaryBlockDecompressor {}

impl BlockDecompressor for BinaryBlockDecompressor {
    fn decompress(&self, data: Option<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        let data = require_block_payload(data, "Binary block")?;
        // In older (not quite stable) versions we stored the bits per offset as a single byte and then the num_values
        // as four bytes.  However, this led to alignment problems and was wasteful since we already store the num_values
        // in higher layers.
        //
        // In the standard scheme we use 4 bytes for the bits per offset and 4 bytes for the bytes_start_offset and we
        // rely on the passed in num_values to be correct.

        // This isn't perfect but it's probably good enough and the best I think we can do.  The bits per offset will
        // never be more than 255 and it's little endian so the last 3 bytes will always be 0.  These will be the least
        // significant 3 bytes of the number of values in the old scheme.  It's pretty unlikely these are all 0 (that would
        // mean there are at least 16M values in a single page) so we'll use this to determine if the old scheme is used.
        //
        // The header fields and the offsets themselves come straight from the file.
        // The structural checks below (all O(1)) reject blocks whose regions do not
        // line up; the offset *values* are validated later, by the mandatory layout
        // validation in `VariableWidthBlock::into_arrow`, so they are not rescanned
        // here.
        if data.len() < 4 {
            return Err(Error::corrupt_file_named(
                "variable-width block",
                format!(
                    "block of {} bytes is too small to hold a header",
                    data.len()
                ),
            ));
        }
        let is_old_scheme = data[1] != 0 || data[2] != 0 || data[3] != 0;

        let ensure_header = |header_len: usize| {
            if data.len() < header_len {
                return Err(Error::corrupt_file_named(
                    "variable-width block",
                    format!(
                        "block of {} bytes is too small for a {} byte header",
                        data.len(),
                        header_len
                    ),
                ));
            }
            Ok(())
        };
        let (bits_per_offset, bytes_start_offset, offset_start) = if is_old_scheme {
            // Old scheme
            let bits_per_offset = data[0];
            match bits_per_offset {
                32 => {
                    ensure_header(9)?;
                    debug_assert_eq!(LittleEndian::read_u32(&data[1..5]), num_values as u32);
                    let bytes_start_offset = LittleEndian::read_u32(&data[5..9]);
                    (bits_per_offset, bytes_start_offset as u64, 9_u64)
                }
                64 => {
                    ensure_header(17)?;
                    debug_assert_eq!(LittleEndian::read_u64(&data[1..9]), num_values);
                    let bytes_start_offset = LittleEndian::read_u64(&data[9..17]);
                    (bits_per_offset, bytes_start_offset, 17)
                }
                _ => {
                    return Err(Error::invalid_input_source(
                        format!("Unsupported bits_per_offset={}", bits_per_offset).into(),
                    ));
                }
            }
        } else {
            // Standard scheme
            let bits_per_offset = LittleEndian::read_u32(&data[0..4]) as u8;
            match bits_per_offset {
                32 => {
                    ensure_header(8)?;
                    let bytes_start_offset = LittleEndian::read_u32(&data[4..8]);
                    (bits_per_offset, bytes_start_offset as u64, 8)
                }
                64 => {
                    ensure_header(16)?;
                    let bytes_start_offset = LittleEndian::read_u64(&data[8..16]);
                    (bits_per_offset, bytes_start_offset, 16)
                }
                _ => {
                    return Err(Error::invalid_input_source(
                        format!("Unsupported bits_per_offset={}", bits_per_offset).into(),
                    ));
                }
            }
        };

        // The offsets region sits between the header and `bytes_start_offset`
        // and must hold exactly `num_values + 1` offsets starting at zero.
        let expected_offsets_bytes = num_values
            .checked_add(1)
            .and_then(|num_offsets| num_offsets.checked_mul(bits_per_offset as u64 / 8))
            .ok_or_else(|| {
                Error::corrupt_file_named(
                    "variable-width block",
                    format!("offsets region size overflows for {num_values} values"),
                )
            })?;
        if bytes_start_offset < offset_start || bytes_start_offset > data.len() as u64 {
            return Err(Error::corrupt_file_named(
                "variable-width block",
                format!(
                    "bytes start offset {} is outside the block (header: {} bytes, block: {} bytes)",
                    bytes_start_offset,
                    offset_start,
                    data.len()
                ),
            ));
        }
        if bytes_start_offset - offset_start != expected_offsets_bytes {
            return Err(Error::corrupt_file_named(
                "variable-width block",
                format!(
                    "expected {} offset bytes for {} values but found {}",
                    expected_offsets_bytes,
                    num_values,
                    bytes_start_offset - offset_start
                ),
            ));
        }

        // the next `bytes_start_offset - offset_start` stores the offsets.
        let offsets = data.slice_with_length(
            offset_start as usize,
            (bytes_start_offset - offset_start) as usize,
        );
        let first_offset = match bits_per_offset {
            32 => LittleEndian::read_u32(&offsets[0..4]) as u64,
            _ => LittleEndian::read_u64(&offsets[0..8]),
        };
        if first_offset != 0 {
            return Err(Error::corrupt_file_named(
                "variable-width block",
                format!("first offset must be 0 but found {first_offset}"),
            ));
        }

        // the rest are the binary bytes.
        let data = data.slice_with_length(
            bytes_start_offset as usize,
            data.len() - bytes_start_offset as usize,
        );

        Ok(DataBlock::VariableWidth(VariableWidthBlock {
            data,
            offsets,
            bits_per_offset,
            num_values,
            block_info: BlockInfo::new(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::{BinaryMiniBlockDecompressor, BinaryMiniBlockEncoder};
    use arrow_array::{
        ArrayRef, StringArray,
        builder::{LargeStringBuilder, StringBuilder},
    };
    use arrow_schema::{DataType, Field};
    use lance_core::{Error, Result};

    use crate::{
        buffer::LanceBuffer,
        compression::MiniBlockDecompressor,
        compression_config::CompressionFieldParams,
        constants::{
            COMPRESSION_META_KEY, DICT_DIVISOR_META_KEY, STRUCTURAL_ENCODING_FULLZIP,
            STRUCTURAL_ENCODING_META_KEY, STRUCTURAL_ENCODING_MINIBLOCK,
        },
        data::{BlockInfo, DataBlock, VariableWidthBlock},
        encodings::logical::primitive::miniblock::{
            MiniBlockCompressed, MiniBlockCompressionContext, MiniBlockCompressor,
        },
        format::pb21::compressive_encoding::Compression,
        format::{ProtobufUtils21, pb21, pb21::CompressiveEncoding},
        testing::{TestEncoding, check_specific_random},
    };
    use rstest::rstest;
    use std::{collections::HashMap, sync::Arc, vec};

    use crate::testing::{
        FnArrayGeneratorProvider, TestCases, TestEncoding, check_basic_random_case,
        check_round_trip_encoding_generated, check_round_trip_encoding_of_data,
    };

    fn miniblock_context() -> MiniBlockCompressionContext {
        MiniBlockCompressionContext::new(0, true, true)
    }

    fn variable_block_u32(lengths: &[usize]) -> VariableWidthBlock {
        let mut offsets = Vec::with_capacity(lengths.len() + 1);
        let mut data = Vec::new();
        offsets.push(0_i32);
        for (index, length) in lengths.iter().copied().enumerate() {
            data.extend(std::iter::repeat_n((index % 251) as u8, length));
            offsets.push(i32::try_from(data.len()).unwrap());
        }
        VariableWidthBlock {
            data: LanceBuffer::from(data),
            offsets: LanceBuffer::reinterpret_vec(offsets),
            bits_per_offset: 32,
            num_values: lengths.len() as u64,
            block_info: BlockInfo::default(),
        }
    }

    fn decode_binary_miniblocks(
        compressed: MiniBlockCompressed,
        encoding: &CompressiveEncoding,
    ) -> Result<Vec<VariableWidthBlock>> {
        let Some(Compression::Variable(variable)) = encoding.compression.as_ref() else {
            return Err(Error::invalid_input("expected Variable encoding"));
        };
        let decoder = BinaryMiniBlockDecompressor::from_variable(variable)?;
        let mut buffer_offsets = vec![0_usize; compressed.data.len()];
        let mut values_seen = 0_u64;
        let mut decoded = Vec::with_capacity(compressed.chunks.len());
        for chunk in compressed.chunks {
            let num_values = chunk.num_values(values_seen, compressed.num_values);
            values_seen += num_values;
            let buffers = chunk
                .buffer_sizes
                .iter()
                .zip(compressed.data.iter().zip(&mut buffer_offsets))
                .map(|(size, (buffer, offset))| {
                    let size = *size as usize;
                    let chunk = buffer.slice_with_length(*offset, size);
                    *offset += size;
                    chunk
                })
                .collect();
            let DataBlock::VariableWidth(block) = decoder.decompress(buffers, num_values)? else {
                return Err(Error::internal(
                    "Binary mini-block decoded a non-variable block".to_string(),
                ));
            };
            decoded.push(block);
        }
        Ok(decoded)
    }

    fn assert_decoded_value_lengths(decoded: &[VariableWidthBlock], expected: &[usize]) {
        let actual = decoded
            .iter()
            .flat_map(|block| {
                let offsets = block.offsets.borrow_to_typed_slice::<i32>();
                offsets
                    .windows(2)
                    .map(|pair| (pair[1] - pair[0]) as usize)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn generic_offsets_use_range_across_chunks() {
        let lengths = vec![3_usize; 2_048];
        let encoder = BinaryMiniBlockEncoder::with_generic_offsets(
            Some(256),
            CompressionFieldParams::default(),
        );
        let (compressed, encoding) = encoder
            .compress(
                miniblock_context(),
                DataBlock::VariableWidth(variable_block_u32(&lengths)),
            )
            .unwrap();
        let Some(Compression::Variable(variable)) = encoding.compression.as_ref() else {
            panic!("expected Variable encoding");
        };
        assert!(matches!(
            variable
                .offsets
                .as_deref()
                .and_then(|offsets| offsets.compression.as_ref()),
            Some(Compression::Range(_))
        ));
        assert_eq!(compressed.data.len(), 1);
        assert!(compressed.chunks.len() > 1);
        assert!(
            compressed
                .chunks
                .iter()
                .all(|chunk| chunk.buffer_sizes.len() == 1)
        );
        let decoded = decode_binary_miniblocks(compressed, &encoding).unwrap();
        assert_decoded_value_lengths(&decoded, &lengths);
    }

    #[test]
    fn generic_offsets_use_delta_payload() {
        let lengths = (0..4_096)
            .map(|index| [1_usize, 7, 2, 5][index % 4])
            .collect::<Vec<_>>();
        let encoder = BinaryMiniBlockEncoder::with_generic_offsets(
            Some(1_024),
            CompressionFieldParams::default(),
        );
        let (compressed, encoding) = encoder
            .compress(
                miniblock_context(),
                DataBlock::VariableWidth(variable_block_u32(&lengths)),
            )
            .unwrap();
        let Some(Compression::Variable(variable)) = encoding.compression.as_ref() else {
            panic!("expected Variable encoding");
        };
        assert!(matches!(
            variable
                .offsets
                .as_deref()
                .and_then(|offsets| offsets.compression.as_ref()),
            Some(Compression::Delta(_))
        ));
        assert_eq!(compressed.data.len(), 2);
        assert!(
            compressed
                .chunks
                .iter()
                .all(|chunk| chunk.buffer_sizes.len() == 2)
        );
        let decoded = decode_binary_miniblocks(compressed, &encoding).unwrap();
        assert_decoded_value_lengths(&decoded, &lengths);
    }

    #[test]
    fn generic_offsets_support_constant_and_u64_range() {
        let encoder = BinaryMiniBlockEncoder::with_generic_offsets(
            Some(256),
            CompressionFieldParams::default(),
        );
        let empty_lengths = vec![0_usize; 1_024];
        let (compressed, encoding) = encoder
            .compress(
                miniblock_context(),
                DataBlock::VariableWidth(variable_block_u32(&empty_lengths)),
            )
            .unwrap();
        let Some(Compression::Variable(variable)) = encoding.compression.as_ref() else {
            panic!("expected Variable encoding");
        };
        assert!(matches!(
            variable
                .offsets
                .as_deref()
                .and_then(|offsets| offsets.compression.as_ref()),
            Some(Compression::Constant(_))
        ));
        let decoded = decode_binary_miniblocks(compressed, &encoding).unwrap();
        assert_decoded_value_lengths(&decoded, &empty_lengths);

        let num_values = 512_usize;
        let block = VariableWidthBlock {
            data: LanceBuffer::from(vec![1_u8; num_values * 2]),
            offsets: LanceBuffer::reinterpret_vec(
                (0..=num_values)
                    .map(|index| (index * 2) as i64)
                    .collect::<Vec<_>>(),
            ),
            bits_per_offset: 64,
            num_values: num_values as u64,
            block_info: BlockInfo::default(),
        };
        let (compressed, encoding) = encoder
            .compress(miniblock_context(), DataBlock::VariableWidth(block))
            .unwrap();
        let Some(Compression::Variable(variable)) = encoding.compression.as_ref() else {
            panic!("expected Variable encoding");
        };
        assert!(matches!(
            variable
                .offsets
                .as_deref()
                .and_then(|offsets| offsets.compression.as_ref()),
            Some(Compression::Range(_))
        ));
        let decoded = decode_binary_miniblocks(compressed, &encoding).unwrap();
        assert!(decoded.iter().all(|block| block.bits_per_offset == 64));
    }

    #[test]
    fn binary_miniblock_preserves_legacy_wire_and_fallible_generic_reader() {
        assert!(BinaryMiniBlockDecompressor::new(16).is_err());
        let encoder = BinaryMiniBlockEncoder::new(Some(4_096));
        let (compressed, encoding) = encoder
            .compress(
                miniblock_context(),
                DataBlock::VariableWidth(variable_block_u32(&[3, 3])),
            )
            .unwrap();
        let Some(Compression::Variable(variable)) = encoding.compression.as_ref() else {
            panic!("expected Variable encoding");
        };
        assert!(matches!(
            variable
                .offsets
                .as_deref()
                .and_then(|offsets| offsets.compression.as_ref()),
            Some(Compression::Flat(_))
        ));
        let mut expected = Vec::new();
        expected.extend_from_slice(&12_i32.to_le_bytes());
        expected.extend_from_slice(&15_i32.to_le_bytes());
        expected.extend_from_slice(&18_i32.to_le_bytes());
        expected.extend_from_slice(&[0, 0, 0, 1, 1, 1]);
        expected.extend_from_slice(&[72, 72]);
        assert_eq!(compressed.chunks[0].buffer_sizes, [20]);
        assert_eq!(compressed.data[0].as_ref(), expected);

        let generic_encoder = BinaryMiniBlockEncoder::with_generic_offsets(
            Some(4_096),
            CompressionFieldParams::default(),
        );
        let (_, encoding) = generic_encoder
            .compress(
                miniblock_context(),
                DataBlock::VariableWidth(variable_block_u32(&[1, 7, 2, 5])),
            )
            .unwrap();
        let Some(Compression::Variable(variable)) = encoding.compression.as_ref() else {
            panic!("expected Variable encoding");
        };
        assert!(matches!(
            variable
                .offsets
                .as_deref()
                .and_then(|offsets| offsets.compression.as_ref()),
            Some(Compression::Flat(_))
        ));

        let variable = pb21::Variable {
            offsets: Some(Box::new(ProtobufUtils21::range(32, 0, 3))),
            values: None,
        };
        let decoder = BinaryMiniBlockDecompressor::from_variable(&variable).unwrap();
        let error = decoder
            .decompress(vec![LanceBuffer::empty(), LanceBuffer::empty()], 2)
            .unwrap_err();
        assert!(error.to_string().contains("2 buffers, expected 1"));
        let error = decoder
            .decompress(vec![LanceBuffer::from(vec![0_u8; 5])], 2)
            .unwrap_err();
        assert!(error.to_string().contains("final generic offset 6"));
    }

    #[rstest]
    #[case::range([16_usize; 4], "range")]
    #[case::delta([4_usize, 10, 5, 8], "delta")]
    #[test_log::test(tokio::test)]
    async fn generic_offsets_support_scan_range_take(
        #[case] lengths: [usize; 4],
        #[case] expected_encoding: &str,
    ) {
        let values = StringArray::from_iter_values((0..10_000).map(|index| {
            let len = lengths[index % lengths.len()];
            format!("{index:04x}{}", "x".repeat(len - 4))
        }));
        let metadata = HashMap::from([
            (
                STRUCTURAL_ENCODING_META_KEY.to_string(),
                STRUCTURAL_ENCODING_MINIBLOCK.to_string(),
            ),
            (COMPRESSION_META_KEY.to_string(), "none".to_string()),
            (DICT_DIVISOR_META_KEY.to_string(), "100000".to_string()),
        ]);
        let test_cases = TestCases::basic()
            .with_encoding(TestEncoding::StructuralSparse)
            .with_expected_encoding(expected_encoding);
        check_round_trip_encoding_of_data(vec![Arc::new(values)], &test_cases, metadata).await;
    }

    #[rstest]
    #[test_log::test(tokio::test)]
    async fn test_utf8_binary(
        #[values(
            TestEncoding::StructuralU16,
            TestEncoding::StructuralU32,
            TestEncoding::StructuralSparse
        )]
        encoding: TestEncoding,
        #[values(4096, 1024 * 1024)] page_size: u64,
        #[values(false, true)] use_slicing: bool,
    ) {
        let field = Field::new("", DataType::Utf8, false);
        check_specific_random(
            field,
            TestCases::basic()
                .with_encoding(encoding)
                .with_page_sizes(vec![page_size])
                .with_slicing_modes([use_slicing]),
        )
        .await;
    }

    #[rstest]
    #[test_log::test(tokio::test)]
    async fn test_binary(
        #[values(STRUCTURAL_ENCODING_MINIBLOCK, STRUCTURAL_ENCODING_FULLZIP)]
        structural_encoding: &str,
        #[values(DataType::Utf8, DataType::Binary)] data_type: DataType,
        #[values(
            TestEncoding::Array,
            TestEncoding::StructuralU16,
            TestEncoding::StructuralU32,
            TestEncoding::StructuralSparse
        )]
        encoding: TestEncoding,
        #[values(4096, 1024 * 1024)] page_size: u64,
        #[values(false, true)] use_slicing: bool,
    ) {
        let mut field_metadata = HashMap::new();
        field_metadata.insert(
            STRUCTURAL_ENCODING_META_KEY.to_string(),
            structural_encoding.into(),
        );

        let field = Field::new("", data_type, false).with_metadata(field_metadata);
        check_basic_random_case(field, encoding, page_size, use_slicing).await;
    }

    #[rstest]
    #[test_log::test(tokio::test)]
    async fn test_binary_fsst(
        #[values(STRUCTURAL_ENCODING_MINIBLOCK, STRUCTURAL_ENCODING_FULLZIP)]
        structural_encoding: &str,
        #[values(DataType::Binary, DataType::Utf8)] data_type: DataType,
        #[values(
            TestEncoding::StructuralU16,
            TestEncoding::StructuralU32,
            TestEncoding::StructuralSparse
        )]
        encoding: TestEncoding,
        #[values(4096, 1024 * 1024)] page_size: u64,
        #[values(false, true)] use_slicing: bool,
    ) {
        let mut field_metadata = HashMap::new();
        field_metadata.insert(
            STRUCTURAL_ENCODING_META_KEY.to_string(),
            structural_encoding.into(),
        );
        field_metadata.insert(COMPRESSION_META_KEY.to_string(), "fsst".into());
        let field = Field::new("", data_type, true).with_metadata(field_metadata);
        // TODO (https://github.com/lance-format/lance/issues/4783)
        let test_cases = TestCases::default()
            .with_encoding(encoding)
            .with_page_sizes(vec![page_size])
            .with_slicing_modes([use_slicing]);
        check_specific_random(field, test_cases).await;
    }

    #[rstest]
    #[test_log::test(tokio::test)]
    async fn test_fsst_large_binary(
        #[values(STRUCTURAL_ENCODING_MINIBLOCK, STRUCTURAL_ENCODING_FULLZIP)]
        structural_encoding: &str,
        #[values(DataType::LargeBinary, DataType::LargeUtf8)] data_type: DataType,
        #[values(
            TestEncoding::StructuralU16,
            TestEncoding::StructuralU32,
            TestEncoding::StructuralSparse
        )]
        encoding: TestEncoding,
        #[values(4096, 1024 * 1024)] page_size: u64,
        #[values(false, true)] use_slicing: bool,
    ) {
        let mut field_metadata = HashMap::new();
        field_metadata.insert(
            STRUCTURAL_ENCODING_META_KEY.to_string(),
            structural_encoding.into(),
        );
        field_metadata.insert(COMPRESSION_META_KEY.to_string(), "fsst".into());
        let field = Field::new("", data_type, true).with_metadata(field_metadata);
        check_specific_random(
            field,
            TestCases::basic()
                .with_encoding(encoding)
                .with_page_sizes(vec![page_size])
                .with_slicing_modes([use_slicing]),
        )
        .await;
    }

    #[rstest]
    #[test_log::test(tokio::test)]
    async fn test_large_binary_types(
        #[values(DataType::LargeBinary, DataType::LargeUtf8)] data_type: DataType,
        #[values(
            TestEncoding::Array,
            TestEncoding::StructuralU16,
            TestEncoding::StructuralU32,
            TestEncoding::StructuralSparse
        )]
        encoding: TestEncoding,
        #[values(4096, 1024 * 1024)] page_size: u64,
        #[values(false, true)] use_slicing: bool,
    ) {
        let field = Field::new("", data_type, true);
        check_basic_random_case(field, encoding, page_size, use_slicing).await;
    }

    #[rstest]
    #[test_log::test(tokio::test)]
    async fn test_small_strings(
        #[values(STRUCTURAL_ENCODING_MINIBLOCK, STRUCTURAL_ENCODING_FULLZIP)]
        structural_encoding: &str,
        #[values(
            TestEncoding::Array,
            TestEncoding::StructuralU16,
            TestEncoding::StructuralU32,
            TestEncoding::StructuralSparse
        )]
        encoding: TestEncoding,
        #[values(4096, 1024 * 1024)] page_size: u64,
        #[values(false, true)] use_slicing: bool,
    ) {
        let mut field_metadata = HashMap::new();
        field_metadata.insert(
            STRUCTURAL_ENCODING_META_KEY.to_string(),
            structural_encoding.into(),
        );
        let field = Field::new("", DataType::Utf8, true).with_metadata(field_metadata);
        check_round_trip_encoding_generated(
            field,
            Box::new(FnArrayGeneratorProvider::new(move || {
                lance_datagen::array::utf8_prefix_plus_counter("user_", /*is_large=*/ false)
            })),
            TestCases::basic()
                .with_encoding(encoding)
                .with_page_sizes(vec![page_size])
                .with_slicing_modes([use_slicing]),
        )
        .await;
    }

    #[rstest]
    #[test_log::test(tokio::test)]
    async fn test_simple_binary(
        #[values(STRUCTURAL_ENCODING_MINIBLOCK, STRUCTURAL_ENCODING_FULLZIP)]
        structural_encoding: &str,
        #[values(DataType::Utf8, DataType::Binary)] data_type: DataType,
    ) {
        let string_array = StringArray::from(vec![Some("abc"), None, Some("pqr"), None, Some("m")]);
        let string_array = arrow_cast::cast(&string_array, &data_type).unwrap();

        let mut field_metadata = HashMap::new();
        field_metadata.insert(
            STRUCTURAL_ENCODING_META_KEY.to_string(),
            structural_encoding.into(),
        );

        let test_cases = TestCases::default()
            .with_range(0..2)
            .with_range(0..3)
            .with_range(1..3)
            .with_indices(vec![0, 1, 3, 4]);
        check_round_trip_encoding_of_data(
            vec![Arc::new(string_array)],
            &test_cases,
            field_metadata,
        )
        .await;
    }

    #[test_log::test(tokio::test)]
    async fn test_sliced_utf8() {
        let string_array = StringArray::from(vec![Some("abc"), Some("de"), None, Some("fgh")]);
        let string_array = string_array.slice(1, 3);

        let test_cases = TestCases::default()
            .with_range(0..1)
            .with_range(0..2)
            .with_range(1..2);
        check_round_trip_encoding_of_data(
            vec![Arc::new(string_array)],
            &test_cases,
            HashMap::new(),
        )
        .await;
    }

    #[rstest]
    #[test_log::test(tokio::test)]
    async fn test_value_bigger_than_max_page_size(
        #[values(
            TestEncoding::Array,
            TestEncoding::StructuralU16,
            TestEncoding::StructuralU32,
            TestEncoding::StructuralSparse
        )]
        encoding: TestEncoding,
    ) {
        // Create one value larger than the configured 1MiB page budget.
        let big_string = String::from_iter((0..(2 * 1024 * 1024)).map(|_| '0'));
        let string_array = StringArray::from(vec![
            Some(big_string),
            Some("abc".to_string()),
            None,
            None,
            Some("xyz".to_string()),
        ]);

        // Drop the max page size to 1MiB
        let test_cases = TestCases::default()
            .with_max_page_size(1024 * 1024)
            .with_encoding(encoding);

        check_round_trip_encoding_of_data(
            vec![Arc::new(string_array)],
            &test_cases,
            HashMap::new(),
        )
        .await;
    }

    #[rstest]
    #[test_log::test(tokio::test)]
    async fn test_page_split_parts_do_not_evenly_divide_rows(
        #[values(
            TestEncoding::Array,
            TestEncoding::StructuralU16,
            TestEncoding::StructuralU32,
            TestEncoding::StructuralSparse
        )]
        encoding: TestEncoding,
    ) {
        // Regression: split 90 rows into four parts, where the part count does
        // not evenly divide the row count.
        let big_string = String::from_iter((0..45_000).map(|_| '0'));
        let string_array = StringArray::from_iter_values((0..90).map(|_| big_string.clone()));

        check_round_trip_encoding_of_data(
            vec![Arc::new(string_array)],
            &TestCases::default()
                .with_max_page_size(1024 * 1024)
                .with_encoding(encoding),
            HashMap::new(),
        )
        .await;
    }

    #[test_log::test(tokio::test)]
    async fn test_empty_strings() {
        // Scenario 1: Some strings are empty

        let values = [Some("abc"), Some(""), None];
        // Test empty list at beginning, middle, and end
        for order in [[0, 1, 2], [1, 0, 2], [2, 0, 1]] {
            let mut string_builder = StringBuilder::new();
            for idx in order {
                string_builder.append_option(values[idx]);
            }
            let string_array = Arc::new(string_builder.finish());
            let test_cases = TestCases::default()
                .with_indices(vec![1])
                .with_indices(vec![0])
                .with_indices(vec![2])
                .with_indices(vec![0, 1]);
            check_round_trip_encoding_of_data(
                vec![string_array.clone()],
                &test_cases,
                HashMap::new(),
            )
            .await;
            let test_cases = test_cases.with_batch_size(1);
            check_round_trip_encoding_of_data(vec![string_array], &test_cases, HashMap::new())
                .await;
        }

        // Scenario 2: All strings are empty

        // When encoding an array of empty strings there are no bytes to encode
        // which is strange and we want to ensure we handle it
        let string_array = Arc::new(StringArray::from(vec![Some(""), None, Some("")]));

        let test_cases = TestCases::default().with_range(0..2).with_indices(vec![1]);
        check_round_trip_encoding_of_data(vec![string_array.clone()], &test_cases, HashMap::new())
            .await;
        let test_cases = test_cases.with_batch_size(1);
        check_round_trip_encoding_of_data(vec![string_array], &test_cases, HashMap::new()).await;
    }

    #[test_log::test(tokio::test)]
    #[ignore] // This test is quite slow in debug mode
    async fn test_jumbo_string() {
        // This is an overflow test.  We have a list of lists where each list
        // has 1Mi items.  We encode 5000 of these lists and so we have over 4Gi in the
        // offsets range
        let mut string_builder = LargeStringBuilder::new();
        // a 1 MiB string
        let giant_string = String::from_iter((0..(1024 * 1024)).map(|_| '0'));
        for _ in 0..5000 {
            string_builder.append_option(Some(&giant_string));
        }
        let giant_array = Arc::new(string_builder.finish()) as ArrayRef;
        let arrs = vec![giant_array];

        // // We can't validate because our validation relies on concatenating all input arrays
        let test_cases = TestCases::default().without_validation();
        check_round_trip_encoding_of_data(arrs, &test_cases, HashMap::new()).await;
    }

    #[rstest]
    #[test_log::test(tokio::test)]
    async fn test_binary_dictionary_encoding(
        #[values(true, false)] with_nulls: bool,
        #[values(100, 500, 35000)] dict_size: u32,
    ) {
        let test_cases = TestCases::default().with_structural_encodings();
        let strings = (0..dict_size)
            .map(|i| i.to_string())
            .collect::<Vec<String>>();

        let repeated_strings: Vec<_> = strings
            .iter()
            .cycle()
            .take(70000)
            .enumerate()
            .map(|(i, s)| {
                if with_nulls && i % 7 == 0 {
                    None
                } else {
                    Some(s.clone())
                }
            })
            .collect();
        let string_array = Arc::new(StringArray::from(repeated_strings)) as ArrayRef;
        check_round_trip_encoding_of_data(vec![string_array], &test_cases, HashMap::new()).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_binary_encoding_verification() {
        use lance_datagen::{ByteCount, RowCount};

        let test_cases = TestCases::default()
            .with_expected_encoding("variable")
            .with_structural_encodings();

        // Test both automatic selection and explicit configuration
        // 1. Test automatic binary encoding selection (small strings that won't trigger FSST)
        let arr_small = lance_datagen::gen_batch()
            .anon_col(lance_datagen::array::rand_utf8(ByteCount::from(10), false))
            .into_batch_rows(RowCount::from(1000))
            .unwrap()
            .column(0)
            .clone();
        check_round_trip_encoding_of_data(vec![arr_small], &test_cases, HashMap::new()).await;

        // 2. Test explicit "none" compression to force binary encoding
        let metadata_explicit =
            HashMap::from([("lance-encoding:compression".to_string(), "none".to_string())]);
        let arr_large = lance_datagen::gen_batch()
            .anon_col(lance_datagen::array::rand_utf8(ByteCount::from(50), false))
            .into_batch_rows(RowCount::from(2000))
            .unwrap()
            .column(0)
            .clone();
        check_round_trip_encoding_of_data(vec![arr_large], &test_cases, metadata_explicit).await;
    }

    #[test]
    fn test_binary_miniblock_with_misaligned_buffer() {
        use super::BinaryMiniBlockDecompressor;
        use crate::buffer::LanceBuffer;
        use crate::compression::MiniBlockDecompressor;
        use crate::data::DataBlock;

        // Test case 1: u32 offsets
        {
            let decompressor = BinaryMiniBlockDecompressor::new(32).unwrap();

            // Create test data with u32 offsets
            // BinaryMiniBlock format: all offsets followed by all string data
            // Need to ensure total size is divisible by 4 for u32
            let mut test_data = Vec::new();

            // Offsets section (3 offsets for 2 values + 1 end offset)
            test_data.extend_from_slice(&12u32.to_le_bytes()); // offset to start of strings (after offsets)
            test_data.extend_from_slice(&15u32.to_le_bytes()); // offset to second string
            test_data.extend_from_slice(&20u32.to_le_bytes()); // offset to end

            // String data section
            test_data.extend_from_slice(b"ABCXYZ"); // 6 bytes of string data
            test_data.extend_from_slice(&[0, 0]); // 2 bytes padding to make total 20 bytes (divisible by 4)

            // Create a misaligned buffer by adding padding and slicing
            let mut padded = Vec::with_capacity(test_data.len() + 1);
            padded.push(0xFF); // Padding byte to misalign
            padded.extend_from_slice(&test_data);

            let bytes = bytes::Bytes::from(padded);
            let misaligned = bytes.slice(1..); // Skip first byte to create misalignment

            // Create LanceBuffer with bytes_per_value=1 to bypass alignment check
            let buffer = LanceBuffer::from_bytes(misaligned, 1);

            // Verify the buffer is actually misaligned
            let ptr = buffer.as_ref().as_ptr();
            assert_ne!(
                ptr.align_offset(4),
                0,
                "Test setup: buffer should be misaligned for u32"
            );

            // Decompress with misaligned buffer - should work with borrow_to_typed_slice
            let result = decompressor.decompress(vec![buffer], 2);
            assert!(
                result.is_ok(),
                "Decompression should succeed with misaligned buffer"
            );

            // Verify the data is correct
            if let Ok(DataBlock::VariableWidth(block)) = result {
                assert_eq!(block.num_values, 2);
                // Data should be the strings (including padding from the original buffer)
                assert_eq!(&block.data.as_ref()[..6], b"ABCXYZ");
            } else {
                panic!("Expected VariableWidth block");
            }
        }

        // Test case 2: u64 offsets
        {
            let decompressor = BinaryMiniBlockDecompressor::new(64).unwrap();

            // Create test data with u64 offsets
            let mut test_data = Vec::new();

            // Offsets section (3 offsets for 2 values + 1 end offset)
            test_data.extend_from_slice(&24u64.to_le_bytes()); // offset to start of strings (after offsets)
            test_data.extend_from_slice(&29u64.to_le_bytes()); // offset to second string
            test_data.extend_from_slice(&40u64.to_le_bytes()); // offset to end (divisible by 8)

            // String data section
            test_data.extend_from_slice(b"HelloWorld"); // 10 bytes of string data
            test_data.extend_from_slice(&[0, 0, 0, 0, 0, 0]); // 6 bytes padding to make total 40 bytes (divisible by 8)

            // Create misaligned buffer
            let mut padded = Vec::with_capacity(test_data.len() + 3);
            padded.extend_from_slice(&[0xFF, 0xFF, 0xFF]); // 3 bytes padding for misalignment
            padded.extend_from_slice(&test_data);

            let bytes = bytes::Bytes::from(padded);
            let misaligned = bytes.slice(3..); // Skip 3 bytes

            let buffer = LanceBuffer::from_bytes(misaligned, 1);

            // Verify misalignment for u64
            let ptr = buffer.as_ref().as_ptr();
            assert_ne!(
                ptr.align_offset(8),
                0,
                "Test setup: buffer should be misaligned for u64"
            );

            // Decompress should succeed
            let result = decompressor.decompress(vec![buffer], 2);
            assert!(
                result.is_ok(),
                "Decompression should succeed with misaligned u64 buffer"
            );

            if let Ok(DataBlock::VariableWidth(block)) = result {
                assert_eq!(block.num_values, 2);
                // Data should be the strings (including padding from the original buffer)
                assert_eq!(&block.data.as_ref()[..10], b"HelloWorld");
            } else {
                panic!("Expected VariableWidth block");
            }
        }
    }

    #[test]
    fn test_binary_miniblock_rejects_corrupt_offsets() {
        use super::BinaryMiniBlockDecompressor;
        use crate::compression::MiniBlockDecompressor;
        use lance_core::Error;

        // Chunk layout mirrors the on-disk format for ["alpha", "beta", "gamma"]:
        // LE u32 offsets [16, 21, 25, 30] followed by the value bytes, padded to
        // a multiple of 8 bytes.
        fn chunk_u32(offsets: &[u32], values: &[u8]) -> LanceBuffer {
            let mut chunk = offsets
                .iter()
                .flat_map(|offset| offset.to_le_bytes())
                .collect::<Vec<u8>>();
            chunk.extend_from_slice(values);
            chunk.resize(chunk.len().next_multiple_of(8), 0);
            LanceBuffer::from(chunk)
        }

        let decompressor = BinaryMiniBlockDecompressor::new(32).unwrap();

        // The tail offset points past the end of the 32-byte chunk.
        let err = decompressor
            .decompress(
                vec![chunk_u32(&[16, 21, 25, 100_000], b"alphabetagamma")],
                3,
            )
            .unwrap_err();
        assert!(matches!(err, Error::CorruptFile { .. }), "{err:?}");
        assert!(err.to_string().contains("out of bounds"), "{err}");

        // Offsets go backwards, which would underflow the rebase subtraction.
        let err = decompressor
            .decompress(vec![chunk_u32(&[16, 25, 21, 30], b"alphabetagamma")], 3)
            .unwrap_err();
        assert!(matches!(err, Error::CorruptFile { .. }), "{err:?}");
        assert!(err.to_string().contains("decreases"), "{err}");

        // The first offset points inside the offset table, which would alias
        // the serialized offsets into the value bytes.
        let err = decompressor
            .decompress(vec![chunk_u32(&[0, 21, 25, 30], b"alphabetagamma")], 3)
            .unwrap_err();
        assert!(matches!(err, Error::CorruptFile { .. }), "{err:?}");
        assert!(err.to_string().contains("overlaps"), "{err}");

        // The chunk stores fewer offsets than the requested value count needs.
        let err = decompressor
            .decompress(vec![chunk_u32(&[8, 8], &[])], 3)
            .unwrap_err();
        assert!(matches!(err, Error::CorruptFile { .. }), "{err:?}");
        assert!(err.to_string().contains("requires 4"), "{err}");

        // The chunk size is not a multiple of the offset width.
        let err = decompressor
            .decompress(vec![LanceBuffer::from(vec![0u8; 10])], 1)
            .unwrap_err();
        assert!(matches!(err, Error::CorruptFile { .. }), "{err:?}");
        assert!(err.to_string().contains("multiple"), "{err}");

        // 64-bit offsets take the same validation path.
        fn chunk_u64(offsets: &[u64], values: &[u8]) -> LanceBuffer {
            let mut chunk = offsets
                .iter()
                .flat_map(|offset| offset.to_le_bytes())
                .collect::<Vec<u8>>();
            chunk.extend_from_slice(values);
            chunk.resize(chunk.len().next_multiple_of(8), 0);
            LanceBuffer::from(chunk)
        }
        let decompressor = BinaryMiniBlockDecompressor::new(64).unwrap();
        let err = decompressor
            .decompress(
                vec![chunk_u64(&[32, 37, 41, 100_000], b"alphabetagamma")],
                3,
            )
            .unwrap_err();
        assert!(matches!(err, Error::CorruptFile { .. }), "{err:?}");
        assert!(err.to_string().contains("out of bounds"), "{err}");
        let err = decompressor
            .decompress(vec![chunk_u64(&[0, 37, 41, 46], b"alphabetagamma")], 3)
            .unwrap_err();
        assert!(matches!(err, Error::CorruptFile { .. }), "{err:?}");
        assert!(err.to_string().contains("overlaps"), "{err}");

        // A valid chunk still decodes: offsets rebase to [0, 5, 9, 14].
        let decompressor = BinaryMiniBlockDecompressor::new(32).unwrap();
        let block = decompressor
            .decompress(vec![chunk_u32(&[16, 21, 25, 30], b"alphabetagamma")], 3)
            .unwrap();
        let DataBlock::VariableWidth(block) = block else {
            panic!("expected a variable-width block");
        };
        assert_eq!(block.data.as_ref(), b"alphabetagamma");
        assert_eq!(
            block.offsets,
            LanceBuffer::reinterpret_vec(vec![0_u32, 5, 9, 14])
        );
    }

    fn encoded_binary_block(bits_per_offset: u8) -> Vec<u8> {
        use crate::compression::BlockCompressor;

        let offsets = match bits_per_offset {
            32 => LanceBuffer::reinterpret_vec(vec![0_i32, 5, 9, 14]),
            64 => LanceBuffer::reinterpret_vec(vec![0_i64, 5, 9, 14]),
            _ => unreachable!(),
        };
        let block = DataBlock::VariableWidth(VariableWidthBlock {
            data: LanceBuffer::copy_slice(b"alphabetagamma"),
            offsets,
            bits_per_offset,
            num_values: 3,
            block_info: BlockInfo::new(),
        });
        BlockCompressor::compress(&super::VariableEncoder::default(), block)
            .unwrap()
            .0
            .as_ref()
            .unwrap()
            .to_vec()
    }

    /// The block decompressor only checks the block structure (all O(1)); bad
    /// offset values inside a structurally-sound block are rejected by the
    /// mandatory layout validation when the block is converted to Arrow.
    #[rstest]
    #[case::i32_tail_out_of_bounds(32, 3, 100_000, "out of bounds")]
    #[case::i64_tail_out_of_bounds(64, 3, 15, "out of bounds")]
    #[case::i32_non_monotonic(32, 2, 4, "non-monotonic")]
    #[case::i64_non_monotonic(64, 2, 4, "non-monotonic")]
    fn test_binary_block_bad_offsets_rejected_at_arrow_conversion(
        #[case] bits_per_offset: u8,
        #[case] mutated_offset_index: usize,
        #[case] mutated_offset_value: u64,
        #[case] expected_message: &str,
    ) {
        use crate::compression::BlockDecompressor;
        use lance_core::Error;

        let mut encoded = encoded_binary_block(bits_per_offset);
        let bytes_per_offset = (bits_per_offset / 8) as usize;
        // The standard scheme header is two offset-width fields.
        let mutated_offset_start = bytes_per_offset * (2 + mutated_offset_index);
        encoded[mutated_offset_start..mutated_offset_start + bytes_per_offset]
            .copy_from_slice(&mutated_offset_value.to_le_bytes()[..bytes_per_offset]);

        let block = super::BinaryBlockDecompressor::default()
            .decompress(Some(LanceBuffer::from(encoded)), 3)
            .unwrap();
        let data_type = match bits_per_offset {
            32 => DataType::Binary,
            _ => DataType::LargeBinary,
        };
        let err = block.into_arrow(data_type, false).unwrap_err();
        assert!(matches!(err, Error::CorruptFile { .. }), "{err:?}");
        assert!(err.to_string().contains(expected_message), "{err}");
    }

    #[test]
    fn test_binary_block_rejects_corrupt_structure() {
        use crate::compression::BlockDecompressor;
        use lance_core::Error;

        let decompressor = super::BinaryBlockDecompressor::default();

        // The first offset must be zero.
        let mut encoded = encoded_binary_block(32);
        encoded[8..12].copy_from_slice(&5_u32.to_le_bytes());
        let err = decompressor
            .decompress(Some(LanceBuffer::from(encoded)), 3)
            .unwrap_err();
        assert!(matches!(err, Error::CorruptFile { .. }), "{err:?}");
        assert!(err.to_string().contains("first offset"), "{err}");

        // The offsets region must hold exactly num_values + 1 offsets.
        let encoded = encoded_binary_block(32);
        let err = decompressor
            .decompress(Some(LanceBuffer::from(encoded)), 4)
            .unwrap_err();
        assert!(matches!(err, Error::CorruptFile { .. }), "{err:?}");
        assert!(err.to_string().contains("offset bytes"), "{err}");

        // A block too small to hold its header is rejected, not a panic.
        let err = decompressor
            .decompress(Some(LanceBuffer::from(vec![0_u8; 2])), 1)
            .unwrap_err();
        assert!(matches!(err, Error::CorruptFile { .. }), "{err:?}");
        assert!(err.to_string().contains("too small"), "{err}");
    }
}
