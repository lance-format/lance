// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;
use std::{env, iter, vec};

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{Array, ArrayRef, PrimitiveArray, make_array, new_null_array};
use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder, NullBuffer};
use arrow_schema::DataType;
use futures::FutureExt;
use futures::future::BoxFuture;
use lance_arrow::DataTypeExt;
use lance_arrow::deepcopy::deep_copy_nulls;
use lance_core::Result;
use lance_core::datatypes::Field;
use lance_core::error::Error;
use lance_core::utils::bit::pad_bytes;
use lance_core::utils::tokio::spawn_cpu;
use log::debug;

use crate::buffer::LanceBuffer;
use crate::compression::{CompressionStrategy, compress_required_block};
use crate::constants::{
    COMPRESSION_LEVEL_META_KEY, COMPRESSION_META_KEY, DICT_DIVISOR_META_KEY,
    DICT_SIZE_RATIO_META_KEY, DICT_VALUES_COMPRESSION_ENV_VAR,
    DICT_VALUES_COMPRESSION_LEVEL_ENV_VAR, DICT_VALUES_COMPRESSION_LEVEL_META_KEY,
    DICT_VALUES_COMPRESSION_META_KEY, STRUCTURAL_ENCODING_FULLZIP, STRUCTURAL_ENCODING_META_KEY,
    STRUCTURAL_ENCODING_MINIBLOCK, STRUCTURAL_ENCODING_SPARSE,
};
use crate::data::{BlockInfo, DataBlock, FixedWidthDataBlock, VariableWidthBlock};
use crate::decoder::PageEncoding;
use crate::encoder::{
    EncodeTask, EncodedColumn, EncodedPage, EncodingOptions, FieldEncoder, OutOfLineBuffers,
};
use crate::encodings::logical::primitive::fullzip::PerValueDataBlock;
use crate::encodings::logical::primitive::miniblock::{
    MiniBlockChunk, MiniBlockCompressed, MiniBlockCompressionContext,
};
use crate::format::pb21::CompressiveEncoding;
use crate::format::{ProtobufUtils21, pb21};
use crate::repdef::{
    ControlWordIterator, DefinitionInterpretation, MiniBlockRepDefBudget, NormalizedStructuralPlan,
    RepDefBuilder, RepDefSlicer, RepDefUnraveler, SerializedRepDefs, build_control_word_iterator,
};
use crate::statistics::{ComputeStat, GetStat, Stat};
use crate::utils::accumulation::AccumulationQueue;
use crate::utils::bytepack::BytepackedIntegerEncoder;

use super::{
    DEFAULT_DICT_DIVISOR, DEFAULT_DICT_MAX_CARDINALITY, DEFAULT_DICT_SIZE_RATIO,
    DEFAULT_DICT_VALUES_COMPRESSION, FILL_BYTE, MINIBLOCK_ALIGNMENT,
};
use super::{constant, dict, layout, miniblock, sparse};

/// The serialized representation of full-zip data
struct SerializedFullZip {
    /// The zipped values buffer
    values: LanceBuffer,
    /// The repetition index (only present if there is repetition)
    repetition_index: Option<LanceBuffer>,
}

/// An encoder for primitive (leaf) arrays
///
/// This encoder is fairly complicated and follows a number of paths depending
/// on the data.
///
/// First, we convert the validity & offsets information into repetition and
/// definition levels.  Then we compress the data itself into a single buffer.
///
/// If the data is narrow then we encode the data in small chunks (each chunk
/// should be a few disk sectors and contains a buffer of repetition, a buffer
/// of definition, and a buffer of value data).  This approach is called
/// "mini-block".  These mini-blocks are stored into a single data buffer.
///
/// If the data is wide then we zip together the repetition and definition value
/// with the value data into a single buffer.  This approach is called "zipped".
///
/// If there is any repetition information then we create a repetition index
///
/// In addition, the compression process may create zero or more metadata buffers.
/// For example, a dictionary compression will create dictionary metadata.  Any
/// mini-block approach has a metadata buffer of block sizes.  This metadata is
/// stored in a separate buffer on disk and read at initialization time.
///
/// TODO: We should concatenate metadata buffers from all pages into a single buffer
/// at (roughly) the end of the file so there is, at most, one read per column of
/// metadata per file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiniblockChunkSize {
    U16,
    U32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ComplexNullEncoding {
    RawLevels,
    CompressedLevels,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FixedWidthDictionaryEncoding {
    Exclude64Bit,
    Include64Bit,
}

trait PrimitivePageEncodingBehavior: Send + Sync + Debug {
    fn validate_field(&self, _field: &Field, _metadata: &HashMap<String, String>) -> Result<()> {
        Ok(())
    }

    fn try_plan_pages(
        &self,
        _ctx: &PrimitivePlanContext<'_>,
        _arrays: &[ArrayRef],
        _normalized: &NormalizedStructuralPlan,
        _row_number: u64,
        _num_rows: u64,
        _num_values: u64,
    ) -> Result<Option<Vec<PrimitivePageData>>> {
        Ok(None)
    }

    fn try_encode_page(
        &self,
        _ctx: &PrimitiveEncodeContext,
        page: PrimitivePageData,
    ) -> Result<PrimitiveEncodeAttempt> {
        Ok(PrimitiveEncodeAttempt::Unhandled(page))
    }
}

/// One executable primitive-page behavior selected by an exact file
/// composition.
#[derive(Debug, Clone)]
pub struct PrimitivePageEncoding {
    behavior: Arc<dyn PrimitivePageEncodingBehavior>,
}

impl PrimitivePageEncoding {
    /// Reject an explicit request for sparse structural encoding.
    pub fn reject_sparse() -> Self {
        Self {
            behavior: Arc::new(RejectSparsePrimitiveEncoding),
        }
    }

    /// Encode constant non-null values as a constant page when applicable.
    pub fn constant() -> Self {
        Self {
            behavior: Arc::new(ConstantPrimitiveEncoding),
        }
    }

    /// Plan and encode sparse structural pages when applicable.
    pub fn sparse(compression: Arc<dyn CompressionStrategy>) -> Self {
        Self {
            behavior: Arc::new(SparsePrimitiveEncoding { compression }),
        }
    }

    /// Encode dense pages with the original u16 miniblock grammar.
    pub fn dense_u16(compression: Arc<dyn CompressionStrategy>) -> Self {
        Self {
            behavior: Arc::new(DenseU16PrimitiveEncoding { compression }),
        }
    }

    /// Encode dense pages with the u32 miniblock grammar.
    pub fn dense_u32(compression: Arc<dyn CompressionStrategy>) -> Self {
        Self {
            behavior: Arc::new(DenseU32PrimitiveEncoding { compression }),
        }
    }
}

#[derive(Debug)]
struct RejectSparsePrimitiveEncoding;

#[derive(Debug)]
struct ConstantPrimitiveEncoding;

#[derive(Debug)]
struct SparsePrimitiveEncoding {
    compression: Arc<dyn CompressionStrategy>,
}

#[derive(Debug)]
struct DenseU16PrimitiveEncoding {
    compression: Arc<dyn CompressionStrategy>,
}

#[derive(Debug)]
struct DenseU32PrimitiveEncoding {
    compression: Arc<dyn CompressionStrategy>,
}

pub struct PrimitiveStructuralEncoder {
    // Accumulates arrays until we have enough data to justify a disk page
    accumulation_queue: AccumulationQueue,

    keep_original_array: bool,
    accumulated_repdefs: Vec<RepDefBuilder>,
    page_encodings: Arc<[PrimitivePageEncoding]>,
    column_index: u32,
    field: Field,
    encoding_metadata: Arc<HashMap<String, String>>,
}

struct CompressedLevelsChunk {
    data: LanceBuffer,
    num_levels: u16,
}

struct CompressedLevels {
    data: Vec<CompressedLevelsChunk>,
    compression: CompressiveEncoding,
    rep_index: Option<LanceBuffer>,
}

struct SerializedMiniBlockPage {
    num_buffers: u64,
    data: LanceBuffer,
    metadata: LanceBuffer,
}

#[derive(Debug, Clone, Copy)]
struct DictEncodingBudget {
    max_dict_entries: u32,
    max_encoded_size: usize,
}

enum PrimitivePageStructure {
    Dense {
        repdef: SerializedRepDefs,
        single_row_miniblock_repdef_levels: Option<u64>,
    },
    Sparse {
        plan: sparse::SparseStructuralPlan,
        prepared_values: Option<sparse::writer::PreparedSparseValues>,
    },
}

// A primitive page after structural encoding selection and optional dense splitting.
struct PrimitivePageData {
    // Arrow leaf arrays that contain this page's visible values.
    arrays: Vec<ArrayRef>,
    // Structural representation aligned to this page.
    structure: PrimitivePageStructure,
    // Top-level row number of the first row in this page.
    row_number: u64,
    // Number of top-level rows in this page.
    num_rows: u64,
}

struct PrimitivePlanContext<'a> {
    column_idx: u32,
    field: &'a Field,
    encoding_metadata: &'a HashMap<String, String>,
}

enum PrimitiveEncodeAttempt {
    Encoded(EncodedPage),
    Unhandled(PrimitivePageData),
}

// Immutable encoder state shared by per-page encode tasks.
//
// Cloning this only clones Arc-backed configuration and field metadata.  Page data
// stays in PrimitivePageData and is moved into exactly one task.
#[derive(Clone)]
struct PrimitiveEncodeContext {
    // Column being encoded.
    column_idx: u32,
    field: Field,
    encoding_metadata: Arc<HashMap<String, String>>,
    is_simple_validity: bool,
    has_repdef_info: bool,
}

impl PrimitiveStructuralEncoder {
    pub fn try_new(
        options: &EncodingOptions,
        page_encodings: Arc<[PrimitivePageEncoding]>,
        column_index: u32,
        field: Field,
        encoding_metadata: Arc<HashMap<String, String>>,
    ) -> Result<Self> {
        for page_encoding in page_encodings.iter() {
            page_encoding
                .behavior
                .validate_field(&field, &encoding_metadata)?;
        }
        Ok(Self {
            accumulation_queue: AccumulationQueue::new(
                options.cache_bytes_per_column,
                column_index,
                options.keep_original_array,
            ),
            keep_original_array: options.keep_original_array,
            accumulated_repdefs: Vec::new(),
            column_index,
            page_encodings,
            field,
            encoding_metadata,
        })
    }

    fn encode_page(
        page_encodings: &[PrimitivePageEncoding],
        ctx: &PrimitiveEncodeContext,
        mut page: PrimitivePageData,
    ) -> Result<EncodedPage> {
        for page_encoding in page_encodings {
            match page_encoding.behavior.try_encode_page(ctx, page)? {
                PrimitiveEncodeAttempt::Encoded(page) => return Ok(page),
                PrimitiveEncodeAttempt::Unhandled(unhandled) => page = unhandled,
            }
        }
        Err(Error::invalid_input_source(
            format!(
                "No primitive page encoding atom supports field '{}'",
                ctx.field.name
            )
            .into(),
        ))
    }

    // TODO: This is a heuristic we may need to tune at some point
    //
    // As data gets narrow then the "zipping" process gets too expensive
    //   and we prefer mini-block
    // As data gets wide then the # of values per block shrinks (very wide)
    //   data doesn't even fit in a mini-block and the block overhead gets
    //   too large and we prefer zipped.
    fn is_narrow(data_block: &DataBlock) -> bool {
        const MINIBLOCK_MAX_BYTE_LENGTH_PER_VALUE: u64 = 256;

        if let Some(max_len_array) = data_block.get_stat(Stat::MaxLength) {
            let max_len_array = max_len_array
                .as_any()
                .downcast_ref::<PrimitiveArray<UInt64Type>>()
                .unwrap();
            if max_len_array.value(0) < MINIBLOCK_MAX_BYTE_LENGTH_PER_VALUE {
                return true;
            }
        }
        false
    }

    fn prefers_miniblock(
        data_block: &DataBlock,
        encoding_metadata: &HashMap<String, String>,
    ) -> bool {
        // If the user specifically requested miniblock then use it
        if let Some(user_requested) = encoding_metadata.get(STRUCTURAL_ENCODING_META_KEY) {
            return user_requested.to_lowercase() == STRUCTURAL_ENCODING_MINIBLOCK;
        }
        // Otherwise only use miniblock if it is narrow
        Self::is_narrow(data_block)
    }

    fn prefers_fullzip(encoding_metadata: &HashMap<String, String>) -> bool {
        // Fullzip is the backup option so the only reason we wouldn't use it is if the
        // user specifically requested not to use it (in which case we're probably going
        // to emit an error)
        if let Some(user_requested) = encoding_metadata.get(STRUCTURAL_ENCODING_META_KEY) {
            return user_requested.to_lowercase() == STRUCTURAL_ENCODING_FULLZIP;
        }
        true
    }

    // Converts value data, repetition levels, and definition levels into a single
    // buffer of mini-blocks.  In addition, creates a buffer of mini-block metadata
    // which tells us the size of each block.  Finally, if repetition is present then
    // we also create a buffer for the repetition index.
    //
    // Each chunk is serialized as:
    // | num_bufs (1 byte) | buf_lens (2 bytes per buffer) | P | buf0 | P | buf1 | ... | bufN | P |
    //
    // P - Padding inserted to ensure each buffer is 8-byte aligned and the buffer size is a multiple
    //     of 8 bytes (so that the next chunk is 8-byte aligned).
    //
    // Each block has a u16 word of metadata.  The upper 12 bits contain the
    // # of 8-byte words in the block (if the block does not fill the final word
    // then up to 7 bytes of padding are added).  The lower 4 bits describe the log_2
    // number of values (e.g. if there are 1024 then the lower 4 bits will be
    // 0xA)  All blocks except the last must have power-of-two number of values.
    // This not only makes metadata smaller but it makes decoding easier since
    // batch sizes are typically a power of 2.  4 bits would allow us to express
    // up to 32Ki values.
    //
    // This means blocks can have 1 to 32Ki values and 8 - 32Ki bytes.
    //
    // All metadata words are serialized (as little endian) into a single buffer
    // of metadata values.
    //
    // If there is repetition then we also create a repetition index.  This is a
    // single buffer of integer vectors (stored in row major order).  There is one
    // entry for each chunk.  The size of the vector is based on the depth of random
    // access we want to support.
    //
    // A vector of size 2 is the minimum and will support row-based random access (e.g.
    // "take the 57th row").  A vector of size 3 will support 1 level of nested access
    // (e.g. "take the 3rd item in the 57th row").  A vector of size 4 will support 2
    // levels of nested access and so on.
    //
    // The first number in the vector is the number of top-level rows that complete in
    // the chunk.  The second number is the number of second-level rows that complete
    // after the final top-level row completed (or beginning of the chunk if no top-level
    // row completes in the chunk).  And so on.  The final number in the vector is always
    // the number of leftover items not covered by earlier entries in the vector.
    //
    // Currently we are limited to 0 levels of nested access but that will change in the
    // future.
    //
    // The repetition index and the chunk metadata are read at initialization time and
    // cached in memory.
    fn serialize_miniblocks(
        miniblocks: MiniBlockCompressed,
        rep: Option<Vec<CompressedLevelsChunk>>,
        def: Option<Vec<CompressedLevelsChunk>>,
        miniblock_chunk_size: MiniblockChunkSize,
    ) -> Result<SerializedMiniBlockPage> {
        let bytes_rep = rep
            .as_ref()
            .map(|rep| rep.iter().map(|r| r.data.len()).sum::<usize>())
            .unwrap_or(0);
        let bytes_def = def
            .as_ref()
            .map(|def| def.iter().map(|d| d.data.len()).sum::<usize>())
            .unwrap_or(0);
        let bytes_data = miniblocks.data.iter().map(|d| d.len()).sum::<usize>();
        let mut num_buffers = miniblocks.data.len();
        if rep.is_some() {
            num_buffers += 1;
        }
        if def.is_some() {
            num_buffers += 1;
        }
        // 2 bytes for the length of each buffer and up to 7 bytes of padding per buffer
        let max_extra = 9 * num_buffers;
        let mut data_buffer = Vec::with_capacity(bytes_rep + bytes_def + bytes_data + max_extra);
        let chunk_size_bytes = match miniblock_chunk_size {
            MiniblockChunkSize::U16 => 2,
            MiniblockChunkSize::U32 => 4,
        };
        let mut meta_buffer = Vec::with_capacity(miniblocks.chunks.len() * chunk_size_bytes);

        let mut rep_iter = rep.map(|r| r.into_iter());
        let mut def_iter = def.map(|d| d.into_iter());

        let mut buffer_offsets = vec![0; miniblocks.data.len()];
        for chunk in miniblocks.chunks {
            let start_pos = data_buffer.len();
            // Start of chunk should be aligned
            debug_assert_eq!(start_pos % MINIBLOCK_ALIGNMENT, 0);

            let rep = rep_iter.as_mut().map(|r| r.next().unwrap());
            let def = def_iter.as_mut().map(|d| d.next().unwrap());

            // Write the number of levels, or 0 if there is no rep/def
            let num_levels = rep
                .as_ref()
                .map(|r| r.num_levels)
                .unwrap_or(def.as_ref().map(|d| d.num_levels).unwrap_or(0));
            data_buffer.extend_from_slice(&num_levels.to_le_bytes());

            // Write the buffer lengths
            if let Some(rep) = rep.as_ref() {
                let bytes_rep = u16::try_from(rep.data.len()).map_err(|_| {
                    Error::internal(format!(
                        "Repetition buffer size ({} bytes) too large",
                        rep.data.len()
                    ))
                })?;
                data_buffer.extend_from_slice(&bytes_rep.to_le_bytes());
            }
            if let Some(def) = def.as_ref() {
                let bytes_def = u16::try_from(def.data.len()).map_err(|_| {
                    Error::internal(format!(
                        "Definition buffer size ({} bytes) too large",
                        def.data.len()
                    ))
                })?;
                data_buffer.extend_from_slice(&bytes_def.to_le_bytes());
            }

            if miniblock_chunk_size == MiniblockChunkSize::U32 {
                for &buffer_size in &chunk.buffer_sizes {
                    data_buffer.extend_from_slice(&buffer_size.to_le_bytes());
                }
            } else {
                for &buffer_size in &chunk.buffer_sizes {
                    let buffer_size = u16::try_from(buffer_size).map_err(|_| {
                        Error::internal(format!(
                            "Mini-block buffer size ({} bytes) too large for 16-bit metadata",
                            buffer_size
                        ))
                    })?;
                    data_buffer.extend_from_slice(&buffer_size.to_le_bytes());
                }
            }

            // Pad
            let add_padding = |data_buffer: &mut Vec<u8>| {
                let pad = pad_bytes::<MINIBLOCK_ALIGNMENT>(data_buffer.len());
                data_buffer.extend(iter::repeat_n(FILL_BYTE, pad));
            };
            add_padding(&mut data_buffer);

            // Write the buffers themselves
            if let Some(rep) = rep.as_ref() {
                data_buffer.extend_from_slice(&rep.data);
                add_padding(&mut data_buffer);
            }
            if let Some(def) = def.as_ref() {
                data_buffer.extend_from_slice(&def.data);
                add_padding(&mut data_buffer);
            }
            for (buffer_size, (buffer, buffer_offset)) in chunk
                .buffer_sizes
                .iter()
                .zip(miniblocks.data.iter().zip(buffer_offsets.iter_mut()))
            {
                let start = *buffer_offset;
                let end = start + *buffer_size as usize;
                *buffer_offset += *buffer_size as usize;
                data_buffer.extend_from_slice(&buffer[start..end]);
                add_padding(&mut data_buffer);
            }

            let chunk_bytes = data_buffer.len() - start_pos;
            let max_chunk_size = match miniblock_chunk_size {
                MiniblockChunkSize::U16 => 32 * 1024,
                MiniblockChunkSize::U32 => 1_u64 << 31,
            };
            if chunk_bytes == 0 || chunk_bytes as u64 > max_chunk_size {
                return Err(Error::internal(format!(
                    "Mini-block chunk size {} bytes exceeds the {} byte metadata limit",
                    chunk_bytes, max_chunk_size
                )));
            }
            if chunk_bytes % MINIBLOCK_ALIGNMENT != 0 {
                return Err(Error::internal(format!(
                    "Mini-block chunk size {} bytes is not aligned to {} bytes",
                    chunk_bytes, MINIBLOCK_ALIGNMENT
                )));
            }
            if chunk.log_num_values > 15 {
                return Err(Error::internal(format!(
                    "Mini-block log_num_values {} exceeds the 4-bit metadata limit",
                    chunk.log_num_values
                )));
            }
            // We subtract 1 here from chunk_bytes because we want to be able to express
            // a size of 32KiB and not (32Ki - 8)B which is what we'd get otherwise with
            // 0xFFF
            let divided_bytes = chunk_bytes / MINIBLOCK_ALIGNMENT;
            let divided_bytes_minus_one = (divided_bytes - 1) as u64;

            let metadata = (divided_bytes_minus_one << 4) | chunk.log_num_values as u64;
            if miniblock_chunk_size == MiniblockChunkSize::U32 {
                meta_buffer.extend_from_slice(&(metadata as u32).to_le_bytes());
            } else {
                meta_buffer.extend_from_slice(&(metadata as u16).to_le_bytes());
            }
        }

        let data_buffer = LanceBuffer::from(data_buffer);
        let metadata_buffer = LanceBuffer::from(meta_buffer);

        Ok(SerializedMiniBlockPage {
            num_buffers: miniblocks.data.len() as u64,
            data: data_buffer,
            metadata: metadata_buffer,
        })
    }

    /// Compresses a buffer of levels into chunks
    ///
    /// If these are repetition levels then we also calculate the repetition index here (that
    /// is the third return value)
    fn compress_levels(
        mut levels: RepDefSlicer<'_>,
        num_elements: u64,
        compression_strategy: &dyn CompressionStrategy,
        chunks: &[MiniBlockChunk],
        // This will be 0 if we are compressing def levels
        max_rep: u16,
    ) -> Result<CompressedLevels> {
        let mut rep_index = if max_rep > 0 {
            Vec::with_capacity(chunks.len())
        } else {
            vec![]
        };
        // Make the levels into a FixedWidth data block
        let num_levels = levels.num_levels() as u64;
        let levels_buf = levels.all_levels().clone();

        let mut fixed_width_block = FixedWidthDataBlock {
            data: levels_buf,
            bits_per_value: 16,
            num_values: num_levels,
            block_info: BlockInfo::new(),
        };
        // Compute statistics to enable optimal compression for rep/def levels
        fixed_width_block.compute_stat();

        let levels_block = DataBlock::FixedWidth(fixed_width_block);
        let levels_field = Field::new_arrow("", DataType::UInt16, false)?;
        // Pick a block compressor
        let compressor =
            compression_strategy.create_block_compressor(&levels_field, &levels_block)?;
        let mut compressor_desc = None;
        // Compress blocks of levels (sized according to the chunks)
        let mut level_chunks = Vec::with_capacity(chunks.len());
        let mut values_counter = 0;
        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            let chunk_num_values = chunk.num_values(values_counter, num_elements);
            debug_assert!(chunk_num_values > 0);
            values_counter += chunk_num_values;
            let chunk_levels = if chunk_idx < chunks.len() - 1 {
                levels.slice_next(chunk_num_values as usize)
            } else {
                levels.slice_rest()
            };
            let num_chunk_levels = (chunk_levels.len() / 2) as u64;
            if max_rep > 0 {
                // If max_rep > 0 then we are working with rep levels and we need
                // to calculate the repetition index.  The repetition index for a
                // chunk is currently 2 values (in the future it may be more).
                //
                // The first value is the number of rows that _finish_ in the
                // chunk.
                //
                // The second value is the number of "leftovers" after the last
                // finished row in the chunk.
                let rep_values = chunk_levels.borrow_to_typed_slice::<u16>();
                let rep_values = rep_values.as_ref();

                // We skip 1 here because a max_rep at spot 0 doesn't count as a finished list (we
                // will count it in the previous chunk)
                let mut num_rows = rep_values.iter().skip(1).filter(|v| **v == max_rep).count();
                let num_leftovers = if chunk_idx < chunks.len() - 1 {
                    rep_values
                        .iter()
                        .rev()
                        .position(|v| *v == max_rep)
                        // # of leftovers includes the max_rep spot
                        .map(|pos| pos + 1)
                        .unwrap_or(rep_values.len())
                } else {
                    // Last chunk can't have leftovers
                    0
                };

                if chunk_idx != 0 && rep_values.first() == Some(&max_rep) {
                    // This chunk starts with a new row and so, if we thought we had leftovers
                    // in the previous chunk, we were mistaken
                    // TODO: Can use unchecked here
                    let rep_len = rep_index.len();
                    if rep_index[rep_len - 1] != 0 {
                        // We thought we had leftovers but that was actually a full row
                        rep_index[rep_len - 2] += 1;
                        rep_index[rep_len - 1] = 0;
                    }
                }

                if chunk_idx == chunks.len() - 1 {
                    // The final list
                    num_rows += 1;
                }
                rep_index.push(num_rows as u64);
                rep_index.push(num_leftovers as u64);
            }
            let mut chunk_fixed_width = FixedWidthDataBlock {
                data: chunk_levels,
                bits_per_value: 16,
                num_values: num_chunk_levels,
                block_info: BlockInfo::new(),
            };
            chunk_fixed_width.compute_stat();
            let chunk_levels_block = DataBlock::FixedWidth(chunk_fixed_width);
            let (compressed_levels, chunk_compressor_desc) =
                compressor.compress(chunk_levels_block)?;
            if let Some(compressor_desc) = compressor_desc.as_ref() {
                if compressor_desc != &chunk_compressor_desc {
                    return Err(Error::internal(
                        "Rep/def block compressor changed encoding between chunks".to_string(),
                    ));
                }
            } else {
                compressor_desc = Some(chunk_compressor_desc);
            }
            let compressed_levels = compressed_levels.ok_or_else(|| {
                Error::internal(
                    "Rep/def block compressor selected a metadata-only codec".to_string(),
                )
            })?;
            let num_levels = u16::try_from(num_chunk_levels).map_err(|_| {
                Error::invalid_input_source(
                    format!(
                        "Mini-block cannot encode {} rep/def levels in one chunk. \
                         This usually means a top-level row contains too much nested structure \
                         for the current layout.",
                        num_chunk_levels
                    )
                    .into(),
                )
            })?;
            level_chunks.push(CompressedLevelsChunk {
                data: compressed_levels,
                num_levels,
            });
        }
        debug_assert_eq!(levels.num_levels_remaining(), 0);
        let rep_index = if rep_index.is_empty() {
            None
        } else {
            Some(LanceBuffer::reinterpret_vec(rep_index))
        };
        Ok(CompressedLevels {
            data: level_chunks,
            compression: compressor_desc.ok_or_else(|| {
                Error::internal("Rep/def compression produced no chunks".to_string())
            })?,
            rep_index,
        })
    }

    fn encode_simple_all_null(
        column_idx: u32,
        num_rows: u64,
        row_number: u64,
    ) -> Result<EncodedPage> {
        let description =
            ProtobufUtils21::constant_layout(&[DefinitionInterpretation::NullableItem], None);
        Ok(EncodedPage {
            column_idx,
            data: vec![],
            description: PageEncoding::Structural(description),
            num_rows,
            row_number,
        })
    }

    fn encode_complex_all_null_vals(
        data: &Arc<[u16]>,
        compression_strategy: &dyn CompressionStrategy,
    ) -> Result<(LanceBuffer, pb21::CompressiveEncoding)> {
        let buffer = LanceBuffer::reinterpret_slice(data.clone());
        let mut fixed_width_block = FixedWidthDataBlock {
            data: buffer,
            bits_per_value: 16,
            num_values: data.len() as u64,
            block_info: BlockInfo::new(),
        };
        fixed_width_block.compute_stat();

        let levels_block = DataBlock::FixedWidth(fixed_width_block);
        let levels_field = Field::new_arrow("", DataType::UInt16, false)?;
        let (compressed_buffer, encoding) =
            compress_required_block(compression_strategy, &levels_field, levels_block)?;
        Ok((compressed_buffer, encoding))
    }

    // Encodes a page where all values are null but we have rep/def
    // information that we need to store (e.g. to distinguish between
    // different kinds of null)
    fn encode_complex_all_null(
        column_idx: u32,
        repdef: crate::repdef::SerializedRepDefs,
        row_number: u64,
        num_rows: u64,
        complex_null_encoding: ComplexNullEncoding,
        compression_strategy: &dyn CompressionStrategy,
    ) -> Result<EncodedPage> {
        if complex_null_encoding == ComplexNullEncoding::RawLevels {
            let rep_bytes = if let Some(rep) = repdef.repetition_levels.as_ref() {
                LanceBuffer::reinterpret_slice(rep.clone())
            } else {
                LanceBuffer::empty()
            };

            let def_bytes = if let Some(def) = repdef.definition_levels.as_ref() {
                LanceBuffer::reinterpret_slice(def.clone())
            } else {
                LanceBuffer::empty()
            };

            let description = ProtobufUtils21::constant_layout(&repdef.def_meaning, None);
            return Ok(EncodedPage {
                column_idx,
                data: vec![rep_bytes, def_bytes],
                description: PageEncoding::Structural(description),
                num_rows,
                row_number,
            });
        }

        let (rep_bytes, rep_encoding, num_rep_values) = if let Some(rep) =
            repdef.repetition_levels.as_ref()
        {
            let num_values = rep.len() as u64;
            let (buffer, encoding) = Self::encode_complex_all_null_vals(rep, compression_strategy)?;
            (buffer, Some(encoding), num_values)
        } else {
            (LanceBuffer::empty(), None, 0)
        };

        let (def_bytes, def_encoding, num_def_values) = if let Some(def) =
            repdef.definition_levels.as_ref()
        {
            let num_values = def.len() as u64;
            let (buffer, encoding) = Self::encode_complex_all_null_vals(def, compression_strategy)?;
            (buffer, Some(encoding), num_values)
        } else {
            (LanceBuffer::empty(), None, 0)
        };

        let description = ProtobufUtils21::compressed_all_null_constant_layout(
            &repdef.def_meaning,
            rep_encoding,
            def_encoding,
            num_rep_values,
            num_def_values,
        );
        Ok(EncodedPage {
            column_idx,
            data: vec![rep_bytes, def_bytes],
            description: PageEncoding::Structural(description),
            num_rows,
            row_number,
        })
    }

    fn leaf_validity(
        repdef: &crate::repdef::SerializedRepDefs,
        num_values: usize,
    ) -> Result<Option<BooleanBuffer>> {
        let rep = repdef
            .repetition_levels
            .as_ref()
            .map(|rep| rep.as_ref().to_vec());
        let def = repdef
            .definition_levels
            .as_ref()
            .map(|def| def.as_ref().to_vec());
        let mut unraveler = RepDefUnraveler::new(
            rep,
            def,
            repdef.def_meaning.clone().into(),
            num_values as u64,
        );
        if unraveler.is_all_valid() {
            return Ok(None);
        }
        let mut validity = BooleanBufferBuilder::new(num_values);
        unraveler.unravel_validity(&mut validity)?;
        Ok(Some(validity.finish()))
    }

    fn is_constant_values(
        arrays: &[ArrayRef],
        scalar: &ArrayRef,
        validity: Option<&BooleanBuffer>,
    ) -> Result<bool> {
        debug_assert_eq!(scalar.len(), 1);
        debug_assert_eq!(scalar.null_count(), 0);

        match scalar.data_type() {
            DataType::Boolean => {
                let mut global_idx = 0usize;
                let scalar_val = scalar.as_boolean().value(0);
                for arr in arrays {
                    let bool_arr = arr.as_boolean();
                    for i in 0..arr.len() {
                        let is_valid = validity.map(|v| v.value(global_idx)).unwrap_or(true);
                        global_idx += 1;
                        if !is_valid {
                            continue;
                        }
                        if bool_arr.value(i) != scalar_val {
                            return Ok(false);
                        }
                    }
                }
                Ok(true)
            }
            DataType::Utf8 => Self::is_constant_utf8::<i32>(arrays, scalar, validity),
            DataType::LargeUtf8 => Self::is_constant_utf8::<i64>(arrays, scalar, validity),
            DataType::Binary => Self::is_constant_binary::<i32>(arrays, scalar, validity),
            DataType::LargeBinary => Self::is_constant_binary::<i64>(arrays, scalar, validity),
            data_type => {
                let mut global_idx = 0usize;
                let Some(byte_width) = data_type.byte_width_opt() else {
                    return Ok(false);
                };
                let scalar_data = scalar.to_data();
                if scalar_data.buffers().len() != 1 || !scalar_data.child_data().is_empty() {
                    return Ok(false);
                }
                let scalar_bytes = scalar_data.buffers()[0].as_slice();
                if scalar_bytes.len() != byte_width {
                    return Ok(false);
                }

                for arr in arrays {
                    let data = arr.to_data();
                    if data.buffers().is_empty() {
                        return Ok(false);
                    }
                    let buf = data.buffers()[0].as_slice();
                    let base = data.offset();
                    for i in 0..arr.len() {
                        let is_valid = validity.map(|v| v.value(global_idx)).unwrap_or(true);
                        global_idx += 1;
                        if !is_valid {
                            continue;
                        }
                        let start = (base + i) * byte_width;
                        if buf[start..start + byte_width] != scalar_bytes[..] {
                            return Ok(false);
                        }
                    }
                }
                Ok(true)
            }
        }
    }

    fn is_constant_utf8<O: arrow_array::OffsetSizeTrait>(
        arrays: &[ArrayRef],
        scalar: &ArrayRef,
        validity: Option<&BooleanBuffer>,
    ) -> Result<bool> {
        debug_assert_eq!(scalar.len(), 1);
        let scalar_val = scalar.as_string::<O>().value(0).as_bytes();
        let mut global_idx = 0usize;
        for arr in arrays {
            let str_arr = arr.as_string::<O>();
            for i in 0..arr.len() {
                let is_valid = validity.map(|v| v.value(global_idx)).unwrap_or(true);
                global_idx += 1;
                if !is_valid {
                    continue;
                }
                if str_arr.value(i).as_bytes() != scalar_val {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    fn is_constant_binary<O: arrow_array::OffsetSizeTrait>(
        arrays: &[ArrayRef],
        scalar: &ArrayRef,
        validity: Option<&BooleanBuffer>,
    ) -> Result<bool> {
        debug_assert_eq!(scalar.len(), 1);
        let scalar_val = scalar.as_binary::<O>().value(0);
        let mut global_idx = 0usize;
        for arr in arrays {
            let bin_arr = arr.as_binary::<O>();
            for i in 0..arr.len() {
                let is_valid = validity.map(|v| v.value(global_idx)).unwrap_or(true);
                global_idx += 1;
                if !is_valid {
                    continue;
                }
                if bin_arr.value(i) != scalar_val {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    fn find_constant_scalar(
        arrays: &[ArrayRef],
        validity: Option<&BooleanBuffer>,
    ) -> Result<Option<ArrayRef>> {
        if arrays.is_empty() {
            return Ok(None);
        }

        let global_scalar_idx = if let Some(validity) = validity {
            let Some(idx) = (0..validity.len()).find(|&i| validity.value(i)) else {
                return Ok(None);
            };
            idx
        } else {
            0
        };

        let mut idx_remaining = global_scalar_idx;
        let mut scalar_arr_idx = 0usize;
        while scalar_arr_idx < arrays.len() {
            let len = arrays[scalar_arr_idx].len();
            if idx_remaining < len {
                break;
            }
            idx_remaining -= len;
            scalar_arr_idx += 1;
        }

        if scalar_arr_idx >= arrays.len() {
            return Ok(None);
        }

        let scalar =
            lance_arrow::scalar::extract_scalar_value(&arrays[scalar_arr_idx], idx_remaining)?;
        if scalar.null_count() != 0 {
            return Ok(None);
        }
        if !Self::is_constant_values(arrays, &scalar, validity)? {
            return Ok(None);
        }
        Ok(Some(scalar))
    }

    fn resolve_dict_values_compression_metadata(
        field_metadata: &HashMap<String, String>,
        env_compression: Option<String>,
        env_compression_level: Option<String>,
    ) -> HashMap<String, String> {
        let mut metadata = HashMap::new();

        let compression = field_metadata
            .get(DICT_VALUES_COMPRESSION_META_KEY)
            .cloned()
            .or(env_compression)
            .unwrap_or_else(|| DEFAULT_DICT_VALUES_COMPRESSION.to_string());
        metadata.insert(COMPRESSION_META_KEY.to_string(), compression);

        if let Some(compression_level) = field_metadata
            .get(DICT_VALUES_COMPRESSION_LEVEL_META_KEY)
            .cloned()
            .or(env_compression_level)
        {
            metadata.insert(COMPRESSION_LEVEL_META_KEY.to_string(), compression_level);
        }

        metadata
    }

    fn build_dict_values_compressor_field(field: &Field) -> Result<Field> {
        // This is an internal synthetic field used only to feed metadata into
        // `create_block_compressor` for dictionary values. The concrete type/name here
        // are not semantically meaningful; we rely on explicit metadata below to control
        // general compression selection for dictionary values.
        let mut dict_values_field = Field::new_arrow("", DataType::UInt16, false)?;
        dict_values_field.metadata = Self::resolve_dict_values_compression_metadata(
            &field.metadata,
            env::var(DICT_VALUES_COMPRESSION_ENV_VAR).ok(),
            env::var(DICT_VALUES_COMPRESSION_LEVEL_ENV_VAR).ok(),
        );
        Ok(dict_values_field)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_miniblock(
        column_idx: u32,
        field: &Field,
        compression_strategy: &dyn CompressionStrategy,
        data: DataBlock,
        repdef: crate::repdef::SerializedRepDefs,
        row_number: u64,
        dictionary_data: Option<DataBlock>,
        num_rows: u64,
        miniblock_chunk_size: MiniblockChunkSize,
    ) -> Result<EncodedPage> {
        if let DataBlock::AllNull(_null_block) = data {
            // We should not be using mini-block for all-null.  There are other structural
            // encodings for that.
            unreachable!()
        }

        let num_items = data.num_values();

        let compressor = compression_strategy.create_miniblock_compressor(field, &data)?;
        let common_chunk_buffers =
            u64::from(repdef.rep_slicer().is_some()) + u64::from(repdef.def_slicer().is_some());
        let support_large_chunk = miniblock_chunk_size == MiniblockChunkSize::U32;
        let compression_context =
            MiniBlockCompressionContext::new(common_chunk_buffers, support_large_chunk, true);
        let (compressed_data, value_encoding) = compressor.compress(compression_context, data)?;

        let max_rep = repdef.def_meaning.iter().filter(|l| l.is_list()).count() as u16;

        let mut compressed_rep = repdef
            .rep_slicer()
            .map(|rep_slicer| {
                Self::compress_levels(
                    rep_slicer,
                    num_items,
                    compression_strategy,
                    &compressed_data.chunks,
                    max_rep,
                )
            })
            .transpose()?;

        let (rep_index, rep_index_depth) =
            match compressed_rep.as_mut().and_then(|cr| cr.rep_index.as_mut()) {
                Some(rep_index) => (Some(rep_index.clone()), 1),
                None => (None, 0),
            };

        let mut compressed_def = repdef
            .def_slicer()
            .map(|def_slicer| {
                Self::compress_levels(
                    def_slicer,
                    num_items,
                    compression_strategy,
                    &compressed_data.chunks,
                    /*max_rep=*/ 0,
                )
            })
            .transpose()?;

        // TODO: Parquet sparsely encodes values here.  We could do the same but
        // then we won't have log2 values per chunk.  This means more metadata
        // and potentially more decoder asymmetry.  However, it may be worth
        // investigating at some point

        let rep_data = compressed_rep
            .as_mut()
            .map(|cr| std::mem::take(&mut cr.data));
        let def_data = compressed_def
            .as_mut()
            .map(|cd| std::mem::take(&mut cd.data));

        let serialized =
            Self::serialize_miniblocks(compressed_data, rep_data, def_data, miniblock_chunk_size)?;
        let has_large_chunk = miniblock_chunk_size == MiniblockChunkSize::U32;

        // Metadata, Data, Dictionary, (maybe) Repetition Index
        let mut data = Vec::with_capacity(4);
        data.push(serialized.metadata);
        data.push(serialized.data);

        if let Some(dictionary_data) = dictionary_data {
            let num_dictionary_items = dictionary_data.num_values();
            let dict_values_field = Self::build_dict_values_compressor_field(field)?;

            let (dictionary_buffer, dictionary_encoding) =
                compress_required_block(compression_strategy, &dict_values_field, dictionary_data)?;

            data.push(dictionary_buffer);
            if let Some(rep_index) = rep_index {
                data.push(rep_index);
            }

            let description = ProtobufUtils21::miniblock_layout(
                compressed_rep.map(|cr| cr.compression),
                compressed_def.map(|cd| cd.compression),
                value_encoding,
                rep_index_depth,
                serialized.num_buffers,
                Some((dictionary_encoding, num_dictionary_items)),
                &repdef.def_meaning,
                num_items,
                has_large_chunk,
            );
            Ok(EncodedPage {
                num_rows,
                column_idx,
                data,
                description: PageEncoding::Structural(description),
                row_number,
            })
        } else {
            let description = ProtobufUtils21::miniblock_layout(
                compressed_rep.map(|cr| cr.compression),
                compressed_def.map(|cd| cd.compression),
                value_encoding,
                rep_index_depth,
                serialized.num_buffers,
                None,
                &repdef.def_meaning,
                num_items,
                has_large_chunk,
            );

            if let Some(rep_index) = rep_index {
                let view = rep_index.borrow_to_typed_slice::<u64>();
                let total = view.as_chunks::<2>().0.iter().map(|c| c[0]).sum::<u64>();
                debug_assert_eq!(total, num_rows);

                data.push(rep_index);
            }

            Ok(EncodedPage {
                num_rows,
                column_idx,
                data,
                description: PageEncoding::Structural(description),
                row_number,
            })
        }
    }

    // For fixed-size data we encode < control word | data > for each value
    fn serialize_full_zip_fixed(
        fixed: FixedWidthDataBlock,
        mut repdef: ControlWordIterator,
        num_values: u64,
    ) -> Result<SerializedFullZip> {
        if !fixed.bits_per_value.is_multiple_of(8) {
            return Err(Error::invalid_input_source(
                format!(
                    "Full-zip fixed-width values must be byte aligned, got {} bits per value",
                    fixed.bits_per_value
                )
                .into(),
            ));
        }

        let len = fixed.data.len() + repdef.bytes_per_word() * num_values as usize;
        let mut zipped_data = Vec::with_capacity(len);

        let max_rep_index_val = if repdef.has_repetition() {
            len as u64
        } else {
            // Setting this to 0 means we won't write a repetition index
            0
        };
        let mut rep_index_builder =
            BytepackedIntegerEncoder::with_capacity(num_values as usize + 1, max_rep_index_val);

        let bytes_per_value = fixed.bits_per_value as usize / 8;
        let mut offset = 0;

        if bytes_per_value == 0 {
            // No data, just dump the repdef into the buffer
            while let Some(control) = repdef.append_next(&mut zipped_data) {
                if control.is_new_row {
                    // We have finished a row
                    debug_assert!(offset <= len);
                    rep_index_builder.append_trusted(offset as u64);
                }
                offset = zipped_data.len();
            }
        } else {
            // We have data, zip it with the repdef
            let mut data_iter = fixed.data.chunks_exact(bytes_per_value);
            while let Some(control) = repdef.append_next(&mut zipped_data) {
                if control.is_new_row {
                    // We have finished a row
                    debug_assert!(offset <= len);
                    rep_index_builder.append_trusted(offset as u64);
                }
                if control.is_visible {
                    let value = data_iter.next().unwrap();
                    zipped_data.extend_from_slice(value);
                }
                offset = zipped_data.len();
            }
        }

        debug_assert_eq!(zipped_data.len(), len);
        // Put the final value in the rep index
        rep_index_builder.append_trusted(zipped_data.len() as u64);

        let zipped_data = LanceBuffer::from(zipped_data);
        let rep_index = rep_index_builder.into_data();
        let rep_index = if rep_index.is_empty() {
            None
        } else {
            Some(LanceBuffer::from(rep_index))
        };
        Ok(SerializedFullZip {
            values: zipped_data,
            repetition_index: rep_index,
        })
    }

    // For variable-size data we encode < control word | length | data > for each value
    //
    // In addition, we create a second buffer, the repetition index
    fn serialize_full_zip_variable(
        variable: VariableWidthBlock,
        mut repdef: ControlWordIterator,
        num_items: u64,
    ) -> Result<SerializedFullZip> {
        let bytes_per_offset = variable.bits_per_offset as usize / 8;
        if !variable.bits_per_offset.is_multiple_of(8) {
            return Err(Error::invalid_input_source(
                format!(
                    "Full-zip variable-width offsets must be byte aligned, got {} bits per offset",
                    variable.bits_per_offset
                )
                .into(),
            ));
        }
        let len = variable.data.len()
            + repdef.bytes_per_word() * num_items as usize
            + bytes_per_offset * variable.num_values as usize;
        let mut buf = Vec::with_capacity(len);

        let max_rep_index_val = len as u64;
        let mut rep_index_builder =
            BytepackedIntegerEncoder::with_capacity(num_items as usize + 1, max_rep_index_val);

        // TODO: byte pack the item lengths with varint encoding
        match bytes_per_offset {
            4 => {
                let offs = variable.offsets.borrow_to_typed_slice::<u32>();
                let mut rep_offset = 0;
                let mut windows_iter = offs.as_ref().windows(2);
                while let Some(control) = repdef.append_next(&mut buf) {
                    if control.is_new_row {
                        // We have finished a row
                        debug_assert!(rep_offset <= len);
                        rep_index_builder.append_trusted(rep_offset as u64);
                    }
                    if control.is_visible {
                        let window = windows_iter.next().unwrap();
                        if control.is_valid_item {
                            buf.extend_from_slice(&(window[1] - window[0]).to_le_bytes());
                            buf.extend_from_slice(
                                &variable.data[window[0] as usize..window[1] as usize],
                            );
                        }
                    }
                    rep_offset = buf.len();
                }
            }
            8 => {
                let offs = variable.offsets.borrow_to_typed_slice::<u64>();
                let mut rep_offset = 0;
                let mut windows_iter = offs.as_ref().windows(2);
                while let Some(control) = repdef.append_next(&mut buf) {
                    if control.is_new_row {
                        // We have finished a row
                        debug_assert!(rep_offset <= len);
                        rep_index_builder.append_trusted(rep_offset as u64);
                    }
                    if control.is_visible {
                        let window = windows_iter.next().unwrap();
                        if control.is_valid_item {
                            buf.extend_from_slice(&(window[1] - window[0]).to_le_bytes());
                            buf.extend_from_slice(
                                &variable.data[window[0] as usize..window[1] as usize],
                            );
                        }
                    }
                    rep_offset = buf.len();
                }
            }
            _ => {
                return Err(Error::invalid_input_source(
                    format!(
                        "Full-zip variable-width offsets must be 32 or 64 bits, got {} bits",
                        variable.bits_per_offset
                    )
                    .into(),
                ));
            }
        }

        // We might have saved a few bytes by not copying lengths when the length was zero.  However,
        // if we are over `len` then we have a bug.
        debug_assert!(buf.len() <= len);
        // Put the final value in the rep index
        rep_index_builder.append_trusted(buf.len() as u64);

        let zipped_data = LanceBuffer::from(buf);
        let rep_index = rep_index_builder.into_data();
        debug_assert!(!rep_index.is_empty());
        let rep_index = Some(LanceBuffer::from(rep_index));
        Ok(SerializedFullZip {
            values: zipped_data,
            repetition_index: rep_index,
        })
    }

    /// Serializes data into a single buffer according to the full-zip format which zips
    /// together the repetition, definition, and value data into a single buffer.
    fn serialize_full_zip(
        compressed_data: PerValueDataBlock,
        repdef: ControlWordIterator,
        num_items: u64,
    ) -> Result<SerializedFullZip> {
        match compressed_data {
            PerValueDataBlock::Fixed(fixed) => {
                Self::serialize_full_zip_fixed(fixed, repdef, num_items)
            }
            PerValueDataBlock::Variable(var) => {
                Self::serialize_full_zip_variable(var, repdef, num_items)
            }
        }
    }

    fn expand_boolean_to_bytes(fixed: FixedWidthDataBlock) -> FixedWidthDataBlock {
        debug_assert_eq!(fixed.bits_per_value, 1);
        let num_values = fixed.num_values as usize;
        let bool_buf = BooleanBuffer::new(fixed.data.into_buffer(), 0, num_values);
        let expanded: Vec<u8> = (0..num_values).map(|i| bool_buf.value(i) as u8).collect();
        FixedWidthDataBlock {
            data: LanceBuffer::from(expanded),
            bits_per_value: 8,
            num_values: fixed.num_values,
            block_info: BlockInfo::new(),
        }
    }

    fn encode_full_zip(
        column_idx: u32,
        field: &Field,
        compression_strategy: &dyn CompressionStrategy,
        data: DataBlock,
        repdef: crate::repdef::SerializedRepDefs,
        row_number: u64,
        num_lists: u64,
    ) -> Result<EncodedPage> {
        let max_rep = repdef
            .repetition_levels
            .as_ref()
            .map_or(0, |r| r.iter().max().copied().unwrap_or(0));
        let max_def = repdef
            .definition_levels
            .as_ref()
            .map_or(0, |d| d.iter().max().copied().unwrap_or(0));

        // To handle FSL we just flatten
        // let data = data.flatten();

        let (num_items, num_visible_items) =
            if let Some(rep_levels) = repdef.repetition_levels.as_ref() {
                // If there are rep levels there may be "invisible" items and we need to encode
                // rep_levels.len() things which might be larger than data.num_values()
                (rep_levels.len() as u64, data.num_values())
            } else {
                // If there are no rep levels then we encode data.num_values() things
                (data.num_values(), data.num_values())
            };

        let max_visible_def = repdef.max_visible_level.unwrap_or(u16::MAX);

        let repdef_iter = build_control_word_iterator(
            repdef.repetition_levels.as_deref(),
            max_rep,
            repdef.definition_levels.as_deref(),
            max_def,
            max_visible_def,
            num_items as usize,
        );
        let bits_rep = repdef_iter.bits_rep();
        let bits_def = repdef_iter.bits_def();

        // Full-zip requires byte-aligned values; expand 1-bit booleans to 1 byte each.
        let data = match data {
            DataBlock::FixedWidth(fixed) if fixed.bits_per_value == 1 => {
                DataBlock::FixedWidth(Self::expand_boolean_to_bytes(fixed))
            }
            other => other,
        };

        let compressor = compression_strategy.create_per_value(field, &data)?;
        let (compressed_data, value_encoding) = compressor.compress(data)?;

        let description = match &compressed_data {
            PerValueDataBlock::Fixed(fixed) => ProtobufUtils21::fixed_full_zip_layout(
                bits_rep,
                bits_def,
                fixed.bits_per_value as u32,
                value_encoding,
                &repdef.def_meaning,
                num_items as u32,
                num_visible_items as u32,
            ),
            PerValueDataBlock::Variable(variable) => ProtobufUtils21::variable_full_zip_layout(
                bits_rep,
                bits_def,
                variable.bits_per_offset as u32,
                value_encoding,
                &repdef.def_meaning,
                num_items as u32,
                num_visible_items as u32,
            ),
        };

        let zipped = Self::serialize_full_zip(compressed_data, repdef_iter, num_items)?;

        let data = if let Some(repindex) = zipped.repetition_index {
            vec![zipped.values, repindex]
        } else {
            vec![zipped.values]
        };

        Ok(EncodedPage {
            num_rows: num_lists,
            column_idx,
            data,
            description: PageEncoding::Structural(description),
            row_number,
        })
    }

    fn should_dictionary_encode(
        data_block: &DataBlock,
        field: &Field,
        fixed_width_dictionary_encoding: FixedWidthDictionaryEncoding,
    ) -> Option<DictEncodingBudget> {
        const DEFAULT_SAMPLE_SIZE: usize = 4096;
        const DEFAULT_SAMPLE_UNIQUE_RATIO: f64 = 0.98;

        // Since we only dictionary encode FixedWidth and VariableWidth blocks for now, we skip
        // estimating the size for other types.
        match data_block {
            DataBlock::FixedWidth(fixed) => {
                if fixed.bits_per_value == 64
                    && fixed_width_dictionary_encoding == FixedWidthDictionaryEncoding::Exclude64Bit
                {
                    return None;
                }
                if fixed.bits_per_value != 64 && fixed.bits_per_value != 128 {
                    return None;
                }
                if fixed.bits_per_value % 8 != 0 {
                    return None;
                }
            }
            DataBlock::VariableWidth(var) => {
                if var.bits_per_offset != 32 && var.bits_per_offset != 64 {
                    return None;
                }
            }
            _ => return None,
        }

        // Don't dictionary encode tiny arrays.
        let too_small = env::var("LANCE_ENCODING_DICT_TOO_SMALL")
            .ok()
            .and_then(|val| val.parse().ok())
            .unwrap_or(100);
        if data_block.num_values() < too_small {
            return None;
        }

        let num_values = data_block.num_values();

        // Apply divisor threshold and cap. This is intentionally conservative: the goal is to
        // avoid spending too much CPU trying to estimate very high cardinalities.
        let divisor: u64 = field
            .metadata
            .get(DICT_DIVISOR_META_KEY)
            .and_then(|val| val.parse().ok())
            .or_else(|| {
                env::var("LANCE_ENCODING_DICT_DIVISOR")
                    .ok()
                    .and_then(|val| val.parse().ok())
            })
            .unwrap_or(DEFAULT_DICT_DIVISOR);

        let max_cardinality: u64 = env::var("LANCE_ENCODING_DICT_MAX_CARDINALITY")
            .ok()
            .and_then(|val| val.parse().ok())
            .unwrap_or(DEFAULT_DICT_MAX_CARDINALITY);

        let threshold_cardinality = num_values
            .checked_div(divisor.max(1))
            .unwrap_or(0)
            .min(max_cardinality);
        if threshold_cardinality == 0 {
            return None;
        }

        // Get size ratio from metadata or env var.
        let threshold_ratio = field
            .metadata
            .get(DICT_SIZE_RATIO_META_KEY)
            .and_then(|val| val.parse::<f64>().ok())
            .or_else(|| {
                env::var("LANCE_ENCODING_DICT_SIZE_RATIO")
                    .ok()
                    .and_then(|val| val.parse().ok())
            })
            .unwrap_or(DEFAULT_DICT_SIZE_RATIO);

        if threshold_ratio <= 0.0 || threshold_ratio > 1.0 {
            panic!(
                "Invalid parameter: dict-size-ratio is {} which is not in the range (0, 1].",
                threshold_ratio
            );
        }

        let data_size = data_block.data_size();
        if data_size == 0 {
            return None;
        }

        let max_encoded_size = (data_size as f64 * threshold_ratio) as u64;
        let max_encoded_size = usize::try_from(max_encoded_size).ok()?;

        // Avoid probing dictionary encoding on data that appears to be near-unique
        // or likely to exceed the dictionary budget.
        if let Some(sample_unique_ratio) =
            Self::sample_unique_ratio(data_block, DEFAULT_SAMPLE_SIZE)?
        {
            if sample_unique_ratio >= DEFAULT_SAMPLE_UNIQUE_RATIO {
                return None;
            }

            let projected_cardinality = (sample_unique_ratio * num_values as f64).ceil() as u64;
            if projected_cardinality > threshold_cardinality {
                return None;
            }
        }

        let max_dict_entries = u32::try_from(threshold_cardinality.min(i32::MAX as u64)).ok()?;
        Some(DictEncodingBudget {
            max_dict_entries,
            max_encoded_size,
        })
    }

    /// Samples whether a page looks near-unique before attempting dictionary encoding.
    ///
    /// The probe uses deterministic block sampling (not RNG sampling), which keeps
    /// the check cheap and reproducible across runs. The result is only a gate for
    /// whether we try dictionary encoding, not a cardinality statistic.
    /// Returns `Some(None)` when there are too few reliable samples or the block type does not
    /// support dictionary encoding. Returns `None` for malformed data.
    fn sample_unique_ratio(data_block: &DataBlock, max_samples: usize) -> Option<Option<f64>> {
        use std::collections::HashSet;

        const NUM_SAMPLE_BLOCKS: usize = 32;
        const MIN_RELIABLE_SAMPLES: usize = 1024;

        let num_values = usize::try_from(data_block.num_values()).ok()?;
        if num_values == 0 {
            return Some(None);
        }

        let sample_count = num_values.min(max_samples).max(1);
        if sample_count < MIN_RELIABLE_SAMPLES {
            return Some(None);
        }

        let block_count = NUM_SAMPLE_BLOCKS.min(sample_count).min(num_values).max(1);
        let samples_per_block = (sample_count / block_count).max(1);
        let mut indices = Vec::with_capacity(sample_count);
        for block_idx in 0..block_count {
            let block_start = block_idx * num_values / block_count;
            let next_block_start = ((block_idx + 1) * num_values / block_count).min(num_values);
            let block_len = next_block_start.saturating_sub(block_start);
            let samples_in_block = samples_per_block.min(block_len);
            indices.extend((0..samples_in_block).map(|offset| block_start + offset));
        }

        if indices.len() < MIN_RELIABLE_SAMPLES {
            return Some(None);
        }

        let ratio = match data_block {
            DataBlock::FixedWidth(fixed) => match fixed.bits_per_value {
                64 => {
                    let values = fixed.data.borrow_to_typed_slice::<u64>();
                    let values = values.as_ref();
                    let mut unique: HashSet<u64> =
                        HashSet::with_capacity(indices.len().min(MIN_RELIABLE_SAMPLES));
                    for idx in indices.iter().copied() {
                        unique.insert(values.get(idx).copied()?);
                    }
                    unique.len() as f64 / indices.len() as f64
                }
                128 => {
                    let values = fixed.data.borrow_to_typed_slice::<u128>();
                    let values = values.as_ref();
                    let mut unique: HashSet<u128> =
                        HashSet::with_capacity(indices.len().min(MIN_RELIABLE_SAMPLES));
                    for idx in indices.iter().copied() {
                        unique.insert(values.get(idx).copied()?);
                    }
                    unique.len() as f64 / indices.len() as f64
                }
                _ => return Some(None),
            },
            DataBlock::VariableWidth(var) => {
                use xxhash_rust::xxh3::xxh3_64;

                // Hash variable-width slices instead of storing borrowed slice keys.
                let mut unique: HashSet<u64> =
                    HashSet::with_capacity(indices.len().min(MIN_RELIABLE_SAMPLES));
                match var.bits_per_offset {
                    32 => {
                        let offsets_ref = var.offsets.borrow_to_typed_slice::<u32>();
                        let offsets: &[u32] = offsets_ref.as_ref();
                        for i in indices.iter().copied() {
                            let start = usize::try_from(*offsets.get(i)?).ok()?;
                            let end = usize::try_from(*offsets.get(i + 1)?).ok()?;
                            if start > end || end > var.data.len() {
                                return None;
                            }
                            unique.insert(xxh3_64(&var.data[start..end]));
                        }
                    }
                    64 => {
                        let offsets_ref = var.offsets.borrow_to_typed_slice::<u64>();
                        let offsets: &[u64] = offsets_ref.as_ref();
                        for i in indices.iter().copied() {
                            let start = usize::try_from(*offsets.get(i)?).ok()?;
                            let end = usize::try_from(*offsets.get(i + 1)?).ok()?;
                            if start > end || end > var.data.len() {
                                return None;
                            }
                            unique.insert(xxh3_64(&var.data[start..end]));
                        }
                    }
                    _ => return Some(None),
                }
                unique.len() as f64 / indices.len() as f64
            }
            _ => return Some(None),
        };

        Some(Some(ratio))
    }

    fn slice_repdef(repdef: &SerializedRepDefs, range: Range<usize>) -> SerializedRepDefs {
        let repetition_levels = repdef
            .repetition_levels
            .as_ref()
            .map(|levels| levels[range.clone()].to_vec());
        let definition_levels = repdef
            .definition_levels
            .as_ref()
            .map(|levels| levels[range].to_vec());
        SerializedRepDefs::new_with_fixed_size_list_levels(
            repetition_levels,
            definition_levels,
            repdef.def_meaning.clone(),
            repdef.has_fixed_size_list_levels(),
        )
    }

    fn slice_arrays(
        arrays: &[ArrayRef],
        value_start: u64,
        num_values: u64,
    ) -> Result<Vec<ArrayRef>> {
        if num_values == 0 {
            return Ok(Vec::new());
        }

        let mut values_to_skip = usize::try_from(value_start).map_err(|_| {
            Error::invalid_input(format!("Value start {} is too large", value_start))
        })?;
        let mut values_remaining = usize::try_from(num_values).map_err(|_| {
            Error::invalid_input(format!("Value count {} is too large", num_values))
        })?;
        let mut sliced = Vec::new();

        for array in arrays {
            if values_to_skip >= array.len() {
                values_to_skip -= array.len();
                continue;
            }

            let offset = values_to_skip;
            let len = (array.len() - offset).min(values_remaining);
            sliced.push(array.slice(offset, len));
            values_remaining -= len;
            values_to_skip = 0;

            if values_remaining == 0 {
                break;
            }
        }

        if values_remaining != 0 {
            return Err(Error::internal(format!(
                "Page split requested {} values starting at {}, but the page did not contain enough values",
                num_values, value_start
            )));
        }

        Ok(sliced)
    }

    fn split_pages_for_miniblock_repdef_budget(
        arrays: Vec<ArrayRef>,
        repdef: SerializedRepDefs,
        budget: MiniBlockRepDefBudget,
        row_number: u64,
        num_rows: u64,
    ) -> Result<Vec<PrimitivePageData>> {
        if budget == MiniBlockRepDefBudget::WithinBudget {
            return Ok(vec![PrimitivePageData {
                arrays,
                structure: PrimitivePageStructure::Dense {
                    repdef,
                    single_row_miniblock_repdef_levels: None,
                },
                row_number,
                num_rows,
            }]);
        }
        if let MiniBlockRepDefBudget::SingleRowOverBudget(num_levels) = budget {
            return Ok(vec![PrimitivePageData {
                arrays,
                structure: PrimitivePageStructure::Dense {
                    repdef,
                    single_row_miniblock_repdef_levels: Some(num_levels),
                },
                row_number,
                num_rows,
            }]);
        }

        let MiniBlockRepDefBudget::RequiresPageSplit(splits) = budget else {
            unreachable!();
        };

        let mut pages = Vec::with_capacity(splits.len());
        for split in splits {
            let arrays = Self::slice_arrays(&arrays, split.value_start, split.num_values)?;
            let repdef = Self::slice_repdef(&repdef, split.level_range);
            pages.push(PrimitivePageData {
                arrays,
                structure: PrimitivePageStructure::Dense {
                    repdef,
                    single_row_miniblock_repdef_levels: None,
                },
                row_number: row_number + split.row_start,
                num_rows: split.num_rows,
            });
        }
        Ok(pages)
    }

    fn encode_dense_page(
        ctx: PrimitiveEncodeContext,
        page: PrimitivePageData,
        compression_strategy: Arc<dyn CompressionStrategy>,
        miniblock_chunk_size: MiniblockChunkSize,
        complex_null_encoding: ComplexNullEncoding,
        fixed_width_dictionary_encoding: FixedWidthDictionaryEncoding,
    ) -> Result<EncodedPage> {
        let PrimitiveEncodeContext {
            column_idx,
            field,
            encoding_metadata,
            is_simple_validity,
            has_repdef_info,
        } = ctx;
        let PrimitivePageData {
            arrays,
            structure,
            row_number,
            num_rows,
        } = page;
        let num_values = arrays.iter().map(|arr| arr.len() as u64).sum();

        let (repdef, single_row_miniblock_repdef_levels) = match structure {
            PrimitivePageStructure::Dense {
                repdef,
                single_row_miniblock_repdef_levels,
            } => (repdef, single_row_miniblock_repdef_levels),
            PrimitivePageStructure::Sparse { .. } => {
                unreachable!("dense atom received sparse page")
            }
        };

        if num_values == 0 {
            // This page contains only structural events, such as empty/null list rows.
            // The existing complex-null layout stores the rep/def stream without value buffers.
            log::debug!(
                "Encoding column {} with {} items ({} rows) using complex-null layout",
                column_idx,
                num_values,
                num_rows
            );
            return Self::encode_complex_all_null(
                column_idx,
                repdef,
                row_number,
                num_rows,
                complex_null_encoding,
                compression_strategy.as_ref(),
            );
        }

        let leaf_validity = Self::leaf_validity(&repdef, num_values as usize)?;
        let all_null = leaf_validity
            .as_ref()
            .map(|validity| validity.count_set_bits() == 0)
            .unwrap_or(false);

        if all_null {
            return if is_simple_validity {
                log::debug!(
                    "Encoding column {} with {} items ({} rows) using simple-null layout",
                    column_idx,
                    num_values,
                    num_rows
                );
                Self::encode_simple_all_null(column_idx, num_values, row_number)
            } else {
                log::debug!(
                    "Encoding column {} with {} items ({} rows) using complex-null layout",
                    column_idx,
                    num_values,
                    num_rows
                );
                Self::encode_complex_all_null(
                    column_idx,
                    repdef,
                    row_number,
                    num_rows,
                    complex_null_encoding,
                    compression_strategy.as_ref(),
                )
            };
        }

        if let DataType::Struct(fields) = &field.data_type()
            && fields.is_empty()
        {
            if has_repdef_info {
                return Err(Error::invalid_input_source(format!("Empty structs with rep/def information are not yet supported.  The field {} is an empty struct that either has nulls or is in a list.", field.name).into()));
            }
            // This is maybe a little confusing but the reader should never look at this anyways and it
            // seems like overkill to invent a new layout just for "empty structs".
            return Self::encode_simple_all_null(column_idx, num_values, row_number);
        }

        let data_block = DataBlock::from_arrays(&arrays, num_values);

        if let Some(num_levels) = single_row_miniblock_repdef_levels {
            let requested_encoding = encoding_metadata
                .get(STRUCTURAL_ENCODING_META_KEY)
                .map(|requested| requested.to_lowercase());
            let fullzip_error = match &data_block {
                // 1-bit booleans are widened to bytes inside `encode_full_zip`.
                DataBlock::FixedWidth(fixed)
                    if fixed.bits_per_value != 1 && !fixed.bits_per_value.is_multiple_of(8) =>
                {
                    Some(format!(
                        "Full-zip fixed-width values must be byte aligned, got {} bits per value",
                        fixed.bits_per_value
                    ))
                }
                DataBlock::VariableWidth(variable)
                    if !variable.bits_per_offset.is_multiple_of(8) =>
                {
                    Some(format!(
                        "Full-zip variable-width offsets must be byte aligned, got {} bits per offset",
                        variable.bits_per_offset
                    ))
                }
                DataBlock::VariableWidth(variable)
                    if variable.bits_per_offset != 32 && variable.bits_per_offset != 64 =>
                {
                    Some(format!(
                        "Full-zip variable-width offsets must be 32 or 64 bits, got {} bits",
                        variable.bits_per_offset
                    ))
                }
                DataBlock::Dictionary(_) => {
                    Some("Full-zip does not encode dictionary data blocks directly".to_string())
                }
                DataBlock::FixedSizeList(fsl) => match fsl.clone().try_into_flat() {
                    Some(flat) if flat.bits_per_value.is_multiple_of(8) => None,
                    Some(flat) => Some(format!(
                        "Full-zip fixed-size-list values must be byte aligned after flattening, got {} bits per value",
                        flat.bits_per_value
                    )),
                    None => Some(
                        "Full-zip fixed-size-list capability requires a flat fixed-width child"
                            .to_string(),
                    ),
                },
                DataBlock::FixedWidth(_) | DataBlock::VariableWidth(_) | DataBlock::Struct(_) => {
                    None
                }
                other => Some(format!(
                    "Full-zip does not support value block type {}",
                    other.name()
                )),
            };
            match requested_encoding.as_deref() {
                Some(STRUCTURAL_ENCODING_FULLZIP) => {
                    if let Some(reason) = fullzip_error {
                        return Err(Error::invalid_input_source(reason.into()));
                    }
                    return Self::encode_full_zip(
                        column_idx,
                        &field,
                        compression_strategy.as_ref(),
                        data_block,
                        repdef,
                        row_number,
                        num_rows,
                    );
                }
                Some(STRUCTURAL_ENCODING_MINIBLOCK) | None => {
                    if requested_encoding.is_none() && fullzip_error.is_none() {
                        log::debug!(
                            "Encoding column {} with {} items using full-zip layout because mini-block cannot split the structural page",
                            column_idx,
                            num_values
                        );
                        return Self::encode_full_zip(
                            column_idx,
                            &field,
                            compression_strategy.as_ref(),
                            data_block,
                            repdef,
                            row_number,
                            num_rows,
                        );
                    }
                    return Err(Error::invalid_input_source(
                        format!(
                            "Mini-block cannot encode {} rep/def levels in one top-level row. \
                             This usually means the row contains too much nested structure \
                             for the current layout.",
                            num_levels
                        )
                        .into(),
                    ));
                }
                _ => {}
            }
        }

        let requires_full_zip_packed_struct =
            if let DataBlock::Struct(ref struct_data_block) = data_block {
                struct_data_block.has_variable_width_child()
            } else {
                false
            };

        if requires_full_zip_packed_struct {
            log::debug!(
                "Encoding column {} with {} items using full-zip packed struct layout",
                column_idx,
                num_values
            );
            return Self::encode_full_zip(
                column_idx,
                &field,
                compression_strategy.as_ref(),
                data_block,
                repdef,
                row_number,
                num_rows,
            );
        }

        if let DataBlock::Dictionary(dict) = data_block {
            log::debug!(
                "Encoding column {} with {} items using dictionary encoding (already dictionary encoded)",
                column_idx,
                num_values
            );
            let (mut indices_data_block, dictionary_data_block) = dict.into_parts();
            // TODO: https://github.com/lancedb/lance/issues/4809
            // If we compute stats on dictionary_data_block => panic.
            // If we don't compute stats on indices_data_block => panic.
            // This is messy.  Don't make me call compute_stat ever.
            indices_data_block.compute_stat();
            return Self::encode_miniblock(
                column_idx,
                &field,
                compression_strategy.as_ref(),
                indices_data_block,
                repdef,
                row_number,
                Some(dictionary_data_block),
                num_rows,
                miniblock_chunk_size,
            );
        }

        // Try dictionary encoding first if applicable. If encoding aborts, fall back to the
        // preferred structural encoding.
        let dict_result = Self::should_dictionary_encode(
            &data_block,
            &field,
            fixed_width_dictionary_encoding,
        )
        .and_then(|budget| {
            log::debug!(
                "Encoding column {} with {} items using dictionary encoding (mini-block layout)",
                column_idx,
                num_values
            );
            dict::dictionary_encode(
                &data_block,
                budget.max_dict_entries,
                budget.max_encoded_size,
            )
        });

        if let Some((indices_data_block, dictionary_data_block)) = dict_result {
            Self::encode_miniblock(
                column_idx,
                &field,
                compression_strategy.as_ref(),
                indices_data_block,
                repdef,
                row_number,
                Some(dictionary_data_block),
                num_rows,
                miniblock_chunk_size,
            )
        } else if Self::prefers_miniblock(&data_block, encoding_metadata.as_ref()) {
            log::debug!(
                "Encoding column {} with {} items using mini-block layout",
                column_idx,
                num_values
            );
            Self::encode_miniblock(
                column_idx,
                &field,
                compression_strategy.as_ref(),
                data_block,
                repdef,
                row_number,
                None,
                num_rows,
                miniblock_chunk_size,
            )
        } else if Self::prefers_fullzip(encoding_metadata.as_ref()) {
            log::debug!(
                "Encoding column {} with {} items using full-zip layout",
                column_idx,
                num_values
            );
            Self::encode_full_zip(
                column_idx,
                &field,
                compression_strategy.as_ref(),
                data_block,
                repdef,
                row_number,
                num_rows,
            )
        } else {
            Err(Error::invalid_input_source(format!("Cannot determine structural encoding for field {}.  This typically indicates an invalid value of the field metadata key {}", field.name, STRUCTURAL_ENCODING_META_KEY).into()))
        }
    }

    /// Rebuilds dictionary chunks whose values array is empty.
    ///
    /// Such chunks are entirely null and retain their key validity until rep/def has been
    /// recorded.  At flush time their keys are zeroed and attached to a neighboring non-empty
    /// dictionary.  If every chunk is empty, one non-null placeholder value makes key zero valid.
    /// Rep/def retains the logical nullness, so these replacement keys are never exposed.
    fn rebuild_empty_dictionary_chunks(arrays: Vec<ArrayRef>) -> Result<Vec<ArrayRef>> {
        if !arrays.iter().any(|array| {
            array
                .as_any_dictionary_opt()
                .is_some_and(|dictionary| dictionary.values().is_empty())
        }) {
            return Ok(arrays);
        }

        let zeroed = |data_type: &DataType, len: usize| {
            new_null_array(data_type, len)
                .to_data()
                .into_builder()
                .nulls(None)
                .build()
                .map(make_array)
        };
        let rebuild =
            |keys: Vec<ArrayRef>, data_type: &DataType, values: ArrayRef| -> Result<ArrayRef> {
                let keys = keys.iter().map(|keys| keys.as_ref()).collect::<Vec<_>>();
                let keys = arrow_select::concat::concat(&keys)?;
                let data = keys
                    .to_data()
                    .into_builder()
                    .data_type(data_type.clone())
                    .child_data(vec![values.to_data()])
                    .build()?;
                Ok(make_array(data))
            };

        let mut rebuilt = Vec::with_capacity(arrays.len());
        let mut pending_keys = Vec::with_capacity(arrays.len());
        let mut empty_data_type = None;
        for array in arrays {
            let Some(dictionary) = array.as_any_dictionary_opt() else {
                return Err(Error::invalid_input_source(
                    "Cannot mix dictionary and non-dictionary chunks".into(),
                ));
            };
            if dictionary.values().is_empty() {
                pending_keys.push(zeroed(dictionary.keys().data_type(), array.len())?);
                empty_data_type.get_or_insert_with(|| array.data_type().clone());
            } else if pending_keys.is_empty() {
                rebuilt.push(array);
            } else {
                pending_keys.push(make_array(dictionary.keys().to_data()));
                rebuilt.push(rebuild(
                    std::mem::take(&mut pending_keys),
                    array.data_type(),
                    dictionary.values().clone(),
                )?);
            }
        }

        if pending_keys.is_empty() {
            return Ok(rebuilt);
        }
        if let Some(array) = rebuilt.pop() {
            let Some(dictionary) = array.as_any_dictionary_opt() else {
                return Err(Error::invalid_input_source(
                    "Cannot mix dictionary and non-dictionary chunks".into(),
                ));
            };
            let mut keys = Vec::with_capacity(pending_keys.len() + 1);
            keys.push(make_array(dictionary.keys().to_data()));
            keys.append(&mut pending_keys);
            rebuilt.push(rebuild(
                keys,
                array.data_type(),
                dictionary.values().clone(),
            )?);
        } else {
            let data_type = empty_data_type.ok_or_else(|| {
                Error::internal("Missing data type for an empty dictionary chunk")
            })?;
            let DataType::Dictionary(_, value_type) = &data_type else {
                return Err(Error::internal(format!(
                    "Expected dictionary data type, got {data_type}"
                )));
            };
            rebuilt.push(rebuild(pending_keys, &data_type, zeroed(value_type, 1)?)?);
        }
        Ok(rebuilt)
    }

    // Creates encode tasks, consuming all buffered data
    fn do_flush(
        &mut self,
        arrays: Vec<ArrayRef>,
        repdefs: Vec<RepDefBuilder>,
        row_number: u64,
        num_rows: u64,
    ) -> Result<Vec<EncodeTask>> {
        let arrays = Self::rebuild_empty_dictionary_chunks(arrays)?;
        DataBlock::validate_arrays(&arrays, &self.field.name)?;
        let num_values = arrays.iter().map(|arr| arr.len() as u64).sum();
        let is_simple_validity = repdefs.iter().all(|rd| rd.is_simple_validity());
        let has_repdef_info = repdefs.iter().any(|rd| !rd.is_empty());
        let normalized = RepDefBuilder::normalize(repdefs);
        let plan_ctx = PrimitivePlanContext {
            column_idx: self.column_index,
            field: &self.field,
            encoding_metadata: &self.encoding_metadata,
        };
        let mut pages = None;
        for page_encoding in self.page_encodings.iter() {
            if let Some(planned) = page_encoding.behavior.try_plan_pages(
                &plan_ctx,
                &arrays,
                &normalized,
                row_number,
                num_rows,
                num_values,
            )? {
                pages = Some(planned);
                break;
            }
        }
        let pages = pages.ok_or_else(|| {
            Error::invalid_input_source(
                format!(
                    "No primitive page planner supports field '{}'",
                    self.field.name
                )
                .into(),
            )
        })?;

        let mut tasks = Vec::with_capacity(pages.len());
        let ctx = PrimitiveEncodeContext {
            column_idx: self.column_index,
            field: self.field.clone(),
            encoding_metadata: self.encoding_metadata.clone(),
            is_simple_validity,
            has_repdef_info,
        };
        for page in pages {
            let ctx = ctx.clone();
            let page_encodings = self.page_encodings.clone();
            let task =
                spawn_cpu(move || Self::encode_page(page_encodings.as_ref(), &ctx, page)).boxed();
            tasks.push(task);
        }
        Ok(tasks)
    }

    fn extract_validity_buf(
        array: Arc<dyn Array>,
        repdef: &mut RepDefBuilder,
        keep_original_array: bool,
    ) -> Result<Arc<dyn Array>> {
        if let Some(validity) = array.nulls() {
            if keep_original_array {
                repdef.add_validity_bitmap(validity.clone());
            } else {
                repdef.add_validity_bitmap(deep_copy_nulls(Some(validity)).unwrap());
            }
            // Empty dictionaries have no valid key payload.  Keep the validity until rep/def is
            // grouped with the buffered page; `do_flush` then rebuilds the keys against either a
            // neighboring values array or an all-empty-page placeholder.
            if let Some(dictionary) = array.as_any_dictionary_opt()
                && dictionary.values().is_empty()
            {
                return Ok(array);
            }
            let data_no_nulls = array.to_data().into_builder().nulls(None).build()?;
            Ok(make_array(data_no_nulls))
        } else {
            repdef.add_no_null(array.len());
            Ok(array)
        }
    }

    fn extract_validity(
        mut array: Arc<dyn Array>,
        repdef: &mut RepDefBuilder,
        keep_original_array: bool,
    ) -> Result<Arc<dyn Array>> {
        match array.data_type() {
            DataType::Null => {
                repdef.add_validity_bitmap(NullBuffer::new(BooleanBuffer::new_unset(array.len())));
                Ok(array)
            }
            DataType::Dictionary(_, _) => {
                array = dict::normalize_dict_nulls(array)?;
                array = dict::clear_out_of_range_null_keys(array)?;
                Self::extract_validity_buf(array, repdef, keep_original_array)
            }
            // Extract our validity buf but NOT any child validity bufs. (they will be encoded in
            // as part of the values).  Note: for FSL we do not use repdef.add_fsl because we do
            // NOT want to increase the repdef depth.
            //
            // This would be quite catasrophic for something like vector embeddings.  Imagine we
            // had thousands of vectors and some were null but no vector contained null items.  If
            // we treated the vectors (primitive FSL) like we treat structural FSL we would end up
            // with a rep/def value for every single item in the vector.
            _ => Self::extract_validity_buf(array, repdef, keep_original_array),
        }
    }
}

impl PrimitivePageEncodingBehavior for RejectSparsePrimitiveEncoding {
    fn validate_field(&self, field: &Field, metadata: &HashMap<String, String>) -> Result<()> {
        if metadata
            .get(STRUCTURAL_ENCODING_META_KEY)
            .is_some_and(|requested| requested.eq_ignore_ascii_case(STRUCTURAL_ENCODING_SPARSE))
        {
            return Err(Error::invalid_input_source(
                format!(
                    "Field '{}' requests sparse structural encoding, which is not enabled by the selected file format",
                    field.name
                )
                .into(),
            ));
        }
        Ok(())
    }
}

fn plan_dense_primitive_pages(
    arrays: &[ArrayRef],
    normalized: &NormalizedStructuralPlan,
    row_number: u64,
    num_rows: u64,
    num_values: u64,
) -> Result<Vec<PrimitivePageData>> {
    let (repdef, miniblock_repdef_budget) = normalized.serialize_with_miniblock_repdef_budget(
        miniblock::max_repdef_levels_per_chunk,
        num_rows,
        num_values,
    )?;
    PrimitiveStructuralEncoder::split_pages_for_miniblock_repdef_budget(
        arrays.to_vec(),
        repdef,
        miniblock_repdef_budget,
        row_number,
        num_rows,
    )
}

impl PrimitivePageEncodingBehavior for DenseU16PrimitiveEncoding {
    fn try_plan_pages(
        &self,
        _ctx: &PrimitivePlanContext<'_>,
        arrays: &[ArrayRef],
        normalized: &NormalizedStructuralPlan,
        row_number: u64,
        num_rows: u64,
        num_values: u64,
    ) -> Result<Option<Vec<PrimitivePageData>>> {
        Ok(Some(plan_dense_primitive_pages(
            arrays, normalized, row_number, num_rows, num_values,
        )?))
    }

    fn try_encode_page(
        &self,
        ctx: &PrimitiveEncodeContext,
        page: PrimitivePageData,
    ) -> Result<PrimitiveEncodeAttempt> {
        if !matches!(&page.structure, PrimitivePageStructure::Dense { .. }) {
            return Ok(PrimitiveEncodeAttempt::Unhandled(page));
        }
        Ok(PrimitiveEncodeAttempt::Encoded(
            PrimitiveStructuralEncoder::encode_dense_page(
                ctx.clone(),
                page,
                self.compression.clone(),
                MiniblockChunkSize::U16,
                ComplexNullEncoding::RawLevels,
                FixedWidthDictionaryEncoding::Exclude64Bit,
            )?,
        ))
    }
}

impl PrimitivePageEncodingBehavior for DenseU32PrimitiveEncoding {
    fn try_plan_pages(
        &self,
        _ctx: &PrimitivePlanContext<'_>,
        arrays: &[ArrayRef],
        normalized: &NormalizedStructuralPlan,
        row_number: u64,
        num_rows: u64,
        num_values: u64,
    ) -> Result<Option<Vec<PrimitivePageData>>> {
        Ok(Some(plan_dense_primitive_pages(
            arrays, normalized, row_number, num_rows, num_values,
        )?))
    }

    fn try_encode_page(
        &self,
        ctx: &PrimitiveEncodeContext,
        page: PrimitivePageData,
    ) -> Result<PrimitiveEncodeAttempt> {
        if !matches!(&page.structure, PrimitivePageStructure::Dense { .. }) {
            return Ok(PrimitiveEncodeAttempt::Unhandled(page));
        }
        Ok(PrimitiveEncodeAttempt::Encoded(
            PrimitiveStructuralEncoder::encode_dense_page(
                ctx.clone(),
                page,
                self.compression.clone(),
                MiniblockChunkSize::U32,
                ComplexNullEncoding::CompressedLevels,
                FixedWidthDictionaryEncoding::Include64Bit,
            )?,
        ))
    }
}

impl PrimitivePageEncodingBehavior for SparsePrimitiveEncoding {
    fn try_plan_pages(
        &self,
        ctx: &PrimitivePlanContext<'_>,
        arrays: &[ArrayRef],
        normalized: &NormalizedStructuralPlan,
        row_number: u64,
        num_rows: u64,
        num_values: u64,
    ) -> Result<Option<Vec<PrimitivePageData>>> {
        let requested_encoding = ctx.encoding_metadata.get(STRUCTURAL_ENCODING_META_KEY);
        let requests_sparse = requested_encoding
            .is_some_and(|requested| requested.eq_ignore_ascii_case(STRUCTURAL_ENCODING_SPARSE));
        if requests_sparse {
            let plan = sparse::writer::plan(normalized, num_values)?;
            if sparse::writer::uses_constant_layout(&plan, ctx.field) {
                return Ok(None);
            }
            return Ok(Some(vec![PrimitivePageData {
                arrays: arrays.to_vec(),
                structure: PrimitivePageStructure::Sparse {
                    plan,
                    prepared_values: None,
                },
                row_number,
                num_rows,
            }]));
        }

        let (_, miniblock_repdef_budget) = normalized.serialize_with_miniblock_repdef_budget(
            miniblock::max_repdef_levels_per_chunk,
            num_rows,
            num_values,
        )?;
        let automatic_sparse = layout::select_automatic_sparse(
            requested_encoding.map(String::as_str),
            &miniblock_repdef_budget,
            || {
                let data = DataBlock::from_arrays(arrays, num_values);
                if !sparse::writer::supports_value_block(&data) {
                    return Ok(None);
                }
                let prepared_values = match sparse::writer::prepare_values(
                    ctx.field,
                    self.compression.as_ref(),
                    data,
                ) {
                    Ok(prepared_values) => prepared_values,
                    Err(error) => {
                        debug!(
                            "Keeping column {} on its dense structural path because sparse value preparation is unavailable: {}",
                            ctx.column_idx, error
                        );
                        return Ok(None);
                    }
                };
                let plan = sparse::writer::plan(normalized, num_values)?;
                if sparse::writer::uses_constant_layout(&plan, ctx.field) {
                    return Ok(None);
                }
                Ok(Some((plan, prepared_values)))
            },
        )?;
        Ok(automatic_sparse.map(|(plan, prepared_values)| {
            vec![PrimitivePageData {
                arrays: arrays.to_vec(),
                structure: PrimitivePageStructure::Sparse {
                    plan,
                    prepared_values: Some(prepared_values),
                },
                row_number,
                num_rows,
            }]
        }))
    }

    fn try_encode_page(
        &self,
        ctx: &PrimitiveEncodeContext,
        page: PrimitivePageData,
    ) -> Result<PrimitiveEncodeAttempt> {
        if !matches!(&page.structure, PrimitivePageStructure::Sparse { .. }) {
            return Ok(PrimitiveEncodeAttempt::Unhandled(page));
        }
        let PrimitivePageData {
            arrays,
            structure:
                PrimitivePageStructure::Sparse {
                    plan,
                    prepared_values,
                },
            row_number,
            num_rows,
        } = page
        else {
            unreachable!()
        };
        let num_values = arrays.iter().map(|array| array.len() as u64).sum();
        log::debug!(
            "Encoding column {} with {} visible items ({} rows) using sparse layout",
            ctx.column_idx,
            num_values,
            num_rows
        );
        Ok(PrimitiveEncodeAttempt::Encoded(
            sparse::writer::encode_page(
                ctx.column_idx,
                &ctx.field,
                self.compression.as_ref(),
                prepared_values.map_or_else(
                    || {
                        sparse::writer::SparseValueInput::Unprepared(DataBlock::from_arrays(
                            &arrays, num_values,
                        ))
                    },
                    sparse::writer::SparseValueInput::Prepared,
                ),
                plan,
                row_number,
                num_rows,
            )?,
        ))
    }
}

impl PrimitivePageEncodingBehavior for ConstantPrimitiveEncoding {
    fn try_encode_page(
        &self,
        ctx: &PrimitiveEncodeContext,
        page: PrimitivePageData,
    ) -> Result<PrimitiveEncodeAttempt> {
        let PrimitivePageStructure::Dense { repdef, .. } = &page.structure else {
            return Ok(PrimitiveEncodeAttempt::Unhandled(page));
        };
        let num_values: u64 = page.arrays.iter().map(|array| array.len() as u64).sum();
        if num_values == 0 {
            return Ok(PrimitiveEncodeAttempt::Unhandled(page));
        }
        let leaf_validity = PrimitiveStructuralEncoder::leaf_validity(repdef, num_values as usize)?;
        if leaf_validity
            .as_ref()
            .is_some_and(|validity| validity.count_set_bits() == 0)
            || matches!(ctx.field.data_type(), DataType::Struct(fields) if fields.is_empty())
        {
            return Ok(PrimitiveEncodeAttempt::Unhandled(page));
        }
        let Some(scalar) =
            PrimitiveStructuralEncoder::find_constant_scalar(&page.arrays, leaf_validity.as_ref())?
        else {
            return Ok(PrimitiveEncodeAttempt::Unhandled(page));
        };
        let PrimitivePageData {
            structure: PrimitivePageStructure::Dense { repdef, .. },
            row_number,
            num_rows,
            ..
        } = page
        else {
            unreachable!()
        };
        log::debug!(
            "Encoding column {} with {} items ({} rows) using constant layout",
            ctx.column_idx,
            num_values,
            num_rows
        );
        Ok(PrimitiveEncodeAttempt::Encoded(
            constant::encode_constant_page(ctx.column_idx, scalar, repdef, row_number, num_rows)?,
        ))
    }
}

impl FieldEncoder for PrimitiveStructuralEncoder {
    // Buffers data, if there is enough to write a page then we create an encode task
    fn maybe_encode(
        &mut self,
        array: ArrayRef,
        _external_buffers: &mut OutOfLineBuffers,
        mut repdef: RepDefBuilder,
        row_number: u64,
        num_rows: u64,
    ) -> Result<Vec<EncodeTask>> {
        let array = Self::extract_validity(array, &mut repdef, self.keep_original_array)?;
        self.accumulated_repdefs.push(repdef);

        if let Some((arrays, row_number, num_rows)) =
            self.accumulation_queue.insert(array, row_number, num_rows)
        {
            let accumulated_repdefs = std::mem::take(&mut self.accumulated_repdefs);
            Ok(self.do_flush(arrays, accumulated_repdefs, row_number, num_rows)?)
        } else {
            Ok(vec![])
        }
    }

    // If there is any data left in the buffer then create an encode task from it
    fn flush(&mut self, _external_buffers: &mut OutOfLineBuffers) -> Result<Vec<EncodeTask>> {
        if let Some((arrays, row_number, num_rows)) = self.accumulation_queue.flush() {
            let accumulated_repdefs = std::mem::take(&mut self.accumulated_repdefs);
            Ok(self.do_flush(arrays, accumulated_repdefs, row_number, num_rows)?)
        } else {
            Ok(vec![])
        }
    }

    fn num_columns(&self) -> u32 {
        1
    }

    fn finish(
        &mut self,
        _external_buffers: &mut OutOfLineBuffers,
    ) -> BoxFuture<'_, Result<Vec<crate::encoder::EncodedColumn>>> {
        std::future::ready(Ok(vec![EncodedColumn::default()])).boxed()
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests;
