// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::any::Any;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;
use std::{iter, vec};

use arrow_buffer::ScalarBuffer;
use bytes::Bytes;
use futures::FutureExt;
use futures::future::BoxFuture;
use itertools::Itertools;
use lance_core::Result;
use lance_core::cache::{Context, DeepSizeOf};
use lance_core::error::Error;
use lance_core::utils::bit::pad_bytes;

use crate::EncodingsIo;
use crate::buffer::LanceBuffer;
use crate::compression::{BlockDecompressor, DecompressionStrategy, MiniBlockDecompressor};
use crate::data::{DataBlock, DataBlockBuilder, DictionaryDataBlock};
use crate::decoder::{DecodePageTask, DecodedPage, StructuralPageDecoder};
use crate::format::pb21::compressive_encoding::Compression;
use crate::format::{ProtobufUtils21, pb21};
use crate::repdef::{DefinitionInterpretation, LevelBuffer, RepDefUnraveler, max_visible_level};

use super::chunk_index::{
    ItemCounts, MiniBlockChunkIndex, PrefixSums, RowMapping, parse_nested_rep,
};
use super::{
    CachedPageData, LoadedChunk, MAX_LEGACY_RLE_LEVELS, MINIBLOCK_ALIGNMENT, PageInitialization,
    PageInitializationBuffers, PageLoadTask, StructuralPageScheduler,
};

/// A mini-block chunk that has been decoded and decompressed
#[derive(Debug, Clone)]
struct DecodedMiniBlockChunk {
    rep: Option<ScalarBuffer<u16>>,
    def: Option<ScalarBuffer<u16>>,
    values: DataBlock,
}

/// A task to decode a one or more mini-blocks of data into an output batch
///
/// Note: Two batches might share the same mini-block of data.  When this happens
/// then each batch gets a copy of the block and each batch decodes the block independently.
///
/// This means we have duplicated work but it is necessary to avoid having to synchronize
/// the decoding of the block. (TODO: test this theory)
#[derive(Debug)]
struct DecodeMiniBlockTask {
    rep_decompressor: Option<Arc<dyn BlockDecompressor>>,
    def_decompressor: Option<Arc<dyn BlockDecompressor>>,
    value_decompressor: Arc<dyn MiniBlockDecompressor>,
    dictionary_data: Option<Arc<DataBlock>>,
    def_meaning: Arc<[DefinitionInterpretation]>,
    num_buffers: u64,
    max_visible_level: u16,
    instructions: Vec<(ChunkDrainInstructions, LoadedChunk)>,
    has_large_chunk: bool,
}

impl DecodeMiniBlockTask {
    fn decoded_size_bytes(&self) -> Option<u64> {
        if self.rep_decompressor.is_some() || self.def_decompressor.is_some() {
            return None;
        }
        let num_values = self
            .instructions
            .iter()
            .try_fold(0_u64, |total, (instruction, _)| {
                total.checked_add(instruction.rows_to_take)
            })?;
        self.value_decompressor.decoded_size_bytes(num_values)
    }

    fn resolve_num_levels(
        rep_decompressor: Option<&dyn BlockDecompressor>,
        rep_levels: Option<&LanceBuffer>,
        def_decompressor: Option<&dyn BlockDecompressor>,
        def_levels: Option<&LanceBuffer>,
        declared_num_levels: u16,
    ) -> Result<u64> {
        let rep_num_levels = match (rep_decompressor, rep_levels) {
            (Some(decompressor), Some(levels)) => decompressor.infer_num_values(levels)?,
            (None, None) => None,
            _ => {
                return Err(Error::invalid_input_source(
                    "miniblock repetition codec and payload presence disagree".into(),
                ));
            }
        };
        let def_num_levels = match (def_decompressor, def_levels) {
            (Some(decompressor), Some(levels)) => decompressor.infer_num_values(levels)?,
            (None, None) => None,
            _ => {
                return Err(Error::invalid_input_source(
                    "miniblock definition codec and payload presence disagree".into(),
                ));
            }
        };

        let inferred_num_levels = match (rep_num_levels, def_num_levels) {
            (Some(rep_num_levels), Some(def_num_levels)) if rep_num_levels != def_num_levels => {
                return Err(Error::invalid_input_source(
                    format!(
                        "miniblock structural streams disagree on the level count: repetition inferred {rep_num_levels}, definition inferred {def_num_levels}"
                    )
                    .into(),
                ));
            }
            (Some(num_levels), _) | (_, Some(num_levels)) => num_levels,
            (None, None) => return Ok(u64::from(declared_num_levels)),
        };

        let declared_num_levels = u64::from(declared_num_levels);
        if inferred_num_levels == declared_num_levels {
            return Ok(inferred_num_levels);
        }
        let u16_modulus = u64::from(u16::MAX) + 1;
        if inferred_num_levels <= declared_num_levels
            || inferred_num_levels % u16_modulus != declared_num_levels
        {
            return Err(Error::invalid_input_source(
                format!(
                    "miniblock payload proves {inferred_num_levels} levels but the header declared {declared_num_levels}; the counts are not congruent modulo 65536"
                )
                .into(),
            ));
        }
        if inferred_num_levels > MAX_LEGACY_RLE_LEVELS {
            return Err(Error::invalid_input_source(
                format!(
                    "miniblock payload proves {inferred_num_levels} levels, exceeding the legacy RLE payload bound of {MAX_LEGACY_RLE_LEVELS}"
                )
                .into(),
            ));
        }
        Ok(inferred_num_levels)
    }

    fn decode_levels(
        decompressor: &dyn BlockDecompressor,
        levels: LanceBuffer,
        expected_num_levels: u64,
    ) -> Result<ScalarBuffer<u16>> {
        let levels = decompressor.decompress(Some(levels), expected_num_levels)?;
        let levels = levels.as_fixed_width().ok_or_else(|| {
            Error::invalid_input_source(
                "miniblock levels did not decode to fixed-width data".into(),
            )
        })?;
        if levels.num_values != expected_num_levels {
            return Err(Error::invalid_input_source(
                format!(
                    "miniblock levels decoded {} values, expected {expected_num_levels}",
                    levels.num_values
                )
                .into(),
            ));
        }
        if levels.bits_per_value != 16 {
            return Err(Error::invalid_input_source(
                format!(
                    "miniblock levels decoded {} bits per value, expected 16",
                    levels.bits_per_value
                )
                .into(),
            ));
        }
        let expected_bytes = expected_num_levels.checked_mul(2).ok_or_else(|| {
            Error::invalid_input_source(
                format!("miniblock level byte count overflowed for {expected_num_levels} levels")
                    .into(),
            )
        })?;
        if levels.data.len() as u64 != expected_bytes {
            return Err(Error::invalid_input_source(
                format!(
                    "miniblock levels decoded {} bytes, expected {expected_bytes}",
                    levels.data.len()
                )
                .into(),
            ));
        }
        Ok(levels.data.borrow_to_typed_slice::<u16>())
    }

    // We are building a LevelBuffer (levels) and want to copy into it `total_len`
    // values from `level_buf` starting at `offset`.
    //
    // We need to handle both the case where `levels` is None (no nulls encountered
    // yet) and the case where `level_buf` is None (the input we are copying from has
    // no nulls)
    fn extend_levels(
        range: Range<u64>,
        levels: &mut Option<LevelBuffer>,
        level_buf: &Option<impl AsRef<[u16]>>,
        dest_offset: usize,
    ) {
        if let Some(level_buf) = level_buf {
            if levels.is_none() {
                // This is the first non-empty def buf we've hit, fill in the past
                // with 0 (valid)
                let mut new_levels_vec =
                    LevelBuffer::with_capacity(dest_offset + (range.end - range.start) as usize);
                new_levels_vec.extend(iter::repeat_n(0, dest_offset));
                *levels = Some(new_levels_vec);
            }
            levels.as_mut().unwrap().extend(
                level_buf.as_ref()[range.start as usize..range.end as usize]
                    .iter()
                    .copied(),
            );
        } else if let Some(levels) = levels {
            let num_values = (range.end - range.start) as usize;
            // This is an all-valid level_buf but we had nulls earlier and so we
            // need to materialize it
            levels.extend(iter::repeat_n(0, num_values));
        }
    }

    /// Maps a range of rows to a range of items and a range of levels
    ///
    /// If there is no repetition information this just returns the range as-is.
    ///
    /// If there is repetition information then we need to do some work to figure out what
    /// range of items corresponds to the requested range of rows.
    ///
    /// For example, if the data is [[1, 2, 3], [4, 5], [6, 7]] and the range is 1..2 (i.e. just row
    /// 1) then the user actually wants items 3..5.  In the above case the rep levels would be:
    ///
    /// Idx: 0 1 2 3 4 5 6
    /// Rep: 1 0 0 1 0 1 0
    ///
    /// So the start (1) maps to the second 1 (idx=3) and the end (2) maps to the third 1 (idx=5)
    ///
    /// If there are invisible items then we don't count them when calculating the range of items we
    /// are interested in but we do count them when calculating the range of levels we are interested
    /// in.  As a result we have to return both the item range (first return value) and the level range
    /// (second return value).
    ///
    /// For example, if the data is [[1, 2, 3], [4, 5], NULL, [6, 7, 8]] and the range is 2..4 then the
    /// user wants items 5..8 but they want levels 5..9.  In the above case the rep/def levels would be:
    ///
    /// Idx: 0 1 2 3 4 5 6 7 8
    /// Rep: 1 0 0 1 0 1 1 0 0
    /// Def: 0 0 0 0 0 1 0 0 0
    /// Itm: 1 2 3 4 5 6 7 8
    ///
    /// Finally, we have to contend with the fact that chunks may or may not start with a "preamble" of
    /// trailing values that finish up a list from the previous chunk.  In this case the first item does
    /// not start at max_rep because it is a continuation of the previous chunk.  For our purposes we do
    /// not consider this a "row" and so the range 0..1 will refer to the first row AFTER the preamble.
    ///
    /// We have a separate parameter (`preamble_action`) to control whether we want the preamble or not.
    ///
    /// Note that the "trailer" is considered a "row" and if we want it we should include it in the range.
    fn map_range(
        range: Range<u64>,
        rep: Option<&impl AsRef<[u16]>>,
        def: Option<&impl AsRef<[u16]>>,
        max_rep: u16,
        max_visible_def: u16,
        // The total number of items (not rows) in the chunk.  This is not quite the same as
        // rep.len() / def.len() because it doesn't count invisible items
        total_items: u64,
        preamble_action: PreambleAction,
    ) -> (Range<u64>, Range<u64>) {
        if let Some(rep) = rep {
            let mut rep = rep.as_ref();
            // If there is a preamble and we need to skip it then do that first.  The work is the same
            // whether there is def information or not
            let mut items_in_preamble = 0_u64;
            let first_row_start = match preamble_action {
                PreambleAction::Skip | PreambleAction::Take => {
                    let first_row_start = if let Some(def) = def.as_ref() {
                        let mut first_row_start = None;
                        for (idx, (rep, def)) in rep.iter().zip(def.as_ref()).enumerate() {
                            if *rep == max_rep {
                                first_row_start = Some(idx as u64);
                                break;
                            }
                            if *def <= max_visible_def {
                                items_in_preamble += 1;
                            }
                        }
                        first_row_start
                    } else {
                        let first_row_start =
                            rep.iter().position(|&r| r == max_rep).map(|r| r as u64);
                        items_in_preamble = first_row_start.unwrap_or(rep.len() as u64);
                        first_row_start
                    };
                    // It is possible for a chunk to be entirely partial values but if it is then it
                    // should never show up as a preamble to skip
                    if first_row_start.is_none() {
                        assert!(preamble_action == PreambleAction::Take);
                        return (0..total_items, 0..rep.len() as u64);
                    }
                    let first_row_start = first_row_start.unwrap();
                    rep = &rep[first_row_start as usize..];
                    first_row_start
                }
                PreambleAction::Absent => {
                    debug_assert!(rep[0] == max_rep);
                    0
                }
            };

            // We hit this case when all we needed was the preamble
            if range.start == range.end {
                debug_assert!(preamble_action == PreambleAction::Take);
                debug_assert!(items_in_preamble <= total_items);
                return (0..items_in_preamble, 0..first_row_start);
            }
            assert!(range.start < range.end);

            let mut rows_seen = 0;
            let mut new_start = 0;
            let mut new_levels_start = 0;

            if let Some(def) = def {
                let def = &def.as_ref()[first_row_start as usize..];

                // range.start == 0 always maps to 0 (even with invis items), otherwise we need to walk
                let mut lead_invis_seen = 0;

                if range.start > 0 {
                    if def[0] > max_visible_def {
                        lead_invis_seen += 1;
                    }
                    for (idx, (rep, def)) in rep.iter().zip(def).skip(1).enumerate() {
                        if *rep == max_rep {
                            rows_seen += 1;
                            if rows_seen == range.start {
                                new_start = idx as u64 + 1 - lead_invis_seen;
                                new_levels_start = idx as u64 + 1;
                                break;
                            }
                        }
                        if *def > max_visible_def {
                            lead_invis_seen += 1;
                        }
                    }
                }

                rows_seen += 1;

                let mut new_end = u64::MAX;
                let mut new_levels_end = rep.len() as u64;
                let new_start_is_visible = def[new_levels_start as usize] <= max_visible_def;
                let mut tail_invis_seen = if new_start_is_visible { 0 } else { 1 };
                for (idx, (rep, def)) in rep[(new_levels_start + 1) as usize..]
                    .iter()
                    .zip(&def[(new_levels_start + 1) as usize..])
                    .enumerate()
                {
                    if *rep == max_rep {
                        rows_seen += 1;
                        if rows_seen == range.end + 1 {
                            new_end = idx as u64 + new_start + 1 - tail_invis_seen;
                            new_levels_end = idx as u64 + new_levels_start + 1;
                            break;
                        }
                    }
                    if *def > max_visible_def {
                        tail_invis_seen += 1;
                    }
                }

                if new_end == u64::MAX {
                    new_levels_end = rep.len() as u64;
                    let total_invis_seen = lead_invis_seen + tail_invis_seen;
                    new_end = rep.len() as u64 - total_invis_seen;
                }

                assert_ne!(new_end, u64::MAX);

                // Adjust for any skipped preamble
                if preamble_action == PreambleAction::Skip {
                    new_start += items_in_preamble;
                    new_end += items_in_preamble;
                    new_levels_start += first_row_start;
                    new_levels_end += first_row_start;
                } else if preamble_action == PreambleAction::Take {
                    debug_assert_eq!(new_start, 0);
                    debug_assert_eq!(new_levels_start, 0);
                    new_end += items_in_preamble;
                    new_levels_end += first_row_start;
                }

                debug_assert!(new_end <= total_items);
                (new_start..new_end, new_levels_start..new_levels_end)
            } else {
                // Easy case, there are no invisible items, so we don't need to check for them
                // The items range and levels range will be the same.  We do still need to walk
                // the rep levels to find the row boundaries

                // range.start == 0 always maps to 0, otherwise we need to walk
                if range.start > 0 {
                    for (idx, rep) in rep.iter().skip(1).enumerate() {
                        if *rep == max_rep {
                            rows_seen += 1;
                            if rows_seen == range.start {
                                new_start = idx as u64 + 1;
                                break;
                            }
                        }
                    }
                }
                let mut new_end = rep.len() as u64;
                // range.end == max_items always maps to rep.len(), otherwise we need to walk
                if range.end < total_items {
                    for (idx, rep) in rep[(new_start + 1) as usize..].iter().enumerate() {
                        if *rep == max_rep {
                            rows_seen += 1;
                            if rows_seen == range.end {
                                new_end = idx as u64 + new_start + 1;
                                break;
                            }
                        }
                    }
                }

                // Adjust for any skipped preamble
                if preamble_action == PreambleAction::Skip {
                    new_start += first_row_start;
                    new_end += first_row_start;
                } else if preamble_action == PreambleAction::Take {
                    debug_assert_eq!(new_start, 0);
                    new_end += first_row_start;
                }

                debug_assert!(new_end <= total_items);
                (new_start..new_end, new_start..new_end)
            }
        } else {
            // No repetition info, easy case, just use the range as-is and the item
            // and level ranges are the same
            (range.clone(), range)
        }
    }

    // read `num_buffers` buffer sizes from `buf` starting at `offset`
    fn read_buffer_sizes<const LARGE: bool>(
        buf: &[u8],
        offset: &mut usize,
        num_buffers: u64,
    ) -> Vec<u32> {
        let read_size = if LARGE { 4 } else { 2 };
        (0..num_buffers)
            .map(|_| {
                let bytes = &buf[*offset..*offset + read_size];
                let size = if LARGE {
                    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                } else {
                    // the buffer size is read from u16 but is stored as u32 after decoding for consistency
                    u16::from_le_bytes([bytes[0], bytes[1]]) as u32
                };
                *offset += read_size;
                size
            })
            .collect()
    }

    // Unserialize a miniblock into a collection of vectors
    fn decode_miniblock_chunk(
        &self,
        buf: &LanceBuffer,
        items_in_chunk: u64,
    ) -> Result<DecodedMiniBlockChunk> {
        let mut offset = 0;
        let num_levels = u16::from_le_bytes([buf[offset], buf[offset + 1]]);
        offset += 2;

        let rep_size = if self.rep_decompressor.is_some() {
            let rep_size = u16::from_le_bytes([buf[offset], buf[offset + 1]]);
            offset += 2;
            Some(rep_size)
        } else {
            None
        };
        let def_size = if self.def_decompressor.is_some() {
            let def_size = u16::from_le_bytes([buf[offset], buf[offset + 1]]);
            offset += 2;
            Some(def_size)
        } else {
            None
        };

        let buffer_sizes = if self.has_large_chunk {
            Self::read_buffer_sizes::<true>(buf, &mut offset, self.num_buffers)
        } else {
            Self::read_buffer_sizes::<false>(buf, &mut offset, self.num_buffers)
        };

        offset += pad_bytes::<MINIBLOCK_ALIGNMENT>(offset);

        let rep = rep_size.map(|rep_size| {
            let rep = buf.slice_with_length(offset, rep_size as usize);
            offset += rep_size as usize;
            offset += pad_bytes::<MINIBLOCK_ALIGNMENT>(offset);
            rep
        });

        let def = def_size.map(|def_size| {
            let def = buf.slice_with_length(offset, def_size as usize);
            offset += def_size as usize;
            offset += pad_bytes::<MINIBLOCK_ALIGNMENT>(offset);
            def
        });

        let num_levels = Self::resolve_num_levels(
            self.rep_decompressor.as_deref(),
            rep.as_ref(),
            self.def_decompressor.as_deref(),
            def.as_ref(),
            num_levels,
        )?;

        let buffers = buffer_sizes
            .into_iter()
            .map(|buf_size| {
                let buf = buf.slice_with_length(offset, buf_size as usize);
                offset += buf_size as usize;
                offset += pad_bytes::<MINIBLOCK_ALIGNMENT>(offset);
                buf
            })
            .collect::<Vec<_>>();

        let values = self
            .value_decompressor
            .decompress(buffers, items_in_chunk)?;

        let rep = rep
            .map(|rep| {
                Self::decode_levels(
                    self.rep_decompressor.as_ref().unwrap().as_ref(),
                    rep,
                    num_levels,
                )
            })
            .transpose()?;
        let def = def
            .map(|def| {
                Self::decode_levels(
                    self.def_decompressor.as_ref().unwrap().as_ref(),
                    def,
                    num_levels,
                )
            })
            .transpose()?;

        if let (Some(rep), Some(def)) = (&rep, &def)
            && rep.len() != def.len()
        {
            return Err(Error::invalid_input_source(
                format!(
                    "miniblock structural streams decoded different level counts: repetition {}, definition {}",
                    rep.len(),
                    def.len()
                )
                .into(),
            ));
        }

        Ok(DecodedMiniBlockChunk { rep, def, values })
    }
}

impl DecodePageTask for DecodeMiniBlockTask {
    fn decode(self: Box<Self>) -> Result<DecodedPage> {
        // First, we create output buffers for the rep and def and data
        let mut repbuf: Option<LevelBuffer> = None;
        let mut defbuf: Option<LevelBuffer> = None;

        let max_rep = self.def_meaning.iter().filter(|l| l.is_list()).count() as u16;

        let estimated_size_bytes = self.decoded_size_bytes().unwrap_or_else(|| {
            // Variable-width and rep/def encoded output sizes are not known before decoding.
            self.instructions
                .iter()
                .map(|(_, chunk)| chunk.data.len() as u64)
                .sum::<u64>()
                * 2
        });
        let mut data_builder = DataBlockBuilder::with_capacity_estimate(estimated_size_bytes);

        // We need to keep track of the offset into repbuf/defbuf that we are building up
        let mut level_offset = 0;

        // Pre-compute caching needs for each chunk by checking if the next chunk is the same
        let needs_caching: Vec<bool> = self
            .instructions
            .windows(2)
            .map(|w| w[0].1.chunk_idx == w[1].1.chunk_idx)
            .chain(std::iter::once(false)) // the last one never needs caching
            .collect();

        // Cache for storing decoded chunks when beneficial
        let mut chunk_cache: Option<(usize, DecodedMiniBlockChunk)> = None;

        // Now we iterate through each instruction and process it
        for (idx, (instructions, chunk)) in self.instructions.iter().enumerate() {
            let should_cache_this_chunk = needs_caching[idx];

            let decoded_chunk = match &chunk_cache {
                Some((cached_chunk_idx, cached_chunk)) if *cached_chunk_idx == chunk.chunk_idx => {
                    // Clone only when we have a cache hit (much cheaper than decoding)
                    cached_chunk.clone()
                }
                _ => {
                    // Cache miss, need to decode
                    let decoded = self.decode_miniblock_chunk(&chunk.data, chunk.items_in_chunk)?;

                    // Only update cache if this chunk will benefit the next access
                    if should_cache_this_chunk {
                        chunk_cache = Some((chunk.chunk_idx, decoded.clone()));
                    }
                    decoded
                }
            };

            let DecodedMiniBlockChunk { rep, def, values } = decoded_chunk;

            // Our instructions tell us which rows we want to take from this chunk
            let row_range_start =
                instructions.rows_to_skip + instructions.chunk_instructions.rows_to_skip;
            let row_range_end = row_range_start + instructions.rows_to_take;

            // We use the rep info to map the row range to an item range / levels range
            let (item_range, level_range) = Self::map_range(
                row_range_start..row_range_end,
                rep.as_ref(),
                def.as_ref(),
                max_rep,
                self.max_visible_level,
                chunk.items_in_chunk,
                instructions.preamble_action,
            );
            if item_range.end - item_range.start > chunk.items_in_chunk {
                return Err(lance_core::Error::internal(format!(
                    "Item range {:?} is greater than chunk items in chunk {:?}",
                    item_range, chunk.items_in_chunk
                )));
            }

            // Now we append the data to the output buffers
            Self::extend_levels(level_range.clone(), &mut repbuf, &rep, level_offset);
            Self::extend_levels(level_range.clone(), &mut defbuf, &def, level_offset);
            level_offset += (level_range.end - level_range.start) as usize;
            data_builder.append(&values, item_range)?;
        }

        let mut data = data_builder.finish();

        let unraveler =
            RepDefUnraveler::new(repbuf, defbuf, self.def_meaning.clone(), data.num_values());

        if let Some(dictionary) = &self.dictionary_data {
            // Don't decode here, that happens later (if needed)
            let DataBlock::FixedWidth(indices) = data else {
                return Err(lance_core::Error::internal(format!(
                    "Expected FixedWidth DataBlock for dictionary indices, got {:?}",
                    data
                )));
            };
            data = DataBlock::Dictionary(DictionaryDataBlock::from_parts(
                indices,
                dictionary.as_ref().clone(),
            ));
        }

        Ok(DecodedPage {
            data,
            repdef: unraveler,
        })
    }
}

/// Decodes mini-block formatted data.  See [`PrimitiveStructuralEncoder`] for more
/// details on the different layouts.
///
/// [`PrimitiveStructuralEncoder`]: crate::encodings::logical::primitive::PrimitiveStructuralEncoder
#[derive(Debug)]
struct MiniBlockDecoder {
    rep_decompressor: Option<Arc<dyn BlockDecompressor>>,
    def_decompressor: Option<Arc<dyn BlockDecompressor>>,
    value_decompressor: Arc<dyn MiniBlockDecompressor>,
    def_meaning: Arc<[DefinitionInterpretation]>,
    loaded_chunks: VecDeque<LoadedChunk>,
    instructions: VecDeque<ChunkInstructions>,
    offset_in_current_chunk: u64,
    num_rows: u64,
    num_buffers: u64,
    dictionary: Option<Arc<DataBlock>>,
    has_large_chunk: bool,
}

/// See [`MiniBlockScheduler`] for more details on the scheduling and decoding
/// process for miniblock encoded data.
impl StructuralPageDecoder for MiniBlockDecoder {
    fn drain(&mut self, num_rows: u64) -> Result<Box<dyn DecodePageTask>> {
        let mut items_desired = num_rows;
        let mut need_preamble = false;
        let mut skip_in_chunk = self.offset_in_current_chunk;
        let mut drain_instructions = Vec::new();
        while items_desired > 0 || need_preamble {
            let (instructions, consumed) = self
                .instructions
                .front()
                .unwrap()
                .drain_from_instruction(&mut items_desired, &mut need_preamble, &mut skip_in_chunk);

            while self.loaded_chunks.front().unwrap().chunk_idx
                != instructions.chunk_instructions.chunk_idx
            {
                self.loaded_chunks.pop_front();
            }
            drain_instructions.push((instructions, self.loaded_chunks.front().unwrap().clone()));
            if consumed {
                self.instructions.pop_front();
            }
        }
        // We can throw away need_preamble here because it must be false.  If it were true it would mean
        // we were still in the middle of loading rows.  We do need to latch skip_in_chunk though.
        self.offset_in_current_chunk = skip_in_chunk;

        let max_visible_level = max_visible_level(&self.def_meaning);

        Ok(Box::new(DecodeMiniBlockTask {
            instructions: drain_instructions,
            def_decompressor: self.def_decompressor.clone(),
            rep_decompressor: self.rep_decompressor.clone(),
            value_decompressor: self.value_decompressor.clone(),
            dictionary_data: self.dictionary.clone(),
            def_meaning: self.def_meaning.clone(),
            num_buffers: self.num_buffers,
            max_visible_level,
            has_large_chunk: self.has_large_chunk,
        }))
    }

    fn num_rows(&self) -> u64 {
        self.num_rows
    }
}

#[derive(Debug, Clone)]
struct MiniBlockSchedulerDictionary {
    // These come from the protobuf
    dictionary_decompressor: Arc<dyn BlockDecompressor>,
    dictionary_buf_position_and_size: (u64, u64),
    dictionary_data_alignment: u64,
    num_dictionary_items: u64,
}

/// State that is loaded once and cached for future lookups
#[derive(Debug)]
struct MiniBlockCacheableState {
    /// Compact per-chunk index (byte ranges + row/item mapping) for the page
    chunk_index: MiniBlockChunkIndex,
    /// The dictionary for the page, if any
    dictionary: Option<Arc<DataBlock>>,
}

impl DeepSizeOf for MiniBlockCacheableState {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.chunk_index.deep_size_of_children(context)
            + self
                .dictionary
                .as_ref()
                .map(|dict| dict.data_size() as usize)
                .unwrap_or(0)
    }
}

impl CachedPageData for MiniBlockCacheableState {
    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }
}

/// A scheduler for a page that has been encoded with the mini-block layout
///
/// Scheduling mini-block encoded data is simple in concept and somewhat complex
/// in practice.
///
/// First, during initialization, we load the chunk metadata, the repetition index,
/// and the dictionary (these last two may not be present)
///
/// Then, during scheduling, we use the user's requested row ranges and the repetition
/// index to determine which chunks we need and which rows we need from those chunks.
///
/// For example, if the repetition index is: [50, 3], [50, 0], [10, 0] and the range
/// from the user is 40..60 then we need to:
///
///  - Read the first chunk and skip the first 40 rows, then read 10 full rows, and
///    then read 3 items for the 11th row of our range.
///  - Read the second chunk and read the remaining items in our 11th row and then read
///    the remaining 9 full rows.
///
/// Then, if we are going to decode that in batches of 5, we need to make decode tasks.
/// The first two decode tasks will just need the first chunk.  The third decode task will
/// need the first chunk (for the trailer which has the 11th row in our range) and the second
/// chunk.  The final decode task will just need the second chunk.
///
/// The above prose descriptions are what are represented by `ChunkInstructions` and
/// `ChunkDrainInstructions`.
#[derive(Debug)]
pub struct MiniBlockScheduler {
    // These come from the protobuf
    buffer_offsets_and_sizes: Vec<(u64, u64)>,
    priority: u64,
    items_in_page: u64,
    repetition_index_depth: u16,
    num_buffers: u64,
    rep_decompressor: Option<Arc<dyn BlockDecompressor>>,
    def_decompressor: Option<Arc<dyn BlockDecompressor>>,
    value_decompressor: Arc<dyn MiniBlockDecompressor>,
    def_meaning: Arc<[DefinitionInterpretation]>,
    dictionary: Option<MiniBlockSchedulerDictionary>,
    // This is set after initialization
    page_meta: Option<Arc<MiniBlockCacheableState>>,
    has_large_chunk: bool,
}

impl MiniBlockScheduler {
    pub(super) fn try_new(
        buffer_offsets_and_sizes: &[(u64, u64)],
        priority: u64,
        items_in_page: u64,
        layout: &pb21::MiniBlockLayout,
        decompressors: &dyn DecompressionStrategy,
    ) -> Result<Self> {
        let rep_decompressor = layout
            .rep_compression
            .as_ref()
            .map(|rep_compression| {
                decompressors
                    .create_block_decompressor(rep_compression)
                    .map(Arc::from)
            })
            .transpose()?;
        let def_decompressor = layout
            .def_compression
            .as_ref()
            .map(|def_compression| {
                decompressors
                    .create_block_decompressor(def_compression)
                    .map(Arc::from)
            })
            .transpose()?;
        let def_meaning = layout
            .layers
            .iter()
            .map(|l| ProtobufUtils21::repdef_layer_to_def_interp(*l))
            .collect::<Vec<_>>();
        let value_decompressor = decompressors.create_miniblock_decompressor(
            layout.value_compression.as_ref().unwrap(),
            decompressors,
        )?;

        let dictionary = if let Some(dictionary_encoding) = layout.dictionary.as_ref() {
            let num_dictionary_items = layout.num_dictionary_items;
            let dictionary_decompressor = decompressors
                .create_block_decompressor(dictionary_encoding)?
                .into();
            let dictionary_data_alignment = match dictionary_encoding.compression.as_ref().unwrap()
            {
                Compression::Variable(_) => 4,
                Compression::Flat(_) => 16,
                Compression::General(_) => 1,
                Compression::InlineBitpacking(_) | Compression::OutOfLineBitpacking(_) => {
                    crate::encoder::MIN_PAGE_BUFFER_ALIGNMENT
                }
                _ => {
                    return Err(Error::invalid_input_source(
                        format!(
                            "Unsupported mini-block dictionary encoding: {:?}",
                            dictionary_encoding.compression.as_ref().unwrap()
                        )
                        .into(),
                    ));
                }
            };
            Some(MiniBlockSchedulerDictionary {
                dictionary_decompressor,
                dictionary_buf_position_and_size: buffer_offsets_and_sizes[2],
                dictionary_data_alignment,
                num_dictionary_items,
            })
        } else {
            None
        };

        Ok(Self {
            buffer_offsets_and_sizes: buffer_offsets_and_sizes.to_vec(),
            rep_decompressor,
            def_decompressor,
            value_decompressor: value_decompressor.into(),
            repetition_index_depth: layout.repetition_index_depth as u16,
            num_buffers: layout.num_buffers,
            priority,
            items_in_page,
            dictionary,
            def_meaning: def_meaning.into(),
            page_meta: None,
            has_large_chunk: layout.has_large_chunk,
        })
    }

    fn lookup_chunks(&self, chunk_indices: &[usize]) -> Vec<LoadedChunk> {
        let chunk_index = &self.page_meta.as_ref().unwrap().chunk_index;
        chunk_indices
            .iter()
            .map(|&chunk_idx| LoadedChunk {
                byte_range: chunk_index.byte_range(chunk_idx),
                items_in_chunk: chunk_index.items_in_chunk(chunk_idx),
                chunk_idx,
                data: LanceBuffer::empty(),
            })
            .collect()
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum PreambleAction {
    Take,
    Skip,
    Absent,
}

// When we schedule a chunk we use the repetition index (or, if none exists, just the # of items
// in each chunk) to map a user requested range into a set of ChunkInstruction objects which tell
// us how exactly to read from the chunk.
//
// Examples:
//
// | Chunk 0     | Chunk 1   | Chunk 2   | Chunk 3 |
// | xxxxyyyyzzz | zzzzzzzzz | zzzzzzzzz | aaabbcc |
//
// Full read (0..6)
//
// Chunk 0: (several rows, ends with trailer)
//   preamble: absent
//   rows_to_skip: 0
//   rows_to_take: 3 (x, y, z)
//   take_trailer: true
//
// Chunk 1: (all preamble, ends with trailer)
//   preamble: take
//   rows_to_skip: 0
//   rows_to_take: 0
//   take_trailer: true
//
// Chunk 2: (all preamble, no trailer)
//   preamble: take
//   rows_to_skip: 0
//   rows_to_take: 0
//   take_trailer: false
//
// Chunk 3: (several rows, no trailer or preamble)
//   preamble: absent
//   rows_to_skip: 0
//   rows_to_take: 3 (a, b, c)
//   take_trailer: false
#[derive(Clone, Debug, PartialEq, Eq)]
struct ChunkInstructions {
    // The index of the chunk to read
    chunk_idx: usize,
    // A "preamble" is when a chunk begins with a continuation of the previous chunk's list.  If there
    // is no repetition index there is never a preamble.
    //
    // It's possible for a chunk to be entirely premable.  For example, if there is a really large list
    // that spans several chunks.
    preamble: PreambleAction,
    // How many complete rows (not including the preamble or trailer) to skip
    //
    // If this is non-zero then premable must not be Take
    rows_to_skip: u64,
    // How many rows to take.  If a row splits across chunks then we will count the row in the first
    // chunk that contains the row.
    rows_to_take: u64,
    // A "trailer" is when a chunk ends with a partial list.  If there is no repetition index there is
    // never a trailer.
    //
    // A chunk that is all preamble may or may not have a trailer.
    //
    // If this is true then we want to include the trailer
    take_trailer: bool,
}

// First, we schedule a bunch of [`ChunkInstructions`] based on the users ranges.  Then we
// start decoding them, based on a batch size, which might not align with what we scheduled.
//
// This results in `ChunkDrainInstructions` which targets a contiguous slice of a `ChunkInstructions`
//
// So if `ChunkInstructions` is "skip preamble, skip 10, take 50, take trailer" and we are decoding in
// batches of size 10 we might have a `ChunkDrainInstructions` that targets that chunk and has its own
// skip of 17 and take of 10.  This would mean we decode the chunk, skip the preamble and 27 rows, and
// then take 10 rows.
//
// One very confusing bit is that `rows_to_take` includes the trailer.  So if we have two chunks:
//  -no preamble, skip 5, take 10, take trailer
//  -take preamble, skip 0, take 50, no trailer
//
// and we are draining 20 rows then the drain instructions for the first batch will be:
//  - no preamble, skip 0 (from chunk 0), take 11 (from chunk 0)
//  - take preamble (from chunk 1), skip 0 (from chunk 1), take 9 (from chunk 1)
#[derive(Debug, PartialEq, Eq)]
struct ChunkDrainInstructions {
    chunk_instructions: ChunkInstructions,
    rows_to_skip: u64,
    rows_to_take: u64,
    preamble_action: PreambleAction,
}

impl ChunkInstructions {
    // Given a repetition index and a set of user ranges we need to figure out how to read from the chunks
    //
    // We assume that `user_ranges` are in sorted order and non-overlapping
    //
    // The output will be a set of `ChunkInstructions` which tell us how to read from the chunks
    fn schedule_instructions(
        chunk_index: &MiniBlockChunkIndex,
        user_ranges: &[Range<u64>],
    ) -> Vec<Self> {
        // Bind the per-page chunk count once; re-deriving it each iteration
        // costs a width match plus a length read.
        let num_chunks = chunk_index.num_chunks();
        // This is an in-exact capacity guess but pretty good.  The actual capacity can be
        // smaller if instructions are merged.  It can be larger if there are multiple instructions
        // per row which can happen with lists.
        let mut chunk_instructions = Vec::with_capacity(user_ranges.len());

        for user_range in user_ranges {
            let mut rows_needed = user_range.end - user_range.start;
            let mut need_preamble = false;

            // Need to find the first chunk with a first row >= user_range.start.  If there are
            // multiple chunks with the same first row we need to take the first one.
            let mut block_index = chunk_index.find_chunk(user_range.start);

            let mut to_skip = user_range.start - chunk_index.first_row(block_index);

            while rows_needed > 0 || need_preamble {
                // Check if we've gone past the last block (should not happen)
                if block_index >= num_chunks {
                    log::warn!(
                        "schedule_instructions inconsistency: block_index >= num_chunks, exiting early"
                    );
                    break;
                }

                let starts_including_trailer = chunk_index.rows_in_chunk(block_index);
                let has_preamble = chunk_index.has_preamble(block_index);
                let has_trailer = chunk_index.has_trailer(block_index);
                let rows_avail = starts_including_trailer.saturating_sub(to_skip);

                // Handle blocks that are entirely preamble (rows_avail = 0)
                // These blocks have no rows to take but may have a preamble we need
                // We only look for preamble if to_skip == 0 (we're not skipping rows)
                if rows_avail == 0 && to_skip == 0 {
                    // Only process if this chunk has a preamble we need
                    if has_preamble && need_preamble {
                        chunk_instructions.push(Self {
                            chunk_idx: block_index,
                            preamble: PreambleAction::Take,
                            rows_to_skip: 0,
                            rows_to_take: 0,
                            // We still need to look at has_trailer to distinguish between "all preamble
                            // and row ends at end of chunk" and "all preamble and row bleeds into next
                            // chunk".  Both cases will have 0 rows available.
                            take_trailer: has_trailer,
                        });
                        // Only set need_preamble = false if the chunk has at least one row,
                        // Or we are reaching the last block,
                        // Otherwise, the chunk is entirely preamble and we need the next chunk's preamble too
                        if starts_including_trailer > 0 || block_index == num_chunks - 1 {
                            need_preamble = false;
                        }
                    }
                    // Move to next block
                    block_index += 1;
                    continue;
                }

                // Edge case: if rows_avail == 0 but to_skip > 0
                // This theoretically shouldn't happen (binary search should avoid it)
                // but handle it for safety
                if rows_avail == 0 && to_skip > 0 {
                    // This block doesn't have enough rows to skip, move to next block
                    // Adjust to_skip by the number of rows in this block
                    to_skip -= starts_including_trailer;
                    block_index += 1;
                    continue;
                }

                let rows_to_take = rows_avail.min(rows_needed);
                rows_needed -= rows_to_take;

                let mut take_trailer = false;
                let preamble = if has_preamble {
                    if need_preamble {
                        PreambleAction::Take
                    } else {
                        PreambleAction::Skip
                    }
                } else {
                    PreambleAction::Absent
                };

                // Are we taking the trailer?  If so, make sure we mark that we need the preamble
                if rows_to_take == rows_avail && has_trailer {
                    take_trailer = true;
                    need_preamble = true;
                } else {
                    need_preamble = false;
                };

                chunk_instructions.push(Self {
                    preamble,
                    chunk_idx: block_index,
                    rows_to_skip: to_skip,
                    rows_to_take,
                    take_trailer,
                });

                to_skip = 0;
                block_index += 1;
            }
        }

        // If there were multiple ranges we may have multiple instructions for a single chunk.  Merge them now if they
        // are _adjacent_ (i.e. don't merge "take first row of chunk 0" and "take third row of chunk 0" into "take 2
        // rows of chunk 0 starting at 0")
        if user_ranges.len() > 1 {
            // Merge adjacent instructions in place.  `write` indexes the last
            // retained instruction; each following instruction is either folded
            // into it (contiguous within the same chunk) or compacted forward.
            let mut write = 0;
            for read in 1..chunk_instructions.len() {
                let merges = {
                    let last = &chunk_instructions[write];
                    let candidate = &chunk_instructions[read];
                    last.chunk_idx == candidate.chunk_idx
                        && last.rows_to_take + last.rows_to_skip == candidate.rows_to_skip
                };
                if merges {
                    let rows_to_take = chunk_instructions[read].rows_to_take;
                    let take_trailer = chunk_instructions[read].take_trailer;
                    let last = &mut chunk_instructions[write];
                    last.rows_to_take += rows_to_take;
                    last.take_trailer |= take_trailer;
                } else {
                    write += 1;
                    if write != read {
                        chunk_instructions.swap(write, read);
                    }
                }
            }
            chunk_instructions.truncate(write + 1);
        }
        chunk_instructions
    }

    fn drain_from_instruction(
        &self,
        rows_desired: &mut u64,
        need_preamble: &mut bool,
        skip_in_chunk: &mut u64,
    ) -> (ChunkDrainInstructions, bool) {
        // If we need the premable then we shouldn't be skipping anything
        debug_assert!(!*need_preamble || *skip_in_chunk == 0);
        let rows_avail = self.rows_to_take - *skip_in_chunk;
        let has_preamble = self.preamble != PreambleAction::Absent;
        let preamble_action = match (*need_preamble, has_preamble) {
            (true, true) => PreambleAction::Take,
            (true, false) => panic!("Need preamble but there isn't one"),
            (false, true) => PreambleAction::Skip,
            (false, false) => PreambleAction::Absent,
        };

        // How many rows are we actually taking in this take step (including the preamble
        // and trailer both as individual rows)
        let rows_taking = if *rows_desired >= rows_avail {
            // We want all the rows.  If there is a trailer we are grabbing it and will need
            // the preamble of the next chunk
            // If there is a trailer and we are taking all the rows then we need the preamble
            // of the next chunk.
            //
            // Also, if this chunk is entirely preamble (rows_avail == 0 && !take_trailer) then we
            // need the preamble of the next chunk.
            *need_preamble = self.take_trailer;
            rows_avail
        } else {
            // We aren't taking all the rows.  Even if there is a trailer we aren't taking
            // it so we will not need the preamble
            *need_preamble = false;
            *rows_desired
        };
        let rows_skipped = *skip_in_chunk;

        // Update the state for the next iteration
        let consumed_chunk = if *rows_desired >= rows_avail {
            *rows_desired -= rows_avail;
            *skip_in_chunk = 0;
            true
        } else {
            *skip_in_chunk += *rows_desired;
            *rows_desired = 0;
            false
        };

        (
            ChunkDrainInstructions {
                chunk_instructions: self.clone(),
                rows_to_skip: rows_skipped,
                rows_to_take: rows_taking,
                preamble_action,
            },
            consumed_chunk,
        )
    }
}

enum Words {
    U16(ScalarBuffer<u16>),
    U32(ScalarBuffer<u32>),
}

struct WordsIter<'a> {
    iter: Box<dyn Iterator<Item = u32> + 'a>,
}

impl Words {
    pub fn len(&self) -> usize {
        match self {
            Self::U16(b) => b.len(),
            Self::U32(b) => b.len(),
        }
    }

    pub fn iter(&self) -> WordsIter<'_> {
        match self {
            Self::U16(buf) => WordsIter {
                iter: Box::new(buf.iter().map(|&x| x as u32)),
            },
            Self::U32(buf) => WordsIter {
                iter: Box::new(buf.iter().copied()),
            },
        }
    }

    pub fn from_bytes(bytes: Bytes, has_large_chunk: bool) -> Result<Self> {
        let bytes_per_value = if has_large_chunk { 4 } else { 2 };
        assert_eq!(bytes.len() % bytes_per_value, 0);
        let buffer = LanceBuffer::from_bytes(bytes, bytes_per_value as u64);
        if has_large_chunk {
            Ok(Self::U32(buffer.borrow_to_typed_slice::<u32>()))
        } else {
            Ok(Self::U16(buffer.borrow_to_typed_slice::<u16>()))
        }
    }
}

impl<'a> Iterator for WordsIter<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

/// Per-chunk leaf value-count analysis derived from the metadata words.
///
/// `values_per_chunk` is the count shared by every non-last chunk (meaningful
/// when `uniform`), and `last_chunk_values` is the final chunk's count.
struct FlatValueCounts {
    logs: Vec<u8>,
    uniform: bool,
    values_per_chunk: u64,
    last_chunk_values: u64,
}

fn analyze_value_counts(words: &Words, items_in_page: u64) -> Result<FlatValueCounts> {
    let num_chunks = words.len();
    let logs = words.iter().map(|w| (w & 0x0F) as u8).collect::<Vec<_>>();
    let mut counted = 0u64;
    for (chunk_index, &log) in logs.iter().take(num_chunks.saturating_sub(1)).enumerate() {
        if log == 0 {
            return Err(Error::corrupt_file_named(
                "miniblock_metadata",
                format!(
                    "non-final chunk {chunk_index} of {num_chunks} has invalid log_num_values=0"
                ),
            ));
        }
        counted = counted.checked_add(1u64 << log).ok_or_else(|| {
            Error::corrupt_file_named(
                "miniblock_metadata",
                format!(
                    "value count overflow at chunk {chunk_index}: counted_values={counted}, \
                     log_num_values={log}, items_in_page={items_in_page}"
                ),
            )
        })?;
    }
    let last_chunk_values = items_in_page.checked_sub(counted).ok_or_else(|| {
        Error::corrupt_file_named(
            "miniblock_metadata",
            format!(
                "non-final chunks account for counted_values={counted}, exceeding \
                 items_in_page={items_in_page}"
            ),
        )
    })?;
    if let Some(&last_log) = logs.last()
        && last_log != 0
        && (1u64 << last_log) != last_chunk_values
    {
        return Err(Error::corrupt_file_named(
            "miniblock_metadata",
            format!(
                "final chunk log_num_values={last_log} does not match \
                 last_chunk_values={last_chunk_values}: counted_values={counted}, \
                 items_in_page={items_in_page}"
            ),
        ));
    }
    let uniform = num_chunks <= 1 || logs[..num_chunks - 1].iter().all(|&log| log == logs[0]);
    // A single-chunk page has no "non-last" chunk to derive a stride from; use the
    // page item count (min 1 so it stays a valid divisor in `find_chunk`).
    let values_per_chunk = if num_chunks <= 1 {
        items_in_page.max(1)
    } else {
        1u64 << logs[0]
    };
    Ok(FlatValueCounts {
        logs,
        uniform,
        values_per_chunk,
        last_chunk_values,
    })
}

/// Iterator over per-chunk value counts for a non-uniform flat page.  Non-last
/// chunks yield `1 << log`; the last yields the validated remaining item count.
fn flat_value_counts_iter(logs: &[u8], last_chunk_values: u64) -> impl Iterator<Item = u64> + '_ {
    let num_chunks = logs.len();
    (0..num_chunks).map(move |i| {
        if i + 1 < num_chunks {
            1u64 << logs[i]
        } else {
            last_chunk_values
        }
    })
}

/// Builds the compact per-chunk index from the metadata words and, for nested
/// pages, the raw repetition-index bytes.  The row axis is picked by page shape:
/// `UniformFlat` when all non-last chunks share a value count (fixed-width /
/// bitpacking), `Flat` for non-uniform flat pages (RLE / FSST), else `Nested`.
fn build_chunk_index(
    words: &Words,
    items_in_page: u64,
    base: u64,
    data_buf_size: u64,
    rep_index_bytes: Option<&[u8]>,
    repetition_index_depth: u16,
) -> Result<MiniBlockChunkIndex> {
    let num_chunks = words.len();
    // Validate item counts before byte sizes because both share a metadata word,
    // and an invalid count must not reach the final-chunk subtraction.
    let value_counts = analyze_value_counts(words, items_in_page)?;

    // Each chunk stores `(divided_bytes + 1) * MINIBLOCK_ALIGNMENT` bytes, so the
    // deltas are the chunk sizes and their grand total is the data buffer size.
    let byte_starts = PrefixSums::from_deltas(
        words
            .iter()
            .map(|word| ((word >> 4) as u64 + 1) * MINIBLOCK_ALIGNMENT as u64),
        num_chunks,
        data_buf_size,
    );

    // Nested pages track rows via the repetition index and keep leaf item counts
    // separately; flat pages have row == value index, so value counts are rows.
    let rows = if let Some(rep_index_data) = rep_index_bytes {
        assert!(rep_index_data.len() % 8 == 0);
        let stride = repetition_index_depth as usize + 1;
        let (row_starts, has_trailer) = parse_nested_rep(rep_index_data, stride);
        let item_counts = if value_counts.uniform {
            ItemCounts::Uniform {
                values_per_chunk: value_counts.values_per_chunk,
                last_chunk_values: value_counts.last_chunk_values,
            }
        } else {
            ItemCounts::PerChunkLog {
                logs: value_counts.logs,
                last_chunk_values: value_counts.last_chunk_values,
            }
        };
        RowMapping::Nested {
            row_starts,
            has_trailer,
            item_counts,
        }
    } else {
        if value_counts.uniform {
            RowMapping::UniformFlat {
                values_per_chunk: value_counts.values_per_chunk,
                last_chunk_values: value_counts.last_chunk_values,
                num_chunks,
            }
        } else {
            let value_starts = PrefixSums::from_deltas(
                flat_value_counts_iter(&value_counts.logs, value_counts.last_chunk_values),
                num_chunks,
                items_in_page,
            );
            RowMapping::Flat { value_starts }
        }
    };

    Ok(MiniBlockChunkIndex::new(base, byte_starts, rows))
}

impl StructuralPageScheduler for MiniBlockScheduler {
    fn init_layout(&self) -> Result<PageInitialization> {
        // We always need to fetch chunk metadata.  We may also need to fetch a dictionary and
        // we may also need to fetch the repetition index.  Here, we gather what buffers we
        // need.
        let (meta_buf_position, meta_buf_size) = self.buffer_offsets_and_sizes[0];
        let dictionary_range = self.dictionary.as_ref().map(|dictionary| {
            dictionary.dictionary_buf_position_and_size.0
                ..dictionary.dictionary_buf_position_and_size.0
                    + dictionary.dictionary_buf_position_and_size.1
        });
        let repetition_index_range = if self.repetition_index_depth > 0 {
            let (rep_index_pos, rep_index_size) =
                self.buffer_offsets_and_sizes.last().ok_or_else(|| {
                    Error::invalid_input_source(
                        "Mini-block page has a repetition index but no buffers".into(),
                    )
                })?;
            Some(*rep_index_pos..*rep_index_pos + *rep_index_size)
        } else {
            None
        };
        Ok(PageInitialization::new([
            Some(meta_buf_position..meta_buf_position + meta_buf_size),
            dictionary_range,
            repetition_index_range,
        ]))
    }

    fn init_from_buffers<'a>(
        &'a mut self,
        buffers: PageInitializationBuffers,
        _io: &Arc<dyn EncodingsIo>,
    ) -> BoxFuture<'a, Result<Arc<dyn CachedPageData>>> {
        let base = self.buffer_offsets_and_sizes[1].0;
        let data_buf_size = self.buffer_offsets_and_sizes[1].1;

        async move {
            let [meta_bytes, dictionary_bytes, rep_index_bytes] =
                buffers.try_into_array("mini-block")?;
            let meta_bytes = meta_bytes.ok_or_else(|| {
                Error::internal("Mini-block page initialization is missing chunk metadata")
            })?;

            let words = Words::from_bytes(meta_bytes, self.has_large_chunk)?;
            let chunk_index = build_chunk_index(
                &words,
                self.items_in_page,
                base,
                data_buf_size,
                rep_index_bytes.as_deref(),
                self.repetition_index_depth,
            )?;

            // decode dictionary
            let dictionary = match (self.dictionary.as_ref(), dictionary_bytes) {
                (Some(dictionary), Some(dictionary_data)) => {
                    Some(Arc::new(dictionary.dictionary_decompressor.decompress(
                        Some(LanceBuffer::from_bytes(
                            dictionary_data,
                            dictionary.dictionary_data_alignment,
                        )),
                        dictionary.num_dictionary_items,
                    )?))
                }
                (None, None) => None,
                _ => {
                    return Err(Error::internal(
                        "Mini-block page initialization dictionary slot does not match its layout",
                    ));
                }
            };

            let page_meta = Arc::new(MiniBlockCacheableState {
                chunk_index,
                dictionary,
            });
            Ok(page_meta as Arc<dyn CachedPageData>)
        }
        .boxed()
    }

    fn try_load(&mut self, data: &Arc<dyn CachedPageData>) -> Result<()> {
        self.page_meta = Some(data.clone().as_arc_any().downcast().map_err(|_| {
            Error::invalid_input_source("Cached mini-block page data has an unexpected type".into())
        })?);
        Ok(())
    }

    fn schedule_ranges(
        &self,
        ranges: &[Range<u64>],
        io: &Arc<dyn EncodingsIo>,
    ) -> Result<Vec<PageLoadTask>> {
        let num_rows = ranges.iter().map(|r| r.end - r.start).sum();

        let page_meta = self.page_meta.as_ref().unwrap();

        let chunk_instructions =
            ChunkInstructions::schedule_instructions(&page_meta.chunk_index, ranges);

        debug_assert_eq!(
            num_rows,
            chunk_instructions
                .iter()
                .map(|ci| ci.rows_to_take)
                .sum::<u64>()
        );

        let chunks_needed = chunk_instructions
            .iter()
            .map(|ci| ci.chunk_idx)
            .unique()
            .collect::<Vec<_>>();

        let mut loaded_chunks = self.lookup_chunks(&chunks_needed);
        let chunk_ranges = loaded_chunks
            .iter()
            .map(|c| c.byte_range.clone())
            .collect::<Vec<_>>();
        let loaded_chunk_data = io.submit_request(chunk_ranges, self.priority);

        let rep_decompressor = self.rep_decompressor.clone();
        let def_decompressor = self.def_decompressor.clone();
        let value_decompressor = self.value_decompressor.clone();
        let num_buffers = self.num_buffers;
        let has_large_chunk = self.has_large_chunk;
        let dictionary = page_meta
            .dictionary
            .as_ref()
            .map(|dictionary| dictionary.clone());
        let def_meaning = self.def_meaning.clone();

        let res = async move {
            let loaded_chunk_data = loaded_chunk_data.await?;
            for (loaded_chunk, chunk_data) in loaded_chunks.iter_mut().zip(loaded_chunk_data) {
                loaded_chunk.data = LanceBuffer::from_bytes(chunk_data, 1);
            }

            Ok(Box::new(MiniBlockDecoder {
                rep_decompressor,
                def_decompressor,
                value_decompressor,
                def_meaning,
                loaded_chunks: VecDeque::from_iter(loaded_chunks),
                instructions: VecDeque::from(chunk_instructions),
                offset_in_current_chunk: 0,
                dictionary,
                num_rows,
                num_buffers,
                has_large_chunk,
            }) as Box<dyn StructuralPageDecoder>)
        }
        .boxed();
        let page_load_task = PageLoadTask {
            decoder_fut: res,
            num_rows,
        };
        Ok(vec![page_load_task])
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Arc;

    use bytes::Bytes;
    use lance_core::cache::{Context, DeepSizeOf};
    use rstest::rstest;

    use crate::buffer::LanceBuffer;
    use crate::compression::{DefaultDecompressionStrategy, MiniBlockDecompressor};
    use crate::data::{BlockInfo, DataBlock, FixedWidthDataBlock};
    use crate::decoder::DecodePageTask;
    use crate::encodings::logical::primitive::chunk_index::{
        MiniBlockChunkIndex, PrefixSums, RowMapping,
    };
    use crate::encodings::logical::primitive::tests::encoded_u16_frame;
    use crate::encodings::logical::primitive::{LoadedChunk, MINIBLOCK_ALIGNMENT};
    use crate::encodings::physical::rle::{RleDecompressor, RunLengthWidth};
    use crate::encodings::physical::value::ValueDecompressor;
    use crate::format::{ProtobufUtils21, pb21};

    use super::{
        ChunkDrainInstructions, ChunkInstructions, DecodeMiniBlockTask, PreambleAction, Words,
        build_chunk_index,
    };

    #[test]
    fn test_miniblock_decode_uses_exact_fixed_width_output_size() {
        #[derive(Debug)]
        struct FixedWidthMiniBlockDecompressor;

        impl MiniBlockDecompressor for FixedWidthMiniBlockDecompressor {
            fn decompress(
                &self,
                data: Vec<LanceBuffer>,
                num_values: u64,
            ) -> crate::Result<DataBlock> {
                assert_eq!(data.len(), 1);
                Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
                    data: data.into_iter().next().unwrap(),
                    bits_per_value: 32,
                    num_values,
                    block_info: BlockInfo::new(),
                }))
            }

            fn decoded_size_bytes(&self, num_values: u64) -> Option<u64> {
                num_values.checked_mul(4)
            }
        }

        let num_rows = 512;
        let expected_size = num_rows * 4;
        let mut chunk_data = Vec::new();
        chunk_data.extend_from_slice(&0_u16.to_le_bytes());
        chunk_data.extend_from_slice(&(expected_size as u16).to_le_bytes());
        let header_padding =
            lance_core::utils::bit::pad_bytes::<{ super::MINIBLOCK_ALIGNMENT }>(chunk_data.len());
        chunk_data.resize(chunk_data.len() + header_padding, 0);
        chunk_data.resize(chunk_data.len() + expected_size as usize, 7);

        let task = DecodeMiniBlockTask {
            rep_decompressor: None,
            def_decompressor: None,
            value_decompressor: Arc::new(FixedWidthMiniBlockDecompressor),
            dictionary_data: None,
            def_meaning: Arc::from([]),
            num_buffers: 1,
            max_visible_level: 0,
            instructions: vec![(
                ChunkDrainInstructions {
                    chunk_instructions: ChunkInstructions {
                        chunk_idx: 0,
                        preamble: PreambleAction::Absent,
                        rows_to_skip: 0,
                        rows_to_take: num_rows,
                        take_trailer: false,
                    },
                    rows_to_skip: 0,
                    rows_to_take: num_rows,
                    preamble_action: PreambleAction::Absent,
                },
                LoadedChunk {
                    byte_range: 0..chunk_data.len() as u64,
                    data: LanceBuffer::from(chunk_data),
                    items_in_chunk: num_rows,
                    chunk_idx: 0,
                },
            )],
            has_large_chunk: false,
        };

        let decoded = Box::new(task).decode().unwrap();
        let values = decoded.data.as_fixed_width_ref().unwrap();
        assert_eq!(values.data.len(), expected_size as usize);
        assert_eq!(
            values.data.clone().into_buffer().capacity(),
            expected_size as usize
        );
    }

    #[test]
    fn test_map_range() {
        // Null in the middle
        // [[A, B, C], [D, E], NULL, [F, G, H]]
        let rep = Some(vec![1, 0, 0, 1, 0, 1, 1, 0, 0]);
        let def = Some(vec![0, 0, 0, 0, 0, 1, 0, 0, 0]);
        let max_visible_def = 0;
        let total_items = 8;
        let max_rep = 1;

        let check = |range, expected_item_range, expected_level_range| {
            let (item_range, level_range) = DecodeMiniBlockTask::map_range(
                range,
                rep.as_ref(),
                def.as_ref(),
                max_rep,
                max_visible_def,
                total_items,
                PreambleAction::Absent,
            );
            assert_eq!(item_range, expected_item_range);
            assert_eq!(level_range, expected_level_range);
        };

        check(0..1, 0..3, 0..3);
        check(1..2, 3..5, 3..5);
        check(2..3, 5..5, 5..6);
        check(3..4, 5..8, 6..9);
        check(0..2, 0..5, 0..5);
        check(1..3, 3..5, 3..6);
        check(2..4, 5..8, 5..9);
        check(0..3, 0..5, 0..6);
        check(1..4, 3..8, 3..9);
        check(0..4, 0..8, 0..9);

        // Null at start
        // [NULL, [A, B], [C]]
        let rep = Some(vec![1, 1, 0, 1]);
        let def = Some(vec![1, 0, 0, 0]);
        let max_visible_def = 0;
        let total_items = 3;

        let check = |range, expected_item_range, expected_level_range| {
            let (item_range, level_range) = DecodeMiniBlockTask::map_range(
                range,
                rep.as_ref(),
                def.as_ref(),
                max_rep,
                max_visible_def,
                total_items,
                PreambleAction::Absent,
            );
            assert_eq!(item_range, expected_item_range);
            assert_eq!(level_range, expected_level_range);
        };

        check(0..1, 0..0, 0..1);
        check(1..2, 0..2, 1..3);
        check(2..3, 2..3, 3..4);
        check(0..2, 0..2, 0..3);
        check(1..3, 0..3, 1..4);
        check(0..3, 0..3, 0..4);

        // Null at end
        // [[A], [B, C], NULL]
        let rep = Some(vec![1, 1, 0, 1]);
        let def = Some(vec![0, 0, 0, 1]);
        let max_visible_def = 0;
        let total_items = 3;

        let check = |range, expected_item_range, expected_level_range| {
            let (item_range, level_range) = DecodeMiniBlockTask::map_range(
                range,
                rep.as_ref(),
                def.as_ref(),
                max_rep,
                max_visible_def,
                total_items,
                PreambleAction::Absent,
            );
            assert_eq!(item_range, expected_item_range);
            assert_eq!(level_range, expected_level_range);
        };

        check(0..1, 0..1, 0..1);
        check(1..2, 1..3, 1..3);
        check(2..3, 3..3, 3..4);
        check(0..2, 0..3, 0..3);
        check(1..3, 1..3, 1..4);
        check(0..3, 0..3, 0..4);

        // No nulls, with repetition
        // [[A, B], [C, D], [E, F]]
        let rep = Some(vec![1, 0, 1, 0, 1, 0]);
        let def: Option<&[u16]> = None;
        let max_visible_def = 0;
        let total_items = 6;

        let check = |range, expected_item_range, expected_level_range| {
            let (item_range, level_range) = DecodeMiniBlockTask::map_range(
                range,
                rep.as_ref(),
                def.as_ref(),
                max_rep,
                max_visible_def,
                total_items,
                PreambleAction::Absent,
            );
            assert_eq!(item_range, expected_item_range);
            assert_eq!(level_range, expected_level_range);
        };

        check(0..1, 0..2, 0..2);
        check(1..2, 2..4, 2..4);
        check(2..3, 4..6, 4..6);
        check(0..2, 0..4, 0..4);
        check(1..3, 2..6, 2..6);
        check(0..3, 0..6, 0..6);

        // No repetition, with nulls (this case is trivial)
        // [A, B, NULL, C]
        let rep: Option<&[u16]> = None;
        let def = Some(vec![0, 0, 1, 0]);
        let max_visible_def = 1;
        let total_items = 4;

        let check = |range, expected_item_range, expected_level_range| {
            let (item_range, level_range) = DecodeMiniBlockTask::map_range(
                range,
                rep.as_ref(),
                def.as_ref(),
                max_rep,
                max_visible_def,
                total_items,
                PreambleAction::Absent,
            );
            assert_eq!(item_range, expected_item_range);
            assert_eq!(level_range, expected_level_range);
        };

        check(0..1, 0..1, 0..1);
        check(1..2, 1..2, 1..2);
        check(2..3, 2..3, 2..3);
        check(0..2, 0..2, 0..2);
        check(1..3, 1..3, 1..3);
        check(0..3, 0..3, 0..3);

        // Tricky case, this chunk is a continuation and starts with a rep-index = 0
        // [[..., A] [B, C], NULL]
        //
        // What we do will depend on the preamble action
        let rep = Some(vec![0, 1, 0, 1]);
        let def = Some(vec![0, 0, 0, 1]);
        let max_visible_def = 0;
        let total_items = 3;

        let check = |range, expected_item_range, expected_level_range| {
            let (item_range, level_range) = DecodeMiniBlockTask::map_range(
                range,
                rep.as_ref(),
                def.as_ref(),
                max_rep,
                max_visible_def,
                total_items,
                PreambleAction::Take,
            );
            assert_eq!(item_range, expected_item_range);
            assert_eq!(level_range, expected_level_range);
        };

        // If we are taking the preamble then the range must start at 0
        check(0..1, 0..3, 0..3);
        check(0..2, 0..3, 0..4);

        let check = |range, expected_item_range, expected_level_range| {
            let (item_range, level_range) = DecodeMiniBlockTask::map_range(
                range,
                rep.as_ref(),
                def.as_ref(),
                max_rep,
                max_visible_def,
                total_items,
                PreambleAction::Skip,
            );
            assert_eq!(item_range, expected_item_range);
            assert_eq!(level_range, expected_level_range);
        };

        check(0..1, 1..3, 1..3);
        check(1..2, 3..3, 3..4);
        check(0..2, 1..3, 1..4);

        // Another preamble case but now it doesn't end with a new list
        // [[..., A], NULL, [D, E]]
        //
        // What we do will depend on the preamble action
        let rep = Some(vec![0, 1, 1, 0]);
        let def = Some(vec![0, 1, 0, 0]);
        let max_visible_def = 0;
        let total_items = 4;

        let check = |range, expected_item_range, expected_level_range| {
            let (item_range, level_range) = DecodeMiniBlockTask::map_range(
                range,
                rep.as_ref(),
                def.as_ref(),
                max_rep,
                max_visible_def,
                total_items,
                PreambleAction::Take,
            );
            assert_eq!(item_range, expected_item_range);
            assert_eq!(level_range, expected_level_range);
        };

        // If we are taking the preamble then the range must start at 0
        check(0..1, 0..1, 0..2);
        check(0..2, 0..3, 0..4);

        let check = |range, expected_item_range, expected_level_range| {
            let (item_range, level_range) = DecodeMiniBlockTask::map_range(
                range,
                rep.as_ref(),
                def.as_ref(),
                max_rep,
                max_visible_def,
                total_items,
                PreambleAction::Skip,
            );
            assert_eq!(item_range, expected_item_range);
            assert_eq!(level_range, expected_level_range);
        };

        // If we are taking the preamble then the range must start at 0
        check(0..1, 1..1, 1..2);
        check(1..2, 1..3, 2..4);
        check(0..2, 1..3, 1..4);

        // Now a preamble case without any definition levels
        // [[..., A] [B, C], [D]]
        let rep = Some(vec![0, 1, 0, 1]);
        let def: Option<Vec<u16>> = None;
        let max_visible_def = 0;
        let total_items = 4;

        let check = |range, expected_item_range, expected_level_range| {
            let (item_range, level_range) = DecodeMiniBlockTask::map_range(
                range,
                rep.as_ref(),
                def.as_ref(),
                max_rep,
                max_visible_def,
                total_items,
                PreambleAction::Take,
            );
            assert_eq!(item_range, expected_item_range);
            assert_eq!(level_range, expected_level_range);
        };

        // If we are taking the preamble then the range must start at 0
        check(0..1, 0..3, 0..3);
        check(0..2, 0..4, 0..4);

        let check = |range, expected_item_range, expected_level_range| {
            let (item_range, level_range) = DecodeMiniBlockTask::map_range(
                range,
                rep.as_ref(),
                def.as_ref(),
                max_rep,
                max_visible_def,
                total_items,
                PreambleAction::Skip,
            );
            assert_eq!(item_range, expected_item_range);
            assert_eq!(level_range, expected_level_range);
        };

        check(0..1, 1..3, 1..3);
        check(1..2, 3..4, 3..4);
        check(0..2, 1..4, 1..4);

        // If we have nested lists then non-top level lists may be empty/null
        // and we need to make sure we still handle them as invisible items (we
        // failed to do this previously)
        let rep = Some(vec![2, 1, 2, 0, 1, 2]);
        let def = Some(vec![0, 1, 2, 0, 0, 0]);
        let max_rep = 2;
        let max_visible_def = 0;
        let total_items = 4;

        let check = |range, expected_item_range, expected_level_range| {
            let (item_range, level_range) = DecodeMiniBlockTask::map_range(
                range,
                rep.as_ref(),
                def.as_ref(),
                max_rep,
                max_visible_def,
                total_items,
                PreambleAction::Absent,
            );
            assert_eq!(item_range, expected_item_range);
            assert_eq!(level_range, expected_level_range);
        };

        check(0..3, 0..4, 0..6);
        check(0..1, 0..1, 0..2);
        check(1..2, 1..3, 2..5);
        check(2..3, 3..4, 5..6);

        // Invisible items in a preamble that we are taking (regressing a previous failure)
        let rep = Some(vec![0, 0, 1, 0, 1, 1]);
        let def = Some(vec![0, 1, 0, 0, 0, 0]);
        let max_rep = 1;
        let max_visible_def = 0;
        let total_items = 5;

        let check = |range, expected_item_range, expected_level_range| {
            let (item_range, level_range) = DecodeMiniBlockTask::map_range(
                range,
                rep.as_ref(),
                def.as_ref(),
                max_rep,
                max_visible_def,
                total_items,
                PreambleAction::Take,
            );
            assert_eq!(item_range, expected_item_range);
            assert_eq!(level_range, expected_level_range);
        };

        check(0..0, 0..1, 0..2);
        check(0..1, 0..3, 0..4);
        check(0..2, 0..4, 0..5);

        // Skip preamble (with invis items) and skip a few rows (with invis items)
        // and then take a few rows but not all the rows
        let rep = Some(vec![0, 1, 0, 1, 0, 1, 0, 1]);
        let def = Some(vec![1, 0, 1, 1, 0, 0, 0, 0]);
        let max_rep = 1;
        let max_visible_def = 0;
        let total_items = 5;

        let check = |range, expected_item_range, expected_level_range| {
            let (item_range, level_range) = DecodeMiniBlockTask::map_range(
                range,
                rep.as_ref(),
                def.as_ref(),
                max_rep,
                max_visible_def,
                total_items,
                PreambleAction::Skip,
            );
            assert_eq!(item_range, expected_item_range);
            assert_eq!(level_range, expected_level_range);
        };

        check(2..3, 2..4, 5..7);
    }

    #[test]
    fn test_schedule_instructions() {
        // Convert repetition index to bytes for testing
        let rep_data: Vec<u64> = vec![5, 2, 3, 0, 4, 7, 2, 0];
        let rep_bytes: Vec<u8> = rep_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let chunk_index = MiniBlockChunkIndex::new_nested_for_test(&rep_bytes, 2);

        let check = |user_ranges, expected_instructions| {
            let instructions = ChunkInstructions::schedule_instructions(&chunk_index, user_ranges);
            assert_eq!(instructions, expected_instructions);
        };

        // The instructions we expect if we're grabbing the whole range
        let expected_take_all = vec![
            ChunkInstructions {
                chunk_idx: 0,
                preamble: PreambleAction::Absent,
                rows_to_skip: 0,
                rows_to_take: 6,
                take_trailer: true,
            },
            ChunkInstructions {
                chunk_idx: 1,
                preamble: PreambleAction::Take,
                rows_to_skip: 0,
                rows_to_take: 2,
                take_trailer: false,
            },
            ChunkInstructions {
                chunk_idx: 2,
                preamble: PreambleAction::Absent,
                rows_to_skip: 0,
                rows_to_take: 5,
                take_trailer: true,
            },
            ChunkInstructions {
                chunk_idx: 3,
                preamble: PreambleAction::Take,
                rows_to_skip: 0,
                rows_to_take: 1,
                take_trailer: false,
            },
        ];

        // Take all as 1 range
        check(&[0..14], expected_take_all.clone());

        // Take all a individual rows
        check(
            &[
                0..1,
                1..2,
                2..3,
                3..4,
                4..5,
                5..6,
                6..7,
                7..8,
                8..9,
                9..10,
                10..11,
                11..12,
                12..13,
                13..14,
            ],
            expected_take_all,
        );

        // Test some partial takes

        // 2 rows in the same chunk but not contiguous
        check(
            &[0..1, 3..4],
            vec![
                ChunkInstructions {
                    chunk_idx: 0,
                    preamble: PreambleAction::Absent,
                    rows_to_skip: 0,
                    rows_to_take: 1,
                    take_trailer: false,
                },
                ChunkInstructions {
                    chunk_idx: 0,
                    preamble: PreambleAction::Absent,
                    rows_to_skip: 3,
                    rows_to_take: 1,
                    take_trailer: false,
                },
            ],
        );

        // Taking just a trailer/preamble
        check(
            &[5..6],
            vec![
                ChunkInstructions {
                    chunk_idx: 0,
                    preamble: PreambleAction::Absent,
                    rows_to_skip: 5,
                    rows_to_take: 1,
                    take_trailer: true,
                },
                ChunkInstructions {
                    chunk_idx: 1,
                    preamble: PreambleAction::Take,
                    rows_to_skip: 0,
                    rows_to_take: 0,
                    take_trailer: false,
                },
            ],
        );

        // Skipping an entire chunk
        check(
            &[7..10],
            vec![
                ChunkInstructions {
                    chunk_idx: 1,
                    preamble: PreambleAction::Skip,
                    rows_to_skip: 1,
                    rows_to_take: 1,
                    take_trailer: false,
                },
                ChunkInstructions {
                    chunk_idx: 2,
                    preamble: PreambleAction::Absent,
                    rows_to_skip: 0,
                    rows_to_take: 2,
                    take_trailer: false,
                },
            ],
        );
    }

    #[test]
    fn test_drain_instructions() {
        fn drain_from_instructions(
            instructions: &mut VecDeque<ChunkInstructions>,
            mut rows_desired: u64,
            need_preamble: &mut bool,
            skip_in_chunk: &mut u64,
        ) -> Vec<ChunkDrainInstructions> {
            // Note: instructions.len() is an upper bound, we typically take much fewer
            let mut drain_instructions = Vec::with_capacity(instructions.len());
            while rows_desired > 0 || *need_preamble {
                let (next_instructions, consumed_chunk) = instructions
                    .front()
                    .unwrap()
                    .drain_from_instruction(&mut rows_desired, need_preamble, skip_in_chunk);
                if consumed_chunk {
                    instructions.pop_front();
                }
                drain_instructions.push(next_instructions);
            }
            drain_instructions
        }

        // Convert repetition index to bytes for testing
        let rep_data: Vec<u64> = vec![5, 2, 3, 0, 4, 7, 2, 0];
        let rep_bytes: Vec<u8> = rep_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let chunk_index = MiniBlockChunkIndex::new_nested_for_test(&rep_bytes, 2);
        let user_ranges = vec![1..7, 10..14];

        // First, schedule the ranges
        let scheduled = ChunkInstructions::schedule_instructions(&chunk_index, &user_ranges);

        let mut to_drain = VecDeque::from(scheduled.clone());

        // Now we drain in batches of 4

        let mut need_preamble = false;
        let mut skip_in_chunk = 0;

        let next_batch =
            drain_from_instructions(&mut to_drain, 4, &mut need_preamble, &mut skip_in_chunk);

        assert!(!need_preamble);
        assert_eq!(skip_in_chunk, 4);
        assert_eq!(
            next_batch,
            vec![ChunkDrainInstructions {
                chunk_instructions: scheduled[0].clone(),
                rows_to_take: 4,
                rows_to_skip: 0,
                preamble_action: PreambleAction::Absent,
            }]
        );

        let next_batch =
            drain_from_instructions(&mut to_drain, 4, &mut need_preamble, &mut skip_in_chunk);

        assert!(!need_preamble);
        assert_eq!(skip_in_chunk, 2);

        assert_eq!(
            next_batch,
            vec![
                ChunkDrainInstructions {
                    chunk_instructions: scheduled[0].clone(),
                    rows_to_take: 1,
                    rows_to_skip: 4,
                    preamble_action: PreambleAction::Absent,
                },
                ChunkDrainInstructions {
                    chunk_instructions: scheduled[1].clone(),
                    rows_to_take: 1,
                    rows_to_skip: 0,
                    preamble_action: PreambleAction::Take,
                },
                ChunkDrainInstructions {
                    chunk_instructions: scheduled[2].clone(),
                    rows_to_take: 2,
                    rows_to_skip: 0,
                    preamble_action: PreambleAction::Absent,
                }
            ]
        );

        let next_batch =
            drain_from_instructions(&mut to_drain, 2, &mut need_preamble, &mut skip_in_chunk);

        assert!(!need_preamble);
        assert_eq!(skip_in_chunk, 0);

        assert_eq!(
            next_batch,
            vec![
                ChunkDrainInstructions {
                    chunk_instructions: scheduled[2].clone(),
                    rows_to_take: 1,
                    rows_to_skip: 2,
                    preamble_action: PreambleAction::Absent,
                },
                ChunkDrainInstructions {
                    chunk_instructions: scheduled[3].clone(),
                    rows_to_take: 1,
                    rows_to_skip: 0,
                    preamble_action: PreambleAction::Take,
                },
            ]
        );

        // Regression case.  Need a chunk with preamble, rows, and trailer (the middle chunk here)
        let rep_data: Vec<u64> = vec![5, 2, 3, 3, 20, 0];
        let rep_bytes: Vec<u8> = rep_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let chunk_index = MiniBlockChunkIndex::new_nested_for_test(&rep_bytes, 2);
        let user_ranges = vec![0..28];

        // First, schedule the ranges
        let scheduled = ChunkInstructions::schedule_instructions(&chunk_index, &user_ranges);

        let mut to_drain = VecDeque::from(scheduled.clone());

        // Drain first chunk and some of second chunk

        let mut need_preamble = false;
        let mut skip_in_chunk = 0;

        let next_batch =
            drain_from_instructions(&mut to_drain, 7, &mut need_preamble, &mut skip_in_chunk);

        assert_eq!(
            next_batch,
            vec![
                ChunkDrainInstructions {
                    chunk_instructions: scheduled[0].clone(),
                    rows_to_take: 6,
                    rows_to_skip: 0,
                    preamble_action: PreambleAction::Absent,
                },
                ChunkDrainInstructions {
                    chunk_instructions: scheduled[1].clone(),
                    rows_to_take: 1,
                    rows_to_skip: 0,
                    preamble_action: PreambleAction::Take,
                },
            ]
        );

        assert!(!need_preamble);
        assert_eq!(skip_in_chunk, 1);

        // Now, the tricky part.  We drain the second chunk, including the trailer, and need to make sure
        // we get a drain task to take the preamble of the third chunk (and nothing else)
        let next_batch =
            drain_from_instructions(&mut to_drain, 2, &mut need_preamble, &mut skip_in_chunk);

        assert_eq!(
            next_batch,
            vec![
                ChunkDrainInstructions {
                    chunk_instructions: scheduled[1].clone(),
                    rows_to_take: 2,
                    rows_to_skip: 1,
                    preamble_action: PreambleAction::Skip,
                },
                ChunkDrainInstructions {
                    chunk_instructions: scheduled[2].clone(),
                    rows_to_take: 0,
                    rows_to_skip: 0,
                    preamble_action: PreambleAction::Take,
                },
            ]
        );

        assert!(!need_preamble);
        assert_eq!(skip_in_chunk, 0);
    }

    /// Builds a `Words` metadata buffer (u16 words) from `(log_num_values, num_bytes)`
    /// pairs, returning the words and the total data-buffer size.
    fn words_from(entries: &[(u32, u32)]) -> (Words, u64) {
        let mut raw = Vec::with_capacity(entries.len() * 2);
        let mut total = 0u64;
        for &(log, num_bytes) in entries {
            assert!(num_bytes > 0 && num_bytes % MINIBLOCK_ALIGNMENT as u32 == 0);
            let divided = num_bytes / MINIBLOCK_ALIGNMENT as u32 - 1;
            let word = (divided << 4) | log;
            assert!(word <= u16::MAX as u32, "test word {word} exceeds u16");
            raw.extend_from_slice(&(word as u16).to_le_bytes());
            total += num_bytes as u64;
        }
        (Words::from_bytes(Bytes::from(raw), false).unwrap(), total)
    }

    fn rep_bytes_from(values: &[u64]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[rstest]
    // Two full chunks of 8 values (log 3) plus a partial last chunk; byte sizes vary
    // independently of value counts.
    #[case::uniform_partial_last(&[(3, 16), (3, 24), (0, 8)], 19, "uniform_flat", 8, 3)]
    // Single chunk covers the whole page.
    #[case::single_chunk(&[(0, 24)], 5, "uniform_flat", 5, 5)]
    // Last chunk is also full (exact multiple).
    #[case::exact_multiple(&[(3, 16), (3, 16)], 16, "uniform_flat", 8, 8)]
    // Non-last chunks differ in size, so this is a non-uniform flat page.
    #[case::non_uniform(&[(4, 16), (2, 16), (0, 8)], 21, "flat", 16, 1)]
    fn test_flat_detection(
        #[case] entries: &[(u32, u32)],
        #[case] items_in_page: u64,
        #[case] expected_kind: &str,
        #[case] expected_first_items: u64,
        #[case] expected_last_items: u64,
    ) {
        let base = 100u64;
        let (words, data_buf_size) = words_from(entries);
        let index = build_chunk_index(&words, items_in_page, base, data_buf_size, None, 0).unwrap();

        assert_eq!(index.row_mapping_debug(), expected_kind);
        assert_eq!(index.num_chunks(), entries.len());
        assert_eq!(index.items_in_chunk(0), expected_first_items);
        assert_eq!(index.items_in_chunk(entries.len() - 1), expected_last_items);

        // Byte ranges are absolute, contiguous, and exactly cover the data buffer.
        let mut expected_start = base;
        for (i, &(_, num_bytes)) in entries.iter().enumerate() {
            let range = index.byte_range(i);
            assert_eq!(range.start, expected_start);
            assert_eq!(range.end - range.start, num_bytes as u64);
            expected_start = range.end;
        }
        assert_eq!(expected_start, base + data_buf_size);

        // For flat pages rows == items, so the per-chunk items sum to the page total.
        let total_items: u64 = (0..index.num_chunks())
            .map(|i| index.items_in_chunk(i))
            .sum();
        assert_eq!(total_items, items_in_page);
    }

    #[test]
    fn test_nested_detection_and_axes() {
        // Repetition index (stride 2): three chunks holding 5, 4, 3 rows, no trailers.
        let rep = rep_bytes_from(&[5, 0, 4, 0, 3, 0]);

        // Uniform leaf chunking: value counts 4, 4, 2.
        let (words, data_buf_size) = words_from(&[(2, 8), (2, 8), (0, 8)]);
        let index = build_chunk_index(&words, 10, 0, data_buf_size, Some(&rep), 1).unwrap();
        assert_eq!(index.row_mapping_debug(), "nested");
        assert_eq!(index.num_chunks(), 3);
        // Rows come from the repetition index, not the value counts.
        assert_eq!(index.first_row(0), 0);
        assert_eq!(index.rows_in_chunk(0), 5);
        assert_eq!(index.first_row(1), 5);
        assert_eq!(index.rows_in_chunk(1), 4);
        assert_eq!(index.first_row(2), 9);
        assert_eq!(index.rows_in_chunk(2), 3);
        // Items come from the value words.
        assert_eq!(index.items_in_chunk(0), 4);
        assert_eq!(index.items_in_chunk(1), 4);
        assert_eq!(index.items_in_chunk(2), 2);

        // Non-uniform leaf chunking: value counts 8, 2, 5.
        let (words_nu, dbs_nu) = words_from(&[(3, 8), (1, 8), (0, 8)]);
        let index_nu = build_chunk_index(&words_nu, 15, 0, dbs_nu, Some(&rep), 1).unwrap();
        assert_eq!(index_nu.row_mapping_debug(), "nested");
        assert_eq!(index_nu.items_in_chunk(0), 8);
        assert_eq!(index_nu.items_in_chunk(1), 2);
        assert_eq!(index_nu.items_in_chunk(2), 5);
        // The row axis is unchanged by the leaf chunking.
        assert_eq!(index_nu.rows_in_chunk(0), 5);
    }

    #[test]
    fn test_uniform_flat_matches_prefix_sum_flat() {
        // Distribution: 4 chunks of 4 values, last chunk 3 (15 items total).
        let (words, data_buf_size) = words_from(&[(2, 8), (2, 8), (2, 8), (0, 8)]);
        let uniform = build_chunk_index(&words, 15, 0, data_buf_size, None, 0).unwrap();
        assert_eq!(uniform.row_mapping_debug(), "uniform_flat");

        // The same distribution expressed as a non-uniform Flat prefix-sum index.
        let byte_starts = PrefixSums::from_deltas([8u64, 8, 8, 8].into_iter(), 4, 32);
        let value_starts = PrefixSums::from_deltas([4u64, 4, 4, 3].into_iter(), 4, 15);
        let flat = MiniBlockChunkIndex::new(0, byte_starts, RowMapping::Flat { value_starts });
        assert_eq!(flat.row_mapping_debug(), "flat");

        // Lookup parity: identical byte ranges and item counts.
        for i in 0..4 {
            assert_eq!(uniform.byte_range(i), flat.byte_range(i));
            assert_eq!(uniform.items_in_chunk(i), flat.items_in_chunk(i));
        }

        // Scheduler parity across scan / single-row / partial / scattered multi-range.
        let range_sets: Vec<Vec<std::ops::Range<u64>>> = vec![
            vec![0..15],
            vec![0..1],
            vec![7..8],
            vec![14..15],
            vec![3..10],
            vec![0..2, 5..6, 12..15],
        ];
        for ranges in &range_sets {
            let from_uniform = ChunkInstructions::schedule_instructions(&uniform, ranges);
            let from_flat = ChunkInstructions::schedule_instructions(&flat, ranges);
            assert_eq!(from_uniform, from_flat, "mismatch for ranges {ranges:?}");
        }

        // A full scan yields one Absent, no-trailer instruction per chunk.
        let full = ChunkInstructions::schedule_instructions(&uniform, &[0..15]);
        assert_eq!(full.len(), 4);
        for (i, inst) in full.iter().enumerate() {
            assert_eq!(inst.chunk_idx, i);
            assert_eq!(inst.preamble, PreambleAction::Absent);
            assert_eq!(inst.rows_to_skip, 0);
            assert!(!inst.take_trailer);
        }
        assert_eq!(full.iter().map(|i| i.rows_to_take).sum::<u64>(), 15);
    }

    #[test]
    fn test_deep_size_per_variant_below_legacy() {
        // The previous representation cached 48 bytes per chunk (24 for ChunkMeta plus
        // 24 for a rep-index block); every variant's heap must be well below that.
        const LEGACY_PER_CHUNK: usize = 48;
        let num_chunks = 3;
        let heap = |index: &MiniBlockChunkIndex| index.deep_size_of_children(&mut Context::new());

        let (uniform_words, uniform_dbs) = words_from(&[(2, 8), (2, 8), (0, 8)]);
        let uniform = build_chunk_index(&uniform_words, 10, 0, uniform_dbs, None, 0).unwrap();
        assert_eq!(uniform.row_mapping_debug(), "uniform_flat");
        assert!(heap(&uniform) < LEGACY_PER_CHUNK * num_chunks);

        let (flat_words, flat_dbs) = words_from(&[(3, 8), (1, 8), (0, 8)]);
        let flat = build_chunk_index(&flat_words, 11, 0, flat_dbs, None, 0).unwrap();
        assert_eq!(flat.row_mapping_debug(), "flat");
        assert!(heap(&flat) < LEGACY_PER_CHUNK * num_chunks);
        // Flat carries a value-starts array that UniformFlat derives arithmetically.
        assert!(heap(&flat) > heap(&uniform));

        let rep = rep_bytes_from(&[4, 0, 3, 0, 3, 0]);
        let (nested_words, nested_dbs) = words_from(&[(2, 8), (2, 8), (0, 8)]);
        let nested = build_chunk_index(&nested_words, 10, 0, nested_dbs, Some(&rep), 1).unwrap();
        assert_eq!(nested.row_mapping_debug(), "nested");
        assert!(heap(&nested) < LEGACY_PER_CHUNK * num_chunks);
    }

    #[test]
    fn test_miniblock_dictionary_out_of_line_bitpacking_decode() {
        let rows = 10_000;
        let unique_values = 2_000;

        let dictionary_encoding =
            ProtobufUtils21::out_of_line_bitpacking(64, ProtobufUtils21::flat(11, None));
        let layout = pb21::MiniBlockLayout {
            rep_compression: None,
            def_compression: None,
            value_compression: Some(ProtobufUtils21::flat(64, None)),
            dictionary: Some(dictionary_encoding),
            num_dictionary_items: unique_values,
            layers: vec![pb21::RepDefLayer::RepdefAllValidItem as i32],
            num_buffers: 1,
            repetition_index_depth: 0,
            num_items: rows,
            has_large_chunk: false,
        };

        let buffer_offsets_and_sizes = vec![(0, 0), (0, 0), (0, 0)];
        let scheduler = super::MiniBlockScheduler::try_new(
            &buffer_offsets_and_sizes,
            /*priority=*/ 0,
            /*items_in_page=*/ rows,
            &layout,
            &DefaultDecompressionStrategy::default(),
        )
        .unwrap();

        let dictionary = scheduler.dictionary.unwrap();
        assert_eq!(dictionary.num_dictionary_items, unique_values);
        assert_eq!(
            dictionary.dictionary_data_alignment,
            crate::encoder::MIN_PAGE_BUFFER_ALIGNMENT
        );
    }

    #[test]
    fn miniblock_levels_use_one_count_for_mixed_codecs() {
        let actual_num_levels = usize::from(u16::MAX) + 8;
        let levels = vec![1_u16; actual_num_levels];
        let rep_frame = encoded_u16_frame(&levels, RunLengthWidth::U8);
        let rep_decompressor = RleDecompressor::new(16);
        let def_frame = LanceBuffer::reinterpret_slice(Arc::from(levels.clone()));
        let def_decompressor = ValueDecompressor::from_flat(&pb21::Flat {
            bits_per_value: 16,
            data: None,
        });

        let num_levels = DecodeMiniBlockTask::resolve_num_levels(
            Some(&rep_decompressor),
            Some(&rep_frame),
            Some(&def_decompressor),
            Some(&def_frame),
            7,
        )
        .unwrap();
        assert_eq!(num_levels, actual_num_levels as u64);

        let rep =
            DecodeMiniBlockTask::decode_levels(&rep_decompressor, rep_frame, num_levels).unwrap();
        let def =
            DecodeMiniBlockTask::decode_levels(&def_decompressor, def_frame, num_levels).unwrap();
        assert_eq!(rep.as_ref(), levels);
        assert_eq!(def, rep);
    }

    #[test]
    fn miniblock_levels_reject_non_wrapped_count_mismatch() {
        let frame = encoded_u16_frame(&[1_u16; 8], RunLengthWidth::U8);
        let decompressor = RleDecompressor::new(16);

        let error = DecodeMiniBlockTask::resolve_num_levels(
            Some(&decompressor),
            Some(&frame),
            None,
            None,
            7,
        )
        .unwrap_err();
        assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
        assert!(error.to_string().contains("not congruent modulo 65536"));
    }

    #[test]
    fn miniblock_levels_reject_cross_stream_count_disagreement() {
        let rep_levels = vec![1_u16; usize::from(u16::MAX) + 8];
        let def_levels = vec![1_u16; rep_levels.len() + usize::from(u16::MAX) + 1];
        let rep_frame = encoded_u16_frame(&rep_levels, RunLengthWidth::U8);
        let def_frame = encoded_u16_frame(&def_levels, RunLengthWidth::U8);
        let decompressor = RleDecompressor::new(16);

        let error = DecodeMiniBlockTask::resolve_num_levels(
            Some(&decompressor),
            Some(&rep_frame),
            Some(&decompressor),
            Some(&def_frame),
            7,
        )
        .unwrap_err();
        assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("structural streams disagree on the level count")
        );
    }
}
