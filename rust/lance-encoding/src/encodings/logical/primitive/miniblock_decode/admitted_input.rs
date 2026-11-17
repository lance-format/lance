// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Prepare selected flat MiniBlock chunks without an uncharged page index/cache.

use std::{future::Future, ops::Range, sync::Arc};

use arrow_array::ArrayRef;
use arrow_schema::DataType;
use lance_core::{
    Error, Result,
    utils::allocation::{Reservation, ReservedBuffer},
};

use super::{
    ChunkDrainInstructions, ChunkInstructions, DecodeMiniBlockTask, LoadedChunk,
    MINIBLOCK_ALIGNMENT, MiniBlockScheduler, PreambleAction,
};
use crate::{
    buffer::LanceBuffer,
    compression::DecompressionStrategy,
    decoder::{DecodePageTask, PageInfo},
    encodings::logical::primitive::fullzip_decode::array_admitted::Shape,
    format::metadata::MiniBlockLayout,
    repdef::{RepDefUnraveler, max_visible_level},
};

fn corrupt(message: &str) -> Error {
    Error::corrupt_file_named("MiniBlock input", message)
}

#[derive(Debug)]
pub(in super::super) struct MiniBlockArrayScheduler {
    decoder: Arc<MiniBlockScheduler>,
    metadata: Range<u64>,
    data: Range<u64>,
    rows: u64,
}

#[derive(Debug)]
struct Selection {
    index: usize,
    bytes: Range<u64>,
    items: u64,
    rows: Range<u64>,
}

impl MiniBlockArrayScheduler {
    pub(in super::super) fn try_new(
        page: &PageInfo,
        layout: &MiniBlockLayout,
        codecs: &dyn DecompressionStrategy,
    ) -> Result<Self> {
        if layout.dictionary.is_some()
            || layout.rep_compression.is_some()
            || layout.repetition_index_depth != 0
        {
            return Err(Error::not_supported(
                "admitted MiniBlock input does not yet support dictionaries or repetition indexes",
            ));
        }
        if page.buffer_offsets_and_sizes.len() != 2
            || layout.num_buffers == 0
            || layout.value_compression.is_none()
            || layout.num_items != page.num_rows
            || page.num_rows == 0
        {
            return Err(corrupt(
                "flat page requires two buffers, a value codec and matching nonzero row/item counts",
            ));
        }
        let mut ranges = page.buffer_offsets_and_sizes.iter().map(|(start, len)| {
            start
                .checked_add(*len)
                .map(|end| *start..end)
                .ok_or_else(|| corrupt("buffer range overflows"))
        });
        let metadata = ranges.next().ok_or_else(|| corrupt("missing metadata"))??;
        let data = ranges.next().ok_or_else(|| corrupt("missing data"))??;
        let width = if layout.has_large_chunk { 4 } else { 2 };
        if metadata.is_empty() || (metadata.end - metadata.start) % width != 0 {
            return Err(corrupt(
                "metadata must contain complete nonempty chunk words",
            ));
        }
        let decoder = MiniBlockScheduler::try_new(
            &page.buffer_offsets_and_sizes,
            page.priority,
            layout.num_items,
            layout,
            codecs,
        )?;
        RepDefUnraveler::validate_dense_levels(None, None, &decoder.def_meaning)?;
        if decoder.def_meaning.len() != 1 || decoder.def_meaning[0].is_list() {
            return Err(Error::not_supported(
                "admitted flat MiniBlock input requires one non-list layer",
            ));
        }
        Ok(Self {
            decoder: Arc::new(decoder),
            metadata,
            data,
            rows: page.num_rows,
        })
    }

    pub(in super::super) fn validate_array_type(&self, data_type: &DataType) -> Result<()> {
        Shape::from_layers(data_type, &self.decoder.def_meaning, None).map(|_| ())
    }

    // Validate every metadata word, including the unselected suffix. Retain only
    // selected routing descriptors; payload scratch never grows with page size.
    async fn selections<F, Fut>(
        &self,
        rows: &Range<u64>,
        work: &mut Reservation,
        read: &mut F,
    ) -> Result<Option<Vec<Selection>>>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        let width = if self.decoder.has_large_chunk { 4 } else { 2 };
        let count = (self.metadata.end - self.metadata.start) / width;
        let mut position = self.metadata.start;
        let mut index = 0_u64;
        let mut item = 0_u64;
        let mut byte = self.data.start;
        let mut selected = Vec::new();
        let available = work.len() as u64;
        let mut selected_bytes = 0_u64;
        let mut fits = true;
        while position < self.metadata.end {
            let len = (self.metadata.end - position).min(4096) as usize;
            let Some(scratch) = work.try_allocate(len)? else {
                return Ok(None);
            };
            let scratch = read(position..position + len as u64, scratch).await?;
            if scratch.as_ref().len() != len {
                return Err(corrupt("metadata read length mismatch"));
            }
            for word in scratch.as_ref().chunks_exact(width as usize) {
                let mut bytes = [0; 4];
                bytes[..word.len()].copy_from_slice(word);
                let word = u32::from_le_bytes(bytes);
                let log = word & 15;
                let items = if index + 1 == count {
                    let remainder = self
                        .rows
                        .checked_sub(item)
                        .filter(|n| *n > 0)
                        .ok_or_else(|| corrupt("chunk item counts exceed page rows"))?;
                    if log != 0 && (1_u64 << log) != remainder {
                        return Err(corrupt("final chunk count disagrees with page rows"));
                    }
                    remainder
                } else {
                    if log == 0 {
                        return Err(corrupt("non-final chunk has zero count exponent"));
                    }
                    1_u64 << log
                };
                let end_item = item
                    .checked_add(items)
                    .filter(|n| *n <= self.rows)
                    .ok_or_else(|| corrupt("chunk item counts exceed page rows"))?;
                let size = (u64::from(word >> 4) + 1) * MINIBLOCK_ALIGNMENT as u64;
                let end_byte = byte
                    .checked_add(size)
                    .filter(|n| *n <= self.data.end)
                    .ok_or_else(|| corrupt("chunk byte sizes exceed data buffer"))?;
                let start = rows.start.max(item);
                let end = rows.end.min(end_item);
                if start < end {
                    selected_bytes = selected_bytes
                        .checked_add(size)
                        .ok_or_else(|| corrupt("selected input size overflows"))?;
                    if fits && selected_bytes > available {
                        // Continue validating the suffix, without retaining a routing
                        // table for an attempt that cannot admit its selected input.
                        selected.clear();
                        fits = false;
                    }
                }
                if start < end && fits {
                    selected.push(Selection {
                        index: usize::try_from(index)
                            .map_err(|_| corrupt("chunk index exceeds usize"))?,
                        bytes: byte..end_byte,
                        items,
                        rows: start - item..end - item,
                    });
                }
                item = end_item;
                byte = end_byte;
                index += 1;
            }
            scratch.recycle(work)?;
            position += len as u64;
            // Cached/in-memory I/O can complete immediately across every window.
            tokio::task::yield_now().await;
        }
        if byte != self.data.end || item != self.rows {
            return Err(corrupt(
                "chunk metadata does not cover the exact data/row domain",
            ));
        }
        Ok(fits.then_some(selected))
    }

    pub(in super::super) async fn prepare<F, Fut>(
        &self,
        rows: Range<u64>,
        data_type: DataType,
        work: &mut Reservation,
        mut read: F,
    ) -> Result<Option<MiniBlockArrayInput>>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        self.validate_array_type(&data_type)?;
        if rows.start >= rows.end || rows.end > self.rows {
            return Err(Error::invalid_input(
                "MiniBlock input range must be nonempty and within page rows",
            ));
        }
        let Some(selected) = self.selections(&rows, work, &mut read).await? else {
            return Ok(None);
        };
        let required = selected.iter().try_fold(0_usize, |sum, chunk| {
            usize::try_from(chunk.bytes.end - chunk.bytes.start)
                .ok()
                .and_then(|bytes| sum.checked_add(bytes))
                .ok_or_else(|| corrupt("selected input size overflows"))
        })?;
        if required > work.len() {
            return Ok(None);
        }
        let mut instructions = Vec::with_capacity(selected.len());
        for chunk in selected {
            let len = usize::try_from(chunk.bytes.end - chunk.bytes.start)
                .map_err(|_| corrupt("chunk length exceeds usize"))?;
            let output = work
                .try_allocate(len)?
                .ok_or_else(|| Error::internal("MiniBlock input preflight changed"))?;
            let output = read(chunk.bytes.clone(), output).await?;
            if output.as_ref().len() != len {
                return Err(corrupt("chunk read length mismatch"));
            }
            let take = chunk.rows.end - chunk.rows.start;
            instructions.push((
                ChunkDrainInstructions {
                    chunk_instructions: ChunkInstructions {
                        chunk_idx: chunk.index,
                        preamble: PreambleAction::Absent,
                        rows_to_skip: chunk.rows.start,
                        rows_to_take: take,
                        take_trailer: false,
                    },
                    rows_to_skip: 0,
                    rows_to_take: take,
                    preamble_action: PreambleAction::Absent,
                },
                LoadedChunk {
                    data: LanceBuffer::from_bytes(output.freeze(), 1),
                    items_in_chunk: chunk.items,
                    byte_range: chunk.bytes,
                    chunk_idx: chunk.index,
                },
            ));
        }
        Ok(Some(MiniBlockArrayInput {
            task: DecodeMiniBlockTask {
                structural_projection: None,
                dictionary_data: None,
                rep_decompressor: None,
                def_decompressor: self.decoder.def_decompressor.clone(),
                value_decompressor: self.decoder.value_decompressor.clone(),
                def_meaning: self.decoder.def_meaning.clone(),
                num_buffers: self.decoder.num_buffers,
                max_visible_level: max_visible_level(&self.decoder.def_meaning),
                instructions,
                has_large_chunk: self.decoder.has_large_chunk,
            },
            data_type,
            count: rows.end - rows.start,
        }))
    }
}

#[derive(Debug)]
pub(in super::super) struct MiniBlockArrayInput {
    task: DecodeMiniBlockTask,
    data_type: DataType,
    count: u64,
}

impl MiniBlockArrayInput {
    pub(in super::super) fn decode_in(self, work: &mut Reservation) -> Result<Option<ArrayRef>> {
        for (_, chunk) in &self.task.instructions {
            let sections = DecodeMiniBlockTask::split_miniblock_chunk(
                &chunk.data,
                false,
                self.task.def_decompressor.is_some(),
                self.task.num_buffers,
                self.task.has_large_chunk,
            )?;
            // The on-disk level count is u16 and may wrap. On flat pages every
            // item has one level; chunk metadata independently proves the count.
            if u64::from(sections.num_levels) != chunk.items_in_chunk % 65536 {
                return Err(corrupt("chunk level header disagrees with flat item count"));
            }
        }
        let page = self.task.try_decode_in(work)?;
        let meanings = self.task.def_meaning.clone();
        // Shared raw output may still retain a chunk owner. Never refund input
        // credit to this workspace while an output alias can keep it alive.
        drop(self.task);
        let Some(page) = page else {
            return Ok(None);
        };
        Shape::from_layers(&self.data_type, &meanings, None)?.finish(page, self.count, work)
    }
}

#[cfg(test)]
mod tests;
