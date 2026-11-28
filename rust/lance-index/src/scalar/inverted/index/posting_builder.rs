// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use smallvec::SmallVec;

use super::super::encoding::{
    decode_full_posting_block, decode_position_stream_deltas_into, decode_varint_u32,
};
use super::*;

/// Inline capacity, in encoded bytes, of a posting list's unflushed tail.
///
/// Most distinct tokens of a real corpus occur in only a few documents, and
/// 16 bytes hold their first three or four varint-encoded postings, so those
/// tokens never allocate a tail buffer.
const INLINE_TAIL_BYTES: usize = 16;

/// Builds the posting list of one token during indexing.
///
/// A partition holds one builder per distinct token and real corpora reach
/// tens of millions of distinct tokens, so the size of this struct and the
/// allocations every token pays dominate index-build memory. Postings that
/// have not filled a block yet are kept as a varint byte stream that starts
/// inline, and the state that only lists with full blocks or positions need
/// lives behind one lazily allocated box.
#[derive(Debug)]
pub struct PostingListBuilder {
    // Varint pairs of (doc id delta, frequency) for the postings that have not
    // filled a block yet. Deltas wrap, so any doc id order round-trips.
    pub(super) tail: SmallVec<[u8; INLINE_TAIL_BYTES]>,
    pub(super) overflow: Option<Box<PostingListOverflow>>,
    // Base of the next doc id delta: the last doc id in `tail`, or 0 while
    // `tail` is empty.
    pub(super) last_tail_doc_id: u32,
    // A tail that reaches `block_size` postings is flushed right away, so the
    // count stays below 256.
    pub(super) tail_len: u8,
    pub(super) with_positions: bool,
    pub(super) posting_tail_codec: PostingTailCodec,
    pub(super) block_size_log2: u8,
}

// There is one builder per distinct token of a partition; keep it small.
const _: () = assert!(std::mem::size_of::<PostingListBuilder>() <= 40);

/// Builder state that only posting lists with full blocks or positions need.
#[derive(Debug, Default)]
pub(super) struct PostingListOverflow {
    // Positions of the postings in `tail` and of the open document; only
    // lists with positions use them.
    pub(super) tail_positions: PositionBlockBuilder,
    pub(super) open_doc: Option<OpenDoc>,
    // Every token of a positional build allocates this box on its first
    // posting, but most tokens never fill a block, so block storage is boxed
    // separately.
    pub(super) full_blocks: Option<Box<FullBlocks>>,
}

// Every distinct token of a positional build allocates one; keep it small.
const _: () = assert!(std::mem::size_of::<PostingListOverflow>() <= 80);

/// The encoded full blocks of a posting list that reached `block_size`
/// postings.
#[derive(Debug, Default)]
pub(super) struct FullBlocks {
    pub(super) encoded_blocks: EncodedBlocks,
    pub(super) encoded_position_blocks: EncodedPositionBlocks,
    pub(super) max_block_scratch_bytes: u64,
}

/// Reusable scratch buffers shared across tokens during [`InnerBuilder::merge_from`].
#[derive(Default)]
pub(in super::super) struct PostingMergeScratch {
    doc_ids: Vec<u32>,
    frequencies: Vec<u32>,
    flush_doc_ids: Vec<u32>,
    flush_frequencies: Vec<u32>,
    deltas: Vec<u32>,
}

fn block_position_scratch_bytes(frequencies: &[u32]) -> Result<u64> {
    let positions: u64 = frequencies.iter().map(|&value| u64::from(value)).sum();
    let largest = u64::from(frequencies.iter().copied().max().unwrap_or(0));
    positions
        .checked_mul(3)
        .and_then(|positions| positions.checked_add(largest))
        .and_then(|positions| positions.checked_mul(4))
        .ok_or_else(|| Error::invalid_input("FTS merge position scratch size overflow"))
}

/// A document whose positions are still streaming in through
/// [`PostingListBuilder::add_occurrence`].
#[derive(Debug, Clone, Copy)]
pub(super) struct OpenDoc {
    doc_id: u32,
    frequency: u32,
    last_position: u32,
}

/// Decodes `remaining` (doc id, frequency) pairs from a builder tail.
struct TailEntries<'a> {
    bytes: &'a [u8],
    offset: usize,
    doc_id: u32,
    remaining: usize,
}

impl<'a> TailEntries<'a> {
    fn new(bytes: &'a [u8], len: usize) -> Self {
        Self {
            bytes,
            offset: 0,
            doc_id: 0,
            remaining: len,
        }
    }
}

impl Iterator for TailEntries<'_> {
    type Item = (u32, u32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let delta = decode_varint_u32(self.bytes, &mut self.offset)
            .expect("posting list tail is encoded by the builder");
        let frequency = decode_varint_u32(self.bytes, &mut self.offset)
            .expect("posting list tail is encoded by the builder");
        self.doc_id = self.doc_id.wrapping_add(delta);
        Some((self.doc_id, frequency))
    }
}

fn push_varint(dst: &mut SmallVec<[u8; INLINE_TAIL_BYTES]>, mut value: u32) {
    while value >= 0x80 {
        dst.push((value as u8) | 0x80);
        value >>= 7;
    }
    dst.push(value as u8);
}

impl PostingListBuilder {
    /// Heap bytes owned by this builder, excluding the builder struct itself.
    pub fn size(&self) -> u64 {
        let tail_size = if self.tail.spilled() {
            self.tail.capacity()
        } else {
            0
        };
        let overflow_size = self.overflow.as_deref().map_or(0, |overflow| {
            let full_blocks_size = overflow.full_blocks.as_deref().map_or(0, |full_blocks| {
                std::mem::size_of::<FullBlocks>()
                    + full_blocks.encoded_blocks.size()
                    + full_blocks.encoded_position_blocks.size()
            });
            std::mem::size_of::<PostingListOverflow>()
                + overflow.tail_positions.size()
                + full_blocks_size
        });
        (tail_size + overflow_size) as u64
    }

    pub fn has_positions(&self) -> bool {
        self.with_positions
    }

    pub fn new(with_position: bool) -> Self {
        Self::new_with_posting_tail_codec_and_block_size(
            with_position,
            current_fts_format_version().posting_tail_codec(),
            LEGACY_BLOCK_SIZE,
        )
    }

    pub fn new_with_posting_tail_codec(
        with_position: bool,
        posting_tail_codec: PostingTailCodec,
    ) -> Self {
        Self::new_with_posting_tail_codec_and_block_size(
            with_position,
            posting_tail_codec,
            LEGACY_BLOCK_SIZE,
        )
    }

    pub fn new_with_block_size(with_position: bool, block_size: usize) -> Self {
        Self::new_with_posting_tail_codec_and_block_size(
            with_position,
            current_fts_format_version().posting_tail_codec(),
            block_size,
        )
    }

    pub fn new_with_posting_tail_codec_and_block_size(
        with_position: bool,
        posting_tail_codec: PostingTailCodec,
        block_size: usize,
    ) -> Self {
        validate_block_size(block_size).expect("invalid posting list block size");
        Self {
            tail: SmallVec::new(),
            overflow: None,
            last_tail_doc_id: 0,
            tail_len: 0,
            with_positions: with_position,
            posting_tail_codec,
            block_size_log2: block_size.trailing_zeros() as u8,
        }
    }

    fn block_size(&self) -> usize {
        1 << self.block_size_log2
    }

    pub fn len(&self) -> usize {
        let tail_len = usize::from(self.tail_len);
        match self.overflow.as_deref() {
            Some(overflow) => {
                let num_full_blocks = overflow
                    .full_blocks
                    .as_deref()
                    .map_or(0, |full_blocks| full_blocks.encoded_blocks.len());
                num_full_blocks * self.block_size()
                    + tail_len
                    + usize::from(overflow.open_doc.is_some())
            }
            None => tail_len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> std::vec::IntoIter<(u32, u32, Option<Vec<u32>>)> {
        self.collect_entries().into_iter()
    }

    fn tail_entries(&self) -> TailEntries<'_> {
        TailEntries::new(&self.tail, usize::from(self.tail_len))
    }

    fn open_doc(&self) -> Option<OpenDoc> {
        self.overflow
            .as_deref()
            .and_then(|overflow| overflow.open_doc)
    }

    fn overflow_mut(&mut self) -> &mut PostingListOverflow {
        self.overflow.get_or_insert_with(Box::default)
    }

    /// Estimate scratch retained when merging or decoding a block, before
    /// decompressing positions. Full blocks track their maximum scratch bound
    /// incrementally so admission does not re-decode compressed blocks.
    pub(in super::super) fn merge_scratch_bytes(&self) -> Result<u64> {
        let block_size = self.block_size();
        let mut maximum = 0u64;
        if self.with_positions {
            if let Some(blocks) = self
                .overflow
                .as_deref()
                .and_then(|overflow| overflow.full_blocks.as_deref())
            {
                maximum = maximum.max(blocks.max_block_scratch_bytes);
            }
            if self.tail_len > 0 {
                let mut positions = 0u64;
                let mut largest = 0u64;
                for (_, frequency) in self.tail_entries() {
                    let frequency = u64::from(frequency);
                    positions += frequency;
                    largest = largest.max(frequency);
                }
                let bytes = positions
                    .checked_mul(3)
                    .and_then(|positions| positions.checked_add(largest))
                    .and_then(|positions| positions.checked_mul(4))
                    .ok_or_else(|| {
                        Error::invalid_input("FTS merge position scratch size overflow")
                    })?;
                maximum = maximum.max(bytes);
            }
        }
        maximum
            .checked_add(
                (block_size as u64)
                    .checked_mul(8)
                    .ok_or_else(|| Error::invalid_input("FTS merge block scratch size overflow"))?,
            )
            .ok_or_else(|| Error::invalid_input("FTS merge scratch size overflow"))
    }

    pub fn for_each_entry<E>(
        &self,
        mut visit: impl FnMut(u32, u32, Option<Vec<u32>>) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        let block_size = self.block_size();
        let mut decoded_positions = Vec::new();

        if let Some(full_blocks) = self
            .overflow
            .as_deref()
            .and_then(|overflow| overflow.full_blocks.as_deref())
        {
            let mut doc_ids = Vec::with_capacity(block_size);
            let mut frequencies = Vec::with_capacity(block_size);
            for (block_index, block) in full_blocks.encoded_blocks.iter().enumerate() {
                doc_ids.clear();
                frequencies.clear();
                decode_full_posting_block(block, &mut doc_ids, &mut frequencies, block_size);
                decoded_positions.clear();
                if self.with_positions {
                    super::super::encoding::decode_position_stream_block(
                        full_blocks.encoded_position_blocks.block(block_index),
                        &frequencies,
                        PositionStreamCodec::PackedDelta,
                        &mut decoded_positions,
                    )
                    .expect("position stream decoding should succeed");
                }
                let mut offset = 0usize;
                for (doc_id, frequency) in doc_ids.iter().copied().zip(frequencies.iter().copied())
                {
                    let positions = self.with_positions.then(|| {
                        let end = offset + frequency as usize;
                        let doc_positions = decoded_positions[offset..end].to_vec();
                        offset = end;
                        doc_positions
                    });
                    visit(doc_id, frequency, positions)?;
                }
            }
        }

        decoded_positions.clear();
        if self.with_positions && self.tail_len > 0 {
            let tail_frequencies = self
                .tail_entries()
                .map(|(_, frequency)| frequency)
                .collect::<Vec<_>>();
            self.overflow
                .as_deref()
                .expect("positions must exist for posting list")
                .tail_positions
                .decode_into(tail_frequencies.as_slice(), &mut decoded_positions)
                .expect("tail position stream decoding should succeed");
        }
        let mut tail_offset = 0usize;
        for (doc_id, frequency) in self.tail_entries() {
            let positions = self.with_positions.then(|| {
                let end = tail_offset + frequency as usize;
                let doc_positions = decoded_positions[tail_offset..end].to_vec();
                tail_offset = end;
                doc_positions
            });
            visit(doc_id, frequency, positions)?;
        }

        Ok(())
    }

    pub fn add(&mut self, doc_id: u32, term_positions: PositionRecorder) {
        let frequency = term_positions.len();
        match term_positions {
            PositionRecorder::Position(positions_in_doc) => {
                self.add_slice(doc_id, frequency, Some(positions_in_doc.as_slice()));
            }
            PositionRecorder::Count(_) => {
                self.add_slice(doc_id, frequency, None);
            }
        }
    }

    pub(in super::super) fn add_slice(
        &mut self,
        doc_id: u32,
        frequency: u32,
        positions: Option<&[u32]>,
    ) {
        debug_assert!(
            self.open_doc().is_none(),
            "cannot add closed doc while a positions doc is still open"
        );
        if let Some(positions_in_doc) = positions {
            debug_assert!(self.with_positions);
            self.overflow_mut()
                .tail_positions
                .append_doc_positions(positions_in_doc)
                .expect("position stream encoding should succeed");
        }
        self.push_tail_entry(doc_id, frequency)
            .expect("posting list block compression should succeed");
    }

    /// Append all postings from `other`, shifting its doc ids by `doc_id_offset`.
    ///
    /// Reuses compressed position blocks verbatim when aligned to block
    /// boundaries, concatenates tail position deltas directly when no boundary
    /// is crossed, and otherwise streams delta slices through reusable scratch
    /// buffers without per-document heap allocations.
    pub(in super::super) fn append_from(
        &mut self,
        other: Self,
        doc_id_offset: u32,
        scratch: &mut PostingMergeScratch,
    ) -> Result<()> {
        debug_assert_eq!(self.with_positions, other.with_positions);
        debug_assert_eq!(self.posting_tail_codec, other.posting_tail_codec);
        debug_assert_eq!(self.block_size_log2, other.block_size_log2);
        debug_assert!(self.open_doc().is_none());
        debug_assert!(other.open_doc().is_none());

        if other.is_empty() {
            return Ok(());
        }
        if self.is_empty() && doc_id_offset == 0 {
            *self = other;
            return Ok(());
        }

        let block_size = self.block_size();
        let with_positions = self.with_positions;
        let Self {
            tail: other_tail,
            overflow: other_overflow,
            tail_len: other_tail_len,
            ..
        } = other;
        let (other_full_blocks, other_tail_positions) = match other_overflow {
            Some(overflow) => {
                let overflow = *overflow;
                (overflow.full_blocks, Some(overflow.tail_positions))
            }
            None => (None, None),
        };

        // Fast path 1: `self` sits on a block boundary (`tail_len == 0`).
        // Full position blocks depend only on intra-document positions and can
        // be copied verbatim without decompressing or recompressing positions.
        if self.tail_len == 0 {
            debug_assert_eq!(self.last_tail_doc_id, 0);
            if let Some(other_full) = other_full_blocks {
                let self_full = self
                    .overflow_mut()
                    .full_blocks
                    .get_or_insert_with(Box::default);
                self_full.max_block_scratch_bytes = self_full
                    .max_block_scratch_bytes
                    .max(other_full.max_block_scratch_bytes);
                for (block_idx, block) in other_full.encoded_blocks.iter().enumerate() {
                    scratch.doc_ids.clear();
                    scratch.frequencies.clear();
                    decode_full_posting_block(
                        block,
                        &mut scratch.doc_ids,
                        &mut scratch.frequencies,
                        block_size,
                    );
                    for doc_id in &mut scratch.doc_ids {
                        *doc_id += doc_id_offset;
                    }
                    self_full
                        .encoded_blocks
                        .push_full_block(&scratch.doc_ids, &scratch.frequencies)?;
                    if with_positions {
                        self_full.encoded_position_blocks.push_encoded_block_slice(
                            other_full.encoded_position_blocks.block(block_idx),
                        )?;
                    }
                }
            }
            if other_tail_len > 0 {
                for (doc_id, frequency) in
                    TailEntries::new(&other_tail, usize::from(other_tail_len))
                {
                    let shifted = doc_id + doc_id_offset;
                    push_varint(&mut self.tail, shifted.wrapping_sub(self.last_tail_doc_id));
                    push_varint(&mut self.tail, frequency);
                    self.last_tail_doc_id = shifted;
                }
                self.tail_len = other_tail_len;
                if with_positions && let Some(other_tp) = other_tail_positions {
                    let tail_positions = &mut self.overflow_mut().tail_positions;
                    debug_assert!(tail_positions.is_empty());
                    *tail_positions = other_tp;
                }
            }
            return Ok(());
        }

        // Fast path 2: neither side completes a block; append tail entries and
        // position deltas directly without decoding or flushing.
        if other_full_blocks.is_none()
            && usize::from(self.tail_len) + usize::from(other_tail_len) < block_size
        {
            for (doc_id, frequency) in TailEntries::new(&other_tail, usize::from(other_tail_len)) {
                let shifted = doc_id + doc_id_offset;
                push_varint(&mut self.tail, shifted.wrapping_sub(self.last_tail_doc_id));
                push_varint(&mut self.tail, frequency);
                self.last_tail_doc_id = shifted;
            }
            self.tail_len += other_tail_len;
            if with_positions && let Some(other_tp) = other_tail_positions {
                self.overflow_mut()
                    .tail_positions
                    .append_from_builder(other_tp)?;
            }
            return Ok(());
        }

        // General path: stream `other`'s full blocks and tail across block
        // boundaries using reusable scratch buffers and raw delta slices.
        if let Some(other_full) = other_full_blocks {
            for (block_idx, block) in other_full.encoded_blocks.iter().enumerate() {
                scratch.doc_ids.clear();
                scratch.frequencies.clear();
                decode_full_posting_block(
                    block,
                    &mut scratch.doc_ids,
                    &mut scratch.frequencies,
                    block_size,
                );
                if with_positions {
                    let total_positions =
                        scratch.frequencies.iter().try_fold(0usize, |sum, &freq| {
                            sum.checked_add(freq as usize).ok_or_else(|| {
                                Error::index("position stream length overflow while merging")
                            })
                        })?;
                    scratch.deltas.clear();
                    decode_position_stream_deltas_into(
                        other_full.encoded_position_blocks.block(block_idx),
                        total_positions,
                        PositionStreamCodec::PackedDelta,
                        &mut scratch.deltas,
                    )?;
                }
                self.append_chunk_entries(
                    &scratch.doc_ids,
                    &scratch.frequencies,
                    with_positions.then_some(scratch.deltas.as_slice()),
                    doc_id_offset,
                    &mut scratch.flush_doc_ids,
                    &mut scratch.flush_frequencies,
                )?;
            }
        }

        if other_tail_len > 0 {
            if usize::from(self.tail_len) + usize::from(other_tail_len) < block_size {
                for (doc_id, frequency) in
                    TailEntries::new(&other_tail, usize::from(other_tail_len))
                {
                    let shifted = doc_id + doc_id_offset;
                    push_varint(&mut self.tail, shifted.wrapping_sub(self.last_tail_doc_id));
                    push_varint(&mut self.tail, frequency);
                    self.last_tail_doc_id = shifted;
                }
                self.tail_len += other_tail_len;
                if with_positions && let Some(other_tp) = other_tail_positions {
                    self.overflow_mut()
                        .tail_positions
                        .append_from_builder(other_tp)?;
                }
            } else {
                scratch.doc_ids.clear();
                scratch.frequencies.clear();
                for (doc_id, frequency) in
                    TailEntries::new(&other_tail, usize::from(other_tail_len))
                {
                    scratch.doc_ids.push(doc_id);
                    scratch.frequencies.push(frequency);
                }
                if with_positions {
                    scratch.deltas.clear();
                    if let Some(other_tp) = other_tail_positions {
                        other_tp.decode_deltas_into(&mut scratch.deltas)?;
                    }
                }
                self.append_chunk_entries(
                    &scratch.doc_ids,
                    &scratch.frequencies,
                    with_positions.then_some(scratch.deltas.as_slice()),
                    doc_id_offset,
                    &mut scratch.flush_doc_ids,
                    &mut scratch.flush_frequencies,
                )?;
            }
        }
        Ok(())
    }

    fn append_chunk_entries(
        &mut self,
        doc_ids: &[u32],
        frequencies: &[u32],
        deltas: Option<&[u32]>,
        doc_id_offset: u32,
        flush_doc_ids: &mut Vec<u32>,
        flush_frequencies: &mut Vec<u32>,
    ) -> Result<()> {
        debug_assert_eq!(doc_ids.len(), frequencies.len());
        let block_size = self.block_size();
        let mut entry_offset = 0usize;
        let mut delta_offset = 0usize;
        while entry_offset < doc_ids.len() {
            let need = block_size - usize::from(self.tail_len);
            let take = need.min(doc_ids.len() - entry_offset);
            let end = entry_offset + take;
            let mut delta_count = 0usize;
            for (&doc_id, &frequency) in doc_ids[entry_offset..end]
                .iter()
                .zip(frequencies[entry_offset..end].iter())
            {
                let shifted = doc_id + doc_id_offset;
                push_varint(&mut self.tail, shifted.wrapping_sub(self.last_tail_doc_id));
                push_varint(&mut self.tail, frequency);
                self.last_tail_doc_id = shifted;
                delta_count += frequency as usize;
            }
            if let Some(deltas) = deltas {
                let delta_end = delta_offset + delta_count;
                self.overflow_mut()
                    .tail_positions
                    .append_deltas(&deltas[delta_offset..delta_end])?;
                delta_offset = delta_end;
            }
            let new_tail_len = usize::from(self.tail_len) + take;
            if new_tail_len == block_size {
                self.flush_tail_block_with_scratch(flush_doc_ids, flush_frequencies)?;
            } else {
                self.tail_len = new_tail_len as u8;
            }
            entry_offset = end;
        }
        if let Some(deltas) = deltas {
            debug_assert_eq!(delta_offset, deltas.len());
        }
        Ok(())
    }

    pub fn add_occurrence(&mut self, doc_id: u32, position: u32) -> Result<bool> {
        if !self.with_positions {
            return Err(Error::index(
                "cannot append streamed positions to a posting list without positions".to_owned(),
            ));
        }

        let overflow = self.overflow_mut();
        match overflow.open_doc.as_mut() {
            Some(open_doc) if open_doc.doc_id == doc_id => {
                overflow
                    .tail_positions
                    .append_position(position, Some(open_doc.last_position))?;
                open_doc.frequency += 1;
                open_doc.last_position = position;
                Ok(false)
            }
            Some(open_doc) => Err(Error::index(format!(
                "posting list received doc {} before finishing open doc {}",
                doc_id, open_doc.doc_id
            ))),
            None => {
                overflow.tail_positions.append_position(position, None)?;
                overflow.open_doc = Some(OpenDoc {
                    doc_id,
                    frequency: 1,
                    last_position: position,
                });
                Ok(true)
            }
        }
    }

    pub fn finish_open_doc(&mut self, doc_id: u32) -> Result<()> {
        if !self.with_positions {
            return Ok(());
        }
        let Some(overflow) = self.overflow.as_deref_mut() else {
            return Ok(());
        };
        match overflow.open_doc {
            Some(open_doc) if open_doc.doc_id == doc_id => {
                overflow.open_doc = None;
                self.push_tail_entry(doc_id, open_doc.frequency)
            }
            Some(open_doc) => Err(Error::index(format!(
                "attempted to finish doc {} while doc {} is still open",
                doc_id, open_doc.doc_id
            ))),
            None => Ok(()),
        }
    }

    fn push_tail_entry(&mut self, doc_id: u32, frequency: u32) -> Result<()> {
        push_varint(&mut self.tail, doc_id.wrapping_sub(self.last_tail_doc_id));
        push_varint(&mut self.tail, frequency);
        self.last_tail_doc_id = doc_id;
        let tail_len = usize::from(self.tail_len) + 1;
        if tail_len == self.block_size() {
            self.flush_tail_block()
        } else {
            self.tail_len = tail_len as u8;
            Ok(())
        }
    }

    /// Encodes the full tail, which holds exactly `block_size` postings, as a
    /// block.
    fn flush_tail_block(&mut self) -> Result<()> {
        let block_size = self.block_size();
        let mut doc_ids = Vec::with_capacity(block_size);
        let mut frequencies = Vec::with_capacity(block_size);
        self.flush_tail_block_with_scratch(&mut doc_ids, &mut frequencies)
    }

    fn flush_tail_block_with_scratch(
        &mut self,
        doc_ids: &mut Vec<u32>,
        frequencies: &mut Vec<u32>,
    ) -> Result<()> {
        let block_size = self.block_size();
        doc_ids.clear();
        frequencies.clear();
        for (doc_id, frequency) in TailEntries::new(&self.tail, block_size) {
            doc_ids.push(doc_id);
            frequencies.push(frequency);
        }
        let with_positions = self.with_positions;
        let block_scratch = if with_positions {
            block_position_scratch_bytes(frequencies)?
        } else {
            0
        };
        let overflow = self.overflow_mut();
        debug_assert!(
            overflow.open_doc.is_none(),
            "cannot flush a posting block while a document is still open"
        );
        let full_blocks = overflow.full_blocks.get_or_insert_with(Box::default);
        full_blocks
            .encoded_blocks
            .push_full_block(doc_ids, frequencies)?;
        if with_positions {
            full_blocks.max_block_scratch_bytes =
                full_blocks.max_block_scratch_bytes.max(block_scratch);
            let tail_position_block = std::mem::take(&mut overflow.tail_positions).finish();
            full_blocks
                .encoded_position_blocks
                .push_encoded_block(tail_position_block)?;
        }
        self.tail.clear();
        self.tail_len = 0;
        self.last_tail_doc_id = 0;
        Ok(())
    }

    fn collect_entries(&self) -> Vec<(u32, u32, Option<Vec<u32>>)> {
        let mut entries = Vec::with_capacity(self.len());
        self.for_each_entry(|doc_id, frequency, positions| {
            entries.push((doc_id, frequency, positions));
            Ok::<(), ()>(())
        })
        .expect("collecting posting list entries should not fail");
        entries
    }

    fn into_parts(self) -> PostingListParts {
        debug_assert!(self.open_doc().is_none());
        let length = self.len();
        let block_size = self.block_size();
        let (tail_doc_ids, tail_frequencies) = self.tail_entries().unzip();
        let overflow = self.overflow.map(|overflow| *overflow).unwrap_or_default();
        let full_blocks = overflow
            .full_blocks
            .map(|full_blocks| *full_blocks)
            .unwrap_or_default();
        PostingListParts {
            with_positions: self.with_positions,
            posting_tail_codec: self.posting_tail_codec,
            block_size,
            length,
            encoded_blocks: full_blocks.encoded_blocks,
            encoded_position_blocks: full_blocks.encoded_position_blocks,
            tail_doc_ids,
            tail_frequencies,
            tail_position_block: self
                .with_positions
                .then(|| overflow.tail_positions.finish()),
        }
    }

    fn build_position_columns(positions: Option<PostingListPositions>) -> Result<Vec<ArrayRef>> {
        let Some(positions) = positions else {
            return Ok(Vec::new());
        };
        match positions {
            PostingListPositions::Legacy(positions) => Ok(vec![Arc::new(ListArray::try_new(
                Arc::new(Field::new("item", positions.data_type().clone(), true)),
                OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, positions.len() as i32])),
                Arc::new(positions) as ArrayRef,
                None,
            )?) as ArrayRef]),
            PostingListPositions::Shared(positions) => positions.into_columns(),
        }
    }

    fn build_batch(
        length: usize,
        compressed: LargeBinaryArray,
        impacts: Option<ImpactSkipData>,
        max_score: f32,
        schema: SchemaRef,
        positions: Option<PostingListPositions>,
    ) -> Result<RecordBatch> {
        let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0, compressed.len() as i32]));
        let mut columns = vec![
            Arc::new(ListArray::try_new(
                Arc::new(Field::new("item", datatypes::DataType::LargeBinary, true)),
                offsets,
                Arc::new(compressed),
                None,
            )?) as ArrayRef,
            Arc::new(Float32Array::from_iter_values(std::iter::once(max_score))) as ArrayRef,
            Arc::new(UInt32Array::from_iter_values(std::iter::once(
                length as u32,
            ))) as ArrayRef,
        ];
        if schema.field_with_name(IMPACT_COL).is_ok() {
            let impacts = impacts.ok_or_else(|| {
                Error::index(format!(
                    "impact column requested without impact data for posting length {}",
                    length
                ))
            })?;
            let impact_offsets =
                OffsetBuffer::new(ScalarBuffer::from(vec![0, impacts.entries().len() as i32]));
            columns.push(Arc::new(ListArray::try_new(
                Arc::new(Field::new("item", datatypes::DataType::LargeBinary, true)),
                impact_offsets,
                Arc::new(impacts.entries().clone()),
                None,
            )?) as ArrayRef);
        }
        columns.extend(Self::build_position_columns(positions)?);

        let batch = RecordBatch::try_new(schema, columns)?;
        Ok(batch)
    }

    fn build_legacy_positions(&self) -> Result<ListArray> {
        let mut positions_builder = ListBuilder::new(LargeBinaryBuilder::new());
        self.for_each_entry(|_doc_id, frequency, positions| {
            let positions = positions.ok_or_else(|| {
                Error::index(format!(
                    "legacy position writer missing positions for frequency {}",
                    frequency
                ))
            })?;
            let compressed = super::super::encoding::compress_positions(positions.as_slice())?;
            for block_idx in 0..compressed.len() {
                positions_builder
                    .values()
                    .append_value(compressed.value(block_idx));
            }
            positions_builder.append(true);
            Ok::<(), Error>(())
        })?;
        Ok(positions_builder.finish())
    }

    pub(in super::super) fn append_to_batch_with_docs(
        self,
        docs: &DocSet,
        batch_builder: &mut PostingListBatchBuilder,
        format_version: InvertedListFormatVersion,
    ) -> Result<()> {
        let legacy_positions =
            if self.with_positions && !format_version.uses_shared_position_stream() {
                Some(self.build_legacy_positions()?)
            } else {
                None
            };
        let parts = self.into_parts();
        let length = parts.length as u32;
        let (compressed, shared_positions, max_score, impacts) =
            Self::build_compressed_with_scores_from_parts(parts, docs)?;
        let positions = match legacy_positions {
            Some(positions) => Some(PostingListPositions::Legacy(positions)),
            None => shared_positions.map(PostingListPositions::Shared),
        };
        batch_builder.append(compressed, Some(&impacts), max_score, length, positions)
    }

    fn build_compressed_with_scores_from_parts(
        parts: PostingListParts,
        docs: &DocSet,
    ) -> Result<(
        LargeBinaryArray,
        Option<EncodedPositionBlocks>,
        f32,
        ImpactSkipData,
    )> {
        let PostingListParts {
            with_positions,
            posting_tail_codec,
            length,
            block_size,
            mut encoded_blocks,
            mut encoded_position_blocks,
            tail_doc_ids,
            tail_frequencies,
            tail_position_block,
        } = parts;
        let avgdl = docs.average_length();
        let idf_scale = idf(length, docs.len()) * (K1 + 1.0);
        let mut max_score = f32::MIN;
        let mut doc_ids = Vec::with_capacity(block_size);
        let mut frequencies = Vec::with_capacity(block_size);
        let mut impact_block = Vec::with_capacity(block_size);
        let mut impact_builder =
            ImpactSkipDataBuilder::with_capacity(length.div_ceil(block_size), block_size);

        for index in 0..encoded_blocks.len() {
            let block = encoded_blocks.block(index);
            doc_ids.clear();
            frequencies.clear();
            decode_full_posting_block(block, &mut doc_ids, &mut frequencies, block_size);
            let block_score = compute_block_score_and_impact_block(
                docs,
                avgdl,
                idf_scale,
                doc_ids.iter().copied(),
                frequencies.iter().copied(),
                &mut impact_block,
            );
            impact_builder.append_block(impact_block.as_slice())?;
            max_score = max_score.max(block_score);
            if super::super::encoding::posting_block_score_prefix_len(block_size) > 0 {
                encoded_blocks.set_block_score(index, block_score);
            }
        }

        if !tail_doc_ids.is_empty() {
            let block_score = compute_block_score_and_impact_block(
                docs,
                avgdl,
                idf_scale,
                tail_doc_ids.iter().copied(),
                tail_frequencies.iter().copied(),
                &mut impact_block,
            );
            impact_builder.append_block(impact_block.as_slice())?;
            max_score = max_score.max(block_score);
            encoded_blocks.append_remainder_block_with_codec(
                tail_doc_ids.as_slice(),
                tail_frequencies.as_slice(),
                posting_tail_codec,
                block_size,
            )?;
            if super::super::encoding::posting_block_score_prefix_len(block_size) > 0 {
                encoded_blocks.set_block_score(encoded_blocks.len() - 1, block_score);
            }
            if with_positions {
                encoded_position_blocks.push_encoded_block(
                    tail_position_block
                        .expect("tail position block must exist for postings with positions"),
                )?;
            }
        }

        let impacts = impact_builder.finish()?;
        Ok((
            encoded_blocks.into_array(),
            with_positions.then_some(encoded_position_blocks),
            max_score,
            impacts,
        ))
    }

    fn build_compressed_with_block_scores_from_parts(
        parts: PostingListParts,
        mut block_max_scores: impl Iterator<Item = f32>,
    ) -> Result<(LargeBinaryArray, Option<EncodedPositionBlocks>, f32)> {
        let PostingListParts {
            with_positions,
            posting_tail_codec,
            block_size,
            mut encoded_blocks,
            mut encoded_position_blocks,
            tail_doc_ids,
            tail_frequencies,
            tail_position_block,
            ..
        } = parts;
        let has_score_prefix =
            super::super::encoding::posting_block_score_prefix_len(block_size) > 0;
        let mut max_score = f32::MIN;

        for index in 0..encoded_blocks.len() {
            let block_score = block_max_scores
                .next()
                .ok_or_else(|| Error::index("missing block max score".to_owned()))?;
            max_score = max_score.max(block_score);
            if has_score_prefix {
                encoded_blocks.set_block_score(index, block_score);
            }
        }

        if !tail_doc_ids.is_empty() {
            let block_score = block_max_scores
                .next()
                .ok_or_else(|| Error::index("missing tail block max score".to_owned()))?;
            max_score = max_score.max(block_score);
            encoded_blocks.append_remainder_block_with_codec(
                tail_doc_ids.as_slice(),
                tail_frequencies.as_slice(),
                posting_tail_codec,
                block_size,
            )?;
            if has_score_prefix {
                encoded_blocks.set_block_score(encoded_blocks.len() - 1, block_score);
            }
            if with_positions {
                encoded_position_blocks.push_encoded_block(
                    tail_position_block
                        .expect("tail position block must exist for postings with positions"),
                )?;
            }
        }

        Ok((
            encoded_blocks.into_array(),
            with_positions.then_some(encoded_position_blocks),
            max_score,
        ))
    }

    pub fn to_batch(self, block_max_scores: Vec<f32>) -> Result<RecordBatch> {
        let format_version = InvertedListFormatVersion::from_posting_tail_codec_and_block_size(
            self.posting_tail_codec,
            self.block_size(),
        )?;
        let schema = inverted_list_schema_for_version_with_block_size_and_impacts(
            self.has_positions(),
            format_version,
            self.block_size(),
            false,
        );
        let legacy_positions =
            if self.with_positions && !format_version.uses_shared_position_stream() {
                Some(self.build_legacy_positions()?)
            } else {
                None
            };
        let parts = self.into_parts();
        let length = parts.length;
        let (compressed, shared_positions, max_score) =
            Self::build_compressed_with_block_scores_from_parts(
                parts,
                block_max_scores.into_iter(),
            )?;
        let positions = match legacy_positions {
            Some(positions) => Some(PostingListPositions::Legacy(positions)),
            None => shared_positions.map(PostingListPositions::Shared),
        };
        Self::build_batch(length, compressed, None, max_score, schema, positions)
    }

    pub fn to_batch_with_docs(self, docs: &DocSet, schema: SchemaRef) -> Result<RecordBatch> {
        let format_version = parse_format_version_from_metadata(schema.metadata())?;
        let legacy_positions =
            if self.with_positions && !format_version.uses_shared_position_stream() {
                Some(self.build_legacy_positions()?)
            } else {
                None
            };
        let parts = self.into_parts();
        let length = parts.length;
        let (compressed, shared_positions, max_score, impacts) =
            Self::build_compressed_with_scores_from_parts(parts, docs)?;
        let positions = match legacy_positions {
            Some(positions) => Some(PostingListPositions::Legacy(positions)),
            None => shared_positions.map(PostingListPositions::Shared),
        };
        Self::build_batch(
            length,
            compressed,
            Some(impacts),
            max_score,
            schema,
            positions,
        )
    }

    pub fn remap(&mut self, removed: &[u32]) {
        if removed.is_empty() {
            return;
        }
        let mut cursor = 0;
        let mut new_builder = Self::new_with_posting_tail_codec_and_block_size(
            self.has_positions(),
            self.posting_tail_codec,
            self.block_size(),
        );
        for (doc_id, freq, positions) in self.iter() {
            cursor += removed_before(&removed[cursor..], doc_id);
            if cursor < removed.len() && removed[cursor] == doc_id {
                continue;
            }
            let positions = match positions {
                Some(positions) => PositionRecorder::Position(positions.into()),
                None => PositionRecorder::Count(freq),
            };
            new_builder.add(doc_id - cursor as u32, positions);
        }

        *self = new_builder;
    }
}

// Every token starts at the beginning of the removal list. A linear walk costs
// O(tokens * removed documents) for rare tokens near the end of a partition.
// Gallop over large gaps, then bisect only that window; dense postings retain
// constant work for small gaps. Both input lists are in ascending doc-id order.
fn removed_before(removed: &[u32], doc_id: u32) -> usize {
    if removed.first().is_none_or(|&removed| removed >= doc_id) {
        return 0;
    }
    let mut start = 1;
    let mut end = 1;
    while end < removed.len() && removed[end] < doc_id {
        start = end + 1;
        end = end.saturating_mul(2);
    }
    start + removed[start..end.min(removed.len())].partition_point(|&removed| removed < doc_id)
}

pub(super) fn compute_block_score_and_impact_block(
    docs: &DocSet,
    avgdl: f32,
    idf_scale: f32,
    doc_ids: impl Iterator<Item = u32>,
    frequencies: impl Iterator<Item = u32>,
    impact_block: &mut Vec<(u32, u32, u32)>,
) -> f32 {
    impact_block.clear();
    let mut block_max_score = f32::MIN;
    for (doc_id, freq) in doc_ids.zip(frequencies) {
        let doc_len = docs.num_tokens(doc_id);
        let doc_norm = K1 * (1.0 - B + B * doc_len as f32 / avgdl);
        let freq_f32 = freq as f32;
        let score = freq_f32 / (freq_f32 + doc_norm);
        block_max_score = block_max_score.max(score);
        impact_block.push((doc_id, freq, doc_len));
    }
    block_max_score * idf_scale
}
