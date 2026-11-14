// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Posting-list traversal: block decoding, block-max bounds and position cursors.

#[cfg(test)]
use std::cell::Cell;
use std::cell::{RefCell, UnsafeCell};
use std::collections::VecDeque;
use std::fmt::Debug;
use std::sync::Arc;
#[cfg(target_arch = "x86_64")]
use std::sync::LazyLock;

use arrow::array::AsArray;
use arrow::datatypes::Int32Type;
use arrow_array::Array;
use lance_core::{Error, Result};

use crate::scalar::inverted::builder::BLOCK_SIZE;
use crate::scalar::inverted::encoding::{
    MAX_POSTING_BLOCK_SIZE, decode_position_stream_block, decompress_positions,
    decompress_posting_block_doc_ids, decompress_posting_block_frequencies,
    decompress_posting_remainder_doc_ids, decompress_posting_remainder_frequencies,
    seek_packed_doc_positions,
};
use crate::scalar::inverted::impact::{IMPACT_LEVEL1_BLOCKS, ImpactScoreCache, ImpactSkipData};
use crate::scalar::inverted::index::PositionStreamCodec;
use crate::scalar::inverted::scorer::{BM25_DOC_WEIGHT_UPPER_BOUND, Scorer, idf};
use crate::scalar::inverted::{
    CompressedPositionStorage, CompressedPostingList, DocInfo, PostingList, RawDocInfo,
};

use super::TERMINATED_DOC_ID;

const LINEAR_BLOCK_SKIP_LIMIT: usize = 8;

#[cfg(target_arch = "x86_64")]
pub(super) static HAS_AVX2: LazyLock<bool> =
    LazyLock::new(|| std::arch::is_x86_feature_detected!("avx2"));

/// First index in `[pos, end)` where `docs[index] >= target` (scalar).
#[inline]
pub(super) unsafe fn find_next_geq_scalar(
    docs: *const u32,
    mut pos: usize,
    end: usize,
    target: u32,
) -> usize {
    unsafe {
        while pos < end && *docs.add(pos) < target {
            pos += 1;
        }
    }
    pos
}

/// AVX2 `find_next_geq` (the analogue of Lucene's VectorUtil.findNextGEQ):
/// scalar lookahead skips windows wholly before the target, then an 8-wide
/// compare locates the first matching lane.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn find_next_geq_avx2(
    docs: *const u32,
    mut pos: usize,
    end: usize,
    target: u32,
) -> usize {
    use core::arch::x86_64::*;
    unsafe {
        // Bias unsigned values around the sign bit so the signed AVX2 compare
        // preserves the ordering of the full u32 domain.
        let sign_bit = _mm256_set1_epi32(i32::MIN);
        let target_lanes = _mm256_xor_si256(_mm256_set1_epi32(target as i32), sign_bit);
        while pos + 8 < end {
            // Skip nine-document groups that end before the target. Only the
            // group containing the lower bound needs a vector load/compare.
            if *docs.add(pos + 8) < target {
                pos += 9;
                continue;
            }
            let docs_lanes = _mm256_xor_si256(
                _mm256_loadu_si256(docs.add(pos) as *const __m256i),
                sign_bit,
            );
            // Lane mask of docs[i] < target.
            let below = _mm256_cmpgt_epi32(target_lanes, docs_lanes);
            let mask = _mm256_movemask_ps(_mm256_castsi256_ps(below)) as u32;
            if mask != 0xFF {
                return pos + mask.trailing_ones() as usize;
            }
            return pos + 8;
        }
        find_next_geq_scalar(docs, pos, end, target)
    }
}

#[inline]
pub(super) unsafe fn find_next_geq(docs: *const u32, pos: usize, end: usize, target: u32) -> usize {
    #[cfg(target_arch = "x86_64")]
    if *HAS_AVX2 {
        return unsafe { find_next_geq_avx2(docs, pos, end, target) };
    }
    unsafe { find_next_geq_scalar(docs, pos, end, target) }
}

/// Search a sorted posting block using AVX2 when available. Keep the binary
/// search fallback because linear catch-up is slower on targets without AVX2.
#[inline]
pub(super) fn find_next_geq_in_block(docs: &[u32], pos: usize, target: u32) -> usize {
    debug_assert!(pos <= docs.len());
    #[cfg(target_arch = "x86_64")]
    if *HAS_AVX2 {
        return unsafe { find_next_geq_avx2(docs.as_ptr(), pos, docs.len(), target) };
    }
    pos + docs[pos..].partition_point(|&doc_id| doc_id < target)
}

#[inline]
fn conservative_bm25_upper_bound(query_weight: f32) -> f32 {
    if query_weight <= 0.0 {
        0.0
    } else {
        query_weight * BM25_DOC_WEIGHT_UPPER_BOUND
    }
}

#[inline]
fn scorer_upper_bound<S: Scorer + ?Sized>(query_weight: f32, scorer: &S) -> f32 {
    if query_weight.is_nan() {
        return f32::INFINITY;
    }
    if query_weight <= 0.0 {
        return 0.0;
    }
    match scorer.doc_weight_upper_bound() {
        Some(bound) if bound.is_finite() && bound >= 0.0 => query_weight * bound,
        _ => f32::INFINITY,
    }
}

/// Per-term data retained after same-position postings are unioned.
#[derive(Debug)]
pub(in crate::scalar::inverted) struct GroupedTermScorer {
    query_weight: f32,
    freqs_by_posting_doc_id: Vec<(u64, u32)>,
}

impl GroupedTermScorer {
    pub(in crate::scalar::inverted) fn new(query_weight: f32, posting: &PostingList) -> Self {
        let freqs_by_posting_doc_id = posting
            .iter()
            .map(|(posting_doc_id, freq, _)| (posting_doc_id, freq))
            .collect();
        Self {
            query_weight,
            freqs_by_posting_doc_id,
        }
    }

    pub(in crate::scalar::inverted) fn query_weight(&self) -> f32 {
        self.query_weight
    }

    pub(in crate::scalar::inverted) fn frequency(&self, posting_doc_id: u64) -> Option<u32> {
        self.freqs_by_posting_doc_id
            .binary_search_by_key(&posting_doc_id, |(doc_id, _)| *doc_id)
            .ok()
            .map(|index| self.freqs_by_posting_doc_id[index].1)
    }

    fn score<S: Scorer + ?Sized>(&self, posting_doc_id: u64, doc_length: u32, scorer: &S) -> f32 {
        self.frequency(posting_doc_id)
            .map(|freq| self.query_weight * scorer.doc_weight(freq, doc_length))
            .unwrap_or_default()
    }
}

pub struct PostingIterator {
    token: String,
    pub(super) token_id: u32,
    pub(super) position: u32,
    pub(super) query_weight: f32,
    pub(super) list: PostingList,
    // the index of current doc, this can be changed only by `next()`
    pub(super) index: usize,
    // the index of current block, this can be changed by `next() and shallow_next()`
    pub(super) block_idx: usize,
    pub(super) current_doc: Option<DocInfo>,
    approximate_upper_bound: f32,
    // Stored block maxima may use partition-local statistics. Queries scored
    // with corpus-wide statistics must use a scorer-provided bound instead.
    pub(super) use_scorer_upper_bound: bool,
    // The union posting drives iteration and pruning; these terms produce the
    // exact score for each candidate document.
    grouped_terms: Option<Arc<[GroupedTermScorer]>>,
    // Position cursors temporarily own this buffer and return it on drop. This
    // keeps repeated cursor creation allocation-free without lending a slice
    // out of the interior-mutable compressed state.
    pub(super) position_scratch: RefCell<Option<Vec<u32>>>,

    // for compressed posting list
    pub(super) compressed: Option<UnsafeCell<CompressedState>>,
}

#[derive(Clone)]
pub(super) struct CompressedState {
    pub(super) block_idx: usize,
    pub(super) doc_ids: Vec<u32>,
    pub(super) freqs: Vec<u32>,
    frequency_offset: usize,
    frequency_block_idx: Option<usize>,
    #[cfg(test)]
    frequency_blocks_decoded: usize,
    #[cfg(test)]
    impact_bound_computations: usize,
    // Scratch for full bitpacked blocks and packed position groups. Most
    // cursors of short posting lists only ever decode a varint tail block or
    // never read positions, so both are allocated on first use.
    buffer: Option<Box<[u32; MAX_POSTING_BLOCK_SIZE]>>,
    position_block_idx: Option<usize>,
    position_values: Vec<u32>,
    position_offsets: Vec<usize>,
    // Seek state for PackedDelta position blocks: the lazily-built group
    // header index, the last unpacked group (memoized), the decoded varint
    // tail, and the block's total delta count. Together these let a phrase
    // check decode just the candidate doc's positions instead of the whole
    // 256-doc position block.
    position_group_offsets: Vec<usize>,
    position_unpacked_group: Option<Box<[u32; BLOCK_SIZE]>>,
    position_unpacked_group_idx: Option<usize>,
    position_tail: Vec<u32>,
    position_total_deltas: usize,
    pub(super) block_max_window: BlockMaxWindow,
    // Lucene-style anchored impact score caches: one slot per level, keyed by
    // the entry the block cursor currently sits in. Each holds
    // (entry_idx, doc_up_to, max_score). See `impact_level0`/`impact_level1`.
    pub(super) level0_cache: Option<(usize, u32, f32)>,
    level1_cache: Option<(usize, u32, f32)>,
}

impl CompressedState {
    fn new(block_size: usize) -> Self {
        Self {
            block_idx: 0,
            doc_ids: Vec::with_capacity(block_size),
            freqs: Vec::with_capacity(block_size),
            frequency_offset: 0,
            frequency_block_idx: None,
            #[cfg(test)]
            frequency_blocks_decoded: 0,
            #[cfg(test)]
            impact_bound_computations: 0,
            buffer: None,
            position_block_idx: None,
            position_values: Vec::new(),
            position_offsets: Vec::new(),
            position_group_offsets: Vec::new(),
            position_unpacked_group: None,
            position_unpacked_group_idx: None,
            position_tail: Vec::new(),
            position_total_deltas: 0,
            block_max_window: BlockMaxWindow::new(),
            level0_cache: None,
            level1_cache: None,
        }
    }

    #[inline]
    fn decompress_doc_ids(
        &mut self,
        block: &[u8],
        block_idx: usize,
        num_blocks: usize,
        length: u32,
        tail_codec: crate::scalar::inverted::PostingTailCodec,
        block_size: usize,
    ) {
        self.doc_ids.clear();
        self.freqs.clear();
        self.frequency_block_idx = None;

        let remainder = length as usize % block_size;
        self.frequency_offset = if block_idx + 1 == num_blocks && remainder != 0 {
            decompress_posting_remainder_doc_ids(
                block,
                remainder,
                tail_codec,
                block_size,
                &mut self.doc_ids,
            )
        } else {
            decompress_posting_block_doc_ids(
                block,
                &mut self
                    .buffer
                    .get_or_insert_with(|| Box::new([0; MAX_POSTING_BLOCK_SIZE]))[..],
                &mut self.doc_ids,
                block_size,
            )
        };
        self.block_idx = block_idx;
        self.position_block_idx = None;
        self.position_values.clear();
        self.position_offsets.clear();
    }

    #[inline]
    fn decompress_frequencies(
        &mut self,
        block: &[u8],
        block_idx: usize,
        num_blocks: usize,
        length: u32,
        tail_codec: crate::scalar::inverted::PostingTailCodec,
        block_size: usize,
    ) {
        debug_assert_eq!(self.block_idx, block_idx);
        if self.frequency_block_idx == Some(block_idx) {
            return;
        }
        self.freqs.clear();
        let remainder = length as usize % block_size;
        if block_idx + 1 == num_blocks && remainder != 0 {
            decompress_posting_remainder_frequencies(
                block,
                self.frequency_offset,
                remainder,
                tail_codec,
                &mut self.freqs,
            );
        } else {
            decompress_posting_block_frequencies(
                block,
                self.frequency_offset,
                &mut self
                    .buffer
                    .get_or_insert_with(|| Box::new([0; MAX_POSTING_BLOCK_SIZE]))[..],
                &mut self.freqs,
                block_size,
            );
        }
        self.frequency_block_idx = Some(block_idx);
        #[cfg(test)]
        {
            self.frequency_blocks_decoded += 1;
        }
    }
}

#[derive(Clone)]
pub(super) struct BlockMaxWindow {
    // Sliding block range used for Lucene-style getMaxScore(upTo). The deque is
    // monotonic by score and covers blocks in [start_block_idx, next_block_idx).
    start_block_idx: usize,
    next_block_idx: usize,
    max_scores: VecDeque<(usize, f32)>,
    pub(super) impact_score_cache: ImpactScoreCache,
}

pub(super) struct BlockMaxScore {
    pub(super) score: f32,
    #[cfg(test)]
    pub(super) blocks_scanned: usize,
}

impl BlockMaxWindow {
    fn new() -> Self {
        Self {
            start_block_idx: 0,
            next_block_idx: 0,
            max_scores: VecDeque::new(),
            impact_score_cache: ImpactScoreCache::default(),
        }
    }

    fn reset(&mut self, start_block_idx: usize) {
        self.start_block_idx = start_block_idx;
        self.next_block_idx = start_block_idx;
        self.max_scores.clear();
    }

    fn max_score_up_to<S: Scorer + ?Sized>(
        &mut self,
        list: &CompressedPostingList,
        start_block_idx: usize,
        up_to: u64,
        query_weight: f32,
        scorer: &S,
    ) -> BlockMaxScore {
        if start_block_idx >= list.blocks.len() {
            self.reset(start_block_idx);
            return BlockMaxScore {
                score: 0.0,
                #[cfg(test)]
                blocks_scanned: 0,
            };
        }
        if start_block_idx < self.start_block_idx || start_block_idx > self.next_block_idx {
            self.reset(start_block_idx);
        }
        self.start_block_idx = start_block_idx;
        while matches!(self.max_scores.front(), Some((block_idx, _)) if *block_idx < start_block_idx)
        {
            self.max_scores.pop_front();
        }

        if list.block_least_doc_id(start_block_idx) as u64 > up_to {
            self.reset(start_block_idx);
            return BlockMaxScore {
                score: 0.0,
                #[cfg(test)]
                blocks_scanned: 0,
            };
        }

        // 256-document-block postings score quantized document lengths, which can be shorter than
        // the exact lengths used to bake a legacy max score. Without impacts,
        // use the score-independent BM25 ceiling instead of that stale bound.
        if list.block_size == MAX_POSTING_BLOCK_SIZE {
            self.reset(start_block_idx);
            return BlockMaxScore {
                score: scorer_upper_bound(query_weight, scorer),
                #[cfg(test)]
                blocks_scanned: 0,
            };
        }

        self.next_block_idx = self.next_block_idx.max(start_block_idx);
        #[cfg(test)]
        let mut blocks_scanned = 0;
        while self.next_block_idx < list.blocks.len()
            && list.block_least_doc_id(self.next_block_idx) as u64 <= up_to
        {
            let score = match list.impacts.as_ref() {
                Some(impacts) => impacts.level0_score_cached(
                    self.next_block_idx,
                    query_weight,
                    scorer,
                    &mut self.impact_score_cache,
                ),
                None => list.block_max_score(self.next_block_idx),
            };
            while matches!(self.max_scores.back(), Some((_, old_score)) if *old_score <= score) {
                self.max_scores.pop_back();
            }
            self.max_scores.push_back((self.next_block_idx, score));
            self.next_block_idx += 1;
            #[cfg(test)]
            {
                blocks_scanned += 1;
            }
        }

        let score = self
            .max_scores
            .front()
            .map(|(_, score)| *score)
            .unwrap_or(0.0);
        BlockMaxScore {
            score,
            #[cfg(test)]
            blocks_scanned,
        }
    }
}

impl Debug for PostingIterator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostingIterator")
            .field(
                "doc",
                &self
                    .doc()
                    .map(|doc| doc.doc_id())
                    .unwrap_or(TERMINATED_DOC_ID),
            )
            .field("approximate_upper_bound", &self.approximate_upper_bound)
            .field("token_id", &self.token_id)
            .finish()
    }
}

impl PartialEq for PostingIterator {
    fn eq(&self, other: &Self) -> bool {
        self.token_id == other.token_id && self.position == other.position
    }
}

impl Eq for PostingIterator {}

impl PartialOrd for PostingIterator {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PostingIterator {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self.doc(), other.doc()) {
            (Some(doc1), Some(doc2)) => doc1
                .cmp(&doc2)
                .then(
                    self.approximate_upper_bound
                        .total_cmp(&other.approximate_upper_bound),
                )
                .then(self.token_id.cmp(&other.token_id))
                .then(self.position.cmp(&other.position)),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => self
                .approximate_upper_bound
                .total_cmp(&other.approximate_upper_bound)
                .then(self.token_id.cmp(&other.token_id))
                .then(self.position.cmp(&other.position)),
        }
    }
}

impl PostingIterator {
    pub(super) fn block_idx_for_doc(
        &self,
        list: &CompressedPostingList,
        mut block_idx: usize,
        least_id: u32,
    ) -> usize {
        let mut linear_skips = 0;
        while block_idx + 1 < list.blocks.len() && linear_skips < LINEAR_BLOCK_SKIP_LIMIT {
            if list.block_least_doc_id(block_idx + 1) > least_id {
                return block_idx;
            }
            block_idx += 1;
            linear_skips += 1;
        }

        if block_idx + 1 >= list.blocks.len() {
            return block_idx;
        }

        if let Some(impacts) = list.impacts.as_ref()
            && let Some(block_idx) =
                self.block_idx_for_doc_with_impacts(list, impacts, block_idx, least_id)
        {
            return block_idx;
        }

        self.block_idx_for_doc_by_least_doc_id(list, block_idx, least_id, list.blocks.len())
    }

    fn block_idx_for_doc_with_impacts(
        &self,
        list: &CompressedPostingList,
        impacts: &ImpactSkipData,
        mut block_idx: usize,
        least_id: u32,
    ) -> Option<usize> {
        while block_idx + 1 < list.blocks.len() {
            let group_idx = (block_idx + 1) / IMPACT_LEVEL1_BLOCKS;
            let group_end = ((group_idx + 1) * IMPACT_LEVEL1_BLOCKS).min(list.blocks.len());
            let group_doc_up_to = impacts.level1_doc_up_to(group_idx)?;
            if group_doc_up_to < least_id {
                block_idx = group_end - 1;
                continue;
            }
            if group_doc_up_to == least_id {
                return Some(group_end - 1);
            }
            return Some(
                self.block_idx_for_doc_by_least_doc_id(list, block_idx, least_id, group_end),
            );
        }
        Some(block_idx)
    }

    fn block_idx_for_doc_by_least_doc_id(
        &self,
        list: &CompressedPostingList,
        block_idx: usize,
        least_id: u32,
        right: usize,
    ) -> usize {
        let mut left = block_idx + 1;
        let mut right = right;
        while left < right {
            let mid = left + (right - left) / 2;
            if list.block_least_doc_id(mid) <= least_id {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        left - 1
    }

    #[inline]
    pub(super) fn block_end_doc(&self) -> u64 {
        self.next_block_first_doc()
            .map(|doc| doc.saturating_sub(1))
            .unwrap_or(TERMINATED_DOC_ID)
    }

    /// Level1 bound of the group holding the current block, for group-wide
    /// skipping: (group doc_up_to, group max score). `None` when the list has
    /// no impact skip data or the group entry is missing/malformed.
    pub(super) fn impact_group_bound<S: Scorer + ?Sized>(&self, scorer: &S) -> Option<(u64, f32)> {
        match self.list {
            PostingList::Compressed(ref list) => {
                let impacts = list.impacts.as_ref()?;
                let (doc_up_to, score) = self.impact_level1(impacts, scorer);
                if doc_up_to == u32::MAX {
                    return None;
                }
                Some((u64::from(doc_up_to), score))
            }
            PostingList::Plain(_) => None,
        }
    }

    #[inline]
    pub(super) fn compressed_state_ptr(&self) -> *mut CompressedState {
        debug_assert!(self.compressed.is_some());
        // this method is called very frequently, so we prefer to use `UnsafeCell` instead of
        // `RefCell` to avoid the overhead of runtime borrow checking
        self.compressed.as_ref().unwrap().get()
    }

    #[inline]
    pub(super) fn ensure_compressed_block_ptr(
        &self,
        list: &CompressedPostingList,
        block_idx: usize,
    ) -> *mut CompressedState {
        let compressed = unsafe { &mut *self.ensure_compressed_doc_ids_ptr(list, block_idx) };
        if compressed.frequency_block_idx != Some(block_idx) {
            let block = list.blocks.value(block_idx);
            compressed.decompress_frequencies(
                block,
                block_idx,
                list.blocks.len(),
                list.length,
                list.posting_tail_codec,
                list.block_size,
            );
        }
        compressed as *mut CompressedState
    }

    #[inline]
    pub(super) fn ensure_compressed_doc_ids_ptr(
        &self,
        list: &CompressedPostingList,
        block_idx: usize,
    ) -> *mut CompressedState {
        let compressed = unsafe { &mut *self.compressed_state_ptr() };
        if compressed.block_idx != block_idx || compressed.doc_ids.is_empty() {
            let block = list.blocks.value(block_idx);
            compressed.decompress_doc_ids(
                block,
                block_idx,
                list.blocks.len(),
                list.length,
                list.posting_tail_codec,
                list.block_size,
            );
        }
        compressed as *mut CompressedState
    }

    #[cfg(test)]
    pub(crate) fn new(
        token: String,
        token_id: u32,
        position: u32,
        list: PostingList,
        num_doc: usize,
    ) -> Self {
        Self::with_query_weight(token, token_id, position, 1.0, list, num_doc)
    }

    pub(crate) fn with_query_weight(
        token: String,
        token_id: u32,
        position: u32,
        query_weight: f32,
        list: PostingList,
        num_doc: usize,
    ) -> Self {
        // Keep a finite scorer-independent BM25 ceiling even when index stats
        // drift after appends.  The ceiling includes f32 evaluation error; an
        // INFINITY bound can never park the iterator in the WAND tail.
        let approximate_upper_bound = match &list {
            PostingList::Compressed(posting) if posting.impacts.is_some() => f32::INFINITY,
            PostingList::Compressed(posting) if posting.block_size == MAX_POSTING_BLOCK_SIZE => {
                conservative_bm25_upper_bound(query_weight)
            }
            _ => match list.max_score() {
                Some(max_score) => max_score,
                None => conservative_bm25_upper_bound(idf(list.len(), num_doc)),
            },
        };
        let compressed = match &list {
            PostingList::Compressed(list) => {
                Some(UnsafeCell::new(CompressedState::new(list.block_size)))
            }
            PostingList::Plain(_) => None,
        };

        let mut posting = Self {
            token,
            token_id,
            position,
            query_weight,
            list,
            index: 0,
            block_idx: 0,
            current_doc: None,
            approximate_upper_bound,
            use_scorer_upper_bound: false,
            grouped_terms: None,
            position_scratch: RefCell::new(Some(Vec::new())),
            compressed,
        };
        posting.refresh_current_doc();
        posting
    }

    pub(in crate::scalar::inverted) fn with_scorer_upper_bound(mut self) -> Self {
        self.use_scorer_upper_bound = true;
        self
    }

    pub(in crate::scalar::inverted) fn with_grouped_terms(
        mut self,
        terms: Arc<[GroupedTermScorer]>,
    ) -> Self {
        // Grouped postings are built at query time with the same scoring
        // lengths and scorer as this cursor.  Their baked list maximum is the
        // authoritative bound, including on 256-document blocks where normal
        // persisted postings must fall back to a scorer-wide ceiling.
        self.approximate_upper_bound = self.list.max_score().unwrap_or(f32::INFINITY);
        self.grouped_terms = Some(terms);
        self
    }

    /// Create an independent cursor over the same immutable posting payload.
    ///
    /// Staged cross-column execution uses one cursor to enumerate exact
    /// membership before asynchronous candidate-side document reads, then a
    /// fresh cursor to compute the canonical score. Arrow buffers and grouped
    /// term tables remain shared; only mutable decode and position state is
    /// recreated.
    pub(in crate::scalar::inverted) fn fork_from_start(&self) -> Self {
        let compressed = match &self.list {
            PostingList::Compressed(list) => {
                Some(UnsafeCell::new(CompressedState::new(list.block_size)))
            }
            PostingList::Plain(_) => None,
        };
        let mut posting = Self {
            token: self.token.clone(),
            token_id: self.token_id,
            position: self.position,
            query_weight: self.query_weight,
            list: self.list.clone(),
            index: 0,
            block_idx: 0,
            current_doc: None,
            approximate_upper_bound: self.approximate_upper_bound,
            use_scorer_upper_bound: self.use_scorer_upper_bound,
            grouped_terms: self.grouped_terms.clone(),
            position_scratch: RefCell::new(Some(Vec::new())),
            compressed,
        };
        posting.refresh_current_doc();
        posting
    }

    #[inline]
    pub(crate) fn term_index(&self) -> u32 {
        self.position
    }

    #[inline]
    pub(crate) fn token(&self) -> &str {
        &self.token
    }

    #[inline]
    pub(super) fn approximate_upper_bound(&self) -> f32 {
        self.approximate_upper_bound
    }

    /// Upper bound for the frequency clamp used by bulk conjunction search.
    /// No-impact postings scored in a corpus-global scorer space cannot reuse
    /// their persisted partition-local maximum. Impact-backed postings retain
    /// their existing fast path.
    #[inline]
    pub(super) fn frequency_clamp_upper_bound<S: Scorer + ?Sized>(&self, scorer: &S) -> f32 {
        if self.use_scorer_upper_bound {
            scorer_upper_bound(self.query_weight, scorer)
        } else {
            self.approximate_upper_bound
        }
    }

    /// Tightest known list-wide score bound. Impact lists answer from the
    /// baked doc-weight slab (the data-driven equivalent of the max_score the
    /// non-impact format bakes at build time); everything else falls back to
    /// `approximate_upper_bound`. A finite, tight global bound lets lagging
    /// iterators park in the WAND tail instead of being force-advanced.
    #[inline]
    pub(super) fn global_upper_bound<S: Scorer + ?Sized>(&self, scorer: &S) -> f32 {
        if self.has_grouped_terms() {
            return self.approximate_upper_bound;
        }
        if self.query_weight <= 0.0 {
            return 0.0;
        }
        if let PostingList::Compressed(ref list) = self.list
            && let Some(impacts) = list.impacts.as_ref()
        {
            let compressed = unsafe { &mut *self.compressed_state_ptr() };
            return self.query_weight
                * impacts.global_max_doc_weight_cached(
                    scorer,
                    &mut compressed.block_max_window.impact_score_cache,
                );
        }
        if self.use_scorer_upper_bound {
            return scorer_upper_bound(self.query_weight, scorer);
        }
        if let PostingList::Compressed(ref list) = self.list
            && list.block_size == MAX_POSTING_BLOCK_SIZE
        {
            return scorer_upper_bound(self.query_weight, scorer);
        }
        self.approximate_upper_bound
    }

    #[inline]
    pub(super) fn score<S: Scorer + ?Sized>(&self, scorer: &S, freq: u32, doc_length: u32) -> f32 {
        if let (Some(grouped_terms), Some(doc)) = (&self.grouped_terms, self.current_doc) {
            return grouped_terms
                .iter()
                .map(|term| term.score(doc.doc_id(), doc_length, scorer))
                .sum();
        }
        self.query_weight * scorer.doc_weight(freq, doc_length)
    }

    #[inline]
    pub(super) fn has_grouped_terms(&self) -> bool {
        self.grouped_terms.is_some()
    }

    #[inline]
    pub(super) fn cost(&self) -> usize {
        self.list.len()
    }

    #[inline]
    fn empty(&self) -> bool {
        self.index >= self.list.len()
    }

    #[inline]
    pub(super) fn doc(&self) -> Option<DocInfo> {
        let current_doc = self.current_doc?;
        match self.list {
            PostingList::Compressed(ref list) => {
                if current_doc.frequency() != 0 {
                    return Some(current_doc);
                }
                let block_idx = self.index >> list.block_shift();
                let block_offset = self.index & list.block_mask();
                let compressed = unsafe { &mut *self.ensure_compressed_block_ptr(list, block_idx) };
                Some(DocInfo::Raw(RawDocInfo {
                    doc_id: current_doc.doc_id() as u32,
                    frequency: compressed.freqs[block_offset],
                }))
            }
            PostingList::Plain(_) => Some(current_doc),
        }
    }

    #[cfg(test)]
    pub(super) fn frequency_blocks_decoded(&self) -> usize {
        self.compressed
            .as_ref()
            .map(|compressed| unsafe { &*compressed.get() }.frequency_blocks_decoded)
            .unwrap_or_default()
    }

    #[cfg(test)]
    pub(super) fn impact_bound_computations(&self) -> usize {
        self.compressed
            .as_ref()
            .map(|compressed| unsafe { &*compressed.get() }.impact_bound_computations)
            .unwrap_or_default()
    }

    /// The posting's current local document before it is moved into a scorer.
    ///
    /// Cross-partition row-address merging uses this only to derive a
    /// conservative lower bound for lazy source activation. It is not an
    /// exact match decision: visibility and phrase confirmation still happen
    /// in [`WandCursor`].
    pub(in crate::scalar::inverted) fn current_doc_id(&self) -> Option<u64> {
        self.current_doc.map(|doc| doc.doc_id())
    }

    fn refresh_current_doc(&mut self) {
        if self.empty() {
            self.current_doc = None;
            return;
        }

        let current_doc = match self.list {
            PostingList::Compressed(ref list) => {
                let block_idx = self.index >> list.block_shift();
                let block_offset = self.index & list.block_mask();
                let compressed =
                    unsafe { &mut *self.ensure_compressed_doc_ids_ptr(list, block_idx) };

                // Frequency decoding is deferred until a scoring or position
                // consumer calls `doc()` / `ensure_compressed_block_ptr`.
                let doc_id = compressed.doc_ids[block_offset];
                let doc = DocInfo::Raw(RawDocInfo {
                    doc_id,
                    frequency: 0,
                });
                Some(doc)
            }
            PostingList::Plain(ref list) => Some(DocInfo::Located(list.doc(self.index))),
        };
        self.current_doc = current_doc;
    }

    pub(super) fn position_cursor(&self) -> Result<PositionCursor<'_>> {
        #[cfg(test)]
        {
            POSITION_CURSOR_CALLS.with(|calls| calls.set(calls.get() + 1));
        }
        match self.list {
            PostingList::Plain(ref list) => {
                let positions = list.positions.as_ref().ok_or_else(|| {
                    Error::index(format!(
                        "positions are missing for token {:?} (token id {}, query position {})",
                        self.token, self.token_id, self.position
                    ))
                })?;
                let start = positions.value_offsets()[self.index] as usize;
                let end = positions.value_offsets()[self.index + 1] as usize;
                Ok(PositionCursor::new(
                    PositionValues::Owned(
                        positions.values().as_primitive::<Int32Type>().values()[start..end]
                            .iter()
                            .map(|value| *value as u32)
                            .collect(),
                    ),
                    self.position as i32,
                ))
            }
            PostingList::Compressed(ref list) => match list.positions.as_ref().ok_or_else(|| {
                Error::index(format!(
                    "positions are missing for token {:?} (token id {}, query position {})",
                    self.token, self.token_id, self.position
                ))
            })? {
                CompressedPositionStorage::LegacyPerDoc(positions) => {
                    let positions = positions.value(self.index);
                    let positions = decompress_positions(positions.as_binary());
                    Ok(PositionCursor::new(
                        PositionValues::Owned(positions),
                        self.position as i32,
                    ))
                }
                CompressedPositionStorage::SharedStream(stream) => {
                    let block_idx = self.index >> list.block_shift();
                    let block_offset = self.index & list.block_mask();
                    let compressed =
                        unsafe { &mut *self.ensure_compressed_block_ptr(list, block_idx) };
                    match stream.codec() {
                        PositionStreamCodec::PackedDelta => {
                            // Seekable layout: decode only the candidate doc's
                            // positions. Per-block seek state resets when the
                            // block cursor moves; the group header index and
                            // varint tail fill in lazily as candidates touch
                            // them.
                            if compressed.position_block_idx != Some(block_idx) {
                                compressed.position_group_offsets.clear();
                                compressed.position_group_offsets.push(0);
                                compressed.position_tail.clear();
                                compressed.position_unpacked_group_idx = None;
                                compressed.position_offsets.clear();
                                compressed
                                    .position_offsets
                                    .reserve(compressed.freqs.len() + 1);
                                compressed.position_offsets.push(0);
                                let mut offset = 0usize;
                                for &freq in &compressed.freqs {
                                    offset += freq as usize;
                                    compressed.position_offsets.push(offset);
                                }
                                compressed.position_total_deltas = offset;
                                compressed.position_block_idx = Some(block_idx);
                            }
                            let delta_start = compressed.position_offsets[block_offset];
                            let delta_end = compressed.position_offsets[block_offset + 1];
                            let mut position_values = self
                                .position_scratch
                                .borrow_mut()
                                .take()
                                .unwrap_or_default();
                            if let Err(error) = seek_packed_doc_positions(
                                stream.block(block_idx),
                                compressed.position_total_deltas,
                                delta_start..delta_end,
                                &mut compressed.position_group_offsets,
                                compressed
                                    .position_unpacked_group
                                    .get_or_insert_with(|| Box::new([0; BLOCK_SIZE])),
                                &mut compressed.position_unpacked_group_idx,
                                &mut compressed.position_tail,
                                &mut position_values,
                            ) {
                                *self.position_scratch.borrow_mut() = Some(position_values);
                                return Err(Error::index(format!(
                                    "failed to decode positions for token {:?} (token id {}, query position {}) at posting index {}: {error}",
                                    self.token, self.token_id, self.position, self.index
                                )));
                            }
                            Ok(PositionCursor::new(
                                PositionValues::Recycled(RecycledPositionValues::new(
                                    position_values,
                                    &self.position_scratch,
                                )),
                                self.position as i32,
                            ))
                        }
                        PositionStreamCodec::VarintDocDelta => {
                            if compressed.position_block_idx != Some(block_idx) {
                                compressed.position_values.clear();
                                decode_position_stream_block(
                                    stream.block(block_idx),
                                    compressed.freqs.as_slice(),
                                    stream.codec(),
                                    &mut compressed.position_values,
                                )
                                .map_err(|error| {
                                    Error::index(format!(
                                        "failed to decode positions for token {:?} (token id {}, query position {}) in block {block_idx}: {error}",
                                        self.token, self.token_id, self.position
                                    ))
                                })?;
                                compressed.position_offsets.clear();
                                compressed
                                    .position_offsets
                                    .reserve(compressed.freqs.len() + 1);
                                compressed.position_offsets.push(0);
                                let mut offset = 0usize;
                                for &freq in &compressed.freqs {
                                    offset += freq as usize;
                                    compressed.position_offsets.push(offset);
                                }
                                compressed.position_block_idx = Some(block_idx);
                            }
                            let start = compressed.position_offsets[block_offset];
                            let end = compressed.position_offsets[block_offset + 1];
                            let mut position_values = self
                                .position_scratch
                                .borrow_mut()
                                .take()
                                .unwrap_or_default();
                            position_values.clear();
                            position_values
                                .extend_from_slice(&compressed.position_values[start..end]);
                            Ok(PositionCursor::new(
                                PositionValues::Recycled(RecycledPositionValues::new(
                                    position_values,
                                    &self.position_scratch,
                                )),
                                self.position as i32,
                            ))
                        }
                    }
                }
            },
        }
    }

    // move to the next doc id that is greater than or equal to least_id
    pub(super) fn next(&mut self, least_id: u64) {
        match self.list {
            PostingList::Compressed(ref list) => {
                debug_assert!(least_id <= u32::MAX as u64);
                let least_id = least_id as u32;
                let shift = list.block_shift();
                let block_idx = self.block_idx_for_doc(list, self.index >> shift, least_id);
                self.index = self.index.max(block_idx << shift);
                let length = list.length as usize;
                while self.index < length {
                    let block_idx = self.index >> shift;
                    let block_offset = self.index & list.block_mask();
                    let compressed =
                        unsafe { &mut *self.ensure_compressed_block_ptr(list, block_idx) };
                    let in_block = &compressed.doc_ids[block_offset..];
                    let offset_in_block = in_block.partition_point(|&doc_id| doc_id < least_id);
                    let new_offset = block_offset + offset_in_block;
                    if new_offset < compressed.doc_ids.len() {
                        self.index = (block_idx << shift) + new_offset;
                        self.block_idx = block_idx;
                        self.current_doc = Some(DocInfo::Raw(RawDocInfo {
                            doc_id: compressed.doc_ids[new_offset],
                            frequency: compressed.freqs[new_offset],
                        }));
                        return;
                    }
                    if block_idx + 1 >= list.blocks.len() {
                        self.index = length;
                        self.block_idx = self.index >> shift;
                        self.current_doc = None;
                        break;
                    }
                    self.index = (block_idx + 1) << shift;
                }
                self.block_idx = self.index >> shift;
                self.current_doc = None;
            }
            PostingList::Plain(ref list) => {
                self.index += list.row_ids[self.index..].partition_point(|&id| id < least_id);
                self.current_doc = (!self.empty()).then(|| DocInfo::Located(list.doc(self.index)));
            }
        }
    }

    /// Move to the next document without materializing its frequency. AND
    /// intersection uses this until a document survives every required list.
    pub(super) fn next_doc_id(&mut self, least_id: u64, is_vectorized_search_enabled: bool) {
        match self.list {
            PostingList::Compressed(ref list) => {
                debug_assert!(least_id <= u32::MAX as u64);
                let least_id = least_id as u32;
                let shift = list.block_shift();
                let block_idx = self.block_idx_for_doc(list, self.index >> shift, least_id);
                self.index = self.index.max(block_idx << shift);
                let length = list.length as usize;
                while self.index < length {
                    let block_idx = self.index >> shift;
                    let block_offset = self.index & list.block_mask();
                    let compressed =
                        unsafe { &mut *self.ensure_compressed_doc_ids_ptr(list, block_idx) };
                    let new_offset = if is_vectorized_search_enabled {
                        find_next_geq_in_block(&compressed.doc_ids, block_offset, least_id)
                    } else {
                        block_offset
                            + compressed.doc_ids[block_offset..]
                                .partition_point(|&doc_id| doc_id < least_id)
                    };
                    if new_offset < compressed.doc_ids.len() {
                        self.index = (block_idx << shift) + new_offset;
                        self.block_idx = block_idx;
                        // Frequencies stay lazy, but once this block's stream is
                        // decoded every later candidate in it can carry its
                        // frequency for free instead of re-materializing in `doc`.
                        let frequency = if compressed.frequency_block_idx == Some(block_idx) {
                            compressed.freqs[new_offset]
                        } else {
                            0
                        };
                        self.current_doc = Some(DocInfo::Raw(RawDocInfo {
                            doc_id: compressed.doc_ids[new_offset],
                            frequency,
                        }));
                        return;
                    }
                    if block_idx + 1 >= list.blocks.len() {
                        self.index = length;
                        self.block_idx = self.index >> shift;
                        self.current_doc = None;
                        break;
                    }
                    self.index = (block_idx + 1) << shift;
                }
                self.block_idx = self.index >> shift;
                self.current_doc = None;
            }
            PostingList::Plain(ref list) => {
                self.index += list.row_ids[self.index..].partition_point(|&id| id < least_id);
                self.current_doc = (!self.empty()).then(|| DocInfo::Located(list.doc(self.index)));
            }
        }
    }

    pub(super) fn shallow_next(&mut self, least_id: u64) {
        match self.list {
            PostingList::Compressed(ref list) => {
                debug_assert!(least_id <= u32::MAX as u64);
                let least_id = least_id as u32;
                self.block_idx = self.block_idx_for_doc(list, self.block_idx, least_id);
            }
            PostingList::Plain(_) => {
                // we don't have block max score for legacy index,
                // and no compression, so just do nothing
            }
        }
    }

    /// Anchored level0 impact bound of the current block: (doc_up_to, max_score),
    /// memoized until the block cursor moves. Malformed entries degrade to
    /// (u32::MAX, INFINITY), which keeps pruning safe by making the block look
    /// unskippable.
    #[inline]
    fn impact_level0<S: Scorer + ?Sized>(
        &self,
        impacts: &ImpactSkipData,
        scorer: &S,
    ) -> (u32, f32) {
        let compressed = unsafe { &mut *self.compressed_state_ptr() };
        if let Some((block_idx, doc_up_to, score)) = compressed.level0_cache
            && block_idx == self.block_idx
        {
            return (doc_up_to, score);
        }
        let doc_up_to = impacts.level0_doc_up_to(self.block_idx).unwrap_or(u32::MAX);
        let score = impacts.level0_score_cached(
            self.block_idx,
            self.query_weight,
            scorer,
            &mut compressed.block_max_window.impact_score_cache,
        );
        #[cfg(test)]
        {
            compressed.impact_bound_computations += 1;
        }
        compressed.level0_cache = Some((self.block_idx, doc_up_to, score));
        (doc_up_to, score)
    }

    /// Anchored level1 impact bound of the group holding the current block,
    /// memoized until the cursor crosses a group boundary.
    #[inline]
    fn impact_level1<S: Scorer + ?Sized>(
        &self,
        impacts: &ImpactSkipData,
        scorer: &S,
    ) -> (u32, f32) {
        let group_idx = self.block_idx / IMPACT_LEVEL1_BLOCKS;
        let compressed = unsafe { &mut *self.compressed_state_ptr() };
        if let Some((cached_group_idx, doc_up_to, score)) = compressed.level1_cache
            && cached_group_idx == group_idx
        {
            return (doc_up_to, score);
        }
        let doc_up_to = impacts.level1_doc_up_to(group_idx).unwrap_or(u32::MAX);
        let score = impacts.level1_score_cached(
            group_idx,
            self.query_weight,
            scorer,
            &mut compressed.block_max_window.impact_score_cache,
        );
        #[cfg(test)]
        {
            compressed.impact_bound_computations += 1;
        }
        compressed.level1_cache = Some((group_idx, doc_up_to, score));
        (doc_up_to, score)
    }

    #[inline]
    pub(super) fn block_max_score<S: Scorer + ?Sized>(&self, scorer: &S) -> f32 {
        match self.list {
            PostingList::Compressed(ref list) => {
                if self.has_grouped_terms() && list.block_size == MAX_POSTING_BLOCK_SIZE {
                    return self.approximate_upper_bound;
                }
                if let Some(impacts) = list.impacts.as_ref() {
                    return self.impact_level0(impacts, scorer).1;
                }
                if self.use_scorer_upper_bound {
                    return scorer_upper_bound(self.query_weight, scorer);
                }
                if list.block_size == MAX_POSTING_BLOCK_SIZE {
                    return scorer_upper_bound(self.query_weight, scorer);
                }
                list.block_max_score(self.block_idx)
            }
            PostingList::Plain(_) if self.use_scorer_upper_bound => {
                scorer_upper_bound(self.query_weight, scorer)
            }
            PostingList::Plain(_) => self.approximate_upper_bound,
        }
    }

    /// Tight max-score bound over `[current block, up_to]`. The common case —
    /// a window ending inside the current block — answers from the anchored
    /// level0 memo; wider windows fall back to the sliding block-max deque,
    /// which scores each block once as it slides forward.
    #[inline]
    pub(super) fn block_max_score_up_to_with_stats<S: Scorer + ?Sized>(
        &self,
        up_to: u64,
        scorer: &S,
    ) -> BlockMaxScore {
        match self.list {
            PostingList::Compressed(ref list) => {
                if self.has_grouped_terms() && list.block_size == MAX_POSTING_BLOCK_SIZE {
                    return BlockMaxScore {
                        score: self.approximate_upper_bound,
                        #[cfg(test)]
                        blocks_scanned: 0,
                    };
                }
                if let Some(impacts) = list.impacts.as_ref() {
                    let (level0_up_to, level0_score) = self.impact_level0(impacts, scorer);
                    if up_to <= u64::from(level0_up_to) {
                        return BlockMaxScore {
                            score: level0_score,
                            #[cfg(test)]
                            blocks_scanned: 0,
                        };
                    }
                }
                if self.use_scorer_upper_bound {
                    return BlockMaxScore {
                        score: scorer_upper_bound(self.query_weight, scorer),
                        #[cfg(test)]
                        blocks_scanned: 0,
                    };
                }
                let compressed = unsafe { &mut *self.compressed_state_ptr() };
                compressed.block_max_window.max_score_up_to(
                    list,
                    self.block_idx,
                    up_to,
                    self.query_weight,
                    scorer,
                )
            }
            PostingList::Plain(_) => BlockMaxScore {
                score: if self.use_scorer_upper_bound {
                    scorer_upper_bound(self.query_weight, scorer)
                } else {
                    self.approximate_upper_bound
                },
                #[cfg(test)]
                blocks_scanned: 0,
            },
        }
    }

    pub(super) fn window_max_score<S: Scorer + ?Sized>(
        &self,
        up_to: Option<u64>,
        scorer: &S,
    ) -> f32 {
        if let Some(up_to) = up_to
            && let PostingList::Compressed(_) = self.list
        {
            // The window end can come from other postings' block ends, so it can
            // cover several of this posting's blocks; bound all of them,
            // impacts or not.
            return self.block_max_score_up_to_with_stats(up_to, scorer).score;
        }
        self.block_max_score(scorer)
    }

    #[inline]
    pub(super) fn is_compressed(&self) -> bool {
        matches!(self.list, PostingList::Compressed(_))
    }

    #[inline]
    pub(super) fn has_next_compressed_block(&self) -> bool {
        match self.list {
            PostingList::Compressed(ref list) => self.block_idx + 1 < list.blocks.len(),
            PostingList::Plain(_) => false,
        }
    }

    pub(super) fn block_first_doc(&self) -> Option<u64> {
        match self.list {
            PostingList::Compressed(ref list) => {
                Some(list.block_least_doc_id(self.block_idx) as u64)
            }
            PostingList::Plain(ref plain) => plain.row_ids.get(self.index).cloned(),
        }
    }

    #[inline]
    pub(super) fn next_block_first_doc(&self) -> Option<u64> {
        match self.list {
            PostingList::Compressed(ref list) => {
                if self.block_idx + 1 >= list.blocks.len() {
                    return None;
                }
                Some(list.block_least_doc_id(self.block_idx + 1) as u64)
            }
            PostingList::Plain(ref plain) => plain.row_ids.get(self.index + 1).cloned(),
        }
    }

    #[cfg(test)]
    pub(super) fn validate_modern_doc_ids(&self, num_docs: usize) -> Result<()> {
        validate_modern_posting_doc_ids(&self.list, &self.token, num_docs)
    }
}

/// Validate the dense-DocId boundary before any document-length lookup.
/// Modern postings are sorted, so decoding only the final physical block is
/// sufficient to validate their maximum DocId.
pub(in crate::scalar::inverted) fn validate_modern_posting_doc_ids(
    posting: &PostingList,
    token: &str,
    num_docs: usize,
) -> Result<()> {
    let PostingList::Compressed(list) = posting else {
        return Err(Error::index(format!(
            "modern FTS posting for token {token:?} uses a legacy row-address layout"
        )));
    };
    if list.length == 0 {
        return Ok(());
    }
    let last_block_idx = list.blocks.len().checked_sub(1).ok_or_else(|| {
        Error::index(format!(
            "modern FTS posting for token {token:?} has length {} but no blocks",
            list.length
        ))
    })?;
    let mut state = CompressedState::new(list.block_size);
    state.decompress_doc_ids(
        list.blocks.value(last_block_idx),
        last_block_idx,
        list.blocks.len(),
        list.length,
        list.posting_tail_codec,
        list.block_size,
    );
    let max_doc_id = state.doc_ids.last().copied().ok_or_else(|| {
        Error::index(format!(
            "modern FTS posting for token {token:?} has an empty final block"
        ))
    })?;
    if max_doc_id as usize >= num_docs {
        return Err(Error::index(format!(
            "modern FTS posting for token {token:?} contains DocId {max_doc_id}, outside [0, {num_docs})"
        )));
    }
    Ok(())
}

#[cfg(test)]
thread_local! {
    static POSITION_CURSOR_CALLS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(super) fn take_position_cursor_calls() -> usize {
    POSITION_CURSOR_CALLS.with(|calls| calls.replace(0))
}

#[derive(Debug)]
pub(super) struct RecycledPositionValues<'a> {
    values: Option<Vec<u32>>,
    pool: &'a RefCell<Option<Vec<u32>>>,
}

impl<'a> RecycledPositionValues<'a> {
    fn new(values: Vec<u32>, pool: &'a RefCell<Option<Vec<u32>>>) -> Self {
        Self {
            values: Some(values),
            pool,
        }
    }

    fn as_slice(&self) -> &[u32] {
        self.values
            .as_deref()
            .expect("position values are present until drop")
    }
}

impl Drop for RecycledPositionValues<'_> {
    fn drop(&mut self) {
        let values = self
            .values
            .take()
            .expect("position values are present until drop");
        let mut pool = self.pool.borrow_mut();
        if pool.is_none() {
            *pool = Some(values);
        }
    }
}

#[derive(Debug)]
pub(super) enum PositionValues<'a> {
    Recycled(RecycledPositionValues<'a>),
    Owned(Vec<u32>),
}

impl<'a> PositionValues<'a> {
    pub(super) fn as_slice(&self) -> &[u32] {
        match self {
            Self::Recycled(values) => values.as_slice(),
            Self::Owned(values) => values.as_slice(),
        }
    }

    fn len(&self) -> usize {
        self.as_slice().len()
    }
}

#[derive(Debug)]
pub(super) struct PositionCursor<'a> {
    pub(super) positions: PositionValues<'a>,
    pub position_in_query: i32,
    index: usize,
}

impl<'a> PositionCursor<'a> {
    fn new(positions: PositionValues<'a>, position_in_query: i32) -> Self {
        Self {
            positions,
            position_in_query,
            index: 0,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.positions.len()
    }

    pub(super) fn relative_position(&self) -> Option<i32> {
        self.positions
            .as_slice()
            .get(self.index)
            .map(|position| *position as i32 - self.position_in_query)
    }

    /// Advance to the first position at or above `target` and return it, or
    /// `None` once the cursor is exhausted.
    ///
    /// `target` must be non-decreasing across calls: the search is confined to
    /// the not-yet-consumed suffix, so the cursor never rewinds and a returned
    /// position is never below `target`. When the cursor already sits past
    /// `target` the search is skipped entirely.
    pub(super) fn advance_to_at_least(&mut self, target: u32) -> Option<u32> {
        if self.index >= self.len() {
            return None;
        }
        let values = self.positions.as_slice();
        // The cursor has reached `target`, or is already past it. Every earlier
        // position is below the previous target and hence below this one, so the
        // insertion point has not moved: no search is needed, and the current
        // position is either the match or the bound the caller wants.
        if values[self.index] >= target {
            return Some(values[self.index]);
        }
        // Targets only grow, so the next position is a frequent answer. Check it
        // before paying for a search: a suffix binary search costs a dependent
        // load per level and its landmarks move as `index` advances, which is
        // what makes interleaved lists slow. On a mismatch here the cursor ends
        // up one position short of the target on every step.
        if let Some(next) = values
            .get(self.index + 1)
            .copied()
            .filter(|&next| next >= target)
        {
            self.index += 1;
            return Some(next);
        }
        self.index += values[self.index..].partition_point(|&pos| pos < target);
        values.get(self.index).copied()
    }

    pub(super) fn advance_to_relative(&mut self, least_relative_pos: i32) {
        if self.index >= self.len() {
            return;
        }
        let least_pos = least_relative_pos + self.position_in_query;
        let least_pos = least_pos.max(0) as u32;
        let values = self.positions.as_slice();
        self.index += values[self.index..].partition_point(|&pos| pos < least_pos);
    }
}
