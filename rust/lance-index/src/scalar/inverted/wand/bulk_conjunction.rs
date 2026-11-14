// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bulk conjunction: block-wise AND over decompressed posting blocks.

use arrow_array::Array;
use lance_core::Result;
use smallvec::SmallVec;

use crate::metrics::MetricsCollector;
use crate::scalar::inverted::builder::{BLOCK_SIZE, ScoredDoc};
use crate::scalar::inverted::encoding::MAX_POSTING_BLOCK_SIZE;
use crate::scalar::inverted::index::dequantize_doc_length;
use crate::scalar::inverted::query::FtsSearchParams;
use crate::scalar::inverted::scorer::{Scorer, bm25_doc_weight_with_norm};
use crate::scalar::inverted::{DocInfo, PostingList, RawDocInfo};

use super::bounds::{exclusive_partial_score_limit, score_sum_upper_bound_factor};
use super::collector::TopKCollector;
use super::doc_source::{DocCandidate, WandDocuments};
#[cfg(target_arch = "x86_64")]
use super::posting::{HAS_AVX2, find_next_geq_avx2};
use super::posting::{PostingIterator, find_next_geq, find_next_geq_scalar};
use super::{TERMINATED_DOC_ID, Wand, intersection};

impl<'a, S: Scorer, D: WandDocuments> Wand<'a, S, D> {
    /// Bulk conjunction search. The window ends at the nearest next-block
    /// boundary across the clauses, so within a window every clause
    /// contributes exactly one decompressed block and the intersection is a
    /// plain merge over `u32` slices — the per-candidate cost drops from a
    /// full `PostingIterator::next` call per clause to a couple of loads.
    /// Window skipping, per-candidate pruning, scoring, and heap semantics
    /// mirror the classic loop exactly, so results are identical.
    pub(super) fn and_bulk_search(
        &mut self,
        params: &FtsSearchParams,
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<DocCandidate<D::Candidate>>> {
        let limit = params.limit.unwrap_or(usize::MAX);
        if limit == 0 {
            return Ok(vec![]);
        }
        let num_lists = self.lead.len();
        let use_pairwise_intersection = num_lists >= 4
            && self.lead.iter().all(|posting| {
                matches!(&posting.list, PostingList::Compressed(list)
                    if list.block_size == MAX_POSTING_BLOCK_SIZE && list.impacts.is_some())
            });
        let phrase_slop = params.phrase_slop;
        // The bulk path never evaluates score-first windows, so keep the block
        // advance from preparing suffix bounds it would not use.
        self.invalidate_score_first_and_window();
        self.score_first_and_enabled = false;
        let mut score_order = (0..num_lists).collect::<Vec<_>>();
        score_order.sort_unstable_by_key(|&index| {
            let posting = &self.lead[index];
            (posting.position, posting.token_id)
        });

        // Per-window view of one clause's current block. Raw pointers into the
        // clause's `CompressedState`; valid for the whole window because the
        // block cursor does not move within a window (position decoding writes
        // to separate fields of the same state).
        struct WindowList {
            docs: *const u32,
            freqs: *const u32,
            // Cursor and exclusive end, as offsets within the block.
            pos: usize,
            end: usize,
            // Absolute posting index of the block's first entry.
            block_start: usize,
        }

        /// Sum one bulk-AND candidate's per-clause BM25 contributions in score
        /// order. Nested so it can name the block-local `WindowList`; the
        /// caller may run it before or after position confirmation.
        #[inline]
        fn window_bm25_score<S: Scorer + ?Sized>(
            lead: &[Box<PostingIterator>],
            wins: &[WindowList],
            offs: &[u8],
            score_order: &[usize],
            scorer: &S,
            norm_addend: Option<f32>,
            doc_length: u32,
        ) -> f32 {
            let mut score = 0.0_f32;
            for &clause_index in score_order {
                let win = &wins[clause_index];
                let posting = &lead[clause_index];
                let off = offs[clause_index];
                let freq = unsafe { *win.freqs.add(off as usize) };
                score += match norm_addend {
                    Some(addend) => posting.query_weight * bm25_doc_weight_with_norm(freq, addend),
                    None => posting.score(scorer, freq, doc_length),
                };
            }
            score
        }

        // Merge kernels: intersect the window's per-clause slices and record
        // each match as (doc, per-clause block offsets). Hand-specialized for
        // two and three clauses so cursors and bounds stay in registers; the
        // generic kernel covers other widths. Offsets fit u8 because a block
        // holds at most `MAX_POSTING_BLOCK_SIZE` (256) entries. The macro
        // stamps a scalar and an AVX2 variant — `#[target_feature]` must
        // cover the whole kernel for the vector catch-up scans to inline.
        // Kernels prune a lead doc before any follower advance when its
        // frequency-bucketed score bound (plus the other clauses' block
        // maxes) cannot beat the threshold — the score-first ordering
        // Lucene's conjunction scorer uses, with doc length dropped from the
        // bound so no doc-length load is involved. The bound is monotone
        // (doc length only shrinks a BM25 weight, and the last bucket holds
        // the clause sup), so every skipped doc would also fail the exact
        // per-candidate prune: results are unchanged, the work never happens.
        macro_rules! merge_kernels {
            ($name2:ident, $name3:ident, $name_n:ident, $docs_name2:ident, $docs_name3:ident, $geq:ident $(, #[$feat:meta])?) => {
                $(#[$feat])?
                unsafe fn $name2(
                    wins: &[WindowList],
                    freq_cannot_beat: &[bool; FREQ_LUT_BUCKETS],
                    docs_out: &mut Vec<u32>,
                    offs_out: &mut Vec<u8>,
                ) {
                    let (d0, mut p0, e0) = (wins[0].docs, wins[0].pos, wins[0].end);
                    let (d1, mut p1, e1) = (wins[1].docs, wins[1].pos, wins[1].end);
                    let f0 = wins[0].freqs;
                    unsafe {
                        while p0 < e0 {
                            let doc = *d0.add(p0);
                            let freq = (*f0.add(p0) as usize).min(FREQ_LUT_BUCKETS - 1);
                            if freq_cannot_beat[freq] {
                                p0 += 1;
                                continue;
                            }
                            p1 = $geq(d1, p1, e1, doc);
                            if p1 >= e1 {
                                return;
                            }
                            let second = *d1.add(p1);
                            if second > doc {
                                p0 = $geq(d0, p0 + 1, e0, second);
                                continue;
                            }
                            docs_out.push(doc);
                            offs_out.push(p0 as u8);
                            offs_out.push(p1 as u8);
                            p0 += 1;
                        }
                    }
                }

                $(#[$feat])?
                unsafe fn $name3(
                    wins: &[WindowList],
                    freq_cannot_beat: &[bool; FREQ_LUT_BUCKETS],
                    docs_out: &mut Vec<u32>,
                    offs_out: &mut Vec<u8>,
                ) {
                    let (d0, mut p0, e0) = (wins[0].docs, wins[0].pos, wins[0].end);
                    let (d1, mut p1, e1) = (wins[1].docs, wins[1].pos, wins[1].end);
                    let (d2, mut p2, e2) = (wins[2].docs, wins[2].pos, wins[2].end);
                    let f0 = wins[0].freqs;
                    unsafe {
                        'outer: while p0 < e0 {
                            let doc = *d0.add(p0);
                            let freq = (*f0.add(p0) as usize).min(FREQ_LUT_BUCKETS - 1);
                            if freq_cannot_beat[freq] {
                                p0 += 1;
                                continue 'outer;
                            }
                            p1 = $geq(d1, p1, e1, doc);
                            if p1 >= e1 {
                                return;
                            }
                            let second = *d1.add(p1);
                            if second > doc {
                                p0 = $geq(d0, p0 + 1, e0, second);
                                continue 'outer;
                            }
                            p2 = $geq(d2, p2, e2, doc);
                            if p2 >= e2 {
                                return;
                            }
                            let third = *d2.add(p2);
                            if third > doc {
                                p0 = $geq(d0, p0 + 1, e0, third);
                                continue 'outer;
                            }
                            docs_out.push(doc);
                            offs_out.push(p0 as u8);
                            offs_out.push(p1 as u8);
                            offs_out.push(p2 as u8);
                            p0 += 1;
                        }
                    }
                }

                $(#[$feat])?
                #[allow(clippy::too_many_arguments)]
                unsafe fn $name_n(
                    wins: &[WindowList],
                    freq_cannot_beat: &[bool; FREQ_LUT_BUCKETS],
                    cursors: &mut Vec<usize>,
                    docs_out: &mut Vec<u32>,
                    offs_out: &mut Vec<u8>,
                ) {
                    cursors.clear();
                    cursors.extend(wins.iter().map(|win| win.pos));
                    let (d0, mut p0, e0) = (wins[0].docs, wins[0].pos, wins[0].end);
                    let f0 = wins[0].freqs;
                    unsafe {
                        'outer: while p0 < e0 {
                            let doc = *d0.add(p0);
                            let freq = (*f0.add(p0) as usize).min(FREQ_LUT_BUCKETS - 1);
                            if freq_cannot_beat[freq] {
                                p0 += 1;
                                continue 'outer;
                            }
                            for j in 1..wins.len() {
                                let win = wins.get_unchecked(j);
                                let pos = $geq(win.docs, *cursors.get_unchecked(j), win.end, doc);
                                *cursors.get_unchecked_mut(j) = pos;
                                if pos >= win.end {
                                    return;
                                }
                                let clause_doc = *win.docs.add(pos);
                                if clause_doc > doc {
                                    p0 = $geq(d0, p0 + 1, e0, clause_doc);
                                    continue 'outer;
                                }
                            }
                            docs_out.push(doc);
                            offs_out.push(p0 as u8);
                            for j in 1..wins.len() {
                                offs_out.push(*cursors.get_unchecked(j) as u8);
                            }
                            p0 += 1;
                        }
                    }
                }

                $(#[$feat])?
                unsafe fn $docs_name2(
                    wins: &[WindowList],
                    docs_out: &mut Vec<u32>,
                    offs_out: &mut Vec<u8>,
                ) {
                    let (d0, mut p0, e0) = (wins[0].docs, wins[0].pos, wins[0].end);
                    let (d1, mut p1, e1) = (wins[1].docs, wins[1].pos, wins[1].end);
                    unsafe {
                        while p0 < e0 {
                            let doc = *d0.add(p0);
                            p1 = $geq(d1, p1, e1, doc);
                            if p1 >= e1 {
                                return;
                            }
                            let second = *d1.add(p1);
                            if second > doc {
                                p0 = $geq(d0, p0 + 1, e0, second);
                                continue;
                            }
                            docs_out.push(doc);
                            offs_out.push(p0 as u8);
                            offs_out.push(p1 as u8);
                            p0 += 1;
                        }
                    }
                }

                $(#[$feat])?
                unsafe fn $docs_name3(
                    wins: &[WindowList],
                    docs_out: &mut Vec<u32>,
                    offs_out: &mut Vec<u8>,
                ) {
                    let (d0, mut p0, e0) = (wins[0].docs, wins[0].pos, wins[0].end);
                    let (d1, mut p1, e1) = (wins[1].docs, wins[1].pos, wins[1].end);
                    let (d2, mut p2, e2) = (wins[2].docs, wins[2].pos, wins[2].end);
                    unsafe {
                        'outer: while p0 < e0 {
                            let doc = *d0.add(p0);
                            p1 = $geq(d1, p1, e1, doc);
                            if p1 >= e1 {
                                return;
                            }
                            let second = *d1.add(p1);
                            if second > doc {
                                p0 = $geq(d0, p0 + 1, e0, second);
                                continue 'outer;
                            }
                            p2 = $geq(d2, p2, e2, doc);
                            if p2 >= e2 {
                                return;
                            }
                            let third = *d2.add(p2);
                            if third > doc {
                                p0 = $geq(d0, p0 + 1, e0, third);
                                continue 'outer;
                            }
                            docs_out.push(doc);
                            offs_out.push(p0 as u8);
                            offs_out.push(p1 as u8);
                            offs_out.push(p2 as u8);
                            p0 += 1;
                        }
                    }
                }
            };
        }
        merge_kernels!(
            merge_window_2,
            merge_window_3,
            merge_window_n,
            merge_window_docs_2,
            merge_window_docs_3,
            find_next_geq_scalar
        );
        #[cfg(target_arch = "x86_64")]
        merge_kernels!(
            merge_window_2_avx2,
            merge_window_3_avx2,
            merge_window_n_avx2,
            merge_window_docs_2_avx2,
            merge_window_docs_3_avx2,
            find_next_geq_avx2,
            #[target_feature(enable = "avx2")]
        );

        #[inline]
        fn merge_window_1(wins: &[WindowList], docs_out: &mut Vec<u32>, offs_out: &mut Vec<u8>) {
            let win = &wins[0];
            for pos in win.pos..win.end {
                docs_out.push(unsafe { *win.docs.add(pos) });
                offs_out.push(pos as u8);
            }
        }

        #[inline]
        fn merge_window_docs_n(
            wins: &[WindowList],
            cursors: &mut Vec<usize>,
            docs_out: &mut Vec<u32>,
            offs_out: &mut Vec<u8>,
        ) {
            cursors.clear();
            cursors.extend(wins.iter().map(|win| win.pos));
            'outer: while cursors[0] < wins[0].end {
                let doc = unsafe { *wins[0].docs.add(cursors[0]) };
                for j in 1..wins.len() {
                    let win = &wins[j];
                    let pos = unsafe { find_next_geq(win.docs, cursors[j], win.end, doc) };
                    cursors[j] = pos;
                    if pos >= win.end {
                        return;
                    }
                    let clause_doc = unsafe { *win.docs.add(pos) };
                    if clause_doc > doc {
                        cursors[0] = unsafe {
                            find_next_geq(wins[0].docs, cursors[0] + 1, wins[0].end, clause_doc)
                        };
                        continue 'outer;
                    }
                }
                docs_out.push(doc);
                for &pos in cursors.iter() {
                    offs_out.push(pos as u8);
                }
                cursors[0] += 1;
            }
        }

        #[inline]
        fn merge_window_docs_pairwise(
            wins: &[WindowList],
            scratch: &mut Vec<u32>,
            docs_out: &mut Vec<u32>,
            offs_out: &mut Vec<u8>,
        ) {
            // Every slice belongs to one decoded block and has at most 256
            // entries. Shrink candidates clause by clause, so later clauses
            // only intersect documents that survived the earlier clauses.
            let first = &wins[0];
            // SAFETY: window construction keeps these decoded blocks alive
            // and unchanged until both intersection and scoring complete.
            docs_out.extend_from_slice(unsafe {
                std::slice::from_raw_parts(first.docs.add(first.pos), first.end - first.pos)
            });
            for win in &wins[1..] {
                let slice =
                    unsafe { std::slice::from_raw_parts(win.docs.add(win.pos), win.end - win.pos) };
                let first_doc = slice[0];
                let last_doc = slice[slice.len() - 1];
                // Sorted distinct IDs with this span contain every integer
                // in the interval. This exact certificate also handles a
                // dense follower covering a sparse set of candidates.
                if u64::from(last_doc) - u64::from(first_doc) + 1 == slice.len() as u64 {
                    if docs_out[0] >= first_doc && docs_out[docs_out.len() - 1] <= last_doc {
                        continue;
                    }
                    let lo = docs_out.partition_point(|&doc| doc < first_doc);
                    let hi = docs_out.partition_point(|&doc| doc <= last_doc);
                    if lo != 0 {
                        docs_out.copy_within(lo..hi, 0);
                    }
                    docs_out.truncate(hi - lo);
                } else {
                    intersection::intersect(docs_out, slice, scratch);
                    std::mem::swap(docs_out, scratch);
                }
                if docs_out.is_empty() {
                    return;
                }
            }

            // Recover posting offsets only for the final intersection. No
            // frequencies or intermediate per-clause offset tables are needed.
            offs_out.resize(docs_out.len() * wins.len(), 0);
            for (clause, win) in wins.iter().enumerate() {
                let first_doc = unsafe { *win.docs.add(win.pos) };
                let last_doc = unsafe { *win.docs.add(win.end - 1) };
                let is_dense =
                    u64::from(last_doc) - u64::from(first_doc) + 1 == (win.end - win.pos) as u64;
                let mut pos = win.pos;
                for (candidate, &doc) in docs_out.iter().enumerate() {
                    pos = if is_dense {
                        win.pos + (doc - first_doc) as usize
                    } else {
                        unsafe { find_next_geq(win.docs, pos, win.end, doc) }
                    };
                    debug_assert!(
                        pos < win.end,
                        "intersection must be present in every clause"
                    );
                    offs_out[candidate * wins.len() + clause] = pos as u8;
                }
            }
        }

        let mut candidates = TopKCollector::new(limit, std::cmp::min(limit, BLOCK_SIZE * 10));
        let mut num_comparisons: usize = 0;
        let mut wins: Vec<WindowList> = Vec::with_capacity(num_lists);
        // Per-window candidate batch. The merge kernel only records matches;
        // scoring then runs in two passes so the doc-length gather issues
        // independent loads (their cache misses overlap) instead of
        // serializing behind per-candidate branching. A window spans at most
        // one block per clause, so a batch holds at most one block's worth.
        let mut batch_docs: Vec<u32> = Vec::with_capacity(MAX_POSTING_BLOCK_SIZE);
        let mut batch_offs: Vec<u8> = Vec::with_capacity(MAX_POSTING_BLOCK_SIZE * num_lists);
        let mut batch_lens: Vec<u32> = Vec::with_capacity(MAX_POSTING_BLOCK_SIZE);
        let mut batch_norms: Vec<u8> = Vec::with_capacity(MAX_POSTING_BLOCK_SIZE);
        let mut cursor_scratch: Vec<usize> = Vec::with_capacity(num_lists);
        let mut intersection_scratch: Vec<u32> = Vec::new();
        // The frequency-bucket prune decisions only depend on the floor and
        // the followers' block maxes, which rarely change between windows.
        let mut freq_cannot_beat = [false; FREQ_LUT_BUCKETS];
        // Keyed separately from the score limit: the per-candidate prune can
        // refresh the limit in the middle of a window without touching the
        // table, so an unchanged limit key does not mean the table is current.
        let mut freq_cannot_beat_key: Option<(u32, u64)> = None;
        // Per clause: the block that was sliced last and the end of that
        // slice. Windows are contiguous, so the next slice of the same block
        // starts where the previous one ended instead of at a binary search.
        let mut slice_prev: SmallVec<[(usize, usize); 8]> =
            std::iter::repeat_n((usize::MAX, 0), num_lists).collect();
        // Norm cache: one byte-norm load plus a cached addend replaces the
        // per-clause BM25 denominator recompute in pass B.
        let mut norm_k = None;
        let mut norm_k_initialized = false;

        // Per-window prune LUT for the merge kernels: an upper bound of the
        // first (rarest) clause's score by clamped frequency. Lead docs whose
        // bound plus the remaining clauses' block maxes cannot beat the
        // threshold are skipped before any follower advances — the same
        // score-first ordering Lucene's conjunction scorer uses, with the
        // doc-length dropped from the bound so no doc-length load is needed.
        // The last bucket holds the frequency-independent sup, so clamping
        // stays a valid upper bound. Skips only avoid work; every emitted
        // candidate still goes through the exact per-candidate prune below.
        const FREQ_LUT_BUCKETS: usize = 64;
        let mut freq_bound_lut: Option<[f32; FREQ_LUT_BUCKETS]> = None;
        // Exclusive limit on the first clause's partial score for the current
        // window: scores at or below it cannot beat the floor. Recomputed only
        // when the floor or the followers' block maxes change.
        let mut first_score_limit: Option<f32> = None;
        let mut first_score_limit_key: Option<(u32, u64)> = None;

        // The conjunction can only start at the max of the clauses' first docs.
        let mut target: u64 = 0;
        for posting in &self.lead {
            match posting.current_doc_id() {
                Some(doc) => target = target.max(doc),
                None => return Ok(vec![]),
            }
        }

        'window: loop {
            #[cfg(test)]
            {
                self.and_window_stats.windows_sliced += 1;
            }
            self.raise_to_shared_floor(params.wand_factor);
            let window_started_with_floor = self.threshold > 0.0;
            if window_started_with_floor {
                if num_lists >= 2 && freq_bound_lut.is_none() {
                    let mut lut = [f32::INFINITY; FREQ_LUT_BUCKETS];
                    for (freq, slot) in lut.iter_mut().enumerate().take(FREQ_LUT_BUCKETS - 1) {
                        *slot = self.lead[0].score(&self.scorer, freq as u32, 0);
                    }
                    lut[FREQ_LUT_BUCKETS - 1] =
                        self.lead[0].frequency_clamp_upper_bound(&self.scorer);
                    freq_bound_lut = Some(lut);
                }
                let advanced = self.and_advance_target(target);
                if advanced == TERMINATED_DOC_ID {
                    break;
                }
                target = advanced;
            }
            debug_assert!(target <= u32::MAX as u64);
            let target32 = target as u32;

            // Position every clause's block cursor at the block that can hold
            // `target`, and end the window at the nearest next-block boundary.
            let mut win_end = TERMINATED_DOC_ID;
            for j in 0..num_lists {
                let (block_idx, block_up_to) = {
                    let posting = &self.lead[j];
                    let PostingList::Compressed(ref list) = posting.list else {
                        unreachable!("bulk AND requires compressed postings");
                    };
                    let block_idx = posting.block_idx_for_doc(list, posting.block_idx, target32);
                    let block_up_to = if block_idx + 1 < list.blocks.len() {
                        u64::from(list.block_least_doc_id(block_idx + 1)).saturating_sub(1)
                    } else {
                        TERMINATED_DOC_ID
                    };
                    (block_idx, block_up_to.max(target))
                };
                self.lead[j].block_idx = block_idx;
                win_end = win_end.min(block_up_to);
            }
            let win_end32 = u32::try_from(win_end).unwrap_or(u32::MAX);

            // Decompress each clause's block and slice it to [target, win_end].
            wins.clear();
            let mut skip_window = false;
            let mut exhausted = false;
            // When a clause has no document in the window, the conjunction
            // cannot match before that clause's next document, so the next
            // window starts there instead of at the next block boundary. A
            // sparse lead then drives the traversal and dense followers skip
            // whole blocks by metadata instead of being sliced block by block.
            let mut jump_to: u64 = 0;
            for (j, posting) in self.lead.iter().enumerate() {
                let PostingList::Compressed(ref list) = posting.list else {
                    unreachable!("bulk AND requires compressed postings");
                };
                let block_idx = posting.block_idx;
                let state = unsafe { &mut *posting.ensure_compressed_doc_ids_ptr(list, block_idx) };
                let doc_ids = state.doc_ids.as_slice();
                let len = doc_ids.len();
                // Everything before the previous slice's end of this block is
                // below the current target, so resume from there.
                let (prev_block, prev_hi) = slice_prev[j];
                let start = if prev_block == block_idx {
                    prev_hi.min(len)
                } else {
                    0
                };
                let lo = if start < len && doc_ids[start] >= target32 {
                    start
                } else {
                    start + doc_ids[start..].partition_point(|&doc| doc < target32)
                };
                let hi = if win_end32 == u32::MAX || len == 0 || doc_ids[len - 1] <= win_end32 {
                    len
                } else {
                    lo + doc_ids[lo..].partition_point(|&doc| doc <= win_end32)
                };
                slice_prev[j] = (block_idx, hi);
                if lo == hi {
                    // No docs of this clause in the window: the whole window
                    // has no conjunction match. If this was the clause's last
                    // block and it is fully behind the target, the clause is
                    // exhausted and the conjunction is done.
                    if lo < state.doc_ids.len() {
                        jump_to = u64::from(state.doc_ids[lo]);
                    } else if block_idx + 1 < list.blocks.len() {
                        jump_to = u64::from(list.block_least_doc_id(block_idx + 1));
                    } else {
                        exhausted = true;
                    }
                    skip_window = true;
                    break;
                }
                wins.push(WindowList {
                    docs: state.doc_ids.as_ptr(),
                    freqs: std::ptr::null(),
                    pos: lo,
                    end: hi,
                    block_start: block_idx << list.block_shift(),
                });
            }
            if exhausted {
                break 'window;
            }
            if !skip_window {
                // Constant within the window (block-anchored); mirrors
                // `and_candidate_cannot_beat_threshold`'s remaining-clause
                // bound of first-clause-exact + rest-block-max.
                let mut others_block_max = if window_started_with_floor {
                    Some(
                        self.lead[1..]
                            .iter()
                            .map(|posting| f64::from(posting.block_max_score(&self.scorer)))
                            .sum::<f64>(),
                    )
                } else {
                    None
                };

                batch_docs.clear();
                batch_offs.clear();
                // Precompute the conservative decision once per frequency
                // bucket so the scalar and SIMD merge kernels stay
                // floating-point-free.
                if window_started_with_floor && num_lists >= 2 {
                    let posting = &self.lead[0];
                    let PostingList::Compressed(ref list) = posting.list else {
                        unreachable!("bulk AND requires compressed postings");
                    };
                    let state = unsafe {
                        &mut *posting.ensure_compressed_block_ptr(list, posting.block_idx)
                    };
                    wins[0].freqs = state.freqs.as_ptr();
                    let others_block_max =
                        others_block_max.expect("positive floor should initialize bounds");
                    let key = (self.threshold.to_bits(), others_block_max.to_bits());
                    if first_score_limit_key != Some(key) {
                        first_score_limit = exclusive_partial_score_limit(
                            others_block_max,
                            self.threshold,
                            score_sum_upper_bound_factor(num_lists),
                        );
                        first_score_limit_key = Some(key);
                    }
                    if freq_cannot_beat_key != Some(key) {
                        let freq_bound_lut = freq_bound_lut
                            .as_ref()
                            .expect("positive threshold should initialize the frequency bound LUT");
                        freq_cannot_beat = match first_score_limit {
                            Some(limit) => {
                                std::array::from_fn(|frequency| freq_bound_lut[frequency] <= limit)
                            }
                            None => [false; FREQ_LUT_BUCKETS],
                        };
                        freq_cannot_beat_key = Some(key);
                    }
                } else if freq_cannot_beat_key.is_some() {
                    freq_cannot_beat = [false; FREQ_LUT_BUCKETS];
                    freq_cannot_beat_key = None;
                }
                #[cfg(test)]
                if freq_cannot_beat.iter().any(|&pruned| pruned) {
                    self.and_window_stats.windows_frequency_pruned += 1;
                }
                #[cfg(target_arch = "x86_64")]
                let use_avx2 = *HAS_AVX2;
                #[cfg(not(target_arch = "x86_64"))]
                let use_avx2 = false;
                match (num_lists, use_avx2, window_started_with_floor) {
                    (1, _, _) => merge_window_1(&wins, &mut batch_docs, &mut batch_offs),
                    #[cfg(target_arch = "x86_64")]
                    (2, true, false) => unsafe {
                        merge_window_docs_2_avx2(&wins, &mut batch_docs, &mut batch_offs)
                    },
                    #[cfg(target_arch = "x86_64")]
                    (3, true, false) => unsafe {
                        merge_window_docs_3_avx2(&wins, &mut batch_docs, &mut batch_offs)
                    },
                    (2, _, false) => unsafe {
                        merge_window_docs_2(&wins, &mut batch_docs, &mut batch_offs)
                    },
                    (3, _, false) => unsafe {
                        merge_window_docs_3(&wins, &mut batch_docs, &mut batch_offs)
                    },
                    (_, _, false) if use_pairwise_intersection => {
                        #[cfg(test)]
                        {
                            self.and_window_stats.pairwise_intersections += 1;
                        }
                        merge_window_docs_pairwise(
                            &wins,
                            &mut intersection_scratch,
                            &mut batch_docs,
                            &mut batch_offs,
                        )
                    }
                    (_, _, false) => merge_window_docs_n(
                        &wins,
                        &mut cursor_scratch,
                        &mut batch_docs,
                        &mut batch_offs,
                    ),
                    #[cfg(target_arch = "x86_64")]
                    (2, true, true) => unsafe {
                        merge_window_2_avx2(
                            &wins,
                            &freq_cannot_beat,
                            &mut batch_docs,
                            &mut batch_offs,
                        )
                    },
                    #[cfg(target_arch = "x86_64")]
                    (3, true, true) => unsafe {
                        merge_window_3_avx2(
                            &wins,
                            &freq_cannot_beat,
                            &mut batch_docs,
                            &mut batch_offs,
                        )
                    },
                    (2, _, true) => unsafe {
                        merge_window_2(&wins, &freq_cannot_beat, &mut batch_docs, &mut batch_offs)
                    },
                    (3, _, true) => unsafe {
                        merge_window_3(&wins, &freq_cannot_beat, &mut batch_docs, &mut batch_offs)
                    },
                    #[cfg(target_arch = "x86_64")]
                    (_, true, true) => unsafe {
                        merge_window_n_avx2(
                            &wins,
                            &freq_cannot_beat,
                            &mut cursor_scratch,
                            &mut batch_docs,
                            &mut batch_offs,
                        )
                    },
                    (_, _, true) => unsafe {
                        merge_window_n(
                            &wins,
                            &freq_cannot_beat,
                            &mut cursor_scratch,
                            &mut batch_docs,
                            &mut batch_offs,
                        )
                    },
                }

                if !batch_docs.is_empty() {
                    for (win, posting) in wins.iter_mut().zip(self.lead.iter()) {
                        let PostingList::Compressed(ref list) = posting.list else {
                            unreachable!("bulk AND requires compressed postings");
                        };
                        let state = unsafe {
                            &mut *posting.ensure_compressed_block_ptr(list, posting.block_idx)
                        };
                        win.freqs = state.freqs.as_ptr();
                    }
                    if !norm_k_initialized {
                        norm_k = self.norm_k_cache();
                        norm_k_initialized = true;
                    }
                }

                // Pass A: gather doc norms/lengths for the whole batch up
                // front so the loads issue back-to-back and their cache
                // misses overlap. With the norm cache, only the byte code is
                // gathered; the exact length decodes from a tiny LUT.
                batch_lens.clear();
                batch_norms.clear();
                let norm_k_ref = norm_k
                    .as_ref()
                    .map(|(norms, cache)| (*norms, cache.as_ref()));
                match norm_k_ref {
                    Some((norms, _)) => {
                        for &doc in batch_docs.iter() {
                            batch_norms.push(norms[doc as usize]);
                        }
                    }
                    None => match self.documents.scoring_norms() {
                        Some(norms) => {
                            for &doc in batch_docs.iter() {
                                batch_lens.push(dequantize_doc_length(norms[doc as usize]));
                            }
                        }
                        None => {
                            for &doc in batch_docs.iter() {
                                batch_lens.push(self.documents.scoring_num_tokens(doc));
                            }
                        }
                    },
                }

                // Pass B: prune / verify / score / insert, in doc order, with
                // exactly the classic loop's semantics.
                for (index, &doc) in batch_docs.iter().enumerate() {
                    let (norm_addend, doc_length) = match norm_k_ref {
                        Some((_, cache)) => {
                            let code = batch_norms[index];
                            (Some(cache[code as usize]), dequantize_doc_length(code))
                        }
                        None => (None, batch_lens[index]),
                    };
                    let offs = &batch_offs[index * num_lists..(index + 1) * num_lists];
                    if self.threshold > 0.0 && num_lists >= 2 && others_block_max.is_none() {
                        others_block_max = Some(
                            self.lead[1..]
                                .iter()
                                .map(|posting| f64::from(posting.block_max_score(&self.scorer)))
                                .sum::<f64>(),
                        );
                    }
                    if self.threshold > 0.0
                        && num_lists >= 2
                        && let Some(others_block_max) = others_block_max
                    {
                        let first_freq = unsafe { *wins[0].freqs.add(offs[0] as usize) };
                        let first_score = match norm_addend {
                            Some(addend) => {
                                self.lead[0].query_weight
                                    * bm25_doc_weight_with_norm(first_freq, addend)
                            }
                            None => self.lead[0].score(&self.scorer, first_freq, doc_length),
                        };
                        let key = (self.threshold.to_bits(), others_block_max.to_bits());
                        if first_score_limit_key != Some(key) {
                            first_score_limit = exclusive_partial_score_limit(
                                others_block_max,
                                self.threshold,
                                score_sum_upper_bound_factor(num_lists),
                            );
                            first_score_limit_key = Some(key);
                        }
                        if first_score_limit.is_some_and(|limit| first_score <= limit) {
                            continue;
                        }
                    }
                    #[cfg(test)]
                    {
                        self.and_window_stats.candidates_returned += 1;
                    }
                    num_comparisons += 1;

                    let Some(document_key) = self.documents.document_key_for_doc_id(doc) else {
                        continue;
                    };

                    // Only front-load the BM25 total when a floor exists: the
                    // skip below can only fire then, and scoring before
                    // position confirmation would otherwise make every
                    // position-failing candidate pay for a score it never uses.
                    let early_score = (self.threshold > 0.0).then(|| {
                        window_bm25_score(
                            &self.lead,
                            &wins,
                            offs,
                            &score_order,
                            &self.scorer,
                            norm_addend,
                            doc_length,
                        )
                    });
                    if let Some(slop) = phrase_slop {
                        // Lance phrase score is term-frequency BM25, so this
                        // total is exact. Skip position decode when it cannot
                        // enter the exclusive top-k heap.
                        if early_score
                            .is_some_and(|score| self.exclusive_score_cannot_beat_floor(score))
                        {
                            continue;
                        }
                        // Park every clause's iterator on this doc so
                        // `position_cursor` reads the right posting entry. The
                        // window block is already decompressed; position blocks
                        // decode lazily and are cached per block.
                        for ((win, posting), &off) in
                            wins.iter().zip(self.lead.iter_mut()).zip(offs.iter())
                        {
                            posting.index = win.block_start + off as usize;
                            posting.current_doc = Some(DocInfo::Raw(RawDocInfo {
                                doc_id: doc,
                                frequency: unsafe { *win.freqs.add(off as usize) },
                            }));
                        }
                        let matched = if slop == 0 {
                            self.check_exact_positions_bulk()?
                        } else {
                            self.check_positions(slop as i32)?
                        };
                        if !matched {
                            continue;
                        }
                    }
                    let score = early_score.unwrap_or_else(|| {
                        window_bm25_score(
                            &self.lead,
                            &wins,
                            offs,
                            &score_order,
                            &self.scorer,
                            norm_addend,
                            doc_length,
                        )
                    });

                    if candidates.insert(
                        ScoredDoc::new(document_key, score),
                        doc_length,
                        u64::from(doc),
                        wins.iter().zip(self.lead.iter()).zip(offs.iter()).map(
                            |((win, posting), &off)| {
                                (posting.term_index(), unsafe {
                                    *win.freqs.add(off as usize)
                                })
                            },
                        ),
                    )? && let Some(kth) = candidates.kth_score_if_full()
                    {
                        self.update_threshold(kth, params.wand_factor);
                    }
                }
            }

            if win_end == TERMINATED_DOC_ID {
                break;
            }
            target = (win_end + 1).max(jump_to);
        }

        metrics.record_comparisons(num_comparisons);

        candidates.into_candidates(|key| self.documents.candidate_from_key(key))
    }
}
