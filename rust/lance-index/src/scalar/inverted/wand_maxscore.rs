// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Inner-window work for MAXSCORE: block skip, two-pointer optional freqs,
//! and SoA completion for one essential plus at most two optionals.
//!
//! Lucene's `scoreInnerWindowSingleEssentialClause`. Two-essential compact
//! union is intentionally not ported.

use smallvec::SmallVec;

use lance_core::Result;

use super::super::builder::ScoredDoc;
use super::super::scorer::{Scorer, bm25_doc_weight_with_norm};
use super::{
    CompetitiveFloorMode, DocInfo, MaxScoreClause, PostingIterator, PostingList, RawDocInfo,
    TopKCollector, Wand, WandDocuments, score_sum_cannot_compete,
};

#[inline]
pub(super) fn exclusive_cannot_compete(
    partial: f32,
    remaining_upper_bound: f64,
    floor: f32,
    upper_bound_factor: f64,
) -> bool {
    floor > 0.0
        && score_sum_cannot_compete(
            partial,
            remaining_upper_bound,
            floor,
            upper_bound_factor,
            CompetitiveFloorMode::Exclusive,
        )
}

fn bm25_tf_from_caches(
    query_weight: f32,
    freq: u32,
    doc: u32,
    quantized: Option<(&[u8], &[f32; 256])>,
) -> Option<f32> {
    let (norms, cache) = quantized?;
    Some(query_weight * bm25_doc_weight_with_norm(freq, cache[norms[doc as usize] as usize]))
}

fn bulk_bm25_tf(
    query_weight: f32,
    hits: &[(u64, u32)],
    quantized: Option<(&[u8], &[f32; 256])>,
    fallback: impl Fn(u64, u32) -> f32,
    out: &mut Vec<f32>,
) {
    out.clear();
    out.reserve(hits.len());
    if let Some((norms, cache)) = quantized {
        for &(doc, freq) in hits {
            out.push(
                query_weight * bm25_doc_weight_with_norm(freq, cache[norms[doc as usize] as usize]),
            );
        }
        return;
    }
    for &(doc, freq) in hits {
        out.push(fallback(doc, freq));
    }
}

/// Lucene `DocAndScoreAccBuffer`: SoA vectors reused across essential blocks.
pub(super) struct MaxScoreSoaScratch {
    docs: Vec<u64>,
    ess_scores: Vec<f32>,
    ess_freqs: Vec<u32>,
    raw_scores: Vec<f32>,
    opt_scores: Vec<f32>,
    opt_freqs: Vec<u32>,
    opt_s: [Vec<f32>; 2],
    opt_f: [Vec<u32>; 2],
    partial: Vec<f32>,
}

impl MaxScoreSoaScratch {
    pub(super) fn new() -> Self {
        Self {
            docs: Vec::with_capacity(128),
            ess_scores: Vec::with_capacity(128),
            ess_freqs: Vec::with_capacity(128),
            raw_scores: Vec::with_capacity(128),
            opt_scores: Vec::with_capacity(128),
            opt_freqs: Vec::with_capacity(128),
            opt_s: [Vec::with_capacity(128), Vec::with_capacity(128)],
            opt_f: [Vec::with_capacity(128), Vec::with_capacity(128)],
            partial: Vec::with_capacity(128),
        }
    }
}

impl PostingIterator {
    /// Lucene `ScorerUtil.applyOptionalClause`: fill `out_freqs[i]` with this
    /// clause's frequency at sorted `docs[i]`, or 0 on a miss.
    pub(super) fn merge_optional_freqs(&mut self, docs: &[u64], out_freqs: &mut [u32]) {
        debug_assert_eq!(docs.len(), out_freqs.len());
        out_freqs.fill(0);
        if docs.is_empty() {
            return;
        }
        if self.doc().is_some_and(|cur| cur.doc_id() < docs[0]) {
            self.next(docs[0]);
        }
        if matches!(self.list, PostingList::Plain(_)) {
            for (i, &doc) in docs.iter().enumerate() {
                if self.doc().is_some_and(|cur| cur.doc_id() < doc) {
                    self.next(doc);
                }
                if let Some(cur) = self.doc()
                    && cur.doc_id() == doc
                {
                    out_freqs[i] = cur.frequency();
                }
            }
            return;
        }
        let shift = match &self.list {
            PostingList::Compressed(list) => list.block_shift(),
            PostingList::Plain(_) => return,
        };
        let length = match &self.list {
            PostingList::Compressed(list) => list.length as usize,
            PostingList::Plain(_) => return,
        };
        let mut i = 0usize;
        while i < docs.len() {
            let Some(cur) = self.current_doc else {
                return;
            };
            let cur_id = cur.doc_id();
            while i < docs.len() && docs[i] < cur_id {
                i += 1;
            }
            if i >= docs.len() {
                return;
            }
            if docs[i] > cur_id {
                self.next(docs[i]);
                continue;
            }
            let block_idx = self.index >> shift;
            let mask = match &self.list {
                PostingList::Compressed(list) => list.block_mask(),
                PostingList::Plain(_) => return,
            };
            let (stay, end_of_block) = match &self.list {
                PostingList::Compressed(list) => {
                    let compressed = unsafe { &*self.ensure_compressed_block_ptr(list, block_idx) };
                    let mut j = self.index & mask;
                    let n = compressed.doc_ids.len();
                    while i < docs.len() && j < n {
                        let block_doc = u64::from(compressed.doc_ids[j]);
                        if block_doc < docs[i] {
                            j += 1;
                        } else if block_doc == docs[i] {
                            out_freqs[i] = compressed.freqs[j];
                            i += 1;
                            j += 1;
                        } else {
                            i += 1;
                        }
                    }
                    let stay = (j < n).then(|| (compressed.doc_ids[j], compressed.freqs[j]));
                    (stay, j)
                }
                PostingList::Plain(_) => return,
            };
            if let Some((doc_id, frequency)) = stay {
                self.index = (block_idx << shift) + end_of_block;
                self.block_idx = block_idx;
                self.current_doc = Some(DocInfo::Raw(RawDocInfo { doc_id, frequency }));
            } else {
                let next_start = (block_idx + 1) << shift;
                if next_start >= length {
                    self.index = length;
                    self.block_idx = self.index >> shift;
                    self.current_doc = None;
                } else {
                    self.index = next_start;
                    self.block_idx = block_idx + 1;
                    let next_doc = match &self.list {
                        PostingList::Compressed(list) => {
                            let next =
                                unsafe { &*self.ensure_compressed_block_ptr(list, block_idx + 1) };
                            (next.doc_ids[0], next.freqs[0])
                        }
                        PostingList::Plain(_) => return,
                    };
                    self.current_doc = Some(DocInfo::Raw(RawDocInfo {
                        doc_id: next_doc.0,
                        frequency: next_doc.1,
                    }));
                }
            }
        }
    }

    /// Copy one decompressed posting block (or a 128-doc Plain slice) with
    /// `doc_id <= window_max` into `out`. Leaves the cursor on the first doc
    /// beyond that chunk.
    pub(super) fn take_docs_one_block_upto(&mut self, window_max: u64, out: &mut Vec<(u64, u32)>) {
        out.clear();
        match self.list {
            PostingList::Compressed(ref list) => {
                let Some(cur) = self.current_doc else {
                    return;
                };
                if cur.doc_id() > window_max {
                    return;
                }
                let shift = list.block_shift();
                let mask = list.block_mask();
                let block_idx = self.index >> shift;
                let block_offset = self.index & mask;
                let compressed = unsafe { &mut *self.ensure_compressed_block_ptr(list, block_idx) };
                for offset in block_offset..compressed.doc_ids.len() {
                    let doc_id = compressed.doc_ids[offset];
                    if u64::from(doc_id) > window_max {
                        self.index = (block_idx << shift) + offset;
                        self.block_idx = block_idx;
                        self.current_doc = Some(DocInfo::Raw(RawDocInfo {
                            doc_id,
                            frequency: compressed.freqs[offset],
                        }));
                        return;
                    }
                    out.push((u64::from(doc_id), compressed.freqs[offset]));
                }
                let next_start = (block_idx + 1) << shift;
                if next_start >= list.length as usize {
                    self.index = list.length as usize;
                    self.block_idx = self.index >> shift;
                    self.current_doc = None;
                    return;
                }
                self.index = next_start;
                self.block_idx = block_idx + 1;
                let compressed =
                    unsafe { &mut *self.ensure_compressed_block_ptr(list, block_idx + 1) };
                self.current_doc = Some(DocInfo::Raw(RawDocInfo {
                    doc_id: compressed.doc_ids[0],
                    frequency: compressed.freqs[0],
                }));
            }
            PostingList::Plain(_) => {
                while let Some(cur) = self.doc() {
                    let doc = cur.doc_id();
                    if doc > window_max {
                        break;
                    }
                    out.push((doc, cur.frequency()));
                    self.next(doc + 1);
                    if out.len() >= 128 {
                        break;
                    }
                }
            }
        }
    }

    /// Advance to the first doc of the next posting block, or exhaust.
    pub(super) fn skip_current_block(&mut self) -> bool {
        match self.next_block_first_doc() {
            Some(doc) => {
                self.next(doc);
                self.current_doc.is_some()
            }
            None => {
                if let PostingList::Compressed(ref list) = self.list {
                    self.index = list.length as usize;
                    self.block_idx = self.index >> list.block_shift();
                }
                self.current_doc = None;
                false
            }
        }
    }
}

impl<'a, S: Scorer, D: WandDocuments> Wand<'a, S, D> {
    /// Skip dead essential blocks and complete surviving docs with at most
    /// two optional clauses via SoA buffers.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn score_single_essential_upto(
        &mut self,
        clauses: &mut [MaxScoreClause],
        first_essential: usize,
        ess_idx: usize,
        upto: u64,
        essential_query_rank: usize,
        essential_term: u32,
        essential_weight: f32,
        essential_window_bound: f32,
        num_query_terms: usize,
        total_non_essential_bound: f64,
        total_sum_upper_bound_factor: f64,
        norm_k_ref: Option<(&[u8], &[f32; 256])>,
        essential_chunk: &mut Vec<(u64, u32)>,
        scratch: &mut MaxScoreSoaScratch,
        candidates: &mut TopKCollector,
        num_comparisons: &mut usize,
        wand_factor: f32,
    ) -> Result<()> {
        loop {
            // Lucene `nextPostings(upTo)`: do not advance past this call's
            // exclusive end. After `take_docs` the cursor already sits on the
            // first doc of the next block, which may belong to a later window
            // whose optional remainder is larger.
            if clauses[ess_idx]
                .posting
                .doc()
                .is_none_or(|doc| doc.doc_id() > upto)
            {
                break;
            }
            if self.threshold > 0.0 {
                let block_max = clauses[ess_idx].posting.block_max_score(&self.scorer);
                if exclusive_cannot_compete(
                    block_max,
                    total_non_essential_bound,
                    self.threshold,
                    total_sum_upper_bound_factor,
                ) {
                    #[cfg(test)]
                    {
                        self.maxscore_essential_blocks_skipped += 1;
                    }
                    let more = clauses[ess_idx].posting.skip_current_block();
                    if !more
                        || clauses[ess_idx]
                            .posting
                            .doc()
                            .is_none_or(|doc| doc.doc_id() > upto)
                    {
                        break;
                    }
                    continue;
                }
            }
            clauses[ess_idx]
                .posting
                .take_docs_one_block_upto(upto, essential_chunk);
            if essential_chunk.is_empty() {
                break;
            }
            let (non_essential, _) = clauses.split_at_mut(first_essential);
            self.complete_maxscore_single_essential(
                essential_chunk,
                non_essential,
                essential_query_rank,
                essential_term,
                essential_weight,
                essential_window_bound,
                num_query_terms,
                total_non_essential_bound,
                total_sum_upper_bound_factor,
                norm_k_ref,
                scratch,
                candidates,
                num_comparisons,
                wand_factor,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn complete_maxscore_single_essential(
        &mut self,
        hits: &[(u64, u32)],
        non_essential: &mut [MaxScoreClause],
        essential_query_rank: usize,
        essential_term: u32,
        essential_weight: f32,
        essential_window_bound: f32,
        num_query_terms: usize,
        total_non_essential_bound: f64,
        total_sum_upper_bound_factor: f64,
        norm_k_ref: Option<(&[u8], &[f32; 256])>,
        scratch: &mut MaxScoreSoaScratch,
        candidates: &mut TopKCollector,
        num_comparisons: &mut usize,
        wand_factor: f32,
    ) -> Result<()> {
        debug_assert!(non_essential.len() <= 2);
        scratch.docs.clear();
        scratch.ess_scores.clear();
        scratch.ess_freqs.clear();
        scratch.opt_scores.clear();
        scratch.opt_freqs.clear();
        bulk_bm25_tf(
            essential_weight,
            hits,
            norm_k_ref,
            |doc, freq| {
                essential_weight
                    * self
                        .scorer
                        .doc_weight(freq, self.documents.scoring_num_tokens(doc as u32))
            },
            &mut scratch.raw_scores,
        );
        for (hit, &score) in hits.iter().zip(scratch.raw_scores.iter()) {
            *num_comparisons += 1;
            if exclusive_cannot_compete(
                score,
                total_non_essential_bound,
                self.threshold,
                total_sum_upper_bound_factor,
            ) {
                continue;
            }
            scratch.docs.push(hit.0);
            scratch.ess_scores.push(score);
            scratch.ess_freqs.push(hit.1);
        }
        if scratch.docs.is_empty() {
            return Ok(());
        }

        let mut opt_term = 0u32;
        let mut opt_rank = 0usize;

        if non_essential.len() == 1 {
            let optional = &mut non_essential[0];
            opt_term = optional.posting.term_index();
            opt_rank = optional.query_rank;
            let require_best = essential_window_bound.is_finite()
                && essential_window_bound > 0.0
                && exclusive_cannot_compete(
                    essential_window_bound,
                    0.0,
                    self.threshold,
                    total_sum_upper_bound_factor,
                );
            let query_weight = optional.posting.query_weight;
            // With a single optional clause `total_non_essential_bound` is that
            // clause's own `prefix_bound`, so the entry filter above already
            // applied this predicate; re-running it here was a no-op pass.
            scratch.opt_scores.clear();
            scratch.opt_scores.resize(scratch.docs.len(), 0.0);
            scratch.opt_freqs.clear();
            scratch.opt_freqs.resize(scratch.docs.len(), 0);
            let probe = &mut optional.posting;
            probe.merge_optional_freqs(&scratch.docs, &mut scratch.opt_freqs);
            let mut w = 0usize;
            for r in 0..scratch.docs.len() {
                let freq = scratch.opt_freqs[r];
                if freq == 0 && require_best {
                    continue;
                }
                let contribution = if freq == 0 {
                    0.0
                } else {
                    bm25_tf_from_caches(query_weight, freq, scratch.docs[r] as u32, norm_k_ref)
                        .unwrap_or_else(|| {
                            probe.score(
                                &self.scorer,
                                freq,
                                self.documents.scoring_num_tokens(scratch.docs[r] as u32),
                            )
                        })
                };
                scratch.docs[w] = scratch.docs[r];
                scratch.ess_scores[w] = scratch.ess_scores[r];
                scratch.ess_freqs[w] = scratch.ess_freqs[r];
                scratch.opt_scores[w] = contribution;
                scratch.opt_freqs[w] = freq;
                w += 1;
            }
            scratch.docs.truncate(w);
            scratch.ess_scores.truncate(w);
            scratch.ess_freqs.truncate(w);
            scratch.opt_scores.truncate(w);
            scratch.opt_freqs.truncate(w);
        } else if non_essential.len() == 2 {
            let n_live = scratch.docs.len();
            for slot in 0..2 {
                scratch.opt_s[slot].clear();
                scratch.opt_s[slot].resize(n_live, 0.0);
                scratch.opt_f[slot].clear();
                scratch.opt_f[slot].resize(n_live, 0);
            }
            scratch.partial.clear();
            scratch.partial.extend_from_slice(&scratch.ess_scores);
            let opt_term = [
                non_essential[0].posting.term_index(),
                non_essential[1].posting.term_index(),
            ];
            let opt_rank = [non_essential[0].query_rank, non_essential[1].query_rank];
            let remaining_without_best = non_essential[0].prefix_bound;
            let require_best = essential_window_bound.is_finite()
                && essential_window_bound > 0.0
                && remaining_without_best.is_finite()
                && exclusive_cannot_compete(
                    essential_window_bound,
                    remaining_without_best,
                    self.threshold,
                    total_sum_upper_bound_factor,
                );
            let apply = [1usize, 0];
            for (step, &idx) in apply.iter().enumerate() {
                let required = require_best && step == 0;
                let query_weight = non_essential[idx].posting.query_weight;
                // Step 0 probes the second optional, whose `prefix_bound` is
                // `total_non_essential_bound`; the entry filter already applied
                // that predicate while `partial` still equalled the essential
                // score, so re-filtering here would be a no-op pass.
                if !required && step > 0 {
                    let prefix_bound = non_essential[idx].prefix_bound;
                    let mut w = 0usize;
                    for r in 0..scratch.docs.len() {
                        if exclusive_cannot_compete(
                            scratch.partial[r],
                            prefix_bound,
                            self.threshold,
                            total_sum_upper_bound_factor,
                        ) {
                            continue;
                        }
                        scratch.docs[w] = scratch.docs[r];
                        scratch.ess_scores[w] = scratch.ess_scores[r];
                        scratch.ess_freqs[w] = scratch.ess_freqs[r];
                        scratch.opt_s[0][w] = scratch.opt_s[0][r];
                        scratch.opt_s[1][w] = scratch.opt_s[1][r];
                        scratch.opt_f[0][w] = scratch.opt_f[0][r];
                        scratch.opt_f[1][w] = scratch.opt_f[1][r];
                        scratch.partial[w] = scratch.partial[r];
                        w += 1;
                    }
                    scratch.docs.truncate(w);
                    scratch.ess_scores.truncate(w);
                    scratch.ess_freqs.truncate(w);
                    scratch.partial.truncate(w);
                    scratch.opt_s[0].truncate(w);
                    scratch.opt_s[1].truncate(w);
                    scratch.opt_f[0].truncate(w);
                    scratch.opt_f[1].truncate(w);
                    if scratch.docs.is_empty() {
                        return Ok(());
                    }
                }
                let n = scratch.docs.len();
                scratch.opt_f[idx].clear();
                scratch.opt_f[idx].resize(n, 0);
                let probe = &mut non_essential[idx].posting;
                probe.merge_optional_freqs(&scratch.docs, &mut scratch.opt_f[idx]);
                let mut w = 0usize;
                for r in 0..n {
                    let freq = scratch.opt_f[idx][r];
                    if freq == 0 && required {
                        continue;
                    }
                    let contribution = if freq == 0 {
                        0.0
                    } else {
                        bm25_tf_from_caches(query_weight, freq, scratch.docs[r] as u32, norm_k_ref)
                            .unwrap_or_else(|| {
                                probe.score(
                                    &self.scorer,
                                    freq,
                                    self.documents.scoring_num_tokens(scratch.docs[r] as u32),
                                )
                            })
                    };
                    scratch.docs[w] = scratch.docs[r];
                    scratch.ess_scores[w] = scratch.ess_scores[r];
                    scratch.ess_freqs[w] = scratch.ess_freqs[r];
                    scratch.opt_s[0][w] = scratch.opt_s[0][r];
                    scratch.opt_s[1][w] = scratch.opt_s[1][r];
                    scratch.opt_f[0][w] = scratch.opt_f[0][r];
                    scratch.opt_f[1][w] = scratch.opt_f[1][r];
                    scratch.opt_s[idx][w] = contribution;
                    scratch.opt_f[idx][w] = freq;
                    scratch.partial[w] = scratch.partial[r] + contribution;
                    w += 1;
                }
                scratch.docs.truncate(w);
                scratch.ess_scores.truncate(w);
                scratch.ess_freqs.truncate(w);
                scratch.partial.truncate(w);
                scratch.opt_s[0].truncate(w);
                scratch.opt_s[1].truncate(w);
                scratch.opt_f[0].truncate(w);
                scratch.opt_f[1].truncate(w);
                if scratch.docs.is_empty() {
                    return Ok(());
                }
            }
            for i in 0..scratch.docs.len() {
                let mut by_rank = SmallVec::<[f32; 8]>::from_elem(0.0, num_query_terms);
                by_rank[essential_query_rank] = scratch.ess_scores[i];
                for (o, &rank) in opt_rank.iter().enumerate() {
                    if scratch.opt_f[o][i] > 0 {
                        by_rank[rank] = scratch.opt_s[o][i];
                    }
                }
                let canonical = by_rank
                    .into_iter()
                    .fold(0.0_f32, |sum, contribution| sum + contribution);
                if canonical <= self.threshold || candidates.rejects_score(canonical) {
                    continue;
                }
                let Some(document_key) = self
                    .documents
                    .document_key_for_doc_id(scratch.docs[i] as u32)
                else {
                    continue;
                };
                let doc_length = self.documents.scoring_num_tokens(scratch.docs[i] as u32);
                let mut freqs = SmallVec::<[(u32, u32); 3]>::new();
                freqs.push((essential_term, scratch.ess_freqs[i]));
                for (o, &term) in opt_term.iter().enumerate() {
                    if scratch.opt_f[o][i] > 0 {
                        freqs.push((term, scratch.opt_f[o][i]));
                    }
                }
                if candidates.insert(
                    ScoredDoc::new(document_key, canonical),
                    doc_length,
                    scratch.docs[i],
                    freqs.into_iter(),
                )? && let Some(kth) = candidates.kth_score_if_full()
                {
                    self.update_threshold(kth, wand_factor);
                }
            }
            return Ok(());
        }

        for i in 0..scratch.docs.len() {
            let opt_freq = if non_essential.len() == 1 {
                scratch.opt_freqs.get(i).copied().unwrap_or(0)
            } else {
                0
            };
            let canonical = if non_essential.is_empty() {
                scratch.ess_scores[i]
            } else {
                let mut by_rank = SmallVec::<[f32; 8]>::from_elem(0.0, num_query_terms);
                by_rank[essential_query_rank] = scratch.ess_scores[i];
                if opt_freq > 0 {
                    by_rank[opt_rank] = scratch.opt_scores[i];
                }
                by_rank
                    .into_iter()
                    .fold(0.0_f32, |sum, contribution| sum + contribution)
            };
            if canonical <= self.threshold || candidates.rejects_score(canonical) {
                continue;
            }
            let Some(document_key) = self
                .documents
                .document_key_for_doc_id(scratch.docs[i] as u32)
            else {
                continue;
            };
            let doc_length = self.documents.scoring_num_tokens(scratch.docs[i] as u32);
            let mut freqs = SmallVec::<[(u32, u32); 2]>::new();
            freqs.push((essential_term, scratch.ess_freqs[i]));
            if opt_freq > 0 {
                freqs.push((opt_term, opt_freq));
            }
            if candidates.insert(
                ScoredDoc::new(document_key, canonical),
                doc_length,
                scratch.docs[i],
                freqs.into_iter(),
            )? && let Some(kth) = candidates.kth_score_if_full()
            {
                self.update_threshold(kth, wand_factor);
            }
        }
        Ok(())
    }
}
