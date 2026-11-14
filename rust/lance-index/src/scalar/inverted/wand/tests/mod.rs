// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use arrow::buffer::ScalarBuffer;
use itertools::Itertools;
use lance_core::Result;
use lance_select::RowAddrMask;
use rstest::rstest;
use smallvec::SmallVec;

use crate::metrics::{LocalMetricsCollector, NoOpMetricsCollector};
use crate::scalar::inverted::builder::{BLOCK_SIZE, PositionRecorder, ScoredDoc};
use crate::scalar::inverted::documents::DocVisibility;
use crate::scalar::inverted::encoding::{
    MAX_POSTING_BLOCK_SIZE, compress_posting_list,
    compress_posting_list_with_tail_codec_and_block_size, encode_position_stream_block_into,
};
use crate::scalar::inverted::impact::IMPACT_LEVEL1_BLOCKS;
use crate::scalar::inverted::index::PositionStreamCodec;
use crate::scalar::inverted::query::{FtsSearchParams, Operator};
use crate::scalar::inverted::scorer::{
    BM25_DOC_WEIGHT_UPPER_BOUND, IndexBM25Scorer, MemBM25Scorer, Scorer, bm25_doc_weight_with_norm,
};
use crate::scalar::inverted::{
    CompressedPositionStorage, CompressedPostingList, DocInfo, DocSet, PlainPostingList,
    PostingList, PostingListBuilder, SharedPositionStream,
};

use super::super::documents::resident_row_address_projection_for_test;
use super::super::impact::build_impact_skip_data;
use super::bounds::{
    ScoreContribution, conservative_score_sum, exclusive_partial_score_limit, next_down_f32,
    next_up_f32, score_contributions_in_query_order,
};
use super::collector::TopKCollector;
use super::maxscore::{MAXSCORE_INNER_WINDOW, MaxScoreClause, MaxScoreSoaScratch};
use super::phrase::counted_anchor_steps;
use super::posting::{find_next_geq, find_next_geq_in_block, take_position_cursor_calls};
use super::*;
use super::{AllModernDocuments, ModernVisibility};

mod bounds;
mod bulk_conjunction;
mod collector;
mod conjunction;
mod cursor;
mod disjunction;
mod maxscore;
mod phrase;
mod posting;
mod search;

struct CostOnlyDocuments {
    total_docs: usize,
    visible_cost_upper_bound: usize,
}

impl WandDocuments for CostOnlyDocuments {
    type Candidate = u64;

    fn len(&self) -> usize {
        self.total_docs
    }

    fn visible_cost_upper_bound(&self) -> usize {
        self.visible_cost_upper_bound
    }

    fn scoring_norms(&self) -> Option<&[u8]> {
        None
    }

    fn scoring_num_tokens(&self, _doc_id: u32) -> u32 {
        1
    }

    fn doc_length(&self, _doc: &DocInfo) -> u32 {
        1
    }

    fn document_key(&self, doc: &DocInfo) -> Option<u64> {
        Some(doc.doc_id())
    }

    fn document_key_for_doc_id(&self, doc_id: u32) -> Option<u64> {
        Some(u64::from(doc_id))
    }

    fn candidate_from_key(&self, key: u64) -> Self::Candidate {
        key
    }

    fn flat_documents(&self) -> Option<FlatDocuments<'_>> {
        None
    }

    fn flat_doc_length(&self, _doc_id: u64, _document_key: u64, _compressed: bool) -> u32 {
        1
    }
}

struct UnitScorer;

impl Scorer for UnitScorer {
    fn query_weight(&self, _token: &str) -> f32 {
        1.0
    }

    fn doc_weight(&self, freq: u32, _doc_tokens: u32) -> f32 {
        freq as f32
    }
}

struct InverseDocLengthScorer;

impl Scorer for InverseDocLengthScorer {
    fn query_weight(&self, _token: &str) -> f32 {
        1.0
    }

    fn doc_weight(&self, freq: u32, doc_tokens: u32) -> f32 {
        freq as f32 / doc_tokens as f32
    }
}

// Inverse-doc-length scorer that counts scored documents, so a test can
// assert that block-max pruning skipped blocks.
struct CountingScorer {
    scored: Arc<AtomicUsize>,
}

impl Scorer for CountingScorer {
    fn query_weight(&self, _token: &str) -> f32 {
        1.0
    }

    fn doc_weight(&self, freq: u32, doc_tokens: u32) -> f32 {
        self.scored.fetch_add(1, Ordering::Relaxed);
        freq as f32 / doc_tokens as f32
    }
}

struct CountingBm25ShapeScorer {
    scored: Arc<AtomicUsize>,
}

impl Scorer for CountingBm25ShapeScorer {
    fn query_weight(&self, _token: &str) -> f32 {
        1.0
    }

    fn doc_weight(&self, freq: u32, _doc_tokens: u32) -> f32 {
        self.scored.fetch_add(1, Ordering::Relaxed);
        bm25_doc_weight_with_norm(freq, 1.2)
    }

    fn doc_weight_upper_bound(&self) -> Option<f32> {
        Some(BM25_DOC_WEIGHT_UPPER_BOUND)
    }

    fn doc_norm(&self, _doc_tokens: u32) -> Option<f32> {
        Some(1.2)
    }
}

struct VariedBm25ShapeScorer;

impl Scorer for VariedBm25ShapeScorer {
    fn query_weight(&self, _token: &str) -> f32 {
        1.0
    }

    fn doc_weight(&self, freq: u32, doc_tokens: u32) -> f32 {
        bm25_doc_weight_with_norm(freq, self.doc_norm(doc_tokens).unwrap())
    }

    fn doc_weight_upper_bound(&self) -> Option<f32> {
        Some(BM25_DOC_WEIGHT_UPPER_BOUND)
    }

    fn doc_norm(&self, doc_tokens: u32) -> Option<f32> {
        Some(0.3 + doc_tokens as f32 * 0.1)
    }
}

fn generate_posting_list(
    doc_ids: Vec<u32>,
    max_score: f32,
    block_max_scores: Option<Vec<f32>>,
    is_compressed: bool,
) -> PostingList {
    let freqs = vec![1; doc_ids.len()];
    generate_posting_list_with_freqs(doc_ids, freqs, max_score, block_max_scores, is_compressed)
}

fn generate_posting_list_with_freqs(
    doc_ids: Vec<u32>,
    freqs: Vec<u32>,
    max_score: f32,
    block_max_scores: Option<Vec<f32>>,
    is_compressed: bool,
) -> PostingList {
    assert_eq!(doc_ids.len(), freqs.len());
    let block_max_scores = block_max_scores.unwrap_or_else(|| vec![max_score; doc_ids.len()]);
    if is_compressed {
        let blocks = compress_posting_list(
            doc_ids.len(),
            doc_ids.iter(),
            freqs.iter(),
            block_max_scores.into_iter(),
        )
        .unwrap();
        PostingList::Compressed(CompressedPostingList::new(
            blocks,
            max_score,
            doc_ids.len() as u32,
            crate::scalar::inverted::PostingTailCodec::VarintDelta,
            crate::scalar::inverted::LEGACY_BLOCK_SIZE,
            None,
            None,
        ))
    } else {
        PostingList::Plain(PlainPostingList::new(
            ScalarBuffer::from_iter(doc_ids.iter().map(|id| *id as u64)),
            ScalarBuffer::from_iter(freqs.iter().map(|freq| *freq as f32)),
            Some(max_score),
            None,
        ))
    }
}

fn generate_impact_posting_list_with_freqs(
    doc_ids: Vec<u32>,
    freqs: Vec<u32>,
    doc_lengths: Vec<u32>,
) -> PostingList {
    generate_impact_posting_list_with_freqs_and_block_size(
        doc_ids,
        freqs,
        doc_lengths,
        crate::scalar::inverted::LEGACY_BLOCK_SIZE,
    )
}

fn generate_impact_posting_list_with_freqs_and_block_size(
    doc_ids: Vec<u32>,
    freqs: Vec<u32>,
    doc_lengths: Vec<u32>,
    block_size: usize,
) -> PostingList {
    assert_eq!(doc_ids.len(), freqs.len());
    assert_eq!(doc_ids.len(), doc_lengths.len());
    let block_max_scores = vec![0.0; doc_ids.len().div_ceil(block_size)];
    let blocks = compress_posting_list_with_tail_codec_and_block_size(
        doc_ids.len(),
        doc_ids.iter(),
        freqs.iter(),
        block_max_scores.into_iter(),
        crate::scalar::inverted::PostingTailCodec::VarintDelta,
        block_size,
    )
    .unwrap();
    let impact_blocks = doc_ids
        .chunks(block_size)
        .zip(freqs.chunks(block_size))
        .zip(doc_lengths.chunks(block_size))
        .map(|((doc_ids, freqs), doc_lengths)| {
            doc_ids
                .iter()
                .copied()
                .zip(freqs.iter().copied())
                .zip(doc_lengths.iter().copied())
                .map(|((doc_id, freq), doc_length)| (doc_id, freq, doc_length))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let impacts = build_impact_skip_data(impact_blocks.as_slice()).unwrap();
    PostingList::Compressed(CompressedPostingList::new(
        blocks,
        0.0,
        doc_ids.len() as u32,
        crate::scalar::inverted::PostingTailCodec::VarintDelta,
        block_size,
        None,
        Some(impacts),
    ))
}

fn generate_posting_list_with_positions(
    doc_ids: Vec<u32>,
    positions_by_doc: Vec<Vec<u32>>,
    max_score: f32,
    is_compressed: bool,
) -> PostingList {
    let freqs = positions_by_doc
        .iter()
        .map(|positions| positions.len() as u32)
        .collect::<Vec<_>>();
    if is_compressed {
        let mut builder = PostingListBuilder::new(true);
        for (doc_id, positions) in doc_ids.iter().copied().zip(positions_by_doc) {
            builder.add(doc_id, PositionRecorder::Position(positions.into()));
        }
        let batch = builder
            .to_batch(vec![max_score; doc_ids.len().div_ceil(BLOCK_SIZE)])
            .unwrap();
        PostingList::from_batch(&batch, Some(max_score), Some(doc_ids.len() as u32)).unwrap()
    } else {
        let mut position_builder =
            arrow::array::ListBuilder::new(arrow::array::Int32Builder::new());
        for positions in positions_by_doc {
            for position in positions {
                position_builder.values().append_value(position as i32);
            }
            position_builder.append(true);
        }
        PostingList::Plain(PlainPostingList::new(
            ScalarBuffer::from_iter(doc_ids.iter().map(|id| *id as u64)),
            ScalarBuffer::from_iter(freqs.iter().map(|freq| *freq as f32)),
            Some(max_score),
            Some(position_builder.finish()),
        ))
    }
}

fn sorted_candidate_row_ids(candidates: Vec<DocCandidate<u64>>) -> Vec<u64> {
    let mut row_ids = candidates
        .into_iter()
        .map(|candidate| candidate.document)
        .collect::<Vec<_>>();
    row_ids.sort_unstable();
    row_ids
}
