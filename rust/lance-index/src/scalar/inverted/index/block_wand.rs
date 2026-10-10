// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashSet;

use super::{BLOCK_SIZE, Operator, V3BlockMeta, V3TermPlan, V3Window};

impl V3TermPlan {
    pub(super) fn overlapping_blocks(
        &self,
        first: u32,
        last: u32,
    ) -> impl Iterator<Item = V3BlockMeta> + '_ {
        let start = self
            .blocks
            .partition_point(|block| block.last_doc_id < first);
        let end = self
            .blocks
            .partition_point(|block| block.first_doc_id <= last);
        self.blocks[start..end]
            .iter()
            .enumerate()
            .map(move |(offset, block)| V3BlockMeta {
                term_idx: self.term_idx,
                block_idx: start + offset,
                block_row: self.block_start + start + offset,
                block_max_score: block.block_max_score,
            })
    }

    fn window_bound(&self, first: u32, last: u32) -> Option<f32> {
        let block = self.overlapping_blocks(first, last).next()?;
        if let Some(points) = self.refined_blocks.get(&block.block_idx) {
            let start = points.partition_point(|(doc, _)| *doc < first);
            let end = points.partition_point(|(doc, _)| *doc <= last);
            points[start..end]
                .iter()
                .map(|(_, score)| *score)
                .reduce(f32::max)
        } else {
            Some(block.block_max_score)
        }
    }
}

/// Partition the document space into disjoint windows. Every term contributes
/// at most one block to each window. Refined sparse blocks contribute only at
/// their actual document ids, rather than inflating the bound across gaps.
pub(super) fn search_windows(terms: &[V3TermPlan], operator: Operator) -> Vec<V3Window> {
    let mut boundaries = terms
        .iter()
        .flat_map(|term| term.blocks.iter())
        .flat_map(|block| {
            [
                u64::from(block.first_doc_id),
                u64::from(block.last_doc_id) + 1,
            ]
        })
        .collect::<Vec<_>>();
    for term in terms {
        boundaries.extend(
            term.refined_blocks
                .values()
                .flatten()
                .flat_map(|(doc, _)| [u64::from(*doc), u64::from(*doc) + 1]),
        );
    }
    boundaries.sort_unstable();
    boundaries.dedup();
    let mut windows = Vec::with_capacity(boundaries.len());
    for interval in boundaries.windows(2) {
        let first = interval[0] as u32;
        let last = (interval[1] - 1) as u32;
        let mut upper_bound = 0.0;
        let mut matching_terms = 0;
        for term in terms {
            if let Some(bound) = term.window_bound(first, last) {
                upper_bound += bound;
                matching_terms += 1;
            }
        }
        if matching_terms == 0 || (operator == Operator::And && matching_terms != terms.len()) {
            continue;
        }
        windows.push(V3Window {
            first_doc_id: first,
            last_doc_id: last,
            upper_bound,
        });
    }
    windows.sort_unstable_by(|left, right| {
        right
            .upper_bound
            .total_cmp(&left.upper_bound)
            .then_with(|| left.first_doc_id.cmp(&right.first_doc_id))
            .then_with(|| left.last_doc_id.cmp(&right.last_doc_id))
    });
    windows
}

/// Resolve the widest sparse bounds in one bounded I/O batch. Rare terms are
/// preferred because their exact document ids can exclude large regions of
/// frequent postings. Dense blocks already have useful interval bounds.
pub(super) fn refinement_blocks(terms: &[V3TermPlan]) -> Vec<V3BlockMeta> {
    const MAX_REFINEMENT_BLOCKS: usize = 128;
    if terms.len() < 2 {
        return Vec::new();
    }
    let mut terms = terms.iter().collect::<Vec<_>>();
    terms.sort_unstable_by_key(|term| term.posting_len);
    let mut selected = Vec::new();
    let mut seen = HashSet::new();
    for term in terms {
        let mut blocks = term.blocks.iter().enumerate().collect::<Vec<_>>();
        blocks.sort_unstable_by(|(left_idx, left), (right_idx, right)| {
            right
                .block_max_score
                .total_cmp(&left.block_max_score)
                .then_with(|| left_idx.cmp(right_idx))
        });
        for (block_idx, block) in blocks {
            let count = (term.posting_len as usize - block_idx * BLOCK_SIZE).min(BLOCK_SIZE);
            let span = u64::from(block.last_doc_id) - u64::from(block.first_doc_id) + 1;
            let block_row = term.block_start + block_idx;
            if span <= count as u64 * 8 || !seen.insert(block_row) {
                continue;
            }
            selected.push(V3BlockMeta {
                term_idx: term.term_idx,
                block_idx,
                block_row,
                block_max_score: block.block_max_score,
            });
            if selected.len() == MAX_REFINEMENT_BLOCKS {
                return selected;
            }
        }
    }
    selected
}
