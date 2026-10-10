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
        (start..end).map(move |block_idx| V3BlockMeta {
            term_idx: self.term_idx,
            block_idx,
            block_row: self.block_start + block_idx,
        })
    }
}

/// Partition the document space into disjoint windows. Every term contributes
/// at most one block to each window. Refined sparse blocks contribute only at
/// their actual document ids, rather than inflating the bound across gaps.
pub(super) fn search_windows(terms: &[V3TermPlan], operator: Operator) -> Vec<V3Window> {
    let mut events = Vec::new();
    for (term_idx, term) in terms.iter().enumerate() {
        for (block_idx, block) in term.blocks.iter().enumerate() {
            if let Some(points) = term.refined_blocks.get(&block_idx) {
                for &(doc, score) in points {
                    events.push((u64::from(doc), term_idx, Some(score)));
                    events.push((u64::from(doc) + 1, term_idx, None));
                }
            } else {
                events.push((
                    u64::from(block.first_doc_id),
                    term_idx,
                    Some(block.block_max_score),
                ));
                events.push((u64::from(block.last_doc_id) + 1, term_idx, None));
            }
        }
    }
    // Sweep each term's changes once. Ends precede starts at the same id, so
    // adjacent blocks and adjacent refined documents preserve their bounds.
    events.sort_unstable_by_key(|(doc, term, bound)| (*doc, *term, bound.is_some()));
    let mut bounds = vec![None; terms.len()];
    let mut windows = Vec::new();
    let mut offset = 0;
    while offset < events.len() {
        let first = events[offset].0;
        while offset < events.len() && events[offset].0 == first {
            let (_, term, bound) = events[offset];
            bounds[term] = bound;
            offset += 1;
        }
        let Some(&(next, _, _)) = events.get(offset) else {
            break;
        };
        let mut upper_bound = 0.0;
        let mut matching_terms = 0;
        // Sum in query-term order, matching candidate scoring and avoiding
        // cumulative rounding error from adding/subtracting event deltas.
        for bound in bounds.iter().flatten() {
            upper_bound += bound;
            matching_terms += 1;
        }
        if matching_terms == 0 || (operator == Operator::And && matching_terms != terms.len()) {
            continue;
        }
        windows.push(V3Window {
            first_doc_id: first as u32,
            last_doc_id: (next - 1) as u32,
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
pub(super) fn refinement_blocks(
    terms: &[V3TermPlan],
    windows: impl Iterator<Item = V3Window>,
) -> Vec<V3BlockMeta> {
    const MAX_REFINEMENT_BLOCKS: usize = 128;
    if terms.len() < 2 {
        return Vec::new();
    }
    let mut windows = windows.collect::<Vec<_>>();
    windows.sort_unstable_by_key(|window| window.first_doc_id);
    let mut active_ranges = Vec::<(u32, u32)>::new();
    for window in windows {
        if let Some((_, last)) = active_ranges.last_mut()
            && u64::from(window.first_doc_id) <= u64::from(*last) + 1
        {
            *last = (*last).max(window.last_doc_id);
        } else {
            active_ranges.push((window.first_doc_id, window.last_doc_id));
        }
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
            let active_idx = active_ranges.partition_point(|(_, last)| *last < block.first_doc_id);
            if span <= count as u64 * 8
                || term.refined_blocks.contains_key(&block_idx)
                || active_ranges
                    .get(active_idx)
                    .is_none_or(|(first, _)| *first > block.last_doc_id)
                || !seen.insert(block_row)
            {
                continue;
            }
            selected.push(V3BlockMeta {
                term_idx: term.term_idx,
                block_idx,
                block_row,
            });
            if selected.len() == MAX_REFINEMENT_BLOCKS {
                return selected;
            }
        }
    }
    selected
}
