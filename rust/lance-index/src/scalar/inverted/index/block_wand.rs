// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BinaryHeap, HashSet};

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

impl PartialEq for V3Window {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for V3Window {}

impl PartialOrd for V3Window {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for V3Window {
    fn cmp(&self, other: &Self) -> Ordering {
        self.upper_bound
            .total_cmp(&other.upper_bound)
            .then_with(|| other.first_doc_id.cmp(&self.first_doc_id))
            .then_with(|| other.last_doc_id.cmp(&self.last_doc_id))
    }
}

/// Partition the document space into disjoint windows. Every term contributes
/// at most one block to each window. Refined sparse blocks contribute only at
/// their actual document ids, rather than inflating the bound across gaps.
pub(super) fn search_windows(
    terms: &[V3TermPlan],
    operator: Operator,
    threshold: f32,
    scored_windows: &BTreeMap<u32, u32>,
) -> BinaryHeap<V3Window> {
    let mut term_events = Vec::with_capacity(terms.len());
    let mut next_events = BinaryHeap::new();
    for (term_idx, term) in terms.iter().enumerate() {
        let mut events = Vec::new();
        for (block_idx, block) in term.blocks.iter().enumerate() {
            if let Some(points) = term.refined_blocks.get(&block_idx) {
                for &(doc, score) in points {
                    events.push((u64::from(doc), Some(score)));
                    events.push((u64::from(doc) + 1, None));
                }
            } else {
                events.push((u64::from(block.first_doc_id), Some(block.block_max_score)));
                events.push((u64::from(block.last_doc_id) + 1, None));
            }
        }
        if let Some(&(doc, _)) = events.first() {
            next_events.push(Reverse((doc, term_idx, 0usize)));
        }
        term_events.push(events);
    }
    // Each term is already ordered by document id, with an end preceding the
    // next start at adjacent ids. Merge these streams instead of sorting every
    // refined document again after each payload batch.
    let mut bounds = vec![None; terms.len()];
    let mut windows = Vec::new();
    let mut scored = scored_windows.iter().peekable();
    while let Some(&Reverse((first, _, _))) = next_events.peek() {
        while let Some(&Reverse((_, term, offset))) =
            next_events.peek().filter(|event| event.0.0 == first)
        {
            next_events.pop();
            bounds[term] = term_events[term][offset].1;
            if let Some(&(doc, _)) = term_events[term].get(offset + 1) {
                next_events.push(Reverse((doc, term, offset + 1)));
            }
        }
        let Some(&Reverse((next, _, _))) = next_events.peek() else {
            break;
        };
        while scored
            .peek()
            .is_some_and(|(_, last)| u64::from(**last) < first)
        {
            scored.next();
        }
        // Refinement adds boundaries but never crosses a previously scored
        // window. Both streams are doc ordered, so exclusion needs no tree lookup.
        if scored.peek().is_some_and(|(start, last)| {
            u64::from(**start) <= first && u64::from(**last) >= next - 1
        }) {
            continue;
        }
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
        if upper_bound <= threshold {
            continue;
        }
        windows.push(V3Window {
            first_doc_id: first as u32,
            last_doc_id: (next - 1) as u32,
            upper_bound,
        });
    }
    // Heap construction is linear; only competitive windows pay the cost of
    // ordered extraction. Most windows are discarded once top-k is established.
    BinaryHeap::from(windows)
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
