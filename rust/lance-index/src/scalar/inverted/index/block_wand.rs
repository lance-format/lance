// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::{Operator, V3BlockMeta, V3TermPlan, V3Window};

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
}

/// Partition the document space into disjoint windows. Every term contributes
/// at most one block to each window, so its bound is tight and a document is
/// scored exactly once, with all matching terms present.
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
    boundaries.sort_unstable();
    boundaries.dedup();
    let mut windows = Vec::with_capacity(boundaries.len());
    for interval in boundaries.windows(2) {
        let first = interval[0] as u32;
        let last = (interval[1] - 1) as u32;
        let mut upper_bound = 0.0;
        let mut matching_terms = 0;
        for term in terms {
            if let Some(block) = term.overlapping_blocks(first, last).next() {
                upper_bound += block.block_max_score;
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
    windows
}
