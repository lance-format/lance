// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Top-k result collection with reusable (term, frequency) slots.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use lance_core::{Error, Result};

use crate::scalar::inverted::builder::ScoredDoc;

use super::doc_source::DocCandidate;

/// Top-k heap entry: (scored doc, doc length, posting doc id, frequency slot).
/// The (term, freq) pairs live outside the heap so heap churn moves a compact
/// tuple instead of a `Vec`.
type TopKHeap = BinaryHeap<Reverse<(ScoredDoc, u32, u64, u32)>>;

/// Reusable (term, freq) storage for live heap candidates. Replacing the k-th
/// result reuses its slot and retained `Vec` capacity, so memory stays bounded
/// by the live top-k rather than every historical heap admission.
struct FrequencySlots {
    slots: Vec<Vec<(u32, u32)>>,
}

impl FrequencySlots {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
        }
    }

    fn push(&mut self, pairs: impl Iterator<Item = (u32, u32)>) -> Result<u32> {
        let slot = u32::try_from(self.slots.len()).map_err(|_| {
            Error::internal(format!(
                "FTS top-k frequency slot count {} exceeds u32::MAX",
                self.slots.len()
            ))
        })?;
        self.slots.push(pairs.collect());
        Ok(slot)
    }

    fn replace(&mut self, slot: u32, pairs: impl Iterator<Item = (u32, u32)>) -> Result<()> {
        let num_slots = self.slots.len();
        let slot = self.slots.get_mut(slot as usize).ok_or_else(|| {
            Error::internal(format!(
                "FTS top-k frequency slot {slot} is out of bounds for {num_slots} slots"
            ))
        })?;
        slot.clear();
        slot.extend(pairs);
        Ok(())
    }

    fn take(&mut self, slot: u32) -> Result<Vec<(u32, u32)>> {
        let num_slots = self.slots.len();
        self.slots
            .get_mut(slot as usize)
            .map(std::mem::take)
            .ok_or_else(|| {
                Error::internal(format!(
                    "FTS top-k frequency slot {slot} is out of bounds for {num_slots} slots"
                ))
            })
    }
}

/// Owns the top-k heap and the frequency slots referenced by its entries.
pub(super) struct TopKCollector {
    limit: usize,
    heap: TopKHeap,
    frequency_slots: FrequencySlots,
}

impl TopKCollector {
    pub(super) fn new(limit: usize, initial_capacity: usize) -> Self {
        let initial_capacity = initial_capacity.min(limit);
        Self {
            limit,
            heap: BinaryHeap::with_capacity(initial_capacity),
            frequency_slots: FrequencySlots::with_capacity(initial_capacity),
        }
    }

    /// Insert a competitive result. When the heap is full, its evicted entry's
    /// frequency slot is cleared and reused before the replacement is pushed.
    pub(super) fn insert(
        &mut self,
        doc: ScoredDoc,
        doc_length: u32,
        posting_doc_id: u64,
        pairs: impl Iterator<Item = (u32, u32)>,
    ) -> Result<bool> {
        if self.limit == 0 {
            return Ok(false);
        }
        if self.heap.len() > self.limit {
            return Err(Error::internal(format!(
                "FTS top-k heap length {} exceeds limit {}",
                self.heap.len(),
                self.limit
            )));
        }

        let frequency_slot = if self.heap.len() == self.limit {
            let Some(kth_score) = self.heap.peek().map(|entry| entry.0.0.score.0) else {
                return Err(Error::internal(
                    "FTS top-k heap is empty while its nonzero limit is reached",
                ));
            };
            // Preserve the existing collector semantics for non-finite custom
            // scorer output: only a strictly greater raw f32 replaces k-th.
            if doc.score.0.partial_cmp(&kth_score) != Some(std::cmp::Ordering::Greater) {
                return Ok(false);
            }
            let Some(Reverse((_, _, _, frequency_slot))) = self.heap.pop() else {
                return Err(Error::internal(
                    "FTS top-k heap entry disappeared during replacement",
                ));
            };
            self.frequency_slots.replace(frequency_slot, pairs)?;
            frequency_slot
        } else {
            self.frequency_slots.push(pairs)?
        };

        self.heap
            .push(Reverse((doc, doc_length, posting_doc_id, frequency_slot)));
        Ok(true)
    }

    pub(super) fn rejects_score(&self, score: f32) -> bool {
        if self.heap.len() < self.limit {
            return false;
        }
        let Some(kth_score) = self.heap.peek().map(|entry| entry.0.0.score.0) else {
            return false;
        };
        score.partial_cmp(&kth_score) != Some(std::cmp::Ordering::Greater)
    }

    pub(super) fn kth_score_if_full(&self) -> Option<f32> {
        if self.heap.len() == self.limit {
            self.heap.peek().map(|entry| entry.0.0.score.0)
        } else {
            None
        }
    }

    pub(super) fn into_candidates<C>(
        self,
        mut to_candidate: impl FnMut(u64) -> C,
    ) -> Result<Vec<DocCandidate<C>>> {
        let Self {
            heap,
            mut frequency_slots,
            ..
        } = self;
        heap.into_iter()
            .map(
                |Reverse((doc, doc_length, posting_doc_id, frequency_slot))| {
                    Ok(DocCandidate {
                        document: to_candidate(doc.row_id),
                        posting_doc_id,
                        freqs: frequency_slots.take(frequency_slot)?,
                        doc_length,
                    })
                },
            )
            .collect()
    }

    #[cfg(test)]
    pub(super) fn num_frequency_slots(&self) -> usize {
        self.frequency_slots.slots.len()
    }
}
