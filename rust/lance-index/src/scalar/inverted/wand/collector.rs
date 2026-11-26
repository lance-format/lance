// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Top-k result collection with reusable (term, frequency) slots.
//!
//! The FTS result order is `(score DESC, row_id ASC)`. The collector applies as
//! much of it as the WAND walk can see and retains what it cannot rank, so the
//! caller can settle the rest once row addresses are known.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use lance_core::{Error, Result};

use crate::scalar::inverted::builder::ScoredDoc;
use crate::vector::graph::OrderedFloat;

use super::doc_source::DocCandidate;

/// The maximum number of documents a deferred collector buffers beyond `limit`
/// while it holds the k-th-score tie band. Matches the compound collector's
/// bound (`compound::SCORE_FLOOR_RESOLUTION_BATCH_SIZE`) so both FTS paths cap
/// their unresolved working set the same way.
pub(in crate::scalar::inverted) const SCORE_FLOOR_BUFFER: usize = 128;

/// As much of the FTS result order `(score DESC, row_id ASC)` as the WAND walk
/// can see. Ordered "best is greatest".
///
/// `order_key` is the row address when the walk can resolve one and a constant
/// otherwise (see [`WandDocuments::tie_order_key`]), so this is a partial picture
/// of the result order: two candidates comparing equal are documents the walk
/// cannot rank, and the collector retains both rather than choosing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TopKRankKey {
    score: OrderedFloat,
    order_key: u64,
}

impl PartialOrd for TopKRankKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TopKRankKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .cmp(&other.score)
            .then_with(|| other.order_key.cmp(&self.order_key))
    }
}

/// The rank key plus the posting DocId, which makes the heap order total.
///
/// The DocId is not part of the result order: it only keeps the heap
/// deterministic and, once the retained band hits its cap, decides which
/// candidates of an unrankable group to keep. Documents are appended in
/// coordinate order, so for an index this writer produces the DocId ascends with
/// `doc_index` inside a row; a rebuilt or merged index preserves that relative
/// order too. It is a proxy, though, not a format guarantee, which is why it is
/// consulted only past the cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TopKKey {
    rank: TopKRankKey,
    posting_doc_id: u64,
}

impl PartialOrd for TopKKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TopKKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.rank
            .cmp(&other.rank)
            .then_with(|| other.posting_doc_id.cmp(&self.posting_doc_id))
    }
}

/// One live top-k candidate. The (term, freq) pairs live outside the heap in a
/// [`FrequencySlots`] slot so heap churn moves a compact record instead of a `Vec`.
#[derive(Debug, Clone)]
struct TopKEntry {
    doc: ScoredDoc,
    order_key: u64,
    doc_length: u32,
    posting_doc_id: u64,
    frequency_slot: u32,
}

impl TopKEntry {
    fn rank_key(&self) -> TopKRankKey {
        TopKRankKey {
            score: self.doc.score,
            order_key: self.order_key,
        }
    }

    fn key(&self) -> TopKKey {
        TopKKey {
            rank: self.rank_key(),
            posting_doc_id: self.posting_doc_id,
        }
    }
}

impl PartialEq for TopKEntry {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

impl Eq for TopKEntry {}

impl PartialOrd for TopKEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TopKEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key().cmp(&other.key())
    }
}

type TopKHeap = BinaryHeap<Reverse<TopKEntry>>;

/// Reusable (term, freq) storage for live heap candidates. Replacing the k-th
/// result reuses its slot and retained `Vec` capacity, so memory stays bounded
/// by the live top-k rather than every historical heap admission.
struct FrequencySlots {
    slots: Vec<Vec<(u32, u32)>>,
    /// Slots whose entry left the heap without a replacement, ready for reuse.
    free: Vec<u32>,
}

impl FrequencySlots {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free: Vec::new(),
        }
    }

    fn push(&mut self, pairs: impl Iterator<Item = (u32, u32)>) -> Result<u32> {
        if let Some(slot) = self.free.pop() {
            self.replace(slot, pairs)?;
            return Ok(slot);
        }
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

    /// Move a slot's pairs out and return the slot to the free list.
    fn release(&mut self, slot: u32) -> Result<Vec<(u32, u32)>> {
        let pairs = self.take(slot)?;
        self.free.push(slot);
        Ok(pairs)
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

/// A document tied at the k-th score that the heap has no room for. It owns its
/// frequencies because it never held a frequency slot.
struct TieCandidate {
    doc: ScoredDoc,
    doc_length: u32,
    posting_doc_id: u64,
    freqs: Vec<(u32, u32)>,
}

/// What the WAND walk knows about the final result order, which decides how it
/// treats a document tied at the k-th score.
///
/// Every walk visits candidates in ascending partition-local DocId order, so the
/// relation between DocId order and row-address order is what separates modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::scalar::inverted) enum TieHandling {
    /// Row addresses are resident, unique, and strictly ascend with DocId. DocId
    /// order is then result order among equal scores, so every candidate still to
    /// come loses a tie against the k-th entry: a k-th-score tie is rejected
    /// outright, the retained band stays empty, and the score floor can sit at the
    /// raw k-th score instead of one ULP below it.
    AscendingRowAddresses,
    /// Row addresses are resident but DocId order does not track them (a remapped
    /// partition), or several documents share one address (the elements of a list
    /// row). `order_key` ranks rows but not what the walk cannot see, so those
    /// candidates are retained. A full band cannot be recovered by rereading
    /// anything, so it degrades to the DocId proxy and reports
    /// [`TopKCollector::band_truncated`].
    ByRowAddress,
    /// No resident addresses, so `order_key` is one constant and every k-th-score
    /// tie is unrankable. Occasional ties are cheap, so they are retained like any
    /// other; once the band fills the corpus is tied at the k-th score rather than
    /// the top-k boundary, and the pass stops there and reports
    /// [`TopKCollector::score_floor_overflow`]. The caller then retries the
    /// partition with addresses, where one of the modes above applies.
    DeferredAddresses,
}

/// Owns the top-k heap and the frequency slots referenced by its entries.
pub(super) struct TopKCollector {
    limit: usize,
    heap: TopKHeap,
    frequency_slots: FrequencySlots,
    tie_handling: TieHandling,
    /// Cap on `limit + tie_band.len()`. Both modes need one: a deferred walk can
    /// tie the whole corpus at the k-th score, and nothing bounds how many elements
    /// a single list row holds.
    max_buffered: usize,
    /// Candidates the walk cannot rank against the current k-th entry and the heap
    /// has no room for. Cleared whenever the k-th key improves, so it holds just
    /// the current tie band and stays empty for a query without ties.
    tie_band: Vec<TieCandidate>,
    score_floor_overflow: bool,
    band_truncated: bool,
    /// Set when the collector gave up on the pass rather than merely dropping
    /// candidates, so a later improvement cannot make the flag obsolete.
    aborted: bool,
}

impl TopKCollector {
    pub(super) fn new(limit: usize, initial_capacity: usize, tie_handling: TieHandling) -> Self {
        let initial_capacity = initial_capacity.min(limit);
        Self {
            limit,
            heap: BinaryHeap::with_capacity(initial_capacity),
            frequency_slots: FrequencySlots::with_capacity(initial_capacity),
            tie_handling,
            max_buffered: limit.saturating_add(SCORE_FLOOR_BUFFER),
            tie_band: Vec::new(),
            score_floor_overflow: false,
            band_truncated: false,
            aborted: false,
        }
    }

    /// True when the walk gave up on part of its k-th-score tie band, either by
    /// stopping at an unrankable tie or by dropping one it could not buffer. The
    /// candidates it returned are then only the exact top-k up to that band, so
    /// the caller must retry the partition with resolved row addresses.
    pub(super) fn score_floor_overflow(&self) -> bool {
        self.score_floor_overflow
    }

    /// True once the walk must stop: [`Self::score_floor_overflow`] is set and
    /// nothing later in this pass can make the result exact. A walk that ignores
    /// this keeps scoring documents whose ties it cannot resolve.
    pub(super) fn aborted(&self) -> bool {
        self.aborted
    }

    /// True when an address-ordered walk filled its band and fell back to the DocId
    /// proxy. Rereading cannot recover it, so the caller reports it rather than
    /// retrying: elements of that one row past the cap may order by DocId instead
    /// of by `doc_index`.
    pub(super) fn band_truncated(&self) -> bool {
        self.band_truncated
    }

    /// Insert a competitive result.
    ///
    /// Returns whether the document entered the heap. When the heap is full, the
    /// evicted entry's frequency slot is cleared and reused before the replacement
    /// is pushed.
    pub(super) fn insert(
        &mut self,
        doc: ScoredDoc,
        order_key: u64,
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

        // A NaN custom-scorer score is never competitive. Dropping it before the
        // heap sees it keeps NaN out of the ordering key, where `total_cmp` would
        // rank it above every real score.
        if doc.score.0.is_nan() {
            return Ok(false);
        }

        if self.heap.len() < self.limit {
            let frequency_slot = self.frequency_slots.push(pairs)?;
            self.heap.push(Reverse(TopKEntry {
                doc,
                order_key,
                doc_length,
                posting_doc_id,
                frequency_slot,
            }));
            return Ok(true);
        }
        let candidate = TopKKey {
            rank: TopKRankKey {
                score: doc.score,
                order_key,
            },
            posting_doc_id,
        };
        let Some(Reverse(worst)) = self.heap.peek() else {
            return Err(Error::internal(
                "FTS top-k heap is empty while its nonzero limit is reached",
            ));
        };
        let worst_key = worst.key();

        // Equal rank keys mean the walk cannot rank these two documents. The heap
        // order still separates them through the DocId proxy, but that is not the
        // result order, so the loser is retained instead of dropped.
        if candidate.rank == worst_key.rank {
            // Unique ascending addresses make rank keys unique, so a walk that
            // visits documents in address order never gets here. Rejecting one
            // anyway keeps the retained band empty if a caller feeds the collector
            // candidates out of order.
            if self.tie_handling == TieHandling::AscendingRowAddresses {
                return Ok(false);
            }
            if self.band_is_full() {
                self.record_band_full();
            } else if candidate > worst_key {
                // Wins the proxy, so it takes the heap slot and the displaced entry
                // becomes the retained one.
                let displaced = self.replace_worst_retaining(
                    doc,
                    order_key,
                    doc_length,
                    posting_doc_id,
                    pairs,
                )?;
                self.tie_band.push(displaced);
                return Ok(true);
            } else {
                self.tie_band.push(TieCandidate {
                    doc,
                    doc_length,
                    posting_doc_id,
                    freqs: pairs.collect(),
                });
                return Ok(false);
            }
        }

        match candidate.cmp(&worst_key) {
            // Ranks below the k-th entry: never competitive.
            std::cmp::Ordering::Less | std::cmp::Ordering::Equal => Ok(false),
            // Ranks above the k-th entry: it always enters the heap.
            std::cmp::Ordering::Greater => {
                // Checked before the pop, which would otherwise make the buffer look
                // one candidate emptier than it is.
                let band_full = self.band_is_full();
                let Some(Reverse(displaced)) = self.heap.pop() else {
                    return Err(Error::internal(
                        "FTS top-k heap entry disappeared during replacement",
                    ));
                };
                // The k-th key after the replacement is the smaller of what the pop
                // exposed and the newcomer, both known without pushing first.
                let kth = self
                    .heap
                    .peek()
                    .map_or(candidate, |Reverse(entry)| entry.key().min(candidate));
                let retain_displaced = if kth.rank > worst_key.rank {
                    // The k-th rank rose past the band: none of its members can win,
                    // and neither can anything a full band dropped earlier, since it
                    // ranked no higher than the band. Clearing the flags keeps a stale
                    // overflow from forcing a needless address-ordered retry. A
                    // deliberate abort is not stale: the pass stopped early, so its
                    // candidates stay incomplete whatever arrived afterwards.
                    self.tie_band.clear();
                    if !self.aborted {
                        self.score_floor_overflow = false;
                        self.band_truncated = false;
                    }
                    false
                } else if band_full {
                    self.record_band_full();
                    false
                } else {
                    true
                };

                if retain_displaced {
                    // The displaced entry keeps its frequencies, so the newcomer
                    // takes another slot and the released one serves the next
                    // replacement.
                    let frequency_slot = self.frequency_slots.push(pairs)?;
                    self.heap.push(Reverse(TopKEntry {
                        doc,
                        order_key,
                        doc_length,
                        posting_doc_id,
                        frequency_slot,
                    }));
                    let freqs = self.frequency_slots.release(displaced.frequency_slot)?;
                    self.tie_band.push(TieCandidate {
                        doc: displaced.doc,
                        doc_length: displaced.doc_length,
                        posting_doc_id: displaced.posting_doc_id,
                        freqs,
                    });
                } else {
                    // Nothing keeps the displaced frequencies, so the slot and its
                    // retained capacity are reused in place. This is the path every
                    // improving candidate of a query without ties takes.
                    let frequency_slot = displaced.frequency_slot;
                    self.frequency_slots.replace(frequency_slot, pairs)?;
                    self.heap.push(Reverse(TopKEntry {
                        doc,
                        order_key,
                        doc_length,
                        posting_doc_id,
                        frequency_slot,
                    }));
                }
                Ok(true)
            }
        }
    }

    /// Pop the worst entry, push `doc` in its place with a fresh slot, and return
    /// the displaced entry owning its frequencies.
    fn replace_worst_retaining(
        &mut self,
        doc: ScoredDoc,
        order_key: u64,
        doc_length: u32,
        posting_doc_id: u64,
        pairs: impl Iterator<Item = (u32, u32)>,
    ) -> Result<TieCandidate> {
        let Some(Reverse(displaced)) = self.heap.pop() else {
            return Err(Error::internal(
                "FTS top-k heap entry disappeared during replacement",
            ));
        };
        let frequency_slot = self.frequency_slots.push(pairs)?;
        self.heap.push(Reverse(TopKEntry {
            doc,
            order_key,
            doc_length,
            posting_doc_id,
            frequency_slot,
        }));
        let freqs = self.frequency_slots.release(displaced.frequency_slot)?;
        Ok(TieCandidate {
            doc: displaced.doc,
            doc_length: displaced.doc_length,
            posting_doc_id: displaced.posting_doc_id,
            freqs,
        })
    }

    /// Record that the band could not take another candidate.
    ///
    /// A walk whose addresses are resident but unordered has nothing to reread, so
    /// it keeps ranking the heap on the DocId proxy: the documents the result order
    /// would pick still reach the final selection, but the rest of the unrankable
    /// group past the cap is dropped.
    fn record_band_full(&mut self) {
        match self.tie_handling {
            // Unreachable: an address-ascending walk keeps an empty band.
            TieHandling::AscendingRowAddresses => {}
            TieHandling::ByRowAddress => self.band_truncated = true,
            // A deferred walk can be retried with addresses, so stop walking rather
            // than keep scoring a corpus that ties at the k-th score throughout.
            TieHandling::DeferredAddresses => self.abort_pass(),
        }
    }

    /// Give up on this pass: the k-th-score tie band is full and this walk cannot
    /// rank its members, so the corpus ties at the k-th score throughout and the
    /// rest of the walk would only score documents it has to throw away.
    fn abort_pass(&mut self) {
        self.aborted = true;
        self.score_floor_overflow = true;
    }

    /// Documents held for the final selection: the heap plus the retained band.
    pub(super) fn buffered(&self) -> usize {
        self.heap.len() + self.tie_band.len()
    }

    /// Whether the retained band has reached its cap.
    fn band_is_full(&self) -> bool {
        self.buffered() >= self.max_buffered
    }

    /// Whether a document with `score` cannot enter the top-k, whatever its
    /// position in the result order.
    ///
    /// When the walk reaches documents in ascending row-address order a tie at the
    /// k-th score has already lost, so only a score strictly above the k-th
    /// competes. Otherwise a tie may still win on address, or has to be retained
    /// until addresses are resolved, and [`Self::insert`] decides it.
    pub(super) fn rejects_score(&self, score: f32) -> bool {
        if score.is_nan() {
            return true;
        }
        self.kth_score_if_full()
            .is_some_and(|kth| match self.tie_handling {
                TieHandling::AscendingRowAddresses => score <= kth,
                TieHandling::ByRowAddress | TieHandling::DeferredAddresses => score < kth,
            })
    }

    pub(super) fn kth_score_if_full(&self) -> Option<f32> {
        if self.heap.len() == self.limit {
            self.heap.peek().map(|entry| entry.0.doc.score.0)
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
            tie_band,
            ..
        } = self;
        let mut candidates = heap
            .into_iter()
            .map(|Reverse(entry)| {
                Ok(DocCandidate {
                    document: to_candidate(entry.doc.row_id),
                    posting_doc_id: entry.posting_doc_id,
                    freqs: frequency_slots.take(entry.frequency_slot)?,
                    doc_length: entry.doc_length,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        // The retained ties may beat a heap entry once addresses are resolved. The
        // caller re-selects by `(score, row_id)`, so returning more than `limit`
        // documents is safe.
        candidates.reserve(tie_band.len());
        for tie in tie_band {
            candidates.push(DocCandidate {
                document: to_candidate(tie.doc.row_id),
                posting_doc_id: tie.posting_doc_id,
                freqs: tie.freqs,
                doc_length: tie.doc_length,
            });
        }
        Ok(candidates)
    }

    #[cfg(test)]
    pub(super) fn tie_band_len(&self) -> usize {
        self.tie_band.len()
    }

    #[cfg(test)]
    pub(super) fn num_frequency_slots(&self) -> usize {
        self.frequency_slots.slots.len()
    }

    /// Smallest retained capacity across the live slots. A slot whose buffer was
    /// handed to a retained tie comes back at zero.
    #[cfg(test)]
    pub(super) fn min_frequency_slot_capacity(&self) -> usize {
        self.frequency_slots
            .slots
            .iter()
            .map(Vec::capacity)
            .min()
            .unwrap_or_default()
    }
}
