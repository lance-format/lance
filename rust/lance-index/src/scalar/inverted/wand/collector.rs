// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Top-k result collection with reusable (term, frequency) slots.
//!
//! The FTS result order is `(score DESC, row_id ASC)`. The collector ranks as
//! much of it as the WAND walk can see and retains the rest, so the caller can
//! rank it once row addresses are known.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use lance_core::{Error, Result};

use crate::scalar::inverted::builder::ScoredDoc;
use crate::scalar::inverted::documents::{DocId, ElementCoordinateKey, ResidentDocCoordinates};
use crate::vector::graph::OrderedFloat;

use super::doc_source::DocCandidate;

/// Slack beyond `limit` that a collector may buffer while it holds the
/// k-th-score tie band. A deferred walk that fills the band stops walking, and
/// the caller re-walks the partition with row addresses attached.
pub(in crate::scalar::inverted) const SCORE_FLOOR_BUFFER: usize = 128;

/// The part of the result order below the score that a walk can rank: the row
/// address, then the element coordinate that separates one row's elements.
/// `coordinate` is `None` where the walk has no element coordinates; a
/// row-granularity partition needs none, its documents own distinct addresses.
/// Ranks deeper than the packed key are compared against the partition's
/// resident columns, see `TopKEntry::cmp_deeper_ranks`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::scalar::inverted) struct TieOrder {
    pub(in crate::scalar::inverted) address: u64,
    pub(in crate::scalar::inverted) coordinate: Option<ElementCoordinateKey>,
}

/// The fixed-size part of the order below the score, greatest first: score, row
/// address, and the element ranks a packed key covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct TopKRankKey {
    score: OrderedFloat,
    order_key: Reverse<u64>,
    coordinate: Option<Reverse<ElementCoordinateKey>>,
}

/// Frequency slot of a candidate that has not been admitted to the heap.
const UNALLOCATED_SLOT: u32 = u32::MAX;

/// A live top-k candidate. Its (term, freq) pairs live in a [`FrequencySlots`]
/// slot, so heap churn moves a compact record instead of a `Vec`.
#[derive(Debug, Clone)]
struct TopKEntry {
    doc: ScoredDoc,
    order: TieOrder,
    doc_length: u32,
    posting_doc_id: u64,
    frequency_slot: u32,
    /// Element ranks past the packed key, shared by every entry of the partition.
    deeper_coordinates: Option<ResidentDocCoordinates>,
}

impl TopKEntry {
    fn rank_key(&self) -> TopKRankKey {
        TopKRankKey {
            score: self.doc.score,
            order_key: Reverse(self.order.address),
            coordinate: self.order.coordinate.map(Reverse),
        }
    }

    /// Compare the ranks deeper than the packed key. DocIds are posting-local and
    /// entries of one partition share one handle. Reversed: the result order takes
    /// the lower `doc_index` first, the heap ranks the winner greatest.
    fn cmp_deeper_ranks(&self, other: &Self) -> std::cmp::Ordering {
        match self
            .deeper_coordinates
            .as_ref()
            .or(other.deeper_coordinates.as_ref())
        {
            Some(columns) => columns
                .compare_deeper_ranks(
                    DocId::new(self.posting_doc_id as u32),
                    DocId::new(other.posting_doc_id as u32),
                )
                .reverse(),
            None => std::cmp::Ordering::Equal,
        }
    }

    /// Compare on the result order below the score: row address, then element path.
    fn cmp_result_order(&self, other: &Self) -> std::cmp::Ordering {
        self.rank_key()
            .cmp(&other.rank_key())
            .then_with(|| self.cmp_deeper_ranks(other))
    }

    /// The result order plus the posting DocId, which makes it total. DocId only
    /// splits candidates the result order leaves tied.
    fn cmp_total(&self, other: &Self) -> std::cmp::Ordering {
        self.cmp_result_order(other)
            .then_with(|| other.posting_doc_id.cmp(&self.posting_doc_id))
    }
}

impl PartialEq for TopKEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp_total(other) == std::cmp::Ordering::Equal
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
        self.cmp_total(other)
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
/// frequencies: a retained tie has no frequency slot.
struct TieCandidate {
    doc: ScoredDoc,
    doc_length: u32,
    posting_doc_id: u64,
    freqs: Vec<(u32, u32)>,
}

/// How much of the result order the walk can rank, which fixes what the
/// collector does with a document tied at the k-th score. Walks visit candidates
/// in ascending partition-local DocId order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::scalar::inverted) enum TieOrdering {
    /// Row addresses are resident, unique and strictly ascend with DocId, so a
    /// k-th-score tie has already lost: it is rejected, the band stays empty, and
    /// the score floor sits at the raw k-th score.
    AscendingRowAddresses,
    /// Row addresses are resident but DocId order does not follow them (a remapped
    /// partition), or one row owns several documents. The order key is total for a
    /// rank-0 partition and for an element partition whose coordinates the walk
    /// reads. A band fills where nothing separates the documents, and the walk then
    /// ranks on the DocId proxy and reports [`TopKCollector::band_truncated`].
    ByRowAddress,
    /// No resident addresses, so `order_key` is a constant and every k-th-score
    /// tie is unrankable. Ties are retained until the band fills, the pass stops
    /// with [`TopKCollector::aborted`], and the caller retries with addresses.
    DeferredAddresses,
}

/// Owns the top-k heap and the frequency slots referenced by its entries.
pub(super) struct TopKCollector {
    limit: usize,
    heap: TopKHeap,
    frequency_slots: FrequencySlots,
    tie_ordering: TieOrdering,
    /// Cap on `limit + tie_band.len()`. Both retaining modes need the cap: a
    /// deferred walk can tie the whole corpus at the k-th score, and a list row
    /// can hold more elements than `limit`.
    max_buffered: usize,
    /// Candidates the walk cannot rank against the current k-th entry and the
    /// heap has no room for. The k-th key improving clears them, so a query
    /// without ties leaves the band empty.
    tie_band: Vec<TieCandidate>,
    band_truncated: bool,
    /// Element ranks past the packed key. Heap records hold a cloned handle: the
    /// compare path allocates nothing.
    deeper_coordinates: Option<ResidentDocCoordinates>,
    /// True when every score is the caller's constant: BM25 ranks nothing.
    constant_score: bool,
    /// Set when the collector gave up on the pass rather than merely dropping
    /// candidates, so a later improvement cannot make the flag obsolete.
    aborted: bool,
}

impl TopKCollector {
    pub(super) fn new(
        limit: usize,
        initial_capacity: usize,
        tie_ordering: TieOrdering,
        deeper_coordinates: Option<ResidentDocCoordinates>,
        constant_score: bool,
    ) -> Self {
        let initial_capacity = initial_capacity.min(limit);
        Self {
            limit,
            heap: BinaryHeap::with_capacity(initial_capacity),
            frequency_slots: FrequencySlots::with_capacity(initial_capacity),
            tie_ordering,
            max_buffered: limit.saturating_add(SCORE_FLOOR_BUFFER),
            tie_band: Vec::new(),
            band_truncated: false,
            deeper_coordinates,
            constant_score,
            aborted: false,
        }
    }

    /// True when the walk gave up on part of its k-th-score tie band, by stopping
    /// at an unrankable tie or by dropping a candidate the band could not hold.
    pub(super) fn aborted(&self) -> bool {
        self.aborted
    }

    /// True when an address-ordered walk filled its band and ranked on the DocId
    /// proxy. A retry cannot recover what the cap dropped. [`TieOrdering`] lists
    /// which partitions reach the cap.
    pub(super) fn band_truncated(&self) -> bool {
        self.band_truncated
    }

    /// Insert a competitive result; true when it entered the heap. A full heap
    /// clears and reuses the evicted entry's frequency slot before pushing the
    /// replacement.
    pub(super) fn insert(
        &mut self,
        doc: ScoredDoc,
        order: TieOrder,
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

        // `OrderedFloat` compares with `total_cmp`, which ranks NaN above every
        // real score, so a NaN custom-scorer score would take the top of the
        // result.
        if doc.score.0.is_nan() {
            return Ok(false);
        }

        // Store the constant: a later merge ranks on the stored score.
        let doc = if self.constant_score {
            ScoredDoc {
                score: OrderedFloat(0.0),
                ..doc
            }
        } else {
            doc
        };

        if self.heap.len() < self.limit {
            let mut candidate = self.new_candidate(doc, order, doc_length, posting_doc_id);
            candidate.frequency_slot = self.frequency_slots.push(pairs)?;
            self.heap.push(Reverse(candidate));
            return Ok(true);
        }
        let mut candidate = self.new_candidate(doc, order, doc_length, posting_doc_id);
        let Some(Reverse(worst)) = self.heap.peek() else {
            return Err(Error::internal(
                "FTS top-k heap is empty while its nonzero limit is reached",
            ));
        };
        // Both comparisons borrow `worst`; the heap order is the same one.
        let order_cmp = candidate.cmp_result_order(worst);
        let total_cmp = order_cmp.then_with(|| worst.posting_doc_id.cmp(&candidate.posting_doc_id));

        // An equal order key is a tie the result order cannot rank. The DocId proxy
        // picks the heap slot and the loser goes to the band.
        if order_cmp == std::cmp::Ordering::Equal {
            // Resident ascending addresses make the order key unique.
            if self.tie_ordering == TieOrdering::AscendingRowAddresses {
                return Ok(false);
            }
            // An equal order key means two documents with the same element path:
            // `InnerBuilder::merge_from` concatenates partitions without dedup.
            if self.band_is_full() {
                self.record_band_full();
            } else if total_cmp == std::cmp::Ordering::Greater {
                // The proxy decides: this candidate takes the heap slot and the
                // displaced entry joins the band.
                let Some(Reverse(displaced)) = self.heap.pop() else {
                    return Err(Error::internal(
                        "FTS top-k heap entry disappeared during replacement",
                    ));
                };
                let displaced = self.retain_displaced(displaced, candidate, pairs)?;
                self.tie_band.push(displaced);
                return Ok(true);
            } else {
                self.tie_band.push(TieCandidate {
                    doc: candidate.doc,
                    doc_length: candidate.doc_length,
                    posting_doc_id: candidate.posting_doc_id,
                    freqs: pairs.collect(),
                });
                return Ok(false);
            }
        }

        match total_cmp {
            std::cmp::Ordering::Less | std::cmp::Ordering::Equal => Ok(false),
            std::cmp::Ordering::Greater => {
                // Checked before the pop, which would otherwise make the buffer look
                // one candidate emptier than it is.
                let band_full = self.band_is_full();
                let Some(Reverse(displaced)) = self.heap.pop() else {
                    return Err(Error::internal(
                        "FTS top-k heap entry disappeared during replacement",
                    ));
                };
                // The k-th order key after the replacement is the worse of what the
                // pop exposed and the newcomer, both known without pushing first.
                let kth_order_rises = match self.heap.peek() {
                    None => candidate.cmp_result_order(&displaced),
                    Some(Reverse(exposed)) => {
                        let kth = if exposed.cmp_total(&candidate) == std::cmp::Ordering::Greater {
                            &candidate
                        } else {
                            exposed
                        };
                        kth.cmp_result_order(&displaced)
                    }
                } == std::cmp::Ordering::Greater;
                let retain_displaced = if kth_order_rises {
                    // The k-th rank rose past the band, so no member of the band
                    // can win, and nothing a full band dropped earlier can win
                    // either: it ranked no higher than the band. The flag
                    // describes that band.
                    self.tie_band.clear();
                    self.band_truncated = false;
                    false
                } else if band_full {
                    self.record_band_full();
                    false
                } else {
                    true
                };

                if retain_displaced {
                    let displaced = self.retain_displaced(displaced, candidate, pairs)?;
                    self.tie_band.push(displaced);
                } else {
                    // The displaced frequencies have no owner, so the slot and its
                    // capacity are reused in place: what every improving candidate
                    // of a query without ties does.
                    let frequency_slot = displaced.frequency_slot;
                    self.frequency_slots.replace(frequency_slot, pairs)?;
                    candidate.frequency_slot = frequency_slot;
                    self.heap.push(Reverse(candidate));
                }
                Ok(true)
            }
        }
    }

    /// A heap record for a candidate, sharing the partition's deeper coordinate
    /// columns and holding no frequency slot until it is admitted.
    fn new_candidate(
        &self,
        doc: ScoredDoc,
        order: TieOrder,
        doc_length: u32,
        posting_doc_id: u64,
    ) -> TopKEntry {
        TopKEntry {
            doc,
            order,
            doc_length,
            posting_doc_id,
            frequency_slot: UNALLOCATED_SLOT,
            deeper_coordinates: self.deeper_coordinates.clone(),
        }
    }

    /// Push `candidate` with a fresh frequency slot and return the displaced entry
    /// as a retained tie, owning the frequencies its slot released.
    fn retain_displaced(
        &mut self,
        displaced: TopKEntry,
        mut candidate: TopKEntry,
        pairs: impl Iterator<Item = (u32, u32)>,
    ) -> Result<TieCandidate> {
        candidate.frequency_slot = self.frequency_slots.push(pairs)?;
        self.heap.push(Reverse(candidate));
        let freqs = self.frequency_slots.release(displaced.frequency_slot)?;
        Ok(TieCandidate {
            doc: displaced.doc,
            doc_length: displaced.doc_length,
            posting_doc_id: displaced.posting_doc_id,
            freqs,
        })
    }

    /// Record that the band could not take another candidate. A walk with resident
    /// but unordered addresses has no pass to retry, so it keeps ranking on the
    /// DocId proxy: the result order's winners still reach the final selection,
    /// and the group past the cap is dropped.
    fn record_band_full(&mut self) {
        match self.tie_ordering {
            // An address-ascending walk ranks its ties, so the band is empty.
            TieOrdering::AscendingRowAddresses => {
                debug_assert!(self.tie_band.is_empty());
            }
            TieOrdering::ByRowAddress => self.band_truncated = true,
            // A deferred walk can be retried with addresses, so stop walking rather
            // than keep scoring a corpus that ties at the k-th score throughout.
            TieOrdering::DeferredAddresses => self.abort_pass(),
        }
    }

    /// Give up on this pass: the k-th-score band is full and this walk cannot rank
    /// its members, so the rest of the walk only scores documents it discards.
    fn abort_pass(&mut self) {
        self.aborted = true;
    }

    /// Documents held for the final selection: the heap plus the retained band.
    pub(super) fn buffered(&self) -> usize {
        self.heap.len() + self.tie_band.len()
    }

    /// True when the retained band has reached its cap.
    fn band_is_full(&self) -> bool {
        self.buffered() >= self.max_buffered
    }

    /// True when a document with `score` cannot enter the top-k at any position in
    /// the result order. A walk that reaches documents in ascending row-address
    /// order has already seen the winner of a k-th-score tie, so only a score
    /// strictly above the k-th competes. Otherwise [`Self::insert`] ranks the tie.
    pub(super) fn rejects_score(&self, score: f32) -> bool {
        if score.is_nan() {
            return true;
        }
        if self.constant_score {
            // Score floors are inert: the k-th score is the caller's constant.
            return false;
        }
        self.kth_score_if_full()
            .is_some_and(|kth| match self.tie_ordering {
                TieOrdering::AscendingRowAddresses => score <= kth,
                TieOrdering::ByRowAddress | TieOrdering::DeferredAddresses => score < kth,
            })
    }

    pub(super) fn kth_score_if_full(&self) -> Option<f32> {
        if self.heap.len() == self.limit {
            self.heap.peek().map(|entry| entry.0.doc.score.0)
        } else {
            None
        }
    }

    /// True when a constant-score walk has nothing left to find: addresses ascend
    /// with DocId, and every document past the k-th entry's address loses.
    pub(super) fn constant_score_walk_is_done(&self, address: u64) -> bool {
        self.constant_score
            && self.tie_ordering == TieOrdering::AscendingRowAddresses
            && self.heap.len() == self.limit
            && self
                .heap
                .peek()
                .is_some_and(|Reverse(worst)| address > worst.order.address)
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
        // A retained tie may beat a heap entry once addresses are resolved, so
        // returning more than `limit` documents is safe: the caller re-selects.
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
