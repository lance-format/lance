// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Owned conjunction state that can leave the CPU pool for position-block I/O.

use super::*;
use crate::scalar::inverted::DocInfo;
use crate::scalar::inverted::index::PositionStreamCodec;
use lance_core::Error;
use smallvec::SmallVec;
use std::ops::ControlFlow;

/// One compact contiguous read containing complete adjacent position blocks.
#[derive(Clone)]
pub(in crate::scalar::inverted) struct PositionWindow {
    pub(in crate::scalar::inverted) first_block: usize,
    pub(in crate::scalar::inverted) codec: PositionStreamCodec,
    pub(in crate::scalar::inverted) offsets: Arc<[u32]>,
    pub(in crate::scalar::inverted) bytes: Arc<[u8]>,
}

impl PositionWindow {
    pub(in crate::scalar::inverted) fn contains(&self, index: usize) -> bool {
        index
            .checked_sub(self.first_block)
            .is_some_and(|relative| relative < self.offsets.len())
    }

    pub(in crate::scalar::inverted) fn block(&self, index: usize) -> Result<&[u8]> {
        let relative = index
            .checked_sub(self.first_block)
            .filter(|relative| *relative < self.offsets.len())
            .ok_or_else(|| {
                Error::index(format!(
                    "position block {index} is outside window starting at {}",
                    self.first_block
                ))
            })?;
        let start = self.offsets[relative] as usize;
        let end = self
            .offsets
            .get(relative + 1)
            .map_or(self.bytes.len(), |end| *end as usize);
        self.bytes.get(start..end).ok_or_else(|| {
            Error::index(format!(
                "position block {index} has invalid window range {start}..{end}"
            ))
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::scalar::inverted) struct PositionBlockRequest {
    pub(in crate::scalar::inverted) token_id: u32,
    pub(in crate::scalar::inverted) index: usize,
}

pub(in crate::scalar::inverted) enum PhraseProgress<C> {
    Read(Vec<PositionBlockRequest>),
    Done(Vec<DocCandidate<C>>),
}

/// No document/scorer/metrics borrow, I/O handle or parked CPU closure survives
/// a suspension. There is at most one encoded window per query clause.
pub(in crate::scalar::inverted) struct PhraseSearch {
    #[allow(clippy::vec_box)]
    lead: Vec<Box<PostingIterator>>,
    up_to: Option<u64>,
    last_doc: Option<u64>,
    and_max_score: f32,
    threshold: f32,
    pending: Option<(DocInfo, f32)>,
    collector: TopKCollector,
    limit: usize,
    doc_range: std::ops::Range<u64>,
    seek_start: Option<u64>,
    has_started: bool,
}

impl PhraseSearch {
    pub(in crate::scalar::inverted) fn new(
        mut postings: Vec<PostingIterator>,
        limit: usize,
    ) -> Result<Self> {
        if postings.is_empty()
            || postings.iter().any(|posting| {
                !posting.is_compressed() || posting.has_grouped_terms() || posting.doc().is_none()
            })
        {
            return Err(Error::invalid_input(
                "selective phrase search requires nonempty ungrouped compressed postings",
            ));
        }
        for posting in &mut postings {
            if let PostingList::Compressed(list) = &mut posting.list {
                list.positions = None;
            }
        }
        let mut lead = postings.into_iter().map(Box::new).collect::<Vec<_>>();
        lead.sort_unstable_by_key(|posting| posting.cost());
        Ok(Self {
            lead,
            up_to: None,
            last_doc: None,
            and_max_score: 0.0,
            threshold: 0.0,
            pending: None,
            collector: TopKCollector::new(limit, limit.min(BLOCK_SIZE * 10)),
            limit,
            doc_range: 0..u64::MAX,
            seek_start: None,
            has_started: false,
        })
    }

    /// Restrict iteration without changing partition/corpus scoring statistics.
    pub(in crate::scalar::inverted) fn with_doc_range(
        mut self,
        range: std::ops::Range<u64>,
    ) -> Result<Self> {
        if self.has_started {
            return Err(Error::invalid_input(
                "selective phrase document range cannot change after search starts",
            ));
        }
        if range.start > range.end || range.end > u64::from(u32::MAX) {
            return Err(Error::invalid_input(format!(
                "selective phrase document range {range:?} must be ordered within the dense DocId domain"
            )));
        }
        self.seek_start = Some(range.start);
        self.doc_range = range;
        Ok(self)
    }

    /// Exclusive position-block limit for each term, including a boundary block.
    pub(in crate::scalar::inverted) fn position_block_limits(&self) -> Vec<(u32, usize)> {
        self.lead
            .iter()
            .filter_map(|posting| match &posting.list {
                PostingList::Compressed(list) => Some((
                    posting.token_id,
                    list.block_first_docs()
                        .partition_point(|doc| u64::from(*doc) < self.doc_range.end),
                )),
                PostingList::Plain(_) => None,
            })
            .collect()
    }

    /// Supply a window containing the requested block. Repeated terms share it.
    pub(in crate::scalar::inverted) fn supply(
        &mut self,
        request: PositionBlockRequest,
        window: PositionWindow,
    ) -> Result<()> {
        if !window.contains(request.index) {
            return Err(Error::invalid_input(
                "supplied position window does not contain requested block",
            ));
        }
        let mut found = false;
        for posting in &mut self.lead {
            if posting.token_id == request.token_id {
                let PostingList::Compressed(list) = &posting.list else {
                    return Err(Error::internal(
                        "selective phrase posting became uncompressed",
                    ));
                };
                if !window.contains(posting.index >> list.block_shift()) {
                    return Err(Error::invalid_input(
                        "supplied position block does not contain the pending document",
                    ));
                }
                posting.position_window = Some(window.clone());
                found = true;
            }
        }
        if !found || self.pending.is_none() {
            return Err(Error::invalid_input(
                "position block was not requested by a pending phrase candidate",
            ));
        }
        Ok(())
    }

    pub(in crate::scalar::inverted) fn advance<S: Scorer, D: WandDocuments>(
        &mut self,
        documents: &D,
        scorer: S,
        shared_threshold: Arc<AtomicU32>,
        metrics: &dyn MetricsCollector,
    ) -> Result<PhraseProgress<D::Candidate>> {
        self.has_started = true;
        let mut wand = Wand::new(Operator::And, std::iter::empty(), documents, scorer)
            .with_shared_threshold(shared_threshold);
        wand.lead = std::mem::take(&mut self.lead);
        wand.num_terms = wand.lead.len();
        wand.up_to = self.up_to;
        wand.and_last_doc = self.last_doc;
        wand.and_max_score = self.and_max_score;
        wand.threshold = self.threshold;
        // The selective path implements exact-phrase classic conjunctions.
        // Score-first's borrowed norm cache and bulk scratch stay in their
        // existing non-suspending paths.
        wand.score_first_and_enabled = false;
        if let Some(start) = self.seek_start.take() {
            wand.seek(start);
        }
        let mut comparisons = 0;
        let progress = loop {
            if self.limit == 0 {
                break PhraseProgress::Done(Vec::new());
            }
            wand.raise_to_shared_floor(1.0);
            let candidate = match self.pending.take() {
                Some(candidate) => Some(candidate),
                None => {
                    let candidate = wand.next()?;
                    if candidate.is_some() {
                        comparisons += 1;
                    }
                    candidate
                }
            };
            let Some((doc, _)) = candidate.filter(|(doc, _)| doc.doc_id() < self.doc_range.end)
            else {
                self.limit = 0;
                let collector = std::mem::replace(&mut self.collector, TopKCollector::new(0, 0));
                break PhraseProgress::Done(
                    collector.into_candidates(|key| documents.candidate_from_key(key))?,
                );
            };
            let Some(document_key) = documents.document_key(&doc) else {
                continue;
            };
            let doc_length = documents.doc_length(&doc);
            let score = wand.score_in_query_order(doc_length);
            if wand.exclusive_score_cannot_beat_floor(score) || self.collector.rejects_score(score)
            {
                continue;
            }
            for posting in &mut wand.lead {
                let PostingList::Compressed(list) = &posting.list else {
                    return Err(Error::internal(
                        "selective phrase posting became uncompressed",
                    ));
                };
                let index = posting.index >> list.block_shift();
                if posting
                    .position_window
                    .as_ref()
                    .is_some_and(|window| !window.contains(index))
                {
                    // Drop stale payloads before requesting replacements,
                    // including clauses that this candidate may not reach.
                    posting.position_window = None;
                }
            }
            // The first pair must align before any later clause can match.
            // Keep those two loads parallel; defer every other clause to the
            // confirmation's actual demand rather than loading all stopwords.
            let mut order = (0..wand.lead.len()).collect::<SmallVec<[usize; 16]>>();
            order.sort_unstable_by_key(|&index| {
                wand.lead[index]
                    .doc()
                    .map(|doc| doc.frequency())
                    .unwrap_or(u32::MAX)
            });
            let mut requests = Vec::with_capacity(2);
            for &index in order.iter().take(2) {
                let posting = &wand.lead[index];
                if posting.position_window.is_some() {
                    continue;
                }
                let PostingList::Compressed(list) = &posting.list else {
                    return Err(Error::internal(
                        "selective phrase posting became uncompressed",
                    ));
                };
                let request = PositionBlockRequest {
                    token_id: posting.token_id,
                    index: posting.index >> list.block_shift(),
                };
                if !requests.contains(&request) {
                    requests.push(request);
                }
            }
            if !requests.is_empty() {
                self.pending = Some((doc, score));
                break PhraseProgress::Read(requests);
            }
            let matched = match wand.check_exact_position_windows()? {
                ControlFlow::Continue(matched) => matched,
                ControlFlow::Break(request) => {
                    self.pending = Some((doc, score));
                    break PhraseProgress::Read(vec![request]);
                }
            };
            if matched
                && self.collector.insert(
                    ScoredDoc::new(document_key, score),
                    doc_length,
                    doc.doc_id(),
                    wand.iter_term_freqs(),
                )?
                && let Some(kth) = self.collector.kth_score_if_full()
            {
                wand.update_threshold(kth, 1.0);
            }
        };
        metrics.record_comparisons(comparisons);
        self.lead = std::mem::take(&mut wand.lead);
        self.up_to = wand.up_to;
        self.last_doc = wand.and_last_doc;
        self.and_max_score = wand.and_max_score;
        self.threshold = wand.threshold;
        Ok(progress)
    }
}
