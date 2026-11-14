// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Owned conjunction state that can leave the CPU pool for position-block I/O.

use super::*;

#[derive(Clone)]
pub(in crate::scalar::inverted) struct PositionBlock {
    pub(in crate::scalar::inverted) index: usize,
    pub(in crate::scalar::inverted) codec: PositionStreamCodec,
    // An exact-size allocation: never retain a slice of a coalesced I/O buffer.
    pub(in crate::scalar::inverted) bytes: Arc<[u8]>,
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
/// a suspension. There is at most one encoded block per query clause.
pub(in crate::scalar::inverted) struct PhraseSearch {
    #[allow(clippy::vec_box)]
    lead: Vec<Box<PostingIterator>>,
    up_to: Option<u64>,
    last_doc: Option<u64>,
    and_max_score: f32,
    threshold: f32,
    pending: Option<(DocInfo, f32)>,
    collector: TopKCollector,
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
        })
    }

    /// Supply only a requested block. Repeated query terms share its allocation.
    pub(in crate::scalar::inverted) fn supply(
        &mut self,
        request: PositionBlockRequest,
        block: PositionBlock,
    ) -> Result<()> {
        if request.index != block.index {
            return Err(Error::invalid_input(
                "supplied position block index differs from request",
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
                if posting.index >> list.block_shift() != block.index {
                    return Err(Error::invalid_input(
                        "supplied position block does not contain the pending document",
                    ));
                }
                posting.selected_position_block = Some(block.clone());
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
        let mut comparisons = 0;
        let progress = loop {
            if self.collector.limit == 0 {
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
            let Some((doc, _)) = candidate else {
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
            let mut requests = Vec::new();
            for posting in &mut wand.lead {
                let PostingList::Compressed(list) = &posting.list else {
                    return Err(Error::internal(
                        "selective phrase posting became uncompressed",
                    ));
                };
                let index = posting.index >> list.block_shift();
                if posting
                    .selected_position_block
                    .as_ref()
                    .is_none_or(|block| block.index != index)
                {
                    // Release the old allocation before asynchronous replacement.
                    posting.selected_position_block = None;
                    let request = PositionBlockRequest {
                        token_id: posting.token_id,
                        index,
                    };
                    if !requests.contains(&request) {
                        requests.push(request);
                    }
                }
            }
            if !requests.is_empty() {
                self.pending = Some((doc, score));
                break PhraseProgress::Read(requests);
            }
            if wand.check_exact_positions()?
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
