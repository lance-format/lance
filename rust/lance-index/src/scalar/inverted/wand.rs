// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#[cfg(test)]
use std::cell::Cell;
use std::cell::OnceCell;
use std::collections::BinaryHeap;
use std::fmt::Debug;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, LazyLock, OnceLock};

use itertools::Itertools;
use lance_core::Result;
use smallvec::SmallVec;

use crate::metrics::MetricsCollector;

use super::PostingList;
use super::builder::{BLOCK_SIZE, ScoredDoc};
use super::encoding::MAX_POSTING_BLOCK_SIZE;
use super::index::dequantize_doc_length;
use super::query::{FtsSearchParams, Operator};
use super::scorer::Scorer;

mod bounds;
mod bulk_conjunction;
mod collector;
mod conjunction;
mod cursor;
mod disjunction;
mod doc_source;
mod intersection;
mod maxscore;
mod phrase;
mod posting;

pub(super) use bounds::{
    CompetitiveFloorMode, outward_f32_upper_bound, score_sum_cannot_compete,
    score_sum_upper_bound_factor,
};
pub(super) use cursor::WandCursor;
pub use doc_source::DocCandidate;
// The pre-split `wand::` paths for these two stay valid (issue #10060) even
// though only `doc_source` itself and the tests name them, so a plain lib build
// sees no use for the restricted re-export.
#[allow(unused_imports)]
pub(super) use doc_source::{AllModernDocuments, ModernVisibility};
pub(super) use doc_source::{
    FlatDocuments, LegacyWandDocuments, ModernWandDocuments, WandDocuments,
};
pub use posting::PostingIterator;
pub(super) use posting::{GroupedTermScorer, validate_modern_posting_doc_ids};

use bounds::{ScoreContribution, score_contributions_in_query_order};
use collector::TopKCollector;

const TERMINATED_DOC_ID: u64 = u64::MAX;

type NormKCache<'a> = (&'a [u8], Arc<[f32; 256]>);

/// Norm-code BM25 denominator addends (Lucene's norm cache) computed once for
/// every partition a query searches with the same scorer. Each partition
/// otherwise rebuilds the same 256 entries for its first scored window.
///
/// Every [`Wand`] given one instance must use a scorer with identical
/// [`Scorer::doc_norm`] results; the first search to need the table fills it.
#[derive(Clone, Default)]
pub struct SharedNormAddends(Arc<OnceLock<Option<Arc<[f32; 256]>>>>);

fn norm_addends<S: Scorer + ?Sized>(scorer: &S) -> Option<Arc<[f32; 256]>> {
    let mut addends = [0f32; 256];
    for (code, slot) in addends.iter_mut().enumerate() {
        *slot = scorer.doc_norm(dequantize_doc_length(code as u8))?;
    }
    Some(Arc::new(addends))
}

pub static FLAT_SEARCH_PERCENT_THRESHOLD: LazyLock<u64> = LazyLock::new(|| {
    std::env::var("LANCE_FLAT_SEARCH_PERCENT_THRESHOLD")
        .unwrap_or_else(|_| "10".to_string())
        .parse::<u64>()
        .unwrap_or(10)
});

// Bulk MAXSCORE path for top-k disjunctions (Lucene MaxScoreBulkScorer
// style). Default on: with right-sized partitions it wins by a wide margin
// (Lucene-parity latency) and its results are score-identical to the classic
// WAND loop. LANCE_FTS_MAXSCORE=0 opts back into the classic loop.
static USE_MAXSCORE_SEARCH: LazyLock<bool> =
    LazyLock::new(|| std::env::var("LANCE_FTS_MAXSCORE").as_deref() != Ok("0"));

// Bulk conjunction path for top-k AND / phrase queries: block-max window
// skipping plus a slice-level merge over decompressed blocks, replacing the
// per-doc `next()` leapfrog. Results are identical to the classic AND loop.
// LANCE_FTS_BULK_AND accepts auto (default), on/1, or off/0. Auto enables the
// bulk path for two and three clauses and for wider current-format conjunctions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum BulkAndMode {
    #[default]
    Auto,
    On,
    Off,
}

impl BulkAndMode {
    fn parse(value: &str) -> Option<Self> {
        let value = value.trim();
        if value.eq_ignore_ascii_case("auto") {
            Some(Self::Auto)
        } else if value.eq_ignore_ascii_case("on") || value == "1" {
            Some(Self::On)
        } else if value.eq_ignore_ascii_case("off") || value == "0" {
            Some(Self::Off)
        } else {
            None
        }
    }

    const fn enabled_for(self, num_clauses: usize) -> bool {
        match self {
            Self::Auto => matches!(num_clauses, 2 | 3),
            Self::On => true,
            Self::Off => false,
        }
    }
}

fn bulk_and_mode_from_env() -> BulkAndMode {
    match std::env::var("LANCE_FTS_BULK_AND") {
        Ok(value) => BulkAndMode::parse(&value).unwrap_or_else(|| {
            log::warn!(
                "Invalid LANCE_FTS_BULK_AND value {value:?}; expected auto, on/1, or off/0; \
                 falling back to auto"
            );
            BulkAndMode::Auto
        }),
        Err(std::env::VarError::NotPresent) => BulkAndMode::Auto,
        Err(std::env::VarError::NotUnicode(value)) => {
            log::warn!(
                "Invalid non-Unicode LANCE_FTS_BULK_AND value {value:?}; expected auto, on/1, \
                 or off/0; falling back to auto"
            );
            BulkAndMode::Auto
        }
    }
}

static BULK_AND_MODE: LazyLock<BulkAndMode> = LazyLock::new(bulk_and_mode_from_env);

struct HeadPosting {
    // Iterators that are already positioned on or after the next candidate doc.
    // The heap is ordered by smallest doc id so the top element determines
    // the next target doc to consider.
    doc_id: u64,
    posting: Box<PostingIterator>,
}

impl HeadPosting {
    fn new(posting: Box<PostingIterator>) -> Self {
        let doc_id = posting
            .doc()
            .map(|doc| doc.doc_id())
            .unwrap_or(TERMINATED_DOC_ID);
        Self { doc_id, posting }
    }

    fn doc_id(&self) -> u64 {
        self.doc_id
    }
}

impl PartialEq for HeadPosting {
    fn eq(&self, other: &Self) -> bool {
        self.doc_id == other.doc_id
            && self.posting.approximate_upper_bound().to_bits()
                == other.posting.approximate_upper_bound().to_bits()
            && self.posting.token_id == other.posting.token_id
            && self.posting.position == other.posting.position
    }
}

impl Eq for HeadPosting {}

impl PartialOrd for HeadPosting {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeadPosting {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .doc_id
            .cmp(&self.doc_id)
            .then_with(|| {
                self.posting
                    .approximate_upper_bound()
                    .total_cmp(&other.posting.approximate_upper_bound())
            })
            .then_with(|| self.posting.token_id.cmp(&other.posting.token_id))
            .then_with(|| self.posting.position.cmp(&other.posting.position))
    }
}

struct TailPosting {
    // Iterators that lag behind the current target doc but may still help the
    // target beat the threshold if advanced to that doc.
    upper_bound: f32,
    // Used as a tie-breaker when upper bounds are equal. Lower-cost iterators
    // are cheaper to advance, so they are preferred.
    cost: usize,
    posting: Box<PostingIterator>,
}

impl TailPosting {
    fn new(upper_bound: f32, cost: usize, posting: Box<PostingIterator>) -> Self {
        Self {
            upper_bound,
            cost,
            posting,
        }
    }
}

impl PartialEq for TailPosting {
    fn eq(&self, other: &Self) -> bool {
        self.upper_bound.to_bits() == other.upper_bound.to_bits()
            && self.cost == other.cost
            && self.posting.token_id == other.posting.token_id
            && self.posting.position == other.posting.position
    }
}

#[cfg(test)]
#[derive(Default)]
struct AndWindowStats {
    windows_wide: usize,
    windows_narrow: usize,
    windows_skipped: usize,
    range_blocks_scanned: usize,
    candidates_returned: usize,
    score_first_rejections: usize,
    pairwise_intersections: usize,
    windows_sliced: usize,
    windows_frequency_pruned: usize,
}

impl Eq for TailPosting {}

impl PartialOrd for TailPosting {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TailPosting {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.upper_bound
            .total_cmp(&other.upper_bound)
            .then_with(|| other.cost.cmp(&self.cost))
            .then_with(|| other.posting.token_id.cmp(&self.posting.token_id))
            .then_with(|| other.posting.position.cmp(&self.posting.position))
    }
}

pub struct Wand<'a, S: Scorer, D: WandDocuments> {
    threshold: f32, // multiple of factor and the minimum score of the top-k documents
    floor_mode: CompetitiveFloorMode,
    operator: Operator,
    num_terms: usize,
    // Posting iterators whose current doc id is >= the next target doc.
    // The heap top gives the smallest current doc id.
    head: BinaryHeap<HeadPosting>,
    #[allow(clippy::vec_box)]
    // Posting iterators that already match the current target doc.
    // Only these iterators participate in scoring / phrase checks for the
    // current candidate.
    lead: Vec<Box<PostingIterator>>,
    // Posting iterators that are behind the current target doc but still kept
    // in play because their score upper bound could affect the decision for the
    // current candidate.
    tail: BinaryHeap<TailPosting>,
    // Conservatively rounded sum of upper bounds for all iterators in `tail`.
    // It is maintained in f64 so candidate checks stay O(1) without allowing
    // repeated f32 add/subtract rounding to underestimate the remaining score.
    tail_max_score: f64,
    // Block-max scores are valid for all candidate docs up to this doc id.
    // `None` means the window has not been initialized yet and the next
    // candidate must refresh block-max state before making pruning decisions.
    up_to: Option<u64>,
    // First unchecked disjunction doc. Head postings may jump beyond this
    // while a clause stays parked in tail under a window-local bound.
    or_next_target: u64,
    // For conjunctions, this is the maximum attainable score for the current
    // block-max window `[target, up_to]`.
    and_max_score: f32,
    // Last conjunction doc returned to the caller. The next conjunction search
    // resumes strictly after this doc, like Lucene's `nextDoc()/advance()`.
    and_last_doc: Option<u64>,
    // Exact query-order score computed while progressively confirming a
    // conjunction candidate. This is valid only for `and_last_doc`.
    and_candidate_score: Option<f32>,
    // Score-first conjunction pruning is restricted to the current impact
    // format and scorers that expose the BM25 denominator contract.
    score_first_and_enabled: bool,
    #[cfg(test)]
    disable_score_first_and: bool,
    // Suffix sums of clause bounds for the current conjunction window. Entry
    // `i` bounds clauses `[i, num_terms)`, without subtracting prior bounds.
    score_first_and_suffix_bounds: SmallVec<[f64; 8]>,
    score_first_and_bounds_up_to: Option<u64>,
    // Inclusive candidate range where every current posting block is dense.
    // This selects score-first scheduling only; membership is still checked.
    score_first_and_dense_range: Option<(u64, u64)>,
    // Initialized only by a certified score-first candidate; cached `None`
    // avoids retrying unsupported norm tables on later candidates.
    score_first_and_norm_k: OnceCell<Option<NormKCache<'a>>>,
    #[cfg(test)]
    and_window_stats: AndWindowStats,
    // Test-only override for comparing bulk and classic conjunctions without
    // mutating the process-wide environment.
    bulk_and_mode_override: Option<BulkAndMode>,
    #[cfg(test)]
    bulk_and_searches: usize,
    #[cfg(test)]
    maxscore_single_essential_windows: usize,
    #[cfg(test)]
    maxscore_general_windows: usize,
    #[cfg(test)]
    maxscore_essential_blocks_skipped: usize,
    // How many windows took the top2-gap shortcut (wand.rs top2 branch),
    // as opposed to the plain single-essential path. Test-only: like the
    // counters beside it, nothing in a release build reads it.
    #[cfg(test)]
    maxscore_top2_gap_windows: usize,
    // How many times the top2-gap shortcut was declined because its span was
    // empty or sat below the window floor (GUARD 2). Test-only.
    #[cfg(test)]
    maxscore_top2_gap_empty_span: usize,
    #[cfg(test)]
    phrase_position_checks: Cell<usize>,
    documents: &'a D,
    scorer: S,
    // Shared cross-partition top-k floor. Each partition publishes its local
    // k-th score (`atomic_store_max_f32`) and prunes against the running value
    // -- a lower bound on the global k-th, so it never drops a real top-k doc.
    shared_threshold: Option<Arc<AtomicU32>>,
    shared_norm_addends: Option<SharedNormAddends>,
}

/// Monotonically raise an f32 stored in an `AtomicU32` to `val`. CAS loop (not a
/// bit-max) so it stays correct for negative scores -- BM25 idf can go negative.
fn atomic_store_max_f32(slot: &AtomicU32, val: f32) {
    let mut cur = slot.load(Ordering::Relaxed);
    while val > f32::from_bits(cur) {
        match slot.compare_exchange_weak(cur, val.to_bits(), Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => cur = actual,
        }
    }
}

// we were using row id as doc id in the past, which is u64,
// but now we are using the index as doc id, which is u32.
// so here WAND is a generic struct that can be used for both u32 and u64 doc ids.
impl<'a, S: Scorer, D: WandDocuments> Wand<'a, S, D> {
    pub(crate) fn new(
        operator: Operator,
        postings: impl Iterator<Item = PostingIterator>,
        documents: &'a D,
        scorer: S,
    ) -> Self {
        let mut head = BinaryHeap::new();
        let mut lead = Vec::new();
        for posting in postings {
            if posting.current_doc_id().is_none() {
                continue;
            }
            let posting = Box::new(posting);
            if operator == Operator::And {
                lead.push(posting);
            } else {
                head.push(HeadPosting::new(posting));
            }
        }
        if operator == Operator::And {
            lead.sort_unstable_by_key(|posting| posting.cost());
        }

        let score_first_and_enabled = operator == Operator::And
            && matches!(lead.len(), 4 | 5)
            && lead.iter().enumerate().all(|(index, posting)| {
                lead[..index]
                    .iter()
                    .all(|prior| prior.token_id != posting.token_id)
            })
            && lead.iter().all(|posting| {
                !posting.has_grouped_terms()
                    && !posting.use_scorer_upper_bound
                    && posting.query_weight.is_finite()
                    && posting.query_weight >= 0.0
                    && matches!(
                        &posting.list,
                        PostingList::Compressed(list)
                            if list.block_size == MAX_POSTING_BLOCK_SIZE
                                && list.impacts.is_some()
                    )
            })
            && scorer
                .doc_norm(1)
                .is_some_and(|norm| norm.is_finite() && norm >= 0.0);

        Self {
            // Standalone leaf top-k keeps its established exclusive floor.
            // Composable cursors switch to an inclusive floor because their
            // downstream collector applies the final document-key tie-breaker.
            threshold: 0.0,
            floor_mode: CompetitiveFloorMode::Exclusive,
            operator,
            num_terms: if operator == Operator::And {
                lead.len()
            } else {
                head.len()
            },
            head,
            lead,
            tail: BinaryHeap::new(),
            tail_max_score: 0.0,
            up_to: None,
            or_next_target: 0,
            and_max_score: f32::INFINITY,
            and_last_doc: None,
            and_candidate_score: None,
            score_first_and_enabled,
            #[cfg(test)]
            disable_score_first_and: false,
            score_first_and_suffix_bounds: SmallVec::new(),
            score_first_and_bounds_up_to: None,
            score_first_and_dense_range: None,
            score_first_and_norm_k: OnceCell::new(),
            #[cfg(test)]
            and_window_stats: AndWindowStats::default(),
            bulk_and_mode_override: None,
            #[cfg(test)]
            bulk_and_searches: 0,
            #[cfg(test)]
            maxscore_single_essential_windows: 0,
            #[cfg(test)]
            maxscore_general_windows: 0,
            #[cfg(test)]
            maxscore_essential_blocks_skipped: 0,
            #[cfg(test)]
            maxscore_top2_gap_windows: 0,
            #[cfg(test)]
            maxscore_top2_gap_empty_span: 0,
            #[cfg(test)]
            phrase_position_checks: Cell::new(0),
            documents,
            scorer,
            shared_threshold: None,
            shared_norm_addends: None,
        }
    }

    /// Test hook: force one conjunction mode so parity tests can compare bulk
    /// and classic search within one process.
    #[cfg(test)]
    fn with_bulk_and_mode(mut self, mode: BulkAndMode) -> Self {
        self.bulk_and_mode_override = Some(mode);
        self
    }

    /// Test hook for old/new score-first parity in one process.
    #[cfg(test)]
    fn with_score_first_and(mut self, enabled: bool) -> Self {
        self.disable_score_first_and = !enabled;
        self
    }

    fn with_floor_mode(mut self, mode: CompetitiveFloorMode) -> Self {
        self.floor_mode = mode;
        self
    }

    /// Share one cross-partition top-k floor across a query's partitions.
    pub(crate) fn with_shared_threshold(mut self, shared: Arc<AtomicU32>) -> Self {
        self.shared_threshold = Some(shared);
        self
    }

    /// Share one norm addend table across a query's partitions. The caller
    /// must pass the same table only to searches whose scorers agree on
    /// [`Scorer::doc_norm`].
    pub(crate) fn with_shared_norm_addends(mut self, shared: SharedNormAddends) -> Self {
        self.shared_norm_addends = Some(shared);
        self
    }

    /// Per-search norm→BM25-denominator cache (Lucene's norm cache): the doc
    /// byte-norm slab plus the 256 possible denominator addends. Available
    /// when the DocSet scores quantized (256-document-block partitions) and the scorer
    /// factors `doc_weight` as `(K1+1)*freq/(freq + addend)`. Scoring through
    /// the cache is bit-identical to `scorer.doc_weight`, because both
    /// evaluate the same expressions on the same quantized lengths.
    fn norm_k_cache(&self) -> Option<NormKCache<'a>> {
        let norms = self.documents.scoring_norms()?;
        let addends = match &self.shared_norm_addends {
            Some(shared) => shared
                .0
                .get_or_init(|| norm_addends(&self.scorer))
                .clone()?,
            None => norm_addends(&self.scorer)?,
        };
        Some((norms, addends))
    }

    /// Set the pruning threshold from this partition's k-th best, raised to the
    /// shared cross-partition floor when one is attached.
    fn update_threshold(&mut self, local_kth: f32, wand_factor: f32) {
        let mut t = local_kth * wand_factor;
        if let Some(shared) = self.shared_threshold.as_ref() {
            atomic_store_max_f32(shared, local_kth);
            let g = f32::from_bits(shared.load(Ordering::Relaxed)) * wand_factor;
            if g > t {
                t = g;
            }
        }
        if self.score_first_and_enabled && self.threshold <= 0.0 && t > 0.0 {
            // A zero-floor window did not prepare block bounds. Force the next
            // conjunction candidate to establish a score-first window.
            self.up_to = None;
            self.invalidate_score_first_and_window();
        }
        self.threshold = t;
    }

    /// Raise the local threshold to the shared cross-partition floor, picking up
    /// updates published by sibling partitions.
    fn raise_to_shared_floor(&mut self, wand_factor: f32) {
        if let Some(shared) = self.shared_threshold.as_ref() {
            let g = f32::from_bits(shared.load(Ordering::Relaxed)) * wand_factor;
            if g > self.threshold {
                if self.score_first_and_enabled && self.threshold <= 0.0 && g > 0.0 {
                    self.up_to = None;
                    self.invalidate_score_first_and_window();
                }
                self.threshold = g;
            }
        }
    }

    // search the top-k documents that contain the query
    // returns the row_id, frequency and doc length
    pub(crate) fn search(
        &mut self,
        params: &FtsSearchParams,
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<DocCandidate<D::Candidate>>> {
        let limit = params.limit.unwrap_or(usize::MAX);
        if limit == 0 {
            return Ok(vec![]);
        }

        if params.phrase_slop.is_some() {
            self.invalidate_score_first_and_window();
            self.score_first_and_enabled = false;
        }

        if self.operator == Operator::Or
            && let Some((num_docs_selected, documents)) = self.documents.flat_documents()
            && num_docs_selected.saturating_mul(100)
                <= (*FLAT_SEARCH_PERCENT_THRESHOLD as usize).saturating_mul(self.documents.len())
        {
            return self.flat_search(params, documents, metrics);
        }

        // Top-k disjunctions over compressed lists can opt into the bulk
        // MAXSCORE path (Lucene MaxScoreBulkScorer style): it streams whole
        // blocks of the essential clauses into a window accumulator instead of
        // advancing doc-at-a-time through a heap.
        if *USE_MAXSCORE_SEARCH
            && self.operator == Operator::Or
            && params.phrase_slop.is_none()
            && !self.head.is_empty()
            && self.head.iter().all(|posting| {
                posting.posting.is_compressed() && !posting.posting.has_grouped_terms()
            })
        {
            return self.maxscore_search(params, metrics);
        }

        // Top-k conjunctions (AND and phrase) over compressed lists use the
        // bulk path: the same block-max window pruning, but candidates come
        // from a slice-level merge over decompressed blocks instead of per-doc
        // `next()` leapfrogging through boxed iterators.
        if self.operator == Operator::And
            && !self.lead.is_empty()
            && self
                .lead
                .iter()
                .all(|posting| posting.is_compressed() && !posting.has_grouped_terms())
            && {
                let mode = self
                    .bulk_and_mode_override
                    .unwrap_or_else(|| *BULK_AND_MODE);
                mode.enabled_for(self.lead.len())
                    || (mode == BulkAndMode::Auto
                        && self.lead.len() >= 4
                        && self.lead.iter().all(|posting| {
                            matches!(&posting.list, PostingList::Compressed(list)
                                if list.block_size == MAX_POSTING_BLOCK_SIZE && list.impacts.is_some())
                        }))
            }
        {
            #[cfg(test)]
            {
                self.bulk_and_searches += 1;
            }
            return self.and_bulk_search(params, metrics);
        }

        let mut candidates = TopKCollector::new(limit, std::cmp::min(limit, BLOCK_SIZE * 10));
        let mut num_comparisons = 0;
        loop {
            self.raise_to_shared_floor(params.wand_factor);
            let Some((doc, and_score)) = self.next()? else {
                break;
            };
            num_comparisons += 1;

            let posting_doc_id = doc.doc_id();
            let Some(document_key) = self.documents.document_key(&doc) else {
                if self.operator == Operator::Or {
                    self.push_back_leads(doc.doc_id() + 1);
                }
                continue;
            };

            let doc_length = self.documents.doc_length(&doc);

            let score = if self.operator == Operator::Or {
                self.advance_all_tail(doc.doc_id(), None, None);
                if let Some(slop) = params.phrase_slop {
                    // Only front-load the BM25 total when a floor exists: with
                    // no floor the skip cannot fire, and scoring before
                    // position confirmation would make every position-failing
                    // candidate pay for a score it never uses.
                    let early_score =
                        (self.threshold > 0.0).then(|| self.score_in_query_order(doc_length));
                    if early_score
                        .is_some_and(|score| self.exclusive_score_cannot_beat_floor(score))
                    {
                        self.push_back_leads(doc.doc_id() + 1);
                        continue;
                    }
                    if !self.check_positions(slop as i32)? {
                        self.push_back_leads(doc.doc_id() + 1);
                        continue;
                    }
                    early_score.unwrap_or_else(|| self.score_in_query_order(doc_length))
                } else {
                    self.score_in_query_order(doc_length)
                }
            } else {
                self.advance_all_tail(doc.doc_id(), None, None);
                if let Some(slop) = params.phrase_slop {
                    // Only front-load the BM25 total when a floor exists: with
                    // no floor the skip cannot fire, and scoring before
                    // position confirmation would make every position-failing
                    // candidate pay for a score it never uses.
                    let early_score = (self.threshold > 0.0).then(|| {
                        if self.and_candidate_score.is_some() {
                            and_score
                        } else {
                            self.score_in_query_order(doc_length)
                        }
                    });
                    if early_score
                        .is_some_and(|score| self.exclusive_score_cannot_beat_floor(score))
                    {
                        continue;
                    }
                    if !self.check_positions(slop as i32)? {
                        continue;
                    }
                    early_score.unwrap_or_else(|| {
                        if self.and_candidate_score.is_some() {
                            and_score
                        } else {
                            self.score_in_query_order(doc_length)
                        }
                    })
                } else if self.and_candidate_score.is_some() {
                    and_score
                } else {
                    self.score_in_query_order(doc_length)
                }
            };

            if candidates.insert(
                ScoredDoc::new(document_key, score),
                doc_length,
                posting_doc_id,
                self.iter_term_freqs(),
            )? && let Some(kth) = candidates.kth_score_if_full()
            {
                self.update_threshold(kth, params.wand_factor);
            }
            if self.operator == Operator::Or {
                self.push_back_leads(doc.doc_id() + 1);
            }
        }
        metrics.record_comparisons(num_comparisons);

        candidates.into_candidates(|key| self.documents.candidate_from_key(key))
    }

    fn flat_search(
        &mut self,
        params: &FtsSearchParams,
        documents: Box<dyn Iterator<Item = (u64, u64)> + '_>,
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<DocCandidate<D::Candidate>>> {
        let limit = params.limit.unwrap_or(usize::MAX);
        if limit == 0 {
            return Ok(vec![]);
        }

        // Posting iterators are forward-only, so selected DocIds are sorted
        // before driving the sparse executor.
        let documents = documents.sorted_unstable().collect::<Vec<_>>();
        let is_compressed = self
            .head
            .peek()
            .map(|posting| matches!(posting.posting.list, PostingList::Compressed(_)))
            .or_else(|| {
                self.lead
                    .first()
                    .map(|posting| matches!(posting.list, PostingList::Compressed(_)))
            })
            .unwrap_or(false);

        let mut num_comparisons = 0;
        let mut candidates = TopKCollector::new(limit, 0);
        for (doc_id, document_key) in documents {
            num_comparisons += 1;
            self.move_head_before_target_to_tail(doc_id);
            self.move_head_doc_to_lead(doc_id);

            if self.lead.is_empty() && self.tail.is_empty() {
                continue;
            }

            if !self.can_target_beat_threshold(doc_id) {
                self.advance_tail_and_lead_to_head(doc_id + 1);
                continue;
            }

            self.collect_tail_matches(doc_id);

            if self.operator == Operator::And && self.lead.len() < self.num_terms {
                self.advance_lead_to_head(doc_id + 1);
                continue;
            }

            // score the doc
            let doc_length = self
                .documents
                .flat_doc_length(doc_id, document_key, is_compressed);
            if self.operator == Operator::Or && !self.refine_or_candidate(doc_id, doc_length) {
                // `flat_search` evaluates an explicit allow-list of doc ids. Unlike the
                // regular WAND path, skipping to the next block boundary is unsafe here
                // because later doc ids from the same block may still be present in the
                // allow-list and need to be evaluated individually.
                self.advance_tail_and_lead_to_head(doc_id + 1);
                continue;
            }

            self.collect_tail_matches(doc_id);
            // Only front-load the BM25 total when a floor exists: with no floor
            // the skip cannot fire, and scoring before position confirmation
            // would make every position-failing candidate pay for a score it
            // never uses.
            let early_score = (self.threshold > 0.0).then(|| self.score_in_query_order(doc_length));
            if let Some(slop) = params.phrase_slop {
                if early_score.is_some_and(|score| self.exclusive_score_cannot_beat_floor(score)) {
                    self.advance_lead_to_head(doc_id + 1);
                    continue;
                }
                if !self.check_positions(slop as i32)? {
                    self.advance_lead_to_head(doc_id + 1);
                    continue;
                }
            }
            let score = early_score.unwrap_or_else(|| self.score_in_query_order(doc_length));

            if candidates.insert(
                ScoredDoc::new(document_key, score),
                doc_length,
                doc_id,
                self.iter_term_freqs(),
            )? && let Some(kth) = candidates.kth_score_if_full()
            {
                self.update_threshold(kth, params.wand_factor);
            }

            self.advance_lead_to_head(doc_id + 1);
        }
        metrics.record_comparisons(num_comparisons);

        candidates.into_candidates(|key| self.documents.candidate_from_key(key))
    }

    /// Exclusive top-k floor: a finite complete score at or below `threshold`
    /// cannot enter the heap, so phrase confirmation is wasted work.
    ///
    /// The score is the same query-order f32 the heap compares, so this uses
    /// `accepts_score` rather than the inflated partial-sum bound. Fail-closed:
    /// `threshold == 0` (heap not full) never skips.
    #[inline]
    fn exclusive_score_cannot_beat_floor(&self, score: f32) -> bool {
        self.threshold > 0.0
            && score.is_finite()
            && !self.floor_mode.accepts_score(score, self.threshold)
    }

    /// Calculate the current document's score in query order.
    ///
    /// WAND deliberately reorders postings by doc id, cost, and score bound.
    /// Floating-point addition is not associative, so that execution order
    /// must not leak into the public score or top-k tie boundary.
    fn score_in_query_order(&self, doc_length: u32) -> f32 {
        if self.lead.windows(2).all(|window| {
            (window[0].position, window[0].token_id) <= (window[1].position, window[1].token_id)
        }) {
            return self.lead.iter().fold(0.0_f32, |score, posting| {
                score
                    + posting.doc().map_or(0.0, |doc| {
                        posting.score(&self.scorer, doc.frequency(), doc_length)
                    })
            });
        }
        let contributions = self
            .lead
            .iter()
            .filter_map(|posting| {
                posting.doc().map(|doc| {
                    (
                        (posting.position, posting.token_id),
                        posting.score(&self.scorer, doc.frequency(), doc_length),
                    )
                })
            })
            .collect::<SmallVec<[ScoreContribution; 8]>>();
        score_contributions_in_query_order(contributions)
    }

    // iterate over all the preceding terms and collect the term index and frequency
    fn iter_term_freqs(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.lead.iter().filter_map(|posting| {
            posting
                .doc()
                .map(|doc| (posting.term_index(), doc.frequency()))
        })
    }
}

#[cfg(test)]
mod tests;
