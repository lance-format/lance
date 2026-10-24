// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Boolean scoring for the `intersection_union` shape: K required term
//! clauses plus N optional ones (named after the Search Benchmark Game's
//! query of that shape).
//!
//! Two kernels cover the family. `iu_tight_search` handles a single
//! required clause: walk MUST until the heap floor proves some SHOULD is
//! individually required (`max(MUST) + Σ_{j≠i} max(SHOULD_j) < floor`),
//! then switch the lead to the cheapest such list. Promotion uses
//! list-wide bounds only, and the optional union is never made required as
//! a whole — that forces a full-union walk before any intersection pruning
//! can start, which measured slower. `iu_tight_search_multi` aligns two or
//! more required clauses by seek-to-max; its only floor-driven pruning is
//! ending the walk once the optional budget runs dry (see the per-candidate
//! check below).

use lance_core::Result;

use super::super::scorer::Scorer;
use super::{
    BulkAndMode, CompetitiveFloorMode, PostingIterator, TERMINATED_DOC_ID, WandDocuments,
    conservative_score_sum, score_sum_cannot_compete, score_sum_upper_bound_factor,
};

/// Whether `partial + remaining` can no longer reach `floor`. Wraps the
/// generic path's predicate so the widening and the inclusive floor semantics
/// live in one audited place; a document scoring exactly at the floor still
/// competes, which is the contract the IU kernels are tested against. Bounds
/// are widened by `conservative_score_sum` so f32 rounding cannot shrink them
/// below the true total and trigger premature IU termination: the consumer
/// scores the same values by sequential `f32` accumulation, so with
/// `n >= 3` clauses the accumulation order alone can exceed the f64 sum by
/// `(n-1) * f32::EPSILON` relative error, and a bound that tight would drop a
/// document tying the floor.
fn iu_cannot_compete(partial: f32, remaining: f32, floor: f32) -> bool {
    score_sum_cannot_compete(
        partial,
        f64::from(remaining),
        floor,
        score_sum_upper_bound_factor(2),
        CompetitiveFloorMode::Inclusive,
    )
}

/// Sum of the upper bounds of the cursors that have not run dry. The closure
/// is consulted only for live cursors, so a dry cursor's bound is neither
/// recomputed nor counted.
fn live_upper_bound(dry: &[bool], upper_bound: impl Fn(usize) -> f32) -> f32 {
    conservative_score_sum((0..dry.len()).filter(|&index| !dry[index]).map(upper_bound))
}

/// Env gate for the IU fast paths, following the `LANCE_FTS_BULK_AND`
/// convention: `auto` (default) and `on`/`1` are equivalent — both run both
/// kernels — and `off`/`0` routes every IU-shaped Boolean back to the generic
/// scorer tree. This is the A/B and rollout switch; it does not change
/// results, only which engine computes them.
static IU_TIGHT_MODE: std::sync::LazyLock<BulkAndMode> =
    std::sync::LazyLock::new(|| match std::env::var("LANCE_FTS_IU_TIGHT") {
        Ok(value) => BulkAndMode::parse(&value).unwrap_or_else(|| {
            log::warn!(
                "Invalid LANCE_FTS_IU_TIGHT value {value:?}; expected auto, on/1, or off/0; \
                 falling back to auto"
            );
            BulkAndMode::Auto
        }),
        Err(std::env::VarError::NotPresent) => BulkAndMode::Auto,
        Err(std::env::VarError::NotUnicode(value)) => {
            log::warn!(
                "Invalid non-Unicode LANCE_FTS_IU_TIGHT value {value:?}; expected auto, on/1, \
                 or off/0; falling back to auto"
            );
            BulkAndMode::Auto
        }
    });

/// Whether the IU fast paths may run in this process.
pub fn iu_tight_enabled() -> bool {
    *IU_TIGHT_MODE != BulkAndMode::Off
}

/// Promote only when the new lead is at least this many times cheaper than
/// MUST. A medium-df SHOULD that just barely becomes required is slower to
/// walk than finishing MUST.
pub const IU_TIGHT_LEAD_COST_RATIO: usize = 3;

pub fn iu_tight_lead_is_cheaper(must_cost: usize, lead_cost: usize) -> bool {
    lead_cost.saturating_mul(IU_TIGHT_LEAD_COST_RATIO) < must_cost
}

struct IuTightScoring<'a, D, S> {
    documents: &'a D,
    scorer: &'a S,
}

impl<'a, D: WandDocuments, S: Scorer> IuTightScoring<'a, D, S> {
    fn term_score(&self, posting: &PostingIterator, doc: u64, freq: u32) -> f32 {
        posting.score(
            self.scorer,
            freq,
            self.documents.scoring_num_tokens(doc as u32),
        )
    }
}

/// Walk every MUST doc, fold `must + sum(matching shoulds)`, and promote
/// to a sparse required SHOULD once the floor makes that list mandatory.
pub fn iu_tight_search<D, S>(
    documents: &D,
    scorer: &S,
    mut must: PostingIterator,
    mut shoulds: Vec<PostingIterator>,
    on_hit: &mut dyn FnMut(u64, f32) -> Result<bool>,
    floor: impl Fn() -> f32,
) -> Result<()>
where
    D: WandDocuments,
    S: Scorer,
{
    let scoring = IuTightScoring { documents, scorer };
    let must_ub = must.global_upper_bound(scorer);
    let should_ubs: Vec<f32> = shoulds
        .iter()
        .map(|should| should.global_upper_bound(scorer))
        .collect();
    let mut chunk = Vec::with_capacity(128);
    let mut last_doc = 0_u64;
    // The promotion decision is a pure function of the floor: the bounds are
    // list-wide constants and the floor only rises. Re-deriving it is O(N^2)
    // in the optional count, so reuse the decision while the floor is
    // unchanged.
    let mut promotion_decision: Option<(f32, Option<Vec<usize>>)> = None;
    // A SHOULD whose cursor ran dry contributes nothing from here on. When
    // even the must alone plus the live optionals cannot reach the floor,
    // the rest of the MUST walk is dead and ends early.
    let mut dry: Vec<bool> = vec![false; shoulds.len()];
    // Candidate doc ids of the current MUST block, ascending.
    let mut candidates: Vec<u64> = Vec::with_capacity(128);
    // Frequency scratch reused by merge_optional_freqs per optional clause.
    let mut freqs: Vec<u32> = Vec::with_capacity(128);
    let mut should_sums: Vec<f32> = Vec::with_capacity(128);
    loop {
        must.take_docs_one_block_upto(TERMINATED_DOC_ID, &mut chunk);
        if chunk.is_empty() {
            break;
        }
        candidates.clear();
        candidates.extend(chunk.iter().map(|(doc, _)| *doc));
        // `merge_optional_freqs` walks each SHOULD posting block-continuously
        // against the candidates: a block of a sparse MUST spans thousands of
        // document ids, so seeking each SHOULD posting once per candidate
        // pays a full block locate for every one of them; the merge turns
        // that into the block decodes that actually contain candidates, and
        // its buffers stay bounded by the candidate count instead of the
        // window's document span. Accumulating per clause in clause order
        // keeps the f32 sum bit-identical to the generic path's
        // `DisjunctionScorer` (Sum) plus `ReqOptScorer::score` accumulation.
        should_sums.clear();
        should_sums.resize(chunk.len(), 0.0);
        for (index, should) in shoulds.iter_mut().enumerate() {
            freqs.clear();
            freqs.resize(candidates.len(), 0);
            should.merge_optional_freqs(&candidates, &mut freqs);
            if should.current_doc_id().is_none() {
                dry[index] = true;
            }
            for (candidate, &freq) in freqs.iter().enumerate() {
                if freq != 0 {
                    should_sums[candidate] +=
                        scoring.term_score(should, candidates[candidate], freq);
                }
            }
        }
        for (candidate, &(doc, freq)) in chunk.iter().enumerate() {
            last_doc = doc;
            let Some(key) = scoring.documents.document_key_for_doc_id(doc as u32) else {
                continue;
            };
            let must_score = scoring.term_score(&must, doc, freq);
            let score = must_score + should_sums[candidate];
            if !on_hit(key, score)? {
                return Ok(());
            }
        }
        let current_floor = floor();
        if current_floor.is_finite() {
            // Dry optionals contribute nothing from here on; if the must
            // alone plus the live optionals cannot reach the floor, every
            // remaining MUST document is dead and the walk ends.
            let live_optional_ub = live_upper_bound(&dry, |index| should_ubs[index]);
            if iu_cannot_compete(must_ub, live_optional_ub, current_floor) {
                return Ok(());
            }
            if promotion_decision
                .as_ref()
                .is_none_or(|(decided_floor, _)| *decided_floor != current_floor)
            {
                promotion_decision = Some((
                    current_floor,
                    iu_tight_required_shoulds(must_ub, &should_ubs, current_floor),
                ));
            }
            if let Some(Some(required)) = promotion_decision.as_ref().map(|(_, decision)| decision)
            {
                if required.is_empty() {
                    return Ok(());
                }
                let lead_cost = required
                    .iter()
                    .map(|&index| shoulds[index].cost())
                    .min()
                    .unwrap_or(usize::MAX);
                if iu_tight_lead_is_cheaper(must.cost(), lead_cost) {
                    return scoring.promoted(
                        must,
                        shoulds,
                        required.clone(),
                        last_doc.saturating_add(1),
                        on_hit,
                    );
                }
            }
        }
    }
    Ok(())
}

/// IU kernel for K >= 2 required clauses plus N optional clauses.
///
/// Every required clause must hit the document; the optional clauses are
/// probed per candidate and summed. The classic seek-to-max convergence
/// aligns all required cursors on one document, then the optional clauses are
/// probed. No optional-to-required promotion: the required set is fixed.
pub fn iu_tight_search_multi<D, S>(
    documents: &D,
    scorer: &S,
    mut required: Vec<PostingIterator>,
    mut optional: Vec<PostingIterator>,
    on_hit: &mut dyn FnMut(u64, f32) -> Result<bool>,
    floor: impl Fn() -> f32,
) -> Result<()>
where
    D: WandDocuments,
    S: Scorer,
{
    debug_assert!(required.len() >= 2);
    let scoring = IuTightScoring { documents, scorer };
    let mut required_ub =
        conservative_score_sum(required.iter().map(|p| p.global_upper_bound(scorer)));
    let mut optional_ub =
        conservative_score_sum(optional.iter().map(|p| p.global_upper_bound(scorer)));
    let mut dry = vec![false; optional.len()];

    // Align every required cursor on one document: seek-to-max until they
    // agree. A required list running out empties the intersection. Alignment
    // uses the doc-id-only seek (`next_doc_id`), which leaves the frequency
    // stream lazy; `doc()` materializes it only for candidates that survive.
    let mut candidate = 0_u64;
    loop {
        let mut max_doc = candidate;
        for posting in required.iter_mut() {
            if posting.current_doc_id().is_none_or(|cur| cur < candidate) {
                posting.next_doc_id(candidate, true);
            }
            let Some(cur) = posting.current_doc_id() else {
                return Ok(());
            };
            max_doc = max_doc.max(cur);
        }
        if max_doc > candidate {
            candidate = max_doc;
            continue;
        }

        // A required posting may name an invisible document (deleted rows
        // stay in postings until compaction); the K=1 kernel skips those too.
        let Some(key) = documents.document_key_for_doc_id(candidate as u32) else {
            candidate += 1;
            continue;
        };

        // Accumulating the required clauses first, then the matching
        // optional clauses in plan order, is the same order as
        // RequiredConjunctionScorer plus DisjunctionScorer inside ReqOptScorer,
        // so the f32 result is bit-identical to the generic path.
        let mut required_sum = 0.0_f32;
        for posting in required.iter() {
            let Some(cur) = posting.doc() else {
                // The alignment loop above guarantees every required cursor
                // sits on `candidate`; a cursor that moved out from under us
                // can only mean the intersection is exhausted.
                return Ok(());
            };
            required_sum += scoring.term_score(posting, candidate, cur.frequency());
        }
        let mut optional_sum = 0.0_f32;
        let mut newly_dry = false;
        for (index, posting) in optional.iter_mut().enumerate() {
            if posting.current_doc_id().is_none_or(|cur| cur < candidate) {
                posting.next_doc_id(candidate, true);
            }
            // A cursor past every document can never contribute again: drop
            // its bound from the remaining budget for good.
            if posting.current_doc_id().is_none() && !dry[index] {
                dry[index] = true;
                newly_dry = true;
            }
            if posting.current_doc_id() == Some(candidate) {
                let Some(cur) = posting.doc() else {
                    continue;
                };
                optional_sum += scoring.term_score(posting, candidate, cur.frequency());
            }
        }
        // With the probes done, an optional that ran dry contributes nothing
        // forever. If even the full required lists cannot lift a document
        // past the floor with the remaining budget, every future candidate is
        // dead and the walk ends.
        if newly_dry {
            optional_ub =
                live_upper_bound(&dry, |index| optional[index].global_upper_bound(scorer));
        }
        let current_floor = floor();
        if current_floor.is_finite() {
            // Promotion (N == 1): MUST-only documents are provably dead —
            // required_ub < floor strictly — so only documents hit by the
            // optional can still compete. Merging the optional into the
            // required set turns the walk into a (K+1)-way seek-to-max
            // driven by the sparse optional cursor; the accumulation order
            // is unchanged (the promoted clause is appended last). N >= 2
            // keeps the list-wide termination: each of several optionals
            // needs its own necessity proof, which v1 does not attempt.
            if optional.len() == 1 && iu_cannot_compete(required_ub, 0.0, current_floor) {
                required.push(optional.remove(0));
                required_ub =
                    conservative_score_sum(required.iter().map(|p| p.global_upper_bound(scorer)));
                optional_ub = 0.0;
                continue;
            }
            if iu_cannot_compete(required_ub, optional_ub, current_floor) {
                return Ok(());
            }
        }
        if !on_hit(key, required_sum + optional_sum)? {
            return Ok(());
        }
        candidate += 1;
    }
}

/// SHOULD indices that are each individually required once MUST-only scores
/// can no longer reach `floor`. `Some(empty)` means even MUST plus every
/// SHOULD clause cannot compete. `None` means keep walking MUST.
pub fn iu_tight_required_shoulds(
    must_ub: f32,
    should_ubs: &[f32],
    floor: f32,
) -> Option<Vec<usize>> {
    if !must_ub.is_finite() || must_ub >= floor {
        return None;
    }
    if should_ubs.iter().any(|upper| !upper.is_finite()) {
        return None;
    }
    let sum_shoulds = conservative_score_sum(should_ubs.iter().copied());
    if iu_cannot_compete(must_ub, sum_shoulds, floor) {
        return Some(Vec::new());
    }
    let required: Vec<usize> = should_ubs
        .iter()
        .enumerate()
        .filter(|(index, _)| {
            let others = conservative_score_sum(
                should_ubs
                    .iter()
                    .enumerate()
                    .filter(|(other, _)| other != index)
                    .map(|(_, upper)| *upper),
            );
            iu_cannot_compete(must_ub, others, floor)
        })
        .map(|(index, _)| index)
        .collect();
    if required.is_empty() {
        None
    } else {
        Some(required)
    }
}

fn iu_tight_seek_eq(posting: &mut PostingIterator, doc: u64) -> Option<u32> {
    if posting.doc().is_some_and(|info| info.doc_id() < doc) {
        posting.next(doc);
    }
    match posting.doc() {
        Some(info) if info.doc_id() == doc => Some(info.frequency()),
        _ => None,
    }
}

impl<'a, D: WandDocuments, S: Scorer> IuTightScoring<'a, D, S> {
    fn promoted(
        &self,
        mut must: PostingIterator,
        mut shoulds: Vec<PostingIterator>,
        required: Vec<usize>,
        start_doc: u64,
        on_hit: &mut dyn FnMut(u64, f32) -> Result<bool>,
    ) -> Result<()> {
        let Some(lead) = required
            .iter()
            .copied()
            .min_by_key(|&index| shoulds[index].cost())
        else {
            return Ok(());
        };
        // `take_docs_one_block_upto` consumed whole MUST blocks, so the live
        // MUST cursor already sits at or past `start_doc`; the forward seek
        // below is a no-op guard. The lead seek is load-bearing: the merge
        // loop stops at the last candidate, which can leave the lead cursor
        // mid-block below `start_doc`.
        if must.doc().is_none_or(|info| info.doc_id() < start_doc) {
            must.next(start_doc);
        }
        if shoulds[lead]
            .doc()
            .is_some_and(|info| info.doc_id() < start_doc)
        {
            shoulds[lead].next(start_doc);
        }
        let mut chunk = Vec::with_capacity(128);
        loop {
            shoulds[lead].take_docs_one_block_upto(TERMINATED_DOC_ID, &mut chunk);
            if chunk.is_empty() {
                break;
            }
            for &(doc, lead_freq) in &chunk {
                if doc < start_doc {
                    continue;
                }
                let Some(must_freq) = iu_tight_seek_eq(&mut must, doc) else {
                    continue;
                };
                let mut all_required = true;
                for &index in &required {
                    if index == lead {
                        continue;
                    }
                    if iu_tight_seek_eq(&mut shoulds[index], doc).is_none() {
                        all_required = false;
                        break;
                    }
                }
                if !all_required {
                    continue;
                }
                let Some(key) = self.documents.document_key_for_doc_id(doc as u32) else {
                    continue;
                };
                let must_score = self.term_score(&must, doc, must_freq);
                let mut should_sum = 0.0_f32;
                for (index, should) in shoulds.iter_mut().enumerate() {
                    let freq = if index == lead {
                        Some(lead_freq)
                    } else {
                        iu_tight_seek_eq(should, doc)
                    };
                    if let Some(freq) = freq {
                        should_sum += self.term_score(should, doc, freq);
                    }
                }
                if !on_hit(key, must_score + should_sum)? {
                    return Ok(());
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{iu_tight_lead_is_cheaper, iu_tight_required_shoulds};

    #[test]
    fn required_shoulds_stays_on_must_while_must_alone_can_compete() {
        assert_eq!(iu_tight_required_shoulds(5.0, &[10.0], 5.0), None);
        assert_eq!(iu_tight_required_shoulds(5.0, &[10.0], 4.0), None);
    }

    #[test]
    fn required_shoulds_promotes_a_should_that_is_individually_necessary() {
        assert_eq!(iu_tight_required_shoulds(1.0, &[10.0], 11.0), Some(vec![0]));
        assert_eq!(
            iu_tight_required_shoulds(1.0, &[10.0, 0.5], 11.0),
            Some(vec![0])
        );
    }

    #[test]
    fn required_shoulds_exits_when_the_combined_bound_cannot_compete() {
        assert_eq!(
            iu_tight_required_shoulds(1.0, &[2.0, 3.0], 10.0),
            Some(Vec::new())
        );
    }

    #[test]
    fn required_shoulds_does_not_promote_when_no_single_should_is_necessary() {
        // MUST + either SHOULD still reaches the floor, so neither list is
        // individually required.
        assert_eq!(iu_tight_required_shoulds(1.0, &[10.0, 10.0], 11.0), None);
    }

    #[test]
    fn required_shoulds_does_not_empty_on_f32_rounding_boundary() {
        // Naive f32 sum rounds 1.0 + 2^-24 down to 1.0, which is below
        // floor = 1 + 2^-23 and would terminate the MUST walk early.
        let must_ub = 2.0_f32.powi(-24);
        let should_ubs = [1.0, 2.0_f32.powi(-24)];
        let floor = 1.0 + 2.0_f32.powi(-23);
        assert_ne!(
            iu_tight_required_shoulds(must_ub, &should_ubs, floor),
            Some(Vec::new()),
            "conservative bounds must not declare the combined bound uncompetitive"
        );
    }

    #[test]
    fn lead_cost_ratio_is_relative_not_an_absolute_df_constant() {
        assert!(iu_tight_lead_is_cheaper(90_000, 10_000));
        assert!(!iu_tight_lead_is_cheaper(20_000, 10_000));
        assert!(!iu_tight_lead_is_cheaper(30_000, 10_000));
        assert!(iu_tight_lead_is_cheaper(31_000, 10_000));
        // Close costs must not hand the promoted walk to the optional list:
        // airport ~56k vs security ~77k is exactly the shape where walking the
        // SHOULD would be slower, so the switch stays relative, not absolute.
        assert!(!iu_tight_lead_is_cheaper(56_000, 77_000));
        assert!(!iu_tight_lead_is_cheaper(56_000, 56_000));
    }
}
