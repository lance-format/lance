// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Term-at-a-time MAXSCORE for `combined_fields`.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::super::scorer::{BM25_DOC_WEIGHT_UPPER_BOUND, CombinedFieldsBM25Scorer};
use super::super::wand::{
    CompetitiveFloorMode, score_sum_cannot_compete, score_sum_upper_bound_factor,
};
use super::cursor::MaterializedTerm;
use super::search::RankedDoc;

/// The constant per-term score ceiling, or 0 when the term cannot raise a score
/// (`idf <= 0`).
///
/// Lucene's `CombinedFieldQuery` uses `idf · (K1 + 1)`, but in f32 the evaluated
/// weight can land above that, and a ceiling one ULP short is enough to drop a
/// better document (see `test_combined_maxscore_ceiling_is_conservative`).
/// `doc_weight` clamps to [`BM25_DOC_WEIGHT_UPPER_BOUND`], so scaling by it is a
/// true bound.
#[inline]
pub(super) fn term_upper_bound(idf: f32) -> f32 {
    if idf > 0.0 {
        idf * BM25_DOC_WEIGHT_UPPER_BOUND
    } else {
        0.0
    }
}

/// Candidate counters for one MAXSCORE run, so tests can check that pruning
/// actually fires.
#[derive(Default, Debug, Clone, Copy)]
pub(super) struct MaxscoreStats {
    /// Candidates pulled from the essential terms' cursors.
    pub(super) discovered: u64,
    /// Discovered candidates rejected by the upper-bound test before probing
    /// the non-essential terms.
    pub(super) pruned: u64,
    /// Candidates fully scored.
    pub(super) scored: u64,
}

/// Term-at-a-time MAXSCORE over the cross-field postings.
///
/// Terms are sorted by ceiling. Once the heap holds `limit` docs, the longest
/// prefix whose summed ceilings cannot beat the k-th score (`threshold`) is
/// non-essential: only the remaining (essential) terms produce candidates, and a
/// candidate is dropped when its essential score plus the non-essential ceilings
/// still cannot beat `threshold`. Survivors are scored exactly, summed in the
/// original term order so scores are bit-identical to an exhaustive scan.
///
/// Both prunes only drop candidates scoring `<= threshold`. Candidates arrive in
/// ascending row-id order, so such a candidate would lose the `(score, row_id)`
/// tiebreak to every incumbent anyway, and the top-k equals the exhaustive one.
/// [`score_sum_upper_bound_factor`] widens the summed ceilings enough to cover
/// f32 rounding of the differently ordered sums.
pub(super) fn combined_maxscore(
    cursors: &mut [MaterializedTerm],
    dl_prime: impl Fn(u64) -> f32,
    limit: usize,
    require_all_terms: bool,
    scorer: &CombinedFieldsBM25Scorer,
) -> (BinaryHeap<Reverse<RankedDoc>>, MaxscoreStats) {
    let num_terms = cursors.len();
    // Index tiebreak keeps the essential split deterministic.
    let mut order: Vec<usize> = (0..num_terms).collect();
    order.sort_by(|&a, &b| {
        cursors[a]
            .upper_bound()
            .partial_cmp(&cursors[b].upper_bound())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });

    let bound_factor = score_sum_upper_bound_factor(num_terms);

    let mut top: BinaryHeap<Reverse<RankedDoc>> = BinaryHeap::new();
    let mut stats = MaxscoreStats::default();
    // The k-th score; pruning is armed only once the heap is full.
    let mut threshold = f32::NEG_INFINITY;
    // Per-term contributions, fully overwritten for every scored candidate.
    let mut contrib = vec![0.0f32; num_terms];

    loop {
        // Recomputed every step because `threshold` only rises.
        let mut nonessential_upper_bound = 0.0f64;
        let mut split = 0;
        if top.len() >= limit {
            while split < num_terms {
                let bound = f64::from(cursors[order[split]].upper_bound());
                if !score_sum_cannot_compete(
                    0.0,
                    nonessential_upper_bound + bound,
                    threshold,
                    bound_factor,
                    CompetitiveFloorMode::Exclusive,
                ) {
                    break;
                }
                nonessential_upper_bound += bound;
                split += 1;
            }
        }

        // Docs matching only non-essential terms are never discovered: their
        // score cannot beat `threshold`.
        let Some(doc) = order[split..]
            .iter()
            .filter_map(|&term| cursors[term].head())
            .min()
        else {
            break;
        };
        stats.discovered += 1;

        let dl = dl_prime(doc);
        let mut essential_score = 0.0f32;
        let mut missing_term = false;
        for &term in &order[split..] {
            let tf = if cursors[term].head() == Some(doc) {
                let tf = cursors[term].head_tf();
                cursors[term].consume(doc);
                tf
            } else {
                0.0
            };
            missing_term |= tf <= 0.0;
            contrib[term] = cursors[term].idf() * scorer.doc_weight(tf, dl);
            essential_score += contrib[term];
        }
        if require_all_terms && missing_term {
            continue;
        }
        if top.len() >= limit
            && score_sum_cannot_compete(
                essential_score,
                nonessential_upper_bound,
                threshold,
                bound_factor,
                CompetitiveFloorMode::Exclusive,
            )
        {
            stats.pruned += 1;
            continue;
        }

        for &term in &order[..split] {
            let tf = cursors[term].probe(doc);
            missing_term |= tf <= 0.0;
            contrib[term] = cursors[term].idf() * scorer.doc_weight(tf, dl);
        }
        if require_all_terms && missing_term {
            continue;
        }
        stats.scored += 1;

        let score: f32 = contrib.iter().sum();
        // Strict `<`: a tie never displaces an incumbent, matching the prune.
        if top.len() < limit {
            top.push(Reverse(RankedDoc::new(doc, score)));
            if top.len() == limit {
                threshold = top.peek().expect("heap is full").0.score.0;
            }
        } else if top.peek().is_some_and(|worst| worst.0.score.0 < score) {
            top.pop();
            top.push(Reverse(RankedDoc::new(doc, score)));
            threshold = top.peek().expect("heap is full").0.score.0;
        }
    }

    (top, stats)
}

#[cfg(test)]
mod tests {
    use super::super::super::scorer::{K1, idf};
    use super::super::testing::{dl_of, exact_topk_scores, maxscore_scores, term, test_scorer};
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_combined_maxscore_matches_exact_scan_or() {
        let scorer = test_scorer();
        // Skewed idf: a rare high-value term, a medium term, a very common term.
        let build = || {
            [
                term(5.0, &[(1, 3.0), (2, 1.0), (3, 2.0), (17, 1.0)]),
                term(1.5, &[(1, 1.0), (3, 1.0), (5, 1.0), (8, 2.0), (17, 1.0)]),
                term(
                    0.05,
                    &(0..40u64).map(|row_id| (row_id, 1.0)).collect::<Vec<_>>(),
                ),
            ]
        };
        // At `limit = 20` the k-th score ties several candidates' ceilings
        // exactly, where a non-conservative bound would bite.
        for limit in [1usize, 2, 3, 5, 10, 20, 50] {
            let expected = exact_topk_scores(&build(), dl_of, limit, false, &scorer);
            let (actual, _stats) = maxscore_scores(&mut build(), dl_of, limit, false, &scorer);
            assert_eq!(
                actual, expected,
                "OR top-{limit} scores diverged from exact scan"
            );
        }
    }

    /// A tie straddling the top-k cutoff must resolve to the lowest row ids,
    /// ascending. Rows `0, 4, 8, ...` share a `dl_of`, so one term with equal
    /// `tf'` scores them all the same.
    #[test]
    fn test_combined_maxscore_ties_keep_lowest_row_ids() {
        let scorer = test_scorer();
        let tied: Vec<(u64, f32)> = (0..6u64).map(|i| (i * 4, 1.0)).collect();
        let build = || [term(5.0, &tied)];

        for limit in [1usize, 3, 5] {
            let (top, stats) = combined_maxscore(&mut build(), dl_of, limit, false, &scorer);
            let hits: Vec<u64> = top
                .into_sorted_vec()
                .into_iter()
                .map(|Reverse(doc)| doc.row_id.0)
                .collect();
            let expected: Vec<u64> = tied.iter().take(limit).map(|(row_id, _)| *row_id).collect();
            assert_eq!(
                hits, expected,
                "top-{limit} over a full tie must be the lowest row ids, ascending"
            );
            // Every tie was reached, so the result is the tiebreak, not discovery.
            assert_eq!(
                stats.discovered,
                tied.len() as u64,
                "top-{limit} did not discover every tied candidate"
            );
        }
    }

    #[test]
    fn test_combined_maxscore_matches_exact_scan_and() {
        let scorer = test_scorer();
        let build = || {
            [
                term(5.0, &[(1, 3.0), (2, 1.0), (3, 2.0), (17, 1.0)]),
                term(1.5, &[(1, 1.0), (3, 1.0), (5, 1.0), (17, 1.0)]),
            ]
        };
        for limit in [1usize, 2, 3, 10] {
            let expected = exact_topk_scores(&build(), dl_of, limit, true, &scorer);
            let (actual, _stats) = maxscore_scores(&mut build(), dl_of, limit, true, &scorer);
            assert_eq!(
                actual, expected,
                "AND top-{limit} scores diverged from exact scan"
            );
            // AND keeps only docs that have both terms: rows 1, 3, 17.
            assert!(actual.len() <= 3);
        }
    }

    #[test]
    fn test_combined_maxscore_no_limit_is_exact_scan() {
        let scorer = test_scorer();
        let build = || {
            [
                term(5.0, &[(1, 3.0), (2, 1.0), (3, 2.0)]),
                term(0.05, &(0..30u64).map(|r| (r, 1.0)).collect::<Vec<_>>()),
            ]
        };
        // The heap never fills, so nothing may be pruned.
        let expected = exact_topk_scores(&build(), dl_of, usize::MAX, false, &scorer);
        let (actual, stats) = maxscore_scores(&mut build(), dl_of, usize::MAX, false, &scorer);
        assert_eq!(actual, expected);
        assert_eq!(stats.pruned, 0, "no limit must not prune");
        // Union of the two terms is rows 0..30.
        assert_eq!(stats.scored, 30);
    }

    #[test]
    fn test_combined_maxscore_discovery_pruning() {
        let scorer = test_scorer();
        // Once the rare term's docs fill the heap, the common term goes
        // non-essential and its 1000 rows are never discovered.
        let build = || {
            [
                term(6.0, &[(1, 3.0), (2, 3.0), (3, 3.0), (4, 3.0), (5, 3.0)]),
                term(0.05, &(0..1000u64).map(|r| (r, 1.0)).collect::<Vec<_>>()),
            ]
        };
        let union = 1000u64; // common covers every row

        let (actual, stats) = maxscore_scores(&mut build(), dl_of, 3, false, &scorer);
        let expected = exact_topk_scores(&build(), dl_of, 3, false, &scorer);
        assert_eq!(
            actual, expected,
            "pruned run must still match the exact top-k"
        );

        // Discovery examines only a handful of candidates, not the full union.
        assert!(
            stats.discovered <= 20,
            "expected heavy discovery pruning, discovered {} of {union}",
            stats.discovered
        );
    }

    #[test]
    fn test_combined_maxscore_per_candidate_pruning() {
        let scorer = test_scorer();
        // The first term stays essential, but its low-tf row 100 plus the
        // non-essential ceiling cannot beat the k-th score, so it is pruned.
        let build = || {
            [
                term(8.0, &[(1, 10.0), (2, 10.0), (100, 1.0)]),
                term(0.1, &(0..50u64).map(|r| (r, 1.0)).collect::<Vec<_>>()),
            ]
        };

        let (actual, stats) = maxscore_scores(&mut build(), dl_of, 2, false, &scorer);
        let expected = exact_topk_scores(&build(), dl_of, 2, false, &scorer);
        assert_eq!(
            actual, expected,
            "pruned run must still match the exact top-k"
        );
        assert!(
            stats.pruned >= 1,
            "expected the per-candidate upper-bound test to fire, stats={stats:?}"
        );
    }

    /// `doc_weight` can exceed `K1 + 1` in f32, so `idf · (K1 + 1)` is not a
    /// safe ceiling. Here the first row scores exactly `idf · (K1 + 1)`; with
    /// that ceiling the only term would go non-essential and the loop would exit
    /// before the one-ULP-higher second row is scored.
    #[test]
    fn test_combined_maxscore_ceiling_is_conservative() {
        const FIRST: (u32, u32) = (91_135_840, 3_324_876_276);
        const BETTER: (u32, u32) = (1_957_490_862, 2_691_694_489);
        let avg_doc_length = ((u64::from(FIRST.1) + u64::from(BETTER.1)) as f64 / 2.0) as f32;
        let scorer = CombinedFieldsBM25Scorer::new(2, avg_doc_length, HashMap::new());
        let idf = idf(2, 2);
        let dl_prime = |row_id: u64| -> f32 {
            if row_id == 0 {
                FIRST.1 as f32
            } else {
                BETTER.1 as f32
            }
        };
        let build = || [term(idf, &[(0, FIRST.0 as f32), (1, BETTER.0 as f32)])];

        // Guard that the fixture still hits the rounding gap.
        let naive_ceiling = idf * (K1 + 1.0);
        let first_score = idf * scorer.doc_weight(FIRST.0 as f32, FIRST.1 as f32);
        let better_score = idf * scorer.doc_weight(BETTER.0 as f32, BETTER.1 as f32);
        assert_eq!(first_score.to_bits(), naive_ceiling.to_bits());
        assert!(
            better_score > naive_ceiling,
            "fixture no longer exercises the rounding gap: {better_score:e} vs {naive_ceiling:e}"
        );

        let expected = exact_topk_scores(&build(), dl_prime, 1, false, &scorer);
        let (actual, _stats) = maxscore_scores(&mut build(), dl_prime, 1, false, &scorer);
        assert_eq!(
            actual, expected,
            "MAXSCORE stopped before the higher-scoring row"
        );
        assert_eq!(actual, vec![better_score]);
        assert!(term_upper_bound(idf) >= better_score);
    }

    /// `term_upper_bound` must dominate every per-term contribution, and no
    /// `(tf', dl')` may give a non-finite score, including the overflowing blends
    /// extreme boosts can create. The scorer guarantees this itself instead of
    /// relying on query-level boost validation.
    #[test]
    fn test_term_upper_bound_dominates_every_doc_weight() {
        // Deterministic xorshift over the u32 range, plus extreme values.
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let corners = [0.0f32, 1.0, f32::MIN_POSITIVE, f32::MAX, f32::INFINITY];
        // Must end up non-zero, so the sweep keeps covering the rounding gap.
        let mut above_naive_saturation = 0u32;
        for avg_doc_length in [
            0.0f32,
            1.0,
            5.0,
            1e9,
            f32::MAX,
            f32::INFINITY,
            f32::NAN,
            -1.0,
        ] {
            let scorer = CombinedFieldsBM25Scorer::new(1000, avg_doc_length, HashMap::new());
            for doc_freq in [1usize, 2, 7, 999, 1000, 4000] {
                let idf = idf(doc_freq, 1000);
                let bound = term_upper_bound(idf);
                let check = |tf: f32, dl: f32| -> f32 {
                    let weight = scorer.doc_weight(tf, dl);
                    assert!(
                        weight.is_finite() && (0.0..=BM25_DOC_WEIGHT_UPPER_BOUND).contains(&weight),
                        "doc_weight({tf:e}, {dl:e}) = {weight:e} escaped \
                         [0, BM25_DOC_WEIGHT_UPPER_BOUND] (avgdl={avg_doc_length:e})"
                    );
                    let contribution = idf.max(0.0) * weight;
                    assert!(
                        contribution.is_finite() && contribution <= bound,
                        "contribution {contribution:e} exceeds ceiling {bound:e} for \
                         tf={tf:e} dl={dl:e} avgdl={avg_doc_length:e} df={doc_freq}"
                    );
                    weight
                };
                for _ in 0..2_000 {
                    let tf = (next() % (u64::from(u32::MAX) + 1)) as u32 as f32;
                    let dl = (next() % (u64::from(u32::MAX) + 1)) as u32 as f32;
                    if check(tf, dl) > K1 + 1.0 {
                        above_naive_saturation += 1;
                    }
                }
                // Only reachable through extreme boosts, signs or NaNs.
                for tf in corners {
                    for dl in corners {
                        check(tf, dl);
                        check(-tf, dl);
                        check(f32::NAN, dl);
                        check(tf, f32::NAN);
                    }
                }
            }
        }
        assert!(
            above_naive_saturation > 0,
            "sweep never reached a doc_weight above K1 + 1, so it no longer covers \
             the rounding gap the ceiling exists for"
        );
    }
}
