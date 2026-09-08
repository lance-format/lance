// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The term-at-a-time MAXSCORE loop shared by both `combined_fields` term
//! cursors, its per-term score ceiling, and its candidate-work accounting.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::super::scorer::{BM25_DOC_WEIGHT_UPPER_BOUND, CombinedFieldsBM25Scorer};
use super::super::wand::{
    CompetitiveFloorMode, score_sum_cannot_compete, score_sum_upper_bound_factor,
};
use super::cursor::TermCursor;
use super::search::RankedDoc;

/// The constant per-term BM25F score ceiling, clamped to 0 when the term cannot
/// raise a score (`idf <= 0`).
///
/// Lucene's `CombinedFieldQuery::getMaxScore` uses `idf · (K1 + 1)` because BM25
/// saturates the `tf'` factor at `K1 + 1`. That product is not an upper bound
/// in f32: the evaluated weight can land above it, and a ceiling one ULP short
/// is enough for MAXSCORE to drop a document that outscores the running k-th (a
/// one-term `limit = 1` query suffices; see
/// `test_combined_maxscore_ceiling_is_conservative`). Scaling by
/// [`BM25_DOC_WEIGHT_UPPER_BOUND`] instead keeps
/// `term_upper_bound(idf) >= idf · doc_weight(tf', dl')` for every `tf'` / `dl'`,
/// because `doc_weight` clamps to that ceiling and `x -> fl(idf · x)` is
/// monotone for `idf >= 0`.
#[inline]
pub(super) fn term_upper_bound(idf: f32) -> f32 {
    if idf > 0.0 {
        idf * BM25_DOC_WEIGHT_UPPER_BOUND
    } else {
        0.0
    }
}

/// Candidate-work accounting for one MAXSCORE run: the scoring-side work the
/// essential/non-essential split saves.
///
/// Per-candidate bookkeeping does not belong on the production
/// `MetricsCollector`, so this never leaves the crate on a normal build: the
/// `maxscore` module is private and only `cfg(test)` or the `test-scan-stats`
/// feature re-exports it, alongside `combined_fields_search_with_stats`. The
/// unit tests below are what keep the pruning honest: a gate that never fired
/// would leave every score correct and silently do nothing.
#[derive(Default, Debug, Clone, Copy)]
pub struct MaxscoreStats {
    /// Candidates pulled from the essential terms' cursors.
    pub discovered: u64,
    /// Discovered candidates skipped by the upper-bound test (no length lookup,
    /// no non-essential probe, no full score).
    pub pruned: u64,
    /// Candidates fully scored.
    pub scored: u64,
}

/// Term-at-a-time MAXSCORE over the cross-field combined postings.
///
/// Terms are ordered ascending by their constant ceiling `upper_bound`. Once the
/// heap holds `limit` docs, the leading prefix whose cumulative ceiling cannot
/// beat the k-th score (`threshold`) becomes non-essential: candidate discovery
/// walks the essential terms' cursors alone, and a candidate is skipped whole
/// (no length lookup, no probe) when the essential terms it matches plus every
/// non-essential ceiling still cannot beat `threshold`. `threshold` only rises,
/// so a term never returns to essential once it crosses over.
///
/// Every surviving candidate is scored exactly, summing `idf'(t) ·
/// doc_weight(tf', dl')` over the terms in their original order, with `dl_prime`
/// supplying the blended length. Absent terms add `doc_weight(0, ·) == 0`, so the
/// score is bit-identical to the exact scan, which skips them.
///
/// Both prunes only ever drop a candidate whose exact `score` is provably
/// `<= threshold`, i.e. one the collector would have rejected anyway, so the top-k
/// is the exact scan's top-k. That rests on two f32 facts, neither of which holds
/// for the textbook `idf · (K1 + 1)` ceiling: the per-term ceiling dominates every
/// per-term contribution ([`term_upper_bound`] over a clamped `doc_weight`), and
/// summing those ceilings is widened enough to dominate the differently-ordered f32
/// summation that produces `score` ([`score_sum_upper_bound_factor`]).
///
/// Generic over [`TermCursor`], so the lazy and eager paths share one algorithm.
pub(super) fn combined_maxscore<C: TermCursor>(
    cursors: &mut [C],
    dl_prime: impl Fn(u64) -> f32,
    limit: usize,
    require_all_terms: bool,
    scorer: &CombinedFieldsBM25Scorer,
) -> (BinaryHeap<Reverse<RankedDoc>>, MaxscoreStats) {
    let num_terms = cursors.len();
    // Term indices ordered ascending by ceiling; ties broken by index so the
    // essential/non-essential split is deterministic.
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
    // The current k-th score; pruning is armed only once the heap is full.
    let mut threshold = f32::NEG_INFINITY;
    // Per-term score contributions, reused across candidates. Every scored
    // candidate overwrites all slots before it is summed.
    let mut contrib = vec![0.0f32; num_terms];

    loop {
        // Split `order` into non-essential (a leading prefix whose widened
        // cumulative ceiling <= threshold) and essential. Recomputed each step
        // because `threshold` may have risen.
        let mut nonessential_upper_bound = 0.0f64;
        let split = if top.len() < limit {
            0
        } else {
            let mut cumulative = 0.0f64;
            let mut boundary = 0;
            while boundary < num_terms {
                let bound = f64::from(cursors[order[boundary]].upper_bound());
                if score_sum_cannot_compete(
                    0.0,
                    cumulative + bound,
                    threshold,
                    bound_factor,
                    CompetitiveFloorMode::Exclusive,
                ) {
                    cumulative += bound;
                    boundary += 1;
                } else {
                    break;
                }
            }
            nonessential_upper_bound = cumulative;
            boundary
        };

        // Next candidate = smallest row id at any essential term's cursor. Docs
        // that match only non-essential terms are never discovered: their score is
        // at most the widened `nonessential_upper_bound`, hence `<= threshold`,
        // which the score-only strict-`<` collector could not admit anyway.
        let mut doc = u64::MAX;
        for &term in &order[split..] {
            if let Some(row_id) = cursors[term].head() {
                doc = doc.min(row_id);
            }
        }
        if doc == u64::MAX {
            break; // essential terms exhausted (or all terms non-essential)
        }
        stats.discovered += 1;

        let dl = dl_prime(doc);
        // Exact contribution of the essential terms, consuming each cursor that
        // sits on `doc`. The widened `essential_score + nonessential_upper_bound`
        // then dominates the full `score`, so the prune below is safe (see this
        // function's doc comment).
        //
        // The prune stays `Exclusive` (reject on `upper_bound <= threshold`) even
        // though the collector now orders on `(score DESC, row_id ASC)`. A
        // candidate that merely ties the k-th score cannot belong in the top-k
        // here: `doc` only ever increases, so every candidate reaching this point
        // has a higher row id than every incumbent, and an incumbent tied at
        // `threshold` therefore wins the tiebreak. Rejecting on the tie is what the
        // collector below does too, so the prune drops nothing the collector would
        // have kept. See `test_combined_maxscore_ties_keep_lowest_row_ids`.
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

        // Probe the non-essential terms on demand to complete the exact score.
        // `doc` increases monotonically, so each probe seeks forward.
        for &term in &order[..split] {
            let tf = cursors[term].probe(doc);
            missing_term |= tf <= 0.0;
            contrib[term] = cursors[term].idf() * scorer.doc_weight(tf, dl);
        }
        if require_all_terms && missing_term {
            continue;
        }
        stats.scored += 1;

        // Sum in original term order so the score is bit-identical to the exact
        // scan.
        let score: f32 = contrib.iter().sum();

        // The replacement test stays score-only and strict, which is what
        // `CompetitiveFloorMode::Exclusive` prunes against: a candidate that only
        // ties the k-th score never displaces an incumbent, so pruning one is
        // sound. [`RankedDoc`] does not weaken that. `doc` increases
        // monotonically, so a tied candidate always carries the higher row id and
        // would lose the tiebreak too; ordering the heap on `(score, row_id)` only
        // fixes which of the retained ties comes out first.
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
        // Every top-k depth must reproduce the exact scan's top-k scores, so the
        // MAXSCORE pruning provably does not change the result. `limit = 20` is
        // the depth where this fixture's k-th score ties several candidates'
        // ceilings exactly, which is where a non-conservative bound would bite.
        for limit in [1usize, 2, 3, 5, 10, 20, 50] {
            let expected = exact_topk_scores(&build(), dl_of, limit, false, &scorer);
            let (actual, _stats) = maxscore_scores(&mut build(), dl_of, limit, false, &scorer);
            assert_eq!(
                actual, expected,
                "OR top-{limit} scores diverged from exact scan"
            );
        }
    }

    /// A tie group straddling the top-k cutoff must resolve to the lowest row
    /// ids, in ascending order.
    ///
    /// This is the interaction the candidate prune has to survive: the prune and
    /// the collector both reject a candidate that only ties `threshold`, and that
    /// is correct precisely because discovery is ascending, so a tied newcomer
    /// always loses the row-id tiebreak to the incumbents.
    ///
    /// `dl_of` depends on `row_id % 4`, so row ids `0, 4, 8, ...` share a blended
    /// length; one term with an equal `tf'` at each then scores them all
    /// identically, making the whole result one tie.
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
            // Every candidate was reached, so the assertion above is about which
            // ties were kept rather than about candidates never being discovered.
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
        // usize::MAX limit: the heap never fills, so nothing is pruned and every
        // matching doc is scored, exactly like the merged scan.
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
        // A rare high-idf term whose docs sit early, and a common low-idf term
        // over a huge posting list. With a small k the common term goes
        // non-essential once the heap fills with the rare docs, so its ~1000
        // rows are never discovered; only the essential rare term drives
        // discovery.
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
        // Two essential terms. `a` stays essential (large ceiling) but has a
        // late doc (row 100) whose tf is low, so its exact contribution plus the
        // non-essential ceiling cannot beat the k-th score: that discovered
        // candidate is skipped by the per-candidate upper-bound test before any
        // non-essential probe.
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

    /// The per-term ceiling must be conservatively rounded, or MAXSCORE can stop
    /// before a strictly higher-scoring document.
    ///
    /// `doc_weight` saturates at `K1 + 1` in exact arithmetic but can exceed it
    /// in f32, so `idf · (K1 + 1)` is not an upper bound. One term and `limit = 1` is enough to lose a document: the
    /// first row's score lands exactly on `idf · (K1 + 1)`, so it sets
    /// `threshold == ceiling`; the split then makes the only term non-essential,
    /// no essential cursor is left to discover a candidate, and the loop exits
    /// before the second, one-ULP-higher row is ever scored.
    #[test]
    fn test_combined_maxscore_ceiling_is_conservative() {
        // Valid u32 frequencies and lengths; `better` is exactly one ULP above
        // the un-rounded ceiling `idf · (K1 + 1)`.
        const FIRST: (u32, u32) = (91_135_840, 3_324_876_276);
        const BETTER: (u32, u32) = (1_957_490_862, 2_691_694_489);
        let avg_doc_length = ((u64::from(FIRST.1) + u64::from(BETTER.1)) as f64 / 2.0) as f32;
        let scorer = CombinedFieldsBM25Scorer::new(2, avg_doc_length, HashMap::new());
        let idf = idf(2, 2);
        // Row 0 carries `FIRST` (so it is scored first and sets the threshold),
        // row 1 carries `BETTER`.
        let dl_prime = |row_id: u64| -> f32 {
            if row_id == 0 {
                FIRST.1 as f32
            } else {
                BETTER.1 as f32
            }
        };
        let build = || [term(idf, &[(0, FIRST.0 as f32), (1, BETTER.0 as f32)])];

        // The fixture is only meaningful while the naive ceiling sits below the
        // achievable score and the first row lands exactly on it.
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
        // The mechanism: the ceiling dominates every achievable contribution.
        assert!(term_upper_bound(idf) >= better_score);
    }

    /// `term_upper_bound` must dominate every per-term contribution it is asked
    /// to bound, and no `(tf', dl')` may produce a non-finite score, not even the
    /// overflowing blends that extreme per-column boosts can create. The MAXSCORE
    /// prune is only conservative if both hold, so the scorer establishes them
    /// itself rather than relying on the query-level validation of the boosts.
    #[test]
    fn test_term_upper_bound_dominates_every_doc_weight() {
        // A deterministic xorshift walk over the whole u32 range of `tf'`/`dl'`,
        // plus the extremes and the values that overflow the blend.
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let corners = [0.0f32, 1.0, f32::MIN_POSITIVE, f32::MAX, f32::INFINITY];
        // How often a realistic `(tf', dl')` pair pushes `doc_weight` past the
        // textbook `K1 + 1` saturation point. Asserted non-zero below so the sweep
        // cannot silently stop exercising the rounding boundary that makes the
        // naive `idf · (K1 + 1)` ceiling unsafe.
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
                // Frequencies and lengths as they actually arrive: `u32` counts
                // widened to f32.
                for _ in 0..2_000 {
                    let tf = (next() % (u64::from(u32::MAX) + 1)) as u32 as f32;
                    let dl = (next() % (u64::from(u32::MAX) + 1)) as u32 as f32;
                    if check(tf, dl) > K1 + 1.0 {
                        above_naive_saturation += 1;
                    }
                }
                // Values only reachable once a boost blows the blend up, plus the
                // signs and NaNs no validation upstream should be trusted to stop.
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
