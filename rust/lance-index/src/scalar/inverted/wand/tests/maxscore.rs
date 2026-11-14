// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

#[rstest]
#[case::all_essential(false)]
#[case::single_essential(true)]
fn maxscore_publishes_query_order_score_bits(#[case] single_essential: bool) {
    let contributions = [0.002_972_301_6_f32, 0.001_982_450_7_f32, 0.001_882_293_f32];
    // Each bound covers its term's exact contribution. Sorted by bound,
    // the two smallest fit below a floor just under the exact score while
    // all three do not, which forces the single-essential MAXSCORE path.
    let bounds = [0.002_99_f32, 0.001_99_f32, 0.003_f32];
    let postings = contributions
        .into_iter()
        .zip(bounds)
        .enumerate()
        .map(|(position, (query_weight, max_score))| {
            PostingIterator::with_query_weight(
                format!("t{position}"),
                position as u32,
                position as u32,
                query_weight,
                generate_posting_list(vec![0], max_score, None, true),
                1,
            )
        })
        .collect::<Vec<_>>();
    let mut docs = DocSet::default();
    docs.append(0, 1);
    let shared_floor = Arc::new(AtomicU32::new(0.0_f32.to_bits()));
    let scored = Arc::new(AtomicUsize::new(0));
    let mut wand = Wand::new(
        Operator::Or,
        postings.into_iter(),
        &docs,
        CountingScorer {
            scored: scored.clone(),
        },
    )
    .with_shared_threshold(shared_floor.clone());
    let query_order_score = contributions
        .into_iter()
        .fold(0.0_f32, |score, contribution| score + contribution);
    if single_essential {
        wand.threshold = f32::from_bits(query_order_score.to_bits() - 1);
    }

    let hits = wand
        .maxscore_search(
            &FtsSearchParams::default().with_limit(Some(1)),
            &NoOpMetricsCollector,
        )
        .unwrap();

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].document, 0);
    let mut terms = hits[0]
        .freqs
        .iter()
        .map(|(term_index, freq)| {
            assert_eq!(*freq, 1);
            *term_index
        })
        .collect::<Vec<_>>();
    terms.sort_unstable();
    assert_eq!(terms, vec![0, 1, 2]);
    assert_eq!(shared_floor.load(Ordering::Relaxed), 0x3be0_094c);
    assert_eq!(scored.load(Ordering::Relaxed), contributions.len());
    assert_eq!(wand.maxscore_single_essential_windows > 0, single_essential);
    assert_eq!(wand.maxscore_general_windows > 0, !single_essential);
}

#[test]
fn maxscore_skips_essential_blocks_below_the_floor() {
    // Lucene ImpactsEnum.setMinCompetitiveScore: a single-essential
    // block whose impact bound plus the optional remainder cannot
    // enter the heap is skipped without decoding.
    let block = crate::scalar::inverted::LEGACY_BLOCK_SIZE as u32;
    let hot: Vec<u32> = (0..block).collect();
    let cold: Vec<u32> = (block..2 * block).collect();
    let ess_docs: Vec<u32> = hot.iter().copied().chain(cold.iter().copied()).collect();
    let ess_freqs: Vec<u32> = std::iter::repeat_n(200, block as usize)
        .chain(std::iter::repeat_n(1, block as usize))
        .collect();
    let opt_docs = ess_docs.clone();
    let essential = PostingIterator::with_query_weight(
        "ess".to_owned(),
        0,
        0,
        1.0,
        generate_impact_posting_list_with_freqs_and_block_size(
            ess_docs,
            ess_freqs,
            vec![1; 2 * block as usize],
            crate::scalar::inverted::LEGACY_BLOCK_SIZE,
        ),
        2 * block as usize,
    );
    let optional = PostingIterator::with_query_weight(
        "opt".to_owned(),
        1,
        1,
        0.01,
        generate_impact_posting_list_with_freqs_and_block_size(
            opt_docs,
            vec![1; 2 * block as usize],
            vec![1; 2 * block as usize],
            crate::scalar::inverted::LEGACY_BLOCK_SIZE,
        ),
        2 * block as usize,
    );
    let mut docs = DocSet::default();
    for doc in 0..2 * block {
        docs.append(doc.into(), 1);
    }
    let shared_floor = Arc::new(AtomicU32::new(5.0_f32.to_bits()));
    let mut wand = Wand::new(
        Operator::Or,
        [essential, optional].into_iter(),
        &docs,
        InverseDocLengthScorer,
    )
    .with_shared_threshold(shared_floor);
    let hits = wand
        .maxscore_search(
            &FtsSearchParams::default().with_limit(Some(10)),
            &NoOpMetricsCollector,
        )
        .unwrap();
    assert!(!hits.is_empty());
    for hit in &hits {
        assert!(
            hit.posting_doc_id < u64::from(block),
            "cold-block doc {} should not enter the heap",
            hit.posting_doc_id
        );
    }
    // Only the single-essential path is pinned here. This fixture cannot
    // prove a block dead -- with `InverseDocLengthScorer` every block looks
    // unbounded -- so asserting on the skip counter as well would make the
    // assertion a tautology. `maxscore_skips_a_dead_essential_block`
    // covers the skip itself.
    assert!(
        wand.maxscore_single_essential_windows > 0,
        "the hot block must take the single-essential path"
    );
}

#[test]
fn maxscore_block_skip_stops_at_the_end_of_the_span() {
    // `skip_current_block()` jumps to the *next* block, so when the block
    // extends past `upto` it would also discard the documents after `upto`.
    // Those belong to a later window: reaching them again through another
    // essential would score them without this clause's contribution, since
    // the overall score accumulates across clauses. The completion must
    // therefore stop at `upto` rather than running to the end of the block.
    let total = 1024u32;
    let mut docs = DocSet::default();
    for doc in 0..total {
        docs.append(doc.into(), 1);
    }
    let ids: Vec<u32> = (0..total).collect();
    let freqs = vec![1u32; total as usize];
    let doc_lengths = vec![1u32; total as usize];
    let mut wand = Wand::new(
        Operator::Or,
        [PostingIterator::with_query_weight(
            "ess".to_owned(),
            1,
            1,
            1.0,
            generate_impact_posting_list_with_freqs_and_block_size(
                ids.clone(),
                freqs.clone(),
                doc_lengths.clone(),
                crate::scalar::inverted::LEGACY_BLOCK_SIZE,
            ),
            total as usize,
        )]
        .into_iter(),
        &docs,
        CountingBm25ShapeScorer {
            scored: Arc::new(AtomicUsize::new(0)),
        },
    );
    // High enough that every block of this clause is dead, so the skip is
    // taken for whole blocks; `upto` sits in the middle of the second one.
    wand.threshold = 20.0;
    let mut clauses = vec![MaxScoreClause {
        posting: Box::new(PostingIterator::with_query_weight(
            "ess".to_owned(),
            1,
            1,
            1.0,
            generate_impact_posting_list_with_freqs_and_block_size(
                ids,
                freqs,
                doc_lengths,
                crate::scalar::inverted::LEGACY_BLOCK_SIZE,
            ),
            total as usize,
        )),
        query_rank: 1,
        bound: 1.0,
        prefix_bound: 1.0,
    }];
    let ess_term = clauses[0].posting.term_index();
    let mut scratch = MaxScoreSoaScratch::new();
    let mut candidates = TopKCollector::new(10, 10);
    let mut chunk: Vec<(u64, u32)> = Vec::new();
    let mut comparisons = 0usize;
    // 200 lands inside the second block (128..=255), whose end is 255.
    let upto = 200u64;
    wand.score_single_essential_upto(
        &mut clauses,
        0,
        0,
        upto,
        1,
        ess_term,
        1.0,
        1.0,
        1,
        0.0,
        1.0,
        None,
        &mut chunk,
        &mut scratch,
        &mut candidates,
        &mut comparisons,
        1.0,
    )
    .unwrap();
    let after = clauses[0].posting.doc().map(|doc| doc.doc_id());
    assert!(
        after.is_none_or(|doc| doc <= upto + 1),
        "the completion must stop at the end of the span (upto={upto}); \
             landing on {after:?} means the skip ran past `upto` to the next block"
    );
}

#[test]
fn maxscore_skips_a_dead_essential_block() {
    // The whole-block skip needs a finite per-block bound. An impact
    // posting derives that bound from the scorer, and `InverseDocLengthScorer`
    // has no finite `doc_weight_upper_bound`, so the bound falls back to
    // infinity and no block can be proven dead. This fixture therefore bakes
    // the block maxima into a plain compressed list: block 0 competes and
    // fills the heap, block 1 cannot reach the resulting floor.
    let block = crate::scalar::inverted::LEGACY_BLOCK_SIZE as u32;
    let total = 2 * block;
    let doc_lengths = vec![1u32; total as usize];
    let ess_freqs: Vec<u32> = std::iter::repeat_n(200, block as usize)
        .chain(std::iter::repeat_n(1, block as usize))
        .collect();
    let mut docs = DocSet::default();
    for doc in 0..total {
        docs.append(doc.into(), 1);
    }
    let essential = |freqs: Vec<u32>| {
        PostingIterator::with_query_weight(
            "ess".to_owned(),
            0,
            0,
            1.0,
            generate_impact_posting_list_with_freqs_and_block_size(
                (0..total).collect(),
                freqs,
                doc_lengths.clone(),
                crate::scalar::inverted::LEGACY_BLOCK_SIZE,
            ),
            total as usize,
        )
    };
    let optional = || {
        PostingIterator::with_query_weight(
            "opt".to_owned(),
            1,
            1,
            0.01,
            generate_impact_posting_list_with_freqs_and_block_size(
                (0..total).collect(),
                vec![1; total as usize],
                doc_lengths.clone(),
                crate::scalar::inverted::LEGACY_BLOCK_SIZE,
            ),
            total as usize,
        )
    };
    let mut wand = Wand::new(
        Operator::Or,
        [essential(ess_freqs.clone()), optional()].into_iter(),
        &docs,
        CountingBm25ShapeScorer {
            scored: Arc::new(AtomicUsize::new(0)),
        },
    );
    // Above what the cold block plus the optional can reach, below what the
    // hot block reaches.
    wand.threshold = 20.0;
    let mut clauses = vec![
        MaxScoreClause {
            posting: Box::new(optional()),
            query_rank: 0,
            bound: 0.01,
            prefix_bound: 0.01,
        },
        MaxScoreClause {
            posting: Box::new(essential(ess_freqs)),
            query_rank: 1,
            bound: 200.0,
            prefix_bound: 200.01,
        },
    ];
    let ess_term = clauses[1].posting.term_index();
    let mut scratch = MaxScoreSoaScratch::new();
    let mut candidates = TopKCollector::new(10, 10);
    let mut chunk: Vec<(u64, u32)> = Vec::new();
    let mut comparisons = 0usize;
    wand.score_single_essential_upto(
        &mut clauses,
        1,
        1,
        u64::from(total - 1),
        1,
        ess_term,
        1.0,
        200.0,
        2,
        0.01,
        1.0,
        None,
        &mut chunk,
        &mut scratch,
        &mut candidates,
        &mut comparisons,
        1.0,
    )
    .unwrap();
    assert!(
        wand.maxscore_essential_blocks_skipped > 0,
        "the cold block, which cannot reach the floor, must be skipped whole"
    );
}

#[test]
fn maxscore_top2_gap_completes_with_an_optional_clause() {
    // The top2-gap shortcut argues that no other essential contributes
    // inside the isolated span. That only says something when a
    // non-essential remainder is left to add afterwards, yet every existing
    // top2-gap case reaches the shortcut with `non_essential` empty. Drive it
    // with a live optional: one essential at doc 0, one far past the gap
    // threshold, and a low-weight optional kept non-essential by a shared
    // floor between their bounds.
    let far = 3000u32;
    let mut docs = DocSet::default();
    for doc in 0..=far {
        docs.append(doc.into(), 1);
    }
    let posting = |term: &str, token_id: u32, weight: f32, ids: Vec<u32>, freqs: Vec<u32>| {
        let len = ids.len();
        PostingIterator::with_query_weight(
            term.to_owned(),
            token_id,
            token_id,
            weight,
            generate_posting_list_with_freqs(ids, freqs, weight, None, true),
            len,
        )
    };
    // `ess_a` owns the two winning documents, `ess_b` owns one a full gap
    // away that cannot reach them, and the optional is live on all of them
    // but kept non-essential by the shared floor. So the shortcut has to
    // add the optional's contribution *after* completing the span, and the
    // published k-th score pins that the addition really happened.
    let optional = posting("opt", 0, 0.01, vec![0, 1, 2, far], vec![1, 1, 1, 1]);
    let ess_a = posting("ess_a", 1, 1.0, vec![0, 1, 2], vec![200, 150, 100]);
    let ess_b = posting("ess_b", 2, 1.0, vec![far], vec![50]);
    let shared_floor = Arc::new(AtomicU32::new(1.0_f32.to_bits()));
    let mut wand = Wand::new(
        Operator::Or,
        [optional, ess_a, ess_b].into_iter(),
        &docs,
        CountingBm25ShapeScorer {
            scored: Arc::new(AtomicUsize::new(0)),
        },
    )
    .with_shared_threshold(shared_floor.clone());
    let hits = wand
        .maxscore_search(
            &FtsSearchParams::default().with_limit(Some(2)),
            &NoOpMetricsCollector,
        )
        .unwrap();
    assert!(
        wand.maxscore_top2_gap_windows > 0,
        "the top2-gap shortcut must run with a non-empty optional remainder"
    );
    let got = hits
        .iter()
        .map(|hit| hit.posting_doc_id)
        .collect::<Vec<_>>();
    assert_eq!(
        got.len(),
        2,
        "the two heaviest documents of `ess_a` must fill the heap, got {got:?}"
    );
    for doc in [0u64, 1] {
        assert!(
            got.contains(&doc),
            "doc {doc} must be in the top 2, got {got:?}"
        );
    }
    // `CountingBm25ShapeScorer` scores every document with norm 1.2, and the
    // canonical fold walks the clauses in query order: optional first, then
    // the essential that owns the document.
    let opt_addend = 0.01 * bm25_doc_weight_with_norm(1, 1.2);
    let kth = opt_addend + bm25_doc_weight_with_norm(150, 1.2);
    assert_eq!(
        shared_floor.load(Ordering::Relaxed),
        kth.to_bits(),
        "the published floor must be the second best score, optional included \
             (expected {kth}, got {})",
        f32::from_bits(shared_floor.load(Ordering::Relaxed))
    );
}

/// The completion asks about visibility before it scores rather than
/// after, so a hidden document must never reach the scorer. The filter is
/// result-preserving -- a hidden document is dropped at insert time either
/// way -- so the only observable difference is how much scoring happened.
/// A limit that keeps the heap from filling leaves the threshold at zero,
/// which makes every surviving document score exactly once.
#[test]
fn maxscore_completion_filters_invisible_documents_before_scoring() {
    struct HiddenDocs<'a> {
        inner: &'a DocSet,
        hidden: Vec<bool>,
    }

    impl WandDocuments for HiddenDocs<'_> {
        type Candidate = u64;

        fn len(&self) -> usize {
            self.inner.len()
        }

        fn scoring_norms(&self) -> Option<&[u8]> {
            self.inner.scoring_norms()
        }

        fn scoring_num_tokens(&self, doc_id: u32) -> u32 {
            self.inner.scoring_num_tokens(doc_id)
        }

        fn doc_length(&self, doc: &DocInfo) -> u32 {
            self.inner.scoring_num_tokens(doc.doc_id() as u32)
        }

        fn document_key(&self, doc: &DocInfo) -> Option<u64> {
            let id = doc.doc_id();
            if self.hidden[id as usize] {
                None
            } else {
                Some(id)
            }
        }

        fn document_key_for_doc_id(&self, doc_id: u32) -> Option<u64> {
            if self.hidden[doc_id as usize] {
                None
            } else {
                Some(u64::from(doc_id))
            }
        }

        fn candidate_from_key(&self, key: u64) -> Self::Candidate {
            key
        }

        fn flat_documents(&self) -> Option<FlatDocuments<'_>> {
            None
        }

        fn flat_doc_length(&self, doc_id: u64, key: u64, compressed: bool) -> u32 {
            if compressed {
                self.inner.scoring_num_tokens(doc_id as u32)
            } else {
                self.inner.scoring_num_tokens(key as u32)
            }
        }
    }

    let total = 4096u32;
    let mut docs = DocSet::default();
    for doc in 0..total {
        docs.append(u64::from(doc), 1);
    }
    let run = |hidden: Vec<bool>| {
        let scored = Arc::new(AtomicUsize::new(0));
        let documents = HiddenDocs {
            inner: &docs,
            hidden,
        };
        let ids = (0..total).collect::<Vec<_>>();
        let len = ids.len();
        let essential = PostingIterator::with_query_weight(
            "ess".to_owned(),
            0,
            0,
            1.0,
            generate_impact_posting_list_with_freqs_and_block_size(
                ids,
                vec![2u32; len],
                vec![1u32; len],
                MAX_POSTING_BLOCK_SIZE,
            ),
            len,
        );
        let mut wand = Wand::new(
            Operator::Or,
            [essential].into_iter(),
            &documents,
            CountingBm25ShapeScorer {
                scored: scored.clone(),
            },
        );
        let hits = wand
            .maxscore_search(
                &FtsSearchParams::default().with_limit(Some(total as usize)),
                &NoOpMetricsCollector,
            )
            .unwrap();
        (hits, scored.load(Ordering::Relaxed))
    };

    let (all_hits, all_scored) = run(vec![false; total as usize]);
    let (half_hits, half_scored) = run((0..total).map(|doc| doc % 2 == 0).collect::<Vec<_>>());

    assert_eq!(all_hits.len(), total as usize);
    assert_eq!(half_hits.len(), total as usize / 2);
    // Hiding N documents has to remove at least N scoring calls; fewer
    // means the hidden documents were scored and then thrown away.
    assert!(
        all_scored - half_scored >= total as usize / 2,
        "hiding {} documents removed only {} of {} scoring calls",
        total as usize / 2,
        all_scored - half_scored,
        all_scored
    );
    for hit in &half_hits {
        assert_eq!(
            hit.posting_doc_id % 2,
            1,
            "hidden document {} was returned",
            hit.posting_doc_id
        );
    }
}

/// GUARD 2: `inner_min` can start a window ahead of cursors that were never
/// consumed, because a window that is skipped wholesale advances
/// `window_min` without touching them and `shallow_next` only moves
/// `block_idx`. The stretch the top2-gap shortcut would complete then sits
/// below the window floor, and taking it would either do nothing or move
/// `inner_min` backwards.
#[test]
fn maxscore_top2_gap_declines_a_span_below_the_window_floor() {
    let total = 8192u32;
    let hot_from = 4096u32;
    let stale_b = 2048u32;
    let mut docs = DocSet::default();
    for doc in 0..total {
        docs.append(u64::from(doc), 1);
    }
    let freqs = |ids: &[u32]| {
        ids.iter()
            .map(|doc| if *doc < hot_from { 1 } else { 200 })
            .collect::<Vec<_>>()
    };
    let lengths = |ids: &[u32]| vec![1u32; ids.len()];
    let build = |term: &str, rank: u32, weight: f32, ids: Vec<u32>| {
        let len = ids.len();
        let list = generate_impact_posting_list_with_freqs_and_block_size(
            ids.clone(),
            freqs(&ids),
            lengths(&ids),
            MAX_POSTING_BLOCK_SIZE,
        );
        PostingIterator::with_query_weight(term.to_owned(), rank, rank, weight, list, len)
    };
    let all = (0..total).collect::<Vec<_>>();
    // `ess_a` and `ess_b` are equally weighted; `ess_b` starts a full
    // top2-gap away so the shortcut is a candidate everywhere, and `opt`
    // is too light to ever be essential.
    let ess_a = build("ess_a", 0, 1.0, all.clone());
    let ess_b = build("ess_b", 1, 1.0, (stale_b..total).collect::<Vec<_>>());
    let opt = build("opt", 2, 0.01, all);
    // Above what two cold blocks can reach, below what one hot block
    // reaches: the first windows are skipped wholesale, the hot one is not.
    let shared_floor = Arc::new(AtomicU32::new(2.1_f32.to_bits()));
    let mut wand = Wand::new(
        Operator::Or,
        [ess_a, ess_b, opt].into_iter(),
        &docs,
        CountingBm25ShapeScorer {
            scored: Arc::new(AtomicUsize::new(0)),
        },
    )
    .with_shared_threshold(shared_floor);
    wand.maxscore_search(
        &FtsSearchParams::default().with_limit(Some(10)),
        &NoOpMetricsCollector,
    )
    .unwrap();
    assert!(
        wand.maxscore_top2_gap_empty_span > 0,
        "the shortcut must decline a span that starts below the window floor"
    );
}

#[test]
fn maxscore_does_not_skip_past_the_current_window() {
    // Optional starts in block 1. Window 0's optional remainder is 0, so
    // the essential's next block cannot compete *in this window*. Skip
    // must leave the cursor on that block for window 1, where the
    // optional is live.
    let block = crate::scalar::inverted::LEGACY_BLOCK_SIZE as u32;
    let ess_docs: Vec<u32> = (0..2 * block).collect();
    let opt_docs: Vec<u32> = (block..2 * block).collect();
    let essential = PostingIterator::with_query_weight(
        "ess".to_owned(),
        0,
        0,
        0.4,
        generate_impact_posting_list_with_freqs_and_block_size(
            ess_docs,
            vec![1; 2 * block as usize],
            vec![1; 2 * block as usize],
            crate::scalar::inverted::LEGACY_BLOCK_SIZE,
        ),
        2 * block as usize,
    );
    let optional = PostingIterator::with_query_weight(
        "opt".to_owned(),
        1,
        1,
        0.3,
        generate_impact_posting_list_with_freqs_and_block_size(
            opt_docs,
            vec![1; block as usize],
            vec![1; block as usize],
            crate::scalar::inverted::LEGACY_BLOCK_SIZE,
        ),
        2 * block as usize,
    );
    let mut docs = DocSet::default();
    for doc in 0..2 * block {
        docs.append(doc.into(), 1);
    }
    let shared_floor = Arc::new(AtomicU32::new(0.5_f32.to_bits()));
    let mut wand = Wand::new(
        Operator::Or,
        [essential, optional].into_iter(),
        &docs,
        InverseDocLengthScorer,
    )
    .with_shared_threshold(shared_floor);
    let hits = wand
        .maxscore_search(
            &FtsSearchParams::default().with_limit(Some(10)),
            &NoOpMetricsCollector,
        )
        .unwrap();
    assert_eq!(hits.len(), 10);
    assert!(
        hits.iter()
            .all(|hit| hit.posting_doc_id >= u64::from(block)),
        "only the optional-overlap block can beat the exclusive floor"
    );
}

#[test]
fn merge_optional_freqs_matches_per_doc_next() {
    let posting_docs = vec![1u32, 2, 4, 7, 11, 20, 21, 40];
    let freqs = vec![3u32, 1, 5, 2, 9, 4, 1, 8];
    let targets = [0u64, 2, 5, 7, 10, 11, 20, 21, 30, 40];
    let build = || {
        PostingIterator::new(
            "hotels".to_owned(),
            0,
            0,
            generate_posting_list_with_freqs(posting_docs.clone(), freqs.clone(), 1.0, None, true),
            64,
        )
    };
    let mut merged = build();
    let mut out = vec![0u32; targets.len()];
    merged.merge_optional_freqs(&targets, &mut out);

    let mut probe = build();
    let mut expected = vec![0u32; targets.len()];
    for (i, &doc) in targets.iter().enumerate() {
        if probe.doc().is_some_and(|cur| cur.doc_id() < doc) {
            probe.next(doc);
        }
        if let Some(cur) = probe.doc()
            && cur.doc_id() == doc
        {
            expected[i] = cur.frequency();
        }
    }
    assert_eq!(out, expected);
    assert_eq!(out, vec![0, 1, 0, 2, 0, 9, 4, 1, 0, 8]);
}

#[test]
fn maxscore_top2_gap_completes_isolated_essentials() {
    // Two essentials 3000 docs apart: Lucene scores each stretch as a
    // single-essential window. The optional sits on both ends so the
    // floor can demote it without dropping either essential's hits.
    let left: Vec<u32> = (0..32).collect();
    let right: Vec<u32> = (3000..3032).collect();
    let optional: Vec<u32> = left.iter().copied().chain(right.iter().copied()).collect();
    let build = |token: &str, rank: u32, docs: Vec<u32>, weight: f32| {
        PostingIterator::with_query_weight(
            token.to_owned(),
            rank,
            rank,
            weight,
            generate_posting_list(docs, weight, None, true),
            4096,
        )
    };
    let mut docs = DocSet::default();
    for doc in 0..4096u64 {
        docs.append(doc, 1);
    }
    let shared_floor = Arc::new(AtomicU32::new(0.15_f32.to_bits()));
    let mut wand = Wand::new(
        Operator::Or,
        [
            build("left", 0, left, 2.0),
            build("right", 1, right, 2.0),
            build("opt", 2, optional, 0.1),
        ]
        .into_iter(),
        &docs,
        InverseDocLengthScorer,
    )
    .with_shared_threshold(shared_floor);
    let hits = wand
        .maxscore_search(
            &FtsSearchParams::default().with_limit(Some(10)),
            &NoOpMetricsCollector,
        )
        .unwrap();
    assert_eq!(hits.len(), 10);
    assert!(
        wand.maxscore_single_essential_windows > 0,
        "a 2048+ gap between essentials must take the Lucene single-clause inner window"
    );
}

#[test]
fn maxscore_first_required_keeps_winners_on_the_buffer() {
    // Optional is non-essential once the floor sits between its bound and
    // the essential bound. Promoting it to required must still keep the
    // document that has both terms; the essential posting is not the
    // leapfrog driver.
    let postings = [
        (0_u32, 0.3_f32, vec![0_u32]),
        (1_u32, 0.4_f32, vec![0_u32, 1]),
    ]
    .into_iter()
    .map(|(position, query_weight, docs)| {
        PostingIterator::with_query_weight(
            format!("t{position}"),
            position,
            position,
            query_weight,
            generate_posting_list(docs, query_weight, None, true),
            2,
        )
    })
    .collect::<Vec<_>>();
    let mut docs = DocSet::default();
    docs.append(0, 1);
    docs.append(1, 1);
    let mut wand = Wand::new(
        Operator::Or,
        postings.into_iter(),
        &docs,
        CountingScorer {
            scored: Arc::new(AtomicUsize::new(0)),
        },
    );
    wand.threshold = 0.45;
    let hits = wand
        .maxscore_search(
            &FtsSearchParams::default().with_limit(Some(10)),
            &NoOpMetricsCollector,
        )
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].document, 0);
    assert!(hits[0].freqs.iter().any(|(term, _)| *term == 0));
    assert!(wand.maxscore_single_essential_windows > 0);
}

/// Oracle: `maxscore_search` must publish the same top-k as an exhaustive
/// fold over every document, scored in query order. The fold shares no code
/// with MAXSCORE, so it pins the hit set, the order, and the k-th score
/// without needing a second search path to compare against.
#[rstest]
fn maxscore_matches_an_exhaustive_query_order_fold(
    #[values(2_usize, 3_usize)] num_terms: usize,
    #[values(4_usize, 10_usize)] limit: usize,
    // `gapped` puts one clause's postings a full top2-gap away from the
    // previous one, so the exhaustive comparison also covers the shortcut
    // that hands a stretch to the single-essential path. The interleaved
    // layout spans 768 docs and can never reach the 2048 gap.
    #[values(false, true)] gapped: bool,
) {
    let block = MAX_POSTING_BLOCK_SIZE as u32;
    let gap_span = (MAXSCORE_INNER_WINDOW / 2) as u32;
    let total_docs = if gapped {
        num_terms as u32 * gap_span + 64
    } else {
        3 * block
    };
    let mut docs = DocSet::default();
    for doc in 0..total_docs {
        docs.append(u64::from(doc), doc % 17 + 1);
    }

    // Clause `term` covers `term, term+num_terms, term+2*num_terms, ...`,
    // so membership is a closed form instead of a posting-list walk. The
    // gapped layout gives clause `term` the 32 docs starting at
    // `term * gap_span`, plus the first doc of the next clause's stretch:
    // without a shared document the shortcut's span could be extended past
    // the second essential without changing any score, and the oracle would
    // not be able to tell a wrong span from a right one.
    let covered = move |doc: u32, term: u32| {
        if gapped {
            let own = doc >= term * gap_span && doc < term * gap_span + 32;
            let shared = term + 1 < num_terms as u32 && doc == (term + 1) * gap_span;
            own || shared
        } else {
            doc >= term && (doc - term).is_multiple_of(num_terms as u32)
        }
    };
    let freq_of = |doc: u32, term: u32| (doc + term) % 5 + 1;

    let mut expected = (0..total_docs)
        .map(|doc| {
            let norm = 0.3 + docs.scoring_num_tokens(doc) as f32 * 0.1;
            let score = (0..num_terms as u32)
                .filter(|term| covered(doc, *term))
                .fold(0.0_f32, |sum, term| {
                    sum + bm25_doc_weight_with_norm(freq_of(doc, term), norm)
                });
            (u64::from(doc), score)
        })
        .filter(|(_, score)| *score > 0.0)
        .collect::<Vec<_>>();
    expected.sort_unstable_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    // Lengths and frequencies are quantized, so many documents tie on
    // score. Which tied documents fill the last slots is not observable,
    // so pin the strict winners and the tie set instead of one arbitrary
    // ordering.
    let kth = expected[limit - 1].1;
    let strictly_better = expected
        .iter()
        .filter(|(_, score)| *score > kth)
        .map(|(doc, _)| *doc)
        .collect::<Vec<_>>();
    let tying = expected
        .iter()
        .filter(|(_, score)| *score == kth)
        .map(|(doc, _)| *doc)
        .collect::<Vec<_>>();

    let postings = (0..num_terms as u32)
        .map(|term| {
            let doc_ids = (0..total_docs)
                .filter(|doc| covered(*doc, term))
                .collect::<Vec<_>>();
            let freqs = doc_ids
                .iter()
                .map(|doc| freq_of(*doc, term))
                .collect::<Vec<_>>();
            let lengths = doc_ids.iter().map(|doc| doc % 17 + 1).collect::<Vec<_>>();
            PostingIterator::with_query_weight(
                format!("t{term}"),
                term,
                term,
                1.0,
                generate_impact_posting_list_with_freqs_and_block_size(
                    doc_ids,
                    freqs,
                    lengths,
                    MAX_POSTING_BLOCK_SIZE,
                ),
                docs.len(),
            )
        })
        .collect::<Vec<_>>();
    let shared_floor = Arc::new(AtomicU32::new(0.0_f32.to_bits()));
    let mut wand = Wand::new(
        Operator::Or,
        postings.into_iter(),
        &docs,
        VariedBm25ShapeScorer,
    )
    .with_shared_threshold(shared_floor.clone());
    let hits = wand
        .maxscore_search(
            &FtsSearchParams::default().with_limit(Some(limit)),
            &NoOpMetricsCollector,
        )
        .unwrap();

    let got = hits
        .iter()
        .map(|hit| hit.posting_doc_id)
        .collect::<Vec<_>>();
    assert_eq!(got.len(), limit);
    for doc in &strictly_better {
        assert!(
            got.contains(doc),
            "doc {doc} scores above the k-th ({kth}) and must be in the top {limit}"
        );
    }
    for doc in &got {
        assert!(
            strictly_better.contains(doc) || tying.contains(doc),
            "doc {doc} is not a top-{limit} candidate (k-th = {kth})"
        );
    }
    // The published floor is the k-th best score, so it pins the value and
    // not just the membership.
    assert_eq!(shared_floor.load(Ordering::Relaxed), kth.to_bits());
    if gapped {
        assert!(
            wand.maxscore_top2_gap_windows > 0,
            "a gap of {gap_span} must hand the stretch to the top2-gap shortcut"
        );
    }
}

/// The gap that hands a stretch to the single-essential inner window is
/// `MAXSCORE_INNER_WINDOW / 2`. Pin the boundary from both sides instead of
/// only testing a comfortable distance away from it.
#[rstest]
fn maxscore_top2_gap_boundary(#[values(2047_u32, 2048_u32, 2049_u32)] gap: u32) {
    let left = (0..32_u32).collect::<Vec<_>>();
    let right = (gap..gap + 32).collect::<Vec<_>>();
    let num_docs = (gap + 64) as usize;
    let build = |token: &str, rank: u32, doc_ids: Vec<u32>, weight: f32| {
        PostingIterator::with_query_weight(
            token.to_owned(),
            rank,
            rank,
            weight,
            generate_posting_list(doc_ids, weight, None, true),
            num_docs,
        )
    };
    let mut docs = DocSet::default();
    for doc in 0..num_docs as u64 {
        docs.append(doc, 1);
    }
    let mut wand = Wand::new(
        Operator::Or,
        // Distinct weights: every document of "left" outranks every document
        // of "right", so the top ten is a known set instead of an arbitrary
        // pick out of 64 tied documents.
        [build("left", 0, left, 3.0), build("right", 1, right, 1.0)].into_iter(),
        &docs,
        InverseDocLengthScorer,
    );
    let hits = wand
        .maxscore_search(
            &FtsSearchParams::default().with_limit(Some(10)),
            &NoOpMetricsCollector,
        )
        .unwrap();

    assert_eq!(hits.len(), 10);
    assert!(
        hits.iter().all(|hit| hit.posting_doc_id < 32),
        "the heavier clause must fill the top ten, got {:?}",
        hits.iter()
            .map(|hit| hit.posting_doc_id)
            .collect::<Vec<_>>()
    );
    if gap >= (MAXSCORE_INNER_WINDOW / 2) as u32 {
        assert!(
            wand.maxscore_single_essential_windows > 0,
            "gap {gap} must take the single-essential inner window"
        );
    } else {
        assert!(
            wand.maxscore_general_windows > 0,
            "gap {gap} must stay on the general window path"
        );
    }
}
