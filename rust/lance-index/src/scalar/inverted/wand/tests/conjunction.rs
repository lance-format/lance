// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

struct CandidateOnlyBm25ShapeScorer {
    norm_calls: Arc<AtomicUsize>,
    unsupported_norm: Option<f32>,
}

impl Scorer for CandidateOnlyBm25ShapeScorer {
    fn query_weight(&self, _token: &str) -> f32 {
        1.0
    }

    fn doc_weight(&self, freq: u32, _doc_tokens: u32) -> f32 {
        bm25_doc_weight_with_norm(freq, 0.4)
    }

    fn doc_weight_upper_bound(&self) -> Option<f32> {
        Some(BM25_DOC_WEIGHT_UPPER_BOUND)
    }

    fn doc_norm(&self, doc_tokens: u32) -> Option<f32> {
        self.norm_calls.fetch_add(1, Ordering::Relaxed);
        if doc_tokens == 1 {
            Some(0.4)
        } else {
            self.unsupported_norm
        }
    }
}

struct NonFiniteBoundBm25ShapeScorer;

impl Scorer for NonFiniteBoundBm25ShapeScorer {
    fn query_weight(&self, _token: &str) -> f32 {
        1.0
    }

    fn doc_weight(&self, _freq: u32, _doc_tokens: u32) -> f32 {
        f32::INFINITY
    }

    fn doc_norm(&self, _doc_tokens: u32) -> Option<f32> {
        Some(1.0)
    }
}

#[cfg_attr(coverage, coverage(off))]
fn unit_length_docs(num_docs: usize) -> DocSet {
    let mut docs = DocSet::default();
    for doc_id in 0..num_docs {
        docs.append(doc_id as u64, 1);
    }
    docs
}

#[cfg_attr(coverage, coverage(off))]
fn unit_length_impact_posting(
    term: u32,
    doc_ids: Vec<u32>,
    frequencies: Vec<u32>,
    num_docs: usize,
) -> PostingIterator {
    let doc_lengths = vec![1; doc_ids.len()];
    PostingIterator::with_query_weight(
        format!("t{term}"),
        term,
        term,
        1.0,
        generate_impact_posting_list_with_freqs_and_block_size(
            doc_ids,
            frequencies,
            doc_lengths,
            MAX_POSTING_BLOCK_SIZE,
        ),
        num_docs,
    )
}

#[test]
fn test_and_search_terminates_for_disjoint_postings() {
    let mut docs = DocSet::default();
    for i in 0..6 {
        docs.append(i, 1);
    }

    let postings = vec![
        PostingIterator::with_query_weight(
            String::from("a"),
            0,
            0,
            1.0,
            generate_posting_list(vec![0, 2, 4], 1.0, None, false),
            docs.len(),
        ),
        PostingIterator::with_query_weight(
            String::from("b"),
            1,
            1,
            1.0,
            generate_posting_list(vec![1, 3, 5], 1.0, None, false),
            docs.len(),
        ),
    ];

    let mut wand = Wand::new(Operator::And, postings.into_iter(), &docs, UnitScorer);
    assert!(wand.next().unwrap().is_none());
}

#[test]
fn test_and_search_prunes_with_threshold_and_keeps_candidate() {
    let mut docs = DocSet::default();
    for i in 0..(2 * BLOCK_SIZE as u64) {
        let doc_tokens = if i < BLOCK_SIZE as u64 { 100 } else { 1 };
        docs.append(i, doc_tokens);
    }
    let all_docs = (0..2 * BLOCK_SIZE as u32).collect::<Vec<_>>();

    let postings = vec![
        PostingIterator::with_query_weight(
            String::from("a"),
            0,
            0,
            1.0,
            generate_posting_list(all_docs.clone(), 1.0, Some(vec![0.02, 1.0]), true),
            docs.len(),
        ),
        PostingIterator::with_query_weight(
            String::from("b"),
            1,
            1,
            1.0,
            generate_posting_list(all_docs, 1.0, Some(vec![0.02, 1.0]), true),
            docs.len(),
        ),
    ];

    let mut wand = Wand::new(
        Operator::And,
        postings.into_iter(),
        &docs,
        InverseDocLengthScorer,
    );
    wand.threshold = 0.5;

    let candidate = wand.next().unwrap().unwrap();
    assert_eq!(candidate.0.doc_id(), BLOCK_SIZE as u64);
}

#[test]
fn test_and_advance_falls_back_to_narrow_when_range_max_loosens_bound() {
    let total = 4 * BLOCK_SIZE as u32;
    let mut docs = DocSet::default();
    for i in 0..total {
        docs.append(i as u64, 1);
    }

    let lead_docs = (0..total).step_by(2).collect::<Vec<_>>();
    let follower_docs = (0..total).collect::<Vec<_>>();
    let postings = vec![
        PostingIterator::with_query_weight(
            String::from("lead"),
            0,
            0,
            1.0,
            generate_posting_list(lead_docs, 1.0, Some(vec![1.0, 1.0]), true),
            docs.len(),
        ),
        PostingIterator::with_query_weight(
            String::from("follower"),
            1,
            1,
            1.0,
            generate_posting_list(follower_docs, 10.0, Some(vec![0.1, 10.0, 0.1, 0.1]), true),
            docs.len(),
        ),
    ];

    let mut wand = Wand::new(Operator::And, postings.into_iter(), &docs, UnitScorer);
    wand.threshold = 5.0;

    let target = wand.and_advance_target(0);

    assert_eq!(target, BLOCK_SIZE as u64);
    assert_eq!(wand.up_to, Some((2 * BLOCK_SIZE - 1) as u64));
    assert!(
        (wand.and_max_score - 11.0).abs() < 1e-6,
        "expected the second narrow window to include the high follower block, got {}",
        wand.and_max_score
    );
    assert_eq!(wand.and_window_stats.windows_wide, 0);
    assert_eq!(wand.and_window_stats.windows_narrow, 2);
    assert_eq!(wand.and_window_stats.windows_skipped, 1);
}

#[test]
fn test_and_advance_uses_narrow_window_for_candidate_ranges() {
    let total = 4 * BLOCK_SIZE as u32;
    let mut docs = DocSet::default();
    for i in 0..total {
        docs.append(i as u64, 1);
    }

    let lead_docs = (0..total).step_by(2).collect::<Vec<_>>();
    let follower_docs = (0..total).collect::<Vec<_>>();
    let postings = vec![
        PostingIterator::with_query_weight(
            String::from("lead"),
            0,
            0,
            1.0,
            generate_posting_list(lead_docs, 1.0, Some(vec![1.0, 1.0]), true),
            docs.len(),
        ),
        PostingIterator::with_query_weight(
            String::from("follower"),
            1,
            1,
            1.0,
            generate_posting_list(follower_docs, 1.0, Some(vec![1.0, 1.0, 1.0, 1.0]), true),
            docs.len(),
        ),
    ];

    let mut wand = Wand::new(Operator::And, postings.into_iter(), &docs, UnitScorer);
    wand.threshold = 1.5;

    let target = wand.and_advance_target(0);

    assert_eq!(target, 0);
    assert_eq!(wand.up_to, Some((BLOCK_SIZE - 1) as u64));
    assert!((wand.and_max_score - 2.0).abs() < 1e-6);
    assert_eq!(wand.and_window_stats.windows_wide, 0);
    assert_eq!(wand.and_window_stats.windows_narrow, 1);
    assert_eq!(wand.and_window_stats.range_blocks_scanned, 0);
}

#[test]
fn test_and_wide_window_only_skips_and_does_not_return_candidates() {
    let total = 4 * BLOCK_SIZE as u32;
    let mut docs = DocSet::default();
    for i in 0..total {
        docs.append(i as u64, 1);
    }

    let lead_docs = (0..total).step_by(2).collect::<Vec<_>>();
    let follower_docs = (0..total).collect::<Vec<_>>();
    let postings = vec![
        PostingIterator::with_query_weight(
            String::from("lead"),
            0,
            0,
            1.0,
            generate_posting_list(lead_docs, 3.0, Some(vec![1.0, 3.0]), true),
            docs.len(),
        ),
        PostingIterator::with_query_weight(
            String::from("follower"),
            1,
            1,
            1.0,
            generate_posting_list(follower_docs, 3.0, Some(vec![0.1, 0.1, 3.0, 3.0]), true),
            docs.len(),
        ),
    ];

    let mut wand = Wand::new(Operator::And, postings.into_iter(), &docs, UnitScorer);
    wand.threshold = 2.0;

    let candidate = wand.next().unwrap().unwrap();

    assert_eq!(candidate.0.doc_id(), (2 * BLOCK_SIZE) as u64);
    assert_eq!(wand.up_to, Some((3 * BLOCK_SIZE - 1) as u64));
    assert_eq!(wand.and_window_stats.windows_wide, 1);
    assert_eq!(wand.and_window_stats.windows_skipped, 1);
    assert_eq!(wand.and_window_stats.windows_narrow, 1);
    assert_eq!(wand.and_window_stats.candidates_returned, 1);
}

#[test]
fn test_and_range_max_preserves_exact_top_k() {
    let total = 4 * BLOCK_SIZE as u32;
    let hot = BLOCK_SIZE as u32..BLOCK_SIZE as u32 + 16;
    let mut docs = DocSet::default();
    for doc_id in 0..total {
        let doc_tokens = if hot.contains(&doc_id) { 1 } else { 1000 };
        docs.append(doc_id as u64, doc_tokens);
    }

    let params = FtsSearchParams::new().with_limit(Some(8));
    let run = |is_compressed: bool| {
        let lead_docs = (0..total).step_by(2).collect::<Vec<_>>();
        let follower_docs = (0..total).collect::<Vec<_>>();
        let lead_scores = is_compressed.then_some(vec![1.0, 0.001]);
        let follower_scores = is_compressed.then_some(vec![0.001, 1.0, 0.001, 0.001]);
        let postings = vec![
            PostingIterator::with_query_weight(
                String::from("lead"),
                0,
                0,
                1.0,
                generate_posting_list(lead_docs, 1.0, lead_scores, is_compressed),
                docs.len(),
            ),
            PostingIterator::with_query_weight(
                String::from("follower"),
                1,
                1,
                1.0,
                generate_posting_list(follower_docs, 1.0, follower_scores, is_compressed),
                docs.len(),
            ),
        ];
        let mut wand = Wand::new(
            Operator::And,
            postings.into_iter(),
            &docs,
            InverseDocLengthScorer,
        );
        sorted_candidate_row_ids(wand.search(&params, &NoOpMetricsCollector).unwrap())
    };

    let compressed = run(true);
    let plain = run(false);
    let expected = hot.step_by(2).map(u64::from).collect::<Vec<_>>();
    assert_eq!(compressed, expected);
    assert_eq!(compressed, plain);
}

#[test]
fn test_and_candidate_prune_scores_first_term_before_full_score() {
    let total_docs = 2 * BLOCK_SIZE as u32 + 1;
    let mut docs = DocSet::default();
    for doc_id in 0..total_docs {
        let doc_tokens = if doc_id == 0 { 1 } else { 1000 };
        docs.append(doc_id as u64, doc_tokens);
    }

    let first_docs = (0..2 * BLOCK_SIZE as u32).collect::<Vec<_>>();
    let second_docs = (0..total_docs).collect::<Vec<_>>();
    let postings = vec![
        PostingIterator::with_query_weight(
            String::from("a"),
            0,
            0,
            1.0,
            generate_posting_list(first_docs, 1.0, Some(vec![1.0, 0.001]), true),
            docs.len(),
        ),
        PostingIterator::with_query_weight(
            String::from("b"),
            1,
            1,
            1.0,
            generate_posting_list(second_docs, 1.0, Some(vec![1.0, 0.001, 0.001]), true),
            docs.len(),
        ),
    ];

    let scored = Arc::new(AtomicUsize::new(0));
    let mut wand = Wand::new(
        Operator::And,
        postings.into_iter(),
        &docs,
        CountingScorer {
            scored: scored.clone(),
        },
    );

    let result = wand
        .search(
            &FtsSearchParams::new().with_limit(Some(1)),
            &NoOpMetricsCollector,
        )
        .unwrap();

    let addrs = result
        .into_iter()
        .map(|doc| doc.document)
        .collect::<Vec<_>>();
    assert_eq!(addrs, vec![0]);
    let scored = scored.load(Ordering::Relaxed);
    // The bulk path evaluates 63 doc weights up front to fill its
    // frequency-bound prune LUT; those are bound computations, not
    // per-candidate scoring.
    assert!(
        scored <= BLOCK_SIZE + 1 + 63,
        "expected candidate pruning to avoid full scoring in the first block, scored {scored}"
    );
}

#[test]
fn test_and_candidate_prune_keeps_top_candidate() {
    let total_docs = 2 * BLOCK_SIZE as u32 + 1;
    let mut docs = DocSet::default();
    for doc_id in 0..total_docs {
        let doc_tokens = if doc_id == 0 { 1 } else { 1000 };
        docs.append(doc_id as u64, doc_tokens);
    }

    let first_docs = (0..2 * BLOCK_SIZE as u32).collect::<Vec<_>>();
    let second_docs = (0..total_docs).collect::<Vec<_>>();
    let postings = vec![
        PostingIterator::with_query_weight(
            String::from("a"),
            0,
            0,
            1.0,
            generate_posting_list(first_docs, 1.0, Some(vec![1.0, 0.001]), true),
            docs.len(),
        ),
        PostingIterator::with_query_weight(
            String::from("b"),
            1,
            1,
            1.0,
            generate_posting_list(second_docs, 1.0, Some(vec![1.0, 0.001, 0.001]), true),
            docs.len(),
        ),
    ];

    let mut wand = Wand::new(
        Operator::And,
        postings.into_iter(),
        &docs,
        InverseDocLengthScorer,
    );
    let metrics = LocalMetricsCollector::default();
    let result = wand
        .search(&FtsSearchParams::new().with_limit(Some(1)), &metrics)
        .unwrap();

    let addrs = result
        .into_iter()
        .map(|doc| doc.document)
        .collect::<Vec<_>>();
    assert_eq!(addrs, vec![0]);

    assert_eq!(metrics.comparisons.load(Ordering::Relaxed), 1);
}

#[test]
fn test_and_candidate_prune_keeps_later_high_score_candidate() {
    let mut docs = DocSet::default();
    for doc_id in 0..3 {
        docs.append(doc_id, 1);
    }

    let postings = vec![
        PostingIterator::with_query_weight(
            String::from("a"),
            0,
            0,
            1.0,
            generate_posting_list_with_freqs(vec![0, 1], vec![10, 1], 10.0, Some(vec![10.0]), true),
            docs.len(),
        ),
        PostingIterator::with_query_weight(
            String::from("b"),
            1,
            1,
            1.0,
            generate_posting_list_with_freqs(
                vec![0, 1, 2],
                vec![1, 20, 1],
                20.0,
                Some(vec![20.0]),
                true,
            ),
            docs.len(),
        ),
    ];

    let mut wand = Wand::new(Operator::And, postings.into_iter(), &docs, UnitScorer);
    let result = wand
        .search(
            &FtsSearchParams::new().with_limit(Some(1)),
            &NoOpMetricsCollector,
        )
        .unwrap();

    let addrs = result
        .into_iter()
        .map(|doc| doc.document)
        .collect::<Vec<_>>();
    assert_eq!(addrs, vec![1]);
}

#[test]
fn score_first_and_checks_first_follower_before_decoding_lead_frequency() {
    let postings = [
        vec![0],
        vec![100, 101],
        vec![100, 101, 102],
        vec![100, 101, 102, 103],
    ]
    .into_iter()
    .enumerate()
    .map(|(term, doc_ids)| {
        unit_length_impact_posting(term as u32, doc_ids.clone(), vec![1; doc_ids.len()], 104)
    });
    let docs = CostOnlyDocuments {
        total_docs: 104,
        visible_cost_upper_bound: 104,
    };
    let mut wand = Wand::new(Operator::And, postings, &docs, VariedBm25ShapeScorer)
        .with_bulk_and_mode(BulkAndMode::Off);
    assert!(wand.score_first_and_enabled);
    wand.threshold = f32::from_bits(1);

    assert!(wand.next().unwrap().is_none());
    assert!(wand.score_first_and_dense_range.is_none());
    assert!(wand.score_first_and_norm_k.get().is_none());
    assert!(
        wand.lead
            .iter()
            .all(|posting| posting.frequency_blocks_decoded() == 0),
        "the first follower miss must leapfrog before any frequency decode"
    );
}

#[test]
fn score_first_and_checks_all_followers_before_decoding_frequencies() {
    let postings = [
        vec![0],
        vec![0, 200],
        vec![100, 101, 102],
        vec![100, 101, 102, 103],
    ]
    .into_iter()
    .enumerate()
    .map(|(term, doc_ids)| {
        unit_length_impact_posting(term as u32, doc_ids.clone(), vec![1; doc_ids.len()], 201)
    });
    let docs = CostOnlyDocuments {
        total_docs: 201,
        visible_cost_upper_bound: 201,
    };
    let mut wand = Wand::new(Operator::And, postings, &docs, VariedBm25ShapeScorer)
        .with_bulk_and_mode(BulkAndMode::Off);
    assert!(wand.score_first_and_enabled);
    wand.threshold = f32::from_bits(1);

    assert!(wand.next().unwrap().is_none());
    assert!(wand.score_first_and_dense_range.is_none());
    assert!(wand.score_first_and_norm_k.get().is_none());
    let decoded_by_term = wand
        .lead
        .iter()
        .map(|posting| (posting.token_id, posting.frequency_blocks_decoded()))
        .sorted_unstable()
        .collect::<Vec<_>>();
    assert_eq!(decoded_by_term, vec![(0, 0), (1, 0), (2, 0), (3, 0)]);
}

#[rstest]
#[case::dense(false, true, true, Some((0, MAX_POSTING_BLOCK_SIZE as u64 - 1)))]
#[case::hole(true, true, true, None)]
#[case::dead_window(false, false, true, None)]
#[case::initial_dead_window(false, false, false, None)]
fn score_first_and_certifies_only_dense_blocks(
    #[case] has_hole: bool,
    #[case] is_live_window: bool,
    #[case] has_prior_live_window: bool,
    #[case] expected: Option<(u64, u64)>,
) {
    let postings = (0..4)
        .map(|term| {
            let doc_ids: Vec<u32> = if term == 1 && has_hole {
                (0..128).chain(129..=256).chain([300]).collect()
            } else {
                (0..MAX_POSTING_BLOCK_SIZE as u32 + term).collect()
            };
            unit_length_impact_posting(term, doc_ids.clone(), vec![1; doc_ids.len()], 301)
        })
        .collect::<Vec<_>>();
    let docs = CostOnlyDocuments {
        total_docs: 301,
        visible_cost_upper_bound: 301,
    };
    let mut wand = Wand::new(
        Operator::And,
        postings.into_iter(),
        &docs,
        VariedBm25ShapeScorer,
    )
    .with_bulk_and_mode(BulkAndMode::Off);
    wand.threshold = f32::from_bits(1);

    if has_prior_live_window {
        assert_eq!(wand.and_advance_target(0), 0);
    } else {
        wand.threshold = f32::MAX;
        wand.and_move_to_next_block(0);
    }
    // Empty decoded buffers make any payload access during preparation observable.
    let decoded_doc_ids = wand
        .lead
        .iter()
        .map(|posting| {
            let compressed = unsafe { &mut *posting.compressed.as_ref().unwrap().get() };
            (
                compressed.block_idx,
                std::mem::take(&mut compressed.doc_ids),
            )
        })
        .collect::<Vec<_>>();
    if !is_live_window {
        wand.threshold = f32::MAX;
    }
    wand.prepare_score_first_and_window();
    assert_eq!(wand.score_first_and_dense_range, expected);
    assert!(wand.score_first_and_norm_k.get().is_none());
    if expected.is_some() {
        assert_eq!(wand.score_first_and_bounds_up_to, wand.up_to);
        assert_eq!(
            wand.score_first_and_suffix_bounds.len(),
            wand.lead.len() + 1
        );
    } else {
        assert!(wand.score_first_and_suffix_bounds.is_empty());
        assert!(wand.score_first_and_bounds_up_to.is_none());
    }
    for (posting, (block_idx, doc_ids)) in wand.lead.iter().zip(decoded_doc_ids) {
        let compressed = unsafe { &mut *posting.compressed.as_ref().unwrap().get() };
        assert_eq!(compressed.block_idx, block_idx);
        assert!(compressed.doc_ids.is_empty());
        compressed.doc_ids = doc_ids;
    }
    assert!(
        wand.lead
            .iter()
            .all(|posting| posting.frequency_blocks_decoded() == 0),
        "metadata certification must not decode frequencies"
    );
    if !has_hole && is_live_window {
        wand.seek(100);
        assert!(wand.score_first_and_dense_range.is_none());
        assert_eq!(wand.and_advance_target(100), 100);
        assert_eq!(wand.score_first_and_dense_range, Some((100, 255)));
    }
}

#[test]
fn score_first_and_certifies_short_tail_without_decoding_it() {
    let postings = (0..4)
        .map(|term| {
            let doc_ids = (0..MAX_POSTING_BLOCK_SIZE as u32 + 1 + term).collect::<Vec<_>>();
            unit_length_impact_posting(
                term,
                doc_ids.clone(),
                vec![1; doc_ids.len()],
                MAX_POSTING_BLOCK_SIZE + 4,
            )
        })
        .collect::<Vec<_>>();
    let docs = CostOnlyDocuments {
        total_docs: MAX_POSTING_BLOCK_SIZE + 4,
        visible_cost_upper_bound: MAX_POSTING_BLOCK_SIZE + 4,
    };
    let mut wand = Wand::new(
        Operator::And,
        postings.into_iter(),
        &docs,
        VariedBm25ShapeScorer,
    )
    .with_bulk_and_mode(BulkAndMode::Off);
    for posting in &mut wand.lead {
        posting.next_doc_id(MAX_POSTING_BLOCK_SIZE as u64, true);
    }
    wand.threshold = f32::from_bits(1);

    assert_eq!(
        wand.and_advance_target(MAX_POSTING_BLOCK_SIZE as u64),
        MAX_POSTING_BLOCK_SIZE as u64
    );
    assert_eq!(
        wand.score_first_and_dense_range,
        Some((MAX_POSTING_BLOCK_SIZE as u64, MAX_POSTING_BLOCK_SIZE as u64))
    );
    assert!(
        wand.lead
            .iter()
            .all(|posting| posting.frequency_blocks_decoded() == 0)
    );
}

#[test]
fn score_first_and_intersects_differently_aligned_dense_blocks() {
    let starts = [0, 100, 50, 80];
    let postings = starts.into_iter().enumerate().map(|(term, start)| {
        let doc_ids =
            (start..start + MAX_POSTING_BLOCK_SIZE as u32 + term as u32).collect::<Vec<_>>();
        unit_length_impact_posting(term as u32, doc_ids.clone(), vec![1; doc_ids.len()], 400)
    });
    let docs = CostOnlyDocuments {
        total_docs: 400,
        visible_cost_upper_bound: 400,
    };
    let mut wand = Wand::new(Operator::And, postings, &docs, VariedBm25ShapeScorer)
        .with_bulk_and_mode(BulkAndMode::Off);
    wand.threshold = f32::from_bits(1);

    assert_eq!(wand.and_advance_target(0), 0);
    assert_eq!(wand.score_first_and_dense_range, Some((100, 255)));
    assert_eq!(wand.next().unwrap().unwrap().0.doc_id(), 100);
}

#[test]
fn score_first_and_certifies_first_live_window_after_impact_skip() {
    let total_docs = 2 * MAX_POSTING_BLOCK_SIZE as u32;
    let postings = (0..4).map(|term| {
        let doc_ids = (0..total_docs + term).collect::<Vec<_>>();
        let freqs = doc_ids
            .iter()
            .map(|doc_id| {
                if *doc_id < MAX_POSTING_BLOCK_SIZE as u32 {
                    1
                } else {
                    20
                }
            })
            .collect::<Vec<_>>();
        unit_length_impact_posting(term, doc_ids, freqs, total_docs as usize + 4)
    });
    let docs = CostOnlyDocuments {
        total_docs: total_docs as usize + 4,
        visible_cost_upper_bound: total_docs as usize + 4,
    };
    let mut wand = Wand::new(Operator::And, postings, &docs, VariedBm25ShapeScorer)
        .with_bulk_and_mode(BulkAndMode::Off);
    wand.threshold = 7.0;

    assert_eq!(wand.and_advance_target(0), MAX_POSTING_BLOCK_SIZE as u64);
    assert!(wand.and_window_stats.windows_skipped > 0);
    assert_eq!(
        wand.score_first_and_dense_range,
        Some((
            MAX_POSTING_BLOCK_SIZE as u64,
            2 * MAX_POSTING_BLOCK_SIZE as u64 - 1
        ))
    );
}

#[test]
fn score_first_and_rejects_malformed_block_count_certificate() {
    let postings = (0..4).map(|term| {
        let doc_ids = (0..MAX_POSTING_BLOCK_SIZE as u32 + term).collect::<Vec<_>>();
        let mut list = generate_impact_posting_list_with_freqs_and_block_size(
            doc_ids.clone(),
            vec![1; doc_ids.len()],
            vec![1; doc_ids.len()],
            MAX_POSTING_BLOCK_SIZE,
        );
        if term == 3 {
            let PostingList::Compressed(list) = &mut list else {
                unreachable!();
            };
            list.length += MAX_POSTING_BLOCK_SIZE as u32;
        }
        PostingIterator::with_query_weight(
            format!("t{term}"),
            term,
            term,
            1.0,
            list,
            2 * MAX_POSTING_BLOCK_SIZE,
        )
    });
    let docs = CostOnlyDocuments {
        total_docs: 2 * MAX_POSTING_BLOCK_SIZE,
        visible_cost_upper_bound: 2 * MAX_POSTING_BLOCK_SIZE,
    };
    let mut wand = Wand::new(Operator::And, postings, &docs, VariedBm25ShapeScorer)
        .with_bulk_and_mode(BulkAndMode::Off);
    wand.threshold = f32::from_bits(1);

    assert_eq!(wand.and_advance_target(0), 0);
    assert!(wand.score_first_and_dense_range.is_none());
}

#[test]
fn dense_certificate_does_not_replace_membership_checks() {
    let postings = [
        vec![1],
        vec![0, 0, 2],
        vec![0, 1, 2, 3],
        vec![0, 1, 2, 3, 4],
    ]
    .into_iter()
    .enumerate()
    .map(|(term, doc_ids)| {
        unit_length_impact_posting(term as u32, doc_ids.clone(), vec![1; doc_ids.len()], 5)
    });
    let docs = CostOnlyDocuments {
        total_docs: 5,
        visible_cost_upper_bound: 5,
    };
    let mut wand = Wand::new(Operator::And, postings, &docs, VariedBm25ShapeScorer)
        .with_bulk_and_mode(BulkAndMode::Off);
    wand.threshold = f32::from_bits(1);

    assert_eq!(wand.and_advance_target(1), 1);
    assert_eq!(wand.score_first_and_dense_range, Some((1, 1)));
    assert!(wand.next().unwrap().is_none());
}

#[rstest]
#[case::unsupported(None, 6)]
#[case::negative(Some(-1.0), 261)]
#[case::non_finite(Some(f32::INFINITY), 261)]
fn score_first_and_norm_cache_matches_uncached_score_bits(
    #[case] unsupported_norm: Option<f32>,
    #[case] expected_norm_calls: usize,
) {
    fn run<S: Scorer>(scorer: S, docs: &DocSet) -> (Vec<(u64, u32)>, Option<bool>) {
        let postings = (0..4).map(|term| {
            let doc_ids = (0..4 + term).collect::<Vec<_>>();
            let freqs = doc_ids
                .iter()
                .map(|doc_id| (doc_id + term) % 4 + 1)
                .collect::<Vec<_>>();
            PostingIterator::with_query_weight(
                format!("t{term}"),
                term,
                term,
                0.001_882_293 + term as f32 * 0.000_173_205,
                generate_impact_posting_list_with_freqs_and_block_size(
                    doc_ids.clone(),
                    freqs,
                    vec![1; doc_ids.len()],
                    MAX_POSTING_BLOCK_SIZE,
                ),
                docs.len(),
            )
        });
        let mut wand =
            Wand::new(Operator::And, postings, docs, scorer).with_bulk_and_mode(BulkAndMode::Off);
        assert!(wand.score_first_and_enabled);
        assert!(wand.score_first_and_norm_k.get().is_none());
        wand.threshold = f32::from_bits(1);

        let mut scored = Vec::new();
        while let Some((doc, score)) = wand.next().unwrap() {
            scored.push((doc.doc_id(), score.to_bits()));
            if doc.doc_id() == 0 {
                let norm_cache_ptr = wand
                    .score_first_and_norm_k
                    .get()
                    .unwrap()
                    .as_ref()
                    .map(|(_, cache)| cache.as_ptr());
                wand.seek(1);
                assert!(wand.score_first_and_suffix_bounds.is_empty());
                assert!(wand.score_first_and_bounds_up_to.is_none());
                assert!(wand.score_first_and_dense_range.is_none());
                assert_eq!(
                    wand.score_first_and_norm_k
                        .get()
                        .unwrap()
                        .as_ref()
                        .map(|(_, cache)| cache.as_ptr()),
                    norm_cache_ptr
                );
            }
        }
        let cache_state = wand
            .score_first_and_norm_k
            .get()
            .map(|cache| cache.is_some());
        (scored, cache_state)
    }

    let mut docs = unit_length_docs(7);
    docs.set_quantized_scoring(true);

    let (cached, cached_state) = run(VariedBm25ShapeScorer, &docs);
    let norm_calls = Arc::new(AtomicUsize::new(0));
    let (uncached, uncached_state) = run(
        CandidateOnlyBm25ShapeScorer {
            norm_calls: norm_calls.clone(),
            unsupported_norm,
        },
        &docs,
    );

    assert_eq!(cached_state, Some(true));
    assert_eq!(uncached_state, Some(false));
    assert_eq!(cached, uncached);
    assert_eq!(cached.len(), 4);
    assert_eq!(
        norm_calls.load(Ordering::Relaxed),
        expected_norm_calls,
        "unsupported norm tables must be attempted once, then use the per-candidate fallback"
    );
}

#[rstest]
fn score_first_and_clears_certified_window_before_phrase_search(#[values(4, 5)] num_terms: u32) {
    let mut docs = unit_length_docs(2);
    docs.set_quantized_scoring(true);
    let postings = (0..num_terms).map(|term| {
        let mut list = generate_impact_posting_list_with_freqs_and_block_size(
            vec![0, 1],
            vec![1, 1],
            vec![1, 1],
            MAX_POSTING_BLOCK_SIZE,
        );
        let PostingList::Compressed(compressed) = &mut list else {
            unreachable!();
        };
        let codec = PositionStreamCodec::PackedDelta;
        let mut encoded = Vec::new();
        encode_position_stream_block_into(&[term, term], &[1, 1], codec, &mut encoded).unwrap();
        compressed.positions = Some(CompressedPositionStorage::SharedStream(
            SharedPositionStream::new(codec, vec![0], bytes::Bytes::from(encoded)),
        ));
        PostingIterator::with_query_weight(format!("t{term}"), term, term, 1.0, list, docs.len())
    });
    let mut wand = Wand::new(Operator::And, postings, &docs, VariedBm25ShapeScorer)
        .with_bulk_and_mode(BulkAndMode::Off);
    wand.threshold = f32::from_bits(1);
    assert_eq!(wand.next().unwrap().unwrap().0.doc_id(), 0);
    assert!(wand.score_first_and_enabled);
    assert_eq!(wand.score_first_and_dense_range, Some((0, 1)));
    assert!(wand.score_first_and_bounds_up_to.is_some());
    assert!(!wand.score_first_and_suffix_bounds.is_empty());
    assert!(wand.and_candidate_score.is_some());
    assert!(wand.score_first_and_norm_k.get().unwrap().is_some());

    let mut params = FtsSearchParams::new().with_limit(Some(10));
    params.phrase_slop = Some(0);
    let results = wand.search(&params, &NoOpMetricsCollector).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].document, 1);
    assert!(!wand.score_first_and_enabled);
    assert!(wand.score_first_and_suffix_bounds.is_empty());
    assert!(wand.score_first_and_bounds_up_to.is_none());
    assert!(wand.score_first_and_dense_range.is_none());
    assert!(wand.and_candidate_score.is_none());
    assert!(wand.score_first_and_norm_k.get().unwrap().is_some());
}

#[test]
fn score_first_and_prunes_before_decoding_final_follower() {
    let total_docs = 2 * MAX_POSTING_BLOCK_SIZE as u32;
    let doc_ids = (0..total_docs).collect::<Vec<_>>();
    let docs = unit_length_docs(doc_ids.len());

    let run = |is_score_first_enabled| {
        let postings = (0..4)
            .map(|term| {
                // Every term contains the frequent-hit range [4, 512),
                // while staggered starts fix a deterministic cost order.
                let term_doc_ids = (4 - term as u32..total_docs).collect::<Vec<_>>();
                let mut freqs = vec![1; term_doc_ids.len()];
                freqs[term] = 2;
                // Each follower has a high-frequency non-winner in the
                // first block, so the lead-only bound cannot reject it.
                if term > 0 {
                    freqs[MAX_POSTING_BLOCK_SIZE - 4] = 3;
                    freqs[term_doc_ids.len() - 4 + term] = 3;
                }
                unit_length_impact_posting(term as u32, term_doc_ids, freqs, docs.len())
            })
            .collect::<Vec<_>>();
        let scored = Arc::new(AtomicUsize::new(0));
        let mut wand = Wand::new(
            Operator::And,
            postings.into_iter(),
            &docs,
            CountingBm25ShapeScorer { scored },
        )
        .with_bulk_and_mode(BulkAndMode::Off)
        .with_score_first_and(is_score_first_enabled);
        let result = wand
            .search(
                &FtsSearchParams::new().with_limit(Some(1)),
                &NoOpMetricsCollector,
            )
            .unwrap();
        (
            sorted_candidate_row_ids(result),
            wand.lead
                .iter()
                .map(|posting| posting.frequency_blocks_decoded())
                .sum::<usize>(),
            wand.and_window_stats.score_first_rejections,
        )
    };

    let (classic_docs, classic_decodes, classic_rejections) = run(false);
    let (score_first_docs, score_first_decodes, score_first_rejections) = run(true);
    assert_eq!(classic_docs, vec![4]);
    assert_eq!(score_first_docs, classic_docs);
    assert_eq!(classic_rejections, 0);
    assert!(score_first_rejections > MAX_POSTING_BLOCK_SIZE);
    assert!(
        score_first_decodes < classic_decodes,
        "partial exact scoring should avoid decoding the final follower block: score-first={score_first_decodes}, classic={classic_decodes}"
    );
}

#[rstest]
#[case::four_k1(4, 1)]
#[case::four_k10(4, 10)]
#[case::four_underfilled(4, 1024)]
#[case::five_k1(5, 1)]
#[case::five_k10(5, 10)]
#[case::five_underfilled(5, 1024)]
fn score_first_and_matches_classic_exact_scores(
    #[case] num_terms: usize,
    #[case] limit: usize,
    #[values(false, true)] is_quantized_scoring: bool,
) {
    let run = |is_score_first_enabled| {
        let total_docs = 3 * MAX_POSTING_BLOCK_SIZE as u32;
        let mut docs = DocSet::default();
        for doc_id in 0..total_docs {
            docs.append(u64::from(doc_id), doc_id % 17 + 1);
        }
        docs.set_quantized_scoring(is_quantized_scoring);
        let postings = (0..num_terms)
            .map(|term| {
                let doc_ids = (term as u32 * 3..total_docs).collect::<Vec<_>>();
                let freqs = doc_ids
                    .iter()
                    .map(|doc_id| {
                        if *doc_id + 1 == total_docs {
                            20 + term as u32
                        } else {
                            (*doc_id + term as u32) % 5 + 1
                        }
                    })
                    .collect::<Vec<_>>();
                let doc_lengths = doc_ids
                    .iter()
                    .map(|doc_id| *doc_id % 17 + 1)
                    .collect::<Vec<_>>();
                // Later terms have shorter postings, so cost order is the
                // reverse of this canonical query order.
                let position = term as u32;
                PostingIterator::with_query_weight(
                    format!("t{term}"),
                    term as u32,
                    position,
                    0.001_882_293 + position as f32 * 0.000_173_205,
                    generate_impact_posting_list_with_freqs_and_block_size(
                        doc_ids,
                        freqs,
                        doc_lengths,
                        MAX_POSTING_BLOCK_SIZE,
                    ),
                    docs.len(),
                )
            })
            .collect::<Vec<_>>();
        let mut wand = Wand::new(
            Operator::And,
            postings.into_iter(),
            &docs,
            VariedBm25ShapeScorer,
        )
        .with_bulk_and_mode(BulkAndMode::Off)
        .with_score_first_and(is_score_first_enabled);
        let mut top = Vec::<(u64, f32)>::with_capacity(limit.min(total_docs as usize));
        while let Some((doc, cached_score)) = wand.next().unwrap() {
            let score = if wand.and_candidate_score.is_some() {
                cached_score
            } else {
                wand.score_in_query_order(wand.documents.doc_length(&doc))
            };
            top.push((doc.doc_id(), score));
            top.sort_unstable_by(|left, right| {
                right
                    .1
                    .total_cmp(&left.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            top.truncate(limit);
            if top.len() == limit {
                wand.update_threshold(top[limit - 1].1, 1.0);
            }
        }
        (
            top.into_iter()
                .map(|(doc, score)| (doc, score.to_bits()))
                .collect::<Vec<_>>(),
            wand.and_window_stats.score_first_rejections,
        )
    };

    let (classic, classic_rejections) = run(false);
    let (score_first, score_first_rejections) = run(true);
    assert_eq!(score_first, classic);
    assert_eq!(classic_rejections, 0);
    if limit == 1 {
        assert_eq!(score_first[0].0, 3 * MAX_POSTING_BLOCK_SIZE as u64 - 1);
    }
    if limit > 3 * MAX_POSTING_BLOCK_SIZE {
        assert_eq!(score_first_rejections, 0);
    } else {
        assert!(score_first_rejections > 0);
    }
}

#[rstest]
fn score_first_and_matches_doc_first_across_dense_and_sparse_windows(
    #[values(4, 5)] num_terms: u32,
) {
    let total_docs = 3 * MAX_POSTING_BLOCK_SIZE as u32 + 1;
    let missing_doc = MAX_POSTING_BLOCK_SIZE as u32 + 128;
    let mut docs = unit_length_docs(total_docs as usize);
    docs.set_quantized_scoring(true);
    let run = |is_score_first_enabled| {
        let postings = (0..num_terms).map(|term| {
            let doc_ids = (0..total_docs)
                .filter(|doc_id| *doc_id != missing_doc)
                .collect::<Vec<_>>();
            let freqs = doc_ids
                .iter()
                .map(|doc_id| (doc_id + term) % 4 + 1)
                .collect::<Vec<_>>();
            PostingIterator::with_query_weight(
                format!("t{term}"),
                term,
                num_terms - term - 1,
                0.001_882_293 + term as f32 * 0.000_173_205,
                generate_impact_posting_list_with_freqs_and_block_size(
                    doc_ids.clone(),
                    freqs,
                    vec![1; doc_ids.len()],
                    MAX_POSTING_BLOCK_SIZE,
                ),
                docs.len(),
            )
        });
        let mut wand = Wand::new(Operator::And, postings, &docs, VariedBm25ShapeScorer)
            .with_bulk_and_mode(BulkAndMode::Off)
            .with_score_first_and(is_score_first_enabled);
        let mut scored = Vec::new();
        while let Some((doc, cached_score)) = wand.next().unwrap() {
            let doc_id = doc.doc_id();
            let is_dense = doc_id < MAX_POSTING_BLOCK_SIZE as u64
                || doc_id > 2 * MAX_POSTING_BLOCK_SIZE as u64;
            assert_eq!(
                wand.and_candidate_score.is_some(),
                is_score_first_enabled && doc_id > 0 && is_dense
            );
            let score = if wand.and_candidate_score.is_some() {
                cached_score
            } else {
                wand.score_in_query_order(1)
            };
            scored.push((doc_id, score.to_bits()));
            if doc_id == 0 {
                assert!(wand.score_first_and_norm_k.get().is_none());
                wand.update_threshold(f32::from_bits(1), 1.0);
            }
        }
        scored
    };

    let classic = run(false);
    let score_first = run(true);
    assert_eq!(score_first, classic);
    assert_eq!(score_first.len(), total_docs as usize - 1);
}

#[test]
fn score_first_and_restores_shallow_probe_and_keeps_later_window() {
    let total_docs = 3 * MAX_POSTING_BLOCK_SIZE as u32;
    let mut docs = DocSet::default();
    for doc_id in 0..total_docs {
        docs.append(u64::from(doc_id), doc_id % 11 + 1);
    }
    let postings = (0..4)
        .map(|term| {
            let doc_ids = (0..total_docs).collect::<Vec<_>>();
            let freqs = doc_ids
                .iter()
                .map(|doc_id| {
                    if *doc_id + 1 == total_docs {
                        32
                    } else {
                        (*doc_id + term) % 4 + 1
                    }
                })
                .collect::<Vec<_>>();
            let doc_lengths = doc_ids
                .iter()
                .map(|doc_id| *doc_id % 11 + 1)
                .collect::<Vec<_>>();
            PostingIterator::with_query_weight(
                format!("t{term}"),
                term,
                term,
                0.002 + term as f32 * 0.000_31,
                generate_impact_posting_list_with_freqs_and_block_size(
                    doc_ids,
                    freqs,
                    doc_lengths,
                    MAX_POSTING_BLOCK_SIZE,
                ),
                docs.len(),
            )
        })
        .collect::<Vec<_>>();
    let mut wand = Wand::new(
        Operator::And,
        postings.into_iter(),
        &docs,
        VariedBm25ShapeScorer,
    )
    .with_score_first_and(true)
    .with_floor_mode(CompetitiveFloorMode::Inclusive);
    wand.threshold = f32::from_bits(1);

    assert_eq!(wand.and_advance_target(0), 0);
    assert!(wand.score_first_and_bounds_up_to.is_some());
    let internal_up_to = wand.up_to;
    let internal_suffix_bounds = wand.score_first_and_suffix_bounds.clone();
    let internal_dense_range = wand.score_first_and_dense_range;
    let (shallow_up_to, _) = wand.compound_shallow_bound(300);
    assert!(shallow_up_to >= 300);
    assert_eq!(wand.up_to, internal_up_to);
    assert_eq!(wand.score_first_and_suffix_bounds, internal_suffix_bounds);
    assert_eq!(wand.score_first_and_dense_range, internal_dense_range);

    wand.seek(300);
    assert!(wand.up_to.is_none());
    assert!(wand.score_first_and_suffix_bounds.is_empty());
    assert!(wand.score_first_and_dense_range.is_none());
    let mut best_score = 0.0_f32;
    let mut saw_late_winner = false;
    while let Some((doc, score)) = wand.next().unwrap() {
        let expected = wand.score_in_query_order(wand.documents.doc_length(&doc));
        assert_eq!(score.to_bits(), expected.to_bits());
        if score > best_score {
            best_score = score;
            wand.threshold = score;
        }
        if doc.doc_id() + 1 == u64::from(total_docs) {
            saw_late_winner = true;
        }
    }
    assert!(saw_late_winner);
    assert!(wand.and_window_stats.score_first_rejections > 0);
}

#[test]
fn score_first_and_reanchors_after_future_shallow_probe() {
    let run = |is_score_first_enabled| {
        let total_docs = 3 * MAX_POSTING_BLOCK_SIZE as u32;
        let postings = (0..4)
            .map(|term| {
                let doc_ids = (0..total_docs).collect::<Vec<_>>();
                let mut freqs = vec![1; doc_ids.len()];
                freqs[0] = 20;
                unit_length_impact_posting(term, doc_ids, freqs, total_docs as usize)
            })
            .collect::<Vec<_>>();
        let docs = unit_length_docs(total_docs as usize);
        let mut wand = Wand::new(
            Operator::And,
            postings.into_iter(),
            &docs,
            VariedBm25ShapeScorer,
        )
        .with_score_first_and(is_score_first_enabled);
        wand.threshold = 7.0;

        let _ = wand.compound_shallow_bound(600);
        let _ = wand.compound_shallow_bound(512);
        let (doc, cached_score) = wand.next().unwrap().unwrap();
        let score = if wand.and_candidate_score.is_some() {
            cached_score
        } else {
            wand.score_in_query_order(wand.documents.doc_length(&doc))
        };
        (doc.doc_id(), score.to_bits())
    };

    let classic = run(false);
    let score_first = run(true);
    assert_eq!(score_first, classic);
    assert_eq!(score_first.0, 0);
}

#[rstest]
fn score_first_and_inclusive_tie_uses_query_order(
    #[values(4_usize, 5_usize)] num_terms: usize,
    #[values(false, true)] is_quantized_scoring: bool,
) {
    let mut contributions = SmallVec::<[ScoreContribution; 8]>::new();
    let postings = (0..num_terms)
        .map(|term| {
            let position = (num_terms - term - 1) as u32;
            let doc_ids = (100 - term as u32..=100).collect::<Vec<_>>();
            let mut freqs = vec![1; doc_ids.len()];
            freqs[term] = term as u32 + 1;
            let query_weight = 0.001_882_293 + position as f32 * 0.000_173_205;
            contributions.push((
                (position, term as u32),
                query_weight * bm25_doc_weight_with_norm(term as u32 + 1, 0.4),
            ));
            PostingIterator::with_query_weight(
                format!("t{term}"),
                term as u32,
                position,
                query_weight,
                generate_impact_posting_list_with_freqs_and_block_size(
                    doc_ids.clone(),
                    freqs,
                    vec![1; doc_ids.len()],
                    MAX_POSTING_BLOCK_SIZE,
                ),
                101,
            )
        })
        .collect::<Vec<_>>();
    let expected_score = score_contributions_in_query_order(contributions);
    let mut docs = unit_length_docs(101);
    docs.set_quantized_scoring(is_quantized_scoring);
    let mut wand = Wand::new(
        Operator::And,
        postings.into_iter(),
        &docs,
        VariedBm25ShapeScorer,
    )
    .with_floor_mode(CompetitiveFloorMode::Inclusive);
    assert!(wand.score_first_and_enabled);
    wand.threshold = expected_score;

    let (doc, score) = wand.next().unwrap().unwrap();
    assert_eq!(doc.doc_id(), 100);
    assert_eq!(score.to_bits(), expected_score.to_bits());
}

#[rstest]
fn score_first_and_handles_u32_max_doc_id(#[values(0.0, 0.1)] floor: f32) {
    let postings = (0..4)
        .map(|term| unit_length_impact_posting(term, vec![u32::MAX], vec![1], 1))
        .collect::<Vec<_>>();
    let docs = CostOnlyDocuments {
        total_docs: 1,
        visible_cost_upper_bound: 1,
    };
    let mut wand = Wand::new(
        Operator::And,
        postings.into_iter(),
        &docs,
        VariedBm25ShapeScorer,
    )
    .with_score_first_and(true);
    wand.threshold = floor;

    assert_eq!(
        wand.next().unwrap().unwrap().0.doc_id(),
        u64::from(u32::MAX)
    );
    assert!(wand.score_first_and_dense_range.is_none());
    if floor == 0.0 {
        assert_eq!(wand.up_to, Some(TERMINATED_DOC_ID));
    }
    assert!(wand.next().unwrap().is_none());
}

#[rstest]
#[case::local_heap("local")]
#[case::shared_floor("shared")]
#[case::compound_floor("compound")]
fn score_first_and_replaces_zero_floor_window_after_floor_update(
    #[case] floor_source: &str,
    #[values(4, 5)] num_terms: u32,
) {
    let total_docs = MAX_POSTING_BLOCK_SIZE as u32 + 3;
    let docs = unit_length_docs(total_docs as usize);
    let postings = (0..num_terms)
        .map(|term| {
            let doc_ids = (0..total_docs).collect::<Vec<_>>();
            unit_length_impact_posting(term, doc_ids, vec![1; total_docs as usize], docs.len())
        })
        .collect();
    let params = FtsSearchParams::default();
    let metrics = NoOpMetricsCollector;
    let mut cursor = WandCursor::new(
        Operator::And,
        postings,
        &docs,
        Arc::new(MemBM25Scorer::new(
            total_docs as u64,
            total_docs as usize,
            std::collections::HashMap::new(),
        )),
        &params,
        &metrics,
    );
    assert!(cursor.wand.score_first_and_enabled);
    assert_eq!(cursor.next().unwrap(), Some(0));
    assert_eq!(cursor.wand.up_to, Some(TERMINATED_DOC_ID));
    assert_eq!(cursor.next().unwrap(), Some(1));
    assert_eq!(cursor.wand.up_to, Some(TERMINATED_DOC_ID));
    assert_eq!(cursor.advance(10).unwrap(), Some(10));
    assert_eq!(cursor.wand.up_to, Some(TERMINATED_DOC_ID));
    let current_score = cursor.current_score().unwrap();
    let shallow_up_to = cursor
        .advance_shallow(MAX_POSTING_BLOCK_SIZE as u64)
        .unwrap();
    let shallow_bound = cursor.score_upper_bound(shallow_up_to).unwrap();
    assert_eq!(cursor.wand.up_to, Some(TERMINATED_DOC_ID));
    assert!(cursor.wand.score_first_and_suffix_bounds.is_empty());
    assert!(cursor.wand.score_first_and_norm_k.get().is_none());

    let floor = f32::from_bits(1);
    match floor_source {
        "local" => cursor.wand.update_threshold(floor, 1.0),
        "shared" => {
            cursor.wand.shared_threshold = Some(Arc::new(AtomicU32::new(floor.to_bits())));
            cursor.wand.raise_to_shared_floor(1.0);
        }
        "compound" => cursor.set_min_competitive_score(floor).unwrap(),
        _ => unreachable!(),
    }
    assert!(cursor.wand.up_to.is_none());
    assert_eq!(cursor.doc(), Some(10));
    assert_eq!(
        cursor.current_score().unwrap().to_bits(),
        current_score.to_bits()
    );
    assert_eq!(
        cursor.score_upper_bound(shallow_up_to).unwrap(),
        shallow_bound
    );
    assert_eq!(cursor.next().unwrap(), Some(11));
    assert_eq!(cursor.wand.up_to, Some(MAX_POSTING_BLOCK_SIZE as u64 - 1));
    assert!(cursor.wand.score_first_and_dense_range.is_some());
    assert!(cursor.wand.and_candidate_score.is_some());
    assert_eq!(
        cursor.current_score().unwrap().to_bits(),
        current_score.to_bits()
    );
}

#[rstest]
fn score_first_and_underfilled_window_stays_unbounded(
    #[values(4, 5)] num_terms: u32,
    #[values(0, 5)] num_matches: u32,
) {
    let sparse_end = 5 * MAX_POSTING_BLOCK_SIZE as u32;
    let total_docs = sparse_end + num_matches;
    let docs = unit_length_docs(total_docs as usize);
    let make_postings = || {
        (0..num_terms)
            .map(|term| {
                let doc_ids = (term..sparse_end)
                    .step_by(num_terms as usize)
                    .chain(sparse_end..total_docs)
                    .collect::<Vec<_>>();
                unit_length_impact_posting(
                    term,
                    doc_ids.clone(),
                    vec![1; doc_ids.len()],
                    docs.len(),
                )
            })
            .collect::<Vec<_>>()
    };
    let run = |is_eligible| {
        let mut wand = Wand::new(
            Operator::And,
            make_postings().into_iter(),
            &docs,
            VariedBm25ShapeScorer,
        )
        .with_bulk_and_mode(BulkAndMode::Off);
        wand.score_first_and_enabled = is_eligible;
        let mut scored = Vec::new();
        while let Some((doc, _)) = wand.next().unwrap() {
            scored.push((doc.doc_id(), wand.score_in_query_order(1).to_bits()));
            if is_eligible {
                assert_eq!(wand.up_to, Some(TERMINATED_DOC_ID));
            }
        }
        if is_eligible {
            assert_eq!(wand.up_to, Some(TERMINATED_DOC_ID));
        }
        assert!(wand.score_first_and_suffix_bounds.is_empty());
        assert!(wand.score_first_and_dense_range.is_none());
        assert!(wand.score_first_and_norm_k.get().is_none());
        assert!(
            wand.lead
                .iter()
                .all(|posting| posting.impact_bound_computations() == 0)
        );
        scored
    };
    let expected = run(false);
    assert_eq!(run(true), expected);
    assert_eq!(expected.len(), num_matches as usize);

    let mut wand = Wand::new(
        Operator::And,
        make_postings().into_iter(),
        &docs,
        VariedBm25ShapeScorer,
    )
    .with_bulk_and_mode(BulkAndMode::Off);
    let results = wand
        .search(
            &FtsSearchParams::new().with_limit(Some(10)),
            &NoOpMetricsCollector,
        )
        .unwrap();
    assert_eq!(results.len(), num_matches as usize);
    assert_eq!(wand.threshold, 0.0);
    assert_eq!(wand.up_to, Some(TERMINATED_DOC_ID));
    let mut actual_docs = results
        .into_iter()
        .map(|candidate| candidate.document)
        .collect::<Vec<_>>();
    actual_docs.sort_unstable();
    assert_eq!(
        actual_docs,
        expected.into_iter().map(|(doc, _)| doc).collect::<Vec<_>>()
    );
}

#[test]
fn score_first_and_requires_bm25_shape_and_positive_floor() {
    let doc_ids = vec![0, 1];
    let make_postings = || {
        (0..4)
            .map(|term| {
                unit_length_impact_posting(term, doc_ids.clone(), vec![1, 1], doc_ids.len())
            })
            .collect::<Vec<_>>()
    };
    let docs = unit_length_docs(2);
    let mut wand = Wand::new(
        Operator::And,
        make_postings().into_iter(),
        &docs,
        UnitScorer,
    );

    assert!(!wand.score_first_and_enabled);
    assert_eq!(wand.next().unwrap().unwrap().0.doc_id(), 0);
    assert_eq!(wand.and_window_stats.score_first_rejections, 0);

    let scored = Arc::new(AtomicUsize::new(0));
    let mut underfilled = Wand::new(
        Operator::And,
        make_postings().into_iter(),
        &docs,
        CountingBm25ShapeScorer { scored },
    );
    assert!(underfilled.score_first_and_enabled);
    let results = underfilled
        .search(
            &FtsSearchParams::new().with_limit(Some(3)),
            &NoOpMetricsCollector,
        )
        .unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(underfilled.threshold, 0.0);
    assert_eq!(underfilled.and_window_stats.score_first_rejections, 0);
    assert!(underfilled.score_first_and_dense_range.is_none());
    assert!(underfilled.score_first_and_norm_k.get().is_none());

    let mut non_finite = Wand::new(
        Operator::And,
        make_postings().into_iter(),
        &docs,
        NonFiniteBoundBm25ShapeScorer,
    );
    assert!(non_finite.score_first_and_enabled);
    non_finite.threshold = 1.0;
    assert_eq!(non_finite.next().unwrap().unwrap().0.doc_id(), 0);
    assert!(non_finite.score_first_and_suffix_bounds.is_empty());
    assert!(non_finite.score_first_and_bounds_up_to.is_none());
    assert!(non_finite.score_first_and_dense_range.is_none());
    assert!(non_finite.and_candidate_score.is_none());
    assert_eq!(non_finite.and_window_stats.score_first_rejections, 0);
}

#[test]
fn score_first_and_selection_stays_within_modern_four_or_five_clause_path() {
    let docs = unit_length_docs(2);
    let modern = |num_terms, block_size| {
        (0..num_terms)
            .map(|term| {
                PostingIterator::with_query_weight(
                    format!("t{term}"),
                    term,
                    term,
                    1.0,
                    generate_impact_posting_list_with_freqs_and_block_size(
                        vec![0, 1],
                        vec![1, 1],
                        vec![1, 1],
                        block_size,
                    ),
                    docs.len(),
                )
            })
            .collect::<Vec<_>>()
    };

    for num_terms in [2, 3] {
        let wand = Wand::new(
            Operator::And,
            modern(num_terms, MAX_POSTING_BLOCK_SIZE).into_iter(),
            &docs,
            VariedBm25ShapeScorer,
        );
        assert!(!wand.score_first_and_enabled);
    }

    let plain = (0..4).map(|term| {
        PostingIterator::with_query_weight(
            format!("t{term}"),
            term,
            term,
            1.0,
            generate_posting_list(vec![0, 1], 1.0, None, false),
            docs.len(),
        )
    });
    assert!(!Wand::new(Operator::And, plain, &docs, VariedBm25ShapeScorer).score_first_and_enabled);

    let no_impact_non_256 = (0..4).map(|term| {
        PostingIterator::with_query_weight(
            format!("t{term}"),
            term,
            term,
            1.0,
            generate_posting_list(vec![0, 1], 1.0, None, true),
            docs.len(),
        )
    });
    assert!(
        !Wand::new(
            Operator::And,
            no_impact_non_256,
            &docs,
            VariedBm25ShapeScorer,
        )
        .score_first_and_enabled
    );

    assert!(
        !Wand::new(
            Operator::And,
            modern(4, crate::scalar::inverted::LEGACY_BLOCK_SIZE).into_iter(),
            &docs,
            VariedBm25ShapeScorer,
        )
        .score_first_and_enabled
    );
}
