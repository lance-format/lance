// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

struct AdjacentScoreScorer;

impl Scorer for AdjacentScoreScorer {
    fn query_weight(&self, _token: &str) -> f32 {
        1.0
    }

    fn doc_weight(&self, freq: u32, _doc_tokens: u32) -> f32 {
        match freq {
            1 => f32::from_bits(1.0_f32.to_bits() - 1),
            2 => 1.0,
            _ => unreachable!("test only defines two score buckets"),
        }
    }
}

#[test]
fn inclusive_floor_keeps_tie_without_admitting_lower_ulp() {
    let posting = PostingIterator::with_query_weight(
        "term".to_owned(),
        0,
        0,
        1.0,
        generate_posting_list_with_freqs(vec![0, 1], vec![1, 2], 1.0, None, false),
        2,
    );
    let mut docs = DocSet::default();
    docs.append(0, 1);
    docs.append(1, 1);
    let mut wand = Wand::new(
        Operator::Or,
        std::iter::once(posting),
        &docs,
        AdjacentScoreScorer,
    )
    .with_floor_mode(CompetitiveFloorMode::Inclusive);
    wand.update_threshold(1.0, 1.0);

    let (doc, score) = wand.next().unwrap().unwrap();
    assert_eq!(doc.doc_id(), 1);
    assert_eq!(score, 1.0);
}

#[test]
fn test_wand_skip_to_next_block() {
    let mut docs = DocSet::default();
    for i in 0..201 {
        docs.append(i as u64, 1);
    }

    let large_posting_docs1: Vec<u32> = (0..=200).collect();

    let postings = vec![
        PostingIterator::new(
            String::from("full"),
            0,
            0,
            generate_posting_list(large_posting_docs1, 1.0, Some(vec![0.5, 0.5]), true),
            docs.len(),
        ),
        PostingIterator::new(
            String::from("text"),
            1,
            1,
            generate_posting_list(vec![0], 1.0, Some(vec![0.5]), true),
            docs.len(),
        ),
    ];

    let bm25 = IndexBM25Scorer::new(std::iter::empty());
    let mut wand = Wand::new(Operator::Or, postings.into_iter(), &docs, bm25);

    // set a threshold that the sum of max scores can hit,
    // but the sum of block max scores is less than the threshold,
    wand.threshold = 1.5;

    let result = wand.search(&FtsSearchParams::default(), &NoOpMetricsCollector);

    assert!(result.is_ok());
}

#[test]
fn test_or_single_term_block_skip_matches_and() {
    // Hot docs occupy the middle block; the flanking blocks score far below
    // the threshold. A single-term disjunction must skip them yet return what
    // the conjunctive path returns.
    let total = 3 * BLOCK_SIZE as u32;
    let hot = BLOCK_SIZE as u32..BLOCK_SIZE as u32 + 12;

    let mut docs = DocSet::default();
    for row_id in 0..total {
        // hot docs get distinct scores 1/1..1/12; the rest score 0.001
        let doc_tokens = if hot.contains(&row_id) {
            row_id - hot.start + 1
        } else {
            1000
        };
        docs.append(row_id as u64, doc_tokens);
    }

    let params = FtsSearchParams::new().with_limit(Some(10));
    let run = |operator| {
        let scored = Arc::new(AtomicUsize::new(0));
        let posting = PostingIterator::with_query_weight(
            String::from("term"),
            0,
            0,
            1.0,
            generate_posting_list(
                (0..total).collect(),
                1.0,
                Some(vec![0.001, 1.0, 0.001]),
                true,
            ),
            docs.len(),
        );
        let mut wand = Wand::new(
            operator,
            std::iter::once(posting),
            &docs,
            CountingScorer {
                scored: scored.clone(),
            },
        );
        let hits = wand.search(&params, &NoOpMetricsCollector).unwrap();
        let mut row_ids = hits.iter().map(|hit| hit.document).collect::<Vec<_>>();
        row_ids.sort_unstable();
        (row_ids, scored.load(Ordering::Relaxed))
    };

    let (or_hits, or_scored) = run(Operator::Or);
    let (and_hits, _) = run(Operator::And);

    let expected = (hot.start..hot.start + 10)
        .map(u64::from)
        .collect::<Vec<_>>();
    assert_eq!(or_hits, expected, "OR must return the top-k");
    assert_eq!(or_hits, and_hits, "OR and AND must agree for a single term");
    // Without pruning OR scores all `total` docs; with it the cold blocks are skipped.
    assert!(
        or_scored <= 2 * BLOCK_SIZE,
        "expected pruning to skip a block, but scored {or_scored} of {total}",
    );
}

#[rstest]
fn test_or_search_records_comparisons(#[values(false, true)] is_compressed: bool) {
    let mut docs = DocSet::default();
    for row_id in 0..6 {
        docs.append(row_id, 1);
    }

    let postings = vec![
        PostingIterator::with_query_weight(
            String::from("alpha"),
            0,
            0,
            1.0,
            generate_posting_list(vec![0, 1, 4], 1.0, None, is_compressed),
            docs.len(),
        ),
        PostingIterator::with_query_weight(
            String::from("beta"),
            1,
            1,
            1.0,
            generate_posting_list(vec![1, 2, 5], 1.0, None, is_compressed),
            docs.len(),
        ),
    ];

    let mut wand = Wand::new(Operator::Or, postings.into_iter(), &docs, UnitScorer);
    let metrics = LocalMetricsCollector::default();
    let candidates = wand.search(&FtsSearchParams::default(), &metrics).unwrap();

    assert_eq!(sorted_candidate_row_ids(candidates), vec![0, 1, 2, 4, 5]);
    assert!(metrics.comparisons.load(Ordering::Relaxed) > 0);
}

#[test]
fn test_up_to_refreshes_on_first_candidate() {
    let mut docs = DocSet::default();
    for i in 0..=(BLOCK_SIZE as u64 + 1) {
        docs.append(i, 1);
    }

    let postings = vec![PostingIterator::with_query_weight(
        String::from("term"),
        0,
        0,
        1.0,
        generate_posting_list(
            (0..=(BLOCK_SIZE as u32 + 1)).collect(),
            1.0,
            Some(vec![1.0, 1.0]),
            true,
        ),
        docs.len(),
    )];

    let mut wand = Wand::new(Operator::Or, postings.into_iter(), &docs, UnitScorer);
    assert!(wand.up_to.is_none());
    let _ = wand.next().unwrap();
    assert!(wand.up_to.is_some());
}

#[test]
fn test_or_push_back_lead_uses_current_block_max_for_tail_bound() {
    let total = 2 * BLOCK_SIZE as u32;
    let mut docs = DocSet::default();
    for doc_id in 0..total {
        docs.append(doc_id as u64, 1);
    }

    let postings = vec![PostingIterator::with_query_weight(
        String::from("term"),
        0,
        0,
        1.0,
        generate_posting_list((0..total).collect(), 10.0, Some(vec![1.0, 10.0]), true),
        docs.len(),
    )];
    let mut wand = Wand::new(Operator::Or, postings.into_iter(), &docs, UnitScorer);
    wand.threshold = 1.5;

    wand.update_max_scores(0);
    wand.move_head_doc_to_lead(0);
    assert_eq!(wand.up_to, Some((BLOCK_SIZE - 1) as u64));

    wand.push_back_leads(1);

    assert_eq!(wand.tail.len(), 1);
    assert!(
        (wand.tail_max_score - 1.0).abs() < 1e-6,
        "tail should use the current block max, got {}",
        wand.tail_max_score
    );
    assert!(wand.head_doc().is_none());
}

#[test]
fn test_or_push_back_lead_falls_back_after_block_window_expires() {
    let total = 2 * BLOCK_SIZE as u32;
    let mut docs = DocSet::default();
    for doc_id in 0..total {
        docs.append(doc_id as u64, 1);
    }

    let freqs = (0..total)
        .map(|doc_id| if doc_id >= BLOCK_SIZE as u32 { 10 } else { 1 })
        .collect::<Vec<_>>();
    let mut posting = PostingIterator::with_query_weight(
        String::from("term"),
        0,
        0,
        1.0,
        generate_posting_list_with_freqs(
            (0..total).collect(),
            freqs,
            10.0,
            Some(vec![1.0, 10.0]),
            true,
        ),
        docs.len(),
    );
    posting.next((BLOCK_SIZE - 1) as u64);
    let mut wand = Wand::new(Operator::Or, std::iter::once(posting), &docs, UnitScorer);
    wand.threshold = 1.5;

    let block_end = (BLOCK_SIZE - 1) as u64;
    wand.update_max_scores(block_end);
    wand.move_head_doc_to_lead(block_end);
    assert_eq!(wand.up_to, Some(block_end));

    wand.push_back_leads(BLOCK_SIZE as u64);

    assert!(wand.tail.is_empty());
    assert_eq!(wand.head_doc(), Some(BLOCK_SIZE as u64));
    let candidate = wand.next().unwrap().unwrap();
    assert_eq!(candidate.0.doc_id(), BLOCK_SIZE as u64);
}

#[rstest]
#[case::head_jump(false)]
#[case::window_skip(true)]
fn or_window_refresh_recovers_competitive_tail_documents(
    #[case] is_window_skip: bool,
    #[values(CompetitiveFloorMode::Exclusive, CompetitiveFloorMode::Inclusive)]
    floor_mode: CompetitiveFloorMode,
) {
    let block_size = MAX_POSTING_BLOCK_SIZE as u32;
    let hot_doc = block_size;
    let next_head_doc = 4 * block_size;
    let mut docs = DocSet::default();
    for doc_id in 0..=next_head_doc {
        docs.append(u64::from(doc_id), 1);
    }

    let parked_docs = (0..block_size).chain([hot_doc]).collect::<Vec<_>>();
    let mut parked_freqs = vec![1; parked_docs.len()];
    *parked_freqs.last_mut().unwrap() = 10;
    let driving_docs = if is_window_skip {
        (0..block_size).chain([next_head_doc]).collect::<Vec<_>>()
    } else {
        vec![0, next_head_doc]
    };
    let mut driving_freqs = vec![1; driving_docs.len()];
    driving_freqs[0] = 6;
    *driving_freqs.last_mut().unwrap() = 8;
    let postings = [(parked_docs, parked_freqs), (driving_docs, driving_freqs)]
        .into_iter()
        .enumerate()
        .map(|(term, (doc_ids, freqs))| {
            let doc_lengths = vec![1; doc_ids.len()];
            PostingIterator::with_query_weight(
                format!("term{term}"),
                term as u32,
                term as u32,
                1.0,
                generate_impact_posting_list_with_freqs_and_block_size(
                    doc_ids,
                    freqs,
                    doc_lengths,
                    MAX_POSTING_BLOCK_SIZE,
                ),
                docs.len(),
            )
        });
    let mut wand = Wand::new(Operator::Or, postings, &docs, UnitScorer).with_floor_mode(floor_mode);

    let (first_doc, _) = wand.next().unwrap().unwrap();
    assert_eq!(first_doc.doc_id(), 0);
    wand.advance_all_tail(0, None, None);
    assert_eq!(wand.score_in_query_order(1), 7.0);
    wand.threshold = 6.5;
    wand.push_back_leads(1);
    assert_eq!(wand.tail.len(), 1);
    assert_eq!(wand.up_to, Some(u64::from(hot_doc - 1)));

    if is_window_skip {
        // A shared collector may raise the floor between candidates. The
        // old window is now dead, but the parked clause's next block wins.
        wand.threshold = 7.5;
        assert!(wand.or_block_window_max() < wand.threshold);
    }

    let mut competitive_docs = Vec::with_capacity(2);
    while let Some((doc, _)) = wand.next().unwrap() {
        wand.advance_all_tail(doc.doc_id(), None, None);
        competitive_docs.push((doc.doc_id(), wand.score_in_query_order(1)));
        wand.push_back_leads(doc.doc_id() + 1);
    }
    assert_eq!(
        competitive_docs,
        vec![(u64::from(hot_doc), 10.0), (u64::from(next_head_doc), 8.0)],
    );
}

#[test]
fn test_non_positive_threshold_advances_without_impact_bound_scoring() {
    let mut docs = DocSet::default();
    for doc_id in 0..3 {
        docs.append(doc_id, 1);
    }
    let make_posting = || {
        PostingIterator::with_query_weight(
            String::from("term"),
            0,
            0,
            1.0,
            generate_impact_posting_list_with_freqs(vec![0, 1, 2], vec![1, 1, 1], vec![1, 1, 1]),
            docs.len(),
        )
    };
    let scored = Arc::new(AtomicUsize::new(0));

    let mut wand = Wand::new(
        Operator::Or,
        std::iter::once(make_posting()),
        &docs,
        CountingScorer {
            scored: scored.clone(),
        },
    );
    wand.move_head_before_target_to_tail(1);
    assert_eq!(wand.head_doc(), Some(1));
    assert!(wand.tail.is_empty());
    assert_eq!(scored.load(Ordering::Relaxed), 0);

    let mut wand = Wand::new(
        Operator::Or,
        std::iter::once(make_posting()),
        &docs,
        CountingScorer {
            scored: scored.clone(),
        },
    );
    wand.move_head_doc_to_lead(0);
    wand.push_back_leads(1);
    assert_eq!(wand.head_doc(), Some(1));
    assert!(wand.tail.is_empty());
    assert_eq!(scored.load(Ordering::Relaxed), 0);
}

#[test]
fn test_or_plain_tail_does_not_advance_headless_window() {
    let mut docs = DocSet::default();
    for doc_id in 0..4 {
        docs.append(doc_id, 1);
    }

    let postings = vec![PostingIterator::with_query_weight(
        String::from("term"),
        0,
        0,
        1.0,
        generate_posting_list(vec![0, 1, 2, 3], 1.0, None, false),
        docs.len(),
    )];
    let mut wand = Wand::new(Operator::Or, postings.into_iter(), &docs, UnitScorer);
    wand.threshold = 2.0;

    wand.update_max_scores(0);
    wand.move_head_doc_to_lead(0);
    wand.push_back_leads(1);

    assert_eq!(wand.tail.len(), 1);
    assert!(wand.head_doc().is_none());
    assert!(!wand.advance_tail_to_next_or_window());
}

#[test]
fn test_or_headless_tail_window_scans_past_final_top_tail() {
    let total = 3 * BLOCK_SIZE as u32;
    let mut docs = DocSet::default();
    for doc_id in 0..total {
        docs.append(doc_id as u64, 1);
    }

    let future_docs = (0..BLOCK_SIZE as u32)
        .chain(2 * BLOCK_SIZE as u32..3 * BLOCK_SIZE as u32)
        .collect::<Vec<_>>();
    let mut future_freqs = vec![1; future_docs.len()];
    future_freqs[0] = 4;
    future_freqs[BLOCK_SIZE] = 20;
    let postings = vec![
        PostingIterator::with_query_weight(
            String::from("future"),
            0,
            0,
            1.0,
            generate_posting_list_with_freqs(
                future_docs,
                future_freqs,
                20.0,
                Some(vec![4.0, 20.0]),
                true,
            ),
            docs.len(),
        ),
        PostingIterator::with_query_weight(
            String::from("final_tail"),
            1,
            1,
            1.0,
            generate_posting_list_with_freqs(vec![0], vec![6], 6.0, Some(vec![6.0]), true),
            docs.len(),
        ),
        PostingIterator::with_query_weight(
            String::from("booster"),
            2,
            2,
            1.0,
            generate_posting_list_with_freqs(vec![0], vec![7], 7.0, Some(vec![7.0]), true),
            docs.len(),
        ),
    ];

    let mut wand = Wand::new(Operator::Or, postings.into_iter(), &docs, UnitScorer);
    let result = wand
        .search(
            &FtsSearchParams::new().with_limit(Some(1)),
            &NoOpMetricsCollector,
        )
        .unwrap();

    assert_eq!(
        sorted_candidate_row_ids(result),
        vec![(2 * BLOCK_SIZE) as u64]
    );
}

#[test]
fn test_or_impact_level1_window_skips_low_group_with_single_score() {
    let total = (IMPACT_LEVEL1_BLOCKS + 1) * BLOCK_SIZE;
    let target = (IMPACT_LEVEL1_BLOCKS * BLOCK_SIZE) as u64;
    let mut docs = DocSet::default();
    for doc_id in 0..total as u64 {
        docs.append(doc_id, 1);
    }

    let doc_ids = (0..total as u32).collect::<Vec<_>>();
    let freqs = doc_ids
        .iter()
        .map(|doc_id| if u64::from(*doc_id) < target { 1 } else { 10 })
        .collect::<Vec<_>>();
    let posting_list = generate_impact_posting_list_with_freqs(doc_ids, freqs, vec![1; total]);
    let mut probe = PostingIterator::with_query_weight(
        String::from("term"),
        0,
        0,
        1.0,
        posting_list.clone(),
        docs.len(),
    );
    probe.shallow_next(0);
    let counting_scorer = CountingScorer {
        scored: Arc::new(AtomicUsize::new(0)),
    };
    let (group_up_to, group_score) = probe.impact_group_bound(&counting_scorer).unwrap();
    assert_eq!(group_up_to, target - 1);
    assert_eq!(group_score, 1.0);
    // A window ending past the current block must answer from level1.
    assert_eq!(
        probe.window_max_score(Some(target - 1), &counting_scorer),
        1.0
    );

    let posting = PostingIterator::with_query_weight(
        String::from("term"),
        0,
        0,
        1.0,
        posting_list,
        docs.len(),
    );
    let scored = Arc::new(AtomicUsize::new(0));
    let mut wand = Wand::new(
        Operator::Or,
        std::iter::once(posting),
        &docs,
        CountingScorer {
            scored: scored.clone(),
        },
    );
    wand.threshold = 2.0;

    let (candidate, score) = wand.next().unwrap().unwrap();
    assert_eq!(candidate.doc_id(), target);
    assert_eq!(score, 10.0);
    // The doc-weight bounds bake exactly once (one doc_weight call per
    // frontier pair across all entries); beyond that only the returned
    // candidate is scored.
    let total_entries = (IMPACT_LEVEL1_BLOCKS + 1) + 2;
    assert!(
        scored.load(Ordering::Relaxed) <= total_entries + 8,
        "bounds should be baked once instead of recomputed per window; scored={}",
        scored.load(Ordering::Relaxed)
    );
}

#[test]
fn or_search_finds_parked_union_hit_in_a_later_block_of_the_window() {
    // A query-time fuzzy union (grouped, 128-doc blocks, no impacts) parks
    // in the tail while the sparse clause's wide block sets the window end.
    // The union's best document sits in a later block of that window, so a
    // bound taken from its current block alone skips it and the top-1 is wrong.
    let block = crate::scalar::inverted::LEGACY_BLOCK_SIZE;
    let codec = crate::scalar::inverted::PostingTailCodec::VarintDelta;
    let non_impact_list = |doc_ids: &[u32], freqs: &[u32]| {
        let block_maxes = freqs
            .chunks(block)
            .map(|chunk| chunk.iter().copied().max().unwrap() as f32)
            .collect::<Vec<_>>();
        let max_score = block_maxes.iter().copied().fold(0.0, f32::max);
        let blocks = compress_posting_list_with_tail_codec_and_block_size(
            doc_ids.len(),
            doc_ids.iter(),
            freqs.iter(),
            block_maxes.into_iter(),
            codec,
            block,
        )
        .unwrap();
        PostingList::Compressed(CompressedPostingList::new(
            blocks,
            max_score,
            doc_ids.len() as u32,
            codec,
            block,
            None,
            None,
        ))
    };

    let block_docs = block as u32;
    let num_docs = 6 * block_docs;
    // Dense union over six blocks; every block max is 1 except block 2.
    let hot_doc = 2 * block_docs + 44;
    let union_docs = (0..num_docs).collect::<Vec<_>>();
    let union_freqs = union_docs
        .iter()
        .map(|&doc| if doc == hot_doc { 12 } else { 1 })
        .collect::<Vec<_>>();
    // Sparse clause: a dense first block whose doc 0 sets the floor, then
    // a stride-4 block spanning union blocks 1..=4, then a short tail. It
    // carries no scorer-bound flag so its block maxes stay finite under
    // `UnitScorer`.
    let sparse_docs = (0..block_docs)
        .chain((0..block_docs).map(|i| block_docs + 4 * i))
        .chain([5 * block_docs, 5 * block_docs + 4])
        .collect::<Vec<_>>();
    let sparse_freqs = sparse_docs
        .iter()
        .map(|&doc| if doc == 0 { 9 } else { 1 })
        .collect::<Vec<_>>();

    let mut docs = DocSet::default();
    let mut exhaustive = vec![0_u32; num_docs as usize];
    for doc_id in 0..num_docs {
        docs.append(u64::from(doc_id), 1);
    }
    for (doc_ids, freqs) in [(&union_docs, &union_freqs), (&sparse_docs, &sparse_freqs)] {
        for (&doc_id, &freq) in doc_ids.iter().zip(freqs.iter()) {
            exhaustive[doc_id as usize] += freq;
        }
    }
    let limit = 1;
    let mut expected = (0..num_docs).map(u64::from).collect::<Vec<_>>();
    expected.sort_by_key(|&doc| (std::cmp::Reverse(exhaustive[doc as usize]), doc));
    expected.truncate(limit);
    assert_eq!(expected, vec![u64::from(hot_doc)]);

    let union_list = non_impact_list(&union_docs, &union_freqs);
    let grouped_terms =
        Arc::<[GroupedTermScorer]>::from([GroupedTermScorer::new(1.0, &union_list)]);
    let union = PostingIterator::with_query_weight(
        String::from("union"),
        0,
        0,
        1.0,
        union_list,
        docs.len(),
    )
    .with_grouped_terms(grouped_terms);
    let sparse = PostingIterator::with_query_weight(
        String::from("sparse"),
        1,
        1,
        1.0,
        non_impact_list(&sparse_docs, &sparse_freqs),
        docs.len(),
    );
    let mut wand = Wand::new(Operator::Or, [union, sparse].into_iter(), &docs, UnitScorer);
    let hits = wand
        .search(
            &FtsSearchParams::default().with_limit(Some(limit)),
            &NoOpMetricsCollector,
        )
        .unwrap();
    let found = hits
        .iter()
        .map(|hit| hit.posting_doc_id)
        .collect::<Vec<_>>();
    assert_eq!(
        found, expected,
        "WAND OR top-{limit} must match exhaustive scoring"
    );
}

#[rstest]
fn test_wand_batches_lagging_iterators(#[values(false, true)] is_compressed: bool) {
    let mut docs = DocSet::default();
    for i in 0..16 {
        docs.append(i as u64, 1);
    }

    let postings = vec![
        PostingIterator::new(
            String::from("a"),
            0,
            0,
            generate_posting_list(vec![1, 10], 1.0, None, is_compressed),
            docs.len(),
        ),
        PostingIterator::new(
            String::from("b"),
            1,
            1,
            generate_posting_list(vec![2, 10], 1.0, None, is_compressed),
            docs.len(),
        ),
        PostingIterator::new(
            String::from("c"),
            2,
            2,
            generate_posting_list(vec![10], 1.0, None, is_compressed),
            docs.len(),
        ),
    ];

    let mut wand = Wand::new(Operator::Or, postings.into_iter(), &docs, UnitScorer);
    wand.threshold = 2.5;

    let candidate = wand.next().unwrap().unwrap();
    assert_eq!(candidate.0.doc_id(), 10);
    assert_eq!(wand.lead.len(), 3);
}
