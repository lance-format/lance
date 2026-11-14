// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

#[rstest]
fn bulk_and_and_classic_publish_identical_query_order_score_bits(
    #[values(3, 4, 5, 8, 16)] num_clauses: usize,
) {
    let contributions = [0.002_972_301_6_f32, 0.001_982_450_7_f32, 0.001_882_293_f32]
        .into_iter()
        .cycle()
        .take(num_clauses)
        .collect::<Vec<_>>();
    let bounds = [0.002_99_f32, 0.001_99_f32, 0.003_f32];
    // Different posting lengths force cost order 1, 2, 0 instead of query
    // order 0, 1, 2.
    let clause_docs = [vec![0, 1, 2], vec![0], vec![0, 1]];
    let mut docs = DocSet::default();
    for doc_id in 0..3 {
        docs.append(doc_id, 1);
    }

    let run = |mode| {
        let postings = contributions
            .iter()
            .copied()
            .zip(bounds.into_iter().cycle())
            .zip(clause_docs.iter().cycle())
            .enumerate()
            .map(|(position, ((query_weight, max_score), doc_ids))| {
                PostingIterator::with_query_weight(
                    format!("t{position}"),
                    position as u32,
                    position as u32,
                    query_weight,
                    generate_posting_list(doc_ids.clone(), max_score, None, true),
                    docs.len(),
                )
            })
            .collect::<Vec<_>>();
        let shared_floor = Arc::new(AtomicU32::new(0.0_f32.to_bits()));
        let mut wand = Wand::new(Operator::And, postings.into_iter(), &docs, UnitScorer)
            .with_bulk_and_mode(mode)
            .with_shared_threshold(shared_floor.clone());
        let hits = wand
            .search(
                &FtsSearchParams::default().with_limit(Some(1)),
                &NoOpMetricsCollector,
            )
            .unwrap();
        (
            hits,
            shared_floor.load(Ordering::Relaxed),
            wand.bulk_and_searches > 0,
        )
    };

    let (bulk, bulk_score, bulk_used) = run(BulkAndMode::On);
    let (classic, classic_score, classic_used) = run(BulkAndMode::Off);
    assert!(bulk_used);
    assert!(!classic_used);
    assert_eq!(bulk.len(), 1);
    assert_eq!(classic.len(), 1);
    assert_eq!(bulk[0].document, 0);
    assert_eq!(classic[0].document, 0);
    assert_eq!(
        bulk_score,
        contributions
            .iter()
            .fold(0.0_f32, |sum, value| sum + value)
            .to_bits()
    );
    assert_eq!(classic_score, bulk_score);
}

/// The bulk conjunction path must return exactly the classic loop's
/// results — same docs, freqs, and doc lengths — for both plain AND and
/// phrase queries, across multi-block lists with heap/threshold pruning
/// in play.
#[rstest]
#[case::and_k10(false, 0, 10, 3)]
#[case::and_k3(false, 0, 3, 3)]
#[case::and_two_clauses(false, 0, 10, 2)]
#[case::and_one_clause(false, 0, 10, 1)]
#[case::and_four_clauses(false, 0, 10, 4)]
#[case::and_five_clauses(false, 0, 10, 5)]
#[case::and_six_clauses(false, 0, 10, 6)]
#[case::and_eight_clauses(false, 0, 10, 8)]
#[case::and_twelve_clauses(false, 0, 10, 12)]
#[case::and_sixteen_clauses(false, 0, 10, 16)]
#[case::phrase_k10(true, 0, 10, 3)]
#[case::phrase_k3(true, 0, 3, 3)]
#[case::phrase_slop_three(true, 3, 10, 3)]
#[case::phrase_two_clauses(true, 0, 10, 2)]
#[case::phrase_four_clauses(true, 0, 10, 4)]
#[case::phrase_five_clauses(true, 0, 10, 5)]
#[case::phrase_six_clauses(true, 0, 10, 6)]
#[case::phrase_eight_clauses(true, 0, 10, 8)]
#[case::phrase_sixteen_clauses(true, 0, 10, 16)]
fn test_bulk_and_matches_classic(
    #[case] phrase: bool,
    #[case] slop: u32,
    #[case] limit: usize,
    #[case] num_clauses: usize,
) {
    let num_docs = (BLOCK_SIZE * 8 + 37) as u32;
    let mut docs = DocSet::default();
    for doc_id in 0..num_docs {
        docs.append(u64::from(doc_id), 32 + doc_id % 57);
    }

    // Clauses with different densities; membership comes from a cheap
    // deterministic mix so docs scatter across blocks. Clause count picks
    // the dedicated (1-3) or generic (4+) merge kernel.
    let clause_docs = |modulus: u32, salt: u32| -> Vec<u32> {
        (0..num_docs)
            .filter(|doc| (doc.wrapping_mul(2654435761).wrapping_add(salt)) % modulus < 2)
            .collect()
    };
    let clauses = [
        clause_docs(3, 7),
        clause_docs(4, 13),
        clause_docs(5, 29),
        clause_docs(3, 41),
        clause_docs(3, 7),
        clause_docs(4, 13),
    ]
    .into_iter()
    .cycle()
    .take(num_clauses)
    .collect::<Vec<_>>();

    let build_postings = || {
        clauses
            .iter()
            .enumerate()
            .map(|(term_pos, doc_ids)| {
                let list = if phrase {
                    // Roughly half of each clause's docs put the token at
                    // position base+term_pos (forming the phrase); the
                    // rest scatter so the position check has misses.
                    let positions = doc_ids
                        .iter()
                        .map(|&doc| {
                            if doc % 2 == 0 {
                                vec![5 + term_pos as u32, 40 + (doc % 3)]
                            } else {
                                vec![20 + (term_pos as u32) * 4]
                            }
                        })
                        .collect::<Vec<_>>();
                    generate_posting_list_with_positions(doc_ids.clone(), positions, 8.0, true)
                } else {
                    generate_posting_list(doc_ids.clone(), 8.0, None, true)
                };
                PostingIterator::with_query_weight(
                    format!("t{term_pos}"),
                    term_pos as u32,
                    term_pos as u32,
                    1.0 + term_pos as f32 * 0.5,
                    list,
                    docs.len(),
                )
            })
            .collect::<Vec<_>>()
    };

    let mut params = FtsSearchParams::default().with_limit(Some(limit));
    if phrase {
        params.phrase_slop = Some(slop);
    }

    let normalize = |result: Vec<DocCandidate<u64>>| {
        let mut rows = result
            .into_iter()
            .map(|candidate| {
                (
                    candidate.posting_doc_id,
                    candidate.doc_length,
                    candidate.freqs,
                    candidate.document,
                )
            })
            .collect::<Vec<_>>();
        rows.sort_unstable();
        rows
    };

    let run = |mode| {
        let shared_floor = Arc::new(AtomicU32::new(0.0_f32.to_bits()));
        let mut wand = Wand::new(
            Operator::And,
            build_postings().into_iter(),
            &docs,
            UnitScorer,
        )
        .with_bulk_and_mode(mode)
        .with_shared_threshold(shared_floor.clone());
        let rows = normalize(wand.search(&params, &NoOpMetricsCollector).unwrap());
        let used_bulk = wand.bulk_and_searches > 0;
        ((rows, shared_floor.load(Ordering::Relaxed)), used_bulk)
    };

    let (bulk, bulk_used) = run(BulkAndMode::On);
    let (classic, classic_used) = run(BulkAndMode::Off);
    let (auto, auto_used) = run(BulkAndMode::Auto);
    assert!(bulk_used, "on should use bulk conjunction search");
    assert!(!classic_used, "off should use classic conjunction search");
    assert_eq!(auto_used, matches!(num_clauses, 2 | 3));
    assert!(!bulk.0.is_empty(), "test corpus should produce matches");
    assert_eq!(bulk, classic);
    assert_eq!(auto, classic);
}

#[rstest]
#[case::bulk_two(BulkAndMode::On, 2)]
#[case::bulk_four(BulkAndMode::On, 4)]
#[case::bulk_eight(BulkAndMode::On, 8)]
#[case::bulk_sixteen(BulkAndMode::On, 16)]
#[case::classic_four(BulkAndMode::Off, 4)]
#[case::classic_five(BulkAndMode::Off, 5)]
fn underfilled_disjoint_and_decodes_no_frequencies_or_bounds(
    #[case] mode: BulkAndMode,
    #[case] num_clauses: usize,
) {
    let num_docs = (BLOCK_SIZE * 3) as u32;
    let mut docs = DocSet::default();
    for doc_id in 0..num_docs {
        docs.append(u64::from(doc_id), 1);
    }
    let even = (0..num_docs).filter(|doc| doc % 2 == 0).collect::<Vec<_>>();
    let odd = (0..num_docs).filter(|doc| doc % 2 == 1).collect::<Vec<_>>();
    let all = (0..num_docs).collect::<Vec<_>>();
    let postings = (0..num_clauses)
        .map(|term| {
            let doc_ids = match term {
                0 => even.clone(),
                1 => odd.clone(),
                _ => all.clone(),
            };
            let len = doc_ids.len();
            PostingIterator::with_query_weight(
                format!("t{term}"),
                term as u32,
                term as u32,
                1.0,
                generate_impact_posting_list_with_freqs(doc_ids, vec![1; len], vec![1; len]),
                docs.len(),
            )
        })
        .collect::<Vec<_>>();
    let scored = Arc::new(AtomicUsize::new(0));
    let mut wand = Wand::new(
        Operator::And,
        postings.into_iter(),
        &docs,
        CountingScorer {
            scored: scored.clone(),
        },
    )
    .with_bulk_and_mode(mode);

    let hits = wand
        .search(
            &FtsSearchParams::default().with_limit(Some(10)),
            &NoOpMetricsCollector,
        )
        .unwrap();

    assert!(hits.is_empty());
    assert_eq!(scored.load(Ordering::Relaxed), 0);
    assert_eq!(
        wand.lead
            .iter()
            .map(|posting| posting.frequency_blocks_decoded())
            .sum::<usize>(),
        0
    );
    assert_eq!(
        wand.lead
            .iter()
            .map(|posting| posting.impact_bound_computations())
            .sum::<usize>(),
        0
    );
}

#[rstest]
#[case::legacy_full(crate::scalar::inverted::LEGACY_BLOCK_SIZE, 0)]
#[case::legacy_tail(crate::scalar::inverted::LEGACY_BLOCK_SIZE, 7)]
#[case::modern_full(MAX_POSTING_BLOCK_SIZE, 0)]
#[case::modern_tail(MAX_POSTING_BLOCK_SIZE, 7)]
fn underfilled_sparse_and_decodes_only_matching_frequency_blocks(
    #[values(BulkAndMode::On, BulkAndMode::Off)] mode: BulkAndMode,
    #[case] block_size: usize,
    #[case] tail_len: usize,
) {
    let posting_len = block_size * 2 + tail_len;
    let first = (0..posting_len)
        .map(|index| (index * 2) as u32)
        .collect::<Vec<_>>();
    let mut second = (0..posting_len)
        .map(|index| (index * 2 + 1) as u32)
        .collect::<Vec<_>>();
    let last_doc = *first.last().unwrap();
    *second.last_mut().unwrap() = last_doc;
    let num_docs = posting_len * 2;
    let mut docs = DocSet::default();
    for doc_id in 0..num_docs {
        docs.append(doc_id as u64, 1);
    }
    let postings = [first, second]
        .into_iter()
        .enumerate()
        .map(|(term, doc_ids)| {
            let len = doc_ids.len();
            PostingIterator::with_query_weight(
                format!("t{term}"),
                term as u32,
                term as u32,
                1.0,
                generate_impact_posting_list_with_freqs_and_block_size(
                    doc_ids,
                    vec![1; len],
                    vec![1; len],
                    block_size,
                ),
                docs.len(),
            )
        })
        .collect::<Vec<_>>();
    let scored = Arc::new(AtomicUsize::new(0));
    let mut wand = Wand::new(
        Operator::And,
        postings.into_iter(),
        &docs,
        CountingScorer {
            scored: scored.clone(),
        },
    )
    .with_bulk_and_mode(mode);

    let hits = wand
        .search(
            &FtsSearchParams::default().with_limit(Some(10)),
            &NoOpMetricsCollector,
        )
        .unwrap();

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].document, u64::from(last_doc));
    assert_eq!(scored.load(Ordering::Relaxed), 2);
    assert_eq!(
        wand.lead
            .iter()
            .map(|posting| posting.frequency_blocks_decoded())
            .sum::<usize>(),
        2,
        "only the two blocks containing the surviving match need frequencies"
    );
    assert_eq!(
        wand.lead
            .iter()
            .map(|posting| posting.impact_bound_computations())
            .sum::<usize>(),
        0
    );
}

/// A sparse lead over dense followers must drive the window traversal:
/// after an empty window the next one starts at the lead's next document,
/// so the number of sliced windows tracks the lead's documents rather than
/// the followers' block count. Results stay identical to the classic loop.
#[rstest]
fn bulk_and_windows_jump_to_next_lead_document(
    #[values(2, 4)] num_clauses: usize,
    #[values(0.0, 0.05)] initial_floor: f32,
) {
    let num_docs = MAX_POSTING_BLOCK_SIZE * 64;
    let lead_stride = 2_000;
    let docs = CostOnlyDocuments {
        total_docs: num_docs,
        visible_cost_upper_bound: num_docs,
    };
    let run = |mode| {
        let postings = (0..num_clauses)
            .map(|term| {
                let doc_ids = if term == 0 {
                    (0..num_docs as u32)
                        .step_by(lead_stride)
                        .collect::<Vec<_>>()
                } else {
                    (0..num_docs as u32)
                        .filter(|doc| doc % 3 != term as u32)
                        .collect()
                };
                let len = doc_ids.len();
                PostingIterator::with_query_weight(
                    format!("t{term}"),
                    term as u32,
                    term as u32,
                    1.0,
                    generate_impact_posting_list_with_freqs_and_block_size(
                        doc_ids,
                        (0..len).map(|index| 1 + (index % 5) as u32).collect(),
                        vec![1; len],
                        MAX_POSTING_BLOCK_SIZE,
                    ),
                    docs.len(),
                )
            })
            .collect::<Vec<_>>();
        let shared_floor = Arc::new(AtomicU32::new(initial_floor.to_bits()));
        let mut wand = Wand::new(Operator::And, postings.into_iter(), &docs, UnitScorer)
            .with_bulk_and_mode(mode)
            .with_shared_threshold(shared_floor.clone());
        let mut rows = wand
            .search(
                &FtsSearchParams::default().with_limit(Some(10)),
                &NoOpMetricsCollector,
            )
            .unwrap()
            .into_iter()
            .map(|hit| (hit.document, hit.doc_length, hit.freqs))
            .collect::<Vec<_>>();
        rows.sort_unstable();
        (
            rows,
            shared_floor.load(Ordering::Relaxed),
            wand.and_window_stats.windows_sliced,
        )
    };
    let (classic, classic_floor, _) = run(BulkAndMode::Off);
    let (bulk, bulk_floor, windows) = run(BulkAndMode::On);
    assert_eq!(bulk, classic);
    assert_eq!(bulk_floor, classic_floor);
    let lead_docs = num_docs / lead_stride + 1;
    let follower_blocks = (num_docs * 2 / 3) / MAX_POSTING_BLOCK_SIZE;
    assert!(
        windows <= 2 * lead_docs + 2,
        "sliced {windows} windows for {lead_docs} lead docs; followers have {follower_blocks} blocks each"
    );
}

#[rstest]
#[case::dense("dense")]
#[case::sparse("sparse")]
#[case::shifted("shifted")]
#[case::underfilled_five("five")]
#[case::underfilled_zero("zero")]
#[case::max_doc_id("max")]
fn pairwise_and_matches_classic(
    #[case] shape: &str,
    #[values(4, 5, 8, 16)] num_clauses: usize,
    #[values(0.0, 0.1)] initial_floor: f32,
) {
    let posting_len = MAX_POSTING_BLOCK_SIZE * 3 + 7;
    let docs = CostOnlyDocuments {
        total_docs: posting_len * 2 + num_clauses,
        visible_cost_upper_bound: posting_len * 2 + num_clauses,
    };
    let run = |mode| {
        let postings = (0..num_clauses)
            .map(|term| {
                let doc_ids = (0..posting_len)
                    .map(|index| match shape {
                        "dense" => index as u32,
                        "shifted" => (index + term) as u32,
                        "sparse" => (index * 3) as u32,
                        "max" => u32::MAX - ((posting_len - index - 1) * 1_000_003) as u32,
                        "zero" | "five" => {
                            let is_last_clause = term + 1 == num_clauses;
                            let is_match = shape == "five" && index >= posting_len - 5;
                            (index * 2 + usize::from(is_last_clause && !is_match)) as u32
                        }
                        _ => unreachable!(),
                    })
                    .collect::<Vec<_>>();
                let frequencies = (0..posting_len)
                    .map(|index| 1 + ((index + term) % 7) as u32)
                    .collect::<Vec<_>>();
                PostingIterator::with_query_weight(
                    format!("t{term}"),
                    term as u32,
                    term as u32,
                    [0.002_972_301_6, 0.001_982_450_7, 0.001_882_293][term % 3],
                    generate_impact_posting_list_with_freqs_and_block_size(
                        doc_ids,
                        frequencies,
                        vec![1; posting_len],
                        MAX_POSTING_BLOCK_SIZE,
                    ),
                    docs.len(),
                )
            })
            .collect::<Vec<_>>();
        let shared_floor = Arc::new(AtomicU32::new(initial_floor.to_bits()));
        let mut wand = Wand::new(Operator::And, postings.into_iter(), &docs, UnitScorer)
            .with_bulk_and_mode(mode)
            .with_shared_threshold(shared_floor.clone());
        let hits = wand
            .search(
                &FtsSearchParams::default().with_limit(Some(10)),
                &NoOpMetricsCollector,
            )
            .unwrap();
        let mut rows = hits
            .into_iter()
            .map(|hit| (hit.document, hit.doc_length, hit.freqs))
            .collect::<Vec<_>>();
        rows.sort_unstable();
        if mode == BulkAndMode::On {
            assert_eq!(
                wand.and_window_stats.pairwise_intersections > 0,
                initial_floor == 0.0
            );
            if initial_floor == 0.0 && matches!(shape, "five" | "zero") {
                assert_eq!(rows.len(), if shape == "five" { 5 } else { 0 });
                assert_eq!(
                    wand.lead
                        .iter()
                        .map(|posting| posting.frequency_blocks_decoded())
                        .sum::<usize>(),
                    if shape == "five" { num_clauses } else { 0 },
                    "only final matching blocks need frequency decoding"
                );
                assert_eq!(
                    wand.lead
                        .iter()
                        .map(|posting| posting.impact_bound_computations())
                        .sum::<usize>(),
                    0
                );
            }
        }
        (rows, shared_floor.load(Ordering::Relaxed))
    };
    let classic = run(BulkAndMode::Off);
    assert_eq!(run(BulkAndMode::On), classic);
    assert_eq!(run(BulkAndMode::Auto), classic);
}

#[rstest]
fn pairwise_and_preserves_phrase_and_visibility(
    #[values(4, 8, 16)] num_clauses: u32,
    #[values(None, Some(0), Some(3))] phrase_slop: Option<u32>,
    #[values(false, true)] is_filtered: bool,
) {
    let num_docs = MAX_POSTING_BLOCK_SIZE * 2 + 7;
    let mut docs = DocSet::default();
    for doc in 0..num_docs {
        docs.append(doc as u64, 1);
    }
    let mask = if is_filtered {
        RowAddrMask::from_block([0, (num_docs - 1) as u64].into_iter().collect())
    } else {
        RowAddrMask::all_rows()
    };
    let documents = LegacyWandDocuments::new(&docs, &mask);
    let run = |mode| {
        let postings = (0..num_clauses)
            .map(|term| {
                let mut list = generate_impact_posting_list_with_freqs_and_block_size(
                    (0..num_docs as u32).collect(),
                    vec![1; num_docs],
                    vec![1; num_docs],
                    MAX_POSTING_BLOCK_SIZE,
                );
                let mut bytes = Vec::new();
                let mut offsets = Vec::new();
                for start in (0..num_docs).step_by(MAX_POSTING_BLOCK_SIZE) {
                    let end = (start + MAX_POSTING_BLOCK_SIZE).min(num_docs);
                    let positions = (start..end)
                        .map(|doc| {
                            if doc >= num_docs - 5 {
                                5 + term
                            } else {
                                5 + term * 8
                            }
                        })
                        .collect::<Vec<_>>();
                    offsets.push(bytes.len() as u32);
                    encode_position_stream_block_into(
                        &positions,
                        &vec![1; end - start],
                        PositionStreamCodec::VarintDocDelta,
                        &mut bytes,
                    )
                    .unwrap();
                }
                if let PostingList::Compressed(list) = &mut list {
                    list.positions = Some(CompressedPositionStorage::SharedStream(
                        SharedPositionStream::new(
                            PositionStreamCodec::VarintDocDelta,
                            offsets,
                            bytes.into(),
                        ),
                    ));
                }
                PostingIterator::with_query_weight(
                    format!("t{term}"),
                    term,
                    term,
                    1.0,
                    list,
                    docs.len(),
                )
            })
            .collect::<Vec<_>>();
        let mut wand = Wand::new(Operator::And, postings.into_iter(), &documents, UnitScorer)
            .with_bulk_and_mode(mode);
        let mut params = FtsSearchParams::default().with_limit(Some(10));
        params.phrase_slop = phrase_slop;
        let mut hits = wand
            .search(&params, &NoOpMetricsCollector)
            .unwrap()
            .into_iter()
            .map(|hit| (hit.document, hit.freqs))
            .collect::<Vec<_>>();
        hits.sort_unstable();
        if mode == BulkAndMode::On {
            assert!(wand.and_window_stats.pairwise_intersections > 0);
        }
        assert_eq!(
            hits.len(),
            if phrase_slop.is_some() {
                5 - usize::from(is_filtered)
            } else {
                10
            }
        );
        (hits, wand.threshold.to_bits())
    };
    let classic = run(BulkAndMode::Off);
    assert_eq!(run(BulkAndMode::On), classic);
    assert_eq!(run(BulkAndMode::Auto), classic);
}

#[rstest]
fn pairwise_and_requires_current_posting_blocks(
    #[values(BulkAndMode::On, BulkAndMode::Auto)] mode: BulkAndMode,
    #[values(2, 3, 4, 16)] num_clauses: usize,
    #[values(crate::scalar::inverted::LEGACY_BLOCK_SIZE, MAX_POSTING_BLOCK_SIZE)] block_size: usize,
    #[values(false, true)] has_impacts: bool,
    #[values(false, true)] has_grouped_terms: bool,
) {
    let docs = CostOnlyDocuments {
        total_docs: 5,
        visible_cost_upper_bound: 5,
    };
    let postings = (0..num_clauses)
        .map(|term| {
            let mut list = generate_impact_posting_list_with_freqs_and_block_size(
                vec![0, 2, 4, 6, 8],
                vec![1; 5],
                vec![1; 5],
                block_size,
            );
            if !has_impacts && let PostingList::Compressed(list) = &mut list {
                list.impacts = None;
            }
            let grouped_terms = has_grouped_terms
                .then(|| Arc::<[GroupedTermScorer]>::from([GroupedTermScorer::new(1.0, &list)]));
            let posting = PostingIterator::with_query_weight(
                format!("t{term}"),
                term as u32,
                term as u32,
                1.0,
                list,
                docs.len(),
            );
            if let Some(grouped_terms) = grouped_terms {
                posting.with_grouped_terms(grouped_terms)
            } else {
                posting
            }
        })
        .collect::<Vec<_>>();
    let mut wand =
        Wand::new(Operator::And, postings.into_iter(), &docs, UnitScorer).with_bulk_and_mode(mode);
    let hits = wand
        .search(
            &FtsSearchParams::default().with_limit(Some(10)),
            &NoOpMetricsCollector,
        )
        .unwrap();
    assert_eq!(hits.len(), 5);
    assert_eq!(
        wand.and_window_stats.pairwise_intersections > 0,
        num_clauses >= 4
            && block_size == MAX_POSTING_BLOCK_SIZE
            && has_impacts
            && !has_grouped_terms
    );
    if has_grouped_terms {
        assert_eq!(wand.bulk_and_searches, 0);
    }
}

#[rstest]
fn bulk_and_activates_frequencies_and_bounds_after_heap_fills(
    #[values(2, 4, 8, 16)] num_clauses: u32,
) {
    let num_docs = (BLOCK_SIZE * 2) as u32;
    let mut docs = DocSet::default();
    for doc_id in 0..num_docs {
        docs.append(u64::from(doc_id), 1);
    }
    let frequencies = (0..num_docs)
        .map(|doc| if doc < BLOCK_SIZE as u32 { 1 } else { 10 })
        .collect::<Vec<_>>();
    let postings = (0..num_clauses)
        .map(|term| {
            PostingIterator::with_query_weight(
                format!("t{term}"),
                term,
                term,
                1.0,
                generate_impact_posting_list_with_freqs(
                    (0..num_docs).collect(),
                    frequencies.clone(),
                    vec![1; num_docs as usize],
                ),
                docs.len(),
            )
        })
        .collect::<Vec<_>>();
    let scored = Arc::new(AtomicUsize::new(0));
    let mut wand = Wand::new(
        Operator::And,
        postings.into_iter(),
        &docs,
        CountingScorer {
            scored: scored.clone(),
        },
    )
    .with_bulk_and_mode(BulkAndMode::On);

    let hits = wand
        .search(
            &FtsSearchParams::default().with_limit(Some(1)),
            &NoOpMetricsCollector,
        )
        .unwrap();

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].document, BLOCK_SIZE as u64);
    assert!(wand.threshold > 0.0);
    assert!(
        wand.lead
            .iter()
            .map(|posting| posting.frequency_blocks_decoded())
            .sum::<usize>()
            >= 4,
        "the competitive second window must decode its frequency blocks"
    );
    assert!(
        wand.lead
            .iter()
            .map(|posting| posting.impact_bound_computations())
            .sum::<usize>()
            > 0,
        "impact bounds must activate on the first window after the heap fills"
    );
    assert!(scored.load(Ordering::Relaxed) > 0);
}

/// The floor turns positive in the middle of the first window, where only
/// the per-candidate prune refreshes the score limit. The second window
/// has the same floor and the same follower block max, so the frequency
/// prune table must still be built for it instead of staying disabled.
#[test]
fn bulk_and_builds_frequency_prune_table_after_mid_window_floor() {
    let num_docs = (MAX_POSTING_BLOCK_SIZE * 2) as u32;
    let mut docs = DocSet::default();
    for doc_id in 0..num_docs {
        docs.append(u64::from(doc_id), 1);
    }
    let run = |mode| {
        // One block spanning both windows; only document 0 can win.
        let sparse_docs = (0..num_docs).step_by(2).collect::<Vec<_>>();
        let sparse_freqs = sparse_docs
            .iter()
            .map(|&doc| if doc == 0 { 10 } else { 1 })
            .collect::<Vec<_>>();
        // Two blocks with the same block max, so the follower bound is
        // identical in both windows. Document 257 keeps the second window
        // competitive by block max without matching the sparse clause.
        let dense_docs = (0..num_docs).collect::<Vec<_>>();
        let dense_freqs = dense_docs
            .iter()
            .map(|&doc| if doc == 0 || doc == 257 { 10 } else { 1 })
            .collect::<Vec<_>>();
        let postings = [(sparse_docs, sparse_freqs), (dense_docs, dense_freqs)]
            .into_iter()
            .enumerate()
            .map(|(term, (doc_ids, freqs))| {
                let len = doc_ids.len();
                PostingIterator::with_query_weight(
                    format!("t{term}"),
                    term as u32,
                    term as u32,
                    1.0,
                    generate_impact_posting_list_with_freqs_and_block_size(
                        doc_ids,
                        freqs,
                        vec![1; len],
                        MAX_POSTING_BLOCK_SIZE,
                    ),
                    docs.len(),
                )
            })
            .collect::<Vec<_>>();
        let mut wand = Wand::new(Operator::And, postings.into_iter(), &docs, UnitScorer)
            .with_bulk_and_mode(mode);
        let hits = wand
            .search(
                &FtsSearchParams::default().with_limit(Some(1)),
                &NoOpMetricsCollector,
            )
            .unwrap()
            .into_iter()
            .map(|hit| (hit.document, hit.doc_length, hit.freqs))
            .collect::<Vec<_>>();
        (
            hits,
            wand.threshold,
            wand.and_window_stats.windows_sliced,
            wand.and_window_stats.windows_frequency_pruned,
        )
    };
    let (classic, classic_floor, _, _) = run(BulkAndMode::Off);
    let (bulk, bulk_floor, windows_sliced, windows_frequency_pruned) = run(BulkAndMode::On);
    assert_eq!(bulk, classic);
    assert_eq!(bulk.len(), 1);
    assert_eq!(bulk[0].0, 0);
    assert_eq!(bulk_floor, 20.0);
    assert_eq!(bulk_floor, classic_floor);
    assert_eq!(
        windows_sliced, 2,
        "the dense clause's block boundary splits two windows"
    );
    assert_eq!(
        windows_frequency_pruned, 1,
        "the second window must apply the frequency prune table"
    );
}
