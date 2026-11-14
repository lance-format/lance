// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

#[test]
fn wand_cursor_cost_uses_materialized_visibility_upper_bound() {
    let total_docs = 10;
    let selected = DocVisibility::Selected(roaring::RoaringBitmap::from_iter([1, 7]));
    let selected_bound = <&DocVisibility as ModernVisibility>::len(&&selected, total_docs);

    let filtered = DocVisibility::Filtered {
        projection: resident_row_address_projection_for_test((0..total_docs as u64).collect()),
        mask: Arc::new(RowAddrMask::all_rows()),
    };
    let filtered_bound = <&DocVisibility as ModernVisibility>::len(&&filtered, total_docs);
    let all_bound = AllModernDocuments.len(total_docs);

    assert_eq!(selected_bound, 2);
    assert_eq!(filtered_bound, total_docs);
    assert_eq!(all_bound, total_docs);

    let scorer = Arc::new(MemBM25Scorer::new(
        total_docs as u64,
        total_docs,
        std::collections::HashMap::from([("term".to_owned(), 8)]),
    ));
    let posting = || {
        PostingIterator::new(
            "term".to_owned(),
            0,
            0,
            generate_posting_list((0..8).collect(), 1.0, None, true),
            total_docs,
        )
    };
    let params = FtsSearchParams::default();
    let metrics = NoOpMetricsCollector;

    let selected_documents = CostOnlyDocuments {
        total_docs,
        visible_cost_upper_bound: selected_bound,
    };
    let selected_cursor = WandCursor::new(
        Operator::Or,
        vec![posting()],
        &selected_documents,
        scorer.clone(),
        &params,
        &metrics,
    );
    assert_eq!(selected_cursor.cost(), 2);

    for visible_cost_upper_bound in [filtered_bound, all_bound] {
        let documents = CostOnlyDocuments {
            total_docs,
            visible_cost_upper_bound,
        };
        let cursor = WandCursor::new(
            Operator::Or,
            vec![posting()],
            &documents,
            scorer.clone(),
            &params,
            &metrics,
        );
        assert_eq!(cursor.cost(), 8);
    }

    let mut complete_docs = DocSet::default();
    for doc_id in 0..total_docs {
        complete_docs.append(doc_id as u64, 1);
    }
    assert_eq!(
        WandDocuments::visible_cost_upper_bound(&complete_docs),
        total_docs
    );
}

#[test]
fn compound_global_bound_rejects_grouped_term_scoring() {
    let mut docs = DocSet::default();
    docs.append(0, 1);
    let list = generate_posting_list(vec![0], 1.0, None, false);
    let grouped_terms = Arc::<[GroupedTermScorer]>::from([GroupedTermScorer::new(1.0, &list)]);
    let posting =
        PostingIterator::with_query_weight(String::from("term"), 0, 0, 1.0, list, docs.len())
            .with_grouped_terms(grouped_terms);
    let wand = Wand::new(Operator::Or, std::iter::once(posting), &docs, UnitScorer);

    assert_eq!(wand.compound_global_score_upper_bound(), None);
}

#[test]
fn compound_global_bound_rejects_late_partial_posting_state() {
    let mut docs = DocSet::default();
    docs.append(0, 1);
    let posting = PostingIterator::with_query_weight(
        String::from("term"),
        0,
        0,
        1.0,
        generate_posting_list(vec![0], 1.0, None, false),
        docs.len(),
    );
    let mut wand = Wand::new(Operator::Or, std::iter::once(posting), &docs, UnitScorer);

    assert!(wand.next().unwrap().is_some());
    wand.push_back_leads(1);
    assert_eq!(wand.compound_global_score_upper_bound(), None);
}

#[test]
fn final_score_uses_query_order_and_keeps_floor_ties() {
    let contributions = [0.002_972_301_6_f32, 0.001_982_450_7_f32, 0.001_882_293_f32];
    // Bounds deliberately make the WAND heap visit positions 2, 0, 1.
    let bounds = [2.0_f32, 1.0, 3.0];
    let postings = || {
        contributions
            .into_iter()
            .zip(bounds)
            .enumerate()
            .map(|(position, (query_weight, max_score))| {
                PostingIterator::with_query_weight(
                    format!("t{position}"),
                    position as u32,
                    position as u32,
                    query_weight,
                    generate_posting_list(vec![0], max_score, None, false),
                    1,
                )
            })
            .collect::<Vec<_>>()
    };
    let mut docs = DocSet::default();
    docs.append(0, 1);
    let mut wand = Wand::new(Operator::Or, postings().into_iter(), &docs, UnitScorer);

    let (_, heap_order_score) = wand.next().unwrap().unwrap();
    let query_order_score = contributions
        .into_iter()
        .fold(0.0_f32, |score, contribution| score + contribution);
    let wide_score = contributions
        .into_iter()
        .fold(0.0_f64, |score, contribution| {
            score + f64::from(contribution)
        }) as f32;

    assert_eq!(query_order_score.to_bits(), 0x3be0_094c);
    assert_eq!(wide_score.to_bits(), 0x3be0_094b);
    assert_eq!(heap_order_score.to_bits(), 0x3be0_094a);
    assert_eq!(wand.score_in_query_order(1), query_order_score);

    let params = FtsSearchParams::default();
    let metrics = NoOpMetricsCollector;
    let mut cursor = WandCursor::new(
        Operator::Or,
        postings(),
        &docs,
        Arc::new(MemBM25Scorer::new(1, 1, std::collections::HashMap::new())),
        &params,
        &metrics,
    );
    // Equal-score candidates remain competitive because row id is the
    // final tie-breaker. The internal heap-order sum is slightly lower.
    cursor.set_min_competitive_score(query_order_score).unwrap();
    assert_eq!(cursor.next().unwrap(), Some(0));
    assert_eq!(cursor.current_score().unwrap(), query_order_score);
}

#[rstest]
#[case::initial_floor(false)]
#[case::explicit_zero_floor(true)]
fn wand_cursor_preserves_zero_score_membership(
    #[case] set_zero_floor: bool,
    #[values(Operator::Or, Operator::And)] operator: Operator,
    #[values(1, 4, 5)] num_terms: u32,
) {
    let postings = (0..num_terms)
        .map(|term| {
            let list = if operator == Operator::And {
                generate_impact_posting_list_with_freqs_and_block_size(
                    vec![0],
                    vec![1],
                    vec![1],
                    MAX_POSTING_BLOCK_SIZE,
                )
            } else {
                generate_posting_list(vec![0], 0.0, None, false)
            };
            PostingIterator::with_query_weight(format!("common{term}"), term, term, 0.0, list, 1)
        })
        .collect();
    let mut docs = DocSet::default();
    docs.append(0, 1);
    let params = FtsSearchParams::default();
    let metrics = NoOpMetricsCollector;
    let mut cursor = WandCursor::new(
        operator,
        postings,
        &docs,
        Arc::new(MemBM25Scorer::new(1, 1, std::collections::HashMap::new())),
        &params,
        &metrics,
    );
    if set_zero_floor {
        cursor.set_min_competitive_score(0.0).unwrap();
    }

    assert_eq!(cursor.next().unwrap(), Some(0));
    assert_eq!(cursor.current_score().unwrap(), 0.0);
}

/// The compound-phrase request from review: drive the exact-phrase check the
/// way `WandCursor::matches()` does - a phrase leaf inside a compound query -
/// instead of calling the entries directly.
#[rstest]
fn wand_cursor_confirmation_rejects_disjoint_phrase_positions(
    #[values(false, true)] is_compressed: bool,
) {
    let n = 4096u32;
    let mut docs = DocSet::default();
    docs.append(0, 1_000_001);
    let make_postings = || {
        vec![
            PostingIterator::new(
                String::from("a"),
                0,
                0,
                generate_posting_list_with_positions(
                    vec![0],
                    vec![(0..n).collect()],
                    1.0,
                    is_compressed,
                ),
                docs.len(),
            ),
            PostingIterator::new(
                String::from("b"),
                1,
                1,
                generate_posting_list_with_positions(
                    vec![0],
                    vec![(2 * n..3 * n).collect()],
                    1.0,
                    is_compressed,
                ),
                docs.len(),
            ),
        ]
    };
    let params = FtsSearchParams {
        phrase_slop: Some(0),
        ..Default::default()
    };
    let metrics = NoOpMetricsCollector;
    let mut cursor = WandCursor::new(
        Operator::And,
        make_postings(),
        &docs,
        Arc::new(MemBM25Scorer::new(1, 1, std::collections::HashMap::new())),
        &params,
        &metrics,
    );
    assert_eq!(cursor.next().unwrap(), Some(0));

    let (matched, steps) = counted_anchor_steps(|| cursor.matches().unwrap());
    assert!(!matched, "disjoint positions cannot form a phrase");
    assert!(
        steps <= 2,
        "the compound path must skip too, took {steps} steps"
    );
}

/// Same path, but the phrase really is present: the confirmation has to
/// survive the jump instead of over-shooting the match.
#[rstest]
fn wand_cursor_confirmation_accepts_aligned_phrase_positions(
    #[values(false, true)] is_compressed: bool,
) {
    let n = 4096u32;
    let mut docs = DocSet::default();
    docs.append(0, 1_000_001);
    let make_postings = || {
        vec![
            PostingIterator::new(
                String::from("a"),
                0,
                0,
                generate_posting_list_with_positions(
                    vec![0],
                    vec![(0..n).collect()],
                    1.0,
                    is_compressed,
                ),
                docs.len(),
            ),
            PostingIterator::new(
                String::from("b"),
                1,
                1,
                generate_posting_list_with_positions(
                    vec![0],
                    vec![(1..n + 1).collect()],
                    1.0,
                    is_compressed,
                ),
                docs.len(),
            ),
        ]
    };
    let params = FtsSearchParams {
        phrase_slop: Some(0),
        ..Default::default()
    };
    let metrics = NoOpMetricsCollector;
    let mut cursor = WandCursor::new(
        Operator::And,
        make_postings(),
        &docs,
        Arc::new(MemBM25Scorer::new(1, 1, std::collections::HashMap::new())),
        &params,
        &metrics,
    );
    assert_eq!(cursor.next().unwrap(), Some(0));

    let (matched, steps) = counted_anchor_steps(|| cursor.matches().unwrap());
    assert!(matched, "the phrase is present at base 0");
    assert!(
        steps <= 2,
        "took {steps} steps to confirm a first-position match"
    );
}
