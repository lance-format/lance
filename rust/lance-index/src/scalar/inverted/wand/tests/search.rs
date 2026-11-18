// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

struct PartialNormScorer;

impl Scorer for PartialNormScorer {
    fn query_weight(&self, _token: &str) -> f32 {
        1.0
    }

    fn doc_weight(&self, freq: u32, _doc_tokens: u32) -> f32 {
        freq as f32
    }

    fn doc_norm(&self, doc_tokens: u32) -> Option<f32> {
        (doc_tokens == 0).then_some(0.0)
    }
}

#[test]
fn test_norm_cache_requires_all_norm_codes() {
    let mut docs = DocSet::default();
    docs.append(0, 1);
    docs.set_quantized_scoring(true);
    let wand = Wand::new(Operator::Or, std::iter::empty(), &docs, PartialNormScorer);

    assert!(wand.norm_k_cache().is_none());

    let shared = SharedNormAddends::default();
    for _ in 0..2 {
        let wand = Wand::new(Operator::Or, std::iter::empty(), &docs, PartialNormScorer)
            .with_shared_norm_addends(shared.clone());
        assert!(wand.norm_k_cache().is_none());
    }
}

#[test]
fn test_shared_norm_addends_are_built_once_and_match_local_table() {
    let mut docs = DocSet::default();
    docs.append(0, 7);
    docs.set_quantized_scoring(true);
    let scorer = Arc::new(MemBM25Scorer::new(70, 10, std::collections::HashMap::new()));
    let shared = SharedNormAddends::default();
    let norm_addends = |shared: Option<SharedNormAddends>| {
        let mut wand = Wand::new(Operator::Or, std::iter::empty(), &docs, scorer.clone());
        if let Some(shared) = shared {
            wand = wand.with_shared_norm_addends(shared);
        }
        wand.norm_k_cache().unwrap().1
    };

    let first = norm_addends(Some(shared.clone()));
    let second = norm_addends(Some(shared));
    let local = norm_addends(None);
    assert!(
        Arc::ptr_eq(&first, &second),
        "partitions of one query must share one table"
    );
    assert_eq!((*first).map(f32::to_bits), (*local).map(f32::to_bits));
}

#[rstest]
#[case::auto("auto", Some(BulkAndMode::Auto))]
#[case::auto_case_and_whitespace(" AUTO ", Some(BulkAndMode::Auto))]
#[case::on("on", Some(BulkAndMode::On))]
#[case::on_legacy("1", Some(BulkAndMode::On))]
#[case::off("off", Some(BulkAndMode::Off))]
#[case::off_legacy("0", Some(BulkAndMode::Off))]
#[case::invalid("true", None)]
#[case::empty("", None)]
fn test_bulk_and_mode_parse(#[case] value: &str, #[case] expected: Option<BulkAndMode>) {
    assert_eq!(BulkAndMode::parse(value), expected);
}

#[rstest]
#[case::auto_one(BulkAndMode::Auto, 1, false)]
#[case::auto_two(BulkAndMode::Auto, 2, true)]
#[case::auto_three(BulkAndMode::Auto, 3, true)]
#[case::auto_four(BulkAndMode::Auto, 4, false)]
#[case::on_one(BulkAndMode::On, 1, true)]
#[case::on_five(BulkAndMode::On, 5, true)]
#[case::off_two(BulkAndMode::Off, 2, false)]
#[case::off_five(BulkAndMode::Off, 5, false)]
fn test_bulk_and_mode_enabled_for(
    #[case] mode: BulkAndMode,
    #[case] num_clauses: usize,
    #[case] expected: bool,
) {
    assert_eq!(mode.enabled_for(num_clauses), expected);
}

struct PanicQueryWeightScorer;

impl Scorer for PanicQueryWeightScorer {
    fn query_weight(&self, _token: &str) -> f32 {
        panic!("query_weight should be precomputed before WAND construction");
    }

    fn doc_weight(&self, freq: u32, _doc_tokens: u32) -> f32 {
        freq as f32
    }
}

#[rstest]
#[tokio::test]
async fn test_wand(#[values(false, true)] is_compressed: bool) {
    let mut docs = DocSet::default();
    for i in 0..2 * BLOCK_SIZE {
        docs.append(i as u64, 1);
    }

    // when the pivot is greater than 0, and the first posting list is exhausted after shallow_next
    let postings = vec![
        PostingIterator::new(
            String::from("test"),
            0,
            0,
            generate_posting_list(
                Vec::from_iter(0..=BLOCK_SIZE as u32 + 1),
                1.0,
                None,
                is_compressed,
            ),
            docs.len(),
        ),
        PostingIterator::new(
            String::from("full"),
            1,
            1,
            generate_posting_list(vec![BLOCK_SIZE as u32 + 2], 1.0, None, is_compressed),
            docs.len(),
        ),
    ];

    let bm25 = IndexBM25Scorer::new(std::iter::empty());
    let mut wand = Wand::new(Operator::And, postings.into_iter(), &docs, bm25);
    // This should trigger the bug when the second posting list becomes empty
    let result = wand
        .search(&FtsSearchParams::default(), &NoOpMetricsCollector)
        .unwrap();
    assert_eq!(result.len(), 0); // Should not panic
}

/// The shared floor prunes partitions that can't reach the global top-k: a
/// high-scoring partition sets the floor and the rest skip their blocks.
/// (Result correctness is covered by the FTS search tests, since sharing is
/// always on.)
#[test]
fn cross_partition_threshold_sharing_prunes() {
    use crate::metrics::MetricsCollector;
    use std::sync::atomic::AtomicUsize;

    #[derive(Default)]
    struct CountComparisons(AtomicUsize);
    impl MetricsCollector for CountComparisons {
        fn record_parts_loaded(&self, _: usize) {}
        fn record_index_loads(&self, _: usize) {}
        fn record_comparisons(&self, n: usize) {
            self.0.fetch_add(n, Ordering::Relaxed);
        }
    }

    let params = FtsSearchParams::default().with_limit(Some(10));
    let part_docs = 4 * BLOCK_SIZE as u32;
    // One high-scoring partition (weight 10) then 7 low-scoring ones.
    let parts: Vec<(f32, std::ops::Range<u32>)> = std::iter::once((10.0, 0..part_docs))
        .chain((1..8).map(|i| (1.0, i * part_docs..(i + 1) * part_docs)))
        .collect();

    let new_floor = || Arc::new(AtomicU32::new(f32::NEG_INFINITY.to_bits()));

    // Total comparisons across all partitions. `Some(floor)` makes every
    // partition share that one floor; `None` gives each its own.
    let total_comparisons = |shared_floor: Option<&Arc<AtomicU32>>| -> usize {
        let metrics = CountComparisons::default();
        for (qw, rows) in &parts {
            let mut docs = DocSet::default();
            for d in rows.clone() {
                docs.append(d as u64, 1);
            }
            let postings = vec![PostingIterator::with_query_weight(
                String::from("t"),
                0,
                0,
                *qw,
                generate_posting_list(rows.clone().collect(), *qw, None, false),
                docs.len(),
            )];
            let floor = shared_floor.cloned().unwrap_or_else(new_floor);
            Wand::new(Operator::Or, postings.into_iter(), &docs, UnitScorer)
                .with_shared_threshold(floor)
                .search(&params, &metrics)
                .unwrap();
        }
        metrics.0.load(Ordering::Relaxed)
    };

    let one_floor = new_floor();
    let with_shared_floor = total_comparisons(Some(&one_floor));
    let with_private_floors = total_comparisons(None);
    assert!(
        with_shared_floor < with_private_floors,
        "shared floor should prune comparisons: \
             shared={with_shared_floor} private={with_private_floors}"
    );
}

#[test]
fn test_wand_new_uses_precomputed_query_weight() {
    let mut docs = DocSet::default();
    docs.append(1, 1);

    let postings = vec![PostingIterator::with_query_weight(
        String::from("term"),
        0,
        0,
        2.0,
        generate_posting_list(vec![0], 1.0, None, false),
        docs.len(),
    )];

    let wand = Wand::new(
        Operator::Or,
        postings.into_iter(),
        &docs,
        PanicQueryWeightScorer,
    );
    assert_eq!(wand.head.len(), 1);
}

#[test]
fn test_flat_search_or_keeps_masked_docs_in_same_block() {
    let mut docs = DocSet::default();
    for i in 0..=(BLOCK_SIZE as u64 + 1) {
        let doc_tokens = if i == 1 { 100 } else { 1 };
        docs.append(i, doc_tokens);
    }

    let posting = PostingIterator::with_query_weight(
        String::from("term"),
        0,
        0,
        1.0,
        generate_posting_list(
            (1..=(BLOCK_SIZE as u32 + 1)).collect(),
            1.0,
            Some(vec![1.0, 1.0]),
            true,
        ),
        docs.len(),
    );

    let mut wand = Wand::new(
        Operator::Or,
        vec![posting].into_iter(),
        &docs,
        InverseDocLengthScorer,
    );
    wand.threshold = 0.5;

    let selected = vec![(1_u64, 1_u64), (2_u64, 2_u64)];
    let result = wand
        .flat_search(
            &FtsSearchParams::default(),
            Box::new(selected.into_iter()),
            &NoOpMetricsCollector,
        )
        .unwrap();

    let matched = result
        .into_iter()
        .map(|doc| doc.document)
        .collect::<Vec<_>>();
    assert_eq!(matched, vec![2]);
}

#[test]
fn test_doc_ids_resolves_every_document_a_row_owns() {
    // A list<string> column indexes each element as its own document, so
    // one row id owns several doc ids. row 100 -> {0, 1}, row 101 -> {2}.
    let row_id_col = arrow_array::UInt64Array::from(vec![100_u64, 100, 101]);
    let num_tokens_col = arrow_array::UInt32Array::from(vec![1_u32, 1, 1]);
    let docs = DocSet::from_columns(&row_id_col, &num_tokens_col, false, None).unwrap();

    assert_eq!(docs.doc_ids(100).collect::<Vec<_>>(), vec![0, 1]);
    assert_eq!(docs.doc_ids(101).collect::<Vec<_>>(), vec![2]);
    assert!(docs.doc_ids(999).next().is_none());

    // legacy shape (row id == doc id) still resolves to a single document.
    let mut legacy = DocSet::default();
    legacy.append(7, 1);
    assert_eq!(legacy.doc_ids(7).collect::<Vec<_>>(), vec![7]);
    assert!(legacy.doc_ids(8).next().is_none());
}

#[rstest]
fn test_flat_search_finds_list_row_with_match_at_non_last_position(
    #[values(false, true)] is_compressed: bool,
) {
    // row 100 owns two element-documents (doc 0, doc 1) that share its row
    // id; row 101 owns doc 2. The query term lives only in doc 0 — the
    // *non-last* element of row 100. Resolving the row to a single doc id
    // would evaluate doc 1, miss the term, and drop the row (lancedb#3352).
    let row_id_col = arrow_array::UInt64Array::from(vec![100_u64, 100, 101]);
    let num_tokens_col = arrow_array::UInt32Array::from(vec![1_u32, 1, 1]);
    let docs = DocSet::from_columns(&row_id_col, &num_tokens_col, false, None).unwrap();

    let posting = PostingIterator::with_query_weight(
        String::from("needle"),
        0,
        0,
        1.0,
        generate_posting_list(vec![0], 1.0, None, is_compressed),
        docs.len(),
    );

    let mut wand = Wand::new(
        Operator::Or,
        vec![posting].into_iter(),
        &docs,
        InverseDocLengthScorer,
    );
    wand.threshold = 0.5;

    let selected = docs
        .doc_ids(100)
        .map(|doc_id| (doc_id, 100_u64))
        .collect::<Vec<_>>();
    let result = wand
        .flat_search(
            &FtsSearchParams::default(),
            Box::new(selected.into_iter()),
            &NoOpMetricsCollector,
        )
        .unwrap();

    // The legacy adapter resolves the prefilter to every owned document,
    // while the candidate identity remains row 100.
    let addrs = result
        .into_iter()
        .map(|doc| doc.document)
        .collect::<Vec<_>>();
    assert!(
        addrs.as_slice() == [100],
        "expected exactly row 100, got {addrs:?}"
    );
}
