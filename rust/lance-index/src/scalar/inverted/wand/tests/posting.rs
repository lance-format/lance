// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

#[test]
fn find_next_geq_matches_partition_point_for_full_u32_domain() {
    let cases = [
        vec![],
        vec![7],
        vec![0, 1, 2, 3, 4, 5, 6, 7],
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8],
        (0..32).collect(),
        vec![
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            i32::MAX as u32,
            i32::MAX as u32 + 1,
            i32::MAX as u32 + 2,
            i32::MAX as u32 + 3,
            i32::MAX as u32 + 4,
            i32::MAX as u32 + 5,
            i32::MAX as u32 + 6,
            u32::MAX - 2,
            u32::MAX - 1,
            u32::MAX,
        ],
    ];
    let targets = [
        0,
        1,
        6,
        7,
        8,
        i32::MAX as u32 - 1,
        i32::MAX as u32,
        i32::MAX as u32 + 1,
        u32::MAX - 1,
        u32::MAX,
    ];

    for docs in &cases {
        for pos in 0..=docs.len() {
            for &target in &targets {
                let expected = pos + docs[pos..].partition_point(|&doc_id| doc_id < target);
                let actual = unsafe { find_next_geq(docs.as_ptr(), pos, docs.len(), target) };
                assert_eq!(
                    actual, expected,
                    "docs={docs:?}, pos={pos}, target={target}"
                );
                assert_eq!(
                    find_next_geq_in_block(docs, pos, target),
                    expected,
                    "block search: docs={docs:?}, pos={pos}, target={target}"
                );
            }
        }
    }
}

fn generate_contiguous_impact_posting_list_with_block_size(
    total: usize,
    block_size: usize,
) -> PostingList {
    generate_impact_posting_list_with_freqs_and_block_size(
        (0..total as u32).collect(),
        vec![1; total],
        vec![1; total],
        block_size,
    )
}

#[rstest]
#[case::packed_delta(PositionStreamCodec::PackedDelta)]
#[case::varint_doc_delta(PositionStreamCodec::VarintDocDelta)]
fn test_shared_position_cursors_use_independent_scratch(
    #[case] codec: PositionStreamCodec,
) -> Result<()> {
    let mut posting_list =
        generate_posting_list_with_positions(vec![0], vec![vec![1_u32, 3, 10]], 1.0, true);
    let PostingList::Compressed(ref mut list) = posting_list else {
        unreachable!("the helper was asked for a compressed posting list");
    };
    let mut encoded = Vec::new();
    encode_position_stream_block_into(&[1, 3, 10], &[3], codec, &mut encoded)?;
    list.positions = Some(CompressedPositionStorage::SharedStream(
        SharedPositionStream::new(codec, vec![0], bytes::Bytes::from(encoded)),
    ));

    let posting = PostingIterator::new(String::from("term"), 0, 0, posting_list, 1);
    let first = posting.position_cursor()?;
    let second = posting.position_cursor()?;

    assert_eq!(second.positions.as_slice(), &[1, 3, 10]);
    assert_eq!(first.positions.as_slice(), &[1, 3, 10]);
    assert!(posting.position_scratch.borrow().is_none());
    drop(second);
    assert!(posting.position_scratch.borrow().is_some());
    drop(first);
    assert!(posting.position_scratch.borrow().is_some());
    Ok(())
}

#[test]
fn test_posting_iterator_next_compressed_partition_point() {
    let mut docs = DocSet::default();
    let num_docs = (BLOCK_SIZE * 2 + 5) as u32;
    for i in 0..num_docs {
        docs.append(i as u64, 1);
    }

    let doc_ids = (0..num_docs).collect::<Vec<_>>();
    let posting = generate_posting_list(doc_ids, 1.0, None, true);
    let mut iter = PostingIterator::new(String::from("term"), 0, 0, posting, docs.len());

    iter.next(10);
    assert_eq!(iter.doc().unwrap().doc_id(), 10);

    let target = BLOCK_SIZE as u64 + 3;
    iter.next(target);
    assert_eq!(iter.doc().unwrap().doc_id(), target);

    iter.next(num_docs as u64 + 10);
    assert!(iter.doc().is_none());
}

#[test]
fn posting_iterator_fork_restarts_shared_compressed_payload() {
    let num_docs = (BLOCK_SIZE * 2 + 5) as u32;
    let posting = generate_posting_list((0..num_docs).collect(), 1.0, None, true);
    let mut original = PostingIterator::new(String::from("term"), 7, 3, posting, num_docs as usize);

    original.next(BLOCK_SIZE as u64 + 3);
    assert_eq!(
        original.doc().map(|doc| doc.doc_id()),
        Some(BLOCK_SIZE as u64 + 3)
    );

    let mut replay = original.fork_from_start();
    assert_eq!(replay.doc().map(|doc| doc.doc_id()), Some(0));
    replay.next(5);
    assert_eq!(replay.doc().map(|doc| doc.doc_id()), Some(5));
    assert_eq!(
        original.doc().map(|doc| doc.doc_id()),
        Some(BLOCK_SIZE as u64 + 3),
        "fork advancement must not mutate the membership cursor"
    );
}

#[test]
fn test_block_max_score_up_to_slides_and_expires_old_max() {
    let total = 5 * BLOCK_SIZE as u32;
    let posting = generate_posting_list(
        (0..total).collect(),
        5.0,
        Some(vec![1.0, 4.0, 2.0, 5.0, 3.0]),
        true,
    );
    let mut posting = PostingIterator::new(String::from("term"), 0, 0, posting, total as usize);

    posting.shallow_next(0);
    assert_eq!(
        posting
            .block_max_score_up_to_with_stats((3 * BLOCK_SIZE - 1) as u64, &UnitScorer)
            .score,
        4.0
    );

    posting.shallow_next((2 * BLOCK_SIZE) as u64);
    assert_eq!(
        posting
            .block_max_score_up_to_with_stats((4 * BLOCK_SIZE - 1) as u64, &UnitScorer)
            .score,
        5.0
    );

    posting.shallow_next((4 * BLOCK_SIZE) as u64);
    assert_eq!(
        posting
            .block_max_score_up_to_with_stats((5 * BLOCK_SIZE - 1) as u64, &UnitScorer)
            .score,
        3.0
    );
}

#[test]
fn test_impact_level1_skip_keeps_boundary_equality_in_group() {
    for block_size in [crate::scalar::inverted::LEGACY_BLOCK_SIZE, 256] {
        let total = (IMPACT_LEVEL1_BLOCKS + 1) * block_size;
        let mut posting = PostingIterator::new(
            String::from("term"),
            0,
            0,
            generate_contiguous_impact_posting_list_with_block_size(total, block_size),
            total,
        );
        let target = (IMPACT_LEVEL1_BLOCKS * block_size - 1) as u64;

        posting.shallow_next(target);
        assert_eq!(posting.block_idx, IMPACT_LEVEL1_BLOCKS - 1);

        posting.next(target);
        assert_eq!(posting.block_idx, IMPACT_LEVEL1_BLOCKS - 1);
        assert_eq!(posting.doc().map(|doc| doc.doc_id()), Some(target));
    }
}

#[test]
fn test_impact_level1_skip_handles_partial_final_group() {
    for block_size in [crate::scalar::inverted::LEGACY_BLOCK_SIZE, 256] {
        let total = (IMPACT_LEVEL1_BLOCKS + 3) * block_size + 17;
        let mut posting = PostingIterator::new(
            String::from("term"),
            0,
            0,
            generate_contiguous_impact_posting_list_with_block_size(total, block_size),
            total,
        );
        let target = (total - 1) as u64;
        let expected_block = total.div_ceil(block_size) - 1;

        posting.shallow_next(target);
        assert_eq!(posting.block_idx, expected_block);

        posting.next(target);
        assert_eq!(posting.block_idx, expected_block);
        assert_eq!(posting.doc().map(|doc| doc.doc_id()), Some(target));
    }
}

#[test]
fn test_impact_level1_skip_reaches_far_target_doc() {
    for block_size in [crate::scalar::inverted::LEGACY_BLOCK_SIZE, 256] {
        let total = (IMPACT_LEVEL1_BLOCKS * 3 + 5) * block_size;
        let target_block = IMPACT_LEVEL1_BLOCKS * 2 + 2;
        let target = (target_block * block_size + 17) as u64;
        let mut posting = PostingIterator::new(
            String::from("term"),
            0,
            0,
            generate_contiguous_impact_posting_list_with_block_size(total, block_size),
            total,
        );

        posting.shallow_next(target);
        assert_eq!(posting.block_idx, target_block);

        posting.next(target);
        assert_eq!(posting.block_idx, target_block);
        assert_eq!(posting.doc().map(|doc| doc.doc_id()), Some(target));
    }
}

#[test]
fn test_compressed_impact_block_max_score_memoizes_current_block() {
    let total = 2 * BLOCK_SIZE as u32;
    let doc_ids = (0..total).collect::<Vec<_>>();
    let freqs = doc_ids
        .iter()
        .map(|doc_id| if *doc_id < BLOCK_SIZE as u32 { 1 } else { 2 })
        .collect::<Vec<_>>();
    let doc_lengths = vec![1; total as usize];
    let posting_list = generate_impact_posting_list_with_freqs(doc_ids, freqs, doc_lengths);
    let mut posting =
        PostingIterator::new(String::from("term"), 0, 0, posting_list, total as usize);
    let scored = Arc::new(AtomicUsize::new(0));
    let scorer = CountingScorer {
        scored: scored.clone(),
    };

    let first_score = posting.block_max_score(&scorer);
    assert_eq!(first_score, 1.0);
    // Baking the query-local doc-weight bounds visits every frontier pair
    // once (two level0 entries plus one level1 entry for this list).
    let baked = scored.load(Ordering::Relaxed);
    assert!(baked >= 2);
    {
        let compressed = unsafe { &mut *posting.compressed_state_ptr() };
        assert_eq!(
            compressed.level0_cache,
            Some((0, BLOCK_SIZE as u32 - 1, first_score))
        );
    }

    let second_score = posting.block_max_score(&scorer);
    assert_eq!(second_score, first_score);
    assert_eq!(
        scored.load(Ordering::Relaxed),
        baked,
        "repeated block max scores must not recompute doc weights"
    );

    posting.shallow_next(BLOCK_SIZE as u64);
    let next_block_score = posting.block_max_score(&scorer);
    assert_eq!(next_block_score, 2.0);
    assert_eq!(
        scored.load(Ordering::Relaxed),
        baked,
        "other blocks answer from the baked bounds without rescoring"
    );
}

#[rstest]
#[case(0.0)]
#[case(-1.0)]
fn test_non_positive_query_weight_skips_global_impact_bound(#[case] query_weight: f32) {
    let posting = PostingIterator::with_query_weight(
        String::from("term"),
        0,
        0,
        query_weight,
        generate_impact_posting_list_with_freqs(vec![0], vec![1], vec![1]),
        1,
    );
    let scored = Arc::new(AtomicUsize::new(0));
    let scorer = CountingScorer {
        scored: scored.clone(),
    };

    assert_eq!(posting.global_upper_bound(&scorer), 0.0);
    assert_eq!(scored.load(Ordering::Relaxed), 0);
}

#[test]
fn test_block_max_score_matches_stored_value() {
    let doc_ids = vec![0_u32];
    let block_max_scores = vec![0.7_f32];
    let posting_list = generate_posting_list(doc_ids, 0.7, Some(block_max_scores), true);
    let expected = match &posting_list {
        PostingList::Compressed(list) => list.block_max_score(0),
        PostingList::Plain(_) => unreachable!("expected compressed posting list"),
    };

    let posting = PostingIterator::new(String::from("test"), 0, 0, posting_list, 1);

    let actual = posting.block_max_score(&UnitScorer);
    assert!(
        (actual - expected).abs() < 1e-6,
        "block max score should match stored value"
    );
}

#[test]
fn test_modern_doc_id_validation_checks_layout_and_upper_bound() {
    let compressed = generate_posting_list(vec![0, 4], 1.0, None, true);
    let posting = PostingIterator::new(String::from("term"), 0, 0, compressed, 5);
    posting
        .validate_modern_doc_ids(5)
        .expect("largest DocId is inside the document table");
    let error = posting.validate_modern_doc_ids(4).unwrap_err();
    assert!(error.to_string().contains("DocId 4"));
    assert!(error.to_string().contains("[0, 4)"));

    let plain = generate_posting_list(vec![0], 1.0, None, false);
    let posting = PostingIterator::new(String::from("legacy"), 0, 0, plain, 1);
    let error = posting.validate_modern_doc_ids(1).unwrap_err();
    assert!(error.to_string().contains("legacy row-address layout"));
}

#[test]
fn test_256_document_blocks_without_impacts_use_conservative_quantized_score_bound() {
    let exact_doc_length = 300;
    let quantized_doc_length = crate::scalar::inverted::index::dequantize_doc_length(
        crate::scalar::inverted::index::quantize_doc_length(exact_doc_length),
    );
    assert!(quantized_doc_length < exact_doc_length);

    let scorer = Arc::new(MemBM25Scorer::new(100, 1, Default::default()));
    let stored_exact_score = scorer.doc_weight(1, exact_doc_length);
    let quantized_score = scorer.doc_weight(1, quantized_doc_length);
    assert!(quantized_score > stored_exact_score);

    let doc_ids = [0_u32];
    let frequencies = [1_u32];
    let blocks = compress_posting_list_with_tail_codec_and_block_size(
        doc_ids.len(),
        doc_ids.iter(),
        frequencies.iter(),
        std::iter::once(stored_exact_score),
        crate::scalar::inverted::PostingTailCodec::VarintDelta,
        MAX_POSTING_BLOCK_SIZE,
    )
    .unwrap();
    let posting_list = PostingList::Compressed(CompressedPostingList::new(
        blocks,
        stored_exact_score,
        doc_ids.len() as u32,
        crate::scalar::inverted::PostingTailCodec::VarintDelta,
        MAX_POSTING_BLOCK_SIZE,
        None,
        None,
    ));
    let posting = PostingIterator::new(String::from("term"), 0, 0, posting_list, doc_ids.len());
    let expected_bound = BM25_DOC_WEIGHT_UPPER_BOUND;

    assert_eq!(posting.approximate_upper_bound(), expected_bound);
    assert_eq!(posting.global_upper_bound(&scorer), expected_bound);
    assert_eq!(posting.block_max_score(&scorer), expected_bound);
    assert_eq!(
        posting.block_max_score_up_to_with_stats(0, &scorer).score,
        expected_bound
    );
    assert!(expected_bound >= quantized_score);
}

#[test]
fn test_256_document_blocks_without_impacts_unknown_scorer_uses_infinite_bound() {
    let doc_ids = [0_u32];
    let frequencies = [10_u32];
    let blocks = compress_posting_list_with_tail_codec_and_block_size(
        doc_ids.len(),
        doc_ids.iter(),
        frequencies.iter(),
        std::iter::once(10.0),
        crate::scalar::inverted::PostingTailCodec::VarintDelta,
        MAX_POSTING_BLOCK_SIZE,
    )
    .unwrap();
    let posting_list = PostingList::Compressed(CompressedPostingList::new(
        blocks,
        10.0,
        doc_ids.len() as u32,
        crate::scalar::inverted::PostingTailCodec::VarintDelta,
        MAX_POSTING_BLOCK_SIZE,
        None,
        None,
    ));
    let posting = PostingIterator::new(String::from("term"), 0, 0, posting_list, doc_ids.len());

    assert!(posting.global_upper_bound(&UnitScorer).is_infinite());
    assert!(posting.block_max_score(&UnitScorer).is_infinite());
    assert!(
        posting
            .block_max_score_up_to_with_stats(0, &UnitScorer)
            .score
            .is_infinite()
    );
}
