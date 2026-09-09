// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::super::partition::validate_no_impact_scorer_upper_bound;
use super::*;

#[tokio::test]
async fn test_bm25_search_uses_global_idf() {
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    // Partition 0: 3 docs, only one contains "alpha".
    let mut builder0 = InnerBuilder::new(0, false, TokenSetFormat::default());
    builder0.tokens.add("alpha".to_owned());
    builder0.tokens.add("beta".to_owned());
    builder0.posting_lists.push(PostingListBuilder::new(false));
    builder0.posting_lists.push(PostingListBuilder::new(false));
    builder0.posting_lists[0].add(0, PositionRecorder::Count(1));
    builder0.posting_lists[1].add(1, PositionRecorder::Count(1));
    builder0.posting_lists[1].add(2, PositionRecorder::Count(1));
    builder0.docs.append(100, 1);
    builder0.docs.append(101, 1);
    builder0.docs.append(102, 1);
    builder0.write(store.as_ref()).await.unwrap();

    // Partition 1: 1 doc, contains "alpha".
    let mut builder1 = InnerBuilder::new(1, false, TokenSetFormat::default());
    builder1.tokens.add("alpha".to_owned());
    builder1.posting_lists.push(PostingListBuilder::new(false));
    builder1.posting_lists[0].add(0, PositionRecorder::Count(1));
    builder1.docs.append(200, 1);
    builder1.write(store.as_ref()).await.unwrap();

    let metadata = std::collections::HashMap::from_iter(vec![
        (
            "partitions".to_owned(),
            serde_json::to_string(&vec![0u64, 1u64]).unwrap(),
        ),
        (
            "params".to_owned(),
            serde_json::to_string(&InvertedIndexParams::default()).unwrap(),
        ),
        (
            TOKEN_SET_FORMAT_KEY.to_owned(),
            TokenSetFormat::default().to_string(),
        ),
    ]);
    let mut writer = store
        .new_index_file(METADATA_FILE, Arc::new(arrow_schema::Schema::empty()))
        .await
        .unwrap();
    writer.finish_with_metadata(metadata).await.unwrap();

    let cache = Arc::new(LanceCache::with_capacity(4096));
    let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
        .await
        .unwrap();

    let tokens = Arc::new(Tokens::new(vec!["alpha".to_string()], DocType::Text));
    let params = Arc::new(FtsSearchParams::new().with_limit(Some(10)));
    let prefilter = Arc::new(NoFilter);
    let metrics = Arc::new(NoOpMetricsCollector);

    let (row_ids, scores) = index
        .bm25_search(tokens, params, Operator::Or, prefilter, metrics, None)
        .await
        .unwrap();

    assert_eq!(row_ids.len(), 2);
    assert!(row_ids.contains(&100));
    assert!(row_ids.contains(&200));
    assert_eq!(row_ids.len(), scores.len());

    let expected_idf = idf(2, 4);
    for score in scores {
        assert!(
            (score - expected_idf).abs() < 1e-6,
            "score: {}, expected: {}",
            score,
            expected_idf
        );
    }
}

async fn write_test_partition_with_optional_impacts(
    store: &Arc<LanceIndexStore>,
    partition_id: u64,
    builder: InnerBuilder,
    token_set_format: TokenSetFormat,
    with_impacts: bool,
) {
    write_test_partition_with_optional_impacts_and_positions(
        store,
        partition_id,
        builder,
        token_set_format,
        with_impacts,
        false,
    )
    .await;
}

async fn write_test_partition_with_optional_impacts_and_positions(
    store: &Arc<LanceIndexStore>,
    partition_id: u64,
    mut builder: InnerBuilder,
    token_set_format: TokenSetFormat,
    with_impacts: bool,
    with_positions: bool,
) {
    let format_version = InvertedListFormatVersion::V1;
    let block_size = LEGACY_BLOCK_SIZE;
    let docs = std::mem::take(&mut builder.docs);
    let schema = inverted_list_schema_for_version_with_block_size_and_impacts(
        with_positions,
        format_version,
        block_size,
        with_impacts,
    );

    let mut posting_writer = store
        .new_index_file(&posting_file_path(partition_id), schema.clone())
        .await
        .unwrap();
    for posting_list in std::mem::take(&mut builder.posting_lists) {
        let batch = posting_list
            .to_batch_with_docs(&docs, schema.clone())
            .unwrap();
        posting_writer.write_record_batch(batch).await.unwrap();
    }
    posting_writer.finish().await.unwrap();

    let token_batch = std::mem::take(&mut builder.tokens)
        .to_batch(token_set_format)
        .unwrap();
    let mut token_writer = store
        .new_index_file(&token_file_path(partition_id), token_batch.schema())
        .await
        .unwrap();
    token_writer.write_record_batch(token_batch).await.unwrap();
    token_writer.finish().await.unwrap();

    let doc_batch = docs.to_batch().unwrap();
    let mut doc_writer = store
        .new_index_file(&doc_file_path(partition_id), doc_batch.schema())
        .await
        .unwrap();
    doc_writer.write_record_batch(doc_batch).await.unwrap();
    doc_writer.finish().await.unwrap();
}

async fn load_single_partition_test_index(
    builder: InnerBuilder,
    with_impacts: bool,
) -> (TempObjDir, Arc<LanceCache>, Arc<InvertedIndex>) {
    load_test_index(vec![(0, builder, with_impacts)]).await
}

async fn load_test_index(
    partitions: Vec<(u64, InnerBuilder, bool)>,
) -> (TempObjDir, Arc<LanceCache>, Arc<InvertedIndex>) {
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    let mut partition_ids = Vec::with_capacity(partitions.len());
    for (partition_id, builder, with_impacts) in partitions {
        write_test_partition_with_optional_impacts(
            &store,
            partition_id,
            builder,
            TokenSetFormat::default(),
            with_impacts,
        )
        .await;
        partition_ids.push(partition_id);
    }
    write_test_metadata(&store, partition_ids, InvertedIndexParams::default()).await;
    let cache = Arc::new(LanceCache::with_capacity(4096));
    let index = InvertedIndex::load(store, None, cache.as_ref())
        .await
        .unwrap();
    (tmpdir, cache, index)
}

async fn load_global_scoring_test_index(
    first_partition_has_impacts: bool,
    second_partition_has_impacts: bool,
) -> (TempObjDir, Arc<LanceCache>, Arc<InvertedIndex>) {
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    let partition_specs = [
        (0, 100, 5_000, 101..111, 5_000, first_partition_has_impacts),
        (1, 200, 1_000, 201..301, 1, second_partition_has_impacts),
    ];
    for (
        partition_id,
        matching_row_id,
        matching_doc_length,
        other_row_ids,
        other_doc_length,
        with_impacts,
    ) in partition_specs
    {
        let mut builder = InnerBuilder::new_with_format_version(
            partition_id,
            false,
            TokenSetFormat::default(),
            InvertedListFormatVersion::V1,
        );
        builder.tokens.add("alpha".to_owned());
        builder
            .posting_lists
            .push(PostingListBuilder::new_with_posting_tail_codec(
                false,
                InvertedListFormatVersion::V1.posting_tail_codec(),
            ));
        builder.posting_lists[0].add(0, PositionRecorder::Count(1));
        builder.docs.append(matching_row_id, matching_doc_length);
        for row_id in other_row_ids {
            builder.docs.append(row_id, other_doc_length);
        }
        write_test_partition_with_optional_impacts(
            &store,
            partition_id,
            builder,
            TokenSetFormat::default(),
            with_impacts,
        )
        .await;
    }

    write_test_metadata(&store, vec![0, 1], InvertedIndexParams::default()).await;
    let cache = Arc::new(LanceCache::with_backend(Arc::new(
        QuickCacheBackend::with_capacity(4096),
    )));
    let index = InvertedIndex::load(store, None, cache.as_ref())
        .await
        .unwrap();
    (tmpdir, cache, index)
}

#[tokio::test]
async fn test_wand_exactness_certificate_support_requires_all_impact_postings() {
    let (_tmpdir, _cache, mixed_impact_index) = load_global_scoring_test_index(true, false).await;
    assert!(!mixed_impact_index.is_legacy());
    assert!(!mixed_impact_index.supports_wand_exactness_certificate());

    let (_tmpdir, _cache, all_impact_index) = load_global_scoring_test_index(true, true).await;
    assert!(!all_impact_index.is_legacy());
    assert!(all_impact_index.supports_wand_exactness_certificate());
}

#[tokio::test]
async fn test_no_impact_segments_preserve_global_bm25_top_k() {
    // Segment-local BM25 strongly favors beta in the first segment, while
    // corpus-wide IDF makes every alpha row the true global winner.
    let mut alpha_partition = InnerBuilder::new_with_format_version(
        0,
        false,
        TokenSetFormat::default(),
        InvertedListFormatVersion::V1,
    );
    alpha_partition.tokens.add("alpha".to_owned());
    alpha_partition
        .posting_lists
        .push(PostingListBuilder::new_with_posting_tail_codec(
            false,
            InvertedListFormatVersion::V1.posting_tail_codec(),
        ));
    for doc_id in 0_u32..98 {
        alpha_partition.posting_lists[0].add(doc_id, PositionRecorder::Count(1));
        alpha_partition.docs.append(u64::from(doc_id), 1);
    }

    let mut beta_partition = InnerBuilder::new_with_format_version(
        1,
        false,
        TokenSetFormat::default(),
        InvertedListFormatVersion::V1,
    );
    beta_partition.tokens.add("beta".to_owned());
    beta_partition
        .posting_lists
        .push(PostingListBuilder::new_with_posting_tail_codec(
            false,
            InvertedListFormatVersion::V1.posting_tail_codec(),
        ));
    beta_partition.posting_lists[0].add(0, PositionRecorder::Count(10));
    beta_partition.docs.append(98, 10);
    beta_partition.posting_lists[0].add(1, PositionRecorder::Count(5));
    beta_partition.docs.append(99, 10);

    let mut second_segment = InnerBuilder::new_with_format_version(
        0,
        false,
        TokenSetFormat::default(),
        InvertedListFormatVersion::V1,
    );
    second_segment.tokens.add("beta".to_owned());
    second_segment
        .posting_lists
        .push(PostingListBuilder::new_with_posting_tail_codec(
            false,
            InvertedListFormatVersion::V1.posting_tail_codec(),
        ));
    for doc_id in 0_u32..10_000 {
        second_segment.posting_lists[0].add(doc_id, PositionRecorder::Count(1));
        second_segment.docs.append(1_001 + u64::from(doc_id), 1);
    }

    // One segment itself mixes a no-impact partition with an impact-backed
    // partition. The second segment is no-impact, matching a rolling upgrade.
    let (_first_tmpdir, _first_cache, first_index) =
        load_test_index(vec![(0, alpha_partition, false), (1, beta_partition, true)]).await;
    let (_second_tmpdir, _second_cache, second_index) =
        load_single_partition_test_index(second_segment, false).await;
    for index in [&first_index, &second_index] {
        assert!(!index.is_legacy());
        assert!(!index.supports_wand_exactness_certificate());
    }

    let scorer = MemBM25Scorer::new(
        10_118,
        10_100,
        HashMap::from([("alpha".to_owned(), 98), ("beta".to_owned(), 10_002)]),
    );
    let tokens = Arc::new(Tokens::new(
        vec!["alpha".to_owned(), "beta".to_owned()],
        DocType::Text,
    ));
    let params = Arc::new(FtsSearchParams::new().with_limit(Some(2)));
    let mut candidates = Vec::new();
    let metrics = Arc::new(NoOpMetricsCollector);
    // Reverse segment visitation so neither the result nor its row-id tie break
    // can accidentally depend on the physical search order.
    for index in [second_index, first_index] {
        let (row_ids, scores) = index
            .bm25_search(
                tokens.clone(),
                params.clone(),
                Operator::Or,
                Arc::new(NoFilter),
                metrics.clone(),
                Some(&scorer),
            )
            .await
            .unwrap();
        candidates.extend(row_ids.into_iter().zip(scores));
    }
    candidates.sort_unstable_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    candidates.truncate(2);

    // Before the no-impact fallback used the corpus scorer, the bounded
    // partition-local candidate set was [98, 1001]. It omitted the alpha tie
    // group even though row 0 is the true global winner.
    assert_eq!(
        candidates
            .iter()
            .map(|candidate| candidate.0)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(candidates[0].1, candidates[1].1);

    let exact_winner_score = scorer.query_weight("alpha") * scorer.doc_weight(1, 1);
    assert!((exact_winner_score - 4.633_705).abs() < 1e-5);
    assert!((exact_winner_score - candidates[0].1).abs() < 1e-5);
}

#[test]
fn test_global_query_weight_validation_rejects_invalid_values() {
    let scorer = MemBM25Scorer::new(1, 1, HashMap::from([("alpha".to_owned(), 10)]));
    let error = validate_no_impact_scorer_upper_bound("alpha", &scorer).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    let message = error.to_string();
    assert!(message.contains("token \"alpha\""), "{message}");
    assert!(message.contains("got -"), "{message}");
}

#[tokio::test]
async fn test_no_impact_search_rejects_injected_negative_query_weight() {
    let mut builder = InnerBuilder::new_with_format_version(
        0,
        false,
        TokenSetFormat::default(),
        InvertedListFormatVersion::V1,
    );
    builder.tokens.add("alpha".to_owned());
    builder
        .posting_lists
        .push(PostingListBuilder::new_with_posting_tail_codec(
            false,
            InvertedListFormatVersion::V1.posting_tail_codec(),
        ));
    builder.posting_lists[0].add(0, PositionRecorder::Count(1));
    builder.docs.append(0, 1);
    let (_tmpdir, _cache, index) = load_single_partition_test_index(builder, false).await;
    let scorer = MemBM25Scorer::new(1, 1, HashMap::from([("alpha".to_owned(), 10)]));
    let error = index
        .bm25_search(
            Arc::new(Tokens::new(vec!["alpha".to_owned()], DocType::Text)),
            Arc::new(FtsSearchParams::new().with_limit(Some(1))),
            Operator::Or,
            Arc::new(NoFilter),
            Arc::new(NoOpMetricsCollector),
            Some(&scorer),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(
        error.to_string().contains(
            "global BM25 query weight for token \"alpha\" must be finite and non-negative"
        )
    );
}

#[tokio::test]
async fn test_chunked_modern_search_preserves_cold_and_prewarmed_results() {
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    let matching_partitions = 17_u64;
    for partition_id in 0..matching_partitions {
        let mut builder = InnerBuilder::new(partition_id, false, TokenSetFormat::default());
        builder.tokens.add("pipeline".to_owned());
        builder.posting_lists.push(PostingListBuilder::new(false));
        builder.posting_lists[0].add(0, PositionRecorder::Count(1));
        builder.docs.append(partition_id * 1_000 + 7, 1);
        builder.write(store.as_ref()).await.unwrap();
    }
    let unmatched_partition = matching_partitions;
    let mut builder = InnerBuilder::new(unmatched_partition, false, TokenSetFormat::default());
    builder.tokens.add("unrelated".to_owned());
    builder.posting_lists.push(PostingListBuilder::new(false));
    builder.posting_lists[0].add(0, PositionRecorder::Count(1));
    builder.docs.append(999_999, 1);
    builder.write(store.as_ref()).await.unwrap();

    write_test_metadata(
        &store,
        (0..=unmatched_partition).collect(),
        InvertedIndexParams::default(),
    )
    .await;
    let cache = Arc::new(LanceCache::with_capacity(64 * 1024 * 1024));
    let index = InvertedIndex::load(store, None, cache.as_ref())
        .await
        .unwrap();
    let tokens = Arc::new(Tokens::new(vec!["pipeline".to_owned()], DocType::Text));
    let params = Arc::new(FtsSearchParams::new().with_limit(Some(matching_partitions as usize)));

    let search = || {
        index.bm25_search(
            tokens.clone(),
            params.clone(),
            Operator::Or,
            Arc::new(NoFilter),
            Arc::new(NoOpMetricsCollector),
            None,
        )
    };
    let (mut cold_row_ids, cold_scores) = search().await.unwrap();
    cold_row_ids.sort_unstable();
    let expected = (0..matching_partitions)
        .map(|partition_id| partition_id * 1_000 + 7)
        .collect::<Vec<_>>();
    assert_eq!(cold_row_ids, expected);
    assert_eq!(cold_scores.len(), expected.len());

    index
        .prewarm_with_options(&FtsPrewarmOptions::default())
        .await
        .unwrap();
    let (mut prewarmed_row_ids, prewarmed_scores) = search().await.unwrap();
    prewarmed_row_ids.sort_unstable();
    assert_eq!(prewarmed_row_ids, expected);
    assert_eq!(prewarmed_scores, cold_scores);
}

#[tokio::test]
async fn test_prewarmed_modern_search_uses_resident_address_projection() {
    let (_tmpdir, cache, index) = load_global_scoring_test_index(true, true).await;
    let tokens = Arc::new(Tokens::new(vec!["alpha".to_owned()], DocType::Text));
    let params = Arc::new(FtsSearchParams::new().with_limit(Some(2)));

    assert!(!index.has_resident_document_projections());
    let deferred = index
        .bm25_search(
            tokens.clone(),
            params.clone(),
            Operator::Or,
            Arc::new(NoFilter),
            Arc::new(NoOpMetricsCollector),
            None,
        )
        .await
        .unwrap();
    assert!(!index.has_resident_document_projections());

    index.partitions[0]
        .docs
        .modern()
        .unwrap()
        .prewarm()
        .await
        .unwrap();
    assert!(index.partitions[0].docs.query_ready());
    assert!(!index.has_resident_document_projections());
    let partially_resident = index
        .bm25_search(
            tokens.clone(),
            params.clone(),
            Operator::Or,
            Arc::new(NoFilter),
            Arc::new(NoOpMetricsCollector),
            None,
        )
        .await
        .unwrap();
    assert_eq!(partially_resident, deferred);

    let prewarm_options = FtsPrewarmOptions::default();
    futures::future::join_all((0..8).map(|_| index.prewarm_with_options(&prewarm_options)))
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()
        .unwrap();
    assert!(index.document_projections_resident.load(Ordering::Acquire));
    assert!(index.has_resident_document_projections());
    assert!(index.corpus_stats.initialized());
    assert!(index.partitions.iter().all(|partition| {
        partition.docs.query_ready() && partition.inverted_list.modern_posting_validation_ready()
    }));
    assert!(index.prewarm_state.lock().await.satisfies(false));

    let resident = index
        .bm25_search(
            tokens.clone(),
            params.clone(),
            Operator::Or,
            Arc::new(NoFilter),
            Arc::new(NoOpMetricsCollector),
            None,
        )
        .await
        .unwrap();
    assert_eq!(resident, deferred);
    assert_eq!(resident.0.len(), 2);
    assert!(resident.0.contains(&100));
    assert!(resident.0.contains(&200));

    cache.clear().await;
    assert!(index.document_projections_resident.load(Ordering::Acquire));
    assert_eq!(cache.size().await, 0);
    let resident_address_owners = index
        .partitions
        .iter()
        .map(|partition| {
            partition
                .docs
                .modern()
                .unwrap()
                .address_buffer_handle()
                .strong_count()
        })
        .collect::<Vec<_>>();
    assert_eq!(resident_address_owners, vec![0, 0]);
    assert!(
        index
            .partitions
            .iter()
            .all(|partition| { !partition.docs.modern().unwrap().projection_resident() })
    );

    let after_eviction = index
        .bm25_search(
            tokens.clone(),
            params.clone(),
            Operator::Or,
            Arc::new(NoFilter),
            Arc::new(NoOpMetricsCollector),
            None,
        )
        .await
        .unwrap();
    assert_eq!(after_eviction, deferred);
    assert!(!index.document_projections_resident.load(Ordering::Acquire));

    cache.clear().await;
    index.prewarm_with_options(&prewarm_options).await.unwrap();
    assert!(index.document_projections_resident_now());
    assert!(index.document_projections_resident.load(Ordering::Acquire));

    let re_prewarms_after_eviction = index
        .bm25_search(
            tokens,
            params,
            Operator::Or,
            Arc::new(NoFilter),
            Arc::new(NoOpMetricsCollector),
            None,
        )
        .await
        .unwrap();
    assert_eq!(re_prewarms_after_eviction, deferred);
}

#[tokio::test]
async fn test_resident_modern_search_loads_partition_stats_without_global_stats() {
    let (_tmpdir, _cache, index) = load_global_scoring_test_index(true, false).await;
    assert!(index.corpus_stats.get().is_none());
    assert!(
        index
            .partitions
            .iter()
            .all(|partition| partition.docs.cached_stats().is_none())
    );

    for partition in &index.partitions {
        partition
            .docs
            .modern()
            .unwrap()
            .address_projection()
            .await
            .unwrap();
    }
    assert!(index.has_resident_document_projections());

    let scorer = MemBM25Scorer::new(56_100, 112, HashMap::from([("alpha".to_owned(), 2)]));
    let result = index
        .bm25_search(
            Arc::new(Tokens::new(vec!["alpha".to_owned()], DocType::Text)),
            Arc::new(FtsSearchParams::new().with_limit(Some(2))),
            Operator::Or,
            Arc::new(NoFilter),
            Arc::new(NoOpMetricsCollector),
            Some(&scorer),
        )
        .await
        .unwrap();

    assert_eq!(result.0.len(), 2);
    assert!(result.0.contains(&100));
    assert!(result.0.contains(&200));
    assert!(index.corpus_stats.get().is_none());
    assert!(
        index
            .partitions
            .iter()
            .all(|partition| partition.docs.cached_stats().is_some())
    );
}

async fn search_test_impact_partition(
    partition: &InvertedPartition,
    tokens: &Tokens,
    params: &FtsSearchParams,
    scorer: Arc<MemBM25Scorer>,
    shared_threshold: Arc<AtomicU32>,
) -> Vec<DocCandidate<DocId>> {
    let LoadedPostings {
        postings,
        grouped_expansions,
        impact_safe,
        exact_scoring_required,
        no_impact_fallback,
    } = partition
        .load_posting_lists(
            tokens,
            params,
            Operator::Or,
            scorer.as_ref(),
            &NoOpMetricsCollector,
            false,
        )
        .await
        .unwrap();
    assert!(impact_safe);
    assert!(!exact_scoring_required);
    assert!(!no_impact_fallback);
    assert!(grouped_expansions.is_empty());

    let documents = partition.docs.modern().unwrap();
    let lengths = documents.lengths().await.unwrap();
    let visibility = documents.visibility(NoFilter.mask(), false).await.unwrap();
    partition
        .bm25_search_modern(
            lengths.as_ref(),
            &visibility,
            params,
            Operator::Or,
            postings,
            Some(scorer),
            &NoOpMetricsCollector,
            shared_threshold,
        )
        .unwrap()
}

async fn load_no_impact_bulk_conjunction_test_index(
    with_positions: bool,
) -> (TempObjDir, Arc<LanceCache>, Arc<InvertedIndex>) {
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    let mut floor_partition = InnerBuilder::new_with_format_version(
        0,
        with_positions,
        TokenSetFormat::default(),
        InvertedListFormatVersion::V1,
    );
    let mut winner_partition = InnerBuilder::new_with_format_version(
        1,
        with_positions,
        TokenSetFormat::default(),
        InvertedListFormatVersion::V1,
    );
    for builder in [&mut floor_partition, &mut winner_partition] {
        for token in ["lead", "follow"] {
            builder.tokens.add(token.to_owned());
            builder
                .posting_lists
                .push(PostingListBuilder::new_with_posting_tail_codec(
                    with_positions,
                    InvertedListFormatVersion::V1.posting_tail_codec(),
                ));
        }
    }

    if with_positions {
        floor_partition.posting_lists[0].add(0, PositionRecorder::Position(vec![0].into()));
        floor_partition.posting_lists[1].add(0, PositionRecorder::Position(vec![1].into()));
    } else {
        floor_partition.posting_lists[0].add(0, PositionRecorder::Count(1));
        floor_partition.posting_lists[1].add(0, PositionRecorder::Count(1));
    }
    floor_partition.docs.append(100, 2);
    for row_id in 101..10_100 {
        floor_partition.docs.append(row_id, 100);
    }

    let lead_positions = (0..63).map(|position| position * 2).collect::<Vec<_>>();
    let follow_positions = (0..63).map(|position| position * 2 + 1).collect::<Vec<_>>();
    for doc_id in 0_u32..100 {
        if with_positions {
            winner_partition.posting_lists[0].add(
                doc_id,
                PositionRecorder::Position(lead_positions.clone().into()),
            );
            winner_partition.posting_lists[1].add(
                doc_id,
                PositionRecorder::Position(follow_positions.clone().into()),
            );
        } else {
            winner_partition.posting_lists[0].add(doc_id, PositionRecorder::Count(63));
            winner_partition.posting_lists[1].add(doc_id, PositionRecorder::Count(63));
        }
        winner_partition
            .docs
            .append(20_000 + u64::from(doc_id), 126);
    }

    for (partition_id, builder) in [(0, floor_partition), (1, winner_partition)] {
        write_test_partition_with_optional_impacts_and_positions(
            &store,
            partition_id,
            builder,
            TokenSetFormat::default(),
            false,
            with_positions,
        )
        .await;
    }
    let params = InvertedIndexParams::default().with_position(with_positions);
    write_test_metadata(&store, vec![0, 1], params).await;
    let cache = Arc::new(LanceCache::with_capacity(4096));
    let index = InvertedIndex::load(store, None, cache.as_ref())
        .await
        .unwrap();
    (tmpdir, cache, index)
}

async fn assert_no_impact_bulk_conjunction_preserves_winner(with_phrase: bool) {
    let (_tmpdir, _cache, index) = load_no_impact_bulk_conjunction_test_index(with_phrase).await;
    let tokens = Arc::new(Tokens::new(
        vec!["lead".to_owned(), "follow".to_owned()],
        DocType::Text,
    ));
    let params = Arc::new(
        FtsSearchParams::new()
            .with_limit(Some(1))
            .with_phrase_slop(with_phrase.then_some(0)),
    );
    let scorer = Arc::new(
        index
            .bm25_base_scorer(tokens.as_ref(), params.as_ref(), None)
            .await
            .unwrap(),
    );
    let shared_threshold = Arc::new(AtomicU32::new(f32::NEG_INFINITY.to_bits()));
    let mut results = Vec::new();
    let mut published_floors = Vec::new();
    for partition_id in [0, 1] {
        let partition = index
            .partitions
            .iter()
            .find(|partition| partition.id() == partition_id)
            .unwrap();
        let LoadedPostings {
            postings,
            grouped_expansions,
            impact_safe,
            exact_scoring_required,
            no_impact_fallback,
        } = partition
            .load_posting_lists(
                tokens.as_ref(),
                params.as_ref(),
                Operator::And,
                scorer.as_ref(),
                &NoOpMetricsCollector,
                false,
            )
            .await
            .unwrap();
        assert!(!impact_safe);
        assert!(exact_scoring_required);
        assert!(no_impact_fallback);
        assert!(grouped_expansions.is_empty());

        let documents = partition.docs.modern().unwrap();
        let lengths = documents.lengths().await.unwrap();
        let visibility = documents.visibility(NoFilter.mask(), false).await.unwrap();
        results.push(
            partition
                .bm25_search_modern(
                    lengths.as_ref(),
                    &visibility,
                    params.as_ref(),
                    Operator::And,
                    postings,
                    Some(scorer.clone()),
                    &NoOpMetricsCollector,
                    shared_threshold.clone(),
                )
                .unwrap(),
        );
        published_floors.push(f32::from_bits(shared_threshold.load(Ordering::Relaxed)));
    }

    assert_eq!(results[0].len(), 1);
    assert!(published_floors[0] > 0.0);
    let winner_score = 2.0 * scorer.query_weight("lead") * scorer.doc_weight(63, 126);
    assert!(winner_score > published_floors[0]);
    assert_eq!(results[1].len(), 1);
    assert_eq!(results[1][0].document, DocId::new(0));
}

#[tokio::test]
async fn test_no_impact_bulk_and_uses_global_frequency_clamp_bound() {
    assert_no_impact_bulk_conjunction_preserves_winner(false).await;
}

#[tokio::test]
async fn test_no_impact_bulk_phrase_uses_global_frequency_clamp_bound() {
    assert_no_impact_bulk_conjunction_preserves_winner(true).await;
}

#[tokio::test]
async fn test_impact_partitions_share_global_threshold_without_pruning_winner() {
    // Partition 0 wins under its local corpus statistics but loses under
    // the global statistics. If its local score escapes into the shared
    // floor, partition 1 will incorrectly prune the real global winner.
    let (_tmpdir, _cache, index) = load_global_scoring_test_index(true, true).await;
    let first_partition = index
        .partitions
        .iter()
        .find(|partition| partition.id() == 0)
        .unwrap();
    let second_partition = index
        .partitions
        .iter()
        .find(|partition| partition.id() == 1)
        .unwrap();

    let tokens = Arc::new(Tokens::new(vec!["alpha".to_owned()], DocType::Text));
    let params = Arc::new(FtsSearchParams::new().with_limit(Some(1)));
    let scorer = Arc::new(
        index
            .bm25_base_scorer(tokens.as_ref(), params.as_ref(), None)
            .await
            .unwrap(),
    );
    first_partition
        .inverted_list
        .ensure_metadata_loaded()
        .await
        .unwrap();
    second_partition
        .inverted_list
        .ensure_metadata_loaded()
        .await
        .unwrap();
    let first_local_scorer = IndexBM25Scorer::new(std::iter::once(first_partition.as_ref()));
    let second_local_scorer = IndexBM25Scorer::new(std::iter::once(second_partition.as_ref()));
    let first_local_score =
        first_local_scorer.query_weight("alpha") * first_local_scorer.doc_weight(1, 5_000);
    let second_local_score =
        second_local_scorer.query_weight("alpha") * second_local_scorer.doc_weight(1, 1_000);
    assert!(first_local_score > second_local_score);
    let shared_threshold = Arc::new(AtomicU32::new(f32::NEG_INFINITY.to_bits()));

    // Search sequentially so partition 0 deterministically publishes its
    // score before partition 1 evaluates its impact upper bound.
    let first_candidates = search_test_impact_partition(
        first_partition,
        tokens.as_ref(),
        params.as_ref(),
        scorer.clone(),
        shared_threshold.clone(),
    )
    .await;
    assert_eq!(first_candidates.len(), 1);
    assert_eq!(first_candidates[0].document, DocId::new(0));
    let first_score =
        scorer.query_weight("alpha") * scorer.doc_weight(1, first_candidates[0].doc_length);
    let published_threshold = f32::from_bits(shared_threshold.load(Ordering::Relaxed));
    assert!(
        (published_threshold - first_score).abs() < 1e-6,
        "published threshold: {published_threshold}, expected global score: {first_score}"
    );

    let second_candidates = search_test_impact_partition(
        second_partition,
        tokens.as_ref(),
        params.as_ref(),
        scorer.clone(),
        shared_threshold.clone(),
    )
    .await;
    assert_eq!(second_candidates.len(), 1);
    assert_eq!(second_candidates[0].document, DocId::new(0));
    let second_score =
        scorer.query_weight("alpha") * scorer.doc_weight(1, second_candidates[0].doc_length);
    assert!(
        second_score > first_score,
        "second score: {second_score}, first score: {first_score}"
    );
    assert!((f32::from_bits(shared_threshold.load(Ordering::Relaxed)) - second_score).abs() < 1e-6);

    let (row_ids, scores) = index
        .bm25_search(
            tokens,
            params,
            Operator::Or,
            Arc::new(NoFilter),
            Arc::new(NoOpMetricsCollector),
            None,
        )
        .await
        .unwrap();
    assert_eq!(row_ids, vec![200]);
    assert_eq!(scores.len(), 1);
    assert!((scores[0] - second_score).abs() < 1e-6);
}

#[tokio::test]
async fn test_mixed_impact_and_legacy_partitions_use_global_final_scores() {
    let (_tmpdir, _cache, index) = load_global_scoring_test_index(true, false).await;

    let impact_partition = index
        .partitions
        .iter()
        .find(|partition| partition.id() == 0)
        .unwrap();
    let legacy_partition = index
        .partitions
        .iter()
        .find(|partition| partition.id() == 1)
        .unwrap();

    let impact_posting = impact_partition
        .inverted_list
        .posting_list(0, false, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(impact_posting.has_impacts());

    let legacy_posting = legacy_partition
        .inverted_list
        .posting_list(0, false, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(!legacy_posting.has_impacts());

    let tokens = Arc::new(Tokens::new(vec!["alpha".to_string()], DocType::Text));
    let params = Arc::new(FtsSearchParams::new().with_limit(Some(1)));
    let metrics = Arc::new(NoOpMetricsCollector);
    let (row_ids, scores) = index
        .bm25_search(
            tokens.clone(),
            params.clone(),
            Operator::Or,
            Arc::new(NoFilter),
            metrics.clone(),
            None,
        )
        .await
        .unwrap();

    assert_eq!(row_ids, vec![200]);
    assert_eq!(row_ids.len(), scores.len());

    let scorer = index
        .bm25_base_scorer(tokens.as_ref(), params.as_ref(), None)
        .await
        .unwrap();
    let expected_score = scorer.query_weight("alpha") * scorer.doc_weight(1, 1_000);
    assert!(
        (scores[0] - expected_score).abs() < 1e-6,
        "score: {}, expected: {}",
        scores[0],
        expected_score
    );
}

#[tokio::test]
async fn test_two_no_impact_partitions_share_global_scorer_and_threshold() {
    // Both no-impact partitions must score and prune in the same corpus-global
    // space before publishing a shared threshold.
    let (_tmpdir, _cache, index) = load_global_scoring_test_index(false, false).await;
    for partition in index.partitions.iter() {
        let posting = partition
            .inverted_list
            .posting_list(0, false, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert!(!posting.has_impacts());
    }

    let tokens = Arc::new(Tokens::new(vec!["alpha".to_string()], DocType::Text));
    let params = Arc::new(FtsSearchParams::new().with_limit(Some(1)));
    let metrics = Arc::new(NoOpMetricsCollector);
    let (row_ids, scores) = index
        .bm25_search(
            tokens.clone(),
            params.clone(),
            Operator::Or,
            Arc::new(NoFilter),
            metrics.clone(),
            None,
        )
        .await
        .unwrap();

    assert_eq!(row_ids, vec![200]);
    assert_eq!(scores.len(), 1);
    let scorer = index
        .bm25_base_scorer(tokens.as_ref(), params.as_ref(), None)
        .await
        .unwrap();
    let expected_score = scorer.query_weight("alpha") * scorer.doc_weight(1, 1_000);
    assert!(
        (scores[0] - expected_score).abs() < 1e-6,
        "score: {}, expected global score: {}",
        scores[0],
        expected_score
    );
}

#[tokio::test]
async fn test_and_query_returns_empty_when_exact_term_missing() {
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());
    builder.tokens.add("alpha".to_owned());
    builder.posting_lists.push(PostingListBuilder::new(false));
    builder.posting_lists[0].add(0, PositionRecorder::Count(1));
    builder.docs.append(100, 1);
    builder.write(store.as_ref()).await.unwrap();

    write_test_metadata(&store, vec![0], InvertedIndexParams::default()).await;
    let cache = Arc::new(LanceCache::with_capacity(4096));
    let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
        .await
        .unwrap();

    let tokens = Arc::new(Tokens::new(
        vec!["alpha".to_owned(), "missing".to_owned()],
        DocType::Text,
    ));
    let params = Arc::new(FtsSearchParams::new().with_limit(Some(10)));
    let prefilter = Arc::new(NoFilter);
    let metrics = Arc::new(NoOpMetricsCollector);

    let (and_row_ids, _) = index
        .bm25_search(
            tokens.clone(),
            params.clone(),
            Operator::And,
            prefilter.clone(),
            metrics.clone(),
            None,
        )
        .await
        .unwrap();
    assert!(
        and_row_ids.is_empty(),
        "AND must not match when any required term is missing"
    );

    let (or_row_ids, _) = index
        .bm25_search(tokens, params, Operator::Or, prefilter, metrics, None)
        .await
        .unwrap();
    assert_eq!(
        or_row_ids,
        vec![100],
        "OR should still match the present term"
    );
}

#[tokio::test]
async fn test_and_query_accepts_same_position_alternatives() {
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());
    for token in ["getusername", "get", "user", "name"] {
        builder.tokens.add(token.to_owned());
        builder.posting_lists.push(PostingListBuilder::new(false));
    }
    // Doc 0 only has the split words. Doc 1 has both the complete
    // identifier and split words. A grouped AND query should accept either
    // `getusername` or `get` at position 0.
    builder.posting_lists[1].add(0, PositionRecorder::Count(1));
    builder.posting_lists[2].add(0, PositionRecorder::Count(1));
    builder.posting_lists[3].add(0, PositionRecorder::Count(1));
    builder.docs.append(100, 3);

    builder.posting_lists[0].add(1, PositionRecorder::Count(1));
    builder.posting_lists[1].add(1, PositionRecorder::Count(1));
    builder.posting_lists[2].add(1, PositionRecorder::Count(1));
    builder.posting_lists[3].add(1, PositionRecorder::Count(1));
    builder.docs.append(101, 4);
    builder.write(store.as_ref()).await.unwrap();

    write_test_metadata(&store, vec![0], InvertedIndexParams::code()).await;
    let index = InvertedIndex::load(store.clone(), None, &LanceCache::no_cache())
        .await
        .unwrap();

    let tokens = Arc::new(Tokens::with_positions(
        vec![
            "getusername".to_string(),
            "get".to_string(),
            "user".to_string(),
            "name".to_string(),
        ],
        vec![0, 0, 1, 2],
        DocType::Text,
    ));
    let params = Arc::new(FtsSearchParams::new().with_limit(Some(10)));
    let (mut row_ids, _) = index
        .bm25_search(
            tokens,
            params,
            Operator::And,
            Arc::new(NoFilter),
            Arc::new(NoOpMetricsCollector),
            None,
        )
        .await
        .unwrap();
    row_ids.sort_unstable();
    assert_eq!(row_ids, vec![100, 101]);
}

#[tokio::test]
async fn test_phrase_query_accepts_same_position_alternatives() {
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    let mut builder = InnerBuilder::new(0, true, TokenSetFormat::default());
    for token in ["getusername", "get", "user", "name"] {
        builder.tokens.add(token.to_owned());
        builder.posting_lists.push(PostingListBuilder::new(true));
    }
    // Doc 0 only has split words. Doc 1 has both the complete identifier
    // and split words at the same position. Doc 2 has the terms but not as
    // an exact phrase.
    builder.posting_lists[1].add(0, PositionRecorder::Position(vec![0].into()));
    builder.posting_lists[2].add(0, PositionRecorder::Position(vec![1].into()));
    builder.posting_lists[3].add(0, PositionRecorder::Position(vec![2].into()));
    builder.docs.append(100, 3);

    builder.posting_lists[0].add(1, PositionRecorder::Position(vec![0].into()));
    builder.posting_lists[1].add(1, PositionRecorder::Position(vec![0].into()));
    builder.posting_lists[2].add(1, PositionRecorder::Position(vec![1].into()));
    builder.posting_lists[3].add(1, PositionRecorder::Position(vec![2].into()));
    builder.docs.append(101, 3);

    builder.posting_lists[0].add(2, PositionRecorder::Position(vec![0].into()));
    builder.posting_lists[2].add(2, PositionRecorder::Position(vec![2].into()));
    builder.posting_lists[3].add(2, PositionRecorder::Position(vec![3].into()));
    builder.docs.append(102, 3);

    builder.write(store.as_ref()).await.unwrap();

    write_test_metadata(
        &store,
        vec![0],
        InvertedIndexParams::code().with_position(true),
    )
    .await;
    let index = InvertedIndex::load(store.clone(), None, &LanceCache::no_cache())
        .await
        .unwrap();

    let tokens = Arc::new(Tokens::with_positions(
        vec![
            "getusername".to_string(),
            "get".to_string(),
            "user".to_string(),
            "name".to_string(),
        ],
        vec![0, 0, 1, 2],
        DocType::Text,
    ));
    let params = Arc::new(
        FtsSearchParams::new()
            .with_limit(Some(10))
            .with_phrase_slop(Some(0)),
    );
    let (mut row_ids, _) = index
        .bm25_search(
            tokens,
            params,
            Operator::And,
            Arc::new(NoFilter),
            Arc::new(NoOpMetricsCollector),
            None,
        )
        .await
        .unwrap();
    row_ids.sort_unstable();
    assert_eq!(row_ids, vec![100, 101]);
}
