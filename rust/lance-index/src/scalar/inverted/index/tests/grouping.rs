// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use lance_core::cache::{CacheCodecImpl, CacheEntryReader, CacheEntryWriter};

#[rstest::rstest]
#[case::default(None, None, Some(PostingGroupPolicy::Bytes(32768)))]
#[case::bytes(Some("8192"), None, Some(PostingGroupPolicy::Bytes(8192)))]
#[case::tokens(None, Some("128"), Some(PostingGroupPolicy::Tokens(128)))]
#[case::both(Some("32768"), Some("128"), None)]
#[case::zero_bytes(Some("0"), None, None)]
#[case::zero_tokens(None, Some("0"), None)]
#[case::invalid(Some("32KiB"), None, None)]
#[case::negative(None, Some("-1"), None)]
#[case::token_overflow(None, Some("4294967296"), None)]
fn test_posting_group_policy(
    #[case] bytes: Option<&str>,
    #[case] tokens: Option<&str>,
    #[case] expected: Option<PostingGroupPolicy>,
) {
    let actual = PostingGroupPolicy::parse(bytes, tokens);
    match expected {
        Some(expected) => assert_eq!(actual.unwrap(), expected),
        None => {
            let error = actual.unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains("LANCE_FTS_POSTING_GROUP_MAX_"));
        }
    }
}

fn sized_posting_batch(payloads: &[usize]) -> RecordBatch {
    let mut posting = ListBuilder::new(LargeBinaryBuilder::new());
    for (row, size) in payloads.iter().enumerate() {
        posting.values().append_value(vec![row as u8; *size]);
        posting.append(true);
    }
    let array = Arc::new(posting.finish());
    RecordBatch::try_from_iter([(POSTING_COL, array as ArrayRef)]).unwrap()
}

#[rstest::rstest]
#[case::single_rows(1)]
#[case::pairs(2)]
#[case::three(3)]
#[case::five(5)]
#[case::whole(6)]
fn test_byte_groups_cross_read_chunks(#[case] chunk_rows: usize) {
    // One binary offset (8), one list offset (4), and score/length (8)
    // accompany each payload. The 80-byte row is intentionally oversized.
    let batch = sized_posting_batch(&[0, 0, 24, 60, 0, 0]);
    assert_eq!(posting_row_bytes(&batch).unwrap(), [20, 20, 44, 80, 20, 20]);
    let mut planner = ByteGroupPlanner::new(64);
    let mut groups = Vec::new();
    for start in (0..6).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(6);
        groups.extend(
            planner
                .push(batch.slice(start, end - start), end == 6)
                .unwrap(),
        );
    }
    assert_eq!(
        groups
            .iter()
            .map(|(start, end, _)| (*start, *end))
            .collect::<Vec<_>>(),
        [(0, 2), (2, 3), (3, 4), (4, 6)]
    );
    for (start, end, group) in groups {
        assert_eq!(group, batch.slice(start as usize, (end - start) as usize));
    }
    assert!(planner.push(batch.slice(0, 0), true).unwrap().is_empty());
}

#[test]
fn test_byte_group_weights_include_impacts_and_exclude_positions() {
    let batch = sized_posting_batch(&[10, 20, 30]);
    let impacts = sized_posting_batch(&[2, 4, 6]).column(0).clone();
    let positions = Arc::new(LargeBinaryArray::from_iter_values([&[1u8; 1024][..]; 3]));
    let batch = batch
        .try_with_column(
            Field::new(IMPACT_COL, impacts.data_type().clone(), false),
            impacts,
        )
        .unwrap()
        .try_with_column(
            Field::new(COMPRESSED_POSITION_COL, DataType::LargeBinary, false),
            positions,
        )
        .unwrap();
    assert_eq!(posting_row_bytes(&batch).unwrap(), [44, 56, 68]);
    assert_eq!(posting_row_bytes(&batch.slice(1, 2)).unwrap(), [56, 68]);
}

#[rstest::rstest]
#[case::valid(vec![2,3,6], true)]
#[case::empty(vec![], false)]
#[case::zero(vec![0,6], false)]
#[case::descending(vec![2,1,6], false)]
#[case::duplicate(vec![2,2,6], false)]
fn test_byte_group_boundaries_codec(#[case] ends: Vec<u32>, #[case] valid: bool) {
    let mut buffer = Vec::new();
    let mut writer = CacheEntryWriter::new(&mut buffer);
    PostingGroupBoundaries { ends: ends.clone() }
        .serialize(&mut writer)
        .unwrap();
    let buffer = bytes::Bytes::from(buffer);
    let mut reader = CacheEntryReader::new(&buffer, 0, 1);
    let decoded = PostingGroupBoundaries::deserialize(&mut reader);
    if valid {
        assert_eq!(decoded.unwrap().ends, ends);
    } else {
        assert!(decoded.is_err());
    }
}

#[tokio::test]
async fn test_byte_group_directory_is_namespaced_by_policy_and_partition() {
    let cache = LanceCache::with_capacity(4096);
    let partition = cache.with_key_prefix("partition-0");
    let key = PostingGroupBoundariesKey {
        max_bytes: 32768,
        token_count: 6,
    };
    partition
        .insert_with_key(
            &key,
            Arc::new(PostingGroupBoundaries {
                ends: vec![2, 3, 6],
            }),
        )
        .await;
    assert!(partition.get_with_key(&key).await.is_some());
    assert!(
        cache
            .with_key_prefix("partition-1")
            .get_with_key(&key)
            .await
            .is_none()
    );
    assert!(
        partition
            .get_with_key(&PostingGroupBoundariesKey {
                max_bytes: 65536,
                token_count: 6
            })
            .await
            .is_none()
    );
    assert!(
        partition
            .get_with_key(&PostingGroupBoundariesKey {
                max_bytes: 32768,
                token_count: 7
            })
            .await
            .is_none()
    );
    let grouping = PostingGrouping::for_reader(false, PositionsLayout::None, 6, &partition)
        .await
        .unwrap();
    assert_eq!(grouping.range_for_token(2, 6), Some((2, 3)));
    assert_eq!(grouping.range_for_token(5, 6), Some((3, 6)));
    assert_eq!(grouping.range_for_token(6, 6), None);
}

#[cfg(target_pointer_width = "64")]
#[tokio::test]
async fn test_posting_group_token_count_overflow_is_rejected() {
    let error = PostingGrouping::for_reader(
        false,
        PositionsLayout::None,
        u32::MAX as usize + 1,
        &LanceCache::no_cache(),
    )
    .await
    .unwrap_err();
    assert!(matches!(error, Error::Index { .. }));
    assert!(error.to_string().contains("4294967296"));
}

fn posting_entries(posting: &PostingList) -> Vec<(u64, u32)> {
    posting.iter().map(|(doc, freq, _)| (doc, freq)).collect()
}

#[tokio::test]
async fn test_modern_posting_validation_is_cached_per_token() {
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());
    builder.tokens.get_or_add("term");
    let mut valid_builder = PostingListBuilder::new(false);
    valid_builder.add(0, PositionRecorder::Count(1));
    builder.posting_lists.push(valid_builder);
    builder.docs.append(1000, 1);
    builder.write(store.as_ref()).await.unwrap();

    let reader = store.open_index_file(&posting_file_path(0)).await.unwrap();
    let mut posting_reader = PostingListReader::try_new(reader, &LanceCache::no_cache())
        .await
        .unwrap();
    posting_reader.modern_num_docs = Some(1);
    let validation = &posting_reader
        .modern_doc_id_validations
        .as_ref()
        .expect("modern readers have per-token validation state")[0];
    assert!(!validation.load(Ordering::Acquire));
    assert!(!posting_reader.modern_posting_is_validated(0).unwrap());

    let mut corrupt_builder = PostingListBuilder::new(false);
    corrupt_builder.add(1, PositionRecorder::Count(1));
    let corrupt_batch = corrupt_builder.to_batch(vec![1.0]).unwrap();
    let corrupt_posting = PostingList::from_batch(&corrupt_batch, Some(1.0), Some(1)).unwrap();
    let error = posting_reader
        .ensure_modern_posting_validated(0, &corrupt_posting)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Index { .. }));
    assert!(error.to_string().contains("DocId 1"));
    assert!(error.to_string().contains("[0, 1)"));
    assert!(!validation.load(Ordering::Acquire));
    assert!(!posting_reader.modern_posting_is_validated(0).unwrap());

    let first = posting_reader
        .posting_list(0, false, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert_eq!(posting_entries(&first), vec![(0, 1)]);
    assert!(validation.load(Ordering::Acquire));
    assert!(posting_reader.modern_posting_is_validated(0).unwrap());

    let second = posting_reader
        .posting_list(0, false, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert_eq!(posting_entries(&second), vec![(0, 1)]);
    assert!(validation.load(Ordering::Acquire));
}

/// Runtime synthetic grouping must return correct posting lists for every
/// token, including across synthetic group boundaries.
#[tokio::test]
async fn test_posting_list_synthetic_grouping_reads_group_boundaries() {
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    let num_tokens = TEST_GROUP_TOKENS as u32 + 4;
    let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());
    for t in 0..num_tokens {
        builder.tokens.get_or_add(&format!("t{t}"));
        let mut pl = PostingListBuilder::new(false);
        pl.add(t, PositionRecorder::Count(1));
        builder.posting_lists.push(pl);
        builder.docs.append(1000 + t as u64, 1);
    }
    builder.write(store.as_ref()).await.unwrap();

    let reader = store.open_index_file(&posting_file_path(0)).await.unwrap();
    let cache = LanceCache::no_cache();
    let mut posting_reader = PostingListReader::try_new(reader, &cache).await.unwrap();
    posting_reader.modern_num_docs = Some(num_tokens as usize);
    posting_reader.grouping = PostingGrouping::SyntheticFixed { group_size: 128 };
    assert!(
        matches!(
            &posting_reader.grouping,
            PostingGrouping::SyntheticFixed { .. }
        ),
        "v2 reader must synthesize runtime posting groups",
    );

    let metrics = NoOpMetricsCollector;
    for token in 0..num_tokens {
        let posting = posting_reader
            .posting_list(token, false, &metrics)
            .await
            .unwrap();
        assert_eq!(
            posting_entries(&posting),
            vec![(token as u64, 1)],
            "synthetic grouping mismatch for token {token}",
        );
        assert_eq!(posting.len(), 1, "length mismatch for token {token}");
    }
}

/// Prewarm must populate exactly the `PostingListGroupKey`s the read path
/// looks up — in particular the final group, whose `end` both paths derive
/// from `self.len()`. If those derivations drifted (e.g. one used
/// `num_rows()` and the other the loaded posting count), the last group's
/// warm entry would be missing and prewarm silently wasted (issue #7040).
#[tokio::test]
async fn test_prewarm_group_keys_match_read_path() {
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    let num_tokens = TEST_GROUP_TOKENS as u32 + 4;
    let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());
    for t in 0..num_tokens {
        builder.tokens.get_or_add(&format!("t{t}"));
        let mut pl = PostingListBuilder::new(false);
        pl.add(t, PositionRecorder::Count(1));
        builder.posting_lists.push(pl);
        builder.docs.append(1000 + t as u64, 1);
    }
    builder.write(store.as_ref()).await.unwrap();

    let reader = store.open_index_file(&posting_file_path(0)).await.unwrap();
    // A real (strong) cache must outlive the reader's weak handle so the
    // prewarmed entries are still resolvable below.
    let cache = LanceCache::with_capacity(1 << 20);
    let mut posting_reader = PostingListReader::try_new(reader, &cache).await.unwrap();
    posting_reader.modern_num_docs = Some(num_tokens as usize);
    assert!(
        matches!(&posting_reader.grouping, PostingGrouping::Bytes { .. }),
        "modern reader should use byte-bounded groups",
    );

    posting_reader
        .prewarm_posting_lists(false, 2)
        .await
        .unwrap();
    assert!(posting_reader.modern_posting_validation_ready());
    assert!(
        posting_reader
            .modern_postings_validated
            .load(Ordering::Acquire)
    );

    for token in 0..num_tokens {
        let (start, end) = posting_reader.group_range_for_token(token).unwrap();
        assert!(
            posting_reader
                .index_cache
                .get_with_key(&posting_list_group_cache_key(
                    start,
                    end,
                    posting_reader.has_impacts,
                ))
                .await
                .is_some(),
            "prewarm did not populate group [{start}, {end}) that the read \
                 path requests for token {token}",
        );
    }

    let (_, last_end) = posting_reader
        .group_range_for_token(num_tokens - 1)
        .unwrap();
    assert_eq!(
        last_end, num_tokens,
        "the last group must end at the posting count ({num_tokens})",
    );
}

/// An empty partition has no synthetic groups because there are no token
/// rows to cache.
#[tokio::test]
async fn test_empty_partition_has_no_synthetic_groups() {
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());
    builder.write(store.as_ref()).await.unwrap();

    let reader = store.open_index_file(&posting_file_path(0)).await.unwrap();
    let posting_reader = PostingListReader::try_new(reader, &LanceCache::no_cache())
        .await
        .unwrap();
    assert!(
        matches!(&posting_reader.grouping, PostingGrouping::None),
        "reader for an empty partition must not create cache groups",
    );
    assert!(posting_reader.is_empty());
}

/// A large posting list can share a runtime synthetic group with neighbors;
/// grouping is token-count based and should still read every member intact.
#[tokio::test]
async fn test_large_posting_reads_inside_synthetic_group() {
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());
    let big_docs = (BLOCK_SIZE * 3 + 5) as u32;
    builder.tokens.get_or_add("big");
    let mut big = PostingListBuilder::new(false);
    for d in 0..big_docs {
        big.add(d, PositionRecorder::Count(1));
    }
    builder.posting_lists.push(big);
    for t in 1..5u32 {
        builder.tokens.get_or_add(&format!("t{t}"));
        let mut pl = PostingListBuilder::new(false);
        pl.add(0, PositionRecorder::Count(1));
        builder.posting_lists.push(pl);
    }
    for d in 0..big_docs as u64 {
        builder.docs.append(1000 + d, 1);
    }
    builder.write(store.as_ref()).await.unwrap();

    let reader = store.open_index_file(&posting_file_path(0)).await.unwrap();
    let mut posting_reader = PostingListReader::try_new(reader, &LanceCache::no_cache())
        .await
        .unwrap();
    posting_reader.grouping = PostingGrouping::SyntheticFixed { group_size: 128 };
    let expected_end = TEST_GROUP_TOKENS.min(5) as u32;

    assert_eq!(
        posting_reader.group_range_for_token(0),
        Some((0, expected_end)),
        "runtime synthetic grouping should group by token count, not posting bytes",
    );
    let big = posting_reader
        .posting_list(0, false, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert_eq!(big.len(), big_docs as usize);
    // A trailing tiny term (in the next, multi-token group) still reads back.
    let tiny = posting_reader
        .posting_list(2, false, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert_eq!(tiny.len(), 1);
}

/// Non-empty v2 indexes should prewarm synthetic `PostingListGroupKey`
/// entries, matching what the read path then looks up without persisted
/// grouping metadata.
#[tokio::test]
async fn test_prewarm_synthetic_grouping_populates_group_entries() {
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    let num_tokens = 3u32;
    let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());
    for t in 0..num_tokens {
        builder.tokens.get_or_add(&format!("t{t}"));
        let mut pl = PostingListBuilder::new(false);
        pl.add(t, PositionRecorder::Count(1));
        builder.posting_lists.push(pl);
        builder.docs.append(1000 + t as u64, 1);
    }
    builder.write(store.as_ref()).await.unwrap();

    let reader = store.open_index_file(&posting_file_path(0)).await.unwrap();
    let cache = LanceCache::with_capacity(1 << 20);
    let mut posting_reader = PostingListReader::try_new(reader, &cache).await.unwrap();
    posting_reader.grouping = PostingGrouping::SyntheticFixed { group_size: 128 };

    posting_reader
        .prewarm_posting_lists(false, 2)
        .await
        .unwrap();

    for token_id in 0..num_tokens {
        let (start, end) = posting_reader.group_range_for_token(token_id).unwrap();
        let group = posting_reader
            .index_cache
            .get_with_key(&posting_list_group_cache_key(
                start,
                end,
                posting_reader.has_impacts,
            ))
            .await
            .unwrap_or_else(|| {
                panic!(
                    "synthetic prewarm should populate group [{start}, {end}) for token {token_id}"
                )
            });
        assert!(
            group.is_packed(),
            "no-position synthetic prewarm should insert a packed group"
        );
        assert!(
            posting_reader
                .index_cache
                .get_with_key(&posting_list_cache_key(
                    token_id,
                    posting_reader.has_impacts,
                ))
                .await
                .is_none(),
            "synthetic prewarm should not populate per-token entry {token_id}",
        );
    }
}

/// End-to-end BM25 search over a grouped multi-group index must return the
/// correct documents, and a warm-cache query must match the cold-cache
/// result exactly (issue #7040).
#[tokio::test]
async fn test_grouped_bm25_search_correct_and_cache_stable() {
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    // Rare tokens (one doc each) plus one common token in every doc. The
    // token count exceeds the runtime group size so scoring must index
    // into the right synthetic group slot.
    let num_rare = TEST_GROUP_TOKENS as u32 + 2;
    let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());
    for t in 0..num_rare {
        builder.tokens.get_or_add(&format!("t{t}"));
        builder.posting_lists.push(PostingListBuilder::new(false));
    }
    let common_id = builder.tokens.get_or_add("common");
    builder.posting_lists.push(PostingListBuilder::new(false));
    for d in 0..num_rare {
        builder.posting_lists[d as usize].add(d, PositionRecorder::Count(1));
        builder.posting_lists[common_id as usize].add(d, PositionRecorder::Count(1));
        builder.docs.append(1000 + d as u64, 2);
    }
    builder.write(store.as_ref()).await.unwrap();

    let metadata = HashMap::from([
        (
            "partitions".to_owned(),
            serde_json::to_string(&vec![0u64]).unwrap(),
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

    let cache = Arc::new(LanceCache::with_capacity(1 << 20));
    let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
        .await
        .unwrap();

    // A rare token in the middle of a group must resolve to its one doc.
    let query = |term: &str| {
        let index = index.clone();
        let term = term.to_string();
        async move {
            index
                .bm25_search(
                    Arc::new(Tokens::new(vec![term], DocType::Text)),
                    Arc::new(FtsSearchParams::new().with_limit(Some(num_rare as usize))),
                    Operator::Or,
                    Arc::new(NoFilter),
                    Arc::new(NoOpMetricsCollector),
                    None,
                )
                .await
                .unwrap()
        }
    };

    let rare_query_id = num_rare / 2;
    let (rare_rows, _) = query(&format!("t{rare_query_id}")).await;
    assert_eq!(
        rare_rows,
        vec![1000 + rare_query_id as u64],
        "rare token must map to its single doc",
    );

    // Cold vs warm cache must agree for the common (large) token.
    let (cold_rows, cold_scores) = query("common").await;
    let (warm_rows, warm_scores) = query("common").await;
    assert_eq!(cold_rows.len(), num_rare as usize);
    assert_eq!(cold_rows, warm_rows, "warm-cache rows must match cold");
    assert_eq!(
        cold_scores, warm_scores,
        "warm-cache scores must match cold"
    );
}

#[tokio::test]
async fn flat_bm25_search_stop_word_query_over_unindexed_rows_returns_empty() {
    let schema = Arc::new(Schema::new(vec![
        ROW_ID_FIELD.clone(),
        Field::new("text", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![0u64, 1, 2])),
            Arc::new(StringArray::from(vec![
                "the quick brown fox",
                "a lazy dog",
                "for the win",
            ])),
        ],
    )
    .unwrap();

    let input: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema.clone(),
        stream::iter(vec![Ok(batch)]),
    ));

    // Analyzer with an English stop-word filter, so the query "the"
    // tokenizes to zero terms -- exactly the production trigger.
    let tokenizer: Box<dyn LanceTokenizer> = Box::new(TextTokenizer::new(
        TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(StopWordFilter::new(Language::English).unwrap())
            .build(),
    ));

    let result_stream = flat_bm25_search_stream_with_metrics(
        input,
        "text".to_string(),
        "the".to_string(),
        tokenizer,
        None,
        100,
        None,
    )
    .await
    .unwrap();

    let batches: Vec<_> = result_stream.try_collect().await.unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total_rows, 0,
        "a stop-word-only query has no searchable terms and must match nothing"
    );
}
