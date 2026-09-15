// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering::Relaxed;

use arrow_array::StringArray;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use lance_core::cache::LanceCache;
use lance_core::utils::tempfile::TempObjDir;
use lance_io::object_store::ObjectStore;
use lance_select::RowAddrTreeMap;
use rstest::rstest;

use super::*;
use crate::metrics::{LocalMetricsCollector, NoOpMetricsCollector};
use crate::scalar::lance_format::LanceIndexStore;

fn test_store() -> (TempObjDir, Arc<LanceIndexStore>) {
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    (tmpdir, store)
}

/// Stream of `(text, row id)` rows split into batches of `batch_rows`.
fn text_stream(rows: &[(Option<&str>, u64)], batch_rows: usize) -> SendableRecordBatchStream {
    let schema = Arc::new(Schema::new(vec![
        Field::new(VALUE_COLUMN_NAME, DataType::Utf8, true),
        Field::new(ROW_ID, DataType::UInt64, false),
    ]));
    let batches: Vec<datafusion::error::Result<RecordBatch>> = rows
        .chunks(batch_rows.max(1))
        .map(|chunk| {
            let texts = StringArray::from(chunk.iter().map(|(text, _)| *text).collect::<Vec<_>>());
            let row_ids = UInt64Array::from(chunk.iter().map(|(_, id)| *id).collect::<Vec<_>>());
            Ok(RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(texts) as ArrayRef, Arc::new(row_ids) as ArrayRef],
            )?)
        })
        .collect();
    Box::pin(RecordBatchStreamAdapter::new(
        schema,
        futures::stream::iter(batches),
    ))
}

fn rows_from<'a>(texts: &[&'a str]) -> Vec<(Option<&'a str>, u64)> {
    texts
        .iter()
        .enumerate()
        .map(|(i, text)| (Some(*text), i as u64))
        .collect()
}

async fn load(
    store: &Arc<LanceIndexStore>,
    details: &prost_types::Any,
    cache: &LanceCache,
) -> Arc<MinHashLshIndex> {
    MinHashLshIndex::load(store.clone(), details, None, cache)
        .await
        .unwrap()
}

/// Build an index and load it against `cache`; the caller keeps the cache
/// alive because the index only holds a weak reference to it.
async fn build_and_load(
    store: &Arc<LanceIndexStore>,
    builder: MinHashLshIndexBuilder,
    rows: &[(Option<&str>, u64)],
    batch_rows: usize,
    cache: &LanceCache,
) -> Arc<MinHashLshIndex> {
    let params = builder.params().clone();
    builder
        .train(text_stream(rows, batch_rows), store.as_ref())
        .await
        .unwrap();
    load(store, &params.details_any().unwrap(), cache).await
}

fn default_builder() -> MinHashLshIndexBuilder {
    MinHashLshIndexBuilder::try_new(MinHashLshIndexParams::default()).unwrap()
}

async fn search(index: &MinHashLshIndex, text: &str, limit: usize) -> Vec<MinHashHit> {
    index
        .search_text(text, limit, &RowAddrMask::all_rows(), &NoOpMetricsCollector)
        .await
        .unwrap()
}

async fn row_ids(index: &MinHashLshIndex, text: &str, limit: usize) -> Vec<u64> {
    ids(&search(index, text, limit).await)
}

fn ids(hits: &[MinHashHit]) -> Vec<u64> {
    hits.iter().map(|hit| hit.row_id).collect()
}

/// Tiny deterministic generator for synthetic corpora.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }
}

/// Exact Jaccard similarity of the 3-shingle sets of two whitespace
/// tokenized documents, independent of the index code under test.
fn shingle_jaccard(a: &str, b: &str) -> f64 {
    let shingles = |text: &str| -> HashSet<String> {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        tokens.windows(3).map(|w| w.join(" ")).collect()
    };
    let (a, b) = (shingles(a), shingles(b));
    a.intersection(&b).count() as f64 / a.union(&b).count() as f64
}

/// Base documents plus one near duplicate of each, with two replaced words.
fn near_duplicate_corpus(num_docs: usize, words_per_doc: usize) -> (Vec<String>, Vec<String>) {
    let mut rng = Lcg(7);
    let mut bases = Vec::with_capacity(num_docs);
    let mut duplicates = Vec::with_capacity(num_docs);
    for _ in 0..num_docs {
        let words: Vec<String> = (0..words_per_doc)
            .map(|_| format!("w{}", rng.below(500)))
            .collect();
        let mut edited = words.clone();
        for _ in 0..2 {
            let position = rng.below(words_per_doc);
            edited[position] = format!("x{}", rng.below(1000));
        }
        bases.push(words.join(" "));
        duplicates.push(edited.join(" "));
    }
    (bases, duplicates)
}

/// Base texts followed by their near duplicates, row id = position.
fn corpus_rows<'a>(bases: &'a [String], duplicates: &'a [String]) -> Vec<(Option<&'a str>, u64)> {
    bases
        .iter()
        .chain(duplicates)
        .enumerate()
        .map(|(i, text)| (Some(text.as_str()), i as u64))
        .collect()
}

#[test]
fn test_signature_estimates_jaccard() {
    let params = MinHashLshIndexParams::default();
    let mut generator = SignatureGenerator::try_new(&params).unwrap();
    let (bases, duplicates) = near_duplicate_corpus(2, 80);
    let mut a = vec![0 as SignatureValue; params.num_hashes as usize];
    let mut b = a.clone();
    assert!(generator.signature(&bases[0], &mut a));
    assert!(generator.signature(&bases[0], &mut b));
    assert_eq!(a, b, "same text must yield the same signature");
    assert!(generator.signature(&duplicates[0], &mut b));
    let exact = shingle_jaccard(&bases[0], &duplicates[0]);
    let estimate = estimate_jaccard(&a, &b) as f64;
    assert!((estimate - exact).abs() < 0.15, "{estimate} vs {exact}");
    assert!(generator.signature(&bases[1], &mut b));
    assert!(estimate_jaccard(&a, &b) < 0.2);

    // The minimum over many shingles is a small number: a signature that
    // stored its raw high bits made unrelated long texts look identical.
    let mut rng = Lcg(5);
    let mut text = |prefix: char| {
        (0..10_000)
            .map(|_| format!("{prefix}{}", rng.next() % 1_000_000))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let (long_a, long_b) = (text('a'), text('b'));
    assert!(generator.signature(&long_a, &mut a));
    assert!(generator.signature(&long_b, &mut b));
    let estimate = estimate_jaccard(&a, &b);
    assert!(estimate < 0.05, "disjoint texts estimated at {estimate}");
}

/// The vectors of the format specification
/// (`docs/src/format/index/scalar/minhash_lsh.md`); any change here is a
/// `SIGNATURE_VERSION` bump.
#[test]
fn test_signature_known_answer_vectors() {
    let params =
        MinHashLshIndexParams::from_json(r#"{"num_hashes": 4, "num_bands": 2, "shingle_size": 2}"#)
            .unwrap();
    let mut generator = SignatureGenerator::try_new(&params).unwrap();
    assert_eq!(generator.multipliers[0], 0xbdd7_3226_2feb_6e95);
    assert_eq!(generator.increments[0], 0x28ef_e333_b266_f103);
    assert_eq!(generator.compressors[0], 0x5705_b877_0b3d_7dd5);

    let mut signature = [0u16; 4];
    let mut keys = Vec::new();
    assert!(generator.signature("The quick brown fox", &mut signature));
    assert_eq!(signature, [0xb00f, 0x59d6, 0x511a, 0xcd2c]);
    generator.band_keys(&signature, &mut keys);
    assert_eq!(keys, [0x00f3_0f35_01d4_58ea, 0x01ae_c5df_28d0_4915]);

    // Fewer tokens than the shingle size: one shingle of all tokens.
    assert!(generator.signature("Fox", &mut signature));
    assert_eq!(signature, [0x2970, 0xd6db, 0x65f8, 0xadfc]);
    keys.clear();
    generator.band_keys(&signature, &mut keys);
    assert_eq!(keys, [0x003c_f6e8_0331_2eca, 0x017c_5504_fbc8_3f36]);

    assert!(!generator.signature("", &mut signature));
    assert!(!generator.signature("  \n", &mut signature));
}

#[test]
fn test_params_defaults_and_details_round_trip() {
    let params = MinHashLshIndexParams::default();
    assert!(params.tokenizer.lower_case);
    assert!(!params.tokenizer.stem && !params.tokenizer.remove_stop_words);

    // Partial tokenizer objects layer over the MinHash defaults, not the
    // full text search defaults.
    let params = MinHashLshIndexParams::from_json(
        r#"{"num_hashes": 64, "num_bands": 8, "tokenizer": {"base_tokenizer": "whitespace"}}"#,
    )
    .unwrap();
    assert_eq!(params.num_hashes, 64);
    assert_eq!(params.tokenizer.base_tokenizer, "whitespace");
    assert!(!params.tokenizer.stem);

    let details = params.to_details().unwrap();
    assert_eq!(details.signature_version, SIGNATURE_VERSION);
    assert_eq!(
        MinHashLshIndexParams::from_details(&details).unwrap(),
        params
    );
    assert_eq!(
        MinHashLshIndexParams::from_details_hex(&params.details_hex().unwrap()).unwrap(),
        params
    );
    let stale = pb::MinHashLshIndexDetails {
        signature_version: SIGNATURE_VERSION + 1,
        ..details
    };
    let err = MinHashLshIndexParams::from_details(&stale).unwrap_err();
    assert!(matches!(err, Error::NotSupported { .. }), "{err}");
    let json = MinHashLshIndexPlugin
        .details_as_json(&params.details_any().unwrap())
        .unwrap();
    assert_eq!(json["tokenizer"]["base_tokenizer"], "whitespace");
}

#[rstest]
#[case::not_divisible(r#"{"num_hashes": 100, "num_bands": 16}"#, "multiple of num_bands")]
#[case::too_many_hashes(r#"{"num_hashes": 8192, "num_bands": 16}"#, "num_hashes <= 4096")]
#[case::too_many_bands(r#"{"num_hashes": 512, "num_bands": 512}"#, "num_bands <= 256")]
#[case::zero_shingle(r#"{"shingle_size": 0}"#, "shingle_size > 0")]
#[case::custom_stop_words(
    r#"{"tokenizer": {"custom_stop_words": ["the"]}}"#,
    "custom_stop_words"
)]
#[case::jieba(r#"{"tokenizer": {"base_tokenizer": "jieba"}}"#, "outside Lance")]
#[case::unknown_key(r#"{"num_hash": 128}"#, "unknown field")]
fn test_params_validation(#[case] json: &str, #[case] message: &str) {
    let err = MinHashLshIndexParams::from_json(json).unwrap_err();
    assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
    assert!(err.to_string().contains(message), "{err}");
}

#[test]
fn test_plugin_rejects_non_string_fields() {
    let plugin = MinHashLshIndexPlugin;
    let Err(err) = plugin.new_training_request("{}", &Field::new("id", DataType::Int64, false))
    else {
        panic!("non-string field must be rejected");
    };
    assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
    assert!(err.to_string().contains("string column"), "{err}");
    assert!(
        plugin
            .new_training_request("", &Field::new("text", DataType::LargeUtf8, true))
            .is_ok()
    );
}

#[tokio::test]
async fn test_build_load_search_roundtrip() {
    let (_tmpdir, store) = test_store();
    let texts = [
        "the quick brown fox jumps over the lazy dog near the river bank",
        "an entirely different sentence about databases and columnar storage formats",
        "the quick brown fox jumps over the lazy dog near the river bank today",
        "short",
        "the quick brown fox jumps over the lazy dog near the river bank",
    ];
    let rows: Vec<(Option<&str>, u64)> = texts
        .iter()
        .enumerate()
        .map(|(i, text)| (Some(*text), (i as u64 % 2) << 32 | i as u64))
        .collect();
    let index = build_and_load(&store, default_builder(), &rows, 2, &LanceCache::no_cache()).await;
    assert_eq!(index.num_docs(), 5);

    let hits = search(&index, texts[0], 10).await;
    assert_eq!(
        hits[0],
        MinHashHit {
            row_id: rows[0].1,
            distance: 0.0
        }
    );
    assert_eq!(
        hits[1],
        MinHashHit {
            row_id: rows[4].1,
            distance: 0.0
        }
    );
    assert_eq!(hits[2].row_id, rows[2].1);
    assert!(hits[2].distance > 0.0 && hits[2].distance < 0.4, "{hits:?}");
    assert!(
        hits.iter()
            .all(|hit| hit.row_id != rows[1].1 && hit.row_id != rows[3].1),
        "unrelated rows must not be candidates: {hits:?}"
    );
    assert_eq!(search(&index, texts[0], 1).await.len(), 1);
    assert!(search(&index, texts[0], 0).await.is_empty());
    assert!(
        search(&index, "nothing in common with the corpus at all", 10)
            .await
            .is_empty()
    );
    // Documents shorter than the shingle size are indexed as one shingle.
    assert_eq!(row_ids(&index, "short", 10).await, vec![rows[3].1]);

    let err = index
        .search(
            &crate::scalar::SargableQuery::IsNull(),
            &NoOpMetricsCollector,
        )
        .await
        .unwrap_err();
    assert!(matches!(err, Error::NotSupported { .. }), "{err}");
}

#[tokio::test]
async fn test_recall_on_near_duplicates() {
    let (_tmpdir, store) = test_store();
    let (bases, duplicates) = near_duplicate_corpus(60, 80);
    let rows = corpus_rows(&bases, &duplicates);
    let index = build_and_load(
        &store,
        default_builder(),
        &rows,
        17,
        &LanceCache::no_cache(),
    )
    .await;
    assert_eq!(index.num_docs(), 120);

    let (mut pairs, mut found) = (0, 0);
    for (i, duplicate) in duplicates.iter().enumerate() {
        if shingle_jaccard(&bases[i], duplicate) < 0.8 {
            continue;
        }
        pairs += 1;
        let hits = row_ids(&index, duplicate, 2).await;
        assert_eq!(hits[0], (bases.len() + i) as u64, "{hits:?}");
        found += hits.contains(&(i as u64)) as usize;
        assert!(
            hits.iter()
                .all(|&hit| hit == i as u64 || hit == (bases.len() + i) as u64),
            "unrelated row surfaced: {hits:?}"
        );
    }
    assert!(pairs >= 50, "{pairs} pairs above 0.8");
    let recall = found as f64 / pairs as f64;
    assert!(recall >= 0.9, "recall {recall} ({found}/{pairs})");
}

#[tokio::test]
async fn test_rows_without_tokens_are_not_indexed() {
    let (_tmpdir, store) = test_store();
    let rows = vec![
        (Some("alpha beta gamma delta"), 10u64),
        (None, 11),
        (Some(""), 12),
        (Some("   \n\t "), 13),
        (Some("alpha beta gamma delta"), 14),
    ];
    let index = build_and_load(&store, default_builder(), &rows, 2, &LanceCache::no_cache()).await;
    assert_eq!(index.num_docs(), 2);
    assert_eq!(
        row_ids(&index, "alpha beta gamma delta", 10).await,
        vec![10, 14]
    );
    assert!(search(&index, "   ", 10).await.is_empty());

    for rows in [vec![], vec![(None, 0), (Some(""), 1)]] {
        let (_tmpdir, store) = test_store();
        let empty =
            build_and_load(&store, default_builder(), &rows, 4, &LanceCache::no_cache()).await;
        assert_eq!(empty.num_docs(), 0);
        assert!(search(&empty, "anything at all here", 5).await.is_empty());
    }
}

#[rstest]
#[case::with_other_texts(5)]
#[case::whole_file(0)]
#[tokio::test]
async fn test_buckets_spanning_pages_and_dense_candidates(#[case] num_others: usize) {
    // Forty identical texts share every bucket: with four rows per page each
    // bucket spans ten pages, and the candidates cover most of the segment,
    // which is refined with a sequential scan. Without other texts every
    // bucket also runs up to the next band or the end of the file.
    let (bases, _) = near_duplicate_corpus(1 + num_others, 40);
    let mut texts = vec![bases[0].as_str(); 40];
    texts.extend(bases[1..].iter().map(String::as_str));
    let (_tmpdir, store) = test_store();
    let index = build_and_load(
        &store,
        default_builder().with_page_rows(4).unwrap(),
        &rows_from(&texts),
        64,
        &LanceCache::no_cache(),
    )
    .await;
    assert!(index.statistics().unwrap()["num_pages"].as_u64().unwrap() > 100);

    let hits = search(&index, &bases[0], 5).await;
    assert!(hits.iter().all(|hit| hit.distance == 0.0), "{hits:?}");
    assert_eq!(ids(&hits), vec![0, 1, 2, 3, 4], "ties are broken by row id");
    assert_eq!(
        row_ids(&index, &bases[0], 100).await,
        (0..40).collect::<Vec<u64>>()
    );
    let mask = RowAddrMask::from_block(RowAddrTreeMap::from_iter(0..20u64));
    let hits = index
        .search_text(&bases[0], 100, &mask, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert_eq!(ids(&hits), (20..40).collect::<Vec<u64>>());
}

#[tokio::test]
async fn test_spilled_multi_page_build_matches_single_run() {
    let (bases, duplicates) = near_duplicate_corpus(20, 40);
    let rows = corpus_rows(&bases, &duplicates);
    let (_single_dir, single_store) = test_store();
    let single = build_and_load(
        &single_store,
        default_builder(),
        &rows,
        64,
        &LanceCache::no_cache(),
    )
    .await;
    assert_eq!(single.statistics().unwrap()["num_pages"], 1);

    let (_split_dir, split_store) = test_store();
    let split = build_and_load(
        &split_store,
        default_builder()
            .with_page_rows(7)
            .unwrap()
            .with_sort_run_records(50)
            .unwrap(),
        &rows,
        3,
        &LanceCache::no_cache(),
    )
    .await;
    assert!(split.statistics().unwrap()["num_pages"].as_u64().unwrap() > 10);
    let files = split_store.list_files_with_sizes().await.unwrap();
    assert_eq!(files.len(), 2, "spill files must be deleted: {files:?}");
    for (text, _) in rows.iter().step_by(5) {
        let text = text.unwrap();
        assert_eq!(
            search(&single, text, 5).await,
            search(&split, text, 5).await
        );
    }
}

#[tokio::test]
async fn test_spill_limit_fails_before_writing() {
    let (_tmpdir, store) = test_store();
    let texts: Vec<String> = (0..2_000)
        .map(|i| format!("doc {i} words {}", i % 7))
        .collect();
    let rows = rows_from(&texts.iter().map(String::as_str).collect::<Vec<_>>());
    let err = default_builder()
        .with_sort_run_records(50)
        .unwrap()
        .with_spill_limit_bytes(100)
        .train(text_stream(&rows, 64), store.as_ref())
        .await
        .unwrap_err();
    assert!(matches!(err, Error::IO { .. }), "{err}");
    assert!(err.to_string().contains(SPILL_LIMIT_ENV), "{err}");
}

#[tokio::test]
async fn test_cache_prewarm_and_mask() {
    let (_tmpdir, store) = test_store();
    let (bases, duplicates) = near_duplicate_corpus(10, 120);
    let rows = corpus_rows(&bases, &duplicates);
    let cache = LanceCache::with_capacity(64 * 1024 * 1024);
    let index = build_and_load(
        &store,
        default_builder().with_page_rows(16).unwrap(),
        &rows,
        8,
        &cache,
    )
    .await;
    let all_rows = RowAddrMask::all_rows();

    // A cold search reads band pages; a repeated one finds them cached.
    let cold = LocalMetricsCollector::default();
    let hits = index
        .search_text(&duplicates[5], 2, &all_rows, &cold)
        .await
        .unwrap();
    assert_eq!(ids(&hits), vec![15, 5]);
    assert!(cold.index_cache_misses() > 0);
    let warm = LocalMetricsCollector::default();
    assert_eq!(
        index
            .search_text(&duplicates[5], 2, &all_rows, &warm)
            .await
            .unwrap(),
        hits
    );
    assert_eq!(warm.index_cache_misses(), 0);
    assert!(warm.index_cache_hits() > 0);

    // After a prewarm the signatures are resident too and a search reads
    // nothing.
    index.prewarm().await.unwrap();
    let resident = LocalMetricsCollector::default();
    assert_eq!(
        index
            .search_text(&duplicates[5], 2, &all_rows, &resident)
            .await
            .unwrap(),
        hits
    );
    assert_eq!(resident.parts_loaded.load(Relaxed), 0);
    assert_eq!(resident.index_cache_misses(), 0);

    let masked = |mask: RowAddrMask| {
        let (index, text) = (&index, &duplicates[5]);
        async move {
            let hits = index
                .search_text(text, 2, &mask, &NoOpMetricsCollector)
                .await
                .unwrap();
            ids(&hits)
        }
    };
    let blocked = masked(RowAddrMask::from_block(RowAddrTreeMap::from_iter([15u64]))).await;
    assert!(!blocked.contains(&15) && blocked[0] == 5, "{blocked:?}");
    let allowed = masked(RowAddrMask::from_allowed(RowAddrTreeMap::from_iter([5u64]))).await;
    assert_eq!(allowed, vec![5]);
    assert!(masked(RowAddrMask::allow_nothing()).await.is_empty());
}

#[tokio::test]
async fn test_load_rejects_mismatched_details_or_schema() {
    let (_tmpdir, store) = test_store();
    let rows = rows_from(&["some text to index for this test"]);
    let builder = default_builder();
    let params = builder.params().clone();
    builder
        .train(text_stream(&rows, 1), store.as_ref())
        .await
        .unwrap();
    let other = MinHashLshIndexParams::from_json(r#"{"shingle_size": 4}"#).unwrap();
    let err = MinHashLshIndex::load(
        store.clone(),
        &other.details_any().unwrap(),
        None,
        &LanceCache::no_cache(),
    )
    .await
    .unwrap_err();
    assert!(matches!(err, Error::CorruptFile { .. }), "{err}");
    assert!(
        err.to_string().contains("do not match index details"),
        "{err}"
    );

    // Replace the signature file with one whose signature is wider.
    let other_schema = signatures_schema(params.num_hashes as i32 + 1);
    let mut writer = store
        .new_index_file(SIGNATURES_FILENAME, other_schema.clone())
        .await
        .unwrap();
    writer
        .write_record_batch(RecordBatch::new_empty(other_schema))
        .await
        .unwrap();
    writer.finish().await.unwrap();
    let err = MinHashLshIndex::load(
        store.clone(),
        &params.details_any().unwrap(),
        None,
        &LanceCache::no_cache(),
    )
    .await
    .unwrap_err();
    assert!(matches!(err, Error::CorruptFile { .. }), "{err}");
    assert!(err.to_string().contains("expected schema"), "{err}");
}

#[tokio::test]
async fn test_update_keeps_filtered_rows_and_adds_new_rows() {
    let (_tmpdir, store) = test_store();
    let (bases, duplicates) = near_duplicate_corpus(6, 120);
    let rows = corpus_rows(&bases, &[]);
    let index = build_and_load(&store, default_builder(), &rows, 4, &LanceCache::no_cache()).await;

    // Drop base rows 0 and 1, append the near duplicates as rows 100..106
    let filter = OldIndexDataFilter::RowIds(RowAddrTreeMap::from_iter(2..6u64));
    let new_rows: Vec<(Option<&str>, u64)> = duplicates
        .iter()
        .enumerate()
        .map(|(i, text)| (Some(text.as_str()), 100 + i as u64))
        .collect();
    let (_dest_dir, dest_store) = test_store();
    let created = index
        .update(text_stream(&new_rows, 3), dest_store.as_ref(), Some(filter))
        .await
        .unwrap();
    let updated = load(&dest_store, &created.index_details, &LanceCache::no_cache()).await;
    assert_eq!(updated.num_docs(), 4 + 6);
    let hits = row_ids(&updated, &bases[0], 3).await;
    assert_eq!(hits[0], 100);
    assert!(!hits.contains(&0) && !hits.contains(&1), "{hits:?}");
    assert_eq!(row_ids(&updated, &duplicates[3], 2).await, vec![103, 3]);
}

#[tokio::test]
async fn test_remap_rewrites_and_drops_row_ids() {
    let (_tmpdir, store) = test_store();
    let (bases, _) = near_duplicate_corpus(4, 40);
    let rows = corpus_rows(&bases, &[]);
    let index = build_and_load(&store, default_builder(), &rows, 4, &LanceCache::no_cache()).await;
    assert!(index.can_remap());

    // Row 1 is deleted, row 2 is absent from the mapping and keeps its id
    let mapping = RowAddrRemap::direct(HashMap::from([
        (0u64, Some(7u64)),
        (1u64, None),
        (3u64, Some(9u64)),
    ]));
    let (_dest_dir, dest_store) = test_store();
    let created = index.remap(&mapping, dest_store.as_ref()).await.unwrap();
    let remapped = load(&dest_store, &created.index_details, &LanceCache::no_cache()).await;
    assert_eq!(remapped.num_docs(), 3);
    assert_eq!(row_ids(&remapped, &bases[0], 1).await, vec![7]);
    assert!(search(&remapped, &bases[1], 5).await.is_empty());
    assert_eq!(row_ids(&remapped, &bases[2], 1).await, vec![2]);
    assert_eq!(row_ids(&remapped, &bases[3], 1).await, vec![9]);
}

/// Row id remapper of a segment opened after a deferred compaction.
#[derive(Debug)]
struct TestRemapper(HashMap<u64, Option<u64>>);

impl RowIdRemapper for TestRemapper {
    fn remap_row_id(&self, row_id: u64) -> Option<u64> {
        self.0.get(&row_id).copied().unwrap_or(Some(row_id))
    }

    fn remap_row_addrs_tree_map(&self, _row_addrs: &RowAddrTreeMap) -> RowAddrTreeMap {
        unreachable!()
    }

    fn remap_row_ids_roaring_tree_map(
        &self,
        _row_ids: &roaring::RoaringTreemap,
    ) -> roaring::RoaringTreemap {
        unreachable!()
    }

    fn remap_row_ids_record_batch(
        &self,
        _batch: RecordBatch,
        _row_id_idx: usize,
    ) -> Result<RecordBatch> {
        unreachable!()
    }
}

#[tokio::test]
async fn test_rebuilds_remap_rows_of_a_deferred_compaction() {
    // The segment stores fragment 0 addresses; a deferred compaction moved
    // its rows to fragment 1 and deleted row 2. Filters are expressed in
    // the compacted address space, so a rebuild applying them to the
    // stored addresses dropped every old row.
    let (_tmpdir, store) = test_store();
    let (bases, duplicates) = near_duplicate_corpus(4, 40);
    let rows = corpus_rows(&bases, &[]);
    build_and_load(&store, default_builder(), &rows, 4, &LanceCache::no_cache()).await;
    let compacted = |row: u64| (1u64 << 32) | row;
    let remapper = TestRemapper(HashMap::from([
        (0u64, Some(compacted(0))),
        (1u64, Some(compacted(1))),
        (2u64, None),
        (3u64, Some(compacted(3))),
    ]));
    let index = MinHashLshIndex::load(
        store.clone(),
        &MinHashLshIndexParams::default().details_any().unwrap(),
        Some(Arc::new(remapper)),
        &LanceCache::no_cache(),
    )
    .await
    .unwrap();
    assert_eq!(row_ids(&index, &bases[0], 1).await, vec![compacted(0)]);

    // Keep the compacted fragment, add a duplicate of row 3
    let filter = OldIndexDataFilter::Fragments {
        to_keep: RoaringBitmap::from_iter([1u32]),
        to_remove: RoaringBitmap::from_iter([0u32]),
    };
    let new_rows = [(Some(duplicates[3].as_str()), 2u64 << 32)];
    let (_dir, dest_store) = test_store();
    let created = index
        .update(text_stream(&new_rows, 1), dest_store.as_ref(), Some(filter))
        .await
        .unwrap();
    let updated = load(&dest_store, &created.index_details, &LanceCache::no_cache()).await;
    assert_eq!(updated.num_docs(), 3 + 1);
    assert_eq!(row_ids(&updated, &bases[0], 1).await, vec![compacted(0)]);
    assert!(search(&updated, &bases[2], 5).await.is_empty());
    assert_eq!(
        row_ids(&updated, &duplicates[3], 2).await,
        vec![2u64 << 32, compacted(3)]
    );
}

#[tokio::test]
async fn test_merge_segments() {
    let (bases, duplicates) = near_duplicate_corpus(8, 120);
    let (_dir_a, store_a) = test_store();
    let rows_a = corpus_rows(&bases, &[]);
    let segment_a = build_and_load(
        &store_a,
        default_builder(),
        &rows_a,
        3,
        &LanceCache::no_cache(),
    )
    .await;
    let (_dir_b, store_b) = test_store();
    let rows_b: Vec<(Option<&str>, u64)> = duplicates
        .iter()
        .enumerate()
        .map(|(i, text)| (Some(text.as_str()), (1u64 << 32) | i as u64))
        .collect();
    let segment_b = build_and_load(
        &store_b,
        default_builder(),
        &rows_b,
        3,
        &LanceCache::no_cache(),
    )
    .await;

    // Segment b keeps only its first four rows
    let filter_b = OldIndexDataFilter::RowIds(RowAddrTreeMap::from_iter(
        (0..4u64).map(|i| (1u64 << 32) | i),
    ));
    let (_dir_merged, merged_store) = test_store();
    let created = merge_minhash_indices(
        &[(&segment_a, None), (&segment_b, Some(&filter_b))],
        merged_store.as_ref(),
    )
    .await
    .unwrap();
    let merged = load(
        &merged_store,
        &created.index_details,
        &LanceCache::no_cache(),
    )
    .await;
    assert_eq!(merged.num_docs(), 8 + 4);
    for (i, duplicate) in duplicates.iter().enumerate() {
        let hits = row_ids(&merged, duplicate, 2).await;
        if i < 4 {
            assert_eq!(hits, vec![(1u64 << 32) | i as u64, i as u64]);
        } else {
            assert_eq!(hits[0], i as u64, "{hits:?}");
            assert!(hits.iter().all(|hit| hit >> 32 == 0), "{hits:?}");
        }
    }

    // Segments built with different parameters cannot be merged
    let (_dir_c, store_c) = test_store();
    let other = MinHashLshIndexBuilder::try_new(
        MinHashLshIndexParams::from_json(r#"{"shingle_size": 4}"#).unwrap(),
    )
    .unwrap();
    let segment_c = build_and_load(&store_c, other, &rows_a, 3, &LanceCache::no_cache()).await;
    let Err(err) = merge_minhash_indices(
        &[(&segment_a, None), (&segment_c, None)],
        merged_store.as_ref(),
    )
    .await
    else {
        panic!("segments with different parameters must not merge");
    };
    assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
    assert!(err.to_string().contains("different parameters"), "{err}");
}
