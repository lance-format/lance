// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::dataset::index::LanceIndexStoreExt;
use arrow_array::{RecordBatch, RecordBatchIterator, StringArray, UInt64Array, record_batch};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::physical_plan::collect;
use lance_index::scalar::IndexStore;
use lance_index::scalar::inverted::METADATA_FILE;
use lance_index::scalar::lance_format::LanceIndexStore;
use rstest::rstest;

#[rstest]
#[case::small_index(1)]
#[case::large_index(20)]
#[case::no_index(0)]
#[tokio::test]
async fn explicit_empty_segments_score_only_input_with_index_tokenizer(#[case] indexed_rows: u32) {
    let mut dataset = lance_datagen::gen_batch()
        .col(
            "text",
            lance_datagen::array::cycle_utf8_literals(&["hello hello hello"]),
        )
        .into_ram_dataset(
            FragmentCount::from(1),
            FragmentRowCount::from(indexed_rows.max(1)),
        )
        .await
        .unwrap();
    let params = InvertedIndexParams::default()
        .lower_case(true)
        .stem(false)
        .remove_stop_words(false);
    if indexed_rows > 0 {
        dataset
            .create_index(&["text"], IndexType::Inverted, None, &params, true)
            .await
            .unwrap();
    }
    let corpus = record_batch!(
        (
            "text",
            Utf8,
            ["hello orange", "hello purple", "HELLO silver"]
        ),
        ("_rowid", UInt64, [10, 11, 12])
    )
    .unwrap();
    let input =
        MemorySourceConfig::try_new_exec(&[vec![corpus.clone()]], corpus.schema(), None).unwrap();
    let query = MatchQuery::new("hello".into())
        .with_column(Some("text".into()))
        .with_document_granularity(DocumentGranularity::Row);
    let exec = Arc::new(
        FlatMatchQueryExec::new_with_segments(
            Arc::new(dataset),
            query,
            FtsSearchParams::default(),
            input,
            vec![],
        )
        .unwrap(),
    );
    // N=3 and every document has length=avgdl=2. Indexed rows must affect
    // neither N nor DF. Only the configured index tokenizer folds case.
    for _ in 0..2 {
        let batches = collect(exec.clone(), Default::default()).await.unwrap();
        let batch = arrow::compute::concat_batches(&exec.schema(), &batches).unwrap();
        let mut ids = batch["_rowid"]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .values()
            .to_vec();
        ids.sort_unstable();
        let (expected_ids, expected_score) = if indexed_rows > 0 {
            (vec![10, 11, 12], (8_f32 / 7.0).ln())
        } else {
            // The unindexed SimpleTokenizer has no lowercase filter: DF=2.
            (vec![10, 11], 1.6_f32.ln())
        };
        assert_eq!(ids, expected_ids);
        let scores = batch["_score"].as_primitive::<arrow_array::types::Float32Type>();
        for score in scores.values() {
            assert!(
                (score - expected_score).abs() < 1e-6,
                "unexpected score {score}"
            );
        }
    }
}

#[rstest]
#[case::custom_list_retains_language_stop_word(
    InvertedIndexParams::default().stem(false).custom_stop_words(Some(vec!["lance".into()])),
    vec!["the lance data", "the orange", "lance purple"],
    "the", vec![10, 11], Some(1.6_f32.ln() * 2.2 / 2.38),
)]
#[case::custom_list_removes_custom_word(
    InvertedIndexParams::default().stem(false).custom_stop_words(Some(vec!["lance".into()])),
    vec!["the lance data", "the orange", "lance purple"],
    "lance", vec![], None,
)]
#[case::json_document_tokenizer(
    InvertedIndexParams::default().stem(false).remove_stop_words(false).lance_tokenizer("json".into()),
    vec![r#"{"body":"apple orange"}"#, r#"{"other":"apple"}"#, r#"{"body":"pear"}"#],
    "body,str,apple", vec![10], Some((8_f32 / 3.0).ln() * 2.2 / 2.65),
)]
#[tokio::test]
async fn explicit_empty_segments_retain_complete_persisted_tokenizer(
    #[case] params: InvertedIndexParams,
    #[case] documents: Vec<&str>,
    #[case] terms: &str,
    #[case] expected_ids: Vec<u64>,
    #[case] expected_score: Option<f32>,
) {
    let text = Arc::new(StringArray::from(documents)) as arrow_array::ArrayRef;
    let batch = RecordBatch::try_from_iter([("text", text.clone())]).unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        "memory://",
        None,
    )
    .await
    .unwrap();
    dataset
        .create_index(&["text"], IndexType::Inverted, None, &params, true)
        .await
        .unwrap();
    let segments = load_segments(&dataset, "text", DocumentGranularity::Row)
        .await
        .unwrap()
        .unwrap();
    let store = LanceIndexStore::from_dataset_for_existing(&dataset, &segments[0])
        .await
        .unwrap();
    let files = store.list_files_with_sizes().await.unwrap();
    assert!(files.iter().any(|file| file.path == METADATA_FILE));
    let mut removed = 0;
    for file in files {
        if file.path != METADATA_FILE {
            store.delete_index_file(&file.path).await.unwrap();
            removed += 1;
        }
    }
    assert!(
        removed > 0,
        "fixture must have unavailable index partitions"
    );
    dataset.session().index_cache.clear().await;
    // The manifest's index listing also lives in index_cache. Bind that
    // metadata before the baseline; no index partition data remains to open.
    dataset.load_indices().await.unwrap();
    let cache_entries = dataset.index_cache_entry_count().await;
    let dataset = Arc::new(dataset);
    let corpus = RecordBatch::try_from_iter([
        ("text", text),
        (
            "_rowid",
            Arc::new(UInt64Array::from(vec![10, 11, 12])) as arrow_array::ArrayRef,
        ),
    ])
    .unwrap();
    let input =
        MemorySourceConfig::try_new_exec(&[vec![corpus.clone()]], corpus.schema(), None).unwrap();
    let query = MatchQuery::new(terms.into())
        .with_column(Some("text".into()))
        .with_document_granularity(DocumentGranularity::Row);
    let exec = Arc::new(
        FlatMatchQueryExec::new_with_segments(
            dataset.clone(),
            query,
            FtsSearchParams::default(),
            input,
            vec![],
        )
        .unwrap(),
    );
    for _ in 0..2 {
        let batches = collect(exec.clone(), Default::default()).await.unwrap();
        let result = arrow::compute::concat_batches(&exec.schema(), &batches).unwrap();
        let mut actual_ids = result["_rowid"]
            .as_primitive::<arrow_array::types::UInt64Type>()
            .values()
            .to_vec();
        actual_ids.sort_unstable();
        assert_eq!(actual_ids, expected_ids);
        if let Some(expected) = expected_score {
            for actual in result["_score"]
                .as_primitive::<arrow_array::types::Float32Type>()
                .values()
            {
                assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
            }
        }
        assert_eq!(
            dataset.index_cache_entry_count().await,
            cache_entries,
            "tokenizer metadata must not open index partitions"
        );
    }
}
