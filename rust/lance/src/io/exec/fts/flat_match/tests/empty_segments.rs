// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use arrow_array::{UInt64Array, record_batch};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::physical_plan::collect;
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
