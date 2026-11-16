// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::support::*;
use super::*;
use crate::dataset::index::LanceIndexStoreExt;
use crate::dataset::transaction::RewriteGroup;
use crate::index::frag_reuse::{
    MissingCoverageReason, frag_reuse_entry_appending, segment_coverage_reason,
};
use crate::index::frag_reuse_reader::tests::{persist_fixture, prepare_partition};
use lance_index::scalar::inverted::METADATA_FILE;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::{IndexStore, index_files_to_table};

#[tokio::test]
async fn test_segmented_optimize_rebalances_only_one_segment() {
    const DIMENSION: i32 = 8;

    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();
    let mut dataset = write_skewed_fragmented_vector_dataset(test_uri, DIMENSION).await;
    create_segmented_vector_index(&mut dataset, "vector_idx", "vector", DIMENSION).await;

    let before_segments = dataset.load_indices_by_name("vector_idx").await.unwrap();
    assert_eq!(before_segments.len(), 2);
    let before_by_fragment = before_segments
        .iter()
        .map(|metadata| {
            let fragments = metadata
                .fragment_bitmap
                .as_ref()
                .unwrap()
                .iter()
                .collect::<Vec<_>>();
            (fragments, metadata.uuid)
        })
        .collect::<HashMap<_, _>>();
    assert_eq!(before_by_fragment.len(), 2);

    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();

    let after_segments = dataset.load_indices_by_name("vector_idx").await.unwrap();
    assert_eq!(after_segments.len(), 2);
    let after_by_fragment = after_segments
        .iter()
        .map(|metadata| {
            let fragments = metadata
                .fragment_bitmap
                .as_ref()
                .unwrap()
                .iter()
                .collect::<Vec<_>>();
            (fragments, metadata.uuid)
        })
        .collect::<HashMap<_, _>>();
    assert_eq!(after_by_fragment.len(), 2);

    assert_ne!(
        before_by_fragment[&vec![0]],
        after_by_fragment[&vec![0]],
        "expected optimize to replace the oversized segment"
    );
    assert_eq!(
        before_by_fragment[&vec![1]],
        after_by_fragment[&vec![1]],
        "expected optimize to leave the smaller segment untouched"
    );

    let logical_index = dataset
        .open_logical_vector_index("vector", "vector_idx")
        .await
        .unwrap();
    let partitions_per_segment = logical_index
        .as_ivf()
        .unwrap()
        .num_partitions_per_segment()
        .into_iter()
        .collect::<HashMap<_, _>>();
    // The 20_000-row partition is split straight to the target size:
    // ceil(20_000 / 4096) = 5 pieces, so 2 + 4 partitions.
    assert_eq!(
        partitions_per_segment[&after_by_fragment[&vec![0]]],
        2 + 20_000_usize.div_ceil(IndexType::IvfFlat.target_partition_size()) - 1
    );
    assert_eq!(partitions_per_segment[&after_by_fragment[&vec![1]]], 2);
}

#[tokio::test]
async fn test_optimize_delta_indices() {
    let dimensions = 16;
    let column_name = "vec";
    let vec_field = Field::new(
        column_name,
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dimensions,
        ),
        false,
    );
    let other_column_name = "other_vec";
    let other_vec_field = Field::new(
        other_column_name,
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dimensions,
        ),
        false,
    );
    let schema = Arc::new(Schema::new(vec![vec_field, other_vec_field]));

    let float_arr = generate_random_array(512 * dimensions as usize);

    let vectors = Arc::new(
        arrow_array::FixedSizeListArray::try_new_from_values(float_arr, dimensions).unwrap(),
    );

    let record_batch =
        RecordBatch::try_new(schema.clone(), vec![vectors.clone(), vectors.clone()]).unwrap();

    let reader = RecordBatchIterator::new(
        vec![record_batch.clone()].into_iter().map(Ok),
        schema.clone(),
    );

    let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();
    let params = VectorIndexParams::ivf_pq(10, 8, 2, MetricType::L2, 10);
    dataset
        .create_index(
            &[column_name],
            IndexType::Vector,
            Some("vec_idx".into()),
            &params,
            true,
        )
        .await
        .unwrap();
    dataset
        .create_index(
            &[other_column_name],
            IndexType::Vector,
            Some("other_vec_idx".into()),
            &params,
            true,
        )
        .await
        .unwrap();

    async fn get_stats(dataset: &Dataset, name: &str) -> serde_json::Value {
        serde_json::from_str(&dataset.index_statistics(name).await.unwrap()).unwrap()
    }
    async fn get_meta(dataset: &Dataset, name: &str) -> Vec<IndexMetadata> {
        dataset
            .load_indices()
            .await
            .unwrap()
            .iter()
            .filter(|m| m.name == name)
            .cloned()
            .collect()
    }
    fn get_bitmap(meta: &IndexMetadata) -> Vec<u32> {
        meta.fragment_bitmap.as_ref().unwrap().iter().collect()
    }
    fn assert_segment_aliases(stats: &serde_json::Value) {
        assert_eq!(stats["num_segments"], stats["num_indices"]);
        assert_eq!(stats["segments"], stats["indices"]);
    }

    let stats = get_stats(&dataset, "vec_idx").await;
    assert_segment_aliases(&stats);
    assert_eq!(stats["num_unindexed_rows"], 0);
    assert_eq!(stats["num_indexed_rows"], 512);
    assert_eq!(stats["num_indexed_fragments"], 1);
    assert_eq!(stats["num_indices"], 1);
    let meta = get_meta(&dataset, "vec_idx").await;
    assert_eq!(meta.len(), 1);
    assert_eq!(get_bitmap(&meta[0]), vec![0]);

    let reader = RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema.clone());
    dataset.append(reader, None).await.unwrap();
    let stats = get_stats(&dataset, "vec_idx").await;
    assert_segment_aliases(&stats);
    assert_eq!(stats["num_unindexed_rows"], 512);
    assert_eq!(stats["num_indexed_rows"], 512);
    assert_eq!(stats["num_indexed_fragments"], 1);
    assert_eq!(stats["num_unindexed_fragments"], 1);
    assert_eq!(stats["num_indices"], 1);
    let meta = get_meta(&dataset, "vec_idx").await;
    assert_eq!(meta.len(), 1);
    assert_eq!(get_bitmap(&meta[0]), vec![0]);

    dataset
        .optimize_indices(&OptimizeOptions::append().index_names(vec![])) // Does nothing because no index name is passed
        .await
        .unwrap();
    let stats = get_stats(&dataset, "vec_idx").await;
    assert_segment_aliases(&stats);
    assert_eq!(stats["num_unindexed_rows"], 512);
    assert_eq!(stats["num_indexed_rows"], 512);
    assert_eq!(stats["num_indexed_fragments"], 1);
    assert_eq!(stats["num_unindexed_fragments"], 1);
    assert_eq!(stats["num_indices"], 1);
    let meta = get_meta(&dataset, "vec_idx").await;
    assert_eq!(meta.len(), 1);
    assert_eq!(get_bitmap(&meta[0]), vec![0]);

    // optimize the other index
    dataset
        .optimize_indices(&OptimizeOptions::append().index_names(vec!["other_vec_idx".to_owned()]))
        .await
        .unwrap();
    let stats = get_stats(&dataset, "vec_idx").await;
    assert_segment_aliases(&stats);
    assert_eq!(stats["num_unindexed_rows"], 512);
    assert_eq!(stats["num_indexed_rows"], 512);
    assert_eq!(stats["num_indexed_fragments"], 1);
    assert_eq!(stats["num_unindexed_fragments"], 1);
    assert_eq!(stats["num_indices"], 1);
    let meta = get_meta(&dataset, "vec_idx").await;
    assert_eq!(meta.len(), 1);
    assert_eq!(get_bitmap(&meta[0]), vec![0]);

    let stats = get_stats(&dataset, "other_vec_idx").await;
    assert_segment_aliases(&stats);
    assert_eq!(stats["num_unindexed_rows"], 0);
    assert_eq!(stats["num_indexed_rows"], 1024);
    assert_eq!(stats["num_indexed_fragments"], 2);
    assert_eq!(stats["num_unindexed_fragments"], 0);
    assert_eq!(stats["num_indices"], 2);
    let meta = get_meta(&dataset, "other_vec_idx").await;
    assert_eq!(meta.len(), 2);
    assert_eq!(get_bitmap(&meta[0]), vec![0]);
    assert_eq!(get_bitmap(&meta[1]), vec![1]);

    dataset
        .optimize_indices(&OptimizeOptions::retrain())
        .await
        .unwrap();

    let stats = get_stats(&dataset, "vec_idx").await;
    assert_segment_aliases(&stats);
    assert_eq!(stats["num_unindexed_rows"], 0);
    assert_eq!(stats["num_indexed_rows"], 1024);
    assert_eq!(stats["num_indexed_fragments"], 2);
    assert_eq!(stats["num_unindexed_fragments"], 0);
    assert_eq!(stats["num_indices"], 1);
    let meta = get_meta(&dataset, "vec_idx").await;
    assert_eq!(meta.len(), 1);
    assert_eq!(get_bitmap(&meta[0]), vec![0, 1]);

    dataset
        .optimize_indices(&OptimizeOptions::retrain())
        .await
        .unwrap();
    let stats = get_stats(&dataset, "other_vec_idx").await;
    assert_segment_aliases(&stats);
    assert_eq!(stats["num_unindexed_rows"], 0);
    assert_eq!(stats["num_indexed_rows"], 1024);
    assert_eq!(stats["num_indexed_fragments"], 2);
    assert_eq!(stats["num_unindexed_fragments"], 0);
    assert_eq!(stats["num_indices"], 1);
    let meta = get_meta(&dataset, "other_vec_idx").await;
    assert_eq!(meta.len(), 1);
    assert_eq!(get_bitmap(&meta[0]), vec![0, 1]);
}

#[tokio::test]
async fn test_optimize_ivf_hnsw_sq_delta_indices() {
    let test_dir = TempStrDir::default();
    let dimensions = 16;
    let column_name = "vec";
    let field = Field::new(
        column_name,
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dimensions,
        ),
        false,
    );
    let schema = Arc::new(Schema::new(vec![field]));

    let float_arr = generate_random_array(512 * dimensions as usize);

    let vectors =
        arrow_array::FixedSizeListArray::try_new_from_values(float_arr, dimensions).unwrap();

    let record_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();

    let reader = RecordBatchIterator::new(
        vec![record_batch.clone()].into_iter().map(Ok),
        schema.clone(),
    );

    let test_uri = &test_dir;
    let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

    let ivf_params = IvfBuildParams::default();
    let hnsw_params = HnswBuildParams::default();
    let sq_params = SQBuildParams::default();
    let params = VectorIndexParams::with_ivf_hnsw_sq_params(
        MetricType::L2,
        ivf_params,
        hnsw_params,
        sq_params,
    );
    dataset
        .create_index(
            &[column_name],
            IndexType::Vector,
            Some("vec_idx".into()),
            &params,
            true,
        )
        .await
        .unwrap();

    let stats: serde_json::Value =
        serde_json::from_str(&dataset.index_statistics("vec_idx").await.unwrap()).unwrap();
    assert_eq!(stats["num_unindexed_rows"], 0);
    assert_eq!(stats["num_indexed_rows"], 512);
    assert_eq!(stats["num_indexed_fragments"], 1);
    assert_eq!(stats["num_indices"], 1);

    let reader = RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema.clone());
    dataset.append(reader, None).await.unwrap();
    let mut dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();
    let stats: serde_json::Value =
        serde_json::from_str(&dataset.index_statistics("vec_idx").await.unwrap()).unwrap();
    assert_eq!(stats["num_unindexed_rows"], 512);
    assert_eq!(stats["num_indexed_rows"], 512);
    assert_eq!(stats["num_indexed_fragments"], 1);
    assert_eq!(stats["num_unindexed_fragments"], 1);
    assert_eq!(stats["num_indices"], 1);

    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();

    let stats: serde_json::Value =
        serde_json::from_str(&dataset.index_statistics("vec_idx").await.unwrap()).unwrap();
    assert_eq!(stats["num_unindexed_rows"], 0);
    assert_eq!(stats["num_indexed_rows"], 1024);
    assert_eq!(stats["num_indexed_fragments"], 2);
    assert_eq!(stats["num_unindexed_fragments"], 0);
    assert_eq!(stats["num_indices"], 2);

    dataset
        .optimize_indices(&OptimizeOptions::retrain())
        .await
        .unwrap();
    let stats: serde_json::Value =
        serde_json::from_str(&dataset.index_statistics("vec_idx").await.unwrap()).unwrap();
    assert_eq!(stats["num_unindexed_rows"], 0);
    assert_eq!(stats["num_indexed_rows"], 1024);
    assert_eq!(stats["num_indexed_fragments"], 2);
    assert_eq!(stats["num_unindexed_fragments"], 0);
    assert_eq!(stats["num_indices"], 1);
}

#[rstest]
#[tokio::test]
async fn test_optimize_fts(#[values(false, true)] with_position: bool) {
    let words = ["apple", "banana", "cherry", "date"];

    let dir = TempStrDir::default();
    let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
    let data = StringArray::from_iter_values(words.iter().map(|s| s.to_string()));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(data)]).unwrap();
    let batch_iterator = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

    let mut dataset = Dataset::write(batch_iterator, &dir, None).await.unwrap();

    let params = InvertedIndexParams::default()
        .lower_case(false)
        .with_position(with_position);
    dataset
        .create_index(&["text"], IndexType::Inverted, None, &params, true)
        .await
        .unwrap();

    async fn assert_indexed_rows(dataset: &Dataset, expected_indexed_rows: usize) {
        let stats = dataset.index_statistics("text_idx").await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
        let indexed_rows = stats["num_indexed_rows"].as_u64().unwrap() as usize;
        let unindexed_rows = stats["num_unindexed_rows"].as_u64().unwrap() as usize;
        let num_rows = dataset.count_all_rows().await.unwrap();
        assert_eq!(indexed_rows, expected_indexed_rows);
        assert_eq!(unindexed_rows, num_rows - expected_indexed_rows);
        for metadata in dataset
            .load_indices()
            .await
            .unwrap()
            .iter()
            .filter(|metadata| metadata.name == "text_idx")
        {
            let index = dataset
                .open_scalar_index("text", &metadata.uuid, &NoOpMetricsCollector)
                .await
                .unwrap();
            let live_index = Arc::downgrade(&index);
            drop(index);
            assert!(
                live_index.upgrade().is_none(),
                "optimization retained a live FTS index in the session cache"
            );
        }
    }

    let num_rows = dataset.count_all_rows().await.unwrap();
    assert_indexed_rows(&dataset, num_rows).await;

    let new_words = ["elephant", "fig", "grape", "honeydew"];
    let new_data = StringArray::from_iter_values(new_words.iter().map(|s| s.to_string()));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(new_data)]).unwrap();
    let batch_iter = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    dataset.append(batch_iter, None).await.unwrap();
    assert_indexed_rows(&dataset, num_rows).await;

    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    let num_rows = dataset.count_all_rows().await.unwrap();
    assert_indexed_rows(&dataset, num_rows).await;

    for &word in words.iter().chain(new_words.iter()) {
        let query_result = dataset
            .scan()
            .project(&["text"])
            .unwrap()
            .full_text_search(FullTextSearchQuery::new(word.to_string()))
            .unwrap()
            .limit(Some(10), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        let texts = query_result["text"]
            .as_string::<i32>()
            .iter()
            .map(|v| match v {
                None => "".to_string(),
                Some(v) => v.to_string(),
            })
            .collect::<Vec<String>>();

        assert_eq!(texts.len(), 1);
        assert_eq!(texts[0], word);
    }

    let uppercase_words = ["Apple", "Banana", "Cherry", "Date"];
    for &word in uppercase_words.iter() {
        let query_result = dataset
            .scan()
            .project(&["text"])
            .unwrap()
            .full_text_search(FullTextSearchQuery::new(word.to_string()))
            .unwrap()
            .limit(Some(10), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        let texts = query_result["text"]
            .as_string::<i32>()
            .iter()
            .map(|v| match v {
                None => "".to_string(),
                Some(v) => v.to_string(),
            })
            .collect::<Vec<String>>();

        assert_eq!(texts.len(), 0);
    }
    let new_data = StringArray::from_iter_values(uppercase_words.iter().map(|s| s.to_string()));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(new_data)]).unwrap();
    let batch_iter = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    dataset.append(batch_iter, None).await.unwrap();
    assert_indexed_rows(&dataset, num_rows).await;

    // we should be able to query the new words
    for &word in uppercase_words.iter() {
        let query_result = dataset
            .scan()
            .project(&["text"])
            .unwrap()
            .full_text_search(FullTextSearchQuery::new(word.to_string()))
            .unwrap()
            .limit(Some(10), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        let texts = query_result["text"]
            .as_string::<i32>()
            .iter()
            .map(|v| match v {
                None => "".to_string(),
                Some(v) => v.to_string(),
            })
            .collect::<Vec<String>>();

        assert_eq!(texts.len(), 1, "query: {}, texts: {:?}", word, texts);
        assert_eq!(texts[0], word, "query: {}, texts: {:?}", word, texts);
    }

    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    let num_rows = dataset.count_all_rows().await.unwrap();
    assert_indexed_rows(&dataset, num_rows).await;

    // we should be able to query the new words after optimization
    for &word in uppercase_words.iter() {
        let query_result = dataset
            .scan()
            .project(&["text"])
            .unwrap()
            .full_text_search(FullTextSearchQuery::new(word.to_string()))
            .unwrap()
            .limit(Some(10), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        let texts = query_result["text"]
            .as_string::<i32>()
            .iter()
            .map(|v| match v {
                None => "".to_string(),
                Some(v) => v.to_string(),
            })
            .collect::<Vec<String>>();

        assert_eq!(texts.len(), 1, "query: {}, texts: {:?}", word, texts);
        assert_eq!(texts[0], word, "query: {}, texts: {:?}", word, texts);

        // we should be able to query the new words after compaction
        compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap();
        assert_indexed_rows(&dataset, num_rows).await;
        for &word in uppercase_words.iter() {
            let query_result = dataset
                .scan()
                .project(&["text"])
                .unwrap()
                .full_text_search(FullTextSearchQuery::new(word.to_string()))
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();
            let texts = query_result["text"]
                .as_string::<i32>()
                .iter()
                .map(|v| match v {
                    None => "".to_string(),
                    Some(v) => v.to_string(),
                })
                .collect::<Vec<String>>();
            assert_eq!(texts.len(), 1, "query: {}, texts: {:?}", word, texts);
            assert_eq!(texts[0], word, "query: {}, texts: {:?}", word, texts);
        }
        assert_indexed_rows(&dataset, num_rows).await;
    }
}

#[tokio::test]
async fn test_optimize_fts_respects_num_indices_to_merge() {
    use lance_index::scalar::inverted::query::PhraseQuery;

    async fn append_texts(dataset: &mut Dataset, schema: Arc<Schema>, rows: &[(i32, &str)]) {
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(rows.iter().map(|(id, _)| *id))),
                Arc::new(StringArray::from_iter_values(
                    rows.iter().map(|(_, text)| *text),
                )),
            ],
        )
        .unwrap();
        let batch_iter = RecordBatchIterator::new(vec![Ok(batch)], schema);
        dataset.append(batch_iter, None).await.unwrap();
    }

    async fn num_fts_segments(dataset: &Dataset) -> usize {
        let stats = dataset.index_statistics("text_idx").await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
        stats["num_indices"].as_u64().unwrap() as usize
    }

    async fn assert_fts_ids(dataset: &Dataset, query: FullTextSearchQuery, expected_ids: &[i32]) {
        let result = dataset
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(query)
            .unwrap()
            .limit(Some(20), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let mut ids = result["id"].as_primitive::<Int32Type>().values().to_vec();
        ids.sort_unstable();
        assert_eq!(ids, expected_ids);
    }

    let dir = TempStrDir::default();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("text", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values([0, 1])),
            Arc::new(StringArray::from_iter_values([
                "alpha base phrase",
                "beta base phrase",
            ])),
        ],
    )
    .unwrap();
    let batch_iterator = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

    let mut dataset = Dataset::write(batch_iterator, &dir, None).await.unwrap();
    let params = InvertedIndexParams::default().with_position(true);
    dataset
        .create_index(&["text"], IndexType::Inverted, None, &params, true)
        .await
        .unwrap();
    assert_eq!(num_fts_segments(&dataset).await, 1);

    append_texts(&mut dataset, schema.clone(), &[(2, "gamma delta phrase")]).await;
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    assert_eq!(num_fts_segments(&dataset).await, 2);

    append_texts(&mut dataset, schema.clone(), &[(3, "epsilon zeta phrase")]).await;
    dataset
        .optimize_indices(&OptimizeOptions::merge(1))
        .await
        .unwrap();
    assert_eq!(num_fts_segments(&dataset).await, 2);
    assert_fts_ids(
        &dataset,
        FullTextSearchQuery::new("epsilon".to_owned()),
        &[3],
    )
    .await;

    append_texts(&mut dataset, schema.clone(), &[(4, "eta theta phrase")]).await;
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    assert_eq!(num_fts_segments(&dataset).await, 3);

    dataset
        .optimize_indices(&OptimizeOptions::merge(2))
        .await
        .unwrap();
    assert_eq!(num_fts_segments(&dataset).await, 2);
    assert_fts_ids(
        &dataset,
        FullTextSearchQuery::new_query(PhraseQuery::new("eta theta".to_owned()).into()),
        &[4],
    )
    .await;

    append_texts(&mut dataset, schema, &[(5, "iota kappa phrase")]).await;
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    assert_eq!(num_fts_segments(&dataset).await, 3);

    dataset
        .optimize_indices(&OptimizeOptions::merge(10))
        .await
        .unwrap();
    assert_eq!(num_fts_segments(&dataset).await, 1);
    assert_fts_ids(&dataset, FullTextSearchQuery::new("alpha".to_owned()), &[0]).await;
    assert_fts_ids(&dataset, FullTextSearchQuery::new("iota".to_owned()), &[5]).await;
}

/// An index built from a partition count, or from centroids that fix one,
/// records the vectors per partition it was built at and keeps its
/// partitions through an optimize with nothing new to index. With no target
/// partition size recorded, optimize joined every partition under a quarter
/// of the index type's default, and 500 vectors a partition is under that
/// for every IVF type. Null rows are not indexed, so they do not count: a
/// target taken from 50,000 rows would join partitions under 625.
#[rstest]
#[case::num_partitions(VectorIndexParams::ivf_flat(20, DistanceType::L2))]
#[case::centroids(ivf_flat_from_centroids(20))]
#[tokio::test]
async fn test_optimize_keeps_trained_partitions(
    #[case] params: VectorIndexParams,
    #[values(0, 40_000)] null_rows: usize,
) {
    let test_dir = tempfile::tempdir().unwrap();
    let mut dataset = partly_null_vector_dataset(test_dir.path(), 10_000 + null_rows, 10_000).await;
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, false)
        .await
        .unwrap();
    assert_eq!(trained_partitions(&dataset).await, vec![20]);
    let details = dataset.load_indices().await.unwrap()[0]
        .index_details
        .clone()
        .unwrap();
    assert_eq!(target_partition_size_from_details(&details), Some(500));

    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    assert_eq!(trained_partitions(&dataset).await, vec![20]);
}

/// The whole point of degrading: the index fills in once the data arrives.
///
/// A definition-only index has no model to append to, so optimizing has to
/// train it from scratch and then cover every fragment, including the small
/// one that existed before the threshold was met.
#[tokio::test]
async fn test_degraded_index_trains_on_optimize_once_data_arrives() {
    let test_dir = tempfile::tempdir().unwrap();
    let mut dataset = small_vector_dataset(test_dir.path(), 100).await;

    let params = VectorIndexParams::ivf_pq(1, 8, 8, DistanceType::L2, 1);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, false)
        .await
        .unwrap();

    // Degraded: 100 rows cannot train a 256-code quantizer.
    let indices = dataset.load_indices().await.unwrap();
    assert!(segment_covers_nothing(&indices[0]));

    // 100 -> 500 across two fragments, clearing the floor.
    let mut dataset = append_vectors(test_dir.path(), 400).await;
    assert_eq!(dataset.count_rows(None).await.unwrap(), 500);
    assert_eq!(dataset.get_fragments().len(), 2);

    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .expect("a definition-only index must train once the data is there");

    // Trained and covering: both fragments, nothing left unindexed.
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    let covered = indices[0].fragment_bitmap.as_ref().unwrap();
    assert_eq!(
        covered.len() as usize,
        dataset.get_fragments().len(),
        "every fragment must be covered, including the pre-threshold one"
    );
    assert!(
        dataset
            .unindexed_fragments("vector_idx")
            .await
            .unwrap()
            .is_empty()
    );
}

/// A table that shrinks below the training threshold keeps working.
///
/// Optimizing was unrunnable in this state: the quantizer cannot train on
/// what is left, and erroring there blocks index maintenance outright, with
/// dropping and recreating the index as the only way out.
#[tokio::test]
async fn test_optimize_indices_after_shrinking_below_the_threshold() {
    let test_dir = tempfile::tempdir().unwrap();
    let mut dataset = small_vector_dataset(test_dir.path(), 1024).await;

    let params = VectorIndexParams::ivf_pq(1, 8, 8, DistanceType::L2, 1);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, false)
        .await
        .unwrap();

    // 1024 -> 224, under a 256-code quantizer's floor.
    dataset.delete("_rowid < 800").await.unwrap();
    assert_eq!(dataset.count_rows(None).await.unwrap(), 224);

    // Retrain, not append: append-mode has no new data to index here and so
    // never reaches the quantizer, which is the path that fails.
    dataset
        .optimize_indices(&OptimizeOptions::retrain())
        .await
        .expect("optimizing a table that shrank past the threshold must not fail");

    assert_eq!(dataset.load_indices().await.unwrap().len(), 1);
}

#[tokio::test]
async fn test_optimize_ivf_pq_up_to_date() {
    // https://github.com/lance-format/lance/issues/4016
    let nrows = 256;
    let dimensions = 16;
    let column_name = "vector";
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            column_name,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dimensions,
            ),
            false,
        ),
    ]));

    let float_arr = generate_random_array(nrows * dimensions as usize);
    let vectors =
        arrow_array::FixedSizeListArray::try_new_from_values(float_arr, dimensions).unwrap();
    let record_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(arrow_array::Int32Array::from_iter_values(0..nrows as i32)),
            Arc::new(vectors),
        ],
    )
    .unwrap();

    let reader = RecordBatchIterator::new(
        vec![record_batch.clone()].into_iter().map(Ok),
        schema.clone(),
    );
    let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

    let params = VectorIndexParams::ivf_pq(1, 8, 2, MetricType::L2, 2);
    dataset
        .create_index(&[column_name], IndexType::Vector, None, &params, true)
        .await
        .unwrap();

    let query_vector = generate_random_array(dimensions as usize);

    let nearest = dataset
        .scan()
        .nearest(column_name, &query_vector, 5)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    let ids = nearest["id"].as_primitive::<Int32Type>();
    let mut seen = HashSet::new();
    for id in ids.values() {
        assert!(seen.insert(*id), "Duplicate id found: {}", id);
    }

    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();

    dataset.validate().await.unwrap();

    let nearest_after = dataset
        .scan()
        .nearest(column_name, &query_vector, 5)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    let ids = nearest_after["id"].as_primitive::<Int32Type>();
    let mut seen = HashSet::new();
    for id in ids.values() {
        assert!(seen.insert(*id), "Duplicate id found: {}", id);
    }
}

#[rstest]
#[case::block_size_128(128)]
#[case::block_size_256(256)]
#[tokio::test]
async fn test_optimize_empty_code_fts_index_preserves_params(#[case] block_size: usize) {
    let dir = TempStrDir::default();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("code", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values([0, 1])),
            Arc::new(StringArray::from_iter_values([
                "fn GetUserEmail() {}",
                "fn ParseConfig() {}",
            ])),
        ],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let mut dataset = Dataset::write(reader, &dir, None).await.unwrap();

    let params = InvertedIndexParams::default()
        .analyzer("code")
        .unwrap()
        .block_size(block_size)
        .unwrap()
        .split_identifiers(true)
        .preserve_original(false);
    dataset
        .create_index_builder(&["code"], IndexType::Inverted, &params)
        .name("code_idx".to_string())
        .train(false)
        .await
        .unwrap();

    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    assert_eq!(indices[0].index_version, INVERTED_INDEX_VERSION_V3 as i32);

    dataset.optimize_indices(&Default::default()).await.unwrap();

    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    assert_eq!(indices[0].index_version, INVERTED_INDEX_VERSION_V3 as i32);
    let details = indices[0]
        .index_details
        .as_ref()
        .expect("optimized FTS index should retain index details")
        .to_msg::<InvertedIndexDetails>()
        .unwrap();
    assert_eq!(details.base_tokenizer.as_deref(), Some("code"));
    assert_eq!(details.block_size, Some(block_size as u32));
    let code_config = details
        .code_config
        .expect("optimized code FTS index should retain code configuration");
    assert!(code_config.split_identifiers);
    assert_eq!(code_config.preserve_original, Some(false));

    let stats: serde_json::Value =
        serde_json::from_str(&dataset.index_statistics("code_idx").await.unwrap()).unwrap();
    assert_eq!(stats["num_unindexed_rows"], 0);

    let result = dataset
        .scan()
        .project(&["id"])
        .unwrap()
        .full_text_search(FullTextSearchQuery::new("email".to_string()))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(result["id"].as_primitive::<Int32Type>().values(), &[0]);
}

#[rstest]
#[case::btree(IndexType::BTree, "id", "page_lookup.lance", false)]
#[case::fts(IndexType::Inverted, "text", "tokens.lance", false)]
#[case::fts_036(IndexType::Inverted, "text", "tokens.lance", true)]
#[tokio::test]
async fn test_optimize_reports_segment_open_errors(
    #[case] index_type: IndexType,
    #[case] column: &str,
    #[case] file_suffix: &str,
    #[case] has_036_type_url: bool,
    #[values(false, true)] is_missing_file: bool,
) {
    let test_dir = TempStrDir::default();
    let mut dataset = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col(
            "text",
            array::cycle_utf8_literals(&["hello world", "lance"]),
        )
        .into_dataset(&test_dir, FragmentCount(1), FragmentRowCount(4))
        .await
        .unwrap();
    let params: Box<dyn IndexParams> = match index_type {
        IndexType::BTree => Box::new(ScalarIndexParams::for_builtin(BuiltinIndexType::BTree)),
        IndexType::Inverted => Box::new(InvertedIndexParams::default()),
        _ => unreachable!(),
    };
    dataset
        .create_index(
            &[column],
            index_type,
            Some("test_idx".into()),
            params.as_ref(),
            false,
        )
        .await
        .unwrap();
    if has_036_type_url {
        let original = dataset.load_indices_by_name("test_idx").await.unwrap();
        let mut segment = original[0].clone();
        Arc::make_mut(segment.index_details.as_mut().unwrap()).type_url =
            "type.googleapis.com/lance.index.pb.InvertedIndexDetails".into();
        dataset
            .apply_commit(
                Transaction::new(
                    dataset.manifest.version,
                    Operation::CreateIndex {
                        new_indices: vec![segment],
                        removed_indices: original,
                    },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap();
    }
    let appended = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col(
            "text",
            array::cycle_utf8_literals(&["hello world", "lance"]),
        )
        .into_reader_rows(RowCount::from(4), BatchCount::from(1));
    dataset.append(appended, None).await.unwrap();

    // Reopen with a fresh session so a cached index cannot hide the missing file.
    let mut dataset = Dataset::open(&test_dir).await.unwrap();
    let original = dataset.load_indices_by_name("test_idx").await.unwrap();
    let version = dataset.version().version;
    let plan = crate::index::optimize::plan_optimize_indices(&dataset, &OptimizeOptions::default())
        .await
        .unwrap();
    assert_eq!(plan.tasks.len(), 1);
    assert_eq!(plan.tasks[0].reference.name, "test_idx");

    if is_missing_file {
        let segment = &original[0];
        let file = segment
            .files
            .as_ref()
            .unwrap()
            .iter()
            .find(|file| file.path.ends_with(file_suffix))
            .unwrap();
        let path = dataset
            .indices_dir()
            .join(segment.uuid.to_string())
            .join(file.path.as_str());
        dataset.object_store.delete(&path).await.unwrap();
    }

    let result = dataset.optimize_indices(&OptimizeOptions::default()).await;
    let dataset = Dataset::open(&test_dir).await.unwrap();
    let indices = dataset.load_indices_by_name("test_idx").await.unwrap();
    if is_missing_file {
        let error = result.unwrap_err();
        assert!(matches!(error, Error::Index { .. }), "{error}");
        let message = error.to_string();
        assert!(message.contains("index 'test_idx'"), "{message}");
        assert!(message.contains(&format!("column '{column}'")), "{message}");
        assert!(message.contains(file_suffix), "{message}");
        assert!(
            message.to_ascii_lowercase().contains("not found"),
            "{message}"
        );
        assert_eq!(dataset.version().version, version);
        assert_eq!(indices, original);
    } else {
        result.unwrap();
        let covered: RoaringBitmap = indices
            .iter()
            .flat_map(|index| index.fragment_bitmap.as_ref().unwrap().iter())
            .collect();
        assert_eq!(covered, RoaringBitmap::from_iter([0, 1]));
    }
}

#[rstest]
#[case::unsupported(false, MissingCoverageReason::Unsupported)]
#[case::withdrawn(false, MissingCoverageReason::Withdrawn)]
#[case::empty(true, MissingCoverageReason::Withdrawn)]
#[tokio::test]
async fn test_optimize_skips_an_index_without_fragment_reuse_translation(
    #[case] is_empty: bool,
    #[case] expected_reason: MissingCoverageReason,
) {
    let mut dataset = gen_batch()
        .col("i", array::step::<Int32Type>())
        .col(
            "text",
            array::cycle_utf8_literals(&["hello world", "lance"]),
        )
        .into_ram_dataset(
            FragmentCount(if is_empty { 0 } else { 2 }),
            FragmentRowCount(2),
        )
        .await
        .unwrap();
    let fm_params = ScalarIndexParams::for_builtin(BuiltinIndexType::Fm);
    dataset
        .create_index(
            &["text"],
            IndexType::Fm,
            Some("text_idx".into()),
            &fm_params,
            false,
        )
        .await
        .unwrap();
    let original = load_all_indices_by_name(&dataset, "text_idx")
        .await
        .unwrap()[0]
        .clone();
    if is_empty {
        assert_eq!(dataset.count_rows(None).await.unwrap(), 0);
        let reader = gen_batch()
            .col("i", array::step::<Int32Type>())
            .col(
                "text",
                array::cycle_utf8_literals(&["hello world", "lance"]),
            )
            .into_reader_rows(RowCount::from(2), BatchCount::from(2));
        dataset
            .append(
                reader,
                Some(WriteParams {
                    max_rows_per_file: 2,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
    }
    dataset
        .create_index(
            &["i"],
            IndexType::BTree,
            Some("i_idx".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    dataset
        .apply_commit(
            Transaction::new(
                dataset.manifest.version,
                Operation::ReserveFragments { num_fragments: 2 },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();
    let old_fragments = dataset.fragments().to_vec();
    let (transition, new_fragments) = prepare_partition(&dataset, &[0, 1], 2).await;
    let frag_reuse_index = Some(
        frag_reuse_entry_appending(&dataset, vec![transition])
            .await
            .unwrap(),
    );
    dataset
        .apply_commit(
            Transaction::new(
                dataset.manifest.version,
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments,
                        new_fragments,
                    }],
                    rewritten_indices: vec![],
                    frag_reuse_index,
                },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();

    let appended = gen_batch()
        .col("i", array::step::<Int32Type>())
        .col(
            "text",
            array::cycle_utf8_literals(&["hello world", "lance"]),
        )
        .into_reader_rows(RowCount::from(2), BatchCount::from(1));
    dataset.append(appended, None).await.unwrap();

    // A sibling on a fragment outside the rewrite history is directly readable.
    // It must also be preserved when the old segment cannot be translated.
    let sibling = dataset
        .create_index_builder(&["text"], IndexType::Fm, &fm_params)
        .name("text_idx".into())
        .fragments(vec![4])
        .replace(true)
        .execute_uncommitted()
        .await
        .unwrap();
    dataset
        .apply_commit(
            Transaction::new(
                dataset.manifest.version,
                Operation::CreateIndex {
                    new_indices: vec![sibling],
                    removed_indices: vec![],
                },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();
    if expected_reason == MissingCoverageReason::Unsupported {
        // Current compaction withdraws this coverage. Preserve a fixture with
        // retained coverage to exercise readers of existing tagged histories.
        let mut indices = crate::index::loading::load_all_indices(&dataset)
            .await
            .unwrap()
            .as_ref()
            .clone();
        let segment = indices
            .iter_mut()
            .find(|segment| segment.uuid == original.uuid)
            .unwrap();
        *segment = original;
        persist_fixture(&mut dataset, indices).await;
    }
    let before = load_all_indices_by_name(&dataset, "text_idx")
        .await
        .unwrap();
    assert_eq!(before.len(), 2);
    assert_eq!(
        segment_coverage_reason(&dataset, &before[0]).await.unwrap(),
        Some(expected_reason)
    );
    assert_eq!(
        segment_coverage_reason(&dataset, &before[1]).await.unwrap(),
        None
    );

    let plan = crate::index::optimize::plan_optimize_indices(&dataset, &OptimizeOptions::default())
        .await
        .unwrap();
    assert_eq!(plan.tasks.len(), 1);
    assert_eq!(plan.tasks[0].reference.name, "i_idx");
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    assert_eq!(
        load_all_indices_by_name(&dataset, "text_idx")
            .await
            .unwrap(),
        before
    );
    assert!(
        dataset
            .unindexed_fragments("i_idx")
            .await
            .unwrap()
            .is_empty()
    );
    dataset.validate().await.unwrap();

    let error = dataset
        .optimize_indices(&OptimizeOptions::default().index_names(vec!["text_idx".into()]))
        .await
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }), "{error}");
    assert!(
        error.to_string().contains("cannot be translated"),
        "{error}"
    );
}

#[tokio::test]
async fn test_optimize_trains_vector_definition_under_tagged_history() {
    let mut dataset = gen_batch()
        .col("i", array::step::<Int32Type>())
        .col("vec", array::rand_vec::<Float32Type>(Dimension::from(4)))
        .into_ram_dataset(FragmentCount(2), FragmentRowCount(4))
        .await
        .unwrap();
    dataset
        .create_index_builder(
            &["vec"],
            IndexType::Vector,
            &VectorIndexParams::ivf_flat(1, MetricType::L2),
        )
        .name("vec_idx".into())
        .train(false)
        .await
        .unwrap();
    dataset
        .apply_commit(
            Transaction::new(
                dataset.manifest.version,
                Operation::ReserveFragments { num_fragments: 2 },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();
    let old_fragments = dataset.fragments().to_vec();
    let (transition, new_fragments) = prepare_partition(&dataset, &[0, 1], 2).await;
    let frag_reuse_index = Some(
        frag_reuse_entry_appending(&dataset, vec![transition])
            .await
            .unwrap(),
    );
    dataset
        .apply_commit(
            Transaction::new(
                dataset.manifest.version,
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments,
                        new_fragments,
                    }],
                    rewritten_indices: vec![],
                    frag_reuse_index,
                },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();
    let before = load_all_indices_by_name(&dataset, "vec_idx").await.unwrap();
    assert_eq!(before.len(), 1);
    assert!(crate::index::append::is_definition_only_segment(&before[0]));
    assert_eq!(
        segment_coverage_reason(&dataset, &before[0]).await.unwrap(),
        Some(MissingCoverageReason::Withdrawn)
    );
    let plan = crate::index::optimize::plan_optimize_indices(&dataset, &OptimizeOptions::default())
        .await
        .unwrap();
    assert_eq!(plan.tasks.len(), 1);
    assert_eq!(plan.tasks[0].reference.name, "vec_idx");

    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    let after = dataset.load_indices_by_name("vec_idx").await.unwrap();
    assert_eq!(after.len(), 1);
    assert_ne!(after[0].uuid, before[0].uuid);
    assert_eq!(
        after[0].fragment_bitmap.as_ref().unwrap(),
        dataset.fragment_bitmap.as_ref()
    );
    dataset.validate().await.unwrap();

    let rows = dataset.scan().try_into_batch().await.unwrap();
    let query = rows["vec"].as_fixed_size_list().value(0);
    let expected = dataset
        .scan()
        .nearest("vec", query.as_ref(), 4)
        .unwrap()
        .use_index(false)
        .project(&["i"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let actual = dataset
        .scan()
        .nearest("vec", query.as_ref(), 4)
        .unwrap()
        .project(&["i"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let expected = expected["i"].as_primitive::<Int32Type>();
    let actual = actual["i"].as_primitive::<Int32Type>();
    let hits = actual
        .values()
        .iter()
        .filter(|id| expected.values().contains(id))
        .count();
    let recall = hits as f64 / expected.len() as f64;
    assert_eq!(recall, 1.0);
}

#[rstest]
#[case::jieba("jieba")]
#[case::lindera("lindera")]
#[case::unknown("unknown-tokenizer")]
#[tokio::test]
async fn test_optimize_skips_an_index_with_an_unavailable_tokenizer(
    #[case] tokenizer: &str,
    #[values(false, true)] is_bounded: bool,
    #[values(false, true)] has_manifest_tokenizer: bool,
    #[values(false, true)] has_stable_row_ids: bool,
) {
    let test_dir = TempStrDir::default();
    let mut dataset = gen_batch()
        .col("i", array::step::<Int32Type>())
        .col(
            "text",
            array::cycle_utf8_literals(&["hello world", "lance"]),
        )
        .into_dataset_with_params(
            &test_dir,
            FragmentCount(3),
            FragmentRowCount(2),
            Some(WriteParams {
                max_rows_per_file: 2,
                enable_stable_row_ids: has_stable_row_ids,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    let mut segments = Vec::with_capacity(2);
    for fragment in [0, 1] {
        let mut segment = dataset
            .create_index_builder(
                &["text"],
                IndexType::Inverted,
                &InvertedIndexParams::default(),
            )
            .name("text_idx".into())
            .fragments(vec![fragment])
            .execute_uncommitted()
            .await
            .unwrap();
        if let Some(files) =
            crate::index::scalar::inverted::finalize_segment_files_if_needed(&dataset, &segment)
                .await
                .unwrap()
        {
            segment.files = Some(index_files_to_table(files));
        }
        segments.push(segment);
    }
    // Model an index written by a process with a tokenizer model installed.
    // Rewrite both persisted copies of its configuration, leaving its data
    // files intact. This process cannot load the uniquely named model.
    let tokenizer = format!("{tokenizer}/missing-{}", Uuid::new_v4());
    let params = InvertedIndexParams::default().base_tokenizer(tokenizer.clone());
    assert!(params.build().is_err());
    let segment = &mut segments[1];
    let store = LanceIndexStore::from_dataset_for_new(&dataset, &segment.uuid).unwrap();
    let reader = store.open_index_file(METADATA_FILE).await.unwrap();
    let schema = reader.schema();
    let batch = reader.read_range(0..1, None).await.unwrap();
    let mut metadata = schema.metadata.clone();
    metadata.insert("params".into(), serde_json::to_string(&params).unwrap());
    let mut writer = store
        .new_index_file(
            METADATA_FILE,
            Arc::new(Schema::new(batch.schema().fields().clone())),
        )
        .await
        .unwrap();
    writer.write_record_batch(batch).await.unwrap();
    let file = writer.finish_with_metadata(metadata).await.unwrap();
    let recorded_file = segment
        .files
        .as_mut()
        .unwrap()
        .iter_mut()
        .find(|existing| existing.path == file.path)
        .unwrap();
    recorded_file.size_bytes = file.size_bytes;
    recorded_file.file_metadata_size_bytes = None;
    let details = if has_manifest_tokenizer {
        InvertedIndexDetails::try_from(&params).unwrap()
    } else {
        InvertedIndexDetails::default()
    };
    segment.index_details = Some(Arc::new(prost_types::Any::from_msg(&details).unwrap()));
    dataset
        .apply_commit(
            Transaction::new(
                dataset.manifest.version,
                Operation::CreateIndex {
                    new_indices: segments.clone(),
                    removed_indices: vec![],
                },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();
    dataset
        .create_index_builder(&["i"], IndexType::BTree, &ScalarIndexParams::default())
        .name("i_idx".into())
        .fragments(vec![0])
        .await
        .unwrap();
    let mut dataset = Dataset::open(&test_dir).await.unwrap();
    let before = load_all_indices_by_name(&dataset, "text_idx")
        .await
        .unwrap();
    let error = dataset
        .open_scalar_index_for_maintenance("text", &segments[1].uuid, &NoOpMetricsCollector)
        .await
        .unwrap_err();
    assert!(
        matches!(error, Error::InvalidInput { .. } | Error::IO { .. }),
        "{error}"
    );
    assert!(error.to_string().contains(&tokenizer), "{error}");

    let options = OptimizeOptions::default().max_rows_per_task(is_bounded.then_some(2));
    let plan = crate::index::optimize::plan_optimize_indices(&dataset, &options)
        .await
        .unwrap();
    assert_eq!(
        plan.tasks
            .iter()
            .filter(|task| task.reference.name == "i_idx")
            .count(),
        if is_bounded { 2 } else { 1 }
    );
    let named = options.clone().index_names(vec!["text_idx".into()]);
    if !is_bounded || has_stable_row_ids {
        // Unsplit tasks can use tokenizer resources installed only on workers.
        // FTS remains unsplit with stable row IDs, even in bounded plans.
        assert_eq!(
            plan.tasks
                .iter()
                .filter(|task| task.reference.name == "text_idx")
                .count(),
            1
        );
        let plan = crate::index::optimize::plan_optimize_indices(&dataset, &named)
            .await
            .unwrap();
        assert_eq!(plan.tasks.len(), 1);
        assert_eq!(plan.tasks[0].reference.name, "text_idx");
    } else {
        assert!(plan.tasks.iter().all(|task| task.reference.name == "i_idx"));
        let error = crate::index::optimize::plan_optimize_indices(&dataset, &named)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        assert!(error.to_string().contains("text_idx"), "{error}");
        assert!(error.to_string().contains(&tokenizer), "{error}");
    }
    dataset.optimize_indices(&options).await.unwrap();
    assert_eq!(
        load_all_indices_by_name(&dataset, "text_idx")
            .await
            .unwrap(),
        before
    );
    assert!(
        dataset
            .unindexed_fragments("i_idx")
            .await
            .unwrap()
            .is_empty()
    );
    dataset.validate().await.unwrap();

    let error = dataset
        .optimize_indices(&OptimizeOptions::default().index_names(vec!["text_idx".into()]))
        .await
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }), "{error}");
    assert!(
        error
            .to_string()
            .contains("which this process cannot build"),
        "{error}"
    );
}

#[tokio::test]
async fn test_optimize_fts_with_legacy_tokenizer_details() {
    let test_dir = copy_test_data_to_tmp("0.27.0/legacy_fts_index").unwrap();
    let mut dataset = Dataset::open(&test_dir.path_str()).await.unwrap();
    let original = dataset.load_indices().await.unwrap();
    assert_eq!(original.len(), 1);
    assert!(
        original[0]
            .index_details
            .as_ref()
            .unwrap()
            .to_msg::<InvertedIndexDetails>()
            .unwrap()
            .base_tokenizer
            .is_none()
    );
    let reader = gen_batch()
        .col("text", array::cycle_utf8_literals(&["happy puppy"]))
        .into_reader_rows(RowCount::from(2), BatchCount::from(1));
    dataset.append(reader, None).await.unwrap();
    let options = OptimizeOptions::default();
    let plan = crate::index::optimize::plan_optimize_indices(&dataset, &options)
        .await
        .unwrap();
    assert!(!plan.tasks.is_empty());
    dataset.optimize_indices(&options).await.unwrap();
    assert!(
        dataset
            .unindexed_fragments(&original[0].name)
            .await
            .unwrap()
            .is_empty()
    );
    dataset.validate().await.unwrap();
}

/// Optimizing a name whose segments this build cannot all read would commit
/// a merged segment overlapping the one it left behind - a state
/// `Dataset::validate` reports as corruption and no later commit can heal.
#[tokio::test]
async fn test_optimize_skips_an_index_with_an_unsupported_segment() {
    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();
    let mut dataset = dataset_with_an_index_from_a_newer_build(test_uri, "id_idx").await;

    // A readable segment beside the hidden one, under the same name, so the
    // group is exactly the mixed case: `id_idx` is now partly readable.
    let hidden = manifest_index(&dataset, "id_idx").await;
    let readable_sibling = IndexMetadata {
        uuid: Uuid::new_v4(),
        index_version: hidden.index_version - 1,
        fragment_bitmap: Some(RoaringBitmap::new()),
        ..hidden
    };
    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: vec![readable_sibling],
            removed_indices: vec![],
        },
        None,
    );
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await
        .unwrap();
    let segments_of = |indices: Vec<IndexMetadata>| {
        indices
            .into_iter()
            .filter(|idx| idx.name == "id_idx")
            .collect::<Vec<_>>()
    };
    let before = segments_of(raw_manifest_indices(&dataset).await);
    assert_eq!(before.len(), 2, "the fixture must build the mixed group");

    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();

    assert_eq!(
        segments_of(raw_manifest_indices(&dataset).await),
        before,
        "optimize touched a name carrying a segment this build cannot read"
    );
    dataset.validate().await.unwrap();
}

#[tokio::test]
async fn test_optimize_rebuilds_dormant_vector_index_instead_of_merging_stale_rows() {
    use crate::dataset::UpdateBuilder;

    let mut dataset = gen_batch()
        .col("vec", array::rand_vec::<Float32Type>(Dimension::from(4)))
        .into_ram_dataset_with_params(
            FragmentCount::from(2),
            FragmentRowCount::from(3),
            Some(WriteParams {
                max_rows_per_file: 3,
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    let original = dataset
        .scan()
        .project(&["vec"])
        .unwrap()
        .limit(Some(1), None)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let query = original["vec"].as_fixed_size_list().value(0);

    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some("vec_idx".to_string()),
            &VectorIndexParams::ivf_flat(1, MetricType::L2),
            true,
        )
        .await
        .unwrap();

    // A full rewrite replaces every covered fragment; the definition is
    // retained as a dormant segment with empty effective coverage.
    let mut updated = UpdateBuilder::new(Arc::new(dataset))
        .set("vec", "array[100.0, 100.0, 100.0, 100.0]")
        .unwrap()
        .build()
        .unwrap()
        .execute()
        .await
        .unwrap()
        .new_dataset
        .as_ref()
        .clone();
    let retained = updated.load_indices_by_name("vec_idx").await.unwrap();
    assert_eq!(retained.len(), 1);
    assert!(
        retained[0]
            .effective_fragment_bitmap(&updated.fragment_bitmap)
            .unwrap()
            .is_empty()
    );

    // Optimize must rebuild from live data and replace the dormant
    // segment, not merge its stale postings back in.
    updated.optimize_indices(&Default::default()).await.unwrap();
    let rebuilt = updated.load_indices_by_name("vec_idx").await.unwrap();
    assert_eq!(rebuilt.len(), 1);
    assert!(
        !rebuilt[0]
            .effective_fragment_bitmap(&updated.fragment_bitmap)
            .unwrap()
            .is_empty()
    );

    let result = updated
        .scan()
        .nearest("vec", query.as_ref(), 1)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let distance = result["_distance"].as_primitive::<Float32Type>().value(0);
    assert!(
        distance > 1_000.0,
        "nearest distance {distance} should reflect the rewritten vectors, not stale postings"
    );
}
