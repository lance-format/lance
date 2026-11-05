// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::support::*;
use super::*;

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
