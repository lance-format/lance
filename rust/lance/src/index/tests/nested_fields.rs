// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

#[rstest]
#[case::simple("value", "data.value")]
#[case::quoted("value.with.dot", "data.`value.with.dot`")]
#[tokio::test]
async fn test_initialize_index_on_nested_field(
    #[case] nested_field_name: &str,
    #[case] nested_field_path: &str,
) {
    let nested_field = Arc::new(Field::new(nested_field_name, DataType::Int32, false));
    let schema = Arc::new(Schema::new(vec![Field::new(
        "data",
        DataType::Struct(vec![nested_field.clone()].into()),
        false,
    )]));
    let nested_values = arrow_array::StructArray::from(vec![(
        nested_field,
        Arc::new(Int32Array::from_iter_values(0..10)) as Arc<dyn arrow_array::Array>,
    )]);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(nested_values)]).unwrap();

    let test_dir = TempStrDir::default();
    let source_uri = format!("{}/source", test_dir);
    let target_uri = format!("{}/target", test_dir);

    let source_reader =
        RecordBatchIterator::new(vec![batch.clone()].into_iter().map(Ok), schema.clone());
    let mut source_dataset = Dataset::write(source_reader, &source_uri, None)
        .await
        .unwrap();
    source_dataset
        .create_index(
            &[nested_field_path],
            IndexType::BTree,
            Some("nested_idx".to_string()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let source_dataset = Dataset::open(&source_uri).await.unwrap();

    let target_reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
    let mut target_dataset = Dataset::write(target_reader, &target_uri, None)
        .await
        .unwrap();
    target_dataset
        .initialize_index(&source_dataset, "nested_idx")
        .await
        .unwrap();

    let target_indices = target_dataset.load_indices().await.unwrap();
    assert_eq!(target_indices.len(), 1);
    assert_eq!(
        target_dataset
            .schema()
            .field_path(target_indices[0].fields[0])
            .unwrap(),
        nested_field_path
    );
}

#[tokio::test]
async fn test_vector_index_on_nested_field_with_dots() {
    let dimensions = 16;
    let num_rows = 256;

    // Create schema with nested field containing dots in the name
    let struct_field = Field::new(
        "embedding_data",
        DataType::Struct(
            vec![
                Field::new(
                    "vector.v1", // Field name with dot
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        dimensions,
                    ),
                    false,
                ),
                Field::new(
                    "vector.v2", // Another field name with dot
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        dimensions,
                    ),
                    false,
                ),
                Field::new("metadata", DataType::Utf8, false),
            ]
            .into(),
        ),
        false,
    );

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        struct_field,
    ]));

    // Generate test data
    let float_arr_v1 = generate_random_array(num_rows * dimensions as usize);
    let vectors_v1 = FixedSizeListArray::try_new_from_values(float_arr_v1, dimensions).unwrap();

    let float_arr_v2 = generate_random_array(num_rows * dimensions as usize);
    let vectors_v2 = FixedSizeListArray::try_new_from_values(float_arr_v2, dimensions).unwrap();

    let ids = Int32Array::from_iter_values(0..num_rows as i32);
    let metadata = StringArray::from_iter_values((0..num_rows).map(|i| format!("meta_{}", i)));

    let struct_array = arrow_array::StructArray::from(vec![
        (
            Arc::new(Field::new(
                "vector.v1",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dimensions,
                ),
                false,
            )),
            Arc::new(vectors_v1) as Arc<dyn arrow_array::Array>,
        ),
        (
            Arc::new(Field::new(
                "vector.v2",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dimensions,
                ),
                false,
            )),
            Arc::new(vectors_v2) as Arc<dyn arrow_array::Array>,
        ),
        (
            Arc::new(Field::new("metadata", DataType::Utf8, false)),
            Arc::new(metadata) as Arc<dyn arrow_array::Array>,
        ),
    ]);

    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(struct_array)]).unwrap();

    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;
    let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

    // Test creating index on nested field with dots using quoted syntax
    let nested_column_path_v1 = "embedding_data.`vector.v1`";
    let params = VectorIndexParams::ivf_pq(10, 8, 2, MetricType::L2, 10);

    dataset
        .create_index(
            &[nested_column_path_v1],
            IndexType::Vector,
            Some("vec_v1_idx".to_string()),
            &params,
            true,
        )
        .await
        .unwrap();

    // Verify index was created
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    assert_eq!(indices[0].name, "vec_v1_idx");

    // Verify the correct field was indexed
    let field_id = indices[0].fields[0];
    let field_path = dataset.schema().field_path(field_id).unwrap();
    assert_eq!(field_path, "embedding_data.`vector.v1`");

    // Test creating index on the second vector field with dots
    let nested_column_path_v2 = "embedding_data.`vector.v2`";
    dataset
        .create_index(
            &[nested_column_path_v2],
            IndexType::Vector,
            Some("vec_v2_idx".to_string()),
            &params,
            true,
        )
        .await
        .unwrap();

    // Verify both indices exist
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 2);

    // Verify we can query using the indexed fields and check the plan
    let query_vector = generate_random_array(dimensions as usize);

    // Check the query plan for the first vector field
    let plan_v1 = dataset
        .scan()
        .nearest(nested_column_path_v1, &query_vector, 5)
        .unwrap()
        .explain_plan(false)
        .await
        .unwrap();

    // Verify the vector index is being used (should show ANNSubIndex or ANNIvfPartition)
    assert!(
        plan_v1.contains("ANNSubIndex") || plan_v1.contains("ANNIvfPartition"),
        "Query plan should use vector index for nested field with dots. Plan: {}",
        plan_v1
    );

    let search_results_v1 = dataset
        .scan()
        .nearest(nested_column_path_v1, &query_vector, 5)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    assert_eq!(search_results_v1.num_rows(), 5);

    // Check the query plan for the second vector field
    let plan_v2 = dataset
        .scan()
        .nearest(nested_column_path_v2, &query_vector, 5)
        .unwrap()
        .explain_plan(false)
        .await
        .unwrap();

    // Verify the vector index is being used
    assert!(
        plan_v2.contains("ANNSubIndex") || plan_v2.contains("ANNIvfPartition"),
        "Query plan should use vector index for second nested field with dots. Plan: {}",
        plan_v2
    );

    let search_results_v2 = dataset
        .scan()
        .nearest(nested_column_path_v2, &query_vector, 5)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    assert_eq!(search_results_v2.num_rows(), 5);
}

#[tokio::test]
async fn test_vector_index_on_simple_nested_field() {
    // This test reproduces the Python test scenario from test_nested_field_vector_index
    // where the nested field path is simple (data.embedding) without dots in field names
    let dimensions = 16;
    let num_rows = 256;

    // Create schema with simple nested field (no dots in field names)
    let struct_field = Field::new(
        "data",
        DataType::Struct(
            vec![
                Field::new(
                    "embedding",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        dimensions,
                    ),
                    false,
                ),
                Field::new("label", DataType::Utf8, false),
            ]
            .into(),
        ),
        false,
    );

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        struct_field,
    ]));

    // Generate test data
    let float_arr = generate_random_array(num_rows * dimensions as usize);
    let vectors = FixedSizeListArray::try_new_from_values(float_arr, dimensions).unwrap();

    let ids = Int32Array::from_iter_values(0..num_rows as i32);
    let labels = StringArray::from_iter_values((0..num_rows).map(|i| format!("label_{}", i)));

    let struct_array = arrow_array::StructArray::from(vec![
        (
            Arc::new(Field::new(
                "embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dimensions,
                ),
                false,
            )),
            Arc::new(vectors) as Arc<dyn arrow_array::Array>,
        ),
        (
            Arc::new(Field::new("label", DataType::Utf8, false)),
            Arc::new(labels) as Arc<dyn arrow_array::Array>,
        ),
    ]);

    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(struct_array)]).unwrap();

    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;
    let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

    // Test creating index on nested field
    let nested_column_path = "data.embedding";
    let params = VectorIndexParams::ivf_pq(2, 8, 2, MetricType::L2, 10);

    dataset
        .create_index(
            &[nested_column_path],
            IndexType::Vector,
            Some("vec_idx".to_string()),
            &params,
            true,
        )
        .await
        .unwrap();

    // Verify index was created
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    assert_eq!(indices[0].name, "vec_idx");

    // Verify the correct field was indexed
    let field_id = indices[0].fields[0];
    let field_path = dataset.schema().field_path(field_id).unwrap();
    assert_eq!(field_path, "data.embedding");

    // Test querying with the index
    let query_vector = generate_random_array(dimensions as usize);

    let plan = dataset
        .scan()
        .nearest(nested_column_path, &query_vector, 5)
        .unwrap()
        .explain_plan(false)
        .await
        .unwrap();

    // Verify the vector index is being used
    assert!(
        plan.contains("ANNSubIndex") || plan.contains("ANNIvfPartition"),
        "Query plan should use vector index for nested field. Plan: {}",
        plan
    );

    let search_results = dataset
        .scan()
        .nearest(nested_column_path, &query_vector, 5)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    assert_eq!(search_results.num_rows(), 5);
}

#[tokio::test]
async fn test_btree_index_on_nested_field_with_dots() {
    // Test creating BTree index on nested field with dots in the name
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;

    // Create schema with nested field containing dots
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "data",
            DataType::Struct(
                vec![
                    Field::new("value.v1", DataType::Int32, false),
                    Field::new("value.v2", DataType::Float32, false),
                    Field::new("text", DataType::Utf8, false),
                ]
                .into(),
            ),
            false,
        ),
    ]));

    // Generate test data
    let num_rows = 1000;
    let ids = Int32Array::from_iter_values(0..num_rows);
    let values_v1 = Int32Array::from_iter_values((0..num_rows).map(|i| i % 100));
    let values_v2 = Float32Array::from_iter_values((0..num_rows).map(|i| (i as f32) * 0.1));
    let texts = StringArray::from_iter_values((0..num_rows).map(|i| format!("text_{}", i)));

    let struct_array = arrow_array::StructArray::from(vec![
        (
            Arc::new(Field::new("value.v1", DataType::Int32, false)),
            Arc::new(values_v1) as Arc<dyn arrow_array::Array>,
        ),
        (
            Arc::new(Field::new("value.v2", DataType::Float32, false)),
            Arc::new(values_v2) as Arc<dyn arrow_array::Array>,
        ),
        (
            Arc::new(Field::new("text", DataType::Utf8, false)),
            Arc::new(texts) as Arc<dyn arrow_array::Array>,
        ),
    ]);

    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(struct_array)]).unwrap();

    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
    let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

    // Create BTree index on nested field with dots
    let nested_column_path = "data.`value.v1`";
    let params = ScalarIndexParams::default();

    dataset
        .create_index(
            &[nested_column_path],
            IndexType::BTree,
            Some("btree_v1_idx".to_string()),
            &params,
            true,
        )
        .await
        .unwrap();

    // Reload dataset to ensure index is loaded
    dataset = Dataset::open(test_uri).await.unwrap();

    // Verify index was created
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    assert_eq!(indices[0].name, "btree_v1_idx");

    // Verify the correct field was indexed
    let field_id = indices[0].fields[0];
    let field_path = dataset.schema().field_path(field_id).unwrap();
    assert_eq!(field_path, "data.`value.v1`");

    // Test querying with the index and verify it's being used
    let plan = dataset
        .scan()
        .filter("data.`value.v1` = 42")
        .unwrap()
        .prefilter(true)
        .explain_plan(false)
        .await
        .unwrap();

    // Verify the query plan (scalar indices on nested fields may use optimized filters)
    // The index may be used internally even if not shown as ScalarIndexQuery
    assert!(
        plan.contains("ScalarIndexQuery"),
        "Query plan should show optimized read. Plan: {}",
        plan
    );

    // Also test that the query returns results
    let results = dataset
        .scan()
        .filter("data.`value.v1` = 42")
        .unwrap()
        .prefilter(true)
        .try_into_batch()
        .await
        .unwrap();

    assert!(results.num_rows() > 0);
}

#[tokio::test]
async fn test_bitmap_index_on_nested_field_with_dots() {
    // Test creating Bitmap index on nested field with dots in the name
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;

    // Create schema with nested field containing dots - using low cardinality for bitmap
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "metadata",
            DataType::Struct(
                vec![
                    Field::new("status.code", DataType::Int32, false),
                    Field::new("category.name", DataType::Utf8, false),
                ]
                .into(),
            ),
            false,
        ),
    ]));

    // Generate test data with low cardinality (good for bitmap index)
    let num_rows = 1000;
    let ids = Int32Array::from_iter_values(0..num_rows);
    // Only 10 unique status codes
    let status_codes = Int32Array::from_iter_values((0..num_rows).map(|i| i % 10));
    // Only 5 unique categories
    let categories =
        StringArray::from_iter_values((0..num_rows).map(|i| format!("category_{}", i % 5)));

    let struct_array = arrow_array::StructArray::from(vec![
        (
            Arc::new(Field::new("status.code", DataType::Int32, false)),
            Arc::new(status_codes) as Arc<dyn arrow_array::Array>,
        ),
        (
            Arc::new(Field::new("category.name", DataType::Utf8, false)),
            Arc::new(categories) as Arc<dyn arrow_array::Array>,
        ),
    ]);

    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(struct_array)]).unwrap();

    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
    let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

    // Create Bitmap index on nested field with dots
    let nested_column_path = "metadata.`status.code`";
    let params = ScalarIndexParams::default();

    dataset
        .create_index(
            &[nested_column_path],
            IndexType::Bitmap,
            Some("bitmap_status_idx".to_string()),
            &params,
            true,
        )
        .await
        .unwrap();

    // Reload dataset to ensure index is loaded
    dataset = Dataset::open(test_uri).await.unwrap();

    // Verify index was created
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    assert_eq!(indices[0].name, "bitmap_status_idx");

    // Verify the correct field was indexed
    let field_id = indices[0].fields[0];
    let field_path = dataset.schema().field_path(field_id).unwrap();
    assert_eq!(field_path, "metadata.`status.code`");

    // Test querying with the index and verify it's being used
    let plan = dataset
        .scan()
        .filter("metadata.`status.code` = 5")
        .unwrap()
        .explain_plan(false)
        .await
        .unwrap();

    // Verify the query plan (scalar indices on nested fields may use optimized filters)
    // The index may be used internally even if not shown as ScalarIndexQuery
    assert!(
        plan.contains("ScalarIndexQuery"),
        "Query plan should show optimized read. Plan: {}",
        plan
    );

    // Also test that the query returns results
    let results = dataset
        .scan()
        .filter("metadata.`status.code` = 5")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    // Should have ~100 rows with status code 5
    assert!(results.num_rows() > 0);
    assert_eq!(results.num_rows(), 100);
}

#[tokio::test]
async fn test_inverted_index_on_nested_field_with_dots() {
    use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;

    // Test creating Inverted index on nested text field with dots in the name
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;

    // Create schema with nested text field containing dots
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "document",
            DataType::Struct(
                vec![
                    Field::new("content.text", DataType::Utf8, false),
                    Field::new("content.summary", DataType::Utf8, false),
                ]
                .into(),
            ),
            false,
        ),
    ]));

    // Generate test data with text content
    let num_rows = 100;
    let ids = Int32Array::from_iter_values(0..num_rows as i32);
    let content_texts = StringArray::from_iter_values((0..num_rows).map(|i| match i % 3 {
        0 => format!("The quick brown fox jumps over the lazy dog {}", i),
        1 => format!(
            "Machine learning and artificial intelligence document {}",
            i
        ),
        _ => format!("Data science and analytics content piece {}", i),
    }));
    let summaries =
        StringArray::from_iter_values((0..num_rows).map(|i| format!("Summary of document {}", i)));

    let struct_array = arrow_array::StructArray::from(vec![
        (
            Arc::new(Field::new("content.text", DataType::Utf8, false)),
            Arc::new(content_texts) as Arc<dyn arrow_array::Array>,
        ),
        (
            Arc::new(Field::new("content.summary", DataType::Utf8, false)),
            Arc::new(summaries) as Arc<dyn arrow_array::Array>,
        ),
    ]);

    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(struct_array)]).unwrap();

    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
    let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

    // Create Inverted index on nested text field with dots
    let nested_column_path = "document.`content.text`";
    let params = InvertedIndexParams::default();

    dataset
        .create_index(
            &[nested_column_path],
            IndexType::Inverted,
            Some("inverted_content_idx".to_string()),
            &params,
            true,
        )
        .await
        .unwrap();

    // Reload the dataset to ensure the index is loaded
    dataset = Dataset::open(test_uri).await.unwrap();

    // Verify index was created
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    assert_eq!(indices[0].name, "inverted_content_idx");

    // Verify the correct field was indexed
    let field_id = indices[0].fields[0];
    let field_path = dataset.schema().field_path(field_id).unwrap();
    assert_eq!(field_path, "document.`content.text`");

    // Test full-text search on the nested field with dots
    // Use the field_path that the index reports
    let query = FullTextSearchQuery::new("machine learning".to_string())
        .with_column(field_path.clone())
        .unwrap();

    // Check the query plan uses the inverted index
    let plan = dataset
        .scan()
        .full_text_search(query.clone())
        .unwrap()
        .explain_plan(false)
        .await
        .unwrap();

    // Verify the inverted index is being used
    assert!(
        plan.contains("MatchQuery") || plan.contains("PhraseQuery"),
        "Query plan should use inverted index for nested field with dots. Plan: {}",
        plan
    );

    let results = dataset
        .scan()
        .full_text_search(query)
        .unwrap()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    // Verify we get results from the full-text search
    assert!(
        !results.is_empty(),
        "Full-text search should return results"
    );

    // Check that we found documents containing "machine learning"
    let mut found_count = 0;
    for batch in results {
        found_count += batch.num_rows();
    }
    // We expect to find approximately 1/3 of documents (those with i % 3 == 1)
    assert!(
        found_count > 0,
        "Should find at least some documents with 'machine learning'"
    );
    assert!(found_count < num_rows, "Should not match all documents");
}
