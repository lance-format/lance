// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::support::*;
use super::*;

#[tokio::test]
async fn test_drop_index() {
    let test_dir = TempStrDir::default();
    let schema = Schema::new(vec![
        sample_vector_field(),
        Field::new("ints", DataType::Int32, false),
    ]);
    let mut dataset = lance_datagen::rand(&schema)
        .into_dataset(
            &test_dir,
            FragmentCount::from(1),
            FragmentRowCount::from(256),
        )
        .await
        .unwrap();

    let idx_name = "name".to_string();
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some(idx_name.clone()),
            &VectorIndexParams::ivf_pq(2, 8, 2, MetricType::L2, 10),
            true,
        )
        .await
        .unwrap();
    dataset
        .create_index(
            &["ints"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();

    assert_eq!(dataset.load_indices().await.unwrap().len(), 2);

    dataset.drop_index(&idx_name).await.unwrap();

    assert_eq!(dataset.load_indices().await.unwrap().len(), 1);

    // Even though we didn't give the scalar index a name it still has an auto-generated one we can use
    let scalar_idx_name = &dataset.load_indices().await.unwrap()[0].name;
    dataset.drop_index(scalar_idx_name).await.unwrap();

    assert_eq!(dataset.load_indices().await.unwrap().len(), 0);

    // Make sure it returns an error if the index doesn't exist
    assert!(dataset.drop_index(scalar_idx_name).await.is_err());
}

#[tokio::test]
async fn test_count_index_rows() {
    let test_dir = TempStrDir::default();
    let dimensions = 16;
    let column_name = "vec";
    let field = sample_vector_field();
    let schema = Arc::new(Schema::new(vec![field]));

    let float_arr = generate_random_array(512 * dimensions as usize);

    let vectors =
        arrow_array::FixedSizeListArray::try_new_from_values(float_arr, dimensions).unwrap();

    let record_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();

    let reader = RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema.clone());
    let test_uri = &test_dir;
    let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();
    dataset.validate().await.unwrap();

    // Make sure it returns None if there's no index with the passed identifier
    assert!(dataset.index_statistics("bad_id").await.is_err());
    // Create an index
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

    let stats: serde_json::Value =
        serde_json::from_str(&dataset.index_statistics("vec_idx").await.unwrap()).unwrap();
    assert_eq!(stats["num_unindexed_rows"], 0);
    assert_eq!(stats["num_indexed_rows"], 512);

    // Now we'll append some rows which shouldn't be indexed and see the
    // count change
    let float_arr = generate_random_array(512 * dimensions as usize);
    let vectors =
        arrow_array::FixedSizeListArray::try_new_from_values(float_arr, dimensions).unwrap();

    let record_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();

    let reader = RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema);
    dataset.append(reader, None).await.unwrap();

    let stats: serde_json::Value =
        serde_json::from_str(&dataset.index_statistics("vec_idx").await.unwrap()).unwrap();
    assert_eq!(stats["num_unindexed_rows"], 512);
    assert_eq!(stats["num_indexed_rows"], 512);
}

#[tokio::test]
async fn test_shallow_clone_with_index() {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;

    // Create a schema with both vector and scalar columns
    let dimensions = 16u32;
    // Generate test data using lance_datagen (300 rows to satisfy PQ training requirements)
    let data = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("category", array::fill_utf8("category_0".to_string()))
        .col(
            "vector",
            array::rand_vec::<Float32Type>(Dimension::from(dimensions)),
        )
        .into_reader_rows(RowCount::from(300), BatchCount::from(1));

    // Create initial dataset
    let mut dataset = Dataset::write(data, test_uri, None).await.unwrap();
    // Create vector index (IVF_PQ)
    let vector_params = VectorIndexParams::ivf_pq(2, 8, 2, MetricType::L2, 10);
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("vector_idx".to_string()),
            &vector_params,
            true,
        )
        .await
        .unwrap();

    // Create scalar index (BTree)
    dataset
        .create_index(
            &["category"],
            IndexType::BTree,
            Some("category_idx".to_string()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();

    // Verify indices were created
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 2, "Should have 2 indices");
    let index_names: HashSet<String> = indices.iter().map(|idx| idx.name.clone()).collect();
    assert!(index_names.contains("vector_idx"));
    assert!(index_names.contains("category_idx"));

    // Test scalar query on source dataset
    let scalar_results = dataset
        .scan()
        .filter("category = 'category_0'")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    let source_scalar_query_rows = scalar_results.num_rows();
    assert!(
        scalar_results.num_rows() > 0,
        "Scalar query should return results"
    );

    // Multiple shallow clone iterations test with chain cloning
    let clone_rounds = 3;
    let mut current_dataset = dataset;

    for round in 1..=clone_rounds {
        let round_clone_dir = format!("{}/clone_round_{}", test_dir, round);
        let round_cloned_uri = &round_clone_dir;
        let tag_name = format!("shallow_clone_test_{}", round);

        // Create tag for this round (use current dataset for chain cloning)
        let current_version = current_dataset.version().version;
        current_dataset
            .tags()
            .create(&tag_name, current_version)
            .await
            .unwrap();

        // Perform shallow clone for this round (chain cloning from current dataset)
        let mut round_cloned_dataset = current_dataset
            .shallow_clone(round_cloned_uri, tag_name.as_str(), None)
            .await
            .unwrap();

        // Immediately validate indices after each clone
        validate_indices_after_clone(
            &round_cloned_dataset,
            round,
            source_scalar_query_rows,
            dimensions,
        )
        .await;

        // Complete validation cycle after each clone: append data, optimize index, validate
        // Append new data to the cloned dataset
        let new_data = gen_batch()
            .col(
                "id",
                array::step_custom::<Int32Type>(300 + (round * 50) as i32, 1),
            )
            .col("category", array::fill_utf8(format!("category_{}", round)))
            .col(
                "vector",
                array::rand_vec::<Float32Type>(Dimension::from(dimensions)),
            )
            .into_reader_rows(RowCount::from(50), BatchCount::from(1));

        round_cloned_dataset = Dataset::write(
            new_data,
            round_cloned_uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Verify row count increased
        let expected_rows = 300 + round * 50;
        let total_rows = round_cloned_dataset.count_rows(None).await.unwrap();
        assert_eq!(
            total_rows, expected_rows,
            "Round {}: Should have {} rows after append",
            round, expected_rows
        );

        let indices_before_optimize = round_cloned_dataset.load_indices().await.unwrap();
        let vector_idx_before = indices_before_optimize
            .iter()
            .find(|idx| idx.name == "vector_idx")
            .unwrap();
        let category_idx_before = indices_before_optimize
            .iter()
            .find(|idx| idx.name == "category_idx")
            .unwrap();

        // Optimize indices
        round_cloned_dataset
            .optimize_indices(&OptimizeOptions::merge(indices_before_optimize.len()))
            .await
            .unwrap();

        // Verify index UUID has changed
        let optimized_indices = round_cloned_dataset.load_indices().await.unwrap();
        let new_vector_idx = optimized_indices
            .iter()
            .find(|idx| idx.name == "vector_idx")
            .unwrap();
        let new_category_idx = optimized_indices
            .iter()
            .find(|idx| idx.name == "category_idx")
            .unwrap();

        assert_ne!(
            new_vector_idx.uuid, vector_idx_before.uuid,
            "Round {}: Vector index should have a new UUID after optimization",
            round
        );
        assert_ne!(
            new_category_idx.uuid, category_idx_before.uuid,
            "Round {}: Category index should have a new UUID after optimization",
            round
        );

        // Verify the location of index files
        use std::path::PathBuf;
        let clone_indices_dir = PathBuf::from(round_cloned_uri).join("_indices");
        let vector_index_dir = clone_indices_dir.join(new_vector_idx.uuid.to_string());
        let category_index_dir = clone_indices_dir.join(new_category_idx.uuid.to_string());

        assert!(
            vector_index_dir.exists(),
            "Round {}: New vector index directory should exist in cloned dataset location: {:?}",
            round,
            vector_index_dir
        );
        assert!(
            category_index_dir.exists(),
            "Round {}: New category index directory should exist in cloned dataset location: {:?}",
            round,
            category_index_dir
        );

        // Verify base id
        assert!(
            new_vector_idx.base_id.is_none(),
            "Round {}: New vector index should not have base_id after optimization in cloned dataset",
            round
        );
        assert!(
            new_category_idx.base_id.is_none(),
            "Round {}: New category index should not have base_id after optimization in cloned dataset",
            round
        );

        // Verify the source location does NOT contain new data
        let original_indices_dir = PathBuf::from(current_dataset.uri()).join("_indices");
        let wrong_vector_dir = original_indices_dir.join(new_vector_idx.uuid.to_string());
        let wrong_category_dir = original_indices_dir.join(new_category_idx.uuid.to_string());

        assert!(
            !wrong_vector_dir.exists(),
            "Round {}: New vector index should NOT be in original dataset location: {:?}",
            round,
            wrong_vector_dir
        );
        assert!(
            !wrong_category_dir.exists(),
            "Round {}: New category index should NOT be in original dataset location: {:?}",
            round,
            wrong_category_dir
        );

        // Validate data integrity and index functionality after optimization
        let old_category_results = round_cloned_dataset
            .scan()
            .filter("category = 'category_0'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        let new_category_results = round_cloned_dataset
            .scan()
            .filter(&format!("category = 'category_{}'", round))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        assert_eq!(
            source_scalar_query_rows,
            old_category_results.num_rows(),
            "Round {}: Should find old category data with {} rows",
            round,
            source_scalar_query_rows
        );
        assert!(
            new_category_results.num_rows() > 0,
            "Round {}: Should find new category data",
            round
        );

        // Test vector search functionality
        let query_vector = generate_random_array(dimensions as usize);
        let search_results = round_cloned_dataset
            .scan()
            .nearest("vector", &query_vector, 10)
            .unwrap()
            .limit(Some(10), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        assert!(
            search_results.num_rows() > 0,
            "Round {}: Vector search should return results after optimization",
            round
        );

        // Verify statistic of indexes
        let vector_stats: serde_json::Value = serde_json::from_str(
            &round_cloned_dataset
                .index_statistics("vector_idx")
                .await
                .unwrap(),
        )
        .unwrap();
        let category_stats: serde_json::Value = serde_json::from_str(
            &round_cloned_dataset
                .index_statistics("category_idx")
                .await
                .unwrap(),
        )
        .unwrap();

        assert_eq!(
            vector_stats["num_indexed_rows"].as_u64().unwrap(),
            expected_rows as u64,
            "Round {}: Vector index should have {} indexed rows",
            round,
            expected_rows
        );
        assert_eq!(
            category_stats["num_indexed_rows"].as_u64().unwrap(),
            expected_rows as u64,
            "Round {}: Category index should have {} indexed rows",
            round,
            expected_rows
        );

        // Prepare for next round: use the cloned dataset as the source for next clone
        current_dataset = round_cloned_dataset;
    }

    // Use the final cloned dataset for any remaining tests
    let final_cloned_dataset = current_dataset;

    // Verify cloned dataset has indices
    let cloned_indices = final_cloned_dataset.load_indices().await.unwrap();
    assert_eq!(
        cloned_indices.len(),
        2,
        "Final cloned dataset should have 2 indices"
    );
    let cloned_index_names: HashSet<String> =
        cloned_indices.iter().map(|idx| idx.name.clone()).collect();
    assert!(cloned_index_names.contains("vector_idx"));
    assert!(cloned_index_names.contains("category_idx"));

    // Test vector search on final cloned dataset
    let query_vector = generate_random_array(dimensions as usize);
    let search_results = final_cloned_dataset
        .scan()
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .limit(Some(5), None)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    assert!(
        search_results.num_rows() > 0,
        "Vector search should return results on final dataset"
    );

    // Test scalar query on final cloned dataset
    let scalar_results = final_cloned_dataset
        .scan()
        .filter("category = 'category_0'")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    assert_eq!(
        source_scalar_query_rows,
        scalar_results.num_rows(),
        "Scalar query should return results on final dataset"
    );
}

#[tokio::test]
async fn test_initialize_indices() {
    use crate::dataset::Dataset;
    use arrow_array::types::Float32Type;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datagen::{BatchCount, RowCount, array};
    use lance_index::scalar::{InvertedIndexParams, ScalarIndexParams};
    use lance_linalg::distance::MetricType;
    use std::collections::HashSet;

    // Create source dataset with various index types
    let test_dir = TempStrDir::default();
    let source_uri = format!("{}/{}", test_dir, "source");
    let target_uri = format!("{}/{}", test_dir, "target");

    // Generate test data using lance_datagen (need at least 256 rows for PQ training)
    let source_reader = lance_datagen::gen_batch()
        .col("vector", array::rand_vec::<Float32Type>(8.into()))
        .col(
            "text",
            array::cycle_utf8_literals(&["hello world", "foo bar", "test data"]),
        )
        .col("id", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(300), BatchCount::from(1));

    // Create source dataset
    let mut source_dataset = Dataset::write(source_reader, &source_uri, None)
        .await
        .unwrap();

    // Create indices on source dataset
    // 1. Vector index
    let vector_params = VectorIndexParams::ivf_pq(4, 8, 2, MetricType::L2, 10);
    source_dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("vec_idx".to_string()),
            &vector_params,
            false,
        )
        .await
        .unwrap();

    // 2. FTS index
    let fts_params = InvertedIndexParams::default();
    source_dataset
        .create_index(
            &["text"],
            IndexType::Inverted,
            Some("text_idx".to_string()),
            &fts_params,
            false,
        )
        .await
        .unwrap();

    // 3. Scalar index
    let scalar_params = ScalarIndexParams::default();
    source_dataset
        .create_index(
            &["id"],
            IndexType::BTree,
            Some("id_idx".to_string()),
            &scalar_params,
            false,
        )
        .await
        .unwrap();

    // Reload source dataset to get updated index metadata
    let source_dataset = Dataset::open(&source_uri).await.unwrap();

    // Verify source has 3 indices
    let source_indices = source_dataset.load_indices().await.unwrap();
    assert_eq!(
        source_indices.len(),
        3,
        "Source dataset should have 3 indices"
    );

    // Create target dataset with same schema but different data (need at least 256 rows for PQ)
    let target_reader = lance_datagen::gen_batch()
        .col("vector", array::rand_vec::<Float32Type>(8.into()))
        .col(
            "text",
            array::cycle_utf8_literals(&["foo bar", "test data", "hello world"]),
        )
        .col("id", array::step_custom::<Int32Type>(100, 1))
        .into_reader_rows(RowCount::from(300), BatchCount::from(1));
    let mut target_dataset = Dataset::write(target_reader, &target_uri, None)
        .await
        .unwrap();

    // Initialize indices from source dataset
    target_dataset
        .initialize_indices(&source_dataset)
        .await
        .unwrap();

    // Verify target has same indices
    let target_indices = target_dataset.load_indices().await.unwrap();
    assert_eq!(
        target_indices.len(),
        3,
        "Target dataset should have 3 indices after initialization"
    );

    // Check index names match
    let source_names: HashSet<String> = source_indices.iter().map(|idx| idx.name.clone()).collect();
    let target_names: HashSet<String> = target_indices.iter().map(|idx| idx.name.clone()).collect();
    assert_eq!(
        source_names, target_names,
        "Index names should match between source and target"
    );

    // Verify indices are functional by running queries
    // 1. Test vector index
    let query_vector = generate_random_array(8);
    let search_results = target_dataset
        .scan()
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .limit(Some(5), None)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert!(
        search_results.num_rows() > 0,
        "Vector index should be functional"
    );

    // 2. Test scalar index
    let scalar_results = target_dataset
        .scan()
        .filter("id = 125")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(
        scalar_results.num_rows(),
        1,
        "Scalar index should find exact match"
    );
}

#[tokio::test]
async fn test_initialize_indices_with_missing_field() {
    use crate::dataset::Dataset;
    use arrow_array::types::Int32Type;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datagen::{BatchCount, RowCount, array};
    use lance_index::scalar::ScalarIndexParams;

    // Test that initialize_indices handles missing fields gracefully
    let test_dir = TempStrDir::default();
    let source_uri = format!("{}/{}", test_dir, "source");
    let target_uri = format!("{}/{}", test_dir, "target");

    // Create source dataset with extra field
    let source_reader = lance_datagen::gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("extra", array::cycle_utf8_literals(&["test"]))
        .into_reader_rows(RowCount::from(10), BatchCount::from(1));
    let mut source_dataset = Dataset::write(source_reader, &source_uri, None)
        .await
        .unwrap();

    // Create index on extra field in source
    source_dataset
        .create_index(
            &["extra"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();

    // Create target dataset without extra field
    let target_reader = lance_datagen::gen_batch()
        .col("id", array::step_custom::<Int32Type>(10, 1))
        .into_reader_rows(RowCount::from(10), BatchCount::from(1));
    let mut target_dataset = Dataset::write(target_reader, &target_uri, None)
        .await
        .unwrap();

    // Initialize indices should skip the index on missing field with an error
    let result = target_dataset.initialize_indices(&source_dataset).await;

    // Should fail when field is missing
    assert!(result.is_err(), "Should error when field is missing");
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("not found in target dataset")
    );
}

/// `initialize_index` must depend only on the keyed prefix of the source
/// index. The rebuild it drives writes fresh, non-covered metadata and never
/// reads the carried columns, so requiring them to exist and type-match in
/// the target rejects a target that can hold the index perfectly well.
#[tokio::test]
async fn test_initialize_index_ignores_carried_columns() {
    use crate::dataset::Dataset;
    use lance_index::scalar::ScalarIndexParams;

    let source_data = gen_batch()
        .col("a", array::step::<Int32Type>())
        .col("carried", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(20), BatchCount::from(1));
    let mut source = Dataset::write(source_data, "memory://source", None)
        .await
        .unwrap();
    source
        .create_index(
            &["a"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();

    // No creation API writes a covering declaration yet, so commit one.
    crate::utils::test::covering::declare_covering(&mut source, "a", "carried").await;
    let index_name = source.load_indices().await.unwrap()[0].name.clone();

    // The target holds the keyed column only -- exactly what the rebuild
    // reads.
    let target_data = gen_batch()
        .col("a", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(20), BatchCount::from(1));
    let mut target = Dataset::write(target_data, "memory://target", None)
        .await
        .unwrap();

    target.initialize_index(&source, &index_name).await.unwrap();

    let initialized = target.load_indices().await.unwrap();
    assert_eq!(initialized.len(), 1);
    assert_eq!(
        initialized[0].fields,
        vec![target.schema().field("a").unwrap().id]
    );
    assert!(
        initialized[0].covering_fields.is_empty(),
        "the rebuild carries nothing, so it must not claim to"
    );
}

#[tokio::test]
async fn test_initialize_single_index() {
    use crate::dataset::Dataset;
    use crate::index::vector::VectorIndexParams;
    use arrow_array::types::{Float32Type, Int32Type};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datagen::{BatchCount, RowCount, array};
    use lance_index::scalar::ScalarIndexParams;
    use lance_linalg::distance::MetricType;

    let test_dir = TempStrDir::default();
    let source_uri = format!("{}/{}", test_dir, "source");
    let target_uri = format!("{}/{}", test_dir, "target");

    // Create source dataset (need at least 256 rows for PQ training)
    let source_reader = lance_datagen::gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("name", array::rand_utf8(4.into(), false))
        .col("vector", array::rand_vec::<Float32Type>(8.into()))
        .into_reader_rows(RowCount::from(300), BatchCount::from(1));
    let mut source_dataset = Dataset::write(source_reader, &source_uri, None)
        .await
        .unwrap();

    // Create multiple indices on source
    let scalar_params = ScalarIndexParams::default();
    source_dataset
        .create_index(
            &["id"],
            IndexType::BTree,
            Some("id_index".to_string()),
            &scalar_params,
            false,
        )
        .await
        .unwrap();

    let vector_params = VectorIndexParams::ivf_pq(16, 8, 4, MetricType::L2, 50);
    source_dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("vector_index".to_string()),
            &vector_params,
            false,
        )
        .await
        .unwrap();

    // Reload source dataset to get updated index metadata
    let source_dataset = Dataset::open(&source_uri).await.unwrap();

    // Create target dataset with same schema
    let target_reader = lance_datagen::gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("name", array::rand_utf8(4.into(), false))
        .col("vector", array::rand_vec::<Float32Type>(8.into()))
        .into_reader_rows(RowCount::from(300), BatchCount::from(1));
    let mut target_dataset = Dataset::write(target_reader, &target_uri, None)
        .await
        .unwrap();

    // Initialize only the vector index
    target_dataset
        .initialize_index(&source_dataset, "vector_index")
        .await
        .unwrap();

    // Verify only vector index was created
    let target_indices = target_dataset.load_indices().await.unwrap();
    assert_eq!(target_indices.len(), 1, "Should have only 1 index");
    assert_eq!(
        target_indices[0].name, "vector_index",
        "Should have the vector index"
    );

    // Initialize the scalar index
    target_dataset
        .initialize_index(&source_dataset, "id_index")
        .await
        .unwrap();

    // Verify both indices now exist
    let target_indices = target_dataset.load_indices().await.unwrap();
    assert_eq!(target_indices.len(), 2, "Should have 2 indices");

    let index_names: HashSet<String> = target_indices.iter().map(|idx| idx.name.clone()).collect();
    assert!(
        index_names.contains("vector_index"),
        "Should have vector index"
    );
    assert!(index_names.contains("id_index"), "Should have id index");

    // Test error case - non-existent index
    let result = target_dataset
        .initialize_index(&source_dataset, "non_existent")
        .await;
    assert!(result.is_err(), "Should error for non-existent index");
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("not found in source dataset")
    );
}

#[rstest::rstest]
#[case::retained_row_ids(false)]
#[case::retained_row_addresses(true)]
#[tokio::test]
async fn test_staged_btree_row_domains_reject_partial_migration(
    #[case] retain_row_addresses: bool,
) {
    let dir = copy_test_data_to_tmp("v14.0.0-beta.15/stable_row_id_btree.lance").unwrap();
    let mut dataset = Dataset::open(&dir.path_str()).await.unwrap();
    assert!(dataset.manifest.uses_stable_row_ids());
    let legacy = dataset.load_indices_by_name("value_idx").await.unwrap();
    assert_eq!(legacy.len(), 2);
    assert!(
        legacy
            .iter()
            .all(|segment| !segment.results_are_row_addrs())
    );
    let params = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::BTree);
    let mut current = Vec::with_capacity(2);
    for fragment_id in [0, 1] {
        current.push(
            dataset
                .create_index_builder(&["value"], IndexType::BTree, &params)
                .name("value_idx".into())
                .replace(true)
                .fragments(vec![fragment_id])
                .execute_uncommitted()
                .await
                .unwrap(),
        );
    }
    assert!(
        current
            .iter()
            .all(|segment| segment.results_are_row_addrs())
    );
    if retain_row_addresses {
        dataset
            .commit_existing_index_segments("value_idx", "value", current.clone())
            .await
            .unwrap();
    }
    dataset = Dataset::open(&dir.path_str()).await.unwrap();
    let before_version = dataset.version_id();
    let before = dataset.load_indices_by_name("value_idx").await.unwrap();
    let incoming = if retain_row_addresses {
        legacy[0].clone()
    } else {
        current[0].clone()
    };
    let error = dataset
        .commit_existing_index_segments("value_idx", "value", vec![incoming])
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(error.to_string().contains("mixes row-ID and row-address"));
    let mut dataset = Dataset::open(&dir.path_str()).await.unwrap();
    assert_eq!(dataset.version_id(), before_version);
    assert_eq!(
        dataset.load_indices_by_name("value_idx").await.unwrap(),
        before
    );
    for (filter, expected) in [("value = 1", 1), ("value = 9", 1), ("value IS NULL", 2)] {
        for indexed in [false, true] {
            let mut scan = dataset.scan();
            scan.filter(filter).unwrap().use_scalar_index(indexed);
            assert_eq!(scan.try_into_batch().await.unwrap().num_rows(), expected);
        }
    }
    // A complete migration remains valid: no legacy sibling survives it.
    dataset
        .commit_existing_index_segments("value_idx", "value", current)
        .await
        .unwrap();
    assert!(
        dataset
            .load_indices_by_name("value_idx")
            .await
            .unwrap()
            .iter()
            .all(|segment| segment.results_are_row_addrs())
    );
    assert_eq!(
        dataset.count_rows(Some("value = 9".into())).await.unwrap(),
        1
    );
}

#[rstest::rstest]
#[case::legacy_first(false, false)]
#[case::address_first(true, false)]
#[case::transaction_boundary(false, true)]
#[tokio::test]
async fn test_staged_btree_row_domains_reject_mixed_incoming(
    #[case] reverse: bool,
    #[case] direct_transaction: bool,
) {
    let dir = copy_test_data_to_tmp("v14.0.0-beta.15/stable_row_id_btree.lance").unwrap();
    let mut dataset = Dataset::open(&dir.path_str()).await.unwrap();
    let legacy = dataset.load_indices_by_name("value_idx").await.unwrap();
    let first = legacy
        .iter()
        .find(|segment| segment.fragment_bitmap.as_ref().unwrap().contains(0))
        .unwrap()
        .clone();
    let params = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::BTree);
    let current = dataset
        .create_index_builder(&["value"], IndexType::BTree, &params)
        .name("value_idx".into())
        .replace(true)
        .fragments(vec![1])
        .execute_uncommitted()
        .await
        .unwrap();
    let mut incoming = vec![first, current];
    if reverse {
        incoming.reverse();
    }
    let version = dataset.version_id();
    let error = if direct_transaction {
        dataset
            .apply_commit(
                Transaction::new(
                    version,
                    Operation::CreateIndex {
                        new_indices: incoming,
                        removed_indices: legacy.clone(),
                    },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap_err()
    } else {
        dataset
            .commit_existing_index_segments("value_idx", "value", incoming)
            .await
            .unwrap_err()
    };
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(error.to_string().contains("mixes row-ID and row-address"));
    let dataset = Dataset::open(&dir.path_str()).await.unwrap();
    assert_eq!(dataset.version_id(), version);
    assert_eq!(
        dataset.load_indices_by_name("value_idx").await.unwrap(),
        legacy
    );
    assert_eq!(
        dataset.count_rows(Some("value = 9".into())).await.unwrap(),
        1
    );
}

#[tokio::test]
async fn test_commit_existing_index_segments_commits_multiple_segments() {
    use lance_datagen::{BatchCount, RowCount, array};

    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();

    let reader = lance_datagen::gen_batch()
        .col("id", array::step::<arrow_array::types::Int32Type>())
        .col(
            "vector",
            array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
        )
        .into_reader_rows(RowCount::from(10), BatchCount::from(2));

    let mut dataset = Dataset::write(
        reader,
        test_uri,
        Some(WriteParams {
            max_rows_per_file: 10,
            max_rows_per_group: 10,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    // Commit validation opens coexisting vector segments, so this fixture must
    // contain real index files rather than placeholder metadata payloads.
    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), 2);
    let params = VectorIndexParams::ivf_flat(1, MetricType::L2);
    let mut segments = Vec::with_capacity(fragments.len());
    for fragment in &fragments {
        segments.push(
            dataset
                .create_index_builder(&["vector"], IndexType::Vector, &params)
                .name("vector_idx".to_string())
                .fragments(vec![fragment.id() as u32])
                .execute_uncommitted()
                .await
                .unwrap(),
        );
    }
    let expected_uuids = segments
        .iter()
        .map(|segment| segment.uuid)
        .collect::<HashSet<_>>();
    dataset
        .commit_existing_index_segments("vector_idx", "vector", segments)
        .await
        .unwrap();

    let committed = dataset.load_indices_by_name("vector_idx").await.unwrap();
    assert_eq!(committed.len(), 2);
    let committed_uuids = committed.iter().map(|idx| idx.uuid).collect::<HashSet<_>>();
    assert_eq!(
        committed_uuids, expected_uuids,
        "all committed segment uuids should be preserved"
    );
    assert_eq!(
        committed
            .iter()
            .map(|idx| idx
                .fragment_bitmap
                .as_ref()
                .unwrap()
                .iter()
                .collect::<Vec<_>>())
            .collect::<HashSet<_>>(),
        HashSet::from([vec![0], vec![1]]),
        "each committed segment should preserve its fragment coverage"
    );
    assert!(
        committed
            .iter()
            .all(|idx| idx.files.as_ref().is_some_and(|files| !files.is_empty())),
        "committed segment metadata should capture on-disk file info"
    );
}

#[tokio::test]
async fn test_commit_existing_index_segments_rejects_duplicate_segment_ids() {
    use lance_datagen::{BatchCount, RowCount, array};

    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();

    let reader = lance_datagen::gen_batch()
        .col("id", array::step::<arrow_array::types::Int32Type>())
        .col(
            "vector",
            array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
        )
        .into_reader_rows(RowCount::from(10), BatchCount::from(1));

    let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

    let field_id = dataset.schema().field("vector").unwrap().id;
    let base = write_vector_segment_metadata(
        &dataset,
        "vector_idx",
        field_id,
        Uuid::new_v4(),
        [0_u32],
        b"base",
    )
    .await;

    let err = dataset
        .commit_existing_index_segments(
            "vector_idx",
            "vector",
            vec![
                segment_from_metadata(&base),
                segment_from_metadata(&IndexMetadata {
                    fragment_bitmap: Some(std::iter::once(1_u32).collect()),
                    ..base
                }),
            ],
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("duplicate segment uuid"));
}

#[tokio::test]
async fn test_commit_existing_index_segments_rejects_empty_segments() {
    use lance_datagen::{BatchCount, RowCount, array};

    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();

    let reader = lance_datagen::gen_batch()
        .col("id", array::step::<arrow_array::types::Int32Type>())
        .col(
            "vector",
            array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
        )
        .into_reader_rows(RowCount::from(10), BatchCount::from(1));

    let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

    let err = dataset
        .commit_existing_index_segments("vector_idx", "vector", Vec::<IndexSegment>::new())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("at least one index segment"));
}

/// A segment may carry extra columns; `build_index_metadata_from_segments`
/// must still commit it, keyed on the field the index is being built for.
/// Calls the metadata builder directly to isolate field validation from
/// commit conflict resolution and manifest writing.
#[tokio::test]
async fn test_build_index_metadata_from_segments_accepts_carried_fields() {
    use lance_datagen::{BatchCount, RowCount, array};

    let test_dir = tempfile::tempdir().unwrap();
    let reader = lance_datagen::gen_batch()
        .col("id", array::step::<arrow_array::types::Int32Type>())
        .col(
            "vector",
            array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
        )
        .into_reader_rows(RowCount::from(10), BatchCount::from(1));
    let dataset = Dataset::write(reader, test_dir.path().to_str().unwrap(), None)
        .await
        .unwrap();

    let vector_field_id = dataset.schema().field("vector").unwrap().id;
    let id_field_id = dataset.schema().field("id").unwrap().id;
    let mut metadata = write_vector_segment_metadata(
        &dataset,
        "vector_idx",
        vector_field_id,
        Uuid::new_v4(),
        [0_u32],
        b"segment",
    )
    .await;
    // Carry `id` alongside the keyed `vector` field.
    metadata.fields = vec![vector_field_id, id_field_id];
    metadata.covering_fields = vec![id_field_id];

    let new_indices = build_index_metadata_from_segments(
        &dataset,
        "vector_idx",
        vector_field_id,
        vec![segment_from_metadata(&metadata)],
    )
    .await
    .unwrap();

    assert_eq!(new_indices.len(), 1);
    assert_eq!(new_indices[0].fields, vec![vector_field_id, id_field_id]);
    assert_eq!(new_indices[0].covering_fields, vec![id_field_id]);
}

/// The per-segment rules only pin the keyed field and its count, so two
/// segments can disagree about the carried columns and each still pass. That
/// pair must be refused at commit, because `IndexDescriptionImpl::try_new`
/// requires `fields` to be identical across segments -- committing it would
/// leave `describe_indices` erroring on metadata this call just wrote.
#[tokio::test]
async fn test_build_index_metadata_from_segments_rejects_mixed_covering_declarations() {
    use lance_datagen::{BatchCount, RowCount, array};

    let test_dir = tempfile::tempdir().unwrap();
    let reader = lance_datagen::gen_batch()
        .col("id", array::step::<arrow_array::types::Int32Type>())
        .col(
            "vector",
            array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
        )
        .into_reader_rows(RowCount::from(10), BatchCount::from(1));
    let dataset = Dataset::write(reader, test_dir.path().to_str().unwrap(), None)
        .await
        .unwrap();

    let vector_field_id = dataset.schema().field("vector").unwrap().id;
    let id_field_id = dataset.schema().field("id").unwrap().id;

    let mut covered = write_vector_segment_metadata(
        &dataset,
        "vector_idx",
        vector_field_id,
        Uuid::new_v4(),
        [0_u32],
        b"covered",
    )
    .await;
    covered.fields = vec![vector_field_id, id_field_id];
    covered.covering_fields = vec![id_field_id];

    // Plain: one keyed field, nothing carried. Passes every per-segment rule
    // on its own -- `keyed == 1` and `fields[0]` is the keyed field -- and so
    // does `covered`; only comparing them catches the disagreement.
    let plain = write_vector_segment_metadata(
        &dataset,
        "vector_idx",
        vector_field_id,
        Uuid::new_v4(),
        [1_u32],
        b"plain",
    )
    .await;
    assert_eq!(plain.fields, vec![vector_field_id]);
    assert!(plain.covering_fields.is_empty());

    let err = build_index_metadata_from_segments(
        &dataset,
        "vector_idx",
        vector_field_id,
        vec![
            segment_from_metadata(&covered),
            segment_from_metadata(&plain),
        ],
    )
    .await
    .expect_err("segments disagreeing about carried columns must not commit");
    assert!(
        err.to_string().contains("must declare the same columns"),
        "unexpected message: {err}"
    );
}

/// A segment keyed on the wrong field must still be rejected. Calls
/// `build_index_metadata_from_segments` directly so the assertion is
/// evidence for *this* guard specifically -- going through
/// `commit_existing_index_segments` would let the untouched-looking
/// duplicate check there mask this guard's removal.
#[tokio::test]
async fn test_build_index_metadata_from_segments_rejects_wrong_field_provenance() {
    use lance_datagen::{BatchCount, RowCount, array};

    let test_dir = tempfile::tempdir().unwrap();
    let reader = lance_datagen::gen_batch()
        .col("id", array::step::<arrow_array::types::Int32Type>())
        .col(
            "vector",
            array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
        )
        .into_reader_rows(RowCount::from(10), BatchCount::from(1));
    let dataset = Dataset::write(reader, test_dir.path().to_str().unwrap(), None)
        .await
        .unwrap();

    let vector_field_id = dataset.schema().field("vector").unwrap().id;
    let id_field_id = dataset.schema().field("id").unwrap().id;
    let mut metadata = write_vector_segment_metadata(
        &dataset,
        "vector_idx",
        vector_field_id,
        Uuid::new_v4(),
        [0_u32],
        b"segment",
    )
    .await;
    metadata.fields = vec![id_field_id];

    let error = build_index_metadata_from_segments(
        &dataset,
        "vector_idx",
        vector_field_id,
        vec![segment_from_metadata(&metadata)],
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("expected keyed field"));
}

/// Matching cardinality is not enough: `covering_fields` must actually be
/// the trailing slice of `fields`. `fields = [vector, id]`,
/// `covering_fields = [999]` has exactly one keyed field but carries a
/// column that is not even present in `fields`, let alone its suffix.
#[tokio::test]
async fn test_build_index_metadata_from_segments_rejects_non_suffix_covering_fields() {
    use lance_datagen::{BatchCount, RowCount, array};

    let test_dir = tempfile::tempdir().unwrap();
    let reader = lance_datagen::gen_batch()
        .col("id", array::step::<arrow_array::types::Int32Type>())
        .col(
            "vector",
            array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
        )
        .into_reader_rows(RowCount::from(10), BatchCount::from(1));
    let dataset = Dataset::write(reader, test_dir.path().to_str().unwrap(), None)
        .await
        .unwrap();

    let vector_field_id = dataset.schema().field("vector").unwrap().id;
    let id_field_id = dataset.schema().field("id").unwrap().id;
    let mut metadata = write_vector_segment_metadata(
        &dataset,
        "vector_idx",
        vector_field_id,
        Uuid::new_v4(),
        [0_u32],
        b"segment",
    )
    .await;
    metadata.fields = vec![vector_field_id, id_field_id];
    metadata.covering_fields = vec![999];

    let error = build_index_metadata_from_segments(
        &dataset,
        "vector_idx",
        vector_field_id,
        vec![segment_from_metadata(&metadata)],
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("are not among its fields"));
}

#[tokio::test]
async fn test_commit_existing_index_segments_rejects_wrong_field_provenance() {
    use lance_datagen::{BatchCount, RowCount, array};

    let test_dir = tempfile::tempdir().unwrap();
    let reader = lance_datagen::gen_batch()
        .col("id", array::step::<arrow_array::types::Int32Type>())
        .col(
            "vector",
            array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
        )
        .into_reader_rows(RowCount::from(10), BatchCount::from(1));
    let mut dataset = Dataset::write(reader, test_dir.path().to_str().unwrap(), None)
        .await
        .unwrap();

    let vector_field_id = dataset.schema().field("vector").unwrap().id;
    let mut metadata = write_vector_segment_metadata(
        &dataset,
        "vector_idx",
        vector_field_id,
        Uuid::new_v4(),
        [0_u32],
        b"segment",
    )
    .await;
    metadata.fields = vec![dataset.schema().field("id").unwrap().id];

    let error = dataset
        .commit_existing_index_segments(
            "vector_idx",
            "vector",
            vec![segment_from_metadata(&metadata)],
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("expected keyed field"));
}

#[tokio::test]
async fn test_commit_existing_index_segments_rejects_future_dataset_version() {
    use lance_datagen::{BatchCount, RowCount, array};

    let test_dir = tempfile::tempdir().unwrap();
    let reader = lance_datagen::gen_batch()
        .col(
            "vector",
            array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
        )
        .into_reader_rows(RowCount::from(10), BatchCount::from(1));
    let mut dataset = Dataset::write(reader, test_dir.path().to_str().unwrap(), None)
        .await
        .unwrap();

    let field_id = dataset.schema().field("vector").unwrap().id;
    let mut metadata = write_vector_segment_metadata(
        &dataset,
        "vector_idx",
        field_id,
        Uuid::new_v4(),
        [0_u32],
        b"segment",
    )
    .await;
    metadata.dataset_version = dataset.manifest.version + 1;

    let error = dataset
        .commit_existing_index_segments(
            "vector_idx",
            "vector",
            vec![segment_from_metadata(&metadata)],
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("future dataset version"));
}

#[tokio::test]
async fn test_commit_existing_index_segments_rejects_overlapping_fragment_coverage() {
    use lance_datagen::{BatchCount, RowCount, array};

    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();

    let reader = lance_datagen::gen_batch()
        .col("id", array::step::<arrow_array::types::Int32Type>())
        .col(
            "vector",
            array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
        )
        .into_reader_rows(RowCount::from(20), BatchCount::from(2));

    let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

    let field_id = dataset.schema().field("vector").unwrap().id;
    let seg0 = write_vector_segment_metadata(
        &dataset,
        "vector_idx",
        field_id,
        Uuid::new_v4(),
        [0_u32, 1_u32],
        b"seg0",
    )
    .await;
    let seg1 = write_vector_segment_metadata(
        &dataset,
        "vector_idx",
        field_id,
        Uuid::new_v4(),
        [1_u32],
        b"seg1",
    )
    .await;

    let err = dataset
        .commit_existing_index_segments(
            "vector_idx",
            "vector",
            vec![segment_from_metadata(&seg0), segment_from_metadata(&seg1)],
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("overlapping fragment coverage"));
}

#[tokio::test]
async fn test_commit_existing_index_segments_rejects_mixed_index_detail_types() {
    use lance_datagen::{BatchCount, RowCount, array};

    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();

    let reader = lance_datagen::gen_batch()
        .col("id", array::step::<arrow_array::types::Int32Type>())
        .col(
            "vector",
            array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
        )
        .into_reader_rows(RowCount::from(20), BatchCount::from(2));

    let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

    let field_id = dataset.schema().field("vector").unwrap().id;
    let seg0 = write_vector_segment_metadata(
        &dataset,
        "vector_idx",
        field_id,
        Uuid::new_v4(),
        [0_u32],
        b"seg0",
    )
    .await;
    let seg1 = IndexMetadata {
        uuid: Uuid::new_v4(),
        name: "vector_idx".to_string(),
        fields: vec![field_id],
        covering_fields: vec![],
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(std::iter::once(1_u32).collect()),
        index_details: Some(Arc::new(
            prost_types::Any::from_msg(&BTreeIndexDetails::default()).unwrap(),
        )),
        index_version: IndexType::BTree.version(),
        created_at: Some(chrono::Utc::now()),
        base_id: None,
        files: seg0.files.clone(),
        clustering_id: None,
        metadata: Default::default(),
    };

    let err = dataset
        .commit_existing_index_segments("vector_idx", "vector", vec![seg0, seg1])
        .await
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("mixes incompatible index detail types")
    );
}

#[rstest]
#[case::analyzer(InvertedIndexParams::default())]
#[case::split_identifiers(InvertedIndexParams::code().split_identifiers(false))]
#[case::positions(InvertedIndexParams::code().split_identifiers(true).with_position(true))]
#[case::block_size(InvertedIndexParams::code().split_identifiers(true).block_size(256).unwrap())]
#[tokio::test]
async fn test_commit_existing_index_segments_validates_inverted_params(
    #[case] incompatible_params: InvertedIndexParams,
    #[values(false, true)] has_existing_index: bool,
) {
    let batch = arrow_array::record_batch!(
        ("id", Int32, [0, 1, 2]),
        (
            "code",
            Utf8,
            [
                Some("def getUserName(): shared"),
                Some("fn fetchUserProfile() { shared }"),
                None
            ]
        )
    )
    .unwrap();
    let reader = RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
    let mut dataset = Dataset::write(
        reader,
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 1,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 3);
    let params = InvertedIndexParams::code().split_identifiers(true);
    let original = dataset
        .create_index_builder(&["code"], IndexType::Inverted, &params)
        .fragments(vec![0])
        .execute_uncommitted()
        .await
        .unwrap();
    if has_existing_index {
        dataset
            .commit_existing_index_segments("cidx", "code", vec![original.clone()])
            .await
            .unwrap();
    }
    let version = dataset.manifest.version;
    let original_indices = dataset.load_indices().await.unwrap();
    let incompatible = dataset
        .create_index_builder(&["code"], IndexType::Inverted, &incompatible_params)
        .fragments(vec![1, 2])
        .execute_uncommitted()
        .await
        .unwrap();
    let segments = if has_existing_index {
        vec![incompatible]
    } else {
        vec![original.clone(), incompatible]
    };
    let error = dataset
        .commit_existing_index_segments("cidx", "code", segments)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(
        error
            .to_string()
            .contains("FTS index cidx has inconsistent inverted index details across segments"),
        "{error}"
    );
    assert_eq!(dataset.manifest.version, version);
    assert_eq!(dataset.load_indices().await.unwrap(), original_indices);
    if has_existing_index {
        let result = dataset
            .scan()
            .full_text_search(FullTextSearchQuery::new("user".to_owned()))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 2);
    }

    let compatible = dataset
        .create_index_builder(&["code"], IndexType::Inverted, &params)
        .fragments(vec![1, 2])
        .execute_uncommitted()
        .await
        .unwrap();
    let segments = if has_existing_index {
        vec![compatible]
    } else {
        vec![original, compatible]
    };
    dataset
        .commit_existing_index_segments("cidx", "code", segments)
        .await
        .unwrap();
    assert_eq!(dataset.load_indices().await.unwrap().len(), 2);
    let result = dataset
        .scan()
        .full_text_search(FullTextSearchQuery::new("user".to_owned()))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let mut ids = result["id"].as_primitive::<Int32Type>().values().to_vec();
    ids.sort_unstable();
    assert_eq!(ids, vec![0, 1]);

    // Removed segments do not constrain a full rebuild's analyzer settings.
    let replacement = dataset
        .create_index_builder(&["code"], IndexType::Inverted, &incompatible_params)
        .execute_uncommitted()
        .await
        .unwrap();
    let replacement_uuid = replacement.uuid;
    dataset
        .commit_existing_index_segments("cidx", "code", vec![replacement])
        .await
        .unwrap();
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    assert_eq!(indices[0].uuid, replacement_uuid);
    let result = dataset
        .scan()
        .full_text_search(FullTextSearchQuery::new("shared".to_owned()))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(result.num_rows(), 2);
}

#[rstest]
#[case::posting_format_version(
    InvertedIndexParams::default().format_version(InvertedListFormatVersion::V3),
    None
)]
#[case::omitted_defaults(InvertedIndexParams::default(), Some(InvertedIndexDetails::default()))]
#[tokio::test]
async fn test_commit_existing_index_segments_accepts_equivalent_inverted_details(
    #[case] incoming_params: InvertedIndexParams,
    #[case] incoming_details: Option<InvertedIndexDetails>,
) {
    let batch = arrow_array::record_batch!(
        ("id", Int32, [0, 1]),
        ("text", Utf8, ["hello world", "hello again"])
    )
    .unwrap();
    let reader = RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
    let mut dataset = Dataset::write(
        reader,
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 1,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let original = dataset
        .create_index_builder(
            &["text"],
            IndexType::Inverted,
            &InvertedIndexParams::default(),
        )
        .fragments(vec![0])
        .execute_uncommitted()
        .await
        .unwrap();
    dataset
        .commit_existing_index_segments("text_idx", "text", vec![original])
        .await
        .unwrap();
    let mut incoming = dataset
        .create_index_builder(&["text"], IndexType::Inverted, &incoming_params)
        .name("staged_text_idx".to_owned())
        .fragments(vec![1])
        .execute_uncommitted()
        .await
        .unwrap();
    if let Some(details) = incoming_details {
        incoming.index_details = Some(Arc::new(prost_types::Any::from_msg(&details).unwrap()));
    }
    dataset
        .commit_existing_index_segments("text_idx", "text", vec![incoming])
        .await
        .unwrap();
    assert_eq!(dataset.load_indices().await.unwrap().len(), 2);
    let result = dataset
        .scan()
        .full_text_search(FullTextSearchQuery::new("hello".to_owned()))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(result.num_rows(), 2);
}

#[tokio::test]
async fn test_commit_existing_index_segments_rejects_partial_replacement_of_wider_segment() {
    use lance_datagen::{BatchCount, RowCount, array};

    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();

    let reader = lance_datagen::gen_batch()
        .col("id", array::step::<arrow_array::types::Int32Type>())
        .col(
            "vector",
            array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
        )
        .into_reader_rows(RowCount::from(20), BatchCount::from(2));

    let mut dataset = Dataset::write(
        reader,
        test_uri,
        Some(WriteParams {
            max_rows_per_file: 20,
            max_rows_per_group: 20,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);

    let field_id = dataset.schema().field("vector").unwrap().id;
    let original = write_vector_segment_metadata(
        &dataset,
        "vector_idx",
        field_id,
        Uuid::new_v4(),
        [0_u32, 1_u32],
        b"original",
    )
    .await;
    dataset
        .commit_existing_index_segments("vector_idx", "vector", vec![original])
        .await
        .unwrap();

    let replacement = write_vector_segment_metadata(
        &dataset,
        "vector_idx",
        field_id,
        Uuid::new_v4(),
        [0_u32],
        b"replacement",
    )
    .await;

    let err = dataset
        .commit_existing_index_segments("vector_idx", "vector", vec![replacement])
        .await
        .unwrap_err();
    assert!(err.to_string().contains("would orphan fragments"));
}

#[tokio::test]
async fn test_commit_existing_index_segments_replaces_different_index_type() {
    use lance_datagen::{BatchCount, RowCount, array};

    let test_dir = tempfile::tempdir().unwrap();
    let reader = lance_datagen::gen_batch()
        .col("id", array::step::<arrow_array::types::Int32Type>())
        .into_reader_rows(RowCount::from(10), BatchCount::from(1));
    let mut dataset = Dataset::write(reader, test_dir.path().to_str().unwrap(), None)
        .await
        .unwrap();

    let index_name = "shared_name";
    let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
    let original = dataset
        .create_index_builder(&["id"], IndexType::BTree, &btree_params)
        .name("staged_btree".to_string())
        .execute_uncommitted()
        .await
        .unwrap();
    dataset
        .commit_existing_index_segments(index_name, "id", vec![original])
        .await
        .unwrap();

    let bitmap_params = ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap);
    let replacement = dataset
        .create_index_builder(&["id"], IndexType::Bitmap, &bitmap_params)
        .name("staged_bitmap".to_string())
        .execute_uncommitted()
        .await
        .unwrap();
    let replacement_uuid = replacement.uuid;
    let replacement_type_url = replacement.index_details.as_ref().unwrap().type_url.clone();

    dataset
        .commit_existing_index_segments(index_name, "id", vec![replacement])
        .await
        .unwrap();

    let committed = dataset
        .load_index_by_name(index_name)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(committed.uuid, replacement_uuid);
    assert_eq!(
        committed.index_details.unwrap().type_url,
        replacement_type_url
    );
}

#[tokio::test]
async fn test_commit_existing_index_segments_rejects_partial_index_type_change() {
    use lance_datagen::{BatchCount, RowCount, array};

    let test_dir = tempfile::tempdir().unwrap();
    let reader = lance_datagen::gen_batch()
        .col("id", array::step::<arrow_array::types::Int32Type>())
        .into_reader_rows(RowCount::from(20), BatchCount::from(2));
    let mut dataset = Dataset::write(
        reader,
        test_dir.path().to_str().unwrap(),
        Some(WriteParams {
            max_rows_per_file: 20,
            max_rows_per_group: 20,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let index_name = "shared_name";
    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), 2);
    let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
    let mut original_segments = Vec::with_capacity(fragments.len());
    for fragment in &fragments {
        original_segments.push(
            dataset
                .create_index_builder(&["id"], IndexType::BTree, &btree_params)
                .fragments(vec![fragment.id() as u32])
                .execute_uncommitted()
                .await
                .unwrap(),
        );
    }
    dataset
        .commit_existing_index_segments(index_name, "id", original_segments)
        .await
        .unwrap();
    let committed_version = dataset.manifest.version;

    let bitmap_params = ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap);
    let replacement = dataset
        .create_index_builder(&["id"], IndexType::Bitmap, &bitmap_params)
        .fragments(vec![fragments[0].id() as u32])
        .execute_uncommitted()
        .await
        .unwrap();
    let error = dataset
        .commit_existing_index_segments(index_name, "id", vec![replacement])
        .await
        .unwrap_err();

    assert!(
        matches!(error, Error::InvalidInput { .. }),
        "expected invalid input error, got: {error}"
    );
    assert!(
        error
            .to_string()
            .contains("cannot change index 'shared_name'")
    );
    assert!(error.to_string().contains("missing current fragments [1]"));
    assert_eq!(dataset.manifest.version, committed_version);

    let committed =
        crate::index::scalar_logical::load_named_scalar_segments(&dataset, "id", index_name)
            .await
            .unwrap();
    assert_eq!(committed.len(), 2);
    assert!(committed.iter().all(|segment| {
        segment
            .index_details
            .as_ref()
            .is_some_and(|details| details.type_url.ends_with("BTreeIndexDetails"))
    }));
}

/// A covered index still names the same column by its keyed prefix. The
/// name-collision guard in `commit_existing_index_segments` must recognize
/// that, not reject on the full `fields` vector including the carried
/// column -- which would misfire "already exists with different fields"
/// before ever reaching the deeper type-compatibility check this test
/// otherwise shares with
/// `test_commit_existing_index_segments_rejects_partial_index_type_change`.
#[tokio::test]
async fn test_commit_existing_index_segments_recognizes_covered_index_by_name() {
    use lance_datagen::{BatchCount, RowCount, array};

    let test_dir = tempfile::tempdir().unwrap();
    let reader = lance_datagen::gen_batch()
        .col("id", array::step::<arrow_array::types::Int32Type>())
        .col("extra", array::step::<arrow_array::types::Int32Type>())
        .into_reader_rows(RowCount::from(20), BatchCount::from(2));
    let mut dataset = Dataset::write(
        reader,
        test_dir.path().to_str().unwrap(),
        Some(WriteParams {
            max_rows_per_file: 20,
            max_rows_per_group: 20,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let index_name = "shared_name";
    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), 2);
    let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
    let mut original_segments = Vec::with_capacity(fragments.len());
    for fragment in &fragments {
        original_segments.push(
            dataset
                .create_index_builder(&["id"], IndexType::BTree, &btree_params)
                .fragments(vec![fragment.id() as u32])
                .execute_uncommitted()
                .await
                .unwrap(),
        );
    }
    dataset
        .commit_existing_index_segments(index_name, "id", original_segments)
        .await
        .unwrap();

    // Hand-declare the committed index as covering an extra carried column
    // -- there is no producer in this phase, so this is done directly.
    let id_field = dataset.schema().field("id").unwrap().id;
    let extra_field = dataset.schema().field("extra").unwrap().id;
    let current = dataset.load_indices_by_name(index_name).await.unwrap();
    let covered_segments = current
        .iter()
        .cloned()
        .map(|mut idx| {
            idx.fields = vec![id_field, extra_field];
            idx.covering_fields = vec![extra_field];
            idx
        })
        .collect::<Vec<_>>();
    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: covered_segments,
            removed_indices: current.to_vec(),
        },
        None,
    );
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await
        .unwrap();

    let bitmap_params = ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap);
    let replacement = dataset
        .create_index_builder(&["id"], IndexType::Bitmap, &bitmap_params)
        .fragments(vec![fragments[0].id() as u32])
        .execute_uncommitted()
        .await
        .unwrap();
    let error = dataset
        .commit_existing_index_segments(index_name, "id", vec![replacement])
        .await
        .unwrap_err();

    // Must reach the deeper type-compatibility error, not misfire the
    // shallow "already exists with different fields" guard this test
    // targets.
    assert!(
        error
            .to_string()
            .contains("cannot change index 'shared_name'"),
        "{error}"
    );
    assert!(
        error.to_string().contains("missing current fragments"),
        "{error}"
    );
    assert!(
        !error
            .to_string()
            .contains("already exists with different fields"),
        "{error}"
    );
}

#[tokio::test]
async fn test_partial_type_change_with_legacy_missing_details_is_rejected() {
    use lance_datagen::{BatchCount, RowCount, array};

    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();
    let reader = lance_datagen::gen_batch()
        .col("id", array::step::<arrow_array::types::Int32Type>())
        .into_reader_rows(RowCount::from(10), BatchCount::from(1));
    let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

    let index_name = "shared_name";
    let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
    let original = dataset
        .create_index_builder(&["id"], IndexType::BTree, &btree_params)
        .execute_uncommitted()
        .await
        .unwrap();
    dataset
        .commit_existing_index_segments(index_name, "id", vec![original])
        .await
        .unwrap();

    let current = dataset.load_indices_by_name(index_name).await.unwrap();
    assert_eq!(current.len(), 1);
    let original_uuid = current[0].uuid;
    let mut legacy = current.clone();
    legacy[0].index_details = None;
    legacy[0].index_version = 0;
    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: legacy,
            removed_indices: current,
        },
        None,
    );
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await
        .unwrap();

    let append_reader = lance_datagen::gen_batch()
        .col("id", array::step::<arrow_array::types::Int32Type>())
        .into_reader_rows(RowCount::from(10), BatchCount::from(1));
    let mut dataset = Dataset::write(
        append_reader,
        test_uri,
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), 2);

    let bitmap_params = ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap);
    let replacement = dataset
        .create_index_builder(&["id"], IndexType::Bitmap, &bitmap_params)
        .fragments(vec![fragments[1].id() as u32])
        .execute_uncommitted()
        .await
        .unwrap();
    let version_before = dataset.manifest.version;
    let error = dataset
        .commit_existing_index_segments(index_name, "id", vec![replacement])
        .await
        .unwrap_err();

    assert!(
        matches!(error, Error::InvalidInput { .. }),
        "expected invalid input error, got: {error}"
    );
    assert!(error.to_string().contains("from type '<unknown>'"));
    assert!(error.to_string().contains("partial fragment coverage"));
    assert!(error.to_string().contains("missing current fragments [0]"));
    assert_eq!(dataset.manifest.version, version_before);

    let committed = dataset.load_indices_by_name(index_name).await.unwrap();
    assert_eq!(committed.len(), 1);
    assert_eq!(committed[0].uuid, original_uuid);
    assert!(committed[0].index_details.is_none());
    assert_eq!(committed[0].index_version, 0);

    let field_id = dataset.schema().field("id").unwrap().id;
    let sentinel_segment = IndexSegment::new(
        Uuid::new_v4(),
        [fragments[1].id() as u32],
        [field_id],
        Arc::new(prost_types::Any {
            type_url: "<unknown>".to_string(),
            value: Vec::new(),
        }),
        0,
        dataset.manifest.version,
        vec![],
    );
    let error = dataset
        .commit_existing_index_segments(index_name, "id", vec![sentinel_segment])
        .await
        .unwrap_err();

    assert!(
        matches!(error, Error::InvalidInput { .. }),
        "expected invalid input error, got: {error}"
    );
    assert!(error.to_string().contains("from type '<unknown>'"));
    assert!(error.to_string().contains("to type '<unknown>'"));
    assert!(error.to_string().contains("missing current fragments [0]"));
    assert_eq!(dataset.manifest.version, version_before);

    let committed = dataset.load_indices_by_name(index_name).await.unwrap();
    assert_eq!(committed.len(), 1);
    assert_eq!(committed[0].uuid, original_uuid);
    assert!(committed[0].index_details.is_none());
    assert_eq!(committed[0].index_version, 0);
}

#[tokio::test]
async fn test_commit_existing_index_segments_removes_empty_segment() {
    use lance_datagen::{BatchCount, RowCount, array};

    let test_dir = tempfile::tempdir().unwrap();
    let reader = lance_datagen::gen_batch()
        .col("id", array::step::<arrow_array::types::Int32Type>())
        .col(
            "vector",
            array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
        )
        .into_reader_rows(RowCount::from(10), BatchCount::from(1));
    let mut dataset = Dataset::write(reader, test_dir.path().to_str().unwrap(), None)
        .await
        .unwrap();
    let field_id = dataset.schema().field("vector").unwrap().id;
    let uuid = Uuid::new_v4();

    // Commit a 0-fragment segment, then a real segment covering the dataset.
    let empty = write_vector_segment_metadata(
        &dataset,
        "vector_idx",
        field_id,
        uuid,
        std::iter::empty::<u32>(),
        b"empty",
    )
    .await;
    dataset
        .commit_existing_index_segments("vector_idx", "vector", vec![segment_from_metadata(&empty)])
        .await
        .unwrap();
    let seg = write_vector_segment_metadata(
        &dataset,
        "vector_idx",
        field_id,
        Uuid::new_v4(),
        [0_u32],
        b"seg",
    )
    .await;
    dataset
        .commit_existing_index_segments("vector_idx", "vector", vec![segment_from_metadata(&seg)])
        .await
        .unwrap();

    // The real segment covers the dataset, so the redundant empty one is removed.
    let committed = dataset.load_indices_by_name("vector_idx").await.unwrap();
    assert_eq!(
        committed.iter().map(|i| i.uuid).collect::<HashSet<_>>(),
        HashSet::from([seg.uuid]),
        "empty segment should be removed once a real segment covers the dataset",
    );
}

/// An index this build cannot read must be hidden, not erased.
///
/// Every commit rebuilds the index list from what it is handed, so filtering
/// the version there turns "ignore it" into "delete it", and the build that
/// could have read the index never gets the chance.
#[rstest]
#[case::newer_version(UnreadableIndexKind::NewerVersion)]
#[case::unknown_type(UnreadableIndexKind::UnknownType)]
#[case::reserved_field(UnreadableIndexKind::ReservedField)]
#[tokio::test]
async fn test_unsupported_index_survives_an_unrelated_commit(#[case] kind: UnreadableIndexKind) {
    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();
    dataset_with_an_unreadable_index(test_uri, "id_idx", kind).await;

    let dataset = Dataset::open(test_uri).await.unwrap();
    assert_eq!(
        dataset
            .load_indices()
            .await
            .unwrap()
            .iter()
            .map(|idx| idx.name.as_str())
            .collect::<Vec<_>>(),
        [READABLE_INDEX],
        "the filter must hide the index this build cannot read, and only it"
    );
    assert_eq!(
        manifest_index_names(&dataset).await,
        ["id_idx", READABLE_INDEX]
    );
    let before = manifest_index(&dataset, "id_idx").await;

    // An unrelated append. Nothing about the index is part of this operation.
    let dataset = Dataset::write(
        two_column_reader(),
        test_uri,
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_eq!(
        manifest_index(&dataset, "id_idx").await,
        before,
        "an unrelated append changed an index this build merely could not read"
    );
}

#[rstest]
#[case::row_id(lance_table::format::ROW_ID_FIELD_ID)]
#[case::unknown_reserved_field(-100)]
#[tokio::test]
async fn test_reserved_field_index_does_not_break_queries_or_maintenance(#[case] field_id: i32) {
    let test_uri = "memory://";
    let reader = || {
        gen_batch()
            .col("rid", array::step::<arrow_array::types::UInt64Type>())
            .col("payload", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(10), BatchCount::from(1))
    };
    let mut dataset = Dataset::write(
        reader(),
        test_uri,
        Some(WriteParams {
            enable_stable_row_ids: true,
            max_rows_per_file: 5,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);
    let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
    for (column, name) in [("rid", "rowid_idx"), ("payload", READABLE_INDEX)] {
        dataset
            .create_index_builder(&[column], IndexType::BTree, &btree_params)
            .name(name.to_string())
            .await
            .unwrap();
    }
    let original = manifest_index(&dataset, "rowid_idx").await;
    let reserved = IndexMetadata {
        fields: vec![field_id],
        ..original.clone()
    };
    dataset
        .apply_commit(
            Transaction::new(
                dataset.manifest.version,
                Operation::CreateIndex {
                    new_indices: vec![reserved.clone()],
                    removed_indices: vec![original],
                },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();

    // The reserved-field segment must not interfere with either an unindexed
    // predicate or the readable companion index.
    for filter in ["rid < 3", "payload < 3"] {
        assert_eq!(
            dataset
                .scan()
                .filter(filter)
                .unwrap()
                .try_into_batch()
                .await
                .unwrap()
                .num_rows(),
            3
        );
    }
    assert_eq!(
        dataset
            .load_indices()
            .await
            .unwrap()
            .iter()
            .map(|index| index.name.as_str())
            .collect::<Vec<_>>(),
        [READABLE_INDEX]
    );

    dataset
        .append(reader(), Some(WriteParams::default()))
        .await
        .unwrap();
    assert_eq!(manifest_index(&dataset, "rowid_idx").await, reserved);
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    assert_eq!(manifest_index(&dataset, "rowid_idx").await, reserved);

    let metrics = compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();
    assert!(metrics.fragments_removed > 0);
    let after = manifest_index(&dataset, "rowid_idx").await;
    assert_eq!(after.uuid, reserved.uuid);
    assert_eq!(after.fields, reserved.fields);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 20);
    assert_eq!(
        dataset
            .scan()
            .filter("payload < 3")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap()
            .num_rows(),
        6
    );
}

/// Carrying an unreadable index forward is not the same as keeping it
/// forever: dropping the column it covers still removes it.
#[tokio::test]
async fn test_unsupported_index_is_dropped_with_its_column() {
    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();
    dataset_with_an_index_from_a_newer_build(test_uri, "id_idx").await;

    let mut dataset = Dataset::open(test_uri).await.unwrap();
    let readable_before = manifest_index(&dataset, READABLE_INDEX).await;

    dataset.drop_columns(&["id"]).await.unwrap();

    assert_eq!(
        manifest_index_names(&dataset).await,
        [READABLE_INDEX],
        "dropping `id` must remove the index over it, and only that one"
    );
    assert_eq!(
        manifest_index(&dataset, READABLE_INDEX).await,
        readable_before,
        "the index over the surviving column was rewritten"
    );
}

/// Carrying it forward must also not drag it through index migration.
///
/// `migrate_indices` recalculates a missing `fragment_bitmap` by opening the
/// index, which is precisely what this build cannot do - so an unreadable
/// index would fail every later commit instead of riding along.
#[rstest]
#[case::newer_version(UnreadableIndexKind::NewerVersion)]
#[case::unknown_type(UnreadableIndexKind::UnknownType)]
#[case::reserved_field(UnreadableIndexKind::ReservedField)]
#[tokio::test]
async fn test_unsupported_index_without_a_bitmap_does_not_fail_later_commits(
    #[case] kind: UnreadableIndexKind,
) {
    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();
    let mut dataset = dataset_with_an_unreadable_index(test_uri, "id_idx", kind).await;

    // Drop the coverage too, so migration would want to rebuild it.
    let hidden = manifest_index(&dataset, "id_idx").await;
    let without_bitmap = IndexMetadata {
        fragment_bitmap: None,
        ..hidden.clone()
    };
    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: vec![without_bitmap.clone()],
            removed_indices: vec![hidden],
        },
        None,
    );
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await
        .unwrap();

    let mut dataset = Dataset::open(test_uri).await.unwrap();
    dataset.delete("false").await.unwrap();

    assert_eq!(
        manifest_index(&dataset, "id_idx").await,
        without_bitmap,
        "a commit rewrote an index it cannot open instead of carrying it through"
    );
}

/// A name an unreadable index already owns cannot be handed out again.
///
/// Nothing in the format stops two entries from sharing a name, and the
/// build that can read both would take them for segments of one index.
#[tokio::test]
async fn test_unsupported_index_name_is_still_taken() {
    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();
    dataset_with_an_index_from_a_newer_build(test_uri, "id_idx").await;

    let mut dataset = Dataset::open(test_uri).await.unwrap();
    let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
    let err = dataset
        .create_index_builder(&["id"], IndexType::BTree, &btree_params)
        .name("id_idx".to_string())
        .await
        .expect_err("a name is taken by an index this build cannot read");
    assert!(
        err.to_string().contains("already exists"),
        "expected a name collision, got: {err}"
    );
    assert_eq!(
        manifest_index_names(&dataset).await,
        ["id_idx", READABLE_INDEX]
    );
}

/// The generated-name loop reads the same view the collision check does, so
/// a name an unreadable index holds is skipped rather than reused.
#[tokio::test]
async fn test_an_auto_generated_name_skips_an_unsupported_index() {
    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();
    dataset_with_an_index_from_a_newer_build(test_uri, "id_idx").await;

    // A different index kind on the same column: the loop steps past the
    // taken name instead of stopping at the collision check.
    let mut dataset = Dataset::open(test_uri).await.unwrap();
    let bitmap_params = ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap);
    dataset
        .create_index_builder(&["id"], IndexType::Bitmap, &bitmap_params)
        .train(false)
        .await
        .unwrap();

    assert_eq!(
        manifest_index_names(&dataset).await,
        ["id_idx", "id_idx_2", READABLE_INDEX],
        "the generated name reused one an unreadable index already holds"
    );
}

/// The multi-segment FM-Index builder reserves names on its own, so it needs
/// the same complete view as the single-segment path.
///
/// The hidden index is an FM index too: the default-name loop steps past a
/// different index kind on the same column, so only an index of the same
/// kind keeps `text_idx` and reaches the collision check. A builder that
/// could not see the hidden index would commit a second `text_idx` instead.
#[tokio::test]
async fn test_multi_segment_fmindex_respects_an_unsupported_index_name() {
    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();

    let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(StringArray::from(vec!["alpha", "beta", "gamma"]))],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema.clone()),
        test_uri,
        None,
    )
    .await
    .unwrap();

    let fm_params = ScalarIndexParams::for_builtin(BuiltinIndexType::Fm);
    dataset
        .create_index_builder(&["text"], IndexType::Fm, &fm_params)
        .name("text_idx".to_string())
        .train(false)
        .await
        .unwrap();
    hide_index_from_this_build(&mut dataset, "text_idx").await;

    let mut dataset = Dataset::open(test_uri).await.unwrap();
    let multi_segment_params = ScalarIndexParams::for_builtin(BuiltinIndexType::Fm)
        .with_params(&serde_json::json!({ "num_segments": 2 }));
    let err = dataset
        .create_index_builder(&["text"], IndexType::Fm, &multi_segment_params)
        .train(false)
        .await
        .expect_err("a name is taken by an index this build cannot read");
    assert!(
        err.to_string().contains("already exists"),
        "expected a name collision, got: {err}"
    );
    assert_eq!(manifest_index_names(&dataset).await, ["text_idx"]);
}

/// Being unreadable must not make an index unremovable.
#[tokio::test]
async fn test_unsupported_index_can_be_dropped_by_name() {
    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();
    dataset_with_an_index_from_a_newer_build(test_uri, "id_idx").await;

    let mut dataset = Dataset::open(test_uri).await.unwrap();
    dataset.drop_index("id_idx").await.unwrap();

    assert_eq!(
        manifest_index_names(&dataset).await,
        [READABLE_INDEX],
        "drop_index removed the wrong set of indices"
    );
}

/// `replace` has to select the index it replaces from the same complete view
/// the name was reserved against, or it adds a twin instead of replacing.
#[tokio::test]
async fn test_replacing_an_unsupported_index_does_not_duplicate_it() {
    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();
    dataset_with_an_index_from_a_newer_build(test_uri, "id_idx").await;

    let mut dataset = Dataset::open(test_uri).await.unwrap();
    let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
    dataset
        .create_index_builder(&["id"], IndexType::BTree, &btree_params)
        .name("id_idx".to_string())
        .replace(true)
        .train(false)
        .await
        .unwrap();

    assert_eq!(
        manifest_index_names(&dataset).await,
        ["id_idx", READABLE_INDEX],
        "replace committed a second index under a name already taken"
    );
    assert!(
        unsupported_index_version(&manifest_index(&dataset, "id_idx").await).is_none(),
        "replace kept the unreadable index and discarded the new one"
    );
}

/// The same selection, through the segment-commit path rather than the
/// builder: full coverage replaces the segments already under that name.
#[tokio::test]
async fn test_committing_a_segment_beside_an_unsupported_index_replaces_it() {
    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();
    dataset_with_an_index_from_a_newer_build(test_uri, "id_idx").await;

    let mut dataset = Dataset::open(test_uri).await.unwrap();
    let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
    let mut segment = dataset
        .create_index_builder(&["id"], IndexType::BTree, &btree_params)
        .name("id_idx".to_string())
        .replace(true)
        .train(false)
        .execute_uncommitted()
        .await
        .unwrap();
    // Full coverage, so the removal decision goes through the fragment
    // overlap branch rather than the empty-bitmap shortcut. Set by hand
    // because training it would take 40 MB of the shared pool to sort ten
    // rows - see `dataset_with_an_index_from_a_newer_build`.
    segment.fragment_bitmap = Some(dataset.fragment_bitmap.as_ref().clone());
    dataset
        .commit_existing_index_segments("id_idx", "id", vec![segment])
        .await
        .unwrap();

    assert_eq!(
        manifest_index_names(&dataset).await,
        ["id_idx", READABLE_INDEX],
        "the incoming segment was committed beside the unreadable one"
    );
    assert!(
        unsupported_index_version(&manifest_index(&dataset, "id_idx").await).is_none(),
        "the incoming segment did not replace the unreadable one"
    );
}

/// The retention path itself: a segment on fragments the existing one does
/// not cover is kept beside it, and agreeing declarations are what makes
/// that legal. This is the case the rejection below must not swallow -
/// partial-coverage builds depend on it.
#[tokio::test]
async fn test_committing_a_segment_on_disjoint_fragments_keeps_the_existing_one() {
    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();

    let mut dataset = Dataset::write(two_column_reader(), test_uri, None)
        .await
        .unwrap();
    let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
    dataset
        .create_index_builder(&["id"], IndexType::BTree, &btree_params)
        .name("id_idx".to_string())
        .train(false)
        .await
        .unwrap();
    // Given by hand: an untrained segment commits with an empty bitmap, which
    // takes the zero-coverage removal branch rather than the disjoint one.
    // See `dataset_with_an_index_from_a_newer_build` for why nothing here trains.
    let untrained = manifest_index(&dataset, "id_idx").await;
    let mut existing = untrained.clone();
    existing.fragment_bitmap = Some(dataset.fragment_bitmap.as_ref().clone());
    dataset
        .apply_commit(
            Transaction::new(
                dataset.manifest.version,
                Operation::CreateIndex {
                    new_indices: vec![existing.clone()],
                    removed_indices: vec![untrained],
                },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();
    let covered = existing.fragment_bitmap.clone().unwrap();
    assert!(!covered.is_empty());

    let mut dataset = Dataset::write(
        two_column_reader(),
        test_uri,
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let appended = dataset.fragment_bitmap.as_ref() - &covered;
    assert!(!appended.is_empty());

    let mut segment = dataset
        .create_index_builder(&["id"], IndexType::BTree, &btree_params)
        .name("id_idx".to_string())
        .replace(true)
        .fragments(appended.iter().collect())
        .train(false)
        .execute_uncommitted()
        .await
        .unwrap();
    segment.fragment_bitmap = Some(appended);
    assert_eq!(segment.fields, existing.fields);
    assert_eq!(segment.covering_fields, existing.covering_fields);

    dataset
        .commit_existing_index_segments("id_idx", "id", vec![segment])
        .await
        .unwrap();

    let uuids = raw_manifest_indices(&dataset)
        .await
        .into_iter()
        .filter(|idx| idx.name == "id_idx")
        .map(|idx| idx.uuid)
        .collect::<Vec<_>>();
    assert_eq!(uuids.len(), 2, "the disjoint existing segment was dropped");
    assert!(uuids.contains(&existing.uuid));
}

/// One logical index needs one declaration, and the complete view is what
/// makes the disagreement reachable: a segment this build cannot read may
/// carry columns, and a plain segment committed beside it on disjoint
/// fragments is retained rather than replaced. Both pass the per-segment
/// rules - `keyed_fields` is the prefix left after the carried ones, so
/// `[id, payload]` carrying `[payload]` keys on `id` exactly as `[id]` does.
/// Committing the pair would leave `describe_indices` erroring on metadata
/// this call just wrote, the same failure
/// `test_build_index_metadata_from_segments_rejects_mixed_covering_declarations`
/// pins for the incoming side.
#[tokio::test]
async fn test_committing_a_plain_segment_beside_a_covered_unsupported_one_is_rejected() {
    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();
    let mut dataset = dataset_with_an_index_from_a_newer_build(test_uri, "id_idx").await;

    // Give the hidden segment a carried column. A newer build is exactly
    // where a covering segment would come from.
    let hidden = manifest_index(&dataset, "id_idx").await;
    let payload_id = dataset.schema().field("payload").unwrap().id;
    let mut hidden_covered = hidden.clone();
    hidden_covered.fields.push(payload_id);
    hidden_covered.covering_fields = vec![payload_id];
    assert_eq!(hidden_covered.keyed_field(), hidden.keyed_field());
    let covered_fragments = hidden_covered.fragment_bitmap.clone().unwrap();
    dataset
        .apply_commit(
            Transaction::new(
                dataset.manifest.version,
                Operation::CreateIndex {
                    new_indices: vec![hidden_covered],
                    removed_indices: vec![hidden],
                },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();

    // Fragments the hidden segment does not cover, so the incoming segment
    // takes the disjoint branch and the hidden one is retained.
    let mut dataset = Dataset::write(
        two_column_reader(),
        test_uri,
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let appended = dataset.fragment_bitmap.as_ref() - &covered_fragments;
    assert!(!appended.is_empty());

    let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
    let mut plain = dataset
        .create_index_builder(&["id"], IndexType::BTree, &btree_params)
        .name("id_idx".to_string())
        .replace(true)
        .fragments(appended.iter().collect())
        .train(false)
        .execute_uncommitted()
        .await
        .unwrap();
    plain.fragment_bitmap = Some(appended);
    assert!(plain.covering_fields.is_empty());

    let err = dataset
        .commit_existing_index_segments("id_idx", "id", vec![plain])
        .await
        .expect_err("a logical index cannot mix covered and plain segment declarations");
    assert!(
        err.to_string().contains("covering_fields"),
        "unexpected error: {err}"
    );
}

/// A cast reassigns the field id, so no index on that column can be carried
/// forward. The guard that makes that explicit has to see the hidden ones
/// too, or they get exactly the silent drop it exists to abolish.
#[tokio::test]
async fn test_casting_a_column_with_an_unsupported_index_is_rejected() {
    use crate::dataset::ColumnAlteration;

    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();
    dataset_with_an_index_from_a_newer_build(test_uri, "id_idx").await;

    let mut dataset = Dataset::open(test_uri).await.unwrap();
    let err = dataset
        .alter_columns(&[ColumnAlteration::new("id".into()).cast_to(DataType::Int64)])
        .await
        .expect_err("a cast must not silently erase an index it cannot read");
    assert!(
        err.to_string().contains("id_idx"),
        "the error should name the index, got: {err}"
    );
    assert_eq!(
        manifest_index_names(&dataset).await,
        ["id_idx", READABLE_INDEX],
        "the rejected cast still changed the manifest"
    );
}

/// `validate` checks the manifest, so it has to see all of it. An index this
/// build cannot read is no less corrupt for being unreadable, and now that
/// such an index is carried forward the corrupt state is durable rather than
/// gone at the next commit.
#[tokio::test]
async fn test_validate_sees_an_unsupported_index() {
    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();
    let mut dataset = dataset_with_an_index_from_a_newer_build(test_uri, "id_idx").await;
    dataset.validate().await.unwrap();

    // A second segment under the same name covering the same fragments. Only
    // `detect_overlapping_fragments` over the complete list can see it.
    let hidden = manifest_index(&dataset, "id_idx").await;
    let overlapping = IndexMetadata {
        uuid: Uuid::new_v4(),
        ..hidden
    };
    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: vec![overlapping],
            removed_indices: vec![],
        },
        None,
    );
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await
        .unwrap();

    let err = dataset
        .validate()
        .await
        .expect_err("two segments of one name covering the same fragments is corrupt");
    assert!(
        err.to_string().contains("id_idx"),
        "the error should name the index, got: {err}"
    );
}

/// A detached commit builds its manifest through its own code path, and it
/// carries the index list forward exactly as an attached one does.
#[tokio::test]
async fn test_a_detached_commit_does_not_erase_an_unsupported_index() {
    use crate::dataset::InsertBuilder;
    use crate::dataset::write::CommitBuilder;

    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();
    dataset_with_an_index_from_a_newer_build(test_uri, "id_idx").await;

    let batches = two_column_reader()
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap();
    let dataset = Arc::new(Dataset::open(test_uri).await.unwrap());
    let before = manifest_index(&dataset, "id_idx").await;
    let transaction = InsertBuilder::new(dataset.clone())
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute_uncommitted(batches)
        .await
        .unwrap();
    let detached = CommitBuilder::new(dataset.clone())
        .with_detached(true)
        .execute(transaction)
        .await
        .unwrap();

    assert_eq!(
        manifest_index_names(&detached).await,
        ["id_idx", READABLE_INDEX],
        "a detached commit erased an index this build merely could not read"
    );
    assert_eq!(
        manifest_index(&detached, "id_idx").await,
        before,
        "a detached commit rewrote an index this build merely could not read"
    );
    assert!(lance_table::format::is_detached_version(
        detached.manifest.version
    ));
    assert_eq!(detached.count_rows(None).await.unwrap(), 20);
}

/// Compaction bins fragments so that no rewrite group splits an index's
/// coverage, and `recalculate_fragment_bitmap` rejects the plan if one does.
/// Both sides therefore have to count the same indices: planning from the
/// filtered view while the commit carries the complete one makes compaction
/// fail outright on a dataset holding an index from a newer build.
#[tokio::test]
async fn test_compaction_survives_an_unsupported_index() {
    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();

    // A text column alongside `two_column_reader`'s numeric ones: every
    // remaining row-id-domain scalar index type (NGram here) requires text,
    // unlike BTree and Bitmap, which moved to row-address domain and no
    // longer exercise this test.
    fn three_column_reader() -> impl arrow_array::RecordBatchReader + Send + 'static {
        lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("payload", array::step::<Int32Type>())
            .col("text", array::rand_utf8(ByteCount::from(8), false))
            .into_reader_rows(RowCount::from(10), BatchCount::from(1))
    }

    let write_params = WriteParams {
        enable_stable_row_ids: true,
        max_rows_per_file: 5,
        ..Default::default()
    };
    let mut dataset = Dataset::write(three_column_reader(), test_uri, Some(write_params.clone()))
        .await
        .unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);

    // Row-id-domain: its coverage is recalculated (not dropped) across a
    // rewrite under stable row ids, which is exactly the property this test
    // checks. BTree and Bitmap moved to row-address domain and no longer
    // exercise it.
    let ngram_params = ScalarIndexParams::for_builtin(BuiltinIndexType::NGram);
    dataset
        .create_index_builder(&["text"], IndexType::NGram, &ngram_params)
        .name("id_idx".to_string())
        .train(false)
        .await
        .unwrap();
    hide_index_from_this_build(&mut dataset, "id_idx").await;

    // A fragment the index does not cover, so a bin holding it together with
    // the covered ones would split the index's coverage.
    let mut dataset = Dataset::write(
        three_column_reader(),
        test_uri,
        Some(WriteParams {
            mode: WriteMode::Append,
            ..write_params
        }),
    )
    .await
    .unwrap();
    let before = manifest_index(&dataset, "id_idx").await;

    let metrics = compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();

    // Without a real rewrite the assertions below hold vacuously: an empty
    // plan commits nothing and re-reads the manifest it started from.
    assert_eq!(metrics.fragments_removed, 4);
    assert_eq!(dataset.get_fragments().len(), 2);

    let after = manifest_index(&dataset, "id_idx").await;
    assert_eq!(after.uuid, before.uuid);
    assert_eq!(after.index_version, before.index_version);
    let live = dataset
        .get_fragments()
        .iter()
        .map(|f| f.id() as u32)
        .collect::<RoaringBitmap>();
    assert!(
        !after.fragment_bitmap.unwrap().is_disjoint(&live),
        "the surviving index covers only fragments the rewrite deleted"
    );
}

/// Without stable row ids a rewrite moves every row address, so each index
/// over the rewritten fragments has to be remapped - and remapping one means
/// opening it. A build with no reader for an index cannot remap it, so
/// compacting the fragments it covers would leave it addressing rows that no
/// longer exist. Those fragments are held back from the plan instead; the
/// rest of the table still compacts.
///
/// The stable-row-id case is the test above: there the fragment-reuse index
/// repairs the coverage afterwards, so nothing has to be held back.
#[rstest]
#[case::newer_version(UnreadableIndexKind::NewerVersion)]
#[case::unknown_type(UnreadableIndexKind::UnknownType)]
#[case::reserved_field(UnreadableIndexKind::ReservedField)]
#[tokio::test]
async fn test_compaction_defers_fragments_an_unsupported_index_covers(
    #[case] kind: UnreadableIndexKind,
) {
    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();

    let write_params = WriteParams {
        enable_stable_row_ids: false,
        max_rows_per_file: 5,
        ..Default::default()
    };
    let mut dataset = Dataset::write(two_column_reader(), test_uri, Some(write_params.clone()))
        .await
        .unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);

    let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
    dataset
        .create_index_builder(&["id"], IndexType::BTree, &btree_params)
        .name("id_idx".to_string())
        .train(false)
        .await
        .unwrap();
    hide_index_as(&mut dataset, "id_idx", kind).await;
    let covered = manifest_index(&dataset, "id_idx")
        .await
        .fragment_bitmap
        .unwrap();

    // Two more fragments the hidden index does not cover: they are the ones
    // compaction is still free to rewrite.
    let mut dataset = Dataset::write(
        two_column_reader(),
        test_uri,
        Some(WriteParams {
            mode: WriteMode::Append,
            ..write_params
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 4);
    let before = manifest_index(&dataset, "id_idx").await;

    let metrics = compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();

    // The two uncovered fragments coalesce; the two the hidden index covers
    // are left alone. Both halves matter: no rewrite at all would satisfy the
    // coverage assertion below for the wrong reason.
    assert_eq!(metrics.fragments_removed, 2);
    assert_eq!(metrics.fragments_added, 1);
    let live = dataset
        .get_fragments()
        .iter()
        .map(|f| f.id() as u32)
        .collect::<RoaringBitmap>();
    assert!(
        covered.is_subset(&live),
        "a fragment the unreadable index covers was rewritten: covered {covered:?}, live {live:?}"
    );

    let after = manifest_index(&dataset, "id_idx").await;
    assert_eq!(after.uuid, before.uuid);
    assert_eq!(after.fragment_bitmap, before.fragment_bitmap);

    // Nothing compactable is left outside the held-back set, so the plan is
    // empty. That has to be an ordinary no-op: on a table the index covers
    // whole - the usual shape - every compaction takes this path.
    let metrics = compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();
    assert_eq!(metrics.fragments_removed, 0);
    assert_eq!(metrics.fragments_added, 0);
    assert_eq!(manifest_index(&dataset, "id_idx").await.uuid, before.uuid);
}

/// Holding back the fragments an unreadable index covers is an optimization
/// in the planner, not the rule: `compact_files_with_planner` takes any
/// planner, `CompactionPlan` is public and serializable, and a distributed
/// driver hands `commit_compaction` results planned on another machine. This
/// takes that last route, so the refusal is pinned to the commit boundary.
///
/// The other half - that the boundary does not refuse a rewrite the index
/// does not cover - is the test above, which compacts the uncovered
/// fragments of this same shape through `compact_files`.
#[tokio::test]
async fn test_committing_a_compaction_an_unsupported_index_covers_is_rejected() {
    use crate::dataset::index::DatasetIndexRemapperOptions;
    use crate::dataset::optimize::{CompactionPlan, TaskData, commit_compaction};

    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();

    let write_params = WriteParams {
        enable_stable_row_ids: false,
        max_rows_per_file: 5,
        ..Default::default()
    };
    let mut dataset = Dataset::write(two_column_reader(), test_uri, Some(write_params.clone()))
        .await
        .unwrap();

    let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
    dataset
        .create_index_builder(&["id"], IndexType::BTree, &btree_params)
        .name("id_idx".to_string())
        .train(false)
        .await
        .unwrap();
    hide_index_from_this_build(&mut dataset, "id_idx").await;
    let covered = manifest_index(&dataset, "id_idx")
        .await
        .fragment_bitmap
        .unwrap();

    // Two more fragments the hidden index does not cover, so the plan below
    // is a genuine selection rather than "every fragment there is".
    let mut dataset = Dataset::write(
        two_column_reader(),
        test_uri,
        Some(WriteParams {
            mode: WriteMode::Append,
            ..write_params
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 4);
    let before = manifest_index(&dataset, "id_idx").await;

    let plan = CompactionPlan {
        tasks: vec![TaskData {
            fragments: dataset
                .fragments()
                .iter()
                .filter(|fragment| covered.contains(fragment.id as u32))
                .cloned()
                .collect(),
            output_clustering_id: None,
            output_metadata: Default::default(),
        }],
        read_version: dataset.version().version,
        options: CompactionOptions::default(),
    };
    assert_eq!(plan.tasks[0].fragments.len(), 2);

    let mut rewrites = Vec::new();
    for task in plan.compaction_tasks() {
        rewrites.push(task.execute(&dataset).await.unwrap());
    }

    let err = commit_compaction(
        &mut dataset,
        rewrites,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &plan.options,
    )
    .await
    .unwrap_err();
    assert!(
        err.to_string().contains("id_idx"),
        "the refusal has to name the index that blocks the rewrite: {err}"
    );

    // Nothing was committed: the fragments the plan named are still there,
    // and the index still covers them.
    assert_eq!(dataset.get_fragments().len(), 4);
    let after = manifest_index(&dataset, "id_idx").await;
    assert_eq!(after.uuid, before.uuid);
    assert_eq!(after.fragment_bitmap, before.fragment_bitmap);
}
