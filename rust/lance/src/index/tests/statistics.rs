// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::support::*;
use super::*;
use serde_json::json;

#[tokio::test]
async fn test_bitmap_index_statistics_minimal_io_via_dataset() {
    const NUM_ROWS: usize = 500_000;
    let test_dir = TempStrDir::default();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "status",
        DataType::Int32,
        false,
    )]));
    let values: Vec<i32> = (0..NUM_ROWS as i32).collect();
    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(values))]).unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

    let mut dataset = Dataset::write(reader, &test_dir, None).await.unwrap();
    let io_tracker = dataset.object_store.as_ref().io_tracker().clone();

    let params = ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap);
    dataset
        .create_index(
            &["status"],
            IndexType::Bitmap,
            Some("status_idx".to_string()),
            &params,
            true,
        )
        .await
        .unwrap();

    let indices = dataset.load_indices().await.unwrap();
    let index_meta = indices
        .iter()
        .find(|idx| idx.name == "status_idx")
        .expect("status_idx should exist");
    let lookup_path = dataset
        .indice_files_dir(index_meta)
        .unwrap()
        .join(index_meta.uuid.to_string())
        .join(BITMAP_LOOKUP_NAME);
    let meta = dataset.object_store.inner.head(&lookup_path).await.unwrap();
    assert!(
        meta.size >= 1_000_000,
        "bitmap index should be large enough to fail without metadata path, size={} bytes",
        meta.size
    );

    // Reset stats collected during index creation
    io_tracker.incremental_stats();

    dataset.index_statistics("status_idx").await.unwrap();

    let stats = io_tracker.incremental_stats();
    assert_io_eq!(
        stats,
        read_bytes,
        4096,
        "index_statistics should only read the index footer; got {} bytes",
        stats.read_bytes
    );
    assert_io_lt!(
        stats,
        read_iops,
        3,
        "index_statistics should only require a head plus one range read; got {} ops",
        stats.read_iops
    );
    assert_io_eq!(
        stats,
        written_bytes,
        0,
        "index_statistics should not perform writes"
    );
}

/// Statistics for an index that is still a definition report no coverage.
///
/// The same shape the scalar side reports: the index is listed, indexed
/// rows are zero, and every row counts as unindexed.
#[tokio::test]
async fn test_statistics_for_a_definition_only_index() {
    let test_dir = tempfile::tempdir().unwrap();
    let rows = 100;
    let mut dataset = small_vector_dataset(test_dir.path(), rows).await;

    let params = VectorIndexParams::ivf_pq(10, 8, 4, DistanceType::L2, 1);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, false)
        .await
        .unwrap();

    let stats: serde_json::Value =
        serde_json::from_str(&dataset.index_statistics("vector_idx").await.unwrap()).unwrap();
    assert_eq!(stats["num_indexed_rows"].as_u64(), Some(0));
    assert_eq!(stats["num_unindexed_rows"].as_u64(), Some(rows as u64));
}

#[tokio::test]
async fn test_index_created_at_timestamp() {
    // Test that created_at is set when creating an index
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("values", DataType::Utf8, false),
    ]));

    let values = StringArray::from_iter_values(["hello", "world", "foo", "bar"]);
    let record_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..4)),
            Arc::new(values),
        ],
    )
    .unwrap();

    let reader = RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema.clone());

    let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

    // Record time before creating index
    let before_index = chrono::Utc::now();

    // Create a scalar index
    dataset
        .create_index(
            &["values"],
            IndexType::Scalar,
            Some("test_idx".to_string()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();

    // Record time after creating index
    let after_index = chrono::Utc::now();

    // Get index metadata
    let indices = dataset.load_indices().await.unwrap();
    let test_index = indices.iter().find(|idx| idx.name == "test_idx").unwrap();

    // Verify created_at is set and within reasonable bounds
    assert!(test_index.created_at.is_some());
    let created_at = test_index.created_at.unwrap();
    assert!(created_at >= before_index);
    assert!(created_at <= after_index);
}

#[tokio::test]
async fn test_index_statistics_updated_at() {
    // Test that updated_at appears in index statistics
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("values", DataType::Utf8, false),
    ]));

    let values = StringArray::from_iter_values(["hello", "world", "foo", "bar"]);
    let record_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..4)),
            Arc::new(values),
        ],
    )
    .unwrap();

    let reader = RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema.clone());

    let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

    // Create a scalar index
    dataset
        .create_index(
            &["values"],
            IndexType::Scalar,
            Some("test_idx".to_string()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();

    // Get index statistics
    let stats_str = dataset.index_statistics("test_idx").await.unwrap();
    let stats: serde_json::Value = serde_json::from_str(&stats_str).unwrap();

    // Verify updated_at_timestamp_ms field exists in statistics
    assert!(stats["updated_at_timestamp_ms"].is_number());
    let updated_at = stats["updated_at_timestamp_ms"].as_u64().unwrap();

    // Get the index metadata to compare with created_at
    let indices = dataset.load_indices().await.unwrap();
    let test_index = indices.iter().find(|idx| idx.name == "test_idx").unwrap();
    let created_at = test_index.created_at.unwrap().timestamp_millis() as u64;

    // For a single index, updated_at should equal created_at
    assert_eq!(updated_at, created_at);
}

#[tokio::test]
async fn test_index_statistics_updated_at_multiple_deltas() {
    // Test that updated_at reflects max(created_at) across multiple delta indices
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
            false,
        ),
    ]));

    // Create initial dataset (need more rows for PQ training)
    let num_rows = 300;
    let float_arr = generate_random_array(4 * num_rows);
    let vectors = FixedSizeListArray::try_new_from_values(float_arr, 4).unwrap();
    let record_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..num_rows as i32)),
            Arc::new(vectors),
        ],
    )
    .unwrap();

    let reader = RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema.clone());

    let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

    // Create vector index
    let params = VectorIndexParams::ivf_pq(1, 8, 2, MetricType::L2, 2);
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("test_vec_idx".to_string()),
            &params,
            false,
        )
        .await
        .unwrap();

    // Get initial statistics
    let stats_str_1 = dataset.index_statistics("test_vec_idx").await.unwrap();
    let stats_1: serde_json::Value = serde_json::from_str(&stats_str_1).unwrap();
    let initial_updated_at = stats_1["updated_at_timestamp_ms"].as_u64().unwrap();

    // Add more data to create additional delta indices
    std::thread::sleep(std::time::Duration::from_millis(10)); // Ensure different timestamp

    let num_rows_2 = 50;
    let float_arr_2 = generate_random_array(4 * num_rows_2);
    let vectors_2 = FixedSizeListArray::try_new_from_values(float_arr_2, 4).unwrap();
    let record_batch_2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(
                num_rows as i32..(num_rows + num_rows_2) as i32,
            )),
            Arc::new(vectors_2),
        ],
    )
    .unwrap();

    let reader_2 =
        RecordBatchIterator::new(vec![record_batch_2].into_iter().map(Ok), schema.clone());

    dataset.append(reader_2, None).await.unwrap();

    // Update the index
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("test_vec_idx".to_string()),
            &params,
            true,
        )
        .await
        .unwrap();

    // Get updated statistics
    let stats_str_2 = dataset.index_statistics("test_vec_idx").await.unwrap();
    let stats_2: serde_json::Value = serde_json::from_str(&stats_str_2).unwrap();
    let final_updated_at = stats_2["updated_at_timestamp_ms"].as_u64().unwrap();

    // The updated_at should be newer than the initial one
    assert!(final_updated_at >= initial_updated_at);
}

#[tokio::test]
async fn test_index_statistics_updated_at_none_when_no_created_at() {
    // Test backward compatibility: when indices were created before the created_at field,
    // updated_at_timestamp_ms should be null

    // Use test data created with Lance 0.29.0 (before created_at field was added)
    let test_dir =
        copy_test_data_to_tmp("v0.30.0_pre_created_at/index_without_created_at").unwrap();
    let test_uri = test_dir.path_str();
    let test_uri = &test_uri;

    let dataset = Dataset::open(test_uri).await.unwrap();

    // Get the index metadata to verify created_at is None
    let indices = dataset.load_indices().await.unwrap();
    assert!(!indices.is_empty(), "Test dataset should have indices");

    // Verify that the index created with old version has no created_at
    for index in indices.iter() {
        assert!(
            index.created_at.is_none(),
            "Index from old version should have created_at = None"
        );
    }

    // Get index statistics - should work even with old indices
    let index_name = &indices[0].name;
    let stats_str = dataset.index_statistics(index_name).await.unwrap();
    let stats: serde_json::Value = serde_json::from_str(&stats_str).unwrap();

    // Verify updated_at_timestamp_ms field is null when no indices have created_at
    assert!(
        stats["updated_at_timestamp_ms"].is_null(),
        "updated_at_timestamp_ms should be null when no indices have created_at timestamps"
    );
}

#[tokio::test]
async fn test_describe_indices_tolerates_missing_index_details() {
    // An index whose manifest entry has no index details (e.g. created
    // before details were persisted) is still described on a best-effort
    // basis rather than causing describe_indices to error.
    use lance_datagen::{BatchCount, RowCount, array};

    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();
    let reader = lance_datagen::gen_batch()
        .col("id", array::step::<arrow_array::types::Int32Type>())
        .into_reader_rows(RowCount::from(10), BatchCount::from(1));
    let dataset = Dataset::write(reader, test_uri, None).await.unwrap();
    let field_id = dataset.schema().field("id").unwrap().id;

    let metadata = IndexMetadata {
        uuid: Uuid::new_v4(),
        name: "mystery_idx".to_string(),
        fields: vec![field_id],
        covering_fields: vec![],
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(std::iter::once(0_u32).collect()),
        index_details: None,
        index_version: 0,
        created_at: None,
        base_id: None,
        files: None,
        clustering_id: None,
        metadata: Default::default(),
    };

    let desc = IndexDescriptionImpl::try_new(vec![metadata], &dataset)
        .await
        .unwrap();
    assert_eq!(desc.index_type(), "Unknown");
    assert_eq!(desc.type_url(), "");
    assert_eq!(desc.details().unwrap(), "{}");
    assert_eq!(desc.rows_indexed(), 10);
}

#[tokio::test]
async fn test_describe_indices_fragment_coverage_unions_segments() {
    let reader = gen_batch()
        .col("id", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(8), BatchCount::from(1));
    let dataset = Dataset::write(
        reader,
        "memory://fragment-coverage-union",
        Some(WriteParams {
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.fragments().len(), 4);

    let field_id = dataset.schema().field("id").unwrap().id;
    let index_details =
        Arc::new(prost_types::Any::from_msg(&BTreeIndexDetails::default()).unwrap());
    let segment = |fragment_ids: &[u32]| IndexMetadata {
        uuid: Uuid::new_v4(),
        name: "id_idx".to_string(),
        fields: vec![field_id],
        covering_fields: vec![],
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(fragment_ids.iter().copied().collect()),
        index_details: Some(index_details.clone()),
        index_version: IndexType::BTree.version(),
        created_at: None,
        base_id: None,
        files: None,
        clustering_id: None,
        metadata: Default::default(),
    };

    // Query-time fragment reuse can make two segment masks overlap. Union
    // them before comparing with the current dataset fragment ids.
    let first = segment(&[0, 1, 99]);
    let second = segment(&[1, 2]);
    let expected_bitmap_size = u64::try_from(first.fragment_bitmap_serialized_size().unwrap())
        .unwrap()
        + u64::try_from(second.fragment_bitmap_serialized_size().unwrap()).unwrap();
    let description = IndexDescriptionImpl::try_new(vec![first, second], &dataset)
        .await
        .unwrap();

    assert_eq!(
        description.fragment_coverage(),
        Some(&IndexFragmentCoverage {
            covered_fragment_count: 3,
            current_fragment_count: 4,
            missing_fragment_count: 1,
            stale_fragment_count: 1,
            fragment_bitmap_size_bytes: expected_bitmap_size,
        })
    );
}

#[tokio::test]
async fn test_describe_indices_derives_type_from_url_without_plugin() {
    // When index details exist but no plugin is registered for the type
    // URL, the index type is derived from the type URL rather than being
    // reported as the opaque "Unknown".
    use lance_datagen::{BatchCount, RowCount, array};

    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();
    let reader = lance_datagen::gen_batch()
        .col("id", array::step::<arrow_array::types::Int32Type>())
        .into_reader_rows(RowCount::from(10), BatchCount::from(1));
    let dataset = Dataset::write(reader, test_uri, None).await.unwrap();
    let field_id = dataset.schema().field("id").unwrap().id;

    let metadata = IndexMetadata {
        uuid: Uuid::new_v4(),
        name: "mystery_idx".to_string(),
        fields: vec![field_id],
        covering_fields: vec![],
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(std::iter::once(0_u32).collect()),
        index_details: Some(Arc::new(prost_types::Any {
            type_url: "/lance.index.pb.MysteryIndexDetails".to_string(),
            value: Vec::new(),
        })),
        index_version: 0,
        created_at: None,
        base_id: None,
        files: None,
        clustering_id: None,
        metadata: Default::default(),
    };

    let desc = IndexDescriptionImpl::try_new(vec![metadata], &dataset)
        .await
        .unwrap();
    assert_eq!(desc.index_type(), "Mystery");
    assert_eq!(desc.type_url(), "/lance.index.pb.MysteryIndexDetails");
}

#[tokio::test]
async fn test_describe_indices_unknown_type_empty_bitmap_has_unknown_coverage() {
    let reader = gen_batch()
        .col("id", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(10), BatchCount::from(1));
    let dataset = Dataset::write(reader, "memory://", None).await.unwrap();
    let field_id = dataset.schema().field("id").unwrap().id;
    let metadata = IndexMetadata {
        uuid: Uuid::new_v4(),
        name: "mystery_idx".to_string(),
        fields: vec![field_id],
        covering_fields: vec![],
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(RoaringBitmap::new()),
        index_details: Some(Arc::new(prost_types::Any {
            type_url: "/lance.index.pb.MysteryIndexDetails".to_string(),
            value: Vec::new(),
        })),
        index_version: 0,
        created_at: None,
        base_id: None,
        files: None,
        clustering_id: None,
        metadata: Default::default(),
    };

    let description = IndexDescriptionImpl::try_new(vec![metadata], &dataset)
        .await
        .unwrap();
    assert_eq!(description.index_type(), "Mystery");
    assert!(
        description.fragment_coverage().is_none(),
        "a missing plugin cannot make an empty bitmap authoritative coverage"
    );
}

#[tokio::test]
async fn test_describe_indices_empty_vector_bitmap_has_known_zero_coverage() {
    let reader = gen_batch()
        .col("id", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(10), BatchCount::from(1));
    let dataset = Dataset::write(reader, "memory://", None).await.unwrap();
    let field_id = dataset.schema().field("id").unwrap().id;
    let metadata = IndexMetadata {
        uuid: Uuid::new_v4(),
        name: "vector_idx".to_string(),
        fields: vec![field_id],
        covering_fields: vec![],
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(RoaringBitmap::new()),
        index_details: Some(Arc::new(vector_index_details_default())),
        index_version: IndexType::Vector.version(),
        created_at: None,
        base_id: None,
        files: None,
        clustering_id: None,
        metadata: Default::default(),
    };
    let expected_bitmap_size =
        u64::try_from(metadata.fragment_bitmap_serialized_size().unwrap()).unwrap();

    let description = IndexDescriptionImpl::try_new(vec![metadata], &dataset)
        .await
        .unwrap();
    assert_eq!(
        description.fragment_coverage(),
        Some(&IndexFragmentCoverage {
            covered_fragment_count: 0,
            current_fragment_count: 1,
            missing_fragment_count: 1,
            stale_fragment_count: 0,
            fragment_bitmap_size_bytes: expected_bitmap_size,
        })
    );
}

/// `fields` lists the carried columns too, but a logical description must
/// advertise only what the index can be searched on. Every binding reads this
/// list as "the index's columns" -- Python's `_default_vector_index_for_column`
/// matches on membership -- so a carried column here hands back the keyed
/// column's model for a query about the carried one.
#[tokio::test]
async fn test_index_description_reports_only_keyed_fields() {
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
    metadata.covering_fields = vec![id_field_id];

    let description = IndexDescriptionImpl::try_new(vec![metadata], &dataset)
        .await
        .unwrap();

    assert_eq!(
        description.field_ids(),
        &[vector_field_id as u32],
        "a covered index must advertise only its keyed column"
    );
}

#[tokio::test]
async fn test_scalar_index_file_sizes_captured() {
    // Test that file sizes are captured when creating a scalar index
    let reader = gen_batch()
        .col("id", array::step::<arrow_array::types::Int32Type>())
        .col("values", array::rand_utf8(ByteCount::from(10), false))
        .into_reader_rows(RowCount::from(4), BatchCount::from(1));

    let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

    // Create a scalar index
    dataset
        .create_index(
            &["values"],
            IndexType::Scalar,
            Some("test_idx".to_string()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();

    // Get index metadata and verify files are populated
    let indices = dataset.load_indices().await.unwrap();
    let test_index = indices.iter().find(|idx| idx.name == "test_idx").unwrap();

    assert!(
        test_index.files.is_some(),
        "Index should have files populated"
    );
    let files = test_index.files.as_ref().unwrap();
    assert!(!files.is_empty(), "Index should have at least one file");

    // Verify each file has a positive size
    for file in files {
        assert!(
            file.size_bytes > 0,
            "File {} should have positive size",
            file.path
        );
    }

    // Verify total_size_bytes works
    let total_size = test_index.total_size_bytes();
    assert!(total_size.is_some(), "total_size_bytes should return Some");
    assert!(total_size.unwrap() > 0, "Total size should be positive");
}

#[tokio::test]
async fn test_vector_index_file_sizes_captured() {
    // Test that file sizes are captured when creating a vector index
    let reader = gen_batch()
        .col("id", array::step::<arrow_array::types::Int32Type>())
        .col(
            "vector",
            array::rand_vec::<arrow_array::types::Float32Type>(4.into()),
        )
        .into_reader_rows(RowCount::from(300), BatchCount::from(1));

    let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

    // Create vector index
    let params = VectorIndexParams::ivf_pq(1, 8, 2, MetricType::L2, 2);
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("test_vec_idx".to_string()),
            &params,
            false,
        )
        .await
        .unwrap();

    // Get index metadata and verify files are populated
    let indices = dataset.load_indices().await.unwrap();
    let test_index = indices
        .iter()
        .find(|idx| idx.name == "test_vec_idx")
        .unwrap();

    assert!(
        test_index.files.is_some(),
        "Index should have files populated"
    );
    let files = test_index.files.as_ref().unwrap();
    assert!(!files.is_empty(), "Index should have at least one file");

    // Verify each file has a positive size
    for file in files {
        assert!(
            file.size_bytes > 0,
            "File {} should have positive size",
            file.path
        );
    }

    // Verify total_size_bytes works
    let total_size = test_index.total_size_bytes();
    assert!(total_size.is_some(), "total_size_bytes should return Some");
    assert!(total_size.unwrap() > 0, "Total size should be positive");
}

#[tokio::test]
async fn test_describe_indices_total_size() {
    // Test that describe_indices returns total_size_bytes
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("values", DataType::Utf8, false),
    ]));

    let values = StringArray::from_iter_values(["hello", "world", "foo", "bar"]);
    let record_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..4)),
            Arc::new(values),
        ],
    )
    .unwrap();

    let reader = RecordBatchIterator::new(vec![record_batch].into_iter().map(Ok), schema.clone());

    let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

    // Create a scalar index
    dataset
        .create_index(
            &["values"],
            IndexType::Scalar,
            Some("test_idx".to_string()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();

    // Use describe_indices to get index info
    let descriptions = dataset.describe_indices(None).await.unwrap();
    assert_eq!(descriptions.len(), 1);

    let desc = &descriptions[0];
    assert_eq!(desc.name(), "test_idx");
    assert_eq!(desc.rows_indexed(), 4);

    // Verify total_size_bytes is available
    let total_size = desc.total_size_bytes();
    assert!(total_size.is_some(), "total_size_bytes should be Some");
    assert!(total_size.unwrap() > 0, "Total size should be positive");
}

#[tokio::test]
async fn test_describe_indices_rows_indexed_multi_fragment() {
    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();

    let reader = lance_datagen::gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("values", array::rand_utf8(ByteCount::from(8), false))
        .into_reader_rows(RowCount::from(10), BatchCount::from(3));

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
    assert_eq!(dataset.fragments().len(), 3);

    dataset
        .create_index(
            &["values"],
            IndexType::Scalar,
            Some("multi_frag_idx".to_string()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();

    let descriptions = dataset.describe_indices(None).await.unwrap();
    assert_eq!(descriptions.len(), 1);
    assert_eq!(descriptions[0].name(), "multi_frag_idx");
    assert_eq!(
        descriptions[0].rows_indexed(),
        30,
        "rows_indexed should sum logical rows across all indexed fragments"
    );
}

#[tokio::test]
async fn test_describe_indices_rows_indexed_with_deletions() {
    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();

    let reader = lance_datagen::gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("values", array::rand_utf8(ByteCount::from(8), false))
        .into_reader_rows(RowCount::from(20), BatchCount::from(1));

    let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();
    dataset.delete("id >= 15").await.unwrap();
    assert_eq!(dataset.count_rows(None).await.unwrap(), 15);

    dataset
        .create_index(
            &["values"],
            IndexType::Scalar,
            Some("deleted_rows_idx".to_string()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();

    let descriptions = dataset.describe_indices(None).await.unwrap();
    assert_eq!(descriptions.len(), 1);
    assert_eq!(descriptions[0].name(), "deleted_rows_idx");
    assert_eq!(
        descriptions[0].rows_indexed(),
        15,
        "rows_indexed should use logical rows (physical_rows - num_deleted_rows)"
    );
}

#[tokio::test]
async fn test_describe_indices_rows_indexed_stale_bitmap_fragment() {
    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();

    let reader = lance_datagen::gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("vector", array::rand_vec::<Float32Type>(8.into()))
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
    assert_eq!(dataset.fragments().len(), 2);

    let field_id = dataset.schema().field("vector").unwrap().id;
    let stale_segment = write_vector_segment_metadata(
        &dataset,
        "vector_idx",
        field_id,
        Uuid::new_v4(),
        [0_u32, 1_u32, 999_u32],
        b"stale-bitmap-segment",
    )
    .await;

    dataset
        .commit_existing_index_segments(
            "vector_idx",
            "vector",
            vec![segment_from_metadata(&stale_segment)],
        )
        .await
        .unwrap();

    let descriptions = dataset.describe_indices(None).await.unwrap();
    assert_eq!(descriptions.len(), 1);
    assert_eq!(descriptions[0].name(), "vector_idx");
    assert_eq!(
        descriptions[0].rows_indexed(),
        20,
        "stale bitmap entries for missing fragments should be skipped without failing"
    );
    assert_eq!(
        descriptions[0].fragment_coverage(),
        Some(&IndexFragmentCoverage {
            covered_fragment_count: 2,
            current_fragment_count: 2,
            missing_fragment_count: 0,
            stale_fragment_count: 1,
            fragment_bitmap_size_bytes: u64::try_from(
                stale_segment.fragment_bitmap_serialized_size().unwrap(),
            )
            .unwrap(),
        })
    );
}

#[tokio::test]
async fn test_scalar_index_create_does_not_list_files() {
    let test_dir = TempStrDir::default();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("category", DataType::Int32, false),
    ]));
    let ids = Int32Array::from_iter_values(0..128);
    let categories = Int32Array::from_iter_values((0..128).map(|value| value % 8));
    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(categories)]).unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let mut dataset = Dataset::write(reader, test_dir.as_str(), None)
        .await
        .unwrap();
    let io_tracker = dataset.object_store.as_ref().io_tracker().clone();

    io_tracker.incremental_stats();
    dataset
        .create_index(
            &["category"],
            IndexType::Bitmap,
            Some("category_bitmap".to_string()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();

    let stats = io_tracker.incremental_stats();
    let list_stats = list_io_stats(&stats);
    assert_io_eq!(
        list_stats,
        read_iops,
        0,
        "new scalar index files should be reported by writer return values"
    );
}

#[tokio::test]
async fn test_vector_index_create_does_not_list_files() {
    let test_dir = TempStrDir::default();
    let dimension = 8;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dimension,
            ),
            false,
        ),
    ]));
    let ids = Int32Array::from_iter_values(0..256);
    let vectors = (0..256)
        .map(|row| {
            Some(
                (0..dimension)
                    .map(|dim| Some((row * dimension + dim) as f32))
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let vector_array =
        FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(vectors, dimension);
    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(vector_array)]).unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let mut dataset = Dataset::write(reader, test_dir.as_str(), None)
        .await
        .unwrap();
    let io_tracker = dataset.object_store.as_ref().io_tracker().clone();

    io_tracker.incremental_stats();
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("vector_ivf_flat".to_string()),
            &VectorIndexParams::ivf_flat(4, MetricType::L2),
            true,
        )
        .await
        .unwrap();

    let stats = io_tracker.incremental_stats();
    let list_stats = list_io_stats(&stats);
    assert_io_eq!(
        list_stats,
        read_iops,
        0,
        "new V3 vector index files should be reported by builder return values"
    );
}

#[tokio::test]
async fn test_index_file_sizes_through_lifecycle() {
    use crate::dataset::WriteDestination;
    use crate::dataset::optimize::{CompactionOptions, compact_files, remapping};
    use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;

    // Create initial dataset with columns for different index types
    let data = gen_batch()
        .col("int_col", array::step::<Int32Type>())
        .col("str_col", array::rand_utf8(8.into(), false))
        .col(
            "vec_col",
            array::rand_vec::<Float32Type>(Dimension::from(32)),
        )
        .into_reader_rows(RowCount::from(1000), BatchCount::from(1));

    let test_dir = TempStrDir::default();
    let mut dataset = Dataset::write(
        data,
        test_dir.as_str(),
        Some(WriteParams {
            max_rows_per_file: 200, // Multiple fragments for compaction
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    // Create BTree index
    dataset
        .create_index(
            &["int_col"],
            IndexType::BTree,
            Some("btree_idx".to_string()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();

    // Create Bitmap index
    dataset
        .create_index(
            &["int_col"],
            IndexType::Bitmap,
            Some("bitmap_idx".to_string()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();

    // Create Inverted index for text search
    dataset
        .create_index(
            &["str_col"],
            IndexType::Inverted,
            Some("inverted_idx".to_string()),
            &InvertedIndexParams::default(),
            false,
        )
        .await
        .unwrap();

    // Validate files are populated after creation
    assert_all_indices_have_files(&dataset, "after initial creation").await;

    // Append more data
    let more_data = gen_batch()
        .col("int_col", array::step::<Int32Type>())
        .col("str_col", array::rand_utf8(8.into(), false))
        .col(
            "vec_col",
            array::rand_vec::<Float32Type>(Dimension::from(32)),
        )
        .into_reader_rows(RowCount::from(500), BatchCount::from(1));

    Dataset::write(
        more_data,
        WriteDestination::Dataset(Arc::new(dataset.clone())),
        Some(WriteParams {
            max_rows_per_file: 200,
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    dataset = DatasetBuilder::from_uri(test_dir.as_str())
        .load()
        .await
        .unwrap();

    // Optimize indices (triggers update/merge)
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();

    // Validate files are still populated after optimize
    assert_all_indices_have_files(&dataset, "after optimize_indices").await;

    // Run compaction with deferred remap
    let options = CompactionOptions {
        target_rows_per_fragment: 500,
        defer_index_remap: true,
        ..Default::default()
    };

    compact_files(&mut dataset, options.clone(), None)
        .await
        .unwrap();

    // Check if frag reuse index exists (indicates remap is needed)
    if dataset
        .load_index_by_name(FRAG_REUSE_INDEX_NAME)
        .await
        .unwrap()
        .is_some()
    {
        // Remap each index
        remapping::remap_column_index(&mut dataset, &["int_col"], Some("btree_idx".to_string()))
            .await
            .unwrap();

        remapping::remap_column_index(&mut dataset, &["int_col"], Some("bitmap_idx".to_string()))
            .await
            .unwrap();

        remapping::remap_column_index(&mut dataset, &["str_col"], Some("inverted_idx".to_string()))
            .await
            .unwrap();

        // Validate files are populated after remap
        assert_all_indices_have_files(&dataset, "after remap").await;
    }
}

#[tokio::test]
async fn test_btree_index_iops() {
    // Test that querying a BTree index uses minimal IOPs (no HEAD requests)
    let test_dir = TempStrDir::default();

    // Create dataset with a column suitable for BTree index
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, false),
    ]));

    let num_rows = 1000;
    let ids = Int32Array::from_iter_values(0..num_rows);
    let values = Int32Array::from_iter_values((0..num_rows).map(|i| i % 100));

    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(values)]).unwrap();

    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
    let mut dataset = Dataset::write(reader, test_dir.as_str(), None)
        .await
        .unwrap();

    // Create BTree index
    dataset
        .create_index(
            &["value"],
            IndexType::BTree,
            Some("btree_idx".to_string()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();

    // Re-open dataset fresh to avoid cached state
    let dataset = DatasetBuilder::from_uri(test_dir.as_str())
        .load()
        .await
        .unwrap();

    // Reset IO stats before query
    let _ = dataset.object_store.as_ref().io_stats_incremental();

    // Query using the BTree index
    let results = dataset
        .scan()
        .filter("value = 50")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert!(results.num_rows() > 0);

    // Verify IOPs - should be minimal (no HEAD requests)
    let stats = dataset.object_store.as_ref().io_stats_incremental();
    // We expect reads for: index metadata + index pages + data files
    // The key assertion is that we don't have extra HEAD requests
    assert_io_lt!(
        stats,
        read_iops,
        10,
        "BTree index query should use minimal IOPs"
    );
}

#[tokio::test]
async fn test_bitmap_index_iops() {
    // Test that querying a Bitmap index uses minimal IOPs (no HEAD requests)
    let test_dir = TempStrDir::default();

    // Create dataset with low-cardinality column for Bitmap index
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("category", DataType::Int32, false),
    ]));

    let num_rows = 1000;
    let ids = Int32Array::from_iter_values(0..num_rows);
    // Low cardinality - only 10 unique values
    let categories = Int32Array::from_iter_values((0..num_rows).map(|i| i % 10));

    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(categories)]).unwrap();

    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
    let mut dataset = Dataset::write(reader, test_dir.as_str(), None)
        .await
        .unwrap();

    // Create Bitmap index
    dataset
        .create_index(
            &["category"],
            IndexType::Bitmap,
            Some("bitmap_idx".to_string()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();

    // Re-open dataset fresh
    let dataset = DatasetBuilder::from_uri(test_dir.as_str())
        .load()
        .await
        .unwrap();

    // Reset IO stats before query
    let _ = dataset.object_store.as_ref().io_stats_incremental();

    // Query using the Bitmap index
    let results = dataset
        .scan()
        .filter("category = 5")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert!(results.num_rows() > 0);

    // Verify IOPs
    let stats = dataset.object_store.as_ref().io_stats_incremental();
    assert_io_lt!(
        stats,
        read_iops,
        10,
        "Bitmap index query should use minimal IOPs"
    );
}

#[tokio::test]
async fn test_inverted_index_iops() {
    // Test that querying an Inverted (FTS) index uses minimal IOPs
    let test_dir = TempStrDir::default();

    // Create dataset with text column for Inverted index
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("text", DataType::Utf8, false),
    ]));

    let num_rows = 100;
    let ids = Int32Array::from_iter_values(0..num_rows);
    let texts = StringArray::from_iter_values((0..num_rows).map(|i| {
        if i % 3 == 0 {
            format!("hello world document {}", i)
        } else if i % 3 == 1 {
            format!("goodbye universe text {}", i)
        } else {
            format!("random content item {}", i)
        }
    }));

    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(texts)]).unwrap();

    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
    let mut dataset = Dataset::write(reader, test_dir.as_str(), None)
        .await
        .unwrap();

    // Create Inverted index
    let params = InvertedIndexParams::default();
    dataset
        .create_index(
            &["text"],
            IndexType::Inverted,
            Some("inverted_idx".to_string()),
            &params,
            true,
        )
        .await
        .unwrap();

    // Re-open dataset fresh
    let dataset = DatasetBuilder::from_uri(test_dir.as_str())
        .load()
        .await
        .unwrap();

    // Reset IO stats before query
    let _ = dataset.object_store.as_ref().io_stats_incremental();

    // Query using the Inverted index (full-text search)
    let results = dataset
        .scan()
        .full_text_search(FullTextSearchQuery::new("hello".to_string()))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert!(results.num_rows() > 0);

    // Verify IOPs. The deferred DocSet loads per-doc num_tokens/row_ids on
    // first use rather than eagerly at index open, so a cold (un-prewarmed)
    // query opens the docs file on demand — a couple more IOPs than the
    // eager path, but constant and only on the first query (prewarm or a
    // warm cache serve it with zero IO).
    let stats = dataset.object_store.as_ref().io_stats_incremental();
    assert_io_lt!(
        stats,
        read_iops,
        18,
        "Inverted index query should use minimal IOPs"
    );
}

#[tokio::test]
async fn test_ivf_pq_index_iops() {
    // Test that querying an IVF_PQ vector index uses minimal IOPs
    let test_dir = TempStrDir::default();

    // Create dataset with vector column
    let dimension = 32;
    let num_rows = 1000;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dimension,
            ),
            false,
        ),
    ]));

    let ids = Int32Array::from_iter_values(0..num_rows);
    let vectors: Vec<Option<Vec<Option<f32>>>> = (0..num_rows)
        .map(|i| {
            Some(
                (0..dimension)
                    .map(|j| Some((i * dimension + j) as f32 / 1000.0))
                    .collect(),
            )
        })
        .collect();
    let vector_array =
        FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(vectors, dimension);

    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(vector_array)]).unwrap();

    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
    let mut dataset = Dataset::write(reader, test_dir.as_str(), None)
        .await
        .unwrap();

    // Create IVF_PQ index
    let params = VectorIndexParams::ivf_pq(4, 8, 4, MetricType::L2, 50);
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("ivf_pq_idx".to_string()),
            &params,
            true,
        )
        .await
        .unwrap();

    // Re-open dataset fresh
    let dataset = DatasetBuilder::from_uri(test_dir.as_str())
        .load()
        .await
        .unwrap();

    // Do a full scan to warm up data file metadata
    let _ = dataset.scan().try_into_batch().await.unwrap();

    // Reset IO stats before query
    let _ = dataset.object_store.as_ref().io_stats_incremental();

    // Query using the IVF_PQ index (KNN search)
    let query_vector: Vec<f32> = (0..dimension).map(|i| i as f32 / 1000.0).collect();
    let results = dataset
        .scan()
        .nearest("vector", &Float32Array::from(query_vector), 10)
        .unwrap()
        .minimum_nprobes(2)
        .maximum_nprobes(2)
        .try_into_batch()
        .await
        .unwrap();
    assert!(results.num_rows() > 0);

    // Verify IOPs
    let stats = dataset.object_store.as_ref().io_stats_incremental();
    assert_io_lt!(
        stats,
        read_iops,
        17,
        "IVF_PQ index query should use minimal IOPs"
    );
}

#[tokio::test]
async fn test_describe_indices_returns_correct_vector_index_type() {
    const DIM: i32 = 8;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), DIM),
            true,
        ),
    ]));

    let data = generate_random_array(256 * DIM as usize);
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..256)),
            Arc::new(FixedSizeListArray::try_new_from_values(data, DIM).unwrap()),
        ],
    )
    .unwrap();

    let test_dir = TempStrDir::default();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let mut dataset = Dataset::write(reader, &test_dir, None).await.unwrap();

    // Create IVF_FLAT index
    let params = VectorIndexParams::ivf_flat(2, MetricType::L2);
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("vector_idx".to_string()),
            &params,
            true,
        )
        .await
        .unwrap();

    // Reload dataset and call describe_indices
    let dataset = Dataset::open(&test_dir).await.unwrap();
    let descriptions = dataset.describe_indices(None).await.unwrap();

    assert_eq!(descriptions.len(), 1);
    let desc = &descriptions[0];
    assert_eq!(desc.name(), "vector_idx");
    // This should be "IVF_FLAT", not "Unknown"
    assert_eq!(desc.index_type(), "IVF_FLAT");
    assert!(!desc.field_ids().is_empty());
}

#[rstest]
#[case::empty_first(true)]
#[case::empty_last(false)]
#[tokio::test]
async fn test_index_statistics_from_segments(#[case] is_empty_first: bool) {
    let test_dir = TempStrDir::default();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "status",
        DataType::Int32,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from_iter_values(0..8))],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], schema.clone());
    let mut dataset = Dataset::write(
        reader,
        &test_dir,
        Some(WriteParams {
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let fragment_ids: Vec<_> = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    let params = ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap);
    let empty_segment =
        CreateIndexBuilder::new(&mut dataset, &["status"], IndexType::Bitmap, &params)
            .name("status_idx".to_string())
            .fragments(Vec::new())
            .execute_uncommitted()
            .await
            .unwrap();
    assert!(empty_segment.fragment_bitmap.as_ref().unwrap().is_empty());
    let empty_uuid = empty_segment.uuid;
    let mut segments = Vec::new();
    for fragment_id in fragment_ids {
        segments.push(
            CreateIndexBuilder::new(&mut dataset, &["status"], IndexType::Bitmap, &params)
                .name("status_idx".to_string())
                .fragments(vec![fragment_id])
                .execute_uncommitted()
                .await
                .unwrap(),
        );
    }
    if is_empty_first {
        segments.insert(0, empty_segment);
    } else {
        segments.push(empty_segment);
    }
    dataset
        .commit_existing_index_segments("status_idx", "status", segments.clone())
        .await
        .unwrap();
    assert_eq!(segments.len(), 5);

    let index_uuids: Vec<_> = segments.iter().rev().map(|segment| segment.uuid).collect();
    let mut statistics = dataset
        .index_segment_statistics("status_idx", &index_uuids)
        .await
        .unwrap();
    assert_eq!(
        statistics
            .iter()
            .map(|statistics| statistics.index_uuid)
            .collect::<Vec<_>>(),
        index_uuids
    );
    let empty_statistics = statistics
        .iter()
        .find(|statistics| statistics.index_uuid == empty_uuid)
        .unwrap();
    assert!(empty_statistics.index_type_uri.is_empty());
    assert_eq!(empty_statistics.statistics, json!({}));
    for result in &statistics {
        let encoded = serde_json::to_string(&result).unwrap();
        assert_eq!(
            serde_json::from_str::<IndexSegmentStatistics>(&encoded).unwrap(),
            *result
        );
    }
    let expected: serde_json::Value =
        serde_json::from_str(&dataset.index_statistics("status_idx").await.unwrap()).unwrap();
    let actual: serde_json::Value = serde_json::from_str(
        &dataset
            .index_statistics_from_segments("status_idx", statistics.clone())
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual["index_type"], "Bitmap");
    assert_eq!(actual["num_segments"], 5);
    assert_eq!(actual["num_indexed_rows"], 8);
    assert_eq!(actual["num_unindexed_rows"], 0);
    let empty_position = usize::from(!is_empty_first) * 4;
    assert_eq!(actual["indices"][empty_position], json!({}));
    assert_eq!(actual["segments"][empty_position], json!({}));

    let mut duplicate = statistics.clone();
    duplicate[1] = duplicate[0].clone();
    let mut foreign = statistics.clone();
    foreign[0].index_uuid = Uuid::new_v4();
    let mut stale = statistics.clone();
    stale[0].read_version += 1;
    for (invalid, expected_message) in [
        (
            Vec::new(),
            "Expected 5 segment statistics for index status_idx, got 0",
        ),
        (
            statistics[..4].to_vec(),
            "Expected 5 segment statistics for index status_idx, got 4",
        ),
        (
            duplicate,
            "Duplicate segment statistics for index status_idx",
        ),
        (
            foreign,
            "does not belong to index status_idx at this snapshot",
        ),
        (stale, "statistics are from dataset version"),
    ] {
        let error = dataset
            .index_statistics_from_segments("status_idx", invalid)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains(expected_message), "{error}");
    }
    let unknown_uuid = Uuid::new_v4();
    let error = dataset
        .index_segment_statistics("status_idx", &[unknown_uuid])
        .await
        .unwrap_err();
    assert!(matches!(error, Error::IndexNotFound { .. }), "{error}");
    assert!(
        error
            .to_string()
            .contains(&format!("name=status_idx, uuid={unknown_uuid}")),
        "{error}"
    );
    for (invalid_uuids, expected_message) in [
        (Vec::new(), "At least one index segment UUID is required"),
        (
            vec![index_uuids[0], index_uuids[0]],
            "Duplicate index segment UUID",
        ),
    ] {
        let error = dataset
            .index_segment_statistics("status_idx", &invalid_uuids)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains(expected_message), "{error}");
    }
    let error = dataset
        .index_statistics_from_segments("missing", statistics.clone())
        .await
        .unwrap_err();
    assert!(matches!(error, Error::IndexNotFound { .. }), "{error}");
    assert!(error.to_string().contains("name=missing"), "{error}");

    dataset.delete("status = 0").await.unwrap();
    let index_uuids: Vec<_> = dataset
        .load_indices_by_name("status_idx")
        .await
        .unwrap()
        .iter()
        .map(|metadata| metadata.uuid)
        .collect();
    statistics.retain(|statistics| index_uuids.contains(&statistics.index_uuid));
    assert_eq!(statistics.len(), 4);
    let error = dataset
        .index_statistics_from_segments("status_idx", statistics)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(
        error
            .to_string()
            .contains("statistics are from dataset version"),
        "{error}"
    );
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let dataset = Dataset::write(
        reader,
        &test_dir,
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let statistics = dataset
        .index_segment_statistics("status_idx", &index_uuids)
        .await
        .unwrap();
    let expected: serde_json::Value =
        serde_json::from_str(&dataset.index_statistics("status_idx").await.unwrap()).unwrap();
    let actual: serde_json::Value = serde_json::from_str(
        &dataset
            .index_statistics_from_segments("status_idx", statistics)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual["num_indexed_rows"], 7);
    assert_eq!(actual["num_unindexed_rows"], 8);
}

#[tokio::test]
async fn test_index_statistics_from_segments_rejects_system_index() {
    let data = gen_batch()
        .col("id", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(8), BatchCount::from(1));
    let mut dataset = Dataset::write(
        data,
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["id"],
            IndexType::BTree,
            Some("id_idx".to_string()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 4,
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    let metadata = dataset
        .load_index_by_name(FRAG_REUSE_INDEX_NAME)
        .await
        .unwrap()
        .unwrap();
    let error = dataset
        .index_segment_statistics(FRAG_REUSE_INDEX_NAME, &[metadata.uuid])
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(
        error
            .to_string()
            .contains("Segment statistics are not supported for system indexes"),
        "{error}"
    );
    let error = dataset
        .index_statistics_from_segments(
            FRAG_REUSE_INDEX_NAME,
            vec![IndexSegmentStatistics {
                read_version: dataset.manifest.version,
                index_uuid: metadata.uuid,
                index_type_uri: String::new(),
                statistics: json!({}),
            }],
        )
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(
        error
            .to_string()
            .contains("Segment statistics are not supported for system indexes"),
        "{error}"
    );
}

#[tokio::test]
async fn test_index_statistics_from_segments_requires_migration() {
    let test_dir = copy_test_data_to_tmp("v0.21.0/bad_index_fragment_bitmap").unwrap();
    let test_uri = test_dir.path_str();
    let mut dataset = Dataset::open(&test_uri).await.unwrap();
    let original_version = dataset.manifest.version;
    let metadata = dataset.load_indices_by_name("vector_idx").await.unwrap();
    let index_uuids: Vec<_> = metadata.iter().map(|metadata| metadata.uuid).collect();
    let statistics = dataset
        .index_segment_statistics("vector_idx", &index_uuids)
        .await
        .unwrap();
    let error = dataset
        .index_statistics_from_segments("vector_idx", statistics)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Index { .. }), "{error}");
    assert!(
        error
            .to_string()
            .contains("collect index statistics in non-distributed mode to migrate"),
        "{error}"
    );
    assert_eq!(dataset.manifest.version, original_version);
    let reopened = Dataset::open(&test_uri).await.unwrap();
    assert_eq!(reopened.manifest.version, original_version);
    let reopened_metadata = reopened.load_indices_by_name("vector_idx").await.unwrap();
    assert_eq!(
        reopened_metadata
            .iter()
            .map(|metadata| &metadata.fragment_bitmap)
            .collect::<Vec<_>>(),
        metadata
            .iter()
            .map(|metadata| &metadata.fragment_bitmap)
            .collect::<Vec<_>>()
    );

    let expected: serde_json::Value =
        serde_json::from_str(&dataset.index_statistics("vector_idx").await.unwrap()).unwrap();
    dataset.checkout_latest().await.unwrap();
    assert!(dataset.manifest.version > original_version);
    let statistics = dataset
        .index_segment_statistics("vector_idx", &index_uuids)
        .await
        .unwrap();
    let actual: serde_json::Value = serde_json::from_str(
        &dataset
            .index_statistics_from_segments("vector_idx", statistics)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(actual, expected);
}
