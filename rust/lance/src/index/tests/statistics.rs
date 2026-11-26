// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::support::*;
use super::*;
use crate::index::frag_reuse_reader::tests::{
    fixture, install, persist_fixture, prepare_partition,
};
use crate::index::vector::details::vector_index_details;
use lance_core::utils::testing::{ProxyObjectStore, ProxyObjectStorePolicy};
use lance_index::pb::{JsonIndexDetails, JsonTargetDataType, VectorIndexDetails, VectorMetricType};
use lance_index::scalar::minhash_lsh::MinHashQuery;
use lance_table::format::overlay::{DataOverlayFile, OverlayCoverage};
use lance_table::format::pb::fragment_reuse_index_details::InlineContent;
use lance_table::format::{Fragment, IndexFile};
use prost::Message;
use serde_json::json;
use std::{num::NonZeroU64, sync::Mutex};

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
    let io_tracker = dataset
        .table_state
        .object_store
        .as_ref()
        .io_tracker()
        .clone();

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
    let meta = dataset
        .table_state
        .object_store
        .stat(&lookup_path)
        .await
        .unwrap();
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
    let scan = dataset
        .index_scan_statistics(
            "vector_idx",
            IndexScanStatisticsOptions::default().with_segment_stats(true),
        )
        .await
        .unwrap();
    assert!(scan.segments[0].ivf.is_none());
    assert_eq!(scan.coverage.unwrap().covered_fragment_live_rows, Some(0));
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
    let error = desc.details().unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(
        error
            .to_string()
            .contains("No scalar index plugin found for name 'mystery'"),
        "{error}"
    );
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

/// `fields` can list the carried columns too, but a logical description must
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
    let io_tracker = dataset
        .table_state
        .object_store
        .as_ref()
        .io_tracker()
        .clone();

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
    let io_tracker = dataset
        .table_state
        .object_store
        .as_ref()
        .io_tracker()
        .clone();

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
    let _ = dataset
        .table_state
        .object_store
        .as_ref()
        .io_stats_incremental();

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
    let stats = dataset
        .table_state
        .object_store
        .as_ref()
        .io_stats_incremental();
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
    let _ = dataset
        .table_state
        .object_store
        .as_ref()
        .io_stats_incremental();

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
    let stats = dataset
        .table_state
        .object_store
        .as_ref()
        .io_stats_incremental();
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
    let _ = dataset
        .table_state
        .object_store
        .as_ref()
        .io_stats_incremental();

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
    let stats = dataset
        .table_state
        .object_store
        .as_ref()
        .io_stats_incremental();
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
    let _ = dataset
        .table_state
        .object_store
        .as_ref()
        .io_stats_incremental();

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
    let stats = dataset
        .table_state
        .object_store
        .as_ref()
        .io_stats_incremental();
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

#[rstest]
#[case::same_declaration(false, "b", true)]
#[case::different_carried_field(false, "c", false)]
#[case::same_fields_different_carried_field(true, "c", false)]
#[tokio::test]
async fn test_index_description_reads_segments_in_either_layout(
    #[case] first_keys_only: bool,
    #[case] carried_by_second: &str,
    #[case] describes: bool,
) {
    let dataset = write_btree_table("memory://", &["a"], None).await;
    let [a, b, carried] =
        ["a", "b", carried_by_second].map(|name| dataset.schema().field_id(name).unwrap());
    let plain = dataset.load_indices().await.unwrap()[0].clone();
    let first = IndexMetadata {
        fields: if first_keys_only { vec![a] } else { vec![a, b] },
        covering_fields: vec![b],
        ..plain.clone()
    };
    let keys_only = IndexMetadata {
        uuid: Uuid::new_v4(),
        fields: vec![a],
        covering_fields: vec![carried],
        ..plain
    };

    let description = IndexDescriptionImpl::try_new(vec![first, keys_only], &dataset).await;

    if !describes {
        let Err(error) = description else {
            panic!("segments that carry different fields must not describe one index");
        };
        assert!(
            error
                .to_string()
                .contains("should declare identical keyed and carried fields"),
            "unexpected error: {error}"
        );
        return;
    }
    assert_eq!(description.unwrap().field_ids(), [a as u32]);
}

#[tokio::test]
async fn test_index_scan_statistics_fragment_union_and_overlay_slots() {
    let data = |rows: std::ops::Range<i32>| {
        let batch = arrow_array::record_batch!(
            ("id", Int32, rows.clone().collect::<Vec<_>>()),
            ("payload", Int32, rows.collect::<Vec<_>>())
        )
        .unwrap();
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema())
    };
    let mut dataset = Dataset::write(
        data(0..14),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 8,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let field = dataset.schema().field("id").unwrap().id;
    let segment = |ids: &[u32]| IndexMetadata {
        uuid: Uuid::new_v4(),
        name: "scan_idx".into(),
        fields: vec![field],
        covering_fields: vec![],
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(ids.iter().copied().collect()),
        index_details: Some(Arc::new(
            prost_types::Any::from_msg(&BTreeIndexDetails::default()).unwrap(),
        )),
        index_version: IndexType::BTree.version(),
        created_at: None,
        base_id: None,
        files: None,
        clustering_id: None,
        metadata: Default::default(),
    };
    let segments = vec![segment(&[0, 1, 99]), segment(&[1])];
    dataset.delete("id = 9").await.unwrap();
    dataset.append(data(14..18), None).await.unwrap();
    persist_fixture(&mut dataset, segments).await;

    let mut manifest = dataset.manifest.as_ref().clone();
    let version = manifest.version;
    let fragments = Arc::make_mut(&mut manifest.fragments);
    let payload = dataset.schema().field("payload").unwrap().id;
    let base_bytes = fragments
        .iter()
        .flat_map(|fragment| &fragment.files)
        .map(|file| file.file_size_bytes.get().unwrap().get())
        .sum::<u64>();
    let overlay_bytes = fragments[1].files[0].file_size_bytes.get().unwrap().get();
    // Offset 1 is deleted and appears in two overlays and two segment masks.
    for (ordinal, field, offsets) in [
        (0, field, vec![1, 2]),
        (1, payload, vec![3]),
        (0, field, vec![1]),
    ] {
        let mut file = fragments[1].files[0].clone();
        file.path = format!("overlay_{ordinal}.lance");
        file.fields = vec![field].into();
        file.column_indices = vec![0].into();
        fragments[1].overlays.push(DataOverlayFile {
            data_file: file,
            coverage: OverlayCoverage::Shared(Arc::new(offsets.into_iter().collect())),
            committed_version: version - 1,
        });
    }
    let residual_bytes = fragments[2].files[0].file_size_bytes.get().unwrap();
    dataset.manifest = Arc::new(manifest);
    let stats = dataset
        .index_scan_statistics("scan_idx", IndexScanStatisticsOptions::default())
        .await
        .unwrap();
    assert_eq!(stats.fragment_count, 3);
    assert_eq!(stats.live_rows, Some(17));
    assert_eq!(stats.deleted_rows, Some(1));
    assert_eq!(stats.total_data_bytes, Some(base_bytes + 2 * overlay_bytes));
    assert_eq!(stats.total_index_bytes, None);
    assert_eq!(stats.segments[0].covered_fragment_live_rows, Some(13));
    assert_eq!(stats.segments[1].covered_fragment_live_rows, Some(5));
    assert_eq!(stats.segments[1].covered_fragment_deleted_rows, Some(1));
    let coverage = stats.coverage.unwrap();
    assert_eq!(coverage.covered_fragment_count, 2);
    assert_eq!(coverage.covered_fragment_live_rows, Some(13));
    assert_eq!(coverage.uncovered_fragment_live_rows, Some(4));
    assert_eq!(coverage.covered_live_row_ratio, Some(13.0 / 17.0));
    assert_eq!(coverage.uncovered_data_bytes, Some(residual_bytes.get()));
    assert_eq!(coverage.uncovered_fragment_ids, [2].into_iter().collect());
    assert_eq!(coverage.overlay_stale_row_slots, 2);

    let manifest = Arc::make_mut(&mut dataset.manifest);
    let fragments = Arc::make_mut(&mut manifest.fragments);
    fragments[0].physical_rows = None;
    fragments[0].files[0].file_size_bytes = None.into();
    fragments[1]
        .deletion_file
        .as_mut()
        .unwrap()
        .num_deleted_rows = None;
    // A current file-less fragment can still carry a known row count.
    fragments[2].files.clear();
    let stats = dataset
        .index_scan_statistics("scan_idx", IndexScanStatisticsOptions::default())
        .await
        .unwrap();
    assert_eq!(stats.live_rows, None);
    assert_eq!(stats.deleted_rows, None);
    assert_eq!(stats.total_data_bytes, None);
    let coverage = stats.coverage.unwrap();
    assert_eq!(coverage.covered_fragment_live_rows, None);
    assert_eq!(coverage.uncovered_fragment_live_rows, Some(4));
    assert_eq!(coverage.uncovered_data_bytes, Some(0));
    assert_eq!(coverage.covered_live_row_ratio, None);
}

#[rstest]
#[case::all_sizes_recorded(Some(101), false)]
#[case::duplicate_size_missing(None, false)]
#[case::conflicting_sizes(Some(303), false)]
#[case::covered_size_missing(Some(101), true)]
#[tokio::test]
async fn test_index_scan_statistics_counts_unique_data_files(
    #[case] duplicate_size: Option<u64>,
    #[case] covered_size_missing: bool,
) {
    let batch = arrow_array::record_batch!(("id", Int32, [0, 1, 2, 3, 4, 5])).unwrap();
    let schema = batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
    let name = "unique_data_files";
    let segment = dataset
        .create_index_builder(&["id"], IndexType::Scalar, &params)
        .name(name.into())
        .fragments(vec![0])
        .execute_uncommitted()
        .await
        .unwrap();
    let manifest = Arc::make_mut(&mut dataset.manifest);
    let fragments = Arc::make_mut(&mut manifest.fragments);
    let base_bytes =
        |fragment: &Fragment| u64::from(fragment.files[0].file_size_bytes.get().unwrap());
    let covered_bytes = base_bytes(&fragments[0]);
    let uncovered_bytes = base_bytes(&fragments[1]) + base_bytes(&fragments[2]);
    let make_overlay = |fragment: &Fragment, base_id, size: Option<u64>| {
        let mut file = fragment.files[0].clone();
        file.path = "shared-overlay.lance".to_string();
        file.base_id = base_id;
        file.file_size_bytes = size.and_then(NonZeroU64::new).into();
        DataOverlayFile {
            data_file: file,
            coverage: OverlayCoverage::Shared(Arc::new([0].into_iter().collect())),
            committed_version: manifest.version,
        }
    };
    for (fragment, size) in fragments
        .iter_mut()
        .zip([Some(101), duplicate_size, Some(101)])
    {
        let overlay = make_overlay(fragment, None, size);
        fragment.overlays.push(overlay);
    }
    let other_base = make_overlay(&fragments[2], Some(42), Some(202));
    fragments[2].overlays.push(other_base);
    if covered_size_missing {
        fragments[0].files[0].file_size_bytes = None.into();
    }
    persist_fixture(&mut dataset, vec![segment]).await;
    let stats = dataset
        .index_scan_statistics(name, IndexScanStatisticsOptions::default())
        .await;
    if duplicate_size == Some(303) {
        let error = stats.unwrap_err();
        assert!(matches!(error, Error::Index { .. }), "{error}");
        let message = error.to_string();
        assert!(message.contains("inconsistent recorded sizes"), "{message}");
        assert!(message.contains("shared-overlay.lance"), "{message}");
        return;
    }
    let stats = stats.unwrap();
    assert_eq!(
        stats.total_data_bytes,
        (!covered_size_missing).then_some(covered_bytes + uncovered_bytes + 101 + 202)
    );
    let coverage = stats.coverage.unwrap();
    assert_eq!(
        coverage.uncovered_fragment_ids,
        [1, 2].into_iter().collect()
    );
    assert_eq!(
        coverage.uncovered_data_bytes,
        Some(uncovered_bytes + 101 + 202)
    );
}

#[derive(Debug)]
struct ScanStatisticsStore(Arc<Mutex<ProxyObjectStorePolicy>>);

impl WrappingObjectStore for ScanStatisticsStore {
    fn wrap(
        &self,
        _storage_prefix: &str,
        original: Arc<dyn object_store::ObjectStore>,
    ) -> Arc<dyn object_store::ObjectStore> {
        Arc::new(ProxyObjectStore::new(original, self.0.clone()))
    }

    fn wrap_paginated(
        &self,
        _store_prefix: &str,
        _original: Arc<dyn object_store::list::PaginatedListStore>,
    ) -> Option<Arc<dyn object_store::list::PaginatedListStore>> {
        None
    }
}

type ScanRequests = Arc<Mutex<Vec<(String, object_store::path::Path)>>>;

fn observe_scan_requests(dataset: &Dataset) -> (Dataset, ScanRequests) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut policy = ProxyObjectStorePolicy::new();
    let tracked = requests.clone();
    policy.set_before_policy(
        "record all requests",
        Arc::new(move |method, path| {
            tracked
                .lock()
                .unwrap()
                .push((method.to_string(), path.clone()));
            Ok(())
        }),
    );
    let tracked = dataset
        .with_object_store_wrappers([Arc::new(ScanStatisticsStore(Arc::new(Mutex::new(policy))))
            as Arc<dyn WrappingObjectStore>]);
    (tracked, requests)
}

fn assert_only_manifest_requests(dataset: &Dataset, requests: &ScanRequests) {
    let requests = requests.lock().unwrap();
    assert!(!requests.is_empty(), "expected a manifest section read");
    for (method, path) in requests.iter() {
        assert!(matches!(method.as_str(), "get_opts" | "get_ranges"));
        assert_eq!(path, &dataset.manifest_location().path);
    }
}

#[rstest]
#[case::v0_21_0_vector_marker(false)]
#[case::v0_18_2_pre_details_manifest(true)]
#[tokio::test]
async fn test_index_scan_statistics_reads_only_pinned_manifest(#[case] pre_details: bool) {
    let temp = copy_test_data_to_tmp("v0.21.0/bad_index_fragment_bitmap").unwrap();
    let mut dataset = DatasetBuilder::from_uri(temp.path_str())
        .with_session(Arc::new(Session::default()))
        .load()
        .await
        .unwrap();
    assert_eq!(dataset.manifest.version, 4);
    if pre_details {
        // Reproduce the 0.18.2 manifest with the fixture's unchanged IVF-PQ files.
        let mut segments = raw_manifest_indices(&dataset).await;
        for segment in &mut segments {
            assert_eq!(segment.index_version, 0);
            segment.index_details = None;
            segment.files = None;
        }
        persist_fixture(&mut dataset, segments).await;
    }
    let original = raw_manifest_indices(&dataset).await;
    if pre_details {
        assert!(
            original
                .iter()
                .all(|segment| { segment.index_details.is_none() && segment.files.is_none() })
        );
    }
    let name = &original[0].name;
    let full = uuid::uuid!("dc833a6e-a710-48aa-af24-9ab80f30700c");
    let delta = uuid::uuid!("ca9b1111-abfc-4fde-b4cc-8e667b84e65d");
    let declared = |indices: &[IndexMetadata], uuid| {
        indices
            .iter()
            .find(|index| index.uuid == uuid)
            .unwrap()
            .fragment_bitmap
            .clone()
            .unwrap()
    };
    assert_eq!(declared(&original, full), [0].into_iter().collect());
    assert_eq!(declared(&original, delta), [0, 1].into_iter().collect());
    let version = dataset.manifest.version;
    let (tracked, requests) = observe_scan_requests(&dataset);
    dataset.index_cache.clear().await;
    for detailed in [false, true, false] {
        requests.lock().unwrap().clear();
        let stats = tracked
            .index_scan_statistics(
                name,
                IndexScanStatisticsOptions::default().with_segment_stats(detailed),
            )
            .await
            .unwrap();
        assert_eq!(stats.dataset_version, version);
        assert_eq!(stats.metric_type, None);
        assert_eq!(stats.live_rows, Some(288));
        assert_eq!(stats.segments.len(), 2);
        for (uuid, fragment, rows) in [(full, 0, 256), (delta, 1, 32)] {
            let segment = stats.segments.iter().find(|s| s.uuid == uuid).unwrap();
            if detailed {
                assert_eq!(
                    segment.covered_fragment_ids,
                    Some([fragment].into_iter().collect())
                );
                assert_eq!(segment.covered_fragment_live_rows, Some(rows));
                assert_eq!(segment.ivf.as_ref().unwrap().partition_entry_counts, [rows]);
            } else {
                assert_eq!(segment.covered_fragment_ids, None);
                assert_eq!(segment.covered_fragment_live_rows, None);
                assert_eq!(segment.covered_fragment_deleted_rows, None);
                assert!(segment.ivf.is_none());
            }
        }
        if detailed {
            let coverage = stats.coverage.unwrap();
            assert_eq!(coverage.covered_fragment_live_rows, Some(288));
            assert_eq!(coverage.uncovered_fragment_live_rows, Some(0));
            assert!(coverage.uncovered_fragment_ids.is_empty());
        } else {
            assert!(stats.coverage.is_none());
            assert_only_manifest_requests(&tracked, &requests);
        }
    }

    let latest = Dataset::open(&temp.path_str()).await.unwrap();
    assert_eq!(latest.manifest.version, version);
    assert_eq!(raw_manifest_indices(&latest).await, original);
}

#[tokio::test]
async fn test_index_scan_statistics_released_vector_marker() {
    let temp = copy_test_data_to_tmp("v0.27.1/pq_in_schema").unwrap();
    let dataset = DatasetBuilder::from_uri(temp.path_str())
        .with_session(Arc::new(Session::default()))
        .load()
        .await
        .unwrap();
    let index = manifest_index(&dataset, "vec_idx").await;
    let (tracked, requests) = observe_scan_requests(&dataset);
    let options = IndexScanStatisticsOptions::default();
    dataset.index_cache.clear().await;
    for warm in [false, true] {
        if warm {
            assert_eq!(
                dataset.load_indices_by_name("vec_idx").await.unwrap()[0].uuid,
                index.uuid
            );
        }
        requests.lock().unwrap().clear();
        let stats = tracked
            .index_scan_statistics("vec_idx", options)
            .await
            .unwrap();
        assert_only_manifest_requests(&tracked, &requests);
        assert_eq!(
            stats.type_url.as_deref(),
            Some("/lance.table.VectorIndexDetails")
        );
        assert_eq!(stats.index_type.as_deref(), Some("Vector"));
        assert_eq!(stats.metric_type, None);
        assert_eq!(stats.details, None);
        assert_eq!(stats.segments[0].uuid, index.uuid);
        assert!(stats.segments[0].ivf.is_none());
        assert_eq!(
            stats.segments[0].covered_fragment_ids,
            index.fragment_bitmap
        );
    }
    let detailed = tracked
        .index_scan_statistics("vec_idx", options.with_segment_stats(true))
        .await
        .unwrap();
    assert_eq!(detailed.segments[0].uuid, index.uuid);
    let layout = detailed.segments[0].ivf.as_ref().unwrap();
    assert_eq!(layout.num_partitions, 1);
    assert_eq!(layout.indexed_vector_entries, 512);
    assert_eq!(layout.partition_entry_counts, [512]);
}

#[rstest]
#[case::translated(0)]
#[case::excluded(i32::MAX)]
#[tokio::test]
async fn test_index_scan_statistics_fragment_reuse(#[case] unsupported_version: i32) {
    let mut dataset = fixture().await;
    let (transition, mut destinations) = prepare_partition(&dataset, &[0], 10).await;
    destinations.push(dataset.fragments()[1].clone());
    install(
        &mut dataset,
        InlineContent {
            legacy_versions: vec![],
            transitions: vec![transition],
        }
        .encode_to_vec(),
        destinations,
        true,
    )
    .await;
    let mut indices = crate::index::loading::load_all_indices(&dataset)
        .await
        .unwrap()
        .as_ref()
        .clone();
    let data_index = indices
        .iter_mut()
        .find(|index| index.name == "i_idx")
        .unwrap();
    assert!(data_index.fragment_bitmap.as_ref().unwrap().contains(1));
    if unsupported_version != 0 {
        data_index.index_version = unsupported_version;
    }
    persist_fixture(&mut dataset, indices).await;
    dataset.index_cache.clear().await;
    let (tracked, requests) = observe_scan_requests(&dataset);

    let default = IndexScanStatisticsOptions::default();
    let metadata_only = tracked
        .index_scan_statistics("i_idx", default)
        .await
        .unwrap();
    assert!(metadata_only.coverage.is_none());
    assert_eq!(metadata_only.segments[0].covered_fragment_ids, None);
    assert_only_manifest_requests(&tracked, &requests);
    let error = dataset
        .index_scan_statistics(FRAG_REUSE_INDEX_NAME, default)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains(FRAG_REUSE_INDEX_NAME));

    let detailed = dataset
        .index_scan_statistics("i_idx", default.with_segment_stats(true))
        .await
        .unwrap();
    assert_eq!(detailed.segments[0].uuid, metadata_only.segments[0].uuid);
    assert!(detailed.segments[0].ivf.is_none());
    let query = dataset.derived_index_listing().await.unwrap();
    let query_segment = query.indices.iter().find(|index| index.name == "i_idx");
    if unsupported_version == 0 {
        let expected: RoaringBitmap = [1, 10, 11].into_iter().collect();
        assert_eq!(
            query_segment.unwrap().fragment_bitmap.as_ref(),
            Some(&expected)
        );
        assert_eq!(detailed.segments[0].covered_fragment_ids, Some(expected));
        let coverage = detailed.coverage.unwrap();
        assert_eq!(coverage.covered_fragment_count, 3);
        assert_eq!(coverage.covered_fragment_live_rows, Some(8));
        assert_eq!(coverage.uncovered_data_bytes, Some(0));
    } else {
        assert!(query_segment.is_none());
        assert!(detailed.coverage.is_none());
        assert!(detailed.segments[0].covered_fragment_ids.is_none());
    }
}

#[tokio::test]
async fn test_index_scan_statistics_partial_configuration_and_open_failure() {
    let batch = arrow_array::record_batch!(("id", Int32, Vec::<i32>::new())).unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        "memory://",
        None,
    )
    .await
    .unwrap();
    let field = dataset.schema().field("id").unwrap().id;
    let vector_segment = |name: &str, metric, target| IndexMetadata {
        uuid: Uuid::new_v4(),
        name: name.into(),
        fields: vec![field],
        covering_fields: vec![],
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(RoaringBitmap::new()),
        index_details: Some(Arc::new(vector_index_details(
            &VectorIndexParams::ivf_flat(1, metric),
            target,
        ))),
        index_version: IndexType::IvfFlat.version(),
        created_at: None,
        base_id: None,
        files: Some(vec![]),
        clustering_id: None,
        metadata: Default::default(),
    };
    let first = vector_segment("mixed", DistanceType::L2, Some(512));
    let second = vector_segment("mixed", DistanceType::Cosine, Some(1024));
    let mut legacy = vector_segment("legacy", DistanceType::L2, None);
    legacy.index_details = Some(Arc::new(vector_index_details_default()));
    legacy.fragment_bitmap = None;
    let mut broken = vector_segment("broken", DistanceType::L2, None);
    let broken_uuid = broken.uuid;
    broken.files = Some(vec![IndexFile {
        path: INDEX_FILE_NAME.into(),
        size_bytes: 123,
        file_metadata_size_bytes: None,
    }]);
    persist_fixture(&mut dataset, vec![first, second, legacy, broken]).await;

    let options = IndexScanStatisticsOptions::default();
    let mixed = dataset
        .index_scan_statistics("mixed", options)
        .await
        .unwrap();
    assert_eq!(mixed.live_rows, Some(0));
    assert_eq!(mixed.fragment_count, 0);
    assert_eq!(mixed.total_data_bytes, Some(0));
    assert_eq!(mixed.total_index_bytes, None);
    assert_eq!(mixed.coverage.unwrap().covered_live_row_ratio, Some(0.0));
    assert_eq!(mixed.index_type.as_deref(), Some("IVF_FLAT"));
    assert_eq!(mixed.metric_type, None);
    assert_eq!(mixed.details, None);
    assert_eq!(
        mixed.segments[0].details.as_ref().unwrap()["target_partition_size"],
        512
    );
    assert_eq!(mixed.segments[1].metric_type, Some(DistanceType::Cosine));

    let legacy = dataset
        .index_scan_statistics("legacy", options)
        .await
        .unwrap();
    assert_eq!(legacy.metric_type, None);
    assert_eq!(legacy.details, None);
    assert!(legacy.coverage.is_none());

    let broken = dataset
        .index_scan_statistics("broken", options)
        .await
        .unwrap();
    assert_eq!(broken.total_index_bytes, Some(123));
    let error = dataset
        .index_scan_statistics("broken", options.with_segment_stats(true))
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Index { .. }));
    let error = error.to_string();
    assert!(error.contains("'broken'"));
    assert!(error.contains(&broken_uuid.to_string()));
}

#[rstest]
#[case::different(16_384)]
#[case::equal(64)]
#[tokio::test]
async fn test_index_scan_statistics_zonemap_settings(#[case] second_size: u64) {
    let mut dataset = fixture().await;
    let mut segments = Vec::new();
    let fragments = dataset.fragment_bitmap().iter().collect::<Vec<_>>();
    assert_eq!(fragments.len(), 2);
    for (fragment_id, rows_per_zone) in fragments.into_iter().zip([64, second_size]) {
        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap)
            .with_params(&json!({"rows_per_zone": rows_per_zone, "use_seeds": false}));
        let segment = dataset
            .create_index_builder(&["i"], IndexType::ZoneMap, &params)
            .name("zones".into())
            .fragments(vec![fragment_id])
            .execute_uncommitted()
            .await
            .unwrap();
        segments.push(segment);
    }
    let expected: HashMap<_, _> = segments
        .iter()
        .map(|s| s.uuid)
        .zip([64, second_size])
        .collect();
    dataset
        .commit_existing_index_segments("zones", "i", segments)
        .await
        .unwrap();
    let stats = dataset
        .index_scan_statistics("zones", IndexScanStatisticsOptions::default())
        .await
        .unwrap();
    assert_eq!(stats.index_type.as_deref(), Some("ZoneMap"));
    for segment in &stats.segments {
        let settings = segment.details.as_ref().unwrap();
        assert_eq!(settings["rows_per_zone"], expected[&segment.uuid]);
        assert_eq!(settings["use_seeds"], false);
        assert_eq!(settings["has_null_bitmap"], true);
    }
    if second_size == 64 {
        assert_eq!(stats.details, stats.segments[0].details);
    } else {
        assert_eq!(stats.details, None);
    }
}

#[tokio::test]
async fn test_index_scan_statistics_json_target_metadata() {
    let mut dataset = fixture().await;
    let prototype = manifest_index(&dataset, "i_idx").await;
    let zone = prost_types::Any::from_msg(&lance_index::pbold::ZoneMapIndexDetails {
        rows_per_zone: Some(64),
        use_seeds: Some(false),
        has_null_bitmap: None,
    })
    .unwrap();
    let foreign = prost_types::Any {
        type_url: "/foreign.ZoneMapIndexDetails".into(),
        value: zone.value.clone(),
    };
    let wrap = |target, version| JsonIndexDetails {
        path: "value".into(),
        target_details: target,
        target_index_version: version,
        ..Default::default()
    };
    let nested_foreign = prost_types::Any::from_msg(&wrap(Some(foreign), None)).unwrap();
    let segment = |name: &str, wrapper: JsonIndexDetails| IndexMetadata {
        uuid: Uuid::new_v4(),
        name: name.into(),
        index_details: Some(Arc::new(prost_types::Any::from_msg(&wrapper).unwrap())),
        index_version: 0,
        files: Some(vec![IndexFile {
            path: "zonemap.lance".into(),
            size_bytes: 123,
            file_metadata_size_bytes: None,
        }]),
        ..prototype.clone()
    };
    let unsupported = segment("unavailable", wrap(Some(nested_foreign), None));
    let too_new = segment("too_new", wrap(Some(zone.clone()), Some(u32::MAX)));
    let supported = segment(
        "supported",
        JsonIndexDetails {
            target_data_type: JsonTargetDataType::Int64 as i32,
            ..wrap(Some(zone), None)
        },
    );
    let missing = segment("missing_target", wrap(None, None));
    let mut malformed = segment("malformed", wrap(None, None));
    Arc::make_mut(malformed.index_details.as_mut().unwrap()).value = vec![0xff];
    persist_fixture(
        &mut dataset,
        vec![
            unsupported.clone(),
            too_new,
            supported,
            missing.clone(),
            malformed.clone(),
        ],
    )
    .await;
    dataset.index_cache.clear().await;
    let (tracked, requests) = observe_scan_requests(&dataset);
    for (warm, detail) in [(false, false), (true, false), (true, true)] {
        if warm {
            dataset.load_indices().await.unwrap();
        }
        let options = IndexScanStatisticsOptions::default().with_segment_stats(detail);
        for name in ["unavailable", "too_new"] {
            requests.lock().unwrap().clear();
            let stats = tracked.index_scan_statistics(name, options).await.unwrap();
            assert_eq!(stats.live_rows, Some(8));
            assert_eq!(stats.total_index_bytes, Some(123));
            assert_eq!(stats.index_type.as_deref(), Some("Json"));
            assert_eq!(stats.details, None);
            assert!(stats.coverage.is_none());
            assert_eq!(stats.segments[0].covered_fragment_ids, None);
            assert_eq!(stats.segments[0].details, None);
            if name == "unavailable" {
                assert_eq!(stats.segments[0].uuid, unsupported.uuid);
            }
            assert_only_manifest_requests(&tracked, &requests);
        }
        let stats = tracked
            .index_scan_statistics("supported", options)
            .await
            .unwrap();
        assert_eq!(
            stats.details.unwrap()["target_details"],
            json!({
                "rows_per_zone": 64, "use_seeds": false
            })
        );
        assert!(stats.coverage.is_some());
        for segment in [&missing, &malformed] {
            let error = tracked
                .index_scan_statistics(&segment.name, options)
                .await
                .unwrap_err();
            assert!(matches!(error, Error::Index { .. }));
            assert!(error.to_string().contains(&segment.name));
            assert!(error.to_string().contains(&segment.uuid.to_string()));
        }
    }
}

#[tokio::test]
async fn test_index_scan_statistics_fts_registered_aliases() {
    let mut dataset = fixture().await;
    let prototype = manifest_index(&dataset, "i_idx").await;
    let payload = InvertedIndexDetails::default().encode_to_vec();
    let canonical = "/lance.table.InvertedIndexDetails";
    let historical = "/lance.index.pb.InvertedIndexDetails";
    let foreign = "/foreign.InvertedIndexDetails";
    let segment = |name: &str, type_url: &str| IndexMetadata {
        uuid: Uuid::new_v4(),
        name: name.into(),
        index_details: Some(Arc::new(prost_types::Any {
            type_url: type_url.into(),
            value: payload.clone(),
        })),
        index_version: 0,
        ..prototype.clone()
    };
    persist_fixture(
        &mut dataset,
        vec![
            segment("canonical", canonical),
            segment("historical", historical),
            segment("mixed", canonical),
            segment("mixed", historical),
            segment("foreign", foreign),
        ],
    )
    .await;
    let (tracked, requests) = observe_scan_requests(&dataset);
    let mut configurations = Vec::new();
    for (name, url) in [
        ("canonical", Some(canonical)),
        ("historical", Some(historical)),
        ("mixed", None),
        ("foreign", Some(foreign)),
    ] {
        requests.lock().unwrap().clear();
        let stats = tracked
            .index_scan_statistics(name, IndexScanStatisticsOptions::default())
            .await
            .unwrap();
        assert_only_manifest_requests(&tracked, &requests);
        assert_eq!(stats.type_url.as_deref(), url);
        if name == "foreign" {
            assert_eq!(stats.index_type, None);
            assert_eq!(stats.details, None);
            assert!(stats.coverage.is_none());
        } else {
            assert_eq!(stats.index_type.as_deref(), Some("Inverted"));
            configurations.push(stats.details.unwrap());
        }
    }
    assert_eq!(configurations[0], configurations[1]);
    assert_eq!(configurations[0], configurations[2]);
}

#[rstest]
#[case::canonical("/lance.table.InvertedIndexDetails", true)]
#[case::historical("/lance.index.pb.InvertedIndexDetails", true)]
#[case::mixed_case("/LANCE.TABLE.invertedindexdetails", true)]
#[case::mixed_case_historical("/LANCE.INDEX.PB.invertedindexdetails", true)]
#[case::foreign("/foreign.InvertedIndexDetails", false)]
#[tokio::test]
async fn test_index_scan_statistics_fts_search_identity(
    #[case] type_url: &str,
    #[case] indexed: bool,
) {
    let uri = TempStrDir::default();
    let batch = arrow_array::record_batch!(
        ("id", Int32, [0, 1, 2, 3]),
        ("text", Utf8, ["needle", "hay", "needle", "hay"])
    )
    .unwrap();
    let schema = batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        &uri,
        Some(WriteParams {
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let name = "fts_idx";
    dataset
        .create_index(
            &["text"],
            IndexType::Inverted,
            Some(name.into()),
            &InvertedIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let mut segment = manifest_index(&dataset, name).await;
    let uuid = segment.uuid;
    Arc::make_mut(segment.index_details.as_mut().unwrap()).type_url = type_url.to_string();
    persist_fixture(&mut dataset, vec![segment]).await;

    let dataset = DatasetBuilder::from_uri(uri.as_str()).load().await.unwrap();
    let (tracked, requests) = observe_scan_requests(&dataset);
    for detailed in [false, true] {
        requests.lock().unwrap().clear();
        let stats = tracked
            .index_scan_statistics(
                name,
                IndexScanStatisticsOptions::default().with_segment_stats(detailed),
            )
            .await
            .unwrap();
        assert_only_manifest_requests(&tracked, &requests);
        assert_eq!(stats.type_url.as_deref(), Some(type_url));
        assert_eq!(stats.segments[0].uuid, uuid);
        assert_eq!(stats.coverage.is_some(), indexed);
        if let Some(coverage) = stats.coverage {
            assert_eq!(coverage.covered_fragment_live_rows, Some(4));
            assert_eq!(coverage.uncovered_fragment_live_rows, Some(0));
        }
    }
    for fast in [false, true] {
        let mut scan = dataset.scan();
        if fast {
            scan.fast_search();
        }
        let mut query = FullTextSearchQuery::new("needle".into());
        if !indexed {
            query = query.with_column("text".into()).unwrap();
        }
        scan.full_text_search(query).unwrap();
        let plan = scan.explain_plan(false).await.unwrap();
        if indexed {
            assert!(
                plan.contains("MatchQuery") && !plan.contains("FlatMatchQuery"),
                "{plan}"
            );
        } else if fast {
            assert!(plan.contains("EmptyExec"), "{plan}");
        } else {
            assert!(plan.contains("FlatMatchQuery"), "{plan}");
        }
        let batch = scan.try_into_batch().await.unwrap();
        let mut ids = batch["id"].as_primitive::<Int32Type>().values().to_vec();
        ids.sort_unstable();
        if indexed || !fast {
            assert_eq!(ids, vec![0, 2]);
        } else {
            assert!(ids.is_empty());
        }
    }
}

#[tokio::test]
async fn test_index_scan_statistics_unsupported_vector_version() {
    let mut dataset = fixture().await;
    let mut segment = manifest_index(&dataset, "i_idx").await;
    let mut details = prost_types::Any::from_msg(&VectorIndexDetails {
        metric_type: VectorMetricType::Cosine.into(),
        ..Default::default()
    })
    .unwrap();
    // A future length-delimited compression arm is ignored by today's protobuf reader.
    details.value.extend([0x52, 0x00]);
    segment.index_details = Some(Arc::new(details));
    segment.index_version = i32::MAX;
    let uuid = segment.uuid;
    let original_size = segment.total_size_bytes();
    persist_fixture(&mut dataset, vec![segment]).await;
    let (tracked, requests) = observe_scan_requests(&dataset);
    for detail in [false, true] {
        requests.lock().unwrap().clear();
        let stats = tracked
            .index_scan_statistics(
                "i_idx",
                IndexScanStatisticsOptions::default().with_segment_stats(detail),
            )
            .await
            .unwrap();
        assert_eq!(stats.total_index_bytes, original_size);
        assert_eq!(stats.metric_type, Some(DistanceType::Cosine));
        assert_eq!(stats.index_type, None);
        assert_eq!(stats.details, None);
        assert!(stats.coverage.is_none());
        let segment = &stats.segments[0];
        assert_eq!(segment.uuid, uuid);
        assert_eq!(segment.metric_type, Some(DistanceType::Cosine));
        assert_eq!(segment.index_type, None);
        assert_eq!(segment.details, None);
        assert_eq!(segment.covered_fragment_ids, None);
        assert!(segment.ivf.is_none());
        assert_only_manifest_requests(&tracked, &requests);
    }
}

#[rstest]
#[case::hnsw("v0.11.0/legacy_hnsw_sq", false)]
#[case::missing_pq("v0.10.15/non_divisible_pq", true)]
#[tokio::test]
async fn test_index_scan_statistics_legacy_ivf(
    #[case] path: &str,
    #[case] remove_index_file: bool,
) {
    let temp = copy_test_data_to_tmp(path).unwrap();
    let dataset = Dataset::open(&temp.path_str()).await.unwrap();
    let segment = raw_manifest_indices(&dataset).await.pop().unwrap();
    if remove_index_file {
        std::fs::remove_file(
            temp.std_path()
                .join("_indices")
                .join(segment.uuid.to_string())
                .join(INDEX_FILE_NAME),
        )
        .unwrap();
    }
    let dataset = DatasetBuilder::from_uri(temp.path_str())
        .with_session(Arc::new(Session::default()))
        .load()
        .await
        .unwrap();
    let (tracked, requests) = observe_scan_requests(&dataset);
    dataset.index_cache.clear().await;
    let options = IndexScanStatisticsOptions::default();
    for warm in [false, true] {
        if warm {
            dataset.load_indices().await.unwrap();
        }
        requests.lock().unwrap().clear();
        let stats = tracked
            .index_scan_statistics(&segment.name, options)
            .await
            .unwrap();
        assert_eq!(stats.segments[0].uuid, segment.uuid);
        assert!(stats.segments[0].ivf.is_none());
        assert_only_manifest_requests(&tracked, &requests);
    }
    let detail = dataset
        .index_scan_statistics(&segment.name, options.with_segment_stats(true))
        .await;
    if remove_index_file {
        let error = detail.unwrap_err();
        assert!(matches!(error, Error::Index { .. }));
        let message = error.to_string();
        assert!(message.contains(&segment.name));
        assert!(message.contains(&segment.uuid.to_string()));
        assert!(message.contains(INDEX_FILE_NAME));
    } else {
        let detailed = detail.unwrap();
        assert_eq!(detailed.segments[0].uuid, segment.uuid);
        let ivf = detailed.segments[0].ivf.as_ref().unwrap();
        assert_eq!(ivf.num_partitions, 2);
        assert_eq!(ivf.indexed_vector_entries, 256);
        assert_eq!(ivf.partition_entry_counts, [119, 137]);
        assert_eq!(dataset.count_rows(None).await.unwrap(), 320);
    }
}

#[tokio::test]
async fn test_index_scan_statistics_json_target_agreement() {
    let mut dataset = fixture().await;
    let prototype = manifest_index(&dataset, "i_idx").await;
    let btree = prost_types::Any::from_msg(&BTreeIndexDetails::default()).unwrap();
    let bitmap =
        prost_types::Any::from_msg(&lance_index::pbold::BitmapIndexDetails::default()).unwrap();
    let wrapper = |target, version, target_type| {
        prost_types::Any::from_msg(&JsonIndexDetails {
            path: "value".into(),
            target_details: Some(target),
            target_data_type: target_type,
            target_index_version: version,
        })
        .unwrap()
    };
    let btree_legacy = wrapper(btree.clone(), None, JsonTargetDataType::Unspecified as i32);
    let btree_zero = wrapper(
        btree.clone(),
        Some(0),
        JsonTargetDataType::Unspecified as i32,
    );
    let bitmap_zero = wrapper(bitmap, Some(0), JsonTargetDataType::Unspecified as i32);
    let blocked = wrapper(btree, Some(0), JsonTargetDataType::LargeBinary as i32);
    let nested = wrapper(blocked, None, JsonTargetDataType::Int64 as i32);
    let segment = |name: &str, details: prost_types::Any| IndexMetadata {
        uuid: Uuid::new_v4(),
        name: name.into(),
        index_details: Some(Arc::new(details)),
        index_version: 0,
        ..prototype.clone()
    };
    persist_fixture(
        &mut dataset,
        vec![
            segment("target_family", btree_zero.clone()),
            segment("target_family", bitmap_zero),
            segment("target_version", btree_legacy),
            segment("target_version", btree_zero.clone()),
            segment("matching", btree_zero.clone()),
            segment("matching", btree_zero),
            segment("nested", nested),
        ],
    )
    .await;

    for detailed in [false, true] {
        let options = IndexScanStatisticsOptions::default().with_segment_stats(detailed);
        let family = dataset
            .index_scan_statistics("target_family", options)
            .await
            .unwrap();
        assert_eq!(family.index_type.as_deref(), Some("Json"));
        assert_eq!(family.details, None);
        assert_eq!(
            family.segments[0].details.as_ref().unwrap()["target_index_type"],
            "BTree"
        );
        assert_eq!(
            family.segments[1].details.as_ref().unwrap()["target_index_type"],
            "Bitmap"
        );
        let version = dataset
            .index_scan_statistics("target_version", options)
            .await
            .unwrap();
        assert_eq!(version.details, None);
        assert!(
            version.segments[0]
                .details
                .as_ref()
                .unwrap()
                .get("target_index_version")
                .is_none()
        );
        assert_eq!(
            version.segments[1].details.as_ref().unwrap()["target_index_version"],
            0
        );
        let matching = dataset
            .index_scan_statistics("matching", options)
            .await
            .unwrap();
        assert_eq!(matching.details, matching.segments[0].details);
        assert_eq!(matching.details.unwrap()["target_index_version"], 0);
        let nested = dataset
            .index_scan_statistics("nested", options)
            .await
            .unwrap();
        assert!(nested.coverage.is_none());
        assert!(nested.segments[0].covered_fragment_ids.is_none());
        let details = nested.details.unwrap();
        assert_eq!(details["target_index_type"], "Json");
        assert_eq!(details["target_details"]["target_index_type"], "BTree");
        assert_eq!(
            details["target_details"]["target_data_type"],
            "large_binary"
        );
    }
}

#[tokio::test]
async fn test_index_scan_statistics_unsupported_scalar_version() {
    let mut dataset = fixture().await;
    let prototype = manifest_index(&dataset, "i_idx").await;
    let segment =
        |name: &str, fragment_id, version, posting_format_version, size_bytes| IndexMetadata {
            uuid: Uuid::new_v4(),
            name: name.into(),
            index_details: Some(Arc::new(
                prost_types::Any::from_msg(&InvertedIndexDetails {
                    posting_format_version: Some(posting_format_version),
                    ..Default::default()
                })
                .unwrap(),
            )),
            fragment_bitmap: Some([fragment_id].into_iter().collect()),
            index_version: version,
            files: Some(vec![IndexFile {
                path: "invert.lance".into(),
                size_bytes,
                file_metadata_size_bytes: None,
            }]),
            ..prototype.clone()
        };
    let supported = segment("future_fts", 0, 3, 3, 123);
    let future = segment("future_fts", 1, 4, 4, 456);
    let future_decodable = segment("future_decodable", 1, 4, 3, 456);
    let mut malformed = segment("malformed_fts", 1, 3, 3, 456);
    Arc::make_mut(malformed.index_details.as_mut().unwrap()).value = vec![0xff];
    persist_fixture(
        &mut dataset,
        vec![
            supported.clone(),
            future.clone(),
            future_decodable,
            malformed.clone(),
        ],
    )
    .await;

    let (tracked, requests) = observe_scan_requests(&dataset);
    for (warm, detail) in [(false, false), (true, false), (true, true)] {
        if warm {
            dataset.load_indices().await.unwrap();
        }
        let options = IndexScanStatisticsOptions::default().with_segment_stats(detail);
        requests.lock().unwrap().clear();
        let stats = tracked
            .index_scan_statistics("future_fts", options)
            .await
            .unwrap();
        assert_only_manifest_requests(&tracked, &requests);
        assert_eq!(stats.live_rows, Some(8));
        assert_eq!(stats.total_index_bytes, Some(579));
        assert_eq!(stats.index_type.as_deref(), Some("Inverted"));
        assert!(stats.coverage.is_none());
        assert!(stats.details.is_none());
        assert_eq!(stats.segments[0].uuid, supported.uuid);
        assert!(stats.segments[0].details.is_some());
        assert_eq!(stats.segments[0].covered_fragment_live_rows, Some(4));
        assert_eq!(stats.segments[1].uuid, future.uuid);
        assert_eq!(stats.segments[1].index_version, 4);
        assert_eq!(stats.segments[1].size_bytes, Some(456));
        assert!(stats.segments[1].details.is_none());
        assert!(stats.segments[1].covered_fragment_ids.is_none());
        assert!(stats.segments.iter().all(|segment| segment.ivf.is_none()));
        let decodable = tracked
            .index_scan_statistics("future_decodable", options)
            .await
            .unwrap();
        assert_eq!(decodable.index_type.as_deref(), Some("Inverted"));
        assert!(decodable.details.is_none());
        let error = tracked
            .index_scan_statistics("malformed_fts", options)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Index { .. }));
        assert!(error.to_string().contains("malformed_fts"));
        assert!(error.to_string().contains(&malformed.uuid.to_string()));
    }
}

#[tokio::test]
async fn test_index_scan_statistics_vector_registered_aliases() {
    let tmp = tempfile::tempdir().unwrap();
    let mut dataset = small_vector_dataset(tmp.path(), 32).await;
    let params = VectorIndexParams::ivf_flat(2, DistanceType::Cosine);
    for name in ["cosine", "second"] {
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(name.into()),
                &params,
                true,
            )
            .await
            .unwrap();
    }
    let original = manifest_index(&dataset, "cosine").await;
    let other = manifest_index(&dataset, "second").await;
    let first_batch = dataset
        .scan()
        .limit(Some(1), None)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let query = first_batch["vector"].as_fixed_size_list().value(0);

    let aliases = [
        "/lance.index.pb.VectorIndexDetails",
        "type.googleapis.com/lance.index.VectorIndexDetails",
        "/LaNcE.InDeX.Pb.vEcToRiNdExDeTaIlS",
        "/lance.index.pb.vectorindexdetails",
    ];
    let options = IndexScanStatisticsOptions::default();
    let mut expected_layout = None;
    let mut expected_config = None;
    let declare = |mut segment: IndexMetadata, url: &str| {
        Arc::make_mut(segment.index_details.as_mut().unwrap()).type_url = url.into();
        segment
    };
    for url in aliases {
        persist_fixture(&mut dataset, vec![declare(original.clone(), url)]).await;
        dataset.index_cache.clear().await;
        let (tracked, requests) = observe_scan_requests(&dataset);
        for warm in [false, true] {
            if warm {
                dataset.load_indices().await.unwrap();
            }
            requests.lock().unwrap().clear();
            let stats = tracked
                .index_scan_statistics("cosine", options)
                .await
                .unwrap();
            assert_only_manifest_requests(&tracked, &requests);
            assert_eq!(stats.metric_type, Some(DistanceType::Cosine));
            assert_eq!(stats.type_url.as_deref(), Some(url));
            assert_eq!(stats.index_type.as_deref(), Some("IVF_FLAT"));
            assert_eq!(stats.coverage.unwrap().covered_fragment_live_rows, Some(32));
            let details = stats.details.unwrap();
            assert_eq!(details["metric_type"], "COSINE");
            assert_eq!(
                &details,
                expected_config.get_or_insert_with(|| details.clone())
            );
        }
        requests.lock().unwrap().clear();
        let detailed = tracked
            .index_scan_statistics("cosine", options.with_segment_stats(true))
            .await
            .unwrap();
        let uuid = original.uuid.to_string();
        {
            let requests = requests.lock().unwrap();
            assert!(
                requests
                    .iter()
                    .any(|(_, path)| path.as_ref().contains(&uuid))
            );
            assert!(requests.iter().all(|(_, path)| {
                path == &tracked.manifest_location().path || path.as_ref().contains(&uuid)
            }));
        }
        let segment = &detailed.segments[0];
        assert_eq!(segment.uuid, original.uuid);
        let layout = segment.ivf.as_ref().unwrap();
        assert_eq!(layout.num_partitions, 2);
        assert_eq!(layout.indexed_vector_entries, 32);
        assert_eq!(
            &layout.partition_entry_counts,
            expected_layout.get_or_insert_with(|| layout.partition_entry_counts.clone())
        );
        let mut scan = dataset.scan();
        scan.nearest("vector", query.as_ref(), 1)
            .unwrap()
            .distance_metric(DistanceType::Cosine)
            .nprobes(2);
        let plan = scan.explain_plan(false).await.unwrap();
        assert!(
            plan.contains("ANNSubIndex") || plan.contains("ANNIvfPartition"),
            "{plan}"
        );
        let batch = scan.try_into_batch().await.unwrap();
        assert_eq!(
            batch["vector"].as_fixed_size_list().value(0).to_data(),
            query.to_data()
        );
    }

    let mixed = IndexMetadata {
        name: original.name.clone(),
        ..declare(other, aliases[1])
    };
    persist_fixture(&mut dataset, vec![original.clone(), mixed]).await;
    let stats = dataset
        .index_scan_statistics("cosine", options)
        .await
        .unwrap();
    assert_eq!(stats.type_url, None);
    assert_eq!(stats.details, expected_config);
    assert_eq!(stats.metric_type, Some(DistanceType::Cosine));

    let foreign = declare(original.clone(), "/foreign.VectorIndexDetails");
    persist_fixture(&mut dataset, vec![foreign]).await;
    let (tracked, requests) = observe_scan_requests(&dataset);
    for detailed in [false, true] {
        requests.lock().unwrap().clear();
        let stats = tracked
            .index_scan_statistics("cosine", options.with_segment_stats(detailed))
            .await
            .unwrap();
        assert_only_manifest_requests(&tracked, &requests);
        assert_eq!(stats.index_type, None);
        assert_eq!(stats.metric_type, None);
        assert!(stats.coverage.is_none());
        assert!(stats.segments[0].ivf.is_none());
    }

    let mut malformed = declare(original, aliases[3]);
    Arc::make_mut(malformed.index_details.as_mut().unwrap()).value = vec![0xff];
    let uuid = malformed.uuid;
    persist_fixture(&mut dataset, vec![malformed]).await;
    let error = dataset
        .index_scan_statistics("cosine", options)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Index { .. }));
    let message = error.to_string();
    assert!(message.contains(&format!("snapshot {}", dataset.manifest().version)));
    assert!(message.contains("cosine"));
    assert!(message.contains(&uuid.to_string()));
}

#[tokio::test]
async fn test_index_scan_statistics_minhash_registered_identity() {
    let test_uri = TempStrDir::default();
    let query = "the quick brown fox jumps over the lazy dog";
    let batch = arrow_array::record_batch!(
        ("id", Int32, vec![0, 1]),
        (
            "text",
            Utf8,
            vec![query, "river banks flood after heavy rain"]
        )
    )
    .unwrap();
    let schema = batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        &test_uri,
        Some(WriteParams {
            max_rows_per_file: 1,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let name = "text_minhash";
    let params = ScalarIndexParams::for_builtin(BuiltinIndexType::MinHashLsh)
        .with_params(&json!({"num_hashes": 16, "num_bands": 4, "shingle_size": 2}));
    dataset
        .create_index(
            &["text"],
            IndexType::MinHashLsh,
            Some(name.into()),
            &params,
            false,
        )
        .await
        .unwrap();
    let original = manifest_index(&dataset, name).await;
    let canonical = original.index_details.as_ref().unwrap().type_url.clone();
    let alias = "/LaNcE.InDeX.Pb.mInHaShLsHiNdExDeTaIlS";
    let options = IndexScanStatisticsOptions::default();
    let mut expected = None;
    let declare = |url: &str| {
        let mut metadata = original.clone();
        Arc::make_mut(metadata.index_details.as_mut().unwrap()).type_url = url.into();
        metadata
    };
    for url in [canonical.as_str(), alias] {
        persist_fixture(&mut dataset, vec![declare(url)]).await;
        dataset.index_cache.clear().await;
        let (tracked, requests) = observe_scan_requests(&dataset);
        for (warm, detail) in [(false, false), (true, false), (true, true)] {
            if warm {
                dataset.load_indices().await.unwrap();
            }
            requests.lock().unwrap().clear();
            let stats = tracked
                .index_scan_statistics(name, options.with_segment_stats(detail))
                .await
                .unwrap();
            assert_only_manifest_requests(&tracked, &requests);
            assert_eq!(stats.type_url.as_deref(), Some(url));
            let details = stats.details.unwrap();
            assert_eq!(details["num_hashes"], 16);
            assert_eq!(&details, expected.get_or_insert_with(|| details.clone()));
            assert_eq!(stats.coverage.unwrap().covered_fragment_live_rows, Some(2));
            assert_eq!(stats.segments[0].uuid, original.uuid);
            assert_eq!(
                stats.segments[0].covered_fragment_ids,
                original.fragment_bitmap
            );
        }
        let mut scan = dataset.scan();
        scan.minhash_search(MinHashQuery::new(query, "text"))
            .unwrap();
        scan.limit(Some(1), None).unwrap();
        let result = scan.try_into_batch().await.unwrap();
        assert_eq!(result["id"].as_primitive::<Int32Type>().values(), &[0]);
    }

    let foreign = declare("/example.MinHashLshIndexDetails");
    persist_fixture(&mut dataset, vec![foreign]).await;
    for detail in [false, true] {
        let stats = dataset
            .index_scan_statistics(name, options.with_segment_stats(detail))
            .await
            .unwrap();
        assert!(stats.index_type.is_none());
        assert!(stats.details.is_none());
        assert!(stats.coverage.is_none());
        assert_eq!(stats.segments[0].uuid, original.uuid);
        assert!(stats.segments[0].covered_fragment_ids.is_none());
    }

    let mut malformed = declare(alias);
    Arc::make_mut(malformed.index_details.as_mut().unwrap()).value = vec![0xff];
    persist_fixture(&mut dataset, vec![malformed]).await;
    for detail in [false, true] {
        let error = dataset
            .index_scan_statistics(name, options.with_segment_stats(detail))
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains(name), "{error}");
        assert!(error.contains(&original.uuid.to_string()), "{error}");
    }
}
