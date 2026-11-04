// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::index::tests::support::*;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use all_asserts::assert_ge;
use arrow::array::AsArray;
use arrow::datatypes::{Float32Type, Int32Type, UInt64Type};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::ROW_ID;
use lance_core::utils::tempfile::TempStrDir;
use lance_datagen::{BatchCount, RowCount, array, gen_batch};
use lance_index::IndexType;
use lance_index::scalar::BuiltinIndexType;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_io::object_store::{ObjectStore, ObjectStoreParams, StorageOptionsAccessor};
use lance_linalg::distance::DistanceType;
use rstest::rstest;

use crate::index::vector::IndexFileVersion;
use crate::index::vector::VectorIndexParams;
use crate::index::vector::ivf::v2::tests::{
    NUM_ROWS, PartitionBypassCacheBackend, append_dataset, generate_test_dataset, ground_truth,
    ivf_partition_cache_keys,
};

#[tokio::test]
async fn test_open_logical_vector_index_single_segment_quality_apis() {
    const DIMENSION: i32 = 8;

    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();
    let mut dataset = write_fragmented_vector_dataset(test_uri, DIMENSION).await;
    let params = VectorIndexParams::with_ivf_flat_params(DistanceType::L2, IvfBuildParams::new(2));

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

    let logical_index = dataset
        .open_logical_vector_index("vector", "vector_idx")
        .await
        .unwrap();

    assert_eq!(logical_index.name(), "vector_idx");
    assert_eq!(logical_index.column(), "vector");
    assert_eq!(logical_index.num_segments(), 1);
    assert_eq!(logical_index.metadatas().len(), 1);

    let rows_per_segment = logical_index.num_rows_per_segment();
    assert_eq!(rows_per_segment.len(), 1);
    assert_eq!(rows_per_segment[0].1, 400);

    let ivf_view = logical_index.as_ivf().unwrap();
    let partitions_per_segment = ivf_view.num_partitions_per_segment();
    assert_eq!(partitions_per_segment, vec![(rows_per_segment[0].0, 2)]);

    let partition_sizes = ivf_view.partition_sizes();
    assert_eq!(partition_sizes.len(), 1);
    assert_eq!(partition_sizes[0].1.len(), 2);
    assert_eq!(
        partition_sizes[0].1.iter().sum::<usize>(),
        rows_per_segment[0].1 as usize
    );
}

#[tokio::test]
async fn test_open_logical_vector_index_segmented_quality_apis() {
    const DIMENSION: i32 = 8;

    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();
    let mut dataset = write_fragmented_vector_dataset(test_uri, DIMENSION).await;
    let segment_ids =
        create_segmented_vector_index(&mut dataset, "vector_idx", "vector", DIMENSION).await;

    let logical_index = dataset
        .open_logical_vector_index("vector", "vector_idx")
        .await
        .unwrap();

    assert_eq!(logical_index.name(), "vector_idx");
    assert_eq!(logical_index.column(), "vector");
    assert_eq!(logical_index.num_segments(), segment_ids.len());

    let metadata_ids = logical_index
        .metadatas()
        .map(|metadata| metadata.uuid)
        .collect::<HashSet<_>>();
    assert_eq!(
        metadata_ids,
        segment_ids.into_iter().collect::<HashSet<_>>()
    );

    let rows_per_segment = logical_index.num_rows_per_segment();
    assert_eq!(rows_per_segment.len(), logical_index.num_segments());
    assert_eq!(
        rows_per_segment
            .iter()
            .map(|(_, num_rows)| *num_rows)
            .sum::<u64>(),
        400
    );
    assert!(
        rows_per_segment.iter().all(|(_, num_rows)| *num_rows > 0),
        "each segment should contain indexed rows"
    );

    let ivf_view = logical_index.as_ivf().unwrap();
    let partitions_per_segment = ivf_view.num_partitions_per_segment();
    assert!(
        partitions_per_segment
            .iter()
            .all(|(_, num_partitions)| *num_partitions == 2)
    );

    let row_count_by_segment = rows_per_segment.into_iter().collect::<HashMap<_, _>>();
    let partition_sizes = ivf_view.partition_sizes();
    assert_eq!(partition_sizes.len(), logical_index.num_segments());
    for (segment_id, sizes) in partition_sizes {
        assert_eq!(sizes.len(), 2);
        assert_eq!(
            sizes.iter().sum::<usize>(),
            row_count_by_segment[&segment_id] as usize
        );
    }
}

#[tokio::test]
async fn test_open_logical_vector_index_rejects_wrong_column() {
    const DIMENSION: i32 = 8;

    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();
    let mut dataset = write_fragmented_vector_dataset(test_uri, DIMENSION).await;
    let params = VectorIndexParams::with_ivf_flat_params(DistanceType::L2, IvfBuildParams::new(2));

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

    let err = dataset
        .open_logical_vector_index("id", "vector_idx")
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("does not belong to column 'id'"),
        "unexpected error: {err}"
    );
}

/// A covered scalar index still answers for its keyed column: only the
/// keyed prefix of `fields` must be a single field for `scalar_index_info`
/// to make it eligible for filter pushdown, not the full vector including
/// carried columns. Before this fix, a covered index was silently excluded
/// -- no error, no failing query, just a plan that quietly stopped using it.
#[tokio::test]
async fn test_scalar_index_info_includes_covered_index() {
    let data = gen_batch()
        .col("a", array::step::<Int32Type>())
        .col("b", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(20), BatchCount::from(1));
    let mut dataset = Dataset::write(data, "memory://", None).await.unwrap();
    dataset
        .create_index(
            &["a"],
            IndexType::BTree,
            None,
            &lance_index::scalar::ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
            false,
        )
        .await
        .unwrap();

    crate::utils::test::covering::declare_covering(&mut dataset, "a", "b").await;

    let indices = dataset.load_indices().await.unwrap();
    let segment_info = dataset
        .scalar_index_info_for_segments(indices.as_ref())
        .await
        .unwrap();
    assert!(
        segment_info.get_index("a").is_some(),
        "the released segment helper must retain its one-argument behavior"
    );

    let index_info = dataset.scalar_index_info().await.unwrap();
    assert!(
        index_info.get_index("a").is_some(),
        "a covered index must still be eligible for filter pushdown on its keyed column"
    );
}

#[tokio::test]
async fn test_resolve_index_column() {
    use lance_datagen::{BatchCount, RowCount, array};

    // Create a test dataset with a vector column
    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();

    let reader = lance_datagen::gen_batch()
        .col("id", array::step::<arrow_array::types::Int32Type>())
        .col(
            "vector",
            array::rand_vec::<arrow_array::types::Float32Type>(32.into()),
        )
        .into_reader_rows(RowCount::from(100), BatchCount::from(1));

    let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

    // Create an index with a custom name
    let params = crate::index::vector::VectorIndexParams::ivf_flat(
        4,
        lance_linalg::distance::MetricType::L2,
    );
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("my_vector_index".to_string()),
            &params,
            false,
        )
        .await
        .unwrap();

    // Reload dataset to get the index metadata
    let dataset = Dataset::open(test_uri).await.unwrap();
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    let index_meta = &indices[0];

    // Test 1: Pass the actual column name
    let (field_path, field) = resolve_index_column(dataset.schema(), index_meta, "vector").unwrap();
    assert_eq!(field_path, "vector");
    assert_eq!(field.name, "vector");

    // Test 2: Pass the index name (should resolve to the actual column)
    let (field_path2, field2) =
        resolve_index_column(dataset.schema(), index_meta, "my_vector_index").unwrap();
    assert_eq!(field_path2, "vector");
    assert_eq!(field2.name, "vector");

    // Test 3: Pass a non-existent column name (should fail)
    let result = resolve_index_column(dataset.schema(), index_meta, "nonexistent");
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("does not exist in the schema")
    );
}

#[tokio::test]
async fn test_resolve_index_column_error_cases() {
    use lance_datagen::{BatchCount, RowCount, array};

    // Create a test dataset
    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();

    let reader = lance_datagen::gen_batch()
        .col("id", array::step::<arrow_array::types::Int32Type>())
        .col(
            "vector",
            array::rand_vec::<arrow_array::types::Float32Type>(32.into()),
        )
        .into_reader_rows(RowCount::from(100), BatchCount::from(1));

    let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

    // Create an index
    let params = crate::index::vector::VectorIndexParams::ivf_flat(
        4,
        lance_linalg::distance::MetricType::L2,
    );
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("my_index".to_string()),
            &params,
            false,
        )
        .await
        .unwrap();

    // Reload dataset
    let dataset = Dataset::open(test_uri).await.unwrap();
    let indices = dataset.load_indices().await.unwrap();
    let index_meta = &indices[0];

    // Test: Pass a column that doesn't exist and is not the index name
    let result = resolve_index_column(dataset.schema(), index_meta, "nonexistent_column");
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("does not exist in the schema"),
        "Error message should mention column doesn't exist, got: {}",
        err_msg
    );
}

#[tokio::test]
async fn test_resolve_index_column_nested_field() {
    use arrow_array::{RecordBatch, StructArray};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};

    // Create a test dataset with nested struct manually
    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();

    // Create schema with nested structure: data.vector
    let vector_field = ArrowField::new(
        "vector",
        DataType::FixedSizeList(
            Arc::new(ArrowField::new("item", DataType::Float32, true)),
            8,
        ),
        false,
    );
    let struct_field = ArrowField::new(
        "data",
        DataType::Struct(vec![vector_field.clone()].into()),
        false,
    );
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        struct_field,
    ]));

    // Create data
    let id_array = arrow_array::Int32Array::from(vec![1, 2, 3, 4, 5]);

    // Create nested vector data
    let mut vector_values = Vec::new();
    for _ in 0..5 {
        for _ in 0..8 {
            vector_values.push(rand::random::<f32>());
        }
    }
    let vector_array = arrow_array::FixedSizeListArray::try_new_from_values(
        arrow_array::Float32Array::from(vector_values),
        8,
    )
    .unwrap();

    let struct_array = StructArray::from(vec![(
        Arc::new(vector_field),
        Arc::new(vector_array) as arrow_array::ArrayRef,
    )]);

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(id_array), Arc::new(struct_array)],
    )
    .unwrap();

    let reader = Box::new(arrow_array::RecordBatchIterator::new(
        vec![Ok(batch)],
        schema,
    ));

    let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

    // Create an index on the nested field
    let params = crate::index::vector::VectorIndexParams::ivf_flat(
        2,
        lance_linalg::distance::MetricType::L2,
    );
    dataset
        .create_index(
            &["data.vector"],
            IndexType::Vector,
            Some("nested_vector_index".to_string()),
            &params,
            false,
        )
        .await
        .unwrap();

    // Reload dataset to get the index metadata
    let dataset = Dataset::open(test_uri).await.unwrap();
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    let index_meta = &indices[0];

    // Test 1: Pass the nested field path directly
    let (field_path, field) =
        resolve_index_column(dataset.schema(), index_meta, "data.vector").unwrap();
    assert_eq!(field_path, "data.vector");
    assert_eq!(field.name, "vector");

    // Test 2: Pass the index name, should resolve to the nested field path
    let (field_path2, field2) =
        resolve_index_column(dataset.schema(), index_meta, "nested_vector_index").unwrap();
    assert_eq!(field_path2, "data.vector");
    assert_eq!(field2.name, "vector");

    // Verify the field path is correct for nested access
    assert!(
        field_path2.contains('.'),
        "Field path should contain '.' for nested field"
    );
}

#[rstest]
#[case::v3(IndexFileVersion::V3)]
#[case::legacy(IndexFileVersion::Legacy)]
#[tokio::test]
async fn test_vector_cache_uses_current_object_store(#[case] index_version: IndexFileVersion) {
    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();
    let (mut dataset, vectors) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;
    append_dataset::<Float32Type>(&mut dataset, NUM_ROWS, 0.0..1.0).await;
    assert_eq!(dataset.get_fragments().len(), 2);

    let params = VectorIndexParams::with_ivf_pq_params(
        DistanceType::L2,
        IvfBuildParams::new(4),
        PQBuildParams::default(),
    )
    .version(index_version.clone())
    .clone();
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("credential_rotation_idx".to_owned()),
            &params,
            true,
        )
        .await
        .unwrap();
    let index_meta = dataset
        .load_indices_by_name("credential_rotation_idx")
        .await
        .unwrap()
        .pop()
        .unwrap();
    let query = vectors.value(0);
    let ground_truth = ground_truth(&dataset, "vector", &query, 20, DistanceType::L2).await;

    let cache_backend = Arc::new(PartitionBypassCacheBackend::new());
    let session = Arc::new(crate::session::Session::with_index_cache_backend(
        cache_backend.clone(),
        128 * 1024 * 1024,
        Arc::new(lance_io::object_store::ObjectStoreRegistry::default()),
    ));
    let dataset = crate::DatasetBuilder::from_uri(test_uri)
        .with_session(session)
        .load()
        .await
        .unwrap();

    let store_params_a = ObjectStoreParams {
        storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
            HashMap::from([(
                "credential_generation".to_owned(),
                "secret-generation-a".to_owned(),
            )]),
        ))),
        ..Default::default()
    };
    let (store_a, _) = ObjectStore::from_uri_and_params(
        dataset.session().store_registry(),
        dataset.uri(),
        &store_params_a,
    )
    .await
    .unwrap();
    let dataset_a = dataset.with_object_store(store_a.clone(), Some(store_params_a));

    let store_params_b = ObjectStoreParams {
        storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
            HashMap::from([(
                "credential_generation".to_owned(),
                "secret-generation-b".to_owned(),
            )]),
        ))),
        ..Default::default()
    };
    let (store_b, _) = ObjectStore::from_uri_and_params(
        dataset.session().store_registry(),
        dataset.uri(),
        &store_params_b,
    )
    .await
    .unwrap();
    assert!(!Arc::ptr_eq(&store_a, &store_b));
    let dataset_b = dataset.with_object_store(store_b.clone(), Some(store_params_b));

    let _ = store_a.io_stats_incremental();
    let _ = store_b.io_stats_incremental();

    dataset_a
        .scan()
        .nearest("vector", &query, 20)
        .unwrap()
        .minimum_nprobes(4)
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();

    let frag_reuse_uuid = dataset_a.frag_reuse_index_uuid().await;
    let state_cache_key = IvfIndexStateCacheKey::new(&index_meta.uuid, frag_reuse_uuid.as_ref());
    let cached_state = if matches!(index_version, IndexFileVersion::V3) {
        Some(
            dataset_a
                .index_cache
                .get_with_key(&state_cache_key)
                .await
                .expect("V3 IVF state should be cached"),
        )
    } else {
        None
    };
    let index_path_fragment = format!("_indices/{}", index_meta.uuid);
    let first_store_stats = store_a.io_stats_incremental();
    assert!(
        first_store_stats
            .requests
            .iter()
            .any(|request| request.path.as_ref().contains(&index_path_fragment)),
        "the first query should read the index through the first object store: {first_store_stats:#?}"
    );
    let partition_keys = ivf_partition_cache_keys(
        dataset.uri(),
        &index_meta.uuid,
        frag_reuse_uuid.as_ref(),
        4,
        &index_version,
    );
    cache_backend.set_partition_keys(partition_keys.clone());
    for partition_key in &partition_keys {
        assert!(
            cache_backend.contains(partition_key).await,
            "the first query should populate portable partition entries"
        );
    }
    let index_entries_after_a = dataset.session().index_cache_stats().await.num_entries;
    let metadata_entries_after_a = dataset.session().metadata_cache_stats().await.num_entries;
    let _ = store_b.io_stats_incremental();

    cache_backend.set_bypass_partitions(true);
    let results = dataset_b
        .scan()
        .nearest("vector", &query, 20)
        .unwrap()
        .minimum_nprobes(4)
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();
    let row_ids = results[ROW_ID]
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let recall = row_ids.intersection(&ground_truth).count() as f32 / 20.0;
    assert_ge!(recall, 0.5);

    let old_store_stats = store_a.io_stats_incremental();
    let old_store_index_reads = old_store_stats
        .requests
        .iter()
        .filter(|request| request.path.as_ref().contains(&index_path_fragment))
        .count();
    let new_store_stats = store_b.io_stats_incremental();
    let new_store_index_reads = new_store_stats
        .requests
        .iter()
        .filter(|request| request.path.as_ref().contains(&index_path_fragment))
        .count();
    if matches!(index_version, IndexFileVersion::V3) {
        assert_eq!(
            old_store_index_reads, 0,
            "the new dataset query must not use readers bound to the old object store: {old_store_stats:#?}"
        );
        assert!(
            new_store_index_reads > 0,
            "the new dataset query should read the index through the new object store: {new_store_stats:#?}"
        );
    } else {
        // Legacy live indices are shared across dataset opens: their
        // readers stay bound to the object store that first populated the
        // cache, so the second dataset keeps reading through the old store.
        assert!(
            old_store_index_reads > 0,
            "the cached legacy index should keep reading through the original object store: {old_store_stats:#?}"
        );
        assert_eq!(
            new_store_index_reads, 0,
            "the cached legacy index must not reopen through the new object store: {new_store_stats:#?}"
        );
    }
    if let Some(cached_state) = cached_state {
        let state_after_rotation = dataset_b
            .index_cache
            .get_with_key(&state_cache_key)
            .await
            .expect("V3 IVF state should remain cached after rotation");
        assert!(
            Arc::ptr_eq(&cached_state, &state_after_rotation),
            "store-free IVF state should be reused across object-store generations"
        );
    }

    // Re-query through the first dataset: V3 portable state is rebound to
    // the store supplied by each reconstruction, while the cached legacy
    // index keeps reading through its original store. Either way the second
    // store must not be touched.
    let _ = store_a.io_stats_incremental();
    let _ = store_b.io_stats_incremental();
    dataset_a
        .scan()
        .nearest("vector", &query, 20)
        .unwrap()
        .minimum_nprobes(4)
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();
    let store_a_stats = store_a.io_stats_incremental();
    let store_a_index_reads = store_a_stats
        .requests
        .iter()
        .filter(|request| request.path.as_ref().contains(&index_path_fragment))
        .count();
    let store_b_stats = store_b.io_stats_incremental();
    let store_b_index_reads = store_b_stats
        .requests
        .iter()
        .filter(|request| request.path.as_ref().contains(&index_path_fragment))
        .count();
    assert!(
        store_a_index_reads > 0,
        "re-querying the first dataset should read the index through its object store: {store_a_stats:#?}"
    );
    assert_eq!(
        store_b_index_reads, 0,
        "re-querying the first dataset must not use the second object store: {store_b_stats:#?}"
    );

    // Cache keys are opaque digests, so they cannot embed credential
    // material by construction. What rotation must not do is mint new
    // entries: the same portable state, partitions, and file metadata
    // serve both object-store generations.
    let index_entries_after_rotation = dataset.session().index_cache_stats().await.num_entries;
    let metadata_entries_after_rotation =
        dataset.session().metadata_cache_stats().await.num_entries;
    assert_eq!(
        index_entries_after_rotation, index_entries_after_a,
        "credential rotation must not create new index cache entries"
    );
    assert_eq!(
        metadata_entries_after_rotation, metadata_entries_after_a,
        "credential rotation must not create new metadata cache entries"
    );

    cache_backend.set_bypass_partitions(false);
    let partition_hits_before = cache_backend.partition_hits();
    dataset_b
        .scan()
        .nearest("vector", &query, 20)
        .unwrap()
        .minimum_nprobes(4)
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();
    assert!(
        cache_backend.partition_hits() > partition_hits_before,
        "the second store should reuse portable partitions populated by the first"
    );
}
