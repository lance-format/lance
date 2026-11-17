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

mod multi_column_planner {
    use std::any::Any;

    use arrow::datatypes::UInt64Type;
    use arrow_array::cast::AsArray;
    use arrow_array::{ArrayRef, Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use datafusion::execution::SendableRecordBatchStream;
    use datafusion::logical_expr::{Expr, utils::conjunction};
    use lance_core::cache::LanceCache;
    use lance_core::deepsize::{Context, DeepSizeOf};
    use lance_core::utils::row_addr_remap::RowAddrRemap;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datafusion::planner::Planner;
    use lance_index::IndexType;
    use lance_index::scalar::expression::{
        ScalarIndexExpr, ScalarIndexLoader, ScalarIndexPlan, ScalarIndexSearch, ScalarQueryParser,
        create_filter_plan,
    };
    use lance_index::scalar::{
        AnyQuery, BuiltinIndexType, CreatedIndex, IndexStore, RowIdRemapper, ScalarIndexParams,
        SearchResult, UpdateCriteria,
    };
    use lance_select::RowAddrTreeMap;
    use rstest::rstest;

    use super::super::*;
    use crate::dataset::WriteParams;
    use crate::dataset::transaction::{Operation, Transaction};

    const PAIR_DETAILS_URL: &str = "/lance.test.PairIndexDetails";

    /// Has no index files: `load_index` panics, and [`PairLoader`] answers
    /// `ab_idx` searches by scanning the dataset.
    #[derive(Debug)]
    struct PairPlugin;

    #[async_trait]
    impl ScalarIndexPlugin for PairPlugin {
        fn name(&self) -> &str {
            "Pair"
        }

        fn provides_exact_answer(&self) -> bool {
            true
        }

        fn version(&self) -> u32 {
            0
        }

        fn new_query_parser(
            &self,
            _index_name: String,
            _index_details: &prost_types::Any,
        ) -> Option<Box<dyn ScalarQueryParser>> {
            None
        }

        fn new_planner(
            &self,
            index_name: String,
            _index_details: &prost_types::Any,
            keys: &Fields,
        ) -> Option<Box<dyn ScalarIndexPlanner>> {
            Some(Box::new(PairPlanner {
                index_name,
                keys: keys.clone(),
            }))
        }

        async fn load_index(
            &self,
            _index_store: Arc<dyn IndexStore>,
            _index_details: &prost_types::Any,
            _index_version: u32,
            _frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
            _cache: &LanceCache,
        ) -> Result<Arc<dyn ScalarIndex>> {
            unimplemented!("PairLoader opens pair indices")
        }
    }

    /// Consumes every conjunct that reads only its key columns.
    #[derive(Debug)]
    struct PairPlanner {
        index_name: String,
        keys: Fields,
    }

    impl ScalarIndexPlanner for PairPlanner {
        fn plan<'a>(&self, conjuncts: &'a [Expr]) -> Option<ScalarIndexPlan<'a>> {
            let consumed = conjuncts
                .iter()
                .filter(|conjunct| {
                    let columns = conjunct.column_refs();
                    !columns.is_empty()
                        && columns
                            .iter()
                            .all(|column| self.keys.find(&column.name).is_some())
                })
                .collect::<Vec<_>>();
            let filter = conjunction(consumed.iter().map(|conjunct| (*conjunct).clone()))?;
            Some(ScalarIndexPlan {
                search: ScalarIndexSearch {
                    column: self.keys[0].name().clone(),
                    index_name: self.index_name.clone(),
                    index_type: "Pair".to_string(),
                    query: Arc::new(PairQuery(filter)),
                    needs_recheck: false,
                    fragment_bitmap: None,
                },
                consumed,
            })
        }
    }

    #[derive(Debug, PartialEq)]
    struct PairQuery(Expr);

    impl AnyQuery for PairQuery {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn format(&self, _col: &str) -> String {
            self.0.to_string()
        }

        fn to_expr(&self, _col: String) -> Expr {
            self.0.clone()
        }

        fn dyn_eq(&self, other: &dyn AnyQuery) -> bool {
            other.as_any().downcast_ref::<Self>() == Some(self)
        }
    }

    #[derive(Debug)]
    struct PairIndex(Dataset);

    impl DeepSizeOf for PairIndex {
        fn deep_size_of_children(&self, _context: &mut Context) -> usize {
            0
        }
    }

    #[async_trait]
    impl Index for PairIndex {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
            self
        }
        fn statistics(&self) -> Result<serde_json::Value> {
            Ok(serde_json::Value::Null)
        }
        async fn prewarm(&self) -> Result<()> {
            Ok(())
        }
        fn index_type(&self) -> IndexType {
            IndexType::Scalar
        }
        async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
            unimplemented!()
        }
    }

    #[async_trait]
    impl ScalarIndex for PairIndex {
        async fn search(
            &self,
            query: &dyn AnyQuery,
            _metrics: &dyn MetricsCollector,
        ) -> Result<SearchResult> {
            let query = query.as_any().downcast_ref::<PairQuery>().unwrap();
            Ok(SearchResult::exact(
                row_addrs(&self.0, query.0.clone()).await,
            ))
        }
        fn can_remap(&self) -> bool {
            false
        }
        async fn remap(&self, _: &RowAddrRemap, _: &dyn IndexStore) -> Result<CreatedIndex> {
            unimplemented!()
        }
        async fn update(
            &self,
            _: SendableRecordBatchStream,
            _: &dyn IndexStore,
            _: Option<lance_index::scalar::OldIndexDataFilter>,
        ) -> Result<CreatedIndex> {
            unimplemented!()
        }
        fn update_criteria(&self) -> UpdateCriteria {
            unimplemented!()
        }
        fn derive_index_params(&self) -> Result<ScalarIndexParams> {
            unimplemented!()
        }
    }

    struct PairLoader<'a>(&'a Dataset);

    #[async_trait]
    impl ScalarIndexLoader for PairLoader<'_> {
        async fn load_index(
            &self,
            column: &str,
            index_name: &str,
            metrics: &dyn MetricsCollector,
        ) -> Result<Arc<dyn ScalarIndex>> {
            if index_name == "ab_idx" {
                return Ok(Arc::new(PairIndex(self.0.clone())));
            }
            ScalarIndexLoader::load_index(self.0, column, index_name, metrics).await
        }
    }

    async fn row_addrs(dataset: &Dataset, filter: Expr) -> RowAddrTreeMap {
        let batch = dataset
            .scan()
            .filter_expr(filter)
            .with_row_address()
            .use_scalar_index(false)
            .try_into_batch()
            .await
            .unwrap();
        let addrs = batch.column_by_name("_rowaddr").unwrap();
        RowAddrTreeMap::from_iter(addrs.as_primitive::<UInt64Type>().values().iter().copied())
    }

    fn sorted(rows: &RowAddrTreeMap) -> Vec<u64> {
        rows.row_addrs().unwrap().map(u64::from).collect()
    }

    async fn pair_dataset(uri: &str) -> Dataset {
        let schema = Arc::new(ArrowSchema::new(
            ["a", "b", "c", "d"]
                .map(|name| Field::new(name, DataType::Int32, false))
                .to_vec(),
        ));
        let column = |modulus: i32| -> ArrayRef {
            Arc::new(Int32Array::from_iter_values(
                (0..600).map(|row| row % modulus),
            ))
        };
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![column(4), column(11), column(5), column(13)],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            uri,
            Some(WriteParams {
                max_rows_per_file: 200,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), 3);
        dataset
            .create_index(
                &["c"],
                IndexType::BTree,
                Some("c_idx".to_string()),
                &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
                false,
            )
            .await
            .unwrap();

        let mut ab_idx = dataset.load_indices().await.unwrap()[0].clone();
        ab_idx.uuid = Uuid::new_v4();
        ab_idx.name = "ab_idx".to_string();
        ab_idx.fields = vec![
            dataset.schema().field("a").unwrap().id,
            dataset.schema().field("b").unwrap().id,
        ];
        ab_idx.index_details = Some(Arc::new(prost_types::Any {
            type_url: PAIR_DETAILS_URL.to_string(),
            value: Vec::new(),
        }));
        ab_idx.files = None;
        let transaction = Transaction::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![ab_idx],
                removed_indices: vec![],
            },
            None,
        );
        dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await
            .unwrap();
        dataset
    }

    fn search<'e>(expr: &'e ScalarIndexExpr, index_name: &str) -> Option<&'e ScalarIndexSearch> {
        match expr {
            ScalarIndexExpr::Query(search) => (search.index_name == index_name).then_some(search),
            ScalarIndexExpr::Not(inner) => search(inner, index_name),
            ScalarIndexExpr::And(lhs, rhs) | ScalarIndexExpr::Or(lhs, rhs) => {
                search(lhs, index_name).or_else(|| search(rhs, index_name))
            }
        }
    }

    #[rstest]
    #[case::with_column_index(
        "a = 1 AND b > 5 AND c = 3",
        "AND([a = Int32(1) AND b > Int32(5)]@ab_idx(Pair),[c = 3]@c_idx(BTree))",
        None
    )]
    #[case::with_row_residual(
        "a = 1 AND b > 5 AND d < 7",
        "[a = Int32(1) AND b > Int32(5)]@ab_idx(Pair)",
        Some("d < Int32(7)")
    )]
    #[tokio::test]
    async fn multi_column_planner_routes_and_matches_rows(
        #[case] filter: &str,
        #[case] index_query: &str,
        #[case] refine: Option<&str>,
    ) {
        let test_dir = TempStrDir::default();
        let dataset = pair_dataset(test_dir.as_str()).await;
        // `load_indices` hides indices whose plugin this build lacks.
        let indices = load_all_indices(&dataset).await.unwrap();
        let info = dataset
            .scalar_index_info_with_plugins(&indices, &HashSet::new(), |details: &IndexDetails| {
                if details.0.type_url == PAIR_DETAILS_URL {
                    Ok(&PairPlugin as &dyn ScalarIndexPlugin)
                } else {
                    details.get_plugin()
                }
            })
            .await
            .unwrap();

        let planner = Planner::new(Arc::new(ArrowSchema::from(dataset.schema())));
        let parsed = planner.parse_filter(filter).unwrap();
        let plan = create_filter_plan(planner.optimize_expr(parsed.clone()).unwrap(), &info, true)
            .unwrap();
        let index_expr = plan.index_query.as_ref().unwrap();
        assert_eq!(index_expr.to_string(), index_query);
        assert_eq!(
            plan.refine_expr
                .as_ref()
                .map(ToString::to_string)
                .as_deref(),
            refine
        );
        let all_fragments = RoaringBitmap::from_iter(0..3);
        assert_eq!(
            search(index_expr, "ab_idx").unwrap().fragment_bitmap,
            Some(all_fragments)
        );

        let result = index_expr
            .evaluate(&PairLoader(&dataset), &NoOpMetricsCollector)
            .await
            .unwrap();
        assert!(result.is_exact());
        let mut rows = result.lower.allow_list().unwrap().clone();
        if let Some(refine) = plan.refine_expr {
            rows &= row_addrs(&dataset, refine).await;
        }
        let expected = row_addrs(&dataset, parsed).await;
        assert!(!sorted(&expected).is_empty());
        assert_eq!(sorted(&rows), sorted(&expected));
    }

    #[tokio::test]
    async fn ignored_multi_column_index_leaves_its_conjuncts_to_refine() {
        let test_dir = TempStrDir::default();
        let dataset = pair_dataset(test_dir.as_str()).await;
        let indices = load_all_indices(&dataset).await.unwrap();
        let ignored = HashSet::from(["ab_idx".to_string()]);
        let info = dataset
            .scalar_index_info_with_plugins(&indices, &ignored, |details: &IndexDetails| {
                if details.0.type_url == PAIR_DETAILS_URL {
                    Ok(&PairPlugin as &dyn ScalarIndexPlugin)
                } else {
                    details.get_plugin()
                }
            })
            .await
            .unwrap();

        let planner = Planner::new(Arc::new(ArrowSchema::from(dataset.schema())));
        let filter = planner.parse_filter("a = 1 AND b > 5 AND c = 3").unwrap();
        let plan = create_filter_plan(planner.optimize_expr(filter).unwrap(), &info, true).unwrap();
        assert_eq!(
            plan.index_query
                .as_ref()
                .map(ToString::to_string)
                .as_deref(),
            Some("[c = 3]@c_idx(BTree)")
        );
        assert_eq!(
            plan.refine_expr
                .as_ref()
                .map(ToString::to_string)
                .as_deref(),
            Some("a = Int32(1) AND b > Int32(5)")
        );
    }
}
