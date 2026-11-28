// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::index::tests::support::*;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use arrow::array::AsArray;
use arrow::datatypes::{Float32Type, Int32Type};
use arrow_array::RecordBatchIterator;
use futures::future::try_join_all;
use lance_core::utils::tempfile::TempStrDir;
use lance_datagen::{array, gen_batch};
use lance_index::INDEX_FILE_NAME;
use lance_index::IndexType;
use lance_index::metrics::LocalMetricsCollector;
use lance_index::scalar::registry::ScalarIndexCacheKey;
use lance_index::scalar::{FullTextSearchQuery, InvertedIndexParams, inverted::InvertedIndex};
use lance_io::object_store::{
    ObjectStore, ObjectStoreParams, StorageOptionsAccessor, WrappingObjectStore,
};
use lance_linalg::distance::DistanceType;
use lance_table::format::pb::fragment_reuse_index_details::InlineContent;
use prost::Message;
use rstest::rstest;

use crate::dataset::WriteParams;
use crate::dataset::builder::DatasetBuilder;
use crate::index::DatasetIndexExt;
use crate::index::create::CreateIndexBuilder;
use crate::index::frag_reuse_reader::tests::{install, prepare_partition};
use crate::index::vector::VectorIndexParams;
use crate::session::Session;
use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount, copy_test_data_to_tmp};

#[rstest]
#[case::credential_stores(false)]
#[case::request_wrappers(true)]
#[tokio::test]
async fn test_scalar_cache_uses_current_object_store(#[case] request_wrappers: bool) {
    async fn search_ids(dataset: &Dataset, term: &str) -> Vec<i32> {
        let result = dataset
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(FullTextSearchQuery::new(term.to_owned()))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let mut ids = result["id"].as_primitive::<Int32Type>().values().to_vec();
        ids.sort_unstable();
        ids
    }

    // Rotated credentials require distinct ObjectStore instances to read the same data.
    // `memory://` creates an isolated in-memory backend for each instance, so this test needs
    // a filesystem-backed URI.
    let test_dir = TempStrDir::default();
    let batches = vec![
        arrow_array::record_batch!(
            ("id", Int32, [0, 1, 2, 3]),
            (
                "text",
                Utf8,
                [
                    "alpha common",
                    "beta common",
                    "gamma common",
                    "alpha common"
                ]
            )
        )
        .unwrap(),
        arrow_array::record_batch!(
            ("id", Int32, [4, 5, 6, 7]),
            (
                "text",
                Utf8,
                ["beta common", "gamma common", "alpha common", "beta common"]
            )
        )
        .unwrap(),
    ];
    let schema = batches[0].schema();
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
    let mut dataset = Dataset::write(
        reader,
        test_dir.as_str(),
        Some(WriteParams {
            max_rows_per_group: 4,
            max_rows_per_file: 4,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);
    dataset
        .create_index(
            &["text"],
            IndexType::Inverted,
            Some("credential_rotation_fts".to_owned()),
            &InvertedIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    let index_meta = dataset
        .load_indices_by_name("credential_rotation_fts")
        .await
        .unwrap()
        .pop()
        .unwrap();

    let session = Arc::new(Session::default());
    let dataset = DatasetBuilder::from_uri(test_dir.as_str())
        .with_session(session)
        .load()
        .await
        .unwrap();

    let index_path_fragment = format!("_indices/{}", index_meta.uuid);
    let wrapper_a = Arc::new(RequestIndexStoreWrapper::new(index_path_fragment.clone()));
    let wrapper_b = Arc::new(RequestIndexStoreWrapper::new(index_path_fragment));
    let mut requests = Vec::with_capacity(2);
    for (generation, wrapper) in [("a", &wrapper_a), ("b", &wrapper_b)] {
        let request = if request_wrappers {
            dataset.with_object_store_wrappers([wrapper.clone() as Arc<dyn WrappingObjectStore>])
        } else {
            let store_params = ObjectStoreParams {
                storage_options_accessor: Some(Arc::new(
                    StorageOptionsAccessor::with_static_options(HashMap::from([(
                        "credential_generation".to_owned(),
                        format!("secret-generation-{generation}"),
                    )])),
                )),
                object_store_wrapper: Some(wrapper.clone()),
                ..Default::default()
            };
            let (store, _) = ObjectStore::from_uri_and_params(
                dataset.session().store_registry(),
                dataset.uri(),
                &store_params,
            )
            .await
            .unwrap();
            dataset.with_object_store(store, Some(store_params))
        };
        requests.push(request);
    }
    let dataset_a = &requests[0];
    let dataset_b = &requests[1];
    assert!(!Arc::ptr_eq(
        &dataset_a.table_state.object_store,
        &dataset_b.table_state.object_store
    ));
    assert!(!Arc::ptr_eq(
        &dataset_a.table_state.object_store.inner,
        &dataset_b.table_state.object_store.inner
    ));

    let initial_metrics = LocalMetricsCollector::default();
    let initial = dataset_a
        .open_scalar_index("text", &index_meta.uuid, &initial_metrics)
        .await
        .unwrap();
    assert_eq!(initial_metrics.index_loads.load(Ordering::Relaxed), 1);
    initial.prewarm().await.unwrap();
    assert_eq!(search_ids(dataset_a, "alpha").await, vec![0, 3, 6]);
    assert!(
        !wrapper_a.requests.lock().unwrap().is_empty(),
        "the first request must read the FTS index through store A"
    );
    wrapper_a.requests.lock().unwrap().clear();
    wrapper_b.requests.lock().unwrap().clear();

    let same_binding_metrics = LocalMetricsCollector::default();
    let same_binding = dataset_a
        .open_scalar_index("text", &index_meta.uuid, &same_binding_metrics)
        .await
        .unwrap();
    assert!(Arc::ptr_eq(&initial, &same_binding));
    assert_eq!(same_binding_metrics.index_loads.load(Ordering::Relaxed), 0);
    assert_eq!(search_ids(dataset_a, "alpha").await, vec![0, 3, 6]);
    assert!(
        wrapper_a.requests.lock().unwrap().is_empty(),
        "the same store binding should reuse the prewarmed scalar index"
    );

    wrapper_a.revoked.store(true, Ordering::Relaxed);
    let rotation_metrics = LocalMetricsCollector::default();
    let opened = try_join_all(
        (0..8).map(|_| dataset_b.open_scalar_index("text", &index_meta.uuid, &rotation_metrics)),
    )
    .await
    .unwrap();
    assert_eq!(opened.len(), 8);
    assert!(opened.iter().all(|index| Arc::ptr_eq(index, &opened[0])));
    assert!(!Arc::ptr_eq(&initial, &opened[0]));
    assert_eq!(
        rotation_metrics.index_loads.load(Ordering::Relaxed),
        0,
        "concurrent opens after rotation should reuse metadata without a full index load"
    );
    assert!(
        opened[0]
            .as_any()
            .downcast_ref::<InvertedIndex>()
            .unwrap()
            .prewarm_residency_result(false)
            .await
            .fully_resident,
        "request B must retain the prewarmed query state before another prewarm"
    );
    // A request can rebind the cached shell between prewarming and the
    // final residency check. B still owns the runtime state prepared by A.
    let result = aggregate_fts_prewarm_results(
        dataset_a,
        vec![OpenedSegmentPrewarmResult {
            index_uuid: index_meta.uuid,
            partition_count: initial
                .as_any()
                .downcast_ref::<InvertedIndex>()
                .unwrap()
                .partition_count(),
            index: initial.clone(),
        }],
        None,
    )
    .await
    .unwrap();
    assert!(result.fully_resident);
    assert_eq!(search_ids(dataset_b, "beta").await, vec![1, 4, 7]);
    assert!(wrapper_a.requests.lock().unwrap().is_empty());
    assert!(
        wrapper_b.requests.lock().unwrap().is_empty(),
        "a new request must serve the prewarmed index without opening partition files"
    );

    let scalar_cache = dataset.index_cache.for_index(&index_meta.uuid, None);
    let cached = scalar_cache
        .get_unsized_with_key(&ScalarIndexCacheKey)
        .await
        .expect("the live scalar index should be cached with store B's binding");
    assert!(Arc::ptr_eq(&cached.index(), &opened[0]));

    // Evict decoded values while retaining the live shell. A fully warm query
    // cannot prove that cache misses use the current request's credentials.
    scalar_cache.clear().await;
    scalar_cache
        .insert_unsized_with_key(&ScalarIndexCacheKey, cached.clone())
        .await;
    wrapper_b.requests.lock().unwrap().clear();
    assert_eq!(search_ids(dataset_b, "beta").await, vec![1, 4, 7]);
    assert!(
        wrapper_a.requests.lock().unwrap().is_empty(),
        "request B must not issue index I/O through revoked store A"
    );
    assert!(
        wrapper_b
            .requests
            .lock()
            .unwrap()
            .iter()
            .any(|method| matches!(method.as_str(), "get_opts" | "get_ranges")),
        "request B must read uncached index data through store B"
    );

    scalar_cache.clear().await;
    scalar_cache
        .insert_unsized_with_key(&ScalarIndexCacheKey, cached)
        .await;
    wrapper_a.revoked.store(false, Ordering::Relaxed);
    wrapper_b.revoked.store(true, Ordering::Relaxed);
    wrapper_a.requests.lock().unwrap().clear();
    wrapper_b.requests.lock().unwrap().clear();
    assert_eq!(search_ids(dataset_a, "gamma").await, vec![2, 5]);
    assert!(
        wrapper_a
            .requests
            .lock()
            .unwrap()
            .iter()
            .any(|method| matches!(method.as_str(), "get_opts" | "get_ranges")),
        "re-querying dataset A must read uncached index data through store A"
    );
    assert!(
        wrapper_b.requests.lock().unwrap().is_empty(),
        "re-querying dataset A must not issue index I/O through revoked store B"
    );
}

#[tokio::test]
async fn test_request_wrappers_reload_legacy_fts_fixture() {
    let test_dir = copy_test_data_to_tmp("0.27.0/legacy_fts_index").unwrap();
    let dataset = Dataset::open(&test_dir.path_str()).await.unwrap();
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    let uuid = indices[0].uuid;

    for _ in 0..2 {
        let wrapper = Arc::new(RequestIndexStoreWrapper::new(format!("_indices/{uuid}")));
        let request =
            dataset.with_object_store_wrappers([wrapper.clone() as Arc<dyn WrappingObjectStore>]);
        let metrics = LocalMetricsCollector::default();
        let index = request
            .open_scalar_index("text", &uuid, &metrics)
            .await
            .unwrap();
        assert!(
            index
                .as_any()
                .downcast_ref::<InvertedIndex>()
                .unwrap()
                .is_legacy()
        );
        assert_eq!(
            metrics.index_loads.load(Ordering::Relaxed),
            1,
            "legacy readers must still be reloaded through each request's store"
        );
        let result = request
            .scan()
            .project(&["text"])
            .unwrap()
            .full_text_search(FullTextSearchQuery::new("happy".to_owned()))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 1);
        assert_eq!(
            result["text"].as_string::<i32>().value(0),
            "frodo was a happy puppy"
        );
        assert!(!wrapper.requests.lock().unwrap().is_empty());
        wrapper.revoked.store(true, Ordering::Relaxed);
    }
}

#[tokio::test]
async fn prewarm_head_rejects_same_version_of_recreated_table() {
    let dir = TempStrDir::default();
    let path = std::path::Path::new(dir.as_str()).join("table");
    let uri = path.to_str().unwrap();
    let batch = arrow_array::record_batch!(("id", Int32, [1, 2, 3])).unwrap();
    let write = || RecordBatchIterator::new([Ok(batch.clone())], batch.schema());

    Dataset::write(write(), uri, None).await.unwrap();
    let original = DatasetBuilder::from_uri(uri).load().await.unwrap();
    assert!(original.prewarm_at_verified_head().await.unwrap());
    std::fs::rename(&path, path.with_file_name("original")).unwrap();
    Dataset::write(write(), uri, None).await.unwrap();
    let replacement = DatasetBuilder::from_uri(uri).load().await.unwrap();

    assert_eq!(original.version_id(), replacement.version_id());
    assert_eq!(
        original.manifest_location().path,
        replacement.manifest_location().path
    );
    assert!(replacement.prewarm_at_verified_head().await.unwrap());
    assert!(!original.prewarm_at_verified_head().await.unwrap());
}

#[tokio::test]
async fn scoped_prewarm_retries_when_a_manifest_segment_format_cannot_be_read() {
    let mut dataset = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("vector", array::rand_vec::<Float32Type>(4.into()))
        .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(8))
        .await
        .unwrap();
    assert!(dataset.index_cache.supports_scoped_prewarm());
    let fragments = dataset.fragments().as_ref().clone();
    let params = VectorIndexParams::ivf_flat(1, DistanceType::L2);
    let mut segments = Vec::new();
    for fragment in &fragments {
        segments.push(
            CreateIndexBuilder::new(&mut dataset, &["vector"], IndexType::Vector, &params)
                .name("vector_idx".into())
                .fragments(vec![fragment.id as u32])
                .execute_uncommitted()
                .await
                .unwrap(),
        );
    }
    let translating = segments[0].clone();
    let direct_uuid = segments[1].uuid;
    dataset
        .commit_existing_index_segments("vector_idx", "vector", segments)
        .await
        .unwrap();
    let (transition, mut live) = prepare_partition(&dataset, &[fragments[0].id], 10).await;
    live.push(fragments[1].clone());
    let history = InlineContent {
        legacy_versions: vec![],
        transitions: vec![transition],
    };
    install(&mut dataset, history.encode_to_vec(), live, false).await;

    let marker = dataset
        .indice_files_dir(&translating)
        .unwrap()
        .join(translating.uuid.to_string())
        .join(INDEX_FILE_NAME);
    let backup = dataset
        .indice_files_dir(&translating)
        .unwrap()
        .join(translating.uuid.to_string())
        .join("marker-backup");
    let store = dataset.object_store_for_index(&translating).await.unwrap();
    store.rename(&marker, &backup).await.unwrap();
    let listing = dataset.derived_index_listing().await.unwrap();
    assert!(listing.unverifiable.contains("vector_idx"));
    assert!(
        listing
            .indices
            .iter()
            .any(|index| index.uuid == direct_uuid)
    );
    let error = dataset
        .prewarm_index_with_cache_budget("vector_idx")
        .await
        .unwrap_err();
    assert!(matches!(error, Error::IO { .. }), "{error}");
    assert!(error.to_string().contains("retry prewarm"), "{error}");

    store.rename(&backup, &marker).await.unwrap();
    dataset
        .prewarm_index_with_cache_budget("vector_idx")
        .await
        .unwrap();
}
