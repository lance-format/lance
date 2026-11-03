// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::support::*;
use super::*;

#[rstest]
#[case::v1("v3.0.1/fts_v1", INVERTED_INDEX_VERSION_V1)]
#[case::v2("v4.0.1/fts_v2", INVERTED_INDEX_VERSION_V2)]
#[tokio::test]
async fn test_read_fts_format_fixture(#[case] fixture_path: &str, #[case] expected_version: u32) {
    async fn search_ids(dataset: &Dataset, query: FullTextSearchQuery) -> Vec<i64> {
        let result = dataset
            .scan()
            .project(&["id"])
            .unwrap()
            .full_text_search(query)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let mut ids = result["id"].as_primitive::<Int64Type>().values().to_vec();
        ids.sort_unstable();
        ids
    }

    let test_dir = copy_test_data_to_tmp(fixture_path).unwrap();
    let dataset = Dataset::open(&test_dir.path_str()).await.unwrap();

    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    assert_eq!(indices[0].index_version, expected_version as i32);

    let match_ids = search_ids(
        &dataset,
        FullTextSearchQuery::new("compatibility".to_string()),
    )
    .await;
    assert_eq!(match_ids, (0..300).collect::<Vec<_>>());

    let phrase =
        PhraseQuery::new("lance database".to_string()).with_column(Some("text".to_string()));
    let phrase_ids = search_ids(
        &dataset,
        FullTextSearchQuery::new_query(FtsQuery::Phrase(phrase)),
    )
    .await;
    assert_eq!(phrase_ids, (0..300).step_by(3).collect::<Vec<_>>());

    for _ in 0..2 {
        let wrapper = Arc::new(RequestIndexStoreWrapper::new(format!(
            "_indices/{}",
            indices[0].uuid
        )));
        let request =
            dataset.with_object_store_wrappers([wrapper.clone() as Arc<dyn WrappingObjectStore>]);
        assert_eq!(
            search_ids(
                &request,
                FullTextSearchQuery::new("compatibility".to_owned())
            )
            .await,
            match_ids
        );
        let phrase =
            PhraseQuery::new("lance database".to_owned()).with_column(Some("text".to_owned()));
        assert_eq!(
            search_ids(
                &request,
                FullTextSearchQuery::new_query(FtsQuery::Phrase(phrase))
            )
            .await,
            phrase_ids
        );
        wrapper.revoked.store(true, Ordering::Relaxed);
    }
}

// #[tokio::test]
#[lance_test_macros::test(tokio::test)]
async fn test_load_indices() {
    let session = Arc::new(Session::default());
    let write_params = WriteParams {
        session: Some(session.clone()),
        ..Default::default()
    };

    let test_dir = TempStrDir::default();
    let field = Field::new("tag", DataType::Utf8, false);
    let schema = Arc::new(Schema::new(vec![field]));
    let array = StringArray::from_iter_values((0..128).map(|i| ["a", "b", "c"][i % 3]));
    let record_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let reader = RecordBatchIterator::new(
        vec![record_batch.clone()].into_iter().map(Ok),
        schema.clone(),
    );

    let test_uri = &test_dir;
    let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
        .await
        .unwrap();
    dataset
        .create_index(
            &["tag"],
            IndexType::Bitmap,
            None,
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    dataset.object_store.as_ref().io_stats_incremental(); // Reset

    let indices = dataset.load_indices().await.unwrap();
    let stats = dataset.object_store.as_ref().io_stats_incremental();
    // We should already have this cached since we just wrote it.
    assert_io_eq!(stats, read_iops, 0);
    assert_io_eq!(stats, read_bytes, 0);
    assert_eq!(indices.len(), 1);

    // Clear the cache
    session.index_cache.clear().await;
    session.file_metadata_cache().clear().await;

    let dataset2 = DatasetBuilder::from_uri(test_uri)
        .with_session(session.clone())
        .load()
        .await
        .unwrap();
    let stats = dataset2.object_store.as_ref().io_stats_incremental(); // Reset
    assert_io_lt!(stats, read_bytes, 64 * 1024);

    // Because the manifest is so small, we should have opportunistically
    // cached the indices in memory already.
    let indices2 = dataset2.load_indices().await.unwrap();
    let stats = dataset2.object_store.as_ref().io_stats_incremental();
    assert_io_eq!(stats, read_iops, 0);
    assert_io_eq!(stats, read_bytes, 0);
    assert_eq!(indices2.len(), 1);
}

#[tokio::test]
async fn test_index_metadata_cache_does_not_survive_drop_recreate_same_uri() {
    fn batch_with_schema_metadata(metadata_value: String) -> RecordBatch {
        let field = Field::new("tag", DataType::Utf8, false);
        let schema = Arc::new(Schema::new_with_metadata(
            vec![field],
            HashMap::from([("large_metadata".to_string(), metadata_value)]),
        ));
        let array = StringArray::from_iter_values((0..128).map(|i| ["a", "b", "c"][i % 3]));
        RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap()
    }

    async fn write_indexed_dataset(uri: &str, session: Arc<Session>, metadata_value: String) {
        let batch = batch_with_schema_metadata(metadata_value);
        let schema = batch.schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let write_params = WriteParams {
            session: Some(session),
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, uri, Some(write_params))
            .await
            .unwrap();
        dataset
            .create_index(
                &["tag"],
                IndexType::Bitmap,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
    }

    let qn_session = Arc::new(Session::default());
    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();

    write_indexed_dataset(test_uri, qn_session.clone(), "old".to_string()).await;
    let first_dataset = DatasetBuilder::from_uri(test_uri)
        .with_session(qn_session.clone())
        .load()
        .await
        .unwrap();
    let first_indices = first_dataset.load_indices().await.unwrap();
    let first_uuid = first_indices[0].uuid;
    assert_eq!(first_dataset.version().version, 2);
    drop(first_dataset);

    std::fs::remove_dir_all(test_uri).unwrap();

    // Use a different writer session so the QN session keeps the previous
    // incarnation's index metadata cache entry. The large schema metadata
    // keeps the manifest index section outside the final read block during
    // open, so the fresh manifest load cannot opportunistically overwrite
    // the stale index metadata entry before load_indices().
    write_indexed_dataset(
        test_uri,
        Arc::new(Session::default()),
        "x".repeat(128 * 1024),
    )
    .await;
    let second_dataset = DatasetBuilder::from_uri(test_uri)
        .with_session(qn_session)
        .load()
        .await
        .unwrap();
    assert_eq!(second_dataset.version().version, 2);

    let raw_second_indices = read_manifest_indexes(
        second_dataset.object_store.as_ref(),
        &second_dataset.manifest_location,
        second_dataset.manifest(),
    )
    .await
    .unwrap();
    let raw_second_uuid = raw_second_indices[0].uuid;
    assert_ne!(
        raw_second_uuid, first_uuid,
        "the recreated dataset should commit a new physical index UUID"
    );

    let cached_second_indices = second_dataset.load_indices().await.unwrap();
    assert_eq!(
        cached_second_indices[0].uuid, raw_second_uuid,
        "load_indices should return index metadata from the recreated dataset, not the previous same-URI incarnation"
    );
}

#[tokio::test]
async fn test_load_indices_singleflights_concurrent_cache_misses() {
    let session = Arc::new(Session::default());
    let write_params = WriteParams {
        session: Some(session.clone()),
        ..Default::default()
    };
    let test_dir = TempStrDir::default();
    let schema = Arc::new(Schema::new(vec![Field::new("tag", DataType::Utf8, false)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(StringArray::from(vec!["a", "b", "c"]))],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        &test_dir,
        Some(write_params),
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["tag"],
            IndexType::Bitmap,
            None,
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();

    session.index_cache.clear().await;
    let before = session.index_cache_stats().await;
    let results = futures::future::join_all((0..32).map(|_| dataset.load_indices())).await;
    assert!(results.iter().all(|result| result.is_ok()));
    let after = session.index_cache_stats().await;

    assert_eq!(after.misses - before.misses, 1);
    assert_eq!(after.hits - before.hits, 31);
}

#[tokio::test]
async fn test_open_frag_reuse_index_with_zero_capacity_cache() {
    let test_dir = TempStrDir::default();
    let data = gen_batch()
        .col("id", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(400), BatchCount::from(1));
    let mut dataset = Dataset::write(
        data,
        &test_dir,
        Some(WriteParams {
            max_rows_per_file: 100,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["id"],
            IndexType::BTree,
            Some("id_idx".to_owned()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 200,
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    assert!(
        dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .is_some()
    );

    let session = Arc::new(Session::with_index_cache_backend(
        Arc::new(lance_core::cache::MokaCacheBackend::no_cache()),
        128 * 1024 * 1024,
        Default::default(),
    ));
    let dataset = DatasetBuilder::from_uri(&test_dir)
        .with_session(session)
        .load()
        .await
        .unwrap();

    let frag_reuse_index = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        dataset.open_frag_reuse_index(&NoOpMetricsCollector),
    )
    .await
    .expect("opening the fragment reuse index deadlocked")
    .unwrap();
    assert!(frag_reuse_index.is_some());
}

#[tokio::test]
async fn test_legacy_vector_index_details_inferred_on_load_and_migration() {
    use lance_linalg::distance::DistanceType;

    // Create a fresh dataset with IVF_HNSW_SQ so inference produces non-default
    // details (HNSW config + SQ compression) that survive proto serialization.
    let test_dir = lance_core::utils::tempfile::TempDir::default();
    let test_uri = test_dir.path_str();
    let data = gen_batch()
        .col("i", array::step::<Int32Type>())
        .col("vec", array::rand_vec::<Float32Type>(16.into()))
        .into_reader_rows(RowCount::from(1024), BatchCount::from(1));
    let mut dataset = Dataset::write(data, &test_uri, None).await.unwrap();

    let params = VectorIndexParams::with_ivf_hnsw_sq_params(
        DistanceType::Cosine,
        IvfBuildParams {
            num_partitions: Some(2),
            ..Default::default()
        },
        HnswBuildParams::default(),
        SQBuildParams::default(),
    );
    dataset
        .create_index(&["vec"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();

    // Verify the index has populated details.
    let descriptions = dataset.describe_indices(None).await.unwrap();
    assert_eq!(descriptions.len(), 1);
    assert_eq!(descriptions[0].index_type(), "IVF_HNSW_SQ");

    // Simulate a legacy dataset by clearing details from the manifest.
    // Write a new manifest with empty VectorIndexDetails value bytes.
    let mut indices = dataset.load_indices().await.unwrap().as_ref().clone();
    for idx in &mut indices {
        if let Some(details) = idx.index_details.as_ref()
            && details.type_url.ends_with("VectorIndexDetails")
        {
            idx.index_details = Some(Arc::new(vector_index_details_default()));
        }
    }
    // Write back via a no-op commit that carries the cleared indices.
    // We commit by doing a delete("false") after replacing the cached indices.
    let metadata_key = crate::session::index_caches::IndexMetadataKey {
        version: dataset.version().version,
        store_identity: &dataset.object_store.store_prefix,
        e_tag: dataset.manifest_location.e_tag.as_deref(),
    };
    dataset
        .index_cache
        .insert_with_key(&metadata_key, Arc::new(indices))
        .await;
    dataset.delete("false").await.unwrap();

    // -- Part 1: Inference on load --
    // Open with a fresh session so nothing is cached.
    let dataset = DatasetBuilder::from_uri(&test_uri)
        .with_session(Arc::new(Session::default()))
        .load()
        .await
        .unwrap();

    // load_indices should detect empty details and infer from index files.
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    let details = indices[0].index_details.as_ref().unwrap();
    assert!(
        !details.value.is_empty(),
        "Details should have been inferred from index files"
    );

    // describe_indices should return a real type (not generic "Vector").
    let descriptions = dataset.describe_indices(None).await.unwrap();
    assert_eq!(descriptions.len(), 1);
    assert_ne!(
        descriptions[0].index_type(),
        "Vector",
        "Should have inferred a specific index type"
    );
    assert_eq!(
        descriptions[0].index_type(),
        "IVF_HNSW_SQ",
        "Inferred type should match the originally-built index"
    );
    let inferred_type = descriptions[0].index_type().to_string();
    let details_json: serde_json::Value =
        serde_json::from_str(&descriptions[0].details().unwrap()).unwrap();
    assert_eq!(details_json["metric_type"], "COSINE");
    assert_eq!(details_json["compression"]["type"], "sq");
    assert!(
        details_json["hnsw"]["max_connections"].is_number(),
        "Inferred HNSW config should have max_connections; got {details_json}"
    );
    assert!(details_json["hnsw"]["construction_ef"].is_number());
    assert!(details_json["hnsw"]["max_level"].is_number());

    // -- Part 2: Migration persists inferred details --
    let mut dataset = dataset;
    dataset.delete("false").await.unwrap();

    // Open with yet another fresh session.
    let dataset = DatasetBuilder::from_uri(&test_uri)
        .with_session(Arc::new(Session::default()))
        .load()
        .await
        .unwrap();

    // The migrated manifest should have non-empty details without
    // needing to read index files again.
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    assert!(
        !indices[0].index_details.as_ref().unwrap().value.is_empty(),
        "Migrated manifest should have non-empty details"
    );
    let descriptions = dataset.describe_indices(None).await.unwrap();
    assert_eq!(descriptions[0].index_type(), inferred_type);
}

/// FRI-straddle corruption (PR #6610) used to panic in `load_indices`.
/// The fixture is a pre-#6610 dataset where a user index's
/// `fragment_bitmap` only partially covers a rewrite group. After the
/// tolerant-load fix `load_indices` returns Ok; affected old-frag IDs
/// are dropped, no new-frag IDs are inserted, and `validate()` succeeds.
#[tokio::test]
async fn test_load_indices_tolerates_fri_straddle() {
    let tmp = copy_test_data_to_tmp("fri_straddle_pre_6610/fri_straddle_dataset").unwrap();
    let uri = format!("file://{}", tmp.std_path().display());
    let dataset = Dataset::open(&uri).await.unwrap();

    let indices = dataset.load_indices().await.unwrap();
    assert!(!indices.is_empty());
    dataset.validate().await.unwrap();
}

/// Any commit reseeds indices via `load_indices` → `build_manifest`,
/// so a single no-op write persists the cleaned bitmap to disk.
#[tokio::test]
async fn test_auto_heal_persists_cleaned_bitmap() {
    use lance_table::io::manifest::read_manifest_indexes;

    let tmp = copy_test_data_to_tmp("fri_straddle_pre_6610/fri_straddle_dataset").unwrap();
    let uri = format!("file://{}", tmp.std_path().display());

    // Sanity: the on-disk fixture has at least one straddling segment.
    let pre = Dataset::open(&uri).await.unwrap();
    let raw_pre = read_manifest_indexes(&pre.object_store, &pre.manifest_location, &pre.manifest)
        .await
        .unwrap();
    let cleaned = pre.load_indices().await.unwrap();
    let any_changed = raw_pre
        .iter()
        .zip(cleaned.iter())
        .any(|(r, c)| r.fragment_bitmap != c.fragment_bitmap);
    assert!(
        any_changed,
        "fixture should have at least one segment whose bitmap is cleaned at load"
    );
    drop(pre);

    // No-op delete commits a fresh manifest seeded from cleaned indices.
    let mut dataset = Dataset::open(&uri).await.unwrap();
    dataset.delete("false").await.unwrap();

    // Reopen and read raw manifest indices: cleaned bitmaps now persisted.
    let healed = Dataset::open(&uri).await.unwrap();
    let raw_post = read_manifest_indexes(
        &healed.object_store,
        &healed.manifest_location,
        &healed.manifest,
    )
    .await
    .unwrap();
    let cleaned_post = healed.load_indices().await.unwrap();
    for (r, c) in raw_post.iter().zip(cleaned_post.iter()) {
        assert_eq!(
            r.fragment_bitmap, c.fragment_bitmap,
            "after auto-heal, raw on-disk bitmap should match cleaned bitmap"
        );
    }
}

/// A cache entry written under the key's previous meaning must cold-miss.
///
/// v1 of `lance.index.metadata-key` held only the indices the writing build
/// could read. The key fields are identical, so on a persistent backend
/// shared with such a build nothing but the schema version stops this one
/// from reading that filtered list as the complete one.
#[tokio::test]
async fn test_a_pre_rotation_cache_entry_is_not_consulted() {
    use lance_core::cache::{CacheCodec, CacheKey, CacheKeySchema, KeyBuilder};
    use std::borrow::Cow;

    struct PreRotationIndexMetadataKey<'a> {
        version: u64,
        store_identity: &'a str,
    }

    impl CacheKey for PreRotationIndexMetadataKey<'_> {
        type ValueType = Vec<IndexMetadata>;

        fn key(&self) -> Cow<'_, str> {
            Cow::Owned(format!(
                "{}:{}/{}",
                self.store_identity.len(),
                self.store_identity,
                self.version
            ))
        }

        fn type_name() -> &'static str {
            "Vec<IndexMetadata>"
        }

        fn schema() -> CacheKeySchema {
            CacheKeySchema::new("lance.index.metadata-key", 1)
        }

        fn write_key(&self, builder: &mut KeyBuilder) {
            builder.write_str(self.store_identity);
            builder.write_u64(self.version);
        }

        fn codec() -> Option<CacheCodec> {
            Some(lance_table::format::index_metadata_codec())
        }
    }

    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();
    dataset_with_an_index_from_a_newer_build(test_uri, "id_idx").await;

    let mut dataset = Dataset::open(test_uri).await.unwrap();
    let complete = raw_manifest_indices(&dataset).await;
    let as_a_released_build_would_cache_it = complete
        .iter()
        .filter(|idx| unsupported_index_version(idx).is_none())
        .cloned()
        .collect::<Vec<_>>();
    assert!(
        as_a_released_build_would_cache_it.len() < complete.len(),
        "the fixture must give the two key versions different values to cache"
    );
    dataset
        .index_cache
        .insert_with_key(
            &PreRotationIndexMetadataKey {
                version: dataset.version().version,
                store_identity: &dataset.object_store.store_prefix,
            },
            Arc::new(as_a_released_build_would_cache_it),
        )
        .await;

    dataset.delete("false").await.unwrap();

    assert_eq!(
        manifest_index_names(&dataset).await,
        ["id_idx", READABLE_INDEX],
        "a commit read a cache entry written under the key's previous meaning"
    );
}
