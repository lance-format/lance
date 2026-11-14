// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

#[rstest]
#[case::default(None, false)]
#[case::serial(Some(1), false)]
#[case::override_default(Some(4), false)]
#[case::auto_config(Some(4), true)]
#[tokio::test]
async fn test_delete_concurrency(#[case] concurrency: Option<usize>, #[case] auto_config: bool) {
    let data = || {
        lance_datagen::gen_batch()
            .col(
                "id",
                lance_datagen::array::step::<arrow_array::types::Int32Type>(),
            )
            .into_reader_rows(2.into(), 2.into())
    };
    let delete_latency = Duration::from_secs(1);
    let mut dataset = Dataset::write(
        data(),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 2,
            store_params: Some(ObjectStoreParams {
                object_store_wrapper: Some(Arc::new(MockObjectStore {
                    delete_latency,
                    ..MockObjectStore::new()
                })),
                ..Default::default()
            }),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset.append(data(), None).await.unwrap();
    dataset = Dataset::write(
        data(),
        Arc::new(dataset),
        Some(WriteParams {
            mode: WriteMode::Overwrite,
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let expected = dataset.scan().try_into_batch().await.unwrap();
    // A small store default exercises explicit windows both below and above it.
    dataset.object_store = Arc::new(ObjectStore::new(
        dataset.object_store.inner.clone(),
        url::Url::parse("memory://").unwrap(),
        None,
        None,
        false,
        true,
        2,
        0,
        None,
    ));
    tokio::time::pause();
    let policy = CleanupPolicy {
        // Keep the pre-existing exhaustive literal source-compatible.
        before_timestamp: None,
        before_version: Some(dataset.version().version),
        versions: None,
        delete_unverified: true,
        error_if_tagged_old_versions: true,
        clean_referenced_branches: false,
        delete_rate_limit: None,
    };
    let operation = if auto_config {
        let mut manifest = dataset.manifest.as_ref().clone();
        manifest.config.extend([
            ("lance.auto_cleanup.interval".into(), "0".into()),
            ("lance.auto_cleanup.retain_versions".into(), "1".into()),
            ("lance.auto_cleanup.delete_concurrency".into(), "4".into()),
        ]);
        let operation = build_cleanup_operation(&dataset, &manifest, false)
            .await
            .unwrap()
            .unwrap();
        for value in ["0", "abc"] {
            manifest
                .config
                .insert("lance.auto_cleanup.delete_concurrency".into(), value.into());
            let error = build_cleanup_operation(&dataset, &manifest, false)
                .await
                .err()
                .unwrap();
            if value == "0" {
                assert!(matches!(error, Error::InvalidInput { .. }));
            } else {
                assert!(matches!(error, Error::Cleanup { .. }));
            }
            assert_contains!(error.to_string(), "delete_concurrency");
            assert_contains!(error.to_string(), value);
        }
        operation
    } else {
        let operation = dataset.cleanup(policy);
        match concurrency {
            Some(value) => operation.with_delete_concurrency(value).unwrap(),
            None => operation,
        }
    };
    let explanation = operation.explain().await.unwrap();
    let window = concurrency.unwrap_or_else(|| dataset.object_store.io_parallelism());
    let candidate_count = explanation.candidate_files.len();
    if concurrency.is_some() {
        assert!(candidate_count >= window);
    }
    let expected_duration = delete_latency * candidate_count.div_ceil(window) as u32;
    let start = tokio::time::Instant::now();
    let stats = operation.execute().await.unwrap();
    let elapsed = start.elapsed();
    assert_eq!(stats.old_versions, 2);
    assert_eq!(stats, explanation.stats);
    assert!(stats.data_files_removed > 0);
    // Bound both sides so serial execution and an ignored override cannot pass.
    assert!(elapsed >= expected_duration);
    assert!(elapsed < expected_duration + Duration::from_millis(50));
    assert_eq!(dataset.scan().try_into_batch().await.unwrap(), expected);
}

#[tokio::test]
async fn cleanup_deletes_a_large_orphan_sweep() {
    let fixture = MockDatasetFixture::try_new().unwrap();
    fixture.create_some_data().await.unwrap();

    let dataset = fixture.open().await.unwrap();
    let data_dir = dataset.data_dir();
    for index in 0..200 {
        let orphan = data_dir.clone().join(format!("orphan-{index:03}.lance"));
        dataset.object_store.put(&orphan, &[0u8; 64]).await.unwrap();
    }
    drop(dataset);

    let removed = fixture
        .run_cleanup_with_override(
            utc_now() + TimeDelta::try_days(1).unwrap(),
            Some(true),
            None,
        )
        .await
        .unwrap();

    // Enough unverified files to take the bulk-delete path.
    assert_eq!(removed.data_files_removed, 200);
    assert_eq!(removed.failed_deletes, 0);
}

#[tokio::test]
async fn bulk_cleanup_removes_local_staging_files() {
    // A local staging name is hidden from listing and rejected by the local backend,
    // so once a sweep has enough candidates to take the bulk path the store must still
    // unlink each staging file, and cleanup must count it as removed.
    let fixture = MockDatasetFixture::try_new().unwrap();
    fixture.create_some_data().await.unwrap();
    let data_dir = std::path::Path::new(fixture.tmpdir.as_str())
        .join("my_db")
        .join("data");
    let staging_count = BULK_DELETE_MIN_FILES + 20;
    let staging_bytes = 7u64;
    for index in 0..staging_count {
        std::fs::write(
            data_dir.join(format!("part-{index:03}#{index}")),
            vec![0u8; staging_bytes as usize],
        )
        .unwrap();
    }

    let removed = fixture
        .run_cleanup_with_override(
            utc_now() + TimeDelta::try_days(1).unwrap(),
            Some(true),
            None,
        )
        .await
        .unwrap();

    assert_eq!(removed.data_files_removed as usize, staging_count);
    assert_eq!(removed.failed_deletes, 0);
    assert_eq!(
        removed.bytes_removed,
        staging_bytes * staging_count as u64,
        "every staging file's bytes must count as removed"
    );
    for index in 0..staging_count {
        let name = format!("part-{index:03}#{index}");
        assert!(!data_dir.join(&name).exists(), "{name} survived cleanup");
    }
}

#[tokio::test]
async fn bulk_delete_counts_one_semaphore_unit_per_request() {
    // The delete semaphore admits one operation: a single manifest delete or one bulk
    // request. Charging one unit per path would make this sweep take a thousand
    // windows; charging one unit per request keeps it to the chunk shape.
    let observations = Arc::new(BulkDeleteObservations::new());
    let delete_latency = Duration::from_secs(1);
    let uri = format!("memory://{}", Uuid::new_v4());
    let dataset = Dataset::write(
        lance_datagen::gen_batch()
            .col(
                "id",
                lance_datagen::array::step::<arrow_array::types::Int32Type>(),
            )
            .into_reader_rows(2.into(), 2.into()),
        uri.as_str(),
        Some(WriteParams {
            store_params: Some(ObjectStoreParams {
                object_store_wrapper: Some(Arc::new(BulkDeleteStoreWrapper {
                    observations: Arc::clone(&observations),
                    latency: delete_latency,
                })),
                ..Default::default()
            }),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let data_dir = dataset.data_dir();
    // Three bulk requests: the chunk is `concurrency * BULK_DELETE_BATCH_FILES` paths,
    // so two requests run in one window and the trailing one in a second window.
    let orphan_count = 2 * BULK_DELETE_BATCH_FILES + BULK_DELETE_MIN_FILES;
    for index in 0..orphan_count {
        let orphan = data_dir.clone().join(format!("orphan-{index:04}.lance"));
        dataset.object_store.put(&orphan, &[0u8; 64]).await.unwrap();
    }

    let policy = CleanupPolicyBuilder::default()
        .before_timestamp(utc_now() + TimeDelta::try_days(1).unwrap())
        .delete_unverified(true)
        .build();
    let window = 2;
    tokio::time::pause();
    let start = tokio::time::Instant::now();
    let stats = dataset
        .cleanup(policy)
        .with_delete_concurrency(window)
        .unwrap()
        .execute()
        .await
        .unwrap();
    let elapsed = start.elapsed();

    assert_eq!(stats.data_files_removed as usize, orphan_count);
    assert_eq!(stats.failed_deletes, 0);
    let mut sizes = observations.batch_sizes();
    sizes.sort_unstable();
    assert_eq!(
        sizes,
        vec![
            BULK_DELETE_MIN_FILES,
            BULK_DELETE_BATCH_FILES,
            BULK_DELETE_BATCH_FILES
        ],
        "the sweep must be split into full bulk requests plus a remainder"
    );
    let expected = 2 * delete_latency;
    assert!(
        elapsed >= expected,
        "elapsed {elapsed:?} below {expected:?}"
    );
    assert!(
        elapsed < expected + Duration::from_millis(50),
        "elapsed {elapsed:?} exceeded two windows of {expected:?}"
    );
}

#[tokio::test(start_paused = true)]
async fn bulk_delete_rate_limit_admits_each_path() {
    // The rate budget is per path even when a request carries the whole batch, so a
    // bulk sweep cannot exceed the configured path rate.
    let observations = Arc::new(BulkDeleteObservations::new());
    let data = || {
        lance_datagen::gen_batch()
            .col(
                "id",
                lance_datagen::array::step::<arrow_array::types::Int32Type>(),
            )
            .into_reader_rows(2.into(), 2.into())
    };
    let uri = format!("memory://{}", Uuid::new_v4());
    let dataset = Dataset::write(
        data(),
        uri.as_str(),
        Some(WriteParams {
            store_params: Some(ObjectStoreParams {
                object_store_wrapper: Some(Arc::new(BulkDeleteStoreWrapper {
                    observations: Arc::clone(&observations),
                    latency: Duration::ZERO,
                })),
                ..Default::default()
            }),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let data_dir = dataset.data_dir();
    let orphan_count = BULK_DELETE_MIN_FILES + 50;
    for index in 0..orphan_count {
        let orphan = data_dir.clone().join(format!("orphan-{index:03}.lance"));
        dataset.object_store.put(&orphan, &[0u8; 64]).await.unwrap();
    }

    let rate = 200;
    let interval = calculate_duration(rate);
    let policy = CleanupPolicyBuilder::default()
        .before_timestamp(utc_now() + TimeDelta::try_days(1).unwrap())
        .delete_unverified(true)
        .build();
    let start = tokio::time::Instant::now();
    let stats = dataset
        .cleanup(policy)
        .with_delete_concurrency(2)
        .unwrap()
        .with_delete_rate_limit(rate)
        .unwrap()
        .execute()
        .await
        .unwrap();
    let elapsed = start.elapsed();

    assert_eq!(stats.data_files_removed as usize, orphan_count);
    assert_eq!(observations.batch_sizes(), vec![orphan_count]);
    // The first permit is immediate, so the sweep costs one interval per remaining
    // path. A request-level budget would instead take a single interval.
    let expected = interval * (orphan_count - 1) as u32;
    assert!(
        elapsed >= expected,
        "elapsed {elapsed:?} below {expected:?}"
    );
    assert!(
        elapsed < expected + Duration::from_millis(50),
        "elapsed {elapsed:?} admitted paths faster than one per interval"
    );
}
