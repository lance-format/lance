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
