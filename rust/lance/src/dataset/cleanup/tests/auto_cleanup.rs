// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

// Helper function to check that the number of files is correct.
async fn check_num_files(fixture: &MockDatasetFixture, num_expected_files: usize) {
    let file_count = fixture.count_files().await.unwrap();

    assert_eq!(file_count.num_data_files, num_expected_files);
    assert_eq!(file_count.num_manifest_files, num_expected_files);
    assert_eq!(file_count.num_tx_files, num_expected_files);
}

#[rstest]
// Aged past the unverified threshold, collected by the default policy.
#[case::aged(false, 20)]
// Fresh, but `delete_unverified` waives the age protection entirely.
#[case::delete_unverified(true, 4)]
#[tokio::test]
async fn cleanup_deletes_aged_orphan_newer_than_retained_manifests(
    #[case] delete_unverified: bool,
    #[case] cleanup_day: i64,
) {
    // A data file referenced by no manifest (e.g. abandoned
    // `write_fragments` output) is deleted based on its age, and its mtime
    // is unrelated to the retained floor.
    MockClock::set_system_time(std::time::Duration::from_secs(0));
    let fixture = MockDatasetFixture::try_new().unwrap();
    fixture.create_some_data().await.unwrap();
    MockClock::set_system_time(TimeDelta::try_days(1).unwrap().to_std().unwrap());
    fixture.append_some_data().await.unwrap();
    MockClock::set_system_time(TimeDelta::try_days(2).unwrap().to_std().unwrap());
    fixture.append_some_data().await.unwrap();

    // Newer than every manifest, and old enough to be aged out below.
    MockClock::set_system_time(TimeDelta::try_days(3).unwrap().to_std().unwrap());
    let dataset = fixture.open().await.unwrap();
    let orphan = dataset.data_dir().join(format!("{}.lance", Uuid::new_v4()));
    dataset
        .object_store
        .put(&orphan, &[0u8; 1024])
        .await
        .unwrap();
    drop(dataset);

    MockClock::set_system_time(TimeDelta::try_days(cleanup_day).unwrap().to_std().unwrap());

    let before_count = fixture.count_files().await.unwrap();
    assert_eq!(before_count.num_data_files, 4);
    assert_eq!(before_count.num_manifest_files, 3);

    // Suffix-only retention: the two oldest versions go, the latest stays.
    let removed = fixture
        .run_cleanup_with_override(
            utc_now() - TimeDelta::try_days(2).unwrap(),
            Some(delete_unverified),
            None,
        )
        .await
        .unwrap();

    assert_eq!(removed.old_versions, 2);
    assert_eq!(removed.data_files_removed, 1);

    let after_count = fixture.count_files().await.unwrap();
    assert_eq!(after_count.num_manifest_files, 1);
    // Appends leave every fragment referenced by the latest version.
    assert_eq!(after_count.num_data_files, 3);
}
#[tokio::test]
async fn auto_cleanup_old_versions() {
    // Every n commits, all versions older than T should be deleted.
    //
    // We first make many commits and check that all of the versions are
    // present. We then wait until the "older_than" period has elapsed and
    // make many more commits. We check that, without explicitly calling
    // `fixture.run_cleanup`, the old versions are automatically cleaned
    // up and only the new ones remain. File counts are made after every
    // commit.
    let fixture = MockDatasetFixture::try_new().unwrap();

    fixture.create_some_data_with_auto_cleanup().await.unwrap();

    let dataset_config = &fixture.open().await.unwrap().manifest.config;
    let cleanup_interval: usize = dataset_config
        .get("lance.auto_cleanup.interval")
        .unwrap()
        .parse()
        .unwrap();

    let cleanup_older_than = TimeDelta::from_std(
        parse_duration(dataset_config.get("lance.auto_cleanup.older_than").unwrap()).unwrap(),
    )
    .unwrap();

    // First, write many files within the "older_than" window. Check that
    // no files are automatically cleaned up.
    for num_expected_files in 2..2 * cleanup_interval {
        fixture.overwrite_some_data().await.unwrap();
        check_num_files(&fixture, num_expected_files).await;
    }

    // Fast forward so we are outside of the "older_than" window.
    MockClock::set_system_time(
        (cleanup_older_than + TimeDelta::minutes(1))
            .to_std()
            .unwrap(),
    );

    // Write more files and check that those outside of the "older_than" window
    // are cleaned up.
    for num_expected_files in 2..cleanup_interval {
        fixture.overwrite_some_data().await.unwrap();
        check_num_files(&fixture, num_expected_files).await;
    }

    // Overwrite auto cleanup params with custom values
    let mut dataset = *(fixture.open().await.unwrap());
    let mut new_autoclean_params = HashMap::new();

    let new_cleanup_older_than_str = "1month 2days 2h 42min 6sec";
    let new_cleanup_older_than =
        TimeDelta::from_std(parse_duration(new_cleanup_older_than_str).unwrap()).unwrap();
    new_autoclean_params.insert(
        "lance.auto_cleanup.older_than".to_string(),
        new_cleanup_older_than_str.to_string(),
    );

    let new_cleanup_interval = 5;
    new_autoclean_params.insert(
        "lance.auto_cleanup.interval".to_string(),
        new_cleanup_interval.to_string(),
    );

    // Convert to new API format
    let config_updates = new_autoclean_params
        .into_iter()
        .map(|(k, v)| (k, Some(v)))
        .collect::<HashMap<String, Option<String>>>();
    dataset.update_config(config_updates).await.unwrap();

    // Fast forward so we are outside of the new "older_than" window.
    MockClock::set_system_time(
        (cleanup_older_than + new_cleanup_older_than + TimeDelta::minutes(2))
            .to_std()
            .unwrap(),
    );

    fixture.overwrite_some_data().await.unwrap();

    for num_expected_files in 2..new_cleanup_interval {
        fixture.overwrite_some_data().await.unwrap();
        check_num_files(&fixture, num_expected_files).await;
    }
}

#[tokio::test]
async fn test_auto_cleanup_interval_zero() {
    let fixture = MockDatasetFixture::try_new().unwrap();

    fixture.create_some_data().await.unwrap();
    fixture.overwrite_some_data().await.unwrap();
    fixture.overwrite_some_data().await.unwrap();
    check_num_files(&fixture, 3).await;

    let mut dataset = fixture.open().await.unwrap();
    let mut config_updates = HashMap::new();
    config_updates.insert(
        "lance.auto_cleanup.interval".to_string(),
        Some("0".to_string()),
    );
    config_updates.insert(
        "lance.auto_cleanup.retain_versions".to_string(),
        Some("1".to_string()),
    );
    dataset
        .update_config(config_updates)
        .replace()
        .await
        .unwrap();

    fixture.overwrite_some_data().await.unwrap();
    fixture.overwrite_some_data().await.unwrap();
    // The last version before the new commit is retained, means we have 2 versions to assert
    check_num_files(&fixture, 2).await;

    fixture.overwrite_some_data().await.unwrap();
    check_num_files(&fixture, 2).await;
}

#[tokio::test]
async fn cleanup_recent_verified_files() {
    let fixture = MockDatasetFixture::try_new().unwrap();
    fixture.create_some_data().await.unwrap();
    MockClock::set_system_time(TimeDelta::try_seconds(1).unwrap().to_std().unwrap());
    fixture.overwrite_some_data().await.unwrap();

    let before_count = fixture.count_files().await.unwrap();
    assert_eq!(before_count.num_data_files, 2);
    assert_eq!(before_count.num_manifest_files, 2);

    // Not much time has passed but we can still delete the old manifest
    // and the related data files
    let before = utc_now();
    let removed = fixture.run_cleanup(before).await.unwrap();

    let after_count = fixture.count_files().await.unwrap();
    assert_eq!(removed.old_versions, 1);
    assert_eq!(
        removed.bytes_removed,
        before_count.num_bytes - after_count.num_bytes
    );

    assert_eq!(after_count.num_data_files, 1);
    assert_eq!(after_count.num_manifest_files, 1);
}

#[tokio::test]
async fn dont_cleanup_recent_unverified_files() {
    for (override_opt, old_files) in [
        (Some(false), false), // User provides false, files are new - do not delete
        (Some(true), false),  // User provides true, files are new - delete
        (None, true),         // Default, files are old - delete
        (None, false),        // Default, files are new - do not delete
    ] {
        MockClock::set_system_time(std::time::Duration::from_secs(0));
        let mut fixture = MockDatasetFixture::try_new().unwrap();
        fixture.create_some_data().await.unwrap();
        let failed_commit_time = TimeDelta::try_seconds(1).unwrap().to_std().unwrap();
        MockClock::set_system_time(failed_commit_time);
        fixture.block_commits();
        assert!(fixture.append_some_data().await.is_err());
        let dataset = fixture.open().await.unwrap();
        let blob_path = dataset.base.clone().join("_blobs").join("orphan.blob");
        dataset
            .object_store
            .put(&blob_path, b"orphan")
            .await
            .unwrap();

        let staging_manifests = dataset
            .object_store
            .read_dir_all(&dataset.versions_dir(), None)
            .try_filter_map(|meta| {
                let relative_path = remove_prefix(&meta.location, &dataset.base);
                future::ready(Ok(staging_manifest_version(&relative_path)
                    .is_some()
                    .then_some(meta.location)))
            })
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(staging_manifests.len(), 1);
        let staging_manifest = &staging_manifests[0];

        let age = if old_files {
            TimeDelta::try_days(UNVERIFIED_THRESHOLD_DAYS + 1).unwrap()
        } else {
            TimeDelta::try_days(UNVERIFIED_THRESHOLD_DAYS - 1).unwrap()
        };
        MockClock::set_system_time(failed_commit_time + age.to_std().unwrap());

        // The above created some unreferenced data files but, since they
        // are not referenced in any manifest, and 7 days has not passed, we
        // cannot safely delete them unless the user overrides the safety check

        let before_count = fixture.count_files().await.unwrap();
        assert_eq!(before_count.num_data_files, 2);
        assert_eq!(before_count.num_manifest_files, 1);

        let before = utc_now();
        let removed = fixture
            .run_cleanup_with_override(before, override_opt, None)
            .await
            .unwrap();

        let should_delete = override_opt.unwrap_or(false) || old_files;

        let after_count = fixture.count_files().await.unwrap();
        assert_eq!(
            dataset.object_store.exists(&blob_path).await.unwrap(),
            !should_delete,
            "override={override_opt:?}, old_files={old_files}"
        );
        assert_eq!(removed.old_versions, 0);
        assert_eq!(
            removed.bytes_removed,
            before_count.num_bytes - after_count.num_bytes
        );

        if should_delete {
            assert_gt!(removed.bytes_removed, 0);
        } else {
            assert_eq!(removed.bytes_removed, 0);
        }
        assert_eq!(
            dataset.object_store.exists(staging_manifest).await.unwrap(),
            !should_delete
        );
    }
}
