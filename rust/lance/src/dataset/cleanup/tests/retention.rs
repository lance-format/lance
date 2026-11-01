// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

#[rstest]
#[tokio::test]
async fn cleanup_default_policy_is_remove_all_after_two_weeks(
    #[values(false, true)] delete_unverified: bool,
) {
    let policy = RetentionPolicy::default();
    assert_eq!(
        policy.rules(),
        &[CleanupRule::RemoveAll {
            older_than: Duration::from_secs(14 * 86_400)
        }]
    );
    let fixture = MockDatasetFixture::try_new_in_memory().unwrap();
    for (index, hour) in [0, 24, 360].into_iter().enumerate() {
        MockClock::set_system_time(Duration::from_secs(hour * 3600));
        if index == 0 {
            fixture.create_some_data().await.unwrap();
        } else {
            fixture.overwrite_some_data().await.unwrap();
        }
    }
    let dataset = fixture.open().await.unwrap();
    let operation = dataset
        .cleanup(
            CleanupPolicyBuilder::default()
                .delete_unverified(delete_unverified)
                .build(),
        )
        .with_retention_policy(policy)
        .unwrap();
    assert_eq!(operation.explain().await.unwrap().stats.old_versions, 1);
    assert_eq!(operation.execute().await.unwrap().old_versions, 1);
    assert_eq!(
        dataset
            .version_refs()
            .await
            .unwrap()
            .iter()
            .map(|version| version.version)
            .collect::<Vec<_>>(),
        vec![2, 3]
    );
    assert_eq!(operation.execute().await.unwrap().old_versions, 0);
}

#[test]
fn cleanup_policy_preserves_large_duration_precision() {
    let policy = parse_cleanup_policy(
        r#"{"rules":[
        {"type":"keep_one","per_interval":"315360000000s 1us","older_than":"2days"}
    ]}"#,
    )
    .unwrap();
    assert_eq!(
        policy.rules(),
        &[CleanupRule::KeepOne {
            per_interval: Duration::new(315_360_000_000, 1_000),
            older_than: MIN_THINNING_AGE,
        }]
    );
}

#[rstest]
#[tokio::test]
async fn cleanup_retention_policy_ladder(
    #[values(false, true)] auto_config: bool,
    #[values(false, true)] protected: bool,
    #[values(false, true)] delete_unverified: bool,
) {
    let fixture = MockDatasetFixture::try_new_in_memory().unwrap();
    for (index, hour) in [
        0, 1, 240, 264, 576, 577, 769, 770, 792, 792, 793, 864, 864, 865, 911, 912, 912, 913, 959,
        960,
    ]
    .into_iter()
    .enumerate()
    {
        MockClock::set_system_time(Duration::from_secs(hour * 3600));
        let data = lance_datagen::gen_batch()
            .col("value", lance_datagen::array::step::<Int32Type>())
            .into_reader_rows(2.into(), 2.into());
        Dataset::write(
            data,
            &fixture.dataset_path,
            Some(WriteParams {
                mode: if index == 0 {
                    WriteMode::Create
                } else {
                    WriteMode::Overwrite
                },
                max_rows_per_file: 2,
                store_params: Some(fixture.os_params()),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    }
    let dataset = fixture.open().await.unwrap();
    let expected_data = dataset.scan().try_into_batch().await.unwrap();
    let orphan = dataset.data_dir().join("uncommitted.lance");
    dataset
        .object_store
        .put(&orphan, b"uncommitted")
        .await
        .unwrap();
    let mut operation = if auto_config {
        let mut manifest = dataset.manifest.as_ref().clone();
        manifest.config.extend([
            ("lance.auto_cleanup.policy_interval".into(), "1".into()),
            (
                "lance.auto_cleanup.policy".into(),
                r#"{"rules":[
                {"type":"keep_one","per_interval":"1h","older_than":"2days"},
                {"type":"keep_one","per_interval":"1day","older_than":"7days"},
                {"type":"remove_all","older_than":"30days"}
            ]}"#
                .into(),
            ),
        ]);
        build_cleanup_operation(&dataset, &manifest, false)
            .await
            .unwrap()
            .unwrap()
    } else {
        dataset
            .cleanup(CleanupPolicy::default())
            .with_retention_policy(RetentionPolicy::new(vec![
                CleanupRule::KeepOne {
                    per_interval: Duration::from_secs(3600),
                    older_than: MIN_THINNING_AGE,
                },
                CleanupRule::KeepOne {
                    per_interval: Duration::from_secs(86_400),
                    older_than: Duration::from_secs(7 * 86_400),
                },
                CleanupRule::RemoveAll {
                    older_than: Duration::from_secs(30 * 86_400),
                },
            ]))
            .unwrap()
    };
    operation.policy.delete_unverified = delete_unverified;
    let mut expected = vec![3, 4, 6, 8, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20];
    if protected {
        dataset.tags().create("keep-oldest", 1).await.unwrap();
        operation.policy.error_if_tagged_old_versions = false;
        operation.policy.versions = Some(
            (1..=20)
                .filter(|version| ![2, 5].contains(version))
                .collect(),
        );
        expected.extend([1, 2, 5]);
        expected.sort_unstable();
    }
    let before = fixture.count_files().await.unwrap();
    let explanation = operation.explain().await.unwrap();
    assert_eq!(before, fixture.count_files().await.unwrap());
    let orphan_candidate = explanation
        .candidate_files
        .iter()
        .find(|file| file.path == orphan.as_ref());
    assert_eq!(orphan_candidate.is_some(), delete_unverified);
    if let Some(file) = orphan_candidate {
        assert!(file.unverified);
        assert_eq!(file.kind, CleanupFileKind::Data);
    }
    let stats = operation.execute().await.unwrap();
    assert_eq!(stats, explanation.stats);
    assert_eq!(stats.old_versions, 20 - expected.len() as u64);
    assert_eq!(
        stats.data_files_removed,
        stats.old_versions * 2 + u64::from(delete_unverified)
    );
    assert_eq!(
        dataset.object_store.exists(&orphan).await.unwrap(),
        !delete_unverified
    );
    assert_eq!(
        dataset
            .version_refs()
            .await
            .unwrap()
            .iter()
            .map(|version| version.version)
            .collect::<Vec<_>>(),
        expected
    );
    for version in expected {
        let snapshot = dataset.checkout_version(version).await.unwrap();
        assert_eq!(snapshot.get_fragments().len(), 2);
        assert_eq!(
            snapshot.scan().try_into_batch().await.unwrap(),
            expected_data
        );
    }
    assert_eq!(operation.execute().await.unwrap().old_versions, 0);
}

#[tokio::test]
async fn cleanup_retention_bands_have_independent_buckets() {
    let fixture = MockDatasetFixture::try_new_in_memory().unwrap();
    for (index, hour) in [200, 800, 801, 960].into_iter().enumerate() {
        MockClock::set_system_time(Duration::from_secs(hour * 3600));
        if index == 0 {
            fixture.create_some_data().await.unwrap();
        } else {
            fixture.overwrite_some_data().await.unwrap();
        }
    }
    let dataset = fixture.open().await.unwrap();
    let operation = dataset
        .cleanup(CleanupPolicy::default())
        .with_retention_policy(RetentionPolicy::new(vec![
            CleanupRule::KeepOne {
                per_interval: Duration::from_secs(3600),
                older_than: MIN_THINNING_AGE,
            },
            CleanupRule::KeepOne {
                per_interval: Duration::from_secs(900),
                older_than: Duration::from_secs(7 * 86_400),
            },
        ]))
        .unwrap();
    assert_eq!(operation.explain().await.unwrap().stats.old_versions, 0);
    assert_eq!(operation.execute().await.unwrap().old_versions, 0);
    assert_eq!(dataset.version_refs().await.unwrap().len(), 4);
}

#[rstest]
#[case::empty("[]", "must not be empty")]
#[case::duplicate(r#"[{"type":"keep_one","per_interval":"1h","older_than":"2days"},{"type":"keep_one","per_interval":"1day","older_than":"2days"}]"#, "strictly increasing")]
#[case::unsorted(r#"[{"type":"keep_one","per_interval":"1h","older_than":"7days"},{"type":"keep_one","per_interval":"1day","older_than":"2days"}]"#, "strictly increasing")]
#[case::early_remove(r#"[{"type":"remove_all","older_than":"30days"},{"type":"keep_one","per_interval":"1h","older_than":"2days"}]"#, "final retention rule")]
#[case::early_expiration(r#"[{"type":"keep_one","per_interval":"1h","older_than":"2days"},{"type":"remove_all","older_than":"1day"}]"#, "strictly increasing")]
#[case::unknown(r#"[{"type":"keep_all","older_than":"2days"}]"#, "unknown variant")]
#[case::extra_field(
    r#"[{"type":"remove_all","older_than":"30days","per_interval":"1h"}]"#,
    "unknown field"
)]
#[case::missing_field(r#"[{"type":"keep_one","older_than":"2days"}]"#, "missing field")]
#[case::invalid_duration(
    r#"[{"type":"remove_all","older_than":"invalid"}]"#,
    "Invalid cleanup retention policy"
)]
#[case::zero_interval(
    r#"[{"type":"keep_one","per_interval":"0s","older_than":"2days"}]"#,
    "keep_one.per_interval"
)]
#[case::young(
    r#"[{"type":"keep_one","per_interval":"1h","older_than":"1day"}]"#,
    "at least 2 days"
)]
fn cleanup_retention_policy_validation(#[case] value: &str, #[case] message: &str) {
    let error = parse_cleanup_policy(&format!(r#"{{"rules":{value}}}"#)).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert_contains!(error.to_string(), message);
}

#[rstest]
#[case::daily(24, vec![4, 7, 8, 9, 10])]
#[case::twice_daily(12, vec![2, 4, 7, 8, 9, 10])]
#[tokio::test]
async fn cleanup_keeps_one_per_period(
    #[case] period_hours: u64,
    #[case] expected_versions: Vec<u64>,
    #[values(false, true)] delete_unverified: bool,
    #[values(false, true)] auto_config: bool,
) {
    let fixture = MockDatasetFixture::try_new_in_memory().unwrap();
    for (index, hour) in [0, 2, 1, 12, 24, 26, 26, 48, 49, 96]
        .into_iter()
        .enumerate()
    {
        MockClock::set_system_time(Duration::from_secs(hour * 3600));
        let data = lance_datagen::gen_batch()
            .col("value", lance_datagen::array::step::<Int32Type>())
            .into_reader_rows(2.into(), 2.into());
        Dataset::write(
            data,
            &fixture.dataset_path,
            Some(WriteParams {
                mode: if index == 0 {
                    WriteMode::Create
                } else {
                    WriteMode::Overwrite
                },
                max_rows_per_file: 2,
                store_params: Some(fixture.os_params()),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    }
    MockClock::set_system_time(Duration::from_secs(96 * 3600));
    let dataset = fixture.open().await.unwrap();
    let expected_data = dataset.scan().try_into_batch().await.unwrap();
    let policy = CleanupPolicyBuilder::default()
        .delete_unverified(delete_unverified)
        .build();
    let operation = if auto_config {
        let mut manifest = dataset.manifest.as_ref().clone();
        manifest.config.extend([
            ("lance.auto_cleanup.policy_interval".into(), "1".into()),
            (
                "lance.auto_cleanup.keep_one.per_interval".into(),
                format!("{period_hours}h"),
            ),
        ]);
        let mut operation = build_cleanup_operation(&dataset, &manifest, false)
            .await
            .unwrap()
            .unwrap();
        operation.policy.delete_unverified = delete_unverified;
        operation
    } else {
        dataset
            .cleanup(policy)
            .with_keep_one(Duration::from_secs(period_hours * 3600), MIN_THINNING_AGE)
            .unwrap()
    };
    let before = fixture.count_files().await.unwrap();
    let explanation = operation.explain().await.unwrap();
    assert_eq!(before, fixture.count_files().await.unwrap());
    let stats = operation.execute().await.unwrap();
    assert_eq!(stats, explanation.stats);
    assert_eq!(stats.old_versions, 10 - expected_versions.len() as u64);
    assert_eq!(stats.data_files_removed, stats.old_versions * 2);
    assert_eq!(
        dataset
            .version_refs()
            .await
            .unwrap()
            .iter()
            .map(|v| v.version)
            .collect::<Vec<_>>(),
        expected_versions
    );
    for version in expected_versions {
        let snapshot = dataset.checkout_version(version).await.unwrap();
        assert_eq!(snapshot.get_fragments().len(), 2);
        assert_eq!(
            snapshot.scan().try_into_batch().await.unwrap(),
            expected_data
        );
    }
    assert_eq!(operation.execute().await.unwrap().old_versions, 0);
}

#[rstest]
#[case::current_hint(None, vec![5, 7, 9, 10, 11, 12, 13, 14])]
#[case::stale_hint(Some(9), vec![5, 7, 8, 9, 10, 11, 12, 13, 14])]
#[tokio::test]
async fn cleanup_separates_expiration_and_thinning(
    #[case] hinted_version: Option<u64>,
    #[case] mut expected_versions: Vec<u64>,
    #[values(false, true)] auto_config: bool,
    #[values(false, true)] filtered: bool,
) {
    let fixture = MockDatasetFixture::try_new().unwrap();
    for (index, hour) in [0, 1, 24, 25, 72, 96, 97, 168, 169, 192, 193, 216, 217, 240]
        .into_iter()
        .enumerate()
    {
        MockClock::set_system_time(Duration::from_secs(hour * 3600));
        if index == 0 {
            fixture.create_some_data().await.unwrap();
        } else {
            fixture.overwrite_some_data().await.unwrap();
        }
    }
    let dataset = fixture.open().await.unwrap();
    if let Some(version) = hinted_version {
        write_version_hint(&dataset.object_store, &dataset.base, version).await;
    }
    let mut operation = if auto_config {
        let mut manifest = dataset.manifest.as_ref().clone();
        manifest.config.extend([
            ("lance.auto_cleanup.policy_interval".into(), "1".into()),
            ("lance.auto_cleanup.older_than".into(), "7days".into()),
            (
                "lance.auto_cleanup.keep_one.per_interval".into(),
                "1day".into(),
            ),
            (
                "lance.auto_cleanup.keep_one.older_than".into(),
                "2days".into(),
            ),
        ]);
        build_cleanup_operation(&dataset, &manifest, false)
            .await
            .unwrap()
            .unwrap()
    } else {
        dataset
            .cleanup(
                CleanupPolicyBuilder::default()
                    .before_timestamp(utc_now() - TimeDelta::days(7))
                    .build(),
            )
            .with_keep_one(Duration::from_secs(86_400), MIN_THINNING_AGE)
            .unwrap()
    };
    if filtered {
        operation.policy.versions = Some((2..14).filter(|version| *version != 6).collect());
        expected_versions.extend([1, 6]);
        expected_versions.sort_unstable();
    }
    let mut expected_data = HashMap::new();
    for version in &expected_versions {
        expected_data.insert(
            *version,
            dataset
                .checkout_version(*version)
                .await
                .unwrap()
                .scan()
                .try_into_batch()
                .await
                .unwrap(),
        );
    }
    let before = fixture.count_files().await.unwrap();
    let explanation = operation.explain().await.unwrap();
    assert_eq!(before, fixture.count_files().await.unwrap());
    let stats = operation.execute().await.unwrap();
    assert_eq!(stats, explanation.stats);
    assert_eq!(stats.old_versions, 14 - expected_versions.len() as u64);
    assert_eq!(
        dataset
            .version_refs()
            .await
            .unwrap()
            .iter()
            .map(|version| version.version)
            .collect::<Vec<_>>(),
        expected_versions
    );
    for version in expected_versions {
        assert_eq!(
            dataset
                .checkout_version(version)
                .await
                .unwrap()
                .scan()
                .try_into_batch()
                .await
                .unwrap(),
            expected_data[&version]
        );
    }
    assert_eq!(operation.execute().await.unwrap().old_versions, 0);
}

#[tokio::test]
async fn cleanup_thinning_respects_filters_and_tags() {
    MockClock::set_system_time(Duration::ZERO);
    let fixture = MockDatasetFixture::try_new_in_memory().unwrap();
    fixture.create_some_data().await.unwrap();
    fixture.overwrite_some_data().await.unwrap();
    fixture.overwrite_some_data().await.unwrap();
    MockClock::set_system_time(Duration::from_secs(3 * 86_400));
    fixture.overwrite_some_data().await.unwrap();
    let dataset = fixture.open().await.unwrap();
    dataset.tags().create("keep-first", 1).await.unwrap();
    let policy = CleanupPolicyBuilder::default()
        .versions(vec![1, 2, 3])
        .unwrap()
        .build();
    let operation = dataset
        .cleanup(policy.clone())
        .with_keep_one(Duration::from_secs(1), MIN_THINNING_AGE)
        .unwrap();
    let error = operation.execute().await.unwrap_err();
    assert!(matches!(error, Error::Cleanup { .. }));
    assert_contains!(error.to_string(), "keep-first");
    assert_eq!(dataset.version_refs().await.unwrap().len(), 4);
    let operation = dataset
        .cleanup(CleanupPolicy {
            error_if_tagged_old_versions: false,
            ..policy
        })
        .with_keep_one(Duration::from_secs(1), MIN_THINNING_AGE)
        .unwrap();
    assert_eq!(operation.explain().await.unwrap().stats.old_versions, 1);
    assert_eq!(operation.execute().await.unwrap().old_versions, 1);
    assert_eq!(
        dataset
            .version_refs()
            .await
            .unwrap()
            .iter()
            .map(|v| v.version)
            .collect::<Vec<_>>(),
        vec![1, 3, 4]
    );
}

#[tokio::test]
async fn cleanup_thinning_rejects_invalid_period() {
    let fixture = MockDatasetFixture::try_new_in_memory().unwrap();
    fixture.create_some_data().await.unwrap();
    let dataset = fixture.open().await.unwrap();
    let error = dataset
        .cleanup(CleanupPolicy::default())
        .with_keep_one(Duration::ZERO, MIN_THINNING_AGE)
        .err()
        .unwrap();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert_contains!(error.to_string(), "keep_one");
    let mut manifest = dataset.manifest.as_ref().clone();
    manifest
        .config
        .insert("lance.auto_cleanup.policy_interval".into(), "1".into());
    manifest.config.insert(
        "lance.auto_cleanup.keep_one.per_interval".into(),
        "1day".into(),
    );
    let error = build_cleanup_policy(&dataset, &manifest).await.unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert_contains!(error.to_string(), "keep_one");
    for value in ["0s", "invalid"] {
        manifest.config.insert(
            "lance.auto_cleanup.keep_one.per_interval".into(),
            value.into(),
        );
        let error = build_cleanup_operation(&dataset, &manifest, false)
            .await
            .err()
            .unwrap();
        assert_contains!(error.to_string(), "keep_one");
    }
}

#[rstest]
#[tokio::test]
async fn cleanup_thinning_preserves_branch_dependencies(
    #[values(false, true)] delete_unverified: bool,
) {
    MockClock::set_system_time(Duration::ZERO);
    let fixture = MockDatasetFixture::try_new_in_memory().unwrap();
    fixture.create_some_data().await.unwrap();
    let mut parent = fixture.open().await.unwrap();
    let branch = fixture
        .create_branch_and_load(&mut parent, "keep-root", (None, None))
        .await
        .unwrap();
    let expected = branch.scan().try_into_batch().await.unwrap();
    MockClock::set_system_time(Duration::from_secs(3600));
    fixture.overwrite_some_data().await.unwrap();
    MockClock::set_system_time(Duration::from_secs(3 * 86_400));
    fixture.overwrite_some_data().await.unwrap();
    let parent = fixture.open().await.unwrap();
    let operation = parent
        .cleanup(
            CleanupPolicyBuilder::default()
                .delete_unverified(delete_unverified)
                .build(),
        )
        .with_keep_one(Duration::from_secs(86_400), MIN_THINNING_AGE)
        .unwrap();
    assert_eq!(operation.execute().await.unwrap().old_versions, 0);
    let branch = parent.checkout_version(("keep-root", None)).await.unwrap();
    assert_eq!(branch.scan().try_into_batch().await.unwrap(), expected);
    assert_eq!(parent.version_refs().await.unwrap().len(), 3);
}

#[rstest]
#[case::default(None, 1)]
#[case::three_days(Some(Duration::from_secs(3 * 86_400)), 0)]
#[tokio::test]
async fn cleanup_thinning_anchors_to_hint(
    #[case] age: Option<Duration>,
    #[case] expected_removed: u64,
    #[values(false, true)] auto_config: bool,
    #[values(false, true)] pinned_snapshot: bool,
) {
    let fixture = MockDatasetFixture::try_new().unwrap();
    for (index, hour) in [0, 1, 24, 25, 48, 1000, 72, 1, 120].into_iter().enumerate() {
        MockClock::set_system_time(Duration::from_secs(hour * 3600));
        if index == 0 {
            fixture.create_some_data().await.unwrap();
        } else {
            fixture.overwrite_some_data().await.unwrap();
        }
    }
    let dataset = fixture.open().await.unwrap();
    assert!(!dataset.object_store.list_is_lexically_ordered);
    write_version_hint(&dataset.object_store, &dataset.base, 7).await;
    let dataset = if pinned_snapshot {
        Box::new(dataset.checkout_version(6).await.unwrap())
    } else {
        dataset
    };
    MockClock::set_system_time(Duration::from_secs(30 * 86_400));
    let operation = if auto_config {
        let mut manifest = dataset.manifest.as_ref().clone();
        manifest.config.extend([
            ("lance.auto_cleanup.policy_interval".into(), "1".into()),
            (
                "lance.auto_cleanup.keep_one.per_interval".into(),
                "1day".into(),
            ),
        ]);
        if let Some(age) = age {
            manifest.config.insert(
                "lance.auto_cleanup.keep_one.older_than".into(),
                format!("{}s", age.as_secs()),
            );
        }
        build_cleanup_operation(&dataset, &manifest, false)
            .await
            .unwrap()
            .unwrap()
    } else {
        dataset
            .cleanup(CleanupPolicy::default())
            .with_keep_one(Duration::from_secs(86_400), age.unwrap_or(MIN_THINNING_AGE))
            .unwrap()
    };
    let before = fixture.count_files().await.unwrap();
    assert_eq!(
        operation.explain().await.unwrap().stats.old_versions,
        expected_removed
    );
    assert_eq!(before, fixture.count_files().await.unwrap());
    assert_eq!(
        operation.execute().await.unwrap().old_versions,
        expected_removed
    );
    let expected: Vec<_> = ((1 + expected_removed)..=9).collect();
    assert_eq!(
        dataset
            .version_refs()
            .await
            .unwrap()
            .iter()
            .map(|v| v.version)
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(dataset.latest_version_id().await.unwrap(), 9);
    for version in expected {
        dataset
            .checkout_version(version)
            .await
            .unwrap()
            .scan()
            .try_into_batch()
            .await
            .unwrap();
    }
}

#[rstest]
#[case::zero(Duration::ZERO)]
#[case::one_day(Duration::from_secs(86_400))]
#[case::just_below(Duration::from_nanos(2 * 86_400 * 1_000_000_000 - 1))]
#[tokio::test]
async fn cleanup_thinning_requires_two_days(#[case] age: Duration) {
    let fixture = MockDatasetFixture::try_new().unwrap();
    fixture.create_some_data().await.unwrap();
    let dataset = fixture.open().await.unwrap();
    let before = fixture.count_files().await.unwrap();
    let error = dataset
        .cleanup(CleanupPolicy::default())
        .with_keep_one(Duration::from_secs(1), age)
        .err()
        .unwrap();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert_contains!(
        error.to_string(),
        "keep_one.older_than must be at least 2 days"
    );
    let mut operation = dataset
        .cleanup(CleanupPolicy::default())
        .with_keep_one(Duration::from_secs(1), MIN_THINNING_AGE)
        .unwrap();
    operation.retention_policy = Some(vec![CleanupRule::KeepOne {
        per_interval: Duration::from_secs(1),
        older_than: age,
    }]);
    assert!(operation.explain().await.is_err());
    assert!(operation.execute().await.is_err());
    let mut manifest = dataset.manifest.as_ref().clone();
    manifest.config.extend([
        ("lance.auto_cleanup.policy_interval".into(), "1".into()),
        (
            "lance.auto_cleanup.keep_one.per_interval".into(),
            "1s".into(),
        ),
        (
            "lance.auto_cleanup.keep_one.older_than".into(),
            "1day".into(),
        ),
    ]);
    let error = build_cleanup_operation(&dataset, &manifest, false)
        .await
        .err()
        .unwrap();
    assert_contains!(error.to_string(), "older_than must be at least 2 days");
    assert_eq!(before, fixture.count_files().await.unwrap());
}

#[rstest]
#[case::malformed(true, 1)]
#[case::missing_current_manifest(false, 2)]
#[tokio::test]
async fn cleanup_thinning_fails_on_invalid_hint(
    #[case] is_malformed: bool,
    #[case] hinted_version: u64,
) {
    let fixture = MockDatasetFixture::try_new().unwrap();
    fixture.create_some_data().await.unwrap();
    fixture.overwrite_some_data().await.unwrap();
    let dataset = fixture.open().await.unwrap();
    if is_malformed {
        dataset
            .object_store
            .put(
                &dataset
                    .base
                    .clone()
                    .join("_versions")
                    .join("latest_version_hint.json"),
                b"invalid",
            )
            .await
            .unwrap();
    } else {
        write_version_hint(&dataset.object_store, &dataset.base, hinted_version).await;
        let location = dataset
            .commit_handler
            .resolve_version_location(
                &dataset.base,
                hinted_version,
                dataset.object_store.inner.as_ref(),
            )
            .await
            .unwrap();
        dataset.object_store.delete(&location.path).await.unwrap();
    }
    let before = fixture.count_files().await.unwrap();
    let operation = dataset
        .cleanup(CleanupPolicy::default())
        .with_keep_one(Duration::from_secs(1), MIN_THINNING_AGE)
        .unwrap();
    let error = operation.explain().await.unwrap_err();
    if is_malformed {
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert_contains!(error.to_string(), "Invalid version hint");
    } else {
        assert!(error.is_not_found(), "{error}");
    }
    assert!(operation.execute().await.is_err());
    assert_eq!(before, fixture.count_files().await.unwrap());
}

#[tokio::test]
async fn cleanup_thinning_falls_back_when_historical_hint_target_was_expired() {
    let fixture = MockDatasetFixture::try_new().unwrap();
    fixture.create_some_data().await.unwrap();
    fixture.overwrite_some_data().await.unwrap();
    let dataset = fixture.open().await.unwrap();
    write_version_hint(&dataset.object_store, &dataset.base, 1).await;
    let location = dataset
        .commit_handler
        .resolve_version_location(&dataset.base, 1, dataset.object_store.inner.as_ref())
        .await
        .unwrap();
    dataset.object_store.delete(&location.path).await.unwrap();
    let before = fixture.count_files().await.unwrap();
    let operation = dataset
        .cleanup(CleanupPolicy::default())
        .with_keep_one(Duration::from_secs(86_400), MIN_THINNING_AGE)
        .unwrap();
    assert_eq!(operation.explain().await.unwrap().stats.old_versions, 0);
    assert_eq!(operation.execute().await.unwrap().old_versions, 0);
    assert_eq!(before, fixture.count_files().await.unwrap());
}

#[tokio::test]
async fn cleanup_thinning_without_hint_uses_snapshot() {
    let fixture = MockDatasetFixture::try_new().unwrap();
    MockClock::set_system_time(Duration::ZERO);
    fixture.create_some_data().await.unwrap();
    MockClock::set_system_time(Duration::from_secs(3600));
    fixture.overwrite_some_data().await.unwrap();
    MockClock::set_system_time(Duration::from_secs(3 * 86_400));
    fixture.overwrite_some_data().await.unwrap();
    let dataset = fixture.open().await.unwrap();
    dataset
        .object_store
        .delete(
            &dataset
                .base
                .clone()
                .join("_versions")
                .join("latest_version_hint.json"),
        )
        .await
        .unwrap();
    assert_eq!(
        read_version_hint(&dataset.object_store, &dataset.base)
            .await
            .unwrap(),
        None
    );
    MockClock::set_system_time(Duration::from_secs(30 * 86_400));
    let operation = dataset
        .cleanup(CleanupPolicy::default())
        .with_keep_one(Duration::from_secs(86_400), MIN_THINNING_AGE)
        .unwrap();
    assert_eq!(operation.explain().await.unwrap().stats.old_versions, 1);
    assert_eq!(operation.execute().await.unwrap().old_versions, 1);
    assert_eq!(
        dataset
            .version_refs()
            .await
            .unwrap()
            .iter()
            .map(|v| v.version)
            .collect::<Vec<_>>(),
        vec![2, 3]
    );
    assert_eq!(dataset.latest_version_id().await.unwrap(), 3);
}

#[rstest]
#[tokio::test]
async fn cleanup_expiration_uses_wall_clock(#[values(false, true)] thinning: bool) {
    let fixture = MockDatasetFixture::try_new().unwrap();
    MockClock::set_system_time(Duration::ZERO);
    fixture.create_some_data().await.unwrap();
    MockClock::set_system_time(Duration::from_secs(3600));
    fixture.overwrite_some_data().await.unwrap();
    let dataset = fixture.open().await.unwrap();
    write_version_hint(&dataset.object_store, &dataset.base, 1).await;
    MockClock::set_system_time(Duration::from_secs(86_400));
    let mut manifest = dataset.manifest.as_ref().clone();
    manifest.config.extend([
        ("lance.auto_cleanup.interval".into(), "1".into()),
        ("lance.auto_cleanup.older_than".into(), "2days".into()),
    ]);
    if thinning {
        manifest.config.remove("lance.auto_cleanup.interval");
        manifest
            .config
            .insert("lance.auto_cleanup.policy_interval".into(), "1".into());
        manifest.config.insert(
            "lance.auto_cleanup.keep_one.per_interval".into(),
            "1day".into(),
        );
    }
    let operation = build_cleanup_operation(&dataset, &manifest, false)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(operation.explain().await.unwrap().stats.old_versions, 0);
    MockClock::set_system_time(Duration::from_secs(3 * 86_400));
    let operation = build_cleanup_operation(&dataset, &manifest, false)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(operation.explain().await.unwrap().stats.old_versions, 1);
    assert_eq!(operation.execute().await.unwrap().old_versions, 1);
    assert_eq!(dataset.latest_version_id().await.unwrap(), 2);
    assert_eq!(operation.explain().await.unwrap().stats.old_versions, 0);
    assert_eq!(operation.execute().await.unwrap().old_versions, 0);
}

#[tokio::test]
async fn cleanup_thinning_honors_hints_with_updates_disabled() {
    let role = env::var("LANCE_THINNING_TEST_ROLE").ok();
    if role.as_deref() == Some("reader") {
        let dataset = Dataset::open(&env::var("LANCE_THINNING_TEST_URI").unwrap())
            .await
            .unwrap();
        assert_eq!(dataset.latest_version_id().await.unwrap(), 9);
        return;
    }
    let test_name = "dataset::cleanup::tests::cleanup_thinning_honors_hints_with_updates_disabled";
    if role.as_deref() != Some("cleaner") {
        let output = Command::new(env::current_exe().unwrap())
            .args([test_name, "--exact", "--nocapture"])
            .env("LANCE_USE_VERSION_HINT", "0")
            .env("LANCE_THINNING_TEST_ROLE", "cleaner")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let fixture = MockDatasetFixture::try_new().unwrap();
    for (index, hour) in [0, 1, 24, 25, 48, 49, 72, 96, 120].into_iter().enumerate() {
        MockClock::set_system_time(Duration::from_secs(hour * 3600));
        if index == 0 {
            fixture.create_some_data().await.unwrap();
        } else {
            fixture.overwrite_some_data().await.unwrap();
        }
    }
    let dataset = fixture.open().await.unwrap();
    assert!(!uses_version_hint(&dataset.object_store));
    dataset
        .object_store
        .put(
            &dataset
                .base
                .clone()
                .join("_versions")
                .join("latest_version_hint.json"),
            br#"{"version":2}"#,
        )
        .await
        .unwrap();
    let operation = dataset
        .cleanup(CleanupPolicy::default())
        .with_keep_one(Duration::from_secs(86_400), MIN_THINNING_AGE)
        .unwrap();
    assert_eq!(operation.explain().await.unwrap().stats.old_versions, 0);
    assert_eq!(operation.execute().await.unwrap().old_versions, 0);
    assert_eq!(
        dataset
            .version_refs()
            .await
            .unwrap()
            .iter()
            .map(|v| v.version)
            .collect::<Vec<_>>(),
        (1..=9).collect::<Vec<_>>()
    );
    let output = Command::new(env::current_exe().unwrap())
        .args([test_name, "--exact", "--nocapture"])
        .env("LANCE_USE_VERSION_HINT", "1")
        .env("LANCE_THINNING_TEST_ROLE", "reader")
        .env("LANCE_THINNING_TEST_URI", &fixture.dataset_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
