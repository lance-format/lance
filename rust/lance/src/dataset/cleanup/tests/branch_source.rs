// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::dataset::refs::branch_contents_path;

#[rstest]
#[case::deletion_files(false)]
#[case::column_files(true)]
#[tokio::test]
async fn cleanup_keeps_branch_source_readable(
    #[case] drop_column: bool,
    #[values(false, true)] delete_unverified: bool,
    #[values(false, true)] by_version: bool,
) {
    MockClock::set_system_time(Duration::ZERO);
    let fixture = MockDatasetFixture::try_new().unwrap();
    let data = lance_datagen::gen_batch()
        .col(
            "id",
            lance_datagen::array::step::<arrow_array::types::Int32Type>(),
        )
        .into_reader_rows(3.into(), 2.into());
    let mut dataset = Dataset::write(
        data,
        &fixture.dataset_path,
        Some(WriteParams {
            max_rows_per_file: 3,
            store_params: Some(fixture.os_params()),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);
    if drop_column {
        dataset
            .add_columns(
                crate::dataset::NewColumnTransform::SqlExpressions(vec![(
                    "added".into(),
                    "id * 2".into(),
                )]),
                None,
                None,
            )
            .await
            .unwrap();
    } else {
        dataset.delete("id = 0").await.unwrap();
    }
    let source_version = dataset.version_id();
    let expected = dataset.scan().try_into_batch().await.unwrap();
    let mut branch = fixture
        .create_branch_and_load(&mut dataset, "child", (None, None))
        .await
        .unwrap();
    if drop_column {
        branch.drop_columns(&["added"]).await.unwrap();
        dataset.drop_columns(&["added"]).await.unwrap();
    } else {
        branch.delete("id = 1").await.unwrap();
        dataset.delete("id = 2").await.unwrap();
    }
    MockClock::advance_system_time(Duration::from_secs(1));
    let policy = |version| CleanupPolicy {
        before_timestamp: (!by_version).then(utc_now),
        before_version: by_version.then_some(version),
        delete_unverified,
        ..Default::default()
    };
    branch
        .cleanup(policy(branch.version_id()))
        .execute()
        .await
        .unwrap();
    assert_eq!(branch.versions().await.unwrap().len(), 1);

    let operation = dataset.cleanup(policy(dataset.version_id()));
    let explanation = operation.explain().await.unwrap();
    let removed = operation.execute().await.unwrap();
    assert_eq!(explanation.stats, removed);
    assert_eq!(
        dataset
            .versions()
            .await
            .unwrap()
            .iter()
            .map(|version| version.version)
            .collect::<Vec<_>>(),
        vec![source_version, dataset.version_id()]
    );
    let mut source = fixture
        .load()
        .await
        .unwrap()
        .checkout_version(source_version)
        .await
        .unwrap();
    assert_eq!(source.scan().try_into_batch().await.unwrap(), expected);
    source.restore().await.unwrap();
    let restored = fixture.load().await.unwrap();
    assert_eq!(restored.scan().try_into_batch().await.unwrap(), expected);
}

#[tokio::test]
async fn cleanup_keeps_files_read_by_legacy_branches() {
    let fixture = MockDatasetFixture::try_new().unwrap();
    let write_params = |mode| {
        Some(WriteParams {
            mode,
            store_params: Some(fixture.os_params()),
            ..Default::default()
        })
    };
    // main:v1 -> legacy:v2 -> legacy/child. Overwriting main and legacy leaves the
    // first data file of each read only by legacy/child.
    fixture.create_some_data().await.unwrap();
    let mut main = fixture.load().await.unwrap();
    let mut legacy = fixture
        .create_branch_and_load(&mut main, "legacy", 1)
        .await
        .unwrap();
    legacy
        .append(some_batch(), write_params(WriteMode::Append))
        .await
        .unwrap();
    let child = fixture
        .create_branch_and_load(&mut legacy, "legacy/child", ("legacy", 2))
        .await
        .unwrap();
    let expected = child.scan().try_into_batch().await.unwrap();
    let legacy = Dataset::write(
        some_batch(),
        Arc::new(legacy),
        write_params(WriteMode::Overwrite),
    )
    .await
    .unwrap();
    fixture.overwrite_some_data().await.unwrap();
    // Lance 2.x and earlier write branch files with neither `identifier` nor `metadata`.
    for name in ["legacy", "legacy/child"] {
        let path = branch_contents_path(&main.base, name);
        let mut contents: serde_json::Value =
            serde_json::from_slice(&main.object_store.read_one_all(&path).await.unwrap()).unwrap();
        let contents_map = contents.as_object_mut().unwrap();
        contents_map.remove("identifier");
        contents_map.remove("metadata");
        main.object_store
            .put(&path, &serde_json::to_vec(&contents).unwrap())
            .await
            .unwrap();
    }

    MockClock::set_system_time(TimeDelta::try_days(10).unwrap().to_std().unwrap());
    let before = utc_now() - TimeDelta::try_days(8).unwrap();
    let policy = CleanupPolicyBuilder::default()
        .before_timestamp(before)
        .build();
    // legacy:v2 is kept for legacy/child and v3 is current, so only v1 goes.
    let removed = cleanup_old_versions(&legacy, policy).await.unwrap();
    assert_eq!(removed.old_versions, 1);
    assert_eq!(removed.data_files_removed, 0);
    // main:v1 is kept for legacy and legacy/child.
    let removed = fixture.run_cleanup(before).await.unwrap();
    assert_eq!(removed.old_versions, 0);
    assert_eq!(removed.data_files_removed, 0);

    let mut main = fixture.load().await.unwrap();
    let child = main.checkout_version(("legacy/child", None)).await.unwrap();
    assert_eq!(child.scan().try_into_batch().await.unwrap(), expected);
    let error = main.delete_branch("legacy").await.unwrap_err();
    assert!(matches!(error, Error::RefConflict { .. }), "{error}");
    assert_contains!(error.to_string(), "legacy/child");
}
