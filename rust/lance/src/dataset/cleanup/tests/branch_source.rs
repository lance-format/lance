// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

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
