// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

#[tokio::test]
async fn test_concurrent_overwrites_retryable() {
    let dataset = test_dataset(5, 1).await;
    let dataset_v1_reader1 = Arc::new(dataset.checkout_version(1).await.unwrap());
    let dataset_v1_reader2 = Arc::new(dataset.checkout_version(1).await.unwrap());

    let data = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, true),
        ])),
        vec![
            Arc::new(Int32Array::from_iter_values(10..15)),
            Arc::new(Int32Array::from_iter_values(std::iter::repeat_n(1, 5))),
        ],
    )
    .unwrap();

    // First overwrite succeeds
    let txn1 = InsertBuilder::new(dataset_v1_reader1.clone())
        .with_params(&WriteParams {
            mode: WriteMode::Overwrite,
            ..Default::default()
        })
        .execute_uncommitted(vec![data.clone()])
        .await
        .unwrap();
    let dataset_v2 = CommitBuilder::new(dataset_v1_reader1)
        .execute(txn1)
        .await
        .unwrap();
    assert_eq!(dataset_v2.manifest.version, 2);

    // Second overwrite should fail with retryable conflict
    let txn2 = InsertBuilder::new(dataset_v1_reader2.clone())
        .with_params(&WriteParams {
            mode: WriteMode::Overwrite,
            ..Default::default()
        })
        .execute_uncommitted(vec![data])
        .await
        .unwrap();
    let result = CommitBuilder::new(dataset_v1_reader2).execute(txn2).await;
    assert!(
        matches!(result, Err(Error::RetryableCommitConflict { .. })),
        "Expected RetryableCommitConflict but got: {:?}",
        result
    );

    assert_eq!(dataset_v2.count_rows(None).await.unwrap(), 5);
}
