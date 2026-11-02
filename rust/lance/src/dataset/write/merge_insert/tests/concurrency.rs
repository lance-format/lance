// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

// For some reason, Windows isn't able to handle the timeout test. Possibly
// a performance bug in their timer implementation?
#[cfg(not(windows))]
#[rstest::rstest]
#[case::all_success(Duration::from_secs(100_000))]
#[case::timeout(Duration::from_millis(200))]
#[tokio::test(start_paused = true)]
async fn test_merge_insert_concurrency(#[case] timeout: Duration) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false),
        Field::new("value", DataType::UInt32, false),
    ]));
    // To benchmark scaling curve: measure how long to run
    //
    // And vary `concurrency` to see how it scales. Compare this again `main`.
    let concurrency = 10;
    let initial_data = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from_iter_values(0..concurrency)),
            Arc::new(UInt32Array::from_iter_values(std::iter::repeat_n(
                0,
                concurrency as usize,
            ))),
        ],
    )
    .unwrap();

    // Increase likelihood of contention by throttling the store
    let throttled = Arc::new(ThrottledStoreWrapper {
        config: ThrottleConfig {
            // For benchmarking: Increase this to simulate object storage.
            wait_list_per_call: Duration::from_millis(20),
            wait_get_per_call: Duration::from_millis(20),
            wait_put_per_call: Duration::from_millis(20),
            ..Default::default()
        },
    });
    let session = Arc::new(Session::default());

    let mut dataset = InsertBuilder::new("memory://")
        .with_params(&WriteParams {
            store_params: Some(ObjectStoreParams {
                object_store_wrapper: Some(throttled.clone()),
                ..Default::default()
            }),
            session: Some(session.clone()),
            ..Default::default()
        })
        .execute(vec![initial_data])
        .await
        .unwrap();

    // do merge inserts in parallel based on the concurrency. Each will open the dataset,
    // signal they have opened, and then wait for a signal to proceed. Once the signal
    // is received, they will do a merge insert and close the dataset.

    let barrier = Arc::new(Barrier::new(concurrency as usize));
    let mut handles = Vec::new();
    for i in 0..concurrency {
        let session_ref = session.clone();
        let schema_ref = schema.clone();
        let barrier_ref = barrier.clone();
        let throttled_ref = throttled.clone();
        let handle = tokio::task::spawn(async move {
            let dataset = DatasetBuilder::from_uri("memory://")
                .with_read_params(ReadParams {
                    store_options: Some(ObjectStoreParams {
                        object_store_wrapper: Some(throttled_ref.clone()),
                        ..Default::default()
                    }),
                    session: Some(session_ref.clone()),
                    ..Default::default()
                })
                .load()
                .await
                .unwrap();
            let dataset = Arc::new(dataset);

            let new_data = RecordBatch::try_new(
                schema_ref.clone(),
                vec![
                    Arc::new(UInt32Array::from(vec![i])),
                    Arc::new(UInt32Array::from(vec![1])),
                ],
            )
            .unwrap();
            let source = Box::new(RecordBatchIterator::new([Ok(new_data)], schema_ref.clone()));

            let job = MergeInsertBuilder::try_new(dataset, vec!["id".to_string()])
                .unwrap()
                .when_matched(WhenMatched::UpdateAll)
                .when_not_matched(WhenNotMatched::InsertAll)
                .conflict_retries(100)
                .retry_timeout(timeout)
                .try_build()
                .unwrap();
            barrier_ref.wait().await;

            job.execute_reader(source)
                .await
                .map(|(_ds, stats)| stats.num_attempts)
        });
        handles.push(handle);
    }

    let results = try_join_all(handles).await.unwrap();

    for attempts in results.iter() {
        match attempts {
            Ok(attempts) => {
                assert!(*attempts <= 10, "Attempt count should be <= 10");
            }
            Err(err) => {
                // If we get an error, it means the task was cancelled
                // due to timeout. This is expected if the timeout is
                // set to a low value.
                assert!(
                    matches!(err, Error::TooMuchWriteContention { message, .. } if message.contains("failed on retry_timeout")),
                    "Expected TooMuchWriteContention error, got: {:?}",
                    err
                );
            }
        }
    }

    if timeout.as_secs() > 10 {
        dataset.checkout_latest().await.unwrap();
        let batches = dataset.scan().try_into_batch().await.unwrap();

        let values = batches["value"].as_primitive::<UInt32Type>();
        assert!(
            values.values().iter().all(|&v| v == 1),
            "All values should be 1 after merge insert. Got: {:?}",
            values
        );
    }
}

#[tokio::test]
async fn test_merge_insert_large_concurrent() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false),
        Field::new("value", DataType::UInt32, false),
    ]));
    let num_rows = 10;
    let initial_data = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from_iter_values(0..num_rows)),
            Arc::new(UInt32Array::from_iter_values(std::iter::repeat_n(
                0,
                num_rows as usize,
            ))),
        ],
    )
    .unwrap();

    // Adding latency helps ensure we get contention
    let throttled = Arc::new(ThrottledStoreWrapper {
        config: ThrottleConfig {
            wait_list_per_call: Duration::from_millis(10),
            wait_get_per_call: Duration::from_millis(10),
            ..Default::default()
        },
    });
    let session = Arc::new(Session::default());

    let dataset = InsertBuilder::new("memory://")
        .with_params(&WriteParams {
            store_params: Some(ObjectStoreParams {
                object_store_wrapper: Some(throttled.clone()),
                ..Default::default()
            }),
            session: Some(session.clone()),
            ..Default::default()
        })
        .execute(vec![initial_data])
        .await
        .unwrap();
    let dataset = Arc::new(dataset);

    // Start one merge insert, but don't commit it yet.
    let new_data1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![1])),
            Arc::new(UInt32Array::from(vec![1])),
        ],
    )
    .unwrap();
    let UncommittedMergeInsert {
        transaction: transaction1,
        ..
    } = MergeInsertBuilder::try_new(dataset.clone(), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap()
        .execute_uncommitted(RecordBatchIterator::new(
            vec![Ok(new_data1)],
            schema.clone(),
        ))
        .await
        .unwrap();

    // Setup a "large" merge insert, with many batches
    let new_data2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from_iter_values(0..1000)),
            Arc::new(UInt32Array::from_iter_values(std::iter::repeat_n(2, 1000))),
        ],
    )
    .unwrap();
    let notify = Arc::new(Notify::new());
    let source = RecordBatchIterator::new(
        (0..10)
            .map(|i| {
                let batch = new_data2.slice(i * 100, 100);
                if i == 9 {
                    notify.notify_one();
                }
                Ok(batch)
            })
            .collect::<Vec<_>>(),
        schema.clone(),
    );
    let dataset2 = DatasetBuilder::from_uri("memory://")
        .with_read_params(ReadParams {
            store_options: Some(ObjectStoreParams {
                object_store_wrapper: Some(throttled.clone()),
                ..Default::default()
            }),
            session: Some(session.clone()),
            ..Default::default()
        })
        .load()
        .await
        .unwrap();
    let job = MergeInsertBuilder::try_new(Arc::new(dataset2), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap()
        .execute_reader(source);
    let task = tokio::task::spawn(job);

    // Right as the large merge insert has finished reading the last batch,
    // we will commit the first merge insert. This should trigger a conflict,
    // but we should resolve it automatically.
    notify.notified().await;
    let mut dataset = CommitBuilder::new(dataset)
        .execute(transaction1)
        .await
        .unwrap();

    task.await.unwrap().unwrap();
    dataset.checkout_latest().await.unwrap();

    let batches = dataset.scan().try_into_batch().await.unwrap();
    let values = batches["value"].as_primitive::<UInt32Type>();
    assert!(
        values.values().iter().all(|&v| v == 2),
        "All values should be 1 after merge insert. Got: {:?}",
        values
    );
}

#[tokio::test]
async fn test_merge_insert_updates_indices() {
    let test_dataset = async || {
        let mut dataset = lance_datagen::gen_batch()
            .col("id", array::step::<UInt32Type>())
            .col("value", array::step::<UInt32Type>())
            .col("other_value", array::step::<UInt32Type>())
            .into_ram_dataset(FragmentCount::from(4), FragmentRowCount::from(20))
            .await
            .unwrap();

        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        dataset
            .create_index(
                &["value"],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        dataset
            .create_index(
                &["other_value"],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        Arc::new(dataset)
    };

    let check_indices = async |dataset: &Dataset, id_frags: &[u32], value_frags: &[u32]| {
        let id_index = dataset
            .load_scalar_index(IndexCriteria::default().with_name("id_idx"))
            .await
            .unwrap();

        if id_frags.is_empty() {
            assert!(id_index.is_none());
        } else {
            let id_index = id_index.unwrap();
            let id_frags_bitmap = RoaringBitmap::from_iter(id_frags.iter().copied());
            // Check the effective bitmap (raw bitmap intersected with existing fragments)
            let effective_bitmap = id_index
                .effective_fragment_bitmap(&dataset.fragment_bitmap)
                .unwrap();
            assert_eq!(effective_bitmap, id_frags_bitmap);
        }

        let value_index = dataset
            .load_scalar_index(IndexCriteria::default().with_name("value_idx"))
            .await
            .unwrap();

        if value_frags.is_empty() {
            assert!(value_index.is_none());
        } else {
            let value_index = value_index.unwrap();
            let value_frags_bitmap = RoaringBitmap::from_iter(value_frags.iter().copied());
            // Check the effective bitmap (raw bitmap intersected with existing fragments)
            let effective_bitmap = value_index
                .effective_fragment_bitmap(&dataset.fragment_bitmap)
                .unwrap();
            assert_eq!(effective_bitmap, value_frags_bitmap);
        }

        let other_value_index = dataset
            .load_scalar_index(IndexCriteria::default().with_name("other_value_idx"))
            .await
            .unwrap()
            .unwrap();

        // The other_value index retains its original bitmap [0,1,2,3] since
        // partial merges that don't modify other_value won't prune it.
        let effective_bitmap = other_value_index
            .effective_fragment_bitmap(&dataset.fragment_bitmap)
            .unwrap();

        // The effective bitmap is the intersection of the index's original bitmap
        // and the current dataset fragments. Since other_value is not modified by
        // partial merges, it retains its validity for fragments it was originally trained on
        // that still exist in the dataset.
        let index_bitmap = other_value_index.fragment_bitmap.as_ref().unwrap();
        let expected_bitmap = index_bitmap & dataset.fragment_bitmap.as_ref();
        assert_eq!(
            effective_bitmap, expected_bitmap,
            "other_value index effective bitmap should be intersection. index_bitmap: {:?}, dataset_fragments: {:?}, effective_bitmap: {:?}",
            index_bitmap, dataset.fragment_bitmap, effective_bitmap
        );
    };

    let dataset = test_dataset().await;

    // Sanity test on the initial dataset
    check_indices(&dataset, &[0, 1, 2, 3], &[0, 1, 2, 3]).await;

    // Vertical merge insert (full schema), one fragment is deleted and should be removed from
    // the index.
    let merge_insert = MergeInsertBuilder::try_new(dataset, vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap();

    let (dataset, _) = merge_insert
        .execute_reader(
            lance_datagen::gen_batch()
                .col("id", array::step_custom::<UInt32Type>(50, 1))
                .col("value", array::step_custom::<UInt32Type>(50, 1))
                .col("other_value", array::step_custom::<UInt32Type>(50, 1))
                .into_df_stream(RowCount::from(40), BatchCount::from(1)),
        )
        .await
        .unwrap();

    // Fragment 3 removed and correctly removed from the index bitmap.
    check_indices(&dataset, &[0, 1, 2], &[0, 1, 2]).await;

    // Now we do the same thing with a partial merge insert (only id and value)
    let dataset = test_dataset().await;

    // Vertical merge insert (full schema), one fragment is deleted and should be removed from
    // the index.
    let merge_insert = MergeInsertBuilder::try_new(dataset, vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap();

    let (dataset, _) = merge_insert
        .execute_reader(
            lance_datagen::gen_batch()
                .col("id", array::step_custom::<UInt32Type>(50, 1))
                .col("value", array::step_custom::<UInt32Type>(50, 1))
                .into_df_stream(RowCount::from(40), BatchCount::from(1)),
        )
        .await
        .unwrap();

    // Join keys stay in their original files, so their index keeps every
    // fragment. Only the rewritten value column loses index coverage.
    check_indices(&dataset, &[0, 1, 2, 3], &[0, 1]).await;

    // Touching all fragments invalidates the value index entirely while
    // preserving the join key index.
    let dataset = test_dataset().await;

    // Vertical merge insert (full schema), one fragment is deleted and should be removed from
    // the index.
    let merge_insert = MergeInsertBuilder::try_new(dataset, vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap();

    let (dataset, _) = merge_insert
        .execute_reader(
            lance_datagen::gen_batch()
                .col("id", array::step_custom::<UInt32Type>(10, 1))
                .col("value", array::step_custom::<UInt32Type>(10, 1))
                .into_df_stream(RowCount::from(80), BatchCount::from(1)),
        )
        .await
        .unwrap();

    check_indices(&dataset, &[0, 1, 2, 3], &[]).await;
}

#[tokio::test]
async fn test_upsert_concurrent_full_frag() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false),
        Field::new("value", DataType::UInt32, false),
    ]));
    let initial_data = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![0, 1])),
            Arc::new(UInt32Array::from(vec![0, 0])),
        ],
    )
    .unwrap();

    // Increase likelihood of contention by throttling the store
    let throttled = Arc::new(ThrottledStoreWrapper {
        config: ThrottleConfig {
            wait_list_per_call: Duration::from_millis(5),
            wait_get_per_call: Duration::from_millis(5),
            wait_put_per_call: Duration::from_millis(5),
            ..Default::default()
        },
    });
    let session = Arc::new(Session::default());

    let mut dataset = InsertBuilder::new("memory://")
        .with_params(&WriteParams {
            store_params: Some(ObjectStoreParams {
                object_store_wrapper: Some(throttled.clone()),
                ..Default::default()
            }),
            session: Some(session.clone()),
            ..Default::default()
        })
        .execute(vec![initial_data])
        .await
        .unwrap();

    // Each merge insert will update one row. Combined, they should delete
    // all rows in the first fragment, and it should be dropped.
    let barrier = Arc::new(Barrier::new(2));
    let mut handles = Vec::new();
    for i in 0..2 {
        let new_data = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![i])),
                Arc::new(UInt32Array::from(vec![1])),
            ],
        )
        .unwrap();
        let source = Box::new(RecordBatchIterator::new([Ok(new_data)], schema.clone()));

        let dataset_ref = Arc::new(dataset.clone());
        let barrier = barrier.clone();
        let handle = tokio::spawn(async move {
            barrier.wait().await;
            MergeInsertBuilder::try_new(dataset_ref, vec!["id".to_string()])
                .unwrap()
                .when_matched(WhenMatched::UpdateAll)
                .when_not_matched(WhenNotMatched::InsertAll)
                .try_build()
                .unwrap()
                .execute_reader(source)
                .await
                .unwrap();
        });
        handles.push(handle);
    }
    try_join_all(handles).await.unwrap();

    dataset.checkout_latest().await.unwrap();
    assert!(
        dataset
            .get_fragments()
            .iter()
            .all(|f| f.metadata().num_rows().unwrap() > 0),
        "No fragments should have zero rows after upsert"
    );

    let batches = dataset.scan().try_into_batch().await.unwrap();
    let values = batches["value"].as_primitive::<UInt32Type>();
    assert!(
        values.values().iter().all(|&v| v == 1),
        "All values should be 1 after merge insert. Got: {:?}",
        values
    );
}

#[tokio::test]
async fn test_skip_auto_cleanup() {
    let tmpdir = TempStrDir::default();
    let dataset_uri = format!("{}/{}", tmpdir, "test_dataset");

    // Create initial dataset with auto cleanup interval of 1 version
    let data = lance_datagen::gen_batch()
        .with_seed(Seed::from(1))
        .col("id", array::step::<UInt32Type>())
        .into_reader_rows(RowCount::from(100), BatchCount::from(1));

    let mut auto_cleanup_params = HashMap::new();
    auto_cleanup_params.insert("lance.auto_cleanup.interval".to_string(), "1".to_string());
    auto_cleanup_params.insert(
        "lance.auto_cleanup.older_than".to_string(),
        "0ms".to_string(),
    );

    let write_params = WriteParams {
        mode: WriteMode::Create,
        auto_cleanup: Some(crate::dataset::AutoCleanupParams {
            interval: 1,
            older_than: chrono::TimeDelta::try_milliseconds(0).unwrap(),
        }),
        ..Default::default()
    };

    // Start at 1 second after epoch
    MockClock::set_system_time(std::time::Duration::from_secs(1));

    let dataset = Dataset::write(data, &dataset_uri, Some(write_params))
        .await
        .unwrap();
    assert_eq!(dataset.version().version, 1);

    // Advance time
    MockClock::set_system_time(std::time::Duration::from_secs(2));

    // First merge insert WITHOUT skip_auto_cleanup - should trigger cleanup
    let new_data = lance_datagen::gen_batch()
        .with_seed(Seed::from(2))
        .col("id", array::step::<UInt32Type>())
        .into_df_stream(RowCount::from(50), BatchCount::from(1));

    let (dataset2, _) = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap()
        .execute(new_data)
        .await
        .unwrap();

    assert_eq!(dataset2.version().version, 2);

    // Advance time
    MockClock::set_system_time(std::time::Duration::from_secs(3));

    // Need to do another merge insert for cleanup to take effect since cleanup runs on the old dataset
    let new_data_extra = lance_datagen::gen_batch()
        .with_seed(Seed::from(4))
        .col("id", array::step::<UInt32Type>())
        .into_df_stream(RowCount::from(10), BatchCount::from(1));

    let (dataset2_extra, _) = MergeInsertBuilder::try_new(dataset2.clone(), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap()
        .execute(new_data_extra)
        .await
        .unwrap();

    assert_eq!(dataset2_extra.version().version, 3);

    // Load the dataset from disk to check versions
    let ds_check1 = DatasetBuilder::from_uri(&dataset_uri).load().await.unwrap();

    // Version 1 should be cleaned up due to auto cleanup (cleanup runs every version)
    assert!(
        ds_check1.checkout_version(1).await.is_err(),
        "Version 1 should have been cleaned up"
    );
    // Version 2 should still exist
    assert!(
        ds_check1.checkout_version(2).await.is_ok(),
        "Version 2 should still exist"
    );

    // Advance time
    MockClock::set_system_time(std::time::Duration::from_secs(4));

    // Second merge insert WITH skip_auto_cleanup - should NOT trigger cleanup
    let new_data2 = lance_datagen::gen_batch()
        .with_seed(Seed::from(3))
        .col("id", array::step::<UInt32Type>())
        .into_df_stream(RowCount::from(30), BatchCount::from(1));

    let (dataset3, _) = MergeInsertBuilder::try_new(dataset2_extra, vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .skip_auto_cleanup(true) // Skip auto cleanup
        .try_build()
        .unwrap()
        .execute(new_data2)
        .await
        .unwrap();

    assert_eq!(dataset3.version().version, 4);

    // Load the dataset from disk to check versions
    let ds_check2 = DatasetBuilder::from_uri(&dataset_uri).load().await.unwrap();

    // Version 2 should still exist because skip_auto_cleanup was enabled
    assert!(
        ds_check2.checkout_version(2).await.is_ok(),
        "Version 2 should still exist because skip_auto_cleanup was enabled"
    );
    // Version 3 should also still exist
    assert!(
        ds_check2.checkout_version(3).await.is_ok(),
        "Version 3 should still exist"
    );
}

#[tokio::test]
async fn test_transaction_inserted_rows_filter_roundtrip() {
    // Create dataset with unenforced primary key on "id" column
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false).with_metadata(
            vec![(
                "lance-schema:unenforced-primary-key".to_string(),
                "true".to_string(),
            )]
            .into_iter()
            .collect(),
        ),
        Field::new("value", DataType::UInt32, false),
    ]));
    let initial = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![0, 1, 2])),
            Arc::new(UInt32Array::from(vec![0, 0, 0])),
        ],
    )
    .unwrap();
    let dataset = InsertBuilder::new("memory://")
        .execute(vec![initial])
        .await
        .unwrap();
    let dataset = Arc::new(dataset);

    // Source with overlapping key 1
    let new_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![1, 3])),
            Arc::new(UInt32Array::from(vec![2, 2])),
        ],
    )
    .unwrap();
    let stream =
        RecordBatchStreamAdapter::new(schema.clone(), futures::stream::iter(vec![Ok(new_batch)]));

    let UncommittedMergeInsert { transaction, .. } =
        MergeInsertBuilder::try_new(dataset.clone(), vec!["id".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .try_build()
            .unwrap()
            .execute_uncommitted(Box::pin(stream) as SendableRecordBatchStream)
            .await
            .unwrap();

    // Commit and read back transaction file
    let committed = CommitBuilder::new(dataset.clone())
        .execute(transaction)
        .await
        .unwrap();
    let tx_path = committed.manifest().transaction_file.clone().unwrap();
    let tx_read = read_transaction_file(dataset.object_store.as_ref(), &dataset.base, &tx_path)
        .await
        .unwrap();
    // Check that inserted_rows_filter is present in the Operation::Update
    if let Operation::Update {
        inserted_rows_filter,
        ..
    } = &tx_read.operation
    {
        assert!(inserted_rows_filter.is_some());
        let filter = inserted_rows_filter.as_ref().unwrap();
        // Field IDs are assigned by Lance schema; check that we tracked exactly 1 key field
        assert_eq!(filter.field_ids.len(), 1);
    } else {
        panic!("Expected Operation::Update");
    }
}

/// Test that two merge insert operations on the same existing key conflict.
/// First merge insert commits successfully, second one fails with conflict error
/// because both operations updated the same key (detected via bloom filter).
#[tokio::test]
async fn test_inserted_rows_filter_bloom_conflict_detection_concurrent() {
    // Create schema with unenforced primary key on "id" column
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false).with_metadata(
            vec![(
                "lance-schema:unenforced-primary-key".to_string(),
                "true".to_string(),
            )]
            .into_iter()
            .collect(),
        ),
        Field::new("value", DataType::UInt32, false),
    ]));
    let initial = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![0, 1, 2, 3])),
            Arc::new(UInt32Array::from(vec![0, 0, 0, 0])),
        ],
    )
    .unwrap();

    let dataset = InsertBuilder::new("memory://")
        .execute(vec![initial])
        .await
        .unwrap();
    let dataset = Arc::new(dataset);

    // Both jobs update/insert the same key 2
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![2])),
            Arc::new(UInt32Array::from(vec![1])),
        ],
    )
    .unwrap();
    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![2])),
            Arc::new(UInt32Array::from(vec![2])),
        ],
    )
    .unwrap();

    // Create second merge insert job based on version 1 with 0 retries
    let b2 = MergeInsertBuilder::try_new(dataset.clone(), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .conflict_retries(0)
        .try_build()
        .unwrap();

    // First merge insert commits (creates version 2)
    let s1 = RecordBatchStreamAdapter::new(
        schema.clone(),
        futures::stream::iter(vec![Ok(batch1.clone())]),
    );
    let b1 = MergeInsertBuilder::try_new(dataset.clone(), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap();
    let result1 = b1.execute(Box::pin(s1) as SendableRecordBatchStream).await;
    assert!(result1.is_ok(), "First merge insert should succeed");

    // Second merge insert tries to commit based on version 1, needs to rebase against version 2
    let s2 = RecordBatchStreamAdapter::new(
        schema.clone(),
        futures::stream::iter(vec![Ok(batch2.clone())]),
    );
    let result2 = b2.execute(Box::pin(s2) as SendableRecordBatchStream).await;

    // Second merge insert should fail because bloom filters show both updated key 2
    assert!(
        matches!(result2, Err(crate::Error::TooMuchWriteContention { .. })),
        "Expected TooMuchWriteContention (retryable conflict exhausted), got: {:?}",
        result2
    );
}

/// Test that two merge insert operations inserting the same NEW key conflict.
/// First merge insert commits successfully (inserts id=100), second one fails
/// with conflict error because both inserted the same new key (detected via bloom filter).
#[tokio::test]
async fn test_concurrent_insert_same_new_key() {
    // Create schema with unenforced primary key on "id" column
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false).with_metadata(
            vec![(
                "lance-schema:unenforced-primary-key".to_string(),
                "true".to_string(),
            )]
            .into_iter()
            .collect(),
        ),
        Field::new("value", DataType::UInt32, false),
    ]));
    // Initial dataset with ids 0, 1, 2, 3 - NOT containing id=100
    let initial = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![0, 1, 2, 3])),
            Arc::new(UInt32Array::from(vec![0, 0, 0, 0])),
        ],
    )
    .unwrap();

    let dataset = InsertBuilder::new("memory://")
        .execute(vec![initial])
        .await
        .unwrap();
    let dataset = Arc::new(dataset);

    // Both jobs try to INSERT the same NEW key id=100 (doesn't exist in initial data)
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![100])), // NEW key id=100
            Arc::new(UInt32Array::from(vec![1])),
        ],
    )
    .unwrap();
    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![100])), // Same NEW key id=100
            Arc::new(UInt32Array::from(vec![2])),
        ],
    )
    .unwrap();

    // Create second merge insert job based on version 1 with 0 retries
    let b2 = MergeInsertBuilder::try_new(dataset.clone(), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .conflict_retries(0)
        .try_build()
        .unwrap();

    // First merge insert commits (creates version 2, inserts id=100)
    let s1 = RecordBatchStreamAdapter::new(
        schema.clone(),
        futures::stream::iter(vec![Ok(batch1.clone())]),
    );
    let b1 = MergeInsertBuilder::try_new(dataset.clone(), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap();
    let result1 = b1.execute(Box::pin(s1) as SendableRecordBatchStream).await;
    assert!(result1.is_ok(), "First merge insert should succeed");

    // Second merge insert tries to commit based on version 1, needs to rebase against version 2
    let s2 = RecordBatchStreamAdapter::new(
        schema.clone(),
        futures::stream::iter(vec![Ok(batch2.clone())]),
    );
    let result2 = b2.execute(Box::pin(s2) as SendableRecordBatchStream).await;

    // Second merge insert should fail because bloom filters show both inserted key 100
    assert!(
        matches!(result2, Err(crate::Error::TooMuchWriteContention { .. })),
        "Expected TooMuchWriteContention (retryable conflict exhausted), got: {:?}",
        result2
    );
}

/// Concurrency regression for lance-format/lance#6441: two concurrent
/// find-or-create jobs (`WhenMatched::DoNothing` + `WhenNotMatched::InsertAll`)
/// both try to insert the same fresh key. The second must fail with
/// `TooMuchWriteContention` because the bloom-filter-backed
/// `inserted_rows_filter` detects the overlap during rebase. Before
/// routing find-or-create through v2 this did not work at all: the v1
/// path returned `inserted_rows_filter=None`, so there was nothing to
/// intersect against during conflict resolution.
#[tokio::test]
async fn test_concurrent_find_or_create_same_new_key() {
    // Schema with an unenforced primary key on "id" — that is what
    // activates bloom-filter conflict detection.
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false).with_metadata(
            vec![(
                "lance-schema:unenforced-primary-key".to_string(),
                "true".to_string(),
            )]
            .into_iter()
            .collect(),
        ),
        Field::new("value", DataType::UInt32, false),
    ]));
    // Initial dataset with ids 0..=3 — id=100 is not present.
    let initial = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![0, 1, 2, 3])),
            Arc::new(UInt32Array::from(vec![0, 0, 0, 0])),
        ],
    )
    .unwrap();

    let dataset = InsertBuilder::new("memory://")
        .execute(vec![initial])
        .await
        .unwrap();
    let dataset = Arc::new(dataset);

    // Both jobs try to find-or-create the same new id=100.
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![100])),
            Arc::new(UInt32Array::from(vec![1])),
        ],
    )
    .unwrap();
    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![100])),
            Arc::new(UInt32Array::from(vec![2])),
        ],
    )
    .unwrap();

    // b2 is built against version 1 with zero retries, so when it needs
    // to rebase against b1's commit the bloom-filter intersection decides
    // the outcome directly.
    let b2 = MergeInsertBuilder::try_new(dataset.clone(), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::DoNothing)
        .when_not_matched(WhenNotMatched::InsertAll)
        .conflict_retries(0)
        .try_build()
        .unwrap();

    // First job commits successfully, producing version 2 with id=100.
    let s1 = RecordBatchStreamAdapter::new(
        schema.clone(),
        futures::stream::iter(vec![Ok(batch1.clone())]),
    );
    let b1 = MergeInsertBuilder::try_new(dataset.clone(), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::DoNothing)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap();
    let result1 = b1.execute(Box::pin(s1) as SendableRecordBatchStream).await;
    assert!(result1.is_ok(), "First find-or-create should succeed");

    // Second job fails because its inserted_rows_filter overlaps b1's.
    let s2 = RecordBatchStreamAdapter::new(
        schema.clone(),
        futures::stream::iter(vec![Ok(batch2.clone())]),
    );
    let result2 = b2.execute(Box::pin(s2) as SendableRecordBatchStream).await;

    assert!(
        matches!(result2, Err(crate::Error::TooMuchWriteContention { .. })),
        "Expected TooMuchWriteContention (bloom-filter conflict) for find-or-create, got: {:?}",
        result2
    );
}

#[test]
fn test_concurrent_insert_different_new_list_key() {
    // Schema for list(string) key column "tags".
    let tags_field = Field::new(
        "tags",
        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
        false,
    );
    let schema = Arc::new(Schema::new(vec![tags_field]));

    // Build two batches inserting list key ["a", "b"] and ["c", "d"].
    let mut builder = ListBuilder::new(StringBuilder::new());
    builder.append_value(["a", "b"].iter().copied().map(Some));
    let tags_array1 = builder.finish();
    let batch1 = RecordBatch::try_new(schema.clone(), vec![Arc::new(tags_array1)]).unwrap();

    let mut builder = ListBuilder::new(StringBuilder::new());
    builder.append_value(["c", "d"].iter().copied().map(Some));
    let tags_array2 = builder.finish();
    let batch2 = RecordBatch::try_new(schema, vec![Arc::new(tags_array2)]).unwrap();

    // Build bloom filters for the list keys.
    let field_ids = vec![0_i32];
    let mut builder1 = KeyExistenceFilterBuilder::new(field_ids.clone());
    let mut builder2 = KeyExistenceFilterBuilder::new(field_ids);

    let key1 = extract_key_value_from_batch(&batch1, 0, &[String::from("tags")])
        .expect("first batch should produce key");
    let key2 = extract_key_value_from_batch(&batch2, 0, &[String::from("tags")])
        .expect("second batch should produce key");

    builder1.insert(key1).unwrap();
    builder2.insert(key2).unwrap();
    let filter1 = KeyExistenceFilter::from_bloom_filter(&builder1);
    let filter2 = KeyExistenceFilter::from_bloom_filter(&builder2);

    let (has_intersection, might_be_fp) = filter1.intersects(&filter2).unwrap();
    assert!(
        !has_intersection,
        "Expected bloom filters not intersect for different list(string) keys",
    );
    assert!(
        !might_be_fp,
        "Bloom filter intersection should be definitively not conflict",
    );
}

#[test]
fn test_concurrent_insert_same_new_list_key() {
    // Schema for list(string) key column "tags".
    let tags_field = Field::new(
        "tags",
        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
        false,
    );
    let schema = Arc::new(Schema::new(vec![tags_field]));

    // Build two batches both inserting the same list key ["a", "b"].
    let mut builder = ListBuilder::new(StringBuilder::new());
    builder.append_value(["a", "b"].iter().copied().map(Some));
    let tags_array1 = builder.finish();
    let batch1 = RecordBatch::try_new(schema.clone(), vec![Arc::new(tags_array1)]).unwrap();

    let mut builder = ListBuilder::new(StringBuilder::new());
    builder.append_value(["a", "b"].iter().copied().map(Some));
    let tags_array2 = builder.finish();
    let batch2 = RecordBatch::try_new(schema, vec![Arc::new(tags_array2)]).unwrap();

    // Build bloom filters for the list key.
    let field_ids = vec![0_i32];
    let mut builder1 = KeyExistenceFilterBuilder::new(field_ids.clone());
    let mut builder2 = KeyExistenceFilterBuilder::new(field_ids);

    let key1 = extract_key_value_from_batch(&batch1, 0, &[String::from("tags")])
        .expect("first batch should produce key");
    let key2 = extract_key_value_from_batch(&batch2, 0, &[String::from("tags")])
        .expect("second batch should produce key");

    builder1.insert(key1).unwrap();
    builder2.insert(key2).unwrap();
    let filter1 = KeyExistenceFilter::from_bloom_filter(&builder1);
    let filter2 = KeyExistenceFilter::from_bloom_filter(&builder2);

    let (has_intersection, might_be_fp) = filter1.intersects(&filter2).unwrap();
    assert!(
        has_intersection,
        "Expected bloom filters to intersect for identical list(string) keys",
    );
    assert!(
        might_be_fp,
        "Bloom filter intersection should be treated as potential conflict",
    );
}

#[test]
fn test_concurrent_insert_same_new_nested_list_key() {
    // Build nested list(list(string)) value [["a", "b"], ["c"]] for the "tags" column.
    let nested_tags = make_nested_array(&[["a", "b"].as_slice(), ["c"].as_slice()]);
    let tags_field = Field::new("tags", nested_tags.data_type().clone(), false);
    let nested_tags2 = make_nested_array(&[["a", "b"].as_slice(), ["c"].as_slice()]);

    let schema = Arc::new(Schema::new(vec![tags_field]));
    let batch1 = RecordBatch::try_new(schema.clone(), vec![Arc::new(nested_tags)]).unwrap();
    let batch2 = RecordBatch::try_new(schema, vec![Arc::new(nested_tags2)]).unwrap();

    // Build bloom filters for the nested list key.
    let field_ids = vec![0_i32];
    let mut builder1 = KeyExistenceFilterBuilder::new(field_ids.clone());
    let mut builder2 = KeyExistenceFilterBuilder::new(field_ids);

    let key1 = extract_key_value_from_batch(&batch1, 0, &[String::from("tags")])
        .expect("first batch should produce key");
    let key2 = extract_key_value_from_batch(&batch2, 0, &[String::from("tags")])
        .expect("second batch should produce key");

    builder1.insert(key1).unwrap();
    builder2.insert(key2).unwrap();
    let filter1 = KeyExistenceFilter::from_bloom_filter(&builder1);
    let filter2 = KeyExistenceFilter::from_bloom_filter(&builder2);

    let (has_intersection, might_be_fp) = filter1.intersects(&filter2).unwrap();
    assert!(
        has_intersection,
        "Expected bloom filters to intersect for identical nested list(list(string)) keys",
    );
    assert!(
        might_be_fp,
        "Bloom filter intersection should be treated as potential conflict",
    );
}

#[test]
fn test_concurrent_insert_different_new_struct_key() {
    let user_field = Field::new(
        "user",
        DataType::Struct(
            vec![
                Field::new("first", DataType::Utf8, false),
                Field::new("last", DataType::Utf8, false),
            ]
            .into(),
        ),
        false,
    );
    let schema = Arc::new(Schema::new(vec![user_field]));

    // Build two batches inserting different struct keys.
    let struct_array1 = make_struct_array_first_last_name(vec!["alice"], vec!["smith"]);
    let batch1 = RecordBatch::try_new(schema.clone(), vec![Arc::new(struct_array1)]).unwrap();

    let struct_array2 = make_struct_array_first_last_name(vec!["bob"], vec!["jones"]);
    let batch2 = RecordBatch::try_new(schema, vec![Arc::new(struct_array2)]).unwrap();

    // Build bloom filters for the struct key.
    let field_ids = vec![0_i32];
    let mut builder1 = KeyExistenceFilterBuilder::new(field_ids.clone());
    let mut builder2 = KeyExistenceFilterBuilder::new(field_ids);

    let key1 = extract_key_value_from_batch(&batch1, 0, &[String::from("user")])
        .expect("first batch should produce key");
    let key2 = extract_key_value_from_batch(&batch2, 0, &[String::from("user")])
        .expect("second batch should produce key");

    builder1.insert(key1).unwrap();
    builder2.insert(key2).unwrap();
    let filter1 = KeyExistenceFilter::from_bloom_filter(&builder1);
    let filter2 = KeyExistenceFilter::from_bloom_filter(&builder2);

    let (has_intersection, might_be_fp) = filter1.intersects(&filter2).unwrap();
    assert!(
        !has_intersection,
        "Expected bloom filters not intersect for different struct keys",
    );
    assert!(
        !might_be_fp,
        "Bloom filter intersection should be definitively not conflict",
    );
}

#[test]
fn test_concurrent_insert_same_new_struct_key() {
    let user_field = Field::new(
        "user",
        DataType::Struct(
            vec![
                Field::new("first", DataType::Utf8, false),
                Field::new("last", DataType::Utf8, false),
            ]
            .into(),
        ),
        false,
    );
    let schema = Arc::new(Schema::new(vec![user_field]));

    // Build two batches both inserting the same struct key {first: "alice", last: "smith"}.
    let struct_array1 = make_struct_array_first_last_name(vec!["alice"], vec!["smith"]);
    let batch1 = RecordBatch::try_new(schema.clone(), vec![Arc::new(struct_array1)]).unwrap();

    let struct_array2 = make_struct_array_first_last_name(vec!["alice"], vec!["smith"]);
    let batch2 = RecordBatch::try_new(schema, vec![Arc::new(struct_array2)]).unwrap();

    // Build bloom filters for the struct key.
    let field_ids = vec![0_i32];
    let mut builder1 = KeyExistenceFilterBuilder::new(field_ids.clone());
    let mut builder2 = KeyExistenceFilterBuilder::new(field_ids);

    let key1 = extract_key_value_from_batch(&batch1, 0, &[String::from("user")])
        .expect("first batch should produce key");
    let key2 = extract_key_value_from_batch(&batch2, 0, &[String::from("user")])
        .expect("second batch should produce key");

    builder1.insert(key1).unwrap();
    builder2.insert(key2).unwrap();
    let filter1 = KeyExistenceFilter::from_bloom_filter(&builder1);
    let filter2 = KeyExistenceFilter::from_bloom_filter(&builder2);

    let (has_intersection, might_be_fp) = filter1.intersects(&filter2).unwrap();
    assert!(
        has_intersection,
        "Expected bloom filters to intersect for identical struct keys",
    );
    assert!(
        might_be_fp,
        "Bloom filter intersection should be treated as potential conflict",
    );
}

#[test]
fn test_concurrent_insert_same_new_nested_struct_key() {
    // Build nested struct value {address: {city: "seattle", zip: 98101}} for the "user" column.
    let outer_struct = make_nested_struct_array_city_zip("seattle", 98101);
    let user_field = Field::new("user", outer_struct.data_type().clone(), false);
    let schema = Arc::new(Schema::new(vec![user_field]));

    let batch1 = RecordBatch::try_new(schema.clone(), vec![Arc::new(outer_struct)]).unwrap();

    let outer_struct2 = make_nested_struct_array_city_zip("seattle", 98101);
    let batch2 = RecordBatch::try_new(schema, vec![Arc::new(outer_struct2)]).unwrap();

    // Build bloom filters for the nested struct key.
    let field_ids = vec![0_i32];
    let mut builder1 = KeyExistenceFilterBuilder::new(field_ids.clone());
    let mut builder2 = KeyExistenceFilterBuilder::new(field_ids);

    let key1 = extract_key_value_from_batch(&batch1, 0, &[String::from("user")])
        .expect("first batch should produce key");
    let key2 = extract_key_value_from_batch(&batch2, 0, &[String::from("user")])
        .expect("second batch should produce key");

    builder1.insert(key1).unwrap();
    builder2.insert(key2).unwrap();
    let filter1 = KeyExistenceFilter::from_bloom_filter(&builder1);
    let filter2 = KeyExistenceFilter::from_bloom_filter(&builder2);

    let (has_intersection, might_be_fp) = filter1.intersects(&filter2).unwrap();
    assert!(
        has_intersection,
        "Expected bloom filters to intersect for identical nested struct keys",
    );
    assert!(
        might_be_fp,
        "Bloom filter intersection should be treated as potential conflict",
    );
}

/// End-to-end test for merge_insert using a struct-typed key column.
#[tokio::test]
async fn test_merge_insert_struct_key_upsert() {
    let user_field = Field::new(
        "user",
        DataType::Struct(
            vec![
                Field::new("first", DataType::Utf8, false),
                Field::new("last", DataType::Utf8, false),
            ]
            .into(),
        ),
        false,
    );
    let schema = Arc::new(Schema::new(vec![
        user_field,
        Field::new("value", DataType::UInt32, false),
    ]));

    // Initial dataset:
    // (alice, smith) -> 1
    // (bob, jones)  -> 1
    // (carla, doe)  -> 1
    let user_array = make_struct_array_first_last_name(
        vec!["alice", "bob", "carla"],
        vec!["smith", "jones", "doe"],
    );
    let values = UInt32Array::from(vec![1, 1, 1]);
    let initial_batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(user_array), Arc::new(values)]).unwrap();

    let test_uri = "memory://test_merge_insert_struct_key.lance";
    let dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(initial_batch)], schema.clone()),
        test_uri,
        None,
    )
    .await
    .unwrap();
    let dataset = Arc::new(dataset);

    // New data: update alice, insert david
    let new_user_array =
        make_struct_array_first_last_name(vec!["alice", "david"], vec!["smith", "brown"]);
    let new_values = UInt32Array::from(vec![10, 2]);
    let new_batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(new_user_array), Arc::new(new_values)],
    )
    .unwrap();

    let reader = RecordBatchIterator::new([Ok(new_batch)], schema.clone());
    let (merged_ds, stats) = MergeInsertBuilder::try_new(dataset, vec!["user".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap()
        .execute(reader_to_stream(Box::new(reader)))
        .await
        .unwrap();

    assert_eq!(stats.num_updated_rows, 1);
    assert_eq!(stats.num_inserted_rows, 1);
    assert_eq!(stats.num_deleted_rows, 0);

    let result = merged_ds.scan().try_into_batch().await.unwrap();
    let user_col = result
        .column_by_name("user")
        .unwrap()
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    let first = user_col
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let last = user_col
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let values = result
        .column_by_name("value")
        .unwrap()
        .as_primitive::<UInt32Type>();

    let mut rows = Vec::new();
    for i in 0..result.num_rows() {
        rows.push((
            first.value(i).to_string(),
            last.value(i).to_string(),
            values.value(i),
        ));
    }
    rows.sort();

    assert_eq!(
        rows,
        vec![
            ("alice".to_string(), "smith".to_string(), 10),
            ("bob".to_string(), "jones".to_string(), 1),
            ("carla".to_string(), "doe".to_string(), 1),
            ("david".to_string(), "brown".to_string(), 2),
        ],
    );
}

fn make_struct_array_first_last_name(first: Vec<&str>, last: Vec<&str>) -> StructArray {
    let first = StringArray::from(first);
    let last = StringArray::from(last);

    StructArray::from(vec![
        (
            Arc::new(Field::new("first", DataType::Utf8, false)),
            Arc::new(first) as Arc<dyn Array>,
        ),
        (
            Arc::new(Field::new("last", DataType::Utf8, false)),
            Arc::new(last) as Arc<dyn Array>,
        ),
    ])
}

fn make_nested_struct_array_city_zip(city: &str, zip: i32) -> StructArray {
    let city = StringArray::from(vec![city]);
    let zip = Int32Array::from(vec![zip]);

    let inner_struct = StructArray::from(vec![
        (
            Arc::new(Field::new("city", DataType::Utf8, false)),
            Arc::new(city) as Arc<dyn Array>,
        ),
        (
            Arc::new(Field::new("zip", DataType::Int32, false)),
            Arc::new(zip) as Arc<dyn Array>,
        ),
    ]);

    StructArray::from(vec![(
        Arc::new(Field::new(
            "address",
            inner_struct.data_type().clone(),
            false,
        )),
        Arc::new(inner_struct) as Arc<dyn Array>,
    )])
}

fn make_nested_array(inner_lists: &[&[&str]]) -> ListArray {
    let mut inner_builder = ListBuilder::new(StringBuilder::new());
    for inner in inner_lists {
        inner_builder.append_value(inner.iter().map(|s| Some(*s)));
    }
    let inner_list_array = inner_builder.finish();

    let offsets = ScalarBuffer::<i32>::from(vec![0, inner_list_array.len() as i32]);
    let offsets = OffsetBuffer::new(offsets);
    ListArray::new(
        Arc::new(Field::new(
            "item",
            inner_list_array.data_type().clone(),
            inner_list_array.nulls().is_some(),
        )),
        offsets,
        Arc::new(inner_list_array),
        None,
    )
}

/// Test that merge_insert with bloom filter fails when committing against
/// an Update transaction that doesn't have a filter. We can't determine if
/// the Update operation conflicted with our inserted rows.
#[tokio::test]
async fn test_merge_insert_conflict_with_update_without_filter() {
    use crate::dataset::UpdateBuilder;

    // Create schema with unenforced primary key on "id" column
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false).with_metadata(
            vec![(
                "lance-schema:unenforced-primary-key".to_string(),
                "true".to_string(),
            )]
            .into_iter()
            .collect(),
        ),
        Field::new("value", DataType::UInt32, false),
    ]));
    let initial = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![0, 1, 2, 3])),
            Arc::new(UInt32Array::from(vec![0, 0, 0, 0])),
        ],
    )
    .unwrap();

    let dataset = InsertBuilder::new("memory://")
        .execute(vec![initial])
        .await
        .unwrap();
    let dataset = Arc::new(dataset);

    // Create merge insert job based on version 1
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![100])),
            Arc::new(UInt32Array::from(vec![1])),
        ],
    )
    .unwrap();

    let b1 = MergeInsertBuilder::try_new(dataset.clone(), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .conflict_retries(0)
        .try_build()
        .unwrap();

    // Regular Update without bloom filter commits first (creates version 2)
    let update_result = UpdateBuilder::new(dataset.clone())
        .update_where("id = 0")
        .unwrap()
        .set("value", "999")
        .unwrap()
        .build()
        .unwrap()
        .execute()
        .await;
    assert!(update_result.is_ok(), "Update should succeed");

    // Now merge insert tries to commit based on version 1, needs to rebase against version 2
    let s1 = RecordBatchStreamAdapter::new(
        schema.clone(),
        futures::stream::iter(vec![Ok(batch1.clone())]),
    );
    let merge_result = b1.execute(Box::pin(s1) as SendableRecordBatchStream).await;

    // Merge insert should fail with retryable conflict because it can't
    // determine if Update conflicted (Update has no inserted_rows_filter)
    assert!(
        matches!(
            merge_result,
            Err(crate::Error::TooMuchWriteContention { .. })
        ),
        "Expected TooMuchWriteContention (retryable conflict exhausted), got: {:?}",
        merge_result
    );
}

/// Test that merge_insert with bloom filter fails when committing against
/// an Append operation. We can't determine if the appended rows conflict
/// with our inserted rows.
#[tokio::test]
async fn test_merge_insert_conflict_with_append() {
    // Create schema with unenforced primary key on "id" column
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false).with_metadata(
            vec![(
                "lance-schema:unenforced-primary-key".to_string(),
                "true".to_string(),
            )]
            .into_iter()
            .collect(),
        ),
        Field::new("value", DataType::UInt32, false),
    ]));
    let initial = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![0, 1, 2, 3])),
            Arc::new(UInt32Array::from(vec![0, 0, 0, 0])),
        ],
    )
    .unwrap();

    let dataset = InsertBuilder::new("memory://")
        .execute(vec![initial])
        .await
        .unwrap();
    let dataset = Arc::new(dataset);

    // Create merge insert job based on version 1
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![100])),
            Arc::new(UInt32Array::from(vec![1])),
        ],
    )
    .unwrap();

    let b1 = MergeInsertBuilder::try_new(dataset.clone(), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .conflict_retries(0)
        .try_build()
        .unwrap();

    // Append commits first (creates version 2)
    let append_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![50])),
            Arc::new(UInt32Array::from(vec![2])),
        ],
    )
    .unwrap();
    let append_result = InsertBuilder::new(dataset.clone())
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute(vec![append_batch])
        .await;
    assert!(append_result.is_ok(), "Append should succeed");

    // Now merge insert tries to commit based on version 1, needs to rebase against version 2
    let s1 = RecordBatchStreamAdapter::new(
        schema.clone(),
        futures::stream::iter(vec![Ok(batch1.clone())]),
    );
    let merge_result = b1.execute(Box::pin(s1) as SendableRecordBatchStream).await;

    // Merge insert should fail with retryable conflict because it can't
    // determine if Append added conflicting keys
    assert!(
        matches!(
            merge_result,
            Err(crate::Error::TooMuchWriteContention { .. })
        ),
        "Expected TooMuchWriteContention (retryable conflict exhausted), got: {:?}",
        merge_result
    );
}

// Regression test: after a partial-schema merge_insert invalidates a fragment,
// compaction should succeed and subsequent searches should return correct results.
//
// The compaction planner separates indexed and unindexed fragments into different
// groups. After invalidating the middle fragment, the indexed fragments on either
// side form separate compactable groups. After compaction the old invalidated
// fragment ID may remain in invalidated_fragment_bitmap but this is harmless
// because the fragment no longer exists and no index results reference it.
#[tokio::test]
async fn test_compaction_after_invalidated_fragment() {
    use crate::dataset::optimize::{CompactionOptions, compact_files};

    // Use 5 small fragments so that after invalidating the middle one (fragment 2),
    // the planner has enough neighbors to form compactable groups on each side:
    // {0,1} (indexed) and {3,4} (indexed), with {2} (unindexed) separate.
    let rows_per_frag = 20;
    let num_frags = 5;
    let total_rows = rows_per_frag * num_frags;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("value_a", DataType::Float64, false),
        Field::new("value_b", DataType::Float64, false),
    ]));

    let make_batch = |frag_idx: usize| {
        let start = frag_idx * rows_per_frag;
        let ids: Vec<String> = (start..start + rows_per_frag)
            .map(|j| format!("id-{j:04}"))
            .collect();
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(ids)),
                Arc::new(StringArray::from(vec!["A"; rows_per_frag])),
                Arc::new(Float64Array::from(
                    (0..rows_per_frag).map(|i| i as f64).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    (0..rows_per_frag)
                        .map(|i| i as f64 * 0.1)
                        .collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap()
    };

    let batch0 = make_batch(0);
    let reader = Box::new(RecordBatchIterator::new([Ok(batch0)], schema.clone()));
    let mut ds = Dataset::write(reader, "memory://compaction_test", None)
        .await
        .unwrap();
    for frag_idx in 1..num_frags {
        let batch = make_batch(frag_idx);
        let reader = Box::new(RecordBatchIterator::new([Ok(batch)], schema.clone()));
        ds.append(reader, None).await.unwrap();
    }
    ds.create_index(
        &["id"],
        IndexType::BTree,
        None,
        &ScalarIndexParams::default(),
        false,
    )
    .await
    .unwrap();

    ds.create_index(
        &["value_a"],
        IndexType::BTree,
        None,
        &ScalarIndexParams::default(),
        false,
    )
    .await
    .unwrap();
    let ds = Arc::new(ds);

    // Invalidate fragment 2 (the middle one) in the rewritten value's index.
    let frag2_start = 2 * rows_per_frag;
    let ds = partial_merge_insert(ds, frag2_start..frag2_start + rows_per_frag, 999.0).await;

    // Verify pre-compaction state
    let indices = ds.load_indices().await.unwrap();
    let idx = indices.iter().find(|i| i.name == "value_a_idx").unwrap();
    assert!(!idx.fragment_bitmap.as_ref().unwrap().contains(2));
    let key_idx = indices.iter().find(|i| i.name == "id_idx").unwrap();
    assert!(key_idx.fragment_bitmap.as_ref().unwrap().contains(2));

    // Run compaction with a target that forces merging of the small fragments.
    let mut ds = (*ds).clone();
    let opts = CompactionOptions {
        target_rows_per_fragment: total_rows,
        ..Default::default()
    };
    compact_files(&mut ds, opts, None).await.unwrap();

    // The indexed fragments (0,1 and 3,4) should be compacted.
    // Fragment 2 (unindexed) may or may not be compacted on its own.
    // Either way, the old fragment IDs in the bitmap should be replaced.
    let indices = ds.load_indices().await.unwrap();
    let idx = indices.iter().find(|i| i.name == "value_a_idx").unwrap();
    let bitmap = idx.fragment_bitmap.as_ref().unwrap();
    for &old_id in &[0u32, 1, 3, 4] {
        assert!(
            !bitmap.contains(old_id),
            "Old indexed fragment {} should not be in bitmap after compaction",
            old_id
        );
    }
    assert!(
        !bitmap.is_empty(),
        "Bitmap should have new compacted fragments"
    );

    // The invalidated bitmap may still reference old fragment 2.
    // This is harmless — fragment 2 no longer exists (or was compacted into
    // a new fragment), so blocking it is a no-op.

    // Verify search works correctly despite stale invalidated entries.
    let ds = Arc::new(ds);
    let ds = partial_merge_insert(ds, frag2_start..frag2_start + rows_per_frag, 888.0).await;

    // All rows present
    let batches = ds
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let combined = concat_batches(&schema, &batches).unwrap();
    assert_eq!(combined.num_rows(), total_rows);

    // Updated rows have correct value
    let result = ds
        .scan()
        .filter(&format!(
            "id >= 'id-{:04}' AND id < 'id-{:04}'",
            frag2_start,
            frag2_start + rows_per_frag
        ))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let result = concat_batches(&schema, &result).unwrap();
    assert_eq!(result.num_rows(), rows_per_frag);
    let values = result
        .column_by_name("value_a")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    for i in 0..rows_per_frag {
        assert_eq!(values.value(i), 888.0, "row {i} should have value_a=888.0");
    }
}

fn count_data_files(base_dir: &str) -> usize {
    let data_dir = std::path::Path::new(base_dir).join("data");
    if !data_dir.exists() {
        return 0;
    }
    std::fs::read_dir(data_dir)
        .unwrap()
        .filter(|e| e.as_ref().unwrap().path().is_file())
        .count()
}

/// Site 3 in PR #6320: when `MergeInsertJob::apply_deletions` fails after
/// the new fragments have been written, the new data files must be cleaned up.
#[tokio::test]
async fn test_merge_insert_cleans_up_data_on_apply_deletions_failure() {
    use crate::utils::test::FailingProxyStore;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let initial = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from_iter_values(0..30)),
            Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
                "foo", 30,
            ))),
        ],
    )
    .unwrap();

    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();
    // Prefix `/` so Windows drive letters (e.g. `C:`) don't get parsed as
    // the URL authority.
    let path_prefix = if test_uri.starts_with('/') { "" } else { "/" };
    let routed_uri = format!("file-object-store://{path_prefix}{test_uri}");

    let batches = RecordBatchIterator::new([Ok(initial)], schema.clone());
    let mut dataset = Dataset::write(
        batches,
        &routed_uri,
        Some(WriteParams {
            max_rows_per_file: 10,
            data_storage_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    // Create a scalar index on the join key. This forces the merge insert
    // to take the slow (non-fast) path, which is the path that has the
    // post-write cleanup we want to exercise.
    dataset
        .create_index(
            &["id"],
            IndexType::Scalar,
            None,
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();

    let baseline_files = count_data_files(test_uri);
    assert!(baseline_files > 0);

    let failing = Arc::new(FailingProxyStore::new());
    failing.fail_when("put", "_deletions", "injected deletions failure");
    failing.fail_when("put_multipart", "_deletions", "injected deletions failure");

    let dataset = DatasetBuilder::from_uri(&routed_uri)
        .with_read_params(ReadParams {
            store_options: Some(ObjectStoreParams {
                object_store_wrapper: Some(failing.clone()),
                ..Default::default()
            }),
            ..Default::default()
        })
        .load()
        .await
        .unwrap();

    // Update existing keys (5..15 already exist) to force the apply_deletions path.
    let new_data = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from_iter_values(5..15)),
            Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
                "bar", 10,
            ))),
        ],
    )
    .unwrap();
    let new_reader = Box::new(RecordBatchIterator::new([Ok(new_data)], schema.clone()));

    let job = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .try_build()
        .unwrap();

    let result = job.execute_reader(new_reader).await;
    assert!(
        result.is_err(),
        "Merge insert should fail when deletion-file write fails"
    );

    assert_eq!(
        count_data_files(test_uri),
        baseline_files,
        "Newly written merge-insert data files should be cleaned up on apply_deletions failure"
    );
}
