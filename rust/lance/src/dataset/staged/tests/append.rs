// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

#[rstest]
#[case::batch(false)]
#[case::stream(true)]
#[tokio::test]
async fn test_append_params_preserve_layout_and_format(
    #[case] streaming: bool,
    #[values(false, true)] stable: bool,
    #[values(None, Some(false), Some(true))] stable_option: Option<bool>,
) {
    let dir = TempStrDir::default();
    let dataset = InsertBuilder::new(dir.as_str())
        .with_params(&WriteParams {
            enable_stable_row_ids: stable,
            data_storage_version: Some(LanceFileVersion::V2_0),
            ..Default::default()
        })
        .execute(vec![record_batch!(("id", Int32, [0])).unwrap()])
        .await
        .unwrap();
    let mut tx = dataset.begin_transaction().await.unwrap();
    let progress = Arc::new(std::sync::Mutex::new(Vec::new()));
    let reported = progress.clone();
    let mut params = WriteParams {
        mode: WriteMode::Append,
        max_rows_per_file: 2,
        max_rows_per_group: 1,
        data_storage_version: Some(LanceFileVersion::V2_1),
        write_progress: Some(WriteProgressFn::new(move |stats| {
            reported.lock().unwrap().push(stats);
        })),
        ..Default::default()
    };
    if let Some(stable_option) = stable_option {
        params.enable_stable_row_ids = stable_option;
    }
    let batch = record_batch!(("id", Int32, [1, 2, 3, 4, 5])).unwrap();
    let ordinary = InsertBuilder::new("memory://")
        .with_params(&WriteParams {
            enable_stable_row_ids: stable,
            data_storage_version: Some(LanceFileVersion::V2_0),
            ..Default::default()
        })
        .execute(vec![record_batch!(("id", Int32, [0])).unwrap()])
        .await
        .unwrap();
    let ordinary = InsertBuilder::new(Arc::new(ordinary))
        .with_params(&WriteParams {
            write_progress: None,
            ..params.clone()
        })
        .execute(vec![batch.clone()])
        .await
        .unwrap();
    if streaming {
        let source =
            RecordBatchStreamAdapter::new(batch.schema(), futures::stream::iter(vec![Ok(batch)]));
        tx.append_stream_with_params(Box::pin(source) as SendableRecordBatchStream, &params)
            .await
            .unwrap();
    } else {
        tx.append_with_params(vec![batch], &params).await.unwrap();
    }
    let first = tx.scan().unwrap();
    let snapshot = tx.staged_dataset().unwrap();
    assert_eq!(
        snapshot
            .manifest
            .fragments
            .iter()
            .map(|fragment| fragment.physical_rows.unwrap())
            .collect::<Vec<_>>(),
        vec![1, 2, 2, 1]
    );
    for fragment in snapshot.manifest.fragments.iter().skip(1) {
        assert_eq!(
            fragment.files[0].file_version().unwrap(),
            ConcreteFileVersion::V2_1
        );
    }
    assert_eq!(progress.lock().unwrap().last().unwrap().rows_written, 5);
    tx.append(vec![record_batch!(("id", Int32, [6])).unwrap()])
        .await
        .unwrap();
    assert_eq!(first.try_into_batch().await.unwrap().num_rows(), 6);
    let committed = tx.commit().await.unwrap();
    assert_eq!(committed.version().version, dataset.version().version + 1);
    let reopened = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(reopened.manifest.fragments, committed.manifest.fragments);
    assert_eq!(
        reopened.manifest.data_storage_format,
        dataset.manifest.data_storage_format
    );
    assert_eq!(reopened.manifest.uses_stable_row_ids(), stable);
    let ordinary = InsertBuilder::new(Arc::new(ordinary))
        .with_params(&append_params())
        .execute(vec![record_batch!(("id", Int32, [6])).unwrap()])
        .await
        .unwrap();
    assert_eq!(
        reopened.manifest.uses_stable_row_ids(),
        ordinary.manifest.uses_stable_row_ids()
    );
    assert_eq!(reopened.manifest.next_row_id, ordinary.manifest.next_row_id);
    assert_eq!(
        reopened.manifest.data_storage_format,
        ordinary.manifest.data_storage_format
    );
    assert_eq!(
        reopened
            .scan()
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap(),
        ordinary
            .scan()
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap()
    );
    assert_eq!(
        reopened.scan().try_into_batch().await.unwrap(),
        record_batch!(("id", Int32, [0, 1, 2, 3, 4, 5, 6])).unwrap()
    );
    if stable {
        assert_eq!(
            reopened
                .count_rows(Some("_rowid = id".into()))
                .await
                .unwrap(),
            7
        );
    }
}

#[rstest]
#[case::create("mode_create")]
#[case::overwrite("mode_overwrite")]
#[case::handler("commit_handler")]
#[case::cleanup("auto_cleanup")]
#[case::properties("transaction_properties")]
#[case::session("session")]
#[case::base_stores("base_store_params")]
#[case::initial_bases("initial_bases")]
#[case::manifest_paths("enable_v2_manifest_paths")]
#[case::skip_cleanup("skip_auto_cleanup")]
#[tokio::test]
async fn test_unsupported_append_params_do_not_write(
    #[case] field: &str,
    #[values(false, true)] streaming: bool,
    #[values(false, true)] stable: bool,
) {
    let dir = TempStrDir::default();
    let dataset = InsertBuilder::new(dir.as_str())
        .with_params(&WriteParams {
            enable_stable_row_ids: stable,
            ..Default::default()
        })
        .execute(vec![record_batch!(("id", Int32, [0])).unwrap()])
        .await
        .unwrap();
    let mut tx = dataset.begin_transaction().await.unwrap();
    tx.append(vec![record_batch!(("id", Int32, [1])).unwrap()])
        .await
        .unwrap();
    let previous = tx.staged_dataset().unwrap();
    let mut params = WriteParams {
        mode: WriteMode::Append,
        enable_stable_row_ids: stable,
        write_progress: Some(WriteProgressFn::new(|_| {
            panic!("invalid params reached writer")
        })),
        ..Default::default()
    };
    match field {
        "mode_create" => params.mode = WriteMode::Create,
        "mode_overwrite" => params.mode = WriteMode::Overwrite,
        "commit_handler" => params.commit_handler = Some(dataset.commit_handler.clone()),
        "auto_cleanup" => params.auto_cleanup = Some(Default::default()),
        "transaction_properties" => params.transaction_properties = Some(Default::default()),
        "session" => params.session = Some(dataset.session.clone()),
        "base_store_params" => params.base_store_params = Some(Default::default()),
        "initial_bases" => params.initial_bases = Some(vec![]),
        "enable_v2_manifest_paths" => params.enable_v2_manifest_paths = false,
        "skip_auto_cleanup" => params.skip_auto_cleanup = true,
        _ => unreachable!(),
    }
    let data_dir = format!("{}/data", dir.as_str());
    let files_before = std::fs::read_dir(&data_dir).unwrap().count();
    let error = if streaming {
        let source = RecordBatchStreamAdapter::new(
            record_batch!(("id", Int32, [2])).unwrap().schema(),
            futures::stream::poll_fn(|_| panic!("invalid params consumed the source")),
        );
        tx.append_stream_with_params(Box::pin(source) as SendableRecordBatchStream, &params)
            .await
    } else {
        tx.append_with_params(vec![record_batch!(("id", Int32, [2])).unwrap()], &params)
            .await
    }
    .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains(if field.starts_with("mode_") {
        "mode"
    } else {
        field
    }));
    assert_eq!(std::fs::read_dir(&data_dir).unwrap().count(), files_before);
    assert!(Arc::ptr_eq(&previous, &tx.staged_dataset().unwrap()));
    tx.append(vec![record_batch!(("id", Int32, [2])).unwrap()])
        .await
        .unwrap();
    assert_eq!(
        tx.commit().await.unwrap().count_rows(None).await.unwrap(),
        3
    );
}

#[rstest]
#[case::batch(false)]
#[case::stream(true)]
#[tokio::test]
async fn test_append_params_use_writer_validation(#[case] streaming: bool) {
    let dataset = InsertBuilder::new("memory://")
        .execute(vec![record_batch!(("id", Int32, [0])).unwrap()])
        .await
        .unwrap();
    let mut tx = dataset.begin_transaction().await.unwrap();
    let previous = tx.staged_dataset().unwrap();
    let params = WriteParams {
        mode: WriteMode::Append,
        file_writer_options: Some(lance_file::writer::FileWriterOptions {
            max_page_bytes: Some(0),
            ..Default::default()
        }),
        ..Default::default()
    };
    let batch = record_batch!(("id", Int32, [1])).unwrap();
    let error = if streaming {
        let source =
            RecordBatchStreamAdapter::new(batch.schema(), futures::stream::iter(vec![Ok(batch)]));
        tx.append_stream_with_params(Box::pin(source) as SendableRecordBatchStream, &params)
            .await
    } else {
        tx.append_with_params(vec![batch], &params).await
    }
    .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(
        error
            .to_string()
            .contains("max_page_bytes must be greater than 0")
    );
    assert!(Arc::ptr_eq(&previous, &tx.staged_dataset().unwrap()));
    tx.append(vec![record_batch!(("id", Int32, [1])).unwrap()])
        .await
        .unwrap();
    assert_eq!(
        tx.commit().await.unwrap().count_rows(None).await.unwrap(),
        2
    );
}

#[rstest]
#[case::physical(false)]
#[case::stable(true)]
#[tokio::test]
async fn test_appends_are_private_until_one_commit(
    #[case] stable: bool,
    #[values(2, 10, 50)] append_count: i32,
) {
    let dir = TempStrDir::default();
    let dataset = InsertBuilder::new(dir.as_str())
        .with_params(&WriteParams {
            max_rows_per_file: 1,
            enable_stable_row_ids: stable,
            ..Default::default()
        })
        .execute(vec![record_batch!(("id", Int32, [0, 1])).unwrap()])
        .await
        .unwrap();
    let version = dataset.version().version;
    let mut tx = dataset.begin_transaction().await.unwrap();
    tx.append(vec![record_batch!(("id", Int32, [2, 3])).unwrap()])
        .await
        .unwrap();
    let first = tx.scan().unwrap();
    for statement in 2..=append_count {
        tx.append(vec![
            record_batch!(("id", Int32, [2 * statement, 2 * statement + 1])).unwrap(),
        ])
        .await
        .unwrap();
    }
    assert_eq!(first.try_into_batch().await.unwrap().num_rows(), 4);
    assert_eq!(
        tx.scan()
            .unwrap()
            .filter("id >= 2")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap()
            .num_rows(),
        2 * append_count as usize
    );
    assert_eq!(
        Dataset::open(dir.as_str())
            .await
            .unwrap()
            .scan()
            .try_into_batch()
            .await
            .unwrap()
            .num_rows(),
        2
    );
    let committed = tx.commit().await.unwrap();
    assert_eq!(committed.version().version, version + 1);
    assert_eq!(
        Dataset::open(dir.as_str())
            .await
            .unwrap()
            .scan()
            .try_into_batch()
            .await
            .unwrap()
            .num_rows(),
        2 + 2 * append_count as usize
    );
}

#[rstest]
#[case::physical(false)]
#[case::stable(true)]
#[tokio::test]
async fn test_read_only_commit_returns_the_original_snapshot_after_a_concurrent_append(
    #[case] stable: bool,
) {
    let dir = TempStrDir::default();
    let dataset = InsertBuilder::new(dir.as_str())
        .with_params(&WriteParams {
            enable_stable_row_ids: stable,
            ..Default::default()
        })
        .execute(vec![record_batch!(("id", Int32, [0])).unwrap()])
        .await
        .unwrap();
    let original_version = dataset.version().version;
    let mut tx = dataset.begin_transaction().await.unwrap();
    let reader = tx.scan().unwrap();
    assert_eq!(
        tx.scan().unwrap().try_into_batch().await.unwrap(),
        record_batch!(("id", Int32, [0])).unwrap()
    );
    let concurrent = InsertBuilder::new(Arc::new(dataset))
        .with_params(&append_params())
        .execute(vec![record_batch!(("id", Int32, [1])).unwrap()])
        .await
        .unwrap();
    let committed = tx.commit().await.unwrap();
    assert_eq!(committed.version().version, original_version);
    let expected = record_batch!(("id", Int32, [0])).unwrap();
    assert_eq!(committed.scan().try_into_batch().await.unwrap(), expected);
    assert_eq!(reader.try_into_batch().await.unwrap(), expected);
    let reopened = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(reopened.version().version, concurrent.version().version);
    assert_eq!(
        reopened.scan().try_into_batch().await.unwrap(),
        record_batch!(("id", Int32, [0, 1])).unwrap()
    );
}

#[rstest]
#[case::blind(false)]
#[case::read_dependent(true)]
#[tokio::test]
async fn test_concurrent_append_obeys_the_read_contract(
    #[case] read: bool,
    #[values(false, true)] stable: bool,
) {
    let dir = TempStrDir::default();
    let dataset = InsertBuilder::new(dir.as_str())
        .with_params(&WriteParams {
            enable_stable_row_ids: stable,
            ..Default::default()
        })
        .execute(vec![record_batch!(("id", Int32, [0])).unwrap()])
        .await
        .unwrap();
    let mut first = dataset.begin_transaction().await.unwrap();
    let mut second = dataset.begin_transaction().await.unwrap();
    first
        .append(vec![record_batch!(("id", Int32, [1])).unwrap()])
        .await
        .unwrap();
    second
        .append(vec![record_batch!(("id", Int32, [2])).unwrap()])
        .await
        .unwrap();
    let staged = second.staged_dataset().unwrap();
    assert_eq!(fragment_ids(&staged), [0, 1]);
    if stable {
        assert_eq!(
            staged
                .scan()
                .filter("_rowid = 1")
                .unwrap()
                .try_into_batch()
                .await
                .unwrap(),
            record_batch!(("id", Int32, [2])).unwrap()
        );
    }
    let old = if read {
        Some(second.scan().unwrap())
    } else {
        None
    };
    first.commit().await.unwrap();
    if let Some(old) = old {
        assert_eq!(old.try_into_batch().await.unwrap().num_rows(), 2);
    }
    let result = second.commit().await;
    if read {
        let error = result.unwrap_err();
        assert!(matches!(error, Error::RetryableCommitConflict { .. }));
    } else {
        let committed = result.unwrap();
        assert_eq!(fragment_ids(&committed), [0, 1, 2]);
        assert_eq!(
            committed
                .scan()
                .filter(&format!("_rowaddr >= {}", 2u64 << 32))
                .unwrap()
                .try_into_batch()
                .await
                .unwrap(),
            record_batch!(("id", Int32, [2])).unwrap()
        );
        assert_eq!(
            committed.scan().try_into_batch().await.unwrap().num_rows(),
            3
        );
    }
    assert_eq!(
        Dataset::open(dir.as_str())
            .await
            .unwrap()
            .scan()
            .try_into_batch()
            .await
            .unwrap()
            .num_rows(),
        if read { 2 } else { 3 }
    );
    if stable {
        assert_eq!(
            Dataset::open(dir.as_str())
                .await
                .unwrap()
                .scan()
                .filter("_rowid = id")
                .unwrap()
                .try_into_batch()
                .await
                .unwrap()
                .num_rows(),
            if read { 2 } else { 3 }
        );
    }
}

#[tokio::test]
async fn test_failed_write_preserves_the_snapshot_and_abort_preserves_the_scanner() {
    let dir = TempStrDir::default();
    let batch = record_batch!(("id", Int32, [0, 1])).unwrap();
    let dataset = InsertBuilder::new(dir.as_str())
        .execute(vec![batch.clone()])
        .await
        .unwrap();
    let mut tx = dataset.begin_transaction().await.unwrap();
    tx.append(vec![batch.clone()]).await.unwrap();
    let source = RecordBatchStreamAdapter::new(
        batch.schema(),
        futures::stream::iter([
            Ok(batch.clone()),
            Err(DataFusionError::Execution("injected stream failure".into())),
        ]),
    );
    let error = tx
        .append_stream(Box::pin(source) as SendableRecordBatchStream)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("injected stream failure"));
    assert_eq!(
        tx.scan()
            .unwrap()
            .try_into_batch()
            .await
            .unwrap()
            .num_rows(),
        4
    );
    tx.append(vec![batch]).await.unwrap();
    let scanner = tx.scan().unwrap();
    tx.abort();
    assert_eq!(scanner.try_into_batch().await.unwrap().num_rows(), 6);
    assert_eq!(
        Dataset::open(dir.as_str())
            .await
            .unwrap()
            .scan()
            .try_into_batch()
            .await
            .unwrap()
            .num_rows(),
        2
    );
}

#[rstest]
#[case::lost_conflict(crate::utils::test::AmbiguousFailure::LandAndConflict, true)]
#[case::lost_io(crate::utils::test::AmbiguousFailure::LandAndError, true)]
#[case::rejected(crate::utils::test::AmbiguousFailure::FailOutright, false)]
#[tokio::test]
async fn test_publication_uses_the_existing_outcome_check(
    #[case] failure: crate::utils::test::AmbiguousFailure,
    #[case] landed: bool,
) {
    let dir = TempStrDir::default();
    let handler = Arc::new(crate::utils::test::AmbiguousCommitHandler::default());
    let dataset = InsertBuilder::new(dir.as_str())
        .with_params(&WriteParams {
            commit_handler: Some(handler.clone()),
            ..Default::default()
        })
        .execute(vec![record_batch!(("id", Int32, [0])).unwrap()])
        .await
        .unwrap();
    let mut tx = dataset.begin_transaction().await.unwrap();
    tx.append(vec![record_batch!(("id", Int32, [1])).unwrap()])
        .await
        .unwrap();
    let identity = tx.id().to_owned();
    let reader = tx.scan().unwrap();
    handler.fail_next(failure);
    let result = tx.commit().await;
    assert_eq!(result.is_ok(), landed);
    if let Err(error) = result {
        assert!(
            error
                .to_string()
                .contains("simulated outright commit failure")
        );
    }
    assert_eq!(reader.try_into_batch().await.unwrap().num_rows(), 2);
    let reopened = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(
        reopened.scan().try_into_batch().await.unwrap().num_rows(),
        if landed { 2 } else { 1 }
    );
    if landed {
        assert_eq!(
            reopened
                .read_transaction_by_version(2)
                .await
                .unwrap()
                .unwrap()
                .uuid,
            identity
        );
    }
}

#[rstest]
#[case::bad_field(
    |fragment: &mut Fragment| fragment.files[0].fields = Arc::from([i32::MAX]),
    "names field"
)]
#[case::missing_file(
    |fragment: &mut Fragment| fragment.files.clear(),
    "does not contain non-nullable field"
)]
#[tokio::test]
async fn test_rejected_append_keeps_the_previous_snapshot(
    #[case] corrupt: fn(&mut Fragment),
    #[case] rejection: &str,
) {
    let dataset = InsertBuilder::new("memory://")
        .execute(vec![id_batch([0])])
        .await
        .unwrap();
    let mut tx = dataset.begin_transaction().await.unwrap();
    tx.append(vec![id_batch([1])]).await.unwrap();
    let before = tx.staged_dataset().unwrap();
    let mut written = InsertBuilder::new(before.clone())
        .with_params(&append_params())
        .execute_uncommitted(vec![id_batch([2])])
        .await
        .unwrap();
    let Operation::Append { fragments } = &mut written.operation else {
        panic!("expected append");
    };
    corrupt(&mut fragments[0]);
    let error = tx.stage_append(written).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(error.to_string().contains(rejection), "{error}");
    assert!(Arc::ptr_eq(&tx.staged_dataset().unwrap(), &before));
    assert_eq!(tx.state.steps.len(), 1);
    tx.append(vec![id_batch([3])]).await.unwrap();
    let after = tx.staged_dataset().unwrap();
    assert_eq!(fragment_ids(&after), [0, 1, 2]);
    let committed = tx.commit().await.unwrap();
    assert_eq!(ids(&committed).await, [0, 1, 3]);
    assert_eq!(ids(&before).await, [0, 1]);
}

#[rstest]
#[tokio::test]
async fn test_two_appends_allocate_distinct_local_fragment_ids_and_commit_together(
    #[values(false, true)] stable: bool,
) {
    let dataset = InsertBuilder::new("memory://")
        .with_params(&WriteParams {
            enable_stable_row_ids: stable,
            ..Default::default()
        })
        .execute(vec![record_batch!(("id", Int32, [0])).unwrap()])
        .await
        .unwrap();
    let mut tx = dataset.begin_transaction().await.unwrap();
    tx.append(vec![record_batch!(("id", Int32, [1])).unwrap()])
        .await
        .unwrap();
    tx.append(vec![record_batch!(("id", Int32, [2])).unwrap()])
        .await
        .unwrap();
    assert_eq!(tx.state.steps.len(), 2);
    assert_eq!(tx.state.next_local_fragment_id, 2);
    let staged = tx.staged_dataset().unwrap();
    assert_eq!(fragment_ids(&staged), [0, 1, 2]);
    assert_eq!(ids(&staged).await, [0, 1, 2]);
    let committed = tx.commit().await.unwrap();
    assert_eq!(fragment_ids(&committed), [0, 1, 2]);
    assert_eq!(ids(&committed).await, [0, 1, 2]);
}

#[tokio::test]
async fn test_unknown_publication_keeps_the_files_readable() {
    let dir = TempStrDir::default();
    let handler = Arc::new(crate::utils::test::AmbiguousCommitHandler::default());
    let dataset = InsertBuilder::new(dir.as_str())
        .with_params(&WriteParams {
            commit_handler: Some(handler.clone()),
            ..Default::default()
        })
        .execute(vec![record_batch!(("id", Int32, [0])).unwrap()])
        .await
        .unwrap();
    let mut tx = dataset.begin_transaction().await.unwrap();
    tx.append(vec![record_batch!(("id", Int32, [1])).unwrap()])
        .await
        .unwrap();
    let identity = tx.id().to_owned();
    let reader = tx.scan().unwrap();
    handler.fail_next(crate::utils::test::AmbiguousFailure::LandAndError);
    handler
        .fail_resolve
        .store(true, std::sync::atomic::Ordering::SeqCst);
    let error = tx.commit().await.unwrap_err();
    assert!(error.is_commit_status_unknown());
    assert!(
        error
            .to_string()
            .contains("simulated ambiguous commit failure")
    );
    handler
        .fail_resolve
        .store(false, std::sync::atomic::Ordering::SeqCst);
    assert_eq!(reader.try_into_batch().await.unwrap().num_rows(), 2);
    let reopened = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(
        reopened.scan().try_into_batch().await.unwrap().num_rows(),
        2
    );
    assert_eq!(
        reopened
            .read_transaction_by_version(2)
            .await
            .unwrap()
            .unwrap()
            .uuid,
        identity
    );
}

#[tokio::test]
async fn test_an_existing_index_reads_staged_appends_and_the_old_snapshot() {
    let dir = TempStrDir::default();
    let mut dataset = InsertBuilder::new(dir.as_str())
        .execute(vec![record_batch!(("id", Int32, [0, 1])).unwrap()])
        .await
        .unwrap();
    dataset
        .create_index(
            &["id"],
            lance_index::IndexType::BTree,
            None,
            &lance_index::scalar::ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let mut tx = dataset.begin_transaction().await.unwrap();
    tx.append(vec![record_batch!(("id", Int32, [2, 3])).unwrap()])
        .await
        .unwrap();
    let mut old = tx.scan().unwrap();
    old.filter("id >= 1").unwrap();
    tx.append(vec![record_batch!(("id", Int32, [4, 5])).unwrap()])
        .await
        .unwrap();
    assert_eq!(old.try_into_batch().await.unwrap().num_rows(), 3);
    assert_eq!(
        tx.scan()
            .unwrap()
            .filter("id >= 1")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap()
            .num_rows(),
        5
    );
    tx.commit().await.unwrap();
    assert_eq!(
        Dataset::open(dir.as_str())
            .await
            .unwrap()
            .scan()
            .filter("id >= 1")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap()
            .num_rows(),
        5
    );
}

#[tokio::test]
async fn test_a_cancelled_append_and_an_open_scan_survive_abort() {
    let dataset = InsertBuilder::new("memory://")
        .execute(vec![record_batch!(("id", Int32, [0])).unwrap()])
        .await
        .unwrap();
    let mut tx = dataset.begin_transaction().await.unwrap();
    let batch = record_batch!(("id", Int32, [1])).unwrap();
    tx.append(vec![batch.clone()]).await.unwrap();
    let previous = tx.staged_dataset().unwrap();
    let source = RecordBatchStreamAdapter::new(batch.schema(), futures::stream::pending());
    let mut pending = Box::pin(tx.append_stream(Box::pin(source) as SendableRecordBatchStream));
    assert!(futures::poll!(&mut pending).is_pending());
    drop(pending);
    assert!(Arc::ptr_eq(&previous, &tx.staged_dataset().unwrap()));
    tx.append(vec![batch]).await.unwrap();
    let stream = tx.scan().unwrap().try_into_stream().await.unwrap();
    tx.abort();
    let count = futures::TryStreamExt::try_fold(stream, 0usize, |count, batch| async move {
        Ok(count + batch.num_rows())
    })
    .await
    .unwrap();
    assert_eq!(count, 3);
}

#[tokio::test]
async fn test_an_open_reader_does_not_keep_aged_staged_files() {
    mock_instant::thread_local::MockClock::set_system_time(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap(),
    );
    let dir = TempStrDir::default();
    let dataset = InsertBuilder::new(dir.as_str())
        .execute(vec![record_batch!(("id", Int32, [0])).unwrap()])
        .await
        .unwrap();
    let mut tx = dataset.begin_transaction().await.unwrap();
    tx.append(vec![record_batch!(("id", Int32, [1])).unwrap()])
        .await
        .unwrap();
    let reader = tx.scan().unwrap();
    let staged_path = tx
        .staged_dataset()
        .unwrap()
        .manifest
        .fragments
        .last()
        .unwrap()
        .files[0]
        .path
        .clone();
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(
            std::path::Path::new(dir.as_str())
                .join("data")
                .join(staged_path.as_str()),
        )
        .unwrap();
    file.set_times(
        std::fs::FileTimes::new()
            .set_modified(std::time::SystemTime::now() - std::time::Duration::from_secs(8 * 86400)),
    )
    .unwrap();
    dataset
        .cleanup_old_versions(chrono::Duration::zero(), Some(false), Some(false))
        .await
        .unwrap();
    let error = reader.try_into_batch().await.unwrap_err();
    assert!(matches!(error.find_root(), Error::IO { .. }), "{error:?}");
    assert!(error.to_string().contains(staged_path.as_str()), "{error}");
    let committed = tx.commit().await.unwrap();
    let error = committed.scan().try_into_batch().await.unwrap_err();
    assert!(matches!(error.find_root(), Error::IO { .. }), "{error:?}");
    assert!(error.to_string().contains(staged_path.as_str()), "{error}");
}

#[derive(Debug, Clone, Copy)]
enum Statement {
    Append,
    DetachedCommit,
    Delete,
    MergeInsert,
    Compaction,
    CreateIndex,
}

#[rstest]
#[case::append(Statement::Append)]
#[case::detached_commit(Statement::DetachedCommit)]
#[case::delete(Statement::Delete)]
#[case::merge_insert(Statement::MergeInsert)]
#[case::compaction(Statement::Compaction)]
#[case::create_index(Statement::CreateIndex)]
#[tokio::test]
async fn test_a_staged_snapshot_cannot_publish_a_statement(#[case] statement: Statement) {
    let dir = TempStrDir::default();
    let (dataset, mut tx) = table_with_staged_append(dir.as_str()).await;
    let staged = snapshot_from_scan_plan(&mut tx).await;
    let mut handle = staged.as_ref().clone();

    let outcome = match statement {
        Statement::Append => InsertBuilder::new(staged)
            .with_params(&append_params())
            .execute(vec![id_batch([60])])
            .await
            .map(drop),
        Statement::DetachedCommit => {
            let on_committed = InsertBuilder::new(Arc::new(dataset.clone()))
                .with_params(&append_params())
                .execute_uncommitted(vec![id_batch([60])])
                .await
                .unwrap();
            CommitBuilder::new(staged)
                .with_detached(true)
                .execute(on_committed)
                .await
                .map(drop)
        }
        Statement::Delete => handle.delete("id = 10").await.map(drop),
        Statement::MergeInsert => MergeInsertBuilder::try_new(staged, vec!["id".into()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .try_build()
            .unwrap()
            .execute_batches(vec![id_batch([10, 60])])
            .await
            .map(drop),
        Statement::Compaction => compact_files(&mut handle, merge_small_fragments(), None)
            .await
            .map(drop),
        Statement::CreateIndex => handle
            .create_index(
                &["id"],
                lance_index::IndexType::BTree,
                None,
                &lance_index::scalar::ScalarIndexParams::default(),
                false,
            )
            .await
            .map(drop),
    };
    let error = outcome.unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(
        error.to_string().contains("uncommitted staged dataset"),
        "{error}"
    );

    let reopened = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(reopened.version_id(), dataset.version_id());
    assert_eq!(ids(&reopened).await, vec![0, 1, 2, 3]);
    assert!(reopened.load_indices().await.unwrap().is_empty());
    let committed = tx.commit().await.unwrap();
    assert_eq!(ids(&committed).await, vec![0, 1, 2, 3, 10, 11]);
}

#[derive(Debug, Clone, Copy)]
enum Prepared {
    Append,
    AppendInBatch,
    Delete,
    DetachedDelete,
    MergeInsert,
    Compaction,
    IndexSegment,
}

#[rstest]
#[case::append(Prepared::Append)]
#[case::append_in_batch(Prepared::AppendInBatch)]
#[case::delete(Prepared::Delete)]
#[case::detached_delete(Prepared::DetachedDelete)]
#[case::merge_insert(Prepared::MergeInsert)]
#[case::compaction(Prepared::Compaction)]
#[case::index_segment(Prepared::IndexSegment)]
#[tokio::test]
async fn test_work_prepared_on_a_staged_snapshot_cannot_be_committed(#[case] prepared: Prepared) {
    let dir = TempStrDir::default();
    let (dataset, mut tx) = table_with_staged_append(dir.as_str()).await;
    let staged = snapshot_from_scan_plan(&mut tx).await;
    let racer = Arc::new(
        InsertBuilder::new(Arc::new(dataset))
            .with_params(&append_params())
            .execute(vec![id_batch([50])])
            .await
            .unwrap(),
    );
    let mut committed_handle = racer.as_ref().clone();

    let outcome = match prepared {
        Prepared::Append | Prepared::AppendInBatch => {
            let on_staged = InsertBuilder::new(staged)
                .with_params(&append_params())
                .execute_uncommitted(vec![id_batch([60])])
                .await
                .unwrap();
            let builder = CommitBuilder::new(racer.clone());
            if matches!(prepared, Prepared::Append) {
                builder.execute(on_staged).await.map(drop)
            } else {
                let on_committed = InsertBuilder::new(racer.clone())
                    .with_params(&append_params())
                    .execute_uncommitted(vec![id_batch([61])])
                    .await
                    .unwrap();
                builder
                    .execute_batch(vec![on_committed, on_staged])
                    .await
                    .map(drop)
            }
        }
        Prepared::Delete | Prepared::DetachedDelete => {
            let delete = DeleteBuilder::new(staged, "id = 10")
                .execute_uncommitted()
                .await
                .unwrap();
            CommitBuilder::new(racer.clone())
                .with_detached(matches!(prepared, Prepared::DetachedDelete))
                .execute(delete.transaction)
                .await
                .map(drop)
        }
        Prepared::MergeInsert => {
            let upsert = MergeInsertBuilder::try_new(staged, vec!["id".into()])
                .unwrap()
                .when_matched(WhenMatched::UpdateAll)
                .when_not_matched(WhenNotMatched::InsertAll)
                .try_build()
                .unwrap()
                .execute_uncommitted_batches(vec![id_batch([10, 60])])
                .await
                .unwrap();
            CommitBuilder::new(racer.clone())
                .execute(upsert.transaction)
                .await
                .map(drop)
        }
        Prepared::Compaction => {
            let options = merge_small_fragments();
            let plan = plan_compaction(&staged, &options).await.unwrap();
            let mut rewrites = Vec::new();
            for task in plan.compaction_tasks() {
                rewrites.push(task.execute(&staged).await.unwrap());
            }
            assert!(!rewrites.is_empty());
            commit_compaction(
                &mut committed_handle,
                rewrites,
                Arc::new(DatasetIndexRemapperOptions::default()),
                &options,
            )
            .await
            .map(drop)
        }
        Prepared::IndexSegment => {
            let mut handle = staged.as_ref().clone();
            let segment = handle
                .create_index_builder(
                    &["id"],
                    lance_index::IndexType::BTree,
                    &lance_index::scalar::ScalarIndexParams::default(),
                )
                .execute_uncommitted()
                .await
                .unwrap();
            committed_handle
                .commit_existing_index_segments("id_idx", "id", vec![segment])
                .await
        }
    };
    let refusal = match prepared {
        Prepared::Compaction => format!("d{STAGED_VERSION}.manifest"),
        Prepared::IndexSegment => format!("future dataset version {STAGED_VERSION}"),
        Prepared::Append
        | Prepared::AppendInBatch
        | Prepared::Delete
        | Prepared::DetachedDelete
        | Prepared::MergeInsert => "prepared on an uncommitted staged dataset".to_owned(),
    };
    let error = outcome.unwrap_err();
    assert!(error.to_string().contains(&refusal), "{error}");

    let reopened = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(reopened.version_id(), racer.version_id());
    assert_eq!(ids(&reopened).await, vec![0, 1, 2, 3, 50]);
    assert!(reopened.load_indices().await.unwrap().is_empty());
}

#[tokio::test]
async fn test_a_handle_rebuilt_from_a_staged_manifest_stays_staged() {
    use prost::Message;

    let dir = TempStrDir::default();
    let (dataset, mut tx) = table_with_staged_append(dir.as_str()).await;
    let staged = snapshot_from_scan_plan(&mut tx).await;
    let serialized = lance_table::format::pb::Manifest::from(staged.manifest()).encode_to_vec();
    let rebuilt = Arc::new(
        crate::dataset::builder::DatasetBuilder::from_uri(dir.as_str())
            .with_serialized_manifest(&serialized)
            .unwrap()
            .load()
            .await
            .unwrap(),
    );
    let policy = crate::dataset::cleanup::CleanupPolicyBuilder::default()
        .before_timestamp(crate::utils::temporal::utc_now() + chrono::TimeDelta::days(1))
        .build();
    let prepared_on_rebuilt = InsertBuilder::new(rebuilt.clone())
        .with_params(&append_params())
        .execute_uncommitted(vec![id_batch([61])])
        .await
        .unwrap();
    for error in [
        rebuilt.cleanup_with_policy(policy).await.unwrap_err(),
        InsertBuilder::new(rebuilt.clone())
            .with_params(&append_params())
            .execute(vec![id_batch([60])])
            .await
            .unwrap_err(),
        CommitBuilder::new(Arc::new(dataset.clone()))
            .execute(prepared_on_rebuilt)
            .await
            .unwrap_err(),
    ] {
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error.to_string().contains("uncommitted staged dataset"),
            "{error}"
        );
    }

    let reopened = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(reopened.version_id(), dataset.version_id());
    assert_eq!(ids(&reopened).await, vec![0, 1, 2, 3]);
}

#[tokio::test]
async fn test_checkout_latest_replaces_staged_index_metadata() {
    let dir = TempStrDir::default();
    let mut dataset = InsertBuilder::new(dir.as_str())
        .execute(vec![record_batch!(("id", Int32, [0, 1])).unwrap()])
        .await
        .unwrap();
    let mut tx = dataset.begin_transaction().await.unwrap();
    tx.append(vec![record_batch!(("id", Int32, [2, 3])).unwrap()])
        .await
        .unwrap();
    let old_reader = tx.scan().unwrap();
    let mut refreshed = tx.staged_dataset().unwrap().as_ref().clone();
    let index = dataset
        .create_index(
            &["id"],
            lance_index::IndexType::BTree,
            None,
            &lance_index::scalar::ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    refreshed.checkout_latest().await.unwrap();
    let indices = refreshed.load_indices().await.unwrap();
    assert_eq!(
        indices.iter().map(|index| index.uuid).collect::<Vec<_>>(),
        vec![index.uuid]
    );
    refreshed.ensure_published().unwrap();
    assert_eq!(refreshed.count_rows(None).await.unwrap(), 2);
    assert_eq!(old_reader.try_into_batch().await.unwrap().num_rows(), 4);
    let error = tx.staged_dataset().unwrap().ensure_published().unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("staged"));
}

#[tokio::test]
async fn test_a_staged_snapshot_rejects_history_questions() {
    let dir = TempStrDir::default();
    let (dataset, mut tx) = table_with_staged_append(dir.as_str()).await;
    let base_version = dataset.version_id();
    let reader = tx.scan().unwrap();
    let staged = snapshot_from_scan_plan(&mut tx).await;
    let committed = InsertBuilder::new(Arc::new(dataset))
        .with_params(&append_params())
        .execute(vec![id_batch([50])])
        .await
        .unwrap();

    let now = crate::utils::temporal::utc_now();
    let deltas = [
        staged.delta().compared_against_version(base_version),
        staged
            .delta()
            .with_begin_version(base_version)
            .with_end_version(committed.version_id()),
        staged.delta().with_begin_date(now).with_end_date(now),
        staged.delta().with_begin_date(now),
    ];
    let mut errors = vec![
        staged.read_transaction().await.unwrap_err(),
        staged.is_stale().await.unwrap_err(),
        staged.has_successor_version().await.unwrap_err(),
    ];
    errors.extend(deltas.map(|delta| delta.build().err().unwrap()));
    for error in errors {
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error.to_string().contains("uncommitted staged dataset"),
            "{error}"
        );
    }

    let rows = reader.try_into_batch().await.unwrap();
    assert_eq!(
        rows["id"].as_primitive::<Int32Type>().values(),
        &[0, 1, 2, 3, 10, 11]
    );
}

#[tokio::test]
async fn test_cleanup_rejects_a_staged_snapshot() {
    let dir = TempStrDir::default();
    let dataset = InsertBuilder::new(dir.as_str())
        .execute(vec![record_batch!(("id", Int32, [0, 1])).unwrap()])
        .await
        .unwrap();
    let mut tx = dataset.begin_transaction().await.unwrap();
    tx.append(vec![record_batch!(("id", Int32, [2])).unwrap()])
        .await
        .unwrap();
    let staged = tx.staged_dataset().unwrap();
    let policy = crate::dataset::cleanup::CleanupPolicyBuilder::default()
        .before_timestamp(crate::utils::temporal::utc_now() + chrono::TimeDelta::days(1))
        .build();
    let operation = staged.cleanup(policy.clone());
    for error in [
        operation.explain().await.unwrap_err(),
        operation.execute().await.unwrap_err(),
        staged.cleanup_with_policy(policy).await.unwrap_err(),
        staged
            .cleanup_old_versions(chrono::TimeDelta::zero(), None, None)
            .await
            .unwrap_err(),
    ] {
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error.to_string().contains("uncommitted staged dataset"),
            "{error}"
        );
    }

    let reopened = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(reopened.version_id(), dataset.version_id());
    let committed = tx.commit().await.unwrap();
    assert_eq!(
        committed.scan().try_into_batch().await.unwrap(),
        record_batch!(("id", Int32, [0, 1, 2])).unwrap()
    );
}

#[tokio::test]
async fn test_blind_appends_build_no_snapshot_until_a_read() {
    let dir = TempStrDir::default();
    let dataset = InsertBuilder::new(dir.as_str())
        .with_params(&WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        })
        .execute(vec![record_batch!(("id", Int32, [0])).unwrap()])
        .await
        .unwrap();
    let mut tx = dataset.begin_transaction().await.unwrap();
    assert!(tx.state.staged_dataset.is_none());

    tx.append(vec![record_batch!(("id", Int32, [1])).unwrap()])
        .await
        .unwrap();
    tx.append(vec![record_batch!(("id", Int32, [2])).unwrap()])
        .await
        .unwrap();
    assert!(tx.state.staged_dataset.is_none());

    let _ = tx.scan().unwrap();
    let first_snapshot = tx.state.staged_dataset.clone().unwrap();
    let _ = tx.scan().unwrap();
    let second_snapshot = tx.state.staged_dataset.clone().unwrap();
    assert!(Arc::ptr_eq(&first_snapshot, &second_snapshot));

    tx.append(vec![record_batch!(("id", Int32, [3])).unwrap()])
        .await
        .unwrap();
    assert!(tx.state.staged_dataset.is_none());

    let committed = tx.commit().await.unwrap();
    committed.validate().await.unwrap();
    assert_eq!(ids(&committed).await, [0, 1, 2, 3]);
}

#[tokio::test]
async fn test_blind_append_preserves_base_indices_across_concurrent_append() {
    let dir = TempStrDir::default();
    let mut dataset = InsertBuilder::new(dir.as_str())
        .execute(vec![
            record_batch!(("id", Int32, [0, 1]), ("val", Int32, [10, 20])).unwrap(),
        ])
        .await
        .unwrap();
    dataset
        .create_index(
            &["val"],
            lance_index::IndexType::BTree,
            Some("val_idx".into()),
            &lance_index::scalar::ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let base_version = dataset.version_id();

    let mut tx = dataset.begin_transaction().await.unwrap();
    assert_eq!(tx.base_version(), base_version);
    assert!(tx.state.staged_dataset.is_none());
    assert!(!tx.state.requires_original_version);

    tx.append(vec![
        record_batch!(("id", Int32, [2, 3]), ("val", Int32, [30, 40])).unwrap(),
    ])
    .await
    .unwrap();
    assert_eq!(tx.base_version(), base_version);
    assert!(tx.state.staged_dataset.is_none());
    assert!(!tx.state.requires_original_version);

    let mut racer = dataset.begin_transaction().await.unwrap();
    racer
        .append(vec![
            record_batch!(("id", Int32, [4]), ("val", Int32, [50])).unwrap(),
        ])
        .await
        .unwrap();
    let after_racer = racer.commit().await.unwrap();

    let committed = tx.commit().await.unwrap();
    committed.validate().await.unwrap();
    assert_eq!(committed.version_id(), after_racer.version_id() + 1);
    assert_eq!(ids(&committed).await, [0, 1, 2, 3, 4]);
    let index_names = committed
        .load_indices()
        .await
        .unwrap()
        .iter()
        .map(|index| index.name.clone())
        .collect::<Vec<_>>();
    assert_eq!(index_names, ["val_idx"]);
}

#[tokio::test]
async fn test_merge_insert_prepared_on_a_staged_snapshot_is_refused() {
    let dir = TempStrDir::default();
    let dataset = InsertBuilder::new(dir.as_str())
        .with_params(&WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        })
        .execute(vec![
            record_batch!(("id", Int32, [1, 2]), ("b", Int32, [10, 20])).unwrap(),
        ])
        .await
        .unwrap();
    let mut tx = dataset.begin_transaction().await.unwrap();
    tx.append(vec![
        record_batch!(("id", Int32, [3]), ("b", Int32, [30])).unwrap(),
    ])
    .await
    .unwrap();
    let leaked = tx.staged_dataset().unwrap();
    let patch = record_batch!(("id", Int32, [1]), ("b", Int32, [99])).unwrap();
    let mut builder = MergeInsertBuilder::try_new(leaked, vec!["id".into()]).unwrap();
    builder
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing);
    let uncommitted = builder
        .try_build()
        .unwrap()
        .execute_uncommitted_batches(vec![patch])
        .await
        .unwrap();
    let error = CommitBuilder::new(Arc::new(dataset))
        .execute(uncommitted.transaction)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(
        error
            .to_string()
            .contains("prepared on an uncommitted staged dataset")
    );
}

#[derive(Debug, Clone, Copy)]
enum Uncommitted {
    StagedSnapshot,
    DetachedVersion,
}

#[rstest]
#[case::staged_snapshot(Uncommitted::StagedSnapshot)]
#[case::detached_version(Uncommitted::DetachedVersion)]
#[tokio::test]
async fn test_begin_transaction_requires_a_committed_mainline_snapshot(
    #[case] uncommitted: Uncommitted,
) {
    let dir = TempStrDir::default();
    let (dataset, mut tx) = table_with_staged_append(dir.as_str()).await;
    let error = match uncommitted {
        Uncommitted::StagedSnapshot => snapshot_from_scan_plan(&mut tx)
            .await
            .begin_transaction()
            .await
            .err()
            .unwrap(),
        Uncommitted::DetachedVersion => {
            let base = Arc::new(dataset);
            let written = InsertBuilder::new(base.clone())
                .with_params(&append_params())
                .execute_uncommitted(vec![id_batch([60])])
                .await
                .unwrap();
            CommitBuilder::new(base)
                .with_detached(true)
                .execute(written)
                .await
                .unwrap()
                .begin_transaction()
                .await
                .err()
                .unwrap()
        }
    };
    match uncommitted {
        Uncommitted::StagedSnapshot => {
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            assert!(
                error.to_string().contains("uncommitted staged dataset"),
                "{error}"
            );
        }
        Uncommitted::DetachedVersion => {
            assert!(matches!(error, Error::NotSupported { .. }), "{error}");
            assert!(
                error.to_string().contains("mainline dataset version"),
                "{error}"
            );
        }
    }
}

#[rstest]
#[case::deletion_file(
    |fragment: &mut Fragment| {
        fragment.deletion_file = Some(lance_table::format::DeletionFile {
            read_version: 1,
            id: 7,
            file_type: lance_table::format::DeletionFileType::Array,
            num_deleted_rows: Some(1),
            base_id: None,
            inline: None,
        })
    },
    "deletion_file"
)]
#[case::overlays(
    |fragment: &mut Fragment| {
        let mut overlay_file = fragment.files[0].clone();
        overlay_file.path = "overlay.lance".to_string();
        fragment.overlays.push(lance_table::format::overlay::DataOverlayFile {
            data_file: overlay_file,
            coverage: lance_table::format::overlay::OverlayCoverage::Shared(Arc::new(
                roaring::RoaringBitmap::from_iter([0_u32]),
            )),
            committed_version: 1,
        })
    },
    "overlays"
)]
#[case::clustering_id(
    |fragment: &mut Fragment| fragment.clustering_id = Some(1),
    "clustering_id"
)]
#[case::metadata(
    |fragment: &mut Fragment| {
        fragment.metadata.insert("shard".into(), "42".into());
    },
    "metadata"
)]
#[tokio::test]
async fn test_append_refuses_a_fragment_its_actions_cannot_express(
    #[case] carry: fn(&mut Fragment),
    #[case] carried_part: &str,
) {
    let dataset = InsertBuilder::new("memory://")
        .execute(vec![id_batch([0])])
        .await
        .unwrap();
    let mut tx = dataset.begin_transaction().await.unwrap();
    tx.append(vec![id_batch([1])]).await.unwrap();
    let before = tx.staged_dataset().unwrap();
    let mut written = InsertBuilder::new(before.clone())
        .with_params(&append_params())
        .execute_uncommitted(vec![id_batch([2])])
        .await
        .unwrap();
    let Operation::Append { fragments } = &mut written.operation else {
        panic!("expected append");
    };
    carry(&mut fragments[0]);
    let error = tx.stage_append(written).unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }), "{error}");
    assert!(error.to_string().contains(carried_part), "{error}");
    assert!(Arc::ptr_eq(&tx.staged_dataset().unwrap(), &before));
    assert_eq!(tx.state.steps.len(), 1);
    tx.append(vec![id_batch([3])]).await.unwrap();
    let committed = tx.commit().await.unwrap();
    assert_eq!(ids(&committed).await, [0, 1, 3]);
}

#[tokio::test]
async fn test_unreserved_row_ids_leave_the_previous_snapshot() {
    let dataset = InsertBuilder::new("memory://")
        .with_params(&WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        })
        .execute(vec![id_batch([0])])
        .await
        .unwrap();
    let mut tx = dataset.begin_transaction().await.unwrap();
    tx.append(vec![id_batch([1])]).await.unwrap();
    let before = tx.staged_dataset().unwrap();
    let mut written = InsertBuilder::new(before.clone())
        .with_params(&append_params())
        .execute_uncommitted(vec![id_batch([2])])
        .await
        .unwrap();
    let Operation::Append { fragments } = &mut written.operation else {
        panic!("expected append");
    };
    let assigned = dataset.manifest.next_row_id;
    let sequence = RowIdSequence::from(assigned..assigned + 1);
    fragments[0].row_id_meta = Some(RowIdMeta::Inline(write_row_ids(&sequence).into()));
    let error = tx.stage_append(written).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(error.to_string().contains("reserved"), "{error}");
    assert!(Arc::ptr_eq(&tx.staged_dataset().unwrap(), &before));
    assert!(!tx.state.requires_original_version);
    tx.append(vec![id_batch([3])]).await.unwrap();
    let committed = tx.commit().await.unwrap();
    assert_eq!(ids(&committed).await, [0, 1, 3]);
}

#[tokio::test]
async fn test_append_routes_files_to_a_committed_base() {
    let dir = lance_core::utils::tempfile::TempDir::default();
    let registered_base = dir.std_path().join("registered");
    std::fs::create_dir_all(&registered_base).unwrap();
    let primary_uri = format!("file://{}", dir.std_path().join("primary").display());
    let dataset = InsertBuilder::new(primary_uri.as_str())
        .with_params(&WriteParams {
            initial_bases: Some(vec![lance_table::format::BasePath {
                id: 1,
                name: Some("registered".to_string()),
                is_dataset_root: true,
                path: format!("file://{}", registered_base.display()),
            }]),
            ..Default::default()
        })
        .execute(vec![id_batch([0])])
        .await
        .unwrap();
    assert_eq!(dataset.manifest.fragments[0].files[0].base_id, None);

    let mut tx = dataset.begin_transaction().await.unwrap();
    tx.append_with_params(
        vec![id_batch([1, 2])],
        &WriteParams {
            target_bases: Some(vec![1]),
            ..append_params()
        },
    )
    .await
    .unwrap();
    let staged = tx.staged_dataset().unwrap();
    assert_eq!(staged.manifest.fragments[1].files[0].base_id, Some(1));
    assert_eq!(ids(&staged).await, [0, 1, 2]);

    tx.commit().await.unwrap();
    let reopened = Dataset::open(primary_uri.as_str()).await.unwrap();
    assert_eq!(reopened.manifest.fragments[1].files[0].base_id, Some(1));
    assert_eq!(ids(&reopened).await, [0, 1, 2]);
}

#[rstest]
#[case::v2_1(LanceFileVersion::V2_1)]
#[case::v2_2(LanceFileVersion::V2_2)]
#[tokio::test]
async fn test_repeated_appends_without_legacy_blob_never_rebuild_snapshot(
    #[case] file_version: LanceFileVersion,
) {
    let dataset = InsertBuilder::new("memory://")
        .with_params(&WriteParams {
            data_storage_version: Some(file_version),
            ..Default::default()
        })
        .execute(vec![id_batch([0])])
        .await
        .unwrap();
    let mut tx = dataset.begin_transaction().await.unwrap();
    assert!(!tx.state.append_context_dirty);
    tx.append(vec![id_batch([1])]).await.unwrap();
    assert!(!tx.state.append_context_dirty);
    assert!(tx.state.staged_dataset.is_none());
    assert!(Arc::ptr_eq(&tx.state.append_base, &tx.base));

    tx.append(vec![id_batch([2])]).await.unwrap();
    assert!(!tx.state.append_context_dirty);
    assert!(tx.state.staged_dataset.is_none());
    assert!(Arc::ptr_eq(&tx.state.append_base, &tx.base));

    let committed = tx.commit().await.unwrap();
    assert_eq!(ids(&committed).await, [0, 1, 2]);
}

#[tokio::test]
async fn test_legacy_blob_promotion_rebuilds_append_schema_once() {
    use arrow_array::LargeBinaryArray;
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_arrow::BLOB_META_KEY;

    let blob_meta = HashMap::from([(BLOB_META_KEY.to_string(), "true".to_string())]);
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new("blob", DataType::LargeBinary, true).with_metadata(blob_meta),
    ]));
    let batch = |id: i32, payload: &[u8]| {
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![id])) as ArrayRef,
                Arc::new(LargeBinaryArray::from(vec![Some(payload)])) as ArrayRef,
            ],
        )
        .unwrap()
    };
    let dataset = InsertBuilder::new("memory://")
        .with_params(&WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        })
        .execute(vec![batch(0, b"v21")])
        .await
        .unwrap();
    assert!(!dataset.schema().field("blob").unwrap().is_blob_v2());

    let v22_params = WriteParams {
        mode: WriteMode::Append,
        data_storage_version: Some(LanceFileVersion::V2_2),
        ..Default::default()
    };
    let mut tx = dataset.begin_transaction().await.unwrap();
    tx.append_with_params(vec![batch(1, b"first-v22")], &v22_params)
        .await
        .unwrap();
    assert!(tx.state.append_context_dirty);
    assert!(tx.state.staged_dataset.is_none());

    // The second 2.2 append rebuilds the snapshot once so the writer sees
    // the promoted blob v2 schema, and does not leave the context dirty
    // because the schema no longer has an unpromoted legacy blob field.
    tx.append_with_params(vec![batch(2, b"second-v22")], &v22_params)
        .await
        .unwrap();
    assert!(!tx.state.append_context_dirty);
    assert!(tx.state.staged_dataset.is_none());
    assert!(
        tx.state
            .append_base
            .schema()
            .field("blob")
            .unwrap()
            .is_blob_v2()
    );
    let promoted_base = tx.state.append_base.clone();

    // A third 2.2 append reuses the promoted append_base without building
    // another snapshot.
    tx.append_with_params(vec![batch(3, b"third-v22")], &v22_params)
        .await
        .unwrap();
    assert!(!tx.state.append_context_dirty);
    assert!(tx.state.staged_dataset.is_none());
    assert!(Arc::ptr_eq(&tx.state.append_base, &promoted_base));

    let committed = tx.commit().await.unwrap();
    assert!(committed.schema().field("blob").unwrap().is_blob_v2());
    assert_eq!(ids(&committed).await, [0, 1, 2, 3]);
}

#[tokio::test]
async fn test_commit_options_apply_to_the_single_publication() {
    let dir = TempStrDir::default();
    let (dataset, tx) = table_with_staged_append(dir.as_str()).await;
    let identity = tx.id().to_owned();
    let properties = HashMap::from([("job".to_string(), "nightly".to_string())]);
    let committed = tx
        .commit_with()
        .with_max_retries(3)
        .with_retry_timeout(Duration::from_secs(5))
        .with_transaction_properties(properties.clone())
        .execute()
        .await
        .unwrap();
    assert_eq!(committed.version_id(), dataset.version_id() + 1);
    let recorded = Dataset::open(dir.as_str())
        .await
        .unwrap()
        .read_transaction_by_version(committed.version_id())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(recorded.uuid, identity);
    assert_eq!(
        recorded.transaction_properties.as_deref(),
        Some(&properties)
    );
}

#[tokio::test]
async fn test_a_zero_commit_timeout_publishes_nothing() {
    let dir = TempStrDir::default();
    let (dataset, tx) = table_with_staged_append(dir.as_str()).await;
    let error = tx
        .commit_with()
        .with_timeout(Some(Duration::ZERO))
        .execute()
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(error.to_string().contains("timeout"), "{error}");
    let reopened = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(reopened.version_id(), dataset.version_id());
    assert_eq!(ids(&reopened).await, [0, 1, 2, 3]);
}
