// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compaction tests for the commit contract area.

use super::*;

/// Distributed compaction on a tagged table: the tasks are planned at V,
/// another writer's disjoint deferred compaction lands at V+1, and the
/// driver commits the planned tasks from a fresh handle at the latest
/// version. The reuse entry is built from the snapshot at V (the
/// transaction's read version), so the two histories merge instead of
/// the other writer's transition being taken for this compaction's own.
#[tokio::test]
async fn distributed_deferred_compaction_merges_with_a_disjoint_concurrent_rewrite() {
    use crate::index::DatasetIndexExt;
    use arrow_array::cast::AsArray;
    let test_dir = TempStrDir::default();
    let uri = test_dir.as_str();
    let fresh = |uri: &str| {
        crate::dataset::builder::DatasetBuilder::from_uri(uri)
            .with_session(Arc::new(crate::session::Session::default()))
            .load()
    };
    let mut dataset = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .into_dataset(uri, FragmentCount::from(4), FragmentRowCount::from(8))
        .await
        .unwrap();
    dataset
        .create_index(
            &["i"],
            IndexType::Scalar,
            Some("i_idx".into()),
            &lance_index::scalar::ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    // Tag the table: the sticky flag decides the entry's form.
    let indices: Vec<IndexMetadata> = crate::index::loading::load_all_indices(&dataset)
        .await
        .unwrap()
        .as_ref()
        .clone();
    {
        let manifest = Arc::make_mut(&mut dataset.manifest);
        manifest.reader_feature_flags |= lance_table::feature_flags::FLAG_FRAGMENT_REUSE_INDEX;
        manifest.writer_feature_flags |= lance_table::feature_flags::FLAG_FRAGMENT_REUSE_INDEX;
    }
    crate::index::frag_reuse_reader::tests::persist_fixture(&mut dataset, indices).await;

    // Planned and executed at V over fragments 0 and 1.
    let ours = CompactionOptions {
        target_rows_per_fragment: 16,
        defer_index_remap: true,
        excluded_fragment_ids: vec![2, 3],
        ..Default::default()
    };
    let planned = fresh(uri).await.unwrap();
    let read_version = planned.manifest.version;
    let completed = execute_compaction_plan(&planned, &ours).await;
    assert_eq!(completed.len(), 1);
    assert_eq!(completed[0].read_version, read_version);

    // Another writer compacts fragments 2 and 3 at V+1.
    let mut other = fresh(uri).await.unwrap();
    let theirs = CompactionOptions {
        target_rows_per_fragment: 16,
        defer_index_remap: true,
        excluded_fragment_ids: vec![0, 1],
        ..Default::default()
    };
    compact_files(&mut other, theirs, None).await.unwrap();
    // The other writer reserved ids and then committed its rewrite.
    assert!(other.manifest.version > read_version);

    // The driver commits from a handle at the latest version.
    let mut driver = fresh(uri).await.unwrap();
    commit_compaction(
        &mut driver,
        completed,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &ours,
    )
    .await
    .unwrap();

    let committed = fresh(uri).await.unwrap();
    assert_eq!(committed.fragments().len(), 2);
    let stored = crate::index::loading::load_all_indices(&committed)
        .await
        .unwrap();
    let entry = stored
        .iter()
        .find(|idx| idx.name == lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME)
        .expect("both compactions recorded their transitions");
    let ledger = crate::index::frag_reuse::decode_frag_reuse_ledger(&committed, entry)
        .await
        .unwrap();
    assert_eq!(ledger.transitions().len(), 2);
    let batch = committed.scan().try_into_batch().await.unwrap();
    let mut values: Vec<i32> = batch["i"]
        .as_primitive::<Int32Type>()
        .iter()
        .map(|v| v.unwrap())
        .collect();
    values.sort_unstable();
    assert_eq!(values, (0..32).collect::<Vec<_>>());
    for predicate in ["i = 3", "i = 20", "i > 10"] {
        let indexed = committed
            .scan()
            .filter(predicate)
            .unwrap()
            .use_scalar_index(true)
            .try_into_batch()
            .await
            .unwrap();
        let scanned = committed
            .scan()
            .filter(predicate)
            .unwrap()
            .use_scalar_index(false)
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(indexed, scanned, "{predicate}");
    }
}
/// When the compaction commit's status is unknown (the commit errored and
/// verification was unavailable), the rewritten files must NOT be deleted:
/// if the commit landed they are referenced by the new version, and
/// deleting them would corrupt the table.
#[tokio::test]
async fn test_compaction_retry_after_ambiguous_success_preserves_live_files() {
    use crate::utils::test::{AmbiguousCommitHandler, AmbiguousFailure};

    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();
    let data = sample_data();
    let num_rows = data.num_rows();
    let handler = Arc::new(AmbiguousCommitHandler::default());

    let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
    let write_params = WriteParams {
        max_rows_per_file: 3_000,
        enable_stable_row_ids: true,
        commit_handler: Some(handler.clone()),
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
        .await
        .unwrap();
    let files_before = list_data_files(test_uri);
    let options = CompactionOptions::default();
    let completed = execute_compaction_plan(&dataset, &options).await;
    let serialized = serde_json::to_vec(&completed).unwrap();
    let retry_results: Vec<RewriteResult> = serde_json::from_slice(&serialized).unwrap();
    let rewritten_files = list_data_files(test_uri)
        .difference(&files_before)
        .cloned()
        .collect::<Vec<_>>();
    assert!(!rewritten_files.is_empty());

    handler.fail_next_rewrite(AmbiguousFailure::LandAndError);
    handler
        .fail_resolve
        .store(true, std::sync::atomic::Ordering::SeqCst);
    let err = commit_compaction(
        &mut dataset,
        completed,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &options,
    )
    .await
    .expect_err("unknown commit status must surface as an error");
    assert!(
        err.is_commit_status_unknown(),
        "expected CommitStatusUnknown, got: {:?}",
        err
    );

    handler
        .fail_resolve
        .store(false, std::sync::atomic::Ordering::SeqCst);

    // Retrying the same serialized RewriteResult now conflicts with the
    // Rewrite that already landed. The retry must not delete those files.
    let retry_error = commit_compaction(
        &mut dataset,
        retry_results,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &options,
    )
    .await
    .expect_err("the replayed rewrite must conflict with its landed predecessor");
    assert!(
        matches!(retry_error, Error::RetryableCommitConflict { .. }),
        "expected RetryableCommitConflict, got: {retry_error:?}"
    );
    let files_after_retry = list_data_files(test_uri);
    assert!(
        rewritten_files
            .iter()
            .all(|file| files_after_retry.contains(file)),
        "a replayed RewriteResult must not delete files referenced by the landed rewrite"
    );

    let ds = Dataset::open(test_uri).await.unwrap();
    assert_eq!(ds.count_rows(None).await.unwrap(), num_rows);
    let scanned = ds.scan().try_into_batch().await.unwrap();
    assert_eq!(scanned.num_rows(), num_rows);
}
/// A deferred (v0 ReplaceEntry) compaction whose commit lands but reports
/// a conflict must be recognized as our own commit. The frag reuse
/// payload is intentionally not serialized, so commit-outcome
/// verification has to compare durable forms; comparing the in-memory
/// transaction would misclassify the landed commit as foreign, retry
/// against ourselves, and delete the landed commit's transaction file.
#[tokio::test]
async fn test_deferred_compaction_recognizes_ambiguous_commit_as_own() {
    use crate::utils::test::{AmbiguousCommitHandler, AmbiguousFailure};

    let handler = Arc::new(AmbiguousCommitHandler::default());
    let mut data_gen =
        BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("i".to_owned())));
    let mut dataset = Dataset::write(
        data_gen.batch(600),
        "memory://test/deferred_ambiguous",
        Some(WriteParams {
            max_rows_per_file: 100, // 6 small files -> compaction has work
            commit_handler: Some(handler.clone()),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    // Indexed data, so the deferred compaction carries a frag reuse
    // (ReplaceEntry) payload on its Rewrite transaction.
    create_scalar_index(&mut dataset, "i", false).await;
    let fragments_before = dataset.get_fragments().len();

    let options = CompactionOptions {
        target_rows_per_fragment: 100_000,
        defer_index_remap: true,
        ..Default::default()
    };
    let completed = execute_compaction_plan(&dataset, &options).await;

    // The commit lands, but the store reports a conflict.
    handler.fail_next_rewrite(AmbiguousFailure::LandAndConflict);
    commit_compaction(
        &mut dataset,
        completed,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &options,
    )
    .await
    .expect("verification must recognize the landed compaction as our own commit");

    assert!(
        dataset.get_fragments().len() < fragments_before,
        "the landed compaction must be visible"
    );
    let fri = dataset
        .load_index_by_name(FRAG_REUSE_INDEX_NAME)
        .await
        .unwrap()
        .expect("the deferred compaction installs the frag reuse entry");
    assert_eq!(fri.index_version, 0);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 600);

    // The landed manifest references its transaction file; it must not
    // have been deleted by the (spurious) conflict cleanup.
    let transaction_file = dataset
        .manifest
        .transaction_file
        .as_deref()
        .expect("the landed manifest records its transaction file");
    let path = dataset
        .base
        .clone()
        .join(crate::dataset::TRANSACTIONS_DIR)
        .join(transaction_file);
    assert!(dataset.object_store.exists(&path).await.unwrap());
}
#[rstest]
#[case::address_style(false, true)]
#[case::stable_row_ids(true, false)]
#[tokio::test]
async fn test_compaction_retains_files_when_fragment_reservation_fails(
    #[case] enable_stable_row_ids: bool,
    #[case] defer_index_remap: bool,
) {
    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();
    let handler = Arc::new(AmbiguousCommitHandler::default());
    let mut dataset = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .into_dataset_with_params(
            test_uri,
            FragmentCount::from(2),
            FragmentRowCount::from(4),
            Some(WriteParams {
                max_rows_per_file: 4,
                data_storage_version: Some(LanceFileVersion::V2_1),
                enable_stable_row_ids,
                commit_handler: Some(handler.clone()),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    let expected = dataset.scan().try_into_batch().await.unwrap();
    let files_before = list_data_files(test_uri);
    let options = CompactionOptions {
        defer_index_remap,
        ..Default::default()
    };
    let completed = execute_compaction_plan(&dataset, &options).await;
    assert_eq!(completed.len(), 1);
    assert_eq!(completed[0].row_addrs.is_some(), defer_index_remap);
    let files_after_rewrite = list_data_files(test_uri);
    assert!(files_after_rewrite.len() > files_before.len());

    handler.fail_next_reserve(AmbiguousFailure::FailOutright);
    let err = commit_compaction(
        &mut dataset,
        completed,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &options,
    )
    .await
    .expect_err("fragment reservation that did not land must fail");
    assert!(matches!(&err, Error::IO { .. }), "{err:?}");
    assert!(
        err.to_string()
            .contains("simulated outright commit failure")
    );
    assert_eq!(list_data_files(test_uri), files_after_rewrite);

    let ds = Dataset::open(test_uri).await.unwrap();
    assert_eq!(ds.scan().try_into_batch().await.unwrap(), expected);
    // No writes remain in progress, so unverified output can be reclaimed.
    ds.cleanup_old_versions(chrono::Duration::zero(), Some(true), None)
        .await
        .unwrap();
    assert_eq!(list_data_files(test_uri), files_before);
    let ds = Dataset::open(test_uri).await.unwrap();
    assert_eq!(ds.scan().try_into_batch().await.unwrap(), expected);
}
#[rstest]
#[case::address_style(false, true)]
#[case::stable_row_ids(true, false)]
#[tokio::test]
async fn test_replayed_compaction_reservation_failure_keeps_published_files(
    #[case] enable_stable_row_ids: bool,
    #[case] defer_index_remap: bool,
) {
    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();
    let handler = Arc::new(AmbiguousCommitHandler::default());
    let mut dataset = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .into_dataset_with_params(
            test_uri,
            FragmentCount::from(2),
            FragmentRowCount::from(4),
            Some(WriteParams {
                max_rows_per_file: 4,
                data_storage_version: Some(LanceFileVersion::V2_1),
                enable_stable_row_ids,
                commit_handler: Some(handler.clone()),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    let expected = dataset.scan().try_into_batch().await.unwrap();
    let options = CompactionOptions {
        defer_index_remap,
        ..Default::default()
    };
    let completed = execute_compaction_plan(&dataset, &options).await;
    assert_eq!(completed.len(), 1);
    assert_eq!(completed[0].row_addrs.is_some(), defer_index_remap);
    let serialized = serde_json::to_vec(&completed).unwrap();
    commit_compaction(
        &mut dataset,
        completed,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &options,
    )
    .await
    .unwrap();
    let published_version = dataset.version().version;
    let published_files = list_data_files(test_uri);
    let ds = Dataset::open(test_uri).await.unwrap();
    assert_eq!(ds.get_fragments().len(), 1);
    assert_eq!(ds.scan().try_into_batch().await.unwrap(), expected);

    handler.fail_next_reserve(AmbiguousFailure::FailOutright);
    let err = commit_compaction(
        &mut dataset,
        serde_json::from_slice(&serialized).unwrap(),
        Arc::new(DatasetIndexRemapperOptions::default()),
        &options,
    )
    .await
    .expect_err("the injected reservation failure must fail the replay");
    assert!(matches!(&err, Error::IO { .. }), "{err:?}");
    assert!(
        err.to_string()
            .contains("simulated outright commit failure")
    );
    assert_eq!(list_data_files(test_uri), published_files);
    let ds = Dataset::open(test_uri).await.unwrap();
    assert_eq!(ds.version().version, published_version);
    assert_eq!(ds.scan().try_into_batch().await.unwrap(), expected);

    // GC must still treat the replayed output as referenced by the published version.
    ds.cleanup_old_versions(chrono::Duration::zero(), Some(true), None)
        .await
        .unwrap();
    let ds = Dataset::open(test_uri).await.unwrap();
    assert_eq!(ds.version().version, published_version);
    assert_eq!(ds.scan().try_into_batch().await.unwrap(), expected);
}
#[rstest::rstest]
#[tokio::test]
async fn test_compact_distributed(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
    #[values(false, true)] use_stable_row_id: bool,
) {
    // Can run the tasks independently
    // Can provide subset of tasks to commit_compaction
    // Once committed, can't commit remaining tasks
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;

    let data = sample_data();

    // Write dataset as 9 1k row fragments
    let reader = RecordBatchIterator::new(vec![Ok(data.slice(0, 9000))], data.schema());
    let write_params = WriteParams {
        max_rows_per_file: 1000,
        data_storage_version: Some(data_storage_version),
        enable_stable_row_ids: use_stable_row_id,
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
        .await
        .unwrap();

    // Plan compaction with 3 tasks
    let options = CompactionOptions {
        target_rows_per_fragment: 3_000,
        ..Default::default()
    };
    let plan = plan_compaction(&dataset, &options).await.unwrap();
    assert_eq!(plan.tasks().len(), 3);

    let dataset_ref = &dataset;
    let mut results = futures::stream::iter(plan.compaction_tasks())
        .then(|task| async move { task.execute(dataset_ref).await.unwrap() })
        .collect::<Vec<_>>()
        .await;

    assert_eq!(results.len(), 3);

    assert_eq!(
        results[0]
            .original_fragments
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(results[0].metrics.files_removed, 3);
    assert_eq!(results[0].metrics.files_added, 1);

    // Just commit the last task
    commit_compaction(
        &mut dataset,
        vec![results.pop().unwrap()],
        Arc::new(IgnoreRemap::default()),
        &options,
    )
    .await
    .unwrap();

    // 1 commit for reserve fragments and 1 for final commit, both
    // from the call to commit_compaction
    assert_eq!(dataset.manifest.version, 3);

    // Can commit the remaining tasks
    commit_compaction(
        &mut dataset,
        results,
        Arc::new(IgnoreRemap::default()),
        &options,
    )
    .await
    .unwrap();
    // 1 commit for reserve fragments and 1 for final commit, both
    // from the call to commit_compaction
    assert_eq!(dataset.manifest.version, 5);

    assert_eq!(dataset.manifest.uses_stable_row_ids(), use_stable_row_id,);
}
#[rstest]
#[case::some_tasks_recorded(
    RecordingMismatch::SomeTasksRecorded,
    "run every task with the same defer_index_remap"
)]
#[case::no_task_recorded(
    RecordingMismatch::NoTaskRecorded,
    "run the compaction with defer_index_remap"
)]
#[case::incompatible_index_created(RecordingMismatch::IncompatibleIndexCreated, "`i_idx`")]
#[case::tagged_history_required(
    RecordingMismatch::TaggedHistoryRequired,
    "requires a tagged fragment reuse history"
)]
#[tokio::test]
async fn test_commit_refuses_recording_mismatch_before_reserving_fragments(
    #[case] mismatch: RecordingMismatch,
    #[case] expected: &str,
) {
    let mut dataset = stable_row_id_dataset("memory://", 4).await;
    let deferred = CompactionOptions {
        target_rows_per_fragment: 20,
        defer_index_remap: true,
        ..Default::default()
    };
    let eager = CompactionOptions {
        defer_index_remap: false,
        ..deferred.clone()
    };
    let plan = plan_compaction(&dataset, &deferred).await.unwrap();
    assert!(plan.tasks().len() > 1);
    let mut results = Vec::new();
    for (position, task) in plan.tasks().iter().enumerate() {
        let options = match mismatch {
            RecordingMismatch::SomeTasksRecorded if position > 0 => &eager,
            RecordingMismatch::NoTaskRecorded => &eager,
            _ => &deferred,
        };
        results.push(
            rewrite_files(Cow::Borrowed(&dataset), task.clone(), options)
                .await
                .unwrap(),
        );
    }
    if matches!(mismatch, RecordingMismatch::IncompatibleIndexCreated) {
        // Inverted: the other scalar index types now always create an
        // address-domain segment on a stable-row-id dataset (safe alongside
        // the recorded moves by construction), so none of them can illustrate
        // this mismatch. Inverted still stores row ids directly here.
        dataset
            .create_index(
                &["cat"],
                IndexType::Inverted,
                Some("i_idx".into()),
                &ScalarIndexParams::for_builtin(BuiltinIndexType::Inverted),
                false,
            )
            .await
            .unwrap();
    }
    let is_tagged = matches!(mismatch, RecordingMismatch::TaggedHistoryRequired);
    if is_tagged {
        let indices = load_all_indices(&dataset).await.unwrap().as_ref().clone();
        let manifest = Arc::make_mut(&mut dataset.manifest);
        manifest.reader_feature_flags |= lance_table::feature_flags::FLAG_FRAGMENT_REUSE_INDEX;
        manifest.writer_feature_flags |= lance_table::feature_flags::FLAG_FRAGMENT_REUSE_INDEX;
        crate::index::frag_reuse_reader::tests::persist_fixture(&mut dataset, indices).await;
        let error = plan_compaction(&dataset, &deferred).await.unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        assert!(error.to_string().contains(expected), "{error}");
    }
    let version = dataset.version().version;

    let error = commit_compaction(
        &mut dataset,
        results,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &deferred,
    )
    .await
    .unwrap_err();
    if is_tagged {
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
    } else {
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    }
    assert!(error.to_string().contains(expected), "{error}");
    dataset.checkout_latest().await.unwrap();
    assert_eq!(dataset.version().version, version);
}
#[rstest]
#[case::remap(CatchUp::Remap, false)]
#[case::merge_segments(CatchUp::MergeSegments, false)]
// Zone maps cannot be remapped.
#[case::merge_zone_map_segments_with_stable_row_ids(CatchUp::MergeSegments, true)]
#[tokio::test]
async fn test_index_built_before_the_rewrite_publishes_is_retained_and_caught_up(
    #[case] catch_up: CatchUp,
    #[case] enable_stable_row_ids: bool,
    #[values(false, true)] is_built_after_an_append: bool,
) {
    let mut data_gen =
        BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("i".to_owned())));
    let mut dataset = Dataset::write(
        data_gen.batch(500),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 100,
            enable_stable_row_ids,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset.delete("i >= 10 AND i < 60").await.unwrap();
    let (index_type, params) = if enable_stable_row_ids {
        (
            IndexType::ZoneMap,
            ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap),
        )
    } else {
        (IndexType::Scalar, ScalarIndexParams::default())
    };
    // An existing index makes the compaction record an FRI.
    dataset
        .create_index(&["i"], index_type, Some("scalar".into()), &params, false)
        .await
        .unwrap();
    let mut stale = dataset.clone();
    if is_built_after_an_append {
        dataset.append(data_gen.batch(100), None).await.unwrap();
    }
    dataset
        .create_index(&["i"], index_type, Some("second".into()), &params, false)
        .await
        .unwrap();
    compact_files(
        &mut stale,
        CompactionOptions {
            target_rows_per_fragment: 100,
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    dataset.checkout_latest().await.unwrap();
    dataset.drop_index("scalar").await.unwrap();

    let frag_reuse = dataset.frag_reuse_index().await.unwrap().unwrap();
    let watermark = frag_reuse.details.versions[0].dataset_version;
    let second = stored_index(&dataset, "second").await;
    assert!(second.dataset_version < watermark);
    let coverage = second.fragment_bitmap.as_ref().unwrap();
    for new_frag in frag_reuse.details.versions[0].new_frag_ids() {
        assert!(coverage.contains(new_frag as u32));
    }
    for old_frag in frag_reuse.details.versions[0].old_frag_ids() {
        assert!(!coverage.contains(old_frag as u32));
    }

    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    assert_eq!(frag_reuse_watermarks(&dataset).await, vec![watermark]);
    assert_eq!(rows_found_through_index(&dataset).await, 11);

    catch_up_index(&mut dataset, "second", catch_up).await;
    let caught_up = dataset.load_index_by_name("second").await.unwrap().unwrap();
    assert_ne!(caught_up.uuid, second.uuid);
    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    assert!(frag_reuse_watermarks(&dataset).await.is_empty());
    assert_eq!(rows_found_through_index(&dataset).await, 11);
}
#[rstest]
#[case::at_watermark_remap(Boundary::AtWatermark, CatchUp::Remap)]
#[case::at_watermark_merge(Boundary::AtWatermark, CatchUp::MergeSegments)]
#[case::after_watermark(Boundary::AfterWatermark, CatchUp::Remap)]
#[tokio::test]
async fn test_index_at_the_watermark_boundary(
    #[case] boundary: Boundary,
    #[case] catch_up: CatchUp,
) {
    let dir = TempStrDir::default();
    let handler = Arc::new(RaceBeforeRewrite::default());
    let mut dataset = racing_dataset(dir.as_str(), handler.clone()).await;
    if matches!(boundary, Boundary::AtWatermark) {
        handler.arm(&dataset, vec![Race::BuildIndex]);
    }
    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 100,
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    let watermark = frag_reuse_watermarks(&dataset).await[0];
    assert_eq!(watermark, dataset.manifest.version - 1);

    match boundary {
        Boundary::AtWatermark => {
            let (mut handle, index) = handler.built.lock().unwrap().take().unwrap();
            let read_version = index.dataset_version;
            handle
                .apply_commit(
                    Transaction::new(
                        read_version,
                        Operation::CreateIndex {
                            new_indices: vec![index],
                            removed_indices: vec![],
                        },
                        None,
                    ),
                    &Default::default(),
                    &Default::default(),
                )
                .await
                .unwrap();
        }
        Boundary::AfterWatermark => {
            dataset
                .create_index(
                    &["i"],
                    IndexType::BTree,
                    Some("second".into()),
                    &ScalarIndexParams::default(),
                    false,
                )
                .await
                .unwrap();
        }
    }
    dataset.checkout_latest().await.unwrap();
    dataset.drop_index("scalar").await.unwrap();
    let second = stored_index(&dataset, "second").await;
    let is_at_watermark = matches!(boundary, Boundary::AtWatermark);
    assert_eq!(
        second.dataset_version,
        watermark + u64::from(!is_at_watermark)
    );

    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    assert_eq!(
        frag_reuse_watermarks(&dataset).await.len(),
        usize::from(is_at_watermark)
    );
    assert_eq!(rows_found_through_index(&dataset).await, 11);
    if is_at_watermark {
        catch_up_index(&mut dataset, "second", catch_up).await;
        let caught_up = dataset.load_index_by_name("second").await.unwrap().unwrap();
        assert_ne!(caught_up.uuid, second.uuid);
        cleanup_frag_reuse_index(&mut dataset).await.unwrap();
        assert!(frag_reuse_watermarks(&dataset).await.is_empty());
        assert_eq!(rows_found_through_index(&dataset).await, 11);
    }
}
#[rstest]
#[case::direct(Publication::Direct)]
#[case::after_repeated_retries(Publication::AfterRetries)]
#[case::over_a_concurrent_cleanup(Publication::OverAConcurrentCleanup)]
#[tokio::test]
async fn test_reuse_version_is_stamped_just_before_its_rewrite_publishes(
    #[case] publication: Publication,
) {
    let dir = TempStrDir::default();
    let handler = Arc::new(RaceBeforeRewrite::default());
    let mut dataset = racing_dataset(dir.as_str(), handler.clone()).await;
    let options = CompactionOptions {
        target_rows_per_fragment: 100,
        defer_index_remap: true,
        ..Default::default()
    };
    compact_files(&mut dataset, options.clone(), None)
        .await
        .unwrap();
    let earlier = frag_reuse_watermarks(&dataset).await;
    assert_eq!(earlier, vec![dataset.manifest.version - 1]);

    dataset.delete("i >= 210 AND i < 260").await.unwrap();
    let rows = dataset.count_rows(None).await.unwrap();
    match publication {
        Publication::Direct => {
            compact_files(&mut dataset, options, None).await.unwrap();
        }
        Publication::AfterRetries => {
            handler.arm(&dataset, vec![Race::Append, Race::Append]);
            compact_files(&mut dataset, options, None).await.unwrap();
            assert!(handler.races.lock().unwrap().is_empty());
        }
        Publication::OverAConcurrentCleanup => {
            let completed = execute_compaction_plan(&dataset, &options).await;
            let mut cleaner = dataset.clone();
            remapping::remap_column_index(&mut cleaner, &["i"], Some("scalar".into()))
                .await
                .unwrap();
            cleanup_frag_reuse_index(&mut cleaner).await.unwrap();
            assert!(frag_reuse_watermarks(&cleaner).await.is_empty());
            commit_compaction(
                &mut dataset,
                completed,
                Arc::new(DatasetIndexRemapperOptions::default()),
                &options,
            )
            .await
            .unwrap();
        }
    }

    let retained = match publication {
        Publication::OverAConcurrentCleanup => vec![],
        _ => earlier,
    };
    assert_eq!(
        frag_reuse_watermarks(&dataset).await,
        [retained, vec![dataset.manifest.version - 1]].concat()
    );
    let appended = match publication {
        Publication::AfterRetries => 4,
        _ => 0,
    };
    assert_eq!(dataset.count_rows(None).await.unwrap(), rows + appended);
} // check_rewrite_txn takes the (None, Some(_)) branch when a Rewrite with
// defer_index_remap=true is committed against a previously committed
// CreateIndex, declaring COMPATIBLE without verifying that the Rewrite's
// FRI groups don't straddle the CreateIndex's fragment bitmap. When a
// group mixes indexed and unindexed fragments, commit succeeds and later
// queries fail at load_indices with "split of indexed and non-indexed
// data".
#[rstest]
#[case::stale(false)]
#[case::fresh(true)]
#[tokio::test]
async fn test_rewrite_fri_vs_create_index_conflict(#[case] is_fresh_handle: bool) {
    use crate::index::DatasetIndexExt;
    use crate::index::vector::VectorIndexParams;
    use futures::TryStreamExt;
    use lance_datagen::{BatchCount, Dimension, RowCount, array, gen_batch};
    use lance_index::IndexType;
    use lance_linalg::distance::MetricType;

    async fn append_fragment(uri: &str, rows: u64) -> Dataset {
        let reader = gen_batch()
            .col("vec", array::rand_vec::<Float32Type>(Dimension::from(16)))
            .into_reader_rows(RowCount::from(rows), BatchCount::from(1));
        let params = WriteParams {
            max_rows_per_file: rows as usize,
            mode: WriteMode::Append,
            ..Default::default()
        };
        Dataset::write(reader, uri, Some(params)).await.unwrap()
    }

    let tmpdir = TempStrDir::default();
    let uri = format!("file://{}", tmpdir.as_str());

    // frag0 (256 rows) with a base IVF index.
    let reader = gen_batch()
        .col("vec", array::rand_vec::<Float32Type>(Dimension::from(16)))
        .into_reader_rows(RowCount::from(256), BatchCount::from(1));
    let mut dataset = Dataset::write(
        reader,
        &uri,
        Some(WriteParams {
            max_rows_per_file: 256,
            mode: WriteMode::Overwrite,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let index_params = VectorIndexParams::ivf_pq(2, 8, 2, MetricType::L2, 50);
    dataset
        .create_index(&["vec"], IndexType::Vector, None, &index_params, true)
        .await
        .unwrap();

    // Append frag1 (unindexed), snapshot a stale handle pointing here,
    // then append frag2 (also unindexed).
    dataset = append_fragment(&uri, 64).await;
    let mut stale = dataset.clone();
    dataset = append_fragment(&uri, 64).await;

    // Plan + execute compaction of frag1+frag2 with deferred remap.
    let options = CompactionOptions {
        defer_index_remap: true,
        ..Default::default()
    };
    let plan = plan_compaction(&dataset, &options).await.unwrap();
    assert!(!plan.tasks.is_empty());
    let snapshot = dataset.clone();
    let completed: Vec<RewriteResult> = futures::stream::iter(plan.tasks.into_iter())
        .map(|task| rewrite_files(Cow::Borrowed(&snapshot), task, &options))
        .buffer_unordered(1)
        .try_collect()
        .await
        .unwrap();

    // optimize_indices on the stale handle indexes frag1 only (frag2
    // didn't exist at that version), commits as CreateIndex. `dataset`
    // stays at its pre-optimize version so the Rewrite commit has to
    // conflict-check against this CreateIndex.
    stale
        .optimize_indices(&lance_index::optimize::OptimizeOptions::append())
        .await
        .unwrap();

    let before_commit = Dataset::open(&uri).await.unwrap();
    let expected_fragments = before_commit.manifest.fragments.clone();
    let expected_indices = load_all_indices(&before_commit).await.unwrap();
    if is_fresh_handle {
        dataset = before_commit;
    }

    // Commit the pre-executed Rewrite. The FRI group [frag1, frag2]
    // straddles the new CreateIndex bitmap (frag1 indexed, frag2 not), so
    // check_rewrite_txn must reject this as a retryable conflict rather
    // than letting it commit into a broken state that fails queries.
    let err = commit_compaction(
        &mut dataset,
        completed,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &options,
    )
    .await
    .expect_err("commit should fail with retryable conflict");
    assert!(
        matches!(err, Error::RetryableCommitConflict { .. }),
        "unexpected error: {err}"
    );
    let latest = Dataset::open(&uri).await.unwrap();
    assert_eq!(latest.manifest.fragments, expected_fragments);
    assert_eq!(load_all_indices(&latest).await.unwrap(), expected_indices);
}
/// Reproduce the distributed-compaction concurrent-delete data-resurrection bug.
///
/// In the distributed (Spark) path the caller opens **two separate** `Dataset` handles:
///
/// 1. dataset_plan  — used for `plan_compaction` (version = V)
/// 2. dataset_commit — opened **fresh** for `commit_compaction` (version = V+N)
///
/// Because `commit_compaction` builds the `Rewrite` transaction with
/// `dataset.manifest.version` (= V+N), `load_and_sort_new_transactions` only
/// scans versions after V+N and finds nothing.  Any concurrent DELETE that
/// happened between V and V+N is silently ignored, causing the deleted rows to
/// reappear in the compacted fragment.
///
/// After the fix, `commit_compaction` uses `min(tasks.read_version)` (= V) as
/// the transaction `read_version`, so the conflict checker correctly loads and
/// rejects the DELETE, returning a retryable conflict error instead of silently
/// resurrecting data.
#[tokio::test]
async fn test_distributed_compact_concurrent_delete_no_resurrection() {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;

    // Write 4 fragments × 1 000 rows each (a=0..4000).
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
    let data = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from_iter_values(0..4_000))],
    )
    .unwrap();
    let mut dataset_plan = Dataset::write(
        RecordBatchIterator::new(vec![Ok(data)], schema.clone()),
        test_uri,
        Some(WriteParams {
            max_rows_per_file: 1_000,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_eq!(dataset_plan.manifest.version, 1);
    assert_eq!(dataset_plan.get_fragments().len(), 4);

    // ── Step 1: plan compaction at version V=1 ───────────────────────────────
    let options = CompactionOptions {
        target_rows_per_fragment: 10_000,
        ..Default::default()
    };
    let plan = plan_compaction(&dataset_plan, &options).await.unwrap();
    assert_eq!(plan.tasks().len(), 1, "expected one compaction task");

    // ── Step 2: execute tasks (simulating distributed executors at V=1) ──────
    // Clone dataset_plan so the closure can own its copy while the original
    // remains available for the concurrent DELETE in Step 3.
    let dataset_for_tasks = dataset_plan.clone();
    let results: Vec<RewriteResult> = futures::stream::iter(plan.compaction_tasks())
        .then(|task| {
            let ds = dataset_for_tasks.clone();
            async move {
                // Executors open the dataset at the planned read_version
                task.execute(&ds).await.unwrap()
            }
        })
        .collect()
        .await;
    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].read_version, 1,
        "tasks must carry read_version=1"
    );

    // ── Step 3: concurrent DELETE commits at V=2 ─────────────────────────────
    // Delete rows where a < 1000 (the first 1 000 rows in fragment 0).
    dataset_plan.delete("a < 1000").await.unwrap();
    assert_eq!(dataset_plan.manifest.version, 2);

    // ── Step 4: the Spark driver opens a *fresh* dataset (latest = V=2) ──────
    // This is exactly what OptimizeExec.scala does for commitCompaction.
    let mut dataset_commit = Dataset::open(test_uri).await.unwrap();
    assert_eq!(
        dataset_commit.manifest.version, 2,
        "fresh dataset must be at the post-delete version"
    );

    // ── Step 5: commit_compaction with the stale results ─────────────────────
    let commit_result = commit_compaction(
        &mut dataset_commit,
        results,
        Arc::new(IgnoreRemap::default()),
        &options,
    )
    .await;

    // ── Step 6: assert correct behaviour ─────────────────────────────────────
    // BEFORE fix: commit_result is Ok(…) and the deleted rows are resurrected.
    // AFTER  fix: commit_result is Err(retryable conflict), protecting data integrity.
    assert!(
        commit_result.is_err(),
        "commit_compaction must fail with a conflict error when a concurrent \
             DELETE touched the same fragments; got Ok instead — deleted rows were \
             silently resurrected"
    );
    let err_msg = commit_result.unwrap_err().to_string();
    assert!(
        err_msg.contains("retryable")
            || err_msg.contains("conflict")
            || err_msg.contains("preempted"),
        "expected a retryable conflict error, got: {err_msg}"
    );

    // The on-disk table must still reflect the DELETE (a < 1000 remains absent).
    let latest = Dataset::open(test_uri).await.unwrap();
    let row_count = latest
        .count_rows(Some("a < 1000".to_string()))
        .await
        .unwrap();
    assert_eq!(
        row_count, 0,
        "rows deleted before compaction must not be resurrected; found {row_count}"
    );
}
/// Once `commit_compaction` reaches the Rewrite commit, its input may be a
/// replay of a result whose earlier commit landed ambiguously. A failure on
/// this call must therefore leave data files for GC instead of deleting
/// files that an existing version may reference.
#[tokio::test]
async fn test_commit_compaction_leaves_data_for_gc_on_commit_failure() {
    use crate::dataset::builder::DatasetBuilder;
    use crate::utils::test::FailingProxyStore;
    use lance_io::object_store::ObjectStoreParams;

    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();
    // Prefix `/` so Windows drive letters (e.g. `C:`) don't get parsed as
    // the URL authority.
    let path_prefix = if test_uri.starts_with('/') { "" } else { "/" };
    let routed_uri = format!("file-object-store://{path_prefix}{test_uri}");

    let data = sample_data();
    let reader = RecordBatchIterator::new(vec![Ok(data.slice(0, 200))], data.schema());
    Dataset::write(
        reader,
        &routed_uri,
        Some(WriteParams {
            max_rows_per_file: 100,
            enable_stable_row_ids: true,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let baseline_files = count_data_files_in(test_uri);

    let failing = Arc::new(FailingProxyStore::new());
    // `commit_compaction` first calls `reserve_fragment_ids` (which writes a
    // ReserveFragments transaction) and then calls `apply_commit` for the
    // rewrite itself. Skip the first transaction write so the reserve
    // succeeds, and fail the second so `apply_commit` errors out — that's
    // the branch we want to exercise cleanup for.
    failing.fail_after_n("put", "_transactions", 1, "injected commit failure");
    failing.fail_after_n(
        "put_multipart",
        "_transactions",
        1,
        "injected commit failure",
    );

    let mut dataset = DatasetBuilder::from_uri(&routed_uri)
        .with_read_params(crate::dataset::ReadParams {
            store_options: Some(ObjectStoreParams {
                object_store_wrapper: Some(failing.clone()),
                ..Default::default()
            }),
            ..Default::default()
        })
        .load()
        .await
        .unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 1000,
        ..Default::default()
    };
    let result = compact_files(&mut dataset, options, None).await;
    assert!(
        result.is_err(),
        "Compaction should fail when transaction commit fails"
    );

    assert!(
        count_data_files_in(test_uri) > baseline_files,
        "Compaction data files should be retained for GC after the Rewrite commit fails"
    );
}
#[tokio::test]
async fn test_commit_compaction_leaves_blob_v2_sidecars_for_gc_on_commit_failure() {
    use crate::BlobArrayBuilder;
    use crate::dataset::builder::DatasetBuilder;
    use crate::utils::test::FailingProxyStore;
    use lance_io::object_store::ObjectStoreParams;

    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();
    let path_prefix = if test_uri.starts_with('/') { "" } else { "/" };
    let routed_uri = format!("file-object-store://{path_prefix}{test_uri}");

    let id_array = Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef;
    // Use one packed blob and one dedicated blob to verify both sidecar layouts.
    let packed_data = vec![0u8; 100 * 1024];
    let dedicated_data = vec![1u8; 5 * 1024 * 1024];
    let mut blob_builder = BlobArrayBuilder::new(2);
    blob_builder.push_bytes(&packed_data).unwrap();
    blob_builder.push_bytes(&dedicated_data).unwrap();
    let blob_array: ArrayRef = blob_builder.finish().unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        crate::blob_field("blob", true),
    ]));
    let batch = RecordBatch::try_new(schema.clone(), vec![id_array, blob_array]).unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

    Dataset::write(
        reader,
        &routed_uri,
        Some(WriteParams {
            max_rows_per_file: 1, // Create 2 fragments
            enable_stable_row_ids: true,
            data_storage_version: Some(lance_file::version::LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let baseline_files = count_data_files_in(test_uri);

    let failing = Arc::new(FailingProxyStore::new());
    failing.fail_after_n("put", "_transactions", 1, "injected commit failure");
    failing.fail_after_n(
        "put_multipart",
        "_transactions",
        1,
        "injected commit failure",
    );

    let mut dataset = DatasetBuilder::from_uri(&routed_uri)
        .with_read_params(crate::dataset::ReadParams {
            store_options: Some(ObjectStoreParams {
                object_store_wrapper: Some(failing.clone()),
                ..Default::default()
            }),
            ..Default::default()
        })
        .load()
        .await
        .unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 1000,
        ..Default::default()
    };
    let result = compact_files(&mut dataset, options, None).await;
    assert!(
        result.is_err(),
        "Compaction should fail when transaction commit fails"
    );

    assert!(
        count_data_files_in(test_uri) > baseline_files,
        "Blob v2 sidecars should be retained for GC after the Rewrite commit fails"
    );
}
