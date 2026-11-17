// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::dataset::{InsertBuilder, NewColumnTransform};
use lance_table::format::{DataFile, DeletionFile, DeletionFileType};
use lance_table::io::commit::CommitHandler;
use lance_table::io::commit::external_manifest::{
    ExternalManifestCommitHandler, ExternalManifestStore,
};

fn id_batch(start: i32) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )])),
        vec![Arc::new(Int32Array::from_iter_values(start..start + 100))],
    )
    .unwrap()
}

/// Commits a detached version of `dataset` that deletes the rows matching
/// `predicate`, returning its version.
pub(super) async fn commit_detached_delete(dataset: &Arc<Dataset>, predicate: &str) -> u64 {
    let delete = DeleteBuilder::new(dataset.clone(), predicate)
        .execute_uncommitted()
        .await
        .unwrap();
    CommitBuilder::new(dataset.clone())
        .with_detached(true)
        .execute(delete.transaction)
        .await
        .unwrap()
        .version_id()
}

async fn scanned_rows(dataset: &Dataset, version: u64) -> usize {
    let checked_out = dataset.checkout_version(version).await.unwrap();
    checked_out
        .scan()
        .try_into_batch()
        .await
        .unwrap()
        .num_rows()
}

#[rstest]
#[tokio::test]
async fn cleanup_keeps_files_of_detached_versions(
    #[values(false, true)] delete_unverified: bool,
    #[values(false, true)] by_version: bool,
) {
    MockClock::set_system_time(std::time::Duration::ZERO);
    let fixture = MockDatasetFixture::try_new().unwrap();
    let batch = id_batch(0);
    fixture
        .create_with_data(RecordBatchIterator::new(
            [Ok(batch.clone())],
            batch.schema(),
        ))
        .await
        .unwrap();
    let dataset = Arc::new(*fixture.open().await.unwrap());

    // One detached version deletes rows of the first version's data file, so
    // it also needs a deletion file; the other appends a data file of its own.
    let deleted = commit_detached_delete(&dataset, "id < 10").await;
    let append = InsertBuilder::new(dataset.clone())
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute_uncommitted(vec![id_batch(100)])
        .await
        .unwrap();
    let appended = CommitBuilder::new(dataset.clone())
        .with_detached(true)
        .execute(append)
        .await
        .unwrap()
        .version_id();
    assert!(is_detached_version(deleted) && is_detached_version(appended));

    // The first version expires, and every file is past the age threshold.
    fixture.overwrite_some_data().await.unwrap();
    MockClock::set_system_time(TimeDelta::days(10).to_std().unwrap());
    // A version cutoff above every detached version lets an unverified cleanup
    // expire manifests from the listing alone.
    let policy = CleanupPolicy {
        before_timestamp: (!by_version).then(utc_now),
        before_version: by_version.then_some(u64::MAX),
        delete_unverified,
        ..Default::default()
    };
    let removed = fixture.run_cleanup_with_policy(policy).await.unwrap();
    assert_eq!(removed.old_versions, 1);

    assert_eq!(scanned_rows(&dataset, deleted).await, 90);
    assert_eq!(scanned_rows(&dataset, appended).await, 200);
    assert_eq!(dataset.list_detached_manifests().await.unwrap().len(), 2);
}

/// A detached version only adds files to keep: it does not move the cutoff of
/// the `_transactions/` and `_deletions/` listings, even when it is older than
/// every retained version.
#[rstest]
#[tokio::test]
async fn cleanup_with_a_detached_version_only_keeps_more(
    #[values(false, true)] delete_unverified: bool,
) {
    let day = |days: f64| std::time::Duration::from_secs_f64(days * 86_400.0);
    MockClock::set_system_time(day(0.0));
    let fixture = MockDatasetFixture::try_new().unwrap();
    fixture.create_some_data().await.unwrap();
    let dataset = Arc::new(*fixture.open().await.unwrap());
    let detached = commit_detached_delete(&dataset, "true").await;
    MockClock::set_system_time(day(1.0));
    fixture.overwrite_some_data().await.unwrap();
    MockClock::set_system_time(day(2.0));
    fixture.overwrite_some_data().await.unwrap();
    // Orphans newer than the oldest retained version.
    MockClock::set_system_time(day(2.5));
    for path in [
        dataset.deletions_dir().join("0-9-99.arrow"),
        dataset.transactions_dir().join("9-orphan.txn"),
    ] {
        dataset.object_store.put(&path, b"orphan").await.unwrap();
    }

    MockClock::set_system_time(day(10.0));
    let policy = CleanupPolicy {
        before_timestamp: Some(DateTime::UNIX_EPOCH + day(1.5)),
        delete_unverified,
        ..Default::default()
    };
    let candidates = || async {
        let dataset = fixture.open().await.unwrap();
        dataset
            .cleanup(policy.clone())
            .with_max_candidate_files(usize::MAX)
            .explain()
            .await
            .unwrap()
            .candidate_files
            .into_iter()
            .map(|file| file.path)
            .collect::<BTreeSet<_>>()
    };
    let with_detached = candidates().await;
    dataset
        .object_store
        .delete(&ManifestNamingScheme::V2.manifest_path(&dataset.base, detached))
        .await
        .unwrap();
    let without_detached = candidates().await;
    assert!(
        with_detached.is_subset(&without_detached) && with_detached != without_detached,
        "{with_detached:?} is not a strict subset of {without_detached:?}"
    );
}

/// A data file an earlier release removed: cleanup still runs when only a
/// detached version holds it, and fails when the latest version does, including
/// a fragment the latest version shares with a detached one.
#[rstest]
#[case::detached(true, false)]
#[case::latest(false, false)]
#[case::shared(true, true)]
#[tokio::test]
async fn managed_cleanup_with_missing_files(#[case] detached: bool, #[case] shared: bool) {
    let fixture = MockDatasetFixture::try_new().unwrap();
    let params = || WriteParams {
        mode: WriteMode::Append,
        store_params: Some(fixture.os_params()),
        data_storage_version: Some(lance_file::version::LanceFileVersion::V2_2),
        ..Default::default()
    };
    let batch = managed_blob_batch();
    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
            &fixture.dataset_path,
            Some(WriteParams {
                mode: WriteMode::Create,
                ..params()
            }),
        )
        .await
        .unwrap(),
    );
    let append = InsertBuilder::new(dataset.clone())
        .with_params(&params())
        .execute_uncommitted(vec![managed_blob_batch()])
        .await
        .unwrap();
    let appended = CommitBuilder::new(dataset.clone())
        .with_detached(detached)
        .execute(append)
        .await
        .unwrap();
    let fragments = &appended.manifest.fragments;
    let fragment = if shared {
        fragments.first()
    } else {
        fragments.last()
    }
    .unwrap();
    dataset
        .object_store
        .delete(&dataset.data_dir().join(fragment.files[0].path.as_str()))
        .await
        .unwrap();

    let result = fixture.run_cleanup(utc_now()).await;
    if detached && !shared {
        result.unwrap();
        let values = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
        assert_eq!(
            values[0].as_ref().unwrap().read().await.unwrap().as_ref(),
            b"first"
        );
        assert_eq!(dataset.list_detached_manifests().await.unwrap().len(), 1);
    } else {
        assert!(result.unwrap_err().is_not_found());
    }
}

/// A detached version's fragment whose blob columns sit in two data files, after
/// an earlier release removed some of its files: cleanup runs once both data files
/// are gone, and fails naming the detached version while one of them can still be
/// read, including when only the deletion file is gone.
#[rstest]
#[case::added_column(&[1], false)]
#[case::first_column(&[0], false)]
#[case::both(&[0, 1], false)]
#[case::deletion_file(&[], true)]
#[tokio::test]
async fn managed_cleanup_with_split_blob_files(
    #[case] removed: &[usize],
    #[case] removes_deletion_file: bool,
) {
    MockClock::set_system_time(std::time::Duration::ZERO);
    let fixture = MockDatasetFixture::try_new().unwrap();
    let batch = managed_blob_batch();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
        &fixture.dataset_path,
        Some(WriteParams {
            store_params: Some(fixture.os_params()),
            data_storage_version: Some(lance_file::version::LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let mut blobs = BlobArrayBuilder::new(2);
    blobs.push_bytes(b"third").unwrap();
    blobs.push_bytes(b"fourth").unwrap();
    let later = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            batch.schema().field(1).clone().with_name("later"),
        ])),
        vec![blobs.finish().unwrap()],
    )
    .unwrap();
    dataset
        .add_columns(
            NewColumnTransform::Reader(Box::new(RecordBatchIterator::new(
                [Ok(later.clone())],
                later.schema(),
            ))),
            None,
            None,
        )
        .await
        .unwrap();
    let files = dataset.manifest.fragments[0].files.clone();
    assert_eq!(files.len(), 2);
    let detached = commit_detached_delete(&Arc::new(dataset.clone()), "id = 1").await;
    let detached_dataset = Arc::new(dataset.checkout_version(detached).await.unwrap());
    let fragment = &detached_dataset.manifest.fragments[0];
    let deletion_file = deletion_file_path(
        &dataset.base,
        fragment.id,
        fragment.deletion_file.as_ref().unwrap(),
    );
    let deletion = dataset
        .object_store
        .read_one_all(&deletion_file)
        .await
        .unwrap();
    // From here on only the detached version references the fragment.
    dataset.delete("id >= 0").await.unwrap();
    assert!(dataset.manifest.fragments.is_empty());
    for index in removed {
        dataset
            .object_store
            .delete(&dataset.data_dir().join(files[*index].path.as_str()))
            .await
            .unwrap();
    }
    if removes_deletion_file {
        dataset.object_store.delete(&deletion_file).await.unwrap();
    }

    // Every attached version but the latest expires.
    MockClock::set_system_time(TimeDelta::days(10).to_std().unwrap());
    let result = fixture.run_cleanup(utc_now()).await;
    if removed.len() == files.len() {
        result.unwrap();
    } else {
        let error = result.unwrap_err();
        assert!(error.is_not_found(), "{error}");
        assert!(matches!(error, Error::IO { .. }), "{error}");
        assert_contains!(error.to_string(), format!("detached version {detached}"));
    }
    if removes_deletion_file {
        // The blobs are kept, so restoring the deletion file restores the version.
        dataset
            .object_store
            .put(&deletion_file, &deletion)
            .await
            .unwrap();
        let values = detached_dataset
            .take_blobs_by_indices(&[0], "blob")
            .await
            .unwrap();
        assert_eq!(
            values[0].as_ref().unwrap().read().await.unwrap().as_ref(),
            b"first"
        );
    }
    assert_eq!(dataset.list_detached_manifests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn blob_descriptors_are_removed_only_when_every_file_is_gone() {
    let fixture = MockDatasetFixture::try_new().unwrap();
    fixture.create_some_data().await.unwrap();
    let dataset = fixture.open().await.unwrap();
    let mut fragment = Fragment::new(9);
    for (path, field) in [("first.lance", 1), ("second.lance", 2), ("other.lance", 3)] {
        fragment
            .files
            .push(DataFile::new_legacy_from_fields(path, vec![field], None));
        dataset
            .object_store
            .put(&dataset.data_dir().join(path), b"data")
            .await
            .unwrap();
    }
    async fn removed(dataset: &Dataset, fragment: &Fragment) -> bool {
        BlobDescriptorFiles::new(dataset, fragment, &[1, 2])
            .removed(&dataset.object_store)
            .await
            .unwrap()
    }
    let data_path = |path: &str| dataset.data_dir().join(path);

    // One blob column's file is gone and the other's is not.
    dataset
        .object_store
        .delete(&data_path("first.lance"))
        .await
        .unwrap();
    assert!(!removed(&dataset, &fragment).await);
    // A file in another base counts as present.
    let mut external = fragment.clone();
    external.files[1].base_id = Some(1);
    dataset
        .object_store
        .delete(&data_path("second.lance"))
        .await
        .unwrap();
    assert!(!removed(&dataset, &external).await);
    assert!(removed(&dataset, &fragment).await);

    // A missing deletion file does not count while a data file remains.
    let mut deleted = Fragment::new(10);
    deleted.files.push(DataFile::new_legacy_from_fields(
        "other.lance",
        vec![1],
        None,
    ));
    deleted.deletion_file = Some(DeletionFile {
        read_version: 1,
        id: 1,
        file_type: DeletionFileType::Array,
        num_deleted_rows: Some(1),
        base_id: None,
        inline: None,
    });
    assert!(!removed(&dataset, &deleted).await);
    dataset
        .object_store
        .delete(&data_path("other.lance"))
        .await
        .unwrap();
    assert!(removed(&dataset, &deleted).await);
}

#[tokio::test]
async fn cleanup_skips_staging_detached_manifests() {
    let fixture = MockDatasetFixture::try_new().unwrap();
    fixture.create_some_data().await.unwrap();
    let dataset = fixture.open().await.unwrap();
    let committed = ManifestNamingScheme::V2.manifest_path(
        &dataset.base,
        lance_table::format::DETACHED_VERSION_MASK | 7,
    );
    let partial = Path::from(format!("{committed}-0b6e3c2a-1d8f-4f0e-9a51-3c7d2e8b4f10"));
    dataset
        .object_store
        .put(&partial, b"partial")
        .await
        .unwrap();

    fixture.run_cleanup(utc_now()).await.unwrap();
    assert!(dataset.object_store.exists(&partial).await.unwrap());
}

/// A detached version whose external store record names its staging copy, as
/// when the copy to the canonical path failed, keeps its files on a store that
/// cannot list its records, also when it is finalized after the listing, before
/// or after cleanup looks up its record.
#[rstest]
#[case::recorded(None)]
#[case::finalized_before_the_lookup(Some(false))]
#[case::finalized_before_the_read(Some(true))]
#[tokio::test]
async fn cleanup_keeps_files_of_a_detached_version_recorded_at_its_staging_copy(
    #[case] finalized_after_lookup: Option<bool>,
) {
    let store = Arc::new(IdentifiedStore {
        is_unlisted: true,
        answers_before_finalizing: finalized_after_lookup == Some(true),
        ..Default::default()
    });
    let handler: Arc<dyn CommitHandler> = Arc::new(ExternalManifestCommitHandler {
        external_manifest_store: store.clone(),
    });
    let params = |mode| WriteParams {
        mode,
        commit_handler: Some(handler.clone()),
        ..Default::default()
    };
    let uri = TempStrDir::default();
    let dataset = Arc::new(
        InsertBuilder::new(uri.as_str())
            .with_params(&params(WriteMode::Create))
            .execute(vec![id_batch(0)])
            .await
            .unwrap(),
    );
    let append = InsertBuilder::new(dataset.clone())
        .with_params(&params(WriteMode::Append))
        .execute_uncommitted(vec![id_batch(100)])
        .await
        .unwrap();
    let detached = CommitBuilder::new(dataset.clone())
        .with_detached(true)
        .execute(append)
        .await
        .unwrap()
        .version_id();

    let canonical = ManifestNamingScheme::V2.manifest_path(&dataset.base, detached);
    let staging = Path::parse(format!("{canonical}-{}", Uuid::new_v4())).unwrap();
    dataset
        .object_store
        .copy(&canonical, &staging)
        .await
        .unwrap();
    dataset.object_store.delete(&canonical).await.unwrap();
    let size = dataset.object_store.size(&staging).await.unwrap();
    store
        .put_if_exists("", detached, staging.as_ref(), size, None)
        .await
        .unwrap();

    // The first version expires, so only the detached version references its
    // data file.
    let latest = InsertBuilder::new(dataset.clone())
        .with_params(&params(WriteMode::Overwrite))
        .execute(vec![id_batch(200)])
        .await
        .unwrap();
    if finalized_after_lookup.is_some() {
        *store.finalize_on_lookup.lock().unwrap() =
            Some((detached, dataset.object_store.clone(), canonical.clone()));
    }
    cleanup_old_versions(
        &latest,
        CleanupPolicyBuilder::default()
            .before_timestamp(chrono::Utc::now())
            .delete_unverified(true)
            .build(),
    )
    .await
    .unwrap();

    let kept = if finalized_after_lookup.is_some() {
        assert!(store.finalize_on_lookup.lock().unwrap().is_none());
        &canonical
    } else {
        &staging
    };
    assert!(dataset.object_store.exists(kept).await.unwrap());
    assert_eq!(scanned_rows(&latest, detached).await, 200);
}

#[tokio::test]
async fn cleanup_keeps_parent_files_of_a_branch_detached_version() {
    MockClock::set_system_time(std::time::Duration::ZERO);
    let fixture = MockDatasetFixture::try_new().unwrap();
    let batch = id_batch(0);
    fixture
        .create_with_data(RecordBatchIterator::new(
            [Ok(batch.clone())],
            batch.schema(),
        ))
        .await
        .unwrap();
    let mut dataset = *fixture.open().await.unwrap();
    let version = dataset.version_id();
    let branch = Arc::new(
        fixture
            .create_branch_and_load(&mut dataset, "branch", version)
            .await
            .unwrap(),
    );
    // The detached version reads the parent's data file through the branch.
    let detached = commit_detached_delete(&branch, "id < 10").await;

    // Once the branch's own history stops referencing the parent's data file,
    // only the detached version does.
    let branch_head = InsertBuilder::new(branch.clone())
        .with_params(&WriteParams {
            mode: WriteMode::Overwrite,
            ..Default::default()
        })
        .execute(vec![id_batch(100)])
        .await
        .unwrap();
    fixture.overwrite_some_data().await.unwrap();
    MockClock::set_system_time(TimeDelta::days(10).to_std().unwrap());
    cleanup_old_versions(
        &branch_head,
        CleanupPolicyBuilder::default()
            .before_timestamp(utc_now())
            .build(),
    )
    .await
    .unwrap();
    fixture.run_cleanup(utc_now()).await.unwrap();

    assert_eq!(scanned_rows(&branch, detached).await, 90);
}

/// A detached manifest that is listed but gone when read, on the dataset or on a
/// referenced branch, is skipped.
#[rstest]
#[tokio::test]
async fn cleanup_skips_removed_detached_manifests(#[values(false, true)] on_branch: bool) {
    MockClock::set_system_time(std::time::Duration::ZERO);
    let fixture = MockDatasetFixture::try_new().unwrap();
    fixture.create_some_data().await.unwrap();
    let mut dataset = *fixture.open().await.unwrap();
    let version = dataset.version_id();
    let branch = fixture
        .create_branch_and_load(&mut dataset, "branch", version)
        .await
        .unwrap();
    fixture.overwrite_some_data().await.unwrap();
    let base = if on_branch {
        &branch.base
    } else {
        &dataset.base
    };
    let placeholder = base.clone().join("_versions").join("removed");
    dataset
        .object_store
        .put(&placeholder, b"removed")
        .await
        .unwrap();
    // The listing reports the placeholder as a detached manifest that no longer
    // exists.
    let removed = ManifestNamingScheme::V2
        .manifest_path(base, lance_table::format::DETACHED_VERSION_MASK | 7);
    fixture
        .mock_store
        .policy
        .lock()
        .unwrap()
        .set_obj_meta_policy(
            "removed_detached_manifest",
            Arc::new(move |_, mut meta: ObjectMeta| {
                if meta.location == placeholder {
                    meta.location = removed.clone();
                }
                Ok(meta)
            }),
        );
    let listed = dataset
        .commit_handler
        .list_detached_manifest_locations(base, &dataset.object_store)
        .map_ok(|location| location.version)
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    assert_eq!(listed, vec![lance_table::format::DETACHED_VERSION_MASK | 7]);

    // The branch's source version expires, so cleanup also walks the branch.
    MockClock::set_system_time(TimeDelta::days(10).to_std().unwrap());
    fixture.run_cleanup(utc_now()).await.unwrap();
}
