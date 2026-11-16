// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::record_batch;
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::{CommitBuilder, InsertBuilder, WriteDestination, WriteMode, WriteParams};
use lance::{Dataset, Error, Result};
use lance_core::utils::tempfile::TempStrDir;
use lance_io::object_store::ObjectStore;
use lance_table::format::{IndexMetadata, Manifest, Transaction as TableTransaction};
use lance_table::io::commit::{
    CommitError, CommitHandler, ManifestLocation, ManifestNamingScheme, ManifestWriter,
};

// Pause a losing clone either before its destination probe or before
// publication, leaving the real manifest resolution and commit intact.
#[derive(Debug)]
struct GatedCloneCommitHandler {
    inner: Arc<dyn CommitHandler>,
    before_commit: bool,
    paused: std::sync::atomic::AtomicBool,
    reached: tokio::sync::Notify,
    resume: tokio::sync::Notify,
}

impl GatedCloneCommitHandler {
    async fn pause(&self) {
        if !self.paused.swap(true, std::sync::atomic::Ordering::SeqCst) {
            self.reached.notify_one();
            self.resume.notified().await;
        }
    }
}

#[async_trait::async_trait]
impl CommitHandler for GatedCloneCommitHandler {
    fn is_version_not_found_definitive(&self) -> bool {
        self.inner.is_version_not_found_definitive()
    }

    async fn resolve_latest_location(
        &self,
        base_path: &object_store::path::Path,
        object_store: &ObjectStore,
    ) -> Result<ManifestLocation> {
        if !self.before_commit {
            self.pause().await;
        }
        self.inner
            .resolve_latest_location(base_path, object_store)
            .await
    }

    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<IndexMetadata>>,
        base_path: &object_store::path::Path,
        object_store: &ObjectStore,
        manifest_writer: ManifestWriter,
        naming_scheme: ManifestNamingScheme,
        transaction: Option<TableTransaction>,
    ) -> std::result::Result<ManifestLocation, CommitError> {
        if self.before_commit {
            self.pause().await;
        }
        self.inner
            .commit(
                manifest,
                indices,
                base_path,
                object_store,
                manifest_writer,
                naming_scheme,
                transaction,
            )
            .await
    }
}

fn clone_transaction(source: &Dataset) -> Transaction {
    Transaction::new(
        source.version().version,
        Operation::Clone {
            is_shallow: true,
            ref_name: None,
            ref_version: source.version().version,
            ref_path: source.uri().to_string(),
            branch_name: None,
        },
        None,
    )
}

#[rstest::rstest]
#[case::destination_probe(false)]
#[case::publication(true)]
#[tokio::test]
async fn clone_commit_preserves_concurrent_winner(#[case] before_commit: bool) {
    let winner_dir = TempStrDir::default();
    let loser_dir = TempStrDir::default();
    let target_dir = TempStrDir::default();
    let winner_batch = record_batch!(
        ("id", Int32, [1, 2]),
        ("payload", Utf8, ["winner-a", "winner-b"])
    )
    .unwrap();
    let loser_batch = record_batch!(
        ("id", Int32, [3, 4]),
        ("payload", Utf8, ["loser-a", "loser-b"])
    )
    .unwrap();
    let winner_source = InsertBuilder::new(winner_dir.as_str())
        .execute(vec![winner_batch.clone()])
        .await
        .unwrap();
    let loser_source = InsertBuilder::new(loser_dir.as_str())
        .execute(vec![loser_batch.clone()])
        .await
        .unwrap();
    let gate = Arc::new(GatedCloneCommitHandler {
        inner: winner_source.commit_handler(),
        before_commit,
        paused: std::sync::atomic::AtomicBool::new(false),
        reached: Default::default(),
        resume: Default::default(),
    });
    let (winner, loser) = tokio::join!(
        async {
            gate.reached.notified().await;
            let result = CommitBuilder::new(target_dir.as_str())
                .execute(clone_transaction(&winner_source))
                .await;
            gate.resume.notify_one();
            result
        },
        CommitBuilder::new(target_dir.as_str())
            .with_commit_handler(gate.clone())
            .execute(clone_transaction(&loser_source)),
    );
    let winner = winner.unwrap();
    let error = loser.unwrap_err();
    assert!(
        matches!(&error, Error::DatasetAlreadyExists { .. }),
        "{error:?}"
    );
    let target_name = std::path::Path::new(target_dir.as_str())
        .file_name()
        .unwrap()
        .to_str()
        .unwrap();
    assert!(error.to_string().contains(target_name));
    let reopened = Dataset::open(target_dir.as_str()).await.unwrap();
    assert_eq!(winner.uri(), reopened.uri());
    assert_eq!(reopened.versions().await.unwrap().len(), 1);
    assert_eq!(winner.scan().try_into_batch().await.unwrap(), winner_batch);
    assert_eq!(
        reopened.scan().try_into_batch().await.unwrap(),
        winner_batch
    );
    assert_eq!(
        loser_source.scan().try_into_batch().await.unwrap(),
        loser_batch
    );
}

#[rstest::rstest]
#[tokio::test]
async fn clone_commit_rejects_existing_destination(
    #[values(false, true)] dataset_destination: bool,
) {
    let source_dir = TempStrDir::default();
    let target_dir = TempStrDir::default();
    let source_batch = record_batch!(("id", Int32, [1, 2])).unwrap();
    let target_batch = record_batch!(("id", Int32, [3, 4])).unwrap();
    InsertBuilder::new(source_dir.as_str())
        .execute(vec![source_batch.clone()])
        .await
        .unwrap();
    // A clone's read version belongs to its source, not the target.
    let source = InsertBuilder::new(source_dir.as_str())
        .with_params(&WriteParams {
            mode: WriteMode::Overwrite,
            ..Default::default()
        })
        .execute(vec![source_batch])
        .await
        .unwrap();
    assert_eq!(source.version().version, 2);
    let target = Arc::new(
        InsertBuilder::new(target_dir.as_str())
            .execute(vec![target_batch.clone()])
            .await
            .unwrap(),
    );
    let destination = if dataset_destination {
        WriteDestination::Dataset(target)
    } else {
        WriteDestination::Uri(target_dir.as_str())
    };
    let error = CommitBuilder::new(destination)
        .execute(clone_transaction(&source))
        .await
        .unwrap_err();
    assert!(
        matches!(&error, Error::DatasetAlreadyExists { .. }),
        "{error:?}"
    );
    let target_name = std::path::Path::new(target_dir.as_str())
        .file_name()
        .unwrap()
        .to_str()
        .unwrap();
    assert!(error.to_string().contains(target_name));
    let reopened = Dataset::open(target_dir.as_str()).await.unwrap();
    assert_eq!(reopened.versions().await.unwrap().len(), 1);
    assert_eq!(
        reopened.scan().try_into_batch().await.unwrap(),
        target_batch
    );
}
