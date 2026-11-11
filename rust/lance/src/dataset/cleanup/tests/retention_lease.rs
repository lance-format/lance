// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::dataset::transaction::TransactionBuilder;
use crate::dataset::{LeaseOptions, RetentionLease};

const START: Duration = Duration::from_secs(1_000_000);
const MINUTE: Duration = Duration::from_secs(60);
const MONTH: Duration = Duration::from_secs(30 * 86_400);

async fn acquire(fixture: &MockDatasetFixture, ttl: Duration) -> RetentionLease {
    let options = LeaseOptions {
        ttl,
        ..Default::default()
    };
    let dataset = fixture.open().await.unwrap();
    dataset.acquire_retention_lease(options).await.unwrap().0
}

async fn versions(fixture: &MockDatasetFixture) -> Vec<u64> {
    let dataset = fixture.open().await.unwrap();
    let mut versions: Vec<u64> = dataset
        .versions()
        .await
        .unwrap()
        .iter()
        .map(|version| version.version)
        .collect();
    versions.sort_unstable();
    versions
}

fn clean_everything_policy(delete_unverified: bool) -> CleanupPolicy {
    CleanupPolicyBuilder::default()
        .before_timestamp(utc_now())
        .delete_unverified(delete_unverified)
        .build()
}

async fn clean_everything(fixture: &MockDatasetFixture, delete_unverified: bool) -> RemovalStats {
    fixture
        .run_cleanup_with_policy(clean_everything_policy(delete_unverified))
        .await
        .unwrap()
}

#[derive(Clone, Copy, Debug)]
enum LeaseEnd {
    Release,
    Publish,
    /// The lease expires; cleanup keeps honoring it until a commit removes it.
    ExpireThenCommit,
}

#[rstest]
#[tokio::test]
async fn cleanup_retains_versions_from_the_oldest_lease_to_head(
    #[values(LeaseEnd::Release, LeaseEnd::Publish, LeaseEnd::ExpireThenCommit)] end: LeaseEnd,
    #[values(false, true)] delete_unverified: bool,
) {
    MockClock::set_system_time(START);
    let fixture = MockDatasetFixture::try_new().unwrap();
    fixture.create_some_data().await.unwrap();
    fixture.append_some_data().await.unwrap();
    let lease = acquire(&fixture, 10 * MINUTE).await;
    assert_eq!(lease.version(), 3);
    let leased_rows = fixture
        .open()
        .await
        .unwrap()
        .checkout_lease(&lease)
        .await
        .unwrap()
        .count_rows(None)
        .await
        .unwrap();
    fixture.append_some_data().await.unwrap();
    fixture.append_some_data().await.unwrap();

    // Cleanup only removes versions committed strictly before its cutoff.
    MockClock::set_system_time(match end {
        LeaseEnd::ExpireThenCommit => START + MONTH,
        _ => START + MINUTE,
    });
    let removed = clean_everything(&fixture, delete_unverified).await;
    assert_eq!(removed.old_versions, 2);
    assert_eq!(versions(&fixture).await, vec![3, 4, 5]);
    let head = fixture.open().await.unwrap();
    let leased = head.checkout_lease(&lease).await.unwrap();
    assert_eq!(leased.count_rows(None).await.unwrap(), leased_rows);
    for version in 4..=5 {
        let later = head.checkout_version(version).await.unwrap();
        assert!(later.read_transaction().await.unwrap().is_some());
    }

    match end {
        LeaseEnd::Release => head.release_retention_lease(&lease).await.unwrap(),
        LeaseEnd::Publish => {
            let publication = TransactionBuilder::for_lease(
                &lease,
                Operation::ReserveFragments { num_fragments: 1 },
            )
            .build();
            CommitBuilder::new(Arc::new(leased))
                .execute(publication)
                .await
                .unwrap();
        }
        LeaseEnd::ExpireThenCommit => fixture.append_some_data().await.unwrap(),
    }
    MockClock::set_system_time(START + 2 * MONTH);
    let removed = clean_everything(&fixture, delete_unverified).await;
    assert_eq!(removed.old_versions, 3);
    assert_eq!(versions(&fixture).await, vec![6]);
}

/// Names of the files under `data/`. Listing, unlike a HEAD, leaves the mock store's
/// recorded modification times alone.
async fn data_files(dataset: &Dataset) -> HashSet<String> {
    dataset
        .object_store
        .read_dir_all(&dataset.data_dir(), None)
        .map_ok(|meta| meta.location.filename().unwrap().to_string())
        .try_collect()
        .await
        .unwrap()
}

#[rstest]
#[tokio::test]
async fn cleanup_keeps_files_staged_after_a_lease_was_created(
    #[values(false, true)] delete_unverified: bool,
    #[values(false, true)] from_before_acquisition: bool,
) {
    MockClock::set_system_time(START);
    let fixture = MockDatasetFixture::try_new().unwrap();
    fixture.create_some_data().await.unwrap();
    let dataset = fixture.open().await.unwrap();
    let orphan = dataset.base.clone().join("data").join("orphan.lance");
    dataset.object_store.put(&orphan, b"orphan").await.unwrap();

    MockClock::set_system_time(START + 2 * Duration::from_secs(3600));
    let lease = acquire(&fixture, 2 * MONTH).await;
    // Only a version newer than the one cleanup reads from lists this lease.
    let cleaner = if from_before_acquisition {
        (*dataset).clone()
    } else {
        *fixture.open().await.unwrap()
    };
    // Inside the skew margin before the lease, as a holder's clock running ahead of the
    // store's would record it.
    MockClock::set_system_time(START + 2 * Duration::from_secs(3600) - MINUTE);
    let staged = dataset.base.clone().join("data").join("staged.lance");
    dataset.object_store.put(&staged, b"staged").await.unwrap();

    // Without the lease, a month-old unreferenced file is reclaimable either way.
    MockClock::set_system_time(START + MONTH);
    cleanup_old_versions(&cleaner, clean_everything_policy(delete_unverified))
        .await
        .unwrap();
    let files = data_files(&dataset).await;
    assert!(!files.contains("orphan.lance"));
    assert!(files.contains("staged.lance"));

    let head = fixture.open().await.unwrap();
    head.release_retention_lease(&lease).await.unwrap();
    let removed = clean_everything(&fixture, delete_unverified).await;
    assert_eq!(removed.data_files_removed, 1);
    assert!(!data_files(&dataset).await.contains("staged.lance"));
}

#[derive(Clone, Copy, Debug)]
enum MissingHistory {
    Manifest,
    TransactionFile,
}

#[rstest]
#[case::manifest(MissingHistory::Manifest)]
#[case::transaction_file(MissingHistory::TransactionFile)]
#[tokio::test]
async fn cleanup_refuses_to_delete_when_lease_history_is_missing(#[case] missing: MissingHistory) {
    MockClock::set_system_time(START);
    let fixture = MockDatasetFixture::try_new().unwrap();
    fixture.create_some_data().await.unwrap();
    fixture.append_some_data().await.unwrap();
    acquire(&fixture, 10 * MINUTE).await;
    fixture.append_some_data().await.unwrap();
    fixture.append_some_data().await.unwrap();
    let dataset = fixture.open().await.unwrap();
    let gap = match missing {
        MissingHistory::Manifest => {
            dataset
                .commit_handler
                .resolve_version_location(&dataset.base, 4, dataset.object_store.inner.as_ref())
                .await
                .unwrap()
                .path
        }
        MissingHistory::TransactionFile => {
            let version = dataset.checkout_version(4).await.unwrap();
            let transaction_file = version.manifest.transaction_file.as_deref().unwrap();
            dataset.base.child(TRANSACTIONS_DIR).child(transaction_file)
        }
    };
    dataset.object_store.delete(&gap).await.unwrap();
    let before = fixture.count_files().await.unwrap();

    MockClock::set_system_time(START + MONTH);
    let error = fixture
        .run_cleanup_with_policy(
            CleanupPolicyBuilder::default()
                .before_timestamp(utc_now())
                .delete_unverified(true)
                .build(),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Cleanup { .. }), "{error}");
    assert_contains!(error.to_string(), "first: [4]");
    assert_eq!(fixture.count_files().await.unwrap(), before);
}

#[tokio::test]
async fn branch_cleanup_uses_only_its_own_leases() {
    MockClock::set_system_time(START);
    let fixture = MockDatasetFixture::try_new().unwrap();
    fixture.create_some_data().await.unwrap();
    let mut main = fixture.load().await.unwrap();
    let branch = fixture
        .create_branch_and_load(&mut main, "branch", ("main", 1))
        .await
        .unwrap();
    let mut branch_head = branch.clone();
    for _ in 0..3 {
        branch_head.append(some_batch(), None).await.unwrap();
    }
    let main_lease = acquire(&fixture, 2 * MONTH).await;
    assert_eq!(main_lease.version(), 2);
    fixture.append_some_data().await.unwrap();

    MockClock::set_system_time(START + MONTH);
    let policy = CleanupPolicyBuilder::default()
        .before_timestamp(utc_now())
        .build();
    // The branch's history overlaps the version numbers main's lease holds.
    let removed = cleanup_old_versions(&branch_head, policy).await.unwrap();
    assert_gt!(removed.old_versions, 0);
    let mut branch_versions: Vec<u64> = branch_head
        .versions()
        .await
        .unwrap()
        .iter()
        .map(|version| version.version)
        .collect();
    branch_versions.sort_unstable();
    assert_eq!(branch_versions, vec![branch_head.manifest.version]);
}
