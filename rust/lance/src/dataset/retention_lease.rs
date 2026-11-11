// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Acquiring, renewing, and releasing retention leases.
//!
//! See the [Retention Leases specification](https://lance.org/format/table/retention_leases/).

mod keeper;

use std::time::Duration;

use chrono::{DateTime, Utc};
pub use keeper::{LeaseKeeper, LeaseStatus};
pub use lance_table::format::RetentionLease;
use uuid::Uuid;

use super::Dataset;
use super::transaction::{Operation, TransactionBuilder};
use crate::utils::temporal::utc_now;
use crate::{Error, Result};

/// Options for [`Dataset::acquire_retention_lease`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeaseOptions {
    /// How long the lease lasts unless renewed, from the time of acquisition.
    /// Must be positive. Defaults to 30 minutes.
    pub ttl: Duration,
    /// Human-readable context stored with the lease; `None` stores no label.
    pub label: Option<String>,
}

impl Default for LeaseOptions {
    fn default() -> Self {
        Self {
            ttl: Duration::from_secs(30 * 60),
            label: None,
        }
    }
}

fn expiry_after(now: DateTime<Utc>, ttl: Duration) -> Result<DateTime<Utc>> {
    let expires_at = chrono::Duration::from_std(ttl)
        .ok()
        .filter(|ttl| *ttl > chrono::Duration::zero())
        .and_then(|ttl| now.checked_add_signed(ttl));
    expires_at.ok_or_else(|| {
        Error::invalid_input(format!(
            "retention lease ttl {ttl:?} must be positive and end before the maximum timestamp"
        ))
    })
}

impl Dataset {
    /// Acquire a retention lease on a new version of this dataset.
    ///
    /// While the lease is listed, cleanup keeps the leased version and every
    /// later one, so the holder can build on the leased version for as long
    /// as it keeps the lease. The lease ends when the holder publishes a
    /// transaction for it (see [`TransactionBuilder::for_lease`]), releases it
    /// with [`Self::release_retention_lease`], or lets it expire without
    /// [`Self::renew_retention_lease`].
    ///
    /// The acquisition commits a new version on the latest version of this
    /// dataset's branch; that version is the leased one. Returns the lease
    /// and the dataset checked out at [`RetentionLease::version`].
    ///
    /// Returns `NotSupported` when this build of Lance cannot honor leases on
    /// this table, for example on a table with stable row IDs.
    ///
    /// ```
    /// # use std::time::Duration;
    /// # use lance::{Dataset, Result};
    /// # use lance::dataset::LeaseOptions;
    /// # async fn example(dataset: &Dataset) -> Result<()> {
    /// let options = LeaseOptions {
    ///     ttl: Duration::from_secs(60 * 60),
    ///     label: Some("index build".to_string()),
    /// };
    /// let (lease, leased) = dataset.acquire_retention_lease(options).await?;
    /// assert_eq!(leased.version().version, lease.version());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn acquire_retention_lease(
        &self,
        options: LeaseOptions,
    ) -> Result<(RetentionLease, Self)> {
        // Fixed once so that every commit attempt requests the same lease.
        let created_at = utc_now();
        let expires_at = expiry_after(created_at, options.ttl)?;
        let lease_id = Uuid::new_v4();
        let head = self
            .commit_lease_operation(Operation::AcquireLease {
                lease_id,
                created_at,
                expires_at,
                label: options.label,
            })
            .await?;
        let lease = listed_lease(&head, lease_id)?;
        let leased = if head.manifest.version == lease.version() {
            head
        } else {
            head.checkout_lease(&lease).await?
        };
        Ok((lease, leased))
    }

    /// Check out the version `lease` retains, for example on a worker that
    /// received the lease from its holder.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use lance::dataset::RetentionLease;
    /// # async fn example(dataset: &Dataset, lease: &RetentionLease) -> Result<()> {
    /// let leased = dataset.checkout_lease(lease).await?;
    /// assert_eq!(leased.version().version, lease.version());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn checkout_lease(&self, lease: &RetentionLease) -> Result<Self> {
        self.checkout_version(lease.version()).await
    }

    /// Extend `lease` to expire `ttl` from now, and return the listed record.
    ///
    /// A time not later than the current expiry changes nothing and writes no
    /// version; the listed record is returned unchanged. Fails if the lease
    /// is no longer listed or has already expired.
    ///
    /// ```
    /// # use std::time::Duration;
    /// # use lance::{Dataset, Result};
    /// # use lance::dataset::RetentionLease;
    /// # async fn example(dataset: &Dataset, lease: &RetentionLease) -> Result<()> {
    /// let renewed = dataset
    ///     .renew_retention_lease(lease, Duration::from_secs(30 * 60))
    ///     .await?;
    /// assert!(renewed.expires_at() >= lease.expires_at());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn renew_retention_lease(
        &self,
        lease: &RetentionLease,
        ttl: Duration,
    ) -> Result<RetentionLease> {
        let expires_at = expiry_after(utc_now(), ttl)?;
        let head = self
            .commit_lease_operation(Operation::RenewLease {
                lease_id: lease.id(),
                expires_at,
            })
            .await?;
        listed_lease(&head, lease.id())
    }

    /// Release `lease`, ending its retention without publishing a result.
    /// Releasing a lease that is not listed changes nothing and writes no
    /// version.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use lance::dataset::RetentionLease;
    /// # async fn example(dataset: &Dataset, lease: &RetentionLease) -> Result<()> {
    /// dataset.release_retention_lease(lease).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn release_retention_lease(&self, lease: &RetentionLease) -> Result<()> {
        self.commit_lease_operation(Operation::ReleaseLease {
            lease_id: lease.id(),
        })
        .await?;
        Ok(())
    }

    /// The leases listed in this version's manifest, including expired ones
    /// that no commit has removed yet. Only the latest version's list is
    /// authoritative.
    ///
    /// ```
    /// # use lance::Dataset;
    /// # fn example(dataset: &Dataset) -> Option<u64> {
    /// let oldest_leased_version = dataset
    ///     .retention_leases()
    ///     .iter()
    ///     .map(|lease| lease.version())
    ///     .min();
    /// # oldest_leased_version
    /// # }
    /// ```
    pub fn retention_leases(&self) -> &[RetentionLease] {
        self.manifest.retention_leases()
    }

    /// Commit a lease operation on the latest version of this branch and
    /// return that branch's head afterwards. Lease operations take `&self`
    /// because they move no handle: a holder keeps reading its leased version.
    async fn commit_lease_operation(&self, operation: Operation) -> Result<Self> {
        let mut head = self.clone();
        // Starting from the latest version spares the commit loop from
        // rebasing over every version since this handle's.
        head.checkout_latest().await?;
        let transaction = TransactionBuilder::new(head.manifest.version, operation).build();
        head.apply_commit(transaction, &Default::default(), &Default::default())
            .await?;
        Ok(head)
    }
}

fn listed_lease(dataset: &Dataset, lease_id: Uuid) -> Result<RetentionLease> {
    dataset
        .retention_leases()
        .iter()
        .find(|lease| lease.id() == lease_id)
        .cloned()
        .ok_or_else(|| {
            Error::internal(format!(
                "retention lease {lease_id} is not listed in version {} after its commit",
                dataset.manifest.version
            ))
        })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::types::Int32Type;
    use lance_datagen::{BatchCount, RowCount, array, gen_batch};
    use lance_table::feature_flags::FLAG_RETENTION_LEASES;
    use mock_instant::thread_local::MockClock;

    use super::*;
    use crate::dataset::write::{CommitBuilder, WriteMode, WriteParams};

    const MINUTE: Duration = Duration::from_secs(60);

    async fn create_dataset(uri: &str) -> Dataset {
        Dataset::write(rows(), uri, None).await.unwrap()
    }

    fn rows() -> impl arrow_array::RecordBatchReader + Send + 'static {
        gen_batch()
            .col("i", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(10), BatchCount::from(1))
    }

    async fn latest(dataset: &Dataset) -> Dataset {
        let mut latest = dataset.clone();
        latest.checkout_latest().await.unwrap();
        latest
    }

    async fn transaction_files(dataset: &Dataset) -> usize {
        dataset
            .object_store
            .read_dir(dataset.base.clone().join(crate::dataset::TRANSACTIONS_DIR))
            .await
            .unwrap()
            .len()
    }

    fn has_lease_flag(dataset: &Dataset) -> bool {
        dataset.manifest.writer_feature_flags & FLAG_RETENTION_LEASES != 0
    }

    #[tokio::test]
    async fn test_publication_releases_lease_across_appends() {
        MockClock::set_system_time(Duration::from_secs(1_000));
        let mut dataset = create_dataset("memory://").await;
        let options = LeaseOptions {
            ttl: 10 * MINUTE,
            label: Some("index build".to_string()),
        };
        let (lease, leased) = dataset.acquire_retention_lease(options).await.unwrap();
        assert_eq!(lease.version(), 2);
        assert_eq!(lease.label(), Some("index build"));
        assert_eq!(
            lease.expires_at() - lease.created_at(),
            chrono::Duration::minutes(10)
        );
        assert_eq!(leased.version().version, 2);
        assert_eq!(leased.retention_leases(), std::slice::from_ref(&lease));
        assert!(has_lease_flag(&leased));

        dataset.append(rows(), None).await.unwrap();
        assert_eq!(dataset.version().version, 3);
        assert_eq!(dataset.retention_leases(), std::slice::from_ref(&lease));

        let worker = dataset.checkout_lease(&lease).await.unwrap();
        assert_eq!(worker.version().version, lease.version());
        let publication =
            TransactionBuilder::for_lease(&lease, Operation::ReserveFragments { num_fragments: 1 })
                .build();
        let published = CommitBuilder::new(Arc::new(worker))
            .execute(publication)
            .await
            .unwrap();
        assert_eq!(published.version().version, 4);
        assert!(published.retention_leases().is_empty());
        assert!(!has_lease_flag(&published));
    }

    #[tokio::test]
    async fn test_noop_lease_requests_write_no_version() {
        MockClock::set_system_time(Duration::from_secs(1_000));
        let dataset = create_dataset("memory://").await;
        let options = LeaseOptions {
            ttl: 10 * MINUTE,
            ..Default::default()
        };
        let (lease, _) = dataset.acquire_retention_lease(options).await.unwrap();
        let transactions_before = transaction_files(&dataset).await;

        let unchanged = dataset.renew_retention_lease(&lease, MINUTE).await.unwrap();
        assert_eq!(unchanged, lease);
        assert_eq!(latest(&dataset).await.version().version, 2);
        // The attempt wrote a transaction file before learning it had nothing
        // to commit; no version references it, so it must not be left behind.
        assert_eq!(transaction_files(&dataset).await, transactions_before);

        let renewed = dataset
            .renew_retention_lease(&lease, 20 * MINUTE)
            .await
            .unwrap();
        assert_eq!(renewed.id(), lease.id());
        assert_eq!(renewed.version(), lease.version());
        assert_eq!(
            renewed.expires_at(),
            lease.created_at() + chrono::Duration::minutes(20)
        );
        let head = latest(&dataset).await;
        assert_eq!(head.version().version, 3);
        assert_eq!(head.retention_leases(), std::slice::from_ref(&renewed));

        dataset.release_retention_lease(&lease).await.unwrap();
        let head = latest(&dataset).await;
        assert_eq!(head.version().version, 4);
        assert!(head.retention_leases().is_empty());
        assert!(!has_lease_flag(&head));

        dataset.release_retention_lease(&lease).await.unwrap();
        assert_eq!(latest(&dataset).await.version().version, 4);
        let error = dataset
            .renew_retention_lease(&lease, 20 * MINUTE)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("it is not listed"), "{error}");
    }

    #[tokio::test]
    async fn test_next_commit_removes_expired_lease() {
        MockClock::set_system_time(Duration::from_secs(1_000));
        let mut dataset = create_dataset("memory://").await;
        let options = LeaseOptions {
            ttl: MINUTE,
            ..Default::default()
        };
        let (lease, _) = dataset.acquire_retention_lease(options).await.unwrap();

        MockClock::set_system_time(Duration::from_secs(1_000) + MINUTE);
        // An expired lease stays listed until a commit removes it.
        assert_eq!(latest(&dataset).await.retention_leases().len(), 1);
        let error = dataset
            .renew_retention_lease(&lease, MINUTE)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("it expired at"), "{error}");

        dataset.append(rows(), None).await.unwrap();
        assert!(dataset.retention_leases().is_empty());
        assert!(!has_lease_flag(&dataset));
    }

    #[tokio::test]
    async fn test_restore_and_overwrite_remove_leases() {
        MockClock::set_system_time(Duration::from_secs(1_000));
        let dataset = create_dataset("memory://").await;
        let (_, mut leased) = dataset
            .acquire_retention_lease(LeaseOptions::default())
            .await
            .unwrap();
        dataset
            .acquire_retention_lease(LeaseOptions::default())
            .await
            .unwrap();

        // Restoring a version that listed a lease must not bring it back.
        leased.restore().await.unwrap();
        assert_eq!(leased.version().version, 4);
        assert!(leased.retention_leases().is_empty());
        assert!(!has_lease_flag(&leased));

        let (_, leased) = leased
            .acquire_retention_lease(LeaseOptions::default())
            .await
            .unwrap();
        let params = WriteParams {
            mode: WriteMode::Overwrite,
            ..Default::default()
        };
        let overwritten = Dataset::write(rows(), Arc::new(leased), Some(params))
            .await
            .unwrap();
        assert!(overwritten.retention_leases().is_empty());
        assert!(!has_lease_flag(&overwritten));
    }

    #[tokio::test]
    async fn test_branches_and_detached_commits_carry_no_leases() {
        MockClock::set_system_time(Duration::from_secs(1_000));
        let (_, mut dataset) = create_dataset("memory://")
            .await
            .acquire_retention_lease(LeaseOptions::default())
            .await
            .unwrap();

        let branch = dataset
            .create_branch("experiment", ("main", None), None)
            .await
            .unwrap();
        assert!(branch.retention_leases().is_empty());
        assert!(!has_lease_flag(&branch));

        let reserve = TransactionBuilder::new(
            dataset.version().version,
            Operation::ReserveFragments { num_fragments: 1 },
        )
        .build();
        let detached = CommitBuilder::new(Arc::new(dataset.clone()))
            .with_detached(true)
            .execute(reserve)
            .await
            .unwrap();
        assert!(detached.retention_leases().is_empty());
        assert!(!has_lease_flag(&detached));

        let release = TransactionBuilder::new(
            dataset.version().version,
            Operation::ReleaseLease {
                lease_id: dataset.retention_leases()[0].id(),
            },
        )
        .build();
        let error = CommitBuilder::new(Arc::new(dataset.clone()))
            .with_detached(true)
            .execute(release)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error.to_string().contains("cannot be committed detached"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn test_acquire_rebased_over_append_leases_the_version_it_creates() {
        MockClock::set_system_time(Duration::from_secs(1_000));
        let stale = create_dataset("memory://").await;
        let mut appender = stale.clone();
        appender.append(rows(), None).await.unwrap();

        let lease_id = Uuid::new_v4();
        let acquire = Operation::AcquireLease {
            lease_id,
            created_at: utc_now(),
            expires_at: utc_now() + chrono::Duration::minutes(10),
            label: None,
        };
        let head = commit_from(&stale, acquire).await.unwrap();

        assert_eq!(head.version().version, 3);
        assert_eq!(head.count_rows(None).await.unwrap(), 20);
        let [lease] = head.retention_leases() else {
            panic!("expected one lease, got {:?}", head.retention_leases());
        };
        assert_eq!(lease.id(), lease_id);
        assert_eq!(lease.version(), 3);
    }

    /// Commit `operation` read at `read`'s version, so it rebases over every
    /// version committed since.
    async fn commit_from(read: &Dataset, operation: Operation) -> Result<Dataset> {
        let transaction = TransactionBuilder::new(read.version().version, operation).build();
        CommitBuilder::new(Arc::new(read.clone()))
            .execute(transaction)
            .await
    }

    #[tokio::test]
    async fn test_renewal_committed_first_outlives_concurrent_expiry_check() {
        MockClock::set_system_time(Duration::from_secs(1_000));
        let dataset = create_dataset("memory://").await;
        let options = LeaseOptions {
            ttl: MINUTE,
            ..Default::default()
        };
        let (lease, mut stale) = dataset.acquire_retention_lease(options).await.unwrap();

        MockClock::set_system_time(Duration::from_secs(1_030));
        let renewed = dataset
            .renew_retention_lease(&lease, 10 * MINUTE)
            .await
            .unwrap();

        // Past the expiry `stale` lists, but before the renewed one: the
        // retry derives from the renewal, not from its read version.
        MockClock::set_system_time(Duration::from_secs(1_000) + 2 * MINUTE);
        stale.append(rows(), None).await.unwrap();
        assert_eq!(stale.version().version, 4);
        assert_eq!(stale.retention_leases(), std::slice::from_ref(&renewed));
        assert!(has_lease_flag(&stale));
    }

    #[tokio::test]
    async fn test_acquire_retried_after_same_id_acquired_returns_stored_lease() {
        MockClock::set_system_time(Duration::from_secs(1_000));
        let dataset = create_dataset("memory://").await;
        let lease_id = Uuid::new_v4();
        let acquire = |label: &str| Operation::AcquireLease {
            lease_id,
            created_at: utc_now(),
            expires_at: utc_now() + chrono::Duration::minutes(10),
            label: Some(label.to_string()),
        };
        let first = commit_from(&dataset, acquire("first")).await.unwrap();
        let stored = first.retention_leases().to_vec();

        MockClock::set_system_time(Duration::from_secs(1_030));
        let retried = commit_from(&dataset, acquire("retried")).await.unwrap();
        assert_eq!(retried.version().version, 2);
        assert_eq!(retried.retention_leases(), stored.as_slice());
        assert_eq!(stored[0].label(), Some("first"));
    }

    enum LeaseEnd {
        Released,
        ExpiredAndRemoved,
        ExpiredButListed,
    }

    #[rstest::rstest]
    #[case::released(LeaseEnd::Released, "it is not listed")]
    #[case::expired_and_removed(LeaseEnd::ExpiredAndRemoved, "it is not listed")]
    #[case::expired_but_listed(LeaseEnd::ExpiredButListed, "it expired at")]
    #[tokio::test]
    async fn test_renew_retried_after_lease_ended_errors(
        #[case] end: LeaseEnd,
        #[case] message: &str,
    ) {
        MockClock::set_system_time(Duration::from_secs(1_000));
        let dataset = create_dataset("memory://").await;
        let options = LeaseOptions {
            ttl: MINUTE,
            ..Default::default()
        };
        let (lease, stale) = dataset.acquire_retention_lease(options).await.unwrap();
        let mut concurrent = stale.clone();
        match end {
            LeaseEnd::Released => dataset.release_retention_lease(&lease).await.unwrap(),
            LeaseEnd::ExpiredAndRemoved => {
                MockClock::set_system_time(Duration::from_secs(1_000) + MINUTE);
                concurrent.append(rows(), None).await.unwrap();
            }
            LeaseEnd::ExpiredButListed => {
                concurrent.append(rows(), None).await.unwrap();
                MockClock::set_system_time(Duration::from_secs(1_000) + MINUTE);
            }
        }

        let renew = Operation::RenewLease {
            lease_id: lease.id(),
            expires_at: utc_now() + chrono::Duration::minutes(10),
        };
        let error = commit_from(&stale, renew).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains(message), "{error}");
        assert_eq!(latest(&dataset).await.version().version, 3);
    }

    #[tokio::test]
    async fn test_publication_retried_after_release_proceeds() {
        MockClock::set_system_time(Duration::from_secs(1_000));
        let dataset = create_dataset("memory://").await;
        let (lease, worker) = dataset
            .acquire_retention_lease(LeaseOptions::default())
            .await
            .unwrap();
        dataset.release_retention_lease(&lease).await.unwrap();

        let publication =
            TransactionBuilder::for_lease(&lease, Operation::ReserveFragments { num_fragments: 1 })
                .build();
        let published = CommitBuilder::new(Arc::new(worker))
            .execute(publication)
            .await
            .unwrap();
        assert_eq!(published.version().version, 4);
        assert!(published.retention_leases().is_empty());
        assert!(!has_lease_flag(&published));
    }

    #[tokio::test]
    async fn test_release_retried_after_publication_writes_no_version() {
        MockClock::set_system_time(Duration::from_secs(1_000));
        let dataset = create_dataset("memory://").await;
        let (lease, worker) = dataset
            .acquire_retention_lease(LeaseOptions::default())
            .await
            .unwrap();
        let publication =
            TransactionBuilder::for_lease(&lease, Operation::ReserveFragments { num_fragments: 1 })
                .build();
        CommitBuilder::new(Arc::new(worker.clone()))
            .execute(publication)
            .await
            .unwrap();

        let release = Operation::ReleaseLease {
            lease_id: lease.id(),
        };
        let head = commit_from(&worker, release).await.unwrap();
        assert_eq!(head.version().version, 3);
        assert!(head.retention_leases().is_empty());
    }

    #[rstest::rstest]
    #[case::zero(Duration::ZERO)]
    #[case::overflow(Duration::MAX)]
    fn test_invalid_ttl_rejected(#[case] ttl: Duration) {
        let error = expiry_after(utc_now(), ttl).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("must be positive"), "{error}");
    }
}
