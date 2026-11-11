// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Keeping a retention lease alive while long-running work uses it.

use tokio::sync::watch;
use tokio::task::JoinHandle;

use super::RetentionLease;
use crate::{Dataset, Result};

/// Whether a [`LeaseKeeper`] still holds its lease.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeaseStatus {
    /// The lease is listed, or a failed renewal can still be retried before
    /// it expires.
    Held,
    /// The lease expired or was removed, so the keeper stopped renewing it.
    /// Publishing a result for it still commits normally, but cleanup and
    /// compaction may have removed what the lease retained, in which case the
    /// commit fails with a conflict.
    Lost { reason: String },
}

#[derive(Debug, Clone)]
struct KeeperState {
    lease: RetentionLease,
    status: LeaseStatus,
}

/// Renews a retention lease in the background until it is dropped.
///
/// The keeper never interrupts the work the lease protects: a renewal that
/// fails for a transient reason is retried, and a lease that ends anyway is
/// reported through [`Self::status`]. Whether the work can still be
/// published is decided when it commits.
pub struct LeaseKeeper {
    dataset: Dataset,
    state: watch::Receiver<KeeperState>,
    renewals: JoinHandle<()>,
}

impl LeaseKeeper {
    /// Start renewing `lease` on `dataset`'s branch. Each renewal extends the
    /// lease by its original time to live (`expires_at - created_at`), and
    /// happens once half of the remaining time has passed.
    ///
    /// Must be called within a Tokio runtime.
    pub fn start(dataset: &Dataset, lease: RetentionLease) -> Self {
        let (_, state) = watch::channel(KeeperState {
            lease,
            status: LeaseStatus::Held,
        });
        Self {
            dataset: dataset.clone(),
            state,
            renewals: tokio::spawn(async {}),
        }
    }

    /// The lease as of its latest renewal.
    pub fn lease(&self) -> RetentionLease {
        self.state.borrow().lease.clone()
    }

    pub fn status(&self) -> LeaseStatus {
        self.state.borrow().status.clone()
    }

    /// Stop renewing and release the lease, for example when the work is
    /// abandoned. Not needed after publishing, since publication releases
    /// the lease.
    pub async fn release(self) -> Result<()> {
        Ok(())
    }

    #[cfg(test)]
    fn subscribe(&self) -> watch::Receiver<KeeperState> {
        self.state.clone()
    }
}

impl Drop for LeaseKeeper {
    fn drop(&mut self) {
        self.renewals.abort();
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use arrow_array::types::Int32Type;
    use lance_datagen::{BatchCount, RowCount, array, gen_batch};
    use mock_instant::thread_local::MockClock;

    use super::*;
    use crate::dataset::LeaseOptions;

    const MINUTE: Duration = Duration::from_secs(60);
    const START: Duration = Duration::from_secs(1_000);

    async fn leased_dataset() -> (RetentionLease, Dataset) {
        MockClock::set_system_time(START);
        let rows = gen_batch()
            .col("i", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(10), BatchCount::from(1));
        let dataset = Dataset::write(rows, "memory://", None).await.unwrap();
        let options = LeaseOptions {
            ttl: 10 * MINUTE,
            label: None,
        };
        dataset.acquire_retention_lease(options).await.unwrap()
    }

    async fn listed_leases(dataset: &Dataset) -> Vec<RetentionLease> {
        let mut latest = dataset.clone();
        latest.checkout_latest().await.unwrap();
        latest.retention_leases().to_vec()
    }

    async fn next_state(state: &mut watch::Receiver<KeeperState>) -> KeeperState {
        tokio::time::timeout(Duration::from_secs(3_600), state.changed())
            .await
            .expect("the keeper never reported a renewal")
            .unwrap();
        state.borrow_and_update().clone()
    }

    #[tokio::test(start_paused = true)]
    async fn renews_once_half_the_remaining_time_has_passed() {
        let (lease, dataset) = leased_dataset().await;
        let keeper = LeaseKeeper::start(&dataset, lease.clone());
        let mut state = keeper.subscribe();

        MockClock::set_system_time(START + 5 * MINUTE);
        let renewed = next_state(&mut state).await;

        assert_eq!(renewed.status, LeaseStatus::Held);
        let extended = lease.expires_at() + 5 * MINUTE;
        assert_eq!(renewed.lease.expires_at(), extended);
        assert_eq!(keeper.lease(), renewed.lease);
        assert_eq!(listed_leases(&dataset).await, vec![renewed.lease]);
    }

    #[tokio::test(start_paused = true)]
    async fn stops_renewing_a_lease_released_elsewhere() {
        let (lease, dataset) = leased_dataset().await;
        let keeper = LeaseKeeper::start(&dataset, lease.clone());
        let mut state = keeper.subscribe();
        dataset.release_retention_lease(&lease).await.unwrap();

        let lost = next_state(&mut state).await;

        let LeaseStatus::Lost { reason } = lost.status else {
            panic!("expected a lost lease, got {:?}", lost.status);
        };
        assert!(reason.contains("it is not listed"), "{reason}");
        assert_eq!(keeper.status(), LeaseStatus::Lost { reason });
        // A new lease would retain a different version than the work reads.
        assert_eq!(listed_leases(&dataset).await, vec![]);
    }

    #[tokio::test(start_paused = true)]
    async fn release_ends_the_lease() {
        let (lease, dataset) = leased_dataset().await;
        let keeper = LeaseKeeper::start(&dataset, lease);

        keeper.release().await.unwrap();

        assert_eq!(listed_leases(&dataset).await, vec![]);
    }
}
