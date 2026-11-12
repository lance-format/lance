// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Keeping a retention lease alive while long-running work uses it.

use std::time::Duration;

use chrono::{DateTime, Utc};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::warn;

use super::RetentionLease;
use crate::utils::temporal::utc_now;
use crate::{Dataset, Error, Result};

/// The shortest wait before retrying a failed renewal, so that failures close
/// to expiry do not turn into a busy loop.
const MIN_RETRY_DELAY: Duration = Duration::from_secs(1);

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
///
/// ```
/// # use lance::{Dataset, Result};
/// # use lance::dataset::{LeaseKeeper, LeaseOptions, LeaseStatus};
/// # async fn example(dataset: &Dataset) -> Result<()> {
/// let (lease, leased) = dataset
///     .acquire_retention_lease(LeaseOptions::default())
///     .await?;
/// let keeper = LeaseKeeper::start(&leased, lease);
/// // ... hours of work reading `leased` ...
/// if let LeaseStatus::Lost { reason } = keeper.status() {
///     eprintln!("publishing may conflict: {reason}");
/// }
/// // Publish with `TransactionBuilder::for_lease(&keeper.lease(), operation)`,
/// // or, to abandon the work:
/// keeper.release().await?;
/// # Ok(())
/// # }
/// ```
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
        let (sender, state) = watch::channel(KeeperState {
            lease: lease.clone(),
            status: LeaseStatus::Held,
        });
        Self {
            dataset: dataset.clone(),
            state,
            renewals: tokio::spawn(renew_until_lost(dataset.clone(), lease, sender)),
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
        self.renewals.abort();
        self.dataset.release_retention_lease(&self.lease()).await
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

async fn renew_until_lost(
    dataset: Dataset,
    mut lease: RetentionLease,
    state: watch::Sender<KeeperState>,
) {
    let ttl = (lease.expires_at() - lease.created_at())
        .to_std()
        .expect("a lease expires after its creation");
    let mut delay = half_remaining(&lease, utc_now());
    loop {
        tokio::time::sleep(delay).await;
        match dataset.renew_retention_lease(&lease, ttl).await {
            Ok(renewed) => {
                lease = renewed;
                state.send_replace(KeeperState {
                    lease: lease.clone(),
                    status: LeaseStatus::Held,
                });
                delay = half_remaining(&lease, utc_now());
            }
            Err(error) => {
                let now = utc_now();
                if let Some(reason) = lost_reason(&error, &lease, now) {
                    warn!(
                        "Stopped renewing retention lease {} on version {}: {reason}",
                        lease.id(),
                        lease.version()
                    );
                    state.send_modify(|state| state.status = LeaseStatus::Lost { reason });
                    return;
                }
                warn!(
                    "Failed to renew retention lease {}, retrying before it expires at {}: {error}",
                    lease.id(),
                    lease.expires_at()
                );
                delay = half_remaining(&lease, now).max(MIN_RETRY_DELAY);
            }
        }
    }
}

fn half_remaining(lease: &RetentionLease, now: DateTime<Utc>) -> Duration {
    (lease.expires_at() - now).to_std().unwrap_or_default() / 2
}

/// Why a failed renewal means the lease is gone, or `None` if it can be
/// retried. Renewal rejects an absent or expired lease as invalid input; any
/// other failure is transient until the lease expires.
fn lost_reason(error: &Error, lease: &RetentionLease, now: DateTime<Utc>) -> Option<String> {
    if matches!(error, Error::InvalidInput { .. }) {
        Some(error.to_string())
    } else if lease.expires_at() <= now {
        Some(format!(
            "it expired at {} before a renewal succeeded; last error: {error}",
            lease.expires_at()
        ))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use arrow_array::types::Int32Type;
    use lance_datagen::{BatchCount, RowCount, array, gen_batch};
    use mock_instant::thread_local::MockClock;
    use rstest::rstest;

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

    #[rstest]
    #[case::rejected_renewal(Error::invalid_input("it is not listed"), 1, true)]
    #[case::transient_failure_before_expiry(Error::io("timeout"), 1, false)]
    #[case::transient_failure_at_expiry(Error::io("timeout"), 10, true)]
    fn renewal_failures_lose_the_lease_only_when_it_cannot_be_renewed(
        #[case] error: Error,
        #[case] minutes_elapsed: u32,
        #[case] lost: bool,
    ) {
        let lease: RetentionLease = serde_json::from_str(
            r#"{
                "id": "8c4f3f53-4a3e-4c55-9f5e-1d6a0a2b7c11",
                "version": 2,
                "created_at": "2026-01-01T00:00:00Z",
                "expires_at": "2026-01-01T00:10:00Z"
            }"#,
        )
        .unwrap();
        let now = lease.created_at() + minutes_elapsed * MINUTE;

        assert_eq!(lost_reason(&error, &lease, now).is_some(), lost);
    }
}
