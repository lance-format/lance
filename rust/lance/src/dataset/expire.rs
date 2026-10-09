// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Remove version history without touching data.
//!
//! [`expire_versions`] deletes manifests and nothing else. It never computes the set of
//! live data files, which is what makes it cheap: on a table taking thousands of commits
//! an hour, [`cleanup_old_versions`](super::cleanup::cleanup_old_versions) must read
//! every retained manifest before it can delete anything, and those manifests can total
//! terabytes.
//!
//! The data files that only an expired version referenced are left behind as garbage.
//! That is deliberate — the next `cleanup_old_versions` reclaims them, and it now has far
//! fewer manifests to read. The intended order is expire, then clean up.
//!
//! ```no_run
//! # use lance::dataset::Dataset;
//! # use lance::dataset::expire::{expire_versions, ExpireVersionsPolicyBuilder};
//! # use std::time::Duration;
//! # async fn example(dataset: &Dataset) -> lance::Result<()> {
//! // Keep everything from the last 7 days, then one version per hour before that.
//! let policy = ExpireVersionsPolicyBuilder::default()
//!     .before_timestamp(chrono::Utc::now() - chrono::TimeDelta::try_days(7).unwrap())
//!     .keep_one_per(Duration::from_secs(3600))
//!     .build();
//! let stats = expire_versions(dataset, policy).await?;
//! println!("removed {} versions", stats.versions_removed);
//! # Ok(())
//! # }
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::time::Duration;

use chrono::{DateTime, Utc};
use futures::{StreamExt, TryStreamExt, stream};
use lance_table::io::commit::ManifestLocation;
use log::{debug, warn};
use tokio::time::{MissedTickBehavior, interval};
use tokio_stream::wrappers::IntervalStream;

use super::Dataset;
use super::cleanup::calculate_duration;
use super::version_retention::{ProtectedVersions, thin_to_one_per};
use crate::{Error, Result};

/// Smallest bucket width [`ExpireVersionsPolicy::keep_one_per`] accepts.
///
/// Thinning competes with the version hint: the floor that protects the top of the history
/// is read once, and a slow committer can publish a lower hint afterwards. Keeping
/// survivors at least an hour apart keeps deletions away from the seconds-wide window in
/// which that can happen. See [`ExpireVersionsPolicy::keep_one_per`].
pub const MIN_KEEP_ONE_PER: Duration = Duration::from_secs(3600);

/// Which versions to expire, and how many to keep behind.
#[derive(Clone, Debug)]
pub struct ExpireVersionsPolicy {
    /// Expire versions written before this time.
    ///
    /// Age comes from the manifest object's write time
    /// ([`ManifestLocation::last_modified`]), not from `Manifest::timestamp`. The two
    /// agree closely in normal operation, but rewriting an object by copy or migration
    /// moves the write time and leaves the commit time alone. Use
    /// [`before_version`](Self::before_version) when an exact boundary matters.
    ///
    /// A manifest whose write time the store did not report is never expired by this
    /// rule; age it out with `before_version` instead.
    pub before_timestamp: Option<DateTime<Utc>>,
    /// Expire versions numbered below this. Exact, and needs no timestamps.
    pub before_version: Option<u64>,
    /// Among the versions this policy would expire, keep the newest one in each bucket of
    /// this width. `None` expires all of them.
    ///
    /// Hourly is `Duration::from_secs(3600)`, daily `86_400`. Applies only beyond the
    /// cutoff, so a `before_timestamp` of 7 days with an hourly bucket reads as "full
    /// history for a week, hourly before that".
    ///
    /// Versions with no reported write time cannot be bucketed and are kept.
    ///
    /// # Thinning carries a rollback risk on an actively committing table
    ///
    /// Resolving the latest version starts at the version hint and probes upward, stopping
    /// at the first version that is missing. [`ProtectedVersions`] therefore refuses to
    /// expire anything at or above the hint — but that floor is read once, and hint writes
    /// are unconditional best effort, so a commit that started earlier can publish a
    /// *lower* hint after the read. A gap above that lowered hint hides every version above
    /// it, and the table reads as an older state while the newer manifests are still there.
    ///
    /// [`MIN_KEEP_ONE_PER`] bounds the exposure rather than removing it: requiring at least
    /// an hour keeps survivors an hour apart, so a deletion is never adjacent to the
    /// second-scale window in which a stale hint can land. It does not make the floor
    /// monotonic, and only a conditional hint write would.
    ///
    /// Treat thinning as a repair for a table that has accumulated far more versions than
    /// it can carry, not as a default to leave switched on.
    pub keep_one_per: Option<Duration>,
    /// Return an error instead of silently keeping a tagged version that the policy
    /// would otherwise expire.
    pub error_if_tagged_old_versions: bool,
    /// Maximum delete requests per second. One permit is one manifest.
    pub delete_rate_limit: Option<u64>,
}

impl Default for ExpireVersionsPolicy {
    fn default() -> Self {
        Self {
            before_timestamp: None,
            before_version: None,
            keep_one_per: None,
            // Matches `CleanupPolicy`: refusing is the safe default when a tag would be
            // silently ignored.
            error_if_tagged_old_versions: true,
            delete_rate_limit: None,
        }
    }
}

/// Builder for [`ExpireVersionsPolicy`].
#[derive(Clone, Debug, Default)]
pub struct ExpireVersionsPolicyBuilder {
    policy: ExpireVersionsPolicy,
}

impl ExpireVersionsPolicyBuilder {
    pub fn before_timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.policy.before_timestamp = Some(timestamp);
        self
    }

    pub fn before_version(mut self, version: u64) -> Self {
        self.policy.before_version = Some(version);
        self
    }

    pub fn keep_one_per(mut self, width: Duration) -> Self {
        self.policy.keep_one_per = Some(width);
        self
    }

    pub fn error_if_tagged_old_versions(mut self, error: bool) -> Self {
        self.policy.error_if_tagged_old_versions = error;
        self
    }

    pub fn delete_rate_limit(mut self, rate: u64) -> Self {
        self.policy.delete_rate_limit = Some(rate);
        self
    }

    pub fn build(self) -> ExpireVersionsPolicy {
        self.policy
    }
}

impl ExpireVersionsPolicy {
    /// True when `location` is old enough for this policy to expire it.
    ///
    /// All conditions must hold, matching `CleanupPolicy::should_clean`. A policy with no
    /// condition at all expires nothing, rather than everything.
    fn is_expirable(&self, location: &ManifestLocation) -> bool {
        if self.before_timestamp.is_none() && self.before_version.is_none() {
            return false;
        }
        if let Some(before_version) = self.before_version
            && location.version >= before_version
        {
            return false;
        }
        if let Some(before_timestamp) = self.before_timestamp {
            // No reported write time means no evidence it is old.
            match location.last_modified {
                Some(last_modified) if last_modified < before_timestamp => {}
                _ => return false,
            }
        }
        true
    }
}

/// What an expiry run did, or would do.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExpireVersionsStats {
    /// Manifests deleted.
    pub versions_removed: u64,
    /// Manifests left in place, for any reason.
    pub versions_retained: u64,
    /// Bytes of manifest removed, as reported by the listing.
    pub bytes_removed: u64,
    /// Manifests this run tried and failed to delete.
    ///
    /// Expiry is best effort: a failure is counted here and the run continues, so a
    /// non-zero value means the run completed without removing everything it selected.
    /// Those versions are still expirable and a later run will retry them.
    pub failed_deletes: u64,
}

/// What an expiry run would remove, without removing it.
#[derive(Clone, Debug)]
pub struct ExpireVersionsPlan {
    /// The versions that would be deleted, ascending.
    pub versions: Vec<u64>,
    /// Stats as they would be after the run.
    pub stats: ExpireVersionsStats,
    /// Tagged versions the policy would have expired but will keep.
    pub tagged_but_kept: Vec<u64>,
}

/// Select the versions to expire, without deleting anything.
pub async fn plan_expire_versions(
    dataset: &Dataset,
    policy: &ExpireVersionsPolicy,
) -> Result<(ExpireVersionsPlan, Vec<ManifestLocation>)> {
    let protected = ProtectedVersions::resolve(dataset).await?;

    let locations: Vec<ManifestLocation> = dataset
        .commit_handler
        .list_manifest_locations(&dataset.base, &dataset.object_store, false)
        .try_collect()
        .await?;

    // Reject before planning, never mid-run: a width the policy cannot honour should be an
    // error the caller sees, not a silently adjusted or skipped thinning pass.
    if let Some(width) = policy.keep_one_per
        && width < MIN_KEEP_ONE_PER
    {
        return Err(Error::invalid_input(format!(
            "keep_one_per must be at least {:?}, got {:?}; thinning more finely than this \
             puts deletions next to the window where a concurrent commit can lower the \
             version hint. Omit keep_one_per to expire every version past the cutoff.",
            MIN_KEEP_ONE_PER, width
        )));
    }

    if locations.is_empty() {
        return Err(Error::internal(
            "no manifests found; refusing to expire versions".to_string(),
        ));
    }

    // The dataset's own version can lag a concurrent writer, so protect whatever the
    // listing says is newest too. Deleting the newest manifest destroys the dataset.
    let newest_listed = locations.iter().map(|l| l.version).max().unwrap_or(0);
    let protected = ProtectedVersions {
        latest: protected.latest.max(newest_listed),
        ..protected
    };

    let (expirable, retained): (Vec<_>, Vec<_>) = locations
        .into_iter()
        .partition(|l| !protected.contains(l.version) && policy.is_expirable(l));

    let tagged_but_kept: Vec<u64> = {
        let mut v: Vec<u64> = protected
            .tagged_but_expired(
                retained
                    .iter()
                    .filter_map(|l| policy.is_expirable(l).then_some(l.version)),
            )
            .into_iter()
            .collect();
        v.sort_unstable();
        v
    };

    if policy.error_if_tagged_old_versions && !tagged_but_kept.is_empty() {
        return Err(Error::invalid_input(format!(
            "tagged versions {:?} are older than the expiry cutoff; \
             pass error_if_tagged_old_versions=false to keep them and continue",
            tagged_but_kept
        )));
    }

    // Thinning runs only over what would otherwise be deleted, so a version inside the
    // retention window is never dropped to satisfy a bucket.
    let (to_delete, thinned_back): (Vec<ManifestLocation>, Vec<ManifestLocation>) =
        match policy.keep_one_per {
            Some(width) => {
                let datable: Vec<(u64, DateTime<Utc>)> = expirable
                    .iter()
                    .filter_map(|l| l.last_modified.map(|t| (l.version, t)))
                    .collect();
                // Versions with no write time cannot be bucketed; keeping them is the
                // conservative choice.
                let undatable: HashSet<u64> = expirable
                    .iter()
                    .filter(|l| l.last_modified.is_none())
                    .map(|l| l.version)
                    .collect();
                let keep = thin_to_one_per(&datable, width);
                expirable
                    .into_iter()
                    .partition(|l| !keep.contains(&l.version) && !undatable.contains(&l.version))
            }
            None => (expirable, Vec::new()),
        };

    let mut versions: Vec<u64> = to_delete.iter().map(|l| l.version).collect();
    versions.sort_unstable();

    let stats = ExpireVersionsStats {
        versions_removed: to_delete.len() as u64,
        versions_retained: (retained.len() + thinned_back.len()) as u64,
        bytes_removed: to_delete.iter().filter_map(|l| l.size).sum(),
        failed_deletes: 0,
    };

    Ok((
        ExpireVersionsPlan {
            versions,
            stats,
            tagged_but_kept,
        },
        to_delete,
    ))
}

/// Report what [`expire_versions`] would remove, without removing it.
pub async fn explain_expire_versions(
    dataset: &Dataset,
    policy: &ExpireVersionsPolicy,
) -> Result<ExpireVersionsPlan> {
    let (plan, _) = plan_expire_versions(dataset, policy).await?;
    Ok(plan)
}

/// Delete expired manifests, and only manifests.
///
/// Data, deletion and transaction files are untouched, including those that only an
/// expired version referenced. Run
/// [`cleanup_old_versions`](super::cleanup::cleanup_old_versions) afterwards to reclaim
/// them.
///
/// Never removes the newest version, a tagged version on the current branch, or a version
/// a branch is rooted at.
pub async fn expire_versions(
    dataset: &Dataset,
    policy: ExpireVersionsPolicy,
) -> Result<ExpireVersionsStats> {
    let (plan, to_delete) = plan_expire_versions(dataset, &policy).await?;
    if to_delete.is_empty() {
        return Ok(plan.stats);
    }

    let identities: HashMap<u64, String> = to_delete
        .iter()
        .filter_map(|l| l.identity.clone().map(|id| (l.version, id)))
        .collect();

    // Count removals from outcomes rather than from the plan, so a manifest that failed
    // to delete is never reported as removed.
    let planned_retained = plan.stats.versions_retained;
    let stats = Mutex::new(ExpireVersionsStats {
        bytes_removed: 0,
        ..Default::default()
    });
    let undeleted: Mutex<HashSet<u64>> = Mutex::new(HashSet::new());

    let source = stream::iter(to_delete.into_iter().map(Ok::<_, Error>));
    let paced = if let Some(rate) = policy.delete_rate_limit {
        let mut ticker = interval(calculate_duration(rate));
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
        IntervalStream::new(ticker)
            .zip(source)
            .map(|(_, location)| location)
            .boxed()
    } else {
        source.boxed()
    };

    let store = &dataset.object_store;
    paced
        .map(|location| async move {
            let location = location?;
            let outcome = match store.delete(&location.path).await {
                Ok(()) => Ok(()),
                // Listed first and deleted after, so a concurrent expiry or cleanup can
                // remove a manifest in between. Already gone is the outcome we wanted.
                Err(Error::NotFound { .. }) => Ok(()),
                Err(error) => Err(error),
            };
            Ok::<_, Error>((location, outcome))
        })
        .buffer_unordered(dataset.object_store.io_parallelism())
        .try_for_each(|(location, outcome)| {
            let mut stats = stats.lock().unwrap();
            match outcome {
                Ok(()) => {
                    stats.versions_removed += 1;
                    stats.bytes_removed += location.size.unwrap_or(0);
                }
                Err(error) => {
                    if stats.failed_deletes == 0 {
                        warn!(
                            "failed to delete manifest for version {}: {}. \
                             Continuing; further failures log at debug.",
                            location.version, error
                        );
                    } else {
                        debug!(
                            "failed to delete manifest for version {}: {}",
                            location.version, error
                        );
                    }
                    stats.failed_deletes += 1;
                    undeleted.lock().unwrap().insert(location.version);
                }
            }
            futures::future::ready(Ok(()))
        })
        .await?;

    let mut stats = stats.into_inner().unwrap();
    // A manifest that could not be deleted is still there, so it counts as retained.
    stats.versions_retained = planned_retained + stats.failed_deletes;

    // Only after the object is gone. A record that outlives its manifest is retired by
    // the next run; the reverse is a lost version.
    let undeleted = undeleted.into_inner().unwrap();
    for (version, identity) in &identities {
        if undeleted.contains(version) {
            continue;
        }
        dataset
            .commit_handler
            .forget_version(&dataset.base, *version, identity)
            .await?;
    }

    Ok(stats)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use all_asserts::assert_gt;
    use arrow_array::{RecordBatch, RecordBatchIterator, UInt32Array};
    use arrow_schema::{DataType, Field, Schema};
    use lance_core::utils::tempfile::TempStrDir;

    use super::*;
    use crate::dataset::WriteParams;

    /// A dataset at version `versions`, each version an overwrite.
    async fn dataset_with_versions(uri: &str, versions: u32) -> Dataset {
        let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::UInt32, false)]));
        let mut dataset = None;
        for v in 0..versions {
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(UInt32Array::from_iter_values(0..(v + 1)))],
            )
            .unwrap();
            let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
            let mode = if v == 0 {
                crate::dataset::WriteMode::Create
            } else {
                crate::dataset::WriteMode::Overwrite
            };
            dataset = Some(
                Dataset::write(
                    reader,
                    uri,
                    Some(WriteParams {
                        mode,
                        ..Default::default()
                    }),
                )
                .await
                .unwrap(),
            );
        }
        dataset.unwrap()
    }

    async fn version_count(dataset: &Dataset) -> usize {
        dataset
            .commit_handler
            .list_manifest_locations(&dataset.base, &dataset.object_store, false)
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
            .len()
    }

    #[tokio::test]
    async fn expire_removes_old_versions_and_keeps_the_latest() {
        let uri = TempStrDir::default();
        let dataset = dataset_with_versions(&uri, 5).await;
        let latest = dataset.manifest.version;
        assert_eq!(version_count(&dataset).await, 5);

        let stats = expire_versions(
            &dataset,
            ExpireVersionsPolicyBuilder::default()
                .before_version(latest)
                .build(),
        )
        .await
        .unwrap();

        assert_eq!(stats.versions_removed, 4);
        assert_eq!(stats.failed_deletes, 0);
        assert_gt!(stats.bytes_removed, 0);

        // Only the latest survives, and the dataset still opens and reads.
        assert_eq!(version_count(&dataset).await, 1);
        let reopened = Dataset::open(&uri).await.unwrap();
        assert_eq!(reopened.manifest.version, latest);
        assert_eq!(reopened.count_rows(None).await.unwrap(), 5);
    }

    #[tokio::test]
    async fn expire_never_removes_the_latest_version() {
        // A policy that would expire everything must still leave the dataset openable.
        let uri = TempStrDir::default();
        let dataset = dataset_with_versions(&uri, 3).await;

        let stats = expire_versions(
            &dataset,
            ExpireVersionsPolicyBuilder::default()
                .before_version(u64::MAX)
                .build(),
        )
        .await
        .unwrap();

        assert_eq!(stats.versions_removed, 2, "the latest must not be removed");
        assert_eq!(version_count(&dataset).await, 1);
        assert!(Dataset::open(&uri).await.is_ok());
    }

    #[tokio::test]
    async fn expire_never_removes_a_tagged_version() {
        let uri = TempStrDir::default();
        let dataset = dataset_with_versions(&uri, 5).await;
        dataset.tags().create("keepme", 2).await.unwrap();

        let stats = expire_versions(
            &dataset,
            ExpireVersionsPolicyBuilder::default()
                .before_version(dataset.manifest.version)
                // Without this the tagged version turns the run into an error instead.
                .error_if_tagged_old_versions(false)
                .build(),
        )
        .await
        .unwrap();

        // Versions 1, 3, 4 go; 2 is tagged and 5 is latest.
        assert_eq!(stats.versions_removed, 3);
        let surviving = dataset
            .commit_handler
            .list_manifest_locations(&dataset.base, &dataset.object_store, false)
            .map_ok(|l| l.version)
            .try_collect::<HashSet<_>>()
            .await
            .unwrap();
        assert!(surviving.contains(&2), "the tagged version must survive");
        let reopened = Dataset::open(&uri).await.unwrap();
        assert!(
            reopened.checkout_version(2).await.is_ok(),
            "a tagged version must still be checkout-able"
        );
    }

    #[tokio::test]
    async fn expire_errors_on_a_tagged_old_version_by_default() {
        let uri = TempStrDir::default();
        let dataset = dataset_with_versions(&uri, 4).await;
        dataset.tags().create("keepme", 2).await.unwrap();

        let result = expire_versions(
            &dataset,
            ExpireVersionsPolicyBuilder::default()
                .before_version(dataset.manifest.version)
                .build(),
        )
        .await;

        assert!(result.is_err(), "a tagged old version must not be ignored");
        assert_eq!(version_count(&dataset).await, 4, "nothing was deleted");
    }

    #[tokio::test]
    async fn expire_does_not_delete_data_files() {
        // The whole point: history goes, data stays for cleanup to reclaim.
        let uri = TempStrDir::default();
        let dataset = dataset_with_versions(&uri, 4).await;
        let data_before = count_data_files(&dataset).await;

        expire_versions(
            &dataset,
            ExpireVersionsPolicyBuilder::default()
                .before_version(dataset.manifest.version)
                .build(),
        )
        .await
        .unwrap();

        assert_eq!(
            count_data_files(&dataset).await,
            data_before,
            "expire_versions must not touch data files"
        );
    }

    async fn count_data_files(dataset: &Dataset) -> usize {
        dataset
            .object_store
            .read_dir_all(&dataset.base.clone().join("data"), None)
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
            .len()
    }

    #[tokio::test]
    async fn expire_with_no_condition_removes_nothing() {
        // A default policy must be inert, not a request to delete all history.
        let uri = TempStrDir::default();
        let dataset = dataset_with_versions(&uri, 3).await;

        let stats = expire_versions(&dataset, ExpireVersionsPolicy::default())
            .await
            .unwrap();

        assert_eq!(stats.versions_removed, 0);
        assert_eq!(version_count(&dataset).await, 3);
    }

    #[tokio::test]
    async fn expire_will_not_delete_at_or_above_the_version_hint() {
        // Resolving the latest version starts at the hint and probes upward, stopping at
        // the first version that is missing. A hole above the hint therefore hides every
        // version above it and the dataset reads as rolled back, so nothing from the hint
        // up may be expired even when the policy asks for it.
        let uri = TempStrDir::default();
        let dataset = dataset_with_versions(&uri, 5).await;
        let latest = dataset.manifest.version;

        // Pin the hint below the latest so the guard has something to protect that the
        // ordinary "never remove the latest" rule would not already cover.
        lance_table::io::commit::write_version_hint(&dataset.object_store, &dataset.base, 3).await;

        let plan = explain_expire_versions(
            &dataset,
            &ExpireVersionsPolicyBuilder::default()
                .before_version(latest)
                .build(),
        )
        .await
        .unwrap();

        // Only 1 and 2 are expirable: 3 is the hint, 4 is above it, 5 is latest.
        assert_eq!(
            plan.versions,
            vec![1, 2],
            "nothing at or above the hint may be expired"
        );
    }

    #[tokio::test]
    async fn expire_on_main_keeps_versions_a_branch_is_rooted_at() {
        // A branch is anchored to a version of its parent. Expiring that version would
        // orphan the branch's history, so it must survive even though the policy covers it.
        let uri = TempStrDir::default();
        let mut dataset = dataset_with_versions(&uri, 5).await;
        let latest = dataset.manifest.version;
        dataset.create_branch("feature", 2u64, None).await.unwrap();

        let plan = explain_expire_versions(
            &dataset,
            &ExpireVersionsPolicyBuilder::default()
                .before_version(latest)
                .build(),
        )
        .await
        .unwrap();

        assert!(
            !plan.versions.contains(&2),
            "the version the branch is rooted at must not be expired, got {:?}",
            plan.versions
        );
    }

    #[tokio::test]
    async fn expire_on_a_branch_leaves_the_parent_alone() {
        // `base` follows the checkout, so expiring on a branch must act on the branch's
        // own manifests and leave the parent's history untouched.
        let uri = TempStrDir::default();
        let mut dataset = dataset_with_versions(&uri, 4).await;
        let main_before = version_count(&dataset).await;

        let branch = dataset.create_branch("feature", 4u64, None).await.unwrap();
        assert_ne!(branch.base, dataset.base, "a branch has its own path");

        let stats = expire_versions(
            &branch,
            ExpireVersionsPolicyBuilder::default()
                .before_version(branch.manifest.version)
                .build(),
        )
        .await
        .unwrap();
        assert_eq!(stats.failed_deletes, 0);

        // The parent still has every version it started with, and still opens.
        assert_eq!(
            version_count(&dataset).await,
            main_before,
            "expiring on a branch must not touch the parent"
        );
        assert!(Dataset::open(&uri).await.is_ok());
    }

    /// Append one small batch to `dataset`, producing one new version.
    async fn append_one(dataset: &mut Dataset) {
        let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::UInt32, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(UInt32Array::from_iter_values(0..3))],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        dataset
            .append(
                reader,
                Some(WriteParams {
                    mode: crate::dataset::WriteMode::Append,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn expire_on_a_branch_keeps_versions_its_own_child_is_rooted_at() {
        // main -> b1 -> b2. Expiring on b1 has to respect b2, which is anchored to a
        // version of b1 rather than of main, so the protection has to resolve against the
        // branch being expired and not just against the root.
        let uri = TempStrDir::default();
        let mut main = dataset_with_versions(&uri, 3).await;

        let mut b1 = main.create_branch("b1", 2u64, None).await.unwrap();
        append_one(&mut b1).await;
        append_one(&mut b1).await;
        let b1_anchor = b1.manifest.version;
        append_one(&mut b1).await;

        // b2 hangs off b1 at the version b1 had after two appends.
        let _b2 = b1.create_branch("b2", b1_anchor, None).await.unwrap();

        let plan = explain_expire_versions(
            &b1,
            &ExpireVersionsPolicyBuilder::default()
                .before_version(b1.manifest.version)
                .build(),
        )
        .await
        .unwrap();

        assert!(
            !plan.versions.contains(&b1_anchor),
            "b1@{} anchors b2 and must not be expired, got {:?}",
            b1_anchor,
            plan.versions
        );
    }

    #[tokio::test]
    async fn expire_on_a_grandchild_branch_leaves_its_ancestors_alone() {
        // Expiring at the bottom of a chain must not disturb anything above it.
        let uri = TempStrDir::default();
        let mut main = dataset_with_versions(&uri, 3).await;
        let main_before = version_count(&main).await;

        let mut b1 = main.create_branch("b1", 2u64, None).await.unwrap();
        append_one(&mut b1).await;
        append_one(&mut b1).await;
        let b1_before = version_count(&b1).await;

        let mut b2 = b1
            .create_branch("b2", b1.manifest.version, None)
            .await
            .unwrap();
        append_one(&mut b2).await;
        append_one(&mut b2).await;

        let stats = expire_versions(
            &b2,
            ExpireVersionsPolicyBuilder::default()
                .before_version(b2.manifest.version)
                .build(),
        )
        .await
        .unwrap();
        assert_eq!(stats.failed_deletes, 0);

        assert_eq!(
            version_count(&b1).await,
            b1_before,
            "expiring on b2 must not touch b1"
        );
        assert_eq!(
            version_count(&main).await,
            main_before,
            "expiring on b2 must not touch main"
        );
        assert!(Dataset::open(&uri).await.is_ok());
    }

    #[tokio::test]
    async fn expire_rejects_a_keep_one_per_below_the_minimum() {
        // Thinning competes with the version hint, whose protective floor is read once and
        // can be lowered afterwards by a slow committer. Keeping survivors at least an
        // hour apart keeps deletions away from that window, so finer widths are refused
        // rather than quietly accepted.
        let uri = TempStrDir::default();
        let dataset = dataset_with_versions(&uri, 4).await;
        let before = version_count(&dataset).await;

        for width in [
            Duration::from_nanos(1),
            Duration::from_secs(0),
            MIN_KEEP_ONE_PER - Duration::from_secs(1),
        ] {
            let result = expire_versions(
                &dataset,
                ExpireVersionsPolicyBuilder::default()
                    .before_version(dataset.manifest.version)
                    .keep_one_per(width)
                    .build(),
            )
            .await;
            assert!(
                result.is_err(),
                "{:?} is below the minimum and must be refused",
                width
            );
        }
        assert_eq!(version_count(&dataset).await, before, "nothing was deleted");

        // Exactly the minimum is accepted.
        let plan = explain_expire_versions(
            &dataset,
            &ExpireVersionsPolicyBuilder::default()
                .before_version(dataset.manifest.version)
                .keep_one_per(MIN_KEEP_ONE_PER)
                .build(),
        )
        .await;
        assert!(plan.is_ok(), "the minimum itself must be accepted");
    }

    #[tokio::test]
    async fn explain_expire_versions_deletes_nothing() {
        let uri = TempStrDir::default();
        let dataset = dataset_with_versions(&uri, 4).await;

        let plan = explain_expire_versions(
            &dataset,
            &ExpireVersionsPolicyBuilder::default()
                .before_version(dataset.manifest.version)
                .build(),
        )
        .await
        .unwrap();

        assert_eq!(plan.versions, vec![1, 2, 3]);
        assert_eq!(plan.stats.versions_removed, 3);
        assert_eq!(version_count(&dataset).await, 4, "explain must not delete");
    }
}
