// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Retention files and restore pins, which stop cleanup from deleting a
//! version that a tag, branch, or restore references concurrently.
//!
//! Cleanup snapshots references before planning, so a reference published
//! after that snapshot could point at a version being deleted. Cleanup writes
//! the rule first and lists references after; a publisher writes its reference
//! first and reads the rule after. Whichever goes second sees the other. See
//! the "History Retention" section of the branch and tag format specification.

use std::collections::HashSet;

use chrono::{DateTime, TimeDelta, Utc};
use futures::{StreamExt, TryStreamExt, stream};
use lance_io::object_store::ObjectStore;
use lance_table::format::history_retention::{HistoryRetention, RETENTION_DIR};
use object_store::Error as ObjectStoreError;
use object_store::path::Path;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::Dataset;
use super::refs::{REFS_DIR, Ref};
use super::version_retention::ProtectedVersions;
use crate::utils::temporal::utc_now;
use crate::{Error, Result};

/// Concurrent cleanups merge their rules on retry, so each retry only loses to
/// a cleanup that wrote in the meantime.
const MAX_WRITE_ATTEMPTS: usize = 10;

/// Pins older than this were left by restores that did not finish.
const STALE_PIN_AGE: TimeDelta = TimeDelta::hours(24);

/// The retention files of one branch.
pub struct RetentionFiles<'a> {
    object_store: &'a ObjectStore,
    dir: Path,
}

impl<'a> RetentionFiles<'a> {
    /// The retention files of the branch whose root is `branch_root`.
    pub fn new(object_store: &'a ObjectStore, branch_root: &Path) -> Self {
        Self {
            object_store,
            dir: branch_root.clone().join(RETENTION_DIR),
        }
    }

    fn path(&self, number: u64) -> Path {
        self.dir.clone().join(format!("{number}.json"))
    }

    /// File numbers present, ascending. Names that are not `{n}.json` are not
    /// retention files and are ignored.
    async fn numbers(&self) -> Result<Vec<u64>> {
        let names = match self.object_store.read_dir(self.dir.clone()).await {
            Ok(names) => names,
            Err(error) if error.is_not_found() => return Ok(Vec::new()),
            Err(error) => return Err(error),
        };
        let mut numbers: Vec<u64> = names
            .iter()
            .filter_map(|name| name.strip_suffix(".json")?.parse().ok())
            .collect();
        numbers.sort_unstable();
        Ok(numbers)
    }

    /// The number of the current file (0 when there is none) and its rule.
    ///
    /// A malformed current file expires nothing. A current file that vanishes
    /// between listing and reading was superseded by a newer one, so the
    /// listing is retried.
    async fn current_numbered(&self) -> Result<(u64, Option<HistoryRetention>)> {
        loop {
            let Some(&number) = self.numbers().await?.last() else {
                return Ok((0, None));
            };
            match self.object_store.read_one_all(&self.path(number)).await {
                Ok(bytes) => return Ok((number, HistoryRetention::from_json(&bytes))),
                Err(error) if error.is_not_found() => continue,
                Err(error) => return Err(error),
            }
        }
    }

    /// The branch's current rule, or `None` when it expires nothing.
    pub async fn current(&self) -> Result<Option<HistoryRetention>> {
        Ok(self.current_numbered().await?.1)
    }

    /// Writes `planned` merged with the current rule, unless that changes
    /// nothing, and returns the rule now in effect.
    ///
    /// The merge keeps expiration monotonic: a version an earlier cleanup
    /// expired may still be in the middle of deletion, so a later plan must
    /// not make it referenceable again.
    pub async fn write(&self, planned: HistoryRetention) -> Result<HistoryRetention> {
        for _ in 0..MAX_WRITE_ATTEMPTS {
            let (number, current) = self.current_numbered().await?;
            let merged = match &current {
                Some(current) => planned.merge(current),
                None => planned.clone(),
            };
            if current.as_ref() == Some(&merged) {
                return Ok(merged);
            }
            match self
                .object_store
                .put_if_absent(&self.path(number + 1), merged.to_json().into())
                .await
            {
                Ok(()) => return Ok(merged),
                Err(
                    ObjectStoreError::AlreadyExists { .. } | ObjectStoreError::Precondition { .. },
                ) => {
                    continue;
                }
                Err(error) => return Err(error.into()),
            }
        }
        Err(Error::internal(format!(
            "could not write the history retention rule after {MAX_WRITE_ATTEMPTS} attempts \
             because of concurrent cleanups; nothing was deleted"
        )))
    }

    /// Deletes every retention file but the current one.
    pub async fn delete_superseded(&self) -> Result<()> {
        let mut numbers = self.numbers().await?;
        numbers.pop();
        for number in numbers {
            match self.object_store.delete(&self.path(number)).await {
                Ok(()) => {}
                Err(error) if error.is_not_found() => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }
}

/// Fails if cleanup has expired `version` of the branch rooted at `branch_root`.
///
/// Publishers call this after writing their reference, so that either cleanup
/// lists the reference or this check sees the rule that cleanup wrote first.
pub async fn ensure_not_expired(
    object_store: &ObjectStore,
    branch_root: &Path,
    branch: Option<&str>,
    version: u64,
) -> Result<()> {
    match RetentionFiles::new(object_store, branch_root)
        .current()
        .await?
    {
        Some(rule) if rule.is_expired(version) => Err(Error::VersionNotFound {
            message: format!(
                "version {} was expired by cleanup (versions below {} are expired) and can \
                 no longer be referenced",
                Ref::Version(branch.map(str::to_string), Some(version)),
                rule.expired_before()
            ),
        }),
        _ => Ok(()),
    }
}

/// The rule a run deleting `deleted` and keeping `kept` must write, or `None`
/// when it deletes no version.
pub fn planned_rule(
    deleted: impl IntoIterator<Item = u64>,
    kept: impl IntoIterator<Item = u64>,
) -> Option<HistoryRetention> {
    let max_deleted = deleted.into_iter().max()?;
    Some(HistoryRetention::new(max_deleted + 1, kept))
}

/// Writes `rule` for the dataset's branch, then lists the references that
/// exist afterwards.
///
/// The caller must keep every version that a reference it did not plan for
/// points at, and must not delete anything if this returns an error.
///
/// `started_at` is when the caller began planning; see [`pinned_versions`].
pub async fn publish(
    dataset: &Dataset,
    rule: HistoryRetention,
    started_at: DateTime<Utc>,
) -> Result<ListedReferences> {
    RetentionFiles::new(&dataset.object_store, &dataset.base)
        .write(rule)
        .await?;
    let tags = dataset.tags().list().await?;
    let mut tagged_or_pinned =
        ProtectedVersions::tagged_versions(&tags, dataset.manifest.branch.as_ref());
    tagged_or_pinned.extend(pinned_versions(dataset, started_at).await?);
    let branches = dataset.branches().list().await?;
    let branch_roots = dataset
        .branch_identifier()
        .await?
        .collect_referenced_versions(&branches);
    Ok(ListedReferences {
        tagged_or_pinned,
        branch_roots,
    })
}

/// The references cleanup lists after writing its rule.
#[derive(Debug, Default)]
pub struct ListedReferences {
    /// Versions that tags and live pins on this branch point at.
    pub tagged_or_pinned: HashSet<u64>,
    /// Branches rooted on this branch, by name, with their root version.
    pub branch_roots: Vec<(String, u64)>,
}

impl ListedReferences {
    /// Every referenced version.
    pub fn versions(&self) -> HashSet<u64> {
        self.tagged_or_pinned
            .iter()
            .copied()
            .chain(self.branch_roots.iter().map(|(_, version)| *version))
            .collect()
    }

    /// Versions referenced by something the plan did not account for.
    ///
    /// The plan kept every tagged and pinned version, so any of those still
    /// counts. Branch roots are compared by branch: the plan may delete the
    /// root of a branch that no longer uses its files, and only a different
    /// branch on that root must keep it.
    pub fn versions_unplanned(&self, planned_branch_roots: &[(String, u64)]) -> HashSet<u64> {
        self.tagged_or_pinned
            .iter()
            .copied()
            .chain(
                self.branch_roots
                    .iter()
                    .filter(|root| !planned_branch_roots.contains(root))
                    .map(|(_, version)| *version),
            )
            .collect()
    }
}

/// Tidies up after a run that deleted versions: drops superseded retention
/// files and stale pins.
///
/// The deletions already happened, so a failure here is only logged; the next
/// run tries again.
pub async fn finish_deleting(dataset: &Dataset, started_at: DateTime<Utc>) {
    if let Err(error) = RetentionFiles::new(&dataset.object_store, &dataset.base)
        .delete_superseded()
        .await
    {
        log::warn!("failed to delete superseded history retention files: {error}");
    }
    if let Err(error) = delete_stale_pins(dataset, started_at).await {
        log::warn!("failed to delete stale restore pins: {error}");
    }
}

/// A short-lived reference that a restore writes before it commits.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PinContents {
    branch: Option<String>,
    version: u64,
    created_at: DateTime<Utc>,
}

fn pins_dir(dataset: &Dataset) -> Result<Path> {
    Ok(dataset.refs.root()?.path.join(REFS_DIR).join("pins"))
}

/// A pin on a version. A restore removes it only when it fails before
/// committing; otherwise cleanup deletes it once stale.
pub struct Pin {
    path: Path,
}

impl Pin {
    /// Pins the dataset's checked-out version on its branch.
    pub async fn write(dataset: &Dataset) -> Result<Self> {
        let contents = PinContents {
            branch: dataset.manifest.branch.clone(),
            version: dataset.manifest.version,
            created_at: utc_now(),
        };
        let path = pins_dir(dataset)?.join(format!("{}.json", Uuid::new_v4()));
        let json = serde_json::to_vec(&contents)
            .map_err(|error| Error::internal(format!("failed to encode pin: {error}")))?;
        dataset.object_store.put(&path, &json).await?;
        Ok(Self { path })
    }

    /// Removes the pin. A pin that cannot be removed only keeps its version
    /// alive until it is stale, so the failure is logged.
    pub async fn remove(self, dataset: &Dataset) {
        if let Err(error) = dataset.object_store.delete(&self.path).await
            && !error.is_not_found()
        {
            log::warn!("failed to remove restore pin {}: {error}", self.path);
        }
    }
}

/// Every pin with its path, skipping files that vanished or cannot be parsed.
async fn list_pins(dataset: &Dataset) -> Result<Vec<(Path, PinContents)>> {
    let dir = pins_dir(dataset)?;
    let names = match dataset.object_store.read_dir(dir.clone()).await {
        Ok(names) => names,
        Err(error) if error.is_not_found() => return Ok(Vec::new()),
        Err(error) => return Err(error),
    };
    stream::iter(names)
        .map(|name| {
            let path = dir.clone().join(name);
            async move {
                match dataset.object_store.read_one_all(&path).await {
                    Ok(bytes) => Ok(serde_json::from_slice(&bytes)
                        .ok()
                        .map(|contents| (path, contents))),
                    Err(error) if error.is_not_found() => Ok(None),
                    Err(error) => Err(error),
                }
            }
        })
        .buffer_unordered(10)
        .try_filter_map(|pin| async move { Ok(pin) })
        .try_collect()
        .await
}

/// Whether `pin` was created more than [`STALE_PIN_AGE`] before `started_at`.
///
/// Age is measured from when cleanup started, not from now: a pin created
/// after cleanup planned guards a restore that planning could not see, so it
/// must hold for the rest of the run however long that takes.
fn is_stale(pin: &PinContents, started_at: DateTime<Utc>) -> bool {
    started_at - pin.created_at > STALE_PIN_AGE
}

/// Versions of the dataset's branch that a pin not stale for a cleanup that
/// started at `started_at` references.
pub async fn pinned_versions(dataset: &Dataset, started_at: DateTime<Utc>) -> Result<HashSet<u64>> {
    Ok(list_pins(dataset)
        .await?
        .into_iter()
        .filter(|(_, pin)| pin.branch == dataset.manifest.branch && !is_stale(pin, started_at))
        .map(|(_, pin)| pin.version)
        .collect())
}

/// Deletes pins left behind by restores that did not finish.
async fn delete_stale_pins(dataset: &Dataset, started_at: DateTime<Utc>) -> Result<()> {
    for (path, pin) in list_pins(dataset).await? {
        if is_stale(&pin, started_at) {
            match dataset.object_store.delete(&path).await {
                Ok(()) => {}
                Err(error) if error.is_not_found() => {}
                Err(error) => return Err(error),
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{RecordBatch, RecordBatchIterator, record_batch};
    use mock_instant::thread_local::MockClock;

    use super::*;
    use crate::dataset::cleanup::{CleanupPolicyBuilder, cleanup_old_versions};
    use crate::dataset::{WriteDestination, WriteMode, WriteParams};

    /// An in-memory dataset at version `versions`, one append per version.
    async fn dataset_with_versions(versions: i32) -> Dataset {
        let reader = |value: i32| {
            let batch: RecordBatch = record_batch!(("x", Int32, [value])).unwrap();
            let schema = batch.schema();
            RecordBatchIterator::new(vec![Ok(batch)], schema)
        };
        let mut dataset = Dataset::write(reader(1), "memory://", None).await.unwrap();
        for value in 2..=versions {
            dataset = Dataset::write(
                reader(value),
                WriteDestination::Dataset(Arc::new(dataset)),
                Some(WriteParams {
                    mode: WriteMode::Append,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        }
        dataset
    }

    fn files(dataset: &Dataset) -> RetentionFiles<'_> {
        RetentionFiles::new(&dataset.object_store, &dataset.base)
    }

    async fn expire(dataset: &Dataset, expired_before: u64, retained: &[u64]) -> HistoryRetention {
        files(dataset)
            .write(HistoryRetention::new(
                expired_before,
                retained.iter().copied(),
            ))
            .await
            .unwrap()
    }

    fn assert_expired_error(error: Error) {
        assert!(matches!(error, Error::VersionNotFound { .. }), "{error}");
        assert!(error.to_string().contains("expired"), "{error}");
    }

    #[test]
    fn a_new_reference_on_a_released_branch_root_is_unplanned() {
        // The plan listed branch "b" on version 50 and decided to delete 50
        // because "b" no longer uses its files.
        let planned = [("b".to_string(), 50)];
        let unchanged = ListedReferences {
            tagged_or_pinned: HashSet::new(),
            branch_roots: planned.to_vec(),
        };
        assert!(unchanged.versions_unplanned(&planned).is_empty());

        let tagged = ListedReferences {
            tagged_or_pinned: HashSet::from([50]),
            branch_roots: planned.to_vec(),
        };
        assert_eq!(tagged.versions_unplanned(&planned), HashSet::from([50]));

        let branched = ListedReferences {
            tagged_or_pinned: HashSet::new(),
            branch_roots: vec![("b".to_string(), 50), ("x".to_string(), 50)],
        };
        assert_eq!(branched.versions_unplanned(&planned), HashSet::from([50]));
    }

    #[tokio::test]
    async fn writes_merge_with_the_current_rule() {
        let dataset = dataset_with_versions(1).await;
        let files = files(&dataset);
        assert_eq!(files.current().await.unwrap(), None);

        expire(&dataset, 3, &[]).await;
        // A later plan that keeps 1 (say, a tag appeared mid-run) must not
        // make it referenceable again.
        let merged = expire(&dataset, 4, &[1, 3]).await;
        assert_eq!(merged, HistoryRetention::new(4, [3]));
        assert_eq!(files.current().await.unwrap(), Some(merged.clone()));
        assert_eq!(files.numbers().await.unwrap(), vec![1, 2]);

        // Writing what is already in effect adds no file.
        expire(&dataset, 2, &[]).await;
        assert_eq!(files.numbers().await.unwrap(), vec![1, 2]);

        files.delete_superseded().await.unwrap();
        assert_eq!(files.numbers().await.unwrap(), vec![2]);
        assert_eq!(files.current().await.unwrap(), Some(merged));
    }

    #[tokio::test]
    async fn malformed_current_file_expires_nothing() {
        let dataset = dataset_with_versions(1).await;
        let files = files(&dataset);
        expire(&dataset, 3, &[]).await;
        dataset
            .object_store
            .put(&files.path(2), b"not json")
            .await
            .unwrap();

        assert_eq!(files.current().await.unwrap(), None);
        expire(&dataset, 2, &[]).await;
        assert_eq!(files.numbers().await.unwrap(), vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn tag_create_and_update_respect_rule() {
        let dataset = dataset_with_versions(4).await;
        expire(&dataset, 3, &[1]).await;
        let tags = dataset.tags();

        assert_expired_error(tags.create("expired", 2).await.unwrap_err());
        assert!(tags.list().await.unwrap().is_empty());

        tags.create("retained", 1).await.unwrap();
        tags.create("above", 3).await.unwrap();
        assert_expired_error(tags.update("above", 2).await.unwrap_err());
        assert_eq!(tags.get_version("above").await.unwrap(), 3);
    }

    #[tokio::test]
    async fn branch_create_rejects_expired_version() {
        let mut dataset = dataset_with_versions(4).await;
        expire(&dataset, 3, &[]).await;

        assert_expired_error(dataset.create_branch("b", 2, None).await.unwrap_err());
        assert!(dataset.list_branches().await.unwrap().is_empty());

        let branch = dataset.create_branch("b", 3, None).await.unwrap();
        assert_eq!(files(&branch).current().await.unwrap(), None);
    }

    #[tokio::test]
    async fn restore_respects_rule_and_keeps_its_pin_after_committing() {
        let dataset = dataset_with_versions(4).await;
        expire(&dataset, 3, &[1]).await;

        let mut expired = dataset.checkout_version(2).await.unwrap();
        assert_expired_error(expired.restore().await.unwrap_err());
        assert!(list_pins(&dataset).await.unwrap().is_empty());

        let mut retained = dataset.checkout_version(1).await.unwrap();
        retained.restore().await.unwrap();
        assert_eq!(retained.version().version, 5);
        assert_eq!(retained.count_rows(None).await.unwrap(), 1);
        // A cleanup that planned before the commit cannot see version 5, so
        // the pin must keep protecting version 1 until it is stale.
        assert_eq!(
            pinned_versions(&dataset, utc_now()).await.unwrap(),
            HashSet::from([1])
        );
    }

    fn retain_latest_policy() -> CleanupPolicyBuilder {
        CleanupPolicyBuilder::default().error_if_tagged_old_versions(false)
    }

    #[tokio::test]
    async fn cleanup_writes_rule_without_committing() {
        let dataset = dataset_with_versions(4).await;
        dataset.tags().create("keep", 2).await.unwrap();

        let policy = retain_latest_policy()
            .retain_n_versions(&dataset, 1)
            .await
            .unwrap()
            .build();
        let removed = cleanup_old_versions(&dataset, policy).await.unwrap();
        assert_eq!(removed.old_versions, 2);

        assert_eq!(dataset.latest_version_id().await.unwrap(), 4);
        assert_eq!(
            files(&dataset).current().await.unwrap(),
            Some(HistoryRetention::new(4, [2]))
        );
        // Version 3 is gone, so the tag fails before the rule is consulted.
        let error = dataset.tags().create("late", 3).await.unwrap_err();
        assert!(matches!(error, Error::VersionNotFound { .. }), "{error}");
    }

    #[tokio::test]
    async fn cleanup_keeps_pinned_versions_until_the_pin_is_stale() {
        let dataset = dataset_with_versions(4).await;
        let pin = Pin::write(&dataset.checkout_version(2).await.unwrap())
            .await
            .unwrap();

        // A pinned version is kept without raising the tagged-version error.
        let policy = CleanupPolicyBuilder::default()
            .retain_n_versions(&dataset, 1)
            .await
            .unwrap()
            .build();
        cleanup_old_versions(&dataset, policy).await.unwrap();
        let versions: Vec<u64> = dataset
            .versions()
            .await
            .unwrap()
            .iter()
            .map(|version| version.version)
            .collect();
        assert_eq!(versions, vec![2, 4]);
        assert!(!list_pins(&dataset).await.unwrap().is_empty());

        MockClock::advance_system_time((STALE_PIN_AGE + TimeDelta::hours(1)).to_std().unwrap());
        let policy = CleanupPolicyBuilder::default()
            .retain_n_versions(&dataset, 1)
            .await
            .unwrap()
            .build();
        cleanup_old_versions(&dataset, policy).await.unwrap();
        assert_eq!(dataset.versions().await.unwrap().len(), 1);
        assert!(list_pins(&dataset).await.unwrap().is_empty());
        pin.remove(&dataset).await;
    }

    #[tokio::test]
    async fn pins_created_during_a_long_cleanup_stay_live_for_it() {
        let dataset = dataset_with_versions(2).await;
        let started_at = utc_now();
        Pin::write(&dataset.checkout_version(1).await.unwrap())
            .await
            .unwrap();

        // The cleanup that started before the pin still honors it after its
        // planning ran past the stale age; a cleanup starting now does not.
        MockClock::advance_system_time((STALE_PIN_AGE + TimeDelta::hours(1)).to_std().unwrap());
        assert_eq!(
            pinned_versions(&dataset, started_at).await.unwrap(),
            HashSet::from([1])
        );
        assert!(
            pinned_versions(&dataset, utc_now())
                .await
                .unwrap()
                .is_empty()
        );
    }
}
