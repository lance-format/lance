// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Explicit-release snapshot retention coordinated with dataset cleanup.
//!
//! All cleanup clients must use this coordination protocol. Older maintenance
//! clients, forced branch/table deletion, and deletion outside Lance are excluded.
//! See the [deployment contract](https://lance.org/guide/snapshot_retention/).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, TimeDelta, Utc};
use futures::future::BoxFuture;
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use object_store::{Error as StoreError, ObjectStoreExt, path::Path};
use prost::Message;
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::Dataset;
use crate::dataset::snapshot::{SnapshotIdentity, canonical_uri, pb};

// Preserve the original public path while general retention remains independent of blob contexts.
pub use crate::blob::reference::BlobSnapshotRetention;
use crate::dataset::refs::{TagContents, tag_path};
use crate::utils::temporal::utc_now;

const PROTOCOL_VERSION: u32 = 1;
const TAG_PREFIX: &str = "lance-query-";
const SNAPSHOT_ID_KEY: &str = "lance.snapshot_retention.snapshot_id";

/// Default maintenance owner lifetime, in hours. Query tags do not expire.
pub const DEFAULT_LOCK_EXPIRATION_HOURS: u64 = 120;
/// Table configuration key for a positive, integer maintenance lifetime in hours.
/// Each owner records its expiry; contenders always honor that recorded deadline.
pub const LOCK_EXPIRATION_HOURS_KEY: &str = "lance.snapshot_retention.lock_expiration_hours";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MaintenanceOwner {
    ticket: u64,
    expires_at: DateTime<Utc>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MaintenanceDeadline {
    expires_at: DateTime<Utc>,
    started_at: Instant,
    lifetime: Duration,
}

impl MaintenanceDeadline {
    fn from_dataset(dataset: &Dataset) -> Result<Self> {
        let hours = dataset.manifest().config.get(LOCK_EXPIRATION_HOURS_KEY)
            .map(|value| value.parse::<u64>().map_err(|error| Error::invalid_input(format!("{LOCK_EXPIRATION_HOURS_KEY} must be a positive integer in hours, got {value:?}: {error}"))))
            .transpose()?.unwrap_or(DEFAULT_LOCK_EXPIRATION_HOURS);
        let lifetime = i64::try_from(hours).ok().filter(|hours| *hours > 0)
            .and_then(TimeDelta::try_hours)
            .ok_or_else(|| Error::invalid_input(format!("{LOCK_EXPIRATION_HOURS_KEY} must be positive and representable in hours, got {hours}")))?;
        let now = utc_now();
        let expires_at = now.checked_add_signed(lifetime).ok_or_else(|| {
            Error::invalid_input(format!(
                "{LOCK_EXPIRATION_HOURS_KEY}={hours} overflows the maintenance expiry timestamp"
            ))
        })?;
        Ok(Self {
            expires_at,
            started_at: Instant::now(),
            lifetime: lifetime.to_std().map_err(|error| {
                Error::invalid_input(format!(
                    "Invalid maintenance lifetime {hours} hours: {error}"
                ))
            })?,
        })
    }

    pub(crate) fn check(&self) -> Result<()> {
        if utc_now() >= self.expires_at || self.started_at.elapsed() >= self.lifetime {
            return Err(Error::RefConflict {
                message: format!(
                    "Snapshot maintenance lock expired at {}; stop protected work and retry the complete operation",
                    self.expires_at
                ),
            });
        }
        Ok(())
    }
}

pub(crate) type MaintenanceDelete =
    dyn Fn(Arc<ObjectStore>, Path) -> BoxFuture<'static, Result<()>> + Send + Sync;

fn retention_dir(dataset: &Dataset) -> Result<Path> {
    Ok(dataset.refs.root()?.path.join("_refs").join("retention-v1"))
}

fn retirement_path(dataset: &Dataset, manifest: &Path) -> Result<Path> {
    Ok(retention_dir(dataset)?
        .join("retiring")
        .join(blake3::hash(manifest.as_ref().as_bytes()).to_hex().as_str()))
}

/// An exclusive, storage-backed fence between retention acquisition and cleanup.
///
/// Contention returns [`Error::RefConflict`]; callers may retry the whole operation.
/// Owners expire after [`DEFAULT_LOCK_EXPIRATION_HOURS`] unless the table config
/// [`LOCK_EXPIRATION_HOURS_KEY`] overrides it. Expired owners are ignored and
/// removed by later attempts. There is no renewal. Dropping the handle leaves its
/// owner until release or expiration. Query tags and retirement markers do not expire.
///
/// Expiration is a bounded-operation deployment contract, not storage-enforced
/// fencing: participating operations, pauses, retries, and submitted storage work
/// must finish before their recorded expiry, and client clocks must agree. An
/// expired owner must stop work; expiry cannot cancel an already submitted delete.
///
/// This hook also lets a maintenance coordinator serialize its own operations with
/// Lance. While holding it, do not call cleanup or retention acquisition, which
/// acquire the same lock. It does not protect files from uncoordinated deletion.
///
/// ```
/// # use lance::{Dataset, Result};
/// # use lance::dataset::retention::SnapshotMaintenanceLock;
/// # async fn example(dataset: &Dataset) -> Result<()> {
/// let fence = SnapshotMaintenanceLock::acquire(dataset).await?;
/// // Perform maintenance that cannot invalidate retained snapshots.
/// fence.release().await?;
/// # Ok(())
/// # }
/// ```
pub struct SnapshotMaintenanceLock {
    store: Arc<ObjectStore>,
    path: Path,
    delete: Option<Arc<MaintenanceDelete>>,
    deadline: MaintenanceDeadline,
}

impl SnapshotMaintenanceLock {
    pub(crate) async fn for_cleanup(
        dataset: &Dataset,
        delete: Arc<MaintenanceDelete>,
    ) -> Result<Option<Self>> {
        // COS cannot acquire a retention root because put_if_absent rejects it.
        // Preserve ordinary cleanup on that backend without promising retention.
        if dataset.object_store.scheme() == "cos" {
            return Ok(None);
        }
        Self::acquire_inner(dataset, Some(delete)).await.map(Some)
    }

    /// Acquire the table-root fence, shared by every branch and worker process.
    /// Storage must implement atomic conditional creation and consistent reads/listing.
    pub async fn acquire(dataset: &Dataset) -> Result<Self> {
        Self::acquire_inner(dataset, None).await
    }

    async fn acquire_inner(
        dataset: &Dataset,
        delete: Option<Arc<MaintenanceDelete>>,
    ) -> Result<Self> {
        let directory = retention_dir(dataset)?.join("maintenance");
        let owner = uuid::Uuid::new_v4().to_string();
        let path = directory.clone().join(owner.as_str());
        let deadline = MaintenanceDeadline::from_dataset(dataset)?;
        let mut record = MaintenanceOwner {
            ticket: 0,
            expires_at: deadline.expires_at,
        };
        // Zero announces that this owner is choosing a ticket. Every attempt has
        // its own key: a retried DELETE must never remove a subsequent owner's lock.
        dataset
            .object_store
            .put_if_absent(&path, serde_json::to_vec(&record)?.into())
            .await
            .map_err(|error| match error {
                StoreError::NotSupported { source } => Error::not_supported(format!(
                    "Snapshot retention requires atomic conditional creation: {source}"
                )),
                error => error.into(),
            })?;
        let fence = Self {
            store: dataset.object_store.clone(),
            path,
            delete,
            deadline,
        };
        let result = async {
            fence.check_valid()?;
            let ticket = maintenance_tickets(&fence, &directory).await?
                .iter().map(|(_, ticket)| *ticket).max().unwrap_or(0)
                .checked_add(1).ok_or_else(|| Error::invalid_input("Snapshot maintenance ticket overflow"))?;
            fence.check_valid()?;
            record.ticket = ticket;
            fence.store.put(&fence.path, &serde_json::to_vec(&record)?).await?;
            // Publishing before checking ensures a later chooser gets a higher
            // ticket. Concurrent choosers break ties by their unique owner IDs.
            for (other_owner, other_ticket) in maintenance_tickets(&fence, &directory).await? {
                if other_owner != owner && (other_ticket == 0 || (other_ticket, &other_owner) < (ticket, &owner)) {
                    return Err(Error::RefConflict { message: format!(
                        "Snapshot maintenance lock {} is held; retry after the operation finishes. \
                         The owner's recorded expiration is honored independently of this caller's configuration",
                        directory.clone().join(other_owner),
                    ) });
                }
            }
            fence.check_valid()?;
            Ok(())
        }.await;
        if let Err(error) = result {
            fence.release().await?;
            return Err(error);
        }
        Ok(fence)
    }

    /// Release after all protected work and submitted storage operations have finished.
    /// Dropping the handle deliberately leaves the lock on storage.
    pub async fn release(self) -> Result<()> {
        self.remove_owner(self.path.clone()).await
    }

    /// Fail before starting protected work after the fixed deadline has passed.
    /// Callers of this hook must also bound their submitted storage operations.
    pub fn check_valid(&self) -> Result<()> {
        self.deadline.check()
    }

    pub(crate) fn deadline(&self) -> MaintenanceDeadline {
        self.deadline
    }

    async fn remove_owner(&self, path: Path) -> Result<()> {
        if let Some(delete) = &self.delete {
            return delete(self.store.clone(), path).await;
        }
        match self.store.delete(&path).await {
            Err(error) if error.is_not_found() => Ok(()),
            result => result,
        }
    }
}

async fn maintenance_tickets(
    fence: &SnapshotMaintenanceLock,
    directory: &Path,
) -> Result<Vec<(String, u64)>> {
    let store = &fence.store;
    let names = store.read_dir(directory.clone()).await?;
    let mut tickets = Vec::with_capacity(names.len());
    for name in names {
        // Local put_if_absent uses temporary files before publishing an owner.
        if uuid::Uuid::parse_str(&name).is_err() {
            continue;
        }
        let path = directory.clone().join(name.as_str());
        let bytes = match store.inner.get(&path).await {
            Ok(result) => result.bytes().await?,
            Err(StoreError::NotFound { .. }) => continue,
            Err(error) => return Err(error.into()),
        };
        let record: MaintenanceOwner = serde_json::from_slice(&bytes).map_err(|error| {
            Error::corrupt_file(
                path.clone(),
                format!("Invalid snapshot maintenance owner record: {error}"),
            )
        })?;
        if record.expires_at <= utc_now() {
            if let Err(error) = fence.remove_owner(path.clone()).await {
                warn!(%path, %error, "Failed to remove expired maintenance owner; ignoring its expired ticket");
            }
            continue;
        }
        tickets.push((name, record.ticket));
    }
    Ok(tickets)
}

/// A durable protected tag for one exact snapshot identity, with explicit release.
///
/// Acquire before starting scans or exporting references. Keep the token until all workers, retries,
/// outstanding reads, and lazy results have finished. Dropping or serializing the
/// token does not release it; a coordinator can recover orphaned tags after a crash.
/// Acquisition is idempotent for the same snapshot identity and globally unique `retention_id`.
/// Reusing that ID for independent queries would give them shared release ownership.
/// Persist that ID and the encoded snapshot identity before acquiring, then persist the token
/// before issuing work. Record release intent before releasing, so recovery can
/// retry release instead of accidentally acquiring protection again.
///
/// This implementation protects primary-storage files and objects owned by this
/// dataset. Snapshots with foreign dataset bases require retention of their owners
/// and are rejected. External objects still require their own immutability/lifetime
/// contract. All maintenance clients must participate in this protocol.
///
/// ```
/// # use lance::{Dataset, Result};
/// # use lance::dataset::{snapshot::SnapshotIdentity, retention::SnapshotRetention};
/// # async fn example(dataset: &Dataset) -> Result<()> {
/// let context = SnapshotIdentity::from_dataset(dataset, "table-incarnation").await?;
/// let retention = SnapshotRetention::acquire(dataset, context, "unique-query-id").await?;
/// let transferred = SnapshotRetention::decode(&retention.encode())?;
/// // Issue references and wait for every consumer to finish before release.
/// transferred.release(dataset).await?;
/// transferred.release(dataset).await?; // Idempotent.
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct SnapshotRetention {
    snapshot: SnapshotIdentity,
    tag_name: String,
}

impl SnapshotRetention {
    /// Protect the exact snapshot and verify its original manifest under the cleanup fence.
    /// An expired/deleted snapshot fails before a usable retention token is returned.
    pub async fn acquire(
        dataset: &Dataset,
        snapshot: SnapshotIdentity,
        retention_id: impl AsRef<str>,
    ) -> Result<Self> {
        snapshot.validate_dataset(dataset)?;
        if retention_id.as_ref().is_empty() {
            return Err(Error::invalid_input("retention_id must not be empty"));
        }
        for base in dataset.manifest().base_paths.values() {
            if base.is_dataset_root && canonical_uri(&base.path)? != snapshot.table_uri() {
                return Err(Error::not_supported(format!(
                    "Snapshot retention requires separate protection for foreign dataset base {}",
                    base.path,
                )));
            }
        }
        let mut identity = blake3::Hasher::new();
        identity.update(snapshot.snapshot_id());
        identity.update(retention_id.as_ref().as_bytes());
        let retention = Self {
            snapshot,
            tag_name: format!("{TAG_PREFIX}{}", identity.finalize().to_hex()),
        };
        let fence = SnapshotMaintenanceLock::acquire(dataset).await?;
        let result = retention.acquire_fenced(dataset, &fence).await;
        // Acquisition never deletes files. An unsuccessful/uncertain tag write can
        // only leave an extra GC root, so its fence can be released on error too.
        fence.release().await?;
        result?;
        Ok(retention)
    }

    async fn acquire_fenced(
        &self,
        dataset: &Dataset,
        fence: &SnapshotMaintenanceLock,
    ) -> Result<()> {
        fence.check_valid()?;
        let manifest_path = dataset.manifest_location().path.clone();
        let location = dataset
            .commit_handler
            .resolve_version_location(
                &dataset.base,
                self.snapshot.version(),
                &dataset.object_store.inner,
            )
            .await?;
        if location.path != manifest_path
            || location.identity != dataset.manifest_location().identity
        {
            return Err(Error::invalid_input(
                "Retained snapshot commit identity changed before acquisition",
            ));
        }
        if dataset
            .object_store
            .exists(&retirement_path(dataset, &manifest_path)?)
            .await?
        {
            return Err(Error::invalid_input(format!(
                "Snapshot manifest {manifest_path} was selected by cleanup; retention cannot be acquired",
            )));
        }
        let observed =
            SnapshotIdentity::from_dataset(dataset, self.snapshot.table_identity()).await?;
        if observed.serialized_manifest() != self.snapshot.serialized_manifest() {
            return Err(Error::invalid_input(format!(
                "Snapshot manifest {manifest_path} changed before retention acquisition",
            )));
        }
        let path = tag_path(&dataset.refs.root()?.path, &self.tag_name);
        let now = utc_now();
        let contents = TagContents {
            branch: self.snapshot.branch().map(str::to_owned),
            version: self.snapshot.version(),
            created_at: Some(now),
            updated_at: Some(now),
            manifest_size: usize::try_from(dataset.object_store.size(&manifest_path).await?)
                .map_err(|_| Error::invalid_input("Retained manifest size exceeds usize"))?,
            metadata: HashMap::from([(SNAPSHOT_ID_KEY.to_string(), self.snapshot_id_hex())]),
        };
        fence.check_valid()?;
        let result = match dataset
            .object_store
            .put_if_absent(&path, serde_json::to_vec(&contents)?.into())
            .await
        {
            Ok(()) => Ok(()),
            Err(StoreError::AlreadyExists { .. } | StoreError::Precondition { .. }) => {
                self.validate_tag(&TagContents::from_path(&path, &dataset.object_store).await?)
            }
            Err(error) => Err(error.into()),
        };
        fence.check_valid()?;
        result
    }

    fn snapshot_id_hex(&self) -> String {
        blake3::Hash::from(*self.snapshot.snapshot_id())
            .to_hex()
            .to_string()
    }

    fn validate_tag(&self, tag: &TagContents) -> Result<()> {
        if tag.branch.as_deref() != self.snapshot.branch()
            || tag.version != self.snapshot.version()
            || tag.metadata.get(SNAPSHOT_ID_KEY) != Some(&self.snapshot_id_hex())
        {
            return Err(Error::invalid_input(format!(
                "Retention tag {} does not match its exact snapshot identity",
                self.tag_name,
            )));
        }
        Ok(())
    }

    /// The general snapshot identity protected by this token.
    pub fn snapshot(&self) -> &SnapshotIdentity {
        &self.snapshot
    }

    /// Protected tag name, useful for query diagnostics and orphan recovery.
    pub fn tag_name(&self) -> &str {
        &self.tag_name
    }

    /// Serialize release ownership for the query coordinator. Contains no credentials.
    pub fn encode(&self) -> Vec<u8> {
        pb::SnapshotRetention {
            protocol_version: PROTOCOL_VERSION,
            snapshot: self.snapshot.encode().into(),
            tag_name: self.tag_name.clone(),
        }
        .encode_to_vec()
    }

    /// Validate a token without acquiring or renewing protection.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let token = pb::SnapshotRetention::decode(bytes)
            .map_err(|error| Error::invalid_input(format!("Invalid retention token: {error}")))?;
        if token.protocol_version != PROTOCOL_VERSION {
            return Err(Error::invalid_input(format!(
                "Unsupported retention protocol version {}",
                token.protocol_version,
            )));
        }
        Self::from_parts(SnapshotIdentity::decode(&token.snapshot)?, token.tag_name)
    }

    pub(crate) fn from_parts(snapshot: SnapshotIdentity, tag_name: String) -> Result<Self> {
        let suffix = tag_name
            .strip_prefix(TAG_PREFIX)
            .ok_or_else(|| Error::invalid_input("Invalid retention tag prefix"))?;
        if suffix.len() != 64 || !suffix.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(Error::invalid_input("Invalid retention tag digest"));
        }
        Ok(Self { snapshot, tag_name })
    }

    /// Idempotently release this query's GC root after its last consumer has stopped.
    /// `dataset` may be a later snapshot, but must use the same table and credentials.
    pub async fn release(&self, dataset: &Dataset) -> Result<()> {
        if canonical_uri(&dataset.refs.root()?.uri)? != self.snapshot.table_uri() {
            return Err(Error::invalid_input(
                "Retention release dataset has a different table URI",
            ));
        }
        let fence = SnapshotMaintenanceLock::acquire(dataset).await?;
        let path = tag_path(&dataset.refs.root()?.path, &self.tag_name);
        let result = async {
            match TagContents::from_path(&path, &dataset.object_store).await {
                Ok(tag) => {
                    self.validate_tag(&tag)?;
                    fence.check_valid()?;
                    dataset.object_store.delete(&path).await
                }
                Err(error) if error.is_not_found() => Ok(()),
                Err(error) => Err(error),
            }
        }
        .await;
        let result = result.and_then(|()| fence.check_valid());
        fence.release().await?;
        result
    }
}

// A failed manifest deletion can leave its payloads partly deleted. Persist this
// barrier before deleting any payload, so a later acquire cannot tag that manifest.
pub(crate) async fn mark_retiring(dataset: &Dataset, manifest: &Path) -> Result<()> {
    dataset
        .object_store
        .put(
            &retirement_path(dataset, manifest)?,
            manifest.as_ref().as_bytes(),
        )
        .await?;
    Ok(())
}

pub(crate) async fn clear_retiring(
    dataset: &Dataset,
    manifest: &Path,
    delete: &MaintenanceDelete,
) -> Result<()> {
    let path = retirement_path(dataset, manifest)?;
    if let Err(error) = delete(dataset.object_store.clone(), path.clone()).await
        && !error.is_not_found()
    {
        warn!(%path, %error, "Failed to remove obsolete snapshot retirement marker");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::{sync::Mutex, time::Duration};

    use arrow_array::{RecordBatchIterator, record_batch};
    use futures::TryStreamExt;
    use lance_core::utils::testing::{ProxyObjectStore, ProxyObjectStorePolicy};
    use lance_file::version::LanceFileVersion;
    use lance_io::object_store::{ObjectStoreParams, WrappingObjectStore};
    use mock_instant::thread_local::MockClock;
    use object_store::throttle::{ThrottleConfig, ThrottledStore};
    use rstest::rstest;

    use crate::dataset::{WriteMode, WriteParams, cleanup::CleanupPolicyBuilder};

    #[derive(Debug)]
    struct MaintenanceTestStore {
        policy: Arc<Mutex<ProxyObjectStorePolicy>>,
    }

    impl WrappingObjectStore for MaintenanceTestStore {
        fn wrap(
            &self,
            _prefix: &str,
            store: Arc<dyn object_store::ObjectStore>,
        ) -> Arc<dyn object_store::ObjectStore> {
            Arc::new(ThrottledStore::new(
                Arc::new(ProxyObjectStore::new(store, self.policy.clone())),
                ThrottleConfig {
                    wait_put_per_call: Duration::from_millis(1),
                    wait_get_per_call: Duration::from_millis(1),
                    ..Default::default()
                },
            ))
        }

        fn wrap_paginated(
            &self,
            _prefix: &str,
            _store: Arc<dyn object_store::list::PaginatedListStore>,
        ) -> Option<Arc<dyn object_store::list::PaginatedListStore>> {
            None
        }
    }

    #[cfg_attr(coverage, coverage(off))]
    async fn maintenance_dataset() -> (Dataset, Arc<Mutex<ProxyObjectStorePolicy>>) {
        let mut policy = ProxyObjectStorePolicy::new();
        policy.set_obj_meta_policy(
            "mock_clock",
            Arc::new(|_, mut meta| {
                meta.last_modified = utc_now();
                Ok(meta)
            }),
        );
        let policy = Arc::new(Mutex::new(policy));
        let batch = record_batch!(("value", Int32, [1, 2])).unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 1,
                data_storage_version: Some(LanceFileVersion::V2_2),
                store_params: Some(ObjectStoreParams {
                    object_store_wrapper: Some(Arc::new(MaintenanceTestStore {
                        policy: policy.clone(),
                    })),
                    ..Default::default()
                }),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        (dataset, policy)
    }

    #[cfg_attr(coverage, coverage(off))]
    async fn overwrite_maintenance_dataset(dataset: &Dataset) -> Dataset {
        let batch = record_batch!(("value", Int32, [3, 4])).unwrap();
        Dataset::write(
            RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
            Arc::new(dataset.clone()),
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn maintenance_contention_never_admits_two_owners() {
        let (dataset, _) = maintenance_dataset().await;
        let (first, second) = tokio::join!(
            SnapshotMaintenanceLock::acquire(&dataset),
            SnapshotMaintenanceLock::acquire(&dataset)
        );
        assert!(first.is_err() || second.is_err());
        for result in [first, second] {
            match result {
                Ok(fence) => {
                    let error = SnapshotMaintenanceLock::acquire(&dataset)
                        .await
                        .err()
                        .unwrap();
                    assert!(matches!(error, Error::RefConflict { .. }));
                    assert!(error.to_string().contains("maintenance lock"));
                    fence.release().await.unwrap();
                }
                Err(error) => {
                    assert!(matches!(error, Error::RefConflict { .. }));
                    assert!(error.to_string().contains("maintenance lock"));
                }
            }
        }
        SnapshotMaintenanceLock::acquire(&dataset)
            .await
            .unwrap()
            .release()
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn maintenance_release_retry_cannot_remove_new_owner() {
        let (dataset, _) = maintenance_dataset().await;
        let first = SnapshotMaintenanceLock::acquire(&dataset).await.unwrap();
        let old_path = first.path.clone();
        first.release().await.unwrap();
        let second = SnapshotMaintenanceLock::acquire(&dataset).await.unwrap();
        assert_ne!(old_path, second.path);
        // Repeat the previous owner's DELETE after a new owner has entered.
        let _ = dataset.object_store.delete(&old_path).await;
        let error = SnapshotMaintenanceLock::acquire(&dataset)
            .await
            .err()
            .unwrap();
        assert!(matches!(error, Error::RefConflict { .. }));
        assert!(error.to_string().contains("maintenance lock"));
        second.release().await.unwrap();
        let choosing = retention_dir(&dataset)
            .unwrap()
            .join("maintenance")
            .join(uuid::Uuid::new_v4().to_string());
        let record = MaintenanceOwner {
            ticket: 0,
            expires_at: utc_now() + TimeDelta::hours(120),
        };
        dataset
            .object_store
            .put(&choosing, &serde_json::to_vec(&record).unwrap())
            .await
            .unwrap();
        let error = SnapshotMaintenanceLock::acquire(&dataset)
            .await
            .err()
            .unwrap();
        assert!(matches!(error, Error::RefConflict { .. }));
        assert!(error.to_string().contains("maintenance lock"));
        dataset.object_store.delete(&choosing).await.unwrap();
        dataset
            .object_store
            .put(&choosing, b"invalid")
            .await
            .unwrap();
        let error = SnapshotMaintenanceLock::acquire(&dataset)
            .await
            .err()
            .unwrap();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(
            error
                .to_string()
                .contains("Invalid snapshot maintenance owner record")
        );
        dataset.object_store.delete(&choosing).await.unwrap();
        SnapshotMaintenanceLock::acquire(&dataset)
            .await
            .unwrap()
            .release()
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn partial_cleanup_prevents_retention_acquisition() {
        let (dataset, policy) = maintenance_dataset().await;
        let source = SnapshotIdentity::from_dataset(&dataset, "catalog/partial")
            .await
            .unwrap();
        let old_manifest = dataset.manifest_location().path.clone();
        let failed_manifest = old_manifest.clone();
        policy.lock().unwrap().set_before_policy(
            "fail_manifest_delete",
            Arc::new(move |method, path| {
                if method == "delete" && path == &failed_manifest {
                    Err(Error::io("injected manifest deletion failure"))
                } else {
                    Ok(())
                }
            }),
        );
        let latest = overwrite_maintenance_dataset(&dataset).await;
        let cleanup = CleanupPolicyBuilder::default()
            .before_timestamp(utc_now() + chrono::Duration::hours(1))
            .build();
        let removed = latest.cleanup(cleanup.clone()).execute().await.unwrap();
        assert_eq!(removed.failed_deletes, 1);
        assert_eq!(removed.data_files_removed, 2);
        assert!(dataset.object_store.exists(&old_manifest).await.unwrap());
        let error = SnapshotRetention::acquire(&dataset, source, "after-partial-delete")
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("selected by cleanup"));
        assert!(dataset.tags().list().await.unwrap().is_empty());
        policy
            .lock()
            .unwrap()
            .clear_before_policy("fail_manifest_delete");
        latest.cleanup(cleanup).execute().await.unwrap();
        assert!(!dataset.object_store.exists(&old_manifest).await.unwrap());
        assert!(
            !dataset
                .object_store
                .exists(&retirement_path(&dataset, &old_manifest).unwrap())
                .await
                .unwrap()
        );
    }

    #[rstest]
    #[case::acquisition_intent(false)]
    #[case::saved_token(true)]
    #[tokio::test]
    async fn recovery_preserves_roots_until_explicit_release(#[case] has_saved_token: bool) {
        let (dataset, _) = maintenance_dataset().await;
        let context = SnapshotIdentity::from_dataset(&dataset, "catalog/recovery")
            .await
            .unwrap();
        // The coordinator records acquisition intent before the storage operation.
        let saved_context = context.encode();
        let query_id = "query-recovery";
        let retention = SnapshotRetention::acquire(&dataset, context.clone(), query_id)
            .await
            .unwrap();
        let tag_name = retention.tag_name().to_string();
        let saved_token = has_saved_token.then(|| retention.encode());
        drop(retention);
        let latest = overwrite_maintenance_dataset(&dataset).await;
        let policy = CleanupPolicyBuilder::default()
            .before_timestamp(utc_now() + chrono::Duration::hours(1))
            .error_if_tagged_old_versions(false)
            .build();

        let fence = SnapshotMaintenanceLock::acquire(&latest).await.unwrap();
        let abandoned_owner = fence.path.clone();
        drop(fence);
        let error = SnapshotRetention::acquire(&dataset, context, query_id)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::RefConflict { .. }));
        assert!(error.to_string().contains(abandoned_owner.as_ref()));
        let error = latest.cleanup(policy.clone()).execute().await.unwrap_err();
        assert!(matches!(error, Error::RefConflict { .. }));
        assert!(error.to_string().contains(abandoned_owner.as_ref()));

        // This owner has stopped and has no outstanding storage calls. Recovery
        // removes its fence explicitly; the durable query tag must survive it.
        dataset.object_store.delete(&abandoned_owner).await.unwrap();
        assert!(dataset.tags().list().await.unwrap().contains_key(&tag_name));
        let protected = latest.cleanup(policy.clone()).execute().await.unwrap();
        assert_eq!(protected.old_versions, 0);
        assert_eq!(protected.data_files_removed, 0);
        let batches = dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(
            batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
            2
        );
        let restored = SnapshotIdentity::decode(&saved_context).unwrap();
        assert_eq!(restored.version(), dataset.version().version);
        let recovered = match saved_token {
            Some(token) => SnapshotRetention::decode(&token).unwrap(),
            None => SnapshotRetention::acquire(
                &dataset,
                SnapshotIdentity::decode(&saved_context).unwrap(),
                query_id,
            )
            .await
            .unwrap(),
        };
        assert_eq!(recovered.tag_name(), tag_name);
        assert_eq!(dataset.tags().list().await.unwrap().len(), 1);

        // A durable release record retains the token until acknowledgement. A
        // lost response is recovered by repeating release, without reacquiring.
        let release_token = recovered.encode();
        recovered.release(&latest).await.unwrap();
        drop(recovered);
        let pending_release = SnapshotRetention::decode(&release_token).unwrap();
        pending_release.release(&latest).await.unwrap();
        assert!(latest.tags().list().await.unwrap().is_empty());
        let removed = latest.cleanup(policy).execute().await.unwrap();
        assert_eq!(removed.old_versions, 1);
        assert_eq!(removed.data_files_removed, 2);
        assert!(
            !dataset
                .object_store
                .exists(&dataset.manifest_location().path)
                .await
                .unwrap()
        );
        pending_release.release(&latest).await.unwrap();
        assert!(latest.tags().list().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn release_validates_exact_source_and_token_protocol() {
        let (dataset, _) = maintenance_dataset().await;
        let source = SnapshotIdentity::from_dataset(&dataset, "catalog/token")
            .await
            .unwrap();
        let retention = SnapshotRetention::acquire(&dataset, source, "query")
            .await
            .unwrap();
        let mut token = pb::SnapshotRetention::decode(retention.encode().as_slice()).unwrap();
        token.protocol_version = 2;
        let error = SnapshotRetention::decode(&token.encode_to_vec()).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("protocol version 2"));
        token.protocol_version = PROTOCOL_VERSION;
        token.snapshot = SnapshotIdentity::from_dataset(&dataset, "catalog/different-incarnation")
            .await
            .unwrap()
            .encode()
            .into();
        let wrong_source = SnapshotRetention::decode(&token.encode_to_vec()).unwrap();
        let error = wrong_source.release(&dataset).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("does not match its exact snapshot identity")
        );
        assert_eq!(dataset.tags().list().await.unwrap().len(), 1);
        retention.release(&dataset).await.unwrap();
        retention.release(&dataset).await.unwrap();
        assert!(dataset.tags().list().await.unwrap().is_empty());
    }

    #[rstest]
    #[case::default_lifetime(None, 120)]
    #[case::configured_lifetime(Some("2"), 2)]
    #[tokio::test]
    async fn maintenance_expiration_uses_recorded_owner_deadline(
        #[case] configured: Option<&str>,
        #[case] hours: u64,
    ) {
        MockClock::set_system_time(Duration::from_secs(3600));
        let (mut dataset, _) = maintenance_dataset().await;
        if let Some(configured) = configured {
            dataset
                .update_config([(LOCK_EXPIRATION_HOURS_KEY, configured)])
                .await
                .unwrap();
        }
        let first = SnapshotMaintenanceLock::acquire(&dataset).await.unwrap();
        let first_path = first.path.clone();
        let expected_expiry = utc_now() + TimeDelta::hours(hours as i64);
        let bytes = dataset
            .object_store
            .inner
            .get(&first_path)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let record: MaintenanceOwner = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(record.expires_at, expected_expiry);
        // A contender's shorter setting must not shorten the existing owner's lifetime.
        dataset
            .update_config([(LOCK_EXPIRATION_HOURS_KEY, "1")])
            .await
            .unwrap();
        MockClock::advance_system_time(Duration::from_secs(3600));
        let error = SnapshotMaintenanceLock::acquire(&dataset)
            .await
            .err()
            .unwrap();
        assert!(matches!(error, Error::RefConflict { .. }));
        assert!(error.to_string().contains(first_path.as_ref()));
        assert!(first.check_valid().is_ok());
        MockClock::advance_system_time(Duration::from_secs((hours - 1) * 3600));
        let error = first.check_valid().unwrap_err();
        assert!(matches!(error, Error::RefConflict { .. }));
        assert!(error.to_string().contains("expired"));
        let successor = SnapshotMaintenanceLock::acquire(&dataset).await.unwrap();
        assert!(!dataset.object_store.exists(&first_path).await.unwrap());
        // A delayed release for the expired owner cannot remove its successor.
        first.release().await.unwrap();
        assert!(dataset.object_store.exists(&successor.path).await.unwrap());
        successor.release().await.unwrap();
    }

    #[rstest]
    #[case::choosing(0)]
    #[case::published(7)]
    #[tokio::test]
    async fn expired_maintenance_owners_do_not_block_query_tags(#[case] ticket: u64) {
        MockClock::set_system_time(Duration::from_secs(3600));
        let (dataset, _) = maintenance_dataset().await;
        let snapshot = SnapshotIdentity::from_dataset(&dataset, "catalog/plain-data")
            .await
            .unwrap();
        let retention = SnapshotRetention::acquire(&dataset, snapshot, "query")
            .await
            .unwrap();
        let path = retention_dir(&dataset)
            .unwrap()
            .join("maintenance")
            .join(uuid::Uuid::new_v4().to_string());
        let record = MaintenanceOwner {
            ticket,
            expires_at: utc_now() + TimeDelta::hours(120),
        };
        dataset
            .object_store
            .put(&path, &serde_json::to_vec(&record).unwrap())
            .await
            .unwrap();
        MockClock::advance_system_time(Duration::from_secs(120 * 3600));
        let fence = SnapshotMaintenanceLock::acquire(&dataset).await.unwrap();
        assert!(!dataset.object_store.exists(&path).await.unwrap());
        assert!(
            dataset
                .tags()
                .list()
                .await
                .unwrap()
                .contains_key(retention.tag_name())
        );
        fence.release().await.unwrap();
        retention.release(&dataset).await.unwrap();
    }

    #[rstest]
    #[case::zero("0")]
    #[case::negative("-1")]
    #[case::fractional("1.5")]
    #[case::empty("")]
    #[case::overflow("18446744073709551615")]
    #[tokio::test]
    async fn invalid_maintenance_expiration_fails_before_owner_creation(#[case] value: &str) {
        let (mut dataset, _) = maintenance_dataset().await;
        dataset
            .update_config([(LOCK_EXPIRATION_HOURS_KEY, value)])
            .await
            .unwrap();
        let error = SnapshotMaintenanceLock::acquire(&dataset)
            .await
            .err()
            .unwrap();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(LOCK_EXPIRATION_HOURS_KEY));
        assert!(
            dataset
                .object_store
                .read_dir(retention_dir(&dataset).unwrap().join("maintenance"))
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn cleanup_stops_when_owner_expires_during_inspection() {
        MockClock::set_system_time(Duration::from_secs(3600));
        let (dataset, policy) = maintenance_dataset().await;
        let latest = overwrite_maintenance_dataset(&dataset).await;
        let manifest = dataset.manifest_location().path.clone();
        policy.lock().unwrap().set_before_policy(
            "expire_during_inspection",
            Arc::new(move |operation, path| {
                if matches!(operation, "get_opts" | "get_ranges") && path == &manifest {
                    MockClock::advance_system_time(Duration::from_secs(120 * 3600));
                }
                Ok(())
            }),
        );
        let cleanup = CleanupPolicyBuilder::default()
            .before_timestamp(utc_now() + TimeDelta::hours(1))
            .build();
        let error = latest.cleanup(cleanup).execute().await.unwrap_err();
        assert!(matches!(error, Error::RefConflict { .. }));
        assert!(error.to_string().contains("expired"));
        policy
            .lock()
            .unwrap()
            .clear_before_policy("expire_during_inspection");
        assert!(
            dataset
                .object_store
                .exists(&dataset.manifest_location().path)
                .await
                .unwrap()
        );
        assert!(
            !dataset
                .object_store
                .exists(&retirement_path(&dataset, &dataset.manifest_location().path).unwrap())
                .await
                .unwrap()
        );
        SnapshotMaintenanceLock::acquire(&latest)
            .await
            .unwrap()
            .release()
            .await
            .unwrap();
    }
}
