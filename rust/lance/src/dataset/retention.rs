// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Explicit-release snapshot retention coordinated with dataset cleanup.
//!
//! All cleanup clients must use this coordination protocol. Older maintenance
//! clients, forced branch/table deletion, and deletion outside Lance are excluded.
//! See the [deployment contract](https://lance.org/guide/blob_references/).

use std::collections::HashMap;
use std::sync::Arc;

use futures::future::BoxFuture;
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use object_store::{Error as StoreError, ObjectStoreExt, path::Path};
use prost::Message;
use tracing::warn;

use crate::Dataset;
use crate::blob::reference::{BlobSourceContext, canonical_uri, pb};
use crate::dataset::refs::{TagContents, tag_path};
use crate::utils::temporal::utc_now;

const PROTOCOL_VERSION: u32 = 1;
const TAG_PREFIX: &str = "lance-query-";
const SOURCE_ID_KEY: &str = "lance.snapshot_retention.source_id";

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
/// There is no expiry or lock stealing. Cancellation, a crash, or an uncertain
/// failure to release can leave the fence in place. An administrator may remove an
/// abandoned owner's object only after its operation and storage requests have
/// stopped. Do not remove retirement markers for manifests still on storage.
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
        // Zero announces that this owner is choosing a ticket. Every attempt has
        // its own key: a retried DELETE must never remove a subsequent owner's lock.
        dataset
            .object_store
            .put_if_absent(&path, "0".into())
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
        };
        let result = async {
            let ticket = maintenance_tickets(&fence.store, &directory).await?
                .iter().map(|(_, ticket)| *ticket).max().unwrap_or(0)
                .checked_add(1).ok_or_else(|| Error::invalid_input("Snapshot maintenance ticket overflow"))?;
            fence.store.put(&fence.path, ticket.to_string().as_bytes()).await?;
            // Publishing before checking ensures a later chooser gets a higher
            // ticket. Concurrent choosers break ties by their unique owner IDs.
            for (other_owner, other_ticket) in maintenance_tickets(&fence.store, &directory).await? {
                if other_owner != owner && (other_ticket == 0 || (other_ticket, &other_owner) < (ticket, &owner)) {
                    return Err(Error::RefConflict { message: format!(
                        "Snapshot maintenance lock {} is held; retry after the operation finishes. \
                         Recover an abandoned lock only after its owner and storage requests have stopped",
                        directory.clone().join(other_owner),
                    ) });
                }
            }
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
        if let Some(delete) = self.delete {
            return delete(self.store, self.path).await;
        }
        match self.store.delete(&self.path).await {
            Err(error) if error.is_not_found() => Ok(()),
            result => result,
        }
    }
}

async fn maintenance_tickets(store: &ObjectStore, directory: &Path) -> Result<Vec<(String, u64)>> {
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
        let ticket = std::str::from_utf8(&bytes)
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .ok_or_else(|| {
                Error::corrupt_file(path.clone(), format!("Invalid snapshot maintenance ticket: expected an unsigned integer, got {} bytes", bytes.len()))
            })?;
        tickets.push((name, ticket));
    }
    Ok(tickets)
}

/// A durable protected tag for one exact source context, with explicit release.
///
/// Acquire before exporting references. Keep the token until all workers, retries,
/// outstanding reads, and lazy results have finished. Dropping or serializing the
/// token does not release it; a coordinator can recover orphaned tags after a crash.
/// Acquisition is idempotent for the same context and globally unique `retention_id`.
/// Reusing that ID for independent queries would give them shared release ownership.
///
/// This implementation protects primary-storage files and objects owned by this
/// dataset. Snapshots with foreign dataset bases require retention of their owners
/// and are rejected. External objects still require their own immutability/lifetime
/// contract. All maintenance clients must participate in this protocol.
///
/// ```
/// # use lance::{Dataset, Result};
/// # use lance::blob::reference::{BlobSourceContext, BlobSnapshotRetention};
/// # async fn example(dataset: &Dataset) -> Result<()> {
/// let context = BlobSourceContext::from_dataset(dataset, "table-incarnation", "storage").await?;
/// let retention = BlobSnapshotRetention::acquire(dataset, context, "unique-query-id").await?;
/// let transferred = BlobSnapshotRetention::decode(&retention.encode())?;
/// // Issue references and wait for every consumer to finish before release.
/// transferred.release(dataset).await?;
/// transferred.release(dataset).await?; // Idempotent.
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct BlobSnapshotRetention {
    context: BlobSourceContext,
    tag_name: String,
}

impl BlobSnapshotRetention {
    /// Protect the exact context and verify its original manifest under the cleanup fence.
    /// An expired/deleted snapshot fails before a usable retention token is returned.
    pub async fn acquire(
        dataset: &Dataset,
        context: BlobSourceContext,
        retention_id: impl AsRef<str>,
    ) -> Result<Self> {
        context.validate_dataset(dataset)?;
        if retention_id.as_ref().is_empty() {
            return Err(Error::invalid_input("retention_id must not be empty"));
        }
        for base in dataset.manifest().base_paths.values() {
            if base.is_dataset_root && canonical_uri(&base.path)? != context.table_uri() {
                return Err(Error::not_supported(format!(
                    "Snapshot retention requires separate protection for foreign dataset base {}",
                    base.path,
                )));
            }
        }
        let mut identity = blake3::Hasher::new();
        identity.update(context.source_id());
        identity.update(retention_id.as_ref().as_bytes());
        let retention = Self {
            context,
            tag_name: format!("{TAG_PREFIX}{}", identity.finalize().to_hex()),
        };
        let fence = SnapshotMaintenanceLock::acquire(dataset).await?;
        let result = retention.acquire_fenced(dataset).await;
        // Acquisition never deletes files. An unsuccessful/uncertain tag write can
        // only leave an extra GC root, so its fence can be released on error too.
        fence.release().await?;
        result?;
        Ok(retention)
    }

    async fn acquire_fenced(&self, dataset: &Dataset) -> Result<()> {
        let manifest_path = dataset.manifest_location().path.clone();
        let location = dataset
            .commit_handler
            .resolve_version_location(
                &dataset.base,
                self.context.version(),
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
        let observed = BlobSourceContext::from_dataset(
            dataset,
            self.context.table_identity(),
            self.context.access_context_key(),
        )
        .await?;
        if observed.serialized_manifest() != self.context.serialized_manifest() {
            return Err(Error::invalid_input(format!(
                "Snapshot manifest {manifest_path} changed before retention acquisition",
            )));
        }
        let path = tag_path(&dataset.refs.root()?.path, &self.tag_name);
        let now = utc_now();
        let contents = TagContents {
            branch: self.context.branch().map(str::to_owned),
            version: self.context.version(),
            created_at: Some(now),
            updated_at: Some(now),
            manifest_size: usize::try_from(dataset.object_store.size(&manifest_path).await?)
                .map_err(|_| Error::invalid_input("Retained manifest size exceeds usize"))?,
            metadata: HashMap::from([(SOURCE_ID_KEY.to_string(), self.source_id())]),
        };
        match dataset
            .object_store
            .put_if_absent(&path, serde_json::to_vec(&contents)?.into())
            .await
        {
            Ok(()) => Ok(()),
            Err(StoreError::AlreadyExists { .. } | StoreError::Precondition { .. }) => {
                self.validate_tag(&TagContents::from_path(&path, &dataset.object_store).await?)
            }
            Err(error) => Err(error.into()),
        }
    }

    fn source_id(&self) -> String {
        blake3::Hash::from(*self.context.source_id())
            .to_hex()
            .to_string()
    }

    fn validate_tag(&self, tag: &TagContents) -> Result<()> {
        if tag.branch.as_deref() != self.context.branch()
            || tag.version != self.context.version()
            || tag.metadata.get(SOURCE_ID_KEY) != Some(&self.source_id())
        {
            return Err(Error::invalid_input(format!(
                "Retention tag {} does not match its exact source context",
                self.tag_name,
            )));
        }
        Ok(())
    }

    /// The source context protected by this token. Forward it with captured batches.
    pub fn context(&self) -> &BlobSourceContext {
        &self.context
    }

    /// Protected tag name, useful for query diagnostics and orphan recovery.
    pub fn tag_name(&self) -> &str {
        &self.tag_name
    }

    /// Serialize release ownership for the query coordinator. Contains no credentials.
    pub fn encode(&self) -> Vec<u8> {
        pb::SnapshotRetention {
            protocol_version: PROTOCOL_VERSION,
            context: self.context.encode().into(),
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
        let suffix = token
            .tag_name
            .strip_prefix(TAG_PREFIX)
            .ok_or_else(|| Error::invalid_input("Invalid retention tag prefix"))?;
        if suffix.len() != 64 || !suffix.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(Error::invalid_input("Invalid retention tag digest"));
        }
        Ok(Self {
            context: BlobSourceContext::decode(&token.context)?,
            tag_name: token.tag_name,
        })
    }

    /// Idempotently release this query's GC root after its last consumer has stopped.
    /// `dataset` may be a later snapshot, but must use the same table and credentials.
    pub async fn release(&self, dataset: &Dataset) -> Result<()> {
        if canonical_uri(&dataset.refs.root()?.uri)? != self.context.table_uri() {
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
                    dataset.object_store.delete(&path).await
                }
                Err(error) if error.is_not_found() => Ok(()),
                Err(error) => Err(error),
            }
        }
        .await;
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
    use lance_core::utils::testing::{ProxyObjectStore, ProxyObjectStorePolicy};
    use lance_file::version::LanceFileVersion;
    use lance_io::object_store::{ObjectStoreParams, WrappingObjectStore};
    use object_store::throttle::{ThrottleConfig, ThrottledStore};

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
        dataset.object_store.put(&choosing, b"0").await.unwrap();
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
        assert!(error.to_string().contains("expected an unsigned integer"));
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
        let source = BlobSourceContext::from_dataset(&dataset, "catalog/partial", "worker")
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
        let batch = record_batch!(("value", Int32, [3, 4])).unwrap();
        let latest = Dataset::write(
            RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
            Arc::new(dataset.clone()),
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let cleanup = CleanupPolicyBuilder::default()
            .before_timestamp(utc_now() + chrono::Duration::hours(1))
            .build();
        let removed = latest.cleanup(cleanup.clone()).execute().await.unwrap();
        assert_eq!(removed.failed_deletes, 1);
        assert_eq!(removed.data_files_removed, 2);
        assert!(dataset.object_store.exists(&old_manifest).await.unwrap());
        let error = BlobSnapshotRetention::acquire(&dataset, source, "after-partial-delete")
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

    #[tokio::test]
    async fn release_validates_exact_source_and_token_protocol() {
        let (dataset, _) = maintenance_dataset().await;
        let source = BlobSourceContext::from_dataset(&dataset, "catalog/token", "worker")
            .await
            .unwrap();
        let retention = BlobSnapshotRetention::acquire(&dataset, source, "query")
            .await
            .unwrap();
        let mut token = pb::SnapshotRetention::decode(retention.encode().as_slice()).unwrap();
        token.protocol_version = 2;
        let error = BlobSnapshotRetention::decode(&token.encode_to_vec()).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("protocol version 2"));
        token.protocol_version = PROTOCOL_VERSION;
        token.context =
            BlobSourceContext::from_dataset(&dataset, "catalog/different-incarnation", "worker")
                .await
                .unwrap()
                .encode()
                .into();
        let wrong_source = BlobSnapshotRetention::decode(&token.encode_to_vec()).unwrap();
        let error = wrong_source.release(&dataset).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("does not match its exact source context")
        );
        assert_eq!(dataset.tags().list().await.unwrap().len(), 1);
        retention.release(&dataset).await.unwrap();
        retention.release(&dataset).await.unwrap();
        assert!(dataset.tags().list().await.unwrap().is_empty());
    }
}
