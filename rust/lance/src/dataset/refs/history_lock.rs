// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use object_store::{Error as ObjectStoreError, ObjectStoreExt, PutMode, UpdateVersion, path::Path};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::dataset::branch_location::BranchLocation;

/// JSON coordination protocol used by cleanup and history reference writers.
const HISTORY_LOCK_FORMAT_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct LockContents {
    format_version: u32,
    owner: Uuid,
    operation: String,
    released: bool,
}

struct Release {
    store: Arc<ObjectStore>,
    path: Path,
    /// Local stores use atomic create/delete; remote stores use conditional
    /// updates so a retried release cannot release a subsequent owner's lock.
    version: Option<UpdateVersion>,
    contents: Vec<u8>,
}

impl Release {
    async fn execute(self) -> Result<()> {
        if let Some(version) = self.version {
            self.store
                .inner
                .put_opts(
                    &self.path,
                    self.contents.into(),
                    PutMode::Update(version).into(),
                )
                .await?;
        } else {
            self.store.delete(&self.path).await?;
        }
        Ok(())
    }
}

/// Excludes physical cleanup from operations that publish references to history.
/// All branches use the table root, since their files can reference each other.
///
/// This lock deliberately has no expiry: a paused owner could resume deleting
/// files after a lease expired. Cancellation or a process crash therefore
/// requires removing the marker after stopping all cleanup and reference mutations.
pub struct HistoryLock {
    release: Option<Box<Release>>,
    /// Before acquisition returns, no protected I/O can have started and a
    /// cancelled acquisition can safely release its ownership in the background.
    is_acquiring: bool,
}

impl HistoryLock {
    pub(crate) async fn acquire(
        store: Arc<ObjectStore>,
        location: &BranchLocation,
        operation: &str,
    ) -> Result<Self> {
        let path = location
            .find_main()?
            .path
            .join("_refs")
            .join("history.lock");
        let operation = operation.to_string();
        // If the caller is cancelled while the create is in flight, the task
        // still constructs and drops the guard after a successful acquisition.
        let mut guard = tokio::spawn(async move {
            let conflict = || Error::RefConflict {
                message: format!(
                    "Cannot {operation}: cleanup or a history reference mutation holds \
                     {path}. Retry after it completes. If its operation was interrupted, stop all \
                     cleanup and history reference mutations before removing this lock."
                ),
            };
            let mut contents = LockContents {
                format_version: HISTORY_LOCK_FORMAT_VERSION,
                owner: Uuid::new_v4(),
                operation: operation.clone(),
                released: false,
            };
            let payload = serde_json::to_vec(&contents)?;
            let acquired = if store.is_local() || store.scheme() == "file-object-store" {
                store.put_if_absent(&path, payload.into()).await.map(|()| None)
            } else {
                // COS can silently ignore conditional requests; never fall back
                // to an unconditional write for coordination.
                if store.scheme() == "cos" {
                    return Err(Error::not_supported(
                        "History locking requires reliable conditional writes; Tencent COS is unsupported",
                    ));
                }
                let version = match store.inner.get(&path).await {
                    Ok(previous) => {
                        let version = UpdateVersion {
                            e_tag: previous.meta.e_tag.clone(),
                            version: previous.meta.version.clone(),
                        };
                        let previous: LockContents =
                            serde_json::from_slice(&previous.bytes().await?)?;
                        if previous.format_version != HISTORY_LOCK_FORMAT_VERSION {
                            return Err(Error::not_supported(format!(
                                "History lock {path} has unsupported format_version {}",
                                previous.format_version,
                            )));
                        }
                        if !previous.released {
                            return Err(conflict());
                        }
                        version
                    }
                    Err(ObjectStoreError::NotFound { .. }) => {
                        // Initialize an unlocked marker, then acquire with an
                        // update. A store that supports create but not update
                        // must fail before any protected work is published.
                        contents.released = true;
                        let initialized = store.inner.put_opts(
                            &path,
                            serde_json::to_vec(&contents)?.into(),
                            PutMode::Create.into(),
                        ).await.map_err(|error| match error {
                            ObjectStoreError::AlreadyExists { .. }
                            | ObjectStoreError::Precondition { .. } => conflict(),
                            error => error.into(),
                        })?;
                        contents.released = false;
                        initialized.into()
                    }
                    Err(error) => return Err(error.into()),
                };
                if version.e_tag.is_none() && version.version.is_none() {
                    return Err(Error::not_supported(
                        "History locking requires an object ETag or version for conditional updates",
                    ));
                }
                store.inner.put_opts(
                    &path,
                    payload.into(),
                    PutMode::Update(version).into(),
                ).await.map(|result| Some(result.into()))
            };
            let version = acquired.map_err(|error| match error {
                ObjectStoreError::AlreadyExists { .. } | ObjectStoreError::Precondition { .. } => conflict(),
                error => error.into(),
            })?;
            contents.released = true;
            Ok(Self {
                release: Some(Box::new(Release {
                    store,
                    path,
                    version,
                    contents: serde_json::to_vec(&contents)?,
                })),
                is_acquiring: true,
            })
        })
        .await
        .map_err(|error| Error::io(format!("History lock acquisition task failed: {error}")))??;
        guard.is_acquiring = false;
        Ok(guard)
    }

    pub(crate) async fn finish<T>(self, result: Result<T>) -> Result<T> {
        let released = self.release().await;
        match result {
            Ok(value) => released.map(|()| value),
            Err(error) => {
                if let Err(release_error) = released {
                    log::warn!("Failed to release history lock: {release_error}");
                }
                Err(error)
            }
        }
    }

    pub(crate) async fn release(mut self) -> Result<()> {
        if let Some(release) = self.release.take() {
            // A caller's cancellation must not cancel an in-flight release.
            tokio::spawn(release.execute()).await.map_err(|error| {
                Error::io(format!("History lock release task failed: {error}"))
            })??;
        }
        Ok(())
    }
}

impl Drop for HistoryLock {
    fn drop(&mut self) {
        let Some(release) = self.release.take() else {
            return;
        };
        if !self.is_acquiring {
            // Dropping a future does not cancel an already submitted storage
            // request. Releasing here could let cleanup race that request.
            log::warn!(
                "History operation cancelled; retaining {} until all cleanup and \
                 history reference mutations stop and the lock can be removed manually",
                release.path,
            );
        } else if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                if let Err(error) = release.execute().await {
                    log::warn!("Failed to release history lock: {error}");
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use lance_core::utils::tempfile::TempStrDir;
    use object_store::throttle::{ThrottleConfig, ThrottledStore};
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::acquire(true)]
    #[case::release(false)]
    #[tokio::test(start_paused = true)]
    async fn cancellation_during_lock_io_releases_ownership(#[case] during_acquire: bool) {
        let mut store = ObjectStore::memory();
        store.inner = Arc::new(ThrottledStore::new(
            store.inner,
            ThrottleConfig {
                wait_put_per_call: Duration::from_secs(1),
                ..Default::default()
            },
        ));
        let store = Arc::new(store);
        let location = BranchLocation {
            path: Path::from("table"),
            uri: "memory://table".into(),
            branch: None,
        };
        let task = if during_acquire {
            tokio::spawn({
                let store = store.clone();
                let location = location.clone();
                async move {
                    let guard = HistoryLock::acquire(store, &location, "cancelled acquire").await?;
                    guard.release().await
                }
            })
        } else {
            let guard = HistoryLock::acquire(store.clone(), &location, "cancelled release")
                .await
                .unwrap();
            tokio::spawn(guard.release())
        };
        tokio::time::sleep(Duration::from_millis(1)).await;
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        tokio::time::sleep(Duration::from_secs(3)).await;
        HistoryLock::acquire(store, &location, "next owner")
            .await
            .unwrap()
            .release()
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn stale_release_cannot_unlock_a_new_owner() {
        let store = Arc::new(ObjectStore::memory());
        let location = BranchLocation {
            path: Path::from("table"),
            uri: "memory://table".into(),
            branch: None,
        };
        let first = HistoryLock::acquire(store.clone(), &location, "first")
            .await
            .unwrap();
        let release = first.release.as_ref().unwrap();
        let delayed_release = Release {
            store: release.store.clone(),
            path: release.path.clone(),
            version: release.version.clone(),
            contents: release.contents.clone(),
        };
        first.release().await.unwrap();
        let second = HistoryLock::acquire(store.clone(), &location, "second")
            .await
            .unwrap();
        let error = delayed_release.execute().await.unwrap_err();
        assert!(matches!(error, Error::IO { .. }));
        assert!(error.to_string().contains("precondition"), "{error}");
        let error = HistoryLock::acquire(store, &location, "third")
            .await
            .err()
            .unwrap();
        assert!(matches!(error, Error::RefConflict { .. }));
        second.release().await.unwrap();
    }

    #[rstest]
    #[tokio::test]
    async fn lock_excludes_all_branches(#[values(false, true)] is_local: bool) {
        let dir = TempStrDir::default();
        let store = Arc::new(if is_local {
            ObjectStore::local()
        } else {
            ObjectStore::memory()
        });
        let location = BranchLocation {
            path: Path::from(dir.as_str()),
            uri: dir.as_str().to_string(),
            branch: None,
        };
        let branch = location.find_branch(Some("child/nested")).unwrap();
        let guard = HistoryLock::acquire(store.clone(), &branch, "branch operation")
            .await
            .unwrap();
        let error = HistoryLock::acquire(store.clone(), &location, "root cleanup")
            .await
            .err()
            .unwrap();
        assert!(matches!(error, Error::RefConflict { .. }));
        guard.release().await.unwrap();
        HistoryLock::acquire(store, &location, "next owner")
            .await
            .unwrap()
            .release()
            .await
            .unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn abandoned_operation_retains_ownership() {
        let store = Arc::new(ObjectStore::memory());
        let location = BranchLocation {
            path: Path::from("table"),
            uri: "memory://table".into(),
            branch: None,
        };
        let guard = HistoryLock::acquire(store.clone(), &location, "interrupted cleanup")
            .await
            .unwrap();
        drop(guard);
        // Let any erroneous background release complete before checking that
        // outstanding storage requests remain excluded from another owner.
        tokio::time::sleep(Duration::from_secs(1)).await;
        let error = HistoryLock::acquire(store, &location, "next owner")
            .await
            .err()
            .unwrap();
        assert!(matches!(error, Error::RefConflict { .. }));
        assert!(error.to_string().contains("history.lock"), "{error}");
    }
}
