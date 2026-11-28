// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fs::{File, TryLockError};
use std::io::ErrorKind;
use std::path::PathBuf;

use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use object_store::path::{Path, PathPart};
use same_file::Handle;

use super::{BranchIdentifier, REFS_DIR};

/// Holds a local branch name across publication, deletion, or a cleanup retry.
pub(super) struct LocalBranchOperation {
    _lock: File,
    path: PathBuf,
    marker: Path,
    pub(super) pending: Option<BranchIdentifier>,
}

impl LocalBranchOperation {
    /// A canceled request may already have dispatched filesystem I/O. Keep
    /// the name locked until that publication finishes.
    pub(super) async fn publish(
        guard: Option<Self>,
        write: impl Future<Output = Result<()>> + Send + 'static,
    ) -> Result<()> {
        let Some(guard) = guard else {
            return write.await;
        };
        tokio::spawn(async move {
            let _guard = guard;
            write.await
        })
        .await
        .map_err(|error| Error::io(format!("Could not publish local branch ref: {error}")))?
    }

    pub(super) async fn acquire(
        store: &ObjectStore,
        root: &Path,
        branch: &str,
    ) -> Result<Option<Self>> {
        if !store.is_local() && store.scheme() != "file-object-store" {
            return Ok(None);
        }
        let Some(local_root) = store.canonicalize_local_path(root).await? else {
            return Ok(None);
        };
        let directory = local_root.join(REFS_DIR).join("deleting");
        let filename = format!("{}.lock", PathPart::from(branch).as_ref());
        let path = directory.join(filename);
        let marker_name = format!("{branch}.json");
        let marker = root
            .clone()
            .join(REFS_DIR)
            .join("deleting")
            .join(marker_name.as_str());
        let local_marker = directory.join(PathPart::from(marker_name.as_str()).as_ref());
        let branch = branch.to_owned();
        let guard = tokio::task::spawn_blocking(move || -> Result<Self> {
            std::fs::create_dir_all(&directory)?;
            if !std::fs::symlink_metadata(&directory)?.is_dir() {
                return Err(Error::io(format!(
                    "Expected a directory at {}",
                    directory.display()
                )));
            }
            match std::fs::symlink_metadata(&path) {
                Ok(metadata) if !metadata.is_file() => {
                    return Err(Error::io(format!(
                        "Invalid branch lock file at {}",
                        path.display()
                    )));
                }
                Err(error) if error.kind() != ErrorKind::NotFound => return Err(error.into()),
                _ => {}
            }
            let file = File::options()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&path)?;
            let busy = || Error::RefConflict {
                message: format!(
                    "Another operation is changing branch {branch}. Retry the operation"
                ),
            };
            match file.try_lock() {
                Ok(()) => {}
                Err(TryLockError::WouldBlock) => return Err(busy()),
                Err(TryLockError::Error(error)) => return Err(error.into()),
            }
            // The previous holder can finish and unlink its file between our
            // open and lock. Only the holder of the *linked* file owns the name.
            let linked = match Handle::from_path(&path) {
                Ok(linked) => linked,
                Err(error) if error.kind() == ErrorKind::NotFound => return Err(busy()),
                Err(error) => return Err(error.into()),
            };
            if linked != Handle::from_file(file.try_clone()?)? {
                return Err(busy());
            }
            if !std::fs::symlink_metadata(&path)?.is_file() || file.metadata()?.len() != 0 {
                return Err(Error::io(format!(
                    "Invalid branch lock file at {}",
                    path.display()
                )));
            }
            let mut guard = Self {
                _lock: file,
                path,
                marker,
                pending: None,
            };
            match std::fs::symlink_metadata(&local_marker) {
                Ok(metadata) if metadata.is_file() => {
                    let (name, identifier): (String, BranchIdentifier) =
                        serde_json::from_reader(File::open(&local_marker)?)?;
                    if name != branch {
                        return Err(Error::RefConflict {
                            message: format!(
                                "Cleanup record for branch {name} aliases branch {branch}"
                            ),
                        });
                    }
                    guard.pending = Some(identifier);
                }
                Ok(_) => {
                    return Err(Error::io(format!(
                        "Invalid branch cleanup record at {}",
                        local_marker.display()
                    )));
                }
                Err(error) if error.kind() == ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
            Ok(guard)
        })
        .await
        .map_err(|error| {
            Error::io(format!("Could not acquire local branch operation: {error}"))
        })??;
        Ok(Some(guard))
    }

    pub(super) async fn begin_delete(
        &mut self,
        store: &ObjectStore,
        branch: &str,
        identifier: &BranchIdentifier,
    ) -> Result<()> {
        if let Some(pending) = &self.pending {
            if pending != identifier {
                self.finish_delete(store).await?;
                return Err(Error::RefConflict {
                    message: format!(
                        "Branch {branch} was recreated during deletion. Its new ref and data were preserved"
                    ),
                });
            }
        } else {
            super::put_ref_if_absent(
                store,
                &self.marker,
                serde_json::to_vec(&(branch, identifier))?,
                format!("Another operation is deleting branch {branch}"),
            )
            .await?;
            self.pending = Some(identifier.clone());
        }
        Ok(())
    }

    pub(super) async fn finish_delete(&mut self, store: &ObjectStore) -> Result<()> {
        if self.pending.is_some() {
            store.delete(&self.marker).await?;
            self.pending = None;
        }
        Ok(())
    }
}

impl Drop for LocalBranchOperation {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_file(&self.path) {
            log::warn!(
                "Could not remove idle branch operation file {}: {error}",
                self.path.display()
            );
        }
    }
}
