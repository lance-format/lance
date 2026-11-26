// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fs::{File, Metadata, Permissions, TryLockError};
use std::io::{self, ErrorKind, Write};
use std::os::unix::fs::{FileExt, MetadataExt, PermissionsExt};
use std::path::PathBuf;
use std::sync::Arc;

use object_store::path::Path as ObjectPath;
use object_store::{GetOptions, GetResultPayload};
use tempfile::TempPath;

use super::{MutationBackend, ObjectStore};
use crate::object_writer::close_file;
use crate::utils::tracking_store::IOTracker;

/// An open generation of a native Unix tag file. The descriptor keeps its
/// inode from being reused while the caller prepares a replacement.
pub struct LocalFileMutation {
    original: Arc<File>,
    path: PathBuf,
    location: ObjectPath,
    size: u64,
    fsync: bool,
    tracker: IOTracker,
}

impl ObjectStore {
    /// Capture an existing tag in an unwrapped native filesystem. Other stores
    /// return `None` and continue through their selected object-store backend.
    ///
    /// ```
    /// # use lance_io::object_store::ObjectStore;
    /// # use object_store::path::Path;
    /// # async fn update_tag(store: &ObjectStore, path: &Path) -> std::io::Result<()> {
    /// if let Some(tag) = store.open_local_mutation(path).await? {
    ///     let mut json = tag.read_all().await?;
    ///     json.push(b'\n');
    ///     tag.publish(Some(json)).await?;
    /// }
    /// # Ok(()) }
    /// ```
    pub async fn open_local_mutation(
        &self,
        location: &ObjectPath,
    ) -> io::Result<Option<LocalFileMutation>> {
        if !self.has_absolute_local_mapping()
            || self.selected_mutation_backend() != Some(MutationBackend::Local)
        {
            return Ok(None);
        }
        let response = self
            .inner
            .get_opts(location, GetOptions::default().with_head(true))
            .await?;
        let GetResultPayload::File(file, path) = response.payload else {
            return Err(io::Error::other("native filesystem HEAD returned no file"));
        };
        Ok(Some(LocalFileMutation {
            original: Arc::new(file),
            path,
            location: location.clone(),
            size: response.meta.size,
            tracker: self.io_tracker.clone(),
            fsync: self.local_fsync,
        }))
    }
}

impl LocalFileMutation {
    /// Read the captured file even if another process has replaced its name.
    pub async fn read_all(&self) -> io::Result<Vec<u8>> {
        let file = self.original.clone();
        let tracker = self.tracker.clone();
        let location = self.location.clone();
        let size = self.size;
        tokio::task::spawn_blocking(move || {
            let metric = tracker.begin_io("get");
            let result = (|| {
                let length = usize::try_from(size).map_err(io::Error::other)?;
                let mut bytes = vec![0; length];
                file.read_exact_at(&mut bytes, 0)?;
                Ok(bytes)
            })();
            let size = result.as_ref().map_or(0, |bytes| bytes.len() as u64);
            metric.record(&result, size);
            if result.is_ok() {
                tracker.record_read("get", location, size, Some(0..size));
            }
            result
        })
        .await?
    }

    /// Atomically replace the captured file, or remove it if `contents` is absent.
    /// Another writer locking, replacing or removing this generation returns `WouldBlock`.
    /// The filesystem must support advisory locking between writers and atomic replacement.
    pub async fn publish(self, contents: Option<Vec<u8>>) -> io::Result<()> {
        let operation = if contents.is_some() { "put" } else { "delete" };
        let size = contents.as_ref().map_or(0, |bytes| bytes.len() as u64);
        let tracker = self.tracker.clone();
        let location = self.location.clone();
        let metric = tracker.begin_io(operation);
        let result = async {
            let staged = if let Some(bytes) = contents {
                let path = self.path.clone();
                let fsync = self.fsync;
                Some(
                    blocking(move || {
                        let parent = path
                            .parent()
                            .ok_or_else(|| io::Error::other("tag has no parent directory"))?;
                        let mut temp = tempfile::Builder::new()
                            .suffix("#0")
                            .permissions(Permissions::from_mode(0o666))
                            .tempfile_in(parent)?;
                        temp.write_all(&bytes)?;
                        if fsync {
                            temp.as_file().sync_all()?;
                        }
                        let (file, temp_path) = temp.into_parts();
                        close_file(file, &path)?;
                        Ok(temp_path)
                    })
                    .await?,
                )
            } else {
                None
            };
            blocking(move || self.publish_staged(staged)).await
        }
        .await;
        metric.record(&result, size);
        if result.is_ok() {
            tracker.record_write(operation, location, size);
        }
        result
    }

    fn publish_staged(self, staged: Option<TempPath>) -> io::Result<()> {
        // Writers that observed the same generation share this inode. A writer
        // that opens the replacement already observes the preceding publication.
        let held = self.original.metadata()?;
        let lock = match self.original.try_lock() {
            Ok(()) => self.original,
            // Some filesystems (notably NFS) require write access for an
            // exclusive lock. Confirm a writable reopen refers to the same generation.
            Err(TryLockError::Error(error))
                if error.raw_os_error() == Some(nix::libc::EBADF)
                    || error.kind() == ErrorKind::PermissionDenied =>
            {
                let writable = File::options()
                    .write(true)
                    .open(&self.path)
                    .map_err(missing_generation)?;
                check_generation(&held, &writable.metadata()?)?;
                writable.try_lock()?;
                Arc::new(writable)
            }
            Err(error) => return Err(error.into()),
        };
        check_generation(
            &held,
            &std::fs::metadata(&self.path).map_err(missing_generation)?,
        )?;
        let sync_parent = self.fsync && staged.is_some();
        match staged {
            Some(staged) => staged.persist(&self.path).map_err(|e| e.error)?,
            None => std::fs::remove_file(&self.path)?,
        }
        drop(lock);
        if sync_parent {
            let parent = self
                .path
                .parent()
                .ok_or_else(|| io::Error::other("tag has no parent directory"))?;
            File::open(parent)?.sync_all()?;
        }
        Ok(())
    }
}

fn missing_generation(error: io::Error) -> io::Error {
    if error.kind() == ErrorKind::NotFound {
        ErrorKind::WouldBlock.into()
    } else {
        error
    }
}

fn check_generation(held: &Metadata, named: &Metadata) -> io::Result<()> {
    if (held.dev(), held.ino()) == (named.dev(), named.ino()) {
        Ok(())
    } else {
        Err(ErrorKind::WouldBlock.into())
    }
}

// Dropping the JoinSet cancels a queued blocking operation. Staging never
// publishes, and a running publication owns the descriptor until the final syscall.
async fn blocking<T: Send + 'static>(
    operation: impl FnOnce() -> io::Result<T> + Send + 'static,
) -> io::Result<T> {
    let mut task = tokio::task::JoinSet::new();
    task.spawn_blocking(operation);
    task.join_next()
        .await
        .ok_or_else(|| io::Error::other("missing local mutation task"))??
}

#[cfg(test)]
mod tests {
    use super::ObjectStore;
    use lance_core::utils::tempfile::TempStdDir;
    use rstest::rstest;

    #[tokio::test]
    async fn test_local_mutation_checks_the_open_generation() {
        let directory = TempStdDir::default();
        let (store, base) = ObjectStore::from_uri(directory.to_str().unwrap())
            .await
            .unwrap();
        let path = base.join("release.json");
        store.put(&path, b"original").await.unwrap();
        let old = store.open_local_mutation(&path).await.unwrap().unwrap();
        let competing = store.open_local_mutation(&path).await.unwrap().unwrap();
        competing.publish(Some(b"new".to_vec())).await.unwrap();
        assert_eq!(old.read_all().await.unwrap().as_slice(), b"original");
        assert_eq!(
            old.publish(None).await.unwrap_err().kind(),
            std::io::ErrorKind::WouldBlock,
        );
        assert_eq!(
            std::fs::read(directory.join("release.json")).unwrap(),
            b"new"
        );
    }

    #[rstest]
    #[case::removed(false)]
    #[case::recreated(true)]
    #[tokio::test]
    async fn test_local_mutation_rejects_removed_generation(#[case] recreate: bool) {
        let directory = TempStdDir::default();
        let (store, base) = ObjectStore::from_uri(directory.to_str().unwrap())
            .await
            .unwrap();
        let path = base.join("release.json");
        store.put(&path, b"original").await.unwrap();
        let old = store.open_local_mutation(&path).await.unwrap().unwrap();
        let remover = store.open_local_mutation(&path).await.unwrap().unwrap();
        remover.publish(None).await.unwrap();
        assert_eq!(
            store.open_local_mutation(&path).await.err().unwrap().kind(),
            std::io::ErrorKind::NotFound
        );
        if recreate {
            store.put(&path, b"successor").await.unwrap();
        }
        assert_eq!(
            old.publish(Some(b"stale".to_vec()))
                .await
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::WouldBlock
        );
        let observed = std::fs::read(directory.join("release.json"));
        if recreate {
            assert_eq!(observed.unwrap(), b"successor");
        } else {
            assert_eq!(observed.unwrap_err().kind(), std::io::ErrorKind::NotFound);
        }
    }

    #[tokio::test]
    async fn test_local_mutation_conflict_is_per_generation() {
        let directory = TempStdDir::default();
        let (store, base) = ObjectStore::from_uri(directory.to_str().unwrap())
            .await
            .unwrap();
        let first = base.clone().join("first.json");
        let second = base.join("second.json");
        store.put(&first, b"first").await.unwrap();
        store.put(&second, b"second").await.unwrap();
        let blocked = store.open_local_mutation(&first).await.unwrap().unwrap();
        let independent = store.open_local_mutation(&second).await.unwrap().unwrap();
        let held = std::fs::File::options()
            .read(true)
            .write(true)
            .open(directory.join("first.json"))
            .unwrap();
        held.try_lock().unwrap();
        assert_eq!(
            blocked
                .publish(Some(b"stale".to_vec()))
                .await
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::WouldBlock
        );
        independent.publish(Some(b"new".to_vec())).await.unwrap();
        assert_eq!(
            std::fs::read(directory.join("first.json")).unwrap(),
            b"first"
        );
        assert_eq!(
            std::fs::read(directory.join("second.json")).unwrap(),
            b"new"
        );
        assert_eq!(std::fs::read_dir(&*directory).unwrap().count(), 2);
    }

    #[rstest]
    #[case::replace(Some(b"updated".to_vec()))]
    #[case::remove(None)]
    #[tokio::test]
    async fn test_local_mutation_of_read_only_tag(#[case] replacement: Option<Vec<u8>>) {
        use std::os::unix::fs::PermissionsExt;
        let directory = TempStdDir::default();
        let (store, base) = ObjectStore::from_uri(directory.to_str().unwrap())
            .await
            .unwrap();
        let path = base.join("release.json");
        let native_path = directory.join("release.json");
        store.put(&path, b"original").await.unwrap();
        std::fs::set_permissions(&native_path, std::fs::Permissions::from_mode(0o444)).unwrap();
        let mutation = store.open_local_mutation(&path).await.unwrap().unwrap();
        mutation.publish(replacement.clone()).await.unwrap();
        if let Some(expected) = replacement {
            assert_eq!(std::fs::read(native_path).unwrap(), expected);
        } else {
            assert!(!native_path.exists());
        }
    }

    #[test]
    fn test_local_mutation_cancels_queued_delete() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .max_blocking_threads(1)
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let directory = TempStdDir::default();
            let (store, base) = ObjectStore::from_uri(directory.to_str().unwrap())
                .await
                .unwrap();
            let path = base.join("release.json");
            store.put(&path, b"original").await.unwrap();
            let mutation = store.open_local_mutation(&path).await.unwrap().unwrap();

            let (entered, ready) = tokio::sync::oneshot::channel();
            let (release, wait) = std::sync::mpsc::channel();
            let blocker = tokio::task::spawn_blocking(move || {
                entered.send(()).unwrap();
                wait.recv().unwrap();
            });
            ready.await.unwrap();
            let mut pending = Box::pin(mutation.publish(None));
            assert!(futures::poll!(&mut pending).is_pending());
            drop(pending);
            release.send(()).unwrap();
            blocker.await.unwrap();
            tokio::task::spawn_blocking(|| {}).await.unwrap();
            assert_eq!(
                std::fs::read(directory.join("release.json")).unwrap(),
                b"original"
            );
        });
    }
}
