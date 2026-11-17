// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{io::ErrorKind, path::PathBuf, sync::Arc};

use futures::{StreamExt, stream, stream::BoxStream};
use lance_core::{Error, Result};
use object_store::{
    ObjectMeta,
    path::{Path, PathPart},
};

use super::{LocalDirOperations, ObjectStore};
use crate::utils::tracking_store::IOTracker;

pub(super) fn is_staging_file_name(name: &str) -> bool {
    name.split_once('#').is_some_and(|(_, suffix)| {
        !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
    })
}

/// The store state that resolves a hidden local staging path to a filesystem path.
///
/// [`ObjectStore::delete_stream`] returns a `'static` stream that cannot borrow the
/// store, so it holds this instead of a second copy of the staging-name and
/// local-root rules. [`Self::for_store`] yields `None` for a store that cannot hold
/// hidden staging files, so its deletes need no local special case.
#[derive(Clone)]
pub(super) struct LocalStagingPaths {
    local_dir_operations: Option<Arc<dyn LocalDirOperations>>,
    /// Whether object paths map directly onto filesystem paths below the root.
    has_local_root: bool,
}

impl LocalStagingPaths {
    pub(super) fn for_store(store: &ObjectStore) -> Option<Self> {
        let has_local_root = store.has_direct_local_paths() || store.scheme == "file-object-store";
        if store.local_dir_operations.is_none() && !has_local_root {
            return None;
        }
        Some(Self {
            local_dir_operations: store.local_dir_operations.clone(),
            has_local_root,
        })
    }

    pub(super) fn resolve(&self, path: &Path) -> Result<Option<PathBuf>> {
        let Some(name) = path.filename().filter(|name| is_staging_file_name(name)) else {
            return Ok(None);
        };
        let Some(parent) = path.parent() else {
            return Ok(None);
        };
        // Resolve the parent: LocalFileSystem rejects the staging filename itself.
        let local_parent = if let Some(operations) = &self.local_dir_operations {
            operations.directory_to_filesystem(&parent)?
        } else if self.has_local_root {
            PathBuf::from(crate::local::to_local_path(&parent))
        } else {
            return Ok(None);
        };
        Ok(Some(local_parent.join(name)))
    }
}

impl ObjectStore {
    pub(super) fn local_staging_path(&self, path: &Path) -> Result<Option<PathBuf>> {
        match LocalStagingPaths::for_store(self) {
            Some(paths) => paths.resolve(path),
            None => Ok(None),
        }
    }

    /// Recursively list local staging files hidden by `object_store`'s normal listing.
    ///
    /// These files have a nonempty decimal suffix after `#` and can survive an
    /// interrupted write. This stream does not follow symbolic links and is empty
    /// for non-local stores or missing directories. Callers must check file age
    /// before deleting candidates, since a staging file may belong to a live write.
    ///
    /// ```
    /// # use lance_core::Result;
    /// # use lance_io::object_store::ObjectStore;
    /// # use object_store::path::Path;
    /// # use futures::TryStreamExt;
    /// # async fn inspect(store: &ObjectStore, dir: &Path) -> Result<()> {
    /// let mut files = store.read_local_staging_files(dir);
    /// let cutoff = chrono::Utc::now() - chrono::TimeDelta::days(7);
    /// while let Some(file) = files.try_next().await? {
    ///     if file.last_modified < cutoff {
    ///         store.delete(&file.location).await?;
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_local_staging_files(&self, root: &Path) -> BoxStream<'_, Result<ObjectMeta>> {
        let local_root = if let Some(operations) = &self.local_dir_operations {
            operations.directory_to_filesystem(root)
        } else if self.has_direct_local_paths() || self.scheme == "file-object-store" {
            Ok(PathBuf::from(crate::local::to_local_path(root)))
        } else {
            return stream::empty().boxed();
        };
        let local_root = match local_root {
            Ok(path) => path,
            Err(error) => return stream::once(std::future::ready(Err(error))).boxed(),
        };

        // Keep one directory iterator and pending directories, rather than collecting
        // every file in a potentially large dataset into memory.
        stream::try_unfold(
            (
                vec![(local_root, root.clone())],
                None::<(tokio::fs::ReadDir, Path)>,
            ),
            |(mut pending, mut current)| async move {
                loop {
                    let Some((entries, object_dir)) = &mut current else {
                        let Some((local_dir, object_dir)) = pending.pop() else {
                            return Ok(None);
                        };
                        match tokio::fs::symlink_metadata(&local_dir).await {
                            Ok(metadata) if metadata.is_dir() => {}
                            Ok(_) => continue,
                            Err(error) if error.kind() == ErrorKind::NotFound => continue,
                            Err(error) => {
                                return Err(Error::io(format!(
                                    "Failed to inspect local staging directory {}: {error}",
                                    local_dir.display()
                                )));
                            }
                        }
                        match tokio::fs::read_dir(&local_dir).await {
                            Ok(entries) => current = Some((entries, object_dir)),
                            Err(error) if error.kind() == ErrorKind::NotFound => continue,
                            Err(error) => {
                                return Err(Error::io(format!(
                                    "Failed to list local staging directory {}: {error}",
                                    local_dir.display()
                                )));
                            }
                        }
                        continue;
                    };
                    let entry = match entries.next_entry().await {
                        Ok(Some(entry)) => entry,
                        Ok(None) => {
                            current = None;
                            continue;
                        }
                        Err(error) if error.kind() == ErrorKind::NotFound => {
                            current = None;
                            continue;
                        }
                        Err(error) => return Err(Error::io(format!(
                            "Failed to read local staging directory {object_dir}: {error}"
                        ))),
                    };
                    let file_type = match entry.file_type().await {
                        Ok(file_type) => file_type,
                        Err(error) if error.kind() == ErrorKind::NotFound => continue,
                        Err(error) => return Err(Error::io(format!(
                            "Failed to inspect local staging path {}: {error}", entry.path().display()
                        ))),
                    };
                    let name = entry.file_name();
                    let Some(name) = name.to_str() else { continue };
                    // Match object_store's is_valid_file_path exactly so files
                    // visible in the ordinary listing are never listed twice.
                    if !(file_type.is_dir() || file_type.is_file() && is_staging_file_name(name)) {
                        continue;
                    }
                    // Converting a string into PathPart would percent-encode '#',
                    // producing a different filename from the native directory entry.
                    let part = PathPart::parse(name).map_err(|error| Error::io(format!(
                        "Invalid local staging path {}: {error}", entry.path().display()
                    )))?;
                    let location = object_dir.clone().join(part);
                    if file_type.is_dir() {
                        pending.push((entry.path(), location));
                        continue;
                    }
                    let metadata = match entry.metadata().await {
                        Ok(metadata) => metadata,
                        Err(error) if error.kind() == ErrorKind::NotFound => continue,
                        Err(error) => return Err(Error::io(format!(
                            "Failed to read local staging file metadata {}: {error}", entry.path().display()
                        ))),
                    };
                    if !metadata.is_file() {
                        continue;
                    }
                    let meta = ObjectMeta {
                        location,
                        last_modified: metadata.modified().map_err(|error| Error::io(format!(
                            "Failed to read local staging file modification time {}: {error}", entry.path().display()
                        )))?.into(),
                        size: metadata.len(),
                        e_tag: None,
                        version: None,
                    };
                    return Ok(Some((meta, (pending, current))));
                }
            },
        )
        .boxed()
    }
}

/// Removes one hidden local staging file and records it as a delete.
///
/// `object_store`'s local implementation rejects these names, so removal bypasses it.
/// Recording the IO here keeps one logical delete in the metrics whether it took this
/// path or the backend path.
pub(super) async fn remove_local_staging_file(
    path: &Path,
    local_path: &std::path::Path,
    io_tracker: &IOTracker,
) -> Result<()> {
    let metrics = io_tracker.begin_io("delete");
    let result = tokio::fs::remove_file(local_path).await.map_err(|error| {
        if error.kind() == ErrorKind::NotFound {
            Error::not_found(path.to_string())
        } else {
            Error::io(format!(
                "Failed to remove local staging file {}: {error}",
                local_path.display()
            ))
        }
    });
    metrics.record(&result, 0);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::TryStreamExt;
    use rstest::rstest;

    #[rstest]
    #[case::local("file")]
    #[case::object_store("file-object-store")]
    #[tokio::test]
    async fn delete_stream_removes_hidden_staging_files(#[case] scheme: &str) {
        // Cleanup hands a whole batch to `delete_stream` once it has enough
        // candidates, so crossing that bound must not lose staging files to
        // LocalFileSystem's rejection of their names.
        let staging_count = 120;
        let directory = tempfile::tempdir().unwrap();
        let local_root = directory.path().join("dataset");
        std::fs::create_dir_all(local_root.join("data")).unwrap();
        let file_url = url::Url::from_directory_path(&local_root).unwrap();
        let uri = file_url
            .as_str()
            .replacen("file:", &format!("{scheme}:"), 1);
        let (store, root) = ObjectStore::from_uri(&uri).await.unwrap();

        let mut paths = Vec::new();
        for index in 0..staging_count {
            let name = format!("data/part-{index:03}#{index}");
            std::fs::write(local_root.join(&name), b"staging").unwrap();
            paths.push(Path::parse(format!("{root}/{name}")).unwrap());
        }
        // A regular object in the same batch must still take the backend delete.
        let regular = root.clone().join("data/part-regular.lance");
        store.put(&regular, b"regular").await.unwrap();
        paths.push(regular.clone());

        let to_remove = stream::iter(paths.clone().into_iter().map(Ok)).boxed();
        let mut removed = store
            .delete_stream(to_remove)
            .try_collect::<Vec<_>>()
            .await
            .expect("a batch of staging files must delete without errors");
        removed.sort_unstable();
        let mut expected = paths.clone();
        expected.sort_unstable();
        assert_eq!(removed, expected);

        for index in 0..staging_count {
            let name = format!("data/part-{index:03}#{index}");
            assert!(!local_root.join(&name).exists(), "{name} survived");
        }
        assert!(!store.exists(&regular).await.unwrap());
        // The listing must not find them either, so the deletion is not merely
        // unreported.
        assert!(
            store
                .read_local_staging_files(&root)
                .try_collect::<Vec<_>>()
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[rstest]
    #[case::local("file")]
    #[case::object_store("file-object-store")]
    #[tokio::test]
    async fn lists_only_hidden_staging_files(#[case] scheme: &str) {
        let directory = tempfile::tempdir().unwrap();
        let local_root = directory.path().join("dataset");
        std::fs::create_dir_all(local_root.join("nested")).unwrap();
        for name in [
            "file#1",
            "nested/file#123",
            "nested/#0",
            "file#",
            "file#1x",
            "file#1#2",
            "file",
        ] {
            std::fs::write(local_root.join(name), b"staging").unwrap();
        }
        let file_url = url::Url::from_directory_path(&local_root).unwrap();
        let uri = file_url
            .as_str()
            .replacen("file:", &format!("{scheme}:"), 1);
        let (store, root) = ObjectStore::from_uri(&uri).await.unwrap();
        let mut files = store
            .read_local_staging_files(&root)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        files.sort_by(|left, right| left.location.cmp(&right.location));
        assert_eq!(
            files
                .iter()
                .map(|file| file.location.clone())
                .collect::<Vec<_>>(),
            [
                Path::parse(format!("{root}/file#1")).unwrap(),
                Path::parse(format!("{root}/nested/#0")).unwrap(),
                Path::parse(format!("{root}/nested/file#123")).unwrap()
            ]
        );
        for file in files {
            assert_eq!(file.size, 7);
            assert!(file.last_modified > chrono::DateTime::<chrono::Utc>::UNIX_EPOCH);
            store.delete(&file.location).await.unwrap();
        }
        assert!(!local_root.join("file#1").exists());
        assert!(!local_root.join("nested/file#123").exists());
        assert!(!local_root.join("nested/#0").exists());
        assert!(local_root.join("file").exists());
        assert!(
            store
                .read_local_staging_files(&root.clone().join("missing"))
                .try_collect::<Vec<_>>()
                .await
                .unwrap()
                .is_empty()
        );
        assert!(
            ObjectStore::memory()
                .read_local_staging_files(&root)
                .try_collect::<Vec<_>>()
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn does_not_follow_staging_symlinks() {
        let directory = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("file#1"), b"outside").unwrap();
        std::os::unix::fs::symlink(outside.path(), directory.path().join("linked_dir")).unwrap();
        std::os::unix::fs::symlink(
            outside.path().join("file#1"),
            directory.path().join("linked#1"),
        )
        .unwrap();
        let store = ObjectStore::local();
        let root = Path::from_absolute_path(directory.path()).unwrap();
        assert!(
            store
                .read_local_staging_files(&root)
                .try_collect::<Vec<_>>()
                .await
                .unwrap()
                .is_empty()
        );
        assert!(
            store
                .read_local_staging_files(&root.join("linked_dir"))
                .try_collect::<Vec<_>>()
                .await
                .unwrap()
                .is_empty()
        );
    }
}
