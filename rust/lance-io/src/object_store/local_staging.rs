// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{io::ErrorKind, path::PathBuf};

use futures::{StreamExt, stream, stream::BoxStream};
use lance_core::{Error, Result};
use object_store::{
    ObjectMeta,
    path::{Path, PathPart},
};

use super::ObjectStore;

pub(super) fn is_staging_file_name(name: &str) -> bool {
    name.split_once('#').is_some_and(|(_, suffix)| {
        !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
    })
}

impl ObjectStore {
    pub(super) fn local_staging_path(&self, path: &Path) -> Result<Option<PathBuf>> {
        let Some(name) = path.filename().filter(|name| is_staging_file_name(name)) else {
            return Ok(None);
        };
        let Some(parent) = path.parent() else {
            return Ok(None);
        };
        // Resolve the parent: LocalFileSystem rejects the staging filename itself.
        let local_parent = if let Some(operations) = &self.local_dir_operations {
            operations.directory_to_filesystem(&parent)?
        } else if self.has_direct_local_paths() || self.scheme == "file-object-store" {
            PathBuf::from(crate::local::to_local_path(&parent))
        } else {
            return Ok(None);
        };
        Ok(Some(local_parent.join(name)))
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

#[cfg(test)]
mod tests {
    use super::*;
    use futures::TryStreamExt;
    use rstest::rstest;

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
