// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Optimized local I/Os

use std::collections::HashSet;
use std::fs::File;
use std::io::{ErrorKind, Read, SeekFrom};
use std::ops::Range;
use std::sync::Arc;

// TODO: Clean up windows/unix stuff
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use chrono::{DateTime, Utc};
use futures::future::BoxFuture;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use object_store::path::Path;
use tokio::io::AsyncSeekExt;
use tokio::sync::OnceCell;
use tracing::instrument;

use crate::object_reader::stream_local_range;
use crate::object_store::DEFAULT_LOCAL_IO_PARALLELISM;
use crate::object_writer::WriteResult;
use crate::traits::{ByteStream, Reader, Writer};
use crate::utils::tracking_store::IOTracker;

/// Convert an [`object_store::path::Path`] to a [`std::path::Path`].
pub fn to_local_path(path: &Path) -> String {
    if cfg!(windows) {
        path.to_string()
    } else {
        format!("/{path}")
    }
}

/// Create a local writer's parent directory and persist any newly created path components.
pub(crate) fn create_dir_all_durable(parent: &std::path::Path) -> std::io::Result<()> {
    #[cfg(unix)]
    let first_existing = parent.ancestors().find(|path| path.exists());

    std::fs::create_dir_all(parent)?;

    #[cfg(unix)]
    if let Some(first_existing) = first_existing.filter(|path| *path != parent) {
        let mut directory = parent;
        loop {
            File::open(directory)?.sync_all()?;
            if directory == first_existing {
                break;
            }
            directory = directory.parent().ok_or_else(|| {
                std::io::Error::other(format!(
                    "cannot find parent of newly created directory {}",
                    directory.display()
                ))
            })?;
        }
    }

    Ok(())
}

/// Recursively remove a directory, specified by [`object_store::path::Path`].
pub fn remove_dir_all(path: &Path) -> Result<()> {
    std::fs::remove_dir_all(to_local_path(path)).map_err(|err| match err.kind() {
        ErrorKind::NotFound => Error::not_found(path.to_string()),
        _ => Error::from(err),
    })?;
    Ok(())
}

/// Remove eligible empty directories below `root` without following symbolic links.
pub(crate) fn remove_empty_dirs(
    root: &Path,
    retained_dirs: &HashSet<Path>,
    verified_dirs: &HashSet<Path>,
    unmodified_since: Option<DateTime<Utc>>,
) -> Result<()> {
    let root_path = std::path::PathBuf::from(to_local_path(root));
    let root_metadata = match std::fs::symlink_metadata(&root_path) {
        Ok(metadata) => metadata,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(Error::from(err)),
    };
    if !root_metadata.file_type().is_dir() {
        return Ok(());
    }
    let canonical_root = std::fs::canonicalize(&root_path)?;

    let mut pending_dirs = vec![(root_path, root.clone(), None::<Path>)];
    let mut discovered_dirs = Vec::new();
    let mut file_bearing_roots = HashSet::new();
    let mut first_error = None;
    while let Some((local_dir, object_store_dir, index_root)) = pending_dirs.pop() {
        let entries = match std::fs::read_dir(&local_dir) {
            Ok(entries) => entries,
            Err(err) if err.kind() == ErrorKind::NotFound => continue,
            Err(err) => return Err(Error::from(err)),
        };
        for entry in entries {
            let entry = entry?;
            // DirEntry::file_type does not follow symbolic links. Non-directories, including
            // symlinks, remain in place and prevent their parent from being removed as empty.
            if !entry.file_type()?.is_dir() {
                if let Some(index_root) = &index_root {
                    file_bearing_roots.insert(index_root.clone());
                }
                continue;
            }
            let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            let child = object_store_dir.clone().join(name);
            if retained_dirs.contains(&child) {
                continue;
            }
            let local_child = entry.path();
            let index_root = index_root.clone().unwrap_or_else(|| child.clone());
            // Removing a child changes its parent's mtime, so capture age eligibility before
            // deepest-first deletion starts.
            let is_old_enough = match std::fs::symlink_metadata(&local_child) {
                Ok(metadata) => unmodified_since.is_none_or(|threshold| {
                    metadata
                        .modified()
                        .ok()
                        .map(DateTime::<Utc>::from)
                        .is_some_and(|modified| modified < threshold)
                }),
                Err(err) if err.kind() == ErrorKind::NotFound => continue,
                Err(err) => {
                    first_error.get_or_insert_with(|| Error::from(err));
                    false
                }
            };
            discovered_dirs.push((
                local_child.clone(),
                child.clone(),
                index_root.clone(),
                is_old_enough,
            ));
            pending_dirs.push((local_child, child, Some(index_root)));
        }
    }

    discovered_dirs.sort_unstable_by_key(|(_, path, _, _)| std::cmp::Reverse(path.parts_count()));
    for (local_dir, object_store_dir, index_root, is_old_enough) in discovered_dirs {
        if file_bearing_roots.contains(&index_root) {
            continue;
        }
        match std::fs::symlink_metadata(&local_dir) {
            Ok(metadata) if metadata.file_type().is_dir() => {}
            Ok(_) => continue,
            Err(err) if err.kind() == ErrorKind::NotFound => continue,
            Err(err) => {
                first_error.get_or_insert_with(|| Error::from(err));
                continue;
            }
        }
        let is_verified = verified_dirs.contains(&object_store_dir);
        if !is_verified && !is_old_enough {
            continue;
        }

        // Canonicalization rejects any directory reached through a symlink outside the supplied
        // root. Removing the canonical path also avoids trusting prefixes returned by an object
        // store listing.
        let canonical_dir = match std::fs::canonicalize(&local_dir) {
            Ok(path) if path != canonical_root && path.starts_with(&canonical_root) => path,
            Ok(_) => continue,
            Err(err) if err.kind() == ErrorKind::NotFound => continue,
            Err(err) => {
                first_error.get_or_insert_with(|| Error::from(err));
                continue;
            }
        };
        if let Err(err) = std::fs::remove_dir(canonical_dir)
            && !matches!(
                err.kind(),
                ErrorKind::NotFound | ErrorKind::DirectoryNotEmpty
            )
        {
            first_error.get_or_insert_with(|| Error::from(err));
        }
    }

    if let Some(error) = first_error {
        Err(error)
    } else {
        Ok(())
    }
}

/// Copy a file from one location to another, supporting cross-filesystem copies.
///
/// Unlike hard links, this function works across filesystem boundaries.
pub fn copy_file(from: &Path, to: &Path) -> Result<()> {
    let from_path = to_local_path(from);
    let to_path = to_local_path(to);

    // Ensure the parent directory exists
    if let Some(parent) = std::path::Path::new(&to_path).parent() {
        std::fs::create_dir_all(parent).map_err(Error::from)?;
    }

    std::fs::copy(&from_path, &to_path).map_err(|err| match err.kind() {
        ErrorKind::NotFound => Error::not_found(from.to_string()),
        _ => Error::from(err),
    })?;
    Ok(())
}

/// Await a filesystem operation running on a blocking thread, flattening the
/// join and IO errors into a single `object_store` error.
///
/// Deliberately not written as `handle.await?` at the call sites: a `JoinError`
/// means the operation panicked, and short-circuiting on it would skip the
/// caller's metrics recording for exactly the failure worth counting.
pub(crate) async fn join_local_io<T>(
    handle: tokio::task::JoinHandle<std::io::Result<T>>,
) -> object_store::Result<T> {
    match handle.await {
        Ok(result) => result.map_err(|err| object_store::Error::Generic {
            store: "LocalFileSystem",
            source: err.into(),
        }),
        Err(err) => Err(err.into()),
    }
}

pub(crate) mod resources;

/// Object reader for local file system.
#[derive(Debug)]
pub struct LocalObjectReader {
    memory_pool: Option<Arc<dyn lance_core::utils::memory::MemoryReserver>>,
    /// File handler.
    file: Arc<File>,

    /// Fie path.
    path: Path,

    /// Known size of the file. This is either passed in on construction or
    /// cached on the first metadata call.
    size: OnceCell<usize>,

    /// Block size, in bytes.
    block_size: usize,

    /// IO tracker for monitoring read operations.
    io_tracker: Arc<IOTracker>,
    _resources: Option<resources::ReaderResources>,
}

impl DeepSizeOf for LocalObjectReader {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        // Skipping `file` as it should just be a file handle
        self.path.as_ref().deep_size_of_children(context)
    }
}

impl LocalObjectReader {
    pub async fn open_local_path(
        path: impl AsRef<std::path::Path>,
        block_size: usize,
        known_size: Option<usize>,
    ) -> Result<Box<dyn Reader>> {
        let path = path.as_ref().to_owned();
        let object_store_path = Path::from_filesystem_path(&path)?;
        Self::open(&object_store_path, block_size, known_size).await
    }

    /// Open a local object reader, with default prefetch size.
    ///
    /// For backward compatibility with existing code that doesn't need tracking.
    #[instrument(level = "debug")]
    pub async fn open(
        path: &Path,
        block_size: usize,
        known_size: Option<usize>,
    ) -> Result<Box<dyn Reader>> {
        Self::open_with_tracker(path, block_size, known_size, Default::default()).await
    }

    /// Open a local object reader with optional IO tracking.
    #[instrument(level = "debug")]
    pub(crate) async fn open_with_tracker(
        path: &Path,
        block_size: usize,
        known_size: Option<usize>,
        io_tracker: Arc<IOTracker>,
    ) -> Result<Box<dyn Reader>> {
        let path = path.clone();
        let local_path = to_local_path(&path);
        tokio::task::spawn_blocking(move || {
            let file = File::open(&local_path).map_err(|e| match e.kind() {
                ErrorKind::NotFound => Error::not_found(path.to_string()),
                _ => e.into(),
            })?;
            let size = OnceCell::new_with(known_size);
            Ok(Box::new(Self {
                memory_pool: None,
                _resources: None,
                file: Arc::new(file),
                block_size,
                size,
                path,
                io_tracker,
            }) as Box<dyn Reader>)
        })
        .await?
    }
}

impl Reader for LocalObjectReader {
    fn memory_reserver(&self) -> Option<Arc<dyn lance_core::utils::memory::MemoryReserver>> {
        self.memory_pool.clone()
    }

    fn configure_memory_reserver(
        &mut self,
        pool: Arc<dyn lance_core::utils::memory::MemoryReserver>,
    ) -> Result<()> {
        if self.memory_pool.is_some() {
            return Err(Error::invalid_input(
                "local reader memory reserver is already configured",
            ));
        }
        self.memory_pool = Some(pool);
        Ok(())
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn io_parallelism(&self) -> usize {
        DEFAULT_LOCAL_IO_PARALLELISM
    }

    /// Returns the file size.
    fn size(&self) -> BoxFuture<'_, object_store::Result<usize>> {
        Box::pin(async move {
            let file = self.file.clone();
            self.size
                .get_or_try_init(|| async move {
                    // The metadata lookup is this reader's equivalent of the HEAD
                    // request a cloud reader makes to learn the object size.
                    let metrics = self.io_tracker.begin_io("head");
                    let result = if let Some(pool) = &self.memory_pool {
                        lance_core::utils::tokio::io_task_with_resources(
                            pool.clone(),
                            "local file metadata",
                            move || file.metadata(),
                        )?
                        .await?
                        .map_err(|source| object_store::Error::Generic {
                            store: "LocalFileSystem",
                            source: Box::new(source),
                        })
                    } else {
                        join_local_io(tokio::task::spawn_blocking(move || file.metadata())).await
                    };
                    metrics.record(&result, 0);
                    Ok(result?.len() as usize)
                })
                .await
                .cloned()
        })
    }

    /// Reads a range of data.
    #[instrument(level = "debug", skip(self))]
    fn get_range(&self, range: Range<usize>) -> BoxFuture<'static, object_store::Result<Bytes>> {
        if let Some(pool) = &self.memory_pool {
            return Self::read_with_resources(
                self.file.clone(),
                self.path.clone(),
                self.io_tracker.clone(),
                pool.clone(),
                range,
            );
        }
        let file = self.file.clone();
        let io_tracker = self.io_tracker.clone();
        let path = self.path.clone();
        let num_bytes = range.len() as u64;
        let range_u64 = (range.start as u64)..(range.end as u64);

        Box::pin(async move {
            let metrics = io_tracker.begin_io("get");
            let result = join_local_io(tokio::task::spawn_blocking(move || {
                let mut buf = BytesMut::with_capacity(range.len());
                // Safety: `buf` is set with appropriate capacity above. It is
                // written to below and we check all data is initialized at that point.
                unsafe { buf.set_len(range.len()) };
                #[cfg(unix)]
                file.read_exact_at(buf.as_mut(), range.start as u64)?;
                #[cfg(windows)]
                read_exact_at(file, buf.as_mut(), range.start as u64)?;

                Ok(buf.freeze())
            }))
            .await;

            metrics.record(&result, num_bytes);
            if result.is_ok() {
                io_tracker.record_read("get_range", path, num_bytes, Some(range_u64));
            }

            result
        })
    }

    /// Reads the entire file.
    #[instrument(level = "debug", skip(self))]
    fn get_all(&self) -> BoxFuture<'_, object_store::Result<Bytes>> {
        if self.memory_pool.is_some() {
            return Box::pin(async move {
                let size = self.size().await?;
                self.get_range(0..size).await
            });
        }
        Box::pin(async move {
            let mut file = self.file.clone();
            let io_tracker = self.io_tracker.clone();
            let path = self.path.clone();

            let metrics = io_tracker.begin_io("get");
            let result = join_local_io(tokio::task::spawn_blocking(move || {
                let mut buf = Vec::new();
                file.read_to_end(buf.as_mut())?;
                Ok(Bytes::from(buf))
            }))
            .await;

            let num_bytes = result.as_ref().map_or(0, |bytes| bytes.len() as u64);
            metrics.record(&result, num_bytes);
            if let Ok(bytes) = &result {
                io_tracker.record_read("get_all", path, bytes.len() as u64, None);
            }

            result
        })
    }

    fn get_stream(&self) -> BoxFuture<'_, object_store::Result<ByteStream>> {
        Box::pin(async move {
            let size = self.size().await?;
            if self.memory_pool.is_some() {
                return Ok(self.stream_with_resources(0..size));
            }
            Ok(stream_local_range(
                self.file.clone(),
                self.path.clone(),
                self.io_tracker.clone(),
                0..size,
                self.block_size.max(8 * 1024),
            ))
        })
    }

    fn get_range_stream(
        &self,
        range: Range<usize>,
    ) -> BoxFuture<'_, object_store::Result<ByteStream>> {
        if self.memory_pool.is_some() {
            let stream = self.stream_with_resources(range);
            return Box::pin(async move { Ok(stream) });
        }
        let file = self.file.clone();
        let path = self.path.clone();
        let io_tracker = self.io_tracker.clone();
        let chunk_size = self.block_size.max(8 * 1024);
        Box::pin(async move {
            Ok(stream_local_range(
                file, path, io_tracker, range, chunk_size,
            ))
        })
    }
}

#[cfg(windows)]
pub(crate) fn read_exact_at(
    file: Arc<File>,
    mut buf: &mut [u8],
    mut offset: u64,
) -> std::io::Result<()> {
    let expected_len = buf.len();
    while !buf.is_empty() {
        match file.seek_read(buf, offset) {
            Ok(0) => break,
            Ok(n) => {
                let tmp = buf;
                buf = &mut tmp[n..];
                offset += n as u64;
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    if !buf.is_empty() {
        Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!(
                "failed to fill whole buffer. Expected {} bytes, got {}",
                expected_len, offset
            ),
        ))
    } else {
        Ok(())
    }
}

#[async_trait]
impl Writer for tokio::fs::File {
    async fn tell(&mut self) -> Result<usize> {
        Ok(self.seek(SeekFrom::Current(0)).await? as usize)
    }

    async fn shutdown(&mut self) -> Result<WriteResult> {
        let size = self.seek(SeekFrom::Current(0)).await? as usize;
        tokio::io::AsyncWriteExt::shutdown(self).await?;
        Ok(WriteResult { size, e_tag: None })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        object_store::{
            ObjectStore, ObjectStoreParams, ObjectStoreRegistry, StorageOptionsAccessor,
        },
        scheduler::{ScanScheduler, SchedulerConfig},
        utils::CachedFileSize,
    };
    use futures::TryStreamExt;
    use lance_core::utils::memory::MemoryReserver;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::AsyncWriteExt;

    #[derive(Debug, Default)]
    struct ReadPool {
        used: Arc<AtomicUsize>,
        limit: usize,
        tasks: Arc<lance_core::utils::tasks::ResourceTaskGroup>,
    }

    struct ReadCharge {
        used: Arc<AtomicUsize>,
        bytes: usize,
    }

    impl Drop for ReadCharge {
        fn drop(&mut self) {
            self.used.fetch_sub(self.bytes, Ordering::Release);
        }
    }

    impl MemoryReserver for ReadPool {
        fn task_group(&self) -> Option<Arc<lance_core::utils::tasks::ResourceTaskGroup>> {
            Some(self.tasks.clone())
        }
        fn reserve(&self, stage: &str, bytes: usize) -> Result<Box<dyn Send + Sync>> {
            self.used
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
                    used.checked_add(bytes).filter(|next| *next <= self.limit)
                })
                .map_err(|used| {
                    Error::execution(format!(
                        "{stage}: requested_bytes={bytes}, used_bytes={used}, budget_bytes={}",
                        self.limit
                    ))
                })?;
            Ok(Box::new(ReadCharge {
                used: self.used.clone(),
                bytes,
            }))
        }
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn resource_local_output_publishes_atomically(
        #[values(0, 3, 70_000)] bytes: usize,
        #[values(false, true)] fsync: bool,
    ) {
        let directory = tempfile::tempdir().unwrap();
        let local = directory.path().join("nested/output");
        let path = Path::from_absolute_path(&local).unwrap();
        let used = Arc::new(AtomicUsize::new(0));
        let pool = Arc::new(ReadPool {
            used: used.clone(),
            limit: 256 * 1024,
            ..Default::default()
        });
        let params = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                std::collections::HashMap::from([("local_fsync".to_owned(), fsync.to_string())]),
            ))),
            ..Default::default()
        };
        let (store, _) = ObjectStore::from_uri_and_params(
            Arc::new(ObjectStoreRegistry::default()),
            directory.path().to_str().unwrap(),
            &params,
        )
        .await
        .unwrap();
        let mut writer = store
            .create_with_memory_reserver(&path, pool)
            .await
            .unwrap();
        let data = vec![7_u8; bytes];
        AsyncWriteExt::write_all(writer.as_mut(), &data)
            .await
            .unwrap();
        assert_eq!(writer.tell().await.unwrap(), bytes);
        assert!(!local.exists());
        let result = Writer::shutdown(writer.as_mut()).await.unwrap();
        assert_eq!(result.size, bytes);
        assert!(result.e_tag.is_some());
        assert_eq!(std::fs::read(&local).unwrap(), data);
        assert!(Writer::shutdown(writer.as_mut()).await.is_ok());
        drop(writer);
        assert_eq!(used.load(Ordering::Acquire), 0);
        assert_eq!(
            std::fs::read_dir(local.parent().unwrap()).unwrap().count(),
            1
        );
    }

    #[derive(Debug)]
    struct PublicationObserver(Arc<AtomicUsize>);
    impl crate::object_writer::OutputLifecycle for PublicationObserver {
        fn published(&self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn resource_publication_mode(
        #[values(false, true)] create: bool,
        #[values(false, true)] exists: bool,
        #[values(false, true)] notify: bool,
    ) {
        let directory = tempfile::tempdir().unwrap();
        let local = directory.path().join("output");
        if exists {
            std::fs::write(&local, b"old").unwrap();
        }
        let path = Path::from_absolute_path(&local).unwrap();
        let pool = Arc::new(ReadPool {
            limit: 256 * 1024,
            ..Default::default()
        });
        let store = ObjectStore::local();
        let notifications = Arc::new(AtomicUsize::new(0));
        let owner = notify.then(|| {
            Arc::new(PublicationObserver(notifications.clone()))
                as Arc<dyn crate::object_writer::OutputLifecycle>
        });
        let mut writer = store
            .create_with_resources_mode(
                &path,
                pool.clone(),
                if create {
                    object_store::PutMode::Create
                } else {
                    object_store::PutMode::Overwrite
                },
                owner,
            )
            .await
            .unwrap();
        AsyncWriteExt::write_all(writer.as_mut(), b"new")
            .await
            .unwrap();
        let result = Writer::shutdown(writer.as_mut()).await;
        assert_eq!(result.is_err(), create && exists);
        assert_eq!(
            notifications.load(Ordering::Relaxed),
            usize::from(notify && !(create && exists))
        );
        drop(writer);
        assert_eq!(
            std::fs::read(&local).unwrap(),
            if create && exists { b"old" } else { b"new" }
        );
        assert_eq!(pool.used.load(Ordering::Acquire), 0);
    }

    #[rstest::rstest]
    #[case::file("file")]
    #[case::missing("missing")]
    #[case::directory("directory")]
    #[tokio::test]
    async fn resource_metadata_matches_default(#[case] kind: &str) {
        let directory = tempfile::tempdir().unwrap();
        let local = directory.path().join("object");
        match kind {
            "file" => std::fs::write(&local, b"data").unwrap(),
            "directory" => std::fs::create_dir(&local).unwrap(),
            _ => {}
        }
        let path = Path::from_absolute_path(&local).unwrap();
        let pool = Arc::new(ReadPool {
            limit: 256 * 1024,
            ..Default::default()
        });
        let store = ObjectStore::local();
        let resource = store.with_memory_reserver(pool.clone()).unwrap();
        assert_eq!(
            store.exists(&path).await.unwrap(),
            resource.exists(&path).await.unwrap()
        );
        let ordinary = store.size(&path).await;
        let charged = resource.size(&path).await;
        if kind == "file" {
            assert_eq!(ordinary.unwrap(), charged.unwrap());
        } else {
            assert!(ordinary.unwrap_err().is_not_found());
            assert!(charged.unwrap_err().is_not_found());
        }
        drop(resource);
        assert_eq!(pool.used.load(Ordering::Acquire), 0);
    }

    #[tokio::test]
    async fn resource_conditional_update_refuses_before_output() {
        let directory = tempfile::tempdir().unwrap();
        let path = Path::from_absolute_path(directory.path().join("output")).unwrap();
        let pool = Arc::new(ReadPool {
            limit: 0,
            ..Default::default()
        });
        let error = ObjectStore::local()
            .create_with_resources_mode(
                &path,
                pool.clone(),
                object_store::PutMode::Update(object_store::UpdateVersion {
                    e_tag: None,
                    version: None,
                }),
                None,
            )
            .await
            .err()
            .unwrap();
        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(error.to_string().contains("PutMode::Update"));
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
        assert_eq!(pool.used.load(Ordering::Acquire), 0);
    }

    #[cfg(unix)]
    #[rstest::rstest]
    #[tokio::test]
    async fn resource_metadata_follows_symlinks(#[values(false, true)] is_directory: bool) {
        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("target");
        if is_directory {
            std::fs::create_dir(&target).unwrap();
        } else {
            std::fs::write(&target, b"data").unwrap();
        }
        let link = directory.path().join("link");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let path = Path::from_absolute_path(&link).unwrap();
        let pool = Arc::new(ReadPool {
            limit: 256 * 1024,
            ..Default::default()
        });
        let store = ObjectStore::local();
        let resource = store.with_memory_reserver(pool.clone()).unwrap();
        assert_eq!(
            store.exists(&path).await.unwrap(),
            resource.exists(&path).await.unwrap()
        );
        assert_eq!(resource.exists(&path).await.unwrap(), !is_directory);
        assert_eq!(pool.used.load(Ordering::Acquire), 0);
    }

    #[tokio::test]
    async fn resource_local_output_failure_preserves_existing_file() {
        let directory = tempfile::tempdir().unwrap();
        let local = directory.path().join("output");
        std::fs::write(&local, b"original").unwrap();
        let path = Path::from_absolute_path(&local).unwrap();
        let used = Arc::new(AtomicUsize::new(0));
        let pool = Arc::new(ReadPool {
            used: used.clone(),
            limit: 16 * 1024,
            ..Default::default()
        });
        let mut writer = ObjectStore::local()
            .create_with_memory_reserver(&path, pool)
            .await
            .unwrap();
        let error = AsyncWriteExt::write_all(writer.as_mut(), &[7; 8192])
            .await
            .unwrap_err();
        assert!(error.to_string().contains("local output write buffer"));
        assert!(
            Writer::shutdown(writer.as_mut())
                .await
                .unwrap_err()
                .to_string()
                .contains("failed or closed")
        );
        drop(writer);
        assert_eq!(std::fs::read(&local).unwrap(), b"original");
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
        assert_eq!(used.load(Ordering::Acquire), 0);
    }

    #[rstest::rstest]
    #[case::write(false)]
    #[case::publication(true)]
    #[test]
    fn resource_local_output_cancelled_io_retains_charge_and_staging_file(
        #[case] publishing: bool,
    ) {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .max_blocking_threads(1)
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let directory = tempfile::tempdir().unwrap();
            let local = directory.path().join("output");
            let path = Path::from_absolute_path(&local).unwrap();
            let used = Arc::new(AtomicUsize::new(0));
            let pool = Arc::new(ReadPool {
                used: used.clone(),
                limit: 64 * 1024,
                ..Default::default()
            });
            let tasks = pool.tasks.clone();
            let mut writer = ObjectStore::local()
                .create_with_memory_reserver(&path, pool)
                .await
                .unwrap();
            if publishing {
                AsyncWriteExt::write_all(writer.as_mut(), &[7; 128])
                    .await
                    .unwrap();
                AsyncWriteExt::flush(writer.as_mut()).await.unwrap();
            }
            let baseline = used.load(Ordering::Acquire);
            let (started_tx, started_rx) = tokio::sync::oneshot::channel();
            let (release_tx, release_rx) = std::sync::mpsc::channel();
            let blocker = tokio::task::spawn_blocking(move || {
                started_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            });
            started_rx.await.unwrap();
            let mut action: std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + '_>> =
                if publishing {
                    Box::pin(async { Writer::shutdown(writer.as_mut()).await.map(|_| ()) })
                } else {
                    Box::pin(async {
                        AsyncWriteExt::write_all(writer.as_mut(), &[7; 128]).await?;
                        AsyncWriteExt::flush(writer.as_mut())
                            .await
                            .map_err(Error::from)
                    })
                };
            assert!(futures::poll!(&mut action).is_pending());
            drop(action);
            drop(writer);
            assert!(used.load(Ordering::Acquire) > baseline);
            assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
            assert!(!local.exists());
            release_tx.send(()).unwrap();
            blocker.await.unwrap();
            tasks.wait().await;
            assert_eq!(used.load(Ordering::Acquire), 0);
            assert_eq!(
                std::fs::read_dir(directory.path()).unwrap().count(),
                usize::from(publishing)
            );
            if publishing {
                assert_eq!(std::fs::read(&local).unwrap(), [7; 128]);
            }
        });
    }

    #[rstest::rstest]
    #[test]
    fn resource_local_output_cancelled_write_can_reuse_writer(
        #[values(4, 70_000)] input_bytes: usize,
    ) {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .max_blocking_threads(1)
            .build()
            .unwrap();
        runtime.block_on(async {
            let directory = tempfile::tempdir().unwrap();
            let local = directory.path().join("output");
            let path = Path::from_absolute_path(&local).unwrap();
            let pool = Arc::new(ReadPool {
                limit: 256 * 1024,
                ..Default::default()
            });
            let used = pool.used.clone();
            let mut writer = ObjectStore::local()
                .create_with_memory_reserver(&path, pool)
                .await
                .unwrap();
            let baseline = used.load(Ordering::Acquire);
            let (started, running) = tokio::sync::oneshot::channel();
            let (release, blocked) = std::sync::mpsc::channel();
            let blocker = tokio::task::spawn_blocking(move || {
                started.send(()).unwrap();
                blocked.recv().unwrap();
            });
            running.await.unwrap();
            let input = vec![b'A'; input_bytes];
            let accepted = {
                let mut write = Box::pin(writer.write(&input));
                let std::task::Poll::Ready(result) = futures::poll!(write.as_mut()) else {
                    panic!("an admitted buffer must be acknowledged before I/O completes");
                };
                result.unwrap()
            };
            assert_eq!(accepted, input_bytes.min(64 * 1024));
            assert_eq!(writer.tell().await.unwrap(), accepted);
            let mut cancelled = Box::pin(writer.write(b"discarded"));
            assert!(futures::poll!(cancelled.as_mut()).is_pending());
            drop(cancelled);
            assert!(used.load(Ordering::Acquire) > baseline);
            release.send(()).unwrap();
            blocker.await.unwrap();
            assert_eq!(writer.write(b"BBBB").await.unwrap(), 4);
            let result = Writer::shutdown(writer.as_mut()).await.unwrap();
            assert_eq!(result.size, accepted + 4);
            let mut expected = input[..accepted].to_vec();
            expected.extend_from_slice(b"BBBB");
            assert_eq!(std::fs::read(local).unwrap(), expected);
            drop(writer);
            assert_eq!(used.load(Ordering::Acquire), 0);
        });
    }

    #[tokio::test]
    async fn resource_store_view_tracks_metadata_and_buffers() {
        let directory = tempfile::tempdir().unwrap();
        let path = Path::from_absolute_path(directory.path().join("output")).unwrap();
        let missing = Path::from_absolute_path(directory.path().join("missing")).unwrap();
        let pool = Arc::new(ReadPool {
            limit: 128 * 1024,
            ..Default::default()
        });
        let store = ObjectStore::local()
            .with_memory_reserver(pool.clone())
            .unwrap();
        assert!(!store.exists(&missing).await.unwrap());
        store.put(&path, &[7_u8; 64]).await.unwrap();
        assert!(store.exists(&path).await.unwrap());
        assert_eq!(store.size(&path).await.unwrap(), 64);
        let reader = store.open_with_size(&path, 64).await.unwrap();
        let bytes = reader.get_all().await.unwrap();
        drop(reader);
        assert_eq!(bytes.as_ref(), &[7_u8; 64]);
        assert!(pool.used.load(Ordering::Acquire) >= 64 + 1024);
        drop(bytes);
        assert_eq!(pool.used.load(Ordering::Acquire), 0);
        let error = ObjectStore::memory()
            .with_memory_reserver(pool)
            .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        assert!(error.to_string().contains("direct local storage"));
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn resource_read_scheduler_retains_slices(#[values(false, true)] lite: bool) {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), [7_u8; 64]).unwrap();
        let used = Arc::new(AtomicUsize::new(0));
        let pool = Arc::new(ReadPool {
            used: used.clone(),
            limit: 16 * 1024,
            ..Default::default()
        });
        let mut config = SchedulerConfig::new(4096);
        config.use_lite_scheduler = Some(lite);
        let scheduler = ScanScheduler::new(Arc::new(ObjectStore::local()), config);
        // Memory observation retains a weak reference before admission is configured.
        let observed = Arc::downgrade(&scheduler);
        let scheduler = scheduler.with_memory_reserver(pool).unwrap();
        assert!(Arc::ptr_eq(&scheduler, &observed.upgrade().unwrap()));
        let reader = scheduler
            .open_file(
                &Path::from_filesystem_path(file.path()).unwrap(),
                &CachedFileSize::unknown(),
            )
            .await
            .unwrap();
        let mut bytes = reader.submit_request(vec![0..4, 8..12], 0).await.unwrap();
        assert_eq!(bytes.len(), 2);
        assert!(used.load(Ordering::Acquire) >= 12 + 1024);
        drop(bytes.remove(0));
        let remaining = bytes.pop().unwrap();
        assert_eq!(remaining.as_ref(), &[7_u8; 4]);
        drop(reader);
        drop(scheduler);
        assert!(used.load(Ordering::Acquire) >= 12 + 1024);
        drop(remaining);
        assert_eq!(used.load(Ordering::Acquire), 0);
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn resource_read_scheduler_rejects_reconfiguration(
        #[values(false, true)] lite: bool,
        #[values(false, true)] configured: bool,
    ) {
        let file = tempfile::NamedTempFile::new().unwrap();
        let pool = Arc::new(ReadPool {
            limit: 16 * 1024,
            ..Default::default()
        });
        let mut config = SchedulerConfig::new(4096);
        config.use_lite_scheduler = Some(lite);
        let scheduler = ScanScheduler::new(Arc::new(ObjectStore::local()), config);
        let scheduler = if configured {
            scheduler.with_memory_reserver(pool.clone()).unwrap()
        } else {
            let _reader = scheduler
                .open_file(
                    &Path::from_filesystem_path(file.path()).unwrap(),
                    &CachedFileSize::unknown(),
                )
                .await
                .unwrap();
            scheduler
        };
        let error = scheduler.with_memory_reserver(pool).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error.to_string().contains(
                "must be configured before opening files and can only be configured once"
            )
        );
    }

    #[rstest::rstest]
    #[case::range(0)]
    #[case::whole_file(1)]
    #[case::range_stream(2)]
    #[case::whole_stream(3)]
    #[tokio::test]
    async fn resource_read_retains_full_parent_charge(#[case] mode: usize) {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), [7_u8; 64]).unwrap();
        let used = Arc::new(AtomicUsize::new(0));
        let pool = Arc::new(ReadPool {
            used: used.clone(),
            limit: 8192,
            ..Default::default()
        });
        let mut reader = LocalObjectReader::open_local_path(file.path(), 64, None)
            .await
            .unwrap();
        reader.configure_memory_reserver(pool).unwrap();
        let bytes = match mode {
            0 => reader.get_range(0..64).await.unwrap(),
            1 => reader.get_all().await.unwrap(),
            2 => reader
                .get_range_stream(0..64)
                .await
                .unwrap()
                .try_next()
                .await
                .unwrap()
                .unwrap(),
            _ => reader
                .get_stream()
                .await
                .unwrap()
                .try_next()
                .await
                .unwrap()
                .unwrap(),
        };
        assert_eq!(bytes.as_ref(), &[7_u8; 64]);
        assert_eq!(used.load(Ordering::Acquire), 64 + 1024);
        let slice = bytes.slice(1..2);
        drop(bytes);
        drop(reader);
        assert_eq!(used.load(Ordering::Acquire), 64 + 1024);
        assert_eq!(slice.as_ref(), &[7]);
        drop(slice);
        assert_eq!(used.load(Ordering::Acquire), 0);
    }

    #[rstest::rstest]
    #[case::cancel_before_poll(0)]
    #[case::unexpected_eof(1)]
    #[case::denied_before_io(2)]
    #[case::overflow(3)]
    #[case::reversed(4)]
    #[tokio::test]
    async fn resource_read_errors_and_cancellation_release_charge(#[case] mode: usize) {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), [7_u8; 64]).unwrap();
        let used = Arc::new(AtomicUsize::new(0));
        let pool = Arc::new(ReadPool {
            used: used.clone(),
            limit: if mode == 2 { 1 } else { 8192 },
            ..Default::default()
        });
        let mut reader = LocalObjectReader::open_local_path(file.path(), 64, None)
            .await
            .unwrap();
        reader.configure_memory_reserver(pool).unwrap();
        let range = match mode {
            3 => 0..usize::MAX,
            4 => Range { start: 2, end: 1 },
            _ => 0..128,
        };
        let read = reader.get_range(range);
        if mode == 0 {
            assert_eq!(used.load(Ordering::Acquire), 128 + 1024);
            drop(read);
        } else {
            let error = read.await.unwrap_err();
            if mode == 1 {
                assert!(error.to_string().contains("fill whole buffer"));
            } else {
                let object_store::Error::Generic { source, .. } = error else {
                    panic!("expected Lance resource error")
                };
                let error = source.downcast_ref::<Error>().unwrap();
                if mode == 2 {
                    assert!(matches!(error, Error::Execution { .. }));
                    assert!(error.to_string().contains("local read buffer"));
                } else {
                    assert!(matches!(error, Error::InvalidInput { .. }));
                }
            }
        }
        assert_eq!(used.load(Ordering::Acquire), 0);
    }
}
