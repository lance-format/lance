// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reclaimable scratch storage.
//!
//! A [`SpillStore`] hands out scratch space for temporary state that is too
//! large to keep in memory and is read back later in the same process (for
//! example, posting lists or shuffle runs accumulated while building an index).
//! The backing storage is reclaimed automatically when the handle is dropped.
//!
//! [`SpillStore::new_spill`] returns a [`Writer`] paired with a [`Spill`]
//! handle: the writer is the byte sink (feed it to `FileWriter::try_new`, or
//! write to it directly); the [`Spill`] reads the bytes back (via
//! [`crate::scheduler::ScanScheduler::open_reader`] for a v2 `FileReader`) and
//! owns the file's lifetime.
//!
//! # Lifecycle
//!
//! - **Write-once.** The only way to obtain a writer is `new_spill`, and each
//!   call allocates a fresh unit of storage, so a single spill cannot be
//!   written twice — there is no second-writer path to guard against.
//! - **Write-before-read.** [`Spill::reader`] fails until the writer has been
//!   shut down, so partially written bytes are never read back.
//! - **RAII.** The last writer, spill, reader, or read operation reclaims the
//!   file and its disk quota. The temp directory is a cleanup backstop.
//!
//! # Disk cap
//!
//! [`LocalSpillStore::with_cap`] enforces a shared byte budget with typed
//! [`lance_core::Error::DiskCapExceeded`] errors. Writes reserve before accepting
//! bytes and roll back unaccepted bytes. Accepted bytes remain charged until
//! deletion succeeds, including when a writer is aborted. Failed deletions are
//! retried on the next spill allocation and on directory cleanup.

use std::fs::File;
use std::future::Future;
use std::io::{self, Write};
use std::ops::Range;
#[cfg(unix)]
use std::os::unix::fs::{DirBuilderExt, FileExt};
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures::{StreamExt, future::BoxFuture};
use lance_core::deepsize::{Context as DeepSizeContext, DeepSizeOf};
use object_store::path::Path;
use tokio::io::AsyncWrite;

use lance_core::{Error, Result};

use crate::object_writer::WriteResult;
use crate::traits::{ByteStream, Reader, Writer};

/// A factory for scratch storage.
///
/// The trait is object-safe and `Send + Sync` so it can be held behind an
/// `Arc<dyn SpillStore>` (e.g. inside a `Session`). Implementations need not be
/// backed by local files (e.g. in-memory buffers, remote object stores).
#[async_trait]
pub trait SpillStore: Send + Sync + 'static {
    /// Allocate a unit of scratch storage.
    ///
    /// Returns the byte sink to write it with and a [`Spill`] handle to read it
    /// back. For a capped store, writes that would exceed the cap fail with
    /// [`lance_core::Error::DiskCapExceeded`]. The storage is reclaimed when the
    /// last backing owner is dropped.
    async fn new_spill(&self) -> Result<(Box<dyn Writer>, Box<dyn Spill>)>;
}

/// The readable half of a spill, and the owner of its backing storage.
///
/// Storage is reclaimed after it and all writers/readers have been dropped.
/// The trait is object-safe so it can be returned as `Box<dyn Spill>` from [`SpillStore::new_spill`].
#[async_trait]
pub trait Spill: Send + Sync {
    /// Open a reader over the spilled bytes.
    ///
    /// Fails until the paired writer has been shut down, since the bytes are not
    /// complete before then.
    async fn reader(&self) -> Result<Box<dyn Reader>>;
}

/// A shared, cloneable byte budget.
///
/// Cloning produces another handle to the *same* underlying counter, so a quota
/// shared across many writers enforces a single combined cap.
#[derive(Debug, Clone)]
struct DiskQuota {
    cap_bytes: u64,
    used: Arc<AtomicU64>,
}

impl DiskQuota {
    fn new(cap_bytes: u64) -> Self {
        Self {
            cap_bytes,
            used: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Try to reserve `n` bytes, failing with [`Error::DiskCapExceeded`] if the
    /// reservation would push total usage past the cap.
    fn try_reserve(&self, n: u64) -> Result<()> {
        self.used
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
                used.checked_add(n).filter(|next| *next <= self.cap_bytes)
            })
            .map(|_| ())
            .map_err(|used| Error::disk_cap_exceeded(self.cap_bytes, used))
    }

    /// Release `n` previously reserved bytes back to the budget.
    fn release(&self, n: u64) {
        let previous = self.used.fetch_sub(n, Ordering::Relaxed);
        debug_assert!(
            previous >= n,
            "spill quota released more bytes than reserved"
        );
    }
}

/// The byte sink handed out by [`SpillStore::new_spill`].
///
/// It optionally reserves a [`DiskQuota`] as bytes are written (keeping cap
/// enforcement inside the spill store rather than in [`ObjectStore`], and
/// working for any backend the store opens), and flips a shared `finished` flag
/// on shutdown so the paired [`Spill`] knows the bytes are complete.
struct SpillWriter {
    inner: Box<dyn Writer>,
    backing: Arc<SpillBacking>,
}

impl AsyncWrite for SpillWriter {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        if this.backing.finished.load(Ordering::Relaxed) {
            return Poll::Ready(Err(io::Error::other(
                "spill write requested after shutdown",
            )));
        }
        let Some(quota) = &this.backing.quota else {
            return Pin::new(this.inner.as_mut()).poll_write(cx, buf);
        };
        // Reserve up-front for the bytes we intend to write, then release the
        // remainder the inner writer did not accept so the reservation tracks
        // bytes actually buffered (and, for a write-once file, the file size).
        if let Err(e) = quota.try_reserve(buf.len() as u64) {
            return Poll::Ready(Err(io::Error::other(e)));
        }
        let poll = Pin::new(this.inner.as_mut()).poll_write(cx, buf);
        match &poll {
            Poll::Ready(Ok(n)) => {
                quota.release((buf.len() - *n) as u64);
                this.backing
                    .accepted
                    .fetch_add(*n as u64, Ordering::Relaxed);
            }
            _ => quota.release(buf.len() as u64),
        }
        poll
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(self.get_mut().inner.as_mut()).poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let poll = Pin::new(this.inner.as_mut()).poll_shutdown(cx);
        if matches!(poll, Poll::Ready(Ok(()))) {
            // Mirrors `Writer::shutdown` so the flag is set whichever shutdown
            // surface the consumer drives (`AsyncWrite` vs the `Writer` trait).
            this.backing.finished.store(true, Ordering::Relaxed);
        }
        poll
    }
}

#[async_trait]
impl Writer for SpillWriter {
    async fn tell(&mut self) -> Result<usize> {
        self.inner.tell().await
    }

    async fn shutdown(&mut self) -> Result<WriteResult> {
        let result = self.inner.shutdown().await?;
        // Signal the paired `Spill` that the bytes are now complete. `Relaxed`
        // is sufficient: this only flags that shutdown happened; the file
        // contents are synchronized through the filesystem, not this flag.
        self.backing.finished.store(true, Ordering::Relaxed);
        Ok(result)
    }
}

/// A [`SpillStore`] that writes temporary files to a local temp directory.
///
/// By default there is no disk cap. Use [`LocalSpillStore::with_cap`] to
/// configure one shared across every handle this store produces.
///
/// The scratch directory is created on the first [`SpillStore::new_spill`]
/// call, not at construction, so opening a dataset does not require a writable
/// temp directory. It is deleted when the store is dropped, cleaning up any
/// files whose handles have already been dropped.
pub struct LocalSpillStore {
    /// Created on first spill. Backstop cleanup removes the directory on drop.
    temp_dir: Mutex<Option<Arc<SpillDirectory>>>,
    file_counter: Arc<AtomicU64>,
    /// Byte budget shared across every handle, enforced while writing.
    quota: Option<DiskQuota>,
}

/// Create a unique scratch directory, retrying `PermissionDenied`.
///
/// Windows GitHub runners (and machines with antivirus) can reject a newly
/// created `%TEMP%` subdirectory with `ERROR_ACCESS_DENIED`. Each attempt uses
/// a fresh random name, so a short retry loop is the usual workaround.
fn create_temp_dir() -> Result<tempfile::TempDir> {
    const ATTEMPTS: u32 = 5;
    let mut attempt = 0;
    loop {
        match tempfile::tempdir() {
            Ok(dir) => return Ok(dir),
            Err(err) if err.kind() == io::ErrorKind::PermissionDenied && attempt + 1 < ATTEMPTS => {
                attempt += 1;
                std::thread::sleep(Duration::from_millis(10 * attempt as u64));
            }
            Err(err) => return Err(err.into()),
        }
    }
}

impl LocalSpillStore {
    /// Create a store with no disk cap.
    ///
    /// Construction does not touch the filesystem; the scratch directory is
    /// created on the first [`SpillStore::new_spill`] call.
    pub fn new() -> Result<Self> {
        Ok(Self::default())
    }

    /// Create a store that returns [`lance_core::Error::DiskCapExceeded`] once
    /// total bytes written across all live handles would exceed `cap_bytes`.
    pub fn with_cap(cap_bytes: u64) -> Result<Self> {
        Ok(Self {
            quota: Some(DiskQuota::new(cap_bytes)),
            ..Self::default()
        })
    }

    fn ensure_temp_dir(&self) -> Result<Arc<SpillDirectory>> {
        // The lock is held only for init (and the rare PermissionDenied retry)
        // and never across an `.await`.
        let mut slot = self.temp_dir.lock().unwrap_or_else(|err| err.into_inner());
        if let Some(dir) = slot.as_ref() {
            dir.retry_cleanup();
            return Ok(dir.clone());
        }
        let directory = create_temp_dir()?;
        let dir = Arc::new(SpillDirectory {
            path: directory.path().to_owned(),
            directory: Some(directory),
            pending: Mutex::new(Vec::new()),
        });
        *slot = Some(dir.clone());
        Ok(dir)
    }
}

impl Default for LocalSpillStore {
    fn default() -> Self {
        Self {
            temp_dir: Mutex::new(None),
            file_counter: Arc::new(AtomicU64::new(0)),
            quota: None,
        }
    }
}

#[async_trait]
impl SpillStore for LocalSpillStore {
    async fn new_spill(&self) -> Result<(Box<dyn Writer>, Box<dyn Spill>)> {
        let temp_dir = self.ensure_temp_dir()?;
        let idx = self
            .file_counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| Error::invalid_input("spill file counter overflow"))?;
        let fs_path = temp_dir.path().join(format!("spill_{idx:06}.bin"));
        let os_path = Path::from_absolute_path(&fs_path)?;
        let backing = Arc::new(SpillBacking {
            fs_path,
            quota: self.quota.clone(),
            accepted: AtomicU64::new(0),
            finished: AtomicBool::new(false),
            directory: temp_dir,
        });
        let write_backing = backing.clone();
        let inner = tokio::task::spawn_blocking(move || {
            // Recover if external temporary-directory cleanup removed the cached directory.
            let directory_path = write_backing.directory.path();
            #[cfg(unix)]
            let create_directory = std::fs::DirBuilder::new()
                .recursive(true)
                .mode(0o700)
                .create(directory_path);
            #[cfg(not(unix))]
            let create_directory = std::fs::create_dir_all(directory_path);
            create_directory.map_err(|err| {
                io::Error::new(
                    err.kind(),
                    format!(
                        "failed to create spill directory {} for spill {}: {err}",
                        directory_path.display(),
                        write_backing.fs_path.display(),
                    ),
                )
            })?;
            let file = File::create(&write_backing.fs_path).map_err(|err| {
                io::Error::new(
                    err.kind(),
                    format!(
                        "failed to create spill file {}: {err}",
                        write_backing.fs_path.display(),
                    ),
                )
            })?;
            Ok::<_, io::Error>(LocalSpillWriter {
                file: Arc::new(SpillWriteFile {
                    file,
                    _backing: write_backing,
                }),
                pending: None,
                position: 0,
            })
        })
        .await??;
        let writer = Box::new(SpillWriter {
            inner: Box::new(inner),
            backing: backing.clone(),
        });
        let spill = Box::new(LocalSpill { os_path, backing });
        Ok((writer, spill))
    }
}

struct SpillWriteFile {
    file: File,
    _backing: Arc<SpillBacking>,
}

/// One outstanding local write, with backing ownership inside the blocking
/// task. Tokio's file writer cannot attach our quota to its detached I/O task.
struct LocalSpillWriter {
    file: Arc<SpillWriteFile>,
    pending: Option<tokio::task::JoinHandle<io::Result<()>>>,
    position: usize,
}

impl AsyncWrite for LocalSpillWriter {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        match self.as_mut().poll_flush(cx) {
            Poll::Ready(Ok(())) => {}
            other => return other.map_ok(|_| 0),
        }
        let Some(position) = self.position.checked_add(buf.len()) else {
            return Poll::Ready(Err(io::Error::other("spill writer position overflow")));
        };
        let bytes = buf.to_vec();
        let file = self.file.clone();
        self.pending = Some(tokio::task::spawn_blocking(move || {
            let mut handle = &file.file;
            handle.write_all(&bytes)
        }));
        self.position = position;
        Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let Some(task) = self.pending.as_mut() else {
            return Poll::Ready(Ok(()));
        };
        let result = match Pin::new(task).poll(cx) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(result) => result,
        };
        self.pending = None;
        Poll::Ready(result.map_err(io::Error::other).and_then(|result| result))
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.poll_flush(cx)
    }
}

#[async_trait]
impl Writer for LocalSpillWriter {
    async fn tell(&mut self) -> Result<usize> {
        Ok(self.position)
    }

    async fn shutdown(&mut self) -> Result<WriteResult> {
        tokio::io::AsyncWriteExt::flush(self).await?;
        Ok(WriteResult {
            size: self.position,
            e_tag: None,
        })
    }
}

#[derive(Debug)]
struct PendingCleanup {
    path: PathBuf,
    quota: Option<DiskQuota>,
    bytes: u64,
}

impl PendingCleanup {
    fn try_cleanup(&self) -> bool {
        match std::fs::remove_file(&self.path) {
            Ok(()) => {}
            Err(err) if err.kind() == io::ErrorKind::NotFound => {}
            Err(err) => {
                tracing::warn!(path = %self.path.display(), error = %err, "spill cleanup failed; retaining disk quota");
                return false;
            }
        }
        if let Some(quota) = &self.quota {
            quota.release(self.bytes);
        }
        true
    }
}

#[derive(Debug)]
struct SpillDirectory {
    path: PathBuf,
    directory: Option<tempfile::TempDir>,
    pending: Mutex<Vec<PendingCleanup>>,
}

impl SpillDirectory {
    fn path(&self) -> &std::path::Path {
        &self.path
    }

    fn retry_cleanup(&self) {
        self.pending
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .retain(|cleanup| !cleanup.try_cleanup());
    }
}

impl Drop for SpillDirectory {
    fn drop(&mut self) {
        if let Some(directory) = self.directory.take() {
            let path = directory.path().to_owned();
            match directory.close() {
                Ok(()) => {
                    for cleanup in self
                        .pending
                        .get_mut()
                        .unwrap_or_else(|err| err.into_inner())
                        .drain(..)
                    {
                        if let Some(quota) = cleanup.quota {
                            quota.release(cleanup.bytes);
                        }
                    }
                }
                Err(err) => {
                    tracing::warn!(path = %path.display(), error = %err, "spill directory cleanup failed")
                }
            }
        }
    }
}

#[derive(Debug)]
struct SpillBacking {
    fs_path: PathBuf,
    quota: Option<DiskQuota>,
    accepted: AtomicU64,
    finished: AtomicBool,
    directory: Arc<SpillDirectory>,
}

impl Drop for SpillBacking {
    fn drop(&mut self) {
        let cleanup = PendingCleanup {
            path: self.fs_path.clone(),
            quota: self.quota.clone(),
            bytes: self.accepted.load(Ordering::Relaxed),
        };
        if !cleanup.try_cleanup() {
            self.directory
                .pending
                .lock()
                .unwrap_or_else(|err| err.into_inner())
                .push(cleanup);
        }
    }
}

struct LocalSpill {
    os_path: Path,
    backing: Arc<SpillBacking>,
}

#[async_trait]
impl Spill for LocalSpill {
    async fn reader(&self) -> Result<Box<dyn Reader>> {
        if !self.backing.finished.load(Ordering::Relaxed) {
            return Err(Error::invalid_input(
                "spill reader requested before the writer was shut down",
            ));
        }
        let backing = self.backing.clone();
        let path = self.os_path.clone();
        // The blocking task owns the backing even if opening is cancelled.
        tokio::task::spawn_blocking(move || {
            let file = File::open(&backing.fs_path)?;
            let size = usize::try_from(file.metadata()?.len())
                .map_err(|err| Error::invalid_input(format!("spill size exceeds usize: {err}")))?;
            Ok(Box::new(SpillReader {
                file: Arc::new(SpillReadFile {
                    file: Arc::new(file),
                    _backing: backing,
                }),
                path,
                size,
            }) as Box<dyn Reader>)
        })
        .await?
    }
}

#[derive(Debug)]
struct SpillReadFile {
    // Close the OS file before releasing its backing (required on Windows).
    file: Arc<File>,
    _backing: Arc<SpillBacking>,
}

impl SpillReadFile {
    fn read_range(
        self: Arc<Self>,
        range: Range<usize>,
    ) -> BoxFuture<'static, object_store::Result<Bytes>> {
        Box::pin(async move {
            crate::local::join_local_io(tokio::task::spawn_blocking(move || {
                // Keep both the file and quota in the blocking closure, whose
                // lifetime can exceed the future awaiting its result.
                let mut bytes = vec![0; range.len()];
                #[cfg(unix)]
                self.file.read_exact_at(&mut bytes, range.start as u64)?;
                #[cfg(windows)]
                crate::local::read_exact_at(self.file.clone(), &mut bytes, range.start as u64)?;
                Ok(Bytes::from(bytes))
            }))
            .await
        })
    }

    fn stream(self: Arc<Self>, range: Range<usize>, chunk_size: usize) -> ByteStream {
        futures::stream::try_unfold((self, range), move |(file, range)| async move {
            if range.is_empty() {
                return Ok(None);
            }
            let end = range.start.saturating_add(chunk_size).min(range.end);
            let bytes = file.clone().read_range(range.start..end).await?;
            Ok(Some((bytes, (file, end..range.end))))
        })
        .boxed()
    }
}

#[derive(Debug)]
struct SpillReader {
    file: Arc<SpillReadFile>,
    path: Path,
    size: usize,
}

impl DeepSizeOf for SpillReader {
    fn deep_size_of_children(&self, context: &mut DeepSizeContext) -> usize {
        // File and quota ownership do not include the file's on-disk contents.
        self.path.as_ref().deep_size_of_children(context)
    }
}

impl Reader for SpillReader {
    fn path(&self) -> &Path {
        &self.path
    }

    fn block_size(&self) -> usize {
        4 * 1024
    }

    fn io_parallelism(&self) -> usize {
        crate::object_store::DEFAULT_LOCAL_IO_PARALLELISM
    }

    fn size(&self) -> BoxFuture<'_, object_store::Result<usize>> {
        Box::pin(async move { Ok(self.size) })
    }

    fn get_range(&self, range: Range<usize>) -> BoxFuture<'static, object_store::Result<Bytes>> {
        self.file.clone().read_range(range)
    }

    fn get_all(&self) -> BoxFuture<'_, object_store::Result<Bytes>> {
        self.get_range(0..self.size)
    }

    fn get_stream(&self) -> BoxFuture<'_, object_store::Result<ByteStream>> {
        self.get_range_stream(0..self.size)
    }

    fn get_range_stream(
        &self,
        range: Range<usize>,
    ) -> BoxFuture<'_, object_store::Result<ByteStream>> {
        let stream = self
            .file
            .clone()
            .stream(range, self.block_size().max(8 * 1024));
        Box::pin(async move { Ok(stream) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object_store::ObjectStore;
    use rstest::rstest;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use tokio::io::AsyncWriteExt;

    /// Write `data` to a fresh writer and shut it down.
    async fn finish_writer(mut writer: Box<dyn Writer>, data: &[u8]) -> Result<()> {
        writer.write_all(data).await?;
        Writer::shutdown(writer.as_mut()).await?;
        Ok(())
    }

    #[test]
    fn test_disk_quota_reserve_release() {
        let quota = DiskQuota::new(100);
        quota.try_reserve(60).unwrap();
        assert!(quota.try_reserve(60).is_err());
        quota.release(60);
        quota.try_reserve(60).unwrap();
        // Reserving exactly up to the cap succeeds; one byte past it fails.
        quota.try_reserve(40).unwrap();
        assert!(quota.try_reserve(1).is_err());
    }

    #[tokio::test]
    async fn test_write_then_read() {
        let store = LocalSpillStore::new().unwrap();
        let (writer, spill) = store.new_spill().await.unwrap();

        let data = b"hello spill world";
        finish_writer(writer, data).await.unwrap();

        let reader = spill.reader().await.unwrap();
        let read_back = reader.get_all().await.unwrap();
        assert_eq!(read_back.as_ref(), data);
    }

    #[rstest]
    #[case::uncapped(false)]
    #[case::capped(true)]
    #[tokio::test]
    async fn test_reader_requires_finished_writer(#[case] is_capped: bool) {
        let store = if is_capped {
            LocalSpillStore::with_cap(100).unwrap()
        } else {
            LocalSpillStore::new().unwrap()
        };
        let (mut writer, spill) = store.new_spill().await.unwrap();
        writer.write_all(b"partial").await.unwrap();

        // Reading before the writer is shut down is rejected.
        let Err(err) = spill.reader().await else {
            panic!("reader before shutdown should be rejected");
        };
        assert!(
            matches!(err, Error::InvalidInput { .. }),
            "expected InvalidInput, got {err:?}"
        );

        // After shutdown the reader sees the bytes.
        Writer::shutdown(writer.as_mut()).await.unwrap();
        let err = writer.write_all(b"suffix").await.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Other);
        assert_eq!(err.to_string(), "spill write requested after shutdown");
        assert_eq!(writer.tell().await.unwrap(), 7);
        if let Some(quota) = &store.quota {
            assert_eq!(quota.used.load(Ordering::Relaxed), 7);
        }

        let reader = spill.reader().await.unwrap();
        assert_eq!(reader.get_all().await.unwrap().as_ref(), b"partial");
    }

    #[rstest]
    #[case::uncapped(false)]
    #[case::capped(true)]
    #[tokio::test]
    async fn test_reader_ready_after_async_shutdown(#[case] is_capped: bool) {
        // Shutting down through the `AsyncWrite` surface (not the `Writer`
        // trait) must also mark the spill readable — covers poll_shutdown's
        // flag set, the path the `Writer::shutdown` tests don't reach.
        let store = if is_capped {
            LocalSpillStore::with_cap(100).unwrap()
        } else {
            LocalSpillStore::new().unwrap()
        };
        let (mut writer, spill) = store.new_spill().await.unwrap();
        writer.write_all(b"async").await.unwrap();
        AsyncWriteExt::shutdown(&mut writer).await.unwrap();
        let err = writer.write_all(b"suffix").await.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Other);
        assert_eq!(err.to_string(), "spill write requested after shutdown");
        assert_eq!(writer.tell().await.unwrap(), 5);
        if let Some(quota) = &store.quota {
            assert_eq!(quota.used.load(Ordering::Relaxed), 5);
        }

        let reader = spill.reader().await.unwrap();
        assert_eq!(reader.get_all().await.unwrap().as_ref(), b"async");
    }

    #[tokio::test]
    async fn test_empty_spill() {
        // A spill written with no bytes round-trips empty, and the capped path
        // handles the zero-byte reserve/stat without error.
        let store = LocalSpillStore::with_cap(100).unwrap();
        let (writer, spill) = store.new_spill().await.unwrap();
        finish_writer(writer, b"").await.unwrap();

        let reader = spill.reader().await.unwrap();
        assert!(reader.get_all().await.unwrap().is_empty());
    }

    #[test]
    fn test_new_does_not_create_temp_dir() {
        // Session construction must not touch the filesystem; Windows CI panics
        // when thousands of tests each eagerly create a `%TEMP%` subdirectory.
        let store = LocalSpillStore::new().unwrap();
        assert!(
            store.temp_dir.lock().unwrap().is_none(),
            "temp directory should be created lazily on first spill"
        );
        assert!(
            LocalSpillStore::default()
                .temp_dir
                .lock()
                .unwrap()
                .is_none(),
            "Default should also defer temp directory creation"
        );
    }

    #[tokio::test]
    async fn test_raii_cleanup() {
        let store = LocalSpillStore::new().unwrap();
        let (writer, spill) = store.new_spill().await.unwrap();
        finish_writer(writer, b"some bytes").await.unwrap();

        // The first spill gets a deterministic name under the store's temp dir.
        let path = store
            .temp_dir
            .lock()
            .unwrap()
            .as_ref()
            .expect("new_spill creates the temp dir")
            .path()
            .join("spill_000000.bin");
        assert!(path.exists());
        drop(spill);
        assert!(!path.exists(), "spill file should be deleted on drop");
    }

    #[rstest]
    #[case::uncapped(false)]
    #[case::capped(true)]
    #[tokio::test]
    async fn test_new_spill_recreates_missing_directory(#[case] is_capped: bool) {
        let store = if is_capped {
            LocalSpillStore::with_cap(100).unwrap()
        } else {
            LocalSpillStore::new().unwrap()
        };
        let (writer, spill) = store.new_spill().await.unwrap();
        finish_writer(writer, b"initial").await.unwrap();
        drop(spill);
        let directory = store.ensure_temp_dir().unwrap().path().to_owned();
        std::fs::remove_dir(&directory).unwrap();
        assert!(!directory.exists());

        let (writer, spill) = store.new_spill().await.unwrap();
        assert_eq!(store.ensure_temp_dir().unwrap().path(), directory);
        assert!(directory.is_dir());
        #[cfg(unix)]
        assert_eq!(
            std::fs::metadata(&directory).unwrap().permissions().mode() & 0o777,
            0o700
        );
        finish_writer(writer, b"recovered").await.unwrap();
        let reader = spill.reader().await.unwrap();
        assert_eq!(reader.get_all().await.unwrap().as_ref(), b"recovered");
        let path = directory.join("spill_000001.bin");
        assert!(path.exists());
        drop(reader);
        drop(spill);
        assert!(!path.exists());
        if let Some(quota) = &store.quota {
            assert_eq!(quota.used.load(Ordering::Relaxed), 0);
        }
    }

    #[rstest]
    #[case::directory_creation(true, "failed to create spill directory")]
    #[case::file_creation(false, "failed to create spill file")]
    #[tokio::test]
    async fn test_new_spill_error_includes_path(
        #[case] is_directory_obstructed: bool,
        #[case] operation: &str,
    ) {
        let store = LocalSpillStore::with_cap(100).unwrap();
        let directory = store.ensure_temp_dir().unwrap().path().to_owned();
        let path = directory.join("spill_000000.bin");
        if is_directory_obstructed {
            std::fs::remove_dir(&directory).unwrap();
            std::fs::write(&directory, b"obstruction").unwrap();
        } else {
            std::fs::create_dir(&path).unwrap();
        }

        let Err(err) = store.new_spill().await else {
            panic!("spill allocation should fail for an obstructed path");
        };
        assert!(matches!(err, Error::IO { .. }), "expected IO, got {err:?}");
        let message = err.to_string();
        assert!(message.contains(operation), "{message}");
        assert!(message.contains(path.to_str().unwrap()), "{message}");
        assert_eq!(
            store.quota.as_ref().unwrap().used.load(Ordering::Relaxed),
            0
        );
        if is_directory_obstructed {
            assert!(message.contains(directory.to_str().unwrap()), "{message}");
            std::fs::remove_file(&directory).unwrap();
        } else {
            std::fs::remove_dir(&path).unwrap();
        }
    }

    #[tokio::test]
    async fn test_cap_exceeded() {
        let store = LocalSpillStore::with_cap(100).unwrap();
        let (writer, _spill) = store.new_spill().await.unwrap();
        let err = finish_writer(writer, &[0u8; 101]).await.unwrap_err();
        assert!(
            matches!(err, Error::DiskCapExceeded { cap_bytes: 100, .. }),
            "expected DiskCapExceeded, got {err:?}"
        );
    }

    #[tokio::test]
    async fn test_cap_shared_across_files() {
        let store = LocalSpillStore::with_cap(100).unwrap();
        let (writer_a, _spill_a) = store.new_spill().await.unwrap();
        let (writer_b, _spill_b) = store.new_spill().await.unwrap();

        finish_writer(writer_a, &[0u8; 60]).await.unwrap();
        // 60 already reserved by `a`; writing 60 more would reach 120 > 100.
        let err = finish_writer(writer_b, &[0u8; 60]).await.unwrap_err();
        assert!(
            matches!(err, Error::DiskCapExceeded { cap_bytes: 100, .. }),
            "expected DiskCapExceeded, got {err:?}"
        );
    }

    #[tokio::test]
    async fn test_cap_freed_on_drop() {
        let store = LocalSpillStore::with_cap(100).unwrap();

        {
            let (writer, spill) = store.new_spill().await.unwrap();
            finish_writer(writer, &[0u8; 80]).await.unwrap();
            // `spill` drops at the end of this block, releasing its 80 bytes.
            drop(spill);
        }

        let (writer, _spill) = store.new_spill().await.unwrap();
        // Succeeds because the cap is no longer under pressure.
        finish_writer(writer, &[0u8; 80]).await.unwrap();
    }

    #[rstest]
    #[case::writer_first(true)]
    #[case::handle_first(false)]
    #[tokio::test]
    async fn test_aborted_writer_reclaims_quota(#[case] writer_first: bool) {
        let store = LocalSpillStore::with_cap(100).unwrap();
        let quota = store.quota.as_ref().unwrap();
        let (mut writer, spill) = store.new_spill().await.unwrap();
        writer.write_all(&[0; 80]).await.unwrap();
        writer.flush().await.unwrap();
        let error = writer.write_all(&[0; 21]).await.unwrap_err();
        assert!(matches!(
            Error::from(error),
            Error::DiskCapExceeded { cap_bytes: 100, .. }
        ));
        assert_eq!(quota.used.load(Ordering::Relaxed), 80);
        if writer_first {
            drop(writer);
            assert_eq!(quota.used.load(Ordering::Relaxed), 80);
            drop(spill);
        } else {
            drop(spill);
            assert_eq!(quota.used.load(Ordering::Relaxed), 80);
            // The spill handle cannot delete a file while its writer is alive.
            writer.write_all(&[0; 20]).await.unwrap();
            writer.flush().await.unwrap();
            drop(writer);
        }
        assert_eq!(quota.used.load(Ordering::Relaxed), 0);
        let (writer, spill) = store.new_spill().await.unwrap();
        finish_writer(writer, &[0; 100]).await.unwrap();
        drop(spill);
        assert_eq!(quota.used.load(Ordering::Relaxed), 0);
    }

    #[rstest]
    #[case::range_future(0)]
    #[case::stream(1)]
    #[case::range_stream(2)]
    #[tokio::test]
    async fn test_read_owners_retain_backing(#[case] kind: usize) {
        let store = LocalSpillStore::with_cap(100).unwrap();
        let quota = store.quota.as_ref().unwrap();
        let (writer, spill) = store.new_spill().await.unwrap();
        finish_writer(writer, b"retained").await.unwrap();
        let reader = spill.reader().await.unwrap();
        drop(spill);
        assert_eq!(quota.used.load(Ordering::Relaxed), 8);
        let mut stream = match kind {
            0 => futures::stream::once(reader.get_range(0..8)).boxed(),
            1 => reader.get_stream().await.unwrap(),
            _ => reader.get_range_stream(0..8).await.unwrap(),
        };
        drop(reader);
        assert_eq!(quota.used.load(Ordering::Relaxed), 8);
        assert_eq!(stream.next().await.unwrap().unwrap().as_ref(), b"retained");
        drop(stream);
        assert_eq!(quota.used.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_cancelled_read_reclaims_backing() {
        let store = LocalSpillStore::with_cap(100).unwrap();
        let (writer, spill) = store.new_spill().await.unwrap();
        finish_writer(writer, &[0; 100]).await.unwrap();
        let reader = spill.reader().await.unwrap();
        let read = reader.get_range(0..100);
        drop(reader);
        drop(spill);
        assert_eq!(
            store.quota.as_ref().unwrap().used.load(Ordering::Relaxed),
            100
        );
        drop(read);
        assert_eq!(
            store.quota.as_ref().unwrap().used.load(Ordering::Relaxed),
            0
        );
        let (writer, _spill) = store.new_spill().await.unwrap();
        finish_writer(writer, &[0; 100]).await.unwrap();
    }

    #[test]
    fn test_cancelled_running_read_keeps_quota_until_io_finishes() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .max_blocking_threads(1)
            .build()
            .unwrap();
        runtime.block_on(async {
            let store = LocalSpillStore::with_cap(100).unwrap();
            let (writer, spill) = store.new_spill().await.unwrap();
            finish_writer(writer, &[0; 100]).await.unwrap();
            let reader = spill.reader().await.unwrap();
            let (release, wait) = std::sync::mpsc::channel();
            let (started, ready) = tokio::sync::oneshot::channel();
            let blocker = tokio::task::spawn_blocking(move || {
                started.send(()).unwrap();
                wait.recv().unwrap();
            });
            ready.await.unwrap();
            let mut read = reader.get_range(0..100);
            // Polling queues the read behind the blocker, so cancellation is
            // deterministic and cannot race an already completed disk read.
            assert!(futures::poll!(&mut read).is_pending());
            drop(read);
            drop(reader);
            drop(spill);
            let quota = store.quota.as_ref().unwrap();
            assert_eq!(quota.used.load(Ordering::Relaxed), 100);
            release.send(()).unwrap();
            blocker.await.unwrap();
            tokio::time::timeout(Duration::from_secs(1), async {
                while quota.used.load(Ordering::Relaxed) != 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            let (writer, _spill) = store.new_spill().await.unwrap();
            finish_writer(writer, &[0; 100]).await.unwrap();
        });
    }

    #[test]
    fn test_aborted_running_write_keeps_quota_until_io_finishes() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .max_blocking_threads(1)
            .build()
            .unwrap();
        runtime.block_on(async {
            let store = LocalSpillStore::with_cap(100).unwrap();
            let (mut writer, spill) = store.new_spill().await.unwrap();
            let (release, wait) = std::sync::mpsc::channel();
            let (started, ready) = tokio::sync::oneshot::channel();
            let blocker = tokio::task::spawn_blocking(move || {
                started.send(()).unwrap();
                wait.recv().unwrap();
            });
            ready.await.unwrap();
            writer.write_all(&[0; 100]).await.unwrap();
            drop(writer);
            drop(spill);
            let quota = store.quota.as_ref().unwrap();
            assert_eq!(quota.used.load(Ordering::Relaxed), 100);
            release.send(()).unwrap();
            blocker.await.unwrap();
            tokio::time::timeout(Duration::from_secs(1), async {
                while quota.used.load(Ordering::Relaxed) != 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            let (writer, _spill) = store.new_spill().await.unwrap();
            finish_writer(writer, &[0; 100]).await.unwrap();
        });
    }

    #[test]
    fn test_failed_cleanup_retains_quota_until_retry() {
        let store = LocalSpillStore::with_cap(100).unwrap();
        let directory = store.ensure_temp_dir().unwrap();
        let path = directory.path().join("failed-cleanup");
        // A directory reliably fails remove_file even when tests run as root.
        std::fs::create_dir(&path).unwrap();
        let quota = store.quota.as_ref().unwrap();
        quota.try_reserve(100).unwrap();
        drop(SpillBacking {
            fs_path: path.clone(),
            quota: Some(quota.clone()),
            accepted: AtomicU64::new(100),
            finished: AtomicBool::new(false),
            directory: directory.clone(),
        });
        assert_eq!(quota.used.load(Ordering::Relaxed), 100);
        assert_eq!(directory.pending.lock().unwrap().len(), 1);
        std::fs::remove_dir(&path).unwrap();
        store.ensure_temp_dir().unwrap();
        assert_eq!(quota.used.load(Ordering::Relaxed), 0);
        assert!(directory.pending.lock().unwrap().is_empty());
        quota.try_reserve(100).unwrap();
        quota.release(100);
    }

    #[test]
    fn test_disk_quota_overflow() {
        let quota = DiskQuota::new(u64::MAX);
        quota.try_reserve(u64::MAX).unwrap();
        let error = quota.try_reserve(1).unwrap_err();
        assert!(matches!(error, Error::DiskCapExceeded { .. }));
        assert_eq!(quota.used.load(Ordering::Relaxed), u64::MAX);
        quota.release(u64::MAX);
    }

    #[tokio::test]
    async fn test_custom_implementation() {
        // A custom store can satisfy the traits without a local file.
        struct MemStore;
        struct MemSpill;

        #[async_trait]
        impl Spill for MemSpill {
            async fn reader(&self) -> Result<Box<dyn Reader>> {
                ObjectStore::memory().open(&Path::from("/mem")).await
            }
        }

        #[async_trait]
        impl SpillStore for MemStore {
            async fn new_spill(&self) -> Result<(Box<dyn Writer>, Box<dyn Spill>)> {
                let writer = ObjectStore::memory().create(&Path::from("/mem")).await?;
                Ok((writer, Box::new(MemSpill)))
            }
        }

        let store = MemStore;
        // Exercise the factory + trait objects; the in-memory store is a fresh
        // instance per call so we don't round-trip data here.
        let (_writer, _spill) = store.new_spill().await.unwrap();
    }

    /// A [`Writer`] whose `poll_write` accepts a fixed number of bytes per call,
    /// or fails, so we can drive the [`SpillWriter`] release arms that the local
    /// backend (which accepts every write in full) never hits.
    struct ControlledWriter {
        outcome: Poll<io::Result<usize>>,
    }

    impl AsyncWrite for ControlledWriter {
        fn poll_write(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<io::Result<usize>> {
            match &self.outcome {
                Poll::Ready(Ok(n)) => Poll::Ready(Ok((*n).min(buf.len()))),
                Poll::Ready(Err(e)) => Poll::Ready(Err(io::Error::new(e.kind(), e.to_string()))),
                Poll::Pending => Poll::Pending,
            }
        }
        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    #[async_trait]
    impl Writer for ControlledWriter {
        async fn tell(&mut self) -> Result<usize> {
            Ok(0)
        }
        async fn shutdown(&mut self) -> Result<WriteResult> {
            Ok(WriteResult::default())
        }
    }

    #[rstest]
    #[case::partial(Poll::Ready(Ok(10)), 10)]
    #[case::error(Poll::Ready(Err(io::Error::other("boom"))), 0)]
    #[case::pending(Poll::Pending, 0)]
    #[tokio::test]
    async fn test_spill_writer_releases_unaccepted_bytes(
        #[case] outcome: Poll<io::Result<usize>>,
        #[case] accepted: u64,
    ) {
        let store = LocalSpillStore::with_cap(100).unwrap();
        let quota = store.quota.as_ref().unwrap().clone();
        let backing = Arc::new(SpillBacking {
            fs_path: store.ensure_temp_dir().unwrap().path().join("controlled"),
            quota: Some(quota.clone()),
            accepted: AtomicU64::new(0),
            finished: AtomicBool::new(false),
            directory: store.ensure_temp_dir().unwrap(),
        });
        let mut writer = SpillWriter {
            inner: Box::new(ControlledWriter { outcome }),
            backing,
        };
        let buffer = [0; 40];
        let _result = futures::poll!(std::pin::pin!(writer.write(&buffer)));
        assert_eq!(quota.used.load(Ordering::Relaxed), accepted);
        drop(writer);
        assert_eq!(quota.used.load(Ordering::Relaxed), 0);
    }
}
