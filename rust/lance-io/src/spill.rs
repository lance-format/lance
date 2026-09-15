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
//! - **RAII.** Dropping the [`Spill`] releases its bytes back to the store's
//!   budget and reclaims the storage: the disk backing deletes the file, with
//!   the store's temp directory as the backstop for anything leaked, and the
//!   in-memory fallback frees the bytes with the last handle to them.
//!
//! # Disk cap
//!
//! [`LocalSpillStore::with_cap`] enforces a byte budget shared across all live
//! handles, returning a typed [`lance_core::Error::DiskCapExceeded`] rather than
//! silently filling the disk. Accounting is reserve-on-write + release-on-drop,
//! by stat on the disk backing and from a counter of accepted bytes in the
//! in-memory fallback, which has no file to stat. The disk backing releases when
//! the handle drops, best effort: it returns what the stat can see then, which is
//! nothing if the stat fails, and nothing before the writer has been shut down,
//! since the file is not in place until then. The in-memory fallback instead
//! charges its bytes until the last owner of that spill's backing drops, writer
//! and readers included, so a retained reader keeps them on the budget.

use std::io;
use std::ops::Range;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Once, OnceLock};
use std::task::{Context, Poll};

use async_trait::async_trait;
use bytes::Bytes;
use futures::future::BoxFuture;
use futures::stream::StreamExt;
use lance_core::deepsize::DeepSizeOf;
use object_store::path::Path;
use tokio::io::AsyncWrite;

use lance_core::{Error, Result};

use crate::object_store::ObjectStore;
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
    /// [`Spill`] is dropped.
    async fn new_spill(&self) -> Result<(Box<dyn Writer>, Box<dyn Spill>)>;
}

/// The readable half of a spill, and the owner of its backing storage.
///
/// Dropping it reclaims the storage. The trait is object-safe so it can be
/// returned as `Box<dyn Spill>` from [`SpillStore::new_spill`].
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
    used: Arc<Mutex<u64>>,
}

impl DiskQuota {
    fn new(cap_bytes: u64) -> Self {
        Self {
            cap_bytes,
            used: Arc::new(Mutex::new(0)),
        }
    }

    /// Try to reserve `n` bytes, failing with [`Error::DiskCapExceeded`] if the
    /// reservation would push total usage past the cap.
    fn try_reserve(&self, n: u64) -> Result<()> {
        // The lock is held only for a couple of arithmetic ops and never across
        // an `.await`, so a std `Mutex` is the simplest correct choice.
        let mut used = self.used.lock().unwrap();
        let next = used.saturating_add(n);
        if next > self.cap_bytes {
            return Err(Error::disk_cap_exceeded(self.cap_bytes, *used));
        }
        *used = next;
        Ok(())
    }

    /// Release `n` previously reserved bytes back to the budget.
    fn release(&self, n: u64) {
        // Saturating sub keeps a stray double-release from underflowing.
        let mut used = self.used.lock().unwrap();
        *used = used.saturating_sub(n);
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
    quota: Option<DiskQuota>,
    /// Present only for a backing with no file to stat, which needs the accepted
    /// byte count to release its reservation. Held rather than just written to,
    /// so bytes this writer adds after its [`Spill`] is dropped are still
    /// counted before the release.
    reservation: Option<Arc<Reservation>>,
    finished: Arc<AtomicBool>,
}

/// A quota reservation released when its last owner drops.
///
/// The in-memory backing is reclaimed by the last `Arc` to it, and a reader is
/// one of those owners, so the reservation has to be tied to the same set of
/// owners: releasing it when the [`Spill`] alone drops would take bytes off the
/// budget while a retained reader still holds them.
#[derive(Debug)]
struct Reservation {
    quota: DiskQuota,
    /// Bytes the paired writer's inner writer accepted.
    written: AtomicU64,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.quota.release(self.written.load(Ordering::Relaxed));
    }
}

impl AsyncWrite for SpillWriter {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let Some(quota) = &this.quota else {
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
                if let Some(reservation) = &this.reservation {
                    reservation.written.fetch_add(*n as u64, Ordering::Relaxed);
                }
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
            this.finished.store(true, Ordering::Relaxed);
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
        self.finished.store(true, Ordering::Relaxed);
        Ok(result)
    }
}

/// The local-disk scratch directory a [`LocalSpillStore`] writes into, resolved
/// on first use.
struct DiskBacking {
    store: Arc<ObjectStore>,
    /// Backstop cleanup: removes the whole scratch directory on drop.
    temp_dir: Arc<tempfile::TempDir>,
}

/// A [`SpillStore`] that writes temporary files to a local temp directory.
///
/// The directory is created on the first [`SpillStore::new_spill`] call rather
/// than at construction, so a process that never spills never touches it. A call
/// that cannot create it keeps that one spill in memory instead, and the next
/// call tries the directory again. A spill kept in memory is bounded only by the
/// store's byte cap, so an uncapped store trades a hard failure for memory
/// pressure.
///
/// By default there is no disk cap. Use [`LocalSpillStore::with_cap`] to
/// configure one shared across every handle this store produces.
///
/// The temp directory is deleted when the store is dropped, cleaning up any
/// files whose handles have already been dropped.
pub struct LocalSpillStore {
    backing: OnceLock<DiskBacking>,
    /// How to create the scratch directory. A field so a test can drive the
    /// failing path without a host whose temp directory is unusable.
    temp_dir_factory: fn() -> io::Result<tempfile::TempDir>,
    /// Gates the fallback warning, which would otherwise repeat per spill.
    warned: Once,
    file_counter: Arc<AtomicU64>,
    /// Byte budget shared across every handle, enforced while writing.
    quota: Option<DiskQuota>,
}

impl LocalSpillStore {
    /// Create a store with no disk cap.
    ///
    /// The `Result` is kept for callers that already handle one; constructing a
    /// store does no I/O and cannot fail.
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

    /// The scratch directory, or `None` when it cannot be created and the caller
    /// should keep this spill in memory.
    ///
    /// Only success is cached, so a transient failure does not decide where every
    /// later spill goes. The returned reference borrows from the cache rather
    /// than from a local, which is what keeps a lost `set` race from handing out
    /// a path under a [`tempfile::TempDir`] that is about to be dropped.
    fn disk_backing(&self) -> Option<&DiskBacking> {
        if let Some(backing) = self.backing.get() {
            return Some(backing);
        }
        match (self.temp_dir_factory)() {
            Ok(temp_dir) => {
                // Losing the race drops this directory again, empty: nothing is
                // written until after this returns.
                let _ = self.backing.set(DiskBacking {
                    store: Arc::new(ObjectStore::local()),
                    temp_dir: Arc::new(temp_dir),
                });
                self.backing.get()
            }
            Err(err) => {
                self.warned.call_once(|| {
                    tracing::warn!(
                        error = %err,
                        "could not create a temp directory for spilling; \
                         keeping this spill in memory and retrying on the next"
                    );
                });
                tracing::debug!(error = %err, "spill directory unavailable");
                None
            }
        }
    }
}

impl Default for LocalSpillStore {
    fn default() -> Self {
        Self {
            backing: OnceLock::new(),
            temp_dir_factory: tempfile::tempdir,
            warned: Once::new(),
            file_counter: Arc::new(AtomicU64::new(0)),
            quota: None,
        }
    }
}

#[async_trait]
impl SpillStore for LocalSpillStore {
    async fn new_spill(&self) -> Result<(Box<dyn Writer>, Box<dyn Spill>)> {
        let idx = self.file_counter.fetch_add(1, Ordering::Relaxed);
        let name = format!("spill_{idx:06}.bin");
        let finished = Arc::new(AtomicBool::new(false));

        match self.disk_backing() {
            Some(DiskBacking { store, temp_dir }) => {
                let fs_path = temp_dir.path().join(name);
                let os_path = Path::from_absolute_path(&fs_path)?;
                let writer = Box::new(SpillWriter {
                    inner: store.create(&os_path).await?,
                    quota: self.quota.clone(),
                    reservation: None,
                    finished: finished.clone(),
                });
                let spill = Box::new(LocalSpill {
                    store: store.clone(),
                    os_path,
                    fs_path,
                    quota: self.quota.clone(),
                    finished,
                    _temp_dir: temp_dir.clone(),
                });
                Ok((writer, spill))
            }
            None => {
                // One backend per spill, so the bytes go with the last `Arc` to
                // it: nothing to delete, and no runtime needed to do it.
                let store = Arc::new(ObjectStore::memory());
                let os_path = Path::from(name);
                let reservation = self.quota.as_ref().map(|quota| {
                    Arc::new(Reservation {
                        quota: quota.clone(),
                        written: AtomicU64::new(0),
                    })
                });
                let writer = Box::new(SpillWriter {
                    inner: store.create(&os_path).await?,
                    quota: self.quota.clone(),
                    reservation: reservation.clone(),
                    finished: finished.clone(),
                });
                let spill = Box::new(MemorySpill {
                    store,
                    os_path,
                    reservation,
                    finished,
                });
                Ok((writer, spill))
            }
        }
    }
}

/// The readable half of a [`LocalSpillStore`] spill; reclaims the file on drop.
struct LocalSpill {
    store: Arc<ObjectStore>,
    os_path: Path,
    fs_path: PathBuf,
    quota: Option<DiskQuota>,
    /// Set by the paired [`SpillWriter`] once it has been shut down.
    finished: Arc<AtomicBool>,
    /// Keep the store's temp directory alive for at least this file's lifetime.
    _temp_dir: Arc<tempfile::TempDir>,
}

#[async_trait]
impl Spill for LocalSpill {
    async fn reader(&self) -> Result<Box<dyn Reader>> {
        // `Relaxed` is sufficient: the flag only gates "has the writer shut
        // down"; the bytes themselves are synchronized through the filesystem,
        // not this load.
        if !self.finished.load(Ordering::Relaxed) {
            return Err(Error::invalid_input(
                "spill reader requested before the writer was shut down",
            ));
        }
        self.store.open(&self.os_path).await
    }
}

impl Drop for LocalSpill {
    fn drop(&mut self) {
        // Release the bytes this file occupied back to the budget. We stat the
        // persisted file rather than tracking writes, which is exact for the
        // write-once contract.
        if let Some(quota) = &self.quota
            && let Ok(metadata) = std::fs::metadata(&self.fs_path)
        {
            quota.release(metadata.len());
        }
        // Best-effort removal; the temp dir is the backstop.
        let _ = std::fs::remove_file(&self.fs_path);
    }
}

/// The readable half of an in-memory spill, and the owner of its backend.
///
/// The backend holds this one object and nothing else, so the bytes go with the
/// last `Arc` to it. A reader keeps its own reference, which puts residency on
/// the same footing as the disk path's unlink with an open descriptor.
struct MemorySpill {
    store: Arc<ObjectStore>,
    os_path: Path,
    /// Shared with the paired writer and with every reader handed out, so the
    /// bytes leave the budget with the last owner of the backend rather than
    /// with this handle.
    reservation: Option<Arc<Reservation>>,
    /// Set by the paired [`SpillWriter`] once it has been shut down.
    finished: Arc<AtomicBool>,
}

#[async_trait]
impl Spill for MemorySpill {
    async fn reader(&self) -> Result<Box<dyn Reader>> {
        if !self.finished.load(Ordering::Relaxed) {
            return Err(Error::invalid_input(
                "spill reader requested before the writer was shut down",
            ));
        }
        let inner = self.store.open(&self.os_path).await?;
        Ok(match &self.reservation {
            Some(reservation) => Box::new(ChargedReader {
                inner,
                reservation: reservation.clone(),
            }),
            None => inner,
        })
    }
}

/// A [`Reader`] that holds a share of its spill's quota reservation.
///
/// Delegates everything; it exists only so the bytes a retained reader keeps
/// resident stay charged to the cap. `get_range` and the two stream methods hand
/// out products that own the backing and outlive this reader, so each of them
/// carries its own share.
#[derive(Debug)]
struct ChargedReader {
    inner: Box<dyn Reader>,
    reservation: Arc<Reservation>,
}

/// Tie `reservation` to the life of `stream`, which is `'static` and can outlive
/// the reader that produced it.
fn charged_stream(stream: ByteStream, reservation: Arc<Reservation>) -> ByteStream {
    Box::pin(stream.map(move |item| {
        let _ = &reservation;
        item
    }))
}

impl DeepSizeOf for ChargedReader {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.inner.deep_size_of_children(context)
    }
}

impl Reader for ChargedReader {
    fn path(&self) -> &Path {
        self.inner.path()
    }

    fn block_size(&self) -> usize {
        self.inner.block_size()
    }

    fn io_parallelism(&self) -> usize {
        self.inner.io_parallelism()
    }

    fn size(&self) -> BoxFuture<'_, object_store::Result<usize>> {
        self.inner.size()
    }

    fn get_range(&self, range: Range<usize>) -> BoxFuture<'static, object_store::Result<Bytes>> {
        let reservation = self.reservation.clone();
        let inner = self.inner.get_range(range);
        Box::pin(async move {
            let result = inner.await;
            drop(reservation);
            result
        })
    }

    fn get_all(&self) -> BoxFuture<'_, object_store::Result<Bytes>> {
        self.inner.get_all()
    }

    fn get_stream(&self) -> BoxFuture<'_, object_store::Result<ByteStream>> {
        let reservation = self.reservation.clone();
        Box::pin(async move {
            let stream = self.inner.get_stream().await?;
            Ok(charged_stream(stream, reservation))
        })
    }

    fn get_range_stream(
        &self,
        range: Range<usize>,
    ) -> BoxFuture<'_, object_store::Result<ByteStream>> {
        let reservation = self.reservation.clone();
        Box::pin(async move {
            let stream = self.inner.get_range_stream(range).await?;
            Ok(charged_stream(stream, reservation))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[tokio::test]
    async fn test_reader_requires_finished_writer() {
        let store = LocalSpillStore::new().unwrap();
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
        let reader = spill.reader().await.unwrap();
        assert_eq!(reader.get_all().await.unwrap().as_ref(), b"partial");
    }

    #[tokio::test]
    async fn test_reader_ready_after_async_shutdown() {
        // Shutting down through the `AsyncWrite` surface (not the `Writer`
        // trait) must also mark the spill readable — covers poll_shutdown's
        // flag set, the path the `Writer::shutdown` tests don't reach.
        let store = LocalSpillStore::new().unwrap();
        let (mut writer, spill) = store.new_spill().await.unwrap();
        writer.write_all(b"async").await.unwrap();
        AsyncWriteExt::shutdown(&mut writer).await.unwrap();

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

    #[tokio::test]
    async fn test_raii_cleanup() {
        let store = LocalSpillStore::new().unwrap();
        let (writer, spill) = store.new_spill().await.unwrap();
        finish_writer(writer, b"some bytes").await.unwrap();

        // The first spill gets a deterministic name under the store's temp dir.
        let path = disk_temp_dir(&store).join("spill_000000.bin");
        assert!(path.exists());
        drop(spill);
        assert!(!path.exists(), "spill file should be deleted on drop");
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
    /// or fails, so the [`SpillWriter`] release arms can be driven directly
    /// rather than by arranging for a real backend to short-write.
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

    #[tokio::test]
    async fn test_spill_writer_releases_unaccepted_bytes() {
        // Short write: the inner writer accepts only 10 of the 40 reserved bytes,
        // so the 30-byte remainder must be returned to the budget, and the
        // counter the in-memory arm releases from must record 10 rather than 40.
        let quota = DiskQuota::new(100);
        let reservation = Arc::new(Reservation {
            quota: quota.clone(),
            written: AtomicU64::new(0),
        });
        let mut writer = SpillWriter {
            inner: Box::new(ControlledWriter {
                outcome: Poll::Ready(Ok(10)),
            }),
            quota: Some(quota.clone()),
            reservation: Some(reservation.clone()),
            finished: Arc::new(AtomicBool::new(false)),
        };
        let n = writer.write(&[0u8; 40]).await.unwrap();
        assert_eq!(n, 10);
        assert_eq!(
            *quota.used.lock().unwrap(),
            10,
            "only the accepted bytes should remain reserved"
        );
        assert_eq!(reservation.written.load(Ordering::Relaxed), 10);

        // Failed write: the full reservation must be released, and nothing may be
        // counted as written.
        let quota = DiskQuota::new(100);
        let reservation = Arc::new(Reservation {
            quota: quota.clone(),
            written: AtomicU64::new(0),
        });
        let mut writer = SpillWriter {
            inner: Box::new(ControlledWriter {
                outcome: Poll::Ready(Err(io::Error::other("boom"))),
            }),
            quota: Some(quota.clone()),
            reservation: Some(reservation.clone()),
            finished: Arc::new(AtomicBool::new(false)),
        };
        writer.write(&[0u8; 40]).await.unwrap_err();
        assert_eq!(reservation.written.load(Ordering::Relaxed), 0);
        assert_eq!(
            *quota.used.lock().unwrap(),
            0,
            "a failed write should release its entire reservation"
        );
    }

    /// A temp-directory factory that always fails, standing in for a host whose
    /// temp directory is unusable.
    fn no_temp_dir() -> io::Result<tempfile::TempDir> {
        Err(io::Error::from(io::ErrorKind::PermissionDenied))
    }

    /// A store that can never resolve a disk backing.
    fn memory_only_store(quota: Option<DiskQuota>) -> LocalSpillStore {
        LocalSpillStore {
            temp_dir_factory: no_temp_dir,
            quota,
            ..Default::default()
        }
    }

    /// The temp directory of a store whose backing has resolved to disk.
    fn disk_temp_dir(store: &LocalSpillStore) -> &std::path::Path {
        match store.backing.get() {
            Some(backing) => backing.temp_dir.path(),
            None => panic!("expected a resolved disk backing"),
        }
    }

    #[test]
    fn test_construction_resolves_no_backing() {
        // Constructing a session used to create the temp directory, so a host
        // that cannot create one aborted on an unrelated path. Nothing may be
        // resolved until the first spill.
        assert!(LocalSpillStore::new().unwrap().backing.get().is_none());
        assert!(LocalSpillStore::default().backing.get().is_none());
        assert!(
            LocalSpillStore::with_cap(1 << 20)
                .unwrap()
                .backing
                .get()
                .is_none()
        );
    }

    #[tokio::test]
    async fn test_first_spill_resolves_the_disk_backing() {
        let store = LocalSpillStore::new().unwrap();
        let (writer, _spill) = store.new_spill().await.unwrap();
        finish_writer(writer, b"resolved").await.unwrap();
        assert!(store.backing.get().is_some());
    }

    #[tokio::test]
    async fn test_memory_backing_round_trips() {
        let store = memory_only_store(None);

        let (writer, spill) = store.new_spill().await.unwrap();
        let data = b"spilled without a temp dir";
        finish_writer(writer, data).await.unwrap();

        let reader = spill.reader().await.unwrap();
        assert_eq!(reader.get_all().await.unwrap().as_ref(), data);
        assert!(
            store.backing.get().is_none(),
            "a failed temp dir must not be cached as a backing"
        );
    }

    #[tokio::test]
    async fn test_memory_backing_rejects_a_reader_before_shutdown() {
        let store = memory_only_store(None);

        let (_writer, spill) = store.new_spill().await.unwrap();
        let err = spill.reader().await.unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput { .. }),
            "expected the write-before-read rejection, got {err:?}"
        );
    }

    #[tokio::test]
    async fn test_memory_backing_releases_the_quota_on_drop() {
        let quota = DiskQuota::new(60);
        let store = memory_only_store(Some(quota.clone()));

        // A second spill stays alive across the drop, so an over-release shows up
        // as its bytes going missing from the budget: `release` saturates, which
        // otherwise makes a `u64::MAX` release indistinguishable from a correct
        // one.
        let (other_writer, _other) = store.new_spill().await.unwrap();
        finish_writer(other_writer, &[0u8; 10]).await.unwrap();

        let (writer, spill) = store.new_spill().await.unwrap();
        finish_writer(writer, &[0u8; 40]).await.unwrap();
        assert_eq!(*quota.used.lock().unwrap(), 50);
        drop(spill);
        assert_eq!(*quota.used.lock().unwrap(), 10);
    }

    #[tokio::test]
    async fn test_a_reader_outlives_the_memory_spill_handle() {
        // The disk path tolerates this because unlink leaves an open descriptor
        // readable; the in-memory backend has to hold its bytes for the reader
        // the same way.
        let store = memory_only_store(None);
        let (writer, spill) = store.new_spill().await.unwrap();
        let data = b"read after the handle is gone";
        finish_writer(writer, data).await.unwrap();

        let reader = spill.reader().await.unwrap();
        drop(spill);
        // The yield is what makes this discriminating: an earlier revision
        // reclaimed through a task spawned in `Drop`, and with no scheduling
        // point that task could not run before the read.
        tokio::task::yield_now().await;
        assert_eq!(reader.get_all().await.unwrap().as_ref(), data);
    }

    #[tokio::test]
    async fn test_a_retained_reader_stays_charged_to_the_cap() {
        let quota = DiskQuota::new(50);
        let store = memory_only_store(Some(quota.clone()));

        let (writer, spill) = store.new_spill().await.unwrap();
        finish_writer(writer, &[1u8; 40]).await.unwrap();
        let retained = spill.reader().await.unwrap();
        drop(spill);

        // The reader still holds the bytes, so releasing here would let the next
        // spill over a cap the process is already using up.
        assert_eq!(retained.get_all().await.unwrap().len(), 40);
        assert_eq!(*quota.used.lock().unwrap(), 40);

        let (writer, _spill) = store.new_spill().await.unwrap();
        let err = finish_writer(writer, &[2u8; 40]).await.unwrap_err();
        assert!(matches!(err, Error::DiskCapExceeded { .. }), "{err:?}");

        drop(retained);
        assert_eq!(*quota.used.lock().unwrap(), 0);
    }

    /// A reader's detached products own the backing too, so each of them has to
    /// keep the charge. One test per API that hands one out.
    #[rstest::rstest]
    #[case::get_range(0)]
    #[case::get_stream(1)]
    #[case::get_range_stream(2)]
    #[tokio::test]
    async fn test_a_detached_reader_product_stays_charged_to_the_cap(#[case] api: u8) {
        let quota = DiskQuota::new(50);
        let store = memory_only_store(Some(quota.clone()));

        let (writer, spill) = store.new_spill().await.unwrap();
        finish_writer(writer, &[1u8; 40]).await.unwrap();
        let reader = spill.reader().await.unwrap();

        // Each of these owns the in-memory backing and outlives the reader.
        let mut detached_future = None;
        let mut detached_stream = None;
        match api {
            0 => detached_future = Some(reader.get_range(0..40)),
            1 => detached_stream = Some(reader.get_stream().await.unwrap()),
            _ => detached_stream = Some(reader.get_range_stream(0..40).await.unwrap()),
        }
        drop(reader);
        drop(spill);

        assert_eq!(
            *quota.used.lock().unwrap(),
            40,
            "the bytes are still reachable, so they stay charged"
        );
        let (writer, _second) = store.new_spill().await.unwrap();
        let err = finish_writer(writer, &[2u8; 40]).await.unwrap_err();
        assert!(matches!(err, Error::DiskCapExceeded { .. }), "{err:?}");

        // And the product still reads what it was created for.
        match (detached_future, detached_stream) {
            (Some(fut), None) => assert_eq!(fut.await.unwrap().as_ref(), &[1u8; 40]),
            (None, Some(stream)) => {
                let chunks: Vec<Bytes> = stream.map(|c| c.unwrap()).collect().await;
                assert_eq!(chunks.concat(), vec![1u8; 40]);
            }
            _ => unreachable!(),
        }
        assert_eq!(*quota.used.lock().unwrap(), 0);
    }

    /// One failing attempt must not decide where later spills go. Uses a process
    /// -wide counter, so it is the only test driving this factory.
    static TEMP_DIR_FAILURES_LEFT: AtomicU64 = AtomicU64::new(1);

    fn temp_dir_failing_once() -> io::Result<tempfile::TempDir> {
        if TEMP_DIR_FAILURES_LEFT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
            .is_ok()
        {
            no_temp_dir()
        } else {
            tempfile::tempdir()
        }
    }

    #[tokio::test]
    async fn test_a_transient_temp_dir_failure_is_retried() {
        // Reset rather than rely on the initial value, so a repeat run of this
        // test in one process still sees exactly one failure.
        TEMP_DIR_FAILURES_LEFT.store(1, Ordering::Relaxed);
        let store = LocalSpillStore {
            temp_dir_factory: temp_dir_failing_once,
            ..Default::default()
        };

        let (writer, spill) = store.new_spill().await.unwrap();
        finish_writer(writer, b"in memory").await.unwrap();
        assert!(
            store.backing.get().is_none(),
            "the first attempt failed, so nothing may be cached"
        );
        drop(spill);

        let (writer, _spill) = store.new_spill().await.unwrap();
        finish_writer(writer, b"on disk").await.unwrap();
        assert!(
            store.backing.get().is_some(),
            "the second attempt succeeded, so the directory must now be cached"
        );
        assert!(disk_temp_dir(&store).join("spill_000001.bin").exists());
    }
}
