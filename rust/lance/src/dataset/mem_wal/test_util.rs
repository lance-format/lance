// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Test-only object store that injects WAL and SSTable write failures and
//! records the paths it serves (for
//! asserting which opens actually resolved through a given `StorageParams`).

use std::fmt::{Debug, Display, Formatter};
use std::ops::Range;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use bytes::Bytes;
use futures::stream::BoxStream;
use lance_core::storage::path::Path;
use lance_io::storage::{
    CopyOptions, GetOptions, GetResult, ListResult, ObjectMeta, PutMultipartOptions, PutOptions,
    PutPayload, PutResult, RenameOptions, Result as OSResult,
};
use tokio::sync::Notify;

use lance_io::storage::{BackendWrapper, Storage, StorageParams, StorageRegistry};

/// Knobs controlling injected write failures, shared with the test.
#[derive(Debug, Default)]
pub struct FailControls {
    /// Hold generation writes while later flushes are queued, then fail a
    /// bounded number of attempts once released. WAL writes remain independent.
    sstable_put_failures: AtomicUsize,
    block_sstable_puts: AtomicBool,
    sstable_put_release: Notify,
    sstable_put_attempts: AtomicUsize,
    /// Remaining WAL-entry `put_opts` calls to fail; saturates at 0. Set high to
    /// "always fail", set to 0 to "recover".
    wal_put_failures: AtomicUsize,
    /// When failing, write to the inner store anyway before reporting the error,
    /// simulating a lost acknowledgement (the PUT actually landed).
    simulate_lost_ack: AtomicBool,
    /// WAL-entry `put_opts` attempts observed, for assertions.
    wal_put_attempts: AtomicUsize,
    /// Remaining Blob sidecar `put_opts` calls to fail.
    blob_put_failures: AtomicUsize,
    /// Block Blob sidecar PUTs forever so cancellation can drop the caller.
    block_blob_puts: AtomicBool,
    /// Blob sidecar `put_opts` attempts observed, for assertions.
    blob_put_attempts: AtomicUsize,
    /// Every location written through this store.
    put_paths: StdMutex<Vec<String>>,
    /// Every location read through this store.
    get_paths: StdMutex<Vec<String>>,
    /// Fail every write whose path contains this substring.
    fail_writes_under: StdMutex<Option<String>>,
}

impl FailControls {
    pub fn fail_sstable_puts(&self, n: usize) {
        self.sstable_put_failures.store(n, Ordering::SeqCst);
    }
    pub fn hold_sstable_puts(&self) {
        self.block_sstable_puts.store(true, Ordering::SeqCst);
    }
    pub fn release_sstable_puts(&self) {
        self.block_sstable_puts.store(false, Ordering::SeqCst);
        self.sstable_put_release.notify_waiters();
    }
    pub fn sstable_attempts(&self) -> usize {
        self.sstable_put_attempts.load(Ordering::SeqCst)
    }

    async fn before_sstable_put(&self, location: &Path) -> OSResult<()> {
        if !location.as_ref().contains("_gen_") {
            return Ok(());
        }
        self.sstable_put_attempts.fetch_add(1, Ordering::SeqCst);
        let released = self.sstable_put_release.notified();
        tokio::pin!(released);
        released.as_mut().enable();
        if self.block_sstable_puts.load(Ordering::SeqCst) {
            released.await;
        }
        if self
            .sstable_put_failures
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
            .is_ok()
        {
            return Err(lance_io::storage::Error::Generic {
                store: "failing-test-store",
                source: "injected SSTable write failure".into(),
            });
        }
        Ok(())
    }

    pub fn fail_wal_puts(&self, n: usize) {
        self.wal_put_failures.store(n, Ordering::SeqCst);
    }
    pub fn recover(&self) {
        self.wal_put_failures.store(0, Ordering::SeqCst);
    }
    pub fn set_lost_ack(&self, v: bool) {
        self.simulate_lost_ack.store(v, Ordering::SeqCst);
    }
    /// Fail every write whose path contains `needle`, until cleared.
    pub fn fail_writes_under(&self, needle: &str) {
        *self.fail_writes_under.lock().unwrap() = Some(needle.to_string());
    }
    /// Stop failing the writes [`Self::fail_writes_under`] targets.
    pub fn clear_write_failures(&self) {
        *self.fail_writes_under.lock().unwrap() = None;
    }
    /// Whether a write to `location` should fail.
    fn fails_write(&self, location: &Path) -> bool {
        self.fail_writes_under
            .lock()
            .unwrap()
            .as_deref()
            .is_some_and(|needle| location.as_ref().contains(needle))
    }
    pub fn attempts(&self) -> usize {
        self.wal_put_attempts.load(Ordering::SeqCst)
    }
    pub fn fail_blob_puts(&self, n: usize) {
        self.blob_put_failures.store(n, Ordering::SeqCst);
    }
    pub fn set_block_blob_puts(&self, value: bool) {
        self.block_blob_puts.store(value, Ordering::SeqCst);
    }
    pub fn blob_attempts(&self) -> usize {
        self.blob_put_attempts.load(Ordering::SeqCst)
    }

    /// Did any write land on a path containing `needle`? An open that resolved
    /// its store from other params never reaches this store, so a `false` here
    /// means the params under test did not reach that open.
    pub fn wrote_under(&self, needle: &str) -> bool {
        self.put_paths
            .lock()
            .unwrap()
            .iter()
            .any(|p| p.contains(needle))
    }

    /// Number of backend reads, including failed requests.
    pub fn read_count(&self) -> usize {
        self.get_paths.lock().unwrap().len()
    }

    /// Did any read land on a path containing `needle`? See [`Self::wrote_under`].
    pub fn read_under(&self, needle: &str) -> bool {
        self.get_paths
            .lock()
            .unwrap()
            .iter()
            .any(|p| p.contains(needle))
    }
}

/// Wraps the inner store with [`FailingBackend`] at construction.
#[derive(Debug)]
struct FailingWrapper {
    controls: Arc<FailControls>,
}

impl BackendWrapper for FailingWrapper {
    fn wrap(
        &self,
        _prefix: &str,
        original: Arc<dyn lance_io::storage::StorageBackend>,
    ) -> Arc<dyn lance_io::storage::StorageBackend> {
        Arc::new(FailingBackend {
            inner: original,
            controls: self.controls.clone(),
        })
    }
}

/// Delegates everything to `inner`, failing WAL-entry PUTs per [`FailControls`].
#[derive(Debug)]
struct FailingBackend {
    inner: Arc<dyn lance_io::storage::StorageBackend>,
    controls: Arc<FailControls>,
}

impl FailingBackend {
    /// WAL entries live under `.../wal/`; manifests (`.../manifest/`) are never
    /// failed so `ShardWriter::open`/`claim_epoch` still work.
    fn is_wal_entry(location: &Path) -> bool {
        location.as_ref().contains("/wal/")
    }

    fn injected_error() -> lance_io::storage::Error {
        lance_io::storage::Error::Generic {
            store: "failing-test-store",
            source: "injected transient WAL put failure".to_string().into(),
        }
    }
}

impl Display for FailingBackend {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "FailingBackend({})", self.inner)
    }
}

#[async_trait::async_trait]
impl lance_io::storage::StorageBackend for FailingBackend {
    async fn list_paginated(
        &self,
        prefix: Option<&str>,
        options: lance_io::storage::PaginatedListOptions,
    ) -> lance_io::storage::Result<lance_io::storage::PaginatedListResult> {
        let _ = (prefix, options);
        Err(lance_io::storage::Error::NotSupported {
            source: "native pagination is unavailable".into(),
        })
    }

    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> OSResult<PutResult> {
        self.controls
            .put_paths
            .lock()
            .unwrap()
            .push(location.to_string());
        self.controls.before_sstable_put(location).await?;
        if self.controls.fails_write(location) {
            return Err(Self::injected_error());
        }
        if location.as_ref().ends_with(".blob") {
            self.controls
                .blob_put_attempts
                .fetch_add(1, Ordering::SeqCst);
            if self.controls.block_blob_puts.load(Ordering::SeqCst) {
                std::future::pending::<()>().await;
            }
            if self.controls.blob_put_failures.load(Ordering::SeqCst) > 0 {
                self.controls
                    .blob_put_failures
                    .fetch_sub(1, Ordering::SeqCst);
                return Err(Self::injected_error());
            }
        }
        if Self::is_wal_entry(location) {
            self.controls
                .wal_put_attempts
                .fetch_add(1, Ordering::SeqCst);
            if self.controls.wal_put_failures.load(Ordering::SeqCst) > 0 {
                self.controls
                    .wal_put_failures
                    .fetch_sub(1, Ordering::SeqCst);
                if self.controls.simulate_lost_ack.load(Ordering::SeqCst) {
                    // The write lands but we still report failure (lost ack).
                    let _ = self.inner.put_opts(location, payload, opts).await;
                }
                return Err(Self::injected_error());
            }
        }
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> OSResult<Box<dyn lance_io::storage::backend::MultipartUpload>> {
        // Data files (`*.lance`) are written multipart, not via `put_opts`.
        self.controls
            .put_paths
            .lock()
            .unwrap()
            .push(location.to_string());
        self.controls.before_sstable_put(location).await?;
        if self.controls.fails_write(location) {
            return Err(Self::injected_error());
        }
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
        self.controls
            .get_paths
            .lock()
            .unwrap()
            .push(location.to_string());
        self.inner.get_opts(location, options).await
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> OSResult<Vec<Bytes>> {
        self.controls
            .get_paths
            .lock()
            .unwrap()
            .push(location.to_string());
        self.inner.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, OSResult<Path>>,
    ) -> BoxStream<'static, OSResult<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
        self.inner.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, OSResult<ObjectMeta>> {
        self.inner.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> OSResult<()> {
        self.inner.copy_opts(from, to, opts).await
    }

    async fn rename_opts(&self, from: &Path, to: &Path, opts: RenameOptions) -> OSResult<()> {
        self.inner.rename_opts(from, to, opts).await
    }
}

/// `StorageParams` carrying the observable store wrapper, plus the controls
/// to drive and inspect it. Open a dataset with these and every store resolved
/// *from these params* — the base and any derived URI they are threaded to —
/// reports its traffic back through the controls.
pub fn observable_store_params() -> (StorageParams, Arc<FailControls>) {
    let controls = Arc::new(FailControls::default());
    let params = StorageParams {
        backend_wrappers: vec![Arc::new(FailingWrapper {
            controls: controls.clone(),
        })],
        ..Default::default()
    };
    (params, controls)
}

/// Build an in-memory `Storage` whose WAL-entry writes can be failed on
/// demand. Returns the store, its base path, and the shared controls.
pub async fn failing_memory_store() -> (Arc<Storage>, Path, Arc<FailControls>) {
    let (params, controls) = observable_store_params();
    let (store, base) =
        Storage::from_uri_and_params(Arc::new(StorageRegistry::default()), "memory:///", &params)
            .await
            .unwrap();
    (store, base, controls)
}
