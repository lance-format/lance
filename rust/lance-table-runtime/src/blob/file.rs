// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::HashMap,
    ops::{DerefMut, Range},
    panic::AssertUnwindSafe,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use bytes::Bytes;
use futures::FutureExt;
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::{FileScheduler, ScanScheduler, SchedulerConfig};
use object_store::path::Path;
use tokio::sync::{Mutex, OnceCell, oneshot};

use lance_core::datatypes::BlobKind;
use lance_core::{Error, Result};

/// Shared physical read context for blob handles that resolve to the same object.
///
/// Blob descriptors are logical slices over a backing object (data file, packed
/// sidecar, dedicated sidecar, or external object). Multiple [`BlobFile`] values
/// can point at different regions of that same object. This struct gives those
/// handles a single lazy-open scheduler plus a lightweight pending queue so
/// concurrent reads can be opportunistically grouped before reaching Lance's
/// existing I/O scheduler.
#[derive(Debug)]
pub struct BlobSource {
    pub object_store: Arc<ObjectStore>,
    pub path: Path,
    scheduler: OnceCell<FileScheduler>,
    pending_reads: Mutex<PendingBlobReads>,
    /// Calls to [`BlobSource::read_ranges`]. Shared by every handle on this
    /// source. The I/O scheduler may combine or split a submission.
    range_submissions: AtomicUsize,
}

impl BlobSource {
    /// Create a shared read context for one physical backing object.
    pub fn new(object_store: Arc<ObjectStore>, path: Path) -> Self {
        Self {
            object_store,
            path,
            scheduler: OnceCell::new(),
            pending_reads: Mutex::new(PendingBlobReads::default()),
            range_submissions: AtomicUsize::new(0),
        }
    }

    /// Read one or more physical ranges from this source.
    ///
    /// Concurrent callers enqueue their requests into `pending_reads`. The first
    /// caller in a drain cycle becomes the leader and spawns the batch drain task.
    /// The mutex critical section only updates in-memory queue bookkeeping; the
    /// actual scheduler submission happens after the lock is released.
    async fn read_ranges(self: &Arc<Self>, ranges: Vec<Range<u64>>) -> Result<Vec<Bytes>> {
        if ranges.is_empty() {
            return Ok(Vec::new());
        }
        self.range_submissions.fetch_add(1, Ordering::Relaxed);

        let scheduler = self
            .scheduler
            .get_or_try_init(|| async {
                let reader = self.object_store.open(&self.path).await?;
                let scheduler = ScanScheduler::new(
                    self.object_store.clone(),
                    SchedulerConfig::max_bandwidth(self.object_store.as_ref()),
                );
                Ok::<_, Error>(scheduler.open_reader(reader.into()))
            })
            .await?;

        let (response_tx, response_rx) = oneshot::channel();
        let should_spawn = {
            let mut pending_reads = self.pending_reads.lock().await;
            pending_reads.requests.push(PendingBlobRead {
                ranges,
                response: response_tx,
            });
            if pending_reads.is_draining {
                false
            } else {
                pending_reads.is_draining = true;
                true
            }
        };

        if should_spawn {
            let source = self.clone();
            let scheduler = scheduler.clone();
            lance_core::utils::tokio::spawn(async move {
                let result = AssertUnwindSafe(source.clone().drain_pending_reads(scheduler))
                    .catch_unwind()
                    .await;
                if let Err(panic) = result {
                    let mut pending_reads = source.pending_reads.lock().await;
                    pending_reads.is_draining = false;
                    std::panic::resume_unwind(panic);
                }
            });
        }

        response_rx.await.map_err(|_| {
            Error::internal("Blob source read task dropped the response".to_string())
        })?
    }

    /// Drain currently queued requests and submit them as scheduler batches.
    ///
    /// Each loop iteration grabs the queued requests with a short mutex hold and
    /// dispatches them without waiting for earlier batches to finish. Awaiting a
    /// batch here would hold later, naturally staggered callers behind its I/O.
    /// [`FileScheduler`] owns the concurrency and backpressure for dispatched I/O.
    async fn drain_pending_reads(self: Arc<Self>, scheduler: FileScheduler) {
        loop {
            let batch = {
                let mut pending_reads = self.pending_reads.lock().await;
                if pending_reads.requests.is_empty() {
                    pending_reads.is_draining = false;
                    return;
                }
                std::mem::take(&mut pending_reads.requests)
            };
            let scheduler = scheduler.clone();
            lance_core::utils::tokio::spawn(async move {
                fulfill_pending_blob_reads(&scheduler, batch).await;
            });
        }
    }
}

/// Queue of pending logical blob reads for one [`BlobSource`].
///
/// `is_draining` marks whether a leader task is already draining the queue.
#[derive(Default, Debug)]
struct PendingBlobReads {
    requests: Vec<PendingBlobRead>,
    is_draining: bool,
}

/// Pending logical blob reads waiting to be grouped into one scheduler batch.
///
/// This queue exists only to combine overlapping concurrent calls. The actual
/// coalescing and physical I/O scheduling still happens in [`FileScheduler`].
#[derive(Debug)]
struct PendingBlobRead {
    ranges: Vec<Range<u64>>,
    response: oneshot::Sender<Result<Vec<Bytes>>>,
}

/// Submit one grouped batch of pending blob reads to Lance's [`FileScheduler`].
///
/// The function flattens all logical requests into one range list, preserves the
/// caller-visible order for each request, and fans the bytes back out after the
/// scheduler completes its own merge / split logic.
async fn fulfill_pending_blob_reads(scheduler: &FileScheduler, batch: Vec<PendingBlobRead>) {
    let total_ranges = batch
        .iter()
        .map(|request| request.ranges.len())
        .sum::<usize>();
    let mut request_ranges = Vec::with_capacity(total_ranges);
    let mut response = batch
        .iter()
        .map(|request| vec![Bytes::new(); request.ranges.len()])
        .collect::<Vec<_>>();

    for (request_idx, request) in batch.iter().enumerate() {
        for (range_idx, range) in request.ranges.iter().enumerate() {
            if range.is_empty() {
                continue;
            }
            request_ranges.push((range.clone(), request_idx, range_idx));
        }
    }

    let result = if request_ranges.is_empty() {
        Ok(())
    } else {
        request_ranges.sort_by_key(|(range, _, _)| (range.start, range.end));
        let priority = request_ranges[0].0.start;
        match scheduler
            .submit_request(
                request_ranges
                    .iter()
                    .map(|(range, _, _)| range.clone())
                    .collect::<Vec<_>>(),
                priority,
            )
            .await
        {
            Ok(bytes_vec) => {
                for ((_, request_idx, range_idx), bytes) in
                    request_ranges.into_iter().zip(bytes_vec)
                {
                    response[request_idx][range_idx] = bytes;
                }
                Ok(())
            }
            Err(err) => Err(err),
        }
    };

    match result {
        Ok(()) => {
            for (request, bytes) in batch.into_iter().zip(response) {
                let _ = request.response.send(Ok(bytes));
            }
        }
        Err(err) => {
            let message = format!(
                "Failed to read blob source {}: {}",
                scheduler.reader().path(),
                err
            );
            for request in batch {
                let _ = request.response.send(Err(Error::io(message.clone())));
            }
        }
    }
}

/// Cache key for sharing one [`BlobSource`] across multiple blob descriptors.
///
/// Store instance identity keeps credentials, endpoints, and wrappers isolated
/// even when two stores use the same prefix and path.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct BlobSourceKey {
    pub(super) store_id: usize,
    pub path: String,
}

impl BlobSourceKey {
    /// Build the cache key for one shared [`BlobSource`].
    pub(super) fn new(source: &BlobSource) -> Self {
        Self {
            store_id: Arc::as_ptr(&source.object_store) as usize,
            path: source.path.to_string(),
        }
    }
}

/// Return a shared [`BlobSource`] for the given physical object.
///
/// This keeps all blob handles that resolve to the same `(store, path)` on a
/// single lazy-open scheduler and pending-read queue.
pub(super) fn shared_blob_source(
    source_cache: &mut HashMap<BlobSourceKey, Arc<BlobSource>>,
    object_store: Arc<ObjectStore>,
    path: &Path,
) -> Arc<BlobSource> {
    let key = BlobSourceKey {
        store_id: Arc::as_ptr(&object_store) as usize,
        path: path.to_string(),
    };
    source_cache
        .entry(key)
        .or_insert_with(|| Arc::new(BlobSource::new(object_store, path.clone())))
        .clone()
}

/// A file-like object that represents a blob in a dataset
#[derive(Debug)]
pub struct BlobFile {
    pub(super) source: Arc<BlobSource>,
    state: Arc<Mutex<BlobFileState>>,
    pub(super) position: u64,
    pub(super) size: u64,
    kind: BlobKind,
    uri: Option<String>,
    base_id: Option<u32>,
}

impl BlobFile {
    /// Default sequential read-ahead size in bytes.
    pub const DEFAULT_READ_BUFFER_SIZE: usize = 512 * 1024;

    pub fn with_source(
        source: Arc<BlobSource>,
        position: u64,
        size: u64,
        kind: BlobKind,
        uri: Option<String>,
    ) -> Self {
        Self {
            source,
            position,
            size,
            kind,
            uri,
            base_id: None,
            state: Arc::new(Mutex::new(BlobFileState::new())),
        }
    }

    pub(super) fn with_base_id(mut self, base_id: Option<u32>) -> Self {
        self.base_id = base_id;
        self
    }

    /// Create an inline blob reader backed by a data file.
    ///
    /// This constructor assumes the caller has already resolved multi-base routing
    /// (base-aware object store and file path). It does not inspect dataset metadata.
    ///
    /// # Parameters
    ///
    /// * `object_store` - The store that owns `path`; reads are issued against this store.
    /// * `path` - Full path to the data file containing inline blob bytes.
    /// * `position` - Byte offset of the blob payload inside the data file.
    /// * `size` - Blob payload length in bytes.
    pub fn new_inline(
        object_store: Arc<ObjectStore>,
        path: Path,
        position: u64,
        size: u64,
    ) -> Self {
        Self::with_source(
            Arc::new(BlobSource::new(object_store, path)),
            position,
            size,
            BlobKind::Inline,
            None,
        )
    }

    /// Create a dedicated blob reader backed by a sidecar `.blob` file.
    ///
    /// Dedicated blobs occupy an entire sidecar file, so the logical read starts
    /// at offset `0` and spans `size` bytes.
    ///
    /// # Parameters
    ///
    /// * `object_store` - The store that owns `path`; reads are issued against this store.
    /// * `path` - Full path to the dedicated sidecar blob file.
    /// * `size` - Total byte length to expose from the sidecar file.
    pub fn new_dedicated(object_store: Arc<ObjectStore>, path: Path, size: u64) -> Self {
        Self::with_source(
            Arc::new(BlobSource::new(object_store, path)),
            0,
            size,
            BlobKind::Dedicated,
            None,
        )
    }

    /// Create a packed blob reader for a slice inside a shared sidecar `.blob` file.
    ///
    /// Packed blobs share one sidecar file; this constructor exposes only the
    /// `[position, position + size)` range that belongs to a single row.
    ///
    /// # Parameters
    ///
    /// * `object_store` - The store that owns `path`; reads are issued against this store.
    /// * `path` - Full path to the packed sidecar blob file.
    /// * `position` - Start offset of this blob within the packed sidecar.
    /// * `size` - Blob payload length in bytes.
    pub fn new_packed(
        object_store: Arc<ObjectStore>,
        path: Path,
        position: u64,
        size: u64,
    ) -> Self {
        Self::with_source(
            Arc::new(BlobSource::new(object_store, path)),
            position,
            size,
            BlobKind::Packed,
            None,
        )
    }

    /// Create an external blob reader backed by a caller-resolved object location.
    ///
    /// External blobs are identified by a URI in metadata, but actual reads happen
    /// against a concrete store/path pair resolved by the caller. This keeps URI
    /// resolution (which may be async) outside of the constructor.
    ///
    /// # Parameters
    ///
    /// * `object_store` - The resolved store used to open and read `path`.
    /// * `path` - The resolved object path that contains external blob bytes.
    /// * `uri` - The original URI recorded in blob metadata for round-tripping.
    /// * `position` - Start offset of the blob payload in the external object.
    /// * `size` - Number of bytes exposed from `position`.
    pub fn new_external(
        object_store: Arc<ObjectStore>,
        path: Path,
        uri: String,
        position: u64,
        size: u64,
    ) -> Self {
        Self::with_source(
            Arc::new(BlobSource::new(object_store, path)),
            position,
            size,
            BlobKind::External,
            Some(uri),
        )
    }

    /// Another handle on the same blob that shares this handle's cursor and
    /// open state, so a planned read can hold a caller's handle without taking
    /// ownership and still observe it being closed.
    pub(super) fn share(&self) -> Self {
        Self {
            source: self.source.clone(),
            state: self.state.clone(),
            position: self.position,
            size: self.size,
            kind: self.kind,
            uri: self.uri.clone(),
            base_id: self.base_id,
        }
    }

    /// Close the blob file, releasing any associated resources
    pub async fn close(&self) -> Result<()> {
        let mut state = self.state.lock().await;
        *state = BlobFileState::Closed;
        Ok(())
    }

    /// Returns true if the blob file is closed
    pub async fn is_closed(&self) -> bool {
        matches!(&*self.state.lock().await, BlobFileState::Closed)
    }

    async fn ensure_open(&self) -> Result<()> {
        let state = self.state.lock().await;
        match &*state {
            BlobFileState::Open { .. } => Ok(()),
            BlobFileState::Closed => Err(Error::invalid_input(
                "Blob file is already closed".to_string(),
            )),
        }
    }

    pub(super) fn read_phys_range(&self, range: Range<u64>) -> Result<Range<u64>> {
        if range.start > range.end {
            return Err(Error::invalid_input(format!(
                "Blob range start {} must be <= end {}",
                range.start, range.end
            )));
        }
        if range.end > self.size {
            return Err(Error::invalid_input(format!(
                "Blob range end {} exceeds blob size {}",
                range.end, self.size
            )));
        }
        let start = self.position.checked_add(range.start).ok_or_else(|| {
            Error::invalid_input(format!(
                "Blob range start overflowed physical position: base={} offset={}",
                self.position, range.start
            ))
        })?;
        let end = self.position.checked_add(range.end).ok_or_else(|| {
            Error::invalid_input(format!(
                "Blob range end overflowed physical position: base={} offset={}",
                self.position, range.end
            ))
        })?;
        Ok(start..end)
    }

    /// Read a byte range relative to the beginning of this blob without changing the cursor.
    ///
    /// The provided range is interpreted in blob-local coordinates, not object
    /// coordinates. Empty ranges are allowed. This method is intended for random
    /// access callers that want deterministic range semantics instead of the
    /// stateful file-like cursor used by [`Self::read`] and [`Self::read_up_to`].
    pub async fn read_range(&self, range: Range<u64>) -> Result<Bytes> {
        let mut data = self.read_ranges(&[range]).await?;
        Ok(data.pop().unwrap_or_default())
    }

    /// Read multiple ranges relative to the beginning of this blob without changing the cursor.
    ///
    /// Empty ranges are allowed and yield empty buffers. The result order always
    /// matches the input order, even though the underlying physical requests may
    /// be reordered, coalesced, or split for efficiency.
    pub async fn read_ranges(&self, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        self.ensure_open().await?;
        let physical_ranges = ranges
            .iter()
            .cloned()
            .map(|range| self.read_phys_range(range))
            .collect::<Result<Vec<_>>>()?;
        self.source.read_ranges(physical_ranges).await
    }

    /// Read the entire blob file from the current cursor position
    /// to the end of the file
    ///
    /// Advances the cursor by the number of bytes returned.
    pub async fn read(&self) -> Result<bytes::Bytes> {
        let mut state = self.state.lock().await;
        match state.deref_mut() {
            BlobFileState::Closed => Err(Error::invalid_input(
                "Blob file is already closed".to_string(),
            )),
            BlobFileState::Open {
                cursor, prefetch, ..
            } => {
                if *cursor >= self.size {
                    *prefetch = None;
                    return Ok(Bytes::new());
                }
                let remaining = (self.size - *cursor) as usize;
                let held = prefetch
                    .as_ref()
                    .and_then(|w| w.slice_from(*cursor, remaining));
                let bytes = match held {
                    Some(held) if held.len() == remaining => held,
                    // Joining copies the unread span while the tail is still
                    // live, so only do it when the tail is no larger than the
                    // bytes it saves re-fetching.
                    Some(held) if remaining <= 2 * held.len() => {
                        self.fetch_after(*cursor, held).await?
                    }
                    _ => self.fetch_after(*cursor, Bytes::new()).await?,
                };
                *cursor = self.size;
                *prefetch = None;
                Ok(bytes)
            }
        }
    }

    async fn fetch_after(&self, cursor: u64, held: Bytes) -> Result<Bytes> {
        let physical = self.read_phys_range(cursor + held.len() as u64..self.size)?;
        let rest = self
            .source
            .read_ranges(vec![physical])
            .await?
            .pop()
            .unwrap();
        if held.is_empty() {
            return Ok(rest);
        }
        Ok([held, rest].concat().into())
    }

    /// Read up to `len` bytes from the current cursor position
    ///
    /// After this call the cursor will be pointing to the end of
    /// the read data.
    pub async fn read_up_to(&self, len: usize) -> Result<bytes::Bytes> {
        let mut state = self.state.lock().await;
        match state.deref_mut() {
            BlobFileState::Closed => Err(Error::invalid_input(
                "Blob file is already closed".to_string(),
            )),
            BlobFileState::Open {
                cursor,
                buffer_size,
                prefetch,
            } => {
                if *cursor >= self.size || len == 0 {
                    return Ok(Bytes::new());
                }
                let remaining = (self.size - *cursor) as usize;
                let read_len = len.min(remaining);
                let prefix = prefetch
                    .as_ref()
                    .and_then(|p| p.slice_from(*cursor, read_len));
                let prefix_len = prefix.as_ref().map(Bytes::len).unwrap_or(0);
                if prefix_len == read_len {
                    *cursor += read_len as u64;
                    return Ok(prefix.unwrap_or_default());
                }

                let fetch_cursor = *cursor + prefix_len as u64;
                let still_need = read_len - prefix_len;
                let remaining = (self.size - fetch_cursor) as usize;
                let fetch_len = if *buffer_size == 0 {
                    still_need
                } else {
                    remaining.min((*buffer_size).max(still_need))
                };
                let start = self.position.checked_add(fetch_cursor).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Blob cursor overflowed physical position: base={} cursor={}",
                        self.position, fetch_cursor
                    ))
                })?;
                let end = start.checked_add(fetch_len as u64).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Blob fetch end overflowed physical position: start={} fetch_len={}",
                        start, fetch_len
                    ))
                })?;
                let data = self
                    .source
                    .read_ranges(vec![start..end])
                    .await?
                    .pop()
                    .unwrap_or_default();
                let result = data.slice(0..still_need.min(data.len()));
                if *buffer_size > 0 {
                    *prefetch = Some(BlobPrefetch {
                        start: fetch_cursor,
                        bytes: data,
                    });
                } else {
                    *prefetch = None;
                }
                *cursor = fetch_cursor + result.len() as u64;
                match prefix {
                    Some(hit) => {
                        let mut out = Vec::with_capacity(hit.len() + result.len());
                        out.extend_from_slice(&hit);
                        out.extend_from_slice(&result);
                        Ok(Bytes::from(out))
                    }
                    None => Ok(result),
                }
            }
        }
    }

    /// Sequential read-ahead size in bytes. `0` disables read-ahead. Changing
    /// the size drops any unused prefetch.
    pub async fn set_buffer_size(&self, buffer_size: usize) -> Result<()> {
        let mut state = self.state.lock().await;
        match state.deref_mut() {
            BlobFileState::Open {
                buffer_size: size,
                prefetch,
                ..
            } => {
                if *size != buffer_size {
                    *size = buffer_size;
                    *prefetch = None;
                }
                Ok(())
            }
            BlobFileState::Closed => Err(Error::invalid_input(
                "Blob file is already closed".to_string(),
            )),
        }
    }

    /// Seek to a new cursor position in the file
    pub async fn seek(&self, new_cursor: u64) -> Result<()> {
        let mut state = self.state.lock().await;
        match state.deref_mut() {
            BlobFileState::Open {
                cursor, prefetch, ..
            } => {
                *cursor = new_cursor;
                match prefetch.as_ref() {
                    Some(p) if p.contains(new_cursor) => {}
                    _ => *prefetch = None,
                }
                Ok(())
            }
            BlobFileState::Closed => Err(Error::invalid_input(
                "Blob file is already closed".to_string(),
            )),
        }
    }

    /// Return the current cursor position in the file
    pub async fn tell(&self) -> Result<u64> {
        let state = self.state.lock().await;
        match &*state {
            BlobFileState::Open { cursor, .. } => Ok(*cursor),
            BlobFileState::Closed => Err(Error::invalid_input(
                "Blob file is already closed".to_string(),
            )),
        }
    }

    /// Number of range-read submissions on this handle's backing object.
    ///
    /// Shared with other handles on the same object. Not a count of
    /// object-store fetches. Test instrumentation, not a supported metric.
    #[doc(hidden)]
    pub fn range_submission_count(&self) -> usize {
        self.source.range_submissions.load(Ordering::Relaxed)
    }

    /// Return the size of the blob file in bytes
    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn position(&self) -> u64 {
        self.position
    }

    /// Store binding for the resolved backing object.
    #[doc(hidden)]
    pub fn object_store(&self) -> &Arc<ObjectStore> {
        &self.source.object_store
    }

    pub fn data_path(&self) -> &Path {
        &self.source.path
    }

    pub fn kind(&self) -> BlobKind {
        self.kind
    }

    /// The registered storage base used to locate this blob within its dataset.
    ///
    /// Inline and sidecar blobs inherit the data file's base. Managed blobs use
    /// their explicit base when present and otherwise inherit the data file's base.
    /// External blobs with a relative URI use the base named by their descriptor.
    /// Returns `None` for the default dataset location, an absolute External URI,
    /// or a handle constructed directly from an object store and path.
    ///
    /// ```
    /// # use lance_table_runtime::{Table, blob::BlobFile};
    /// # fn example(table: &Table, file: &BlobFile) {
    /// let base = file.base_id().and_then(|id| table.base_paths().get(&id));
    /// # }
    /// ```
    pub fn base_id(&self) -> Option<u32> {
        self.base_id
    }

    pub fn uri(&self) -> Option<&str> {
        self.uri.as_deref()
    }
}

/// Bytes fetched by a sequential [`BlobFile`] read, still valid for later
/// `read_up_to` calls until the cursor leaves this span.
#[derive(Debug)]
struct BlobPrefetch {
    start: u64,
    bytes: Bytes,
}

impl BlobPrefetch {
    fn slice_from(&self, cursor: u64, want: usize) -> Option<Bytes> {
        if cursor < self.start {
            return None;
        }
        let offset = (cursor - self.start) as usize;
        if offset >= self.bytes.len() {
            return None;
        }
        let available = self.bytes.len() - offset;
        let n = want.min(available);
        if n == 0 {
            return None;
        }
        Some(self.bytes.slice(offset..offset + n))
    }

    fn contains(&self, cursor: u64) -> bool {
        let end = self.start + self.bytes.len() as u64;
        cursor >= self.start && cursor < end
    }
}

/// Mutable state for a [`BlobFile`] cursor.
///
/// The cursor is logical to the blob slice, not the backing object. Once closed,
/// subsequent cursor-based and range-based reads are rejected, but reads that
/// were already in flight may still complete.
#[derive(Debug)]
enum BlobFileState {
    Open {
        cursor: u64,
        buffer_size: usize,
        prefetch: Option<BlobPrefetch>,
    },
    Closed,
}

impl BlobFileState {
    fn new() -> Self {
        Self::Open {
            cursor: 0,
            buffer_size: BlobFile::DEFAULT_READ_BUFFER_SIZE,
            prefetch: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::blob::read::{BlobEntry, BlobReadRange, execute_blob_entries};
    use crate::blob::test_util::{gated_range_store, recording_range_store};
    use lance_core::utils::tempfile::TempDir;
    use rstest::rstest;
    use std::time::Duration;

    #[tokio::test]
    async fn test_blob_file_read_empty_range_returns_empty_bytes() {
        let store = Arc::new(ObjectStore::memory());
        let path = Path::from("blobs/test.bin");

        let empty_blob = BlobFile::new_packed(store.clone(), path.clone(), 1, 0);
        assert!(empty_blob.read().await.unwrap().is_empty());
        assert!(empty_blob.read_up_to(16).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_blob_file_read_tracks_relative_cursor() {
        let test_dir = TempDir::default();
        let file_path = test_dir.std_path().join("blob.bin");
        std::fs::write(&file_path, b"abcd").unwrap();

        let path = Path::from_absolute_path(file_path).unwrap();
        let blob = BlobFile::new_packed(Arc::new(ObjectStore::local()), path, 1, 2);

        assert_eq!(blob.read().await.unwrap().as_ref(), b"bc");
        assert_eq!(blob.tell().await.unwrap(), 2);
        assert!(blob.read().await.unwrap().is_empty());
        assert!(blob.read_up_to(1).await.unwrap().is_empty());
        assert_eq!(blob.tell().await.unwrap(), 2);
    }

    #[tokio::test]
    async fn test_blob_file_read_range_does_not_change_cursor() {
        let (store, _) = recording_range_store(Bytes::from_static(b"abcdefgh"));
        let path = Path::from("blobs/test.bin");
        let blob = BlobFile::new_packed(store, path, 1, 6);

        let bytes = blob.read_range(2..5).await.unwrap();
        assert_eq!(bytes.as_ref(), b"def");
        assert_eq!(blob.tell().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_blob_file_read_ranges_preserves_input_order() {
        let (store, inner) = recording_range_store(Bytes::from_static(b"abcdefghij"));
        let path = Path::from("blobs/test.bin");
        let blob = BlobFile::new_packed(store, path, 1, 6);

        let chunks = blob.read_ranges(&[4..6, 0..2, 2..4, 2..2]).await.unwrap();
        assert_eq!(chunks[0].as_ref(), b"fg");
        assert_eq!(chunks[1].as_ref(), b"bc");
        assert_eq!(chunks[2].as_ref(), b"de");
        assert!(chunks[3].is_empty());
        assert_eq!(inner.requested_ranges(), vec![1..7]);
        assert_eq!(inner.head_requests.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_blob_file_read_range_rejects_out_of_bounds() {
        let (store, inner) = recording_range_store(Bytes::from_static(b"abcdef"));
        let path = Path::from("blobs/test.bin");
        let blob = BlobFile::new_packed(store, path, 0, 4);

        let err = blob.read_range(1..5).await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }));
        assert!(err.to_string().contains("exceeds blob size"));
        assert!(inner.requested_ranges().is_empty());
        assert_eq!(inner.head_requests.load(Ordering::Relaxed), 0);
    }

    #[rstest]
    #[case::partial(3)]
    #[case::empty(0)]
    #[tokio::test]
    async fn test_blob_range_reads_reject_truncated_backing_object(#[case] truncated_size: u64) {
        let test_dir = TempDir::default();
        let file_path = test_dir.std_path().join("blob.bin");
        std::fs::write(&file_path, b"abcdefgh").unwrap();
        let path = Path::from_absolute_path(&file_path).unwrap();
        let blob = BlobFile::new_packed(Arc::new(ObjectStore::local()), path, 2, 4);
        std::fs::OpenOptions::new()
            .write(true)
            .open(&file_path)
            .unwrap()
            .set_len(truncated_size)
            .unwrap();

        let error = blob.read_range(0..4).await.unwrap_err();
        assert!(matches!(error, Error::IO { .. }), "{error:?}");
        assert!(
            error.to_string().contains("failed to fill whole buffer"),
            "{error}"
        );

        let error = execute_blob_entries(
            vec![BlobEntry {
                selection_index: 0,
                file: blob,
                requested_range: Some(BlobReadRange::new(0, 4)),
            }],
            1,
            None,
        )
        .await
        .unwrap_err();
        assert!(matches!(error, Error::IO { .. }), "{error:?}");
        assert!(
            error.to_string().contains("failed to fill whole buffer"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn test_blob_file_staggered_same_source_ranges_run_concurrently() {
        let (store, inner, gate) = gated_range_store(
            Bytes::from_static(b"abcdefgh"),
            "mock://same-source/blob-range-tests",
        );
        let blob = Arc::new(BlobFile::new_dedicated(
            store,
            Path::from("blobs/test.bin"),
            8,
        ));

        let first_blob = blob.clone();
        let first = tokio::spawn(async move { first_blob.read_range(0..3).await });
        inner.wait_for_blob_requests(1).await;

        let second_blob = blob.clone();
        let second = tokio::spawn(async move { second_blob.read_range(4..7).await });
        tokio::time::timeout(Duration::from_secs(1), inner.wait_for_blob_requests(2))
            .await
            .expect("the second same-source range should start while the first is in flight");

        assert_eq!(inner.peak_active_blob_requests(), 2);
        gate.add_permits(2);
        assert_eq!(first.await.unwrap().unwrap().as_ref(), b"abc");
        assert_eq!(second.await.unwrap().unwrap().as_ref(), b"efg");
    }

    #[tokio::test]
    async fn test_blob_file_staggered_multiple_source_ranges_run_concurrently() {
        let (first_store, first_inner, first_gate) = gated_range_store(
            Bytes::from_static(b"abcdefgh"),
            "mock://first-source/blob-range-tests",
        );
        let (second_store, second_inner, second_gate) = gated_range_store(
            Bytes::from_static(b"ijklmnop"),
            "mock://second-source/blob-range-tests",
        );
        let first_blob = Arc::new(BlobFile::new_dedicated(
            first_store,
            Path::from("blobs/first.bin"),
            8,
        ));
        let second_blob = Arc::new(BlobFile::new_dedicated(
            second_store,
            Path::from("blobs/second.bin"),
            8,
        ));

        let first = tokio::spawn(async move { first_blob.read_range(0..3).await });
        first_inner.wait_for_blob_requests(1).await;

        let second = tokio::spawn(async move { second_blob.read_range(4..7).await });
        tokio::time::timeout(
            Duration::from_secs(1),
            second_inner.wait_for_blob_requests(1),
        )
        .await
        .expect("a read from another source should start while the first is in flight");

        first_gate.add_permits(1);
        second_gate.add_permits(1);
        assert_eq!(first.await.unwrap().unwrap().as_ref(), b"abc");
        assert_eq!(second.await.unwrap().unwrap().as_ref(), b"mno");
    }

    #[tokio::test]
    async fn test_blob_files_share_source_and_coalesce() {
        let (store, inner) = recording_range_store(Bytes::from_static(b"abcdefghij"));
        let source = Arc::new(BlobSource::new(store, Path::from("blobs/test.bin")));
        let blob1 = BlobFile::with_source(source.clone(), 1, 3, BlobKind::Packed, None);
        let blob2 = BlobFile::with_source(source, 4, 3, BlobKind::Packed, None);

        let (data1, data2) = tokio::join!(blob1.read(), blob2.read());
        assert_eq!(data1.unwrap().as_ref(), b"bcd");
        assert_eq!(data2.unwrap().as_ref(), b"efg");
        assert_eq!(inner.requested_blob_ranges(), vec![1..7]);
    }

    #[tokio::test]
    async fn managed_empty_range_does_not_discover_object_length() {
        let (store, recording) = recording_range_store(Bytes::from_static(b"not empty"));
        let source = Arc::new(BlobSource::new(store, Path::from("data/a/payload.blob")));
        let blob = BlobFile::with_source(source, 4, 0, BlobKind::Managed, None);
        assert!(blob.read().await.unwrap().is_empty());
        assert!(blob.read_range(0..0).await.unwrap().is_empty());
        assert_eq!(recording.head_requests.load(Ordering::Relaxed), 0);
        assert!(recording.requested_ranges().is_empty());
    }
}
