// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::io;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::task::Poll;
use std::time::Instant;

use crate::object_store::ObjectStore as LanceObjectStore;
use async_trait::async_trait;
use bytes::Bytes;
use futures::FutureExt;
use futures::future::BoxFuture;
use object_store::{MultipartUpload, ObjectStoreExt};
use object_store::{ObjectStore, path::Path};
use tokio::io::{AsyncWrite, AsyncWriteExt};
use tokio::task::JoinSet;

use lance_core::{Error, Result};
use tracing::Instrument;

use crate::traits::Writer;
use crate::utils::tracking_store::{IOTracker, IoMetricsGuard};
use tokio::runtime::Handle;

/// Start at 5MB.
const INITIAL_UPLOAD_STEP: usize = 1024 * 1024 * 5;

pub(crate) fn max_upload_parallelism() -> usize {
    static MAX_UPLOAD_PARALLELISM: OnceLock<usize> = OnceLock::new();
    *MAX_UPLOAD_PARALLELISM.get_or_init(|| {
        std::env::var("LANCE_UPLOAD_CONCURRENCY")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(10)
    })
}

/// Maximum body size for a single S3 PUT: strictly less than 5 GiB.
/// AWS rejects single-PUT bodies of exactly 5 GiB (= 5 * 1024^3) with
/// `EntityTooLarge`, so we clamp `LANCE_INITIAL_UPLOAD_SIZE` one byte
/// below that threshold to keep the buffer-fills-to-clamp single-PUT
/// path safe. See lance#6750 for the related txn-file write fix.
const MAX_UPLOAD_PART_SIZE: usize = 1024 * 1024 * 1024 * 5 - 1;

/// Clamps a requested upload part size to the valid [5MB, 5GB] range.
/// Returns the clamped value and whether clamping was necessary.
fn clamp_initial_upload_size(raw: usize) -> (usize, bool) {
    let clamped = raw.clamp(INITIAL_UPLOAD_STEP, MAX_UPLOAD_PART_SIZE);
    (clamped, clamped != raw)
}

pub(crate) fn initial_upload_size() -> usize {
    static LANCE_INITIAL_UPLOAD_SIZE: OnceLock<usize> = OnceLock::new();
    *LANCE_INITIAL_UPLOAD_SIZE.get_or_init(|| {
        let Some(raw) = std::env::var("LANCE_INITIAL_UPLOAD_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        else {
            return INITIAL_UPLOAD_STEP;
        };
        let (clamped, was_clamped) = clamp_initial_upload_size(raw);
        if was_clamped {
            // OnceLock caches the result, so this warning fires at most once per process.
            tracing::warn!(
                requested = raw,
                clamped,
                "LANCE_INITIAL_UPLOAD_SIZE must be between 5MB and 5GB; clamping to valid range"
            );
        }
        clamped
    })
}

/// Writer to an object in an object store.
///
/// If the object is small enough, the writer will upload the object in a single
/// PUT request. If the object is larger, the writer will create a multipart
/// upload and upload parts in parallel.
///
/// Parts stay in flight across writes and flushes, so a writer can hold up to
/// `LANCE_UPLOAD_CONCURRENCY` part bodies in memory at once. With a large
/// `LANCE_INITIAL_UPLOAD_SIZE` that product is what bounds the writer's
/// footprint, not the part size alone.
///
/// This implements the `AsyncWrite` trait.
pub struct ObjectWriter {
    state: UploadState,
    path: Arc<Path>,
    cursor: usize,
    buffer: Vec<u8>,
    // TODO: use constant size to support R2
    use_constant_size_upload_parts: bool,
}

#[derive(Debug, Clone, Default)]
pub struct WriteResult {
    pub size: usize,
    pub e_tag: Option<String>,
}

/// An object-store upload failure, annotated with what Lance was uploading.
///
/// `object_store` reports its own elapsed time, but its clock starts inside
/// `RetryContext::new`, which runs on the *first poll* of the request future.
/// The `elapsed` reported here is measured from the moment Lance handed the
/// request to the uploader, so the two together tell a slow request (both
/// durations agree) apart from one whose task sat unpolled before it ever
/// issued (this duration is much larger). That distinction is what identifies
/// runtime starvation as the cause of a whole-request timeout, and it is not
/// recoverable from the object-store error alone.
#[derive(Debug)]
struct UploadFailure {
    context: String,
    /// The kind `into_io_error` restores. Without it every contextualized
    /// failure would collapse to `ErrorKind::Other`, changing what callers
    /// matching on the kind observe.
    kind: io::ErrorKind,
    source: Box<dyn std::error::Error + Send + Sync>,
}

impl UploadFailure {
    /// Wraps an object store error.
    ///
    /// The `io::ErrorKind` is taken from `object_store`'s own conversion rather
    /// than a local copy of its mapping, and the error itself is kept as the
    /// source, so both callers matching on the kind and `Error::is_not_found`
    /// (which downcasts along the source chain) keep working.
    fn new(context: String, source: object_store::Error) -> Self {
        let mapped = io::Error::from(source);
        let kind = mapped.kind();
        let source: Box<dyn std::error::Error + Send + Sync> =
            match mapped.downcast::<object_store::Error>() {
                Ok(source) => Box::new(source),
                Err(mapped) => Box::new(mapped),
            };
        Self {
            context,
            kind,
            source,
        }
    }

    /// Wraps a failure that carries no object store error to map a kind from.
    fn from_task(context: String, source: tokio::task::JoinError) -> Self {
        Self {
            context,
            kind: io::ErrorKind::Other,
            source: Box::new(source),
        }
    }

    fn into_io_error(self) -> io::Error {
        let kind = self.kind;
        io::Error::new(kind, self)
    }
}

impl std::fmt::Display for UploadFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.context, self.source)
    }
}

impl std::error::Error for UploadFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.source.as_ref())
    }
}

type UploadResult<T> = std::result::Result<T, UploadFailure>;

/// Identifies a single part upload, for its failure message.
struct PartUpload {
    path: Arc<Path>,
    part_idx: u16,
    /// Concurrent part uploads, counting this one, when it was submitted.
    parts_in_flight: usize,
    /// The part size in effect, which is the buffer capacity this part was
    /// filled to. [`ObjectWriter::next_part_buffer`] grows the part size every
    /// 100 parts so one upload can cover a very large object within the
    /// 10,000-part limit, so on a long upload this is a multiple of
    /// `LANCE_INITIAL_UPLOAD_SIZE` rather than equal to it. A body smaller than
    /// this is the final flush.
    part_size: usize,
}

/// Describes the upload knobs in effect, for inclusion in failure messages.
///
/// Both are process-global and read from the environment, so a failure that is
/// sensitive to either is impossible to interpret without them. Note that
/// `LANCE_INITIAL_UPLOAD_SIZE` is the starting part size, not the size in
/// effect: see [`PartUpload::part_size`].
fn upload_settings() -> String {
    format!(
        "LANCE_INITIAL_UPLOAD_SIZE={} bytes, LANCE_UPLOAD_CONCURRENCY={}",
        initial_upload_size(),
        max_upload_parallelism()
    )
}

enum UploadState {
    /// The writer has been opened but no data has been written yet. Will be in
    /// this state until the buffer is full or the writer is shut down.
    Started(Arc<dyn ObjectStore>),
    /// The writer is in the process of creating a multipart upload.
    CreatingUpload(BoxFuture<'static, UploadResult<Box<dyn MultipartUpload>>>),
    /// The writer is in the process of uploading parts.
    InProgress {
        part_idx: u16,
        upload: Box<dyn MultipartUpload>,
        futures: JoinSet<UploadResult<()>>,
    },
    /// The writer is in the process of uploading data in a single PUT request.
    /// This happens when shutdown is called before the buffer is full.
    PuttingSingle(BoxFuture<'static, UploadResult<WriteResult>>),
    /// The writer is in the process of completing the multipart upload.
    Completing(BoxFuture<'static, UploadResult<WriteResult>>),
    /// The writer has been shut down and all data has been written.
    Done(WriteResult),
}

/// Methods for state transitions.
impl UploadState {
    fn started_to_putting_single(&mut self, path: Arc<Path>, buffer: Vec<u8>) {
        // To get owned self, we temporarily swap with Done.
        let this = std::mem::replace(self, Self::Done(WriteResult::default()));
        *self = match this {
            Self::Started(store) => {
                tracing::Span::current().record("part_count", 1_u64);
                let started_at = Instant::now();
                let fut = async move {
                    let size = buffer.len();
                    let res = store.put(&path, buffer.into()).await.map_err(|source| {
                        UploadFailure::new(
                            format!(
                                "single PUT of {path} failed after {:?} ({size} bytes, {})",
                                started_at.elapsed(),
                                upload_settings()
                            ),
                            source,
                        )
                    })?;
                    Ok(WriteResult {
                        size,
                        e_tag: res.e_tag,
                    })
                };
                Self::PuttingSingle(Box::pin(fut))
            }
            _ => unreachable!(),
        }
    }

    fn in_progress_to_completing(&mut self, path: Arc<Path>, bytes_written: usize) {
        // To get owned self, we temporarily swap with Done.
        let this = std::mem::replace(self, Self::Done(WriteResult::default()));
        *self = match this {
            Self::InProgress {
                mut upload,
                futures,
                part_idx,
            } => {
                debug_assert!(futures.is_empty());
                tracing::Span::current().record("part_count", part_idx as u64);
                let started_at = Instant::now();
                let fut = async move {
                    let res = upload.complete().await.map_err(|source| {
                        UploadFailure::new(
                            format!(
                                "completing multipart upload of {path} failed after {:?} \
                                 ({part_idx} parts, {bytes_written} bytes, {})",
                                started_at.elapsed(),
                                upload_settings()
                            ),
                            source,
                        )
                    })?;
                    Ok(WriteResult {
                        size: 0, // This will be set properly later.
                        e_tag: res.e_tag,
                    })
                };
                Self::Completing(Box::pin(fut))
            }
            _ => unreachable!(),
        };
    }
}

impl ObjectWriter {
    pub async fn new(object_store: &LanceObjectStore, path: &Path) -> Result<Self> {
        Ok(Self {
            state: UploadState::Started(object_store.inner.clone()),
            cursor: 0,
            path: Arc::new(path.clone()),
            buffer: Vec::with_capacity(initial_upload_size()),
            use_constant_size_upload_parts: object_store.use_constant_size_upload_parts,
        })
    }

    /// Returns the contents of `buffer` as a `Bytes` object and resets `buffer`.
    /// The new capacity of `buffer` is determined by the current part index.
    fn next_part_buffer(buffer: &mut Vec<u8>, part_idx: u16, constant_upload_size: bool) -> Bytes {
        let new_capacity = if constant_upload_size {
            // The store does not support variable part sizes, so use the initial size.
            initial_upload_size()
        } else {
            // Increase the upload size every 100 parts. This gives maximum part size of 2.5TB.
            initial_upload_size().max(((part_idx / 100) as usize + 1) * INITIAL_UPLOAD_STEP)
        };
        let new_buffer = Vec::with_capacity(new_capacity);
        let part = std::mem::replace(buffer, new_buffer);
        Bytes::from(part)
    }

    fn put_part(
        upload: &mut dyn MultipartUpload,
        buffer: Bytes,
        part: PartUpload,
    ) -> BoxFuture<'static, UploadResult<()>> {
        let body_size = buffer.len();
        log::debug!("MultipartUpload submitting part with {} bytes", body_size);
        // Stamped before the future is spawned so the reported duration covers
        // any time the task spent waiting to be polled, not just the request.
        let queued_at = Instant::now();
        let fut = upload.put_part(buffer.into());
        Box::pin(async move {
            fut.await.map_err(|source| {
                let PartUpload {
                    path,
                    part_idx,
                    parts_in_flight,
                    part_size,
                } = part;
                UploadFailure::new(
                    format!(
                        "multipart upload of part {part_idx} of {path} failed after {:?} \
                         ({body_size} bytes, part_size={part_size} bytes, \
                          parts_in_flight={parts_in_flight} at submission, {})",
                        queued_at.elapsed(),
                        upload_settings()
                    ),
                    source,
                )
            })
        })
    }

    fn poll_tasks(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::result::Result<(), io::Error> {
        let mut_self = &mut *self;
        loop {
            match &mut mut_self.state {
                UploadState::Started(_) | UploadState::Done(_) => break,
                UploadState::CreatingUpload(fut) => match fut.poll_unpin(cx) {
                    Poll::Ready(Ok(mut upload)) => {
                        let mut futures = JoinSet::new();

                        // Read before the buffer is swapped out: capacity is the
                        // part size this body was filled to.
                        let part_size = mut_self.buffer.capacity();
                        let data = Self::next_part_buffer(
                            &mut mut_self.buffer,
                            0,
                            mut_self.use_constant_size_upload_parts,
                        );
                        futures.spawn(Self::put_part(
                            upload.as_mut(),
                            data,
                            PartUpload {
                                path: mut_self.path.clone(),
                                part_idx: 0,
                                parts_in_flight: 1,
                                part_size,
                            },
                        ));

                        mut_self.state = UploadState::InProgress {
                            part_idx: 1, // We just used 0
                            futures,
                            upload,
                        };
                    }
                    Poll::Ready(Err(err)) => return Err(err.into_io_error()),
                    Poll::Pending => break,
                },
                UploadState::InProgress { futures, .. } => {
                    while let Poll::Ready(Some(res)) = futures.poll_join_next(cx) {
                        match res {
                            Ok(Ok(())) => {}
                            Err(err) => {
                                return Err(UploadFailure::from_task(
                                    format!(
                                        "multipart upload task for {} did not complete",
                                        mut_self.path
                                    ),
                                    err,
                                )
                                .into_io_error());
                            }
                            Ok(Err(err)) => return Err(err.into_io_error()),
                        }
                    }
                    break;
                }
                UploadState::PuttingSingle(fut) | UploadState::Completing(fut) => {
                    match fut.poll_unpin(cx) {
                        Poll::Ready(Ok(mut res)) => {
                            res.size = mut_self.cursor;
                            mut_self.state = UploadState::Done(res)
                        }
                        Poll::Ready(Err(err)) => return Err(err.into_io_error()),
                        Poll::Pending => break,
                    }
                }
            }
        }
        Ok(())
    }

    pub async fn abort(&mut self) {
        let state = std::mem::replace(&mut self.state, UploadState::Done(WriteResult::default()));
        if let UploadState::InProgress { mut upload, .. } = state {
            let _ = upload.abort().await;
        }
    }
}

impl Drop for ObjectWriter {
    fn drop(&mut self) {
        // If there is a multipart upload started but not finished, we should abort it.
        if matches!(self.state, UploadState::InProgress { .. }) {
            // Take ownership of the state.
            let state =
                std::mem::replace(&mut self.state, UploadState::Done(WriteResult::default()));
            if let UploadState::InProgress { mut upload, .. } = state
                && let Ok(handle) = Handle::try_current()
            {
                handle.spawn(async move {
                    let _ = upload.abort().await;
                });
            }
        }
    }
}

impl AsyncWrite for ObjectWriter {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::result::Result<usize, std::io::Error>> {
        self.as_mut().poll_tasks(cx)?;

        // Fill buffer up to remaining capacity.
        let remaining_capacity = self.buffer.capacity() - self.buffer.len();
        let bytes_to_write = std::cmp::min(remaining_capacity, buf.len());
        self.buffer.extend_from_slice(&buf[..bytes_to_write]);
        self.cursor += bytes_to_write;

        // Rust needs a little help to borrow self mutably and immutably at the same time
        // through a Pin.
        let mut_self = &mut *self;

        // Instantiate next request, if available.
        if mut_self.buffer.capacity() == mut_self.buffer.len() {
            match &mut mut_self.state {
                UploadState::Started(store) => {
                    let path = mut_self.path.clone();
                    let store = store.clone();
                    let started_at = Instant::now();
                    let fut = Box::pin(async move {
                        store.put_multipart(path.as_ref()).await.map_err(|source| {
                            UploadFailure::new(
                                format!(
                                    "failed to create multipart upload for {path} after {:?} ({})",
                                    started_at.elapsed(),
                                    upload_settings()
                                ),
                                source,
                            )
                        })
                    });
                    self.state = UploadState::CreatingUpload(fut);
                }
                // TODO: Make max concurrency configurable from storage options.
                UploadState::InProgress {
                    upload,
                    part_idx,
                    futures,
                    ..
                } if futures.len() < max_upload_parallelism() => {
                    // Read before the buffer is swapped out: capacity is the
                    // part size this body was filled to, which grows as the
                    // upload progresses.
                    let part_size = mut_self.buffer.capacity();
                    let data = Self::next_part_buffer(
                        &mut mut_self.buffer,
                        *part_idx,
                        mut_self.use_constant_size_upload_parts,
                    );
                    let part = PartUpload {
                        path: mut_self.path.clone(),
                        part_idx: *part_idx,
                        parts_in_flight: futures.len() + 1,
                        part_size,
                    };
                    futures.spawn(
                        Self::put_part(upload.as_mut(), data, part)
                            .instrument(tracing::Span::current()),
                    );
                    *part_idx += 1;
                }
                _ => {}
            }
        }

        self.poll_tasks(cx)?;

        match bytes_to_write {
            0 => Poll::Pending,
            _ => Poll::Ready(Ok(bytes_to_write)),
        }
    }

    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::result::Result<(), std::io::Error>> {
        self.as_mut().poll_tasks(cx)?;

        match &self.state {
            UploadState::Started(_) | UploadState::Done(_) => Poll::Ready(Ok(())),
            UploadState::CreatingUpload(_)
            | UploadState::Completing(_)
            | UploadState::PuttingSingle(_) => Poll::Pending,
            // In-flight parts are spawned tasks, so the runtime drives them
            // whether or not this writer is polled again; `poll_tasks` above
            // only reaps them. Waiting for them here would serialize every part
            // upload behind the caller's next batch, because callers flush once
            // per batch. `poll_shutdown` still drains them before completing the
            // upload, which is the only point at which the object becomes
            // readable. Note this never flushed the tail buffer either, so it
            // was not a "all data has reached the destination" barrier to begin
            // with.
            UploadState::InProgress { .. } => Poll::Ready(Ok(())),
        }
    }

    fn poll_shutdown(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::result::Result<(), std::io::Error>> {
        loop {
            self.as_mut().poll_tasks(cx)?;

            // Rust needs a little help to borrow self mutably and immutably at the same time
            // through a Pin.
            let mut_self = &mut *self;
            match &mut mut_self.state {
                UploadState::Done(_) => return Poll::Ready(Ok(())),
                UploadState::CreatingUpload(_)
                | UploadState::PuttingSingle(_)
                | UploadState::Completing(_) => return Poll::Pending,
                UploadState::Started(_) => {
                    // If we didn't start a multipart upload, we can just do a single put.
                    let part = std::mem::take(&mut mut_self.buffer);
                    let path = mut_self.path.clone();
                    self.state.started_to_putting_single(path, part);
                }
                UploadState::InProgress {
                    upload,
                    futures,
                    part_idx,
                } => {
                    // Flush final batch
                    if !mut_self.buffer.is_empty() && futures.len() < max_upload_parallelism() {
                        // We can just use `take` since we don't need the buffer anymore.
                        let part_size = mut_self.buffer.capacity();
                        let data = Bytes::from(std::mem::take(&mut mut_self.buffer));
                        let part = PartUpload {
                            path: mut_self.path.clone(),
                            part_idx: *part_idx,
                            parts_in_flight: futures.len() + 1,
                            part_size,
                        };
                        // Counted like every other part so the part total
                        // reported when completing the upload is accurate.
                        *part_idx += 1;
                        futures.spawn(
                            Self::put_part(upload.as_mut(), data, part)
                                .instrument(tracing::Span::current()),
                        );
                        // We need to go back to beginning of loop to poll the
                        // new feature and get the waker registered on the ctx.
                        continue;
                    }

                    // We handle the transition from in progress to completing here.
                    if futures.is_empty() {
                        let path = mut_self.path.clone();
                        let bytes_written = mut_self.cursor;
                        self.state.in_progress_to_completing(path, bytes_written);
                    } else {
                        return Poll::Pending;
                    }
                }
            }
        }
    }
}

#[async_trait]
impl Writer for ObjectWriter {
    async fn tell(&mut self) -> Result<usize> {
        Ok(self.cursor)
    }

    async fn shutdown(&mut self) -> Result<WriteResult> {
        // Propagated structurally rather than formatted into a message: every
        // failure from this writer already names the path, and stringifying it
        // would flatten the object store error out of the source chain.
        AsyncWriteExt::shutdown(self).await?;
        if let UploadState::Done(result) = &self.state {
            Ok(result.clone())
        } else {
            unreachable!()
        }
    }
}

pub struct LocalWriter {
    path: Path,
    state: LocalWriteState,
}

#[derive(Default)]
enum LocalWriteState {
    Writing(Box<WritingState>),
    Finishing {
        size: usize,
        future: BoxFuture<'static, Result<WriteResult>>,
    },
    Done(WriteResult),
    #[default]
    Poisoned,
}

struct WritingState {
    writer: tokio::io::BufWriter<tokio::fs::File>,
    cursor: usize,
    /// Temp path that auto-deletes on drop. Set to `None` after `persist()`.
    temp_path: tempfile::TempPath,
    io_tracker: Arc<IOTracker>,
    /// The whole file is reported as a single `put`, so this covers everything
    /// from opening the file to it being durable under its final path. A writer
    /// dropped before `persist()` records nothing, like an aborted upload.
    metrics: IoMetricsGuard,
}

impl LocalWriter {
    pub fn new(
        file: tokio::fs::File,
        path: Path,
        temp_path: tempfile::TempPath,
        io_tracker: Arc<IOTracker>,
    ) -> Self {
        Self {
            path,
            state: LocalWriteState::Writing(Box::new(WritingState {
                writer: tokio::io::BufWriter::new(file),
                cursor: 0,
                temp_path,
                metrics: io_tracker.begin_io("put"),
                io_tracker,
            })),
        }
    }

    fn already_closed_err(path: &Path) -> io::Error {
        io::Error::other(format!(
            "cannot write to LocalWriter for {} after shutdown",
            path
        ))
    }

    fn poisoned_err(path: &Path) -> io::Error {
        io::Error::other(format!("LocalWriter for {} is in poisoned state", path))
    }

    async fn persist(
        temp_path: tempfile::TempPath,
        final_path: Path,
        size: usize,
        io_tracker: Arc<IOTracker>,
        metrics: IoMetricsGuard,
    ) -> Result<WriteResult> {
        let local_path = crate::local::to_local_path(&final_path);
        let persisted = tokio::task::spawn_blocking(move || -> Result<String> {
            temp_path.persist(&local_path).map_err(|e| {
                Error::io(format!(
                    "failed to persist temp file to {}: {}",
                    local_path, e.error
                ))
            })?;

            let metadata = std::fs::metadata(&local_path).map_err(|e| {
                Error::io(format!("failed to read metadata for {}: {}", local_path, e))
            })?;
            Ok(get_etag(&metadata))
        })
        .await
        .map_err(|e| Error::io(format!("spawn_blocking failed: {}", e)))
        .and_then(|e_tag| e_tag);

        metrics.record(&persisted, size as u64);
        let e_tag = persisted?;

        io_tracker.record_write("put", final_path, size as u64);

        Ok(WriteResult {
            size,
            e_tag: Some(e_tag),
        })
    }
}

impl AsyncWrite for LocalWriter {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> Poll<std::result::Result<usize, std::io::Error>> {
        if let LocalWriteState::Writing(state) = &mut self.state {
            let poll = Pin::new(&mut state.writer).poll_write(cx, buf);
            if let Poll::Ready(Ok(n)) = &poll {
                state.cursor += *n;
            }
            poll
        } else {
            Poll::Ready(Err(Self::already_closed_err(&self.path)))
        }
    }

    fn poll_flush(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<std::result::Result<(), std::io::Error>> {
        if let LocalWriteState::Writing(state) = &mut self.state {
            Pin::new(&mut state.writer).poll_flush(cx)
        } else {
            Poll::Ready(Err(Self::already_closed_err(&self.path)))
        }
    }

    fn poll_shutdown(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<std::result::Result<(), std::io::Error>> {
        let mut_self = &mut *self;
        loop {
            match &mut mut_self.state {
                LocalWriteState::Writing(state) => {
                    if Pin::new(&mut state.writer).poll_shutdown(cx).is_pending() {
                        return Poll::Pending;
                    }

                    // Write is complete, we can transition to persisting.
                    let LocalWriteState::Writing(state) =
                        std::mem::replace(&mut mut_self.state, LocalWriteState::Poisoned)
                    else {
                        unreachable!()
                    };
                    let size = state.cursor;
                    mut_self.state = LocalWriteState::Finishing {
                        size,
                        future: Box::pin(Self::persist(
                            state.temp_path,
                            mut_self.path.clone(),
                            size,
                            state.io_tracker,
                            state.metrics,
                        )),
                    };
                }
                LocalWriteState::Finishing { future, .. } => match future.poll_unpin(cx) {
                    Poll::Ready(Ok(result)) => mut_self.state = LocalWriteState::Done(result),
                    Poll::Ready(Err(e)) => {
                        return Poll::Ready(Err(io::Error::other(e)));
                    }
                    Poll::Pending => return Poll::Pending,
                },
                LocalWriteState::Done(_) => return Poll::Ready(Ok(())),
                LocalWriteState::Poisoned => {
                    return Poll::Ready(Err(Self::poisoned_err(&self.path)));
                }
            }
        }
    }
}

#[async_trait]
impl Writer for LocalWriter {
    async fn tell(&mut self) -> Result<usize> {
        match &mut self.state {
            LocalWriteState::Writing(state) => Ok(state.cursor),
            LocalWriteState::Finishing { size, .. } => Ok(*size),
            LocalWriteState::Done(result) => Ok(result.size),
            LocalWriteState::Poisoned => Err(Self::poisoned_err(&self.path).into()),
        }
    }

    async fn shutdown(&mut self) -> Result<WriteResult> {
        AsyncWriteExt::shutdown(self).await.map_err(|e| {
            Error::io(format!(
                "failed to shutdown local writer for {}: {}",
                self.path, e
            ))
        })?;

        match &self.state {
            LocalWriteState::Done(result) => Ok(result.clone()),
            _ => unreachable!(),
        }
    }
}

// Based on object store's implementation.
pub fn get_etag(metadata: &std::fs::Metadata) -> String {
    let inode = get_inode(metadata);
    let size = metadata.len();
    let mtime = metadata
        .modified()
        .ok()
        .and_then(|mtime| mtime.duration_since(std::time::SystemTime::UNIX_EPOCH).ok())
        .unwrap_or_default()
        .as_micros();

    // Use an ETag scheme based on that used by many popular HTTP servers
    // <https://httpd.apache.org/docs/2.2/mod/core.html#fileetag>
    format!("{inode:x}-{mtime:x}-{size:x}")
}

#[cfg(unix)]
fn get_inode(metadata: &std::fs::Metadata) -> u64 {
    std::os::unix::fs::MetadataExt::ino(metadata)
}

#[cfg(not(unix))]
fn get_inode(_metadata: &std::fs::Metadata) -> u64 {
    0
}

#[cfg(test)]
mod tests {
    use futures::stream::BoxStream;
    use object_store::{
        CopyOptions, GetOptions, GetResult, ListResult, ObjectMeta, PutMultipartOptions,
        PutOptions, PutPayload, PutResult, RenameOptions, Result as OSResult, UploadPart,
    };
    use std::sync::Mutex;
    use std::time::Duration;
    use tokio::io::AsyncWriteExt;
    use tokio::sync::Semaphore;

    use super::*;

    /// Which stage of an upload the mock store rejects.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum FailAt {
        Nothing,
        CreateMultipart,
        PutPart,
        Complete,
        SinglePut,
    }

    /// What the mock store saw, so a test can assert on uploads that are still
    /// in flight as well as on the object they eventually assemble.
    #[derive(Debug)]
    struct UploadObservations {
        /// One permit per part upload that has begun. A test waits on this
        /// rather than sampling a counter: `write_all` and a non-waiting
        /// `flush` can both complete without ever returning `Pending`, so on a
        /// current-thread runtime the spawned upload tasks may not have run yet.
        started: Semaphore,
        /// `(part index, body)` in completion order. The index is recorded
        /// because it, not completion order, determines the assembled object.
        parts: Mutex<Vec<(usize, Vec<u8>)>>,
    }

    impl Default for UploadObservations {
        fn default() -> Self {
            Self {
                started: Semaphore::new(0),
                parts: Mutex::new(Vec::new()),
            }
        }
    }

    fn rejected(stage: &'static str) -> object_store::Error {
        object_store::Error::Generic {
            store: "FailingUploadStore",
            source: format!("{stage} rejected by test").into(),
        }
    }

    #[derive(Debug)]
    struct FailingUpload {
        fail_at: FailAt,
        /// When set, a part upload does not resolve until the gate is given a
        /// permit, so a test can hold requests in flight.
        gate: Option<Arc<Semaphore>>,
        observations: Arc<UploadObservations>,
        next_part: usize,
    }

    #[async_trait]
    impl MultipartUpload for FailingUpload {
        fn put_part(&mut self, data: PutPayload) -> UploadPart {
            let fails = self.fail_at == FailAt::PutPart;
            let part_idx = self.next_part;
            self.next_part += 1;
            let gate = self.gate.clone();
            let observations = self.observations.clone();
            Box::pin(async move {
                observations.started.add_permits(1);
                if let Some(gate) = gate {
                    // `forget` keeps the permit from being returned on drop, so
                    // adding N permits releases exactly N parts.
                    gate.acquire_owned().await.unwrap().forget();
                }
                if fails {
                    return Err(rejected("part"));
                }
                let body = data
                    .iter()
                    .flat_map(|chunk| chunk.iter().copied())
                    .collect();
                observations.parts.lock().unwrap().push((part_idx, body));
                Ok(())
            })
        }

        async fn complete(&mut self) -> OSResult<PutResult> {
            if self.fail_at == FailAt::Complete {
                Err(rejected("complete"))
            } else {
                Ok(PutResult {
                    e_tag: None,
                    version: None,
                })
            }
        }

        async fn abort(&mut self) -> OSResult<()> {
            Ok(())
        }
    }

    /// Rejects exactly one stage of an upload so each failure site can be
    /// exercised on its own, and optionally holds part uploads open.
    #[derive(Debug)]
    struct FailingUploadStore {
        fail_at: FailAt,
        gate: Option<Arc<Semaphore>>,
        observations: Arc<UploadObservations>,
    }

    impl FailingUploadStore {
        fn new(fail_at: FailAt) -> Self {
            Self {
                fail_at,
                gate: None,
                observations: Arc::new(UploadObservations::default()),
            }
        }

        /// Builds a store whose part uploads stay in flight until the returned
        /// gate is given permits.
        fn gated(fail_at: FailAt) -> (Self, Arc<Semaphore>) {
            let gate = Arc::new(Semaphore::new(0));
            let store = Self {
                fail_at,
                gate: Some(gate.clone()),
                observations: Arc::new(UploadObservations::default()),
            };
            (store, gate)
        }
    }

    impl std::fmt::Display for FailingUploadStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "FailingUploadStore")
        }
    }

    #[async_trait]
    impl ObjectStore for FailingUploadStore {
        async fn put_opts(
            &self,
            _location: &Path,
            _bytes: PutPayload,
            _opts: PutOptions,
        ) -> OSResult<PutResult> {
            if self.fail_at == FailAt::SinglePut {
                Err(rejected("single put"))
            } else {
                Ok(PutResult {
                    e_tag: None,
                    version: None,
                })
            }
        }

        async fn put_multipart_opts(
            &self,
            _location: &Path,
            _opts: PutMultipartOptions,
        ) -> OSResult<Box<dyn MultipartUpload>> {
            if self.fail_at == FailAt::CreateMultipart {
                Err(rejected("create multipart"))
            } else {
                Ok(Box::new(FailingUpload {
                    fail_at: self.fail_at,
                    gate: self.gate.clone(),
                    observations: self.observations.clone(),
                    next_part: 0,
                }))
            }
        }

        async fn get_opts(&self, _location: &Path, _options: GetOptions) -> OSResult<GetResult> {
            unimplemented!()
        }

        fn delete_stream(
            &self,
            _locations: BoxStream<'static, OSResult<Path>>,
        ) -> BoxStream<'static, OSResult<Path>> {
            unimplemented!()
        }

        fn list(&self, _prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
            unimplemented!()
        }

        fn list_with_offset(
            &self,
            _prefix: Option<&Path>,
            _offset: &Path,
        ) -> BoxStream<'static, OSResult<ObjectMeta>> {
            unimplemented!()
        }

        async fn list_with_delimiter(&self, _prefix: Option<&Path>) -> OSResult<ListResult> {
            unimplemented!()
        }

        async fn copy_opts(&self, _from: &Path, _to: &Path, _opts: CopyOptions) -> OSResult<()> {
            unimplemented!()
        }

        async fn rename_opts(
            &self,
            _from: &Path,
            _to: &Path,
            _opts: RenameOptions,
        ) -> OSResult<()> {
            unimplemented!()
        }
    }

    const FAILING_UPLOAD_PATH: &str = "part_7_invert.lance";

    /// Enough bytes for two full multipart parts, so a failing part has a
    /// sibling in flight. Derived from the configured part size rather than the
    /// default, since `LANCE_INITIAL_UPLOAD_SIZE` may raise it.
    fn two_parts() -> usize {
        initial_upload_size() * 2
    }

    /// Drives a write against a store that rejects `fail_at`, returning the
    /// error. The failure can surface either from a write or from shutdown
    /// depending on when the rejected request is reaped, so both are checked.
    async fn failing_upload(fail_at: FailAt, num_bytes: usize) -> io::Error {
        let mut store = LanceObjectStore::memory();
        store.inner = Arc::new(FailingUploadStore::new(fail_at));

        let mut writer = ObjectWriter::new(&store, &Path::from(FAILING_UPLOAD_PATH))
            .await
            .unwrap();
        let buf = vec![0u8; num_bytes];
        match writer.write_all(buf.as_slice()).await {
            Err(err) => err,
            Ok(()) => AsyncWriteExt::shutdown(&mut writer)
                .await
                .expect_err("upload should have failed"),
        }
    }

    #[tokio::test]
    async fn test_part_upload_failure_reports_upload_context() {
        let err = failing_upload(FailAt::PutPart, two_parts()).await;
        let message = err.to_string();

        assert!(
            message.contains("multipart upload of part"),
            "should name the failing stage: {message}"
        );
        assert!(
            message.contains(FAILING_UPLOAD_PATH),
            "should name the object: {message}"
        );
        assert!(
            message.contains(&format!("{} bytes", initial_upload_size())),
            "should report the body size: {message}"
        );
        assert!(
            message.contains(&format!("part_size={} bytes", initial_upload_size())),
            "should report the part size in effect: {message}"
        );
        assert!(
            message.contains("parts_in_flight="),
            "should report upload concurrency in use: {message}"
        );
        assert!(
            message.contains("LANCE_INITIAL_UPLOAD_SIZE")
                && message.contains("LANCE_UPLOAD_CONCURRENCY"),
            "should report the knobs governing the request: {message}"
        );
        assert!(
            message.contains("part rejected by test"),
            "should keep the underlying object store error: {message}"
        );
    }

    // The elapsed time is the whole point of the added context: it is what tells
    // a slow request apart from one whose task was never polled.
    #[tokio::test]
    async fn test_part_upload_failure_reports_elapsed_time() {
        let err = failing_upload(FailAt::PutPart, two_parts()).await;
        let message = err.to_string();
        assert!(
            message.contains("failed after"),
            "should report how long the request took: {message}"
        );
    }

    // `part_size` in the failure message is the live buffer capacity, which is
    // what makes it report the size actually in effect. The part size grows
    // every 100 parts, so on a long upload that diverges from
    // LANCE_INITIAL_UPLOAD_SIZE by a multiple; reporting only the configured
    // value would understate a late part by that factor. Reaching part 100
    // through the writer would mean allocating hundreds of MiB, so the growth
    // is asserted on the buffer the message reads from.
    #[test]
    fn test_part_buffer_capacity_tracks_grown_part_size() {
        let mut buffer = Vec::<u8>::with_capacity(initial_upload_size());
        assert_eq!(buffer.capacity(), initial_upload_size());

        let _ = ObjectWriter::next_part_buffer(&mut buffer, 0, false);
        assert_eq!(
            buffer.capacity(),
            initial_upload_size(),
            "early parts stay at the configured size"
        );

        let _ = ObjectWriter::next_part_buffer(&mut buffer, 100, false);
        assert_eq!(
            buffer.capacity(),
            initial_upload_size().max(2 * INITIAL_UPLOAD_STEP),
            "the part size has grown past the first step"
        );

        // A store pinned to constant part sizes never grows, so the reported
        // size stays equal to the configured one.
        let _ = ObjectWriter::next_part_buffer(&mut buffer, 100, true);
        assert_eq!(buffer.capacity(), initial_upload_size());
    }

    #[tokio::test]
    async fn test_part_upload_failure_preserves_source_chain() {
        let err = failing_upload(FailAt::PutPart, two_parts()).await;

        let failure = err
            .get_ref()
            .expect("io error should carry the upload failure");
        let source = std::error::Error::source(failure)
            .expect("upload failure should expose the object store error");
        assert!(
            source.downcast_ref::<object_store::Error>().is_some(),
            "source should still be the object store error, got: {source}"
        );
    }

    /// Adding context must not flatten the `io::ErrorKind` that `object_store`
    /// maps an error to, since that kind is observable to callers.
    #[test]
    fn test_upload_failure_preserves_error_kind() {
        fn not_found() -> object_store::Error {
            object_store::Error::NotFound {
                path: FAILING_UPLOAD_PATH.to_string(),
                source: "not found".into(),
            }
        }

        let unwrapped = io::Error::from(not_found());
        assert_eq!(unwrapped.kind(), io::ErrorKind::NotFound);

        let wrapped =
            UploadFailure::new("part upload failed".to_string(), not_found()).into_io_error();
        assert_eq!(wrapped.kind(), unwrapped.kind());
    }

    /// `Writer::shutdown` is the public boundary most callers see. The object
    /// store error has to remain reachable through it, not be flattened into a
    /// message.
    #[tokio::test]
    async fn test_writer_shutdown_preserves_object_store_source() {
        let mut store = LanceObjectStore::memory();
        store.inner = Arc::new(FailingUploadStore::new(FailAt::SinglePut));
        let mut writer = ObjectWriter::new(&store, &Path::from(FAILING_UPLOAD_PATH))
            .await
            .unwrap();
        writer.write_all(&[0u8; 256]).await.unwrap();
        let err = Writer::shutdown(&mut writer).await.unwrap_err();

        let mut current: Option<&(dyn std::error::Error + 'static)> = Some(&err);
        let mut found_object_store = false;
        while let Some(source) = current {
            if source.downcast_ref::<object_store::Error>().is_some() {
                found_object_store = true;
                break;
            }
            current = source.source();
        }
        assert!(found_object_store, "source chain was flattened: {err:?}");

        assert!(
            err.to_string().contains(FAILING_UPLOAD_PATH),
            "should still name the object: {err}"
        );
    }

    #[tokio::test]
    async fn test_create_multipart_failure_reports_upload_context() {
        let err = failing_upload(FailAt::CreateMultipart, two_parts()).await;
        let message = err.to_string();

        assert!(
            message.contains("failed to create multipart upload for"),
            "should name the failing stage: {message}"
        );
        assert!(
            message.contains(FAILING_UPLOAD_PATH),
            "should name the object: {message}"
        );
        assert!(
            message.contains("create multipart rejected by test"),
            "should keep the underlying object store error: {message}"
        );
    }

    #[tokio::test]
    async fn test_complete_multipart_failure_reports_upload_context() {
        let num_bytes = two_parts();
        let err = failing_upload(FailAt::Complete, num_bytes).await;
        let message = err.to_string();

        assert!(
            message.contains("completing multipart upload of"),
            "should name the failing stage: {message}"
        );
        assert!(
            message.contains(&format!("{num_bytes} bytes")),
            "should report how much had been written: {message}"
        );
        assert!(
            message.contains("complete rejected by test"),
            "should keep the underlying object store error: {message}"
        );
    }

    #[tokio::test]
    async fn test_single_put_failure_reports_upload_context() {
        // Below the multipart threshold, so shutdown takes the single-PUT path.
        let err = failing_upload(FailAt::SinglePut, 256).await;
        let message = err.to_string();

        assert!(
            message.contains("single PUT of"),
            "should name the failing stage: {message}"
        );
        assert!(
            message.contains(FAILING_UPLOAD_PATH),
            "should name the object: {message}"
        );
        assert!(
            message.contains("256 bytes"),
            "should report the body size: {message}"
        );
        assert!(
            message.contains("single put rejected by test"),
            "should keep the underlying object store error: {message}"
        );
    }

    /// Released permits, comfortably above the part count of any test here so
    /// no test depends on the exact number of parts a payload produces.
    const GATE_RELEASE: usize = 64;

    /// How long a flush is allowed to take before it counts as waiting. A flush
    /// that does not wait resolves immediately; this bound only has to be short
    /// of the test harness timeout.
    const FLUSH_BOUND: Duration = Duration::from_secs(10);

    /// Blocks until a part upload has begun, so the assertions that follow are
    /// made against a request that is genuinely in flight.
    async fn await_part_in_flight(observations: &UploadObservations) {
        tokio::time::timeout(FLUSH_BOUND, observations.started.acquire())
            .await
            .expect("a part upload should have started")
            .unwrap()
            .forget();
    }

    #[tokio::test]
    async fn test_flush_does_not_wait_for_in_flight_parts() {
        let (store, gate) = FailingUploadStore::gated(FailAt::Nothing);
        let observations = store.observations.clone();
        let mut lance_store = LanceObjectStore::memory();
        lance_store.inner = Arc::new(store);

        let mut writer = ObjectWriter::new(&lance_store, &Path::from("gated.lance"))
            .await
            .unwrap();
        // Distinct bytes so a part landing out of order is detectable.
        let payload = (0..two_parts()).map(|i| i as u8).collect::<Vec<_>>();
        writer.write_all(payload.as_slice()).await.unwrap();
        await_part_in_flight(&observations).await;

        tokio::time::timeout(FLUSH_BOUND, AsyncWriteExt::flush(&mut writer))
            .await
            .expect("flush must not wait for in-flight part uploads")
            .unwrap();

        assert!(
            observations.parts.lock().unwrap().is_empty(),
            "no gated part may have completed before the gate opened"
        );

        gate.add_permits(GATE_RELEASE);
        let result = Writer::shutdown(&mut writer).await.unwrap();
        assert_eq!(result.size, payload.len());

        let mut parts = observations.parts.lock().unwrap().clone();
        parts.sort_by_key(|(part_idx, _)| *part_idx);
        let assembled = parts
            .into_iter()
            .flat_map(|(_, body)| body)
            .collect::<Vec<_>>();
        assert_eq!(
            assembled, payload,
            "parts must reassemble into the original bytes"
        );
    }

    #[tokio::test]
    async fn test_part_failure_after_flush_surfaces_at_shutdown() {
        let (store, gate) = FailingUploadStore::gated(FailAt::PutPart);
        let observations = store.observations.clone();
        let mut lance_store = LanceObjectStore::memory();
        lance_store.inner = Arc::new(store);

        let mut writer = ObjectWriter::new(&lance_store, &Path::from(FAILING_UPLOAD_PATH))
            .await
            .unwrap();
        writer
            .write_all(vec![0u8; two_parts()].as_slice())
            .await
            .unwrap();
        await_part_in_flight(&observations).await;
        // The parts are still gated, so nothing has failed yet and flush passes.
        AsyncWriteExt::flush(&mut writer).await.unwrap();

        // Now let them fail. Shutdown is the first place that can report it, so
        // no longer waiting in flush must not lose the error.
        gate.add_permits(GATE_RELEASE);
        let err = AsyncWriteExt::shutdown(&mut writer)
            .await
            .expect_err("a failed part upload must still surface");
        let message = err.to_string();
        assert!(
            message.contains(FAILING_UPLOAD_PATH),
            "should name the object being written: {message}"
        );
    }

    #[tokio::test]
    async fn test_write() {
        let store = LanceObjectStore::memory();

        let mut object_writer = ObjectWriter::new(&store, &Path::from("/foo"))
            .await
            .unwrap();
        assert_eq!(object_writer.tell().await.unwrap(), 0);

        let buf = vec![0; 256];
        assert_eq!(object_writer.write(buf.as_slice()).await.unwrap(), 256);
        assert_eq!(object_writer.tell().await.unwrap(), 256);

        assert_eq!(object_writer.write(buf.as_slice()).await.unwrap(), 256);
        assert_eq!(object_writer.tell().await.unwrap(), 512);

        assert_eq!(object_writer.write(buf.as_slice()).await.unwrap(), 256);
        assert_eq!(object_writer.tell().await.unwrap(), 256 * 3);

        let res = Writer::shutdown(&mut object_writer).await.unwrap();
        assert_eq!(res.size, 256 * 3);

        // Trigger multi part upload
        let mut object_writer = ObjectWriter::new(&store, &Path::from("/bar"))
            .await
            .unwrap();
        let buf = vec![0; INITIAL_UPLOAD_STEP / 3 * 2];
        for i in 0..5 {
            // Write more data to trigger the multipart upload
            // This should be enough to trigger a multipart upload
            object_writer.write_all(buf.as_slice()).await.unwrap();
            // Check the cursor
            assert_eq!(object_writer.tell().await.unwrap(), (i + 1) * buf.len());
        }
        let res = Writer::shutdown(&mut object_writer).await.unwrap();
        assert_eq!(res.size, buf.len() * 5);
    }

    #[tokio::test]
    async fn test_abort_write() {
        let store = LanceObjectStore::memory();

        let mut object_writer = ObjectWriter::new(&store, &Path::from("/foo"))
            .await
            .unwrap();
        object_writer.abort().await;
    }

    #[tokio::test]
    async fn test_local_writer_shutdown() {
        let tmp = lance_core::utils::tempfile::TempStdDir::default();
        let file_path = tmp.join("test_local_writer.bin");
        let os_path = Path::from_absolute_path(&file_path).unwrap();
        let io_tracker = Arc::new(IOTracker::default());

        let named_temp = tempfile::NamedTempFile::new_in(&*tmp).unwrap();
        let temp_file_path = named_temp.path().to_owned();
        let (std_file, temp_path) = named_temp.into_parts();
        let file = tokio::fs::File::from_std(std_file);
        let mut writer = LocalWriter::new(file, os_path, temp_path, io_tracker.clone());

        let data = b"hello local writer";
        writer.write_all(data).await.unwrap();

        // Before shutdown, the final path should not exist
        assert!(!file_path.exists());
        // But the temp file should exist
        assert!(temp_file_path.exists());

        let result = Writer::shutdown(&mut writer).await.unwrap();
        assert_eq!(result.size, data.len());
        assert!(result.e_tag.is_some());
        assert!(!result.e_tag.as_ref().unwrap().is_empty());

        // After shutdown, the final path should exist and temp should be gone
        assert!(file_path.exists());
        assert!(!temp_file_path.exists());

        let stats = io_tracker.stats();
        assert_eq!(stats.write_iops, 1);
        assert_eq!(stats.written_bytes, data.len() as u64);
    }

    #[tokio::test]
    async fn test_local_writer_drop_cleans_up() {
        let tmp = lance_core::utils::tempfile::TempStdDir::default();
        let file_path = tmp.join("test_drop.bin");
        let os_path = Path::from_absolute_path(&file_path).unwrap();
        let io_tracker = Arc::new(IOTracker::default());

        let named_temp = tempfile::NamedTempFile::new_in(&*tmp).unwrap();
        let temp_file_path = named_temp.path().to_owned();
        let (std_file, temp_path) = named_temp.into_parts();
        let file = tokio::fs::File::from_std(std_file);
        let mut writer = LocalWriter::new(file, os_path, temp_path, io_tracker);

        writer.write_all(b"some data").await.unwrap();
        assert!(temp_file_path.exists());

        // Drop without shutdown should clean up the temp file
        drop(writer);
        assert!(!temp_file_path.exists());
        assert!(!file_path.exists());
    }

    #[test]
    fn clamp_initial_upload_size_below_min_is_clamped_up() {
        assert_eq!(clamp_initial_upload_size(0), (INITIAL_UPLOAD_STEP, true));
        assert_eq!(
            clamp_initial_upload_size(INITIAL_UPLOAD_STEP - 1),
            (INITIAL_UPLOAD_STEP, true)
        );
    }

    #[test]
    fn clamp_initial_upload_size_within_range_is_unchanged() {
        assert_eq!(
            clamp_initial_upload_size(INITIAL_UPLOAD_STEP),
            (INITIAL_UPLOAD_STEP, false)
        );
        assert_eq!(
            clamp_initial_upload_size(MAX_UPLOAD_PART_SIZE),
            (MAX_UPLOAD_PART_SIZE, false)
        );
        let mid = INITIAL_UPLOAD_STEP * 8; // 40MB, in range
        assert_eq!(clamp_initial_upload_size(mid), (mid, false));
    }

    #[test]
    fn clamp_initial_upload_size_above_max_is_clamped_down() {
        assert_eq!(
            clamp_initial_upload_size(MAX_UPLOAD_PART_SIZE + 1),
            (MAX_UPLOAD_PART_SIZE, true)
        );
        assert_eq!(
            clamp_initial_upload_size(usize::MAX),
            (MAX_UPLOAD_PART_SIZE, true)
        );
    }

    /// Regression for the foot-gun where `LANCE_INITIAL_UPLOAD_SIZE=5368709120`
    /// (exactly 5 GiB, Pucheng's setting) caused a single-PUT of 5 GiB on
    /// shutdown — which S3 rejects with `EntityTooLarge`. After tightening
    /// `MAX_UPLOAD_PART_SIZE` to 5 GiB - 1, raw 5 GiB must clamp DOWN.
    #[test]
    fn clamp_initial_upload_size_at_5gib_clamps_down() {
        let exactly_5_gib: usize = 5 * 1024 * 1024 * 1024;
        assert_eq!(
            clamp_initial_upload_size(exactly_5_gib),
            (MAX_UPLOAD_PART_SIZE, true)
        );
    }
}
