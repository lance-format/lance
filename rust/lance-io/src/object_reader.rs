// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fs::File;
use std::ops::Range;
use std::sync::Arc;

use crate::local::join_local_io;
#[cfg(windows)]
use crate::local::read_exact_at;
#[cfg(unix)]
use std::os::unix::fs::FileExt;

use bytes::{Bytes, BytesMut};
use futures::{
    FutureExt,
    future::{BoxFuture, Shared},
    stream::{self, StreamExt},
};
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result, error::CloneableError};
use object_store::ObjectStoreExt;
use object_store::{GetOptions, GetResult, ObjectStore, Result as OSResult, path::Path};
use tokio::sync::OnceCell;
use tracing::instrument;

use crate::{
    object_store::DEFAULT_CLOUD_IO_PARALLELISM,
    object_store::hf_throttle::is_huggingface_rate_limited,
    traits::{ByteStream, Reader},
};

mod streaming;

trait StaticGetRange {
    fn path(&self) -> &Path;
    fn get_range(&self) -> BoxFuture<'static, OSResult<GetResult>>;
}

/// A wrapper around an object store and a path that implements a static
/// get_range method by assuming self is stored in an Arc.
struct GetRequest {
    object_store: Arc<dyn ObjectStore>,
    path: Path,
    options: GetOptions,
}

impl StaticGetRange for Arc<GetRequest> {
    fn path(&self) -> &Path {
        &self.path
    }

    fn get_range(&self) -> BoxFuture<'static, OSResult<GetResult>> {
        let store_and_path = self.clone();
        Box::pin(async move {
            store_and_path
                .object_store
                .get_opts(&store_and_path.path, store_and_path.options.clone())
                .await
        })
    }
}

/// Object Reader
///
/// Object Store + Base Path
#[derive(Debug)]
pub struct CloudObjectReader {
    // Object Store.
    pub object_store: Arc<dyn ObjectStore>,
    // File path
    pub path: Path,
    // File size, if known.
    size: OnceCell<usize>,

    block_size: usize,
    io_parallelism: usize,
    download_retry_count: usize,
}

impl DeepSizeOf for CloudObjectReader {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        // Skipping object_store because there is no easy way to do that and it shouldn't be too big
        self.path.as_ref().deep_size_of_children(context)
    }
}

impl CloudObjectReader {
    /// Create an ObjectReader from URI
    pub fn new(
        object_store: Arc<dyn ObjectStore>,
        path: Path,
        block_size: usize,
        known_size: Option<usize>,
        download_retry_count: usize,
    ) -> Result<Self> {
        Ok(Self {
            object_store,
            path,
            size: OnceCell::new_with(known_size),
            block_size,
            io_parallelism: DEFAULT_CLOUD_IO_PARALLELISM,
            download_retry_count,
        })
    }

    /// Override the I/O parallelism this reader advertises.
    ///
    /// `ObjectStore::open` / `open_with_size` pass their normalized effective
    /// parallelism (`LANCE_IO_THREADS` override applied, at least 1) so
    /// consumers that size concurrency windows off the reader honor the
    /// configured request limit instead of the hardcoded cloud default.
    pub fn with_io_parallelism(mut self, io_parallelism: usize) -> Self {
        self.io_parallelism = io_parallelism;
        self
    }
}

// Retries for the initial request are handled by object store, but
// there are no retries for failures that occur during the streaming
// of the response body. Thus we add an outer retry loop here.
async fn do_with_retry<'a, O>(f: impl Fn() -> BoxFuture<'a, OSResult<O>> + Clone) -> OSResult<O> {
    let mut retries = 3;
    loop {
        let f = f.clone();
        match f().await {
            Ok(val) => return Ok(val),
            Err(err) => {
                // OpenDAL owns the Hugging Face rate-limit response: it either
                // spent the retry budget or surfaced an over-limit Retry-After.
                // Repeating the request here would bypass that decision.
                if retries == 0 || is_huggingface_rate_limited(&err) {
                    return Err(err);
                }
                retries -= 1;
            }
        }
    }
}

// We have a separate retry loop here.  This is because object_store does not
// attempt retries on downloads that fail during streaming of the response body.
//
// However, this failure is pretty common (e.g. timeout) and we want to retry in these
// situations.  In addition, we provide additional logging information in these
// failures cases.
async fn do_get_with_outer_retry<E: From<object_store::Error> + From<Error>>(
    download_retry_count: usize,
    get_request: Arc<GetRequest>,
    expected_size: Option<usize>,
    desc: impl Fn() -> String,
) -> std::result::Result<Bytes, E> {
    let mut retries = download_retry_count;
    loop {
        let get_request_clone = get_request.clone();
        let get_result = do_with_retry(move || get_request_clone.get_range())
            .await
            .map_err(E::from)?;
        if let Some(size) = expected_size {
            // Reject stale lengths before consuming or allocating for the body.
            // A partial response cannot establish the complete object's length.
            if get_result.meta.size != size as u64 || get_result.range != (0..size as u64) {
                return Err(Error::invalid_input(format!(
                    "object at '{}' has {} bytes and response range {:?}, expected complete file of {size} bytes",
                    get_request.path(), get_result.meta.size, get_result.range
                )).into());
            }
        }
        let body = match expected_size {
            Some(size) => read_sized_body(get_result, size).await,
            None => get_result.bytes().await,
        };
        match body {
            Ok(bytes) => return Ok(bytes),
            Err(err) => {
                if retries == 0 {
                    log::warn!(
                        "Failed to download {} from {} after {} attempts.  This may indicate that cloud storage is overloaded or your timeout settings are too restrictive.  Error details: {:?}",
                        desc(),
                        get_request.path(),
                        download_retry_count,
                        err
                    );
                    return Err(err.into());
                }
                log::debug!(
                    "Retrying {} from {} (remaining retries: {}).  Error details: {:?}",
                    desc(),
                    get_request.path(),
                    retries,
                    err
                );
                retries -= 1;
            }
        }
    }
}

// Keep the existing small-file buffer bounded even if a store returns a body
// inconsistent with its metadata. Body failures retain the download retry loop.
async fn read_sized_body(result: GetResult, size: usize) -> OSResult<Bytes> {
    let path = result.meta.location.clone();
    let mut stream = result.into_stream();
    let mut bytes = BytesMut::with_capacity(size);
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if chunk.len() > size - bytes.len() {
            return Err(Error::io(format!(
                "object body at '{path}' exceeds expected {size} bytes"
            ))
            .into());
        }
        bytes.extend_from_slice(&chunk);
    }
    if bytes.len() != size {
        return Err(Error::io(format!(
            "object body at '{path}' has {} bytes, expected {size}",
            bytes.len()
        ))
        .into());
    }
    Ok(bytes.freeze())
}

impl Reader for CloudObjectReader {
    fn path(&self) -> &Path {
        &self.path
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn io_parallelism(&self) -> usize {
        self.io_parallelism
    }

    /// Object/File Size.
    fn size(&self) -> BoxFuture<'_, object_store::Result<usize>> {
        Box::pin(async move {
            self.size
                .get_or_try_init(|| async move {
                    let meta =
                        do_with_retry(|| Box::pin(self.object_store.head(&self.path))).await?;
                    Ok(meta.size as usize)
                })
                .await
                .cloned()
        })
    }

    #[instrument(level = "debug", skip(self))]
    fn get_range(&self, range: Range<usize>) -> BoxFuture<'static, OSResult<Bytes>> {
        let range = Range {
            start: range.start as u64,
            end: range.end as u64,
        };
        // Downloading a range is the same two layers as downloading the whole
        // object, so it goes through the same retry loop: `do_with_retry`
        // retries the request, and the outer loop retries a body that stops
        // partway.
        let get_request = Arc::new(GetRequest {
            object_store: self.object_store.clone(),
            path: self.path.clone(),
            options: GetOptions {
                range: Some(range.clone().into()),
                ..Default::default()
            },
        });
        let download_retry_count = self.download_retry_count;
        Box::pin(async move {
            do_get_with_outer_retry::<object_store::Error>(
                download_retry_count,
                get_request,
                None,
                move || format!("range {}..{}", range.start, range.end),
            )
            .await
        })
    }

    #[instrument(level = "debug", skip_all)]
    fn get_all(&self) -> BoxFuture<'_, OSResult<Bytes>> {
        let get_request = Arc::new(GetRequest {
            object_store: self.object_store.clone(),
            path: self.path.clone(),
            options: GetOptions::default(),
        });
        Box::pin(async move {
            do_get_with_outer_retry::<object_store::Error>(
                self.download_retry_count,
                get_request,
                None,
                || "read_all".to_string(),
            )
            .await
        })
    }

    fn get_stream(&self) -> BoxFuture<'_, OSResult<ByteStream>> {
        let get_request = GetRequest {
            object_store: self.object_store.clone(),
            path: self.path.clone(),
            options: GetOptions::default(),
        };
        Box::pin(streaming::get_stream(
            get_request,
            self.download_retry_count,
        ))
    }

    fn get_range_stream(&self, range: Range<usize>) -> BoxFuture<'_, OSResult<ByteStream>> {
        let get_request = GetRequest {
            object_store: self.object_store.clone(),
            path: self.path.clone(),
            options: GetOptions {
                range: Some(
                    Range {
                        start: range.start as u64,
                        end: range.end as u64,
                    }
                    .into(),
                ),
                ..Default::default()
            },
        };
        Box::pin(streaming::get_stream(
            get_request,
            self.download_retry_count,
        ))
    }
}

#[derive(Debug)]
pub struct SmallReaderInner {
    path: Path,
    size: usize,
    state: std::sync::Mutex<SmallReaderState>,
}

/// A reader for a file so small, we just eagerly read it all into memory.
///
/// When created, it represents a future that will read the whole file into memory.
///
/// On the first read call, it will start the read. Multiple threads can call read at the same time.
///
/// Once the read is complete, any thread can call read again to get the result.
#[derive(Clone, Debug)]
pub struct SmallReader {
    inner: Arc<SmallReaderInner>,
}

enum SmallReaderState {
    Loading(Shared<BoxFuture<'static, std::result::Result<Bytes, CloneableError>>>),
    Finished(std::result::Result<Bytes, CloneableError>),
}

impl std::fmt::Debug for SmallReaderState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Loading(_) => write!(f, "Loading"),
            Self::Finished(Ok(data)) => {
                write!(f, "Finished({} bytes)", data.len())
            }
            Self::Finished(Err(err)) => {
                write!(f, "Finished({})", err.inner())
            }
        }
    }
}

impl SmallReader {
    pub fn new(
        store: Arc<dyn ObjectStore>,
        path: Path,
        download_retry_count: usize,
        size: usize,
    ) -> Self {
        let path_ref = path.clone();
        Self::from_download(path, size, async move {
            let reader = CloudObjectReader::new(store, path_ref, 0, None, download_retry_count)?;
            reader.get_all().await.map_err(Error::from)
        })
    }

    /// Admit a checkpoint length before exposing the reader to its caller.
    /// Ordinary known-size readers do not require this stricter contract.
    pub(crate) async fn open_with_expected_size(
        store: Arc<dyn ObjectStore>,
        path: Path,
        download_retry_count: usize,
        expected_size: usize,
    ) -> Result<Self> {
        let request = Arc::new(GetRequest {
            object_store: store,
            path: path.clone(),
            options: GetOptions::default(),
        });
        let reader = Self::from_download(path, expected_size, async move {
            do_get_with_outer_retry::<Error>(
                download_retry_count,
                request,
                Some(expected_size),
                || "read_all".to_string(),
            )
            .await
        });
        reader.inner.wait().await?;
        Ok(reader)
    }

    fn from_download(
        path: Path,
        size: usize,
        download: impl std::future::Future<Output = Result<Bytes>> + Send + 'static,
    ) -> Self {
        let state = SmallReaderState::Loading(
            async move { download.await.map_err(CloneableError::new) }
                .boxed()
                .shared(),
        );
        Self {
            inner: Arc::new(SmallReaderInner {
                path,
                size,
                state: std::sync::Mutex::new(state),
            }),
        }
    }
}

impl SmallReaderInner {
    async fn wait(&self) -> Result<Bytes> {
        let future = {
            let state = self.state.lock().unwrap();
            match &*state {
                SmallReaderState::Loading(future) => future.clone(),
                SmallReaderState::Finished(result) => {
                    return result.clone().map_err(|err| err.into_inner());
                }
            }
        };

        let result = future.await;
        let result_to_return = result.clone().map_err(|err| err.into_inner());
        let mut state = self.state.lock().unwrap();
        if matches!(*state, SmallReaderState::Loading(_)) {
            *state = SmallReaderState::Finished(result);
        }
        result_to_return
    }
}

impl Reader for SmallReader {
    fn path(&self) -> &Path {
        &self.inner.path
    }

    fn block_size(&self) -> usize {
        64 * 1024
    }

    fn io_parallelism(&self) -> usize {
        1024
    }

    /// Object/File Size.
    fn size(&self) -> BoxFuture<'_, OSResult<usize>> {
        let size = self.inner.size;
        Box::pin(async move { Ok(size) })
    }

    fn get_range(&self, range: Range<usize>) -> BoxFuture<'static, OSResult<Bytes>> {
        let inner = self.inner.clone();
        Box::pin(async move {
            let bytes = inner.wait().await.map_err(object_store::Error::from)?;
            let start = range.start;
            let end = range.end;
            if start >= bytes.len() || end > bytes.len() {
                return Err(object_store::Error::Generic {
                    store: "memory",
                    source: format!(
                        "Invalid range {}..{} for object '{}' of size {} bytes",
                        start,
                        end,
                        inner.path,
                        bytes.len()
                    )
                    .into(),
                });
            }
            Ok(bytes.slice(range))
        })
    }

    fn get_all(&self) -> BoxFuture<'_, OSResult<Bytes>> {
        Box::pin(async move { self.inner.wait().await.map_err(object_store::Error::from) })
    }
}

pub(crate) fn stream_local_range(
    file: Arc<File>,
    path: Path,
    io_tracker: Arc<crate::utils::tracking_store::IOTracker>,
    range: Range<usize>,
    chunk_size: usize,
) -> ByteStream {
    stream::try_unfold(
        (file, path, io_tracker, range.start, range.end),
        move |state| async move {
            let (file, path, io_tracker, start, end) = state;
            if start >= end {
                return Ok(None);
            }

            let next = (start + chunk_size).min(end);
            let file_clone = file.clone();
            let path_clone = path.clone();
            let num_bytes = (next - start) as u64;
            let metrics = io_tracker.begin_io("get");
            let result = join_local_io(tokio::task::spawn_blocking(move || {
                let mut buf = bytes::BytesMut::with_capacity(next - start);
                // Safety: buffer capacity matches the exact number of bytes we read below.
                unsafe { buf.set_len(next - start) };
                #[cfg(unix)]
                file_clone.read_exact_at(buf.as_mut(), start as u64)?;
                #[cfg(windows)]
                read_exact_at(file_clone, buf.as_mut(), start as u64)?;
                Ok::<_, std::io::Error>(buf.freeze())
            }))
            .await;
            metrics.record(&result, num_bytes);
            let bytes = result?;

            io_tracker.record_read(
                "get_range_stream",
                path_clone,
                num_bytes,
                Some(start as u64..next as u64),
            );

            Ok(Some((bytes, (file, path, io_tracker, next, end))))
        },
    )
    .boxed()
}

impl DeepSizeOf for SmallReader {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        let mut size = self.inner.path.as_ref().deep_size_of_children(context);

        if let Ok(guard) = self.inner.state.try_lock()
            && let SmallReaderState::Finished(Ok(data)) = &*guard
        {
            size += data.len();
        }

        size
    }
}

/// A [`Reader`] that keeps no file descriptor open between reads.
///
/// Each read reopens the object through the [`ObjectStore`](crate::object_store::ObjectStore)
/// and closes it when the read completes. Use this for readers that are stored
/// inside long-lived cache entries: a reader holding an open file pins the
/// inode even after the file is deleted, so cached readers would otherwise
/// accumulate descriptors for files cleanup has already removed.
///
/// The trade-off is one extra open per read. Reads of a warm cache entry are
/// served without touching the reader at all, so the cost only shows up when
/// data is actually re-read.
#[derive(Debug)]
pub struct LazyReader {
    object_store: Arc<crate::object_store::ObjectStore>,
    path: Path,
    size: OnceCell<usize>,
    block_size: usize,
    io_parallelism: usize,
}

impl LazyReader {
    /// Create a lazy reader over `path` in `object_store`.
    ///
    /// `known_size`, when given, saves the HEAD request that the first
    /// [`Reader::size`] call would otherwise make.
    pub fn new(
        object_store: Arc<crate::object_store::ObjectStore>,
        path: Path,
        known_size: Option<usize>,
    ) -> Self {
        Self {
            block_size: object_store.block_size(),
            io_parallelism: object_store.io_parallelism(),
            object_store,
            path,
            size: OnceCell::new_with(known_size),
        }
    }
}

impl Reader for LazyReader {
    fn path(&self) -> &Path {
        &self.path
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn io_parallelism(&self) -> usize {
        self.io_parallelism
    }

    fn size(&self) -> BoxFuture<'_, OSResult<usize>> {
        Box::pin(async move {
            self.size
                .get_or_try_init(|| async {
                    let meta = do_with_retry(|| Box::pin(self.object_store.inner.head(&self.path)))
                        .await?;
                    Ok(meta.size as usize)
                })
                .await
                .cloned()
        })
    }

    fn get_range(&self, range: Range<usize>) -> BoxFuture<'static, OSResult<Bytes>> {
        let object_store = self.object_store.clone();
        let path = self.path.clone();
        Box::pin(async move {
            let reader = object_store
                .open(&path)
                .await
                .map_err(object_store::Error::from)?;
            reader.get_range(range).await
        })
    }

    fn get_all(&self) -> BoxFuture<'_, OSResult<Bytes>> {
        let object_store = self.object_store.clone();
        let path = self.path.clone();
        Box::pin(async move {
            let reader = object_store
                .open(&path)
                .await
                .map_err(object_store::Error::from)?;
            reader.get_all().await
        })
    }
}

impl DeepSizeOf for LazyReader {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.path.as_ref().deep_size_of_children(context)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::fmt::{Display, Formatter};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use futures::stream::BoxStream;
    use lance_core::utils::testing::{ProxyObjectStore, ProxyObjectStorePolicy};
    use object_store::memory::InMemory;
    use object_store::{
        Attributes, CopyOptions, GetResult, GetResultPayload, ListResult, MultipartUpload,
        ObjectMeta, ObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult,
        Result as OSResult, path::Path,
    };
    use rstest::rstest;

    use super::*;
    use crate::traits::{WriteExt, Writer};
    use crate::utils::tracking_store::{IoStats, IoTrackingStore};

    /// Injects request failures, response metadata, and partial streaming bodies.
    #[derive(Debug)]
    struct PartialBodyStore {
        responses: Option<Mutex<VecDeque<OSResult<GetResult>>>>,
        request_options: Mutex<Vec<GetOptions>>,
        metadata_size: u64,
        returned_range: Option<Range<u64>>,
        body: Vec<Bytes>,
        failed_downloads: usize,
        requests: AtomicUsize,
        body_polls: Arc<AtomicUsize>,
        head_requests: AtomicUsize,
        body_gate: Option<Arc<tokio::sync::Notify>>,
    }

    impl Default for PartialBodyStore {
        fn default() -> Self {
            Self {
                responses: None,
                request_options: Mutex::new(Vec::new()),
                metadata_size: 1024,
                returned_range: None,
                body: vec![Bytes::from(vec![7; 1024])],
                failed_downloads: usize::MAX,
                requests: AtomicUsize::new(0),
                body_polls: Arc::new(AtomicUsize::new(0)),
                head_requests: AtomicUsize::new(0),
                body_gate: None,
            }
        }
    }

    impl Display for PartialBodyStore {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "PartialBodyStore")
        }
    }

    #[async_trait::async_trait]
    impl ObjectStore for PartialBodyStore {
        async fn put_opts(
            &self,
            _location: &Path,
            _payload: PutPayload,
            _opts: PutOptions,
        ) -> OSResult<PutResult> {
            unimplemented!("put_opts is not used by these tests")
        }
        async fn put_multipart_opts(
            &self,
            _location: &Path,
            _opts: PutMultipartOptions,
        ) -> OSResult<Box<dyn MultipartUpload>> {
            unimplemented!("put_multipart_opts is not used by these tests")
        }
        async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
            let request = self.requests.fetch_add(1, Ordering::SeqCst);
            if let Some(responses) = &self.responses {
                self.request_options.lock().unwrap().push(options);
                return responses.lock().unwrap().pop_front().unwrap();
            }
            if options.head {
                self.head_requests.fetch_add(1, Ordering::SeqCst);
            }
            let range = self.returned_range.clone().unwrap_or_else(|| {
                options
                    .range
                    .map(|r| r.as_range(self.metadata_size).unwrap())
                    .unwrap_or(0..self.metadata_size)
            });
            let chunks = if request < self.failed_downloads {
                vec![Err(object_store::Error::Generic {
                    store: "PartialBodyStore",
                    source: "connection reset".into(),
                })]
            } else {
                self.body.iter().cloned().map(Ok).collect()
            };
            let polls = self.body_polls.clone();
            let gate = self.body_gate.clone();
            Ok(GetResult {
                payload: GetResultPayload::Stream(
                    stream::iter(chunks)
                        .then(move |chunk| {
                            let gate = gate.clone();
                            async move {
                                if let Some(gate) = gate {
                                    gate.notified().await;
                                }
                                chunk
                            }
                        })
                        .inspect(move |_| {
                            polls.fetch_add(1, Ordering::SeqCst);
                        })
                        .boxed(),
                ),
                meta: ObjectMeta {
                    location: location.clone(),
                    last_modified: Default::default(),
                    size: self.metadata_size,
                    e_tag: None,
                    version: None,
                },
                range,
                attributes: Attributes::default(),
                extensions: Default::default(),
            })
        }
        fn delete_stream(
            &self,
            _locations: BoxStream<'static, OSResult<Path>>,
        ) -> BoxStream<'static, OSResult<Path>> {
            unimplemented!("delete_stream is not used by these tests")
        }
        fn list(&self, _prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
            unimplemented!("list is not used by these tests")
        }
        async fn list_with_delimiter(&self, _prefix: Option<&Path>) -> OSResult<ListResult> {
            unimplemented!("list_with_delimiter is not used by these tests")
        }
        async fn copy_opts(&self, _from: &Path, _to: &Path, _opts: CopyOptions) -> OSResult<()> {
            unimplemented!("copy_opts is not used by these tests")
        }
    }

    fn reader(store: Arc<dyn ObjectStore>, download_retry_count: usize) -> CloudObjectReader {
        CloudObjectReader::new(
            store,
            Path::from("test/data.lance"),
            4096,
            None,
            download_retry_count,
        )
        .unwrap()
    }

    fn streaming_store(responses: Vec<OSResult<GetResult>>) -> Arc<PartialBodyStore> {
        Arc::new(PartialBodyStore {
            responses: Some(Mutex::new(responses.into())),
            ..Default::default()
        })
    }

    fn stream_response(range: Range<u64>, chunks: Vec<OSResult<Bytes>>) -> OSResult<GetResult> {
        Ok(GetResult {
            payload: GetResultPayload::Stream(stream::iter(chunks).boxed()),
            meta: ObjectMeta {
                location: Path::from("test/data.lance"),
                last_modified: Default::default(),
                size: 16,
                e_tag: Some("original-etag".into()),
                version: Some("original-version".into()),
            },
            range,
            attributes: Attributes::default(),
            extensions: Default::default(),
        })
    }

    fn connection_reset() -> object_store::Error {
        object_store::Error::Generic {
            store: "PartialBodyStore",
            source: "connection reset".into(),
        }
    }

    #[rstest]
    #[case::whole_object(None)]
    #[case::range(Some(3..13))]
    #[tokio::test]
    async fn streaming_copy_resumes_at_the_first_undelivered_byte(
        #[case] range: Option<Range<usize>>,
        #[values(0, 4)] delivered: usize,
        #[values(false, true)] is_truncated: bool,
    ) {
        let data = Bytes::from_static(b"abcdefghijklmnop");
        let requested = range.clone().unwrap_or(0..data.len());
        let offset = requested.start + delivered;
        let mut partial = vec![Ok(data.slice(requested.start..offset))];
        if !is_truncated {
            partial.push(Err(connection_reset()));
        }
        let store = streaming_store(vec![
            stream_response(requested.start as u64..requested.end as u64, partial),
            stream_response(
                offset as u64..requested.end as u64,
                vec![Ok(data.slice(offset..requested.end))],
            ),
        ]);
        let reader = reader(store.clone(), 1);
        let target = crate::object_store::ObjectStore::memory();
        let target_path = Path::from("copied");
        let mut writer = crate::object_writer::ObjectWriter::new(&target, &target_path)
            .await
            .unwrap();
        let copied = match range {
            Some(range) => writer.copy_range_from_reader(&reader, range).await,
            None => writer.copy_from_reader(&reader).await,
        }
        .unwrap();
        writer.shutdown().await.unwrap();
        assert_eq!(copied, requested.len());
        assert_eq!(
            target
                .inner
                .get(&target_path)
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap(),
            data.slice(requested.clone())
        );
        let options = store.request_options.lock().unwrap();
        assert_eq!(options.len(), 2);
        assert_eq!(
            options[1].range,
            Some((offset as u64..requested.end as u64).into())
        );
        assert_eq!(options[1].if_match.as_deref(), Some("original-etag"));
        assert_eq!(options[1].version.as_deref(), Some("original-version"));
    }

    #[rstest]
    #[case::no_retries(0)]
    #[case::one_retry(1)]
    #[case::default(3)]
    #[case::raised(5)]
    #[tokio::test]
    async fn streaming_request_retries_follow_configuration(
        #[case] retry_count: usize,
        #[values(false, true)] is_range: bool,
    ) {
        let store = streaming_store((0..=retry_count).map(|_| Err(connection_reset())).collect());
        let reader = reader(store.clone(), retry_count);
        let result = if is_range {
            reader.get_range_stream(3..13).await
        } else {
            reader.get_stream().await
        };
        let error = result.err().unwrap();
        assert!(matches!(error, object_store::Error::Generic { .. }));
        assert!(error.to_string().contains("connection reset"), "{error}");
        assert_eq!(store.requests.load(Ordering::SeqCst), retry_count + 1);
    }

    #[rstest]
    #[case::no_retries(0)]
    #[case::one_retry(1)]
    #[case::default(3)]
    #[case::raised(5)]
    #[tokio::test]
    async fn streaming_body_retries_are_bounded_across_delivered_chunks(
        #[case] retry_count: usize,
        #[values(false, true)] is_range: bool,
    ) {
        let start = if is_range { 3 } else { 0 };
        let end = if is_range { 13 } else { 16 };
        let store = streaming_store(
            (0..=retry_count)
                .map(|attempt| {
                    stream_response(
                        start + attempt as u64..end,
                        vec![Ok(Bytes::from_static(b"a")), Err(connection_reset())],
                    )
                })
                .collect(),
        );
        let reader = reader(store.clone(), retry_count);
        let mut stream = if is_range {
            reader.get_range_stream(start as usize..end as usize).await
        } else {
            reader.get_stream().await
        }
        .unwrap();
        for _ in 0..=retry_count {
            assert_eq!(stream.next().await.unwrap().unwrap(), "a");
        }
        let error = stream.next().await.unwrap().unwrap_err();
        assert!(matches!(error, object_store::Error::Generic { .. }));
        assert!(error.to_string().contains("connection reset"), "{error}");
        assert!(stream.next().await.is_none());
        assert_eq!(store.requests.load(Ordering::SeqCst), retry_count + 1);
    }

    #[rstest]
    #[case::exhausted(2, false)]
    #[case::recovered(3, true)]
    #[tokio::test]
    async fn streaming_request_and_body_failures_share_one_retry_budget(
        #[case] retry_count: usize,
        #[case] is_recovered: bool,
    ) {
        let store = streaming_store(vec![
            Err(connection_reset()),
            stream_response(
                0..16,
                vec![Ok(Bytes::from_static(b"abcd")), Err(connection_reset())],
            ),
            Err(connection_reset()),
            stream_response(4..16, vec![Ok(Bytes::from_static(b"efghijklmnop"))]),
        ]);
        let reader = reader(store.clone(), retry_count);
        let mut stream = reader.get_stream().await.unwrap();
        assert_eq!(stream.next().await.unwrap().unwrap(), "abcd");
        let next = stream.next().await.unwrap();
        if is_recovered {
            assert_eq!(next.unwrap(), "efghijklmnop");
        } else {
            let error = next.unwrap_err();
            assert!(matches!(error, object_store::Error::Generic { .. }));
            assert!(error.to_string().contains("connection reset"), "{error}");
        }
        assert!(stream.next().await.is_none());
        assert_eq!(store.requests.load(Ordering::SeqCst), retry_count + 1);
        let options = store.request_options.lock().unwrap();
        for options in &options[2..] {
            assert_eq!(options.range, Some((4..16).into()));
            assert_eq!(options.if_match.as_deref(), Some("original-etag"));
        }
    }

    #[rstest]
    #[case::etag_only(true, false)]
    #[case::version_only(false, true)]
    #[case::neither(false, false)]
    #[tokio::test]
    async fn streaming_resume_requires_an_object_identifier(
        #[case] has_etag: bool,
        #[case] has_version: bool,
    ) {
        let mut initial = stream_response(0..16, vec![Err(connection_reset())]).unwrap();
        let mut resumed =
            stream_response(0..16, vec![Ok(Bytes::from_static(b"abcdefghijklmnop"))]).unwrap();
        for result in [&mut initial, &mut resumed] {
            if !has_etag {
                result.meta.e_tag = None;
            }
            if !has_version {
                result.meta.version = None;
            }
        }
        let store = streaming_store(vec![Ok(initial), Ok(resumed)]);
        let reader = reader(store.clone(), 1);
        let mut stream = reader.get_stream().await.unwrap();
        let result = stream.next().await.unwrap();
        if has_etag || has_version {
            assert_eq!(result.unwrap(), "abcdefghijklmnop");
            let options = store.request_options.lock().unwrap();
            assert_eq!(options.len(), 2);
            assert_eq!(options[1].if_match.is_some(), has_etag);
            assert_eq!(options[1].version.is_some(), has_version);
        } else {
            let error = result.unwrap_err();
            assert!(matches!(error, object_store::Error::Generic { .. }));
            assert!(
                error.to_string().contains("without an ETag or version"),
                "{error}"
            );
            assert_eq!(store.requests.load(Ordering::SeqCst), 1);
        }
        assert!(stream.next().await.is_none());
    }

    #[rstest]
    #[case::etag("etag")]
    #[case::version("version")]
    #[case::size("size")]
    #[case::precondition("precondition")]
    #[tokio::test]
    async fn streaming_resume_rejects_object_replacement(#[case] changed: &str) {
        let mut resumed =
            stream_response(4..16, vec![Ok(Bytes::from_static(b"NEW CONTENT!"))]).unwrap();
        match changed {
            "etag" => resumed.meta.e_tag = Some("replacement-etag".into()),
            "version" => resumed.meta.version = Some("replacement-version".into()),
            "size" => resumed.meta.size += 1,
            "precondition" => (),
            _ => unreachable!(),
        }
        let resumed = if changed == "precondition" {
            Err(object_store::Error::Precondition {
                path: "test/data.lance".into(),
                source: "object changed".into(),
            })
        } else {
            // Deliberately ignore the request's conditions to exercise response validation.
            Ok(resumed)
        };
        let store = streaming_store(vec![
            stream_response(
                0..16,
                vec![Ok(Bytes::from_static(b"abcd")), Err(connection_reset())],
            ),
            resumed,
        ]);
        let reader = reader(store.clone(), 3);
        let mut stream = reader.get_stream().await.unwrap();
        assert_eq!(stream.next().await.unwrap().unwrap(), "abcd");
        let error = stream.next().await.unwrap().unwrap_err();
        assert!(matches!(error, object_store::Error::Precondition { .. }));
        assert!(error.to_string().contains("object changed"), "{error}");
        assert!(stream.next().await.is_none());
        assert_eq!(store.requests.load(Ordering::SeqCst), 2);
    }

    #[rstest]
    #[case::initial_range(false, false, "expected 0..16")]
    #[case::resumed_range(true, false, "expected remaining range 4..16")]
    #[case::overlong_body(false, true, "exceeds remaining range")]
    #[tokio::test]
    async fn streaming_rejects_inconsistent_responses(
        #[case] is_retry: bool,
        #[case] is_overlong: bool,
        #[case] message: &str,
    ) {
        let bad_response = if is_overlong {
            stream_response(
                0..16,
                vec![Ok(Bytes::from_static(b"too many bytes for this object"))],
            )
        } else {
            stream_response(0..15, vec![])
        };
        let responses = if is_retry {
            vec![
                stream_response(
                    0..16,
                    vec![Ok(Bytes::from_static(b"abcd")), Err(connection_reset())],
                ),
                bad_response,
            ]
        } else {
            vec![bad_response]
        };
        let store = streaming_store(responses);
        let reader = reader(store.clone(), 3);
        let error = match reader.get_stream().await {
            Err(error) => error,
            Ok(mut stream) => {
                if is_retry {
                    assert_eq!(stream.next().await.unwrap().unwrap(), "abcd");
                }
                let error = stream.next().await.unwrap().unwrap_err();
                assert!(stream.next().await.is_none());
                error
            }
        };
        assert!(matches!(error, object_store::Error::Generic { .. }));
        assert!(error.to_string().contains(message), "{error}");
        assert_eq!(
            store.requests.load(Ordering::SeqCst),
            if is_retry { 2 } else { 1 }
        );
    }

    #[tokio::test]
    async fn streaming_does_not_resume_until_polled() {
        let store = streaming_store(vec![stream_response(
            0..16,
            vec![Ok(Bytes::from_static(b"abcd")), Err(connection_reset())],
        )]);
        let reader = reader(store.clone(), 3);
        let mut stream = reader.get_stream().await.unwrap();
        assert_eq!(store.requests.load(Ordering::SeqCst), 1);
        assert_eq!(stream.next().await.unwrap().unwrap(), "abcd");
        drop(stream);
        assert_eq!(store.requests.load(Ordering::SeqCst), 1);
    }

    #[rstest]
    #[case::empty(b"")]
    #[case::nonempty(b"abcdefghijklmnop")]
    #[tokio::test]
    async fn streaming_reads_without_retries(#[case] data: &'static [u8]) {
        let store = Arc::new(InMemory::new());
        let path = Path::from("test/data.lance");
        store
            .put(&path, Bytes::from_static(data).into())
            .await
            .unwrap();
        let reader = reader(store, 0);
        let mut stream = reader.get_stream().await.unwrap();
        let mut actual = BytesMut::new();
        while let Some(chunk) = stream.next().await {
            actual.extend_from_slice(&chunk.unwrap());
        }
        assert_eq!(actual.as_ref(), data);
    }

    #[tokio::test]
    async fn streaming_does_not_retry_after_delivering_the_complete_body() {
        let store = streaming_store(vec![stream_response(
            0..16,
            vec![
                Ok(Bytes::from_static(b"abcdefghijklmnop")),
                Err(connection_reset()),
            ],
        )]);
        let reader = reader(store.clone(), 0);
        let mut stream = reader.get_stream().await.unwrap();
        assert_eq!(stream.next().await.unwrap().unwrap(), "abcdefghijklmnop");
        assert!(stream.next().await.is_none());
        assert_eq!(store.requests.load(Ordering::SeqCst), 1);
    }

    /// A body that stops partway is what `download_retry_count` documents, and
    /// `get_range` has to honor it like `get_all` already did.
    #[rstest]
    #[case::no_retries(0, 1)]
    #[case::one_retry(1, 2)]
    #[case::default(3, 4)]
    #[case::raised(10, 11)]
    #[tokio::test]
    async fn get_range_retries_a_partial_download_as_configured(
        #[case] download_retry_count: usize,
        #[case] expected_requests: u64,
    ) {
        let stats = Arc::new(Mutex::new(IoStats::default()));
        let store = IoTrackingStore::new(Arc::new(PartialBodyStore::default()), stats.clone());
        let reader = reader(Arc::new(store), download_retry_count);

        reader.get_range(0..128).await.unwrap_err();

        assert_eq!(stats.lock().unwrap().read_iops, expected_requests);
    }

    /// `do_with_retry` gives the request its own fixed retries, so the
    /// configured download count does not stretch or shrink them.
    #[tokio::test]
    async fn get_range_leaves_the_request_retries_alone() {
        // Counted by the policy: `IoTrackingStore` records only requests that
        // succeed, and none of these do.
        let requests = Arc::new(AtomicUsize::new(0));
        let counted = requests.clone();
        let mut policy = ProxyObjectStorePolicy::new();
        policy.set_before_policy(
            "fail_every_request",
            Arc::new(move |_, _| {
                counted.fetch_add(1, Ordering::Relaxed);
                Err(Error::io("connection reset"))
            }),
        );
        let store = ProxyObjectStore::new(Arc::new(InMemory::new()), Arc::new(Mutex::new(policy)));
        let reader = reader(Arc::new(store), 10);

        reader.get_range(0..128).await.unwrap_err();

        assert_eq!(requests.load(Ordering::Relaxed), 4);
    }
    #[rstest]
    #[case::same_size(1024, None)]
    #[case::stale_hint(1023, None)]
    #[case::partial_metadata_range(1024, Some(0..512))]
    #[tokio::test]
    async fn known_size_open_preserves_body_collection_and_shared_bytes(
        #[case] known_size: usize,
        #[case] returned_range: Option<Range<u64>>,
    ) {
        let body = Bytes::from(vec![7; 1024]);
        let inner = Arc::new(PartialBodyStore {
            body: vec![body.clone()],
            returned_range,
            failed_downloads: 0,
            ..Default::default()
        });
        let mut store = crate::object_store::ObjectStore::memory();
        store.inner = inner.clone();
        let reader = store
            .open_with_size(&Path::from("part"), known_size)
            .await
            .unwrap();
        assert_eq!(inner.requests.load(Ordering::SeqCst), 0);
        let (first, second) = tokio::join!(reader.get_all(), reader.get_range(8..16));
        let first = first.unwrap();
        let second = second.unwrap();
        assert_eq!(first, body);
        assert_eq!(first.as_ptr(), body.as_ptr());
        assert_eq!(second.as_ptr(), body.slice(8..16).as_ptr());
        assert_eq!(inner.requests.load(Ordering::SeqCst), 1);
        assert_eq!(inner.head_requests.load(Ordering::SeqCst), 0);
        assert_eq!(inner.body_polls.load(Ordering::SeqCst), 1);
    }

    #[rstest]
    #[case::short(1023, None)]
    #[case::long(1025, None)]
    #[case::partial_range(1024, Some(0..512))]
    #[tokio::test]
    async fn small_reader_rejects_metadata_before_body(
        #[case] metadata_size: u64,
        #[case] returned_range: Option<Range<u64>>,
    ) {
        let store = Arc::new(PartialBodyStore {
            metadata_size,
            returned_range,
            failed_downloads: 0,
            ..Default::default()
        });
        let error =
            SmallReader::open_with_expected_size(store.clone(), Path::from("part"), 3, 1024)
                .await
                .unwrap_err();
        assert!(
            matches!(error.find_root(), Error::InvalidInput { .. }),
            "{error}"
        );
        assert!(
            error.to_string().contains("expected complete file"),
            "{error}"
        );
        assert_eq!(store.requests.load(Ordering::SeqCst), 1);
        assert_eq!(store.body_polls.load(Ordering::SeqCst), 0);
    }

    #[rstest]
    #[case::truncated(1023, "has 1023 bytes")]
    #[case::overlong(1025, "exceeds expected")]
    #[tokio::test]
    async fn small_reader_rejects_body_length_with_existing_retry_budget(
        #[case] size: usize,
        #[case] message: &str,
    ) {
        let store = Arc::new(PartialBodyStore {
            body: vec![Bytes::from(vec![0; size])],
            failed_downloads: 0,
            ..Default::default()
        });
        let error =
            SmallReader::open_with_expected_size(store.clone(), Path::from("part"), 1, 1024)
                .await
                .unwrap_err();
        assert!(matches!(error.find_root(), Error::IO { .. }), "{error}");
        assert!(error.to_string().contains(message), "{error}");
        assert_eq!(store.requests.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn small_reader_retries_body_then_shares_the_validated_bytes() {
        let store = Arc::new(PartialBodyStore {
            failed_downloads: 1,
            ..Default::default()
        });
        let reader =
            SmallReader::open_with_expected_size(store.clone(), Path::from("part"), 1, 1024)
                .await
                .unwrap();
        let (first, second) = tokio::join!(reader.get_all(), reader.get_range(8..16));
        assert_eq!(first.unwrap(), Bytes::from(vec![7; 1024]));
        assert_eq!(second.unwrap(), Bytes::from(vec![7; 8]));
        assert_eq!(store.requests.load(Ordering::SeqCst), 2);
    }

    #[rstest]
    #[case::truncated(3)]
    #[case::grown(5)]
    #[tokio::test]
    async fn small_reader_checks_replaced_object_on_first_read(#[case] size: usize) {
        let store = Arc::new(InMemory::new());
        let path = Path::from("part");
        store
            .put(&path, Bytes::from_static(b"old!").into())
            .await
            .unwrap();
        let opening = SmallReader::open_with_expected_size(store.clone(), path.clone(), 3, 4);
        store
            .put(&path, Bytes::from(vec![0; size]).into())
            .await
            .unwrap();
        let error = opening.await.unwrap_err();
        assert!(
            matches!(error.find_root(), Error::InvalidInput { .. }),
            "{error}"
        );
        assert!(
            error.to_string().contains("expected complete file"),
            "{error}"
        );
    }

    #[rstest]
    #[case::small(0, 0)]
    #[case::large(1, 1)]
    #[tokio::test]
    async fn checked_open_keeps_the_existing_small_file_threshold(
        #[case] extra: usize,
        #[case] expected_heads: usize,
    ) {
        let mut store = crate::object_store::ObjectStore::memory();
        let size = store.block_size() + extra;
        let inner = Arc::new(PartialBodyStore {
            metadata_size: size as u64,
            body: vec![Bytes::from(vec![9; size])],
            failed_downloads: 0,
            ..Default::default()
        });
        store.inner = inner.clone();
        let reader = store
            .open_with_expected_size(&Path::from("part"), size as u64)
            .await
            .unwrap();
        assert_eq!(reader.get_all().await.unwrap(), Bytes::from(vec![9; size]));
        assert_eq!(inner.head_requests.load(Ordering::SeqCst), expected_heads);
        assert_eq!(inner.requests.load(Ordering::SeqCst), 1 + expected_heads);
    }

    #[tokio::test]
    async fn small_reader_cancellation_drops_the_pending_body() {
        let gate = Arc::new(tokio::sync::Notify::new());
        let store = Arc::new(PartialBodyStore {
            failed_downloads: 0,
            body_gate: Some(gate.clone()),
            ..Default::default()
        });
        let mut opening = Box::pin(SmallReader::open_with_expected_size(
            store.clone(),
            Path::from("part"),
            3,
            1024,
        ));
        assert!(opening.as_mut().now_or_never().is_none());
        assert_eq!(store.requests.load(Ordering::SeqCst), 1);
        drop(opening);
        assert_eq!(Arc::strong_count(&gate), 2); // Test and store only; no pending stream.
        assert_eq!(store.body_polls.load(Ordering::SeqCst), 0);
    }
}
