// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Range;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use async_trait::async_trait;
use bytes::Bytes;
use chrono::Utc;
use futures::StreamExt;
use lance_io::object_store::ObjectStore;
use object_store::{
    Attributes, CopyOptions, GetOptions, GetRange, GetResult, GetResultPayload, ListResult,
    MultipartUpload, ObjectMeta, PutMultipartOptions, PutOptions, PutPayload, PutResult,
    path::Path,
};
use tokio::sync::{Notify, Semaphore};
use url::Url;

#[derive(Debug)]
pub struct RecordingRangeObjectStore {
    pub data: Bytes,
    pub gate: Option<Arc<Semaphore>>,
    pub head_requests: AtomicUsize,
    pub requested_ranges: std::sync::Mutex<Vec<Range<u64>>>,
    pub started_blob_requests: AtomicUsize,
    pub active_blob_requests: AtomicUsize,
    pub peak_active_blob_requests: AtomicUsize,
    pub request_started: Notify,
}

impl RecordingRangeObjectStore {
    pub fn new(data: Bytes) -> Self {
        Self {
            data,
            gate: None,
            head_requests: AtomicUsize::new(0),
            requested_ranges: std::sync::Mutex::new(Vec::new()),
            started_blob_requests: AtomicUsize::new(0),
            active_blob_requests: AtomicUsize::new(0),
            peak_active_blob_requests: AtomicUsize::new(0),
            request_started: Notify::new(),
        }
    }

    pub fn with_gate(data: Bytes, gate: Arc<Semaphore>) -> Self {
        Self {
            data,
            gate: Some(gate),
            head_requests: AtomicUsize::new(0),
            requested_ranges: std::sync::Mutex::new(Vec::new()),
            started_blob_requests: AtomicUsize::new(0),
            active_blob_requests: AtomicUsize::new(0),
            peak_active_blob_requests: AtomicUsize::new(0),
            request_started: Notify::new(),
        }
    }

    pub async fn wait_for_blob_requests(&self, expected: usize) {
        loop {
            let request_started = self.request_started.notified();
            if self.started_blob_requests.load(Ordering::Acquire) >= expected {
                return;
            }
            request_started.await;
        }
    }

    pub fn peak_active_blob_requests(&self) -> usize {
        self.peak_active_blob_requests.load(Ordering::Acquire)
    }

    pub fn requested_ranges(&self) -> Vec<Range<u64>> {
        self.requested_ranges.lock().unwrap().clone()
    }

    pub fn requested_blob_ranges(&self) -> Vec<Range<u64>> {
        let full_object = 0..self.data.len() as u64;
        self.requested_ranges()
            .into_iter()
            .filter(|range| range != &full_object)
            .collect()
    }

    pub fn object_meta(&self, location: &Path) -> ObjectMeta {
        ObjectMeta {
            location: location.clone(),
            last_modified: Utc::now(),
            size: self.data.len() as u64,
            e_tag: None,
            version: None,
        }
    }
}

impl std::fmt::Display for RecordingRangeObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RecordingRangeObjectStore")
    }
}

#[async_trait]
impl object_store::ObjectStore for RecordingRangeObjectStore {
    async fn put_opts(
        &self,
        _location: &Path,
        _bytes: PutPayload,
        _opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        unimplemented!("put_opts is not used by these tests")
    }

    async fn put_multipart_opts(
        &self,
        _location: &Path,
        _opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        unimplemented!("put_multipart_opts is not used by these tests")
    }

    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        if options.head {
            self.head_requests.fetch_add(1, Ordering::Relaxed);
        }
        let range = match options.range {
            Some(GetRange::Bounded(range)) => range,
            None => 0..self.data.len() as u64,
            Some(other) => {
                return Err(object_store::Error::NotSupported {
                    source: format!("unsupported range request {other:?}").into(),
                });
            }
        };
        let is_full_object_probe = range.start == 0 && range.end == self.data.len() as u64;
        self.requested_ranges.lock().unwrap().push(range.clone());
        if !is_full_object_probe {
            let active = self.active_blob_requests.fetch_add(1, Ordering::AcqRel) + 1;
            self.peak_active_blob_requests
                .fetch_max(active, Ordering::AcqRel);
            self.started_blob_requests.fetch_add(1, Ordering::AcqRel);
            self.request_started.notify_waiters();
            if let Some(gate) = &self.gate {
                gate.acquire()
                    .await
                    .expect("test gate should remain open")
                    .forget();
            }
            self.active_blob_requests.fetch_sub(1, Ordering::AcqRel);
        }
        let bytes = self.data.slice(range.start as usize..range.end as usize);
        Ok(GetResult {
            payload: GetResultPayload::Stream(
                futures::stream::once(async move { Ok(bytes) }).boxed(),
            ),
            meta: self.object_meta(location),
            range,
            attributes: Attributes::default(),
            extensions: Default::default(),
        })
    }

    fn delete_stream(
        &self,
        _locations: futures::stream::BoxStream<'static, object_store::Result<Path>>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<Path>> {
        unimplemented!("delete_stream is not used by these tests")
    }

    fn list(
        &self,
        _prefix: Option<&Path>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<ObjectMeta>> {
        unimplemented!("list is not used by these tests")
    }

    async fn list_with_delimiter(
        &self,
        _prefix: Option<&Path>,
    ) -> object_store::Result<ListResult> {
        unimplemented!("list_with_delimiter is not used by these tests")
    }

    async fn copy_opts(
        &self,
        _from: &Path,
        _to: &Path,
        _options: CopyOptions,
    ) -> object_store::Result<()> {
        unimplemented!("copy_opts is not used by these tests")
    }
}

pub fn recording_range_store_with_url_and_block_size(
    data: Bytes,
    url: &str,
    block_size: Option<usize>,
) -> (Arc<ObjectStore>, Arc<RecordingRangeObjectStore>) {
    const TEST_RANGE_STORE_SIZE: usize = 128 * 1024;
    let mut padded = vec![0; TEST_RANGE_STORE_SIZE.max(data.len())];
    padded[..data.len()].copy_from_slice(data.as_ref());
    let inner = Arc::new(RecordingRangeObjectStore::new(Bytes::from(padded)));
    let store = Arc::new(ObjectStore::new(
        inner.clone() as Arc<dyn object_store::ObjectStore>,
        Url::parse(url).unwrap(),
        block_size,
        None,
        false,
        true,
        lance_io::object_store::DEFAULT_LOCAL_IO_PARALLELISM,
        lance_io::object_store::DEFAULT_DOWNLOAD_RETRY_COUNT,
        None,
    ));
    (store, inner)
}

pub fn recording_range_store_with_url(
    data: Bytes,
    url: &str,
) -> (Arc<ObjectStore>, Arc<RecordingRangeObjectStore>) {
    recording_range_store_with_url_and_block_size(data, url, None)
}

pub fn recording_range_store(data: Bytes) -> (Arc<ObjectStore>, Arc<RecordingRangeObjectStore>) {
    recording_range_store_with_url(data, "mock://recording/blob-range-tests")
}

pub fn gated_range_store(
    data: Bytes,
    url: &str,
) -> (
    Arc<ObjectStore>,
    Arc<RecordingRangeObjectStore>,
    Arc<Semaphore>,
) {
    const TEST_RANGE_STORE_SIZE: usize = 128 * 1024;
    let mut padded = vec![0; TEST_RANGE_STORE_SIZE.max(data.len())];
    padded[..data.len()].copy_from_slice(data.as_ref());
    let gate = Arc::new(Semaphore::new(0));
    let inner = Arc::new(RecordingRangeObjectStore::with_gate(
        Bytes::from(padded),
        gate.clone(),
    ));
    let store = Arc::new(ObjectStore::new(
        inner.clone() as Arc<dyn object_store::ObjectStore>,
        Url::parse(url).unwrap(),
        None,
        None,
        false,
        true,
        lance_io::object_store::DEFAULT_LOCAL_IO_PARALLELISM,
        lance_io::object_store::DEFAULT_DOWNLOAD_RETRY_COUNT,
        None,
    ));
    (store, inner, gate)
}
