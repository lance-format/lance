// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    ops::Range,
    sync::{Arc, Mutex},
    time::Duration,
};

use crate::storage::{
    CopyOptions, GetOptions, GetResult, ListResult, ObjectMeta, PutMultipartOptions, PutOptions,
    PutPayload, PutResult, RenameOptions, Result, UploadPart, path::Path,
};
use async_trait::async_trait;
use bytes::Bytes;
use futures::{StreamExt, stream::BoxStream};

use crate::storage::backend::{MultipartUpload, StorageBackend};

/// Per-operation delays for exercising concurrent IO paths with [`DelayedBackend`].
#[derive(Debug, Clone, Copy, Default)]
pub struct DelayConfig {
    /// Delay before each metadata, object, or multi-range read.
    pub wait_get_per_call: Duration,
    /// Delay before each put, uploaded part, copy, or rename.
    pub wait_put_per_call: Duration,
    /// Delay before each valid path in a deletion stream.
    pub wait_delete_per_call: Duration,
    /// Delay before starting a streaming list, including lists with an offset.
    pub wait_list_per_call: Duration,
    /// Delay before each successful entry in a streaming list.
    pub wait_list_per_entry: Duration,
}

/// Injects latency while preserving backend payloads, conditions and bulk operations.
/// Delimiter listings are passed through without delay.
///
/// ```
/// use std::{sync::Arc, time::Duration};
/// use lance_io::storage::Storage;
/// use lance_io::utils::testing::{DelayConfig, DelayedBackend};
///
/// let backend = DelayedBackend::new(lance_io::storage::backend::object_store::from_object_store(Arc::new(object_store::memory::InMemory::new())), DelayConfig {
///     wait_get_per_call: Duration::from_millis(10),
///     ..Default::default()
/// });
/// backend.config_mut(|config| config.wait_get_per_call = Duration::ZERO);
/// ```
#[derive(Debug)]
pub struct DelayedBackend {
    inner: Arc<dyn StorageBackend>,
    config: Mutex<DelayConfig>,
}

impl DelayedBackend {
    /// Wrap a backend with the specified operation delays.
    pub fn new(inner: Arc<dyn StorageBackend>, config: DelayConfig) -> Self {
        Self {
            inner,
            config: Mutex::new(config),
        }
    }

    /// Update delays for subsequent operations; existing uploads and listing
    /// streams retain the delays they captured when created.
    pub fn config_mut(&self, update: impl FnOnce(&mut DelayConfig)) {
        update(&mut self.config.lock().unwrap());
    }

    fn config(&self) -> DelayConfig {
        *self.config.lock().unwrap()
    }
}

impl std::fmt::Display for DelayedBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DelayedBackend({})", self.inner)
    }
}

#[async_trait]
impl StorageBackend for DelayedBackend {
    async fn list_paginated(
        &self,
        prefix: Option<&str>,
        options: crate::storage::PaginatedListOptions,
    ) -> crate::storage::Result<crate::storage::PaginatedListResult> {
        let delay = self.config.lock().unwrap().wait_list_per_call;
        tokio::time::sleep(delay).await;
        self.inner.list_paginated(prefix, options).await
    }

    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> Result<PutResult> {
        tokio::time::sleep(self.config().wait_put_per_call).await;
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        Ok(Box::new(DelayedUpload {
            inner: self.inner.put_multipart_opts(location, opts).await?,
            delay: self.config().wait_put_per_call,
        }))
    }

    async fn get_opts(&self, location: &Path, opts: GetOptions) -> Result<GetResult> {
        tokio::time::sleep(self.config().wait_get_per_call).await;
        self.inner.get_opts(location, opts).await
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        tokio::time::sleep(self.config().wait_get_per_call).await;
        self.inner.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>> {
        let delay = self.config().wait_delete_per_call;
        self.inner.delete_stream(
            locations
                .then(move |location| async move {
                    if location.is_ok() {
                        tokio::time::sleep(delay).await;
                    }
                    location
                })
                .boxed(),
        )
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        let config = self.config();
        let stream = self.inner.list(prefix);
        futures::stream::once(async move {
            tokio::time::sleep(config.wait_list_per_call).await;
            stream.then(move |item| async move {
                if item.is_ok() {
                    tokio::time::sleep(config.wait_list_per_entry).await;
                }
                item
            })
        })
        .flatten()
        .boxed()
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, Result<ObjectMeta>> {
        let config = self.config();
        let stream = self.inner.list_with_offset(prefix, offset);
        futures::stream::once(async move {
            tokio::time::sleep(config.wait_list_per_call).await;
            stream.then(move |item| async move {
                if item.is_ok() {
                    tokio::time::sleep(config.wait_list_per_entry).await;
                }
                item
            })
        })
        .flatten()
        .boxed()
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> Result<()> {
        tokio::time::sleep(self.config().wait_put_per_call).await;
        self.inner.copy_opts(from, to, opts).await
    }

    async fn rename_opts(&self, from: &Path, to: &Path, opts: RenameOptions) -> Result<()> {
        tokio::time::sleep(self.config().wait_put_per_call).await;
        self.inner.rename_opts(from, to, opts).await
    }
}

#[derive(Debug)]
struct DelayedUpload {
    inner: Box<dyn MultipartUpload>,
    delay: Duration,
}

#[async_trait]
impl MultipartUpload for DelayedUpload {
    fn put_part(&mut self, payload: PutPayload) -> UploadPart {
        let upload = self.inner.put_part(payload);
        let delay = self.delay;
        Box::pin(async move {
            tokio::time::sleep(delay).await;
            upload.await
        })
    }

    async fn complete(&mut self) -> Result<PutResult> {
        self.inner.complete().await
    }

    async fn abort(&mut self) -> Result<()> {
        self.inner.abort().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::GetResultPayload;
    use crate::storage::backend::object_store::from_object_store;
    use ::object_store::local::LocalFileSystem;
    use futures::TryStreamExt;
    use tokio::time::Instant;

    #[tokio::test(start_paused = true)]
    async fn local_payload_and_updated_delay() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("file"), b"contents").unwrap();
        let backend = DelayedBackend::new(
            from_object_store(Arc::new(
                LocalFileSystem::new_with_prefix(dir.path()).unwrap(),
            )),
            DelayConfig {
                wait_get_per_call: Duration::from_secs(1),
                ..Default::default()
            },
        );
        let start = Instant::now();
        let result = backend
            .get_opts(&"file".into(), GetOptions::default())
            .await
            .unwrap();
        assert!(start.elapsed() >= Duration::from_secs(1));
        assert!(matches!(&result.payload, GetResultPayload::File(..)));
        assert_eq!(result.bytes().await.unwrap(), "contents");

        backend.config_mut(|config| config.wait_get_per_call = Duration::ZERO);
        let start = Instant::now();
        backend
            .get_opts(&"file".into(), GetOptions::default())
            .await
            .unwrap();
        assert!(start.elapsed() < Duration::from_secs(1));
    }

    #[tokio::test(start_paused = true)]
    async fn multipart_lifecycle_and_list_delays() {
        let backend = DelayedBackend::new(
            crate::storage::backend::object_store::from_object_store(Arc::new(
                object_store::memory::InMemory::new(),
            )),
            DelayConfig {
                wait_put_per_call: Duration::from_secs(1),
                wait_list_per_call: Duration::from_secs(2),
                wait_list_per_entry: Duration::from_secs(3),
                ..Default::default()
            },
        );
        let backend = &backend as &dyn StorageBackend;
        let mut upload = backend.put_multipart(&"a".into()).await.unwrap();
        let start = Instant::now();
        upload
            .put_part(Bytes::from_static(b"payload").into())
            .await
            .unwrap();
        assert!(start.elapsed() >= Duration::from_secs(1));
        upload.complete().await.unwrap();
        assert_eq!(
            backend
                .get(&"a".into())
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap(),
            "payload"
        );

        let mut aborted = backend.put_multipart(&"b".into()).await.unwrap();
        aborted
            .put_part(Bytes::from_static(b"discard").into())
            .await
            .unwrap();
        aborted.abort().await.unwrap();
        assert!(matches!(
            backend.head(&"b".into()).await,
            Err(crate::storage::Error::NotFound { .. })
        ));

        let start = Instant::now();
        let objects = backend
            .list_with_offset(None, &Path::default())
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].location, Path::from("a"));
        assert!(start.elapsed() >= Duration::from_secs(5));
        assert!(start.elapsed() < Duration::from_secs(6));
    }
}
