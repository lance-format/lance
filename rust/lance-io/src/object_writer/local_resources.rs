// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Local destination buffers whose owners follow actual blocking writes.

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::{
    fs::File,
    future::Future,
    io::{self},
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use async_trait::async_trait;
use futures::future::BoxFuture;
use lance_core::{
    Error, Result,
    utils::{memory::MemoryReserver, tokio::io_task_with_resources},
};
use object_store::{PutMode, path::Path};
use tokio::io::{AsyncWrite, AsyncWriteExt};

use super::{OutputLifecycle, WriteResult, get_etag};
use crate::{
    traits::Writer,
    utils::tracking_store::{IOTracker, IoMetricsGuard},
};

// One bounded copy per write; no multipart buffers on local filesystems.
const WRITE_CHUNK_BYTES: usize = 64 * 1024;
const WRITE_CONTROL_BYTES: usize = 2048;

struct StagedFile {
    // Close the file before dropping its temporary path, including on Windows.
    file: File,
    temp_path: tempfile::TempPath,
    path: Path,
    tracker: Arc<IOTracker>,
    metrics: IoMetricsGuard,
    _control: Box<dyn Send + Sync>,
    lifecycle: Option<Arc<dyn OutputLifecycle>>,
    mode: PutMode,
    fsync: bool,
}

struct FinishedOutput {
    result: Result<WriteResult>,
    control: Box<dyn Send + Sync>,
}

#[doc(hidden)]
pub struct ResourceLocalWriter {
    file: Option<Arc<StagedFile>>,
    pending: Option<BoxFuture<'static, io::Result<()>>>,
    finishing: Option<BoxFuture<'static, Result<FinishedOutput>>>,
    published_control: Option<Box<dyn Send + Sync>>,
    result: Option<WriteResult>,
    pool: Arc<dyn MemoryReserver>,
    position: usize,
    failed: bool,
}

impl ResourceLocalWriter {
    #[doc(hidden)]
    pub async fn create(
        path: &Path,
        pool: Arc<dyn MemoryReserver>,
        tracker: Arc<IOTracker>,
        fsync: bool,
    ) -> Result<Self> {
        Self::create_with_mode(path, pool, tracker, None, PutMode::Overwrite, fsync).await
    }

    pub(crate) async fn create_with_mode(
        path: &Path,
        pool: Arc<dyn MemoryReserver>,
        tracker: Arc<IOTracker>,
        lifecycle: Option<Arc<dyn OutputLifecycle>>,
        mode: PutMode,
        fsync: bool,
    ) -> Result<Self> {
        if matches!(mode, PutMode::Update(_)) {
            return Err(Error::not_supported(
                "resource-aware local output does not support PutMode::Update",
            ));
        }
        let bytes = path
            .as_ref()
            .len()
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_add(8192))
            .ok_or_else(|| Error::invalid_input("local output state size overflow"))?;
        let control = pool.reserve("local output state", bytes)?;
        let path = path.clone();
        let file = io_task_with_resources(
            pool.clone(),
            "local output creation",
            move || -> Result<Arc<StagedFile>> {
                let local = std::path::PathBuf::from(crate::local::to_local_path(&path));
                let parent = local.parent().ok_or_else(|| {
                    Error::invalid_input(format!("local output path {path} has no parent"))
                })?;
                if fsync {
                    crate::local::create_dir_all_durable(parent)?;
                } else {
                    std::fs::create_dir_all(parent)?;
                }
                let mut builder = tempfile::Builder::new();
                #[cfg(unix)]
                builder.permissions(std::fs::Permissions::from_mode(0o666));
                let (file, temp_path) = builder.tempfile_in(parent)?.into_parts();
                Ok(Arc::new(StagedFile {
                    file,
                    temp_path,
                    path,
                    metrics: tracker.begin_io("put"),
                    tracker,
                    _control: control,
                    lifecycle,
                    mode,
                    fsync,
                }))
            },
        )?
        .await??;
        Ok(Self {
            file: Some(file),
            pending: None,
            finishing: None,
            result: None,
            published_control: None,
            pool,
            position: 0,
            failed: false,
        })
    }

    fn poll_pending(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let Some(task) = self.pending.as_mut() else {
            return Poll::Ready(Ok(()));
        };
        let result = match Pin::new(task).poll(cx) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(result) => result,
        };
        self.pending = None;
        if result.is_err() {
            self.failed = true;
        }
        Poll::Ready(result)
    }

    fn closed_error() -> io::Error {
        io::Error::other("resource-aware local output is failed or closed")
    }
}

impl AsyncWrite for ResourceLocalWriter {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bytes: &[u8],
    ) -> Poll<io::Result<usize>> {
        if self.failed || self.file.is_none() {
            return Poll::Ready(Err(Self::closed_error()));
        }
        match self.poll_pending(cx) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(Err(error)) => return Poll::Ready(Err(error)),
            Poll::Ready(Ok(())) => {}
        }
        if bytes.is_empty() {
            return Poll::Ready(Ok(0));
        }
        let length = bytes.len().min(WRITE_CHUNK_BYTES);
        if self.position.checked_add(length).is_none() {
            self.failed = true;
            return Poll::Ready(Err(io::Error::other("local output position overflow")));
        }
        let charge = match self.pool.reserve(
            "local output write buffer and task",
            length + WRITE_CONTROL_BYTES,
        ) {
            Ok(charge) => charge,
            Err(error) => {
                self.failed = true;
                return Poll::Ready(Err(io::Error::other(error)));
            }
        };
        let data = crate::local::resources::OwnedIoBuffer {
            bytes: bytes[..length].to_vec(),
            _reservation: Some(charge),
        };
        let Some(file) = self.file.clone() else {
            self.failed = true;
            return Poll::Ready(Err(Self::closed_error()));
        };
        let task = io_task_with_resources(self.pool.clone(), "local output write", move || {
            let result = crate::local::resources::write_all(&file.file, data.as_ref());
            (result, file, data)
        });
        let task = match task {
            Ok(task) => task,
            Err(error) => {
                self.failed = true;
                return Poll::Ready(Err(io::Error::other(error)));
            }
        };
        self.pending = Some(Box::pin(async move {
            let (result, _file, _data) = task.await.map_err(io::Error::other)?;
            result
        }));
        // Acknowledge the admitted copy now. A later call must drain it before
        // accepting its own buffer; flush/shutdown report actual I/O failures.
        self.position += length;
        Poll::Ready(Ok(length))
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        if self.failed {
            return Poll::Ready(Err(Self::closed_error()));
        }
        self.poll_pending(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        if self.failed {
            return Poll::Ready(Err(Self::closed_error()));
        }
        if self.result.is_some() {
            return Poll::Ready(Ok(()));
        }
        match self.poll_pending(cx) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(Err(error)) => return Poll::Ready(Err(error)),
            Poll::Ready(Ok(_)) => {}
        }
        if self.finishing.is_none() {
            let Some(file) = self.file.take() else {
                self.failed = true;
                return Poll::Ready(Err(Self::closed_error()));
            };
            let Ok(file) = Arc::try_unwrap(file) else {
                self.failed = true;
                return Poll::Ready(Err(io::Error::other(
                    "local output still has an outstanding file owner",
                )));
            };
            let size = self.position;
            let task =
                io_task_with_resources(self.pool.clone(), "local output publication", move || {
                    let StagedFile {
                        file,
                        temp_path,
                        path,
                        tracker,
                        metrics,
                        _control,
                        lifecycle,
                        mode,
                        fsync,
                    } = file;
                    let result = (|| -> Result<WriteResult> {
                        if fsync {
                            file.sync_all()?;
                        }
                        drop(file);
                        let local = crate::local::to_local_path(&path);
                        let published = if matches!(mode, PutMode::Create) {
                            temp_path.persist_noclobber(&local)
                        } else {
                            temp_path.persist(&local)
                        };
                        published.map_err(|error| {
                            Error::io(format!(
                                "failed to publish local output {path}: {}",
                                error.error
                            ))
                        })?;
                        if let Some(lifecycle) = &lifecycle {
                            lifecycle.published();
                        }
                        #[cfg(unix)]
                        if fsync {
                            let parent =
                                std::path::Path::new(&local).parent().ok_or_else(|| {
                                    Error::io(format!("local output {path} has no parent"))
                                })?;
                            File::open(parent)?.sync_all()?;
                        }
                        let metadata = std::fs::metadata(&local)?;
                        Ok(WriteResult {
                            size,
                            e_tag: Some(get_etag(&metadata)),
                        })
                    })();
                    metrics.record(&result, size as u64);
                    if result.is_ok() {
                        tracker.record_write("put", path, size as u64);
                    }
                    // Ownership accompanies the actual publication and its result,
                    // including an unobserved result after the writer is dropped.
                    FinishedOutput {
                        result,
                        control: _control,
                    }
                });
            self.finishing = Some(match task {
                Ok(task) => task,
                Err(error) => {
                    self.failed = true;
                    return Poll::Ready(Err(io::Error::other(error)));
                }
            });
        }
        let Some(task) = self.finishing.as_mut() else {
            self.failed = true;
            return Poll::Ready(Err(Self::closed_error()));
        };
        let result = match Pin::new(task).poll(cx) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(result) => result.map_err(io::Error::other),
        };
        self.finishing = None;
        match result {
            Ok(FinishedOutput {
                result: Ok(result),
                control,
            }) => {
                self.result = Some(result);
                self.published_control = Some(control);
                Poll::Ready(Ok(()))
            }
            Ok(FinishedOutput {
                result: Err(error),
                control: _,
            }) => {
                self.failed = true;
                Poll::Ready(Err(io::Error::other(error)))
            }
            Err(error) => {
                self.failed = true;
                Poll::Ready(Err(error))
            }
        }
    }
}

#[async_trait]
impl Writer for ResourceLocalWriter {
    async fn tell(&mut self) -> Result<usize> {
        Ok(self.position)
    }

    async fn shutdown(&mut self) -> Result<WriteResult> {
        AsyncWriteExt::shutdown(self).await?;
        self.result.clone().ok_or_else(|| {
            Error::internal("local output shutdown completed without a write result")
        })
    }
}
