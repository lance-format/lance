// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reservations accompany the actual local read, including detached blocking I/O.

use std::{fs::File, ops::Range, sync::Arc};

use bytes::Bytes;
use futures::{StreamExt, future::BoxFuture};
use lance_core::{
    Error,
    utils::{memory::MemoryReserver, tokio::io_task_with_resources},
};
use object_store::path::Path;

#[cfg(windows)]
use super::read_exact_at;
#[cfg(unix)]
use std::os::unix::fs::FileExt;

use super::LocalObjectReader;
use crate::{traits::ByteStream, utils::tracking_store::IOTracker};

pub(super) struct ReaderResources {
    _control: Box<dyn Send + Sync>,
}

impl std::fmt::Debug for ReaderResources {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReaderResources")
            .finish_non_exhaustive()
    }
}

pub struct OwnedIoBuffer {
    pub(crate) bytes: Vec<u8>,
    pub(crate) _reservation: Option<Box<dyn Send + Sync>>,
}

impl AsRef<[u8]> for OwnedIoBuffer {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

impl LocalObjectReader {
    pub(crate) async fn open_with_resources(
        path: &Path,
        block_size: usize,
        known_size: Option<usize>,
        io_tracker: Arc<IOTracker>,
        pool: Arc<dyn MemoryReserver>,
    ) -> lance_core::Result<Box<dyn crate::traits::Reader>> {
        let bytes = path
            .as_ref()
            .len()
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_add(8192))
            .ok_or_else(|| Error::invalid_input("local reader state size overflow"))?;
        let control = pool.reserve("local reader file and path state", bytes)?;
        let path = path.clone();
        let reader_pool = pool.clone();
        io_task_with_resources(pool, "local file open", move || {
            let file = File::open(super::to_local_path(&path)).map_err(|error| {
                if error.kind() == std::io::ErrorKind::NotFound {
                    Error::not_found(path.to_string())
                } else {
                    error.into()
                }
            })?;
            Ok(Box::new(Self {
                memory_pool: Some(reader_pool),
                _resources: Some(ReaderResources { _control: control }),
                file: Arc::new(file),
                block_size,
                size: tokio::sync::OnceCell::new_with(known_size),
                path,
                io_tracker,
            }) as Box<dyn crate::traits::Reader>)
        })?
        .await?
    }

    pub(super) fn read_with_resources(
        file: Arc<File>,
        path: Path,
        tracker: Arc<IOTracker>,
        pool: Arc<dyn MemoryReserver>,
        range: Range<usize>,
    ) -> BoxFuture<'static, object_store::Result<Bytes>> {
        let admitted = reserve_read(
            Some(pool.as_ref()),
            &range,
            "local read buffer and task",
            1024,
        );
        Box::pin(async move {
            let owner = admitted?;
            let length = range.end - range.start;
            let offset = range.start as u64;
            let metrics = tracker.begin_io("get");
            let result = io_task_with_resources(
                pool,
                "local range read",
                move || -> std::io::Result<Bytes> {
                    // The reservation moves into the actual blocking closure. Dropping
                    // its JoinHandle cannot return the allowance while it still runs.
                    let mut data = OwnedIoBuffer {
                        bytes: vec![0; length],
                        _reservation: owner,
                    };
                    read_exact(file, &mut data.bytes, offset)?;
                    Ok(Bytes::from_owner(data))
                },
            )?
            .await?
            .map_err(|source| object_store::Error::Generic {
                store: "LocalFileSystem",
                source: Box::new(source),
            });
            metrics.record(&result, length as u64);
            if result.is_ok() {
                tracker.record_read(
                    "get_range",
                    path,
                    length as u64,
                    Some(offset..offset + length as u64),
                );
            }
            result
        })
    }

    pub(super) fn stream_with_resources(&self, range: Range<usize>) -> ByteStream {
        let file = self.file.clone();
        let path = self.path.clone();
        let tracker = self.io_tracker.clone();
        let pool = self.memory_pool.clone();
        let chunk_bytes = self.block_size.max(8 * 1024);
        futures::stream::try_unfold(range.start, move |start| {
            let file = file.clone();
            let path = path.clone();
            let tracker = tracker.clone();
            let pool = pool.clone();
            let end = range.end;
            async move {
                if start > end {
                    return Err(Error::invalid_input(format!(
                        "local read range is reversed: {start}..{end}"
                    ))
                    .into());
                }
                if start == end {
                    return Ok(None);
                }
                let pool = pool.ok_or_else(|| {
                    Error::internal("resource-aware local stream lost its memory pool")
                })?;
                let next = start + (end - start).min(chunk_bytes);
                let bytes =
                    Self::read_with_resources(file, path, tracker, pool, start..next).await?;
                Ok(Some((bytes, next)))
            }
        })
        .boxed()
    }
}

pub fn reserve_read(
    pool: Option<&dyn MemoryReserver>,
    range: &Range<usize>,
    stage: &str,
    control_bytes: usize,
) -> lance_core::Result<Option<Box<dyn Send + Sync>>> {
    let bytes = range
        .end
        .checked_sub(range.start)
        .ok_or_else(|| Error::invalid_input(format!("{stage}: read range is reversed: {range:?}")))?
        .checked_add(control_bytes)
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "{stage}: read reservation size overflow: {range:?}"
            ))
        })?;
    pool.map(|pool| pool.reserve(stage, bytes)).transpose()
}

// These native operations are shared by scratch and destination I/O. Their
// callers carry the admitted buffer and file owners into the completion task.
pub fn write_all(mut writer: impl std::io::Write, mut bytes: &[u8]) -> std::io::Result<()> {
    while !bytes.is_empty() {
        match writer.write(bytes) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "local write made no progress",
                ));
            }
            Ok(n) => bytes = &bytes[n..],
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

pub fn read_exact(file: Arc<File>, bytes: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        file.read_exact_at(bytes, offset)
    }
    #[cfg(windows)]
    {
        read_exact_at(file, bytes, offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct PartialWriter {
        interrupted: bool,
        fail: Option<std::io::ErrorKind>,
        bytes: Vec<u8>,
    }
    impl std::io::Write for PartialWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if !self.interrupted {
                self.interrupted = true;
                return Err(std::io::ErrorKind::Interrupted.into());
            }
            if let Some(kind) = self.fail {
                return if kind == std::io::ErrorKind::WriteZero {
                    Ok(0)
                } else {
                    Err(kind.into())
                };
            }
            self.bytes.push(bytes[0]);
            Ok(1)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    #[rstest::rstest]
    #[case::short(None)]
    #[case::zero(Some(std::io::ErrorKind::WriteZero))]
    #[case::failure(Some(std::io::ErrorKind::PermissionDenied))]
    #[test]
    fn native_write_retries_and_errors(#[case] fail: Option<std::io::ErrorKind>) {
        let mut writer = PartialWriter {
            interrupted: false,
            fail,
            bytes: Vec::new(),
        };
        let result = write_all(&mut writer, b"abc");
        if let Some(kind) = fail {
            assert_eq!(result.unwrap_err().kind(), kind);
        } else {
            result.unwrap();
            assert_eq!(writer.bytes, b"abc");
        }
    }
}
