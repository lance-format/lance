// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::{Attributes, ObjectMeta};
use crate::object_store::{Error, Result};
use bytes::{Bytes, BytesMut};
use futures::{StreamExt, TryStreamExt, stream::BoxStream};
use http::Extensions;
use std::ops::Range;

#[derive(Debug, thiserror::Error)]
#[error("Read {} at {range:?}: {source}", path.display())]
struct FileReadError {
    path: std::path::PathBuf,
    range: Range<u64>,
    #[source]
    source: std::io::Error,
}

fn read_file_range(
    file: &mut std::fs::File,
    path: &std::path::Path,
    range: Range<u64>,
) -> Result<Bytes> {
    use std::io::{Read, Seek, SeekFrom};
    let size = range
        .end
        .checked_sub(range.start)
        .and_then(|len| usize::try_from(len).ok())
        .ok_or_else(|| Error::Generic {
            store: "local",
            source: format!("Invalid response range {range:?} for {}", path.display()).into(),
        })?;
    let mut buffer = vec![0; size];
    file.seek(SeekFrom::Start(range.start))
        .and_then(|_| file.read_exact(&mut buffer))
        .map_err(|source| Error::Generic {
            store: "local",
            source: Box::new(FileReadError {
                path: path.into(),
                range,
                source,
            }),
        })?;
    Ok(Bytes::from(buffer))
}

/// A read response, retaining the returned range, metadata and streaming body.
#[derive(Debug)]
pub struct GetResult {
    /// The local file or streaming response body.
    pub payload: GetResultPayload,
    /// Metadata for the complete object.
    pub meta: ObjectMeta,
    /// Byte range represented by this response.
    pub range: Range<u64>,
    /// Content headers and user metadata.
    pub attributes: Attributes,
    /// Context propagated by the backend, such as cache observations.
    pub extensions: Extensions,
}

/// A local file or an asynchronous body stream.
pub enum GetResultPayload {
    /// An opened file and its physical path; the response range still applies.
    File(std::fs::File, std::path::PathBuf),
    /// A body that can fail after the initial request succeeds.
    Stream(BoxStream<'static, Result<Bytes>>),
}

impl std::fmt::Debug for GetResultPayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::File(_, path) => f.debug_tuple("File").field(path).finish(),
            Self::Stream(_) => f.write_str("Stream"),
        }
    }
}

impl GetResult {
    /// Consume and collect this response, propagating errors from the body.
    /// Local files read the entire response range in a single blocking task.
    pub async fn bytes(self) -> Result<Bytes> {
        let size_hint = self
            .range
            .end
            .checked_sub(self.range.start)
            .and_then(|size| usize::try_from(size).ok());
        let mut stream = match self.payload {
            GetResultPayload::File(mut file, path) => {
                let read = move || read_file_range(&mut file, &path, self.range);
                return match tokio::runtime::Handle::try_current() {
                    Ok(handle) => {
                        handle
                            .spawn_blocking(read)
                            .await
                            .map_err(|source| Error::Generic {
                                store: "local",
                                source: Box::new(source),
                            })?
                    }
                    Err(_) => read(),
                };
            }
            GetResultPayload::Stream(stream) => stream,
        };
        let first = stream.try_next().await?.unwrap_or_default();
        let Some(second) = stream.try_next().await? else {
            return Ok(first);
        };
        let mut buffer =
            BytesMut::with_capacity(size_hint.unwrap_or_else(|| first.len() + second.len()));
        buffer.extend_from_slice(&first);
        buffer.extend_from_slice(&second);
        // Release consumed chunks instead of retaining a second complete body.
        drop((first, second));
        while let Some(chunk) = stream.try_next().await? {
            buffer.extend_from_slice(&chunk);
        }
        Ok(buffer.freeze())
    }

    /// Consume the body as a stream without buffering the entire object.
    ///
    /// Local files are read in bounded chunks on Tokio's blocking pool when a
    /// runtime is available. Only the response's byte range is read.
    pub fn into_stream(self) -> BoxStream<'static, Result<Bytes>> {
        match self.payload {
            GetResultPayload::Stream(stream) => stream,
            GetResultPayload::File(file, path) => {
                futures::stream::try_unfold(
                    (file, path, self.range),
                    |(mut file, path, range)| async move {
                        if range.start == range.end {
                            return Ok(None);
                        }
                        let read = move || {
                            let remaining =
                                range.end.checked_sub(range.start).ok_or_else(|| {
                                    Error::Generic {
                                        store: "local",
                                        source: format!(
                                            "Invalid response range {range:?} for {}",
                                            path.display()
                                        )
                                        .into(),
                                    }
                                })?;
                            let end = range.start + remaining.min(64 * 1024);
                            let buffer = read_file_range(&mut file, &path, range.start..end)?;
                            Ok(Some((buffer, (file, path, end..range.end))))
                        };
                        match tokio::runtime::Handle::try_current() {
                            Ok(handle) => handle.spawn_blocking(read).await.map_err(|source| {
                                Error::Generic {
                                    store: "local",
                                    source: Box::new(source),
                                }
                            })?,
                            Err(_) => read(),
                        }
                    },
                )
                .boxed()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object_store::Path;

    fn response(payload: GetResultPayload, range: Range<u64>, size: u64) -> GetResult {
        GetResult {
            payload,
            meta: ObjectMeta {
                location: Path::from("file"),
                last_modified: chrono::Utc::now(),
                size,
                e_tag: None,
                version: None,
            },
            range,
            attributes: Default::default(),
            extensions: Default::default(),
        }
    }

    #[tokio::test]
    async fn a_single_chunk_keeps_its_allocation() {
        let bytes = Bytes::from(vec![7; 123]);
        let original = bytes.clone();
        let result = response(
            GetResultPayload::Stream(futures::stream::once(async { Ok(bytes) }).boxed()),
            0..123,
            123,
        )
        .bytes()
        .await
        .unwrap();
        assert_eq!(result, original);
        assert_eq!(result.as_ptr(), original.as_ptr());
    }

    #[tokio::test]
    async fn collecting_a_body_releases_consumed_chunks() {
        let first = Bytes::from(vec![1; 16]);
        let second = Bytes::from(vec![2; 24]);
        let stream = futures::stream::iter(0..3).map(move |index| {
            Ok(match index {
                0 => first.clone(),
                1 => second.clone(),
                _ => {
                    assert!(first.is_unique());
                    assert!(second.is_unique());
                    Bytes::from_static(b"tail")
                }
            })
        });
        let result = response(GetResultPayload::Stream(stream.boxed()), 0..44, 44)
            .bytes()
            .await
            .unwrap();
        assert_eq!(
            result.as_ref(),
            [vec![1; 16], vec![2; 24], b"tail".to_vec()].concat()
        );
    }

    #[tokio::test]
    async fn file_stream_is_bounded_and_reports_truncation() {
        let file = tempfile::NamedTempFile::new().unwrap();
        file.as_file().set_len(128 * 1024).unwrap();
        let mut stream = response(
            GetResultPayload::File(file.reopen().unwrap(), file.path().into()),
            2..100_000,
            128 * 1024,
        )
        .into_stream();
        let first = stream.try_next().await.unwrap().unwrap();
        assert_eq!(first.len(), 64 * 1024);
        file.as_file().set_len(64 * 1024 + 2).unwrap();
        let error = stream.try_next().await.unwrap_err();
        let Error::Generic { source, .. } = error else {
            panic!("expected a file read error")
        };
        assert_eq!(
            source
                .downcast_ref::<FileReadError>()
                .unwrap()
                .source
                .kind(),
            std::io::ErrorKind::UnexpectedEof
        );
    }

    #[rstest::rstest]
    #[case::empty(5..5, false)]
    #[case::large(2..100_000, false)]
    #[case::truncated(2..150_000, true)]
    #[tokio::test]
    async fn file_bytes_preserves_range_and_truncation(
        #[case] range: Range<u64>,
        #[case] truncated: bool,
    ) {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        let data: Vec<u8> = (0..128 * 1024).map(|i| (i % 251) as u8).collect();
        std::io::Write::write_all(&mut file, &data).unwrap();
        let result = response(
            GetResultPayload::File(file.reopen().unwrap(), file.path().into()),
            range.clone(),
            data.len() as u64,
        )
        .bytes()
        .await;
        if truncated {
            let Error::Generic { source, .. } = result.unwrap_err() else {
                panic!("expected file read failure")
            };
            assert_eq!(
                source
                    .downcast_ref::<FileReadError>()
                    .unwrap()
                    .source
                    .kind(),
                std::io::ErrorKind::UnexpectedEof
            );
        } else {
            assert_eq!(
                result.unwrap().as_ref(),
                &data[range.start as usize..range.end as usize]
            );
        }
    }

    #[test]
    fn local_range_can_be_read_without_a_tokio_runtime() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut file, b"0123456789").unwrap();
        let result = response(
            GetResultPayload::File(file.reopen().unwrap(), file.path().into()),
            2..5,
            10,
        );
        assert_eq!(
            futures::executor::block_on(result.bytes())
                .unwrap()
                .as_ref(),
            b"234"
        );
    }
}
