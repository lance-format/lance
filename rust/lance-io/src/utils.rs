// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{cmp::min, num::NonZero, ops::Range, sync::atomic::AtomicU64};

use byteorder::{ByteOrder, LittleEndian};
use bytes::{Bytes, BytesMut};
use futures::{Stream, StreamExt, TryStreamExt};
use lance_core::deepsize::DeepSizeOf;
use prost::Message;
use serde::{Deserialize, Serialize};

use crate::traits::{ProtoStruct, Reader};
use lance_core::{Error, Result};

pub mod tracking_store;

/// Chunk size for splitting a large metadata read into concurrent range requests.
///
/// A single object-store GET streams its body over one connection, so its
/// throughput is capped by the TCP window over the round-trip time; on
/// high-latency links that tops out in the tens of MB/s. Fetching the range as
/// a window of concurrent chunk requests multiplies that per-connection limit.
/// 16 MiB keeps per-request overhead negligible (a ~1 GiB manifest costs ~64
/// GET requests) while a `Reader::io_parallelism` window of such chunks is
/// enough to saturate the link.
pub const METADATA_READ_CHUNK_SIZE: usize = 16 * 1024 * 1024;

/// Read `range` from `reader` as `chunk_size`-sized concurrent range requests,
/// yielding the chunks in file order. Concurrency is bounded by
/// [`Reader::io_parallelism`], clamped to at least 1: a `buffered(0)` window
/// never polls its input, so an unvalidated reader value (e.g.
/// `LANCE_URING_IO_PARALLELISM=0`) would hang the read.
pub fn read_range_in_chunks(
    reader: &dyn Reader,
    range: Range<usize>,
    chunk_size: usize,
) -> impl Stream<Item = object_store::Result<Bytes>> + '_ {
    let end = range.end;
    let chunk_ranges = range
        .step_by(chunk_size)
        .map(move |start| start..min(start + chunk_size, end));
    futures::stream::iter(chunk_ranges.map(|chunk| reader.get_range(chunk)))
        .buffered(reader.io_parallelism().max(1))
}

/// Read a protobuf message at file position 'pos'.
///
/// We write protobuf by first writing the length of the message as a u32,
/// followed by the message itself.
pub async fn read_message<M: Message + Default>(reader: &dyn Reader, pos: usize) -> Result<M> {
    let file_size = reader.size().await?;
    // A message is a u32 length prefix followed by its body; both must lie before
    // the end. A `pos` too close to the end means the reader size is too small
    // (e.g. a stale cached size). Reject it rather than slice a short buffer and
    // panic.
    if pos + 4 > file_size {
        return Err(Error::io("file size is too small".to_string()));
    }

    let range = pos..min(pos + reader.block_size(), file_size);
    let buf = reader.get_range(range.clone()).await?;
    let msg_len = LittleEndian::read_u32(&buf) as usize;

    if msg_len + 4 > buf.len() {
        let remaining_range = range.end..min(4 + pos + msg_len, file_size);
        // Assemble into one pre-allocated buffer; fetching the remainder as
        // concurrent chunks lifts the single-connection throughput cap on
        // large messages (e.g. manifests of datasets with many fragments).
        let mut full = BytesMut::with_capacity(buf.len() + remaining_range.len());
        full.extend_from_slice(&buf);
        let mut chunks = read_range_in_chunks(reader, remaining_range, METADATA_READ_CHUNK_SIZE);
        while let Some(chunk) = chunks.try_next().await? {
            full.extend_from_slice(&chunk);
        }
        if full.len() < msg_len + 4 {
            return Err(Error::io("file size is too small".to_string()));
        }
        Ok(M::decode(&full[4..4 + msg_len])?)
    } else {
        Ok(M::decode(&buf[4..4 + msg_len])?)
    }
}

/// Read a Protobuf-backed struct at file position: `pos`.
// TODO: pub(crate)
pub async fn read_struct<
    M: Message + Default + 'static,
    T: ProtoStruct<Proto = M> + TryFrom<M, Error = Error>,
>(
    reader: &dyn Reader,
    pos: usize,
) -> Result<T> {
    let msg = read_message::<M>(reader, pos).await?;
    T::try_from(msg)
}

pub async fn read_last_block(reader: &dyn Reader) -> object_store::Result<Bytes> {
    let file_size = reader.size().await?;
    let block_size = reader.block_size();
    let begin = file_size.saturating_sub(block_size);
    reader.get_range(begin..file_size).await
}

pub fn read_metadata_offset(bytes: &Bytes) -> Result<usize> {
    let len = bytes.len();
    if len < 16 {
        return Err(Error::io(format!(
            "does not have sufficient data, len: {}, bytes: {:?}",
            len, bytes
        )));
    }
    let offset_bytes = bytes.slice(len - 16..len - 8);
    Ok(LittleEndian::read_u64(offset_bytes.as_ref()) as usize)
}

/// Read the version from the footer bytes
pub fn read_version(bytes: &Bytes) -> Result<(u16, u16)> {
    let len = bytes.len();
    if len < 8 {
        return Err(Error::io(format!(
            "does not have sufficient data, len: {}, bytes: {:?}",
            len, bytes
        )));
    }

    let major_version = LittleEndian::read_u16(bytes.slice(len - 8..len - 6).as_ref());
    let minor_version = LittleEndian::read_u16(bytes.slice(len - 6..len - 4).as_ref());
    Ok((major_version, minor_version))
}

/// Read protobuf from a buffer.
pub fn read_message_from_buf<M: Message + Default>(buf: &Bytes) -> Result<M> {
    let msg_len = LittleEndian::read_u32(buf) as usize;
    Ok(M::decode(&buf[4..4 + msg_len])?)
}

/// Read a Protobuf-backed struct from a buffer.
pub fn read_struct_from_buf<
    M: Message + Default,
    T: ProtoStruct<Proto = M> + TryFrom<M, Error = Error>,
>(
    buf: &Bytes,
) -> Result<T> {
    let msg: M = read_message_from_buf(buf)?;
    T::try_from(msg)
}

/// A cached file size.
///
/// This wraps an atomic u64 to allow setting the cached file size without
/// needed a mutable reference.
///
/// Zero is interpreted as unknown.
#[derive(Debug, DeepSizeOf)]
pub struct CachedFileSize(AtomicU64);

impl<'de> Deserialize<'de> for CachedFileSize {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let size = Option::<u64>::deserialize(deserializer)?.unwrap_or(0);
        Ok(Self::new(size))
    }
}

impl Serialize for CachedFileSize {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let size = self.0.load(std::sync::atomic::Ordering::Relaxed);
        if size == 0 {
            serializer.serialize_none()
        } else {
            serializer.serialize_u64(size)
        }
    }
}

impl From<Option<NonZero<u64>>> for CachedFileSize {
    fn from(size: Option<NonZero<u64>>) -> Self {
        match size {
            Some(size) => Self(AtomicU64::new(size.into())),
            None => Self(AtomicU64::new(0)),
        }
    }
}

impl Default for CachedFileSize {
    fn default() -> Self {
        Self(AtomicU64::new(0))
    }
}

impl Clone for CachedFileSize {
    fn clone(&self) -> Self {
        Self(AtomicU64::new(
            self.0.load(std::sync::atomic::Ordering::Relaxed),
        ))
    }
}

impl PartialEq for CachedFileSize {
    fn eq(&self, other: &Self) -> bool {
        self.0.load(std::sync::atomic::Ordering::Relaxed)
            == other.0.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Eq for CachedFileSize {}

impl CachedFileSize {
    /// Create a `CachedFileSize` from a raw byte count.
    ///
    /// Passing `0` is equivalent to calling [`unknown`](Self::unknown): the
    /// type interprets zero as "size not yet known".
    pub fn new(size: u64) -> Self {
        Self(AtomicU64::new(size))
    }

    pub fn unknown() -> Self {
        Self(AtomicU64::new(0))
    }

    pub fn get(&self) -> Option<NonZero<u64>> {
        NonZero::new(self.0.load(std::sync::atomic::Ordering::Relaxed))
    }

    pub fn set(&self, size: NonZero<u64>) {
        self.0
            .store(size.into(), std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use bytes::{Bytes, BytesMut};
    use futures::TryStreamExt;
    use object_store::path::Path;

    use crate::{
        Error, Result,
        object_reader::CloudObjectReader,
        object_store::{DEFAULT_DOWNLOAD_RETRY_COUNT, ObjectStore},
        object_writer::ObjectWriter,
        traits::{ProtoStruct, WriteExt, Writer},
        utils::{METADATA_READ_CHUNK_SIZE, read_range_in_chunks, read_struct},
    };

    // Bytes is a prost::Message, since we don't have any .proto files in this crate we
    // can use it to simulate a real message object.
    #[derive(Debug, PartialEq)]
    struct BytesWrapper(Bytes);

    impl ProtoStruct for BytesWrapper {
        type Proto = Bytes;
    }

    impl From<&BytesWrapper> for Bytes {
        fn from(value: &BytesWrapper) -> Self {
            value.0.clone()
        }
    }

    impl TryFrom<Bytes> for BytesWrapper {
        type Error = Error;
        fn try_from(value: Bytes) -> Result<Self> {
            Ok(Self(value))
        }
    }

    #[tokio::test]
    async fn test_write_proto_structs() {
        let store = ObjectStore::memory();
        let path = Path::from("/foo");

        let mut object_writer = ObjectWriter::new(&store, &path).await.unwrap();
        assert_eq!(object_writer.tell().await.unwrap(), 0);

        let some_message = BytesWrapper(Bytes::from(vec![10, 20, 30]));

        let pos = object_writer.write_struct(&some_message).await.unwrap();
        assert_eq!(pos, 0);
        object_writer.shutdown().await.unwrap();

        let object_reader =
            CloudObjectReader::new(store.inner, path, 1024, None, DEFAULT_DOWNLOAD_RETRY_COUNT)
                .unwrap();
        let actual: BytesWrapper = read_struct(&object_reader, pos).await.unwrap();
        assert_eq!(some_message, actual);
    }

    #[tokio::test]
    async fn test_read_range_in_chunks_reassembles_in_order() {
        let store = ObjectStore::memory();
        let path = Path::from("/chunked");
        // Patterned data with a range that neither starts nor ends on a chunk
        // boundary, so ordering or off-by-one mistakes change the bytes.
        let data: Vec<u8> = (0..10 * 1024 + 37).map(|i| (i % 251) as u8).collect();
        store.put(&path, &data).await.unwrap();
        let reader = store.open(&path).await.unwrap();

        let range = 5..data.len() - 3;
        let mut assembled = BytesMut::new();
        let mut chunks = read_range_in_chunks(reader.as_ref(), range.clone(), 1024);
        while let Some(chunk) = chunks.try_next().await.unwrap() {
            assembled.extend_from_slice(&chunk);
        }
        assert_eq!(assembled.as_ref(), &data[range]);
    }

    #[tokio::test]
    async fn test_read_range_in_chunks_zero_parallelism_reader() {
        // A reader advertising io_parallelism 0 (e.g. LANCE_URING_IO_PARALLELISM=0)
        // must not hang the chunked read: the window is clamped to at least 1.
        let store = ObjectStore::memory();
        let path = Path::from("/zero_parallelism");
        let data: Vec<u8> = (0..4096).map(|i| (i % 249) as u8).collect();
        store.put(&path, &data).await.unwrap();
        let reader =
            CloudObjectReader::new(store.inner, path, 1024, None, DEFAULT_DOWNLOAD_RETRY_COUNT)
                .unwrap()
                .with_io_parallelism(0);

        let assembled = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            let mut buf = BytesMut::new();
            let mut chunks = read_range_in_chunks(&reader, 0..data.len(), 1024);
            while let Some(chunk) = chunks.try_next().await.unwrap() {
                buf.extend_from_slice(&chunk);
            }
            buf
        })
        .await
        .expect("chunked read with a zero-parallelism reader must not hang");
        assert_eq!(assembled.as_ref(), &data[..]);
    }

    #[tokio::test]
    async fn test_read_message_larger_than_chunk_size() {
        // A message body crossing METADATA_READ_CHUNK_SIZE forces read_message
        // to fetch the remainder as multiple concurrent chunks.
        let store = ObjectStore::memory();
        let path = Path::from("/large_message");

        let mut object_writer = ObjectWriter::new(&store, &path).await.unwrap();
        let payload: Vec<u8> = (0..METADATA_READ_CHUNK_SIZE + 5 * 1024 * 1024)
            .map(|i| (i % 253) as u8)
            .collect();
        let message = BytesWrapper(Bytes::from(payload));
        let pos = object_writer.write_struct(&message).await.unwrap();
        object_writer.shutdown().await.unwrap();

        let object_reader =
            CloudObjectReader::new(store.inner, path, 4096, None, DEFAULT_DOWNLOAD_RETRY_COUNT)
                .unwrap();
        let actual: BytesWrapper = read_struct(&object_reader, pos).await.unwrap();
        assert_eq!(message, actual);
    }

    #[tokio::test]
    async fn test_copy_reader_to_writer() {
        let store = ObjectStore::memory();
        let src = Path::from("/src");
        let dst = Path::from("/dst");
        store.put(&src, b"abcdef").await.unwrap();

        let reader = store.open(&src).await.unwrap();
        let mut writer = store.create(&dst).await.unwrap();
        let copied = writer.copy_from_reader(reader.as_ref()).await.unwrap();
        writer.shutdown().await.unwrap();

        assert_eq!(copied, 6);
        assert_eq!(store.read_one_all(&dst).await.unwrap().as_ref(), b"abcdef");
    }

    #[tokio::test]
    async fn test_copy_reader_range_to_writer() {
        let store = ObjectStore::memory();
        let src = Path::from("/src-range");
        let dst = Path::from("/dst-range");
        store.put(&src, b"abcdef").await.unwrap();

        let reader = store.open(&src).await.unwrap();
        let mut writer = store.create(&dst).await.unwrap();
        let copied = writer
            .copy_range_from_reader(reader.as_ref(), 2..5)
            .await
            .unwrap();
        writer.shutdown().await.unwrap();

        assert_eq!(copied, 3);
        assert_eq!(store.read_one_all(&dst).await.unwrap().as_ref(), b"cde");
    }
}
