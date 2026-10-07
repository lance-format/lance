// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{cmp::min, collections::VecDeque, num::NonZero, ops::Range, sync::atomic::AtomicU64};

use byteorder::{ByteOrder, LittleEndian};
use bytes::{Buf, Bytes};
use futures::{Stream, StreamExt, TryStreamExt};
use lance_core::deepsize::DeepSizeOf;
use prost::Message;
use serde::{Deserialize, Serialize};

use crate::traits::{ProtoStruct, Reader};
use lance_core::{Error, Result};

#[cfg(feature = "test-util")]
pub mod failpoint;
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

/// A [`Buf`] over a sequence of [`Bytes`] chunks in file order.
///
/// Lets a protobuf message fetched as concurrent range requests be decoded
/// without first copying the chunks into one contiguous buffer. A copy of a
/// large message is not free: the destination pages are touched for the first
/// time, so the copy costs a page fault per 4 KiB on top of the `memmove`
/// (measured at ~0.5 s and ~1 GiB of peak memory for a 945 MiB manifest).
/// `copy_to_bytes` hands out a slice of the underlying chunk when the request
/// lies within one, so `bytes`-typed fields decode zero-copy as well.
#[derive(Debug, Default)]
pub struct ChunkedBuf {
    chunks: VecDeque<Bytes>,
    remaining: usize,
}

impl ChunkedBuf {
    /// Append `chunk` after the chunks pushed so far. Empty chunks are dropped
    /// so `chunk()` never reports an empty slice while data remains.
    pub fn push(&mut self, chunk: Bytes) {
        if !chunk.is_empty() {
            self.remaining += chunk.len();
            self.chunks.push_back(chunk);
        }
    }

    /// Keep only the first `len` bytes.
    ///
    /// Prost decodes until the buffer is exhausted, so a message followed by
    /// trailing bytes (a footer, the next section) must be cut to its exact
    /// length before decoding.
    pub fn truncate(&mut self, len: usize) {
        if len >= self.remaining {
            return;
        }
        let mut to_drop = self.remaining - len;
        while to_drop > 0 {
            let Some(back) = self.chunks.back_mut() else {
                break;
            };
            if to_drop >= back.len() {
                to_drop -= back.len();
                self.chunks.pop_back();
            } else {
                back.truncate(back.len() - to_drop);
                to_drop = 0;
            }
        }
        self.remaining = len;
    }
}

impl Buf for ChunkedBuf {
    fn remaining(&self) -> usize {
        self.remaining
    }

    fn chunk(&self) -> &[u8] {
        self.chunks.front().map(Bytes::as_ref).unwrap_or(&[])
    }

    fn advance(&mut self, mut cnt: usize) {
        assert!(
            cnt <= self.remaining,
            "cannot advance ChunkedBuf by {cnt} bytes: only {} remain",
            self.remaining
        );
        self.remaining -= cnt;
        while cnt > 0 {
            let front = self
                .chunks
                .front_mut()
                .expect("remaining bytes imply a chunk");
            if cnt < front.len() {
                front.advance(cnt);
                return;
            }
            cnt -= front.len();
            self.chunks.pop_front();
        }
    }

    fn copy_to_bytes(&mut self, len: usize) -> Bytes {
        assert!(
            len <= self.remaining,
            "cannot copy {len} bytes from ChunkedBuf: only {} remain",
            self.remaining
        );
        let within_front = self.chunks.front().is_some_and(|front| len <= front.len());
        if within_front {
            let front = self
                .chunks
                .front_mut()
                .expect("checked that a front chunk exists");
            let bytes = front.split_to(len);
            self.remaining -= len;
            if front.is_empty() {
                self.chunks.pop_front();
            }
            return bytes;
        }
        // Straddles a chunk boundary: a copy is unavoidable.
        let mut out = Vec::with_capacity(len);
        let mut left = len;
        while left > 0 {
            let chunk = self.chunk();
            let take = min(left, chunk.len());
            out.extend_from_slice(&chunk[..take]);
            self.advance(take);
            left -= take;
        }
        Bytes::from(out)
    }
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
        // Fetching the remainder as concurrent chunks lifts the
        // single-connection throughput cap on large messages (e.g. manifests of
        // datasets with many fragments); decoding straight from the chunks
        // avoids re-copying the whole message into one buffer.
        let mut full = ChunkedBuf::default();
        full.push(buf.slice(4..));
        let mut chunks = read_range_in_chunks(reader, remaining_range, METADATA_READ_CHUNK_SIZE);
        while let Some(chunk) = chunks.try_next().await? {
            full.push(chunk);
        }
        if full.remaining() < msg_len {
            return Err(Error::io("file size is too small".to_string()));
        }
        full.truncate(msg_len);
        Ok(M::decode(full)?)
    } else {
        Ok(M::decode(buf.slice(4..4 + msg_len))?)
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
    use std::cmp::min;

    use bytes::{Buf, Bytes, BytesMut};
    use futures::TryStreamExt;
    use object_store::path::Path;
    use prost::Message;

    use super::{ChunkedBuf, read_message};

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

    #[test]
    fn chunked_buf_advances_and_copies_across_chunks() {
        let mut buf = ChunkedBuf::default();
        buf.push(Bytes::from_static(b"abc"));
        buf.push(Bytes::new());
        buf.push(Bytes::from_static(b"defgh"));
        buf.push(Bytes::from_static(b"ij"));
        assert_eq!(buf.remaining(), 10);
        assert_eq!(buf.chunk(), b"abc");

        // Within one chunk: zero-copy slice of that chunk.
        let first_chunk_ptr = buf.chunk().as_ptr();
        let head = buf.copy_to_bytes(2);
        assert_eq!(head.as_ref(), b"ab");
        assert_eq!(head.as_ptr(), first_chunk_ptr);

        // Straddling a boundary: copied, contents preserved, cursor advanced.
        let cross = buf.copy_to_bytes(4);
        assert_eq!(cross.as_ref(), b"cdef");
        assert_eq!(buf.remaining(), 4);
        assert_eq!(buf.chunk(), b"gh");

        buf.advance(3);
        assert_eq!(buf.chunk(), b"j");
        assert_eq!(buf.copy_to_bytes(1).as_ref(), b"j");
        assert_eq!(buf.remaining(), 0);
        assert_eq!(buf.chunk(), b"");
    }

    #[test]
    fn chunked_buf_truncate_drops_trailing_bytes() {
        let mut buf = ChunkedBuf::default();
        buf.push(Bytes::from_static(b"0123"));
        buf.push(Bytes::from_static(b"4567"));
        buf.push(Bytes::from_static(b"89"));
        buf.truncate(20);
        assert_eq!(buf.remaining(), 10);
        buf.truncate(5);
        assert_eq!(buf.remaining(), 5);
        assert_eq!(buf.copy_to_bytes(5).as_ref(), b"01234");
        assert_eq!(buf.remaining(), 0);
    }

    #[test]
    fn chunked_buf_decodes_message_split_across_chunks() {
        // `Bytes` implements `prost::Message` (one bytes field), which is
        // exactly the shape whose decoding should be zero-copy within a chunk.
        let payload: Vec<u8> = (0..20_000u32).map(|i| (i % 251) as u8).collect();
        let msg = Bytes::from(payload);
        let encoded = Bytes::from(msg.encode_to_vec());
        for chunk_size in [1usize, 7, 64, 4096, usize::MAX] {
            let mut buf = ChunkedBuf::default();
            for start in (0..encoded.len()).step_by(chunk_size) {
                buf.push(encoded.slice(start..min(start + chunk_size, encoded.len())));
            }
            // Trailing bytes (a footer) must be cut off before decoding.
            buf.push(Bytes::from_static(b"footer"));
            buf.truncate(encoded.len());
            let decoded = Bytes::decode(buf).unwrap();
            assert_eq!(decoded, msg, "chunk_size={chunk_size}");
        }
    }

    #[tokio::test]
    async fn read_message_decodes_body_larger_than_one_block_from_chunks() {
        // A message larger than the reader block size takes the chunked path.
        let store = ObjectStore::memory();
        let path = Path::from("/large_message");
        let payload = Bytes::from(
            (0..300_000u32)
                .map(|i| (i % 253) as u8)
                .collect::<Vec<u8>>(),
        );
        let mut writer = store.create(&path).await.unwrap();
        let pos = writer.write_protobuf(&payload).await.unwrap();
        writer.write_magics(pos, 0, 1, b"LANC").await.unwrap();
        writer.shutdown().await.unwrap();

        let reader = store.open(&path).await.unwrap();
        assert!(payload.len() > reader.block_size());
        let decoded: Bytes = read_message(reader.as_ref(), pos).await.unwrap();
        assert_eq!(decoded, payload);
    }
}
