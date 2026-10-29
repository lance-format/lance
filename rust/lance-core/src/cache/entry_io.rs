// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Streaming readers/writers for cache entry bodies.
//!
//! [`CacheCodecImpl`](super::CacheCodecImpl) bodies are written and read
//! through these wrappers. They keep serialization streaming (no buffering of
//! the whole entry) and reads zero-copy (sections borrow from the input
//! [`Bytes`]), while tracking the byte position needed to keep Arrow IPC
//! sections 64-byte aligned (see [`lance_arrow::ipc`]).
//!
//! Body layout primitives:
//!
//! ```text
//! HEADER    : [header_len: u32 LE][header proto bytes]
//! ARROW_IPC : [pad to 64B][self-delimiting IPC stream]
//! RAW_BLOB  : [len: u64 LE][bytes]
//! ```

use std::io::Write;

use arrow_array::RecordBatch;
use bytes::Bytes;
use prost::Message;

use crate::{Error, Result};

/// Writes a cache entry body: a header followed by sections, streaming
/// directly to the underlying writer.
///
/// The envelope is written by the [`CacheCodec`](super::CacheCodec) wrapper
/// before this writer is handed to
/// [`CacheCodecImpl::serialize`](super::CacheCodecImpl::serialize).
pub struct CacheEntryWriter<'a> {
    writer: &'a mut dyn Write,
    /// Absolute byte offset within the entry, used to align IPC sections.
    pos: usize,
}

impl<'a> CacheEntryWriter<'a> {
    /// Create a writer positioned at the start of an entry (offset 0).
    ///
    /// Use this for nested serialization into a standalone buffer. The
    /// envelope-aware entry point is [`CacheCodec::serialize`](super::CacheCodec::serialize).
    pub fn new(writer: &'a mut dyn Write) -> Self {
        Self { writer, pos: 0 }
    }

    /// Create a writer whose section alignment accounts for `pos` bytes
    /// already written ahead of the body (i.e. the envelope).
    pub(crate) fn with_pos(writer: &'a mut dyn Write, pos: usize) -> Self {
        Self { writer, pos }
    }

    /// Write a single discriminant byte (e.g. a variant tag).
    pub fn write_u8(&mut self, value: u8) -> Result<()> {
        self.writer.write_all(&[value])?;
        self.pos += 1;
        Ok(())
    }

    /// Write a protobuf header as `[len: u32 LE][bytes]`.
    pub fn write_header<P: Message>(&mut self, header: &P) -> Result<()> {
        let bytes = header.encode_to_vec();
        let len = u32::try_from(bytes.len())
            .map_err(|_| Error::io(format!("cache header too large: {} bytes", bytes.len())))?;
        self.writer.write_all(&len.to_le_bytes())?;
        self.writer.write_all(&bytes)?;
        self.pos += 4 + bytes.len();
        Ok(())
    }

    /// Write `batch` as a 64-byte-aligned Arrow IPC section.
    pub fn write_ipc(&mut self, batch: &RecordBatch) -> Result<()> {
        lance_arrow::ipc::write_ipc_section(self.writer, &mut self.pos, batch)
            .map_err(|e| Error::io(e.to_string()))
    }

    /// Write `batches` as a single 64-byte-aligned multi-batch Arrow IPC
    /// section. The iterator must yield at least one batch.
    pub fn write_ipc_batches<I>(&mut self, batches: I) -> Result<()>
    where
        I: IntoIterator<Item = RecordBatch>,
    {
        lance_arrow::ipc::write_ipc_section_batches(self.writer, &mut self.pos, batches)
            .map_err(|e| Error::io(e.to_string()))
    }

    /// Write a raw blob as `[len: u64 LE][bytes]`.
    ///
    /// Only for byte payloads that already have their own stable, portable
    /// encoding (e.g. a roaring bitmap, a varint-packed stream).
    pub fn write_raw(&mut self, bytes: &[u8]) -> Result<()> {
        lance_arrow::ipc::write_len_prefixed_bytes(self.writer, bytes)
            .map_err(|e| Error::io(e.to_string()))?;
        self.pos += 8 + bytes.len();
        Ok(())
    }

    /// The underlying writer, for a payload that carries its own framing.
    ///
    /// Use this only when the codec writes a self-delimiting or whole-body
    /// payload — e.g. streaming a roaring bitmap as the entire body, where the
    /// length prefix of [`write_raw`](Self::write_raw) would be redundant and
    /// buffering to measure that length would force an extra copy. For
    /// structured bodies prefer [`write_header`](Self::write_header) /
    /// [`write_ipc`](Self::write_ipc) / [`write_raw`](Self::write_raw), which
    /// give you versioning and 64-byte IPC alignment.
    ///
    /// Bytes written through this do **not** advance the section-alignment
    /// position, so it must not be interleaved with [`write_ipc`](Self::write_ipc).
    pub fn raw_writer(&mut self) -> &mut dyn Write {
        self.writer
    }
}

/// Reads a cache entry body, tracking an offset into the input and exposing
/// the entry's `type_version` so implementors can branch for backward compat.
///
/// All reads are zero-copy: returned [`Bytes`] and the buffers behind decoded
/// [`RecordBatch`]es borrow from the input allocation.
pub struct CacheEntryReader<'a> {
    data: &'a Bytes,
    offset: usize,
    version: u32,
}

impl<'a> CacheEntryReader<'a> {
    /// Create a reader over `data`, starting at body byte `offset`, for an
    /// entry written at `version`.
    pub fn new(data: &'a Bytes, offset: usize, version: u32) -> Self {
        Self {
            data,
            offset,
            version,
        }
    }

    /// The `type_version` from the envelope. Branch on this for backward compat.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Read a single discriminant byte written by [`CacheEntryWriter::write_u8`].
    pub fn read_u8(&mut self) -> Result<u8> {
        let bytes = self.data.as_ref();
        let v = *bytes
            .get(self.offset)
            .ok_or_else(|| Error::io("cache entry: truncated, missing tag byte".to_string()))?;
        self.offset += 1;
        Ok(v)
    }

    /// Read a protobuf header written by [`CacheEntryWriter::write_header`].
    pub fn read_header<P: Message + Default>(&mut self) -> Result<P> {
        let bytes = self.data.as_ref();
        let len_end = self
            .offset
            .checked_add(4)
            .filter(|&e| e <= bytes.len())
            .ok_or_else(|| Error::io("cache header: truncated length prefix".to_string()))?;
        let len = u32::from_le_bytes(bytes[self.offset..len_end].try_into().unwrap()) as usize;
        let data_end = len_end
            .checked_add(len)
            .filter(|&e| e <= bytes.len())
            .ok_or_else(|| Error::io("cache header: truncated body".to_string()))?;
        let msg = P::decode(&bytes[len_end..data_end])
            .map_err(|e| Error::io(format!("cache header decode failed: {e}")))?;
        self.offset = data_end;
        Ok(msg)
    }

    /// Read one [`RecordBatch`] from a 64-byte-aligned IPC section.
    pub fn read_ipc(&mut self) -> Result<RecordBatch> {
        lance_arrow::ipc::read_ipc_section_at(self.data, &mut self.offset)
            .map_err(|e| Error::io(e.to_string()))
    }

    /// Read all [`RecordBatch`]es from a 64-byte-aligned multi-batch IPC
    /// section written by [`CacheEntryWriter::write_ipc_batches`].
    pub fn read_ipc_batches(&mut self) -> Result<Vec<RecordBatch>> {
        lance_arrow::ipc::read_ipc_section_batches_at(self.data, &mut self.offset)
            .map_err(|e| Error::io(e.to_string()))
    }

    /// Read a raw blob written by [`CacheEntryWriter::write_raw`], zero-copy.
    pub fn read_raw(&mut self) -> Result<Bytes> {
        lance_arrow::ipc::read_len_prefixed_bytes_at(self.data, &mut self.offset)
            .map_err(|e| Error::io(e.to_string()))
    }

    /// The not-yet-consumed body bytes as a zero-copy slice.
    ///
    /// For a payload that carries its own framing and is parsed with the
    /// codec's own cursor — the read counterpart of
    /// [`CacheEntryWriter::raw_writer`]. For structured bodies prefer
    /// [`read_header`](Self::read_header) / [`read_ipc`](Self::read_ipc) /
    /// [`read_raw`](Self::read_raw).
    pub fn body(&self) -> Bytes {
        self.data.slice(self.offset..)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow_array::{Int32Array, UInt64Array};
    use arrow_schema::{DataType, Field, Schema};
    use lance_arrow::ipc::IPC_SECTION_ALIGNMENT;

    /// Write a body starting at entry offset `pos` and return the bytes.
    ///
    /// `pos` models the envelope the [`CacheCodec`](super::CacheCodec) wrapper
    /// writes ahead of the body; it only affects section alignment.
    fn write_body(pos: usize, f: impl FnOnce(&mut CacheEntryWriter<'_>)) -> Bytes {
        let mut buf = Vec::new();
        let mut writer = CacheEntryWriter::with_pos(&mut buf, pos);
        f(&mut writer);
        Bytes::from(buf)
    }

    fn int_batch(values: Vec<i32>) -> RecordBatch {
        let schema = Schema::new(vec![Field::new("i", DataType::Int32, false)]);
        RecordBatch::try_new(schema.into(), vec![Arc::new(Int32Array::from(values))]).unwrap()
    }

    #[test]
    fn test_u8_roundtrip_and_truncation() {
        let data = write_body(0, |w| {
            w.write_u8(7).unwrap();
            w.write_u8(255).unwrap();
        });
        assert_eq!(data.as_ref(), &[7, 255]);

        let mut reader = CacheEntryReader::new(&data, 0, 1);
        assert_eq!(reader.read_u8().unwrap(), 7);
        assert_eq!(reader.read_u8().unwrap(), 255);

        // A third read has nothing left and must say so rather than wrap around.
        let message = reader.read_u8().unwrap_err().to_string();
        assert!(message.contains("missing tag byte"), "{message}");
    }

    /// Headers are framed as `[len: u32 LE][bytes]`; `u64` stands in for a real
    /// header proto here (prost encodes it as `google.protobuf.UInt64Value`).
    #[test]
    fn test_header_roundtrip_is_length_prefixed() {
        let data = write_body(0, |w| w.write_header(&1234u64).unwrap());

        let encoded_len = 1234u64.encoded_len();
        assert_eq!(
            u32::from_le_bytes(data[..4].try_into().unwrap()) as usize,
            encoded_len
        );
        assert_eq!(data.len(), 4 + encoded_len);

        let mut reader = CacheEntryReader::new(&data, 0, 1);
        assert_eq!(reader.read_header::<u64>().unwrap(), 1234);
        // The reader consumed exactly the prefix plus the payload.
        assert!(reader.body().is_empty());
    }

    #[test]
    fn test_read_header_rejects_truncated_length_prefix() {
        let data = Bytes::from_static(&[0, 0]);
        let message = CacheEntryReader::new(&data, 0, 1)
            .read_header::<u64>()
            .unwrap_err()
            .to_string();
        assert!(message.contains("truncated length prefix"), "{message}");
    }

    #[test]
    fn test_read_header_rejects_truncated_body() {
        // Prefix claims 16 payload bytes; only 3 follow.
        let mut data = 16u32.to_le_bytes().to_vec();
        data.extend_from_slice(&[1, 2, 3]);
        let data = Bytes::from(data);

        let message = CacheEntryReader::new(&data, 0, 1)
            .read_header::<u64>()
            .unwrap_err()
            .to_string();
        assert!(message.contains("truncated body"), "{message}");
    }

    /// A length prefix that is in range but whose payload is not valid protobuf
    /// must surface as a decode error, not a panic inside prost.
    #[test]
    fn test_read_header_rejects_undecodable_payload() {
        // Field 1 tagged as a varint, then a varint that never terminates.
        let payload = [0x08u8, 0xFF, 0xFF, 0xFF];
        let mut data = (payload.len() as u32).to_le_bytes().to_vec();
        data.extend_from_slice(&payload);
        let data = Bytes::from(data);

        let message = CacheEntryReader::new(&data, 0, 1)
            .read_header::<u64>()
            .unwrap_err()
            .to_string();
        assert!(message.contains("decode failed"), "{message}");
    }

    #[test]
    fn test_raw_roundtrip_leaves_the_rest_as_body() {
        let data = write_body(0, |w| {
            w.write_raw(&[1, 2, 3]).unwrap();
            w.raw_writer().write_all(&[9, 9]).unwrap();
        });
        // 8-byte length prefix + 3 payload + 2 trailing.
        assert_eq!(data.len(), 13);

        let mut reader = CacheEntryReader::new(&data, 0, 1);
        assert_eq!(reader.read_raw().unwrap().as_ref(), &[1, 2, 3]);
        assert_eq!(reader.body().as_ref(), &[9, 9]);
    }

    #[test]
    fn test_reader_exposes_the_entry_version() {
        let data = Bytes::from_static(&[0]);
        assert_eq!(CacheEntryReader::new(&data, 0, 7).version(), 7);
    }

    /// The reason `pos` is tracked at all: an IPC section must begin on a
    /// 64-byte boundary *of the whole entry*, so the envelope bytes ahead of the
    /// body count toward the padding. Writer and reader have to agree on that,
    /// and only a non-multiple-of-64 prefix makes a disagreement visible.
    #[test]
    fn test_ipc_section_is_aligned_against_the_envelope() {
        const ENVELOPE: usize = 13;
        let batch = int_batch(vec![1, 2, 3]);

        let mut buf = vec![0xAAu8; ENVELOPE];
        let mut writer = CacheEntryWriter::with_pos(&mut buf, ENVELOPE);
        writer.write_header(&1234u64).unwrap();
        writer.write_ipc(&batch).unwrap();
        let data = Bytes::from(buf);

        let header_end = ENVELOPE + 4 + 1234u64.encoded_len();
        let stream_start = header_end.next_multiple_of(IPC_SECTION_ALIGNMENT);
        assert!(stream_start > header_end, "padding should be non-empty");
        assert!(
            data[header_end..stream_start].iter().all(|b| *b == 0),
            "the gap must be zero padding"
        );

        let mut reader = CacheEntryReader::new(&data, ENVELOPE, 1);
        assert_eq!(reader.read_header::<u64>().unwrap(), 1234);
        assert_eq!(reader.read_ipc().unwrap(), batch);
        assert!(reader.body().is_empty());
    }

    #[test]
    fn test_ipc_batches_roundtrip() {
        let batches = vec![int_batch(vec![1, 2]), int_batch(vec![3])];
        let data = write_body(0, |w| w.write_ipc_batches(batches.clone()).unwrap());

        let mut reader = CacheEntryReader::new(&data, 0, 1);
        assert_eq!(reader.read_ipc_batches().unwrap(), batches);
    }

    /// The shape a real codec writes: a discriminant, a header, an arrow section
    /// and a blob. Each reader step has to pick up exactly where the previous one
    /// stopped, and the arrow section still has to land on its boundary even
    /// though a `write_u8` moved the position by one.
    #[test]
    fn test_mixed_sections_stay_in_sync() {
        let schema = Schema::new(vec![Field::new("u", DataType::UInt64, false)]);
        let batch = RecordBatch::try_new(
            schema.into(),
            vec![Arc::new(UInt64Array::from(vec![u64::MAX, 0]))],
        )
        .unwrap();

        let data = write_body(0, |w| {
            w.write_u8(2).unwrap();
            w.write_header(&99u64).unwrap();
            w.write_ipc(&batch).unwrap();
            w.write_raw(&[7, 7, 7]).unwrap();
        });

        let mut reader = CacheEntryReader::new(&data, 0, 1);
        assert_eq!(reader.read_u8().unwrap(), 2);
        assert_eq!(reader.read_header::<u64>().unwrap(), 99);
        assert_eq!(reader.read_ipc().unwrap(), batch);
        assert_eq!(reader.read_raw().unwrap().as_ref(), &[7, 7, 7]);
        assert!(reader.body().is_empty());
    }
}
