// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Range;

use async_trait::async_trait;
use bytes::Bytes;
use futures::{StreamExt, future::BoxFuture, stream::BoxStream};
use lance_core::deepsize::DeepSizeOf;
use object_store::path::Path;
use prost::Message;
use tokio::io::{AsyncWrite, AsyncWriteExt};

use lance_core::{Error, Result, utils::allocation::ReservedBuffer};

use crate::object_writer::WriteResult;

pub trait ProtoStruct {
    type Proto: Message;
}

pub type ByteStream = BoxStream<'static, object_store::Result<Bytes>>;

/// A trait for writing to a file on local file system or object store.
#[async_trait]
pub trait Writer: AsyncWrite + Unpin + Send {
    /// Tell the current offset.
    async fn tell(&mut self) -> Result<usize>;

    /// Flush all buffered data and finalize the write, returning metadata about
    /// the written object.
    async fn shutdown(&mut self) -> Result<WriteResult>;
}

#[async_trait]
impl Writer for Box<dyn Writer> {
    async fn tell(&mut self) -> Result<usize> {
        self.as_mut().tell().await
    }

    async fn shutdown(&mut self) -> Result<WriteResult> {
        self.as_mut().shutdown().await
    }
}

/// Lance Write Extension.
#[async_trait]
pub trait WriteExt {
    /// Write a Protobuf message to the [Writer], and returns the file position
    /// where the protobuf is written.
    async fn write_protobuf(&mut self, msg: &impl Message) -> Result<usize>;

    async fn write_struct<
        'b,
        M: Message + From<&'b T>,
        T: ProtoStruct<Proto = M> + Send + Sync + 'b,
    >(
        &mut self,
        obj: &'b T,
    ) -> Result<usize> {
        let msg: M = M::from(obj);
        self.write_protobuf(&msg).await
    }
    /// Write magics to the tail of a file before closing the file.
    async fn write_magics(
        &mut self,
        pos: usize,
        major_version: i16,
        minor_version: i16,
        magic: &[u8],
    ) -> Result<()>;

    async fn copy_from_reader(&mut self, reader: &dyn Reader) -> Result<usize>;

    async fn copy_range_from_reader(
        &mut self,
        reader: &dyn Reader,
        range: Range<usize>,
    ) -> Result<usize>;
}

#[async_trait]
impl<W: Writer + ?Sized> WriteExt for W {
    async fn write_protobuf(&mut self, msg: &impl Message) -> Result<usize> {
        let offset = self.tell().await?;

        let buf = msg.encode_to_vec();

        self.write_u32_le(buf.len() as u32).await?;
        self.write_all(&buf).await?;

        Ok(offset)
    }

    async fn write_magics(
        &mut self,
        pos: usize,
        major_version: i16,
        minor_version: i16,
        magic: &[u8],
    ) -> Result<()> {
        self.write_i64_le(pos as i64).await?;
        self.write_i16_le(major_version).await?;
        self.write_i16_le(minor_version).await?;
        self.write_all(magic).await?;
        Ok(())
    }

    async fn copy_from_reader(&mut self, reader: &dyn Reader) -> Result<usize> {
        let mut stream = reader.get_stream().await?;
        let mut copied = 0usize;
        while let Some(chunk) = stream.next().await {
            let bytes = chunk?;
            copied += bytes.len();
            self.write_all(&bytes).await?;
        }
        Ok(copied)
    }

    async fn copy_range_from_reader(
        &mut self,
        reader: &dyn Reader,
        range: Range<usize>,
    ) -> Result<usize> {
        let mut stream = reader.get_range_stream(range).await?;
        let mut copied = 0usize;
        while let Some(chunk) = stream.next().await {
            let bytes = chunk?;
            copied += bytes.len();
            self.write_all(&bytes).await?;
        }
        Ok(copied)
    }
}

pub trait Reader: std::fmt::Debug + Send + Sync + DeepSizeOf {
    fn path(&self) -> &Path;

    /// Suggest optimal I/O size per storage device.
    fn block_size(&self) -> usize;

    /// Suggest optimal I/O parallelism per storage device.
    fn io_parallelism(&self) -> usize;

    /// Object/File Size.
    fn size(&self) -> BoxFuture<'_, object_store::Result<usize>>;

    /// Read a range of bytes from the object.
    ///
    /// TODO: change to read_at()?
    fn get_range(&self, range: Range<usize>) -> BoxFuture<'static, object_store::Result<Bytes>>;

    /// Fill an already admitted owned payload without detaching its credit.
    /// The default copies streamed transport chunks; transport allocations are not
    /// covered by the payload reservation. Readers with direct read-into support
    /// should override this to avoid the intermediate buffer. Cancellation must
    /// keep `output` alive until any underlying operation has stopped using it.
    fn get_range_into(
        &self,
        range: Range<usize>,
        mut output: ReservedBuffer,
    ) -> BoxFuture<'_, Result<ReservedBuffer>> {
        Box::pin(async move {
            if range.end.checked_sub(range.start) != Some(output.capacity()) {
                return Err(Error::invalid_input(
                    "read range does not match reserved payload",
                ));
            }
            if range.is_empty() {
                return Ok(output);
            }
            let mut body = self.get_range_stream(range.clone()).await?;
            let mut offset = 0;
            while let Some(chunk) = body.next().await {
                let chunk = chunk?;
                if chunk.len() > range.len() - offset {
                    return Err(Error::io(format!(
                        "body for {} range {range:?} exceeds requested length",
                        self.path()
                    )));
                }
                output.as_mut()[offset..offset + chunk.len()].copy_from_slice(&chunk);
                offset += chunk.len();
            }
            if offset != range.len() {
                return Err(Error::io(format!(
                    "body for {} range {range:?} has {offset} bytes, expected {}",
                    self.path(),
                    range.len()
                )));
            }
            Ok(output)
        })
    }

    /// Read all bytes from the object.
    ///
    /// By default this reads the size in a separate IOP but some implementations
    /// may not need the size beforehand.
    fn get_all(&self) -> BoxFuture<'_, object_store::Result<Bytes>>;

    /// Read the entire object as a byte stream.
    fn get_stream(&self) -> BoxFuture<'_, object_store::Result<ByteStream>> {
        Box::pin(async move {
            let bytes = self.get_all().await?;
            Ok(futures::stream::once(async move { Ok(bytes) }).boxed())
        })
    }

    /// Read a byte range as a byte stream.
    fn get_range_stream(
        &self,
        range: Range<usize>,
    ) -> BoxFuture<'_, object_store::Result<ByteStream>> {
        Box::pin(async move {
            let bytes = self.get_range(range).await?;
            Ok(futures::stream::once(async move { Ok(bytes) }).boxed())
        })
    }
}
