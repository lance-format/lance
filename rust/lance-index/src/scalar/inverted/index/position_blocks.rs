// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::io::Write;

use super::*;

// Coalesce tiny encoded blocks without retaining an independently growing
// large allocation per term. Oversized blocks keep their own buffer.
const POSITION_CHUNK_BYTES: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(super) struct EncodedPositionBlocks {
    offsets: Vec<u32>,
    chunks: Vec<PositionChunk>,
    tail: Vec<u8>,
    byte_len: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PositionChunk {
    start: usize,
    bytes: Box<[u8]>,
}

impl EncodedPositionBlocks {
    pub(super) fn size(&self) -> usize {
        self.byte_len + self.tail.capacity() - self.tail.len()
            + self.offsets.capacity() * std::mem::size_of::<u32>()
            + self.chunks.capacity() * std::mem::size_of::<PositionChunk>()
    }

    pub(super) fn block(&self, index: usize) -> &[u8] {
        let start = self.offsets[index] as usize;
        let end = self
            .offsets
            .get(index + 1)
            .map_or(self.byte_len, |&x| x as usize);
        let tail_start = self.byte_len - self.tail.len();
        if start >= tail_start {
            &self.tail[start - tail_start..end - tail_start]
        } else {
            let chunk = &self.chunks[self.chunks.partition_point(|c| c.start <= start) - 1];
            &chunk.bytes[start - chunk.start..end - chunk.start]
        }
    }

    pub(super) fn push_encoded_block(&mut self, block: Vec<u8>) -> Result<usize> {
        let len = block.len();
        let offset = u32::try_from(self.byte_len).map_err(|_| {
            Error::index(format!(
                "position block offset {} exceeds the u32 format limit",
                self.byte_len
            ))
        })?;
        let byte_len = self.byte_len.checked_add(len).ok_or_else(|| {
            Error::index(format!(
                "position stream length {} + {len} overflows usize",
                self.byte_len
            ))
        })?;
        if !self.tail.is_empty() && len > POSITION_CHUNK_BYTES - self.tail.len() {
            self.chunks.push(PositionChunk {
                start: self.byte_len - self.tail.len(),
                bytes: std::mem::take(&mut self.tail).into_boxed_slice(),
            });
        }
        self.offsets.push(offset);
        if len >= POSITION_CHUNK_BYTES {
            self.chunks.push(PositionChunk {
                start: self.byte_len,
                bytes: block.into_boxed_slice(),
            });
        } else {
            if len > self.tail.capacity() - self.tail.len() {
                let capacity = (self.tail.len() + len)
                    .next_power_of_two()
                    .min(POSITION_CHUNK_BYTES);
                self.tail.reserve_exact(capacity - self.tail.len());
            }
            self.tail.extend_from_slice(&block);
        }
        self.byte_len = byte_len;
        Ok(len)
    }

    pub(super) fn append_to(
        self,
        bytes: &mut LargeBinaryBuilder,
        block_offsets: &mut ListBuilder<UInt32Builder>,
    ) -> Result<()> {
        block_offsets.values().append_slice(&self.offsets);
        for chunk in self.chunks {
            bytes.write_all(&chunk.bytes)?;
        }
        bytes.write_all(&self.tail)?;
        // Complete one logical value after all its chunks; no intermediate
        // contiguous copy of the term's complete position stream is needed.
        bytes.append_value([]);
        block_offsets.append(true);
        Ok(())
    }

    pub(super) fn into_columns(self) -> Result<Vec<ArrayRef>> {
        let mut bytes = LargeBinaryBuilder::with_capacity(1, self.byte_len);
        let mut offsets = ListBuilder::new(UInt32Builder::with_capacity(self.offsets.len()));
        self.append_to(&mut bytes, &mut offsets)?;
        Ok(vec![Arc::new(bytes.finish()), Arc::new(offsets.finish())])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_offsets_reject_overflow_before_mutation() {
        // Exercise the persisted offset limit without allocating several GiB.
        let mut blocks = EncodedPositionBlocks {
            byte_len: u32::MAX as usize + 1,
            ..Default::default()
        };
        let error = blocks.push_encoded_block(vec![1]).unwrap_err();
        assert!(matches!(error, Error::Index { .. }));
        assert!(
            error
                .to_string()
                .contains("4294967296 exceeds the u32 format limit")
        );
        assert!(blocks.offsets.is_empty());
        assert!(blocks.chunks.is_empty());
        assert!(blocks.tail.is_empty());
    }
}
