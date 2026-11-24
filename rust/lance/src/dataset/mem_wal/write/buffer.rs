// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bounded input buffering for a single atomic write.

use arrow_array::RecordBatch;
use lance_core::{Error, Result};

use super::ShardWriterConfig;

/// Buffers an atomic put or delete while enforcing input byte and row limits.
///
/// This bounds conversion of a reader before handing its batches to
/// [`super::ShardWriter`]. It does not write anything itself.
#[derive(Debug)]
pub struct WriteBatchBuffer {
    batches: Vec<RecordBatch>,
    rows: usize,
    bytes: usize,
    max_bytes: usize,
    max_rows: Option<usize>,
}

impl WriteBatchBuffer {
    /// Create an empty buffer with a positive byte budget and the writer's row cap.
    ///
    /// WAL-only writers have no row cap. Bytes count Arrow array memory and
    /// batch structures, conservatively counting shared buffers more than once.
    /// The producer must allocate each batch before it can be checked, so peak
    /// input memory can exceed `max_bytes` by one incoming batch.
    ///
    /// ```
    /// # use lance::Result;
    /// # use lance::dataset::mem_wal::write::{ShardWriterConfig, WriteBatchBuffer};
    /// # use arrow_array::RecordBatch;
    /// # fn example(config: &ShardWriterConfig, batch: RecordBatch) -> Result<()> {
    /// let mut buffer = WriteBatchBuffer::new(config, 64 * 1024 * 1024)?;
    /// buffer.push(batch)?;
    /// let batches = buffer.into_batches();
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(config: &ShardWriterConfig, max_bytes: usize) -> Result<Self> {
        if max_bytes == 0 {
            return Err(Error::invalid_input("max_bytes must be positive, got 0"));
        }
        Ok(Self {
            batches: Vec::new(),
            rows: 0,
            bytes: 0,
            max_bytes,
            max_rows: config.enable_memtable.then_some(config.max_memtable_rows),
        })
    }

    /// Retain a batch only if the resulting input fits both limits.
    ///
    /// A single oversized batch is rejected without splitting it. On error,
    /// the batch is dropped and the previously buffered input is unchanged.
    pub fn push(&mut self, batch: RecordBatch) -> Result<()> {
        let rows = self
            .rows
            .checked_add(batch.num_rows())
            .ok_or_else(|| Error::invalid_input("write buffer row count exceeds usize::MAX"))?;
        if let Some(max_rows) = self.max_rows
            && rows > max_rows
        {
            return Err(Error::invalid_input(format!(
                "write buffer of {rows} rows exceeds max_memtable_rows={max_rows}: \
                 split the write, or raise max_memtable_rows"
            )));
        }
        let bytes = self
            .bytes
            .checked_add(batch.get_array_memory_size())
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<RecordBatch>()))
            .ok_or_else(|| Error::invalid_input("write buffer byte count exceeds usize::MAX"))?;
        if bytes > self.max_bytes {
            return Err(Error::invalid_input(format!(
                "write buffer of {bytes} bytes exceeds max_bytes={}: \
                 split the write, or raise max_bytes",
                self.max_bytes
            )));
        }
        self.rows = rows;
        self.bytes = bytes;
        self.batches.push(batch);
        Ok(())
    }

    /// Finish buffering without writing; pass these batches to put or delete.
    pub fn into_batches(self) -> Vec<RecordBatch> {
        self.batches
    }
}
