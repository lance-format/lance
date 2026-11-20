// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

/// Incremental traversal for merging structural sequences. The caller preflights
/// the sum of live cursors' `scratch_bytes` and retained output in one workspace.
/// Fully draining returns scratch credit; an error/cancel discards the attempt.
pub struct WindowCursor<'a> {
    sequence: &'a WindowSequence,
    index: u64,
    byte_position: u64,
    scratch: Option<ReservedBuffer>,
    chunk_count: u64,
    chunk_index: u64,
    base_pending: bool,
    value: u64,
}

impl<'a> WindowCursor<'a> {
    pub(crate) fn new(sequence: &'a WindowSequence) -> Self {
        Self {
            sequence,
            index: 0,
            byte_position: sequence.payload.start,
            scratch: None,
            chunk_count: 0,
            chunk_index: 0,
            base_pending: sequence.base.is_some(),
            value: sequence.base.unwrap_or(0),
        }
    }

    pub(crate) async fn next<F, Fut>(
        &mut self,
        work: &mut Reservation,
        read: &mut F,
    ) -> Result<Option<u64>>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        if self.base_pending {
            self.base_pending = false;
            return Ok(Some(self.value));
        }
        if self.index == self.sequence.count {
            if let Some(scratch) = self.scratch.take() {
                scratch.recycle(work)?;
            }
            if self.byte_position != self.sequence.payload.end {
                return Err(Error::corrupt_file_named(
                    "unsigned cursor",
                    "unconsumed payload bytes",
                ));
            }
            return Ok(None);
        }
        let next = if let Leaf::Range { start, step } = self.sequence.leaf {
            if self.index > 0 && self.index.is_multiple_of(1024) {
                tokio::task::yield_now().await;
            }
            step.checked_mul(self.index)
                .and_then(|v| start.checked_add(v))
                .ok_or_else(|| {
                    Error::corrupt_file_named("unsigned cursor", "Range value overflows")
                })?
        } else {
            if self.scratch.is_none() || self.chunk_index == self.chunk_count {
                if let Some(scratch) = self.scratch.take() {
                    scratch.recycle(work)?;
                }
                let (count, bytes) = self.sequence.window(self.sequence.count - self.index);
                let scratch = work
                    .try_allocate(bytes)?
                    .ok_or_else(|| Error::internal("cursor scratch was not preflighted"))?;
                let end = self
                    .byte_position
                    .checked_add(bytes as u64)
                    .filter(|end| *end <= self.sequence.payload.end)
                    .ok_or_else(|| {
                        Error::corrupt_file_named("unsigned cursor", "window exceeds payload")
                    })?;
                let scratch = read(self.byte_position..end, scratch).await?;
                if scratch.as_ref().len() != bytes {
                    return Err(Error::corrupt_file_named(
                        "unsigned cursor",
                        "short window read",
                    ));
                }
                self.byte_position = end;
                self.chunk_count = count;
                self.chunk_index = 0;
                self.scratch = Some(scratch);
            }
            let scratch = self
                .scratch
                .as_ref()
                .ok_or_else(|| Error::internal("missing cursor input"))?;
            let next = match self.sequence.leaf {
                Leaf::Flat => unsigned::flat_at(
                    scratch.as_ref(),
                    self.sequence.bits,
                    self.chunk_count,
                    self.chunk_index,
                )?,
                #[cfg(feature = "bitpacking")]
                Leaf::Bitpacked { width, raw_tail } => {
                    if self.chunk_count < 1024 && raw_tail {
                        unsigned::flat_at(
                            scratch.as_ref(),
                            self.sequence.bits,
                            self.chunk_count,
                            self.chunk_index,
                        )?
                    } else {
                        unsigned::bitpacked_at(
                            scratch.as_ref(),
                            self.sequence.bits,
                            width,
                            self.chunk_count,
                            self.chunk_index,
                        )?
                    }
                }
                Leaf::Range { .. } => {
                    return Err(Error::internal("metadata cursor unexpectedly read payload"));
                }
            };
            self.chunk_index += 1;
            next
        };
        self.index += 1;
        if self.sequence.base.is_some() {
            self.value = self
                .value
                .checked_add(next)
                .filter(|v| self.sequence.bits == 64 || *v <= u64::from(u32::MAX))
                .ok_or_else(|| {
                    Error::corrupt_file_named("unsigned cursor", "Delta prefix sum overflows")
                })?;
            Ok(Some(self.value))
        } else {
            Ok(Some(next))
        }
    }
}
