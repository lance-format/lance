// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bounded input traversal for independently framed u64 structural sequences.

use std::{future::Future, ops::Range};

use lance_core::{
    Error, Result,
    utils::allocation::{Reservation, ReservedBuffer},
};

use super::SequenceDecoder;
#[cfg(feature = "bitpacking")]
use crate::encodings::physical::bitpacking::out_of_line_payload_byte_lengths;
use crate::{
    compression::unsigned,
    format::metadata::{CompressiveEncoding, compressive_encoding::Compression},
};

#[derive(Clone, Debug)]
enum Leaf {
    Flat,
    #[cfg(feature = "bitpacking")]
    Bitpacked {
        width: u64,
        raw_tail: bool,
    },
}

/// Reads at most one encoded chunk at a time. Visitors must discard their partial
/// result on error/cancellation. This accounts input scratch, not visitor output
/// or transport/cache allocations. There is no secondary admission or credit wait.
#[derive(Clone, Debug)]
pub(crate) struct WindowSequence {
    leaf: Leaf,
    payload: Range<u64>,
    count: u64,
    base: Option<u64>,
}

impl WindowSequence {
    pub(crate) fn try_new(
        encoding: &CompressiveEncoding,
        count: u64,
        payload: Range<u64>,
    ) -> Result<Self> {
        // Reuse the current codec grammar/cardinality validation before planning I/O.
        SequenceDecoder::try_new(encoding, 64, count)?;
        if payload.start > payload.end {
            return Err(Error::corrupt_file_named(
                "unsigned sequence",
                "reversed payload range",
            ));
        }
        let (encoding, count, base) = if let Some(Compression::Delta(delta)) = &encoding.compression
        {
            (
                delta
                    .deltas
                    .as_deref()
                    .ok_or_else(|| Error::invalid_input("missing Delta child"))?,
                count
                    .checked_sub(1)
                    .ok_or_else(|| Error::invalid_input("empty Delta"))?,
                Some(delta.base),
            )
        } else {
            (encoding, count, None)
        };
        let bytes = payload.end - payload.start;
        let leaf = match &encoding.compression {
            Some(Compression::Flat(flat)) if flat.bits_per_value == 64 => {
                if count.checked_mul(8) != Some(bytes) {
                    return Err(Error::corrupt_file_named(
                        "unsigned sequence",
                        "Flat payload length disagrees with count",
                    ));
                }
                Leaf::Flat
            }
            #[cfg(feature = "bitpacking")]
            Some(Compression::OutOfLineBitpacking(bitpacked)) => {
                let Some(Compression::Flat(flat)) = bitpacked
                    .values
                    .as_deref()
                    .and_then(|c| c.compression.as_ref())
                else {
                    return Err(Error::not_supported(
                        "windowed bitpacking requires a Flat child",
                    ));
                };
                let width = flat.bits_per_value;
                if width == 0 || width >= 64 {
                    return Err(Error::invalid_input(
                        "windowed bitpacking requires a narrower positive width",
                    ));
                }
                let [raw, padded] = out_of_line_payload_byte_lengths(count, 64, width)?;
                if bytes != raw && bytes != padded {
                    return Err(Error::corrupt_file_named(
                        "unsigned sequence",
                        "bitpacked payload length disagrees with count",
                    ));
                }
                Leaf::Bitpacked {
                    width,
                    raw_tail: bytes == raw,
                }
            }
            _ => {
                return Err(Error::not_supported(
                    "windowed unsigned traversal supports Flat/Bitpacked and their Delta transform",
                ));
            }
        };
        Ok(Self {
            leaf,
            payload,
            count,
            base,
        })
    }

    fn window(&self, remaining: u64) -> (u64, usize) {
        if remaining == 0 {
            return (0, 0);
        }
        match self.leaf {
            Leaf::Flat => {
                let count = remaining.min(512);
                (count, count as usize * 8)
            }
            #[cfg(feature = "bitpacking")]
            Leaf::Bitpacked { width, raw_tail } => {
                let count = remaining.min(1024);
                let raw = count < 1024 && raw_tail;
                (
                    count,
                    if raw {
                        count as usize * 8
                    } else {
                        width as usize * 128
                    },
                )
            }
        }
    }

    pub(crate) fn scratch_bytes(&self) -> usize {
        self.window(self.count)
            .1
            .max(self.window(self.count % 1024).1)
    }

    pub(crate) async fn visit<F, Fut, V>(
        &self,
        work: &mut Reservation,
        read: &mut F,
        visitor: &mut V,
    ) -> Result<Option<()>>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
        V: FnMut(u64) -> Result<()>,
    {
        // Flat windows are <=4KiB; packed chunks are <8KiB. A raw tail can
        // be larger than its preceding packed chunk, so preflight both.
        if self.scratch_bytes() > work.len() {
            return Ok(None);
        }
        let mut value = self.base.unwrap_or(0);
        if self.base.is_some() {
            visitor(value)?;
        }
        let mut consume = |next| {
            let next = if self.base.is_some() {
                value = value.checked_add(next).ok_or_else(|| {
                    Error::corrupt_file_named("unsigned sequence", "Delta prefix sum overflows")
                })?;
                value
            } else {
                next
            };
            visitor(next)
        };
        let mut remaining = self.count;
        let mut position = self.payload.start;
        while remaining > 0 {
            let (count, bytes) = self.window(remaining);
            let scratch = work
                .try_allocate(bytes)?
                .ok_or_else(|| Error::internal("window scratch changed after preflight"))?;
            let end = position
                .checked_add(bytes as u64)
                .filter(|end| *end <= self.payload.end)
                .ok_or_else(|| {
                    Error::corrupt_file_named("unsigned sequence", "window exceeds payload")
                })?;
            let scratch = read(position..end, scratch).await?;
            if scratch.as_ref().len() != bytes {
                return Err(Error::corrupt_file_named(
                    "unsigned sequence",
                    "short window read",
                ));
            }
            match self.leaf {
                Leaf::Flat => unsigned::flat(scratch.as_ref(), 64, count, &mut consume)?,
                #[cfg(feature = "bitpacking")]
                Leaf::Bitpacked { width, raw_tail } => {
                    if count < 1024 && raw_tail {
                        unsigned::flat(scratch.as_ref(), 64, count, &mut consume)?;
                    } else {
                        unsigned::bitpacked(scratch.as_ref(), 64, width, count, &mut consume)?;
                    }
                }
            }
            scratch.recycle(work)?;
            position = end;
            remaining -= count;
        }
        if position != self.payload.end {
            return Err(Error::corrupt_file_named(
                "unsigned sequence",
                "unconsumed payload bytes",
            ));
        }
        Ok(Some(()))
    }
}

#[cfg(test)]
mod tests;
