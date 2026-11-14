// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Prepare the first buffer while keeping its exact charge through inner decode.

use std::{ops::Range, sync::Arc};

use arrow_buffer::Buffer;
use bytes::Bytes;
use lance_core::{
    Error, Result,
    utils::allocation::{Reservation, ReservedBuffer},
};

use super::GeneralMiniBlockDecompressor;
use crate::{
    buffer::LanceBuffer,
    compression::{MiniBlockDecompressor, PreparedMiniBlockRange, prepared::PreparedExecution},
    data::DataBlock,
    encodings::physical::block::GeneralBufferCompressor,
};

// A temporary read-only view cannot detach the charge. Keep another Arc so that
// after all views are dropped we can recover exclusive scratch for recycling.
struct ScratchOwner(Arc<ReservedBuffer>);

impl AsRef<[u8]> for ScratchOwner {
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref().as_ref()
    }
}

pub(super) fn prepare<'a>(
    codec: &'a GeneralMiniBlockDecompressor,
    data: &[LanceBuffer],
    num_values: u64,
    range: Range<u64>,
    workspace: &mut Reservation,
) -> Result<Option<PreparedMiniBlockRange<'a>>> {
    if data.len() != codec.num_buffers() || data.is_empty() {
        return Err(Error::invalid_input(format!(
            "General mini-block has {} buffers, expected {}",
            data.len(),
            codec.num_buffers()
        )));
    }
    if range.start > range.end || range.end > num_values {
        return Err(Error::invalid_input(
            "General mini-block selection exceeds value count",
        ));
    }
    let layout = codec.inner.admitted_output_layout()?;
    let decompressor = GeneralBufferCompressor::get_compressor(codec.compression)?;
    let expected = decompressor
        .decompressed_size_of_value(&data[0])
        .ok_or_else(|| Error::invalid_input("General mini-block has no readable decoded length"))?;
    let expected = usize::try_from(expected)
        .map_err(|_| Error::invalid_input("General scratch exceeds addressable size"))?;
    let Some(scratch) = decompressor.try_decompress_exact_in(&data[0], expected, workspace)? else {
        return Ok(None);
    };
    let scratch = Arc::new(scratch);
    let view = Bytes::from_owner(ScratchOwner(scratch.clone()));
    let mut prepared = data.to_vec();
    prepared[0] = LanceBuffer::from(Buffer::from(view));
    let result = (|| {
        // Keep pure inner preflight: nested preparation remains unsupported.
        let plan = PreparedMiniBlockRange::direct(
            codec.inner.as_ref(),
            &prepared,
            num_values,
            range.clone(),
        )?;
        layout.validate_range_bytes(range.end - range.start, plan.budget().output_bytes)?;
        Ok(plan)
    })();
    drop(prepared);
    match result {
        Ok(inner) => Ok(Some(PreparedMiniBlockRange::new(
            inner.budget(),
            Box::new(GeneralPrepared { inner, scratch }),
        ))),
        Err(error) => {
            recover_scratch(scratch, workspace)?;
            Err(error)
        }
    }
}

struct GeneralPrepared<'a> {
    inner: PreparedMiniBlockRange<'a>,
    scratch: Arc<ReservedBuffer>,
}

fn recover_scratch(scratch: Arc<ReservedBuffer>, workspace: &mut Reservation) -> Result<()> {
    let scratch = Arc::try_unwrap(scratch).map_err(|_| {
        // An external alias still holds the charge. Never refund live bytes.
        Error::internal("General inner decoder retained an external scratch alias")
    })?;
    scratch.recycle(workspace)
}

impl PreparedExecution for GeneralPrepared<'_> {
    fn decode(self: Box<Self>, workspace: &mut Reservation) -> Result<Option<DataBlock>> {
        let Self { inner, scratch } = *self;
        let result = inner.try_decode(workspace);
        // Consuming the inner plan drops its read-only views before checking
        // exclusivity. A codec cannot publish preparation as its admitted output.
        let result = if Arc::strong_count(&scratch) == 1 {
            result
        } else {
            drop(result);
            Err(Error::internal(
                "General inner decoder retained preparation scratch",
            ))
        };
        recover_scratch(scratch, workspace)?;
        result
    }

    fn recycle(self: Box<Self>, workspace: &mut Reservation) -> Result<()> {
        let Self { inner, scratch } = *self;
        let result = inner.recycle(workspace);
        recover_scratch(scratch, workspace)?;
        result
    }
}

#[cfg(test)]
mod tests;
