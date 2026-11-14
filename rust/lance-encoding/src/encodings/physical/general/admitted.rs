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
    buffer::LanceBuffer, compression::MiniBlockDecompressor, data::DataBlock,
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

pub(super) fn decode(
    codec: &GeneralMiniBlockDecompressor,
    data: &[LanceBuffer],
    num_values: u64,
    range: Range<u64>,
    workspace: &mut Reservation,
) -> Result<Option<DataBlock>> {
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
        // Pure inner preflight sees decompressed bytes but cannot publish scratch.
        // Nested preparation and unaccounted codecs deliberately remain unsupported.
        let plan = codec
            .inner
            .plan_range(&prepared, num_values, range.clone())?;
        layout.validate_range_bytes(range.end - range.start, plan.output_bytes)?;
        if plan.workspace_bytes < plan.output_bytes {
            return Err(Error::internal("General inner plan omits retained output"));
        }
        if plan.workspace_bytes > workspace.len() {
            return Ok(None);
        }
        codec
            .inner
            .try_decompress_range_in(&prepared, num_values, range, workspace)
    })();
    drop(prepared);
    // Reject an inner codec that publishes aliases of preparation instead of its
    // admitted output. Drop that result before attempting to regain exclusivity.
    let result = if Arc::strong_count(&scratch) == 1 {
        result
    } else {
        drop(result);
        Err(Error::internal(
            "General inner decoder retained preparation scratch",
        ))
    };
    let scratch = Arc::try_unwrap(scratch).map_err(|_| {
        // A misbehaving custom codec may retain an external alias. Its owner still
        // holds the charge; never refund live bytes merely to restore headroom.
        Error::internal("General inner decoder retained an external scratch alias")
    })?;
    scratch.recycle(workspace)?;
    result
}

#[cfg(test)]
mod tests;
