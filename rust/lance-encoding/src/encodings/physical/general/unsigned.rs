// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_buffer::Buffer;
use bytes::Bytes;
use lance_core::{
    Error, Result,
    utils::allocation::{Reservation, ReservedBuffer},
};

use super::{
    FixedWidthGeneralBlockDecompressor,
    admitted::{ScratchOwner, recover_scratch},
};
use crate::{
    buffer::LanceBuffer,
    compression::{PreparedUnsigned, prepared_unsigned::UnsignedExecution},
    encodings::physical::block::GeneralBufferCompressor,
};

pub(super) fn prepare<'a>(
    codec: &'a FixedWidthGeneralBlockDecompressor,
    data: Option<&LanceBuffer>,
    bits: u64,
    count: u64,
    workspace: &mut Reservation,
) -> Result<Option<PreparedUnsigned<'a>>> {
    if !matches!(bits, 32 | 64) || bits != codec.bits_per_value {
        return Err(Error::invalid_input(
            "General unsigned width disagrees with codec",
        ));
    }
    let data =
        data.ok_or_else(|| Error::corrupt_file_named("General unsigned", "missing payload"))?;
    // Match the ordinary fixed-width General reader's exact decoded size. This
    // supports the writer's General(Flat) candidate, not arbitrary nested frames.
    let bytes = count
        .checked_mul(bits / 8)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| Error::invalid_input("General unsigned scratch size overflows"))?;
    let compressor = GeneralBufferCompressor::get_compressor(codec.compression)?;
    let Some(scratch) = compressor.try_decompress_exact_in(data, bytes, workspace)? else {
        return Ok(None);
    };
    let scratch = Arc::new(scratch);
    let view = LanceBuffer::from(Buffer::from(Bytes::from_owner(ScratchOwner(
        scratch.clone(),
    ))));
    let inner = PreparedUnsigned::direct(codec.child.as_ref(), Some(&view), bits, count);
    Ok(Some(PreparedUnsigned::new(Box::new(GeneralUnsigned {
        inner,
        scratch,
    }))))
}

struct GeneralUnsigned<'a> {
    inner: PreparedUnsigned<'a>,
    scratch: Arc<ReservedBuffer>,
}

impl UnsignedExecution for GeneralUnsigned<'_> {
    fn visit(&self, visitor: &mut dyn FnMut(u64) -> Result<()>) -> Result<()> {
        self.inner.visit(visitor)
    }

    fn recycle(self: Box<Self>, workspace: &mut Reservation) -> Result<()> {
        let Self { inner, scratch } = *self;
        let result = inner.recycle(workspace);
        recover_scratch(scratch, workspace)?;
        result
    }
}
