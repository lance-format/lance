// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! General per-value decode with exact output and caller-owned codec context.
//!
//! Raw copies and Zstd initialize output once. The writer publishes only complete
//! output; partial decoder failures recycle its allocation into the same workspace.
//! Zstd reuses one admitted context within a decode; frames remain independent.

use arrow_buffer::Buffer;
use lance_core::{
    Error, Result,
    utils::allocation::{Reservation, ReservedBufferWriter},
};

use super::CompressionScheme;
use crate::{
    buffer::LanceBuffer,
    compression::{MiniBlockOutputLayout, PerValueDecodeBudget, variable},
    data::{BlockInfo, DataBlock, VariableWidthBlock},
};

fn context_bytes(scheme: CompressionScheme) -> Result<usize> {
    match scheme {
        CompressionScheme::None => Ok(0),
        #[cfg(feature = "lz4")]
        CompressionScheme::Lz4 => Ok(0),
        #[cfg(feature = "zstd")]
        CompressionScheme::Zstd => {
            // SAFETY: this pure size query has no pointer arguments or allocations.
            Ok(unsafe { ::zstd::zstd_safe::zstd_sys::ZSTD_estimateDCtxSize() })
        }
        _ => Err(Error::not_supported(
            "general per-value codec is not enabled",
        )),
    }
}

// Empty slots represent empty/null values and have no codec frame. Nonempty
// frames retain the current writer's framing; no unadmitted fallback is used.
fn frame(scheme: CompressionScheme, input: &[u8]) -> Result<(&[u8], usize)> {
    if input.is_empty() {
        return Ok((input, 0));
    }
    match scheme {
        CompressionScheme::None => Ok((input, input.len())),
        #[cfg(feature = "lz4")]
        CompressionScheme::Lz4 => {
            let prefix = input.get(..4).ok_or_else(|| {
                Error::invalid_input("LZ4 per-value frame is missing its length prefix")
            })?;
            let len = u32::from_le_bytes(prefix.try_into().unwrap()) as usize;
            super::admitted::validate_lz4(input, len)?;
            Ok((input, len))
        }
        #[cfg(feature = "zstd")]
        CompressionScheme::Zstd => {
            let raw = super::zstd::ZstdBufferCompressor::new(0).is_raw_stream_format(input);
            let (compressed, declared) = if raw {
                (input, None)
            } else {
                let prefix = input.get(..8).ok_or_else(|| {
                    Error::invalid_input("Zstd per-value frame is missing its length prefix")
                })?;
                (
                    &input[8..],
                    Some(u64::from_le_bytes(prefix.try_into().unwrap())),
                )
            };
            let frame_size = ::zstd::zstd_safe::find_frame_compressed_size(compressed)
                .map_err(|code| Error::invalid_input(::zstd::zstd_safe::get_error_name(code)))?;
            if frame_size != compressed.len() {
                return Err(Error::not_supported(
                    "admitted per-value Zstd requires one complete frame",
                ));
            }
            let content = ::zstd::zstd_safe::get_frame_content_size(compressed)
                .map_err(|_| Error::invalid_input("invalid Zstd content size"))?;
            if let (Some(declared), Some(content)) = (declared, content)
                && declared != content
            {
                return Err(Error::invalid_input(
                    "Zstd prefix disagrees with frame content size",
                ));
            }
            let len = declared.or(content).ok_or_else(|| {
                Error::not_supported("admitted raw Zstd needs a known frame content size")
            })?;
            let len = usize::try_from(len)
                .map_err(|_| Error::invalid_input("Zstd output length does not fit usize"))?;
            Ok((compressed, len))
        }
        _ => Err(Error::not_supported(
            "general per-value codec is not enabled",
        )),
    }
}

pub(super) fn plan(
    scheme: CompressionScheme,
    data: &VariableWidthBlock,
) -> Result<PerValueDecodeBudget> {
    let scratch = context_bytes(scheme)?;
    let mut payload = 0_usize;
    variable::visit(data, &mut |input| {
        payload = payload
            .checked_add(frame(scheme, input)?.1)
            .ok_or_else(|| Error::invalid_input("general per-value output size overflows"))?;
        Ok(())
    })?;
    let output_bytes = payload
        .checked_add(data.offsets.len())
        .ok_or_else(|| Error::invalid_input("general per-value allocation size overflows"))?;
    MiniBlockOutputLayout::VariableWidth {
        bits_per_offset: data.bits_per_offset,
    }
    .validate_range_bytes(data.num_values, output_bytes)?;
    let workspace_bytes = output_bytes
        .checked_add(scratch)
        .ok_or_else(|| Error::invalid_input("general per-value workspace size overflows"))?;
    Ok(PerValueDecodeBudget {
        output_bytes,
        workspace_bytes,
    })
}

#[cfg(feature = "zstd")]
fn zstd_append(
    input: &[u8],
    len: usize,
    output: &mut ReservedBufferWriter,
    context: &mut [u8],
    initialized: &mut bool,
) -> Result<()> {
    use ::zstd::zstd_safe::zstd_sys;
    // SAFETY: Reservation supplies exclusive, 64-byte-aligned storage, sized by
    // ZSTD_estimateDCtxSize. Input, output and context are disjoint. The pinned
    // ZSTD_decompressDCtx accepts uninitialized output (as in zstd-safe's WriteBuf)
    // and returns its initialized length. Never form an uninitialized u8 slice.
    // Only an exact successful length advances the writer; errors leave the
    // private suffix unreadable. Static DCtx never allocates or resizes. No
    // dictionary/streaming/legacy context is installed. decode_with supplies the
    // same allocation on every call and initialized is local to that decode.
    // ZSTD_decompressDCtx starts each frame with ZSTD_decompressBegin, which
    // clears previousDstEnd, prefixStart, virtualStart and dictionary/entropy
    // state before checking output continuity (pinned Zstd 1.5.7).
    unsafe {
        let ctx = if *initialized {
            context.as_mut_ptr().cast::<zstd_sys::ZSTD_DCtx>()
        } else {
            let ctx = zstd_sys::ZSTD_initStaticDCtx(context.as_mut_ptr().cast(), context.len());
            if ctx.is_null() {
                return Err(Error::internal("Zstd rejected the admitted static context"));
            }
            *initialized = true;
            ctx
        };
        output.try_extend_with(len, |target| {
            let written = zstd_sys::ZSTD_decompressDCtx(
                ctx,
                target.as_mut_ptr().cast(),
                target.len(),
                input.as_ptr().cast(),
                input.len(),
            );
            if zstd_sys::ZSTD_isError(written) != 0 {
                return Err(Error::invalid_input(::zstd::zstd_safe::get_error_name(
                    written,
                )));
            }
            Ok(written)
        })
    }
}

pub(super) fn decode(
    scheme: CompressionScheme,
    data: &VariableWidthBlock,
    workspace: &mut Reservation,
) -> Result<Option<DataBlock>> {
    match scheme {
        CompressionScheme::None => decode_with(scheme, data, workspace, |input, _, output, _| {
            output.extend_from_slice(input)
        }),
        #[cfg(feature = "lz4")]
        CompressionScheme::Lz4 => decode_with(scheme, data, workspace, |input, len, output, _| {
            let start = output.initialized_mut().len();
            output.extend_zeros(len)?;
            super::admitted::lz4_into(input, &mut output.initialized_mut()[start..])
        }),
        #[cfg(feature = "zstd")]
        CompressionScheme::Zstd => {
            let mut initialized = false;
            decode_with(scheme, data, workspace, |input, len, output, context| {
                zstd_append(input, len, output, context, &mut initialized)
            })
        }
        _ => Err(Error::not_supported(
            "general per-value codec is not enabled",
        )),
    }
}

fn decode_with<F>(
    scheme: CompressionScheme,
    data: &VariableWidthBlock,
    workspace: &mut Reservation,
    mut append: F,
) -> Result<Option<DataBlock>>
where
    F: FnMut(&[u8], usize, &mut ReservedBufferWriter, &mut [u8]) -> Result<()>,
{
    let budget = plan(scheme, data)?;
    if budget.workspace_bytes > workspace.len() {
        return Ok(None);
    }
    let mut offsets = workspace
        .try_allocate(data.offsets.len())?
        .ok_or_else(|| Error::internal("general offsets exceeded complete admission"))?;
    let mut values = match workspace.try_allocate_writer(budget.output_bytes - data.offsets.len()) {
        Ok(Some(value)) => value,
        other => {
            offsets.recycle(workspace)?;
            return other.map(|_| None);
        }
    };
    let mut context = match workspace.try_allocate(budget.workspace_bytes - budget.output_bytes) {
        Ok(Some(context)) => context,
        other => {
            offsets.recycle(workspace)?;
            values.recycle(workspace)?;
            return other.map(|_| None);
        }
    };
    let width = usize::from(data.bits_per_offset / 8);
    offsets.as_mut()[..width].fill(0);
    let mut row = 0;
    let mut written = 0;
    let result = variable::visit(data, &mut |input| {
        let (compressed, len) = frame(scheme, input)?;
        let end = written + len; // the immutable input was fully checked by plan
        if !input.is_empty() {
            append(compressed, len, &mut values, context.as_mut())?;
        }
        written = end;
        row += 1;
        offsets.as_mut()[row * width..(row + 1) * width]
            .copy_from_slice(&(written as u64).to_le_bytes()[..width]);
        Ok(())
    });
    context.recycle(workspace)?;
    if let Err(error) = result {
        offsets.recycle(workspace)?;
        values.recycle(workspace)?;
        return Err(error);
    }
    Ok(Some(DataBlock::VariableWidth(VariableWidthBlock {
        data: LanceBuffer::from(Buffer::from(values.finish()?)),
        offsets: LanceBuffer::from(Buffer::from(offsets.freeze())),
        bits_per_offset: data.bits_per_offset,
        num_values: data.num_values,
        block_info: BlockInfo::new(),
    })))
}

#[cfg(test)]
mod tests;
