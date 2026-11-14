// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Preflight and owned output for dense u16 structural-level decompression.

use lance_core::{
    Error, Result,
    utils::allocation::{Reservation, ReservedBuffer},
};

use crate::buffer::LanceBuffer;

#[cfg(feature = "bitpacking")]
use crate::encodings::physical::bitpacking::{ELEMS_PER_CHUNK, out_of_line_payload_byte_lengths};
#[cfg(feature = "bitpacking")]
use lance_bitpacking::BitPacking;

#[derive(Debug)]
enum Layout {
    Flat,
    #[cfg(feature = "bitpacking")]
    Bitpacked {
        width: usize,
        chunk_bytes: usize,
        has_raw_tail: bool,
        alignment_bytes: usize,
        tail_bytes: usize,
    },
}

/// A validated borrowed structural stream with a complete payload workspace size.
/// Input owners are outside this amount and must already be charged. Decoded
/// output keeps its reservation; alignment/tail scratch returns to the same
/// workspace before the method returns. Metadata and allocator overhead are
/// excluded. Planning never decompresses or allocates payload buffers.
///
/// ```
/// use lance_core::utils::allocation::AllocationBudget;
/// use lance_encoding::{buffer::LanceBuffer, compression::BlockDecompressor,
///     encodings::physical::value::ValueDecompressor, format::metadata};
/// let codec = ValueDecompressor::from_flat(&metadata::Flat {
///     bits_per_value: 16,
///     data: None,
/// });
/// let budget = AllocationBudget::new(12);
/// let mut work = budget.try_reserve(12).unwrap();
/// let mut input = work.try_allocate(6).unwrap().unwrap();
/// input.as_mut().copy_from_slice(bytemuck::cast_slice(&[1_u16, 0, 1]));
/// let input = LanceBuffer::from(arrow_buffer::Buffer::from(input.freeze()));
/// let plan = codec.plan_levels(&input, 3).unwrap();
/// let output = plan.try_decode_in(&mut work).unwrap().unwrap();
/// assert_eq!(bytemuck::cast_slice::<u8, u16>(output.as_ref()), &[1, 0, 1]);
/// drop(input);
/// drop(work);
/// assert_eq!(budget.reserved(), 6);
/// drop(output);
/// assert_eq!(budget.reserved(), 0);
/// ```
#[derive(Debug)]
pub struct LevelDecodePlan<'a> {
    data: &'a LanceBuffer,
    output_bytes: usize,
    workspace_bytes: usize,
    layout: Layout,
}

impl<'a> LevelDecodePlan<'a> {
    fn output_layout(num_values: u64) -> Result<usize> {
        let bytes = num_values
            .checked_mul(2)
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| Error::invalid_input("structural level output length overflows"))?;
        std::alloc::Layout::from_size_align(bytes, arrow_buffer::alloc::ALIGNMENT).map_err(
            |_| Error::invalid_input("structural level output exceeds addressable allocation size"),
        )?;
        Ok(bytes)
    }

    pub(crate) fn flat(data: &'a LanceBuffer, num_values: u64) -> Result<Self> {
        let output_bytes = Self::output_layout(num_values)?;
        if data.len() != output_bytes {
            return Err(Error::invalid_input(format!(
                "flat structural stream has {} bytes, expected {output_bytes}",
                data.len()
            )));
        }
        Ok(Self {
            data,
            output_bytes,
            workspace_bytes: output_bytes,
            layout: Layout::Flat,
        })
    }

    #[cfg(feature = "bitpacking")]
    pub(crate) fn bitpacked(data: &'a LanceBuffer, num_values: u64, width: u64) -> Result<Self> {
        let output_bytes = Self::output_layout(num_values)?;
        let [raw, padded] = out_of_line_payload_byte_lengths(num_values, 16, width)?;
        if data.len() as u64 != raw && data.len() as u64 != padded {
            return Err(Error::corrupt_file_named(
                "out-of-line bitpacking",
                format!(
                    "structural payload has {} bytes, expected {raw} raw-tail or {padded} padded-tail bytes",
                    data.len()
                ),
            ));
        }
        let tail = num_values % ELEMS_PER_CHUNK;
        // Match the ordinary reader when the raw and padded sizes are equal.
        let has_raw_tail = tail > 0 && data.len() as u64 == raw;
        let has_packed_tail = tail > 0 && !has_raw_tail;
        let chunk_bytes = (ELEMS_PER_CHUNK * width / 8) as usize;
        let alignment_bytes = if !data.as_ptr().addr().is_multiple_of(2)
            && (num_values >= ELEMS_PER_CHUNK || has_packed_tail)
        {
            chunk_bytes
        } else {
            0
        };
        let tail_bytes = if has_packed_tail {
            ELEMS_PER_CHUNK as usize * 2
        } else {
            0
        };
        let workspace_bytes = output_bytes
            .checked_add(alignment_bytes)
            .and_then(|v| v.checked_add(tail_bytes))
            .ok_or_else(|| Error::invalid_input("structural level workspace overflows"))?;
        Ok(Self {
            data,
            output_bytes,
            workspace_bytes,
            layout: Layout::Bitpacked {
                width: width as usize,
                chunk_bytes,
                has_raw_tail,
                alignment_bytes,
                tail_bytes,
            },
        })
    }

    /// Retained decoded payload after temporary scratch has been recycled.
    pub fn output_bytes(&self) -> usize {
        self.output_bytes
    }

    /// Peak additional credit needed while the borrowed encoded input remains live.
    pub fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    /// Decode directly into admitted output. `None` preserves input and workspace.
    /// No ordinary decoder or untracked alignment allocation is used.
    pub fn try_decode_in(&self, workspace: &mut Reservation) -> Result<Option<ReservedBuffer>> {
        if self.workspace_bytes > workspace.len() {
            return Ok(None);
        }
        let mut output = allocate(workspace, self.output_bytes)?;
        match self.layout {
            Layout::Flat => output.as_mut().copy_from_slice(self.data),
            #[cfg(feature = "bitpacking")]
            Layout::Bitpacked {
                width,
                chunk_bytes,
                has_raw_tail,
                alignment_bytes,
                tail_bytes,
            } => {
                let mut alignment = if alignment_bytes > 0 {
                    Some(allocate(workspace, alignment_bytes)?)
                } else {
                    None
                };
                let mut tail = if tail_bytes > 0 {
                    Some(allocate(workspace, tail_bytes)?)
                } else {
                    None
                };
                let values: &mut [u16] = bytemuck::cast_slice_mut(output.as_mut());
                let (full, remaining) = values.as_chunks_mut::<1024>();
                for (i, destination) in full.iter_mut().enumerate() {
                    unpack(
                        width,
                        &self.data[i * chunk_bytes..(i + 1) * chunk_bytes],
                        destination,
                        alignment.as_mut(),
                    );
                }
                let start = full.len() * chunk_bytes;
                if !remaining.is_empty() {
                    if has_raw_tail {
                        bytemuck::cast_slice_mut(remaining).copy_from_slice(&self.data[start..]);
                    } else {
                        let scratch = tail.as_mut().expect("packed tail scratch was preflighted");
                        let decoded: &mut [u16] = bytemuck::cast_slice_mut(scratch.as_mut());
                        unpack(width, &self.data[start..], decoded, alignment.as_mut());
                        remaining.copy_from_slice(&decoded[..remaining.len()]);
                    }
                }
                if let Some(buffer) = alignment {
                    buffer.recycle(workspace)?;
                }
                if let Some(buffer) = tail {
                    buffer.recycle(workspace)?;
                }
            }
        }
        Ok(Some(output))
    }
}

fn allocate(workspace: &mut Reservation, len: usize) -> Result<ReservedBuffer> {
    workspace
        .try_allocate(len)?
        .ok_or_else(|| Error::internal("level workspace changed after complete admission"))
}

#[cfg(feature = "bitpacking")]
fn unpack(width: usize, input: &[u8], output: &mut [u16], alignment: Option<&mut ReservedBuffer>) {
    let input = if let Some(alignment) = alignment {
        let bytes = <ReservedBuffer as AsMut<[u8]>>::as_mut(alignment);
        bytes.copy_from_slice(input);
        bytes
    } else {
        input
    };
    let words = if input.is_empty() {
        &[]
    } else {
        bytemuck::cast_slice::<u8, u16>(input)
    };
    // SAFETY: planning checked width <= 16 and every full/padded encoded chunk
    // size. Both direct output chunks and tail scratch contain exactly 1024 u16.
    // Alignment repair uses an admitted Arrow-aligned buffer, including width 0.
    unsafe { u16::unchecked_unpack(width, words, output) };
}

#[cfg(test)]
mod tests;
