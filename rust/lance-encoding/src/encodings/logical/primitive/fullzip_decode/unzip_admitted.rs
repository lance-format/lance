// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Select rows directly from charged zipped input without a whole-page index.

use std::ops::Range;

use arrow_buffer::{Buffer, ScalarBuffer};
use lance_core::{Error, Result, utils::allocation::Reservation};

use super::{
    FullZipScheduler, PerValueDecompressor, VariableFullZipDecodeTask, VariableFullZipDecoder,
};
use crate::{
    buffer::LanceBuffer,
    compression::MiniBlockOutputLayout,
    decoder::DecodePageTask,
    repdef::{ControlWordParser, RepDefUnraveler, max_visible_level},
};

#[derive(Default)]
struct Plan {
    values: usize,
    levels: usize,
    payload: usize,
    offsets: usize,
    rep: usize,
    def: usize,
    total: usize,
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| Error::invalid_input("FullZip unzip allocation size overflows"))
}

fn has_def(parser: ControlWordParser) -> bool {
    !matches!(
        parser,
        ControlWordParser::NIL
            | ControlWordParser::REP8
            | ControlWordParser::REP16
            | ControlWordParser::REP32
    )
}

// Decode into scalars rather than using the ordinary parser's growing Vecs.
fn levels(parser: ControlWordParser, src: &[u8]) -> Result<(u32, u32)> {
    let width = parser.bytes_per_word();
    let bytes = src
        .get(..width)
        .ok_or_else(|| Error::corrupt_file_named("FullZip", "truncated control word"))?;
    let mut word = [0; 4];
    word[..width].copy_from_slice(bytes);
    let word = u32::from_le_bytes(word);
    Ok(match parser {
        ControlWordParser::BOTH8(shift, mask)
        | ControlWordParser::BOTH16(shift, mask)
        | ControlWordParser::BOTH32(shift, mask) => (word >> shift, word & mask),
        ControlWordParser::REP8 | ControlWordParser::REP16 | ControlWordParser::REP32 => (word, 0),
        ControlWordParser::DEF8 | ControlWordParser::DEF16 | ControlWordParser::DEF32 => (0, word),
        ControlWordParser::NIL => (0, 0),
    })
}

/// Walk complete row-aligned buffers; callbacks see only the requested rows.
/// Validate all framing, including unselected tails, before admitting output.
fn visit(
    scheduler: &FullZipScheduler,
    data: &[LanceBuffer],
    num_rows: u64,
    range: Range<u64>,
    mut visitor: impl FnMut(u16, u16, bool, &[u8]) -> Result<()>,
) -> Result<()> {
    if !matches!(
        scheduler.details.value_decompressor,
        PerValueDecompressor::Variable(_)
    ) {
        return Err(Error::not_supported(
            "admitted FullZip unzip requires variable values",
        ));
    }
    if scheduler.structural_projection.is_some() {
        return Err(Error::not_supported(
            "admitted FullZip unzip does not apply structural projection",
        ));
    }
    if !matches!(scheduler.bits_per_offset, 32 | 64)
        || range.start > range.end
        || range.end > num_rows
    {
        return Err(Error::invalid_input(
            "invalid FullZip offset width or unzip row range",
        ));
    }
    let details = &scheduler.details;
    RepDefUnraveler::validate_dense_levels(None, None, &details.def_meaning)?;
    let max_rep = details.def_meaning.iter().filter(|m| m.is_list()).count() as u32;
    let max_def = details
        .def_meaning
        .iter()
        .map(|m| u32::from(m.num_def_levels()))
        .sum::<u32>();
    if u32::from(details.max_rep) != max_rep
        || details.max_visible_def != max_visible_level(&details.def_meaning)
    {
        return Err(Error::invalid_input(
            "FullZip levels disagree with structural interpretation",
        ));
    }
    let parser = details.ctrl_word_parser;
    let width = usize::from(scheduler.bits_per_offset / 8);
    let mut rows = 0_u64;
    for buffer in data {
        let mut bytes = buffer.as_ref();
        let mut first = true;
        while !bytes.is_empty() {
            let (rep, def) = levels(parser, bytes)?;
            if rep > max_rep || def > max_def {
                return Err(Error::corrupt_file_named(
                    "FullZip",
                    "control level exceeds schema",
                ));
            }
            let new_row = !parser.has_rep() || rep == max_rep;
            if first && !new_row {
                return Err(Error::corrupt_file_named(
                    "FullZip",
                    "input buffer starts inside a row",
                ));
            }
            first = false;
            if new_row {
                rows = rows
                    .checked_add(1)
                    .ok_or_else(|| Error::invalid_input("FullZip row count overflows"))?;
                if rows > num_rows {
                    return Err(Error::corrupt_file_named(
                        "FullZip",
                        "input has more rows than declared",
                    ));
                }
            }
            bytes = &bytes[parser.bytes_per_word()..];
            let visible = def <= u32::from(details.max_visible_def);
            let value = if visible && def == 0 {
                let length =
                    VariableFullZipDecoder::parse_length(bytes, scheduler.bits_per_offset)?;
                bytes = &bytes[width..];
                let length = usize::try_from(length).map_err(|_| {
                    Error::corrupt_file_named("FullZip", "item length exceeds addressable size")
                })?;
                let value = bytes.get(..length).ok_or_else(|| {
                    Error::corrupt_file_named("FullZip", "item length exceeds remaining input")
                })?;
                bytes = &bytes[length..];
                value
            } else {
                &[]
            };
            if range.contains(&(rows - 1)) {
                visitor(rep as u16, def as u16, visible, value)?;
            }
        }
    }
    if rows != num_rows {
        return Err(Error::corrupt_file_named(
            "FullZip",
            "input row count differs from declared rows",
        ));
    }
    Ok(())
}

fn plan(
    scheduler: &FullZipScheduler,
    data: &[LanceBuffer],
    rows: u64,
    range: Range<u64>,
) -> Result<Plan> {
    let mut plan = Plan::default();
    visit(scheduler, data, rows, range, |_, _, visible, value| {
        plan.levels = add(plan.levels, 1)?;
        if visible {
            plan.values = add(plan.values, 1)?;
            plan.payload = add(plan.payload, value.len())?;
        }
        Ok(())
    })?;
    plan.offsets = add(plan.values, 1)?
        .checked_mul(usize::from(scheduler.bits_per_offset / 8))
        .ok_or_else(|| Error::invalid_input("FullZip offsets size overflows"))?;
    let value_bytes = add(plan.offsets, plan.payload)?;
    MiniBlockOutputLayout::VariableWidth {
        bits_per_offset: scheduler.bits_per_offset,
    }
    .validate_range_bytes(plan.values as u64, value_bytes)?;
    let level_bytes = plan
        .levels
        .checked_mul(2)
        .ok_or_else(|| Error::invalid_input("FullZip levels size overflows"))?;
    let parser = scheduler.details.ctrl_word_parser;
    plan.rep = if parser.has_rep() { level_bytes } else { 0 };
    plan.def = if has_def(parser) { level_bytes } else { 0 };
    plan.total = add(add(value_bytes, plan.rep)?, plan.def)?;
    std::alloc::Layout::from_size_align(plan.total, arrow_buffer::alloc::ALIGNMENT)
        .map_err(|_| Error::invalid_input("FullZip unzip exceeds addressable allocation size"))?;
    Ok(plan)
}

impl FullZipScheduler {
    /// Exact additional payload bytes for unzipping a row range, before codec
    /// decompression. Input buffers must each contain complete rows, in selection
    /// order; `range` indexes their concatenation of `num_rows` rows. All input
    /// framing is validated, including unselected rows, without payload allocation.
    /// The plan includes selected encoded values, offsets and dense rep/def levels,
    /// but excludes the already charged physical input and later decode output.
    /// See [`Self::try_unzip_range_in`] for staged admission and usage.
    pub fn plan_unzip_range(
        &self,
        data: &[LanceBuffer],
        num_rows: u64,
        range: Range<u64>,
    ) -> Result<usize> {
        Ok(plan(self, data, num_rows, range)?.total)
    }

    /// Unzip only selected rows into one exact admitted allocation. No whole-page
    /// data copy or row index is built. `None` preserves input and workspace;
    /// corrupt input is rejected even without credit. Each input buffer must
    /// contain complete rows and its entire backing allocation must be charged.
    /// The returned task retains its unzip allocation until its final alias drops.
    /// Calling its admitted decode requires separate workspace for decoded values
    /// and copied levels, overlapping input and unzip storage. Refusal is a signal
    /// to drop/drain/replan, never to wait while holding the same budget.
    /// This opt-in entry point does not change ordinary scheduling or bound I/O,
    /// caches, metadata or overall process memory. Structural projections and
    /// fixed-width FullZip are not supported here.
    ///
    /// ```
    /// use lance_core::{Result, utils::allocation::Reservation};
    /// use lance_encoding::{buffer::LanceBuffer, decoder::DecodePageTask,
    ///     encodings::logical::primitive::FullZipScheduler};
    /// fn prepare(scheduler: &FullZipScheduler, charged_input: &[LanceBuffer],
    ///     rows: u64, work: &mut Reservation) -> Result<Option<Box<dyn DecodePageTask>>> {
    ///     scheduler.try_unzip_range_in(charged_input, rows, 0..rows, work)
    /// }
    /// ```
    pub fn try_unzip_range_in(
        &self,
        data: &[LanceBuffer],
        num_rows: u64,
        range: Range<u64>,
        workspace: &mut Reservation,
    ) -> Result<Option<Box<dyn DecodePageTask>>> {
        let plan = plan(self, data, num_rows, range.clone())?;
        let Some(mut output) = workspace.try_allocate(plan.total)? else {
            return Ok(None);
        };
        let rep_start = plan.offsets;
        let def_start = rep_start + plan.rep;
        let data_start = def_start + plan.def;
        let width = usize::from(self.bits_per_offset / 8);
        let mut row = 0;
        let mut level = 0;
        let mut written = 0;
        let replay = visit(self, data, num_rows, range, |rep, def, visible, value| {
            if plan.rep != 0 {
                output.as_mut()[rep_start + 2 * level..rep_start + 2 * level + 2]
                    .copy_from_slice(&rep.to_ne_bytes());
            }
            if plan.def != 0 {
                output.as_mut()[def_start + 2 * level..def_start + 2 * level + 2]
                    .copy_from_slice(&def.to_ne_bytes());
            }
            level += 1;
            if visible {
                output.as_mut()[data_start + written..data_start + written + value.len()]
                    .copy_from_slice(value);
                written += value.len();
                row += 1;
                output.as_mut()[row * width..(row + 1) * width]
                    .copy_from_slice(&(written as u64).to_le_bytes()[..width]);
            }
            Ok(())
        });
        if let Err(error) = replay {
            output.recycle(workspace)?;
            return Err(error);
        }
        if row != plan.values || level != plan.levels || written != plan.payload {
            output.recycle(workspace)?;
            return Err(Error::internal(
                "FullZip unzip replay disagrees with preflight",
            ));
        }
        let PerValueDecompressor::Variable(codec) = &self.details.value_decompressor else {
            output.recycle(workspace)?;
            return Err(Error::internal("validated variable FullZip codec changed"));
        };
        let storage = Buffer::from(output.freeze());
        Ok(Some(Box::new(VariableFullZipDecodeTask {
            details: self.details.clone(),
            decompressor: codec.clone(),
            offsets: LanceBuffer::from(storage.slice_with_length(0, plan.offsets)),
            data: LanceBuffer::from(storage.slice_with_length(data_start, plan.payload)),
            rep: ScalarBuffer::new(
                storage.slice_with_length(rep_start, plan.rep),
                0,
                plan.rep / 2,
            ),
            def: ScalarBuffer::new(
                storage.slice_with_length(def_start, plan.def),
                0,
                plan.def / 2,
            ),
            bits_per_offset: self.bits_per_offset,
            num_visible_items: plan.values as u64,
        })))
    }
}

#[cfg(test)]
mod tests;
