// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Select rows directly from charged zipped input without a whole-page index.

use std::{ops::Range, sync::Arc};

use arrow_buffer::{Buffer, ScalarBuffer};
use bytes::Bytes;
use lance_core::{
    Error, Result,
    utils::allocation::{Reservation, ReservedBuffer},
};

use super::{FullZipScheduler, PerValueDecompressor, VariableFullZipDecodeTask};
use crate::{
    buffer::LanceBuffer,
    compression::MiniBlockOutputLayout,
    decoder::{DecodePageTask, DecodedPage},
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

// Construction is private: all slices share exactly one newly admitted unzip
// allocation. This proof must not be inferred from a generic task's visible size.
pub(super) struct AdmittedUnzipTask {
    task: VariableFullZipDecodeTask,
    allocation: Arc<ReservedBuffer>,
}

// Bytes and Arrow views share this owner, so uniqueness proves that no decoded
// output or codec-retained alias can still access the allocation.
struct UnzipOwner(Arc<ReservedBuffer>);

impl AsRef<[u8]> for UnzipOwner {
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref().as_ref()
    }
}

impl AdmittedUnzipTask {
    pub(super) fn decode(self, workspace: &mut Reservation) -> Result<Option<DecodedPage>> {
        let Self { task, allocation } = self;
        let result = super::admitted::decode_reusing_values(&task, workspace);
        drop(task);
        if let Ok(buffer) = Arc::try_unwrap(allocation) {
            buffer.recycle(workspace)?;
        }
        result
    }
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

/// Walk complete row-aligned buffers; callbacks see only the requested rows.
/// Validate all framing, including unselected tails, before admitting output.
pub(super) fn visit(
    scheduler: &FullZipScheduler,
    data: &[impl AsRef<[u8]>],
    num_rows: u64,
    range: Range<u64>,
    visitor: impl FnMut(u16, u16, bool, &[u8]) -> Result<()>,
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
    // Dispatch once per buffer walk so neither framing width nor parser kind
    // needs a dynamic branch for each value in both preflight and replay.
    match scheduler.bits_per_offset {
        32 => visit_control_width::<4>(scheduler, data, num_rows, range, visitor, max_rep, max_def),
        64 => visit_control_width::<8>(scheduler, data, num_rows, range, visitor, max_rep, max_def),
        _ => Err(Error::internal("validated FullZip offset width changed")),
    }
}

fn visit_control_width<const OFFSET: usize>(
    scheduler: &FullZipScheduler,
    data: &[impl AsRef<[u8]>],
    num_rows: u64,
    range: Range<u64>,
    visitor: impl FnMut(u16, u16, bool, &[u8]) -> Result<()>,
    max_rep: u32,
    max_def: u32,
) -> Result<()> {
    match scheduler.details.ctrl_word_parser.bytes_per_word() {
        0 => visit_bytes::<0, OFFSET>(scheduler, data, num_rows, range, visitor, max_rep, max_def),
        1 => visit_bytes::<1, OFFSET>(scheduler, data, num_rows, range, visitor, max_rep, max_def),
        2 => visit_bytes::<2, OFFSET>(scheduler, data, num_rows, range, visitor, max_rep, max_def),
        4 => visit_bytes::<4, OFFSET>(scheduler, data, num_rows, range, visitor, max_rep, max_def),
        _ => Err(Error::internal("unsupported FullZip control word width")),
    }
}

fn visit_bytes<const CONTROL: usize, const OFFSET: usize>(
    scheduler: &FullZipScheduler,
    data: &[impl AsRef<[u8]>],
    num_rows: u64,
    range: Range<u64>,
    mut visitor: impl FnMut(u16, u16, bool, &[u8]) -> Result<()>,
    max_rep: u32,
    max_def: u32,
) -> Result<()> {
    let details = &scheduler.details;
    let parser = details.ctrl_word_parser;
    let (rep_shift, def_mask) = match parser {
        ControlWordParser::BOTH8(shift, mask)
        | ControlWordParser::BOTH16(shift, mask)
        | ControlWordParser::BOTH32(shift, mask) => (u32::from(shift), u64::from(mask)),
        ControlWordParser::REP8 | ControlWordParser::REP16 | ControlWordParser::REP32 => (0, 0),
        ControlWordParser::DEF8 | ControlWordParser::DEF16 | ControlWordParser::DEF32 => {
            (32, u64::from(u32::MAX))
        }
        ControlWordParser::NIL => (0, 0),
    };
    let has_rep = parser.has_rep();
    let mut rows = 0_u64;
    for buffer in data {
        let mut bytes = buffer.as_ref();
        let mut first = true;
        while !bytes.is_empty() {
            let control = bytes
                .get(..CONTROL)
                .ok_or_else(|| Error::corrupt_file_named("FullZip", "truncated control word"))?;
            let mut word = [0; 8];
            word[..CONTROL].copy_from_slice(control);
            let word = u64::from_le_bytes(word);
            let rep = (word >> rep_shift) as u32;
            let def = (word & def_mask) as u32;
            if rep > max_rep || def > max_def {
                return Err(Error::corrupt_file_named(
                    "FullZip",
                    "control level exceeds schema",
                ));
            }
            let new_row = !has_rep || rep == max_rep;
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
            bytes = &bytes[CONTROL..];
            let visible = def <= u32::from(details.max_visible_def);
            let value = if visible && def == 0 {
                let prefix = bytes.get(..OFFSET).ok_or_else(|| {
                    Error::corrupt_file_named("FullZip", "truncated length prefix")
                })?;
                let mut length = [0; 8];
                length[..OFFSET].copy_from_slice(prefix);
                let length = u64::from_le_bytes(length);
                bytes = &bytes[OFFSET..];
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
    data: &[impl AsRef<[u8]>],
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
        self.try_unzip_buffers_in(data, num_rows, range, workspace)
    }

    // Borrow exclusive read allocations so the read stage can recycle their
    // credit after unzipping, including when output admission is refused.
    pub(super) fn try_unzip_buffers_in(
        &self,
        data: &[impl AsRef<[u8]>],
        num_rows: u64,
        range: Range<u64>,
        workspace: &mut Reservation,
    ) -> Result<Option<Box<dyn DecodePageTask>>> {
        Ok(self
            .try_unzip_owned_in(data, num_rows, range, workspace)?
            .map(|task| Box::new(task.task) as Box<dyn DecodePageTask>))
    }

    pub(super) fn try_unzip_owned_in(
        &self,
        data: &[impl AsRef<[u8]>],
        num_rows: u64,
        range: Range<u64>,
        workspace: &mut Reservation,
    ) -> Result<Option<AdmittedUnzipTask>> {
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
        let allocation = Arc::new(output);
        let storage = Buffer::from(Bytes::from_owner(UnzipOwner(allocation.clone())));
        Ok(Some(AdmittedUnzipTask {
            task: VariableFullZipDecodeTask {
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
            },
            allocation,
        }))
    }
}

#[cfg(test)]
mod tests;
