// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Admitted range decoding for one drained MiniBlock instruction. Levels are
//! decoded into charged scratch to resolve the selected value range. A later
//! refusal recycles this scratch and selected-level copies into the same
//! workspace. No task cursor is advanced, and no ordinary decoder is called.
//! Multiple instructions need admitted concatenation/builder support first.

use std::ops::Range;

use lance_core::{
    Error, Result,
    utils::allocation::{Reservation, ReservedBuffer},
};

use super::{DecodeMiniBlockTask, PreambleAction};
use crate::{
    compression::levels::LevelDecodePlan,
    decoder::DecodedPage,
    repdef::{RepDefUnraveler, max_visible_level},
};

fn decode_levels(
    plan: Option<&LevelDecodePlan<'_>>,
    workspace: &mut Reservation,
) -> Result<Option<ReservedBuffer>> {
    plan.map(|plan| {
        plan.try_decode_in(workspace)?
            .ok_or_else(|| Error::internal("MiniBlock level preflight changed during decode"))
    })
    .transpose()
}

fn copy_levels(
    levels: Option<&ReservedBuffer>,
    range: &Range<u64>,
    workspace: &mut Reservation,
) -> Result<Option<ReservedBuffer>> {
    levels
        .map(|levels| {
            let bytes = &levels.as_ref()[range.start as usize * 2..range.end as usize * 2];
            let mut output = workspace
                .try_allocate(bytes.len())?
                .ok_or_else(|| Error::internal("MiniBlock selected levels exceeded preflight"))?;
            output.as_mut().copy_from_slice(bytes);
            Ok(output)
        })
        .transpose()
}

fn recycle(buffer: &mut Option<ReservedBuffer>, workspace: &mut Reservation) -> Result<()> {
    if let Some(buffer) = buffer.take() {
        buffer.recycle(workspace)?;
    }
    Ok(())
}

pub(super) fn decode(
    task: &DecodeMiniBlockTask,
    workspace: &mut Reservation,
) -> Result<Option<DecodedPage>> {
    if task.dictionary_data.is_some() || task.structural_projection.is_some() {
        return Err(Error::not_supported(
            "admitted MiniBlock dictionary/projection tasks need separate output planning",
        ));
    }
    let [(instruction, chunk)] = task.instructions.as_slice() else {
        return Err(Error::not_supported(
            "admitted MiniBlock tasks currently require one instruction; split before drain",
        ));
    };
    RepDefUnraveler::validate_dense_levels(None, None, &task.def_meaning)?;
    let max_rep = task.def_meaning.iter().filter(|m| m.is_list()).count() as u16;
    if task.max_visible_level != max_visible_level(&task.def_meaning) {
        return Err(Error::invalid_input(
            "MiniBlock visibility metadata disagrees with schema",
        ));
    }
    let start = instruction
        .rows_to_skip
        .checked_add(instruction.chunk_instructions.rows_to_skip)
        .ok_or_else(|| Error::invalid_input("MiniBlock selected row start overflows"))?;
    let end = start
        .checked_add(instruction.rows_to_take)
        .ok_or_else(|| Error::invalid_input("MiniBlock selected row end overflows"))?;
    let sections = DecodeMiniBlockTask::split_miniblock_chunk(
        &chunk.data,
        task.rep_decompressor.is_some(),
        task.def_decompressor.is_some(),
        task.num_buffers,
        task.has_large_chunk,
    )?;
    // The supported level codecs recover counts using bytes/metadata only;
    // retain the existing reader's exact count and wrapped-header validation.
    let count = DecodeMiniBlockTask::resolve_num_levels(
        task.rep_decompressor.as_deref(),
        sections.rep.as_ref(),
        task.def_decompressor.as_deref(),
        sections.def.as_ref(),
        sections.num_levels,
    )?;
    let rep_plan = task
        .rep_decompressor
        .as_ref()
        .zip(sections.rep.as_ref())
        .map(|(codec, input)| codec.plan_levels(input, count))
        .transpose()?;
    let def_plan = task
        .def_decompressor
        .as_ref()
        .zip(sections.def.as_ref())
        .map(|(codec, input)| codec.plan_levels(input, count))
        .transpose()?;
    let required = rep_plan
        .as_ref()
        .map_or(0, LevelDecodePlan::output_bytes)
        .checked_add(
            def_plan
                .as_ref()
                .map_or(0, LevelDecodePlan::workspace_bytes),
        )
        .ok_or_else(|| Error::invalid_input("MiniBlock level workspace overflows"))?
        .max(
            rep_plan
                .as_ref()
                .map_or(0, LevelDecodePlan::workspace_bytes),
        );
    if required > workspace.len() {
        return Ok(None);
    }

    let mut rep = None;
    let mut def = None;
    let mut selected_rep = None;
    let mut selected_def = None;
    let result = (|| {
        rep = decode_levels(rep_plan.as_ref(), workspace)?;
        def = decode_levels(def_plan.as_ref(), workspace)?;
        let rep_values = rep
            .as_ref()
            .map(|b| bytemuck::cast_slice::<u8, u16>(b.as_ref()));
        let def_values = def
            .as_ref()
            .map(|b| bytemuck::cast_slice::<u8, u16>(b.as_ref()));
        RepDefUnraveler::validate_dense_levels(rep_values, def_values, &task.def_meaning)?;
        let (items, levels) = selected_ranges(
            start..end,
            rep_values,
            def_values,
            max_rep,
            task.max_visible_level,
            chunk.items_in_chunk,
            instruction.preamble_action,
        )?;
        let selected_bytes = (levels.end - levels.start)
            .checked_mul(2 * (u64::from(rep.is_some()) + u64::from(def.is_some())))
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| Error::invalid_input("MiniBlock selected level size overflows"))?;
        if selected_bytes > workspace.len() {
            return Ok(None);
        }
        selected_rep = copy_levels(rep.as_ref(), &levels, workspace)?;
        selected_def = copy_levels(def.as_ref(), &levels, workspace)?;
        // This codec API preflights all value output/scratch before it allocates.
        // Full and selected levels remain charged throughout the call. Refusal
        // can therefore safely return all level credit without re-admission.
        let Some(data) = task.value_decompressor.try_decompress_range_in(
            &sections.value_buffers,
            chunk.items_in_chunk,
            items.clone(),
            workspace,
        )?
        else {
            return Ok(None);
        };
        let repdef = RepDefUnraveler::from_admitted_levels(
            selected_rep.take(),
            selected_def.take(),
            task.def_meaning.clone(),
            items.end - items.start,
        )?;
        Ok(Some(DecodedPage { data, repdef }))
    })();
    // Keep ownership until this point, including errors in mapping/codec checks.
    // Shared input owners remain with the borrowed task and are never refunded here.
    recycle(&mut rep, workspace)?;
    recycle(&mut def, workspace)?;
    recycle(&mut selected_rep, workspace)?;
    recycle(&mut selected_def, workspace)?;
    result
}

/// Establish the existing range mapper's invariants before using decoded levels.
fn selected_ranges(
    rows: Range<u64>,
    rep: Option<&[u16]>,
    def: Option<&[u16]>,
    max_rep: u16,
    max_visible: u16,
    items: u64,
    preamble: PreambleAction,
) -> Result<(Range<u64>, Range<u64>)> {
    let visible = def.map_or_else(
        || rep.map_or(items, |v| v.len() as u64),
        |v| v.iter().filter(|v| **v <= max_visible).count() as u64,
    );
    if visible != items {
        return Err(Error::invalid_input(
            "MiniBlock visible levels disagree with item count",
        ));
    }
    if let Some(rep) = rep {
        let starts = rep.iter().filter(|v| **v == max_rep).count() as u64;
        if max_rep == 0 || rows.end > starts {
            return Err(Error::invalid_input(
                "MiniBlock selected rows exceed repetition boundaries",
            ));
        }
        if rep.is_empty() {
            if preamble != PreambleAction::Absent {
                return Err(Error::invalid_input("empty MiniBlock has no preamble"));
            }
            return Ok((0..0, 0..0));
        }
        if (preamble == PreambleAction::Absent && rep[0] != max_rep)
            || (preamble == PreambleAction::Skip && starts == 0)
            || (preamble == PreambleAction::Take && rows.start != 0)
        {
            return Err(Error::invalid_input(
                "MiniBlock preamble disagrees with selection",
            ));
        }
    } else if max_rep != 0 || preamble != PreambleAction::Absent {
        return Err(Error::not_supported(
            "admitted list selection requires repetition levels",
        ));
    } else if rows.end > items {
        return Err(Error::invalid_input(
            "MiniBlock selected rows exceed item count",
        ));
    }
    let (selected_items, levels) = DecodeMiniBlockTask::map_range(
        rows,
        rep.as_ref(),
        def.as_ref(),
        max_rep,
        max_visible,
        items,
        preamble,
    );
    if selected_items.start > selected_items.end
        || selected_items.end > items
        || levels.start > levels.end
        || rep.is_some_and(|v| levels.end > v.len() as u64)
        || def.is_some_and(|v| levels.end > v.len() as u64)
    {
        return Err(Error::invalid_input(
            "MiniBlock mapped selection exceeds decoded levels",
        ));
    }
    Ok((selected_items, levels))
}

#[cfg(test)]
mod tests;
