// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Keep planning buffers exclusive until every instruction and allocation is admitted.

use std::ops::Range;

use lance_core::{
    Error, Result,
    utils::allocation::{Reservation, ReservedBuffer},
};

use super::super::{ChunkDrainInstructions, DecodeMiniBlockTask, LoadedChunk};
use super::{ValidateRepetition, copy_levels, decode_levels, recycle, selected_ranges};
use crate::{
    compression::{
        MiniBlockDecodeBudget, MiniBlockOutputLayout, PreparedMiniBlockRange,
        levels::LevelDecodePlan,
    },
    data::DataBlock,
    decoder::DecodedPage,
    repdef::{RepDefUnraveler, max_visible_level},
};

struct Prepared<'a> {
    values: Option<PreparedMiniBlockRange<'a>>,
    range: Range<u64>,
    budget: MiniBlockDecodeBudget,
    rep: Option<ReservedBuffer>,
    def: Option<ReservedBuffer>,
}

fn add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| Error::invalid_input("MiniBlock task workspace overflows"))
}

fn prepare<'a>(
    task: &'a DecodeMiniBlockTask,
    instruction: &ChunkDrainInstructions,
    chunk: &LoadedChunk,
    layout: &MiniBlockOutputLayout,
    workspace: &mut Reservation,
    instruction_index: usize,
    validate: Option<&ValidateRepetition<'_>>,
) -> Result<Option<Prepared<'a>>> {
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
        .map(|(codec, data)| codec.plan_levels(data, count))
        .transpose()?;
    let def_plan = task
        .def_decompressor
        .as_ref()
        .zip(sections.def.as_ref())
        .map(|(codec, data)| codec.plan_levels(data, count))
        .transpose()?;
    let required = add(
        rep_plan.as_ref().map_or(0, LevelDecodePlan::output_bytes),
        def_plan
            .as_ref()
            .map_or(0, LevelDecodePlan::workspace_bytes),
    )?
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
    let mut values = None;
    let result = (|| {
        rep = decode_levels(rep_plan.as_ref(), workspace)?;
        def = decode_levels(def_plan.as_ref(), workspace)?;
        let rep_values = rep
            .as_ref()
            .map(|b| bytemuck::cast_slice::<u8, u16>(b.as_ref()));
        let def_values = def
            .as_ref()
            .map(|b| bytemuck::cast_slice::<u8, u16>(b.as_ref()));
        if let Some(validate) = validate {
            validate(instruction_index, rep_values)?;
        }
        RepDefUnraveler::validate_dense_levels(rep_values, def_values, &task.def_meaning)?;
        let (range, levels) = selected_ranges(
            start..end,
            rep_values,
            def_values,
            task.def_meaning.iter().filter(|m| m.is_list()).count() as u16,
            task.max_visible_level,
            chunk.items_in_chunk,
            instruction.preamble_action,
        )?;
        values = task.value_decompressor.try_prepare_range_in(
            &sections.value_buffers,
            chunk.items_in_chunk,
            range.clone(),
            workspace,
        )?;
        let Some(prepared) = values.as_ref() else {
            return Ok(None);
        };
        let budget = prepared.budget();
        layout.validate_range_bytes(range.end - range.start, budget.output_bytes)?;
        let selected = (levels.end - levels.start)
            .checked_mul(2 * (u64::from(rep.is_some()) + u64::from(def.is_some())))
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| Error::invalid_input("MiniBlock selected level size overflows"))?;
        if selected > workspace.len() {
            return Ok(None);
        }
        selected_rep = copy_levels(rep.as_ref(), &levels, workspace)?;
        selected_def = copy_levels(def.as_ref(), &levels, workspace)?;
        Ok(Some(Prepared {
            values: values.take(),
            range,
            budget,
            rep: selected_rep.take(),
            def: selected_def.take(),
        }))
    })();
    if let Some(values) = values {
        values.recycle(workspace)?;
    }
    recycle(&mut rep, workspace)?;
    recycle(&mut def, workspace)?;
    recycle(&mut selected_rep, workspace)?;
    recycle(&mut selected_def, workspace)?;
    result
}

fn combine_levels(
    plans: &[Prepared<'_>],
    is_rep: bool,
    bytes: usize,
    is_present: bool,
    workspace: &mut Reservation,
) -> Result<Option<ReservedBuffer>> {
    if !is_present {
        return Ok(None);
    }
    let mut output = workspace
        .try_allocate(bytes)?
        .ok_or_else(|| Error::internal("MiniBlock combined levels exceeded preflight"))?;
    let mut cursor = 0;
    for plan in plans {
        let input = if is_rep { &plan.rep } else { &plan.def };
        if let Some(input) = input {
            let end = cursor + input.as_ref().len();
            output.as_mut()[cursor..end].copy_from_slice(input.as_ref());
            cursor = end;
        }
    }
    Ok(Some(output))
}

pub(super) fn decode(
    task: &DecodeMiniBlockTask,
    workspace: &mut Reservation,
    validate: Option<&ValidateRepetition<'_>>,
) -> Result<Option<DecodedPage>> {
    RepDefUnraveler::validate_dense_levels(None, None, &task.def_meaning)?;
    if task.max_visible_level != max_visible_level(&task.def_meaning) {
        return Err(Error::invalid_input(
            "MiniBlock visibility metadata disagrees with schema",
        ));
    }
    let layout = task.value_decompressor.admitted_output_layout()?;
    if task.instructions.is_empty() && matches!(layout, MiniBlockOutputLayout::VariableWidth { .. })
    {
        return Err(Error::not_supported(
            "empty variable-width tasks need an explicit empty range for typed output",
        ));
    }
    let mut plans = Vec::with_capacity(task.instructions.len());
    let mut rep = None;
    let mut def = None;
    let result = (|| {
        let mut retained = 0;
        let mut value_peak = 0;
        let mut rep_bytes = 0;
        let mut def_bytes = 0;
        let mut values = 0_u64;
        for (index, (instruction, chunk)) in task.instructions.iter().enumerate() {
            let Some(plan) = prepare(
                task,
                instruction,
                chunk,
                &layout,
                workspace,
                index,
                validate,
            )?
            else {
                return Ok(None);
            };
            // Store the owned scratch before any fallible arithmetic so an error
            // returns all credit to this exact workspace, never the global pool.
            plans.push(plan);
            let plan = plans
                .last()
                .ok_or_else(|| Error::internal("missing prepared MiniBlock"))?;
            value_peak = value_peak.max(add(retained, plan.budget.workspace_bytes)?);
            retained = add(retained, plan.budget.output_bytes)?;
            values = values
                .checked_add(plan.range.end - plan.range.start)
                .ok_or_else(|| Error::invalid_input("MiniBlock selected value count overflows"))?;
            rep_bytes = add(rep_bytes, plan.rep.as_ref().map_or(0, |b| b.as_ref().len()))?;
            def_bytes = add(def_bytes, plan.def.as_ref().map_or(0, |b| b.as_ref().len()))?;
        }
        let concat_bytes = layout.concat_payload_bytes(
            plans
                .iter()
                .map(|plan| (plan.range.end - plan.range.start, plan.budget.output_bytes)),
        )?;
        let level_bytes = add(rep_bytes, def_bytes)?;
        let required = value_peak
            .max(add(retained, concat_bytes)?)
            .max(level_bytes);
        // Preparation is already deducted from workspace. Keep that charge
        // throughout this conservative output/concat preflight, even though each
        // decoded range will recycle its own preparation before the next decode.
        if required > workspace.len() {
            return Ok(None);
        }

        rep = combine_levels(
            &plans,
            true,
            rep_bytes,
            task.rep_decompressor.is_some(),
            workspace,
        )?;
        def = combine_levels(
            &plans,
            false,
            def_bytes,
            task.def_decompressor.is_some(),
            workspace,
        )?;
        for plan in &mut plans {
            recycle(&mut plan.rep, workspace)?;
            recycle(&mut plan.def, workspace)?;
        }
        // Check the combined representation before freezing any value output.
        RepDefUnraveler::validate_dense_levels(
            rep.as_ref()
                .map(|b| bytemuck::cast_slice::<u8, u16>(b.as_ref())),
            def.as_ref()
                .map(|b| bytemuck::cast_slice::<u8, u16>(b.as_ref())),
            &task.def_meaning,
        )?;
        let mut decoded = Vec::with_capacity(plans.len());
        for plan in &mut plans {
            let data = plan
                .values
                .take()
                .ok_or_else(|| Error::internal("missing MiniBlock value preparation"))?
                .try_decode(workspace)?
                .ok_or_else(|| {
                    Error::internal("MiniBlock value decoder refused its complete plan")
                })?;
            decoded.push(data);
        }
        let data = DataBlock::try_concat_in(&decoded, workspace)?
            .ok_or_else(|| Error::internal("MiniBlock concatenation refused its complete plan"))?;
        // Shared intermediate values refund the global pool when dropped. Their
        // credit is not reused for a later operation in this exclusive workspace.
        drop(decoded);
        let repdef = RepDefUnraveler::from_admitted_levels(
            rep.take(),
            def.take(),
            task.def_meaning.clone(),
            values,
        )?;
        Ok(Some(DecodedPage { data, repdef }))
    })();
    for plan in &mut plans {
        if let Some(values) = plan.values.take() {
            values.recycle(workspace)?;
        }
        recycle(&mut plan.rep, workspace)?;
        recycle(&mut plan.def, workspace)?;
    }
    recycle(&mut rep, workspace)?;
    recycle(&mut def, workspace)?;
    result
}
