// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Admission for an already drained variable FullZip task. Unzipping and its
//! input owners are caller prerequisites, not covered by this output workspace.

use lance_core::{
    Error, Result,
    utils::allocation::{Reservation, ReservedBuffer},
};

use super::VariableFullZipDecodeTask;
use crate::{
    data::{BlockInfo, VariableWidthBlock},
    decoder::DecodedPage,
    repdef::{ControlWordParser, RepDefUnraveler, max_visible_level},
};

fn copy_levels(
    input: Option<&[u16]>,
    workspace: &mut Reservation,
) -> Result<Option<ReservedBuffer>> {
    input
        .map(|levels| {
            let mut output = workspace
                .try_allocate(std::mem::size_of_val(levels))?
                .ok_or_else(|| Error::internal("FullZip levels exceeded complete admission"))?;
            output
                .as_mut()
                .copy_from_slice(bytemuck::cast_slice(levels));
            Ok(output)
        })
        .transpose()
}

pub(super) fn decode(
    task: &VariableFullZipDecodeTask,
    workspace: &mut Reservation,
) -> Result<Option<DecodedPage>> {
    // An empty selection still has the encoded structural streams. Keep their
    // presence so list reconstruction can emit its leading zero offset.
    let parser = task.details.ctrl_word_parser;
    let has_def = !matches!(
        parser,
        ControlWordParser::NIL
            | ControlWordParser::REP8
            | ControlWordParser::REP16
            | ControlWordParser::REP32
    );
    let rep = (parser.has_rep() || !task.rep.is_empty()).then_some(task.rep.as_ref());
    let def = (has_def || !task.def.is_empty()).then_some(task.def.as_ref());
    RepDefUnraveler::validate_dense_levels(rep, def, &task.details.def_meaning)?;
    let max_visible = max_visible_level(&task.details.def_meaning);
    if task.details.max_visible_def != max_visible {
        return Err(Error::invalid_input(
            "FullZip visibility metadata disagrees with schema",
        ));
    }
    let visible = def.map_or_else(
        || rep.map_or(task.num_visible_items, |v| v.len() as u64),
        |v| v.iter().filter(|v| **v <= max_visible).count() as u64,
    );
    if visible != task.num_visible_items {
        return Err(Error::invalid_input(
            "FullZip visible levels disagree with value count",
        ));
    }
    let block = VariableWidthBlock {
        data: task.data.clone(),
        offsets: task.offsets.clone(),
        bits_per_offset: task.bits_per_offset,
        num_values: task.num_visible_items,
        block_info: BlockInfo::new(),
    };
    let plan = task.decompressor.plan_decode(&block)?;
    let required = plan
        .workspace_bytes
        .checked_add(rep.map_or(0, std::mem::size_of_val))
        .and_then(|n| n.checked_add(def.map_or(0, std::mem::size_of_val)))
        .ok_or_else(|| Error::invalid_input("FullZip workspace size overflows"))?;
    if required > workspace.len() {
        return Ok(None);
    }
    let mut rep_output = None;
    let mut def_output = None;
    let result = (|| {
        rep_output = copy_levels(rep, workspace)?;
        def_output = copy_levels(def, workspace)?;
        let Some(data) = task.decompressor.try_decompress_in(&block, workspace)? else {
            return Ok(None);
        };
        let repdef = RepDefUnraveler::from_admitted_levels(
            rep_output.take(),
            def_output.take(),
            task.details.def_meaning.clone(),
            task.num_visible_items,
        )?;
        Ok(Some(DecodedPage { data, repdef }))
    })();
    // Refusal or codec failure returns exclusive levels to this same workspace.
    for buffer in [rep_output, def_output].into_iter().flatten() {
        buffer.recycle(workspace)?;
    }
    result
}

#[cfg(test)]
mod tests;
