// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::compression::{PreparedMiniBlockRange, PreparedUnsigned, prepared::PreparedExecution};

pub fn prepare<'a>(
    codec: &'a BinaryMiniBlockDecompressor,
    data: &[LanceBuffer],
    num_values: u64,
    range: Range<u64>,
    workspace: &mut Reservation,
) -> Result<Option<PreparedMiniBlockRange<'a>>> {
    // Preserve pure planning for codecs that can traverse encoded offsets directly.
    match PreparedMiniBlockRange::direct(codec, data, num_values, range.clone()) {
        Ok(plan) => return Ok(Some(plan)),
        Err(Error::NotSupported { .. }) => {}
        Err(error) => return Err(error),
    }
    let BinaryMiniBlockLayout::Generic {
        offsets,
        bits_per_offset,
        offsets_have_payload: true,
    } = &codec.layout
    else {
        return Err(Error::not_supported(
            "binary offsets have no admitted preparation",
        ));
    };
    let [input, values] = data else {
        return Err(Error::corrupt_file_named(
            "binary mini-block",
            "expected offset and value buffers",
        ));
    };
    let count = num_values
        .checked_add(1)
        .ok_or_else(|| Error::invalid_input("binary offset count overflows"))?;
    let Some(prepared) = offsets.try_prepare_unsigned_in(
        Some(input),
        u64::from(*bits_per_offset),
        count,
        workspace,
    )?
    else {
        return Ok(None);
    };
    // Validate the complete offset sequence before publishing selected values.
    let plan = payload::plan_visit(codec, num_values, range, values.len(), |visitor| {
        prepared.visit(visitor)
    });
    match plan {
        Ok(plan) => Ok(Some(PreparedMiniBlockRange::new(
            plan.budget,
            Box::new(BinaryPrepared {
                offsets: prepared,
                values: values.clone(),
                plan,
                num_values,
            }),
        ))),
        Err(error) => {
            prepared.recycle(workspace)?;
            Err(error)
        }
    }
}

struct BinaryPrepared<'a> {
    offsets: PreparedUnsigned<'a>,
    values: LanceBuffer,
    plan: RangePlan,
    num_values: u64,
}

impl PreparedExecution for BinaryPrepared<'_> {
    fn decode(self: Box<Self>, workspace: &mut Reservation) -> Result<Option<DataBlock>> {
        let Self {
            offsets,
            values,
            plan,
            num_values,
        } = *self;
        let result = materialize(&plan, &values, workspace, |output| {
            payload::write_visit(num_values, &plan, output, |visitor| offsets.visit(visitor))
        });
        offsets.recycle(workspace)?;
        result
    }

    fn recycle(self: Box<Self>, workspace: &mut Reservation) -> Result<()> {
        self.offsets.recycle(workspace)
    }
}
