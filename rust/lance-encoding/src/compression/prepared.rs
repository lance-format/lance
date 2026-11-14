// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Keep allocating preflight state exclusive until a complete task is admitted.

use std::ops::Range;

use lance_core::{Error, Result, utils::allocation::Reservation};

use super::{MiniBlockDecodeBudget, MiniBlockDecompressor};
use crate::{buffer::LanceBuffer, data::DataBlock};

/// A validated range retaining any admitted preparation scratch and input owners.
///
/// Preparation has already deducted its payload from the caller's workspace.
/// [`Self::budget`] prices the additional decode allocations, not those retained
/// inputs. Keep all prepared ranges alive while planning a task's output/concat
/// overlap. Use [`Self::recycle`] on refusal to return exclusive scratch to the
/// same workspace. Dropping a plan instead returns its scratch to the global pool.
/// See [`MiniBlockDecompressor::try_prepare_range_in`] for an example.
#[must_use = "decode or recycle preparation before reusing its workspace"]
pub struct PreparedMiniBlockRange<'a> {
    budget: MiniBlockDecodeBudget,
    execution: Box<dyn PreparedExecution + 'a>,
}

pub trait PreparedExecution: Send {
    fn decode(self: Box<Self>, workspace: &mut Reservation) -> Result<Option<DataBlock>>;
    fn recycle(self: Box<Self>, workspace: &mut Reservation) -> Result<()>;
}

impl<'a> PreparedMiniBlockRange<'a> {
    pub(crate) fn new(
        budget: MiniBlockDecodeBudget,
        execution: Box<dyn PreparedExecution + 'a>,
    ) -> Self {
        Self { budget, execution }
    }

    pub(crate) fn direct<T: MiniBlockDecompressor + ?Sized>(
        codec: &'a T,
        data: &[LanceBuffer],
        count: u64,
        range: Range<u64>,
    ) -> Result<Self> {
        let budget = codec.plan_range(data, count, range.clone())?;
        if budget.workspace_bytes < budget.output_bytes {
            return Err(Error::internal("MiniBlock plan omits retained output"));
        }
        Ok(Self::new(
            budget,
            Box::new(Direct {
                codec,
                data: data.to_vec(),
                count,
                range,
            }),
        ))
    }

    /// Additional output/scratch required while this prepared input remains live.
    pub fn budget(&self) -> MiniBlockDecodeBudget {
        self.budget
    }

    /// Decode the validated range, recycling preparation after all views drop.
    /// Insufficient workspace recycles preparation and returns `None`, never EOF.
    /// The supplied workspace must be the one that funded preparation.
    pub fn try_decode(self, workspace: &mut Reservation) -> Result<Option<DataBlock>> {
        if self.budget.workspace_bytes > workspace.len() {
            self.recycle(workspace)?;
            return Ok(None);
        }
        self.execution.decode(workspace)
    }

    /// Abandon the range and return exclusive preparation to its original workspace.
    /// No output has been published. Input owners are released with their last alias.
    pub fn recycle(self, workspace: &mut Reservation) -> Result<()> {
        self.execution.recycle(workspace)
    }
}

struct Direct<'a, T: ?Sized> {
    codec: &'a T,
    data: Vec<LanceBuffer>,
    count: u64,
    range: Range<u64>,
}

impl<T: MiniBlockDecompressor + ?Sized> PreparedExecution for Direct<'_, T> {
    fn decode(self: Box<Self>, workspace: &mut Reservation) -> Result<Option<DataBlock>> {
        self.codec
            .try_decompress_range_in(&self.data, self.count, self.range.clone(), workspace)
    }

    fn recycle(self: Box<Self>, _workspace: &mut Reservation) -> Result<()> {
        Ok(())
    }
}
