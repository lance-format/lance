// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Borrowed unsigned traversal after explicitly admitted input preparation.

use lance_core::{Result, utils::allocation::Reservation};

use super::BlockDecompressor;
use crate::buffer::LanceBuffer;

/// A repeatable unsigned visitor retaining its charged input preparation.
///
/// Construction does not validate every value. Complete a visit before publishing
/// dependent output: a late error can follow a valid prefix. Visits allocate no
/// payload; the callback owns any allocations it makes. Preparation is already
/// deducted from the workspace and must remain live during dependent admission.
/// [`Self::recycle`] returns exclusive scratch to the same workspace; dropping
/// this object returns its charge to the global pool instead.
/// See [`BlockDecompressor::try_prepare_unsigned_in`] for an example.
#[must_use = "visit and recycle preparation before reusing its workspace"]
pub struct PreparedUnsigned<'a> {
    execution: Box<dyn UnsignedExecution + 'a>,
}

pub trait UnsignedExecution: Send {
    fn visit(&self, visitor: &mut dyn FnMut(u64) -> Result<()>) -> Result<()>;
    fn recycle(self: Box<Self>, workspace: &mut Reservation) -> Result<()>;
}

impl<'a> PreparedUnsigned<'a> {
    pub(crate) fn new(execution: Box<dyn UnsignedExecution + 'a>) -> Self {
        Self { execution }
    }

    pub(crate) fn direct<T: BlockDecompressor + ?Sized>(
        codec: &'a T,
        data: Option<&LanceBuffer>,
        bits: u64,
        count: u64,
    ) -> Self {
        Self::new(Box::new(Direct {
            codec,
            data: data.cloned(),
            bits,
            count,
        }))
    }

    /// Visit the immutable sequence without allocating payload or alignment copies.
    pub fn visit(&self, visitor: &mut dyn FnMut(u64) -> Result<()>) -> Result<()> {
        self.execution.visit(visitor)
    }

    /// Release all views, then return exclusive preparation to its original workspace.
    pub fn recycle(self, workspace: &mut Reservation) -> Result<()> {
        self.execution.recycle(workspace)
    }
}

struct Direct<'a, T: ?Sized> {
    codec: &'a T,
    data: Option<LanceBuffer>,
    bits: u64,
    count: u64,
}

impl<T: BlockDecompressor + ?Sized> UnsignedExecution for Direct<'_, T> {
    fn visit(&self, visitor: &mut dyn FnMut(u64) -> Result<()>) -> Result<()> {
        self.codec
            .visit_unsigned(self.data.as_ref(), self.bits, self.count, visitor)
    }

    fn recycle(self: Box<Self>, _workspace: &mut Reservation) -> Result<()> {
        Ok(())
    }
}
