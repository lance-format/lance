// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Explicit dispatch for admitted byte arrays, without ordinary decoder fallback.

use std::{future::Future, ops::Range, sync::Arc};

use arrow_array::ArrayRef;
use arrow_schema::DataType;
use lance_core::{
    Result,
    utils::allocation::{Reservation, ReservedBuffer},
};

use super::{
    FullZipArrayInput, FullZipScheduler,
    constant::{ConstantArrayInput, ConstantArrayScheduler},
};
use crate::{
    compression::DecompressionStrategy,
    decoder::{PageEncoding, PageInfo},
    format::metadata::page_layout::Layout,
};

mod all_null;
use all_null::{AllNullInput, AllNullScheduler};

#[derive(Debug)]
enum Scheduler {
    AllNull(AllNullScheduler),
    FullZip(Arc<FullZipScheduler>),
    Constant(Arc<ConstantArrayScheduler>),
}

/// Metadata-only dispatch for dense FullZip and Constant byte/string pages.
/// List/LargeList nesting is supported for pages with values. Flat all-null
/// byte/string pages with metadata-only validity are also supported. Other layouts
/// return an error; this API never falls back to unadmitted decoding. Metadata is
/// not charged.
#[derive(Debug)]
pub struct AdmittedBytePageScheduler(Scheduler);

impl AdmittedBytePageScheduler {
    /// Validate the stored layout before payload reads. Use with
    /// [`Self::prepare_array_range_in`] inside an existing operation reservation.
    pub fn try_from_page(page: &PageInfo, codecs: &dyn DecompressionStrategy) -> Result<Self> {
        if let Some(scheduler) = AllNullScheduler::try_from_page(page)? {
            return Ok(Self(Scheduler::AllNull(scheduler)));
        }
        let planar;
        let layout = match &page.encoding {
            PageEncoding::Structural(layout) => Some(layout),
            PageEncoding::Planar(layout) if layout.sparse().is_none() => {
                planar = layout.dense_execution_layout()?;
                Some(&planar)
            }
            _ => None,
        };
        if let Some(Layout::ConstantLayout(layout)) =
            layout.and_then(|layout| layout.layout.as_ref())
        {
            Ok(Self(Scheduler::Constant(Arc::new(
                ConstantArrayScheduler::try_new(page, layout)?,
            ))))
        } else {
            Ok(Self(Scheduler::FullZip(Arc::new(
                FullZipScheduler::try_from_page(page, codecs)?,
            ))))
        }
    }

    /// Check the projected Arrow shape without payload I/O or allocation.
    pub fn validate_array_type(&self, data_type: &DataType) -> Result<()> {
        match &self.0 {
            Scheduler::AllNull(scheduler) => scheduler.validate_array_type(data_type),
            Scheduler::FullZip(scheduler) => scheduler.validate_array_type(data_type),
            Scheduler::Constant(scheduler) => scheduler.validate_array_type(data_type),
        }
    }

    /// Prepare one exact nonempty page-local range using caller-owned I/O buffers.
    /// `None` requires dropping/replanning the complete attempt. All input, output
    /// and assembly must use this same reservation; no additional credit is awaited.
    /// Constant list boundaries use bounded scratch, not a whole-page level cache.
    /// Transport/cache and metadata overhead are outside this payload accounting.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use arrow_schema::DataType;
    /// use lance_core::{Result, utils::allocation::Reservation};
    /// use lance_encoding::{EncodingsIo, encodings::logical::primitive::AdmittedBytePageScheduler};
    /// async fn prepare(page: &AdmittedBytePageScheduler, io: Arc<dyn EncodingsIo>,
    ///     work: &mut Reservation) -> Result<()> {
    ///     let input = page.prepare_array_range_in(0..1, DataType::Binary, work,
    ///         |range, buffer| io.submit_single_into(range, 0, buffer)).await?;
    ///     drop(input); // or move into the caller's one outer CPU decode job
    ///     Ok(())
    /// }
    /// ```
    pub async fn prepare_array_range_in<F, Fut>(
        &self,
        rows: Range<u64>,
        data_type: DataType,
        workspace: &mut Reservation,
        read: F,
    ) -> Result<Option<AdmittedBytePageInput>>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        Ok(match &self.0 {
            Scheduler::AllNull(scheduler) => Some(AdmittedBytePageInput(Input::AllNull(
                scheduler.prepare(rows, data_type)?,
            ))),
            Scheduler::FullZip(scheduler) => scheduler
                .clone()
                .prepare_array_range_in(rows, data_type, workspace, read)
                .await?
                .map(|input| AdmittedBytePageInput(Input::FullZip(input))),
            Scheduler::Constant(scheduler) => scheduler
                .prepare(rows, data_type, workspace, read)
                .await?
                .map(|input| AdmittedBytePageInput(Input::Constant(input))),
        })
    }
}

#[derive(Debug)]
enum Input {
    AllNull(AllNullInput),
    FullZip(FullZipArrayInput),
    Constant(ConstantArrayInput),
}

/// Charged page input prepared by [`AdmittedBytePageScheduler`].
/// Dropping it releases its physical buffers, including on cancellation.
#[derive(Debug)]
pub struct AdmittedBytePageInput(Input);

impl AdmittedBytePageInput {
    /// Consume input and reconstruct Arrow within its original workspace.
    /// Pure CPU work: no I/O, nested dispatch, additional admission or waits.
    /// `None` means discard/replan the whole operation, never partial success.
    pub fn decode_in(self, workspace: &mut Reservation) -> Result<Option<ArrayRef>> {
        match self.0 {
            Input::AllNull(input) => input.decode_in(workspace),
            Input::FullZip(input) => input.decode_in(workspace),
            Input::Constant(input) => input.decode_in(workspace),
        }
    }
}
