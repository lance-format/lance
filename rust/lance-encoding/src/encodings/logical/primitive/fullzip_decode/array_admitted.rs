// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Keep structural reconstruction inside the read/decode admission envelope.

use std::{future::Future, ops::Range, sync::Arc};

use arrow_array::{ArrayRef, LargeListArray, ListArray, make_array};
use arrow_schema::{DataType, FieldRef};
use lance_core::{
    Error, Result,
    utils::{
        allocation::{AllocationBudget, Reservation, ReservedBuffer},
        tokio::spawn_cpu,
    },
};

use super::FullZipScheduler;
use crate::{
    decoder::{DecodePageTask, DecodedPage},
    repdef::CompositeRepDefUnraveler,
};

/// A complete Arrow array for an exact, nonempty page-local top-level row prefix.
pub struct FullZipArrayPrefix {
    /// Advance the caller's page cursor only to this end, never the requested end.
    pub rows: Range<u64>,
    /// All owned value, offset and validity payloads keep their shared-budget charge.
    pub array: ArrayRef,
}

enum ListLayer<'a> {
    Small(&'a FieldRef),
    Large(&'a FieldRef),
}

struct Shape<'a> {
    leaf: &'a DataType,
    // Outer to inner, rebuilt in reverse after leaf validity is applied.
    lists: Vec<ListLayer<'a>>,
}

impl<'a> Shape<'a> {
    fn new(scheduler: &FullZipScheduler, data_type: &'a DataType) -> Result<Self> {
        let mut leaf = data_type;
        let mut lists = Vec::new();
        loop {
            match leaf {
                DataType::List(field) => {
                    lists.push(ListLayer::Small(field));
                    leaf = field.data_type();
                }
                DataType::LargeList(field) => {
                    lists.push(ListLayer::Large(field));
                    leaf = field.data_type();
                }
                _ => break,
            }
        }
        let offset_bits = match leaf {
            DataType::Binary | DataType::Utf8 => 32,
            DataType::LargeBinary | DataType::LargeUtf8 => 64,
            _ => {
                return Err(Error::not_supported(
                    "admitted FullZip arrays require binary/string values with optional List/LargeList nesting",
                ));
            }
        };
        let layers = &scheduler.details.def_meaning;
        if offset_bits != scheduler.bits_per_offset
            || layers.len() != lists.len() + 1
            || layers.first().is_none_or(|layer| layer.is_list())
            || layers.iter().skip(1).any(|layer| !layer.is_list())
        {
            return Err(Error::invalid_input(
                "FullZip Arrow type does not match page offset width or structural layers",
            ));
        }
        Ok(Self { leaf, lists })
    }

    fn materialize(
        &self,
        page: DecodedPage,
        workspace: &mut Reservation,
    ) -> Result<Option<ArrayRef>> {
        let Some(leaf) = page.data.try_to_arrow_in(self.leaf, workspace)? else {
            return Ok(None);
        };
        let mut levels = CompositeRepDefUnraveler::new(vec![page.repdef]);
        let Some(validity) = levels.try_unravel_validity_in(leaf.len(), workspace)? else {
            return Ok(None);
        };
        // Only supported binary/string leaves reach this path. Their pinned Arrow
        // validation and these List constructors borrow buffers without payload
        // scratch. FSL/struct reconstruction has different overlap requirements.
        let mut array = make_array(leaf.into_builder().nulls(validity).build()?);
        for layer in self.lists.iter().rev() {
            array = match layer {
                ListLayer::Small(field) => {
                    let Some((offsets, validity)) =
                        levels.try_unravel_offsets_in::<i32>(workspace)?
                    else {
                        return Ok(None);
                    };
                    Arc::new(ListArray::try_new(
                        (*field).clone(),
                        offsets,
                        array,
                        validity,
                    )?)
                }
                ListLayer::Large(field) => {
                    let Some((offsets, validity)) =
                        levels.try_unravel_offsets_in::<i64>(workspace)?
                    else {
                        return Ok(None);
                    };
                    Arc::new(LargeListArray::try_new(
                        (*field).clone(),
                        offsets,
                        array,
                        validity,
                    )?)
                }
            };
        }
        Ok(Some(array))
    }
}

impl Shape<'_> {
    fn decode(
        &self,
        task: Box<dyn DecodePageTask>,
        count: u64,
        workspace: &mut Reservation,
    ) -> Result<Option<ArrayRef>> {
        let page = task.try_decode_in(workspace)?;
        drop(task);
        let Some(page) = page else {
            return Ok(None);
        };
        let Some(array) = self.materialize(page, workspace)? else {
            return Ok(None);
        };
        if array.len() as u64 != count {
            return Err(Error::corrupt_file_named(
                "FullZip",
                "reconstructed Arrow row count differs from selected page rows",
            ));
        }
        Ok(Some(array))
    }
}

pub(super) struct ArrayDecodeJob {
    scheduler: Arc<FullZipScheduler>,
    input: ReservedBuffer,
    data_type: DataType,
    count: u64,
    workspace: Reservation,
}

impl ArrayDecodeJob {
    pub(super) fn decode(mut self) -> Result<(Reservation, Option<ArrayRef>)> {
        let task = self.scheduler.try_unzip_buffers_in(
            std::slice::from_ref(&self.input),
            self.count,
            0..self.count,
            &mut self.workspace,
        );
        self.input.recycle(&mut self.workspace)?;
        let array = match task? {
            Some(task) => Shape::new(&self.scheduler, &self.data_type)?.decode(
                task,
                self.count,
                &mut self.workspace,
            )?,
            None => None,
        };
        Ok((self.workspace, array))
    }
}

impl FullZipScheduler {
    /// Read one exact nonempty row range and reconstruct it within an existing
    /// reservation. Unlike the prefix API, this never shrinks the selection or
    /// acquires more credit. A caller composing columns can admit one complete
    /// workspace, keep every column aligned, then retry the whole batch smaller.
    ///
    /// `None` is refusal, not EOF. Drop the entire compound attempt (including
    /// previously returned columns) and its reservation before retrying. A failed
    /// reconstruction may release frozen intermediate credit to the shared pool
    /// instead of restoring it to this exclusive reservation. Errors propagate;
    /// no ordinary decode fallback or dependent allocation wait is introduced.
    /// The supported types and I/O ownership contract match the prefix API.
    ///
    /// ```
    /// use std::ops::Range;
    /// use arrow_array::ArrayRef;
    /// use arrow_schema::DataType;
    /// use lance_core::{Result, utils::allocation::Reservation};
    /// use lance_encoding::{EncodingsIo,
    ///     encodings::logical::primitive::FullZipScheduler};
    /// async fn column(decoder: &FullZipScheduler, io: &dyn EncodingsIo,
    ///     rows: Range<u64>, ty: &DataType, work: &mut Reservation)
    ///     -> Result<Option<ArrayRef>> {
    ///     decoder.try_read_array_range_in(rows, ty, work,
    ///         |range, output| io.submit_single_into(range, 0, output)).await
    /// }
    /// ```
    pub async fn try_read_array_range_in<F, Fut>(
        &self,
        rows: Range<u64>,
        data_type: &DataType,
        workspace: &mut Reservation,
        read: F,
    ) -> Result<Option<ArrayRef>>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        let shape = Shape::new(self, data_type)?;
        if rows.start >= rows.end || rows.end > self.rows_in_page {
            return Err(Error::invalid_input(
                "FullZip array requires a nonempty range within the page",
            ));
        }
        let count = rows.end - rows.start;
        let Some(task) = self.try_read_unzip_range_in(rows, workspace, read).await? else {
            return Ok(None);
        };
        shape.decode(task, count, workspace)
    }

    /// Read an exact range asynchronously, then unzip, decode and reconstruct
    /// Arrow output on the CPU pool. Unlike [`Self::try_read_array_range_in`],
    /// this takes ownership of the workspace and returns its remainder.
    ///
    /// Cancellation after dispatch does not abort CPU work. Input and unused
    /// credit belong to the CPU job until it finishes; output aliases retain
    /// their own charge. No allocation wait or smaller-range retry occurs here.
    /// The read callback has the same owned-buffer contract as the inline API.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use arrow_schema::DataType;
    /// use lance_core::{Result, utils::allocation::Reservation};
    /// use lance_encoding::{EncodingsIo,
    ///     encodings::logical::primitive::FullZipScheduler};
    /// async fn column(decoder: Arc<FullZipScheduler>, io: &dyn EncodingsIo,
    ///     work: Reservation) -> Result<()> {
    ///     let (remaining, array) = decoder.try_read_array_range_on_cpu(
    ///         0..1, DataType::Binary, work,
    ///         |range, output| io.submit_single_into(range, 0, output)).await?;
    ///     drop((remaining, array));
    ///     Ok(())
    /// }
    /// ```
    pub async fn try_read_array_range_on_cpu<F, Fut>(
        self: Arc<Self>,
        rows: Range<u64>,
        data_type: DataType,
        workspace: Reservation,
        read: F,
    ) -> Result<(Reservation, Option<ArrayRef>)>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        self.read_array_range_with(rows, data_type, workspace, read, |job| {
            spawn_cpu(move || job.decode())
        })
        .await
    }

    // Keep the owned handoff separate so queued cancellation can be tested
    // deterministically without blocking a CPU worker or racing a timer.
    pub(super) async fn read_array_range_with<F, Fut, Run, Done>(
        self: Arc<Self>,
        rows: Range<u64>,
        data_type: DataType,
        mut workspace: Reservation,
        read: F,
        dispatch: Run,
    ) -> Result<(Reservation, Option<ArrayRef>)>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
        Run: FnOnce(ArrayDecodeJob) -> Done,
        Done: Future<Output = Result<(Reservation, Option<ArrayRef>)>>,
    {
        Shape::new(&self, &data_type)?;
        if rows.start >= rows.end || rows.end > self.rows_in_page {
            return Err(Error::invalid_input(
                "FullZip array requires a nonempty range within the page",
            ));
        }
        let count = rows.end - rows.start;
        let Some(input) = self.read_selected_range(rows, &mut workspace, read).await? else {
            return Ok((workspace, None));
        };
        dispatch(ArrayDecodeJob {
            scheduler: self,
            input,
            data_type,
            count,
            workspace,
        })
        .await
    }

    /// Validate the Arrow shape and structural unzip metadata without I/O.
    /// This does not validate payload framing or data-dependent codec support.
    /// File-level cursors use this before returning any batch from a selection.
    ///
    /// ```
    /// use arrow_schema::DataType;
    /// use lance_core::Result;
    /// use lance_encoding::encodings::logical::primitive::FullZipScheduler;
    /// fn validate(decoder: &FullZipScheduler, ty: &DataType) -> Result<()> {
    ///     decoder.validate_array_type(ty)
    /// }
    /// ```
    pub fn validate_array_type(&self, data_type: &DataType) -> Result<()> {
        Shape::new(self, data_type)?;
        self.plan_unzip_range(&[], 0, 0..0)?;
        Ok(())
    }

    /// Read a fitting prefix and finish its Arrow reconstruction in one admission.
    ///
    /// Supports Binary/Utf8/LargeBinary/LargeUtf8 leaves nested in List/LargeList.
    /// The supplied type must describe the whole raw leaf path, not a projection
    /// or a scalarized list. Unsupported/mismatched shapes fail before I/O. Reader
    /// projections, filters, struct/multi-column alignment and CPU dispatch are
    /// not implemented by this low-level API; ordinary scans are unchanged.
    ///
    /// Input, unzip, codec, rep/def, typed values, validity and list offsets share
    /// the same `workspace_bytes` reservation. Reconstruction refusal drops all
    /// intermediate owners before retrying a smaller prefix. It never reacquires
    /// credit while holding intermediate data. `None` is refusal, not EOF; keep the
    /// cursor and drain/replan. Errors are propagated without prefix retry.
    /// Metadata, transport/cache and allocator overhead remain outside accounting.
    /// Repeated reads and inline CPU have not been performance qualified.
    ///
    /// ```
    /// use std::{ops::Range, sync::Arc};
    /// use arrow_schema::DataType;
    /// use lance_core::{Result, utils::allocation::AllocationBudget};
    /// use lance_encoding::{EncodingsIo,
    ///     encodings::logical::primitive::{FullZipArrayPrefix, FullZipScheduler}};
    /// async fn next(scheduler: &FullZipScheduler, io: &dyn EncodingsIo,
    ///     rows: Range<u64>, ty: &DataType, budget: &Arc<AllocationBudget>,
    ///     priority: u64) -> Result<Option<FullZipArrayPrefix>> {
    ///     scheduler.try_read_array_prefix_in(rows, ty, budget, 65536,
    ///         |range, output| io.submit_single_into(range, priority, output)).await
    /// }
    /// ```
    pub async fn try_read_array_prefix_in<F, Fut>(
        &self,
        rows: Range<u64>,
        data_type: &DataType,
        budget: &Arc<AllocationBudget>,
        workspace_bytes: usize,
        read: F,
    ) -> Result<Option<FullZipArrayPrefix>>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        let shape = Shape::new(self, data_type)?;
        let Some((rows, array)) = self
            .try_read_prefix_in(rows, budget, workspace_bytes, read, |page, workspace| {
                shape.materialize(page, workspace)
            })
            .await?
        else {
            return Ok(None);
        };
        if array.len() as u64 != rows.end - rows.start {
            return Err(Error::corrupt_file_named(
                "FullZip",
                "reconstructed Arrow row count differs from selected page rows",
            ));
        }
        Ok(Some(FullZipArrayPrefix { rows, array }))
    }
}
