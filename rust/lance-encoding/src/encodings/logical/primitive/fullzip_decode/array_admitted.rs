// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Keep structural reconstruction inside the read/decode admission envelope.

pub mod batch;
mod flat;
mod list;

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

pub(in super::super) struct Shape<'a> {
    pub(in super::super) leaf: &'a DataType,
    // Outer to inner, rebuilt in reverse after leaf validity is applied.
    lists: Vec<ListLayer<'a>>,
}

impl<'a> Shape<'a> {
    fn new(scheduler: &FullZipScheduler, data_type: &'a DataType) -> Result<Self> {
        Self::from_layers(
            data_type,
            &scheduler.details.def_meaning,
            Some(scheduler.bits_per_offset),
        )
    }

    pub(in super::super) fn from_layers(
        data_type: &'a DataType,
        layers: &[crate::repdef::DefinitionInterpretation],
        bits: Option<u8>,
    ) -> Result<Self> {
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
        if bits.is_some_and(|bits| offset_bits != bits)
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
        self.finish(page, count, workspace)
    }

    pub(in super::super) fn finish(
        &self,
        page: DecodedPage,
        count: u64,
        workspace: &mut Reservation,
    ) -> Result<Option<ArrayRef>> {
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

/// Charged input for one exact FullZip row range, ready for CPU decoding.
/// Construct with [`FullZipScheduler::prepare_array_range_in`]. Several columns
/// may be read into the same workspace before dispatching one pure CPU job.
/// Dropping this input refunds its physical charge, including after cancellation.
#[derive(Debug)]
pub struct FullZipArrayInput {
    scheduler: Arc<FullZipScheduler>,
    input: ReservedBuffer,
    data_type: DataType,
    count: u64,
}

impl FullZipArrayInput {
    // Count repeated byte/level passes and per-row work, including Arrow
    // reconstruction. Restrict this heuristic to shallow identity codecs: a
    // small compressed input does not imply a small decode. This is neither an
    // allocation plan nor permission to skip full payload validation.
    pub(in super::super) fn decode_work_estimate(&self) -> Option<usize> {
        let super::PerValueDecompressor::Variable(codec) =
            &self.scheduler.details.value_decompressor
        else {
            return None;
        };
        let depth = self.scheduler.details.def_meaning.len();
        if !codec.is_identity() || depth > 3 || self.count > 256 {
            return None;
        }
        self.input
            .capacity()
            .checked_mul(4 * depth)
            .and_then(|bytes| bytes.checked_add(self.count as usize * 32))
            .and_then(|bytes| bytes.checked_add(4096))
    }

    /// Consume input and decode using the remainder of its original workspace.
    /// Performs no I/O, pool dispatch or allocation wait. `None` means the entire
    /// compound attempt must be dropped and replanned, never EOF. Flat raw byte
    /// arrays materialize offsets, values and optional validity in one allocation,
    /// without intermediate definition levels. One List/LargeList layer over raw
    /// bytes similarly uses only the final parent and child allocations, without
    /// dense repetition/definition levels. These raw paths initialize metadata
    /// before copying values once; only complete allocations become Arrow buffers.
    /// Deeper/compressed shapes retain the
    /// general admitted decoder. Consumed unzip scratch returns to this workspace
    /// only after its last shared view is gone; output aliases retain their full
    /// backing allocation.
    /// Errors consume input; previously decoded sibling columns remain the
    /// caller's responsibility. See [`FullZipScheduler::prepare_array_range_in`].
    pub fn decode_in(self, workspace: &mut Reservation) -> Result<Option<ArrayRef>> {
        if matches!(
            &self.scheduler.details.value_decompressor,
            super::PerValueDecompressor::Variable(codec)
                if codec.supports_direct_byte_materialization()
        ) {
            let direct = match &self.data_type {
                DataType::Binary | DataType::Utf8 | DataType::LargeBinary | DataType::LargeUtf8 => {
                    Shape::new(&self.scheduler, &self.data_type)?;
                    Some(flat::decode(&self, workspace))
                }
                DataType::List(field) | DataType::LargeList(field)
                    if matches!(
                        field.data_type(),
                        DataType::Binary
                            | DataType::Utf8
                            | DataType::LargeBinary
                            | DataType::LargeUtf8
                    ) =>
                {
                    Shape::new(&self.scheduler, &self.data_type)?;
                    Some(list::decode(&self, workspace))
                }
                _ => None,
            };
            if let Some(array) = direct {
                self.input.recycle(workspace)?;
                return array;
            }
        }
        let task = self.scheduler.try_unzip_owned_in(
            std::slice::from_ref(&self.input),
            self.count,
            0..self.count,
            workspace,
        );
        self.input.recycle(workspace)?;
        let array = match task? {
            Some(task) => match task.decode(workspace)? {
                Some(page) => Shape::new(&self.scheduler, &self.data_type)?
                    .finish(page, self.count, workspace)?,
                None => None,
            },
            None => None,
        };
        Ok(array)
    }
}

pub(super) struct ArrayDecodeJob {
    input: FullZipArrayInput,
    workspace: Reservation,
}

impl ArrayDecodeJob {
    pub(super) fn decode(mut self) -> Result<(Reservation, Option<ArrayRef>)> {
        let array = self.input.decode_in(&mut self.workspace)?;
        Ok((self.workspace, array))
    }
}

impl FullZipScheduler {
    /// Read an exact nonempty row range into charged input without dispatching
    /// CPU work. This separates asynchronous I/O from batch-wide CPU decoding.
    /// Use the same workspace for every column's input, unzip, codec and Arrow
    /// reconstruction; all input remains charged while other columns are read.
    /// `None` is admission refusal. Drop all prepared inputs and the remainder
    /// before replanning; never wait for more credit while holding this attempt.
    /// Shape/range errors precede I/O; payload validation occurs during decode.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use arrow_schema::DataType;
    /// use lance_core::{Result, utils::{allocation::Reservation, tokio::spawn_cpu}};
    /// use lance_encoding::{EncodingsIo,
    ///     encodings::logical::primitive::FullZipScheduler};
    /// async fn decode(scheduler: Arc<FullZipScheduler>, io: &dyn EncodingsIo,
    ///     mut work: Reservation) -> Result<()> {
    ///     let input = scheduler.prepare_array_range_in(0..1, DataType::Binary,
    ///         &mut work, |range, out| io.submit_single_into(range, 0, out)).await?;
    ///     if let Some(input) = input {
    ///         let array = spawn_cpu(move || input.decode_in(&mut work)).await?;
    ///         drop(array); // consume before admitting the next operation
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub async fn prepare_array_range_in<F, Fut>(
        self: Arc<Self>,
        rows: Range<u64>,
        data_type: DataType,
        workspace: &mut Reservation,
        read: F,
    ) -> Result<Option<FullZipArrayInput>>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        Shape::new(&self, &data_type)?;
        if rows.start >= rows.end || rows.end > self.rows_in_page {
            return Err(Error::invalid_input(
                "FullZip array requires a nonempty range within the page",
            ));
        }
        let count = rows.end - rows.start;
        let Some(input) = self.read_selected_range(rows, workspace, read).await? else {
            return Ok(None);
        };
        Ok(Some(FullZipArrayInput {
            scheduler: self,
            input,
            data_type,
            count,
        }))
    }

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
    /// Raw values reuse this job's compact, admitted unzip allocation. Its full
    /// charge (including structural streams) follows every output alias; generic
    /// task decoding still detaches values from arbitrary input owners.
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
        let Some(input) = self
            .prepare_array_range_in(rows, data_type, &mut workspace, read)
            .await?
        else {
            return Ok((workspace, None));
        };
        dispatch(ArrayDecodeJob { input, workspace }).await
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
