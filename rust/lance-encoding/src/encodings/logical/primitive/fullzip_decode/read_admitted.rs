// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Read only selected FullZip rows using one already admitted workspace.

use std::{future::Future, ops::Range, sync::Arc};

use lance_core::{
    Error, Result,
    utils::allocation::{AllocationBudget, Reservation, ReservedBuffer},
};

use super::FullZipScheduler;
use crate::decoder::{DecodePageTask, DecodedPage};

/// A successfully decoded, nonempty prefix of requested top-level rows.
///
/// See [`FullZipScheduler::try_read_decode_prefix_in`] for bounded retries and
/// a usage example. The values and levels keep their allocation charges until
/// their final owners drop; Arrow/structural reconstruction needs separate credit.
pub struct DecodedFullZipPrefix {
    /// Exact page-row interval represented by `page`; advance only to this end.
    pub rows: Range<u64>,
    /// Decoded leaf values and repetition/definition levels for those rows.
    pub page: DecodedPage,
}

async fn fill<F, Fut>(
    read: &mut F,
    range: Range<u64>,
    output: ReservedBuffer,
) -> Result<ReservedBuffer>
where
    F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
    Fut: Future<Output = Result<ReservedBuffer>>,
{
    let address = output.as_ref().as_ptr() as usize;
    let len = output.capacity();
    let output = read(range, output).await?;
    if output.as_ref().as_ptr() as usize != address || output.capacity() != len {
        return Err(Error::invalid_input(
            "FullZip reader must return the supplied allocation unchanged",
        ));
    }
    Ok(output)
}

fn endpoint(buffer: &[u8]) -> u64 {
    let mut value = [0; 8];
    value[..buffer.len()].copy_from_slice(buffer);
    u64::from_le_bytes(value)
}

impl FullZipScheduler {
    /// Read and decode a fitting prefix without keeping failed attempts alive.
    /// Each attempt reserves `workspace_bytes` atomically from the shared budget.
    /// Admission refusal at input, unzip or codec output releases the entire
    /// attempt before halving its row count. Corruption/reader/unsupported errors
    /// propagate without retrying a smaller prefix that could hide bad input.
    ///
    /// `Some` always makes progress and identifies exactly which rows succeeded.
    /// `None` means current credit is unavailable or even the one-row attempt
    /// cannot fit this workspace; it is never EOF. Keep the original cursor and
    /// drain outstanding results or replan. No waiting or credit increase occurs
    /// while holding inputs. This is a finite size search, not a fairness policy.
    /// A generous workspace keeps the requested batch intact rather than forcing
    /// one-row output. Repeated reads/decodes can cost CPU/I/O; performance is not
    /// qualified by this API's payload admission guarantee.
    ///
    /// `rows` must be nonempty and within this page. On success only the returned
    /// values/levels retain credit; input, unzip and unused workspace are released.
    /// Read callbacks obey [`Self::try_read_unzip_range_in`]'s ownership contract.
    /// Transport/cache, metadata and later materialization are still separate.
    ///
    /// ```
    /// use std::{future::Future, ops::Range, sync::Arc};
    /// use lance_core::{Result, utils::allocation::{AllocationBudget, ReservedBuffer}};
    /// use lance_encoding::encodings::logical::primitive::{FullZipScheduler, DecodedFullZipPrefix};
    /// async fn next<F, Fut>(scheduler: &FullZipScheduler, remaining: Range<u64>,
    ///     budget: &Arc<AllocationBudget>, read: F) -> Result<Option<DecodedFullZipPrefix>>
    /// where F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
    ///       Fut: Future<Output = Result<ReservedBuffer>> {
    ///     // Advance remaining.start only to a returned prefix's rows.end.
    ///     scheduler.try_read_decode_prefix_in(remaining, budget, 65536, read).await
    /// }
    /// ```
    pub async fn try_read_decode_prefix_in<F, Fut>(
        &self,
        rows: Range<u64>,
        budget: &Arc<AllocationBudget>,
        workspace_bytes: usize,
        read: F,
    ) -> Result<Option<DecodedFullZipPrefix>>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        Ok(self
            .try_read_prefix_in(rows, budget, workspace_bytes, read, |page, _| {
                Ok(Some(page))
            })
            .await?
            .map(|(rows, page)| DecodedFullZipPrefix { rows, page }))
    }

    // All dependent stages share one admission. In particular a caller cannot
    // retain decoded inputs, release the headroom, then wait to admit Arrow output.
    pub(super) async fn try_read_prefix_in<F, Fut, T>(
        &self,
        rows: Range<u64>,
        budget: &Arc<AllocationBudget>,
        workspace_bytes: usize,
        mut read: F,
        mut finish: impl FnMut(DecodedPage, &mut Reservation) -> Result<Option<T>>,
    ) -> Result<Option<(Range<u64>, T)>>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        self.plan_unzip_range(&[], 0, 0..0)?;
        if rows.start >= rows.end || rows.end > self.rows_in_page || workspace_bytes == 0 {
            return Err(Error::invalid_input(
                "FullZip prefix requires nonempty page rows and workspace",
            ));
        }
        let mut candidate = rows;
        loop {
            let Some(mut workspace) = budget.try_reserve(workspace_bytes) else {
                return Ok(None);
            };
            let page = if let Some(task) = self
                .try_read_unzip_range_in(candidate.clone(), &mut workspace, &mut read)
                .await?
            {
                task.try_decode_in(&mut workspace)?
            } else {
                None
            };
            let output = match page {
                Some(page) => finish(page, &mut workspace)?,
                None => None,
            };
            // Task, decoded page and any refused reconstruction are gone before
            // retry. Only a successful output keeps an independent payload owner.
            drop(workspace);
            if let Some(output) = output {
                return Ok(Some((candidate, output)));
            }
            let count = candidate.end - candidate.start;
            if count == 1 {
                return Ok(None);
            }
            candidate.end = candidate.start + count / 2;
        }
    }

    /// Read and unzip selected variable-width rows with pre-reserved credit.
    /// Repetition-index endpoints are read together when their span is at most
    /// 4 KiB and fits the existing workspace. Otherwise one entry allocation is
    /// reused for two reads. Index scratch is recycled before the data read.
    /// Only the selected row-aligned data is read;
    /// it overlaps unzip output, then is recycled before this method returns.
    /// Index coalescing gaps are charged in full; no ordinary allocating decoder
    /// or unbounded full-page index load is used.
    ///
    /// `read` must fill the supplied allocation exactly and return it unchanged,
    /// without retaining aliases. It must keep the allocation alive until any
    /// underlying I/O has stopped using it, including on cancellation. An adapter
    /// can use `lance_io::traits::Reader::get_range_into`. Transport/cache memory
    /// inside the reader remains outside this application-payload budget.
    ///
    /// `None` releases inputs and restores this operation's workspace credit.
    /// The caller must shrink/drain/replan; this method never waits for credit.
    /// Reserve enough for the maximum of one index entry and selected input plus
    /// unzip output. Later codec decode needs separate output headroom alongside
    /// the retained task. Error/cancellation drops in-flight allocations, refunding
    /// the shared budget; drop the remaining workspace before retrying.
    /// Fixed-width values and structural projections remain unsupported here.
    ///
    /// ```
    /// use std::{future::Future, ops::Range};
    /// use lance_core::{Result, utils::allocation::{Reservation, ReservedBuffer}};
    /// use lance_encoding::{decoder::DecodePageTask,
    ///     encodings::logical::primitive::FullZipScheduler};
    /// async fn load<F, Fut>(scheduler: &FullZipScheduler, work: &mut Reservation,
    ///     read: F) -> Result<Option<Box<dyn DecodePageTask>>>
    /// where F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
    ///       Fut: Future<Output = Result<ReservedBuffer>> {
    ///     scheduler.try_read_unzip_range_in(0..1, work, read).await
    /// }
    /// ```
    pub async fn try_read_unzip_range_in<F, Fut>(
        &self,
        rows: Range<u64>,
        workspace: &mut Reservation,
        read: F,
    ) -> Result<Option<Box<dyn DecodePageTask>>>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        let Some(input) = self
            .read_selected_range(rows.clone(), workspace, read)
            .await?
        else {
            return Ok(None);
        };
        let count = rows.end - rows.start;
        let result =
            self.try_unzip_buffers_in(std::slice::from_ref(&input), count, 0..count, workspace);
        input.recycle(workspace)?;
        result
    }

    // Only I/O and index endpoint validation belong on the async executor.
    // The returned exclusive input can move with its workspace to a CPU job.
    pub(super) async fn read_selected_range<F, Fut>(
        &self,
        rows: Range<u64>,
        workspace: &mut Reservation,
        mut read: F,
    ) -> Result<Option<ReservedBuffer>>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        // Validate schema/codec metadata before any read, even on zero credit.
        self.plan_unzip_range(&[], 0, 0..0)?;
        if rows.start > rows.end || rows.end > self.rows_in_page {
            return Err(Error::invalid_input(
                "FullZip read rows are outside the page",
            ));
        }
        if rows.is_empty() {
            return workspace.try_allocate(0);
        }
        let index = self.rep_index.ok_or_else(|| {
            Error::invalid_input("variable FullZip read requires a repetition index")
        })?;
        if !matches!(index.bytes_per_value, 1 | 2 | 4 | 8) {
            return Err(Error::invalid_input(
                "invalid FullZip repetition index width",
            ));
        }
        let index_range = |row: u64| -> Result<Range<u64>> {
            let start = row
                .checked_mul(index.bytes_per_value)
                .and_then(|n| index.buf_position.checked_add(n));
            start
                .and_then(|start| {
                    start
                        .checked_add(index.bytes_per_value)
                        .map(|end| start..end)
                })
                .ok_or_else(|| Error::invalid_input("FullZip repetition index address overflows"))
        };
        let first = index_range(rows.start)?;
        let last = index_range(rows.end)?;
        self.data_buf_position
            .checked_add(self.data_buf_size)
            .ok_or_else(|| Error::invalid_input("FullZip data address overflows"))?;
        let width = index.bytes_per_value as usize;
        // Avoid a second storage round trip for nearby endpoints. The fixed
        // cap bounds over-read independently of page/selection size; using only
        // existing workspace preserves progress at the original minimum budget.
        const MAX_INDEX_SPAN: u64 = 4096;
        let span = last.end - first.start;
        let combined = if span <= MAX_INDEX_SPAN {
            workspace.try_allocate(span as usize)?
        } else {
            None
        };
        let (start, end, index_buffer) = if let Some(index_buffer) = combined {
            let index_buffer = fill(&mut read, first.start..last.end, index_buffer).await?;
            let bytes = index_buffer.as_ref();
            (
                endpoint(&bytes[..width]),
                endpoint(&bytes[bytes.len() - width..]),
                index_buffer,
            )
        } else {
            let Some(index_buffer) = workspace.try_allocate(width)? else {
                return Ok(None);
            };
            let index_buffer = fill(&mut read, first, index_buffer).await?;
            let start = endpoint(index_buffer.as_ref());
            let index_buffer = fill(&mut read, last, index_buffer).await?;
            (start, endpoint(index_buffer.as_ref()), index_buffer)
        };
        index_buffer.recycle(workspace)?;
        if start >= end
            || end > self.data_buf_size
            || (rows.start == 0 && start != 0)
            || (rows.end == self.rows_in_page && end != self.data_buf_size)
        {
            return Err(Error::corrupt_file_named(
                "FullZip",
                "invalid repetition index endpoints",
            ));
        }
        let len = usize::try_from(end - start)
            .map_err(|_| Error::invalid_input("FullZip input exceeds addressable size"))?;
        let Some(input) = workspace.try_allocate(len)? else {
            return Ok(None);
        };
        // The page-end check above bounds both additions without overflow.
        let range = (self.data_buf_position + start)..(self.data_buf_position + end);
        fill(&mut read, range, input).await.map(Some)
    }
}
