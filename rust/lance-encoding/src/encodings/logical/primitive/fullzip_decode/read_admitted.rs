// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Read only selected FullZip rows using one already admitted workspace.

use std::{future::Future, ops::Range};

use lance_core::{
    Error, Result,
    utils::allocation::{Reservation, ReservedBuffer},
};

use super::FullZipScheduler;
use crate::decoder::DecodePageTask;

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

fn endpoint(buffer: &ReservedBuffer) -> u64 {
    let mut value = [0; 8];
    value[..buffer.capacity()].copy_from_slice(buffer.as_ref());
    u64::from_le_bytes(value)
}

impl FullZipScheduler {
    /// Read and unzip selected variable-width rows with pre-reserved credit.
    /// One small allocation is reused for both repetition-index endpoints and
    /// recycled before the data read. Only the selected row-aligned data is read;
    /// it overlaps unzip output, then is recycled before this method returns.
    /// No full-page index, coalescing gap or ordinary allocating decoder is used.
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
        mut read: F,
    ) -> Result<Option<Box<dyn DecodePageTask>>>
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
            return self.try_unzip_range_in(&[], 0, 0..0, workspace);
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
        let Some(index_buffer) = workspace.try_allocate(index.bytes_per_value as usize)? else {
            return Ok(None);
        };
        let index_buffer = fill(&mut read, first, index_buffer).await?;
        let start = endpoint(&index_buffer);
        let index_buffer = fill(&mut read, last, index_buffer).await?;
        let end = endpoint(&index_buffer);
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
        let input = fill(&mut read, range, input).await?;
        let count = rows.end - rows.start;
        let result =
            self.try_unzip_buffers_in(std::slice::from_ref(&input), count, 0..count, workspace);
        input.recycle(workspace)?;
        result
    }
}
