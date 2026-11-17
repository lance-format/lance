// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{future::Future, ops::Range, sync::Arc};

use arrow_array::{RecordBatch, UInt32Array};
use arrow_buffer::{Buffer, ScalarBuffer};
use lance_core::{
    Error, Result,
    utils::allocation::{AllocationBudget, Reservation},
};

use super::FragmentReader;

/// A complete physical prefix, a retryable refusal, or the end of a fragment scan.
#[derive(Debug)]
pub enum AdmittedFragmentScanOutcome {
    /// The complete result for these physical offsets. Deletion filtering may
    /// remove some or all output rows; even an empty batch consumes this prefix.
    Batch {
        rows: Range<u64>,
        batch: RecordBatch,
    },
    /// No position was committed. Drain independently held output or replan the
    /// workspace before retrying. This is never EOF or a total-result limit.
    NeedsCapacity,
    /// Every requested physical offset has been consumed.
    Finished,
}

/// Opt-in fragment cursor with charged per-batch selection and no read-ahead.
///
/// The reader's projection, overlays, deletion and system-column snapshots remain
/// fixed for the cursor lifetime. Decoder support and budget exclusions are those
/// of [`FragmentReader::try_take_admitted_fullzip`]. This does not install a default
/// scanner path or bound metadata, transport/cache, allocator overhead or RSS.
pub struct AdmittedFragmentScan<'a> {
    reader: &'a FragmentReader,
    cursor: u64,
    end: u64,
    max_rows: u64,
    workspace_bytes: usize,
    budget: Arc<AllocationBudget>,
}

impl FragmentReader {
    /// Scan a physical range through the admitted fragment take path.
    ///
    /// Each attempt atomically reserves `workspace_bytes` before generating its
    /// physical offsets. Those offsets, dependent input, decode scratch and output
    /// share that reservation. A refused attempt drops completely before retrying
    /// a smaller prefix, down to one row. No whole-fragment selection is built.
    /// The row ceiling is a per-batch candidate, not a total-result cap.
    ///
    /// `try_next` never waits for allocation credit. Consume/drop returned output
    /// before requesting the next batch, or provide an independent drain path.
    /// `NeedsCapacity` also covers a minimal row whose operation cannot fit the
    /// configured workspace; waiting for a refund alone cannot resolve that case.
    /// Unsupported codecs and errors propagate without skipping physical rows.
    /// Cancellation leaves the cursor unchanged; work already dispatched to I/O
    /// or CPU retains its charge until it actually ends.
    ///
    /// ```
    /// use lance::dataset::fragment::{FragmentReader, AdmittedFragmentScanOutcome};
    /// use lance_core::{Result, utils::allocation::AllocationBudget};
    /// // Some(offset) means incomplete: resume there after drain/replanning.
    /// async fn consume_until_blocked(reader: &FragmentReader, rows: std::ops::Range<u64>)
    ///     -> Result<Option<u64>> {
    ///     let budget = AllocationBudget::new(65536);
    ///     let mut scan = reader.try_scan_admitted_fullzip(rows, 256, 65536, budget)?;
    ///     loop {
    ///         match scan.try_next().await? {
    ///             AdmittedFragmentScanOutcome::Batch { batch, .. } => drop(batch),
    ///             AdmittedFragmentScanOutcome::NeedsCapacity => {
    ///                 return Ok(Some(scan.position()));
    ///             }
    ///             AdmittedFragmentScanOutcome::Finished => return Ok(None),
    ///         }
    ///     }
    /// }
    /// ```
    pub fn try_scan_admitted_fullzip(
        &self,
        rows: Range<u64>,
        max_rows: u32,
        workspace_bytes: usize,
        budget: Arc<AllocationBudget>,
    ) -> Result<AdmittedFragmentScan<'_>> {
        if rows.start > rows.end
            || rows.end > self.num_physical_rows as u64
            || rows.end > u64::from(u32::MAX) + 1
        {
            return Err(Error::invalid_input(format!(
                "admitted fragment scan range {rows:?} exceeds physical domain {}",
                self.num_physical_rows
            )));
        }
        if max_rows == 0 || workspace_bytes == 0 {
            return Err(Error::invalid_input(
                "admitted fragment scan requires positive max_rows and workspace_bytes",
            ));
        }
        Ok(AdmittedFragmentScan {
            reader: self,
            cursor: rows.start,
            end: rows.end,
            max_rows: u64::from(max_rows),
            workspace_bytes,
            budget,
        })
    }
}

impl AdmittedFragmentScan<'_> {
    /// Next unconsumed physical offset. Errors, cancellation and refusals preserve it.
    pub fn position(&self) -> u64 {
        self.cursor
    }

    /// Attempt the next complete prefix, without waiting for credit or reading ahead.
    pub async fn try_next(&mut self) -> Result<AdmittedFragmentScanOutcome> {
        Ok(self
            .try_next_admitted()
            .await?
            .unwrap_or(AdmittedFragmentScanOutcome::NeedsCapacity))
    }

    // None is exclusively a failed pool admission. An admitted minimal prefix
    // that cannot fit returns Some(NeedsCapacity), which must not wait for credit.
    pub(crate) async fn try_next_admitted(
        &mut self,
    ) -> Result<Option<AdmittedFragmentScanOutcome>> {
        let reader = self.reader;
        self.try_next_with(|indices, workspace| async move {
            reader.try_take_admitted_fullzip(&indices, workspace).await
        })
        .await
    }

    // Own the handoff so cancellation tests can suspend an attempt without
    // blocking a CPU worker. Production uses the actual fragment take above.
    pub(in crate::dataset::fragment) async fn try_next_with<Run, Done>(
        &mut self,
        mut take: Run,
    ) -> Result<Option<AdmittedFragmentScanOutcome>>
    where
        Run: FnMut(UInt32Array, Reservation) -> Done,
        Done: Future<Output = Result<Option<RecordBatch>>>,
    {
        if self.cursor == self.end {
            return Ok(Some(AdmittedFragmentScanOutcome::Finished));
        }
        let mut count = self
            .max_rows
            .min(self.end - self.cursor)
            .min((self.workspace_bytes / size_of::<u32>()).max(1) as u64);
        loop {
            let Some(mut workspace) = self.budget.try_reserve(self.workspace_bytes) else {
                return Ok(None);
            };
            let bytes = usize::try_from(count)
                .ok()
                .and_then(|n| n.checked_mul(size_of::<u32>()))
                .ok_or_else(|| Error::invalid_input("physical selection exceeds address space"))?;
            let batch = if let Some(mut storage) = workspace.try_allocate(bytes)? {
                for (index, slot) in storage
                    .as_mut()
                    .as_chunks_mut::<4>()
                    .0
                    .iter_mut()
                    .enumerate()
                {
                    // The constructor admits at most the complete u32 domain.
                    let row = (self.cursor + index as u64) as u32;
                    slot.copy_from_slice(&row.to_ne_bytes());
                }
                let indices = UInt32Array::new(
                    ScalarBuffer::new(Buffer::from(storage.freeze()), 0, count as usize),
                    None,
                );
                take(indices, workspace).await?
            } else {
                drop(workspace);
                None
            };
            if let Some(batch) = batch {
                let rows = self.cursor..self.cursor + count;
                // No await follows the commit. Empty filtered output is still a
                // fully consumed physical prefix, not EOF or a retry.
                self.cursor = rows.end;
                return Ok(Some(AdmittedFragmentScanOutcome::Batch { rows, batch }));
            }
            if count == 1 {
                return Ok(Some(AdmittedFragmentScanOutcome::NeedsCapacity));
            }
            count /= 2;
        }
    }
}
