// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Demand-driven FullZip cursor with one admission for all projected columns.

use std::{collections::HashSet, future::Future, ops::Range, sync::Arc};

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Schema as ArrowSchema};
use lance_core::{
    Error, Result,
    utils::{
        allocation::{AllocationBudget, Reservation},
        deletion::DeletionVector,
        tokio::spawn_cpu,
    },
};
use lance_encoding::{
    EncodingsIo,
    compression::DefaultDecompressionStrategy,
    decoder::ColumnInfo,
    encodings::logical::primitive::{FullZipArrayInput, FullZipScheduler},
};

use super::FileReader;
use crate::version::ConcreteFileVersion;

/// One step of an admitted file cursor. Refusal and EOF are deliberately distinct.
#[derive(Debug)]
pub enum FullZipScanOutcome {
    /// Complete output for exactly these file-local top-level rows.
    Batch {
        rows: Range<u64>,
        batch: RecordBatch,
    },
    /// The cursor is unchanged. Drain retained output or replan before retrying;
    /// this also covers a minimal row that cannot fit the configured workspace.
    NeedsCapacity,
    /// Every non-deleted row in the requested physical range has been returned.
    Finished,
}

/// Opt-in FullZip cursor with no read-ahead or whole-result buffer.
///
/// The same shared allocation budget covers each operation through final Arrow
/// reconstruction. Retained output keeps its charge. This is not a process RSS
/// limit: file metadata, transport/cache and allocator overhead are excluded.
/// Each batch prepares charged column inputs asynchronously, then runs all
/// unzip/decode/Arrow reconstruction in one CPU job. Input overlap stays within
/// the same workspace. Prefix attempts and dispatch are not performance qualified.
/// Dropping a pending `try_next` leaves the cursor unchanged; backend operations
/// that outlive cancellation retain their allocation owners until access stops.
#[derive(Debug)]
pub struct FullZipScan {
    columns: Vec<FullZipColumn>,
    schema: Arc<ArrowSchema>,
    budget: Arc<AllocationBudget>,
    workspace_bytes: usize,
    is_partial_workspace_enabled: bool,
    max_rows: u64,
    read_len: u64,
    end: u64,
    cursor: u64,
    deleted_rows: Option<Arc<DeletionVector>>,
    has_started: bool,
}

#[derive(Debug)]
struct FullZipColumn {
    io: Arc<dyn EncodingsIo>,
    column: Arc<ColumnInfo>,
    page_index: usize,
    page_start: u64,
    decoder: Option<Arc<FullZipScheduler>>,
}

pub(super) struct BatchDecodeJob {
    inputs: Vec<FullZipArrayInput>,
    schema: Arc<ArrowSchema>,
    workspace: Reservation,
}

impl BatchDecodeJob {
    pub(super) fn decode(mut self) -> Result<Option<RecordBatch>> {
        let mut arrays = Vec::with_capacity(self.inputs.len());
        for input in self.inputs {
            let Some(array) = input.decode_in(&mut self.workspace)? else {
                // Drop earlier output and all remaining inputs before replanning.
                return Ok(None);
            };
            arrays.push(array);
        }
        Ok(Some(RecordBatch::try_new(self.schema, arrays)?))
    }
}

impl FileReader {
    /// Open an admitted cursor for this reader's base projection.
    ///
    /// Each selected field must be Binary/Utf8/LargeBinary/LargeUtf8, optionally
    /// nested in List/LargeList, stored in dense FullZip pages in v2.2/v2.3.
    /// Every selected page's layout and Arrow shape are checked before output.
    /// Payload corruption and data-dependent codec errors can still fail later.
    /// Other layouts, Blob descriptors, structural-only projections and structs
    /// return errors instead of switching to ordinary unadmitted decoding.
    /// Existing version-specific projection validation and read length are used.
    /// This API takes physical file-row ranges and starts without deletion
    /// handling; use [`FullZipScan::with_deleted_rows`] to bind a snapshot. It
    /// does not apply query filters, takes or dataset-level row IDs. Ordinary
    /// scans are unchanged.
    ///
    /// `max_rows` limits a single batch, never the total selection. A batch ends
    /// at the earliest projected page boundary, or earlier when its complete
    /// workspace cannot fit. All columns use one atomic reservation and the same
    /// top-level row range. A refused column releases the whole attempt before
    /// reducing that range; it never waits for credit while holding earlier columns.
    /// No payload I/O is submitted until `try_next` is polled. A slow consumer
    /// therefore triggers no speculative reads. Use one shared `budget` across
    /// related cursors; a separate pool per file does not bound aggregate usage.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use lance_core::{Result, utils::allocation::AllocationBudget};
    /// use lance_file::reader::{FileReader, FullZipScanOutcome};
    /// async fn consume(reader: &FileReader, budget: Arc<AllocationBudget>) -> Result<()> {
    ///     let mut scan = reader.open_admitted_fullzip_scan(
    ///         0..reader.num_rows(), budget, 65536, 1024).await?;
    ///     loop {
    ///         match scan.try_next().await? {
    ///             FullZipScanOutcome::Batch { batch, .. } => drop(batch), // consume first
    ///             FullZipScanOutcome::Finished => return Ok(()),
    ///             FullZipScanOutcome::NeedsCapacity => {
    ///                 // Return to an owner that can drain/replan; do not spin or
    ///                 // report successful completion for this unfinished selection.
    ///                 return Err(lance_core::Error::io("drain or replan required"));
    ///             }
    ///         }
    ///     }
    /// }
    /// ```
    pub async fn open_admitted_fullzip_scan(
        &self,
        rows: Range<u64>,
        budget: Arc<AllocationBudget>,
        workspace_bytes: usize,
        max_rows: u64,
    ) -> Result<FullZipScan> {
        if workspace_bytes == 0 || max_rows == 0 || rows.start > rows.end {
            return Err(Error::invalid_input(
                "FullZip scan requires positive workspace/max_rows and an ordered row range",
            ));
        }
        if !matches!(
            self.version(),
            ConcreteFileVersion::V2_2 | ConcreteFileVersion::V2_3
        ) || !self.core.options.decoder_config.list_structure.is_empty()
        {
            return Err(Error::not_supported(
                "admitted FullZip scan requires v2.2/v2.3 full-value reads",
            ));
        }
        let projection = self.base_projection();
        if projection.schema.fields.is_empty()
            || projection.column_indices.len() != projection.schema.fields.len()
            || projection
                .schema
                .fields_pre_order()
                .any(|field| field.is_blob())
        {
            return Err(Error::not_supported(
                "admitted FullZip scan requires one non-Blob leaf path per field",
            ));
        }
        let (prepared, read_len) = self.prepare(projection.clone()).await?;
        if rows.end > read_len {
            return Err(Error::invalid_input(format!(
                "FullZip scan range {}..{} exceeds projected length {read_len}",
                rows.start, rows.end
            )));
        }
        let schema = Arc::new(ArrowSchema::from(
            prepared.decoder_projection.schema.as_ref(),
        ));
        let mut columns = Vec::with_capacity(schema.fields().len());
        for (field, column_index) in schema
            .fields()
            .iter()
            .zip(&prepared.decoder_projection.column_indices)
        {
            let data_type = field.data_type();
            let mut leaf = data_type;
            while let DataType::List(field) | DataType::LargeList(field) = leaf {
                leaf = field.data_type();
            }
            if !matches!(
                leaf,
                DataType::Binary | DataType::LargeBinary | DataType::Utf8 | DataType::LargeUtf8
            ) {
                return Err(Error::not_supported(
                    "admitted FullZip scan requires binary/string leaves with optional List/LargeList nesting",
                ));
            }
            let column = prepared
                .column_infos
                .get(*column_index as usize)
                .ok_or_else(|| {
                    Error::invalid_input("FullZip projection refers to missing column metadata")
                })?
                .clone();
            let mut page_start = 0_u64;
            for page in column.page_infos.iter() {
                let page_end = page_start.checked_add(page.num_rows).ok_or_else(|| {
                    Error::corrupt_file_named("FullZip", "column row count overflows")
                })?;
                if !rows.is_empty() && page_start < rows.end && page_end > rows.start {
                    FullZipScheduler::try_from_page(page, &DefaultDecompressionStrategy {})?
                        .validate_array_type(data_type)?;
                }
                page_start = page_end;
            }
            if page_start != read_len {
                return Err(Error::corrupt_file_named(
                    "FullZip",
                    "physical page lengths differ from projected read length",
                ));
            }
            columns.push(FullZipColumn {
                io: self.core.scheduler.clone(),
                column,
                page_index: 0,
                page_start: 0,
                decoder: None,
            });
        }
        Ok(FullZipScan {
            columns,
            schema,
            budget,
            workspace_bytes,
            is_partial_workspace_enabled: false,
            max_rows,
            read_len,
            end: rows.end,
            cursor: rows.start,
            deleted_rows: None,
            has_started: false,
        })
    }
}

impl FullZipColumn {
    // Validate a future interval without changing the current decoder cache or
    // reading payload. Pages wholly inside the skipped gap are irrelevant.
    fn validate_range(&self, rows: &Range<u64>, data_type: &DataType) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let mut page_start = self.page_start;
        for page in self.column.page_infos.iter().skip(self.page_index) {
            if page_start >= rows.end {
                break;
            }
            let page_end = page_start.checked_add(page.num_rows).ok_or_else(|| {
                Error::corrupt_file_named("FullZip", "column row count overflows")
            })?;
            if page_end > rows.start {
                FullZipScheduler::try_from_page(page, &DefaultDecompressionStrategy {})?
                    .validate_array_type(data_type)?;
            }
            page_start = page_end;
        }
        Ok(())
    }

    // Select metadata only. Advancing this cache never consumes logical rows.
    fn select(&mut self, cursor: u64) -> Result<u64> {
        loop {
            let page = self.column.page_infos.get(self.page_index).ok_or_else(|| {
                Error::corrupt_file_named("FullZip", "page table ended before selected rows")
            })?;
            let page_end = self
                .page_start
                .checked_add(page.num_rows)
                .ok_or_else(|| Error::corrupt_file_named("FullZip", "page row count overflows"))?;
            if cursor >= page_end {
                self.page_index += 1;
                self.page_start = page_end;
                self.decoder = None;
                continue;
            }
            if self.decoder.is_none() {
                self.decoder = Some(Arc::new(FullZipScheduler::try_from_page(
                    page,
                    &DefaultDecompressionStrategy {},
                )?));
            }
            return Ok(page_end);
        }
    }
}

impl FullZipScan {
    /// Exclude deleted physical file-row offsets before reading payload.
    ///
    /// Bind the snapshot before the first `try_next` attempt, including a refused
    /// or cancelled attempt. It then applies to all forward ranges. Nonempty
    /// snapshots require at most 2^32 file rows and every deleted offset must be
    /// within that file. The caller must establish that these are the file's
    /// physical offsets, not stable row IDs or logical post-deletion positions.
    ///
    /// Each batch covers one contiguous live interval and reports its physical
    /// range. All-deleted suffixes finish without I/O or credit. Refusal, errors
    /// and cancellation preserve the cursor, including leading deleted rows.
    /// No mask, filtered payload or full list of live ranges is allocated. The
    /// supplied shared deletion metadata is outside the payload budget.
    ///
    /// This removes rows; it does not implement make-deletions-null or remap a
    /// logical range/limit. Opening still validates all selected page layouts.
    /// Fragmented deletions can reduce batch sizes and increase I/O; performance
    /// is not qualified. Ordinary fragment/scanner execution is unchanged.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use lance_core::{Result, utils::deletion::DeletionVector};
    /// use lance_file::reader::FullZipScan;
    /// fn exclude(scan: FullZipScan) -> Result<FullZipScan> {
    ///     scan.with_deleted_rows(Arc::new([2_u32, 5].into_iter().collect::<DeletionVector>()))
    /// }
    /// ```
    pub fn with_deleted_rows(mut self, deleted: Arc<DeletionVector>) -> Result<Self> {
        if self.has_started {
            return Err(Error::invalid_input(
                "FullZip deletion snapshot must be bound before the first read attempt",
            ));
        }
        if deleted.is_empty() {
            self.deleted_rows = None;
        } else {
            let has_invalid_offset = match deleted.as_ref() {
                DeletionVector::NoDeletions => false,
                DeletionVector::Set(set) => set.iter().any(|row| u64::from(*row) >= self.read_len),
                DeletionVector::Bitmap(bitmap) => bitmap
                    .max()
                    .is_some_and(|row| u64::from(row) >= self.read_len),
            };
            if self.read_len > u64::from(u32::MAX) + 1 || has_invalid_offset {
                return Err(Error::invalid_input(format!(
                    "FullZip deleted offsets require a file length within 2^32 and offsets below {}",
                    self.read_len
                )));
            }
            self.deleted_rows = Some(deleted);
        }
        Ok(self)
    }

    /// Combine disjoint columns from row-aligned files into one admitted cursor.
    ///
    /// Both cursors must use the same shared budget, workspace size, partial-mode
    /// setting and maximum batch rows. Their physical file length, next row and
    /// selected end must match, with either no deletions or the same shared
    /// deletion snapshot. Output fields follow `self` then `other`; duplicate
    /// names and conflicting schema metadata are rejected. This consumes both
    /// cursors, including on error, but never reads payload or releases their
    /// previously returned Arrow aliases.
    ///
    /// Each batch ends at the earliest page boundary across all files. One
    /// reservation covers every input and output, and one CPU job reconstructs
    /// the complete batch. It never retains one file's completed batch while
    /// requesting another workspace for the next file. Cancellation or refusal
    /// leaves the combined row position unchanged.
    ///
    /// The caller must establish physical row alignment. Matching lengths alone
    /// do not prove that unrelated files represent the same rows. This does not
    /// apply fragment overlays, stable row IDs or query filters, and
    /// does not adopt this path in ordinary fragment/scanner execution.
    ///
    /// ```
    /// use lance_core::Result;
    /// use lance_file::reader::FullZipScan;
    /// fn combine(labels: FullZipScan, frames: FullZipScan) -> Result<FullZipScan> {
    ///     // Both files cover the same physical rows, with disjoint projections.
    ///     labels.try_merge(frames)
    /// }
    /// ```
    pub fn try_merge(mut self, other: Self) -> Result<Self> {
        let same_deletions = match (&self.deleted_rows, &other.deleted_rows) {
            (None, None) => true,
            (Some(left), Some(right)) => Arc::ptr_eq(left, right),
            _ => false,
        };
        if !same_deletions {
            return Err(Error::invalid_input(
                "FullZip merge requires the same shared deletion snapshot",
            ));
        }
        if !Arc::ptr_eq(&self.budget, &other.budget)
            || self.workspace_bytes != other.workspace_bytes
            || self.is_partial_workspace_enabled != other.is_partial_workspace_enabled
            || self.max_rows != other.max_rows
        {
            return Err(Error::invalid_input(
                "FullZip merge requires the same budget and workspace/partial/max_rows settings",
            ));
        }
        if self.read_len != other.read_len || self.cursor != other.cursor || self.end != other.end {
            return Err(Error::invalid_input(format!(
                "FullZip merge requires aligned physical lengths and positions: left {}/{}..{}, right {}/{}..{}",
                self.read_len, self.cursor, self.end, other.read_len, other.cursor, other.end
            )));
        }
        let mut names = HashSet::new();
        for field in self.schema.fields().iter().chain(other.schema.fields()) {
            if !names.insert(field.name()) {
                return Err(Error::invalid_input(format!(
                    "FullZip merge requires disjoint field names: {}",
                    field.name()
                )));
            }
        }
        let schema =
            ArrowSchema::try_merge([self.schema.as_ref().clone(), other.schema.as_ref().clone()])
                .map_err(|error| Error::invalid_input(format!("FullZip merge schema: {error}")))?;
        self.columns.extend(other.columns);
        self.schema = Arc::new(schema);
        self.has_started |= other.has_started;
        Ok(self)
    }

    /// Reuse a drained cursor for another forward, non-overlapping file-row range.
    ///
    /// The current range must have been completely delivered. The next range
    /// must start at or after its end and stay within the projected file length.
    /// Empty ranges are allowed. Invalid or unsupported selections leave the
    /// cursor unchanged. All selected page layouts are validated before switching;
    /// payload/codec errors can still occur when the range is read.
    ///
    /// This only updates metadata. It neither reads ahead nor drains retained
    /// output: aliases from earlier ranges still charge the same shared budget.
    /// Each `try_next` batch stays inside the active range. `Finished` means that
    /// range is complete; the caller owns the selection iterator and decides
    /// whether to advance again or finish. No whole-selection list is retained.
    /// A bound deletion snapshot still applies. This does not implement query
    /// filters, logical range remapping or arbitrary-order takes.
    ///
    /// ```
    /// use lance_core::Result;
    /// use lance_file::reader::{FullZipScan, FullZipScanOutcome};
    /// async fn consume(scan: &mut FullZipScan) -> Result<()> {
    ///     // The caller has already drained the current range ending at row 10.
    ///     scan.advance_to_range(20..30)?;
    ///     loop {
    ///         match scan.try_next().await? {
    ///             FullZipScanOutcome::Batch { batch, .. } => drop(batch),
    ///             FullZipScanOutcome::Finished => return Ok(()),
    ///             FullZipScanOutcome::NeedsCapacity => {
    ///                 return Err(lance_core::Error::io("drain or replan required"));
    ///             }
    ///         }
    ///     }
    /// }
    /// ```
    pub fn advance_to_range(&mut self, rows: Range<u64>) -> Result<()> {
        if self.cursor != self.end {
            return Err(Error::invalid_input(format!(
                "FullZip range must be drained before advancing: next row {}, end {}",
                self.cursor, self.end
            )));
        }
        if rows.start > rows.end || rows.start < self.end || rows.end > self.read_len {
            return Err(Error::invalid_input(format!(
                "FullZip next range {}..{} must be ordered, start at or after {}, and end within {} rows",
                rows.start, rows.end, self.end, self.read_len
            )));
        }
        for (column, field) in self.columns.iter().zip(self.schema.fields()) {
            column.validate_range(&rows, field.data_type())?;
        }
        // Commit only after every projected column validates, with no await or
        // payload ownership change between validation and cursor replacement.
        self.cursor = rows.start;
        self.end = rows.end;
        Ok(())
    }

    /// Allow an attempt to use currently available credit below its configured
    /// workspace size. This lets a file make progress while another file's output
    /// still retains part of the shared budget. The entire smaller workspace is
    /// admitted atomically before I/O; dependent work never waits for more credit.
    ///
    /// Disabled by default. Enabling can produce smaller batches and more prefix
    /// attempts/reads; performance is not qualified. If even one row cannot fit,
    /// `try_next` returns `NeedsCapacity` without advancing, never EOF. The caller
    /// still owns drain/replan scheduling and fairness between competing scans.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use lance_core::{Result, utils::allocation::AllocationBudget};
    /// use lance_file::reader::FileReader;
    /// async fn scan(reader: &FileReader, budget: Arc<AllocationBudget>) -> Result<()> {
    ///     let mut scan = reader.open_admitted_fullzip_scan(
    ///         0..reader.num_rows(), budget, 65536, 1024).await?
    ///         .with_partial_workspace(true);
    ///     let outcome = scan.try_next().await?;
    ///     drop(outcome); // consume or drain before requesting more
    ///     Ok(())
    /// }
    /// ```
    pub fn with_partial_workspace(mut self, is_enabled: bool) -> Self {
        self.is_partial_workspace_enabled = is_enabled;
        self
    }

    /// The next file-local top-level row; unchanged on refusal, error or cancellation.
    pub fn position(&self) -> u64 {
        self.cursor
    }

    async fn try_read_batch_in<Run, Done>(
        &self,
        rows: Range<u64>,
        mut workspace: Reservation,
        dispatch: &mut Run,
    ) -> Result<Option<RecordBatch>>
    where
        Run: FnMut(BatchDecodeJob) -> Done,
        Done: Future<Output = Result<Option<RecordBatch>>>,
    {
        let mut inputs = Vec::with_capacity(self.columns.len());
        for (column, field) in self.columns.iter().zip(self.schema.fields()) {
            let decoder = column
                .decoder
                .as_ref()
                .ok_or_else(|| Error::internal("missing FullZip decoder"))?;
            let priority = column.column.page_infos[column.page_index].priority;
            let local = (rows.start - column.page_start)..(rows.end - column.page_start);
            let input = decoder
                .clone()
                .prepare_array_range_in(
                    local,
                    field.data_type().clone(),
                    &mut workspace,
                    |range, buffer| column.io.submit_single_into(range, priority, buffer),
                )
                .await?;
            let Some(input) = input else {
                // Prepared inputs must drop before the outer retry admits again.
                return Ok(None);
            };
            inputs.push(input);
        }
        dispatch(BatchDecodeJob {
            inputs,
            schema: self.schema.clone(),
            workspace,
        })
        .await
    }

    /// Attempt one complete batch. Never waits for allocation credit.
    /// Call again after consuming/dropping a batch, or after resolving a refusal.
    pub async fn try_next(&mut self) -> Result<FullZipScanOutcome> {
        self.try_next_with(|job| spawn_cpu(move || job.decode()))
            .await
    }

    // An owned handoff lets cancellation tests control the queue without parking
    // a CPU worker. Every retry transfers the whole batch's input and workspace.
    pub(super) async fn try_next_with<Run, Done>(
        &mut self,
        mut dispatch: Run,
    ) -> Result<FullZipScanOutcome>
    where
        Run: FnMut(BatchDecodeJob) -> Done,
        Done: Future<Output = Result<Option<RecordBatch>>>,
    {
        self.has_started = true;
        let live = match &self.deleted_rows {
            Some(deleted) => next_live_range(deleted, self.cursor..self.end),
            None => (self.cursor < self.end).then_some(self.cursor..self.end),
        };
        let Some(live) = live else {
            // No await or allocation: there is no outstanding row to deliver.
            self.cursor = self.end;
            return Ok(FullZipScanOutcome::Finished);
        };
        let mut end = live.start + self.max_rows.min(live.end - live.start);
        for column in &mut self.columns {
            end = end.min(column.select(live.start)?);
        }
        let mut rows = live.start..end;
        loop {
            let workspace = if self.is_partial_workspace_enabled {
                self.budget.try_reserve_up_to(self.workspace_bytes)
            } else {
                self.budget.try_reserve(self.workspace_bytes)
            };
            let Some(workspace) = workspace else {
                return Ok(FullZipScanOutcome::NeedsCapacity);
            };
            let batch = self
                .try_read_batch_in(rows.clone(), workspace, &mut dispatch)
                .await?;
            if let Some(batch) = batch {
                // No await follows the commit: cancellation cannot consume rows
                // without returning the complete, aligned batch to the caller.
                self.cursor = rows.end;
                return Ok(FullZipScanOutcome::Batch { rows, batch });
            }
            let count = rows.end - rows.start;
            if count == 1 {
                return Ok(FullZipScanOutcome::NeedsCapacity);
            }
            rows.end = rows.start + count / 2;
        }
    }
}

// Snapshot binding validates the u32 domain. Return only the first live run;
// never materialize a complement bitmap or enumerate a large deleted prefix.
fn next_live_range(deleted: &DeletionVector, mut rows: Range<u64>) -> Option<Range<u64>> {
    if rows.is_empty() {
        return None;
    }
    match deleted {
        DeletionVector::NoDeletions => Some(rows),
        DeletionVector::Set(set) => {
            while rows.start < rows.end && set.contains(&(rows.start as u32)) {
                rows.start += 1;
            }
            rows.end = set
                .iter()
                .map(|row| u64::from(*row))
                .filter(|row| rows.contains(row))
                .min()
                .unwrap_or(rows.end);
            (!rows.is_empty()).then_some(rows)
        }
        DeletionVector::Bitmap(bitmap) => {
            let mut iter = bitmap.range(rows.start as u32..=(rows.end - 1) as u32);
            if let Some(run) = iter.next_range() {
                if u64::from(*run.start()) > rows.start {
                    rows.end = u64::from(*run.start());
                } else {
                    rows.start = u64::from(*run.end()) + 1;
                    rows.end = iter.next().map(u64::from).unwrap_or(rows.end);
                }
            }
            (!rows.is_empty()).then_some(rows)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::empty(0..0)]
    #[case::small(0..24)]
    #[case::container_boundary(65530..65555)]
    #[case::u32_boundary(u64::from(u32::MAX) - 24..u64::from(u32::MAX) + 1)]
    fn live_ranges_match_independent_membership(#[case] rows: Range<u64>) {
        let offsets = rows
            .clone()
            .filter(|row| (row - rows.start) % 7 < 4)
            .map(|row| row as u32)
            .collect::<Vec<_>>();
        for deleted in [
            DeletionVector::NoDeletions,
            DeletionVector::Set(offsets.iter().copied().collect()),
            DeletionVector::Bitmap(offsets.iter().copied().collect()),
        ] {
            let expected = rows
                .clone()
                .filter(|row| !deleted.contains(*row as u32))
                .collect::<Vec<_>>();
            let mut actual = Vec::new();
            let mut start = rows.start;
            while let Some(live) = next_live_range(&deleted, start..rows.end) {
                assert!(live.start >= start && live.start < live.end && live.end <= rows.end);
                actual.extend(live.clone());
                start = live.end;
            }
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn large_deleted_run_does_not_expand_the_complement() {
        let mut deleted = DeletionVector::Bitmap(Default::default());
        if let DeletionVector::Bitmap(bitmap) = &mut deleted {
            bitmap.insert_range(0..1_000_000);
        }
        assert_eq!(
            next_live_range(&deleted, 0..1_000_001),
            Some(1_000_000..1_000_001)
        );
        assert_eq!(next_live_range(&deleted, 19..65540), None);
    }
}
