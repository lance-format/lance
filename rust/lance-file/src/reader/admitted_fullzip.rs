// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Demand-driven FullZip cursor with one admission for all projected columns.

use std::{future::Future, ops::Range, sync::Arc};

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Schema as ArrowSchema};
use lance_core::{
    Error, Result,
    utils::{
        allocation::{AllocationBudget, Reservation},
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
    /// Every row in the requested range has been returned.
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
    io: Arc<dyn EncodingsIo>,
    columns: Vec<FullZipColumn>,
    schema: Arc<ArrowSchema>,
    budget: Arc<AllocationBudget>,
    workspace_bytes: usize,
    is_partial_workspace_enabled: bool,
    max_rows: u64,
    end: u64,
    cursor: u64,
}

#[derive(Debug)]
struct FullZipColumn {
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
    /// This API takes physical file-row ranges; it does not apply query filters,
    /// deletion vectors, takes or dataset-level row IDs. Ordinary scans are unchanged.
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
                column,
                page_index: 0,
                page_start: 0,
                decoder: None,
            });
        }
        Ok(FullZipScan {
            io: self.core.scheduler.clone(),
            columns,
            schema,
            budget,
            workspace_bytes,
            is_partial_workspace_enabled: false,
            max_rows,
            end: rows.end,
            cursor: rows.start,
        })
    }
}

impl FullZipColumn {
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
                    |range, buffer| self.io.submit_single_into(range, priority, buffer),
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
        if self.cursor == self.end {
            return Ok(FullZipScanOutcome::Finished);
        }
        let mut end = self.cursor + self.max_rows.min(self.end - self.cursor);
        for column in &mut self.columns {
            end = end.min(column.select(self.cursor)?);
        }
        let mut rows = self.cursor..end;
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
