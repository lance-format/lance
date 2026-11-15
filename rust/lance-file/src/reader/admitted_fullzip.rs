// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Explicit demand-driven cursor for a single admitted FullZip leaf path.

use std::{ops::Range, sync::Arc};

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Schema as ArrowSchema};
use lance_core::{Error, Result, utils::allocation::AllocationBudget};
use lance_encoding::{
    EncodingsIo, compression::DefaultDecompressionStrategy, decoder::ColumnInfo,
    encodings::logical::primitive::FullZipScheduler,
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

/// Opt-in single-column FullZip cursor with no read-ahead or whole-result buffer.
///
/// The same shared allocation budget covers each operation through final Arrow
/// reconstruction. Retained output keeps its charge. This is not a process RSS
/// limit: file metadata, transport/cache and allocator overhead are excluded.
/// CPU remains inline and repeated prefix attempts are not performance qualified.
/// Dropping a pending `try_next` leaves the cursor unchanged; backend operations
/// that outlive cancellation retain their allocation owners until access stops.
#[derive(Debug)]
pub struct FullZipScan {
    io: Arc<dyn EncodingsIo>,
    column: Arc<ColumnInfo>,
    schema: Arc<ArrowSchema>,
    budget: Arc<AllocationBudget>,
    workspace_bytes: usize,
    max_rows: u64,
    end: u64,
    cursor: u64,
    page_index: usize,
    page_start: u64,
    decoder: Option<FullZipScheduler>,
}

impl FileReader {
    /// Open an admitted cursor for this reader's single-field base projection.
    ///
    /// The selected field must be Binary/Utf8/LargeBinary/LargeUtf8, optionally
    /// nested in List/LargeList, stored in dense FullZip pages in v2.2/v2.3.
    /// Every selected page's layout and Arrow shape are checked before output.
    /// Payload corruption and data-dependent codec errors can still fail later.
    /// Other layouts,
    /// Blob descriptors, structural-only projections and multi-field schemas
    /// return errors instead of switching to ordinary unadmitted decoding.
    /// Existing version-specific projection validation and read length are used.
    /// This API takes physical file-row ranges; it does not apply query filters,
    /// deletion vectors, takes or dataset-level row IDs. Ordinary scans are unchanged.
    ///
    /// `max_rows` limits a single batch, never the total selection. A batch ends
    /// at a page boundary or earlier when its complete workspace cannot fit.
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
        if projection.schema.fields.len() != 1
            || projection.column_indices.len() != 1
            || projection
                .schema
                .fields_pre_order()
                .any(|field| field.is_blob())
        {
            return Err(Error::not_supported(
                "admitted FullZip scan requires one non-Blob leaf path",
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
        let data_type = schema.field(0).data_type();
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
        let column_index = prepared.decoder_projection.column_indices[0] as usize;
        let column = prepared
            .column_infos
            .get(column_index)
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
        Ok(FullZipScan {
            io: self.core.scheduler.clone(),
            column,
            schema,
            budget,
            workspace_bytes,
            max_rows,
            end: rows.end,
            cursor: rows.start,
            page_index: 0,
            page_start: 0,
            decoder: None,
        })
    }
}

impl FullZipScan {
    /// The next file-local top-level row; unchanged on refusal, error or cancellation.
    pub fn position(&self) -> u64 {
        self.cursor
    }

    /// Attempt one complete batch. Never waits for allocation credit.
    /// Call again after consuming/dropping a batch, or after resolving a refusal.
    pub async fn try_next(&mut self) -> Result<FullZipScanOutcome> {
        if self.cursor == self.end {
            return Ok(FullZipScanOutcome::Finished);
        }
        loop {
            let page = self.column.page_infos.get(self.page_index).ok_or_else(|| {
                Error::corrupt_file_named("FullZip", "page table ended before selected rows")
            })?;
            let page_end = self
                .page_start
                .checked_add(page.num_rows)
                .ok_or_else(|| Error::corrupt_file_named("FullZip", "page row count overflows"))?;
            if self.cursor >= page_end {
                self.page_index += 1;
                self.page_start = page_end;
                self.decoder = None;
                continue;
            }
            if self.decoder.is_none() {
                self.decoder = Some(FullZipScheduler::try_from_page(
                    page,
                    &DefaultDecompressionStrategy {},
                )?);
            }
            let decoder = self
                .decoder
                .as_ref()
                .ok_or_else(|| Error::internal("missing FullZip decoder"))?;
            let rows = (self.cursor - self.page_start)
                ..(self.cursor + self.max_rows.min(page_end.min(self.end) - self.cursor)
                    - self.page_start);
            let Some(output) = decoder
                .try_read_array_prefix_in(
                    rows,
                    self.schema.field(0).data_type(),
                    &self.budget,
                    self.workspace_bytes,
                    |range, buffer| self.io.submit_single_into(range, page.priority, buffer),
                )
                .await?
            else {
                return Ok(FullZipScanOutcome::NeedsCapacity);
            };
            let rows = self.cursor..self.page_start + output.rows.end;
            let batch = RecordBatch::try_new(self.schema.clone(), vec![output.array])?;
            // No await follows the commit: cancellation can never consume rows
            // without returning the corresponding complete batch to the caller.
            self.cursor = rows.end;
            return Ok(FullZipScanOutcome::Batch { rows, batch });
        }
    }
}
