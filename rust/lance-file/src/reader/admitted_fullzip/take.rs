// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Sparse reads composed with routing/output inside an existing admission.

use arrow_array::{Array, make_array};
use lance_encoding::data::{AllNullDataBlock, DataBlock, try_concat_nested_binary_in};

use super::*;

/// Charged sparse inputs awaiting CPU decoding in the caller's reservation.
///
/// Created by [`FileReader::prepare_admitted_fullzip_take_in`]. Each projected
/// column owns inputs for its selected contiguous in-page runs. Dropping this
/// value releases those owners; it does not retain a cursor or consume rows.
/// Descriptor/Arrow metadata is outside the payload budget and scales with the
/// selected run count, columns and nesting depth. This is not an RSS bound.
pub struct PreparedFullZipTake {
    schema: Arc<ArrowSchema>,
    columns: Vec<Vec<AdmittedBytePageInput>>,
}

impl PreparedFullZipTake {
    /// Decode and concatenate exactly the prepared selection, without I/O or waits.
    ///
    /// Use the same original reservation as preparation and include routing,
    /// other files, decoded chunks and final output overlap in that admission.
    /// This is pure CPU work suitable for one outer `spawn_cpu` call. `None`
    /// means discard the complete compound attempt and its workspace before
    /// replanning; it is never EOF or partial output. Freed intermediate credit
    /// may return to the shared budget, not this exclusive reservation. Empty
    /// selections still charge the required typed offset buffers. Single runs
    /// retain their decoded arrays without an additional concatenation copy.
    ///
    /// See [`FileReader::prepare_admitted_fullzip_take_in`] for a complete example.
    pub fn decode_in(self, workspace: &mut Reservation) -> Result<Option<RecordBatch>> {
        let mut arrays = Vec::with_capacity(self.columns.len());
        for (inputs, field) in self.columns.into_iter().zip(self.schema.fields()) {
            let mut chunks = Vec::with_capacity(inputs.len());
            for input in inputs {
                let Some(array) = input.decode_in(workspace)? else {
                    return Ok(None);
                };
                chunks.push(array);
            }
            let array = match chunks.len() {
                0 => {
                    let block = DataBlock::AllNull(AllNullDataBlock { num_values: 0 });
                    let Some(data) = block.try_to_arrow_in(field.data_type(), workspace)? else {
                        return Ok(None);
                    };
                    make_array(data)
                }
                1 => chunks.remove(0),
                _ => {
                    let sources: Vec<&dyn Array> = chunks.iter().map(AsRef::as_ref).collect();
                    let Some(array) = try_concat_nested_binary_in(&sources, workspace)? else {
                        return Ok(None);
                    };
                    array
                }
            };
            arrays.push(array);
        }
        Ok(Some(RecordBatch::try_new(self.schema, arrays)?))
    }
}

impl FileReader {
    /// Prepare sorted, unique physical file-row offsets within an existing admission.
    ///
    /// Supports the same projection/types as [`Self::open_admitted_fullzip_scan`].
    /// All indices and selected page layouts are checked before payload reads;
    /// unselected pages are not decoded or read. Adjacent offsets within a page
    /// share one range read, with the FullZip offset-index reads needed to locate
    /// it. Input buffers retain their full physical charges. No new reservation,
    /// wait, prefix reduction, ordinary decode fallback or cursor is introduced.
    ///
    /// Pass dense file offsets, not stable row IDs or pre-overlay physical rows.
    /// No deletions, row IDs or row-version columns are implicitly applied.
    /// `None` is insufficient capacity: discard all prior files/routing/output
    /// and the original reservation before replanning the compound operation.
    /// Errors/cancellation also discard this preparation's inputs. Backend work
    /// that survives cancellation retains its owners until it stops using them.
    /// Empty selections read no payload; decoding still charges typed offsets.
    /// Transport/cache, file/descriptor metadata and allocator overhead remain
    /// excluded. Sparse read amplification and performance are not qualified.
    ///
    /// ```
    /// use arrow_array::RecordBatch;
    /// use lance_core::{Result, utils::{allocation::AllocationBudget, tokio::spawn_cpu}};
    /// use lance_file::reader::FileReader;
    /// async fn take(reader: &FileReader) -> Result<Option<RecordBatch>> {
    ///     let budget = AllocationBudget::new(65536);
    ///     let mut work = budget.try_reserve(65536).unwrap();
    ///     let Some(input) = reader.prepare_admitted_fullzip_take_in(&[0, 2], &mut work).await?
    ///         else { return Ok(None) };
    ///     // Compose additional files before dispatch; include their overlap in work.
    ///     spawn_cpu(move || input.decode_in(&mut work)).await
    /// }
    /// ```
    pub async fn prepare_admitted_fullzip_take_in(
        &self,
        indices: &[u32],
        workspace: &mut Reservation,
    ) -> Result<Option<PreparedFullZipTake>> {
        self.core
            .prepare_admitted_fullzip_take_in(indices, self.base_projection().clone(), workspace)
            .await
    }
}

impl crate::reader::ProjectedFileReader {
    /// Prepare sparse FullZip inputs using the projection-scoped metadata path.
    ///
    /// This has the admission, supported-type, sorted-unique-offset and failure
    /// contract of [`FileReader::prepare_admitted_fullzip_take_in`]. `projection`
    /// follows this reader's ordinary per-read projection rules, including exact
    /// version validation and lazy metadata loading. No full-file metadata reader
    /// or ordinary payload decode is constructed as a fallback.
    ///
    /// ```
    /// use lance_core::{Result, utils::allocation::Reservation};
    /// use lance_file::reader::{ProjectedFileReader, PreparedFullZipTake};
    /// async fn prepare(reader: &ProjectedFileReader, work: &mut Reservation)
    ///     -> Result<Option<PreparedFullZipTake>> {
    ///     reader.prepare_admitted_fullzip_take_in(&[0, 2], None, work).await
    /// }
    /// ```
    pub async fn prepare_admitted_fullzip_take_in(
        &self,
        indices: &[u32],
        projection: Option<ReaderProjection>,
        workspace: &mut Reservation,
    ) -> Result<Option<PreparedFullZipTake>> {
        self.core
            .prepare_admitted_fullzip_take_in(
                indices,
                projection.unwrap_or_else(|| self.base_projection().clone()),
                workspace,
            )
            .await
    }
}

impl DecodeEngine {
    async fn prepare_admitted_fullzip_take_in(
        &self,
        indices: &[u32],
        projection: ReaderProjection,
        workspace: &mut Reservation,
    ) -> Result<Option<PreparedFullZipTake>> {
        if indices.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(Error::invalid_input(
                "admitted FullZip take requires sorted unique physical file offsets",
            ));
        }
        let (schema, mut columns, read_len) = self.prepare_fullzip_columns(projection).await?;
        if let Some(last) = indices.last()
            && u64::from(*last) >= read_len
        {
            return Err(Error::invalid_input(format!(
                "FullZip take offset {last} exceeds projected length {read_len}",
            )));
        }
        // Preflight every selected page/column before any payload access. Retain
        // no expanded routing table; a second pass regenerates the in-page runs.
        for (column, field) in columns.iter_mut().zip(schema.fields()) {
            let mut remaining = indices;
            while let Some(first) = remaining.first() {
                let end = column.select(u64::from(*first))?;
                column
                    .decoder
                    .as_ref()
                    .ok_or_else(|| Error::internal("missing FullZip take decoder"))?
                    .validate_array_type(field.data_type())?;
                remaining = &remaining[remaining.partition_point(|row| u64::from(*row) < end)..];
            }
            column.page_index = 0;
            column.page_start = 0;
            column.decoder = None;
        }
        let mut inputs = Vec::with_capacity(columns.len());
        for (column, field) in columns.iter_mut().zip(schema.fields()) {
            let mut chunks = Vec::new();
            let mut position = 0;
            while let Some(first) = indices.get(position) {
                let start = u64::from(*first);
                let page_end = column.select(start)?;
                // Every selected offset is u32, so its exclusive end fits u64.
                let mut end = start + 1;
                position += 1;
                while end < page_end
                    && indices.get(position).map(|row| u64::from(*row)) == Some(end)
                {
                    end += 1;
                    position += 1;
                }
                let decoder = column
                    .decoder
                    .as_ref()
                    .ok_or_else(|| Error::internal("missing FullZip take decoder"))?;
                let priority = column.column.page_infos[column.page_index].priority;
                let Some(input) = decoder
                    .prepare_array_range_in(
                        (start - column.page_start)..(end - column.page_start),
                        field.data_type().clone(),
                        workspace,
                        |range, output| column.io.submit_single_into(range, priority, output),
                    )
                    .await?
                else {
                    return Ok(None);
                };
                chunks.push(input);
            }
            inputs.push(chunks);
        }
        Ok(Some(PreparedFullZipTake {
            schema,
            columns: inputs,
        }))
    }
}
