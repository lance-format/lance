// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    fmt,
    sync::{Arc, OnceLock},
};

use arrow_array::{
    ArrayRef, BooleanArray, RecordBatch, RecordBatchOptions, UInt64Array, make_array,
};
use arrow_buffer::NullBuffer;
use arrow_schema::{Field, Schema, SchemaRef};
use futures::{
    FutureExt, Stream, StreamExt,
    future::{BoxFuture, Shared},
    stream::{BoxStream, FuturesOrdered},
};
use lance_arrow::RecordBatchExt;
use lance_core::{
    Error, ROW_ADDR, ROW_ADDR_FIELD, ROW_CREATED_AT_VERSION_FIELD, ROW_ID, ROW_ID_FIELD,
    ROW_LAST_UPDATED_AT_VERSION_FIELD, Result,
    utils::{address::RowAddress, deletion::DeletionVector},
};
use lance_io::ReadBatchParams;
use tracing::instrument;

use crate::rowids::{RowIdSequence, RowIdSequenceCursor};

pub type ReadBatchFut = BoxFuture<'static, Result<RecordBatch>>;
/// A task, emitted by a file reader, that will produce a batch (of the
/// given size)
pub struct ReadBatchTask {
    pub task: ReadBatchFut,
    pub num_rows: u32,
}
pub type ReadBatchTaskStream = BoxStream<'static, ReadBatchTask>;
pub type ReadBatchFutStream = BoxStream<'static, ReadBatchFut>;

type SharedReadBatchFut = Shared<BoxFuture<'static, std::result::Result<RecordBatch, Arc<Error>>>>;

#[derive(Debug)]
struct SharedReadError(Arc<Error>);

impl fmt::Display for SharedReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for SharedReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref())
    }
}

struct PendingReadBatch {
    task: Option<ReadBatchFut>,
    shared_task: Option<SharedReadBatchFut>,
    offset: u32,
    num_rows: u32,
}

impl PendingReadBatch {
    fn new(task: ReadBatchTask) -> Self {
        Self {
            task: Some(task.task),
            shared_task: None,
            offset: 0,
            num_rows: task.num_rows,
        }
    }

    fn take(&mut self, num_rows: u32) -> ReadBatchFut {
        debug_assert!(num_rows <= self.num_rows);

        if self.offset == 0 && num_rows == self.num_rows && self.shared_task.is_none() {
            self.num_rows = 0;
            let Some(task) = self.task.take() else {
                return async {
                    Err(Error::internal(
                        "missing read task while merging aligned streams".to_string(),
                    ))
                }
                .boxed();
            };
            return task;
        }

        let shared_task = self
            .shared_task
            .get_or_insert_with(|| {
                let task = self.task.take();
                async move {
                    let Some(task) = task else {
                        return Err(Arc::new(Error::internal(
                            "missing read task while splitting a merged stream".to_string(),
                        )));
                    };
                    task.await.map_err(Arc::new)
                }
                .boxed()
                .shared()
            })
            .clone();
        let offset = self.offset;
        self.offset += num_rows;
        self.num_rows -= num_rows;

        async move {
            match shared_task.await {
                Ok(batch) => Ok(batch.slice(offset as usize, num_rows as usize)),
                Err(error) => Err(Error::wrapped(Box::new(SharedReadError(error)))),
            }
        }
        .boxed()
    }
}

struct MergeStream {
    streams: Vec<ReadBatchTaskStream>,
    pending: Vec<Option<PendingReadBatch>>,
    index: usize,
}

impl MergeStream {
    fn emit(&mut self) -> ReadBatchTask {
        let num_rows = self
            .pending
            .iter()
            .filter_map(|pending| pending.as_ref().map(|pending| pending.num_rows))
            .min()
            .unwrap_or_default();
        let mut batches = FuturesOrdered::new();
        for pending in &mut self.pending {
            let Some(pending_batch) = pending.as_mut() else {
                continue;
            };
            batches.push_back(pending_batch.take(num_rows));
            if pending_batch.num_rows == 0 {
                *pending = None;
            }
        }
        let task = async move {
            let Some(first) = batches.next().await else {
                return Err(Error::internal(
                    "cannot merge an empty set of read batches".to_string(),
                ));
            };
            let mut batch = first?;
            while let Some(next) = batches.next().await {
                let next = next?;
                batch = batch.merge(&next)?;
            }
            Ok(batch)
        }
        .boxed();
        ReadBatchTask { task, num_rows }
    }
}

impl Stream for MergeStream {
    type Item = ReadBatchTask;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        loop {
            if self.pending.iter().all(Option::is_some) {
                return std::task::Poll::Ready(Some(self.emit()));
            }

            let index = self.index;
            if self.pending[index].is_some() {
                self.index = (index + 1) % self.streams.len();
                continue;
            }
            match self.streams[index].poll_next_unpin(cx) {
                std::task::Poll::Ready(Some(batch_task)) => {
                    self.pending[index] = Some(PendingReadBatch::new(batch_task));
                    self.index = (index + 1) % self.streams.len();
                }
                std::task::Poll::Ready(None) => {
                    return std::task::Poll::Ready(None);
                }
                std::task::Poll::Pending => {
                    return std::task::Poll::Pending;
                }
            }
        }
    }
}

/// Given multiple streams of batch tasks, merge them into a single stream
///
/// This pulls one batch from each stream and then combines the columns from
/// all of the batches into a single batch.  The order of the batches in the
/// streams is maintained and the merged batch columns will be in order from first
/// to last stream. If the streams use different batch boundaries then batches are
/// sliced so each merged output remains row-aligned.
///
/// This stream ends as soon as any of the input streams ends (we do not
/// verify that the other input streams are finished as well)
pub fn merge_streams(streams: Vec<ReadBatchTaskStream>) -> ReadBatchTaskStream {
    if streams.is_empty() {
        return futures::stream::empty().boxed();
    }
    let pending = (0..streams.len()).map(|_| None).collect();
    MergeStream {
        streams,
        pending,
        index: 0,
    }
    .boxed()
}

/// Apply a mask to the batch, where rows are "deleted" by the _rowid column null.
///
/// This is used partly as a performance optimization (cheaper to null than to filter)
/// but also because there are cases where we want to load the physical rows.  For example,
/// we may be replacing a column based on some UDF and we want to provide a value for the
/// deleted rows to ensure the fragments are aligned.
fn apply_deletions_as_nulls(batch: RecordBatch, mask: &BooleanArray) -> Result<RecordBatch> {
    // Transform mask into null buffer. Null means deleted, though note that
    // null buffers are actually validity buffers, so True means not null
    // and thus not deleted.
    let mask_buffer = NullBuffer::new(mask.values().clone());

    if mask_buffer.null_count() == 0 {
        // No rows are deleted
        return Ok(batch);
    }

    // For each column convert to data
    let new_columns = batch
        .schema()
        .fields()
        .iter()
        .zip(batch.columns())
        .map(|(field, col)| {
            if field.name() == ROW_ID || field.name() == ROW_ADDR {
                let col_data = col.to_data();
                // If it already has a validity bitmap, then AND it with the mask.
                // Otherwise, use the boolean buffer as the mask.
                let null_buffer = NullBuffer::union(col_data.nulls(), Some(&mask_buffer));

                Ok(col_data
                    .into_builder()
                    .null_bit_buffer(null_buffer.map(|b| b.buffer().clone()))
                    .build()
                    .map(make_array)?)
            } else {
                Ok(col.clone())
            }
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(RecordBatch::try_new_with_options(
        batch.schema(),
        new_columns,
        &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )?)
}

/// Extract version values for a batch selection with a reusable RLE cursor.
/// Single-run fragments (the common case) take the O(1) fast path.
fn version_values_for_selection_with_cursor(
    sequence: &crate::rowids::version::RowDatasetVersionSequence,
    cursor: &mut crate::rowids::version::RowDatasetVersionCursor,
    params: &ReadBatchParams,
    batch_offset: u32,
    num_rows: u32,
) -> Result<Vec<u64>> {
    let selection = params
        .slice(batch_offset as usize, num_rows as usize)
        .unwrap()
        .to_ranges()
        .unwrap();

    if sequence.runs.len() == 1 {
        return Ok(vec![sequence.runs[0].version(); num_rows as usize]);
    }

    let mut versions = Vec::with_capacity(num_rows as usize);
    for r in &selection {
        cursor.extend_range(sequence, r.start as usize..r.end as usize, &mut versions)?;
    }
    Ok(versions)
}

fn version_values_for_selection(
    sequence: &crate::rowids::version::RowDatasetVersionSequence,
    params: &ReadBatchParams,
    batch_offset: u32,
    num_rows: u32,
) -> Result<Vec<u64>> {
    // Preserve the common direct-call path without constructing a cursor.
    // Keep the selection validation in the same order as the general path.
    let _selection = params
        .slice(batch_offset as usize, num_rows as usize)
        .unwrap()
        .to_ranges()
        .unwrap();
    if sequence.runs.len() == 1 {
        return Ok(vec![sequence.runs[0].version(); num_rows as usize]);
    }
    version_values_for_selection_with_cursor(
        sequence,
        &mut sequence.cursor(),
        params,
        batch_offset,
        num_rows,
    )
}

/// Configuration needed to apply row ids and deletions to a batch
#[derive(Debug)]
pub struct RowIdAndDeletesConfig {
    /// The row ids that were requested
    pub params: ReadBatchParams,
    /// Whether to include the row id column in the final batch
    pub with_row_id: bool,
    /// Whether to include the row address column in the final batch
    pub with_row_addr: bool,
    /// Whether to include the last updated at version column in the final batch
    pub with_row_last_updated_at_version: bool,
    /// Whether to include the created at version column in the final batch
    pub with_row_created_at_version: bool,
    /// An optional deletion vector to apply to the batch
    pub deletion_vector: Option<Arc<DeletionVector>>,
    /// An optional row id sequence to use for the row id column.
    pub row_id_sequence: Option<Arc<RowIdSequence>>,
    /// The last_updated_at version sequence
    pub last_updated_at_sequence: Option<Arc<crate::rowids::version::RowDatasetVersionSequence>>,
    /// The created_at version sequence
    pub created_at_sequence: Option<Arc<crate::rowids::version::RowDatasetVersionSequence>>,
    /// Whether to make deleted rows null instead of filtering them out
    pub make_deletions_null: bool,
    /// The total number of rows that will be loaded
    ///
    /// This is needed to convert ReadbatchParams::RangeTo into a valid range
    pub total_num_rows: u32,
}

impl RowIdAndDeletesConfig {
    fn has_system_cols(&self) -> bool {
        self.with_row_id
            || self.with_row_addr
            || self.with_row_last_updated_at_version
            || self.with_row_created_at_version
    }
}

pub fn apply_row_id_and_deletes(
    batch: RecordBatch,
    batch_offset: u32,
    fragment_id: u32,
    config: &RowIdAndDeletesConfig,
) -> Result<RecordBatch> {
    apply_row_id_and_deletes_with_system_columns(
        batch,
        batch_offset,
        fragment_id,
        config,
        PrecomputedSystemColumns::default(),
        None,
    )
}

#[derive(Default)]
struct PrecomputedSystemColumns {
    row_ids: Option<Result<Arc<UInt64Array>>>,
    last_updated_versions: Option<Result<Arc<UInt64Array>>>,
    created_versions: Option<Result<Arc<UInt64Array>>>,
}

const ROW_ID_READ_AHEAD_ROWS: usize = 64 * 1024;

struct PrecomputedRowIdChunk {
    logical_offset: usize,
    values: Arc<UInt64Array>,
}

struct CachedOutputSchema {
    input: SchemaRef,
    output: SchemaRef,
}

impl PrecomputedRowIdChunk {
    fn end_offset(&self) -> usize {
        self.logical_offset + self.values.len()
    }

    fn slice(&self, logical_offset: usize, num_rows: usize) -> Option<Arc<UInt64Array>> {
        let offset_in_chunk = logical_offset.checked_sub(self.logical_offset)?;
        if offset_in_chunk + num_rows > self.values.len() {
            return None;
        }
        if offset_in_chunk == 0 && num_rows == self.values.len() {
            return Some(self.values.clone());
        }
        Some(Arc::new(self.values.slice(offset_in_chunk, num_rows)))
    }
}

fn decode_row_id_chunk<const USE_DENSE_ROW_ID_EXPANSION: bool>(
    sequence: &RowIdSequence,
    cursor: &mut RowIdSequenceCursor,
    params: &ReadBatchParams,
    logical_offset: usize,
    chunk_len: usize,
) -> Result<PrecomputedRowIdChunk> {
    let selection = params
        .slice(logical_offset, chunk_len)
        .unwrap()
        .to_ranges()
        .unwrap();
    let values = match selection.as_slice() {
        [range] if USE_DENSE_ROW_ID_EXPANSION => UInt64Array::from(
            sequence
                .select_dense_range_with_cursor(cursor, range.start as usize..range.end as usize),
        ),
        [range] => UInt64Array::from(
            sequence.select_range_with_cursor(cursor, range.start as usize..range.end as usize),
        ),
        _ => sequence
            .select_with_cursor(
                cursor,
                selection
                    .iter()
                    .flat_map(|range| range.start as usize..range.end as usize),
            )
            .collect::<UInt64Array>(),
    };
    if values.len() != chunk_len {
        return Err(Error::corrupt_file_named(
            "row ID metadata",
            format!(
                "decoded row ID chunk at selected offset {logical_offset} contains {} rows, but the selection requires {chunk_len} rows",
                values.len()
            ),
        ));
    }
    Ok(PrecomputedRowIdChunk {
        logical_offset,
        values: Arc::new(values),
    })
}

fn selected_row_count(params: &ReadBatchParams, total_num_rows: usize) -> usize {
    match params {
        ReadBatchParams::Range(range) => range.len(),
        ReadBatchParams::Ranges(ranges) => ranges
            .iter()
            .map(|range| (range.end - range.start) as usize)
            .sum(),
        ReadBatchParams::RangeFull => total_num_rows,
        ReadBatchParams::RangeTo(range) => range.end,
        ReadBatchParams::RangeFrom(range) => total_num_rows.saturating_sub(range.start),
        ReadBatchParams::Indices(indices) => indices.len(),
    }
}

#[instrument(name = "apply_row_id_and_deletes", level = "debug", skip_all)]
fn apply_row_id_and_deletes_with_system_columns(
    batch: RecordBatch,
    batch_offset: u32,
    fragment_id: u32,
    config: &RowIdAndDeletesConfig,
    precomputed: PrecomputedSystemColumns,
    output_schema_cache: Option<&OnceLock<CachedOutputSchema>>,
) -> Result<RecordBatch> {
    let PrecomputedSystemColumns {
        row_ids: precomputed_row_ids,
        last_updated_versions,
        created_versions,
    } = precomputed;
    let mut deletion_vector = config.deletion_vector.as_ref();
    // Convert Some(NoDeletions) into None to simplify logic below
    if let Some(deletion_vector_inner) = deletion_vector
        && matches!(deletion_vector_inner.as_ref(), DeletionVector::NoDeletions)
    {
        deletion_vector = None;
    }
    let has_deletions = deletion_vector.is_some();
    debug_assert!(batch.num_columns() > 0 || config.has_system_cols() || has_deletions);

    // If row id sequence is None, then row id IS row address.
    let should_fetch_row_addr = config.with_row_addr
        || (config.with_row_id && config.row_id_sequence.is_none())
        || has_deletions;

    let num_rows = batch.num_rows() as u32;

    let row_addrs =
        if should_fetch_row_addr {
            let _rowaddrs = tracing::span!(tracing::Level::DEBUG, "fetch_row_addrs").entered();
            let mut row_addrs = Vec::with_capacity(num_rows as usize);
            for offset_range in config
                .params
                .slice(batch_offset as usize, num_rows as usize)
                .unwrap()
                .iter_offset_ranges()?
            {
                row_addrs.extend(offset_range.map(|row_offset| {
                    u64::from(RowAddress::new_from_parts(fragment_id, row_offset))
                }));
            }

            Some(Arc::new(UInt64Array::from(row_addrs)))
        } else {
            None
        };

    let row_ids = if config.with_row_id {
        let _rowids = tracing::span!(tracing::Level::DEBUG, "fetch_row_ids").entered();
        if let Some(row_ids) = precomputed_row_ids {
            let row_ids = row_ids?;
            debug_assert_eq!(row_ids.len(), num_rows as usize);
            Some(row_ids)
        } else if let Some(row_id_sequence) = &config.row_id_sequence {
            let selection = config
                .params
                .slice(batch_offset as usize, num_rows as usize)
                .unwrap()
                .to_ranges()
                .unwrap();
            let row_ids = row_id_sequence
                .select(
                    selection
                        .iter()
                        .flat_map(|r| r.start as usize..r.end as usize),
                )
                .collect::<UInt64Array>();
            Some(Arc::new(row_ids))
        } else {
            // If we don't have a row id sequence, can assume the row ids are
            // the same as the row addresses.
            row_addrs.clone()
        }
    } else {
        None
    };

    let span = tracing::span!(tracing::Level::DEBUG, "apply_deletions");
    let _enter = span.enter();
    let deletion_mask = deletion_vector.and_then(|v| {
        let row_addrs: &[u64] = row_addrs.as_ref().unwrap().values();
        v.build_predicate(row_addrs.iter())
    });

    let mut system_columns: Vec<(Field, ArrayRef)> = Vec::with_capacity(4);
    if config.with_row_id {
        system_columns.push((ROW_ID_FIELD.clone(), row_ids.unwrap()));
    }
    if config.with_row_addr {
        system_columns.push((ROW_ADDR_FIELD.clone(), row_addrs.unwrap()));
    }
    if config.with_row_last_updated_at_version {
        let version_arr = if let Some(version_arr) = last_updated_versions {
            version_arr?
        } else if let Some(sequence) = &config.last_updated_at_sequence {
            Arc::new(UInt64Array::from(version_values_for_selection(
                sequence,
                &config.params,
                batch_offset,
                num_rows,
            )?))
        } else {
            // Default to version 1 if sequence not provided
            Arc::new(UInt64Array::from(vec![1u64; num_rows as usize]))
        };
        system_columns.push((ROW_LAST_UPDATED_AT_VERSION_FIELD.clone(), version_arr));
    }
    if config.with_row_created_at_version {
        let version_arr = if let Some(version_arr) = created_versions {
            version_arr?
        } else if let Some(sequence) = &config.created_at_sequence {
            Arc::new(UInt64Array::from(version_values_for_selection(
                sequence,
                &config.params,
                batch_offset,
                num_rows,
            )?))
        } else {
            // Default to version 1 if sequence not provided
            Arc::new(UInt64Array::from(vec![1u64; num_rows as usize]))
        };
        system_columns.push((ROW_CREATED_AT_VERSION_FIELD.clone(), version_arr));
    }

    let batch = if system_columns.is_empty() {
        batch
    } else if let Some(output_schema_cache) = output_schema_cache {
        let input_schema = batch.schema();
        let make_output_schema = || {
            let mut fields = input_schema
                .fields()
                .iter()
                .map(|field| field.as_ref().clone())
                .collect::<Vec<_>>();
            fields.extend(system_columns.iter().map(|(field, _)| field.clone()));
            Arc::new(Schema::new_with_metadata(
                fields,
                input_schema.metadata().clone(),
            ))
        };
        let cached = output_schema_cache.get_or_init(|| CachedOutputSchema {
            input: input_schema.clone(),
            output: make_output_schema(),
        });
        let output_schema = if Arc::ptr_eq(&cached.input, &input_schema)
            || cached.input.as_ref() == input_schema.as_ref()
        {
            cached.output.clone()
        } else {
            make_output_schema()
        };
        let mut columns = Vec::with_capacity(batch.num_columns() + system_columns.len());
        columns.extend_from_slice(batch.columns());
        columns.extend(system_columns.into_iter().map(|(_, array)| array));
        RecordBatch::try_new_with_options(
            output_schema,
            columns,
            &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
        )?
    } else {
        system_columns
            .into_iter()
            .try_fold(batch, |batch, (field, array)| {
                batch.try_with_column(field, array)
            })?
    };

    match (deletion_mask, config.make_deletions_null) {
        (None, _) => Ok(batch),
        (Some(mask), false) => Ok(arrow::compute::filter_record_batch(&batch, &mask)?),
        (Some(mask), true) => Ok(apply_deletions_as_nulls(batch, &mask)?),
    }
}

/// Given a stream of batch tasks this function will add a row ids column (if requested)
/// and also apply a deletions vector to the batch.
///
/// This converts from BatchTaskStream to BatchFutStream because, if we are applying a
/// deletion vector, it is impossible to know how many output rows we will have.
pub fn wrap_with_row_id_and_delete(
    stream: ReadBatchTaskStream,
    fragment_id: u32,
    config: RowIdAndDeletesConfig,
) -> ReadBatchFutStream {
    let (row_id_cursor, use_dense_row_id_expansion) = config
        .row_id_sequence
        .as_ref()
        .filter(|_| config.with_row_id)
        .map(|sequence| {
            let (cursor, use_dense_range_expansion) = sequence.cursor_with_dense_range_expansion();
            (Some(cursor), use_dense_range_expansion)
        })
        .unwrap_or((None, false));
    if use_dense_row_id_expansion {
        wrap_with_row_id_and_delete_impl::<true>(stream, fragment_id, config, row_id_cursor)
    } else {
        wrap_with_row_id_and_delete_impl::<false>(stream, fragment_id, config, row_id_cursor)
    }
}

fn wrap_with_row_id_and_delete_impl<const USE_DENSE_ROW_ID_EXPANSION: bool>(
    stream: ReadBatchTaskStream,
    fragment_id: u32,
    config: RowIdAndDeletesConfig,
    mut row_id_cursor: Option<RowIdSequenceCursor>,
) -> ReadBatchFutStream {
    let config = Arc::new(config);
    let output_schema_cache = Arc::new(OnceLock::new());
    let mut row_id_chunk: Option<PrecomputedRowIdChunk> = None;
    let mut uniform_batch_size = None;
    let mut use_uniform_batch_fast_paths = true;
    let selected_rows = selected_row_count(&config.params, config.total_num_rows as usize);
    let mut last_updated_cursor = config
        .last_updated_at_sequence
        .as_ref()
        .filter(|sequence| config.with_row_last_updated_at_version && sequence.runs.len() > 1)
        .map(|sequence| sequence.cursor());
    let mut created_cursor = config
        .created_at_sequence
        .as_ref()
        .filter(|sequence| config.with_row_created_at_version && sequence.runs.len() > 1)
        .map(|sequence| sequence.cursor());
    let mut offset = 0;
    stream
        .map(move |batch_task| {
            let config = config.clone();
            let this_offset = offset;
            let num_rows = batch_task.num_rows;
            offset += num_rows;
            let logical_offset = this_offset as usize;
            let num_rows_usize = num_rows as usize;
            if num_rows_usize != 0 && use_uniform_batch_fast_paths {
                if let Some(batch_size) = uniform_batch_size {
                    // A shorter final batch is expected. Any other task-size change can
                    // repeatedly straddle read-ahead boundaries. Drain the current row-ID
                    // cache, then use exact per-task decoding and the original incremental
                    // system-column assembly from this point forward.
                    if num_rows_usize != batch_size
                        && logical_offset + num_rows_usize < selected_rows
                    {
                        use_uniform_batch_fast_paths = false;
                    }
                } else {
                    uniform_batch_size = Some(num_rows_usize);
                }
            }
            let output_schema_cache = use_uniform_batch_fast_paths
                .then(|| output_schema_cache.clone());
            // Build row ids while pulling the ordered task stream, before the
            // batch futures can run concurrently. Adjacent batches share a
            // bounded chunk and take zero-copy Arrow slices from it.
            let row_ids = config.row_id_sequence.as_ref().and_then(|sequence| {
                row_id_cursor.as_mut().map(|cursor| {
                    if num_rows_usize == 0 {
                        return Ok(Arc::new(UInt64Array::from(Vec::<u64>::new())));
                    }
                    if !use_uniform_batch_fast_paths
                        && row_id_chunk
                            .as_ref()
                            .is_none_or(|chunk| chunk.end_offset() <= logical_offset)
                    {
                        row_id_chunk = None;
                        return decode_row_id_chunk::<USE_DENSE_ROW_ID_EXPANSION>(
                            sequence,
                            cursor,
                            &config.params,
                            logical_offset,
                            num_rows_usize,
                        )
                        .map(|chunk| chunk.values);
                    }
                    if let Some(row_ids) = row_id_chunk
                        .as_ref()
                        .and_then(|chunk| chunk.slice(logical_offset, num_rows_usize))
                    {
                        return Ok(row_ids);
                    }

                    let prefix = row_id_chunk.as_ref().and_then(|chunk| {
                        let prefix_len = chunk.end_offset().checked_sub(logical_offset)?;
                        if prefix_len == 0 {
                            None
                        } else {
                            chunk.slice(logical_offset, prefix_len)
                        }
                    });
                    let decode_offset = logical_offset
                        + prefix
                            .as_ref()
                            .map(|row_ids| row_ids.len())
                            .unwrap_or_default();
                    let required_end = logical_offset + num_rows_usize;
                    let missing_rows = required_end.saturating_sub(decode_offset);
                    let chunk_len = if use_uniform_batch_fast_paths {
                        let batch_size = uniform_batch_size.unwrap_or(num_rows_usize);
                        let batches_per_chunk = (ROW_ID_READ_AHEAD_ROWS / batch_size).max(1);
                        let chunk_rows = batch_size.saturating_mul(batches_per_chunk);
                        missing_rows.max(chunk_rows)
                    } else {
                        missing_rows
                    }
                    .min(selected_rows.saturating_sub(decode_offset));
                    let chunk = decode_row_id_chunk::<USE_DENSE_ROW_ID_EXPANSION>(
                        sequence,
                        cursor,
                        &config.params,
                        decode_offset,
                        chunk_len,
                    )?;
                    let suffix = chunk.slice(decode_offset, missing_rows).ok_or_else(|| {
                        Error::corrupt_file_named(
                            "row ID metadata",
                            format!(
                                "decoded row ID chunk at selected offset {decode_offset} contains {} rows, but the current batch requires {missing_rows} more rows",
                                chunk.values.len()
                            ),
                        )
                    })?;
                    let row_ids = if let Some(prefix) = prefix {
                        let mut values = Vec::with_capacity(num_rows_usize);
                        values.extend_from_slice(prefix.values());
                        values.extend_from_slice(suffix.values());
                        Arc::new(UInt64Array::from(values))
                    } else {
                        suffix
                    };
                    row_id_chunk = Some(chunk);
                    Ok(row_ids)
                })
            });
            let last_updated_versions =
                config
                    .last_updated_at_sequence
                    .as_ref()
                    .and_then(|sequence| {
                        last_updated_cursor.as_mut().map(|cursor| {
                            version_values_for_selection_with_cursor(
                                sequence,
                                cursor,
                                &config.params,
                                this_offset,
                                num_rows,
                            )
                            .map(UInt64Array::from)
                            .map(Arc::new)
                        })
                    });
            let created_versions = config.created_at_sequence.as_ref().and_then(|sequence| {
                created_cursor.as_mut().map(|cursor| {
                    version_values_for_selection_with_cursor(
                        sequence,
                        cursor,
                        &config.params,
                        this_offset,
                        num_rows,
                    )
                    .map(UInt64Array::from)
                    .map(Arc::new)
                })
            });
            batch_task
                .task
                .map(move |batch| {
                    apply_row_id_and_deletes_with_system_columns(
                        batch?,
                        this_offset,
                        fragment_id,
                        config.as_ref(),
                        PrecomputedSystemColumns {
                            row_ids,
                            last_updated_versions,
                            created_versions,
                        },
                        output_schema_cache.as_deref(),
                    )
                })
                .boxed()
        })
        .boxed()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{array::AsArray, datatypes::UInt64Type};
    use arrow_array::{RecordBatch, UInt32Array, types::Int32Type};
    use arrow_schema::ArrowError;
    use futures::{
        FutureExt, StreamExt, TryStreamExt,
        stream::{self, BoxStream},
    };
    use lance_core::{
        ROW_ID,
        utils::{address::RowAddress, deletion::DeletionVector},
    };
    use lance_datagen::{BatchCount, RowCount};
    use lance_io::{ReadBatchParams, stream::arrow_stream_to_lance_stream};
    use roaring::RoaringBitmap;

    use crate::{rowids::RowIdSequence, utils::stream::ReadBatchTask};

    use super::RowIdAndDeletesConfig;

    fn batch_task_stream(
        datagen_stream: BoxStream<'static, std::result::Result<RecordBatch, ArrowError>>,
    ) -> super::ReadBatchTaskStream {
        arrow_stream_to_lance_stream(datagen_stream)
            .map(|batch| ReadBatchTask {
                num_rows: batch.as_ref().unwrap().num_rows() as u32,
                task: std::future::ready(batch).boxed(),
            })
            .boxed()
    }

    #[tokio::test]
    async fn test_basic_zip() {
        let left = batch_task_stream(
            lance_datagen::gen_batch()
                .col("x", lance_datagen::array::step::<Int32Type>())
                .into_reader_stream(RowCount::from(100), BatchCount::from(10))
                .0,
        );
        let right = batch_task_stream(
            lance_datagen::gen_batch()
                .col("y", lance_datagen::array::step::<Int32Type>())
                .into_reader_stream(RowCount::from(100), BatchCount::from(10))
                .0,
        );

        let merged = super::merge_streams(vec![left, right])
            .map(|batch_task| batch_task.task)
            .buffered(1)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        let expected = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .col("y", lance_datagen::array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(100), BatchCount::from(10))
            .collect::<Result<Vec<_>, ArrowError>>()
            .unwrap();
        assert_eq!(merged, expected);
    }

    #[tokio::test]
    async fn test_stable_row_ids_across_concurrent_batches_and_deletes() {
        let expected = (10_000..120_000)
            .filter(|row_id| row_id % 13 != 0)
            .collect::<Vec<u64>>();
        let row_id_sequence = Arc::new(RowIdSequence::try_from_iter(expected.clone()).unwrap());
        let deletion_offsets = (0..expected.len() as u32).step_by(997).collect::<Vec<_>>();
        let deletion_vector = Some(Arc::new(DeletionVector::Bitmap(
            deletion_offsets.iter().copied().collect(),
        )));

        let batches = expected
            .chunks(257)
            .map(|chunk| arrow_array::record_batch!(("x", Int32, vec![0; chunk.len()])).unwrap())
            .map(Ok)
            .collect::<Vec<std::result::Result<RecordBatch, ArrowError>>>();
        let data = batch_task_stream(stream::iter(batches).boxed());
        let config = RowIdAndDeletesConfig {
            params: ReadBatchParams::RangeFull,
            with_row_id: true,
            with_row_addr: true,
            with_row_last_updated_at_version: false,
            with_row_created_at_version: false,
            deletion_vector,
            row_id_sequence: Some(row_id_sequence),
            last_updated_at_sequence: None,
            created_at_sequence: None,
            make_deletions_null: false,
            total_num_rows: expected.len() as u32,
        };

        let batches = super::wrap_with_row_id_and_delete(data, 7, config)
            .buffered(8)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let actual_row_ids = batches
            .iter()
            .flat_map(|batch| batch[ROW_ID].as_primitive::<UInt64Type>().values())
            .copied()
            .collect::<Vec<_>>();
        let actual_row_addrs = batches
            .iter()
            .flat_map(|batch| {
                batch[lance_core::ROW_ADDR]
                    .as_primitive::<UInt64Type>()
                    .values()
            })
            .copied()
            .collect::<Vec<_>>();
        let expected_survivors = expected
            .iter()
            .enumerate()
            .filter(|(offset, _)| deletion_offsets.binary_search(&(*offset as u32)).is_err())
            .map(|(offset, row_id)| {
                (
                    *row_id,
                    u64::from(RowAddress::new_from_parts(7, offset as u32)),
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(
            actual_row_ids,
            expected_survivors
                .iter()
                .map(|(row_id, _)| *row_id)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            actual_row_addrs,
            expected_survivors
                .iter()
                .map(|(_, row_addr)| *row_addr)
                .collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn test_stable_row_ids_with_unsorted_indices() {
        let expected = (100..140)
            .filter(|row_id| row_id % 3 != 0)
            .collect::<Vec<u64>>();
        let indices = UInt32Array::from(vec![8, 2, 9, 1, 6]);
        let batches = [2, 2, 1].into_iter().map(|num_rows| ReadBatchTask {
            num_rows,
            task: std::future::ready(Ok(arrow_array::record_batch!((
                "x",
                Int32,
                vec![0; num_rows as usize]
            ))
            .unwrap()))
            .boxed(),
        });
        let config = RowIdAndDeletesConfig {
            params: ReadBatchParams::Indices(indices.clone()),
            with_row_id: true,
            with_row_addr: false,
            with_row_last_updated_at_version: false,
            with_row_created_at_version: false,
            deletion_vector: None,
            row_id_sequence: Some(Arc::new(
                RowIdSequence::try_from_iter(expected.clone()).unwrap(),
            )),
            last_updated_at_sequence: None,
            created_at_sequence: None,
            make_deletions_null: false,
            total_num_rows: expected.len() as u32,
        };

        let actual = super::wrap_with_row_id_and_delete(stream::iter(batches).boxed(), 7, config)
            .buffered(3)
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
            .iter()
            .flat_map(|batch| batch[ROW_ID].as_primitive::<UInt64Type>().values())
            .copied()
            .collect::<Vec<_>>();
        let expected = indices
            .values()
            .iter()
            .map(|index| expected[*index as usize])
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn test_repeated_row_id_after_bulk_segment_boundary() {
        let mut row_ids = RowIdSequence::from(0..5);
        row_ids.extend(RowIdSequence::from(10..20));
        let batches = [1_u32, 2].into_iter().map(|num_rows| ReadBatchTask {
            num_rows,
            task: std::future::ready(Ok(arrow_array::record_batch!((
                "x",
                Int32,
                vec![0; num_rows as usize]
            ))
            .unwrap()))
            .boxed(),
        });
        let config = RowIdAndDeletesConfig {
            params: ReadBatchParams::Indices(UInt32Array::from(vec![4, 4, 5])),
            with_row_id: true,
            with_row_addr: false,
            with_row_last_updated_at_version: false,
            with_row_created_at_version: false,
            deletion_vector: None,
            row_id_sequence: Some(Arc::new(row_ids)),
            last_updated_at_sequence: None,
            created_at_sequence: None,
            make_deletions_null: false,
            total_num_rows: 15,
        };

        let actual = super::wrap_with_row_id_and_delete(stream::iter(batches).boxed(), 0, config)
            .buffered(1)
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
            .iter()
            .flat_map(|batch| batch[ROW_ID].as_primitive::<UInt64Type>().values())
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(actual, vec![4, 4, 10]);
    }

    #[tokio::test]
    async fn test_stable_row_id_read_ahead_range_boundary_and_tail() {
        let all_row_ids = (10_000..120_000)
            .filter(|row_id| row_id % 11 != 0)
            .collect::<Vec<u64>>();
        let selection = 1_234..71_237;
        let selected_len = selection.len();
        let mut remaining = selected_len;
        let tasks = std::iter::from_fn(move || {
            if remaining == 0 {
                return None;
            }
            let num_rows = remaining.min(1_025);
            remaining -= num_rows;
            Some(ReadBatchTask {
                num_rows: num_rows as u32,
                task: std::future::ready(Ok(arrow_array::record_batch!((
                    "x",
                    Int32,
                    vec![0; num_rows]
                ))
                .unwrap()))
                .boxed(),
            })
        });
        let config = RowIdAndDeletesConfig {
            params: ReadBatchParams::Range(selection.clone()),
            with_row_id: true,
            with_row_addr: false,
            with_row_last_updated_at_version: false,
            with_row_created_at_version: false,
            deletion_vector: None,
            row_id_sequence: Some(Arc::new(
                RowIdSequence::try_from_iter(all_row_ids.clone()).unwrap(),
            )),
            last_updated_at_sequence: None,
            created_at_sequence: None,
            make_deletions_null: false,
            total_num_rows: all_row_ids.len() as u32,
        };

        let batches = super::wrap_with_row_id_and_delete(stream::iter(tasks).boxed(), 3, config)
            .buffered(8)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(batches.len(), 69);
        assert_eq!(batches.last().unwrap().num_rows(), 303);

        fn row_ids(batch: &RecordBatch) -> &arrow_array::UInt64Array {
            batch[ROW_ID].as_primitive::<UInt64Type>()
        }
        assert_eq!(
            row_ids(&batches[62]).values().as_ptr(),
            row_ids(&batches[0])
                .values()
                .as_ptr()
                .wrapping_add(62 * 1_025)
        );
        assert_eq!(
            row_ids(&batches[68]).values().as_ptr(),
            row_ids(&batches[64])
                .values()
                .as_ptr()
                .wrapping_add(4 * 1_025)
        );

        let actual = batches
            .iter()
            .flat_map(|batch| row_ids(batch).values())
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(actual, all_row_ids[selection]);
    }

    #[tokio::test]
    async fn test_stable_row_id_read_ahead_empty_task() {
        let tasks = [0_u32, 1].into_iter().map(|num_rows| ReadBatchTask {
            num_rows,
            task: std::future::ready(Ok(arrow_array::record_batch!((
                "x",
                Int32,
                vec![0; num_rows as usize]
            ))
            .unwrap()))
            .boxed(),
        });
        let config = RowIdAndDeletesConfig {
            params: ReadBatchParams::RangeFull,
            with_row_id: true,
            with_row_addr: false,
            with_row_last_updated_at_version: false,
            with_row_created_at_version: false,
            deletion_vector: None,
            row_id_sequence: Some(Arc::new(RowIdSequence::try_from_iter([42]).unwrap())),
            last_updated_at_sequence: None,
            created_at_sequence: None,
            make_deletions_null: false,
            total_num_rows: 1,
        };

        let batches = super::wrap_with_row_id_and_delete(stream::iter(tasks).boxed(), 0, config)
            .buffered(2)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(batches[0].num_rows(), 0);
        assert_eq!(
            batches[1][ROW_ID].as_primitive::<UInt64Type>().values(),
            &[42]
        );
    }

    #[tokio::test]
    async fn test_truncated_stable_row_ids_returns_error() {
        let task = ReadBatchTask {
            num_rows: 10,
            task: std::future::ready(Ok(
                arrow_array::record_batch!(("x", Int32, vec![0; 10])).unwrap()
            ))
            .boxed(),
        };
        let config = RowIdAndDeletesConfig {
            params: ReadBatchParams::RangeFull,
            with_row_id: true,
            with_row_addr: false,
            with_row_last_updated_at_version: false,
            with_row_created_at_version: false,
            deletion_vector: None,
            row_id_sequence: Some(Arc::new(RowIdSequence::try_from_iter(0_u64..5).unwrap())),
            last_updated_at_sequence: None,
            created_at_sequence: None,
            make_deletions_null: false,
            total_num_rows: 10,
        };

        let error = super::wrap_with_row_id_and_delete(stream::iter([task]).boxed(), 0, config)
            .buffered(1)
            .try_collect::<Vec<_>>()
            .await
            .unwrap_err();
        assert!(matches!(error, lance_core::Error::CorruptFile { .. }));
        assert!(error.to_string().contains(
            "decoded row ID chunk at selected offset 0 contains 5 rows, but the selection requires 10 rows"
        ));
    }

    #[tokio::test]
    async fn test_truncated_stable_row_ids_with_unsorted_indices_returns_error() {
        let tasks = (0..4).map(|_| ReadBatchTask {
            num_rows: 1,
            task: std::future::ready(Ok(
                arrow_array::record_batch!(("x", Int32, vec![0])).unwrap()
            ))
            .boxed(),
        });
        let config = RowIdAndDeletesConfig {
            params: ReadBatchParams::Indices(UInt32Array::from(vec![0, 5, 1, 2])),
            with_row_id: true,
            with_row_addr: false,
            with_row_last_updated_at_version: false,
            with_row_created_at_version: false,
            deletion_vector: None,
            row_id_sequence: Some(Arc::new(RowIdSequence::try_from_iter(0_u64..5).unwrap())),
            last_updated_at_sequence: None,
            created_at_sequence: None,
            make_deletions_null: false,
            total_num_rows: 6,
        };

        let error = super::wrap_with_row_id_and_delete(stream::iter(tasks).boxed(), 0, config)
            .buffered(1)
            .try_collect::<Vec<_>>()
            .await
            .unwrap_err();
        assert!(matches!(error, lance_core::Error::CorruptFile { .. }));
    }

    #[tokio::test]
    async fn test_stable_row_id_read_ahead_with_variable_task_boundaries() {
        let total_rows = super::ROW_ID_READ_AHEAD_ROWS * 3 + 41;
        let expected = (0_u64..)
            .filter(|row_id| row_id % 17 != 0)
            .take(total_rows)
            .collect::<Vec<_>>();
        let mut remaining = total_rows;
        let mut use_short_task = true;
        let tasks = std::iter::from_fn(move || {
            if remaining == 0 {
                return None;
            }
            let requested = if use_short_task { 32_768 } else { 32_769 };
            use_short_task = !use_short_task;
            let num_rows = remaining.min(requested);
            remaining -= num_rows;
            Some(ReadBatchTask {
                num_rows: num_rows as u32,
                task: std::future::ready(Ok(arrow_array::record_batch!((
                    "x",
                    Int32,
                    vec![0; num_rows]
                ))
                .unwrap()))
                .boxed(),
            })
        });
        let config = RowIdAndDeletesConfig {
            params: ReadBatchParams::RangeFull,
            with_row_id: true,
            with_row_addr: false,
            with_row_last_updated_at_version: false,
            with_row_created_at_version: false,
            deletion_vector: None,
            row_id_sequence: Some(Arc::new(
                RowIdSequence::try_from_iter(expected.clone()).unwrap(),
            )),
            last_updated_at_sequence: None,
            created_at_sequence: None,
            make_deletions_null: false,
            total_num_rows: total_rows as u32,
        };

        let batches = super::wrap_with_row_id_and_delete(stream::iter(tasks).boxed(), 0, config)
            .buffered(3)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let second_schema = batches[1].schema();
        let third_schema = batches[2].schema();
        assert!(!Arc::ptr_eq(&second_schema, &third_schema));

        let actual = batches
            .iter()
            .flat_map(|batch| batch[ROW_ID].as_primitive::<UInt64Type>().values())
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn test_system_columns_share_schema_for_equivalent_payload_batches() {
        let batches = (0..3)
            .map(|batch_index| {
                arrow_array::record_batch!((
                    "payload",
                    Int32,
                    (batch_index * 10..(batch_index + 1) * 10).collect::<Vec<_>>()
                ))
                .unwrap()
            })
            .collect::<Vec<_>>();
        assert!(!Arc::ptr_eq(&batches[0].schema(), &batches[1].schema()));
        let tasks = batches.into_iter().map(|batch| ReadBatchTask {
            num_rows: batch.num_rows() as u32,
            task: std::future::ready(Ok(batch)).boxed(),
        });
        let config = RowIdAndDeletesConfig {
            params: ReadBatchParams::RangeFull,
            with_row_id: true,
            with_row_addr: true,
            with_row_last_updated_at_version: true,
            with_row_created_at_version: true,
            deletion_vector: None,
            row_id_sequence: Some(Arc::new(
                RowIdSequence::try_from_iter((0..30).map(|row_id| 100 + row_id + row_id / 7))
                    .unwrap(),
            )),
            last_updated_at_sequence: None,
            created_at_sequence: None,
            make_deletions_null: false,
            total_num_rows: 30,
        };

        let batches = super::wrap_with_row_id_and_delete(stream::iter(tasks).boxed(), 7, config)
            .buffered(3)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let expected_fields = [
            "payload",
            lance_core::ROW_ID,
            lance_core::ROW_ADDR,
            lance_core::ROW_LAST_UPDATED_AT_VERSION,
            lance_core::ROW_CREATED_AT_VERSION,
        ];
        assert_eq!(
            batches[0]
                .schema()
                .fields()
                .iter()
                .map(|field| field.name().as_str())
                .collect::<Vec<_>>(),
            expected_fields
        );
        assert!(
            batches
                .windows(2)
                .all(|pair| Arc::ptr_eq(&pair[0].schema(), &pair[1].schema()))
        );
        assert!(batches.iter().all(|batch| batch.num_columns() == 5));
    }

    #[tokio::test]
    async fn test_zip_with_different_batch_boundaries() {
        let left_batch =
            arrow_array::record_batch!(("x", Int32, (0..10).collect::<Vec<_>>())).unwrap();
        let right_batch =
            arrow_array::record_batch!(("y", Int32, (10..20).collect::<Vec<_>>())).unwrap();
        let left = batch_task_stream(
            stream::iter([Ok(left_batch.slice(0, 6)), Ok(left_batch.slice(6, 4))]).boxed(),
        );
        let right = batch_task_stream(
            stream::iter([Ok(right_batch.slice(0, 4)), Ok(right_batch.slice(4, 6))]).boxed(),
        );

        let merged = super::merge_streams(vec![left, right])
            .map(|batch_task| batch_task.task)
            .buffered(3)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        let expected = vec![
            arrow_array::record_batch!(
                ("x", Int32, (0..4).collect::<Vec<_>>()),
                ("y", Int32, (10..14).collect::<Vec<_>>())
            )
            .unwrap(),
            arrow_array::record_batch!(
                ("x", Int32, (4..6).collect::<Vec<_>>()),
                ("y", Int32, (14..16).collect::<Vec<_>>())
            )
            .unwrap(),
            arrow_array::record_batch!(
                ("x", Int32, (6..10).collect::<Vec<_>>()),
                ("y", Int32, (16..20).collect::<Vec<_>>())
            )
            .unwrap(),
        ];
        assert_eq!(merged, expected);
    }

    async fn check_row_id(params: ReadBatchParams, expected: impl IntoIterator<Item = u32>) {
        let expected = Vec::from_iter(expected);

        for has_columns in [false, true] {
            for fragment_id in [0, 10] {
                // 100 rows across 10 batches of 10 rows
                let mut datagen = lance_datagen::gen_batch();
                if has_columns {
                    datagen = datagen.col("x", lance_datagen::array::rand::<Int32Type>());
                }
                let data = batch_task_stream(
                    datagen
                        .into_reader_stream(RowCount::from(10), BatchCount::from(10))
                        .0,
                );

                let config = RowIdAndDeletesConfig {
                    params: params.clone(),
                    with_row_id: true,
                    with_row_addr: false,
                    with_row_last_updated_at_version: false,
                    with_row_created_at_version: false,
                    deletion_vector: None,
                    row_id_sequence: None,
                    last_updated_at_sequence: None,
                    created_at_sequence: None,
                    make_deletions_null: false,
                    total_num_rows: 100,
                };
                let stream = super::wrap_with_row_id_and_delete(data, fragment_id, config);
                let batches = stream.buffered(1).try_collect::<Vec<_>>().await.unwrap();

                let mut offset = 0;
                let expected = expected.clone();
                for batch in batches {
                    let actual_row_ids =
                        batch[ROW_ID].as_primitive::<UInt64Type>().values().to_vec();
                    let expected_row_ids = expected[offset..offset + 10]
                        .iter()
                        .map(|row_offset| {
                            RowAddress::new_from_parts(fragment_id, *row_offset).into()
                        })
                        .collect::<Vec<u64>>();
                    assert_eq!(actual_row_ids, expected_row_ids);
                    offset += batch.num_rows();
                }
            }
        }
    }

    #[tokio::test]
    async fn test_row_id() {
        let some_indices = (0..100).rev().collect::<Vec<u32>>();
        let some_indices_arr = UInt32Array::from(some_indices.clone());
        check_row_id(ReadBatchParams::RangeFull, 0..100).await;
        check_row_id(ReadBatchParams::Indices(some_indices_arr), some_indices).await;
        check_row_id(ReadBatchParams::Range(1000..1100), 1000..1100).await;
        check_row_id(
            ReadBatchParams::RangeFrom(std::ops::RangeFrom { start: 1000 }),
            1000..1100,
        )
        .await;
        check_row_id(
            ReadBatchParams::RangeTo(std::ops::RangeTo { end: 1000 }),
            0..100,
        )
        .await;
    }

    #[tokio::test]
    async fn test_deletes() {
        let no_deletes: Option<Arc<DeletionVector>> = None;
        let no_deletes_2 = Some(Arc::new(DeletionVector::NoDeletions));
        let delete_some_bitmap = Some(Arc::new(DeletionVector::Bitmap(RoaringBitmap::from_iter(
            0..35,
        ))));
        let delete_some_set = Some(Arc::new(DeletionVector::Set((0..35).collect())));

        for deletion_vector in [
            no_deletes,
            no_deletes_2,
            delete_some_bitmap,
            delete_some_set,
        ] {
            for has_columns in [false, true] {
                for with_row_id in [false, true] {
                    for make_deletions_null in [false, true] {
                        for frag_id in [0, 1] {
                            let has_deletions = if let Some(dv) = &deletion_vector {
                                !matches!(dv.as_ref(), DeletionVector::NoDeletions)
                            } else {
                                false
                            };
                            if !has_columns && !has_deletions && !with_row_id {
                                // This is an invalid case and should be prevented upstream,
                                // no meaningful work is being done!
                                continue;
                            }
                            if make_deletions_null && !with_row_id {
                                // This is an invalid case and should be prevented upstream
                                // we cannot make the row_id column null if it isn't present
                                continue;
                            }

                            let mut datagen = lance_datagen::gen_batch();
                            if has_columns {
                                datagen =
                                    datagen.col("x", lance_datagen::array::rand::<Int32Type>());
                            }
                            // 100 rows across 10 batches of 10 rows
                            let data = batch_task_stream(
                                datagen
                                    .into_reader_stream(RowCount::from(10), BatchCount::from(10))
                                    .0,
                            );

                            let config = RowIdAndDeletesConfig {
                                params: ReadBatchParams::RangeFull,
                                with_row_id,
                                with_row_addr: false,
                                with_row_last_updated_at_version: false,
                                with_row_created_at_version: false,
                                deletion_vector: deletion_vector.clone(),
                                row_id_sequence: None,
                                last_updated_at_sequence: None,
                                created_at_sequence: None,
                                make_deletions_null,
                                total_num_rows: 100,
                            };
                            let stream = super::wrap_with_row_id_and_delete(data, frag_id, config);
                            let batches = stream
                                .buffered(1)
                                .filter_map(|batch| {
                                    std::future::ready(
                                        batch
                                            .map(|batch| {
                                                if batch.num_rows() == 0 {
                                                    None
                                                } else {
                                                    Some(batch)
                                                }
                                            })
                                            .transpose(),
                                    )
                                })
                                .try_collect::<Vec<_>>()
                                .await
                                .unwrap();

                            let total_num_rows =
                                batches.iter().map(|b| b.num_rows()).sum::<usize>();
                            let total_num_nulls = if make_deletions_null {
                                batches
                                    .iter()
                                    .map(|b| b[ROW_ID].null_count())
                                    .sum::<usize>()
                            } else {
                                0
                            };
                            let total_actually_deleted = total_num_nulls + (100 - total_num_rows);

                            let expected_deletions = match &deletion_vector {
                                None => 0,
                                Some(deletion_vector) => match deletion_vector.as_ref() {
                                    DeletionVector::NoDeletions => 0,
                                    DeletionVector::Bitmap(b) => b.len() as usize,
                                    DeletionVector::Set(s) => s.len(),
                                },
                            };
                            assert_eq!(total_actually_deleted, expected_deletions);
                            if expected_deletions > 0 && with_row_id {
                                if make_deletions_null {
                                    // If we make deletions null we get 3 batches of all-null and then
                                    // a batch of half-null
                                    assert_eq!(
                                        batches[3][ROW_ID].as_primitive::<UInt64Type>().value(0),
                                        u64::from(RowAddress::new_from_parts(frag_id, 30))
                                    );
                                    assert_eq!(batches[3][ROW_ID].null_count(), 5);
                                } else {
                                    // If we materialize deletions the first row will be 35
                                    assert_eq!(
                                        batches[0][ROW_ID].as_primitive::<UInt64Type>().value(0),
                                        u64::from(RowAddress::new_from_parts(frag_id, 35))
                                    );
                                }
                            }
                            if !with_row_id {
                                assert!(batches[0].column_by_name(ROW_ID).is_none());
                            }
                        }
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_version_column_with_deletions() {
        use crate::rowids::segment::U64Segment;
        use crate::rowids::version::{RowDatasetVersionRun, RowDatasetVersionSequence};

        let seq = Arc::new(RowDatasetVersionSequence {
            runs: vec![RowDatasetVersionRun {
                span: U64Segment::Range(0..100),
                version: 42,
            }],
        });

        let data = batch_task_stream(
            lance_datagen::gen_batch()
                .col("x", lance_datagen::array::rand::<Int32Type>())
                .into_reader_stream(RowCount::from(10), BatchCount::from(10))
                .0,
        );

        let config = RowIdAndDeletesConfig {
            params: ReadBatchParams::RangeFull,
            with_row_id: true,
            with_row_addr: false,
            with_row_last_updated_at_version: false,
            with_row_created_at_version: true,
            deletion_vector: Some(Arc::new(DeletionVector::Bitmap(RoaringBitmap::from_iter(
                0..35,
            )))),
            row_id_sequence: None,
            last_updated_at_sequence: None,
            created_at_sequence: Some(seq),
            make_deletions_null: false,
            total_num_rows: 100,
        };
        let stream = super::wrap_with_row_id_and_delete(data, 0, config);
        let batches: Vec<_> = stream
            .buffered(1)
            .try_filter(|b| std::future::ready(b.num_rows() > 0))
            .try_collect()
            .await
            .unwrap();

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 65);

        for batch in &batches {
            let versions = batch
                .column_by_name("_row_created_at_version")
                .unwrap()
                .as_primitive::<UInt64Type>()
                .values();
            assert!(versions.iter().all(|&v| v == 42));
        }
    }

    #[tokio::test]
    async fn test_version_column_multi_run() {
        use crate::rowids::segment::U64Segment;
        use crate::rowids::version::{RowDatasetVersionRun, RowDatasetVersionSequence};

        // Exercise the worst-case created-at shape: one run per row.
        let created_seq = Arc::new(RowDatasetVersionSequence {
            runs: (0..100)
                .map(|position| RowDatasetVersionRun {
                    span: U64Segment::Range(position..position + 1),
                    version: 1_000 + position,
                })
                .collect(),
        });
        // Also exercise irregular boundaries for last-updated-at.
        let last_updated_seq = Arc::new(RowDatasetVersionSequence {
            runs: vec![
                RowDatasetVersionRun {
                    span: U64Segment::Range(0..7),
                    version: 11,
                },
                RowDatasetVersionRun {
                    span: U64Segment::Range(7..20),
                    version: 22,
                },
                RowDatasetVersionRun {
                    span: U64Segment::Range(20..21),
                    version: 33,
                },
                RowDatasetVersionRun {
                    span: U64Segment::Range(21..50),
                    version: 44,
                },
                RowDatasetVersionRun {
                    span: U64Segment::Range(50..100),
                    version: 55,
                },
            ],
        });

        // Delete 0..20 and 60..80 (spans run boundary).
        // Survivors: 20..40 (v1), 40..60 (v2), 80..100 (v3) = 60 rows
        let mut deletions = RoaringBitmap::from_iter(0..20);
        deletions.extend(60..80);

        let data = batch_task_stream(
            lance_datagen::gen_batch()
                .col("x", lance_datagen::array::rand::<Int32Type>())
                .into_reader_stream(RowCount::from(10), BatchCount::from(10))
                .0,
        );

        let config = RowIdAndDeletesConfig {
            params: ReadBatchParams::RangeFull,
            with_row_id: true,
            with_row_addr: false,
            with_row_last_updated_at_version: true,
            with_row_created_at_version: true,
            deletion_vector: Some(Arc::new(DeletionVector::Bitmap(deletions))),
            row_id_sequence: None,
            last_updated_at_sequence: Some(last_updated_seq),
            created_at_sequence: Some(created_seq),
            make_deletions_null: false,
            total_num_rows: 100,
        };
        let stream = super::wrap_with_row_id_and_delete(data, 0, config);
        let batches: Vec<_> = stream
            .buffered(8)
            .try_filter(|b| std::future::ready(b.num_rows() > 0))
            .try_collect()
            .await
            .unwrap();

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 60);

        let created_versions: Vec<u64> = batches
            .iter()
            .flat_map(|b| {
                b.column_by_name("_row_created_at_version")
                    .unwrap()
                    .as_primitive::<UInt64Type>()
                    .values()
                    .to_vec()
            })
            .collect();
        let last_updated_versions: Vec<u64> = batches
            .iter()
            .flat_map(|b| {
                b.column_by_name("_row_last_updated_at_version")
                    .unwrap()
                    .as_primitive::<UInt64Type>()
                    .values()
                    .to_vec()
            })
            .collect();
        let surviving_positions: Vec<u64> = (20..60).chain(80..100).collect();
        let expected_created: Vec<u64> = surviving_positions
            .iter()
            .map(|position| 1_000 + position)
            .collect();
        let expected_last_updated: Vec<u64> = surviving_positions
            .iter()
            .map(|position| match position {
                0..=6 => 11,
                7..=19 => 22,
                20 => 33,
                21..=49 => 44,
                _ => 55,
            })
            .collect();

        assert_eq!(created_versions, expected_created);
        assert_eq!(last_updated_versions, expected_last_updated);
    }

    #[tokio::test]
    async fn test_version_column_with_unsorted_indices_across_batches() {
        use crate::rowids::segment::U64Segment;
        use crate::rowids::version::{RowDatasetVersionRun, RowDatasetVersionSequence};

        let sequence = Arc::new(RowDatasetVersionSequence {
            runs: (0..10)
                .map(|position| RowDatasetVersionRun {
                    span: U64Segment::Range(position..position + 1),
                    version: 100 + position,
                })
                .collect(),
        });
        let indices = UInt32Array::from(vec![8, 2, 9, 1, 6]);
        let batches = [2, 2, 1].into_iter().map(|num_rows| ReadBatchTask {
            num_rows,
            task: std::future::ready(Ok(arrow_array::record_batch!((
                "x",
                Int32,
                vec![0; num_rows as usize]
            ))
            .unwrap()))
            .boxed(),
        });
        let config = RowIdAndDeletesConfig {
            params: ReadBatchParams::Indices(indices.clone()),
            with_row_id: false,
            with_row_addr: false,
            with_row_last_updated_at_version: true,
            with_row_created_at_version: false,
            deletion_vector: None,
            row_id_sequence: None,
            last_updated_at_sequence: Some(sequence),
            created_at_sequence: None,
            make_deletions_null: false,
            total_num_rows: 10,
        };

        let actual = super::wrap_with_row_id_and_delete(stream::iter(batches).boxed(), 0, config)
            .buffered(3)
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
            .iter()
            .flat_map(|batch| {
                batch["_row_last_updated_at_version"]
                    .as_primitive::<UInt64Type>()
                    .values()
            })
            .copied()
            .collect::<Vec<_>>();
        let expected = indices
            .values()
            .iter()
            .map(|position| 100 + u64::from(*position))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_apply_version_column_direct_call_fallback() {
        use crate::rowids::segment::U64Segment;
        use crate::rowids::version::{RowDatasetVersionRun, RowDatasetVersionSequence};

        let sequence = Arc::new(RowDatasetVersionSequence {
            runs: (0..5)
                .map(|position| RowDatasetVersionRun {
                    span: U64Segment::Range(position..position + 1),
                    version: 10 + position,
                })
                .collect(),
        });
        let config = RowIdAndDeletesConfig {
            params: ReadBatchParams::Indices(UInt32Array::from(vec![4, 1, 3])),
            with_row_id: false,
            with_row_addr: false,
            with_row_last_updated_at_version: true,
            with_row_created_at_version: false,
            deletion_vector: None,
            row_id_sequence: None,
            last_updated_at_sequence: Some(sequence),
            created_at_sequence: None,
            make_deletions_null: false,
            total_num_rows: 5,
        };
        let batch = arrow_array::record_batch!(("x", Int32, vec![0; 3])).unwrap();

        let actual = super::apply_row_id_and_deletes(batch, 0, 0, &config).unwrap();
        assert_eq!(
            actual["_row_last_updated_at_version"]
                .as_primitive::<UInt64Type>()
                .values(),
            &[14, 11, 13]
        );
    }
}
