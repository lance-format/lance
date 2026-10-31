// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

mod plan;
mod reference;

pub use reference::{
    BlobRefBinding, BlobRefDemand, BlobRefReadStream, BlobRefReader, BlobRefRequest, ReadBlobRef,
};

use std::{
    collections::{BTreeMap, HashMap},
    ops::Range,
    sync::Arc,
};

use arrow::array::AsArray;
use arrow::datatypes::UInt64Type;
use arrow_array::{Array, RecordBatch, StructArray};
use bytes::Bytes;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt, stream};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};

pub(super) use self::plan::BlobEntry;
use self::plan::{
    BlobReadBatch, BlobReadPlan, ResolvedBlobSelection, plan_blob_read_batches,
    plan_blob_read_plans, plan_disjoint_blob_reads,
};
use super::file::{BlobFile, BlobSource};
use super::locate::{blob_version_from_descriptions, collect_blob_files_v1, collect_blob_files_v2};
use crate::dataset::take::{MissingRowPolicy, TakeBuilder};
use crate::dataset::{Dataset, ProjectionRequest};
use lance_core::datatypes::{BlobVersion, parse_field_path};
use lance_core::{Error, Result};

/// Blob bytes materialized by [`ReadBlobsBuilder`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadBlob {
    /// Row address of the blob that was read.
    pub row_address: u64,
    /// Blob payload bytes, or `None` when the selected blob value is null.
    ///
    /// A valid empty blob is represented as `Some(Bytes::new())`.
    pub data: Option<Bytes>,
}

/// A byte range relative to the beginning of one logical blob value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlobReadRange {
    /// Byte offset from the beginning of the blob value.
    pub offset: u64,
    /// Number of bytes to read.
    pub length: u64,
}

impl BlobReadRange {
    /// Create a blob-local byte range from an offset and length.
    pub const fn new(offset: u64, length: u64) -> Self {
        Self { offset, length }
    }

    fn checked_range(self, request_index: usize) -> Result<Range<u64>> {
        let end = self.offset.checked_add(self.length).ok_or_else(|| {
            Error::invalid_input(format!(
                "Blob range request {request_index} offset + length overflowed u64: offset={}, length={}",
                self.offset, self.length
            ))
        })?;
        Ok(self.offset..end)
    }
}

/// One row-specific blob range read request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlobRangeRequest {
    /// Row selector interpreted by the selection method on
    /// [`ReadBlobRangesBuilder`].
    pub row: u64,
    /// Blob-local byte range to read from the selected row.
    pub range: BlobReadRange,
}

impl BlobRangeRequest {
    /// Create a request for `length` bytes at `offset` in the selected row.
    pub const fn new(row: u64, offset: u64, length: u64) -> Self {
        Self {
            row,
            range: BlobReadRange::new(offset, length),
        }
    }
}

/// Bytes materialized for one request submitted through [`ReadBlobRangesBuilder`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadBlobRange {
    /// Zero-based position of this request in the caller's request list.
    pub request_index: usize,
    /// Physical row address of the selected blob value.
    pub row_address: u64,
    /// Blob-local range supplied for this request.
    pub range: BlobReadRange,
    /// Bytes in `range`, or `None` when the selected blob value is null.
    ///
    /// An empty range on a non-null blob is represented as
    /// `Some(Bytes::new())`.
    pub data: Option<Bytes>,
}

/// Stream returned by [`ReadBlobsBuilder::try_into_stream`].
pub type ReadBlobsStream = BoxStream<'static, Result<ReadBlob>>;

/// Stream returned by [`ReadBlobRangesBuilder::try_into_stream`].
pub type ReadBlobRangesStream = BoxStream<'static, Result<ReadBlobRange>>;

/// Row selector configured on [`ReadBlobsBuilder`].
#[derive(Debug, Clone)]
enum ReadBlobsSelection {
    None,
    RowIds(Vec<u64>),
    RowIndices(Vec<u64>),
    RowAddresses(Vec<u64>),
}

#[derive(Debug, Clone)]
enum ReadBlobRangesSelection {
    None,
    RowIds(Vec<BlobRangeRequest>),
    RowIndices(Vec<BlobRangeRequest>),
    RowAddresses(Vec<BlobRangeRequest>),
}

impl ReadBlobRangesSelection {
    fn requests(&self) -> Option<&[BlobRangeRequest]> {
        match self {
            Self::None => None,
            Self::RowIds(requests) | Self::RowIndices(requests) | Self::RowAddresses(requests) => {
                Some(requests)
            }
        }
    }

    fn into_row_selection(self) -> Option<(ReadBlobsSelection, Vec<BlobRangeRequest>)> {
        match self {
            Self::None => None,
            Self::RowIds(requests) => {
                let rows = requests.iter().map(|request| request.row).collect();
                Some((ReadBlobsSelection::RowIds(rows), requests))
            }
            Self::RowIndices(requests) => {
                let rows = requests.iter().map(|request| request.row).collect();
                Some((ReadBlobsSelection::RowIndices(rows), requests))
            }
            Self::RowAddresses(requests) => {
                let rows = requests.iter().map(|request| request.row).collect();
                Some((ReadBlobsSelection::RowAddresses(rows), requests))
            }
        }
    }
}

/// Planner knobs for [`ReadBlobsBuilder`].
///
/// Options that shape how `read_blobs` uses Lance's existing schedulers.
#[derive(Debug, Clone)]
struct ReadBlobsOptions {
    io_buffer_size_bytes: Option<u64>,
    preserve_order: bool,
}

impl Default for ReadBlobsOptions {
    fn default() -> Self {
        Self {
            io_buffer_size_bytes: None,
            preserve_order: true,
        }
    }
}

/// Builder for sequential / planned blob reads.
///
/// Unlike [`Dataset::take_blobs`], which returns [`BlobFile`] handles for
/// caller-driven random access, this builder plans object-store reads across a
/// selected row set and yields fully materialized blob payloads.
#[derive(Debug, Clone)]
pub struct ReadBlobsBuilder {
    dataset: Arc<Dataset>,
    column: String,
    blob_field_id: u32,
    selection: ReadBlobsSelection,
    options: ReadBlobsOptions,
}

impl ReadBlobsBuilder {
    pub(crate) fn new(dataset: Arc<Dataset>, column: String, blob_field_id: u32) -> Self {
        Self {
            dataset,
            column,
            blob_field_id,
            selection: ReadBlobsSelection::None,
            options: ReadBlobsOptions::default(),
        }
    }

    /// Read blobs for the provided stable row ids.
    pub fn with_row_ids(mut self, row_ids: impl Into<Vec<u64>>) -> Self {
        self.selection = ReadBlobsSelection::RowIds(row_ids.into());
        self
    }

    /// Read blobs for the provided row offsets in dataset order.
    pub fn with_row_indices(mut self, row_indices: impl Into<Vec<u64>>) -> Self {
        self.selection = ReadBlobsSelection::RowIndices(row_indices.into());
        self
    }

    /// Read blobs for the provided physical row addresses.
    pub fn with_row_addresses(mut self, row_addrs: impl Into<Vec<u64>>) -> Self {
        self.selection = ReadBlobsSelection::RowAddresses(row_addrs.into());
        self
    }

    /// Set the scheduler I/O buffer size used while materializing blobs.
    pub fn with_io_buffer_size_bytes(mut self, bytes: u64) -> Self {
        self.options.io_buffer_size_bytes = Some(bytes);
        self
    }

    /// Whether results must follow the caller's requested row order.
    pub fn preserve_order(mut self, preserve: bool) -> Self {
        self.options.preserve_order = preserve;
        self
    }

    /// Execute the planned blob read and return a stream of blob payloads.
    ///
    /// The stream yields one [`ReadBlob`] per selected row. Null blob values
    /// have `data` set to `None`; valid empty blobs contain an empty buffer.
    pub async fn try_into_stream(self) -> Result<ReadBlobsStream> {
        self.validate()?;
        let collected = collect_blob_selection_for_selection(
            &self.dataset,
            self.blob_field_id,
            &self.column,
            &self.selection,
        )
        .await?;
        let partitioned =
            partition_blob_selections(collected, |selection_index, row_address, range| {
                if range.is_some() {
                    return Err(Error::internal(format!(
                        "Whole-blob selection {} unexpectedly carried range metadata",
                        selection_index
                    )));
                }
                Ok(ReadBlob {
                    row_address,
                    data: None,
                })
            })?;
        let physical_buffer_size = self.options.io_buffer_size_bytes.unwrap_or_else(|| {
            SchedulerConfig::max_bandwidth(self.dataset.object_store.as_ref()).io_buffer_size_bytes
        });
        let batches = plan_blob_read_batches(partitioned.reads, physical_buffer_size)?;
        let execution = Arc::new(ReadBlobsExecution::new(self.options.io_buffer_size_bytes));
        let non_null_blobs = execute_blob_read_batches_stream(
            batches,
            execution,
            self.dataset.object_store.io_parallelism(),
        )
        .map_ok(|blob| {
            let selection_index = blob.selection_index;
            (selection_index, into_read_blob(blob))
        })
        .boxed();
        Ok(totalize_blob_selection_stream(
            non_null_blobs,
            partitioned.ready_values,
            partitioned.selection_count,
            self.options.preserve_order,
        ))
    }

    /// Execute the planned blob read and collect the full result in memory.
    pub async fn execute(self) -> Result<Vec<ReadBlob>> {
        self.try_into_stream().await?.try_collect().await
    }

    fn validate(&self) -> Result<()> {
        match self.selection {
            ReadBlobsSelection::None => Err(Error::invalid_input(
                "ReadBlobsBuilder requires a row selection; call one of with_row_ids, with_row_indices, or with_row_addresses".to_string(),
            )),
            _ if self.options.io_buffer_size_bytes == Some(0) => Err(Error::invalid_input(
                "ReadBlobsBuilder io_buffer_size must be greater than 0".to_string(),
            )),
            _ => Ok(()),
        }
    }
}

/// Builder for planned blob-local range reads.
///
/// Planning retains `O(request_count)` metadata. [`Self::try_into_stream`]
/// bounds each scheduler-visible physical batch by the configured I/O buffer
/// size after accounting for range coalescing, except that one request larger
/// than the buffer must be materialized by itself. The existing scheduler
/// separately applies byte backpressure to physical I/O. [`Self::execute`]
/// additionally retains all returned payload bytes.
#[derive(Debug, Clone)]
pub struct ReadBlobRangesBuilder {
    inner: ReadBlobsBuilder,
    selection: ReadBlobRangesSelection,
}

impl ReadBlobRangesBuilder {
    pub(crate) fn new(inner: ReadBlobsBuilder) -> Self {
        Self {
            inner,
            selection: ReadBlobRangesSelection::None,
        }
    }

    /// Read requests whose row values are stable row ids.
    pub fn with_row_ids(mut self, requests: impl IntoIterator<Item = BlobRangeRequest>) -> Self {
        self.selection = ReadBlobRangesSelection::RowIds(requests.into_iter().collect());
        self
    }

    /// Read requests whose row values are offsets in dataset order.
    pub fn with_row_indices(
        mut self,
        requests: impl IntoIterator<Item = BlobRangeRequest>,
    ) -> Self {
        self.selection = ReadBlobRangesSelection::RowIndices(requests.into_iter().collect());
        self
    }

    /// Read requests whose row values are physical row addresses.
    pub fn with_row_addresses(
        mut self,
        requests: impl IntoIterator<Item = BlobRangeRequest>,
    ) -> Self {
        self.selection = ReadBlobRangesSelection::RowAddresses(requests.into_iter().collect());
        self
    }

    /// Set the scheduler I/O buffer size used while materializing ranges.
    pub fn with_io_buffer_size_bytes(mut self, bytes: u64) -> Self {
        self.inner.options.io_buffer_size_bytes = Some(bytes);
        self
    }

    /// Whether results must follow the caller's request order.
    pub fn preserve_order(mut self, preserve: bool) -> Self {
        self.inner.options.preserve_order = preserve;
        self
    }

    /// Execute the planned range read and return one result per request.
    ///
    /// Null blob requests have `data` set to `None`. Empty ranges on non-null
    /// blobs contain an empty buffer without issuing payload I/O. By default,
    /// results follow input order; `request_index` identifies the original
    /// request even when ordering is disabled. Blob-local bounds are not
    /// evaluated for null values because they have no logical payload length.
    pub async fn try_into_stream(self) -> Result<ReadBlobRangesStream> {
        self.validate()?;
        let (row_selection, requests) = self.selection.into_row_selection().ok_or_else(|| {
            Error::internal("Validated blob range selection was missing".to_string())
        })?;
        let mut selections = collect_blob_selection_for_selection(
            &self.inner.dataset,
            self.inner.blob_field_id,
            &self.inner.column,
            &row_selection,
        )
        .await?;
        if selections.len() != requests.len() {
            return Err(Error::internal(format!(
                "Resolved blob selection count {} did not match range request count {}",
                selections.len(),
                requests.len()
            )));
        }
        for (selection, request) in selections.iter_mut().zip(requests) {
            selection.requested_range = Some(request.range);
        }
        let partitioned = partition_blob_selections(
            selections,
            |request_index, row_address, requested_range| {
                let range = requested_range.ok_or_else(|| {
                    Error::internal(format!(
                        "Blob range request {} was missing range metadata",
                        request_index
                    ))
                })?;
                Ok(ReadBlobRange {
                    request_index,
                    row_address,
                    range,
                    data: None,
                })
            },
        )?;
        let physical_buffer_size = self.inner.options.io_buffer_size_bytes.unwrap_or_else(|| {
            SchedulerConfig::max_bandwidth(self.inner.dataset.object_store.as_ref())
                .io_buffer_size_bytes
        });
        let batches = plan_blob_read_batches(partitioned.reads, physical_buffer_size)?;
        let execution = Arc::new(ReadBlobsExecution::new(
            self.inner.options.io_buffer_size_bytes,
        ));
        let non_null_ranges = execute_blob_read_batches_stream(
            batches,
            execution,
            self.inner.dataset.object_store.io_parallelism(),
        )
        .map(|result| {
            result.and_then(|blob| {
                let selection_index = blob.selection_index;
                into_read_blob_range(blob).map(|range| (selection_index, range))
            })
        })
        .boxed();
        Ok(totalize_blob_selection_stream(
            non_null_ranges,
            partitioned.ready_values,
            partitioned.selection_count,
            self.inner.options.preserve_order,
        ))
    }

    /// Execute the planned range read and collect all returned bytes in memory.
    pub async fn execute(self) -> Result<Vec<ReadBlobRange>> {
        self.try_into_stream().await?.try_collect().await
    }

    fn validate(&self) -> Result<()> {
        let requests = self.selection.requests().ok_or_else(|| {
            Error::invalid_input(
                "ReadBlobRangesBuilder requires requests; call one of with_row_ids, with_row_indices, or with_row_addresses".to_string(),
            )
        })?;
        if self.inner.options.io_buffer_size_bytes == Some(0) {
            return Err(Error::invalid_input(
                "ReadBlobRangesBuilder io_buffer_size must be greater than 0".to_string(),
            ));
        }
        for (request_index, request) in requests.iter().enumerate() {
            request.range.checked_range(request_index)?;
        }
        Ok(())
    }
}

/// Scheduler cache shared by one operation or materialization context.
///
/// Each authenticated object-store instance gets its own [`ScanScheduler`]. A
/// reference-reader batch can attach reservations retained by its active readers.
#[derive(Debug)]
pub(super) struct ReadBlobsExecution {
    io_buffer_size_bytes: Option<u64>,
    schedulers: Arc<std::sync::Mutex<HashMap<usize, Arc<ScanScheduler>>>>,
    reservation: Option<Arc<reference::ReadReservation>>,
}

impl ReadBlobsExecution {
    pub(super) fn new(io_buffer_size_bytes: Option<u64>) -> Self {
        Self {
            io_buffer_size_bytes,
            schedulers: Arc::new(std::sync::Mutex::new(HashMap::new())),
            reservation: None,
        }
    }

    fn scheduler_for(&self, source: &BlobSource) -> Arc<ScanScheduler> {
        let mut schedulers = self.schedulers.lock().unwrap();
        schedulers
            .entry(Arc::as_ptr(&source.object_store) as usize)
            .or_insert_with(|| {
                let config = self
                    .io_buffer_size_bytes
                    .map(SchedulerConfig::new)
                    .unwrap_or_else(|| {
                        SchedulerConfig::max_bandwidth(source.object_store.as_ref())
                    });
                ScanScheduler::new(source.object_store.clone(), config)
            })
            .clone()
    }
}

/// Materialized blob bytes plus the original selection index used to restore
/// caller ordering after per-source reads complete.
#[derive(Debug)]
pub(super) struct IndexedReadBlob {
    pub(super) selection_index: usize,
    row_address: u64,
    requested_range: Option<BlobReadRange>,
    pub(super) data: Bytes,
}

fn into_read_blob(blob: IndexedReadBlob) -> ReadBlob {
    debug_assert!(
        blob.requested_range.is_none(),
        "whole-blob reads must not carry a requested range"
    );
    ReadBlob {
        row_address: blob.row_address,
        data: Some(blob.data),
    }
}

struct PartitionedBlobSelection<T> {
    selection_count: usize,
    reads: Vec<BlobEntry>,
    ready_values: Vec<(usize, T)>,
}

fn partition_blob_selections<T>(
    selections: Vec<ResolvedBlobSelection>,
    mut make_ready_value: impl FnMut(usize, u64, Option<BlobReadRange>) -> Result<T>,
) -> Result<PartitionedBlobSelection<T>> {
    let selection_count = selections.len();
    let read_count = selections
        .iter()
        .filter(|selection| selection.file.is_some())
        .count();
    let mut reads = Vec::with_capacity(read_count);
    let mut ready_values = Vec::with_capacity(selection_count - read_count);
    for (selection_index, selection) in selections.into_iter().enumerate() {
        match selection.file {
            Some(file) => reads.push(BlobEntry {
                selection_index,
                row_address: selection.row_address,
                file,
                requested_range: selection.requested_range,
            }),
            None => {
                let value = make_ready_value(
                    selection_index,
                    selection.row_address,
                    selection.requested_range,
                )?;
                ready_values.push((selection_index, value));
            }
        }
    }
    Ok(PartitionedBlobSelection {
        selection_count,
        reads,
        ready_values,
    })
}

fn totalize_blob_selection_stream<T: Send + 'static>(
    read_values: BoxStream<'static, Result<(usize, T)>>,
    ready_values: Vec<(usize, T)>,
    selection_count: usize,
    preserve_order: bool,
) -> BoxStream<'static, Result<T>> {
    if preserve_order {
        return totalize_ordered_blob_selection_stream(read_values, ready_values, selection_count);
    }

    let ready_values = stream::iter(ready_values.into_iter().map(Ok::<(usize, T), Error>));
    let values = stream::select(ready_values, read_values).boxed();
    totalize_unordered_blob_selection_stream(values, selection_count)
}

struct UnorderedBlobSelectionState<T> {
    values: BoxStream<'static, Result<(usize, T)>>,
    seen_selection_indices: Vec<bool>,
    remaining: usize,
}

fn totalize_unordered_blob_selection_stream<T: Send + 'static>(
    values: BoxStream<'static, Result<(usize, T)>>,
    selection_count: usize,
) -> BoxStream<'static, Result<T>> {
    let state = UnorderedBlobSelectionState {
        values,
        seen_selection_indices: vec![false; selection_count],
        remaining: selection_count,
    };
    stream::try_unfold(state, move |mut state| async move {
        let Some((selection_index, value)) = state.values.try_next().await? else {
            return if state.remaining == 0 {
                Ok(None)
            } else {
                Err(Error::internal(format!(
                    "planned blob read stream completed after producing {} of {} selections",
                    selection_count - state.remaining,
                    selection_count
                )))
            };
        };

        if selection_index >= selection_count {
            return Err(Error::internal(format!(
                "Blob selection index {} exceeded selected row count {}",
                selection_index, selection_count
            )));
        }
        if state.seen_selection_indices[selection_index] {
            return Err(Error::internal(format!(
                "Blob selection index {} was produced more than once",
                selection_index
            )));
        }
        state.seen_selection_indices[selection_index] = true;
        state.remaining -= 1;
        Ok(Some((value, state)))
    })
    .boxed()
}

struct OrderedBlobSelectionState<T> {
    read_values: BoxStream<'static, Result<(usize, T)>>,
    ready: BTreeMap<usize, T>,
    next_selection_index: usize,
}

fn totalize_ordered_blob_selection_stream<T: Send + 'static>(
    read_values: BoxStream<'static, Result<(usize, T)>>,
    ready_values: Vec<(usize, T)>,
    selection_count: usize,
) -> BoxStream<'static, Result<T>> {
    // Seed the reorder buffer so ordered consumers do not start later physical
    // I/O while the next logical selection is already available.
    let mut ready = BTreeMap::new();
    for (selection_index, value) in ready_values {
        if selection_index >= selection_count {
            let error = Error::internal(format!(
                "Blob selection index {} exceeded selected row count {}",
                selection_index, selection_count
            ));
            return stream::once(async move { Err(error) }).boxed();
        }
        if ready.insert(selection_index, value).is_some() {
            let error = Error::internal(format!(
                "Blob selection index {} was produced more than once",
                selection_index
            ));
            return stream::once(async move { Err(error) }).boxed();
        }
    }
    let state = OrderedBlobSelectionState {
        read_values,
        ready,
        next_selection_index: 0,
    };
    stream::try_unfold(state, move |mut state| async move {
        loop {
            if let Some(value) = state.ready.remove(&state.next_selection_index) {
                state.next_selection_index += 1;
                return Ok(Some((value, state)));
            }

            match state.read_values.try_next().await? {
                Some((selection_index, value)) => {
                    if selection_index >= selection_count {
                        return Err(Error::internal(format!(
                            "Blob selection index {} exceeded selected row count {}",
                            selection_index, selection_count
                        )));
                    }
                    if selection_index < state.next_selection_index {
                        return Err(Error::internal(format!(
                            "Blob selection index {} was produced more than once",
                            selection_index
                        )));
                    }
                    if state.ready.insert(selection_index, value).is_some() {
                        return Err(Error::internal(format!(
                            "Blob selection index {} was produced more than once",
                            selection_index
                        )));
                    }
                }
                None if state.next_selection_index == selection_count => return Ok(None),
                None => {
                    return Err(Error::internal(format!(
                        "planned blob read stream completed before selection index {} was produced",
                        state.next_selection_index
                    )));
                }
            }
        }
    })
    .boxed()
}

fn into_read_blob_range(blob: IndexedReadBlob) -> Result<ReadBlobRange> {
    let range = blob.requested_range.ok_or_else(|| {
        Error::internal(format!(
            "Blob range request {} completed without range metadata",
            blob.selection_index
        ))
    })?;
    Ok(ReadBlobRange {
        request_index: blob.selection_index,
        row_address: blob.row_address,
        range,
        data: Some(blob.data),
    })
}

/// Execute one per-source blob read plan with a single scheduler submission.
async fn execute_blob_read_plan(
    task: BlobReadPlan,
    execution: Arc<ReadBlobsExecution>,
) -> Result<Vec<IndexedReadBlob>> {
    let (physical_ranges, slices) = plan_disjoint_blob_reads(&task.reads);
    let mut bytes = vec![Bytes::new(); task.reads.len()];
    if let Some(first_range) = physical_ranges.first() {
        let scheduler = execution.scheduler_for(&task.source);
        // The plan already bounds physical reads; probing the backing object's
        // size would add a metadata request without validating the blob slice.
        let reader = task.source.object_store.open(&task.source.path).await?;
        let reader = match &execution.reservation {
            Some(reservation) => Arc::new(reference::ReservedReader {
                reader: reader.into(),
                reservation: reservation.clone(),
            }) as Arc<dyn lance_io::traits::Reader>,
            None => reader.into(),
        };
        let file_scheduler = scheduler.open_reader(reader);
        let priority = first_range.start;
        let physical_range_count = physical_ranges.len();
        let returned = file_scheduler
            .submit_request(physical_ranges, priority)
            .await?;
        if returned.len() != physical_range_count {
            return Err(Error::internal(format!(
                "Blob read scheduler returned {} ranges for {} disjoint physical ranges from {}",
                returned.len(),
                physical_range_count,
                task.source.path
            )));
        }
        for slice in slices {
            let start = usize::try_from(slice.relative_range.start).map_err(|_| {
                Error::internal(format!(
                    "Blob read slice start {} does not fit into usize for {}",
                    slice.relative_range.start, task.source.path
                ))
            })?;
            let end = usize::try_from(slice.relative_range.end).map_err(|_| {
                Error::internal(format!(
                    "Blob read slice end {} does not fit into usize for {}",
                    slice.relative_range.end, task.source.path
                ))
            })?;
            let data = &returned[slice.physical_range_index];
            if end > data.len() {
                return Err(Error::internal(format!(
                    "Blob read slice {:?} exceeds the {} bytes returned for physical range {} from {}",
                    slice.relative_range,
                    data.len(),
                    slice.physical_range_index,
                    task.source.path
                )));
            }
            bytes[slice.read_index] = data.slice(start..end);
        }
    }

    Ok(task
        .reads
        .into_iter()
        .zip(bytes)
        .map(|(read, data)| IndexedReadBlob {
            selection_index: read.selection_index,
            row_address: read.row_address,
            requested_range: read.requested_range,
            data,
        })
        .collect())
}

fn execute_blob_read_batches_stream(
    batches: Vec<BlobReadBatch>,
    execution: Arc<ReadBlobsExecution>,
    io_parallelism: usize,
) -> BoxStream<'static, Result<IndexedReadBlob>> {
    let streams = batches.into_iter().map(move |batch| {
        execute_blob_read_plans_stream(batch.plans, execution.clone(), io_parallelism)
    });
    stream::iter(streams).flatten().boxed()
}

fn execute_blob_read_plans_stream(
    plans: Vec<BlobReadPlan>,
    execution: Arc<ReadBlobsExecution>,
    io_parallelism: usize,
) -> BoxStream<'static, Result<IndexedReadBlob>> {
    if plans.is_empty() {
        return stream::empty().boxed();
    }

    let plan_stream = stream::iter(plans.into_iter().map(move |plan| {
        let execution = execution.clone();
        execute_blob_read_plan(plan, execution)
    }))
    .buffer_unordered(io_parallelism.max(1));

    plan_stream
        .map_ok(|blobs| stream::iter(blobs.into_iter().map(Ok)))
        .try_flatten()
        .boxed()
}

#[cfg(test)]
pub(super) async fn execute_blob_entries(
    entries: Vec<BlobEntry>,
    io_parallelism: usize,
    io_buffer_size_bytes: Option<u64>,
) -> Result<Vec<IndexedReadBlob>> {
    execute_blob_entries_with_execution(
        entries,
        io_parallelism,
        Arc::new(ReadBlobsExecution::new(io_buffer_size_bytes)),
    )
    .await
}

pub(super) async fn execute_blob_entries_with_execution(
    entries: Vec<BlobEntry>,
    io_parallelism: usize,
    execution: Arc<ReadBlobsExecution>,
) -> Result<Vec<IndexedReadBlob>> {
    let plans = plan_blob_read_plans(entries)?;
    if plans.is_empty() {
        return Ok(Vec::new());
    }

    let batches = stream::iter(plans.into_iter().map(move |plan| {
        let execution = execution.clone();
        execute_blob_read_plan(plan, execution)
    }))
    .buffer_unordered(io_parallelism.max(1))
    .try_collect::<Vec<_>>()
    .await?;
    Ok(batches.into_iter().flatten().collect())
}

pub(in crate::dataset) async fn take_blobs(
    dataset: &Arc<Dataset>,
    row_ids: &[u64],
    column: &str,
) -> Result<Vec<Option<BlobFile>>> {
    let blob_field_id = validate_blob_column(dataset, column)?;
    let collected = collect_blob_selection_for_selection(
        dataset,
        blob_field_id,
        column,
        &ReadBlobsSelection::RowIds(row_ids.to_vec()),
    )
    .await?;
    Ok(collected
        .into_iter()
        .map(|selection| selection.file)
        .collect())
}

/// Take [BlobFile] by row addresses.
///
/// Row addresses are `u64` values encoding `(fragment_id << 32) | row_offset`.
/// Use this method when you already have row addresses, for example from
/// a scan with `with_row_address()`. For row IDs (stable identifiers), use
/// [`Dataset::take_blobs`]. For row indices (offsets), use
/// [`Dataset::take_blobs_by_indices`].
pub async fn take_blobs_by_addresses(
    dataset: &Arc<Dataset>,
    row_addrs: &[u64],
    column: &str,
) -> Result<Vec<Option<BlobFile>>> {
    let blob_field_id = validate_blob_column(dataset, column)?;
    let collected = collect_blob_selection_for_selection(
        dataset,
        blob_field_id,
        column,
        &ReadBlobsSelection::RowAddresses(row_addrs.to_vec()),
    )
    .await?;
    Ok(collected
        .into_iter()
        .map(|selection| selection.file)
        .collect())
}

/// Validate that `column` exists and is a blob column, returning its field id.
pub(in crate::dataset) fn validate_blob_column(
    dataset: &Arc<Dataset>,
    column: &str,
) -> Result<u32> {
    let schema = dataset.schema();
    let blob_field = schema
        .field(column)
        .ok_or_else(|| Error::field_not_found(column, schema.field_paths()))?;
    if !blob_field.is_blob() {
        return Err(Error::invalid_input_source(
            format!("the column '{}' is not a blob column", column).into(),
        ));
    }
    Ok(blob_field.id as u32)
}

/// Load blob descriptor rows for a stable-row-id selection.
async fn take_blob_descriptions_by_row_ids(
    dataset: &Arc<Dataset>,
    row_ids: &[u64],
    column: &str,
) -> Result<RecordBatch> {
    let projection = dataset.schema().project(&[column])?;
    dataset
        .take_builder(row_ids, projection)?
        .with_missing_row_policy(MissingRowPolicy::Error)
        .with_row_address(true)
        .execute()
        .await
}

/// Load blob descriptor rows for a physical-row-address selection.
async fn take_blob_descriptions_by_row_addresses(
    dataset: &Arc<Dataset>,
    row_addrs: &[u64],
    column: &str,
) -> Result<RecordBatch> {
    let projection = dataset.schema().project(&[column])?;
    let projection_request = ProjectionRequest::from(projection);
    let projection_plan = Arc::new(projection_request.into_projection_plan(dataset.clone())?);
    TakeBuilder::try_new_from_addresses(dataset.clone(), row_addrs.to_vec(), projection_plan)?
        .with_row_address(true)
        .execute()
        .await
}

/// Resolve every selected row address and the non-null blob entries among them.
async fn collect_blob_selection_for_selection(
    dataset: &Arc<Dataset>,
    blob_field_id: u32,
    column: &str,
    selection: &ReadBlobsSelection,
) -> Result<Vec<ResolvedBlobSelection>> {
    let description_and_addr = match selection {
        ReadBlobsSelection::None => {
            return Err(Error::invalid_input(
                "Blob row selection is required".to_string(),
            ));
        }
        ReadBlobsSelection::RowIds(row_ids) => {
            take_blob_descriptions_by_row_ids(dataset, row_ids, column).await?
        }
        ReadBlobsSelection::RowIndices(row_indices) => {
            let row_addrs = crate::dataset::take::row_offsets_to_row_addresses(
                &dataset.get_fragments(),
                row_indices,
            )
            .await?;
            take_blob_descriptions_by_row_addresses(dataset, &row_addrs, column).await?
        }
        ReadBlobsSelection::RowAddresses(row_addrs) => {
            take_blob_descriptions_by_row_addresses(dataset, row_addrs, column).await?
        }
    };

    if description_and_addr.num_rows() == 0 {
        return Ok(Vec::new());
    }

    let descriptions = leaf_descriptor_struct(&description_and_addr, column)?;
    let row_addrs = description_and_addr.column(1).as_primitive::<UInt64Type>();

    let files = match blob_version_from_descriptions(descriptions)? {
        BlobVersion::V1 => collect_blob_files_v1(dataset, blob_field_id, descriptions, row_addrs),
        BlobVersion::V2 => {
            collect_blob_files_v2(dataset, blob_field_id, descriptions, row_addrs).await
        }
    }?;
    if files.len() != row_addrs.len() {
        return Err(Error::internal(format!(
            "Resolved blob file count {} did not match row address count {}",
            files.len(),
            row_addrs.len()
        )));
    }
    Ok(row_addrs
        .values()
        .iter()
        .copied()
        .zip(files)
        .map(|(row_address, file)| ResolvedBlobSelection {
            row_address,
            file,
            requested_range: None,
        })
        .collect())
}

/// Walk into the descriptor `RecordBatch` at `column` and return the leaf
/// descriptor `StructArray`, descending through nested struct children for
/// dotted paths.
fn leaf_descriptor_struct<'a>(batch: &'a RecordBatch, column: &str) -> Result<&'a StructArray> {
    let current = leaf_descriptor_array(batch, column)?;
    current
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            Error::invalid_input_source(
                format!(
                    "Blob column '{}' expected descriptor struct but got {}",
                    column,
                    current.data_type()
                )
                .into(),
            )
        })
}

fn leaf_descriptor_array<'a>(batch: &'a RecordBatch, column: &str) -> Result<&'a dyn Array> {
    let path = parse_field_path(column)?;
    let mut current: &dyn Array = batch
        .column_by_name(&path[0])
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "Blob column '{}' was not found in descriptor batch",
                column
            ))
        })?
        .as_ref();
    for segment in &path[1..] {
        let struct_array = current
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| {
                Error::invalid_input_source(
                    format!(
                        "Blob column path '{}' expected struct before segment '{}' but got {}",
                        column,
                        segment,
                        current.data_type()
                    )
                    .into(),
                )
            })?;
        current = struct_array
            .column_by_name(segment)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "Blob column path '{}' missing segment '{}'",
                    column, segment
                ))
            })?
            .as_ref();
    }
    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::dataset::blob::file::{BlobFile, BlobSource};
    use crate::dataset::blob::locate::collect_blob_files_v1;
    use crate::dataset::blob::read::plan::{BlobEntry, plan_blob_read_plans};
    use crate::dataset::blob::test_util::{
        gated_range_store, managed_blob_schema, recording_range_store,
        recording_range_store_with_url,
    };
    use crate::{
        Dataset,
        blob::{BlobArrayBuilder, BlobDescriptorArrayBuilder, ManagedBlobWriter, blob_field},
        dataset::WriteParams,
        dataset::optimize::compact_files,
    };
    use arrow_array::RecordBatch;
    use arrow_array::{ArrayRef, LargeBinaryArray, RecordBatchIterator, StructArray, UInt64Array};
    use arrow_buffer::NullBuffer;
    use arrow_schema::{DataType, Field, Schema};
    use bytes::Bytes;
    use futures::{StreamExt, TryStreamExt};
    use lance_arrow::{
        BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY, BLOB_INLINE_SIZE_THRESHOLD_META_KEY, BLOB_META_KEY,
    };
    use lance_core::datatypes::BlobKind;
    use lance_core::{Error, utils::tempfile::TempStrDir};
    use lance_file::version::LanceFileVersion;
    use lance_io::object_store::ObjectStore;
    use lance_table::format::BasePath;
    use object_store::path::Path;
    use rstest::rstest;
    use std::collections::HashMap;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use std::time::Duration;

    #[rstest]
    #[tokio::test]
    async fn prepared_managed_sharing_reuses_physical_ranges(
        #[values(None, Some(7), Some(u32::MAX))] base_id: Option<u32>,
        #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
    ) {
        let dir = TempStrDir::default();
        let root = Path::from_absolute_path(dir.to_string()).unwrap();
        let mut writer = ManagedBlobWriter::try_new(ObjectStore::local(), root, base_id)
            .await
            .unwrap();
        let payload_path = writer.path().clone();
        writer
            .write_blobs([
                Some(b"shared".as_slice()),
                Some(b"".as_slice()),
                None,
                Some(b"unused".as_slice()),
            ])
            .await
            .unwrap();
        let values = writer.finish().await.unwrap();
        let mut builder = BlobDescriptorArrayBuilder::new("blob");
        builder
            .extend([0, 0, 1, 2, 0].map(|i| values[i].clone()))
            .unwrap();
        let (field, array) = builder.finish().unwrap().into_parts();
        let schema = Arc::new(managed_blob_schema(vec![field]));
        let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            &dir,
            Some(WriteParams {
                data_storage_version: Some(version),
                max_rows_per_file: 2,
                initial_bases: base_id
                    .map(|id| vec![BasePath::new(id, dir.to_string(), None, false)]),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), 3);

        for compacted in [false, true] {
            if compacted {
                let payload_store = dataset.object_store(base_id).await.unwrap();
                let _ = payload_store.io_stats_incremental();
                compact_files(&mut dataset, Default::default(), None)
                    .await
                    .unwrap();
                assert_eq!(dataset.get_fragments().len(), 1);
                let stats = payload_store.io_stats_incremental();
                assert!(
                    stats.requests.iter().all(|request| {
                        !request.path.as_ref().contains("/_blobs/")
                            || !request.method.starts_with("put")
                    }),
                    "{stats:?}"
                );
            }
            let dataset = Arc::new(dataset.clone());
            let blobs = dataset
                .take_blobs_by_indices(&[0, 1, 2, 3, 4], "blob")
                .await
                .unwrap();
            assert!(blobs[3].is_none());
            assert!(blobs[2].as_ref().unwrap().read().await.unwrap().is_empty());
            for i in [0, 1, 4] {
                assert_eq!(blobs[i].as_ref().unwrap().source.path, payload_path);
                assert_eq!(
                    blobs[i].as_ref().unwrap().read().await.unwrap(),
                    b"shared"[..]
                );
            }
            // Descriptor lookup is warm, but payload reads are submitted through
            // the real dataset planner. Duplicate and overlapping selections
            // must share the same physical range in this single read plan.
            let payload_store = dataset.object_store(base_id).await.unwrap();
            let _ = payload_store.io_stats_incremental();
            let results = dataset
                .read_blob_ranges("blob")
                .unwrap()
                .with_row_indices(vec![
                    BlobRangeRequest::new(4, 1, 3),
                    BlobRangeRequest::new(0, 1, 3),
                    BlobRangeRequest::new(1, 2, 2),
                ])
                .execute()
                .await
                .unwrap();
            let stats = payload_store.io_stats_incremental();
            assert_eq!(results[0].data.as_deref(), Some(b"har".as_slice()));
            assert_eq!(results[1].data.as_deref(), Some(b"har".as_slice()));
            assert_eq!(results[2].data.as_deref(), Some(b"ar".as_slice()));
            let requests = stats
                .requests
                .iter()
                .filter(|request| request.path == payload_path)
                .collect::<Vec<_>>();
            assert_eq!(requests.len(), 1, "{requests:?}");
            assert_eq!(requests[0].range, Some(1..4));
            let objects = std::fs::read_dir(std::path::Path::new(dir.as_str()).join("_blobs"))
                .unwrap()
                .collect::<Vec<_>>();
            assert_eq!(objects.len(), 1);
            assert_eq!(
                std::fs::read(objects[0].as_ref().unwrap().path()).unwrap(),
                b"sharedunused"
            );
        }
    }

    #[tokio::test]
    async fn test_execute_blob_entries_preserves_order_and_coalesces() {
        let (store, inner) = recording_range_store(Bytes::from_static(b"abcdefghij"));
        let source = Arc::new(BlobSource::new(store, Path::from("blobs/test.bin")));
        let entries = vec![
            BlobEntry {
                selection_index: 0,
                row_address: 10,
                file: BlobFile::with_source(source.clone(), 4, 3, BlobKind::Packed, None),
                requested_range: None,
            },
            BlobEntry {
                selection_index: 1,
                row_address: 11,
                file: BlobFile::with_source(source, 1, 3, BlobKind::Packed, None),
                requested_range: None,
            },
        ];
        let mut blobs = execute_blob_entries(entries, 2, None).await.unwrap();
        blobs.sort_by_key(|blob| blob.selection_index);

        assert_eq!(blobs.len(), 2);
        assert_eq!(blobs[0].row_address, 10);
        assert_eq!(blobs[0].data.as_ref(), b"efg");
        assert_eq!(blobs[1].row_address, 11);
        assert_eq!(blobs[1].data.as_ref(), b"bcd");
        assert_eq!(inner.requested_blob_ranges(), vec![1..7]);
    }

    #[tokio::test]
    async fn test_blob_selection_apis_preserve_v1_nulls_and_empty_values() {
        let test_dir = TempStrDir::default();
        let blob_metadata = HashMap::from([(BLOB_META_KEY.to_string(), "true".to_string())]);
        let schema = Arc::new(Schema::new(vec![
            Field::new("blob", DataType::LargeBinary, true).with_metadata(blob_metadata),
        ]));
        let blobs =
            LargeBinaryArray::from(vec![None, Some(b"".as_slice()), Some(b"abc".as_slice())]);
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(blobs)]).unwrap();
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch)], schema),
                &test_dir,
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_0),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        let blob_files = dataset
            .take_blobs_by_indices(&[2, 1, 0], "blob")
            .await
            .unwrap();
        assert_eq!(blob_files.len(), 3);
        assert_eq!(
            blob_files[0]
                .as_ref()
                .unwrap()
                .read()
                .await
                .unwrap()
                .as_ref(),
            b"abc"
        );
        assert!(
            blob_files[1]
                .as_ref()
                .unwrap()
                .read()
                .await
                .unwrap()
                .is_empty()
        );
        assert!(blob_files[2].is_none());

        let blobs = dataset
            .read_blobs("blob")
            .unwrap()
            .with_row_indices([2, 1, 0])
            .execute()
            .await
            .unwrap();
        assert_eq!(
            blobs
                .iter()
                .map(|blob| (blob.row_address, blob.data.as_deref()))
                .collect::<Vec<_>>(),
            vec![
                (2, Some(b"abc".as_slice())),
                (1, Some(b"".as_slice())),
                (0, None),
            ]
        );

        let results = dataset
            .read_blob_ranges("blob")
            .unwrap()
            .with_row_indices([
                BlobRangeRequest::new(0, 0, 0),
                BlobRangeRequest::new(0, 0, 1),
                BlobRangeRequest::new(1, 0, 0),
                BlobRangeRequest::new(2, 1, 1),
            ])
            .execute()
            .await
            .unwrap();

        assert_eq!(
            results
                .iter()
                .map(|result| (result.request_index, result.data.as_deref()))
                .collect::<Vec<_>>(),
            vec![
                (0, None),
                (1, None),
                (2, Some(b"".as_slice())),
                (3, Some(b"b".as_slice())),
            ]
        );

        let descriptions = StructArray::try_new(
            vec![
                Arc::new(Field::new("position", DataType::UInt64, false)),
                Arc::new(Field::new("size", DataType::UInt64, false)),
            ]
            .into(),
            vec![
                Arc::new(UInt64Array::from(vec![0])) as ArrayRef,
                Arc::new(UInt64Array::from(vec![0])) as ArrayRef,
            ],
            Some(NullBuffer::from(vec![false])),
        )
        .unwrap();
        let row_addrs = UInt64Array::from(vec![u64::MAX]);
        let files = collect_blob_files_v1(&dataset, u32::MAX, &descriptions, &row_addrs).unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].is_none());
    }

    #[tokio::test]
    async fn test_read_blob_ranges_handles_overlaps_across_scheduler_splits() {
        let max_iop_size = *lance_io::object_store::DEFAULT_MAX_IOP_SIZE;
        let value_size = max_iop_size + 12;
        let nested_start = value_size / 2 - 1;
        let nested_end = value_size;
        let (store, inner) = recording_range_store(Bytes::from(vec![7; value_size as usize]));
        let source = Arc::new(BlobSource::new(store, Path::from("blobs/overlapping.bin")));
        let entries = vec![
            BlobEntry {
                selection_index: 0,
                row_address: 10,
                file: BlobFile::with_source(
                    source.clone(),
                    0,
                    value_size,
                    BlobKind::Dedicated,
                    None,
                ),
                requested_range: Some(BlobReadRange::new(0, value_size)),
            },
            BlobEntry {
                selection_index: 1,
                row_address: 10,
                file: BlobFile::with_source(source, 0, value_size, BlobKind::Dedicated, None),
                requested_range: Some(BlobReadRange::new(nested_start, nested_end - nested_start)),
            },
        ];

        let mut plans = plan_blob_read_plans(entries).unwrap();
        let results = execute_blob_read_plan(
            plans.pop().unwrap(),
            Arc::new(ReadBlobsExecution::new(None)),
        )
        .await
        .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].selection_index, 0);
        assert_eq!(results[0].data.len(), value_size as usize);
        assert!(results[0].data.iter().all(|byte| *byte == 7));
        assert_eq!(results[1].selection_index, 1);
        assert_eq!(results[1].data.len(), (nested_end - nested_start) as usize);
        assert!(results[1].data.iter().all(|byte| *byte == 7));

        let mut physical_ranges = inner.requested_blob_ranges();
        physical_ranges.sort_by_key(|range| range.start);
        assert_eq!(
            physical_ranges.len(),
            value_size.div_ceil(max_iop_size) as usize
        );
        assert_eq!(physical_ranges[0].start, 0);
        assert_eq!(physical_ranges.last().unwrap().end, value_size);
        assert!(
            physical_ranges
                .windows(2)
                .all(|ranges| ranges[0].end == ranges[1].start)
        );
    }

    #[derive(Clone, Copy, Debug)]
    enum BlobSelectionStreamCase {
        Normal,
        Empty,
        Missing,
        Duplicate,
        DuplicateReplacesMissing,
        OutOfRange,
        UpstreamError,
    }

    #[rstest]
    #[case::normal(BlobSelectionStreamCase::Normal)]
    #[case::empty(BlobSelectionStreamCase::Empty)]
    #[case::missing(BlobSelectionStreamCase::Missing)]
    #[case::duplicate(BlobSelectionStreamCase::Duplicate)]
    #[case::duplicate_replaces_missing(BlobSelectionStreamCase::DuplicateReplacesMissing)]
    #[case::out_of_range(BlobSelectionStreamCase::OutOfRange)]
    #[case::upstream_error(BlobSelectionStreamCase::UpstreamError)]
    #[tokio::test]
    async fn test_blob_selection_stream(
        #[case] case: BlobSelectionStreamCase,
        #[values(true, false)] preserve_order: bool,
    ) {
        let (non_null, null_values, selection_count, mut expected_values, expected_error) =
            match case {
                BlobSelectionStreamCase::Normal => (
                    vec![Ok((2, "row2")), Ok((0, "row0"))],
                    vec![(1, "null1")],
                    3,
                    vec!["row0", "null1", "row2"],
                    None,
                ),
                BlobSelectionStreamCase::Empty => (Vec::new(), Vec::new(), 0, Vec::new(), None),
                BlobSelectionStreamCase::Missing => (
                    vec![Ok((0, "row0"))],
                    vec![(1, "null1")],
                    3,
                    vec!["row0", "null1"],
                    Some(if preserve_order {
                        "completed before selection index 2 was produced"
                    } else {
                        "completed after producing 2 of 3 selections"
                    }),
                ),
                BlobSelectionStreamCase::Duplicate => (
                    vec![Ok((0, "row0")), Ok((0, "duplicate"))],
                    Vec::new(),
                    1,
                    vec!["row0"],
                    Some("selection index 0 was produced more than once"),
                ),
                BlobSelectionStreamCase::DuplicateReplacesMissing => (
                    vec![Ok((0, "row0")), Ok((0, "duplicate"))],
                    Vec::new(),
                    2,
                    vec!["row0"],
                    Some("selection index 0 was produced more than once"),
                ),
                BlobSelectionStreamCase::OutOfRange => (
                    vec![Ok((1, "out-of-range"))],
                    Vec::new(),
                    1,
                    Vec::new(),
                    Some("selection index 1 exceeded selected row count 1"),
                ),
                BlobSelectionStreamCase::UpstreamError => (
                    vec![
                        Ok((0, "row0")),
                        Err(Error::internal("planned blob read failed".to_string())),
                    ],
                    Vec::new(),
                    1,
                    vec!["row0"],
                    Some("planned blob read failed"),
                ),
            };
        let non_null = futures::stream::iter(non_null).boxed();
        let results =
            totalize_blob_selection_stream(non_null, null_values, selection_count, preserve_order)
                .take(8)
                .collect::<Vec<_>>()
                .await;

        let mut actual_values = results
            .iter()
            .filter_map(|result| result.as_ref().ok().copied())
            .collect::<Vec<_>>();
        if !preserve_order {
            actual_values.sort_unstable();
            expected_values.sort_unstable();
        }
        assert_eq!(actual_values, expected_values);

        let errors = results
            .iter()
            .filter_map(|result| result.as_ref().err())
            .collect::<Vec<_>>();
        assert_eq!(
            errors.len(),
            if expected_error.is_some() { 1 } else { 0 },
            "stream must terminate after its first error: {errors:?}"
        );
        if let Some(expected_error) = expected_error {
            assert!(errors[0].to_string().contains(expected_error));
        }
    }

    #[tokio::test]
    async fn test_ordered_blob_selection_stream_drains_ready_prefix_before_reads() {
        let read_poll_count = Arc::new(AtomicUsize::new(0));
        let read_poll_count_for_stream = read_poll_count.clone();
        let read_values = futures::stream::once(async move {
            read_poll_count_for_stream.fetch_add(1, Ordering::SeqCst);
            Ok::<_, Error>((2, "row2"))
        })
        .boxed();
        let mut values =
            totalize_blob_selection_stream(read_values, vec![(0, "null0"), (1, "null1")], 3, true);

        assert_eq!(values.try_next().await.unwrap(), Some("null0"));
        assert_eq!(read_poll_count.load(Ordering::SeqCst), 0);
        assert_eq!(values.try_next().await.unwrap(), Some("null1"));
        assert_eq!(read_poll_count.load(Ordering::SeqCst), 0);
        assert_eq!(values.try_next().await.unwrap(), Some("row2"));
        assert_eq!(read_poll_count.load(Ordering::SeqCst), 1);
        assert_eq!(values.try_next().await.unwrap(), None);
    }

    #[tokio::test]
    async fn test_planned_blob_range_avoids_whole_value_read_amplification() {
        const VALUE_SIZE: u64 = 500 * 1024 * 1024;
        const WINDOW_SIZE: u64 = 100 * 1024;

        let (store, inner) = recording_range_store(Bytes::from(vec![7; WINDOW_SIZE as usize]));
        let source = Arc::new(BlobSource::new(store, Path::from("blobs/large.bin")));
        let entries = vec![BlobEntry {
            selection_index: 0,
            row_address: 10,
            file: BlobFile::with_source(source, 0, VALUE_SIZE, BlobKind::Dedicated, None),
            requested_range: Some(BlobReadRange::new(0, WINDOW_SIZE)),
        }];
        let mut plans = plan_blob_read_plans(entries).unwrap();
        let results = execute_blob_read_plan(
            plans.pop().unwrap(),
            Arc::new(ReadBlobsExecution::new(None)),
        )
        .await
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].data.len(), WINDOW_SIZE as usize);
        let physical_ranges = inner.requested_ranges();
        assert_eq!(inner.head_requests.load(Ordering::Relaxed), 0);
        assert_eq!(physical_ranges, vec![0..WINDOW_SIZE]);
        assert_eq!(
            physical_ranges
                .iter()
                .map(|range| range.end - range.start)
                .sum::<u64>(),
            WINDOW_SIZE
        );
        assert!(!physical_ranges.contains(&(0..VALUE_SIZE)));
    }

    #[rstest]
    #[case::inline(BlobKind::Inline, 1024 * 1024, 1024 * 1024)]
    #[case::packed(BlobKind::Packed, 0, 1024 * 1024)]
    #[case::dedicated(BlobKind::Dedicated, 0, 1)]
    #[tokio::test]
    async fn test_read_blob_ranges_avoids_whole_value_read_amplification_end_to_end(
        #[case] kind: BlobKind,
        #[case] inline_threshold: usize,
        #[case] dedicated_threshold: usize,
    ) {
        const VALUE_SIZE: usize = 256 * 1024;
        const WINDOW_SIZE: u64 = 4 * 1024;
        const OFFSET: u64 = 64 * 1024;

        let test_dir = TempStrDir::default();
        let mut blob_builder = BlobArrayBuilder::new(4);
        for value in 0xA5..0xA9 {
            blob_builder.push_bytes(vec![value; VALUE_SIZE]).unwrap();
        }
        let mut blob_field = blob_field("blob", false);
        let mut metadata = blob_field.metadata().clone();
        metadata.insert(
            BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY.to_string(),
            dedicated_threshold.to_string(),
        );
        metadata.insert(
            BLOB_INLINE_SIZE_THRESHOLD_META_KEY.to_string(),
            inline_threshold.to_string(),
        );
        blob_field = blob_field.with_metadata(metadata);
        let schema = Arc::new(Schema::new(vec![blob_field]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(blob_builder.finish().unwrap())],
        )
        .unwrap();
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch)], schema),
                &test_dir,
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    max_rows_per_file: 2,
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        assert_eq!(dataset.get_fragments().len(), 2);
        let indices = [3, 0, 2, 1];
        let blobs = dataset
            .take_blobs_by_indices(&indices, "blob")
            .await
            .unwrap();
        for blob in &blobs {
            assert_eq!(blob.as_ref().unwrap().kind(), kind);
        }
        let _ = dataset.object_store.io_stats_incremental();
        let results = dataset
            .read_blob_ranges("blob")
            .unwrap()
            .with_row_indices(indices.map(|row| BlobRangeRequest::new(row, OFFSET, WINDOW_SIZE)))
            .execute()
            .await
            .unwrap();
        let stats = dataset.object_store.io_stats_incremental();

        assert_eq!(results.len(), indices.len());
        for ((result, row), blob) in results.iter().zip(indices).zip(blobs) {
            let data = result.data.as_ref().unwrap();
            assert_eq!(data.as_ref(), vec![0xA5 + row as u8; WINDOW_SIZE as usize]);
            let blob = blob.unwrap();
            let range = blob.position + OFFSET..blob.position + OFFSET + WINDOW_SIZE;
            assert!(
                stats.requests.iter().any(|request| {
                    request.path == blob.source.path && request.range.as_ref() == Some(&range)
                }),
                "{stats:?}"
            );
            assert!(
                stats
                    .requests
                    .iter()
                    .filter(|request| request.path == blob.source.path)
                    .all(|request| {
                        request.method != "head"
                            && (request.method != "get_opts" || request.range.is_some())
                    }),
                "{stats:?}"
            );
        }
    }

    #[tokio::test]
    async fn test_read_blobs_stream_emits_ready_plan_without_waiting_for_slower_ones() {
        let (slow_store, _, slow_gate) = gated_range_store(
            Bytes::from_static(b"abcdef"),
            "mock://slow/blob-range-tests",
        );
        let (fast_store, _) = recording_range_store_with_url(
            Bytes::from_static(b"uvwxyz"),
            "mock://fast/blob-range-tests",
        );
        let entries = vec![
            BlobEntry {
                selection_index: 0,
                row_address: 10,
                file: BlobFile::with_source(
                    Arc::new(BlobSource::new(slow_store, Path::from("blobs/slow.bin"))),
                    0,
                    3,
                    BlobKind::Packed,
                    None,
                ),
                requested_range: None,
            },
            BlobEntry {
                selection_index: 1,
                row_address: 11,
                file: BlobFile::with_source(
                    Arc::new(BlobSource::new(fast_store, Path::from("blobs/fast.bin"))),
                    0,
                    3,
                    BlobKind::Packed,
                    None,
                ),
                requested_range: None,
            },
        ];
        let execution = Arc::new(ReadBlobsExecution::new(None));
        let mut stream: ReadBlobsStream = futures::stream::iter(
            plan_blob_read_plans(entries)
                .unwrap()
                .into_iter()
                .map(move |plan| execute_blob_read_plan(plan, execution.clone())),
        )
        .buffer_unordered(2)
        .map_ok(|blobs: Vec<IndexedReadBlob>| {
            futures::stream::iter(
                blobs
                    .into_iter()
                    .map(|blob| Ok::<ReadBlob, Error>(into_read_blob(blob))),
            )
        })
        .try_flatten()
        .boxed();

        let first = tokio::time::timeout(Duration::from_secs(1), stream.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(first.row_address, 11);
        assert_eq!(first.data.as_deref(), Some(b"uvw".as_slice()));

        slow_gate.add_permits(1);
        let second = tokio::time::timeout(Duration::from_secs(1), stream.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(second.row_address, 10);
        assert_eq!(second.data.as_deref(), Some(b"abc".as_slice()));
    }
}
