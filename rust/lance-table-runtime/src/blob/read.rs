// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

mod plan;
mod selection;

use std::{
    borrow::Borrow,
    collections::{BTreeMap, HashMap},
    ops::Range,
    sync::Arc,
};

use bytes::Bytes;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt, stream};
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};

pub use self::plan::BlobEntry;
use self::plan::{
    BlobReadBatch, BlobReadPlan, ResolvedBlobSelection, plan_blob_read_batches,
    plan_blob_read_plans, plan_disjoint_blob_reads,
};
use self::selection::{collect_blob_selection, validate_blob_column};
use super::file::{BlobFile, BlobSource};
use crate::Table;
use crate::take::TakeSelection;
use lance_core::{Error, Result};

impl Table {
    /// Prepare a streaming read of a blob column's payloads.
    ///
    /// ```
    /// # use lance_table_runtime::Table;
    /// # async fn example(table: &Table) -> lance_core::Result<()> {
    /// let blobs = table.read_blobs("images")?.with_row_indices(vec![0, 2]).execute().await?;
    /// # let _ = blobs;
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_blobs(&self, column: impl AsRef<str>) -> Result<ReadBlobsBuilder> {
        let column = column.as_ref();
        let field_id = validate_blob_column(self, column)?;
        Ok(ReadBlobsBuilder::new(
            Arc::new(self.clone()),
            column.to_string(),
            field_id,
        ))
    }

    /// Prepare a streaming read of byte ranges relative to selected blobs.
    ///
    /// ```
    /// # use lance_table_runtime::{Table, blob::BlobRangeRequest};
    /// # async fn example(table: &Table) -> lance_core::Result<()> {
    /// let chunks = table.read_blob_ranges("images")?
    ///     .with_row_indices([BlobRangeRequest::new(0, 0, 16)])
    ///     .execute().await?;
    /// # let _ = chunks;
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_blob_ranges(&self, column: impl AsRef<str>) -> Result<ReadBlobRangesBuilder> {
        Ok(ReadBlobRangesBuilder::new(self.read_blobs(column)?))
    }
}

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

/// Bytes materialized for one request submitted through [`ReadBlobFilesBuilder`].
///
/// A request names a [`BlobFile`] handle rather than a row, so the result is
/// identified by its position in the request list instead of a row address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadBlobFile {
    /// Zero-based position of this request in the caller's request list.
    pub request_index: usize,
    /// Requested bytes, or `None` when the request passed `None` instead of a
    /// handle, which is how [`Table::take_blobs`] reports a null blob.
    ///
    /// A valid empty blob, or an empty range on any handle, is
    /// `Some(Bytes::new())`.
    pub data: Option<Bytes>,
}

/// Stream returned by [`ReadBlobsBuilder::try_into_stream`].
pub type ReadBlobsStream = BoxStream<'static, Result<ReadBlob>>;

/// Stream returned by [`ReadBlobRangesBuilder::try_into_stream`].
pub type ReadBlobRangesStream = BoxStream<'static, Result<ReadBlobRange>>;

/// Stream returned by [`ReadBlobFilesBuilder::try_into_stream`].
pub type ReadBlobFilesStream = BoxStream<'static, Result<ReadBlobFile>>;

/// Row selector configured on [`ReadBlobsBuilder`].
#[derive(Debug, Clone)]
enum ReadBlobsSelection {
    None,
    RowIds(Vec<u64>),
    RowIndices(Vec<u64>),
    RowAddresses(Vec<u64>),
}

impl ReadBlobsSelection {
    fn as_take_selection(&self) -> Result<TakeSelection<'_>> {
        match self {
            Self::None => Err(Error::internal("Validated blob row selection was missing")),
            Self::RowIds(rows) => Ok(TakeSelection::RowIds(rows)),
            Self::RowIndices(rows) => Ok(TakeSelection::Indices(rows)),
            Self::RowAddresses(rows) => Ok(TakeSelection::RowAddresses(rows)),
        }
    }
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
    io_parallelism: Option<usize>,
    preserve_order: bool,
}

impl Default for ReadBlobsOptions {
    fn default() -> Self {
        Self {
            io_buffer_size_bytes: None,
            io_parallelism: None,
            preserve_order: true,
        }
    }
}

/// Builder for sequential / planned blob reads.
///
/// Unlike [`Table::take_blobs`], which returns [`BlobFile`] handles for
/// caller-driven random access, this builder plans object-store reads across a
/// selected row set and yields fully materialized blob payloads.
#[derive(Debug, Clone)]
pub struct ReadBlobsBuilder {
    table: Arc<Table>,
    column: String,
    blob_field_id: u32,
    selection: ReadBlobsSelection,
    options: ReadBlobsOptions,
}

impl ReadBlobsBuilder {
    pub(crate) fn new(table: Arc<Table>, column: String, blob_field_id: u32) -> Self {
        Self {
            table,
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
        let collected = collect_blob_selection(
            &self.table,
            self.blob_field_id,
            &self.column,
            self.selection.as_take_selection()?,
        )
        .await?;
        let row_addresses: Vec<_> = collected
            .iter()
            .map(|selection| selection.row_address)
            .collect();
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
        read_partitioned_blob_selection(
            Some(self.table.state.object_store.as_ref()),
            &self.options,
            partitioned,
            move |blob| {
                let row_address = row_addresses[blob.selection_index];
                Ok(into_read_blob(blob, row_address))
            },
        )
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
        let mut selections = collect_blob_selection(
            &self.inner.table,
            self.inner.blob_field_id,
            &self.inner.column,
            row_selection.as_take_selection()?,
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
        let row_addresses: Vec<_> = selections
            .iter()
            .map(|selection| selection.row_address)
            .collect();
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
        read_partitioned_blob_selection(
            Some(self.inner.table.state.object_store.as_ref()),
            &self.inner.options,
            partitioned,
            move |blob| {
                let row_address = row_addresses[blob.selection_index];
                into_read_blob_range(blob, row_address)
            },
        )
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

/// Plan payload reads through existing handles without reading dataset descriptors.
///
/// Null handles yield null results. Defaults come from the first non-null handle
/// and can be overridden on the builder. Empty and all-null inputs issue no I/O.
/// Handles from different stores and snapshots retain their own storage credentials.
///
/// ```
/// # use lance_core::Result;
/// # use lance_table_runtime::Table;
/// # use lance_table_runtime::blob::read_blob_files;
/// # async fn example(table: &Table) -> Result<()> {
/// let handles = table.take_blobs_by_indices(&[0, 1], "images").await?;
/// let blobs = read_blob_files(handles.iter().map(Option::as_ref)).execute().await?;
/// # Ok(())
/// # }
/// ```
pub fn read_blob_files<B: Borrow<BlobFile>>(
    blob_files: impl IntoIterator<Item = Option<B>>,
) -> ReadBlobFilesBuilder {
    ReadBlobFilesBuilder::new(blob_files)
}

/// Plan blob-local `(handle, range)` requests without changing handle cursors.
///
/// Preserves request order by default, including duplicates and null handles.
/// Ranges must fit their non-null handle; offset overflow is rejected even for
/// null handles. Empty ranges issue no payload I/O.
///
/// ```
/// # use lance_core::Result;
/// # use lance_table_runtime::Table;
/// # use lance_table_runtime::blob::{read_blob_file_ranges, BlobReadRange};
/// # async fn example(table: &Table) -> Result<()> {
/// let handles = table.take_blobs_by_indices(&[0], "images").await?;
/// let ranges = read_blob_file_ranges([
///     (handles[0].as_ref(), BlobReadRange::new(0, 16)),
/// ]).execute().await?;
/// # Ok(())
/// # }
/// ```
pub fn read_blob_file_ranges<B: Borrow<BlobFile>>(
    requests: impl IntoIterator<Item = (Option<B>, BlobReadRange)>,
) -> ReadBlobFilesBuilder {
    ReadBlobFilesBuilder::new_ranges(requests)
}

/// One read requested through [`ReadBlobFilesBuilder`]. `range` is `None` for
/// a whole-blob read.
#[derive(Debug)]
struct BlobFileRead {
    blob_file: Option<BlobFile>,
    range: Option<BlobReadRange>,
}

/// Builder for planned reads through [`BlobFile`] handles the caller already
/// holds, created by [`read_blob_files`] or [`read_blob_file_ranges`].
///
/// [`ReadBlobsBuilder`] and [`ReadBlobRangesBuilder`] resolve their row
/// selection by reading the blob column's descriptors on every execution, a
/// request that the payload reads then wait for. A handle returned by
/// [`Table::take_blobs`] already holds its descriptor, so this builder issues
/// only payload reads. They are planned the same way: grouped by backing
/// object, coalesced when nearby, and batched by the I/O buffer size.
///
/// Handles retain their resolved object locations and byte ranges. Later
/// dataset commits do not re-resolve their descriptors; referenced payloads
/// must remain retained. External objects follow their provider's consistency
/// semantics.
/// Handles from different datasets or versions can be mixed in one read. The
/// first non-null handle supplies I/O defaults (parallelism and buffer size).
#[derive(Debug, Clone)]
pub struct ReadBlobFilesBuilder {
    reads: Arc<[BlobFileRead]>,
    options: ReadBlobsOptions,
}

impl ReadBlobFilesBuilder {
    pub(crate) fn new<B: Borrow<BlobFile>>(
        blob_files: impl IntoIterator<Item = Option<B>>,
    ) -> Self {
        let reads = blob_files
            .into_iter()
            .map(|blob_file| BlobFileRead {
                blob_file: blob_file.map(|blob_file| blob_file.borrow().share()),
                range: None,
            })
            .collect();
        Self {
            reads,
            options: ReadBlobsOptions::default(),
        }
    }

    pub(crate) fn new_ranges<B: Borrow<BlobFile>>(
        requests: impl IntoIterator<Item = (Option<B>, BlobReadRange)>,
    ) -> Self {
        let reads = requests
            .into_iter()
            .map(|(blob_file, range)| BlobFileRead {
                blob_file: blob_file.map(|blob_file| blob_file.borrow().share()),
                range: Some(range),
            })
            .collect();
        Self {
            reads,
            options: ReadBlobsOptions::default(),
        }
    }

    /// Set the positive byte budget for planned physical reads.
    ///
    /// Defaults to the first non-null handle's object store. A single request
    /// larger than this budget is materialized alone.
    pub fn with_io_buffer_size_bytes(mut self, bytes: u64) -> Self {
        self.options.io_buffer_size_bytes = Some(bytes);
        self
    }

    /// Set the maximum number of concurrent physical read plans. Must be positive.
    pub fn with_io_parallelism(mut self, io_parallelism: usize) -> Self {
        self.options.io_parallelism = Some(io_parallelism);
        self
    }

    /// Whether results must follow the caller's request order.
    pub fn preserve_order(mut self, preserve: bool) -> Self {
        self.options.preserve_order = preserve;
        self
    }

    /// Execute the planned read and return one result per request.
    ///
    /// A `None` request has `data` set to `None` and issues no I/O. Empty
    /// ranges and empty blobs contain an empty buffer without payload I/O. By
    /// default, results follow request order; `request_index` identifies the
    /// original request even when ordering is disabled.
    ///
    /// Returns an error before any payload I/O when a range's `offset + length`
    /// overflows `u64` (checked for every request), when a range extends past
    /// its handle's blob size (not checked for `None` requests, which have no
    /// size), or when a handle was already closed when the stream was created.
    /// Closing a handle after that point does not cancel reads that were
    /// already planned.
    pub async fn try_into_stream(self) -> Result<ReadBlobFilesStream> {
        if self.options.io_buffer_size_bytes == Some(0) {
            return Err(Error::invalid_input(
                "ReadBlobFilesBuilder io_buffer_size must be greater than 0".to_string(),
            ));
        }
        if self.options.io_parallelism == Some(0) {
            return Err(Error::invalid_input(
                "ReadBlobFilesBuilder io_parallelism must be greater than 0".to_string(),
            ));
        }
        let mut reads = Vec::with_capacity(self.reads.len());
        let mut ready_values = Vec::new();
        for (request_index, read) in self.reads.iter().enumerate() {
            if let Some(range) = read.range {
                range.checked_range(request_index)?;
            }
            let Some(blob_file) = &read.blob_file else {
                ready_values.push((
                    request_index,
                    ReadBlobFile {
                        request_index,
                        data: None,
                    },
                ));
                continue;
            };
            if blob_file.is_closed().await {
                return Err(Error::invalid_input(format!(
                    "Blob file in request {request_index} is already closed"
                )));
            }
            reads.push(BlobEntry {
                selection_index: request_index,
                file: blob_file.share(),
                requested_range: read.range,
            });
        }
        let partitioned = PartitionedBlobSelection {
            selection_count: self.reads.len(),
            reads,
            ready_values,
        };
        let store = self
            .reads
            .iter()
            .find_map(|read| read.blob_file.as_ref())
            .map(|file| file.source.object_store.as_ref());
        let mut options = self.options;
        options.io_buffer_size_bytes = options.io_buffer_size_bytes.or_else(|| {
            store.map(|store| SchedulerConfig::max_bandwidth(store).io_buffer_size_bytes)
        });
        read_partitioned_blob_selection(store, &options, partitioned, |blob| {
            Ok(ReadBlobFile {
                request_index: blob.selection_index,
                data: Some(blob.data),
            })
        })
    }

    /// Execute the planned read and collect all returned bytes in memory.
    pub async fn execute(self) -> Result<Vec<ReadBlobFile>> {
        self.try_into_stream().await?.try_collect().await
    }
}

/// Operation-scoped scheduler cache for one [`ReadBlobsBuilder`] execution.
///
/// We reuse one [`ScanScheduler`] per store instance, keeping different
/// credentials and endpoints isolated without retaining a scheduler for every
/// object visited by a streaming scan. Payload grouping remains per source.
#[derive(Debug)]
pub struct ReadBlobsExecution {
    io_buffer_size_bytes: Option<u64>,
    schedulers: std::sync::Mutex<HashMap<usize, Arc<ScanScheduler>>>,
}

impl ReadBlobsExecution {
    pub fn new(io_buffer_size_bytes: Option<u64>) -> Self {
        Self {
            io_buffer_size_bytes,
            schedulers: std::sync::Mutex::new(HashMap::new()),
        }
    }

    #[cfg(test)]
    fn num_schedulers(&self) -> usize {
        self.schedulers.lock().unwrap().len()
    }

    fn scheduler_for(&self, source: &BlobSource) -> Arc<ScanScheduler> {
        let mut schedulers = self.schedulers.lock().unwrap();
        // Each cached scheduler retains its store Arc, so its address cannot be
        // reused for another store while this execution holds the entry.
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
pub struct IndexedReadBlob {
    pub selection_index: usize,
    requested_range: Option<BlobReadRange>,
    pub data: Bytes,
}

fn into_read_blob(blob: IndexedReadBlob, row_address: u64) -> ReadBlob {
    debug_assert!(
        blob.requested_range.is_none(),
        "whole-blob reads must not carry a requested range"
    );
    ReadBlob {
        row_address,
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

/// Plan and execute the payload reads of a partitioned selection, returning one
/// value per selection. `into_value` maps each completed payload read;
/// `partitioned.ready_values` already holds the values that need no payload I/O.
fn read_partitioned_blob_selection<T: Send + 'static>(
    store: Option<&ObjectStore>,
    options: &ReadBlobsOptions,
    partitioned: PartitionedBlobSelection<T>,
    into_value: impl Fn(IndexedReadBlob) -> Result<T> + Send + 'static,
) -> Result<BoxStream<'static, Result<T>>> {
    let physical_buffer_size = options.io_buffer_size_bytes.unwrap_or_else(|| {
        store
            .map(|store| SchedulerConfig::max_bandwidth(store).io_buffer_size_bytes)
            .unwrap_or(1)
    });
    let batches = plan_blob_read_batches(partitioned.reads, physical_buffer_size)?;
    let execution = Arc::new(ReadBlobsExecution::new(options.io_buffer_size_bytes));
    let io_parallelism = options
        .io_parallelism
        .unwrap_or_else(|| store.map(ObjectStore::io_parallelism).unwrap_or(1));
    let read_values = execute_blob_read_batches_stream(batches, execution, io_parallelism)
        .map(move |result| {
            result.and_then(|blob| {
                let selection_index = blob.selection_index;
                into_value(blob).map(|value| (selection_index, value))
            })
        })
        .boxed();
    Ok(totalize_blob_selection_stream(
        read_values,
        partitioned.ready_values,
        partitioned.selection_count,
        options.preserve_order,
    ))
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

fn into_read_blob_range(blob: IndexedReadBlob, row_address: u64) -> Result<ReadBlobRange> {
    let range = blob.requested_range.ok_or_else(|| {
        Error::internal(format!(
            "Blob range request {} completed without range metadata",
            blob.selection_index
        ))
    })?;
    Ok(ReadBlobRange {
        request_index: blob.selection_index,
        row_address,
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
        let file_scheduler = scheduler.open_reader(reader.into());
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

pub async fn execute_blob_entries_with_execution(
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

#[cfg(test)]
mod tests;
