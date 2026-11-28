// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Point reads over a table snapshot using offsets, row IDs, or physical addresses.

mod offsets;
mod projection;

use std::borrow::Cow;
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{Array, RecordBatch, StructArray, UInt64Array};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer};
use arrow_schema::{DataType, Field};
use arrow_select::concat::concat_batches;
use futures::{FutureExt, StreamExt, TryStreamExt, future::BoxFuture};
use lance_arrow::RecordBatchExt;
use lance_core::datatypes::{Projection, Schema};
use lance_core::utils::address::RowAddress;
use lance_core::utils::io_stats::IoStatsRecorder;
use lance_core::{Error, ROW_ADDR, ROW_OFFSET, Result};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_table::rowids::compute_row_offset_array;

use crate::rowids::resolve_take_row_ids;
use crate::{FileFragment, FragReadConfig, Table};

#[doc(hidden)]
pub use offsets::row_offsets_to_row_addresses;
pub use projection::TakeProjection;

/// Select rows within the checked-out snapshot.
#[derive(Clone, Copy, Debug)]
pub enum TakeSelection<'a> {
    /// Zero-based positions among live rows in manifest order.
    Indices(&'a [u64]),
    /// Stable row IDs, or physical addresses if stable IDs are disabled.
    RowIds(&'a [u64]),
    /// Physical addresses formed from a fragment ID and physical offset.
    RowAddresses(&'a [u64]),
}

impl TakeSelection<'_> {
    fn is_empty(self) -> bool {
        match self {
            Self::Indices(rows) | Self::RowIds(rows) | Self::RowAddresses(rows) => rows.is_empty(),
        }
    }
}

/// How to handle missing stable row IDs in a point take.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MissingRowPolicy {
    /// Drop unresolved IDs from the result.
    #[default]
    Ignore,
    /// Fail if an ID is missing or deleted.
    Error,
}

/// Options for [`Table::take_with_options`].
#[derive(Clone, Copy, Debug, Default)]
pub struct TakeOptions {
    /// Append `_rowaddr` when it is absent from the projection. Every resolved
    /// address must then produce a row.
    pub with_row_address: bool,
    /// Permit deleted rows to be read. Use only when the IDs are known to be
    /// live. For stable IDs, overlapping fragment spans still consult deletions
    /// to distinguish current rows from superseded copies.
    pub ignore_deletions: bool,
    /// How to handle stable IDs that do not resolve to a physical address.
    pub missing_row_policy: MissingRowPolicy,
    /// Cap primary-base data read concurrency for current-format files.
    ///
    /// By default the object store chooses the limit. Additional bases, legacy
    /// data files, and auxiliary metadata reads use their own schedulers.
    pub io_parallelism: Option<NonZeroU32>,
}

impl Table {
    /// Take live rows by zero-based position in the snapshot.
    ///
    /// Results preserve request order and duplicates. The schema controls the
    /// output names and order, including any requested system columns. An
    /// empty projection preserves the row count. Out-of-bounds positions
    /// return an error.
    ///
    /// ```
    /// # use lance_table_runtime::Table;
    /// # async fn example(table: &Table) -> lance_core::Result<()> {
    /// let projection = table.schema().project_preserve_system_columns(&["id", "_rowoffset"])?;
    /// let batch = table.take(&[3, 0, 3], &projection).await?;
    /// assert_eq!(batch.num_rows(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn take(&self, indices: &[u64], projection: &Schema) -> Result<RecordBatch> {
        self.take_with_io_parallelism(indices, projection, None)
            .await
    }

    /// Take rows by stable row ID. Tables without stable IDs interpret the IDs
    /// as physical addresses. Missing stable IDs and deleted rows are omitted.
    ///
    /// ```
    /// # use lance_table_runtime::Table;
    /// # async fn example(table: &Table, row_id: u64) -> lance_core::Result<()> {
    /// let projection = table.schema().project_preserve_system_columns(&["_rowid", "id"])?;
    /// let batch = table.take_rows(&[row_id], &projection).await?;
    /// # let _ = batch;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn take_rows(&self, row_ids: &[u64], projection: &Schema) -> Result<RecordBatch> {
        let projection = TakeProjection::try_new(Arc::new(self.clone()), projection)?;
        self.take_with_options(
            TakeSelection::RowIds(row_ids),
            &projection,
            TakeOptions::default(),
        )
        .await
    }

    /// Take live rows by position with an optional positive I/O concurrency cap.
    ///
    /// See [`TakeOptions::io_parallelism`] for the reads covered by the cap.
    ///
    /// ```
    /// # use std::num::NonZeroU32;
    /// # use lance_table_runtime::Table;
    /// # async fn example(table: &Table) -> lance_core::Result<()> {
    /// let batch = table.take_with_io_parallelism(&[4], table.schema(), NonZeroU32::new(8)).await?;
    /// # let _ = batch;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn take_with_io_parallelism(
        &self,
        indices: &[u64],
        projection: &Schema,
        io_parallelism: Option<NonZeroU32>,
    ) -> Result<RecordBatch> {
        let projection = TakeProjection::try_new(Arc::new(self.clone()), projection)?;
        self.take_with_options(
            TakeSelection::Indices(indices),
            &projection,
            TakeOptions {
                io_parallelism,
                ..Default::default()
            },
        )
        .await
    }

    /// Take rows using a resolved schema and explicit read options.
    ///
    /// The projection must have been resolved against this snapshot's schema.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use lance_table_runtime::{Table, TakeOptions, TakeProjection, TakeSelection};
    /// # async fn example(table: Arc<Table>, address: u64) -> lance_core::Result<()> {
    /// let projection = TakeProjection::try_new(table.clone(), table.schema())?;
    /// let batch = table
    ///     .take_with_options(
    ///         TakeSelection::RowAddresses(&[address]),
    ///         &projection,
    ///         TakeOptions::default(),
    ///     )
    ///     .await?;
    /// # let _ = batch;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn take_with_options(
        &self,
        selection: TakeSelection<'_>,
        projection: &TakeProjection,
        options: TakeOptions,
    ) -> Result<RecordBatch> {
        self.take_with_options_with_io_stats(selection, projection, options, None)
            .await
    }

    /// Take rows while attributing current-format data-file I/O to `io_stats`.
    #[doc(hidden)]
    pub async fn take_with_options_with_io_stats(
        &self,
        selection: TakeSelection<'_>,
        projection: &TakeProjection,
        options: TakeOptions,
        io_stats: Option<Arc<dyn IoStatsRecorder>>,
    ) -> Result<RecordBatch> {
        projection.take(self, selection, options, io_stats).await
    }

    /// Read the physical inputs for a caller's output expressions. An empty
    /// address selection returns `None` so the caller can supply its output schema.
    #[doc(hidden)]
    pub fn take_physical<'a>(
        &'a self,
        selection: TakeSelection<'a>,
        projection: &'a Projection,
        with_row_offset: bool,
        options: TakeOptions,
    ) -> BoxFuture<'a, Result<Option<RecordBatch>>> {
        self.take_physical_with_io_stats(selection, projection, with_row_offset, options, None)
    }

    /// Read physical inputs while attributing current-format data-file I/O to `io_stats`.
    #[doc(hidden)]
    pub fn take_physical_with_io_stats<'a>(
        &'a self,
        selection: TakeSelection<'a>,
        projection: &'a Projection,
        with_row_offset: bool,
        options: TakeOptions,
        io_stats: Option<Arc<dyn IoStatsRecorder>>,
    ) -> BoxFuture<'a, Result<Option<RecordBatch>>> {
        // Keep fragment reader futures out of the layout of callers that
        // compose takes with deeply nested scan and index operations.
        self.take_physical_impl(selection, projection, with_row_offset, options, io_stats)
            .boxed()
    }

    async fn take_physical_impl(
        &self,
        selection: TakeSelection<'_>,
        projection: &Projection,
        with_row_offset: bool,
        options: TakeOptions,
        io_stats: Option<Arc<dyn IoStatsRecorder>>,
    ) -> Result<Option<RecordBatch>> {
        if selection.is_empty() {
            return Ok(None);
        }
        let row_addrs = match selection {
            TakeSelection::Indices(indices) => {
                Cow::Owned(row_offsets_to_row_addresses(&self.get_fragments(), indices).await?)
            }
            TakeSelection::RowIds(row_ids) => self.resolve_row_ids(row_ids, options).await?,
            TakeSelection::RowAddresses(row_addrs) => Cow::Borrowed(row_addrs),
        };
        if row_addrs.is_empty() {
            return Ok(None);
        }

        // Keep an internal address so empty and constant projections retain row counts.
        let mut read_config = FragReadConfig::default()
            .with_row_id(projection.with_row_id)
            .with_row_address(projection.with_row_addr || projection.is_empty())
            .with_row_created_at_version(projection.with_row_created_at_version)
            .with_row_last_updated_at_version(projection.with_row_last_updated_at_version)
            .with_ignore_deletions(options.ignore_deletions);
        let store = &self.state.object_store;
        let mut fragment_concurrency = store.io_parallelism();
        if let Some(io_parallelism) = options.io_parallelism {
            fragment_concurrency = fragment_concurrency.min(io_parallelism.get() as usize);
            let config = SchedulerConfig::new(SchedulerConfig::max_bandwidth_bytes(
                io_parallelism.get() as usize,
            ));
            read_config = read_config.with_scan_scheduler(ScanScheduler::new_with_io_parallelism(
                store.clone(),
                config,
                io_parallelism,
            ));
        }

        let physical_schema = Arc::new(projection.to_bare_schema());
        let mut batch = self
            .take_addresses(
                &row_addrs,
                physical_schema,
                read_config,
                fragment_concurrency,
                io_stats,
            )
            .await?;

        if options.with_row_address || with_row_offset {
            if batch.num_rows() != row_addrs.len() {
                return Err(Error::not_supported_source(format!(
                    "Expected {} rows, got {}.  A take operation that includes row addresses must not target deleted rows.",
                    row_addrs.len(),
                    batch.num_rows()
                ).into()));
            }

            let row_addr_col = UInt64Array::from(row_addrs.into_owned());
            if with_row_offset && batch.column_by_name(ROW_OFFSET).is_none() {
                let offsets = self.fragment_row_offsets().await?;
                let row_offset_col = compute_row_offset_array(&row_addr_col, &offsets)?;
                batch = batch.try_with_column(
                    Field::new(ROW_OFFSET, DataType::UInt64, false),
                    Arc::new(row_offset_col),
                )?;
            }
            if options.with_row_address && batch.column_by_name(ROW_ADDR).is_none() {
                batch = batch.try_with_column(
                    Field::new(ROW_ADDR, DataType::UInt64, false),
                    Arc::new(row_addr_col),
                )?;
            }
        }

        Ok(Some(batch))
    }

    async fn resolve_row_ids<'a>(
        &self,
        row_ids: &'a [u64],
        options: TakeOptions,
    ) -> Result<Cow<'a, [u64]>> {
        let Some(resolved) = resolve_take_row_ids(self, row_ids, !options.ignore_deletions).await?
        else {
            return Ok(Cow::Borrowed(row_ids));
        };
        if options.missing_row_policy == MissingRowPolicy::Error
            && let Some(first_missing_index) = resolved.iter().position(Option::is_none)
        {
            let resolved_count = resolved.iter().filter(|address| address.is_some()).count();
            return Err(Error::invalid_input(format!(
                "Could not resolve all requested row IDs: requested {}, resolved {}; first missing row ID {} was deleted or not found",
                row_ids.len(),
                resolved_count,
                row_ids[first_missing_index]
            )));
        }
        Ok(Cow::Owned(
            resolved.into_iter().flatten().map(u64::from).collect(),
        ))
    }

    async fn take_addresses(
        &self,
        row_addrs: &[u64],
        schema: Arc<Schema>,
        config: FragReadConfig,
        concurrency: usize,
        io_stats: Option<Arc<dyn IoStatsRecorder>>,
    ) -> Result<RecordBatch> {
        let first = RowAddress::from(row_addrs[0]);
        let last = RowAddress::from(row_addrs[row_addrs.len() - 1]);
        let mut sorted = true;
        let mut contiguous = first.fragment_id() == last.fragment_id();
        for pair in row_addrs.windows(2) {
            sorted &= pair[0] <= pair[1];
            contiguous &= pair[0].checked_add(1) == Some(pair[1]);
        }
        if contiguous {
            let fragment_id = first.fragment_id();
            let fragment = self.get_fragment(fragment_id as usize).ok_or_else(|| {
                Error::invalid_input(format!(
                    "rowaddr start: {} belongs to non-existent fragment: {fragment_id}",
                    row_addrs[0]
                ))
            })?;
            if let Some(io_stats) = io_stats.as_ref() {
                io_stats.record_fragments_selected(1);
            }
            let reader = fragment
                .open_with_io_stats(&schema, config, io_stats)
                .await?;
            return reader
                .read_range_as_batch(first.row_offset() as usize..last.row_offset() as usize + 1)
                .await;
        }

        if sorted {
            return self
                .take_fragment_groups(row_addrs, schema, config, concurrency, false, io_stats)
                .await;
        }

        let mut unique = row_addrs.to_vec();
        unique.sort_unstable();
        unique.dedup();
        let batch = self
            .take_fragment_groups(
                &unique,
                schema,
                config.with_row_address(true),
                concurrency,
                true,
                io_stats,
            )
            .await?;

        // Some rows may have been deleted. Reorder using returned addresses
        // and restore duplicates from the original request.
        let returned = batch
            .column_by_name(ROW_ADDR)
            .ok_or_else(|| Error::internal("_rowaddr column not found"))?
            .as_primitive::<UInt64Type>()
            .values();
        let positions: HashMap<u64, u64> = returned
            .iter()
            .enumerate()
            .map(|(index, &address)| (address, index as u64))
            .collect();
        let indices: UInt64Array = row_addrs
            .iter()
            .filter_map(|address| positions.get(address).copied())
            .collect();
        debug_assert!(indices.len() >= batch.num_rows());

        // Arrow's take kernel does not handle empty structs.
        let reordered = take_struct_array(&batch.into(), &indices)?;
        Ok(reordered.into())
    }

    async fn take_fragment_groups(
        &self,
        sorted_addrs: &[u64],
        schema: Arc<Schema>,
        config: FragReadConfig,
        concurrency: usize,
        reorder: bool,
        io_stats: Option<Arc<dyn IoStatsRecorder>>,
    ) -> Result<RecordBatch> {
        // Own each fragment for the lifetime of its buffered read.
        let table = Arc::new(self.clone());
        let mut groups = Vec::new();
        for addresses in sorted_addrs.chunk_by(|a, b| {
            RowAddress::from(*a).fragment_id() == RowAddress::from(*b).fragment_id()
        }) {
            let fragment_id = RowAddress::from(addresses[0]).fragment_id();
            let metadata = self.view().find_fragment(u64::from(fragment_id)).ok_or_else(|| {
                if reorder {
                    Error::invalid_input(format!(
                        "Unknown fragment id {fragment_id} in fragment filter; not part of the current dataset version"
                    ))
                } else {
                    Error::invalid_input(format!(
                        "rowaddr {} belongs to non-existent fragment: {fragment_id}", addresses[0]
                    ))
                }
            })?;
            let fragment = FileFragment::new(table.clone(), metadata.clone());
            let offsets = addresses
                .iter()
                .map(|&address| RowAddress::from(address).row_offset())
                .collect();
            groups.push((fragment, offsets));
        }
        if let Some(io_stats) = io_stats.as_ref() {
            io_stats.record_fragments_selected(groups.len());
        }

        let batches: Vec<RecordBatch> = futures::stream::iter(groups)
            .map(|(fragment, offsets)| {
                read_fragment(
                    fragment,
                    offsets,
                    schema.clone(),
                    config.clone(),
                    io_stats.clone(),
                )
            })
            .buffered(concurrency)
            .try_collect()
            .await?;
        let first = batches
            .first()
            .ok_or_else(|| Error::internal("No fragments resolved for a nonempty take"))?;
        if batches.len() == 1 {
            return Ok(first.clone());
        }
        Ok(concat_batches(first.schema_ref(), &batches)?)
    }
}

// Spell out the Send bound for callers buffering reads across fragments.
#[allow(clippy::manual_async_fn)]
fn read_fragment(
    fragment: FileFragment,
    offsets: Vec<u32>,
    schema: Arc<Schema>,
    config: FragReadConfig,
    io_stats: Option<Arc<dyn IoStatsRecorder>>,
) -> impl Future<Output = Result<RecordBatch>> + Send {
    async move {
        fragment
            .take_rows_with_io_stats(&offsets, &schema, config, io_stats)
            .await
    }
}

fn take_struct_array(array: &StructArray, indices: &UInt64Array) -> Result<StructArray> {
    let nulls = array.nulls().map(|nulls| {
        let is_valid = indices
            .iter()
            .map(|index| index.is_some_and(|index| nulls.is_valid(index as usize)));
        NullBuffer::new(BooleanBuffer::new(
            Buffer::from_iter(is_valid),
            0,
            indices.len(),
        ))
    });

    if array.fields().is_empty() {
        return Ok(StructArray::new_empty_fields(indices.len(), nulls));
    }

    let arrays = array
        .columns()
        .iter()
        .map(|array| {
            let array = match array.data_type() {
                DataType::Struct(_) => Arc::new(take_struct_array(array.as_struct(), indices)?),
                _ => arrow_select::take::take(array, indices, None)?,
            };
            Ok(array)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(StructArray::new(array.fields().clone(), arrays, nulls))
}

#[cfg(test)]
mod tests;
