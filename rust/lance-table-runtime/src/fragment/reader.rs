// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchOptions, UInt32Array};
use arrow_schema::Schema as ArrowSchema;
use arrow_select::concat::concat_batches;
use futures::future::BoxFuture;
use futures::{FutureExt, StreamExt, TryFutureExt, TryStreamExt, join, stream};
use lance_arrow::{RecordBatchExt, SchemaExt};
use lance_core::datatypes::Schema;
use lance_core::utils::deletion::DeletionVector;
use lance_core::utils::io_stats::IoStatsRecorder;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::{
    Error, ROW_ADDR_FIELD, ROW_CREATED_AT_VERSION_FIELD, ROW_ID_FIELD,
    ROW_LAST_UPDATED_AT_VERSION_FIELD, Result,
};
use lance_file::version::ConcreteFileVersion;
use lance_io::ReadBatchParams;
use lance_table::rowids::RowIdSequence;
use lance_table::utils::stream::{
    ReadBatchFutStream, ReadBatchTask, ReadBatchTaskStream, RowIdAndDeletesConfig,
    wrap_with_row_id_and_delete,
};
use tokio::sync::OnceCell;

use super::{FileFragment, FragReadConfig, GenericFileReader};
use crate::overlay::{
    OverlayReadPlanner, OverlayReadSelection, merge_overlay_batch, plan_overlays, resolve_overlays,
};
use crate::rowids::{RowVersionKind, load_row_id_sequence, load_row_version_sequence};

const DEFAULT_BATCH_READ_SIZE: u32 = 1024;

impl FileFragment {
    /// Open a FileFragment with a given default projection.
    ///
    /// The projection selects data columns. Use [`FragReadConfig`] to request
    /// system columns and share I/O schedulers across fragment reads. An empty
    /// projection can be used to read only system columns.
    pub async fn open(
        &self,
        projection: &Schema,
        read_config: FragReadConfig,
    ) -> Result<FragmentReader> {
        self.open_with_io_stats(projection, read_config, None).await
    }

    pub(crate) async fn open_with_io_stats(
        &self,
        projection: &Schema,
        read_config: FragReadConfig,
        io_stats: Option<Arc<dyn IoStatsRecorder>>,
    ) -> Result<FragmentReader> {
        if read_config.make_deletions_null
            && !read_config.with_row_id
            && !read_config.with_row_address
        {
            return Err(Error::invalid_input(
                "make_deletions_null requires with_row_id or with_row_address".to_string(),
            ));
        }
        let open_files = self.open_readers(projection, &read_config, io_stats.clone());
        let deletion_vec_load = if read_config.ignore_deletions {
            futures::future::Either::Left(futures::future::ready(Ok(None)))
        } else {
            futures::future::Either::Right(self.get_deletion_vector())
        };

        // Preserve the legacy reader's eager initialization. Current readers
        // can defer the sequence until a read actually requests stable row IDs.
        let needs_row_ids = read_config.with_row_id
            || self.table.manifest.data_storage_format.lance_file_format()
                == ConcreteFileVersion::V1;
        let row_id_load = if self.table.manifest.uses_stable_row_ids() && needs_row_ids {
            futures::future::Either::Left(
                load_row_id_sequence(&self.table, &self.metadata).map_ok(Some),
            )
        } else {
            futures::future::Either::Right(futures::future::ready(Ok(None)))
        };

        let version_load = |kind: RowVersionKind, wanted: bool| {
            if wanted {
                futures::future::Either::Left(load_row_version_sequence(
                    &self.table,
                    &self.metadata,
                    kind,
                ))
            } else {
                futures::future::Either::Right(futures::future::ready(Ok(None)))
            }
        };
        let last_updated_at_load = version_load(
            RowVersionKind::LastUpdatedAt,
            read_config.with_row_last_updated_at_version,
        );
        let created_at_load = version_load(
            RowVersionKind::CreatedAt,
            read_config.with_row_created_at_version,
        );

        let (
            opened_files,
            deletion_vec,
            row_id_sequence,
            last_updated_at_sequence,
            created_at_sequence,
        ) = join!(
            open_files,
            deletion_vec_load,
            row_id_load,
            last_updated_at_load,
            created_at_load
        );
        let opened_files = opened_files?;
        let deletion_vec = deletion_vec?;
        let row_id_sequence = row_id_sequence?;
        let last_updated_at_sequence = last_updated_at_sequence?;
        let created_at_sequence = created_at_sequence?;

        if opened_files.is_empty() && !read_config.has_system_cols() {
            return Err(Error::not_found(format!(
                "No data files found for schema: {}, fragment_id={}",
                projection,
                self.id()
            )));
        }

        let num_physical_rows = self.physical_rows().await?;

        let mut reader = FragmentReader::try_new(
            self.id(),
            deletion_vec,
            row_id_sequence,
            opened_files,
            ArrowSchema::from(projection),
            self.count_rows().await?,
            num_physical_rows,
            Arc::new(self.clone()),
        )?;

        // Plan overlay resolution from coverage metadata (no files opened here); the
        // readers are opened lazily on read, pruned to the rows each read touches.
        if !self.metadata.overlays.is_empty() {
            let planner = plan_overlays(self, projection)?;
            if !planner.is_empty() {
                reader.overlay = Some(OverlayReadState {
                    planner: Arc::new(planner),
                    read_config: Arc::new(read_config.clone()),
                    io_stats,
                });
            }
        }

        if read_config.with_row_id {
            reader.with_row_id();
        }
        if read_config.with_row_address {
            reader.with_row_address();
        }
        if read_config.with_row_last_updated_at_version {
            reader.with_row_last_updated_at_version(last_updated_at_sequence);
        }
        if read_config.with_row_created_at_version {
            reader.with_row_created_at_version(created_at_sequence);
        }
        if read_config.make_deletions_null {
            reader.with_make_deletions_null();
        }

        Ok(reader)
    }
}

/// [`FragmentReader`] is an abstract reader for a [`FileFragment`].
///
/// It opens the data files that contains the columns of the projection schema, and
/// reconstruct the RecordBatch from columns read from each data file.
#[derive(Debug)]
pub struct FragmentReader {
    /// Readers and schema of each opened data file.
    readers: Vec<Box<dyn GenericFileReader>>,

    /// The output schema. The defines the order in which the columns are returned.
    output_schema: ArrowSchema,

    /// The deleted row IDs
    deletion_vec: Option<Arc<DeletionVector>>,

    /// Stable row IDs, shared by clones and loaded on demand by current readers.
    /// An empty cell means not loaded; it does not mean row IDs are addresses.
    row_id_sequence: Arc<OnceCell<Arc<RowIdSequence>>>,

    /// ID of the fragment
    fragment_id: usize,

    /// True if we should generate a row id for the output
    with_row_id: bool,

    /// True if we should generate a row address column in output
    with_row_addr: bool,

    /// True if we should generate a last updated at version column in output
    with_row_last_updated_at_version: bool,

    /// True if we should generate a created at version column in output
    with_row_created_at_version: bool,

    /// If true, deleted rows will be set to null, which is fast
    /// If false, deleted rows will be removed from the batch, requiring a copy
    make_deletions_null: bool,

    /// Snapshot and request-specific store binding for deferred sequence reads.
    fragment: Arc<FileFragment>,

    /// The last_updated_at version sequence (loaded from fragment metadata)
    last_updated_at_sequence: Option<Arc<lance_table::rowids::version::RowDatasetVersionSequence>>,

    /// The created_at version sequence (loaded from fragment metadata)
    created_at_sequence: Option<Arc<lance_table::rowids::version::RowDatasetVersionSequence>>,

    // total number of real rows in the fragment (num_physical_rows - num_deleted_rows)
    num_rows: usize,

    // total number of physical rows in the fragment (all rows, ignoring deletions)
    num_physical_rows: usize,

    /// Read-time state for resolving data overlay files: the coverage plan plus
    /// what is needed to open overlay readers. `None` when the fragment has no
    /// overlays. Overlays are merged into base batches (by `offset_in_frag`) before
    /// deletion filtering, opening only the files each read's rows touch.
    overlay: Option<OverlayReadState>,
}

/// What [`FragmentReader`] needs to resolve overlays at read time: the coverage
/// plan and configuration for opening overlay readers once the read's rows are known.
#[derive(Clone, Debug)]
struct OverlayReadState {
    planner: Arc<OverlayReadPlanner>,
    read_config: Arc<FragReadConfig>,
    io_stats: Option<Arc<dyn IoStatsRecorder>>,
}

// Custom clone impl needed because it is not easy to clone Box<dyn GenericFileReader>
//
// We currently need FragmentReader to be Clone because the pushdown scan clones it
// to reuse the fragment reader for both "scan with row id" and "scan without row id"
impl Clone for FragmentReader {
    fn clone(&self) -> Self {
        Self {
            readers: self
                .readers
                .iter()
                .map(|reader| reader.clone_box())
                .collect::<Vec<_>>(),
            output_schema: self.output_schema.clone(),
            deletion_vec: self.deletion_vec.clone(),
            row_id_sequence: self.row_id_sequence.clone(),
            fragment_id: self.fragment_id,
            with_row_id: self.with_row_id,
            with_row_addr: self.with_row_addr,
            with_row_last_updated_at_version: self.with_row_last_updated_at_version,
            with_row_created_at_version: self.with_row_created_at_version,
            make_deletions_null: self.make_deletions_null,
            fragment: self.fragment.clone(),
            last_updated_at_sequence: self.last_updated_at_sequence.clone(),
            created_at_sequence: self.created_at_sequence.clone(),
            num_rows: self.num_rows,
            num_physical_rows: self.num_physical_rows,
            overlay: self.overlay.clone(),
        }
    }
}

impl std::fmt::Display for FragmentReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FragmentReader(id={})", self.fragment_id)
    }
}

impl FragmentReader {
    #[allow(clippy::too_many_arguments)]
    fn try_new(
        fragment_id: usize,
        deletion_vec: Option<Arc<DeletionVector>>,
        row_id_sequence: Option<Arc<RowIdSequence>>,
        readers: Vec<Box<dyn GenericFileReader>>,
        output_schema: ArrowSchema,
        num_rows: usize,
        num_physical_rows: usize,
        fragment: Arc<FileFragment>,
    ) -> Result<Self> {
        Ok(Self {
            readers,
            output_schema,
            deletion_vec,
            row_id_sequence: Arc::new(OnceCell::new_with(row_id_sequence)),
            fragment_id,
            with_row_id: false,
            with_row_addr: false,
            with_row_last_updated_at_version: false,
            with_row_created_at_version: false,
            make_deletions_null: false,
            fragment,
            last_updated_at_sequence: None,
            created_at_sequence: None,
            num_rows,
            num_physical_rows,
            overlay: None,
        })
    }

    async fn row_id_sequence(&self) -> Result<Option<Arc<RowIdSequence>>> {
        if !self.with_row_id || !self.fragment.table.manifest.uses_stable_row_ids() {
            return Ok(None);
        }
        // Readers may enable row IDs after open, including on a clone. Keep
        // successful loads even when the session cache is disabled, but allow
        // retries after a failed or cancelled load.
        let sequence = self
            .row_id_sequence
            .get_or_try_init(|| load_row_id_sequence(&self.fragment.table, &self.fragment.metadata))
            .await?;
        Ok(Some(sequence.clone()))
    }

    #[doc(hidden)]
    pub fn with_row_id(&mut self) -> &mut Self {
        self.with_row_id = true;
        self.output_schema = self
            .output_schema
            .try_with_column(ROW_ID_FIELD.clone())
            .expect("Table already has a column named _rowid");
        self
    }

    #[doc(hidden)]
    pub fn with_row_address(&mut self) -> &mut Self {
        self.with_row_addr = true;
        self.output_schema = self
            .output_schema
            .try_with_column(ROW_ADDR_FIELD.clone())
            .expect("Table already has a column named _rowaddr");
        self
    }

    #[doc(hidden)]
    pub fn with_make_deletions_null(&mut self) -> &mut Self {
        self.make_deletions_null = true;
        self
    }

    /// Emit the `_row_last_updated_at_version` column, served from `sequence`;
    /// `None` means the fragment has no version metadata and every row reads
    /// as version 1.
    pub(crate) fn with_row_last_updated_at_version(
        &mut self,
        sequence: Option<Arc<lance_table::rowids::version::RowDatasetVersionSequence>>,
    ) -> &mut Self {
        self.with_row_last_updated_at_version = true;
        self.last_updated_at_sequence = sequence;

        // Add the version column to the output schema
        self.output_schema = self
            .output_schema
            .try_with_column(ROW_LAST_UPDATED_AT_VERSION_FIELD.clone())
            .expect("Table already has a column named _row_last_updated_at_version");

        self
    }

    /// Emit the `_row_created_at_version` column, served from `sequence`;
    /// `None` means the fragment has no version metadata and every row reads
    /// as version 1.
    pub(crate) fn with_row_created_at_version(
        &mut self,
        sequence: Option<Arc<lance_table::rowids::version::RowDatasetVersionSequence>>,
    ) -> &mut Self {
        self.with_row_created_at_version = true;
        self.created_at_sequence = sequence;

        // Add the version column to the output schema
        self.output_schema = self
            .output_schema
            .try_with_column(ROW_CREATED_AT_VERSION_FIELD.clone())
            .expect("Table already has a column named _row_created_at_version");

        self
    }

    /// Merge data overlay values onto a stream of base batches.
    ///
    /// Runs on physical rows in read order, *before* deletion filtering, so each
    /// row can be addressed by its position in the fragment (its `offset_in_frag`,
    /// derived from `params`) and deletions take precedence naturally: an overlay
    /// value for a deleted row is dropped along with the row downstream. A no-op
    /// when the fragment has no overlays.
    ///
    /// The read's `offset_in_frag` values are known from `params` up front, so
    /// overlays are resolved here to just the files this read's rows touch — an
    /// overlay whose cells fall outside the read is not opened at all. Within each
    /// batch, the overlay reads (only the values that batch needs) are then issued
    /// concurrently with the base read rather than after it.
    async fn merge_overlays(
        &self,
        merged: ReadBatchTaskStream,
        params: &ReadBatchParams,
        total_num_rows: u32,
    ) -> Result<ReadBatchTaskStream> {
        let Some(overlay) = &self.overlay else {
            return Ok(merged);
        };
        let selection = Arc::new(OverlayReadSelection::new(params, total_num_rows)?);

        // Open only the overlay readers this read touches (pruned by row selection).
        let plans = resolve_overlays(
            &overlay.planner,
            &selection,
            &self.fragment,
            &overlay.read_config,
            overlay.io_stats.clone(),
        )
        .await?;
        if plans.is_empty() {
            return Ok(merged);
        }
        let plans = Arc::new(plans);

        // Batches arrive in physical read order, so a running total of the rows seen
        // so far identifies its selection slice. Materialize only the batch's
        // offsets when its future is polled, not while scheduling the whole read.
        let mut rows_seen = 0usize;
        let stream = merged
            .map(move |task| {
                let num_rows = task.num_rows;
                let start = rows_seen;
                rows_seen += num_rows as usize;
                let selection = selection.clone();
                let plans = plans.clone();
                let inner = task.task;
                ReadBatchTask {
                    num_rows,
                    task: async move {
                        let batch_selection = selection.batch(start, num_rows as usize)?;
                        merge_overlay_batch(inner, &batch_selection, &plans).await
                    }
                    .boxed(),
                }
            })
            .boxed();
        Ok(stream)
    }

    async fn new_read_impl<'a, F>(
        &'a self,
        params: ReadBatchParams,
        batch_size: u32,
        read_fn: F,
    ) -> Result<ReadBatchFutStream>
    where
        F: Fn(&'a dyn GenericFileReader) -> BoxFuture<'a, Result<ReadBatchTaskStream>>,
    {
        let total_num_rows = self.num_physical_rows as u32;
        // Note that the fragment length might be considerably smaller if there are deleted rows.
        // E.g. if a fragment has 100 rows but rows 0..10 are deleted we still need to make
        // sure it is valid to read / take 0..100
        if !params.valid_given_len(total_num_rows as usize) {
            return Err(Error::invalid_input(format!(
                "Invalid read params {} for fragment with {} addressable rows",
                params, total_num_rows
            )));
        }
        // If just the row id or address there is no need to actually read any data
        // and we don't need to involve the readers at all.
        //
        // The v1 reader does not support reading batches with zero columns, so
        // we need this as a separate code path.
        // In these cases, we can just emit batches with zero columns and rely
        // on `wrap_with_row_id_and_delete` to add the row id or address column.
        //
        // We could potentially delete the support for no-columns in the wrap function or
        // we can delete this path once we migrate away from any support of v1.
        let system_columns_only = self.num_system_cols() == self.output_schema.fields.len();
        let merged = if system_columns_only {
            let selected_rows = params.to_offsets_total(total_num_rows).len();
            let empty_schema = Arc::new(ArrowSchema::empty());
            let full_batch = RecordBatch::try_new_with_options(
                empty_schema.clone(),
                Vec::new(),
                &RecordBatchOptions::new().with_row_count(Some(batch_size as usize)),
            )?;
            let tasks = (0..selected_rows)
                .step_by(batch_size as usize)
                .map(move |offset| {
                    let num_rows = (batch_size as usize).min(selected_rows - offset);
                    let batch = if num_rows == batch_size as usize {
                        full_batch.clone()
                    } else {
                        RecordBatch::try_new_with_options(
                            empty_schema.clone(),
                            Vec::new(),
                            &RecordBatchOptions::new().with_row_count(Some(num_rows)),
                        )
                        .expect("an empty schema accepts an explicit row count")
                    };
                    ReadBatchTask {
                        task: std::future::ready(Ok(batch)).boxed(),
                        num_rows: num_rows as u32,
                    }
                });
            stream::iter(tasks).boxed()
        } else {
            // Read each data file, these reads should produce streams of equal sized
            // tasks.  In other words, if we get 3 tasks of 20 rows and then a task
            // of 10 rows from one data file we should get the same from the other.
            //
            // We launch all readers' scheduling work concurrently — for v2 files
            // this is where the decode scheduler's `initialize` I/O happens, so
            // running them in parallel keeps the per-file scheduling I/Os from
            // serializing.
            let read_futs = self.readers.iter().filter_map(|reader| {
                // Normally we filter out empty readers in the open_readers method
                // However, we will keep the first empty reader to use for row id
                // purposes on some legacy paths and so we need to filter that out
                // here.
                if reader.projection().fields.is_empty() {
                    None
                } else {
                    Some(read_fn(reader.as_ref()))
                }
            });
            let read_streams = futures::future::try_join_all(read_futs).await?;
            // Merge the streams, this merges the generated batches
            lance_table::utils::stream::merge_streams(read_streams)
        };

        let merged = self.merge_overlays(merged, &params, total_num_rows).await?;

        // Add the row id column (if needed) and delete rows (if a deletion
        // vector is present).
        let config = RowIdAndDeletesConfig {
            deletion_vector: self.deletion_vec.clone(),
            row_id_sequence: self.row_id_sequence().await?,
            make_deletions_null: self.make_deletions_null,
            with_row_id: self.with_row_id,
            with_row_addr: self.with_row_addr,
            with_row_last_updated_at_version: self.with_row_last_updated_at_version,
            with_row_created_at_version: self.with_row_created_at_version,
            last_updated_at_sequence: self.last_updated_at_sequence.clone(),
            created_at_sequence: self.created_at_sequence.clone(),
            params,
            total_num_rows,
        };
        let output_schema = Arc::new(self.output_schema.clone());
        Ok(
            wrap_with_row_id_and_delete(merged, self.fragment_id as u32, config)
                // Finally, reorder the columns to match the order specified in the projection
                .map(move |batch_fut| {
                    let output_schema = output_schema.clone();
                    batch_fut
                        .map(move |batch| {
                            let batch = batch?;
                            if system_columns_only
                                && batch.schema().as_ref() == output_schema.as_ref()
                            {
                                Ok(batch)
                            } else {
                                batch.project_by_schema(&output_schema).map_err(Error::from)
                            }
                        })
                        .boxed()
                })
                .boxed(),
        )
    }

    fn patch_range_for_deletions(&self, range: Range<u32>, dv: &DeletionVector) -> Range<u32> {
        let mut start = range.start;
        let mut end = range.end;
        for val in dv.to_sorted_iter() {
            if val <= start {
                start += 1;
                end += 1;
            } else if val < end {
                end += 1;
            } else {
                break;
            }
        }
        start..end
    }

    async fn do_read_range(
        &self,
        mut range: Range<u32>,
        batch_size: u32,
        skip_deleted_rows: bool,
    ) -> Result<ReadBatchFutStream> {
        if skip_deleted_rows && let Some(deletion_vector) = self.deletion_vec.as_ref() {
            range = self.patch_range_for_deletions(range, deletion_vector.as_ref());
        }
        self.new_read_impl(
            ReadBatchParams::Range(range.start as usize..range.end as usize),
            batch_size,
            move |reader| {
                reader.read_range_tasks(
                    range.start as u64..range.end as u64,
                    batch_size,
                    reader.projection().clone(),
                )
            },
        )
        .await
    }

    fn num_system_cols(&self) -> usize {
        self.with_row_id as usize
            + self.with_row_addr as usize
            + self.with_row_created_at_version as usize
            + self.with_row_last_updated_at_version as usize
    }

    /// Reads a range of rows from the fragment
    ///
    /// This function interprets the request as the Xth to the Nth row of the fragment (after deletions)
    /// and will always return range.len().min(self.num_rows()) rows.
    ///
    /// This is async because it drives the per-data-file decode scheduler
    /// `initialize` work before returning the stream — see
    /// [`GenericFileReader`].
    pub async fn read_range(
        &self,
        range: Range<u32>,
        batch_size: u32,
    ) -> Result<ReadBatchFutStream> {
        self.do_read_range(range, batch_size, true).await
    }

    /// Takes a range of rows from the fragment
    ///
    /// Unlike [`Self::read_range`], this function will NOT skip deleted rows.  If rows are deleted they will
    /// be filtered or set to null.  This function may return less than range.len() rows as a result.
    ///
    /// This is async for the same reason as [`Self::read_range`].
    pub async fn take_range(
        &self,
        range: Range<u32>,
        batch_size: u32,
    ) -> Result<ReadBatchFutStream> {
        self.do_read_range(range, batch_size, false).await
    }

    /// Reads all rows from the fragment.
    ///
    /// This is async for the same reason as [`Self::read_range`].
    pub async fn read_all(&self, batch_size: u32) -> Result<ReadBatchFutStream> {
        self.new_read_impl(ReadBatchParams::RangeFull, batch_size, move |reader| {
            reader.read_all_tasks(batch_size, reader.projection().clone())
        })
        .await
    }

    // This method is a clone of new_read_impl but returns tasks instead of batches
    //
    // It also only supports v2 files
    ///
    /// This is async for the same reason as [`Self::read_range`].
    pub async fn read_ranges(
        &self,
        ranges: Arc<[Range<u64>]>,
        batch_size: u32,
    ) -> Result<ReadBatchFutStream> {
        let total_num_rows = self.num_physical_rows as u32;
        let mut num_requested_rows = 0;
        // Note that row ranges at this point are physical and not logical.
        for range in ranges.as_ref() {
            if range.end > total_num_rows as u64 {
                return Err(Error::internal(format!(
                    "Invalid read of range {:?} for fragment {} with {} addressable rows",
                    range, self.fragment_id, total_num_rows
                )));
            }
            num_requested_rows += range.end - range.start;
        }

        let system_columns_only = self.num_system_cols() == self.output_schema.fields.len();
        let merged_stream = if system_columns_only {
            let empty_schema = Arc::new(ArrowSchema::empty());
            let full_batch = RecordBatch::try_new_with_options(
                empty_schema.clone(),
                Vec::new(),
                &RecordBatchOptions::new().with_row_count(Some(batch_size as usize)),
            )?;
            let tasks = (0..num_requested_rows)
                .step_by(batch_size as usize)
                .map(move |offset| {
                    let num_rows = (batch_size as u64).min(num_requested_rows - offset);
                    let batch = if num_rows == batch_size as u64 {
                        full_batch.clone()
                    } else {
                        RecordBatch::try_new_with_options(
                            empty_schema.clone(),
                            Vec::new(),
                            &RecordBatchOptions::new().with_row_count(Some(num_rows as usize)),
                        )
                        .expect("an empty schema accepts an explicit row count")
                    };
                    ReadBatchTask {
                        task: std::future::ready(Ok(batch)).boxed(),
                        num_rows: num_rows as u32,
                    }
                });
            stream::iter(tasks).boxed()
        } else {
            // Read each data file, these reads should produce streams of equal sized
            // tasks.  In other words, if we get 3 tasks of 20 rows and then a task
            // of 10 rows from one data file we should get the same from the other.
            //
            // Run all readers' scheduling concurrently so the per-file
            // `initialize` I/Os overlap.
            let read_futs = self.readers.iter().map(|reader| {
                reader.read_ranges_tasks(ranges.clone(), batch_size, reader.projection().clone())
            });
            let read_streams = futures::future::try_join_all(read_futs).await?;
            // Merge the streams, this merges the generated batches
            lance_table::utils::stream::merge_streams(read_streams)
        };

        let params = ReadBatchParams::Ranges(ranges);
        let merged_stream = self
            .merge_overlays(merged_stream, &params, total_num_rows)
            .await?;

        // Add the row id column (if needed) and delete rows (if a deletion
        // vector is present).
        let config = RowIdAndDeletesConfig {
            deletion_vector: self.deletion_vec.clone(),
            row_id_sequence: self.row_id_sequence().await?,
            make_deletions_null: self.make_deletions_null,
            with_row_id: self.with_row_id,
            with_row_addr: self.with_row_addr,
            with_row_last_updated_at_version: self.with_row_last_updated_at_version,
            with_row_created_at_version: self.with_row_created_at_version,
            last_updated_at_sequence: self.last_updated_at_sequence.clone(),
            created_at_sequence: self.created_at_sequence.clone(),
            params,
            total_num_rows,
        };
        let output_schema = Arc::new(self.output_schema.clone());
        Ok(
            wrap_with_row_id_and_delete(merged_stream, self.fragment_id as u32, config)
                // Finally, reorder the columns to match the order specified in the projection
                .map(move |batch_fut| {
                    let output_schema = output_schema.clone();
                    batch_fut
                        .map(move |batch| {
                            let batch = batch?;
                            if system_columns_only
                                && batch.schema().as_ref() == output_schema.as_ref()
                            {
                                Ok(batch)
                            } else {
                                batch.project_by_schema(&output_schema).map_err(Error::from)
                            }
                        })
                        .boxed()
                })
                .boxed(),
        )
    }

    /// Reads a range and concatenates the result into one batch.
    pub async fn read_range_as_batch(&self, range: Range<usize>) -> Result<RecordBatch> {
        let batches = self
            .take_range(
                range.start as u32..range.end as u32,
                DEFAULT_BATCH_READ_SIZE,
            )
            .await?
            .buffered(get_num_compute_intensive_cpus())
            .try_collect::<Vec<_>>()
            .await?;
        concat_batches(&Arc::new(self.output_schema.clone()), batches.iter()).map_err(Error::from)
    }

    /// Take rows from this fragment.
    pub async fn take(
        &self,
        indices: &[u32],
        batch_size: u32,
        take_priority: Option<u32>,
    ) -> Result<ReadBatchFutStream> {
        let indices_arr = UInt32Array::from(indices.to_vec());
        self.new_read_impl(
            ReadBatchParams::Indices(indices_arr),
            batch_size,
            move |reader| {
                reader.take_all_tasks(
                    indices,
                    batch_size,
                    reader.projection().clone(),
                    take_priority,
                )
            },
        )
        .await
    }

    /// Take rows from this fragment, will perform a copy if the underlying reader returns multiple
    /// batches.  May return an error if the taken rows do not fit into a single batch.
    ///
    /// Indices may be unordered or duplicated. Rows are returned in the requested order.
    pub async fn take_as_batch(
        &self,
        indices: &[u32],
        take_priority: Option<u32>,
    ) -> Result<RecordBatch> {
        // The v2 encoding layer requires strictly increasing indices. Normalize
        // here so callers can request arbitrary row order and duplicate rows.
        let is_sorted_and_unique = indices.windows(2).all(|w| w[0] < w[1]);
        let (unique_indices, restore_order) = if is_sorted_and_unique {
            (Cow::Borrowed(indices), None)
        } else {
            let mut sorted_positions = indices
                .iter()
                .copied()
                .filter(|&index| {
                    self.make_deletions_null
                        || self
                            .deletion_vec
                            .as_ref()
                            .is_none_or(|deletion_vec| !deletion_vec.contains(index))
                })
                // Restoration positions only count surviving requests.
                .enumerate()
                .collect::<Vec<_>>();
            sorted_positions.sort_unstable_by_key(|&(_, index)| index);

            let mut unique: Vec<u32> = Vec::with_capacity(sorted_positions.len());
            let mut restore_indices = vec![0_u32; sorted_positions.len()];
            for (position, idx) in sorted_positions {
                if unique.last() != Some(&idx) {
                    unique.push(idx);
                }
                restore_indices[position] = (unique.len() - 1) as u32;
            }
            (Cow::Owned(unique), Some(UInt32Array::from(restore_indices)))
        };

        let batches = self
            .take(&unique_indices, u32::MAX, take_priority)
            .await?
            .buffered(get_num_compute_intensive_cpus())
            .try_collect::<Vec<_>>()
            .await?;
        let mut batch = concat_batches(&Arc::new(self.output_schema.clone()), batches.iter())?;

        if let Some(restore_order) = restore_order {
            batch = arrow_select::take::take_record_batch(&batch, &restore_order)?;
        }

        Ok(batch)
    }

    pub(super) async fn take_rows(&self, row_offsets: &[u32]) -> Result<RecordBatch> {
        if row_offsets.len() > 1 && row_ids_contiguous(row_offsets) {
            let range =
                (row_offsets[0] as usize)..(row_offsets[row_offsets.len() - 1] as usize + 1);
            self.read_range_as_batch(range).await
        } else {
            self.take_as_batch(row_offsets, None).await
        }
    }
}

fn row_ids_contiguous(row_ids: &[u32]) -> bool {
    if row_ids.is_empty() {
        return false;
    }

    let mut last_id = row_ids[0];

    for id in row_ids.iter().skip(1) {
        if last_id.checked_add(1) != Some(*id) {
            return false;
        }
        last_id = *id;
    }

    true
}
