// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Wraps a Fragment of the dataset.

mod aligned_update;
pub mod session;
pub mod write;

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use arrow_array::cast::as_primitive_array;
use arrow_array::{RecordBatch, RecordBatchReader, UInt64Array};
use datafusion::logical_expr::Expr;
use datafusion::scalar::ScalarValue;
use futures::future::try_join_all;
use futures::{FutureExt, Stream, StreamExt, TryStreamExt};
use lance_core::datatypes::{BlobHandling, Schema, SchemaCompareOptions};
use lance_core::utils::deletion::DeletionVector;
use lance_core::{Error, ROW_ADDR, ROW_ID, Result};
use lance_datafusion::utils::StreamingWriteSource;
use lance_encoding::decoder::DecoderPlugins;
use lance_file::reader::{CachedFileMetadata, FileMetadataIndex};
use lance_file::version::ConcreteFileVersion;
use lance_file::writer::FileWriterOptions;
use lance_file::{determine_file_version, versions as file_versions};
use lance_io::scheduler::{FileScheduler, ScanScheduler, SchedulerConfig};
use lance_io::stream::RecordBatchStream;
use lance_io::utils::CachedFileSize;
use lance_table::format::{DataFile, Fragment};
use lance_table::utils::stream::ReadBatchFutStream;
use lance_table_runtime::Table;
use roaring::RoaringBitmap;

use self::write::FragmentCreateBuilder;

use super::hash_joiner::HashJoiner;
use super::scanner::Scanner;
use super::updater::Updater;
use super::write::seeds::LanceWriteSeedHook;
use super::{NewColumnTransform, WriteParams, schema_evolution, versions};
use crate::blob::BlobWriteSources;
use crate::dataset::fragment::session::FragmentSession;
use crate::dataset::overlay::writer::{OverlayWriter, WriteOverlayError};
use crate::dataset::{Dataset, UpdateJoinOptions};

/// Result of a fragment column update: updated fragment metadata, modified field ids, and physical
/// row offsets that matched the join (for stable row-id version metadata).
#[derive(Debug, Clone)]
pub struct FragmentUpdateColumnsResult {
    pub fragment: Fragment,
    pub fields_modified: Vec<u32>,
    /// Physical row offsets (0-based within this fragment) whose columns were rewritten from the right-hand stream.
    pub matched_offsets: RoaringBitmap,
}

/// Cleanup ownership for files staged by [`FileFragment::add_columns_with_cleanup`].
///
/// Call [`Self::cleanup`] only after establishing that the outer commit did not
/// land. Dropping this value does not remove any files because a failed commit
/// response can have an ambiguous outcome.
#[derive(Debug)]
#[must_use = "retain this token until the outer fragment merge has committed"]
pub struct FragmentAddColumnsCleanup {
    original_fragment: FileFragment,
    fragments_to_cleanup: Vec<Fragment>,
}

impl FragmentAddColumnsCleanup {
    /// Remove only the files created by the fragment-level add-columns operation.
    ///
    /// Pre-existing fragment files and files stored in an external base are
    /// preserved. Cleanup is best effort; failures are logged for later dataset
    /// garbage collection.
    pub async fn cleanup(self) {
        schema_evolution::cleanup_new_column_data_files(
            std::slice::from_ref(&self.original_fragment),
            &self.fragments_to_cleanup,
        )
        .await;
    }
}

/// A staged fragment-level add-columns result and its cleanup ownership.
///
/// The fragment and schema can be used to build an outer [`Operation::Merge`].
/// Keep `cleanup` until that commit succeeds. If the commit definitively fails,
/// invoking the token removes only files staged by this operation.
///
/// [`Operation::Merge`]: crate::dataset::transaction::Operation::Merge
#[derive(Debug)]
#[must_use = "the cleanup token must be retained until the outer merge commits"]
pub struct FragmentAddColumnsResult {
    /// Fragment metadata containing the staged columns.
    pub fragment: Fragment,
    /// Dataset schema containing the staged columns.
    pub schema: Schema,
    /// Ownership token for the files written by this operation.
    pub cleanup: FragmentAddColumnsCleanup,
}

/// A Fragment of a Lance [`Dataset`].
///
/// The interface is modeled after `pyarrow.dataset.Fragment`.
#[derive(Debug, Clone)]
pub struct FileFragment {
    core: lance_table_runtime::FileFragment,
    dataset: Arc<Dataset>,
}

pub(crate) use lance_table_runtime::fragment::V1FragmentReader;
pub use lance_table_runtime::fragment::{
    BaseSchedulers, FragReadConfig, FragmentReader, GenericFileReader,
};

pub(crate) use lance_table_runtime::fragment::{duplicate_field_path, relax_nullability};

impl FileFragment {
    /// Creates a new FileFragment.
    pub fn new(dataset: Arc<Dataset>, metadata: Fragment) -> Self {
        let table = Arc::new(dataset.table_view().to_table());
        Self::from_table(dataset, table, metadata)
    }

    pub(crate) fn from_table(dataset: Arc<Dataset>, table: Arc<Table>, metadata: Fragment) -> Self {
        let core = lance_table_runtime::FileFragment::new(table, metadata);
        Self { dataset, core }
    }

    pub(super) fn as_runtime(&self) -> &lance_table_runtime::FileFragment {
        &self.core
    }

    /// Create a new [`FileFragment`] from a [`StreamingWriteSource`].
    ///
    /// This method can be used before a `Dataset` is created. For example,
    /// Fragments can be created distributed first, before a central machine to
    /// commit the dataset with these fragments.
    ///
    pub async fn create(
        dataset_uri: &str,
        id: usize,
        source: impl StreamingWriteSource,
        params: Option<WriteParams>,
    ) -> Result<Fragment> {
        let mut builder = FragmentCreateBuilder::new(dataset_uri);

        if let Some(params) = params.as_ref() {
            builder = builder.write_params(params);
        }

        builder.write(source, Some(id as u64)).await
    }

    /// Create a list of [`FileFragment`] from a [`StreamingWriteSource`].
    pub async fn create_fragments(
        dataset_uri: &str,
        source: impl StreamingWriteSource,
        params: Option<WriteParams>,
    ) -> Result<Vec<Fragment>> {
        let mut builder = FragmentCreateBuilder::new(dataset_uri);

        if let Some(params) = params.as_ref() {
            builder = builder.write_params(params);
        }

        builder.write_fragments(source).await
    }

    pub async fn create_from_file(
        filename: &str,
        dataset: &Dataset,
        fragment_id: usize,
        physical_rows: Option<usize>,
    ) -> Result<Fragment> {
        let filepath = dataset.data_dir().join(filename);
        let file_version =
            determine_file_version(dataset.table_state.object_store.as_ref(), &filepath, None)
                .await?;

        super::versions::create_fragment_from_file(
            file_version,
            dataset.manifest.data_storage_format.lance_file_format(),
            filename,
            dataset,
            fragment_id,
            physical_rows,
        )
        .await
    }

    pub(crate) async fn create_from_v1_file(
        filename: &str,
        dataset: &Dataset,
        fragment_id: usize,
        physical_rows: Option<usize>,
    ) -> Result<Fragment> {
        Ok(Fragment::with_file_legacy(
            fragment_id as u64,
            filename,
            dataset.schema(),
            physical_rows,
        ))
    }

    pub(crate) async fn create_from_current_file(
        filename: &str,
        dataset: &Dataset,
        fragment_id: usize,
    ) -> Result<Fragment> {
        let filepath = dataset.data_dir().join(filename);
        // Load the file metadata, confirm the schema is compatible, and
        // determine the column offsets
        let mut frag = Fragment::new(fragment_id as u64);
        let scheduler = ScanScheduler::new(
            dataset.table_state.object_store.clone(),
            SchedulerConfig::max_bandwidth(&dataset.table_state.object_store),
        );
        let file_scheduler = scheduler
            .open_file(&filepath, &CachedFileSize::unknown())
            .await?;
        let reader = lance_file::reader::FileReader::try_open(
            file_scheduler,
            None,
            Arc::<DecoderPlugins>::default(),
            &dataset
                .table_state
                .metadata_cache
                .file_metadata_cache(&filepath),
            dataset
                .table_state
                .file_reader_options
                .as_deref()
                .cloned()
                .unwrap_or_default(),
        )
        .await?;
        // If the schemas are not compatible we can't calculate field id offsets
        reader
            .schema()
            .check_compatible(dataset.schema(), &SchemaCompareOptions::default())?;
        let projection = file_versions::reader_projection_from_whole_schema(
            dataset.schema(),
            reader.metadata().version(),
        );
        frag.physical_rows = Some(reader.metadata().num_rows as usize);

        let column_indices = projection
            .column_indices
            .into_iter()
            .map(|column| column as i32)
            .collect();
        frag.add_file(
            filename,
            dataset.schema().field_ids(),
            column_indices,
            reader.metadata().version(),
            None,
        );
        Ok(frag)
    }

    /// On-disk bytes for each field of the current dataset schema.
    ///
    /// The ids are the schema's fields, the same set as
    /// [`DatasetStatisticsExt::calculate_data_stats`](crate::dataset::statistics::DatasetStatisticsExt::calculate_data_stats).
    /// A field with no bytes in this fragment is included as `0`. Ids that have been dropped, and
    /// tombstoned slots still stored in a data file, are omitted. A field that
    /// appears in more than one data file is summed, including bytes in data
    /// overlays. The pairs are sorted by field id.
    ///
    /// V1 datasets do not record per-field sizes. This returns the initialized
    /// zeros and does not open data files, the same as dataset statistics.
    pub async fn storage_stats(&self) -> Result<Vec<(u32, u64)>> {
        // V1 files have no per-field byte statistics. Dataset statistics leave
        // the initialized zeros in place and skip the data files.
        if self
            .dataset
            .manifest
            .data_storage_format
            .lance_file_format()
            == ConcreteFileVersion::V1
        {
            let mut totals: Vec<(u32, u64)> = self
                .dataset
                .schema()
                .field_ids()
                .into_iter()
                .map(|id| (id as u32, 0))
                .collect();
            totals.sort_unstable_by_key(|(field_id, _)| *field_id);
            return Ok(totals);
        }
        let scan_scheduler = ScanScheduler::new(
            self.dataset.table_state.object_store.clone(),
            SchedulerConfig::max_bandwidth(self.dataset.table_state.object_store.as_ref()),
        );
        let pairs = self
            .storage_stats_with(self.dataset.schema(), scan_scheduler)
            .await?;
        let mut totals = HashMap::<u32, u64>::from_iter(
            self.dataset
                .schema()
                .field_ids()
                .into_iter()
                .map(|id| (id as u32, 0)),
        );
        for (field_id, bytes) in pairs {
            if let Some(total) = totals.get_mut(&field_id) {
                *total += bytes;
            }
        }
        let mut totals: Vec<(u32, u64)> = totals.into_iter().collect();
        totals.sort_unstable_by_key(|(field_id, _)| *field_id);
        Ok(totals)
    }

    /// Storage stats as `(field_id, bytes_on_disk)` pairs for this fragment.
    ///
    /// The same field id can appear more than once when it is stored in more
    /// than one data file.
    pub(crate) async fn storage_stats_with(
        &self,
        dataset_schema: &Schema,
        scan_scheduler: Arc<ScanScheduler>,
    ) -> Result<Vec<(u32, u64)>> {
        let read_config = FragReadConfig::default().with_scan_scheduler(scan_scheduler);
        let mut stats = Vec::new();
        for reader in self
            .core
            .open_readers_with_full_metadata(dataset_schema, &read_config)
            .await?
        {
            stats.extend(reader.storage_stats()?);
        }
        // An overlay's values live outside the base data files, but they are
        // bytes their fields occupy on disk all the same.
        let overlay_readers = try_join_all(self.metadata().overlays.iter().map(|overlay| {
            self.core.open_reader_with_full_metadata(
                &overlay.data_file,
                Some(dataset_schema),
                &read_config,
            )
        }))
        .await?;
        for reader in overlay_readers.into_iter().flatten() {
            stats.extend(reader.storage_stats()?);
        }
        Ok(stats)
    }

    pub fn dataset(&self) -> &Dataset {
        self.dataset.as_ref()
    }

    pub fn schema(&self) -> &Schema {
        self.core.schema()
    }

    /// Returns the fragment's metadata.
    pub fn metadata(&self) -> &Fragment {
        self.core.metadata()
    }

    pub(super) fn metadata_mut(&mut self) -> &mut Fragment {
        self.core.metadata_mut()
    }

    /// The id of this [`FileFragment`].
    pub fn id(&self) -> usize {
        self.core.id()
    }

    /// The number of data files in this fragment.
    pub fn num_data_files(&self) -> usize {
        self.core.num_data_files()
    }

    /// Gets the data file for a given field.
    pub fn data_file_for_field(&self, field_id: u32) -> Option<&DataFile> {
        self.core.data_file_for_field(field_id)
    }

    /// Open a reader for the supplied projection and read options.
    ///
    /// Use [`FragReadConfig`] to include system columns or share I/O schedulers.
    pub async fn open(
        &self,
        projection: &Schema,
        read_config: FragReadConfig,
    ) -> Result<FragmentReader> {
        self.core.open(projection, read_config).await
    }

    pub(crate) async fn open_v1_fragment_reader(
        &self,
        projection: &Schema,
        read_config: &FragReadConfig,
    ) -> Result<V1FragmentReader> {
        self.core
            .open_v1_fragment_reader(projection, read_config)
            .await
    }

    /// Count the rows in this fragment.
    pub async fn count_rows(&self, filter: Option<String>) -> Result<usize> {
        match filter {
            Some(expr) => self
                .scan()
                .project(&Vec::<String>::default())
                .unwrap()
                .with_row_id()
                .filter(&expr)?
                .count_rows()
                .await
                .map(|v| v as usize),
            None => self.core.count_rows().await,
        }
    }

    /// Get the number of rows that have been deleted in this fragment.
    pub async fn count_deletions(&self) -> Result<usize> {
        self.core.count_deletions().await
    }

    /// Return the physical row count from trusted manifest metadata.
    pub fn fast_physical_rows(&self) -> Result<usize> {
        self.core.fast_physical_rows()
    }

    /// Return the deleted row count from manifest metadata.
    pub fn fast_num_deletions(&self) -> Result<usize> {
        self.core.fast_num_deletions()
    }

    /// Return the number of live rows using the counts in manifest metadata.
    pub fn fast_logical_rows(&self) -> Result<usize> {
        self.core.fast_logical_rows()
    }

    /// Get the number of physical rows in this [`FileFragment`].
    ///
    /// Physical rows include deleted rows. If there are no deleted rows, this is
    /// equal to the number of logical rows in the fragment.
    ///
    /// When the dataset manifest contains writer-version metadata, this method
    /// trusts [`Fragment::physical_rows`] when it is present. This includes
    /// file-less fragments, which can represent rows whose columns are all null.
    /// Use [`Self::metadata`] to inspect the persisted [`Fragment`] metadata.
    ///
    /// Legacy manifests without writer-version metadata may contain an incorrect
    /// cached count, so this method ignores it and reads the count from a data
    /// file. If such a fragment has no data files, this method returns
    /// [`Error::NotFound`]. See [`Self::validate`] for the corresponding metadata
    /// checks.
    ///
    /// # Example
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # async fn example(dataset: &Dataset) -> Result<()> {
    /// for fragment in dataset.get_fragments() {
    ///     let physical_rows = fragment.physical_rows().await?;
    ///     if fragment.num_data_files() == 0 {
    ///         assert_eq!(fragment.metadata().physical_rows, Some(physical_rows));
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn physical_rows(&self) -> Result<usize> {
        self.core.physical_rows().await
    }

    /// Validate this [`FileFragment`] and its persisted [`Fragment`] metadata.
    ///
    /// Verifies:
    /// * All field ids in the fragment are distinct.
    /// * Within each data file, field ids are in increasing order.
    /// * All data files holding user data exist and have the same length. A
    ///   file kept only for the spilled row lineage it carries is not opened;
    ///   [`Dataset::validate`] reads that lineage back, which checks its length.
    /// * Field ids are distinct between data files.
    /// * The deletion file exists and has row offsets in the correct range.
    /// * [`Fragment::physical_rows`], when present, matches the data-file length.
    /// * [`DeletionFile::num_deleted_rows`](lance_table::format::DeletionFile::num_deleted_rows),
    ///   when present, matches the deletion-vector length.
    ///
    /// When no file holds user data, the fragment is accepted if the manifest contains
    /// writer-version metadata, [`Fragment::physical_rows`] is present, and the
    /// remaining metadata is valid. The cached count supplies the expected length
    /// for deletion metadata validation. A missing count returns
    /// [`Error::CorruptFile`].
    ///
    /// For legacy manifests without writer-version metadata, a file-less fragment
    /// returns [`Error::CorruptFile`] even when [`Fragment::physical_rows`] is
    /// present, because the cached count cannot be trusted. See
    /// [`Self::physical_rows`] for the corresponding row-count lookup behavior.
    ///
    /// # Example
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # async fn example(dataset: &Dataset) -> Result<()> {
    /// for fragment in dataset.get_fragments() {
    ///     fragment.validate().await?;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn validate(&self) -> Result<()> {
        self.core.validate().await
    }

    /// Open a [`FragmentSession`], which manages a short-lived session of [`FileFragment`].
    ///
    /// This API works well for users making repeated requests over the same projected schema.
    pub async fn open_session(
        &self,
        projection: &Schema,
        with_row_address: bool,
    ) -> Result<FragmentSession> {
        FragmentSession::open(Arc::new(self.clone()), projection, with_row_address).await
    }

    /// Take rows from this fragment based on the offset in the file.
    ///
    /// This will always return the same number of rows as the input indices.
    /// If indices are out-of-bounds, this will return an error.
    ///
    /// The projection may include the `_rowid` and `_rowaddr` system columns
    /// (for example from [`Schema::project_preserve_system_columns`]). They
    /// are returned in the order they appear in the projection.
    pub async fn take(&self, indices: &[u32], projection: &Schema) -> Result<RecordBatch> {
        self.core.take(indices, projection).await
    }

    /// Read a fragment-local half-open physical row interval without applying deletions.
    ///
    /// Unlike logical range reads, offsets address the immutable rows stored in
    /// the fragment's files. Deleted positions remain present with their stored
    /// column values. Callers can stream the batches into
    /// [`Dataset::write_data_file_part`](super::Dataset::write_data_file_part)
    /// when independently computing a physical-row part.
    ///
    /// ```
    /// # use lance::{dataset::fragment::FileFragment, Result};
    /// # use lance_core::datatypes::Schema;
    /// # async fn read(fragment: &FileFragment, schema: &Schema) -> Result<()> {
    /// let batches = fragment.read_physical_slice(0..100, schema, 1024).await?;
    /// # let _ = batches;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn read_physical_slice(
        &self,
        rows: Range<u64>,
        projection: &Schema,
        batch_size: u32,
    ) -> Result<ReadBatchFutStream> {
        if batch_size == 0 {
            return Err(Error::invalid_input(
                "read_physical_slice batch_size must be greater than zero",
            ));
        }
        let physical_rows = self.physical_rows().await? as u64;
        if rows.start > rows.end || rows.end > physical_rows {
            return Err(Error::invalid_input(format!(
                "physical slice {}..{} is outside fragment {} with {} physical rows",
                rows.start,
                rows.end,
                self.id(),
                physical_rows
            )));
        }
        let offset = i64::try_from(rows.start).map_err(|_| {
            Error::invalid_input(format!(
                "physical slice start {} exceeds the supported scan offset range",
                rows.start
            ))
        })?;
        let limit = i64::try_from(rows.end - rows.start).map_err(|_| {
            Error::invalid_input(format!(
                "physical slice length {} exceeds the supported scan limit range",
                rows.end - rows.start
            ))
        })?;

        // Build a read-only view of this exact fragment without its deletion
        // file. The normal scanner can then apply overlays and Blob descriptor
        // materialization while offsets still address immutable physical rows.
        let mut physical_metadata = self.metadata().clone();
        physical_metadata.deletion_file = None;
        let mut physical_dataset = self.dataset.as_ref().clone();
        let mut physical_manifest = self.dataset.manifest.as_ref().clone();
        physical_manifest.fragments = Arc::new(vec![physical_metadata.clone()]);
        physical_dataset.manifest = Arc::new(physical_manifest);
        let physical_dataset = Arc::new(physical_dataset);
        let fragment = Self::new(physical_dataset.clone(), physical_metadata);
        let mut scanner = fragment.scan();
        let columns = projection
            .fields
            .iter()
            .map(|field| field.name.as_str())
            .collect::<Vec<_>>();
        scanner.project(&columns)?;
        scanner.batch_size(batch_size as usize);
        scanner.limit(Some(limit), Some(offset))?;

        let has_blob_columns = projection.fields_pre_order().any(|field| field.is_blob());
        if has_blob_columns {
            scanner.with_row_address();
        }
        let stream = scanner.try_into_stream().await?;
        if has_blob_columns {
            let rewrite_plan = Arc::new(super::blob::BlobV2BatchRewritePlan::try_new(
                projection,
                stream.schema().as_ref(),
                false,
                false,
            )?);
            Ok(stream
                .map(move |batch_result| {
                    let physical_dataset = physical_dataset.clone();
                    let rewrite_plan = rewrite_plan.clone();
                    async move {
                        rewrite_plan
                            .transform_batch(&physical_dataset, batch_result?)
                            .await
                    }
                    .boxed()
                })
                .boxed())
        } else {
            Ok(stream
                .map(|batch_result| async move { batch_result }.boxed())
                .boxed())
        }
    }

    /// Get the deletion vector for this fragment, using the cache if available.
    pub async fn get_deletion_vector(&self) -> Result<Option<Arc<DeletionVector>>> {
        self.core.get_deletion_vector().await
    }

    /// Get the file metadata for this fragment, using the cache if available.
    pub async fn get_file_metadata(
        &self,
        file_scheduler: &FileScheduler,
    ) -> Result<Arc<CachedFileMetadata>> {
        self.core.get_file_metadata(file_scheduler).await
    }

    pub async fn get_file_metadata_index(
        &self,
        file_scheduler: &FileScheduler,
        known_schema: Option<(Arc<Schema>, u64)>,
    ) -> Result<Arc<FileMetadataIndex>> {
        self.core
            .get_file_metadata_index(file_scheduler, known_schema)
            .await
    }

    pub(crate) async fn take_rows(
        &self,
        row_offsets: &[u32],
        projection: &Schema,
        read_config: FragReadConfig,
    ) -> Result<RecordBatch> {
        self.core
            .take_rows(row_offsets, projection, read_config)
            .await
    }

    /// Scan this [`FileFragment`].
    ///
    /// See [`Dataset::scan`].
    pub fn scan(&self) -> Scanner {
        Scanner::from_fragment(self.dataset.clone(), self.metadata().clone())
    }

    /// Create an [`Updater`] to append new columns.
    ///
    /// The `columns` parameter is a list of existing columns to be read from
    /// the fragment. They can be used to derive new columns. This is allowed to
    /// be empty.
    ///
    /// The columns `_rowaddr` and `_rowid` can be used to load the row id or row address
    ///
    /// The `schemas` parameter is a tuple of the write schema (just the new fields)
    /// and the full schema (the target schema after the update). If the write
    /// schema is None, it is inferred from the first batch of results. The full
    /// schema is inferred by appending the write schema to the existing schema.
    ///
    /// The `batch_size` parameter can be used to influence how much data is processed
    /// at a time. This can be useful to control memory usage when processing very large
    /// fields. The batch_size will only be used if the dataset is a v2 dataset.  It will
    /// be ignored for v1 datasets.
    ///
    /// The `blob_handling` parameter controls the in-memory representation of blob
    /// columns read by the updater. If unset, the dataset schema is used unchanged.
    pub(crate) async fn updater<T: AsRef<str>>(
        &self,
        columns: Option<&[T]>,
        schemas: Option<(Schema, Schema)>,
        batch_size: Option<u32>,
        blob_handling: Option<BlobHandling>,
    ) -> Result<Updater> {
        let write_version = self
            .dataset
            .manifest
            .data_storage_format
            .lance_file_format();
        self.updater_with_version(columns, schemas, batch_size, blob_handling, write_version)
            .await
    }

    pub(crate) async fn updater_with_version<T: AsRef<str>>(
        &self,
        columns: Option<&[T]>,
        schemas: Option<(Schema, Schema)>,
        batch_size: Option<u32>,
        blob_handling: Option<BlobHandling>,
        write_version: ConcreteFileVersion,
    ) -> Result<Updater> {
        let updater = Box::pin(self.core.updater_with_version(
            columns,
            schemas,
            batch_size,
            blob_handling,
            write_version,
            || super::scanner::get_default_batch_size().unwrap_or(1024),
        ))
        .await?;
        let hook = Arc::new(LanceWriteSeedHook::new(self.dataset.session.clone()));
        Ok(Updater::from_runtime(
            self.clone(),
            updater.with_seed_hook(hook),
        ))
    }

    pub async fn merge_columns(
        &mut self,
        stream: impl RecordBatchReader + Send + 'static,
        left_on: &str,
        right_on: &str,
    ) -> Result<(Fragment, Schema)> {
        let stream = Box::new(stream);
        if self.schema().field(left_on).is_none() && left_on != ROW_ID && left_on != ROW_ADDR {
            return Err(Error::invalid_input(format!(
                "Column {} does not exist in the left side fragment",
                left_on
            )));
        };
        let right_schema = stream.schema();
        if right_schema.field_with_name(right_on).is_err() {
            return Err(Error::invalid_input(format!(
                "Column {} does not exist in the right side fragment",
                right_on
            )));
        };

        for field in right_schema.fields() {
            if field.name() == right_on {
                // right_on is allowed to exist in the dataset, since it may be
                // the same as left_on.
                continue;
            }
            if self.schema().field(field.name()).is_some() {
                return Err(Error::invalid_input(format!(
                    "Column {} exists in left side fragment and right side dataset",
                    field.name()
                )));
            }
        }
        // Hash join
        let joiner = Arc::new(HashJoiner::try_new(stream, right_on).await?);
        // Final schema is union of current schema, plus the RHS schema without
        // the right_on key.
        let mut new_schema: Schema = self.schema().merge(joiner.out_schema().as_ref())?;
        // Use the same starting id as the updater so schema and data file ids match.
        new_schema.try_set_field_id(Some(self.dataset.manifest.max_field_id()))?;

        let new_fragment = self
            .clone()
            .merge(left_on, &joiner)
            .await
            .map(Fragment::from)?;

        Ok((new_fragment, new_schema))
    }

    pub(crate) async fn merge(mut self, join_column: &str, joiner: &HashJoiner) -> Result<Self> {
        let mut updater = self.updater(Some(&[join_column]), None, None, None).await?;

        while let Some(batch) = updater.next().await? {
            let batch = joiner
                .collect(&self.dataset, batch[join_column].clone())
                .await?;
            updater.update(batch).await?;
        }

        *self.metadata_mut() = updater.finish().await?;

        Ok(self)
    }

    /// Same as [`Self::update_columns_with_offsets`] but discards the matched row offsets.
    /// Use [`Self::update_columns_with_offsets`] if you need per-row version metadata for stable row IDs.
    pub async fn update_columns(
        &mut self,
        right_stream: impl RecordBatchReader + Send + 'static,
        left_on: &str,
        right_on: &str,
    ) -> Result<(Fragment, Vec<u32>)> {
        let r = self
            .update_columns_with_offsets(right_stream, left_on, right_on)
            .await?;
        Ok((r.fragment, r.fields_modified))
    }

    /// Same operation as [`Self::update_columns`], and also returns matched physical row offsets for stable row IDs.
    pub async fn update_columns_with_offsets(
        &mut self,
        right_stream: impl RecordBatchReader + Send + 'static,
        left_on: &str,
        right_on: &str,
    ) -> Result<FragmentUpdateColumnsResult> {
        self.update_columns_with_options(
            right_stream,
            left_on,
            right_on,
            UpdateJoinOptions::default(),
        )
        .await
    }

    /// Updates columns using explicit join-strategy and external-resource controls.
    ///
    /// The returned offsets are physical row offsets within this fragment. Existing callers can
    /// continue to use [`Self::update_columns`] or [`Self::update_columns_with_offsets`], which use
    /// [`UpdateJoinOptions::default`].
    ///
    /// ```
    /// # use arrow_array::RecordBatchReader;
    /// # use lance::Result;
    /// # use lance::dataset::UpdateJoinOptions;
    /// # use lance::dataset::fragment::FileFragment;
    /// # async fn update(
    /// #     fragment: &mut FileFragment,
    /// #     updates: impl RecordBatchReader + Send + 'static,
    /// # ) -> Result<()> {
    /// let options = UpdateJoinOptions::default()
    ///     .with_hash_thresholds(500_000, 1024 * 1024 * 1024)
    ///     .with_external_memory_pool_bytes(256 * 1024 * 1024);
    /// fragment
    ///     .update_columns_with_options(updates, "id", "id", options)
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn update_columns_with_options(
        &mut self,
        right_stream: impl RecordBatchReader + Send + 'static,
        left_on: &str,
        right_on: &str,
        options: UpdateJoinOptions,
    ) -> Result<FragmentUpdateColumnsResult> {
        super::update_join::update_columns(self, Box::new(right_stream), left_on, right_on, options)
            .await
    }

    /// Append new columns to the fragment
    ///
    /// This is the fragment-level version of [`Dataset::add_columns`].
    /// Call [`Self::add_columns_with_cleanup`] when the resulting files will be
    /// committed by a separate fallible operation and explicit cleanup is needed.
    pub async fn add_columns(
        &self,
        transforms: NewColumnTransform,
        read_columns: Option<Vec<String>>,
        batch_size: Option<u32>,
    ) -> Result<(Fragment, Schema)> {
        let result = self
            .add_columns_with_cleanup(transforms, read_columns, batch_size)
            .await?;
        Ok((result.fragment, result.schema))
    }

    /// Append new columns while retaining ownership of the staged files.
    ///
    /// This variant is intended for distributed or multi-step callers that
    /// perform the final [`Operation::Merge`] separately. The returned cleanup
    /// token should be dropped after a successful commit and invoked only when
    /// the caller can establish that the commit did not land.
    ///
    /// # Example
    ///
    /// ```
    /// # use lance::Result;
    /// # use lance::dataset::{NewColumnTransform, fragment::FileFragment};
    /// # async fn stage(
    /// #     fragment: &FileFragment,
    /// #     transform: NewColumnTransform,
    /// # ) -> Result<()> {
    /// let staged = fragment
    ///     .add_columns_with_cleanup(transform, None, None)
    ///     .await?;
    ///
    /// // Use staged.fragment and staged.schema in an outer Merge commit.
    /// // If that commit definitively fails, release the staged files:
    /// staged.cleanup.cleanup().await;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [`Operation::Merge`]: crate::dataset::transaction::Operation::Merge
    pub async fn add_columns_with_cleanup(
        &self,
        transforms: NewColumnTransform,
        read_columns: Option<Vec<String>>,
        batch_size: Option<u32>,
    ) -> Result<FragmentAddColumnsResult> {
        let (fragments, schema, fragments_to_cleanup, _) =
            schema_evolution::add_columns_to_fragments(
                self.dataset.as_ref(),
                transforms,
                read_columns,
                std::slice::from_ref(self),
                batch_size,
            )
            .await?;
        let fragment_count = fragments.len();
        let Ok([fragment]) = <Vec<Fragment> as TryInto<[Fragment; 1]>>::try_into(fragments) else {
            return Err(Error::internal(format!(
                "fragment add-columns produced {} fragments for fragment {}",
                fragment_count,
                self.id()
            )));
        };
        Ok(FragmentAddColumnsResult {
            fragment,
            schema,
            cleanup: FragmentAddColumnsCleanup {
                original_fragment: self.clone(),
                fragments_to_cleanup,
            },
        })
    }

    /// Write new data for columns of this fragment as a standalone data file,
    /// without committing it, and return the
    /// [`DataReplacementGroup`](super::transaction::DataReplacementGroup)
    /// describing it.
    ///
    /// Unlike [`Self::add_columns`], the staged file answers for a field that
    /// already exists, so this recomputes a column rather than appending one.
    ///
    /// `schema` names the fields being written. Each must be a top-level
    /// column the dataset schema already defines, matching its manifest
    /// definition; a column is staged whole, so a nested field cannot be
    /// staged on its own. To recompute a new column, declare it first with an
    /// all-null [`Self::add_columns`], then stage its data. Physical layout
    /// comes from the manifest, so staging cannot change a field's storage
    /// encoding. Batch columns are matched by name at every level, so struct
    /// children may arrive in any order, but a batch whose fields are not
    /// exactly the target's, at every level, is rejected.
    ///
    /// `data` must produce exactly the fragment's physical row count, nulls
    /// included: the file is positionally aligned with the fragment and no
    /// deletion vector is applied on the way in. Batches are pulled one at a
    /// time, so the full column need not be held in memory.
    ///
    /// Callers should take care to set the read version correctly. If this is
    /// not done then multiple replacements to the same field will not be
    /// detected as a conflict.
    pub async fn write_columns(
        &self,
        data: impl Stream<Item = Result<RecordBatch>> + Send,
        schema: &Schema,
    ) -> Result<super::transaction::DataReplacementGroup> {
        self.write_columns_with_file_writer_options(data, schema, FileWriterOptions::default())
            .await
    }

    /// Write replacement column data with runtime file-writer options.
    ///
    /// This has the same format and commit semantics as [`Self::write_columns`].
    /// The options control buffering and execution only; the fragment's
    /// manifest still selects the file grammar and physical layout.
    pub async fn write_columns_with_file_writer_options(
        &self,
        data: impl Stream<Item = Result<RecordBatch>> + Send,
        schema: &Schema,
        file_writer_options: FileWriterOptions,
    ) -> Result<super::transaction::DataReplacementGroup> {
        let write_version = self
            .dataset
            .manifest
            .data_storage_format
            .lance_file_format();
        let data = data.map(|result| result.map(|batch| (batch, None)));
        self.write_column_with_version_and_options(
            data,
            schema,
            write_version,
            file_writer_options,
            false,
        )
        .await
    }

    /// Stage replacement columns using prepared Blob v2 leaves and trusted
    /// payload sources associated with each batch.
    ///
    /// See [`Self::write_columns`] for the schema, row alignment and commit
    /// contract. Each batch can supply [`BlobWriteSources`] for payloads
    /// uploaded into primary dataset storage or resolved from a prior write.
    /// Only the dataset schema's Managed opt-in determines whether the
    /// writer can adopt a completed upload. Requires file version 2.2 or newer.
    ///
    /// ```
    /// # use arrow_array::RecordBatch;
    /// # use lance::{Result, blob::BlobWriteSources, dataset::fragment::FileFragment};
    /// # use lance_core::datatypes::Schema;
    /// # use lance_file::writer::FileWriterOptions;
    /// # async fn example(fragment: &FileFragment, batch: RecordBatch, schema: &Schema,
    /// #   sources: BlobWriteSources) -> Result<()> {
    /// let replacement = fragment.write_columns_with_blob_sources(
    ///     futures::stream::iter([Ok((batch, sources))]), schema, FileWriterOptions::default(),
    /// ).await?;
    /// # let _ = replacement;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn write_columns_with_blob_sources(
        &self,
        data: impl Stream<Item = Result<(RecordBatch, BlobWriteSources)>> + Send,
        schema: &Schema,
        file_writer_options: FileWriterOptions,
    ) -> Result<super::transaction::DataReplacementGroup> {
        let write_version = self
            .dataset
            .manifest
            .data_storage_format
            .lance_file_format();
        if !versions::writes_blob_v2(write_version) {
            return Err(Error::not_supported(
                "Column writes with Blob sources require Lance file version 2.2 or later",
            ));
        }
        let data = data.map(|result| result.map(|(batch, sources)| (batch, Some(sources))));
        self.write_column_with_version_and_options(
            data,
            schema,
            write_version,
            file_writer_options,
            true,
        )
        .await
    }

    /// Write replacement column data using an exact V2 data file version.
    ///
    /// The input and commit requirements are the same as [`Self::write_columns`].
    /// V1 targets and legacy datasets are not supported.
    /// When the version differs from the manifest default, the commit that
    /// publishes the returned replacement group derives the mixed-version
    /// capability from its final manifest.
    ///
    /// ```
    /// # use lance::{dataset::fragment::FileFragment, Result};
    /// # use lance_core::datatypes::Schema;
    /// # use lance_file::version::ConcreteFileVersion;
    /// # async fn example(fragment: &FileFragment, batch: arrow_array::RecordBatch, schema: &Schema) -> Result<()> {
    /// let replacement = fragment.write_column_with_version(
    ///     futures::stream::iter([Ok(batch)]), schema, ConcreteFileVersion::V2_2,
    /// ).await?;
    /// // Commit the replacement with Operation::DataReplacement.
    /// # Ok(())
    /// # }
    /// ```
    pub async fn write_column_with_version(
        &self,
        data: impl Stream<Item = Result<RecordBatch>> + Send,
        schema: &Schema,
        write_version: ConcreteFileVersion,
    ) -> Result<super::transaction::DataReplacementGroup> {
        let data = data.map(|result| result.map(|batch| (batch, None)));
        self.write_column_with_version_and_options(
            data,
            schema,
            write_version,
            FileWriterOptions::default(),
            false,
        )
        .await
    }

    async fn write_column_with_version_and_options(
        &self,
        data: impl Stream<Item = Result<(RecordBatch, Option<BlobWriteSources>)>> + Send,
        schema: &Schema,
        write_version: ConcreteFileVersion,
        file_writer_options: FileWriterOptions,
        has_blob_sources: bool,
    ) -> Result<super::transaction::DataReplacementGroup> {
        let hook = LanceWriteSeedHook::new(self.dataset.session.clone());
        Box::pin(self.core.write_column_with_version_and_options(
            data,
            schema,
            write_version,
            file_writer_options,
            has_blob_sources,
            Some(&hook),
        ))
        .await
    }

    /// Stage a data overlay supplying new values for a subset of this fragment's
    /// cells, without rewriting its base data files.
    ///
    /// `schema` declares the fields the overlay may supply, by the dataset's own
    /// field ids, exactly as [`Self::write_columns`] does. Values are then fed to
    /// the returned writer keyed by `_rowaddr`; see [`OverlayWriter`] for the
    /// ordering and coverage rules it enforces, and for how to finish or discard
    /// what it stages.
    ///
    /// An overlay is the cheap way to fill a small fraction of a column: write
    /// cost is proportional to the cells supplied, rows keep their addresses, and
    /// no existing values are re-read to carry them forward. It costs a merge on
    /// every later read of the cells it covers, so a dense fill of a whole column
    /// is better served by [`Self::write_columns`].
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use arrow_array::RecordBatch;
    /// # use futures::TryStreamExt;
    /// # use lance::Dataset;
    /// # use lance::dataset::transaction::Operation;
    /// # use lance::dataset::WriteDestination;
    /// # use lance_core::Result;
    /// # async fn backfill(
    /// #     dataset: Arc<Dataset>,
    /// #     mut values: impl futures::TryStream<Ok = RecordBatch, Error = lance_core::Error> + Unpin,
    /// #     field_id: i32,
    /// # ) -> Result<()> {
    /// let read_version = dataset.version().version;
    /// let fragment = dataset.get_fragment(0).expect("fragment 0 exists");
    /// let schema = dataset.schema().project_by_ids(&[field_id], true);
    ///
    /// let mut overlay = fragment.write_overlay(&schema).await?;
    /// // Each batch carries `_rowaddr` plus the fields being filled, for the
    /// // rows it has values for. Addresses must strictly ascend per field.
    /// while let Some(batch) = values.try_next().await? {
    ///     overlay.write_batch(&batch).await?;
    /// }
    ///
    /// if let Some(group) = overlay.finish().await? {
    ///     Dataset::commit(
    ///         WriteDestination::Dataset(dataset),
    ///         Operation::DataOverlay { groups: vec![group] },
    ///         Some(read_version),
    ///         None,
    ///         None,
    ///         Arc::new(Default::default()),
    ///         false,
    ///     )
    ///     .await?;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn write_overlay(
        &self,
        schema: &Schema,
    ) -> std::result::Result<OverlayWriter, WriteOverlayError> {
        Box::pin(self.core.write_overlay(schema)).await
    }

    /// Delete rows from the fragment.
    ///
    /// If all rows are deleted, returns `Ok(None)`. Otherwise, returns a new
    /// fragment with the updated deletion vector. This must be persisted to
    /// the manifest.
    pub async fn delete(self, predicate: &str) -> Result<Option<Self>> {
        // Load existing deletion vector
        let mut deletion_vector = self
            .get_deletion_vector()
            .await?
            .unwrap_or_default()
            .as_ref()
            .clone();

        let starting_length = deletion_vector.len();

        // scan with predicate and row addresses
        let mut scanner = self.scan();

        let predicate_lower = predicate.trim().to_lowercase();
        if predicate_lower == "true" {
            return Ok(None);
        } else if predicate_lower == "false" {
            return Ok(Some(self));
        }

        scanner
            .with_row_address()
            .filter(predicate)?
            .project::<&str>(&[])?;

        // if predicate is `true`, delete the whole fragment
        // else if predicate is `false`, filter the predicate
        // We do this on the expression level after expression optimization has
        // occurred so we also catch expressions that are equivalent to `true`
        if let Some(predicate) = &scanner.get_expr_filter()? {
            if matches!(
                predicate,
                Expr::Literal(ScalarValue::Boolean(Some(false)), _)
            ) {
                return Ok(Some(self));
            }
            if matches!(
                predicate,
                Expr::Literal(ScalarValue::Boolean(Some(true)), _)
            ) {
                return Ok(None);
            }
        }

        // As we get row addrs, add them into our deletion vector
        scanner
            .try_into_stream()
            .await?
            .try_for_each(|batch| {
                let array = batch[ROW_ADDR].clone();
                let int_array: &UInt64Array = as_primitive_array(array.as_ref());

                // _rowaddr is global, not within fragment level. The high bits
                // are the fragment_id, the low bits are the row_id within the
                // fragment.
                let local_row_ids = int_array.values().iter().map(|v| *v as u32);

                deletion_vector.extend(local_row_ids);
                futures::future::ready(Ok(()))
            })
            .await?;

        // If we haven't deleted any additional rows, we can return the fragment as-is.
        if deletion_vector.len() == starting_length {
            return Ok(Some(self));
        }

        self.write_deletions(deletion_vector).await
    }

    pub async fn extend_deletions(
        self,
        new_deletions: impl IntoIterator<Item = u32>,
    ) -> Result<Option<Self>> {
        let Self { dataset, core } = self;
        Ok(Box::pin(core.extend_deletions(new_deletions))
            .await?
            .map(|core| Self { dataset, core }))
    }

    async fn write_deletions(self, deletion_vector: DeletionVector) -> Result<Option<Self>> {
        let Self { dataset, core } = self;
        Ok(Box::pin(core.write_deletions(deletion_vector))
            .await?
            .map(|core| Self { dataset, core }))
    }
}

impl From<FileFragment> for Fragment {
    fn from(fragment: FileFragment) -> Self {
        fragment.core.into()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::dataset::rowids::load_row_id_sequence;
    use arrow::compute::concat_batches;
    use arrow_arith::numeric::mul;
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Int32Type, UInt64Type};
    use arrow_array::{
        Array, ArrayRef, BinaryArray, BooleanArray, Int32Array, Int64Array, RecordBatchIterator,
        StringArray, UInt32Array,
    };
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_arrow::RecordBatchExt;
    use lance_core::utils::address::RowAddress;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_core::{ROW_ID, ROW_ID_FIELD};
    use lance_datagen::{RowCount, array, gen_batch};
    use lance_file::version::{ConcreteFileVersion, LanceFileVersion};
    use lance_file::versions::v1::reader::FileReader as V1FileReader;
    use lance_file::writer::FileWriterOptions;
    use lance_io::{assert_io_eq, assert_io_lt, object_store::ObjectStore};
    use lance_table::format::DeletionFile;
    use pretty_assertions::assert_eq;
    use rstest::rstest;
    use std::collections::HashMap;
    use std::collections::HashSet;

    use super::*;
    use crate::{
        dataset::{
            CommitBuilder, InsertBuilder,
            transaction::{Operation, Transaction, UpdateMode, UpdatedFragmentOffsets},
        },
        session::Session,
        utils::test::TestDatasetGenerator,
    };

    async fn create_dataset(test_uri: &str, data_storage_version: LanceFileVersion) -> Dataset {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, true),
            ArrowField::new("s", DataType::Utf8, true),
        ]));

        let batches: Vec<RecordBatch> = (0..10)
            .map(|i| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20)),
                        Arc::new(StringArray::from_iter_values(
                            (i * 20..(i + 1) * 20).map(|v| format!("s-{}", v)),
                        )),
                    ],
                )
                .unwrap()
            })
            .collect();

        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        Dataset::open(test_uri).await.unwrap()
    }

    fn assert_error_contains(err: Error, expected: &str) {
        assert!(
            err.to_string().contains(expected),
            "expected error to contain {expected:?}, got {err:?}"
        );
    }

    async fn create_dataset_v2(test_uri: &str) -> Dataset {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            true,
        )]));

        let batches: Vec<RecordBatch> = (0..10)
            .map(|i| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20))],
                )
                .unwrap()
            })
            .collect();

        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            data_storage_version: Some(LanceFileVersion::Stable),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        Dataset::open(test_uri).await.unwrap()
    }

    /// End-to-end tests for reading data overlay files (OSS-1324): overlays are
    /// written, committed via the `DataOverlay` transaction, and then resolved on
    /// the `take` and scan read paths.
    mod overlay_read {
        use std::sync::Arc;

        use arrow_array::{
            Array, ArrayRef, Int32Array, RecordBatch, RecordBatchIterator, StructArray, UInt64Array,
        };
        use arrow_buffer::NullBuffer;
        use arrow_schema::{DataType, Field as ArrowField, Fields, Schema as ArrowSchema};
        use futures::{StreamExt, TryStreamExt};
        use lance_core::datatypes::Schema;
        use lance_file::version::LanceFileVersion;
        use lance_file::writer::FileWriterOptions;
        use lance_io::utils::CachedFileSize;
        use lance_table::format::DataFile;
        use lance_table::format::overlay::{DataOverlayFile, OverlayCoverage};
        use object_store::path::Path;
        use roaring::RoaringBitmap;
        use rstest::rstest;

        use crate::dataset::fragment::FragReadConfig;
        use crate::dataset::transaction::{DataOverlayGroup, Operation};
        use crate::dataset::{Dataset, WriteDestination, WriteParams};

        fn bitmap(offsets: impl IntoIterator<Item = u32>) -> RoaringBitmap {
            RoaringBitmap::from_iter(offsets)
        }

        fn i32_array(values: impl IntoIterator<Item = Option<i32>>) -> ArrayRef {
            Arc::new(Int32Array::from_iter(values))
        }

        /// Two-fragment Int32 dataset: `id` (field 0) = 0..12 and `val` (field 1)
        /// = id * 10, written 6 rows per file (fragments 0 and 1).
        ///
        async fn create_base_dataset(version: LanceFileVersion) -> Dataset {
            let schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new("id", DataType::Int32, true),
                ArrowField::new("val", DataType::Int32, true),
            ]));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(0..12)),
                    Arc::new(Int32Array::from_iter_values((0..12).map(|v| v * 10))),
                ],
            )
            .unwrap();
            let write_params = WriteParams {
                max_rows_per_file: 6,
                max_rows_per_group: 6,
                data_storage_version: Some(version),
                ..Default::default()
            };
            let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
            Dataset::write(reader, "memory://", Some(write_params))
                .await
                .unwrap()
        }

        /// Write an overlay file covering `fields` (dataset field ids) of
        /// `fragment_id` with the given coverage and per-field value columns, then
        /// commit it as a `DataOverlay` transaction. `name` makes the file unique.
        #[allow(clippy::too_many_arguments)]
        async fn commit_overlay(
            dataset: Dataset,
            name: &str,
            fragment_id: u64,
            fields: &[i32],
            coverage: OverlayCoverage,
            columns: Vec<ArrayRef>,
            version: LanceFileVersion,
        ) -> Dataset {
            let read_version = dataset.version().version;
            let overlay_schema = dataset.schema().project_by_ids(fields, true);

            let filename = format!("{name}.lance");
            // The manifest records the bare filename; only the physical write is
            // data-dir qualified.
            let path = dataset.data_dir().join(filename.as_str());
            let obj_writer = dataset
                .table_state
                .object_store
                .create(&path)
                .await
                .unwrap();
            let file_version = version.resolve();
            let mut writer = lance_file::versions::create_writer(
                file_version,
                obj_writer,
                overlay_schema,
                FileWriterOptions::default(),
            )
            .unwrap();
            for (column_index, array) in columns.into_iter().enumerate() {
                writer.write_column(column_index, array).await.unwrap();
            }
            let summary = writer.finish().await.unwrap();

            let mut data_file = DataFile::new_unstarted(filename, file_version);
            data_file.fields = writer
                .field_id_to_column_indices()
                .iter()
                .map(|(field_id, _)| *field_id as i32)
                .collect::<Vec<_>>()
                .into();
            data_file.column_indices = writer
                .field_id_to_column_indices()
                .iter()
                .map(|(_, column_index)| *column_index as i32)
                .collect::<Vec<_>>()
                .into();
            data_file.file_size_bytes = CachedFileSize::new(summary.size_bytes);

            let overlay = DataOverlayFile {
                data_file,
                coverage,
                committed_version: 0,
            };
            Dataset::commit(
                WriteDestination::Dataset(Arc::new(dataset)),
                Operation::DataOverlay {
                    groups: vec![DataOverlayGroup {
                        fragment_id,
                        overlays: vec![overlay],
                    }],
                },
                Some(read_version),
                None,
                None,
                Arc::new(Default::default()),
                false,
            )
            .await
            .unwrap()
        }

        /// `collect_paths` feeds `deep_clone`'s copy loop, so an overlay data file
        /// it omits is referenced by the clone's manifest but never copied. The
        /// clone then fails to read the overlaid values.
        #[tokio::test]
        async fn deep_clone_copies_overlay_files() {
            use lance_core::utils::tempfile::TempStdDir;

            let version = LanceFileVersion::Stable;
            let test_dir = TempStdDir::default();
            let source_uri = test_dir.join("source").to_str().unwrap().to_string();
            let clone_uri = test_dir.join("clone").to_str().unwrap().to_string();

            let schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new("id", DataType::Int32, true),
                ArrowField::new("val", DataType::Int32, true),
            ]));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(0..6)),
                    Arc::new(Int32Array::from_iter_values((0..6).map(|v| v * 10))),
                ],
            )
            .unwrap();
            let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
            let dataset = Dataset::write(
                reader,
                &source_uri,
                Some(WriteParams {
                    max_rows_per_file: 6,
                    max_rows_per_group: 6,
                    data_storage_version: Some(version),
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

            // Overlay `val` on offsets 0 and 1 of fragment 0.
            let mut dataset = commit_overlay(
                dataset,
                "overlay0",
                0,
                &[1],
                OverlayCoverage::Shared(Arc::new(bitmap([0, 1]))),
                vec![i32_array([Some(700), Some(701)])],
                version,
            )
            .await;

            let source_version = dataset.version().version;
            dataset
                .tags()
                .create("clone-me", source_version)
                .await
                .unwrap();
            let cloned = dataset
                .deep_clone(&clone_uri, "clone-me", None)
                .await
                .unwrap();

            let batch = cloned
                .scan()
                .try_into_batch()
                .await
                .expect("the clone must be readable, including its overlay files");
            let val = col(&batch, "val");
            assert_eq!(
                val.values()[..2],
                [700, 701],
                "the clone must return the overlaid values, not the base ones"
            );
        }

        /// Deep-cloning a shallow clone has to drop every `base_id`, since the
        /// result owns its files. An overlay whose `base_id` survives points at a
        /// base the new manifest no longer lists.
        #[tokio::test]
        async fn deep_clone_of_shallow_clone_clears_overlay_base_id() {
            use lance_core::utils::tempfile::TempStdDir;

            let version = LanceFileVersion::Stable;
            let test_dir = TempStdDir::default();
            let source_uri = test_dir.join("source").to_str().unwrap().to_string();
            let shallow_uri = test_dir.join("shallow").to_str().unwrap().to_string();
            let deep_uri = test_dir.join("deep").to_str().unwrap().to_string();

            let schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new("id", DataType::Int32, true),
                ArrowField::new("val", DataType::Int32, true),
            ]));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(0..6)),
                    Arc::new(Int32Array::from_iter_values((0..6).map(|v| v * 10))),
                ],
            )
            .unwrap();
            let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
            let dataset = Dataset::write(
                reader,
                &source_uri,
                Some(WriteParams {
                    max_rows_per_file: 6,
                    data_storage_version: Some(version),
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

            let mut dataset = commit_overlay(
                dataset,
                "overlay0",
                0,
                &[1],
                OverlayCoverage::Shared(Arc::new(bitmap([0, 1]))),
                vec![i32_array([Some(700), Some(701)])],
                version,
            )
            .await;

            let source_version = dataset.version().version;
            dataset.tags().create("v", source_version).await.unwrap();
            let mut shallow = dataset
                .shallow_clone(&shallow_uri, "v", None)
                .await
                .unwrap();
            // The shallow clone reaches the overlay through the parent.
            assert!(
                shallow.get_fragments()[0].metadata().overlays[0]
                    .data_file
                    .base_id
                    .is_some(),
                "the shallow clone must reference the parent's overlay by base_id"
            );

            let shallow_version = shallow.version().version;
            shallow.tags().create("v", shallow_version).await.unwrap();
            let deep = shallow.deep_clone(&deep_uri, "v", None).await.unwrap();

            assert_eq!(
                deep.get_fragments()[0].metadata().overlays[0]
                    .data_file
                    .base_id,
                None,
                "a deep clone owns its files, so the overlay must carry no base_id"
            );
            assert!(
                deep.manifest.base_paths.is_empty(),
                "a deep clone lists no external bases"
            );
        }

        fn full_schema(dataset: &Dataset) -> Schema {
            dataset.schema().clone()
        }

        fn col(batch: &RecordBatch, name: &str) -> Int32Array {
            let idx = batch.schema().index_of(name).unwrap();
            batch
                .column(idx)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .clone()
        }

        #[rstest]
        #[tokio::test]
        async fn test_take_covered_and_uncovered(
            #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
        ) {
            let dataset = create_base_dataset(version).await;
            // Overlay fragment 0's `val` at physical offsets {1, 4}.
            let dataset = commit_overlay(
                dataset,
                "ov",
                0,
                &[1],
                OverlayCoverage::dense(bitmap([1, 4])),
                vec![i32_array([Some(111), Some(444)])],
                version,
            )
            .await;

            let frag = dataset.get_fragment(0).unwrap();
            let batch = frag
                .take(&[0, 1, 2, 4], &full_schema(&dataset))
                .await
                .unwrap();
            // Offsets 1 and 4 take overlay values; 0 and 2 fall through to base.
            assert_eq!(col(&batch, "val").values(), &[0, 111, 20, 444]);
            // The unrelated `id` column is untouched.
            assert_eq!(col(&batch, "id").values(), &[0, 1, 2, 4]);
        }

        #[rstest]
        #[tokio::test]
        async fn test_take_newest_overlay_wins(
            #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
        ) {
            let dataset = create_base_dataset(version).await;
            let dataset = commit_overlay(
                dataset,
                "older",
                0,
                &[1],
                OverlayCoverage::dense(bitmap([1, 4])),
                vec![i32_array([Some(111), Some(444)])],
                version,
            )
            .await;
            // A newer overlay (later commit -> higher committed_version) re-covers
            // offset 1.
            let dataset = commit_overlay(
                dataset,
                "newer",
                0,
                &[1],
                OverlayCoverage::dense(bitmap([1])),
                vec![i32_array([Some(999)])],
                version,
            )
            .await;

            let frag = dataset.get_fragment(0).unwrap();
            let batch = frag.take(&[1, 4], &full_schema(&dataset)).await.unwrap();
            // Offset 1 -> newest overlay (999); offset 4 -> only older covers it.
            assert_eq!(col(&batch, "val").values(), &[999, 444]);
        }

        #[rstest]
        #[tokio::test]
        async fn test_take_per_field_coverage(
            #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
        ) {
            let dataset = create_base_dataset(version).await;
            // Sparse overlay: `id` covers {2}, `val` covers {2, 3} — different
            // offset sets and therefore unequal-length value columns.
            let dataset = commit_overlay(
                dataset,
                "sparse",
                0,
                &[0, 1],
                OverlayCoverage::sparse(vec![bitmap([2]), bitmap([2, 3])]),
                vec![i32_array([Some(777)]), i32_array([Some(220), Some(330)])],
                version,
            )
            .await;

            let frag = dataset.get_fragment(0).unwrap();
            let batch = frag.take(&[2, 3], &full_schema(&dataset)).await.unwrap();
            // id: offset 2 covered (777), offset 3 falls through (3).
            assert_eq!(col(&batch, "id").values(), &[777, 3]);
            // val: both offsets covered (220, 330).
            assert_eq!(col(&batch, "val").values(), &[220, 330]);
        }

        #[rstest]
        #[tokio::test]
        async fn test_take_null_override(
            #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
        ) {
            let dataset = create_base_dataset(version).await;
            let dataset = commit_overlay(
                dataset,
                "nullov",
                0,
                &[1],
                OverlayCoverage::dense(bitmap([0])),
                vec![i32_array([None])],
                version,
            )
            .await;

            let frag = dataset.get_fragment(0).unwrap();
            let batch = frag.take(&[0, 1], &full_schema(&dataset)).await.unwrap();
            let val = col(&batch, "val");
            // Offset 0 is covered with a NULL value -> resolves to NULL; offset 1
            // falls through to the base value.
            assert!(val.is_null(0));
            assert_eq!(val.value(1), 10);
        }

        /// Overlays interact correctly with NULL *base* cells (distinct from a NULL
        /// overlay value): a covered row whose base value is NULL is overridden to the
        /// overlay's non-null value, while an uncovered NULL base cell falls through
        /// and stays NULL.
        #[rstest]
        #[tokio::test]
        async fn test_take_null_base_cell(
            #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
        ) {
            let schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new("id", DataType::Int32, true),
                ArrowField::new("val", DataType::Int32, true),
            ]));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(0..6)),
                    // `val` is NULL at offsets 1 and 3.
                    Arc::new(Int32Array::from_iter([
                        Some(0),
                        None,
                        Some(20),
                        None,
                        Some(40),
                        Some(50),
                    ])),
                ],
            )
            .unwrap();
            let write_params = WriteParams {
                max_rows_per_file: 6,
                max_rows_per_group: 6,
                data_storage_version: Some(version),
                ..Default::default()
            };
            let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
            let dataset = Dataset::write(reader, "memory://", Some(write_params))
                .await
                .unwrap();

            // Cover offset 1 (NULL base) and offset 4 (non-null base); leave offset
            // 3's NULL base uncovered.
            let dataset = commit_overlay(
                dataset,
                "nullbase",
                0,
                &[1],
                OverlayCoverage::dense(bitmap([1, 4])),
                vec![i32_array([Some(111), Some(444)])],
                version,
            )
            .await;

            let frag = dataset.get_fragment(0).unwrap();
            let batch = frag.take(&[1, 3, 4], &full_schema(&dataset)).await.unwrap();
            let val = col(&batch, "val");
            // Offset 1: NULL base overridden to 111. Offset 3: uncovered NULL base
            // stays NULL. Offset 4: non-null base overridden to 444.
            assert_eq!(val.value(0), 111);
            assert!(val.is_null(1));
            assert_eq!(val.value(2), 444);
        }

        #[rstest]
        #[tokio::test]
        async fn test_overlay_on_deleted_row_is_inert(
            #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
        ) {
            let mut dataset = create_base_dataset(version).await;
            // Delete global row 1 (fragment 0, physical offset 1).
            dataset.delete("id = 1").await.unwrap();
            // Overlay covers the deleted offset 1 and the live offset 4.
            let dataset = commit_overlay(
                dataset,
                "delov",
                0,
                &[1],
                OverlayCoverage::dense(bitmap([1, 4])),
                vec![i32_array([Some(111), Some(444)])],
                version,
            )
            .await;

            // Scan fragment 0: row 1 is gone, and offset 4's overlay value survives
            // even though the deletion shifts logical positions — coverage is keyed
            // by physical offset.
            let frag = dataset.get_fragment(0).unwrap();
            let mut scanner = frag.scan();
            let batch = scanner
                .project(&["id", "val"])
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();
            assert_eq!(col(&batch, "id").values(), &[0, 2, 3, 4, 5]);
            assert_eq!(col(&batch, "val").values(), &[0, 20, 30, 444, 50]);
        }

        #[rstest]
        #[tokio::test]
        async fn test_scan_multi_fragment_overlays(
            #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
        ) {
            let dataset = create_base_dataset(version).await;
            // Overlay fragment 0 at offset 0 and fragment 1 at offset 0 (global
            // row 6). Each fragment's coverage is independent.
            let dataset = commit_overlay(
                dataset,
                "frag0",
                0,
                &[1],
                OverlayCoverage::dense(bitmap([0])),
                vec![i32_array([Some(1000)])],
                version,
            )
            .await;
            let dataset = commit_overlay(
                dataset,
                "frag1",
                1,
                &[1],
                OverlayCoverage::dense(bitmap([0])),
                vec![i32_array([Some(6000)])],
                version,
            )
            .await;

            let batch = dataset
                .scan()
                .project(&["id", "val"])
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();
            assert_eq!(batch.num_rows(), 12);
            let expected: Vec<i32> = (0..12)
                .map(|i| match i {
                    0 => 1000,
                    6 => 6000,
                    other => other * 10,
                })
                .collect();
            assert_eq!(col(&batch, "val").values(), &expected);
        }

        /// A `take` of a few rows must read only the overlay values those rows
        /// touch — not the whole column. Uses v2.1 (which slices pages on read) and
        /// an incompressible, all-covering overlay, so reading the full column would
        /// be far more bytes than reading a couple of values. This is the regression
        /// guard for the lazy, value-pushdown overlay read.
        #[tokio::test]
        async fn test_take_reads_only_needed_overlay_values() {
            let version = LanceFileVersion::V2_1;
            const N: usize = 100_000;

            let schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new("id", DataType::Int32, true),
                ArrowField::new("val", DataType::Int32, true),
            ]));
            let base = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(0..N as i32)),
                    Arc::new(Int32Array::from_iter_values((0..N as i32).map(|v| v * 10))),
                ],
            )
            .unwrap();
            let write_params = WriteParams {
                max_rows_per_file: N,
                max_rows_per_group: N,
                data_storage_version: Some(version),
                ..Default::default()
            };
            let reader = RecordBatchIterator::new(vec![Ok(base)], schema.clone());
            let dataset = Dataset::write(reader, "memory://", Some(write_params))
                .await
                .unwrap();

            // Overlay `val` over ALL N offsets with incompressible values, so the
            // value column is ~N*4 bytes on disk.
            let values: Vec<i32> = (0..N as u64)
                .map(|i| {
                    let mut x = i;
                    x ^= x >> 33;
                    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
                    x ^= x >> 33;
                    x as i32
                })
                .collect();
            let dataset = commit_overlay(
                dataset,
                "big",
                0,
                &[1],
                OverlayCoverage::dense(bitmap(0..N as u32)),
                vec![Arc::new(Int32Array::from(values.clone())) as ArrayRef],
                version,
            )
            .await;

            let frag = dataset.get_fragment(0).unwrap();
            let val_only = dataset.schema().project_by_ids(&[1], true);

            // Measure only the reads that resolve the take.
            dataset.table_state.object_store.io_stats_incremental();
            let batch = frag.take(&[0, 1], &val_only).await.unwrap();
            let io = dataset.table_state.object_store.io_stats_incremental();

            // The overlay's `val` column alone is N*4 bytes; resolving two adjacent
            // offsets must read only a small fraction of it.
            let full_column_bytes = (N * std::mem::size_of::<i32>()) as u64;
            assert!(
                io.read_bytes > 0 && io.read_bytes < full_column_bytes / 4,
                "take read {} bytes; expected far less than the {}-byte overlay \
                 column (a take must not read the whole value column)",
                io.read_bytes,
                full_column_bytes,
            );

            // ...and it still resolves correctly.
            let val = col(&batch, "val");
            assert_eq!(val.value(0), values[0]);
            assert_eq!(val.value(1), values[1]);
        }

        /// Row-selection pruning: an overlay whose coverage is disjoint from the
        /// requested rows must not be opened at all. Proven by deleting the overlay's
        /// data file — a `take` that misses its coverage still succeeds (the file is
        /// never touched), while a `take` that hits it then fails because the file is
        /// genuinely needed.
        #[rstest]
        #[tokio::test]
        async fn test_take_prunes_overlays_outside_row_selection(
            #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
        ) {
            let dataset = create_base_dataset(version).await;
            // Overlay on fragment 0 (offsets 0..6) covering only offset_in_frag 5.
            let dataset = commit_overlay(
                dataset,
                "miss",
                0,
                &[1],
                OverlayCoverage::dense(bitmap([5])),
                vec![i32_array([Some(5000)])],
                version,
            )
            .await;

            // Delete the overlay's data file: opening it now fails.
            dataset
                .table_state
                .object_store
                .delete(&Path::from("data/miss.lance"))
                .await
                .unwrap();

            let frag = dataset.get_fragment(0).unwrap();
            let val_only = dataset.schema().project_by_ids(&[1], true);

            // A take that misses the overlay's coverage must not open it, so it
            // succeeds and returns base values (val = offset * 10).
            let batch = frag.take(&[0, 1], &val_only).await.unwrap();
            assert_eq!(col(&batch, "val").values(), &[0, 10]);

            // A take that hits the coverage does need the file, so it now fails with
            // a not-found error naming the missing overlay file.
            let err = frag.take(&[5], &val_only).await.unwrap_err();
            let message = format!("{err:?}");
            assert!(
                err.is_not_found() && message.contains("miss.lance"),
                "take hitting the overlay's coverage should fail with a not-found error \
                 for its missing file, got: {message}",
            );
        }

        /// The overlay merge runs before `wrap_with_row_id_and_delete`, so the
        /// `_rowid` system column must coexist with overlay-resolved data columns:
        /// the row ids are unaffected by the merge and the overlay value still wins.
        #[rstest]
        #[tokio::test]
        async fn test_scan_with_row_id_alongside_overlay(
            #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
        ) {
            let dataset = create_base_dataset(version).await;
            let dataset = commit_overlay(
                dataset,
                "rowidov",
                0,
                &[1],
                OverlayCoverage::dense(bitmap([0])),
                vec![i32_array([Some(1000)])],
                version,
            )
            .await;

            let frag = dataset.get_fragment(0).unwrap();
            let batch = frag
                .scan()
                .with_row_id()
                .project(&["id", "val"])
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();
            // Overlay value resolves...
            assert_eq!(col(&batch, "val").values()[0], 1000);
            assert_eq!(&col(&batch, "val").values()[1..], &[10, 20, 30, 40, 50]);
            // ...and the row ids for fragment 0 are the untouched physical offsets.
            let row_ids = batch
                .column(batch.schema().index_of("_rowid").unwrap())
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            assert_eq!(row_ids.values(), &[0, 1, 2, 3, 4, 5]);
        }

        /// When the newest overlay covers every requested offset, an older overlay
        /// in the same plan needs zero values and its value column must not be read
        /// (the empty-input branch of `fetch_overlay_values`). The result still
        /// resolves to the newest overlay.
        #[rstest]
        #[tokio::test]
        async fn test_take_older_overlay_contributes_no_values(
            #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
        ) {
            let dataset = create_base_dataset(version).await;
            // Older covers {1, 4}; newer re-covers {1}. A take of only offset 1
            // routes entirely to the newer overlay, leaving the older one with no
            // values to fetch even though it is part of the field's plan.
            let dataset = commit_overlay(
                dataset,
                "older",
                0,
                &[1],
                OverlayCoverage::dense(bitmap([1, 4])),
                vec![i32_array([Some(111), Some(444)])],
                version,
            )
            .await;
            let dataset = commit_overlay(
                dataset,
                "newer",
                0,
                &[1],
                OverlayCoverage::dense(bitmap([1])),
                vec![i32_array([Some(999)])],
                version,
            )
            .await;

            let frag = dataset.get_fragment(0).unwrap();
            let batch = frag.take(&[1], &full_schema(&dataset)).await.unwrap();
            assert_eq!(col(&batch, "val").values(), &[999]);
        }

        /// A newest overlay whose value is NULL must shadow an older overlay's
        /// non-null value at the same offset — the merge resolves to NULL, it does
        /// not fall back to the older overlay.
        #[rstest]
        #[tokio::test]
        async fn test_take_newest_null_shadows_older(
            #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
        ) {
            let dataset = create_base_dataset(version).await;
            let dataset = commit_overlay(
                dataset,
                "older",
                0,
                &[1],
                OverlayCoverage::dense(bitmap([1])),
                vec![i32_array([Some(111)])],
                version,
            )
            .await;
            let dataset = commit_overlay(
                dataset,
                "newer_null",
                0,
                &[1],
                OverlayCoverage::dense(bitmap([1])),
                vec![i32_array([None])],
                version,
            )
            .await;

            let frag = dataset.get_fragment(0).unwrap();
            let batch = frag.take(&[1], &full_schema(&dataset)).await.unwrap();
            let val = col(&batch, "val");
            assert!(val.is_null(0), "newest NULL must win over older 111");
        }

        /// Newest-wins is resolved independently per field across multiple sparse
        /// overlays: for the same offset, `id` can resolve to one overlay while
        /// `val` resolves to the other, depending on which overlay newly covers
        /// that field at that offset.
        #[rstest]
        #[tokio::test]
        async fn test_take_multi_sparse_per_field_newest_wins(
            #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
        ) {
            let dataset = create_base_dataset(version).await;
            // Older: id covers {3}, val covers {2}.
            let dataset = commit_overlay(
                dataset,
                "older",
                0,
                &[0, 1],
                OverlayCoverage::sparse(vec![bitmap([3]), bitmap([2])]),
                vec![i32_array([Some(7773)]), i32_array([Some(2772)])],
                version,
            )
            .await;
            // Newer: id covers {2}, val covers {3} — the mirror image.
            let dataset = commit_overlay(
                dataset,
                "newer",
                0,
                &[0, 1],
                OverlayCoverage::sparse(vec![bitmap([2]), bitmap([3])]),
                vec![i32_array([Some(9992)]), i32_array([Some(9993)])],
                version,
            )
            .await;

            let frag = dataset.get_fragment(0).unwrap();
            let batch = frag.take(&[2, 3], &full_schema(&dataset)).await.unwrap();
            // id: offset 2 -> newer (9992), offset 3 -> older (7773).
            assert_eq!(col(&batch, "id").values(), &[9992, 7773]);
            // val: offset 2 -> older (2772), offset 3 -> newer (9993).
            assert_eq!(col(&batch, "val").values(), &[2772, 9993]);
        }

        /// A fragment with an overlay plan, but a take that touches only uncovered
        /// offsets, must fall entirely through to the base values (the
        /// `!routing.any_overlay` early-return with a plan present).
        #[rstest]
        #[tokio::test]
        async fn test_take_plan_present_all_offsets_uncovered(
            #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
        ) {
            let dataset = create_base_dataset(version).await;
            let dataset = commit_overlay(
                dataset,
                "ov",
                0,
                &[1],
                OverlayCoverage::dense(bitmap([1, 4])),
                vec![i32_array([Some(111), Some(444)])],
                version,
            )
            .await;

            let frag = dataset.get_fragment(0).unwrap();
            // None of {0, 2, 5} are covered: the plan exists but contributes nothing.
            let batch = frag.take(&[0, 2, 5], &full_schema(&dataset)).await.unwrap();
            assert_eq!(col(&batch, "val").values(), &[0, 20, 50]);
            assert_eq!(col(&batch, "id").values(), &[0, 2, 5]);
        }

        /// A dataset-level `take` spanning multiple fragments, each with its own
        /// overlay, routes every global row index to the right fragment's overlay.
        #[rstest]
        #[tokio::test]
        async fn test_dataset_take_multi_fragment_overlays(
            #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
        ) {
            let dataset = create_base_dataset(version).await;
            let dataset = commit_overlay(
                dataset,
                "frag0",
                0,
                &[1],
                OverlayCoverage::dense(bitmap([0])),
                vec![i32_array([Some(1000)])],
                version,
            )
            .await;
            let dataset = commit_overlay(
                dataset,
                "frag1",
                1,
                &[1],
                OverlayCoverage::dense(bitmap([0])),
                vec![i32_array([Some(6000)])],
                version,
            )
            .await;

            // Global rows 0 and 6 are the overlaid offset-0 rows of fragments 0 and
            // 1; rows 1 and 7 fall through to base.
            let batch = dataset
                .take(&[0, 1, 6, 7], full_schema(&dataset))
                .await
                .unwrap();
            assert_eq!(col(&batch, "id").values(), &[0, 1, 6, 7]);
            assert_eq!(col(&batch, "val").values(), &[1000, 10, 6000, 70]);
        }

        /// A scan whose read splits into multiple batches must slice
        /// `offsets_in_frag` per batch correctly — the running `rows_seen`
        /// accumulator in `merge_overlays` gives each batch its start. Every other
        /// scan test uses single-batch fragments, so this is the only guard for the
        /// cross-batch (`start > 0`) path.
        #[rstest]
        #[case::full(None)]
        #[case::ranges(Some(vec![1..6, 8..10]))]
        #[tokio::test]
        async fn test_scan_multi_batch_overlay_slicing(
            #[values(
                LanceFileVersion::V2_0,
                LanceFileVersion::V2_1,
                LanceFileVersion::V2_2,
                LanceFileVersion::V2_3
            )]
            version: LanceFileVersion,
            #[case] ranges: Option<Vec<std::ops::Range<u64>>>,
        ) {
            // One fragment of 10 rows so the read can be chunked below.
            let schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new("id", DataType::Int32, true),
                ArrowField::new("val", DataType::Int32, true),
            ]));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(0..10)),
                    Arc::new(Int32Array::from_iter_values((0..10).map(|v| v * 10))),
                ],
            )
            .unwrap();
            let write_params = WriteParams {
                max_rows_per_file: 100,
                max_rows_per_group: 100,
                data_storage_version: Some(version),
                ..Default::default()
            };
            let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
            let dataset = Dataset::write(reader, "memory://", Some(write_params))
                .await
                .unwrap();

            // Overlay one offset in each batch that batch_size 4 produces (batches
            // [0,4), [4,8), [8,10)): offsets 1, 5, 9 with distinct values. A wrong
            // per-batch slice would misalign these.
            let dataset = commit_overlay(
                dataset,
                "multibatch",
                0,
                &[1],
                OverlayCoverage::dense(bitmap([1, 5, 9])),
                vec![i32_array([Some(111), Some(555), Some(999)])],
                version,
            )
            .await;

            let frag = dataset.get_fragment(0).unwrap();
            let expected_rows: Vec<i32> = ranges.as_ref().map_or_else(
                || (0..10).collect(),
                |ranges| {
                    ranges
                        .iter()
                        .flat_map(|r| r.clone().map(|i| i as i32))
                        .collect()
                },
            );
            let batches: Vec<RecordBatch> = if let Some(ranges) = ranges {
                frag.open(&full_schema(&dataset), FragReadConfig::default())
                    .await
                    .unwrap()
                    .read_ranges(ranges.into(), 4)
                    .await
                    .unwrap()
                    .buffered(2)
                    .try_collect()
                    .await
                    .unwrap()
            } else {
                let mut scanner = frag.scan();
                scanner.batch_size(4).project(&["val"]).unwrap();
                scanner
                    .try_into_stream()
                    .await
                    .unwrap()
                    .try_collect()
                    .await
                    .unwrap()
            };
            // Guard the guard: the read must actually span multiple batches, else
            // this would not exercise the cross-batch slice at all.
            assert!(
                batches.len() > 1,
                "expected a multi-batch scan, got {} batch(es)",
                batches.len()
            );

            let merged =
                arrow_select::concat::concat_batches(&batches[0].schema(), &batches).unwrap();
            let expected: Vec<i32> = expected_rows
                .into_iter()
                .map(|i| match i {
                    1 => 111,
                    5 => 555,
                    9 => 999,
                    other => other * 10,
                })
                .collect();
            assert_eq!(col(&merged, "val").values(), &expected);
        }

        /// An empty selection must not trip over the overlay path: the plan exists
        /// but there are no offsets to route, so the result is an empty batch.
        #[rstest]
        #[tokio::test]
        async fn test_take_empty_selection(
            #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
        ) {
            let dataset = create_base_dataset(version).await;
            let dataset = commit_overlay(
                dataset,
                "ov",
                0,
                &[1],
                OverlayCoverage::dense(bitmap([1, 4])),
                vec![i32_array([Some(111), Some(444)])],
                version,
            )
            .await;

            let frag = dataset.get_fragment(0).unwrap();
            let batch = frag.take(&[], &full_schema(&dataset)).await.unwrap();
            assert_eq!(batch.num_rows(), 0);
        }

        /// Overlays resolve variable-width columns end-to-end, not just fixed-width
        /// ones: the value column is fetched through the real file reader (a
        /// different value-pushdown path than the fixed-width case) and assembled.
        #[rstest]
        #[tokio::test]
        async fn test_string_overlay_end_to_end(
            #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
        ) {
            use arrow_array::StringArray;

            let schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new("id", DataType::Int32, true),
                ArrowField::new("name", DataType::Utf8, true),
            ]));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(0..6)),
                    Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e", "f"])),
                ],
            )
            .unwrap();
            let write_params = WriteParams {
                max_rows_per_file: 6,
                max_rows_per_group: 6,
                data_storage_version: Some(version),
                ..Default::default()
            };
            let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
            let dataset = Dataset::write(reader, "memory://", Some(write_params))
                .await
                .unwrap();

            // Overlay `name` at offsets {1, 4}, one of the values NULL.
            let dataset = commit_overlay(
                dataset,
                "strov",
                0,
                &[1],
                OverlayCoverage::dense(bitmap([1, 4])),
                vec![Arc::new(StringArray::from(vec![Some("B"), None])) as ArrayRef],
                version,
            )
            .await;

            let frag = dataset.get_fragment(0).unwrap();
            let batch = frag.take(&[0, 1, 4], &full_schema(&dataset)).await.unwrap();
            let name = batch
                .column(batch.schema().index_of("name").unwrap())
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            assert_eq!(name.value(0), "a"); // falls through to base
            assert_eq!(name.value(1), "B"); // overlay value
            assert!(name.is_null(2)); // overlay NULL wins
        }

        /// Projection pruning must do NO IO to overlay files whose fields are not
        /// projected. Proven the same way as row-selection pruning: delete the
        /// overlay's data file, then read projecting only the *unrelated* `id`
        /// column — it must succeed (the `val` overlay file is never opened), while
        /// projecting the overlaid `val` column then fails because its file is gone.
        #[rstest]
        #[tokio::test]
        async fn test_projection_prunes_overlay_files_no_io(
            #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
        ) {
            let dataset = create_base_dataset(version).await;
            // Overlay covers `val` (field 1) only.
            let dataset = commit_overlay(
                dataset,
                "valov",
                0,
                &[1],
                OverlayCoverage::dense(bitmap([0, 1])),
                vec![i32_array([Some(1000), Some(1010)])],
                version,
            )
            .await;

            // Delete the overlay's data file: opening it now fails.
            dataset
                .table_state
                .object_store
                .delete(&Path::from("data/valov.lance"))
                .await
                .unwrap();

            let frag = dataset.get_fragment(0).unwrap();
            let id_only = dataset.schema().project_by_ids(&[0], true);
            let val_only = dataset.schema().project_by_ids(&[1], true);

            // Projecting only `id` must not open the `val` overlay file, so it
            // succeeds and returns untouched base values.
            let batch = frag.take(&[0, 1], &id_only).await.unwrap();
            assert_eq!(col(&batch, "id").values(), &[0, 1]);
            // A scan projecting only `id` must likewise never touch the file.
            let batch = frag
                .scan()
                .project(&["id"])
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();
            assert_eq!(col(&batch, "id").values(), &[0, 1, 2, 3, 4, 5]);

            // Projecting the overlaid `val` column does need the file, so it fails
            // with a not-found error naming the missing overlay file.
            let err = frag.take(&[0], &val_only).await.unwrap_err();
            let message = format!("{err:?}");
            assert!(
                err.is_not_found() && message.contains("valov.lance"),
                "projecting the overlaid column should fail with a not-found error \
                 for its missing file, got: {message}",
            );
        }

        /// A top-level struct column resolves through overlays: the overlay stores
        /// the struct's leaf columns (under V2_1 those are the only ids in
        /// `data_file.fields`), and `plan_overlays` maps them back to the top-level
        /// struct so the whole value is fetched and replaced as a unit.
        #[rstest]
        #[tokio::test]
        async fn test_struct_overlay_end_to_end(
            #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
        ) {
            let struct_fields = Fields::from(vec![
                ArrowField::new("x", DataType::Int32, true),
                ArrowField::new("y", DataType::Int32, true),
            ]);
            let schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new("id", DataType::Int32, true),
                ArrowField::new("info", DataType::Struct(struct_fields.clone()), true),
            ]));
            let info = Arc::new(StructArray::new(
                struct_fields.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(0..6)),
                    Arc::new(Int32Array::from_iter_values((0..6).map(|v| v * 100))),
                ],
                None,
            ));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from_iter_values(0..6)), info],
            )
            .unwrap();
            let write_params = WriteParams {
                max_rows_per_file: 6,
                max_rows_per_group: 6,
                data_storage_version: Some(version),
                ..Default::default()
            };
            let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
            let dataset = Dataset::write(reader, "memory://", Some(write_params))
                .await
                .unwrap();

            // Overlay the whole `info` struct (top-level field id 1) at offset 2.
            let overlay_info = Arc::new(StructArray::new(
                struct_fields,
                vec![
                    Arc::new(Int32Array::from(vec![777])),
                    Arc::new(Int32Array::from(vec![888])),
                ],
                None,
            )) as ArrayRef;
            let dataset = commit_overlay(
                dataset,
                "structov",
                0,
                &[1],
                OverlayCoverage::dense(bitmap([2])),
                vec![overlay_info],
                version,
            )
            .await;

            let frag = dataset.get_fragment(0).unwrap();
            let batch = frag.take(&[1, 2], &full_schema(&dataset)).await.unwrap();
            let info = batch
                .column(batch.schema().index_of("info").unwrap())
                .as_any()
                .downcast_ref::<StructArray>()
                .unwrap();
            let x = info
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let y = info
                .column(1)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            // Offset 1 falls through to base {1, 100}; offset 2 takes the overlay.
            assert_eq!(x.values(), &[1, 777]);
            assert_eq!(y.values(), &[100, 888]);
        }

        /// A struct overlay replaces the struct cell itself, including its validity,
        /// rather than only replacing the child values.
        #[tokio::test]
        async fn test_nullable_struct_overlay_end_to_end() {
            let version = LanceFileVersion::V2_1;
            let struct_fields = Fields::from(vec![ArrowField::new("x", DataType::Int32, true)]);
            let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "info",
                DataType::Struct(struct_fields.clone()),
                true,
            )]));
            let base_info = Arc::new(StructArray::new(
                struct_fields.clone(),
                vec![Arc::new(Int32Array::from(vec![1, 2]))],
                Some(NullBuffer::from(vec![false, true])),
            ));
            let batch = RecordBatch::try_new(schema.clone(), vec![base_info]).unwrap();
            let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
            let dataset = Dataset::write(
                reader,
                "memory://",
                Some(WriteParams {
                    data_storage_version: Some(version),
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

            let overlay_info = Arc::new(StructArray::new(
                struct_fields,
                vec![Arc::new(Int32Array::from(vec![10, 20]))],
                Some(NullBuffer::from(vec![true, false])),
            )) as ArrayRef;
            let dataset = commit_overlay(
                dataset,
                "nullable_struct",
                0,
                &[0],
                OverlayCoverage::dense(bitmap([0, 1])),
                vec![overlay_info],
                version,
            )
            .await;

            let batch = dataset.scan().try_into_batch().await.unwrap();
            let info = struct_col(&batch, "info");
            assert_eq!(i32_child(info, 0).values(), &[10, 20]);
            assert!(!info.is_null(0));
            assert!(info.is_null(1));
        }

        /// A top-level list column resolves through overlays the same way — the
        /// overlay's leaf (item) id maps back to the top-level list, and the whole
        /// list value at a covered offset is replaced.
        #[rstest]
        #[tokio::test]
        async fn test_list_overlay_end_to_end(
            #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
        ) {
            use arrow_array::ListArray;
            use arrow_array::types::Int32Type;

            let item = Arc::new(ArrowField::new("item", DataType::Int32, true));
            let schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new("id", DataType::Int32, true),
                ArrowField::new("tags", DataType::List(item.clone()), true),
            ]));
            let base_tags = ListArray::from_iter_primitive::<Int32Type, _, _>(
                (0..6i32).map(|i| Some(vec![Some(i), Some(i * 10)])),
            );
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(0..6)),
                    Arc::new(base_tags),
                ],
            )
            .unwrap();
            let write_params = WriteParams {
                max_rows_per_file: 6,
                max_rows_per_group: 6,
                data_storage_version: Some(version),
                ..Default::default()
            };
            let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
            let dataset = Dataset::write(reader, "memory://", Some(write_params))
                .await
                .unwrap();

            // Overlay `tags` (top-level field id 1) at offset 2 with a new list.
            let overlay_tags =
                ListArray::from_iter_primitive::<Int32Type, _, _>(std::iter::once(Some(vec![
                    Some(77),
                    Some(88),
                    Some(99),
                ])));
            let dataset = commit_overlay(
                dataset,
                "listov",
                0,
                &[1],
                OverlayCoverage::dense(bitmap([2])),
                vec![Arc::new(overlay_tags) as ArrayRef],
                version,
            )
            .await;

            let frag = dataset.get_fragment(0).unwrap();
            let batch = frag.take(&[1, 2], &full_schema(&dataset)).await.unwrap();
            let tags = batch
                .column(batch.schema().index_of("tags").unwrap())
                .as_any()
                .downcast_ref::<ListArray>()
                .unwrap();
            let row1 = tags.value(0);
            let row1 = row1.as_any().downcast_ref::<Int32Array>().unwrap();
            let row2 = tags.value(1);
            let row2 = row2.as_any().downcast_ref::<Int32Array>().unwrap();
            // Offset 1 falls through to base [1, 10]; offset 2 takes the overlay.
            assert_eq!(row1.values(), &[1, 10]);
            assert_eq!(row2.values(), &[77, 88, 99]);
        }

        /// A top-level Map column resolves as a single atomic field even though its
        /// value spans two leaves (key and value): both leaf ids map back to the one
        /// Map atomic field, and the whole map value at a covered offset is replaced.
        /// Maps require
        /// the 2.2+ file format, so this runs only at V2_2 (unlike the V2_0/V2_1
        /// parametrized tests).
        #[tokio::test]
        async fn test_map_overlay_end_to_end() {
            use arrow_array::MapArray;
            use arrow_array::builder::{Int32Builder, MapBuilder};

            let version = LanceFileVersion::V2_2;

            // Base row i holds the single entry {i: i * 10}.
            let mut builder = MapBuilder::new(None, Int32Builder::new(), Int32Builder::new());
            for i in 0..6i32 {
                builder.keys().append_value(i);
                builder.values().append_value(i * 10);
                builder.append(true).unwrap();
            }
            let base_attrs = builder.finish();
            let schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new("id", DataType::Int32, true),
                ArrowField::new("attrs", base_attrs.data_type().clone(), true),
            ]));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(0..6)),
                    Arc::new(base_attrs),
                ],
            )
            .unwrap();
            let write_params = WriteParams {
                max_rows_per_file: 6,
                max_rows_per_group: 6,
                data_storage_version: Some(version),
                ..Default::default()
            };
            let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
            let dataset = Dataset::write(reader, "memory://", Some(write_params))
                .await
                .unwrap();

            // Overlay `attrs` (top-level field id 1) at offset 2 with a two-entry map.
            let mut ov = MapBuilder::new(None, Int32Builder::new(), Int32Builder::new());
            ov.keys().append_value(7);
            ov.values().append_value(77);
            ov.keys().append_value(8);
            ov.values().append_value(88);
            ov.append(true).unwrap();
            let overlay_attrs = ov.finish();
            let dataset = commit_overlay(
                dataset,
                "mapov",
                0,
                &[1],
                OverlayCoverage::dense(bitmap([2])),
                vec![Arc::new(overlay_attrs) as ArrayRef],
                version,
            )
            .await;

            let frag = dataset.get_fragment(0).unwrap();
            let batch = frag.take(&[1, 2], &full_schema(&dataset)).await.unwrap();
            let attrs = batch
                .column(batch.schema().index_of("attrs").unwrap())
                .as_any()
                .downcast_ref::<MapArray>()
                .unwrap();

            let entries = |i: usize| -> (Vec<i32>, Vec<i32>) {
                let row = attrs.value(i);
                let keys = row.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                let vals = row.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
                (keys.values().to_vec(), vals.values().to_vec())
            };
            // Offset 1 falls through to the base entry {1: 10}; offset 2 takes the
            // overlay map {7: 77, 8: 88}.
            assert_eq!(entries(0), (vec![1], vec![10]));
            assert_eq!(entries(1), (vec![7, 8], vec![77, 88]));
        }

        /// Base `id` + a struct `s { a, b }` (6 rows). Field ids: s=1, a=2, b=3.
        async fn create_struct_dataset(version: LanceFileVersion) -> (Dataset, Fields) {
            let s_fields = Fields::from(vec![
                ArrowField::new("a", DataType::Int32, true),
                ArrowField::new("b", DataType::Int32, true),
            ]);
            let schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new("id", DataType::Int32, true),
                ArrowField::new("s", DataType::Struct(s_fields.clone()), true),
            ]));
            let s = Arc::new(StructArray::new(
                s_fields.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(0..6)),
                    Arc::new(Int32Array::from_iter_values((0..6).map(|v| v * 100))),
                ],
                None,
            ));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from_iter_values(0..6)), s],
            )
            .unwrap();
            let write_params = WriteParams {
                max_rows_per_file: 6,
                max_rows_per_group: 6,
                data_storage_version: Some(version),
                ..Default::default()
            };
            let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
            let dataset = Dataset::write(reader, "memory://", Some(write_params))
                .await
                .unwrap();
            (dataset, s_fields)
        }

        fn struct_col<'a>(batch: &'a RecordBatch, name: &str) -> &'a StructArray {
            batch
                .column(batch.schema().index_of(name).unwrap())
                .as_any()
                .downcast_ref::<StructArray>()
                .unwrap()
        }

        fn i32_child(s: &StructArray, i: usize) -> Int32Array {
            s.column(i)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .clone()
        }

        /// The reviewer's core case (r3553495147): an overlay stores only sub-field
        /// `s.a`, but the read projects the whole struct `s`. The overlay must splice
        /// into `a` and leave `b` untouched (previously this panicked because the merge
        /// fetched the whole `s` from an overlay file holding only `a`).
        #[rstest]
        #[tokio::test]
        async fn test_overlay_subfield_projecting_parent_struct(
            #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
        ) {
            let (dataset, _) = create_struct_dataset(version).await;
            // Overlay ONLY `s.a` (field id 2) at offset 2.
            let a_only = Fields::from(vec![ArrowField::new("a", DataType::Int32, true)]);
            let overlay = Arc::new(StructArray::new(
                a_only,
                vec![Arc::new(Int32Array::from(vec![777]))],
                None,
            )) as ArrayRef;
            let dataset = commit_overlay(
                dataset,
                "aov",
                0,
                &[2],
                OverlayCoverage::dense(bitmap([2])),
                vec![overlay],
                version,
            )
            .await;

            let frag = dataset.get_fragment(0).unwrap();
            let batch = frag.take(&[1, 2], &full_schema(&dataset)).await.unwrap();
            let s = struct_col(&batch, "s");
            // a: offset 1 base (1), offset 2 overlaid (777).
            assert_eq!(i32_child(s, 0).values(), &[1, 777]);
            // b: untouched base (100, 200).
            assert_eq!(i32_child(s, 1).values(), &[100, 200]);
        }

        /// An overlay on a non-projected sibling leaf contributes shared struct
        /// validity, but its child value remains pruned from the result.
        #[rstest]
        #[tokio::test]
        async fn test_overlay_nonprojected_sibling_value_pruned(
            #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
        ) {
            let (dataset, _) = create_struct_dataset(version).await;
            let b_only = Fields::from(vec![ArrowField::new("b", DataType::Int32, true)]);
            let overlay = Arc::new(StructArray::new(
                b_only,
                vec![Arc::new(Int32Array::from(vec![888]))],
                None,
            )) as ArrayRef;
            let dataset = commit_overlay(
                dataset,
                "bov",
                0,
                &[3],
                OverlayCoverage::dense(bitmap([2])),
                vec![overlay],
                version,
            )
            .await;

            let frag = dataset.get_fragment(0).unwrap();
            let a_only = dataset.schema().project_by_ids(&[2], true);
            let batch = frag.take(&[1, 2], &a_only).await.unwrap();
            let s = struct_col(&batch, "s");
            // Only `a` is projected, so the sibling value remains absent and `a`
            // retains its unchanged base values.
            assert_eq!(s.num_columns(), 1);
            assert_eq!(i32_child(s, 0).values(), &[1, 2]);
        }

        /// Two overlays target different sub-fields of the same struct, and a third
        /// re-overlays `s.a`. Each leaf resolves independently and newest wins on `a`.
        #[rstest]
        #[tokio::test]
        async fn test_overlay_multiple_subfields_newest_wins(
            #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
        ) {
            let (dataset, _) = create_struct_dataset(version).await;
            let a_field = Fields::from(vec![ArrowField::new("a", DataType::Int32, true)]);
            let b_field = Fields::from(vec![ArrowField::new("b", DataType::Int32, true)]);
            // Older: a := 700 at offset 2.
            let dataset = commit_overlay(
                dataset,
                "a_old",
                0,
                &[2],
                OverlayCoverage::dense(bitmap([2])),
                vec![Arc::new(StructArray::new(
                    a_field.clone(),
                    vec![Arc::new(Int32Array::from(vec![700]))],
                    None,
                )) as ArrayRef],
                version,
            )
            .await;
            // b := 800 at offset 2.
            let dataset = commit_overlay(
                dataset,
                "b_ov",
                0,
                &[3],
                OverlayCoverage::dense(bitmap([2])),
                vec![Arc::new(StructArray::new(
                    b_field,
                    vec![Arc::new(Int32Array::from(vec![800]))],
                    None,
                )) as ArrayRef],
                version,
            )
            .await;
            // Newest: a := 999 at offset 2 (shadows the older `a` overlay).
            let dataset = commit_overlay(
                dataset,
                "a_new",
                0,
                &[2],
                OverlayCoverage::dense(bitmap([2])),
                vec![Arc::new(StructArray::new(
                    a_field,
                    vec![Arc::new(Int32Array::from(vec![999]))],
                    None,
                )) as ArrayRef],
                version,
            )
            .await;

            let frag = dataset.get_fragment(0).unwrap();
            let batch = frag.take(&[2], &full_schema(&dataset)).await.unwrap();
            let s = struct_col(&batch, "s");
            assert_eq!(i32_child(s, 0).values(), &[999]); // newest `a` wins
            assert_eq!(i32_child(s, 1).values(), &[800]); // `b` from its own overlay
        }

        /// Struct validity shared by independently overlaid children follows the
        /// same newest-overlay precedence as values.
        #[tokio::test]
        async fn test_struct_validity_uses_newest_child_overlay() {
            let version = LanceFileVersion::V2_1;
            let (dataset, _) = create_struct_dataset(version).await;
            let b_field = Fields::from(vec![ArrowField::new("b", DataType::Int32, true)]);
            let dataset = commit_overlay(
                dataset,
                "b_null",
                0,
                &[3],
                OverlayCoverage::dense(bitmap([2])),
                vec![Arc::new(StructArray::new(
                    b_field,
                    vec![Arc::new(Int32Array::from(vec![800]))],
                    Some(NullBuffer::from(vec![false])),
                )) as ArrayRef],
                version,
            )
            .await;

            let a_field = Fields::from(vec![ArrowField::new("a", DataType::Int32, true)]);
            let dataset = commit_overlay(
                dataset,
                "a_valid",
                0,
                &[2],
                OverlayCoverage::dense(bitmap([2])),
                vec![Arc::new(StructArray::new(
                    a_field,
                    vec![Arc::new(Int32Array::from(vec![999]))],
                    Some(NullBuffer::from(vec![true])),
                )) as ArrayRef],
                version,
            )
            .await;

            let frag = dataset.get_fragment(0).unwrap();
            let batch = frag.take(&[2], &full_schema(&dataset)).await.unwrap();
            let s = struct_col(&batch, "s");
            assert_eq!(i32_child(s, 0).values(), &[999]);
            assert!(s.is_valid(0));

            // Projecting only the older `b` value must still consult the newer
            // `a` overlay for their shared parent validity.
            let b_only = dataset.schema().project_by_ids(&[3], true);
            let batch = frag.take(&[2], &b_only).await.unwrap();
            let s = struct_col(&batch, "s");
            assert!(s.is_valid(0));
        }

        /// Three levels of nesting: `outer { middle { a, b } }`. An overlay on the
        /// deep leaf `outer.middle.a` splices correctly when the whole `outer` is read.
        #[rstest]
        #[tokio::test]
        async fn test_overlay_deeply_nested_subfield(
            #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
        ) {
            let mid_fields = Fields::from(vec![
                ArrowField::new("a", DataType::Int32, true),
                ArrowField::new("b", DataType::Int32, true),
            ]);
            let outer_fields = Fields::from(vec![ArrowField::new(
                "middle",
                DataType::Struct(mid_fields.clone()),
                true,
            )]);
            let schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new("id", DataType::Int32, true),
                ArrowField::new("outer", DataType::Struct(outer_fields.clone()), true),
            ]));
            // Field ids: outer=1, middle=2, a=3, b=4.
            let base_validity = (version == LanceFileVersion::V2_1)
                .then(|| NullBuffer::from(vec![true, true, false, true, true, true]));
            let middle = Arc::new(StructArray::new(
                mid_fields.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(0..6)),
                    Arc::new(Int32Array::from_iter_values((0..6).map(|v| v * 100))),
                ],
                base_validity.clone(),
            ));
            let outer = Arc::new(StructArray::new(outer_fields, vec![middle], base_validity));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from_iter_values(0..6)), outer],
            )
            .unwrap();
            let write_params = WriteParams {
                max_rows_per_file: 6,
                max_rows_per_group: 6,
                data_storage_version: Some(version),
                ..Default::default()
            };
            let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
            let dataset = Dataset::write(reader, "memory://", Some(write_params))
                .await
                .unwrap();

            // Overlay the deep leaf `outer.middle.a` (field id 3) at offset 2.
            let a_leaf = Fields::from(vec![ArrowField::new("a", DataType::Int32, true)]);
            let mid_a = Fields::from(vec![ArrowField::new(
                "middle",
                DataType::Struct(a_leaf.clone()),
                true,
            )]);
            let overlay = Arc::new(StructArray::new(
                mid_a,
                vec![Arc::new(StructArray::new(
                    a_leaf,
                    vec![Arc::new(Int32Array::from(vec![777]))],
                    None,
                ))],
                None,
            )) as ArrayRef;
            let dataset = commit_overlay(
                dataset,
                "deepov",
                0,
                &[3],
                OverlayCoverage::dense(bitmap([2])),
                vec![overlay],
                version,
            )
            .await;

            let frag = dataset.get_fragment(0).unwrap();
            let batch = frag.take(&[1, 2], &full_schema(&dataset)).await.unwrap();
            let outer = struct_col(&batch, "outer");
            let middle = outer
                .column(0)
                .as_any()
                .downcast_ref::<StructArray>()
                .unwrap();
            // a: offset 1 base (1), offset 2 overlaid (777); b untouched.
            assert_eq!(i32_child(middle, 0).values(), &[1, 777]);
            assert_eq!(i32_child(middle, 1).values(), &[100, 200]);
            if version == LanceFileVersion::V2_1 {
                assert!(outer.is_valid(1));
                assert!(middle.is_valid(1));
            }

            // Projecting the *intermediate* struct `outer.middle` (field id 2) while
            // the overlay targets a deeper field (id 3) must still apply: the
            // overlay's leaf id falls inside the projected subtree, so it maps to a
            // projected atomic field. (This is the case wjones127/westonpace flagged where a
            // top-level-only mapping would miss the overlay.)
            let middle_only = dataset.schema().project_by_ids(&[2], true);
            let batch = frag.take(&[2], &middle_only).await.unwrap();
            let middle = struct_col(&batch, "outer")
                .column(0)
                .as_any()
                .downcast_ref::<StructArray>()
                .unwrap();
            assert_eq!(i32_child(middle, 0).values(), &[777]);
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_fragment_scan(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let dataset = create_dataset(test_uri, data_storage_version).await;
        let fragment = &dataset.get_fragments()[2];
        let mut scanner = fragment.scan();
        let batches = scanner
            .with_row_id()
            .filter(" i < 105")
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        if data_storage_version == LanceFileVersion::Legacy {
            assert_eq!(batches.len(), 3);

            assert_eq!(
                batches[0].column_by_name("i").unwrap().as_ref(),
                &Int32Array::from_iter_values(80..90)
            );
            assert_eq!(
                batches[1].column_by_name("i").unwrap().as_ref(),
                &Int32Array::from_iter_values(90..100)
            );
            assert_eq!(
                batches[2].column_by_name("i").unwrap().as_ref(),
                &Int32Array::from_iter_values(100..105)
            );
        } else {
            assert_eq!(batches.len(), 1);

            assert_eq!(
                batches[0].column_by_name("i").unwrap().as_ref(),
                &Int32Array::from_iter_values(80..105)
            )
        }
    }

    #[tokio::test]
    async fn test_fragment_scan_v2() {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let dataset = create_dataset_v2(test_uri).await;
        let fragment = &dataset.get_fragments()[2];
        let mut scanner = fragment.scan();
        let batches = scanner
            .with_row_id()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(batches.len(), 1);

        assert_eq!(
            batches[0].column_by_name("i").unwrap().as_ref(),
            &Int32Array::from_iter_values(80..120)
        );

        let mut scanner = fragment.scan();
        let batches = scanner
            .with_row_id()
            .batch_size(20)
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(batches.len(), 2);

        assert_eq!(
            batches[0].column_by_name("i").unwrap().as_ref(),
            &Int32Array::from_iter_values(80..100)
        );
        assert_eq!(
            batches[1].column_by_name("i").unwrap().as_ref(),
            &Int32Array::from_iter_values(100..120)
        );
    }

    /// The dataset declares `payload` non-nullable, so the blank restored for the
    /// deleted row cannot be a null no matter how the update stream marks the field.
    /// `test_write_schema_controls_blank_nullability` in `updater.rs` pins the value
    /// itself; what matters here is that the rewritten data file still accounts for
    /// every physical row, which is what `validate` checks.
    #[tokio::test]
    async fn test_update_columns_respects_dataset_nullability() {
        let dataset_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("payload", DataType::Binary, false),
        ]));
        let batch = RecordBatch::try_new(
            dataset_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(BinaryArray::from(vec![&b"a"[..], &b"b"[..], &b"c"[..]])),
            ],
        )
        .unwrap();

        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], dataset_schema),
            "memory://",
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset.delete("id = 2").await.unwrap();
        let dataset = Arc::new(dataset);

        // Update streams commonly carry nullable Arrow fields even when the dataset
        // field is non-nullable.
        let right_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("payload", DataType::Binary, true),
        ]));
        let right = RecordBatch::try_new(
            right_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 3])),
                Arc::new(BinaryArray::from(vec![&b"A"[..], &b"C"[..]])),
            ],
        )
        .unwrap();

        let mut fragment = dataset.get_fragments().pop().unwrap();
        let (updated, _) = fragment
            .update_columns(
                RecordBatchIterator::new([Ok(right)], right_schema),
                "id",
                "id",
            )
            .await
            .unwrap();

        // `payload` moved to a new file, leaving a tombstone behind in the old one.
        let layout = updated
            .files
            .iter()
            .map(|file| file.fields.as_ref().to_vec())
            .collect::<Vec<_>>();
        assert_eq!(layout, vec![vec![0, -2], vec![1]]);
        FileFragment::new(dataset, updated)
            .validate()
            .await
            .unwrap();
    }

    #[rstest]
    #[case::empty(&[])]
    #[case::contiguous(&[0, 1])]
    #[case::sorted(&[0, 1, 1])]
    #[case::unsorted(&[1, 0, 1])]
    #[tokio::test]
    async fn test_take_empty_projection(
        #[case] indices: &[u32],
        #[values(false, true)] enable_stable_row_ids: bool,
        #[values(false, true)] delete_row: bool,
    ) {
        let data = gen_batch()
            .col("i", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(6), lance_datagen::BatchCount::from(1));
        let mut dataset = Dataset::write(
            data,
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 3,
                enable_stable_row_ids,
                data_storage_version: Some(LanceFileVersion::Stable),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), 2);
        if delete_row {
            dataset.delete("i = 1").await.unwrap();
        }
        let fragment = dataset.get_fragment(0).unwrap();
        let projection = dataset.schema().project::<&str>(&[]).unwrap();
        dataset.table_state.object_store.io_stats_incremental();
        let batch = fragment.take(indices, &projection).await.unwrap();
        assert_eq!(batch.num_columns(), 0);
        assert_eq!(batch.num_rows(), indices.len());
        assert!(
            dataset
                .table_state
                .object_store
                .io_stats_incremental()
                .requests
                .iter()
                .all(|request| !request.path.as_ref().contains("/data/"))
        );
        let invalid_offset = if delete_row { 2 } else { 3 };
        let error = fragment
            .take(&[invalid_offset], &projection)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("Invalid read params"), "{error}");
    }

    #[rstest]
    #[case::contiguous(&[0, 1], &[3, 4])]
    #[case::unsorted_duplicates(&[1, 0, 1], &[4, 3, 4])]
    #[tokio::test]
    async fn test_take_system_columns(
        #[case] indices: &[u32],
        #[case] dataset_offsets: &[u64],
        #[values(false, true)] enable_stable_row_ids: bool,
        #[values(
            &["_rowaddr", "i", "_rowid"][..],
            &["_rowid"][..],
            &["_rowaddr"][..],
            &["_rowid", "_rowaddr"][..]
        )]
        columns: &[&str],
    ) {
        let data = gen_batch()
            .col("i", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(6), lance_datagen::BatchCount::from(1));
        let mut dataset = Dataset::write(
            data,
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 3,
                enable_stable_row_ids,
                data_storage_version: Some(LanceFileVersion::Stable),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        // Shift the fragment offsets away from the physical offsets.
        dataset.delete("i = 3").await.unwrap();
        let fragment = dataset.get_fragment(1).unwrap();

        let projection = dataset
            .schema()
            .project_preserve_system_columns(columns)
            .unwrap();
        let batch = fragment.take(indices, &projection).await.unwrap();
        let expected = dataset
            .take(
                dataset_offsets,
                crate::dataset::ProjectionRequest::from_columns(
                    columns.iter().copied(),
                    dataset.schema(),
                ),
            )
            .await
            .unwrap();
        assert_eq!(batch, expected);

        let field_names = batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect::<Vec<_>>();
        assert_eq!(field_names, columns);
        let expected_addrs = indices
            .iter()
            .map(|offset| u64::from(RowAddress::new_from_parts(1, offset + 1)))
            .collect::<Vec<_>>();
        if let Some(addrs) = batch.column_by_name(ROW_ADDR) {
            assert_eq!(addrs.as_primitive::<UInt64Type>().values(), &expected_addrs);
        }

        let row_offset_projection = dataset
            .schema()
            .project_preserve_system_columns(&["i", "_rowoffset"])
            .unwrap();
        let error = fragment
            .take(indices, &row_offset_projection)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    }

    /// A fragment with one data file per column group reads only the files
    /// that hold a requested field: the others are not even opened.
    #[tokio::test]
    async fn test_take_opens_only_files_with_projected_fields() {
        let test_dir = TempStrDir::default();
        let mut dataset = create_dataset_v2(&test_dir).await;
        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![("col1".into(), "i * 2".into())]),
                None,
                None,
            )
            .await
            .unwrap();
        let fragment = dataset.get_fragment(0).unwrap();
        assert_eq!(fragment.metadata().files.len(), 2);
        let added_file = fragment.metadata().files[1].path.clone();

        let projection = dataset.schema().project(&["col1"]).unwrap();
        dataset.table_state.object_store.io_stats_incremental();
        let reader = fragment
            .open(&projection, FragReadConfig::default())
            .await
            .unwrap();
        let rows = reader.take_as_batch(&[1, 5], None).await.unwrap();
        assert_eq!(rows.num_rows(), 2);
        let data_files_read: HashSet<String> = dataset
            .table_state
            .object_store
            .io_stats_incremental()
            .requests
            .iter()
            .filter(|request| request.path.as_ref().contains("/data/"))
            .map(|request| request.path.filename().unwrap().to_string())
            .collect();
        assert_eq!(data_files_read, HashSet::from([added_file]));
    }

    /// Projecting only a child that lives in a later file must still open the
    /// file holding the parent's other leaves: a list takes its offsets and a
    /// struct its validity from whichever leaf a file has.
    #[rstest]
    #[case::take(false)]
    #[case::scan(true)]
    #[tokio::test]
    async fn test_projected_missing_list_child_keeps_offsets(#[case] use_scan: bool) {
        use arrow_array::{ListArray, StructArray};
        use arrow_buffer::OffsetBuffer;

        let old_values = StructArray::new(
            vec![ArrowField::new("a", DataType::Int32, true)].into(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
            None,
        );
        let list = ListArray::new(
            Arc::new(ArrowField::new(
                "item",
                old_values.data_type().clone(),
                true,
            )),
            OffsetBuffer::new(vec![0i32, 2, 2, 3].into()),
            Arc::new(old_values),
            None,
        );
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "c",
            list.data_type().clone(),
            true,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(list)]).unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 2,
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let added = ArrowSchema::new(vec![ArrowField::new(
            "c",
            DataType::List(Arc::new(ArrowField::new(
                "item",
                DataType::Struct(vec![ArrowField::new("b", DataType::Int32, true)].into()),
                true,
            ))),
            true,
        )]);
        dataset
            .add_columns(NewColumnTransform::AllNulls(Arc::new(added)), None, None)
            .await
            .unwrap();

        let projection = dataset.schema().project(&["c.item.b"]).unwrap();
        let result = if use_scan {
            let mut scan = dataset.scan();
            scan.project_with_schema(&projection).unwrap();
            scan.try_into_batch().await.unwrap()
        } else {
            dataset.take(&[0, 1, 2], projection).await.unwrap()
        };
        let list = result["c"].as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(list.value_offsets(), &[0, 2, 2, 3][..]);
        assert_eq!(list.null_count(), 0);
    }

    #[tokio::test]
    async fn test_fragment_update() {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let mut dataset = create_dataset_v2(test_uri).await;

        // Test update with _rowid
        let _ = dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![("col1".into(), "-1".into())]),
                None,
                None,
            )
            .await;
        let mut fragment1 = dataset.get_fragment(0).unwrap();

        let schema1 = Arc::new(ArrowSchema::new(vec![
            ArrowField::new(ROW_ID, DataType::UInt64, false),
            ArrowField::new("col1", DataType::Int64, true),
        ]));
        let update_batch1 = RecordBatch::try_new(
            schema1.clone(),
            vec![
                Arc::new(UInt64Array::from(
                    (0..40).filter(|&v| v != 0 && v != 3).collect::<Vec<_>>(),
                )),
                Arc::new(Int64Array::from(vec![2; 38])),
            ],
        )
        .unwrap();
        let right_stream1: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
            vec![Ok(update_batch1)].into_iter(),
            schema1,
        ));
        let u1 = fragment1
            .update_columns_with_offsets(right_stream1, ROW_ID, ROW_ID)
            .await
            .unwrap();
        assert_eq!(u1.matched_offsets.iter().count(), 38);
        assert!(!u1.matched_offsets.contains(0));
        assert!(!u1.matched_offsets.contains(3));
        assert!(u1.matched_offsets.contains(1));
        assert!(u1.matched_offsets.contains(39));
        let frag_id_1 = u1.fragment.id;
        let matched_1 = u1.matched_offsets;
        let op1 = Operation::Update {
            removed_fragment_ids: vec![],
            updated_fragments: vec![u1.fragment],
            new_fragments: vec![],
            fields_modified: u1.fields_modified,
            mem_wal_dir_id: None,
            compacted_sstables: Vec::new(),
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: Some(UpdateMode::RewriteColumns),
            inserted_rows_filter: None,
            updated_fragment_offsets: Some(UpdatedFragmentOffsets(HashMap::from([(
                frag_id_1, matched_1,
            )]))),
        };
        let mut dataset1 = Dataset::commit(
            test_uri,
            op1,
            Some(dataset.version().version),
            None,
            None,
            Default::default(),
            true,
        )
        .await
        .unwrap();
        assert_eq!(dataset1.get_fragments().len(), 5);
        let scanner1 = dataset1.get_fragment(0).unwrap().scan();
        let batches1 = scanner1
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(batches1.len(), 1);
        let mut expected_col1 = vec![2; 40];
        expected_col1[0] = -1;
        expected_col1[3] = -1;
        assert_eq!(
            batches1[0].column_by_name("col1").unwrap().as_ref(),
            &Int64Array::from(expected_col1)
        );

        // Test update with user specified keys
        let _ = dataset1
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![("col2".into(), "false".into())]),
                None,
                None,
            )
            .await;
        let mut fragment2 = dataset1.get_fragment(0).unwrap();

        let schema2 = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i1", DataType::Int32, true),
            ArrowField::new("col2", DataType::Boolean, true),
            ArrowField::new("col1", DataType::Int64, true),
        ]));
        let update_batch2 = RecordBatch::try_new(
            schema2.clone(),
            vec![
                Arc::new(Int32Array::from(
                    (0..40).filter(|&v| v != 0 && v != 3).collect::<Vec<_>>(),
                )),
                Arc::new(BooleanArray::from(vec![true; 38])),
                Arc::new(Int64Array::from(vec![3; 38])),
            ],
        )
        .unwrap();
        let right_stream2: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
            vec![Ok(update_batch2)].into_iter(),
            schema2,
        ));
        let u2 = fragment2
            .update_columns_with_offsets(right_stream2, "i", "i1")
            .await
            .unwrap();
        assert_eq!(u2.matched_offsets.iter().count(), 38);
        assert!(!u2.matched_offsets.contains(0));
        assert!(!u2.matched_offsets.contains(3));
        let frag_id_2 = u2.fragment.id;
        let matched_2 = u2.matched_offsets;
        let op = Operation::Update {
            removed_fragment_ids: vec![],
            updated_fragments: vec![u2.fragment],
            new_fragments: vec![],
            fields_modified: u2.fields_modified,
            mem_wal_dir_id: None,
            compacted_sstables: Vec::new(),
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: Some(UpdateMode::RewriteColumns),
            inserted_rows_filter: None,
            updated_fragment_offsets: Some(UpdatedFragmentOffsets(HashMap::from([(
                frag_id_2, matched_2,
            )]))),
        };
        let dataset2 = Dataset::commit(
            test_uri,
            op,
            Some(dataset1.version().version),
            None,
            None,
            Default::default(),
            true,
        )
        .await
        .unwrap();
        assert_eq!(dataset2.get_fragments().len(), 5);
        let scanner2 = dataset2.get_fragment(0).unwrap().scan();
        let batches2 = scanner2
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(batches2.len(), 1);

        expected_col1 = vec![3; 40];
        expected_col1[0] = -1;
        expected_col1[3] = -1;
        assert_eq!(
            batches2[0].column_by_name("col1").unwrap().as_ref(),
            &Int64Array::from(expected_col1)
        );
        let mut expected_col2 = vec![true; 40];
        expected_col2[0] = false;
        expected_col2[3] = false;
        assert_eq!(
            batches2[0].column_by_name("col2").unwrap().as_ref(),
            &BooleanArray::from(expected_col2)
        );
    }

    #[tokio::test]
    async fn test_out_of_range() {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        // Creates 400 rows in 10 fragments
        let mut dataset = create_dataset(test_uri, LanceFileVersion::Legacy).await;
        // Delete last 20 rows in first fragment
        dataset.delete("i >= 20").await.unwrap();
        // Last fragment has 20 rows but 40 addressable rows
        let fragment = &dataset.get_fragments()[0];
        assert_eq!(fragment.metadata().num_rows().unwrap(), 20);

        // Test with take_range (all rows addressable)
        for with_row_id in [false, true] {
            let reader = fragment
                .open(
                    fragment.schema(),
                    FragReadConfig::default().with_row_id(with_row_id),
                )
                .await
                .unwrap();
            for valid_range in [0..40, 20..40] {
                reader
                    .take_range(valid_range, 100)
                    .await
                    .unwrap()
                    .buffered(1)
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap();
            }
            for invalid_range in [0..41, 41..42] {
                assert!(reader.take_range(invalid_range, 100).await.is_err());
            }
        }

        // Test with read_range (only non-deleted rows addressable)
        for with_row_id in [false, true] {
            let reader = fragment
                .open(
                    fragment.schema(),
                    FragReadConfig::default().with_row_id(with_row_id),
                )
                .await
                .unwrap();
            for valid_range in [0..20, 0..10, 10..20] {
                reader
                    .read_range(valid_range, 100)
                    .await
                    .unwrap()
                    .buffered(1)
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap();
            }
            for invalid_range in [0..21, 21..22] {
                assert!(reader.read_range(invalid_range, 100).await.is_err());
            }
        }
    }

    #[tokio::test]
    async fn test_rowid_rowaddr_only() {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        // Creates 400 rows in 10 fragments
        let mut dataset = create_dataset(test_uri, LanceFileVersion::Legacy).await;
        // Delete last 20 rows in first fragment
        dataset.delete("i >= 20").await.unwrap();
        // Last fragment has 20 rows but 40 addressable rows
        let fragment = &dataset.get_fragments()[0];
        assert_eq!(fragment.metadata().num_rows().unwrap(), 20);

        // Test with take_range (all rows addressable)
        for (with_row_id, with_row_address) in [(false, true), (true, false), (true, true)] {
            let reader = fragment
                .open(
                    &fragment.schema().project::<&str>(&[]).unwrap(),
                    FragReadConfig::default()
                        .with_row_id(with_row_id)
                        .with_row_address(with_row_address),
                )
                .await
                .unwrap();
            for valid_range in [0..40, 20..40] {
                reader
                    .take_range(valid_range, 100)
                    .await
                    .unwrap()
                    .buffered(1)
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap();
            }
            for invalid_range in [0..41, 41..42] {
                assert!(reader.take_range(invalid_range, 100).await.is_err());
            }
        }

        // Test with read_range (only non-deleted rows addressable)
        for (with_row_id, with_row_address) in [(false, true), (true, false), (true, true)] {
            let reader = fragment
                .open(
                    &fragment.schema().project::<&str>(&[]).unwrap(),
                    FragReadConfig::default()
                        .with_row_id(with_row_id)
                        .with_row_address(with_row_address),
                )
                .await
                .unwrap();
            for valid_range in [0..20, 0..10, 10..20] {
                reader
                    .read_range(valid_range, 100)
                    .await
                    .unwrap()
                    .buffered(1)
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap();
            }
            for invalid_range in [0..21, 21..22] {
                assert!(reader.read_range(invalid_range, 100).await.is_err());
            }
        }
    }

    #[tokio::test]
    async fn test_mixed_payload_and_system_columns_preserve_schema_and_order() {
        let test_dir = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, false),
            ArrowField::new("s", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..23)),
                Arc::new(StringArray::from_iter_values(
                    (0..23).map(|value| format!("s-{value}")),
                )),
            ],
        )
        .unwrap();
        let write_params = WriteParams {
            data_storage_version: Some(LanceFileVersion::Stable),
            enable_stable_row_ids: true,
            max_rows_per_file: 23,
            max_rows_per_group: 7,
            ..Default::default()
        };
        let dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            test_dir.as_ref(),
            Some(write_params),
        )
        .await
        .unwrap();
        let fragment = dataset.get_fragment(0).unwrap();
        let projection = fragment.schema().project(&["s", "i"]).unwrap();
        let reader = fragment
            .open(
                &projection,
                FragReadConfig::default()
                    .with_row_id(true)
                    .with_row_address(true)
                    .with_row_last_updated_at_version(true)
                    .with_row_created_at_version(true),
            )
            .await
            .unwrap();

        let ranges: Arc<[Range<u64>]> = Arc::from([2..6, 10..13]);
        let batches = reader
            .read_ranges(ranges, 3)
            .await
            .unwrap()
            .buffered(3)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 7);
        assert!(
            batches
                .windows(2)
                .all(|pair| pair[0].schema().as_ref() == pair[1].schema().as_ref())
        );
        let expected_fields = [
            "s",
            "i",
            ROW_ID,
            ROW_ADDR,
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

        let selected_offsets = [2_u64, 3, 4, 5, 10, 11, 12];
        let actual_row_ids = batches
            .iter()
            .flat_map(|batch| batch[ROW_ID].as_primitive::<UInt64Type>().values())
            .copied()
            .collect::<Vec<_>>();
        let actual_row_addrs = batches
            .iter()
            .flat_map(|batch| batch[ROW_ADDR].as_primitive::<UInt64Type>().values())
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(actual_row_ids, selected_offsets);
        assert_eq!(actual_row_addrs, selected_offsets);
        for version_column in [
            lance_core::ROW_LAST_UPDATED_AT_VERSION,
            lance_core::ROW_CREATED_AT_VERSION,
        ] {
            assert!(batches.iter().all(|batch| {
                batch[version_column]
                    .as_primitive::<UInt64Type>()
                    .values()
                    .iter()
                    .all(|version| *version == 1)
            }));
        }

        let range_batches = reader
            .read_range(1..8, 3)
            .await
            .unwrap()
            .buffered(3)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(
            range_batches
                .iter()
                .flat_map(|batch| batch[ROW_ID].as_primitive::<UInt64Type>().values())
                .copied()
                .collect::<Vec<_>>(),
            (1..8).collect::<Vec<_>>()
        );
        assert_eq!(range_batches.last().unwrap().num_rows(), 1);

        let empty_projection = fragment.schema().project::<&str>(&[]).unwrap();
        let system_reader = fragment
            .open(
                &empty_projection,
                FragReadConfig::default()
                    .with_row_id(true)
                    .with_row_address(true),
            )
            .await
            .unwrap();
        let system_batches = system_reader
            .read_all(7)
            .await
            .unwrap()
            .buffered(4)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(
            system_batches
                .iter()
                .map(RecordBatch::num_rows)
                .collect::<Vec<_>>(),
            [7, 7, 7, 2]
        );
        assert!(
            system_batches
                .windows(2)
                .all(|pair| Arc::ptr_eq(&pair[0].schema(), &pair[1].schema()))
        );

        let system_ranges: Arc<[Range<u64>]> = Arc::from([2..6, 10..13]);
        let system_range_batches = system_reader
            .read_ranges(system_ranges, 3)
            .await
            .unwrap()
            .buffered(3)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert!(
            system_range_batches
                .windows(2)
                .all(|pair| Arc::ptr_eq(&pair[0].schema(), &pair[1].schema()))
        );
    }

    #[rstest]
    #[case::row_id(true, false, false, LanceFileVersion::V2_1)]
    #[case::row_address(false, true, true, LanceFileVersion::V2_1)]
    #[case::both(true, true, true, LanceFileVersion::V2_1)]
    #[case::stable(true, true, false, LanceFileVersion::Stable)]
    #[tokio::test]
    async fn test_frag_read_config_make_deletions_null(
        #[case] with_row_id: bool,
        #[case] with_row_address: bool,
        #[case] enable_stable_row_ids: bool,
        #[case] data_storage_version: LanceFileVersion,
    ) {
        let batch = arrow_array::record_batch!(
            ("i", Int32, [0, 1, 2, 3, 4, 5, 6, 7]),
            (
                "value",
                Int32,
                [
                    Some(0),
                    None,
                    Some(20),
                    None,
                    Some(40),
                    None,
                    Some(60),
                    None
                ]
            )
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 4,
                data_storage_version: Some(data_storage_version),
                enable_stable_row_ids,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset.delete("i IN (0, 1, 4, 5)").await.unwrap();
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 2);

        for (fragment_index, fragment) in fragments.iter().enumerate() {
            for invalid_config in [
                FragReadConfig {
                    make_deletions_null: true,
                    ..Default::default()
                },
                FragReadConfig::default()
                    .with_make_deletions_null(true)
                    .with_row_created_at_version(true),
            ] {
                let error = fragment
                    .open(dataset.schema(), invalid_config)
                    .await
                    .unwrap_err();
                assert!(matches!(error, Error::InvalidInput { .. }));
                assert!(
                    error
                        .to_string()
                        .contains("make_deletions_null requires with_row_id or with_row_address")
                );
            }

            let config = FragReadConfig::default()
                .with_row_id(with_row_id)
                .with_row_address(with_row_address);
            assert!(!config.make_deletions_null);
            let reader = fragment
                .open(
                    dataset.schema(),
                    config.clone().with_make_deletions_null(true),
                )
                .await
                .unwrap();
            let reads = [
                (reader.read_all(2).await.unwrap(), vec![0, 1, 2, 3]),
                (reader.take_range(0..2, 2).await.unwrap(), vec![0, 1]),
                (reader.take_range(1..3, 2).await.unwrap(), vec![1, 2]),
                (
                    reader
                        .read_ranges(Arc::from([0..1, 2..4]), 2)
                        .await
                        .unwrap(),
                    vec![0, 2, 3],
                ),
            ];
            for (stream, offsets) in reads {
                let batches = stream.buffered(2).try_collect::<Vec<_>>().await.unwrap();
                let actual = concat_batches(&batches[0].schema(), &batches).unwrap();
                let expected_indices = UInt32Array::from(
                    offsets
                        .iter()
                        .map(|offset| fragment_index as u32 * 4 + offset)
                        .collect::<Vec<_>>(),
                );
                let expected =
                    arrow_select::take::take_record_batch(&batch, &expected_indices).unwrap();
                assert_eq!(actual.num_rows(), offsets.len());
                for column in ["i", "value"] {
                    assert_eq!(actual[column].as_ref(), expected[column].as_ref());
                }
                assert_eq!(actual.column_by_name(ROW_ID).is_some(), with_row_id);
                assert_eq!(actual.column_by_name(ROW_ADDR).is_some(), with_row_address);
                for (column, enabled) in [(ROW_ID, with_row_id), (ROW_ADDR, with_row_address)] {
                    if !enabled {
                        continue;
                    }
                    let expected_markers = UInt64Array::from(
                        offsets
                            .iter()
                            .map(|&offset| {
                                (offset >= 2).then(|| {
                                    if column == ROW_ID && enable_stable_row_ids {
                                        fragment_index as u64 * 4 + u64::from(offset)
                                    } else {
                                        u64::from(RowAddress::new_from_parts(
                                            fragment.id() as u32,
                                            offset,
                                        ))
                                    }
                                })
                            })
                            .collect::<Vec<_>>(),
                    );
                    assert_eq!(actual[column].as_ref(), &expected_markers);
                }
            }

            for filtered_config in [
                config,
                FragReadConfig::default(),
                FragReadConfig::default()
                    .with_make_deletions_null(true)
                    .with_make_deletions_null(false),
            ] {
                let filtered = fragment
                    .open(dataset.schema(), filtered_config)
                    .await
                    .unwrap()
                    .read_range_as_batch(0..4)
                    .await
                    .unwrap();
                assert_eq!(filtered.num_rows(), 2);
                let expected = batch.slice(fragment_index * 4 + 2, 2);
                for column in ["i", "value"] {
                    assert_eq!(filtered[column].as_ref(), expected[column].as_ref());
                }
                for column in [ROW_ID, ROW_ADDR] {
                    if let Some(markers) = filtered.column_by_name(column) {
                        assert_eq!(markers.null_count(), 0);
                    }
                }
            }
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_fragment_take_range_deletions(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let mut dataset = create_dataset(test_uri, data_storage_version).await;
        dataset.delete("i >= 0 and i < 15").await.unwrap();

        let fragment = &dataset.get_fragments()[0];
        let read_config = FragReadConfig::default().with_row_id(true);

        if data_storage_version == LanceFileVersion::Legacy {
            let mut reader = fragment
                .open_v1_fragment_reader(dataset.schema(), &read_config)
                .await
                .unwrap();
            reader.with_make_deletions_null();
            // The first batch is entirely deleted, deleted rows will be marked null with null row ids.
            let batch1 = reader
                .read_batch_projected(0, .., dataset.schema())
                .await
                .unwrap();
            assert_eq!(
                batch1.column_by_name(ROW_ID).unwrap().as_ref(),
                &UInt64Array::from_iter(std::iter::repeat_n(None, 10))
            );

            // The second batch is partially deleted, so the deleted rows will be
            // marked null with null row ids.
            let batch2 = reader
                .read_batch_projected(1, .., dataset.schema())
                .await
                .unwrap();
            assert_eq!(
                batch2.column_by_name(ROW_ID).unwrap().as_ref(),
                &UInt64Array::from_iter((10..20).map(|v| if v < 15 { None } else { Some(v) }))
            );

            // The final batch is not deleted, so it will be returned as-is.
            let batch3 = reader
                .read_batch_projected(2, .., dataset.schema())
                .await
                .unwrap();
            assert_eq!(
                batch3.column_by_name(ROW_ID).unwrap().as_ref(),
                &UInt64Array::from_iter_values(20..30)
            );
        } else {
            let mut reader = fragment.open(dataset.schema(), read_config).await.unwrap();
            reader.with_make_deletions_null();
            let to_batches = |range: Range<u32>| {
                let batch_size = range.len() as u32;
                let fut = reader.take_range(range, batch_size);
                async move { fut.await.unwrap().buffered(1).try_collect::<Vec<_>>().await }
            };

            // Since the first batch is all deleted, it will return all nulls row ids.
            let batches = to_batches(0..10).await.unwrap();
            assert_eq!(batches.len(), 1);
            let batch = batches.into_iter().next().unwrap();
            assert_eq!(
                batch.column_by_name(ROW_ID).unwrap().as_ref(),
                &UInt64Array::from_iter(std::iter::repeat_n(None, 10))
            );

            let batches = to_batches(10..20).await.unwrap();
            assert_eq!(batches.len(), 1);
            let batch = batches.into_iter().next().unwrap();
            // The second batch is partially deleted, so the deleted rows will be
            // marked null with null row ids.
            assert_eq!(
                batch.column_by_name(ROW_ID).unwrap().as_ref(),
                &UInt64Array::from_iter((10..20).map(|v| if v < 15 { None } else { Some(v) }))
            );

            // The final batch is not deleted, so it will be returned as-is.
            let batches = to_batches(20..30).await.unwrap();
            assert_eq!(batches.len(), 1);
            let batch = batches.into_iter().next().unwrap();
            assert_eq!(
                batch.column_by_name(ROW_ID).unwrap().as_ref(),
                &UInt64Array::from_iter_values(20..30)
            );
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_range_scan_deletions(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let dataset = create_dataset(test_uri, data_storage_version).await;

        let version = dataset.version().version;

        let check = |cond: &'static str, range: Range<u32>, expected: Vec<i32>| async {
            let mut dataset = dataset.checkout_version(version).await.unwrap();
            dataset.restore().await.unwrap();
            dataset.delete(cond).await.unwrap();

            let fragment = &dataset.get_fragments()[0];
            let reader = fragment
                .open(
                    dataset.schema(),
                    FragReadConfig::default().with_row_id(true),
                )
                .await
                .unwrap();

            // Using batch_size=20 here.  If we use batch_size=range.len() we get
            // multiple batches because we might have to read from a larger range
            // to satisfy the request
            let mut stream = reader.read_range(range, 20).await.unwrap();
            let mut batches = Vec::new();
            while let Some(next) = stream.next().await {
                batches.push(next.await.unwrap());
            }
            let schema = Arc::new(dataset.schema().into());
            let batch = arrow_select::concat::concat_batches(&schema, batches.iter()).unwrap();

            assert_eq!(batch.num_rows(), expected.len());
            assert_eq!(
                batch.column_by_name("i").unwrap().as_ref(),
                &Int32Array::from(expected)
            );
        };
        // Deleting from the start
        check("i < 5", 0..2, vec![5, 6]).await;
        check("i < 5", 0..15, (5..20).collect()).await;
        // Deleting from the middle
        check("i >= 5 and i < 15", 7..9, vec![17, 18]).await;
        check("i >= 5 and i < 15", 3..5, vec![3, 4]).await;
        check("i >= 5 and i < 15", 3..6, vec![3, 4, 15]).await;
        check("i >= 5 and i < 15", 5..6, vec![15]).await;
        check("i >= 5 and i < 15", 5..10, vec![15, 16, 17, 18, 19]).await;
        check(
            "i >= 5 and i < 15",
            0..10,
            vec![0, 1, 2, 3, 4, 15, 16, 17, 18, 19],
        )
        .await;
        // Deleting from the end
        check("i >= 15", 10..15, vec![10, 11, 12, 13, 14]).await;
        check("i >= 15", 0..15, (0..15).collect()).await;
    }

    #[rstest]
    #[tokio::test]
    async fn test_fragment_take_indices(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let mut dataset = create_dataset(test_uri, data_storage_version).await;
        let fragment = dataset
            .get_fragments()
            .into_iter()
            .find(|f| f.id() == 3)
            .unwrap();

        // Repeated indices are repeated in result.
        let batch = fragment
            .take(&[1, 2, 4, 5, 5, 8], dataset.schema())
            .await
            .unwrap();
        assert_eq!(
            batch.column_by_name("i").unwrap().as_ref(),
            &Int32Array::from(vec![121, 122, 124, 125, 125, 128])
        );

        // Unordered indices are returned in the requested order.
        let batch = fragment
            .take(&[8, 1, 5, 1, 4, 2], dataset.schema())
            .await
            .unwrap();
        assert_eq!(
            batch.column_by_name("i").unwrap().as_ref(),
            &Int32Array::from(vec![128, 121, 125, 121, 124, 122])
        );

        dataset.delete("i in (122, 123, 125)").await.unwrap();
        dataset.validate().await.unwrap();

        // Deleted rows are skipped
        let fragment = dataset
            .get_fragments()
            .into_iter()
            .find(|f| f.id() == 3)
            .unwrap();
        assert!(fragment.metadata().deletion_file.is_some());
        let batch = fragment
            .take(&[1, 2, 4, 5, 8], dataset.schema())
            .await
            .unwrap();
        assert_eq!(
            batch.column_by_name("i").unwrap().as_ref(),
            &Int32Array::from(vec![121, 124, 127, 128, 131])
        );

        // Empty indices gives empty result
        let batch = fragment.take(&[], dataset.schema()).await.unwrap();
        assert_eq!(
            batch.column_by_name("i").unwrap().as_ref(),
            &Int32Array::from(Vec::<i32>::new())
        );
    }

    #[rstest]
    #[case::unordered(&[8, 2, 5, 4, 1], &[128, 125, 121])]
    #[case::duplicates_and_deletions(&[2, 8, 2, 5, 4, 1, 8, 4], &[128, 125, 121, 128])]
    #[case::all_deleted(&[4, 2, 4], &[])]
    #[tokio::test]
    async fn test_fragment_take_rows(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[case] unordered_indices: &[u32],
        #[case] expected_values: &[i32],
    ) {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let mut dataset = create_dataset(test_uri, data_storage_version).await;
        let fragment = dataset
            .get_fragments()
            .into_iter()
            .find(|f| f.id() == 3)
            .unwrap();

        // Repeated indices are repeated in result.
        let batch = fragment
            .take_rows(
                &[1, 2, 4, 5, 5, 8],
                dataset.schema(),
                FragReadConfig::default().with_row_address(false),
            )
            .await
            .unwrap();
        assert_eq!(
            batch.column_by_name("i").unwrap().as_ref(),
            &Int32Array::from(vec![121, 122, 124, 125, 125, 128])
        );

        dataset.delete("i in (122, 124)").await.unwrap();
        dataset.validate().await.unwrap();

        // Cannot get rows 2 and 4 anymore
        let fragment = dataset
            .get_fragments()
            .into_iter()
            .find(|f| f.id() == 3)
            .unwrap();
        assert!(fragment.metadata().deletion_file.is_some());
        let batch = fragment
            .take_rows(
                &[1, 2, 4, 5, 8],
                dataset.schema(),
                FragReadConfig::default().with_row_address(false),
            )
            .await
            .unwrap();
        assert_eq!(
            batch.column_by_name("i").unwrap().as_ref(),
            &Int32Array::from(vec![121, 125, 128])
        );

        // Unordered deleted rows are omitted while surviving rows retain request order.
        let batch = fragment
            .take_rows(
                unordered_indices,
                dataset.schema(),
                FragReadConfig::default(),
            )
            .await
            .unwrap();
        assert_eq!(
            batch.column_by_name("i").unwrap().as_ref(),
            &Int32Array::from(expected_values.to_vec())
        );

        // Readers configured to materialize deletions retain their requested positions.
        let mut reader = fragment
            .open(
                dataset.schema(),
                FragReadConfig::default().with_row_id(true),
            )
            .await
            .unwrap();
        reader.with_make_deletions_null();
        let batch = reader.take_as_batch(&[8, 2, 5, 4, 1], None).await.unwrap();
        assert_eq!(
            batch.column_by_name("i").unwrap().as_ref(),
            &Int32Array::from(vec![128, 122, 125, 124, 121])
        );
        let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
        assert!(row_ids.is_valid(0));
        assert!(row_ids.is_null(1));
        assert!(row_ids.is_valid(2));
        assert!(row_ids.is_null(3));
        assert!(row_ids.is_valid(4));

        // Empty indices gives empty result
        let batch = fragment
            .take_rows(&[], dataset.schema(), FragReadConfig::default())
            .await
            .unwrap();
        assert_eq!(
            batch.column_by_name("i").unwrap().as_ref(),
            &Int32Array::from(Vec::<i32>::new())
        );

        // Can get row ids
        let batch = fragment
            .take_rows(
                &[1, 2, 4, 5, 8],
                dataset.schema(),
                FragReadConfig::default().with_row_address(true),
            )
            .await
            .unwrap();
        assert_eq!(
            batch.column_by_name("i").unwrap().as_ref(),
            &Int32Array::from(vec![121, 125, 128])
        );
        assert_eq!(
            batch.column_by_name(ROW_ADDR).unwrap().as_ref(),
            &UInt64Array::from(vec![(3 << 32) + 1, (3 << 32) + 5, (3 << 32) + 8])
        );
    }

    #[tokio::test]
    async fn test_recommit_from_file() {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let dataset = create_dataset(test_uri, LanceFileVersion::Legacy).await;
        let schema = dataset.schema();
        let dataset_rows = dataset.count_rows(None).await.unwrap();

        let mut paths: Vec<String> = Vec::new();
        for f in dataset.get_fragments() {
            for file in Fragment::from(f.clone()).files {
                let p = file.path.clone();
                paths.push(p);
            }
        }

        let mut fragments: Vec<Fragment> = Vec::new();
        for (idx, path) in paths.iter().enumerate() {
            let f = FileFragment::create_from_file(path, &dataset, idx, None)
                .await
                .unwrap();
            fragments.push(f)
        }

        let op = Operation::Overwrite {
            schema: schema.clone(),
            fragments,
            config_upsert_values: None,
            initial_bases: None,
        };

        let new_dataset =
            Dataset::commit(test_uri, op, None, None, None, Default::default(), false)
                .await
                .unwrap();

        assert_eq!(new_dataset.count_rows(None).await.unwrap(), dataset_rows);

        // Fragments will have number of rows recorded in metadata, even though
        // we passed `None` when constructing the `FileFragment`.
        let fragments = new_dataset.get_fragments();
        assert_eq!(fragments.len(), 5);
        for f in fragments {
            assert_eq!(f.metadata().num_rows(), Some(40));
            assert_eq!(f.count_rows(None).await.unwrap(), 40);
            assert_eq!(f.metadata().deletion_file, None);
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_fragment_count(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let dataset = create_dataset(test_uri, data_storage_version).await;
        let fragment = dataset.get_fragments().pop().unwrap();

        assert_eq!(fragment.count_rows(None).await.unwrap(), 40);
        assert_eq!(fragment.physical_rows().await.unwrap(), 40);
        assert!(fragment.metadata().deletion_file.is_none());

        assert_eq!(
            fragment
                .count_rows(Some("i < 170".to_string()))
                .await
                .unwrap(),
            10
        );

        let fragment = fragment
            .delete("i >= 160 and i <= 172")
            .await
            .unwrap()
            .unwrap();

        fragment.validate().await.unwrap();

        assert_eq!(fragment.count_rows(None).await.unwrap(), 27);
        assert_eq!(fragment.physical_rows().await.unwrap(), 40);
        assert!(fragment.metadata().deletion_file.is_some());
        assert_eq!(
            fragment
                .metadata()
                .deletion_file
                .as_ref()
                .unwrap()
                .num_deleted_rows,
            Some(13)
        );
    }

    #[rstest]
    #[case::empty(0)]
    #[case::non_empty(7)]
    #[tokio::test]
    async fn test_file_less_fragment_physical_rows_uses_cached_value(#[case] physical_rows: usize) {
        let test_dir = TempStrDir::default();
        let mut dataset = create_dataset(&test_dir, LanceFileVersion::Stable).await;
        assert!(dataset.manifest.writer_version.is_some());

        let fragment = FileFragment::new(
            Arc::new(dataset.clone()),
            Fragment::new(99).with_physical_rows(physical_rows),
        );
        assert_eq!(fragment.physical_rows().await.unwrap(), physical_rows);
        fragment.validate().await.unwrap();

        Arc::make_mut(&mut dataset.manifest).writer_version = None;
        let fragment = FileFragment::new(
            Arc::new(dataset),
            Fragment::new(99).with_physical_rows(physical_rows),
        );
        let err = fragment.physical_rows().await.unwrap_err();
        assert!(matches!(err, Error::NotFound { .. }));
        assert_error_contains(err, "Fragment 99 does not contain any data");

        let err = fragment.validate().await.unwrap_err();
        assert!(matches!(err, Error::CorruptFile { .. }));
        assert_error_contains(
            err,
            &format!(
                "physical_rows=Some({physical_rows}) cannot be trusted because manifest.writer_version is missing"
            ),
        );
    }

    #[tokio::test]
    async fn test_fragment_validate_checks_physical_rows_against_data_file() {
        let test_dir = TempStrDir::default();
        let dataset = create_dataset(&test_dir, LanceFileVersion::Stable).await;
        let mut metadata = dataset.manifest.fragments[0].clone();
        let actual_rows = metadata.physical_rows.unwrap();
        metadata.physical_rows = Some(actual_rows + 1);
        let fragment = FileFragment::new(Arc::new(dataset), metadata);
        assert_eq!(fragment.physical_rows().await.unwrap(), actual_rows + 1);

        let err = fragment.validate().await.unwrap_err();
        assert!(matches!(err, Error::CorruptFile { .. }));
        assert_error_contains(err, "Fragment metadata has incorrect physical_rows");
    }

    #[tokio::test]
    async fn test_file_less_fragment_validate_requires_physical_rows() {
        let test_dir = TempStrDir::default();
        let dataset = Arc::new(create_dataset(&test_dir, LanceFileVersion::Stable).await);
        let fragment = FileFragment::new(dataset, Fragment::new(99));

        let err = fragment.physical_rows().await.unwrap_err();
        assert!(matches!(err, Error::NotFound { .. }));
        assert_error_contains(err, "Fragment 99 does not contain any data");

        let err = fragment.validate().await.unwrap_err();
        assert!(matches!(err, Error::CorruptFile { .. }));
        assert_error_contains(
            err,
            "Fragment 99 has no user data files and is missing physical_rows",
        );
    }

    #[tokio::test]
    async fn test_file_less_fragment_validate_checks_deletion_metadata() {
        let test_dir = TempStrDir::default();
        let dataset = Arc::new(create_dataset(&test_dir, LanceFileVersion::Stable).await);

        let deletion_vector = DeletionVector::from_iter([2, 9]);
        let deletion_file = lance_table::io::deletion::write_deletion_file(
            &dataset.table_state.base,
            99,
            dataset.version().version,
            &deletion_vector,
            &dataset.table_state.object_store,
            0,
        )
        .await
        .unwrap()
        .unwrap();

        let mut metadata = Fragment::new(99).with_physical_rows(10);
        metadata.deletion_file = Some(deletion_file.clone());
        FileFragment::new(dataset.clone(), metadata)
            .validate()
            .await
            .unwrap();

        let mut metadata = Fragment::new(99).with_physical_rows(10);
        metadata.deletion_file = Some(DeletionFile {
            num_deleted_rows: Some(3),
            ..deletion_file
        });
        let err = FileFragment::new(dataset, metadata)
            .validate()
            .await
            .unwrap_err();
        assert!(matches!(err, Error::CorruptFile { .. }));
        assert_error_contains(
            err,
            "deletion vector length does not match metadata. Metadata: 3 Deletion vector: 2",
        );
    }

    #[tokio::test]
    async fn test_file_less_fragment_validate_rejects_out_of_range_deletions() {
        let test_dir = TempStrDir::default();
        let dataset = Arc::new(create_dataset(&test_dir, LanceFileVersion::Stable).await);

        let deletion_vector = DeletionVector::from_iter([2, 10]);
        let deletion_file = lance_table::io::deletion::write_deletion_file(
            &dataset.table_state.base,
            99,
            dataset.version().version,
            &deletion_vector,
            &dataset.table_state.object_store,
            0,
        )
        .await
        .unwrap()
        .unwrap();

        let mut metadata = Fragment::new(99).with_physical_rows(10);
        metadata.deletion_file = Some(deletion_file);
        let err = FileFragment::new(dataset, metadata)
            .validate()
            .await
            .unwrap_err();
        assert!(matches!(err, Error::CorruptFile { .. }));
        assert_error_contains(
            err,
            "deletion vector contains an offset that is out of range. Offset: 10 Fragment length: 10",
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_append_new_columns(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        for with_delete in [true, false] {
            let test_dir = TempStrDir::default();
            let test_uri = &test_dir;
            let mut dataset = create_dataset(test_uri, data_storage_version).await;
            dataset.validate().await.unwrap();
            assert_eq!(dataset.count_rows(None).await.unwrap(), 200);

            if with_delete {
                dataset.delete("i >= 15 and i < 20").await.unwrap();
                dataset.validate().await.unwrap();
                assert_eq!(dataset.count_rows(None).await.unwrap(), 195);
            }

            let new_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "double_i",
                DataType::Int32,
                true,
            )]));
            // Merge keeps the fragment list intact, so every fragment gets the new
            // column. Fragment 0 is the one carrying the deletions.
            let fragment_ids = dataset
                .manifest
                .fragments
                .iter()
                .map(|f| f.id as usize)
                .collect::<Vec<_>>();
            let mut merged_fragments = Vec::new();
            for fragment_id in fragment_ids {
                let fragment = &mut dataset.get_fragment(fragment_id).unwrap();
                let mut updater = fragment
                    .updater(Some(&["i"]), None, None, None)
                    .await
                    .unwrap();
                while let Some(batch) = updater.next().await.unwrap() {
                    let input_col = batch.column_by_name("i").unwrap();
                    let result_col = mul(input_col, &Int32Array::new_scalar(2)).unwrap();
                    let batch = RecordBatch::try_new(
                        new_schema.clone(),
                        vec![Arc::new(result_col) as ArrayRef],
                    )
                    .unwrap();
                    updater.update(batch).await.unwrap();
                }
                let new_fragment = updater.finish().await.unwrap();

                assert_eq!(new_fragment.files.len(), 2);
                merged_fragments.push(new_fragment);
            }

            // Scan again
            let mut full_schema = dataset.schema().merge(new_schema.as_ref()).unwrap();
            full_schema.set_field_id(None);
            let before_version = dataset.version().version;

            let op = Operation::Merge {
                fragments: merged_fragments,
                schema: full_schema.clone(),
                preserves_nullability: true,
            };

            let dataset = Dataset::commit(
                test_uri,
                op,
                Some(before_version),
                None,
                None,
                Default::default(),
                false,
            )
            .await
            .unwrap();

            assert_eq!(
                dataset.count_rows(None).await.unwrap(),
                if with_delete { 195 } else { 200 }
            );
            assert_eq!(dataset.version().version, before_version + 1);
            dataset.validate().await.unwrap();
            let new_projection = full_schema.project(&["i", "double_i"]).unwrap();

            let stream = dataset
                .scan()
                .batch_size(10)
                .project(&["i", "double_i"])
                .unwrap()
                .try_into_stream()
                .await
                .unwrap();
            let batches = stream.try_collect::<Vec<_>>().await.unwrap();

            assert_eq!(batches[1].schema().as_ref(), &(&new_projection).into());
            let expected_i = match (with_delete, data_storage_version) {
                // Legacy format uses old scan node which deletes after read and
                // so the batch is truncated
                (true, LanceFileVersion::Legacy) => vec![10, 11, 12, 13, 14],
                // Newer formats delete before read and so we get a full batch of 10
                (true, _) => vec![10, 11, 12, 13, 14, 20, 21, 22, 23, 24],
                (false, _) => vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            };
            let expected_batch = RecordBatch::try_new(
                Arc::new(ArrowSchema::new(vec![
                    ArrowField::new("i", DataType::Int32, true),
                    ArrowField::new("double_i", DataType::Int32, true),
                ])),
                vec![
                    Arc::new(Int32Array::from_iter_values(expected_i.iter().copied())),
                    Arc::new(Int32Array::from_iter_values(
                        expected_i.iter().map(|i| 2 * i),
                    )),
                ],
            )
            .unwrap();
            assert_eq!(batches[1], expected_batch);
        }
    }

    /// A deletion vector naming a row the fragment does not have leaves the restorer
    /// with rows it can never account for, so `Updater::next` has to refuse at the end
    /// of the stream rather than let a data file short of those rows be written.
    ///
    /// `write_deletions` rejects an over-long vector, so the file is written directly
    /// to get a fragment into this state.
    #[tokio::test]
    async fn test_updater_rejects_deletion_vector_past_end_of_fragment() {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let mut dataset = create_dataset(test_uri, LanceFileVersion::Stable).await;

        // Point a fragment's deletion file at a row it does not have. 200 rows are
        // spread over several 40-row fragments, so 10_000 is past the end of any of
        // them. Pick a fragment whose id is not zero, so the assertion below cannot
        // pass on a message that dropped the id entirely.
        let deletion_vector: DeletionVector = [10_000].into_iter().collect();
        let fragment_index = 1;
        let fragment_id = dataset.manifest.fragments[fragment_index].id;
        assert_ne!(fragment_id, 0, "need a non-zero fragment id");
        let deletion_file = lance_table::io::deletion::write_deletion_file(
            &dataset.table_state.base,
            fragment_id,
            dataset.version().version,
            &deletion_vector,
            dataset.table_state.object_store.as_ref(),
            0,
        )
        .await
        .unwrap();
        let mut fragments = dataset.manifest.fragments.as_ref().clone();
        fragments[fragment_index].deletion_file = deletion_file;
        let mut manifest = dataset.manifest.as_ref().clone();
        manifest.fragments = Arc::new(fragments);
        dataset.manifest = Arc::new(manifest);

        let new_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "double_i",
            DataType::Int32,
            true,
        )]));
        let fragment = dataset.get_fragment(fragment_id as usize).unwrap();
        let mut updater = fragment
            .updater(Some(&["i"]), None, None, None)
            .await
            .unwrap();

        // Every live row is handed back, so the loop only ends when next() gives up.
        let err = loop {
            match updater.next().await {
                Ok(Some(batch)) => {
                    let input_col = batch.column_by_name("i").unwrap();
                    let result_col = mul(input_col, &Int32Array::new_scalar(2)).unwrap();
                    let batch = RecordBatch::try_new(
                        new_schema.clone(),
                        vec![Arc::new(result_col) as ArrayRef],
                    )
                    .unwrap();
                    updater.update(batch).await.unwrap();
                }
                Ok(None) => panic!("expected next() to refuse the unaccounted-for row"),
                Err(err) => break err,
            }
        };

        assert!(matches!(err, Error::NotSupported { .. }), "{err:?}");
        let message = err.to_string();
        assert!(
            message.contains("unaccounted for"),
            "expected the stream-ended wording, got: {message}"
        );
        assert!(
            message.contains(&format!("fragment {fragment_id}")),
            "message should name the fragment: {message}"
        );
    }

    /// Stopping early is diagnosed as unread input, not as unmaterialized deletions:
    /// the caller's remedy is to keep polling, not to run compaction. `finish` probes the
    /// stream before consulting the restorer for exactly that reason -- otherwise which
    /// message a caller saw would depend only on where the deletions happened to fall.
    #[tokio::test]
    async fn test_updater_finish_rejects_stopping_early() -> Result<()> {
        let test_dir = TempStrDir::default();
        let mut dataset = create_dataset(&test_dir, LanceFileVersion::Stable).await;
        dataset.delete("i >= 5 AND i < 40").await?;

        let fragment_id = dataset.manifest.fragments[0].id;
        let fragment = dataset.get_fragment(fragment_id as usize).unwrap();
        let mut updater = fragment.updater(Some(&["i"]), None, Some(5), None).await?;

        let num_rows = updater.next().await?.unwrap().num_rows();
        assert_eq!(num_rows, 5);
        let new_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "double_i",
            DataType::Int32,
            true,
        )]));
        updater
            .update(RecordBatch::try_new(
                new_schema,
                vec![Arc::new(Int32Array::from_iter_values(0..num_rows as i32))],
            )?)
            .await?;

        let err = updater.finish().await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err:?}");
        let message = err.to_string();
        assert!(message.contains("cannot finish"), "{message}");
        assert!(message.contains("unread input batches"), "{message}");
        assert!(
            !message.contains("run compaction"),
            "stopping early is not a compaction problem: {message}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_updater_rejects_unpaired_and_unread_batches() -> Result<()> {
        let test_dir = TempStrDir::default();
        let dataset = create_dataset(&test_dir, LanceFileVersion::Stable).await;
        let fragment = dataset.get_fragment(0).unwrap();
        let mut updater = fragment.updater(Some(&["i"]), None, Some(5), None).await?;

        let num_rows = updater.next().await?.unwrap().num_rows();
        assert_eq!(num_rows, 5);

        let error = updater.next().await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error.to_string().contains("previous input batch"),
            "{error}"
        );

        let output_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "double_i",
            DataType::Int32,
            false,
        )]));
        let output = RecordBatch::try_new(
            output_schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..num_rows as i32))],
        )?;
        updater.update(output.clone()).await?;

        let error = updater.update(output).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("no input data"), "{error}");

        let error = updater.finish().await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error.to_string().contains("unread input batches"),
            "{error}"
        );

        // Probing must not consume a batch. In particular, repeatedly calling
        // `finish` cannot walk the stream to its end and commit a short file.
        let repeated = updater.finish().await.unwrap_err();
        assert!(
            repeated.to_string().contains("unread input batches"),
            "{repeated}"
        );

        let mut rows_written = num_rows;
        while let Some(input) = updater.next().await? {
            if rows_written == num_rows {
                let values = arrow_array::cast::as_primitive_array::<arrow_array::types::Int32Type>(
                    input.column_by_name("i").unwrap().as_ref(),
                );
                assert_eq!(values.value(0), 5);
            }
            let rows = input.num_rows();
            updater
                .update(RecordBatch::try_new(
                    output_schema.clone(),
                    vec![Arc::new(Int32Array::from_iter_values(0..rows as i32))],
                )?)
                .await?;
            rows_written += rows;
        }
        assert_eq!(rows_written, 40);
        let updated = updater.finish().await?;
        assert_eq!(updated.files.len(), 2);
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn test_merge_fragment(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let mut dataset = create_dataset(test_uri, data_storage_version).await;
        dataset.validate().await.unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 200);

        let deleted_range = 15..20;
        dataset.delete("i >= 15 and i < 20").await.unwrap();
        dataset.validate().await.unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 195);

        // Create data to merge: merge in double the data
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, true),
            ArrowField::new("double_i", DataType::Int32, true),
        ]));
        let to_merge = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..200)),
                Arc::new(Int32Array::from_iter_values((0..400).step_by(2))),
            ],
        )
        .unwrap();

        let stream = RecordBatchIterator::new(vec![Ok(to_merge)], schema.clone());
        dataset.merge(stream, "i", "i").await.unwrap();
        dataset.validate().await.unwrap();

        // Validate the resulting data
        let batches = dataset
            .scan()
            .project(&["i", "double_i"])
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let batch = concat_batches(&schema, &batches).unwrap();

        let mut row_id: i32 = 0;
        let mut i: usize = 0;
        let array_i: &Int32Array = as_primitive_array(&batch["i"]);
        let array_double_i: &Int32Array = as_primitive_array(&batch["double_i"]);
        while row_id < 200 {
            if deleted_range.contains(&row_id) {
                row_id += 1;
                continue;
            }
            assert_eq!(array_i.value(i), row_id);
            assert_eq!(array_double_i.value(i), 2 * row_id);
            row_id += 1;
            i += 1;
        }
    }

    #[tokio::test]
    async fn test_merge_columns_field_ids_match_data_file() {
        let test_dir = TempStrDir::default();
        let mut dataset = create_dataset(&test_dir, LanceFileVersion::Stable).await;
        // The dropped column's id stays in the data files, so the manifest's
        // max field id is larger than the schema's.
        dataset.drop_columns(&["s"]).await.unwrap();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, true),
            ArrowField::new("double_i", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..200)),
                Arc::new(Int32Array::from_iter_values((0..400).step_by(2))),
            ],
        )
        .unwrap();
        let fragments = dataset.get_fragments();
        let mut merged_fragments = Vec::with_capacity(fragments.len());
        let mut merged_schema = dataset.schema().clone();
        for mut frag in fragments {
            let stream = RecordBatchIterator::new(vec![Ok(batch.clone())], schema.clone());
            let (new_frag, new_schema) = frag.merge_columns(stream, "i", "i").await.unwrap();
            let new_field_id = new_schema.field("double_i").unwrap().id;
            assert_eq!(new_field_id, dataset.manifest.max_field_id() + 1);
            assert_eq!(
                new_frag.files.last().unwrap().fields.as_ref(),
                &[new_field_id]
            );
            merged_fragments.push(new_frag);
            merged_schema = new_schema;
        }

        let dataset = Dataset::commit(
            &test_dir,
            Operation::Merge {
                fragments: merged_fragments,
                schema: merged_schema,
                preserves_nullability: true,
            },
            Some(dataset.manifest.version),
            None,
            None,
            Default::default(),
            false,
        )
        .await
        .unwrap();
        let actual = dataset
            .scan()
            .project(&["i", "double_i"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(actual, batch);
    }

    #[tokio::test]
    async fn test_write_batch_size() {
        // V1 ONLY
        //
        // This test is only for the legacy version of the file format.
        // It ensures that the `max_rows_per_group` property is respected
        // and this property does not exist in V2.
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            true,
        )]));

        let in_memory_batch = 1024;
        let batches: Vec<RecordBatch> = (0..10)
            .map(|i| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(Int32Array::from_iter_values(
                        i * in_memory_batch..(i + 1) * in_memory_batch,
                    ))],
                )
                .unwrap()
            })
            .collect();

        let batch_iter = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());

        let fragment = FileFragment::create(
            test_uri,
            10,
            batch_iter,
            Some(WriteParams {
                max_rows_per_group: 100,
                data_storage_version: Some(LanceFileVersion::Legacy),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let (object_store, base_path) = ObjectStore::from_uri(test_uri).await.unwrap();
        let file_reader = V1FileReader::try_new_with_fragment_id(
            &object_store,
            &base_path
                .clone()
                .join("data")
                .join(fragment.files[0].path.as_str()),
            schema.as_ref().try_into().unwrap(),
            10,
            0,
            1,
            None,
        )
        .await
        .unwrap();

        for i in 0..file_reader.num_batches() - 1 {
            assert_eq!(file_reader.num_rows_in_batch(i as i32), 100);
        }
        assert_eq!(
            file_reader.num_rows_in_batch(file_reader.num_batches() as i32 - 1) as i32,
            in_memory_batch * 10 % 100
        );
    }

    #[tokio::test]
    async fn test_shuffled_columns() -> Result<()> {
        // Validates we can handle datasets where the order of columns is not
        // aligned with the order of the data files. This can happen when replacing
        // columns in a dataset.
        let batch_i = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "i",
                DataType::Int32,
                true,
            )])),
            vec![Arc::new(Int32Array::from_iter_values(0..20))],
        )?;

        let batch_s = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "s",
                DataType::Utf8,
                true,
            )])),
            vec![Arc::new(StringArray::from_iter_values(
                (0..20).map(|v| format!("s-{}", v)),
            ))],
        )?;

        // Write batch_i as a fragment
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch_i.clone())], batch_i.schema().clone()),
            test_uri,
            None,
        )
        .await?;

        let fragment = dataset.get_fragments().pop().unwrap();

        // Write batch_s using add_columns
        let mut updater = fragment.updater(Some(&["i"]), None, None, None).await?;
        updater.next().await?;
        updater.update(batch_s.clone()).await?;
        let frag = updater.finish().await?;

        // Rearrange schema so it's `s` then `i`.
        let schema = updater.schema().unwrap().clone().project(&["s", "i"])?;

        let dataset = Dataset::commit(
            test_uri,
            Operation::Merge {
                schema,
                fragments: vec![frag],
                preserves_nullability: true,
            },
            Some(dataset.manifest.version),
            None,
            None,
            Default::default(),
            false,
        )
        .await?;

        let expected_data = batch_s.merge(&batch_i)?;
        let actual_data = dataset.scan().try_into_batch().await?;
        assert_eq!(expected_data, actual_data);

        // Also take, read_range, and read_batch_projected
        let reader = dataset
            .get_fragments()
            .first()
            .unwrap()
            .open(dataset.schema(), FragReadConfig::default())
            .await?;
        let actual_data = reader.take_as_batch(&[0, 1, 2], None).await?;
        assert_eq!(expected_data.slice(0, 3), actual_data);

        let actual_data = reader
            .read_range(0..3, 3)
            .await
            .unwrap()
            .next()
            .await
            .unwrap()
            .await
            .unwrap();
        assert_eq!(expected_data.slice(0, 3), actual_data);

        // Also check case of row_id.
        let expected_data = expected_data.try_with_column(
            ROW_ID_FIELD.clone(),
            Arc::new(UInt64Array::from_iter_values(0..20)),
        )?;
        let actual_data = dataset.scan().with_row_id().try_into_batch().await?;
        assert_eq!(expected_data, actual_data);

        Ok(())
    }

    pub mod demand_row_ids {
        use super::*;
        use arrow_array::types::Int32Type;
        use lance_datagen::ArrayGeneratorExt;
        use lance_io::object_store::{ObjectStoreRegistry, WrappingObjectStore, uri_to_url};
        use lance_io::utils::tracking_store::{IOTracker, IoStats};
        use lance_table::format::{ROW_ID_FIELD_ID, RowIdMeta};
        use pretty_assertions::assert_eq;

        use crate::dataset::builder::DatasetBuilder;
        use crate::dataset::optimize::{CompactionMode, CompactionOptions, compact_files};
        use crate::dataset::rowids::{
            INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, SPILL_ROW_LINEAGE_CONFIG_KEY,
        };
        use crate::index::DatasetIndexExt;
        use crate::utils::test::FailingProxyStore;
        use lance_index::{IndexType, scalar::ScalarIndexParams};

        // Binary-copy compaction gives each output a separate lineage-only file,
        // so its reads cannot be confused with user-column or footer reads.
        pub async fn fixture(spill: bool) -> (TempStrDir, Dataset) {
            let dir = TempStrDir::default();
            // Normalize the file URL first so Windows drive letters remain in
            // the path, rather than being parsed as the custom scheme's host.
            let uri = uri_to_url(dir.as_str()).unwrap().as_str().replacen(
                "file:",
                "file-object-store:",
                1,
            );
            let data = gen_batch()
                .col("i", array::step::<Int32Type>())
                .col("n", array::fill::<Int32Type>(0).with_random_nulls(1.0))
                .into_reader_rows(RowCount::from(6), lance_datagen::BatchCount::from(4));
            let mut dataset = Dataset::write(
                data,
                &uri,
                Some(WriteParams {
                    enable_stable_row_ids: true,
                    max_rows_per_file: 6,
                    data_storage_version: Some(LanceFileVersion::Stable),
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
            if spill {
                dataset
                    .update_config([
                        (SPILL_ROW_LINEAGE_CONFIG_KEY, "true"),
                        (INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, "0"),
                    ])
                    .await
                    .unwrap();
            }
            compact_files(
                &mut dataset,
                CompactionOptions {
                    target_rows_per_fragment: 12,
                    // The assertions use the original row order; concurrent
                    // compaction commits its outputs in completion order.
                    num_threads: Some(1),
                    compaction_mode: Some(CompactionMode::ForceBinaryCopy),
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();
            assert_eq!(dataset.get_fragments().len(), 2);
            for fragment in dataset.get_fragments() {
                assert_eq!(
                    matches!(fragment.metadata().row_id_meta, Some(RowIdMeta::Column)),
                    spill
                );
            }
            dataset.delete("i % 6 = 1").await.unwrap();
            // Disable both caches so sharing across cloned readers, including
            // concurrent reads, cannot accidentally be supplied by the session.
            let dataset = DatasetBuilder::from_uri(&uri)
                .with_session(Arc::new(Session::new(
                    0,
                    0,
                    Arc::new(ObjectStoreRegistry::default()),
                )))
                .load()
                .await
                .unwrap();
            (dir, dataset)
        }

        pub fn lineage_reads(stats: &IoStats, fragment: &FileFragment) -> usize {
            let file = fragment
                .metadata()
                .row_lineage_file(ROW_ID_FIELD_ID)
                .unwrap()
                .unwrap();
            stats
                .requests
                .iter()
                .filter(|request| request.path.as_ref().ends_with(&file.path))
                .count()
        }

        #[rstest]
        #[case::payload(false)]
        #[case::row_address_only(true)]
        #[tokio::test]
        async fn skips_unused_sequence(#[case] addresses_only: bool) {
            let (_dir, dataset) = fixture(true).await;
            let fragment = dataset.get_fragments().remove(0);
            let projection = if addresses_only {
                dataset.schema().project::<&str>(&[]).unwrap()
            } else {
                dataset.schema().clone()
            };
            dataset.table_state.object_store.io_stats_incremental();
            let reader = fragment
                .open(
                    &projection,
                    FragReadConfig::default().with_row_address(addresses_only),
                )
                .await
                .unwrap();
            let batch = reader.read_range_as_batch(0..12).await.unwrap();
            assert_eq!(batch.num_rows(), 10);
            let stats = dataset.table_state.object_store.io_stats_incremental();
            assert_eq!(lineage_reads(&stats, &fragment), 0, "{stats:?}");
            if addresses_only {
                let expected: Vec<u64> = (0..12)
                    .filter(|i| i % 6 != 1)
                    .map(|i| RowAddress::new_from_parts(fragment.id() as u32, i).into())
                    .collect();
                assert_eq!(
                    batch[ROW_ADDR]
                        .as_primitive::<UInt64Type>()
                        .values()
                        .as_ref(),
                    expected
                );
            } else {
                assert_eq!(batch["n"].null_count(), 10);
                assert!(
                    stats.read_bytes > 0,
                    "user data and metadata must still be read"
                );
            }
        }

        #[rstest]
        #[case::payload(false)]
        #[case::row_address_only(true)]
        #[tokio::test]
        async fn skips_unused_scan_sequence(#[case] addresses_only: bool) {
            let (_dir, dataset) = fixture(true).await;
            let mut scanner = dataset.scan();
            if addresses_only {
                scanner.project::<&str>(&[]).unwrap().with_row_address();
            }
            dataset.table_state.object_store.io_stats_incremental();
            let batch = scanner.try_into_batch().await.unwrap();
            assert_eq!(batch.num_rows(), 20);
            let stats = dataset.table_state.object_store.io_stats_incremental();
            for fragment in dataset.get_fragments() {
                assert_eq!(lineage_reads(&stats, &fragment), 0, "{stats:?}");
            }
        }

        #[rstest]
        #[case::inline(false)]
        #[case::spilled(true)]
        #[tokio::test]
        async fn late_ids_on_cloned_reader(#[case] spill: bool) {
            let (_dir, dataset) = fixture(spill).await;
            for (index, fragment) in dataset.get_fragments().into_iter().enumerate() {
                let reader = fragment
                    .open(dataset.schema(), FragReadConfig::default())
                    .await
                    .unwrap();
                let mut ids_reader = reader.clone();
                ids_reader.with_row_id().with_row_address();
                let batch = ids_reader
                    .take_as_batch(&[0, 2, 2, 5, 11], None)
                    .await
                    .unwrap();
                let expected = [0, 2, 2, 5, 11].map(|i| (index * 12 + i) as u64);
                assert_eq!(
                    batch[ROW_ID].as_primitive::<UInt64Type>().values().as_ref(),
                    expected
                );
                assert_eq!(batch["n"].null_count(), expected.len());
                assert!(
                    reader
                        .read_range_as_batch(0..2)
                        .await
                        .unwrap()
                        .column_by_name(ROW_ID)
                        .is_none()
                );
                let batches = ids_reader
                    .read_ranges(Arc::from([0..3, 5..8]), 2)
                    .await
                    .unwrap()
                    .buffered(2)
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap();
                let actual = batches
                    .iter()
                    .flat_map(|batch| {
                        batch[ROW_ID]
                            .as_primitive::<UInt64Type>()
                            .values()
                            .iter()
                            .copied()
                    })
                    .collect::<Vec<_>>();
                assert_eq!(actual, [0, 2, 5, 6].map(|i| (index * 12 + i) as u64));
                let logical = ids_reader
                    .read_range(1..3, 2)
                    .await
                    .unwrap()
                    .buffered(1)
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap();
                assert_eq!(
                    logical[0][ROW_ID]
                        .as_primitive::<UInt64Type>()
                        .values()
                        .as_ref(),
                    [(index * 12 + 2) as u64, (index * 12 + 3) as u64]
                );
                let mut null_reader = ids_reader.clone();
                null_reader.with_make_deletions_null();
                let null_batch = null_reader.read_range_as_batch(0..3).await.unwrap();
                assert_eq!(
                    null_batch[ROW_ID]
                        .as_primitive::<UInt64Type>()
                        .iter()
                        .collect::<Vec<_>>(),
                    [
                        Some((index * 12) as u64),
                        None,
                        Some((index * 12 + 2) as u64)
                    ]
                );
                assert_eq!(
                    ids_reader
                        .take_as_batch(&[], None)
                        .await
                        .unwrap()
                        .num_rows(),
                    0
                );
            }
            let batch = dataset
                .scan()
                .filter("i >= 10 AND n IS NULL")
                .unwrap()
                .with_row_id()
                .try_into_batch()
                .await
                .unwrap();
            let expected: Vec<u64> = (10..24).filter(|i| i % 6 != 1).collect();
            assert_eq!(
                batch[ROW_ID].as_primitive::<UInt64Type>().values().as_ref(),
                expected
            );
        }

        #[tokio::test]
        async fn indexed_scan_skips_stable_id_mapping_for_a_fresh_index() {
            // A freshly built scalar index is address-domain, even on a
            // stable-row-id dataset, so the scalar-index scan path consumes
            // its result directly — no per-fragment row-id-sequence read is
            // needed to map it back to physical offsets.
            let (_dir, mut dataset) = fixture(true).await;
            dataset
                .create_index(
                    &["i"],
                    IndexType::Scalar,
                    None,
                    &ScalarIndexParams::default(),
                    false,
                )
                .await
                .unwrap();
            let mut scanner = dataset.scan();
            scanner
                .project(&["i"])
                .unwrap()
                .filter("i >= 10 AND i < 17")
                .unwrap();
            let plan = scanner.explain_plan(false).await.unwrap();
            assert!(plan.contains("ScalarIndex"), "{plan}");
            dataset.table_state.object_store.io_stats_incremental();
            let batch = scanner.try_into_batch().await.unwrap();
            assert_eq!(
                batch["i"].as_primitive::<Int32Type>().values().as_ref(),
                [10, 11, 12, 14, 15, 16]
            );
            let stats = dataset.table_state.object_store.io_stats_incremental();
            for fragment in dataset.get_fragments() {
                assert_eq!(
                    lineage_reads(&stats, &fragment),
                    0,
                    "address-domain index results need no stable-to-physical mapping: {stats:?}"
                );
            }
        }

        #[tokio::test]
        async fn shared_load_retries_and_preserves_request_store() {
            let (_dir, dataset) = fixture(true).await;
            let failures = Arc::new(FailingProxyStore::new());
            let tracker = Arc::new(IOTracker::default());
            let request_dataset = dataset.with_object_store_wrappers([
                failures.clone() as Arc<dyn WrappingObjectStore>,
                tracker.clone() as Arc<dyn WrappingObjectStore>,
            ]);
            let fragment = request_dataset.get_fragments().remove(0);
            let mut reader = fragment
                .open(
                    &dataset.schema().project::<&str>(&[]).unwrap(),
                    FragReadConfig::default().with_row_address(true),
                )
                .await
                .unwrap();
            reader.with_row_id();
            tracker.incremental_stats();
            failures.fail_when("get_opts", ".lance", "retry deferred lineage read");
            let error = reader.read_range_as_batch(0..3).await.unwrap_err();
            assert!(matches!(error, Error::IO { .. }), "{error:?}");
            assert!(
                error.to_string().contains("retry deferred lineage read"),
                "{error}"
            );
            failures.clear_fail_when("get_opts", ".lance");
            tracker.incremental_stats();
            let cloned = reader.clone();
            let (first, second) = tokio::join!(
                reader.read_range_as_batch(0..3),
                cloned.read_range_as_batch(5..8)
            );
            assert_eq!(
                first.unwrap()[ROW_ID]
                    .as_primitive::<UInt64Type>()
                    .values()
                    .as_ref(),
                [0, 2]
            );
            assert_eq!(
                second.unwrap()[ROW_ID]
                    .as_primitive::<UInt64Type>()
                    .values()
                    .as_ref(),
                [5, 6]
            );
            let first_load = tracker.incremental_stats();
            assert!(
                lineage_reads(&first_load, &fragment) > 0,
                "late load must use this request's store"
            );
            reader.read_range_as_batch(2..3).await.unwrap();
            assert_eq!(
                tracker.incremental_stats().read_iops,
                0,
                "clones must retain the loaded sequence even with the session cache disabled"
            );
            load_row_id_sequence(&request_dataset, fragment.metadata())
                .await
                .unwrap();
            let single_load = tracker.incremental_stats();
            assert_eq!(
                first_load.read_iops, single_load.read_iops,
                "concurrent clones must perform only one sequence load"
            );
            assert_eq!(first_load.read_bytes, single_load.read_bytes);
        }
    }

    #[tokio::test]
    async fn test_row_id_reader() -> Result<()> {
        // Make sure we can create a fragment reader that only captures the row_id.
        let batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "i",
                DataType::Int32,
                true,
            )])),
            vec![Arc::new(Int32Array::from_iter_values(0..20))],
        )?;

        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema().clone()),
            test_uri,
            None,
        )
        .await?;

        let fragment = dataset.get_fragments().pop().unwrap();

        let reader = fragment
            .open(
                &dataset.schema().project::<&str>(&[])?,
                FragReadConfig::default().with_row_id(true),
            )
            .await?;
        let batch = reader.read_range_as_batch(0..20).await?;

        let expected_data = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ROW_ID_FIELD.clone()])),
            vec![Arc::new(UInt64Array::from_iter_values(0..20))],
        )?;
        assert_eq!(expected_data, batch);

        // We should get error if we pass empty schema and with_row_id false
        let res = fragment
            .open(
                &dataset.schema().project::<&str>(&[])?,
                FragReadConfig::default(),
            )
            .await;
        assert!(matches!(res, Err(Error::NotFound { .. })));

        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn create_from_file_v2(
        #[values(
            LanceFileVersion::V2_0,
            LanceFileVersion::V2_1,
            LanceFileVersion::V2_2,
            LanceFileVersion::V2_3
        )]
        file_version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let make_gen = || {
            gen_batch()
                .col("str", array::rand_type(&DataType::Utf8))
                .col("int", array::rand_type(&DataType::Int32))
        };

        let batch = make_gen().into_batch_rows(RowCount::from(128)).unwrap();
        let dataset = TestDatasetGenerator::new(vec![batch], LanceFileVersion::Stable)
            .make_hostile(test_uri)
            .await;

        let new_data = make_gen().into_batch_rows(RowCount::from(128)).unwrap();
        let store = ObjectStore::local();
        let file_path = dataset.data_dir().join("some_file.lance");
        let object_writer = store.create(&file_path).await.unwrap();
        let mut file_writer = lance_file::versions::create_lazy_writer(
            file_version.resolve(),
            object_writer,
            FileWriterOptions::default(),
        )
        .unwrap();
        file_writer.write_batch(&new_data).await.unwrap();
        file_writer.finish().await.unwrap();

        let frag = FileFragment::create_from_file("some_file.lance", &dataset, 0, Some(128))
            .await
            .unwrap();

        assert_eq!(
            Fragment::try_infer_version(std::slice::from_ref(&frag))
                .unwrap()
                .unwrap(),
            file_version.resolve()
        );

        let op = Operation::Append {
            fragments: vec![frag],
        };
        let transaction = Transaction::new_from_version(dataset.version().version, op);
        let dataset = CommitBuilder::new(Arc::new(dataset))
            .with_storage_format(file_version)
            .execute(transaction)
            .await
            .unwrap();

        assert_eq!(
            dataset
                .count_rows(Some("int IS NOT NULL".to_string()))
                .await
                .unwrap(),
            256
        );
    }

    #[tokio::test]
    async fn test_lazy_column_metadata_scan_reads_less_than_full_projection() {
        let num_columns = 512;
        let rows_per_batch = 100;
        let num_batches = 10;
        let schema = Arc::new(ArrowSchema::new(
            (0..num_columns)
                .map(|i| ArrowField::new(format!("col_{i}"), DataType::Int32, true))
                .collect::<Vec<_>>(),
        ));
        let batches = (0..num_batches)
            .map(|batch_idx| {
                let columns = (0..num_columns)
                    .map(|column_idx| {
                        Arc::new(Int32Array::from_iter_values((0..rows_per_batch).map(
                            |row_idx| (batch_idx * rows_per_batch + row_idx) as i32 + column_idx,
                        ))) as ArrayRef
                    })
                    .collect::<Vec<_>>();
                RecordBatch::try_new(schema.clone(), columns).unwrap()
            })
            .collect::<Vec<_>>();

        let test_dir = TempStrDir::default();
        let write_params = WriteParams {
            max_rows_per_file: rows_per_batch * num_batches,
            max_rows_per_group: rows_per_batch,
            data_storage_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        };
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
        let dataset = Dataset::write(reader, &test_dir, Some(write_params))
            .await
            .unwrap();

        let projection = dataset.schema().project(&["col_0"]).unwrap();
        let fragment = dataset.get_fragment(0).unwrap();

        dataset
            .table_state
            .object_store
            .as_ref()
            .io_stats_incremental();
        let narrow_reader = fragment
            .open(&projection, FragReadConfig::default())
            .await
            .unwrap();
        let mut empty_narrow_stream = narrow_reader.take_range(0..0, 1024).await.unwrap();
        assert!(empty_narrow_stream.next().await.is_none());
        let narrow_metadata_stats = dataset
            .table_state
            .object_store
            .as_ref()
            .io_stats_incremental();
        assert!(
            narrow_metadata_stats.read_iops <= 3,
            "expected lazy metadata open to skip the schema buffer read, iops={}, bytes={}",
            narrow_metadata_stats.read_iops,
            narrow_metadata_stats.read_bytes
        );

        let full_projection = dataset.schema().clone();
        let full_reader = fragment
            .open(&full_projection, FragReadConfig::default())
            .await
            .unwrap();
        let mut empty_full_stream = full_reader.take_range(0..0, 1024).await.unwrap();
        assert!(empty_full_stream.next().await.is_none());
        let full_metadata_stats = dataset
            .table_state
            .object_store
            .as_ref()
            .io_stats_incremental();

        assert!(
            full_metadata_stats.read_bytes > narrow_metadata_stats.read_bytes * 4,
            "expected narrow lazy metadata read to fetch much less than full metadata, narrow={} bytes, full={} bytes",
            narrow_metadata_stats.read_bytes,
            full_metadata_stats.read_bytes
        );

        let mut narrow_scan = dataset.scan();
        let narrow_batch = narrow_scan
            .project(&["col_0"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(narrow_batch.num_columns(), 1);
        assert_eq!(narrow_batch.num_rows(), rows_per_batch * num_batches);

        let taken = fragment.take(&[0, 777, 999], &projection).await.unwrap();
        let taken_values = taken
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(taken_values.values(), &[0, 777, 999]);

        let object_store = dataset.object_store(None).await.unwrap();
        let scheduler = ScanScheduler::new(
            object_store.clone(),
            SchedulerConfig::max_bandwidth(&object_store),
        );
        let stats = fragment
            .storage_stats_with(&projection, scheduler)
            .await
            .unwrap();
        assert_eq!(stats.len(), num_columns as usize);
        assert!(stats.iter().all(|(_, size)| *size > 0));
    }

    #[tokio::test]
    async fn v1_fragment_storage_stats_skip_data_files() {
        use crate::dataset::statistics::DatasetStatisticsExt;

        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let dataset = create_dataset(test_uri, LanceFileVersion::Legacy).await;
        let mut expected: Vec<(u32, u64)> = dataset
            .schema()
            .field_ids()
            .into_iter()
            .map(|id| (id as u32, 0))
            .collect();
        expected.sort_unstable_by_key(|(field_id, _)| *field_id);

        dataset
            .table_state
            .object_store
            .as_ref()
            .io_stats_incremental();
        let stats = dataset
            .get_fragment(0)
            .unwrap()
            .storage_stats()
            .await
            .unwrap();
        let io_stats = dataset
            .table_state
            .object_store
            .as_ref()
            .io_stats_incremental();
        assert_eq!(
            io_stats.read_iops, 0,
            "v1 fragment stats should not open data files, read {} bytes",
            io_stats.read_bytes
        );
        assert_eq!(stats, expected);

        let data_stats = Arc::new(dataset).calculate_data_stats().await.unwrap();
        let data_pairs: Vec<(u32, u64)> = data_stats
            .fields
            .iter()
            .map(|field| (field.id, field.bytes_on_disk))
            .collect();
        assert_eq!(data_pairs, stats);
    }

    #[test]
    fn test_indexed_metadata_heuristic_counts_selected_physical_columns() {
        let schema = Schema::try_from(&ArrowSchema::new(vec![
            ArrowField::new(
                "s",
                DataType::Struct(
                    vec![
                        ArrowField::new("x", DataType::Int32, true),
                        ArrowField::new("y", DataType::Int32, true),
                    ]
                    .into(),
                ),
                true,
            ),
            ArrowField::new("a", DataType::Int32, true),
            ArrowField::new("b", DataType::Int32, true),
            ArrowField::new("c", DataType::Int32, true),
        ]))
        .unwrap();
        let data_file = DataFile {
            path: "wide.lance".to_string(),
            fields: Arc::from([0, 1, 2, 3, 4, 5]),
            column_indices: Arc::from([-1, 0, 1, 2, 3, 4]),
            file_major_version: 2,
            file_minor_version: 1,
            file_size_bytes: CachedFileSize::unknown(),
            file_metadata_size_bytes: None,
            base_id: None,
        };

        let full_struct = file_versions::reader_projection_from_column_names(
            ConcreteFileVersion::V2_1,
            &schema,
            &["s"],
        )
        .unwrap();
        assert_eq!(full_struct.column_indices.len(), 2);
        let valid_column_count = data_file
            .column_indices
            .iter()
            .filter(|column_index| **column_index >= 0)
            .count();
        assert!(full_struct.column_indices.len().saturating_mul(4) >= valid_column_count);

        let partial_struct = file_versions::reader_projection_from_column_names(
            ConcreteFileVersion::V2_1,
            &schema,
            &["s.x"],
        )
        .unwrap();
        assert_eq!(partial_struct.column_indices.len(), 1);
        assert!(partial_struct.column_indices.len().saturating_mul(4) < valid_column_count);
    }

    #[tokio::test]
    async fn test_iops_read_small() {
        // Create a file that has 8 columns.
        let schema = Arc::new(ArrowSchema::new(
            (0..8)
                .map(|i| ArrowField::new(format!("col_{}", i), DataType::Int32, true))
                .collect::<Vec<_>>(),
        ));

        // Single row batch
        let batch = RecordBatch::try_new(
            schema.clone(),
            (0..8)
                .map(|i| Arc::new(Int32Array::from(vec![i])) as ArrayRef)
                .collect(),
        )
        .unwrap();
        let session = Arc::new(Session::default());
        let write_params = WriteParams {
            session: Some(session.clone()),
            ..Default::default()
        };
        let dataset = InsertBuilder::new("memory://test")
            .with_params(&write_params)
            .execute(vec![batch])
            .await
            .unwrap();
        let fragment = dataset.get_fragments().pop().unwrap();

        // Assert file is small (< 4300 bytes)
        {
            let stats = dataset
                .table_state
                .object_store
                .as_ref()
                .io_stats_incremental();
            assert_io_eq!(stats, write_iops, 3);
            assert_io_lt!(stats, written_bytes, 4300);
        }

        // Measure IOPS needed to scan all data first time.
        let projection = Schema::try_from(schema.as_ref())
            .unwrap()
            .project_by_ids(&[0, 1, 2, 3, 4, 6, 7, 8, 9], true);
        let reader = fragment
            .open(&projection, Default::default())
            .await
            .unwrap();
        let mut data = reader
            .read_all(1024)
            .await
            .unwrap()
            .buffered(1)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(data.len(), 1);
        let data = data.pop().unwrap();
        assert_eq!(data.num_rows(), 1);
        assert_eq!(data.num_columns(), 7);

        let stats = dataset
            .table_state
            .object_store
            .as_ref()
            .io_stats_incremental();
        assert_io_eq!(stats, read_iops, 1);
        assert_io_lt!(stats, read_bytes, 4096);
    }

    #[tokio::test]
    async fn test_update_columns_with_json_extension_type() {
        use arrow_array::{StringViewArray, UInt64Array};
        use lance_arrow::ARROW_EXT_NAME_KEY;
        use lance_arrow::json::ARROW_JSON_EXT_NAME;
        use lance_core::ROW_ID;
        use std::collections::HashMap;

        // Create a dataset with an Arrow JSON extension column and a Utf8View
        // column. Both are logical types that Lance stores in a different
        // physical form (JSONB LargeBinary / Utf8), so the update path must
        // convert the right-hand stream to match.
        let test_dir = TempStrDir::default();
        let mut json_metadata = HashMap::new();
        json_metadata.insert(
            ARROW_EXT_NAME_KEY.to_string(),
            ARROW_JSON_EXT_NAME.to_string(),
        );
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int64, false),
            ArrowField::new("name", DataType::Utf8, true),
            ArrowField::new("meta", DataType::Utf8, true).with_metadata(json_metadata.clone()),
            ArrowField::new("desc", DataType::Utf8View, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e"])),
                Arc::new(StringArray::from(vec![
                    r#"{"x":1}"#,
                    r#"{"x":2}"#,
                    r#"{"x":3}"#,
                    r#"{"x":4}"#,
                    r#"{"x":5}"#,
                ])),
                Arc::new(StringViewArray::from(vec!["d1", "d2", "d3", "d4", "d5"])),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let dataset = Dataset::write(reader, test_dir.as_ref(), None)
            .await
            .unwrap();

        // Build the right stream with an Arrow JSON column (Utf8 + arrow.json
        // extension) and a Utf8View column. Only update rows with row_id 1 and 3.
        let update_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new(ROW_ID, DataType::UInt64, false),
            ArrowField::new("meta", DataType::Utf8, true).with_metadata(json_metadata),
            ArrowField::new("desc", DataType::Utf8View, true),
        ]));
        let update_batch = RecordBatch::try_new(
            update_schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![1, 3])),
                Arc::new(StringArray::from(vec![
                    r#"{"updated":true,"id":2}"#,
                    r#"{"updated":true,"id":4}"#,
                ])),
                Arc::new(StringViewArray::from(vec!["d2-new", "d4-new"])),
            ],
        )
        .unwrap();
        let right_stream: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
            vec![Ok(update_batch)],
            update_schema,
        ));

        // Perform update_columns - this should NOT fail with a type mismatch.
        // Previously the JSON column would error with:
        //   "It is not possible to interleave arrays of different data types (Utf8 and LargeBinary)"
        // and the view column would likewise mismatch (Utf8View vs stored Utf8).
        let mut fragment = dataset.get_fragment(0).unwrap();
        let (updated_fragment, fields_modified) = fragment
            .update_columns(right_stream, ROW_ID, ROW_ID)
            .await
            .unwrap();

        // Verify the operation produced valid results
        assert!(!fields_modified.is_empty());
        assert!(!updated_fragment.files.is_empty());
    }

    #[rstest]
    #[tokio::test]
    async fn test_fragment_logical_offset_overflow(
        #[values(false, true)] stable_ids: bool,
        #[values(false, true)] delete_row: bool,
        #[values(false, true)] use_session: bool,
    ) {
        let data = gen_batch()
            .col("i", array::step::<arrow_array::types::Int32Type>())
            .into_reader_rows(RowCount::from(6), lance_datagen::BatchCount::from(1));
        let mut dataset = Dataset::write(
            data,
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 3,
                enable_stable_row_ids: stable_ids,
                data_storage_version: Some(LanceFileVersion::Stable),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), 2);
        if delete_row {
            dataset.delete("i = 1").await.unwrap();
        }
        let fragment = dataset.get_fragment(0).unwrap();
        let session = fragment
            .open_session(dataset.schema(), false)
            .await
            .unwrap();
        let live = if use_session {
            session.take(&[0, 1]).await.unwrap()
        } else {
            fragment.take(&[0, 1], dataset.schema()).await.unwrap()
        };
        assert_eq!(
            live["i"]
                .as_primitive::<arrow_array::types::Int32Type>()
                .values(),
            if delete_row { &[0, 2] } else { &[0, 1] }
        );
        let error = if use_session {
            session.take(&[u32::MAX]).await.unwrap_err()
        } else {
            fragment
                .take(&[u32::MAX], dataset.schema())
                .await
                .unwrap_err()
        };
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("4294967295"), "{error}");
    }
}
