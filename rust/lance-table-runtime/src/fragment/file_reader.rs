// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

mod v1;
mod v2;

#[doc(hidden)]
pub use v1::V1FragmentReader;

use std::collections::{BTreeMap, HashSet};
use std::ops::Range;
use std::sync::Arc;

use arrow_array::{RecordBatch, new_null_array};
use arrow_schema::{DataType, Schema as ArrowSchema};
use futures::future::{BoxFuture, try_join_all};
use futures::{FutureExt, StreamExt};
use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder};
use lance_core::utils::io_stats::IoStatsRecorder;
use lance_core::{Result, datatypes::Schema};
use lance_encoding::decoder::DecoderPlugins;
use lance_file::reader::{CachedFileMetadata, FileMetadataIndex, ProjectedFileReader};
use lance_file::version::ConcreteFileVersion;
use lance_file::{LanceEncodingsIo, versions as file_versions};
use lance_io::scheduler::{FileScheduler, ScanScheduler, SchedulerConfig};
use lance_table::format::DataFile;
use lance_table::utils::stream::{ReadBatchTask, ReadBatchTaskStream};

use super::{FileFragment, FragReadConfig};

/// A trait for file readers to be implemented by both the v1 and v2 readers
///
/// The `read_*_tasks` methods are async because for v2 files they drive
/// the decode scheduler's `initialize` step (and, for small reads, the
/// synchronous scheduling that follows) before returning the stream.
/// Doing that work here keeps it on whichever task awaits this call —
/// typically a per-fragment `tokio::spawn` — instead of smuggling it
/// into the first poll of the returned stream.
#[allow(clippy::len_without_is_empty)]
pub trait GenericFileReader: std::fmt::Debug + Send + Sync {
    /// Reads the requested range of rows from the file, returning as a stream
    /// of tasks.
    fn read_range_tasks(
        &self,
        range: Range<u64>,
        batch_size: u32,
        projection: Arc<lance_core::datatypes::Schema>,
    ) -> BoxFuture<'_, Result<ReadBatchTaskStream>>;
    /// Reads the requested ranges of rows from the file, only supported by v2
    fn read_ranges_tasks(
        &self,
        ranges: Arc<[Range<u64>]>,
        batch_size: u32,
        projection: Arc<lance_core::datatypes::Schema>,
    ) -> BoxFuture<'_, Result<ReadBatchTaskStream>>;
    /// Reads all rows from the file, returning as a stream of tasks
    fn read_all_tasks(
        &self,
        batch_size: u32,
        projection: Arc<lance_core::datatypes::Schema>,
    ) -> BoxFuture<'_, Result<ReadBatchTaskStream>>;
    /// Take specific rows from the file, returning as a stream of tasks
    fn take_all_tasks(
        &self,
        indices: &[u32],
        batch_size: u32,
        projection: Arc<lance_core::datatypes::Schema>,
        take_priority: Option<u32>,
    ) -> BoxFuture<'_, Result<ReadBatchTaskStream>>;

    /// Return the number of rows in the file
    fn len(&self) -> u32;

    /// Schema of the reader
    fn projection(&self) -> &Arc<Schema>;

    /// Get storage statistics for this file (ignored by v1 reader)
    fn storage_stats(&self) -> Result<Vec<(u32, u64)>>;

    // Clone the reader, this is needed because Box<dyn Foo: Clone> doesn't
    // implement Clone
    fn clone_box(&self) -> Box<dyn GenericFileReader>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetadataMode {
    LazyAllowed,
    Full,
}

impl FileFragment {
    pub(crate) async fn open_reader(
        &self,
        data_file: &DataFile,
        projection: Option<&Schema>,
        read_config: &FragReadConfig,
        io_stats: Option<Arc<dyn IoStatsRecorder>>,
    ) -> Result<Option<Box<dyn GenericFileReader>>> {
        self.open_reader_impl(
            data_file,
            projection,
            read_config,
            MetadataMode::LazyAllowed,
            io_stats,
        )
        .await
    }

    /// Open a file with the full metadata required for storage statistics.
    #[doc(hidden)]
    pub async fn open_reader_with_full_metadata(
        &self,
        data_file: &DataFile,
        projection: Option<&Schema>,
        read_config: &FragReadConfig,
    ) -> Result<Option<Box<dyn GenericFileReader>>> {
        self.open_reader_impl(data_file, projection, read_config, MetadataMode::Full, None)
            .await
    }

    fn open_reader_impl<'a>(
        &'a self,
        data_file: &'a DataFile,
        projection: Option<&'a Schema>,
        read_config: &'a FragReadConfig,
        metadata_mode: MetadataMode,
        io_stats: Option<Arc<dyn IoStatsRecorder>>,
    ) -> BoxFuture<'a, Result<Option<Box<dyn GenericFileReader>>>> {
        async move {
            match data_file.file_version()? {
                ConcreteFileVersion::V1 => Ok(self
                    .open_v1_reader(data_file, projection)
                    .await?
                    .map(|reader| Box::new(reader) as Box<dyn GenericFileReader>)),
                ConcreteFileVersion::V2_0
                | ConcreteFileVersion::V2_1
                | ConcreteFileVersion::V2_2
                | ConcreteFileVersion::V2_3 => {
                    self.open_current_file_reader(
                        data_file,
                        projection,
                        read_config,
                        metadata_mode,
                        io_stats,
                    )
                    .await
                }
            }
        }
        .boxed()
    }

    async fn open_current_file_reader(
        &self,
        data_file: &DataFile,
        projection: Option<&Schema>,
        read_config: &FragReadConfig,
        metadata_mode: MetadataMode,
        io_stats: Option<Arc<dyn IoStatsRecorder>>,
    ) -> Result<Option<Box<dyn GenericFileReader>>> {
        let full_schema = self.table.schema();
        let data_file_schema = Arc::new(data_file.schema(full_schema));
        let projection = projection.unwrap_or(full_schema);
        let schema_per_file =
            Arc::new(data_file_projection(projection, data_file_schema.as_ref())?);
        if schema_per_file.fields.is_empty() {
            return Ok(None);
        }

        let path = self
            .table
            .data_file_dir(data_file)?
            .join(data_file.path.as_str());
        let (store_scheduler, reader_priority) = if let Some(base_id) = data_file.base_id {
            let object_store = self.table.object_store(Some(base_id)).await?;
            // Reuse one scheduler per base for the scan's lifetime so the
            // scheduler budget scales with the number of bases, not files. When
            // there is no scan cache (a one-off read), fall back to a dedicated
            // scheduler for this file.
            let scheduler = match read_config.base_schedulers.as_ref() {
                Some(cache) => cache.get_or_create(base_id, object_store),
                None => ScanScheduler::new(
                    object_store.clone(),
                    SchedulerConfig::max_bandwidth(&object_store),
                ),
            };
            (scheduler, read_config.reader_priority.unwrap_or(0))
        } else if let Some(scan_scheduler) = read_config.scan_scheduler.as_ref() {
            (
                scan_scheduler.clone(),
                read_config.reader_priority.unwrap_or(0),
            )
        } else {
            (
                ScanScheduler::new(
                    self.table.state.object_store.clone(),
                    SchedulerConfig::max_bandwidth(&self.table.state.object_store),
                ),
                0,
            )
        };
        let file_reader_options = read_config
            .file_reader_options
            .clone()
            .or_else(|| self.table.state.file_reader_options.as_deref().cloned())
            .unwrap_or_default();
        let file_scheduler = if file_reader_options.decoder_config.list_structure.is_empty() {
            store_scheduler
                .open_file_with_priority(&path, reader_priority as u64, &data_file.file_size_bytes)
                .await?
        } else {
            // Even a small file must remain a range reader: SmallReader fetches
            // its entire body on the first metadata request.
            let reader = self
                .table
                .object_store(data_file.base_id)
                .await?
                .open(&path)
                .await?;
            // Current-format footers occupy 40 bytes. A full tail block may
            // otherwise fetch image payload before the structural read begins.
            store_scheduler
                .open_reader(reader.into())
                .with_priority(reader_priority as u64)
                .with_block_size(0)
                .with_metadata_read_size(lance_file::reader::FOOTER_LEN)?
        };
        let file_scheduler = match io_stats {
            Some(stats) => file_scheduler.with_io_stats(stats),
            None => file_scheduler,
        };
        let path = file_scheduler.reader().path().clone();
        let metadata_cache = self.table.state.metadata_cache.file_metadata_cache(&path);
        let field_id_to_column_idx = Arc::new(BTreeMap::from_iter(
            data_file
                .fields
                .iter()
                .copied()
                .zip(data_file.column_indices.iter().copied())
                .filter_map(|(field_id, column_index)| {
                    (column_index >= 0).then_some((field_id as u32, column_index as u32))
                }),
        ));
        let file_version = data_file.file_version()?;
        // Blob projections always open the reader with the full footer. A 2.2 file reads
        // it first, to see which of its blob columns keep the legacy layout.
        let blob_file_metadata = if file_version == ConcreteFileVersion::V2_2
            && schema_per_file
                .fields_pre_order()
                .any(|field| field.is_blob_v2())
        {
            Some(self.get_file_metadata(&file_scheduler).await?)
        } else {
            None
        };
        let legacy_blobs = match file_version {
            ConcreteFileVersion::V2_0 | ConcreteFileVersion::V2_1 => schema_per_file
                .fields_pre_order()
                .filter(|field| field.is_blob_v2())
                .map(|field| field.id)
                .collect(),
            _ => blob_file_metadata
                .as_ref()
                .map_or_else(HashSet::new, |file_metadata| {
                    v2::legacy_blob_fields(
                        &schema_per_file,
                        &field_id_to_column_idx,
                        &file_metadata.column_infos,
                    )
                }),
        };
        let physical_schema = v2::legacy_blob_read_schema(&schema_per_file, &legacy_blobs);
        let reader_projection = file_versions::reader_projection_from_field_ids(
            file_version,
            &physical_schema,
            field_id_to_column_idx.as_ref(),
        )?;
        let prefer_indexed = metadata_mode == MetadataMode::LazyAllowed
            && reader_projection.column_indices.len().saturating_mul(4)
                < data_file
                    .column_indices
                    .iter()
                    .filter(|column_index| **column_index >= 0)
                    .count();
        let known_schema = self
            .metadata
            .physical_rows
            .map(|num_rows| (data_file_schema.clone(), num_rows as u64));

        let encodings_io = Arc::new(
            LanceEncodingsIo::new(file_scheduler.clone())
                .with_read_chunk_size(file_reader_options.read_chunk_size),
        );
        let reader = file_versions::open_projected_reader(
            file_version,
            &reader_projection,
            prefer_indexed,
            || async {
                let metadata_index = self
                    .get_file_metadata_index(&file_scheduler, known_schema.clone())
                    .await?;
                if (reader_projection.column_indices.len() as u32).saturating_mul(4)
                    >= metadata_index.num_columns()
                {
                    return Ok(None);
                }
                Ok(Some(
                    ProjectedFileReader::try_open_with_metadata_index(
                        encodings_io.clone(),
                        path.clone(),
                        Some(reader_projection.clone()),
                        Arc::<DecoderPlugins>::default(),
                        metadata_index,
                        &metadata_cache,
                        file_reader_options.clone(),
                    )
                    .await?,
                ))
            },
            || async {
                let file_metadata = match blob_file_metadata {
                    Some(file_metadata) => file_metadata,
                    None => self.get_file_metadata(&file_scheduler).await?,
                };
                ProjectedFileReader::try_open_with_file_metadata(
                    encodings_io.clone(),
                    path.clone(),
                    None,
                    Arc::<DecoderPlugins>::default(),
                    file_metadata,
                    &metadata_cache,
                    file_reader_options.clone(),
                )
                .await
            },
        )
        .await?;
        Ok(Some(Box::new(v2::Reader::new(
            Arc::new(reader),
            schema_per_file,
            field_id_to_column_idx,
            Arc::new(legacy_blobs),
            reader_priority,
            file_scheduler,
        ))))
    }

    pub(super) async fn open_readers(
        &self,
        projection: &Schema,
        read_config: &FragReadConfig,
        io_stats: Option<Arc<dyn IoStatsRecorder>>,
    ) -> Result<Vec<Box<dyn GenericFileReader>>> {
        self.open_readers_impl(projection, read_config, MetadataMode::LazyAllowed, io_stats)
            .await
    }

    /// The id of `field` and of every field under it.
    fn collect_field_ids(field: &lance_core::datatypes::Field, out: &mut HashSet<i32>) {
        out.insert(field.id);
        for child in &field.children {
            Self::collect_field_ids(child, out);
        }
    }

    /// Open the base data files with the full metadata required for storage statistics.
    #[doc(hidden)]
    pub async fn open_readers_with_full_metadata(
        &self,
        projection: &Schema,
        read_config: &FragReadConfig,
    ) -> Result<Vec<Box<dyn GenericFileReader>>> {
        self.open_readers_impl(projection, read_config, MetadataMode::Full, None)
            .await
    }

    fn open_readers_impl<'a>(
        &'a self,
        projection: &'a Schema,
        read_config: &'a FragReadConfig,
        metadata_mode: MetadataMode,
        io_stats: Option<Arc<dyn IoStatsRecorder>>,
    ) -> BoxFuture<'a, Result<Vec<Box<dyn GenericFileReader>>>> {
        async move {
            // A fragment written through several `add_columns` holds one file
            // per column group. Projecting the full schema through every file
            // only to learn that it holds none of the requested fields costs
            // more CPU than the read itself on a wide table, so test the ids
            // first. A projected struct or list takes its validity and offsets
            // from whichever of its leaves a file holds, so every leaf under a
            // projected parent counts as wanted even when it is not projected.
            let mut wanted_ids: HashSet<i32> = projection.field_ids().into_iter().collect();
            for parent in projection
                .fields_pre_order()
                .filter(|field| !field.children.is_empty())
            {
                if let Some(full) = self.table.schema().field_by_id(parent.id) {
                    Self::collect_field_ids(full, &mut wanted_ids);
                }
            }
            // Each open is at least one object-store round trip for the file's
            // metadata; a fragment with several data files (one per
            // `add_columns` / merge) would otherwise pay them back to back.
            // `try_join_all` keeps the readers in data-file order.
            let opens = self
                .metadata
                .files
                .iter()
                .filter(|data_file| data_file.fields.iter().any(|id| wanted_ids.contains(id)))
                .map(|data_file| {
                    let io_stats = io_stats.clone();
                    async move {
                        match metadata_mode {
                            MetadataMode::LazyAllowed => {
                                self.open_reader(data_file, Some(projection), read_config, io_stats)
                                    .await
                            }
                            MetadataMode::Full => {
                                self.open_reader_with_full_metadata(
                                    data_file,
                                    Some(projection),
                                    read_config,
                                )
                                .await
                            }
                        }
                    }
                });
            let mut opened_files: Vec<Box<dyn GenericFileReader>> =
                try_join_all(opens).await?.into_iter().flatten().collect();

            // This should return immediately on modern datasets.  Need to use physical_rows because
            // deletions will be applied later
            let num_rows = self.physical_rows().await?;

            // Check if there are any fields that are not in any data files
            let field_ids_in_files = opened_files
                .iter()
                .flat_map(|r| r.projection().fields_pre_order().map(|f| f.id))
                .filter(|id| *id >= 0)
                .collect::<HashSet<_>>();
            let mut missing_fields = projection.field_ids();
            missing_fields.retain(|f| !field_ids_in_files.contains(f) && *f >= 0);
            if !missing_fields.is_empty() {
                let missing_projection = projection.project_by_ids(&missing_fields, true);
                let null_reader = NullReader::new(Arc::new(missing_projection), num_rows as u32);
                opened_files.push(Box::new(null_reader));
            }

            Ok(opened_files)
        }
        .boxed()
    }

    /// Get the file metadata for this fragment, using the cache if available.
    pub async fn get_file_metadata(
        &self,
        file_scheduler: &FileScheduler,
    ) -> Result<Arc<CachedFileMetadata>> {
        let path = file_scheduler.reader().path();
        let cache = self.table.state.metadata_cache.file_metadata_cache(path);

        let file_metadata = cache
            .get_or_insert_with_key(FileMetadataCacheKey, || async {
                let file_metadata: CachedFileMetadata =
                    lance_file::reader::FileReader::read_all_metadata(file_scheduler).await?;
                Ok(file_metadata)
            })
            .await?;
        Ok(file_metadata)
    }

    pub async fn get_file_metadata_index(
        &self,
        file_scheduler: &FileScheduler,
        known_schema: Option<(Arc<Schema>, u64)>,
    ) -> Result<Arc<FileMetadataIndex>> {
        let path = file_scheduler.reader().path();
        let cache = self.table.state.metadata_cache.file_metadata_cache(path);

        let metadata_index = cache
            .get_or_insert_with_key(FileMetadataIndexCacheKey, || async {
                let metadata_index = if let Some((file_schema, num_rows)) = known_schema {
                    lance_file::reader::FileReader::read_metadata_index_with_schema(
                        file_scheduler,
                        file_schema,
                        num_rows,
                    )
                    .await?
                } else {
                    lance_file::reader::FileReader::read_metadata_index(file_scheduler).await?
                };
                Ok(metadata_index)
            })
            .await?;
        Ok(metadata_index)
    }
}

/// A reader where all rows are null. Used when there are fields that have no
/// data files in a fragment.
#[derive(Debug, Clone)]
struct NullReader {
    schema: Arc<Schema>,
    num_rows: u32,
}

impl NullReader {
    fn new(schema: Arc<Schema>, num_rows: u32) -> Self {
        Self { schema, num_rows }
    }

    /// The Arrow schema of the placeholder batches for `projection`.
    ///
    /// When a nested child is added to an existing parent, the parent lands
    /// here as well (all-null) even though the dataset schema may declare it
    /// NOT NULL; the later merge with the data files rebuilds it from the
    /// parent's real data. Mark the placeholder fields nullable so the batch
    /// is not rejected for violating a constraint the merged result satisfies.
    fn nullable_schema(projection: &Schema) -> Arc<ArrowSchema> {
        let schema = ArrowSchema::from(projection);
        let fields = schema
            .fields()
            .iter()
            .map(|f| f.as_ref().clone().with_nullable(true))
            .collect::<Vec<_>>();
        Arc::new(ArrowSchema::new_with_metadata(fields, schema.metadata))
    }

    fn batch(projection: Arc<ArrowSchema>, num_rows: usize) -> RecordBatch {
        let columns = projection
            .fields()
            .iter()
            .map(|f| new_null_array(f.data_type(), num_rows))
            .collect::<Vec<_>>();
        RecordBatch::try_new(projection, columns).unwrap()
    }
}

impl GenericFileReader for NullReader {
    fn read_range_tasks(
        &self,
        range: Range<u64>,
        batch_size: u32,
        projection: Arc<Schema>,
    ) -> BoxFuture<'_, Result<ReadBatchTaskStream>> {
        self.read_ranges_tasks(vec![range].into(), batch_size, projection)
    }

    fn read_ranges_tasks(
        &self,
        ranges: Arc<[Range<u64>]>,
        batch_size: u32,
        projection: Arc<Schema>,
    ) -> BoxFuture<'_, Result<ReadBatchTaskStream>> {
        let mut remaining_rows = ranges.iter().map(|r| r.end - r.start).sum::<u64>();
        let projection = Self::nullable_schema(projection.as_ref());

        let task_iter = std::iter::from_fn(move || {
            if remaining_rows == 0 {
                return None;
            }

            let num_rows = remaining_rows.min(batch_size as u64) as usize;
            remaining_rows -= num_rows as u64;
            let batch = Self::batch(projection.clone(), num_rows);
            let task = ReadBatchTask {
                task: futures::future::ready(Ok(batch)).boxed(),
                num_rows: num_rows as u32,
            };
            Some(task)
        });

        async move { Ok(futures::stream::iter(task_iter).boxed()) }.boxed()
    }

    fn read_all_tasks(
        &self,
        batch_size: u32,
        projection: Arc<Schema>,
    ) -> BoxFuture<'_, Result<ReadBatchTaskStream>> {
        self.read_ranges_tasks(vec![0..self.num_rows as u64].into(), batch_size, projection)
    }

    fn take_all_tasks(
        &self,
        indices: &[u32],
        batch_size: u32,
        projection: Arc<Schema>,
        _take_priority: Option<u32>,
    ) -> BoxFuture<'_, Result<ReadBatchTaskStream>> {
        let num_rows = indices.len() as u64;
        self.read_ranges_tasks(vec![0..num_rows].into(), batch_size, projection)
    }

    fn storage_stats(&self) -> Result<Vec<(u32, u64)>> {
        // No-op for null reader
        Ok(Vec::new())
    }

    fn projection(&self) -> &Arc<Schema> {
        &self.schema
    }

    fn len(&self) -> u32 {
        self.num_rows
    }

    fn clone_box(&self) -> Box<dyn GenericFileReader> {
        Box::new(self.clone())
    }
}

/// `projection` intersected with a data file's schema. A nested field none of
/// whose projected children are in the file keeps one file leaf, which still
/// carries its validity and offsets; the final projection drops it.
fn data_file_projection(projection: &Schema, file_schema: &Schema) -> Result<Schema> {
    fn keep_a_leaf(field: &mut lance_core::datatypes::Field, file_schema: &Schema) -> Result<()> {
        if field.is_blob() || field.is_packed_struct() {
            return Ok(());
        }
        if !field.children.is_empty() {
            return field
                .children
                .iter_mut()
                .try_for_each(|child| keep_a_leaf(child, file_schema));
        }
        let Some(file_field) = file_schema.field_by_id(field.id) else {
            return Ok(());
        };
        if file_field.children.is_empty() {
            return Ok(());
        }
        let path = file_field.validity_leaf_path().unwrap_or_else(|| {
            // No fixed-width leaf: any leaf carries the validity.
            let mut path = vec![];
            let mut node = file_field;
            while !(node.is_blob()
                || node.is_packed_struct()
                || matches!(node.data_type(), DataType::Map(..)))
                && let Some(child) = node.children.first()
            {
                path.push(child.name.as_str());
                node = child;
            }
            path
        });
        *field = file_field.project(&path)?;
        Ok(())
    }
    let mut schema = projection.intersection_ignore_types(file_schema)?;
    for field in &mut schema.fields {
        keep_a_leaf(field, file_schema)?;
    }
    Ok(schema)
}

// Cache key for file metadata
#[derive(Debug, Clone)]
struct FileMetadataCacheKey;

impl CacheKey for FileMetadataCacheKey {
    type ValueType = CachedFileMetadata;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        "".into()
    }

    fn type_name() -> &'static str {
        "FileMetadata"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.dataset.fragment-file-metadata-key", 1)
    }

    fn write_key(&self, _builder: &mut KeyBuilder) {}
}

#[derive(Debug, Clone)]
struct FileMetadataIndexCacheKey;

impl CacheKey for FileMetadataIndexCacheKey {
    type ValueType = FileMetadataIndex;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        "metadata_index".into()
    }

    fn type_name() -> &'static str {
        "FileMetadataIndex"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.dataset.fragment-file-metadata-index-key", 1)
    }

    fn write_key(&self, _builder: &mut KeyBuilder) {}
}
