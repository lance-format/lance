// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::num::NonZero;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use bytes::Bytes;
use lance_core::{
    Error, Result,
    datatypes::{Field, Schema},
};
use lance_file::version::ConcreteFileVersion;
use lance_file::versions::v1::writer::{
    FileWriter as V1FileWriter, ManifestProvider as V1ManifestProvider,
};
use lance_file::{
    versions as file_versions,
    writer::{self as current_writer, FileWriteSummary, FileWriter, FileWriterOptions},
};
use lance_io::object_store::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry};
use lance_io::traits::Writer;
use lance_table::{
    feature_flags::managed_blobs_enabled, format::DataFile, io::manifest::ManifestDescribing,
};
use object_store::ObjectStoreExt;
use object_store::path::Path;

use super::bases::blob_v2_external_base_resolver;
use super::schema::validate_decimal32_and_decimal64;
use super::schema_adapter::SchemaAdapter;
use super::{ExternalBlobMode, WriteParams};
use crate::Table;
use crate::blob::{
    BlobPayloadInventory, BlobPreprocessor, BlobWriteSources, ExternalBaseResolver, SidecarRewrite,
};
use crate::fragment::write::generate_random_filename;
use crate::paths::DATA_DIR;

#[async_trait::async_trait]
pub trait GenericWriter: Send {
    /// Write the given batches to the file
    async fn write(&mut self, batches: &[RecordBatch]) -> Result<()>;

    #[doc(hidden)]
    async fn write_with_blob_sources(
        &mut self,
        batch: &RecordBatch,
        _root: &Path,
        sources: &BlobWriteSources,
    ) -> Result<()> {
        if !sources.is_empty() {
            return Err(Error::not_supported(
                "This writer does not accept Blob sources",
            ));
        }
        self.write(std::slice::from_ref(batch)).await
    }

    /// Mark the next batch as ending the current file page.
    fn prepare_for_flush(&mut self) {}
    /// Get the file path and base ID for the data file being written.
    fn data_file_path(&self) -> (&str, Option<u32>);
    /// Get the current position in the file
    ///
    /// We use this to know when the file is too large and we need to start
    /// a new file
    async fn tell(&mut self) -> Result<u64>;
    /// Finish writing the file (flush the remaining data and write footer)
    async fn finish(&mut self) -> Result<(u32, DataFile)>;

    /// Append `array` to a single top-level column, leaving the others where
    /// they are.
    ///
    /// `column_index` is the column's position in the writer's schema. Columns
    /// advance independently, so a file written this way can end with a
    /// different number of rows in each column, and the row count
    /// [`finish`](Self::finish) reports is then the longest column rather than a
    /// count every column shares. That raggedness is the point: it is what lets
    /// an overlay carry a different subset of cells per field. Callers that want
    /// every column to stay aligned should use [`Self::write`] instead.
    ///
    /// Only supported on V2 files without blob preprocessing. This is a
    /// defaulted method rather than a required one so that the writers which
    /// cannot offer it -- and any implementor outside this crate -- keep
    /// compiling and reject it at runtime instead.
    async fn write_column(&mut self, _column_index: usize, _array: ArrayRef) -> Result<()> {
        Err(Error::not_supported(
            "writing a single column: this writer only accepts whole batches",
        ))
    }

    /// Add a global buffer to the current file. Returns the 1-based buffer index.
    /// Must be called before `finish`. No-op on legacy (V1) files (returns `Ok(1)`).
    async fn add_global_buffer(&mut self, _buffer: Bytes) -> Result<u32> {
        Ok(1)
    }

    /// Add a key-value pair to the file's schema metadata.
    /// Must be called before `finish`. No-op on legacy (V1) files.
    fn add_schema_metadata(&mut self, _key: String, _value: String) {}
}

struct V1WriterAdapter<M>
where
    M: V1ManifestProvider + Send + Sync,
{
    writer: V1FileWriter<M>,
    path: String,
    base_id: Option<u32>,
}

#[async_trait::async_trait]
impl<M> GenericWriter for V1WriterAdapter<M>
where
    M: V1ManifestProvider + Send + Sync,
{
    async fn write(&mut self, batches: &[RecordBatch]) -> Result<()> {
        self.writer.write(batches).await
    }
    fn data_file_path(&self) -> (&str, Option<u32>) {
        (&self.path, self.base_id)
    }
    async fn tell(&mut self) -> Result<u64> {
        Ok(self.writer.tell().await? as u64)
    }
    async fn finish(&mut self) -> Result<(u32, DataFile)> {
        let summary = self.writer.finish().await?;
        Ok((
            u32::try_from(summary.num_rows).map_err(|_| {
                Error::internal(format!(
                    "data file has {} rows, exceeding the fragment capacity of {} rows",
                    summary.num_rows,
                    u32::MAX
                ))
            })?,
            DataFile::new_legacy(
                self.path.clone(),
                self.writer.schema(),
                NonZero::new(summary.size_bytes),
                self.base_id,
            ),
        ))
    }
}

/// Writes dataset data files in the current (V2) file formats.
///
/// Every current-format data file a dataset writes goes through this type, so
/// it is the single write boundary: each batch is converted from the
/// representation the caller supplied to the one Lance stores (Arrow JSON to
/// Lance JSONB, view arrays to offset arrays) before blob
/// preprocessing and encoding. The file writer then rejects any array that
/// still differs from the file schema.
pub(super) struct V2WriterAdapter {
    writer: current_writer::FileWriter,
    data_file: Option<DataFile>,
    preprocessor: Option<BlobPreprocessor>,
    promotion: Option<FilePromotion>,
}

pub(super) struct FilePromotion {
    object_store: ObjectStore,
    staging_path: Path,
    final_path: Path,
}

impl FilePromotion {
    async fn promote(&self) -> Result<()> {
        self.object_store
            .copy_if_not_exists_or_same_size(&self.staging_path, &self.final_path)
            .await?;
        self.object_store
            .inner
            .delete(&self.staging_path)
            .await
            .map_err(|error| Error::io(error.to_string()))?;
        Ok(())
    }

    async fn abort(&self) {
        let _ = self.object_store.inner.delete(&self.staging_path).await;
    }
}

impl V2WriterAdapter {
    /// `data_file` describes the file being written and is completed by
    /// [`GenericWriter::finish`]. Writers that describe their output
    /// themselves pass `None` and use [`Self::finish_file`].
    pub(crate) fn new(
        writer: current_writer::FileWriter,
        data_file: Option<DataFile>,
        preprocessor: Option<BlobPreprocessor>,
        promotion: Option<FilePromotion>,
    ) -> Self {
        Self {
            writer,
            data_file,
            preprocessor,
            promotion,
        }
    }

    pub(crate) async fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        self.write_batch_with_sources(batch, None).await
    }

    pub(super) async fn write_batch_with_sources(
        &mut self,
        batch: &RecordBatch,
        sources: Option<(&Path, &BlobWriteSources)>,
    ) -> Result<()> {
        let batch = SchemaAdapter::new(batch.schema()).to_physical_batch(batch.clone())?;
        let batch = match self.preprocessor.as_mut() {
            Some(pre) => pre.preprocess_batch_with_sources(&batch, sources).await?,
            None if sources.is_some_and(|(_, sources)| !sources.is_empty()) => {
                return Err(Error::invalid_input("Blob sources require a Blob v2 field"));
            }
            None => batch,
        };
        self.writer.write_batch(&batch).await
    }

    /// Finish the blob sidecars and the data file.
    pub(crate) async fn finish_file(&mut self) -> Result<FileWriteSummary> {
        let result = async {
            if let Some(pre) = self.preprocessor.as_mut() {
                pre.finish().await?;
            }
            let summary = self.writer.finish().await?;
            if let Some(promotion) = &self.promotion {
                promotion.promote().await?;
            }
            Ok(summary)
        }
        .await;
        if result.is_err()
            && let Some(promotion) = &self.promotion
        {
            promotion.abort().await;
        }
        result
    }

    /// Abandon the data file and any blob sidecars in progress.
    pub(crate) async fn abort(&mut self) {
        self.writer.abort().await;
        if let Some(pre) = self.preprocessor.as_mut() {
            pre.abort();
        }
        if let Some(promotion) = &self.promotion {
            promotion.abort().await;
        }
    }
}

#[async_trait::async_trait]
impl GenericWriter for V2WriterAdapter {
    async fn write(&mut self, batches: &[RecordBatch]) -> Result<()> {
        for batch in batches {
            self.write_batch(batch).await?;
        }
        Ok(())
    }

    async fn write_with_blob_sources(
        &mut self,
        batch: &RecordBatch,
        root: &Path,
        sources: &BlobWriteSources,
    ) -> Result<()> {
        self.write_batch_with_sources(batch, Some((root, sources)))
            .await
    }

    fn prepare_for_flush(&mut self) {
        self.writer.prepare_for_flush();
    }
    async fn write_column(&mut self, column_index: usize, array: ArrayRef) -> Result<()> {
        if self.preprocessor.is_some() {
            // The preprocessor rewrites a batch as a whole, splitting blob
            // values out to a sidecar and replacing them with descriptions; it
            // has no meaning applied to one column in isolation.
            return Err(Error::not_supported(
                "writing a single column: this file stores blob data in a sidecar",
            ));
        }
        self.writer.write_column(column_index, array).await
    }
    fn data_file_path(&self) -> (&str, Option<u32>) {
        self.data_file
            .as_ref()
            .map(|data_file| (data_file.path.as_str(), data_file.base_id))
            .unwrap_or(("", None))
    }
    async fn tell(&mut self) -> Result<u64> {
        Ok(self.writer.tell().await?)
    }
    async fn finish(&mut self) -> Result<(u32, DataFile)> {
        let field_ids = self
            .writer
            .field_id_to_column_indices()
            .iter()
            .map(|(field_id, _)| *field_id as i32)
            .collect::<Vec<_>>();
        let column_indices = self
            .writer
            .field_id_to_column_indices()
            .iter()
            .map(|(_, column_index)| *column_index as i32)
            .collect::<Vec<_>>();
        let write_summary = self.finish_file().await?;
        let mut data_file = self
            .data_file
            .take()
            .ok_or_else(|| Error::internal("current writer was already finished"))?;
        data_file.fields = field_ids.into();
        data_file.column_indices = column_indices.into();
        data_file.file_size_bytes = NonZero::new(write_summary.size_bytes).into();
        let num_rows = u32::try_from(write_summary.num_rows).map_err(|_| {
            Error::internal(format!(
                "data file has {} rows, exceeding the fragment capacity of {} rows",
                write_summary.num_rows,
                u32::MAX
            ))
        })?;
        Ok((num_rows, data_file))
    }

    async fn add_global_buffer(&mut self, buffer: Bytes) -> Result<u32> {
        self.writer.add_global_buffer(buffer).await
    }

    fn add_schema_metadata(&mut self, key: String, value: String) {
        self.writer.add_schema_metadata(key, value);
    }
}

#[derive(Clone, Default)]
pub struct WriterOptions {
    pub(super) add_data_dir: bool,
    pub(super) base_id: Option<u32>,
    pub(super) preassigned_data_file_name: Option<Arc<String>>,
    pub(super) external_base_resolver: Option<Arc<ExternalBaseResolver>>,
    pub(super) allow_external_blob_outside_bases: bool,
    pub(super) external_blob_mode: ExternalBlobMode,
    pub(super) source_store_registry: Arc<ObjectStoreRegistry>,
    pub(super) source_store_params: ObjectStoreParams,
    pub(super) blob_pack_file_size_threshold: Option<usize>,
    pub(super) file_writer_options: FileWriterOptions,
    pub(super) created_blob_payloads: Option<BlobPayloadInventory>,
    pub(super) sidecar_rewrite: Option<Arc<SidecarRewrite>>,
}

impl WriterOptions {
    pub(super) fn update(
        source_store_registry: Arc<ObjectStoreRegistry>,
        external_base_resolver: Option<Arc<ExternalBaseResolver>>,
        allow_external_blob_outside_bases: bool,
        file_writer_options: FileWriterOptions,
    ) -> Self {
        Self {
            add_data_dir: true,
            external_base_resolver,
            allow_external_blob_outside_bases,
            source_store_registry,
            file_writer_options,
            ..Default::default()
        }
    }

    /// Options for data files written from `params`, under the base's `data/`
    /// directory. Writes into a registered base override `add_data_dir` and
    /// `base_id`; dataset writes override the source store registry.
    pub(crate) fn from_params(
        params: &WriteParams,
        external_base_resolver: Option<Arc<ExternalBaseResolver>>,
    ) -> Self {
        Self {
            add_data_dir: true,
            external_base_resolver,
            allow_external_blob_outside_bases: params.allow_external_blob_outside_bases,
            external_blob_mode: params.external_blob_mode,
            source_store_registry: params.store_registry(),
            source_store_params: params.store_params.clone().unwrap_or_default(),
            blob_pack_file_size_threshold: params.blob_pack_file_size_threshold,
            file_writer_options: params.file_writer_options.clone().unwrap_or_default(),
            ..Default::default()
        }
    }
}

pub(super) async fn open_v1_writer(
    object_store: &ObjectStore,
    schema: &Schema,
    base_dir: &Path,
    options: WriterOptions,
) -> Result<Box<dyn GenericWriter>> {
    let WriterOptions {
        add_data_dir,
        base_id,
        preassigned_data_file_name,
        ..
    } = options;
    let (_data_file_key, filename, _data_dir, full_path) = prepare_data_file_path(
        base_dir,
        add_data_dir,
        preassigned_data_file_name.as_deref().map(String::as_str),
    );
    Ok(Box::new(V1WriterAdapter {
        writer: V1FileWriter::<ManifestDescribing>::try_new(
            object_store,
            &full_path,
            schema.clone(),
            &Default::default(),
        )
        .await?,
        path: filename,
        base_id,
    }))
}

pub(super) async fn open_current_writer<F>(
    create_file_writer: F,
    object_store: &ObjectStore,
    schema: &Schema,
    base_dir: &Path,
    options: WriterOptions,
) -> Result<Box<dyn GenericWriter>>
where
    F: FnOnce(
        Box<dyn Writer>,
        Schema,
        String,
        Option<u32>,
        FileWriterOptions,
    ) -> Result<(current_writer::FileWriter, DataFile)>,
{
    let WriterOptions {
        add_data_dir,
        base_id,
        preassigned_data_file_name,
        file_writer_options,
        ..
    } = options;
    let (_data_file_key, filename, data_dir, final_path) = prepare_data_file_path(
        base_dir,
        add_data_dir,
        preassigned_data_file_name.as_deref().map(String::as_str),
    );
    let (writer_path, promotion_target) =
        staged_writer_path(data_dir, final_path, preassigned_data_file_name.is_some());
    let writer = object_store.create(&writer_path).await?;
    let promotion = promotion_target.map(|final_path| FilePromotion {
        object_store: object_store.clone(),
        staging_path: writer_path,
        final_path,
    });
    let (file_writer, data_file) = create_file_writer(
        writer,
        schema.clone(),
        filename,
        base_id,
        file_writer_options,
    )?;
    Ok(Box::new(V2WriterAdapter::new(
        file_writer,
        Some(data_file),
        None,
        promotion,
    )))
}

pub(super) async fn open_current_blob_v2_writer<F>(
    version: ConcreteFileVersion,
    create_file_writer: F,
    object_store: &ObjectStore,
    schema: &Schema,
    base_dir: &Path,
    options: WriterOptions,
) -> Result<Box<dyn GenericWriter>>
where
    F: FnOnce(
        Box<dyn Writer>,
        Schema,
        String,
        Option<u32>,
        FileWriterOptions,
    ) -> Result<(current_writer::FileWriter, DataFile)>,
{
    let WriterOptions {
        add_data_dir,
        base_id,
        preassigned_data_file_name,
        external_base_resolver,
        allow_external_blob_outside_bases,
        external_blob_mode,
        source_store_registry,
        source_store_params,
        blob_pack_file_size_threshold,
        file_writer_options,
        created_blob_payloads,
        sidecar_rewrite,
    } = options;
    let (data_file_key, filename, data_dir, final_path) = prepare_data_file_path(
        base_dir,
        add_data_dir,
        preassigned_data_file_name.as_deref().map(String::as_str),
    );
    let (writer_path, promotion_target) = staged_writer_path(
        data_dir.clone(),
        final_path,
        preassigned_data_file_name.is_some(),
    );
    let writer = object_store.create(&writer_path).await?;
    let promotion = promotion_target.map(|final_path| FilePromotion {
        object_store: object_store.clone(),
        staging_path: writer_path,
        final_path,
    });
    let (file_writer, data_file) = create_file_writer(
        writer,
        schema.clone(),
        filename,
        base_id,
        file_writer_options,
    )?;
    let mut preprocessor = BlobPreprocessor::new(
        object_store.clone(),
        data_dir,
        data_file_key,
        schema,
        external_base_resolver,
        allow_external_blob_outside_bases,
        external_blob_mode,
        source_store_registry,
        source_store_params,
        blob_pack_file_size_threshold,
    )?;
    preprocessor = preprocessor.with_sidecar_rewrite(sidecar_rewrite);
    if managed_blobs_enabled(schema)?
        && matches!(
            version,
            ConcreteFileVersion::V2_2 | ConcreteFileVersion::V2_3
        )
    {
        preprocessor = preprocessor
            .with_managed_base(base_id, base_dir.clone())
            .with_created_payloads(created_blob_payloads);
    }
    Ok(Box::new(V2WriterAdapter::new(
        file_writer,
        Some(data_file),
        Some(preprocessor),
        promotion,
    )))
}

fn prepare_data_file_path(
    base_dir: &Path,
    add_data_dir: bool,
    preassigned_data_file_name: Option<&str>,
) -> (String, String, Path, Path) {
    let filename = preassigned_data_file_name
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| format!("{}.lance", generate_random_filename()));
    let data_file_key = filename
        .strip_suffix(".lance")
        .unwrap_or(filename.as_str())
        .to_string();
    let data_dir = if add_data_dir {
        base_dir.clone().join(DATA_DIR)
    } else {
        base_dir.clone()
    };
    let full_path = data_dir.clone().join(filename.as_str());
    (data_file_key, filename, data_dir, full_path)
}

fn staged_writer_path(data_dir: Path, final_path: Path, stage: bool) -> (Path, Option<Path>) {
    if !stage {
        return (final_path, None);
    }
    let staging_path = data_dir.join(format!("{}.lance-stage", generate_random_filename()));
    (staging_path, Some(final_path))
}

fn create_current_file_writer(
    version: ConcreteFileVersion,
    object_writer: Box<dyn Writer>,
    schema: Schema,
    filename: String,
    base_id: Option<u32>,
    options: FileWriterOptions,
) -> Result<(FileWriter, DataFile)> {
    let writer = file_versions::create_writer(version, object_writer, schema, options)?;
    let mut data_file = DataFile::new_unstarted(filename, version);
    data_file.base_id = base_id;
    Ok((writer, data_file))
}

pub async fn open_writer(
    version: ConcreteFileVersion,
    object_store: &ObjectStore,
    schema: &Schema,
    base_dir: &Path,
    options: WriterOptions,
) -> Result<Box<dyn GenericWriter>> {
    validate_decimal32_and_decimal64(schema, version)?;
    match version {
        ConcreteFileVersion::V1 => open_v1_writer(object_store, schema, base_dir, options).await,
        ConcreteFileVersion::V2_0 | ConcreteFileVersion::V2_1 => {
            open_current_writer(
                move |object_writer, schema, filename, base_id, options| {
                    create_current_file_writer(
                        version,
                        object_writer,
                        schema,
                        filename,
                        base_id,
                        options,
                    )
                },
                object_store,
                schema,
                base_dir,
                options,
            )
            .await
        }
        ConcreteFileVersion::V2_2 | ConcreteFileVersion::V2_3 => {
            let create_file_writer = move |object_writer, schema, filename, base_id, options| {
                create_current_file_writer(
                    version,
                    object_writer,
                    schema,
                    filename,
                    base_id,
                    options,
                )
            };
            if schema.fields_pre_order().any(Field::is_blob_v2) {
                open_current_blob_v2_writer(
                    version,
                    create_file_writer,
                    object_store,
                    schema,
                    base_dir,
                    options,
                )
                .await
            } else {
                open_current_writer(create_file_writer, object_store, schema, base_dir, options)
                    .await
            }
        }
    }
}

pub async fn open_update_writer(
    version: ConcreteFileVersion,
    dataset: &Table,
    schema: &Schema,
    allow_external_blob_outside_bases: bool,
) -> Result<Box<dyn GenericWriter>> {
    open_update_writer_with_options(
        version,
        dataset,
        schema,
        allow_external_blob_outside_bases,
        FileWriterOptions::default(),
    )
    .await
}

pub async fn open_update_writer_with_options(
    version: ConcreteFileVersion,
    dataset: &Table,
    schema: &Schema,
    allow_external_blob_outside_bases: bool,
    file_writer_options: FileWriterOptions,
) -> Result<Box<dyn GenericWriter>> {
    // Blob v1 columns are stored as Blob v2 from 2.2 on, as `Dataset::write`
    // does. The promoted children take ids above the table's so the file's
    // schema reuses none of them; the commit assigns the manifest's own.
    let promoted;
    let schema = if writes_blob_v2(version)
        && schema
            .fields_pre_order()
            .any(|field| field.is_blob() && !field.is_blob_v2())
    {
        let mut schema = schema.clone();
        for field in &mut schema.fields {
            field.promote_blob_v2()?;
        }
        schema.try_set_field_id(Some(dataset.manifest.max_field_id()))?;
        promoted = schema;
        &promoted
    } else {
        schema
    };
    let external_base_resolver = match version {
        ConcreteFileVersion::V2_2 | ConcreteFileVersion::V2_3 => {
            blob_v2_external_base_resolver(Some(dataset), &WriteParams::default(), schema).await?
        }
        ConcreteFileVersion::V1 | ConcreteFileVersion::V2_0 | ConcreteFileVersion::V2_1 => None,
    };
    let options = WriterOptions::update(
        dataset.session().store_registry(),
        external_base_resolver,
        allow_external_blob_outside_bases,
        file_writer_options,
    );
    open_writer(
        version,
        &dataset.state.object_store,
        schema,
        &dataset.state.base,
        options,
    )
    .await
}

/// Whether `version` writes blob columns in the Blob v2 layout. Earlier
/// versions only have the legacy layout; 2.2 reads both but writes v2.
pub fn writes_blob_v2(version: ConcreteFileVersion) -> bool {
    match version {
        ConcreteFileVersion::V1 | ConcreteFileVersion::V2_0 | ConcreteFileVersion::V2_1 => false,
        ConcreteFileVersion::V2_2 | ConcreteFileVersion::V2_3 => true,
    }
}
