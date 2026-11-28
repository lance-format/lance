// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

mod metadata;
mod target;

pub use metadata::DataFilePart;
pub use target::DataFileTarget;

use std::num::NonZeroU64;
use std::ops::Range;

use arrow_array::RecordBatch;
use futures::{Stream, StreamExt};
use lance_core::{Error, Result};
use lance_file::concat::{
    FileConcatOptions, FileConcatReason, FileConcatResult, FileConcatTarget,
    concat_data_file_parts as concat_parts,
};
use lance_file::{versions as file_versions, writer::FileWriterOptions};
use lance_table::{feature_flags::managed_blobs_enabled, format::DataFile};

use crate::Table;
use crate::blob::{BlobPreprocessor, BlobWriteSources};
use crate::fragment::write::generate_random_filename;

use super::file::V2WriterAdapter;
use super::schema::validate_blob_v2_write_schema;
use super::{ExternalBlobMode, WriteParams, blob_v2_external_base_resolver, writes_blob_v2};

impl Table {
    fn validate_data_file_target(&self, target: &DataFileTarget) -> Result<()> {
        let dataset_version = self.manifest.data_storage_format.lance_file_format();
        if target.version != dataset_version {
            return Err(Error::invalid_input(format!(
                "DataFileTarget.version is {}, but dataset version {} uses {}",
                target.version,
                self.version_id(),
                dataset_version
            )));
        }
        self.view().data_file_dir_for_base(target.base_id)?;

        if target.schema.metadata != self.schema().metadata {
            return Err(Error::invalid_input(
                "DataFileTarget.schema metadata differs from the dataset schema metadata",
            ));
        }
        for target_field in &target.schema.fields {
            let Some(dataset_field) = self
                .schema()
                .fields
                .iter()
                .find(|field| field.id == target_field.id)
            else {
                return Err(Error::invalid_input(format!(
                    "DataFileTarget.schema field ID {} is not a top-level dataset field",
                    target_field.id
                )));
            };
            if dataset_field != target_field {
                return Err(Error::invalid_input(format!(
                    "DataFileTarget.schema field ID {} differs from the current dataset field",
                    target_field.id
                )));
            }
        }

        Ok(())
    }

    /// Encode one managed part and return its serializable description.
    ///
    /// Lance generates a unique staging name in the target's base. With schema
    /// metadata `lance.blob.managed=true`, Managed Blob payloads use independent
    /// `_blobs/<uuid>.blob` objects in that base. Otherwise payloads use sidecars
    /// addressed relative to the final data file. Every
    /// non-empty logical Inline value is spilled so final concatenation never
    /// copies Blob payload bytes. Managed parts accept `blob_ids=None`; only
    /// file-relative sidecars require disjoint reservations. Supplied ranges
    /// are ignored for Managed parts.
    /// Every use of `target` must refer to the same dataset and resolved base;
    /// associating a target with that storage context is the caller's
    /// responsibility.
    /// Persist the target before writing. A failed write may leave files; after
    /// stopping all users of the target, [`DataFileTarget::cleanup`] removes
    /// staging files and file-relative sidecars without a completed part description.
    /// File-relative retries must use fresh, disjoint Blob ID ranges, including
    /// ranges from failed writes. Independent Managed objects follow ordinary
    /// dataset GC rules, which must be coordinated with uncommitted writes
    /// and checkpoints.
    ///
    /// # Example
    ///
    /// ```
    /// use arrow_array::RecordBatch;
    /// use futures::stream;
    /// use lance_table_runtime::{Table, write::DataFileTarget};
    ///
    /// # async fn write_part(
    /// #     dataset: &Table,
    /// #     target: &DataFileTarget,
    /// #     batch: RecordBatch,
    /// # ) -> lance_core::Result<()> {
    /// let part = dataset
    ///     .write_data_file_part(target, None, stream::iter([Ok(batch)]))
    ///     .await?;
    /// // Serialize part into the caller's checkpoint.
    /// # let _ = part;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn write_data_file_part(
        &self,
        target: &DataFileTarget,
        blob_ids: Option<Range<u32>>,
        data: impl Stream<Item = Result<RecordBatch>> + Send,
    ) -> Result<DataFilePart> {
        self.write_data_file_part_with_file_writer_options(
            target,
            blob_ids,
            data,
            FileWriterOptions::default(),
        )
        .await
    }

    /// Encode one managed part with runtime file-writer options.
    ///
    /// This has the same part description, Blob handling and concat semantics
    /// as [`Self::write_data_file_part`]. The options control buffering and
    /// execution only: the target still selects the file grammar and physical
    /// layout, so an assembled target is indistinguishable from one whose parts
    /// were written without them.
    ///
    /// Use this when the caller owns a memory budget for the writer. A part
    /// retains its encoded pages until [`Self::concat_data_file_parts`], so
    /// without an explicit budget each cache-owning field encoder buffers its
    /// own default allowance regardless of how many fields the schema has.
    pub async fn write_data_file_part_with_file_writer_options(
        &self,
        target: &DataFileTarget,
        blob_ids: Option<Range<u32>>,
        data: impl Stream<Item = Result<RecordBatch>> + Send,
        file_writer_options: FileWriterOptions,
    ) -> Result<DataFilePart> {
        let data = data.map(|result| result.map(|batch| (batch, None)));
        self.write_data_file_part_impl(target, blob_ids, data, file_writer_options)
            .await
    }

    /// Encode one part with trusted payload sources accompanying its batches.
    ///
    /// Each source map belongs to one batch. Use prepared Blob v2 leaves and
    /// insert the descriptors returned by [`BlobWriteSources`] at the relevant
    /// slots. Uploaded keys are relative to the base of `target`. On an ordinary
    /// table every non-empty payload is spilled to sidecars using the target's
    /// allocated Blob IDs. A Managed-enabled table can reference the completed
    /// upload directly. The part and its sidecars use the same lifetime and
    /// concatenation rules as [`Self::write_data_file_part`].
    ///
    /// ```
    /// # use std::ops::Range;
    /// # use arrow_array::RecordBatch;
    /// # use lance_core::Result;
    /// # use lance_table_runtime::{Table, blob::BlobWriteSources, write::DataFileTarget};
    /// # use lance_file::writer::FileWriterOptions;
    /// # async fn example(dataset: &Table, target: &DataFileTarget, ids: Option<Range<u32>>,
    /// #   batch: RecordBatch, sources: BlobWriteSources) -> Result<()> {
    /// let part = dataset.write_data_file_part_with_blob_sources(
    ///     target, ids, futures::stream::iter([Ok((batch, sources))]),
    ///     FileWriterOptions::default(),
    /// ).await?;
    /// # let _ = part;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn write_data_file_part_with_blob_sources(
        &self,
        target: &DataFileTarget,
        blob_ids: Option<Range<u32>>,
        data: impl Stream<Item = Result<(RecordBatch, BlobWriteSources)>> + Send,
        file_writer_options: FileWriterOptions,
    ) -> Result<DataFilePart> {
        if !writes_blob_v2(target.version) {
            return Err(Error::not_supported(
                "Part writes with Blob sources require Lance file version 2.2 or later",
            ));
        }
        let data = data.map(|result| result.map(|(batch, sources)| (batch, Some(sources))));
        self.write_data_file_part_impl(target, blob_ids, data, file_writer_options)
            .await
    }

    async fn write_data_file_part_impl(
        &self,
        target: &DataFileTarget,
        blob_ids: Option<Range<u32>>,
        data: impl Stream<Item = Result<(RecordBatch, Option<BlobWriteSources>)>> + Send,
        file_writer_options: FileWriterOptions,
    ) -> Result<DataFilePart> {
        self.validate_data_file_target(target)?;
        validate_blob_v2_write_schema(target.schema.as_ref())?;
        let managed = managed_blobs_enabled(target.schema.as_ref())?;
        let part_blob_ids = if managed { None } else { blob_ids.clone() };
        let has_blob = target
            .schema
            .fields_pre_order()
            .any(|field| field.is_blob_v2());
        if has_blob && !managed && blob_ids.is_none() {
            return Err(Error::invalid_input(
                "write_data_file_part requires a non-empty Blob ID range for a schema containing Blob v2 fields",
            ));
        }

        let mut preprocessor = if has_blob || blob_ids.is_some() {
            let data_dir = self.view().data_file_dir_for_base(target.base_id)?;
            let object_store = self.object_store(target.base_id).await?;
            let external_base_resolver = blob_v2_external_base_resolver(
                Some(self),
                &WriteParams::default(),
                target.schema.as_ref(),
            )
            .await?;
            Some(
                BlobPreprocessor::new(
                    object_store.as_ref().clone(),
                    data_dir,
                    target.data_file_key().to_string(),
                    target.schema.as_ref(),
                    external_base_resolver,
                    false,
                    ExternalBlobMode::Reference,
                    self.session().store_registry(),
                    self.store_params().cloned().unwrap_or_default(),
                    None,
                )?
                .with_part_blob_ids(if managed { None } else { blob_ids })?,
            )
        } else {
            None
        };
        let root = self.view().blob_base_path(target.base_id)?;
        if managed && let Some(writer) = preprocessor.take() {
            preprocessor = Some(writer.with_managed_base(target.base_id, root.clone()));
        }

        let file_name = format!("{}.part", generate_random_filename());
        let path = target
            .parts_dir(&self.view().data_file_dir_for_base(target.base_id)?)
            .join(file_name.as_str());
        let store = self.object_store(target.base_id).await?;
        let mut writer = V2WriterAdapter::new(
            file_versions::create_writer(
                target.version,
                store.create(&path).await?,
                target.schema.as_ref().clone(),
                file_writer_options,
            )?,
            None,
            preprocessor,
            None,
        );
        let mut data = Box::pin(data);
        let write_result = async {
            while let Some(batch) = data.next().await {
                let (batch, sources) = batch?;
                let sources = sources.as_ref().map(|sources| (&root, sources));
                writer.write_batch_with_sources(&batch, sources).await?;
            }
            writer.finish_file().await
        }
        .await;

        match write_result {
            Ok(summary) => Ok(DataFilePart {
                target_file_name: target.file_name.clone(),
                base_id: target.base_id,
                file_name,
                blob_ids: part_blob_ids,
                num_rows: summary.num_rows,
                size_bytes: NonZeroU64::new(summary.size_bytes)
                    .ok_or_else(|| Error::internal("completed part has zero file size"))?,
            }),
            Err(error) => {
                writer.abort().await;
                Err(error)
            }
        }
    }

    /// Reopen, validate, and concatenate parts into the final data file.
    ///
    /// Part order is the final physical row order. The operation copies
    /// encoded page buffers and regenerates metadata and the footer; incompatible
    /// inputs fail without a decode/re-encode fallback or dataset commit. The
    /// caller owns cleanup of all durable part, Blob, and final-file objects.
    /// The caller must also assemble the target through the same dataset and
    /// resolved base used to write managed Blob payloads.
    /// The target must still be uncommitted and have no concurrent assembler.
    /// Assembly can overwrite a previous uncommitted output for the same target;
    /// it does not check current or historical manifests for references.
    ///
    /// To replace a fragment's columns, wrap the returned file in a
    /// [`DataReplacementGroup`](lance_table::transaction::DataReplacementGroup).
    /// The caller must check that the ordered parts cover the fragment's physical
    /// rows exactly, including deleted rows; this operation does not check
    /// fragment coverage or commit the replacement.
    ///
    /// # Example
    ///
    /// ```
    /// use lance_table_runtime::{Table, write::{DataFilePart, DataFileTarget}};
    ///
    /// # async fn concat(
    /// #     dataset: &Table,
    /// #     target: &DataFileTarget,
    /// #     ordered_parts: &[DataFilePart],
    /// # ) -> lance_core::Result<()> {
    /// let data_file = dataset.concat_data_file_parts(target, ordered_parts).await?;
    /// // The caller decides when and how to commit `data_file`.
    /// # let _ = data_file;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn concat_data_file_parts(
        &self,
        target: &DataFileTarget,
        ordered_parts: &[DataFilePart],
    ) -> Result<DataFile> {
        let opened = DataFilePart::open_all(self, target, ordered_parts).await?;
        let data_dir = self.view().data_file_dir_for_base(target.base_id)?;
        let output_path = target.object_path(&data_dir);
        let object_store = self.object_store(target.base_id).await?;
        let mut concat_target = FileConcatTarget::new(target.version, target.schema.clone());
        if let Some(blob_target_id) = target.blob_target_id() {
            concat_target = concat_target.with_blob_target_id(blob_target_id);
        }
        let result = concat_parts(
            &concat_target,
            &opened,
            {
                let object_store = object_store.clone();
                let output_path = output_path.clone();
                move || async move { object_store.create(&output_path).await }
            },
            FileConcatOptions::default(),
        )
        .await;

        let output = match result {
            Ok(FileConcatResult::Written(output)) => output,
            Ok(FileConcatResult::Reused(_, _)) => {
                return Err(Error::internal(
                    "data-file part concatenation unexpectedly reused an input".to_string(),
                ));
            }
            Ok(FileConcatResult::Unsupported(reason)) => {
                let message = format!(
                    "parts cannot be concatenated into target {:?}: {reason}",
                    target.file_name
                );
                return Err(match reason {
                    FileConcatReason::VersionMismatch { actual, .. } => {
                        let (major, minor) = actual.to_standard_footer_numbers();
                        Error::version_conflict(message, major, minor)
                    }
                    FileConcatReason::SchemaMismatch { .. } => Error::schema_mismatch(message),
                    FileConcatReason::LegacyVersion
                    | FileConcatReason::ColumnLayoutMismatch { .. }
                    | FileConcatReason::ColumnEncodingMismatch { .. }
                    | FileConcatReason::ColumnBuffers { .. }
                    | FileConcatReason::ExtraGlobalBuffers { .. }
                    | FileConcatReason::BlobColumns => Error::not_supported(message),
                });
            }
            Err(error) => return Err(error),
        };
        let (fields, column_indices) =
            file_versions::data_file_columns(target.version, target.schema.as_ref());
        Ok(DataFile::new(
            target.file_name.clone(),
            fields,
            column_indices,
            target.version,
            NonZeroU64::new(output.size_bytes),
            target.base_id,
        ))
    }
}

#[cfg(test)]
mod tests;
