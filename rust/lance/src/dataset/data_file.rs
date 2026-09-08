// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stateless writing and concatenation of complete encoded data-file parts.

use std::{collections::HashSet, num::NonZeroU64, ops::Range, sync::Arc};

use arrow_array::RecordBatch;
use futures::{Stream, StreamExt};
use lance_core::{Error, Result, datatypes::Schema};
use lance_file::{
    concat::{
        BlobTargetId, EncodedFileInput, FileConcatOptions, FileConcatReason, FileConcatResult,
        FileConcatTarget, concat_data_file_parts as concat_parts,
    },
    version::ConcreteFileVersion,
    versions as file_versions,
    writer::{FileWriteSummary, FileWriterOptions},
};
use lance_io::traits::Writer;
use lance_table::format::DataFile;
use object_store::path::Path;

pub use lance_file::concat::DataFilePart;

use super::{
    Dataset,
    fragment::{FileFragment, write::generate_random_filename},
    transaction::DataReplacementGroup,
};
use crate::{
    blob::prepared_to_logical_blob_schema,
    dataset::{
        blob::BlobPreprocessor,
        write::{
            ExternalBlobMode, WriteParams, blob_v2_external_base_resolver,
            validate_blob_v2_write_schema,
        },
    },
};

/// Runtime identity and logical schema of a final concatenated data file.
///
/// Reuse the same live value for every part write and final concatenation. Lance
/// defines no serialization or recovery contract for this type. The caller must
/// keep every use associated with the same dataset and resolved base; Lance does
/// not validate that association across [`Dataset`] instances.
#[derive(Debug, Clone)]
pub struct DataFileTarget {
    file_name: String,
    base_id: Option<u32>,
    schema: Arc<Schema>,
    version: ConcreteFileVersion,
    blob_target_id: Option<BlobTargetId>,
}

impl DataFileTarget {
    /// Create a final data-file target with Lance's ordinary random file naming.
    ///
    /// This only creates a runtime identity; it does not create, reserve, or
    /// register an object. The caller owns the target lifetime, part storage,
    /// cleanup, and commit state. Prepared Blob v2 schemas are normalized to
    /// their caller-visible logical form; the persisted descriptor schema remains
    /// an internal writer detail.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use lance::dataset::DataFileTarget;
    /// use lance_core::datatypes::Schema;
    /// use lance_file::version::ConcreteFileVersion;
    ///
    /// # fn target(schema: Arc<Schema>) -> lance_core::Result<DataFileTarget> {
    /// DataFileTarget::new(
    ///     None,
    ///     schema,
    ///     ConcreteFileVersion::V2_2,
    /// )
    /// # }
    /// ```
    pub fn new(
        base_id: Option<u32>,
        schema: Arc<Schema>,
        version: ConcreteFileVersion,
    ) -> Result<Self> {
        if version == ConcreteFileVersion::V1 {
            return Err(Error::not_supported(
                "data-file part concatenation does not support Lance v1".to_string(),
            ));
        }
        if base_id == Some(0) {
            return Err(Error::invalid_input(
                "DataFileTarget.base_id must not use reserved ID 0",
            ));
        }
        if schema.fields.is_empty() {
            return Err(Error::invalid_input(
                "DataFileTarget.schema must contain at least one top-level field",
            ));
        }
        let mut field_ids = HashSet::with_capacity(schema.fields.len());
        for field in &schema.fields {
            if !field_ids.insert(field.id) {
                return Err(Error::invalid_input(format!(
                    "DataFileTarget.schema contains duplicate top-level field ID {}",
                    field.id
                )));
            }
        }
        let schema = Arc::new(prepared_to_logical_blob_schema(schema.as_ref())?);
        let has_blob_v2 = schema.fields_pre_order().any(|field| field.is_blob_v2());
        if schema
            .fields_pre_order()
            .any(|field| field.is_blob() && !field.is_blob_v2())
        {
            return Err(Error::not_supported(
                "DataFileTarget does not support legacy Blob v1 fields",
            ));
        }
        let file_name = format!("{}.lance", generate_random_filename());
        let blob_target_id = has_blob_v2.then(|| {
            let base = base_id
                .map(|id| format!("base:{id}"))
                .unwrap_or_else(|| "primary".to_string());
            BlobTargetId::new(format!("{base}/{file_name}"))
        });
        Ok(Self {
            file_name,
            base_id,
            schema,
            version,
            blob_target_id,
        })
    }

    /// Relative path of the final data file within its selected base.
    pub fn file_name(&self) -> &str {
        &self.file_name
    }

    /// Optional registered dataset base that owns the final data file.
    pub fn base_id(&self) -> Option<u32> {
        self.base_id
    }

    /// Caller-visible logical schema encoded by every part.
    pub fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    /// Exact Lance file grammar used by parts and final output.
    pub fn version(&self) -> ConcreteFileVersion {
        self.version
    }

    /// Open one caller-provided part and associate its managed Blob descriptors
    /// with this runtime target.
    ///
    /// The caller must ensure that Blob payloads were written through this target
    /// using the same dataset and resolved base that will assemble the part.
    pub async fn open_part(
        &self,
        input: EncodedFileInput,
        blob_ids: Option<Range<u32>>,
    ) -> Result<DataFilePart> {
        DataFilePart::open(input, blob_ids, self.blob_target_id.clone()).await
    }

    fn object_path(&self, data_dir: &Path) -> Path {
        data_dir.clone().join(self.file_name.as_str())
    }
}

impl Dataset {
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
        self.data_file_dir_for_base(target.base_id)?;

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

    /// Encode one independently persisted part for a future data file.
    ///
    /// The caller owns `output` and its storage path. Managed Blob payloads are
    /// written directly beneath the sidecar directory selected by the final
    /// target using IDs from `blob_ids`; every non-empty logical Inline value is
    /// spilled to Packed or Dedicated storage so final concatenation never copies
    /// Blob payload bytes.
    /// Every use of `target` must refer to the same dataset and resolved base;
    /// associating a runtime target with that storage context is the caller's
    /// responsibility.
    ///
    /// # Example
    ///
    /// ```
    /// use arrow_array::RecordBatch;
    /// use futures::stream;
    /// use lance::{Dataset, dataset::DataFileTarget};
    /// use lance_io::traits::Writer;
    ///
    /// # async fn write_part(
    /// #     dataset: &Dataset,
    /// #     target: &DataFileTarget,
    /// #     output: Box<dyn Writer>,
    /// #     batch: RecordBatch,
    /// # ) -> lance_core::Result<()> {
    /// dataset
    ///     .write_data_file_part(target, output, None, stream::iter([Ok(batch)]))
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn write_data_file_part(
        &self,
        target: &DataFileTarget,
        output: Box<dyn Writer>,
        blob_ids: Option<Range<u32>>,
        data: impl Stream<Item = Result<RecordBatch>> + Send,
    ) -> Result<FileWriteSummary> {
        self.validate_data_file_target(target)?;
        validate_blob_v2_write_schema(target.schema.as_ref())?;
        let has_blob = target
            .schema
            .fields_pre_order()
            .any(|field| field.is_blob_v2());
        if has_blob && blob_ids.is_none() {
            return Err(Error::invalid_input(
                "write_data_file_part requires a non-empty Blob ID range for a schema containing Blob v2 fields",
            ));
        }

        let mut preprocessor = if let Some(blob_ids) = blob_ids {
            let data_dir = self.data_file_dir_for_base(target.base_id)?;
            let data_file_key = target.file_name.strip_suffix(".lance").ok_or_else(|| {
                Error::invalid_input("DataFileTarget.file_name must end in '.lance'")
            })?;
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
                    data_file_key.to_string(),
                    target.schema.as_ref(),
                    external_base_resolver,
                    false,
                    ExternalBlobMode::Reference,
                    self.session().store_registry(),
                    self.store_params().cloned().unwrap_or_default(),
                    None,
                )?
                .with_part_blob_ids(blob_ids)?,
            )
        } else {
            None
        };

        let mut writer = file_versions::create_writer(
            target.version,
            output,
            target.schema.as_ref().clone(),
            FileWriterOptions::default(),
        )?;
        let mut data = Box::pin(data);
        let write_result = async {
            while let Some(batch) = data.next().await {
                let batch = batch?;
                if let Some(preprocessor) = preprocessor.as_mut() {
                    let batch = preprocessor.preprocess_batch(&batch).await?;
                    writer.write_batch(&batch).await?;
                } else {
                    writer.write_batch(&batch).await?;
                }
            }
            if let Some(preprocessor) = preprocessor.as_mut() {
                preprocessor.finish().await?;
            }
            writer.finish().await
        }
        .await;

        match write_result {
            Ok(summary) => Ok(summary),
            Err(error) => {
                writer.abort().await;
                if let Some(preprocessor) = preprocessor.as_mut() {
                    preprocessor.abort();
                }
                Err(error)
            }
        }
    }

    /// Concatenate validated parts into the Lance-generated final data file.
    ///
    /// Part order is the final physical row order. The operation copies
    /// encoded page buffers and regenerates metadata and the footer; incompatible
    /// inputs fail without a decode/re-encode fallback or dataset commit. The
    /// caller owns cleanup of all durable part, Blob, and final-file objects.
    /// The caller must also assemble the target through the same dataset and
    /// resolved base used to write managed Blob payloads.
    ///
    /// # Example
    ///
    /// ```
    /// use lance::{Dataset, dataset::{DataFilePart, DataFileTarget}};
    ///
    /// # async fn concat(
    /// #     dataset: &Dataset,
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
        self.validate_data_file_target(target)?;
        if ordered_parts.is_empty() {
            return Err(Error::invalid_input(
                "concat_data_file_parts requires at least one part",
            ));
        }
        let data_dir = self.data_file_dir_for_base(target.base_id)?;
        let output_path = target.object_path(&data_dir);
        let object_store = self.object_store(target.base_id).await?;
        let mut concat_target = FileConcatTarget::new(target.version, target.schema.clone());
        if let Some(blob_target_id) = target.blob_target_id.clone() {
            concat_target = concat_target.with_blob_target_id(blob_target_id);
        }
        let result = concat_parts(
            &concat_target,
            ordered_parts,
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

impl FileFragment {
    /// Write parts as a complete replacement for existing top-level columns.
    ///
    /// The target schema must name current top-level fields, and the sum of
    /// part footer row counts must equal this fragment's physical row count.
    /// The returned group is uncommitted; the caller retains snapshot fencing.
    ///
    /// # Example
    ///
    /// ```
    /// use lance::dataset::{DataFilePart, DataFileTarget};
    /// use lance::dataset::fragment::FileFragment;
    ///
    /// # async fn replace(
    /// #     fragment: &FileFragment,
    /// #     target: &DataFileTarget,
    /// #     ordered_parts: &[DataFilePart],
    /// # ) -> lance_core::Result<()> {
    /// let replacement = fragment.write_columns_from_parts(target, ordered_parts).await?;
    /// // The caller includes `replacement` in its fenced transaction.
    /// # let _ = replacement;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn write_columns_from_parts(
        &self,
        target: &DataFileTarget,
        ordered_parts: &[DataFilePart],
    ) -> Result<DataReplacementGroup> {
        let expected_rows = self.physical_rows().await? as u64;
        let actual_rows = ordered_parts.iter().try_fold(0u64, |total, part| {
            total
                .checked_add(part.num_rows())
                .ok_or_else(|| Error::invalid_input("part physical row count overflows u64"))
        })?;
        if actual_rows != expected_rows {
            return Err(Error::invalid_input(format!(
                "parts contain {actual_rows} physical rows, but fragment {} contains {expected_rows}",
                self.id()
            )));
        }
        let data_file = self
            .dataset()
            .concat_data_file_parts(target, ordered_parts)
            .await?;
        Ok(DataReplacementGroup(self.id() as u64, data_file))
    }
}
