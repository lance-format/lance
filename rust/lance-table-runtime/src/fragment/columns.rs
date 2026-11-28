// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::borrow::Cow;
use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field as ArrowField, Fields as ArrowFields, Schema as ArrowSchema};
use futures::{Stream, StreamExt};
use lance_arrow::RecordBatchExt;
use lance_core::{
    Error, Result,
    datatypes::{NullabilityComparison, Schema, SchemaCompareOptions},
};
use lance_file::{version::ConcreteFileVersion, writer::FileWriterOptions};
use lance_table::transaction::DataReplacementGroup;
use object_store::path::Path;

use crate::blob::{BlobWriteSources, logical_to_prepared_blob_schema};
use crate::write::{SeedCollector, WriteSeedHook, open_update_writer_with_options, writes_blob_v2};
use crate::{FileFragment, Table};

/// The first path in `fields` that names a sibling twice. Projection picks
/// children by name, so a duplicate makes that choice arbitrary, and the
/// name-set comparison the schema check uses cannot see one at all.
#[doc(hidden)]
pub fn duplicate_field_path(fields: &ArrowFields, path: &str) -> Option<String> {
    let mut seen = HashSet::new();
    for field in fields {
        let qualified = if path.is_empty() {
            field.name().clone()
        } else {
            format!("{path}.{}", field.name())
        };
        if !seen.insert(field.name()) {
            return Some(qualified);
        }
        if let Some(nested) = duplicate_nested_path(field.data_type(), &qualified) {
            return Some(nested);
        }
    }
    None
}

fn duplicate_nested_path(data_type: &DataType, path: &str) -> Option<String> {
    match data_type {
        DataType::Struct(children) => duplicate_field_path(children, path),
        DataType::List(item)
        | DataType::LargeList(item)
        | DataType::FixedSizeList(item, _)
        | DataType::Map(item, _) => {
            duplicate_nested_path(item.data_type(), &format!("{path}.item"))
        }
        _ => None,
    }
}

/// Remove a staged file that will not be committed. Best effort: it is
/// unreachable either way, and must not mask the error that caused it.
pub async fn discard_staged_file(dataset: &Table, path: &Path) {
    // Blob v2 spills sidecars into data/<file-stem>/ beside the file, and
    // those are the large ones; leaving them is what makes a routine
    // rejection expensive.
    if let Some(stem) = path
        .filename()
        .and_then(|name| name.strip_suffix(".lance"))
        .map(|stem| dataset.data_dir().join(stem))
        && let Err(delete_error) = dataset
            .state
            .object_store
            .remove_dir_all(stem.clone())
            .await
    {
        log::warn!("failed to delete staged blob sidecars '{stem}': {delete_error}");
    }
    if let Err(delete_error) = dataset.state.object_store.delete(path).await {
        log::warn!("failed to delete staged column file '{path}': {delete_error}");
    }
}

/// `field` with nullability dropped at every level: the projector rebuilds
/// arrays against its target and panics rather than reports on a constraint,
/// so it gets a shape that cannot fail and the writer objects instead.
#[doc(hidden)]
pub fn relax_nullability(field: &ArrowField) -> ArrowField {
    let relax = |field: &Arc<ArrowField>| Arc::new(relax_nullability(field));
    let data_type = match field.data_type() {
        DataType::Struct(children) => DataType::Struct(children.iter().map(relax).collect()),
        DataType::List(item) => DataType::List(relax(item)),
        DataType::LargeList(item) => DataType::LargeList(relax(item)),
        DataType::FixedSizeList(item, width) => DataType::FixedSizeList(relax(item), *width),
        // A Map's entries struct and its key stay required -- Arrow rejects a
        // map whose entries or keys are nullable -- so only the value relaxes.
        DataType::Map(entries, sorted) => match entries.data_type() {
            DataType::Struct(kv) if kv.len() == 2 => {
                let value = Arc::new(relax_nullability(&kv[1]));
                let entries = ArrowField::new(
                    entries.name(),
                    DataType::Struct(vec![kv[0].clone(), value].into()),
                    false,
                )
                .with_metadata(entries.metadata().clone());
                DataType::Map(Arc::new(entries), *sorted)
            }
            _ => field.data_type().clone(),
        },
        other => other.clone(),
    };
    ArrowField::new(field.name(), data_type, true).with_metadata(field.metadata().clone())
}

/// Keep per-operation targets within the dataset's existing reader family.
pub fn validate_write_version(
    default_version: ConcreteFileVersion,
    target: ConcreteFileVersion,
) -> Result<()> {
    if (default_version == ConcreteFileVersion::V1) != (target == ConcreteFileVersion::V1) {
        return Err(Error::invalid_input(format!(
            "Cannot write data files in version {target} to a dataset with default version {default_version}: V1 and V2 storage versions cannot be mixed"
        )));
    }
    Ok(())
}

impl FileFragment {
    fn schema_mismatch(&self, detail: impl std::fmt::Display) -> Error {
        Error::invalid_input(format!(
            "column data for fragment {} does not match the requested schema: {detail}",
            self.id()
        ))
    }

    /// Write new data for columns of this fragment as a standalone data file,
    /// without committing it, and return the
    /// [`DataReplacementGroup`]
    /// describing it.
    ///
    /// Unlike `add_columns`, the staged file answers for a field that
    /// already exists, so this recomputes a column rather than appending one.
    ///
    /// `schema` names the fields being written. Each must be a top-level
    /// column the dataset schema already defines, matching its manifest
    /// definition; a column is staged whole, so a nested field cannot be
    /// staged on its own. To recompute a new column, declare it first with an
    /// all-null `add_columns`, then stage its data. Physical layout
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
    ) -> Result<DataReplacementGroup> {
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
    ) -> Result<DataReplacementGroup> {
        let write_version = self
            .table()
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
            None,
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
    /// # use lance_core::Result;
    /// # use lance_table_runtime::{blob::BlobWriteSources, FileFragment};
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
    ) -> Result<DataReplacementGroup> {
        let write_version = self
            .table()
            .manifest
            .data_storage_format
            .lance_file_format();
        if !writes_blob_v2(write_version) {
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
            None,
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
    /// # use lance_core::Result;
    /// # use lance_table_runtime::FileFragment;
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
    ) -> Result<DataReplacementGroup> {
        let data = data.map(|result| result.map(|batch| (batch, None)));
        self.write_column_with_version_and_options(
            data,
            schema,
            write_version,
            FileWriterOptions::default(),
            false,
            None,
        )
        .await
    }

    #[doc(hidden)]
    pub async fn write_column_with_version_and_options(
        &self,
        data: impl Stream<Item = Result<(RecordBatch, Option<BlobWriteSources>)>> + Send,
        schema: &Schema,
        write_version: ConcreteFileVersion,
        file_writer_options: FileWriterOptions,
        has_blob_sources: bool,
        seed_hook: Option<&dyn WriteSeedHook>,
    ) -> Result<DataReplacementGroup> {
        validate_write_version(
            self.table()
                .manifest
                .data_storage_format
                .lance_file_format(),
            write_version,
        )?;
        let expected_rows = self.physical_rows().await? as u64;

        let mut writer_schema = self.resolve_writer_schema(schema)?;
        if has_blob_sources {
            writer_schema.metadata.clone_from(&self.schema().metadata);
        }
        let batch_schema = ArrowSchema::from(&writer_schema);
        let (batch_schema, expected_schema) = if has_blob_sources {
            let prepared = logical_to_prepared_blob_schema(&batch_schema)?;
            let expected_schema = Cow::Owned(Schema::try_from(&prepared)?);
            (prepared, expected_schema)
        } else {
            (batch_schema, Cow::Borrowed(&writer_schema))
        };
        let projection_schema = ArrowSchema::new(
            batch_schema
                .fields()
                .iter()
                .map(|field| relax_nullability(field))
                .collect::<Vec<_>>(),
        );

        if write_version == ConcreteFileVersion::V1 {
            // The legacy reader pairs a fragment's files by batch boundary, so a
            // staged file chunked to the caller's batches leaves the fragment
            // unreadable. Rechunking is the legacy update path's job, not this
            // one's.
            return Err(Error::not_supported(format!(
                "write_columns is not supported for fragment {} in the legacy file format",
                self.id()
            )));
        }

        // The update writer, not a raw file writer: that boundary carries the
        // version's write policies (blob v2 columns arrive logical and must be
        // prepared for the encoders) and returns a populated `DataFile`.
        // Blob v2 descriptors land under the dataset root, outside any
        // registered external base, as on the other update paths.
        let has_blob_v2 = writer_schema
            .fields_pre_order()
            .any(|field| field.is_blob_v2());
        // Resolve the seed writers before the staged file exists, so a failure
        // here leaves nothing to discard.
        let mut seeds =
            SeedCollector::for_write(write_version, Some(self.table()), &writer_schema, seed_hook)
                .await?;
        let mut writer = open_update_writer_with_options(
            write_version,
            self.table(),
            &writer_schema,
            has_blob_v2 && !has_blob_sources,
            file_writer_options,
        )
        .await?;
        let staged_path = {
            let (file_name, _) = writer.data_file_path();
            self.table().data_dir().join(file_name)
        };

        // From here every failure -- a stream error, a rejected batch, a write
        // or finish error, a row-count mismatch -- owns the same staged
        // artifacts: the data file and any Blob sidecars already finalized
        // beside it. One exit cleans them all.
        let mut data = std::pin::pin!(data);
        let staged: Result<_> = async {
            while let Some(batch_result) = data.next().await {
                let (batch, sources) = batch_result?;
                // Struct encoders consume children positionally, so a batch
                // ordered differently from the manifest lands under the wrong
                // field ids. Projection fixes that by name, but it downcasts by
                // shape, so the whole tree is compared first. Nullability is the
                // writer's to enforce, against the data rather than the
                // declared schema.
                if let Some(duplicate) = duplicate_field_path(batch.schema_ref().fields(), "") {
                    return Err(self.schema_mismatch(format!("column '{duplicate}' appears twice")));
                }
                Schema::try_from(batch.schema_ref().as_ref())
                    .and_then(|staged| {
                        staged.check_compatible(
                            expected_schema.as_ref(),
                            &SchemaCompareOptions {
                                compare_nullability: NullabilityComparison::Ignore,
                                ignore_field_order: true,
                                ..Default::default()
                            },
                        )
                    })
                    .map_err(|mismatch| self.schema_mismatch(mismatch))?;
                let batch = batch
                    .project_by_schema(&projection_schema)
                    .map_err(|err| self.schema_mismatch(err))?;
                if let Some(sources) = sources.as_ref() {
                    writer
                        .write_with_blob_sources(&batch, &self.table().state.base, sources)
                        .await?;
                } else {
                    writer.write(std::slice::from_ref(&batch)).await?;
                }
                seeds.observe(&batch)?;
            }
            seeds.flush(writer.as_mut()).await?;
            let (num_rows, data_file) = writer.finish().await?;
            if num_rows as u64 != expected_rows {
                return Err(Error::invalid_input(format!(
                    "column data for fragment {} has {} rows but the fragment has {} physical rows",
                    self.id(),
                    num_rows,
                    expected_rows
                )));
            }
            Ok(data_file)
        }
        .await;

        match staged {
            Ok(data_file) => Ok(DataReplacementGroup(self.id() as u64, data_file)),
            Err(err) => {
                // The writer may still hold the file open (a buffered upload,
                // an unflushed local handle); release it before deleting.
                drop(writer);
                discard_staged_file(self.table(), &staged_path).await;
                Err(err)
            }
        }
    }

    /// Resolve the fields a staged file will carry against the manifest.
    ///
    /// Shared by [`Self::write_columns`] and [`Self::write_overlay`]: both stage
    /// a data file that the commit path publishes without publishing schema, so
    /// both need the same guarantee that every field is one the dataset already
    /// defines, exactly as it defines it. Returns the manifest's own fields, so
    /// physical layout comes from the manifest rather than from the request.
    pub(crate) fn resolve_writer_schema(&self, schema: &Schema) -> Result<Schema> {
        // Readers take everything but the field id from the manifest, so a
        // staged field reusing an id is decoded as the manifest's version rather
        // than rejected. Compare full identity, not just the storage type.
        let compare_options = SchemaCompareOptions {
            compare_field_ids: true,
            ..Default::default()
        };
        // Top-level requests match top-level manifest fields only: resolving an
        // id from anywhere lets a caller reuse a field at a path the dataset
        // never gave it, staging a file covering the borrowed field. Layout then
        // comes from the manifest, since the metadata the identity check ignores
        // -- packed structs, blob encoding -- decides physical field coverage.
        let dataset_schema = self.schema();
        let mut writer_fields = Vec::with_capacity(schema.fields.len());
        let mut requested = HashSet::with_capacity(schema.fields.len());
        for field in &schema.fields {
            // The per-field identity check cannot see the request naming an
            // id twice, and the set-based batch comparison downstream would
            // match one batch column against both copies.
            if !requested.insert(field.id) {
                return Err(Error::invalid_input(format!(
                    "column data for fragment {} names field id {} ('{}') more than once",
                    self.id(),
                    field.id,
                    field.name
                )));
            }
            if lance_core::is_system_column(&field.name) {
                return Err(Error::invalid_input(format!(
                    "column data for fragment {} names reserved column '{}'",
                    self.id(),
                    field.name
                )));
            }
            let Some(existing) = dataset_schema
                .fields
                .iter()
                .find(|existing| existing.id == field.id)
            else {
                // The commit path publishes data files, never schema, so a
                // field the manifest does not define would commit as a file no
                // live field answers for -- and a concurrent schema change
                // could never be checked against it.
                return Err(Error::invalid_input(format!(
                    "column data for fragment {} names field id {} ('{}') that the dataset schema \
                     does not define; declare the column with add_columns before staging its data",
                    self.id(),
                    field.id,
                    field.name
                )));
            };
            // `explain_difference` recurses, covering the whole subtree.
            if let Some(difference) = field.explain_difference(existing, &compare_options) {
                return Err(Error::invalid_input(format!(
                    "column data for fragment {} does not match dataset field id {}: {}",
                    self.id(),
                    field.id,
                    difference
                )));
            }
            writer_fields.push(existing.clone());
        }
        Ok(Schema {
            fields: writer_fields,
            metadata: schema.metadata.clone(),
        })
    }
}
