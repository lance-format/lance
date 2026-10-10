// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Overwriting existing columns from a stream aligned with a fragment's live
//! rows. See [`FileFragment::update_columns_from_stream`].

use std::collections::HashSet;
use std::sync::Arc;

use arrow::compute::concat_batches;
use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field as ArrowField, Fields, Schema as ArrowSchema, SchemaRef};
use datafusion::execution::SendableRecordBatchStream;
use futures::StreamExt;
use lance_arrow::{FieldExt, RecordBatchExt};
use lance_core::datatypes::{
    BLOB_V2_LOGICAL_FIELDS, BLOB_V2_LOGICAL_MINIMAL_FIELDS, OnMissing, OnTypeMismatch, Schema,
};
use lance_core::utils::address::RowAddress;
use lance_core::{Error, ROW_ADDR, Result, is_system_column};
use lance_datafusion::utils::StreamingWriteSource;
use lance_file::version::ConcreteFileVersion;
use lance_table::format::overlay::TOMBSTONE_FIELD_ID;
use roaring::RoaringBitmap;

use super::{FileFragment, FragmentUpdateColumnsResult, duplicate_field_path, relax_nullability};
use crate::dataset::utils::SchemaAdapter;

/// How the source's columns map onto the fragment, worked out before any
/// batch is pulled.
struct AlignedColumns {
    /// The source schema every batch must match.
    source_schema: SchemaRef,
    row_addr_index: usize,
    /// Positions of the value columns in the source, in source order.
    value_indices: Vec<usize>,
    /// The fields written, resolved against the dataset schema.
    write_schema: Schema,
    /// `write_schema` as Arrow, with nullability relaxed.
    write_arrow_schema: SchemaRef,
    /// Converts the value columns to the form the file stores them in.
    adapter: SchemaAdapter,
    /// Fed back to the updater for a read batch with no live rows.
    empty_values: RecordBatch,
}

impl FileFragment {
    /// Overwrite existing columns of this fragment from a stream aligned with
    /// its live rows.
    ///
    /// `source` carries `_rowaddr` plus the columns to overwrite, and yields
    /// every live row of the fragment exactly once, in ascending `_rowaddr`
    /// order (what an in-order scan of the fragment, the default, returns).
    /// Each row's `_rowaddr` is checked against the row it lands on, so a
    /// stream that skips, reorders, or adds a row is rejected instead of
    /// writing values onto other rows. Columns are replaced whole: a struct
    /// must be supplied with all of its children. Blob v2 columns take the
    /// logical layout (`data`, `uri`, and optionally `position` and `size`).
    ///
    /// Unlike [`Self::update_columns`], there is no join and the old values
    /// are never read, so memory is bounded by the stream's batch size and
    /// `batch_size` rather than by the fragment. Unlike
    /// [`Self::write_columns`], deleted rows are not supplied. The legacy file
    /// format is not supported.
    ///
    /// `batch_size` sets how many rows are processed at a time.
    ///
    /// Commit the result as [`Self::update_columns_with_offsets`] does: an
    /// `Operation::Update` with `UpdateMode::RewriteColumns`, passing
    /// `matched_offsets` as `updated_fragment_offsets`, against the version the
    /// fragment was read from. Without `RewriteColumns`, a data overlay on a
    /// rewritten field keeps overriding the new values; without it or the
    /// offsets, a dataset with stable row ids does not advance
    /// `_row_last_updated_at_version`.
    ///
    /// ```
    /// # use std::collections::HashMap;
    /// # use std::sync::Arc;
    /// # use arrow_array::RecordBatchReader;
    /// # use lance::Result;
    /// # use lance::dataset::fragment::FileFragment;
    /// # use lance::dataset::transaction::{Operation, UpdateMode, UpdatedFragmentOffsets};
    /// # use lance::dataset::{Dataset, WriteDestination};
    /// # async fn rewrite(
    /// #     dataset: Arc<Dataset>,
    /// #     values: Box<dyn RecordBatchReader + Send>,
    /// # ) -> Result<Dataset> {
    /// let fragment = dataset.get_fragment(0).unwrap();
    /// let update = fragment.update_columns_from_stream(values, None).await?;
    /// let offsets = HashMap::from([(update.fragment.id, update.matched_offsets)]);
    /// let operation = Operation::Update {
    ///     removed_fragment_ids: vec![],
    ///     updated_fragments: vec![update.fragment],
    ///     new_fragments: vec![],
    ///     fields_modified: update.fields_modified,
    ///     compacted_sstables: vec![],
    ///     fields_for_preserving_frag_bitmap: vec![],
    ///     update_mode: Some(UpdateMode::RewriteColumns),
    ///     inserted_rows_filter: None,
    ///     updated_fragment_offsets: Some(UpdatedFragmentOffsets(offsets)),
    /// };
    /// let read_version = dataset.version().version;
    /// Dataset::commit(
    ///     WriteDestination::Dataset(dataset),
    ///     operation,
    ///     Some(read_version),
    ///     None,
    ///     None,
    ///     Default::default(),
    ///     false,
    /// )
    /// .await
    /// # }
    /// ```
    pub async fn update_columns_from_stream(
        &self,
        source: impl StreamingWriteSource,
        batch_size: Option<u32>,
    ) -> Result<FragmentUpdateColumnsResult> {
        let columns = self.aligned_columns(source.arrow_schema())?;
        let mut stream = source.into_stream();
        let mut updater = self
            .updater::<String>(
                Some(&[]),
                Some((columns.write_schema.clone(), self.schema().clone())),
                batch_size,
                None,
            )
            .await?;

        let mut matched_offsets = RoaringBitmap::new();
        let written = async {
            let mut leftover = None;
            let mut rows_consumed = 0;
            // An empty projection reads only the live rows' `_rowaddr`.
            while let Some(batch) = updater.next().await? {
                let expected = batch[ROW_ADDR].as_primitive::<UInt64Type>().clone();
                let values = if expected.is_empty() {
                    columns.empty_values.clone()
                } else {
                    let supplied = self
                        .next_rows(&mut stream, &mut leftover, expected.len(), &columns)
                        .await?;
                    self.check_alignment(&expected, &supplied, &columns, rows_consumed)?;
                    rows_consumed += supplied.num_rows();
                    self.value_columns(&supplied, &columns)?
                };
                matched_offsets.extend(
                    expected
                        .values()
                        .iter()
                        .map(|addr| RowAddress::from(*addr).row_offset()),
                );
                updater.update(values).await?;
            }
            self.check_exhausted(&mut stream, leftover).await?;
            updater.finish().await
        }
        .await;
        let mut fragment = match written {
            Ok(fragment) => fragment,
            Err(err) => {
                updater.cleanup_unfinished_writer().await;
                return Err(err);
            }
        };

        // The updater writes one file holding every replaced field, appended
        // after the fragment's existing files.
        if fragment.files.len() != self.metadata.files.len() + 1 {
            return Err(Error::internal(format!(
                "update_columns_from_stream wrote no data file for fragment {}",
                self.id()
            )));
        }
        let (written_file, existing_files) = fragment.files.split_last_mut().unwrap();
        let written_fields = written_file.fields.clone();
        for file in existing_files {
            if file
                .fields
                .iter()
                .any(|field| written_fields.contains(field))
            {
                file.fields = file
                    .fields
                    .iter()
                    .map(|field| {
                        if written_fields.contains(field) {
                            TOMBSTONE_FIELD_ID
                        } else {
                            *field
                        }
                    })
                    .collect::<Vec<_>>()
                    .into();
            }
        }
        fragment
            .files
            .retain(|file| file.fields.iter().any(|&field| field != TOMBSTONE_FIELD_ID));

        Ok(FragmentUpdateColumnsResult {
            fragment,
            fields_modified: written_fields
                .iter()
                .filter_map(|&field| u32::try_from(field).ok())
                .collect(),
            matched_offsets,
        })
    }

    fn aligned_columns(&self, source_schema: SchemaRef) -> Result<AlignedColumns> {
        let id = self.id();
        // Legacy files keep no validity for lists, fixed-size lists, fixed-size
        // binary or strings, and read an empty string back as null, so values
        // written there do not all read back as supplied.
        let write_version = self
            .dataset
            .manifest
            .data_storage_format
            .lance_file_format();
        if write_version == ConcreteFileVersion::V1 {
            return Err(Error::not_supported(format!(
                "update_columns_from_stream is not supported for fragment {id} in the legacy \
                 file format"
            )));
        }
        if let Some(duplicate) = duplicate_field_path(source_schema.fields(), "") {
            return Err(Error::invalid_input(format!(
                "column '{duplicate}' appears twice in the stream for fragment {id}"
            )));
        }
        let row_addr_index = source_schema.index_of(ROW_ADDR).map_err(|_| {
            Error::invalid_input(format!(
                "the stream for fragment {id} has no '{ROW_ADDR}' column naming the row each \
                 value belongs to"
            ))
        })?;
        let row_addr_type = source_schema.field(row_addr_index).data_type();
        if row_addr_type != &DataType::UInt64 {
            return Err(Error::invalid_input(format!(
                "'{ROW_ADDR}' in the stream for fragment {id} must be UInt64, got {row_addr_type}"
            )));
        }

        let value_indices = (0..source_schema.fields().len())
            .filter(|&index| index != row_addr_index)
            .collect::<Vec<_>>();
        if value_indices.is_empty() {
            return Err(Error::invalid_input(format!(
                "the stream for fragment {id} has no columns to update besides '{ROW_ADDR}'"
            )));
        }
        for &index in &value_indices {
            let name = source_schema.field(index).name();
            if is_system_column(name) {
                return Err(Error::invalid_input(format!(
                    "column '{name}' is a reserved metadata column and cannot be updated"
                )));
            }
            if !self.schema().fields.iter().any(|field| &field.name == name) {
                return Err(Error::invalid_input(format!(
                    "column '{name}' does not exist in fragment {id}"
                )));
            }
        }

        let value_schema = Arc::new(source_schema.project(&value_indices)?);
        let supplied = self.schema().project_by_schema(
            value_schema.as_ref(),
            OnMissing::Error,
            OnTypeMismatch::Error,
        )?;
        // Supplying part of a struct would leave the rest of it in the old
        // file, under a parent whose validity now comes from the new one.
        for field in &supplied.fields {
            let mut supplied_ids = Schema {
                fields: vec![field.clone()],
                metadata: Default::default(),
            }
            .field_ids();
            let mut declared_ids = self.schema().project_by_ids(&[field.id], true).field_ids();
            supplied_ids.sort_unstable();
            declared_ids.sort_unstable();
            if supplied_ids != declared_ids {
                return Err(Error::invalid_input(format!(
                    "column '{}' must be supplied whole, but the stream for fragment {id} omits \
                     some of its nested fields",
                    field.name
                )));
            }
        }

        // The fields as the manifest defines them, in its order. Batches are
        // projected onto this by name, so struct children may arrive in any
        // order; nullability is the writer's to enforce against the data.
        let top_level_ids = supplied
            .fields
            .iter()
            .map(|field| field.id)
            .collect::<Vec<_>>();
        let write_schema = self.schema().project_by_ids(&top_level_ids, true);
        let write_fields = ArrowSchema::from(&write_schema)
            .fields()
            .iter()
            .map(|target| {
                let source = value_schema.field_with_name(target.name())?;
                with_source_blob_layout(&relax_nullability(target), source)
            })
            .collect::<Result<Vec<_>>>()?;
        let write_arrow_schema = Arc::new(ArrowSchema::new(write_fields));

        Ok(AlignedColumns {
            source_schema,
            row_addr_index,
            value_indices,
            write_schema,
            adapter: SchemaAdapter::new(value_schema),
            empty_values: RecordBatch::new_empty(write_arrow_schema.clone()),
            write_arrow_schema,
        })
    }

    /// Pull exactly `num_rows` rows from `stream`, carrying any surplus of the
    /// last batch over in `leftover`.
    async fn next_rows(
        &self,
        stream: &mut SendableRecordBatchStream,
        leftover: &mut Option<RecordBatch>,
        num_rows: usize,
        columns: &AlignedColumns,
    ) -> Result<RecordBatch> {
        let mut parts = Vec::new();
        let mut remaining = num_rows;
        while remaining > 0 {
            let batch = match leftover.take() {
                Some(batch) => batch,
                None => {
                    let batch = stream.next().await.ok_or_else(|| {
                        Error::invalid_input(format!(
                            "the stream for fragment {} ended before supplying a row for every \
                             live row",
                            self.id()
                        ))
                    })??;
                    self.check_batch_schema(&batch, columns)?;
                    // An empty slice still holds its parent's buffers, so it is
                    // dropped here instead of being kept in `parts`.
                    if batch.num_rows() == 0 {
                        continue;
                    }
                    batch
                }
            };
            if batch.num_rows() > remaining {
                parts.push(batch.slice(0, remaining));
                *leftover = Some(batch.slice(remaining, batch.num_rows() - remaining));
                remaining = 0;
            } else {
                remaining -= batch.num_rows();
                parts.push(batch);
            }
        }
        // Rebuilt against the declared schema even for a single part (which
        // `concat_batches` does without copying): field metadata such as the
        // JSON extension decides the physical conversion, and a batch may
        // arrive without it.
        Ok(concat_batches(&columns.source_schema, &parts)?)
    }

    /// Columns are taken by position, so a batch whose columns differ from the
    /// declared schema would write each column's values under another's field.
    fn check_batch_schema(&self, batch: &RecordBatch, columns: &AlignedColumns) -> Result<()> {
        let declared = columns.source_schema.fields();
        let actual = batch.schema_ref().fields();
        let matches = declared.len() == actual.len()
            && declared
                .iter()
                .zip(actual.iter())
                .all(|(declared, actual)| {
                    declared.name() == actual.name() && declared.data_type() == actual.data_type()
                });
        if matches {
            return Ok(());
        }
        Err(Error::invalid_input(format!(
            "a batch in the stream for fragment {} does not match the stream schema: expected \
             {}, got {}",
            self.id(),
            ArrowSchema::new(declared.clone()),
            ArrowSchema::new(actual.clone()),
        )))
    }

    /// `supplied` must name, row for row, the live rows in `expected`.
    /// `rows_consumed` is the stream position of its first row.
    fn check_alignment(
        &self,
        expected: &UInt64Array,
        supplied: &RecordBatch,
        columns: &AlignedColumns,
        rows_consumed: usize,
    ) -> Result<()> {
        let supplied_addrs = supplied
            .column(columns.row_addr_index)
            .as_primitive::<UInt64Type>();
        if let Some(row) = (0..supplied_addrs.len()).find(|&row| supplied_addrs.is_null(row)) {
            return Err(Error::invalid_input(format!(
                "stream row {} for fragment {} has a null '{ROW_ADDR}'",
                rows_consumed + row,
                self.id()
            )));
        }
        let mismatch = expected
            .values()
            .iter()
            .zip(supplied_addrs.values().iter())
            .position(|(expected, supplied)| expected != supplied);
        if let Some(row) = mismatch {
            return Err(Error::invalid_input(format!(
                "stream row {} for fragment {} has '{ROW_ADDR}' {}, but the live row at that \
                 position is {}; the stream must supply every live row once, in ascending \
                 '{ROW_ADDR}' order",
                rows_consumed + row,
                self.id(),
                RowAddress::from(supplied_addrs.value(row)),
                RowAddress::from(expected.value(row)),
            )));
        }
        Ok(())
    }

    /// The columns to write, in the form and field order the file stores them.
    fn value_columns(
        &self,
        supplied: &RecordBatch,
        columns: &AlignedColumns,
    ) -> Result<RecordBatch> {
        let values = columns
            .adapter
            .to_physical_batch(supplied.project(&columns.value_indices)?)?
            .project_by_schema(&columns.write_arrow_schema)?;
        Ok(values)
    }

    /// Every live row has been written, so the stream must have nothing left.
    async fn check_exhausted(
        &self,
        stream: &mut SendableRecordBatchStream,
        leftover: Option<RecordBatch>,
    ) -> Result<()> {
        let surplus = || {
            Error::invalid_input(format!(
                "the stream for fragment {} has more rows than the fragment has live rows",
                self.id()
            ))
        };
        if leftover.is_some_and(|batch| batch.num_rows() > 0) {
            return Err(surplus());
        }
        while let Some(batch) = stream.next().await {
            if batch?.num_rows() > 0 {
                return Err(surplus());
            }
        }
        Ok(())
    }
}

/// `target` with each blob v2 node, at any depth, given the logical layout
/// the source supplies, so projecting onto it keeps `position` and `size`
/// instead of cutting the struct down to the manifest's `data, uri`.
fn with_source_blob_layout(target: &ArrowField, source: &ArrowField) -> Result<ArrowField> {
    let nested = |target: &ArrowField, source: &ArrowField| {
        with_source_blob_layout(target, source).map(Arc::new)
    };
    let data_type = match (target.data_type(), source.data_type()) {
        _ if target.is_blob_v2() => DataType::Struct(logical_blob_children(target, source)?),
        (DataType::Struct(targets), DataType::Struct(sources)) => DataType::Struct(
            targets
                .iter()
                .map(
                    |target| match sources.iter().find(|s| s.name() == target.name()) {
                        Some(source) => nested(target, source),
                        None => Ok(target.clone()),
                    },
                )
                .collect::<Result<Vec<_>>>()?
                .into(),
        ),
        (DataType::List(target), DataType::List(source)) => DataType::List(nested(target, source)?),
        (DataType::LargeList(target), DataType::LargeList(source)) => {
            DataType::LargeList(nested(target, source)?)
        }
        (DataType::FixedSizeList(target, size), DataType::FixedSizeList(source, _)) => {
            DataType::FixedSizeList(nested(target, source)?, *size)
        }
        (DataType::Map(target, sorted), DataType::Map(source, _)) => {
            DataType::Map(nested(target, source)?, *sorted)
        }
        _ => return Ok(target.clone()),
    };
    Ok(target.clone().with_data_type(data_type))
}

/// The logical blob layout whose children the source names, in canonical
/// order. Prepared and descriptor input are refused: their packed and
/// dedicated rows point at sidecars relative to a data file the caller cannot
/// know the name of.
fn logical_blob_children(target: &ArrowField, source: &ArrowField) -> Result<Fields> {
    let supplied = match source.data_type() {
        DataType::Struct(children) => children
            .iter()
            .map(|child| (child.name().as_str(), child.data_type()))
            .collect::<HashSet<_>>(),
        _ => HashSet::new(),
    };
    for layout in [&*BLOB_V2_LOGICAL_MINIMAL_FIELDS, &*BLOB_V2_LOGICAL_FIELDS] {
        if supplied.len() == layout.len()
            && layout
                .iter()
                .all(|f| supplied.contains(&(f.name().as_str(), f.data_type())))
        {
            return Ok(layout
                .iter()
                .map(|field| Arc::new(relax_nullability(field)))
                .collect());
        }
    }
    Err(Error::invalid_input(format!(
        "blob column '{}' must be supplied as logical blobs (data, uri, and optionally \
         position and size), got {}",
        target.name(),
        source.data_type()
    )))
}
