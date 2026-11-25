// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Writing data overlay files.
//!
//! An overlay supplies new values for a subset of a fragment's
//! `(offset_in_frag, field)` cells without rewriting its base data files. This
//! module stages one: [`OverlayWriter`] takes batches of values keyed by
//! `_rowaddr`, writes them into a value file, and accumulates the coverage the
//! read path resolves them by. See the [module documentation](super) for the
//! coordinate spaces involved.
//!
//! Two invariants are the caller's to satisfy and this writer's to enforce,
//! because nothing downstream can:
//!
//! - **Values are addressed by rank.** An overlay stores a field's values
//!   densely, in ascending `offset_in_frag` order, and a covered cell's value is
//!   found by counting the covered cells below it. A stream that revisited an
//!   offset it had passed would put its values on the wrong rows, and every
//!   later value with them. Offsets must therefore strictly ascend, per field,
//!   across the writer's whole lifetime.
//! - **Coverage is not value-nullness.** A covered offset holding NULL overrides
//!   that cell *to* NULL; an offset absent from the coverage falls through to
//!   the base value. Passing a null to mean "leave this cell alone" silently
//!   erases live data.
//!
//! Neither the commit path nor the read path can catch a violation of either:
//! `build_manifest` checks that an overlay's target fragment exists and that a
//! fragment's overlays are ordered newest-last, and resolution trusts the
//! coverage it is handed. This writer is where they are checked.

use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{Array, RecordBatch};
use arrow_schema::{DataType, FieldRef as ArrowFieldRef, Schema as ArrowSchema};
use object_store::path::Path;
use roaring::RoaringBitmap;
use snafu::Snafu;

use lance_arrow::RecordBatchExt;
use lance_core::datatypes::{
    NullabilityComparison, Schema, Schema as LanceSchema, SchemaCompareOptions,
};
use lance_core::utils::address::RowAddress;
use lance_core::{Error, ROW_ADDR};
use lance_file::version::ConcreteFileVersion;
use lance_table::format::overlay::{DataOverlayFile, OverlayCoverage};

use crate::fragment::{discard_staged_file, duplicate_field_path, relax_nullability};
use crate::write::{GenericWriter, open_update_writer, writes_blob_v2};
use crate::{FileFragment, Table};
use lance_table::transaction::DataOverlayGroup;

/// Why staging a data overlay failed.
#[derive(Debug, Snafu)]
pub enum WriteOverlayError {
    /// The declared schema names a field the dataset does not define, or defines
    /// differently. `source` names the field and the difference.
    #[snafu(display(
        "overlay for fragment {fragment_id} declared a schema the dataset does not match: {source}"
    ))]
    SchemaMismatch { fragment_id: u64, source: Error },

    /// Overlays address cells by physical offset within a V2 file. The legacy
    /// reader pairs a fragment's files by batch boundary and has no notion of
    /// per-field coverage.
    #[snafu(display(
        "cannot write an overlay for fragment {fragment_id}: the dataset uses the legacy file format"
    ))]
    LegacyFileFormat { fragment_id: u64 },

    /// Blob v2 columns carry external descriptors that overlay resolution does
    /// not read.
    #[snafu(display(
        "cannot overlay field '{name}' of fragment {fragment_id}: blob columns are not supported"
    ))]
    BlobField { fragment_id: u64, name: String },

    /// A batch arrived without the `_rowaddr` column naming the cells its values
    /// apply to.
    #[snafu(display("overlay batch for fragment {fragment_id} has no '{ROW_ADDR}' column"))]
    MissingRowAddr { fragment_id: u64 },

    /// `_rowaddr` was present but not `UInt64`, so it holds something other than
    /// row addresses.
    #[snafu(display(
        "overlay batch for fragment {fragment_id} has a '{ROW_ADDR}' of type {data_type}, \
         expected {expected}"
    ))]
    RowAddrType {
        fragment_id: u64,
        data_type: DataType,
        expected: DataType,
    },

    /// `_rowaddr` is nullable — scans use its validity bitmap as a selection
    /// vector, so a deleted row can read back as a null address. Skipping those
    /// rows would drop a value the caller computed and shift every later value's
    /// rank by one, so they are refused instead.
    #[snafu(display(
        "overlay batch for fragment {fragment_id} has a null '{ROW_ADDR}' at batch offset \
         {offset_in_batch}; filter deleted rows out before staging them"
    ))]
    NullRowAddr {
        fragment_id: u64,
        offset_in_batch: usize,
    },

    /// An address belongs to a different fragment. Reached when one scan spans a
    /// range of fragments and its batches are not split before being staged.
    #[snafu(display(
        "overlay for fragment {fragment_id} received a row address for fragment {}: {row_addr}",
        RowAddress::new_from_u64(*row_addr).fragment_id()
    ))]
    ForeignRowAddr { fragment_id: u64, row_addr: u64 },

    /// An address names an offset past the end of the fragment.
    #[snafu(display(
        "overlay for fragment {fragment_id} received offset {offset_in_frag}, but the fragment has \
         {physical_rows} physical rows"
    ))]
    OffsetOutOfRange {
        fragment_id: u64,
        offset_in_frag: u32,
        physical_rows: u64,
    },

    /// Offsets did not strictly ascend, so values would be stored at a rank
    /// other than the one they will be read back at. `name` is the field the
    /// ordering broke for, or `_rowaddr` when a single batch is out of order.
    #[snafu(display(
        "overlay for fragment {fragment_id} received offset {offset_in_frag} for '{name}' after \
         offset {previous}; offsets must strictly ascend per field"
    ))]
    OffsetNotAscending {
        fragment_id: u64,
        name: String,
        previous: u32,
        offset_in_frag: u32,
    },

    /// A batch carried a column the overlay was not opened for. An overlay's
    /// fields are fixed when it opens, because its value file's columns are.
    #[snafu(display(
        "overlay batch for fragment {fragment_id} has column '{name}', which the overlay was not \
         opened for"
    ))]
    UndeclaredColumn { fragment_id: u64, name: String },

    /// A batch named the same column twice, at the top level or under a struct.
    /// Columns and their children are matched to fields by name, so a repeat has
    /// no single field to stage against and projection would silently keep the
    /// first. `name` is the dotted path to the repeated name.
    #[snafu(display("overlay batch for fragment {fragment_id} has column '{name}' twice"))]
    DuplicateColumn { fragment_id: u64, name: String },

    /// A batch's columns matched the overlay's fields by name but not by shape.
    /// Projection descends by name and downcasts by the *target's* type, so a
    /// mismatch below the top level would panic in the Arrow cast rather than
    /// surface here. The whole tree is compared before projecting so it does
    /// not. `source` names the field and the difference.
    #[snafu(display(
        "overlay batch for fragment {fragment_id} does not match the fields it was staged \
         for: {source}"
    ))]
    BatchSchemaMismatch { fragment_id: u64, source: Error },

    /// Anything not overlay-specific: I/O, encoding, or values that do not match
    /// the field they are staged for.
    #[snafu(display("{source}"))]
    Other { source: Error },
}

impl From<Error> for WriteOverlayError {
    fn from(source: Error) -> Self {
        Self::Other { source }
    }
}

impl From<WriteOverlayError> for Error {
    fn from(error: WriteOverlayError) -> Self {
        match error {
            // Already a lance error, and one that already names the fragment
            // and the offending field. Re-wrapping would bury the variant
            // behind an InvalidInput that says less than the error it holds,
            // and restate what the message already carries.
            WriteOverlayError::Other { source } => source,
            WriteOverlayError::SchemaMismatch { source, .. } => source,
            WriteOverlayError::LegacyFileFormat { .. } | WriteOverlayError::BlobField { .. } => {
                Self::not_supported(error.to_string())
            }
            caller_error => Self::invalid_input(caller_error.to_string()),
        }
    }
}

type Result<T> = std::result::Result<T, WriteOverlayError>;

/// Stages one data overlay file for one fragment.
///
/// Open it with [`FileFragment::write_overlay`], feed it batches keyed by
/// `_rowaddr`, and [`finish`](Self::finish) it into a [`DataOverlayGroup`] ready
/// for `Operation::DataOverlay`. On an error of the caller's own,
/// [`abort`](Self::abort) discards the staged file. If an error occurs during
/// the call to [`finish`](Self::finish), the staged file is cleaned up automatically.
///
/// # The overlay is bound to the dataset version it was opened on
///
/// Coverage is addressed by physical offset, so a staged overlay only means
/// what it says against the fragment layout it was staged over: `open` range-
/// checks every address against that snapshot's `physical_rows`, and the read
/// path resolves the offsets literally. The writer does not track a read
/// version of its own — the caller commits one, and it must be the version of
/// the `Table` this writer was opened on, not the latest version at commit
/// time. That is what makes the conflict resolver replay the intervening
/// commits and reject a concurrent `Rewrite`, `Merge`, or row-moving `Update`
/// that moved the rows out from under these offsets. Committing a newer read
/// version skips those checks and lands values on the wrong rows.
///
/// A caller staging several fragments, or several groups, commits them under
/// one read version, so every writer in that batch must be opened on the same
/// `Table` snapshot.
pub struct OverlayWriter {
    dataset: Table,
    fragment_id: u64,
    physical_rows: u64,
    /// The overlay's fields, resolved against the manifest. Column `i` of the
    /// value file stores `schema.fields[i]`.
    schema: Schema,
    /// `schema` as Arrow fields with nullability relaxed, for projecting an
    /// incoming batch onto the field it is staged for. Built once.
    arrow_fields: Vec<ArrowFieldRef>,
    writer: Box<dyn GenericWriter>,
    staged_path: Path,
    /// Coverage accumulated per field, indexed as `schema.fields`.
    coverage: Vec<RoaringBitmap>,
    /// The last offset staged for each field, for the strict-ascension check.
    /// `None` until that field is first written.
    last_offset_in_frag: Vec<Option<u32>>,
}

impl OverlayWriter {
    async fn open(fragment: &FileFragment, schema: &Schema) -> Result<Self> {
        let fragment_id = fragment.id() as u64;
        let dataset = fragment.table();

        let file_version = dataset.manifest.data_storage_format.lance_file_format();
        if file_version == ConcreteFileVersion::V1 {
            return Err(WriteOverlayError::LegacyFileFormat { fragment_id });
        }

        let schema = fragment.resolve_writer_schema(schema).map_err(|source| {
            WriteOverlayError::SchemaMismatch {
                fragment_id,
                source,
            }
        })?;
        // Blob v1 columns are stored as Blob v2 from 2.2 on, and the Blob v2
        // writer only accepts whole batches.
        let blob_v2_file = writes_blob_v2(file_version);
        if let Some(blob) = schema
            .fields_pre_order()
            .find(|field| field.is_blob_v2() || (blob_v2_file && field.is_blob()))
        {
            return Err(WriteOverlayError::BlobField {
                fragment_id,
                name: blob.name.clone(),
            });
        }
        let physical_rows = fragment.physical_rows().await? as u64;
        // Blob v2 is rejected above, so no external base resolution is needed.
        let writer = open_update_writer(file_version, dataset, &schema, false).await?;
        let staged_path = {
            let (file_name, _) = writer.data_file_path();
            dataset.data_dir().join(file_name)
        };

        let arrow_fields = ArrowSchema::from(&schema)
            .fields()
            .iter()
            .map(|field| Arc::new(relax_nullability(field)))
            .collect::<Vec<_>>();
        let field_count = schema.fields.len();
        Ok(Self {
            dataset: dataset.clone(),
            fragment_id,
            physical_rows,
            schema,
            arrow_fields,
            writer,
            staged_path,
            coverage: vec![RoaringBitmap::new(); field_count],
            last_offset_in_frag: vec![None; field_count],
        })
    }

    /// Supply values for some of the overlay's fields.
    ///
    /// `_rowaddr` is required and names the cell each row's values apply to. The
    /// addresses must belong to this fragment and strictly ascend. Every other
    /// column must be one of the fields the overlay was opened for, and a batch
    /// may carry any subset of them — a field's coverage is the offsets it is
    /// given, accumulated across batches.
    ///
    /// Passing a NULL value covers that cell *with* NULL, overriding the base
    /// value. To leave a cell alone, omit its row.
    pub async fn write_batch(&mut self, data: &RecordBatch) -> Result<()> {
        let offsets_in_frag = self.validated_offsets(data)?;
        let positions = self.validated_columns(data)?;
        let Some(&first_offset_in_frag) = offsets_in_frag.first() else {
            return Ok(());
        };

        for &position in &positions {
            if let Some(previous) = self.last_offset_in_frag[position]
                && previous >= first_offset_in_frag
            {
                return Err(WriteOverlayError::OffsetNotAscending {
                    fragment_id: self.fragment_id,
                    name: self.schema.fields[position].name.clone(),
                    previous,
                    offset_in_frag: first_offset_in_frag,
                });
            }
        }

        // Project rather than take columns as they arrive: a struct's children
        // may be in any order, and values that do not match the field they are
        // staged for are rejected here instead of being encoded.
        let projection = ArrowSchema::new(
            positions
                .iter()
                .map(|&position| self.arrow_fields[position].clone())
                .collect::<Vec<_>>(),
        );
        let projected = data.project_by_schema(&projection).map_err(Error::from)?;

        for (projected_index, &position) in positions.iter().enumerate() {
            self.writer
                .write_column(position, projected.column(projected_index).clone())
                .await?;
            self.coverage[position].extend(offsets_in_frag.iter().copied());
            self.last_offset_in_frag[position] = offsets_in_frag.last().copied();
        }
        Ok(())
    }

    /// Finish the overlay.
    ///
    /// Fields given identical coverage yield a dense overlay
    /// ([`OverlayCoverage::Shared`]); otherwise coverage is recorded per field.
    /// A field that was never written keeps an empty coverage and a zero-length
    /// value column, contributing nothing on read.
    ///
    /// Returns `Ok(None)` when no cell was staged at all: the file is discarded
    /// and there is nothing to commit. The returned overlay's
    /// `committed_version` is a placeholder — the commit stamps the version it
    /// produces, and re-stamps it on a conflict retry.
    ///
    /// The group carries no read version. Commit it with the version of the
    /// `Table` this writer was opened on; see the [type
    /// documentation](Self) for why a later one is unsafe.
    pub async fn finish(mut self) -> Result<Option<DataOverlayGroup>> {
        let coverage_by_schema_field = std::mem::take(&mut self.coverage);
        if coverage_by_schema_field.iter().all(RoaringBitmap::is_empty) {
            self.discard().await;
            return Ok(None);
        }
        // Either shape resolves correctly; a shared bitmap is the cheaper one,
        // storing a single bitmap in the manifest instead of N identical copies
        // of it. Most overlays fill every declared field on the rows they touch.
        let is_dense = coverage_by_schema_field
            .windows(2)
            .all(|pair| pair[0] == pair[1]);

        let (_, data_file) = match self.writer.finish().await {
            Ok(finished) => finished,
            Err(error) => {
                self.discard().await;
                return Err(error.into());
            }
        };

        let coverage = if is_dense {
            // `windows(2)` is vacuously true for one field, and the zero-field
            // case returned above, so there is always a first bitmap here.
            let Some(shared) = coverage_by_schema_field.into_iter().next() else {
                return Err(Error::internal(format!(
                    "overlay for fragment {} staged values for no field at all",
                    self.fragment_id
                ))
                .into());
            };
            OverlayCoverage::dense(shared)
        } else {
            // Sparse coverage is indexed by position in `data_file.fields`, which
            // lists every field the file holds -- a struct's children as well as
            // the struct itself -- in the encoder's order, not the schema's. A
            // column is staged whole, so every field under a top-level column
            // shares that column's coverage.
            let mut coverage_by_file_field = Vec::with_capacity(data_file.fields.len());
            for &field_id in data_file.fields.iter() {
                let position = self
                    .schema
                    .fields
                    .iter()
                    .position(|top| top.id == field_id || top.field_by_id(field_id).is_some());
                let Some(position) = position else {
                    return Err(Error::internal(format!(
                        "overlay for fragment {} staged a file holding field id {field_id}, which \
                         is not one of the fields it was opened for",
                        self.fragment_id
                    ))
                    .into());
                };
                coverage_by_file_field.push(coverage_by_schema_field[position].clone());
            }
            OverlayCoverage::sparse(coverage_by_file_field)
        };

        Ok(Some(DataOverlayGroup {
            fragment_id: self.fragment_id,
            overlays: vec![DataOverlayFile {
                data_file,
                coverage,
                committed_version: 0,
            }],
        }))
    }

    /// Discard the staged file without committing anything.
    ///
    /// Best effort: a file left behind is unreferenced either way, and cleanup reclaims it.
    ///
    /// This is not callable after calling [`finish`](Self::finish). That method
    /// will invoke the cleanup in the case of errors automatically.
    pub async fn abort(self) {
        self.discard().await
    }

    async fn discard(self) {
        let Self {
            dataset,
            writer,
            staged_path,
            ..
        } = self;
        // The writer may still hold the file open (a buffered upload, an
        // unflushed local handle); release it before deleting.
        drop(writer);
        discard_staged_file(&dataset, &staged_path).await;
    }

    /// The batch's addresses as fragment-local offsets, checked for nullness,
    /// fragment, range, and strict ascension within the batch.
    fn validated_offsets(&self, batch: &RecordBatch) -> Result<Vec<u32>> {
        let column = batch
            .column_by_name(ROW_ADDR)
            .ok_or(WriteOverlayError::MissingRowAddr {
                fragment_id: self.fragment_id,
            })?;
        let addrs = column.as_primitive_opt::<UInt64Type>().ok_or_else(|| {
            WriteOverlayError::RowAddrType {
                fragment_id: self.fragment_id,
                data_type: column.data_type().clone(),
                expected: DataType::UInt64,
            }
        })?;

        let mut offsets_in_frag = Vec::with_capacity(addrs.len());
        let mut previous: Option<u32> = None;
        for offset_in_batch in 0..addrs.len() {
            if addrs.is_null(offset_in_batch) {
                return Err(WriteOverlayError::NullRowAddr {
                    fragment_id: self.fragment_id,
                    offset_in_batch,
                });
            }
            let row_addr = addrs.value(offset_in_batch);
            let address = RowAddress::new_from_u64(row_addr);
            if u64::from(address.fragment_id()) != self.fragment_id {
                return Err(WriteOverlayError::ForeignRowAddr {
                    fragment_id: self.fragment_id,
                    row_addr,
                });
            }
            let offset_in_frag = address.row_offset();
            if u64::from(offset_in_frag) >= self.physical_rows {
                return Err(WriteOverlayError::OffsetOutOfRange {
                    fragment_id: self.fragment_id,
                    offset_in_frag,
                    physical_rows: self.physical_rows,
                });
            }
            if let Some(previous) = previous
                && previous >= offset_in_frag
            {
                return Err(WriteOverlayError::OffsetNotAscending {
                    fragment_id: self.fragment_id,
                    name: ROW_ADDR.to_string(),
                    previous,
                    offset_in_frag,
                });
            }
            previous = Some(offset_in_frag);
            offsets_in_frag.push(offset_in_frag);
        }
        Ok(offsets_in_frag)
    }

    /// The overlay-field positions this batch supplies, in schema order.
    ///
    /// Admits the batch before anything downcasts it: a duplicate name anywhere
    /// in the tree, a column the overlay was not opened for, and any shape
    /// difference against the fields it was opened for are all rejected here.
    /// `write_batch` projects by name and Arrow's projection then downcasts to
    /// the *target's* type, so a nested mismatch left to it would panic instead
    /// of returning an error, and a duplicate child would be resolved to
    /// whichever came first.
    fn validated_columns(&self, batch: &RecordBatch) -> Result<Vec<usize>> {
        if let Some(duplicate) = duplicate_field_path(batch.schema_ref().fields(), "") {
            return Err(WriteOverlayError::DuplicateColumn {
                fragment_id: self.fragment_id,
                name: duplicate,
            });
        }

        let mut present = HashSet::with_capacity(batch.num_columns());
        for field in batch.schema_ref().fields() {
            if field.name() == ROW_ADDR {
                continue;
            }
            if !self
                .schema
                .fields
                .iter()
                .any(|declared| &declared.name == field.name())
            {
                return Err(WriteOverlayError::UndeclaredColumn {
                    fragment_id: self.fragment_id,
                    name: field.name().clone(),
                });
            }
            present.insert(field.name().clone());
        }

        let positions = self
            .schema
            .fields
            .iter()
            .enumerate()
            .filter(|(_, field)| present.contains(&field.name))
            .map(|(position, _)| position)
            .collect::<Vec<_>>();

        // Compare the whole tree, not just the names checked above. A column is
        // staged whole, so a struct must arrive with every child the overlay was
        // opened for -- a partial struct would encode NULLs into the children it
        // omitted while claiming coverage for them. Field order is the caller's
        // to choose, and nullability is enforced by the writer against the data
        // rather than against the declared schema.
        let staged = ArrowSchema::new(
            batch
                .schema_ref()
                .fields()
                .iter()
                .filter(|field| field.name() != ROW_ADDR)
                .cloned()
                .collect::<Vec<_>>(),
        );
        let expected = ArrowSchema::new(
            positions
                .iter()
                .map(|&position| self.arrow_fields[position].clone())
                .collect::<Vec<_>>(),
        );
        LanceSchema::try_from(&staged)
            .and_then(|staged| {
                staged.check_compatible(
                    &LanceSchema::try_from(&expected)?,
                    &SchemaCompareOptions {
                        compare_nullability: NullabilityComparison::Ignore,
                        ignore_field_order: true,
                        ..Default::default()
                    },
                )
            })
            .map_err(|source| WriteOverlayError::BatchSchemaMismatch {
                fragment_id: self.fragment_id,
                source,
            })?;

        Ok(positions)
    }
}

impl FileFragment {
    /// Open a sparse overlay writer for these existing top-level columns.
    ///
    /// Feed it batches keyed by physical row address and commit the completed
    /// group against the snapshot this fragment belongs to.
    pub async fn write_overlay(&self, schema: &Schema) -> Result<OverlayWriter> {
        OverlayWriter::open(self, schema).await
    }
}
