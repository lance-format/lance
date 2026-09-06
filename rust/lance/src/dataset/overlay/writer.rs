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
use lance_core::datatypes::Schema;
use lance_core::utils::address::RowAddress;
use lance_core::{Error, ROW_ADDR};
use lance_file::version::ConcreteFileVersion;
use lance_table::format::overlay::{DataOverlayFile, OverlayCoverage};

use crate::Dataset;
use crate::dataset::fragment::{FileFragment, discard_staged_file, relax_nullability};
use crate::dataset::transaction::DataOverlayGroup;
use crate::dataset::versions;
use crate::dataset::write::GenericWriter;

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

    /// A batch named the same column twice. Columns are matched to fields by
    /// name, so a repeat has no single field to stage against.
    #[snafu(display("overlay batch for fragment {fragment_id} has column '{name}' twice"))]
    DuplicateColumn { fragment_id: u64, name: String },

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
/// [`abort`](Self::abort) discards the staged file; `finish` cleans up after its
/// own failures.
///
/// The writer owns everything it needs, so it is `Send + 'static` and several
/// can be open at once — one per fragment when a single scan spans a range.
pub struct OverlayWriter {
    dataset: Arc<Dataset>,
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
    pub(crate) async fn open(
        dataset: Arc<Dataset>,
        fragment: &FileFragment,
        schema: &Schema,
    ) -> Result<Self> {
        let fragment_id = fragment.id() as u64;

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
        if let Some(blob) = schema.fields_pre_order().find(|field| field.is_blob_v2()) {
            return Err(WriteOverlayError::BlobField {
                fragment_id,
                name: blob.name.clone(),
            });
        }

        let physical_rows = fragment.physical_rows().await? as u64;
        // Blob v2 is rejected above, so no external base resolution is needed.
        let writer =
            versions::open_update_writer(file_version, dataset.as_ref(), &schema, false).await?;
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
            dataset,
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
    ///
    /// The batch is checked in full before anything is written, so a batch
    /// rejected by any of the rules above contributes nothing to the staged
    /// file and the writer can be used again. An error from underneath —
    /// [`Other`](WriteOverlayError::Other), an I/O or encoding failure — can
    /// arrive with part of the batch already written, so the writer must be
    /// discarded rather than reused: call [`abort`](Self::abort), or drop it
    /// and let cleanup reclaim the staged file.
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
    /// For the caller's own error paths — [`finish`](Self::finish) already
    /// cleans up after its own failures. Best effort: a file left behind is
    /// unreferenced either way, and cleanup reclaims it.
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
        discard_staged_file(dataset.as_ref(), &staged_path).await;
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
    fn validated_columns(&self, batch: &RecordBatch) -> Result<Vec<usize>> {
        let mut present = HashSet::with_capacity(batch.num_columns());
        for field in batch.schema_ref().fields() {
            if field.name() == ROW_ADDR {
                continue;
            }
            if !present.insert(field.name().clone()) {
                return Err(WriteOverlayError::DuplicateColumn {
                    fragment_id: self.fragment_id,
                    name: field.name().clone(),
                });
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
        }
        Ok(self
            .schema
            .fields
            .iter()
            .enumerate()
            .filter(|(_, field)| present.contains(&field.name))
            .map(|(position, _)| position)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::types::Int32Type;
    use arrow_array::{
        ArrayRef, Int32Array, RecordBatchIterator, StructArray, UInt64Array, record_batch,
    };
    use arrow_schema::Field as ArrowField;
    use lance_file::version::LanceFileVersion;
    use rstest::rstest;
    use tempfile::{TempDir, tempdir};

    use crate::dataset::transaction::Operation;
    use crate::dataset::{DATA_DIR, WriteDestination, WriteParams};

    /// Two six-row fragments of `id` (field 0) = 0..12, `val` (field 1) = id * 10,
    /// and `tag` (field 2) = id * 100. Backed by a temp dir so tests can count the
    /// files left in `data/`.
    async fn test_dataset() -> (TempDir, Arc<Dataset>) {
        test_dataset_with(None).await
    }

    async fn test_dataset_with(
        data_storage_version: Option<LanceFileVersion>,
    ) -> (TempDir, Arc<Dataset>) {
        let test_dir = tempdir().unwrap();
        let uri = test_dir.path().to_str().unwrap().to_string();
        let batch = record_batch!(
            ("id", Int32, (0..12).collect::<Vec<_>>()),
            ("val", Int32, (0..12).map(|v| v * 10).collect::<Vec<_>>()),
            ("tag", Int32, (0..12).map(|v| v * 100).collect::<Vec<_>>())
        )
        .unwrap();
        let schema = batch.schema();
        let params = WriteParams {
            max_rows_per_file: 6,
            max_rows_per_group: 6,
            data_storage_version,
            ..Default::default()
        };
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let dataset = Dataset::write(reader, &uri, Some(params)).await.unwrap();
        (test_dir, Arc::new(dataset))
    }

    /// The overlay schema for `val` alone, or for `val` and `tag`.
    fn overlay_schema(dataset: &Dataset, ids: &[i32]) -> Schema {
        dataset.schema().project_by_ids(ids, true)
    }

    fn addrs(fragment_id: u32, offsets: &[u32]) -> Vec<u64> {
        offsets
            .iter()
            .map(|&offset| u64::from(RowAddress::new_from_parts(fragment_id, offset)))
            .collect()
    }

    fn int32(values: Vec<Option<i32>>) -> ArrayRef {
        Arc::new(Int32Array::from(values))
    }

    /// `_rowaddr` for `offsets` of `fragment_id`, alongside the given columns.
    fn batch(fragment_id: u32, offsets: &[u32], columns: Vec<(&str, ArrayRef)>) -> RecordBatch {
        let addrs: ArrayRef = Arc::new(UInt64Array::from(addrs(fragment_id, offsets)));
        let columns = std::iter::once((ROW_ADDR, addrs, true))
            .chain(columns.into_iter().map(|(name, array)| (name, array, true)));
        RecordBatch::try_from_iter_with_nullable(columns).unwrap()
    }

    fn val_batch(offsets: &[u32], values: Vec<Option<i32>>) -> RecordBatch {
        batch(0, offsets, vec![("val", int32(values))])
    }

    fn tag_batch(offsets: &[u32], values: Vec<Option<i32>>) -> RecordBatch {
        batch(0, offsets, vec![("tag", int32(values))])
    }

    async fn open_writer(dataset: &Arc<Dataset>, ids: &[i32]) -> OverlayWriter {
        open_writer_on(dataset, 0, ids).await
    }

    async fn open_writer_on(
        dataset: &Arc<Dataset>,
        fragment_id: usize,
        ids: &[i32],
    ) -> OverlayWriter {
        let schema = overlay_schema(dataset, ids);
        dataset
            .get_fragment(fragment_id)
            .unwrap()
            .write_overlay(&schema)
            .await
            .unwrap()
    }

    async fn commit(dataset: Arc<Dataset>, group: DataOverlayGroup) -> Dataset {
        commit_groups(dataset, vec![group]).await
    }

    async fn commit_groups(dataset: Arc<Dataset>, groups: Vec<DataOverlayGroup>) -> Dataset {
        let read_version = dataset.version().version;
        Dataset::commit(
            WriteDestination::Dataset(dataset),
            Operation::DataOverlay { groups },
            Some(read_version),
            None,
            None,
            Arc::new(Default::default()),
            false,
        )
        .await
        .unwrap()
    }

    /// `id -> (val, tag)`, nulls included, ordered by `id`.
    async fn scanned(dataset: &Dataset) -> Vec<(i32, Option<i32>, Option<i32>)> {
        let batch = dataset
            .scan()
            .project(&["id", "val", "tag"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let column = |name: &str| batch[name].as_primitive::<Int32Type>().clone();
        let (ids, vals, tags) = (column("id"), column("val"), column("tag"));
        let mut rows = (0..batch.num_rows())
            .map(|i| {
                (
                    ids.value(i),
                    vals.is_valid(i).then(|| vals.value(i)),
                    tags.is_valid(i).then(|| tags.value(i)),
                )
            })
            .collect::<Vec<_>>();
        rows.sort_by_key(|row| row.0);
        rows
    }

    /// Every entry in `data/`, in-progress temp files included: on local storage
    /// a file being written lives there under a temporary name until it is
    /// finished, so this catches a leak whether or not the writer got as far as
    /// publishing the `.lance` path.
    fn staged_file_count(test_dir: &TempDir) -> usize {
        std::fs::read_dir(test_dir.path().join(DATA_DIR))
            .unwrap()
            .count()
    }

    fn shared_coverage(group: &DataOverlayGroup) -> &RoaringBitmap {
        match &group.overlays[0].coverage {
            OverlayCoverage::Shared(bitmap) => bitmap,
            other => panic!("expected shared coverage, got {other:?}"),
        }
    }

    /// The coverage the read path will resolve for `field_id`, looked up the way
    /// `plan_overlays` looks it up: by the field's position in `data_file.fields`.
    fn coverage_of(group: &DataOverlayGroup, field_id: i32) -> RoaringBitmap {
        let overlay = &group.overlays[0];
        let position = overlay
            .data_file
            .fields
            .iter()
            .position(|&id| id == field_id)
            .unwrap_or_else(|| {
                panic!(
                    "field {field_id} is not in the overlay's fields {:?}",
                    overlay.data_file.fields
                )
            });
        (*overlay.coverage_for_field(position).unwrap()).clone()
    }

    fn field_id(dataset: &Dataset, name: &str) -> i32 {
        dataset.schema().field(name).unwrap().id
    }

    #[tokio::test]
    async fn test_overlay_overrides_covered_cells() {
        let (_test_dir, dataset) = test_dataset().await;
        let mut writer = open_writer(&dataset, &[1]).await;
        writer
            .write_batch(&val_batch(&[1, 3], vec![Some(-1), Some(-3)]))
            .await
            .unwrap();
        let group = writer.finish().await.unwrap().unwrap();

        assert_eq!(group.fragment_id, 0);
        assert_eq!(*shared_coverage(&group), RoaringBitmap::from_iter([1, 3]));

        let dataset = commit(dataset, group).await;
        let rows = scanned(&dataset).await;
        assert_eq!(rows[1], (1, Some(-1), Some(100)));
        assert_eq!(rows[3], (3, Some(-3), Some(300)));
        // Uncovered rows keep their base values, in this fragment and the next.
        assert_eq!(rows[2], (2, Some(20), Some(200)));
        assert_eq!(rows[7], (7, Some(70), Some(700)));
    }

    #[tokio::test]
    async fn test_covering_a_cell_with_null_overrides_it() {
        let (_test_dir, dataset) = test_dataset().await;
        let mut writer = open_writer(&dataset, &[1]).await;
        writer
            .write_batch(&val_batch(&[2], vec![None]))
            .await
            .unwrap();
        let group = writer.finish().await.unwrap().unwrap();

        let dataset = commit(dataset, group).await;
        assert_eq!(scanned(&dataset).await[2], (2, None, Some(200)));
    }

    #[tokio::test]
    async fn test_batches_accumulate_coverage() {
        let (_test_dir, dataset) = test_dataset().await;
        let mut writer = open_writer(&dataset, &[1]).await;
        writer
            .write_batch(&val_batch(&[0, 1], vec![Some(-1), Some(-2)]))
            .await
            .unwrap();
        writer
            .write_batch(&val_batch(&[4], vec![Some(-5)]))
            .await
            .unwrap();
        let group = writer.finish().await.unwrap().unwrap();
        assert_eq!(
            *shared_coverage(&group),
            RoaringBitmap::from_iter([0, 1, 4])
        );

        let dataset = commit(dataset, group).await;
        let vals = scanned(&dataset)
            .await
            .into_iter()
            .map(|(_, val, _)| val)
            .collect::<Vec<_>>();
        assert_eq!(
            vals[..6],
            [Some(-1), Some(-2), Some(20), Some(30), Some(-5), Some(50)]
        );
    }

    #[tokio::test]
    async fn test_fields_covered_alike_share_one_bitmap() {
        let (_test_dir, dataset) = test_dataset().await;
        let mut writer = open_writer(&dataset, &[1, 2]).await;
        let both = batch(
            0,
            &[0, 2],
            vec![
                ("val", int32(vec![Some(-1), Some(-3)])),
                ("tag", int32(vec![Some(-10), Some(-30)])),
            ],
        );
        writer.write_batch(&both).await.unwrap();
        let group = writer.finish().await.unwrap().unwrap();
        assert_eq!(*shared_coverage(&group), RoaringBitmap::from_iter([0, 2]));

        let dataset = commit(dataset, group).await;
        let rows = scanned(&dataset).await;
        assert_eq!(rows[0], (0, Some(-1), Some(-10)));
        assert_eq!(rows[2], (2, Some(-3), Some(-30)));
        assert_eq!(rows[1], (1, Some(10), Some(100)));
    }

    #[tokio::test]
    async fn test_fields_covered_differently_get_their_own_bitmaps() {
        let (_test_dir, dataset) = test_dataset().await;
        let mut writer = open_writer(&dataset, &[1, 2]).await;
        writer
            .write_batch(&val_batch(&[0, 1], vec![Some(-1), Some(-2)]))
            .await
            .unwrap();
        writer
            .write_batch(&tag_batch(&[5], vec![Some(-50)]))
            .await
            .unwrap();
        let group = writer.finish().await.unwrap().unwrap();

        assert!(matches!(
            group.overlays[0].coverage,
            OverlayCoverage::PerField(_)
        ));
        assert_eq!(
            coverage_of(&group, field_id(&dataset, "val")),
            RoaringBitmap::from_iter([0, 1])
        );
        assert_eq!(
            coverage_of(&group, field_id(&dataset, "tag")),
            RoaringBitmap::from_iter([5])
        );

        let dataset = commit(dataset, group).await;
        let rows = scanned(&dataset).await;
        assert_eq!(rows[0], (0, Some(-1), Some(0)));
        assert_eq!(rows[1], (1, Some(-2), Some(100)));
        assert_eq!(rows[5], (5, Some(50), Some(-50)));
    }

    #[tokio::test]
    async fn test_declared_field_may_go_unwritten() {
        let (_test_dir, dataset) = test_dataset().await;
        let mut writer = open_writer(&dataset, &[1, 2]).await;
        writer
            .write_batch(&val_batch(&[3], vec![Some(-4)]))
            .await
            .unwrap();
        let group = writer.finish().await.unwrap().unwrap();

        // `tag` still gets a column in the value file, covering nothing -- the
        // shape resolution has to tolerate an empty bitmap rather than a missing
        // field.
        assert_eq!(
            coverage_of(&group, field_id(&dataset, "val")),
            RoaringBitmap::from_iter([3])
        );
        assert!(coverage_of(&group, field_id(&dataset, "tag")).is_empty());

        let dataset = commit(dataset, group).await;
        assert_eq!(scanned(&dataset).await[3], (3, Some(-4), Some(300)));
    }

    #[tokio::test]
    async fn test_nothing_staged_leaves_nothing_behind() {
        let (test_dir, dataset) = test_dataset().await;
        let before = staged_file_count(&test_dir);
        let writer = open_writer(&dataset, &[1]).await;
        assert!(writer.finish().await.unwrap().is_none());
        assert_eq!(staged_file_count(&test_dir), before);
    }

    #[tokio::test]
    async fn test_abort_leaves_nothing_behind() {
        let (test_dir, dataset) = test_dataset().await;
        let before = staged_file_count(&test_dir);
        let mut writer = open_writer(&dataset, &[1]).await;
        writer
            .write_batch(&val_batch(&[0], vec![Some(-1)]))
            .await
            .unwrap();
        writer.abort().await;
        assert_eq!(staged_file_count(&test_dir), before);
    }

    /// Batches an overlay must refuse. Each case writes its batches in turn and
    /// expects the last to be rejected, so a case can first set up the state --
    /// accumulated coverage, a per-field cursor -- that makes it wrong.
    #[rstest]
    #[case::no_row_addr(
        vec![record_batch!(("val", Int32, vec![Some(-1)])).unwrap()],
        |error| matches!(error, WriteOverlayError::MissingRowAddr { .. }),
        "has no '_rowaddr' column"
    )]
    #[case::row_addr_of_the_wrong_type(
        vec![record_batch!(
            (ROW_ADDR, Int32, vec![Some(0)]),
            ("val", Int32, vec![Some(-1)])
        ).unwrap()],
        |error| matches!(error, WriteOverlayError::RowAddrType { .. }),
        "has a '_rowaddr' of type Int32, expected UInt64"
    )]
    #[case::null_row_addr(
        vec![record_batch!(
            (ROW_ADDR, UInt64, vec![Some(addrs(0, &[0])[0]), None]),
            ("val", Int32, vec![Some(-1), Some(-2)])
        ).unwrap()],
        |error| matches!(error, WriteOverlayError::NullRowAddr { offset_in_batch: 1, .. }),
        "null '_rowaddr' at batch offset 1"
    )]
    #[case::row_addr_from_another_fragment(
        vec![batch(1, &[0], vec![("val", int32(vec![Some(-1)]))])],
        |error| matches!(error, WriteOverlayError::ForeignRowAddr { .. }),
        "overlay for fragment 0 received a row address for fragment 1"
    )]
    #[case::offset_past_the_end_of_the_fragment(
        vec![val_batch(&[6], vec![Some(-1)])],
        |error| matches!(
            error,
            WriteOverlayError::OffsetOutOfRange { offset_in_frag: 6, physical_rows: 6, .. }
        ),
        "received offset 6, but the fragment has 6 physical rows"
    )]
    #[case::offsets_descending_within_a_batch(
        vec![val_batch(&[3, 1], vec![Some(-1), Some(-2)])],
        |error| matches!(
            error,
            WriteOverlayError::OffsetNotAscending { previous: 3, offset_in_frag: 1, .. }
        ),
        "received offset 1 for '_rowaddr' after offset 3"
    )]
    #[case::offset_repeated_across_batches(
        vec![
            val_batch(&[1, 3], vec![Some(-1), Some(-2)]),
            val_batch(&[3], vec![Some(-3)]),
        ],
        |error| matches!(
            error,
            WriteOverlayError::OffsetNotAscending { previous: 3, offset_in_frag: 3, .. }
        ),
        "received offset 3 for 'val' after offset 3"
    )]
    #[case::undeclared_column(
        vec![tag_batch(&[0], vec![Some(-1)])],
        |error| matches!(error, WriteOverlayError::UndeclaredColumn { .. }),
        "has column 'tag', which the overlay was not opened for"
    )]
    #[case::duplicate_column(
        vec![record_batch!(
            (ROW_ADDR, UInt64, addrs(0, &[0])),
            ("val", Int32, vec![Some(-1)]),
            ("val", Int32, vec![Some(-2)])
        ).unwrap()],
        |error| matches!(error, WriteOverlayError::DuplicateColumn { .. }),
        "has column 'val' twice"
    )]
    #[case::value_of_the_wrong_type(
        vec![record_batch!(
            (ROW_ADDR, UInt64, addrs(0, &[0])),
            ("val", Utf8, vec![Some("not an int")])
        ).unwrap()],
        |error| matches!(error, WriteOverlayError::Other { .. }),
        "val"
    )]
    #[tokio::test]
    async fn test_bad_batch_is_rejected(
        #[case] batches: Vec<RecordBatch>,
        #[case] expected: fn(WriteOverlayError) -> bool,
        #[case] message: &str,
    ) {
        let (_test_dir, dataset) = test_dataset().await;
        let mut writer = open_writer(&dataset, &[1]).await;
        let (last, accepted) = batches.split_last().unwrap();
        for batch in accepted {
            writer.write_batch(batch).await.unwrap();
        }

        let error = writer.write_batch(last).await.unwrap_err();
        let display = error.to_string();
        assert!(display.contains(message), "{display}");
        assert!(expected(error), "unexpected variant: {display}");
    }

    #[tokio::test]
    async fn test_undeclared_field_id_is_rejected_at_open() {
        let (_test_dir, dataset) = test_dataset().await;
        let mut schema = overlay_schema(&dataset, &[1]);
        schema.fields[0].id = 99;
        let Err(error) = dataset
            .get_fragment(0)
            .unwrap()
            .write_overlay(&schema)
            .await
        else {
            panic!("expected opening the overlay to fail");
        };
        assert!(matches!(error, WriteOverlayError::SchemaMismatch { .. }));
        assert!(
            error
                .to_string()
                .contains("names field id 99 ('val') that the dataset schema does not define"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn test_legacy_file_format_is_rejected() {
        let (_test_dir, dataset) = test_dataset_with(Some(LanceFileVersion::Legacy)).await;
        let schema = overlay_schema(&dataset, &[1]);
        let Err(error) = dataset
            .get_fragment(0)
            .unwrap()
            .write_overlay(&schema)
            .await
        else {
            panic!("expected opening the overlay to fail");
        };
        assert!(matches!(error, WriteOverlayError::LegacyFileFormat { .. }));
        assert!(
            error
                .to_string()
                .contains("the dataset uses the legacy file format"),
            "{error}"
        );
        assert!(matches!(Error::from(error), Error::NotSupported { .. }));
    }

    #[tokio::test]
    async fn test_offsets_are_local_to_the_fragment() {
        let (_test_dir, dataset) = test_dataset().await;
        let mut writer = open_writer_on(&dataset, 1, &[1]).await;
        let values = batch(1, &[0, 2], vec![("val", int32(vec![Some(-1), Some(-3)]))]);
        writer.write_batch(&values).await.unwrap();
        let group = writer.finish().await.unwrap().unwrap();

        assert_eq!(group.fragment_id, 1);
        assert_eq!(*shared_coverage(&group), RoaringBitmap::from_iter([0, 2]));

        let dataset = commit(dataset, group).await;
        let rows = scanned(&dataset).await;
        // Fragment 1 holds ids 6..12, so its offsets 0 and 2 are ids 6 and 8 --
        // not the dataset's rows 0 and 2, which belong to fragment 0.
        assert_eq!(rows[6], (6, Some(-1), Some(600)));
        assert_eq!(rows[8], (8, Some(-3), Some(800)));
        assert_eq!(rows[0], (0, Some(0), Some(0)));
        assert_eq!(rows[2], (2, Some(20), Some(200)));
    }

    #[tokio::test]
    async fn test_one_commit_can_overlay_several_fragments() {
        let (_test_dir, dataset) = test_dataset().await;
        // Both open at once, as a scan spanning the fragments would leave them.
        let mut first = open_writer_on(&dataset, 0, &[1]).await;
        let mut second = open_writer_on(&dataset, 1, &[1]).await;
        first
            .write_batch(&val_batch(&[1], vec![Some(-1)]))
            .await
            .unwrap();
        second
            .write_batch(&batch(1, &[1], vec![("val", int32(vec![Some(-2)]))]))
            .await
            .unwrap();
        let groups = vec![
            first.finish().await.unwrap().unwrap(),
            second.finish().await.unwrap().unwrap(),
        ];

        let dataset = commit_groups(dataset, groups).await;
        let rows = scanned(&dataset).await;
        assert_eq!(rows[1], (1, Some(-1), Some(100)));
        assert_eq!(rows[7], (7, Some(-2), Some(700)));
    }

    #[tokio::test]
    async fn test_offsets_are_physical_so_deletions_do_not_shift_them() {
        let (_test_dir, dataset) = test_dataset().await;
        let mut dataset = Arc::try_unwrap(dataset).unwrap();
        dataset.delete("id = 1").await.unwrap();
        let dataset = Arc::new(dataset);

        let mut writer = open_writer(&dataset, &[1]).await;
        // Physical offsets, so the deleted row still occupies offset 1: id 2 is
        // at offset 2, not at the offset 1 it would hold if deletions closed up.
        // Offset 1 itself stays addressable -- it is in range, and its value is
        // simply unreachable while the row is deleted.
        writer
            .write_batch(&val_batch(&[1, 2, 5], vec![Some(-2), Some(-3), Some(-6)]))
            .await
            .unwrap();
        let group = writer.finish().await.unwrap().unwrap();
        assert_eq!(
            *shared_coverage(&group),
            RoaringBitmap::from_iter([1, 2, 5])
        );

        let dataset = commit(dataset, group).await;
        let rows = scanned(&dataset).await;
        assert!(!rows.iter().any(|row| row.0 == 1), "id 1 is deleted");
        assert_eq!(rows[1], (2, Some(-3), Some(200)));
        assert_eq!(rows[4], (5, Some(-6), Some(500)));
        assert_eq!(rows[0], (0, Some(0), Some(0)));
    }

    #[tokio::test]
    async fn test_range_is_bounded_by_physical_rows_not_live_rows() {
        let (_test_dir, dataset) = test_dataset().await;
        let mut dataset = Arc::try_unwrap(dataset).unwrap();
        dataset.delete("id < 3").await.unwrap();
        let dataset = Arc::new(dataset);

        let mut writer = open_writer(&dataset, &[1]).await;
        // Three of six rows are deleted, but the last physical offset is still 5.
        writer
            .write_batch(&val_batch(&[5], vec![Some(-6)]))
            .await
            .unwrap();
        let error = writer
            .write_batch(&val_batch(&[6], vec![Some(-7)]))
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            WriteOverlayError::OffsetOutOfRange {
                offset_in_frag: 6,
                physical_rows: 6,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn test_ascension_is_tracked_per_field() {
        let (_test_dir, dataset) = test_dataset().await;
        let mut writer = open_writer(&dataset, &[1, 2]).await;
        writer
            .write_batch(&val_batch(&[5], vec![Some(-6)]))
            .await
            .unwrap();
        // A lower offset for a different field is legal: each field has its own
        // cursor. A single shared cursor would reject this.
        writer
            .write_batch(&tag_batch(&[0], vec![Some(-10)]))
            .await
            .unwrap();
        let group = writer.finish().await.unwrap().unwrap();

        let dataset = commit(dataset, group).await;
        let rows = scanned(&dataset).await;
        assert_eq!(rows[5], (5, Some(-6), Some(500)));
        assert_eq!(rows[0], (0, Some(0), Some(-10)));
    }

    #[tokio::test]
    async fn test_newest_overlay_wins() {
        let (_test_dir, dataset) = test_dataset().await;
        let mut writer = open_writer(&dataset, &[1]).await;
        writer
            .write_batch(&val_batch(&[2], vec![Some(-1)]))
            .await
            .unwrap();
        let dataset = Arc::new(commit(dataset, writer.finish().await.unwrap().unwrap()).await);
        assert_eq!(scanned(&dataset).await[2], (2, Some(-1), Some(200)));

        // Overlaying an already-overlaid fragment stacks a second file on it.
        let mut writer = open_writer(&dataset, &[1]).await;
        writer
            .write_batch(&val_batch(&[2], vec![Some(-2)]))
            .await
            .unwrap();
        let dataset = commit(dataset, writer.finish().await.unwrap().unwrap()).await;
        assert_eq!(scanned(&dataset).await[2], (2, Some(-2), Some(200)));
    }

    #[tokio::test]
    async fn test_empty_batch_is_a_no_op() {
        let (_test_dir, dataset) = test_dataset().await;
        let mut writer = open_writer(&dataset, &[1]).await;
        writer
            .write_batch(&val_batch(&[3], vec![Some(-4)]))
            .await
            .unwrap();
        // Neither an empty batch nor one carrying no values disturbs the
        // coverage or the per-field cursor already accumulated.
        writer.write_batch(&val_batch(&[], vec![])).await.unwrap();
        writer.write_batch(&batch(0, &[4], vec![])).await.unwrap();
        writer
            .write_batch(&val_batch(&[5], vec![Some(-6)]))
            .await
            .unwrap();
        let group = writer.finish().await.unwrap().unwrap();
        assert_eq!(*shared_coverage(&group), RoaringBitmap::from_iter([3, 5]));

        let dataset = commit(dataset, group).await;
        let rows = scanned(&dataset).await;
        assert_eq!(rows[3], (3, Some(-4), Some(300)));
        assert_eq!(rows[4], (4, Some(40), Some(400)));
        assert_eq!(rows[5], (5, Some(-6), Some(500)));
    }

    fn nested_array(a: Vec<i32>, b: Vec<i32>) -> ArrayRef {
        let field = |name| Arc::new(ArrowField::new(name, DataType::Int32, true));
        Arc::new(StructArray::from(vec![
            (field("a"), Arc::new(Int32Array::from(a)) as ArrayRef),
            (field("b"), Arc::new(Int32Array::from(b)) as ArrayRef),
        ]))
    }

    #[tokio::test]
    async fn test_struct_column_shares_its_coverage_with_its_children() {
        let test_dir = tempdir().unwrap();
        let uri = test_dir.path().to_str().unwrap().to_string();
        let base = RecordBatch::try_from_iter_with_nullable(vec![
            ("val", int32((0..4).map(Some).collect()), true),
            (
                "nested",
                nested_array((0..4).collect(), (0..4).map(|v| v * 10).collect()),
                true,
            ),
        ])
        .unwrap();
        let arrow_schema = base.schema();
        let reader = RecordBatchIterator::new(vec![Ok(base)], arrow_schema);
        let dataset = Arc::new(Dataset::write(reader, &uri, None).await.unwrap());

        let val_id = field_id(&dataset, "val");
        let nested_id = field_id(&dataset, "nested");
        let child_ids = dataset
            .schema()
            .field("nested")
            .unwrap()
            .children
            .iter()
            .map(|child| child.id)
            .collect::<Vec<_>>();
        assert_eq!(child_ids.len(), 2);

        let mut writer = open_writer_on(&dataset, 0, &[val_id, nested_id]).await;
        writer
            .write_batch(&batch(0, &[0], vec![("val", int32(vec![Some(-1)]))]))
            .await
            .unwrap();
        writer
            .write_batch(&batch(
                0,
                &[2],
                vec![("nested", nested_array(vec![-7], vec![-70]))],
            ))
            .await
            .unwrap();
        let group = writer.finish().await.unwrap().unwrap();

        // The value file lists leaves, so `nested` contributes two entries
        // rather than one: sparse coverage is longer than the overlay's two
        // declared fields, and a child's bitmap has to be found through its
        // parent.
        let overlay = &group.overlays[0];
        assert_eq!(
            overlay.data_file.fields.len(),
            1 + child_ids.len(),
            "expected leaf fields in {:?}",
            overlay.data_file.fields
        );
        assert!(matches!(overlay.coverage, OverlayCoverage::PerField(_)));
        assert_eq!(coverage_of(&group, val_id), RoaringBitmap::from_iter([0]));
        for id in child_ids {
            assert_eq!(
                coverage_of(&group, id),
                RoaringBitmap::from_iter([2]),
                "field {id} should carry the struct's coverage"
            );
        }

        let dataset = commit(dataset, group).await;
        let read_back = dataset
            .scan()
            .project(&["val", "nested"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let vals = read_back["val"].as_primitive::<Int32Type>();
        assert_eq!((vals.value(0), vals.value(2)), (-1, 2));
        let nested = read_back["nested"].as_struct();
        let child = |i: usize| {
            nested
                .column(i)
                .as_primitive::<Int32Type>()
                .values()
                .to_vec()
        };
        assert_eq!((child(0)[2], child(1)[2]), (-7, -70));
        assert_eq!((child(0)[0], child(1)[0]), (0, 0));
    }

    #[tokio::test]
    async fn test_blob_field_is_rejected() {
        let test_dir = tempdir().unwrap();
        let uri = test_dir.path().to_str().unwrap().to_string();
        let field = |name, data_type| Arc::new(ArrowField::new(name, data_type, true));
        let descriptor: ArrayRef = Arc::new(StructArray::from(vec![
            (
                field("data", DataType::LargeBinary),
                Arc::new(arrow_array::LargeBinaryArray::from(vec![Some(
                    b"hello".as_slice(),
                )])) as ArrayRef,
            ),
            (
                field("uri", DataType::Utf8),
                Arc::new(arrow_array::StringArray::from(vec![None::<&str>])) as ArrayRef,
            ),
            (
                field("position", DataType::UInt64),
                Arc::new(UInt64Array::from(vec![None::<u64>])) as ArrayRef,
            ),
            (
                field("size", DataType::UInt64),
                Arc::new(UInt64Array::from(vec![None::<u64>])) as ArrayRef,
            ),
        ]));
        let blob = ArrowField::new("blob", descriptor.data_type().clone(), true).with_metadata(
            [(
                lance_arrow::ARROW_EXT_NAME_KEY.to_string(),
                lance_arrow::BLOB_V2_EXT_NAME.to_string(),
            )]
            .into_iter()
            .collect(),
        );
        let arrow_schema = Arc::new(ArrowSchema::new(vec![blob]));
        let base = RecordBatch::try_new(arrow_schema.clone(), vec![descriptor]).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(base)], arrow_schema);
        // Blob v2 columns need a 2.2 or newer file.
        let params = WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        };
        let dataset = Arc::new(Dataset::write(reader, &uri, Some(params)).await.unwrap());

        let schema = overlay_schema(&dataset, &[field_id(&dataset, "blob")]);
        let Err(error) = dataset
            .get_fragment(0)
            .unwrap()
            .write_overlay(&schema)
            .await
        else {
            panic!("expected opening the overlay to fail");
        };
        assert!(matches!(error, WriteOverlayError::BlobField { .. }));
        assert!(
            error.to_string().contains("blob columns are not supported"),
            "{error}"
        );
        assert!(matches!(Error::from(error), Error::NotSupported { .. }));
    }
}
