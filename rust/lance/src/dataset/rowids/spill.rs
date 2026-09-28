// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Spilling a fragment's row lineage sequences -- its row ids and its
//! created-at and last-updated-at versions -- out of the manifest and into
//! hidden columns of a Lance data file.
//!
//! A row id sequence is run-encoded, so an appended fragment costs about 20
//! bytes of manifest and never needs to leave it. A fragment whose rows came
//! from many places -- the output of compacting a shuffled table, for instance
//! -- has no runs to exploit and falls back to 8 bytes per row. The version
//! sequences are run-length encoded and degrade the same way once a fragment
//! interleaves rows written at many versions. Inline, that cost is paid again
//! in every manifest version, so the manifest grows with the table and every
//! commit rewrites all of it.
//!
//! Spilled, each sequence is an ordinary `UInt64` column carrying one of the
//! reserved field ids [`ROW_ID_FIELD_ID`], [`ROW_CREATED_AT_VERSION_FIELD_ID`]
//! or [`ROW_LAST_UPDATED_AT_VERSION_FIELD_ID`], written with the same encodings
//! and read with the same reader as user data. The file is one of the
//! fragment's own data files, found by that field id; the fragment's metadata
//! only marks the sequence as spilled. A fragment's spilled sequences written
//! here share one such file.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use futures::{FutureExt, StreamExt, TryStreamExt};
use lance_core::datatypes::Schema;
use lance_core::{ROW_CREATED_AT_VERSION, ROW_ID, ROW_LAST_UPDATED_AT_VERSION};
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::LanceEncodingsIo;
use lance_file::reader::{FileReader, ProjectedFileReader, ReaderProjection};
use lance_file::version::ConcreteFileVersion;
use lance_file::versions;
use lance_file::writer::FileWriterOptions;
use lance_io::ReadBatchParams;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_table::format::{
    DataFile, Fragment, ROW_CREATED_AT_VERSION_FIELD_ID, ROW_ID_FIELD_ID,
    ROW_LAST_UPDATED_AT_VERSION_FIELD_ID, RowDatasetVersionMeta, RowDatasetVersionSequence,
    RowIdMeta,
};
use lance_table::rowids::version::write_dataset_versions;
use lance_table::rowids::{RowIdSequence, write_row_ids};
use object_store::path::Path;

use super::super::Dataset;
use crate::dataset::fragment::write::generate_random_filename;
use crate::{Error, Result};

/// Rows per batch handed to the file writer and read back from it.
const SPILL_BATCH_ROWS: usize = 64 * 1024;

/// Encoded sequences at or below this size stay in the manifest, unless
/// [`INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY`] says otherwise.
///
/// Matches the inline limit the format has always documented for the
/// lineage oneofs. A `Range` row id sequence -- every appended fragment --
/// encodes to a few dozen bytes and is nowhere near it, and so does a
/// single-run version sequence.
pub const DEFAULT_INLINE_ROW_LINEAGE_MAX_BYTES: usize = 200 * 1024;

/// Table config key that turns spilling on: `"true"` lets compaction and
/// update move the oversized lineage sequences they carry over from existing
/// rows out of the manifest -- row ids and created-at versions, and at
/// compaction last-updated-at versions too; an update's last-updated-at
/// version is the commit's and stays inline. Absent or anything else, every
/// sequence stays inline however large it grows, which is what every released
/// build does; a table that never sets it stays readable by them.
pub const SPILL_ROW_LINEAGE_CONFIG_KEY: &str = "lance.row_lineage.spill";

/// Table config key overriding [`DEFAULT_INLINE_ROW_LINEAGE_MAX_BYTES`], as a
/// byte count. Only consulted when [`SPILL_ROW_LINEAGE_CONFIG_KEY`] is on.
pub const INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY: &str = "lance.row_lineage.inline_max_bytes";

/// The largest encoded lineage sequence `dataset` keeps inline, or `None` when
/// the table does not spill at all.
///
/// Spilling needs the table's opt-in, a build that understands the feature
/// flag (a build that does not would write a dataset it then refuses to
/// open), and v2 data files: the format allows the columns only there, since
/// a legacy v1 file has no `column_indices` to locate them by.
pub fn inline_row_lineage_max_bytes(dataset: &Dataset) -> Result<Option<usize>> {
    let config = dataset.config();
    let enabled = config
        .get(SPILL_ROW_LINEAGE_CONFIG_KEY)
        .is_some_and(|value| value.eq_ignore_ascii_case("true"));
    if !enabled
        || !lance_table::feature_flags::spilled_row_lineage_enabled()
        || dataset.manifest.data_storage_format.lance_file_format() == ConcreteFileVersion::V1
    {
        return Ok(None);
    }
    let Some(value) = config.get(INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY) else {
        return Ok(Some(DEFAULT_INLINE_ROW_LINEAGE_MAX_BYTES));
    };
    value.parse().map(Some).map_err(|error| {
        Error::invalid_input(format!(
            "table config {INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY}={value:?} is not a byte \
             count: {error}"
        ))
    })
}

/// The hidden row lineage columns by name and reserved field id, in the order
/// a data file written by compaction stores them.
const LINEAGE_COLUMNS: [(&str, i32); 3] = [
    (ROW_ID, ROW_ID_FIELD_ID),
    (ROW_CREATED_AT_VERSION, ROW_CREATED_AT_VERSION_FIELD_ID),
    (
        ROW_LAST_UPDATED_AT_VERSION,
        ROW_LAST_UPDATED_AT_VERSION_FIELD_ID,
    ),
];

/// Which of a compaction task's sequences leave the manifest as hidden columns
/// of the data files it writes. Every file the task writes carries the same
/// columns, so this holds for all of its output fragments.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RowLineageSpill {
    pub row_ids: bool,
    pub created_at: bool,
    pub last_updated_at: bool,
}

impl RowLineageSpill {
    pub fn any(&self) -> bool {
        self.row_ids || self.created_at || self.last_updated_at
    }

    /// The reserved field ids of the spilled columns, in column order.
    pub fn field_ids(&self) -> impl Iterator<Item = i32> {
        let flags = [self.row_ids, self.created_at, self.last_updated_at];
        LINEAGE_COLUMNS
            .into_iter()
            .zip(flags)
            .filter_map(|((_, field_id), spilled)| spilled.then_some(field_id))
    }

    /// Move the spilled sequences out of `lineages` as the values of the
    /// hidden columns, one vector per column in column order, concatenated
    /// over `lineages`. The spilled sequences are left empty: nothing places
    /// them inline, and their run-length form would otherwise stay alive
    /// through the write next to the values built from it.
    pub fn take_columns(&self, lineages: &mut [RowLineage]) -> Vec<Vec<u64>> {
        let total_rows = lineages
            .iter()
            .map(|lineage| lineage.row_ids.len() as usize)
            .sum::<usize>();
        self.field_ids()
            .map(|field_id| {
                let mut values = Vec::with_capacity(total_rows);
                for lineage in lineages.iter_mut() {
                    match field_id {
                        ROW_ID_FIELD_ID => {
                            let row_ids = std::mem::take(&mut lineage.row_ids);
                            values.extend(row_ids.iter());
                        }
                        ROW_CREATED_AT_VERSION_FIELD_ID => {
                            let created_at = std::mem::take(&mut lineage.created_at);
                            values.extend(created_at.versions());
                        }
                        // The only other id `field_ids` yields.
                        _ => {
                            let last_updated_at = std::mem::take(&mut lineage.last_updated_at);
                            values.extend(last_updated_at.versions());
                        }
                    }
                }
                values
            })
            .collect()
    }

    /// The spilled columns as write-schema fields, in column order and under
    /// their reserved ids.
    pub fn schema_fields(&self) -> Result<Vec<lance_core::datatypes::Field>> {
        self.field_ids().map(lineage_field).collect()
    }

    /// The placement for one fragment whose own data file was written with
    /// the spilled columns: those sequences are marked as spilled, the rest
    /// are placed inline. There is no lineage file to add to the fragment.
    pub fn place_in_file(&self, lineage: &RowLineage) -> PlacedRowLineage {
        PlacedRowLineage {
            row_ids: if self.row_ids {
                RowIdMeta::Column
            } else {
                RowIdMeta::Inline(write_row_ids(&lineage.row_ids).into())
            },
            created_at: if self.created_at {
                RowDatasetVersionMeta::Column
            } else {
                RowDatasetVersionMeta::Inline(write_dataset_versions(&lineage.created_at).into())
            },
            last_updated_at: if self.last_updated_at {
                RowDatasetVersionMeta::Column
            } else {
                RowDatasetVersionMeta::Inline(
                    write_dataset_versions(&lineage.last_updated_at).into(),
                )
            },
            file: None,
        }
    }
}

/// The hidden column a spilled lineage sequence is stored in: a non-nullable
/// `UInt64` named after the sequence in [`LINEAGE_COLUMNS`], under its
/// reserved `field_id`.
fn lineage_field(field_id: i32) -> Result<lance_core::datatypes::Field> {
    let (name, _) = LINEAGE_COLUMNS
        .into_iter()
        .find(|(_, id)| *id == field_id)
        .ok_or_else(|| {
            Error::internal(format!(
                "field id {field_id} is not a reserved row lineage field id"
            ))
        })?;
    let mut field =
        lance_core::datatypes::Field::try_from(&ArrowField::new(name, DataType::UInt64, false))?;
    field.id = field_id;
    Ok(field)
}

/// How a compaction task places the lineage of the fragments it writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowLineagePlan {
    /// Each sequence type the spill names is over the inline budget in every
    /// output fragment and is written into the data files as a hidden column;
    /// every other type is under the budget in all of them and stays inline.
    InFile(RowLineageSpill),
    /// Some sequence type is over the budget in some output fragments but not
    /// in others. A column in every file would also spill the sequences that
    /// fit inline, and their readers would then pay a read per row for what a
    /// few bytes of manifest hold. The task writes no hidden columns instead,
    /// and after the write places each fragment's lineage on its own with
    /// [`place_row_lineage`], which spills only what is over the budget, to a
    /// separate lineage file.
    PerFragment,
}

/// Plan how the lineage of a compaction task's output fragments, `lineages`,
/// is placed when every encoded sequence over `limit` bytes has to leave the
/// manifest (see [`inline_row_lineage_max_bytes`]).
pub fn plan_row_lineage_spill(limit: usize, lineages: &[RowLineage]) -> RowLineagePlan {
    let over = |encoded: Vec<u8>| encoded.len() > limit;
    let row_ids = over_in_all_or_none(lineages.iter().map(|l| over(write_row_ids(&l.row_ids))));
    let created_at = over_in_all_or_none(
        lineages
            .iter()
            .map(|l| over(write_dataset_versions(&l.created_at))),
    );
    let last_updated_at = over_in_all_or_none(
        lineages
            .iter()
            .map(|l| over(write_dataset_versions(&l.last_updated_at))),
    );
    match (row_ids, created_at, last_updated_at) {
        (Some(row_ids), Some(created_at), Some(last_updated_at)) => {
            RowLineagePlan::InFile(RowLineageSpill {
                row_ids,
                created_at,
                last_updated_at,
            })
        }
        _ => RowLineagePlan::PerFragment,
    }
}

/// Whether one sequence type is over the budget in every output fragment
/// (`Some(true)`) or in none (`Some(false)`), given each output's verdict in
/// `over`; `None` when the outputs disagree. It stops encoding at the first
/// disagreement.
fn over_in_all_or_none(mut over: impl Iterator<Item = bool>) -> Option<bool> {
    let Some(first) = over.next() else {
        return Some(false);
    };
    over.all(|next| next == first).then_some(first)
}

/// The per-row lineage of one fragment, in row offset order.
pub struct RowLineage {
    pub row_ids: RowIdSequence,
    pub created_at: RowDatasetVersionSequence,
    pub last_updated_at: RowDatasetVersionSequence,
}

/// Where each of a fragment's lineage sequences ended up, and the data file
/// the spilled ones were written to, which the fragment has to list among its
/// files. [`Self::apply`] does both.
pub struct PlacedRowLineage {
    pub row_ids: RowIdMeta,
    pub created_at: RowDatasetVersionMeta,
    pub last_updated_at: RowDatasetVersionMeta,
    pub file: Option<DataFile>,
}

impl PlacedRowLineage {
    /// Put the placement on `fragment`: its three lineage arms, and the
    /// lineage file, if any, as one more of its data files.
    pub fn apply(self, fragment: &mut Fragment) {
        fragment.row_id_meta = Some(self.row_ids);
        fragment.created_at_version_meta = Some(self.created_at);
        fragment.last_updated_at_version_meta = Some(self.last_updated_at);
        if let Some(file) = self.file {
            fragment.files.push(file);
        }
    }
}

/// Place each sequence of `lineage` either inline in the manifest or in a
/// hidden column of a new data file, as the table's spill policy (see
/// [`inline_row_lineage_max_bytes`]) and the sequence's encoded size call for.
/// Every sequence that spills goes into one file, which the caller adds to the
/// fragment's data files (see [`PlacedRowLineage::apply`]).
///
/// Only correct for lineage that a commit conflict cannot change: row ids and
/// versions carried over from existing rows. Lineage assigned at commit time --
/// an appended fragment's row ids, an inserted row's created-at version -- has
/// to stay inline, where the commit can still rewrite it.
pub async fn place_row_lineage(
    dataset: &Dataset,
    lineage: &RowLineage,
) -> Result<PlacedRowLineage> {
    let (can_spill, limit) = match inline_row_lineage_max_bytes(dataset)? {
        Some(limit) => (true, limit),
        None => (false, usize::MAX),
    };
    let inline_row_ids = write_row_ids(&lineage.row_ids);
    let inline_created_at = write_dataset_versions(&lineage.created_at);
    let inline_last_updated_at = write_dataset_versions(&lineage.last_updated_at);

    // Materialized up front rather than streamed from the sequence iterators:
    // `RowIdSequence::iter` returns a boxed `dyn DoubleEndedIterator`, which is
    // not `Send`, so holding it across the write below would make this future
    // non-`Send` and every caller of `compact_files` along with it --
    // including the Python bindings, which spawn that future.
    let mut columns: Vec<(i32, &str, ArrayRef)> = Vec::with_capacity(3);
    if can_spill && inline_row_ids.len() > limit {
        let ids = UInt64Array::from(lineage.row_ids.iter().collect::<Vec<u64>>());
        columns.push((ROW_ID_FIELD_ID, ROW_ID, Arc::new(ids)));
    }
    if can_spill && inline_created_at.len() > limit {
        let versions = UInt64Array::from(lineage.created_at.versions().collect::<Vec<u64>>());
        columns.push((
            ROW_CREATED_AT_VERSION_FIELD_ID,
            ROW_CREATED_AT_VERSION,
            Arc::new(versions),
        ));
    }
    if can_spill && inline_last_updated_at.len() > limit {
        let versions = UInt64Array::from(lineage.last_updated_at.versions().collect::<Vec<u64>>());
        columns.push((
            ROW_LAST_UPDATED_AT_VERSION_FIELD_ID,
            ROW_LAST_UPDATED_AT_VERSION,
            Arc::new(versions),
        ));
    }

    let spilled = if columns.is_empty() {
        None
    } else {
        Some(write_lineage_file(dataset, &columns).await?)
    };
    let holds = |field_id: i32| {
        spilled
            .as_ref()
            .is_some_and(|file| file.fields.contains(&field_id))
    };

    Ok(PlacedRowLineage {
        row_ids: if holds(ROW_ID_FIELD_ID) {
            RowIdMeta::Column
        } else {
            RowIdMeta::Inline(inline_row_ids.into())
        },
        created_at: if holds(ROW_CREATED_AT_VERSION_FIELD_ID) {
            RowDatasetVersionMeta::Column
        } else {
            RowDatasetVersionMeta::Inline(inline_created_at.into())
        },
        last_updated_at: if holds(ROW_LAST_UPDATED_AT_VERSION_FIELD_ID) {
            RowDatasetVersionMeta::Column
        } else {
            RowDatasetVersionMeta::Inline(inline_last_updated_at.into())
        },
        file: spilled,
    })
}

/// The lineage an update carries over into one of its new fragments, as
/// [`place_carried_row_lineage`] placed it.
///
/// `pub` rather than `pub(crate)` only because `clippy::redundant_pub_crate`
/// fires inside the private `spill` module; the crate-private re-export of
/// [`place_carried_row_lineage`] keeps it off the public API.
pub struct CarriedRowLineage {
    row_ids: RowIdMeta,
    /// `None` when neither sequence spilled: the commit then resolves the
    /// created-at versions from the inline row ids, as on a table that has not
    /// opted in. `Some`, inline or spilled, once either sequence spilled.
    created_at: Option<RowDatasetVersionMeta>,
    /// The lineage file holding the spilled sequences, `None` when nothing
    /// spilled.
    file: Option<DataFile>,
}

impl CarriedRowLineage {
    /// Put the placement on `fragment`: its row ids, its created-at versions
    /// when they were placed, and the lineage file, if any, as one more of its
    /// data files. The last-updated-at versions are left for the commit.
    pub fn apply(self, fragment: &mut Fragment) {
        fragment.row_id_meta = Some(self.row_ids);
        fragment.created_at_version_meta = self.created_at;
        fragment.last_updated_at_version_meta = None;
        if let Some(file) = self.file {
            fragment.files.push(file);
        }
    }
}

/// Place the row ids and created-at versions that rewritten rows carry over
/// into a new fragment, spilling each whose encoding exceeds `limit` into a
/// hidden column of one new data file.
///
/// When neither exceeds `limit`, only the row ids are placed, inline, and the
/// created-at versions are left for the commit to resolve from them, as it
/// does on a table that has not opted in. Once either spills the commit can no
/// longer resolve them, since it cannot read spilled row ids, so the created-at
/// versions are placed as well, spilled or inline.
///
/// The last-updated-at version is never placed: it is the commit's, and a
/// conflict retry moves it.
pub async fn place_carried_row_lineage(
    dataset: &Dataset,
    limit: usize,
    row_ids: &RowIdSequence,
    created_at: &RowDatasetVersionSequence,
) -> Result<CarriedRowLineage> {
    let inline_row_ids = write_row_ids(row_ids);
    let inline_created_at = write_dataset_versions(created_at);
    let spill_row_ids = inline_row_ids.len() > limit;
    let spill_created_at = inline_created_at.len() > limit;
    if !spill_row_ids && !spill_created_at {
        return Ok(CarriedRowLineage {
            row_ids: RowIdMeta::Inline(inline_row_ids.into()),
            created_at: None,
            file: None,
        });
    }

    // Materialized before the write, as in `place_row_lineage`, so the future
    // stays `Send`.
    let mut columns: Vec<(i32, &str, ArrayRef)> = Vec::with_capacity(2);
    if spill_row_ids {
        let ids = UInt64Array::from(row_ids.iter().collect::<Vec<u64>>());
        columns.push((ROW_ID_FIELD_ID, ROW_ID, Arc::new(ids)));
    }
    if spill_created_at {
        let versions = UInt64Array::from(created_at.versions().collect::<Vec<u64>>());
        columns.push((
            ROW_CREATED_AT_VERSION_FIELD_ID,
            ROW_CREATED_AT_VERSION,
            Arc::new(versions),
        ));
    }
    let file = write_lineage_file(dataset, &columns).await?;

    Ok(CarriedRowLineage {
        row_ids: if spill_row_ids {
            RowIdMeta::Column
        } else {
            RowIdMeta::Inline(inline_row_ids.into())
        },
        created_at: Some(if spill_created_at {
            RowDatasetVersionMeta::Column
        } else {
            RowDatasetVersionMeta::Inline(inline_created_at.into())
        }),
        file: Some(file),
    })
}

/// Write `columns` as the hidden columns of one new data file and return the
/// [`DataFile`] that locates them, listing the columns' field ids in order.
async fn write_lineage_file(
    dataset: &Dataset,
    columns: &[(i32, &str, ArrayRef)],
) -> Result<DataFile> {
    let file_version = dataset.manifest.data_storage_format.version;
    let filename = format!("{}.lance", generate_random_filename());
    let full_path = dataset.data_dir().join(filename.as_str());

    let arrow_schema = Arc::new(ArrowSchema::new(
        columns
            .iter()
            .map(|(_, name, _)| ArrowField::new(*name, DataType::UInt64, false))
            .collect::<Vec<_>>(),
    ));
    let schema = Schema::try_from(arrow_schema.as_ref())?;
    let object_writer = dataset.object_store.create(&full_path).await?;
    let mut writer = versions::create_writer(
        file_version,
        object_writer,
        schema,
        FileWriterOptions::default(),
    )?;

    let num_rows = columns[0].2.len();
    for offset in (0..num_rows).step_by(SPILL_BATCH_ROWS) {
        let len = SPILL_BATCH_ROWS.min(num_rows - offset);
        let batch = RecordBatch::try_new(
            arrow_schema.clone(),
            columns
                .iter()
                .map(|(_, _, array)| array.slice(offset, len))
                .collect(),
        )?;
        writer.write_batch(&batch).await?;
    }
    let summary = writer.finish().await?;

    Ok(DataFile::new(
        filename,
        columns.iter().map(|(field_id, _, _)| *field_id).collect(),
        (0..columns.len() as i32).collect(),
        file_version,
        std::num::NonZero::new(summary.size_bytes),
        None,
    ))
}

/// Read back `fragment`'s row id sequence from the data file that carries it.
pub async fn read_spilled_row_ids(dataset: &Dataset, fragment: &Fragment) -> Result<RowIdSequence> {
    let ids = read_spilled_column(dataset, fragment, ROW_ID_FIELD_ID)
        .boxed()
        .await?;
    Ok(RowIdSequence::from(ids.as_slice()))
}

/// Read back one of `fragment`'s version sequences from the data file that
/// carries it; `field_id` says which of the two it is.
pub async fn read_spilled_versions(
    dataset: &Dataset,
    fragment: &Fragment,
    field_id: i32,
) -> Result<RowDatasetVersionSequence> {
    let versions = read_spilled_column(dataset, fragment, field_id)
        .boxed()
        .await?;
    Ok(RowDatasetVersionSequence::from_versions(&versions))
}

/// Read the hidden `UInt64` column `field_id` of `fragment` in full: one
/// value per physical row, from the one data file that carries the id.
///
/// Callers box this future: it drives the data file reader, and inlined
/// into the row id index build (reached from `take`, and from there from index
/// builds and `optimize_indices`) it makes those futures too deep for the trait
/// solver to prove `Send`/`Sync` (E0275).
async fn read_spilled_column(
    dataset: &Dataset,
    fragment: &Fragment,
    field_id: i32,
) -> Result<Vec<u64>> {
    let data_file = fragment.row_lineage_file(field_id)?.ok_or_else(|| {
        Error::corrupt_file(
            dataset.base.clone(),
            format!(
                "fragment {} marks row lineage field {field_id} as spilled but none of its \
                 data files carries it",
                fragment.id
            ),
        )
    })?;
    let column_index = data_file
        .fields
        .iter()
        .position(|field| *field == field_id)
        .and_then(|position| data_file.column_indices.get(position))
        .ok_or_else(|| {
            Error::corrupt_file_named(
                &data_file.path,
                format!("spilled row lineage file does not carry field id {field_id}"),
            )
        })?;
    let column_index = u32::try_from(*column_index).map_err(|_| {
        Error::corrupt_file_named(
            &data_file.path,
            format!("field id {field_id} has no column index in the spilled row lineage file"),
        )
    })?;

    // The projected field is built from the reserved id rather than taken
    // from the file schema. A column index counts physical columns -- one per
    // leaf, and in 2.0 one per list or struct as well -- so once the lineage
    // columns follow nested user columns, as in a compaction output, it is no
    // longer the field's position among the file's top-level fields. Nor can
    // the file schema be searched by id: a lineage-only file stores its
    // fields under ids 0, 1 and 2.
    let projection = ReaderProjection {
        schema: Arc::new(Schema {
            fields: vec![lineage_field(field_id)?],
            metadata: Default::default(),
        }),
        column_indices: vec![column_index],
    };

    // Resolved through `data_file_dir` rather than `data_dir` so a shallow
    // clone, which rewrites `base_id` on every referenced file, still finds it.
    let path: Path = dataset
        .data_file_dir(data_file)?
        .join(data_file.path.as_str());
    let object_store = dataset.object_store_for_data_file(data_file).await?;
    let scheduler = ScanScheduler::new(
        object_store.clone(),
        SchedulerConfig::max_bandwidth(&object_store),
    );
    let file = scheduler
        .open_file(&path, &data_file.file_size_bytes)
        .await?;
    let options = dataset.file_reader_options.clone().unwrap_or_default();
    let cache = dataset.metadata_cache.file_metadata_cache(&path);
    let io =
        Arc::new(LanceEncodingsIo::new(file.clone()).with_read_chunk_size(options.read_chunk_size));
    // An index past the file's columns means the data file entry and the file
    // disagree. The reader would reject the projection as invalid input without
    // naming the file, so it is checked here, once the column count is known.
    let check_column_index = |num_columns: usize| {
        if (column_index as usize) < num_columns {
            Ok(())
        } else {
            Err(Error::corrupt_file_named(
                &data_file.path,
                format!(
                    "spilled row lineage column {field_id} is at column index {column_index}, \
                     but the file has only {num_columns} columns"
                ),
            ))
        }
    };
    // A compaction output holds the lineage next to every user column, and
    // decoding all their metadata to read one column would cost as much as
    // opening the file for a scan. Past a few columns, only the one read here
    // has its metadata fetched, by the same threshold the fragment reader uses.
    let live_columns = data_file
        .column_indices
        .iter()
        .filter(|column_index| **column_index >= 0)
        .count();
    let reader = versions::open_projected_reader(
        data_file.file_version()?,
        &projection,
        projection.column_indices.len().saturating_mul(4) < live_columns,
        || async {
            let metadata_index = FileReader::read_metadata_index(&file).await?;
            check_column_index(metadata_index.num_columns() as usize)?;
            let reader = ProjectedFileReader::try_open_with_metadata_index(
                io.clone(),
                path.clone(),
                Some(projection.clone()),
                Arc::<DecoderPlugins>::default(),
                Arc::new(metadata_index),
                &cache,
                options.clone(),
            )
            .await?;
            Ok(Some(reader))
        },
        || async {
            let metadata = FileReader::read_all_metadata(&file).await?;
            check_column_index(metadata.column_infos.len())?;
            ProjectedFileReader::try_open_with_file_metadata(
                io.clone(),
                path.clone(),
                Some(projection.clone()),
                Arc::<DecoderPlugins>::default(),
                Arc::new(metadata),
                &cache,
                options.clone(),
            )
            .await
        },
    )
    .await?;

    let mut values: Vec<u64> = Vec::with_capacity(reader.num_rows() as usize);
    let mut batches = reader
        .read_tasks(
            ReadBatchParams::RangeFull,
            SPILL_BATCH_ROWS as u32,
            None,
            FilterExpression::no_filter(),
        )
        .await?
        .map(|task| task.task)
        .buffered(8);
    while let Some(batch) = batches.try_next().await? {
        let column = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                Error::corrupt_file_named(
                    &data_file.path,
                    format!("spilled row lineage column {field_id} is not UInt64"),
                )
            })?;
        // A null has no row id or version to stand for; the format requires a
        // value for every physical row.
        if column.null_count() > 0 {
            return Err(Error::corrupt_file_named(
                &data_file.path,
                format!("spilled row lineage column {field_id} holds nulls"),
            ));
        }
        values.extend_from_slice(column.values());
    }
    if let Some(physical_rows) = fragment.physical_rows
        && values.len() != physical_rows
    {
        return Err(Error::corrupt_file_named(
            &data_file.path,
            format!(
                "spilled row lineage column {field_id} holds {} values for a fragment of {} \
                 physical rows",
                values.len(),
                physical_rows
            ),
        ));
    }

    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::cleanup::{CleanupPolicyBuilder, cleanup_old_versions};
    use crate::dataset::fragment::FileFragment;
    use crate::dataset::optimize::{
        CompactionMode, CompactionOptions, compact_files, plan_compaction,
    };
    use crate::dataset::rowids::{RowVersionKind, load_row_id_sequence, load_row_version_sequence};
    use crate::dataset::transaction::Operation;
    use crate::dataset::{
        ColumnAlteration, NewColumnTransform, UpdateBuilder, WriteMode, WriteParams,
    };
    use arrow_array::builder::{ListBuilder, StringBuilder};
    use arrow_array::{Int32Array, RecordBatchIterator, StringArray, StructArray};
    use arrow_schema::Field;
    use chrono::Utc;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_core::{ROW_CREATED_AT_VERSION, ROW_ID, ROW_LAST_UPDATED_AT_VERSION};
    use lance_file::version::LanceFileVersion;
    use lance_table::feature_flags::FLAG_UNSTABLE_SPILLED_ROW_LINEAGE;
    use lance_table::format::overlay::TOMBSTONE_FIELD_ID;
    use rstest::rstest;

    /// A sequence with no runs to exploit, which is what a globally shuffled
    /// table produces and what forces the spill path.
    fn scattered_row_ids(len: u64) -> RowIdSequence {
        // A stride coprime with `len` visits every id exactly once in an order
        // with no ascending run longer than one.
        let ids: Vec<u64> = (0..len).map(|i| (i * 7919) % len).collect();
        RowIdSequence::from(ids.as_slice())
    }

    /// A version per row that alternates, so every row is its own run.
    fn alternating_versions(len: u64, first: u64) -> RowDatasetVersionSequence {
        let versions: Vec<u64> = (0..len).map(|i| first + i % 2).collect();
        RowDatasetVersionSequence::from_versions(&versions)
    }

    fn test_schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![Field::new(
            "i",
            DataType::Int32,
            false,
        )]))
    }

    async fn tiny_dataset(uri: &str) -> Dataset {
        tiny_dataset_with_version(uri, LanceFileVersion::default()).await
    }

    async fn tiny_dataset_with_version(uri: &str, version: LanceFileVersion) -> Dataset {
        let schema = test_schema();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4]))],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        Dataset::write(
            reader,
            uri,
            Some(WriteParams {
                enable_stable_row_ids: true,
                data_storage_version: Some(version),
                ..Default::default()
            }),
        )
        .await
        .unwrap()
    }

    fn versions_of(sequence: &RowDatasetVersionSequence) -> Vec<u64> {
        sequence.versions().collect()
    }

    #[tokio::test]
    async fn spilled_lineage_shares_one_file_and_round_trips() {
        let dir = TempStrDir::default();
        let mut dataset = tiny_dataset(dir.as_str()).await;
        spill_everything(&mut dataset).await;

        let lineage = RowLineage {
            row_ids: scattered_row_ids(20_000),
            created_at: alternating_versions(20_000, 1),
            last_updated_at: alternating_versions(20_000, 3),
        };
        let placed = place_row_lineage(&dataset, &lineage).await.unwrap();
        assert!(
            matches!(placed.row_ids, RowIdMeta::Column)
                && matches!(placed.created_at, RowDatasetVersionMeta::Column)
                && matches!(placed.last_updated_at, RowDatasetVersionMeta::Column),
            "expected every sequence to spill"
        );
        // One file carries all three, and it becomes one of the fragment's
        // data files; the metadata only marks the sequences as spilled.
        let mut fragment = Fragment::new(0);
        fragment.physical_rows = Some(20_000);
        placed.apply(&mut fragment);
        assert_eq!(fragment.files.len(), 1);
        assert_eq!(
            fragment.files[0].fields.as_ref(),
            [
                ROW_ID_FIELD_ID,
                ROW_CREATED_AT_VERSION_FIELD_ID,
                ROW_LAST_UPDATED_AT_VERSION_FIELD_ID
            ]
        );

        let row_ids = read_spilled_row_ids(&dataset, &fragment).await.unwrap();
        assert_eq!(
            row_ids.iter().collect::<Vec<_>>(),
            lineage.row_ids.iter().collect::<Vec<_>>()
        );
        let created_at =
            read_spilled_versions(&dataset, &fragment, ROW_CREATED_AT_VERSION_FIELD_ID)
                .await
                .unwrap();
        assert_eq!(versions_of(&created_at), versions_of(&lineage.created_at));
        let last_updated_at =
            read_spilled_versions(&dataset, &fragment, ROW_LAST_UPDATED_AT_VERSION_FIELD_ID)
                .await
                .unwrap();
        assert_eq!(
            versions_of(&last_updated_at),
            versions_of(&lineage.last_updated_at)
        );

        // A data file entry that points past the file's columns is corruption
        // of that file, not a bad projection from the caller.
        fragment.files[0].column_indices = vec![3, 1, 2].into();
        let error = read_spilled_row_ids(&dataset, &fragment).await.unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains("is at column index 3, but the file has only 3 columns"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn only_the_sequences_over_the_limit_spill() {
        let dir = TempStrDir::default();
        let mut dataset = tiny_dataset(dir.as_str()).await;
        // An appended fragment's row ids are a single `Range` and its versions
        // a single run, so they encode to a few dozen bytes and must never
        // leave the manifest, even next to a sequence that does.
        let lineage = RowLineage {
            row_ids: RowIdSequence::from(0..20_000),
            created_at: alternating_versions(20_000, 1),
            last_updated_at: RowDatasetVersionSequence::from_uniform_row_count(20_000, 1),
        };

        // A table that has not opted in never spills, whatever the size.
        let placed = place_row_lineage(&dataset, &lineage).await.unwrap();
        assert!(matches!(
            placed.created_at,
            RowDatasetVersionMeta::Inline(_)
        ));
        assert!(placed.file.is_none());

        dataset
            .update_config([(SPILL_ROW_LINEAGE_CONFIG_KEY, "true")])
            .await
            .unwrap();
        let placed = place_row_lineage(&dataset, &lineage).await.unwrap();
        assert!(
            matches!(placed.row_ids, RowIdMeta::Inline(_)),
            "a range sequence must stay inline, got {:?}",
            placed.row_ids
        );
        assert!(
            matches!(placed.last_updated_at, RowDatasetVersionMeta::Inline(_)),
            "a single-run sequence must stay inline, got {:?}",
            placed.last_updated_at
        );
        assert!(
            matches!(placed.created_at, RowDatasetVersionMeta::Column),
            "an alternating version sequence encodes past 200 KiB"
        );
        assert_eq!(
            placed.file.as_ref().unwrap().fields.as_ref(),
            [ROW_CREATED_AT_VERSION_FIELD_ID]
        );
    }

    /// What an update carries over is placed in full only once something
    /// spills: the commit resolves created-at versions from inline row ids but
    /// cannot from spilled ones, and it always stamps last-updated-at itself.
    #[rstest]
    #[case::nothing_spills(false, false)]
    #[case::row_ids_spill(true, false)]
    #[case::created_at_spills(false, true)]
    #[case::both_spill(true, true)]
    #[tokio::test]
    async fn carried_lineage_places_created_at_once_anything_spills(
        #[case] spill_row_ids: bool,
        #[case] spill_created_at: bool,
    ) {
        let dir = TempStrDir::default();
        let dataset = tiny_dataset(dir.as_str()).await;
        // Under a 100-byte budget a scattered or alternating sequence of 1,000
        // rows spills, and a range or a single run stays inline.
        let row_ids = if spill_row_ids {
            scattered_row_ids(1_000)
        } else {
            RowIdSequence::from(0..1_000)
        };
        let created_at = if spill_created_at {
            alternating_versions(1_000, 1)
        } else {
            RowDatasetVersionSequence::from_uniform_row_count(1_000, 1)
        };

        let mut fragment = Fragment::new(42);
        fragment.physical_rows = Some(1_000);
        place_carried_row_lineage(&dataset, 100, &row_ids, &created_at)
            .await
            .unwrap()
            .apply(&mut fragment);

        assert_eq!(
            matches!(fragment.row_id_meta, Some(RowIdMeta::Column)),
            spill_row_ids
        );
        assert_eq!(fragment.last_updated_at_version_meta, None);
        let placed_row_ids = load_row_id_sequence(&dataset, &fragment).await.unwrap();
        assert_eq!(
            placed_row_ids.iter().collect::<Vec<_>>(),
            row_ids.iter().collect::<Vec<_>>()
        );
        if !spill_row_ids && !spill_created_at {
            assert_eq!(fragment.created_at_version_meta, None);
            assert!(fragment.files.is_empty());
        } else {
            let spilled = [
                (spill_row_ids, ROW_ID_FIELD_ID),
                (spill_created_at, ROW_CREATED_AT_VERSION_FIELD_ID),
            ]
            .into_iter()
            .filter_map(|(spills, field_id)| spills.then_some(field_id))
            .collect::<Vec<_>>();
            assert_eq!(fragment.files.len(), 1);
            assert_eq!(fragment.files[0].fields.as_ref(), spilled.as_slice());
            assert_eq!(
                matches!(
                    fragment.created_at_version_meta,
                    Some(RowDatasetVersionMeta::Column)
                ),
                spill_created_at
            );
            let placed_created_at =
                load_row_version_sequence(&dataset, &fragment, RowVersionKind::CreatedAt)
                    .await
                    .unwrap()
                    .expect("created-at versions are placed once anything spills");
            assert_eq!(versions_of(&placed_created_at), versions_of(&created_at));
        }
    }

    /// The format allows the columns only in v2 files, so a legacy v1 table
    /// keeps everything inline however it is configured.
    #[tokio::test]
    async fn a_legacy_v1_table_never_spills() {
        let dir = TempStrDir::default();
        let mut dataset = tiny_dataset_with_version(dir.as_str(), LanceFileVersion::Legacy).await;
        spill_everything(&mut dataset).await;
        assert_eq!(inline_row_lineage_max_bytes(&dataset).unwrap(), None);

        let lineage = RowLineage {
            row_ids: RowIdSequence::from(0..20_000),
            created_at: alternating_versions(20_000, 1),
            last_updated_at: RowDatasetVersionSequence::from_uniform_row_count(20_000, 1),
        };
        let placed = place_row_lineage(&dataset, &lineage).await.unwrap();
        assert!(matches!(placed.row_ids, RowIdMeta::Inline(_)));
        assert!(matches!(
            placed.created_at,
            RowDatasetVersionMeta::Inline(_)
        ));
        assert!(matches!(
            placed.last_updated_at,
            RowDatasetVersionMeta::Inline(_)
        ));
        assert!(placed.file.is_none());
    }

    /// Compaction writes a sequence type into its data files only when it is
    /// over the budget in every output fragment. A type over it in only some
    /// of them sends the task down the per-fragment path, so the sequences
    /// that fit stay inline.
    #[rstest]
    #[case::all_over(
        vec![scattered_row_ids(500), scattered_row_ids(500)],
        RowLineagePlan::InFile(RowLineageSpill { row_ids: true, ..Default::default() })
    )]
    #[case::none_over(
        vec![RowIdSequence::from(0..500), RowIdSequence::from(500..1000)],
        RowLineagePlan::InFile(RowLineageSpill::default())
    )]
    #[case::mixed(
        vec![RowIdSequence::from(0..500), scattered_row_ids(500)],
        RowLineagePlan::PerFragment
    )]
    fn plan_row_lineage_spill_decides_per_kind(
        #[case] row_ids: Vec<RowIdSequence>,
        #[case] expected: RowLineagePlan,
    ) {
        // Single-run version sequences encode to a few bytes, far under the
        // budget, so the row ids alone decide.
        let lineages = row_ids
            .into_iter()
            .map(|row_ids| {
                let created_at =
                    RowDatasetVersionSequence::from_uniform_row_count(row_ids.len(), 1);
                RowLineage {
                    row_ids,
                    last_updated_at: created_at.clone(),
                    created_at,
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(plan_row_lineage_spill(100, &lineages), expected);
    }

    /// Opt the table into spilling, at a zero inline budget so every sequence
    /// spills regardless of size: reaching the natural 200 KiB threshold needs
    /// ~25k scattered rows, more than these tests need to prove.
    async fn spill_everything(dataset: &mut Dataset) {
        dataset
            .update_config([
                (SPILL_ROW_LINEAGE_CONFIG_KEY, "true"),
                (INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, "0"),
            ])
            .await
            .unwrap();
    }

    /// The user columns of a test table: its key column `i`, and possibly one
    /// more column next to it.
    #[derive(Clone, Copy, Debug)]
    enum UserColumns {
        KeyOnly,
        /// Adds `meta: struct<a: int32, b: utf8>`, which from 2.1 on takes one
        /// physical column per child and none for the struct itself.
        WithStruct,
        /// Adds `tags: list<utf8>`, which in 2.0 takes one physical column for
        /// the list and one for its items.
        WithList,
        /// Adds `j: int32` holding the key again, so a schema change has a
        /// column to rename, drop or cast while `i` still identifies every
        /// row.
        WithCopy,
    }

    /// A batch of `columns` holding `keys` in column `i`.
    fn keyed_batch(columns: UserColumns, keys: std::ops::Range<i32>) -> RecordBatch {
        let mut fields = vec![Field::new("i", DataType::Int32, false)];
        let mut arrays: Vec<ArrayRef> = vec![Arc::new(Int32Array::from_iter_values(keys.clone()))];
        match columns {
            UserColumns::KeyOnly => {}
            UserColumns::WithStruct => {
                let a: ArrayRef = Arc::new(Int32Array::from_iter_values(keys.clone()));
                let b: ArrayRef = Arc::new(StringArray::from_iter_values(
                    keys.map(|key| format!("b{key}")),
                ));
                let meta = StructArray::from(vec![
                    (Arc::new(Field::new("a", DataType::Int32, true)), a),
                    (Arc::new(Field::new("b", DataType::Utf8, true)), b),
                ]);
                fields.push(Field::new("meta", meta.data_type().clone(), true));
                arrays.push(Arc::new(meta));
            }
            UserColumns::WithList => {
                let mut tags = ListBuilder::new(StringBuilder::new());
                for key in keys {
                    tags.values().append_value(format!("t{key}"));
                    tags.append(true);
                }
                let tags = tags.finish();
                fields.push(Field::new("tags", tags.data_type().clone(), true));
                arrays.push(Arc::new(tags));
            }
            UserColumns::WithCopy => {
                fields.push(Field::new("j", DataType::Int32, true));
                arrays.push(Arc::new(Int32Array::from_iter_values(keys)));
            }
        }
        RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), arrays).unwrap()
    }

    /// A stable-row-id dataset built from `chunks` separate appends, so
    /// compacting it has several sequences to concatenate.
    async fn appended_dataset(uri: &str, chunks: i32, rows_per_chunk: i32) -> Dataset {
        appended_dataset_with(uri, chunks, rows_per_chunk, UserColumns::KeyOnly, None).await
    }

    /// [`appended_dataset`] with the given user columns, written in `version`,
    /// or the default version when that is `None`.
    async fn appended_dataset_with(
        uri: &str,
        chunks: i32,
        rows_per_chunk: i32,
        columns: UserColumns,
        version: Option<LanceFileVersion>,
    ) -> Dataset {
        let mut dataset: Option<Dataset> = None;
        for chunk in 0..chunks {
            let keys = (chunk * rows_per_chunk)..((chunk + 1) * rows_per_chunk);
            let batch = keyed_batch(columns, keys);
            let schema = batch.schema();
            let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
            dataset = Some(
                Dataset::write(
                    reader,
                    uri,
                    Some(WriteParams {
                        enable_stable_row_ids: true,
                        data_storage_version: version,
                        mode: if chunk == 0 {
                            WriteMode::Create
                        } else {
                            WriteMode::Append
                        },
                        ..Default::default()
                    }),
                )
                .await
                .unwrap(),
            );
        }
        dataset.unwrap()
    }

    fn one_fragment() -> CompactionOptions {
        CompactionOptions {
            target_rows_per_fragment: 1_000,
            ..Default::default()
        }
    }

    /// The lineage columns of every row, in scan order.
    async fn collect_lineage(dataset: &Dataset) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        let mut scanner = dataset.scan();
        scanner
            .project(&[ROW_ID, ROW_CREATED_AT_VERSION, ROW_LAST_UPDATED_AT_VERSION])
            .unwrap();
        let batch = scanner.try_into_batch().await.unwrap();
        let column = |name: &str| {
            batch
                .column_by_name(name)
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .values()
                .to_vec()
        };
        (
            column(ROW_ID),
            column(ROW_CREATED_AT_VERSION),
            column(ROW_LAST_UPDATED_AT_VERSION),
        )
    }

    /// Compaction writes the spilled columns after the user columns of its
    /// output file. A column index counts physical columns, so after a
    /// struct, or a list in 2.0, a lineage column's index no longer matches
    /// its position among the file's top-level fields.
    #[rstest]
    #[case::flat(UserColumns::KeyOnly, None)]
    #[case::nested_v2_2(UserColumns::WithStruct, Some(LanceFileVersion::V2_2))]
    #[case::list_v2_0(UserColumns::WithList, Some(LanceFileVersion::V2_0))]
    #[tokio::test]
    async fn compaction_spills_and_reads_back_row_lineage(
        #[case] columns: UserColumns,
        #[case] version: Option<LanceFileVersion>,
    ) {
        let dir = TempStrDir::default();
        let uri = dir.as_str();
        let mut dataset = appended_dataset_with(uri, 4, 250, columns, version).await;
        spill_everything(&mut dataset).await;
        // Four appends at four versions, so the compacted created-at sequence
        // has four runs rather than one.
        let before = collect_lineage(&dataset).await;
        assert_eq!(
            before
                .1
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            4
        );

        compact_files(&mut dataset, one_fragment(), None)
            .await
            .unwrap();

        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        let metadata = fragments[0].metadata();
        assert!(
            matches!(metadata.row_id_meta, Some(RowIdMeta::Column))
                && matches!(
                    metadata.created_at_version_meta,
                    Some(RowDatasetVersionMeta::Column)
                )
                && matches!(
                    metadata.last_updated_at_version_meta,
                    Some(RowDatasetVersionMeta::Column)
                ),
            "compaction must spill every sequence under a zero inline budget, got {metadata:?}"
        );
        // The three columns ride in the fragment's own data file, after the
        // user columns, so the fragment has no extra file to reference. Their
        // column indices continue from the last physical user column.
        assert_eq!(metadata.files.len(), 1);
        let data_file = &metadata.files[0];
        let (user_fields, lineage_fields) = data_file.fields.split_at(data_file.fields.len() - 3);
        assert!(
            user_fields.iter().all(|field| *field >= 0),
            "unexpected data file fields {:?}",
            data_file.fields
        );
        assert_eq!(
            lineage_fields,
            [
                ROW_ID_FIELD_ID,
                ROW_CREATED_AT_VERSION_FIELD_ID,
                ROW_LAST_UPDATED_AT_VERSION_FIELD_ID
            ]
        );
        let (user_columns, lineage_columns) = data_file.column_indices.split_at(user_fields.len());
        let next = user_columns.iter().filter(|column| **column >= 0).count() as i32;
        assert_eq!(lineage_columns, [next, next + 1, next + 2]);
        for field_id in data_file.fields.iter().filter(|id| **id < 0) {
            assert_eq!(
                metadata.row_lineage_file(*field_id).unwrap(),
                Some(data_file)
            );
        }
        assert_ne!(
            dataset.manifest.reader_feature_flags & FLAG_UNSTABLE_SPILLED_ROW_LINEAGE,
            0,
            "a spilled sequence must set the reader feature flag"
        );
        assert_ne!(
            dataset.manifest.writer_feature_flags & FLAG_UNSTABLE_SPILLED_ROW_LINEAGE,
            0,
            "a spilled sequence must set the writer feature flag"
        );

        // The lineage survives the rewrite and is still served through the
        // ordinary scan path, now from the data file columns.
        assert_eq!(collect_lineage(&dataset).await, before);
        // `validate_stable_row_ids` reads every fragment's sequences back and
        // checks them against the fragment length, so this covers the loaders
        // independently of the scan.
        dataset.validate().await.unwrap();

        // Re-opened cold, so nothing is served from this process's caches.
        let reopened = Dataset::open(uri).await.unwrap();
        assert_eq!(collect_lineage(&reopened).await, before);
        let fragment = &reopened.get_fragments()[0];
        let row_ids = load_row_id_sequence(&reopened, fragment.metadata())
            .await
            .unwrap();
        assert_eq!(row_ids.iter().collect::<Vec<_>>(), before.0);
        let created_at =
            load_row_version_sequence(&reopened, fragment.metadata(), RowVersionKind::CreatedAt)
                .await
                .unwrap()
                .expect("a compacted fragment carries created-at versions");
        assert_eq!(versions_of(&created_at), before.1);
    }

    /// Binary-copy compaction copies the input files page by page and cannot
    /// add columns to them, so its spilled lineage goes to a separate file.
    #[tokio::test]
    async fn binary_copy_compaction_spills_to_a_separate_file() {
        let dir = TempStrDir::default();
        let uri = dir.as_str();
        let mut dataset = appended_dataset(uri, 4, 250).await;
        spill_everything(&mut dataset).await;
        let before = collect_lineage(&dataset).await;

        compact_files(
            &mut dataset,
            CompactionOptions {
                compaction_mode: Some(CompactionMode::ForceBinaryCopy),
                ..one_fragment()
            },
            None,
        )
        .await
        .unwrap();

        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        let metadata = fragments[0].metadata();
        assert!(
            matches!(metadata.row_id_meta, Some(RowIdMeta::Column)),
            "compaction must spill the row ids under a zero inline budget, got {metadata:?}"
        );
        // The copied data file carries only the user column; the lineage
        // follows it as a file of its own.
        assert_eq!(metadata.files.len(), 2);
        assert!(metadata.files[0].fields.iter().all(|field| *field >= 0));
        assert_eq!(
            metadata.files[1].fields.as_ref(),
            [
                ROW_ID_FIELD_ID,
                ROW_CREATED_AT_VERSION_FIELD_ID,
                ROW_LAST_UPDATED_AT_VERSION_FIELD_ID
            ]
        );
        assert_eq!(collect_lineage(&dataset).await, before);
        dataset.validate().await.unwrap();

        // The lineage-only file holds no dead user column, so it is no reason
        // to compact the fragment again.
        let plan = plan_compaction(&dataset, &one_fragment()).await.unwrap();
        assert_eq!(plan.num_tasks(), 0);
    }

    /// Cleanup decides what to delete by walking
    /// [`Fragment::referenced_lance_files`], so a spilled sequence has to be
    /// reachable from there. If it were not, an ordinary cleanup would delete a
    /// live file and leave the fragment claiming row ids it can no longer read.
    #[tokio::test]
    async fn cleanup_keeps_a_live_spilled_file() {
        let dir = TempStrDir::default();
        let uri = dir.as_str();
        let mut dataset = appended_dataset(uri, 4, 250).await;
        spill_everything(&mut dataset).await;
        let before = collect_lineage(&dataset).await;

        compact_files(&mut dataset, one_fragment(), None)
            .await
            .unwrap();

        let spilled = dataset.get_fragments()[0]
            .metadata()
            .row_lineage_file(ROW_ID_FIELD_ID)
            .unwrap()
            .expect("compaction must spill under a zero inline budget")
            .path
            .clone();
        let on_disk = std::path::Path::new(uri).join("data").join(&spilled);
        assert!(on_disk.exists(), "no spilled file written at {on_disk:?}");

        // Everything written so far is older than this instant, so the
        // pre-compaction versions and their data files are all candidates.
        let removed = cleanup_old_versions(
            &dataset,
            CleanupPolicyBuilder::default()
                .before_timestamp(Utc::now())
                .delete_unverified(true)
                .build(),
        )
        .await
        .unwrap();
        assert!(
            removed.old_versions > 0,
            "expected the pre-compaction versions to be cleaned up"
        );
        assert!(
            on_disk.exists(),
            "cleanup deleted the live spilled row lineage file at {on_disk:?}"
        );

        let reopened = Dataset::open(uri).await.unwrap();
        assert_eq!(collect_lineage(&reopened).await, before);
    }
    /// Every row's key, row id, created-at and last-updated-at version.
    async fn collect_rows(dataset: &Dataset) -> Vec<(i32, u64, u64, u64)> {
        let mut scanner = dataset.scan();
        scanner
            .project(&[
                "i",
                ROW_ID,
                ROW_CREATED_AT_VERSION,
                ROW_LAST_UPDATED_AT_VERSION,
            ])
            .unwrap();
        let batch = scanner.try_into_batch().await.unwrap();
        let u64s = |name: &str| {
            batch
                .column_by_name(name)
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .values()
                .to_vec()
        };
        let keys = batch
            .column_by_name("i")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .to_vec();
        let (ids, created, updated) = (
            u64s(ROW_ID),
            u64s(ROW_CREATED_AT_VERSION),
            u64s(ROW_LAST_UPDATED_AT_VERSION),
        );
        keys.into_iter()
            .zip(ids)
            .zip(created)
            .zip(updated)
            .map(|(((key, id), created), updated)| (key, id, created, updated))
            .collect()
    }

    fn by_key(rows: &[(i32, u64, u64, u64)]) -> std::collections::BTreeMap<i32, (u64, u64, u64)> {
        rows.iter()
            .map(|(key, id, created, updated)| (*key, (*id, *created, *updated)))
            .collect()
    }

    /// Updating rows whose lineage is spilled keeps every row's created-at
    /// version. The update's scan reads each rewritten row's created-at
    /// version, spilled column included. When what the update carries over
    /// spills again, the writer places those versions and the commit only
    /// stamps last-updated-at; when it fits inline, the commit resolves them
    /// from the row ids, reading the source fragment's spilled sequences ahead
    /// of the build. Deleted rows and a selection across the compacted
    /// fragment's created-at runs mean a version read at the wrong offset
    /// would show up as a neighbour's.
    #[rstest]
    #[case::writer_spills(true)]
    #[case::commit_resolves(false)]
    #[tokio::test]
    async fn updating_rows_with_spilled_lineage_keeps_their_created_at(
        #[case] update_spills: bool,
    ) {
        let dir = TempStrDir::default();
        let uri = dir.as_str();
        let mut dataset = appended_dataset(uri, 4, 250).await;
        spill_everything(&mut dataset).await;
        compact_files(&mut dataset, one_fragment(), None)
            .await
            .unwrap();
        // Deleted rows move the compacted fragment's scan positions away from
        // its physical offsets, which its lineage is indexed by.
        dataset.delete("i % 7 = 0").await.unwrap();
        if !update_spills {
            // A budget the update's carried-over lineage fits in, which leaves
            // resolving the created-at versions to the commit.
            dataset
                .update_config([(INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, "1000000")])
                .await
                .unwrap();
        }
        let before = by_key(&collect_rows(&dataset).await);
        let selected = |key: i32| (240..260).contains(&key) || (745..755).contains(&key);
        // The selection spans the run boundaries at 250, between the first
        // and second appends, and at 750, between the third and fourth.
        assert_eq!(
            before
                .iter()
                .filter(|(key, _)| selected(**key))
                .map(|(_, (_, created, _))| *created)
                .collect::<std::collections::BTreeSet<_>>(),
            std::collections::BTreeSet::from([1, 2, 3, 4])
        );

        let updated = UpdateBuilder::new(Arc::new(dataset))
            .update_where("(i >= 240 AND i < 260) OR (i >= 745 AND i < 755)")
            .unwrap()
            .set("i", "i + 10000")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        let updated = updated.new_dataset.as_ref();
        let update_version = updated.version().version;
        let rewritten = updated
            .manifest
            .fragments
            .last()
            .expect("the update adds a fragment");
        assert_eq!(
            matches!(rewritten.row_id_meta, Some(RowIdMeta::Column)),
            update_spills,
            "{rewritten:?}"
        );
        assert_eq!(
            matches!(
                rewritten.created_at_version_meta,
                Some(RowDatasetVersionMeta::Column)
            ),
            update_spills,
            "{rewritten:?}"
        );

        // Each rewritten row keeps its id and its created-at, and is stamped
        // with the update's version; every other row is untouched.
        let after = by_key(&collect_rows(updated).await);
        assert_eq!(after.len(), before.len());
        for (key, (id, created, updated_at)) in before.iter() {
            if selected(*key) {
                assert_eq!(
                    after[&(key + 10000)],
                    (*id, *created, update_version),
                    "row {key}"
                );
            } else {
                assert_eq!(after[key], (*id, *created, *updated_at), "row {key}");
            }
        }
        updated.validate().await.unwrap();
    }

    /// merge_insert rewrites the matched rows and appends the inserted ones in
    /// one fragment; the former keep their lineage, the latter start at the
    /// commit version. Both resolve through the read-ahead spilled sequences.
    #[tokio::test]
    async fn merge_insert_on_spilled_table_keeps_matched_lineage() {
        use crate::dataset::{MergeInsertBuilder, WhenMatched, WhenNotMatched};

        let dir = TempStrDir::default();
        let uri = dir.as_str();
        let mut dataset = appended_dataset(uri, 4, 250).await;
        spill_everything(&mut dataset).await;
        compact_files(&mut dataset, one_fragment(), None)
            .await
            .unwrap();
        let before = by_key(&collect_rows(&dataset).await);

        // Keys 300 and 700 exist and were appended at different versions;
        // 5000 does not exist and is inserted.
        let schema = test_schema();
        let source = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![300, 700, 5000]))],
        )
        .unwrap();
        let (merged, stats) = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["i".into()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .try_build()
            .unwrap()
            .execute_reader(Box::new(RecordBatchIterator::new([Ok(source)], schema)))
            .await
            .unwrap();
        assert_eq!((stats.num_updated_rows, stats.num_inserted_rows), (2, 1));
        let merge_version = merged.version().version;
        let after = by_key(&collect_rows(&merged).await);

        for key in [300, 700] {
            let (id, created, _) = before[&key];
            assert_eq!(after[&key], (id, created, merge_version), "row {key}");
        }
        let (_, created, updated) = after[&5000];
        assert_eq!((created, updated), (merge_version, merge_version));
        for (key, lineage) in before.iter().filter(|(key, _)| ![300, 700].contains(key)) {
            assert_eq!(after[key], *lineage, "row {key} must be untouched");
        }
        merged.validate().await.unwrap();
    }

    /// A partial column rewrite patches an existing fragment in place and
    /// stamps only the patched rows' last-updated-at, which means overlaying
    /// the fragment's existing sequence; on a spilled fragment that sequence
    /// is read ahead of the commit and the refreshed one goes back inline.
    #[tokio::test]
    async fn partial_column_rewrite_on_spilled_fragment_stamps_only_matched_rows() {
        use crate::dataset::{
            MergeInsertBuilder, MergeInsertWriteMode, WhenMatched, WhenNotMatched,
        };

        let dir = TempStrDir::default();
        let uri = dir.as_str();
        // A third column keeps the patch source a strict subset of the schema,
        // which is what makes this an in-place column rewrite.
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("i", DataType::Int32, false),
            Field::new("tag", DataType::Utf8, true),
            Field::new("other", DataType::Utf8, true),
        ]));
        let mut dataset: Option<Dataset> = None;
        for chunk in 0..4 {
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(
                        (chunk * 250)..((chunk + 1) * 250),
                    )),
                    Arc::new(StringArray::from(vec!["t"; 250])),
                    Arc::new(StringArray::from(vec!["o"; 250])),
                ],
            )
            .unwrap();
            let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
            dataset = Some(
                Dataset::write(
                    reader,
                    uri,
                    Some(WriteParams {
                        enable_stable_row_ids: true,
                        mode: if chunk == 0 {
                            WriteMode::Create
                        } else {
                            WriteMode::Append
                        },
                        ..Default::default()
                    }),
                )
                .await
                .unwrap(),
            );
        }
        let mut dataset = dataset.unwrap();
        spill_everything(&mut dataset).await;
        compact_files(&mut dataset, one_fragment(), None)
            .await
            .unwrap();
        assert!(
            matches!(
                dataset.get_fragments()[0]
                    .metadata()
                    .last_updated_at_version_meta,
                Some(RowDatasetVersionMeta::Column)
            ),
            "the fixture must start with spilled last-updated-at versions"
        );
        let before = by_key(&collect_rows(&dataset).await);

        let source_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("i", DataType::Int32, false),
            Field::new("tag", DataType::Utf8, true),
        ]));
        let source = RecordBatch::try_new(
            source_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![300, 700])),
                Arc::new(StringArray::from(vec!["patched"; 2])),
            ],
        )
        .unwrap();
        let (patched, stats) = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["i".into()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing)
            .write_mode(MergeInsertWriteMode::RewriteColumns)
            .try_build()
            .unwrap()
            .execute_reader(Box::new(RecordBatchIterator::new(
                [Ok(source)],
                source_schema,
            )))
            .await
            .unwrap();
        assert_eq!(stats.num_updated_rows, 2);
        assert_eq!(
            patched.get_fragments().len(),
            1,
            "in-place patches add no fragment"
        );
        let patch_version = patched.version().version;
        let after = by_key(&collect_rows(&patched).await);

        for key in [300, 700] {
            let (id, created, _) = before[&key];
            assert_eq!(after[&key], (id, created, patch_version), "row {key}");
        }
        for (key, lineage) in before.iter().filter(|(key, _)| ![300, 700].contains(key)) {
            assert_eq!(after[key], *lineage, "row {key} must be untouched");
        }
        patched.validate().await.unwrap();
    }

    /// An update carries the rewritten rows' ids and created-at versions over
    /// and the commit stamps their last-updated-at version. On an opted-in
    /// table what the update carries over spills at write time -- it is known
    /// before the commit and a retry cannot change it -- while the commit's
    /// stamp stays inline. Without the opt-in everything stays inline, as
    /// every release has, and the lineage is the same.
    #[rstest]
    #[case::opted_in(true)]
    #[case::not_opted_in(false)]
    #[tokio::test]
    async fn update_places_the_rewritten_rows_lineage(#[case] opted_in: bool) {
        let dir = TempStrDir::default();
        let uri = dir.as_str();
        let mut dataset = appended_dataset(uri, 4, 250).await;
        if opted_in {
            spill_everything(&mut dataset).await;
        }
        let before = by_key(&collect_rows(&dataset).await);

        let updated = UpdateBuilder::new(Arc::new(dataset))
            .update_where("i >= 500")
            .unwrap()
            .set("i", "i + 10000")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        let updated = updated.new_dataset.as_ref();
        let update_version = updated.version().version;

        if opted_in {
            let rewritten = updated
                .get_fragments()
                .into_iter()
                .map(|fragment| fragment.metadata().clone())
                .find(|metadata| matches!(metadata.row_id_meta, Some(RowIdMeta::Column)))
                .expect("the rewritten rows' fragment must spill its row ids");
            assert!(
                matches!(
                    rewritten.created_at_version_meta,
                    Some(RowDatasetVersionMeta::Column)
                ),
                "the created-at versions must spill with the row ids"
            );
            // The lineage file is one of the fragment's files, after its data,
            // and holds only what the update carried over.
            assert_eq!(rewritten.files.len(), 2);
            assert_eq!(
                rewritten.files[1].fields.as_ref(),
                [ROW_ID_FIELD_ID, ROW_CREATED_AT_VERSION_FIELD_ID]
            );
            assert!(
                matches!(
                    rewritten.last_updated_at_version_meta,
                    Some(RowDatasetVersionMeta::Inline(_))
                ),
                "the commit stamps last-updated-at inline, got {:?}",
                rewritten.last_updated_at_version_meta
            );
            // The source fragments were plain appends, so this update is the
            // commit that first spills anything and has to raise the flag.
            assert_ne!(
                updated.manifest.reader_feature_flags & FLAG_UNSTABLE_SPILLED_ROW_LINEAGE,
                0,
                "a spilled sequence must set the reader feature flag"
            );
            assert_ne!(
                updated.manifest.writer_feature_flags & FLAG_UNSTABLE_SPILLED_ROW_LINEAGE,
                0,
                "a spilled sequence must set the writer feature flag"
            );
        } else {
            for fragment in updated.get_fragments() {
                assert!(
                    !fragment.metadata().has_spilled_row_lineage(),
                    "fragment {} spilled without the table opting in",
                    fragment.id()
                );
            }
        }

        let after = by_key(&collect_rows(updated).await);
        for (key, (id, created, updated_at)) in before.iter() {
            if *key >= 500 {
                assert_eq!(
                    after[&(key + 10000)],
                    (*id, *created, update_version),
                    "row {key}"
                );
            } else {
                assert_eq!(after[key], (*id, *created, *updated_at), "row {key}");
            }
        }
        updated.validate().await.unwrap();

        // Re-opened cold, so the lineage is read through the committed
        // manifest rather than from this process's caches.
        let reopened = Dataset::open(uri).await.unwrap();
        assert_eq!(by_key(&collect_rows(&reopened).await), after);
        reopened.validate().await.unwrap();
    }

    /// The write that spills the fixture's lineage ahead of a schema change.
    /// Both leave it in a lineage-only file next to the fragment's data.
    #[derive(Debug, Clone, Copy)]
    enum SpillingWrite {
        /// An update rewriting the rows with `i >= 500`.
        Update,
        /// A binary-copy compaction into a single fragment. A reencoding one
        /// would write the lineage next to `i` in the data file, which the
        /// schema change keeps for `i` alone.
        Compaction,
    }

    /// A change to column `j` that adds or removes no rows.
    #[derive(Debug, Clone, Copy)]
    enum SchemaChange {
        Rename,
        Drop,
        Cast,
    }

    /// A schema change keeps only the data files that still hold a schema
    /// field, and the reserved ids of spilled lineage never are one. Renames
    /// and drops commit a projection and a cast rewrites the column; each must
    /// keep the file carrying the lineage, which is its only copy.
    #[rstest]
    #[case::update_then_rename(SpillingWrite::Update, SchemaChange::Rename)]
    #[case::update_then_drop(SpillingWrite::Update, SchemaChange::Drop)]
    #[case::update_then_cast(SpillingWrite::Update, SchemaChange::Cast)]
    #[case::compact_then_drop(SpillingWrite::Compaction, SchemaChange::Drop)]
    #[tokio::test]
    async fn schema_change_keeps_spilled_lineage(
        #[case] spilling_write: SpillingWrite,
        #[case] change: SchemaChange,
    ) {
        let dir = TempStrDir::default();
        let uri = dir.as_str();
        let mut dataset = appended_dataset_with(uri, 4, 250, UserColumns::WithCopy, None).await;
        spill_everything(&mut dataset).await;
        match spilling_write {
            SpillingWrite::Update => {
                let updated = UpdateBuilder::new(Arc::new(dataset))
                    .update_where("i >= 500")
                    .unwrap()
                    .set("i", "i + 10000")
                    .unwrap()
                    .build()
                    .unwrap()
                    .execute()
                    .await
                    .unwrap();
                dataset = updated.new_dataset.as_ref().clone();
            }
            SpillingWrite::Compaction => {
                let options = CompactionOptions {
                    compaction_mode: Some(CompactionMode::ForceBinaryCopy),
                    ..one_fragment()
                };
                compact_files(&mut dataset, options, None).await.unwrap();
            }
        }
        // Each spilled fragment keeps its lineage in a file of its own, which
        // nothing but the lineage keeps alive through the schema change.
        let spilled = dataset
            .manifest
            .fragments
            .iter()
            .filter(|fragment| fragment.has_spilled_row_lineage())
            .collect::<Vec<_>>();
        assert!(
            !spilled.is_empty(),
            "the fixture must spill before the schema change"
        );
        for fragment in spilled {
            assert_eq!(fragment.files.len(), 2, "{fragment:?}");
            assert!(
                fragment.files[1].fields.iter().all(|field| *field < 0),
                "expected a lineage-only file: {fragment:?}"
            );
        }
        let before = by_key(&collect_rows(&dataset).await);

        let j = || ColumnAlteration::new("j".to_string());
        let changed = match change {
            SchemaChange::Rename => dataset.alter_columns(&[j().rename("k".to_string())]).await,
            SchemaChange::Drop => dataset.drop_columns(&["j"]).await,
            SchemaChange::Cast => dataset.alter_columns(&[j().cast_to(DataType::Int64)]).await,
        };
        changed.unwrap();

        let mut expected = before;
        if matches!(change, SchemaChange::Cast) {
            // A cast rewrites `j` in every fragment, which the commit records
            // as an update of every row; ids and created-at stay.
            let cast_version = dataset.version().version;
            for (_, _, updated_at) in expected.values_mut() {
                *updated_at = cast_version;
            }
        }
        assert_eq!(by_key(&collect_rows(&dataset).await), expected);
        dataset.validate().await.unwrap();

        // Re-opened cold, so the lineage is read back from the committed files.
        let reopened = Dataset::open(uri).await.unwrap();
        assert_eq!(by_key(&collect_rows(&reopened).await), expected);
    }

    /// How [`dropping_every_column_of_a_lineage_carrier_keeps_lineage_until_compaction`]
    /// takes every user column away from the data file that compaction wrote
    /// the lineage into.
    #[derive(Debug, Clone, Copy)]
    enum CarrierChange {
        /// `k` is added in a file of its own and `i` and `j` are dropped. A
        /// drop leaves the files it keeps as they are, so the data file still
        /// lists the ids of `i` and `j`, which the schema no longer has.
        Drop,
        /// `i` and `j` are cast. A cast rewrites the columns under new field
        /// ids into a new file and leaves their old ids, now dead, in the data
        /// file.
        Cast,
        /// `i` and `j` are rewritten under their own field ids by a
        /// `DataReplacement`, which tombstones them in the data file.
        Replace,
    }

    /// Compaction writes the lineage into the fragment's data file, which
    /// then outlives its user columns. It stays, as the lineage's only copy,
    /// and neither reads nor `validate` open it for user data. The next
    /// compaction rewrites the fragment, which moves the lineage next to the
    /// live columns and lets the old file go. Binary copy cannot rewrite it,
    /// so a compaction restricted to binary copy leaves the fragment alone
    /// rather than planning a task that fails.
    #[rstest]
    #[case::drop(CarrierChange::Drop)]
    #[case::cast(CarrierChange::Cast)]
    #[case::replace(CarrierChange::Replace)]
    #[tokio::test]
    async fn dropping_every_column_of_a_lineage_carrier_keeps_lineage_until_compaction(
        #[case] change: CarrierChange,
    ) {
        let dir = TempStrDir::default();
        let uri = dir.as_str();
        let mut dataset = appended_dataset_with(uri, 4, 250, UserColumns::WithCopy, None).await;
        spill_everything(&mut dataset).await;
        compact_files(&mut dataset, one_fragment(), None)
            .await
            .unwrap();
        let (row_ids, created_at, _) = collect_lineage(&dataset).await;

        // The ids the data file lists for `i` and `j` once the change is made.
        let carrier_user_fields = match change {
            CarrierChange::Drop => {
                // `k` lives in a file of its own, so dropping `i` and `j`
                // leaves the compacted file with no field in the schema.
                dataset
                    .add_columns(
                        NewColumnTransform::SqlExpressions(vec![("k".into(), "i + 1".into())]),
                        None,
                        None,
                    )
                    .await
                    .unwrap();
                dataset.drop_columns(&["i", "j"]).await.unwrap();
                [0, 1]
            }
            CarrierChange::Cast => {
                let alterations = [
                    ColumnAlteration::new("i".into()).cast_to(DataType::Int64),
                    ColumnAlteration::new("j".into()).cast_to(DataType::Int64),
                ];
                dataset.alter_columns(&alterations).await.unwrap();
                [0, 1]
            }
            CarrierChange::Replace => {
                let batch = keyed_batch(UserColumns::WithCopy, 0..1_000);
                let fragments = dataset.get_fragments();
                let replacement = fragments[0]
                    .write_columns(futures::stream::iter([Ok(batch)]), dataset.schema())
                    .await
                    .unwrap();
                let read_version = dataset.manifest.version;
                let operation = Operation::DataReplacement {
                    replacements: vec![replacement],
                };
                dataset = Dataset::commit(
                    Arc::new(dataset),
                    operation,
                    Some(read_version),
                    None,
                    None,
                    Arc::new(Default::default()),
                    false,
                )
                .await
                .unwrap();
                [TOMBSTONE_FIELD_ID, TOMBSTONE_FIELD_ID]
            }
        };

        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        let metadata = fragments[0].metadata();
        let carrier = metadata
            .row_lineage_file(ROW_ID_FIELD_ID)
            .unwrap()
            .expect("the schema change must keep the lineage carrier");
        let (user_fields, lineage_fields) = carrier.fields.split_at(2);
        assert_eq!(user_fields, carrier_user_fields, "{metadata:?}");
        assert_eq!(
            lineage_fields,
            [
                ROW_ID_FIELD_ID,
                ROW_CREATED_AT_VERSION_FIELD_ID,
                ROW_LAST_UPDATED_AT_VERSION_FIELD_ID
            ]
        );
        assert!(
            carrier
                .fields
                .iter()
                .all(|field_id| dataset.schema().field_by_id(*field_id).is_none()),
            "the carrier must have lost every user column: {metadata:?}"
        );
        dataset.validate().await.unwrap();
        // A change may stamp every row as updated, so only the row ids and
        // created-at versions are compared.
        let changed = collect_lineage(&dataset).await;
        assert_eq!((&changed.0, &changed.1), (&row_ids, &created_at));

        let binary_copy_only = CompactionOptions {
            compaction_mode: Some(CompactionMode::ForceBinaryCopy),
            ..one_fragment()
        };
        let plan = plan_compaction(&dataset, &binary_copy_only).await.unwrap();
        assert_eq!(plan.num_tasks(), 0);

        compact_files(&mut dataset, one_fragment(), None)
            .await
            .unwrap();

        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        let metadata = fragments[0].metadata();
        let mut expected_fields = dataset.schema().field_ids();
        expected_fields.extend([
            ROW_ID_FIELD_ID,
            ROW_CREATED_AT_VERSION_FIELD_ID,
            ROW_LAST_UPDATED_AT_VERSION_FIELD_ID,
        ]);
        assert_eq!(metadata.files.len(), 1, "{metadata:?}");
        assert_eq!(
            metadata.files[0].fields.as_ref(),
            expected_fields.as_slice()
        );
        assert_eq!(collect_lineage(&dataset).await, changed);
        dataset.validate().await.unwrap();
        let reopened = Dataset::open(uri).await.unwrap();
        assert_eq!(collect_lineage(&reopened).await, changed);

        // What compaction wrote holds no dead user column, so it plans no
        // further rewrite.
        let plan = plan_compaction(&dataset, &one_fragment()).await.unwrap();
        assert_eq!(plan.num_tasks(), 0);
    }

    /// `FileFragment::validate` leaves a file without a schema field unopened
    /// only when the fragment keeps it for a spilled sequence it carries, as
    /// it keeps a lineage carrier whose user columns are gone. Any other file
    /// whose user fields are all outside the schema is opened and reported, as
    /// it always was, even when it lists a lineage id the fragment does not
    /// spill.
    #[rstest]
    #[case::stale_user_file(vec![7], false)]
    #[case::dead_lineage_carrier(vec![7, ROW_ID_FIELD_ID], true)]
    #[case::unspilled_lineage_id(vec![7, ROW_ID_FIELD_ID], false)]
    #[tokio::test]
    async fn validate_skips_only_files_kept_for_spilled_lineage(
        #[case] fields: Vec<i32>,
        #[case] spills_row_ids: bool,
    ) {
        let dir = TempStrDir::default();
        let dataset = tiny_dataset(dir.as_str()).await;
        let mut fragment = dataset.manifest.fragments[0].clone();
        // A second entry for the fragment's data file, under field ids the
        // schema does not have.
        let mut extra = fragment.files[0].clone();
        extra.column_indices = (0..fields.len() as i32).collect();
        extra.fields = fields.into();
        fragment.files.push(extra);
        if spills_row_ids {
            fragment.row_id_meta = Some(RowIdMeta::Column);
        }

        let result = FileFragment::new(Arc::new(dataset), fragment)
            .validate()
            .await;
        if spills_row_ids {
            result.unwrap();
        } else {
            let error = result.unwrap_err();
            assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
            assert!(
                error
                    .to_string()
                    .contains("did not have any fields in common with the dataset schema"),
                "{error}"
            );
        }
    }
}
