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

use super::lineage_field;
use arrow_array::{Array, ArrayRef, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use lance_core::datatypes::Schema;
use lance_core::{Error, ROW_CREATED_AT_VERSION, ROW_ID, ROW_LAST_UPDATED_AT_VERSION, Result};
use lance_file::version::ConcreteFileVersion;
use lance_file::versions;
use lance_file::writer::FileWriterOptions;
use lance_table::format::{
    DataFile, Fragment, ROW_CREATED_AT_VERSION_FIELD_ID, ROW_ID_FIELD_ID,
    ROW_LAST_UPDATED_AT_VERSION_FIELD_ID, RowDatasetVersionMeta, RowDatasetVersionSequence,
    RowIdMeta,
};
use lance_table::rowids::version::write_dataset_versions;
use lance_table::rowids::{RowIdSequence, write_row_ids};

use crate::Table;
use crate::fragment::generate_random_filename;

/// Rows per batch handed to the file writer.
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
/// sequence stays inline however large it grows. This affects new writes only;
/// readers still load any sequences the table has already spilled.
pub const SPILL_ROW_LINEAGE_CONFIG_KEY: &str = "lance.row_lineage.spill";

/// Table config key overriding [`DEFAULT_INLINE_ROW_LINEAGE_MAX_BYTES`], as a
/// byte count. Only consulted when [`SPILL_ROW_LINEAGE_CONFIG_KEY`] is on.
pub const INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY: &str = "lance.row_lineage.inline_max_bytes";

/// The largest encoded lineage sequence `dataset` keeps inline, or `None` when
/// the table does not spill at all.
///
/// Spilling needs the table's opt-in and v2 data files: the format allows the
/// columns only there, since a legacy v1 file has no `column_indices` to locate
/// them by.
pub fn inline_row_lineage_max_bytes(dataset: &Table) -> Result<Option<usize>> {
    let config = &dataset.manifest.config;
    let enabled = config
        .get(SPILL_ROW_LINEAGE_CONFIG_KEY)
        .is_some_and(|value| value.eq_ignore_ascii_case("true"));
    if !enabled
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
        [
            ROW_ID_FIELD_ID,
            ROW_CREATED_AT_VERSION_FIELD_ID,
            ROW_LAST_UPDATED_AT_VERSION_FIELD_ID,
        ]
        .into_iter()
        .zip(flags)
        .filter_map(|(field_id, spilled)| spilled.then_some(field_id))
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
pub async fn place_row_lineage(dataset: &Table, lineage: &RowLineage) -> Result<PlacedRowLineage> {
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
    dataset: &Table,
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
    dataset: &Table,
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
    let object_writer = dataset.state.object_store.create(&full_path).await?;
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
