// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Read row lineage stored as reserved columns in data files.

use std::sync::Arc;

use arrow_array::{Array, UInt64Array};
use arrow_schema::{DataType, Field as ArrowField};
use futures::{FutureExt, StreamExt, TryStreamExt};
use lance_core::datatypes::Schema;
use lance_core::{Error, ROW_CREATED_AT_VERSION, ROW_ID, ROW_LAST_UPDATED_AT_VERSION, Result};
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::LanceEncodingsIo;
use lance_file::reader::{FileReader, ProjectedFileReader, ReaderProjection};
use lance_file::versions;
use lance_io::ReadBatchParams;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_table::format::{
    Fragment, ROW_CREATED_AT_VERSION_FIELD_ID, ROW_ID_FIELD_ID,
    ROW_LAST_UPDATED_AT_VERSION_FIELD_ID, RowDatasetVersionSequence,
};
use lance_table::rowids::RowIdSequence;

use crate::Table;

/// Number of rows to decode per batch when reading a whole lineage column.
const READ_BATCH_ROWS: u32 = 64 * 1024;

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

/// The hidden column a spilled lineage sequence is stored in: a non-nullable
/// `UInt64` named after the sequence in `LINEAGE_COLUMNS`, under its
/// reserved `field_id`.
pub fn lineage_field(field_id: i32) -> Result<lance_core::datatypes::Field> {
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

/// Read back `fragment`'s row id sequence from the data file that carries it.
pub async fn read_spilled_row_ids(table: &Table, fragment: &Fragment) -> Result<RowIdSequence> {
    let ids = read_spilled_column(table, fragment, ROW_ID_FIELD_ID)
        .boxed()
        .await?;
    Ok(RowIdSequence::from(ids.as_slice()))
}

/// Read back one of `fragment`'s version sequences from the data file that
/// carries it; `field_id` says which of the two it is.
pub async fn read_spilled_versions(
    table: &Table,
    fragment: &Fragment,
    field_id: i32,
) -> Result<RowDatasetVersionSequence> {
    let versions = read_spilled_column(table, fragment, field_id)
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
    table: &Table,
    fragment: &Fragment,
    field_id: i32,
) -> Result<Vec<u64>> {
    let data_file = fragment.row_lineage_file(field_id)?.ok_or_else(|| {
        Error::corrupt_file(
            table.state.base.clone(),
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
    let path = table
        .data_file_dir(data_file)?
        .join(data_file.path.as_str());
    let object_store = table.object_store(data_file.base_id).await?;
    let scheduler = ScanScheduler::new(
        object_store.clone(),
        SchedulerConfig::max_bandwidth(&object_store),
    );
    let file = scheduler
        .open_file(&path, &data_file.file_size_bytes)
        .await?;
    let options = table
        .state
        .file_reader_options
        .as_deref()
        .cloned()
        .unwrap_or_default();
    let cache = table.state.metadata_cache.file_metadata_cache(&path);
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
            READ_BATCH_ROWS,
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
