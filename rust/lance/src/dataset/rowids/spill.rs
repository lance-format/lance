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
//! and read with the same reader as user data. A fragment's spilled sequences
//! share one file, and the manifest keeps only the [`DataFile`] that locates
//! them.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use futures::TryStreamExt;
use lance_core::datatypes::Schema;
use lance_core::{ROW_CREATED_AT_VERSION, ROW_ID, ROW_LAST_UPDATED_AT_VERSION};
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::reader::{FileReader, ReaderProjection};
use lance_file::versions;
use lance_file::writer::FileWriterOptions;
use lance_io::ReadBatchParams;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_table::format::{
    DataFile, ROW_CREATED_AT_VERSION_FIELD_ID, ROW_ID_FIELD_ID,
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

/// Table config key that turns spilling on: `"true"` lets compaction move
/// oversized lineage sequences out of the manifest. Absent or anything else,
/// every sequence stays inline however large it grows, which is what every
/// released build does; a table that never sets it stays readable by them.
pub const SPILL_ROW_LINEAGE_CONFIG_KEY: &str = "lance.row_lineage.spill";

/// Table config key overriding [`DEFAULT_INLINE_ROW_LINEAGE_MAX_BYTES`], as a
/// byte count. Only consulted when [`SPILL_ROW_LINEAGE_CONFIG_KEY`] is on.
pub const INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY: &str = "lance.row_lineage.inline_max_bytes";

/// The largest encoded lineage sequence `dataset` keeps inline, or `None` when
/// the table does not spill at all.
///
/// Spilling needs both the table's opt-in and a build that understands the
/// feature flag; a build that does not would write a dataset it then refuses
/// to open.
pub fn inline_row_lineage_max_bytes(dataset: &Dataset) -> Result<Option<usize>> {
    let config = dataset.config();
    let enabled = config
        .get(SPILL_ROW_LINEAGE_CONFIG_KEY)
        .is_some_and(|value| value.eq_ignore_ascii_case("true"));
    if !enabled || !lance_table::feature_flags::spilled_row_lineage_enabled() {
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

/// The per-row lineage of one fragment, in row offset order.
pub struct RowLineage {
    pub row_ids: RowIdSequence,
    pub created_at: RowDatasetVersionSequence,
    pub last_updated_at: RowDatasetVersionSequence,
}

/// Where each of a fragment's lineage sequences ended up.
pub struct RowLineageMeta {
    pub row_ids: RowIdMeta,
    pub created_at: RowDatasetVersionMeta,
    pub last_updated_at: RowDatasetVersionMeta,
}

/// Place each sequence of `lineage` either inline in the manifest or in a
/// hidden column of a new data file, as the table's spill policy (see
/// [`inline_row_lineage_max_bytes`]) and the sequence's encoded size call for.
/// Every sequence that spills goes into one file.
///
/// Only correct for lineage that a commit conflict cannot change: row ids and
/// versions carried over from existing rows. Lineage assigned at commit time --
/// an appended fragment's row ids, an inserted row's created-at version -- has
/// to stay inline, where the commit can still rewrite it.
pub async fn place_row_lineage(dataset: &Dataset, lineage: &RowLineage) -> Result<RowLineageMeta> {
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
            .filter(|file| file.fields.contains(&field_id))
            .cloned()
    };

    Ok(RowLineageMeta {
        row_ids: match holds(ROW_ID_FIELD_ID) {
            Some(file) => RowIdMeta::Column(file),
            None => RowIdMeta::Inline(inline_row_ids.into()),
        },
        created_at: match holds(ROW_CREATED_AT_VERSION_FIELD_ID) {
            Some(file) => RowDatasetVersionMeta::Column(file),
            None => RowDatasetVersionMeta::Inline(inline_created_at.into()),
        },
        last_updated_at: match holds(ROW_LAST_UPDATED_AT_VERSION_FIELD_ID) {
            Some(file) => RowDatasetVersionMeta::Column(file),
            None => RowDatasetVersionMeta::Inline(inline_last_updated_at.into()),
        },
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

/// Read back a row id sequence spilled by [`place_row_lineage`].
pub async fn read_spilled_row_ids(
    dataset: &Dataset,
    data_file: &DataFile,
) -> Result<RowIdSequence> {
    let ids = read_spilled_column(dataset, data_file, ROW_ID_FIELD_ID).await?;
    Ok(RowIdSequence::from(ids.as_slice()))
}

/// Read back a version sequence spilled by [`place_row_lineage`]; `field_id`
/// says which of the two it is.
pub async fn read_spilled_versions(
    dataset: &Dataset,
    data_file: &DataFile,
    field_id: i32,
) -> Result<RowDatasetVersionSequence> {
    let versions = read_spilled_column(dataset, data_file, field_id).await?;
    Ok(RowDatasetVersionSequence::from_versions(&versions))
}

/// Read one hidden `UInt64` column of `data_file` in full.
async fn read_spilled_column(
    dataset: &Dataset,
    data_file: &DataFile,
    field_id: i32,
) -> Result<Vec<u64>> {
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
    let reader = FileReader::try_open(
        file,
        None,
        Arc::<DecoderPlugins>::default(),
        &dataset.metadata_cache.file_metadata_cache(&path),
        dataset.file_reader_options.clone().unwrap_or_default(),
    )
    .await?;

    // The lineage columns are flat primitives, so the file schema's column
    // position is the column index in every file version.
    let field = reader
        .schema()
        .fields
        .get(column_index as usize)
        .ok_or_else(|| {
            Error::corrupt_file_named(
                &data_file.path,
                format!("spilled row lineage file has no column at index {column_index}"),
            )
        })?;
    let projection = ReaderProjection {
        schema: Arc::new(Schema {
            fields: vec![field.clone()],
            metadata: Default::default(),
        }),
        column_indices: vec![column_index],
    };

    let mut values: Vec<u64> = Vec::with_capacity(reader.num_rows() as usize);
    let mut stream = reader
        .read_stream_projected(
            ReadBatchParams::RangeFull,
            SPILL_BATCH_ROWS as u32,
            8,
            projection,
            FilterExpression::no_filter(),
        )
        .await?;
    while let Some(batch) = stream.try_next().await? {
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
        values.extend_from_slice(column.values());
    }

    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::cleanup::{CleanupPolicyBuilder, cleanup_old_versions};
    use crate::dataset::optimize::{CompactionOptions, compact_files};
    use crate::dataset::rowids::{RowVersionKind, load_row_id_sequence, load_row_version_sequence};
    use crate::dataset::{UpdateBuilder, WriteMode, WriteParams};
    use arrow_array::{Int32Array, RecordBatchIterator};
    use arrow_schema::Field;
    use chrono::Utc;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_table::feature_flags::FLAG_UNSTABLE_SPILLED_ROW_LINEAGE;

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
        let meta = place_row_lineage(&dataset, &lineage).await.unwrap();
        let (RowIdMeta::Column(row_id_file), RowDatasetVersionMeta::Column(created_at_file)) =
            (&meta.row_ids, &meta.created_at)
        else {
            panic!("expected every sequence to spill");
        };
        assert_eq!(meta.last_updated_at.column_file(), Some(row_id_file));
        assert_eq!(created_at_file, row_id_file);
        assert_eq!(
            row_id_file.fields.as_ref(),
            [
                ROW_ID_FIELD_ID,
                ROW_CREATED_AT_VERSION_FIELD_ID,
                ROW_LAST_UPDATED_AT_VERSION_FIELD_ID
            ]
        );

        let row_ids = read_spilled_row_ids(&dataset, row_id_file).await.unwrap();
        assert_eq!(
            row_ids.iter().collect::<Vec<_>>(),
            lineage.row_ids.iter().collect::<Vec<_>>()
        );
        let created_at =
            read_spilled_versions(&dataset, row_id_file, ROW_CREATED_AT_VERSION_FIELD_ID)
                .await
                .unwrap();
        assert_eq!(versions_of(&created_at), versions_of(&lineage.created_at));
        let last_updated_at =
            read_spilled_versions(&dataset, row_id_file, ROW_LAST_UPDATED_AT_VERSION_FIELD_ID)
                .await
                .unwrap();
        assert_eq!(
            versions_of(&last_updated_at),
            versions_of(&lineage.last_updated_at)
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
        let meta = place_row_lineage(&dataset, &lineage).await.unwrap();
        assert!(matches!(meta.created_at, RowDatasetVersionMeta::Inline(_)));

        dataset
            .update_config([(SPILL_ROW_LINEAGE_CONFIG_KEY, "true")])
            .await
            .unwrap();
        let meta = place_row_lineage(&dataset, &lineage).await.unwrap();
        assert!(
            matches!(meta.row_ids, RowIdMeta::Inline(_)),
            "a range sequence must stay inline, got {:?}",
            meta.row_ids
        );
        assert!(
            matches!(meta.last_updated_at, RowDatasetVersionMeta::Inline(_)),
            "a single-run sequence must stay inline, got {:?}",
            meta.last_updated_at
        );
        let created_at_file = meta
            .created_at
            .column_file()
            .expect("an alternating version sequence encodes past 200 KiB");
        assert_eq!(
            created_at_file.fields.as_ref(),
            [ROW_CREATED_AT_VERSION_FIELD_ID]
        );
    }

    /// A stable-row-id dataset built from `chunks` separate appends, so
    /// compacting it has several sequences to concatenate.
    async fn appended_dataset(uri: &str, chunks: i32, rows_per_chunk: i32) -> Dataset {
        let schema = test_schema();
        let mut dataset: Option<Dataset> = None;
        for chunk in 0..chunks {
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from_iter_values(
                    (chunk * rows_per_chunk)..((chunk + 1) * rows_per_chunk),
                ))],
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
        dataset.unwrap()
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

    #[tokio::test]
    async fn compaction_spills_and_reads_back_row_lineage() {
        let dir = TempStrDir::default();
        let uri = dir.as_str();
        let mut dataset = appended_dataset(uri, 4, 250).await;
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
        let spilled = metadata
            .row_id_meta
            .as_ref()
            .and_then(RowIdMeta::column_file)
            .expect("compaction must spill the row ids under a zero inline budget");
        assert_eq!(
            metadata
                .created_at_version_meta
                .as_ref()
                .and_then(RowDatasetVersionMeta::column_file),
            Some(spilled),
            "the created-at versions must share the row id file"
        );
        assert_eq!(
            metadata
                .last_updated_at_version_meta
                .as_ref()
                .and_then(RowDatasetVersionMeta::column_file),
            Some(spilled),
            "the last-updated-at versions must share the row id file"
        );
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
            .row_id_meta
            .as_ref()
            .and_then(RowIdMeta::column_file)
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

    /// Resolving the rewritten rows' original created-at versions happens at
    /// commit time, inside `lance-table`, which cannot read a data file. Until
    /// that path can, updating rows whose lineage is spilled must refuse rather
    /// than silently stamp them with a default version.
    #[tokio::test]
    async fn updating_rows_with_spilled_lineage_is_refused() {
        let dir = TempStrDir::default();
        let uri = dir.as_str();
        let mut dataset = appended_dataset(uri, 4, 250).await;
        spill_everything(&mut dataset).await;
        compact_files(&mut dataset, one_fragment(), None)
            .await
            .unwrap();

        let error = UpdateBuilder::new(Arc::new(dataset))
            .update_where("i = 7")
            .unwrap()
            .set("i", "70")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap_err();
        assert!(
            matches!(error, Error::NotSupported { .. }),
            "expected NotSupported, got {error:?}"
        );
    }
}
