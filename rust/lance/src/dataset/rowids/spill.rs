// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Spilling a fragment's row id sequence out of the manifest and into a hidden
//! column of a Lance data file.
//!
//! A row id sequence is run-encoded, so an appended fragment costs about 20
//! bytes of manifest and never needs to leave it. A fragment whose rows came
//! from many places -- the output of compacting a shuffled table, for instance
//! -- has no runs to exploit and falls back to 8 bytes per row. Inline, that
//! cost is paid again in every manifest version, so the manifest grows with the
//! table and every commit rewrites all of it.
//!
//! Spilled, the sequence is an ordinary `UInt64` column carrying field id
//! [`ROW_ID_FIELD_ID`], written with the same encodings and read with the same
//! reader as user data. The manifest keeps only the [`DataFile`] that locates
//! it.

use std::sync::Arc;

use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use futures::TryStreamExt;
use lance_core::datatypes::Schema;
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::reader::FileReader;
use lance_file::versions;
use lance_file::writer::FileWriterOptions;
use lance_io::ReadBatchParams;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_table::format::{DataFile, ROW_ID_FIELD_ID, RowIdMeta};
use lance_table::rowids::{RowIdSequence, write_row_ids};
use object_store::path::Path;

use super::super::Dataset;
use crate::dataset::fragment::write::generate_random_filename;
use crate::{Error, Result};

/// Name of the hidden column holding a spilled row id sequence. Only the field
/// id is load-bearing; the name is for humans reading a file dump.
const ROW_ID_COLUMN_NAME: &str = "_rowid";

/// Rows per batch handed to the file writer.
const SPILL_BATCH_ROWS: usize = 64 * 1024;

/// Encoded sequences at or below this size stay in the manifest.
///
/// Matches the inline limit the format has always documented for the
/// `row_id_sequence` oneof. A `Range` sequence -- every appended fragment --
/// encodes to a few dozen bytes and is nowhere near it.
pub const DEFAULT_INLINE_ROW_IDS_MAX_BYTES: usize = 200 * 1024;

/// Place `sequence` either inline in the manifest or in a data file column,
/// whichever its encoded size calls for. `inline_max_bytes` defaults to
/// [`DEFAULT_INLINE_ROW_IDS_MAX_BYTES`].
pub async fn build_row_id_meta(
    dataset: &Dataset,
    sequence: &RowIdSequence,
    inline_max_bytes: Option<usize>,
) -> Result<RowIdMeta> {
    let encoded = write_row_ids(sequence);
    let limit = inline_max_bytes.unwrap_or(DEFAULT_INLINE_ROW_IDS_MAX_BYTES);
    if encoded.len() <= limit || !lance_table::feature_flags::spilled_row_ids_enabled() {
        return Ok(RowIdMeta::Inline(encoded.into()));
    }
    Ok(RowIdMeta::Column(
        spill_row_id_sequence(dataset, sequence).await?,
    ))
}

/// Write `sequence` as a hidden column and return the data file that holds it.
async fn spill_row_id_sequence(dataset: &Dataset, sequence: &RowIdSequence) -> Result<DataFile> {
    let file_version = dataset.manifest.data_storage_format.version;
    let filename = format!("{}.lance", generate_random_filename());
    let full_path = dataset.data_dir().join(filename.as_str());

    let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        ROW_ID_COLUMN_NAME,
        DataType::UInt64,
        false,
    )]));
    let schema = Schema::try_from(arrow_schema.as_ref())?;
    let object_writer = dataset.object_store.create(&full_path).await?;
    let mut writer = versions::create_writer(
        file_version,
        object_writer,
        schema,
        FileWriterOptions::default(),
    )?;

    // Materialized up front rather than streamed from `sequence.iter()`: that
    // returns a boxed `dyn DoubleEndedIterator`, which is not `Send`, so holding
    // it across the write below would make this future non-`Send` and every
    // caller of `compact_files` along with it -- including the Python bindings,
    // which spawn that future. The batches are zero-copy slices of it.
    let ids = UInt64Array::from(sequence.iter().collect::<Vec<u64>>());
    for offset in (0..ids.len()).step_by(SPILL_BATCH_ROWS) {
        let len = SPILL_BATCH_ROWS.min(ids.len() - offset);
        let batch =
            RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(ids.slice(offset, len))])?;
        writer.write_batch(&batch).await?;
    }
    let summary = writer.finish().await?;

    Ok(DataFile::new(
        filename,
        vec![ROW_ID_FIELD_ID],
        vec![0],
        file_version,
        std::num::NonZero::new(summary.size_bytes),
        None,
    ))
}

/// Read back a sequence spilled by [`spill_row_id_sequence`].
pub async fn read_spilled_row_id_sequence(
    dataset: &Dataset,
    data_file: &DataFile,
) -> Result<RowIdSequence> {
    let column_index = data_file
        .fields
        .iter()
        .position(|field| *field == ROW_ID_FIELD_ID)
        .and_then(|position| data_file.column_indices.get(position))
        .ok_or_else(|| {
            Error::corrupt_file_named(
                &data_file.path,
                format!("spilled row id file does not carry field id {ROW_ID_FIELD_ID}"),
            )
        })?;
    if *column_index != 0 {
        return Err(Error::not_supported(format!(
            "spilled row ids at column index {column_index}; only a dedicated file is supported"
        )));
    }

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

    let mut ids: Vec<u64> = Vec::with_capacity(reader.num_rows() as usize);
    let mut stream = reader
        .read_stream(
            ReadBatchParams::RangeFull,
            SPILL_BATCH_ROWS as u32,
            8,
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
                    "spilled row id column is not UInt64".to_string(),
                )
            })?;
        ids.extend_from_slice(column.values());
    }

    Ok(RowIdSequence::from(ids.as_slice()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::cleanup::{CleanupPolicyBuilder, cleanup_old_versions};
    use crate::dataset::optimize::{CompactionOptions, compact_files};
    use crate::dataset::{WriteMode, WriteParams};
    use arrow_array::{Int32Array, RecordBatchIterator};
    use arrow_schema::Field;
    use chrono::Utc;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_table::feature_flags::FLAG_UNSTABLE_SPILLED_ROW_IDS;

    /// A sequence with no runs to exploit, which is what a globally shuffled
    /// table produces and what forces the spill path.
    fn scattered_sequence(len: u64) -> RowIdSequence {
        // A stride coprime with `len` visits every id exactly once in an order
        // with no ascending run longer than one.
        let ids: Vec<u64> = (0..len).map(|i| (i * 7919) % len).collect();
        RowIdSequence::from(ids.as_slice())
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

    #[tokio::test]
    async fn spilled_sequence_round_trips() {
        let dir = TempStrDir::default();
        let dataset = tiny_dataset(dir.as_str()).await;

        let sequence = scattered_sequence(20_000);
        let meta = build_row_id_meta(&dataset, &sequence, Some(0))
            .await
            .unwrap();
        let RowIdMeta::Column(data_file) = &meta else {
            panic!("expected the sequence to spill, got {meta:?}");
        };
        assert_eq!(data_file.fields.as_ref(), [ROW_ID_FIELD_ID]);

        let restored = read_spilled_row_id_sequence(&dataset, data_file)
            .await
            .unwrap();
        assert_eq!(
            restored.iter().collect::<Vec<_>>(),
            sequence.iter().collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn sequence_under_the_limit_stays_inline() {
        let dir = TempStrDir::default();
        let dataset = tiny_dataset(dir.as_str()).await;

        // An appended fragment's sequence is a single `Range`, so it encodes to
        // a few dozen bytes and must never leave the manifest.
        let meta = build_row_id_meta(&dataset, &RowIdSequence::from(0..1_000_000), None)
            .await
            .unwrap();
        assert!(
            matches!(meta, RowIdMeta::Inline(_)),
            "a range sequence must stay inline, got {meta:?}"
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

    /// Force the spill path regardless of size: reaching the natural 200 KiB
    /// threshold needs ~25k scattered rows, more than these tests need to prove.
    fn spill_everything() -> CompactionOptions {
        CompactionOptions {
            target_rows_per_fragment: 1_000,
            inline_row_ids_max_bytes: Some(0),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn compaction_spills_and_reads_back_row_ids() {
        let dir = TempStrDir::default();
        let uri = dir.as_str();
        let mut dataset = appended_dataset(uri, 4, 250).await;
        let before = collect_row_ids(&dataset).await;

        compact_files(&mut dataset, spill_everything(), None)
            .await
            .unwrap();

        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        assert!(
            matches!(
                fragments[0].metadata().row_id_meta,
                Some(RowIdMeta::Column(_))
            ),
            "compaction must spill under a zero inline budget, got {:?}",
            fragments[0].metadata().row_id_meta
        );
        assert_ne!(
            dataset.manifest.reader_feature_flags & FLAG_UNSTABLE_SPILLED_ROW_IDS,
            0,
            "a spilled sequence must set the reader feature flag"
        );

        // The ids survive the rewrite and are still readable through the
        // ordinary scan path, now served from the data file column.
        assert_eq!(collect_row_ids(&dataset).await, before);
        // `validate_stable_row_ids` reads every fragment's sequence back and
        // checks it against the fragment length, so this covers the read path
        // for a spilled sequence independently of the scan.
        dataset.validate().await.unwrap();
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
        let before = collect_row_ids(&dataset).await;

        compact_files(&mut dataset, spill_everything(), None)
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
            "cleanup deleted the live spilled row id file at {on_disk:?}"
        );

        let reopened = Dataset::open(uri).await.unwrap();
        assert_eq!(collect_row_ids(&reopened).await, before);
    }

    async fn collect_row_ids(dataset: &Dataset) -> Vec<u64> {
        let mut scanner = dataset.scan();
        scanner.with_row_id().project::<&str>(&[]).unwrap();
        let batch = scanner.try_into_batch().await.unwrap();
        batch
            .column_by_name("_rowid")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .values()
            .to_vec()
    }
}
