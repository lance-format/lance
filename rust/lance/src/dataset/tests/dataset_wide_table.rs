// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Characterization tests for metadata amplification on wide tables.
//!
//! The manifest is a full snapshot: adding a single column re-serializes the
//! whole manifest (and the resulting transaction carries the full fragment
//! list). These tests pin down the *observable* metadata costs so that future
//! work (e.g. column-group manifests) can be validated against them:
//!
//! 1. [`test_add_column_rewrites_full_manifest`]: a one-column add rewrites the
//!    whole manifest object, so the new manifest is a full snapshot rather than
//!    a delta;
//! 2. [`test_cold_open_reads_full_manifest`]: a cold open reads at least the
//!    manifest size; a projected query still has to open the dataset first;
//! 3. [`test_version_refresh_rereads_manifest`]: after another writer commits
//!    a new version, the next `checkout_latest` re-reads the full manifest;
//! 4. [`test_compaction_rewrites_every_column`]: the one in-tree remedy for the
//!    data-file fragmentation that `add_columns` creates is a full rewrite, so
//!    repairing the layout costs a write of every column, not just the one that
//!    was added.
//!
//! These document current behavior. A format change that improves metadata
//! amplification is expected to *relax* these assertions in the same PR that
//! lands the improvement.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};

use crate::Dataset;
use crate::dataset::optimize::{CompactionOptions, compact_files};
use crate::dataset::{NewColumnTransform, WriteParams};

const NUM_COLUMNS: usize = 256;
const NUM_FRAGMENTS: usize = 8;
const ROWS_PER_FRAGMENT: usize = 16;

async fn create_wide_dataset(uri: &str) -> Dataset {
    let fields: Vec<ArrowField> = (0..NUM_COLUMNS)
        .map(|i| ArrowField::new(format!("col_{i:04}"), DataType::Int32, true))
        .collect();
    let schema = Arc::new(ArrowSchema::new(fields));

    let mut batches = Vec::with_capacity(NUM_FRAGMENTS);
    for _ in 0..NUM_FRAGMENTS {
        let columns: Vec<ArrayRef> = (0..NUM_COLUMNS)
            .map(|_| {
                Arc::new(Int32Array::from_iter_values(
                    (0..ROWS_PER_FRAGMENT).map(|i| (i % 100) as i32),
                )) as ArrayRef
            })
            .collect();
        batches.push(Ok(RecordBatch::try_new(schema.clone(), columns).unwrap()));
    }
    let reader = RecordBatchIterator::new(batches, schema);

    let write_params = WriteParams {
        max_rows_per_file: ROWS_PER_FRAGMENT,
        ..Default::default()
    };
    Dataset::write(reader, uri, Some(write_params))
        .await
        .unwrap()
}

#[tokio::test]
async fn test_add_column_rewrites_full_manifest() {
    let tmpdir = tempfile::tempdir().unwrap();
    let uri = tmpdir.path().to_str().unwrap();
    let mut dataset = create_wide_dataset(uri).await;

    let base_snapshot = dataset.manifest().serialized().len() as u64;
    assert!(
        base_snapshot > 10_000,
        "expected a non-trivial manifest for {NUM_COLUMNS} columns, got {base_snapshot} bytes"
    );

    // Size of the manifest object that v1 points at, as reported by the writer.
    // This is the whole manifest: the flat format has no way to express "one
    // column changed".
    let base_manifest = dataset
        .manifest_location
        .size
        .expect("the writer should report the size of the manifest it wrote");

    dataset
        .add_columns(
            NewColumnTransform::SqlExpressions(vec![(
                "new_col".to_string(),
                "col_0000".to_string(),
            )]),
            None,
            None,
        )
        .await
        .unwrap();

    // The commit rewrites the entire manifest file, so the object v2 points at
    // must be a full snapshot no smaller than v1's. That object holds the
    // serialized manifest plus the inline transaction, which carries the full
    // fragment list again - it is the real per-commit metadata payload.
    //
    // Measuring it directly, rather than the operation's total written bytes, is
    // what keeps this assertion meaningful: the total is dominated by data and
    // transaction objects and stays above `base_manifest` even if the manifest
    // stops being rewritten, which would leave this test silently green. The
    // manifest object itself collapses to a small delta in that case and fails
    // here - the signal this test exists to provide.
    let after_manifest = dataset
        .manifest_location
        .size
        .expect("the writer should report the size of the manifest it wrote");
    assert!(
        after_manifest >= base_manifest,
        "expected a one-column add to rewrite the whole {base_manifest} byte manifest, \
         but v{}'s manifest is only {after_manifest} bytes",
        dataset.manifest_location.version
    );

    let grown = dataset.manifest().serialized().len();
    assert!(
        grown > base_snapshot as usize,
        "manifest should grow after adding a column"
    );
}

#[tokio::test]
async fn test_cold_open_reads_full_manifest() {
    let tmpdir = tempfile::tempdir().unwrap();
    let uri = tmpdir.path().to_str().unwrap();
    let dataset = create_wide_dataset(uri).await;
    let manifest = dataset.manifest().serialized().len() as u64;
    drop(dataset);

    // Cold open with a fresh session: opening the dataset reads the full
    // manifest before any query can run.
    let ds = Dataset::open(uri).await.unwrap();
    let io = ds.object_store.as_ref().io_stats_snapshot();
    assert!(
        io.read_bytes >= manifest,
        "expected a cold open to read at least the manifest size \
         ({manifest} bytes), but read {} bytes",
        io.read_bytes
    );
}

#[tokio::test]
async fn test_version_refresh_rereads_manifest() {
    let tmpdir = tempfile::tempdir().unwrap();
    let uri = tmpdir.path().to_str().unwrap();
    create_wide_dataset(uri).await;

    // Reader with its own session, pinned to the initial version.
    let mut reader = Dataset::open(uri).await.unwrap();
    assert_eq!(reader.version().version, 1);

    // Another writer commits a one-column addition.
    let mut writer = Dataset::open(uri).await.unwrap();
    writer
        .add_columns(
            NewColumnTransform::SqlExpressions(vec![(
                "new_col".to_string(),
                "col_0000".to_string(),
            )]),
            None,
            None,
        )
        .await
        .unwrap();
    let manifest_v2 = writer.manifest().serialized().len() as u64;

    // The reader's next version refresh reads the full manifest for the new
    // version.
    reader.object_store.as_ref().io_stats_incremental();
    reader.checkout_latest().await.unwrap();
    assert_eq!(reader.version().version, 2);
    let io = reader.object_store.as_ref().io_stats_incremental();
    assert!(
        io.read_bytes >= manifest_v2,
        "expected version refresh to re-read the full manifest \
         ({manifest_v2} bytes), but read {} bytes",
        io.read_bytes
    );
}

/// `add_columns` leaves one extra data file per fragment and `compact_files`
/// is the only in-tree remedy. It repairs the layout by rewriting the
/// fragments, so the commit writes a data file that holds *every* column of the
/// table - the cost of the repair scales with the schema, not with the column
/// that caused the fragmentation.
///
/// This is the baseline that a column-repack task would improve on: a repack
/// merges the same files without moving rows and without touching the columns
/// it does not move. Until that lands, this pins what the rewrite path pays.
#[tokio::test]
async fn test_compaction_rewrites_every_column() {
    let tmpdir = tempfile::tempdir().unwrap();
    let uri = tmpdir.path().to_str().unwrap();
    let mut dataset = create_wide_dataset(uri).await;

    // A single backfilled column gives every fragment one more data file:
    // the base file with all `NUM_COLUMNS` columns, plus one file with just
    // `new_col`.
    dataset
        .add_columns(
            NewColumnTransform::SqlExpressions(vec![(
                "new_col".to_string(),
                "col_0000".to_string(),
            )]),
            None,
            None,
        )
        .await
        .unwrap();

    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), NUM_FRAGMENTS);
    let files_before: usize = fragments.iter().map(|f| f.metadata().files.len()).sum();
    assert_eq!(
        files_before,
        2 * NUM_FRAGMENTS,
        "the backfill should add exactly one data file per fragment"
    );

    // The file the backfill wrote holds one column of this fragment's rows.
    // Its size is the yardstick for what "adding a column" costs.
    let backfill_bytes = fragments
        .iter()
        .flat_map(|f| f.metadata().files.iter())
        .filter(|file| file.fields.len() == 1)
        .filter_map(|file| file.file_size_bytes.get())
        .map(|size| size.get())
        .max()
        .expect("the backfilled file should report its size");

    dataset.object_store.as_ref().io_stats_incremental();
    let metrics = compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();
    let io = dataset.object_store.as_ref().io_stats_incremental();

    assert_eq!(metrics.fragments_removed, NUM_FRAGMENTS);
    assert_eq!(metrics.fragments_added, 1);
    assert_eq!(metrics.files_added, 1);
    assert_eq!(metrics.files_removed, files_before);

    // The rewrite collapsed the dataset into a single fragment with a single
    // data file.
    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), 1);
    let files = &fragments[0].metadata().files;
    assert_eq!(files.len(), 1);

    // That file holds every column of the table, not just the backfilled one,
    // so the commit had to read and write the whole table.
    assert_eq!(
        files[0].fields.len(),
        NUM_COLUMNS + 1,
        "expected the compacted file to hold every column of the table"
    );

    let rewritten_bytes = files[0]
        .file_size_bytes
        .get()
        .expect("the compacted file should report its size")
        .get();

    // The bytes were actually written by this commit: the repair is a data
    // write, not a metadata-only operation.
    assert!(
        io.written_bytes >= rewritten_bytes,
        "expected the compaction commit to write its {} byte data file, \
         but it wrote {} bytes",
        rewritten_bytes,
        io.written_bytes
    );

    // Repairing the layout costs far more than creating it: the backfill wrote
    // one column, the rewrite writes all of them.
    assert!(
        rewritten_bytes > backfill_bytes * 5,
        "expected the rewrite ({} bytes) to cost far more than the backfilled \
         column ({} bytes)",
        rewritten_bytes,
        backfill_bytes
    );

    // The metadata side is unchanged: the commit still writes a full manifest
    // object, not a delta describing the rewrite. The object is smaller than
    // the one v2 pointed at, because the dataset now has one fragment instead
    // of eight - but it still carries the whole schema and every fragment, so
    // it is at least as large as the manifest it serializes.
    let after_manifest = dataset
        .manifest_location
        .size
        .expect("the writer should report the size of the manifest it wrote");
    let rewritten_manifest = dataset.manifest().serialized().len() as u64;
    assert!(
        after_manifest >= rewritten_manifest,
        "expected the compaction commit to write the full {rewritten_manifest} byte \
         manifest, but it wrote {after_manifest} bytes"
    );
}
