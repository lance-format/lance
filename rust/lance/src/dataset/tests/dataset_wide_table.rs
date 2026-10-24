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
//!    manifest size, even for a narrow query;
//! 3. [`test_version_refresh_rereads_manifest`]: after another writer commits
//!    a new version, the next `checkout_latest` re-reads the full manifest,
//!    even though this reader's projection did not change.
//!
//! These document current behavior. A format change that improves metadata
//! amplification is expected to *relax* these assertions in the same PR that
//! lands the improvement.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};

use crate::Dataset;
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

    // Cold open with a fresh session: even though any real query would
    // project a narrow set of columns, opening the dataset must read the
    // full manifest first.
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

    // The reader's next version refresh re-reads the full manifest even
    // though its projection (and the queried columns) did not change.
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
