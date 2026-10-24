// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reproduction for a scan-scheduler I/O-buffer deadlock.
//!
//! Theory (from https://github.com/lance-format/lance/pull/9576#issuecomment-6265846131):
//! every data file in a fragment is opened with the same base read priority, so
//! only a file's row offset separates its reads from its siblings'. When a
//! fragment has several data files (as happens when columns were added one at a
//! time via `add_columns`, each call appending one more file to every fragment),
//! `MergeStream` polls those files in a fixed order and returns `Pending` as soon
//! as the first one isn't ready, without polling any sibling behind it. If a
//! later sibling's read has already completed but is never polled, its admitted
//! response is never dropped: the I/O scheduler's byte-budget reservation stays
//! pinned on that response, the budget never frees up, and the earlier file's
//! next page can never be admitted. Nothing can make progress.
//!
//! This builds a dataset entirely through public `lance::Dataset` APIs: 10
//! fragments, each made of 10 data files (one base write plus nine
//! `add_columns` calls — the first adding a "wide" column that needs several
//! pages to read, the rest adding small "narrow" columns that each fit in a
//! single page), then compacts the dataset with a very small `io_buffer_size`.
//! The compaction is wrapped in a timeout so a reproduced deadlock fails the
//! test quickly instead of hanging the run.

use std::sync::Arc;
use std::time::Duration;

use arrow_array::{Int64Array, LargeBinaryArray, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use lance::Dataset;
use lance::dataset::optimize::{CompactionOptions, compact_files};
use lance::dataset::{BatchUDF, NewColumnTransform, WriteParams};
use lance_core::utils::tempfile::TempStrDir;

const NUM_FRAGMENTS: i64 = 10;
const ROWS_PER_FRAGMENT: i64 = 600;
// One base write + this many `add_columns` calls == 10 data files per fragment.
const NUM_ADDED_COLUMNS: i64 = 9;
// Deliberately not a divisor of ROWS_PER_FRAGMENT, so each added column's
// row-group boundaries land at row offsets that don't line up with the other
// files in the fragment. That misalignment is what lets `MergeStream` stop at
// a `Pending` sibling while another sibling's already-finished read sits
// unconsumed, per the linked comment's trace.
const ADD_COLUMNS_BATCH_SIZE: u32 = 257;
// Large enough to force the first added column's file into multiple pages;
// the deadlock needs at least one file whose decode can block mid-fragment.
const WIDE_VALUE_SIZE: usize = 16 * 1024;
// Small enough that each of these files reads in a single page.
const NARROW_VALUE_SIZE: usize = 256;
// Small enough to force the budget contention described above; io_buffer_size
// of 0 disables buffering entirely and is known (per the linked comment) to
// avoid the bug.
const TINY_IO_BUFFER_SIZE: u64 = 64 * 1024;

fn blob_value(seed: i64, size: usize) -> Vec<u8> {
    (0..size)
        .map(|i| (seed.wrapping_add(i as i64)) as u8)
        .collect()
}

fn add_blob_column(column_name: &str, value_size: usize) -> NewColumnTransform {
    let output_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        column_name,
        DataType::LargeBinary,
        false,
    )]));
    let mapper_schema = output_schema.clone();
    let mapper = move |batch: &RecordBatch| {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let values: Vec<Vec<u8>> = ids
            .values()
            .iter()
            .map(|id| blob_value(*id, value_size))
            .collect();
        let array = LargeBinaryArray::from_iter_values(values.iter().map(|v| v.as_slice()));
        Ok(RecordBatch::try_new(
            mapper_schema.clone(),
            vec![Arc::new(array)],
        )?)
    };
    NewColumnTransform::BatchUDF(BatchUDF {
        mapper: Box::new(mapper),
        output_schema,
        result_checkpoint: None,
    })
}

fn make_base_batch(start_row: i64, num_rows: i64) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("base0", DataType::LargeBinary, false),
    ]));
    let ids = Int64Array::from_iter_values(start_row..start_row + num_rows);
    let base: Vec<Vec<u8>> = (start_row..start_row + num_rows)
        .map(|i| blob_value(i, NARROW_VALUE_SIZE))
        .collect();
    let base = LargeBinaryArray::from_iter_values(base.iter().map(|v| v.as_slice()));
    RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(base)]).unwrap()
}

/// Builds a dataset with `NUM_FRAGMENTS` fragments, each containing a base
/// data file plus `NUM_ADDED_COLUMNS` files added one column at a time:
/// `num_wide_columns` "wide" columns first (several pages per file, each
/// with genuinely ongoing, not-yet-loaded I/O work across several merge
/// rounds), then narrow columns that each fit in a single page (nothing left
/// to schedule after their first read).
async fn make_fragmented_dataset(test_uri: &str, num_wide_columns: i64) -> Dataset {
    assert!((1..=NUM_ADDED_COLUMNS).contains(&num_wide_columns));
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("base0", DataType::LargeBinary, false),
    ]));
    let batches = (0..NUM_FRAGMENTS)
        .map(|frag| Ok(make_base_batch(frag * ROWS_PER_FRAGMENT, ROWS_PER_FRAGMENT)))
        .collect::<Vec<_>>();
    let reader = RecordBatchIterator::new(batches, schema);
    let write_params = WriteParams {
        max_rows_per_file: ROWS_PER_FRAGMENT as usize,
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
        .await
        .unwrap();
    assert_eq!(dataset.get_fragments().len(), NUM_FRAGMENTS as usize);

    for wide_idx in 0..num_wide_columns {
        let column_name = format!("wide{wide_idx}");
        dataset
            .add_columns(
                add_blob_column(&column_name, WIDE_VALUE_SIZE),
                Some(vec!["id".to_string()]),
                Some(ADD_COLUMNS_BATCH_SIZE),
            )
            .await
            .unwrap();
    }

    for col_idx in 0..(NUM_ADDED_COLUMNS - num_wide_columns) {
        let column_name = format!("narrow{col_idx}");
        dataset
            .add_columns(
                add_blob_column(&column_name, NARROW_VALUE_SIZE),
                Some(vec!["id".to_string()]),
                Some(ADD_COLUMNS_BATCH_SIZE),
            )
            .await
            .unwrap();
    }

    dataset.validate().await.unwrap();
    for fragment in dataset.get_fragments() {
        assert_eq!(
            fragment.metadata().files.len(),
            1 + NUM_ADDED_COLUMNS as usize,
            "expected one base file plus one file per add_columns call"
        );
    }
    dataset
}

/// Runs compaction with a tiny `io_buffer_size` against a dataset built by
/// [`make_fragmented_dataset`], panicking with a description of the deadlock
/// if it times out, and otherwise verifying the result.
async fn compact_and_check_for_deadlock(mut dataset: Dataset, num_wide_columns: i64) {
    let total_rows = (NUM_FRAGMENTS * ROWS_PER_FRAGMENT) as usize;

    let options = CompactionOptions {
        // Big enough that every fragment is a compaction candidate, so the
        // whole dataset gets merged into a single new fragment and the
        // per-fragment, multi-file merge path in the scan scheduler is
        // actually exercised.
        target_rows_per_fragment: total_rows * 2,
        io_buffer_size: Some(TINY_IO_BUFFER_SIZE),
        ..Default::default()
    };

    // The deadlock, if reproduced, is a permanent park with nothing on the
    // wire, so a generous-but-bounded timeout turns a would-be CI hang into a
    // clear, fast test failure instead.
    let result = tokio::time::timeout(
        Duration::from_secs(60),
        compact_files(&mut dataset, options, None),
    )
    .await;

    let metrics = match result {
        Err(_) => panic!(
            "compact_files deadlocked with a tiny io_buffer_size on a dataset with \
             multi-file fragments (one base write + {NUM_ADDED_COLUMNS} add_columns calls \
             per fragment, {num_wide_columns} of them adding a \"wide\" multi-page column). \
             This reproduces the scan-scheduler I/O-buffer deadlock described in \
             https://github.com/lance-format/lance/pull/9576#issuecomment-6265846131: \
             MergeStream stops polling at the first file that returns Pending and never \
             reaches a sibling file whose already-completed read is holding the io \
             scheduler's byte-budget reservation, so the budget can never free up."
        ),
        Ok(result) => result.unwrap(),
    };
    assert_eq!(metrics.fragments_removed, NUM_FRAGMENTS as usize);

    // Sanity check: once compaction completes, all rows and columns must
    // still be there and correct.
    let scanner = dataset.scan();
    let batches = scanner
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let scanned_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(scanned_rows, total_rows);
}

#[tokio::test]
async fn test_compact_multi_file_fragments_with_tiny_io_buffer_does_not_deadlock() {
    let test_dir = TempStrDir::default();
    let test_uri: &str = &test_dir;
    let dataset = make_fragmented_dataset(test_uri, 1).await;
    Box::pin(compact_and_check_for_deadlock(dataset, 1)).await;
}

/// Same shape, but with *two* multi-page "wide" columns per fragment instead
/// of one. This stresses the case where more than one file in the merge has
/// genuinely ongoing (not-yet-loaded) I/O work at the same time, rather than
/// one multi-page file alongside several already-fully-loaded single-page
/// ones. See the discussion of whether a stream sitting on leftover rows
/// could still stall a sibling's forward progress.
#[tokio::test]
async fn test_compact_multi_file_fragments_with_two_wide_columns_does_not_deadlock() {
    let test_dir = TempStrDir::default();
    let test_uri: &str = &test_dir;
    let dataset = make_fragmented_dataset(test_uri, 2).await;
    Box::pin(compact_and_check_for_deadlock(dataset, 2)).await;
}
