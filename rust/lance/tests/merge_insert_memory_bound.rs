// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Regression test: a small partial-schema upsert into a table of wide rows
//! must run under a memory pool far smaller than the table, on every merge
//! path, without holding the table in memory.
//!
//! The merge plan hashes one side of a source/target join. Building that hash
//! table from the target decoded the whole table per merge, outside any pool:
//! a table with few rows of several MB each looked small to DataFusion's
//! row-count heuristics, and the merge session had an unbounded pool. The
//! scalar-index path scanned the fragments its index does not cover with the
//! scanner's defaults and coalesced the batches outside the pool, which for
//! an index covering nothing is the table again. The plan now builds on the
//! source under `LANCE_MEM_POOL_SIZE`, and both target scans run under a
//! share of that pool.
//!
//! This file must stay a single-test integration binary: the merge reads the
//! pool size from the process-global `LANCE_MEM_POOL_SIZE`, so a sibling test
//! could race the `set_var` or inherit the tiny pool.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicIsize, Ordering};

use arrow_array::types::UInt32Type;
use arrow_array::{Array, BinaryArray, RecordBatch, RecordBatchIterator, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lance::Dataset;
use lance::dataset::{
    MergeInsertBuilder, MergeInsertWriteMode, WhenMatched, WhenNotMatched, WriteParams,
};
use lance::index::DatasetIndexExt;
use lance_datafusion::exec::LanceExecutionOptions;
use lance_datagen::{BatchCount, ByteCount, RowCount, array, gen_batch};
use lance_index::IndexType;
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};

/// Counts live heap bytes so a merge's peak can be asserted, not just the
/// pool's view of it: the target scan is not pool-tracked.
struct CountingAllocator;

static LIVE_BYTES: AtomicIsize = AtomicIsize::new(0);
static PEAK_BYTES_SEEN: AtomicIsize = AtomicIsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let live = LIVE_BYTES.fetch_add(layout.size() as isize, Ordering::Relaxed)
            + layout.size() as isize;
        PEAK_BYTES_SEEN.fetch_max(live, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

const MEM_POOL_SIZE: u64 = 48 * 1024 * 1024;
const ROW_BYTES: u64 = 1024 * 1024;
const ROWS_PER_FRAGMENT: usize = 8;
/// One fragment is written before the index, the rest appended after it.
const FRAGMENTS: u32 = 25;
const ROWS: usize = ROWS_PER_FRAGMENT * FRAGMENTS as usize;
/// Heap growth allowed during a merge; the table is 200 MiB.
const PEAK_BYTES: isize = 128 * 1024 * 1024;
const NEW_BYTE: u8 = 0x42;

fn rows(first_key: u32, fragments: u32) -> impl arrow_array::RecordBatchReader {
    gen_batch()
        .col("key", array::step_custom::<UInt32Type>(first_key, 1))
        .col(
            "payload",
            array::rand_varbin(ByteCount::from(ROW_BYTES), ByteCount::from(ROW_BYTES)),
        )
        .col("tag", array::fill_utf8("old".to_string()))
        .into_reader_rows(
            RowCount::from(ROWS_PER_FRAGMENT as u64),
            BatchCount::from(fragments),
        )
}

async fn create_key_index(dataset: &mut Dataset) {
    dataset
        .create_index(
            &["key"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
            true,
        )
        .await
        .unwrap();
}

/// Upserts a new `payload` for four keys of one appended fragment, so the
/// target side of the merge carries the wide column while the column patch
/// touches one fragment; `tag` stays on the target. Returns the peak heap
/// growth during the merge.
async fn upsert(dataset: Arc<Dataset>, write_mode: MergeInsertWriteMode) -> (Arc<Dataset>, isize) {
    let payload = vec![NEW_BYTE; ROW_BYTES as usize];
    let source = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("key", DataType::UInt32, false),
            Field::new("payload", DataType::Binary, true),
        ])),
        vec![
            Arc::new(UInt32Array::from(vec![40, 41, 42, 43])),
            Arc::new(BinaryArray::from_vec(vec![&payload; 4])),
        ],
    )
    .unwrap();
    let job = MergeInsertBuilder::try_new(dataset, vec!["key".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(write_mode)
        .try_build()
        .unwrap();
    let source = RecordBatchIterator::new(vec![Ok(source.clone())], source.schema());
    let live_before = LIVE_BYTES.load(Ordering::Relaxed);
    PEAK_BYTES_SEEN.store(live_before, Ordering::Relaxed);
    let (dataset, stats) = job.execute_reader(source).await.unwrap();
    let peak = PEAK_BYTES_SEEN.load(Ordering::Relaxed) - live_before;
    assert_eq!(stats.num_updated_rows, 4);
    assert_eq!(stats.num_inserted_rows, 0);
    (dataset, peak)
}

async fn check(dataset: &Dataset) {
    let batches: Vec<RecordBatch> = dataset
        .scan()
        .project(&["payload", "tag"])
        .unwrap()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    let mut rows = 0;
    let mut replaced = 0;
    for batch in batches {
        let payload = batch["payload"]
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        let tag = batch["tag"].as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..batch.num_rows() {
            assert_eq!(payload.value(i).len() as u64, ROW_BYTES);
            assert_eq!(tag.value(i), "old");
            if payload.value(i).iter().all(|byte| *byte == NEW_BYTE) {
                replaced += 1;
            }
        }
        rows += batch.num_rows();
    }
    assert_eq!(rows, ROWS);
    assert_eq!(replaced, 4);
}

#[tokio::test]
async fn test_partial_upsert_into_wide_rows_fits_pool() {
    unsafe {
        std::env::set_var("LANCE_MEM_POOL_SIZE", MEM_POOL_SIZE.to_string());
        std::env::remove_var("LANCE_BYPASS_SPILLING");
    }
    // The merge builds its execution options from the env vars above; fail
    // loudly if that plumbing changes, or the pool would be the default one.
    let options = LanceExecutionOptions {
        use_spilling: true,
        ..Default::default()
    };
    assert_eq!(options.mem_pool_size(), MEM_POOL_SIZE);
    assert!(options.use_spilling());

    // 200 rows x 1 MiB in 25 fragments: the table is four times the pool, the
    // source 4 rows. The btree is built on the first fragment only, so it
    // covers none of the rows the merge touches.
    let write_params = WriteParams {
        max_rows_per_file: ROWS_PER_FRAGMENT,
        ..Default::default()
    };
    let mut dataset = Dataset::write(rows(0, 1), "memory://", Some(write_params.clone()))
        .await
        .unwrap();
    create_key_index(&mut dataset).await;
    dataset
        .append(
            rows(ROWS_PER_FRAGMENT as u32, FRAGMENTS - 1),
            Some(write_params),
        )
        .await
        .unwrap();
    assert_eq!(dataset.get_fragments().len(), FRAGMENTS as usize);

    // Indexed key, index covering nothing: the scalar-index path scans every
    // appended fragment and patches the column.
    let (dataset, peak) = upsert(Arc::new(dataset), MergeInsertWriteMode::Auto).await;
    check(&dataset).await;
    assert!(peak < PEAK_BYTES, "index path held {peak} bytes");

    // Row rewrite: the plan path, which fills `payload` from the target side
    // of the join.
    let (dataset, peak) = upsert(dataset, MergeInsertWriteMode::RewriteRows).await;
    check(&dataset).await;
    assert!(peak < PEAK_BYTES, "plan path held {peak} bytes");

    // Column patch on the plan path: the target scan carries only the key.
    let mut dataset = Arc::unwrap_or_clone(dataset);
    dataset.drop_index("key_idx").await.unwrap();
    let (dataset, _) = upsert(Arc::new(dataset), MergeInsertWriteMode::RewriteColumns).await;
    check(&dataset).await;

    // Covering index: the take path.
    let mut dataset = Arc::unwrap_or_clone(dataset);
    create_key_index(&mut dataset).await;
    let (dataset, _) = upsert(Arc::new(dataset), MergeInsertWriteMode::Auto).await;
    check(&dataset).await;
}
