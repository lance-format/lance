// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Regression test: a small partial-schema upsert into a table of wide rows
//! must run under a memory pool far smaller than the table, through both the
//! non-indexed plan and the covering-index take path, without holding the
//! table in memory.
//!
//! The merge plan hashes one side of a source/target join. Building that hash
//! table from the target decoded the whole table per merge, outside any pool:
//! a table with few rows of several MB each looked small to DataFusion's
//! row-count heuristics, and the merge session had an unbounded pool. The
//! indexed path full-joined the target rows it took, emitting the whole target
//! scan through the join. The plan now builds on the source under
//! `LANCE_MEM_POOL_SIZE`, and both target scans are budgeted from that pool.
//!
//! This file must stay a single-test integration binary: the merge reads the
//! pool size from the process-global `LANCE_MEM_POOL_SIZE`, so a sibling test
//! could race the `set_var` or inherit the tiny pool.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicU64, Ordering};

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

async fn tag_counts(dataset: &Dataset) -> (usize, usize) {
    let batch = dataset
        .scan()
        .project(&["tag"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let tags = batch["tag"].as_any().downcast_ref::<StringArray>().unwrap();
    tags.iter().fold((0, 0), |(new, old), tag| match tag {
        Some("new") => (new + 1, old),
        Some("old") => (new, old + 1),
        other => panic!("unexpected tag {other:?}"),
    })
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
    // source 4 rows. The btree is created after the plan-path checks and covers
    // every fragment for the final take-path check.
    let write_params = WriteParams {
        max_rows_per_file: ROWS_PER_FRAGMENT,
        ..Default::default()
    };
    let mut dataset = Dataset::write(rows(0, 1), "memory://", Some(write_params.clone()))
        .await
        .unwrap();
    dataset
        .append(
            rows(ROWS_PER_FRAGMENT as u32, FRAGMENTS - 1),
            Some(write_params),
        )
        .await
        .unwrap();
    assert_eq!(dataset.get_fragments().len(), FRAGMENTS as usize);

    // Row rewrite: the plan path, which fills `payload` from the target side
    // of the join.
    let (dataset, peak) = upsert(Arc::new(dataset), MergeInsertWriteMode::RewriteRows).await;
    check(&dataset).await;
    assert!(peak < PEAK_BYTES, "plan path held {peak} bytes");

    // Column patch on the plan path: the target scan carries only the key.
    let (dataset, peak) = upsert(dataset, MergeInsertWriteMode::RewriteColumns).await;
    check(&dataset).await;
    assert!(
        peak < PEAK_BYTES,
        "RewriteColumns plan path held {peak} bytes"
    );

    // Covering index: the take path.
    let mut dataset = Arc::unwrap_or_clone(dataset);
    create_key_index(&mut dataset).await;
    let (dataset, peak) = upsert(Arc::new(dataset), MergeInsertWriteMode::Auto).await;
    check(&dataset).await;
    assert!(
        peak < PEAK_BYTES,
        "covering-index take path held {peak} bytes"
    );

    // The source can be larger than both the target and the pool. Building the
    // source would fail even though the small target fits comfortably.
    unsafe {
        std::env::set_var("LANCE_MEM_POOL_SIZE", (16 * 1024 * 1024).to_string());
    }
    let wide_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("payload", DataType::Binary, true),
        Field::new("tag", DataType::Utf8, false),
    ]));
    let source_payload = vec![NEW_BYTE; ROW_BYTES as usize];
    let source_rows = 64;
    let source = RecordBatch::try_new(
        wide_schema.clone(),
        vec![
            Arc::new(UInt32Array::from_iter_values(0..source_rows)),
            Arc::new(BinaryArray::from(vec![
                source_payload.as_slice();
                source_rows as usize
            ])),
            Arc::new(StringArray::from(vec!["new"; source_rows as usize])),
        ],
    )
    .unwrap();
    for (target_rows, spill_for_retry) in [(1, true), (4, true), (4, false)] {
        let target = gen_batch()
            .col("key", array::step::<UInt32Type>())
            .col(
                "payload",
                array::rand_varbin(ByteCount::from(ROW_BYTES), ByteCount::from(ROW_BYTES)),
            )
            .col("tag", array::fill_utf8("old".to_string()))
            .into_reader_rows(RowCount::from(target_rows), BatchCount::from(1));
        let dataset = Arc::new(Dataset::write(target, "memory://", None).await.unwrap());
        let target_file_bytes: u64 = dataset
            .get_fragments()
            .iter()
            .flat_map(|fragment| fragment.metadata().files.iter())
            .map(|file| file.file_size_bytes.get().unwrap().get())
            .sum();
        assert_eq!(
            target_file_bytes > 2 * 1024 * 1024,
            target_rows == 4,
            "the four-row target must cross the old fixed cutoff"
        );

        let version_before = dataset.version().version;
        let mut builder = MergeInsertBuilder::try_new(dataset, vec!["key".to_string()]).unwrap();
        builder
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing)
            .use_index(false);
        builder.spill_for_retry(spill_for_retry);
        let (dataset, stats) = builder
            .try_build()
            .unwrap()
            .execute_reader(RecordBatchIterator::new(
                vec![Ok(source.clone())],
                source.schema(),
            ))
            .await
            .unwrap();
        assert_eq!(stats.num_updated_rows, target_rows);
        assert_eq!(
            dataset.count_rows(None).await.unwrap(),
            target_rows as usize
        );
        assert_eq!(dataset.version().version, version_before + 1);
        assert_eq!(tag_counts(&dataset).await, (target_rows as usize, 0));
    }

    // Repeated payloads make the encoded target smaller than its decoded Arrow
    // batches. The file size alone must not send this source through the target
    // hash-table build.
    let compressed_rows = 32;
    let compressed_target = RecordBatch::try_new(
        wide_schema.clone(),
        vec![
            Arc::new(UInt32Array::from_iter_values(0..compressed_rows)),
            Arc::new(BinaryArray::from(vec![
                source_payload.as_slice();
                compressed_rows as usize
            ])),
            Arc::new(StringArray::from(vec!["old"; compressed_rows as usize])),
        ],
    )
    .unwrap();
    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(compressed_target)], wide_schema.clone()),
            "memory://",
            None,
        )
        .await
        .unwrap(),
    );
    let encoded_bytes: u64 = dataset
        .get_fragments()
        .iter()
        .flat_map(|fragment| fragment.metadata().files.iter())
        .map(|file| file.file_size_bytes.get().unwrap().get())
        .sum();
    assert!(
        encoded_bytes <= 2 * 1024 * 1024,
        "fixture must compress below the old cutoff, got {encoded_bytes}"
    );
    let compressed_source_rows = 2;
    let compressed_source = RecordBatch::try_new(
        wide_schema.clone(),
        vec![
            Arc::new(UInt32Array::from_iter_values(0..compressed_source_rows)),
            Arc::new(BinaryArray::from(vec![
                source_payload.as_slice();
                compressed_source_rows as usize
            ])),
            Arc::new(StringArray::from(vec![
                "new";
                compressed_source_rows as usize
            ])),
        ],
    )
    .unwrap();
    let version_before = dataset.version().version;
    let mut builder = MergeInsertBuilder::try_new(dataset, vec!["key".to_string()]).unwrap();
    builder
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .use_index(false);
    let (dataset, stats) = builder
        .try_build()
        .unwrap()
        .execute_reader(RecordBatchIterator::new(
            vec![Ok(compressed_source.clone())],
            compressed_source.schema(),
        ))
        .await
        .unwrap();
    assert_eq!(stats.num_updated_rows, compressed_source_rows as u64);
    assert_eq!(
        dataset.count_rows(None).await.unwrap(),
        compressed_rows as usize
    );
    assert_eq!(dataset.version().version, version_before + 1);
    assert_eq!(
        tag_counts(&dataset).await,
        (
            compressed_source_rows as usize,
            (compressed_rows - compressed_source_rows) as usize
        )
    );

    // Source replay and patch staging must share one operation-scoped disk
    // quota across the source-build attempt and its target-build fallback.
    fn disk_bytes(path: &std::path::Path) -> u64 {
        std::fs::read_dir(path)
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| {
                let path = entry.path();
                if path.is_dir() {
                    disk_bytes(&path)
                } else {
                    entry.metadata().map(|metadata| metadata.len()).unwrap_or(0)
                }
            })
            .sum()
    }

    let scratch = tempfile::tempdir().unwrap();
    unsafe {
        std::env::set_var("TMPDIR", scratch.path());
        std::env::set_var("LANCE_MEM_POOL_SIZE", (512 * 1024 * 1024).to_string());
        std::env::set_var(
            "LANCE_MAX_TEMP_DIRECTORY_SIZE",
            (160 * 1024 * 1024).to_string(),
        );
        std::env::remove_var("LANCE_BYPASS_SPILLING");
    }
    let quota_rows = 112usize;
    let target = RecordBatch::try_new(
        wide_schema.clone(),
        vec![
            Arc::new(UInt32Array::from_iter_values(0..quota_rows as u32)),
            Arc::new(BinaryArray::from_vec(vec![&[]; quota_rows])),
            Arc::new(StringArray::from(vec!["old"; quota_rows])),
        ],
    )
    .unwrap();
    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(target)], wide_schema.clone()),
            "memory://",
            None,
        )
        .await
        .unwrap(),
    );
    let quota_source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("payload", DataType::Binary, true),
    ]));
    let source = RecordBatch::try_new(
        quota_source_schema.clone(),
        vec![
            Arc::new(UInt32Array::from_iter_values(0..quota_rows as u32)),
            Arc::new(BinaryArray::from(vec![
                source_payload.as_slice();
                quota_rows
            ])),
        ],
    )
    .unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let peak_spill_bytes = Arc::new(AtomicU64::new(0));
    let monitor = {
        let stop = stop.clone();
        let peak_spill_bytes = peak_spill_bytes.clone();
        let path = scratch.path().to_owned();
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                peak_spill_bytes.fetch_max(disk_bytes(&path), Ordering::Relaxed);
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        })
    };
    let result = MergeInsertBuilder::try_new(dataset, vec!["key".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .use_index(false)
        .try_build()
        .unwrap()
        .execute_reader(RecordBatchIterator::new(
            vec![Ok(source)],
            quota_source_schema,
        ))
        .await;
    stop.store(true, Ordering::Relaxed);
    monitor.join().unwrap();

    match result {
        Ok((dataset, stats)) => panic!(
            "merge succeeded despite replay plus patch staging exceeding 160 MiB: \
             version={}, updated={}, peak_spill_bytes={}",
            dataset.version().version,
            stats.num_updated_rows,
            peak_spill_bytes.load(Ordering::Relaxed)
        ),
        Err(error) => assert!(
            error.to_string().contains("allowable limit"),
            "unexpected error: {error}"
        ),
    }
}
