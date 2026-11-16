// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! An invalid `LANCE_MEMORY_OBSERVE` fails every observed operation. Lance
//! reads the switch once per process, so the operations run in a child
//! process.

use std::process::Command;
use std::sync::Arc;

use arrow_array::RecordBatchReader;
use arrow_array::types::Int64Type;
use lance_core::utils::tempfile::TempStrDir;
use lance_datagen::{BatchCount, RowCount, array, gen_batch};
use lance_index::IndexType;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::ScalarIndexParams;
use uuid::Uuid;

use crate::dataset::fragment::FileFragment;
use crate::dataset::mem_wal::{DatasetMemWalExt, ShardWriterConfig};
use crate::dataset::optimize::{CompactionOptions, compact_files, plan_compaction};
use crate::dataset::{
    DeleteBuilder, InsertBuilder, MergeInsertBuilder, UpdateBuilder, WhenMatched, WriteMode,
    WriteParams,
};
use crate::index::DatasetIndexExt;
use crate::index::optimize::plan_optimize_indices;
use crate::{Dataset, Error};

const CHILD_URI: &str = "LANCE_INVALID_MEMORY_SWITCH_TEST_URI";
const CHILD_DONE: &str = "every observed operation refused the invalid switch";

fn batches() -> impl arrow_array::RecordBatchReader + Send + 'static {
    gen_batch()
        .col("id", array::step::<Int64Type>())
        .into_reader_rows(RowCount::from(8), BatchCount::from(2))
}

#[tokio::test]
async fn test_an_invalid_memory_switch_fails_observed_operations() {
    let dir = TempStrDir::default();
    let params = WriteParams {
        max_rows_per_file: 8,
        ..Default::default()
    };
    let mut dataset = Dataset::write(batches(), &dir, Some(params)).await.unwrap();
    dataset
        .initialize_mem_wal()
        .unsharded()
        .execute()
        .await
        .unwrap();
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "dataset::tests::dataset_memory_switch::test_an_invalid_memory_switch_fails_observed_operations_child",
            "--nocapture",
        ])
        .env(CHILD_URI, &*dir)
        .env("LANCE_MEMORY_OBSERVE", "maybe")
        .output()
        .unwrap();
    assert!(output.status.success(), "child process failed: {output:?}");
    assert!(
        String::from_utf8_lossy(&output.stdout).contains(CHILD_DONE),
        "the child test did not run: {output:?}"
    );
}

#[tokio::test]
#[allow(clippy::print_stdout)]
async fn test_an_invalid_memory_switch_fails_observed_operations_child() {
    let Ok(uri) = std::env::var(CHILD_URI) else {
        return;
    };
    let refused = |error: Error| {
        assert!(
            matches!(error, Error::InvalidInput { .. })
                && error.to_string().contains("LANCE_MEMORY_OBSERVE"),
            "{error:?}"
        );
    };
    let mut dataset = Dataset::open(&uri).await.unwrap();
    let btree = ScalarIndexParams::default();
    refused(
        dataset
            .create_index(&["id"], IndexType::BTree, None, &btree, false)
            .await
            .unwrap_err(),
    );
    refused(
        compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap_err(),
    );
    refused(
        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap_err(),
    );
    // The distributed pieces: planning a compaction is not observed, so it
    // still hands out the task that the switch then refuses.
    let plan = plan_compaction(&dataset, &CompactionOptions::default())
        .await
        .unwrap();
    let task = plan
        .compaction_tasks()
        .next()
        .expect("the two fragments make a compaction task");
    refused(task.execute(&dataset).await.unwrap_err());
    refused(
        plan_optimize_indices(&dataset, &OptimizeOptions::default())
            .await
            .unwrap_err(),
    );

    // Every entry point of a write, those that commit nothing included.
    refused(
        Dataset::write(batches(), "memory://refused", None)
            .await
            .unwrap_err(),
    );
    let target = Arc::new(dataset.clone());
    let batch = || batches().next().unwrap().unwrap();
    let source = || Box::new(batches()) as Box<dyn RecordBatchReader + Send>;
    let append = WriteParams {
        mode: WriteMode::Append,
        ..Default::default()
    };
    let insert = || InsertBuilder::new(target.clone()).with_params(&append);
    refused(insert().execute(vec![batch()]).await.unwrap_err());
    refused(
        insert()
            .execute_uncommitted(vec![batch()])
            .await
            .unwrap_err(),
    );
    refused(
        insert()
            .execute_uncommitted_stream(source())
            .await
            .unwrap_err(),
    );
    refused(
        FileFragment::create(&uri, 10, source(), Some(append.clone()))
            .await
            .unwrap_err(),
    );
    refused(
        FileFragment::create_fragments(&uri, source(), Some(append.clone()))
            .await
            .unwrap_err(),
    );
    refused(
        UpdateBuilder::new(target.clone())
            .update_where("id < 2")
            .unwrap()
            .set("id", "id + 100")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap_err(),
    );
    refused(
        DeleteBuilder::new(target.clone(), "id < 2")
            .execute()
            .await
            .unwrap_err(),
    );
    refused(
        DeleteBuilder::new(target.clone(), "id < 2")
            .execute_uncommitted()
            .await
            .unwrap_err(),
    );
    let upsert = || {
        MergeInsertBuilder::try_new(target.clone(), vec!["id".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .try_build()
            .unwrap()
    };
    refused(upsert().execute_reader(source()).await.unwrap_err());
    refused(upsert().execute_dataset(target.clone()).await.unwrap_err());
    refused(upsert().execute_batches(vec![batch()]).await.unwrap_err());
    // An uncommitted merge has no `Debug`, so `unwrap_err` cannot show it.
    refused(upsert().execute_uncommitted(source()).await.err().unwrap());
    refused(
        upsert()
            .execute_uncommitted_batches(vec![batch()])
            .await
            .err()
            .unwrap(),
    );
    refused(
        upsert()
            .execute_uncommitted_dataset(target.clone())
            .await
            .err()
            .unwrap(),
    );

    // A writer that flushes memtables is refused before it claims the shard,
    // so the shard keeps the epoch of the WAL-only writer holding it, which
    // flushes none and keeps writing. That writer only checks the epoch when
    // a WAL write collides, so the check below reads it.
    let shard_id = Uuid::new_v4();
    let wal_only = ShardWriterConfig {
        enable_memtable: false,
        durable_write: true,
        ..Default::default()
    };
    let holder = dataset.mem_wal_writer(shard_id, wal_only).await.unwrap();
    refused(
        dataset
            .mem_wal_writer(shard_id, ShardWriterConfig::default())
            .await
            .err()
            .expect("a writer with a memtable opened"),
    );
    holder
        .manifest_store()
        .check_fenced(holder.epoch())
        .await
        .unwrap();
    let batch = batches().next().unwrap().unwrap();
    holder.put(vec![batch]).await.unwrap();
    holder.close().await.unwrap();
    println!("{CHILD_DONE}");
}
