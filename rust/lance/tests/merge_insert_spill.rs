// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#![recursion_limit = "256"]

//! Regression test for spilling an indexed merge-insert under a bounded pool.
//!
//! This file must stay a single-test integration binary because the indexed
//! merge path reads its pool size from the process-global `LANCE_MEM_POOL_SIZE`.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{
    RecordBatch, RecordBatchIterator, RecordBatchReader,
    cast::AsArray,
    types::{Float32Type, UInt32Type, UInt64Type},
};
use lance::Dataset;
use lance::dataset::write::merge_insert::{WhenMatched, WhenNotMatched};
use lance::dataset::{MergeInsertBuilder, WriteParams};
use lance::index::DatasetIndexExt;
use lance_datagen::{BatchCount, ByteCount, Dimension, RowCount, Seed, array, gen_batch};
use lance_index::IndexType;
use lance_index::scalar::ScalarIndexParams;

const MEM_POOL_SIZE: u64 = 16 * 1024 * 1024;
const LARGE_ROW_MEM_POOL_SIZE: u64 = 512 * 1024 * 1024;
const LARGE_PAYLOAD_BYTES: u64 = 30 * 1024 * 1024;
const PARTIAL_UPDATE_MEM_POOL_SIZE: u64 = 150 * 1024 * 1024;
const PARTIAL_ROWS_PER_BATCH: u64 = 50_000;
const PARTIAL_BATCHES: u32 = 4;
const WIDE_KEY_BYTES: u64 = 2048;
const NUM_ROWS: u64 = 8192;
const SPILLED_SOURCE_MEM_POOL_SIZE: u64 = 8 * 1024 * 1024;
const SPILLED_SOURCE_ROWS: u64 = 1024;
// Vector dimensions of the rows below. A fixed width decodes to what the file holds, so the
// batches have the sizes these comments rely on.
//
// 8192 rows of this width are about 2.7 MiB, above the 2 MiB byte cap of a sort input in a
// 16 MiB pool, so the sort reserves 4 MiB for a capped slice.
const TAIL_DIM: u32 = 80;
const SPILLED_SOURCE_DIM: u32 = 1536;
// A sort reserves twice a batch. Twice this 3 MiB row is more than a third of a 16 MiB pool
// and less than two thirds.
const WIDE_ROW_DIM: u32 = 768 * 1024;

fn rows(first_id: u64, num_rows: u64, dim: u32, updated: u32) -> RecordBatch {
    gen_batch()
        .with_seed(Seed::from(first_id))
        .col("id", array::step_custom::<UInt64Type>(first_id, 1))
        .col(
            "payload",
            array::rand_vec::<Float32Type>(Dimension::from(dim)),
        )
        .col("updated", array::fill::<UInt32Type>(updated))
        .into_batch_rows(RowCount::from(num_rows))
        .unwrap()
}

fn reader(batches: Vec<RecordBatch>) -> Box<dyn RecordBatchReader + Send> {
    let schema = batches[0].schema();
    Box::new(RecordBatchIterator::new(
        batches.into_iter().map(Ok),
        schema,
    ))
}

fn ids(batch: &RecordBatch) -> &[u64] {
    batch["id"].as_primitive::<UInt64Type>().values()
}

/// Upserts `source` by `id` under `pool_bytes` and checks every row. The dataset has `indexed`
/// under an index on `id`, and `unindexed` in a fragment appended after the index build.
async fn upsert(
    pool_bytes: u64,
    indexed: RecordBatch,
    unindexed: Option<RecordBatch>,
    source: Vec<RecordBatch>,
) {
    let mut expected = ids(&indexed)
        .iter()
        .chain(unindexed.iter().flat_map(ids))
        .map(|id| (*id, 0))
        .collect::<BTreeMap<u64, u32>>();
    let (mut expected_updates, mut expected_inserts) = (0, 0);
    for id in source.iter().flat_map(ids) {
        match expected.insert(*id, 1) {
            Some(_) => expected_updates += 1,
            None => expected_inserts += 1,
        }
    }

    let mut dataset = Dataset::write(reader(vec![indexed]), "memory://", None)
        .await
        .unwrap();
    dataset
        .create_index(
            &["id"],
            IndexType::Scalar,
            None,
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    if let Some(unindexed) = unindexed {
        dataset.append(reader(vec![unindexed]), None).await.unwrap();
    }

    unsafe { std::env::set_var("LANCE_MEM_POOL_SIZE", pool_bytes.to_string()) };
    let (dataset, stats) = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap()
        .execute_reader(reader(source))
        .await
        .unwrap();

    assert_eq!(stats.num_updated_rows, expected_updates);
    assert_eq!(stats.num_inserted_rows, expected_inserts);
    let mut scan = dataset.scan();
    scan.project(&["id", "updated"]).unwrap();
    let merged = scan.try_into_batch().await.unwrap();
    let mut actual = ids(&merged)
        .iter()
        .copied()
        .zip(
            merged["updated"]
                .as_primitive::<UInt32Type>()
                .values()
                .iter()
                .copied(),
        )
        .collect::<Vec<_>>();
    actual.sort_unstable();
    assert_eq!(actual, expected.into_iter().collect::<Vec<_>>());
}

#[tokio::test]
async fn test_indexed_merge_insert_spills_wide_source() {
    let target = gen_batch()
        .with_seed(Seed::from(1))
        .col("id", array::step_custom::<UInt64Type>(NUM_ROWS, 1))
        .col("payload", array::rand_utf8(ByteCount::from(2048), false))
        .col("updated", array::fill::<arrow_array::types::UInt32Type>(0))
        .into_reader_rows(RowCount::from(NUM_ROWS), BatchCount::from(1));

    let mut dataset = Dataset::write(
        target,
        "memory://",
        Some(WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["id"],
            IndexType::Scalar,
            None,
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();

    unsafe {
        std::env::set_var("LANCE_MEM_POOL_SIZE", MEM_POOL_SIZE.to_string());
        std::env::remove_var("LANCE_BYPASS_SPILLING");
    }

    // The source is larger than the 16 MiB fair pool. The indexed path
    // must spill instead of collecting it in a non-spillable hash-join input.
    // Lower-key inserts precede every update in the join, so stable row-ID
    // preservation must also remain bounded while staging the write order.
    let source: Box<dyn arrow_array::RecordBatchReader + Send> = Box::new(
        gen_batch()
            .with_seed(Seed::from(2))
            .col("id", array::step::<arrow_array::types::UInt64Type>())
            .col("payload", array::rand_utf8(ByteCount::from(2048), false))
            .col("updated", array::fill::<arrow_array::types::UInt32Type>(1))
            .into_reader_rows(RowCount::from(NUM_ROWS * 2), BatchCount::from(1)),
    );

    let (dataset, stats) = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap()
        .execute_reader(source)
        .await
        .unwrap();

    assert_eq!(stats.num_updated_rows, NUM_ROWS);
    assert_eq!(stats.num_inserted_rows, NUM_ROWS);
    assert_eq!(
        dataset
            .count_rows(Some("updated = 1".to_string()))
            .await
            .unwrap(),
        (NUM_ROWS * 2) as usize
    );
    let original_rows = dataset
        .take_rows(&[0, NUM_ROWS - 1], dataset.schema().clone())
        .await
        .unwrap();
    assert_eq!(
        original_rows["id"].as_primitive::<UInt64Type>().values(),
        &[NUM_ROWS, NUM_ROWS * 2 - 1]
    );

    unsafe {
        std::env::set_var("LANCE_MEM_POOL_SIZE", LARGE_ROW_MEM_POOL_SIZE.to_string());
    }

    // The rechunk target is a batching threshold, not a row-size validity
    // limit. A single 30 MiB payload must remain valid when it fits the actual
    // execution pool.
    let large_target = gen_batch()
        .with_seed(Seed::from(3))
        .col("id", array::step::<arrow_array::types::UInt64Type>())
        .col(
            "payload",
            array::rand_utf8(ByteCount::from(LARGE_PAYLOAD_BYTES), false),
        )
        .col("updated", array::fill::<arrow_array::types::UInt32Type>(0))
        .into_reader_rows(RowCount::from(1), BatchCount::from(1));
    let mut large_dataset = Dataset::write(large_target, "memory://", None)
        .await
        .unwrap();
    large_dataset
        .create_index(
            &["id"],
            IndexType::Scalar,
            None,
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();

    let large_source: Box<dyn arrow_array::RecordBatchReader + Send> = Box::new(
        gen_batch()
            .with_seed(Seed::from(4))
            .col("id", array::step::<arrow_array::types::UInt64Type>())
            .col(
                "payload",
                array::rand_utf8(ByteCount::from(LARGE_PAYLOAD_BYTES), false),
            )
            .col("updated", array::fill::<arrow_array::types::UInt32Type>(1))
            .into_reader_rows(RowCount::from(1), BatchCount::from(1)),
    );
    let (large_dataset, large_stats) =
        MergeInsertBuilder::try_new(Arc::new(large_dataset), vec!["id".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing)
            .try_build()
            .unwrap()
            .execute_reader(large_source)
            .await
            .unwrap();

    assert_eq!(large_stats.num_updated_rows, 1);
    assert_eq!(
        large_dataset
            .count_rows(Some("updated = 1".to_string()))
            .await
            .unwrap(),
        1
    );

    // Partial updates add a downstream fragment sort after the indexed join.
    // Both wide-key join inputs spill here; their merge streams must be fully
    // released before the fragment sort starts using the same bounded pool.
    let partial_target = gen_batch()
        .with_seed(Seed::from(5))
        .col(
            "id",
            array::rand_utf8(ByteCount::from(WIDE_KEY_BYTES), false),
        )
        .col("payload", array::fill::<arrow_array::types::UInt32Type>(0))
        .col("updated", array::fill::<arrow_array::types::UInt32Type>(0))
        .into_reader_rows(
            RowCount::from(PARTIAL_ROWS_PER_BATCH),
            BatchCount::from(PARTIAL_BATCHES),
        );
    let mut partial_dataset = Dataset::write(partial_target, "memory://", None)
        .await
        .unwrap();
    partial_dataset
        .create_index(
            &["id"],
            IndexType::Scalar,
            None,
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();

    unsafe {
        std::env::set_var(
            "LANCE_MEM_POOL_SIZE",
            PARTIAL_UPDATE_MEM_POOL_SIZE.to_string(),
        );
    }

    let partial_source: Box<dyn arrow_array::RecordBatchReader + Send> = Box::new(
        gen_batch()
            .with_seed(Seed::from(5))
            .col(
                "id",
                array::rand_utf8(ByteCount::from(WIDE_KEY_BYTES), false),
            )
            .col("updated", array::fill::<arrow_array::types::UInt32Type>(1))
            .into_reader_rows(
                RowCount::from(PARTIAL_ROWS_PER_BATCH),
                BatchCount::from(PARTIAL_BATCHES),
            ),
    );
    let (partial_dataset, partial_stats) =
        MergeInsertBuilder::try_new(Arc::new(partial_dataset), vec!["id".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing)
            .try_build()
            .unwrap()
            .execute_reader(partial_source)
            .await
            .unwrap();

    let partial_total_rows = PARTIAL_ROWS_PER_BATCH * u64::from(PARTIAL_BATCHES);
    assert_eq!(partial_stats.num_updated_rows, partial_total_rows);
    assert_eq!(partial_stats.num_inserted_rows, 0);
    assert_eq!(
        partial_dataset
            .count_rows(Some("updated = 1".to_string()))
            .await
            .unwrap(),
        partial_total_rows as usize
    );

    // Rows appended after the index build reach the target sort through a scan, whose batches
    // hit the byte cap of the sort input. A capped batch fits the sort's share of the pool next
    // to one other spillable consumer, but not next to two.
    upsert(
        MEM_POOL_SIZE,
        rows(0, 1024, TAIL_DIM, 0),
        Some(rows(1024, NUM_ROWS, TAIL_DIM, 0)),
        vec![
            rows(512, 1024, TAIL_DIM, 1),
            rows(1024 + NUM_ROWS, 512, TAIL_DIM, 1),
        ],
    )
    .await;

    // A source spilled to several runs keeps a merge reservation for each run of its last merge
    // pass for as long as its stream is open. The target sort must not have to fit next to it.
    // The keys arrive out of order, so the join is right only if the source reaches it sorted.
    upsert(
        SPILLED_SOURCE_MEM_POOL_SIZE,
        rows(0, SPILLED_SOURCE_ROWS, SPILLED_SOURCE_DIM, 0),
        None,
        vec![
            rows(
                SPILLED_SOURCE_ROWS,
                SPILLED_SOURCE_ROWS,
                SPILLED_SOURCE_DIM,
                1,
            ),
            rows(0, SPILLED_SOURCE_ROWS, SPILLED_SOURCE_DIM, 1),
        ],
    )
    .await;

    // The byte cap cannot split a row, so a row of up to a quarter of the pool must fit on either
    // side of the join and in either arm of the target: a sort reserves twice the row, and
    // merging two runs of it takes twice that.
    upsert(
        MEM_POOL_SIZE,
        rows(0, 1, WIDE_ROW_DIM, 0),
        Some(rows(1, 1, WIDE_ROW_DIM, 0)),
        vec![rows(0, 2, WIDE_ROW_DIM, 1), rows(2, 1, WIDE_ROW_DIM, 1)],
    )
    .await;
}
