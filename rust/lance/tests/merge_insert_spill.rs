// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#![recursion_limit = "256"]

//! Regression test for spilling an indexed merge-insert under a bounded pool.
//!
//! This file must stay a single-test integration binary because the indexed
//! merge path reads its pool size from the process-global `LANCE_MEM_POOL_SIZE`.

use std::sync::Arc;

use lance::Dataset;
use lance::dataset::MergeInsertBuilder;
use lance::dataset::write::merge_insert::{WhenMatched, WhenNotMatched};
use lance::index::DatasetIndexExt;
use lance_datagen::{BatchCount, ByteCount, RowCount, Seed, array, gen_batch};
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

#[tokio::test]
async fn test_indexed_merge_insert_spills_wide_source() {
    let target = gen_batch()
        .with_seed(Seed::from(1))
        .col("id", array::step::<arrow_array::types::UInt64Type>())
        .col("payload", array::rand_utf8(ByteCount::from(2048), false))
        .col("updated", array::fill::<arrow_array::types::UInt32Type>(0))
        .into_reader_rows(RowCount::from(NUM_ROWS), BatchCount::from(1));

    let mut dataset = Dataset::write(target, "memory://", None).await.unwrap();
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
    let source: Box<dyn arrow_array::RecordBatchReader + Send> = Box::new(
        gen_batch()
            .with_seed(Seed::from(2))
            .col("id", array::step::<arrow_array::types::UInt64Type>())
            .col("payload", array::rand_utf8(ByteCount::from(2048), false))
            .col("updated", array::fill::<arrow_array::types::UInt32Type>(1))
            .into_reader_rows(RowCount::from(NUM_ROWS), BatchCount::from(1)),
    );

    let (dataset, stats) = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .try_build()
        .unwrap()
        .execute_reader(source)
        .await
        .unwrap();

    assert_eq!(stats.num_updated_rows, NUM_ROWS);
    assert_eq!(stats.num_inserted_rows, 0);
    assert_eq!(
        dataset
            .count_rows(Some("updated = 1".to_string()))
            .await
            .unwrap(),
        NUM_ROWS as usize
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
}
