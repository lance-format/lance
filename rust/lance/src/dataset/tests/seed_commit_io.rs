// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Seed flag activation must not add object-store reads to ordinary commits:
//! a table whose ZoneMap index requests seeds issues the same read and write
//! requests per append, update and merge-insert as a table without indices.

use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use lance_core::utils::tempfile::TempStrDir;
use lance_index::IndexType;
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};

use crate::Dataset;
use crate::dataset::write::merge_insert::{WhenMatched, WhenNotMatched};
use crate::dataset::{MergeInsertBuilder, UpdateBuilder, WriteParams};
use crate::index::DatasetIndexExt;

fn schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new("val", DataType::Int32, true),
    ]))
}

fn rows(ids: std::ops::Range<i32>) -> RecordBatch {
    let values: Vec<Option<i32>> = ids.clone().map(|i| Some(i * 10)).collect();
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int32Array::from_iter_values(ids)),
            Arc::new(Int32Array::from(values)),
        ],
    )
    .unwrap()
}

/// (read_iops, write_iops) of one append, one update and one merge-insert.
async fn commit_requests(dataset: &mut Dataset) -> [(u64, u64); 3] {
    let mut out = [(0, 0); 3];
    // The first write of a process reads each seeded zone map's
    // null-position policy once (cached afterwards); measure steady state.
    dataset
        .append(
            RecordBatchIterator::new([Ok(rows(90..100))], schema()),
            None,
        )
        .await
        .unwrap();
    dataset.object_store.io_stats_incremental();
    dataset
        .append(
            RecordBatchIterator::new([Ok(rows(100..110))], schema()),
            None,
        )
        .await
        .unwrap();
    let io = dataset.object_store.io_stats_incremental();
    out[0] = (io.read_iops, io.write_iops);

    let updated = UpdateBuilder::new(Arc::new(dataset.clone()))
        .update_where("id = 1")
        .unwrap()
        .set("val", "val + 1")
        .unwrap()
        .build()
        .unwrap()
        .execute()
        .await
        .unwrap();
    *dataset = updated.new_dataset.as_ref().clone();
    let io = dataset.object_store.io_stats_incremental();
    out[1] = (io.read_iops, io.write_iops);

    let (merged, _) =
        MergeInsertBuilder::try_new(Arc::new(dataset.clone()), vec!["id".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .try_build()
            .unwrap()
            .execute_reader(RecordBatchIterator::new([Ok(rows(5..15))], schema()))
            .await
            .unwrap();
    *dataset = merged.as_ref().clone();
    let io = dataset.object_store.io_stats_incremental();
    out[2] = (io.read_iops, io.write_iops);
    out
}

async fn fresh_dataset(dir: &TempStrDir) -> Dataset {
    Dataset::write(
        RecordBatchIterator::new([Ok(rows(0..10))], schema()),
        dir.as_str(),
        Some(WriteParams::default()),
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn test_seed_activation_adds_no_commit_requests() {
    let dir = TempStrDir::default();
    let mut plain = fresh_dataset(&dir).await;
    let without_index = commit_requests(&mut plain).await;

    let dir = TempStrDir::default();
    let mut seeded = fresh_dataset(&dir).await;
    seeded
        .create_index(
            &["val"],
            IndexType::ZoneMap,
            None,
            &ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap)
                .with_params(&serde_json::json!({"rows_per_zone": 4, "use_seeds": true})),
            false,
        )
        .await
        .unwrap();
    let with_seeded_index = commit_requests(&mut seeded).await;

    assert_eq!(
        with_seeded_index, without_index,
        "(read_iops, write_iops) per append / update / merge-insert must not change \
         when a seed-requesting index exists"
    );
}
