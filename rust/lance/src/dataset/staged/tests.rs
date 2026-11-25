// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::dataset::index::DatasetIndexRemapperOptions;
use crate::dataset::optimize::{
    CompactionOptions, commit_compaction, compact_files, plan_compaction,
};
use crate::dataset::write::WriteProgressFn;
use crate::dataset::{DeleteBuilder, MergeInsertBuilder, WhenMatched, WhenNotMatched};
use crate::index::DatasetIndexExt;
use crate::io::exec::filtered_read::FilteredReadExec;
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{ArrayRef, Int32Array, record_batch};
use datafusion::error::DataFusionError;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use lance_core::utils::tempfile::TempStrDir;
use lance_file::version::LanceFileVersion;
use lance_table::format::RowIdMeta;
use lance_table::rowids::{RowIdSequence, write_row_ids};
use rstest::rstest;

fn id_batch(ids: impl IntoIterator<Item = i32>) -> RecordBatch {
    let ids = Arc::new(Int32Array::from_iter_values(ids)) as ArrayRef;
    RecordBatch::try_from_iter([("id", ids)]).unwrap()
}

async fn ids(dataset: &Dataset) -> Vec<i32> {
    let batch = dataset.scan().try_into_batch().await.unwrap();
    let mut ids = batch["id"].as_primitive::<Int32Type>().values().to_vec();
    ids.sort_unstable();
    ids
}

fn fragment_ids(dataset: &Dataset) -> Vec<u64> {
    dataset
        .manifest
        .fragments
        .iter()
        .map(|fragment| fragment.id)
        .collect()
}

async fn table_with_staged_append(uri: &str) -> (Dataset, StagedTransaction) {
    let dataset = InsertBuilder::new(uri)
        .with_params(&WriteParams {
            max_rows_per_file: 2,
            ..Default::default()
        })
        .execute(vec![id_batch(0..4)])
        .await
        .unwrap();
    let mut tx = dataset.begin_transaction().await.unwrap();
    tx.append(vec![id_batch([10, 11])]).await.unwrap();
    (dataset, tx)
}

async fn snapshot_from_scan_plan(tx: &mut StagedTransaction) -> Arc<Dataset> {
    let mut plans = vec![tx.scan().unwrap().create_plan().await.unwrap()];
    loop {
        let plan = plans.pop().expect("scan must contain a FilteredReadExec");
        if let Some(read) = plan.downcast_ref::<FilteredReadExec>() {
            return read.dataset().clone();
        }
        plans.extend(plan.children().into_iter().cloned());
    }
}

fn merge_small_fragments() -> CompactionOptions {
    CompactionOptions {
        target_rows_per_fragment: 100,
        ..Default::default()
    }
}

mod append;
