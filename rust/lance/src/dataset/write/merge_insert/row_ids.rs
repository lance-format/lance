// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::RecordBatch;
use datafusion::common::Result;
use datafusion::execution::disk_manager::DiskManager;
use datafusion::physical_plan::{SendableRecordBatchStream, stream::RecordBatchStreamAdapter};
use futures::{StreamExt, stream};
use lance_core::ROW_ID;
use lance_datafusion::{
    exec::provider_to_stream, spill::spilling_table_provider_with_job_disk_manager,
};

/// Put all updates before inserts without relying on join output order.
///
/// The commit assigns carried IDs to the output prefix and fresh IDs to its
/// tail. The nullable row-ID column distinguishes the two groups until this
/// final write boundary; order within each group must stay unchanged so the
/// captured update IDs still align with their rows.
pub(super) async fn updates_before_inserts(
    source: SendableRecordBatchStream,
    disk_manager: Arc<DiskManager>,
) -> Result<SendableRecordBatchStream> {
    let schema = source.schema();
    let row_id_col = schema.index_of(ROW_ID)?;
    let projection: Arc<[usize]> = (0..schema.fields().len())
        .filter(|&col| col != row_id_col)
        .collect();
    let output_schema = Arc::new(schema.project(&projection)?);

    // Heap-buffered batches are outside the job pool's accounting. Spill from
    // the first batch, charging the same quota as replay and join sort files.
    let staged = spilling_table_provider_with_job_disk_manager(source, 0, disk_manager).await?;
    let mut streams = Vec::with_capacity(2);
    for is_update in [true, false] {
        let replay = provider_to_stream(staged.clone()).await?;
        let projection = projection.clone();
        let selected = replay
            .map(move |batch| -> Result<Option<RecordBatch>> {
                let batch = batch?;
                let row_ids = batch.column(row_id_col);
                let selection = if is_update {
                    arrow::compute::is_not_null(row_ids)?
                } else {
                    arrow::compute::is_null(row_ids)?
                };
                if selection.true_count() == 0 {
                    return Ok(None);
                }
                let batch = arrow::compute::filter_record_batch(&batch, &selection)?;
                Ok(Some(batch.project(&projection)?))
            })
            .filter_map(|batch| std::future::ready(batch.transpose()));
        streams.push(selected);
    }

    Ok(Box::pin(RecordBatchStreamAdapter::new(
        output_schema,
        stream::iter(streams).flatten(),
    )))
}
